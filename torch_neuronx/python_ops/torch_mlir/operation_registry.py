"""Single-source operation registration system for torch-mlir backend."""

import inspect
from collections.abc import Callable

import torch
import torch.library

# Global registry to collect decorated operations
_PENDING_OPERATIONS = []


def register_aten(
    operations: str | list[str],
    output_params: tuple[str, ...] | None = None,
    static_argnums: tuple[int, ...] | None = None,
    static_argnames: tuple[str, ...] | None = None,
    uses_preprocessing: bool = False,
    validate_static_inputs: Callable | None = None,
    skip_dispatch_registration: bool = False,
    translate_args_fn: Callable | None = None,
):
    """
    Decorator that registers torch-mlir implementations for ATen operations.

    Args:
        operations: Single operation name or list of names
        output_params: Tuple of parameter names that are output tensors
                      (for operations with .output or .grad_input variants)
        static_argnums: Tuple of argument indices that should be treated as
                       static for cache key
        static_argnames: Tuple of keyword argument names that should be treated as
                        static for cache key

    Usage:
        @register_aten("aten::add")
        def torch_add(x, y):
            return torch.add(x, y)

        @register_aten(["aten::mul.Tensor", "aten::mul.out"])
        def torch_mul(x, y):
            return torch.mul(x, y)
    """
    if isinstance(operations, str):
        operations = [operations]

    def decorator(func: Callable):
        # Extract metadata from function
        func_name = func.__name__
        signature = inspect.signature(func)

        # Store registration info for each operation
        for op_name in operations:
            registration_info = {
                "aten_name": op_name,
                "torch_function": func,
                "function_name": func_name,
                "signature": signature,
                "output_params": output_params,
                "static_argnums": static_argnums,
                "static_argnames": static_argnames,
                "uses_preprocessing": uses_preprocessing,
                "validate_static_inputs": validate_static_inputs,
                "translate_args_fn": translate_args_fn,
            }

            # Phase 1: Add to torch-mlir registry immediately
            from .registry import register_torch_mlir_op

            register_torch_mlir_op(op_name, func, output_params)

            # Phase 2: Queue for PyTorch registration (happens at module init)
            # Skip if the op handles its own dispatch (e.g. via @neuron_op)
            if not skip_dispatch_registration:
                _PENDING_OPERATIONS.append(registration_info)

        # Mark function with metadata
        func._aten_operations = operations
        func.uses_preprocessing = uses_preprocessing

        return func

    return decorator


def finalize_registrations(aten_lib=None, verbose: bool = False):
    """
    Complete PyTorch registration for all pending operations.
    Called during module initialization.

    Args:
        aten_lib: PyTorch library to register operations with
        verbose: If True, print registration progress
    """
    if not _PENDING_OPERATIONS:
        return 0

    from torch_neuronx.python_ops.auto_registration import (
        create_auto_operation,
        create_wrapper_function,
    )

    registered_count = 0

    # Group operations by their implementation function
    func_to_ops = {}
    for reg_info in _PENDING_OPERATIONS:
        func_key = reg_info["function_name"]
        if func_key not in func_to_ops:
            func_to_ops[func_key] = []
        func_to_ops[func_key].append(reg_info)

    # Create one class per unique function, register for all operations
    for func_name, ops_list in func_to_ops.items():
        # Get output_params from the first operation
        output_params = ops_list[0].get("output_params")
        static_argnums = ops_list[0].get("static_argnums")
        static_argnames = ops_list[0].get("static_argnames")

        # Import TorchMlirOpImpl
        from .op_impl import TorchMlirOpImpl

        # Register each operation variant
        for reg_info in ops_list:
            aten_name = reg_info["aten_name"]
            torch_fn = reg_info["torch_function"]

            # Create a wrapper class
            ta_fn = reg_info.get("translate_args_fn")

            def make_wrapper(
                name, fn, params, static_argnums, static_argnames, translate_args_fn=None
            ):
                class ImplementationWrapper(TorchMlirOpImpl):
                    def __init__(self):
                        super().__init__(name, fn, params, static_argnums, static_argnames)
                        if translate_args_fn is not None:
                            self.translate_args = translate_args_fn

                return ImplementationWrapper

            implementation_wrapper = make_wrapper(
                aten_name, torch_fn, output_params, static_argnums, static_argnames, ta_fn
            )

            # Create the operation wrapper
            operation = create_auto_operation(aten_name, [implementation_wrapper])

            # Create wrapper function
            wrapper_func = create_wrapper_function(aten_name, operation)

            # Register with PyTorch
            op_name = aten_name.replace("aten::", "")
            try:
                # import pdb; pdb.set_trace()
                aten_lib.impl(op_name, wrapper_func, "PrivateUse1")
                registered_count += 1
                if verbose:
                    print(f"[OK] Registered torch-mlir op: {aten_name} -> {func_name}")
            except Exception as e:
                if verbose:
                    print(f"[WARN] Failed to register {op_name}: {e}")

    _PENDING_OPERATIONS.clear()
    return registered_count


# ── Autograd formula registry ────────────────────────────────────────────
# Ops register their autograd formula (setup_context + backward) here.
# At finalize time, _maybe_register_autograd picks these up and registers
# them on AutogradPrivateUse1 so the C++ forward path gets a grad_fn.

_AUTOGRAD_FORMULAS = {}  # aten_name -> {setup_context, backward}
_AUTOGRAD_LIBS = {}  # aten_name -> Library (prevent GC)


def register_autograd_formula(
    forward_op: str,
    setup_context: Callable,
    backward: Callable,
    wrapper: Callable | None = None,
):
    """Register an autograd formula for a forward op.

    Args:
        forward_op: ATen op name (e.g. "aten::cross_entropy_loss")
        setup_context: Called during forward to save state for backward.
            Signature: setup_context(ctx, inputs, output)
        backward: Called during backward to compute gradients.
            Signature: backward(ctx, *grad_outputs) -> tuple of gradients
        wrapper: Optional callable that wraps the autograd kernel.
            Signature: wrapper(autograd_kernel) -> wrapped_kernel
            Useful for intercepting DTensor inputs before the standard
            autograd redispatch (which would hit DTensor dispatch and fail
            on mixed Tensor/DTensor args).
    """
    _AUTOGRAD_FORMULAS[forward_op] = {
        "setup_context": setup_context,
        "backward": backward,
        "wrapper": wrapper,
    }


def _maybe_register_autograd(aten_name):
    """Register autograd for an op if it has a formula in _AUTOGRAD_FORMULAS.

    Uses torch.library internals to register on AutogradPrivateUse1 so the
    C++ forward path runs unchanged while PyTorch attaches a grad_fn.
    """
    formula = _AUTOGRAD_FORMULAS.get(aten_name)
    if formula is None:
        return

    ns, opname = aten_name.split("::")
    op_overload = getattr(getattr(torch.ops, ns), opname).default

    autograd_kernel = torch.library._library.autograd.make_autograd_impl(
        op_overload,
        torch.library._library.autograd.Info(formula["backward"], formula["setup_context"]),
    )

    # If a wrapper is provided, wrap the autograd kernel (e.g. to handle
    # DTensor inputs before the standard redispatch).
    if formula.get("wrapper"):
        autograd_kernel = formula["wrapper"](autograd_kernel)

    lib = torch.library.Library("aten", "IMPL")
    lib.impl(opname, autograd_kernel, "AutogradPrivateUse1", with_keyset=True)
    _AUTOGRAD_LIBS[aten_name] = lib


def finalize_registrations_cpp(aten_lib=None, verbose: bool = False):
    """
    Complete registration for all pending operations via C++ NeuronDispatcher.
    Called during module initialization.

    Instead of aten_lib.impl(op_name, wrapper, "PrivateUse1"), this calls
    _C._register_neuron_op(op_name, wrapper, lower_func) which registers:
    1. A C++ boxed handler that calls the Python wrapper (Phase 1 fallback)
    2. A Python lower function (kernel.lower) that the C++ dispatcher calls
       on cache miss to get StableHLO bytes + cache key

    In steady state, all lowerings are cached and no Python calls are needed.

    Ops with uses_preprocessing=True are registered via aten_lib.impl instead
    (the Python dispatch path) because their preprocessing protocol (factory
    functions, out= writeback, void returns) is not yet supported by the C++
    dispatch pipeline.

    Args:
        aten_lib: PyTorch library for Python-path registration of preprocessing ops
        verbose: If True, print registration progress

    Returns:
        Number of operations registered
    """
    if not _PENDING_OPERATIONS:
        return 0

    # Import _C module for C++ dispatcher
    import torch_neuronx._C as _C
    from torch_neuronx.python_ops.auto_registration import (
        create_auto_operation,
        create_wrapper_function,
    )

    registered_count = 0

    # Group operations by their implementation function
    func_to_ops = {}
    for reg_info in _PENDING_OPERATIONS:
        func_key = reg_info["function_name"]
        if func_key not in func_to_ops:
            func_to_ops[func_key] = []
        func_to_ops[func_key].append(reg_info)

    # Create one class per unique function, register for all operations
    for func_name, ops_list in func_to_ops.items():
        # Get output_params from the first operation
        output_params = ops_list[0].get("output_params")
        static_argnums = ops_list[0].get("static_argnums")
        static_argnames = ops_list[0].get("static_argnames")

        # Import TorchMlirOpImpl
        from .op_impl import TorchMlirOpImpl

        # Register each operation variant
        for reg_info in ops_list:
            aten_name = reg_info["aten_name"]
            torch_fn = reg_info["torch_function"]

            # Create a wrapper class
            ta_fn = reg_info.get("translate_args_fn")

            def make_wrapper(
                name, fn, params, static_argnums, static_argnames, translate_args_fn=None
            ):
                class ImplementationWrapper(TorchMlirOpImpl):
                    def __init__(self):
                        super().__init__(name, fn, params, static_argnums, static_argnames)
                        if translate_args_fn is not None:
                            self.translate_args = translate_args_fn

                return ImplementationWrapper

            implementation_wrapper = make_wrapper(
                aten_name, torch_fn, output_params, static_argnums, static_argnames, ta_fn
            )

            # Create the operation wrapper
            operation = create_auto_operation(aten_name, [implementation_wrapper])

            # Create wrapper function (Phase 1 fallback)
            wrapper_func = create_wrapper_function(aten_name, operation)

            uses_preprocessing = reg_info.get("uses_preprocessing", False)
            op_name = aten_name.replace("aten::", "")

            # Ops with uses_preprocessing=True use a factory-function protocol
            # (returning compute_fn + out= + postprocess_fn) that the C++ dispatch
            # pipeline doesn't support.  Register them via the Python aten_lib path.
            if uses_preprocessing and aten_lib is not None:
                # Wrap with validate_static_inputs if registered, so validation
                # runs automatically on raw args before Operation.__call__.
                validate_fn = reg_info.get("validate_static_inputs")
                if validate_fn is not None:
                    original_wrapper = wrapper_func

                    def make_validated_wrapper(vfn, orig):
                        def validated_wrapper(*args, **kwargs):
                            vfn(*args, **kwargs)
                            return orig(*args, **kwargs)

                        validated_wrapper.__name__ = orig.__name__
                        return validated_wrapper

                    wrapper_func = make_validated_wrapper(validate_fn, original_wrapper)

                try:
                    aten_lib.impl(op_name, wrapper_func, "PrivateUse1")
                    registered_count += 1
                except Exception as e:
                    if verbose:
                        print(f"[WARN] Failed to register {op_name} via aten_lib: {e}")
                    continue
                try:
                    _maybe_register_autograd(aten_name)
                except Exception as e:
                    print(f"[WARN] Failed to register autograd for {aten_name}: {e}")
                    import traceback

                    traceback.print_exc()
                if verbose:
                    print(
                        f"[OK] Registered via Python dispatch (uses_preprocessing): "
                        f"{aten_name} -> {func_name}"
                    )
                continue

            # Extract kernel.lower from the first implementation that has a kernel.
            # Attach validate_static_inputs to the kernel so it fires during
            # dtype inference (on original pre-downcast args) on cache miss.
            lower_func = None
            validate_fn = reg_info.get("validate_static_inputs")
            for impl in operation._implementations:
                if hasattr(impl, "kernel") and impl.kernel is not None:
                    lower_func = impl.kernel.lower
                    if validate_fn is not None:
                        impl.kernel.validate_static_inputs_func = validate_fn
                    break

            # Register with C++ NeuronDispatcher.
            # Dtype inference is unified into lower_func: on cache miss, C++
            # passes both downcasted and original args, and lower_func infers
            # output dtypes at original precision inside the single Python call.
            try:
                _C._register_neuron_op(
                    op_name,
                    wrapper_func,
                    lower_func,
                    list(static_argnums) if static_argnums else [],
                    list(static_argnames) if static_argnames else [],
                )
                registered_count += 1

                # Register autograd for ops that have a backward implementation.
                # This lets the C++ forward path run unchanged while PyTorch
                # automatically attaches a grad_fn that calls our Python backward.
                _maybe_register_autograd(aten_name)

                if verbose:
                    has_lower = "with lower()" if lower_func else "wrapper only"
                    print(
                        f"[OK] Registered via C++ dispatcher: "
                        f"{aten_name} -> {func_name} ({has_lower})"
                    )
            except Exception as e:
                if verbose:
                    print(f"[WARN] Failed to register {op_name} via C++ dispatcher: {e}")

    _PENDING_OPERATIONS.clear()
    return registered_count


def get_pending_operations():
    """Get list of operations pending registration (for debugging)."""
    return list(_PENDING_OPERATIONS)


def clear_pending_operations():
    """Clear pending operations (mainly for testing)."""
    _PENDING_OPERATIONS.clear()
