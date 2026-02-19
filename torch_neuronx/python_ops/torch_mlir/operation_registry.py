"""Single-source operation registration system for torch-mlir backend."""

import inspect
from collections.abc import Callable

# Global registry to collect decorated operations
_PENDING_OPERATIONS = []


def register_aten(
    operations: str | list[str],
    output_params: tuple[str, ...] | None = None,
    static_argnums: tuple[int, ...] | None = None,
    static_argnames: tuple[str, ...] | None = None,
    uses_preprocessing: bool = False,
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
            }

            # Phase 1: Add to torch-mlir registry immediately
            from .registry import register_torch_mlir_op

            register_torch_mlir_op(op_name, func, output_params)

            # Phase 2: Queue for PyTorch registration (happens at module init)
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
            def make_wrapper(name, fn, params, static_argnums, static_argnames):
                class ImplementationWrapper(TorchMlirOpImpl):
                    def __init__(self):
                        super().__init__(name, fn, params, static_argnums, static_argnames)

                return ImplementationWrapper

            implementation_wrapper = make_wrapper(
                aten_name, torch_fn, output_params, static_argnums, static_argnames
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


def get_pending_operations():
    """Get list of operations pending registration (for debugging)."""
    return list(_PENDING_OPERATIONS)


def clear_pending_operations():
    """Clear pending operations (mainly for testing)."""
    _PENDING_OPERATIONS.clear()
