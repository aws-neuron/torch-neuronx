"""Single-source operation registration system.

This module provides decorators for registering JAX implementations directly
on the implementation functions, eliminating boilerplate registration code.
"""

import inspect
from collections.abc import Callable
from typing import Any

# Global registry to collect decorated operations
_PENDING_OPERATIONS = []


def register_aten(
    operations: str | list[str],
    operation_type: str = "auto",
    reduction_identity: Any | None = None,
    static_argnums: tuple[int, ...] | None = None,
    static_argnames: tuple[str, ...] | None = None,
    output_params: tuple[str, ...] | None = None,
    uses_preprocessing: bool = False,
):
    """
    Decorator that registers JAX implementations for ATen operations.

    Args:
        operations: Single operation name or list of names
        operation_type: "general", "reduction", "comparison", or "auto" (inferred)
        reduction_identity: For reduction ops, the identity value
        static_argnums: Tuple of argument indices that should be treated as
                       static (compile-time constants) for JAX compilation
        static_argnames: Tuple of keyword argument names that should be treated as
                        static (compile-time constants) for JAX compilation
        output_params: Tuple of parameter names that are output tensors
                      (for operations with .output or .grad_input variants)
        uses_preprocessing: Whether the function returns (jax_fn, processed_args)
                           instead of computing directly

    Usage:
        @register_aten("aten::sqrt")
        def _aten_sqrt(x):
            return jnp.sqrt(x)

        @register_aten(["aten::mul.Tensor", "aten::mul.Scalar",
                       "aten::mul.out", "aten::mul.Scalar_out"])
        def _aten_mul(x, y):
            return x * y

        @register_aten("aten::dist", static_argnums=(2,))
        def _aten_dist(input, other, p=2):
            # p (arg index 2) will be static during compilation
            return compute_dist(input, other, p)

        @register_aten("aten::nll_loss_forward.output",
                      static_argnums=(3, 4),
                      output_params=('output', 'total_weight'))
        def _aten_nll_loss_forward(input, target, weight, reduction=1, ignore_index=-100, **kwargs):
            # kwargs contains output tensors that will be written in-place
            return compute_nll_loss(input, target, weight, reduction, ignore_index)
    """
    if isinstance(operations, str):
        operations = [operations]

    # Explicit-only registration: do not auto-add variants. All overloads
    # (e.g., ".unary_out", ".dim_max", "_out") must be listed explicitly.

    def decorator(func: Callable):
        # Extract metadata from function
        func_name = func.__name__
        signature = inspect.signature(func)

        # Infer operation type if auto
        if operation_type == "auto":
            op_type = _infer_operation_type(func_name, signature)
        else:
            op_type = operation_type

        # Store registration info for each operation
        for op_name in operations:
            registration_info = {
                "aten_name": op_name,
                "jax_function": func,
                "operation_type": op_type,
                "function_name": func_name,
                "signature": signature,
                "reduction_identity": reduction_identity,
                "static_argnums": static_argnums,
                "static_argnames": static_argnames,
                "output_params": output_params,
                "uses_preprocessing": uses_preprocessing,
            }

            # Phase 1: Add to JAX registry immediately
            # This import is delayed to avoid circular dependencies
            from .registry import add_jax_operation

            add_jax_operation(op_name, func)

            # Phase 2: Queue for PyTorch registration (happens at module init)
            _PENDING_OPERATIONS.append(registration_info)

        # Mark function with metadata
        func._aten_operations = operations
        func._operation_type = op_type
        func.uses_preprocessing = uses_preprocessing

        return func

    return decorator


def _infer_operation_type(func_name: str, signature) -> str:
    """Return default operation type - always 'general'.

    All non-general operations must explicitly specify their type in the decorator.
    """
    # No inference - everything defaults to general
    return "general"


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

    # Group operations by their implementation function to avoid duplicate class creation
    func_to_ops = {}
    for reg_info in _PENDING_OPERATIONS:
        func_key = (reg_info["function_name"], reg_info["operation_type"])
        if func_key not in func_to_ops:
            func_to_ops[func_key] = []
        func_to_ops[func_key].append(reg_info)

    # Create one class per unique function, but register it for all operations
    for (func_name, op_type), ops_list in func_to_ops.items():
        # Get static_argnums, static_argnames and output_params from the first operation
        # (they should all be the same)
        static_argnums = ops_list[0].get("static_argnums")
        static_argnames = ops_list[0].get("static_argnames")
        output_params = ops_list[0].get("output_params")

        # Import the unified JaxOpImpl class
        from torch_neuronx.python_ops.legacy.jax_base import JaxOpImpl

        # Use unified JaxOpImpl for all operation types
        base_class = JaxOpImpl
        init_kwargs = {
            "static_argnums": static_argnums,
            "static_argnames": static_argnames,
            "output_params": output_params,
        }

        # For reductions, add the identity value if specified
        if op_type == "reduction" and ops_list[0].get("reduction_identity") is not None:
            init_kwargs["identity_value"] = ops_list[0].get("reduction_identity")
        else:
            # Use auto identity inference (the default)
            init_kwargs["identity_value"] = "auto"

        # Register each operation variant
        for reg_info in ops_list:
            aten_name = reg_info["aten_name"]

            # Create a wrapper class that can be instantiated without arguments
            # This is what create_auto_operation expects
            def make_wrapper(name, kwargs, cls):
                class ImplementationWrapper(cls):
                    def __init__(self):
                        super().__init__(name, **kwargs)

                return ImplementationWrapper

            implementation_wrapper = make_wrapper(aten_name, init_kwargs, base_class)

            # Create the operation wrapper
            operation = create_auto_operation(aten_name, [implementation_wrapper])

            # Create wrapper function
            wrapper_func = create_wrapper_function(aten_name, operation)

            # Register with PyTorch
            op_name = aten_name.replace("aten::", "")
            try:
                aten_lib.impl(op_name, wrapper_func, "PrivateUse1")
                registered_count += 1
                if verbose:
                    print(f"[OK] Registered from decorator: {aten_name} -> {func_name}")
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
