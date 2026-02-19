"""Auto-registration system for neuron operations.

This module provides automatic registration of operations decorated with @neuron_op.
It discovers implementations, creates Operation wrappers, and registers with PyTorch.
"""

import importlib
import pkgutil
import warnings
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import torch
import torch.library

from .base import (
    BinaryOperation,
    ComparisonOperation,
    Operation,
    OperationImplementation,
    ReductionOperation,
    UnaryOperation,
)

# Global registry for all decorated implementations
_NEURON_OPS_REGISTRY: dict[str, dict[str, Any]] = defaultdict(
    lambda: {
        "implementations": [],
        "wrapper_override": None,
        "operation_class": None,
    }
)


def register_implementation(aten_name: str, impl_class: type[OperationImplementation]):
    """Register an implementation class for an aten operation.

    Called automatically by the @neuron_op decorator.
    """
    _NEURON_OPS_REGISTRY[aten_name]["implementations"].append(impl_class)


def neuron_op(
    aten_name: str,
    priority: int = 50,
    *,
    disable_dtype_autocast: bool | None = None,
    disable_empty_shortcircuit: bool | None = None,
):
    """Enhanced decorator that auto-registers operations.

    Args:
        aten_name: The aten operation name (e.g., "aten::add", "aten::add.Tensor")
        priority: Priority for this implementation (higher = tried first)

    Example:
        @neuron_op("aten::add")
        class AddXLAImpl(BinaryOpImplementation):
            ...
    """

    def decorator(cls):
        # Add metadata to class
        cls._aten_op_name = aten_name
        cls._has_neuron_op = True
        cls._auto_priority = priority

        # Optional execution policy flags
        if disable_dtype_autocast is not None:
            cls.DISABLE_DTYPE_AUTOCAST = disable_dtype_autocast
        if disable_empty_shortcircuit is not None:
            cls.DISABLE_EMPTY_SHORTCIRCUIT = disable_empty_shortcircuit

        # Auto-register the implementation
        register_implementation(aten_name, cls)

        return cls

    return decorator


def create_auto_operation(
    aten_name: str, implementations: list[type[OperationImplementation]]
) -> Operation:
    """Create an Operation instance dynamically for the given implementations."""

    class AutoOperation(Operation):
        def __init__(self):
            # Store the actual aten name for this specific operation variant
            self._actual_aten_name = aten_name
            super().__init__()

        def _setup_implementations(self):
            """Register all implementations for this operation."""

            def create_priority_impl(base_class, priority):
                """Factory function to create implementation with custom priority."""

                class PriorityImpl(base_class):
                    @property
                    def priority(self) -> int:
                        return priority

                return PriorityImpl

            for impl_class in implementations:
                # Check if we need to override priority
                if hasattr(impl_class, "_auto_priority"):
                    # Use factory to create subclass with correct priority
                    priority_impl_class = create_priority_impl(
                        impl_class, impl_class._auto_priority
                    )
                    impl = priority_impl_class()
                else:
                    impl = impl_class()
                self._implementations.append(impl)

        @property
        def op_name(self) -> str:
            # Extract clean op name from aten name
            # "aten::add" -> "add", "aten::add.Tensor" -> "add"
            clean_name = aten_name.replace("aten::", "")
            if "." in clean_name:
                clean_name = clean_name.split(".")[0]
            return clean_name

    # Create instance
    return AutoOperation()


def create_auto_operation_typed(
    aten_name: str,
    implementations: list[type[OperationImplementation]],
    base_class: type[Operation] = Operation,
) -> Operation:
    """Create an Operation instance dynamically with a specific base class.

    Args:
        aten_name: The aten operation name
        implementations: List of implementation classes
        base_class: The base Operation class to inherit from (UnaryOperation, BinaryOperation, etc.)
    """

    class TypedAutoOperation(base_class):
        def __init__(self):
            # Store the actual aten name for this specific operation variant
            self._actual_aten_name = aten_name
            super().__init__()

        def _setup_implementations(self):
            """Register all implementations for this operation."""

            def create_priority_impl(impl_base_class, priority):
                """Factory function to create implementation with custom priority."""

                class PriorityImpl(impl_base_class):
                    @property
                    def priority(self) -> int:
                        return priority

                return PriorityImpl

            for impl_class in implementations:
                # Check if we need to override priority
                if hasattr(impl_class, "_auto_priority"):
                    # Use factory to create subclass with correct priority
                    priority_impl_class = create_priority_impl(
                        impl_class, impl_class._auto_priority
                    )
                    impl = priority_impl_class()
                else:
                    impl = impl_class()
                self._implementations.append(impl)

        @property
        def op_name(self) -> str:
            # Extract clean op name from aten name
            # "aten::add" -> "add", "aten::add.Tensor" -> "add"
            clean_name = aten_name.replace("aten::", "")
            if "." in clean_name:
                clean_name = clean_name.split(".")[0]
            return clean_name

    # Create instance
    return TypedAutoOperation()


def neuron_unary_op(aten_name: str, priority: int = 50):
    """Decorator for unary operations (sqrt, neg, relu, etc.).

    Operations registered with this decorator will automatically get
    proper output shape calculation for unary operations.
    """

    def decorator(cls):
        # Add metadata to class
        cls._aten_op_name = aten_name
        cls._has_neuron_op = True
        cls._auto_priority = priority
        cls._op_base_class = UnaryOperation

        # Auto-register the implementation
        register_implementation(aten_name, cls)

        return cls

    return decorator


def neuron_binary_op(aten_name: str, priority: int = 50):
    """Decorator for binary operations (add, mul, div, etc.).

    Operations registered with this decorator will automatically get
    proper output shape calculation using broadcasting rules.
    """

    def decorator(cls):
        # Add metadata to class
        cls._aten_op_name = aten_name
        cls._has_neuron_op = True
        cls._auto_priority = priority
        cls._op_base_class = BinaryOperation

        # Auto-register the implementation
        register_implementation(aten_name, cls)

        return cls

    return decorator


def neuron_reduction_op(aten_name: str, priority: int = 50):
    """Decorator for reduction operations (sum, mean, etc.).

    Operations registered with this decorator will automatically get
    proper output shape calculation for reductions.
    """

    def decorator(cls):
        # Add metadata to class
        cls._aten_op_name = aten_name
        cls._has_neuron_op = True
        cls._auto_priority = priority
        cls._op_base_class = ReductionOperation

        # Auto-register the implementation
        register_implementation(aten_name, cls)

        return cls

    return decorator


def neuron_comparison_op(aten_name: str, priority: int = 50):
    """Decorator for comparison operations (eq, lt, gt, etc.).

    Operations registered with this decorator will automatically get
    proper output shape calculation using broadcasting rules.
    """

    def decorator(cls):
        # Add metadata to class
        cls._aten_op_name = aten_name
        cls._has_neuron_op = True
        cls._auto_priority = priority
        cls._op_base_class = ComparisonOperation

        # Auto-register the implementation
        register_implementation(aten_name, cls)

        return cls

    return decorator


def create_wrapper_function(aten_name: str, operation: Operation) -> Callable:
    """Create a wrapper function for PyTorch registration.

    Simply passes all arguments through to the operation without
    making any assumptions about the operation type.
    """

    op_name = aten_name.replace("aten::", "")

    if operation.is_inplace:
        # In-place operations need special handling to ensure they modify the input tensor
        def wrapper(self, *args, **kwargs):
            # Call the operation with out=self to ensure in-place modification
            kwargs["out"] = self
            result = operation(self, *args, **kwargs)
            if result is None:
                return result
            else:
                return self
    else:
        # Standard wrapper - just pass everything through
        def wrapper(*args, **kwargs):
            return operation(*args, **kwargs)

    wrapper.__name__ = f"{op_name.replace('.', '_').replace(':', '_')}_neuron"

    return wrapper


def discover_all_implementations(package_name: str = "torch_neuronx.python_ops"):
    """Discover all implementations by importing all modules in the package."""

    # Import the base package
    package = importlib.import_module(package_name)

    # Walk through all submodules
    for _importer, modname, _ispkg in pkgutil.walk_packages(
        path=package.__path__, prefix=package.__name__ + ".", onerror=lambda x: None
    ):
        # Skip __pycache__ and test modules
        if "__pycache__" in modname or "test" in modname:
            continue

        # Skip legacy and JAX packages
        if ".legacy" in modname or modname.endswith(".legacy"):
            continue
        if ".jax" in modname or modname.endswith(".jax"):
            continue

        try:
            # Import the module - this triggers decorator registration
            importlib.import_module(modname)
        except Exception as e:
            # Log but don't fail - some modules might have import issues
            warnings.warn(f"Failed to import {modname}: {e}", stacklevel=2)


def auto_register_neuron_ops(aten_lib: torch.library.Library, verbose: bool = False):
    """Auto-register all operations decorated with @neuron_op.

    Args:
        aten_lib: The PyTorch library to register operations with
        verbose: If True, print registration details
    """

    # First, discover all implementations by importing modules
    discover_all_implementations()

    # Now register everything in the registry
    registered_count = 0
    for aten_name, reg_info in _NEURON_OPS_REGISTRY.items():
        # Skip if there's a manual wrapper override
        if reg_info.get("wrapper_override"):
            wrapper_func = reg_info["wrapper_override"]
        else:
            implementations = reg_info["implementations"]
            if not implementations:
                continue

            # Use existing Operation class if specified, otherwise create one
            if reg_info.get("operation_class"):
                operation = reg_info["operation_class"]
            else:
                # Check if any implementation specifies a base class
                base_class = Operation
                for impl_class in implementations:
                    if hasattr(impl_class, "_op_base_class"):
                        base_class = impl_class._op_base_class
                        break

                # Use typed operation creation if a specific base class is found
                if base_class != Operation:
                    operation = create_auto_operation_typed(aten_name, implementations, base_class)
                else:
                    operation = create_auto_operation(aten_name, implementations)

            # Create wrapper function
            wrapper_func = create_wrapper_function(aten_name, operation)

        # Register with PyTorch
        op_name = aten_name.replace("aten::", "")
        try:
            aten_lib.impl(op_name, wrapper_func, "PrivateUse1")
            registered_count += 1
            if verbose:
                print(f"✓ Auto-registered: {op_name} -> {wrapper_func.__name__}")
        except Exception as e:
            warnings.warn(f"Failed to register {op_name}: {e}", stacklevel=2)

    if verbose:
        print(f"\nAuto-registered {registered_count} operations")

    return registered_count


def set_wrapper_override(aten_name: str, wrapper_func: Callable):
    """Set a custom wrapper function for an operation.

    Use this for complex operations that need custom dispatch logic.
    """
    _NEURON_OPS_REGISTRY[aten_name]["wrapper_override"] = wrapper_func


def set_operation_class(aten_name: str, operation_class: Operation):
    """Set a custom Operation class for an aten operation.

    Use this when you have a hand-written Operation class with custom logic.
    """
    _NEURON_OPS_REGISTRY[aten_name]["operation_class"] = operation_class


def get_registry_info() -> dict[str, dict[str, Any]]:
    """Get information about all registered operations (for debugging)."""
    return dict(_NEURON_OPS_REGISTRY)
