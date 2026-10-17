"""Auto-registration system for neuron operations.

This module provides automatic registration of operations decorated with @neuron_op.
It discovers implementations, creates Operation wrappers, and registers with PyTorch.
"""

import importlib
import pkgutil
import sys
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


def _register_dynamic_class(cls, cls_name):
    """Register a dynamically created class at module level for pickle.

    torch.compile's caching_precompile requires all types referenced in
    guards to be resolvable by pickle via module-level attribute lookup.
    Dynamically created classes (via ``type()`` or local ``class`` statements)
    have ``__qualname__`` containing ``<locals>`` which fails this check.

    This function sets ``__name__`` and ``__qualname__``, so that
    ``__name__ == __qualname__`` (required by Dynamo's TYPE_MATCH guard)
    and registers the class in the module's namespace.
    """
    cls.__name__ = cls_name
    cls.__qualname__ = cls_name
    sys.modules[cls.__module__].__dict__[cls_name] = cls


# Global registry for all decorated implementations
_NEURON_OPS_REGISTRY: dict[str, dict[str, Any]] = defaultdict(
    lambda: {
        "implementations": [],
        "wrapper_override": None,
        "operation_class": None,
    }
)


def register_implementation(aten_name: str, impl_class: type[OperationImplementation]):
    """Register an implementation class for an ATen operation.

    Args:
        aten_name (str): Qualified ATen operation name used as the registry key.
        impl_class (type[OperationImplementation]): Implementation class to append.

    Note:
        The ``neuron_op`` decorators call this function automatically.
    """
    _NEURON_OPS_REGISTRY[aten_name]["implementations"].append(impl_class)


def neuron_op(
    aten_name: str,
    priority: int = 50,
    *,
    disable_dtype_autocast: bool | None = None,
    disable_empty_shortcircuit: bool | None = None,
):
    """Decorate and register an operation implementation class.

    Args:
        aten_name (str): Qualified ATen operation name, such as ``aten::add.Tensor``.
        priority (int, optional): Selection priority; higher values are tried first.
            Defaults to ``50``.
        disable_dtype_autocast (bool, optional): Whether to disable dtype autocasting.
            Defaults to ``None``, which leaves the class policy unchanged.
        disable_empty_shortcircuit (bool, optional): Whether to disable the empty-input
            short circuit. Defaults to ``None``, which leaves the class policy unchanged.

    Returns:
        Callable: Class decorator that adds registration metadata and registers the class.
    """

    def decorator(cls):
        # Add metadata to class
        """Add operation metadata to and register an implementation class.

        Args:
            cls (type): Operation implementation class to decorate.

        Returns:
            type: The same class after metadata and policy attributes are set.
        """
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
    """Create a dynamically typed operation for implementation classes.

    Args:
        aten_name (str): Qualified ATen operation name represented by the instance.
        implementations (list[type[OperationImplementation]]): Implementation classes
            to instantiate in registry order.

    Returns:
        Operation: Dynamic operation instance with pickle-resolvable generated classes.
    """

    class AutoOperation(Operation):
        """Represent a generated operation backed by registered implementations."""

        def __init__(self):
            # Store the actual aten name for this specific operation variant
            self._actual_aten_name = aten_name
            super().__init__()

        def _setup_implementations(self):
            """Register all implementations for this operation."""

            def create_priority_impl(base_class, priority):
                """Create a pickle-resolvable implementation subclass with a priority.

                Args:
                    base_class (type): Implementation class to subclass.
                    priority (int): Priority returned by the generated property.

                Returns:
                    type: Generated implementation subclass registered at module scope.
                """

                class PriorityImpl(base_class):
                    """Represent an implementation with a captured priority."""

                    @property
                    def priority(self) -> int:
                        """Return the captured implementation priority.

                        Returns:
                            int: Priority supplied when the subclass was generated.
                        """
                        return priority

                # Register at module level for pickle resolution
                _register_dynamic_class(
                    PriorityImpl, f"PriorityImpl_{base_class.__name__}_{priority}"
                )

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
            """Return the unqualified base ATen operation name.

            Returns:
                str: Name with the ``aten::`` prefix and overload suffix removed.
            """
            clean_name = aten_name.replace("aten::", "")
            if "." in clean_name:
                clean_name = clean_name.split(".")[0]
            return clean_name

    # Register class at module level so pickle can resolve it.
    _register_dynamic_class(
        AutoOperation,
        f"AutoOperation_{aten_name.replace('::', '_').replace('.', '_')}",
    )
    return AutoOperation()


def create_auto_operation_typed(
    aten_name: str,
    implementations: list[type[OperationImplementation]],
    base_class: type[Operation] = Operation,
) -> Operation:
    """Create a dynamically typed operation with a specific base class.

    Args:
        aten_name (str): Qualified ATen operation name represented by the instance.
        implementations (list[type[OperationImplementation]]): Implementation classes
            to instantiate in registry order.
        base_class (type[Operation], optional): Operation class for the generated type.
            Defaults to ``Operation``.

    Returns:
        Operation: Dynamic typed operation with pickle-resolvable generated classes.
    """

    class TypedAutoOperation(base_class):
        """Represent a generated operation with a caller-selected base class."""

        def __init__(self):
            # Store the actual aten name for this specific operation variant
            self._actual_aten_name = aten_name
            super().__init__()

        def _setup_implementations(self):
            """Register all implementations for this operation."""

            def create_priority_impl(impl_base_class, priority):
                """Create a pickle-resolvable implementation subclass with a priority.

                Args:
                    impl_base_class (type): Implementation class to subclass.
                    priority (int): Priority returned by the generated property.

                Returns:
                    type: Generated implementation subclass registered at module scope.
                """

                class PriorityImpl(impl_base_class):
                    """Represent an implementation with a captured priority."""

                    @property
                    def priority(self) -> int:
                        """Return the captured implementation priority.

                        Returns:
                            int: Priority supplied when the subclass was generated.
                        """
                        return priority

                _register_dynamic_class(
                    PriorityImpl, f"PriorityImpl_{impl_base_class.__name__}_{priority}"
                )

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
            """Return the unqualified base ATen operation name.

            Returns:
                str: Name with the ``aten::`` prefix and overload suffix removed.
            """
            clean_name = aten_name.replace("aten::", "")
            if "." in clean_name:
                clean_name = clean_name.split(".")[0]
            return clean_name

    # Create instance
    _register_dynamic_class(
        TypedAutoOperation,
        f"TypedAutoOperation_{aten_name.replace('::', '_').replace('.', '_')}",
    )
    return TypedAutoOperation()


def neuron_unary_op(aten_name: str, priority: int = 50):
    """Decorate and register a unary operation implementation class.

    Args:
        aten_name (str): Qualified ATen operation name to register.
        priority (int, optional): Selection priority; higher values are tried first.
            Defaults to ``50``.

    Returns:
        Callable: Class decorator that selects unary output-shape handling.
    """

    def decorator(cls):
        # Add metadata to class
        """Add unary-operation metadata to and register an implementation class.

        Args:
            cls (type): Operation implementation class to decorate.

        Returns:
            type: The class configured to use ``UnaryOperation``.
        """
        cls._aten_op_name = aten_name
        cls._has_neuron_op = True
        cls._auto_priority = priority
        cls._op_base_class = UnaryOperation

        # Auto-register the implementation
        register_implementation(aten_name, cls)

        return cls

    return decorator


def neuron_binary_op(aten_name: str, priority: int = 50):
    """Decorate and register a binary operation implementation class.

    Args:
        aten_name (str): Qualified ATen operation name to register.
        priority (int, optional): Selection priority; higher values are tried first.
            Defaults to ``50``.

    Returns:
        Callable: Class decorator that selects broadcast output-shape handling.
    """

    def decorator(cls):
        # Add metadata to class
        """Add binary-operation metadata to and register an implementation class.

        Args:
            cls (type): Operation implementation class to decorate.

        Returns:
            type: The class configured to use ``BinaryOperation``.
        """
        cls._aten_op_name = aten_name
        cls._has_neuron_op = True
        cls._auto_priority = priority
        cls._op_base_class = BinaryOperation

        # Auto-register the implementation
        register_implementation(aten_name, cls)

        return cls

    return decorator


def neuron_reduction_op(aten_name: str, priority: int = 50):
    """Decorate and register a reduction operation implementation class.

    Args:
        aten_name (str): Qualified ATen operation name to register.
        priority (int, optional): Selection priority; higher values are tried first.
            Defaults to ``50``.

    Returns:
        Callable: Class decorator that selects reduction output-shape handling.
    """

    def decorator(cls):
        # Add metadata to class
        """Add reduction metadata to and register an implementation class.

        Args:
            cls (type): Operation implementation class to decorate.

        Returns:
            type: The class configured to use ``ReductionOperation``.
        """
        cls._aten_op_name = aten_name
        cls._has_neuron_op = True
        cls._auto_priority = priority
        cls._op_base_class = ReductionOperation

        # Auto-register the implementation
        register_implementation(aten_name, cls)

        return cls

    return decorator


def neuron_comparison_op(aten_name: str, priority: int = 50):
    """Decorate and register a comparison operation implementation class.

    Args:
        aten_name (str): Qualified ATen operation name to register.
        priority (int, optional): Selection priority; higher values are tried first.
            Defaults to ``50``.

    Returns:
        Callable: Class decorator that selects comparison output-shape handling.
    """

    def decorator(cls):
        # Add metadata to class
        """Add comparison metadata to and register an implementation class.

        Args:
            cls (type): Operation implementation class to decorate.

        Returns:
            type: The class configured to use ``ComparisonOperation``.
        """
        cls._aten_op_name = aten_name
        cls._has_neuron_op = True
        cls._auto_priority = priority
        cls._op_base_class = ComparisonOperation

        # Auto-register the implementation
        register_implementation(aten_name, cls)

        return cls

    return decorator


def make_composite_lower(operation):
    """Create a composite lowering callable for an operation.

    Args:
        operation (Operation): Operation whose implementations are tried in order.

    Returns:
        Callable: Dynamo-disabled lowerer used by the C++ Neuron dispatcher on a
            cache miss. It returns ``None`` when the first handling implementation
            has no lowerable function or kernel, or when no implementation handles
            the arguments.

    Note:
        The lowerer follows ``Operation.__call__`` order, preserves the prior Dynamo
        evaluation frame, and forwards original arguments to kernel lowering.
    """

    @torch._dynamo.disable
    def composite_lower(*args, **kwargs):
        """Lower with the first handling implementation, or request CPU fallback.

        Args:
            *args (Any): Positional operation arguments.
            **kwargs (Any): Keyword arguments, including optional original arguments
                injected by the C++ dispatcher.

        Returns:
            Any: Lowering output, or ``None`` when CPU fallback is needed.
        """
        prior = torch._dynamo.eval_frame.set_eval_frame(None)
        try:
            # Extract original_args/original_kwargs injected by C++ before
            # translate_args strips them (they're not normal op kwargs).
            original_args = kwargs.pop("original_args", None)
            original_kwargs = kwargs.pop("original_kwargs", None)

            for impl in operation._implementations:
                impl_args, impl_kwargs = impl.translate_args(*args, **kwargs)
                can_handle_result = impl.can_handle(*impl_args, **impl_kwargs)
                if isinstance(can_handle_result, tuple):
                    can_handle_ok = can_handle_result[0]
                else:
                    can_handle_ok = can_handle_result

                if not can_handle_ok:
                    continue

                if hasattr(impl, "get_lower_func"):
                    lower_fn = impl.get_lower_func()
                    if lower_fn is not None:
                        return lower_fn(
                            *args,
                            **kwargs,
                        )
                if hasattr(impl, "kernel") and impl.kernel is not None:
                    return impl.kernel.lower(
                        *impl_args,
                        original_args=original_args,
                        original_kwargs=original_kwargs,
                        **impl_kwargs,
                    )

                # This impl can handle but has no kernel — can't lower via C++.
                return None

            return None
        finally:
            torch._dynamo.eval_frame.set_eval_frame(prior)

    return composite_lower


def create_wrapper_function(aten_name: str, operation: Operation) -> Callable:
    """Create a PyTorch registration wrapper for an operation.

    Args:
        aten_name (str): Qualified ATen name used to name the wrapper.
        operation (Operation): Operation invoked by the generated wrapper.

    Returns:
        Callable: Generated wrapper named from the ATen operation.

    Note:
        Standard wrappers forward all arguments. In-place wrappers pass the first
        tensor as ``out`` and return it unless the operation returns ``None``.
    """
    op_name = aten_name.replace("aten::", "")

    if operation.is_inplace:
        # In-place operations need special handling to ensure they modify the input tensor
        def wrapper(self, *args, **kwargs):
            """Invoke an in-place operation with ``self`` as the output.

            Args:
                self (torch.Tensor): Destination tensor passed as ``out``.
                *args (Any): Additional positional operation arguments.
                **kwargs (Any): Additional keyword operation arguments.

            Returns:
                Any: ``None`` when the operation returns ``None``; otherwise, ``self``.
            """
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
            """Forward arguments to the registered operation.

            Args:
                *args (Any): Positional operation arguments.
                **kwargs (Any): Keyword operation arguments.

            Returns:
                Any: Result returned by the operation.
            """
            return operation(*args, **kwargs)

    wrapper.__name__ = f"{op_name.replace('.', '_').replace(':', '_')}_neuron"

    return wrapper


def discover_all_implementations(package_name: str = "torch_neuronx.python_ops"):
    """Discover operation implementations by importing package modules.

    Args:
        package_name (str, optional): Package tree to import recursively. Defaults to
            ``torch_neuronx.python_ops``.

    Note:
        Module names containing ``test`` or ``__pycache__`` are skipped. Failed
        imports emit warnings, and discovery continues because imports trigger registration.
    """
    # Import the base package
    package = importlib.import_module(package_name)

    # Walk through all submodules
    for _importer, modname, _ispkg in pkgutil.walk_packages(
        path=package.__path__, prefix=package.__name__ + ".", onerror=lambda x: None
    ):
        # Skip __pycache__ and test modules
        if "__pycache__" in modname or "test" in modname:
            continue

        try:
            # Import the module - this triggers decorator registration
            importlib.import_module(modname)
        except Exception as e:
            # Log but don't fail - some modules might have import issues
            warnings.warn(f"Failed to import {modname}: {e}", stacklevel=2)


def auto_register_neuron_ops(
    aten_lib: torch.library.Library,
    verbose: bool = False,
    torch_mlir_ops: set[str] | None = None,
):
    """Register all decorated operations with PyTorch dispatch.

    Args:
        aten_lib (torch.library.Library): Library used to register operations that have no
            lowerable kernel (Python dispatch path).
        verbose (bool, optional): Whether to print registration details. Defaults to
            ``False``.
        torch_mlir_ops (set[str], optional): ATen names registered by torch-mlir; these are
            skipped here. Defaults to ``None``.

    Returns:
        int: Number of operations successfully registered.

    Note:
        Registration failures emit warnings. Operations for which no implementation
        has a kernel or lower-function attribute are registered through ``aten_lib``
        instead of the C++ Neuron dispatcher.
    """
    # First, discover all implementations by importing modules
    discover_all_implementations()

    import torch_neuronx._C as _C

    # Now register everything in the registry
    registered_count = 0
    for aten_name, reg_info in _NEURON_OPS_REGISTRY.items():
        # Skip operations that have @register_aten torch-mlir registrations when using C++ dispatch
        # Those will be registered via C++ dispatcher by torch-mlir's finalize_registrations_cpp
        if torch_mlir_ops and aten_name in torch_mlir_ops:
            if verbose:
                print(f"⏭ Skipping {aten_name} - will use torch-mlir C++ dispatch")
            continue
        # Skip if there's a manual wrapper override
        if reg_info.get("wrapper_override"):
            wrapper_func = reg_info["wrapper_override"]
            operation = None
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
            lower_func = None
            if operation is not None:
                has_any_lowerable = any(
                    (hasattr(impl, "kernel") and impl.kernel is not None)
                    or hasattr(impl, "get_lower_func")
                    for impl in operation._implementations
                )
                if has_any_lowerable:
                    lower_func = make_composite_lower(operation)
            # Extract static_argnums/static_argnames from the highest-priority
            # implementation.  These tell C++ PreprocessArgs to skip converting
            # scalar args (like dim, non_blocking) to device tensors.
            static_argnums = []
            static_argnames = []
            if operation is not None:
                for impl in operation._implementations:
                    if hasattr(impl, "static_argnums") and impl.static_argnums:
                        static_argnums = list(impl.static_argnums)
                        break
                for impl in operation._implementations:
                    if hasattr(impl, "static_argnames") and impl.static_argnames:
                        static_argnames = list(impl.static_argnames)
                        break
            # Ops with no lowerable kernel (pure Python/NRT path) cannot go
            # through the C++ dispatcher which requires a lower_func.
            # Fall back to the Python aten_lib path for these.
            if lower_func is None:
                aten_lib.impl(op_name, wrapper_func, "PrivateUse1")
                registered_count += 1
                if verbose:
                    print(f"✓ Auto-registered (Python path): {op_name} -> {wrapper_func.__name__}")
                continue
            _C._register_neuron_op(
                op_name,
                wrapper_func,
                lower_func,
                static_argnums,
                static_argnames,
            )
            registered_count += 1
            if verbose:
                print(f"✓ Auto-registered: {op_name} -> {wrapper_func.__name__}")
        except Exception as e:
            warnings.warn(f"Failed to register {op_name}: {e}", stacklevel=2)

    if verbose:
        print(f"\nAuto-registered {registered_count} operations")

    return registered_count


def set_wrapper_override(aten_name: str, wrapper_func: Callable):
    """Set a custom dispatch wrapper for an ATen operation.

    Args:
        aten_name (str): Qualified ATen operation name to update.
        wrapper_func (Callable): Wrapper to store as the dispatch override.

    Note:
        Registration uses the override only when it evaluates truthy.
    """
    _NEURON_OPS_REGISTRY[aten_name]["wrapper_override"] = wrapper_func


def set_operation_class(aten_name: str, operation_class: Operation):
    """Set a custom operation object for an ATen operation.

    Args:
        aten_name (str): Qualified ATen operation name to update.
        operation_class (Operation): Hand-written operation to store in the registry.

    Note:
        Registration uses the custom object only when it evaluates truthy.
    """
    _NEURON_OPS_REGISTRY[aten_name]["operation_class"] = operation_class


def get_registry_info() -> dict[str, dict[str, Any]]:
    """Return a shallow copy of the operation registry.

    Returns:
        dict[str, dict[str, Any]]: Registry entries keyed by qualified ATen name.
    """
    return dict(_NEURON_OPS_REGISTRY)
