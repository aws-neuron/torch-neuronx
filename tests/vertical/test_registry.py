import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

from torch_neuronx.python_ops.auto_registration import discover_all_implementations

logger = logging.getLogger(__name__)


@dataclass
class RegistryComponent:
    """Component representing a registered test with its associated op specs.

    Attributes:
        op_specs: Dictionary mapping op spec names to their classes
        test_class: The test class that will execute the ops for the given specs
    """

    op_specs: dict[str, type[Any]] = field(default_factory=dict)
    test_class: type[Any] | None = None


class VerticalTestsRegistry:
    """Registry for managing vertical tests and their op specs.

    This class maintains a central registry of test classes and their associated
    op specs, providing functionality for automatic discovery
    and registration of tests.

    Class Attributes:
        test_registry: Main registry mapping test names to RegistryComponent instances
        pending_specs: Temporary storage for specs waiting to be associated with tests
        pending_tests: Temporary storage for test classes waiting to be registered
    """

    test_registry: ClassVar[dict[str, RegistryComponent]] = {}
    xfail_tests: ClassVar[defaultdict[str, dict[str, str]]] = defaultdict(lambda: defaultdict(str))
    pending_specs: ClassVar[defaultdict[str, dict[str, type[Any]]]] = defaultdict(dict)
    pending_tests: ClassVar[dict[str, type[Any]]] = {}

    @classmethod
    def auto_register_tests(cls) -> None:
        """Automatically register all pending tests with their associated op specs."""

        for test_name, test_class in cls.pending_tests.items():
            associated_specs = cls.pending_specs.get(test_name, {})
            cls.test_registry[test_name] = RegistryComponent(
                op_specs=associated_specs.copy(), test_class=test_class
            )

            cls.test_registry[test_name] = RegistryComponent(
                op_specs=cls.pending_specs[test_name], test_class=test_class
            )
            if test_name in cls.pending_specs:
                del cls.pending_specs[test_name]

            logger.debug(f"Registered test '{test_name}' with {len(associated_specs)} specs")

        cls.pending_tests.clear()

    @classmethod
    def get_registry(cls) -> dict[str, RegistryComponent]:
        """Get the complete test registry, performing auto-discovery if needed.

        If the registry is empty, this method triggers automatic discovery of
        all vertical tests and op specs under test.vertical subdirectory.

        Returns:
            Dictionary mapping test names to RegistryComponent instances
        """

        if cls.test_registry:
            return cls.test_registry

        discover_all_implementations("tests.vertical")
        cls.auto_register_tests()
        return cls.test_registry


def get_test_registry() -> dict[str, RegistryComponent]:
    """Get the vertical tests registry.

    Returns:
        Dictionary mapping test names to RegistryComponent instances
    """
    return VerticalTestsRegistry.get_registry()


def get_xfail_tests() -> dict[str, RegistryComponent]:
    """Get the xfail test list.

    Returns:
        Dictionary mapping test names to xfail reasons
    """
    # Need to force discover all modules first
    if not VerticalTestsRegistry.test_registry:
        VerticalTestsRegistry.get_registry()

    return VerticalTestsRegistry.xfail_tests


def register_spec(vertical_test: str, mark_xfail_reason: str | None = None):
    """Decorator to register an op spec for a vertical test.

    Args:
        vertical_test: Name of the vertical test this spec is associated with.
        mark_xfail_reason: Optional string to mark a spec to xfail.

    Returns:
        Decorator function that registers the specification class

    Raises:
        RuntimeError: If a spec with the same name is already registered

    Example:
        @register_spec("ShapeTest")
        class AddOpSpec:
            pass
    """

    def decorator(cls):
        spec_name = cls.__name__
        if spec_name in VerticalTestsRegistry.pending_specs[vertical_test]:
            raise RuntimeError(
                f"Error: attempt to re-define previously registered {spec_name} "
                f"old: {VerticalTestsRegistry.pending_specs[vertical_test][spec_name]}, new: {cls}"
            )

        VerticalTestsRegistry.pending_specs[vertical_test][spec_name] = cls
        if mark_xfail_reason is not None:
            VerticalTestsRegistry.xfail_tests[vertical_test][spec_name] = mark_xfail_reason

        return cls

    return decorator


def register_test():
    """Decorator to register a vertical test class.

    Returns:
        Decorator function that registers the test class

    Raises:
        RuntimeError: If a test with the same name is already registered

    Example:
        @register_test()
        class ShapeTest(VerticalTestBase):
            pass
    """

    def decorator(cls):
        test_name = cls.__name__
        if test_name in VerticalTestsRegistry.pending_tests:
            raise RuntimeError(
                f"Error: attempt to re-define previously registered {test_name} "
                f"old: {VerticalTestsRegistry.pending_tests[test_name]}, new: {cls}"
            )

        VerticalTestsRegistry.pending_tests[test_name] = cls
        return cls

    return decorator
