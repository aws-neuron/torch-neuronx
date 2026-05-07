"""Tests for pickle support of dynamically created classes in auto_registration."""

import pickle
import sys

from torch_neuronx.python_ops.auto_registration import (
    _register_dynamic_class,
    create_auto_operation,
    create_auto_operation_typed,
)
from torch_neuronx.python_ops.base import (
    BinaryOperation,
    OperationImplementation,
    UnaryOperation,
)

MODULE_NAME = "torch_neuronx.python_ops.auto_registration"


class TestRegisterDynamicClass:
    """Tests for _register_dynamic_class."""

    def test_sets_name_qualname(self):
        cls = type("Original", (), {})
        _register_dynamic_class(cls, "RegisteredName")

        assert cls.__name__ == "RegisteredName"
        assert cls.__qualname__ == "RegisteredName"

    def test_name_equals_qualname(self):
        """Dynamo's TYPE_MATCH guard requires __name__ == __qualname__."""
        cls = type("X", (), {})
        _register_dynamic_class(cls, "MyClass")
        assert cls.__name__ == cls.__qualname__

    def test_class_registered_in_module_dict(self):
        cls = type("Temp", (), {})
        _register_dynamic_class(cls, "TestRegisteredClass")

        module = sys.modules[cls.__module__]
        assert module.__dict__["TestRegisteredClass"] is cls

    def test_registered_class_is_pickleable(self):
        cls = type("Unpickleable", (), {})
        _register_dynamic_class(cls, "PickleableClass")

        restored = pickle.loads(pickle.dumps(cls))
        assert restored is cls

    def test_instance_of_registered_class_is_pickleable(self):
        cls = type("Inst", (), {})
        _register_dynamic_class(cls, "PickleableInstanceClass")

        obj = cls()
        restored = pickle.loads(pickle.dumps(obj))
        assert isinstance(restored, cls)


class _StubImpl(OperationImplementation):
    """Minimal implementation for testing create_auto_operation."""

    _auto_priority = 10

    def can_handle(self, *args, **kwargs):
        return True

    def execute(self, *args, **kwargs):
        return None

    def _execute_impl(self, *args, **kwargs):
        return None


class TestCreateAutoOperationPickle:
    """Tests that create_auto_operation registers classes for pickle."""

    def test_auto_operation_class_in_module(self):
        op = create_auto_operation("aten::test_pickle_op", [_StubImpl])
        cls = type(op)
        expected_name = "AutoOperation_aten_test_pickle_op"

        assert cls.__name__ == expected_name
        module = sys.modules[MODULE_NAME]
        assert module.__dict__[expected_name] is cls

    def test_auto_operation_class_is_pickleable(self):
        op = create_auto_operation("aten::test_pickle_cls", [_StubImpl])
        cls = type(op)

        restored = pickle.loads(pickle.dumps(cls))
        assert restored is cls

    def test_priority_impl_registered_for_pickle(self):
        _ = create_auto_operation("aten::test_priority_pickle", [_StubImpl])
        expected_name = f"PriorityImpl_{_StubImpl.__name__}_{_StubImpl._auto_priority}"

        module = sys.modules[MODULE_NAME]
        assert expected_name in module.__dict__

        priority_cls = module.__dict__[expected_name]
        restored = pickle.loads(pickle.dumps(priority_cls))
        assert restored is priority_cls

    def test_dotted_aten_name(self):
        """Dots in aten names like 'aten::add.Tensor' are replaced with underscores."""
        op = create_auto_operation("aten::add.Tensor", [_StubImpl])
        cls = type(op)
        assert cls.__name__ == "AutoOperation_aten_add_Tensor"


class TestCreateAutoOperationTypedPickle:
    """Tests that create_auto_operation_typed registers classes for pickle."""

    def test_typed_operation_class_in_module(self):
        op = create_auto_operation_typed("aten::test_typed_pickle", [_StubImpl], UnaryOperation)
        cls = type(op)
        expected_name = "TypedAutoOperation_aten_test_typed_pickle"

        assert cls.__name__ == expected_name
        module = sys.modules[MODULE_NAME]
        assert module.__dict__[expected_name] is cls

    def test_typed_operation_class_is_pickleable(self):
        op = create_auto_operation_typed(
            "aten::test_typed_pickle_cls", [_StubImpl], BinaryOperation
        )
        cls = type(op)

        restored = pickle.loads(pickle.dumps(cls))
        assert restored is cls

    def test_typed_priority_impl_registered(self):
        _ = create_auto_operation_typed("aten::test_typed_priority", [_StubImpl], UnaryOperation)
        expected_name = f"PriorityImpl_{_StubImpl.__name__}_{_StubImpl._auto_priority}"

        module = sys.modules[MODULE_NAME]
        assert expected_name in module.__dict__
