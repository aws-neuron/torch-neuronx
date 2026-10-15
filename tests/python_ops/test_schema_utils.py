"""Tests for schema-driven op introspection (torch_neuronx.python_ops.schema_utils)."""

import pytest

from torch_neuronx.python_ops.schema_utils import _inplace_from_name, is_inplace_op

# Ops whose first argument is annotated Tensor(a!) — genuinely mutating.
INPLACE_OPS = [
    "aten::add_.Tensor",
    "aten::mul_.Tensor",
    "aten::zero_",
    "aten::fill_.Scalar",
    "aten::copy_",
    "aten::normal_",
    "aten::uniform_",
    "aten::random_",
    "aten::relu_",
    "aten::bitwise_right_shift_.Tensor",
    # The operator-form compound-assignment dunders really are in-place, and
    # must stay classified as such now that detection is schema-based.
    "aten::__irshift__.Tensor",
    "aten::__ilshift__.Scalar",
    "aten::__iand__.Tensor",
]

# Pure ops, including the operator-form dunders whose names end in "_".
FUNCTIONAL_OPS = [
    "aten::add.Tensor",
    "aten::mul.Tensor",
    "aten::bitwise_right_shift.Tensor",
    "aten::bitwise_right_shift.Tensor_out",
    "aten::__rshift__.Tensor",
    "aten::__rshift__.Scalar",
    "aten::__rshift__.Tensor_out",
    "aten::__rshift__.Scalar_out",
    "aten::__lshift__.Tensor",
    "aten::__lshift__.Scalar",
    "aten::__lshift__.Tensor_out",
    "aten::__lshift__.Scalar_out",
    "aten::__and__.Tensor",
]


@pytest.mark.parametrize("aten_name", INPLACE_OPS)
def test_inplace_ops_detected(aten_name):
    assert is_inplace_op(aten_name) is True


@pytest.mark.parametrize("aten_name", FUNCTIONAL_OPS)
def test_functional_ops_not_inplace(aten_name):
    """Regression: a trailing "_" in a dunder name must not imply in-place.

    The old name-based heuristic classified `__rshift__` as in-place because
    the name ends in "_", which made the dispatcher alias `out` to `self`
    (corrupting `self`) and size the output buffer from `self` (breaking
    broadcasting).
    """
    assert is_inplace_op(aten_name) is False


def test_names_without_a_schema_fall_back_to_the_name_heuristic():
    """Unqualified aliases and internal helpers have no resolvable schema."""
    # No such overload — must not raise.
    assert is_inplace_op("aten::bitwise_and") is False
    assert is_inplace_op("aten::add_") is True
    assert is_inplace_op("aten::not_a_real_op_xyz") is False
    assert is_inplace_op("aten::not_a_real_op_xyz_") is True


def test_fallback_heuristic_matches_the_expression_it_replaced():
    """Pin `_inplace_from_name` to the original name-based expression.

    Schema-less names (unqualified aliases, internal helpers) still go through
    this heuristic, so it must stay byte-for-byte equivalent to the code it
    replaced in `Operation.is_inplace` / `TorchMlirKernel._is_inplace_op`.
    Without this, a later tidy-up of the heuristic could silently change
    behavior for every op that has no resolvable schema.
    """

    def original(name):
        return "_." in name or (name.endswith("_") and "." not in name)

    names = [
        "add_",
        "add_.Tensor",
        "add.Tensor",
        "zero_",
        "relu_",
        "copy_",
        "__rshift__",
        "__irshift__",
        "bitwise_and",
        "foo.Tensor_",
        "_foreach_add_.List",
        "cross_entropy_loss_backward_",
        "a_.b",
        "",
    ]
    mismatches = [n for n in names if _inplace_from_name(n) != original(n)]
    assert not mismatches, f"heuristic diverged from the original for: {mismatches}"


def test_accepts_names_with_and_without_the_aten_prefix():
    assert is_inplace_op("aten::add_.Tensor") is is_inplace_op("add_.Tensor") is True
    assert is_inplace_op("aten::__rshift__.Tensor") is is_inplace_op("__rshift__.Tensor") is False
