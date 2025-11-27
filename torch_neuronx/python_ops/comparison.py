"""Comparison operations (eq, ne, lt, le, gt, ge) with XLA implementation."""

from .base import ComparisonOperation
from .xla_ops.comparison_xla import (
    EqScalarXLAImpl,
    EqTensorXLAImpl,
    GeScalarXLAImpl,
    GeTensorXLAImpl,
    GtScalarXLAImpl,
    GtTensorXLAImpl,
    LeScalarXLAImpl,
    LeTensorXLAImpl,
    LtScalarXLAImpl,
    LtTensorXLAImpl,
    NeScalarXLAImpl,
    NeTensorXLAImpl,
)


def create_comparison_op(impl_class, op_name, description):
    """Factory function to create comparison operation classes.

    Args:
        impl_class: The XLA implementation class to use
        op_name: The operation name (e.g., "eq_scalar")
        description: The operation description
    """

    class ComparisonOp(ComparisonOperation):
        __doc__ = description

        def _setup_implementations(self):
            """Register available implementations."""
            self._implementations.append(impl_class())

        @property
        def op_name(self) -> str:
            return op_name

    return ComparisonOp


# Scalar comparison operations
EqScalarOp = create_comparison_op(
    EqScalarXLAImpl, "eq_scalar", "Element-wise equality comparison with scalar."
)

NeScalarOp = create_comparison_op(
    NeScalarXLAImpl, "ne_scalar", "Element-wise not-equal comparison with scalar."
)

LtScalarOp = create_comparison_op(
    LtScalarXLAImpl, "lt_scalar", "Element-wise less-than comparison with scalar."
)

LeScalarOp = create_comparison_op(
    LeScalarXLAImpl, "le_scalar", "Element-wise less-than-or-equal comparison with scalar."
)

GtScalarOp = create_comparison_op(
    GtScalarXLAImpl, "gt_scalar", "Element-wise greater-than comparison with scalar."
)

GeScalarOp = create_comparison_op(
    GeScalarXLAImpl, "ge_scalar", "Element-wise greater-than-or-equal comparison with scalar."
)

# Tensor comparison operations
EqTensorOp = create_comparison_op(
    EqTensorXLAImpl, "eq_tensor", "Element-wise equality comparison between tensors."
)

NeTensorOp = create_comparison_op(
    NeTensorXLAImpl, "ne_tensor", "Element-wise not-equal comparison between tensors."
)

LtTensorOp = create_comparison_op(
    LtTensorXLAImpl, "lt_tensor", "Element-wise less-than comparison between tensors."
)

LeTensorOp = create_comparison_op(
    LeTensorXLAImpl, "le_tensor", "Element-wise less-than-or-equal comparison between tensors."
)

GtTensorOp = create_comparison_op(
    GtTensorXLAImpl, "gt_tensor", "Element-wise greater-than comparison between tensors."
)

GeTensorOp = create_comparison_op(
    GeTensorXLAImpl, "ge_tensor", "Element-wise greater-than-or-equal comparison between tensors."
)
