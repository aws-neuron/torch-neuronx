import torch
import torch._C as _C
import torch.library
from packaging.version import Version

from torch_neuronx.utils import TORCH_VERSION

from .addmm import AddmmOp
from .auto_registration import auto_register_neuron_ops
from .comparison import (
    EqScalarOp,
    EqTensorOp,
    GeScalarOp,
    GeTensorOp,
    GtScalarOp,
    GtTensorOp,
    LeScalarOp,
    LeTensorOp,
    LtScalarOp,
    LtTensorOp,
    NeScalarOp,
    NeTensorOp,
)
from .div import DivOp, DivScalarOp
from .min import MinOp
from .mul import MulOp, MulScalarOp
from .native_multi_head_attention import (
    native_multi_head_attention_neuron,
    native_multi_head_attention_out_neuron,
)
from .scaled_dot_product_fused_attention import (
    scaled_dot_product_fused_attention_overrideable_backward_meta,
    scaled_dot_product_fused_attention_overrideable_backward_neuron,
    scaled_dot_product_fused_attention_overrideable_neuron,
)

# Create library instance at module level to prevent garbage collection
aten_lib = torch.library.Library("aten", "IMPL")


def register_python_operations():
    """Register Python operations with PyTorch dispatcher"""

    # Create operation instances
    div_op = DivOp()
    div_scalar_op = DivScalarOp()
    # Create reduction operation instances
    addmm_op = AddmmOp()

    # Create comparison operation instances
    eq_scalar_op = EqScalarOp()
    ne_scalar_op = NeScalarOp()
    lt_scalar_op = LtScalarOp()
    le_scalar_op = LeScalarOp()
    gt_scalar_op = GtScalarOp()
    ge_scalar_op = GeScalarOp()

    eq_tensor_op = EqTensorOp()
    ne_tensor_op = NeTensorOp()
    lt_tensor_op = LtTensorOp()
    le_tensor_op = LeTensorOp()
    gt_tensor_op = GtTensorOp()
    ge_tensor_op = GeTensorOp()

    # Register div operations for Neuron backend
    def div_tensor_neuron(self, other):
        # Check if other is a scalar
        if isinstance(other, int | float | bool):
            return div_scalar_op(self, other)
        else:
            return div_op(self, other)

    def div_out_neuron(self, other, *, out):
        # Check if other is a scalar
        if isinstance(other, int | float | bool):
            return div_scalar_op(self, other, out=out)
        else:
            return div_op(self, other, out=out)

    def div_scalar_neuron(self, scalar):
        import logging

        logger = logging.getLogger(__name__)
        logger.debug(
            f"div_scalar_neuron called with "
            f"self={self.shape if hasattr(self, 'shape') else self}, scalar={scalar}"
        )
        return div_scalar_op(self, scalar)

    def div_scalar_out_neuron(self, scalar, *, out):
        return div_scalar_op(self, scalar, out=out)

    # Register scalar comparison operations
    def eq_scalar_out_neuron(self, other, *, out):
        return eq_scalar_op(self, other, out=out)

    def ne_scalar_out_neuron(self, other, *, out):
        return ne_scalar_op(self, other, out=out)

    def lt_scalar_out_neuron(self, other, *, out):
        return lt_scalar_op(self, other, out=out)

    def le_scalar_out_neuron(self, other, *, out):
        return le_scalar_op(self, other, out=out)

    def gt_scalar_out_neuron(self, other, *, out):
        return gt_scalar_op(self, other, out=out)

    def ge_scalar_out_neuron(self, other, *, out):
        return ge_scalar_op(self, other, out=out)

    # Register tensor comparison operations
    def eq_tensor_out_neuron(self, other, *, out):
        return eq_tensor_op(self, other, out=out)

    def ne_tensor_out_neuron(self, other, *, out):
        return ne_tensor_op(self, other, out=out)

    def lt_tensor_out_neuron(self, other, *, out):
        return lt_tensor_op(self, other, out=out)

    def le_tensor_out_neuron(self, other, *, out):
        return le_tensor_op(self, other, out=out)

    def gt_tensor_out_neuron(self, other, *, out):
        return gt_tensor_op(self, other, out=out)

    def ge_tensor_out_neuron(self, other, *, out):
        return ge_tensor_op(self, other, out=out)

    # Register addmm operation for Neuron backend
    def addmm_out_neuron(self, mat1, mat2, *, beta=1, alpha=1, out):
        return addmm_op(self, mat1, mat2, beta=beta, alpha=alpha, out=out)

    # Register addmm implementation
    aten_lib.impl("addmm.out", addmm_out_neuron, "PrivateUse1")

    # Register scalar comparison implementations
    aten_lib.impl("eq.Scalar_out", eq_scalar_out_neuron, "PrivateUse1")
    aten_lib.impl("ne.Scalar_out", ne_scalar_out_neuron, "PrivateUse1")
    aten_lib.impl("lt.Scalar_out", lt_scalar_out_neuron, "PrivateUse1")
    aten_lib.impl("le.Scalar_out", le_scalar_out_neuron, "PrivateUse1")
    aten_lib.impl("gt.Scalar_out", gt_scalar_out_neuron, "PrivateUse1")
    aten_lib.impl("ge.Scalar_out", ge_scalar_out_neuron, "PrivateUse1")

    # Register tensor comparison implementations
    aten_lib.impl("eq.Tensor_out", eq_tensor_out_neuron, "PrivateUse1")
    aten_lib.impl("ne.Tensor_out", ne_tensor_out_neuron, "PrivateUse1")
    aten_lib.impl("lt.Tensor_out", lt_tensor_out_neuron, "PrivateUse1")
    aten_lib.impl("le.Tensor_out", le_tensor_out_neuron, "PrivateUse1")
    aten_lib.impl("gt.Tensor_out", gt_tensor_out_neuron, "PrivateUse1")
    aten_lib.impl("ge.Tensor_out", ge_tensor_out_neuron, "PrivateUse1")

    # Register reduction implementations

    # Register the fused attention operations - with qkv and out projection for inference
    aten_lib.impl("_native_multi_head_attention", native_multi_head_attention_neuron, "PrivateUse1")
    aten_lib.impl(
        "_native_multi_head_attention.out", native_multi_head_attention_out_neuron, "PrivateUse1"
    )
    # For training - without qkv and out projection - with dropout and LSE - for training
    aten_lib.impl(
        "_scaled_dot_product_fused_attention_overrideable",
        scaled_dot_product_fused_attention_overrideable_neuron,
        "PrivateUse1",
    )
    aten_lib.impl(
        "_scaled_dot_product_fused_attention_overrideable_backward",
        scaled_dot_product_fused_attention_overrideable_backward_neuron,
        "PrivateUse1",
    )

    # We add for meta device because dtensor's sharding propagator runs
    # the aten op with meta device to get the output shapes and create
    # a graph
    aten_lib.impl(
        "_scaled_dot_product_fused_attention_overrideable_backward",
        scaled_dot_product_fused_attention_overrideable_backward_meta,
        "Meta",
    )
    if Version("2.9.0") <= TORCH_VERSION:
        from .rms_norm import rms_norm_neuron

        aten_lib.impl("rms_norm", rms_norm_neuron, "PrivateUse1")

    # Auto-register operations decorated with @neuron_op
    auto_register_neuron_ops(aten_lib)

    from torch_neuronx.python_ops.jax import (
        jax_impl,
        ops,  # Import to trigger decorator registration
    )
    from torch_neuronx.python_ops.jax.operation_registry import finalize_registrations
    from torch_neuronx.python_ops.meta import meta_impls  # Import meta implementations

    finalize_registrations(aten_lib)
