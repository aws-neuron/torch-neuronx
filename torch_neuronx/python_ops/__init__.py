import torch
import torch._C as _C
import torch.library
from packaging.version import Version

from torch_neuronx.utils import TORCH_VERSION

# Create library instance at module level to prevent garbage collection
aten_lib = torch.library.Library("aten", "IMPL")


def _register_common_ops():
    """Register operations shared between legacy and MLIR paths."""
    from .gather import gather_neuron, gather_out_neuron
    from .index import index_op
    from .native_multi_head_attention import (
        native_multi_head_attention_neuron,
        native_multi_head_attention_out_neuron,
    )
    from .nonzero import nonzero_op
    from .scaled_dot_product_fused_attention import (
        scaled_dot_product_fused_attention_overrideable_backward_meta,
        scaled_dot_product_fused_attention_overrideable_backward_neuron,
        scaled_dot_product_fused_attention_overrideable_neuron,
    )
    from .scatter_add import (
        scatter_add_inplace_neuron,
        scatter_add_neuron,
        scatter_add_out_neuron,
    )

    # Register the fused attention operations - with qkv and out projection for inference
    aten_lib.impl("_native_multi_head_attention", native_multi_head_attention_neuron, "PrivateUse1")
    aten_lib.impl(
        "_native_multi_head_attention.out",
        native_multi_head_attention_out_neuron,
        "PrivateUse1",
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

    # Register scatter_add operation
    aten_lib.impl("scatter_add", scatter_add_neuron, "PrivateUse1")
    aten_lib.impl("scatter_add_", scatter_add_inplace_neuron, "PrivateUse1")
    aten_lib.impl("scatter_add.out", scatter_add_out_neuron, "PrivateUse1")
    # Register gather operation
    aten_lib.impl("gather", gather_neuron, "PrivateUse1")
    aten_lib.impl("gather.out", gather_out_neuron, "PrivateUse1")

    # Register nonzero ops
    aten_lib.impl("nonzero", nonzero_op, "PrivateUse1")
    aten_lib.impl("nonzero.out", nonzero_op, "PrivateUse1")

    # Register index op
    aten_lib.impl("index.Tensor", index_op, "PrivateUse1")
    aten_lib.impl("index.Tensor_out", index_op, "PrivateUse1")

    if Version("2.9.0") <= TORCH_VERSION:
        from .rms_norm import rms_norm_neuron

        aten_lib.impl("rms_norm", rms_norm_neuron, "PrivateUse1")
        # Also register for AutogradPrivateUse1 to ensure gradient tracking works
        # when called through DTensor dispatch
        aten_lib.impl("rms_norm", rms_norm_neuron, "AutogradPrivateUse1")


def register_mlir_ops():
    """Register MLIR-based operations with PyTorch dispatcher."""
    from torch_neuronx.python_ops.torch_mlir import initialize_torch_mlir_backend

    from . import contiguous_broadcast, contiguous_slice, contiguous_transpose
    from .auto_registration import auto_register_neuron_ops

    # Register common ops (attention, scatter, gather)
    _register_common_ops()

    # Auto-register operations decorated with @neuron_op
    auto_register_neuron_ops(aten_lib)

    initialize_torch_mlir_backend(aten_lib)


def register_python_operations():
    """Main entry point for registering all Python operations."""
    register_mlir_ops()
