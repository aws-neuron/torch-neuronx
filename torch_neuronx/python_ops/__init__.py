import logging

import torch
import torch.library
from packaging.version import Version

from torch_neuronx.utils import TORCH_VERSION

logger = logging.getLogger(__name__)

# Create library instance at module level to prevent garbage collection
aten_lib = torch.library.Library("aten", "IMPL")


def _register_common_ops():
    """Register operations shared between legacy and MLIR paths."""
    import os

    from .gather import (
        _gather_op,
        _gather_out_op,
        gather_neuron,
        gather_out_neuron,
    )
    from .index import index_op
    from .native_multi_head_attention import (
        native_multi_head_attention_neuron,
        native_multi_head_attention_out_neuron,
    )
    from .nonzero import nonzero_op
    from .scaled_dot_product_fused_attention import (
        _fused_sdpa_op,
        _sdpa_backward_op,
        match_layout,
        scaled_dot_product_fused_attention_overrideable_backward_meta,
        scaled_dot_product_fused_attention_overrideable_backward_neuron,
        scaled_dot_product_fused_attention_overrideable_neuron,
    )
    from .scatter_add import (
        _scatter_add_inplace_op,
        _scatter_add_op,
        _scatter_add_out_op,
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

    use_cpp_dispatch = os.environ.get("TORCH_NEURONX_CPP_DISPATCH", "1") == "1"

    if use_cpp_dispatch:
        # gather.out has no NKI impl — register via _register_neuron_op (MLIR only)
        import torch_neuronx._C as _C

        from .auto_registration import make_composite_lower
        from .cpp_dispatch_helpers import make_lower_func

        _C._register_neuron_op(
            "gather.out",
            gather_out_neuron,
            make_composite_lower(_gather_out_op),
            static_argnums=[1],
            static_argnames=["sparse_grad"],
        )

        # scatter_add.out has no NKI impl — MLIR only
        _C._register_neuron_op(
            "scatter_add.out",
            scatter_add_out_neuron,
            make_composite_lower(_scatter_add_out_op),
            static_argnums=[1],
        )

        # ── SDPA Forward (NKI) — C++ dispatch handler ──
        # Registered via _register_neuron_op so the C++ DispatchSdpaFwdImpl
        # handles stride fixup and 9-tuple output without Python overhead.
        _C._register_neuron_op(
            "_scaled_dot_product_fused_attention_overrideable",
            scaled_dot_product_fused_attention_overrideable_neuron,
            make_composite_lower(_fused_sdpa_op),
            static_argnums=[],
            static_argnames=["dropout_p", "is_causal", "return_debug_mask", "scale"],
        )

        # ── SDPA Backward (NKI) — C++ dispatch handler ──
        _C._register_neuron_op(
            "_scaled_dot_product_fused_attention_overrideable_backward",
            scaled_dot_product_fused_attention_overrideable_backward_neuron,
            make_composite_lower(_sdpa_backward_op),
            static_argnums=[5, 8, 9, 10, 11, 12, 13],
            static_argnames=["scale"],
        )
        # ── Nonzero (MLIR with C++ post-processing) ──
        from .torch_mlir.ops.nonzero import NonzeroMLIRImpl

        nonzero_mlir = NonzeroMLIRImpl()
        nonzero_lower = nonzero_mlir.get_lower_func()
        _C._register_neuron_op(
            "nonzero",
            nonzero_op,
            nonzero_lower,
            static_argnames=["size", "with_count"],
        )
        _C._register_neuron_op(
            "nonzero.out",
            nonzero_op,
            nonzero_lower,
            static_argnames=["size", "with_count"],
        )

        # ── Index (MLIR with C++ boolean→integer preprocessing) ──
        from .torch_mlir.ops.index import IndexMLIRImpl

        index_mlir = IndexMLIRImpl()
        index_lower = index_mlir.get_lower_func()
        _C._register_neuron_op(
            "index.Tensor",
            index_op,
            index_lower,
        )
        _C._register_neuron_op(
            "index.Tensor_out",
            index_op,
            index_lower,
        )

    else:
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
        aten_lib.impl("scatter_add", scatter_add_neuron, "PrivateUse1")
        aten_lib.impl("scatter_add_", scatter_add_inplace_neuron, "PrivateUse1")
        aten_lib.impl("scatter_add.out", scatter_add_out_neuron, "PrivateUse1")
        aten_lib.impl("gather", gather_neuron, "PrivateUse1")
        aten_lib.impl("gather.out", gather_out_neuron, "PrivateUse1")

        # nonzero and index: Python dispatch fallback
        aten_lib.impl("nonzero", nonzero_op, "PrivateUse1")
        aten_lib.impl("nonzero.out", nonzero_op, "PrivateUse1")
        aten_lib.impl("index.Tensor", index_op, "PrivateUse1")
        aten_lib.impl("index.Tensor_out", index_op, "PrivateUse1")

    # Meta registration for backward — DTensor's sharding propagator needs it.
    # PyTorch 2.12+ already provides a Meta kernel for this op.
    if Version("2.12.0.dev0") > TORCH_VERSION:
        aten_lib.impl(
            "_scaled_dot_product_fused_attention_overrideable_backward",
            scaled_dot_product_fused_attention_overrideable_backward_meta,
            "Meta",
        )

    if Version("2.9.0") <= TORCH_VERSION:
        from .rms_norm import rms_norm_neuron

        if use_cpp_dispatch:
            import torch_neuronx._C as _C

            from .cpp_dispatch_helpers import make_lower_func

            _C._register_neuron_op(
                "rms_norm",
                rms_norm_neuron,
                make_lower_func(
                    rms_norm_neuron, "rms_norm", static_argnums=(1,), static_argnames=("eps",)
                ),
                static_argnums=[1],
                static_argnames=["eps"],
            )
        else:
            aten_lib.impl("rms_norm", rms_norm_neuron, "PrivateUse1")
        # Also register for AutogradPrivateUse1 to ensure gradient tracking works
        # when called through DTensor dispatch
        aten_lib.impl("rms_norm", rms_norm_neuron, "AutogradPrivateUse1")

        # cross_entropy_loss AutogradPrivateUse1 is registered via
        # _maybe_register_autograd in finalize_registrations_cpp instead.


def register_mlir_ops():
    """Register MLIR-based operations with PyTorch dispatcher."""
    from torch_neuronx.python_ops.torch_mlir import initialize_torch_mlir_backend
    from torch_neuronx.python_ops.torch_mlir.operation_registry import get_pending_operations

    from . import (
        contiguous_broadcast,
        contiguous_slice,
        contiguous_transpose,
    )
    from .auto_registration import auto_register_neuron_ops

    # Register common ops (attention, scatter, gather)
    _register_common_ops()

    import os

    use_cpp_dispatch = os.environ.get("TORCH_NEURONX_CPP_DISPATCH", "1") == "1"

    # The empty-HLO contiguous handler (priority 200) is only for the C++ dispatch
    # path — it produces a trivial clone kernel and relies on the C++ prologue to
    # handle all non-contiguous patterns.  When C++ dispatch is disabled, the
    # specialized Python handlers (broadcast, transpose, slice) handle contiguous.
    if use_cpp_dispatch:
        from . import contiguous_empty_hlo

    # Get the set of torch-mlir registered operations BEFORE registering @neuron_op ops
    # to skip @neuron_op registrations for ops that have torch-mlir implementations
    torch_mlir_ops = {reg_info["aten_name"] for reg_info in get_pending_operations()}

    # Auto-register @neuron_op ops
    auto_register_neuron_ops(
        aten_lib,
        skip_torch_mlir_ops=use_cpp_dispatch,
        torch_mlir_ops=torch_mlir_ops,
        use_cpp_dispatch=use_cpp_dispatch,
    )

    initialize_torch_mlir_backend(aten_lib, use_cpp_dispatch=use_cpp_dispatch)


def register_python_operations():
    """Main entry point for registering all Python operations."""
    register_mlir_ops()
