"""Auto-generated shape registry entries for pending ops."""

# Keep co-varying list lengths (e.g. foreach scalars[] alongside self[]) aligned
# with the tensor-list default count used by _tensor_list_from_spec. If this
# constant changes, regenerate this file.
from .neuron_op_info import TENSOR_LIST_DEFAULT_COUNT as _N


def register_all_defaults(r, smoke_subsets=()):
    """Register all shape entries (756 total).

    ``smoke_subsets`` names the accuracy smoke test subsets to register on top of the builtin
    defaults (see ``SMOKE_SUBSET_SHAPES``). It is an allowlist and it is empty by default, so a
    plain run sees exactly the builtin pool it always has. ``--traced-model`` names a subset.
    The allowlist keeps the ops of a subset out of the pre-existing vertical tests.
    """

    # Elementwise (self-only ops)
    unary_ops = [
        "aten::abs",
        "aten::neg",
        "aten::relu",
        "aten::sigmoid",
        "aten::tanh",
        "aten::exp",
        "aten::log",
        "aten::sqrt",
        "aten::rsqrt",
        "aten::sin",
        "aten::cos",
        "aten::gelu",
        "aten::silu",
        "aten::erf",
        "aten::ceil",
        "aten::floor",
    ]
    # (M, H) shapes: original coverage dims + dynamic-NKI MoE sweep dims
    # (H=2048/4096/12288), swept by test_dynamic_aten_ops.py (silu among these).
    elem_shapes = [
        (2374, 4096), (64, 4096), (256, 1024),
        (128, 2048), (256, 2048), (512, 2048), (1024, 2048), (2048, 2048),
        (512, 4096), (1024, 4096), (2048, 4096), (4096, 4096),
        (512, 12288), (1024, 12288), (2048, 12288),
    ]  # fmt: skip
    for op in unary_ops:
        for s in elem_shapes:
            r(op, {"self": s}, source="builtin")

    # Binary elementwise — same (M, H) dims (add / mul swept by the dynamic test).
    binary_ops = ["aten::add.Tensor", "aten::sub.Tensor", "aten::mul.Tensor", "aten::div.Tensor"]
    for op in binary_ops:
        for s in elem_shapes:
            r(op, {"self": s, "other": s}, source="builtin")

    # Matmul — hidden/intermediate dims + dynamic-NKI MoE sweep dims (mm swept by
    # the dynamic test).
    mm_shapes = [
        # shapes commented out until the M-tiled at 128 constraint for dynamic mm is addressed
        # (2374, 4096, 4096),  # seq x hidden x hidden
        # (64, 4096, 4096),  # short seq
        (256, 1024, 3072),  # smaller model, intermediate
        # (192, 4096, 11008),  # llama gate_proj
        # (16, 4096, 4096),  # very short seq
        (128, 2048, 2048), (256, 2048, 4096), (512, 2048, 4096), (1024, 2048, 4096),
        (2048, 2048, 4096), (512, 4096, 4096), (1024, 4096, 4096), (2048, 4096, 4096),
        (256, 1024, 1024), (512, 2048, 2048),
    ]  # fmt: skip
    for m, k, n in mm_shapes:
        r("aten::mm", {"self": (m, k), "mat2": (k, n)}, source="builtin")
        r("aten::addmm", {"self": (n,), "mat1": (m, k), "mat2": (k, n)}, source="builtin")
        r("aten::linear", {"input": (m, k), "weight": (n, k), "bias": (n,)}, source="builtin")
        r("aten::linear", {"input": (m, k), "weight": (n, k), "bias": None}, source="builtin")

    for b, m, k, n in [(2, 2374, 128, 128), (4, 256, 64, 64), (8, 64, 128, 128)]:
        r("aten::bmm", {"self": (b, m, k), "mat2": (b, k, n)}, source="builtin")

    # Normalization — (batch, channels/hidden, seq/features)
    norm_shapes = [
        (1, 32, 4096),  # batch=1, heads=32, hidden=4096
        (2, 64, 1024),  # batch=2, channels=64
        (1, 128, 3072),  # larger intermediate
    ]
    for n, ch, h in norm_shapes:
        r(
            "aten::layer_norm",
            {"input": (n, ch, h), "normalized_shape": (h,), "weight": (h,), "bias": (h,)},
            source="builtin",
        )
        r(
            "aten::native_layer_norm",
            {"input": (n, ch, h), "weight": (h,), "bias": (h,)},
            source="builtin",
        )
        r(
            "aten::batch_norm_elemt",
            {"input": (n, ch, h), "weight": (ch,), "bias": (ch,), "mean": (ch,), "invstd": (ch,)},
            source="builtin",
        )
        r("aten::batch_norm_stats", {"input": (n, ch, h)}, source="builtin")

    # SDPA — with 8K and 16K seq lengths
    sdpa_shapes = [
        (1, 32, 8192, 128),  # llama-7b 8K context
        (2, 8, 4096, 64),  # smaller model, batch>1
    ]
    for b, h, s, d in sdpa_shapes:
        r(
            "aten::_scaled_dot_product_fused_attention_overrideable",
            {"query": (b, h, s, d), "key": (b, h, s, d), "value": (b, h, s, d)},
            source="builtin",
        )

    # Reductions
    for op in ["aten::sum", "aten::mean", "aten::amax", "aten::amin"]:
        r(op, {"self": (2374, 4096)}, source="builtin")
        r(op, {"self": (64, 4096)}, source="builtin")
        r(op, {"self": (256, 3072)}, source="builtin")

    r(
        "aten::native_group_norm",
        {
            "input": (1, 64, 4096),
            "weight": None,
            "bias": None,
            "N": 1,
            "C": 64,
            "HxW": 4096,
            "group": 8,
            "eps": 1e-5,
        },
        source="builtin",
    )
    r(
        "aten::_native_batch_norm_legit",
        {
            "input": (2, 64, 1024),
            "weight": (64,),
            "bias": (64,),
            "running_mean": (64,),
            "running_var": (64,),
            "training": True,
            "momentum": 0.1,
            "eps": 1e-5,
        },
        source="builtin",
    )
    r(
        "aten::_native_batch_norm_legit",
        {
            "input": {"shape": (2, 64, 1024), "dtype": "bfloat16"},
            "weight": {"shape": (64,), "dtype": "bfloat16"},
            "bias": {"shape": (64,), "dtype": "bfloat16"},
            "running_mean": (64,),
            "running_var": (64,),
            "training": True,
            "momentum": 0.1,
            "eps": 1e-5,
        },
        source="builtin",
    )
    r(
        "aten::_native_batch_norm_legit_functional",
        {
            "input": (2, 64, 1024),
            "weight": (64,),
            "bias": (64,),
            "running_mean": (64,),
            "running_var": (64,),
            "training": True,
            "momentum": 0.1,
            "eps": 1e-5,
        },
        source="builtin",
    )
    r(
        "aten::_native_batch_norm_legit_no_training",
        {
            "input": (2, 64, 1024),
            "weight": (64,),
            "bias": (64,),
            "running_mean": (64,),
            "running_var": (64,),
            "momentum": 0.1,
            "eps": 1e-5,
        },
        source="builtin",
    )
    r(
        "aten::native_batch_norm_backward",
        {
            "grad_out": (2, 64, 1024),
            "input": (2, 64, 1024),
            "weight": (64,),
            "running_mean": (64,),
            "running_var": (64,),
            "save_mean": (64,),
            "save_invstd": (64,),
            "train": True,
            "eps": 1e-5,
            "output_mask": [True, True, True],
        },
        source="builtin",
    )

    # Loss ops
    r(
        "aten::binary_cross_entropy",
        {
            "self": {"shape": (256, 1024), "sigmoid": True},
            "target": {"shape": (256, 1024), "dtype": "float", "sigmoid": True},
        },
        source="builtin",
    )
    r(
        "aten::binary_cross_entropy.out",
        {
            "self": {"shape": (256, 1024), "sigmoid": True},
            "target": {"shape": (256, 1024), "dtype": "float", "sigmoid": True},
            "out": (0,),
        },
        source="builtin",
    )
    r(
        "aten::binary_cross_entropy_with_logits",
        {"self": (256, 1024), "target": {"shape": (256, 1024), "dtype": "float"}},
        source="builtin",
    )
    r(
        "aten::kl_div",
        {"self": (256, 1024), "target": {"shape": (256, 1024), "dtype": "float"}},
        source="builtin",
    )
    r(
        "aten::huber_loss_backward",
        {
            "grad_output": (256, 1024),
            "self": (256, 1024),
            "target": (256, 1024),
            "reduction": 1,
            "delta": 1.0,
        },
        source="builtin",
    )
    r(
        "aten::nll_loss_forward.output",
        {
            "self": (256, 1024),
            "target": {"shape": (256,), "dtype": "long", "low": 0, "high": 1024},
            "weight": None,
            "reduction": 1,
            "ignore_index": -100,
            "output": (0,),
            "total_weight": (0,),
        },
        source="builtin",
    )
    r(
        "aten::nll_loss_backward.grad_input",
        {
            "grad_output": (1,),
            "self": (256, 1024),
            "target": {"shape": (256,), "dtype": "long", "low": 0, "high": 1024},
            "weight": None,
            "reduction": 1,
            "ignore_index": -100,
            "total_weight": (1,),
            "grad_input": (256, 1024),
        },
        source="builtin",
    )
    r(
        "aten::soft_margin_loss_backward.grad_input",
        {
            "grad_output": (256, 1024),
            "self": (256, 1024),
            "target": (256, 1024),
            "reduction": 1,
            "grad_input": (256, 1024),
        },
        source="builtin",
    )
    r(
        "aten::index.Tensor",
        {
            "self": (100, 64),
            "indices": [{"shape": (1, 3), "dtype": "long", "low": -100, "high": -1}],
        },
        source="builtin",
    )
    # (T, H, n) gather/scatter MoE sweep dims (index / scatter_add_ swept by
    # test_dynamic_aten_ops.py). n rows selected from T.
    gather_shapes = [
        (2048, 2048, 512), (2048, 2048, 2048), (4096, 2048, 1024), (2048, 4096, 1024),
        (4096, 4096, 2048), (1024, 2048, 512), (8192, 2048, 4096), (4096, 4096, 4096),
        (2048, 12288, 1024), (512, 2048, 256),
    ]  # fmt: skip
    for t, h, n in gather_shapes:
        r(
            "aten::index.Tensor",
            {
                "self": (t, h),
                "indices": [{"shape": (n,), "dtype": "long", "low": 0, "high": t}],
            },
            source="builtin",
        )
        # scatter_add_ index is a 1-D (n,) row index broadcast to (n, H) with
        # stride(1)==0 (via the ``expand`` directive). The static NKI scatter_add
        # only routes to its fast kernel when that stride is 0; a materialized
        # contiguous (n, H) index falls back to the slow GPSIMD path.
        # ``allocate_to_device`` re-applies the broadcast on-device so the zero
        # stride survives the move. ``distinct`` draws the n row indices without
        # replacement so no two src rows accumulate into the same output row
        r(
            "aten::scatter_add_",
            {
                "self": (t, h),
                "dim": 0,
                "index": {
                    "shape": (n,),
                    "dtype": "long",
                    "low": 0,
                    "high": t,
                    "distinct": True,
                    "expand": [n, h],
                },
                "src": (n, h),
            },
            source="builtin",
        )
        # _index_put_impl_ shares the dynamic scatter_add kernel, but only on its
        # accumulate=True path (accumulate=False is a scatter/overwrite the dynamic kernel
        # declines)
        r(
            "aten::_index_put_impl_",
            {
                "self": (t, h),
                "indices": [
                    {
                        "shape": (n,),
                        "dtype": "long",
                        "low": 0,
                        "high": t,
                        "distinct": True,
                    }
                ],
                "values": (n, h),
                "accumulate": True,
                "unsafe": False,
            },
            source="builtin",
        )
    r(
        "aten::index.Tensor_out",
        {
            "self": (2374, 4096),
            "indices": [{"shape": (64,), "dtype": "long", "low": 0, "high": 2374}],
            "out": (0,),
        },
        source="builtin",
    )
    r(
        "aten::_index_put_impl",
        {
            "self": (2374, 4096),
            "indices": [{"shape": (64,), "dtype": "long", "low": 0, "high": 2374}],
            "values": (64, 4096),
            "accumulate": False,
            "unsafe": False,
        },
        source="builtin",
    )
    r(
        "aten::index_put_",
        {
            "self": (2374, 4096),
            "indices": [{"shape": (64,), "dtype": "long", "low": 0, "high": 2374}],
            "values": (64, 4096),
        },
        source="builtin",
    )
    r(
        "aten::index_put_",
        {
            "self": (100, 64),
            "indices": [{"shape": (8,), "dtype": "long", "low": -100, "high": -1}],
            "values": (8, 64),
        },
        source="builtin",
    )
    r(
        "aten::_unsafe_index.Tensor",
        {
            "self": (2374, 4096),
            "indices": [{"shape": (64,), "dtype": "long", "low": 0, "high": 2374}],
        },
        source="builtin",
    )
    r(
        "aten::_unsafe_index_put",
        {
            "self": (2374, 4096),
            "indices": [{"shape": (64,), "dtype": "long", "low": 0, "high": 2374}],
            "values": (64, 4096),
        },
        source="builtin",
    )
    r(
        "aten::index_copy",
        {
            "self": (2374, 4096),
            "dim": 0,
            "index": {"shape": (64,), "dtype": "long", "low": 0, "high": 2374},
            "source": (64, 4096),
        },
        source="builtin",
    )
    r(
        "aten::index_select",
        {
            "self": (2374, 4096),
            "dim": 0,
            "index": {"shape": (64,), "dtype": "long", "low": 0, "high": 2374},
        },
        source="builtin",
    )

    # Scatter ops
    r(
        "aten::scatter.value_out",
        {
            "self": (256, 1024),
            "dim": 0,
            "index": {"shape": (64, 1024), "dtype": "long", "low": 0, "high": 256},
            "value": 1.0,
            "out": (256, 1024),
        },
        source="builtin",
    )
    r(
        "aten::scatter.src_out",
        {
            "self": (256, 1024),
            "dim": 0,
            "index": {"shape": (64, 1024), "dtype": "long", "low": 0, "high": 256},
            "src": (64, 1024),
            "out": (256, 1024),
        },
        source="builtin",
    )

    # Stack/view ops
    r("aten::_stack", {"tensors": (256, 4096), "dim": 0}, source="builtin")
    r("aten::_unsafe_view", {"self": (16, 4096), "size": [65536]}, source="builtin")
    r("aten::chain_matmul", {"matrices": (256, 256)}, source="builtin")

    # Pooling
    r("aten::avg_pool1d", {"self": (1, 64, 4096), "kernel_size": [2]}, source="builtin")

    # Upsample
    r(
        "aten::upsample_bilinear2d.vec",
        {
            "input": (1, 64, 32, 32),
            "output_size": None,
            "align_corners": False,
            "scale_factors": [2.0, 2.0],
        },
        source="builtin",
    )
    r(
        "aten::upsample_nearest2d.vec",
        {"input": (1, 64, 32, 32), "output_size": None, "scale_factors": [2.0, 2.0]},
        source="builtin",
    )

    # Col2im
    r(
        "aten::col2im",
        {
            "self": (1, 576, 256),
            "output_size": [16, 16],
            "kernel_size": [3, 3],
            "dilation": [1, 1],
            "padding": [1, 1],
            "stride": [1, 1],
        },
        source="builtin",
    )

    # Foreach ops
    for suffix in ["", "_"]:
        r(
            f"aten::_foreach_clamp_max{suffix}.ScalarList",
            {"self": (256, 4096), "scalars": [1.0] * _N},
            source="builtin",
        )
        r(
            f"aten::_foreach_clamp_min{suffix}.ScalarList",
            {"self": (256, 4096), "scalars": [1.0] * _N},
            source="builtin",
        )
        r(
            f"aten::_foreach_pow{suffix}.ScalarList",
            {"self": (256, 4096), "exponent": [2.0] * _N},
            source="builtin",
        )
    for suffix in ["", "_"]:
        r(
            f"aten::_foreach_addcdiv{suffix}.Tensor",
            {
                "self": (256, 4096),
                "tensor1": (256, 4096),
                "tensor2": (256, 4096),
                "scalars": {"shape": (_N,), "dtype": "float"},
            },
            source="builtin",
        )
        r(
            f"aten::_foreach_addcmul{suffix}.Tensor",
            {
                "self": (256, 4096),
                "tensor1": (256, 4096),
                "tensor2": (256, 4096),
                "scalars": {"shape": (_N,), "dtype": "float"},
            },
            source="builtin",
        )
    r("aten::_foreach_sub.Tensor", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r("aten::_foreach_sub_.Tensor", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")

    # --- Category A: registered ops missing shapes ---

    # Fused RMS norm
    r(
        "aten::_fused_rms_norm",
        {"input": (4, 4096, 4096), "normalized_shape": [4096], "weight": (4096,), "eps": 1e-5},
        source="builtin",
    )
    r(
        "aten::_fused_rms_norm_backward",
        {
            "grad_out": (4, 4096, 4096),
            "input": (4, 4096, 4096),
            "normalized_shape": [4096],
            "rstd": (4, 4096, 1),
            "weight": (4096,),
            "output_mask": [True, True],
        },
        source="builtin",
    )

    # addmv_: matrix-vector multiply accumulate
    r("aten::addmv_", {"self": (8192,), "mat": (8192, 8192), "vec": (8192,)}, source="builtin")

    # Batch norm internals
    r(
        "aten::batch_norm_backward_elemt",
        {
            "grad_out": (32, 64, 4096),
            "input": (32, 64, 4096),
            "mean": (64,),
            "invstd": (64,),
            "weight": (64,),
            "sum_dy": (64,),
            "sum_dy_xmu": (64,),
            "count": (64,),
        },
        source="builtin",
    )
    r(
        "aten::batch_norm_backward_reduce",
        {
            "grad_out": (32, 64, 4096),
            "input": (32, 64, 4096),
            "mean": (64,),
            "invstd": (64,),
            "weight": (64,),
            "input_g": True,
            "weight_g": True,
            "bias_g": True,
        },
        source="builtin",
    )
    r(
        "aten::batch_norm_elemt",
        {
            "input": (32, 64, 4096),
            "weight": (64,),
            "bias": (64,),
            "mean": (64,),
            "invstd": (64,),
            "eps": 1e-5,
        },
        source="builtin",
    )
    r(
        "aten::batch_norm_gather_stats_with_counts",
        {
            "input": (32, 64, 4096),
            "mean": (1, 64),
            "invstd": (1, 64),
            "running_mean": (64,),
            "running_var": (64,),
            "momentum": 0.1,
            "eps": 1e-5,
            "counts": (1,),
        },
        source="builtin",
    )
    r("aten::batch_norm_stats", {"input": (32, 64, 4096), "eps": 1e-5}, source="builtin")
    r(
        "aten::_weight_norm_interface",
        {"v": (8192, 8192), "g": (8192, 1), "dim": 0},
        source="manual",
    )
    # CPU correctness is skipped for this op (see _OPS_SKIP_CPU_CORRECTNESS): the
    # harness fills saved_norms with independent randn, violating
    # saved_norms==‖saved_v‖; the 1/norm**3 term then diverges. Correctness is
    # covered by upstream gradcheck + unit/e2e tests; lowering verified exact vs
    # CPU with consistent inputs.
    r(
        "aten::_weight_norm_interface_backward",
        {
            "grad_w": (8192, 8192),
            "saved_v": (8192, 8192),
            "saved_g": (8192, 1),
            "saved_norms": (8192, 1),
            "dim": 0,
        },
        source="manual",
    )

    # Bitwise shift with scalar LHS
    r(
        "aten::bitwise_left_shift.Scalar_Tensor",
        {"self": 8, "other": {"shape": (4096, 4096), "dtype": "int"}},
        source="builtin",
    )
    r(
        "aten::bitwise_right_shift.Scalar_Tensor",
        {"self": 256, "other": {"shape": (4096, 4096), "dtype": "int"}},
        source="builtin",
    )

    # Clamp with tensor bounds
    r(
        "aten::clamp.Tensor",
        {"self": (4096, 4096), "min": (4096, 4096), "max": (4096, 4096)},
        source="builtin",
    )

    # Convolution backward
    r(
        "aten::convolution_backward",
        {
            "grad_output": (8, 64, 56, 56),
            "input": (8, 3, 224, 224),
            "weight": (64, 3, 7, 7),
            "bias_sizes": [64],
            "stride": [2, 2],
            "padding": [3, 3],
            "dilation": [1, 1],
            "transposed": False,
            "output_padding": [0, 0],
            "groups": 1,
            "output_mask": [True, True, True],
        },
        source="builtin",
    )

    # Embedding backward
    r(
        "aten::embedding_dense_backward",
        {
            "grad_output": (4, 2048, 4096),
            "indices": {"shape": (4, 2048), "dtype": "long", "low": 0, "high": 32000},
            "num_weights": 32000,
            "padding_idx": -1,
            "scale_grad_by_freq": False,
        },
        source="builtin",
    )

    # Eye
    r("aten::eye.m", {"n": 4096, "m": 4096}, source="builtin")
    # Explicit .m_out because FakeTensorMode probe picks the wrong functional
    # variant (aten::eye.default only takes n, not (n, m)), so auto-injection
    # of out= fails for the .m_out overload.
    r("aten::eye.m_out", {"n": 4096, "m": 4096, "out": (4096, 4096)}, source="builtin")

    # Lerp
    r(
        "aten::lerp.Scalar",
        {"self": (4096, 4096), "end": (4096, 4096), "weight": 0.5},
        source="builtin",
    )
    r(
        "aten::lerp.Tensor",
        {"self": (4096, 4096), "end": (4096, 4096), "weight": (4096, 4096)},
        source="builtin",
    )
    r(
        "aten::lerp_.Tensor",
        {"self": (4096, 4096), "end": (4096, 4096), "weight": (4096, 4096)},
        source="builtin",
    )

    # Linear backward
    r(
        "aten::linear_backward",
        {
            "self": (4, 2048, 4096),
            "grad_output": (4, 2048, 4096),
            "weight": (4096, 4096),
            "output_mask": [True, True, True],
        },
        source="builtin",
    )

    # Max/min with dim
    r("aten::max.dim", {"self": (4096, 4096), "dim": 1}, source="builtin")
    r("aten::min.dim", {"self": (4096, 4096), "dim": 1}, source="builtin")
    r("aten::mean.dim", {"self": (4096, 4096), "dim": [1]}, source="builtin")

    # Norm backward ops
    r(
        "aten::native_group_norm_backward",
        {
            "grad_out": (4, 64, 4096),
            "input": (4, 64, 4096),
            # mean/rstd carry per-group stats with shape (N, group). PyTorch's
            # meta_native_group_norm_backward (torch/_meta_registrations.py)
            # rejects the (N, group, 1) shape a previous entry used with
            # ``Expect mean to have shape (N, group)``, which the harness then
            # surfaces as a Dynamo FakeTensor RuntimeError. Verified 2026-07-20.
            "mean": (4, 4),
            "rstd": (4, 4),
            "weight": (64,),
            "N": 4,
            "C": 64,
            "HxW": 4096,
            "group": 4,
            "output_mask": [True, True, True],
        },
        source="builtin",
    )
    r(
        "aten::native_layer_norm_backward",
        {
            "grad_out": (4, 2048, 4096),
            "input": (4, 2048, 4096),
            "normalized_shape": [4096],
            "mean": (4, 2048, 1),
            "rstd": (4, 2048, 1),
            "weight": (4096,),
            "bias": (4096,),
            "output_mask": [True, True, True],
        },
        source="builtin",
    )

    # Scatter
    r(
        "aten::scatter.src",
        {
            "self": (4096, 4096),
            "dim": 1,
            "index": {"shape": (4096, 256), "dtype": "long", "low": 0, "high": 4096},
            "src": (4096, 256),
        },
        source="builtin",
    )
    r(
        "aten::scatter.reduce",
        {
            "self": (4096, 4096),
            "dim": 1,
            "index": {"shape": (4096, 256), "dtype": "long", "low": 0, "high": 4096},
            "src": (4096, 256),
            "reduce": "add",
        },
        source="builtin",
    )

    # Select/slice backward — output is input_sizes, grad_output is a slice of it
    r(
        "aten::select_backward",
        {"grad_output": (4096,), "input_sizes": [8192, 4096], "dim": 0, "index": 0},
        source="builtin",
    )
    r(
        "aten::slice_backward",
        {
            "grad_output": (4096, 4096),
            "input_sizes": [8192, 4096],
            "dim": 0,
            "start": 0,
            "end": 4096,
            "step": 1,
        },
        source="builtin",
    )

    # Sum with dim
    r("aten::sum.dim_IntList", {"self": (4096, 4096), "dim": [1]}, source="builtin")

    # Value selecting reduction backward (for max/min backward)
    r(
        "aten::value_selecting_reduction_backward",
        {
            "grad": (4096,),
            "dim": 1,
            "indices": {"shape": (4096,), "dtype": "long", "low": 0, "high": 4096},
            "sizes": [4096, 4096],
            "keepdim": False,
        },
        source="builtin",
    )

    # Index put impl (in-place) — accumulate=True + a 1-D distinct row index
    # so it routes to the dynamic scatter_add kernel (the accumulate=False /
    # materialized-index form declines to the default lowering).
    r(
        "aten::_index_put_impl_",
        {
            "self": (4096, 4096),
            "indices": [
                {
                    "shape": (256,),
                    "dtype": "long",
                    "low": 0,
                    "high": 4096,
                    "distinct": True,
                }
            ],
            "values": (256, 4096),
            "accumulate": True,
            "unsafe": False,
        },
        source="builtin",
    )

    # Log softmax backward
    r(
        "aten::_log_softmax_backward_data",
        {"grad_output": (4096, 4096), "output": (4096, 4096), "dim": 1, "input_dtype": 6},
        source="builtin",
    )

    # Bitwise ops (need int tensors)
    for op in ["aten::bitwise_and.Tensor", "aten::bitwise_or.Tensor", "aten::bitwise_xor.Tensor"]:
        r(
            op,
            {
                "self": {"shape": (256, 1024), "dtype": "int"},
                "other": {"shape": (256, 1024), "dtype": "int"},
            },
            source="builtin",
        )
    for op in ["aten::bitwise_and.Scalar", "aten::bitwise_or.Scalar", "aten::bitwise_xor.Scalar"]:
        r(op, {"self": {"shape": (256, 1024), "dtype": "int"}, "other": 7}, source="builtin")
    r("aten::bitwise_not", {"self": {"shape": (256, 1024), "dtype": "int"}}, source="builtin")
    r(
        "aten::__rshift__.Scalar",
        {"self": {"shape": (256, 1024), "dtype": "int"}, "other": 2},
        source="builtin",
    )
    r(
        "aten::__rshift__.Tensor",
        {"self": {"shape": (256, 1024), "dtype": "int"}, "other": {"shape": (1,), "dtype": "int"}},
        source="builtin",
    )

    # Int-only ops
    r(
        "aten::gcd",
        {
            "self": {"shape": (256, 1024), "dtype": "int"},
            "other": {"shape": (256, 1024), "dtype": "int"},
        },
        source="builtin",
    )
    r(
        "aten::lcm",
        {
            # Bound the range so the LCM (a*b/gcd) stays within int32; full-range
            # random ints overflow int32.
            "self": {"shape": (256, 1024), "dtype": "int", "low": 1, "high": 1024},
            "other": {"shape": (256, 1024), "dtype": "int", "low": 1, "high": 1024},
        },
        source="builtin",
    )
    r(
        "aten::bincount",
        {"self": {"shape": (8192,), "dtype": "long", "low": 0, "high": 1024}},
        source="builtin",
    )

    # Factory-like ops
    r("aten::tril_indices", {"row": 256, "col": 256}, source="builtin")
    r("aten::triu_indices", {"row": 256, "col": 256}, source="builtin")
    r("aten::rand", {"size": [64, 4096]}, source="builtin")

    # Nonzero
    r("aten::nonzero", {"self": {"shape": (4096,), "dtype": "float"}}, source="builtin")
    r(
        "aten::nonzero_static",
        {"self": {"shape": (4096,), "dtype": "float"}, "size": 4096},
        source="builtin",
    )

    r(
        "aten::masked_fill_.Tensor",
        {
            "self": (256, 4096),
            "mask": {"shape": (256, 4096), "dtype": "bool"},
            "value": {"shape": (), "dtype": "float"},
        },
        source="builtin",
    )

    # masked_scatter_ and its backward have no OpInfo. Give the in-place variant a
    # source that is large enough for any mask, and give the backward a source
    # buffer size (sizes) that holds every True position.
    r(
        "aten::masked_scatter_",
        {
            "self": (128, 256),
            "mask": {"shape": (128, 256), "dtype": "bool"},
            "source": (32768,),
        },
        source="builtin",
    )
    r(
        "aten::masked_scatter_backward",
        {
            "grad_output": (128, 256),
            "mask": {"shape": (128, 256), "dtype": "bool"},
            "sizes": [128, 256],
        },
        source="builtin",
    )

    r(
        "aten::div_.Scalar_mode",
        {"self": (256, 4096), "other": 2.0, "rounding_mode": "floor"},
        source="builtin",
    )
    r(
        "aten::div_.Tensor_mode",
        {"self": (256, 4096), "other": (256, 4096), "rounding_mode": "floor"},
        source="builtin",
    )

    r(
        "aten::reflection_pad1d.out",
        {"self": (1, 64, 4096), "padding": [1, 1], "out": (1, 64, 4098)},
        source="builtin",
    )
    r(
        "aten::reflection_pad2d.out",
        {"self": (1, 64, 32, 32), "padding": [1, 1, 1, 1], "out": (1, 64, 34, 34)},
        source="builtin",
    )
    r(
        "aten::reflection_pad3d.out",
        {"self": (1, 32, 16, 16, 16), "padding": [1, 1, 1, 1, 1, 1], "out": (1, 32, 18, 18, 18)},
        source="builtin",
    )
    r(
        "aten::replication_pad1d.out",
        {"self": (1, 64, 4096), "padding": [1, 1], "out": (1, 64, 4098)},
        source="builtin",
    )
    r(
        "aten::replication_pad3d.out",
        {"self": (1, 32, 16, 16, 16), "padding": [1, 1, 1, 1, 1, 1], "out": (1, 32, 18, 18, 18)},
        source="builtin",
    )

    # SVD
    r(
        "aten::linalg_svd.U",
        {"A": (256, 256), "full_matrices": True, "U": (0,), "S": (0,), "Vh": (0,)},
        source="builtin",
    )

    # ── Auto-generated shapes ─────────────────────────────────────────────
    r("aten::_aminmax", {"self": (256, 4096)}, source="builtin")
    r("aten::_aminmax.dim", {"self": (256, 4096), "dim": 0, "keepdim": False}, source="builtin")
    r(
        "aten::_aminmax.dim_out",
        {"self": (256, 4096), "dim": 0, "keepdim": False, "out0": (256, 4096), "out1": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::_aminmax.out",
        {"self": (256, 4096), "out0": (256, 4096), "out1": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::_fft_r2c",
        {"self": (256, 4096), "dim": [0], "normalization": 1, "onesided": False},
        source="builtin",
    )
    r("aten::_foreach_abs.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_abs.out", {"self": (256, 4096), "out": (256, 4096)}, source="builtin")
    r("aten::_foreach_abs_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_acos.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_acos.out", {"self": (256, 4096), "out": (256, 4096)}, source="builtin")
    r("aten::_foreach_acos_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_add.List", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::_foreach_add.List_out",
        {"self": (256, 4096), "other": (256, 4096), "out": (256, 4096)},
        source="builtin",
    )
    r("aten::_foreach_add.Scalar", {"self": (256, 4096), "scalar": 1.0}, source="builtin")
    r(
        "aten::_foreach_add.ScalarList",
        {"self": (256, 4096), "scalars": [1.0] * _N},
        source="builtin",
    )
    r(
        "aten::_foreach_add.ScalarList_out",
        {"self": (256, 4096), "scalars": [1.0] * _N, "out": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::_foreach_add.Scalar_out",
        {"self": (256, 4096), "scalar": 1.0, "out": (256, 4096)},
        source="builtin",
    )
    r("aten::_foreach_add.Tensor", {"self": (256, 4096), "other": ()}, source="builtin")
    r(
        "aten::_foreach_add.Tensor_out",
        {"self": (256, 4096), "other": (), "out": (256, 4096)},
        source="builtin",
    )
    r("aten::_foreach_add_.List", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r("aten::_foreach_add_.Scalar", {"self": (256, 4096), "scalar": 1.0}, source="builtin")
    r(
        "aten::_foreach_add_.ScalarList",
        {"self": (256, 4096), "scalars": [1.0] * _N},
        source="builtin",
    )
    r("aten::_foreach_add_.Tensor", {"self": (256, 4096), "other": ()}, source="builtin")
    r(
        "aten::_foreach_addcdiv.Scalar",
        {"self": (256, 4096), "tensor1": (256, 4096), "tensor2": (256, 4096), "value": 1.0},
        source="builtin",
    )
    r(
        "aten::_foreach_addcdiv.ScalarList",
        {
            "self": (256, 4096),
            "tensor1": (256, 4096),
            "tensor2": (256, 4096),
            "scalars": [1.0] * _N,
        },
        source="builtin",
    )
    r(
        "aten::_foreach_addcdiv.ScalarList_out",
        {
            "self": (256, 4096),
            "tensor1": (256, 4096),
            "tensor2": (256, 4096),
            "scalars": [1.0] * _N,
            "out": (256, 4096),
        },
        source="builtin",
    )
    r(
        "aten::_foreach_addcdiv.Scalar_out",
        {
            "self": (256, 4096),
            "tensor1": (256, 4096),
            "tensor2": (256, 4096),
            "value": 1.0,
            "out": (256, 4096),
        },
        source="builtin",
    )
    r(
        "aten::_foreach_addcdiv.Tensor_out",
        {
            "self": (256, 4096),
            "tensor1": (256, 4096),
            "tensor2": (256, 4096),
            "scalars": {"shape": (_N,), "dtype": "float"},
            "out": (256, 4096),
        },
        source="builtin",
    )
    r(
        "aten::_foreach_addcdiv_.Scalar",
        {"self": (256, 4096), "tensor1": (256, 4096), "tensor2": (256, 4096), "value": 1.0},
        source="builtin",
    )
    r(
        "aten::_foreach_addcdiv_.ScalarList",
        {
            "self": (256, 4096),
            "tensor1": (256, 4096),
            "tensor2": (256, 4096),
            "scalars": [1.0] * _N,
        },
        source="builtin",
    )
    r(
        "aten::_foreach_addcmul.Scalar",
        {"self": (256, 4096), "tensor1": (256, 4096), "tensor2": (256, 4096), "value": 1.0},
        source="builtin",
    )
    r(
        "aten::_foreach_addcmul.ScalarList",
        {
            "self": (256, 4096),
            "tensor1": (256, 4096),
            "tensor2": (256, 4096),
            "scalars": [1.0] * _N,
        },
        source="builtin",
    )
    r(
        "aten::_foreach_addcmul.ScalarList_out",
        {
            "self": (256, 4096),
            "tensor1": (256, 4096),
            "tensor2": (256, 4096),
            "scalars": [1.0] * _N,
            "out": (256, 4096),
        },
        source="builtin",
    )
    r(
        "aten::_foreach_addcmul.Scalar_out",
        {
            "self": (256, 4096),
            "tensor1": (256, 4096),
            "tensor2": (256, 4096),
            "value": 1.0,
            "out": (256, 4096),
        },
        source="builtin",
    )
    r(
        "aten::_foreach_addcmul.Tensor_out",
        {
            "self": (256, 4096),
            "tensor1": (256, 4096),
            "tensor2": (256, 4096),
            "scalars": {"shape": (_N,), "dtype": "float"},
            "out": (256, 4096),
        },
        source="builtin",
    )
    r(
        "aten::_foreach_addcmul_.Scalar",
        {"self": (256, 4096), "tensor1": (256, 4096), "tensor2": (256, 4096), "value": 1.0},
        source="builtin",
    )
    r(
        "aten::_foreach_addcmul_.ScalarList",
        {
            "self": (256, 4096),
            "tensor1": (256, 4096),
            "tensor2": (256, 4096),
            "scalars": [1.0] * _N,
        },
        source="builtin",
    )
    r("aten::_foreach_asin.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_asin.out", {"self": (256, 4096), "out": (256, 4096)}, source="builtin")
    r("aten::_foreach_asin_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_atan.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_atan.out", {"self": (256, 4096), "out": (256, 4096)}, source="builtin")
    r("aten::_foreach_atan_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_ceil.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_ceil.out", {"self": (256, 4096), "out": (256, 4096)}, source="builtin")
    r("aten::_foreach_ceil_.default", {"self": (256, 4096)}, source="builtin")
    r(
        "aten::_foreach_clamp_max.List",
        {"self": (256, 4096), "other": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::_foreach_clamp_max.List_out",
        {"self": (256, 4096), "other": (256, 4096), "out": (256, 4096)},
        source="builtin",
    )
    r("aten::_foreach_clamp_max.Scalar", {"self": (256, 4096), "scalar": 1.0}, source="builtin")
    r(
        "aten::_foreach_clamp_max.ScalarList_out",
        {"self": (256, 4096), "scalars": [1.0] * _N, "out": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::_foreach_clamp_max.Scalar_out",
        {"self": (256, 4096), "scalar": 1.0, "out": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::_foreach_clamp_max_.List",
        {"self": (256, 4096), "other": (256, 4096)},
        source="builtin",
    )
    r("aten::_foreach_clamp_max_.Scalar", {"self": (256, 4096), "scalar": 1.0}, source="builtin")
    r(
        "aten::_foreach_clamp_min.List",
        {"self": (256, 4096), "other": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::_foreach_clamp_min.List_out",
        {"self": (256, 4096), "other": (256, 4096), "out": (256, 4096)},
        source="builtin",
    )
    r("aten::_foreach_clamp_min.Scalar", {"self": (256, 4096), "scalar": 1.0}, source="builtin")
    r(
        "aten::_foreach_clamp_min.ScalarList_out",
        {"self": (256, 4096), "scalars": [1.0] * _N, "out": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::_foreach_clamp_min.Scalar_out",
        {"self": (256, 4096), "scalar": 1.0, "out": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::_foreach_clamp_min_.List",
        {"self": (256, 4096), "other": (256, 4096)},
        source="builtin",
    )
    r("aten::_foreach_clamp_min_.Scalar", {"self": (256, 4096), "scalar": 1.0}, source="builtin")
    r("aten::_foreach_cos.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_cos.out", {"self": (256, 4096), "out": (256, 4096)}, source="builtin")
    r("aten::_foreach_cos_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_cosh.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_cosh.out", {"self": (256, 4096), "out": (256, 4096)}, source="builtin")
    r("aten::_foreach_cosh_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_div.List", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::_foreach_div.List_out",
        {"self": (256, 4096), "other": (256, 4096), "out": (256, 4096)},
        source="builtin",
    )
    r("aten::_foreach_div.Scalar", {"self": (256, 4096), "scalar": 1.0}, source="builtin")
    r(
        "aten::_foreach_div.ScalarList",
        {"self": (256, 4096), "scalars": [1.0] * _N},
        source="builtin",
    )
    r(
        "aten::_foreach_div.ScalarList_out",
        {"self": (256, 4096), "scalars": [1.0] * _N, "out": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::_foreach_div.Scalar_out",
        {"self": (256, 4096), "scalar": 1.0, "out": (256, 4096)},
        source="builtin",
    )
    r("aten::_foreach_div.Tensor", {"self": (256, 4096), "other": ()}, source="builtin")
    r(
        "aten::_foreach_div.Tensor_out",
        {"self": (256, 4096), "other": (), "out": (256, 4096)},
        source="builtin",
    )
    r("aten::_foreach_div_.List", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r("aten::_foreach_div_.Scalar", {"self": (256, 4096), "scalar": 1.0}, source="builtin")
    r(
        "aten::_foreach_div_.ScalarList",
        {"self": (256, 4096), "scalars": [1.0] * _N},
        source="builtin",
    )
    r("aten::_foreach_div_.Tensor", {"self": (256, 4096), "other": ()}, source="builtin")
    r("aten::_foreach_erf.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_erf.out", {"self": (256, 4096), "out": (256, 4096)}, source="builtin")
    r("aten::_foreach_erf_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_erfc.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_erfc.out", {"self": (256, 4096), "out": (256, 4096)}, source="builtin")
    r("aten::_foreach_erfc_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_exp.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_exp.out", {"self": (256, 4096), "out": (256, 4096)}, source="builtin")
    r("aten::_foreach_exp_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_expm1.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_expm1.out", {"self": (256, 4096), "out": (256, 4096)}, source="builtin")
    r("aten::_foreach_expm1_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_floor.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_floor.out", {"self": (256, 4096), "out": (256, 4096)}, source="builtin")
    r("aten::_foreach_floor_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_frac.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_frac.out", {"self": (256, 4096), "out": (256, 4096)}, source="builtin")
    r("aten::_foreach_frac_.default", {"self": (256, 4096)}, source="builtin")
    r(
        "aten::_foreach_lerp.List",
        {"self": (256, 4096), "tensors1": (256, 4096), "weights": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::_foreach_lerp.List_out",
        {"self": (256, 4096), "tensors1": (256, 4096), "weights": (256, 4096), "out": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::_foreach_lerp.Scalar",
        {"self": (256, 4096), "tensors1": (256, 4096), "weight": 1.0},
        source="builtin",
    )
    r(
        "aten::_foreach_lerp.ScalarList",
        {"self": (256, 4096), "tensors1": (256, 4096), "weight": [1.0] * _N},
        source="builtin",
    )
    r(
        "aten::_foreach_lerp.ScalarList_out",
        {"self": (256, 4096), "tensors1": (256, 4096), "weight": [1.0] * _N, "out": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::_foreach_lerp.Scalar_out",
        {"self": (256, 4096), "tensors1": (256, 4096), "weight": 1.0, "out": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::_foreach_lerp_.List",
        {"self": (256, 4096), "tensors1": (256, 4096), "weights": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::_foreach_lerp_.Scalar",
        {"self": (256, 4096), "tensors1": (256, 4096), "weight": 1.0},
        source="builtin",
    )
    r(
        "aten::_foreach_lerp_.ScalarList",
        {"self": (256, 4096), "tensors1": (256, 4096), "weight": [1.0] * _N},
        source="builtin",
    )
    r("aten::_foreach_lgamma.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_lgamma.out", {"self": (256, 4096), "out": (256, 4096)}, source="builtin")
    r("aten::_foreach_lgamma_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_log.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_log.out", {"self": (256, 4096), "out": (256, 4096)}, source="builtin")
    r("aten::_foreach_log10.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_log10.out", {"self": (256, 4096), "out": (256, 4096)}, source="builtin")
    r("aten::_foreach_log10_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_log1p.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_log1p.out", {"self": (256, 4096), "out": (256, 4096)}, source="builtin")
    r("aten::_foreach_log1p_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_log2.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_log2.out", {"self": (256, 4096), "out": (256, 4096)}, source="builtin")
    r("aten::_foreach_log2_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_log_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_maximum.List", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::_foreach_maximum.List_out",
        {"self": (256, 4096), "other": (256, 4096), "out": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::_foreach_maximum.ScalarList_out",
        {"self": (256, 4096), "scalars": [1.0] * _N, "out": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::_foreach_maximum.Scalar_out",
        {"self": (256, 4096), "scalar": 1.0, "out": (256, 4096)},
        source="builtin",
    )
    r("aten::_foreach_maximum_.List", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r("aten::_foreach_minimum.List", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::_foreach_minimum.List_out",
        {"self": (256, 4096), "other": (256, 4096), "out": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::_foreach_minimum.ScalarList_out",
        {"self": (256, 4096), "scalars": [1.0] * _N, "out": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::_foreach_minimum.Scalar_out",
        {"self": (256, 4096), "scalar": 1.0, "out": (256, 4096)},
        source="builtin",
    )
    r("aten::_foreach_minimum_.List", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r("aten::_foreach_mul.List", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::_foreach_mul.List_out",
        {"self": (256, 4096), "other": (256, 4096), "out": (256, 4096)},
        source="builtin",
    )
    r("aten::_foreach_mul.Scalar", {"self": (256, 4096), "scalar": 1.0}, source="builtin")
    r(
        "aten::_foreach_mul.ScalarList",
        {"self": (256, 4096), "scalars": [1.0] * _N},
        source="builtin",
    )
    r(
        "aten::_foreach_mul.ScalarList_out",
        {"self": (256, 4096), "scalars": [1.0] * _N, "out": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::_foreach_mul.Scalar_out",
        {"self": (256, 4096), "scalar": 1.0, "out": (256, 4096)},
        source="builtin",
    )
    r("aten::_foreach_mul.Tensor", {"self": (256, 4096), "other": ()}, source="builtin")
    r(
        "aten::_foreach_mul.Tensor_out",
        {"self": (256, 4096), "other": (), "out": (256, 4096)},
        source="builtin",
    )
    r("aten::_foreach_mul_.List", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r("aten::_foreach_mul_.Scalar", {"self": (256, 4096), "scalar": 1.0}, source="builtin")
    r(
        "aten::_foreach_mul_.ScalarList",
        {"self": (256, 4096), "scalars": [1.0] * _N},
        source="builtin",
    )
    r("aten::_foreach_mul_.Tensor", {"self": (4096, 4096), "other": ()}, source="builtin")
    r("aten::_foreach_neg.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_neg.out", {"self": (256, 4096), "out": (256, 4096)}, source="builtin")
    r("aten::_foreach_neg_.default", {"self": (256, 4096)}, source="builtin")
    # foreach_copy_: both self (write-alias list) and src list must be populated.
    r(
        "aten::_foreach_copy_.default",
        {"self": (256, 4096), "src": (256, 4096), "non_blocking": False},
        source="builtin",
    )
    # foreach_norm with GPT-OSS shapes
    r(
        "aten::_foreach_norm.Scalar",
        {
            "self": [
                {"shape": (4096, 2880)},
                {"shape": (4096,)},
                {"shape": (512, 2880)},
                {"shape": (512,)},
                {"shape": (512, 2880)},
                {"shape": (512,)},
                {"shape": (2880, 4096)},
                {"shape": (2880,)},
                {"shape": (2880,)},
                {"shape": (2880,)},
                {"shape": (32, 5760, 2880)},
                {"shape": (32, 5760)},
                {"shape": (32, 2880, 2880)},
                {"shape": (32, 2880)},
                {"shape": (32, 2880)},
            ],
            "ord": 2.0,
        },
        source="builtin",
    )
    r("aten::_foreach_pow.List", {"self": (256, 4096), "exponent": (256, 4096)}, source="builtin")
    r(
        "aten::_foreach_pow.List_out",
        {"self": (256, 4096), "exponent": (256, 4096), "out": (256, 4096)},
        source="builtin",
    )
    r("aten::_foreach_pow.Scalar", {"self": (256, 4096), "exponent": 1.0}, source="builtin")
    r(
        "aten::_foreach_pow.ScalarList_out",
        {"self": (256, 4096), "exponent": [1.0] * _N, "out": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::_foreach_pow.Scalar_out",
        {"self": (256, 4096), "exponent": 1.0, "out": (256, 4096)},
        source="builtin",
    )
    r("aten::_foreach_pow_.List", {"self": (256, 4096), "exponent": (256, 4096)}, source="builtin")
    r("aten::_foreach_pow_.Scalar", {"self": (256, 4096), "exponent": 1.0}, source="builtin")
    r("aten::_foreach_reciprocal.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_reciprocal.out", {"self": (256, 4096), "out": (256, 4096)}, source="builtin")
    r("aten::_foreach_reciprocal_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_round.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_round.out", {"self": (256, 4096), "out": (256, 4096)}, source="builtin")
    r("aten::_foreach_round_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_rsqrt.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_rsqrt.out", {"self": (256, 4096), "out": (256, 4096)}, source="builtin")
    r("aten::_foreach_rsqrt_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_sigmoid.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_sigmoid.out", {"self": (256, 4096), "out": (256, 4096)}, source="builtin")
    r("aten::_foreach_sigmoid_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_sign.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_sign.out", {"self": (256, 4096), "out": (256, 4096)}, source="builtin")
    r("aten::_foreach_sign_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_sin.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_sin.out", {"self": (256, 4096), "out": (256, 4096)}, source="builtin")
    r("aten::_foreach_sin_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_sinh.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_sinh.out", {"self": (256, 4096), "out": (256, 4096)}, source="builtin")
    r("aten::_foreach_sinh_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_sqrt.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_sqrt.out", {"self": (256, 4096), "out": (256, 4096)}, source="builtin")
    r("aten::_foreach_sqrt_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_sub.List", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::_foreach_sub.List_out",
        {"self": (256, 4096), "other": (256, 4096), "out": (256, 4096)},
        source="builtin",
    )
    r("aten::_foreach_sub.Scalar", {"self": (256, 4096), "scalar": 1.0}, source="builtin")
    r(
        "aten::_foreach_sub.ScalarList",
        {"self": (256, 4096), "scalars": [1.0] * _N},
        source="builtin",
    )
    r(
        "aten::_foreach_sub.ScalarList_out",
        {"self": (256, 4096), "scalars": [1.0] * _N, "out": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::_foreach_sub.Scalar_out",
        {"self": (256, 4096), "scalar": 1.0, "out": (256, 4096)},
        source="builtin",
    )
    r("aten::_foreach_sub_.List", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r("aten::_foreach_sub_.Scalar", {"self": (256, 4096), "scalar": 1.0}, source="builtin")
    r(
        "aten::_foreach_sub_.ScalarList",
        {"self": (256, 4096), "scalars": [1.0] * _N},
        source="builtin",
    )
    r("aten::_foreach_tan.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_tan.out", {"self": (256, 4096), "out": (256, 4096)}, source="builtin")
    r("aten::_foreach_tan_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_tanh.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_tanh.out", {"self": (256, 4096), "out": (256, 4096)}, source="builtin")
    r("aten::_foreach_tanh_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_trunc.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_trunc.out", {"self": (256, 4096), "out": (256, 4096)}, source="builtin")
    r("aten::_foreach_trunc_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::_foreach_zero_.default", {"self": (256, 4096)}, source="builtin")
    r(
        "aten::_log_softmax",
        {"self": (256, 4096), "dim": 0, "half_to_float": False},
        source="builtin",
    )
    r(
        "aten::_log_softmax.out",
        {"self": (256, 4096), "dim": 0, "half_to_float": False},
        source="builtin",
    )
    r("aten::_logcumsumexp", {"self": (256, 4096), "dim": 0}, source="builtin")
    r("aten::_logcumsumexp.out", {"self": (256, 4096), "dim": 0}, source="builtin")
    r("aten::_prelu_kernel", {"self": (256, 4096), "weight": (1,)}, source="builtin")
    r(
        "aten::_prelu_kernel_backward",
        {"grad_output": (256, 4096), "self": (256, 4096), "weight": (1,)},
        source="builtin",
    )
    r("aten::_softmax", {"self": (256, 4096), "dim": 0, "half_to_float": False}, source="builtin")
    r(
        "aten::_softmax.out",
        {"self": (256, 4096), "dim": 0, "half_to_float": False},
        source="builtin",
    )
    r("aten::_to_copy", {"self": (256, 4096)}, source="builtin")
    r("aten::_to_copy.out", {"self": (256, 4096)}, source="builtin")
    r("aten::abs.out", {"self": (256, 4096)}, source="builtin")
    r("aten::acos", {"self": (256, 4096)}, source="builtin")
    r("aten::acos.out", {"self": (256, 4096)}, source="builtin")
    r("aten::acosh", {"self": (256, 4096)}, source="builtin")
    r("aten::acosh.out", {"self": (256, 4096)}, source="builtin")
    r("aten::add.Scalar", {"self": (256, 4096), "other": 1.0, "alpha": 1.0}, source="builtin")
    r(
        "aten::add.Scalar_out",
        {"self": (256, 4096), "other": 1.0, "alpha": 1.0},
        source="builtin",
    )
    r("aten::add.out", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r("aten::add_.Scalar", {"self": (256, 4096), "other": 1.0, "alpha": 1.0}, source="builtin")
    r("aten::add_.Tensor", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::addcdiv_.default",
        {"self": (256, 4096), "tensor1": (256, 4096), "tensor2": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::addcmul_.default",
        {"self": (256, 4096), "tensor1": (256, 4096), "tensor2": (256, 4096)},
        source="builtin",
    )
    r("aten::alias", {"self": (256, 4096)}, source="builtin")
    r("aten::asin", {"self": (256, 4096)}, source="builtin")
    r("aten::asin.out", {"self": (256, 4096)}, source="builtin")
    r("aten::asinh", {"self": (256, 4096)}, source="builtin")
    r("aten::asinh.out", {"self": (256, 4096)}, source="builtin")
    r("aten::atan", {"self": (256, 4096)}, source="builtin")
    r("aten::atan.out", {"self": (256, 4096)}, source="builtin")
    r("aten::atan2", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r("aten::atan2.out", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r("aten::atanh", {"self": (256, 4096)}, source="builtin")
    r("aten::atanh.out", {"self": (256, 4096)}, source="builtin")
    r(
        "aten::avg_pool1d.out",
        {
            "self": (1, 64, 4096),
            "kernel_size": [2],
            "stride": [1],
            "padding": [0],
            "ceil_mode": False,
            "count_include_pad": False,
            "out": (0,),
        },
        source="builtin",
    )
    r(
        "aten::binary_cross_entropy_backward",
        {"grad_output": (256, 4096), "self": (256, 4096), "target": (256, 4096), "reduction": 1},
        source="builtin",
    )
    r(
        "aten::binary_cross_entropy_backward.grad_input",
        {
            "grad_output": (256, 4096),
            "self": (256, 4096),
            "target": (256, 4096),
            "reduction": 1,
            "grad_input": (256, 4096),
        },
        source="builtin",
    )
    r(
        "aten::binary_cross_entropy_with_logits.out",
        {"self": (256, 4096), "target": (256, 4096), "reduction": 1},
        source="builtin",
    )
    r(
        "aten::bitwise_and.Tensor_out",
        {
            "self": {"shape": (256, 1024), "dtype": "int"},
            "other": {"shape": (256, 1024), "dtype": "int"},
            "out": (0,),
        },
        source="builtin",
    )
    r(
        "aten::bitwise_left_shift.Tensor",
        {
            "self": {"shape": (256, 1024), "dtype": "int"},
            "other": {"shape": (256, 1024), "dtype": "int"},
        },
        source="builtin",
    )
    r(
        "aten::bitwise_left_shift.Tensor_Scalar",
        {"self": {"shape": (256, 1024), "dtype": "int"}, "other": 1},
        source="builtin",
    )
    r(
        "aten::bitwise_left_shift.Tensor_Scalar_out",
        {"self": {"shape": (256, 1024), "dtype": "int"}, "other": 1},
        source="builtin",
    )
    r(
        "aten::bitwise_left_shift.Tensor_out",
        {
            "self": {"shape": (256, 1024), "dtype": "int"},
            "other": {"shape": (256, 1024), "dtype": "int"},
            "out": (0,),
        },
        source="builtin",
    )
    r(
        "aten::bitwise_not.out",
        {"self": {"shape": (256, 1024), "dtype": "int"}},
        source="builtin",
    )
    r(
        "aten::bitwise_or.Tensor_out",
        {
            "self": {"shape": (256, 1024), "dtype": "int"},
            "other": {"shape": (256, 1024), "dtype": "int"},
            "out": (0,),
        },
        source="builtin",
    )
    r(
        "aten::bitwise_right_shift.Tensor",
        {
            "self": {"shape": (256, 1024), "dtype": "int"},
            "other": {"shape": (256, 1024), "dtype": "int"},
        },
        source="builtin",
    )
    r(
        "aten::bitwise_right_shift.Tensor_Scalar",
        {"self": {"shape": (256, 1024), "dtype": "int"}, "other": 1},
        source="builtin",
    )
    r(
        "aten::bitwise_right_shift.Tensor_Scalar_out",
        {"self": {"shape": (256, 1024), "dtype": "int"}, "other": 1},
        source="builtin",
    )
    r(
        "aten::bitwise_right_shift.Tensor_out",
        {
            "self": {"shape": (256, 1024), "dtype": "int"},
            "other": {"shape": (256, 1024), "dtype": "int"},
            "out": (0,),
        },
        source="builtin",
    )
    r(
        "aten::bitwise_xor.Tensor_out",
        {
            "self": {"shape": (256, 1024), "dtype": "int"},
            "other": {"shape": (256, 1024), "dtype": "int"},
            "out": (0,),
        },
        source="builtin",
    )
    r("aten::cauchy_", {"self": (256, 4096), "median": 1.0, "sigma": 1.0}, source="builtin")
    r("aten::ceil.out", {"self": (256, 4096)}, source="builtin")
    r("aten::celu", {"self": (256, 4096), "alpha": 1.0}, source="builtin")
    r("aten::celu.out", {"self": (256, 4096), "alpha": 1.0}, source="builtin")
    r("aten::clamp_max", {"self": (256, 4096), "max": 1.0}, source="builtin")
    r("aten::clamp_max.Tensor", {"self": (256, 4096), "max": (256, 4096)}, source="builtin")
    r(
        "aten::clamp_max.Tensor_out",
        {"self": (256, 4096), "max": (256, 4096)},
        source="builtin",
    )
    r("aten::clamp_max.out", {"self": (256, 4096), "max": 1.0}, source="builtin")
    r("aten::clamp_min", {"self": (256, 4096), "min": 1.0}, source="builtin")
    r("aten::clamp_min.Tensor", {"self": (256, 4096), "min": (256, 4096)}, source="builtin")
    r(
        "aten::clamp_min.Tensor_out",
        {"self": (256, 4096), "min": (256, 4096)},
        source="builtin",
    )
    r("aten::clamp_min.out", {"self": (256, 4096), "min": 1.0}, source="builtin")
    r("aten::conj_physical", {"self": (256, 4096)}, source="builtin")
    r("aten::conj_physical.out", {"self": (256, 4096)}, source="builtin")
    r("aten::copysign.Scalar", {"self": (256, 4096), "other": 1.0}, source="builtin")
    r(
        "aten::copysign.Scalar_out",
        {"self": (256, 4096), "other": 1.0},
        source="builtin",
    )
    r("aten::copysign.Tensor", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::copysign.out",
        {"self": (256, 4096), "other": (256, 4096)},
        source="builtin",
    )
    r("aten::copysign_.Scalar", {"self": (256, 4096), "other": 1.0}, source="builtin")
    r("aten::copysign_.Tensor", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r("aten::cos.out", {"self": (256, 4096)}, source="builtin")
    r("aten::cosh", {"self": (256, 4096)}, source="builtin")
    r("aten::cosh.out", {"self": (256, 4096)}, source="builtin")
    r(
        "aten::cross_entropy_loss",
        {
            "self": (256, 4096),
            "target": {"shape": (256,), "dtype": "long", "low": 0, "high": 4096},
            "reduction": 1,
            "ignore_index": 1,
            "label_smoothing": 1.0,
        },
        source="builtin",
    )
    r("aten::deg2rad", {"self": (256, 4096)}, source="builtin")
    r("aten::deg2rad.out", {"self": (256, 4096)}, source="builtin")
    r("aten::deg2rad_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::digamma", {"self": (256, 4096)}, source="builtin")
    r("aten::digamma.out", {"self": (256, 4096)}, source="builtin")
    r("aten::digamma_.default", {"self": (256, 4096)}, source="builtin")
    r(
        "aten::div.Scalar_mode_out",
        {"self": (256, 4096), "other": 1.0, "rounding_mode": None},
        source="builtin",
    )
    r("aten::div.Scalar_out", {"self": (256, 4096), "other": 1.0}, source="builtin")
    r(
        "aten::div.Tensor_mode",
        {"self": (256, 4096), "other": (256, 4096), "rounding_mode": None},
        source="builtin",
    )
    r("aten::div.out", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::div.out_mode",
        {"self": (256, 4096), "other": (256, 4096), "rounding_mode": None},
        source="builtin",
    )
    r("aten::div_.Scalar", {"self": (256, 4096), "other": 1.0}, source="builtin")
    r("aten::div_.Tensor", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::elu",
        {"self": (256, 4096), "alpha": 1.0, "scale": 1.0, "input_scale": 1.0},
        source="builtin",
    )
    r(
        "aten::elu.out",
        {"self": (256, 4096), "alpha": 1.0, "scale": 1.0, "input_scale": 1.0},
        source="builtin",
    )
    r(
        "aten::elu_backward",
        {
            "grad_output": (256, 4096),
            "alpha": 1.0,
            "scale": 1.0,
            "input_scale": 1.0,
            "is_result": False,
            "self_or_result": (256, 4096),
        },
        source="builtin",
    )
    r(
        "aten::elu_backward.grad_input",
        {
            "grad_output": (256, 4096),
            "alpha": 1.0,
            "scale": 1.0,
            "input_scale": 1.0,
            "is_result": False,
            "self_or_result": (256, 4096),
            "grad_input": (256, 4096),
        },
        source="builtin",
    )
    r("aten::eq.Scalar", {"self": (256, 4096), "other": 1.0}, source="builtin")
    r("aten::eq.Scalar_out", {"self": (256, 4096), "other": 1.0}, source="builtin")
    r("aten::eq.Tensor", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::eq.Tensor_out",
        {"self": (256, 4096), "other": (256, 4096)},
        source="builtin",
    )
    r("aten::erf.out", {"self": (256, 4096)}, source="builtin")
    r("aten::erfc", {"self": (256, 4096)}, source="builtin")
    r("aten::erfc.out", {"self": (256, 4096)}, source="builtin")
    r("aten::erfc_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::erfinv", {"self": (256, 4096)}, source="builtin")
    r("aten::erfinv.out", {"self": (256, 4096)}, source="builtin")
    r("aten::erfinv_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::exp.out", {"self": (256, 4096)}, source="builtin")
    r("aten::exp2", {"self": (256, 4096)}, source="builtin")
    r("aten::exp2.out", {"self": (256, 4096)}, source="builtin")
    r("aten::exp2_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::expm1", {"self": (256, 4096)}, source="builtin")
    r("aten::expm1.out", {"self": (256, 4096)}, source="builtin")
    r("aten::exponential_", {"self": (256, 4096), "lambd": 1.0}, source="builtin")
    r("aten::fill.Scalar", {"self": (256, 4096), "value": 1.0}, source="builtin")
    r("aten::fill.Scalar_out", {"self": (256, 4096), "value": 1.0}, source="builtin")
    r("aten::fill.Tensor", {"self": (256, 4096), "value": ()}, source="builtin")
    r("aten::fill.Tensor_out", {"self": (256, 4096), "value": ()}, source="builtin")
    r("aten::fill_.Scalar", {"self": (256, 4096), "value": 1.0}, source="builtin")
    r("aten::fill_.Tensor", {"self": (256, 4096), "value": ()}, source="builtin")
    r("aten::fix", {"self": (256, 4096)}, source="builtin")
    r("aten::fix.out", {"self": (256, 4096)}, source="builtin")
    r("aten::fix_", {"self": (256, 4096)}, source="builtin")
    r("aten::float_power.Tensor_Scalar", {"self": (256, 4096), "exponent": 1.0}, source="builtin")
    r(
        "aten::float_power.Tensor_Scalar_out",
        {"self": (256, 4096), "exponent": 1.0, "out": {"shape": (256, 4096), "dtype": "double"}},
        source="builtin",
    )
    r(
        "aten::float_power.Tensor_Tensor",
        {"self": (256, 4096), "exponent": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::float_power.Tensor_Tensor_out",
        {
            "self": (256, 4096),
            "exponent": (256, 4096),
            "out": {"shape": (256, 4096), "dtype": "double"},
        },
        source="builtin",
    )
    r("aten::floor.out", {"self": (256, 4096)}, source="builtin")
    r("aten::floor_divide", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::floor_divide.Scalar_out",
        {"self": (256, 4096), "other": 1.0},
        source="builtin",
    )
    r(
        "aten::floor_divide.out",
        {"self": (256, 4096), "other": (256, 4096)},
        source="builtin",
    )
    r("aten::floor_divide_.Tensor", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r("aten::fmax", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r("aten::fmax.out", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r("aten::fmin", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r("aten::fmin.out", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r("aten::fmod.Scalar", {"self": (256, 4096), "other": 1.0}, source="builtin")
    r("aten::fmod.Scalar_out", {"self": (256, 4096), "other": 1.0}, source="builtin")
    r("aten::fmod.Tensor", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::fmod.Tensor_out",
        {"self": (256, 4096), "other": (256, 4096)},
        source="builtin",
    )
    r("aten::frac", {"self": (256, 4096)}, source="builtin")
    r("aten::frac.out", {"self": (256, 4096)}, source="builtin")
    r("aten::frac_.default", {"self": (256, 4096)}, source="builtin")
    r(
        "aten::gcd.out",
        {
            "self": {"shape": (256, 4096), "dtype": "int"},
            "other": {"shape": (256, 4096), "dtype": "int"},
            "out": {"shape": (256, 4096), "dtype": "int"},
        },
        source="builtin",
    )
    r(
        "aten::gcd_.default",
        {
            "self": {"shape": (256, 1024), "dtype": "int"},
            "other": {"shape": (256, 1024), "dtype": "int"},
        },
        source="builtin",
    )
    r("aten::ge.Scalar", {"self": (256, 4096), "other": 1.0}, source="builtin")
    r("aten::ge.Scalar_out", {"self": (256, 4096), "other": 1.0}, source="builtin")
    r("aten::ge.Tensor", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::ge.Tensor_out",
        {"self": (256, 4096), "other": (256, 4096)},
        source="builtin",
    )
    r("aten::gelu.out", {"self": (256, 4096)}, source="builtin")
    r("aten::gelu_", {"self": (256, 4096)}, source="builtin")
    r("aten::gelu_backward", {"grad_output": (256, 4096), "self": (256, 4096)}, source="builtin")
    r(
        "aten::gelu_backward.grad_input",
        {"grad_output": (256, 4096), "self": (256, 4096), "grad_input": (256, 4096)},
        source="builtin",
    )
    r("aten::geometric_", {"self": (256, 4096), "p": 0.5}, source="builtin")
    r("aten::glu", {"self": (256, 4096), "dim": 0}, source="builtin")
    r("aten::glu.out", {"self": (256, 4096), "dim": 0}, source="builtin")
    r(
        "aten::glu_backward",
        {"grad_output": (128, 4096), "self": (256, 4096), "dim": 0},
        source="builtin",
    )
    r(
        "aten::glu_backward.grad_input",
        {"grad_output": (128, 4096), "self": (256, 4096), "dim": 0, "grad_input": (256, 4096)},
        source="builtin",
    )
    r("aten::gt.Scalar", {"self": (256, 4096), "other": 1.0}, source="builtin")
    r("aten::gt.Scalar_out", {"self": (256, 4096), "other": 1.0}, source="builtin")
    r("aten::gt.Tensor", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::gt.Tensor_out",
        {"self": (256, 4096), "other": (256, 4096)},
        source="builtin",
    )
    r("aten::hardshrink", {"self": (256, 4096), "lambd": 1.0}, source="builtin")
    r("aten::hardshrink.out", {"self": (256, 4096), "lambd": 1.0}, source="builtin")
    r(
        "aten::hardshrink_backward",
        {"grad_out": (256, 4096), "self": (256, 4096), "lambd": 1.0},
        source="builtin",
    )
    r(
        "aten::hardshrink_backward.grad_input",
        {"grad_out": (256, 4096), "self": (256, 4096), "lambd": 1.0, "grad_input": (256, 4096)},
        source="builtin",
    )
    r("aten::hardsigmoid", {"self": (256, 4096)}, source="builtin")
    r("aten::hardsigmoid.out", {"self": (256, 4096)}, source="builtin")
    r("aten::hardsigmoid_.default", {"self": (256, 4096)}, source="builtin")
    r(
        "aten::hardsigmoid_backward",
        {"grad_output": (256, 4096), "self": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::hardsigmoid_backward.grad_input",
        {"grad_output": (256, 4096), "self": (256, 4096), "grad_input": (256, 4096)},
        source="builtin",
    )
    r("aten::hardswish", {"self": (256, 4096)}, source="builtin")
    r("aten::hardswish.out", {"self": (256, 4096)}, source="builtin")
    r("aten::hardswish_.default", {"self": (256, 4096)}, source="builtin")
    r(
        "aten::hardswish_backward",
        {"grad_output": (256, 4096), "self": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::hardswish_backward.out",
        {"grad_output": (256, 4096), "self": (256, 4096)},
        source="builtin",
    )
    r("aten::hardtanh", {"self": (256, 4096), "min_val": -1.0, "max_val": 1.0}, source="builtin")
    r(
        "aten::hardtanh.out",
        {"self": (256, 4096), "min_val": -1.0, "max_val": 1.0},
        source="builtin",
    )
    r(
        "aten::hardtanh_.default",
        {"self": (256, 4096), "min_val": -1.0, "max_val": 1.0},
        source="builtin",
    )
    r(
        "aten::hardtanh_backward",
        {"grad_output": (256, 4096), "self": (256, 4096), "min_val": -1.0, "max_val": 1.0},
        source="builtin",
    )
    r(
        "aten::hardtanh_backward.grad_input",
        {
            "grad_output": (256, 4096),
            "self": (256, 4096),
            "min_val": -1.0,
            "max_val": 1.0,
            "grad_input": (256, 4096),
        },
        source="builtin",
    )
    r("aten::heaviside", {"self": (256, 4096), "values": (256, 4096)}, source="builtin")
    r(
        "aten::heaviside.out",
        {"self": (256, 4096), "values": (256, 4096)},
        source="builtin",
    )
    r("aten::heaviside_", {"self": (256, 4096), "values": (256, 4096)}, source="builtin")
    r(
        "aten::hinge_embedding_loss",
        {"self": (256, 4096), "target": (256, 4096), "margin": 1.0, "reduction": 1},
        source="builtin",
    )
    r(
        "aten::huber_loss",
        {"self": (256, 4096), "target": (256, 4096), "reduction": 1, "delta": 1.0},
        source="builtin",
    )
    r(
        "aten::huber_loss.out",
        {"self": (256, 4096), "target": (256, 4096), "reduction": 1, "delta": 1.0},
        source="builtin",
    )
    r(
        "aten::huber_loss_backward.out",
        {
            "grad_output": (256, 4096),
            "self": (256, 4096),
            "target": (256, 4096),
            "reduction": 1,
            "delta": 1.0,
            "grad_input": (256, 4096),
        },
        source="builtin",
    )
    r("aten::hypot", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r("aten::hypot.out", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r("aten::hypot_", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r("aten::i0_", {"self": (256, 4096)}, source="builtin")
    r("aten::igamma", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::igamma.out",
        {"self": (256, 4096), "other": (256, 4096)},
        source="builtin",
    )
    r("aten::igamma_", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r("aten::igammac", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::igammac.out",
        {"self": (256, 4096), "other": (256, 4096)},
        source="builtin",
    )
    r("aten::igammac_", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::index_add_",
        {
            "self": (256, 4096),
            "dim": 0,
            "index": {"shape": (64,), "dtype": "long", "low": 0, "high": 256},
            "source": (64, 4096),
        },
        source="builtin",
    )
    r(
        "aten::index_copy_",
        {
            "self": (256, 4096),
            "dim": 0,
            "index": {"shape": (64,), "dtype": "long", "low": 0, "high": 256},
            "source": (64, 4096),
        },
        source="builtin",
    )
    r(
        "aten::index_fill_.int_Scalar",
        {
            "self": (256, 4096),
            "dim": 0,
            "index": {"shape": (64,), "dtype": "long", "low": 0, "high": 256},
            "value": 1.0,
        },
        source="builtin",
    )
    r(
        "aten::index_fill_.int_Tensor",
        {
            "self": (256, 4096),
            "dim": 0,
            "index": {"shape": (64,), "dtype": "long", "low": 0, "high": 256},
            "value": (),
        },
        source="builtin",
    )
    r("aten::isfinite", {"self": (256, 4096)}, source="builtin")
    r("aten::isinf", {"self": (256, 4096)}, source="builtin")
    r(
        "aten::isinf.out",
        {"self": (256, 4096), "out": {"shape": (256, 4096), "dtype": "bool"}},
        source="builtin",
    )
    r("aten::isnan", {"self": (256, 4096)}, source="builtin")
    r(
        "aten::isnan.out",
        {"self": (256, 4096), "out": {"shape": (256, 4096), "dtype": "bool"}},
        source="builtin",
    )
    r("aten::isneginf", {"self": (256, 4096)}, source="builtin")
    r(
        "aten::isneginf.out",
        {"self": (256, 4096), "out": {"shape": (256, 4096), "dtype": "bool"}},
        source="builtin",
    )
    r("aten::isposinf", {"self": (256, 4096)}, source="builtin")
    r(
        "aten::isposinf.out",
        {"self": (256, 4096), "out": {"shape": (256, 4096), "dtype": "bool"}},
        source="builtin",
    )
    r(
        "aten::l1_loss",
        {"self": (256, 4096), "target": (256, 4096), "reduction": 1},
        source="builtin",
    )
    r(
        "aten::lcm.out",
        {
            "self": {"shape": (256, 4096), "dtype": "int"},
            "other": {"shape": (256, 4096), "dtype": "int"},
            "out": {"shape": (256, 4096), "dtype": "int"},
        },
        source="builtin",
    )
    r(
        "aten::lcm_.default",
        {
            "self": {"shape": (256, 1024), "dtype": "int"},
            "other": {"shape": (256, 1024), "dtype": "int"},
        },
        source="builtin",
    )
    r("aten::ldexp.Tensor", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r("aten::ldexp.out", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r("aten::le.Scalar", {"self": (256, 4096), "other": 1.0}, source="builtin")
    r("aten::le.Scalar_out", {"self": (256, 4096), "other": 1.0}, source="builtin")
    r("aten::le.Tensor", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::le.Tensor_out",
        {"self": (256, 4096), "other": (256, 4096)},
        source="builtin",
    )
    r("aten::leaky_relu", {"self": (256, 4096), "negative_slope": 1.0}, source="builtin")
    r(
        "aten::leaky_relu.out",
        {"self": (256, 4096), "negative_slope": 1.0},
        source="builtin",
    )
    r(
        "aten::leaky_relu_backward",
        {
            "grad_output": (256, 4096),
            "self": (256, 4096),
            "negative_slope": 1.0,
            "self_is_result": False,
        },
        source="builtin",
    )
    r(
        "aten::leaky_relu_backward.grad_input",
        {
            "grad_output": (256, 4096),
            "self": (256, 4096),
            "negative_slope": 1.0,
            "self_is_result": False,
            "grad_input": (256, 4096),
        },
        source="builtin",
    )
    r(
        "aten::lerp_.Scalar",
        {"self": (256, 4096), "end": (256, 4096), "weight": 1.0},
        source="builtin",
    )
    r("aten::lgamma", {"self": (256, 4096)}, source="builtin")
    r("aten::lgamma.out", {"self": (256, 4096)}, source="builtin")
    r("aten::lgamma_", {"self": (256, 4096)}, source="builtin")
    r("aten::linalg_matrix_exp", {"self": (256, 256)}, source="builtin")
    r("aten::linalg_matrix_exp.out", {"self": (256, 256)}, source="builtin")
    r(
        "aten::linalg_norm",
        {"self": (256, 4096), "ord": 1.0, "dim": [0], "keepdim": False},
        source="builtin",
    )
    r(
        "aten::linalg_norm.out",
        {"self": (256, 4096), "ord": 1.0, "dim": [0], "keepdim": False},
        source="builtin",
    )
    r("aten::linalg_svd", {"A": (256, 4096), "full_matrices": False}, source="builtin")
    r("aten::linalg_svdvals", {"A": (256, 4096)}, source="builtin")
    r("aten::linalg_svdvals.out", {"A": (256, 4096)}, source="builtin")
    r(
        "aten::linalg_vector_norm",
        {"self": (256, 4096), "ord": 1.0, "dim": [0], "keepdim": False},
        source="builtin",
    )
    r(
        "aten::linalg_vector_norm.out",
        {"self": (256, 4096), "ord": 1.0, "dim": [0], "keepdim": False},
        source="builtin",
    )
    r(
        "aten::linear.out",
        {"input": (256, 4096), "weight": (4096, 4096)},
        source="builtin",
    )
    r("aten::log.out", {"self": (256, 4096)}, source="builtin")
    r("aten::log10", {"self": (256, 4096)}, source="builtin")
    r("aten::log10.out", {"self": (256, 4096)}, source="builtin")
    r("aten::log1p", {"self": (256, 4096)}, source="builtin")
    r("aten::log1p.out", {"self": (256, 4096)}, source="builtin")
    r("aten::log2", {"self": (256, 4096)}, source="builtin")
    r("aten::log2.out", {"self": (256, 4096)}, source="builtin")
    r("aten::log_normal_", {"self": (256, 4096), "mean": 1.0, "std": 1.0}, source="builtin")
    r("aten::log_sigmoid", {"self": (256, 4096)}, source="builtin")
    r("aten::log_sigmoid.out", {"self": (256, 4096)}, source="builtin")
    r(
        "aten::log_sigmoid_backward",
        {"grad_output": (256, 4096), "self": (256, 4096), "buffer": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::log_sigmoid_backward.grad_input",
        {
            "grad_output": (256, 4096),
            "self": (256, 4096),
            "buffer": (256, 4096),
            "grad_input": (256, 4096),
        },
        source="builtin",
    )
    r("aten::log_sigmoid_forward", {"self": (256, 4096)}, source="builtin")
    r(
        "aten::log_sigmoid_forward.output",
        {"self": (256, 4096), "output": (256, 4096), "buffer": (256, 4096)},
        source="builtin",
    )
    r("aten::logaddexp", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::logaddexp.out",
        {"self": (256, 4096), "other": (256, 4096)},
        source="builtin",
    )
    r("aten::logical_and", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::logical_and.out",
        {"self": (256, 4096), "other": (256, 4096)},
        source="builtin",
    )
    r("aten::logical_not", {"self": (256, 4096)}, source="builtin")
    r("aten::logical_not.out", {"self": (256, 4096)}, source="builtin")
    r("aten::logical_or", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::logical_or.out",
        {"self": (256, 4096), "other": (256, 4096)},
        source="builtin",
    )
    r("aten::logical_xor", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::logical_xor.out",
        {"self": (256, 4096), "other": (256, 4096)},
        source="builtin",
    )
    r("aten::logit", {"self": (256, 4096), "eps": 1e-5}, source="builtin")
    r("aten::logit.out", {"self": (256, 4096), "eps": 1e-5}, source="builtin")
    r("aten::logit_", {"self": (256, 4096), "eps": 1e-5}, source="builtin")
    r(
        "aten::logit_backward",
        {"grad_output": (256, 4096), "self": (256, 4096), "eps": 1e-5},
        source="builtin",
    )
    r(
        "aten::logit_backward.grad_input",
        {"grad_output": (256, 4096), "self": (256, 4096), "eps": 1e-5, "grad_input": (256, 4096)},
        source="builtin",
    )
    r("aten::lt.Scalar", {"self": (256, 4096), "other": 1.0}, source="builtin")
    r("aten::lt.Scalar_out", {"self": (256, 4096), "other": 1.0}, source="builtin")
    r("aten::lt.Tensor", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::lt.Tensor_out",
        {"self": (256, 4096), "other": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::margin_ranking_loss",
        {
            "input1": (256, 4096),
            "input2": (256, 4096),
            "target": (256, 4096),
            "margin": 1.0,
            "reduction": 1,
        },
        source="builtin",
    )
    r(
        "aten::masked_fill_.Scalar",
        {"self": (256, 4096), "mask": {"shape": (256, 4096), "dtype": "bool"}, "value": 1.0},
        source="builtin",
    )
    r("aten::maximum", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::maximum.out",
        {"self": (256, 4096), "other": (256, 4096)},
        source="builtin",
    )
    r("aten::minimum", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::minimum.out",
        {"self": (256, 4096), "other": (256, 4096)},
        source="builtin",
    )
    r("aten::mish", {"self": (256, 4096)}, source="builtin")
    r("aten::mish.out", {"self": (256, 4096)}, source="builtin")
    r("aten::mish_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::mish_backward", {"grad_output": (256, 4096), "self": (256, 4096)}, source="builtin")
    r(
        "aten::mse_loss",
        {"self": (256, 4096), "target": (256, 4096), "reduction": 1},
        source="builtin",
    )
    r(
        "aten::mse_loss.out",
        {"self": (256, 4096), "target": (256, 4096), "reduction": 1},
        source="builtin",
    )
    r(
        "aten::mse_loss_backward",
        {"grad_output": (256, 4096), "self": (256, 4096), "target": (256, 4096), "reduction": 1},
        source="builtin",
    )
    r(
        "aten::mse_loss_backward.grad_input",
        {
            "grad_output": (256, 4096),
            "self": (256, 4096),
            "target": (256, 4096),
            "reduction": 1,
            "grad_input": (256, 4096),
        },
        source="builtin",
    )
    r("aten::mul.Scalar_out", {"self": (256, 4096), "other": 1.0}, source="builtin")
    r("aten::mul.out", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r("aten::mul_.Scalar", {"self": (256, 4096), "other": 1.0}, source="builtin")
    r("aten::mul_.Tensor", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::multi_margin_loss",
        {
            "self": (256, 4096),
            "target": {"shape": (256,), "dtype": "long", "low": 0, "high": 4096},
            "p": 1.0,
            "margin": 1.0,
            "reduction": 1,
        },
        source="builtin",
    )
    r(
        "aten::multi_margin_loss.out",
        {
            "self": (256, 4096),
            "target": {"shape": (256,), "dtype": "long", "low": 0, "high": 4096},
            "p": 1.0,
            "margin": 1.0,
            "reduction": 1,
            "out": (0,),
        },
        source="builtin",
    )
    r(
        "aten::multilabel_margin_loss_forward",
        {
            "self": (256, 4096),
            "target": {"shape": (256, 4096), "dtype": "long", "low": -1, "high": 4096},
            "reduction": 1,
        },
        source="builtin",
    )
    r(
        "aten::multilabel_margin_loss_forward.output",
        {
            "self": (256, 4096),
            "target": {"shape": (256, 4096), "dtype": "long", "low": -1, "high": 4096},
            "reduction": 1,
            "output": (256, 4096),
            "is_target": (256, 4096),
        },
        source="builtin",
    )
    r(
        "aten::nan_to_num",
        {"self": (256, 4096), "nan": 1.0, "posinf": 1.0, "neginf": 1.0},
        source="builtin",
    )
    r(
        "aten::nan_to_num.out",
        {"self": (256, 4096), "nan": 1.0, "posinf": 1.0, "neginf": 1.0},
        source="builtin",
    )
    r(
        "aten::nan_to_num_",
        {"self": (256, 4096), "nan": 1.0, "posinf": 1.0, "neginf": 1.0},
        source="builtin",
    )
    r("aten::native_dropout", {"input": (256, 4096), "p": 1.0, "train": False}, source="builtin")
    r(
        "aten::native_dropout.out",
        {
            "input": (256, 4096),
            "p": 0.5,
            "train": False,
            "out0": (256, 4096),
            "out1": {"shape": (256, 4096), "dtype": "bool"},
        },
        source="builtin",
    )
    r("aten::ne.Scalar", {"self": (256, 4096), "other": 1.0}, source="builtin")
    r("aten::ne.Scalar_out", {"self": (256, 4096), "other": 1.0}, source="builtin")
    r("aten::ne.Tensor", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::ne.Tensor_out",
        {"self": (256, 4096), "other": (256, 4096)},
        source="builtin",
    )
    r("aten::neg.out", {"self": (256, 4096)}, source="builtin")
    r("aten::nextafter", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::nextafter.out",
        {"self": (256, 4096), "other": (256, 4096)},
        source="builtin",
    )
    r("aten::nextafter_", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::nll_loss",
        {
            "self": (256, 4096),
            "target": {"shape": (256,), "dtype": "long", "low": 0, "high": 4096},
            "reduction": 1,
            "ignore_index": -100,
        },
        source="builtin",
    )
    r(
        "aten::nll_loss.out",
        {
            "self": (256, 4096),
            "target": {"shape": (256,), "dtype": "long", "low": 0, "high": 4096},
            "reduction": 1,
            "ignore_index": -100,
            "out": (0,),
        },
        source="builtin",
    )
    r(
        "aten::nll_loss2d_backward",
        {
            "grad_output": (1,),
            "self": (2, 4, 8, 8),
            "target": {"shape": (2, 8, 8), "dtype": "long", "low": 0, "high": 4},
            "weight": None,
            "reduction": 1,
            "ignore_index": 1,
            "total_weight": (1,),
        },
        source="builtin",
    )
    r(
        "aten::nll_loss2d_backward.grad_input",
        {
            "grad_output": (1,),
            "self": (2, 4, 8, 8),
            "target": {"shape": (2, 8, 8), "dtype": "long", "low": 0, "high": 4},
            "weight": None,
            "reduction": 1,
            "ignore_index": 1,
            "total_weight": (1,),
            "grad_input": (2, 4, 8, 8),
        },
        source="builtin",
    )
    r(
        "aten::nll_loss2d_forward",
        {
            "self": (2, 4, 8, 8),
            "target": {"shape": (2, 8, 8), "dtype": "long", "low": 0, "high": 4},
            "weight": None,
            "reduction": 1,
            "ignore_index": 1,
        },
        source="builtin",
    )
    r(
        "aten::nll_loss2d_forward.output",
        {
            "self": (2, 4, 8, 8),
            "target": {"shape": (2, 8, 8), "dtype": "long", "low": 0, "high": 4},
            "weight": None,
            "reduction": 1,
            "ignore_index": 1,
            "output": (0,),
            "total_weight": (1,),
        },
        source="builtin",
    )
    r(
        "aten::nll_loss_backward",
        {
            "grad_output": (1,),
            "self": (256, 4096),
            "target": {"shape": (256,), "dtype": "long", "low": 0, "high": 4096},
            "weight": None,
            "reduction": 1,
            "ignore_index": -100,
            "total_weight": (1,),
        },
        source="builtin",
    )
    r(
        "aten::nll_loss_forward",
        {
            "self": (256, 4096),
            "target": {"shape": (256,), "dtype": "long", "low": 0, "high": 4096},
            "weight": None,
            "reduction": 1,
            "ignore_index": -100,
        },
        source="builtin",
    )
    r("aten::normal_", {"self": (256, 4096), "mean": 1.0, "std": 1.0}, source="builtin")
    r(
        "aten::pairwise_distance",
        {"x1": (256, 4096), "x2": (256, 4096), "p": 1.0, "eps": 1e-5, "keepdim": False},
        source="builtin",
    )
    r(
        "aten::poisson_nll_loss",
        {
            "input": (256, 4096),
            "target": (256, 4096),
            "log_input": False,
            "full": False,
            "eps": 1e-5,
            "reduction": 1,
        },
        source="builtin",
    )
    r("aten::pow.Scalar", {"self": 1.0, "exponent": (256, 4096)}, source="builtin")
    r("aten::pow.Scalar_out", {"self": 1.0, "exponent": (256, 4096)}, source="builtin")
    r("aten::pow.Tensor_Scalar", {"self": (256, 4096), "exponent": 1.0}, source="builtin")
    r(
        "aten::pow.Tensor_Scalar_out",
        {"self": (256, 4096), "exponent": 1.0},
        source="builtin",
    )
    r("aten::pow.Tensor_Tensor", {"self": (256, 4096), "exponent": (256, 4096)}, source="builtin")
    r(
        "aten::pow.Tensor_Tensor_out",
        {"self": (256, 4096), "exponent": (256, 4096)},
        source="builtin",
    )
    r("aten::prelu", {"self": (256, 4096), "weight": (1,)}, source="builtin")
    r("aten::prelu.out", {"self": (256, 4096), "weight": (1,)}, source="builtin")
    r(
        "aten::put_",
        {
            "self": (256, 4096),
            "index": {"shape": (64,), "dtype": "long", "low": 0, "high": 256},
            "source": (64,),
            "accumulate": False,
        },
        source="builtin",
    )
    r("aten::rad2deg", {"self": (256, 4096)}, source="builtin")
    r("aten::rad2deg.out", {"self": (256, 4096)}, source="builtin")
    r("aten::random_", {"self": (256, 4096)}, source="builtin")
    r("aten::random_.from", {"self": (256, 4096), "from": 0, "to": 10}, source="builtin")
    r("aten::random_.to", {"self": (256, 4096), "to": 1}, source="builtin")
    r("aten::reciprocal", {"self": (256, 4096)}, source="builtin")
    r("aten::reciprocal.out", {"self": (256, 4096)}, source="builtin")
    r("aten::reciprocal_", {"self": (256, 4096)}, source="builtin")
    r("aten::reflection_pad1d", {"self": (1, 64, 4096), "padding": [1, 1]}, source="builtin")
    r(
        "aten::reflection_pad2d",
        {"self": (1, 64, 32, 32), "padding": [1, 1, 1, 1]},
        source="builtin",
    )
    r(
        "aten::reflection_pad3d",
        {"self": (1, 32, 16, 16, 16), "padding": [1, 1, 1, 1, 1, 1]},
        source="builtin",
    )
    r("aten::relu.out", {"self": (256, 4096)}, source="builtin")
    r("aten::relu6", {"self": (256, 4096)}, source="builtin")
    r("aten::relu6_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::relu_", {"self": (256, 4096)}, source="builtin")
    r("aten::remainder.Scalar", {"self": (256, 4096), "other": 1.0}, source="builtin")
    r("aten::remainder.Scalar_Tensor", {"self": 1.0, "other": (256, 4096)}, source="builtin")
    r(
        "aten::remainder.Scalar_Tensor_out",
        {"self": 1.0, "other": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::remainder.Scalar_out",
        {"self": (256, 4096), "other": 1.0},
        source="builtin",
    )
    r(
        "aten::remainder.Tensor",
        {"self": (256, 4096), "other": {"shape": (256, 4096), "low": 1}},
        source="builtin",
    )
    r(
        "aten::remainder.Tensor_out",
        {"self": (256, 4096), "other": {"shape": (256, 4096), "low": 1}},
        source="builtin",
    )
    r(
        "aten::replication_pad2d",
        {"self": (1, 64, 32, 32), "padding": [1, 1, 1, 1]},
        source="builtin",
    )
    r(
        "aten::replication_pad2d.out",
        {"self": (1, 64, 32, 32), "padding": [1, 1, 1, 1]},
        source="builtin",
    )
    r(
        "aten::replication_pad3d",
        {"self": (1, 32, 16, 16, 16), "padding": [1, 1, 1, 1, 1, 1]},
        source="builtin",
    )
    r("aten::round", {"self": (256, 4096)}, source="builtin")
    r("aten::round.decimals", {"self": (256, 4096), "decimals": 1}, source="builtin")
    r(
        "aten::round.decimals_out",
        {"self": (256, 4096), "decimals": 1},
        source="builtin",
    )
    r("aten::round.out", {"self": (256, 4096)}, source="builtin")
    r(
        "aten::rrelu",
        {"self": (256, 4096), "lower": 1.0, "upper": 1.0, "training": False},
        source="builtin",
    )
    r(
        "aten::rrelu_.default",
        {"self": (256, 4096), "lower": 1.0, "upper": 1.0, "training": False},
        source="builtin",
    )
    r(
        "aten::rrelu_with_noise",
        {"self": (256, 4096), "noise": (256, 4096), "lower": 1.0, "upper": 1.0, "training": False},
        source="builtin",
    )
    r(
        "aten::rrelu_with_noise.out",
        {
            "self": (256, 4096),
            "noise": (256, 4096),
            "lower": 1.0,
            "upper": 1.0,
            "training": False,
            "out": (0,),
        },
        source="builtin",
    )
    r(
        "aten::rrelu_with_noise_.default",
        {"self": (256, 4096), "noise": (256, 4096), "lower": 1.0, "upper": 1.0, "training": False},
        source="builtin",
    )
    r(
        "aten::rrelu_with_noise_backward",
        {
            "grad_output": (256, 4096),
            "self": (256, 4096),
            "noise": (256, 4096),
            "lower": 1.0,
            "upper": 1.0,
            "training": False,
            "self_is_result": False,
        },
        source="builtin",
    )
    r(
        "aten::rrelu_with_noise_backward.out",
        {
            "grad_output": (256, 4096),
            "self": (256, 4096),
            "noise": (256, 4096),
            "lower": 1.0,
            "upper": 1.0,
            "training": False,
            "self_is_result": False,
            "out": (0,),
        },
        source="builtin",
    )
    r(
        "aten::rrelu_with_noise_functional",
        {"self": (256, 4096), "noise": (256, 4096), "lower": 1.0, "upper": 1.0, "training": False},
        source="builtin",
    )
    r("aten::rsqrt.out", {"self": (256, 4096)}, source="builtin")
    r("aten::rsub.Scalar", {"self": (256, 4096), "other": 1.0, "alpha": 1.0}, source="builtin")
    r(
        "aten::rsub.Scalar_out",
        {"self": (256, 4096), "other": 1.0, "alpha": 1.0},
        source="builtin",
    )
    r("aten::rsub.Tensor", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::rsub.Tensor_out",
        {"self": (256, 4096), "other": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::scatter_add_",
        {
            "self": (256, 4096),
            "dim": 0,
            "index": {"shape": (64, 4096), "dtype": "long", "low": 0, "high": 256},
            "src": (64, 4096),
        },
        source="builtin",
    )
    r("aten::selu", {"self": (256, 4096)}, source="builtin")
    r("aten::selu_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::sgn", {"self": (256, 4096)}, source="builtin")
    r("aten::sgn.out", {"self": (256, 4096)}, source="builtin")
    r("aten::sgn_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::sigmoid.out", {"self": (256, 4096)}, source="builtin")
    r(
        "aten::sigmoid_backward",
        {"grad_output": (256, 4096), "output": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::sigmoid_backward.grad_input",
        {"grad_output": (256, 4096), "output": (256, 4096), "grad_input": (256, 4096)},
        source="builtin",
    )
    r("aten::sign", {"self": (256, 4096)}, source="builtin")
    r("aten::sign.out", {"self": (256, 4096)}, source="builtin")
    r("aten::signbit", {"self": (256, 4096)}, source="builtin")
    r("aten::signbit.out", {"self": (256, 4096)}, source="builtin")
    r("aten::silu.out", {"self": (256, 4096)}, source="builtin")
    r("aten::silu_backward", {"grad_output": (256, 4096), "self": (256, 4096)}, source="builtin")
    r(
        "aten::silu_backward.grad_input",
        {"grad_output": (256, 4096), "self": (256, 4096), "grad_input": (256, 4096)},
        source="builtin",
    )
    r("aten::sin.out", {"self": (256, 4096)}, source="builtin")
    r("aten::sinc", {"self": (256, 4096)}, source="builtin")
    r("aten::sinc.out", {"self": (256, 4096)}, source="builtin")
    r("aten::sinh", {"self": (256, 4096)}, source="builtin")
    r("aten::sinh.out", {"self": (256, 4096)}, source="builtin")
    r(
        "aten::smooth_l1_loss",
        {"self": (256, 4096), "target": (256, 4096), "reduction": 1, "beta": 1.0},
        source="builtin",
    )
    r(
        "aten::smooth_l1_loss.out",
        {"self": (256, 4096), "target": (256, 4096), "reduction": 1, "beta": 1.0},
        source="builtin",
    )
    r(
        "aten::smooth_l1_loss_backward",
        {
            "grad_output": (256, 4096),
            "self": (256, 4096),
            "target": (256, 4096),
            "reduction": 1,
            "beta": 1.0,
        },
        source="builtin",
    )
    r(
        "aten::smooth_l1_loss_backward.grad_input",
        {
            "grad_output": (256, 4096),
            "self": (256, 4096),
            "target": (256, 4096),
            "reduction": 1,
            "beta": 1.0,
            "grad_input": (256, 4096),
        },
        source="builtin",
    )
    r(
        "aten::soft_margin_loss",
        {"self": (256, 4096), "target": (256, 4096), "reduction": 1},
        source="builtin",
    )
    r(
        "aten::soft_margin_loss.out",
        {"self": (256, 4096), "target": (256, 4096), "reduction": 1},
        source="builtin",
    )
    r(
        "aten::soft_margin_loss_backward",
        {"grad_output": (256, 4096), "self": (256, 4096), "target": (256, 4096), "reduction": 1},
        source="builtin",
    )
    r("aten::softplus", {"self": (256, 4096), "beta": 1.0, "threshold": 1.0}, source="builtin")
    r(
        "aten::softplus.out",
        {"self": (256, 4096), "beta": 1.0, "threshold": 1.0},
        source="builtin",
    )
    r(
        "aten::softplus_backward",
        {"grad_output": (256, 4096), "self": (256, 4096), "beta": 1.0, "threshold": 1.0},
        source="builtin",
    )
    r(
        "aten::softplus_backward.grad_input",
        {
            "grad_output": (256, 4096),
            "self": (256, 4096),
            "beta": 1.0,
            "threshold": 1.0,
            "grad_input": (256, 4096),
        },
        source="builtin",
    )
    r("aten::softshrink", {"self": (256, 4096), "lambd": 1.0}, source="builtin")
    r("aten::softshrink.out", {"self": (256, 4096), "lambd": 1.0}, source="builtin")
    r(
        "aten::softshrink_backward",
        {"grad_output": (256, 4096), "self": (256, 4096), "lambd": 1.0},
        source="builtin",
    )
    r(
        "aten::softshrink_backward.grad_input",
        {"grad_output": (256, 4096), "self": (256, 4096), "lambd": 1.0, "grad_input": (256, 4096)},
        source="builtin",
    )
    r("aten::special_bessel_j0", {"self": (256, 4096)}, source="builtin")
    r("aten::special_bessel_j0.out", {"self": (256, 4096)}, source="builtin")
    r("aten::special_bessel_j1", {"self": (256, 4096)}, source="builtin")
    r("aten::special_bessel_j1.out", {"self": (256, 4096)}, source="builtin")
    r("aten::special_entr", {"self": (256, 4096)}, source="builtin")
    r("aten::special_entr.out", {"self": (256, 4096)}, source="builtin")
    r("aten::special_erfc", {"self": (256, 4096)}, source="builtin")
    r("aten::special_erfc.out", {"self": (256, 4096)}, source="builtin")
    r("aten::special_erfcx", {"self": (256, 4096)}, source="builtin")
    r("aten::special_erfcx.out", {"self": (256, 4096)}, source="builtin")
    r("aten::special_erfinv", {"self": (256, 4096)}, source="builtin")
    r("aten::special_erfinv.out", {"self": (256, 4096)}, source="builtin")
    r("aten::special_gammainc", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::special_gammainc.out",
        {"self": (256, 4096), "other": (256, 4096)},
        source="builtin",
    )
    r("aten::special_gammaincc", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::special_gammaincc.out",
        {"self": (256, 4096), "other": (256, 4096)},
        source="builtin",
    )
    r("aten::special_i0e", {"self": (256, 4096)}, source="builtin")
    r("aten::special_i0e.out", {"self": (256, 4096)}, source="builtin")
    r("aten::special_i1", {"self": (256, 4096)}, source="builtin")
    r("aten::special_i1.out", {"self": (256, 4096)}, source="builtin")
    r("aten::special_i1e", {"self": (256, 4096)}, source="builtin")
    r("aten::special_i1e.out", {"self": (256, 4096)}, source="builtin")
    r("aten::special_laguerre_polynomial_l", {"x": (256, 4096), "n": (256, 4096)}, source="builtin")
    r(
        "aten::special_laguerre_polynomial_l.n_scalar",
        {"x": (256, 4096), "n": 1.0},
        source="builtin",
    )
    r(
        "aten::special_laguerre_polynomial_l.n_scalar_out",
        {"x": (256, 4096), "n": 1.0},
        source="builtin",
    )
    r(
        "aten::special_laguerre_polynomial_l.out",
        {"x": (256, 4096), "n": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::special_laguerre_polynomial_l.x_scalar",
        {"x": 1.0, "n": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::special_laguerre_polynomial_l.x_scalar_out",
        {"x": 1.0, "n": (256, 4096)},
        source="builtin",
    )
    r("aten::special_log_ndtr", {"self": (256, 4096)}, source="builtin")
    r("aten::special_log_ndtr.out", {"self": (256, 4096)}, source="builtin")
    r("aten::special_ndtri", {"self": (256, 4096)}, source="builtin")
    r("aten::special_ndtri.out", {"self": (256, 4096)}, source="builtin")
    r("aten::special_spherical_bessel_j0", {"x": (256, 4096)}, source="builtin")
    r("aten::special_spherical_bessel_j0.out", {"x": (256, 4096)}, source="builtin")
    r("aten::special_xlog1py", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::special_xlog1py.other_scalar_out",
        {"self": (256, 4096), "other": 1.0},
        source="builtin",
    )
    r(
        "aten::special_xlog1py.out",
        {"self": (256, 4096), "other": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::special_xlog1py.self_scalar_out",
        {"self": 1.0, "other": (256, 4096)},
        source="builtin",
    )
    r("aten::special_zeta", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::special_zeta.other_scalar_out",
        {"self": (256, 4096), "other": 1.0},
        source="builtin",
    )
    r(
        "aten::special_zeta.out",
        {"self": (256, 4096), "other": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::special_zeta.self_scalar_out",
        {"self": 1.0, "other": (256, 4096)},
        source="builtin",
    )
    r("aten::sqrt.out", {"self": (256, 4096)}, source="builtin")
    r("aten::square", {"self": (256, 4096)}, source="builtin")
    r("aten::square.out", {"self": (256, 4096)}, source="builtin")
    r("aten::square_.default", {"self": (256, 4096)}, source="builtin")
    r("aten::sub.Scalar", {"self": (256, 4096), "other": 1.0, "alpha": 1.0}, source="builtin")
    r(
        "aten::sub.Scalar_out",
        {"self": (256, 4096), "other": 1.0, "alpha": 1.0},
        source="builtin",
    )
    r("aten::sub.out", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r("aten::sub_.Scalar", {"self": (256, 4096), "other": 1.0, "alpha": 1.0}, source="builtin")
    r("aten::sub_.Tensor", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r("aten::tan", {"self": (256, 4096)}, source="builtin")
    r("aten::tan.out", {"self": (256, 4096)}, source="builtin")
    r("aten::tanh.out", {"self": (256, 4096)}, source="builtin")
    r("aten::tanh_backward", {"grad_output": (256, 4096), "output": (256, 4096)}, source="builtin")
    r(
        "aten::tanh_backward.grad_input",
        {"grad_output": (256, 4096), "output": (256, 4096), "grad_input": (256, 4096)},
        source="builtin",
    )
    r("aten::threshold", {"self": (256, 4096), "threshold": 1.0, "value": 1.0}, source="builtin")
    r(
        "aten::threshold.out",
        {"self": (256, 4096), "threshold": 1.0, "value": 1.0},
        source="builtin",
    )
    r(
        "aten::threshold_backward",
        {"grad_output": (8192, 8192), "self": (8192, 8192), "threshold": 1.0},
        source="builtin",
    )
    r(
        "aten::threshold_backward.grad_input",
        {
            "grad_output": (8192, 8192),
            "self": (8192, 8192),
            "threshold": 1.0,
            "grad_input": (8192, 8192),
        },
        source="builtin",
    )
    r("aten::tril_", {"self": (256, 4096), "diagonal": 1}, source="builtin")
    r("aten::trunc", {"self": (256, 4096)}, source="builtin")
    r("aten::trunc.out", {"self": (256, 4096)}, source="builtin")
    r("aten::uniform_", {"self": (256, 4096), "from": 1.0, "to": 1.0}, source="builtin")
    r(
        "aten::xlogy.OutScalar_Other",
        {"self": (256, 4096), "other": 1.0},
        source="builtin",
    )
    r(
        "aten::xlogy.OutScalar_Self",
        {"self": 1.0, "other": (256, 4096)},
        source="builtin",
    )
    r(
        "aten::xlogy.OutTensor",
        {"self": (256, 4096), "other": (256, 4096)},
        source="builtin",
    )
    r("aten::xlogy.Scalar_Other", {"self": (256, 4096), "other": 1.0}, source="builtin")
    r("aten::xlogy.Scalar_Self", {"self": 1.0, "other": (256, 4096)}, source="builtin")
    r("aten::xlogy.Tensor", {"self": (256, 4096), "other": (256, 4096)}, source="builtin")
    r(
        "aten::where.self",
        {
            "condition": {"shape": (256, 4096), "dtype": "bool"},
            "self": (256, 4096),
            "other": (256, 4096),
        },
        source="builtin",
    )
    r(
        "aten::where.self_out",
        {
            "condition": {"shape": (256, 4096), "dtype": "bool"},
            "self": (256, 4096),
            "other": (256, 4096),
            "out": (256, 4096),
        },
        source="builtin",
    )
    r(
        "aten::index_select.out",
        {
            "self": (256, 4096),
            "dim": 0,
            "index": {"shape": (64,), "dtype": "long", "low": 0, "high": 256},
            "out": (64, 4096),
        },
        source="builtin",
    )
    r(
        "aten::bincount.out",
        {
            "self": {"shape": (8192,), "dtype": "long", "low": 0, "high": 1024},
            "out": {"shape": (1024,), "dtype": "long"},
        },
        source="builtin",
    )
    r(
        "aten::index_copy.out",
        {
            "self": (256, 4096),
            "dim": 0,
            "index": {"shape": (64,), "dtype": "long", "low": 0, "high": 256},
            "source": (64, 4096),
            "out": (256, 4096),
        },
        source="builtin",
    )
    r(
        "aten::_grouped_mm",
        {
            # _grouped_mm requires BF16 inputs — the meta kernel
            # (torch/_meta_registrations.py::_meta_grouped_mm_common) rejects
            # fp32 with "Expected inputs of BF16 type but got mat_a.dtype=..."
            # during Dynamo FakeTensor propagation, so the sample must force
            # bfloat16. Verified 2026-07-20.
            "self": {"shape": (64, 256), "dtype": "bfloat16"},
            "mat2": {"shape": (256, 128), "dtype": "bfloat16"},
            # offs values must be sorted and divisible by 128 (NKI tile alignment).
            "offs": {
                "shape": (3,),
                "dtype": "int",
                "low": 0,
                "high": 3,
                "multiple_of": 128,
                "sorted": True,
            },
        },
        source="builtin",
    )
    # DeepSeek-V3-16B MoE forward grouped_mm (2Dx3D ragged): a[R,d1] @ b[E,d1,d2].
    # R=T*top_k=24576, E=64, h=2048, moe_intermediate=1408. offs [E] cumulative,
    # sorted, x128-aligned (high=192 -> max 192*128=24448 ~ R).
    r(
        "aten::_grouped_mm",  # DeepSeek gate/up: [R,D] @ [E,D,I] -> [R,I]
        {
            "self": {"shape": (24576, 2048), "dtype": "bfloat16"},
            "mat2": {"shape": (64, 2048, 1408), "dtype": "bfloat16"},
            "offs": {
                "shape": (64,),
                "dtype": "int",
                "low": 0,
                "high": 192,
                "multiple_of": 128,
                "sorted": True,
            },
        },
        source="builtin",
    )
    r(
        "aten::_grouped_mm",  # DeepSeek down: [R,I] @ [E,I,D] -> [R,D]
        {
            "self": {"shape": (24576, 1408), "dtype": "bfloat16"},
            "mat2": {"shape": (64, 1408, 2048), "dtype": "bfloat16"},
            "offs": {
                "shape": (64,),
                "dtype": "int",
                "low": 0,
                "high": 192,
                "multiple_of": 128,
                "sorted": True,
            },
        },
        source="builtin",
    )
    r(
        "aten::embedding",
        {
            "weight": (32000, 4096),
            "indices": {"shape": (1, 2048), "dtype": "long", "low": 0, "high": 32000},
        },
        source="builtin",
    )
    # convolution: NCHW
    r(
        "aten::convolution",
        {
            "input": (1, 3, 224, 224),
            "weight": (64, 3, 7, 7),
            "bias": (64,),
            "stride": [2, 2],
            "padding": [3, 3],
            "dilation": [1, 1],
            "transposed": False,
            "output_padding": [0, 0],
            "groups": 1,
        },
        source="builtin",
    )
    r(
        "aten::convolution.out",
        {
            "input": (1, 3, 224, 224),
            "weight": (64, 3, 7, 7),
            "bias": (64,),
            "stride": [2, 2],
            "padding": [3, 3],
            "dilation": [1, 1],
            "transposed": False,
            "output_padding": [0, 0],
            "groups": 1,
            "out": (1, 64, 112, 112),
        },
        source="builtin",
    )
    r(
        "aten::one_hot",
        {"self": {"shape": (256,), "dtype": "long", "low": 0, "high": 1024}, "num_classes": 1024},
        source="builtin",
    )
    r(
        "aten::_fused_adamw_",
        {
            "self": [(256, 4096)],
            "grads": [(256, 4096)],
            "exp_avgs": [(256, 4096)],
            "exp_avg_sqs": [(256, 4096)],
            "max_exp_avg_sqs": [(256, 4096)],
            "state_steps": [{"shape": (1,), "dtype": "long"}],
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.01,
            "eps": 1e-8,
            "amsgrad": False,
            "maximize": False,
        },
        source="builtin",
    )
    # _fused_adam_
    r(
        "aten::_fused_adam_",
        {
            "self": [(256, 4096)],
            "grads": [(256, 4096)],
            "exp_avgs": [(256, 4096)],
            "exp_avg_sqs": [(256, 4096)],
            "max_exp_avg_sqs": [(256, 4096)],
            "state_steps": [{"shape": (1,), "dtype": "long"}],
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.01,
            "eps": 1e-8,
            "amsgrad": False,
            "maximize": False,
        },
        source="builtin",
    )
    # _native_multi_head_attention
    r(
        "aten::_native_multi_head_attention",
        {
            "query": (2, 64, 256),
            "key": (2, 64, 256),
            "value": (2, 64, 256),
            "embed_dim": 256,
            "num_head": 4,
            "qkv_weight": (768, 256),
            "qkv_bias": (768,),
            "proj_weight": (256, 256),
            "proj_bias": (256,),
            "mask": None,
            "need_weights": False,
        },
        source="builtin",
    )
    r(
        "aten::scatter_reduce.two",
        {
            "self": (256, 4096),
            "dim": 0,
            "index": {"shape": (64, 4096), "dtype": "long", "low": 0, "high": 256},
            "src": (64, 4096),
            "reduce": "sum",
        },
        source="builtin",
    )
    r(
        "aten::scatter_reduce.two_out",
        {
            "self": (256, 4096),
            "dim": 0,
            "index": {"shape": (64, 4096), "dtype": "long", "low": 0, "high": 256},
            "src": (64, 4096),
            "reduce": "sum",
            "out": (256, 4096),
        },
        source="builtin",
    )
    r(
        "aten::repeat_interleave.Tensor",
        {
            # Deterministic repeats so that sum(repeats) == output_size == 512.
            # Random low/high produced a variable sum that triggered
            # "allocated size does not match required size" when a model-trace
            # entry supplied a fixed output_size.
            "repeats": {"shape": (256,), "dtype": "long", "low": 2, "high": 3},
            # ``repeat_interleave(repeats)`` has a data-dependent output shape
            # (``sum(repeats)``). Without an explicit ``output_size``, the
            # decomposition uses repeats.sum() as a symbolic tensor, producing
            # a dynamic shape (tensor<?xi64>) that StableHLO rejects under
            # fullgraph=True compilation.
            "output_size": 512,
        },
        source="builtin",
    )
    r(
        "aten::repeat_interleave.Tensor_out",
        {
            # Deterministic repeats so that sum(repeats) == output_size == 512
            # matches the pre-allocated out buffer. Random low/high produced a
            # variable sum that the CPU oracle rejected as
            # "allocated size does not match required size". Fixed 2026-07-20.
            "repeats": {"shape": (256,), "dtype": "long", "low": 2, "high": 3},
            # ``repeat_interleave(repeats)`` has a data-dependent output shape
            # (``sum(repeats)``). Without an explicit ``output_size``, meta
            # (torch/_meta_registrations.py::meta_repeat_interleave_Tensor)
            # raises "cannot repeat_interleave a meta tensor without
            # output_size" during Dynamo FakeTensor propagation. Verified
            # 2026-07-20: with output_size=512 the FakeTensor call succeeds.
            "output_size": 512,
            "out": {"shape": (512,), "dtype": "long"},
        },
        source="builtin",
    )
    r(
        "aten::repeat_interleave.self_int",
        {"self": (256, 4096), "repeats": 2, "dim": 0},
        source="builtin",
    )

    # ── Linalg ops missing shape entries ───────────────────────────────────
    # Sizes chosen to satisfy each op's geometric constraints (square matrices
    # for det/slogdet/lstsq; last dim = 3 for cross; compatible (M,K)@(K,N)
    # for matmul_backward and einsum). Multi-output write-alias kwargs are
    # pre-populated explicitly because `_with_out_kwargs` skips multi-out
    # schemas.

    # linalg_cross — 3D cross product; last dim must be 3
    r(
        "aten::linalg_cross",
        {"self": (32, 3), "other": (32, 3), "dim": -1},
        source="builtin",
    )
    r(
        "aten::linalg_cross.out",
        {"self": (32, 3), "other": (32, 3), "dim": -1, "out": (32, 3)},
        source="builtin",
    )

    # _linalg_det — returns (result, LU, pivots); LU shape = A shape,
    # pivots shape = (N,) with int32, result is 0-d scalar
    r("aten::_linalg_det", {"A": (256, 256)}, source="builtin")
    r(
        "aten::_linalg_det.result",
        {
            "A": (256, 256),
            "result": (),
            "LU": (256, 256),
            "pivots": {"shape": (256,), "dtype": "int"},
        },
        source="builtin",
    )

    # linalg_lstsq.out — least-squares with multiple RHS (K=8).
    # solution: (N, K); residuals: (K,); rank: 0-d int; singular_values: (N,)
    r(
        "aten::linalg_lstsq.out",
        {
            "self": (256, 256),
            "b": (256, 8),
            "rcond": None,
            "solution": (256, 8),
            "residuals": (8,),
            "rank": {"shape": (), "dtype": "int"},
            "singular_values": (256,),
        },
        source="builtin",
    )

    # linalg_slogdet — returns (sign, logabsdet); both scalars for 2D A
    r("aten::linalg_slogdet", {"A": (256, 256)}, source="builtin")
    r(
        "aten::linalg_slogdet.out",
        {"A": (256, 256), "sign": (), "logabsdet": ()},
        source="builtin",
    )

    # matmul_backward — grads for (256, 1024) @ (1024, 4096) → (256, 4096)
    r(
        "aten::matmul_backward",
        {
            "grad": (256, 4096),
            "self": (256, 1024),
            "other": (1024, 4096),
            "mask": [True, True],
        },
        source="builtin",
    )
    r(
        "aten::matmul_backward.out",
        {
            "grad": (256, 4096),
            "self": (256, 1024),
            "other": (1024, 4096),
            "mask": [True, True],
            "out0": (256, 1024),
            "out1": (1024, 4096),
        },
        source="builtin",
    )

    # _addmm_activation — self (bias, N,) + mat1 (M,K) @ mat2 (K,N) → (M, N)
    r(
        "aten::_addmm_activation",
        {"self": (4096,), "mat1": (256, 4096), "mat2": (4096, 4096)},
        source="builtin",
    )
    r(
        "aten::_addmm_activation.out",
        {
            "self": (4096,),
            "mat1": (256, 4096),
            "mat2": (4096, 4096),
            "out": (256, 4096),
        },
        source="builtin",
    )

    # einsum — explicit 2-tensor matmul via einsum string
    r(
        "aten::einsum",
        {
            "equation": "ij,jk->ik",
            "tensors": [{"shape": (256, 1024)}, {"shape": (1024, 4096)}],
        },
        source="builtin",
    )

    # linalg_norm.ord_str_out — Frobenius norm of a matrix; scalar output
    r(
        "aten::linalg_norm.ord_str_out",
        {
            "self": (256, 256),
            "ord": "fro",
            "dim": None,
            "keepdim": False,
            "out": (),
        },
        source="builtin",
    )

    # ── Indexing ops missing shape entries ─────────────────────────────────
    # 40 previously-missing ops + 2 with bad propagated entries. Named-tensor
    # variants (.dimname, .dimname_out) use a placeholder dim name so the test
    # reaches the dispatcher (which will surface as failed_lowering for
    # unnamed tensors) rather than sitting in "no shapes registered" limbo.

    # Sparsification (Neuron-side likely no-ops; shapes just need to parse)
    r("aten::_to_sparse", {"self": (256, 256)}, source="builtin")
    r("aten::_to_sparse.out", {"self": (256, 256)}, source="builtin")
    r(
        "aten::_to_sparse.sparse_dim_out",
        {"self": (256, 256), "sparse_dim": 2},
        source="builtin",
    )

    # Sort / nonzero
    r(
        "aten::argsort.stable_out",
        {
            "self": (256, 256),
            "stable": True,
            "dim": -1,
            "descending": False,
            "out": {"shape": (256, 256), "dtype": "long"},
        },
        source="builtin",
    )
    r(
        "aten::nonzero.out",
        {"self": (256, 256), "out": {"shape": (0, 2), "dtype": "long"}},
        source="builtin",
    )

    # Bucketize / searchsorted
    r(
        "aten::bucketize.Tensor",
        {"self": (256, 256), "boundaries": {"shape": (32,), "sorted": True}},
        source="builtin",
    )
    r(
        "aten::bucketize.Scalar",
        {"self": 0.5, "boundaries": {"shape": (32,), "sorted": True}},
        source="builtin",
    )
    r(
        "aten::bucketize.Scalar_out",
        {
            "self": 0.5,
            "boundaries": {"shape": (32,), "sorted": True},
            "out": {"shape": (), "dtype": "long"},
        },
        source="builtin",
    )
    r(
        "aten::searchsorted.Tensor",
        {"sorted_sequence": {"shape": (32,), "sorted": True}, "self": (256, 256)},
        source="builtin",
    )
    r(
        "aten::searchsorted.Scalar",
        {"sorted_sequence": {"shape": (32,), "sorted": True}, "self": 0.5},
        source="builtin",
    )

    # embedding_backward: grad has shape of embedded output (bag, dim)
    r(
        "aten::embedding_backward",
        {
            "grad": (64, 128),
            "indices": {"shape": (64,), "dtype": "long", "low": 0, "high": 256},
            "num_weights": 256,
            "padding_idx": 0,
            "scale_grad_by_freq": False,
            "sparse": False,
        },
        source="builtin",
    )

    # index_fill integer-dim variants only. Dimname variants are not testable:
    # torch.ops.aten.* dispatch rejects named tensors with "NYI: Named tensors
    # are currently unsupported in TorchScript". They need the tensor-method
    # syntax (tensor.index_fill(dim="name", ...)) which our test framework
    # doesn't use. Entries removed; ops stay as no_input_sample on the dashboard.
    r(
        "aten::index_fill.int_Scalar",
        {
            "self": (256, 256),
            "dim": 0,
            "index": {"shape": (32,), "dtype": "long", "low": 0, "high": 256},
            "value": 1.0,
        },
        source="builtin",
    )
    r(
        "aten::index_fill.int_Tensor",
        {
            "self": (256, 256),
            "dim": 0,
            "index": {"shape": (32,), "dtype": "long", "low": 0, "high": 256},
            "value": {"shape": (), "dtype": "float"},
        },
        source="builtin",
    )

    # index_reduce: scatter-reduce variant. reduce ∈ {"prod", "mean", "amax", "amin"}
    for _variant in ("aten::index_reduce", "aten::index_reduce_"):
        r(
            _variant,
            {
                "self": (256, 256),
                "dim": 0,
                "index": {"shape": (64,), "dtype": "long", "low": 0, "high": 256},
                "source": (64, 256),
                "reduce": "prod",
            },
            source="builtin",
        )
    r(
        "aten::index_reduce.out",
        {
            "self": (256, 256),
            "dim": 0,
            "index": {"shape": (64,), "dtype": "long", "low": 0, "high": 256},
            "source": (64, 256),
            "reduce": "prod",
            "out": (256, 256),
        },
        source="builtin",
    )

    # gather / index_select dimname_out variants are not testable via
    # torch.ops.aten.* dispatch (it rejects named tensors with NYI); entries
    # omitted so these stay as no_input_sample rather than producing
    # misleading failed_lowering rows.

    # masked_fill / masked_select
    r(
        "aten::masked_fill.Scalar",
        {
            "self": (256, 256),
            "mask": {"shape": (256, 256), "dtype": "bool"},
            "value": 1.0,
        },
        source="builtin",
    )
    r(
        "aten::masked_fill.Tensor",
        {
            "self": (256, 256),
            "mask": {"shape": (256, 256), "dtype": "bool"},
            "value": {"shape": (), "dtype": "float"},
        },
        source="builtin",
    )
    # Explicit entry: the out= fallback derives its args from masked_fill.Scalar,
    # whose float `value` doesn't type-check against the Tensor_out schema.
    r(
        "aten::masked_fill.Tensor_out",
        {
            "self": (256, 256),
            "mask": {"shape": (256, 256), "dtype": "bool"},
            "value": {"shape": (), "dtype": "float"},
            "out": (256, 256),
        },
        source="builtin",
    )
    r(
        "aten::masked_select.out",
        {
            "self": (256, 256),
            "mask": {"shape": (256, 256), "dtype": "bool"},
            "out": (0,),
        },
        source="builtin",
    )

    # scatter value / value_reduce
    r(
        "aten::scatter.value",
        {
            "self": (256, 256),
            "dim": 0,
            "index": {"shape": (64, 256), "dtype": "long", "low": 0, "high": 256},
            "value": 1.0,
        },
        source="builtin",
    )
    r(
        "aten::scatter.value_reduce",
        {
            "self": (256, 256),
            "dim": 0,
            "index": {"shape": (64, 256), "dtype": "long", "low": 0, "high": 256},
            "value": 1.0,
            "reduce": "add",
        },
        source="builtin",
    )

    # where (scalar / tensor combinations)
    r(
        "aten::where.Scalar",
        {
            "condition": {"shape": (256, 256), "dtype": "bool"},
            "self": 0.0,
            "other": 1.0,
        },
        source="builtin",
    )
    r(
        "aten::where.ScalarOther",
        {
            "condition": {"shape": (256, 256), "dtype": "bool"},
            "self": (256, 256),
            "other": 0.0,
        },
        source="builtin",
    )
    r(
        "aten::where.ScalarSelf",
        {
            "condition": {"shape": (256, 256), "dtype": "bool"},
            "self": 0.0,
            "other": (256, 256),
        },
        source="builtin",
    )

    # count_nonzero (IntList variant)
    r(
        "aten::count_nonzero.dim_IntList",
        {"self": (256, 256), "dim": [0]},
        source="builtin",
    )
    r(
        "aten::count_nonzero.dim_IntList_out",
        {
            "self": (256, 256),
            "dim": [0],
            "out": {"shape": (256,), "dtype": "long"},
        },
        source="builtin",
    )

    # _embedding_bag — 4-output: (output, offset2bag, bag_size, max_indices)
    # mode=0 (sum): max_indices is empty. weight=(N, D), indices=(B_total,),
    # offsets=(num_bags,) monotonic, output=(num_bags, D).
    _eb_weight = (256, 128)
    _eb_indices = {"shape": (1024,), "dtype": "long", "low": 0, "high": 256}
    _eb_offsets = {
        "shape": (32,),
        "dtype": "long",
        "low": 0,
        "high": 1024,
        "sorted": True,
        "starts_at": 0,
    }
    for _eb_op in ("aten::_embedding_bag", "aten::_embedding_bag_forward_only"):
        r(
            _eb_op,
            {
                "weight": _eb_weight,
                "indices": _eb_indices,
                "offsets": _eb_offsets,
                "scale_grad_by_freq": False,
                "mode": 0,
                "sparse": False,
                "per_sample_weights": None,
                "include_last_offset": False,
                "padding_idx": -1,
            },
            source="builtin",
        )
        r(
            _eb_op + ".out",
            {
                "weight": _eb_weight,
                "indices": _eb_indices,
                "offsets": _eb_offsets,
                "scale_grad_by_freq": False,
                "mode": 0,
                "sparse": False,
                "per_sample_weights": None,
                "include_last_offset": False,
                "padding_idx": -1,
                "out0": (32, 128),  # output (num_bags, D)
                "out1": {"shape": (1024,), "dtype": "long"},  # offset2bag
                "out2": {"shape": (32,), "dtype": "long"},  # bag_size
                "out3": {"shape": (0,), "dtype": "long"},  # max_indices (empty for mode=0)
            },
            source="builtin",
        )

    # _unique2 / unique_consecutive — dynamic output sizes
    r(
        "aten::_unique2",
        {
            "self": (256, 256),
            "sorted": True,
            "return_inverse": False,
            "return_counts": False,
        },
        source="builtin",
    )
    r(
        "aten::_unique2.out",
        {
            "self": (256, 256),
            "sorted": True,
            "return_inverse": False,
            "return_counts": False,
            "out0": (0,),
            "out1": {"shape": (0,), "dtype": "long"},
            "out2": {"shape": (0,), "dtype": "long"},
        },
        source="builtin",
    )
    r(
        "aten::unique_consecutive.out",
        {
            "self": (256, 256),
            "return_inverse": False,
            "return_counts": False,
            "dim": None,
            "out0": (0,),
            "out1": {"shape": (0,), "dtype": "long"},
            "out2": {"shape": (0,), "dtype": "long"},
        },
        source="builtin",
    )

    # ── NN ops missing shape entries ───────────────────────────────────────
    # Pool 2D forward/backward. (N, C, H, W) with 2x2 kernel/stride →
    # output (N, C, H/2, W/2).
    _pool2d_in = (2, 64, 32, 32)
    _pool2d_out = (2, 64, 16, 16)
    _pool2d_adapt_out = [7, 7]
    _pool2d_adapt_shape = (2, 64, 7, 7)
    # Pool 3D
    _pool3d_in = (1, 32, 16, 16, 16)
    _pool3d_out = (1, 32, 8, 8, 8)
    _pool3d_adapt_out = [4, 4, 4]
    _pool3d_adapt_shape = (1, 32, 4, 4, 4)
    # Batch norm (input shape uses (N, C, HxW) = (2, 64, 1024), C=64)
    _bn_in = (2, 64, 1024)
    _bn_c = (64,)
    # Group norm (N=1, C=64, HxW=4096, group=8)
    _gn_in = (1, 64, 4096)
    _gn_stats = (1, 8)
    # Layer norm
    _ln_in = (4, 4096, 4096)
    _ln_mean = (4, 4096, 1)
    _ln_weight = (4096,)
    # SDPA
    _sdpa = (1, 32, 2048, 128)
    _sdpa_lse = (1, 32, 2048)

    # Adaptive pool 2D
    r(
        "aten::_adaptive_avg_pool2d",
        {"self": _pool2d_in, "output_size": _pool2d_adapt_out},
        source="builtin",
    )
    r(
        "aten::_adaptive_avg_pool2d.out",
        {"self": _pool2d_in, "output_size": _pool2d_adapt_out},
        source="builtin",
    )
    r(
        "aten::_adaptive_avg_pool2d_backward",
        {"grad_output": _pool2d_adapt_shape, "self": _pool2d_in},
        source="builtin",
    )
    r(
        "aten::_adaptive_avg_pool2d_backward.out",
        {"grad_output": _pool2d_adapt_shape, "self": _pool2d_in},
        source="builtin",
    )
    r(
        "aten::adaptive_max_pool2d",
        {"self": _pool2d_in, "output_size": _pool2d_adapt_out},
        source="builtin",
    )
    r(
        "aten::adaptive_max_pool2d.out",
        {
            "self": _pool2d_in,
            "output_size": _pool2d_adapt_out,
            "out": (0,),
            "indices": {"shape": (0,), "dtype": "long"},
        },
        source="builtin",
    )
    r(
        "aten::adaptive_max_pool2d_backward",
        {
            "grad_output": _pool2d_adapt_shape,
            "self": _pool2d_in,
            "indices": {"shape": _pool2d_adapt_shape, "dtype": "long", "low": 0, "high": 1024},
        },
        source="builtin",
    )
    r(
        "aten::adaptive_max_pool2d_backward.grad_input",
        {
            "grad_output": _pool2d_adapt_shape,
            "self": _pool2d_in,
            "indices": {"shape": _pool2d_adapt_shape, "dtype": "long", "low": 0, "high": 1024},
            "grad_input": (0,),
        },
        source="builtin",
    )

    # Adaptive pool 3D
    r(
        "aten::_adaptive_avg_pool3d",
        {"self": _pool3d_in, "output_size": _pool3d_adapt_out},
        source="builtin",
    )
    r(
        "aten::_adaptive_avg_pool3d.out",
        {"self": _pool3d_in, "output_size": _pool3d_adapt_out},
        source="builtin",
    )
    r(
        "aten::_adaptive_avg_pool3d_backward",
        {"grad_output": _pool3d_adapt_shape, "self": _pool3d_in},
        source="builtin",
    )
    r(
        "aten::_adaptive_avg_pool3d_backward.out",
        {"grad_output": _pool3d_adapt_shape, "self": _pool3d_in},
        source="builtin",
    )
    r(
        "aten::adaptive_max_pool3d",
        {"self": _pool3d_in, "output_size": _pool3d_adapt_out},
        source="builtin",
    )
    r(
        "aten::adaptive_max_pool3d.out",
        {
            "self": _pool3d_in,
            "output_size": _pool3d_adapt_out,
            "out": (0,),
            "indices": {"shape": (0,), "dtype": "long"},
        },
        source="builtin",
    )
    r(
        "aten::adaptive_max_pool3d_backward",
        {
            "grad_output": _pool3d_adapt_shape,
            "self": _pool3d_in,
            "indices": {"shape": _pool3d_adapt_shape, "dtype": "long", "low": 0, "high": 4096},
        },
        source="builtin",
    )
    r(
        "aten::adaptive_max_pool3d_backward.grad_input",
        {
            "grad_output": _pool3d_adapt_shape,
            "self": _pool3d_in,
            "indices": {"shape": _pool3d_adapt_shape, "dtype": "long", "low": 0, "high": 4096},
            "grad_input": (0,),
        },
        source="builtin",
    )

    # Adaptive avg pool 1D
    r("aten::adaptive_avg_pool1d", {"self": (1, 64, 128), "output_size": [8]}, source="builtin")
    r(
        "aten::adaptive_avg_pool1d.out",
        {"self": (1, 64, 128), "output_size": [8]},
        source="builtin",
    )

    # avg_pool2d / 3d
    _avg2d_kwargs = {
        "kernel_size": [2, 2],
        "stride": [2, 2],
        "padding": [0, 0],
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
    }
    r("aten::avg_pool2d", {"self": _pool2d_in, **_avg2d_kwargs}, source="builtin")
    r("aten::avg_pool2d.out", {"self": _pool2d_in, **_avg2d_kwargs}, source="builtin")
    r(
        "aten::avg_pool2d_backward",
        {"grad_output": _pool2d_out, "self": _pool2d_in, **_avg2d_kwargs},
        source="builtin",
    )
    r(
        "aten::avg_pool2d_backward.grad_input",
        {"grad_output": _pool2d_out, "self": _pool2d_in, **_avg2d_kwargs, "grad_input": (0,)},
        source="builtin",
    )

    _avg3d_kwargs = {
        "kernel_size": [2, 2, 2],
        "stride": [2, 2, 2],
        "padding": [0, 0, 0],
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
    }
    r("aten::avg_pool3d", {"self": _pool3d_in, **_avg3d_kwargs}, source="builtin")
    r("aten::avg_pool3d.out", {"self": _pool3d_in, **_avg3d_kwargs}, source="builtin")
    r(
        "aten::avg_pool3d_backward",
        {"grad_output": _pool3d_out, "self": _pool3d_in, **_avg3d_kwargs},
        source="builtin",
    )
    r(
        "aten::avg_pool3d_backward.grad_input",
        {"grad_output": _pool3d_out, "self": _pool3d_in, **_avg3d_kwargs, "grad_input": (0,)},
        source="builtin",
    )

    # max_pool2d / 3d with indices
    _max2d_kwargs = {
        "kernel_size": [2, 2],
        "stride": [2, 2],
        "padding": [0, 0],
        "dilation": [1, 1],
        "ceil_mode": False,
    }
    r("aten::max_pool2d_with_indices", {"self": _pool2d_in, **_max2d_kwargs}, source="builtin")
    r(
        "aten::max_pool2d_with_indices.out",
        {
            "self": _pool2d_in,
            **_max2d_kwargs,
            "out": (0,),
            "indices": {"shape": (0,), "dtype": "long"},
        },
        source="builtin",
    )
    r(
        "aten::max_pool2d_with_indices_backward",
        {
            "grad_output": _pool2d_out,
            "self": _pool2d_in,
            **_max2d_kwargs,
            "indices": {"shape": _pool2d_out, "dtype": "long", "low": 0, "high": 1024},
        },
        source="builtin",
    )

    _max3d_kwargs = {
        "kernel_size": [2, 2, 2],
        "stride": [2, 2, 2],
        "padding": [0, 0, 0],
        "dilation": [1, 1, 1],
        "ceil_mode": False,
    }
    r("aten::max_pool3d_with_indices", {"self": _pool3d_in, **_max3d_kwargs}, source="builtin")
    r(
        "aten::max_pool3d_with_indices.out",
        {
            "self": _pool3d_in,
            **_max3d_kwargs,
            "out": (0,),
            "indices": {"shape": (0,), "dtype": "long"},
        },
        source="builtin",
    )
    r(
        "aten::max_pool3d_with_indices_backward",
        {
            "grad_output": _pool3d_out,
            "self": _pool3d_in,
            **_max3d_kwargs,
            "indices": {"shape": _pool3d_out, "dtype": "long", "low": 0, "high": 4096},
        },
        source="builtin",
    )
    r(
        "aten::max_pool3d_with_indices_backward.grad_input",
        {
            "grad_output": _pool3d_out,
            "self": _pool3d_in,
            **_max3d_kwargs,
            "indices": {"shape": _pool3d_out, "dtype": "long", "low": 0, "high": 4096},
            "grad_input": (0,),
        },
        source="builtin",
    )

    # Fractional max pool: random_samples is (N, C, dim) where dim=2 for 2D, 3 for 3D
    r(
        "aten::fractional_max_pool2d.output",
        {
            "self": _pool2d_in,
            "kernel_size": [2, 2],
            "output_size": [16, 16],
            "random_samples": (2, 64, 2),
            "output": (0,),
            "indices": {"shape": (0,), "dtype": "long"},
        },
        source="builtin",
    )
    r(
        "aten::fractional_max_pool3d.output",
        {
            "self": _pool3d_in,
            "kernel_size": [2, 2, 2],
            "output_size": [8, 8, 8],
            "random_samples": (1, 32, 3),
            "output": (0,),
            "indices": {"shape": (0,), "dtype": "long"},
        },
        source="builtin",
    )

    # Upsample 2D
    _ups2d_in = (1, 64, 32, 32)
    _ups2d_out = [64, 64]
    # NOTE: write-alias kwargs (out=/grad_input=) are intentionally omitted.
    # NeuronOpInfo._with_out_kwargs probes the correct output shape via
    # FakeTensorMode and allocates the tensor. A placeholder like out=(0,) is a
    # valid eager input (PyTorch resizes it) but graph-breaks under
    # torch.compile(fullgraph=True) ("Shape mismatch with out= tensor variant"),
    # so leaving the slot empty is required for the compile-coverage path.
    r(
        "aten::upsample_bicubic2d.out",
        {
            "self": _ups2d_in,
            "output_size": _ups2d_out,
            "align_corners": False,
            "scales_h": None,
            "scales_w": None,
        },
        source="builtin",
    )
    r(
        "aten::upsample_bilinear2d.out",
        {
            "self": _ups2d_in,
            "output_size": _ups2d_out,
            "align_corners": False,
            "scales_h": None,
            "scales_w": None,
        },
        source="builtin",
    )
    r(
        "aten::upsample_bilinear2d.vec_out",
        {
            "input": _ups2d_in,
            "output_size": _ups2d_out,
            "align_corners": False,
            "scale_factors": None,
        },
        source="builtin",
    )
    r(
        "aten::_upsample_bilinear2d_aa.vec",
        {
            "input": _ups2d_in,
            "output_size": _ups2d_out,
            "align_corners": False,
            "scale_factors": None,
        },
        source="builtin",
    )
    r(
        "aten::_upsample_bilinear2d_aa_backward",
        {
            "grad_output": (1, 64, 64, 64),
            "output_size": _ups2d_out,
            "input_size": [1, 64, 32, 32],
            "align_corners": False,
            "scales_h": None,
            "scales_w": None,
        },
        source="builtin",
    )
    r(
        "aten::_upsample_bilinear2d_aa_backward.grad_input",
        {
            "grad_output": (1, 64, 64, 64),
            "output_size": _ups2d_out,
            "input_size": [1, 64, 32, 32],
            "align_corners": False,
            "scales_h": None,
            "scales_w": None,
        },
        source="builtin",
    )
    r(
        "aten::upsample_nearest2d.out",
        {
            "self": _ups2d_in,
            "output_size": _ups2d_out,
            "scales_h": None,
            "scales_w": None,
        },
        source="builtin",
    )
    r(
        "aten::upsample_nearest2d.vec_out",
        {"input": _ups2d_in, "output_size": _ups2d_out, "scale_factors": None},
        source="builtin",
    )
    r(
        "aten::upsample_nearest1d_backward",
        {
            "grad_output": (1, 64, 256),
            "output_size": [256],
            "input_size": [1, 64, 128],
            "scales": None,
        },
        source="builtin",
    )
    r(
        "aten::upsample_nearest1d_backward.grad_input",
        {
            "grad_output": (1, 64, 256),
            "output_size": [256],
            "input_size": [1, 64, 128],
            "scales": None,
        },
        source="builtin",
    )
    r(
        "aten::_upsample_nearest_exact1d_backward",
        {
            "grad_output": (1, 64, 256),
            "output_size": [256],
            "input_size": [1, 64, 128],
            "scales": None,
        },
        source="builtin",
    )
    r(
        "aten::_upsample_nearest_exact1d_backward.grad_input",
        {
            "grad_output": (1, 64, 256),
            "output_size": [256],
            "input_size": [1, 64, 128],
            "scales": None,
        },
        source="builtin",
    )
    r(
        "aten::upsample_linear1d_backward",
        {
            "grad_output": (1, 64, 256),
            "output_size": [256],
            "input_size": [1, 64, 128],
            "align_corners": False,
            "scales": None,
        },
        source="builtin",
    )
    r(
        "aten::upsample_linear1d_backward.grad_input",
        {
            "grad_output": (1, 64, 256),
            "output_size": [256],
            "input_size": [1, 64, 128],
            "align_corners": False,
            "scales": None,
        },
        source="builtin",
    )
    r(
        "aten::_upsample_nearest_exact2d_backward",
        {
            "grad_output": (1, 64, 64, 64),
            "output_size": _ups2d_out,
            "input_size": [1, 64, 32, 32],
            "scales_h": None,
            "scales_w": None,
        },
        source="builtin",
    )
    r(
        "aten::_upsample_nearest_exact2d_backward.grad_input",
        {
            "grad_output": (1, 64, 64, 64),
            "output_size": _ups2d_out,
            "input_size": [1, 64, 32, 32],
            "scales_h": None,
            "scales_w": None,
        },
        source="builtin",
    )
    r(
        "aten::upsample_bilinear2d_backward",
        {
            "grad_output": (1, 64, 64, 64),
            "output_size": _ups2d_out,
            "input_size": [1, 64, 32, 32],
            "align_corners": False,
            "scales_h": None,
            "scales_w": None,
        },
        source="builtin",
    )
    r(
        "aten::upsample_bilinear2d_backward.grad_input",
        {
            "grad_output": (1, 64, 64, 64),
            "output_size": _ups2d_out,
            "input_size": [1, 64, 32, 32],
            "align_corners": False,
            "scales_h": None,
            "scales_w": None,
        },
        source="builtin",
    )
    r(
        "aten::upsample_bicubic2d_backward",
        {
            "grad_output": (1, 64, 64, 64),
            "output_size": _ups2d_out,
            "input_size": [1, 64, 32, 32],
            "align_corners": False,
            "scales_h": None,
            "scales_w": None,
        },
        source="builtin",
    )
    r(
        "aten::upsample_bicubic2d_backward.grad_input",
        {
            "grad_output": (1, 64, 64, 64),
            "output_size": _ups2d_out,
            "input_size": [1, 64, 32, 32],
            "align_corners": False,
            "scales_h": None,
            "scales_w": None,
        },
        source="builtin",
    )
    r(
        "aten::upsample_nearest2d_backward",
        {
            "grad_output": (1, 64, 64, 64),
            "output_size": _ups2d_out,
            "input_size": [1, 64, 32, 32],
            "scales_h": None,
            "scales_w": None,
        },
        source="builtin",
    )
    r(
        "aten::upsample_nearest2d_backward.grad_input",
        {
            "grad_output": (1, 64, 64, 64),
            "output_size": _ups2d_out,
            "input_size": [1, 64, 32, 32],
            "scales_h": None,
            "scales_w": None,
        },
        source="builtin",
    )

    # Upsample 1D
    r(
        "aten::upsample_linear1d.out",
        {
            "self": (1, 64, 128),
            "output_size": [256],
            "align_corners": False,
            "scales": None,
        },
        source="builtin",
    )
    r(
        "aten::upsample_nearest1d.out",
        {"self": (1, 64, 128), "output_size": [256], "scales": None},
        source="builtin",
    )
    r(
        "aten::_upsample_nearest_exact1d.out",
        {"self": (1, 64, 128), "output_size": [256], "scales": None},
        source="builtin",
    )

    # Upsample 3D
    r(
        "aten::upsample_trilinear3d.out",
        {
            "self": (1, 16, 16, 16, 16),
            "output_size": [32, 32, 32],
            "align_corners": False,
            "scales_d": None,
            "scales_h": None,
            "scales_w": None,
        },
        source="builtin",
    )

    # BatchNorm variants. input (N, C, HxW) = (2, 64, 1024); C=64.
    r(
        "aten::_native_batch_norm_legit.no_stats",
        {
            "input": _bn_in,
            "weight": _bn_c,
            "bias": _bn_c,
            "training": True,
            "momentum": 0.1,
            "eps": 1e-5,
        },
        source="builtin",
    )
    r(
        "aten::_native_batch_norm_legit_no_training.out",
        {
            "input": _bn_in,
            "weight": _bn_c,
            "bias": _bn_c,
            "running_mean": _bn_c,
            "running_var": _bn_c,
            "momentum": 0.1,
            "eps": 1e-5,
            "out0": _bn_in,  # output has input shape
            "out1": (0,),  # save_mean (empty in inference)
            "out2": (0,),  # save_invstd (empty in inference)
        },
        source="builtin",
    )
    r(
        "aten::batch_norm_backward",
        {
            "grad_out": _bn_in,
            "input": _bn_in,
            "weight": _bn_c,
            "running_mean": _bn_c,
            "running_var": _bn_c,
            "save_mean": _bn_c,
            "save_var": _bn_c,
            "update": False,
            "eps": 1e-5,
            "output_mask": [True, True, True],
            "reserve": (0,),  # cudnn reserve; empty on CPU
        },
        source="builtin",
    )
    r(
        "aten::batch_norm_backward_reduce.out",
        {
            "grad_out": _bn_in,
            "input": _bn_in,
            "mean": _bn_c,
            "invstd": _bn_c,
            "weight": _bn_c,
            "input_g": True,
            "weight_g": True,
            "bias_g": True,
            "out0": _bn_c,
            "out1": _bn_c,
            "out2": _bn_c,
            "out3": _bn_c,
        },
        source="builtin",
    )
    r(
        "aten::batch_norm_gather_stats_with_counts.out",
        {
            "input": _bn_in,
            "mean": _bn_c,
            "invstd": _bn_c,
            "running_mean": _bn_c,
            "running_var": _bn_c,
            "momentum": 0.1,
            "eps": 1e-5,
            "counts": (1,),
            "out0": _bn_c,
            "out1": _bn_c,
        },
        source="builtin",
    )
    r(
        "aten::batch_norm_stats.out",
        {"input": _bn_in, "eps": 1e-5, "out0": _bn_c, "out1": _bn_c},
        source="builtin",
    )
    r(
        "aten::_weight_norm_interface.out",
        {"v": (8192, 8192), "g": (8192, 1), "dim": 0, "out0": (8192, 8192), "out1": (8192, 1)},
        source="manual",
    )
    # CPU correctness is skipped for this op (see _OPS_SKIP_CPU_CORRECTNESS): the
    # harness fills saved_norms with independent randn, violating
    # saved_norms==‖saved_v‖; the 1/norm**3 term then diverges. Correctness is
    # covered by upstream gradcheck + unit/e2e tests; lowering verified exact vs
    # CPU with consistent inputs.
    r(
        "aten::_weight_norm_interface_backward.out",
        {
            "grad_w": (8192, 8192),
            "saved_v": (8192, 8192),
            "saved_g": (8192, 1),
            "saved_norms": (8192, 1),
            "dim": 0,
            "out0": (8192, 8192),
            "out1": (8192, 1),
        },
        source="manual",
    )
    r(
        "aten::native_batch_norm_backward.out",
        {
            "grad_out": _bn_in,
            "input": _bn_in,
            "weight": _bn_c,
            "running_mean": _bn_c,
            "running_var": _bn_c,
            "save_mean": _bn_c,
            "save_invstd": _bn_c,
            "train": True,
            "eps": 1e-5,
            "output_mask": [True, True, True],
            "out0": _bn_in,
            "out1": _bn_c,
            "out2": _bn_c,
        },
        source="builtin",
    )

    # GroupNorm / LayerNorm .out backward
    r(
        "aten::native_group_norm.out",
        {
            "input": _gn_in,
            "weight": (64,),
            "bias": (64,),
            "N": 1,
            "C": 64,
            "HxW": 4096,
            "group": 8,
            "eps": 1e-5,
            "out0": _gn_in,
            "out1": _gn_stats,
            "out2": _gn_stats,
        },
        source="builtin",
    )
    r(
        "aten::native_group_norm_backward.out",
        {
            "grad_out": _gn_in,
            "input": _gn_in,
            "mean": _gn_stats,
            "rstd": _gn_stats,
            "weight": (64,),
            "N": 1,
            "C": 64,
            "HxW": 4096,
            "group": 8,
            "output_mask": [True, True, True],
            "out0": _gn_in,
            "out1": (64,),
            "out2": (64,),
        },
        source="builtin",
    )
    r(
        "aten::native_layer_norm.out",
        {
            "input": _ln_in,
            "normalized_shape": [4096],
            "weight": _ln_weight,
            "bias": _ln_weight,
            "eps": 1e-5,
            "out0": _ln_in,
            "out1": _ln_mean,
            "out2": _ln_mean,
        },
        source="builtin",
    )
    r(
        "aten::native_layer_norm_backward.out",
        {
            "grad_out": _ln_in,
            "input": _ln_in,
            "normalized_shape": [4096],
            "mean": _ln_mean,
            "rstd": _ln_mean,
            "weight": _ln_weight,
            "bias": _ln_weight,
            "output_mask": [True, True, True],
            "out0": _ln_in,
            "out1": _ln_weight,
            "out2": _ln_weight,
        },
        source="builtin",
    )

    # RNN cells (GRU has 3 gates, LSTM has 4). batch=32, hidden=512.
    r(
        "aten::_thnn_fused_gru_cell",
        {
            "input_gates": (32, 3 * 512),
            "hidden_gates": (32, 3 * 512),
            "hx": (32, 512),
            "input_bias": None,
            "hidden_bias": None,
        },
        source="builtin",
    )
    r(
        "aten::_thnn_fused_gru_cell_backward",
        {"grad_hy": (32, 512), "workspace": (32, 3 * 512), "has_bias": False},
        source="builtin",
    )
    r(
        "aten::_thnn_fused_lstm_cell",
        {
            "input_gates": (32, 4 * 512),
            "hidden_gates": (32, 4 * 512),
            "cx": (32, 512),
            # PyTorch's meta (torch/_meta_registrations.py::rnn_cell_checkSizes)
            # iterates over ``input_bias``/``hidden_bias`` and dereferences
            # ``.device`` unconditionally, blowing up with ``AttributeError:
            # 'NoneType' object has no attribute 'device'`` when either is
            # None. Passing concrete bias tensors here sidesteps the upstream
            # meta bug so the FakeTensor call succeeds. Verified 2026-07-20.
            "input_bias": (4 * 512,),
            "hidden_bias": (4 * 512,),
        },
        source="builtin",
    )
    r(
        "aten::_thnn_fused_lstm_cell_backward",
        {
            "grad_hy": (32, 512),
            "grad_cy": (32, 512),
            "cx": (32, 512),
            "cy": (32, 512),
            "workspace": (32, 4 * 512),
            "has_bias": False,
        },
        source="builtin",
    )
    r(
        "aten::_thnn_fused_lstm_cell_backward_impl",
        {
            "grad_hy": (32, 512),
            "grad_cy": (32, 512),
            "cx": (32, 512),
            "cy": (32, 512),
            "workspace": (32, 4 * 512),
            "has_bias": False,
        },
        source="builtin",
    )

    # SDPA backward: Q/K/V at positions 1/2/3 (grad_out at 0).
    r(
        "aten::_scaled_dot_product_fused_attention_overrideable_backward",
        {
            "grad_out": _sdpa,
            "query": _sdpa,
            "key": _sdpa,
            "value": _sdpa,
            "attn_bias": None,
            "grad_input_mask": [True, True, True, False],
            "out": _sdpa,
            "logsumexp": _sdpa_lse,
            "cum_seq_q": (0,),
            "cum_seq_k": (0,),
            "max_q": 2048,
            "max_k": 2048,
            "dropout_p": 0.0,
            "is_causal": False,
            "philox_seed": {"shape": (), "dtype": "long"},
            "philox_offset": {"shape": (), "dtype": "long"},
        },
        source="builtin",
    )

    # _trilinear: einsum-like tri-input contraction with expand/sum dims
    r(
        "aten::_trilinear",
        {
            "i1": (8, 4),
            "i2": (8, 4),
            "i3": (8, 4),
            "expand1": [1, 2],
            "expand2": [0, 2],
            "expand3": [0, 1],
            "sumdim": [0],
            "unroll_dim": 1,
        },
        source="builtin",
    )
    r(
        "aten::_trilinear.out",
        {
            "i1": (8, 4),
            "i2": (8, 4),
            "i3": (8, 4),
            "expand1": [1, 2],
            "expand2": [0, 2],
            "expand3": [0, 1],
            "sumdim": [0],
            "unroll_dim": 1,
            "out": (0,),
        },
        source="builtin",
    )

    # ── Shape ops missing shape entries ────────────────────────────────────
    # Factory / window / arange-family ops. Window ops produce a 1D tensor
    # of length `window_length`. Skipped: eye.m_out (already registered),
    # cat.names_out (named-tensor variant, untestable via torch.ops.aten).

    # slice
    r(
        "aten::slice.Tensor",
        {"self": (256, 4096), "dim": 0, "start": 0, "end": 128, "step": 1},
        source="builtin",
    )

    # split family (multi-output)
    r(
        "aten::split.Tensor",
        {"self": (256, 4096), "split_size": 64, "dim": 0},
        source="builtin",
    )
    r(
        "aten::split_with_sizes_copy.out",
        {
            "self": (256, 4096),
            "split_sizes": [64, 128, 64],
            "dim": 0,
            "out": [(64, 4096), (128, 4096), (64, 4096)],
        },
        source="builtin",
    )

    # unfold_backward (reverses unfold: input (256, 4096), unfold dim=1 size=4
    # step=4 → grad_in (256, 1024, 4); backward returns original shape)
    r(
        "aten::unfold_backward",
        {
            "grad_in": (256, 1024, 4),
            "input_sizes": [256, 4096],
            "dim": 1,
            "size": 4,
            "step": 4,
        },
        source="builtin",
    )
    r(
        "aten::unfold_backward.out",
        {
            "grad_in": (256, 1024, 4),
            "input_sizes": [256, 4096],
            "dim": 1,
            "size": 4,
            "step": 4,
            "out": (256, 4096),
        },
        source="builtin",
    )

    # arange family (excluding arange.out which is already registered)
    r("aten::arange", {"end": 128}, source="builtin")
    r(
        "aten::arange.start",
        {"start": 0, "end": 128},
        source="builtin",
    )
    r(
        "aten::arange.start_step",
        {"start": 0, "end": 128, "step": 1},
        source="builtin",
    )
    r(
        "aten::arange.start_out",
        {"start": 0, "end": 128, "step": 1, "out": (128,)},
        source="builtin",
    )

    # range family
    r("aten::range", {"start": 0, "end": 127}, source="builtin")
    r("aten::range.step", {"start": 0, "end": 127, "step": 1}, source="builtin")
    r("aten::range.out", {"start": 0, "end": 127, "step": 1, "out": (128,)}, source="builtin")

    # aten::range.out_ hangs indefinitely on Neuron for all schema-valid inputs
    # (verified: any scalar dtype x any out dtype x size 10 and 128 all hang;
    # same call returns <1 ms on CPU). Registered with empty arg_shapes so
    # (a) Tier-3 propagation from aten::range skips this op, and (b) sample
    # building returns None (start/end missing) → op surfaces as
    # no_input_sample instead of hanging the pytest-xdist worker.
    # TODO: file torch_neuronx bug and restore the entry once fixed.
    r("aten::range.out_", {}, source="builtin")

    # linspace / logspace with mixed Tensor / Scalar endpoints. The Tensor
    # endpoints are 0-d scalar tensors.
    _scalar_t = {"shape": (), "dtype": "float"}
    r(
        "aten::linspace.Scalar_Tensor_out",
        {"start": 0.0, "end": _scalar_t, "steps": 100, "out": (100,)},
        source="builtin",
    )
    r(
        "aten::linspace.Tensor_Scalar_out",
        {"start": _scalar_t, "end": 1.0, "steps": 100, "out": (100,)},
        source="builtin",
    )
    r(
        "aten::linspace.Tensor_Tensor_out",
        {"start": _scalar_t, "end": _scalar_t, "steps": 100, "out": (100,)},
        source="builtin",
    )
    r(
        "aten::logspace.Scalar_Tensor_out",
        {"start": 0.0, "end": _scalar_t, "steps": 100, "base": 10.0, "out": (100,)},
        source="builtin",
    )
    r(
        "aten::logspace.Tensor_Scalar_out",
        {"start": _scalar_t, "end": 1.0, "steps": 100, "base": 10.0, "out": (100,)},
        source="builtin",
    )
    r(
        "aten::logspace.Tensor_Tensor_out",
        {"start": _scalar_t, "end": _scalar_t, "steps": 100, "base": 10.0, "out": (100,)},
        source="builtin",
    )

    # Window functions. Basic, .out, .periodic, .periodic_out variants across
    # 4 window types (bartlett, blackman, hann, hamming).
    _wl = 128
    _win_out_shape = (_wl,)
    for _win in ("bartlett_window", "blackman_window", "hann_window", "hamming_window"):
        r(f"aten::{_win}", {"window_length": _wl}, source="builtin")
        r(f"aten::{_win}.out", {"window_length": _wl, "out": _win_out_shape}, source="builtin")
        r(
            f"aten::{_win}.periodic",
            {"window_length": _wl, "periodic": True},
            source="builtin",
        )
        r(
            f"aten::{_win}.periodic_out",
            {"window_length": _wl, "periodic": True, "out": _win_out_shape},
            source="builtin",
        )

    # hamming_window has extra alpha / alpha_beta variants
    r(
        "aten::hamming_window.periodic_alpha",
        {"window_length": _wl, "periodic": True, "alpha": 0.54},
        source="builtin",
    )
    r(
        "aten::hamming_window.periodic_alpha_out",
        {"window_length": _wl, "periodic": True, "alpha": 0.54, "out": _win_out_shape},
        source="builtin",
    )
    r(
        "aten::hamming_window.periodic_alpha_beta",
        {"window_length": _wl, "periodic": True, "alpha": 0.54, "beta": 0.46},
        source="builtin",
    )
    r(
        "aten::hamming_window.periodic_alpha_beta_out",
        {
            "window_length": _wl,
            "periodic": True,
            "alpha": 0.54,
            "beta": 0.46,
            "out": _win_out_shape,
        },
        source="builtin",
    )

    # kaiser_window has .beta variant
    r("aten::kaiser_window", {"window_length": _wl}, source="builtin")
    r(
        "aten::kaiser_window.out",
        {"window_length": _wl, "out": _win_out_shape},
        source="builtin",
    )
    r(
        "aten::kaiser_window.periodic",
        {"window_length": _wl, "periodic": True},
        source="builtin",
    )
    r(
        "aten::kaiser_window.periodic_out",
        {"window_length": _wl, "periodic": True, "out": _win_out_shape},
        source="builtin",
    )
    r(
        "aten::kaiser_window.beta",
        {"window_length": _wl, "periodic": True, "beta": 12.0},
        source="builtin",
    )
    r(
        "aten::kaiser_window.beta_out",
        {"window_length": _wl, "periodic": True, "beta": 12.0, "out": _win_out_shape},
        source="builtin",
    )

    # ── Random ops missing shape entries ───────────────────────────────────
    # Skipped: rand.generator_out, rand.generator_with_names_out, rand.names_out
    # (already registered).
    _rng_out_f = (256, 4096)
    _rng_out_i = {"shape": (256, 4096), "dtype": "long"}
    _rng_size = [256, 4096]

    # normal variants. For variants where std is a Tensor, use sigmoid to
    # constrain values to [0, 1] (non-negative as the op requires).
    _pos_t = {"shape": (256, 4096), "sigmoid": True}
    r(
        "aten::normal.Tensor_float",
        {"mean": _rng_out_f, "std": 1.0, "generator": None},
        source="builtin",
    )
    r(
        "aten::normal.Tensor_Tensor",
        {"mean": _rng_out_f, "std": _pos_t, "generator": None},
        source="builtin",
    )
    r(
        "aten::normal.float_Tensor",
        {"mean": 0.0, "std": _pos_t, "generator": None},
        source="builtin",
    )
    r(
        "aten::normal.float_float",
        {"mean": 0.0, "std": 1.0, "size": _rng_size, "generator": None},
        source="builtin",
    )
    r(
        "aten::normal.float_float_out",
        {"mean": 0.0, "std": 1.0, "size": _rng_size, "generator": None, "out": _rng_out_f},
        source="builtin",
    )
    r(
        "aten::normal.out",
        {"self": _rng_out_f, "mean": 0.0, "std": 1.0, "generator": None, "out": _rng_out_f},
        source="builtin",
    )

    # rand generator-variant (plus override for .out variants which
    # propagation left without out= kwargs)
    r("aten::rand.generator", {"size": _rng_size, "generator": None}, source="builtin")
    r(
        "aten::rand.generator_out",
        {"size": _rng_size, "generator": None, "out": _rng_out_f},
        source="builtin",
    )
    r(
        "aten::rand.names_out",
        {"size": _rng_size, "names": None, "out": _rng_out_f},
        source="builtin",
    )
    r(
        "aten::rand.generator_with_names_out",
        {"size": _rng_size, "generator": None, "names": None, "out": _rng_out_f},
        source="builtin",
    )

    # randint variants. dtype defaults to long/int; out tensor is a long tensor.
    r(
        "aten::randint.generator",
        {"high": 100, "size": _rng_size, "generator": None},
        source="builtin",
    )
    r(
        "aten::randint.generator_out",
        {"high": 100, "size": _rng_size, "generator": None, "out": _rng_out_i},
        source="builtin",
    )
    r("aten::randint.low", {"low": 0, "high": 100, "size": _rng_size}, source="builtin")
    r(
        "aten::randint.low_generator",
        {"low": 0, "high": 100, "size": _rng_size, "generator": None},
        source="builtin",
    )
    r(
        "aten::randint.low_generator_out",
        {"low": 0, "high": 100, "size": _rng_size, "generator": None, "out": _rng_out_i},
        source="builtin",
    )
    r(
        "aten::randint.low_out",
        {"low": 0, "high": 100, "size": _rng_size, "out": _rng_out_i},
        source="builtin",
    )

    # randn variants (names_out / generator_with_names_out pass names=None)
    r("aten::randn.generator", {"size": _rng_size, "generator": None}, source="builtin")
    r(
        "aten::randn.generator_out",
        {"size": _rng_size, "generator": None, "out": _rng_out_f},
        source="builtin",
    )
    r(
        "aten::randn.names_out",
        {"size": _rng_size, "names": None, "out": _rng_out_f},
        source="builtin",
    )
    r(
        "aten::randn.generator_with_names_out",
        {"size": _rng_size, "generator": None, "names": None, "out": _rng_out_f},
        source="builtin",
    )

    # ── Elementwise ops missing shape entries ──────────────────────────────
    # Skipped: named-tensor variants (.names_dim_*, kthvalue.dimname_out —
    # untestable via torch.ops.aten); ops already registered via propagation
    # or earlier entries (clamp.out, _fused_adam_, linear_backward.out, etc.)
    _ew_in = (256, 4096)

    # Unary ops (Tensor → Tensor)
    r("aten::_conj_physical", {"self": _ew_in}, source="builtin")
    r("aten::_conj_physical.out", {"self": _ew_in, "out": _ew_in}, source="builtin")
    r("aten::angle", {"self": _ew_in}, source="builtin")
    r("aten::angle.out", {"self": _ew_in, "out": _ew_in}, source="builtin")
    r("aten::max", {"self": _ew_in}, source="builtin")
    r("aten::max.unary_out", {"self": _ew_in}, source="builtin")
    r("aten::min", {"self": _ew_in}, source="builtin")
    r("aten::min.unary_out", {"self": _ew_in}, source="builtin")
    r("aten::sym_storage_offset", {"self": _ew_in}, source="builtin")

    # clamp with optional scalar bounds (clamp.out already registered)
    r("aten::clamp", {"self": _ew_in, "min": -1.0, "max": 1.0}, source="builtin")

    # Binary element-at-a-time ops
    r("aten::complex", {"real": _ew_in, "imag": _ew_in}, source="builtin")
    # Note: complex.out / polar.out require ComplexFloat out tensors, which
    # the current spec language doesn't express cleanly. Left unregistered;
    # propagation will supply a Float out and the dashboard will show the
    # honest "Expected ComplexFloat" error.
    r(
        "aten::polar",
        {"abs": {"shape": (256, 4096), "sigmoid": True}, "angle": _ew_in},
        source="builtin",
    )

    # max/min binary variants
    r("aten::max.other", {"self": _ew_in, "other": _ew_in}, source="builtin")
    r("aten::max.out", {"self": _ew_in, "other": _ew_in, "out": _ew_in}, source="builtin")
    r("aten::min.other", {"self": _ew_in, "other": _ew_in}, source="builtin")
    r("aten::min.out", {"self": _ew_in, "other": _ew_in, "out": _ew_in}, source="builtin")

    # isin variants (tensor ∈ test_elements)
    r(
        "aten::isin.Tensor_Tensor",
        {"elements": _ew_in, "test_elements": (64,), "assume_unique": False, "invert": False},
        source="builtin",
    )
    r(
        "aten::isin.Scalar_Tensor",
        {"element": 0.5, "test_elements": (64,), "assume_unique": False, "invert": False},
        source="builtin",
    )
    r(
        "aten::isin.Tensor_Scalar",
        {"elements": _ew_in, "test_element": 0.5, "assume_unique": False, "invert": False},
        source="builtin",
    )

    # bitwise Scalar_Tensor_out — int dtype for both sides
    _int_in = {"shape": (256, 4096), "dtype": "int"}
    for _op in (
        "aten::bitwise_and.Scalar_Tensor_out",
        "aten::bitwise_or.Scalar_Tensor_out",
        "aten::bitwise_xor.Scalar_Tensor_out",
    ):
        r(_op, {"self": 5, "other": _int_in, "out": _int_in}, source="builtin")

    # float_power.Scalar (base scalar, tensor exponent). .out already registered.
    r("aten::float_power.Scalar", {"self": 2.0, "exponent": _ew_in}, source="builtin")

    # frexp — multi-output (mantissa + exponent). exponent dtype is int32.
    r("aten::frexp.Tensor", {"self": _ew_in}, source="builtin")
    r(
        "aten::frexp.Tensor_out",
        {"self": _ew_in, "mantissa": _ew_in, "exponent": {"shape": (256, 4096), "dtype": "int"}},
        source="builtin",
    )

    # median (int-dim variant) — returns (values, indices)
    r("aten::median.dim", {"self": _ew_in, "dim": 1, "keepdim": False}, source="builtin")

    # addbmm_ in-place: (B,M,K) @ (B,K,N) summed across B → self (M,N)
    r(
        "aten::addbmm_",
        {"self": (32, 64), "batch1": (4, 32, 16), "batch2": (4, 16, 64)},
        source="builtin",
    )

    # _fused_adam_.tensor_lr / _fused_adamw_.tensor_lr — same structure as the
    # existing _fused_adam_ entries but with lr as a 0-d tensor. max_exp_avg_sqs
    # must be empty when amsgrad=False.
    _adam_list = [{"shape": (256, 4096)}] * 4
    for _op in ("aten::_fused_adam_.tensor_lr", "aten::_fused_adamw_.tensor_lr"):
        r(
            _op,
            {
                "self": _adam_list,
                "grads": _adam_list,
                "exp_avgs": _adam_list,
                "exp_avg_sqs": _adam_list,
                "max_exp_avg_sqs": [],  # empty when amsgrad=False
                "state_steps": [{"shape": (), "dtype": "float"}] * 4,
                "lr": {"shape": (), "dtype": "float"},  # 0-d tensor lr
                "beta1": 0.9,
                "beta2": 0.999,
                "weight_decay": 0.01,
                "eps": 1e-8,
                "amsgrad": False,
                "maximize": False,
                "grad_scale": None,
                "found_inf": None,
            },
            source="builtin",
        )

    # Image / channel ops. Standard N,C,H,W
    # channel_shuffle / pixel_shuffle / pixel_unshuffle
    r("aten::channel_shuffle", {"self": (1, 16, 8, 8), "groups": 4}, source="builtin")
    r(
        "aten::channel_shuffle.out",
        {"self": (1, 16, 8, 8), "groups": 4, "out": (1, 16, 8, 8)},
        source="builtin",
    )
    r("aten::pixel_shuffle", {"self": (1, 16, 8, 8), "upscale_factor": 2}, source="builtin")
    r(
        "aten::pixel_shuffle.out",
        {"self": (1, 16, 8, 8), "upscale_factor": 2, "out": (1, 4, 16, 16)},
        source="builtin",
    )
    r("aten::pixel_unshuffle", {"self": (1, 4, 16, 16), "downscale_factor": 2}, source="builtin")
    r(
        "aten::pixel_unshuffle.out",
        {"self": (1, 4, 16, 16), "downscale_factor": 2, "out": (1, 16, 8, 8)},
        source="builtin",
    )

    # im2col: input (N, C, H, W) unfolded to (N, C*kH*kW, L)
    r(
        "aten::im2col",
        {
            "self": (1, 3, 16, 16),
            "kernel_size": [3, 3],
            "dilation": [1, 1],
            "padding": [1, 1],
            "stride": [1, 1],
        },
        source="builtin",
    )
    r(
        "aten::im2col.out",
        {
            "self": (1, 3, 16, 16),
            "kernel_size": [3, 3],
            "dilation": [1, 1],
            "padding": [1, 1],
            "stride": [1, 1],
            "out": (0,),
        },
        source="builtin",
    )

    # max_unpool2d / 3d: self + indices (long) + output_size
    r(
        "aten::max_unpool2d",
        {
            "self": (1, 1, 4, 4),
            "indices": {"shape": (1, 1, 4, 4), "dtype": "long", "low": 0, "high": 64},
            "output_size": [8, 8],
        },
        source="builtin",
    )
    r(
        "aten::max_unpool2d.out",
        {
            "self": (1, 1, 4, 4),
            "indices": {"shape": (1, 1, 4, 4), "dtype": "long", "low": 0, "high": 64},
            "output_size": [8, 8],
            "out": (1, 1, 8, 8),
        },
        source="builtin",
    )
    r(
        "aten::max_unpool3d",
        {
            "self": (1, 1, 2, 2, 2),
            "indices": {"shape": (1, 1, 2, 2, 2), "dtype": "long", "low": 0, "high": 64},
            "output_size": [4, 4, 4],
            "stride": [2, 2, 2],
            "padding": [0, 0, 0],
        },
        source="builtin",
    )
    r(
        "aten::max_unpool3d.out",
        {
            "self": (1, 1, 2, 2, 2),
            "indices": {"shape": (1, 1, 2, 2, 2), "dtype": "long", "low": 0, "high": 64},
            "output_size": [4, 4, 4],
            "stride": [2, 2, 2],
            "padding": [0, 0, 0],
            "out": (1, 1, 4, 4, 4),
        },
        source="builtin",
    )

    # polygamma: n (int) + self (tensor)
    r("aten::polygamma", {"n": 1, "self": _ew_in}, source="builtin")

    # factory .names_out variants — pass names=None since names is Optional
    r(
        "aten::full.names_out",
        {"size": [256, 4096], "fill_value": 1.0, "names": None, "out": _ew_in},
        source="builtin",
    )
    r("aten::ones.names_out", {"size": [256, 4096], "names": None, "out": _ew_in}, source="builtin")
    r(
        "aten::zeros.names_out",
        {"size": [256, 4096], "names": None, "out": _ew_in},
        source="builtin",
    )

    # ── Overrides for pre-existing broken propagated entries ──────────────
    # These ops were already in the registry via propagation but the
    # propagated entries were missing required write-alias / kwargs, causing
    # the op to fail with "missing value for argument 'X'" at dispatch.

    # clamp.out needs out kwarg
    r("aten::clamp.out", {"self": _ew_in, "min": -1.0, "max": 1.0, "out": _ew_in}, source="builtin")
    # clamp.Tensor_out: min/max are Optional[Tensor], pass tensors
    r(
        "aten::clamp.Tensor_out",
        {"self": _ew_in, "min": _ew_in, "max": _ew_in, "out": _ew_in},
        source="builtin",
    )
    # float_power.Scalar_out needs out
    r(
        "aten::float_power.Scalar_out",
        {
            "self": 2.0,
            "exponent": _ew_in,
            "out": {"shape": (256, 4096), "dtype": "double"},
        },  # float_power returns f64
        source="builtin",
    )
    # lerp.Tensor_out: weight is a Tensor (not float)
    r(
        "aten::lerp.Tensor_out",
        {"self": _ew_in, "end": _ew_in, "weight": _ew_in, "out": _ew_in},
        source="builtin",
    )
    # linear_backward.out: all three write-aliases
    r(
        "aten::linear_backward.out",
        {
            "self": (256, 4096),
            "grad_output": (256, 4096),
            "weight": (4096, 4096),
            "output_mask": [True, True, True],
            "out0": (256, 4096),
            "out1": (4096, 4096),
            "out2": (4096,),
        },
        source="builtin",
    )
    # max.dim_max / min.dim_min: multi-out (value + index)
    r(
        "aten::max.dim_max",
        {
            "self": _ew_in,
            "dim": 1,
            "keepdim": False,
            "max": (256,),
            "max_values": {"shape": (256,), "dtype": "long"},
        },
        source="builtin",
    )
    r(
        "aten::min.dim_min",
        {
            "self": _ew_in,
            "dim": 1,
            "keepdim": False,
            "min": (256,),
            "min_indices": {"shape": (256,), "dtype": "long"},
        },
        source="builtin",
    )
    # _fused_adam_ / _fused_adamw_: max_exp_avg_sqs must be empty when amsgrad=False
    _adam_list_4 = [{"shape": (256, 4096)}] * 4
    _adam_steps_4 = [{"shape": (), "dtype": "float"}] * 4
    for _op in ("aten::_fused_adam_", "aten::_fused_adamw_"):
        r(
            _op,
            {
                "self": _adam_list_4,
                "grads": _adam_list_4,
                "exp_avgs": _adam_list_4,
                "exp_avg_sqs": _adam_list_4,
                "max_exp_avg_sqs": [],  # empty when amsgrad=False
                "state_steps": _adam_steps_4,
                "lr": 0.001,
                "beta1": 0.9,
                "beta2": 0.999,
                "weight_decay": 0.01,
                "eps": 1e-8,
                "amsgrad": False,
                "maximize": False,
                "grad_scale": None,
                "found_inf": None,
            },
            source="builtin",
        )

    # ── Reduction ops missing shape entries ───────────────────────────────
    # Common input shape. Most reductions along dim=1 with keepdim=False.
    _red_in = (256, 4096)

    # ── std / var with correction (explicit Bessel term) ──────────────
    for _op in ("aten::std.correction", "aten::var.correction"):
        r(_op, {"self": _red_in, "dim": [1], "correction": 1, "keepdim": False}, source="builtin")
    for _op in (
        "aten::std_mean.correction",
        "aten::var_mean.correction",
        "aten::std_mean.dim",
        "aten::var_mean.dim",
    ):
        r(_op, {"self": _red_in, "dim": [1], "correction": 1, "keepdim": False}, source="builtin")
    r(
        "aten::std_mean.correction_out",
        {
            "self": _red_in,
            "dim": [1],
            "correction": 1,
            "keepdim": False,
            "out0": (256,),
            "out1": (256,),
        },
        source="builtin",
    )
    r(
        "aten::var_mean.correction_out",
        {
            "self": _red_in,
            "dim": [1],
            "correction": 1,
            "keepdim": False,
            "out0": (256,),
            "out1": (256,),
        },
        source="builtin",
    )

    # std / var .dim and .out (unbiased signature)
    for _op in ("aten::std.dim", "aten::var.dim"):
        r(_op, {"self": _red_in, "dim": [1], "unbiased": True, "keepdim": False}, source="builtin")
    for _op in ("aten::std.out", "aten::var.out"):
        r(
            _op,
            {"self": _red_in, "dim": [1], "unbiased": True, "keepdim": False, "out": (256,)},
            source="builtin",
        )

    # ── all / any ─────────────────────────────────────────────────────
    # Base full-reduction variants: input only → 0-d bool scalar output.
    for _op in ("aten::all", "aten::any"):
        r(_op, {"self": _red_in}, source="builtin")
    for _op in ("aten::all.all_out", "aten::any.all_out"):
        r(_op, {"self": _red_in, "out": {"shape": (), "dtype": "bool"}}, source="builtin")
    for _op in ("aten::all.dim", "aten::any.dim"):
        r(_op, {"self": _red_in, "dim": 1, "keepdim": False}, source="builtin")
    for _op in ("aten::all.dims", "aten::any.dims"):
        r(_op, {"self": _red_in, "dim": [1], "keepdim": False}, source="builtin")
    for _op in ("aten::all.out", "aten::any.out"):
        r(
            _op,
            {
                "self": _red_in,
                "dim": 1,
                "keepdim": False,
                "out": {"shape": (256,), "dtype": "bool"},
            },
            source="builtin",
        )

    # ── cumsum ────────────────────────────────────────────────────────
    # Prefix-sum along dim; output same shape as input.
    r("aten::cumsum", {"self": _red_in, "dim": 1}, source="builtin")
    r("aten::cumsum.out", {"self": _red_in, "dim": 1, "out": _red_in}, source="builtin")

    # ── mean / sum / prod out-variants ────────────────────────────────
    r(
        "aten::mean.out",
        {"self": _red_in, "dim": [1], "keepdim": False, "out": (256,)},
        source="builtin",
    )
    r(
        "aten::sum.IntList_out",
        {"self": _red_in, "dim": [1], "keepdim": False, "out": (256,)},
        source="builtin",
    )
    r(
        "aten::prod.dim_int",
        {"self": _red_in, "dim": 1, "keepdim": False},
        source="builtin",
    )

    # ── softmax / log_softmax / _safe_softmax (int variants) ──────────
    for _op in ("aten::softmax.int", "aten::log_softmax.int", "aten::_safe_softmax"):
        r(_op, {"self": _red_in, "dim": 1, "dtype": None}, source="builtin")

    # ── mode / topk (multi-output with values + indices) ─────────────
    r(
        "aten::mode.values",
        {
            "self": _red_in,
            "dim": 1,
            "keepdim": False,
            "values": (0,),
            "indices": {"shape": (0,), "dtype": "long"},
        },
        source="builtin",
    )
    r(
        "aten::topk.values",
        {
            "self": _red_in,
            "k": 5,
            "dim": 1,
            "largest": True,
            "sorted": True,
            "values": (0,),
            "indices": {"shape": (0,), "dtype": "long"},
        },
        source="builtin",
    )

    # ── Cumulative ops ────────────────────────────────────────────────
    r(
        "aten::cumprod_backward",
        {"grad": _red_in, "input": _red_in, "dim": 1, "output": _red_in},
        source="builtin",
    )
    # _cummax_helper / _cummin_helper write into values/indices, return ()
    for _op in ("aten::_cummax_helper", "aten::_cummin_helper"):
        r(
            _op,
            {
                "self": _red_in,
                "values": _red_in,
                "indices": {"shape": _red_in, "dtype": "long"},
                "dim": 1,
            },
            source="builtin",
        )

    # ── pdist / cdist ─────────────────────────────────────────────────
    # pdist: pairwise distances of N points in D-dim; output (N*(N-1)/2,)
    r("aten::_pdist_forward", {"self": (32, 16), "p": 2.0}, source="builtin")
    r("aten::_pdist_forward.out", {"self": (32, 16), "p": 2.0}, source="builtin")
    # cdist: pairwise distance between two point sets; x1 (B, P, D), x2 (B, R, D)
    r(
        "aten::_cdist_forward",
        {"x1": (4, 32, 16), "x2": (4, 32, 16), "p": 2.0, "compute_mode": None},
        source="builtin",
    )
    r(
        "aten::_cdist_forward.out",
        {"x1": (4, 32, 16), "x2": (4, 32, 16), "p": 2.0, "compute_mode": None},
        source="builtin",
    )

    # ── segment_reduce / backward ─────────────────────────────────────
    # lengths=[32]*8 sums to 256 = data.shape[0]. randint(32, 33) yields all 32s.
    _sr_data = (256, 4096)
    _sr_lengths = {"shape": (8,), "dtype": "long", "low": 32, "high": 33}
    r(
        "aten::segment_reduce",
        {
            "data": _sr_data,
            "reduce": "sum",
            "lengths": _sr_lengths,
            "indices": None,
            "offsets": None,
            "axis": 0,
            "unsafe": False,
            "initial": None,
        },
        source="builtin",
    )
    r(
        "aten::segment_reduce.out",
        {
            "data": _sr_data,
            "reduce": "sum",
            "lengths": _sr_lengths,
            "indices": None,
            "offsets": None,
            "axis": 0,
            "unsafe": False,
            "initial": None,
            "out": (8, 4096),
        },
        source="builtin",
    )
    r(
        "aten::_segment_reduce_backward",
        {
            "grad": (8, 4096),
            "output": (8, 4096),
            "data": _sr_data,
            "reduce": "sum",
            "lengths": _sr_lengths,
            "offsets": None,
            "axis": 0,
            "initial": None,
        },
        source="builtin",
    )
    r(
        "aten::_segment_reduce_backward.out",
        {
            "grad": (8, 4096),
            "output": (8, 4096),
            "data": _sr_data,
            "reduce": "sum",
            "lengths": _sr_lengths,
            "offsets": None,
            "axis": 0,
            "initial": None,
            "out": _sr_data,
        },
        source="builtin",
    )
    r(
        "aten::index_fill.int_Tensor_out",
        {
            "self": (256, 256),
            "dim": 0,
            "index": {"shape": (32,), "dtype": "long", "low": 0, "high": 256},
            "value": {"shape": (), "dtype": "float"},
            "out": (256, 256),
        },
        source="builtin",
    )
    r(
        "aten::gather.out",
        {
            "self": (256, 4096),
            "dim": 0,
            "index": {"shape": (64, 4096), "dtype": "long", "low": 0, "high": 256},
            "out": (64, 4096),
        },
        source="builtin",
    )
    r(
        "aten::histc",
        {"self": (4096,), "bins": 100, "min": 0.0, "max": 1.0},
        source="builtin",
    )
    r(
        "aten::baddbmm.dtype_out",
        {
            "self": (8, 256, 64),  # (B, M, N)
            "batch1": (8, 256, 128),  # (B, M, K)
            "batch2": (8, 128, 64),  # (B, K, N)
            "out_dtype": "float",
            "beta": 1.0,
            "alpha": 1.0,
            "out": (8, 256, 64),
        },
        source="builtin",
    )
    r(
        "aten::_batch_norm_with_update.out",
        {
            "input": (2, 64, 1024),
            "weight": (64,),
            "bias": (64,),
            "running_mean": (64,),
            "running_var": (64,),
            "momentum": 0.1,
            "eps": 1e-5,
            "out": (2, 64, 1024),
            "save_mean": (64,),
            "save_invstd": (64,),
            "reserve": (0,),
        },
        source="builtin",
    )

    # ── Dimname / names_out variants ──────────────────────────────────
    # These require named tensors, which our test dispatches (torch.ops.aten.*)
    # doesn't support — it rejects with "NYI: Named tensors are currently
    # unsupported in TorchScript". Only the Python tensor-method syntax
    # (tensor.sum(dim="name", ...)) reaches these ops. Entries omitted so
    # ops stay as no_input_sample (honest: "can't test with current framework")
    # rather than producing misleading failed_lowering rows.

    # Accuracy smoke test subsets, opt-in via --traced-model. An unknown name fails loud.
    for _name in smoke_subsets:
        try:
            _register_model = SMOKE_SUBSET_SHAPES[_name]
        except KeyError:
            raise KeyError(
                f"unknown --traced-model {_name!r}; available: {sorted(SMOKE_SUBSET_SHAPES)}"
            ) from None
        _register_model(r)


# ── Accuracy smoke test subset ────────────────────────────────────────────────
#
# This is a smoke test subset. It is not a model onboarding framework. Do not add a model
# here to get accuracy coverage for that model. The first subset comes from a Qwen3-235B
# EP64/TP4 capture, You can extend the subset, but the intent is one small frozen subset.
#
# Captured end-to-end from NeuronTorchTitan ``qwen3_235b_6layers_ep64_nle1_tp4_neuron_ep`` on
# trn3-dev1.48xlarge: SelectiveAC, seq_len 2048.
#
# **Op mode** property of ops type
#
#   eager_only   -- seen in eager dispatch, never in a post-AOT graph.
#   compile_only -- post-AOT graphs only, i.e. introduced by AOT. Either a decomposition
#                   product (``aten::sigmoid`` from ``aten::silu``; ``aten::mm`` and
#                   ``aten::t`` in backward graphs from ``aten::linear_backward``) or a
#                   functionalization rewrite (``aten::index_put`` is the out-of-place form of
#                   eager-only ``aten::index_put_``).
#   both_modes   -- seen in eager dispatch **and** in a post-AOT graph.
#
# **Capture path** property of test *shape*
#
#   e.g. eager / compiled / both, which path the test shape is captured.
#
# These two axes are not interchangeable.
#
# **Entry count.** Entries include 328 distinct shapes (trimmed down w/ top 10 per op
# of largest tensor sizes of collected).
#
# **No float dtype keys.** Int/bool specs are pinned because an index or mask operand sampled
# as bfloat16 is rejected by the op.
#
# **Gap.** Known zero-shape op: ``aten::bmm``. Its captured entries are global shapes over the
# 512Mi cap, we still register this op, report result will say ``no_input_sample``, so we can
# follow up fix instead of silently ignore.
QWEN3_235B_EP64_TP4 = "qwen3_235b_ep64_tp4"

QWEN3_235B_EP64_TP4_OP_MODE = {
    "aten::_foreach_copy_": "eager_only",
    "aten::_foreach_mul_.Tensor": "eager_only",
    "aten::_foreach_norm.Scalar": "eager_only",
    "aten::_fused_rms_norm": "both_modes",
    "aten::_fused_rms_norm_backward": "eager_only",
    "aten::_grouped_mm": "eager_only",
    "aten::_scaled_dot_product_fused_attention_overrideable": "eager_only",
    "aten::_scaled_dot_product_fused_attention_overrideable_backward": "eager_only",
    "aten::_softmax": "both_modes",
    "aten::_softmax_backward_data": "eager_only",
    "aten::_to_copy": "both_modes",
    "aten::_unsafe_view": "compile_only",
    "aten::add.Tensor": "both_modes",
    "aten::add_.Tensor": "eager_only",
    "aten::all": "eager_only",
    "aten::amax": "both_modes",
    "aten::any.dim": "both_modes",
    "aten::arange": "both_modes",
    "aten::as_strided": "eager_only",
    "aten::bitwise_and.Tensor": "eager_only",
    "aten::bitwise_or.Tensor": "both_modes",
    "aten::bmm": "compile_only",
    "aten::cat": "both_modes",
    "aten::clamp": "eager_only",
    "aten::clone": "both_modes",
    "aten::constant_pad_nd": "compile_only",
    "aten::copy_": "both_modes",
    "aten::cumsum": "both_modes",
    "aten::detach": "both_modes",
    "aten::div.Tensor": "both_modes",
    "aten::embedding": "eager_only",
    "aten::embedding_dense_backward": "eager_only",
    "aten::eq.Tensor": "both_modes",
    "aten::exp": "both_modes",
    "aten::expand": "both_modes",
    "aten::floor_divide": "both_modes",
    "aten::full": "both_modes",
    "aten::gather": "both_modes",
    "aten::ge.Scalar": "both_modes",
    "aten::index.Tensor": "both_modes",
    "aten::index_put": "compile_only",
    "aten::index_put_": "eager_only",
    "aten::le.Scalar": "both_modes",
    "aten::le.Tensor": "both_modes",
    "aten::linalg_vector_norm": "eager_only",
    "aten::linear": "both_modes",
    "aten::linear_backward": "eager_only",
    "aten::log": "both_modes",
    "aten::lt.Scalar": "both_modes",
    "aten::maximum": "compile_only",
    "aten::mean": "eager_only",
    "aten::minimum": "compile_only",
    "aten::mm": "compile_only",
    "aten::mul.Tensor": "both_modes",
    "aten::ne.Scalar": "both_modes",
    "aten::neg": "both_modes",
    "aten::new_zeros": "both_modes",
    "aten::one_hot": "both_modes",
    "aten::ones_like": "eager_only",
    "aten::pow.Tensor_Scalar": "eager_only",
    "aten::pow_.Scalar": "eager_only",
    "aten::reciprocal": "eager_only",
    "aten::remainder.Scalar": "both_modes",
    "aten::rsub.Scalar": "compile_only",
    "aten::scalar_tensor": "eager_only",
    "aten::scatter_add": "both_modes",
    "aten::select.int": "eager_only",
    "aten::sigmoid": "compile_only",
    "aten::sign": "eager_only",
    "aten::silu": "eager_only",
    "aten::silu_backward": "eager_only",
    "aten::slice.Tensor": "both_modes",
    "aten::slice_backward": "eager_only",
    "aten::split.Tensor": "eager_only",
    "aten::split_with_sizes": "eager_only",
    "aten::squeeze.dim": "both_modes",
    "aten::stack": "eager_only",
    "aten::sub.Tensor": "both_modes",
    "aten::sum": "both_modes",
    "aten::sum.dim_IntList": "both_modes",
    "aten::t": "compile_only",
    "aten::topk": "both_modes",
    "aten::transpose.int": "both_modes",
    "aten::unsqueeze": "both_modes",
    "aten::view": "both_modes",
    "aten::where.self": "both_modes",
    "aten::zero_": "eager_only",
    "aten::zeros": "both_modes",
    "aten::zeros_like": "both_modes",
}

QWEN3_235B_EP64_TP4_OPS = frozenset(QWEN3_235B_EP64_TP4_OP_MODE)

# Ops registered w/ no shape: e.g. ``aten::bmm``'s entry traced shapes are over the
# 512Mi cap. Define it as no shape input to track ops w/ input_sample generation issues;
# the report verdict shows ``no_input_sample`` instead of silently skipping.
QWEN3_235B_EP64_TP4_NO_SHAPE_OPS = frozenset(
    {
        "aten::bmm",
    }
)


# Register op shapes. The "path" routes the entry result as a row in generated csv
# file. e.g. "both" will make that row write to both "op_accuracy_eager_*.csv" and
# "op_accuracy_compiled_*.csv".
def _register_qwen3_235b_ep64_tp4(r):
    def t(op, arg_shapes, count=1, path="eager"):
        paths = ("eager", "compiled") if path == "both" else (path,)
        r(op, arg_shapes, source=QWEN3_235B_EP64_TP4, count=count, paths=paths)

    t(
        "aten::_foreach_copy_",
        {
            "self": [(589824,), (524288,), (8,), (8,), (16384,), (256,), (256,)],
            "src": [(589824,), (524288,), (8,), (8,), (16384,), (256,), (256,)],
        },
        count=60,
        path="eager",
    )
    t("aten::_foreach_copy_", {"self": [(9723904,)], "src": [(9723904,)]}, count=9, path="eager")
    t("aten::_foreach_copy_", {"self": [(256,)], "src": [(256,)]}, count=6, path="eager")
    t(
        "aten::_foreach_mul_.Tensor",
        {
            "other": (),
            "self": [
                (64, 1536, 4096),
                (64, 4096, 1536),
                (64, 1536, 4096),
                (64, 1536, 4096),
                (64, 4096, 1536),
                (64, 1536, 4096),
                (64, 1536, 4096),
                (64, 4096, 1536),
                (64, 1536, 4096),
                (64, 1536, 4096),
                (64, 4096, 1536),
                (64, 1536, 4096),
                (64, 1536, 4096),
                (64, 4096, 1536),
                (64, 1536, 4096),
                (64, 1536, 4096),
                (64, 4096, 1536),
                (64, 1536, 4096),
            ],
        },
        count=3,
        path="eager",
    )
    t(
        "aten::_foreach_norm.Scalar",
        {
            "ord": 2.0,
            "self": [
                (64, 1536, 4096),
                (64, 4096, 1536),
                (64, 1536, 4096),
                (64, 1536, 4096),
                (64, 4096, 1536),
                (64, 1536, 4096),
                (64, 1536, 4096),
                (64, 4096, 1536),
                (64, 1536, 4096),
                (64, 1536, 4096),
                (64, 4096, 1536),
                (64, 1536, 4096),
                (64, 1536, 4096),
                (64, 4096, 1536),
                (64, 1536, 4096),
                (64, 1536, 4096),
                (64, 4096, 1536),
                (64, 1536, 4096),
            ],
        },
        count=3,
        path="eager",
    )
    t(
        "aten::_fused_rms_norm",
        {"eps": 1e-06, "input": (1, 2048, 4096), "normalized_shape": (4096,), "weight": (4096,)},
        count=75,
        path="eager",
    )
    t(
        "aten::_fused_rms_norm",
        {"eps": 1e-06, "input": (1, 2048, 4, 128), "normalized_shape": (128,), "weight": (128,)},
        count=36,
        path="eager",
    )
    t(
        "aten::_fused_rms_norm",
        {"eps": 1e-06, "input": (1, 2048, 64, 128), "normalized_shape": (128,), "weight": (128,)},
        count=36,
        path="eager",
    )
    t(
        "aten::_fused_rms_norm",
        {"eps": 1e-06, "input": (1, 512, 4096), "normalized_shape": (4096,), "weight": (4096,)},
        count=1,
        path="compiled",
    )
    t(
        "aten::_fused_rms_norm",
        {"eps": 1e-06, "input": (1, 2048, 16, 128), "normalized_shape": (128,), "weight": (128,)},
        count=1,
        path="compiled",
    )
    t(
        "aten::_fused_rms_norm",
        {"eps": 1e-06, "input": (1, 2048, 1, 128), "normalized_shape": (128,), "weight": (128,)},
        count=1,
        path="compiled",
    )
    t(
        "aten::_fused_rms_norm_backward",
        {
            "grad_out": (1, 2048, 4096),
            "input": (1, 2048, 4096),
            "normalized_shape": (4096,),
            "output_mask": (True, True),
            "rstd": (1, 2048, 1),
            "weight": (4096,),
        },
        count=39,
        path="eager",
    )
    t(
        "aten::_fused_rms_norm_backward",
        {
            "grad_out": (1, 2048, 4, 128),
            "input": (1, 2048, 4, 128),
            "normalized_shape": (128,),
            "output_mask": (True, True),
            "rstd": (1, 2048, 4, 1),
            "weight": (128,),
        },
        count=18,
        path="eager",
    )
    t(
        "aten::_fused_rms_norm_backward",
        {
            "grad_out": (1, 2048, 64, 128),
            "input": (1, 2048, 64, 128),
            "normalized_shape": (128,),
            "output_mask": (True, True),
            "rstd": (1, 2048, 64, 1),
            "weight": (128,),
        },
        count=18,
        path="eager",
    )
    t(
        "aten::_grouped_mm",
        {
            "mat2": (1, 4096, 1536),
            "offs": {"dtype": "int32", "high": 2, "low": 1, "multiple_of": 32768, "shape": (1,)},
            "self": (32768, 4096),
        },
        count=90,
        path="eager",
    )
    t(
        "aten::_grouped_mm",
        {
            "mat2": (1, 1536, 4096),
            "offs": {"dtype": "int32", "high": 2, "low": 1, "multiple_of": 32768, "shape": (1,)},
            "self": (32768, 1536),
        },
        count=72,
        path="eager",
    )
    t(
        "aten::_grouped_mm",
        {
            "mat2": (32768, 4096),
            "offs": {"dtype": "int32", "high": 2, "low": 1, "multiple_of": 1536, "shape": (1,)},
            "self": (1536, 32768),
        },
        count=36,
        path="eager",
    )
    t(
        "aten::_grouped_mm",
        {
            "mat2": (32768, 1536),
            "offs": {"dtype": "int32", "high": 2, "low": 1, "multiple_of": 4096, "shape": (1,)},
            "self": (4096, 32768),
        },
        count=18,
        path="eager",
    )
    t(
        "aten::_scaled_dot_product_fused_attention_overrideable",
        {
            "attn_bias": None,
            "dropout_p": 0.0,
            "is_causal": True,
            "key": (1, 1, 2048, 128),
            "query": (1, 16, 2048, 128),
            "scale": 0.08838834764831845,
            "value": (1, 1, 2048, 128),
        },
        count=18,
        path="eager",
    )
    t(
        "aten::_scaled_dot_product_fused_attention_overrideable_backward",
        {
            "attn_bias": None,
            "cum_seq_k": None,
            "cum_seq_q": None,
            "dropout_p": 0.0,
            "grad_input_mask": (True, True, True, False),
            "grad_out": (1, 16, 2048, 128),
            "is_causal": True,
            "key": (1, 1, 2048, 128),
            "logsumexp": (1, 16, 2048),
            "max_k": 2048,
            "max_q": 2048,
            "out": (1, 16, 2048, 128),
            "philox_offset": {"dtype": "int32", "shape": ()},
            "philox_seed": {"dtype": "int32", "shape": ()},
            "query": (1, 16, 2048, 128),
            "scale": 0.08838834764831845,
            "value": (1, 1, 2048, 128),
        },
        count=18,
        path="eager",
    )
    t(
        "aten::_softmax",
        {"dim": -1, "half_to_float": False, "self": (1, 2048, 64)},
        count=36,
        path="eager",
    )
    t(
        "aten::_softmax",
        {"dim": -1, "half_to_float": False, "self": (1, 512, 64)},
        count=1,
        path="compiled",
    )
    t(
        "aten::_softmax_backward_data",
        {"dim": -1, "grad_output": (1, 2048, 64), "input_dtype": 6, "output": (1, 2048, 64)},
        count=18,
        path="eager",
    )
    t("aten::_to_copy", {"dtype": 15, "self": (1, 2048, 64, 128)}, count=3456, path="eager")
    t("aten::_to_copy", {"dtype": 15, "self": (256, 37984)}, count=1536, path="both")
    t("aten::_to_copy", {"dtype": 6, "self": (1, 2048, 64, 128)}, count=1188, path="eager")
    t("aten::_to_copy", {"dtype": 6, "self": (4096, 4096)}, count=1188, path="both")
    t("aten::_to_copy", {"dtype": 15, "self": (4096, 4096)}, count=1188, path="both")
    t("aten::_to_copy", {"dtype": 15, "self": (1, 2048, 4096)}, count=1155, path="eager")
    t("aten::_to_copy", {"dtype": 6, "self": (1, 2048, 4096)}, count=36, path="eager")
    t("aten::_to_copy", {"dtype": 6, "self": (256, 37984)}, count=24, path="both")
    t("aten::_to_copy", {"dtype": 6, "self": (32768, 1536)}, count=1, path="compiled")
    t("aten::_to_copy", {"dtype": 15, "self": (32768, 1536)}, count=1, path="compiled")
    t(
        "aten::_unsafe_view",
        {"self": (1, 16, 16, 128), "size": (1, 16, 2048)},
        count=1,
        path="compiled",
    )
    t("aten::_unsafe_view", {"self": (512, 4096), "size": (1, 512, 4096)}, count=1, path="compiled")
    t(
        "aten::_unsafe_view",
        {"self": (2048, 2048), "size": (1, 2048, 2048)},
        count=1,
        path="compiled",
    )
    t(
        "aten::_unsafe_view",
        {"self": (2048, 4096), "size": (1, 2048, 4096)},
        count=1,
        path="compiled",
    )
    t(
        "aten::add.Tensor",
        {"other": (1, 2048, 4096), "self": (1, 2048, 4096)},
        count=126,
        path="eager",
    )
    t(
        "aten::add.Tensor",
        {"other": (1, 2048, 64, 128), "self": (1, 2048, 64, 128)},
        count=72,
        path="eager",
    )
    t(
        "aten::add.Tensor",
        {"other": (1, 2048, 4, 128), "self": (1, 2048, 4, 128)},
        count=72,
        path="eager",
    )
    t("aten::add.Tensor", {"other": (64,), "self": (1, 2048, 64)}, count=36, path="eager")
    t("aten::add.Tensor", {"other": (256, 37984), "self": (256, 37984)}, count=24, path="both")
    t("aten::add.Tensor", {"other": (32768, 4096), "self": (32768, 4096)}, count=18, path="both")
    t("aten::add.Tensor", {"other": (), "self": (32768, 1536)}, count=1, path="compiled")
    t(
        "aten::add.Tensor",
        {"other": (1, 2048, 16, 128), "self": (1, 2048, 16, 128)},
        count=1,
        path="compiled",
    )
    t(
        "aten::add.Tensor",
        {"other": (1, 512, 4096), "self": (1, 512, 4096)},
        count=1,
        path="compiled",
    )
    t(
        "aten::add.Tensor",
        {"other": (1, 2048, 1, 128), "self": (1, 2048, 1, 128)},
        count=1,
        path="compiled",
    )
    t(
        "aten::add_.Tensor",
        {"other": {"dtype": "int64", "shape": (64,)}, "self": (64,)},
        count=36,
        path="eager",
    )
    t("aten::add_.Tensor", {"other": (64,), "self": (64,)}, count=18, path="eager")
    t(
        "aten::add_.Tensor",
        {"other": {"dtype": "int64", "shape": ()}, "self": {"dtype": "int64", "shape": ()}},
        count=3,
        path="eager",
    )
    t("aten::all", {"self": {"dtype": "bool", "shape": (1, 2048)}}, count=36, path="eager")
    t("aten::amax", {"dim": (-1,), "keepdim": True, "self": (256, 37984)}, count=24, path="both")
    t(
        "aten::any.dim",
        {"dim": -2, "self": {"dtype": "bool", "shape": (1, 2048, 8, 64)}},
        count=36,
        path="eager",
    )
    t(
        "aten::any.dim",
        {"dim": -2, "self": {"dtype": "bool", "shape": (1, 512, 8, 64)}},
        count=1,
        path="compiled",
    )
    t("aten::arange", {"end": 4096, "pin_memory": False}, count=4608, path="both")
    t("aten::arange", {"dtype": 3, "end": 4096, "pin_memory": False}, count=2304, path="both")
    t("aten::arange", {"dtype": 4, "end": 64, "pin_memory": False}, count=2304, path="both")
    t("aten::arange", {"end": 256, "pin_memory": False}, count=1536, path="both")
    t(
        "aten::as_strided",
        {
            "self": (1, 2048, 16, 128),
            "size": (1, 2048, 18, 128),
            "storage_offset": 0,
            "stride": (4718592, 2304, 128, 1),
        },
        count=36,
        path="eager",
    )
    t(
        "aten::as_strided",
        {
            "self": (18874368,),
            "size": (1, 1536, 4096),
            "storage_offset": 0,
            "stride": (6291456, 4096, 1),
        },
        count=18,
        path="eager",
    )
    t(
        "aten::as_strided",
        {
            "self": (18874368,),
            "size": (1, 1536, 4096),
            "storage_offset": 12582912,
            "stride": (6291456, 4096, 1),
        },
        count=18,
        path="eager",
    )
    t(
        "aten::as_strided",
        {
            "self": (18874368,),
            "size": (1, 4096, 1536),
            "storage_offset": 6291456,
            "stride": (6291456, 1536, 1),
        },
        count=18,
        path="eager",
    )
    t(
        "aten::as_strided",
        {
            "self": (6291456,),
            "size": (1, 1536, 4096),
            "storage_offset": 0,
            "stride": (6291456, 4096, 1),
        },
        count=12,
        path="eager",
    )
    t(
        "aten::as_strided",
        {"self": (9723904,), "size": (2374, 4096), "storage_offset": 0, "stride": (4096, 1)},
        count=6,
        path="eager",
    )
    t(
        "aten::as_strided",
        {"self": (9437184,), "size": (2304, 4096), "storage_offset": 0, "stride": (4096, 1)},
        count=6,
        path="eager",
    )
    t(
        "aten::as_strided",
        {"self": (8388608,), "size": (4096, 2048), "storage_offset": 0, "stride": (2048, 1)},
        count=6,
        path="eager",
    )
    t(
        "aten::as_strided",
        {
            "self": (6291456,),
            "size": (1, 4096, 1536),
            "storage_offset": 0,
            "stride": (6291456, 1536, 1),
        },
        count=6,
        path="eager",
    )
    t(
        "aten::as_strided",
        {"self": (155582464,), "size": (37984, 4096), "storage_offset": 0, "stride": (4096, 1)},
        count=2,
        path="eager",
    )
    t(
        "aten::bitwise_and.Tensor",
        {
            "other": {"dtype": "bool", "shape": (1, 2048)},
            "self": {"dtype": "bool", "shape": (1, 2048)},
        },
        count=3,
        path="eager",
    )
    t(
        "aten::bitwise_or.Tensor",
        {"other": {"dtype": "bool", "shape": (256,)}, "self": {"dtype": "bool", "shape": (256,)}},
        count=48,
        path="both",
    )
    t(
        "aten::cat",
        {"tensors": [(1, 512, 4096), (1, 512, 4096), (1, 512, 4096), (1, 512, 4096)]},
        count=96,
        path="both",
    )
    t(
        "aten::cat",
        {"dim": -1, "tensors": [(1, 2048, 4, 64), (1, 2048, 4, 64)]},
        count=36,
        path="eager",
    )
    t(
        "aten::cat",
        {"dim": -1, "tensors": [(1, 2048, 64, 64), (1, 2048, 64, 64)]},
        count=36,
        path="eager",
    )
    t(
        "aten::cat",
        {"dim": 3, "tensors": [(1, 2048, 1, 16, 128), (1, 2048, 1, 1, 128), (1, 2048, 1, 1, 128)]},
        count=18,
        path="both",
    )
    t(
        "aten::cat",
        {"tensors": [(1, 64), (1, 64), (1, 64), (1, 64), (1, 64), (1, 64)]},
        count=3,
        path="eager",
    )
    t(
        "aten::cat",
        {"dim": -1, "tensors": [(1, 2048, 16, 64), (1, 2048, 16, 64)]},
        count=1,
        path="compiled",
    )
    t(
        "aten::cat",
        {"dim": -1, "tensors": [(1, 2048, 1, 64), (1, 2048, 1, 64)]},
        count=1,
        path="compiled",
    )
    t("aten::clamp", {"max": 1.0, "min": None, "self": ()}, count=6, path="eager")
    t(
        "aten::clamp",
        {"max": 37983, "min": 0, "self": {"dtype": "int64", "shape": (1, 2048)}},
        count=3,
        path="eager",
    )
    t("aten::clone", {"self": (512, 8, 4096)}, count=36, path="both")
    t("aten::clone", {"self": (1, 2048, 16, 128)}, count=1, path="compiled")
    t("aten::clone", {"self": (1, 2048, 1, 128)}, count=1, path="compiled")
    t("aten::clone", {"self": (1, 16, 16, 128)}, count=1, path="compiled")
    t("aten::clone", {"self": (1, 512, 64)}, count=1, path="compiled")
    t("aten::constant_pad_nd", {"pad": (64, 0), "self": (1, 2048, 1, 64)}, count=1, path="compiled")
    t("aten::constant_pad_nd", {"pad": (0, 64), "self": (1, 2048, 1, 64)}, count=1, path="compiled")
    t(
        "aten::constant_pad_nd",
        {"pad": (64, 0), "self": (1, 2048, 16, 64)},
        count=1,
        path="compiled",
    )
    t(
        "aten::constant_pad_nd",
        {"pad": (0, 64), "self": (1, 2048, 16, 64)},
        count=1,
        path="compiled",
    )
    t("aten::copy_", {"self": (6291456,), "src": (6291456,)}, count=108, path="eager")
    t("aten::copy_", {"self": (1, 256, 4096), "src": (1, 256, 4096)}, count=24, path="eager")
    t("aten::copy_", {"self": (18874368,), "src": (18874368,)}, count=18, path="eager")
    t("aten::copy_", {"self": (64,), "src": (64,)}, count=1, path="compiled")
    t(
        "aten::cumsum",
        {"dim": 0, "dtype": 3, "self": {"dtype": "int32", "shape": (1,)}},
        count=36,
        path="both",
    )
    t(
        "aten::cumsum",
        {"dim": 0, "self": {"dtype": "int64", "shape": (4096, 64)}},
        count=36,
        path="both",
    )
    t("aten::detach", {"self": (32768, 1536)}, count=72, path="eager")
    t("aten::detach", {"self": (9216, 4096)}, count=60, path="eager")
    t("aten::detach", {"self": (4096, 8192)}, count=60, path="eager")
    t("aten::detach", {"self": (64, 1536, 4096)}, count=48, path="eager")
    t("aten::detach", {"self": (32768, 4096)}, count=36, path="eager")
    t("aten::detach", {"self": (1, 2048, 64, 128)}, count=36, path="eager")
    t("aten::detach", {"self": (1, 2048, 8192)}, count=36, path="eager")
    t("aten::detach", {"self": (64, 4096, 1536)}, count=24, path="eager")
    t("aten::detach", {"self": (1, 2048, 9216)}, count=18, path="eager")
    t("aten::detach", {"self": (4096, 4096)}, count=18, path="eager")
    t("aten::div.Tensor", {"other": (1, 2048, 1), "self": (1, 2048, 8)}, count=90, path="eager")
    t("aten::div.Tensor", {"other": (), "self": ()}, count=48, path="eager")
    t("aten::div.Tensor", {"other": (), "self": (18874368,)}, count=18, path="eager")
    t(
        "aten::div.Tensor",
        {"other": {"dtype": "int64", "shape": ()}, "self": ()},
        count=3,
        path="eager",
    )
    t("aten::div.Tensor", {"other": (1, 512, 1), "self": (1, 512, 8)}, count=1, path="compiled")
    t("aten::div.Tensor", {"other": (), "self": (1, 512, 4096)}, count=1, path="compiled")
    t("aten::div.Tensor", {"other": (), "self": (1, 2048, 1, 128)}, count=1, path="compiled")
    t("aten::div.Tensor", {"other": (), "self": (1, 2048, 16, 128)}, count=1, path="compiled")
    t(
        "aten::embedding",
        {"indices": {"dtype": "int64", "shape": (1, 2048)}, "weight": (37984, 4096)},
        count=3,
        path="eager",
    )
    t(
        "aten::embedding_dense_backward",
        {
            "grad_output": (1, 2048, 4096),
            "indices": {"dtype": "int64", "shape": (1, 2048)},
            "num_weights": 37984,
            "padding_idx": -1,
            "scale_grad_by_freq": False,
        },
        count=3,
        path="eager",
    )
    t(
        "aten::eq.Tensor",
        {
            "other": {"dtype": "int64", "shape": (64,)},
            "self": {"dtype": "int64", "shape": (1, 2048, 8, 1)},
        },
        count=36,
        path="eager",
    )
    t(
        "aten::eq.Tensor",
        {
            "other": {"dtype": "int64", "shape": (64,)},
            "self": {"dtype": "int64", "shape": (1, 512, 8, 1)},
        },
        count=1,
        path="compiled",
    )
    t("aten::exp", {"self": (256, 37984)}, count=48, path="both")
    t("aten::expand", {"self": (512, 1, 4096), "size": (512, 8, 4096)}, count=36, path="both")
    t(
        "aten::expand",
        {"self": {"dtype": "int64", "shape": (4096, 1)}, "size": (-1, 4096)},
        count=36,
        path="both",
    )
    t("aten::expand", {"self": (1, 2048, 1), "size": (1, 2048, 8)}, count=18, path="eager")
    t(
        "aten::expand",
        {"self": (256, 128, 4096), "size": (256, 128, 4096)},
        count=1,
        path="compiled",
    )
    t(
        "aten::expand",
        {"self": (256, 128, 1536), "size": (256, 128, 1536)},
        count=1,
        path="compiled",
    )
    t("aten::expand", {"self": (1, 512, 1), "size": (1, 512, 8)}, count=1, path="compiled")
    t(
        "aten::floor_divide",
        {"other": (), "self": {"dtype": "int64", "shape": (4096,)}},
        count=72,
        path="both",
    )
    t("aten::floor_divide", {"other": (), "self": (1, 64)}, count=36, path="both")
    t("aten::floor_divide", {"other": (), "self": (64,)}, count=18, path="eager")
    t(
        "aten::floor_divide",
        {"other": (), "self": {"dtype": "int32", "shape": (1, 1)}},
        count=1,
        path="compiled",
    )
    t(
        "aten::full",
        {"dtype": 3, "fill_value": 32768, "pin_memory": False, "size": (1,)},
        count=2304,
        path="both",
    )
    t(
        "aten::full",
        {"dtype": 3, "fill_value": 4096, "pin_memory": False, "size": (4096, 64)},
        count=2304,
        path="both",
    )
    t(
        "aten::full",
        {"dtype": 4, "fill_value": 0, "pin_memory": False, "size": ()},
        count=64,
        path="compiled",
    )
    t(
        "aten::full",
        {"dtype": 6, "fill_value": 0, "pin_memory": False, "size": ()},
        count=64,
        path="compiled",
    )
    t(
        "aten::gather",
        {"dim": -1, "index": {"dtype": "int64", "shape": (1, 2048, 8)}, "self": (1, 2048, 64)},
        count=36,
        path="eager",
    )
    t(
        "aten::gather",
        {"dim": -1, "index": {"dtype": "int64", "shape": (256, 1)}, "self": (256, 37984)},
        count=24,
        path="both",
    )
    t(
        "aten::gather",
        {"dim": 0, "index": {"dtype": "int64", "shape": (4096, 4096)}, "self": (2048, 4096)},
        count=18,
        path="both",
    )
    t(
        "aten::gather",
        {"dim": -1, "index": {"dtype": "int64", "shape": (1, 512, 8)}, "self": (1, 512, 64)},
        count=1,
        path="compiled",
    )
    t(
        "aten::ge.Scalar",
        {"other": 37984, "self": {"dtype": "int64", "shape": (256,)}},
        count=48,
        path="both",
    )
    t(
        "aten::ge.Scalar",
        {"other": 75968, "self": {"dtype": "int64", "shape": (256,)}},
        count=48,
        path="both",
    )
    t(
        "aten::ge.Scalar",
        {"other": 113952, "self": {"dtype": "int64", "shape": (256,)}},
        count=48,
        path="both",
    )
    t(
        "aten::ge.Scalar",
        {"other": 151936, "self": {"dtype": "int64", "shape": (256,)}},
        count=48,
        path="both",
    )
    t(
        "aten::ge.Scalar",
        {"other": 0, "self": {"dtype": "int64", "shape": (1, 2048)}},
        count=3,
        path="eager",
    )
    t(
        "aten::ge.Scalar",
        {"other": 37984, "self": {"dtype": "int64", "shape": (1, 2048)}},
        count=3,
        path="eager",
    )
    t(
        "aten::ge.Scalar",
        {"other": 75968, "self": {"dtype": "int64", "shape": (1, 2048)}},
        count=3,
        path="eager",
    )
    t(
        "aten::ge.Scalar",
        {"other": 113952, "self": {"dtype": "int64", "shape": (1, 2048)}},
        count=3,
        path="eager",
    )
    t(
        "aten::index.Tensor",
        {"indices": [{"dtype": "int64", "shape": (2048,)}], "self": (2048, 256)},
        count=36,
        path="both",
    )
    t(
        "aten::index.Tensor",
        {
            "indices": [{"dtype": "int64", "shape": (4096,)}, {"dtype": "int64", "shape": (4096,)}],
            "self": {"dtype": "int64", "shape": (4096, 64)},
        },
        count=36,
        path="both",
    )
    t(
        "aten::index_put",
        {
            "indices": [{"dtype": "int64", "shape": (4096,)}, {"dtype": "int64", "shape": (4096,)}],
            "self": {"dtype": "int32", "shape": (4096, 64)},
            "values": {"dtype": "int32", "shape": (4096,)},
        },
        count=1,
        path="compiled",
    )
    t(
        "aten::index_put",
        {
            "indices": [{"dtype": "bool", "shape": (256,)}],
            "self": {"dtype": "int64", "shape": (256,)},
            "values": {"dtype": "int64", "shape": ()},
        },
        count=1,
        path="compiled",
    )
    t(
        "aten::index_put",
        {"indices": [{"dtype": "bool", "shape": (256, 1)}], "self": (256, 1), "values": ()},
        count=1,
        path="compiled",
    )
    t(
        "aten::index_put",
        {
            "indices": [{"dtype": "int64", "shape": (256,)}, {"dtype": "int64", "shape": (256,)}],
            "self": (256, 37984),
            "values": (256,),
        },
        count=1,
        path="compiled",
    )
    t(
        "aten::index_put_",
        {
            "indices": [{"dtype": "bool", "shape": (256,)}],
            "self": {"dtype": "int64", "shape": (256,)},
            "values": {"dtype": "int64", "shape": ()},
        },
        count=48,
        path="eager",
    )
    t(
        "aten::index_put_",
        {
            "indices": [{"dtype": "int64", "shape": (4096,)}, {"dtype": "int64", "shape": (4096,)}],
            "self": {"dtype": "int32", "shape": (4096, 64)},
            "values": {"dtype": "int32", "shape": (4096,)},
        },
        count=36,
        path="eager",
    )
    t(
        "aten::index_put_",
        {"indices": [{"dtype": "bool", "shape": (256, 1)}], "self": (256, 1), "values": ()},
        count=24,
        path="eager",
    )
    t(
        "aten::index_put_",
        {
            "indices": [{"dtype": "int64", "shape": (256,)}, {"dtype": "int64", "shape": (256,)}],
            "self": (256, 37984),
            "values": (256,),
        },
        count=24,
        path="eager",
    )
    t(
        "aten::le.Scalar",
        {"other": 2047, "self": {"dtype": "int64", "shape": (1, 2048)}},
        count=36,
        path="eager",
    )
    t(
        "aten::le.Tensor",
        {"other": {"dtype": "int64", "shape": (256,)}, "self": {"dtype": "int32", "shape": (1, 1)}},
        count=1,
        path="compiled",
    )
    t("aten::linalg_vector_norm", {"ord": 2.0, "self": (18,)}, count=3, path="eager")
    t("aten::linalg_vector_norm", {"ord": 2.0, "self": (45,)}, count=3, path="eager")
    t("aten::linear", {"input": (1, 2048, 4096), "weight": (64, 4096)}, count=36, path="eager")
    t("aten::linear", {"input": (1, 2048, 4096), "weight": (9216, 4096)}, count=18, path="eager")
    t("aten::linear", {"input": (1, 2048, 8192), "weight": (4096, 8192)}, count=18, path="eager")
    t("aten::linear", {"input": (1, 2048, 4096), "weight": (2304, 4096)}, count=1, path="compiled")
    t("aten::linear", {"input": (1, 2048, 2048), "weight": (4096, 2048)}, count=1, path="compiled")
    t(
        "aten::linear_backward",
        {
            "grad_output": (1, 2048, 9216),
            "output_mask": (True, True, False),
            "self": (1, 2048, 4096),
            "weight": (9216, 4096),
        },
        count=18,
        path="eager",
    )
    t(
        "aten::linear_backward",
        {
            "grad_output": (1, 2048, 4096),
            "output_mask": (True, True, False),
            "self": (1, 2048, 8192),
            "weight": (4096, 8192),
        },
        count=18,
        path="eager",
    )
    t(
        "aten::linear_backward",
        {
            "grad_output": (1, 2048, 64),
            "output_mask": (True, True, False),
            "self": (1, 2048, 4096),
            "weight": (64, 4096),
        },
        count=18,
        path="eager",
    )
    t("aten::log", {"self": (256, 1)}, count=24, path="both")
    t("aten::log", {"self": (16, 128, 16)}, count=1, path="compiled")
    t(
        "aten::lt.Scalar",
        {"other": 0, "self": {"dtype": "int64", "shape": (256,)}},
        count=48,
        path="both",
    )
    t(
        "aten::lt.Scalar",
        {"other": 37984, "self": {"dtype": "int64", "shape": (256,)}},
        count=48,
        path="both",
    )
    t(
        "aten::lt.Scalar",
        {"other": 75968, "self": {"dtype": "int64", "shape": (256,)}},
        count=48,
        path="both",
    )
    t(
        "aten::lt.Scalar",
        {"other": 113952, "self": {"dtype": "int64", "shape": (256,)}},
        count=48,
        path="both",
    )
    t(
        "aten::lt.Scalar",
        {"other": 37984, "self": {"dtype": "int64", "shape": (1, 2048)}},
        count=3,
        path="eager",
    )
    t(
        "aten::lt.Scalar",
        {"other": 75968, "self": {"dtype": "int64", "shape": (1, 2048)}},
        count=3,
        path="eager",
    )
    t(
        "aten::lt.Scalar",
        {"other": 113952, "self": {"dtype": "int64", "shape": (1, 2048)}},
        count=3,
        path="eager",
    )
    t(
        "aten::lt.Scalar",
        {"other": 151936, "self": {"dtype": "int64", "shape": (1, 2048)}},
        count=3,
        path="eager",
    )
    t(
        "aten::maximum",
        {"other": {"dtype": "int64", "shape": ()}, "self": {"dtype": "int64", "shape": (256,)}},
        count=1,
        path="compiled",
    )
    t("aten::mean", {"self": (64,)}, count=36, path="eager")
    t(
        "aten::minimum",
        {"other": {"dtype": "int64", "shape": ()}, "self": {"dtype": "int64", "shape": (256,)}},
        count=1,
        path="compiled",
    )
    t("aten::mm", {"mat2": (64, 4096), "self": (512, 64)}, count=1, path="compiled")
    t("aten::mm", {"mat2": (512, 4096), "self": (64, 512)}, count=1, path="compiled")
    t("aten::mm", {"mat2": (4096, 2048), "self": (2048, 4096)}, count=1, path="compiled")
    t("aten::mm", {"mat2": (2048, 2048), "self": (4096, 2048)}, count=1, path="compiled")
    t("aten::mm", {"mat2": (2304, 4096), "self": (2048, 2304)}, count=1, path="compiled")
    t("aten::mm", {"mat2": (2048, 4096), "self": (2304, 2048)}, count=1, path="compiled")
    t(
        "aten::mul.Tensor",
        {"other": (1, 2048, 1, 128), "self": (1, 2048, 64, 128)},
        count=108,
        path="eager",
    )
    t("aten::mul.Tensor", {"other": (32768, 1536), "self": (32768, 1536)}, count=73, path="both")
    t("aten::mul.Tensor", {"other": (4096, 1), "self": (4096, 4096)}, count=54, path="both")
    t("aten::mul.Tensor", {"other": (256, 1), "self": (256, 37984)}, count=24, path="both")
    t("aten::mul.Tensor", {"other": (4096, 4096), "self": (4096, 4096)}, count=18, path="both")
    t("aten::mul.Tensor", {"other": (1, 2048, 1), "self": (1, 2048, 4096)}, count=6, path="eager")
    t(
        "aten::mul.Tensor",
        {"other": (1, 2048, 1, 128), "self": (1, 2048, 16, 128)},
        count=1,
        path="compiled",
    )
    t("aten::mul.Tensor", {"other": (128,), "self": (1, 2048, 16, 128)}, count=1, path="compiled")
    t(
        "aten::mul.Tensor",
        {"other": (1, 2048, 16, 1), "self": (1, 2048, 16, 128)},
        count=1,
        path="compiled",
    )
    t(
        "aten::mul.Tensor",
        {"other": (1, 2048, 16, 128), "self": (1, 2048, 16, 128)},
        count=1,
        path="compiled",
    )
    t(
        "aten::ne.Scalar",
        {"other": -100, "self": {"dtype": "int64", "shape": (256,)}},
        count=96,
        path="both",
    )
    t(
        "aten::ne.Scalar",
        {"other": -100, "self": {"dtype": "int64", "shape": (1, 2048)}},
        count=3,
        path="eager",
    )
    t("aten::neg", {"self": (1, 2048, 4, 64)}, count=54, path="eager")
    t("aten::neg", {"self": (1, 2048, 64, 64)}, count=54, path="eager")
    t("aten::neg", {"self": (256,)}, count=24, path="both")
    t("aten::neg", {"self": (1, 2048, 8)}, count=18, path="eager")
    t("aten::neg", {"self": (1, 2048, 16, 64)}, count=1, path="compiled")
    t("aten::neg", {"self": (1, 2048, 1, 64)}, count=1, path="compiled")
    t("aten::neg", {"self": (16, 128, 16)}, count=1, path="compiled")
    t("aten::neg", {"self": (1, 512, 8)}, count=1, path="compiled")
    t("aten::new_zeros", {"self": (1, 2048, 8), "size": (1, 2048, 64)}, count=18, path="eager")
    t(
        "aten::new_zeros",
        {"dtype": 6, "pin_memory": False, "self": (1, 2048, 4096), "size": ()},
        count=3,
        path="eager",
    )
    t("aten::new_zeros", {"self": (1, 512, 8), "size": (1, 2048, 64)}, count=1, path="compiled")
    t(
        "aten::one_hot",
        {"num_classes": 64, "self": {"dtype": "int64", "shape": (4096,)}},
        count=36,
        path="both",
    )
    t("aten::ones_like", {"pin_memory": False, "self": ()}, count=24, path="eager")
    t("aten::pow.Tensor_Scalar", {"exponent": 2.0, "self": ()}, count=6, path="eager")
    t("aten::pow_.Scalar", {"exponent": 0.5, "self": ()}, count=3, path="eager")
    t("aten::reciprocal", {"self": ()}, count=6, path="eager")
    t(
        "aten::remainder.Scalar",
        {"other": 512, "self": {"dtype": "int64", "shape": (4096,)}},
        count=36,
        path="both",
    )
    t("aten::rsub.Scalar", {"other": 1, "self": (32768, 1536)}, count=1, path="compiled")
    t("aten::scalar_tensor", {"dtype": 4, "s": 0}, count=6144, path="eager")
    t("aten::scalar_tensor", {"dtype": 6, "s": 0}, count=4608, path="eager")
    t(
        "aten::scatter_add",
        {
            "dim": 0,
            "index": {"dtype": "int64", "shape": (4096, 4096)},
            "self": (2048, 4096),
            "src": (4096, 4096),
        },
        count=36,
        path="both",
    )
    t(
        "aten::scatter_add",
        {
            "dim": -1,
            "index": {"dtype": "int64", "shape": (1, 2048, 8)},
            "self": (1, 2048, 64),
            "src": (1, 2048, 8),
        },
        count=18,
        path="eager",
    )
    t(
        "aten::scatter_add",
        {
            "dim": -1,
            "index": {"dtype": "int64", "shape": (1, 512, 8)},
            "self": (1, 512, 64),
            "src": (1, 512, 8),
        },
        count=1,
        path="compiled",
    )
    t("aten::select.int", {"dim": 0, "index": 0, "self": (6, 64)}, count=3, path="eager")
    t("aten::select.int", {"dim": 0, "index": 1, "self": (6, 64)}, count=3, path="eager")
    t("aten::select.int", {"dim": 0, "index": 2, "self": (6, 64)}, count=3, path="eager")
    t("aten::select.int", {"dim": 0, "index": 3, "self": (6, 64)}, count=3, path="eager")
    t("aten::select.int", {"dim": 0, "index": 4, "self": (6, 64)}, count=3, path="eager")
    t("aten::select.int", {"dim": 0, "index": 5, "self": (6, 64)}, count=3, path="eager")
    t("aten::sigmoid", {"self": (32768, 1536)}, count=1, path="compiled")
    t("aten::sign", {"self": (64,)}, count=18, path="eager")
    t("aten::silu", {"self": (32768, 1536)}, count=36, path="eager")
    t(
        "aten::silu_backward",
        {"grad_output": (32768, 1536), "self": (32768, 1536)},
        count=18,
        path="eager",
    )
    t(
        "aten::slice.Tensor",
        {"dim": 1, "end": 2048, "self": (1, 2048, 4096), "start": 0},
        count=78,
        path="both",
    )
    t(
        "aten::slice.Tensor",
        {"dim": 3, "end": 64, "self": (1, 2048, 64, 128), "start": 0},
        count=54,
        path="eager",
    )
    t(
        "aten::slice.Tensor",
        {"dim": 3, "end": 9223372036854775807, "self": (1, 2048, 64, 128), "start": 64},
        count=36,
        path="eager",
    )
    t(
        "aten::slice.Tensor",
        {"dim": 3, "end": 128, "self": (1, 2048, 64, 128), "start": 64},
        count=18,
        path="eager",
    )
    t(
        "aten::slice.Tensor",
        {"dim": 1, "end": 256, "self": (1, 2048, 4096), "start": 0},
        count=3,
        path="eager",
    )
    t(
        "aten::slice.Tensor",
        {"dim": 1, "end": 1280, "self": (1, 2048, 4096), "start": 1024},
        count=3,
        path="eager",
    )
    t(
        "aten::slice.Tensor",
        {"dim": 1, "end": 1536, "self": (1, 2048, 4096), "start": 1280},
        count=3,
        path="eager",
    )
    t(
        "aten::slice.Tensor",
        {"dim": 1, "end": 1792, "self": (1, 2048, 4096), "start": 1536},
        count=3,
        path="eager",
    )
    t(
        "aten::slice.Tensor",
        {"dim": 1, "end": 2048, "self": (1, 2048, 4096), "start": 1792},
        count=3,
        path="eager",
    )
    t(
        "aten::slice.Tensor",
        {"dim": 1, "end": 512, "self": (1, 2048, 4096), "start": 256},
        count=3,
        path="eager",
    )
    t(
        "aten::slice_backward",
        {
            "dim": 3,
            "end": 64,
            "grad_output": (1, 2048, 4, 64),
            "input_sizes": (1, 2048, 4, 128),
            "start": 0,
            "step": 1,
        },
        count=18,
        path="eager",
    )
    t(
        "aten::slice_backward",
        {
            "dim": 3,
            "end": 9223372036854775807,
            "grad_output": (1, 2048, 4, 64),
            "input_sizes": (1, 2048, 4, 128),
            "start": 64,
            "step": 1,
        },
        count=18,
        path="eager",
    )
    t(
        "aten::slice_backward",
        {
            "dim": 3,
            "end": 64,
            "grad_output": (1, 2048, 64, 64),
            "input_sizes": (1, 2048, 64, 128),
            "start": 0,
            "step": 1,
        },
        count=18,
        path="eager",
    )
    t(
        "aten::slice_backward",
        {
            "dim": 3,
            "end": 9223372036854775807,
            "grad_output": (1, 2048, 64, 64),
            "input_sizes": (1, 2048, 64, 128),
            "start": 64,
            "step": 1,
        },
        count=18,
        path="eager",
    )
    t(
        "aten::split.Tensor",
        {"dim": 1, "self": (1, 2048, 4096), "split_size": 512},
        count=96,
        path="eager",
    )
    t(
        "aten::split_with_sizes",
        {"self": (1131024,), "split_sizes": (589824, 524288, 8, 8, 16384, 256, 256)},
        count=60,
        path="eager",
    )
    t(
        "aten::split_with_sizes",
        {"dim": -2, "self": (1, 2048, 1, 18, 128), "split_sizes": (16, 1, 1)},
        count=36,
        path="eager",
    )
    t(
        "aten::split_with_sizes",
        {"self": (9723904,), "split_sizes": (9723904,)},
        count=9,
        path="eager",
    )
    t("aten::split_with_sizes", {"self": (256,), "split_sizes": (256,)}, count=6, path="eager")
    t(
        "aten::split_with_sizes",
        {
            "dim": 1,
            "self": (1, 2048, 4096),
            "split_sizes": (256, 256, 256, 256, 256, 256, 256, 256),
        },
        count=3,
        path="eager",
    )
    t(
        "aten::split_with_sizes",
        {
            "dim": 1,
            "self": {"dtype": "int64", "shape": (1, 2048)},
            "split_sizes": (256, 256, 256, 256, 256, 256, 256, 256),
        },
        count=3,
        path="eager",
    )
    t(
        "aten::squeeze.dim",
        {"dim": 0, "self": {"dtype": "int64", "shape": (1, 2048)}},
        count=36,
        path="eager",
    )
    t("aten::squeeze.dim", {"dim": -1, "self": (256, 1)}, count=24, path="both")
    t("aten::squeeze.dim", {"dim": 1, "self": (512, 1, 4096)}, count=18, path="both")
    t("aten::stack", {"tensors": [{"dtype": "int64", "shape": (2048,)}]}, count=9, path="eager")
    t(
        "aten::stack",
        {"tensors": [(), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), ()]},
        count=3,
        path="eager",
    )
    t("aten::stack", {"tensors": [()]}, count=3, path="eager")
    t(
        "aten::sub.Tensor",
        {"other": (), "self": {"dtype": "int64", "shape": (256,)}},
        count=192,
        path="both",
    )
    t("aten::sub.Tensor", {"other": (256, 1), "self": (256, 37984)}, count=48, path="both")
    t(
        "aten::sub.Tensor",
        {"other": (), "self": {"dtype": "int64", "shape": (4096,)}},
        count=36,
        path="both",
    )
    t("aten::sub.Tensor", {"other": (), "self": (256,)}, count=24, path="both")
    t("aten::sub.Tensor", {"other": (), "self": (64,)}, count=18, path="eager")
    t(
        "aten::sub.Tensor",
        {"other": (), "self": {"dtype": "int64", "shape": (1, 2048)}},
        count=12,
        path="eager",
    )
    t(
        "aten::sub.Tensor",
        {"other": (1, 2048, 16, 128), "self": (1, 2048, 16, 128)},
        count=1,
        path="compiled",
    )
    t(
        "aten::sub.Tensor",
        {"other": (1, 512, 4096), "self": (1, 512, 4096)},
        count=1,
        path="compiled",
    )
    t(
        "aten::sub.Tensor",
        {"other": (1, 2048, 1, 128), "self": (1, 2048, 1, 128)},
        count=1,
        path="compiled",
    )
    t("aten::sub.Tensor", {"other": (1, 512, 64), "self": (1, 512, 64)}, count=1, path="compiled")
    t("aten::sum", {"self": (256,)}, count=24, path="both")
    t("aten::sum", {"self": {"dtype": "bool", "shape": (1, 2048)}}, count=3, path="eager")
    t("aten::sum", {"self": (1,)}, count=3, path="eager")
    t(
        "aten::sum.dim_IntList",
        {"dim": (0,), "self": {"dtype": "int64", "shape": (4096, 64)}},
        count=36,
        path="both",
    )
    t(
        "aten::sum.dim_IntList",
        {"dim": (-1,), "keepdim": True, "self": (256, 37984)},
        count=24,
        path="both",
    )
    t(
        "aten::sum.dim_IntList",
        {"dim": (1,), "keepdim": True, "self": (512, 8, 4096)},
        count=18,
        path="both",
    )
    t(
        "aten::sum.dim_IntList",
        {"dim": (1,), "keepdim": True, "self": (4096, 4096)},
        count=18,
        path="both",
    )
    t(
        "aten::sum.dim_IntList",
        {"dim": (3,), "keepdim": True, "self": (1, 2048, 16, 128)},
        count=1,
        path="compiled",
    )
    t(
        "aten::sum.dim_IntList",
        {"dim": (0, 1, 2), "self": (1, 2048, 16, 128)},
        count=1,
        path="compiled",
    )
    t(
        "aten::sum.dim_IntList",
        {"dim": (2,), "keepdim": True, "self": (1, 512, 4096)},
        count=1,
        path="compiled",
    )
    t("aten::sum.dim_IntList", {"dim": (0, 1), "self": (1, 512, 4096)}, count=1, path="compiled")
    t(
        "aten::sum.dim_IntList",
        {"dim": (3,), "keepdim": True, "self": (1, 2048, 1, 128)},
        count=1,
        path="compiled",
    )
    t(
        "aten::sum.dim_IntList",
        {"dim": (0, 1, 2), "self": (1, 2048, 1, 128)},
        count=1,
        path="compiled",
    )
    t("aten::t", {"self": (512, 64)}, count=1, path="compiled")
    t("aten::t", {"self": (2048, 4096)}, count=1, path="compiled")
    t("aten::t", {"self": (2048, 2304)}, count=1, path="compiled")
    t(
        "aten::topk",
        {"dim": -1, "k": 8, "largest": True, "self": (1, 2048, 64), "sorted": False},
        count=18,
        path="eager",
    )
    t("aten::topk", {"k": 8, "self": (1, 512, 64)}, count=1, path="compiled")
    t(
        "aten::transpose.int",
        {"dim0": -2, "dim1": -1, "self": (1, 1536, 4096)},
        count=144,
        path="both",
    )
    t(
        "aten::transpose.int",
        {"dim0": -2, "dim1": -1, "self": (1, 4096, 1536)},
        count=126,
        path="both",
    )
    t(
        "aten::transpose.int",
        {"dim0": 1, "dim1": 2, "self": (1, 2048, 1, 128)},
        count=72,
        path="both",
    )
    t(
        "aten::transpose.int",
        {"dim0": 1, "dim1": 2, "self": (1, 16, 2048, 128)},
        count=54,
        path="both",
    )
    t(
        "aten::transpose.int",
        {"dim0": 1, "dim1": 2, "self": (1, 2048, 16, 128)},
        count=54,
        path="both",
    )
    t("aten::transpose.int", {"dim0": -2, "dim1": -1, "self": (32768, 1536)}, count=36, path="both")
    t(
        "aten::transpose.int",
        {"dim0": 1, "dim1": 2, "self": (1, 1, 2048, 128)},
        count=36,
        path="both",
    )
    t("aten::transpose.int", {"dim0": -2, "dim1": -1, "self": (32768, 4096)}, count=18, path="both")
    t(
        "aten::transpose.int",
        {"dim0": -2, "dim1": -1, "self": (1, 16, 2048, 128)},
        count=1,
        path="compiled",
    )
    t(
        "aten::transpose.int",
        {"dim0": -2, "dim1": -1, "self": (1, 16, 128, 2048)},
        count=1,
        path="compiled",
    )
    t(
        "aten::unsqueeze",
        {"dim": -1, "self": {"dtype": "bool", "shape": (256,)}},
        count=48,
        path="both",
    )
    t("aten::unsqueeze", {"dim": 1, "self": (512, 4096)}, count=36, path="both")
    t(
        "aten::unsqueeze",
        {"dim": -1, "self": {"dtype": "int64", "shape": (1, 2048, 8)}},
        count=36,
        path="eager",
    )
    t(
        "aten::unsqueeze",
        {"dim": -1, "self": {"dtype": "int64", "shape": (256,)}},
        count=24,
        path="both",
    )
    t("aten::unsqueeze", {"dim": 0, "self": (64,)}, count=18, path="eager")
    t(
        "aten::unsqueeze",
        {"dim": -1, "self": {"dtype": "bool", "shape": (1, 2048)}},
        count=3,
        path="eager",
    )
    t(
        "aten::unsqueeze",
        {"dim": -1, "self": {"dtype": "int64", "shape": (1, 512, 8)}},
        count=1,
        path="compiled",
    )
    t(
        "aten::unsqueeze",
        {"dim": 1, "self": {"dtype": "int32", "shape": (1,)}},
        count=1,
        path="compiled",
    )
    t(
        "aten::view",
        {"self": (1, 2048, 9216), "size": (1, 2048, -1, 18, 128)},
        count=36,
        path="eager",
    )
    t("aten::view", {"self": (155582464,), "size": (16, -1)}, count=24, path="eager")
    t(
        "aten::view",
        {"self": (1, 2048, 4, 18, 128), "size": (1, 2048, 9216)},
        count=18,
        path="eager",
    )
    t("aten::view", {"self": (37984, 4096), "size": (37984, 4096)}, count=6, path="eager")
    t("aten::view", {"self": (32768, 4096), "size": (256, 128, 4096)}, count=1, path="compiled")
    t("aten::view", {"self": (256, 128, 4096), "size": (256, 128, 4096)}, count=1, path="compiled")
    t("aten::view", {"self": (256, 128, 4096), "size": (32768, 4096)}, count=1, path="compiled")
    t("aten::view", {"self": (256, 128, 1536), "size": (256, 128, 1536)}, count=1, path="compiled")
    t("aten::view", {"self": (256, 128, 1536), "size": (32768, 1536)}, count=1, path="compiled")
    t("aten::view", {"self": (32768, 1536), "size": (256, 128, 1536)}, count=1, path="compiled")
    t(
        "aten::where.self",
        {
            "condition": {"dtype": "bool", "shape": (256,)},
            "other": {"dtype": "int64", "shape": ()},
            "self": {"dtype": "int64", "shape": (256,)},
        },
        count=48,
        path="both",
    )
    t(
        "aten::where.self",
        {"condition": {"dtype": "bool", "shape": (256, 1)}, "other": (), "self": ()},
        count=24,
        path="both",
    )
    t(
        "aten::where.self",
        {"condition": {"dtype": "bool", "shape": (256,)}, "other": (), "self": (256,)},
        count=24,
        path="both",
    )
    t("aten::zero_", {"self": (64,)}, count=18, path="eager")
    t("aten::zeros", {"dtype": 6, "pin_memory": False, "size": ()}, count=4032, path="eager")
    t(
        "aten::zeros",
        {"dtype": 15, "pin_memory": False, "size": (2048, 4096)},
        count=2304,
        path="both",
    )
    t("aten::zeros", {"dtype": 3, "size": (1, 64)}, count=1152, path="eager")
    t("aten::zeros_like", {"pin_memory": False, "self": (4096,)}, count=26, path="eager")
    t("aten::zeros_like", {"pin_memory": False, "self": (128,)}, count=24, path="eager")
    t("aten::zeros_like", {"pin_memory": False, "self": (256, 37984)}, count=24, path="both")
    t("aten::zeros_like", {"pin_memory": False, "self": (64, 1536, 4096)}, count=24, path="eager")
    t("aten::zeros_like", {"pin_memory": False, "self": (4096, 8192)}, count=12, path="eager")
    t("aten::zeros_like", {"pin_memory": False, "self": (64, 4096, 1536)}, count=12, path="eager")
    t("aten::zeros_like", {"pin_memory": False, "self": (64, 4096)}, count=12, path="eager")
    t("aten::zeros_like", {"pin_memory": False, "self": (9216, 4096)}, count=12, path="eager")
    t(
        "aten::zeros_like",
        {"dtype": 6, "pin_memory": False, "self": (1, 2048, 4096)},
        count=3,
        path="eager",
    )


# ── provenance ───────────────────────────────────────────────────────────────────────────────
# The block above is generated output of ``gen_ep64_registry.py --model qwen3_235b_ep64_tp4``
# from the reference branch ``opcap/qwen3-235b-ep64``. Details in ``tests/vertical/README.md``.


# The accuracy smoke test subsets, named by ``--traced-model``. A key is the subset name, and
# it carries the model and the distributed degrees of the capture (``ep64_tp4``).
SMOKE_SUBSET_SHAPES = {
    QWEN3_235B_EP64_TP4: _register_qwen3_235b_ep64_tp4,
}

SMOKE_SUBSET_OPS = {
    QWEN3_235B_EP64_TP4: QWEN3_235B_EP64_TP4_OPS,
}

# Op mode per subset, e.g. ``aten::silu`` is ``eager_only`` in this subset because AOT lowers
# it to sigmoid+mul. The mode tells you if an op exists in eager, in compile, or in both.
SMOKE_SUBSET_OP_MODE = {
    QWEN3_235B_EP64_TP4: QWEN3_235B_EP64_TP4_OP_MODE,
}


# BEGIN GENERATED CACHE-KEY SHAPES
# Do not edit. Regenerate with tests/vertical/internal/refresh_cache_key_shapes.py.
CACHE_KEY_SOURCE = "cache_keys"


def register_cache_key_shapes(r):
    """Register cache-key model shapes (145 ops, 739 shapes).

    These come from the HuggingFace and torchtitan model runs in CI. Each model job
    writes its Neuron compiler cache keys; each key records an op, its argument shapes,
    and dtypes. refresh_cache_key_shapes.py maps those to op schema arg names, caps the
    shapes per op, and regenerates this function. The vertical suite calls it by default
    (opt out with --no-cache-key-shapes).
    """

    def t(op, arg_shapes, count=1, paths=()):
        r(op, arg_shapes, source=CACHE_KEY_SOURCE, count=count, paths=paths)

    t(
        "aten::_foreach_add_.List",
        {
            "alpha": 1.0,
            "other": [
                {"dtype": "bfloat16", "shape": [3000, 2048]},
                {"dtype": "bfloat16", "shape": [32]},
                {"dtype": "bfloat16", "shape": [32]},
                {"dtype": "bfloat16", "shape": [8192, 1, 4]},
                {"dtype": "bfloat16", "shape": [12288, 2048]},
                {"dtype": "bfloat16", "shape": [64, 2048]},
            ],
            "self": [
                {"dtype": "bfloat16", "shape": [3000, 2048]},
                {"dtype": "bfloat16", "shape": [32]},
                {"dtype": "bfloat16", "shape": [32]},
                {"dtype": "bfloat16", "shape": [8192, 1, 4]},
                {"dtype": "bfloat16", "shape": [12288, 2048]},
                {"dtype": "bfloat16", "shape": [64, 2048]},
            ],
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::_foreach_add_.List",
        {
            "alpha": 1.0,
            "other": [
                {"dtype": "bfloat16", "shape": [3000, 4088]},
                {"dtype": "bfloat16", "shape": [1536, 4088]},
                {"dtype": "bfloat16", "shape": [1536]},
                {"dtype": "bfloat16", "shape": [14016, 1536]},
                {"dtype": "bfloat16", "shape": [576, 4088]},
                {"dtype": "bfloat16", "shape": [512]},
            ],
            "self": [
                {"dtype": "bfloat16", "shape": [3000, 4088]},
                {"dtype": "bfloat16", "shape": [1536, 4088]},
                {"dtype": "bfloat16", "shape": [1536]},
                {"dtype": "bfloat16", "shape": [14016, 1536]},
                {"dtype": "bfloat16", "shape": [576, 4088]},
                {"dtype": "bfloat16", "shape": [512]},
            ],
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::_foreach_add_.List",
        {
            "alpha": 1.0,
            "other": [
                {"dtype": "bfloat16", "shape": [3000, 3072]},
                {"dtype": "bfloat16", "shape": [6144, 3072]},
                {"dtype": "bfloat16", "shape": [1024, 3072]},
                {"dtype": "bfloat16", "shape": [1024, 3072]},
                {"dtype": "bfloat16", "shape": [3072, 6144]},
                {"dtype": "bfloat16", "shape": [6144]},
            ],
            "self": [
                {"dtype": "bfloat16", "shape": [3000, 3072]},
                {"dtype": "bfloat16", "shape": [6144, 3072]},
                {"dtype": "bfloat16", "shape": [1024, 3072]},
                {"dtype": "bfloat16", "shape": [1024, 3072]},
                {"dtype": "bfloat16", "shape": [3072, 6144]},
                {"dtype": "bfloat16", "shape": [6144]},
            ],
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::_foreach_add_.List",
        {
            "alpha": 1.0,
            "other": [
                {"dtype": "bfloat16", "shape": [3000, 4032]},
                {"dtype": "bfloat16", "shape": [4352, 4032]},
                {"dtype": "bfloat16", "shape": [4032, 2688]},
                {"dtype": "bfloat16", "shape": [4096, 4032]},
                {"dtype": "bfloat16", "shape": [4096, 4032]},
                {"dtype": "bfloat16", "shape": [4032, 4096]},
            ],
            "self": [
                {"dtype": "bfloat16", "shape": [3000, 4032]},
                {"dtype": "bfloat16", "shape": [4352, 4032]},
                {"dtype": "bfloat16", "shape": [4032, 2688]},
                {"dtype": "bfloat16", "shape": [4096, 4032]},
                {"dtype": "bfloat16", "shape": [4096, 4032]},
                {"dtype": "bfloat16", "shape": [4032, 4096]},
            ],
        },
        count=9,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::_foreach_add_.List",
        {
            "alpha": 1.0,
            "other": [
                {"dtype": "bfloat16", "shape": [3000, 4096]},
                {"dtype": "bfloat16", "shape": [4096, 4096]},
                {"dtype": "bfloat16", "shape": [1024, 4096]},
                {"dtype": "bfloat16", "shape": [1024, 4096]},
                {"dtype": "bfloat16", "shape": [4096, 4096]},
                {"dtype": "bfloat16", "shape": [128]},
            ],
            "self": [
                {"dtype": "bfloat16", "shape": [3000, 4096]},
                {"dtype": "bfloat16", "shape": [4096, 4096]},
                {"dtype": "bfloat16", "shape": [1024, 4096]},
                {"dtype": "bfloat16", "shape": [1024, 4096]},
                {"dtype": "bfloat16", "shape": [4096, 4096]},
                {"dtype": "bfloat16", "shape": [128]},
            ],
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::_foreach_add_.List",
        {
            "alpha": 1.0,
            "other": [
                {"dtype": "bfloat16", "shape": [3000, 4096]},
                {"dtype": "bfloat16", "shape": [4096, 4096]},
                {"dtype": "bfloat16", "shape": [1024, 4096]},
                {"dtype": "bfloat16", "shape": [1024, 4096]},
                {"dtype": "bfloat16", "shape": [4096, 4096]},
                {"dtype": "bfloat16", "shape": [64]},
            ],
            "self": [
                {"dtype": "bfloat16", "shape": [3000, 4096]},
                {"dtype": "bfloat16", "shape": [4096, 4096]},
                {"dtype": "bfloat16", "shape": [1024, 4096]},
                {"dtype": "bfloat16", "shape": [1024, 4096]},
                {"dtype": "bfloat16", "shape": [4096, 4096]},
                {"dtype": "bfloat16", "shape": [64]},
            ],
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::_foreach_add_.Scalar",
        {
            "scalar": 0.0,
            "self": [
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
            ],
        },
        count=78,
        paths=(
            "test_kaizen_hf_onboarding_reduced_eager",
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-dev",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-compile-trn3",
            "text-generation-torch-titan-multi-core-compile-trn3-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-compile-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
            "text-generation-torch-titan-qwen3-235b-trn3",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::_foreach_add_.Scalar",
        {
            "scalar": 0.0,
            "self": [
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
            ],
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::_foreach_copy_",
        {"non_blocking": False, "self": {"dtype": "bfloat16", "shape": [12155200]}},
        count=7,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::_foreach_copy_",
        {"non_blocking": False, "self": {"dtype": "float32", "shape": [12155200]}},
        count=7,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::_foreach_copy_",
        {"non_blocking": False, "self": {"dtype": "bfloat16", "shape": [12154960]}},
        count=1,
        paths=("text-generation-torch-titan-multi-core-trn3",),
    )
    t(
        "aten::_foreach_copy_",
        {"non_blocking": False, "self": {"dtype": "float32", "shape": [12154960]}},
        count=1,
        paths=("text-generation-torch-titan-multi-core-trn3",),
    )
    t(
        "aten::_foreach_copy_",
        {"non_blocking": False, "self": {"dtype": "bfloat16", "shape": [12154880]}},
        count=7,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::_foreach_copy_",
        {"non_blocking": False, "self": {"dtype": "float32", "shape": [12154880]}},
        count=7,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::_foreach_mul_.Tensor",
        {
            "other": {"dtype": "float32", "shape": []},
            "self": [
                {"dtype": "float32", "shape": [8016, 4096]},
                {"dtype": "float32", "shape": [256, 4096]},
                {"dtype": "float32", "shape": [64, 4096]},
                {"dtype": "float32", "shape": [64, 4096]},
                {"dtype": "float32", "shape": [1024, 1024]},
                {"dtype": "float32", "shape": [896, 4096]},
            ],
        },
        count=1,
        paths=("text-generation-torch-titan-multi-node-4N-async-hostcc",),
    )
    t(
        "aten::_foreach_mul_.Tensor",
        {
            "other": {"dtype": "float32", "shape": []},
            "self": [
                {"dtype": "float32", "shape": [2374, 5120]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [128, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
            ],
        },
        count=3,
        paths=(
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::_foreach_mul_.Tensor",
        {
            "other": {"dtype": "float32", "shape": []},
            "self": [
                {"dtype": "float32", "shape": [2374, 5120]},
                {"dtype": "float32", "shape": [2]},
                {"dtype": "float32", "shape": [2]},
                {"dtype": "float32", "shape": [128, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
            ],
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::_foreach_mul_.Tensor",
        {
            "other": {"dtype": "float32", "shape": []},
            "self": [
                {"dtype": "float32", "shape": [2374, 5120]},
                {"dtype": "float32", "shape": [160, 5120]},
                {"dtype": "float32", "shape": [320, 2048]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [400, 5120]},
            ],
        },
        count=2,
        paths=(
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::_foreach_mul_.Tensor",
        {
            "other": {"dtype": "float32", "shape": []},
            "self": [
                {"dtype": "float32", "shape": [2374, 5120]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [80, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
            ],
        },
        count=1,
        paths=("text-generation-torch-titan-multi-core-async-hostcc",),
    )
    t(
        "aten::_foreach_mul_.Tensor",
        {
            "other": {"dtype": "float64", "shape": []},
            "self": [
                {"dtype": "float32", "shape": [2374, 5120]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [128, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
            ],
        },
        count=1,
        paths=("text-generation-torch-titan-multi-core-async",),
    )
    t(
        "aten::_foreach_norm.Scalar",
        {
            "dtype": None,
            "ord": 2.0,
            "self": [
                {"dtype": "float32", "shape": [8016, 4096]},
                {"dtype": "float32", "shape": [256, 4096]},
                {"dtype": "float32", "shape": [64, 4096]},
                {"dtype": "float32", "shape": [64, 4096]},
                {"dtype": "float32", "shape": [1024, 1024]},
                {"dtype": "float32", "shape": [896, 4096]},
            ],
        },
        count=1,
        paths=("text-generation-torch-titan-multi-node-4N-async-hostcc",),
    )
    t(
        "aten::_foreach_norm.Scalar",
        {
            "dtype": None,
            "ord": 2.0,
            "self": [
                {"dtype": "float32", "shape": [2374, 5120]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [128, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
            ],
        },
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::_foreach_norm.Scalar",
        {
            "dtype": None,
            "ord": 2.0,
            "self": [
                {"dtype": "float32", "shape": [2374, 5120]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [80, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
            ],
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::_foreach_norm.Scalar",
        {
            "dtype": None,
            "ord": 2.0,
            "self": [
                {"dtype": "float32", "shape": [2374, 5120]},
                {"dtype": "float32", "shape": [2]},
                {"dtype": "float32", "shape": [2]},
                {"dtype": "float32", "shape": [128, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
            ],
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::_foreach_norm.Scalar",
        {
            "dtype": None,
            "ord": 2.0,
            "self": [
                {"dtype": "float32", "shape": [2374, 5120]},
                {"dtype": "float32", "shape": [160, 5120]},
                {"dtype": "float32", "shape": [320, 2048]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [400, 5120]},
            ],
        },
        count=2,
        paths=(
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::_foreach_norm.Scalar",
        {
            "dtype": None,
            "ord": 2.0,
            "self": [
                {"dtype": "float32", "shape": [2374, 4096]},
                {"dtype": "float32", "shape": [2]},
                {"dtype": "float32", "shape": [2]},
                {"dtype": "float32", "shape": [64, 4096]},
                {"dtype": "float32", "shape": [16, 4096]},
                {"dtype": "float32", "shape": [16, 4096]},
            ],
        },
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::_fused_adamw_",
        {
            "amsgrad": False,
            "beta1": 0.9,
            "beta2": 0.95,
            "eps": 1e-08,
            "exp_avg_sqs": [
                {"dtype": "float32", "shape": [8016, 4096]},
                {"dtype": "float32", "shape": [256, 4096]},
                {"dtype": "float32", "shape": [64, 4096]},
                {"dtype": "float32", "shape": [64, 4096]},
                {"dtype": "float32", "shape": [1024, 1024]},
                {"dtype": "float32", "shape": [896, 4096]},
            ],
            "exp_avgs": [
                {"dtype": "float32", "shape": [8016, 4096]},
                {"dtype": "float32", "shape": [256, 4096]},
                {"dtype": "float32", "shape": [64, 4096]},
                {"dtype": "float32", "shape": [64, 4096]},
                {"dtype": "float32", "shape": [1024, 1024]},
                {"dtype": "float32", "shape": [896, 4096]},
            ],
            "found_inf": None,
            "grad_scale": None,
            "grads": [
                {"dtype": "float32", "shape": [8016, 4096]},
                {"dtype": "float32", "shape": [256, 4096]},
                {"dtype": "float32", "shape": [64, 4096]},
                {"dtype": "float32", "shape": [64, 4096]},
                {"dtype": "float32", "shape": [1024, 1024]},
                {"dtype": "float32", "shape": [896, 4096]},
            ],
            "lr": 0.001,
            "max_exp_avg_sqs": [],
            "maximize": False,
            "self": [
                {"dtype": "float32", "shape": [8016, 4096]},
                {"dtype": "float32", "shape": [256, 4096]},
                {"dtype": "float32", "shape": [64, 4096]},
                {"dtype": "float32", "shape": [64, 4096]},
                {"dtype": "float32", "shape": [1024, 1024]},
                {"dtype": "float32", "shape": [896, 4096]},
            ],
            "state_steps": [
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
            ],
            "weight_decay": 0.1,
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-node-4N-async-compile-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
        ),
    )
    t(
        "aten::_fused_adamw_",
        {
            "amsgrad": False,
            "beta1": 0.9,
            "beta2": 0.95,
            "eps": 1e-08,
            "exp_avg_sqs": [
                {"dtype": "float32", "shape": [2374, 5120]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [128, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
            ],
            "exp_avgs": [
                {"dtype": "float32", "shape": [2374, 5120]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [128, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
            ],
            "found_inf": None,
            "grad_scale": None,
            "grads": [
                {"dtype": "float32", "shape": [2374, 5120]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [128, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
            ],
            "lr": 0.001,
            "max_exp_avg_sqs": [],
            "maximize": False,
            "self": [
                {"dtype": "float32", "shape": [2374, 5120]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [128, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
            ],
            "state_steps": [
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
            ],
            "weight_decay": 0.1,
        },
        count=6,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::_fused_adamw_",
        {
            "amsgrad": False,
            "beta1": 0.9,
            "beta2": 0.95,
            "eps": 1e-08,
            "exp_avg_sqs": [
                {"dtype": "float32", "shape": [2374, 5120]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [80, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
            ],
            "exp_avgs": [
                {"dtype": "float32", "shape": [2374, 5120]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [80, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
            ],
            "found_inf": None,
            "grad_scale": None,
            "grads": [
                {"dtype": "float32", "shape": [2374, 5120]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [80, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
            ],
            "lr": 0.001,
            "max_exp_avg_sqs": [],
            "maximize": False,
            "self": [
                {"dtype": "float32", "shape": [2374, 5120]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [80, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
            ],
            "state_steps": [
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
            ],
            "weight_decay": 0.1,
        },
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::_fused_adamw_",
        {
            "amsgrad": False,
            "beta1": 0.9,
            "beta2": 0.95,
            "eps": 1e-08,
            "exp_avg_sqs": [
                {"dtype": "float32", "shape": [2374, 5120]},
                {"dtype": "float32", "shape": [160, 5120]},
                {"dtype": "float32", "shape": [320, 2048]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [400, 5120]},
            ],
            "exp_avgs": [
                {"dtype": "float32", "shape": [2374, 5120]},
                {"dtype": "float32", "shape": [160, 5120]},
                {"dtype": "float32", "shape": [320, 2048]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [400, 5120]},
            ],
            "found_inf": None,
            "grad_scale": None,
            "grads": [
                {"dtype": "float32", "shape": [2374, 5120]},
                {"dtype": "float32", "shape": [160, 5120]},
                {"dtype": "float32", "shape": [320, 2048]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [400, 5120]},
            ],
            "lr": 0.001,
            "max_exp_avg_sqs": [],
            "maximize": False,
            "self": [
                {"dtype": "float32", "shape": [2374, 5120]},
                {"dtype": "float32", "shape": [160, 5120]},
                {"dtype": "float32", "shape": [320, 2048]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [400, 5120]},
            ],
            "state_steps": [
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
            ],
            "weight_decay": 0.1,
        },
        count=4,
        paths=(
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::_fused_adamw_",
        {
            "amsgrad": False,
            "beta1": 0.9,
            "beta2": 0.95,
            "eps": 1e-08,
            "exp_avg_sqs": [
                {"dtype": "float32", "shape": [2374, 5120]},
                {"dtype": "float32", "shape": [2]},
                {"dtype": "float32", "shape": [2]},
                {"dtype": "float32", "shape": [128, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
            ],
            "exp_avgs": [
                {"dtype": "float32", "shape": [2374, 5120]},
                {"dtype": "float32", "shape": [2]},
                {"dtype": "float32", "shape": [2]},
                {"dtype": "float32", "shape": [128, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
            ],
            "found_inf": None,
            "grad_scale": None,
            "grads": [
                {"dtype": "float32", "shape": [2374, 5120]},
                {"dtype": "float32", "shape": [2]},
                {"dtype": "float32", "shape": [2]},
                {"dtype": "float32", "shape": [128, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
            ],
            "lr": 0.001,
            "max_exp_avg_sqs": [],
            "maximize": False,
            "self": [
                {"dtype": "float32", "shape": [2374, 5120]},
                {"dtype": "float32", "shape": [2]},
                {"dtype": "float32", "shape": [2]},
                {"dtype": "float32", "shape": [128, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
                {"dtype": "float32", "shape": [16, 5120]},
            ],
            "state_steps": [
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
            ],
            "weight_decay": 0.1,
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::_fused_adamw_",
        {
            "amsgrad": False,
            "beta1": 0.9,
            "beta2": 0.95,
            "eps": 1e-08,
            "exp_avg_sqs": [
                {"dtype": "float32", "shape": [2374, 4096]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [64, 4096]},
                {"dtype": "float32", "shape": [16, 4096]},
                {"dtype": "float32", "shape": [16, 4096]},
            ],
            "exp_avgs": [
                {"dtype": "float32", "shape": [2374, 4096]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [64, 4096]},
                {"dtype": "float32", "shape": [16, 4096]},
                {"dtype": "float32", "shape": [16, 4096]},
            ],
            "found_inf": None,
            "grad_scale": None,
            "grads": [
                {"dtype": "float32", "shape": [2374, 4096]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [64, 4096]},
                {"dtype": "float32", "shape": [16, 4096]},
                {"dtype": "float32", "shape": [16, 4096]},
            ],
            "lr": 0.001,
            "max_exp_avg_sqs": [],
            "maximize": False,
            "self": [
                {"dtype": "float32", "shape": [2374, 4096]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [8]},
                {"dtype": "float32", "shape": [64, 4096]},
                {"dtype": "float32", "shape": [16, 4096]},
                {"dtype": "float32", "shape": [16, 4096]},
            ],
            "state_steps": [
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
                {"dtype": "float32", "shape": []},
            ],
            "weight_decay": 0.1,
        },
        count=7,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-dev",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::_fused_rms_norm",
        {
            "eps": 1e-06,
            "input": {"dtype": "bfloat16", "shape": [1, 8192, 64, 128]},
            "normalized_shape": [128],
            "weight": {"dtype": "bfloat16", "shape": [128]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::_fused_rms_norm",
        {
            "eps": 1e-06,
            "input": {"dtype": "bfloat16", "shape": [1, 8192, 5120]},
            "normalized_shape": [5120],
            "weight": {"dtype": "bfloat16", "shape": [5120]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::_fused_rms_norm",
        {
            "eps": 1e-06,
            "input": {"dtype": "bfloat16", "shape": [1, 8192, 4096]},
            "normalized_shape": [4096],
            "weight": {"dtype": "bfloat16", "shape": [4096]},
        },
        count=6,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::_fused_rms_norm",
        {
            "eps": 1e-06,
            "input": {"dtype": "bfloat16", "shape": [1, 8192, 32, 128]},
            "normalized_shape": [128],
            "weight": {"dtype": "bfloat16", "shape": [128]},
        },
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::_fused_rms_norm",
        {
            "eps": 1e-06,
            "input": {"dtype": "bfloat16", "shape": [1, 4096, 64, 128]},
            "normalized_shape": [128],
            "weight": {"dtype": "bfloat16", "shape": [128]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::_fused_rms_norm",
        {
            "eps": 1e-06,
            "input": {"dtype": "bfloat16", "shape": [1, 4096, 5120]},
            "normalized_shape": [5120],
            "weight": {"dtype": "bfloat16", "shape": [5120]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::_fused_rms_norm_backward",
        {
            "grad_out": {"dtype": "bfloat16", "shape": [1, 8192, 64, 128]},
            "input": {"dtype": "bfloat16", "shape": [1, 8192, 64, 128]},
            "normalized_shape": [128],
            "output_mask": [True, True],
            "rstd": {"dtype": "float32", "shape": [1, 8192, 64, 1]},
            "weight": {"dtype": "bfloat16", "shape": [128]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::_fused_rms_norm_backward",
        {
            "grad_out": {"dtype": "bfloat16", "shape": [1, 8192, 5120]},
            "input": {"dtype": "bfloat16", "shape": [1, 8192, 5120]},
            "normalized_shape": [5120],
            "output_mask": [True, True],
            "rstd": {"dtype": "float32", "shape": [1, 8192, 1]},
            "weight": {"dtype": "bfloat16", "shape": [5120]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::_fused_rms_norm_backward",
        {
            "grad_out": {"dtype": "bfloat16", "shape": [1, 8192, 4096]},
            "input": {"dtype": "bfloat16", "shape": [1, 8192, 4096]},
            "normalized_shape": [4096],
            "output_mask": [True, True],
            "rstd": {"dtype": "float32", "shape": [1, 8192, 1]},
            "weight": {"dtype": "bfloat16", "shape": [4096]},
        },
        count=6,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::_fused_rms_norm_backward",
        {
            "grad_out": {"dtype": "bfloat16", "shape": [1, 8192, 32, 128]},
            "input": {"dtype": "bfloat16", "shape": [1, 8192, 32, 128]},
            "normalized_shape": [128],
            "output_mask": [True, True],
            "rstd": {"dtype": "float32", "shape": [1, 8192, 32, 1]},
            "weight": {"dtype": "bfloat16", "shape": [128]},
        },
        count=3,
        paths=(
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::_fused_rms_norm_backward",
        {
            "grad_out": {"dtype": "bfloat16", "shape": [1, 4096, 64, 128]},
            "input": {"dtype": "bfloat16", "shape": [1, 4096, 64, 128]},
            "normalized_shape": [128],
            "output_mask": [True, True],
            "rstd": {"dtype": "float32", "shape": [1, 4096, 64, 1]},
            "weight": {"dtype": "bfloat16", "shape": [128]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::_fused_rms_norm_backward",
        {
            "grad_out": {"dtype": "bfloat16", "shape": [1, 4096, 5120]},
            "input": {"dtype": "bfloat16", "shape": [1, 4096, 5120]},
            "normalized_shape": [5120],
            "output_mask": [True, True],
            "rstd": {"dtype": "float32", "shape": [1, 4096, 1]},
            "weight": {"dtype": "bfloat16", "shape": [5120]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::_grouped_mm",
        {
            "bias": None,
            "mat2": {"dtype": "bfloat16", "shape": [16, 2048, 5120]},
            "offs": {
                "dtype": "int32",
                "high": 49,
                "low": 0,
                "multiple_of": 128,
                "shape": [16],
                "sorted": True,
            },
            "out_dtype": None,
            "self": {"dtype": "bfloat16", "shape": [6144, 2048]},
        },
        count=1,
        paths=("text-generation-torch-titan-multi-node-4N-async-hostcc",),
    )
    t(
        "aten::_grouped_mm",
        {
            "bias": None,
            "mat2": {"dtype": "bfloat16", "shape": [16, 5120, 2048]},
            "offs": {
                "dtype": "int32",
                "high": 49,
                "low": 0,
                "multiple_of": 128,
                "shape": [16],
                "sorted": True,
            },
            "out_dtype": None,
            "self": {"dtype": "bfloat16", "shape": [6144, 5120]},
        },
        count=1,
        paths=("text-generation-torch-titan-multi-node-4N-async-hostcc",),
    )
    t(
        "aten::_grouped_mm",
        {
            "bias": None,
            "mat2": {"dtype": "bfloat16", "shape": [128, 2048, 192]},
            "offs": {
                "dtype": "int32",
                "high": 385,
                "low": 0,
                "multiple_of": 128,
                "shape": [128],
                "sorted": True,
            },
            "out_dtype": None,
            "self": {"dtype": "bfloat16", "shape": [49152, 2048]},
        },
        count=1,
        paths=("text-generation-torch-titan-multi-core-async",),
    )
    t(
        "aten::_grouped_mm",
        {
            "bias": None,
            "mat2": {"dtype": "bfloat16", "shape": [64, 2048, 352]},
            "offs": {
                "dtype": "int32",
                "high": 257,
                "low": 0,
                "multiple_of": 128,
                "shape": [64],
                "sorted": True,
            },
            "out_dtype": None,
            "self": {"dtype": "bfloat16", "shape": [32768, 2048]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::_grouped_mm",
        {
            "bias": None,
            "mat2": {"dtype": "bfloat16", "shape": [128, 192, 2048]},
            "offs": {
                "dtype": "int32",
                "high": 385,
                "low": 0,
                "multiple_of": 128,
                "shape": [128],
                "sorted": True,
            },
            "out_dtype": None,
            "self": {"dtype": "bfloat16", "shape": [49152, 192]},
        },
        count=1,
        paths=("text-generation-torch-titan-multi-core-async",),
    )
    t(
        "aten::_grouped_mm",
        {
            "bias": None,
            "mat2": {"dtype": "bfloat16", "shape": [64, 352, 2048]},
            "offs": {
                "dtype": "int32",
                "high": 257,
                "low": 0,
                "multiple_of": 128,
                "shape": [64],
                "sorted": True,
            },
            "out_dtype": None,
            "self": {"dtype": "bfloat16", "shape": [32768, 352]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::_index_put_impl_",
        {
            "accumulate": True,
            "indices": [{"dtype": "int64", "shape": [64]}],
            "self": {"dtype": "bfloat16", "shape": [8, 4096, 4088]},
            "unsafe": True,
            "values": {"dtype": "bfloat16", "shape": [64, 4096, 4088]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::_index_put_impl_",
        {
            "accumulate": True,
            "indices": [{"dtype": "int64", "shape": [64]}],
            "self": {"dtype": "bfloat16", "shape": [8, 2880, 5760]},
            "unsafe": True,
            "values": {"dtype": "bfloat16", "shape": [64, 2880, 5760]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::_index_put_impl_",
        {
            "accumulate": True,
            "indices": [{"dtype": "int64", "shape": [64]}],
            "self": {"dtype": "bfloat16", "shape": [8, 4096, 4032]},
            "unsafe": True,
            "values": {"dtype": "bfloat16", "shape": [64, 4096, 4032]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::_index_put_impl_",
        {
            "accumulate": False,
            "indices": [{"dtype": "int64", "shape": [16384]}, {"dtype": "int64", "shape": [16384]}],
            "self": {"dtype": "float32", "shape": [16384, 37984]},
            "unsafe": False,
            "values": {"dtype": "float32", "shape": [16384]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::_index_put_impl_",
        {
            "accumulate": True,
            "indices": [{"dtype": "int64", "shape": [64]}],
            "self": {"dtype": "bfloat16", "shape": [8, 3072, 3072]},
            "unsafe": True,
            "values": {"dtype": "bfloat16", "shape": [64, 3072, 3072]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::_index_put_impl_",
        {
            "accumulate": True,
            "indices": [{"dtype": "int64", "shape": [64]}],
            "self": {"dtype": "bfloat16", "shape": [8, 4088, 2048]},
            "unsafe": True,
            "values": {"dtype": "bfloat16", "shape": [64, 4088, 2048]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::_scaled_dot_product_fused_attention_overrideable",
        {
            "attn_bias": None,
            "dropout_p": 0.0,
            "is_causal": False,
            "key": {"dtype": "float32", "shape": [1, 20, 1500, 64]},
            "query": {"dtype": "float32", "shape": [1, 20, 1500, 64]},
            "return_debug_mask": False,
            "scale": 1.0,
            "value": {"dtype": "float32", "shape": [1, 20, 1500, 64]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::_scaled_dot_product_fused_attention_overrideable",
        {
            "attn_bias": None,
            "dropout_p": 0.0,
            "is_causal": False,
            "key": {"dtype": "float32", "shape": [1, 20, 1500, 64]},
            "query": {"dtype": "float32", "shape": [1, 20, 448, 64]},
            "return_debug_mask": False,
            "scale": 1.0,
            "value": {"dtype": "float32", "shape": [1, 20, 1500, 64]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::_scaled_dot_product_fused_attention_overrideable",
        {
            "attn_bias": None,
            "dropout_p": 0.0,
            "is_causal": True,
            "key": {"dtype": "float32", "shape": [1, 20, 448, 64]},
            "query": {"dtype": "float32", "shape": [1, 20, 448, 64]},
            "return_debug_mask": False,
            "scale": 1.0,
            "value": {"dtype": "float32", "shape": [1, 20, 448, 64]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::_scaled_dot_product_fused_attention_overrideable",
        {
            "attn_bias": None,
            "dropout_p": 0.0,
            "is_causal": True,
            "key": {"dtype": "bfloat16", "shape": [1, 1, 96, 128]},
            "query": {"dtype": "bfloat16", "shape": [1, 16, 96, 128]},
            "return_debug_mask": False,
            "scale": 0.0883883,
            "value": {"dtype": "bfloat16", "shape": [1, 1, 96, 128]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::_scaled_dot_product_fused_attention_overrideable",
        {
            "attn_bias": None,
            "dropout_p": 0.0,
            "is_causal": True,
            "key": {"dtype": "bfloat16", "shape": [1, 1, 96, 256]},
            "query": {"dtype": "bfloat16", "shape": [1, 8, 96, 256]},
            "return_debug_mask": False,
            "scale": 0.0625,
            "value": {"dtype": "bfloat16", "shape": [1, 1, 96, 256]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::_scaled_dot_product_fused_attention_overrideable",
        {
            "attn_bias": None,
            "dropout_p": 0.0,
            "is_causal": True,
            "key": {"dtype": "bfloat16", "shape": [1, 1, 96, 256]},
            "query": {"dtype": "bfloat16", "shape": [1, 4, 96, 256]},
            "return_debug_mask": False,
            "scale": 0.0625,
            "value": {"dtype": "bfloat16", "shape": [1, 1, 96, 256]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::_softmax",
        {"dim": -1, "half_to_float": False, "self": {"dtype": "float32", "shape": [16384, 128]}},
        count=3,
        paths=(
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3",
        ),
    )
    t(
        "aten::_softmax",
        {"dim": 1, "half_to_float": False, "self": {"dtype": "float32", "shape": [4096, 128]}},
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::_softmax",
        {"dim": 1, "half_to_float": False, "self": {"dtype": "float32", "shape": [4096, 64]}},
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::_softmax",
        {
            "dim": -1,
            "half_to_float": False,
            "self": {"dtype": "float32", "shape": [1, 128, 32, 33]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::_softmax",
        {
            "dim": -1,
            "half_to_float": False,
            "self": {"dtype": "float32", "shape": [3, 1, 8, 16, 212]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::_softmax",
        {"dim": -1, "half_to_float": False, "self": {"dtype": "float32", "shape": [1, 73, 32, 32]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::_softmax_backward_data",
        {
            "dim": -1,
            "grad_output": {"dtype": "float32", "shape": [16384, 128]},
            "input_dtype": 6,
            "output": {"dtype": "float32", "shape": [16384, 128]},
        },
        count=3,
        paths=(
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3",
        ),
    )
    t(
        "aten::_softmax_backward_data",
        {
            "dim": 1,
            "grad_output": {"dtype": "float32", "shape": [4096, 128]},
            "input_dtype": 6,
            "output": {"dtype": "float32", "shape": [4096, 128]},
        },
        count=1,
        paths=("text-generation-torch-titan-multi-core-async",),
    )
    t(
        "aten::_softmax_backward_data",
        {
            "dim": 1,
            "grad_output": {"dtype": "float32", "shape": [4096, 64]},
            "input_dtype": 6,
            "output": {"dtype": "float32", "shape": [4096, 64]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::_softmax_backward_data",
        {
            "dim": -1,
            "grad_output": {"dtype": "float32", "shape": [1, 128, 32, 33]},
            "input_dtype": 6,
            "output": {"dtype": "float32", "shape": [1, 128, 32, 33]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::_softmax_backward_data",
        {
            "dim": -1,
            "grad_output": {"dtype": "float32", "shape": [3, 1, 8, 16, 212]},
            "input_dtype": 6,
            "output": {"dtype": "float32", "shape": [3, 1, 8, 16, 212]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::_softmax_backward_data",
        {
            "dim": -1,
            "grad_output": {"dtype": "float32", "shape": [1, 73, 32, 32]},
            "input_dtype": 6,
            "output": {"dtype": "float32", "shape": [1, 73, 32, 32]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::_to_copy",
        {
            "device": None,
            "dtype": 6,
            "layout": None,
            "memory_format": None,
            "non_blocking": False,
            "pin_memory": None,
            "self": {"dtype": "bfloat16", "shape": [8192, 151936]},
        },
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::_to_copy",
        {
            "device": None,
            "dtype": 15,
            "layout": None,
            "memory_format": None,
            "non_blocking": False,
            "pin_memory": None,
            "self": {"dtype": "float32", "shape": [8192, 151936]},
        },
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::_to_copy",
        {
            "device": None,
            "dtype": 6,
            "layout": None,
            "memory_format": None,
            "non_blocking": False,
            "pin_memory": None,
            "self": {"dtype": "bfloat16", "shape": [64, 12154960]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::_to_copy",
        {
            "device": None,
            "dtype": 6,
            "layout": None,
            "memory_format": None,
            "non_blocking": False,
            "pin_memory": None,
            "self": {"dtype": "bfloat16", "shape": [64, 12154880]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::_to_copy",
        {
            "device": None,
            "dtype": 6,
            "layout": None,
            "memory_format": None,
            "non_blocking": False,
            "pin_memory": None,
            "self": {"dtype": "bfloat16", "shape": [64, 9723968]},
        },
        count=6,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::_to_copy",
        {
            "device": None,
            "dtype": 6,
            "layout": None,
            "memory_format": None,
            "non_blocking": False,
            "pin_memory": None,
            "self": {"dtype": "bfloat16", "shape": [64, 9723904]},
        },
        count=10,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::abs",
        {"self": {"dtype": "float32", "shape": [1, 1024, 640]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::abs",
        {"self": {"dtype": "bfloat16", "shape": [1000, 48]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::abs",
        {"self": {"dtype": "bfloat16", "shape": [1000, 5]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::abs",
        {"self": {"dtype": "int64", "shape": [32, 32]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::abs",
        {"self": {"dtype": "float32", "shape": [1, 448]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::abs",
        {"self": {"dtype": "float32", "shape": [1, 96]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::add.Tensor",
        {
            "alpha": 1,
            "other": {"dtype": "float32", "shape": [16384, 37984]},
            "self": {"dtype": "float32", "shape": [16384, 37984]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::add.Tensor",
        {
            "alpha": 1,
            "other": {"dtype": "float32", "shape": [8192, 37984]},
            "self": {"dtype": "float32", "shape": [8192, 37984]},
        },
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::add.Tensor",
        {
            "alpha": 1,
            "other": {"dtype": "float32", "shape": [8192, 32064]},
            "self": {"dtype": "float32", "shape": [8192, 32064]},
        },
        count=5,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
        ),
    )
    t(
        "aten::add.Tensor",
        {
            "alpha": 1,
            "other": {"dtype": "float32", "shape": [4096, 50512]},
            "self": {"dtype": "float32", "shape": [4096, 50512]},
        },
        count=1,
        paths=("text-generation-torch-titan-multi-node-4N-async-hostcc",),
    )
    t(
        "aten::add.Tensor",
        {
            "alpha": 1,
            "other": {"dtype": "float32", "shape": [4096, 37984]},
            "self": {"dtype": "float32", "shape": [4096, 37984]},
        },
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::add.Tensor",
        {
            "alpha": 1,
            "other": {"dtype": "float32", "shape": [4096, 25600]},
            "self": {"dtype": "float32", "shape": [4096, 25600]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::add_.Tensor",
        {
            "alpha": 1,
            "other": {"dtype": "bfloat16", "shape": [151936, 4096]},
            "self": {"dtype": "float32", "shape": [151936, 4096]},
        },
        count=4,
        paths=(
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::add_.Tensor",
        {
            "alpha": 1,
            "other": {"dtype": "bfloat16", "shape": [151936, 2048]},
            "self": {"dtype": "bfloat16", "shape": [151936, 2048]},
        },
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::add_.Tensor",
        {
            "alpha": 1,
            "other": {"dtype": "bfloat16", "shape": [37984, 5120]},
            "self": {"dtype": "float32", "shape": [37984, 5120]},
        },
        count=4,
        paths=(
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::add_.Tensor",
        {
            "alpha": 1,
            "other": {"dtype": "bfloat16", "shape": [37984, 4096]},
            "self": {"dtype": "float32", "shape": [37984, 4096]},
        },
        count=5,
        paths=(
            "text-generation-torch-titan-qwen3-235b-trn3",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::add_.Tensor",
        {
            "alpha": 1,
            "other": {"dtype": "bfloat16", "shape": [151936, 1024]},
            "self": {"dtype": "bfloat16", "shape": [151936, 1024]},
        },
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::add_.Tensor",
        {
            "alpha": 1,
            "other": {"dtype": "bfloat16", "shape": [49152, 2048]},
            "self": {"dtype": "bfloat16", "shape": [49152, 2048]},
        },
        count=1,
        paths=("text-generation-torch-titan-multi-core-async",),
    )
    t(
        "aten::addmm",
        {
            "alpha": 1,
            "beta": 1,
            "mat1": {"dtype": "bfloat16", "shape": [32, 768]},
            "mat2": {"dtype": "bfloat16", "shape": [768, 3072]},
            "self": {"dtype": "bfloat16", "shape": [3072]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::addmm",
        {
            "alpha": 1,
            "beta": 1,
            "mat1": {"dtype": "bfloat16", "shape": [32, 3072]},
            "mat2": {"dtype": "bfloat16", "shape": [3072, 768]},
            "self": {"dtype": "bfloat16", "shape": [768]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::addmm",
        {
            "alpha": 1,
            "beta": 1,
            "mat1": {"dtype": "bfloat16", "shape": [32, 768]},
            "mat2": {"dtype": "bfloat16", "shape": [768, 2304]},
            "self": {"dtype": "bfloat16", "shape": [2304]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::addmm",
        {
            "alpha": 1,
            "beta": 1,
            "mat1": {"dtype": "bfloat16", "shape": [32, 768]},
            "mat2": {"dtype": "bfloat16", "shape": [768, 768]},
            "self": {"dtype": "bfloat16", "shape": [768]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::all",
        {"self": {"dtype": "bool", "shape": [1, 96]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::all",
        {"self": {"dtype": "bool", "shape": [1, 65]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::all",
        {"self": {"dtype": "bool", "shape": [1, 36]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::all",
        {"self": {"dtype": "bool", "shape": [1, 32]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::all",
        {"self": {"dtype": "bool", "shape": [1]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::all",
        {"self": {"dtype": "bool", "shape": []}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::all.out",
        {
            "dim": 1,
            "keepdim": False,
            "out": {"dtype": "bool", "shape": [1, 4]},
            "self": {"dtype": "bool", "shape": [1, 36, 4]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::all.out",
        {
            "dim": -1,
            "keepdim": False,
            "out": {"dtype": "bool", "shape": [1, 36]},
            "self": {"dtype": "bool", "shape": [1, 36, 2]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::amax",
        {"dim": [1], "keepdim": True, "self": {"dtype": "float32", "shape": [16384, 37984]}},
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::amax",
        {"dim": [1], "keepdim": True, "self": {"dtype": "float32", "shape": [8192, 37984]}},
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::amax",
        {"dim": [1], "keepdim": True, "self": {"dtype": "float32", "shape": [8192, 32064]}},
        count=5,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
        ),
    )
    t(
        "aten::amax",
        {"dim": [1], "keepdim": True, "self": {"dtype": "float32", "shape": [4096, 50512]}},
        count=1,
        paths=("text-generation-torch-titan-multi-node-4N-async-hostcc",),
    )
    t(
        "aten::amax",
        {"dim": [1], "keepdim": True, "self": {"dtype": "float32", "shape": [4096, 37984]}},
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::amax",
        {"dim": [1], "keepdim": True, "self": {"dtype": "float32", "shape": [4096, 25600]}},
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::any",
        {"self": {"dtype": "bool", "shape": [8192, 5120]}},
        count=4,
        paths=(
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::any",
        {"self": {"dtype": "bool", "shape": [8192, 4096]}},
        count=1,
        paths=("text-generation-torch-titan-qwen3-235b-trn3",),
    )
    t(
        "aten::any",
        {"self": {"dtype": "bool", "shape": [8016, 4096]}},
        count=2,
        paths=(
            "text-generation-torch-titan-multi-node-4N-async-compile-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
        ),
    )
    t(
        "aten::any",
        {"self": {"dtype": "bool", "shape": [4096, 4096]}},
        count=4,
        paths=(
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::any",
        {"self": {"dtype": "bool", "shape": [2374, 5120]}},
        count=10,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::any",
        {"self": {"dtype": "bool", "shape": [2374, 4096]}},
        count=12,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-dev",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-qwen3-235b-trn3",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::arange",
        {"device": None, "dtype": 6, "end": 4096, "layout": None, "pin_memory": None},
        count=15,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::arange",
        {"device": None, "dtype": None, "end": 8192, "layout": None, "pin_memory": None},
        count=10,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-compile-trn3",
            "text-generation-torch-titan-multi-core-compile-trn3-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-compile-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
        ),
    )
    t(
        "aten::arange",
        {"device": None, "dtype": 6, "end": 8192, "layout": None, "pin_memory": None},
        count=7,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-dev",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::arange",
        {"device": None, "dtype": None, "end": 4096, "layout": None, "pin_memory": None},
        count=6,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile-moe",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
        ),
    )
    t(
        "aten::arange",
        {"device": None, "dtype": 3, "end": 128, "layout": None, "pin_memory": None},
        count=3,
        paths=(
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3",
        ),
    )
    t(
        "aten::arange",
        {"device": None, "dtype": None, "end": 2048, "layout": None, "pin_memory": None},
        count=3,
        paths=(
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3",
        ),
    )
    t(
        "aten::arange.start_out",
        {"end": 128, "out": {"dtype": "int64", "shape": [0]}, "start": 0, "step": 2},
        count=22,
        paths=(
            "test_kaizen_hf_onboarding_reduced_eager",
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-dev",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-compile-trn3",
            "text-generation-torch-titan-multi-core-compile-trn3-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-compile-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
            "text-generation-torch-titan-qwen3-235b-trn3",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::arange.start_out",
        {"end": 64, "out": {"dtype": "int64", "shape": [0]}, "start": 0, "step": 2},
        count=5,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile-moe",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::arange.start_out",
        {"end": 1024, "out": {"dtype": "int64", "shape": [0]}, "start": 0, "step": 16},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::arange.start_out",
        {"end": 12, "out": {"dtype": "float32", "shape": [0]}, "start": 0, "step": 2},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::arange.start_out",
        {"end": 32, "out": {"dtype": "int64", "shape": [0]}, "start": 0, "step": 1},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::arange.start_out",
        {"end": 64, "out": {"dtype": "int64", "shape": [0]}, "start": 0, "step": 1},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::avg_pool2d",
        {
            "ceil_mode": False,
            "count_include_pad": True,
            "divisor_override": None,
            "kernel_size": [1, 1],
            "padding": 0,
            "self": {"dtype": "float32", "shape": [1, 72, 8, 8]},
            "stride": [1, 1],
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::bitwise_and.Tensor_out",
        {
            "other": {"dtype": "bool", "shape": [1, 1, 96, 96]},
            "out": {"dtype": "bool", "shape": [1, 16, 96, 96]},
            "self": {"dtype": "bool", "shape": [1, 16, 96, 96]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::bitwise_and.Tensor_out",
        {
            "other": {"dtype": "bool", "shape": [4, 4096]},
            "out": {"dtype": "bool", "shape": [4, 4096]},
            "self": {"dtype": "bool", "shape": [4, 4096]},
        },
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3-hostcc",
        ),
    )
    t(
        "aten::bitwise_and.Tensor_out",
        {
            "other": {"dtype": "bool", "shape": [1, 1, 1, 96]},
            "out": {"dtype": "bool", "shape": [1, 1, 96, 96]},
            "self": {"dtype": "bool", "shape": [1, 1, 96, 96]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::bitwise_and.Tensor_out",
        {
            "other": {"dtype": "bool", "shape": [1, 1, 96, 1]},
            "out": {"dtype": "bool", "shape": [1, 1, 96, 96]},
            "self": {"dtype": "bool", "shape": [1, 1, 96, 96]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::bitwise_and.Tensor_out",
        {
            "other": {"dtype": "bool", "shape": [1, 1, 96, 96]},
            "out": {"dtype": "bool", "shape": [1, 1, 96, 96]},
            "self": {"dtype": "bool", "shape": [1, 1, 96, 96]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::bitwise_and.Tensor_out",
        {
            "other": {"dtype": "bool", "shape": [1, 1, 96]},
            "out": {"dtype": "bool", "shape": [1, 96, 96]},
            "self": {"dtype": "bool", "shape": [1, 96, 96]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::bitwise_not",
        {"self": {"dtype": "bool", "shape": [1, 16, 96, 96]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::bitwise_not",
        {"self": {"dtype": "bool", "shape": [256, 256]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::bitwise_not",
        {"self": {"dtype": "bool", "shape": [96, 384]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::bitwise_not",
        {"self": {"dtype": "bool", "shape": [128, 128]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::bitwise_not",
        {"self": {"dtype": "bool", "shape": [1, 1, 96, 96]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::bitwise_not",
        {"self": {"dtype": "bool", "shape": [32, 8]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::bitwise_or.Tensor_out",
        {
            "other": {"dtype": "bool", "shape": [8192, 5120]},
            "out": {"dtype": "bool", "shape": [8192, 5120]},
            "self": {"dtype": "bool", "shape": [8192, 5120]},
        },
        count=4,
        paths=(
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::bitwise_or.Tensor_out",
        {
            "other": {"dtype": "bool", "shape": [8192, 4096]},
            "out": {"dtype": "bool", "shape": [8192, 4096]},
            "self": {"dtype": "bool", "shape": [8192, 4096]},
        },
        count=1,
        paths=("text-generation-torch-titan-qwen3-235b-trn3",),
    )
    t(
        "aten::bitwise_or.Tensor_out",
        {
            "other": {"dtype": "bool", "shape": [8016, 4096]},
            "out": {"dtype": "bool", "shape": [8016, 4096]},
            "self": {"dtype": "bool", "shape": [8016, 4096]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-node-4N-async-compile-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
        ),
    )
    t(
        "aten::bitwise_or.Tensor_out",
        {
            "other": {"dtype": "bool", "shape": [4096, 4096]},
            "out": {"dtype": "bool", "shape": [4096, 4096]},
            "self": {"dtype": "bool", "shape": [4096, 4096]},
        },
        count=4,
        paths=(
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::bitwise_or.Tensor_out",
        {
            "other": {"dtype": "bool", "shape": [2374, 5120]},
            "out": {"dtype": "bool", "shape": [2374, 5120]},
            "self": {"dtype": "bool", "shape": [2374, 5120]},
        },
        count=10,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::bitwise_or.Tensor_out",
        {
            "other": {"dtype": "bool", "shape": [2374, 4096]},
            "out": {"dtype": "bool", "shape": [2374, 4096]},
            "self": {"dtype": "bool", "shape": [2374, 4096]},
        },
        count=12,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-dev",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-qwen3-235b-trn3",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::bitwise_right_shift.Tensor_Scalar",
        {"other": 0.0, "self": {"dtype": "int32", "shape": [1, 32, 32]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::bitwise_right_shift.Tensor_Scalar",
        {"other": 0.0, "self": {"dtype": "int32", "shape": [32, 32]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::bitwise_right_shift.Tensor_Scalar",
        {"other": 0.0, "self": {"dtype": "int32", "shape": []}},
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
        ),
    )
    t(
        "aten::bmm",
        {
            "mat2": {"dtype": "bfloat16", "shape": [64, 4088, 1]},
            "self": {"dtype": "bfloat16", "shape": [64, 4096, 4088]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::bmm",
        {
            "mat2": {"dtype": "bfloat16", "shape": [64, 2880, 5760]},
            "self": {"dtype": "bfloat16", "shape": [64, 1, 2880]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::bmm",
        {
            "mat2": {"dtype": "bfloat16", "shape": [64, 4032, 1]},
            "self": {"dtype": "bfloat16", "shape": [64, 4096, 4032]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::bmm",
        {
            "mat2": {"dtype": "bfloat16", "shape": [64, 3072, 1]},
            "self": {"dtype": "bfloat16", "shape": [64, 3072, 3072]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::bmm",
        {
            "mat2": {"dtype": "bfloat16", "shape": [64, 2048, 1]},
            "self": {"dtype": "bfloat16", "shape": [64, 4088, 2048]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::bmm",
        {
            "mat2": {"dtype": "bfloat16", "shape": [64, 2880, 2880]},
            "self": {"dtype": "bfloat16", "shape": [64, 1, 2880]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::cat",
        {
            "dim": 0,
            "tensors": [
                {"dtype": "bfloat16", "shape": [32768, 2048]},
                {"dtype": "bfloat16", "shape": [1, 2048]},
            ],
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::cat",
        {
            "dim": 0,
            "tensors": [
                {"dtype": "bfloat16", "shape": [24576, 2048]},
                {"dtype": "bfloat16", "shape": [1, 2048]},
            ],
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::cat",
        {
            "dim": 1,
            "tensors": [
                {"dtype": "float32", "shape": [8, 8, 128, 5120]},
                {"dtype": "float32", "shape": [8, 1, 128, 5120]},
                {"dtype": "float32", "shape": [8, 1, 128, 5120]},
            ],
        },
        count=4,
        paths=(
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::cat",
        {
            "dim": 1,
            "tensors": [
                {"dtype": "float32", "shape": [4, 16, 128, 4096]},
                {"dtype": "float32", "shape": [4, 1, 128, 4096]},
                {"dtype": "float32", "shape": [4, 1, 128, 4096]},
            ],
        },
        count=1,
        paths=("text-generation-torch-titan-qwen3-235b-trn3",),
    )
    t(
        "aten::cat",
        {
            "dim": 0,
            "tensors": [
                {"dtype": "bfloat16", "shape": [4096, 5120]},
                {"dtype": "bfloat16", "shape": [1, 5120]},
            ],
        },
        count=1,
        paths=("text-generation-torch-titan-multi-node-4N-async-hostcc",),
    )
    t(
        "aten::cat",
        {
            "dim": 1,
            "tensors": [
                {"dtype": "float32", "shape": [8, 4, 128, 4096]},
                {"dtype": "float32", "shape": [8, 1, 128, 4096]},
                {"dtype": "float32", "shape": [8, 1, 128, 4096]},
            ],
        },
        count=4,
        paths=(
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::cat.out",
        {
            "dim": 1,
            "out": {"dtype": "float32", "shape": [64, 12154960]},
            "tensors": [
                {"dtype": "bfloat16", "shape": [64, 80]},
                {"dtype": "bfloat16", "shape": [64, 12154880]},
            ],
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::cat.out",
        {
            "dim": 1,
            "out": {"dtype": "float32", "shape": [64, 12154880]},
            "tensors": [{"dtype": "bfloat16", "shape": [64, 12154880]}],
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::cat.out",
        {
            "dim": 1,
            "out": {"dtype": "float32", "shape": [64, 9723968]},
            "tensors": [
                {"dtype": "bfloat16", "shape": [64, 64]},
                {"dtype": "bfloat16", "shape": [64, 9723904]},
            ],
        },
        count=6,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::cat.out",
        {
            "dim": 1,
            "out": {"dtype": "float32", "shape": [64, 9723904]},
            "tensors": [{"dtype": "bfloat16", "shape": [64, 9723904]}],
        },
        count=10,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::cat.out",
        {
            "dim": 1,
            "out": {"dtype": "float32", "shape": [64, 9723904]},
            "tensors": [{"dtype": "float32", "shape": [64, 9723904]}],
        },
        count=4,
        paths=(
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::cat.out",
        {
            "dim": 1,
            "out": {"dtype": "float32", "shape": [64, 7864320]},
            "tensors": [
                {"dtype": "bfloat16", "shape": [64, 2621440]},
                {"dtype": "bfloat16", "shape": [64, 2621440]},
                {"dtype": "bfloat16", "shape": [64, 2621440]},
            ],
        },
        count=1,
        paths=("text-generation-torch-titan-multi-node-4N-async-hostcc",),
    )
    t(
        "aten::clamp",
        {"max": 1.0, "min": None, "self": {"dtype": "bfloat16", "shape": [1, 32, 1024]}},
        count=2,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::clamp",
        {"max": 1.0, "min": None, "self": {"dtype": "int64", "shape": [4, 4096]}},
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3-hostcc",
        ),
    )
    t(
        "aten::clamp",
        {"max": 1.0, "min": None, "self": {"dtype": "int64", "shape": [1, 4096]}},
        count=4,
        paths=(
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::clamp",
        {"max": 1.0, "min": None, "self": {"dtype": "bfloat16", "shape": [1, 32, 128]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::clamp",
        {"max": 1.0, "min": None, "self": {"dtype": "int64", "shape": [2048]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::clamp",
        {"max": 1.0, "min": None, "self": {"dtype": "int64", "shape": [1, 2048]}},
        count=1,
        paths=("text-generation-torch-titan-qwen3-235b-trn3",),
    )
    t(
        "aten::clamp.Tensor",
        {
            "max": {"dtype": "bfloat16", "shape": []},
            "min": {"dtype": "bfloat16", "shape": []},
            "self": {"dtype": "bfloat16", "shape": [1, 36, 192]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::clamp.Tensor",
        {
            "max": {"dtype": "bfloat16", "shape": []},
            "min": {"dtype": "bfloat16", "shape": []},
            "self": {"dtype": "bfloat16", "shape": [1, 36, 48]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::clamp_min",
        {"min": 0.0, "self": {"dtype": "bfloat16", "shape": [1, 32, 8, 1]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::clamp_min",
        {"min": 0.0, "self": {"dtype": "bfloat16", "shape": [1, 32, 2, 1]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::clamp_min.Tensor_out",
        {
            "min": {"dtype": "float32", "shape": []},
            "out": {"dtype": "float32", "shape": [1, 1]},
            "self": {"dtype": "float32", "shape": [1, 1]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::constant_pad_nd",
        {
            "pad": [0, 0, 0, 0, 0, -224],
            "self": {"dtype": "float32", "shape": [1, 256, 128, 128]},
            "value": 0,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::constant_pad_nd",
        {
            "pad": [0, 0, 0, 0, 0, -224],
            "self": {"dtype": "float32", "shape": [1, 256, 128, 64]},
            "value": 0,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::constant_pad_nd",
        {
            "pad": [0, 0, 0, -32],
            "self": {"dtype": "float32", "shape": [1, 64, 128, 128]},
            "value": 0,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::constant_pad_nd",
        {
            "pad": [0, 0, 0, 32],
            "self": {"dtype": "float32", "shape": [1, 64, 96, 128]},
            "value": 0.0,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::constant_pad_nd",
        {
            "pad": [0, 0, 0, -32],
            "self": {"dtype": "float32", "shape": [1, 32, 128, 128]},
            "value": 0,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::constant_pad_nd",
        {
            "pad": [0, 0, 0, 0, 0, 224],
            "self": {"dtype": "float32", "shape": [1, 32, 128, 128]},
            "value": 0.0,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::convolution",
        {
            "bias": {"dtype": "float32", "shape": [1280]},
            "dilation": [1],
            "groups": 1,
            "input": {"dtype": "float32", "shape": [1, 1280, 3000]},
            "output_padding": [0],
            "padding": [1],
            "stride": [2],
            "transposed": False,
            "weight": {"dtype": "float32", "shape": [1280, 1280, 3]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::convolution",
        {
            "bias": {"dtype": "float32", "shape": [480]},
            "dilation": [1, 1],
            "groups": 1,
            "input": {"dtype": "float32", "shape": [1, 480, 32, 25]},
            "output_padding": [0, 0],
            "padding": [1, 1],
            "stride": [2, 2],
            "transposed": False,
            "weight": {"dtype": "float32", "shape": [480, 480, 3, 3]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::convolution",
        {
            "bias": {"dtype": "float32", "shape": [480]},
            "dilation": [1, 1],
            "groups": 1,
            "input": {"dtype": "float32", "shape": [1, 480, 64, 50]},
            "output_padding": [0, 0],
            "padding": [1, 1],
            "stride": [2, 2],
            "transposed": False,
            "weight": {"dtype": "float32", "shape": [480, 480, 3, 3]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::convolution",
        {
            "bias": {"dtype": "bfloat16", "shape": [1024]},
            "dilation": [1, 1, 1],
            "groups": 1,
            "input": {"dtype": "bfloat16", "shape": [256, 3, 2, 16, 16]},
            "output_padding": [0, 0, 0],
            "padding": [0, 0, 0],
            "stride": [2, 16, 16],
            "transposed": False,
            "weight": {"dtype": "bfloat16", "shape": [1024, 3, 2, 16, 16]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::convolution",
        {
            "bias": {"dtype": "bfloat16", "shape": [1152]},
            "dilation": [1, 1],
            "groups": 1,
            "input": {"dtype": "bfloat16", "shape": [1, 3, 224, 224]},
            "output_padding": [0],
            "padding": [0],
            "stride": [14, 14],
            "transposed": False,
            "weight": {"dtype": "bfloat16", "shape": [1152, 3, 14, 14]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::convolution",
        {
            "bias": {"dtype": "bfloat16", "shape": [1152]},
            "dilation": [1, 1],
            "groups": 1,
            "input": {"dtype": "bfloat16", "shape": [64, 3, 14, 14]},
            "output_padding": [0, 0],
            "padding": [0, 0],
            "stride": [14, 14],
            "transposed": False,
            "weight": {"dtype": "bfloat16", "shape": [1152, 3, 14, 14]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::convolution_backward",
        {
            "bias_sizes": [1280],
            "dilation": [1],
            "grad_output": {"dtype": "float32", "shape": [1, 1280, 1500]},
            "groups": 1,
            "input": {"dtype": "float32", "shape": [1, 1280, 3000]},
            "output_mask": [True, True, True],
            "output_padding": [0],
            "padding": [1],
            "stride": [2],
            "transposed": False,
            "weight": {"dtype": "float32", "shape": [1280, 1280, 3]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::convolution_backward",
        {
            "bias_sizes": [1280],
            "dilation": [1],
            "grad_output": {"dtype": "float32", "shape": [1, 1280, 3000]},
            "groups": 1,
            "input": {"dtype": "float32", "shape": [1, 80, 3000]},
            "output_mask": [False, True, True],
            "output_padding": [0],
            "padding": [1],
            "stride": [1],
            "transposed": False,
            "weight": {"dtype": "float32", "shape": [1280, 80, 3]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::convolution_backward",
        {
            "bias_sizes": [480],
            "dilation": [1, 1],
            "grad_output": {"dtype": "float32", "shape": [1, 480, 16, 13]},
            "groups": 1,
            "input": {"dtype": "float32", "shape": [1, 480, 32, 25]},
            "output_mask": [True, True, True],
            "output_padding": [0, 0],
            "padding": [1, 1],
            "stride": [2, 2],
            "transposed": False,
            "weight": {"dtype": "float32", "shape": [480, 480, 3, 3]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::convolution_backward",
        {
            "bias_sizes": [480],
            "dilation": [1, 1],
            "grad_output": {"dtype": "float32", "shape": [1, 480, 32, 25]},
            "groups": 1,
            "input": {"dtype": "float32", "shape": [1, 480, 64, 50]},
            "output_mask": [True, True, True],
            "output_padding": [0, 0],
            "padding": [1, 1],
            "stride": [2, 2],
            "transposed": False,
            "weight": {"dtype": "float32", "shape": [480, 480, 3, 3]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::convolution_backward",
        {
            "bias_sizes": [1024],
            "dilation": [1, 1, 1],
            "grad_output": {"dtype": "bfloat16", "shape": [256, 1024, 1, 1, 1]},
            "groups": 1,
            "input": {"dtype": "bfloat16", "shape": [256, 3, 2, 16, 16]},
            "output_mask": [False, True, True],
            "output_padding": [0, 0, 0],
            "padding": [0, 0, 0],
            "stride": [2, 16, 16],
            "transposed": False,
            "weight": {"dtype": "bfloat16", "shape": [1024, 3, 2, 16, 16]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::convolution_backward",
        {
            "bias_sizes": [480],
            "dilation": [1, 1],
            "grad_output": {"dtype": "float32", "shape": [1, 480, 64, 50]},
            "groups": 1,
            "input": {"dtype": "float32", "shape": [1, 1, 128, 100]},
            "output_mask": [False, True, True],
            "output_padding": [0, 0],
            "padding": [1, 1],
            "stride": [2, 2],
            "transposed": False,
            "weight": {"dtype": "float32", "shape": [480, 1, 3, 3]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::copy_",
        {
            "non_blocking": False,
            "self": {"dtype": "float32", "shape": [64, 12154960]},
            "src": {"dtype": "float32", "shape": [64, 12154960]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::copy_",
        {
            "non_blocking": False,
            "self": {"dtype": "float32", "shape": [64, 12154880]},
            "src": {"dtype": "float32", "shape": [64, 12154880]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::copy_",
        {
            "non_blocking": False,
            "self": {"dtype": "float32", "shape": [64, 9723968]},
            "src": {"dtype": "float32", "shape": [64, 9723968]},
        },
        count=6,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::copy_",
        {
            "non_blocking": False,
            "self": {"dtype": "float32", "shape": [64, 9723904]},
            "src": {"dtype": "float32", "shape": [64, 9723904]},
        },
        count=10,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::copy_",
        {
            "non_blocking": False,
            "self": {"dtype": "float32", "shape": [64, 7864320]},
            "src": {"dtype": "float32", "shape": [64, 7864320]},
        },
        count=1,
        paths=("text-generation-torch-titan-multi-node-4N-async-hostcc",),
    )
    t(
        "aten::copy_",
        {
            "non_blocking": False,
            "self": {"dtype": "float32", "shape": [64, 7618724]},
            "src": {"dtype": "float32", "shape": [64, 7618724]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::cos",
        {"self": {"dtype": "float32", "shape": [8192, 128]}},
        count=7,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-dev",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::cos",
        {"self": {"dtype": "float32", "shape": [4096, 128]}},
        count=15,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::cos",
        {"self": {"dtype": "float32", "shape": [4096, 64]}},
        count=5,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile-moe",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::cos",
        {"self": {"dtype": "float32", "shape": [2048, 128]}},
        count=1,
        paths=("text-generation-torch-titan-qwen3-235b-trn3",),
    )
    t(
        "aten::cos",
        {"self": {"dtype": "float32", "shape": [1, 256, 512]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::cos",
        {"self": {"dtype": "float32", "shape": [1, 256, 256]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::cross_entropy_loss",
        {
            "ignore_index": -100,
            "label_smoothing": 0.0,
            "reduction": 2,
            "self": {"dtype": "float32", "shape": [8192, 151936]},
            "target": {"dtype": "int64", "shape": [8192]},
            "weight": None,
        },
        count=5,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::cross_entropy_loss",
        {
            "ignore_index": -100,
            "label_smoothing": 0.0,
            "reduction": 2,
            "self": {"dtype": "float32", "shape": [4096, 151936]},
            "target": {"dtype": "int64", "shape": [4096]},
            "weight": None,
        },
        count=5,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::cross_entropy_loss",
        {
            "ignore_index": -100,
            "label_smoothing": 0.0,
            "reduction": 2,
            "self": {"dtype": "float32", "shape": [512, 151936]},
            "target": {"dtype": "int64", "shape": [512]},
            "weight": None,
        },
        count=2,
        paths=(
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::cross_entropy_loss",
        {
            "ignore_index": -100,
            "label_smoothing": 0.0,
            "reduction": 1,
            "self": {"dtype": "float32", "shape": [448, 30000]},
            "target": {"dtype": "int64", "shape": [448]},
            "weight": None,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::cross_entropy_loss",
        {
            "ignore_index": -100,
            "label_smoothing": 0.0,
            "reduction": 1,
            "self": {"dtype": "float32", "shape": [32, 30000]},
            "target": {"dtype": "int64", "shape": [32]},
            "weight": None,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::cross_entropy_loss",
        {
            "ignore_index": -100,
            "label_smoothing": 0.0,
            "reduction": 1,
            "self": {"dtype": "float32", "shape": [256, 3000]},
            "target": {"dtype": "int64", "shape": [256]},
            "weight": None,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::cumsum",
        {"dim": -2, "dtype": None, "self": {"dtype": "float32", "shape": [1, 128, 1, 256, 256]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::cumsum",
        {"dim": -2, "dtype": None, "self": {"dtype": "float32", "shape": [1, 16, 1, 128, 128]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::cumsum",
        {"dim": -2, "dtype": None, "self": {"dtype": "float32", "shape": [1, 8, 1, 128, 128]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::cumsum",
        {"dim": -1, "dtype": None, "self": {"dtype": "float32", "shape": [1, 128, 1, 256]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::cumsum",
        {"dim": -1, "dtype": None, "self": {"dtype": "float32", "shape": [1, 64, 2, 64]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::cumsum",
        {"dim": -1, "dtype": 6, "self": {"dtype": "float32", "shape": [1, 64, 2, 64]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::div.out",
        {
            "other": {"dtype": "int64", "shape": []},
            "out": {"dtype": "bfloat16", "shape": [16384, 2048]},
            "self": {"dtype": "bfloat16", "shape": [16384, 2048]},
        },
        count=3,
        paths=(
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3",
        ),
    )
    t(
        "aten::div.out",
        {
            "other": {"dtype": "float64", "shape": []},
            "out": {"dtype": "float32", "shape": [18874368]},
            "self": {"dtype": "float32", "shape": [18874368]},
        },
        count=1,
        paths=("text-generation-torch-titan-qwen3-235b-trn3",),
    )
    t(
        "aten::div.out",
        {
            "other": {"dtype": "float32", "shape": [16384, 1]},
            "out": {"dtype": "float32", "shape": [16384, 128]},
            "self": {"dtype": "float32", "shape": [16384, 128]},
        },
        count=3,
        paths=(
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3",
        ),
    )
    t(
        "aten::div.out",
        {
            "other": {"dtype": "int64", "shape": []},
            "out": {"dtype": "bfloat16", "shape": [1, 4096, 512]},
            "self": {"dtype": "bfloat16", "shape": [1, 4096, 512]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::div.out",
        {
            "other": {"dtype": "float64", "shape": []},
            "out": {"dtype": "float32", "shape": [1, 256, 3000]},
            "self": {"dtype": "float32", "shape": [1, 256, 3000]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::div.out",
        {
            "other": {"dtype": "float64", "shape": []},
            "out": {"dtype": "bfloat16", "shape": [1, 96, 3000]},
            "self": {"dtype": "bfloat16", "shape": [1, 96, 3000]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::div.out_mode",
        {
            "other": {"dtype": "int64", "shape": []},
            "out": {"dtype": "int64", "shape": [1, 36, 2]},
            "rounding_mode": "floor",
            "self": {"dtype": "int64", "shape": [1, 36, 2]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::div.out_mode",
        {
            "other": {"dtype": "int64", "shape": [64]},
            "out": {"dtype": "int64", "shape": [64]},
            "rounding_mode": "floor",
            "self": {"dtype": "int64", "shape": [64]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::div_.Tensor",
        {
            "other": {"dtype": "float32", "shape": [16384, 1]},
            "self": {"dtype": "float32", "shape": [16384, 128]},
        },
        count=3,
        paths=(
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3",
        ),
    )
    t(
        "aten::div_.Tensor",
        {
            "other": {"dtype": "float32", "shape": [256, 1]},
            "self": {"dtype": "float32", "shape": [256, 8]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::div_.Tensor",
        {
            "other": {"dtype": "float32", "shape": [96, 1]},
            "self": {"dtype": "float32", "shape": [96, 10]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::div_.Tensor",
        {
            "other": {"dtype": "float32", "shape": [96, 1]},
            "self": {"dtype": "bfloat16", "shape": [96, 8]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::div_.Tensor",
        {
            "other": {"dtype": "float32", "shape": [96, 1]},
            "self": {"dtype": "float32", "shape": [96, 8]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::div_.Tensor",
        {
            "other": {"dtype": "float32", "shape": [96, 1]},
            "self": {"dtype": "float32", "shape": [96, 4]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::embedding",
        {
            "indices": {"dtype": "int64", "shape": [1, 4096]},
            "padding_idx": -1,
            "scale_grad_by_freq": False,
            "sparse": False,
            "weight": {"dtype": "bfloat16", "shape": [151936, 5120]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::embedding",
        {
            "indices": {"dtype": "int64", "shape": [1, 8192]},
            "padding_idx": -1,
            "scale_grad_by_freq": False,
            "sparse": False,
            "weight": {"dtype": "bfloat16", "shape": [151936, 5120]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::embedding",
        {
            "indices": {"dtype": "int64", "shape": [1, 4096]},
            "padding_idx": -1,
            "scale_grad_by_freq": False,
            "sparse": False,
            "weight": {"dtype": "bfloat16", "shape": [151936, 4096]},
        },
        count=10,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::embedding",
        {
            "indices": {"dtype": "int64", "shape": [1, 8192]},
            "padding_idx": -1,
            "scale_grad_by_freq": False,
            "sparse": False,
            "weight": {"dtype": "bfloat16", "shape": [151936, 4096]},
        },
        count=6,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::embedding",
        {
            "indices": {"dtype": "int64", "shape": [1, 4096]},
            "padding_idx": -1,
            "scale_grad_by_freq": False,
            "sparse": False,
            "weight": {"dtype": "bfloat16", "shape": [151936, 2048]},
        },
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::embedding",
        {
            "indices": {"dtype": "int64", "shape": [1, 8192]},
            "padding_idx": -1,
            "scale_grad_by_freq": False,
            "sparse": False,
            "weight": {"dtype": "bfloat16", "shape": [151936, 2048]},
        },
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::embedding_dense_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [1, 8192, 8192]},
            "indices": {"dtype": "int64", "shape": [1, 8192]},
            "num_weights": 128256,
            "padding_idx": -1,
            "scale_grad_by_freq": False,
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-node-4N-async-compile-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
        ),
    )
    t(
        "aten::embedding_dense_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [1, 8192, 5120]},
            "indices": {"dtype": "int64", "shape": [1, 8192]},
            "num_weights": 151936,
            "padding_idx": -1,
            "scale_grad_by_freq": False,
        },
        count=6,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::embedding_dense_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [2, 8192, 2560]},
            "indices": {"dtype": "int64", "shape": [2, 8192]},
            "num_weights": 151936,
            "padding_idx": -1,
            "scale_grad_by_freq": False,
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
        ),
    )
    t(
        "aten::embedding_dense_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [1, 8192, 4096]},
            "indices": {"dtype": "int64", "shape": [1, 8192]},
            "num_weights": 128256,
            "padding_idx": -1,
            "scale_grad_by_freq": False,
        },
        count=8,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-compile-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
        ),
    )
    t(
        "aten::embedding_dense_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [1, 8192, 4096]},
            "indices": {"dtype": "int64", "shape": [1, 8192]},
            "num_weights": 151936,
            "padding_idx": -1,
            "scale_grad_by_freq": False,
        },
        count=7,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-dev",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::embedding_dense_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [4, 4096, 2048]},
            "indices": {"dtype": "int64", "shape": [4, 4096]},
            "num_weights": 37984,
            "padding_idx": -1,
            "scale_grad_by_freq": False,
        },
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3-hostcc",
        ),
    )
    t(
        "aten::eq.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [8192]},
            "self": {"dtype": "int64", "shape": [8192]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
        ),
    )
    t(
        "aten::eq.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [4096]},
            "self": {"dtype": "int64", "shape": [4096]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
        ),
    )
    t(
        "aten::eq.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [768]},
            "self": {"dtype": "int64", "shape": [768]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::eq.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [1, 448]},
            "self": {"dtype": "float32", "shape": [1, 448]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::eq.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [1, 4, 96, 1]},
            "self": {"dtype": "bfloat16", "shape": [1, 4, 96, 1]},
        },
        count=2,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::eq.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [1, 264]},
            "self": {"dtype": "int64", "shape": [1, 264]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::eq.Tensor_out",
        {
            "other": {"dtype": "int64", "shape": [1, 1, 1, 96]},
            "out": {"dtype": "bool", "shape": [1, 1, 96, 96]},
            "self": {"dtype": "int64", "shape": [1, 1, 96, 1]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::eq.Tensor_out",
        {
            "other": {"dtype": "int64", "shape": [1, 1, 96]},
            "out": {"dtype": "bool", "shape": [1, 96, 96]},
            "self": {"dtype": "int64", "shape": [1, 96, 1]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::exp",
        {"self": {"dtype": "float32", "shape": [16384, 37984]}},
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::exp",
        {"self": {"dtype": "float32", "shape": [8192, 37984]}},
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::exp",
        {"self": {"dtype": "float32", "shape": [8192, 32064]}},
        count=5,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
        ),
    )
    t(
        "aten::exp",
        {"self": {"dtype": "float32", "shape": [4096, 50512]}},
        count=1,
        paths=("text-generation-torch-titan-multi-node-4N-async-hostcc",),
    )
    t(
        "aten::exp",
        {"self": {"dtype": "float32", "shape": [4096, 37984]}},
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::exp",
        {"self": {"dtype": "float32", "shape": [4096, 25600]}},
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::exp.out",
        {
            "out": {"dtype": "float32", "shape": [1, 448, 30000]},
            "self": {"dtype": "float32", "shape": [1, 448, 30000]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::exp.out",
        {
            "out": {"dtype": "float32", "shape": [1, 32, 30000]},
            "self": {"dtype": "float32", "shape": [1, 32, 30000]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::exp.out",
        {
            "out": {"dtype": "float32", "shape": [1, 16, 30000]},
            "self": {"dtype": "float32", "shape": [1, 16, 30000]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::exp.out",
        {
            "out": {"dtype": "float32", "shape": [1, 96, 3000]},
            "self": {"dtype": "float32", "shape": [1, 96, 3000]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::exp.out",
        {
            "out": {"dtype": "float32", "shape": [1, 32, 3000]},
            "self": {"dtype": "float32", "shape": [1, 32, 3000]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::eye",
        {"device": None, "dtype": 6, "layout": None, "n": 64, "pin_memory": None},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::fill_.Scalar",
        {"self": {"dtype": "bfloat16", "shape": [1, 8, 8, 512]}, "value": float("-inf")},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::fill_.Scalar",
        {"self": {"dtype": "bfloat16", "shape": [1, 8, 8, 128]}, "value": float("-inf")},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::fill_.Scalar",
        {"self": {"dtype": "float32", "shape": [1024]}, "value": 1.0},
        count=2,
        paths=(
            "text-generation-torch-titan-multi-node-4N-async-compile-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
        ),
    )
    t(
        "aten::fill_.Scalar",
        {"self": {"dtype": "bool", "shape": [1, 32, 32]}, "value": 1.0},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::fill_.Scalar",
        {"self": {"dtype": "int64", "shape": [32, 32]}, "value": 15},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::fill_.Scalar",
        {"self": {"dtype": "bfloat16", "shape": [1, 4, 96, 2]}, "value": float("-inf")},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::flip",
        {"dims": [-2], "self": {"dtype": "float32", "shape": [1, 128, 1, 256, 256]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::flip",
        {"dims": [-2], "self": {"dtype": "float32", "shape": [1, 16, 1, 128, 128]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::flip",
        {"dims": [-2], "self": {"dtype": "float32", "shape": [1, 8, 1, 128, 128]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::flip",
        {"dims": [-1], "self": {"dtype": "float32", "shape": [1, 128, 1, 256]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::flip",
        {"dims": [-1], "self": {"dtype": "float32", "shape": [1, 64, 2, 64]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::flip",
        {"dims": [-1], "self": {"dtype": "float32", "shape": [1, 32, 2, 64]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::floor_divide",
        {"other": {"dtype": "int64", "shape": []}, "self": {"dtype": "float32", "shape": [262144]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::floor_divide",
        {"other": {"dtype": "int64", "shape": []}, "self": {"dtype": "int64", "shape": [32768]}},
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::floor_divide",
        {"other": {"dtype": "int64", "shape": []}, "self": {"dtype": "int64", "shape": [24576]}},
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::floor_divide",
        {"other": {"dtype": "int64", "shape": []}, "self": {"dtype": "int64", "shape": [4096]}},
        count=1,
        paths=("text-generation-torch-titan-multi-node-4N-async-hostcc",),
    )
    t(
        "aten::floor_divide",
        {"other": {"dtype": "int64", "shape": []}, "self": {"dtype": "int32", "shape": [128]}},
        count=3,
        paths=(
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3",
        ),
    )
    t(
        "aten::floor_divide",
        {"other": {"dtype": "int64", "shape": []}, "self": {"dtype": "int64", "shape": [1, 96]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::full",
        {
            "device": None,
            "dtype": None,
            "fill_value": 0.0,
            "layout": None,
            "pin_memory": None,
            "size": [1, 96],
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::gather",
        {
            "dim": 1,
            "index": {"dtype": "int64", "shape": [16384, 1]},
            "self": {"dtype": "float32", "shape": [16384, 37984]},
            "sparse_grad": False,
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::gather",
        {
            "dim": 1,
            "index": {"dtype": "int64", "shape": [8192, 1]},
            "self": {"dtype": "float32", "shape": [8192, 37984]},
            "sparse_grad": False,
        },
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::gather",
        {
            "dim": 1,
            "index": {"dtype": "int64", "shape": [8192, 1]},
            "self": {"dtype": "float32", "shape": [8192, 32064]},
            "sparse_grad": False,
        },
        count=5,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
        ),
    )
    t(
        "aten::gather",
        {
            "dim": 1,
            "index": {"dtype": "int64", "shape": [4096, 1]},
            "self": {"dtype": "float32", "shape": [4096, 50512]},
            "sparse_grad": False,
        },
        count=1,
        paths=("text-generation-torch-titan-multi-node-4N-async-hostcc",),
    )
    t(
        "aten::gather",
        {
            "dim": 1,
            "index": {"dtype": "int64", "shape": [4096, 1]},
            "self": {"dtype": "float32", "shape": [4096, 37984]},
            "sparse_grad": False,
        },
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::gather",
        {
            "dim": 1,
            "index": {"dtype": "int64", "shape": [4096, 1]},
            "self": {"dtype": "float32", "shape": [4096, 25600]},
            "sparse_grad": False,
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::ge.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [1, 96, 12288]},
            "self": {"dtype": "bfloat16", "shape": [1, 96, 12288]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::ge.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [96, 3072]},
            "self": {"dtype": "bfloat16", "shape": [96, 3072]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::ge.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [64, 2880]},
            "self": {"dtype": "bfloat16", "shape": [64, 2880]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::ge.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [64, 1024]},
            "self": {"dtype": "bfloat16", "shape": [64, 1024]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::ge.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [1, 32, 1024]},
            "self": {"dtype": "bfloat16", "shape": [1, 32, 1024]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::ge.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [384, 48]},
            "self": {"dtype": "bfloat16", "shape": [384, 48]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::ge.Tensor_out",
        {
            "other": {"dtype": "bfloat16", "shape": []},
            "out": {"dtype": "bool", "shape": [1, 36, 192]},
            "self": {"dtype": "bfloat16", "shape": [1, 36, 192]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::ge.Tensor_out",
        {
            "other": {"dtype": "bfloat16", "shape": []},
            "out": {"dtype": "bool", "shape": [1, 36, 48]},
            "self": {"dtype": "bfloat16", "shape": [1, 36, 48]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::ge.Tensor_out",
        {
            "other": {"dtype": "int64", "shape": [1, 32, 1]},
            "out": {"dtype": "bool", "shape": [1, 32, 8]},
            "self": {"dtype": "int64", "shape": [1, 1, 8]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::ge.Tensor_out",
        {
            "other": {"dtype": "int64", "shape": [1, 32, 1]},
            "out": {"dtype": "bool", "shape": [1, 32, 8]},
            "self": {"dtype": "int64", "shape": [1, 32, 8]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::ge.Tensor_out",
        {
            "other": {"dtype": "int64", "shape": [1, 16]},
            "out": {"dtype": "bool", "shape": [16, 16]},
            "self": {"dtype": "int64", "shape": [16, 1]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::gelu",
        {"approximate": "none", "self": {"dtype": "float32", "shape": [1, 1500, 5120]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::gelu",
        {"approximate": "none", "self": {"dtype": "float32", "shape": [1, 1280, 3000]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::gelu",
        {"approximate": "none", "self": {"dtype": "float32", "shape": [1, 448, 5120]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::gelu",
        {"approximate": "tanh", "self": {"dtype": "bfloat16", "shape": [1, 264, 8192]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::gelu",
        {"approximate": "none", "self": {"dtype": "bfloat16", "shape": [1, 1025, 2048]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::gelu",
        {"approximate": "none", "self": {"dtype": "float32", "shape": [1, 1280, 1500]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::gelu_backward",
        {
            "approximate": "none",
            "grad_output": {"dtype": "float32", "shape": [1, 1500, 5120]},
            "self": {"dtype": "float32", "shape": [1, 1500, 5120]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::gelu_backward",
        {
            "approximate": "none",
            "grad_output": {"dtype": "float32", "shape": [1, 1280, 3000]},
            "self": {"dtype": "float32", "shape": [1, 1280, 3000]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::gelu_backward",
        {
            "approximate": "none",
            "grad_output": {"dtype": "float32", "shape": [1, 448, 5120]},
            "self": {"dtype": "float32", "shape": [1, 448, 5120]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::gelu_backward",
        {
            "approximate": "tanh",
            "grad_output": {"dtype": "bfloat16", "shape": [1, 264, 8192]},
            "self": {"dtype": "bfloat16", "shape": [1, 264, 8192]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::gelu_backward",
        {
            "approximate": "none",
            "grad_output": {"dtype": "bfloat16", "shape": [1, 1025, 2048]},
            "self": {"dtype": "bfloat16", "shape": [1, 1025, 2048]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::gelu_backward",
        {
            "approximate": "none",
            "grad_output": {"dtype": "float32", "shape": [1, 480, 64, 50]},
            "self": {"dtype": "float32", "shape": [1, 480, 64, 50]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::gt.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [8192, 5120]},
            "self": {"dtype": "float32", "shape": [8192, 5120]},
        },
        count=4,
        paths=(
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::gt.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [8192, 4096]},
            "self": {"dtype": "float32", "shape": [8192, 4096]},
        },
        count=1,
        paths=("text-generation-torch-titan-qwen3-235b-trn3",),
    )
    t(
        "aten::gt.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [8016, 4096]},
            "self": {"dtype": "float32", "shape": [8016, 4096]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-node-4N-async-compile-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
        ),
    )
    t(
        "aten::gt.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [4096, 4096]},
            "self": {"dtype": "float32", "shape": [4096, 4096]},
        },
        count=4,
        paths=(
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::gt.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [2374, 5120]},
            "self": {"dtype": "float32", "shape": [2374, 5120]},
        },
        count=10,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::gt.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [2374, 4096]},
            "self": {"dtype": "float32", "shape": [2374, 4096]},
        },
        count=12,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-dev",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-qwen3-235b-trn3",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::gt.Tensor_out",
        {
            "other": {"dtype": "int64", "shape": [1, 1, 96, 1]},
            "out": {"dtype": "bool", "shape": [1, 1, 96, 96]},
            "self": {"dtype": "int64", "shape": [1, 1, 1, 96]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::gt.Tensor_out",
        {
            "other": {"dtype": "int64", "shape": [96, 1]},
            "out": {"dtype": "bool", "shape": [96, 96]},
            "self": {"dtype": "int64", "shape": [96]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::gt.Tensor_out",
        {
            "other": {"dtype": "int64", "shape": [1, 1, 32, 1]},
            "out": {"dtype": "bool", "shape": [1, 1, 32, 32]},
            "self": {"dtype": "int64", "shape": [1, 1, 1, 32]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::gt.Tensor_out",
        {
            "other": {"dtype": "int64", "shape": [1]},
            "out": {"dtype": "bool", "shape": [1]},
            "self": {"dtype": "int64", "shape": [1]},
        },
        count=19,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-compile-trn3",
            "text-generation-torch-titan-multi-core-compile-trn3-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-compile-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::gt.Tensor_out",
        {
            "other": {"dtype": "int64", "shape": []},
            "out": {"dtype": "bool", "shape": []},
            "self": {"dtype": "int64", "shape": []},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::histc",
        {"bins": 128, "max": 0, "min": 0, "self": {"dtype": "int64", "shape": [32768]}},
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::histc",
        {"bins": 64, "max": 0, "min": 0, "self": {"dtype": "int64", "shape": [24576]}},
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::histc",
        {"bins": 16, "max": 0, "min": 0, "self": {"dtype": "int64", "shape": [4096]}},
        count=1,
        paths=("text-generation-torch-titan-multi-node-4N-async-hostcc",),
    )
    t(
        "aten::index.Tensor",
        {
            "indices": [{"dtype": "int64", "high": 32, "low": 0, "shape": [1]}],
            "self": {"dtype": "bfloat16", "shape": [32, 3072, 1536]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::index.Tensor",
        {
            "indices": [{"dtype": "int64", "high": 8, "low": 0, "shape": [64]}],
            "self": {"dtype": "bfloat16", "shape": [8, 4096, 4088]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::index.Tensor",
        {
            "indices": [{"dtype": "int64", "high": 8, "low": 0, "shape": [64]}],
            "self": {"dtype": "bfloat16", "shape": [8, 2880, 5760]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::index.Tensor",
        {
            "indices": [{"dtype": "int64", "high": 8, "low": 0, "shape": [64]}],
            "self": {"dtype": "bfloat16", "shape": [8, 4096, 4032]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::index.Tensor",
        {
            "indices": [{"dtype": "int64", "high": 32, "low": 0, "shape": [1]}],
            "self": {"dtype": "bfloat16", "shape": [32, 1536, 1536]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::index.Tensor",
        {
            "indices": [{"dtype": "int64", "high": 8, "low": 0, "shape": [64]}],
            "self": {"dtype": "bfloat16", "shape": [8, 3072, 3072]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::index_add_",
        {
            "alpha": 1,
            "dim": 0,
            "index": {"dtype": "int64", "shape": [10]},
            "self": {"dtype": "float32", "shape": [32, 4032]},
            "source": {"dtype": "float32", "shape": [10, 4032]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::index_add_",
        {
            "alpha": 1,
            "dim": 0,
            "index": {"dtype": "int64", "shape": [11]},
            "self": {"dtype": "float32", "shape": [32, 4032]},
            "source": {"dtype": "float32", "shape": [11, 4032]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::index_add_",
        {
            "alpha": 1,
            "dim": 0,
            "index": {"dtype": "int64", "shape": [12]},
            "self": {"dtype": "float32", "shape": [32, 4032]},
            "source": {"dtype": "float32", "shape": [12, 4032]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::index_add_",
        {
            "alpha": 1,
            "dim": 0,
            "index": {"dtype": "int64", "shape": [13]},
            "self": {"dtype": "float32", "shape": [32, 4032]},
            "source": {"dtype": "float32", "shape": [13, 4032]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::index_add_",
        {
            "alpha": 1,
            "dim": 0,
            "index": {"dtype": "int64", "shape": [14]},
            "self": {"dtype": "float32", "shape": [32, 4032]},
            "source": {"dtype": "float32", "shape": [14, 4032]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::index_add_",
        {
            "alpha": 1,
            "dim": 0,
            "index": {"dtype": "int64", "shape": [15]},
            "self": {"dtype": "float32", "shape": [32, 4032]},
            "source": {"dtype": "float32", "shape": [15, 4032]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::index_select",
        {
            "dim": 1,
            "index": {"dtype": "int64", "shape": [3]},
            "self": {"dtype": "bfloat16", "shape": [1, 10, 256, 256]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::index_select",
        {
            "dim": 0,
            "index": {"dtype": "int64", "shape": [10]},
            "self": {"dtype": "float32", "shape": [32, 4032]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::index_select",
        {
            "dim": 0,
            "index": {"dtype": "int64", "shape": [11]},
            "self": {"dtype": "float32", "shape": [32, 4032]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::index_select",
        {
            "dim": 0,
            "index": {"dtype": "int64", "shape": [12]},
            "self": {"dtype": "float32", "shape": [32, 4032]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::index_select",
        {
            "dim": 0,
            "index": {"dtype": "int64", "shape": [13]},
            "self": {"dtype": "float32", "shape": [32, 4032]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::index_select",
        {
            "dim": 0,
            "index": {"dtype": "int64", "shape": [14]},
            "self": {"dtype": "float32", "shape": [32, 4032]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::index_select_backward",
        {
            "dim": 0,
            "grad": {"dtype": "float32", "shape": [13, 1024]},
            "index": {"dtype": "int64", "shape": [13]},
            "self_sizes": [13, 1024],
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::le.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [1, 96, 12288]},
            "self": {"dtype": "bfloat16", "shape": [1, 96, 12288]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::le.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [96, 3072]},
            "self": {"dtype": "bfloat16", "shape": [96, 3072]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::le.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [64, 2880]},
            "self": {"dtype": "bfloat16", "shape": [64, 2880]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::le.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [64, 1024]},
            "self": {"dtype": "bfloat16", "shape": [64, 1024]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::le.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [1, 32, 1024]},
            "self": {"dtype": "bfloat16", "shape": [1, 32, 1024]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::le.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [384, 48]},
            "self": {"dtype": "bfloat16", "shape": [384, 48]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::le.Tensor_out",
        {
            "other": {"dtype": "int64", "shape": [1, 1, 96, 1]},
            "out": {"dtype": "bool", "shape": [1, 1, 96, 96]},
            "self": {"dtype": "int64", "shape": [1, 1, 1, 96]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::le.Tensor_out",
        {
            "other": {"dtype": "int64", "shape": [96, 1]},
            "out": {"dtype": "bool", "shape": [96, 96]},
            "self": {"dtype": "int64", "shape": [96]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::le.Tensor_out",
        {
            "other": {"dtype": "bfloat16", "shape": []},
            "out": {"dtype": "bool", "shape": [1, 36, 192]},
            "self": {"dtype": "bfloat16", "shape": [1, 36, 192]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::le.Tensor_out",
        {
            "other": {"dtype": "bfloat16", "shape": []},
            "out": {"dtype": "bool", "shape": [1, 36, 48]},
            "self": {"dtype": "bfloat16", "shape": [1, 36, 48]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::le.Tensor_out",
        {
            "other": {"dtype": "int64", "shape": [1, 1, 32, 1]},
            "out": {"dtype": "bool", "shape": [1, 1, 32, 32]},
            "self": {"dtype": "int64", "shape": [1, 1, 1, 32]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::linalg_vector_norm",
        {
            "dim": [-1],
            "dtype": None,
            "keepdim": True,
            "ord": 2,
            "self": {"dtype": "bfloat16", "shape": [1, 32, 8, 128]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::linalg_vector_norm",
        {
            "dim": [-1],
            "dtype": None,
            "keepdim": True,
            "ord": 2,
            "self": {"dtype": "bfloat16", "shape": [1, 32, 2, 128]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::linalg_vector_norm",
        {
            "dim": [1],
            "dtype": None,
            "keepdim": True,
            "ord": 2,
            "self": {"dtype": "float32", "shape": [1, 1024]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::linalg_vector_norm",
        {
            "dim": None,
            "dtype": None,
            "keepdim": False,
            "ord": 2.0,
            "self": {"dtype": "float32", "shape": [723]},
        },
        count=1,
        paths=("text-generation-torch-titan-multi-node-4N-async-hostcc",),
    )
    t(
        "aten::linalg_vector_norm",
        {
            "dim": None,
            "dtype": None,
            "keepdim": False,
            "ord": 2.0,
            "self": {"dtype": "float32", "shape": [707]},
        },
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::linalg_vector_norm",
        {
            "dim": None,
            "dtype": None,
            "keepdim": False,
            "ord": 2.0,
            "self": {"dtype": "float32", "shape": [627]},
        },
        count=1,
        paths=("text-generation-torch-titan-multi-node-4N-async-hostcc",),
    )
    t(
        "aten::linear",
        {
            "bias": None,
            "input": {"dtype": "bfloat16", "shape": [1, 4096, 5120]},
            "weight": {"dtype": "bfloat16", "shape": [151936, 5120]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::linear",
        {
            "bias": None,
            "input": {"dtype": "bfloat16", "shape": [1, 8192, 5120]},
            "weight": {"dtype": "bfloat16", "shape": [151936, 5120]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::linear",
        {
            "bias": None,
            "input": {"dtype": "bfloat16", "shape": [1, 4096, 4096]},
            "weight": {"dtype": "bfloat16", "shape": [151936, 4096]},
        },
        count=6,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::linear",
        {
            "bias": None,
            "input": {"dtype": "bfloat16", "shape": [1, 8192, 4096]},
            "weight": {"dtype": "bfloat16", "shape": [151936, 4096]},
        },
        count=6,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::linear",
        {
            "bias": None,
            "input": {"dtype": "bfloat16", "shape": [1, 512, 4096]},
            "weight": {"dtype": "bfloat16", "shape": [151936, 4096]},
        },
        count=4,
        paths=(
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::linear",
        {
            "bias": None,
            "input": {"dtype": "bfloat16", "shape": [1, 4096, 2048]},
            "weight": {"dtype": "bfloat16", "shape": [151936, 2048]},
        },
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::linear_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [1, 8192, 151936]},
            "output_mask": [True, True, False],
            "self": {"dtype": "bfloat16", "shape": [1, 8192, 4096]},
            "weight": {"dtype": "bfloat16", "shape": [151936, 4096]},
        },
        count=6,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::linear_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [1, 8192, 151936]},
            "output_mask": [True, True, False],
            "self": {"dtype": "bfloat16", "shape": [1, 8192, 1024]},
            "weight": {"dtype": "bfloat16", "shape": [151936, 1024]},
        },
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::linear_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [1, 8192, 151936]},
            "output_mask": [True, True, False],
            "self": {"dtype": "bfloat16", "shape": [1, 8192, 2048]},
            "weight": {"dtype": "bfloat16", "shape": [151936, 2048]},
        },
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::linear_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [1, 8192, 151936]},
            "output_mask": [True, True, False],
            "self": {"dtype": "bfloat16", "shape": [1, 8192, 5120]},
            "weight": {"dtype": "bfloat16", "shape": [151936, 5120]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::linear_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [1, 4096, 151936]},
            "output_mask": [True, True, False],
            "self": {"dtype": "bfloat16", "shape": [1, 4096, 5120]},
            "weight": {"dtype": "bfloat16", "shape": [151936, 5120]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::linear_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [1, 4096, 151936]},
            "output_mask": [True, True, False],
            "self": {"dtype": "bfloat16", "shape": [1, 4096, 4096]},
            "weight": {"dtype": "bfloat16", "shape": [151936, 4096]},
        },
        count=6,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::log",
        {"self": {"dtype": "float32", "shape": [16384, 1]}},
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::log",
        {"self": {"dtype": "float32", "shape": [8192, 1]}},
        count=5,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
        ),
    )
    t(
        "aten::log",
        {"self": {"dtype": "float32", "shape": [4096, 1]}},
        count=5,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
        ),
    )
    t(
        "aten::log",
        {"self": {"dtype": "float32", "shape": [2048, 1]}},
        count=3,
        paths=(
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3",
        ),
    )
    t(
        "aten::log",
        {"self": {"dtype": "float32", "shape": [32, 32]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::log",
        {"self": {"dtype": "float32", "shape": [512, 1]}},
        count=2,
        paths=(
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::log.out",
        {
            "out": {"dtype": "float32", "shape": [1, 448]},
            "self": {"dtype": "float32", "shape": [1, 448]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::log.out",
        {
            "out": {"dtype": "float32", "shape": [1, 96]},
            "self": {"dtype": "float32", "shape": [1, 96]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::log.out",
        {
            "out": {"dtype": "float32", "shape": [1, 32]},
            "self": {"dtype": "float32", "shape": [1, 32]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::log.out",
        {
            "out": {"dtype": "float32", "shape": [1, 16]},
            "self": {"dtype": "float32", "shape": [1, 16]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::logical_and.out",
        {
            "other": {"dtype": "bool", "shape": [1, 96, 12288]},
            "out": {"dtype": "bool", "shape": [1, 96, 12288]},
            "self": {"dtype": "bool", "shape": [1, 96, 12288]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::logical_and.out",
        {
            "other": {"dtype": "bool", "shape": [96, 3072]},
            "out": {"dtype": "bool", "shape": [96, 3072]},
            "self": {"dtype": "bool", "shape": [96, 3072]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::logical_and.out",
        {
            "other": {"dtype": "bool", "shape": [64, 2880]},
            "out": {"dtype": "bool", "shape": [64, 2880]},
            "self": {"dtype": "bool", "shape": [64, 2880]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::logical_and.out",
        {
            "other": {"dtype": "bool", "shape": [64, 1024]},
            "out": {"dtype": "bool", "shape": [64, 1024]},
            "self": {"dtype": "bool", "shape": [64, 1024]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::logical_and.out",
        {
            "other": {"dtype": "bool", "shape": [1, 32, 1024]},
            "out": {"dtype": "bool", "shape": [1, 32, 1024]},
            "self": {"dtype": "bool", "shape": [1, 32, 1024]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::logical_and.out",
        {
            "other": {"dtype": "bool", "shape": [384, 48]},
            "out": {"dtype": "bool", "shape": [384, 48]},
            "self": {"dtype": "bool", "shape": [384, 48]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::logical_not",
        {"self": {"dtype": "bool", "shape": [1, 4]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::lt.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [8192, 5120]},
            "self": {"dtype": "float32", "shape": [8192, 5120]},
        },
        count=4,
        paths=(
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::lt.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [8192, 4096]},
            "self": {"dtype": "float32", "shape": [8192, 4096]},
        },
        count=1,
        paths=("text-generation-torch-titan-qwen3-235b-trn3",),
    )
    t(
        "aten::lt.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [8016, 4096]},
            "self": {"dtype": "float32", "shape": [8016, 4096]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-node-4N-async-compile-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
        ),
    )
    t(
        "aten::lt.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [4096, 4096]},
            "self": {"dtype": "float32", "shape": [4096, 4096]},
        },
        count=4,
        paths=(
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::lt.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [2374, 5120]},
            "self": {"dtype": "float32", "shape": [2374, 5120]},
        },
        count=10,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::lt.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [2374, 4096]},
            "self": {"dtype": "float32", "shape": [2374, 4096]},
        },
        count=12,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-dev",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-qwen3-235b-trn3",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::lt.Tensor_out",
        {
            "other": {"dtype": "int32", "shape": [127]},
            "out": {"dtype": "bool", "shape": [127]},
            "self": {"dtype": "int32", "shape": [127]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::lt.Tensor_out",
        {
            "other": {"dtype": "int32", "shape": [63]},
            "out": {"dtype": "bool", "shape": [63]},
            "self": {"dtype": "int32", "shape": [63]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::lt.Tensor_out",
        {
            "other": {"dtype": "int32", "shape": [15]},
            "out": {"dtype": "bool", "shape": [15]},
            "self": {"dtype": "int32", "shape": [15]},
        },
        count=1,
        paths=("text-generation-torch-titan-multi-node-4N-async-hostcc",),
    )
    t(
        "aten::lt.Tensor_out",
        {
            "other": {"dtype": "int64", "shape": [1, 1]},
            "out": {"dtype": "bool", "shape": [1, 13]},
            "self": {"dtype": "int64", "shape": [13]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::masked_fill.Scalar",
        {
            "mask": {"dtype": "bool", "shape": [256, 256]},
            "self": {"dtype": "float32", "shape": [1, 128, 1, 256, 256]},
            "value": 0.0,
        },
        count=2,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::masked_fill.Scalar",
        {
            "mask": {"dtype": "bool", "shape": [64, 64]},
            "self": {"dtype": "float32", "shape": [1, 64, 2, 64, 64]},
            "value": 0.0,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::masked_fill.Scalar",
        {
            "mask": {"dtype": "bool", "shape": [128, 128]},
            "self": {"dtype": "float32", "shape": [1, 16, 1, 128, 128]},
            "value": 0.0,
        },
        count=2,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::masked_fill.Scalar",
        {
            "mask": {"dtype": "bool", "shape": [64, 64]},
            "self": {"dtype": "float32", "shape": [1, 32, 2, 64, 64]},
            "value": 0.0,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::masked_fill.Scalar",
        {
            "mask": {"dtype": "bool", "shape": [1, 96, 1]},
            "self": {"dtype": "bfloat16", "shape": [1, 96, 2048]},
            "value": 0.0,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::masked_fill.Scalar",
        {
            "mask": {"dtype": "bool", "shape": [1, 16, 96, 96]},
            "self": {"dtype": "bfloat16", "shape": [1, 16, 96, 96]},
            "value": 0.0,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::masked_fill_.Scalar",
        {
            "mask": {"dtype": "bool", "shape": [1, 32, 8, 1]},
            "self": {"dtype": "bfloat16", "shape": [1, 32, 8, 128]},
            "value": 0.0,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::masked_fill_.Scalar",
        {
            "mask": {"dtype": "bool", "shape": [1, 32, 2, 1]},
            "self": {"dtype": "bfloat16", "shape": [1, 32, 2, 128]},
            "value": 0.0,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::masked_fill_.Scalar",
        {
            "mask": {"dtype": "bool", "shape": [1, 1]},
            "self": {"dtype": "float32", "shape": [1, 1024]},
            "value": 0.0,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::masked_fill_.Scalar",
        {
            "mask": {"dtype": "bool", "shape": [1, 448]},
            "self": {"dtype": "float32", "shape": [1, 448]},
            "value": 0.0,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::masked_fill_.Scalar",
        {
            "mask": {"dtype": "bool", "shape": [1, 96]},
            "self": {"dtype": "float32", "shape": [1, 96]},
            "value": 0.0,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::masked_fill_.Scalar",
        {
            "mask": {"dtype": "bool", "shape": [1, 32]},
            "self": {"dtype": "float32", "shape": [1, 32]},
            "value": 0.0,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::masked_select",
        {
            "mask": {"dtype": "bool", "shape": [1, 264, 1]},
            "self": {"dtype": "bfloat16", "shape": [1, 264, 2048]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::masked_select",
        {
            "mask": {"dtype": "bool", "shape": [1, 96, 1]},
            "self": {"dtype": "bfloat16", "shape": [1, 96, 2048]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::masked_select",
        {
            "mask": {"dtype": "bool", "shape": [1, 96, 1]},
            "self": {"dtype": "bfloat16", "shape": [1, 96, 640]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::masked_select",
        {
            "mask": {"dtype": "bool", "shape": [1, 96, 1]},
            "self": {"dtype": "bfloat16", "shape": [1, 96, 512]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::masked_select",
        {
            "mask": {"dtype": "bool", "shape": [1, 96, 1]},
            "self": {"dtype": "bfloat16", "shape": [1, 96, 448]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::masked_select",
        {
            "mask": {"dtype": "bool", "shape": [1, 96, 1]},
            "self": {"dtype": "bfloat16", "shape": [1, 96, 336]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::max",
        {"self": {"dtype": "int64", "shape": [8192]}},
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
        ),
    )
    t(
        "aten::max",
        {"self": {"dtype": "int64", "shape": [4096]}},
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
        ),
    )
    t(
        "aten::max",
        {"self": {"dtype": "int64", "shape": [3, 96]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::max",
        {"self": {"dtype": "int64", "shape": [64, 2]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::max",
        {"self": {"dtype": "int64", "shape": [1]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::max.dim",
        {"dim": -1, "keepdim": True, "self": {"dtype": "bfloat16", "shape": [1, 128, 32, 33]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::max.dim",
        {"dim": -1, "keepdim": True, "self": {"dtype": "bfloat16", "shape": [1, 64, 32, 33]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::max.dim",
        {"dim": -1, "keepdim": True, "self": {"dtype": "bfloat16", "shape": [1, 21, 32, 32]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::max.dim",
        {"dim": -1, "keepdim": True, "self": {"dtype": "bfloat16", "shape": [1, 4, 32, 41]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::max.dim",
        {"dim": -1, "keepdim": True, "self": {"dtype": "bfloat16", "shape": [1, 4, 32, 33]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::max.dim",
        {"dim": -1, "keepdim": True, "self": {"dtype": "int64", "shape": [1, 36]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::mean",
        {"dtype": None, "self": {"dtype": "float32", "shape": [1, 448]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::mean",
        {"dtype": None, "self": {"dtype": "float32", "shape": [1, 96]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::mean",
        {"dtype": None, "self": {"dtype": "float32", "shape": [64]}},
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile-moe",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-qwen3-235b-trn3",
        ),
    )
    t(
        "aten::mean",
        {"dtype": None, "self": {"dtype": "float32", "shape": [32]}},
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async-compile-moe",
            "text-generation-torch-titan-multi-core-trn3",
        ),
    )
    t(
        "aten::mean",
        {"dtype": None, "self": {"dtype": "float32", "shape": [1, 32]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::mean",
        {"dtype": None, "self": {"dtype": "float32", "shape": [1, 16]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::mean.dim",
        {
            "dim": [-1],
            "dtype": None,
            "keepdim": True,
            "self": {"dtype": "float32", "shape": [6144, 128]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::mean.dim",
        {
            "dim": [-1],
            "dtype": None,
            "keepdim": False,
            "self": {"dtype": "float32", "shape": [1, 1024, 640]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::mean.dim",
        {
            "dim": [-1],
            "dtype": None,
            "keepdim": True,
            "self": {"dtype": "float32", "shape": [1, 264, 16, 128]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::mean.dim",
        {
            "dim": [-1],
            "dtype": None,
            "keepdim": True,
            "self": {"dtype": "float32", "shape": [1, 264, 2048]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::mean.dim",
        {
            "dim": [-1],
            "dtype": None,
            "keepdim": True,
            "self": {"dtype": "float32", "shape": [1, 256, 4, 512]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::mean.dim",
        {
            "dim": [-1],
            "dtype": None,
            "keepdim": True,
            "self": {"dtype": "float32", "shape": [1, 96, 8, 512]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::min",
        {"self": {"dtype": "int64", "shape": [8192]}},
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
        ),
    )
    t(
        "aten::min",
        {"self": {"dtype": "int64", "shape": [4096]}},
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
        ),
    )
    t(
        "aten::minimum",
        {
            "other": {"dtype": "int64", "shape": [32, 32]},
            "self": {"dtype": "int64", "shape": [32, 32]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::minimum",
        {
            "other": {"dtype": "int64", "shape": [16, 16]},
            "self": {"dtype": "int64", "shape": [16, 16]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::mm",
        {
            "mat2": {"dtype": "bfloat16", "shape": [32, 256]},
            "self": {"dtype": "bfloat16", "shape": [2048, 32]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::mm",
        {
            "mat2": {"dtype": "bfloat16", "shape": [72, 336]},
            "self": {"dtype": "bfloat16", "shape": [64, 72]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::mse_loss",
        {
            "reduction": 0,
            "self": {"dtype": "float32", "shape": [1, 40, 132]},
            "target": {"dtype": "float32", "shape": [1, 40, 132]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::mse_loss",
        {
            "reduction": 0,
            "self": {"dtype": "float32", "shape": [1, 16, 32]},
            "target": {"dtype": "float32", "shape": [1, 16, 32]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::mse_loss",
        {
            "reduction": 1,
            "self": {"dtype": "float32", "shape": [1, 3, 7, 7]},
            "target": {"dtype": "float32", "shape": [1, 3, 7, 7]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::mse_loss_backward",
        {
            "grad_output": {"dtype": "float32", "shape": [1, 40, 132]},
            "reduction": 0,
            "self": {"dtype": "float32", "shape": [1, 40, 132]},
            "target": {"dtype": "float32", "shape": [1, 40, 132]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::mse_loss_backward",
        {
            "grad_output": {"dtype": "float32", "shape": [1, 16, 32]},
            "reduction": 0,
            "self": {"dtype": "float32", "shape": [1, 16, 32]},
            "target": {"dtype": "float32", "shape": [1, 16, 32]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::mse_loss_backward",
        {
            "grad_output": {"dtype": "float32", "shape": []},
            "reduction": 1,
            "self": {"dtype": "float32", "shape": [1, 3, 7, 7]},
            "target": {"dtype": "float32", "shape": [1, 3, 7, 7]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::mul.Scalar",
        {"other": 0.0, "self": {"dtype": "float32", "shape": [6144, 128]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::mul.Scalar",
        {"other": 0.0, "self": {"dtype": "float32", "shape": [1, 264, 16, 128]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::mul.Scalar",
        {"other": 0.0, "self": {"dtype": "float32", "shape": [1, 264, 2048]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::mul.Scalar",
        {"other": 0.0, "self": {"dtype": "float32", "shape": [1, 256, 4, 512]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::mul.Scalar",
        {"other": 0.0, "self": {"dtype": "float32", "shape": [1, 96, 8, 512]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::mul.Scalar",
        {"other": 0.0, "self": {"dtype": "float32", "shape": [3072, 128]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::mul.Tensor",
        {
            "other": {"dtype": "float32", "shape": [16384, 1]},
            "self": {"dtype": "float32", "shape": [16384, 37984]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::mul.Tensor",
        {
            "other": {"dtype": "float32", "shape": [8192, 1]},
            "self": {"dtype": "float32", "shape": [8192, 37984]},
        },
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::mul.Tensor",
        {
            "other": {"dtype": "float32", "shape": [8192, 1]},
            "self": {"dtype": "float32", "shape": [8192, 32064]},
        },
        count=5,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
        ),
    )
    t(
        "aten::mul.Tensor",
        {
            "other": {"dtype": "bfloat16", "shape": [1, 8192, 25600]},
            "self": {"dtype": "bfloat16", "shape": [1, 8192, 25600]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::mul.Tensor",
        {
            "other": {"dtype": "float32", "shape": [4096, 1]},
            "self": {"dtype": "float32", "shape": [4096, 50512]},
        },
        count=1,
        paths=("text-generation-torch-titan-multi-node-4N-async-hostcc",),
    )
    t(
        "aten::mul.Tensor",
        {
            "other": {"dtype": "float32", "shape": [4096, 1]},
            "self": {"dtype": "float32", "shape": [4096, 37984]},
        },
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::mul_.Tensor",
        {
            "other": {"dtype": "float32", "shape": [1, 4096, 16]},
            "self": {"dtype": "float32", "shape": [1, 4096, 16]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::mul_.Tensor",
        {
            "other": {"dtype": "bfloat16", "shape": [1]},
            "self": {"dtype": "bfloat16", "shape": [1, 256, 176]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::mul_.Tensor",
        {
            "other": {"dtype": "bfloat16", "shape": [1]},
            "self": {"dtype": "bfloat16", "shape": [1, 96, 336]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::mul_.Tensor",
        {
            "other": {"dtype": "bfloat16", "shape": [1]},
            "self": {"dtype": "bfloat16", "shape": [1, 96, 176]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::mul_.Tensor",
        {
            "other": {"dtype": "bfloat16", "shape": [1]},
            "self": {"dtype": "bfloat16", "shape": [1, 96, 96]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::mul_.Tensor",
        {"other": {"dtype": "int64", "shape": []}, "self": {"dtype": "int64", "shape": []}},
        count=2,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::native_dropout",
        {"input": {"dtype": "bfloat16", "shape": [1, 96, 160]}, "p": 0.1, "train": True},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::native_dropout",
        {"input": {"dtype": "bfloat16", "shape": [96, 160]}, "p": 0.1, "train": True},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::native_dropout",
        {"input": {"dtype": "bfloat16", "shape": [1, 96, 48]}, "p": 0.1, "train": True},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::native_dropout",
        {"input": {"dtype": "float32", "shape": [1, 96, 48]}, "p": 0.1, "train": True},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::native_dropout",
        {"input": {"dtype": "bfloat16", "shape": [1, 64, 48]}, "p": 0.1, "train": True},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::native_dropout",
        {"input": {"dtype": "bfloat16", "shape": [1, 32, 48]}, "p": 0.1, "train": True},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::native_dropout_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [1, 96, 160]},
            "mask": {"dtype": "bool", "shape": [1, 96, 160]},
            "scale": 1.11111,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::native_dropout_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [96, 160]},
            "mask": {"dtype": "bool", "shape": [96, 160]},
            "scale": 1.11111,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::native_dropout_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [1, 96, 48]},
            "mask": {"dtype": "bool", "shape": [1, 96, 48]},
            "scale": 1.11111,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::native_dropout_backward",
        {
            "grad_output": {"dtype": "float32", "shape": [1, 96, 48]},
            "mask": {"dtype": "bool", "shape": [1, 96, 48]},
            "scale": 1.11111,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::native_dropout_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [1, 64, 48]},
            "mask": {"dtype": "bool", "shape": [1, 64, 48]},
            "scale": 1.11111,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::native_dropout_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [1, 32, 48]},
            "mask": {"dtype": "bool", "shape": [1, 32, 48]},
            "scale": 1.11111,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::native_layer_norm",
        {
            "bias": {"dtype": "float32", "shape": [1280]},
            "eps": 1e-05,
            "input": {"dtype": "float32", "shape": [1, 1500, 1280]},
            "normalized_shape": [1280],
            "weight": {"dtype": "float32", "shape": [1280]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::native_layer_norm",
        {
            "bias": {"dtype": "float32", "shape": [1280]},
            "eps": 1e-05,
            "input": {"dtype": "float32", "shape": [1, 448, 1280]},
            "normalized_shape": [1280],
            "weight": {"dtype": "float32", "shape": [1280]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::native_layer_norm",
        {
            "bias": {"dtype": "float32", "shape": [2048]},
            "eps": 1e-05,
            "input": {"dtype": "float32", "shape": [1, 264, 2048]},
            "normalized_shape": [2048],
            "weight": {"dtype": "float32", "shape": [2048]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::native_layer_norm",
        {
            "bias": {"dtype": "bfloat16", "shape": [512]},
            "eps": 1e-05,
            "input": {"dtype": "bfloat16", "shape": [1, 1025, 512]},
            "normalized_shape": [512],
            "weight": {"dtype": "bfloat16", "shape": [512]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::native_layer_norm",
        {
            "bias": {"dtype": "float32", "shape": [768]},
            "eps": 1e-06,
            "input": {"dtype": "float32", "shape": [3, 197, 768]},
            "normalized_shape": [768],
            "weight": {"dtype": "float32", "shape": [768]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::native_layer_norm",
        {
            "bias": {"dtype": "float32", "shape": [768]},
            "eps": 1e-05,
            "input": {"dtype": "float32", "shape": [3, 1, 196, 768]},
            "normalized_shape": [768],
            "weight": {"dtype": "float32", "shape": [768]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::native_layer_norm_backward",
        {
            "bias": {"dtype": "float32", "shape": [1280]},
            "grad_out": {"dtype": "float32", "shape": [1, 1500, 1280]},
            "input": {"dtype": "float32", "shape": [1, 1500, 1280]},
            "mean": {"dtype": "float32", "shape": [1, 1500, 1]},
            "normalized_shape": [1280],
            "output_mask": [True, True, True],
            "rstd": {"dtype": "float32", "shape": [1, 1500, 1]},
            "weight": {"dtype": "float32", "shape": [1280]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::native_layer_norm_backward",
        {
            "bias": {"dtype": "float32", "shape": [1280]},
            "grad_out": {"dtype": "float32", "shape": [1, 448, 1280]},
            "input": {"dtype": "float32", "shape": [1, 448, 1280]},
            "mean": {"dtype": "float32", "shape": [1, 448, 1]},
            "normalized_shape": [1280],
            "output_mask": [True, True, True],
            "rstd": {"dtype": "float32", "shape": [1, 448, 1]},
            "weight": {"dtype": "float32", "shape": [1280]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::native_layer_norm_backward",
        {
            "bias": {"dtype": "float32", "shape": [2048]},
            "grad_out": {"dtype": "float32", "shape": [1, 264, 2048]},
            "input": {"dtype": "float32", "shape": [1, 264, 2048]},
            "mean": {"dtype": "float32", "shape": [1, 264, 1]},
            "normalized_shape": [2048],
            "output_mask": [True, True, True],
            "rstd": {"dtype": "float32", "shape": [1, 264, 1]},
            "weight": {"dtype": "float32", "shape": [2048]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::native_layer_norm_backward",
        {
            "bias": {"dtype": "bfloat16", "shape": [512]},
            "grad_out": {"dtype": "bfloat16", "shape": [1, 1025, 512]},
            "input": {"dtype": "bfloat16", "shape": [1, 1025, 512]},
            "mean": {"dtype": "float32", "shape": [1, 1025, 1]},
            "normalized_shape": [512],
            "output_mask": [True, True, True],
            "rstd": {"dtype": "float32", "shape": [1, 1025, 1]},
            "weight": {"dtype": "bfloat16", "shape": [512]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::native_layer_norm_backward",
        {
            "bias": {"dtype": "float32", "shape": [1152]},
            "grad_out": {"dtype": "float32", "shape": [1, 256, 1152]},
            "input": {"dtype": "float32", "shape": [1, 256, 1152]},
            "mean": {"dtype": "float32", "shape": [1, 256, 1]},
            "normalized_shape": [1152],
            "output_mask": [True, True, True],
            "rstd": {"dtype": "float32", "shape": [1, 256, 1]},
            "weight": {"dtype": "float32", "shape": [1152]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::native_layer_norm_backward",
        {
            "bias": {"dtype": "bfloat16", "shape": [1024]},
            "grad_out": {"dtype": "bfloat16", "shape": [1, 257, 1024]},
            "input": {"dtype": "bfloat16", "shape": [1, 257, 1024]},
            "mean": {"dtype": "float32", "shape": [1, 257, 1]},
            "normalized_shape": [1024],
            "output_mask": [True, True, True],
            "rstd": {"dtype": "float32", "shape": [1, 257, 1]},
            "weight": {"dtype": "bfloat16", "shape": [1024]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::ne.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [16384, 1]},
            "self": {"dtype": "int64", "shape": [16384, 1]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::ne.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [16384]},
            "self": {"dtype": "int64", "shape": [16384]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::ne.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [8192, 1]},
            "self": {"dtype": "int64", "shape": [8192, 1]},
        },
        count=5,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
        ),
    )
    t(
        "aten::ne.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [8192]},
            "self": {"dtype": "int64", "shape": [8192]},
        },
        count=5,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
        ),
    )
    t(
        "aten::ne.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [4096, 1]},
            "self": {"dtype": "int64", "shape": [4096, 1]},
        },
        count=5,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
        ),
    )
    t(
        "aten::ne.Scalar_out",
        {
            "other": 0.0,
            "out": {"dtype": "bool", "shape": [4096]},
            "self": {"dtype": "int64", "shape": [4096]},
        },
        count=5,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
        ),
    )
    t(
        "aten::neg",
        {"self": {"dtype": "float32", "shape": [16384, 128]}},
        count=3,
        paths=(
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3",
        ),
    )
    t(
        "aten::neg",
        {"self": {"dtype": "float32", "shape": [1, 1024, 640]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::neg",
        {"self": {"dtype": "float32", "shape": [1, 64, 2, 64, 64]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::neg",
        {"self": {"dtype": "bfloat16", "shape": [1, 64, 64, 128]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::neg",
        {"self": {"dtype": "float32", "shape": [1, 32, 2, 64, 64]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::neg",
        {"self": {"dtype": "bfloat16", "shape": [1, 32, 64, 128]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::nonzero",
        {"self": {"dtype": "bool", "shape": [768]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::nonzero",
        {"self": {"dtype": "bool", "shape": [1, 96]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::nonzero",
        {"self": {"dtype": "bool", "shape": [96]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::nonzero",
        {"self": {"dtype": "int64", "shape": [32, 2]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::nonzero",
        {"self": {"dtype": "bool", "shape": [13]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::nonzero",
        {"self": {"dtype": "bool", "shape": [1, 4]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::one_hot",
        {"num_classes": 8, "self": {"dtype": "int64", "shape": [32, 2]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::one_hot",
        {"num_classes": 4, "self": {"dtype": "int64", "shape": [1, 36]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::ones",
        {"device": None, "dtype": 4, "layout": None, "pin_memory": None, "size": [1, 32]},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::ones",
        {"device": None, "dtype": 4, "layout": None, "pin_memory": None, "size": [1, 65]},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::ones",
        {"device": None, "dtype": 11, "layout": None, "pin_memory": None, "size": [128, 128]},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::ones",
        {"device": None, "dtype": 11, "layout": None, "pin_memory": None, "size": [2, 2]},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::ones",
        {"device": None, "dtype": 11, "layout": None, "pin_memory": None, "size": [256, 256]},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::ones",
        {"device": None, "dtype": 4, "layout": None, "pin_memory": None, "size": [3, 1, 96]},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::ones_like",
        {
            "device": None,
            "dtype": None,
            "layout": None,
            "memory_format": None,
            "pin_memory": None,
            "self": {"dtype": "float32", "shape": [262144, 32]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::ones_like",
        {
            "device": None,
            "dtype": None,
            "layout": None,
            "memory_format": None,
            "pin_memory": None,
            "self": {"dtype": "float32", "shape": [8192, 64]},
        },
        count=10,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-compile-trn3",
            "text-generation-torch-titan-multi-core-compile-trn3-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-compile-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
        ),
    )
    t(
        "aten::ones_like",
        {
            "device": None,
            "dtype": None,
            "layout": None,
            "memory_format": None,
            "pin_memory": None,
            "self": {"dtype": "float32", "shape": [4096, 64]},
        },
        count=1,
        paths=("text-generation-torch-titan-multi-node-4N-async-hostcc",),
    )
    t(
        "aten::ones_like",
        {
            "device": None,
            "dtype": None,
            "layout": None,
            "memory_format": None,
            "pin_memory": None,
            "self": {"dtype": "float32", "shape": [4096, 32]},
        },
        count=1,
        paths=("text-generation-torch-titan-multi-core-async-compile-moe",),
    )
    t(
        "aten::ones_like",
        {
            "device": None,
            "dtype": None,
            "layout": None,
            "memory_format": None,
            "pin_memory": None,
            "self": {"dtype": "float32", "shape": []},
        },
        count=22,
        paths=(
            "test_kaizen_hf_onboarding_reduced_eager",
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-dev",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-compile-trn3",
            "text-generation-torch-titan-multi-core-compile-trn3-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-compile-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
            "text-generation-torch-titan-qwen3-235b-trn3",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::pow.Scalar",
        {"exponent": {"dtype": "float32", "shape": [64]}, "self": 0.0},
        count=24,
        paths=(
            "test_kaizen_hf_onboarding_reduced_eager",
            "text-generation-torch-titan-multi-core-async-compile-dev",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-compile-trn3",
            "text-generation-torch-titan-multi-core-compile-trn3-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-compile-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
            "text-generation-torch-titan-qwen3-235b-trn3",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::pow.Scalar",
        {"exponent": {"dtype": "float32", "shape": [32]}, "self": 0.0},
        count=5,
        paths=(
            "test_kaizen_hf_onboarding_reduced_eager",
            "text-generation-torch-titan-multi-core-async-compile-moe",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::pow.Scalar",
        {"exponent": {"dtype": "float32", "shape": [10]}, "self": 0.0},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::pow.Scalar",
        {"exponent": {"dtype": "float32", "shape": [6]}, "self": 0.0},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::pow.Tensor_Scalar_out",
        {
            "exponent": 1.0,
            "out": {"dtype": "float32", "shape": [6144, 128]},
            "self": {"dtype": "float32", "shape": [6144, 128]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::pow.Tensor_Scalar_out",
        {
            "exponent": 2,
            "out": {"dtype": "float32", "shape": [6144, 128]},
            "self": {"dtype": "float32", "shape": [6144, 128]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::pow.Tensor_Scalar_out",
        {
            "exponent": 1.0,
            "out": {"dtype": "float32", "shape": [1, 264, 16, 128]},
            "self": {"dtype": "float32", "shape": [1, 264, 16, 128]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::pow.Tensor_Scalar_out",
        {
            "exponent": 2,
            "out": {"dtype": "float32", "shape": [1, 264, 16, 128]},
            "self": {"dtype": "float32", "shape": [1, 264, 16, 128]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::pow.Tensor_Scalar_out",
        {
            "exponent": 1.0,
            "out": {"dtype": "float32", "shape": [1, 264, 2048]},
            "self": {"dtype": "float32", "shape": [1, 264, 2048]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::pow.Tensor_Scalar_out",
        {
            "exponent": 2,
            "out": {"dtype": "float32", "shape": [1, 264, 2048]},
            "self": {"dtype": "float32", "shape": [1, 264, 2048]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::reciprocal",
        {"self": {"dtype": "bfloat16", "shape": [1, 32, 8, 1]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::reciprocal",
        {"self": {"dtype": "float32", "shape": [64]}},
        count=22,
        paths=(
            "test_kaizen_hf_onboarding_reduced_eager",
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-dev",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-compile-trn3",
            "text-generation-torch-titan-multi-core-compile-trn3-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-compile-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
            "text-generation-torch-titan-qwen3-235b-trn3",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::reciprocal",
        {"self": {"dtype": "bfloat16", "shape": [1, 32, 2, 1]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::reciprocal",
        {"self": {"dtype": "float32", "shape": [32]}},
        count=6,
        paths=(
            "test_kaizen_hf_onboarding_reduced_eager",
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile-moe",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::reciprocal",
        {"self": {"dtype": "float32", "shape": [10]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::reciprocal",
        {"self": {"dtype": "float32", "shape": [6]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::relu",
        {"self": {"dtype": "bfloat16", "shape": [1, 32, 4096]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::relu",
        {"self": {"dtype": "bfloat16", "shape": [1, 32, 3712]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::relu",
        {"self": {"dtype": "bfloat16", "shape": [64, 1856]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::relu",
        {"self": {"dtype": "float32", "shape": [1, 32, 64, 32]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::relu",
        {"self": {"dtype": "bfloat16", "shape": [1, 49, 1024]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::relu",
        {"self": {"dtype": "bfloat16", "shape": [1, 41, 1024]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::remainder.Scalar",
        {"other": 0.0, "self": {"dtype": "float32", "shape": [262144]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::remainder.Scalar",
        {"other": 0.0, "self": {"dtype": "int32", "shape": [128]}},
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::remainder.Scalar",
        {"other": 0.0, "self": {"dtype": "int32", "shape": [64]}},
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::remainder.Scalar",
        {"other": 0.0, "self": {"dtype": "int32", "shape": [16]}},
        count=1,
        paths=("text-generation-torch-titan-multi-node-4N-async-hostcc",),
    )
    t(
        "aten::remainder.Scalar",
        {"other": 0.0, "self": {"dtype": "int64", "shape": [1]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::remainder.Scalar",
        {"other": 0.0, "self": {"dtype": "int64", "shape": []}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::remainder.Tensor_out",
        {
            "other": {"dtype": "int64", "shape": [64]},
            "out": {"dtype": "int64", "shape": [64]},
            "self": {"dtype": "int64", "shape": [64]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::repeat",
        {"repeats": [1, 2], "self": {"dtype": "float32", "shape": [96, 64]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::repeat",
        {"repeats": [1, 1], "self": {"dtype": "float32", "shape": [64, 80]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::repeat",
        {"repeats": [1, 1, 1], "self": {"dtype": "float32", "shape": [1, 1, 1408]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::repeat",
        {"repeats": [1, 1, 1], "self": {"dtype": "bfloat16", "shape": [1, 1, 1152]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::repeat",
        {"repeats": [1, 2], "self": {"dtype": "float32", "shape": [64, 10]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::repeat",
        {"repeats": [1, 1], "self": {"dtype": "int64", "shape": [1, 448]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::repeat_interleave.Tensor",
        {"output_size": None, "repeats": {"dtype": "int32", "shape": [128]}},
        count=3,
        paths=(
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3",
        ),
    )
    t(
        "aten::rsqrt",
        {"self": {"dtype": "float32", "shape": [1, 96, 64, 1]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::rsqrt",
        {"self": {"dtype": "float32", "shape": [6144, 1]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::rsqrt",
        {"self": {"dtype": "float32", "shape": [1, 264, 16, 1]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::rsqrt",
        {"self": {"dtype": "float32", "shape": [1, 96, 32, 1]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::rsqrt",
        {"self": {"dtype": "float32", "shape": [3072, 1]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::rsqrt",
        {"self": {"dtype": "float32", "shape": [1, 264, 8, 1]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::scatter.value_out",
        {
            "dim": -1,
            "index": {"dtype": "int64", "shape": [16384, 8]},
            "out": {"dtype": "bool", "shape": [16384, 128]},
            "self": {"dtype": "bool", "shape": [16384, 128]},
            "value": True,
        },
        count=3,
        paths=(
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3",
        ),
    )
    t(
        "aten::scatter.value_out",
        {
            "dim": -1,
            "index": {"dtype": "int64", "shape": [1, 32, 32]},
            "out": {"dtype": "bool", "shape": [1, 32, 32]},
            "self": {"dtype": "bool", "shape": [1, 32, 32]},
            "value": False,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::scatter.value_out",
        {
            "dim": -1,
            "index": {"dtype": "int64", "shape": [1, 4, 96, 1]},
            "out": {"dtype": "bfloat16", "shape": [1, 4, 96, 2]},
            "self": {"dtype": "bfloat16", "shape": [1, 4, 96, 2]},
            "value": 0.0,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::scatter.value_out",
        {
            "dim": -1,
            "index": {"dtype": "int64", "shape": [1, 4, 96, 1]},
            "out": {"dtype": "bfloat16", "shape": [1, 4, 96, 1]},
            "self": {"dtype": "bfloat16", "shape": [1, 4, 96, 1]},
            "value": 0.0,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::scatter.value_out",
        {
            "dim": -1,
            "index": {"dtype": "int64", "shape": [1, 1, 32, 8]},
            "out": {"dtype": "bfloat16", "shape": [1, 1, 32, 9]},
            "self": {"dtype": "bfloat16", "shape": [1, 1, 32, 9]},
            "value": 0.0,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::scatter.value_out",
        {
            "dim": 1,
            "index": {"dtype": "int64", "shape": [96, 1]},
            "out": {"dtype": "float32", "shape": [96, 1]},
            "self": {"dtype": "float32", "shape": [96, 1]},
            "value": 0.0,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::scatter_add_",
        {
            "dim": 1,
            "index": {"dtype": "int64", "shape": [1, 256, 1024]},
            "self": {"dtype": "bfloat16", "shape": [1, 1024, 1024]},
            "src": {"dtype": "bfloat16", "shape": [1, 256, 1024]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::scatter_add_",
        {
            "dim": 1,
            "index": {"dtype": "int64", "shape": [4096, 8]},
            "self": {"dtype": "float32", "shape": [4096, 128]},
            "src": {"dtype": "float32", "shape": [4096, 8]},
        },
        count=1,
        paths=("text-generation-torch-titan-multi-core-async",),
    )
    t(
        "aten::scatter_add_",
        {
            "dim": 1,
            "index": {"dtype": "int64", "shape": [4096, 6]},
            "self": {"dtype": "float32", "shape": [4096, 64]},
            "src": {"dtype": "float32", "shape": [4096, 6]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::scatter_add_",
        {
            "dim": 1,
            "index": {"dtype": "int64", "shape": [4096, 1]},
            "self": {"dtype": "float32", "shape": [4096, 16]},
            "src": {"dtype": "float32", "shape": [4096, 1]},
        },
        count=1,
        paths=("text-generation-torch-titan-multi-node-4N-async-hostcc",),
    )
    t(
        "aten::scatter_add_",
        {
            "dim": 1,
            "index": {"dtype": "int64", "shape": [96, 8]},
            "self": {"dtype": "bfloat16", "shape": [96, 384]},
            "src": {"dtype": "bfloat16", "shape": [96, 8]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::scatter_add_",
        {
            "dim": 1,
            "index": {"dtype": "int64", "shape": [96, 4]},
            "self": {"dtype": "float32", "shape": [96, 128]},
            "src": {"dtype": "float32", "shape": [96, 4]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::select_backward",
        {
            "dim": 0,
            "grad_output": {"dtype": "bfloat16", "shape": [10240, 72]},
            "index": 0,
            "input_sizes": [2, 10240, 72],
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::select_backward",
        {
            "dim": 0,
            "grad_output": {"dtype": "bfloat16", "shape": [10240, 72]},
            "index": 1,
            "input_sizes": [2, 10240, 72],
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::select_backward",
        {
            "dim": 2,
            "grad_output": {"dtype": "bfloat16", "shape": [1, 64, 64, 128]},
            "index": 0,
            "input_sizes": [1, 64, 2, 64, 128],
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::select_backward",
        {
            "dim": 2,
            "grad_output": {"dtype": "bfloat16", "shape": [1, 64, 64, 128]},
            "index": 1,
            "input_sizes": [1, 64, 2, 64, 128],
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::select_backward",
        {
            "dim": 2,
            "grad_output": {"dtype": "float32", "shape": [1, 64, 64, 128]},
            "index": 0,
            "input_sizes": [1, 64, 2, 64, 128],
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::select_backward",
        {
            "dim": 2,
            "grad_output": {"dtype": "float32", "shape": [1, 64, 64, 128]},
            "index": 1,
            "input_sizes": [1, 64, 2, 64, 128],
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::sgn",
        {"self": {"dtype": "float32", "shape": [1, 1024, 640]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::sigmoid",
        {"self": {"dtype": "bfloat16", "shape": [1, 96, 12288]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::sigmoid",
        {"self": {"dtype": "bfloat16", "shape": [96, 3072]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::sigmoid",
        {"self": {"dtype": "bfloat16", "shape": [1, 96, 2048]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::sigmoid",
        {"self": {"dtype": "bfloat16", "shape": [64, 2880]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::sigmoid",
        {"self": {"dtype": "bfloat16", "shape": [1, 32, 4096]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::sigmoid",
        {"self": {"dtype": "bfloat16", "shape": [1, 96, 1024]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::sigmoid_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [1, 96, 12288]},
            "output": {"dtype": "bfloat16", "shape": [1, 96, 12288]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::sigmoid_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [96, 3072]},
            "output": {"dtype": "bfloat16", "shape": [96, 3072]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::sigmoid_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [1, 96, 2048]},
            "output": {"dtype": "bfloat16", "shape": [1, 96, 2048]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::sigmoid_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [64, 2880]},
            "output": {"dtype": "bfloat16", "shape": [64, 2880]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::sigmoid_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [1, 32, 4096]},
            "output": {"dtype": "bfloat16", "shape": [1, 32, 4096]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::sigmoid_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [1, 96, 1024]},
            "output": {"dtype": "bfloat16", "shape": [1, 96, 1024]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::sign",
        {"self": {"dtype": "float32", "shape": [64]}},
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile-moe",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-qwen3-235b-trn3",
        ),
    )
    t(
        "aten::sign",
        {"self": {"dtype": "float32", "shape": [32]}},
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async-compile-moe",
            "text-generation-torch-titan-multi-core-trn3",
        ),
    )
    t(
        "aten::sign",
        {"self": {"dtype": "float32", "shape": [16]}},
        count=1,
        paths=("text-generation-torch-titan-multi-node-4N-async-hostcc",),
    )
    t(
        "aten::silu",
        {"self": {"dtype": "bfloat16", "shape": [1, 8192, 25600]}},
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::silu",
        {"self": {"dtype": "bfloat16", "shape": [1, 4096, 25600]}},
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::silu",
        {"self": {"dtype": "bfloat16", "shape": [1, 8192, 12288]}},
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::silu",
        {"self": {"dtype": "bfloat16", "shape": [1, 8192, 7168]}},
        count=1,
        paths=("text-generation-torch-titan-multi-node-4N-async-hostcc",),
    )
    t(
        "aten::silu",
        {"self": {"dtype": "bfloat16", "shape": [1, 8192, 6400]}},
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::silu",
        {"self": {"dtype": "bfloat16", "shape": [1, 4096, 12288]}},
        count=6,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::silu_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [1, 8192, 25600]},
            "self": {"dtype": "bfloat16", "shape": [1, 8192, 25600]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::silu_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [1, 4096, 25600]},
            "self": {"dtype": "bfloat16", "shape": [1, 4096, 25600]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::silu_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [1, 8192, 12288]},
            "self": {"dtype": "bfloat16", "shape": [1, 8192, 12288]},
        },
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::silu_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [1, 8192, 7168]},
            "self": {"dtype": "bfloat16", "shape": [1, 8192, 7168]},
        },
        count=1,
        paths=("text-generation-torch-titan-multi-node-4N-async-hostcc",),
    )
    t(
        "aten::silu_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [1, 8192, 6400]},
            "self": {"dtype": "bfloat16", "shape": [1, 8192, 6400]},
        },
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::silu_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [1, 4096, 12288]},
            "self": {"dtype": "bfloat16", "shape": [1, 4096, 12288]},
        },
        count=6,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::sin",
        {"self": {"dtype": "float32", "shape": [8192, 128]}},
        count=7,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-dev",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::sin",
        {"self": {"dtype": "float32", "shape": [4096, 128]}},
        count=15,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::sin",
        {"self": {"dtype": "float32", "shape": [4096, 64]}},
        count=5,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile-moe",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::sin",
        {"self": {"dtype": "float32", "shape": [2048, 128]}},
        count=1,
        paths=("text-generation-torch-titan-qwen3-235b-trn3",),
    )
    t(
        "aten::sin",
        {"self": {"dtype": "float32", "shape": [1, 256, 512]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::sin",
        {"self": {"dtype": "float32", "shape": [1, 256, 256]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::slice_backward",
        {
            "dim": 0,
            "end": 16384,
            "grad_output": {"dtype": "bfloat16", "shape": [16384, 2048]},
            "input_sizes": [16384, 2048],
            "start": 0,
            "step": 1,
        },
        count=3,
        paths=(
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3",
        ),
    )
    t(
        "aten::slice_backward",
        {
            "dim": 3,
            "end": 9223372036854775807,
            "grad_output": {"dtype": "bfloat16", "shape": [1, 8192, 64, 64]},
            "input_sizes": [1, 8192, 64, 128],
            "start": 64,
            "step": 1,
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::slice_backward",
        {
            "dim": 3,
            "end": 9223372036854775807,
            "grad_output": {"dtype": "bfloat16", "shape": [1, 8192, 32, 64]},
            "input_sizes": [1, 8192, 32, 128],
            "start": 64,
            "step": 1,
        },
        count=3,
        paths=(
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::slice_backward",
        {
            "dim": 3,
            "end": 9223372036854775807,
            "grad_output": {"dtype": "bfloat16", "shape": [1, 4096, 64, 64]},
            "input_sizes": [1, 4096, 64, 128],
            "start": 64,
            "step": 1,
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::slice_backward",
        {
            "dim": 3,
            "end": 9223372036854775807,
            "grad_output": {"dtype": "bfloat16", "shape": [1, 4096, 32, 64]},
            "input_sizes": [1, 4096, 32, 128],
            "start": 64,
            "step": 1,
        },
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::slice_backward",
        {
            "dim": 3,
            "end": 9223372036854775807,
            "grad_output": {"dtype": "bfloat16", "shape": [1, 8192, 16, 64]},
            "input_sizes": [1, 8192, 16, 128],
            "start": 64,
            "step": 1,
        },
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::softplus",
        {"beta": 1, "self": {"dtype": "bfloat16", "shape": [1, 4096, 32]}, "threshold": 20},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::softplus",
        {"beta": 1, "self": {"dtype": "bfloat16", "shape": [1, 2048, 32]}, "threshold": 20},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::softplus",
        {"beta": 1, "self": {"dtype": "float32", "shape": [1, 96, 64]}, "threshold": 20},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::softplus",
        {"beta": 1, "self": {"dtype": "bfloat16", "shape": [1, 32, 128]}, "threshold": 20},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::softplus",
        {"beta": 1, "self": {"dtype": "float32", "shape": [1, 96, 32]}, "threshold": 20},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::softplus",
        {"beta": 1, "self": {"dtype": "float32", "shape": [1, 32, 32]}, "threshold": 20},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::softplus_backward",
        {
            "beta": 1,
            "grad_output": {"dtype": "bfloat16", "shape": [1, 2048, 32]},
            "self": {"dtype": "bfloat16", "shape": [1, 2048, 32]},
            "threshold": 20,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::softplus_backward",
        {
            "beta": 1,
            "grad_output": {"dtype": "float32", "shape": [1, 96, 64]},
            "self": {"dtype": "float32", "shape": [1, 96, 64]},
            "threshold": 20,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::softplus_backward",
        {
            "beta": 1,
            "grad_output": {"dtype": "bfloat16", "shape": [1, 32, 128]},
            "self": {"dtype": "bfloat16", "shape": [1, 32, 128]},
            "threshold": 20,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::softplus_backward",
        {
            "beta": 1,
            "grad_output": {"dtype": "float32", "shape": [1, 96, 32]},
            "self": {"dtype": "float32", "shape": [1, 96, 32]},
            "threshold": 20,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::softplus_backward",
        {
            "beta": 1,
            "grad_output": {"dtype": "float32", "shape": [1, 32, 32]},
            "self": {"dtype": "float32", "shape": [1, 32, 32]},
            "threshold": 20,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::softplus_backward",
        {
            "beta": 1,
            "grad_output": {"dtype": "bfloat16", "shape": [1, 32, 16]},
            "self": {"dtype": "bfloat16", "shape": [1, 32, 16]},
            "threshold": 20,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::split_with_sizes_copy.out",
        {
            "dim": 0,
            "out": None,
            "self": {"dtype": "bfloat16", "shape": [64, 12154960]},
            "split_sizes": [80, 12154880],
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::split_with_sizes_copy.out",
        {
            "dim": 0,
            "out": None,
            "self": {"dtype": "bfloat16", "shape": [64, 12154880]},
            "split_sizes": [12154880],
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::split_with_sizes_copy.out",
        {
            "dim": 0,
            "out": None,
            "self": {"dtype": "bfloat16", "shape": [64, 9723968]},
            "split_sizes": [64, 9723904],
        },
        count=10,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::split_with_sizes_copy.out",
        {
            "dim": 0,
            "out": None,
            "self": {"dtype": "bfloat16", "shape": [64, 9723904]},
            "split_sizes": [9723904],
        },
        count=10,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::split_with_sizes_copy.out",
        {
            "dim": 0,
            "out": None,
            "self": {"dtype": "bfloat16", "shape": [64, 7864320]},
            "split_sizes": [2621440, 2621440, 2621440],
        },
        count=1,
        paths=("text-generation-torch-titan-multi-node-4N-async-hostcc",),
    )
    t(
        "aten::split_with_sizes_copy.out",
        {
            "dim": 0,
            "out": None,
            "self": {"dtype": "bfloat16", "shape": [64, 7618724]},
            "split_sizes": [2, 2, 655360, 81920, 81920, 655360],
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::sqrt",
        {"self": {"dtype": "bfloat16", "shape": [32, 4]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::stack",
        {
            "dim": 0,
            "tensors": [
                {"dtype": "bfloat16", "shape": [196, 1, 128]},
                {"dtype": "bfloat16", "shape": [196, 1, 128]},
                {"dtype": "bfloat16", "shape": [196, 1, 128]},
            ],
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::stack",
        {"dim": 0, "tensors": [{"dtype": "bfloat16", "shape": [64, 256]}]},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::stack",
        {"dim": 0, "tensors": [{"dtype": "float32", "shape": [96, 128]}]},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::stack",
        {
            "dim": -1,
            "tensors": [
                {"dtype": "bfloat16", "shape": [1, 2048]},
                {"dtype": "bfloat16", "shape": [1, 2048]},
                {"dtype": "bfloat16", "shape": [1, 2048]},
                {"dtype": "bfloat16", "shape": [1, 2048]},
                {"dtype": "bfloat16", "shape": [1, 2048]},
                {"dtype": "bfloat16", "shape": [1, 2048]},
            ],
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::stack",
        {
            "dim": -1,
            "tensors": [
                {"dtype": "int32", "shape": [1, 32, 32]},
                {"dtype": "int32", "shape": [1, 32, 32]},
            ],
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::stack",
        {
            "dim": -1,
            "tensors": [
                {"dtype": "int32", "shape": [32, 32]},
                {"dtype": "int32", "shape": [32, 32]},
            ],
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::sub.Tensor",
        {
            "alpha": 1,
            "other": {"dtype": "float32", "shape": [16384, 1]},
            "self": {"dtype": "float32", "shape": [16384, 37984]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::sub.Tensor",
        {
            "alpha": 1,
            "other": {"dtype": "float32", "shape": [8192, 1]},
            "self": {"dtype": "float32", "shape": [8192, 37984]},
        },
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::sub.Tensor",
        {
            "alpha": 1,
            "other": {"dtype": "float32", "shape": [8192, 1]},
            "self": {"dtype": "float32", "shape": [8192, 32064]},
        },
        count=5,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
        ),
    )
    t(
        "aten::sub.Tensor",
        {
            "alpha": 1,
            "other": {"dtype": "float32", "shape": [4096, 1]},
            "self": {"dtype": "float32", "shape": [4096, 50512]},
        },
        count=1,
        paths=("text-generation-torch-titan-multi-node-4N-async-hostcc",),
    )
    t(
        "aten::sub.Tensor",
        {
            "alpha": 1,
            "other": {"dtype": "float32", "shape": [4096, 1]},
            "self": {"dtype": "float32", "shape": [4096, 37984]},
        },
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::sub.Tensor",
        {
            "alpha": 1,
            "other": {"dtype": "float32", "shape": [4096, 1]},
            "self": {"dtype": "float32", "shape": [4096, 25600]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::sum",
        {"dtype": None, "self": {"dtype": "bool", "shape": [16384]}},
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::sum",
        {"dtype": None, "self": {"dtype": "float32", "shape": [16384]}},
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::sum",
        {"dtype": None, "self": {"dtype": "bool", "shape": [8192]}},
        count=5,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
        ),
    )
    t(
        "aten::sum",
        {"dtype": None, "self": {"dtype": "float32", "shape": [8192]}},
        count=5,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
        ),
    )
    t(
        "aten::sum",
        {"dtype": 6, "self": {"dtype": "bfloat16", "shape": [1, 40, 132]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::sum",
        {"dtype": 6, "self": {"dtype": "float32", "shape": [1, 40, 132]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::sum.IntList_out",
        {
            "dim": [-1],
            "dtype": None,
            "keepdim": False,
            "out": {"dtype": "float32", "shape": [0]},
            "self": {"dtype": "float32", "shape": [1, 448, 30000]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::sum.IntList_out",
        {
            "dim": [-1],
            "dtype": None,
            "keepdim": False,
            "out": {"dtype": "float32", "shape": [0]},
            "self": {"dtype": "float32", "shape": [1, 32, 30000]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::sum.IntList_out",
        {
            "dim": [-1],
            "dtype": None,
            "keepdim": False,
            "out": {"dtype": "float32", "shape": [0]},
            "self": {"dtype": "float32", "shape": [1, 16, 30000]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::sum.IntList_out",
        {
            "dim": [-1],
            "dtype": None,
            "keepdim": False,
            "out": {"dtype": "float32", "shape": [0]},
            "self": {"dtype": "float32", "shape": [1, 96, 3000]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::sum.IntList_out",
        {
            "dim": [-1],
            "dtype": None,
            "keepdim": False,
            "out": {"dtype": "float32", "shape": [0]},
            "self": {"dtype": "float32", "shape": [1, 32, 3000]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::sum.dim_IntList",
        {
            "dim": [1],
            "dtype": None,
            "keepdim": True,
            "self": {"dtype": "float32", "shape": [16384, 37984]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::sum.dim_IntList",
        {
            "dim": [1],
            "dtype": None,
            "keepdim": True,
            "self": {"dtype": "float32", "shape": [8192, 37984]},
        },
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::sum.dim_IntList",
        {
            "dim": [1],
            "dtype": None,
            "keepdim": True,
            "self": {"dtype": "float32", "shape": [8192, 32064]},
        },
        count=5,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
        ),
    )
    t(
        "aten::sum.dim_IntList",
        {
            "dim": [1],
            "dtype": None,
            "keepdim": True,
            "self": {"dtype": "float32", "shape": [4096, 50512]},
        },
        count=1,
        paths=("text-generation-torch-titan-multi-node-4N-async-hostcc",),
    )
    t(
        "aten::sum.dim_IntList",
        {
            "dim": [1],
            "dtype": None,
            "keepdim": True,
            "self": {"dtype": "float32", "shape": [4096, 37984]},
        },
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::sum.dim_IntList",
        {
            "dim": [1],
            "dtype": None,
            "keepdim": True,
            "self": {"dtype": "float32", "shape": [4096, 25600]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::tanh",
        {"self": {"dtype": "float32", "shape": [1, 256, 3000]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::tanh",
        {"self": {"dtype": "bfloat16", "shape": [1, 96, 3000]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::tanh",
        {"self": {"dtype": "float32", "shape": [1, 32, 5120]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::tanh",
        {"self": {"dtype": "bfloat16", "shape": [1, 32, 3072]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::tanh",
        {"self": {"dtype": "float32", "shape": [1, 16, 5120]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::tanh_backward",
        {
            "grad_output": {"dtype": "float32", "shape": [1, 256, 3000]},
            "output": {"dtype": "float32", "shape": [1, 256, 3000]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::tanh_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [1, 96, 3000]},
            "output": {"dtype": "bfloat16", "shape": [1, 96, 3000]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::tanh_backward",
        {
            "grad_output": {"dtype": "float32", "shape": [1, 32, 5120]},
            "output": {"dtype": "float32", "shape": [1, 32, 5120]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::tanh_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [1, 32, 3072]},
            "output": {"dtype": "bfloat16", "shape": [1, 32, 3072]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::tanh_backward",
        {
            "grad_output": {"dtype": "float32", "shape": [1, 16, 5120]},
            "output": {"dtype": "float32", "shape": [1, 16, 5120]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::threshold_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [1, 32, 4096]},
            "self": {"dtype": "bfloat16", "shape": [1, 32, 4096]},
            "threshold": 0,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::threshold_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [1, 32, 3712]},
            "self": {"dtype": "bfloat16", "shape": [1, 32, 3712]},
            "threshold": 0,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::threshold_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [64, 1856]},
            "self": {"dtype": "bfloat16", "shape": [64, 1856]},
            "threshold": 0,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::threshold_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [1, 49, 1024]},
            "self": {"dtype": "bfloat16", "shape": [1, 49, 1024]},
            "threshold": 0,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::threshold_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [1, 41, 1024]},
            "self": {"dtype": "bfloat16", "shape": [1, 41, 1024]},
            "threshold": 0,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::threshold_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": [96, 160]},
            "self": {"dtype": "bfloat16", "shape": [96, 160]},
            "threshold": 0,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::topk",
        {
            "dim": -1,
            "k": 8,
            "largest": True,
            "self": {"dtype": "float32", "shape": [16384, 128]},
            "sorted": False,
        },
        count=3,
        paths=(
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3",
        ),
    )
    t(
        "aten::topk",
        {
            "dim": -1,
            "k": 8,
            "largest": True,
            "self": {"dtype": "float32", "shape": [4096, 128]},
            "sorted": False,
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::topk",
        {
            "dim": -1,
            "k": 6,
            "largest": True,
            "self": {"dtype": "float32", "shape": [4096, 64]},
            "sorted": False,
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::topk",
        {
            "dim": -1,
            "k": 1,
            "largest": True,
            "self": {"dtype": "float32", "shape": [4096, 16]},
            "sorted": False,
        },
        count=1,
        paths=("text-generation-torch-titan-multi-node-4N-async-hostcc",),
    )
    t(
        "aten::topk",
        {
            "dim": -1,
            "k": 10,
            "largest": True,
            "self": {"dtype": "float32", "shape": [96, 512]},
            "sorted": True,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::topk",
        {
            "dim": -1,
            "k": 2,
            "largest": True,
            "self": {"dtype": "bfloat16", "shape": [96, 1, 384]},
            "sorted": True,
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::tril",
        {"diagonal": 0, "self": {"dtype": "float32", "shape": [1, 64, 2, 64, 64]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::tril",
        {"diagonal": 0, "self": {"dtype": "float32", "shape": [1, 32, 2, 64, 64]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::tril",
        {"diagonal": 0, "self": {"dtype": "float32", "shape": [1, 32, 1, 64, 64]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::tril",
        {"diagonal": -1, "self": {"dtype": "bool", "shape": [256, 256]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::tril",
        {"diagonal": 0, "self": {"dtype": "bool", "shape": [256, 256]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::tril",
        {"diagonal": -1, "self": {"dtype": "bool", "shape": [128, 128]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::triu",
        {"diagonal": 0, "self": {"dtype": "bool", "shape": [64, 64]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::triu",
        {"diagonal": 1, "self": {"dtype": "bool", "shape": [64, 64]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::upsample_bilinear2d",
        {
            "align_corners": False,
            "output_size": [224, 224],
            "scales_h": None,
            "scales_w": None,
            "self": {"dtype": "float32", "shape": [1, 3, 256, 256]},
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::value_selecting_reduction_backward",
        {
            "dim": -1,
            "grad": {"dtype": "bfloat16", "shape": [1, 128, 32, 1]},
            "indices": {"dtype": "int64", "shape": [1, 128, 32, 1]},
            "keepdim": True,
            "sizes": [1, 128, 32, 33],
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::value_selecting_reduction_backward",
        {
            "dim": -1,
            "grad": {"dtype": "bfloat16", "shape": [1, 64, 32, 1]},
            "indices": {"dtype": "int64", "shape": [1, 64, 32, 1]},
            "keepdim": True,
            "sizes": [1, 64, 32, 33],
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::value_selecting_reduction_backward",
        {
            "dim": -1,
            "grad": {"dtype": "float32", "shape": [256, 8]},
            "indices": {"dtype": "int64", "shape": [256, 8]},
            "keepdim": True,
            "sizes": [256, 128],
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::value_selecting_reduction_backward",
        {
            "dim": -1,
            "grad": {"dtype": "float32", "shape": [96, 10]},
            "indices": {"dtype": "int64", "shape": [96, 10]},
            "keepdim": True,
            "sizes": [96, 512],
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::value_selecting_reduction_backward",
        {
            "dim": -1,
            "grad": {"dtype": "float32", "shape": [96, 8]},
            "indices": {"dtype": "int64", "shape": [96, 8]},
            "keepdim": True,
            "sizes": [96, 128],
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::value_selecting_reduction_backward",
        {
            "dim": -1,
            "grad": {"dtype": "float32", "shape": [96, 8]},
            "indices": {"dtype": "int64", "shape": [96, 8]},
            "keepdim": True,
            "sizes": [96, 8],
        },
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::where.self",
        {
            "condition": {"dtype": "bool", "shape": [8016, 4096]},
            "other": {"dtype": "float32", "shape": [8016, 4096]},
            "self": {"dtype": "float32", "shape": [8016, 4096]},
        },
        count=2,
        paths=(
            "text-generation-torch-titan-multi-node-4N-async-compile-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
        ),
    )
    t(
        "aten::where.self",
        {
            "condition": {"dtype": "bool", "shape": [2374, 5120]},
            "other": {"dtype": "float32", "shape": [2374, 5120]},
            "self": {"dtype": "float32", "shape": [2374, 5120]},
        },
        count=10,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::where.self",
        {
            "condition": {"dtype": "bool", "shape": [2374, 4096]},
            "other": {"dtype": "float32", "shape": [2374, 4096]},
            "self": {"dtype": "float32", "shape": [2374, 4096]},
        },
        count=12,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-dev",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-qwen3-235b-trn3",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::where.self",
        {
            "condition": {"dtype": "bool", "shape": [3142, 2880]},
            "other": {"dtype": "float32", "shape": [3142, 2880]},
            "self": {"dtype": "float32", "shape": [3142, 2880]},
        },
        count=3,
        paths=(
            "text-generation-torch-titan-multi-core-async-compile-moe",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::where.self",
        {
            "condition": {"dtype": "bool", "shape": [2004, 4096]},
            "other": {"dtype": "float32", "shape": [2004, 4096]},
            "self": {"dtype": "float32", "shape": [2004, 4096]},
        },
        count=9,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-compile-trn3",
            "text-generation-torch-titan-multi-core-compile-trn3-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-compile-hostcc",
        ),
    )
    t(
        "aten::where.self",
        {
            "condition": {"dtype": "bool", "shape": [2374, 2560]},
            "other": {"dtype": "float32", "shape": [2374, 2560]},
            "self": {"dtype": "float32", "shape": [2374, 2560]},
        },
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::zero_",
        {"self": {"dtype": "float32", "shape": [16384, 37984]}},
        count=2,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
        ),
    )
    t(
        "aten::zero_",
        {"self": {"dtype": "float32", "shape": [8192, 37984]}},
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
        ),
    )
    t(
        "aten::zero_",
        {"self": {"dtype": "bfloat16", "shape": [262144, 1024]}},
        count=1,
        paths=("test_kaizen_hf_onboarding_reduced_eager",),
    )
    t(
        "aten::zero_",
        {"self": {"dtype": "float32", "shape": [8192, 32064]}},
        count=5,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
        ),
    )
    t(
        "aten::zero_",
        {"self": {"dtype": "bfloat16", "shape": [50560, 5120]}},
        count=1,
        paths=("text-generation-torch-titan-multi-node-4N-async-hostcc",),
    )
    t(
        "aten::zero_",
        {"self": {"dtype": "float32", "shape": [4096, 50512]}},
        count=1,
        paths=("text-generation-torch-titan-multi-node-4N-async-hostcc",),
    )
    t(
        "aten::zeros",
        {"device": None, "dtype": 6, "layout": None, "pin_memory": None, "size": []},
        count=22,
        paths=(
            "test_kaizen_hf_onboarding_reduced_eager",
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-dev",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-compile-trn3",
            "text-generation-torch-titan-multi-core-compile-trn3-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-compile-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-hostcc",
            "text-generation-torch-titan-qwen3-235b-trn3",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::zeros",
        {"device": None, "dtype": 4, "layout": None, "pin_memory": None, "size": [64]},
        count=17,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3-hostcc",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-multi-core-compile-trn3",
            "text-generation-torch-titan-multi-core-compile-trn3-hostcc",
            "text-generation-torch-titan-multi-core-trn3",
            "text-generation-torch-titan-multi-core-trn3-hostcc",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn2",
            "text-generation-torch-titan-upstream_sync-dense-compile-trn3",
            "text-generation-torch-titan-upstream_sync-dense-trn2",
            "text-generation-torch-titan-upstream_sync-dense-trn3",
        ),
    )
    t(
        "aten::zeros",
        {"device": None, "dtype": 6, "layout": None, "pin_memory": None, "size": [291]},
        count=5,
        paths=(
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
            "text-generation-torch-titan-multi-core-compile-trn3",
            "text-generation-torch-titan-multi-core-compile-trn3-hostcc",
            "text-generation-torch-titan-multi-node-4N-async-compile-hostcc",
        ),
    )
    t(
        "aten::zeros",
        {"device": None, "dtype": 6, "layout": None, "pin_memory": None, "size": [435]},
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-hostcc",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3",
            "text-generation-torch-titan-multi-core-async-compile-moe-upstream_sync-trn3-hostcc",
        ),
    )
    t(
        "aten::zeros",
        {"device": None, "dtype": 6, "layout": None, "pin_memory": None, "size": [64]},
        count=4,
        paths=(
            "text-generation-torch-titan-multi-core-async",
            "text-generation-torch-titan-multi-core-async-compile-moe",
            "text-generation-torch-titan-multi-core-async-hostcc",
            "text-generation-torch-titan-qwen3-235b-trn3",
        ),
    )
    t(
        "aten::zeros",
        {"device": None, "dtype": 6, "layout": None, "pin_memory": None, "size": [399]},
        count=3,
        paths=(
            "text-generation-torch-titan-multi-core-async-compile",
            "text-generation-torch-titan-multi-core-async-compile-dev",
            "text-generation-torch-titan-multi-core-async-compile-hostcc",
        ),
    )


# END GENERATED CACHE-KEY SHAPES
