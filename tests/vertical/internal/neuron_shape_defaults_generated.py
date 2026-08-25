"""Auto-generated shape registry entries for pending ops."""

# Keep co-varying list lengths (e.g. foreach scalars[] alongside self[]) aligned
# with the tensor-list default count used by _tensor_list_from_spec. If this
# constant changes, regenerate this file.
from .neuron_op_info import TENSOR_LIST_DEFAULT_COUNT as _N


def register_all_defaults(r, traced_models=()):
    """Register all shape entries (756 total).

    ``traced_models`` names model-traced shape sets to register on top of the builtin defaults
    (see ``TRACED_MODEL_SHAPES``). It is empty by default so a plain run sees exactly the
    builtin pool it always has; naming one model uses ``--traced-model``. Keeping this an
    allowlist rather than a default-on, which make model added ops not impact
    pre-existing vertical tests.
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

    # Index put impl (in-place)
    r(
        "aten::_index_put_impl_",
        {
            "self": (4096, 4096),
            "indices": [{"shape": (256,), "dtype": "long", "low": 0, "high": 4096}],
            "values": (256, 4096),
            "accumulate": False,
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
        {"self": (256, 256), "boundaries": (32,)},
        source="builtin",
    )
    r(
        "aten::bucketize.Scalar",
        {"self": 0.5, "boundaries": (32,)},
        source="builtin",
    )
    r(
        "aten::bucketize.Scalar_out",
        {"self": 0.5, "boundaries": (32,), "out": {"shape": (), "dtype": "long"}},
        source="builtin",
    )
    r(
        "aten::searchsorted.Tensor",
        {"sorted_sequence": (32,), "self": (256, 256)},
        source="builtin",
    )
    r(
        "aten::searchsorted.Scalar",
        {"sorted_sequence": (32,), "self": 0.5},
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

    # Model-traced sets, opt-in via --traced-model. Unknown names fail loud.
    for _name in traced_models:
        try:
            _register_model = TRACED_MODEL_SHAPES[_name]
        except KeyError:
            raise KeyError(
                f"unknown --traced-model {_name!r}; available: {sorted(TRACED_MODEL_SHAPES)}"
            ) from None
        _register_model(r)


QWEN3_235B_A22B = "qwen3_235b_a22b"

QWEN3_235B_A22B_OPS = frozenset(
    {
        "aten::_softmax",
        "aten::_to_copy",
        "aten::add.Scalar",
        "aten::add.Tensor",
        "aten::add_.Tensor",
        "aten::arange",
        "aten::arange.start",
        "aten::bitwise_and.Tensor",
        "aten::bmm",
        "aten::cat",
        "aten::clamp_min",
        "aten::constant_pad_nd",
        "aten::cumsum",
        "aten::div.Tensor",
        "aten::eq.Scalar",
        "aten::eq.Tensor",
        "aten::exp",
        "aten::floor_divide",
        "aten::full",
        "aten::gather",
        "aten::ge.Scalar",
        "aten::index.Tensor",
        "aten::le.Scalar",
        "aten::le.Tensor",
        "aten::linear",
        "aten::logical_and",
        "aten::logical_not",
        "aten::max",
        "aten::max.dim",
        "aten::maximum",
        "aten::mean.dim",
        "aten::min",
        "aten::minimum",
        "aten::mm",
        "aten::mul.Tensor",
        "aten::neg",
        "aten::new_ones",
        "aten::new_zeros",
        "aten::ones",
        "aten::ones_like",
        "aten::pow.Tensor_Scalar",
        "aten::rsub.Scalar",
        "aten::sigmoid",
        "aten::silu",
        "aten::silu_backward",
        "aten::slice_backward",
        "aten::split.Tensor",
        "aten::stack",
        "aten::sub.Tensor",
        "aten::tril",
        "aten::triu",
        "aten::where.self",
        "aten::zeros",
        "aten::zeros_like",
    }
)


def _register_qwen3_235b_a22b(r):
    """Qwen3-235B-A22B traced shapes: 181 entries over 54 ops."""
    r(
        "aten::_softmax",
        {"dim": 1, "half_to_float": False, "self": {"dtype": "float32", "shape": (128, 8)}},
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::_to_copy",
        {"dtype": "bfloat16", "self": {"dtype": "float32", "shape": (1024, 1536)}},
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::_to_copy",
        {"dtype": "bfloat16", "self": {"dtype": "float32", "shape": (1024, 4096)}},
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::_to_copy",
        {"dtype": "bfloat16", "self": {"dtype": "float32", "shape": (2048, 4096)}},
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::_to_copy",
        {"dtype": "bfloat16", "self": {"dtype": "float32", "shape": (8, 1536, 4096)}},
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::_to_copy",
        {"dtype": "bfloat16", "self": {"dtype": "float32", "shape": (8, 4096, 1536)}},
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::_to_copy",
        {"dtype": "float32", "self": {"dtype": "bfloat16", "shape": (1024, 1536)}},
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::_to_copy",
        {"dtype": "float32", "self": {"dtype": "bfloat16", "shape": (1024, 4096)}},
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::_to_copy",
        {"dtype": "float32", "self": {"dtype": "bfloat16", "shape": (2048, 4096)}},
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::_to_copy",
        {"dtype": "float32", "self": {"dtype": "bfloat16", "shape": (8, 1536, 4096)}},
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::_to_copy",
        {"dtype": "float32", "self": {"dtype": "bfloat16", "shape": (8, 4096, 1536)}},
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::_to_copy",
        {"dtype": "float32", "self": {"dtype": "int64", "shape": (128,)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::_to_copy",
        {"dtype": "int32", "self": {"dtype": "int32", "shape": (8,)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::_to_copy",
        {"dtype": "int32", "self": {"dtype": "int64", "shape": (128,)}},
        source=QWEN3_235B_A22B,
        count=3,
    )
    r(
        "aten::_to_copy",
        {"dtype": "int32", "self": {"dtype": "int64", "shape": (8,)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::_to_copy",
        {"dtype": "int64", "self": {"dtype": "float32", "shape": (1024,)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::_to_copy",
        {"dtype": "int64", "self": {"dtype": "int64", "shape": (2048,)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::_to_copy",
        {"dtype": "int64", "self": {"dtype": "int64", "shape": (2,)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::_to_copy",
        {"dtype": "int64", "self": {"dtype": "int64", "shape": (8,)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::add.Scalar",
        {"other": 1e-06, "self": {"dtype": "float32", "shape": (1, 128, 1)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::add.Scalar",
        {"other": 1e-06, "self": {"dtype": "float32", "shape": (1, 128, 4, 1)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::add.Scalar",
        {"other": 1e-06, "self": {"dtype": "float32", "shape": (1, 128, 64, 1)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::add.Tensor",
        {"other": 1, "self": {"dtype": "float32", "shape": (1024, 1536)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::add.Tensor",
        {"other": 128, "self": {"dtype": "int64", "shape": (8,)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::add.Tensor",
        {"other": 1e-20, "self": {"dtype": "float32", "shape": (128, 1)}},
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::add.Tensor",
        {
            "other": {"dtype": "float32", "shape": (1, 128, 4, 128)},
            "self": {"dtype": "float32", "shape": (1, 128, 4, 128)},
        },
        source=QWEN3_235B_A22B,
        count=4,
    )
    r(
        "aten::add.Tensor",
        {
            "other": {"dtype": "float32", "shape": (1, 128, 4096)},
            "self": {"dtype": "float32", "shape": (1, 128, 4096)},
        },
        source=QWEN3_235B_A22B,
        count=4,
    )
    r(
        "aten::add.Tensor",
        {
            "other": {"dtype": "float32", "shape": (1, 128, 64, 128)},
            "self": {"dtype": "float32", "shape": (1, 128, 64, 128)},
        },
        source=QWEN3_235B_A22B,
        count=4,
    )
    r(
        "aten::add.Tensor",
        {
            "other": {"dtype": "float32", "shape": (1, 64, 128, 1)},
            "self": {"dtype": "float32", "shape": (1, 64, 128, 1)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::add.Tensor",
        {
            "other": {"dtype": "float32", "shape": (1, 64, 128)},
            "self": {"dtype": "float32", "shape": (1, 64, 128)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::add.Tensor",
        {
            "other": {"dtype": "float32", "shape": (1024, 4096)},
            "self": {"dtype": "float32", "shape": (1024, 4096)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::add.Tensor",
        {
            "other": {"dtype": "float32", "shape": (128, 4096)},
            "self": {"dtype": "float32", "shape": (128, 4096)},
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::add.Tensor",
        {
            "other": {"dtype": "float32", "shape": (128, 8)},
            "self": {"dtype": "float32", "shape": (128, 8)},
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::add.Tensor",
        {
            "other": {"dtype": "float32", "shape": (2048, 4096)},
            "self": {"dtype": "float32", "shape": (2048, 4096)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::add.Tensor",
        {
            "other": {"dtype": "float32", "shape": (8,)},
            "self": {"dtype": "float32", "shape": (128, 8)},
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::add.Tensor",
        {"other": {"dtype": "int64", "shape": (8,)}, "self": {"dtype": "float32", "shape": (8,)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::add_.Tensor",
        {"other": {"dtype": "int64", "shape": (8,)}, "self": {"dtype": "float32", "shape": (8,)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::arange",
        {"dtype": "int64", "end": 8},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::arange",
        {"end": 8},
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::arange.start",
        {"dtype": "int64", "end": 1024, "start": 896},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::arange.start",
        {"dtype": "int64", "end": 128, "start": 0},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::arange.start",
        {"dtype": "int64", "end": 256, "start": 128},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::arange.start",
        {"dtype": "int64", "end": 384, "start": 256},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::arange.start",
        {"dtype": "int64", "end": 512, "start": 384},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::arange.start",
        {"dtype": "int64", "end": 640, "start": 512},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::arange.start",
        {"dtype": "int64", "end": 768, "start": 640},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::arange.start",
        {"dtype": "int64", "end": 896, "start": 768},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::bitwise_and.Tensor",
        {"other": {"dtype": "bool", "shape": (1024,)}, "self": {"dtype": "bool", "shape": (1024,)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::bmm",
        {
            "mat2": {"dtype": "bfloat16", "shape": (8, 1536, 4096)},
            "self": {"dtype": "bfloat16", "shape": (8, 128, 1536)},
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::bmm",
        {
            "mat2": {"dtype": "bfloat16", "shape": (8, 4096, 1536)},
            "self": {"dtype": "bfloat16", "shape": (8, 128, 4096)},
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::bmm",
        {
            "mat2": {"dtype": "float32", "shape": (128, 1, 4096)},
            "self": {"dtype": "float32", "shape": (128, 8, 1)},
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::bmm",
        {
            "mat2": {"dtype": "float32", "shape": (128, 4096, 8)},
            "self": {"dtype": "float32", "shape": (128, 1, 4096)},
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::bmm",
        {
            "mat2": {"dtype": "float32", "shape": (128, 8, 4096)},
            "self": {"dtype": "float32", "shape": (128, 1, 8)},
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::bmm",
        {
            "mat2": {"dtype": "float32", "shape": (64, 128, 128)},
            "self": {"dtype": "float32", "shape": (64, 128, 128)},
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::cat",
        {
            "dim": -1,
            "tensors": [
                {"dtype": "float32", "shape": (1, 128, 4, 64)},
                {"dtype": "float32", "shape": (1, 128, 4, 64)},
            ],
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::cat",
        {
            "dim": -1,
            "tensors": [
                {"dtype": "float32", "shape": (1, 128, 64, 64)},
                {"dtype": "float32", "shape": (1, 128, 64, 64)},
            ],
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::cat",
        {
            "tensors": [
                {"dtype": "float32", "shape": (1024, 4096)},
                {"dtype": "float32", "shape": (1, 4096)},
            ]
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::clamp_min",
        {"min": 128, "self": {"dtype": "int64", "shape": (8,)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::constant_pad_nd",
        {"pad": (0, 64), "self": {"dtype": "float32", "shape": (1, 128, 4, 64)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::constant_pad_nd",
        {"pad": (0, 64), "self": {"dtype": "float32", "shape": (1, 128, 64, 64)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::constant_pad_nd",
        {"pad": (64, 0), "self": {"dtype": "float32", "shape": (1, 128, 4, 64)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::constant_pad_nd",
        {"pad": (64, 0), "self": {"dtype": "float32", "shape": (1, 128, 64, 64)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::cumsum",
        {"dim": 0, "dtype": "int32", "self": {"dtype": "int32", "shape": (8,)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::cumsum",
        {"dim": 0, "dtype": "int32", "self": {"dtype": "int64", "shape": (8,)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::cumsum",
        {"dim": 0, "self": {"dtype": "int32", "shape": (8,)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::cumsum",
        {"dim": 0, "self": {"dtype": "int64", "shape": (8,)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::div.Tensor",
        {"other": 1.0, "self": {"dtype": "int64", "shape": (1024,)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::div.Tensor",
        {"other": 128, "self": {"dtype": "float32", "shape": (1, 128, 4, 128)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::div.Tensor",
        {"other": 128, "self": {"dtype": "float32", "shape": (1, 128, 64, 128)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::div.Tensor",
        {"other": 4096, "self": {"dtype": "float32", "shape": (1, 128, 4096)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::div.Tensor",
        {
            "other": {"dtype": "float32", "shape": (1, 64, 128, 1)},
            "self": {"dtype": "float32", "shape": (1, 64, 128, 128)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::div.Tensor",
        {
            "other": {"dtype": "float32", "shape": (128, 1)},
            "self": {"dtype": "float32", "shape": (128, 8)},
        },
        source=QWEN3_235B_A22B,
        count=4,
    )
    r(
        "aten::eq.Scalar",
        {"other": -100, "self": {"dtype": "int64", "shape": (128,)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::eq.Tensor",
        {
            "other": {"dtype": "int64", "shape": (1, 1024)},
            "self": {"dtype": "int64", "shape": (8, 1)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::exp",
        {"self": {"dtype": "float32", "shape": (1, 64, 128, 128)}},
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::floor_divide",
        {"other": 128, "self": {"dtype": "int32", "shape": (8, 1)}},
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::floor_divide",
        {"other": 128, "self": {"dtype": "int64", "shape": (8,)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::floor_divide",
        {"other": 8, "self": {"dtype": "int64", "shape": (1024,)}},
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::full",
        {"dtype": "int64", "fill_value": -1, "size": (2048,)},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::full",
        {"dtype": "int64", "fill_value": 0, "size": []},
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::full",
        {"dtype": "int64", "fill_value": 7, "size": []},
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::gather",
        {
            "dim": 1,
            "index": {"dtype": "int64", "shape": (128, 8)},
            "self": {"dtype": "float32", "shape": (128, 8)},
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::ge.Scalar",
        {"other": 0, "self": {"dtype": "int64", "shape": (1024,)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::index.Tensor",
        {
            "indices": [{"dtype": "int64", "shape": (1, 128)}],
            "self": {"dtype": "float32", "shape": (151936, 4096)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::index.Tensor",
        {
            "indices": [{"dtype": "int64", "shape": (1024,)}],
            "self": {"dtype": "float32", "shape": (1024, 4096)},
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::index.Tensor",
        {
            "indices": [{"dtype": "int64", "shape": (1024,)}],
            "self": {"dtype": "float32", "shape": (1024,)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::index.Tensor",
        {
            "indices": [{"dtype": "int64", "shape": (1024,)}],
            "self": {"dtype": "float32", "shape": (128, 4096)},
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::index.Tensor",
        {
            "indices": [{"dtype": "int64", "shape": (128,)}],
            "self": {"dtype": "float32", "shape": (128, 4096)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::index.Tensor",
        {
            "indices": [{"dtype": "int64", "shape": (128,)}],
            "self": {"dtype": "int64", "shape": (128,)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::index.Tensor",
        {
            "indices": [{"dtype": "int64", "shape": (2048,)}],
            "self": {"dtype": "float32", "shape": (1025, 4096)},
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::index.Tensor",
        {
            "indices": [{"dtype": "int64", "shape": (8,)}],
            "self": {"dtype": "bfloat16", "shape": (8, 1536, 4096)},
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::index.Tensor",
        {
            "indices": [{"dtype": "int64", "shape": (8,)}],
            "self": {"dtype": "bfloat16", "shape": (8, 4096, 1536)},
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::le.Scalar",
        {"other": 8, "self": {"dtype": "int64", "shape": (1024,)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::le.Tensor",
        {"other": {"dtype": "int64", "shape": (8,)}, "self": {"dtype": "int32", "shape": (8, 1)}},
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::linear",
        {
            "input": {"dtype": "float32", "shape": (1, 128, 4096)},
            "weight": {"dtype": "float32", "shape": (151936, 4096)},
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::linear",
        {
            "input": {"dtype": "float32", "shape": (1, 128, 4096)},
            "weight": {"dtype": "float32", "shape": (512, 4096)},
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::linear",
        {
            "input": {"dtype": "float32", "shape": (1, 128, 4096)},
            "weight": {"dtype": "float32", "shape": (8192, 4096)},
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::linear",
        {
            "input": {"dtype": "float32", "shape": (1, 128, 8192)},
            "weight": {"dtype": "float32", "shape": (4096, 8192)},
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::linear",
        {
            "input": {"dtype": "float32", "shape": (128, 4096)},
            "weight": {"dtype": "float32", "shape": (8, 4096)},
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::logical_and",
        {
            "other": {"dtype": "bool", "shape": (8, 1024)},
            "self": {"dtype": "bool", "shape": (1, 1024)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::logical_not",
        {"self": {"dtype": "bool", "shape": (128, 128)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::max",
        {"self": {"dtype": "int64", "shape": (128,)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::max.dim",
        {"dim": -1, "keepdim": True, "self": {"dtype": "float32", "shape": (1, 64, 128, 128)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::maximum",
        {"other": {"dtype": "int64", "shape": []}, "self": {"dtype": "int64", "shape": (1024,)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::maximum",
        {"other": {"dtype": "int64", "shape": []}, "self": {"dtype": "int64", "shape": (8,)}},
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::mean.dim",
        {"dim": (2,), "keepdim": True, "self": {"dtype": "float32", "shape": (1, 128, 4096)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::mean.dim",
        {"dim": (3,), "keepdim": True, "self": {"dtype": "float32", "shape": (1, 128, 4, 128)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::mean.dim",
        {"dim": (3,), "keepdim": True, "self": {"dtype": "float32", "shape": (1, 128, 64, 128)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::min",
        {"self": {"dtype": "int64", "shape": (128,)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::minimum",
        {"other": {"dtype": "int64", "shape": []}, "self": {"dtype": "int64", "shape": (1024,)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::minimum",
        {"other": {"dtype": "int64", "shape": []}, "self": {"dtype": "int64", "shape": (8,)}},
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::mm",
        {
            "mat2": {"dtype": "float32", "shape": (128, 4096)},
            "self": {"dtype": "float32", "shape": (151936, 128)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::mm",
        {
            "mat2": {"dtype": "float32", "shape": (128, 4096)},
            "self": {"dtype": "float32", "shape": (512, 128)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::mm",
        {
            "mat2": {"dtype": "float32", "shape": (128, 4096)},
            "self": {"dtype": "float32", "shape": (8, 128)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::mm",
        {
            "mat2": {"dtype": "float32", "shape": (128, 4096)},
            "self": {"dtype": "float32", "shape": (8192, 128)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::mm",
        {
            "mat2": {"dtype": "float32", "shape": (128, 8192)},
            "self": {"dtype": "float32", "shape": (4096, 128)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::mm",
        {
            "mat2": {"dtype": "float32", "shape": (151936, 4096)},
            "self": {"dtype": "float32", "shape": (128, 151936)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::mm",
        {
            "mat2": {"dtype": "float32", "shape": (4096, 8192)},
            "self": {"dtype": "float32", "shape": (128, 4096)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::mm",
        {
            "mat2": {"dtype": "float32", "shape": (512, 4096)},
            "self": {"dtype": "float32", "shape": (128, 512)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::mm",
        {
            "mat2": {"dtype": "float32", "shape": (8, 4096)},
            "self": {"dtype": "float32", "shape": (128, 8)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::mm",
        {
            "mat2": {"dtype": "float32", "shape": (8192, 4096)},
            "self": {"dtype": "float32", "shape": (128, 8192)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::mul.Tensor",
        {"other": 0.08838834764831845, "self": {"dtype": "float32", "shape": (1, 64, 128, 128)}},
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::mul.Tensor",
        {"other": 1.0, "self": {"dtype": "float32", "shape": (128, 8)}},
        source=QWEN3_235B_A22B,
        count=4,
    )
    r(
        "aten::mul.Tensor",
        {"other": 128, "self": {"dtype": "int64", "shape": (8,)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::mul.Tensor",
        {
            "other": {"dtype": "bfloat16", "shape": (1024, 1536)},
            "self": {"dtype": "bfloat16", "shape": (1024, 1536)},
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::mul.Tensor",
        {
            "other": {"dtype": "bfloat16", "shape": (2048, 1536)},
            "self": {"dtype": "bfloat16", "shape": (2048, 1536)},
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::mul.Tensor",
        {
            "other": {"dtype": "float32", "shape": (1, 128, 1, 128)},
            "self": {"dtype": "float32", "shape": (1, 128, 4, 128)},
        },
        source=QWEN3_235B_A22B,
        count=4,
    )
    r(
        "aten::mul.Tensor",
        {
            "other": {"dtype": "float32", "shape": (1, 128, 1, 128)},
            "self": {"dtype": "float32", "shape": (1, 128, 64, 128)},
        },
        source=QWEN3_235B_A22B,
        count=4,
    )
    r(
        "aten::mul.Tensor",
        {
            "other": {"dtype": "float32", "shape": (1, 128, 1)},
            "self": {"dtype": "float32", "shape": (1, 128, 4096)},
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::mul.Tensor",
        {
            "other": {"dtype": "float32", "shape": (1, 128, 4, 128)},
            "self": {"dtype": "float32", "shape": (1, 128, 4, 128)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::mul.Tensor",
        {
            "other": {"dtype": "float32", "shape": (1, 128, 4, 1)},
            "self": {"dtype": "float32", "shape": (1, 128, 4, 128)},
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::mul.Tensor",
        {
            "other": {"dtype": "float32", "shape": (1, 128, 4096)},
            "self": {"dtype": "float32", "shape": (1, 128, 4096)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::mul.Tensor",
        {
            "other": {"dtype": "float32", "shape": (1, 128, 64, 128)},
            "self": {"dtype": "float32", "shape": (1, 128, 64, 128)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::mul.Tensor",
        {
            "other": {"dtype": "float32", "shape": (1, 128, 64, 1)},
            "self": {"dtype": "float32", "shape": (1, 128, 64, 128)},
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::mul.Tensor",
        {
            "other": {"dtype": "float32", "shape": (1, 64, 128, 128)},
            "self": {"dtype": "float32", "shape": (1, 64, 128, 128)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::mul.Tensor",
        {
            "other": {"dtype": "float32", "shape": (1024, 1536)},
            "self": {"dtype": "float32", "shape": (1024, 1536)},
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::mul.Tensor",
        {
            "other": {"dtype": "float32", "shape": (128, 1)},
            "self": {"dtype": "float32", "shape": (128, 8)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::mul.Tensor",
        {
            "other": {"dtype": "float32", "shape": (128, 8)},
            "self": {"dtype": "float32", "shape": (128, 8)},
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::mul.Tensor",
        {
            "other": {"dtype": "float32", "shape": (128,)},
            "self": {"dtype": "float32", "shape": (1, 128, 4, 128)},
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::mul.Tensor",
        {
            "other": {"dtype": "float32", "shape": (128,)},
            "self": {"dtype": "float32", "shape": (1, 128, 64, 128)},
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::mul.Tensor",
        {
            "other": {"dtype": "float32", "shape": (4096,)},
            "self": {"dtype": "float32", "shape": (1, 128, 4096)},
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::neg",
        {"self": {"dtype": "float32", "shape": (1, 128, 4, 64)}},
        source=QWEN3_235B_A22B,
        count=4,
    )
    r(
        "aten::neg",
        {"self": {"dtype": "float32", "shape": (1, 128, 64, 64)}},
        source=QWEN3_235B_A22B,
        count=4,
    )
    r(
        "aten::neg",
        {"self": {"dtype": "float32", "shape": (128, 8)}},
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::neg",
        {"self": {"dtype": "float32", "shape": (128,)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::new_ones",
        {
            "dtype": "bool",
            "self": {"dtype": "float32", "shape": (1, 64, 128, 128)},
            "size": (128, 128),
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::new_zeros",
        {
            "dtype": "float32",
            "self": {"dtype": "float32", "shape": (1024, 4096)},
            "size": (128, 4096),
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::new_zeros",
        {
            "dtype": "float32",
            "self": {"dtype": "float32", "shape": (2048, 4096)},
            "size": (1025, 4096),
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::new_zeros",
        {"self": {"dtype": "float32", "shape": (1024, 4096)}, "size": (4096,)},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::new_zeros",
        {"self": {"dtype": "float32", "shape": (128, 8)}, "size": (128, 8)},
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::ones",
        {"dtype": "bool", "size": (128, 128)},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::ones_like",
        {"self": {"dtype": "float32", "shape": []}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::pow.Tensor_Scalar",
        {"exponent": 2, "self": {"dtype": "float32", "shape": (1, 128, 4, 128)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::pow.Tensor_Scalar",
        {"exponent": 2, "self": {"dtype": "float32", "shape": (1, 128, 4096)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::pow.Tensor_Scalar",
        {"exponent": 2, "self": {"dtype": "float32", "shape": (1, 128, 64, 128)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::rsub.Scalar",
        {"other": 1, "self": {"dtype": "float32", "shape": (1024, 1536)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::sigmoid",
        {"self": {"dtype": "float32", "shape": (1024, 1536)}},
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::silu",
        {"self": {"dtype": "bfloat16", "shape": (2048, 1536)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::silu_backward",
        {
            "grad_output": {"dtype": "bfloat16", "shape": (2048, 1536)},
            "self": {"dtype": "bfloat16", "shape": (2048, 1536)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::slice_backward",
        {
            "dim": 3,
            "end": 64,
            "grad_output": {"dtype": "float32", "shape": (1, 128, 4, 64)},
            "input_sizes": (1, 128, 4, 128),
            "start": 0,
            "step": 1,
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::slice_backward",
        {
            "dim": 3,
            "end": 64,
            "grad_output": {"dtype": "float32", "shape": (1, 128, 64, 64)},
            "input_sizes": (1, 128, 64, 128),
            "start": 0,
            "step": 1,
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::slice_backward",
        {
            "dim": 3,
            "end": 9223372036854775807,
            "grad_output": {"dtype": "float32", "shape": (1, 128, 4, 64)},
            "input_sizes": (1, 128, 4, 128),
            "start": 64,
            "step": 1,
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::slice_backward",
        {
            "dim": 3,
            "end": 9223372036854775807,
            "grad_output": {"dtype": "float32", "shape": (1, 128, 64, 64)},
            "input_sizes": (1, 128, 64, 128),
            "start": 64,
            "step": 1,
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::split.Tensor",
        {"self": {"dtype": "float32", "shape": (1025, 4096)}, "split_size": 1024},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::stack",
        {"tensors": [{"dtype": "int64", "shape": []}, {"dtype": "int64", "shape": []}]},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::sub.Tensor",
        {"other": 0, "self": {"dtype": "int64", "shape": (1024,)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::sub.Tensor",
        {"other": 1, "self": {"dtype": "int64", "shape": (8,)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::sub.Tensor",
        {
            "other": {"dtype": "float32", "shape": (1, 128, 4, 128)},
            "self": {"dtype": "float32", "shape": (1, 128, 4, 128)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::sub.Tensor",
        {
            "other": {"dtype": "float32", "shape": (1, 128, 4096)},
            "self": {"dtype": "float32", "shape": (1, 128, 4096)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::sub.Tensor",
        {
            "other": {"dtype": "float32", "shape": (1, 128, 64, 128)},
            "self": {"dtype": "float32", "shape": (1, 128, 64, 128)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::sub.Tensor",
        {
            "other": {"dtype": "float32", "shape": (1, 64, 128, 128)},
            "self": {"dtype": "float32", "shape": (1, 64, 128, 128)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::sub.Tensor",
        {
            "other": {"dtype": "float32", "shape": (1, 64, 128, 1)},
            "self": {"dtype": "float32", "shape": (1, 64, 128, 128)},
        },
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::sub.Tensor",
        {
            "other": {"dtype": "float32", "shape": (128, 8)},
            "self": {"dtype": "float32", "shape": (128, 8)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::sub.Tensor",
        {"other": {"dtype": "int32", "shape": (8,)}, "self": {"dtype": "int64", "shape": (8,)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::sub.Tensor",
        {"other": {"dtype": "int64", "shape": (8,)}, "self": {"dtype": "int64", "shape": (8,)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::tril",
        {"self": {"dtype": "bool", "shape": (128, 128)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::triu",
        {"diagonal": 1, "self": {"dtype": "bool", "shape": (128, 128)}},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::where.self",
        {
            "condition": {"dtype": "bool", "shape": (128,)},
            "other": {"dtype": "int64", "shape": (128,)},
            "self": {"dtype": "int64", "shape": (128,)},
        },
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::zeros",
        {"dtype": "float32", "size": (1, 4096)},
        source=QWEN3_235B_A22B,
        count=1,
    )
    r(
        "aten::zeros",
        {"dtype": "float32", "size": (1024, 4096)},
        source=QWEN3_235B_A22B,
        count=2,
    )
    r(
        "aten::zeros_like",
        {"self": {"dtype": "int64", "shape": (128,)}},
        source=QWEN3_235B_A22B,
        count=1,
    )


# Model-traced shape sets, keyed by the name ``--traced-model`` accepts.
TRACED_MODEL_SHAPES = {
    QWEN3_235B_A22B: _register_qwen3_235b_a22b,
}

TRACED_MODEL_OPS = {
    QWEN3_235B_A22B: QWEN3_235B_A22B_OPS,
}
