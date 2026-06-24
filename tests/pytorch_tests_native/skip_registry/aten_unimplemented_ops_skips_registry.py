"""Skip registry for ATen ops not natively lowered on Neuron (CPU fallback).

These 232 ATen ops do not have a native Neuron lowering. When invoked on
neuron tensors they bounce through CPU fallback, producing correct
results but providing no signal about Neuron's dispatch or execution
path. Running their OpInfo tests wastes ~61h per full suite run and
pollutes results with false xfails/xpasses.

Source: Athena query on ``aten_ops_generality_coverage``.

To re-enable tests for an op once Neuron support is added, delete the
op's pattern from the appropriate SkipGroup below and verify the tests
pass.

Ops are grouped by the upstream ``op_group`` tag (elementwise, nn,
foreach, etc.) for maintainability. Each SkipGroup lists concrete
fnmatch patterns that target the test method names produced by OpInfo.
"""

from tests.pytorch_tests_native.neuron_test_registry import SkipGroup

SKIP_GROUPS: list[SkipGroup] = [
    # ═══════════════════════════════════════════════════════════════
    # Elementwise ops not natively lowered on Neuron (125 ops).
    # ═══════════════════════════════════════════════════════════════
    # Includes trig/hyperbolic (acos, acosh, asin, asinh, atan, atan2,
    # atanh, cosh, sinh, tan), special functions (bessel, entr, erfc,
    # erfcx, gammainc/c, i0/i0e/i1/i1e, laguerre, log_ndtr, ndtri,
    # xlog1py, zeta, digamma, lgamma, mvlgamma, xlogy, polygamma,
    # igamma/c), secondary activations (hardshrink, hardsigmoid,
    # hardswish, hardtanh, leaky_relu, mish, selu, celu, elu, glu,
    # softshrink, rrelu, relu6, prelu), rounding/floor (floor, round,
    # frac, nan_to_num, fmod), log/exp variants (exp2, expm1, log10/1p/2,
    # logaddexp, logaddexp2), bitwise/integer-only (__rshift__, gcd, lcm,
    # hash_tensor), and misc (copysign, fmax, fmin, heaviside, hypot,
    # ldexp, nextafter, angle, isposinf, isin, sinc, addbmm, baddbmm,
    # logit, col2im, im2col, max_unpool2d/3d, kl_div, pairwise_distance,
    # ormqr, smooth_l1_loss_backward, huber_loss_backward,
    # multilabel_margin_loss_forward).
    # Core elementwise ops ARE supported natively: add, sub, mul, div,
    # abs, neg, exp, log, sqrt, rsqrt, pow, erf, erfinv, sin, cos, relu,
    # gelu, sigmoid, tanh, silu, softmax.
    SkipGroup(
        reason=(
            "Elementwise ops not natively lowered on Neuron — bounce"
            " through CPU fallback, providing no Neuron signal."
        ),
        patterns=[
            "*___rshift___neuron*",
            "*__rshift___neuron*",
            "*_acosh_neuron*",
            "*_addbmm_neuron*",
            "*_addbmm__neuron*",
            "*_angle_neuron*",
            "*_asin_neuron*",
            "*_asinh_neuron*",
            "*_atan_neuron*",
            "*_atan2_neuron*",
            "*_atanh_neuron*",
            "*_baddbmm_neuron*",
            "*_celu_neuron*",
            "*_col2im_neuron*",
            "*_copysign_neuron*",
            "*_copysign__neuron*",
            "*_cosh_neuron*",
            "*_digamma_neuron*",
            "*_digamma__neuron*",
            "*_elu_neuron*",
            "*_elu_backward_neuron*",
            "*_erfc_neuron*",
            "*_erfc__neuron*",
            "*_exp2_neuron*",
            "*_exp2__neuron*",
            "*_expm1_neuron*",
            "*_floor_neuron*",
            "*_floor_divide__neuron*",
            "*_fmax_neuron*",
            "*_fmin_neuron*",
            "*_fmod_neuron*",
            "*_frac_neuron*",
            "*_frac__neuron*",
            "*_gcd_neuron*",
            "*_gcd__neuron*",
            "*_glu_neuron*",
            "*_glu_backward_neuron*",
            "*_hardshrink_neuron*",
            "*_hardshrink_backward_neuron*",
            "*_hardsigmoid_neuron*",
            "*_hardsigmoid__neuron*",
            "*_hardsigmoid_backward_neuron*",
            "*_hardswish_neuron*",
            "*_hardswish__neuron*",
            "*_hardswish_backward_neuron*",
            "*_hardtanh_neuron*",
            "*_hardtanh__neuron*",
            "*_hardtanh_backward_neuron*",
            "*_hash_tensor_neuron*",
            "*_heaviside_neuron*",
            "*_heaviside__neuron*",
            "*_huber_loss_backward_neuron*",
            "*_hypot_neuron*",
            "*_hypot__neuron*",
            "*_i0_neuron*",
            "*_i0__neuron*",
            "*_igamma_neuron*",
            "*_igamma__neuron*",
            "*_igammac_neuron*",
            "*_igammac__neuron*",
            "*_im2col_neuron*",
            "*_isin_neuron*",
            "*_isposinf_neuron*",
            "*_kl_div_neuron*",
            "*_lcm_neuron*",
            "*_lcm__neuron*",
            "*_ldexp_neuron*",
            "*_leaky_relu_neuron*",
            "*_leaky_relu_backward_neuron*",
            "*_lgamma_neuron*",
            "*_lgamma__neuron*",
            "*_log10_neuron*",
            "*_log1p_neuron*",
            "*_log2_neuron*",
            "*_logaddexp_neuron*",
            "*_logaddexp2_neuron*",
            "*_logit_neuron*",
            "*_logit__neuron*",
            "*_logit_backward_neuron*",
            "*_max_unpool2d_neuron*",
            "*_max_unpool3d_neuron*",
            "*_mish_neuron*",
            "*_mish__neuron*",
            "*_multilabel_margin_loss_forward_neuron*",
            "*_mvlgamma_neuron*",
            "*_nan_to_num_neuron*",
            "*_nan_to_num__neuron*",
            "*_nextafter_neuron*",
            "*_nextafter__neuron*",
            "*_ormqr_neuron*",
            "*_pairwise_distance_neuron*",
            "*_polygamma_neuron*",
            "*_relu6_neuron*",
            "*_round_neuron*",
            "*_rrelu_neuron*",
            "*_rrelu__neuron*",
            "*_rrelu_with_noise_neuron*",
            "*_rrelu_with_noise__neuron*",
            "*_rrelu_with_noise_backward_neuron*",
            "*_rrelu_with_noise_functional_neuron*",
            "*_selu_neuron*",
            "*_selu__neuron*",
            "*_sinc_neuron*",
            "*_sinh_neuron*",
            "*_smooth_l1_loss_backward_neuron*",
            "*_softshrink_neuron*",
            "*_softshrink_backward_neuron*",
            "*_special_bessel_j0_neuron*",
            "*_special_bessel_j1_neuron*",
            "*_special_entr_neuron*",
            "*_special_erfc_neuron*",
            "*_special_erfcx_neuron*",
            "*_special_gammainc_neuron*",
            "*_special_gammaincc_neuron*",
            "*_special_i0e_neuron*",
            "*_special_i1_neuron*",
            "*_special_i1e_neuron*",
            "*_special_laguerre_polynomial_l_neuron*",
            "*_special_log_ndtr_neuron*",
            "*_special_ndtri_neuron*",
            "*_special_spherical_bessel_j0_neuron*",
            "*_special_xlog1py_neuron*",
            "*_special_zeta_neuron*",
            "*_tan_neuron*",
            "*_xlogy_neuron*",
        ],
        match_target="test_method",
    ),
    # ═══════════════════════════════════════════════════════════════
    # NN ops not natively lowered on Neuron (41 ops).
    # ═══════════════════════════════════════════════════════════════
    # Includes pooling variants (avg_pool1d/2d/3d + backward,
    # max_pool*_with_indices + backward, adaptive_max_pool2d/3d +
    # backward, fractional_max_pool2d/3d), fused RNN/LSTM cells
    # (_thnn_fused_gru_cell/backward, _thnn_fused_lstm_cell/backward_impl),
    # batch norm variants (_batch_norm_with_update,
    # _native_batch_norm_legit_functional, batch_norm_backward,
    # native_batch_norm), loss functions (binary_cross_entropy[_backward],
    # huber_loss, smooth_l1_loss, soft_margin_loss, multi_margin_loss,
    # nll_loss2d_forward/backward), upsample variants (upsample_bicubic2d,
    # upsample_linear1d, upsample_nearest1d, upsample_nearest2d_backward,
    # upsample_trilinear3d, _upsample_bilinear2d_aa_backward,
    # _upsample_nearest_exact1d), and prelu family (prelu, _prelu_kernel,
    # _prelu_kernel_backward).
    # Core NN ops ARE supported natively: max_pool, native_layer_norm,
    # native_group_norm, native_dropout, convolution, linear.
    SkipGroup(
        reason=(
            "NN ops not natively lowered on Neuron — bounce through CPU"
            " fallback, providing no Neuron signal."
        ),
        patterns=[
            "*__batch_norm_with_update_neuron*",
            "*__native_batch_norm_legit_functional_neuron*",
            "*__prelu_kernel_neuron*",
            "*__prelu_kernel_backward_neuron*",
            "*__thnn_fused_gru_cell_neuron*",
            "*__thnn_fused_gru_cell_backward_neuron*",
            "*__thnn_fused_lstm_cell_neuron*",
            "*__thnn_fused_lstm_cell_backward_impl_neuron*",
            "*__upsample_bicubic2d_aa_neuron*",
            "*__upsample_bicubic2d_aa_backward_neuron*",
            "*__upsample_bilinear2d_aa_neuron*",
            "*__upsample_bilinear2d_aa_backward_neuron*",
            "*__upsample_nearest_exact1d_neuron*",
            "*_adaptive_max_pool2d_neuron*",
            "*_adaptive_max_pool2d_backward_neuron*",
            "*_adaptive_max_pool3d_neuron*",
            "*_adaptive_max_pool3d_backward_neuron*",
            "*_avg_pool1d_neuron*",
            "*nn_functional_avg_pool2d_neuron*",
            "*_avg_pool2d_backward_neuron*",
            "*_avg_pool3d_neuron*",
            "*_avg_pool3d_backward_neuron*",
            "*_batch_norm_backward_neuron*",
            "*_binary_cross_entropy_neuron*",
            "*_binary_cross_entropy_backward_neuron*",
            "*_fractional_max_pool2d_neuron*",
            "*_fractional_max_pool3d_neuron*",
            "*_huber_loss_neuron*",
            "*_max_pool2d_with_indices_neuron*",
            "*_max_pool2d_with_indices_backward_neuron*",
            "*_max_pool3d_with_indices_neuron*",
            "*_max_pool3d_with_indices_backward_neuron*",
            "*_multi_margin_loss_neuron*",
            "*_native_batch_norm_neuron*",
            "*_nll_loss2d_backward_neuron*",
            "*_nll_loss2d_forward_neuron*",
            "*_prelu_neuron*",
            "*_smooth_l1_loss_neuron*",
            "*_soft_margin_loss_neuron*",
            "*_upsample_bicubic2d_neuron*",
            "*_upsample_bicubic2d_backward_neuron*",
            "*_upsample_bilinear2d_neuron*",
            "*_upsample_bilinear2d_backward_neuron*",
            "*_upsample_linear1d_neuron*",
            "*_upsample_nearest1d_neuron*",
            "*_upsample_nearest2d_backward_neuron*",
            "*_upsample_trilinear3d_neuron*",
        ],
        match_target="test_method",
    ),
    # ═══════════════════════════════════════════════════════════════
    # nn.LSTM / nn.GRU module tests — internally call _thnn_fused_*_cell
    # which has no native Neuron lowering AND CPU fallback is disabled.
    # ═══════════════════════════════════════════════════════════════
    # Some of these tests run for 60-150s before failing, wasting
    # significant compute. Skip rather than xfail to avoid that cost.
    SkipGroup(
        reason=(
            "nn.LSTM/nn.GRU module tests bottom out on _thnn_fused_*_cell"
            " (no native Neuron lowering, CPU fallback disabled). Skipped"
            " rather than xfailed because some variants run for 60-150s"
            " before failing."
        ),
        patterns=[
            "*nn_LSTM*neuron*",
            "*nn_GRU*neuron*",
            "*_LSTM_grad_and_gradgrad_neuron*",
            "*_GRU_grad_and_gradgrad_neuron*",
            "*_rnn_retain_variables_neuron*",
            "*_variable_sequence_neuron*",
        ],
        match_target="test_method",
    ),
    # ═══════════════════════════════════════════════════════════════
    # Foreach ops not natively lowered on Neuron (30 ops).
    # ═══════════════════════════════════════════════════════════════
    # All unary _foreach_* variants of ops missing from the unary list —
    # each decomposes to a primitive that itself requires CPU fallback.
    # Covered bases: acos, asin, atan, cosh, sinh, tan, erfc, expm1,
    # floor, round, frac, lgamma, log10, log1p, log2. Each has
    # non-mutating and in-place (_) variants.
    SkipGroup(
        reason=(
            "Foreach ops decomposing to non-lowered primitives — bounce"
            " through CPU fallback, providing no Neuron signal."
        ),
        patterns=[
            "*__foreach_acos_neuron*",
            "*__foreach_acos__neuron*",
            "*__foreach_asin_neuron*",
            "*__foreach_asin__neuron*",
            "*__foreach_atan_neuron*",
            "*__foreach_atan__neuron*",
            "*__foreach_cosh_neuron*",
            "*__foreach_cosh__neuron*",
            "*__foreach_erfc_neuron*",
            "*__foreach_erfc__neuron*",
            "*__foreach_expm1_neuron*",
            "*__foreach_expm1__neuron*",
            "*__foreach_floor_neuron*",
            "*__foreach_floor__neuron*",
            "*__foreach_frac_neuron*",
            "*__foreach_frac__neuron*",
            "*__foreach_lgamma_neuron*",
            "*__foreach_lgamma__neuron*",
            "*__foreach_log10_neuron*",
            "*__foreach_log10__neuron*",
            "*__foreach_log1p_neuron*",
            "*__foreach_log1p__neuron*",
            "*__foreach_log2_neuron*",
            "*__foreach_log2__neuron*",
            "*__foreach_round_neuron*",
            "*__foreach_round__neuron*",
            "*__foreach_sinh_neuron*",
            "*__foreach_sinh__neuron*",
            "*__foreach_tan_neuron*",
            "*__foreach_tan__neuron*",
        ],
        match_target="test_method",
    ),
    # ═══════════════════════════════════════════════════════════════
    # Reduction ops not natively lowered on Neuron (10 ops).
    # ═══════════════════════════════════════════════════════════════
    # Includes index-returning reductions (argmax, argmin, aminmax,
    # _aminmax), and special reduction helpers (_cummax_helper,
    # _cummin_helper, _logcumsumexp, nansum, bincount, renorm).
    # Core reductions ARE supported natively: sum, mean, amax, amin, all,
    # any, cumsum.
    SkipGroup(
        reason=(
            "Reduction ops not natively lowered on Neuron — bounce"
            " through CPU fallback, providing no Neuron signal."
        ),
        patterns=[
            "*__aminmax_neuron*",
            "*__cummax_helper_neuron*",
            "*__cummin_helper_neuron*",
            "*__logcumsumexp_neuron*",
            "*_aminmax_neuron*",
            "*_argmax_neuron*",
            "*_argmin_neuron*",
            "*_bincount_neuron*",
            "*_nansum_neuron*",
            "*_renorm_neuron*",
        ],
        match_target="test_method",
    ),
    # ═══════════════════════════════════════════════════════════════
    # Linalg ops not natively lowered on Neuron (7 ops).
    # ═══════════════════════════════════════════════════════════════
    # Advanced linalg operations. Core linalg IS supported: mm, bmm, dot,
    # mv, addmm, addmv, matmul, linalg_vector_norm, chain_matmul.
    SkipGroup(
        reason=(
            "Advanced linalg ops not natively lowered on Neuron — bounce"
            " through CPU fallback, providing no Neuron signal."
        ),
        patterns=[
            "*__addmm_activation_neuron*",
            "*__linalg_det_neuron*",
            "*_linalg_cross_neuron*",
            "*_linalg_lstsq_neuron*",
            "*_linalg_matrix_exp_neuron*",
            "*_linalg_svd_neuron*",
            "*_linalg_svdvals_neuron*",
        ],
        match_target="test_method",
    ),
    # ═══════════════════════════════════════════════════════════════
    # Indexing ops not natively lowered on Neuron (7 ops).
    # ═══════════════════════════════════════════════════════════════
    # Advanced indexing variants. Core indexing IS supported: index,
    # index_add, index_copy, index_put, embedding, gather, scatter,
    # as_strided_scatter, diagonal_scatter.
    SkipGroup(
        reason=(
            "Advanced indexing ops not natively lowered on Neuron —"
            " bounce through CPU fallback, providing no Neuron signal."
        ),
        patterns=[
            "*_index_reduce_neuron*",
            "*_index_reduce__neuron*",
            "*_put_neuron*",
            "*_put__neuron*",
            "*_scatter_reduce_neuron*",
            "*_searchsorted_neuron*",
            "*_take_neuron*",
        ],
        match_target="test_method",
    ),
    # ═══════════════════════════════════════════════════════════════
    # Random ops not natively lowered on Neuron (7 ops).
    # ═══════════════════════════════════════════════════════════════
    # No native Neuron RNG — all random in-place ops bounce through CPU.
    SkipGroup(
        reason=(
            "Random ops not natively lowered on Neuron — no native RNG,"
            " all variants bounce through CPU fallback."
        ),
        patterns=[
            "*_cauchy__neuron*",
            "*_exponential__neuron*",
            "*_geometric__neuron*",
            "*_log_normal__neuron*",
            "*_normal__neuron*",
            "*_random__neuron*",
            "*_uniform__neuron*",
        ],
        match_target="test_method",
    ),
    # ═══════════════════════════════════════════════════════════════
    # FFT ops not natively lowered on Neuron (3 ops).
    # ═══════════════════════════════════════════════════════════════
    # FFT is entirely unsupported on Neuron hardware.
    SkipGroup(
        reason=(
            "FFT ops not supported on Neuron hardware — bounce through"
            " CPU fallback, providing no Neuron signal."
        ),
        patterns=[
            "*__fft_c2c_neuron*",
            "*__fft_c2r_neuron*",
            "*__fft_r2c_neuron*",
        ],
        match_target="test_method",
    ),
    # ═══════════════════════════════════════════════════════════════
    # Shape ops not natively lowered on Neuron (1 op).
    # ═══════════════════════════════════════════════════════════════
    # Only replication_pad1d — 2d/3d variants ARE supported natively.
    SkipGroup(
        reason=(
            "replication_pad1d not natively lowered on Neuron (2d/3d"
            " variants are supported) — bounces through CPU fallback."
        ),
        patterns=[
            "*_replication_pad1d_neuron*",
        ],
        match_target="test_method",
    ),
]
