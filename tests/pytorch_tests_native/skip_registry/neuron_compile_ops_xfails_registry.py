from tests.pytorch_tests_native.neuron_test_registry import SkipGroup

XFAIL_GROUPS: list[SkipGroup] = [
    # TODO: Update this test to compare only the metadata for an empty result.
    SkipGroup(
        reason=("empty returns uninitialized data. The CPU and Neuron values are not comparable."),
        tests=[
            "test_neuron_compile_ops.py::TestCompileOpInfoNEURON::test_compile_matches_eager_empty_neuron_float64",
        ],
    ),
    # TODO: Remove this XFAIL after the compiler error is fixed.
    SkipGroup(
        reason="Compiler error: int64 slice operations fail with NCC_IBCG902.",
        tests=[
            "test_neuron_compile_ops.py::TestCompileOpInfoNEURON::test_compile_matches_eager_select_neuron_int64",
            "test_neuron_compile_ops.py::TestCompileOpInfoNEURON::test_compile_matches_eager_unbind_neuron_int64",
            "test_neuron_compile_ops.py::TestCompileOpInfoNEURON::test_compile_matches_eager_vsplit_neuron_int64",
        ],
    ),
    # TODO: Remove this XFAIL once the histogramdd graph has a Neuron lowering.
    SkipGroup(
        reason=(
            "histogramdd fails to legalize: 'failed to legalize operation torch.operator'. "
            "This passed before the FX capture kept the Python dispatch key, because the "
            "capture collapsed the computation to a single tensor constant and compared that "
            "against eager rather than compiling the op."
        ),
        tests=[
            "test_neuron_compile_ops.py::TestCompileOpInfoNEURON::test_compile_matches_eager_histogramdd_neuron_float32",
            "test_neuron_compile_ops.py::TestCompileOpInfoNEURON::test_compile_matches_eager_histogramdd_neuron_float64",
        ],
    ),
    # TODO: Remove this XFAIL after the compiler error is fixed.
    SkipGroup(
        reason=(
            "Compiler error: cat backward sample 2 fails with NCC_INLA001 during BIR verification."
        ),
        tests=[
            "test_neuron_compile_ops.py::TestCompileOpInfoGradNEURON::test_compile_fn_grad_cat_neuron_bfloat16",
            "test_neuron_compile_ops.py::TestCompileOpInfoGradNEURON::test_compile_fn_grad_cat_neuron_float32",
            "test_neuron_compile_ops.py::TestCompileOpInfoGradNEURON::test_compile_fn_grad_cat_neuron_float64",
        ],
    ),
    SkipGroup(
        reason=(
            "[accuracy: on-device RNG input divergence] torch.compile path: the on-device RNG"
            " draws different (still-valid) make_tensor inputs than the CPU RNG, so inputs land"
            " on precision-marginal cases -- bf16/fp16 rounding of div/topk/trapezoid/trapz/"
            "layer_norm/nll_loss/poisson_nll, and fp32 floor_divide/remainder near division"
            " poles. A precision floor surfaced by the RNG change, not an op bug. strict=False."
        ),
        tests=[
            "test_neuron_compile_ops.py::TestCompileBinaryUfuncInfoNEURON::test_reference_numerics_normal_div_floor_rounding_neuron_float32",
            "test_neuron_compile_ops.py::TestCompileBinaryUfuncInfoNEURON::test_reference_numerics_normal_div_no_rounding_mode_neuron_bfloat16",
            "test_neuron_compile_ops.py::TestCompileBinaryUfuncInfoNEURON::test_reference_numerics_normal_div_trunc_rounding_neuron_float32",
            "test_neuron_compile_ops.py::TestCompileBinaryUfuncInfoNEURON::test_reference_numerics_normal_floor_divide_neuron_float32",
            "test_neuron_compile_ops.py::TestCompileBinaryUfuncInfoNEURON::test_reference_numerics_normal_remainder_neuron_float32",
            "test_neuron_compile_ops.py::TestCompileBinaryUfuncInfoNEURON::test_reference_numerics_normal_true_divide_neuron_bfloat16",
            "test_neuron_compile_ops.py::TestCompileNNFunctionalOpInfoNEURON::test_compile_nn_functional_nn_functional_layer_norm_neuron_bfloat16",
            "test_neuron_compile_ops.py::TestCompileOpInfoGradNEURON::test_compile_fn_grad_div_no_rounding_mode_neuron_bfloat16",
            "test_neuron_compile_ops.py::TestCompileOpInfoGradNEURON::test_compile_fn_grad_mode_neuron_bfloat16",
            "test_neuron_compile_ops.py::TestCompileOpInfoGradNEURON::test_compile_fn_grad_topk_neuron_bfloat16",
            "test_neuron_compile_ops.py::TestCompileOpInfoGradNEURON::test_compile_fn_grad_true_divide_neuron_bfloat16",
            "test_neuron_compile_ops.py::TestCompileOpInfoNEURON::test_compile_matches_eager_cumulative_trapezoid_neuron_bfloat16",
            "test_neuron_compile_ops.py::TestCompileOpInfoNEURON::test_compile_matches_eager_nn_functional_poisson_nll_loss_neuron_bfloat16",
            "test_neuron_compile_ops.py::TestCompileOpInfoNEURON::test_compile_matches_eager_topk_neuron_bfloat16",
            "test_neuron_compile_ops.py::TestCompileOpInfoNEURON::test_compile_matches_eager_trapezoid_neuron_bfloat16",
            "test_neuron_compile_ops.py::TestCompileOpInfoNEURON::test_compile_matches_eager_trapz_neuron_bfloat16",
            "test_neuron_compile_ops.py::TestCompileNNFunctionalOpInfoNEURON::test_compile_nn_functional_nn_functional_nll_loss_neuron_bfloat16",
        ],
    ),
]
