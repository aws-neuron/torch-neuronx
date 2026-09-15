from tests.pytorch_tests_native.neuron_test_registry import SkipGroup
from tests.pytorch_tests_native.skip_registry import xfail_tests as x

XFAIL_GROUPS: list[SkipGroup] = [
    SkipGroup(
        reason=(
            "SiLU forward lowers to the @AwsNeuronSilu hardware activation"
            " (custom-call), whose bounded fp32 datapath diverges from the"
            " reference formula x/(1+exp(-x)) only at synthetic extremal"
            " inputs: -inf -> 0 (the true limit) instead of nan in all 4"
            " dtypes, and +inf -> FP32_MAX instead of inf for"
            " float32/float64/bfloat16 (float16 preserves +inf)."
            " Real-range values are unaffected -- the"
            " normal/small/large reference-numerics tests pass, and such"
            " extremal inputs only arise from already-diverged (overflow/NaN)"
            " computation."
        ),
        tests=[
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_extremal_nn_functional_silu_neuron_float16",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_extremal_nn_functional_silu_neuron_float32",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_extremal_nn_functional_silu_neuron_float64",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_extremal_nn_functional_silu_neuron_bfloat16",
        ],
    ),
    SkipGroup(
        reason=(
            "[platform.float64] clamp with a float64 bound promotes (via"
            " torch.result_type) to float64, which Neuron does not support."
            " The non-float64 clamp promotion cases are fixed in this PR; these"
            " float64-bound residuals are a platform limitation. xfail (not skip)"
            " so a future float64 story would xpass and flag it."
        ),
        tests=[
            "test_type_promotion.py::TestTypePromotionDeviceNEURON::test_clamp_type_promotion_neuron_bool_float64_float32",
            "test_type_promotion.py::TestTypePromotionDeviceNEURON::test_clamp_type_promotion_neuron_int32_float64_float32",
        ],
    ),
    SkipGroup(
        reason=(
            "[platform.float64] Neuron has no native float64; the dispatch layer"
            " downcasts float64 to float32 (DispatchUtils CastTarget). The float64"
            " forward runs in float32 and still meets the test tolerance, so it"
            " passes. Only the backward gradient misses the float64 reference on"
            " precision. xfail (not skip): the op runs, so a future float64 story"
            " would xpass and flag it."
        ),
        tests=[
            "test_decomp.py::TestDecompNEURON::test_quick_core_backward_logaddexp2_neuron_float64",
        ],
    ),
    SkipGroup(
        reason=(
            "these are @unittest.expectedFailure"
            " upstream but run on Neuron. The expected failure no longer occurs and"
            " unittest reports an Unexpected success. xfailing so the test"
            " still runs and an xpass flags if/when upstream drops the decorator."
        ),
        tests=[
            "test_dataloader.py::TestDataLoader::test_multiprocessing_iterdatapipe_with_dill",
            "test_dataloader.py::TestDataLoaderPersistentWorkers::test_multiprocessing_iterdatapipe_with_dill",
        ],
    ),
    SkipGroup(
        reason=(
            "Neuron intentionally supports a wider dtype set than aten's CPU"
            " anti-aliased kernel for _upsample_bilinear2d_aa. The CPU AA kernel"
            " raises NotImplementedError for bfloat16/float16 (its"
            " compute_index_ranges_weights has no reduced-precision path), so the"
            " OpInfo dtype table omits them; the Neuron decomposition builds the"
            " resample matrix and accumulates the matmul in fp32, so bf16/fp16"
            " work correctly in both forward and backward. test_dtypes flags this"
            " superset as a failure. xfail (not skip) so that an xpass fires if"
            " upstream ever adds bf16/fp16 to the OpInfo table and the two align."
        ),
        tests=[
            "test_ops.py::TestCommonNEURON::test_dtypes__upsample_bilinear2d_aa_neuron",
        ],
    ),
    SkipGroup(
        reason=(
            "Error-expectation test: the test asserts a specific op-native error"
            " message (e.g., 'torch.frexp() only supports floating-point dtypes'),"
            " but Neuron's CPU fallback layer intercepts complex-dtype inputs"
            " with its own message ('CPU fallback disabled...') when"
            " TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1. The op is"
            " implementable on Neuron; the fallback layer should surface the"
            " op's native error for unsupported dtypes. xfail (not skip) so that"
            " a future fallback improvement will xpass and flag the fix."
        ),
        tests=[
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_frexp_assert_raises_neuron",
            "test_binary_ufuncs.py::TestBinaryUfuncs*NEURON::test_float_power_exceptions_neuron",
            "test_nn.py::TestNNDeviceTypeNEURON::test_invalid_reduction_strings_neuron",
            "test_ops.py::TestCommonNEURON::test_errors_masked_fill_neuron",
        ],
    ),
    SkipGroup(
        reason=(
            "PyTorch 2.13 changed avg_pool2d validation error messages from PyTorch "
            "2.12. Neuron requires version-specific validation behavior; temporarily "
            "xfail while PT2.13 semantics are added without regressing PT2.12."
        ),
        tests=[
            "test_nn.py::TestNNDeviceTypeNEURON::test_avg_pool2d_invalid_parameters_neuron",
        ],
    ),
    SkipGroup(
        reason=(
            "Neuron only supports contiguous/preserve memory format"
            " Conv2d with channels_last layout fails through forward/backward."
        ),
        tests=[
            "test_c10d_nccl.py::DistributedDataParallelTest::test_grad_layout_1devicemodule_1replicaperprocess",
            "test_c10d_nccl.py::DistributedDataParallelTest::test_param_layout_mismatch_error",
        ],
    ),
    SkipGroup(
        reason=(
            "torch-neuronx registers fused LSTM for PrivateUse1, displacing"
            " CPU registration — test runs reference model on CPU first."
        ),
        tests=[
            "test_c10d_nccl.py::DistributedDataParallelTest::test_ddp_packed_sequence",
        ],
    ),
    SkipGroup(
        reason=(
            "[accuracy.gradcheck] Numerical vs analytical gradient"
            " mismatch due to float64→float32 precision loss"
        ),
        tests=x.ACCURACY_GRADCHECK,
    ),
    SkipGroup(
        reason=(
            "[accuracy.jacobian_mismatch] Jacobian mismatch due to float64→float32 precision loss"
        ),
        tests=x.ACCURACY_JACOBIAN_MISMATCH,
    ),
    SkipGroup(
        reason="[accuracy.scalar_not_close] Scalar precision difference exceeds tolerance",
        tests=x.ACCURACY_SCALAR_NOT_CLOSE,
    ),
    SkipGroup(
        reason=(
            "[accuracy.scalar_not_equal] Scalar values differ due to reduced precision on Neuron"
        ),
        tests=x.ACCURACY_SCALAR_NOT_EQUAL,
    ),
    SkipGroup(
        reason=(
            "Temporary parent xfail: unary min/max all-NaN backward produces incorrect gradients"
        ),
        tests=[
            "test_autograd.py::TestAutogradDeviceTypeNEURON::test_min_max_aminmax_median_backprops_to_all_values_neuron",
        ],
    ),
    SkipGroup(
        reason=(
            "[accuracy.tensor_not_close] Element-wise difference"
            " exceeds tolerance due to float64→float32 precision loss"
        ),
        tests=x.ACCURACY_TENSOR_NOT_CLOSE,
    ),
    SkipGroup(
        reason=(
            "Extremal inputs feed inf and NaN to celu, hardsigmoid, logit, round,"
            " and round.decimals. Neuron and the CPU _refs golden disagree on inf"
            " and NaN handling. Real-range inputs pass. These boundary cases are"
            " not wrong lowerings."
        ),
        tests=[
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_extremal__refs_nn_functional_hardsigmoid_neuron_bfloat16",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_extremal__refs_nn_functional_hardsigmoid_neuron_float16",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_extremal__refs_nn_functional_hardsigmoid_neuron_float32",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_extremal__refs_round_neuron_bfloat16",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_extremal__refs_round_neuron_float16",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_extremal__refs_round_neuron_float32",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_extremal_logit_neuron_bfloat16",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_extremal_nn_functional_celu_neuron_bfloat16",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_extremal_nn_functional_celu_neuron_float16",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_extremal_nn_functional_celu_neuron_float32",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_extremal_nn_functional_hardsigmoid_neuron_bfloat16",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_extremal_nn_functional_hardsigmoid_neuron_float16",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_extremal_nn_functional_hardsigmoid_neuron_float32",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_extremal_round_decimals_0_neuron_bfloat16",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_extremal_round_decimals_0_neuron_float16",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_extremal_round_decimals_0_neuron_float32",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_extremal_round_decimals_3_neuron_bfloat16",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_extremal_round_decimals_3_neuron_float32",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_extremal_round_decimals_neg_3_neuron_bfloat16",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_extremal_round_decimals_neg_3_neuron_float32",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_extremal_round_neuron_bfloat16",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_extremal_round_neuron_float16",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_extremal_round_neuron_float32",
        ],
    ),
    SkipGroup(
        reason=(
            "Large finite inputs to round and frac. At that magnitude the float"
            " mantissa has no fraction bits, so the result is precision-bound, not"
            " incorrect. The normal and small ranges pass."
        ),
        tests=[
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_large__refs_frac_neuron_bfloat16",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_large__refs_frac_neuron_float32",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_large__refs_round_neuron_bfloat16",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_large__refs_round_neuron_float32",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_large_frac_neuron_bfloat16",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_large_frac_neuron_float32",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_large_round_decimals_0_neuron_bfloat16",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_large_round_decimals_0_neuron_float32",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_large_round_decimals_3_neuron_bfloat16",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_large_round_decimals_3_neuron_float32",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_large_round_decimals_neg_3_neuron_bfloat16",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_large_round_decimals_neg_3_neuron_float32",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_large_round_neuron_bfloat16",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_large_round_neuron_float32",
        ],
    ),
    SkipGroup(
        reason=(
            "Low precision in bfloat16 and float16 for celu, round.decimals, and"
            " the ldexp decomposition cross-reference. Only these low-precision"
            " dtypes exceed the strict tolerance. The float32 normal and small"
            " ranges pass."
        ),
        tests=[
            "test_decomp.py::TestDecompNEURON::test_comprehensive_ldexp_neuron_bfloat16",
            "test_decomp.py::TestDecompNEURON::test_comprehensive_ldexp_neuron_float16",
            "test_decomp.py::TestDecompNEURON::test_quick_ldexp_neuron_bfloat16",
            "test_decomp.py::TestDecompNEURON::test_quick_ldexp_neuron_float16",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_normal_nn_functional_celu_neuron_bfloat16",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_normal_round_decimals_3_neuron_bfloat16",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_small_nn_functional_celu_neuron_bfloat16",
        ],
    ),
    SkipGroup(
        reason=(
            "[accuracy.tensor_not_equal] Tensor values differ due to reduced precision on Neuron"
        ),
        tests=x.ACCURACY_TENSOR_NOT_EQUAL,
    ),
    SkipGroup(
        reason="[assertion_misc.false_not_true] Generic assertion failure on Neuron",
        tests=x.ASSERTION_MISC_FALSE_NOT_TRUE,
    ),
    SkipGroup(
        reason="[assertion_misc.tensor_false] Tensor boolean assertion returns False",
        tests=x.ASSERTION_MISC_TENSOR_FALSE,
    ),
    SkipGroup(
        reason="[compiler_error.neuron] Neuron compiler failed to compile HLO graph",
        tests=x.COMPILER_ERROR_NEURON,
    ),
    SkipGroup(
        reason="[compiler_error.torch_mlir] TorchMLIR op lowering or MLIR conversion error",
        tests=x.COMPILER_ERROR_TORCH_MLIR,
    ),
    SkipGroup(
        reason="[device_mismatch.cpu_vs_neuron] Tensor on wrong device (CPU vs Neuron)",
        tests=x.DEVICE_MISMATCH_CPU_VS_NEURON,
    ),
    SkipGroup(
        reason=(
            "[dtype_mismatch.float64_downcast] float64→float32"
            " downcast due to Neuron 64-bit to 32-bit cast"
        ),
        tests=x.DTYPE_MISMATCH_FLOAT64_DOWNCAST,
    ),
    SkipGroup(
        reason="[dtype_mismatch.object_comparison] Output dtype differs from expected",
        tests=x.DTYPE_MISMATCH_OBJECT_COMPARISON,
    ),
    SkipGroup(
        reason="[error_not_raised.assertion_error] Expected AssertionError not raised on Neuron",
        tests=x.ERROR_NOT_RAISED_ASSERTION_ERROR,
    ),
    SkipGroup(
        reason="[error_not_raised.runtime_error] Expected RuntimeError not raised on Neuron",
        tests=x.ERROR_NOT_RAISED_RUNTIME_ERROR,
    ),
    SkipGroup(
        reason=(
            "Temporary parent xfail: advertised-unsupported dtypes allocate or convert "
            "without raising"
        ),
        # TODO: Replace with per-subtest xfails and reject every dtype absent from
        # torch.accelerator.get_device_capability()["supported_dtypes"].
        tests=[
            "test_accelerator.py::TestAccelerator*::test_device_capability_supported_dtypes*",
        ],
    ),
    SkipGroup(
        reason="[error_not_raised.warning] Expected UserWarning not triggered on Neuron",
        tests=x.ERROR_NOT_RAISED_WARNING,
    ),
    SkipGroup(
        reason="[index_error] Index out of range during op execution on Neuron",
        tests=x.INDEX_ERROR,
    ),
    SkipGroup(
        reason=(
            "[meta_mismatch.output] Meta function output"
            " shape/stride disagrees with real implementation"
        ),
        tests=x.META_MISMATCH_OUTPUT,
    ),
    SkipGroup(
        reason="[opinfo_broken] Upstream OpInfo is broken, not Neuron-specific",
        tests=x.OPINFO_BROKEN,
    ),
    SkipGroup(
        reason="[other] Miscellaneous failures not fitting other categories",
        tests=x.OTHER,
    ),
    SkipGroup(
        reason="[runtime_misc.comparison] RuntimeError during tensor comparison on Neuron",
        tests=x.RUNTIME_MISC_COMPARISON,
    ),
    SkipGroup(
        reason="[runtime_misc.failed_to_run] Op execution returned None on Neuron",
        tests=x.RUNTIME_MISC_FAILED_TO_RUN,
    ),
    SkipGroup(
        reason=(
            "[runtime_misc.module_input_count] NEFF module input"
            " count does not match function arguments"
        ),
        tests=x.RUNTIME_MISC_MODULE_INPUT_COUNT,
    ),
    SkipGroup(
        reason="[runtime_misc.no_refs] No _refs decomposition support for the op",
        tests=x.RUNTIME_MISC_NO_REFS,
    ),
    SkipGroup(
        reason="[runtime_misc.null_storage] Tensor has no backing storage on Neuron",
        tests=x.RUNTIME_MISC_NULL_STORAGE,
    ),
    SkipGroup(
        reason="[runtime_misc.shape_broadcast] Shape or broadcast mismatch during op execution",
        tests=x.RUNTIME_MISC_SHAPE_BROADCAST,
    ),
    SkipGroup(
        reason="[runtime_nrt_error] Neuron Runtime execution error during op execution",
        tests=x.RUNTIME_NRT_ERROR,
    ),
    SkipGroup(
        reason=(
            "[unsupported_op.cpu_fallback_disabled] CPU fallback disabled and op failed on Neuron"
        ),
        tests=x.UNSUPPORTED_OP_CPU_FALLBACK_DISABLED,
    ),
    SkipGroup(
        reason="[unsupported_op.not_implemented] Op not implemented for Neuron backend",
        tests=x.UNSUPPORTED_OP_NOT_IMPLEMENTED,
    ),
    SkipGroup(
        reason=(
            "Neuron permanently registers the sequence-dim Shard(2)->Shard(2)"
            " SDPA strategy (required so the CP ring handler's not-needs-redistribute"
            " assert holds), so SDPA on seq-sharded DTensors without CP still yields"
            " Shard(2) output. Upstream expects seq sharding to be unsupported without"
            " CP (it registers cp sharding rules only transiently), so the test's"
            " assertNotEqual(out.placements[0], Shard(2)) fails."
            "LU without pivoting (pivot=False) is CUDA-only; Neuron's CPU-LAPACK"
            " fallback raises 'LU without pivoting is not implemented on the CPU'"
        ),
        tests=[
            "test_linalg.py::TestLinalgNEURON::test_linalg_lu_family_neuron_float32",
        ],
    ),
    SkipGroup(
        reason=(
            "SDPA sharding strategy arg mismatch:"
            " aten._scaled_dot_product_fused_attention_overrideable.default"
            " passes 3 args but Neuron strategy unpacks ≥4"
        ),
        tests=[
            "test_attention_shard_without_cp",
        ],
    ),
    SkipGroup(
        reason="NLL loss backward shape incompatibility",
        tests=[
            "test_nll_loss_and_cross_entropy",
        ],
    ),
    SkipGroup(
        reason="RNG tracker is None - random ops initialization",
        tests=[
            "test_deterministic_uniform_2d",
        ],
    ),
    SkipGroup(
        reason=(
            "Upstream test bug: TestDifferentiableOptimizer uses beta=1.0 which"
            " is rejected by Adam/AdamW validation (must be < 1.0). This is not"
            " Neuron-specific - test needs to be fixed upstream."
        ),
        tests=[
            "test_optim.py::TestDifferentiableOptimizer::test_adam_differentiable_betas",
            "test_optim.py::TestDifferentiableOptimizer::test_adam_differentiable_all_hyperparams",
            "test_optim.py::TestDifferentiableOptimizer::test_adamw_differentiable_betas",
            "test_optim.py::TestDifferentiableOptimizer::test_adamw_differentiable_all_hyperparams",
        ],
    ),
    SkipGroup(
        reason=(
            "Backward ops (huber_loss, logit, binary_cross_entropy, softplus) are"
            " not decomposed on Neuron - test sees aten.zeros_like.default instead"
            " of the expected decomposed backward. Need to add to CROSS_REF_EXCLUDE_SET"
            " or implement the decomposition."
        ),
        tests=[
            "test_decomp.py::TestDecompNEURON::test_quick_nn_functional_huber_loss_neuron_float32",
            "test_decomp.py::TestDecompNEURON::test_quick_nn_functional_binary_cross_entropy_neuron_float16",
        ],
    ),
    SkipGroup(
        reason=(
            "NRT execution failures on Neuron runtime — compilation succeeds but"
            " runtime rejects the NEFF ('Invalid NEFF, instruction, or input')."
            " Error is reported asynchronously so the listed op may not be the"
            " actual source of the failure."
        ),
        tests=[
            "test_reductions.py::TestReductionsNEURON::test_noncontiguous_transposed_nansum_neuron_int16",
            "test_torch.py::TestTorchDeviceTypeNEURON::test_cdist_large_batch_neuron",
        ],
    ),
    SkipGroup(
        reason=(
            "test_index_copy_scalars iterates 8 combinations of 0-d/1-d"
            " (target, index, source). Neuron's Python-side index_copy handler"
            " in torch_neuronx/python_ops/torch_mlir/ops/indexing.py assigns"
            " expected_shape[actual_dim] without guarding against 0-d target,"
            " raising 'IndexError: list assignment index out of range' instead"
            " of the expected 'Dimension out of range' that CPU/CUDA raise."
            " Affects all dtypes since the bug is in setup, not dispatch."
        ),
        tests=[
            "test_indexing.py::TestIndexingNEURON::test_index_copy_scalars_neuron_bfloat16",
            "test_indexing.py::TestIndexingNEURON::test_index_copy_scalars_neuron_bool",
            "test_indexing.py::TestIndexingNEURON::test_index_copy_scalars_neuron_float16",
            "test_indexing.py::TestIndexingNEURON::test_index_copy_scalars_neuron_float32",
            "test_indexing.py::TestIndexingNEURON::test_index_copy_scalars_neuron_float64",
            "test_indexing.py::TestIndexingNEURON::test_index_copy_scalars_neuron_int16",
            "test_indexing.py::TestIndexingNEURON::test_index_copy_scalars_neuron_int32",
            "test_indexing.py::TestIndexingNEURON::test_index_copy_scalars_neuron_int64",
            "test_indexing.py::TestIndexingNEURON::test_index_copy_scalars_neuron_int8",
            "test_indexing.py::TestIndexingNEURON::test_index_copy_scalars_neuron_uint8",
        ],
    ),
    SkipGroup(
        reason=(
            "Neuron SDPA dispatcher (NeuronDispatcher.cpp) rejects float64"
            " query/key/value with 'SDPA: query dtype must be float32,"
            " float16, or bfloat16, got Double'. Tests create tensors as"
            " float64 (Python default) and hit this guard. Hardware does not"
            " natively support float64 SDPA; if a float64→float32 fallback"
            " is ever added, these xfails will xpass and flag the change."
        ),
        patterns=[
            "test_scaled_dot_product_attention_batch_size_0_*",
        ],
    ),
    SkipGroup(
        reason=(
            "PT 2.11 accuracy regression: 33 bfloat16/float16"
            " reciprocal-family tests (div, reciprocal, rsqrt, fmod,"
            " true_divide, lgamma, floor_divide, remainder,"
            " pairwise_distance) pass on nightly but fail on PT 2.11"
            " with 'tensor-likes are not close' or 'Difference from float64"
            " is larger with decomposition'. Greatest-diff pattern is"
            " 'inf at index N' and decomposed div.Tensor_mode produces ~3e38"
            " where the original is finite — suggests bf16 reciprocal kernel"
            " now saturates to inf where it didn't before."
        ),
        tests=x.PT211_BF16_RECIPROCAL_REGRESSION,
    ),
    SkipGroup(
        reason=(
            "masked_sum on Neuron fp16 returns NaN when masked-out positions"
            " contain Inf, while torch CPU and numpy references correctly"
            " return Inf"
        ),
        tests=[
            "test_reductions.py::TestReductionsNEURON::test_reference_masked_masked_sum_neuron_float16",
        ],
    ),
    SkipGroup(
        reason=(
            "Module1 input address count mismatch: the compiled NEFF expects"
            " a different number of input addresses than the caller provides."
            " Hits the sync-warning path on float32 tensors on PT 2.11. Likely"
            " a torch_neuronx dispatcher / lazy-materialization arg-binding bug."
        ),
        tests=[
            "test_torch.py::TestTorchDeviceTypeNEURON::test_sync_warning_neuron",
        ],
    ),
    SkipGroup(
        reason=(
            "_refs.nn.functional.margin_ranking_loss decomposition returns a"
            " float64 tensor regardless of the input dtype, while the Neuron"
            " eager path correctly preserves the input dtype. This is a bug in"
            " the upstream PrimTorch reference decomposition (casts intermediate"
            " values to float64), not a Neuron backend bug. Deterministic across"
            " all numeric input dtypes."
        ),
        tests=[
            "test_ops.py::TestCommonNEURON::test_python_ref_torch_fallback__refs_nn_functional_margin_ranking_loss_neuron_bfloat16",
            "test_ops.py::TestCommonNEURON::test_python_ref_torch_fallback__refs_nn_functional_margin_ranking_loss_neuron_float16",
            "test_ops.py::TestCommonNEURON::test_python_ref_torch_fallback__refs_nn_functional_margin_ranking_loss_neuron_float32",
            "test_ops.py::TestCommonNEURON::test_python_ref_torch_fallback__refs_nn_functional_margin_ranking_loss_neuron_int8",
            "test_ops.py::TestCommonNEURON::test_python_ref_torch_fallback__refs_nn_functional_margin_ranking_loss_neuron_int32",
            "test_ops.py::TestCommonNEURON::test_python_ref_torch_fallback__refs_nn_functional_margin_ranking_loss_neuron_int64",
            "test_ops.py::TestCommonNEURON::test_python_ref_torch_fallback__refs_nn_functional_margin_ranking_loss_neuron_uint8",
        ],
    ),
    SkipGroup(
        reason=(
            "scipy 1.18.0rc1 regression (PR #24481): kstest maps 'norm' to"
            " scipy.special.ndtr which only accepts (x, out=None), not loc/scale."
            " stats.kstest(data, 'norm', args=(mean, std)) raises TypeError."
            " Not Neuron-specific — upstream PyTorch test + scipy incompatibility."
        ),
        tests=[
            "test_init.py::TestNNInit::test_xavier_normal",
            "test_init.py::TestNNInit::test_kaiming_normal",
            "test_init.py::TestNNInit::test_normal",
            "test_init.py::TestNNInit::test_sparse_default_std",
            "test_torch.py::TestTorchDeviceTypeNEURON::test_normal_kstest_neuron_float16",
            "test_torch.py::TestTorchDeviceTypeNEURON::test_normal_kstest_neuron_float32",
            "test_torch.py::TestTorchDeviceTypeNEURON::test_normal_kstest_neuron_float64",
            "test_torch.py::TestTorchDeviceTypeNEURON::test_normal_kstest_neuron_bfloat16",
            "test_torch.py::TestTorchDeviceTypeNEURON::test_normal_small_std_kstest_neuron_float16",
            "test_torch.py::TestTorchDeviceTypeNEURON::test_normal_small_std_kstest_neuron_bfloat16",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT2.12 only] ExecutionTraceObserver disables itself mid-test"
            " when Neuron's infer_outputs_from_meta() runs torch.gather with"
            " meta tensors while the profiler is active. The C++ observer"
            " (execution_trace_observer.cpp:770) intercepts the meta op and"
            " calls .item() to record tensor_range, which raises"
            " 'Tensor.item() cannot be called on meta tensors'. The observer"
            " catches this and shuts down, so the real gather's tensor_range"
            " is never recorded. This is a PT 2.12 observer bug — PT 2.13"
            " handles meta tensors gracefully. Not a Neuron bug."
        ),
        tests=[
            "profiler/test_execution_trace.py::TestExecutionTraceNEURON::test_execution_trace_record_integral_tensor_range_neuron",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT2.12 only] Automatic dynamic shapes recompilation triggers"
            " neuron backend compilation with unbounded dynamic dims"
            " (f32[?,8]) which neuron-cc rejects. Test uses aot_eager backend"
            " but neuron executor intercepts the .to(device) path. Passes on"
            " PT 2.13 nightly where the routing is handled correctly."
        ),
        tests=[
            "test_autograd.py::TestAutograd::test_checkpoint_automatic_dynamic_graph_shadowing",
        ],
    ),
    SkipGroup(
        reason=(
            "An Aten op (e.g. torch.empty) on a CPU tensor executed inside a TorchDispatchMode"
            " wrapper internally triggers make_variable -> TensorImpl::shallow_copy_and_detach,"
            " only when a prior test runs TorchDispatchMode on the same worker. This re-entry"
            " creates this extra detach op, which is captured and appended to the returned log"
            " list by the LoggingTensor handler. The test then fails the exact match check in"
            " assertExpectedInline. The same test passes under --forked (process isolation)."
            " xfail (not skip) so that a future upstream fix, or an infra change (e.g. --forked)"
            " run, xpasses and flags the change."
        ),
        tests=[
            "test_python_dispatch.py::TestPythonDispatch::test_torch_dispatch_mode_basic",
            "test_python_dispatch.py::TestPythonDispatch::test_nested_push_logging_tensor_mode",
            "test_python_dispatch.py::TestPythonDispatch::test_nesting_same_mode",
            "test_python_dispatch.py::TestPythonDispatch::test_capture_logs_with_torch_dispatch_mode",
        ],
    ),
    SkipGroup(
        reason=(
            "Neuron profiler records bfloat16 norm type promotion as aten::to,"
            " while the test expects promotion to occur internally within the"
            " norm operation, as it does on CUDA."
        ),
        tests=[
            "test_linalg.py::TestLinalgNEURON::test_norm_fused_type_promotion_neuron_bfloat16",
        ],
    ),
    SkipGroup(
        reason=(
            "Neuron eager functionalization does not write BatchNorm running_mean "
            "and running_var mutations back to module buffers. SWA update_bn "
            "therefore leaves stale running statistics and fails CPU parity. "
            "Tracked as an xfail pending mutation-aliasing write-back support."
        ),
        tests=[
            "test_optim.py::TestSWAUtilsNEURON::test_update_bn_cnn_neuron",
            "test_optim.py::TestSWAUtilsNEURON::test_update_bn_dnn_neuron",
        ],
    ),
    SkipGroup(
        reason=(
            "[cpu-only] Reduction-order artifact of OMP_NUM_THREADS=1, which the"
            " conftest sets before importing torch so that --forked children cannot"
            " deadlock on an OpenMP barrier inherited across fork(). The CPU batchnorm"
            " kernel accumulates its reductions according to at::get_num_threads(), so"
            " single-threading changes the summation order and the two paths diverge past"
            " the test's tolerance (2.7e-05 vs 1e-05 allowed on 1 of 8 elements). No Neuron"
            " device is involved. xfail (not skip) so it still runs and xpasses if"
            " the thread pinning is ever narrowed or the upstream tolerance relaxes."
        ),
        tests=[
            "test_nn.py::TestNN::test_batchnorm_nhwc_cpu",
        ],
    ),
    SkipGroup(
        reason=(
            "int64 comparison magnitude limitation. On the native-int64 path,"
            " comparison ops (eq/ne/gt/ge/lt/le) are demoted to the packed int32"
            " view (native S64 compare+transpose hangs the compiler), and the"
            " packed compare uses the low 32 bits only. test_abs_big_number does"
            " abs(2**31+1) (native, correct full-width = 2147483649) then '> 0',"
            " whose low word (0x80000001) is negative as int32 -> False. On main"
            " int64 is downcast at creation (value truncated below 2**31) so '> 0'"
            " is trivially true. xfail (not skip) so it xpasses and flags the fix"
            " once native int64 comparisons (or a sign-correct packed compare) land."
        ),
        tests=[
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_abs_big_number_neuron_int64",
        ],
    ),
    SkipGroup(
        reason=(
            "Neuron sqrt is not IEEE-754 correctly rounded (undershoots exact"
            " squares); sqrt is a mandated correctly-rounded op. tril/triu_indices"
            " floor(sqrt) drops to the wrong integer at row boundaries, and"
            " linalg_vector_norm's pow(.,0.5) (same sqrt kernel) exceeds the fp64"
            " tolerance. Compiler bug P496209905 / NCC-9408; temp xfail until sqrt"
            " is correctly rounded."
        ),
        tests=[
            # tril_indices -- floor(sqrt) off-by-one at exact-square boundaries
            "test_decomp.py::TestDecompNEURON::test_quick_tril_indices_neuron_int32",
            "test_decomp.py::TestDecompNEURON::test_quick_tril_indices_neuron_int64",
            "test_ops.py::TestCommonNEURON::test_python_ref__refs_tril_indices_neuron_int32",
            "test_ops.py::TestCommonNEURON::test_python_ref__refs_tril_indices_neuron_int64",
            "test_ops.py::TestCommonNEURON::test_python_ref_torch_fallback__refs_tril_indices_neuron_int32",
            "test_ops.py::TestCommonNEURON::test_python_ref_torch_fallback__refs_tril_indices_neuron_int64",
            # triu_indices -- same decomposition (relocated from accuracy.tensor_not_equal)
            "test_decomp.py::TestDecompNEURON::test_comprehensive_triu_indices_neuron_int32",
            "test_decomp.py::TestDecompNEURON::test_comprehensive_triu_indices_neuron_int64",
            "test_decomp.py::TestDecompNEURON::test_quick_triu_indices_neuron_int32",
            "test_decomp.py::TestDecompNEURON::test_quick_triu_indices_neuron_int64",
            "test_ops.py::TestCommonNEURON::test_python_ref__refs_triu_indices_neuron_int32",
            "test_ops.py::TestCommonNEURON::test_python_ref__refs_triu_indices_neuron_int64",
            "test_ops.py::TestCommonNEURON::test_python_ref_torch_fallback__refs_triu_indices_neuron_int32",
            "test_ops.py::TestCommonNEURON::test_python_ref_torch_fallback__refs_triu_indices_neuron_int64",
            # linalg_vector_norm -- pow(.,0.5) == the same sqrt kernel, vs fp64 tol
            "test_decomp.py::TestDecompNEURON::test_quick_linalg_vector_norm_neuron_float64",
        ],
    ),
]
