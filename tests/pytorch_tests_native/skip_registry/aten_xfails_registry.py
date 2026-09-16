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
        tests=x.EXTREMAL_REF_NUMERICS,
    ),
    SkipGroup(
        reason=(
            "Large finite inputs to round and frac. At that magnitude the float"
            " mantissa has no fraction bits, so the result is precision-bound, not"
            " incorrect. The normal and small ranges pass."
        ),
        tests=x.LARGE_ROUND_FRAC,
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
        reason=(
            "[PT 2.14][runtime_misc.foreach_norm_integer] foreach_norm on integer/bool"
            " inputs calls linalg.vector_norm which requires floating-point."
            " CPU casts to float first; Neuron's foreach dispatch does not."
        ),
        tests=x.FOREACH_NORM_INTEGER,
    ),
    SkipGroup(
        reason=(
            "[PT 2.14][runtime_misc.foreach_copy_mixed_dtype] foreach_copy with mixed"
            " dtypes within a tensor list produces incorrect results on Neuron."
        ),
        tests=[
            "test_foreach.py::TestForeachNEURON::test_foreach_copy_with_mixed_dtypes_within_tensor__foreach_copy_neuron_bfloat16",
            "test_foreach.py::TestForeachNEURON::test_foreach_copy_with_mixed_dtypes_within_tensor__foreach_copy_neuron_float16",
            "test_foreach.py::TestForeachNEURON::test_foreach_copy_with_mixed_dtypes_within_tensor__foreach_copy_neuron_float32",
        ],
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
            "test_scaled_dot_product_attention_batch_size_*",
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
            "Neuron dropout does not propagate NaN through dropped positions."
            " native_dropout computes 'input * mask * scale', but the compiler"
            " folds the multiply by a 0/1 mask into a select, so a dropped NaN"
            " becomes 0.0 while CPU keeps NaN. This test builds a random bool"
            " attn_mask, and when a row is fully masked the softmax reference is"
            " all-NaN. Dropout then zeroes part of that row, so its"
            " 'expected_all_nan' assertion fails depending on the RNG draw."
            " Only the dropout_p>0 rows are affected; the dropout_p=0 rows pass."
        ),
        tests=[
            "test_transformers.py::TestTransformersNEURON::test_scaled_dot_product_attention_batch_size_5_4D_input_dim_4D_attn_mask_dropout_p_0_5_neuron",
        ],
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
            "hardtanh (and relu6, which decomposes to hardtanh(0,6)) lowers to the"
            " Neuron clamp. On the extremal OpInfo reference sample the torch._refs"
            " decomposition runs on the Neuron clamp, which does not propagate a"
            " NaN input (it returns a clamp bound). test_python_ref then measures"
            " the _refs reference as farther (nan) from the float64-precise result"
            " than the eager op (distance 0.0). The eager lowering is exact on real"
            " ranges (unit tests and vertical coverage pass). Do not mask the NaN"
            " with torch.where."
        ),
        tests=[
            "test_ops.py::TestCommonNEURON::test_python_ref__refs_nn_functional_hardtanh_neuron_bfloat16",
            "test_ops.py::TestCommonNEURON::test_python_ref__refs_nn_functional_hardtanh_neuron_float16",
            "test_ops.py::TestCommonNEURON::test_python_ref__refs_nn_functional_hardtanh_neuron_float32",
            "test_ops.py::TestCommonNEURON::test_python_ref__refs_nn_functional_hardtanh_neuron_float64",
            "test_ops.py::TestCommonNEURON::test_python_ref__refs_nn_functional_relu6_neuron_bfloat16",
            "test_ops.py::TestCommonNEURON::test_python_ref__refs_nn_functional_relu6_neuron_float16",
            "test_ops.py::TestCommonNEURON::test_python_ref__refs_nn_functional_relu6_neuron_float32",
            "test_ops.py::TestCommonNEURON::test_python_ref__refs_nn_functional_relu6_neuron_float64",
        ],
    ),
    SkipGroup(
        reason=(
            "hardshrink lowers to a Neuron compare. On the extremal OpInfo"
            " reference sample the torch._refs decomposition runs on the Neuron"
            " compare, which treats a NaN input as inside the shrink band and zeros"
            " it, while aten keeps the NaN. test_python_ref then measures the _refs"
            " reference as farther (nan) from the float64-precise result than the"
            " eager op (distance 0.0). The eager lowering is exact on real ranges"
            " (unit tests and vertical coverage pass). Do not mask the NaN with"
            " torch.where."
        ),
        tests=[
            "test_ops.py::TestCommonNEURON::test_python_ref__refs_nn_functional_hardshrink_neuron_bfloat16",
            "test_ops.py::TestCommonNEURON::test_python_ref__refs_nn_functional_hardshrink_neuron_float16",
            "test_ops.py::TestCommonNEURON::test_python_ref__refs_nn_functional_hardshrink_neuron_float32",
            "test_ops.py::TestCommonNEURON::test_python_ref__refs_nn_functional_hardshrink_neuron_float64",
            "test_ops.py::TestCommonNEURON::test_python_ref_torch_fallback__refs_nn_functional_hardshrink_neuron_bfloat16",
            "test_ops.py::TestCommonNEURON::test_python_ref_torch_fallback__refs_nn_functional_hardshrink_neuron_float16",
            "test_ops.py::TestCommonNEURON::test_python_ref_torch_fallback__refs_nn_functional_hardshrink_neuron_float32",
            "test_ops.py::TestCommonNEURON::test_python_ref_torch_fallback__refs_nn_functional_hardshrink_neuron_float64",
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
        tests=x.SQRT_NOT_IEEE754,
    ),
    SkipGroup(
        reason=(
            "[PT 2.14][platform.exception_type] Neuron raises RuntimeError for"
            " .imag on non-complex tensors; upstream 2.14+ changed the"
            " expected exception to TypeError. xfail so a dispatch fix"
            " would xpass and flag it."
        ),
        tests=[
            "test_autograd.py::TestAutograd::test_dir",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT 2.14][accuracy.decomp_reference] New test_python_ref instantiations in"
            " PT 2.14. The test computes the op in float64 as a reference and"
            " checks that the device result is closer to it than the PyTorch"
            " CPU decomposition result. Neuron's reduced-precision hardware"
            " (float16/bfloat16/float32) produces results farther from the"
            " float64 reference than CPU. Platform precision limitation."
        ),
        tests=x.DECOMP_REFERENCE_PRECISION,
    ),
    SkipGroup(
        reason=(
            "[PT 2.14][refs_executor.type_cast] _refs.linspace integer-output path calls"
            " _refs.fill which rejects float->int cast that ATen allows."
            " Only hits the refs executor tracing path, not direct op calls."
        ),
        tests=[
            "test_ops.py::TestCommonNEURON::test_python_ref_executor__refs_linspace_executor_aten_neuron_int16",
            "test_ops.py::TestCommonNEURON::test_python_ref_executor__refs_linspace_executor_aten_neuron_int32",
            "test_ops.py::TestCommonNEURON::test_python_ref_executor__refs_linspace_executor_aten_neuron_int64",
            "test_ops.py::TestCommonNEURON::test_python_ref_executor__refs_linspace_executor_aten_neuron_int8",
            "test_ops.py::TestCommonNEURON::test_python_ref_executor__refs_linspace_executor_aten_neuron_uint8",
            "test_ops.py::TestCommonNEURON::test_python_ref_executor__refs_linspace_tensor_overload_executor_aten_neuron_int16",
            "test_ops.py::TestCommonNEURON::test_python_ref_executor__refs_linspace_tensor_overload_executor_aten_neuron_int32",
            "test_ops.py::TestCommonNEURON::test_python_ref_executor__refs_linspace_tensor_overload_executor_aten_neuron_int64",
            "test_ops.py::TestCommonNEURON::test_python_ref_executor__refs_linspace_tensor_overload_executor_aten_neuron_int8",
            "test_ops.py::TestCommonNEURON::test_python_ref_executor__refs_linspace_tensor_overload_executor_aten_neuron_uint8",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT 2.14][platform.output_stride] FakeTensor crossref backward stride mismatch."
            " PT 2.14 widened these tests from @onlyCUDA to @onlyAccelerator."
            " Neuron's XLA compilation makes inputs contiguous before lowering,"
            " so outputs are always row-major. ATen contract requires elementwise"
            " ops to preserve input strides (e.g. column-major in, column-major"
            " out). The correct fix is to fuse the output transpose into the NEFF"
            " via the epilogue, avoiding a post-hoc device copy. xfail (not skip)"
            " so the epilogue fix gives xpass."
        ),
        tests=x.OUTPUT_STRIDE_MISMATCH,
    ),
    SkipGroup(
        reason=(
            "[PT 2.14][runtime_misc.nki_rng_meta] NKI set_rng_state kernel cannot run"
            " in meta crossref dispatch path. PT 2.14 widened these tests"
            " from @onlyCUDA to @onlyAccelerator."
        ),
        tests=[
            "test_meta.py::TestMetaNEURON::test_dispatch_meta_outplace_randint_like_neuron_bool",
            "test_meta.py::TestMetaNEURON::test_dispatch_meta_outplace_randint_neuron_bool",
            "test_meta.py::TestMetaNEURON::test_dispatch_symbolic_meta_outplace_randint_like_neuron_bool",
            "test_meta.py::TestMetaNEURON::test_dispatch_symbolic_meta_outplace_randint_neuron_bool",
            "test_meta.py::TestMetaNEURON::test_meta_outplace_randint_like_neuron_bool",
            "test_meta.py::TestMetaNEURON::test_meta_outplace_randint_neuron_bool",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT 2.14][unsupported_op.lbfgs] LBFGS optimizer hits IndexError at"
            " lbfgs.py:199 on Neuron. PT 2.14 new test."
        ),
        tests=[
            "test_optim.py::TestOptimRenewedNEURON::test_tensor_lr_num_dim_0_LBFGS_neuron_float32",
            "test_optim.py::TestOptimRenewedNEURON::test_tensor_lr_num_dim_1_LBFGS_neuron_float32",
            "test_optim.py::TestOptimRenewedNEURON::test_tensor_lr_num_dim_2_LBFGS_neuron_float32",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT 2.14][accuracy.tensor_not_close] PT 2.14 cpu_gpu_parity tests:"
            " CrossEntropyLoss fp16 tolerance and fp32 grad None mismatch."
        ),
        tests=[
            "test_modules.py::TestModuleNEURON::test_cpu_gpu_parity_nn_CrossEntropyLoss_neuron_float16",
            "test_modules.py::TestModuleNEURON::test_cpu_gpu_parity_nn_CrossEntropyLoss_neuron_float32",
            "test_optim.py::TestOptimRenewedNEURON::test_state_dict_cross_device_Muon_neuron_float32",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT 2.14][runtime_misc.serialization] PT 2.14 serialization tests:"
            " Unsupported ScalarType 8 (float8) and dtype view restrictions."
        ),
        tests=[
            "test_serialization.py::TestSerializationDeviceTypeNEURON::test_save_different_dtype_unallocated_neuron",
            "test_serialization.py::TestSerializationDeviceTypeNEURON::test_skip_data_serialization_materialize_fake_False_neuron",
            "test_serialization.py::TestSerializationDeviceTypeNEURON::test_skip_data_serialization_materialize_fake_True_neuron",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT 2.14] _refs.special.logit integer/bool: decomposition path"
            " produces different results than aten on Neuron for non-float"
            " dtypes. Direct torch.special.logit works correctly."
        ),
        tests=x.LOGIT_INTEGER_BOOL,
    ),
    SkipGroup(
        reason=(
            "[PT 2.14] refs executor: arange negative step boundary produces"
            " shape [2] vs [3] on CI. Not reproduced locally."
        ),
        tests=[
            "test_ops.py::TestCommonNEURON::test_python_ref_executor__refs_arange_executor_aten_neuron_bfloat16",
            "test_ops.py::TestCommonNEURON::test_python_ref_executor__refs_arange_executor_aten_neuron_float16",
            "test_ops.py::TestCommonNEURON::test_python_ref_executor__refs_arange_executor_aten_neuron_float32",
            "test_ops.py::TestCommonNEURON::test_python_ref_executor__refs_arange_executor_aten_neuron_int16",
            "test_ops.py::TestCommonNEURON::test_python_ref_executor__refs_arange_executor_aten_neuron_int32",
            "test_ops.py::TestCommonNEURON::test_python_ref_executor__refs_arange_executor_aten_neuron_int64",
            "test_ops.py::TestCommonNEURON::test_python_ref_executor__refs_arange_executor_aten_neuron_int8",
            "test_ops.py::TestCommonNEURON::test_python_ref_executor__refs_arange_executor_aten_neuron_uint8",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT 2.14] test_compare_cpu numerical mismatches on new"
            " @onlyAccelerator tests. Not reproduced locally."
        ),
        tests=x.COMPARE_CPU_NUMERICAL,
    ),
    SkipGroup(
        reason=(
            "[PT 2.14] refs executor: integer type mismatches in"
            " addcmul, floor_divide, triu/tril_indices, constant_pad_nd,"
            " rsqrt bool, alpha_dropout dtype, nll_loss tracing."
        ),
        tests=x.REFS_EXECUTOR_INTEGER,
    ),
    SkipGroup(
        reason=(
            "[PT 2.14] test_decomp logit decomposition: same root cause as"
            " test_ops logit — decomposition produces wrong results for"
            " integer/bool/float dtypes. Direct op works correctly."
        ),
        tests=x.DECOMP_LOGIT,
    ),
    SkipGroup(
        reason=(
            "[PT 2.14] neuron_compile_ops: logit decomposition + matrix_sqrth"
            " NEFF execution + max_reduction int64 overflow + abs int64."
        ),
        tests=[
            "neuron_compile_ops/test_neuron_compile_ops.py::TestCompileLinalgOpInfoNEURON::test_compile_linalg_linalg_matrix_sqrth_neuron_float32",
            "neuron_compile_ops/test_neuron_compile_ops.py::TestCompileOpInfoGradNEURON::test_compile_fn_grad_linalg_matrix_sqrth_neuron_float32",
            "neuron_compile_ops/test_neuron_compile_ops.py::TestCompileOpInfoNEURON::test_compile_matches_eager_abs_neuron_int64",
            "neuron_compile_ops/test_neuron_compile_ops.py::TestCompileOpInfoNEURON::test_compile_matches_eager_linalg_matrix_sqrth_neuron_float32",
            "neuron_compile_ops/test_neuron_compile_ops.py::TestCompileOpInfoNEURON::test_compile_matches_eager_logit_neuron_bfloat16",
            "neuron_compile_ops/test_neuron_compile_ops.py::TestCompileOpInfoNEURON::test_compile_matches_eager_logit_neuron_float32",
            "neuron_compile_ops/test_neuron_compile_ops.py::TestCompileOpInfoNEURON::test_compile_matches_eager_logit_neuron_float64",
            "neuron_compile_ops/test_neuron_compile_ops.py::TestCompileOpInfoNEURON::test_compile_matches_eager_logit_neuron_int64",
            "neuron_compile_ops/test_neuron_compile_ops.py::TestCompileOpInfoNEURON::test_compile_matches_eager_max_reduction_no_dim_neuron_int64",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT 2.14] test_modules cpu_gpu_parity: LPPool 88-100% mismatch,"
            " Conv1d backward compiler error, ReplicationPad3d numerical."
            " Not reproduced locally with basic inputs."
        ),
        tests=[
            "test_modules.py::TestModuleNEURON::test_cpu_gpu_parity_nn_Conv1d_neuron_float32",
            "test_modules.py::TestModuleNEURON::test_cpu_gpu_parity_nn_LPPool1d_neuron_float32",
            "test_modules.py::TestModuleNEURON::test_cpu_gpu_parity_nn_LPPool2d_neuron_float32",
            "test_modules.py::TestModuleNEURON::test_cpu_gpu_parity_nn_LPPool3d_neuron_float32",
            "test_modules.py::TestModuleNEURON::test_cpu_gpu_parity_nn_LazyConv1d_neuron_float32",
            "test_modules.py::TestModuleNEURON::test_cpu_gpu_parity_nn_ReplicationPad3d_neuron_float32",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT 2.14] test_reductions: NRT errors on large inputs,"
            " sum device mismatch error text, sum_noncontig timeout."
            " min_max_nan: stablehlo.maximum/minimum use quiet NaN"
            " semantics (NaN loses to any real number) instead of IEEE"
            " 754 propagating NaN. Fix requires neuronx-cc compiler"
            " change to emit NaN-propagating reduce; a Python-level"
            " workaround would add isnan+where overhead to every"
            " min/max call. xfail pending compiler fix."
        ),
        tests=[
            "test_reductions.py::TestReductionsNEURON::test_argminmax_large_axis_neuron",
            "test_reductions.py::TestReductionsNEURON::test_min_max_nan_neuron",
            "test_reductions.py::TestReductionsNEURON::test_quantile_large_input_neuron_float32",
            "test_reductions.py::TestReductionsNEURON::test_sum_cpu_device_mismatch_neuron",
            "test_reductions.py::TestReductionsNEURON::test_sum_noncontig_lowp_neuron_bfloat16",
            "test_reductions.py::TestReductionsNEURON::test_sum_noncontig_lowp_neuron_float16",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT 2.14] test_ops_gradients: float64 Jacobian mismatches"
            " for nn_functional_linear and matrix_sqrth."
        ),
        tests=[
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_nn_functional_linear_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_linalg_matrix_sqrth_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_nn_functional_linear_neuron_float64",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT 2.14] misc remaining: foreach_l2_norm precision,"
            " foreach_neg bool different-devices, bmm COW inputs,"
            " checkpoint, profiler, type_promotion, native_mha,"
            " FSDP prefetch_writeback, distributed checkpoint."
        ),
        tests=x.PT214_MISC_REMAINING,
    ),
    SkipGroup(
        reason=(
            "[PT 2.14][signed_zero.relu] Neuron loses the -0.0 sign bit through relu."
            " relu(-0.0) should return -0.0 per IEEE 754 but Neuron returns"
            " +0.0. Signed-zero handling difference on Neuron hardware."
        ),
        tests=[
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_relu_signed_zero_neuron_bfloat16",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_relu_signed_zero_neuron_float16",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_relu_signed_zero_neuron_float32",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT 2.14][signed_zero.clamp] Neuron loses the -0.0 sign bit through clamp."
            " clamp(-0.0, min=0) should return -0.0 per IEEE 754 but Neuron"
            " returns +0.0. Signed-zero handling difference on Neuron hardware."
        ),
        tests=[
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_clamp_signed_zero_neuron_bfloat16",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_clamp_signed_zero_neuron_float16",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_clamp_signed_zero_neuron_float32",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT 2.14] scatter_add_large triggers NRT allocator crash on"
            " large tensor inputs. The test creates tensors exceeding"
            " the default allocation threshold."
        ),
        tests=[
            "test_scatter_gather_ops.py::TestScatterGatherNEURON::test_scatter_add_large_neuron_bfloat16",
            "test_scatter_gather_ops.py::TestScatterGatherNEURON::test_scatter_add_large_neuron_float16",
            "test_scatter_gather_ops.py::TestScatterGatherNEURON::test_scatter_add_large_neuron_float32",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT 2.14][foreach_norm] foreach_reduce_large_input norm tests fail on"
            " bf16/fp16 due to reduced-precision accumulation on Neuron."
        ),
        tests=[
            "test_foreach.py::TestForeachNEURON::test_foreach_reduce_large_input__foreach_norm_w_empty_False_neuron_bfloat16",
            "test_foreach.py::TestForeachNEURON::test_foreach_reduce_large_input__foreach_norm_w_empty_False_neuron_float16",
            "test_foreach.py::TestForeachNEURON::test_foreach_reduce_large_input__foreach_norm_w_empty_True_neuron_bfloat16",
            "test_foreach.py::TestForeachNEURON::test_foreach_reduce_large_input__foreach_norm_w_empty_True_neuron_float16",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT 2.14][CTCLoss] test_CTCLoss_long_targets triggers precision loss"
            " on Neuron due to reduced-precision log-sum-exp accumulation"
            " over long target sequences."
        ),
        tests=[
            "test_nn.py::TestNNDeviceTypeNEURON::test_CTCLoss_long_targets_neuron",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT 2.14][foreach_different_devices] Test creates interleaved"
            " [device_t, cpu_t] tensor lists. foreach raises for mixed"
            " devices, then asserts element-wise native_op also raises."
            " On Neuron, element-wise pairs are same-device (device+device,"
            " cpu+cpu) so no error is raised. Platform difference, not a bug."
        ),
        tests=x.FOREACH_DIFFERENT_DEVICES,
    ),
    SkipGroup(
        reason=(
            "[PT 2.14][baddbmm_out] PT 2.14 added dtype guard for out= argument."
            " Test expects RuntimeError for float32->long unsafe cast"
            " but Neuron dispatch does not enforce the same out-dtype check."
        ),
        tests=[
            "test_ops.py::TestCommonNEURON::test_out_baddbmm_neuron_float32",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT 2.14][resize_overflow] PT 2.14 resize overflow check not raised"
            " on Neuron — Neuron's resize_ implementation does not validate"
            " the product of dimensions for overflow."
        ),
        tests=[
            "test_view_ops.py::TestOldViewOpsNEURON::test_resize_overflow_neuron",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT 2.14][foreach_sub_autodiff] foreach_sub outplace float64 — test"
            " expects RuntimeError for autodiff on float64 but Neuron"
            " does not raise because the op falls back to CPU for float64."
        ),
        tests=[
            "test_foreach.py::TestForeachNEURON::test_autodiff__foreach_sub_outplace_neuron_float64",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT 2.14][foreach_different_devices_misc] foreach_sub bool and"
            " foreach pointwise different-devices — same mixed-device"
            " pattern as binary_op_tensors_on_different_devices."
        ),
        tests=[
            "test_foreach.py::TestForeachNEURON::test_binary_op_tensors_on_different_devices__foreach_sub_neuron_bool",
            "test_foreach.py::TestForeachNEURON::test_pointwise_op_tensors_on_different_devices__foreach_addcdiv_neuron_float32",
            "test_foreach.py::TestForeachNEURON::test_pointwise_op_tensors_on_different_devices__foreach_addcmul_neuron_float32",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT 2.14][isin_different_devices] Test expects RuntimeError for"
            " mixed-device isin but Neuron does not raise because the"
            " element-wise comparison pairs are same-device."
        ),
        tests=[
            "test_sort_and_select.py::TestSortAndSelectDeviceNEURON::test_isin_different_devices_neuron_float32",
            "test_sort_and_select.py::TestSortAndSelectDeviceNEURON::test_isin_different_devices_neuron_int16",
            "test_sort_and_select.py::TestSortAndSelectDeviceNEURON::test_isin_different_devices_neuron_int32",
            "test_sort_and_select.py::TestSortAndSelectDeviceNEURON::test_isin_different_devices_neuron_int64",
            "test_sort_and_select.py::TestSortAndSelectDeviceNEURON::test_isin_different_devices_neuron_int8",
            "test_sort_and_select.py::TestSortAndSelectDeviceNEURON::test_isin_different_devices_neuron_uint8",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT 2.14][error_message_mismatch] Neuron raises a valid error but"
            " with different wording than the regex the test expects."
        ),
        tests=[
            "test_indexing.py::TestIndexingNEURON::test_gather_take_along_dim_cross_device_neuron_float32",
            "test_indexing.py::TestIndexingNEURON::test_index_put_deterministic_with_optional_tensors_neuron",
            "test_nn.py::TestNNDeviceTypeNEURON::test_cosine_embedding_loss_error_on_nonexpandable_shapes_neuron",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT 2.14][unsupported_dtype] eye float8_e4m3fnuz/e5m2fnuz"
            " — ScalarType 25/26 not supported on Neuron."
        ),
        tests=[
            "test_ops.py::TestCommonNEURON::test_python_ref_executor__refs_eye_executor_aten_neuron_float8_e4m3fnuz",
            "test_ops.py::TestCommonNEURON::test_python_ref_executor__refs_eye_executor_aten_neuron_float8_e5m2fnuz",
        ],
    ),
    SkipGroup(
        reason="[PT 2.14][unsupported_dtype] uint64 ScalarType not supported on Neuron.",
        tests=[
            "test_indexing.py::TestIndexingNEURON::test_setitem_uint64_above_int64_max_neuron",
            "test_torch.py::TestTorchDeviceTypeNEURON::test_fill_barebones_unsigned_neuron_uint64",
        ],
    ),
    SkipGroup(
        reason="[PT 2.14][not_implemented] Fused GRU cell not implemented on Neuron backend.",
        tests=[
            "test_nn.py::TestNNDeviceTypeNEURON::test_RNN_input_size_zero_neuron",
        ],
    ),
    SkipGroup(
        reason="[PT 2.14][not_implemented] avg_pool3d not implemented for BFloat16 on Neuron.",
        tests=[
            "nn/test_pooling.py::TestPoolingNNDeviceTypeNEURON::test_pooling_bfloat16_neuron",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT 2.14][not_implemented] Unbatched dilated conv not supported"
            " — Neuron expects batched input."
        ),
        tests=[
            "nn/test_convolution.py::TestConvolutionNNDeviceTypeNEURON::test_slow_conv_dilated2d_unbatched_neuron",
            "nn/test_convolution.py::TestConvolutionNNDeviceTypeNEURON::test_slow_conv_dilated3d_unbatched_neuron",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT 2.14][not_implemented] geometric check_uniform_bounds"
            " not implemented for integer types on Neuron."
        ),
        tests=[
            "test_ops.py::TestCommonNEURON::test_python_ref_executor__refs_geometric_executor_aten_neuron_int16",
            "test_ops.py::TestCommonNEURON::test_python_ref_executor__refs_geometric_executor_aten_neuron_int32",
            "test_ops.py::TestCommonNEURON::test_python_ref_executor__refs_geometric_executor_aten_neuron_int64",
            "test_ops.py::TestCommonNEURON::test_python_ref_executor__refs_geometric_executor_aten_neuron_int8",
            "test_ops.py::TestCommonNEURON::test_python_ref_executor__refs_geometric_executor_aten_neuron_uint8",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT 2.14][unsupported_dtype] gather uint16/32/64 — ScalarType"
            " not supported in Torch-MLIR lowering."
        ),
        tests=[
            "test_scatter_gather_ops.py::TestScatterGatherNEURON::test_gather_neuron_uint16",
            "test_scatter_gather_ops.py::TestScatterGatherNEURON::test_gather_neuron_uint32",
            "test_scatter_gather_ops.py::TestScatterGatherNEURON::test_gather_neuron_uint64",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT 2.14][foreach_0dim] Test expects RuntimeError for CPU scalar"
            " tensor passed to foreach op with device tensors. Neuron"
            " does not raise the same scalar-device validation error."
        ),
        tests=[
            "test_foreach.py::TestForeachNEURON::test_0dim_tensor_overload_exception_neuron",
        ],
    ),
]
