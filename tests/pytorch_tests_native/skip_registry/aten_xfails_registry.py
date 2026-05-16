from tests.pytorch_tests_native.neuron_test_registry import SkipGroup
from tests.pytorch_tests_native.skip_registry import xfail_tests as x

XFAIL_GROUPS: list[SkipGroup] = [
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
            "test_binary_ufuncs.py::TestBinaryUfuncsNEURON::test_float_power_exceptions_neuron",
            "test_nn.py::TestNNDeviceTypeNEURON::test_invalid_reduction_strings_neuron",
            "test_ops.py::TestCommonNEURON::test_errors_masked_fill_neuron",
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
            "[accuracy.jacobian_mismatch] Jacobian mismatch"
            " due to float64→float32 precision loss"
        ),
        tests=x.ACCURACY_JACOBIAN_MISMATCH,
    ),
    SkipGroup(
        reason="[accuracy.scalar_not_close] Scalar precision difference exceeds tolerance",
        tests=x.ACCURACY_SCALAR_NOT_CLOSE,
    ),
    SkipGroup(
        reason=(
            "[accuracy.scalar_not_equal] Scalar values differ" " due to reduced precision on Neuron"
        ),
        tests=x.ACCURACY_SCALAR_NOT_EQUAL,
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
            "[accuracy.tensor_not_equal] Tensor values differ" " due to reduced precision on Neuron"
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
        reason="[error_not_raised.runtime_error] Expected RuntimeError not raised on Neuron",
        tests=x.ERROR_NOT_RAISED_RUNTIME_ERROR,
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
            "[unsupported_op.cpu_fallback_disabled]"
            " CPU fallback disabled and op failed on Neuron"
        ),
        tests=x.UNSUPPORTED_OP_CPU_FALLBACK_DISABLED,
    ),
    SkipGroup(
        reason="[unsupported_op.not_implemented] Op not implemented for Neuron backend",
        tests=x.UNSUPPORTED_OP_NOT_IMPLEMENTED,
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
            "index_put with None in the indices "
            "list generates a torch.constant.bool op in the IR, and neuron's "
            "torch-mlir → StableHLO pass has no legalization pattern for it, "
            "so compilation aborts."
        ),
        tests=[
            "test_tensor_ops.py::DistTensorOpsTest::test_index_put_noncontiguous_indexed_dims",
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
            "test_decomp.py::TestDecompNEURON::test_quick_logit_neuron_bfloat16",
            "test_decomp.py::TestDecompNEURON::test_quick_nn_functional_binary_cross_entropy_neuron_float16",
            "test_decomp.py::TestDecompNEURON::test_quick_nn_functional_hardsigmoid_neuron_float32",
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
            "Neuron's Event.query() returns False after the associated"
            " stream work has completed. Violates the Accelerator API"
            " contract that Event.query() should return True once the"
            " recorded work has finished. Bug in torch_neuronx stream/event"
            " signaling."
        ),
        tests=[
            "test_accelerator.py::TestAccelerator::test_generic_stream_behavior",
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
            "NRT Execution error on aten::ne.Tensor_out when prod reduction hits"
            " the high-dim-count limit (>ndim). Likely a Neuron kernel bug in"
            " the out-variant of ne for high-rank tensors."
        ),
        tests=[
            "test_reductions.py::TestReductionsNEURON::test_dim_ndim_limit_prod_neuron",
        ],
    ),
    SkipGroup(
        reason=(
            "Test expects _foreach_addcmul with a bool tensor to raise"
            " RuntimeError, but Neuron's fastpath inplace variant silently"
            " succeeds. Indicates either a missing dtype check in the Neuron"
            " foreach_addcmul kernel or a mismatch between the upstream OpInfo"
            " expectation and the Neuron implementation."
        ),
        tests=[
            "test_foreach.py::TestForeachNEURON::test_parity__foreach_addcmul_fastpath_inplace_neuron_bool",
        ],
    ),
    SkipGroup(
        reason=(
            "[prologue] Data-dependent NaN in standalone transpose NEFF for"
            " non-contiguous fp16 tensors of shape (357, 789). The _refs"
            " decomposition (promote->op->demote) triggers CopyToNonContiguous"
            " which uses an on-device transpose that produces spurious NaN at"
            " row 128 for specific data patterns. Affects any _refs op that"
            " does promote/demote on this shape. Native ops with epilogue work"
            " correctly."
        ),
        tests=[
            "test_ops.py::TestCommonNEURON::test_python_ref__refs_div_no_rounding_mode_neuron_float16",
            "test_ops.py::TestCommonNEURON::test_python_ref__refs_pow_neuron_float16",
        ],
    ),
]
