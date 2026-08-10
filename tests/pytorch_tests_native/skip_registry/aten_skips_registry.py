from tests.pytorch_tests_native.neuron_test_registry import SkipGroup
from tests.pytorch_tests_native.skip_registry import skipped_tests as s
from tests.pytorch_tests_native.skip_registry import xfail_tests as x

SKIP_GROUPS: list[SkipGroup] = [
    SkipGroup(
        reason=(
            "Neuron does the index bounds check on-device (async), like CUDA/XPU,"
            " so an out-of-bounds fancy index does not raise a synchronous"
            " IndexError at the op call. The upstream test already skips CUDA/XPU"
            " for the same reason ('CUDA/XPU asserts instead of raising an"
            " exception'); Neuron is in the same category."
        ),
        tests=[
            "test_indexing.py::NumpyTestsNEURON::test_trivial_fancy_out_of_bounds_neuron",
        ],
    ),
    SkipGroup(
        reason=(
            "FFT ops and complex dtypes (cfloat, cdouble, chalf)"
            " are not supported on Neuron hardware. Includes tests that"
            " primarily exercise complex output ops (polar, linalg_eigvals,"
            " stft) or iterate all dtypes including complex."
        ),
        patterns=["*cdouble*", "*cfloat*", "*chalf*", "*complex*", "*fft_*"],
        match_target="test_method",
        tests=s.ATEN_COMPLEX_DTYPE_UNSUPPORTED,
    ),
    SkipGroup(
        reason="CUDA graphs are not supported on Neuron hardware",
        patterns=["*cuda_graph*"],
        match_target="test_method",
    ),
    SkipGroup(
        reason="Long compilation - causes test suite timeout",
        tests=["test_ReplicationPad2d_large_neuron"],
    ),
    SkipGroup(
        reason="Segfault in view ops - requires investigation",
        patterns=["*conj_imag_view*", "*conj_view*"],
        match_target="test_method",
    ),
    SkipGroup(
        reason=(
            "Forward-mode AD (JVP) is not supported on Neuron. Tests fail with"
            " 'InferenceMode::is_enabled() && primal.is_inference() && tangent.is_inference()'"
            " internal assert. Neuron's tensor dispatch interferes with forward AD"
            " inference mode tracking during gradcheck."
        ),
        patterns=["*forward_mode_AD*"],
        tests=[
            "test_ops.py::TestForwardADWithScalarsNEURON::test_0d_tensor_with_python_scalar_div_trunc_rounding_neuron_float32",
        ],
    ),
    SkipGroup(
        reason=(
            "nn.functional.linear_cross_entropy not fully supported on Neuron."
            " Failures include: cross_entropy_loss CPU fallback disabled,"
            " backward fails with broadcast errors, gradcheck failures,"
            " dtype mismatch."
        ),
        patterns=["*linear_cross_entropy*", "*LinearCrossEntropyLoss*"],
    ),
    SkipGroup(
        reason=(
            "Special Bessel functions (scaled_modified_bessel_k0/k1,"
            " spherical_bessel_j0) produce different results on contiguous"
            " vs transposed tensors on Neuron."
        ),
        patterns=[
            "test_contig_vs_transposed_special_scaled_modified_bessel_k0_*",
            "test_contig_vs_transposed_special_scaled_modified_bessel_k1_*",
            "test_contig_vs_transposed_special_spherical_bessel_j0_*",
        ],
    ),
    SkipGroup(
        reason=(
            "TestForeachNEURON::test_all_zero_size_tensors_do_not_launch_kernel_*"
            " is unstable and expensive on Neuron. Each test compiles per"
            " foreach op before reaching the skip/pass decision — observed"
            " durations ranging from 0.02s to 298s on the same test across"
            " runs. Subset of variants (div/sigmoid/sqrt) have flipped from"
            " slow-skip to actual failure with 'silent worker death' (empty"
            " error output). Pattern-skip the whole family to save compile"
            " time and eliminate the flake source. 46 variants covered."
        ),
        patterns=["test_all_zero_size_tensors_do_not_launch_kernel__foreach_*"],
    ),
    SkipGroup(
        reason="Long compilation/execution for conv transpose",
        tests=[
            "test_conv_transpose_with_output_size_and_no_batch_dim",
            "test_conv_transpose_with_output_size_and_no_batch_dim_ConvTranspose2d_neuron",
            "test_conv_transpose_with_output_size_and_no_batch_dim_ConvTranspose3d_neuron",
        ],
    ),
    SkipGroup(
        reason="Functionality Gap: We do not support quantization_per_channel format currently",
        tests=[
            "test_ddp_hooks.py::DistributedDataParallelCommHookTest::test_ddp_comm_hook_quantize_per_channel_hook",
        ],
    ),
    SkipGroup(
        reason=(
            "NotImplementedError: Could not run "
            "aten::_sparse_coo_tensor_with_dims_and_tensors "
            "with arguments from the Sparseneuron backend"
        ),
        tests=[
            "test_c10d_nccl.py::SparseCollective::test_ddp_set_sparse_metadata",
        ],
    ),
    SkipGroup(
        reason="Dynamic shapes not supported on Neuron",
        tests=[
            "DynamicShapesAotAutogradFallbackTests",
            "DynamicShapesCtxManagerTests",
            "DynamicShapesExportTests",
            "DynamicShapesFuncTorchHigherOrderOpTests",
            "DynamicShapesFunctionTests",
            "DynamicShapesHigherOrderOpTests",
            "DynamicShapesMiscTests",
            "DynamicShapesNNModuleTests",
            "DynamicShapesReproTests",
            "DynamicShapesSubGraphTests",
            "DynamicShapesTestSDPA",
            # Compiles with dynamic=True (symbolic shapes); MHA's Q/K/V reshapes
            # then need an i64->index materialization torch-mlir can't legalize.
            # A dynamic-shape config, not a per-op lowering bug (static lowers fine).
            "test_transformers.py::TestTransformersNEURON::test_transformer_encoder_layer_fwd_fake_neuron",
        ],
    ),
    SkipGroup(
        reason="Sparse tensors not supported on Neuron",
        tests=[
            "SparseCollective",
            "TestSparseAnyNEURON",
            "TestSparseMaskedReductionsNEURON",
            "TestSparseNEURON",
            "TestSparseUnaryUfuncsNEURON",
        ],
    ),
    SkipGroup(
        reason="NestedTensor not supported on Neuron",
        tests=[
            "TestNestedTensor",
            "TestNestedTensorAutogradNEURON",
            "TestNestedTensorDeviceTypeNEURON",
            "TestNestedTensorOpInfoNEURON",
            "TestNestedTensorSubclassNEURON",
        ],
    ),
    SkipGroup(
        reason="AOTAutograd cache tests not supported on Neuron",
        tests=[
            "AOTAutogradCacheTests",
            "AOTAutogradCacheBundledTests",
            "AOTAutogradCachePicklerTests",
            "AotAutogradFallbackTests",
        ],
    ),
    SkipGroup(
        reason="Misc tests not supported on Neuron",
        tests=[
            "TestJitNEURON",
            "TestMaskedNEURON",
            "TestMathBitsNEURON",
            "TestCompositeComplianceNEURON",
            "TestTorchDlPackNEURON",
            "TestTagsNEURON",
            "FxGraphRunnableTest",
            "TestPackage",
            "TestFakeDistributedSingleProc",
        ],
    ),
    SkipGroup(
        reason="Sparse tensor ops not supported on Neuron",
        tests=x.UNSUPPORTED_OP_SPARSE,
    ),
    SkipGroup(
        reason=(
            "int64→int32 downcast overflow: test values exceed int32 range,"
            " producing wrong results due to two's complement truncation"
        ),
        tests=s.ATEN_INT64_OVERFLOW,
    ),
    SkipGroup(
        reason=(
            "Needs investigation: Skipping as these tests segfault causing pytest worker crashes"
        ),
        tests=s.ATEN_SEGFAULT,
    ),
    # TODO: Test Groups below are actually functional bugs that need investigation
    SkipGroup(
        reason=(
            "PT2.12 - Conv same_padding backward corrupts NRT state, causing all"
            " subsequent tests on the same xdist worker to fail with"
            " 'NRT model scheduling failed, Unknown status (status: 101)'"
        ),
        tests=s.ATEN_CONV_BACKWARD_NRT_CORRUPTION,
    ),
    SkipGroup(
        reason=(
            "PT2.12 - non-reentrant checkpoint(DataParallel) hits CUDA-only"
            " torch._C._scatter on Neuron. Must skip not xfail: running the body"
            " leaks a recompute frame into the xdist worker, crashing the"
            " subsequent test_full_backward_hook_double_backward"
        ),
        tests=s.ATEN_CHECKPOINT_DATAPARALLEL_LEAK,
    ),
    SkipGroup(
        reason=(
            "Needs investigation: Skipping as these tests are hanging and the"
            " pytest timeout does not kill them"
        ),
        tests=s.ATEN_HANG_NEEDS_INVESTIGATION,
    ),
    SkipGroup(
        reason=(
            "Foreach tests where pytest-xdist workers die during execution. "
            "Observed across multiple ToD runs — workers either crash with an empty-error "
            "'failed' row or fail to report results at all. "
            "Crash mechanism (segfault, abort, OOM) is not yet confirmed."
        ),
        tests=s.ATEN_FOREACH_WORKER_CRASH,
    ),
    SkipGroup(
        reason=("Test hangs indefinitely - worker never reports back, causing workload timeout."),
        tests=[
            "test_binary_op_list_error_cases__foreach_div_neuron_uint8",
            "test_comparison_ops_type_promotion_and_broadcasting_neuron_uint8_float16",
            "test_parity__foreach_tanh_fastpath_outplace_neuron_float64",
            "test_parity__foreach_sqrt_slowpath_inplace_neuron_int64",
            "test_dtypes_nn_functional_dropout3d_neuron",
        ],
    ),
    SkipGroup(
        reason=(
            "Neuron compiler error 'Access pattern out of bounds' in"
            " nkilib foreach_elementwise.addcdiv_kernel — DMACopy bug crashes the worker."
        ),
        tests=[
            "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcdiv_is_fastpath_False_neuron_int32",
        ],
    ),
    SkipGroup(
        reason=(
            "NRT timeout leaves execution unit in unrecoverable state (status 101),"
            " cascading failures to all subsequent tests on the same worker"
        ),
        tests=[
            "test_binary_ufuncs.py::TestBinaryUfuncsNEURON::test_remainder_overflow_neuron",
            "test_ops.py::TestCommonNEURON::test_dtypes___rmod___neuron",
        ],
    ),
    SkipGroup(
        reason=(
            "Ops internally produce complex tensors (polar outputs complex,"
            " linalg_eig returns complex eigenvalues). Neuron does not"
            " support complex dtypes natively; with "
            " TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1"
            " CPU fallback is disabled for these implemented ops so they fail"
            " when complex intermediate tensors flow through aten::stack,"
            " aten::abs, or aten::contiguous."
        ),
        tests=[
            "test_binary_ufuncs.py::TestBinaryUfuncsNEURON::test_batch_vs_slicing__refs__conversions_polar_neuron_float32",
            "test_binary_ufuncs.py::TestBinaryUfuncsNEURON::test_batch_vs_slicing__refs__conversions_polar_neuron_float64",
            "test_binary_ufuncs.py::TestBinaryUfuncsNEURON::test_batch_vs_slicing_polar_neuron_float32",
            "test_binary_ufuncs.py::TestBinaryUfuncsNEURON::test_batch_vs_slicing_polar_neuron_float64",
            "test_decomp.py::TestDecompNEURON::test_comprehensive_linalg_eig_neuron_float32",
            "test_decomp.py::TestDecompNEURON::test_comprehensive_linalg_eig_neuron_float64",
            "test_ops.py::TestCommonNEURON::test_out_linalg_eig_neuron_float32",
        ],
    ),
    SkipGroup(
        reason=(
            "OpInfo metadata mismatch: supported dtypes for __getitem__"
            " and lu_unpack list complex64/complex128 as working in forward"
            " but they are not declared as supported in the OpInfo."
            " Needs OpInfo update upstream."
        ),
        tests=[
            "test_ops.py::TestCommonNEURON::test_dtypes___getitem___neuron",
            "test_ops.py::TestCommonNEURON::test_dtypes_lu_unpack_neuron",
        ],
    ),
    SkipGroup(
        reason=(
            "Extreme duration (>15 min): these tests take 15-84 minutes to complete."
            " They sporadically complete and block workers for long periods."
            " Needs investigation to determine if it's a Neuron compiler issue,"
            " test input size or something else"
        ),
        tests=[
            "test_decomp.py::TestDecompNEURON::test_comprehensive_nn_functional_max_pool3d_neuron_int8",
            "test_decomp.py::TestDecompNEURON::test_comprehensive_max_pool2d_with_indices_backward_neuron_float32",
            "test_decomp.py::TestDecompNEURON::test_comprehensive_nn_functional_max_pool2d_neuron_int16",
            "test_linalg.py::TestLinalgNEURON::test_bmm_neuron_float64",
            "test_decomp.py::TestDecompNEURON::test_comprehensive_linalg_householder_product_neuron_float32",
        ],
    ),
    SkipGroup(
        reason="[timeout] Compilation or execution exceeded 600s timeout on Neuron",
        tests=x.TIMEOUT,
    ),
    SkipGroup(
        reason="[unknown] Failure reason not captured in test output - may cause crashes/segfaults",
        tests=x.UNKNOWN,
    ),
    SkipGroup(
        reason="Neuron compiler fails on Bool dtype inputs for certain ops"
        " (aten::min.dim, aten::convolution). The async error surfaces"
        " during setUp of the next test, poisoning the worker.",
        tests=[
            "test_dtypes_normal_neuron",
            "test_dtypes_nn_functional_hardsigmoid_neuron",
            "test_dtypes_nn_functional_interpolate_trilinear_neuron",
        ],
    ),
    SkipGroup(
        reason=(
            "torch.signal is a beta PyTorch feature that is not supported"
            " on Neuron. The torch.signal.windows.* functions internally"
            " compile to primitives (mul/arange/cos/etc.) that trigger"
            " neuronx-cc failures. Covers all test_signal_windows_functions"
            "_window_* variants and test_kaiser."
        ),
        patterns=[
            "test_signal_windows_functions_window_*",
            "test_kaiser_neuron_*",
        ],
    ),
    SkipGroup(
        reason=(
            "Creates FakeTensors to symbolically trace the op→"
            " FakeTensor.__init__ accesses invalid memory leading to SEGFAULT"
        ),
        tests=[
            "test_out_warning__refs_atleast_3d_neuron",
            "test_python_ref_torch_fallback__refs_arange_neuron_bfloat16",
        ],
    ),
    SkipGroup(
        reason=(
            "Upstream OpInfo for zeros/ones/full declares"
            " @expectedFailure for test_out_warning because CPU/CUDA do not"
            " fire the 'Resized a non-empty tensor but did not warn about it'"
            " UserWarning. Neuron correctly fires the warning, so the test"
            " passes — but pytest reports this as 'Unexpected success' /"
            " failure because the upstream decorator expected it to fail."
            " Skip to avoid the false-positive failure. Revisit if the"
            " upstream OpInfo is corrected, or if we start patching OpInfo"
            " skips in neuron_overrides.py (see TODO)."
        ),
        tests=[
            "test_ops.py::TestCommonNEURON::test_out_warning_zeros_neuron",
            "test_ops.py::TestCommonNEURON::test_out_warning_ones_neuron",
            "test_ops.py::TestCommonNEURON::test_out_warning_full_neuron",
        ],
    ),
    SkipGroup(
        reason=(
            "neuronx-cc assertion NCC_IBCG901 'Incorrect partition set!'"
            " when compiling aten::convolution_backward for 3D conv-transpose"
            " with non-contiguous layout. Test also takes an extreme amount"
            " of time (>10000s) to complete, blocking pytest workers."
        ),
        tests=[
            "test_ops.py::TestCommonNEURON::test_noncontiguous_samples_nn_functional_conv_transpose3d_neuron_float32",
        ],
    ),
    SkipGroup(
        reason=(
            "Upstream PyTorch has a strict xfail marker on this OpInfo entry"
            " (test_meta.py expected_failures dicts / OpInfo decorator) that"
            " assumes the op fails on this backend. Neuron's implementation"
            " actually succeeds, so pytest reports 'Unexpected success' (XPASS)."
            " We cannot remove the upstream marker, so skip on Neuron."
        ),
        tests=s.ATEN_UPSTREAM_STRICT_XFAIL_PASSES_ON_NEURON,
    ),
    SkipGroup(
        reason=(
            "Segfault in torch-mlir ``context_init_hook`` during"
            " ``convert_fx_to_stablehlo`` on the backward pass. Root cause"
            " not yet identified; surfaces as 800s timeout in xdist,"
            " segfault in isolated runs. TODO: investigate."
        ),
        tests=s.ATEN_TEST_DTYPES_MLIR_CONTEXT_SEGFAULT,
    ),
    SkipGroup(
        reason=(
            "Neuron's native_multi_head_attention does not support arbitrary"
            " mask tensors — only mask=None or mask_type=1 (causal masking)."
            " Parametrized variants exercise padding masks and raise"
            " NotImplementedError from"
            " torch_neuronx/python_ops/native_multi_head_attention.py:79."
            " Skipped because the failure is consistent (real implementation"
            " gap, not a flake) and subtest failures don't retry cleanly."
            " Remove this skip when arbitrary mask support is added."
        ),
        patterns=[
            "test_native_multihead_self_attention_*",
        ],
    ),
    SkipGroup(
        reason=(
            "No Neuron lowering for the upsample_bilinear2d / upsample_bicubic2d /"
            " _upsample_bilinear2d_aa / _upsample_bicubic2d_aa ops."
        ),
        tests=s.ATEN_UPSAMPLINGBIMODE2D_NO_LOWERING,
    ),
]
