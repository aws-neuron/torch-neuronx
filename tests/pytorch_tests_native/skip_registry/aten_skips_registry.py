from tests.pytorch_tests_native.neuron_test_registry import SkipGroup
from tests.pytorch_tests_native.skip_registry import skipped_tests as s
from tests.pytorch_tests_native.skip_registry import xfail_tests as x

SKIP_GROUPS: list[SkipGroup] = [
    SkipGroup(
        reason=(
            "[Compiler Bug] Neuron min/max/clamp drop NaN. min/max/clamp -- and"
            " the reductions (amax/amin), clamp-based activations"
            " (hardsigmoid/celu/hardtanh/relu6), and losses (margin_ranking_loss/"
            "hinge_embedding_loss) that decompose to them -- follow IEEE-754"
            " maxNum: a NaN operand is dropped and the non-NaN operand / clamp"
            " bound is returned instead of propagating NaN. StableHLO requires"
            " NaN propagation, so the fix is a spec-conformant NaN-propagating"
            " stablehlo.min/max in the compiler, which would clear all of these"
            " at once (eager + compile). Tracked as unsupported for now."
        ),
        tests=x.MINMAX_CLAMP_NAN_PROPAGATION,
    ),
    SkipGroup(
        reason=(
            "[platform.float64] float64 is not natively supported on Neuron (64-bit types"
            " are computed on a 32-bit datapath), so float64 floor_divide/remainder/div-floor"
            " leave a residual that exceeds test_decomp's strict DecompCrossRefMode tolerance."
        ),
        tests=[
            "test_decomp.py::TestDecompNEURON::test_comprehensive_div_floor_rounding_neuron_float64",
            "test_decomp.py::TestDecompNEURON::test_comprehensive_floor_divide_neuron_float64",
            "test_decomp.py::TestDecompNEURON::test_comprehensive_remainder_neuron_float64",
            "test_decomp.py::TestDecompNEURON::test_quick_div_floor_rounding_neuron_float64",
        ],
    ),
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
            " primarily exercise complex output ops (polar, linalg_eig,"
            " linalg_eigvals, stft) or iterate all dtypes including complex."
            " Also covers linalg_eig/eigvals backward and forward-mode AD:"
            " their backward pass produces complex intermediates"
            " (aten::_linalg_solve_ex) that Neuron cannot execute natively."
        ),
        # ``_fft_r2c`` (rfft) now lowers natively, so it is unskipped. Keep the
        # other FFT variants skipped: c2c, c2r, rfft2, and rfftn are not lowered.
        patterns=[
            "*cdouble*",
            "*cfloat*",
            "*chalf*",
            "*complex*",
            "*fft_fft*",
            "*fft_ifft*",
            "*fft_hfft*",
            "*fft_ihfft*",
            "*fft_irfft*",
            "*fft_rfft2*",
            "*fft_rfftn*",
        ],
        match_target="test_method",
        tests=s.ATEN_COMPLEX_DTYPE_UNSUPPORTED,
    ),
    SkipGroup(
        reason=(
            "uint16, uint32 and uint64 are not supported on Neuron hardware."
            " Tensor creation already raises 'Unsupported ScalarType', and the"
            " dispatcher cast policy has no target for them: unlike int64 they"
            " cannot fold to int32 without changing the value range."
        ),
        tests=[
            "test_decomp.py::TestDecompNEURON::test_comprehensive_unique_neuron_uint16",
            "test_decomp.py::TestDecompNEURON::test_comprehensive_unique_neuron_uint32",
            "test_decomp.py::TestDecompNEURON::test_comprehensive_unique_neuron_uint64",
            "test_meta.py::TestMetaNEURON::test_dispatch_meta_outplace_unique_neuron_uint16",
            "test_meta.py::TestMetaNEURON::test_dispatch_meta_outplace_unique_neuron_uint32",
            "test_meta.py::TestMetaNEURON::test_dispatch_meta_outplace_unique_neuron_uint64",
            "test_meta.py::TestMetaNEURON::test_dispatch_symbolic_meta_outplace_unique_neuron_uint16",
            "test_meta.py::TestMetaNEURON::test_dispatch_symbolic_meta_outplace_unique_neuron_uint32",
            "test_meta.py::TestMetaNEURON::test_dispatch_symbolic_meta_outplace_unique_neuron_uint64",
            "test_meta.py::TestMetaNEURON::test_meta_outplace_unique_neuron_uint16",
            "test_meta.py::TestMetaNEURON::test_meta_outplace_unique_neuron_uint32",
            "test_meta.py::TestMetaNEURON::test_meta_outplace_unique_neuron_uint64",
            "test_tensor_creation_ops.py::TestTensorCreationNEURON::test_cat_out_fast_path_dim0_dim1_neuron_uint16",
            "test_tensor_creation_ops.py::TestTensorCreationNEURON::test_cat_out_fast_path_dim0_dim1_neuron_uint32",
            "test_tensor_creation_ops.py::TestTensorCreationNEURON::test_cat_out_fast_path_dim0_dim1_neuron_uint64",
            "test_torch.py::TestTorchDeviceTypeNEURON::test_item_neuron_uint16",
            "test_torch.py::TestTorchDeviceTypeNEURON::test_item_neuron_uint32",
            "test_torch.py::TestTorchDeviceTypeNEURON::test_item_neuron_uint64",
        ],
    ),
    SkipGroup(
        reason=(
            "float8_e4m3fnuz and float8_e5m2fnuz are AMD ROCm-specific float8"
            " variants. Neuron implements the OCP float8 types, so tensor"
            " creation raises 'Unsupported ScalarType' for these two."
        ),
        tests=[
            "test_decomp.py::TestDecompNEURON::test_comprehensive_eye_neuron_float8_e4m3fnuz",
            "test_decomp.py::TestDecompNEURON::test_comprehensive_eye_neuron_float8_e5m2fnuz",
            "test_decomp.py::TestDecompNEURON::test_quick_eye_neuron_float8_e4m3fnuz",
            "test_decomp.py::TestDecompNEURON::test_quick_eye_neuron_float8_e5m2fnuz",
            "test_ops.py::TestCommonNEURON::test_python_ref__refs_eye_neuron_float8_e4m3fnuz",
            "test_ops.py::TestCommonNEURON::test_python_ref__refs_eye_neuron_float8_e5m2fnuz",
            "test_ops.py::TestCommonNEURON::test_python_ref_torch_fallback__refs_eye_neuron_float8_e4m3fnuz",
            "test_ops.py::TestCommonNEURON::test_python_ref_torch_fallback__refs_eye_neuron_float8_e5m2fnuz",
            "test_torch.py::TestTorchDeviceTypeNEURON::test_item_neuron_float8_e4m3fnuz",
            "test_torch.py::TestTorchDeviceTypeNEURON::test_item_neuron_float8_e5m2fnuz",
        ],
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
            # to_sparse builds a sparse result, so the real op raises before the
            # meta comparison starts: aten::_sparse_coo_tensor_with_dims_and_tensors
            # has no kernel for the Sparseneuron backend.
            "test_meta.py::TestMetaNEURON::test_dispatch_meta_outplace_to_sparse_neuron_*",
            "test_meta.py::TestMetaNEURON::test_dispatch_symbolic_meta_outplace_to_sparse_neuron_*",
            "test_meta.py::TestMetaNEURON::test_meta_outplace_to_sparse_neuron_*",
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
            "int64->int32 downcast overflow: test values exceed int32 range."
            " Neuron wraps with two's complement or raises a conversion error"
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
            "test_binary_ufuncs.py::TestBinaryUfuncs*NEURON::test_remainder_overflow_neuron",
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
        tests=s.COMPLEX_OUTPUT_OPS,
    ),
    SkipGroup(
        reason=(
            "aten::_fft_r2c (rfft) runs natively on Neuron for the common cases."
            " These rfft variants still fail. Each needs a feature that Neuron"
            " does not have yet: complex-to-complex FFT (aten::_fft_c2c) for the"
            " noncontiguous-samples and variant-consistency checks, the rfft"
            " backward in the fake crossref and gradient checks, and the decomp"
            " crossref; the .out compile path; the"
            " OpInfo declared dtype set; and more than one Neuron device."
        ),
        tests=[
            "test_ops.py::TestCommonNEURON::test_dtypes_fft_rfft_neuron",
            "test_ops.py::TestCommonNEURON::test_multiple_devices_fft_rfft_neuron_float32",
            "test_ops.py::TestCommonNEURON::test_multiple_devices_fft_rfft_neuron_int64",
            "test_ops.py::TestCommonNEURON::test_noncontiguous_samples_fft_rfft_neuron_float32",
            "test_ops.py::TestCommonNEURON::test_out_fft_rfft_neuron_float32",
            "test_ops.py::TestCommonNEURON::test_variant_consistency_eager_fft_rfft_neuron_float32",
            "test_ops.py::TestFakeTensorNEURON::test_fake_crossref_backward_amp_fft_rfft_neuron_float32",
            "test_ops.py::TestFakeTensorNEURON::test_fake_crossref_backward_no_amp_fft_rfft_neuron_float32",
            "test_decomp.py::TestDecompNEURON::test_comprehensive_fft_rfft_neuron_float32",
            "test_decomp.py::TestDecompNEURON::test_comprehensive_fft_rfft_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_fft_rfft_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_fft_rfft_neuron_float64",
            "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad_fft_rfft_neuron_float64",
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
            "Upstream OpInfo for zeros/ones/full/eye declares"
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
            "test_ops.py::TestCommonNEURON::test_out_warning_eye_neuron",
        ],
    ),
    SkipGroup(
        reason=(
            "Upstream OpInfo declares @expectedFailure for test_compare_cpu"
            " on these ops (linalg.lu family and special polynomials)."
            " Neuron correctly computes the result, so the test passes —"
            " but pytest reports 'Unexpected success' as a failure."
            " PT 2.14 widened test_compare_cpu from @onlyCUDA to"
            " @onlyAccelerator, exposing this on Neuron for the first time."
        ),
        tests=s.ATEN_COMPARE_CPU_UNEXPECTED_SUCCESS,
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
            "No Neuron lowering for the anti-aliased _upsample_bilinear2d_aa /"
            " _upsample_bicubic2d_aa ops."
        ),
        tests=s.ATEN_UPSAMPLINGBIMODE2D_AA_NO_LOWERING,
    ),
    SkipGroup(
        reason=(
            "The upstream test makes a tensor with 65 dimensions. These masked"
            " reductions build their mask with torch.ones, so a 65-dimension"
            " aten::ones reaches neuronx-cc, which supports a maximum of 64."
            " The test still passes, because the RuntimeError it asserts comes"
            " from the host check in ATen/WrapDimUtilsMulti.h. The compilation"
            " failure stays queued and appears in a later test. A skip stops the"
            " dispatch. An xfail keeps the dispatch, so the test still queues"
            " the error."
        ),
        tests=[
            "test_reductions.py::TestReductionsNEURON::test_dim_ndim_limit_masked_mean_neuron",
            "test_reductions.py::TestReductionsNEURON::test_dim_ndim_limit_masked_std_neuron",
            "test_reductions.py::TestReductionsNEURON::test_dim_ndim_limit_masked_var_neuron",
        ],
    ),
    SkipGroup(
        reason=(
            "[platform.float64 + on-device RNG] float64 is not natively supported on Neuron "
            "(64-bit tensors run on a 32-bit datapath), and the on-device RNG draws "
            "different inputs than the CPU-RNG reference, so these fp64 gradcheck/reduction/"
            "comparison tests exceed tolerance. Skipped as a platform limitation (not an op bug). "
        ),
        tests=[
            "test_foreach.py::TestForeachNEURON::test_autodiff__foreach_lgamma_outplace_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_linalg_lu_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_linalg_lu_factor_ex_neuron_float64",
            "test_tensor_creation_ops.py::TestTensorCreationNEURON::test_random_default_neuron_float64",
            "test_decomp.py::TestDecompNEURON::test_quick_core_backward_logaddexp2_neuron_float64",
            "test_decomp.py::TestDecompNEURON::test_quick_logaddexp2_neuron_float64",
            "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad_nn_functional_hardtanh_neuron_float64",
            "test_reductions.py::TestReductionsNEURON::test_ref_large_input_1D__refs_sum_neuron_float64",
            "test_reductions.py::TestReductionsNEURON::test_ref_large_input_1D_masked_sum_neuron_float64",
            "test_reductions.py::TestReductionsNEURON::test_ref_large_input_1D_sum_neuron_float64",
            "nn/test_init.py::TestNNInitDeviceTypeNEURON::test_trunc_normal_all_dtypes_dims_1_float64_neuron_float64",
            "nn/test_init.py::TestNNInitDeviceTypeNEURON::test_trunc_normal_all_dtypes_dims_2_float64_neuron_float64",
            "nn/test_init.py::TestNNInitDeviceTypeNEURON::test_trunc_normal_unique_values_float64_min_unique_5000_neuron_float64",
            "test_binary_ufuncs.py::TestBinaryUfuncs*NEURON::test_div_rounding_numpy_neuron_float64",
            "test_binary_ufuncs.py::TestBinaryUfuncs*NEURON::test_reference_numerics_extremal_values__refs_remainder_neuron_float64",
            "test_binary_ufuncs.py::TestBinaryUfuncs*NEURON::test_reference_numerics_extremal_values_floor_divide_neuron_float64",
            "test_binary_ufuncs.py::TestBinaryUfuncs*NEURON::test_reference_numerics_extremal_values_remainder_neuron_float64",
            "test_binary_ufuncs.py::TestBinaryUfuncs*NEURON::test_reference_numerics_large_values_floor_divide_neuron_float64",
            "test_binary_ufuncs.py::TestBinaryUfuncsDeviceNEURON::test_reference_numerics_floor_divide_neuron_float64",
            "test_decomp.py::TestDecompNEURON::test_comprehensive_addmm_decomposed_neuron_float64",
            "test_decomp.py::TestDecompNEURON::test_comprehensive_addmm_neuron_float64",
            "test_decomp.py::TestDecompNEURON::test_comprehensive_baddbmm_neuron_float64",
            "test_decomp.py::TestDecompNEURON::test_comprehensive_constant_pad_nd_neuron_float64",
            "test_decomp.py::TestDecompNEURON::test_comprehensive_native_layer_norm_neuron_float64",
            "test_decomp.py::TestDecompNEURON::test_comprehensive_nn_functional_layer_norm_neuron_float64",
            "test_decomp.py::TestDecompNEURON::test_comprehensive_nn_functional_pad_constant_neuron_float64",
            "test_decomp.py::TestDecompNEURON::test_comprehensive_nn_functional_rms_norm_neuron_float64",
            "test_decomp.py::TestDecompNEURON::test_comprehensive_nn_functional_softmin_neuron_float64",
            "test_decomp.py::TestDecompNEURON::test_comprehensive_nn_functional_softmin_with_dtype_neuron_float64",
            "test_decomp.py::TestDecompNEURON::test_quick_addmm_decomposed_neuron_float64",
            "test_decomp.py::TestDecompNEURON::test_quick_addmm_neuron_float64",
            "test_decomp.py::TestDecompNEURON::test_quick_baddbmm_neuron_float64",
            "test_decomp.py::TestDecompNEURON::test_quick_constant_pad_nd_neuron_float64",
            "test_decomp.py::TestDecompNEURON::test_quick_mv_neuron_float64",
            "test_decomp.py::TestDecompNEURON::test_quick_nn_functional_pad_constant_neuron_float64",
            "test_decomp.py::TestDecompNEURON::test_quick_vdot_neuron_float64",
            "test_linalg.py::TestLinalgNEURON::test_linalg_lu_solve_neuron_float64",
            "test_modules.py::TestModuleNEURON::test_grad_nn_BCEWithLogitsLoss_neuron_float64",
            "test_modules.py::TestModuleNEURON::test_grad_nn_GaussianNLLLoss_neuron_float64",
            "test_modules.py::TestModuleNEURON::test_grad_nn_HuberLoss_neuron_float64",
            "test_modules.py::TestModuleNEURON::test_grad_nn_LeakyReLU_neuron_float64",
            "test_modules.py::TestModuleNEURON::test_grad_nn_NLLLoss_neuron_float64",
            "test_modules.py::TestModuleNEURON::test_grad_nn_PReLU_neuron_float64",
            "test_modules.py::TestModuleNEURON::test_grad_nn_SmoothL1Loss_neuron_float64",
            "test_modules.py::TestModuleNEURON::test_gradgrad_nn_AvgPool2d_neuron_float64",
            "test_modules.py::TestModuleNEURON::test_gradgrad_nn_BatchNorm1d_eval_mode_neuron_float64",
            "test_modules.py::TestModuleNEURON::test_gradgrad_nn_BatchNorm3d_eval_mode_neuron_float64",
            "test_modules.py::TestModuleNEURON::test_gradgrad_nn_GaussianNLLLoss_neuron_float64",
            "test_modules.py::TestModuleNEURON::test_gradgrad_nn_PoissonNLLLoss_neuron_float64",
            "test_modules.py::TestModuleNEURON::test_gradgrad_nn_RNN_eval_mode_neuron_float64",
            "test_ops.py::TestCommonNEURON::test_python_ref__refs_constant_pad_nd_neuron_float64",
            "test_ops.py::TestCommonNEURON::test_python_ref__refs_floor_divide_neuron_float64",
            "test_ops.py::TestCommonNEURON::test_python_ref__refs_linalg_vector_norm_neuron_float64",
            "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad___getitem___neuron_float64",
            "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad___rmul___neuron_float64",
            "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad_acos_neuron_float64",
            "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad_addcmul_neuron_float64",
            "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad_cholesky_solve_neuron_float64",
            "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad_clamp_max_neuron_float64",
            "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad_clamp_min_neuron_float64",
            "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad_clamp_neuron_float64",
            "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad_frexp_neuron_float64",
            "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad_kron_neuron_float64",
            "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad_linalg_eigh_neuron_float64",
            "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad_linalg_tensorsolve_neuron_float64",
            "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad_linalg_vector_norm_neuron_float64",
            "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad_lu_neuron_float64",
            "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad_max_binary_neuron_float64",
            "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad_maximum_neuron_float64",
            "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad_min_binary_neuron_float64",
            "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad_minimum_neuron_float64",
            "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad_mul_neuron_float64",
            "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad_nn_functional_max_pool3d_neuron_float64",
            "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad_nn_functional_threshold_neuron_float64",
            "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad_remainder_neuron_float64",
            "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad_scatter_reduce_amax_neuron_float64",
            "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad_scatter_reduce_amin_neuron_float64",
            "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad_svd_lowrank_neuron_float64",
            "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad_nn_functional_interpolate_trilinear_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad___getitem___neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad__native_batch_norm_legit_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad__upsample_bilinear2d_aa_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_addmm_decomposed_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_addmm_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_addmv_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_clamp_max_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_clamp_min_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_dist_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_dot_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_frexp_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_inner_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_linalg_solve_triangular_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_linalg_tensorsolve_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_linalg_vector_norm_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_lu_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_masked_std_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_masked_var_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_matrix_exp_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_mode_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_native_layer_norm_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_nn_functional_batch_norm_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_nn_functional_binary_cross_entropy_with_logits_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_nn_functional_grid_sample_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_nn_functional_interpolate_trilinear_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_nn_functional_threshold_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_norm_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_scatter_reduce_amax_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_scatter_reduce_amin_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_trapezoid_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_trapz_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad___getitem___neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad__native_batch_norm_legit_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_cholesky_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_clamp_max_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_clamp_min_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_clamp_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_corrcoef_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_cumprod_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_div_no_rounding_mode_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_frexp_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_index_reduce_amax_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_index_reduce_amin_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_lerp_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_linalg_cholesky_ex_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_linalg_cholesky_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_masked_var_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_matrix_exp_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_max_binary_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_max_reduction_no_dim_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_maximum_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_min_binary_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_min_reduction_no_dim_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_minimum_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_nn_functional_batch_norm_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_nn_functional_grid_sample_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_nn_functional_interpolate_trilinear_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_nn_functional_layer_norm_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_nn_functional_threshold_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_nn_functional_unfold_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_pca_lowrank_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_scatter_reduce_amax_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_scatter_reduce_amin_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_std_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_triangular_solve_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_true_divide_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_inplace_grad_addmm_decomposed_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_inplace_grad_addmm_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_inplace_grad_clamp_max_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_inplace_grad_clamp_min_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_inplace_grad_mul_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_inplace_grad_nn_functional_threshold_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_inplace_grad_polygamma_polygamma_n_0_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_inplace_grad_scatter_reduce_amax_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_inplace_grad_scatter_reduce_amin_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_inplace_grad_tan_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_inplace_gradgrad_clamp_max_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_inplace_gradgrad_clamp_min_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_inplace_gradgrad_clamp_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_inplace_gradgrad_index_reduce_amax_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_inplace_gradgrad_index_reduce_amin_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_inplace_gradgrad_mul_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_inplace_gradgrad_nn_functional_threshold_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_inplace_gradgrad_polygamma_polygamma_n_0_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_inplace_gradgrad_remainder_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_inplace_gradgrad_scatter_reduce_amax_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_inplace_gradgrad_scatter_reduce_amin_neuron_float64",
            "test_ops_gradients.py::TestBwdGradientsNEURON::test_inplace_gradgrad_tan_neuron_float64",
            "test_reductions.py::TestReductionsNEURON::test_histc_min_max_corner_cases_neuron_float64",
            "test_torch.py::TestTorchDeviceTypeNEURON::test_diff_neuron_float64",
            "test_torch.py::TestTorchDeviceTypeNEURON::test_diff_noncontig_neuron_float64",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_extremal__refs_tan_neuron_float64",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_extremal__refs_trunc_neuron_float64",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_extremal_tan_neuron_float64",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_extremal_trunc_neuron_float64",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_large__refs_ceil_neuron_float64",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_large__refs_trunc_neuron_float64",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_large_ceil_neuron_float64",
            "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_large_trunc_neuron_float64",
        ],
    ),
    SkipGroup(
        reason="Tests use torch.compile with a non-Neuron backend (eager/inductor/aot_eager)",
        tests=s.ATEN_NON_NEURON_COMPILE_BACKEND,
    ),
    SkipGroup(
        reason="Assumes torch is linked with CUDA — not applicable on Neuron",
        tests=[
            "FakeTensorOperatorInvariants::test_fake_gpu_no_init",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT 2.14][platform.float64] float64 gradgrad tests added in PT 2.14."
            " Neuron does not support native float64; the float64->float32"
            " precision loss causes gradcheck failures. xfail (not skip)"
            " so a future float64 story would xpass and flag it."
        ),
        tests=[
            "test_autograd.py::TestAutogradDeviceTypeNEURON::test_cdist_gradgrad_neuron_float64",
            "test_autograd.py::TestAutogradDeviceTypeNEURON::test_pdist_gradgrad_neuron_float64",
            # test_pdist_neuron uses torch.double internally; gradcheck fails
            # due to float64→float32 precision loss (same root cause)
            "test_nn.py::TestNNDeviceTypeNEURON::test_pdist_neuron",
        ],
    ),
    SkipGroup(
        reason=(
            "Neuron registers ops without schemas (schemaless boxed dispatch)."
            " test_find_dangling_impls expects all registered ops to have a schema."
        ),
        tests=[
            "TestDispatch::test_find_dangling_impls",
            "TestDispatch::test_find_dangling_impls_ext",
        ],
    ),
    SkipGroup(
        reason="Requires z3-solver package (not installed on Neuron hosts)",
        tests=[
            "TestSymbolicTracing::test_constant_specialization",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT 2.14][platform.float64] New NEURON device-type test instantiations added"
            " in PT 2.14. Neuron does not support native float64; the"
            " float64->float32 precision loss causes accuracy mismatches,"
            " Jacobian failures, or compilation errors. xfail (not skip)"
            " so a future float64 story would xpass and flag it."
        ),
        tests=s.ATEN_PLATFORM_FLOAT64_NEW_INSTANTIATIONS,
    ),
    SkipGroup(
        reason=(
            "[PT 2.14][platform.float64] New NEURON device-type test instantiations added"
            " in PT 2.14 iter2. Neuron does not support native float64; the"
            " float64->float32 precision loss causes accuracy mismatches,"
            " Jacobian failures, ComplexDouble cast errors, or compilation"
            " errors. xfail (not skip) so a future float64 story would xpass."
        ),
        tests=s.ATEN_PLATFORM_FLOAT64_NEW_INSTANTIATIONS_ITER2,
    ),
    SkipGroup(
        reason=(
            "float8_e4m3fnuz and float8_e5m2fnuz are AMD ROCm-specific float8"
            " variants not supported on Neuron hardware. Unlike the standard OCP"
            " float8 formats (e4m3fn, e5m2), these fnuz variants use a different"
            " NaN/zero encoding and are not a Neuron roadmap item."
        ),
        patterns=[
            "*float8_e4m3fnuz*",
            "*float8_e5m2fnuz*",
        ],
    ),
]
