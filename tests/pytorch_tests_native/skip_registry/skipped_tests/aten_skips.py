"""Aten skip lists — extracted from aten_skips_registry.py for readability."""

ATEN_INT64_OVERFLOW = [
    "test_comprehensive_argwhere_neuron_int64",
    "test_comprehensive_index_copy_neuron_int64",
    "test_comprehensive_index_select_neuron_int64",
    "test_comprehensive_masked_select_neuron_int64",
    "test_comprehensive_nonzero_neuron_int64",
    "test_comprehensive_scatter_neuron_int64",
    "test_deterministic_empty_neuron_int64",
    "test_dispatch_meta_inplace_index_copy_neuron_int64",
    "test_dispatch_meta_inplace_scatter_neuron_int64",
    "test_dispatch_meta_outplace_argwhere_neuron_int64",
    "test_dispatch_meta_outplace_index_copy_neuron_int64",
    "test_dispatch_meta_outplace_masked_select_neuron_int64",
    "test_dispatch_meta_outplace_nonzero_neuron_int64",
    "test_dispatch_meta_outplace_scatter_neuron_int64",
    "test_dispatch_symbolic_meta_inplace_index_copy_neuron_int64",
    "test_dispatch_symbolic_meta_inplace_scatter_neuron_int64",
    "test_dispatch_symbolic_meta_outplace_argwhere_neuron_int64",
    "test_dispatch_symbolic_meta_outplace_index_copy_neuron_int64",
    "test_dispatch_symbolic_meta_outplace_masked_select_neuron_int64",
    "test_dispatch_symbolic_meta_outplace_nonzero_neuron_int64",
    "test_dispatch_symbolic_meta_outplace_scatter_neuron_int64",
    "test_div_rounding_modes_neuron_int64",
    "test_div_rounding_numpy_neuron_int64",
    "test_fmod_remainder_overflow_neuron_int64",
    "test_index_reduce_reduce_amax_neuron_int64",
    "test_index_reduce_reduce_amin_neuron_int64",
    "test_large_linspace_neuron_int64",
    "test_low_high_boolean_integral2_neuron_int64",
    "test_meta_inplace_index_copy_neuron_int64",
    "test_meta_inplace_scatter_neuron_int64",
    "test_meta_outplace_argwhere_neuron_int64",
    "test_meta_outplace_index_copy_neuron_int64",
    "test_meta_outplace_masked_select_neuron_int64",
    "test_meta_outplace_nonzero_neuron_int64",
    "test_meta_outplace_scatter_neuron_int64",
    "test_noncontiguous_samples_argwhere_neuron_int64",
    "test_noncontiguous_samples_index_copy_neuron_int64",
    "test_noncontiguous_samples_masked_select_neuron_int64",
    "test_noncontiguous_samples_nonzero_neuron_int64",
    "test_noncontiguous_samples_scatter_neuron_int64",
    "test_numpy_ref_argwhere_neuron_int64",
    "test_python_ref_meta__refs_index_copy_neuron_int64",
    "test_python_ref_meta__refs_index_select_neuron_int64",
    "test_python_ref_torch_fallback__refs_index_copy_neuron_int64",
    "test_python_ref_torch_fallback__refs_index_select_neuron_int64",
    "test_quick_index_copy_neuron_int64",
    "test_quick_index_select_neuron_int64",
    "test_reference_numerics_small__refs_exp_neuron_int64",
    "test_reference_numerics_small_exp_neuron_int64",
]

ATEN_SEGFAULT = [
    "test_foreach_reduce_large_input__foreach_norm_w_empty_False_neuron_int32",
    "test_meta_inplace__foreach_sub_neuron_int8",
    "test_dtypes_constant_pad_nd_neuron",
    "test_dtypes_pca_lowrank_neuron",
    "test_mm_bmm_non_memory_dense_neuron",
    "test_mm_conjtranspose_neuron",
    "test_numpy_ref_argwhere_neuron_float64",
    "test_new_empty_strided_neuron",
]

# Foreach tests where pytest-xdist workers die during execution.
# Observed across multiple ToD runs on test_foreach.py — workers either crash with
# an empty-error "failed" row or fail to report results at all
ATEN_FOREACH_WORKER_CRASH = [
    "test_parity__foreach_addcdiv_fastpath_outplace_neuron_float16",
    "test_parity__foreach_addcdiv_slowpath_inplace_neuron_bfloat16",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcdiv_is_fastpath_False_neuron_bool",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcdiv_is_fastpath_False_neuron_float16",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcdiv_is_fastpath_False_neuron_float32",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcdiv_is_fastpath_False_neuron_float64",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcdiv_is_fastpath_False_neuron_int16",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcdiv_is_fastpath_False_neuron_int32",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcdiv_is_fastpath_False_neuron_int64",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcdiv_is_fastpath_False_neuron_int8",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcdiv_is_fastpath_False_neuron_uint8",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcdiv_is_fastpath_True_neuron_bfloat16",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcdiv_is_fastpath_True_neuron_bool",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcdiv_is_fastpath_True_neuron_float16",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcdiv_is_fastpath_True_neuron_float32",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcdiv_is_fastpath_True_neuron_float64",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcdiv_is_fastpath_True_neuron_int16",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcdiv_is_fastpath_True_neuron_int32",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcdiv_is_fastpath_True_neuron_int64",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcdiv_is_fastpath_True_neuron_int8",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcdiv_is_fastpath_True_neuron_uint8",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcmul_is_fastpath_False_neuron_bfloat16",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcmul_is_fastpath_False_neuron_bool",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcmul_is_fastpath_False_neuron_float16",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcmul_is_fastpath_False_neuron_float32",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcmul_is_fastpath_False_neuron_float64",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcmul_is_fastpath_False_neuron_int16",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcmul_is_fastpath_False_neuron_int32",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcmul_is_fastpath_False_neuron_int64",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcmul_is_fastpath_False_neuron_int8",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcmul_is_fastpath_False_neuron_uint8",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcmul_is_fastpath_True_neuron_bfloat16",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcmul_is_fastpath_True_neuron_bool",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcmul_is_fastpath_True_neuron_float16",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcmul_is_fastpath_True_neuron_float32",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcmul_is_fastpath_True_neuron_float64",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcmul_is_fastpath_True_neuron_int16",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcmul_is_fastpath_True_neuron_int32",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcmul_is_fastpath_True_neuron_int64",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcmul_is_fastpath_True_neuron_int8",
    "test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcmul_is_fastpath_True_neuron_uint8",
    "test_binary_op_list_error_cases__foreach_mul_neuron_float32",
]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Complex dtype unsupported on Neuron
# ---------------------------------------------------------------------------

# Tests whose primary purpose is to exercise complex-output ops
# (polar, linalg_eigvals, stft) or complex type-promotion behavior.
# These are added alongside the *complex*/*fft_* pattern-based skip because
# their test method names don't match those patterns.
ATEN_COMPLEX_DTYPE_UNSUPPORTED = [
    # test_ones iterates every dtype, including complex32 (ScalarType 8)
    "test_tensor_creation_ops.py::TestTensorCreationNEURON::test_ones_neuron",
    # polar: outputs complex by definition (re^iθ → complex)
    "test_binary_ufuncs.py::TestBinaryUfuncs*NEURON::test_contig_size1__refs__conversions_polar_neuron_float64",
    "test_binary_ufuncs.py::TestBinaryUfuncs*NEURON::test_contig_size1_large_dim__refs__conversions_polar_neuron_float64",
    "test_binary_ufuncs.py::TestBinaryUfuncs*NEURON::test_contig_size1_large_dim_polar_neuron_float64",
    "test_binary_ufuncs.py::TestBinaryUfuncs*NEURON::test_contig_size1_polar_neuron_float64",
    "test_binary_ufuncs.py::TestBinaryUfuncs*NEURON::test_contig_vs_every_other__refs__conversions_polar_neuron_float64",
    "test_binary_ufuncs.py::TestBinaryUfuncs*NEURON::test_contig_vs_every_other_polar_neuron_float64",
    "test_binary_ufuncs.py::TestBinaryUfuncs*NEURON::test_contig_vs_transposed__refs__conversions_polar_neuron_float64",
    "test_binary_ufuncs.py::TestBinaryUfuncs*NEURON::test_contig_vs_transposed_polar_neuron_float64",
    "test_binary_ufuncs.py::TestBinaryUfuncs*NEURON::test_non_contig__refs__conversions_polar_neuron_float64",
    "test_binary_ufuncs.py::TestBinaryUfuncs*NEURON::test_non_contig_expand__refs__conversions_polar_neuron_float64",
    "test_binary_ufuncs.py::TestBinaryUfuncs*NEURON::test_non_contig_expand_polar_neuron_float64",
    "test_binary_ufuncs.py::TestBinaryUfuncs*NEURON::test_non_contig_index__refs__conversions_polar_neuron_float64",
    "test_binary_ufuncs.py::TestBinaryUfuncs*NEURON::test_non_contig_index_polar_neuron_float64",
    "test_binary_ufuncs.py::TestBinaryUfuncs*NEURON::test_non_contig_polar_neuron_float64",
    "test_decomp.py::TestDecompNEURON::test_comprehensive_polar_neuron_float64",
    "test_ops.py::TestCommonNEURON::test_out_polar_neuron_float32",
    # linalg_eig / linalg_eigvals: outputs complex eigenvalues for real input.
    # The backward pass calls aten::_linalg_solve_ex with a complex intermediate.
    # With TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1 there is no
    # CPU fallback, so all tests that reach the backward or eigenvalue output fail.
    "test_ops.py::TestCommonNEURON::test_noncontiguous_samples_linalg_eigvals_neuron_float32",
    "test_ops.py::TestCommonNEURON::test_out_linalg_eigvals_neuron_float32",
    "test_ops.py::TestCommonNEURON::test_out_warning_linalg_eig_neuron",
    "test_ops.py::TestCommonNEURON::test_out_warning_linalg_eigvals_neuron",
    "test_decomp.py::TestDecompNEURON::test_comprehensive_linalg_eigvals_neuron_float32",
    "test_decomp.py::TestDecompNEURON::test_comprehensive_linalg_eigvals_neuron_float64",
    "test_ops.py::TestCommonNEURON::test_dtypes_linalg_eig_neuron",
    "test_ops.py::TestCommonNEURON::test_dtypes_linalg_eigvals_neuron",
    "test_ops.py::TestCommonNEURON::test_dtypes_stft_neuron",
    "test_ops.py::TestCommonNEURON::test_noncontiguous_samples_linalg_eig_neuron_float32",
    "test_ops.py::TestCommonNEURON::test_variant_consistency_eager_linalg_eigvals_neuron_float32",
    "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_linalg_eig_neuron_float64",
    "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_linalg_eigvals_neuron_float64",
    "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_linalg_eig_neuron_float64",
    "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_linalg_eigvals_neuron_float64",
    "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad_linalg_eig_neuron_float64",
    "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad_linalg_eigvals_neuron_float64",
    "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_forward_mode_AD_linalg_eig_neuron_float64",
    "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_forward_mode_AD_linalg_eigvals_neuron_float64",
    # block_diag: iterates all dtypes including complex64/complex128
    "test_tensor_creation_ops.py::TestTensorCreationNEURON::test_block_diag_neuron",
    "test_tensor_creation_ops.py::TestTensorCreationNEURON::test_block_diag_scipy_neuron",
    # linspace deduction: helper passes complex endpoints (1j, 2j) expecting a
    # cfloat result; Neuron rejects complex with TypeError (crash, not wrong
    # output) so this is a skip, not an xfail.
    "test_tensor_creation_ops.py::TestTensorCreationNEURON::test_linspace_deduction_neuron",
    # stft: FFT produces complex output
    "test_ops.py::TestCommonNEURON::test_python_ref_torch_fallback__refs_stft_neuron_float32",
    "test_ops.py::TestCommonNEURON::test_python_ref_torch_fallback__refs_stft_neuron_float64",
    # clone/copy: test iterates all dtypes including complex
    "test_torch.py::TestTorchDeviceTypeNEURON::test_clone_all_dtypes_and_devices_neuron",
    "test_torch.py::TestTorchDeviceTypeNEURON::test_copy_all_dtypes_and_devices_neuron",
    # roll / cat / addcdiv: test iterates all dtypes including complex; same
    # pattern as `where` but via the op's own sample_inputs_func rather than
    # reference_inputs_where, so the where-override doesn't apply.
    "test_tensor_creation_ops.py::TestTensorCreationNEURON::test_cat_all_dtypes_and_devices_neuron",
    "test_tensor_creation_ops.py::TestTensorCreationNEURON::test_roll_neuron",
    "test_type_promotion.py::TestTypePromotionNEURON::test_ternary_out_promotion_neuron",
    # where scalar handcrafted values: hardcodes complex64/complex128 dtypes
    # and complex scalar values (1+0.5j, 2+0.5j) directly in the test body,
    # bypassing OpInfo-based dtype filtering.
    "test_torch.py::TestTorchDeviceTypeNEURON::test_where_scalar_handcrafted_values_neuron",
]


# Tests with a strict upstream xfail marker that actually pass on Neuron.
# The xfail decorator lives in upstream PyTorch (OpInfo / test_meta.py
# expected_failures dicts), which assumes the op fails on this backend. The
# Neuron implementation succeeds, so pytest reports "Unexpected success". We
# cannot remove the marker without upstream patching, so we skip these
# tests on Neuron to avoid false XPASS failures.
ATEN_UPSTREAM_STRICT_XFAIL_PASSES_ON_NEURON = [
    # Upstream's comprehensive_failures marks uint8 bilinear interpolate as
    # xfail with the comment "off by one error". The Neuron lowering reproduces
    # aten's half-up rounding and its narrowing after each 1-D pass, so it matches
    # element for element and the test succeeds unexpectedly. (uint8 *bicubic* is
    # still off by one and its upstream xfail continues to hold.)
    "test_decomp.py::TestDecompNEURON::test_comprehensive_nn_functional_interpolate_bilinear_neuron_uint8",
    "test_decomp.py::TestDecompNEURON::test_comprehensive_nn_functional_upsample_bilinear_neuron_uint8",
    "test_meta.py::TestMetaNEURON::test_dispatch_meta_outplace__foreach_lerp_neuron_int64",
    "test_meta.py::TestMetaNEURON::test_dispatch_meta_outplace__foreach_sub_neuron_float64",
    "test_meta.py::TestMetaNEURON::test_dispatch_meta_outplace__foreach_sub_neuron_int64",
    "test_meta.py::TestMetaNEURON::test_dispatch_symbolic_meta_outplace__foreach_lerp_neuron_int64",
    "test_meta.py::TestMetaNEURON::test_dispatch_symbolic_meta_outplace__foreach_sub_neuron_float64",
    "test_meta.py::TestMetaNEURON::test_dispatch_symbolic_meta_outplace__foreach_sub_neuron_int64",
    "test_meta.py::TestMetaNEURON::test_meta_outplace__foreach_lerp_neuron_int64",
    "test_meta.py::TestMetaNEURON::test_meta_outplace__foreach_sub_neuron_float64",
    "test_meta.py::TestMetaNEURON::test_meta_outplace__foreach_sub_neuron_int64",
    # matmul OpInfo xfails test_out ("Strides are not the same!"); Neuron now
    # binds the strided, non-contiguous out= correctly, so the test passes.
    "test_ops.py::TestCommonNEURON::test_out_matmul_neuron_float32",
]


# Conv same_padding backward tests corrupt NRT state, causing all subsequent
# tests on the same pytest-xdist worker to fail with "NRT model scheduling
# failed, Unknown status (status: 101)". Skip to prevent cascading failures.
ATEN_CONV_BACKWARD_NRT_CORRUPTION = [
    "nn/test_convolution.py::TestConvolutionNNDeviceTypeNEURON::test_conv1d_same_padding_backward_neuron_float32",
    "nn/test_convolution.py::TestConvolutionNNDeviceTypeNEURON::test_conv2d_same_padding_backward_neuron_float32",
]


# PT2.12 - non-reentrant checkpoint(DataParallel) hits CUDA-only
# torch._C._scatter. Skip (not xfail): its body leaks a recompute frame
# into the worker, later crashing test_full_backward_hook_double_backward.
ATEN_CHECKPOINT_DATAPARALLEL_LEAK = [
    "test_autograd.py::TestAutograd::test_checkpointing_without_reentrant_dataparallel",
]


# test_dtypes variants that segfault in torch-mlir's context_init_hook
# during convert_fx_to_stablehlo on the backward pass. Reproduces across
# all nightly and PT 2.11 runs. Root cause not yet identified.
# TODO: investigate MLIR context / StableHLO dialect registration.
ATEN_TEST_DTYPES_MLIR_CONTEXT_SEGFAULT = [
    "test_ops.py::TestCommonNEURON::test_dtypes_linalg_matrix_norm_neuron",
    "test_ops.py::TestCommonNEURON::test_dtypes_masked_normalize_neuron",
]

# No Neuron lowering for the anti-aliased _upsample_bicubic2d_aa. The
# antialias=False subtests lower natively via upsample_bicubic2d, and the
# _upsample_bilinear2d_aa (mode=bilinear) subtests now lower via the AA
# bilinear decomposition — both are no longer skipped.
ATEN_UPSAMPLINGBIMODE2D_AA_NO_LOWERING = [
    "test_nn.py::TestNNDeviceTypeNEURON::test_upsamplingBiMode2d_antialias_True_align_corners_False_mode_bicubic_memory_format0_neuron",
    "test_nn.py::TestNNDeviceTypeNEURON::test_upsamplingBiMode2d_antialias_True_align_corners_False_mode_bicubic_memory_format1_neuron",
    "test_nn.py::TestNNDeviceTypeNEURON::test_upsamplingBiMode2d_antialias_True_align_corners_True_mode_bicubic_memory_format0_neuron",
    "test_nn.py::TestNNDeviceTypeNEURON::test_upsamplingBiMode2d_antialias_True_align_corners_True_mode_bicubic_memory_format1_neuron",
]

# Tests that explicitly pass backend="eager", backend="inductor", or
# backend="aot_eager" to torch.compile. These backends target CPU/GPU and
# are not applicable on Neuron.
ATEN_NON_NEURON_COMPILE_BACKEND = [
    # test_custom_ops.py
    "TestCustomOpAPI::test_compile",
    "TestCustomOpAPI::test_custom_op_inplace_tag_compile_eager",
    "TestCustomOpAPI::test_custom_op_inplace_tag_compile_eager_fullgraph",
    "TestCustomOpAPI::test_custom_op_inplace_tag_compile_inductor_fullgraph",
    "TestCustomOpAPI::test_custom_op_out_tag_compile_eager",
    "TestCustomOpAPI::test_custom_op_out_tag_compile_eager_fullgraph",
    "TestCustomOpAPI::test_custom_op_out_tag_compile_inductor_fullgraph",
    # test_fake_tensor.py
    "FakeTensorTest::test_embedding_meta_indices_torch_compile",
    # test_opaque_obj_v2.py
    "TestOpaqueObject::test_compile1",
    "TestOpaqueObject::test_compile_contiguous_strides",
    "TestOpaqueObject::test_compile_create_intermediate",
    "TestOpaqueObject::test_compile_exact_strides",
    "TestOpaqueObject::test_compile_fixed_stride_order",
    "TestOpaqueObject::test_compile_inline_methods",
    "TestOpaqueObject::test_custom_op_reference_return",
    "TestOpaqueObject::test_hoist_basic",
    "TestOpaqueObject::test_hoist_cache_hits",
    "TestOpaqueObject::test_multiple_opaque_objects_in_backward",
    "TestOpaqueObject::test_op_passthrough_counter_in_tuple",
    "TestOpaqueObject::test_opaque_obj_saved_for_backward",
    "TestOpaqueObject::test_opaque_produced_by_call_function_saved_for_backward",
    "TestOpaqueObject::test_script_object_intermediate_exposed_from_checkpoint",
    "TestOpaqueObject::test_subclass_opaque_attrs_cache_hit",
    "TestOpaqueObject::test_tensor_subclass_opaque_backward_compiled_autograd",
    "TestOpaqueObject::test_tensor_subclass_shared_opaque_backward",
    "TestOpaqueObject::test_tensor_subclass_shared_opaque_remapping",
    "TestOpaqueObject::test_value_type_graph_intermediate",
    "TestOpaqueObject::test_value_type_graph_output_backend_aot_eager",
    "TestOpaqueObject::test_value_type_graph_output_backend_eager",
    "TestOpaqueObject::test_value_type_graph_output_backend_inductor",
    "TestOpaqueObject::test_value_type_list",
    "TestOpaqueObject::test_value_type_nested",
    "TestOpaqueObject::test_value_type_subclass",
    # test_out_dtype_op.py
    "TestOutDtypeOp::test_out_dtype_dynamo",
]
