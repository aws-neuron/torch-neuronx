DTYPE_MISMATCH_FLOAT64_DOWNCAST = [
    "test_binary_ufuncs.py::TestBinaryUfuncs*NEURON::test_divmul_scalar_neuron_float16",
    "test_decomp.py::TestDecompNEURON::test_comprehensive__native_batch_norm_legit_neuron_float64",
    # baddbmm joins its matmul-accumulate siblings addmm/addmv/addcmul above,
    # for the identical root cause. test_decomp's DecompCrossRefMode runs the
    # decomposition against a float64 reference and asserts every intermediate
    # keeps its dtype; the `beta * self` multiply comes back float32 because
    # Neuron down-casts 64-bit types on entry (docs/packed_int64_design.md,
    # "Neuron computes on 32-bit types"). Messages measured:
    #   bfloat16/float16: "0 Operation: aten.mul.Tensor dtype mismatch:
    #                      torch.float64 != torch.float32"
    #   float32:          "Object comparison failed: torch.float64 !=
    #                      torch.float32"
    # test_comprehensive_addmm_neuron_float32 above fails with that same
    # "Object comparison failed" message and is filed here, so the grouping is
    # by root cause, not by message wording.
    "test_decomp.py::TestDecompNEURON::test_comprehensive_nn_functional_nll_loss_neuron_float64",
    "test_decomp.py::TestDecompNEURON::test_quick__native_batch_norm_legit_neuron_float64",
    "test_foreach.py::TestForeachNEURON::test_parity__foreach_sqrt_fastpath_outplace_neuron_float64",
    "test_foreach.py::TestForeachNEURON::test_parity__foreach_sqrt_slowpath_outplace_neuron_float64",
    "test_ops.py::TestCommonNEURON::test_out__refs_allclose_neuron_float32",
    "test_ops.py::TestCommonNEURON::test_out_warning__refs_allclose_neuron",
    "test_ops.py::TestCommonNEURON::test_python_ref__refs_nn_functional_triplet_margin_loss_neuron_uint8",
    "test_type_promotion.py::TestTypePromotionNEURON::test_bfloat16_neuron",
    "test_ops.py::TestFakeTensorNEURON::test_fake_autocast_pinverse_neuron_float32",
    "test_tensor_creation_ops.py::TestRandomTensorCreationNEURON::test_uniform_from_to_neuron_float64",
    "test_decomp.py::TestDecompNEURON::test_comprehensive_nn_functional_glu_neuron_bfloat16",
    "test_decomp.py::TestDecompNEURON::test_comprehensive_nn_functional_glu_neuron_float16",
    "test_decomp.py::TestDecompNEURON::test_comprehensive_nn_functional_glu_neuron_float64",
    "test_decomp.py::TestDecompNEURON::test_comprehensive_nn_functional_mish_neuron_bfloat16",
    "test_decomp.py::TestDecompNEURON::test_comprehensive_nn_functional_mish_neuron_float16",
    "test_decomp.py::TestDecompNEURON::test_comprehensive_nn_functional_mish_neuron_float64",
    # Neuron has no float64 (docs/packed_int64_design.md). It runs these ops in
    # float32. The forward and gradient results, and the test_decomp float64
    # reference, do not match the float64 CPU value. Same limit as the entries
    # above.
    "test_decomp.py::TestDecompNEURON::test_comprehensive_nn_functional_celu_neuron_float64",
    "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad_exp2_neuron_float64",
    "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad_ldexp_neuron_float64",
    "test_ops_fwd_gradients.py::TestFwdGradientsNEURON::test_fn_fwgrad_bwgrad_logit_neuron_float64",
    "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_ldexp_neuron_float64",
    "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_exp2_neuron_float64",
    "test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_gradgrad_ldexp_neuron_float64",
    "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_large__refs_frac_neuron_float64",
    "test_unary_ufuncs.py::TestUnaryUfuncsNEURON::test_reference_numerics_large_frac_neuron_float64",
    # foreach ops: float64 inputs downcast to float32 on Neuron; parity check
    # against CPU float64 reference fails due to precision loss.
    # clip_grad_norm: the test uses a 2e22 float64 grad. After the downcast to
    # float32, (2e22)^2 overflows to inf, so the order-2 norm is non-finite and
    # clip_grad_norm_ raises. CPU keeps float64 and returns a finite 6.32e22.
    "test_nn.py::TestNNDeviceTypeNEURON::test_clip_grad_norm_error_if_nonfinite_neuron",
]
