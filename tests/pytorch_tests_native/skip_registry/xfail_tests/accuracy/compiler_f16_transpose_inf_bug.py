"""Compiler bug: stablehlo.transpose on f16 corrupts inf values to NaN.

These tests fail because the neuron compiler's transpose lowering corrupts
inf values in float16 tensors, producing NaN instead.
"""

COMPILER_F16_TRANSPOSE_INF_BUG = [
    "test_binary_ufuncs.py::TestBinaryUfuncsNEURON::test_contig_vs_transposed__refs_true_divide_neuron_float16",
    "test_binary_ufuncs.py::TestBinaryUfuncsNEURON::test_contig_vs_transposed__refs_div_floor_rounding_neuron_float16",
    "test_binary_ufuncs.py::TestBinaryUfuncsNEURON::test_contig_vs_transposed__refs_div_no_rounding_mode_neuron_float16",
    "test_binary_ufuncs.py::TestBinaryUfuncsNEURON::test_contig_vs_transposed__refs_div_trunc_rounding_neuron_float16",
    "test_binary_ufuncs.py::TestBinaryUfuncsNEURON::test_contig_vs_transposed__refs_pow_neuron_float16",
    "test_ops.py::TestCommonNEURON::test_python_ref_torch_fallback__refs_div_no_rounding_mode_neuron_float16",
    "test_ops.py::TestCommonNEURON::test_python_ref_torch_fallback__refs_true_divide_neuron_float16",
    "test_ops.py::TestCommonNEURON::test_python_ref_torch_fallback__refs_pow_neuron_float16",
]
