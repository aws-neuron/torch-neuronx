"""PT 2.11 accuracy regression in bfloat16/float16 reciprocal-family ops.

33 tests (div, reciprocal, rsqrt, fmod, true_divide, lgamma,
floor_divide, remainder, pairwise_distance) pass on nightly but fail on
PT 2.11 with 'tensor-likes are not close' or 'Difference from float64 is
larger with decomposition'. Greatest-diff pattern is 'inf at index N' and
decomposed div.Tensor_mode produces ~3e38 where the original is finite —
suggests bf16 reciprocal kernel now saturates to inf where it didn't
before.
"""

PT211_BF16_RECIPROCAL_REGRESSION = [
    "test_binary_ufuncs.py::TestBinaryUfuncsNEURON::test_div_rounding_modes_neuron_bfloat16",
    "test_binary_ufuncs.py::TestBinaryUfuncsNEURON::test_div_rounding_modes_neuron_float16",
    "test_binary_ufuncs.py::TestBinaryUfuncsNEURON::test_non_contig_div_floor_rounding_neuron_float16",
    "test_binary_ufuncs.py::TestBinaryUfuncsNEURON::test_non_contig_div_no_rounding_mode_neuron_float16",
    "test_binary_ufuncs.py::TestBinaryUfuncsNEURON::test_non_contig_div_trunc_rounding_neuron_float16",
    "test_binary_ufuncs.py::TestBinaryUfuncsNEURON::test_non_contig_true_divide_neuron_float16",
]
