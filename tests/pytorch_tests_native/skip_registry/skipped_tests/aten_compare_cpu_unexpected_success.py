"""Upstream OpInfo declares @expectedFailure for test_compare_cpu on these
ops (linalg.lu family and special polynomials). Neuron computes the
result correctly so the test passes, reported as Unexpected success."""

ATEN_COMPARE_CPU_UNEXPECTED_SUCCESS = [
    "test_ops.py::TestCommonNEURON::test_compare_cpu_linalg_lu_factor_ex_neuron_float32",
    "test_ops.py::TestCommonNEURON::test_compare_cpu_linalg_lu_factor_neuron_float32",
    "test_ops.py::TestCommonNEURON::test_compare_cpu_linalg_lu_neuron_float32",
    "test_ops.py::TestCommonNEURON::test_compare_cpu_special_chebyshev_polynomial_t_neuron_float32",
    "test_ops.py::TestCommonNEURON::test_compare_cpu_special_chebyshev_polynomial_u_neuron_float32",
    "test_ops.py::TestCommonNEURON::test_compare_cpu_special_chebyshev_polynomial_v_neuron_float32",
    "test_ops.py::TestCommonNEURON::test_compare_cpu_special_chebyshev_polynomial_w_neuron_float32",
    "test_ops.py::TestCommonNEURON::test_compare_cpu_special_hermite_polynomial_h_neuron_float32",
    "test_ops.py::TestCommonNEURON::test_compare_cpu_special_hermite_polynomial_he_neuron_float32",
    "test_ops.py::TestCommonNEURON::test_compare_cpu_special_legendre_polynomial_p_neuron_float32",
    "test_ops.py::TestCommonNEURON::test_compare_cpu_special_shifted_chebyshev_polynomial_t_neuron_float32",
    "test_ops.py::TestCommonNEURON::test_compare_cpu_special_shifted_chebyshev_polynomial_u_neuron_float32",
    "test_ops.py::TestCommonNEURON::test_compare_cpu_special_shifted_chebyshev_polynomial_v_neuron_float32",
    "test_ops.py::TestCommonNEURON::test_compare_cpu_special_shifted_chebyshev_polynomial_w_neuron_float32",
    "test_nn.py::TestNNDeviceTypeNEURON::test_pdist_gradgrad_unimplemented_neuron",
]
