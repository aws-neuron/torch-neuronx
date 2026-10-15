# Split per test class: a single ~4700-entry file exceeds the pre-commit large-file limit.
from tests.pytorch_tests_native.neuron_test_registry import SkipGroup
from tests.pytorch_tests_native.skip_registry.neuron_compile_ops import (
    binary_ufunc,
    foreach,
    grad,
    linalg,
    nn_functional,
    op_info,
    reduction,
    unary_ufunc,
)

SKIP_GROUPS: list[SkipGroup] = [
    SkipGroup(
        reason=(
            "[platform limitation] int64 and float64 are not natively supported on Neuron "
            "(64-bit tensors run on a 32-bit datapath), so these compile-path cases exceed "
            "tolerance. Expected platform limitation, not op bugs."
        ),
        tests=[
            "test_neuron_compile_ops.py::TestCompileOpInfoNEURON::test_compile_matches_eager_as_strided_scatter_neuron_int64",
            "test_neuron_compile_ops.py::TestCompileOpInfoNEURON::test_compile_matches_eager_special_polygamma_special_polygamma_n_0_neuron_float64",
            "test_neuron_compile_ops.py::TestCompileOpInfoNEURON::test_compile_matches_eager_polygamma_polygamma_n_4_neuron_float64",
            "test_neuron_compile_ops.py::TestCompileOpInfoNEURON::test_compile_matches_eager_polygamma_polygamma_n_3_neuron_float64",
            "test_neuron_compile_ops.py::TestCompileOpInfoNEURON::test_compile_matches_eager_index_copy_neuron_int64",
            "test_neuron_compile_ops.py::TestCompileOpInfoNEURON::test_compile_matches_eager_polygamma_polygamma_n_0_neuron_float64",
            "test_neuron_compile_ops.py::TestCompileOpInfoNEURON::test_compile_matches_eager_polygamma_polygamma_n_2_neuron_float64",
            "test_neuron_compile_ops.py::TestCompileOpInfoNEURON::test_compile_matches_eager_polygamma_polygamma_n_1_neuron_float64",
            "test_neuron_compile_ops.py::TestCompileOpInfoNEURON::test_compile_matches_eager_masked_softmin_neuron_float64",
        ],
    ),
    SkipGroup(
        reason="Neuron torch.compile failure/hang in the in-tree op suite.",
        tests=[
            *binary_ufunc.SKIP,
            *foreach.SKIP,
            *grad.SKIP,
            *linalg.SKIP,
            *nn_functional.SKIP,
            *op_info.SKIP,
            *reduction.SKIP,
            *unary_ufunc.SKIP,
        ],
    ),
]
