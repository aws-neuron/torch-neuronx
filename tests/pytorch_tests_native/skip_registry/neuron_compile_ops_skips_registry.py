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
