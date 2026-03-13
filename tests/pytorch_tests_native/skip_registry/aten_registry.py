"""ATen test skips for upstream PyTorch tests on Neuron.

Tests that crash, timeout, or segfault.
"""

from tests.pytorch_tests_native.neuron_test_registry import SkipGroup

SKIP_GROUPS: list[SkipGroup] = [
    SkipGroup(
        reason="Long compilation - causes test suite timeout",
        tests=["test_ReplicationPad2d_large_neuron"],
    ),
    SkipGroup(
        reason="Segfault in view ops - requires investigation",
        patterns=["*conj_imag_view*", "*conj_view*"],
    ),
    SkipGroup(
        reason="Long compilation/execution for conv transpose",
        tests=[
            "test_conv_transpose_with_output_size_and_no_batch_dim",
            "test_conv_transpose_with_output_size_and_no_batch_dim_ConvTranspose2d_neuron",
            "test_conv_transpose_with_output_size_and_no_batch_dim_ConvTranspose3d_neuron",
        ],
    ),
]

XFAIL_GROUPS: list[SkipGroup] = []
