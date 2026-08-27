"""Skips for upstream distributed/pipelining tests on Neuron."""

from tests.pytorch_tests_native.neuron_test_registry import SkipGroup

SKIP_GROUPS: list[SkipGroup] = [
    SkipGroup(
        reason=(
            "create_block_mask() argsorts an int32 dense masks which"
            " lowers to the AwsNeuronTopK custom call. neuronx-cc rejects it:"
            " '[NCC_EVRF013] TopK custom operation does not support 32-bit and 64-bit"
            " integer types'. Every BlockMask construction on Neuron hits this, so it"
            " needs an integer upcast in the Neuron sort lowering (or TopK integer"
            " support), not a test change."
        ),
        tests=[
            "test_microbatch.py::MicrobatchTestsNEURON::test_split_block_mask_neuron",
            "test_microbatch.py::MicrobatchTestsNEURON::test_split_block_mask_batch_size_one_neuron",
        ],
    ),
]
