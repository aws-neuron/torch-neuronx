"""Test suite for TorchNeuronx error handling and propagation mechanisms."""

import html
import os
import re

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_raises


@pytest.mark.skipif(
    os.environ.get("NEURON_LAUNCH_BLOCKING") == "1",
    reason="Error Handler is used only during async mode",
)
class TestErrorHandler:
    """
    Tests for TorchNeuronx's error handling pipeline.

    Validates that compilation errors in the async execution pipeline are properly
    caught, formatted, and propagated to user code with meaningful error messages.
    """

    def setup_method(self):
        """Configure environment to trigger compilation errors for testing."""
        os.environ["TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS"] = "1"
        # Inject invalid --verbose flag to force neuronxcc compilation failure
        # This triggers the ErrorHandler pathway we want to test
        incorrect_argument = "--verbose not_a_valid_arg"
        os.environ["NEURON_CC_FLAGS"] = incorrect_argument

    def teardown_method(self):
        """Reset environment variables to avoid test interference."""
        os.environ.pop("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS", None)
        os.environ.pop("NEURON_CC_FLAGS", None)

    @assert_raises(
        (IndexError, RuntimeError),
        match=(
            re.compile(
                r"COMPILATION FAILED: neuronxcc\.logging\.Assert\.NeuronAssertionError:"
                r".*Unknown verbosity level.*python stack trace=",
                flags=re.DOTALL,
            )
        ),
    )
    def test_compilation_error_handling(self):
        """
        Verify ErrorHandler correctly propagates neuronxcc compilation failures.

        Tests that when neuronxcc fails due to invalid compiler flags, the error
        is properly caught by ErrorHandler and formatted with stack trace information
        before being raised to user code.
        """
        default_stream = torch_neuronx.default_stream()
        assert default_stream.stream_id == 0
        assert default_stream.device_index == 0

        dt0 = torch.randn(2048, 1024).to(device="neuron")
        dt1 = torch.randn(1024, 2048).to(device="neuron")
        # Trigger compilation with invalid flags - this should fail
        _ = torch.mm(dt0, dt1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
