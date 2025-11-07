"""Test suite for TorchNeuron error handling and propagation mechanisms."""

import html
import os
import re

import pytest
import torch

import torch_neuron


# TODO - re-eanble once async execution is merged
@pytest.mark.xfail(reason="Will fail until all changes related to async exec are merged")
class TestErrorHandler:
    """
    Tests for TorchNeuron's error handling pipeline.

    Validates that compilation errors in the async execution pipeline are properly
    caught, formatted, and propagated to user code with meaningful error messages.
    """

    def setup_method(self):
        """Configure environment to trigger compilation errors for testing."""
        os.environ["TORCH_NEURON_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS"] = "1"
        # Inject invalid --verbose flag to force neuronxcc compilation failure
        # This triggers the ErrorHandler pathway we want to test
        incorrect_argument = "--verbose not_a_valid_arg"
        os.environ["NEURON_CC_FLAGS"] = incorrect_argument

    def teardown_method(self):
        """Reset environment variables to avoid test interference."""
        os.environ.pop("TORCH_NEURON_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS", None)
        os.environ.pop("NEURON_CC_FLAGS", None)

    def test_compilation_error_handling(self):
        """
        Verify ErrorHandler correctly propagates neuronxcc compilation failures.

        Tests that when neuronxcc fails due to invalid compiler flags, the error
        is properly caught by ErrorHandler and formatted with stack trace information
        before being raised to user code.
        """
        with pytest.raises(Exception) as exc_info:
            default_stream = torch_neuron.default_stream()
            assert default_stream.stream_id == 0
            assert default_stream.device_index == 0

            dt0 = torch.randn(2048, 1024).to(device="neuron")
            dt1 = torch.randn(1024, 2048).to(device="neuron")
            # Trigger compilation with invalid flags - this should fail
            _ = torch.mm(dt0, dt1)
            torch.neuron.synchronize()

        # Verify error message contains expected neuronxcc failure signature
        error_message = html.unescape(str(exc_info.value))
        pattern = (
            r"COMPILATION FAILED: neuronxcc\.logging\.Assert\.NeuronAssertionError:"
            r".*Unknown verbosity level.*python stack trace="
        )
        assert re.search(
            pattern, error_message, re.DOTALL
        ), f"Expected error pattern not found in: {error_message}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
