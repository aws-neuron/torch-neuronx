"""Test suite for TorchNeuronx error handling and propagation mechanisms."""

import html
import os
import re

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

import torch_neuronx
from tests.utils.neuron_test_utils import assert_raises


@pytest.mark.skipif(
    os.environ.get("NEURON_LAUNCH_BLOCKING") == "1",
    reason="Error Handler is used only during async mode",
)
@pytest.mark.skipif(
    os.environ.get("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS") != "1",
    reason=(
        "Requires TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1 at process start "
        "(C++ reads it once at library load)"
    ),
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
                r"compilation failed"
                r".*neuronxcc\.logging\.Assert\.NeuronAssertionError:"
                r".*Unknown verbosity level",
                flags=re.DOTALL | re.IGNORECASE,
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


@pytest.mark.skipif(
    os.environ.get("NEURON_LAUNCH_BLOCKING") == "1",
    reason="Error Handler is used only during async mode",
)
@pytest.mark.skipif(
    os.environ.get("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS") != "1",
    reason=(
        "Requires TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1 at process start "
        "(C++ reads it once at library load)"
    ),
)
class TestCompilerDiagnosticReachesUser:
    """
    Guard that the compiler diagnostic reaches the user.

    A neuronx-cc failure prints the actionable "[NCC_*]" diagnostic first and
    the failing HLO instruction last. ErrorHandler.CleanErrorMessage must keep
    the diagnostic. Before the fix it kept only the last line, so the "[NCC_*]"
    code and the reason were lost.
    """

    def setup_method(self):
        # Force a loud failure instead of a silent CPU fallback.
        os.environ["TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS"] = "1"

    def teardown_method(self):
        os.environ.pop("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS", None)

    # A compiler diagnostic line carries an "[NCC_*]" or "[ERROR]" code. The
    # failing HLO instruction (the pre-fix message) has neither. So this token
    # is what proves the diagnostic reached the user. Match only the token, not
    # the reason text, because the compiler owns the wording.
    _DIAGNOSTIC_CODE = re.compile(r"\[NCC_[A-Z0-9]+\]|\[ERROR\]")

    def test_compiler_diagnostic_reaches_user(self):
        """
        A transposed conv maps stride -> input dilation and dilation -> kernel
        dilation. Both > 1 makes neuronx-cc reject the op with an "[NCC_*]"
        diagnostic. Confirm the code reaches the raised RuntimeError. See
        P324476620.

        Do not assert the compiler limitation itself. If a later compiler
        supports the op, no error occurs, so skip instead of fail. Then update
        the trigger to another op that still emits an "[NCC_*]" diagnostic.
        """
        dtype = torch.bfloat16
        inp = torch.randn(2, 2, 4, dtype=dtype, device="neuron")
        weight = torch.randn(2, 2, 4, dtype=dtype, device="neuron")
        bias = torch.randn(4, dtype=dtype, device="neuron")

        # stride=3 -> lhs_dilate, dilation=4 -> rhs_dilate. Both > 1 -> reject.
        out = F.conv_transpose1d(
            inp, weight, bias, stride=3, padding=1, output_padding=2, groups=2, dilation=4
        )
        try:
            # Move to CPU to force compilation and to surface the async error.
            _ = out.cpu()
        except RuntimeError as error:
            message = str(error)
            assert self._DIAGNOSTIC_CODE.search(message), (
                "compiler diagnostic code missing from the error; "
                f"CleanErrorMessage dropped it:\n{message}"
            )
            return
        pytest.skip(
            "neuronx-cc no longer rejects a dilated transposed conv. "
            "Replace the trigger with another op that emits an [NCC_*] diagnostic."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
