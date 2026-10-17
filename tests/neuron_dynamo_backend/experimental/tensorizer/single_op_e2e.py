"""End-to-end tests for tensorizer backend single operations."""

import os

import pytest
import torch

# Check if neuron device is available
assert torch.neuron.device_count() > 0, "No neuron devices were discovered."


def _to_neuron(*tensors):
    """Move tensors to neuron device."""
    return tuple(t.to(torch.neuron.current_device()) for t in tensors)


# Single op test cases: (op_function, input_generator)
# Using lambdas for simple operations instead of full Module classes
SINGLE_OP_TESTS = [
    # Arithmetic ops
    pytest.param(lambda x, y: x + y, lambda: (torch.randn(4, 4), torch.randn(4, 4)), id="add"),
    pytest.param(lambda x, y: x - y, lambda: (torch.randn(4, 4), torch.randn(4, 4)), id="subtract"),
    pytest.param(lambda x, y: x * y, lambda: (torch.randn(4, 4), torch.randn(4, 4)), id="multiply"),
    pytest.param(
        lambda x, y: x / y, lambda: (torch.randn(4, 4), torch.randn(4, 4) + 1.0), id="divide"
    ),
    pytest.param(lambda x: -x, lambda: (torch.randn(4, 4),), id="negate"),
    pytest.param(lambda x: x**2, lambda: (torch.randn(4, 4),), id="power"),
    # Comparison ops
    pytest.param(lambda x, y: x > y, lambda: (torch.randn(4, 4), torch.randn(4, 4)), id="compare"),
    pytest.param(
        lambda x, y: x & y,
        lambda: (torch.randn(4, 4) > 0, torch.randn(4, 4) > 0),
        id="and",
    ),
    # Shape ops
    pytest.param(lambda x: x.expand(4, 4), lambda: (torch.randn(1, 4),), id="broadcast"),
    pytest.param(
        lambda x, y: torch.cat([x, y], dim=0),
        lambda: (torch.randn(2, 4), torch.randn(2, 4)),
        id="concatenate",
    ),
    pytest.param(lambda x: x.reshape(2, 8), lambda: (torch.randn(4, 4),), id="reshape"),
    pytest.param(lambda x: x.transpose(0, 1), lambda: (torch.randn(4, 8),), id="transpose"),
    pytest.param(lambda x: x[1:3, 0:2], lambda: (torch.randn(4, 4),), id="slice"),
    # Type conversion
    pytest.param(
        lambda x: x.to(torch.float32),
        lambda: (torch.randn(4, 4, dtype=torch.float64),),
        id="convert",
    ),
    # Linear algebra ops
    pytest.param(
        lambda x, y: torch.dot(x, y), lambda: (torch.randn(16), torch.randn(16)), id="dot"
    ),
    pytest.param(lambda x, y: x @ y, lambda: (torch.randn(4, 4), torch.randn(4, 4)), id="matmul"),
    # Elementwise math ops
    pytest.param(lambda x: torch.exp(x), lambda: (torch.randn(4, 4),), id="exp"),
    pytest.param(lambda x: torch.log(x), lambda: (torch.rand(4, 4) + 0.1,), id="log"),
    pytest.param(lambda x: torch.log1p(x), lambda: (torch.rand(4, 4) + 0.1,), id="log1p"),
    pytest.param(lambda x: torch.sqrt(x), lambda: (torch.rand(4, 4) + 0.1,), id="sqrt"),
    pytest.param(lambda x: torch.rsqrt(x), lambda: (torch.rand(4, 4) + 0.1,), id="rsqrt"),
    pytest.param(lambda x: torch.cos(x), lambda: (torch.randn(4, 4),), id="cos"),
    pytest.param(lambda x: torch.sin(x), lambda: (torch.randn(4, 4),), id="sin"),
    pytest.param(lambda x: torch.tanh(x), lambda: (torch.randn(4, 4),), id="tanh"),
    pytest.param(lambda x: torch.sigmoid(x), lambda: (torch.randn(4, 4),), id="sigmoid"),
    pytest.param(
        lambda x, y: torch.maximum(x, y),
        lambda: (torch.randn(4, 4), torch.randn(4, 4)),
        id="maximum",
    ),
    pytest.param(
        lambda x, idx, src: x.scatter(1, idx, src),
        lambda: (torch.randn(4, 4), torch.randint(0, 4, (4, 2)), torch.randn(4, 2)),
        id="scatter",
    ),
    # Iota-like op
    pytest.param(
        lambda x: x + torch.arange(4, device=x.device, dtype=x.dtype),
        lambda: (torch.randn(4),),
        id="iota",
    ),
    # Select/where
    pytest.param(
        lambda cond, x, y: torch.where(cond, x, y),
        lambda: (torch.randn(4, 4) > 0, torch.randn(4, 4), torch.randn(4, 4)),
        id="select",
    ),
    # Pooling ops
    pytest.param(
        lambda x: torch.nn.functional.max_pool2d(x, kernel_size=2),
        lambda: (torch.randn(1, 1, 4, 4),),
        id="max_pool2d",
    ),
    # Conv op
    pytest.param(
        lambda x, w: torch.nn.functional.conv2d(x, w),
        lambda: (torch.randn(1, 1, 8, 8), torch.randn(1, 1, 3, 3)),
        id="conv2d",
    ),
]


class TestTensorizerSingleOpE2E:
    """End-to-end tests for tensorizer backend with single operations."""

    @pytest.mark.skipif(
        os.environ.get("RUN_TENSORIZER_TESTS") != "1",
        reason="Tensorizer backend test - set RUN_TENSORIZER_TESTS=1 to run",
    )
    @pytest.mark.parametrize("op_fn,input_gen", SINGLE_OP_TESTS)
    def test_single_op(self, op_fn, input_gen):
        """Test single operation with tensorizer backend."""
        torch.manual_seed(42)

        # Generate inputs on CPU for reference, then move to neuron device
        cpu_inputs = input_gen()
        neuron_inputs = _to_neuron(*cpu_inputs)

        with torch.inference_mode():
            # Reference output on CPU
            reference_output = op_fn(*cpu_inputs)

            # Compile with tensorizer backend
            compiled_fn = torch.compile(
                op_fn, backend="neuron", options={"use_tensorizer_backend": True}
            )
            compiled_output = compiled_fn(*neuron_inputs)

        # Verify output is on neuron device
        assert compiled_output.device.type == "neuron"
        assert compiled_output.shape == reference_output.shape

        # Compare results
        if reference_output.dtype == torch.bool:
            assert torch.equal(compiled_output.cpu(), reference_output)
        else:
            torch.testing.assert_close(
                compiled_output.cpu(), reference_output, rtol=1e-4, atol=1e-4
            )
