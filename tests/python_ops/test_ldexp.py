import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    track_neuron_ops,
)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestLdexp:
    """Test cases for ldexp (input * 2 ** other) operation."""

    @pytest.mark.parametrize(
        "input_data,other_data",
        [
            pytest.param([1.0, 2.0, 3.0], [0, 1, 2], id="basic_1d"),
            pytest.param([[1.0, 2.0], [3.0, 4.0]], [[1, 2], [3, 0]], id="basic_2d"),
            pytest.param([1.0, 2.0, 3.0], [-1, -2, -3], id="negative_exponents"),
            pytest.param([1.5, 2.5, 3.5], [0, 0, 0], id="zero_exponents"),
        ],
    )
    def test_ldexp_runs_on_neuron(self, input_data, other_data):
        """Test that ldexp runs on Neuron without CPU fallback."""
        input_cpu = torch.tensor(input_data)
        other_cpu = torch.tensor(other_data)
        expected = torch.ldexp(input_cpu, other_cpu)

        with track_neuron_ops():
            input_neuron = torch.tensor(input_data, device="neuron")
            other_neuron = torch.tensor(other_data, device="neuron")
            result = torch.ldexp(input_neuron, other_neuron)

            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron("aten::ldexp")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_ldexp_different_dtypes(self, dtype):
        """Test ldexp with different data types."""
        input_cpu = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
        other_cpu = torch.tensor([1, 2, 3])
        expected = torch.ldexp(input_cpu, other_cpu)

        with track_neuron_ops():
            input_neuron = input_cpu.to("neuron")
            other_neuron = other_cpu.to("neuron")
            result = torch.ldexp(input_neuron, other_neuron)

            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron("aten::ldexp")

    def test_ldexp_empty_tensor(self):
        """Test ldexp with empty tensors."""
        input_cpu = torch.empty(0)
        other_cpu = torch.empty(0, dtype=torch.int64)
        expected = torch.ldexp(input_cpu, other_cpu)

        with track_neuron_ops():
            input_neuron = torch.empty(0, device="neuron")
            other_neuron = torch.empty(0, dtype=torch.int64, device="neuron")
            result = torch.ldexp(input_neuron, other_neuron)

            assert result.shape == expected.shape
            assert_op_runs_on_neuron("aten::ldexp")
