import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


class TestPow:
    def test_pow_runs_on_neuron(self):
        """Test that pow runs on Neuron without CPU fallback"""
        with track_neuron_ops():
            a = torch.tensor([2.0, 3.0, 4.0], device="neuron")
            b = torch.tensor([2.0, 2.0, 2.0], device="neuron")
            result = torch.pow(a, b)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::pow")

    def test_pow_tensor_scalar(self):
        """Test pow with tensor input and scalar exponent"""
        input_tensor = torch.tensor([2.0, 3.0, 4.0, 5.0], device="neuron")

        result = torch.pow(input_tensor, 2)
        expected = torch.tensor([4.0, 9.0, 16.0, 25.0], device="neuron")

        torch.testing.assert_close(result, expected)

    def test_pow_tensor_scalar_out(self):
        """Test pow with tensor input and scalar exponent with preallocated out"""
        input_tensor = torch.tensor([2.0, 3.0, 4.0, 5.0], device="neuron")
        output = torch.empty_like(input_tensor)

        torch.pow(input_tensor, 2, out=output)
        expected = torch.tensor([4.0, 9.0, 16.0, 25.0], device="neuron")

        torch.testing.assert_close(output, expected)

    def test_pow_tensor_scalar_inplace(self):
        """Test in-place pow operation"""
        tensor = torch.tensor([2.0, 4.0, 8.0], device="neuron")
        expected = torch.tensor([4.0, 16.0, 64.0], device="neuron")

        tensor.pow_(2)

        torch.testing.assert_close(tensor, expected)

    def test_pow_tensor_tensor(self):
        """Test pow with tensor input and tensor exponent"""
        input_tensor = torch.tensor([2.0, 3.0, 4.0, 5.0], device="neuron")
        exponent = torch.tensor([2.0, 2.0, 3.0, 3.0], device="neuron")

        result = torch.pow(input_tensor, exponent)
        expected = torch.tensor([4.0, 9.0, 64.0, 125.0], device="neuron")

        torch.testing.assert_close(result, expected)

    def test_pow_tensor_tensor_out(self):
        """Test pow with tensor input and scalar exponent with preallocated out"""
        input_tensor = torch.tensor([2.0, 3.0, 4.0, 5.0], device="neuron")
        exponent = torch.tensor([2.0, 2.0, 3.0, 3.0], device="neuron")
        output = torch.empty_like(input_tensor)

        torch.pow(input_tensor, exponent, out=output)
        expected = torch.tensor([4.0, 9.0, 64.0, 125.0], device="neuron")

        torch.testing.assert_close(output, expected)

    def test_pow_tensor_tensor_inplace(self):
        """Test in-place pow operation"""
        tensor = torch.tensor([2.0, 4.0, 8.0], device="neuron")
        exponent = torch.tensor([3.0, 2.0, 1.0], device="neuron")
        expected = torch.tensor([8.0, 16.0, 8.0], device="neuron")

        tensor.pow_(exponent)

        torch.testing.assert_close(tensor, expected)

    def test_pow_scalar_tensor(self):
        """Test in-place pow operation"""
        input = 2
        exponent = torch.tensor([3.0, 2.0, 1.0], device="neuron")
        expected = torch.tensor([8.0, 4.0, 2.0], device="neuron")

        output = torch.pow(input, exponent)

        torch.testing.assert_close(output, expected)
