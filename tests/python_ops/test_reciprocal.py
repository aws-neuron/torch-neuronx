import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestReciprocalRegistration:
    """Test reciprocal operation registration and functionality."""

    def test_reciprocal_runs_on_neuron(self):
        """Test that reciprocal runs on Neuron"""
        with track_neuron_ops():
            input_tensor = torch.tensor([1.0, 2.0, 4.0, 0.5]).to("neuron")
            result = torch.reciprocal(input_tensor)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::reciprocal")

    @pytest.mark.parametrize(
        "shape",
        [
            (0,),
            (1,),
            (2, 2),
            (2, 1, 2),
            (1, 2, 1, 2),
            (0, 5),
            (3, 0),
            (0, 0),
        ],
    )
    def test_reciprocal_different_shapes(self, shape):
        """Test reciprocal operator with different tensor shapes"""
        with track_neuron_ops():
            input_tensor = torch.randn(shape).abs() + 0.1
            input_tensor_device = input_tensor.to("neuron")

            result = torch.reciprocal(input_tensor_device)
            expected = torch.reciprocal(input_tensor)

            assert result.shape == shape
            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron("aten::reciprocal")

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            None,
        ],
    )
    def test_reciprocal_different_dtypes(self, dtype):
        """Test reciprocal operator with different data types."""
        with track_neuron_ops():
            input_tensor = torch.tensor([1.0, 2.0, 4.0, 0.5], dtype=dtype)
            input_tensor_device = input_tensor.to("neuron")

            result = torch.reciprocal(input_tensor_device)
            expected = torch.reciprocal(input_tensor)

            assert result.device.type == "neuron"
            if dtype is None:
                dtype = torch.get_default_dtype()
            assert result.dtype == dtype
            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron("aten::reciprocal")

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            (1.0, 1.0),
            (2.0, 0.5),
            (0.25, 4.0),
            (-1.0, -1.0),
            (-0.5, -2.0),
            (0.0, float("inf")),
            (float("inf"), 0.0),
            (1e-6, 1e6),
            (1e6, 1e-6),
        ],
    )
    def test_reciprocal_specific_values(self, input_val, expected):
        """Test specific reciprocal values including edge cases."""
        with track_neuron_ops():
            input_tensor = torch.tensor([input_val])
            input_tensor_device = input_tensor.to("neuron")
            result = torch.reciprocal(input_tensor_device)

            if torch.isinf(torch.tensor(expected)):
                assert torch.isinf(result.cpu()).item()
            elif expected == 0.0:
                assert torch.isclose(result.cpu(), torch.tensor([0.0])).item()
            else:
                assert torch.isclose(result.cpu(), torch.tensor([expected])).item()
            assert_op_runs_on_neuron("aten::reciprocal")

    def test_reciprocal_backward(self):
        """Test reciprocal backward pass (gradient computation)."""
        with track_neuron_ops():
            x = torch.tensor([1.0, 2.0, 4.0], requires_grad=True)
            x_neuron = x.detach().clone().to("neuron")
            x_neuron.requires_grad = True
            y = torch.reciprocal(x).sum()
            y_neuron = torch.reciprocal(x_neuron).sum()
            y.backward()
            y_neuron.backward()
            torch.testing.assert_close(x_neuron.grad.cpu(), x.grad)
            assert_op_runs_on_neuron("aten::reciprocal")

    def test_reciprocal_special_values(self):
        """Test reciprocal with special floating point values."""
        with track_neuron_ops():
            special_values = torch.tensor([float("nan"), float("inf"), float("-inf")])
            special_device = special_values.to("neuron")
            result = torch.reciprocal(special_device)
            expected = torch.reciprocal(special_values)

            # Check nan handling
            assert torch.isnan(result.cpu()[0]).item()
            assert torch.isnan(expected[0]).item()

            # Check inf handling
            assert result.cpu()[1].item() == 0.0
            assert result.cpu()[2].item() == 0.0
            assert_op_runs_on_neuron("aten::reciprocal")
