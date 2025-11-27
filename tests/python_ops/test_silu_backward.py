import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

TEST_CONFIGS = [
    # Test different tensor shapes with normal values
    pytest.param(
        (1000,),
        "normal",
        id="1d_normal",
    ),
    pytest.param(
        (32, 128),
        "normal",
        id="2d_normal",
    ),
    pytest.param(
        (8, 64, 32),
        "normal",
        id="3d_normal",
    ),
    pytest.param(
        (4, 16, 32, 32),
        "normal",
        id="4d_normal",
    ),
    # Test with negative values
    pytest.param(
        (100,),
        "negative",
        id="1d_negative",
    ),
    pytest.param(
        (16, 64),
        "negative",
        id="2d_negative",
    ),
    pytest.param(
        (4, 32, 16),
        "negative",
        id="3d_negative",
    ),
    # Test with extreme positive values
    pytest.param(
        (50,),
        "extreme_positive",
        id="1d_extreme_positive",
    ),
    pytest.param(
        (8, 32),
        "extreme_positive",
        id="2d_extreme_positive",
    ),
    pytest.param(
        (2, 16, 8),
        "extreme_positive",
        id="3d_extreme_positive",
    ),
    # Test with extreme negative values
    pytest.param(
        (50,),
        "extreme_negative",
        id="1d_extreme_negative",
    ),
    pytest.param(
        (8, 32),
        "extreme_negative",
        id="2d_extreme_negative",
    ),
    pytest.param(
        (2, 16, 8),
        "extreme_negative",
        id="3d_extreme_negative",
    ),
    # Test with mixed positive and negative values
    pytest.param(
        (200,),
        "mixed",
        id="1d_mixed",
    ),
    pytest.param(
        (16, 32),
        "mixed",
        id="2d_mixed",
    ),
    pytest.param(
        (4, 8, 16),
        "mixed",
        id="3d_mixed",
    ),
    # Test with values close to zero
    pytest.param(
        (100,),
        "near_zero",
        id="1d_near_zero",
    ),
    pytest.param(
        (8, 16),
        "near_zero",
        id="2d_near_zero",
    ),
    # Test with zero values
    pytest.param(
        (50,),
        "zeros",
        id="1d_zeros",
    ),
    pytest.param(
        (4, 8),
        "zeros",
        id="2d_zeros",
    ),
    # Test with very large tensors and extreme values
    pytest.param(
        (10000,),
        "extreme_positive",
        id="large_1d_extreme_positive",
    ),
    pytest.param(
        (10000,),
        "extreme_negative",
        id="large_1d_extreme_negative",
    ),
]

"""
Setting loose thresholds for low precision because fp32 passes. Input distribution
is very varied so global thresholds would not work. Need a mechanism to calculate
thresholds based on the data distribution. To be included in vertical testing.
"""
DTYPE_TOLERANCE_CONFIGS = [
    pytest.param(torch.float32, 1e-5, 1.3e-6, id="float32"),
    pytest.param(torch.float16, 1e-1, 1e-2, id="float16"),
    pytest.param(torch.bfloat16, 1e-1, 1e-2, id="bfloat16"),
]


def generate_test_input(shape, value_type, dtype=torch.float32):
    """Generate test input tensor based on shape and value type"""
    if value_type == "normal":
        return torch.randn(shape, dtype=dtype, requires_grad=True)
    elif value_type == "negative":
        return -torch.abs(torch.randn(shape, dtype=dtype, requires_grad=True))
    elif value_type == "extreme_positive":
        return (
            torch.randn(shape, dtype=dtype, requires_grad=True) * 50 + 100
        )  # Values around 100 ± 50
    elif value_type == "extreme_negative":
        return -(
            torch.randn(shape, dtype=dtype, requires_grad=True) * 50 + 100
        )  # Values around -100 ± 50
    elif value_type == "mixed":
        tensor = torch.randn(shape, dtype=dtype, requires_grad=True) * 10
        # Ensure we have both positive and negative values
        mask = torch.rand(shape) > 0.5
        tensor_data = tensor.detach()  # Create a new tensor without grad to modify values
        tensor_data[mask] = torch.abs(tensor_data[mask])
        tensor_data[~mask] = -torch.abs(tensor_data[~mask])
        return torch.tensor(tensor_data, dtype=dtype, requires_grad=True)
    elif value_type == "near_zero":
        return (
            torch.randn(shape, dtype=dtype, requires_grad=True) * 0.01
        )  # Small values around zero
    elif value_type == "zeros":
        return torch.zeros(shape, dtype=dtype, requires_grad=True)
    else:
        raise ValueError(f"Unknown value_type: {value_type}")


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestSiluRegistration:
    """Test cases for silu operation"""

    @pytest.mark.parametrize("input_shape, value_type", TEST_CONFIGS)
    @pytest.mark.parametrize("dtype, atol, rtol", DTYPE_TOLERANCE_CONFIGS)
    def test_silu_run_on_neuron(self, input_shape, value_type, dtype, atol, rtol):
        """
        Test if the op runs on neuron and output matches CPU for varying
        dtypes, tensor shapes, sizes, and value ranges.
        """

        def run_silu(device):
            torch.manual_seed(0)
            x = generate_test_input(input_shape, value_type, dtype=dtype)
            x.retain_grad()

            if device == "neuron":
                x = x.to(device)
                x.retain_grad()
                # Track neuron ops only for neuron device
                with track_neuron_ops():
                    output = torch.nn.functional.silu(x)
                    grad_output = torch.randn_like(output)
                    output.backward(grad_output)
            else:
                output = torch.nn.functional.silu(x)
                grad_output = torch.randn_like(output)
                output.backward(grad_output)

            return output, x.grad

        # Run on both devices
        neuron_output, neuron_grad = run_silu("neuron")
        cpu_output, cpu_grad = run_silu("cpu")
        assert neuron_output.dtype == dtype

        torch.testing.assert_close(neuron_grad.cpu(), cpu_grad, atol=atol, rtol=rtol)
        assert_op_runs_on_neuron("aten::silu_backward")
