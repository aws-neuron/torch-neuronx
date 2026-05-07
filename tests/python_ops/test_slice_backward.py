import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


def slice_tensor(t, dim, start, end, step):
    """Helper to slice tensor along a dimension."""
    slices = [slice(None)] * t.dim()
    slices[dim] = slice(start, end, step)
    return t[tuple(slices)]


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestSliceBackward:
    """Test cases for slice_backward operation"""

    @pytest.mark.parametrize(
        "input_sizes,dim,start,end,step",
        [
            ((4, 4), 0, 0, 2, 1),  # Basic slice on dim 0
            ((4, 4), 1, 1, 3, 1),  # Basic slice on dim 1
            ((8,), 0, 2, 6, 1),  # 1D tensor
            ((2, 3, 4), 2, 0, 2, 1),  # 3D tensor
            ((4, 4), 0, 0, 4, 2),  # Slice with step > 1
        ],
    )
    def test_slice_backward_basic(self, input_sizes, dim, start, end, step):
        """Test slice_backward with various configurations."""
        with track_neuron_ops():
            # Create input tensor that requires grad
            x_cpu = torch.randn(input_sizes, requires_grad=True)
            x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

            # Forward slice
            sliced_cpu = slice_tensor(x_cpu, dim, start, end, step)
            sliced_neuron = slice_tensor(x_neuron, dim, start, end, step)

            # Create gradient for backward
            grad_output = torch.randn_like(sliced_cpu)
            grad_output_neuron = grad_output.to("neuron")

            # Backward
            sliced_cpu.backward(grad_output)
            sliced_neuron.backward(grad_output_neuron)

            # Compare gradients
            torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)
            assert_op_runs_on_neuron("aten::slice_backward")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_slice_backward_dtypes(self, dtype):
        """Test slice_backward with different dtypes."""
        with track_neuron_ops():
            x_cpu = torch.randn(4, 4, dtype=dtype, requires_grad=True)
            x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

            sliced_cpu = x_cpu[0:2, :]
            sliced_neuron = x_neuron[0:2, :]

            grad_output = torch.randn_like(sliced_cpu)
            grad_output_neuron = grad_output.to("neuron")

            sliced_cpu.backward(grad_output)
            sliced_neuron.backward(grad_output_neuron)

            torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-2, atol=1e-2)
            assert_op_runs_on_neuron("aten::slice_backward")

    def test_slice_backward_negative_indices(self):
        """Test slice_backward with negative indices."""
        with track_neuron_ops():
            x_cpu = torch.randn(4, 4, requires_grad=True)
            x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

            # Slice with negative end index (equivalent to end=3)
            sliced_cpu = x_cpu[:, 1:-1]
            sliced_neuron = x_neuron[:, 1:-1]

            grad_output = torch.randn_like(sliced_cpu)
            grad_output_neuron = grad_output.to("neuron")

            sliced_cpu.backward(grad_output)
            sliced_neuron.backward(grad_output_neuron)

            torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)

    def test_slice_backward_full_slice(self):
        """Test slice_backward when slicing entire dimension."""
        with track_neuron_ops():
            x_cpu = torch.randn(4, 4, requires_grad=True)
            x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

            # Full slice (no-op effectively)
            sliced_cpu = x_cpu[:, :]
            sliced_neuron = x_neuron[:, :]

            grad_output = torch.randn_like(sliced_cpu)
            grad_output_neuron = grad_output.to("neuron")

            sliced_cpu.backward(grad_output)
            sliced_neuron.backward(grad_output_neuron)

            torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)
