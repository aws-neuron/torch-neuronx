"""Test RMS normalization operations not covered by PyTorch tests."""

import pytest
import torch
import torch.nn.functional as func

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

# Check if PyTorch version supports _fused_rms_norm (2.9+)
TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
HAS_FUSED_RMS_NORM = TORCH_VERSION >= (2, 9)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestRMSNormOperations:
    """Test RMS norm cases not covered by PyTorch's test_rmsnorm_numeric."""

    def setup_method(self):
        torch.manual_seed(42)

    @pytest.mark.parametrize("normalized_shape", [(8,), (4, 8)])
    def test_rms_norm_multi_dim_normalization(self, normalized_shape):
        """Test single vs multi-dimensional normalization."""
        input_cpu = torch.randn(2, 4, 8)
        input_neuron = input_cpu.to("neuron")
        weight_cpu = torch.randn(normalized_shape)
        weight_neuron = weight_cpu.to("neuron")

        with track_neuron_ops():
            output_cpu = func.rms_norm(input_cpu, normalized_shape, weight_cpu, 0.5)
            output_neuron = func.rms_norm(input_neuron, normalized_shape, weight_neuron, 0.5)
            if HAS_FUSED_RMS_NORM:
                assert_op_runs_on_neuron("aten::_fused_rms_norm")
        torch.testing.assert_close(output_neuron.cpu(), output_cpu, rtol=1e-4, atol=1e-4)

    def test_rms_norm_without_weight(self):
        """Test without weight parameter."""
        input_cpu = torch.randn(4, 8)
        input_neuron = input_cpu.to("neuron")

        with track_neuron_ops():
            output_cpu = func.rms_norm(input_cpu, (8,), None, 0.5)
            output_neuron = func.rms_norm(input_neuron, (8,), None, 0.5)
            if HAS_FUSED_RMS_NORM:
                assert_op_runs_on_neuron("aten::_fused_rms_norm")

        torch.testing.assert_close(output_neuron.cpu(), output_cpu, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("eps", [1e-8, 1e-3])
    def test_rms_norm_epsilon_values(self, eps):
        """Test extreme epsilon values."""
        input_cpu = torch.randn(4, 8)
        input_neuron = input_cpu.to("neuron")
        weight_cpu = torch.randn(8)
        weight_neuron = weight_cpu.to("neuron")

        with track_neuron_ops():
            output_cpu = func.rms_norm(input_cpu, (8,), weight_cpu, eps)
            output_neuron = func.rms_norm(input_neuron, (8,), weight_neuron, eps)
            if HAS_FUSED_RMS_NORM:
                assert_op_runs_on_neuron("aten::_fused_rms_norm")

        torch.testing.assert_close(output_neuron.cpu(), output_cpu, rtol=1e-4, atol=1e-4)

    def test_rms_norm_backward(self):
        """Test backward pass."""
        input_cpu = torch.randn(4, 8, requires_grad=True)
        input_neuron = input_cpu.detach().clone().to("neuron").requires_grad_(True)
        weight_cpu = torch.randn(8, requires_grad=True)
        weight_neuron = weight_cpu.detach().clone().to("neuron").requires_grad_(True)
        with track_neuron_ops():
            output_cpu = func.rms_norm(input_cpu, (8,), weight_cpu, 1e-5)
            output_neuron = func.rms_norm(input_neuron, (8,), weight_neuron, 1e-5)
            if HAS_FUSED_RMS_NORM:
                assert_op_runs_on_neuron("aten::_fused_rms_norm")
            output_cpu.sum().backward()
            output_neuron.sum().backward()
            if HAS_FUSED_RMS_NORM:
                assert_op_runs_on_neuron("aten::_fused_rms_norm_backward")
        torch.testing.assert_close(input_neuron.grad.cpu(), input_cpu.grad, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(weight_neuron.grad.cpu(), weight_cpu.grad, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize(
        "test_case",
        [
            "both_grads",  # Both input and weight need gradients
            "input_grad_only",  # Only input needs gradients
            "weight_grad_only",  # Only weight needs gradients
        ],
    )
    def test_rms_norm_backward_gradient_scenarios(self, test_case):
        """Test backward pass with different gradient requirements."""
        input_cpu = torch.randn(
            4, 8, requires_grad=(test_case in ["both_grads", "input_grad_only"])
        )
        input_neuron = (
            input_cpu.detach().clone().to("neuron").requires_grad_(input_cpu.requires_grad)
        )

        weight_cpu = torch.randn(8, requires_grad=(test_case in ["both_grads", "weight_grad_only"]))
        weight_neuron = (
            weight_cpu.detach().clone().to("neuron").requires_grad_(weight_cpu.requires_grad)
        )

        with track_neuron_ops():
            output_cpu = func.rms_norm(input_cpu, (8,), weight_cpu, 1e-5)
            output_neuron = func.rms_norm(input_neuron, (8,), weight_neuron, 1e-5)

            output_cpu.sum().backward()
            output_neuron.sum().backward()

            if HAS_FUSED_RMS_NORM:
                assert_op_runs_on_neuron("aten::_fused_rms_norm_backward")

        # Check gradients based on requires_grad settings
        if input_cpu.requires_grad:
            assert input_cpu.grad is not None
            assert input_neuron.grad is not None
            torch.testing.assert_close(
                input_neuron.grad.cpu(), input_cpu.grad, rtol=1e-3, atol=1e-3
            )
        else:
            assert input_cpu.grad is None
            assert input_neuron.grad is None

        if weight_cpu.requires_grad:
            assert weight_cpu.grad is not None
            assert weight_neuron.grad is not None
            torch.testing.assert_close(
                weight_neuron.grad.cpu(), weight_cpu.grad, rtol=1e-3, atol=1e-3
            )
        else:
            assert weight_cpu.grad is None
            assert weight_neuron.grad is None

    @pytest.mark.parametrize(
        "input_dtype,weight_dtype",
        [
            (torch.float32, torch.float32),
            (torch.bfloat16, torch.float32),
            (torch.float16, torch.float32),
            (torch.bfloat16, torch.bfloat16),
            (torch.float16, torch.float16),
        ],
    )
    def test_rms_norm_dtype_combinations_forward(self, input_dtype, weight_dtype):
        """Test different dtype combinations for input and weight."""
        input_cpu = torch.randn(4, 8, dtype=input_dtype)
        input_neuron = input_cpu.to("neuron")
        weight_cpu = torch.randn(8, dtype=weight_dtype)
        weight_neuron = weight_cpu.to("neuron")

        with track_neuron_ops():
            output_cpu = func.rms_norm(input_cpu, (8,), weight_cpu, 1e-5)
            output_neuron = func.rms_norm(input_neuron, (8,), weight_neuron, 1e-5)
            if HAS_FUSED_RMS_NORM:
                assert_op_runs_on_neuron("aten::_fused_rms_norm")

        torch.testing.assert_close(output_neuron.cpu(), output_cpu, rtol=1.3e-6, atol=1e-5)

    @pytest.mark.parametrize(
        "input_dtype,weight_dtype",
        [
            (torch.float32, torch.float32),
            (torch.bfloat16, torch.float32),
            (torch.float16, torch.float32),
            (torch.bfloat16, torch.bfloat16),
            (torch.float16, torch.float16),
        ],
    )
    def test_rms_norm_dtype_combinations_backward(self, input_dtype, weight_dtype):
        """Test different dtype combinations for backward pass."""
        input_cpu = torch.randn(4, 8, dtype=input_dtype, requires_grad=True)
        weight_cpu = torch.randn(8, dtype=weight_dtype, requires_grad=True)
        input_neuron = input_cpu.detach().clone().to("neuron").requires_grad_(True)
        weight_neuron = weight_cpu.detach().clone().to("neuron").requires_grad_(True)

        with track_neuron_ops():
            output_cpu = func.rms_norm(input_cpu, (8,), weight_cpu, 1e-5)
            output_neuron = func.rms_norm(input_neuron, (8,), weight_neuron, 1e-5)
            if HAS_FUSED_RMS_NORM:
                assert_op_runs_on_neuron("aten::_fused_rms_norm")
            torch.testing.assert_close(output_neuron.cpu(), output_cpu, rtol=1.3e-6, atol=1e-5)
            output_cpu.sum().backward()
            output_neuron.sum().backward()
            if HAS_FUSED_RMS_NORM:
                assert_op_runs_on_neuron("aten::_fused_rms_norm_backward")

        torch.testing.assert_close(input_neuron.grad.cpu(), input_cpu.grad, rtol=1.3e-6, atol=1e-5)
        torch.testing.assert_close(
            weight_neuron.grad.cpu(), weight_cpu.grad, rtol=1.3e-6, atol=1e-5
        )
