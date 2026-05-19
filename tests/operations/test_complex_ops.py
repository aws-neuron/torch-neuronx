"""Tests for complex ops that run natively on Neuron via the complex allowlist."""

import torch

import torch_neuronx


class TestComplexOpsOnNeuron:
    """Verify complex ops on the allowlist compile and run on Neuron hardware."""

    def test_complex_mul_tensor(self):
        """mul.Tensor: element-wise complex * complex"""
        x_cpu = torch.randn(4, 4, dtype=torch.complex64)
        y_cpu = torch.randn(4, 4, dtype=torch.complex64)
        result = x_cpu.to("neuron") * y_cpu.to("neuron")
        assert result.dtype == torch.complex64
        torch.testing.assert_close(result.cpu(), x_cpu * y_cpu, atol=1e-5, rtol=1e-4)

    def test_complex_mul_scalar(self):
        """mul.Scalar: complex tensor * real scalar"""
        x_cpu = torch.randn(4, 4, dtype=torch.complex64)
        scalar = 2.5
        result = x_cpu.to("neuron") * scalar
        assert result.dtype == torch.complex64
        torch.testing.assert_close(result.cpu(), x_cpu * scalar, atol=1e-5, rtol=1e-4)

    def test_complex_mul_inplace(self):
        """mul_: in-place complex multiplication"""
        x_cpu = torch.randn(4, 4, dtype=torch.complex64)
        y_cpu = torch.randn(4, 4, dtype=torch.complex64)
        x_neuron = x_cpu.clone().to("neuron")
        x_neuron.mul_(y_cpu.to("neuron"))
        torch.testing.assert_close(x_neuron.cpu(), x_cpu * y_cpu, atol=1e-5, rtol=1e-4)

    def test_complex_mul_broadcast(self):
        """mul.Tensor with broadcasting"""
        x_cpu = torch.randn(4, 4, dtype=torch.complex64)
        y_cpu = torch.randn(1, 4, dtype=torch.complex64)
        result = x_cpu.to("neuron") * y_cpu.to("neuron")
        assert result.dtype == torch.complex64
        torch.testing.assert_close(result.cpu(), x_cpu * y_cpu, atol=1e-5, rtol=1e-4)

    def test_complex_div_tensor(self):
        """div.Tensor: element-wise complex / complex"""
        x_cpu = torch.randn(4, 4, dtype=torch.complex64)
        y_cpu = torch.randn(4, 4, dtype=torch.complex64) + 0.1  # avoid div by zero
        result = x_cpu.to("neuron") / y_cpu.to("neuron")
        assert result.dtype == torch.complex64
        torch.testing.assert_close(result.cpu(), x_cpu / y_cpu, atol=1e-5, rtol=1e-4)

    def test_complex_div_scalar(self):
        """div by real scalar"""
        x_cpu = torch.randn(4, 4, dtype=torch.complex64)
        result = x_cpu.to("neuron") / 2.0
        assert result.dtype == torch.complex64
        torch.testing.assert_close(result.cpu(), x_cpu / 2.0, atol=1e-5, rtol=1e-4)

    def test_complex_div_inplace(self):
        """div_.Tensor: in-place complex division"""
        x_cpu = torch.randn(4, 4, dtype=torch.complex64)
        y_cpu = torch.randn(4, 4, dtype=torch.complex64) + 0.1
        x_neuron = x_cpu.clone().to("neuron")
        x_neuron.div_(y_cpu.to("neuron"))
        torch.testing.assert_close(x_neuron.cpu(), x_cpu / y_cpu, atol=1e-5, rtol=1e-4)

    def test_complex_div_broadcast(self):
        """div.Tensor with broadcasting"""
        x_cpu = torch.randn(4, 4, dtype=torch.complex64)
        y_cpu = torch.randn(1, 4, dtype=torch.complex64) + 0.1
        result = x_cpu.to("neuron") / y_cpu.to("neuron")
        assert result.dtype == torch.complex64
        torch.testing.assert_close(result.cpu(), x_cpu / y_cpu, atol=1e-5, rtol=1e-4)

    def test_complex_clone(self):
        x_cpu = torch.randn(4, 4, dtype=torch.complex64)
        x_neuron = x_cpu.to("neuron")
        result = x_neuron.clone()
        assert result.dtype == torch.complex64
        assert result.data_ptr() != x_neuron.data_ptr()
        torch.testing.assert_close(result.cpu(), x_cpu)

    def test_complex_clone_detach(self):
        """clone().detach() is a common pattern in test frameworks and autograd."""
        x_cpu = torch.randn(4, 4, dtype=torch.complex64)
        x_neuron = x_cpu.to("neuron")
        result = x_neuron.clone().detach()
        assert result.dtype == torch.complex64
        assert not result.requires_grad
        torch.testing.assert_close(result.cpu(), x_cpu)
