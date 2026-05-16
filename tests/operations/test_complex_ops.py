"""Tests for complex ops that run natively on Neuron via the complex allowlist."""

import pytest
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


class TestComplexAutogradOnNeuron:
    """Regression tests for complex autograd correctness.

    The backward of complex mul uses .conj() which must be materialized
    during clone/copy. Previously the conj bit was silently dropped,
    causing gradient corruption (swapped real/imag components).
    """

    @pytest.fixture(autouse=True)
    def _allow_complex_fallback(self, monkeypatch):
        monkeypatch.delenv("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS", raising=False)

    def test_complex_mul_backward_grad_matches_cpu(self):
        """grad of view_as_real(t * freq).sum() w.r.t. t must match CPU."""
        t_cpu = torch.randn(4, dtype=torch.complex64, requires_grad=True)
        freq_cpu = torch.randn(4, dtype=torch.complex64)

        t_neuron = t_cpu.detach().clone().to("neuron").requires_grad_(True)
        freq_neuron = freq_cpu.to("neuron")

        torch.view_as_real(t_cpu * freq_cpu).sum().backward()
        torch.view_as_real(t_neuron * freq_neuron).sum().backward()

        torch.testing.assert_close(t_neuron.grad.cpu(), t_cpu.grad, atol=1e-5, rtol=1e-4)

    def test_view_as_complex_mul_backward(self):
        """view_as_complex → mul → backward: the original bug scenario."""
        freq = torch.tensor([1.0 + 1.0j, 0.5 - 0.5j], dtype=torch.complex64)

        x_cpu = torch.randn(1, 4, requires_grad=True)
        z_cpu = torch.view_as_complex(x_cpu.view(1, 2, 2)) * freq
        torch.view_as_real(z_cpu).sum().backward()

        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)
        z_neuron = torch.view_as_complex(x_neuron.view(1, 2, 2)) * freq.to("neuron")
        torch.view_as_real(z_neuron).sum().backward()

        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, atol=1e-5, rtol=1e-4)

    def test_linear_view_as_complex_mul_backward(self):
        """Linear → view_as_complex → mul → backward: weight grad must match CPU.

        Regression: complex mul backward computes grad * freq.conj(). If copy_
        doesn't materialize the conj bit, the multiply uses freq instead of
        conj(freq), producing pairwise-swapped rows in weight.grad.
        """
        torch.manual_seed(42)
        freq = torch.tensor([1.0 + 1.0j, 0.5 - 0.5j], dtype=torch.complex64)
        x = torch.randn(1, 4)

        # CPU reference
        linear_cpu = torch.nn.Linear(4, 4, bias=False)
        y = linear_cpu(x).view(1, 2, 2)
        torch.view_as_real(torch.view_as_complex(y) * freq).sum().backward()

        # Neuron
        linear_neuron = torch.nn.Linear(4, 4, bias=False)
        linear_neuron.weight = torch.nn.Parameter(linear_cpu.weight.detach().clone().to("neuron"))
        y_n = linear_neuron(x.to("neuron")).view(1, 2, 2)
        torch.view_as_real(torch.view_as_complex(y_n) * freq.to("neuron")).sum().backward()

        torch.testing.assert_close(
            linear_neuron.weight.grad.cpu(), linear_cpu.weight.grad, atol=1e-5, rtol=1e-4
        )

    def test_conj_grad_accumulation(self):
        """Two backward passes accumulate correctly (add_ with conj'd tensor)."""
        torch.manual_seed(42)
        freq = torch.tensor([1.0 + 1.0j, 0.5 - 0.5j], dtype=torch.complex64)
        x = torch.randn(1, 4)

        # CPU reference
        linear_cpu = torch.nn.Linear(4, 4, bias=False)
        y = linear_cpu(x).view(1, 2, 2)
        loss = torch.view_as_real(torch.view_as_complex(y) * freq).sum()
        loss.backward()
        loss = torch.view_as_real(torch.view_as_complex(linear_cpu(x).view(1, 2, 2)) * freq).sum()
        loss.backward()

        # Neuron
        linear_neuron = torch.nn.Linear(4, 4, bias=False)
        linear_neuron.weight = torch.nn.Parameter(linear_cpu.weight.detach().clone().to("neuron"))
        x_n = x.to("neuron")
        freq_n = freq.to("neuron")
        y_n = linear_neuron(x_n).view(1, 2, 2)
        torch.view_as_real(torch.view_as_complex(y_n) * freq_n).sum().backward()
        y_n2 = linear_neuron(x_n).view(1, 2, 2)
        torch.view_as_real(torch.view_as_complex(y_n2) * freq_n).sum().backward()

        torch.testing.assert_close(
            linear_neuron.weight.grad.cpu(), linear_cpu.weight.grad, atol=1e-5, rtol=1e-4
        )

    def test_complex_mul_conj_explicit(self):
        """Explicitly verify that mul with a conjugated tensor is correct."""
        a = torch.randn(4, 4, dtype=torch.complex64)
        b = torch.randn(4, 4, dtype=torch.complex64)

        expected = (a * b.conj()).resolve_conj()
        result = a.to("neuron") * b.to("neuron").conj()

        torch.testing.assert_close(result.cpu(), expected, atol=1e-5, rtol=1e-4)
