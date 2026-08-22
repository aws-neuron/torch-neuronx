"""Tests for complex ops that run natively on Neuron via the complex allowlist."""

import json
import os

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


class TestComplexCopyBits:
    """copy_ must reconcile the conj/neg bits of a complex src and dst.

    aten::copy_ is a fallthrough at ATen's Conjugate and Negative keys, so
    materializing those bits belongs to copy_ — and it is copy_ that ATen's
    conjugate fallback calls, through clone, for complex mul backward.
    """

    @staticmethod
    def _empty_like_neuron(t_cpu):
        return torch.empty(t_cpu.shape, dtype=t_cpu.dtype, device="neuron")

    def test_copy_from_conj_view(self):
        src_cpu = torch.randn(8, 4, dtype=torch.complex64)
        dst = self._empty_like_neuron(src_cpu)
        dst.copy_(src_cpu.to("neuron").conj())
        torch.testing.assert_close(dst.cpu(), src_cpu.conj().resolve_conj())

    def test_copy_from_neg_view(self):
        src_cpu = torch.randn(8, 4, dtype=torch.complex64)
        dst = self._empty_like_neuron(src_cpu)
        dst.copy_(torch._neg_view(src_cpu.to("neuron")))
        torch.testing.assert_close(dst.cpu(), -src_cpu)

    def test_copy_from_conj_and_neg_view(self):
        """conj ∘ neg flips only the real lane — the case a single flip misses."""
        src_cpu = torch.randn(8, 4, dtype=torch.complex64)
        dst = self._empty_like_neuron(src_cpu)
        dst.copy_(torch._neg_view(src_cpu.to("neuron").conj()))
        torch.testing.assert_close(dst.cpu(), (-src_cpu.conj()).resolve_conj())

    def test_copy_into_conj_view(self):
        """A conj bit on dst means the written physical values must be conjugated."""
        src_cpu = torch.randn(8, 4, dtype=torch.complex64)
        dst = self._empty_like_neuron(src_cpu)
        dst.conj().copy_(src_cpu.to("neuron"))
        torch.testing.assert_close(dst.cpu(), src_cpu.conj().resolve_conj())

    def test_copy_conj_view_into_conj_view(self):
        """Matching bits on both sides cancel out — a plain copy."""
        src_cpu = torch.randn(8, 4, dtype=torch.complex64)
        dst = self._empty_like_neuron(src_cpu)
        dst.conj().copy_(src_cpu.to("neuron").conj())
        torch.testing.assert_close(dst.cpu(), src_cpu)

    def test_clone_of_conj_view(self):
        """What ATen's conjugate fallback emits for `x * y.conj()`."""
        src_cpu = torch.randn(8, 4, dtype=torch.complex64)
        result = src_cpu.to("neuron").conj().clone()
        assert not result.is_conj()
        torch.testing.assert_close(result.cpu(), src_cpu.conj().resolve_conj())

    def test_resolve_conj(self):
        src_cpu = torch.randn(8, 4, dtype=torch.complex64)
        result = src_cpu.to("neuron").conj().resolve_conj()
        torch.testing.assert_close(result.cpu(), src_cpu.conj().resolve_conj())

    def test_copy_conj_broadcast(self):
        src_cpu = torch.randn(1, 4, dtype=torch.complex64)
        dst = torch.empty(8, 4, dtype=torch.complex64, device="neuron")
        dst.copy_(src_cpu.to("neuron").conj())
        torch.testing.assert_close(dst.cpu(), src_cpu.conj().resolve_conj().expand(8, 4).clone())

    def test_copy_conj_noncontiguous_dst(self):
        src_cpu = torch.randn(8, 4, dtype=torch.complex64)
        dst = torch.empty(4, 8, dtype=torch.complex64, device="neuron").t()
        dst.copy_(src_cpu.to("neuron").conj())
        torch.testing.assert_close(dst.cpu(), src_cpu.conj().resolve_conj())

    def test_copy_conj_noncontiguous_src(self):
        src_cpu = torch.randn(4, 8, dtype=torch.complex64)
        dst = torch.empty(8, 4, dtype=torch.complex64, device="neuron")
        dst.copy_(src_cpu.to("neuron").t().conj())
        torch.testing.assert_close(dst.cpu(), src_cpu.t().conj().resolve_conj())

    def test_copy_conj_zero_dim(self):
        src_cpu = torch.randn((), dtype=torch.complex64)
        dst = torch.empty((), dtype=torch.complex64, device="neuron")
        dst.copy_(src_cpu.to("neuron").conj())
        torch.testing.assert_close(dst.cpu(), src_cpu.conj().resolve_conj())

    def test_copy_conj_of_self(self):
        """src aliasing dst — declined by the on-device path, still correct."""
        src_cpu = torch.randn(8, 4, dtype=torch.complex64)
        x = src_cpu.to("neuron")
        x.copy_(x.conj())
        torch.testing.assert_close(x.cpu(), src_cpu.conj().resolve_conj())

    # The declined cases with a flagged dst. The CPU bounce materializes only the
    # source, so dst's bits are moved onto src before declining; without that,
    # these write unconjugated values.

    def test_copy_into_conj_view_of_self(self):
        """Reverse of test_copy_conj_of_self: the alias guard with a flagged dst."""
        src_cpu = torch.randn(8, 4, dtype=torch.complex64)
        x = src_cpu.to("neuron")
        x.conj().copy_(x)
        torch.testing.assert_close(x.cpu(), src_cpu.conj().resolve_conj())

    def test_copy_into_neg_view_of_self(self):
        src_cpu = torch.randn(8, 4, dtype=torch.complex64)
        x = src_cpu.to("neuron")
        torch._neg_view(x).copy_(x)
        torch.testing.assert_close(x.cpu(), -src_cpu)

    def test_copy_into_conj_view_of_wider_dtype(self):
        """Reverse direction for the dtype guard: complex64 src into a c128 conj dst."""
        src_cpu = torch.randn(8, 4, dtype=torch.complex64)
        dst = torch.empty(8, 4, dtype=torch.complex128, device="neuron")
        dst.conj().copy_(src_cpu.to("neuron"))
        torch.testing.assert_close(dst.cpu(), src_cpu.conj().resolve_conj().to(torch.complex128))

    def test_copy_complex128_conj_keeps_float64_precision(self):
        """complex128 stays on the bounce: DowncastArgs would round its float64 lanes."""
        src_cpu = torch.tensor([1.0000000001 + 2.0000000002j], dtype=torch.complex128)
        assert (
            src_cpu.real.item() != src_cpu.real.to(torch.float32).item()
        ), "pick a value float32 cannot represent, or this test cannot detect the downcast"
        dst = torch.empty(1, dtype=torch.complex128, device="neuron")
        dst.copy_(src_cpu.to("neuron").conj())
        torch.testing.assert_close(dst.cpu(), src_cpu.conj().resolve_conj(), rtol=0, atol=0)

    def test_contiguous_of_conj_view(self):
        """`contiguous` on a flagged non-contiguous view now goes through copy_."""
        src_cpu = torch.randn(4, 5, dtype=torch.complex64)
        result = src_cpu.to("neuron").t().conj().contiguous()
        assert result.is_contiguous()
        torch.testing.assert_close(result.cpu(), src_cpu.t().conj().resolve_conj())


class TestComplexCopyCpuToNeuronBits:
    """Known-broken: CPU→Neuron `copy_` ignores conj/neg on both sides (#1697).

    Strict xfails so a fix flips them to failures — remove the markers when #1697 lands.
    """

    @pytest.mark.xfail(strict=True, reason="#1697: raw upload ignores dst.is_conj()")
    def test_copy_cpu_src_into_conj_view(self):
        """dst reads as conj(base), so base must physically hold conj(src)."""
        src = torch.tensor([[1.0 + 2.0j]], dtype=torch.complex64)
        base = torch.empty(1, 1, dtype=torch.complex64, device="neuron")
        base.conj().copy_(src)
        torch.testing.assert_close(base.cpu(), src.conj().resolve_conj())

    @pytest.mark.xfail(strict=True, reason="#1697: raw upload ignores dst.is_neg()")
    def test_copy_cpu_src_into_neg_view(self):
        src = torch.tensor([[3.0 + 0.0j]], dtype=torch.complex64)
        base = torch.empty(1, 1, dtype=torch.complex64, device="neuron")
        torch._neg_view(base).copy_(src)
        torch.testing.assert_close(base.cpu(), -src)

    @pytest.mark.xfail(strict=True, reason="#1697: raw upload ignores src.is_conj()")
    def test_copy_from_contiguous_conj_cpu_src(self):
        """A contiguous conj'd CPU src: nothing resolves the bit before the upload."""
        src = torch.tensor([[1.0 + 2.0j]], dtype=torch.complex64)
        dst = torch.empty(1, 1, dtype=torch.complex64, device="neuron")
        dst.copy_(src.conj())
        torch.testing.assert_close(dst.cpu(), src.conj().resolve_conj())

    def test_copy_from_noncontiguous_conj_cpu_src(self):
        """Control: `.contiguous()` incidentally resolves the bit here, masking the bug."""
        src = torch.randn(2, 2, dtype=torch.complex64)
        dst = torch.empty(2, 2, dtype=torch.complex64, device="neuron")
        dst.copy_(src.t().conj())
        torch.testing.assert_close(dst.cpu(), src.t().conj().resolve_conj())


class TestComplexCopyStaysOnDevice:
    """The conj materialization must not round-trip through the host.

    Complex copy_ used to land in DispatchCopyImpl's CPU bounce: blocking D2H,
    `conj_physical_()` on a host temporary, H2D. RoPE with the complex backend
    paid two of those per layer per step, plus the drain each blocking read forces.
    """

    @pytest.fixture(autouse=True)
    def _profile_dir(self, tmp_path):
        self.profile_dir = str(tmp_path)

    def _host_copies_during(self, fn):
        """Names of host<->device transfer events emitted while running `fn`."""
        from torch.profiler import ProfilerActivity, profile

        from torch_neuronx.profiling import NeuronConfig, ProfileMode

        torch.neuron.synchronize()
        config = NeuronConfig(modes=[ProfileMode.RUNTIME], profile_output_dir=self.profile_dir)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            experimental_config=config,
        ) as prof:
            fn()
            torch.neuron.synchronize()

        trace_file = os.path.join(self.profile_dir, "trace.json")
        prof.export_chrome_trace(trace_file)
        with open(trace_file) as f:
            events = [e for e in json.load(f)["traceEvents"] if e.get("ph") == "X"]

        device_events = [e for e in events if e.get("cat") == "privateuse1_driver"]
        assert device_events, "no device events captured — the assertion below would be vacuous"
        return sorted(
            {
                e["name"]
                for e in events
                if "neuron_to_cpu" in e["name"] or "cpu_to_neuron" in e["name"]
            }
        )

    def test_clone_of_conj_view_issues_no_host_copy(self):
        src = torch.randn(64, 8, dtype=torch.complex64).to("neuron")
        src.conj().clone()  # warm the kernels the path needs before profiling
        assert self._host_copies_during(lambda: src.conj().clone()) == []

    def test_mul_by_conj_issues_no_host_copy(self):
        """The complex mul backward shape: ATen clones `other.conj()` for us."""
        a = torch.randn(64, 8, dtype=torch.complex64).to("neuron")
        b = torch.randn(64, 8, dtype=torch.complex64).to("neuron")
        a * b.conj()  # warm
        assert self._host_copies_during(lambda: a * b.conj()) == []

    def test_contiguous_of_conj_view_issues_no_host_copy(self):
        src = torch.randn(64, 8, dtype=torch.complex64).to("neuron")
        src.t().conj().contiguous()  # warm
        assert self._host_copies_during(lambda: src.t().conj().contiguous()) == []

    def test_aliased_copy_still_uses_host_bounce(self):
        """Positive control: the declined case proves the check above can fail."""
        x = torch.randn(64, 8, dtype=torch.complex64).to("neuron")
        x.copy_(x.conj())  # warm
        assert self._host_copies_during(lambda: x.copy_(x.conj())) == [
            "neuron::copy::cpu_to_neuron",
            "neuron::copy::neuron_to_cpu",
        ]
