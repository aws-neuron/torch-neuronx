"""Tests for aten::_pdist_forward on Neuron."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestPdistForward:
    # Finite-p pdist accumulates sqrt(sum(|diff|^p)); Neuron's fp32/fp64
    # reduction drifts a few ULP past the strict default rtol (observed max
    # ~2e-6 fp32, ~1e-6 fp64), so a small bump above defaults is required. The
    # exact cases (p=0/inf, empty output) below keep the strict default.
    _RTOL = 1e-5
    _ATOL = 1e-5

    @pytest.mark.parametrize("p", [1.0, 2.0, 3.0, 0.5])
    @pytest.mark.parametrize("shape", [(5, 4), (8, 16), (2, 3)])
    def test_forward(self, p, shape):
        x_cpu = torch.randn(shape)
        with track_neuron_ops():
            result = torch.ops.aten._pdist_forward(x_cpu.to("neuron"), p)
            assert_op_runs_on_neuron("aten::_pdist_forward")
        expected = torch.ops.aten._pdist_forward(x_cpu, p)
        torch.testing.assert_close(result.cpu(), expected, rtol=self._RTOL, atol=self._ATOL)

    def test_default_p(self):
        x_cpu = torch.randn(6, 8)
        with track_neuron_ops():
            result = torch.ops.aten._pdist_forward(x_cpu.to("neuron"))
            assert_op_runs_on_neuron("aten::_pdist_forward")
        expected = torch.ops.aten._pdist_forward(x_cpu)
        torch.testing.assert_close(result.cpu(), expected, rtol=self._RTOL, atol=self._ATOL)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtypes(self, dtype):
        # pdist (and its OpInfo) supports float32 and float64; both run on Neuron.
        x_cpu = torch.randn(6, 8, dtype=dtype)
        with track_neuron_ops():
            result = torch.ops.aten._pdist_forward(x_cpu.to("neuron"), 2.0)
            assert_op_runs_on_neuron("aten::_pdist_forward")
        assert result.dtype == dtype
        expected = torch.ops.aten._pdist_forward(x_cpu, 2.0)
        torch.testing.assert_close(result.cpu(), expected, rtol=self._RTOL, atol=self._ATOL)

    @pytest.mark.parametrize("p", [0.0, float("inf")])
    @pytest.mark.parametrize("shape", [(5, 4), (8, 6), (2, 3)])
    def test_p_zero_and_inf(self, p, shape):
        """p=0 (count of differing coords) and p=inf (max abs coord diff) can't
        use the generic sum(|diff|^p)^(1/p) path -- it degenerates to wrong
        results. They're decomposed explicitly and must run on Neuron (not fall
        back) while matching CPU exactly."""
        x_cpu = torch.randn(shape)
        with track_neuron_ops():
            result = torch.ops.aten._pdist_forward(x_cpu.to("neuron"), p)
            assert_op_runs_on_neuron("aten::_pdist_forward")
        expected = torch.ops.aten._pdist_forward(x_cpu, p)
        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize("p", [0.0, 2.0, float("inf")])
    def test_single_row_empty_output(self, p):
        """A single row has no pairs, so pdist returns an empty (0,) tensor."""
        x_cpu = torch.randn(1, 4)
        result = torch.ops.aten._pdist_forward(x_cpu.to("neuron"), p)
        expected = torch.ops.aten._pdist_forward(x_cpu, p)
        assert result.shape == (0,)
        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int32, torch.int64])
    def test_unsupported_dtypes_rejected(self, dtype):
        """pdist supports only fp32/fp64 (per its OpInfo); everything else must
        raise on Neuron rather than silently running (fp16/bf16) or hitting an
        internal compiler error (integers).

        Capture the error from both CPU and Neuron and assert they match exactly
        (same exception type and message), rather than hardcoding a string that
        would drift across torch upgrades."""
        if dtype.is_floating_point:
            x = torch.randn(6, 8, dtype=dtype)
        else:
            x = torch.randint(0, 5, (6, 8), dtype=dtype)

        # Reference error from CPU.
        with pytest.raises(Exception) as cpu_exc:
            torch.ops.aten._pdist_forward(x, 2.0)

        # Same op on Neuron.
        with pytest.raises(Exception) as neuron_exc:
            torch.ops.aten._pdist_forward(x.to("neuron"), 2.0)

        # Neuron must reject identically to CPU.
        assert type(neuron_exc.value) is type(cpu_exc.value)
        assert str(neuron_exc.value) == str(cpu_exc.value)

    def test_out_variant(self):
        x_cpu = torch.randn(5, 4)
        expected = torch.ops.aten._pdist_forward(x_cpu, 2.0)
        out_n = torch.empty_like(expected).to("neuron")
        with track_neuron_ops():
            result = torch.ops.aten._pdist_forward.out(x_cpu.to("neuron"), 2.0, out=out_n)
            assert_op_runs_on_neuron("aten::_pdist_forward.out")
        torch.testing.assert_close(result.cpu(), expected, rtol=self._RTOL, atol=self._ATOL)
        torch.testing.assert_close(out_n.cpu(), expected, rtol=self._RTOL, atol=self._ATOL)
