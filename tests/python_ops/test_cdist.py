"""Tests for aten::_cdist_forward on Neuron."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestCdistForward:
    # cdist accumulates sqrt(sum(|diff|^p)); Neuron's fp32/fp64 reduction drifts
    # a few ULP past the strict default rtol (observed max ~7e-6 rel fp32,
    # ~5e-6 rel fp64), so a small bump above defaults is required.
    _RTOL = 1e-5
    _ATOL = 1e-5

    @pytest.mark.parametrize("p", [1.0, 2.0, 3.0, 0.5])
    @pytest.mark.parametrize("shape1,shape2", [((3, 4), (5, 4)), ((1, 2), (6, 2))])
    def test_forward(self, p, shape1, shape2):
        x1_cpu = torch.randn(shape1)
        x2_cpu = torch.randn(shape2)
        with track_neuron_ops():
            result = torch.ops.aten._cdist_forward(
                x1_cpu.to("neuron"), x2_cpu.to("neuron"), p, None
            )
            assert_op_runs_on_neuron("aten::_cdist_forward")
        expected = torch.ops.aten._cdist_forward(x1_cpu, x2_cpu, p, None)
        torch.testing.assert_close(result.cpu(), expected, rtol=self._RTOL, atol=self._ATOL)

    @pytest.mark.parametrize("compute_mode", [None, 1, 2])
    def test_compute_mode(self, compute_mode):
        x1_cpu = torch.randn(4, 8)
        x2_cpu = torch.randn(6, 8)
        with track_neuron_ops():
            result = torch.ops.aten._cdist_forward(
                x1_cpu.to("neuron"), x2_cpu.to("neuron"), 2.0, compute_mode
            )
            assert_op_runs_on_neuron("aten::_cdist_forward")
        expected = torch.ops.aten._cdist_forward(x1_cpu, x2_cpu, 2.0, compute_mode)
        torch.testing.assert_close(result.cpu(), expected, rtol=self._RTOL, atol=self._ATOL)

    def test_batched(self):
        x1_cpu = torch.randn(2, 3, 4)
        x2_cpu = torch.randn(2, 5, 4)
        with track_neuron_ops():
            result = torch.ops.aten._cdist_forward(
                x1_cpu.to("neuron"), x2_cpu.to("neuron"), 2.0, None
            )
            assert_op_runs_on_neuron("aten::_cdist_forward")
        expected = torch.ops.aten._cdist_forward(x1_cpu, x2_cpu, 2.0, None)
        torch.testing.assert_close(result.cpu(), expected, rtol=self._RTOL, atol=self._ATOL)

    @pytest.mark.parametrize("p", [0.0, float("inf")])
    def test_p_zero_and_inf(self, p):
        """p=0 (count of differing coords) and p=inf (max abs coord diff) can't
        use the generic sum(|diff|^p)^(1/p) path -- it degenerates to wrong
        results. They're decomposed explicitly and must run on Neuron while
        matching CPU."""
        x1_cpu = torch.randn(4, 8)
        x2_cpu = torch.randn(6, 8)
        with track_neuron_ops():
            result = torch.ops.aten._cdist_forward(
                x1_cpu.to("neuron"), x2_cpu.to("neuron"), p, None
            )
            assert_op_runs_on_neuron("aten::_cdist_forward")
        expected = torch.ops.aten._cdist_forward(x1_cpu, x2_cpu, p, None)
        torch.testing.assert_close(result.cpu(), expected, rtol=self._RTOL, atol=self._ATOL)

    @pytest.mark.parametrize("p", [1.0, 2.0, float("inf")])
    @pytest.mark.parametrize(
        "shape1,shape2",
        [
            ((1, 3, 4), (2, 5, 4)),  # differing size-1 batch, equal rank
            ((3, 4), (2, 5, 4)),  # differing rank (2D vs 3D)
            ((2, 1, 2, 2), (1, 3, 2, 2)),  # broadcast batch dims both sides
        ],
    )
    def test_broadcast_batch_dims(self, p, shape1, shape2):
        """torch-mlir marks differing-rank / broadcast batch dims illegal; these
        are decomposed so they run on Neuron and match CPU broadcasting."""
        x1_cpu = torch.randn(shape1)
        x2_cpu = torch.randn(shape2)
        with track_neuron_ops():
            result = torch.ops.aten._cdist_forward(
                x1_cpu.to("neuron"), x2_cpu.to("neuron"), p, None
            )
            assert_op_runs_on_neuron("aten::_cdist_forward")
        expected = torch.ops.aten._cdist_forward(x1_cpu, x2_cpu, p, None)
        torch.testing.assert_close(result.cpu(), expected, rtol=self._RTOL, atol=self._ATOL)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtypes(self, dtype):
        # float32/float64 are supported on both CPU and Neuron; compare directly.
        x1_cpu = torch.randn(4, 8, dtype=dtype)
        x2_cpu = torch.randn(6, 8, dtype=dtype)
        with track_neuron_ops():
            result = torch.ops.aten._cdist_forward(
                x1_cpu.to("neuron"), x2_cpu.to("neuron"), 2.0, None
            )
            assert_op_runs_on_neuron("aten::_cdist_forward")
        assert result.dtype == dtype
        expected = torch.ops.aten._cdist_forward(x1_cpu, x2_cpu, 2.0, None)
        torch.testing.assert_close(result.cpu(), expected, rtol=self._RTOL, atol=self._ATOL)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_extended_dtypes(self, dtype):
        # Neuron supports fp16/bf16 cdist; CPU does not, so reference against fp32 CPU.
        x1_cpu = torch.randn(4, 8, dtype=dtype)
        x2_cpu = torch.randn(6, 8, dtype=dtype)
        with track_neuron_ops():
            result = torch.ops.aten._cdist_forward(
                x1_cpu.to("neuron"), x2_cpu.to("neuron"), 2.0, None
            )
            assert_op_runs_on_neuron("aten::_cdist_forward")
        assert result.dtype == dtype
        expected = torch.ops.aten._cdist_forward(x1_cpu.float(), x2_cpu.float(), 2.0, None)
        # Tolerances reflect each dtype's precision vs the fp32 reference.
        rtol, atol = (1e-3, 3e-3) if dtype == torch.float16 else (1e-2, 3e-2)
        torch.testing.assert_close(result.cpu().float(), expected, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_integer_dtypes_rejected(self, dtype):
        """Integer inputs are unsupported; Neuron must raise like CPU (not silently run)."""
        x1 = torch.randint(0, 5, (4, 8), dtype=dtype)
        x2 = torch.randint(0, 5, (6, 8), dtype=dtype)
        # CPU: "cdist only supports floating-point dtypes". Neuron raises the same.
        with pytest.raises(RuntimeError, match="only supports floating-point"):
            torch.ops.aten._cdist_forward(x1, x2, 2.0, None)
        with pytest.raises(RuntimeError, match="only supports floating-point"):
            torch.ops.aten._cdist_forward(x1.to("neuron"), x2.to("neuron"), 2.0, None)

    def test_out_variant(self):
        x1_cpu = torch.randn(3, 4)
        x2_cpu = torch.randn(5, 4)
        expected = torch.ops.aten._cdist_forward(x1_cpu, x2_cpu, 2.0, None)
        out_n = torch.empty_like(expected).to("neuron")
        with track_neuron_ops():
            result = torch.ops.aten._cdist_forward.out(
                x1_cpu.to("neuron"), x2_cpu.to("neuron"), 2.0, None, out=out_n
            )
            assert_op_runs_on_neuron("aten::_cdist_forward.out")
        torch.testing.assert_close(result.cpu(), expected, rtol=self._RTOL, atol=self._ATOL)
        torch.testing.assert_close(out_n.cpu(), expected, rtol=self._RTOL, atol=self._ATOL)
