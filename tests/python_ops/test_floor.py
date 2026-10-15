"""Tests for floor operation."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestFloor:
    """Test suite for aten::floor, aten::floor.out, aten::floor_ operations."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 5), (100, 100)])
    def test_floor_basic_shapes(self, device, shape):
        """Test floor with various tensor shapes."""
        with track_neuron_ops():
            x = torch.randn(shape, device=device) * 10
            neuron_result = torch.floor(x)
            cpu_result = torch.floor(x.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::floor")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_floor_float_dtypes(self, device, dtype):
        """Test floor with floating-point data types."""
        with track_neuron_ops():
            x = torch.tensor([1.7, -1.7, 2.5, -2.5, 0.9, -0.1], dtype=dtype, device=device)
            neuron_result = torch.floor(x)
            cpu_result = torch.floor(x.cpu())

            assert neuron_result.device.type == "neuron"
            assert neuron_result.dtype == dtype
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::floor")

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_floor_int_passthrough(self, device, dtype):
        """Test floor returns input unchanged for integer dtypes."""
        with track_neuron_ops():
            x = torch.tensor([1, -1, 0, 5, -5], dtype=dtype, device=device)
            neuron_result = torch.floor(x)
            cpu_result = torch.floor(x.cpu())

            assert neuron_result.device.type == "neuron"
            assert neuron_result.dtype == dtype
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::floor")

    def test_floor_out_variant(self, device):
        """Test floor .out variant writes to pre-allocated tensor."""
        with track_neuron_ops():
            x = torch.tensor([1.7, -1.7, 2.5, -2.5], device=device)
            out = torch.empty_like(x)
            result = torch.floor(x, out=out)

            expected = torch.floor(x.cpu())
            assert out.data_ptr() == result.data_ptr()
            torch.testing.assert_close(out.cpu(), expected)
            assert_op_runs_on_neuron("aten::floor")

    @pytest.mark.parametrize(
        "in_dtype,out_dtype",
        [(torch.float32, torch.int64), (torch.int32, torch.float32)],
    )
    def test_floor_out_dtype_mismatch_rejected(self, device, in_dtype, out_dtype):
        """floor preserves the input dtype, so an out= with a different dtype
        must be rejected. Assert Neuron raises the same exception type and
        message as CPU (rather than a hardcoded string) — the framework skips
        this structured-kernel guard, so the op registers it explicitly."""
        if in_dtype.is_floating_point:
            x = torch.randn(6, 8, dtype=in_dtype)
        else:
            x = torch.randint(0, 5, (6, 8), dtype=in_dtype)
        out = torch.empty(0, dtype=out_dtype)

        # Reference error from CPU.
        with pytest.raises(Exception) as cpu_exc:
            torch.floor(x, out=out)

        # Neuron must reject identically.
        with pytest.raises(Exception) as neuron_exc:
            torch.floor(x.to(device), out=out.to(device))

        assert type(neuron_exc.value) is type(cpu_exc.value)
        assert str(neuron_exc.value) == str(cpu_exc.value)

    def test_floor_inplace(self, device):
        """Test floor_ in-place variant."""
        with track_neuron_ops():
            x = torch.tensor([1.7, -1.7, 2.5, -2.5], device=device)
            original_ptr = x.data_ptr()
            x.floor_()

            expected = torch.tensor([1.0, -2.0, 2.0, -3.0])
            assert x.data_ptr() == original_ptr
            torch.testing.assert_close(x.cpu(), expected)
            assert_op_runs_on_neuron("aten::floor")

    def test_floor_empty_tensor(self, device):
        """Test floor with empty tensor (numel=0)."""
        with track_neuron_ops():
            x = torch.tensor([], device=device)
            neuron_result = torch.floor(x)
            cpu_result = torch.floor(x.cpu())

            assert neuron_result.shape == cpu_result.shape
            assert neuron_result.numel() == 0
            assert_op_runs_on_neuron("aten::floor")

    def test_floor_single_element(self, device):
        """Test floor with single element tensor."""
        with track_neuron_ops():
            x = torch.tensor([3.7], device=device)
            neuron_result = torch.floor(x)
            cpu_result = torch.floor(x.cpu())

            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::floor")

    def test_floor_large_float_values(self, device):
        """Test floor with float32 values outside int32 range.

        Regression test for neuronx-cc stablehlo.floor bug where values
        outside int32 range get clamped to ±2.1475e+09. The workaround
        bypasses floor for |x| >= 2^23 (where floats are already integers).
        """
        with track_neuron_ops():
            x = torch.tensor(
                [2.5e9, -2.5e9, 1e10, -1e10, 8388608.0, -8388608.0, 1e38, -1e38],
                device=device,
            )
            neuron_result = torch.floor(x)
            cpu_result = torch.floor(x.cpu())

            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::floor")

    def test_floor_boundary_values(self, device):
        """Test floor at the 2^23 boundary where workaround activates."""
        with track_neuron_ops():
            # Values just below and at the boundary
            x = torch.tensor(
                [8388607.5, 8388608.0, 8388609.0, -8388607.5, -8388608.0, -8388609.0],
                device=device,
            )
            neuron_result = torch.floor(x)
            cpu_result = torch.floor(x.cpu())

            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::floor")

    def test_floor_special_values(self, device):
        """Test floor with inf, -inf, nan — the op passes non-finite values
        through unchanged (floor is identity on inf/-inf/nan), working around a
        neuronx-cc bug where floor(nan) returns 2^31."""
        with track_neuron_ops():
            x = torch.tensor([float("inf"), float("-inf"), float("nan")], device=device)
            neuron_result = torch.floor(x)
            cpu_result = torch.floor(x.cpu())

            torch.testing.assert_close(neuron_result.cpu(), cpu_result, equal_nan=True)
            assert_op_runs_on_neuron("aten::floor")

    def test_floor_no_recompilation(self, device):
        """Test that calling floor twice with same shape does not recompile."""
        import time

        with track_neuron_ops():
            x = torch.randn(64, 64, device=device)
            # First call — compiles
            torch.floor(x)
            torch.neuron.synchronize()

            # Second call — should use cached NEFF
            start = time.perf_counter()
            torch.floor(x)
            torch.neuron.synchronize()
            elapsed = time.perf_counter() - start

            assert elapsed < 0.1, f"Recompiled: {elapsed:.3f}s"
            assert_op_runs_on_neuron("aten::floor")

    def test_floor_does_not_break_autograd(self, device):
        """Test that floor registration does not break autograd chains.

        floor is non-differentiable (gradient is zero), but it must not
        disconnect the autograd graph for upstream ops.
        """
        with track_neuron_ops():
            x = torch.randn(4, 4, device=device, requires_grad=True)
            y = x * 2.0  # differentiable
            z = torch.floor(y)
            loss = (z + x).sum()
            loss.backward()
            assert x.grad is not None
