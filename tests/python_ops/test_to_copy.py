import os

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_did_not_run_on_neuron,
    assert_op_runs_on_neuron,
    count_cpu_bounce_calls,
    track_neuron_ops,
)
from torch_neuronx.utils import is_native_int64_enabled


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestToCopy:
    """Test suite for to_copy operation."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.mark.parametrize(
        "torch_dtype", [torch.float32, torch.bfloat16, torch.float8_e5m2, torch.float8_e4m3fn]
    )
    def test_nonzero_static_basic_shapes(self, device, torch_dtype):
        """Test nonzero_static with various tensor shapes."""

        with track_neuron_ops():
            # TESTING CPU -> NEURON
            x = torch.randn(128, dtype=torch.float32).to(torch_dtype)
            neuron_result = x.to(device=device)
            assert neuron_result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::_to_copy")

            # TESTING NEURON -> NEURON
            x = torch.randn(128, dtype=torch.float32, device=device)
            neuron_result = x.to(torch_dtype)
            assert neuron_result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::_to_copy")
            assert_op_runs_on_neuron("aten::copy_")


def test_to_copy_blocking_uses_mlir():
    """Test non_blocking=False uses MLIR implementation."""
    from torch_neuronx.python_ops.torch_mlir.ops.to_copy import ToCopyMLIRImpl

    src = torch.randn(4, 4, device="neuron")
    impl = ToCopyMLIRImpl()
    assert impl.can_handle(src, dtype=torch.float16, non_blocking=False)

    dst = src.to(dtype=torch.float16, non_blocking=False)

    assert dst.device.type == "neuron"
    assert dst.dtype == torch.float16
    torch.testing.assert_close(src.cpu(), dst.cpu().to(torch.float32), rtol=1e-2, atol=1e-2)


def test_to_copy_fp64_to_fp32():
    """Test neuron fp64 to neuron fp32 - MLIR impl should reject float64"""
    from torch_neuronx.python_ops.torch_mlir.ops.to_copy import ToCopyMLIRImpl

    # Create fp64 tensor on neuron, convert to neuron fp32
    src = torch.randn(4, 4, dtype=torch.float64, device="neuron")

    # Verify MLIR implementation rejects float64
    impl = ToCopyMLIRImpl()
    assert not impl.can_handle(src, dtype=torch.float32, device="neuron")

    # Operation still works via non-MLIR implementation
    dst = src.to(dtype=torch.float32, device="neuron")
    torch_neuronx.synchronize()
    assert dst.device.type == "neuron"
    assert dst.dtype == torch.float32
    torch.testing.assert_close(src.cpu().to(torch.float32), dst.cpu())


def test_to_copy_fp32_to_fp64():
    """Test neuron fp32 to neuron fp64 - MLIR impl should reject float64"""
    from torch_neuronx.python_ops.torch_mlir.ops.to_copy import ToCopyMLIRImpl

    # Create fp32 tensor on neuron, convert to neuron fp64
    src = torch.randn(4, 4, dtype=torch.float32, device="neuron")

    # Verify MLIR implementation rejects float64
    impl = ToCopyMLIRImpl()
    assert not impl.can_handle(src, dtype=torch.float64, device="neuron")

    # Operation still works via non-MLIR implementation
    dst = src.to(dtype=torch.float64, device="neuron")
    torch_neuronx.synchronize()
    assert dst.device.type == "neuron"
    assert dst.dtype == torch.float64
    torch.testing.assert_close(src.cpu().to(torch.float64), dst.cpu())


def test_to_copy_cpu_fp64_to_neuron_fp32():
    """Test CPU fp64 to neuron fp32"""
    src = torch.randn(4, 4, dtype=torch.float64)
    dst = src.to(dtype=torch.float32, device="neuron")
    torch_neuronx.synchronize()
    assert dst.device.type == "neuron"
    assert dst.dtype == torch.float32
    torch.testing.assert_close(src.to(torch.float32), dst.cpu())


def test_to_copy_cpu_fp32_to_neuron_fp64():
    """Test CPU fp32 to neuron fp64"""
    src = torch.randn(4, 4, dtype=torch.float32)
    dst = src.to(dtype=torch.float64, device="neuron")
    torch_neuronx.synchronize()
    assert dst.device.type == "neuron"
    assert dst.dtype == torch.float64
    torch.testing.assert_close(src.to(torch.float64), dst.cpu())


def test_to_copy_neuron_fp64_to_cpu_fp32():
    """Test neuron fp64 to CPU fp32"""
    src = torch.randn(4, 4, dtype=torch.float64, device="neuron")
    dst = src.to(dtype=torch.float32, device="cpu")
    torch_neuronx.synchronize()
    assert dst.device.type == "cpu"
    assert dst.dtype == torch.float32
    torch.testing.assert_close(src.cpu().to(torch.float32), dst)


def test_to_copy_neuron_fp32_to_cpu_fp64():
    """Test neuron fp32 to CPU fp64"""
    src = torch.randn(4, 4, dtype=torch.float32, device="neuron")
    dst = src.to(dtype=torch.float64, device="cpu")
    torch_neuronx.synchronize()
    assert dst.device.type == "cpu"
    assert dst.dtype == torch.float64
    torch.testing.assert_close(src.cpu().to(torch.float64), dst)


def test_to_copy_neuron_fp64_to_neuron_fp64():
    """Test neuron fp64 to neuron fp64 (same dtype)"""
    src = torch.randn(4, 4, dtype=torch.float64, device="neuron")
    dst = src.to(dtype=torch.float64, device="neuron")
    torch_neuronx.synchronize()
    assert dst.device.type == "neuron"
    assert dst.dtype == torch.float64
    torch.testing.assert_close(src.cpu(), dst.cpu())


def test_to_copy_cpu_fp64_to_neuron_fp64():
    """Test CPU fp64 to neuron fp64"""
    src = torch.randn(4, 4, dtype=torch.float64)
    dst = src.to(device="neuron")
    torch_neuronx.synchronize()
    assert dst.device.type == "neuron"
    assert dst.dtype == torch.float64
    torch.testing.assert_close(src, dst.cpu())


def test_to_copy_neuron_fp64_to_cpu_fp64():
    """Test neuron fp64 to CPU fp64"""
    src = torch.randn(4, 4, dtype=torch.float64, device="neuron")
    dst = src.to(device="cpu")
    torch_neuronx.synchronize()
    assert dst.device.type == "cpu"
    assert dst.dtype == torch.float64
    torch.testing.assert_close(src.cpu(), dst)


def test_to_copy_neuron_fp64_empty():
    """Test empty tensor with fp64"""
    src = torch.empty(0, dtype=torch.float64, device="neuron")
    dst = src.to(dtype=torch.float32, device="neuron")
    assert dst.numel() == 0
    assert dst.dtype == torch.float32
    assert dst.device.type == "neuron"


@pytest.mark.parametrize(
    "src_dtype",
    [
        torch.float32,
        torch.float64,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.int64,
        torch.bool,
    ],
)
@pytest.mark.parametrize(
    "dst_dtype",
    [
        torch.float32,
        torch.float64,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.int64,
        torch.bool,
    ],
)
def test_to_copy_neuron_to_neuron_dtype_combinations(src_dtype, dst_dtype):
    """Test neuron-to-neuron dtype conversions."""
    from torch_neuronx.python_ops.torch_mlir.ops.to_copy import ToCopyMLIRImpl

    if src_dtype == torch.bool:
        src = torch.randint(0, 2, (4, 4), device="neuron").bool()
    elif src_dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
        src = torch.randn(4, 4, dtype=src_dtype, device="neuron")
    else:
        src = torch.randint(0, 10, (4, 4), dtype=src_dtype, device="neuron")

    # Verify MLIR rejects unsupported dtypes
    impl = ToCopyMLIRImpl()
    mlir_unsupported = {torch.bool, torch.float64, torch.int32, torch.int64, torch.uint64}
    if src_dtype in mlir_unsupported or dst_dtype in mlir_unsupported:
        assert not impl.can_handle(src, dtype=dst_dtype, device="neuron")

    dst = src.to(dtype=dst_dtype, device="neuron")
    torch_neuronx.synchronize()
    assert dst.device.type == "neuron"
    assert dst.dtype == dst_dtype
    assert dst.shape == src.shape

    # Only validate correctness for same-type or safe conversions
    # Skip validation for float->int conversions due to known rounding differences
    is_float_src = src_dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]
    is_int_dst = dst_dtype in [torch.int32, torch.int64]

    if not (is_float_src and is_int_dst):
        expected = src.cpu().to(dst_dtype)
        actual = dst.cpu()
        if dst_dtype == torch.bool or dst_dtype in [torch.int32, torch.int64]:
            assert torch.equal(expected, actual)
        else:
            torch.testing.assert_close(expected, actual)


@pytest.mark.skipif(
    not is_native_int64_enabled(),
    reason="Test requires native int64 passthrough (TORCH_NEURONX_DISABLE_NATIVE_INT64=1)",
)
def test_to_copy_int64_small_values_to_float32():
    """Test int64 to float32 conversion with small values (no CPU bounce, compiler handles S64)."""
    src = torch.tensor([0, 1, -1, 42, -42], dtype=torch.int64, device="neuron")
    dst = None

    def do_copy():
        nonlocal dst
        dst = src.to(dtype=torch.float32, device="neuron")

    bounces = count_cpu_bounce_calls(do_copy)
    torch_neuronx.synchronize()
    assert dst.dtype == torch.float32
    assert dst.device.type == "neuron"
    expected = src.cpu().to(torch.float32)
    torch.testing.assert_close(dst.cpu(), expected)
    # int64 passes through to compiler natively (S64 HLO), no CPU bounce needed
    assert bounces["reads"] == 0, f"Expected 0 CPU reads, got {bounces['reads']}"
    assert bounces["writes"] == 0, f"Expected 0 CPU writes, got {bounces['writes']}"


@pytest.mark.skipif(
    not is_native_int64_enabled(),
    reason="Test requires native int64 passthrough (TORCH_NEURONX_DISABLE_NATIVE_INT64=1)",
)
def test_to_copy_int64_2d_small_values_to_float32():
    """Test int64 to float32 with 2D tensor (no CPU bounce, compiler handles S64)."""
    src = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64, device="neuron")
    dst = None

    def do_copy():
        nonlocal dst
        dst = src.to(dtype=torch.float32, device="neuron")

    bounces = count_cpu_bounce_calls(do_copy)
    torch_neuronx.synchronize()
    assert dst.dtype == torch.float32
    assert dst.shape == (2, 3)
    expected = src.cpu().to(torch.float32)
    torch.testing.assert_close(dst.cpu(), expected)
    # int64 passes through to compiler natively (S64 HLO), no CPU bounce needed
    assert bounces["reads"] == 0, f"Expected 0 CPU reads, got {bounces['reads']}"
    assert bounces["writes"] == 0, f"Expected 0 CPU writes, got {bounces['writes']}"
