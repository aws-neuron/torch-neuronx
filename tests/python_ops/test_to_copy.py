import os
import time

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
            # The neuron->neuron cast is serviced entirely by aten::_to_copy.
            # The previous aten::copy_ assertion only passed because randn fell back to cpu.
            assert_op_runs_on_neuron("aten::_to_copy")


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

    src = torch.randn(4, 4, dtype=torch.float64, device="neuron")

    impl = ToCopyMLIRImpl()
    assert not impl.can_handle(src, dtype=torch.float32, device="neuron")

    dst = src.to(dtype=torch.float32, device="neuron")
    torch_neuronx.synchronize()
    assert dst.device.type == "neuron"
    assert dst.dtype == torch.float32
    torch.testing.assert_close(src.cpu().to(torch.float32), dst.cpu())


def test_to_copy_fp32_to_fp64():
    """Test neuron fp32 to neuron fp64 - MLIR impl should reject float64"""
    from torch_neuronx.python_ops.torch_mlir.ops.to_copy import ToCopyMLIRImpl

    src = torch.randn(4, 4, dtype=torch.float32, device="neuron")

    impl = ToCopyMLIRImpl()
    assert not impl.can_handle(src, dtype=torch.float64, device="neuron")

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
    mlir_unsupported = {torch.float64, torch.int64, torch.uint64}
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


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.bfloat16, torch.int32, torch.int64, torch.bool],
)
def test_to_copy_neuron_channels_last_same_dtype(dtype):
    """_to_copy with memory_format=channels_last must preserve dtype, values, and layout.

    Regression for a packed-int64 bug: int64->int64 with channels_last went
    through the MLIR pipeline (which cannot carry int64) and silently came back
    as int32. The MLIR _to_copy path also drops the channels_last layout for
    *all* dtypes (returns contiguous), so this also guards the layout for the
    supported dtypes. Mirrors how aten::contiguous(channels_last) is handled
    (NRT strided copy), not the MLIR path.
    """
    # 4D tensor so channels_last is meaningful; input is contiguous (NCHW).
    if dtype == torch.bool:
        src = torch.randint(0, 2, (2, 3, 4, 5), device="neuron").bool()
    elif dtype in (torch.float32, torch.bfloat16):
        src = torch.randn(2, 3, 4, 5, dtype=dtype, device="neuron")
    else:
        src = torch.randint(0, 100, (2, 3, 4, 5), dtype=dtype, device="neuron")

    dst = src.to(dtype=dtype, memory_format=torch.channels_last)
    torch_neuronx.synchronize()

    expected = src.cpu().to(dtype=dtype, memory_format=torch.channels_last)
    assert dst.device.type == "neuron"
    # dtype must be preserved (the int64->int32 corruption this guards against)
    assert dst.dtype == dtype, f"dtype changed: {dst.dtype} != {dtype}"
    assert dst.shape == src.shape
    # layout must honor channels_last (MLIR path silently returns contiguous)
    assert dst.cpu().is_contiguous(
        memory_format=torch.channels_last
    ), f"expected channels_last layout, got strides {dst.cpu().stride()}"
    # values must be correct
    assert torch.equal(dst.cpu(), expected)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    [
        (torch.float32, torch.int64),  # → packed int64, + layout
        (torch.int64, torch.float32),  # int64 source, + layout
        (torch.int64, torch.int32),  # int64 → int32, + layout
        (torch.float32, torch.int32),  # supported → supported, + layout
        (torch.int32, torch.float32),
    ],
)
def test_to_copy_neuron_channels_last_dtype_change(src_dtype, dst_dtype):
    """_to_copy with BOTH a dtype change and memory_format=channels_last.

    CPU honors channels_last regardless of any dtype change, so Neuron must too.
    The dtype conversion (incl. packed int64 materialize/extract) runs first,
    then the channels_last layout is applied — see DispatchToCopyImpl Phase 1/2.
    """
    # 4D, contiguous source (NCHW).
    if src_dtype in (torch.float32, torch.bfloat16):
        src = torch.randn(2, 3, 4, 5, dtype=src_dtype, device="neuron")
    else:
        # small magnitudes so float→int and packed-int64 (low 32 bits) match CPU
        src = torch.randint(0, 50, (2, 3, 4, 5), dtype=src_dtype, device="neuron")

    dst = src.to(dtype=dst_dtype, memory_format=torch.channels_last)
    torch_neuronx.synchronize()

    expected = src.cpu().to(dtype=dst_dtype, memory_format=torch.channels_last)
    assert dst.device.type == "neuron"
    assert dst.dtype == dst_dtype, f"dtype: {dst.dtype} != {dst_dtype}"
    assert dst.shape == src.shape
    assert dst.cpu().is_contiguous(
        memory_format=torch.channels_last
    ), f"expected channels_last layout, got strides {dst.cpu().stride()}"
    if dst_dtype in (torch.int32, torch.int64):
        assert torch.equal(dst.cpu(), expected)
    else:
        torch.testing.assert_close(dst.cpu(), expected)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
@pytest.mark.parametrize("src_dtype", [torch.float32, torch.float64])
def test_to_copy_float_to_int64_beyond_int32_range(src_dtype):
    """float→int64 must preserve values outside int32 range (full 64 bits).

    Regression: float→int64 used to go float→int32→materialize, which saturates
    at int32 range (e.g. 2**40 → -2147483648). packed int64 storage holds both
    lanes, so the conversion must keep full precision (via CPU bounce), matching
    CPU. Only *compute* on packed int64 is limited to the low 32 bits.
    """
    vals = [2.0**40, -(2.0**40), 3.0e9, -3.0e9, 1.0, -1.0, 0.0, 16777217.0]
    src = torch.tensor(vals, dtype=src_dtype, device="neuron")
    dst = src.to(torch.int64)
    torch_neuronx.synchronize()
    expected = src.cpu().to(torch.int64)
    assert dst.dtype == torch.int64
    assert torch.equal(
        dst.cpu(), expected
    ), f"{src_dtype}→int64 mismatch: got {dst.cpu().tolist()} exp {expected.tolist()}"


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
@pytest.mark.parametrize("trigger", ["n2n_preserve", "n2n_contiguous_format", "cpu_to_neuron"])
def test_to_copy_noncontiguous_strided_cast_is_fast(trigger):
    """_to_copy of a non-contiguous (transposed) view must stay fast."""
    shape = (1, 24, 2048, 256)  # transposed to [1,24,256,2048] like an attention key

    def do_cast():
        if trigger == "n2n_preserve":
            # Neuron->Neuron cross-dtype cast, MemoryFormat::Preserve (autocast path).
            src = torch.randn(*shape, dtype=torch.float32, device="neuron").transpose(-1, -2)
            return src, src.to(torch.bfloat16)
        elif trigger == "n2n_contiguous_format":
            # Same-dtype + contiguous_format materialize. High-level .to() no-ops
            # same-dtype, so drive the dispatcher directly (as functionalization does).
            src = torch.randn(*shape, dtype=torch.float32, device="neuron").transpose(-1, -2)
            out = torch.ops.aten._to_copy(
                src, dtype=torch.float32, memory_format=torch.contiguous_format
            )
            return src, out
        else:  # cpu_to_neuron
            # CPU non-contiguous source -> Neuron with dtype change (Preserve strides).
            src = torch.randn(*shape, dtype=torch.float32).transpose(-1, -2)
            return src, src.to(device="neuron", dtype=torch.bfloat16)

    # Warm up one-time compilation; the per-element regression is NOT
    # compile-cacheable, so it stays ~90s even when warm.
    _, _ = do_cast()
    torch_neuronx.synchronize()

    t0 = time.time()
    src, dst = do_cast()
    torch_neuronx.synchronize()
    elapsed = time.time() - t0

    assert dst.shape == src.shape
    expected = src.cpu().to(dst.dtype)
    torch.testing.assert_close(dst.cpu(), expected)
    assert elapsed < 1.0, (
        f"[{trigger}] strided cast took {elapsed:.2f}s "
        f"(regressed to per-element nrt copies?); expected ~60ms"
    )


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
@pytest.mark.parametrize("path", ["cpu_to_neuron", "n2n_same_dtype", "n2n_cross_dtype"])
def test_to_copy_expand_zero_stride_is_correct(path):
    """_to_copy of an expanded (stride-0 / overlapping) view must be correct."""
    # [1,1,512,512] -> expand to [2,1,512,512]: dim 0 stride 0 (overlapping).
    if path == "cpu_to_neuron":
        base = torch.randn(1, 1, 8, 8, dtype=torch.float32)
        src = base.expand(2, 1, 8, 8)
        dst = src.to("neuron")  # CPU -> Neuron, same dtype, Preserve
        expected = src  # value/stride reference on CPU
    elif path == "n2n_same_dtype":
        base = torch.randn(1, 1, 8, 8, dtype=torch.float32, device="neuron")
        src = base.expand(2, 1, 8, 8)
        dst = torch.ops.aten._to_copy(src, dtype=torch.float32)  # same dtype, Preserve
        expected = src.cpu()
    else:  # n2n_cross_dtype
        base = torch.randn(1, 1, 8, 8, dtype=torch.float32, device="neuron")
        src = base.expand(2, 1, 8, 8)
        dst = src.to(torch.bfloat16)  # cross dtype, Preserve
        expected = src.cpu().to(torch.bfloat16)

    torch_neuronx.synchronize()
    assert dst.device.type == "neuron"
    assert dst.shape == expected.shape
    torch.testing.assert_close(dst.cpu(), expected)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
def test_to_copy_complex_strided_does_not_redispatch():
    """Strided complex _to_copy must use the byte-wise path, not aten::copy_.

    Fence for the linalg_eig regression: redispatching a complex strided copy
    hits aten::contiguous (unsupported for complex; hard error under
    TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1, silent CPU fallback
    otherwise — hence the op-tracking check).
    """
    # F-contiguous complex source, like LAPACK eigenvector output: strides (1, 5).
    src = torch.randn(5, 5, dtype=torch.complex64).t()

    with track_neuron_ops():
        dst = src.to("neuron")
        torch_neuronx.synchronize()
        executed = torch_neuronx.get_executed_ops()
        fallback = torch_neuronx.get_fallback_ops()

    assert "aten::contiguous" not in executed and "aten::contiguous" not in fallback, (
        "complex strided _to_copy redispatched into aten::contiguous "
        f"(executed={executed}, fallback={fallback})"
    )
    assert dst.device.type == "neuron"
    assert dst.stride() == src.stride()
    torch.testing.assert_close(dst.cpu(), src)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
@pytest.mark.skipif(
    is_native_int64_enabled(),
    reason="Packed int64 path; native int64 passes through to the compiler (different path).",
)
@pytest.mark.parametrize("n", [64, 1026])
def test_noncontiguous_packed_int64_downcast(n):
    """Non-contiguous packed int64 must contiguify correctly on the downcast path."""
    big = torch.arange(2 * n, device="neuron", dtype=torch.int64)
    t = big[::2]
    assert not t.is_contiguous()
    t.masked_fill_(t > 4, 1)

    big_ref = torch.arange(2 * n, dtype=torch.int64)
    t_ref = big_ref[::2]
    t_ref.masked_fill_(t_ref > 4, 1)

    # The view holds the correct logical values ...
    assert torch.equal(t.cpu(), t_ref)
    # ... and only the sliced (even) positions of the backing storage changed.
    assert torch.equal(big.cpu(), big_ref)
