import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_did_not_run_on_neuron,
    track_neuron_ops,
)


def test_to_copy_cpu_to_neuron():
    """Test .to() for CPU to Neuron transfer"""
    cpu_tensor = torch.randn(4, 4)
    neuron_tensor = cpu_tensor.to("neuron:0")

    assert neuron_tensor.device.type == "neuron"
    assert neuron_tensor.device.index == 0
    assert neuron_tensor.shape == cpu_tensor.shape

    # Verify data
    back_to_cpu = neuron_tensor.to("cpu")
    torch.testing.assert_close(cpu_tensor, back_to_cpu)


def test_to_copy_neuron_to_cpu():
    """Test .to() for Neuron to CPU transfer"""
    neuron_tensor = torch.randn(4, 4).to("neuron:0")
    cpu_tensor = neuron_tensor.to("cpu")

    assert cpu_tensor.device.type == "cpu"
    assert cpu_tensor.shape == neuron_tensor.shape


def test_to_copy_dtype_conversion():
    """Test .to() with dtype conversion"""
    cpu_float = torch.randn(4, 4, dtype=torch.float32)
    neuron_int = cpu_float.to(device="neuron:0", dtype=torch.int32)

    assert neuron_int.dtype == torch.int32
    assert neuron_int.device.type == "neuron"

    # Verify conversion
    expected = cpu_float.to(torch.int32)
    result = neuron_int.to("cpu")
    torch.testing.assert_close(expected, result)


def test_to_copy_non_contiguous_cpu_to_neuron():
    """Test .to() handles non-contiguous CPU tensors"""
    # Create a non-contiguous CPU tensor
    cpu_tensor = torch.randn(4, 4).transpose(0, 1)
    assert not cpu_tensor.is_contiguous()

    # Should handle non-contiguous by making it contiguous internally
    neuron_tensor = cpu_tensor.to("neuron:0")

    assert neuron_tensor.device.type == "neuron"
    assert neuron_tensor.shape == cpu_tensor.shape

    # Verify data is correct
    back_to_cpu = neuron_tensor.to("cpu")
    torch.testing.assert_close(cpu_tensor, back_to_cpu)


def test_to_copy_no_op_same_device():
    """Test that .to() with same device returns same tensor"""
    cpu_tensor = torch.randn(4, 4)

    # Same device, dtype, layout - should return same object
    result = cpu_tensor.to(device="cpu")
    assert result is cpu_tensor

    # Same device but different dtype - should create new tensor
    result_int = cpu_tensor.to(device="cpu", dtype=torch.int32)
    assert result_int is not cpu_tensor
    assert result_int.dtype == torch.int32


@pytest.mark.xfail(reason="_to_copy does not yet propagate memory_format to output layout")
def test_to_copy_memory_format_channels_last():
    """Test that channels_last memory format is preserved for to() operations"""
    tensor = torch.randn(4, 4, 4, 4)

    result = tensor.to("neuron:0", memory_format=torch.channels_last)
    assert result.device.type == "neuron"
    assert result.is_contiguous(memory_format=torch.channels_last)


def test_to_copy_combined_dtype_and_device():
    """Test .to() with both dtype and device change"""
    cpu_float = torch.randn(4, 4, dtype=torch.float32)

    # Change both device and dtype to bfloat16
    neuron_bfloat = cpu_float.to(device="neuron:0", dtype=torch.bfloat16)

    assert neuron_bfloat.device.type == "neuron"
    assert neuron_bfloat.dtype == torch.bfloat16

    # Verify data
    expected = cpu_float.to(dtype=torch.bfloat16)
    result = neuron_bfloat.to("cpu")
    torch.testing.assert_close(expected, result)


def test_to_copy_empty_tensor():
    """Test .to() with empty tensors"""
    empty_cpu = torch.empty(0)
    empty_neuron = empty_cpu.to("neuron:0")

    assert empty_neuron.device.type == "neuron"
    assert empty_neuron.numel() == 0

    # And back
    back_to_cpu = empty_neuron.to("cpu")
    assert back_to_cpu.device.type == "cpu"
    assert back_to_cpu.numel() == 0


def test_to_copy_scalar_tensor():
    """Test .to() with scalar tensors"""
    scalar_cpu = torch.tensor(3.14)
    scalar_neuron = scalar_cpu.to("neuron:0")

    assert scalar_neuron.device.type == "neuron"
    assert scalar_neuron.dim() == 0

    # Verify value
    result = scalar_neuron.to("cpu")
    assert result.item() == pytest.approx(3.14)


def test_to_copy_preserves_strides_transpose():
    """Test that .to('neuron') preserves strides from transpose, matching CUDA semantics."""
    t = torch.randn(2, 4, 512, 512).transpose(1, 2)
    assert not t.is_contiguous()
    t_neu = t.to("neuron")
    assert (
        t_neu.stride() == t.stride()
    ), f"stride mismatch: neuron={t_neu.stride()} cpu={t.stride()}"
    torch.testing.assert_close(t_neu.cpu(), t)


def _make_tensor_with_stride(shape, dtype, pattern):
    """Create a CPU tensor with the given stride pattern and random data."""

    def _rand(s, dt):
        if dt == torch.bool:
            return torch.randint(0, 2, s, dtype=dt)
        if dt in (torch.int32,):
            return torch.randint(0, 4, s, dtype=dt)
        return torch.randn(s, dtype=dt)

    if pattern == "contiguous":
        return _rand(shape, dtype)
    elif pattern == "transpose":
        return _rand((shape[1], shape[0]), dtype).t()
    elif pattern == "slice_even_cols":
        return _rand((shape[0], shape[1] * 2), dtype)[:, 0::2]
    elif pattern == "slice_even_rows":
        return _rand((shape[0] * 2, shape[1]), dtype)[0::2, :]
    elif pattern == "as_strided_gaps":
        base = _rand((shape[0], shape[1] * 2), dtype)
        return base.as_strided(shape, (shape[1] * 2, 1))
    elif pattern == "chunk":
        return _rand((shape[0], shape[1] * 3), dtype).chunk(3, dim=-1)[1]


@pytest.mark.parametrize(
    "stride_pattern",
    ["transpose", "slice_even_cols", "slice_even_rows", "as_strided_gaps", "chunk"],
)
def test_to_cpu_preserves_strides(stride_pattern):
    """Test that neuron .cpu() produces correct strides matching PyTorch semantics.

    Non-overlapping-and-dense (e.g. transpose): strides preserved.
    Others (e.g. slices with gaps): output is contiguous.
    """
    shape = (64, 64)
    t_cpu = _make_tensor_with_stride(shape, torch.float32, stride_pattern)
    t_neuron = t_cpu.to("neuron")
    result = t_neuron.cpu()

    if torch.ops.aten.is_non_overlapping_and_dense(t_cpu):
        expected_stride = t_cpu.stride()
    else:
        expected_stride = torch.empty(t_cpu.shape).stride()

    assert (
        result.stride() == expected_stride
    ), f"stride mismatch: got {result.stride()}, expected {expected_stride}"
    torch.testing.assert_close(result, t_cpu)


_COPY_DTYPES = [
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.bool,
]
_NO_CPU_BOUNCE_PATTERNS = {"contiguous", "transpose"}


@pytest.mark.parametrize("src_dtype", _COPY_DTYPES, ids=[str(d) for d in _COPY_DTYPES])
@pytest.mark.parametrize("dst_dtype", _COPY_DTYPES, ids=[str(d) for d in _COPY_DTYPES])
@pytest.mark.parametrize(
    "stride_pattern",
    ["contiguous", "transpose", "slice_even_cols", "slice_even_rows", "as_strided_gaps", "chunk"],
)
def test_copy_neuron_to_neuron_dtype_cast_no_cpu_bounce(src_dtype, dst_dtype, stride_pattern):
    """copy_ between Neuron tensors with dtype mismatch and non-contiguous dst
    must cast on-device without CPU bounce for supported patterns, and produce
    correct results for all patterns."""
    shape = (64, 64)
    src_cpu = _make_tensor_with_stride(shape, src_dtype, "contiguous")
    dst_cpu = _make_tensor_with_stride(shape, dst_dtype, stride_pattern)

    src_neuron = src_cpu.to("neuron")
    dst_neuron = dst_cpu.to("neuron")

    with track_neuron_ops():
        dst_neuron.copy_(src_neuron)

    if stride_pattern in _NO_CPU_BOUNCE_PATTERNS:
        assert_op_did_not_run_on_neuron("neuron::copy::neuron_to_cpu")

    # CPU reference — always check accuracy
    dst_cpu.copy_(src_cpu)
    assert (
        dst_neuron.stride() == dst_cpu.stride()
    ), f"stride mismatch: neuron={dst_neuron.stride()} cpu={dst_cpu.stride()}"
    torch.testing.assert_close(dst_neuron.cpu(), dst_cpu)


@pytest.mark.parametrize("src_dtype", _COPY_DTYPES, ids=[str(d) for d in _COPY_DTYPES])
@pytest.mark.parametrize("dst_dtype", _COPY_DTYPES, ids=[str(d) for d in _COPY_DTYPES])
@pytest.mark.parametrize(
    "stride_pattern",
    ["contiguous", "transpose", "slice_even_cols", "slice_even_rows", "as_strided_gaps", "chunk"],
)
def test_to_dtype_neuron_no_cpu_bounce(src_dtype, dst_dtype, stride_pattern):
    """Neuron .to(dtype) with non-contiguous source must cast on-device without CPU bounce
    for supported patterns, and produce correct results for all patterns."""
    shape = (64, 64)
    src_cpu = _make_tensor_with_stride(shape, src_dtype, stride_pattern)

    src_neuron = src_cpu.to("neuron")

    with track_neuron_ops():
        result_neuron = src_neuron.to(dst_dtype)

    if stride_pattern in _NO_CPU_BOUNCE_PATTERNS:
        assert_op_did_not_run_on_neuron("neuron::copy::neuron_to_cpu")
    assert result_neuron.dtype == dst_dtype

    # CPU reference — always check accuracy
    expected = src_cpu.to(dst_dtype)
    assert (
        result_neuron.stride() == expected.stride()
    ), f"stride mismatch: neuron={result_neuron.stride()} cpu={expected.stride()}"
    torch.testing.assert_close(result_neuron.cpu(), expected)


@pytest.mark.parametrize("dtype", _COPY_DTYPES, ids=[str(d) for d in _COPY_DTYPES])
@pytest.mark.parametrize(
    "src_shape,dst_base_shape,dst_slice",
    [
        ((5,), (1, 5, 5, 3), (slice(None), slice(None), slice(None), 0)),
        ((5, 1), (1, 5, 5, 3), (slice(None), slice(None), slice(None), 0)),
        ((5,), (5, 3), (slice(None), 0)),
    ],
    ids=["1d_into_3d_slice", "2d_into_3d_slice", "1d_into_2d_slice"],
)
def test_copy_broadcast_into_non_contiguous_dst(dtype, src_shape, dst_base_shape, dst_slice):
    """copy_ with broadcast src into a non-contiguous dst (slice of a larger tensor)."""
    if dtype in (torch.bool, torch.int32):
        src_cpu = torch.randint(0, 2, src_shape, dtype=dtype)
        base_cpu = torch.randint(0, 2, dst_base_shape, dtype=dtype)
    else:
        src_cpu = torch.randn(src_shape, dtype=dtype)
        base_cpu = torch.randn(dst_base_shape, dtype=dtype)

    dst_cpu = base_cpu[dst_slice]
    base_neuron = base_cpu.to("neuron")
    dst_neuron = base_neuron[dst_slice]
    src_neuron = src_cpu.to("neuron")

    dst_neuron.copy_(src_neuron)
    dst_cpu.copy_(src_cpu)

    torch.testing.assert_close(dst_neuron.cpu(), dst_cpu)
