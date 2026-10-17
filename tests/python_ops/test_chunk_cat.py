"""aten::_chunk_cat on Neuron: one _foreach_copy_ over chunk segments, matching the CPU op."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

# FSDP v2 reduce-scatter copy-in of one Qwen3 layer under TP4/FSDP16, scaled down 1/40 per
# dim: fused qkv, wo, q_norm, k_norm, w1, w2, w3, attention_norm, ffn_norm.
FSDP_SHAPES = [
    (64, 128),
    (128, 48),
    (128,),
    (128,),
    (160, 128),
    (128, 160),
    (160, 128),
    (128,),
    (128,),
]


def _reference(tensors, dim, num_chunks, out_dtype=None):
    cpu = [t.cpu() for t in tensors]
    ref = torch._chunk_cat(cpu, dim, num_chunks)
    return ref if out_dtype is None else ref.to(out_dtype)


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
def test_fsdp_copy_in_matches_cpu(out_dtype):
    """Divisible bf16 grads into a [num_chunks, width] fp32/bf16 out (the FSDP v2 case)."""
    num_chunks = 16
    grads = [torch.randn(s, dtype=torch.bfloat16, device="neuron") for s in FSDP_SHAPES]
    ref = _reference(grads, 0, num_chunks, out_dtype)
    out = torch.empty(ref.shape, dtype=out_dtype, device="neuron")
    with track_neuron_ops():
        torch._chunk_cat(grads, 0, num_chunks, out=out)
        assert_op_runs_on_neuron("aten::_foreach_copy_")
        assert "aten::cat.out" not in torch_neuronx.get_executed_ops()
    torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)


def test_padding_is_zero_filled():
    """size(dim) that is not a multiple of num_chunks is zero-padded per chunk."""
    num_chunks = 4
    tensors = [torch.randn(s, device="neuron") for s in [(7,), (5, 3), (2,), (130,)]]
    out = torch._chunk_cat(tensors, 0, num_chunks)
    assert out.shape == (num_chunks, 2 + 2 * 3 + 1 + 33)  # size(dim) pads to 8, 8, 4, 132
    torch.testing.assert_close(out.cpu(), _reference(tensors, 0, num_chunks), rtol=0, atol=0)


def test_leading_dims_and_mixed_ndims():
    """dim > 0 keeps the shared leading dims; tensors may differ in ndim past dim."""
    num_chunks = 3
    tensors = [torch.randn(2, 5, 3, device="neuron"), torch.randn(2, 7, device="neuron")]
    out = torch._chunk_cat(tensors, 1, num_chunks)
    assert out.shape == (2, num_chunks, 2 * 3 + 3)  # size(dim) pads to 6 and 9
    torch.testing.assert_close(out.cpu(), _reference(tensors, 1, num_chunks), rtol=0, atol=0)


def test_negative_dim_and_noncontiguous_inputs():
    """Regression: strided sources plus the repeated zeros source read wrong in _foreach_copy_."""
    num_chunks = 2
    base = torch.randn(4, 6, device="neuron")
    tensors = [base[:, ::2], base.t()[:3].t()]  # (4, 3) strided, (4, 3) strided
    out = torch._chunk_cat(tensors, -1, num_chunks)
    assert out.shape == (4, num_chunks, 2 + 2)
    torch.testing.assert_close(out.cpu(), _reference(tensors, -1, num_chunks), rtol=0, atol=0)


def test_out_with_wrong_shape_or_strides_falls_back():
    """cat(out=) resizes a mis-sized out and writes a strided one; both keep working."""
    num_chunks = 2
    tensors = [torch.randn(8, device="neuron"), torch.randn(4, device="neuron")]
    ref = _reference(tensors, 0, num_chunks)

    wrong = torch.empty(1, device="neuron")
    torch._chunk_cat(tensors, 0, num_chunks, out=wrong)
    torch.testing.assert_close(wrong.cpu(), ref, rtol=0, atol=0)

    strided = torch.empty(num_chunks, 12, device="neuron")[:, ::2]
    torch._chunk_cat(tensors, 0, num_chunks, out=strided)
    torch.testing.assert_close(strided.cpu(), ref, rtol=0, atol=0)


def test_many_segments_fall_back_to_composite(monkeypatch):
    """A segment count over the fast-path cap takes PyTorch's composite cat path."""
    from torch_neuronx.python_ops import chunk_cat

    monkeypatch.setattr(chunk_cat, "_MAX_FAST_PATH_SEGMENTS", 8)
    num_chunks = 4
    tensors = [torch.randn(3, 8, device="neuron") for _ in range(2)]  # 3 * 4 * 2 = 24 segments
    with track_neuron_ops():
        out = torch._chunk_cat(tensors, 1, num_chunks)
        assert_op_runs_on_neuron("aten::cat")
        assert "aten::_foreach_copy_" not in torch_neuronx.get_executed_ops()
    torch.testing.assert_close(out.cpu(), _reference(tensors, 1, num_chunks), rtol=0, atol=0)


def test_empty_list_with_out_raises_like_cpu():
    """PyTorch's input checks run before the out checks: an empty list raises the upstream error."""
    for device in ["neuron", "cpu"]:
        with pytest.raises(RuntimeError, match="non-empty input tensor list"):
            torch._chunk_cat([], 0, 2, out=torch.empty(2, 4, device=device))


def test_mixed_dtypes_raise_like_cpu():
    """PyTorch's input checks run first: mixed input dtypes raise the upstream error."""
    tensors = [torch.randn(8, device="neuron"), torch.randn(8, device="neuron").to(torch.bfloat16)]
    with pytest.raises(RuntimeError, match="same dtype"):
        torch._chunk_cat(tensors, 0, 2)
    with pytest.raises(RuntimeError, match="same dtype"):
        torch._chunk_cat([t.cpu() for t in tensors], 0, 2)
