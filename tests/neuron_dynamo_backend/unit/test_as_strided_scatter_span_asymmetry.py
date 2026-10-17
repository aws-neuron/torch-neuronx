"""as_strided_scatter writes whose untouched span is much shorter than the written one.

A short untouched remainder is what makes the backend shard on it, so these cover the
asymmetry at rank 1 and 2 in both directions, asserting on values only. CPU-only.
"""

import pytest
import torch

from torch_neuronx.neuron_dynamo_backend import decompositions

# (storage, size, stride, storage_offset, label)
ASYMMETRIC = [
    (3072, (2880,), (1,), 0, "192-element tail"),
    (3072, (2880,), (1,), 192, "192-element head"),
    (3072, (15, 192), (192, 1), 0, "rank 2, 192-element tail"),
    (3072, (2, 1440), (1440, 1), 192, "rank 2, 192-element head"),
    (3072, (2688,), (1,), 192, "192 on both sides"),
    (3072, (3072,), (1,), 0, "whole buffer, nothing to mask"),
    (3072, (1,), (1,), 3071, "single element at the very end"),
    (3072, (1,), (1,), 0, "single element at the very start"),
]


@pytest.mark.parametrize(
    "storage,size,stride,offset,label", ASYMMETRIC, ids=[c[4] for c in ASYMMETRIC]
)
def test_asymmetric_span_write_matches_reference(storage, size, stride, offset, label):
    buf = torch.arange(storage, dtype=torch.float32)
    src = torch.full(size, -7.0)

    expected = torch.ops.aten.as_strided_scatter.default(buf, src, size, stride, offset)
    got = decompositions.as_strided_scatter_decomp(buf, src, list(size), list(stride), offset)

    assert got.shape == expected.shape, label
    assert torch.equal(got, expected), label
    assert got.data_ptr() != buf.data_ptr(), f"{label}: output aliases buf"


@pytest.mark.parametrize(
    "storage,size,stride,offset,label", ASYMMETRIC, ids=[c[4] for c in ASYMMETRIC]
)
def test_asymmetric_span_write_leaves_the_rest_untouched(storage, size, stride, offset, label):
    """The masked-in region is exactly the view; everything outside it keeps its value."""
    buf = torch.arange(storage, dtype=torch.float32)
    src = torch.full(size, -7.0)
    numel = torch.Size(size).numel()

    got = decompositions.as_strided_scatter_decomp(buf, src, list(size), list(stride), offset)

    assert torch.equal(got[:offset], buf[:offset]), f"{label}: head changed"
    assert torch.equal(got[offset + numel :], buf[offset + numel :]), f"{label}: tail changed"
    assert torch.all(got[offset : offset + numel] == -7.0), f"{label}: written region wrong"


def test_short_span_write_casts_src_to_destination_dtype():
    """A bf16 destination keeps its dtype when src is fp32."""
    buf = torch.zeros(3072, dtype=torch.bfloat16)
    src = torch.full((2880,), 3.0, dtype=torch.float32)

    expected = torch.ops.aten.as_strided_scatter.default(
        buf, src.to(torch.bfloat16), [2880], [1], 0
    )
    got = decompositions.as_strided_scatter_decomp(buf, src, [2880], [1], 0)

    assert got.dtype == expected.dtype == torch.bfloat16
    assert torch.equal(got, expected)


# Views that are not one row-major span, so the write must keep index_copy.
# (size, stride, offset, label)
NON_CONTIGUOUS = [
    ((1440,), (2,), 0, "step 2 over the buffer"),
    ((15, 192), (1, 15), 0, "transposed rank 2"),
    ((16,), (192,), 0, "gappy stride"),
]


@pytest.mark.parametrize(
    "size,stride,offset,label", NON_CONTIGUOUS, ids=[c[3] for c in NON_CONTIGUOUS]
)
def test_non_contiguous_view_still_falls_back(size, stride, offset, label):
    """Only a row-major span may take the pad+select path; the rest keep index_copy."""
    buf = torch.arange(3072, dtype=torch.float32)
    src = torch.full(size, -7.0)

    expected = torch.ops.aten.as_strided_scatter.default(buf, src, list(size), list(stride), offset)
    got = decompositions.as_strided_scatter_decomp(buf, src, list(size), list(stride), offset)

    assert torch.equal(got, expected), label
