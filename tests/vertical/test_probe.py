"""Unit tests for the FakeTensorMode output-shape probe used to size rooflines.

These are ordinary PyTorch-only tests — no torch_neuronx device required — so
they run fast and can guard against regressions in the shape-inference path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch.testing._internal.opinfo.core import SampleInput

sys.path.insert(0, str(Path(__file__).resolve().parent))
from internal.common import flatten_tensors, probe_output_meta


class TestFlattenTensors:
    def test_single_tensor(self):
        t = torch.randn(4, 4)
        assert flatten_tensors(t) == [t]

    def test_tuple_of_tensors(self):
        a, b = torch.randn(2), torch.randn(3)
        assert flatten_tensors((a, b)) == [a, b]

    def test_non_tensor_dropped(self):
        t = torch.randn(4)
        assert flatten_tensors((t, None, 42, "x")) == [t]

    def test_none_returns_empty(self):
        assert flatten_tensors(None) == []

    def test_non_tensor_list_returns_empty(self):
        assert flatten_tensors([{"not": "a tensor"}]) == []


class TestProbeOutputMeta:
    def test_mm(self):
        op = torch.ops.aten.mm.default
        sample = SampleInput(torch.randn(8, 16), args=(torch.randn(16, 32),))
        out = probe_output_meta(op, sample)
        assert len(out) == 1
        assert tuple(out[0].shape) == (8, 32)
        assert out[0].dtype == torch.float32

    def test_linear(self):
        op = torch.ops.aten.linear.default
        sample = SampleInput(
            torch.randn(4, 16),
            args=(torch.randn(32, 16),),
            kwargs={"bias": torch.randn(32)},
        )
        out = probe_output_meta(op, sample)
        assert len(out) == 1
        assert tuple(out[0].shape) == (4, 32)

    def test_sum_scalar(self):
        op = torch.ops.aten.sum.default
        sample = SampleInput(torch.randn(4, 8))
        out = probe_output_meta(op, sample)
        assert len(out) == 1
        assert tuple(out[0].shape) == ()

    def test_layer_norm_returns_input_shape(self):
        op = torch.ops.aten.layer_norm.default
        sample = SampleInput(
            torch.randn(2, 4, 16),
            args=([16], torch.randn(16), torch.randn(16)),
        )
        out = probe_output_meta(op, sample)
        assert len(out) >= 1
        assert tuple(out[0].shape) == (2, 4, 16)

    def test_unknown_op_returns_empty(self):
        class BrokenOp:
            def __call__(self, *a, **kw):
                raise RuntimeError("no such op")

        sample = SampleInput(torch.randn(4))
        assert probe_output_meta(BrokenOp(), sample) == []

    def test_returns_tensor_with_no_storage(self):
        """The probe must not allocate storage for outputs."""
        op = torch.ops.aten.mm.default
        # Large matmul — would be expensive if storage were allocated.
        sample = SampleInput(torch.randn(1024, 4096), args=(torch.randn(4096, 4096),))
        out = probe_output_meta(op, sample)
        assert len(out) == 1
        assert out[0].numel() == 1024 * 4096
        # FakeTensor (no storage) or meta tensor — both have zero storage.
        assert "Fake" in type(out[0]).__name__ or out[0].is_meta
