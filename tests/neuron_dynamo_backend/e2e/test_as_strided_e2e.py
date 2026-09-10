"""E2E tests for as_strided / as_strided_scatter through torch.compile(backend='neuron')."""

import math

import pytest
import torch

# Reads over a flat buffer. (size, stride, storage_offset)
READ_CASES = [
    pytest.param((2, 2, 8, 8, 2), (256, 128, 16, 2, 1), 0, id="kv-cache-rank5-contiguous"),
    pytest.param((5, 3), (3, 1), 17, id="row-major-at-offset"),
    pytest.param((3, 4), (1, 3), 0, id="transpose-gathers"),
    pytest.param((2, 2), (5, 1), 0, id="gappy-gathers"),
]

# Writes into a flat buffer. (size, stride, storage_offset, storage)
WRITE_CASES = [
    pytest.param((2, 4), (4, 1), 0, 16, id="suffix-only"),
    pytest.param((2, 4), (4, 1), 4, 16, id="prefix-and-suffix"),
    pytest.param((4, 4), (4, 1), 0, 16, id="whole-buffer"),
    pytest.param((2, 2), (5, 1), 0, 16, id="gappy-index-copies"),
]


def _storage_needed(size, stride, offset):
    """Elements the view addresses: its highest reachable index, plus one."""
    return offset + sum((s - 1) * t for s, t in zip(size, stride, strict=False)) + 1


class TestAsStridedE2E:
    """as_strided and as_strided_scatter must match CPU through the neuron backend."""

    @pytest.mark.parametrize("size,stride,offset", READ_CASES)
    def test_read(self, size, stride, offset):
        torch.manual_seed(0)
        x = torch.randn(_storage_needed(size, stride, offset), dtype=torch.float32)

        def fn(a):
            # The add keeps a bare view out of the graph outputs.
            return torch.as_strided(a, size, stride, offset) + 1.0

        with torch.no_grad():
            cpu_out = fn(x)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out)

    def test_read_bfloat16(self):
        """The fast path is a reinterpretation, so it must not disturb a narrower dtype."""
        torch.manual_seed(0)
        size, stride, offset = (5, 3), (3, 1), 17
        x = torch.randn(_storage_needed(size, stride, offset), dtype=torch.bfloat16)

        def fn(a):
            return torch.as_strided(a, size, stride, offset) + 1.0

        with torch.no_grad():
            cpu_out = fn(x)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=5e-2, atol=5e-2)

    @pytest.mark.parametrize("size,stride,offset,storage", WRITE_CASES)
    def test_write(self, size, stride, offset, storage):
        torch.manual_seed(0)
        x = torch.randn(storage, dtype=torch.float32)
        src = torch.arange(1, math.prod(size) + 1, dtype=torch.float32).reshape(size)

        def fn(a, s):
            return torch.as_strided_scatter(a, s, size, stride, offset)

        with torch.no_grad():
            cpu_out = fn(x, src)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"), src.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out)

    def test_write_into_non_contiguous_destination(self):
        """A transposed destination is addressed in storage order, not logical order."""
        torch.manual_seed(0)
        x = torch.randn(16, dtype=torch.float32)
        src = torch.arange(1, 9, dtype=torch.float32).reshape(2, 4)

        def fn(a, s):
            return torch.as_strided_scatter(a.reshape(4, 4).t(), s, (2, 4), (4, 1), 4)

        with torch.no_grad():
            cpu_out = fn(x, src)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"), src.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out)

    def test_read_from_non_contiguous_input(self):
        """The read side addresses storage too, so its flatten must match CPU's."""
        torch.manual_seed(0)
        x = torch.randn(24, dtype=torch.float32)

        def fn(a):
            return torch.as_strided(a.reshape(2, 3, 4).permute(2, 0, 1), (2, 3), (3, 1), 0) + 1.0

        with torch.no_grad():
            cpu_out = fn(x)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out)

    @pytest.mark.parametrize(
        "size,stride",
        [
            pytest.param((2, 3), (3, 1), id="default-stride"),
            pytest.param((2, 3), (1, 2), id="transpose-stride"),
            pytest.param((4, 5), (1, 4), id="col-major-stride"),
        ],
    )
    def test_new_empty_strided_decomposes(self, size, stride):
        """new_empty_strided returns uninitialized memory; a following copy_
        makes the result observable. Torch-mlir must decompose the op for any
        stride, or StableHLO legalization fails on the surviving stride list."""
        torch.manual_seed(0)
        x = torch.randn(size, dtype=torch.float32)

        def fn(a):
            y = a.new_empty_strided(size, stride)
            y.copy_(a)
            return y + 1.0

        cpu_out = fn(x)
        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        neuron_out = compiled_fn(x.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out)

    @pytest.mark.parametrize(
        "size,stride",
        [
            pytest.param((2, 3), (3, 1), id="default-stride"),
            pytest.param((2, 3), (1, 2), id="transpose-stride"),
            pytest.param((4, 5), (1, 4), id="col-major-stride"),
        ],
    )
    def test_empty_strided_decomposes(self, size, stride):
        """empty_strided is the factory form of new_empty_strided; same
        uninitialized-memory reasoning applies."""
        torch.manual_seed(0)
        x = torch.randn(size, dtype=torch.float32)

        def fn(a):
            y = torch.empty_strided(size, stride, dtype=torch.float32, device=a.device)
            y.copy_(a)
            return y + 1.0

        cpu_out = fn(x)
        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        neuron_out = compiled_fn(x.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out)
