"""E2E tests for slice_scatter through torch.compile(backend='neuron')."""

import pytest
import torch

# Contiguous writes the decomposition must serve. (input_shape, dim, start, end)
CONTIGUOUS_WRITES = [
    pytest.param((4, 3072), -1, 0, 2880, id="2880-of-3072"),
    pytest.param((4, 3072), -1, 2880, None, id="tail-after-2880"),
    pytest.param((4, 3072), -1, 64, 2944, id="prefix-and-suffix-of-3072"),
    pytest.param((8,), 0, 2, 6, id="rank1-middle-span"),
    pytest.param((8,), 0, 3, None, id="suffix"),
    pytest.param((8,), 0, None, 4, id="prefix"),
    pytest.param((8,), 0, -4, None, id="negative-start"),
    pytest.param((2, 3, 4), 1, 1, 3, id="middle-dim-of-rank3"),
    pytest.param((4, 8), -1, 0, 8, id="whole-dim"),
]


def _src_like(inp, dim, start, end, step=1):
    """src must have exactly the extent aten expects; aten does not broadcast it."""
    extent = torch.ops.aten.slice.Tensor(inp, dim, start, end, step)
    return torch.arange(1, extent.numel() + 1, dtype=inp.dtype).reshape(extent.shape)


class TestSliceScatterE2E:
    """slice_scatter must match CPU through the neuron backend, decomposed or not."""

    @pytest.mark.parametrize("shape,dim,start,end", CONTIGUOUS_WRITES)
    def test_contiguous_write(self, shape, dim, start, end):
        torch.manual_seed(0)
        x = torch.randn(shape, dtype=torch.float32)
        src = _src_like(x, dim, start, end)

        def fn(a, s):
            return torch.ops.aten.slice_scatter.default(a, s, dim, start, end, 1)

        with torch.no_grad():
            cpu_out = fn(x, src)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"), src.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out)

    def test_contiguous_write_bfloat16(self):
        """A bf16 destination must round-trip through the padded copy and the mask."""
        torch.manual_seed(0)
        x = torch.randn(4, 3072, dtype=torch.bfloat16)
        src = _src_like(x, -1, 0, 2880)

        def fn(a, s):
            return torch.ops.aten.slice_scatter.default(a, s, -1, 0, 2880, 1)

        with torch.no_grad():
            cpu_out = fn(x, src)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"), src.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=5e-2, atol=5e-2)

    def test_strided_write_keeps_the_scatter(self):
        """step != 1 is declined, so this exercises the scatter lowering instead."""
        torch.manual_seed(0)
        x = torch.randn(8, dtype=torch.float32)
        src = _src_like(x, 0, 0, 6, 2)

        def fn(a, s):
            return torch.ops.aten.slice_scatter.default(a, s, 0, 0, 6, 2)

        with torch.no_grad():
            cpu_out = fn(x, src)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"), src.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out)

    def test_inplace_slice_write(self):
        """The in-place form functionalization produces, rather than a direct op call."""
        torch.manual_seed(0)
        x = torch.randn(4, 3072, dtype=torch.float32)
        src = torch.randn(4, 2880, dtype=torch.float32)

        def fn(a, s):
            out = a.clone()
            out[..., :2880] = s
            return out

        with torch.no_grad():
            cpu_out = fn(x, src)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"), src.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out)
