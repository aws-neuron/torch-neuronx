import neuronxcc.nki.typing as nt
import pytest
import torch
from neuronxcc import nki

from torch_neuronx import wrap_nki


def _make_identity_kernel():
    """Create a simple load/store kernel that copies x -> y."""

    @wrap_nki
    @nki.jit
    def copy_kernel(x, y: nt.mutable_tensor):
        import neuronxcc.nki.language as nl

        x_tile = nl.load(x)
        nl.store(y, value=x_tile)
        return y

    return copy_kernel


def test_slice_input_execution_succeeds_with_view_offset():
    """Pass a non-zero-offset slice to an NKI kernel and expect success.

    Setup: base has 10 elements; view takes [2:7] -> shape 5, offset 2.
    Correct behavior: allocate an NRT slice of exactly 5 elements at the offset,
    execute, and match expected values.
    """
    device = torch.device("neuron", 0)
    base = torch.arange(10, dtype=torch.float32, device=device)
    view = base[2:7]  # shape=5, storage_offset=2
    out = torch.empty_like(view)

    copy_kernel = _make_identity_kernel()
    copy_kernel(view, out)

    expected = torch.arange(2, 7, dtype=torch.float32)
    assert torch.equal(out.to("cpu"), expected)


def test_contiguous_input_execution_succeeds():
    """Control test: using a contiguous tensor (not a view)"""
    device = torch.device("neuron", 0)
    base = torch.arange(64, dtype=torch.float32, device=device)
    view = base[0:32]
    x = view.contiguous()  # materialize a correctly-sized tensor on device
    out = torch.empty_like(x)

    copy_kernel = _make_identity_kernel()
    copy_kernel(x, out)

    assert torch.equal(out.to("cpu"), torch.arange(32, dtype=torch.float32))
