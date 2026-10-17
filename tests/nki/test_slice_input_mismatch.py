import nki
import nki.isa as nisa
import nki.language as nl
import pytest
import torch


@nki.jit
def _copy_kernel(x):
    x_sbuf = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sbuf, src=x)

    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=x_sbuf)
    return out


def test_slice_input_execution_succeeds_with_view_offset():
    """Pass a non-zero-offset slice to an NKI kernel and expect success.

    Setup: base has shape (1, 10); view takes [:, 2:7] -> shape (1, 5), offset 2.
    Correct behavior: allocate an NRT slice of exactly 5 elements at the offset,
    execute, and match expected values.
    """
    device = torch.device("neuron", 0)
    base = torch.arange(10, dtype=torch.float32, device=device).reshape(1, 10)
    view = base[:, 2:7]  # shape=(1, 5), storage_offset=2
    view = view.contiguous()

    result = _copy_kernel(view)

    expected = torch.arange(2, 7, dtype=torch.float32).reshape(1, 5)
    assert torch.equal(result.to("cpu"), expected)


def test_contiguous_input_execution_succeeds():
    """Control test: using a contiguous tensor (not a view)"""
    device = torch.device("neuron", 0)
    base = torch.arange(64, dtype=torch.float32, device=device).reshape(1, 64)
    view = base[:, 0:32]
    x = view.contiguous()

    result = _copy_kernel(x)

    assert torch.equal(result.to("cpu"), torch.arange(32, dtype=torch.float32).reshape(1, 32))
