"""Shared helpers for lazy-alloc defer-retry tests: VNC-relative sizing,
the pending-kDeallocs-behind-compute staging pattern, and a clean-device
fixture. Used by ``TestLazyAllocDeferRetry`` and ``TestMultiStreamDeferRetry``."""

from __future__ import annotations

import pytest
import torch

import torch_neuronx

# Staged-kDealloc chunk size and per-tensor compute ops. Tuned for FIFO
# pressure without drowning in per-op overhead.
DEFAULT_CHUNK_MIB = 256
DEFAULT_OPS_PER_TENSOR = 40

# Fractions of per-VNC HBM (see ``vnc_hbm_mib``).
# big + staged ≈ 1.04 x VNC: reliable NRT_RESOURCE on the big alloc.
# oversize > 1 x VNC: guaranteed AttemptOOMRecovery failure even after cache drain.
_BIG_FRACTION_NUM = 42
_STAGED_FRACTION_NUM = 62
_OVERSIZE_FRACTION_NUM = 150
_FRACTION_DENOM = 100


def vnc_hbm_mib() -> int:
    """Per-VNC HBM budget in MiB. Sizing baseline for all pressure ratios."""
    total_bytes = torch.neuron.get_device_properties(0).total_memory
    return total_bytes // (1024 * 1024)


def big_mib() -> int:
    """Pressure alloc: fits alone, fails with staged chunks resident."""
    return (vnc_hbm_mib() * _BIG_FRACTION_NUM) // _FRACTION_DENOM


def staged_mib() -> int:
    """Total MiB of pending-dealloc chunks staged behind compute."""
    return (vnc_hbm_mib() * _STAGED_FRACTION_NUM) // _FRACTION_DENOM


def n_chunks(chunk_mib: int = DEFAULT_CHUNK_MIB) -> int:
    """Number of chunks needed to total :func:`staged_mib` MiB."""
    return staged_mib() // chunk_mib


def oversize_mib() -> int:
    """Alloc guaranteed to exceed VNC budget; exercises the hard-OOM path."""
    return (vnc_hbm_mib() * _OVERSIZE_FRACTION_NUM) // _FRACTION_DENOM


def floats_for_mib(mib: int) -> int:
    """float32 element count for a tensor of the given MiB size."""
    return mib * 1024 * 1024 // 4


def stage_pending_kdeallocs(
    chunk_mib: int | None = None,
    num_chunks: int | None = None,
    ops_per_tensor: int | None = None,
) -> None:
    """Stage "pending kDeallocs behind compute" on the ambient stream.

    Allocates chunks, pumps in-place ops to park compute on the worker FIFO,
    then drops Python refs without syncing. On return, the worker has compute
    + kDealloc ops submitted but not drained NRT-side — the precondition a
    subsequent kAlloc needs to hit NRT_RESOURCE.
    """
    if chunk_mib is None:
        chunk_mib = DEFAULT_CHUNK_MIB
    if num_chunks is None:
        num_chunks = n_chunks(chunk_mib)
    if ops_per_tensor is None:
        ops_per_tensor = DEFAULT_OPS_PER_TENSOR
    floats = floats_for_mib(chunk_mib)
    holds = [torch.zeros(floats, device="neuron") for _ in range(num_chunks)]
    for _ in range(ops_per_tensor):
        for x in holds:
            x.add_(1.0)
            x.mul_(2.0)
    holds.clear()  # kDealloc submitted behind compute; not drained


@pytest.fixture
def clean_device_state():
    """Sync, drop allocator cache, and zero peak/accumulated counters before
    and after the test."""
    torch_neuronx.synchronize()
    torch_neuronx.empty_cache()
    torch_neuronx.reset_peak_memory_stats()
    torch_neuronx.reset_accumulated_memory_stats()
    yield
    torch_neuronx.synchronize()
    torch_neuronx.empty_cache()
