"""Tests for HLO generation file lock in TorchMlirKernel._get_or_compile_hlo.

Verifies that only one process traces/lowers HLO on a cache miss,
while other processes wait and read from disk after the lock is released.
"""

import os
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from torch_neuronx.python_ops.compilation import hlo_cache as _hlo_cache_mod
from torch_neuronx.python_ops.compilation.hlo_cache import get as _hlo_cache_get
from torch_neuronx.python_ops.compilation.hlo_cache import put as _hlo_cache_put


@pytest.fixture(autouse=True)
def _enable_hlo_cache(tmp_path, monkeypatch):
    # Point the on-disk HLO cache at a fresh, per-test directory. The disk
    # cache (C++-backed, keyed on cache_key) persists across processes at the
    # default /tmp/hlo_cache, so without isolation a cache_key written by one
    # test — or one earlier run in a reused CI runner — turns a later run into
    # a disk hit, and _generate_hlo is never called (generate_count == 0). The
    # C++ backend re-reads TORCH_NEURONX_HLO_CACHE_DIR on every call, so setting
    # it here guarantees a cold cache; CACHE_DIR (captured at import for the
    # lock-file path) is repointed to match.
    cache_dir = str(tmp_path / "hlo_cache")
    os.makedirs(cache_dir, exist_ok=True)
    monkeypatch.setenv("TORCH_NEURONX_HLO_CACHE_DIR", cache_dir)
    monkeypatch.setattr(_hlo_cache_mod, "CACHE_DIR", cache_dir)
    monkeypatch.setattr("torch_neuronx.python_ops.torch_mlir.kernel.CACHE_DIR", cache_dir)

    original = _hlo_cache_mod._enabled
    _hlo_cache_mod._enabled = True
    yield
    _hlo_cache_mod._enabled = original


def _make_dummy_entry():
    from torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils import FunctionIO, TensorSpec

    hlo_bytes = b"module @test { func.func @main(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> }"
    io_spec = FunctionIO(
        inputs=(TensorSpec(shape=(4, 8), dtype=torch.float32),),
        outputs=(TensorSpec(shape=(4, 8), dtype=torch.float32),),
    )
    return (
        hlo_bytes,
        io_spec,
        [0],
        [0],
        torch.empty(4, 8, dtype=torch.float32, device="meta"),
        torch.empty(4, 8, dtype=torch.float32, device="meta"),
        {},
    )


def _make_generate_hlo_result():
    """Return an 8-tuple matching _generate_hlo output (includes cast_spec)."""
    entry = _make_dummy_entry()
    # Insert cast_spec=None at index 2
    return entry[:2] + (None,) + entry[2:]


def _make_kernel():
    """Create a TorchMlirKernel with mocked internals."""
    from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

    kernel = TorchMlirKernel(lambda x: x, "test_op")
    return kernel


class TestHloFileLockSingleGeneration:
    """Only one thread should call _generate_hlo on concurrent cache misses."""

    def test_only_one_thread_generates_hlo(self):
        kernel = _make_kernel()
        generate_count = 0
        generate_lock = threading.Lock()
        cache_key = "test_concurrent_key"
        generate_result = _make_generate_hlo_result()

        def slow_generate_hlo(*args, **kwargs):
            nonlocal generate_count
            with generate_lock:
                generate_count += 1
            time.sleep(0.2)
            return generate_result

        def run_get_or_compile():
            return kernel._get_or_compile_hlo((torch.empty(4, 8, device="meta"),), {}, None)

        num_threads = 5
        results = [None] * num_threads
        errors = [None] * num_threads

        # Clear any existing cache for this key
        kernel.hlo_cache.clear()

        threads = []
        for i in range(num_threads):

            def target(idx=i):
                try:
                    results[idx] = run_get_or_compile()
                except Exception as e:
                    errors[idx] = e

            t = threading.Thread(target=target)
            threads.append(t)

        # Patch the shared kernel ONCE, around the whole thread lifecycle. Doing this
        # inside each thread's own `with` block races: one thread's patch.object __exit__
        # delattr's the method off the shared instance while another thread is still
        # calling it, giving a transient AttributeError. A single __enter__/__exit__ pair
        # on the shared instance — entered before any thread starts, exited only after all
        # have joined — removes that race while preserving the shared-kernel/shared-cache
        # intent this test checks.
        with (
            patch.object(kernel, "_generate_unified_cache_key", return_value=cache_key),
            patch.object(kernel, "_generate_hlo", side_effect=slow_generate_hlo),
        ):
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

        # No errors
        for i, err in enumerate(errors):
            assert err is None, f"Thread {i} raised: {err}"

        # Only one thread should have called _generate_hlo
        assert generate_count == 1

        # All threads should get a valid result (8-tuple from _get_or_compile_hlo)
        for i, r in enumerate(results):
            assert r is not None, f"Thread {i} got None result"
            assert len(r) == 8  # hlo_bytes, io_spec, kept_in, kept_out, down, orig, pt, cache_key

    def test_disk_cache_hit_skips_lock(self):
        """When disk cache has the entry, no lock or _generate_hlo should be used."""
        kernel = _make_kernel()
        cache_key = "test_disk_hit_key"
        entry = _make_dummy_entry()

        # Pre-populate disk cache
        _hlo_cache_put(cache_key, *entry)

        # Clear in-memory cache
        kernel.hlo_cache.clear()

        with (
            patch.object(kernel, "_generate_unified_cache_key", return_value=cache_key),
            patch.object(kernel, "_generate_hlo") as mock_generate,
            patch("torch.utils._filelock.FileLock") as mock_filelock,
        ):
            result = kernel._get_or_compile_hlo((torch.empty(4, 8, device="meta"),), {}, None)

        mock_generate.assert_not_called()
        mock_filelock.assert_not_called()
        assert result is not None

    def test_in_memory_cache_hit_skips_everything(self):
        """When in-memory cache has the entry, nothing else is called."""
        kernel = _make_kernel()
        cache_key = "test_mem_hit_key"
        entry = _make_dummy_entry()

        # Pre-populate in-memory cache
        kernel.hlo_cache[cache_key] = entry

        with (
            patch.object(kernel, "_generate_unified_cache_key", return_value=cache_key),
            patch.object(kernel, "_generate_hlo") as mock_generate,
            patch("torch_neuronx.python_ops.compilation.hlo_cache.get") as mock_disk_get,
        ):
            result = kernel._get_or_compile_hlo((torch.empty(4, 8, device="meta"),), {}, None)

        mock_generate.assert_not_called()
        mock_disk_get.assert_not_called()
        assert result is not None

    def test_winner_does_not_read_from_disk(self):
        """The process that generates HLO should not do a redundant disk read."""
        kernel = _make_kernel()
        cache_key = "test_winner_no_read_key"
        generate_result = _make_generate_hlo_result()

        kernel.hlo_cache.clear()

        with (
            patch.object(kernel, "_generate_unified_cache_key", return_value=cache_key),
            patch.object(kernel, "_generate_hlo", return_value=generate_result),
            patch("torch_neuronx.python_ops.compilation.hlo_cache.get") as mock_get,
            patch("torch_neuronx.python_ops.compilation.hlo_cache.put"),
        ):
            # First call returns None (miss), should not be called again for winner
            mock_get.return_value = None

            kernel._get_or_compile_hlo((torch.empty(4, 8, device="meta"),), {}, None)

        # _hlo_cache_get called once (initial check), not twice
        assert mock_get.call_count == 1
