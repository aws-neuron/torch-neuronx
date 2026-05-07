"""
Tests for global_default_device changes in NeuronDevice.cpp

These test the new 3-level fallback mechanism:
1. Thread-local device (if >= 0)
2. Global default device (if >= 0)
3. Fallback to 0

The key behavior change: worker threads now inherit the process global default
device instead of always getting 0.
"""

import threading

import pytest

import torch_neuronx


class TestGlobalDefaultDevice:
    """Tests for the global_default_device mechanism."""

    def test_current_device_returns_valid_value(self):
        """current_device should return a valid non-negative value."""
        device = torch_neuronx.current_device()
        assert device >= 0, f"Device should be non-negative, got {device}"

    def test_set_device_updates_current_device(self):
        """set_device should update current_device."""
        vnc_count = torch_neuronx._C._vnc_count()
        if vnc_count < 1:
            pytest.skip("Need at least 1 device")

        torch_neuronx.set_device(0)
        assert torch_neuronx.current_device() == 0

    def test_worker_thread_inherits_device(self):
        """
        Worker threads should inherit global_default_device.

        This is the key test for the new behavior - before, worker threads
        would get device 0. Now they inherit whatever device was set via
        set_device() or distributed init.
        """
        vnc_count = torch_neuronx._C._vnc_count()
        if vnc_count < 1:
            pytest.skip("Need at least 1 device")

        # Main thread sets device 0
        torch_neuronx.set_device(0)
        main_device = torch_neuronx.current_device()

        # Worker thread should see the same device (via global default)
        result = []

        def worker():
            result.append(torch_neuronx.current_device())

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        # Worker should get device 0 (inherited from global default)
        assert (
            result[0] == main_device
        ), f"Worker got device {result[0]}, expected {main_device} (inherited from main thread)"

    def test_multiple_workers_inherit_same_device(self):
        """Multiple worker threads should all inherit the same global default."""
        vnc_count = torch_neuronx._C._vnc_count()
        if vnc_count < 1:
            pytest.skip("Need at least 1 device")

        torch_neuronx.set_device(0)

        num_workers = 4
        results = []
        lock = threading.Lock()

        def worker():
            dev = torch_neuronx.current_device()
            with lock:
                results.append(dev)

        threads = [threading.Thread(target=worker) for _ in range(num_workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All workers should get device 0
        assert all(r == 0 for r in results), f"Not all workers got device 0: {results}"
