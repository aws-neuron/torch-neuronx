"""End-to-end tests for persistent caching.

Note: Cross-process caching tests are in tests/distributed/test_cross_process_caching.py
which uses the multi-process framework to properly test cache sharing between processes.
"""

import os

import pytest
import torch


def _list_neff_files(directory: str) -> list:
    """List all .neff files in a directory recursively."""
    neff_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(".neff"):
                neff_files.append(os.path.join(root, f))
    return neff_files


@pytest.mark.skipif(
    os.environ.get("NEURON_LAUNCH_BLOCKING") == "1",
    reason="Async pipeline persistent caching tests",
)
@pytest.mark.usefixtures("enable_metrics_for_class")
class TestPersistentCachingE2E:
    """End-to-end tests for persistent caching."""

    def test_second_execution_cache_hit(self, cache_dir):
        """Test that second execution with same shapes produces persistent cache hit."""
        import torch_neuronx

        torch_neuronx._C._clear_compilation_cache()

        a1 = torch.randn(4, 8, dtype=torch.float32, device="neuron")
        b1 = torch.randn(8, 4, dtype=torch.float32, device="neuron")
        result1 = torch.matmul(a1, b1)
        torch.neuron.synchronize()

        stats1 = torch_neuronx._C._get_compilation_cache_stats()
        assert (
            stats1["persistent_misses"] == 1
        ), f"Expected 1 persistent miss, got {stats1['persistent_misses']}"

        torch_neuronx._C._clear_compilation_memory_cache()

        a2 = torch.randn(4, 8, dtype=torch.float32, device="neuron")
        b2 = torch.randn(8, 4, dtype=torch.float32, device="neuron")
        result2 = torch.matmul(a2, b2)
        torch.neuron.synchronize()

        stats2 = torch_neuronx._C._get_compilation_cache_stats()
        assert (
            stats2["persistent_hits"] == 1
        ), f"Expected 1 persistent hit, got {stats2['persistent_hits']}"
        assert stats2["persistent_misses"] == 1
        assert result1.shape == (4, 4)
        assert result2.shape == (4, 4)

    def test_different_shapes_different_cache_entries(self, cache_dir):
        """Test that different shapes produce different cache entries."""
        import torch_neuronx

        torch_neuronx._C._clear_compilation_cache()

        a1 = torch.randn(4, 8, dtype=torch.float32, device="neuron")
        b1 = torch.randn(8, 4, dtype=torch.float32, device="neuron")
        torch.matmul(a1, b1)
        torch.neuron.synchronize()

        a2 = torch.randn(8, 16, dtype=torch.float32, device="neuron")
        b2 = torch.randn(16, 8, dtype=torch.float32, device="neuron")
        torch.matmul(a2, b2)
        torch.neuron.synchronize()

        stats = torch_neuronx._C._get_compilation_cache_stats()
        assert stats["persistent_misses"] == 2

        torch_neuronx._C._clear_compilation_memory_cache()

        a3 = torch.randn(4, 8, dtype=torch.float32, device="neuron")
        b3 = torch.randn(8, 4, dtype=torch.float32, device="neuron")
        torch.matmul(a3, b3)
        torch.neuron.synchronize()

        a4 = torch.randn(8, 16, dtype=torch.float32, device="neuron")
        b4 = torch.randn(16, 8, dtype=torch.float32, device="neuron")
        torch.matmul(a4, b4)
        torch.neuron.synchronize()

        stats2 = torch_neuronx._C._get_compilation_cache_stats()
        assert stats2["persistent_hits"] == 2
        assert stats2["persistent_misses"] == 2

    def test_cache_file_exists_on_disk(self, cache_dir):
        """Verify that cache files are written to disk."""
        import torch_neuronx

        torch_neuronx._C._clear_compilation_cache()

        neff_files_before = _list_neff_files(cache_dir)

        a = torch.randn(13, 17, dtype=torch.float32, device="neuron")
        b = torch.randn(17, 11, dtype=torch.float32, device="neuron")
        torch.matmul(a, b)
        torch.neuron.synchronize()

        stats = torch_neuronx._C._get_compilation_cache_stats()
        assert stats["persistent_misses"] == 1, "Expected 1 persistent miss"

        neff_files_after = _list_neff_files(cache_dir)
        assert len(neff_files_after) > len(neff_files_before), "No new NEFF files created on disk"

    def test_deterministic_cache_key(self, cache_dir):
        """Test that cache keys are deterministic.

        Running the same operation multiple times should produce exactly 1 persistent
        cache miss (one compilation) and subsequent memory cache hits.
        """
        import torch_neuronx

        torch_neuronx._C._clear_compilation_cache()

        num_iterations = 5
        for _ in range(num_iterations):
            a = torch.randn(10, 20, dtype=torch.float32, device="neuron")
            b = torch.randn(20, 15, dtype=torch.float32, device="neuron")
            torch.matmul(a, b)
            torch.neuron.synchronize()

        stats = torch_neuronx._C._get_compilation_cache_stats()

        assert (
            stats["persistent_misses"] == 1
        ), f"Expected 1 persistent miss, got {stats['persistent_misses']}."
