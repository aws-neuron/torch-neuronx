"""
Tests for MegaCache integration with Neuron NEFF cache.

Tests the wrap-serialize approach where artifacts are collected just-in-time
when save_cache_artifacts() is called.
"""

import uuid

import pytest
import torch

import torch_neuronx

# Enable metrics for cache statistics in tests.
torch_neuronx._C._neuron_set_metrics_enabled(True)

from torch_neuronx.neuron_dynamo_backend.cache_artifact import ArtifactType  # noqa: E402
from torch_neuronx.neuron_dynamo_backend.compile import (  # noqa: E402
    clear_compiled_cache_keys,
    get_compiled_cache_keys,
)


def get_cache_stats():
    """Get compilation cache statistics."""
    return torch_neuronx._C._get_compilation_cache_stats()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def reset_dynamo():
    """Reset dynamo state before test."""
    torch._dynamo.reset()
    yield


@pytest.fixture
def clear_caches():
    """Clear all caches before and after test."""
    from torch.compiler._cache import CacheArtifactManager

    CacheArtifactManager.clear()
    clear_compiled_cache_keys()  # Clears all artifact types
    torch_neuronx._C._clear_compilation_cache()
    yield
    CacheArtifactManager.clear()
    clear_compiled_cache_keys()
    torch_neuronx._C._clear_compilation_cache()


@pytest.fixture
def simple_model():
    """Simple model on neuron device."""
    return torch.nn.Linear(4, 2).to("neuron")


@pytest.fixture
def simple_input():
    """Simple input tensor on neuron device."""
    return torch.randn(1, 4, device="neuron")


# =============================================================================
# Low-level C++ Binding Tests
# =============================================================================


class TestNeffCacheBindings:
    """Test put_neff_cache and get_neff_cache C++ bindings."""

    def test_write_and_read(self):
        key = f"test_key_{uuid.uuid4().hex[:8]}"
        neff_bytes = b"fake_neff_content"

        assert torch_neuronx._C.put_neff_cache(key, neff_bytes) is True
        assert torch_neuronx._C.get_neff_cache(key) == neff_bytes

    def test_missing_key_returns_none(self):
        key = f"nonexistent_{uuid.uuid4().hex[:8]}"
        assert torch_neuronx._C.get_neff_cache(key) is None

    def test_overwrite(self):
        key = f"test_overwrite_{uuid.uuid4().hex[:8]}"
        torch_neuronx._C.put_neff_cache(key, b"original")
        torch_neuronx._C.put_neff_cache(key, b"updated")
        assert torch_neuronx._C.get_neff_cache(key) == b"updated"

    def test_empty_content_treated_as_miss(self):
        """Empty content is invalid NEFF, treated as cache miss."""
        key = f"test_empty_{uuid.uuid4().hex[:8]}"
        torch_neuronx._C.put_neff_cache(key, b"")
        assert torch_neuronx._C.get_neff_cache(key) is None


class TestGetNeffInfoBinding:
    """Test get_neff_info C++ binding."""

    def test_returns_none_for_missing_key(self):
        result = torch_neuronx._C.get_neff_info(f"nonexistent_{uuid.uuid4().hex[:8]}")
        assert result is None

    def test_returns_tuple_after_compilation(self, reset_dynamo, simple_model, simple_input):
        """Returns (persistent_key, neff_bytes) after compilation."""
        clear_compiled_cache_keys()
        torch_neuronx._C._clear_compilation_cache()

        compiled = torch.compile(simple_model, backend="neuron")
        compiled(simple_input)
        # Async compilation: NEFF may not be in cache immediately after compile() returns.
        # synchronize() blocks until all pending compilations complete.
        torch.neuron.synchronize()

        cache_keys = get_compiled_cache_keys(ArtifactType.NEURON_NEFF)
        assert len(cache_keys) >= 1

        result = torch_neuronx._C.get_neff_info(next(iter(cache_keys)))
        assert result is not None

        persistent_key, neff_bytes = result
        assert len(persistent_key) == 32  # XXH3_128 hex
        assert len(neff_bytes) > 0
        assert get_cache_stats()["total_entries"] >= 1


# =============================================================================
# Cache Key Tracking Tests
# =============================================================================


class TestCacheKeyTracking:
    """Test that compiled cache keys are tracked in Python."""

    def test_key_tracked_after_compile(self, reset_dynamo, simple_model, simple_input):
        clear_compiled_cache_keys()
        torch_neuronx._C._clear_compilation_cache()

        compiled = torch.compile(simple_model, backend="neuron")
        compiled(simple_input)

        assert len(get_compiled_cache_keys(ArtifactType.NEURON_NEFF)) >= 1

        torch.neuron.synchronize()
        assert get_cache_stats()["total_entries"] >= 1

    def test_clear_tracking(self, reset_dynamo, simple_model, simple_input):
        compiled = torch.compile(simple_model, backend="neuron")
        compiled(simple_input)

        clear_compiled_cache_keys()
        assert len(get_compiled_cache_keys(ArtifactType.NEURON_NEFF)) == 0


# =============================================================================
# NeuronNeffCacheArtifact Tests
# =============================================================================


class TestNeuronNeffCacheArtifact:
    """Test NeuronNeffCacheArtifact for PyTorch MegaCache integration."""

    def test_type(self):
        from torch_neuronx.neuron_dynamo_backend.cache_artifact import NeuronNeffCacheArtifact

        assert NeuronNeffCacheArtifact.type() == "neuron_neff"

    def test_registered_with_factory(self):
        from torch.compiler._cache import CacheArtifactFactory

        from torch_neuronx.neuron_dynamo_backend.cache_artifact import NeuronNeffCacheArtifact

        artifact = CacheArtifactFactory.create("neuron_neff", "key", b"content")
        assert isinstance(artifact, NeuronNeffCacheArtifact)

    def test_populate_cache(self):
        from torch_neuronx.neuron_dynamo_backend.cache_artifact import NeuronNeffCacheArtifact

        key = f"test_populate_{uuid.uuid4().hex[:8]}"
        NeuronNeffCacheArtifact(key, b"test_content").populate_cache()
        assert torch_neuronx._C.get_neff_cache(key) == b"test_content"


# =============================================================================
# Serialize Wrapper Tests
# =============================================================================


class TestSerializeWrapper:
    """Test that serialize() wrapper collects Neuron artifacts."""

    def test_serialize_collects_artifacts(
        self, reset_dynamo, clear_caches, simple_model, simple_input
    ):
        compiled = torch.compile(simple_model, backend="neuron")
        compiled(simple_input)

        artifacts, info = torch.compiler.save_cache_artifacts()

        assert artifacts is not None
        assert "neuron_neff" in info.artifacts
        assert len(info.artifacts["neuron_neff"]) >= 1
        assert get_cache_stats()["total_entries"] >= 1

    def test_serialize_syncs_automatically(
        self, reset_dynamo, clear_caches, simple_model, simple_input
    ):
        """Wrapper calls synchronize() - no explicit sync needed."""
        compiled = torch.compile(simple_model, backend="neuron")
        compiled(simple_input)
        # No torch.neuron.synchronize() - wrapper handles it

        artifacts, info = torch.compiler.save_cache_artifacts()

        assert "neuron_neff" in info.artifacts
        assert get_cache_stats()["total_entries"] >= 1


# =============================================================================
# End-to-End Tests
# =============================================================================


class TestMegaCacheEndToEnd:
    """End-to-end tests for save/load_cache_artifacts."""

    def test_roundtrip(self, reset_dynamo, clear_caches, simple_model, simple_input):
        """Compile → save → clear → load → verify."""
        compiled = torch.compile(simple_model, backend="neuron")
        compiled(simple_input)

        # Save
        artifacts, save_info = torch.compiler.save_cache_artifacts()
        assert "neuron_neff" in save_info.artifacts
        saved_keys = save_info.artifacts["neuron_neff"]
        assert len(saved_keys) >= 1
        assert get_cache_stats()["total_entries"] >= 1

        # Verify keys exist
        for key in saved_keys:
            assert torch_neuronx._C.get_neff_cache(key) is not None

        # Clear (simulate fresh machine)
        torch_neuronx._C._clear_compilation_cache()

        # Load and verify repopulated
        torch.compiler.load_cache_artifacts(artifacts)
        for key in saved_keys:
            assert torch_neuronx._C.get_neff_cache(key) is not None
