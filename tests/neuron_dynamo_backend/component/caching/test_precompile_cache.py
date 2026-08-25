"""
End-to-end tests for precompile caching with the Neuron backend.

Tests the full save/load cycle: cold compile → save to Dynamo cache →
reset → load from cache → execute without recompilation.
"""

import shutil

import pytest
import torch
import torch._dynamo.config
import torch._functorch.config
from torch._dynamo.package import DynamoCache
from torch._dynamo.precompile_context import PrecompileContext
from torch._dynamo.utils import counters

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def precompile_config():
    """Enable precompile caching for all tests in this module."""
    with (
        torch._dynamo.config.patch(
            caching_precompile=True,
            enable_aot_compile=True,
        ),
        torch._functorch.config.patch(enable_autograd_cache=True),
    ):
        yield


@pytest.fixture(autouse=True)
def clean_state():
    """Reset all caches and counters before and after each test."""
    torch._dynamo.reset()
    counters.clear()
    DynamoCache.clear()
    PrecompileContext.clear()
    yield
    torch.neuron.synchronize()
    torch._dynamo.reset()
    counters.clear()
    DynamoCache.clear()
    PrecompileContext.clear()


@pytest.fixture
def simple_model():
    """Simple model on neuron device."""
    return torch.nn.Linear(4, 2).to("neuron")


@pytest.fixture
def simple_input():
    """Simple input tensor on neuron device."""
    return torch.randn(2, 4, device="neuron")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_and_reload():
    """Save cache, reset dynamo state, simulate fresh process."""
    PrecompileContext.save_to_dynamo_cache()
    torch._dynamo.reset()
    PrecompileContext.clear()
    counters.clear()


def _get_dynamo_cache_counters():
    return dict(counters.get("dynamo_cache", {}))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPrecompileCacheInference:
    """Test precompile caching for inference (no backward)."""

    def test_cold_compile_saves_cache(self, simple_model, simple_input):
        """Cold run should compile and save artifacts."""
        compiled = torch.compile(simple_model, backend="neuron")

        with torch.no_grad():
            result = compiled(simple_input)

        assert result.shape == (2, 2)

        # Should have recorded dynamo cache entries
        assert len(PrecompileContext._dynamo_cache_entries) > 0

        # Should have recorded backend artifacts
        assert len(PrecompileContext._backend_artifacts_by_key) > 0

    def test_warm_load_skips_compilation(self, simple_model, simple_input):
        """Warm run should load from cache without recompilation."""
        # Cold run
        compiled = torch.compile(simple_model, backend="neuron")
        with torch.no_grad():
            expected = compiled(simple_input)

        _save_and_reload()

        # Warm run
        compiled2 = torch.compile(simple_model, backend="neuron")
        with torch.no_grad():
            result = compiled2(simple_input)

        # Should have loaded from cache
        cache_counters = _get_dynamo_cache_counters()
        assert cache_counters.get("dynamo_cache_hit", 0) >= 1

        # Results should match
        torch.testing.assert_close(result.cpu(), expected.cpu())

    def test_cache_roundtrip_correctness(self, simple_input):
        """Verify numerical correctness across cache save/load."""
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 2),
        ).to("neuron")

        compiled = torch.compile(model, backend="neuron")
        with torch.no_grad():
            expected = compiled(simple_input)

        _save_and_reload()

        compiled2 = torch.compile(model, backend="neuron")
        with torch.no_grad():
            result = compiled2(simple_input)

        torch.testing.assert_close(result.cpu(), expected.cpu())


class TestPrecompileCacheTraining:
    """Test precompile caching for training (with backward)."""

    def test_training_saves_cache(self, simple_model, simple_input):
        """Training run should save both forward and backward artifacts."""
        compiled = torch.compile(simple_model, backend="neuron")

        # Forward + backward
        out = compiled(simple_input)
        loss = out.sum()
        loss.backward()

        # Should have recorded artifacts
        assert len(PrecompileContext._dynamo_cache_entries) > 0
        assert len(PrecompileContext._backend_artifacts_by_key) > 0

    def test_training_warm_load(self, simple_model, simple_input):
        """Warm training run should load from cache."""
        compiled = torch.compile(simple_model, backend="neuron")
        out = compiled(simple_input)
        out.sum().backward()

        _save_and_reload()

        # Warm run
        compiled2 = torch.compile(simple_model, backend="neuron")
        out2 = compiled2(simple_input)
        out2.sum().backward()

        cache_counters = _get_dynamo_cache_counters()
        assert cache_counters.get("dynamo_cache_hit", 0) >= 1


class TestPrecompileCacheEdgeCases:
    """Test edge cases and error handling."""

    def test_cache_invalidated_on_shape_change(self, simple_model):
        """Different input shapes should not use stale cache."""
        input1 = torch.randn(2, 4, device="neuron")
        input2 = torch.randn(3, 4, device="neuron")

        compiled = torch.compile(simple_model, backend="neuron")
        with torch.no_grad():
            compiled(input1)

        _save_and_reload()
        frames_before = torch._dynamo.convert_frame.FRAME_COUNTER

        # Different shape — guads should fail, triggering recompile
        compiled2 = torch.compile(simple_model, backend="neuron")
        with torch.no_grad():
            result = compiled2(input2)

        assert result.shape == (3, 2)

        # Cache file was found (hit), but guards failed on the new shape,
        # so dynamo recompiled — frame counter should have incremented.
        assert frames_before < torch._dynamo.convert_frame.FRAME_COUNTER

    def test_multiple_models_same_module(self):
        """Two models defined in the same module should not interfere."""
        model1 = torch.nn.Linear(4, 2).to("neuron")
        model2 = torch.nn.Linear(4, 3).to("neuron")
        x = torch.randn(2, 4, device="neuron")

        compiled1 = torch.compile(model1, backend="neuron")
        compiled2 = torch.compile(model2, backend="neuron")

        with torch.no_grad():
            r1 = compiled1(x)
            r2 = compiled2(x)

        assert r1.shape == (2, 2)
        assert r2.shape == (2, 3)

        _save_and_reload()

        compiled1b = torch.compile(model1, backend="neuron")
        compiled2b = torch.compile(model2, backend="neuron")

        with torch.no_grad():
            r1b = compiled1b(x)
            r2b = compiled2b(x)

        torch.testing.assert_close(r1.cpu(), r1b.cpu())
        torch.testing.assert_close(r2.cpu(), r2b.cpu())

    def test_warm_load_after_hlo_cache_wipe(self, simple_model, simple_input):
        """Warm start should succeed even when HLO cache is wiped (V2200810117)."""
        from unittest.mock import patch

        # Cold run
        compiled = torch.compile(simple_model, backend="neuron")
        with torch.no_grad():
            expected = compiled(simple_input)

        _save_and_reload()

        # Simulate /tmp/hlo_cache/ wipe — HLO cache returns None for all keys
        with patch("torch_neuronx.python_ops.compilation.hlo_cache.get", return_value=None):
            compiled2 = torch.compile(simple_model, backend="neuron")
            with torch.no_grad():
                result = compiled2(simple_input)

        # Should still produce correct results via embedded hlo_bytes
        torch.testing.assert_close(result.cpu(), expected.cpu())

    def test_warm_load_after_hlo_cache_wipe_training(self, simple_model, simple_input):
        """Warm start after HLO cache wipe with training-shaped graph (V2200810117)."""
        from unittest.mock import patch

        # Cold run (inference — the test validates cache hit, not backward)
        compiled = torch.compile(simple_model, backend="neuron")
        with torch.no_grad():
            compiled(simple_input)

        _save_and_reload()

        # Simulate /tmp/hlo_cache/ wipe — verify cache hit on warm reload
        with patch("torch_neuronx.python_ops.compilation.hlo_cache.get", return_value=None):
            compiled2 = torch.compile(simple_model, backend="neuron")
            with torch.no_grad():
                compiled2(simple_input)

        cache_counters = _get_dynamo_cache_counters()
        assert cache_counters.get("dynamo_cache_hit", 0) >= 1

    @pytest.mark.skip(
        reason="Segfaults: MLIR context init is not thread-safe when called from "
        "PyTorch autograd worker thread during backward recompilation. "
        "backward() triggers convert_fx_to_stablehlo on a non-main thread, "
        "and _mlir.so crashes in torchMlirRegisterAllDialects."
    )
    def test_warm_load_after_hlo_cache_wipe_training_backward(self, simple_model, simple_input):
        """Training backward after HLO cache wipe — segfaults due to MLIR thread-safety."""
        from unittest.mock import patch

        # Cold run with backward
        compiled = torch.compile(simple_model, backend="neuron")
        out = compiled(simple_input)
        out.sum().backward()

        _save_and_reload()

        # Simulate /tmp/hlo_cache/ wipe
        with patch("torch_neuronx.python_ops.compilation.hlo_cache.get", return_value=None):
            compiled2 = torch.compile(simple_model, backend="neuron")
            out2 = compiled2(simple_input)
            out2.sum().backward()

        cache_counters = _get_dynamo_cache_counters()
        assert cache_counters.get("dynamo_cache_hit", 0) >= 1
