"""
Unit tests for cache_artifact.py precompile caching support.

Tests NeuronOutputCode serialization, monkey-patches, and the
_apply_precompile_patches gating logic.
"""

import dataclasses
import pickle
from unittest.mock import patch

import pytest
import torch

from torch_neuronx.neuron_dynamo_backend.cache_artifact import NeuronOutputCode


class TestNeuronOutputCode:
    """Tests for NeuronOutputCode dataclass."""

    def _make_output_code(self, **kwargs):
        defaults = {
            "graph_name": "test_model",
            "cache_key": "abc123",
            "hlo_cache_key": "abc123",
            "has_collectives": False,
            "retain_device": False,
        }
        defaults.update(kwargs)
        return NeuronOutputCode(**defaults)

    def test_basic_creation(self):
        oc = self._make_output_code()
        assert oc.graph_name == "test_model"
        assert oc.cache_key == "abc123"
        assert oc.hlo_cache_key == "abc123"
        assert oc._boxed_call is True
        assert oc._executor is None

    def test_picklable(self):
        """NeuronOutputCode should survive a pickle round-trip."""
        oc = self._make_output_code(
            graph_name="my_model",
            cache_key="key123",
            hlo_cache_key="hlo456",
            has_collectives=True,
        )
        data = pickle.dumps(oc)
        restored = pickle.loads(data)
        assert restored.graph_name == "my_model"
        assert restored.cache_key == "key123"
        assert restored.hlo_cache_key == "hlo456"
        assert restored.has_collectives is True
        assert restored._executor is None
        assert restored._boxed_call is True

    def test_prepare_for_serialization_clears_executor(self):
        oc = self._make_output_code()
        oc._executor = object()
        oc.prepare_for_serialization()
        assert oc._executor is None

    def test_pickle_after_prepare_for_serialization(self):
        oc = self._make_output_code()
        oc._executor = object()
        oc.prepare_for_serialization()
        data = pickle.dumps(oc)
        restored = pickle.loads(data)
        assert restored._executor is None
        assert restored.graph_name == "test_model"

    def test_post_compile_is_noop(self):
        oc = self._make_output_code()
        oc.post_compile(None, None, None)

    def test_set_triton_bundle_is_noop(self):
        oc = self._make_output_code()
        oc.set_triton_bundle(None)

    def test_boxed_call_attribute(self):
        """OutputCode must have _boxed_call=True for AOTAutograd."""
        oc = self._make_output_code()
        assert hasattr(oc, "_boxed_call")
        assert oc._boxed_call is True

    def test_ensure_neff_loaded_raises_on_missing_hlo(self):
        """_ensure_neff_loaded should raise RuntimeError if HLO cache misses."""
        oc = self._make_output_code(hlo_cache_key="nonexistent_key")
        with (
            patch("torch_neuronx.python_ops.compilation.hlo_cache.get", return_value=None),
            pytest.raises(RuntimeError, match="HLO cache miss"),
        ):
            oc._ensure_neff_loaded()

    def test_ensure_neff_loaded_skips_if_no_key(self):
        """_ensure_neff_loaded should be a no-op if hlo_cache_key is empty."""
        oc = self._make_output_code(hlo_cache_key="")
        with patch("torch_neuronx.python_ops.compilation.hlo_cache.get") as mock_hlo_get:
            oc._ensure_neff_loaded()
            mock_hlo_get.assert_not_called()
