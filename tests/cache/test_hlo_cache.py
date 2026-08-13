"""Tests for HLO persistent cache."""

import os

import pytest
import torch

from torch_neuronx.python_ops.compilation import hlo_cache as _hlo_cache_mod
from torch_neuronx.python_ops.compilation.hlo_cache import (
    _deserialize_meta_tensor,
    _serialize_meta_tensor,
    get,
    put,
)


@pytest.fixture(autouse=True)
def _enable_hlo_cache():
    """Enable HLO cache for each test, restore original state after."""
    original = _hlo_cache_mod._enabled
    _hlo_cache_mod._enabled = True
    yield
    _hlo_cache_mod._enabled = original


def _make_dummy_entry():
    """Create a minimal valid eager HLO cache entry."""
    from torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils import FunctionIO, TensorSpec
    from torch_neuronx.python_ops.compilation.hlo_cache import CachedHLO

    return CachedHLO(
        hlo_bytes=b"module @test { func.func @main(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> }",
        io_spec=FunctionIO(
            inputs=(TensorSpec(shape=(4, 8), dtype=torch.float32),),
            outputs=(TensorSpec(shape=(4, 8), dtype=torch.float32),),
        ),
        kept_input_indices=[0],
        kept_output_indices=[0],
        downcasted_outputs=torch.empty(4, 8, dtype=torch.float32, device="meta"),
        original_outputs=torch.empty(4, 8, dtype=torch.float32, device="meta"),
        passthrough_mappings={},
    )


class TestHloCachePutGet:
    def test_put_and_get(self):
        entry = _make_dummy_entry()
        key = "test_put_get_key"
        put(key, entry)
        result = get(key)
        assert result is not None
        assert result.hlo_bytes == entry.hlo_bytes
        assert result.kept_input_indices == entry.kept_input_indices
        assert result.kept_output_indices == entry.kept_output_indices

    def test_get_miss(self):
        result = get("nonexistent_key_12345")
        assert result is None

    def test_duplicate_put_skips(self):
        entry = _make_dummy_entry()
        key = "test_duplicate_key"
        put(key, entry)

        hlo_cache_dir = os.environ.get("TORCH_NEURONX_HLO_CACHE_DIR", "/tmp/hlo_cache")
        cache_files = []
        for root, _, files in os.walk(hlo_cache_dir):
            for f in files:
                if f.endswith(".bin"):
                    cache_files.append(os.path.join(root, f))

        assert len(cache_files) >= 1
        mtime_before = os.path.getmtime(cache_files[-1])

        put(key, entry)
        mtime_after = os.path.getmtime(cache_files[-1])
        assert mtime_before == mtime_after

    def test_different_keys_different_entries(self):
        entry = _make_dummy_entry()
        put("key_a", entry)
        put("key_b", entry)
        assert get("key_a") is not None
        assert get("key_b") is not None


class TestMetaTensorSerialization:
    def test_single_tensor(self):
        t = torch.empty(4, 8, dtype=torch.float32, device="meta")
        serialized = _serialize_meta_tensor(t)
        deserialized = _deserialize_meta_tensor(serialized)
        assert deserialized.shape == t.shape
        assert deserialized.dtype == t.dtype
        assert deserialized.device.type == "meta"

    def test_tuple_of_tensors(self):
        t = (
            torch.empty(4, 8, dtype=torch.float32, device="meta"),
            torch.empty(2, 3, dtype=torch.bfloat16, device="meta"),
        )
        serialized = _serialize_meta_tensor(t)
        deserialized = _deserialize_meta_tensor(serialized)
        assert len(deserialized) == 2
        assert deserialized[0].shape == (4, 8)
        assert deserialized[1].dtype == torch.bfloat16

    def test_none(self):
        assert _serialize_meta_tensor(None) is None
        assert _deserialize_meta_tensor(None) is None


class TestCacheDisabled:
    def test_get_returns_none_when_disabled(self):
        _hlo_cache_mod._enabled = False
        result = get("any_key")
        assert result is None

    def test_put_noop_when_disabled(self):
        _hlo_cache_mod._enabled = False
        entry = _make_dummy_entry()
        put("disabled_key", entry)
        _hlo_cache_mod._enabled = True
        assert get("disabled_key") is None
