"""Unit tests for torch_neuronx.packed_storage layout computation.

These tests verify packed physical layout math on CPU — no Neuron device needed.
"""

import os

import pytest
import torch

from torch_neuronx.int64_policy import (
    Int64Mode,
    get_int64_mode,
    is_native_int64_mode,
    is_packed_int64_mode,
    reset_int64_mode_cache,
)
from torch_neuronx.packed_storage import (
    is_packed_logical_dtype,
    packed_int64_high_word_alias,
    packed_int64_low_word_alias,
    packed_physical_dtype,
    packed_physical_layout,
    packed_physical_view,
)


class TestPackedPhysicalDtype:
    def test_int64_maps_to_int32(self):
        assert packed_physical_dtype(torch.int64) == torch.int32

    def test_float64_returns_none(self):
        assert packed_physical_dtype(torch.float64) is None

    def test_int32_returns_none(self):
        assert packed_physical_dtype(torch.int32) is None

    def test_float32_returns_none(self):
        assert packed_physical_dtype(torch.float32) is None

    def test_bool_returns_none(self):
        assert packed_physical_dtype(torch.bool) is None


class TestIsPackedLogicalDtype:
    def test_int64_is_packed(self):
        assert is_packed_logical_dtype(torch.int64) is True

    def test_float64_not_packed(self):
        assert is_packed_logical_dtype(torch.float64) is False

    def test_int32_not_packed(self):
        assert is_packed_logical_dtype(torch.int32) is False


class TestPackedPhysicalLayout:
    def test_int64_contiguous_2d(self):
        shape, stride, offset = packed_physical_layout(
            torch.int64, shape=(3, 4), stride=(4, 1), storage_offset=0
        )
        assert shape == (3, 4, 2)
        assert stride == (8, 2, 1)
        assert offset == 0

    def test_int64_contiguous_1d(self):
        shape, stride, offset = packed_physical_layout(
            torch.int64, shape=(10,), stride=(1,), storage_offset=0
        )
        assert shape == (10, 2)
        assert stride == (2, 1)
        assert offset == 0

    def test_int64_scalar(self):
        # Scalar () packs to (2,) — a single [low, high] lane pair. NOT (1, 2):
        # that would collide with a 1-D (1,) tensor and disagree with the
        # output-packing / logical_view conventions.
        shape, stride, offset = packed_physical_layout(
            torch.int64, shape=(), stride=(), storage_offset=0
        )
        assert shape == (2,)
        assert stride == (1,)
        assert offset == 0

    def test_int64_scalar_with_offset(self):
        shape, stride, offset = packed_physical_layout(
            torch.int64, shape=(), stride=(), storage_offset=5
        )
        assert shape == (2,)
        assert stride == (1,)
        assert offset == 10  # offset doubled

    def test_int64_with_offset(self):
        shape, stride, offset = packed_physical_layout(
            torch.int64, shape=(5,), stride=(1,), storage_offset=3
        )
        assert shape == (5, 2)
        assert stride == (2, 1)
        assert offset == 6  # offset doubled

    def test_int64_transposed(self):
        shape, stride, offset = packed_physical_layout(
            torch.int64, shape=(3, 4), stride=(1, 3), storage_offset=0
        )
        assert shape == (3, 4, 2)
        assert stride == (2, 6, 1)  # strides doubled
        assert offset == 0


class TestPackedPhysicalView:
    def test_int64_view_shape(self):
        t = torch.tensor([1, 2, 3], dtype=torch.int64)
        v = packed_physical_view(t)
        assert v.dtype == torch.int32
        assert v.shape == (3, 2)
        assert v.stride() == (2, 1)

    def test_int64_view_shares_storage(self):
        t = torch.tensor([1, 2, 3], dtype=torch.int64)
        v = packed_physical_view(t)
        assert v.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()

    def test_int64_2d_view(self):
        t = torch.zeros((3, 4), dtype=torch.int64)
        v = packed_physical_view(t)
        assert v.shape == (3, 4, 2)
        assert v.stride() == (8, 2, 1)

    def test_non_packed_dtype_raises(self):
        t = torch.zeros(3, dtype=torch.int32)
        with pytest.raises(TypeError, match="does not have a packed"):
            packed_physical_view(t)


class TestPackedInt64WordAliases:
    def test_low_word_extracts_low_bits(self):
        # Create int64 tensor with known values
        t = torch.tensor([0x0000_0002_0000_0001], dtype=torch.int64)
        low = packed_int64_low_word_alias(t)
        assert low.dtype == torch.int32
        assert low.shape == (1,)
        assert int(low[0]) == 1  # low word of 0x200000001

    def test_high_word_extracts_high_bits(self):
        t = torch.tensor([0x0000_0002_0000_0001], dtype=torch.int64)
        high = packed_int64_high_word_alias(t)
        assert high.dtype == torch.int32
        assert high.shape == (1,)
        assert int(high[0]) == 2  # high word of 0x200000001

    def test_low_word_negative_value(self):
        t = torch.tensor([-1], dtype=torch.int64)
        low = packed_int64_low_word_alias(t)
        high = packed_int64_high_word_alias(t)
        # -1 in int64 = 0xFFFFFFFF_FFFFFFFF
        assert int(low[0]) == -1  # 0xFFFFFFFF as int32
        assert int(high[0]) == -1

    def test_low_word_shape_preserved(self):
        t = torch.zeros((3, 4), dtype=torch.int64)
        low = packed_int64_low_word_alias(t)
        assert low.shape == (3, 4)
        assert low.dtype == torch.int32

    def test_non_int64_raises(self):
        t = torch.zeros(3, dtype=torch.float32)
        with pytest.raises(TypeError, match="expected int64"):
            packed_int64_low_word_alias(t)


class TestInt64Policy:
    def setup_method(self):
        """Start each test from the true default (no env set)."""
        os.environ.pop("TORCH_NEURONX_INT64_MODE", None)
        reset_int64_mode_cache()

    def teardown_method(self):
        """Restore env and clear cache after each test."""
        os.environ.pop("TORCH_NEURONX_INT64_MODE", None)
        reset_int64_mode_cache()

    def test_default_is_native(self):
        assert get_int64_mode() == Int64Mode.NATIVE
        assert is_native_int64_mode() is True

    def test_packed_mode(self):
        os.environ["TORCH_NEURONX_INT64_MODE"] = "packed"
        assert get_int64_mode() == Int64Mode.PACKED
        assert is_packed_int64_mode() is True

    def test_native_mode(self):
        os.environ["TORCH_NEURONX_INT64_MODE"] = "native"
        assert get_int64_mode() == Int64Mode.NATIVE
        assert is_native_int64_mode() is True

    def test_invalid_mode_raises(self):
        os.environ["TORCH_NEURONX_INT64_MODE"] = "invalid"
        with pytest.raises(ValueError, match="Invalid TORCH_NEURONX_INT64_MODE"):
            get_int64_mode()
