"""Unit tests for torch_neuronx.utils dtype conversion helpers."""

import numpy as np
import pytest
import torch

from torch_neuronx.utils import _STR_TO_TORCH_MAP, map_external_dtype_to_torch


class TestStrToTorchMap:
    """Tests for _STR_TO_TORCH_MAP and its use in map_external_dtype_to_torch."""

    @pytest.mark.parametrize(
        "dtype_str, expected",
        [
            ("bfloat16", torch.bfloat16),
            ("float16", torch.float16),
            ("float32", torch.float32),
            ("float64", torch.float64),
            ("int32", torch.int32),
            ("int64", torch.int64),
            ("int16", torch.int16),
            ("int8", torch.int8),
            ("uint8", torch.uint8),
            ("bool", torch.bool),
            ("bool_", torch.bool),
        ],
    )
    def test_str_to_torch_map_entries(self, dtype_str, expected):
        assert _STR_TO_TORCH_MAP[dtype_str] is expected

    @pytest.mark.parametrize(
        "dtype_str, expected",
        [
            ("bfloat16", torch.bfloat16),
            ("float32", torch.float32),
            ("int8", torch.int8),
            ("bool", torch.bool),
        ],
    )
    def test_map_external_dtype_string_input(self, dtype_str, expected):
        assert map_external_dtype_to_torch(dtype_str) is expected

    def test_map_external_dtype_unknown_string_raises(self):
        with pytest.raises(RuntimeError, match="Unrecognized dtype"):
            map_external_dtype_to_torch("not_a_dtype")


class TestMapExternalDtypeToTorch:
    """Tests for other map_external_dtype_to_torch paths."""

    def test_torch_dtype_passthrough(self):
        assert map_external_dtype_to_torch(torch.float32) is torch.float32

    @pytest.mark.parametrize(
        "np_dtype, expected",
        [
            (np.float32, torch.float32),
            (np.int64, torch.int64),
            (np.bool_, torch.bool),
        ],
    )
    def test_numpy_scalar_type(self, np_dtype, expected):
        assert map_external_dtype_to_torch(np_dtype) is expected

    @pytest.mark.parametrize(
        "np_dtype_str, expected",
        [
            ("float32", torch.float32),
            ("int16", torch.int16),
        ],
    )
    def test_numpy_dtype_instance(self, np_dtype_str, expected):
        assert map_external_dtype_to_torch(np.dtype(np_dtype_str)) is expected
