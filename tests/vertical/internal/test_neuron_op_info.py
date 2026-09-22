"""Unit tests for neuron_op_info sample construction — index dtype correctness."""

from __future__ import annotations

import pytest
import torch

from .neuron_op_info import _DTYPE_MAP, _build_value


class TestIndexDtypeCorrectness:
    """Integer-typed tensor specs must produce integer tensors, never float.

    The shape registry generator emits standard dtype strings ("int64", "int32")
    for index tensors. The sample builder must map these correctly so ops like
    embedding, gather, scatter, and index_select receive integer indices.
    """

    @pytest.mark.parametrize(
        "dtype_str,expected",
        [
            ("int64", torch.long),
            ("int32", torch.int32),
            ("int16", torch.int16),
            ("int8", torch.int8),
            ("uint8", torch.uint8),
            ("long", torch.long),
            ("int", torch.int32),
        ],
    )
    def test_integer_dtype_spec_produces_integer_tensor(self, dtype_str, expected):
        raw = {"shape": [4, 2048], "dtype": dtype_str}
        t = _build_value("Tensor", raw, "cpu", torch.float32)
        assert (
            t.dtype == expected
        ), f"dtype spec {dtype_str!r} produced {t.dtype}, expected {expected}"

    @pytest.mark.parametrize(
        "dtype_str,expected",
        [
            ("int64", torch.long),
            ("int32", torch.int32),
        ],
    )
    def test_index_tensor_with_bounds(self, dtype_str, expected):
        raw = {"shape": [1, 2048], "dtype": dtype_str, "low": 0, "high": 32000}
        t = _build_value("Tensor", raw, "cpu", torch.float32)
        assert t.dtype == expected
        assert t.min() >= 0
        assert t.max() < 32000

    def test_integer_dtype_not_overridden_by_default(self):
        """Even when default dtype is bfloat16, an int64 spec must stay int64."""
        raw = {"shape": [8, 512], "dtype": "int64"}
        t = _build_value("Tensor", raw, "cpu", torch.bfloat16)
        assert t.dtype == torch.long

    def test_bool_dtype_spec(self):
        raw = {"shape": [4, 4], "dtype": "bool"}
        t = _build_value("Tensor", raw, "cpu", torch.float32)
        assert t.dtype == torch.bool

    @pytest.mark.parametrize("dtype_str", ["float32", "float"])
    def test_float_dtype_spec_uses_default(self, dtype_str):
        """Float specs use the caller's default dtype (sentinel behavior)."""
        raw = {"shape": [4, 4], "dtype": dtype_str}
        t = _build_value("Tensor", raw, "cpu", torch.bfloat16)
        assert t.dtype == torch.bfloat16

    def test_bare_shape_list_uses_default_dtype(self):
        """Shape without dtype metadata uses whatever default is passed in."""
        raw = [4, 4]
        t = _build_value("Tensor", raw, "cpu", torch.float32)
        assert t.dtype == torch.float32

    def test_dtype_map_covers_generator_output(self):
        """All dtype strings the generator can emit must be in _DTYPE_MAP."""
        generator_dtypes = [
            "float32",
            "float64",
            "float16",
            "bfloat16",
            "int64",
            "int32",
            "int16",
            "int8",
            "uint8",
            "bool",
        ]
        for dtype_str in generator_dtypes:
            assert dtype_str in _DTYPE_MAP, f"{dtype_str!r} missing from _DTYPE_MAP"
