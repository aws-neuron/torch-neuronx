"""Unit tests for torch_neuronx.utils dtype conversion helpers."""

import numpy as np
import pytest
import torch

import torch_neuronx.utils as utils


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
        assert utils._STR_TO_TORCH_MAP[dtype_str] is expected

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
        assert utils.map_external_dtype_to_torch(dtype_str) is expected

    def test_map_external_dtype_unknown_string_raises(self):
        with pytest.raises(RuntimeError, match="Unrecognized dtype"):
            utils.map_external_dtype_to_torch("not_a_dtype")


class TestMapExternalDtypeToTorch:
    """Tests for other map_external_dtype_to_torch paths."""

    def test_torch_dtype_passthrough(self):
        assert utils.map_external_dtype_to_torch(torch.float32) is torch.float32

    @pytest.mark.parametrize(
        "np_dtype, expected",
        [
            (np.float32, torch.float32),
            (np.int64, torch.int64),
            (np.bool_, torch.bool),
        ],
    )
    def test_numpy_scalar_type(self, np_dtype, expected):
        assert utils.map_external_dtype_to_torch(np_dtype) is expected

    @pytest.mark.parametrize(
        "np_dtype_str, expected",
        [
            ("float32", torch.float32),
            ("int16", torch.int16),
        ],
    )
    def test_numpy_dtype_instance(self, np_dtype_str, expected):
        assert utils.map_external_dtype_to_torch(np.dtype(np_dtype_str)) is expected


def _fresh_str(value: str) -> str:
    """Return a string equal to ``value`` but guaranteed to be a distinct object.

    ``str.join`` builds the result at runtime rather than reusing an interned
    literal, so ``_fresh_str("x") == "x"`` is True while ``_fresh_str("x") is "x"``
    is False.
    """
    return "".join(list(value))


class TestStarfishDtypeStringComparison:
    """Regression tests for starfish (NeuronXCC) dtype mapping.

    NKI emits dtypes as plain ``str`` objects. The ``_ndt`` attributes they
    must be matched against are also strings, so ``map_external_dtype_to_torch``
    must compare with ``==`` (equality), not ``is`` (identity).
    """

    class _NdtStub:
        """Stand-in for ``neuronxcc.starfish.support.dtype`` (string dtypes)."""

        bfloat16 = "bfloat16"
        float8_e5m2 = "float8_e5m2"
        float8_e4m3 = "float8_e4m3"
        float8_e4m3fn = "float8_e4m3fn"
        float8_e8m0fnu = "float8_e8m0fnu"

    @pytest.fixture
    def _patched(self, monkeypatch):
        """Install the string-dtype stub as ``_ndt`` in the utils module."""
        monkeypatch.setattr(utils, "_ndt", self._NdtStub, raising=False)
        return utils

    def test_bfloat16_string_matches(self, _patched):
        assert _patched.map_external_dtype_to_torch(_fresh_str("bfloat16")) is torch.bfloat16

    def test_float8_e5m2_string_matches(self, _patched):
        assert _patched.map_external_dtype_to_torch(_fresh_str("float8_e5m2")) is torch.float8_e5m2

    def test_float8_e4m3_string_matches_on_trn2(self, _patched, monkeypatch):
        """The reported bug: nl.float8_e4m3 output on trn2 must map to torch.float8_e4m3fn."""
        monkeypatch.setattr(_patched, "is_float8_e4m3_supported", lambda: True)
        result = _patched.map_external_dtype_to_torch(_fresh_str("float8_e4m3"))
        assert result is torch.float8_e4m3fn

    def test_float8_e4m3_string_raises_when_unsupported(self, _patched, monkeypatch):
        monkeypatch.setattr(_patched, "is_float8_e4m3_supported", lambda: False)
        monkeypatch.setattr(_patched, "get_platform_target", lambda: "trn3")
        with pytest.raises(RuntimeError, match="Unexpected dtype"):
            _patched.map_external_dtype_to_torch(_fresh_str("float8_e4m3"))

    def test_float8_e4m3fn_string_matches_when_supported(self, _patched, monkeypatch):
        monkeypatch.setattr(_patched, "is_float8_e4m3fn_supported", lambda: True)
        result = _patched.map_external_dtype_to_torch(_fresh_str("float8_e4m3fn"))
        assert result is torch.float8_e4m3fn

    @pytest.mark.skipif(
        not hasattr(torch, "float8_e8m0fnu"),
        reason="torch.float8_e8m0fnu not available in this PyTorch build",
    )
    def test_float8_e8m0fnu_string_matches_when_supported(self, _patched, monkeypatch):
        monkeypatch.setattr(_patched, "is_float8_e8m0fnu_supported", lambda: True)
        result = _patched.map_external_dtype_to_torch(_fresh_str("float8_e8m0fnu"))
        assert result is torch.float8_e8m0fnu
