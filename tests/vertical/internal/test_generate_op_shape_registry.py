"""Unit tests for generate_op_shape_registry cache key parser."""

from __future__ import annotations

from .generate_op_shape_registry import (
    _parse_tensor_token,
    _parse_token,
    _tokenize_signature,
    collect_from_cache_keys_file,
    parse_cache_key,
)


class TestParseTensorToken:
    def test_basic_tensor(self):
        t = _parse_tensor_token("[1,2048,4096]:BFloat16:S(C)")
        assert t is not None
        assert t.shape == [1, 2048, 4096]
        assert t.dtype == "bfloat16"
        assert t.contiguous is True

    def test_scalar_tensor(self):
        t = _parse_tensor_token("[]:BFloat16:S(C)")
        assert t is not None
        assert t.shape == []

    def test_non_contiguous(self):
        t = _parse_tensor_token("[2,3]:Float:S(3,1)")
        assert t is not None
        assert t.contiguous is False
        assert t.strides == [3, 1]

    def test_not_a_tensor(self):
        assert _parse_tensor_token("i42") is None
        assert _parse_tensor_token("None") is None


class TestParseToken:
    def test_tensor_returns_shape(self):
        assert _parse_token("[1,2048,4096]:BFloat16:S(C)") == [1, 2048, 4096]

    def test_scalar_tensor(self):
        assert _parse_token("[]:BFloat16:S(C)") == []

    def test_none(self):
        assert _parse_token("None") is None

    def test_undef(self):
        assert _parse_token("undef") is None

    def test_bool(self):
        assert _parse_token("b0") is False
        assert _parse_token("b1") is True

    def test_static_int(self):
        assert _parse_token("i42") == 42
        assert _parse_token("i0") == 0

    def test_dynamic_int(self):
        assert _parse_token("int") is None

    def test_static_float(self):
        assert _parse_token("d1e-06") == 1e-06
        assert _parse_token("d0.5") == 0.5

    def test_dynamic_float(self):
        assert _parse_token("float") is None

    def test_string(self):
        assert _parse_token("s:floor") == "floor"

    def test_dtype(self):
        assert _parse_token("dtype:BFloat16") == "bfloat16"
        assert _parse_token("dtype:Float") == "float32"

    def test_device(self):
        assert _parse_token("dev:neuron:0") == "neuron:0"

    def test_int_list(self):
        assert _parse_token("IL[i4096]") == [4096]
        assert _parse_token("IL[i1,i2,i3]") == [1, 2, 3]
        assert _parse_token("IL[]") == []

    def test_bool_list(self):
        assert _parse_token("BL[b1,b1,b0]") == [True, True, False]

    def test_double_list(self):
        assert _parse_token("DL[d1.0,d2.0]") == [1.0, 2.0]

    def test_tensor_list(self):
        result = _parse_token("TL[[64,2]:BFloat16:S(C),[64,4096]:BFloat16:S(C)]")
        assert result == [[64, 2], [64, 4096]]


class TestTokenizeSignature:
    def test_simple_args(self):
        tokens = _tokenize_signature("_[1,2]:Float:S(C)_i42_None")
        assert len(tokens) == 3
        assert tokens[0] == (None, "[1,2]:Float:S(C)")
        assert tokens[1] == (None, "i42")
        assert tokens[2] == (None, "None")

    def test_kwargs(self):
        tokens = _tokenize_signature("_[1]:Float:S(C)_kw:scale=d0.5")
        assert len(tokens) == 2
        assert tokens[1] == ("scale", "d0.5")

    def test_nested_brackets(self):
        tokens = _tokenize_signature("_[1,2]:Float:S(3,1)_i0")
        assert len(tokens) == 2
        assert tokens[0] == (None, "[1,2]:Float:S(3,1)")

    def test_tensor_list(self):
        tokens = _tokenize_signature("_TL[[1]:Float:S(C),[2]:Float:S(C)]_i0")
        assert len(tokens) == 2
        assert "TL[" in tokens[0][1]


class TestParseCacheKey:
    def test_basic(self):
        result = parse_cache_key("linear--_[1,2048,4096]:BFloat16:S(C)_[4096,4096]:BFloat16:S(C)")
        assert result is not None
        assert result.op_name == "linear"
        assert len(result.positional) == 2

    def test_with_hash(self):
        key = (
            "zero_--_[2]:BFloat16:S(C)"
            ":a0c5bd8381c4ce4119b6a16cff9dea0b86537da57e0686c07b75fc5b14c73a04"
        )
        result = parse_cache_key(key)
        assert result is not None
        assert result.positional[0] == [2]

    def test_with_kwargs(self):
        key = "_fused_rms_norm--_[1,2048,4096]:BFloat16:S(C)_IL[i4096]_[4096]:BFloat16:S(C)_d1e-06"
        result = parse_cache_key(key)
        assert result is not None
        assert result.op_name == "_fused_rms_norm"
        assert len(result.positional) == 4

    def test_nki_filtered(self):
        assert parse_cache_key("nki_20_something--_[1]:Float:S(C)") is None

    def test_no_separator(self):
        assert parse_cache_key("no_separator_here") is None

    def test_prologue(self):
        result = parse_cache_key("prologue--i0_1x2048x4096x_bf16")
        assert result is not None
        assert result.op_name == "prologue"


class TestCollectFromCacheKeysFile:
    def test_basic(self, tmp_path):
        cache_file = tmp_path / "cache_keys.txt"
        cache_file.write_text(
            "linear--_[1,2048,4096]:BFloat16:S(C)_[4096,4096]:BFloat16:S(C):abc123\n"
            "nki_20_something--_[1]:Float:S(C)\n"
            "prologue--i0_stuff\n"
            "add.Tensor--_[1,2048]:BFloat16:S(C)_[1,2048]:BFloat16:S(C)_i1\n"
        )
        results = collect_from_cache_keys_file(str(cache_file))
        # nki and prologue filtered, linear hash stripped but no aten:: prefix from line 1
        op_names = {r.op_name for r in results}
        assert "aten::linear" in op_names
        assert "aten::add.Tensor" in op_names
        assert len(results) == 2
