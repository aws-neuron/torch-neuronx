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
        assert _parse_token("[1,2048,4096]:BFloat16:S(C)") == {
            "shape": [1, 2048, 4096],
            "dtype": "bfloat16",
        }

    def test_scalar_tensor(self):
        assert _parse_token("[]:BFloat16:S(C)") == {"shape": [], "dtype": "bfloat16"}

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
        assert _parse_token("dev:neuron:0") is None

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
        assert result == [
            {"shape": [64, 2], "dtype": "bfloat16"},
            {"shape": [64, 4096], "dtype": "bfloat16"},
        ]

    def test_empty_tensor_list(self):
        # Empty TL[]/GL[] must parse to [] (legitimate empty list), not the
        # literal token string — otherwise downstream _build_value fails when
        # the schema expects Tensor[] (e.g. aten::_fused_adamw_.max_exp_avg_sqs
        # when AMSGrad is disabled).
        assert _parse_token("TL[]") == []
        assert _parse_token("GL[]") == []


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
        assert result.positional[0] == {"shape": [2], "dtype": "bfloat16"}

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


class TestScalarDefaultFilling:
    """Tests for the default-filling pass in ``_build_arg_shapes``.

    Cache keys encode dynamic (SymFloat/SymInt) scalars as None; this pass
    replaces None with a concrete value for required non-Tensor args so the
    registered entry can actually execute at test time.
    """

    def _build(self, op_name, positional=None, kwargs=None):
        """Run _build_arg_shapes for a given op with parsed positional + kwargs."""
        from .generate_op_shape_registry import ParsedCacheKey, _build_arg_shapes

        parsed = ParsedCacheKey(
            op_name=op_name,
            positional=positional or [],
            kwargs=kwargs or {},
        )
        return _build_arg_shapes(op_name, parsed)

    def test_op_specific_fused_adamw(self):
        """_fused_adamw_'s lr/beta1/beta2/weight_decay/eps get filled with
        canonical AdamW hyperparameters when the cache key captured them as
        dynamic."""
        args = self._build(
            "aten::_fused_adamw_",
            positional=[
                [[256, 4096]],  # self
                [[256, 4096]],  # grads
                [[256, 4096]],  # exp_avgs
                [[256, 4096]],  # exp_avg_sqs
                [],  # max_exp_avg_sqs
                [[]],  # state_steps
            ],
            kwargs={
                "lr": None,
                "beta1": None,
                "beta2": None,
                "weight_decay": None,
                "eps": None,
                "amsgrad": None,
                "maximize": None,
            },
        )
        # Required scalars must be concrete.
        assert args["lr"] == 0.001
        assert args["beta1"] == 0.9
        assert args["beta2"] == 0.999
        assert args["weight_decay"] == 0.01
        assert args["eps"] == 1e-8
        assert args["amsgrad"] is False
        assert args["maximize"] is False
        # Optional Tensor kwargs stay None.
        assert args["grad_scale"] is None
        assert args["found_inf"] is None

    def test_name_scoped_eps_on_norm_op(self):
        """A norm op's `eps` gets the norm-canonical value 1e-5, not 1e-8."""
        args = self._build(
            "aten::layer_norm",
            positional=[
                [1, 4, 16],  # input
                [16],  # normalized_shape
                [16],  # weight
                [16],  # bias
                None,  # eps captured as dynamic
            ],
        )
        assert args["eps"] == 1e-5

    def test_type_based_fallback(self):
        """For an op/arg combo not in any table, the type-based fallback kicks
        in and fills ``float`` → 0.0, ``int`` → 0, ``bool`` → False."""
        # Use a simple op with scalar args not in our tables.
        args = self._build(
            "aten::clamp",
            positional=[
                [4, 4],  # self
                None,  # min (Optional[Scalar])
                None,  # max (Optional[Scalar])
            ],
        )
        # min/max are Optional → stay None (not filled).
        assert args["min"] is None
        assert args["max"] is None

    def test_optional_tensor_stays_none(self):
        """Optional[Tensor] args don't get a filled default — None is valid."""
        args = self._build(
            "aten::linear",
            positional=[[2, 16], [32, 16], None],  # input, weight, bias=None
        )
        assert args["bias"] is None

    def test_tensor_null_stays_null(self):
        """Required Tensor args captured as None stay None — the consumer
        (sample build) decides what to do. We don't invent shapes."""
        args = self._build(
            "aten::add.Tensor",
            positional=[None, [4, 4]],  # self=None, other=shape
        )
        assert args["self"] is None


class TestKwargUnderscoreParsing:
    """Bug fix: tokenizer must not split on '_' inside kw:name=value when name contains '_'."""

    def test_ignore_index_kwarg(self):
        tokens = _tokenize_signature(
            "_[1,2048]:BFloat16:S(C)_[1,2048]:Long:S(C)_kw:ignore_index=i-100_kw:reduction=i1"
        )
        kw_tokens = {name: val for name, val in tokens if name}
        assert "ignore_index" in kw_tokens
        assert kw_tokens["ignore_index"] == "i-100"
        assert kw_tokens["reduction"] == "i1"

    def test_use_deterministic_algorithms(self):
        tokens = _tokenize_signature("_[4,4]:Float:S(C)_kw:use_deterministic_algorithms=b0")
        kw_tokens = {name: val for name, val in tokens if name}
        assert "use_deterministic_algorithms" in kw_tokens
        assert kw_tokens["use_deterministic_algorithms"] == "b0"

    def test_cross_entropy_full_parse(self):
        ck = (
            "cross_entropy_loss--_[1,2048,4096]:BFloat16:S(C)_[1,2048]:Long:S(C)"
            "_kw:ignore_index=i-100_kw:reduction=i1"
            ":abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
        )
        parsed = parse_cache_key(ck)
        assert parsed is not None
        assert parsed.kwargs["ignore_index"] == -100
        assert parsed.kwargs["reduction"] == 1


class TestScalarTensorConversion:
    """Bug fix: 0-d tensor specs must be converted to numeric for Scalar-typed args."""

    def test_0d_tensor_parse(self):
        result = _parse_token("[]:Int:S(C)")
        assert result == {"shape": [], "dtype": "int32"}

    def test_0d_tensor_converted_for_scalar_arg(self):
        from .generate_op_shape_registry import ParsedCacheKey, _build_arg_shapes

        # _foreach_add_.Scalar expects (Tensor[] self, Scalar scalar)
        # Cache key encodes scalar as 0-d tensor []:Int:S(C)
        parsed = ParsedCacheKey(
            op_name="aten::_foreach_add_.Scalar",
            positional=[
                [{"shape": [4096], "dtype": "bfloat16"}],  # self (Tensor[])
                {"shape": [], "dtype": "int32"},  # scalar (0-d tensor → None → filled by defaults)
            ],
            kwargs={},
        )
        args = _build_arg_shapes("aten::_foreach_add_.Scalar", parsed)
        # The Scalar arg should be numeric, not a dict
        assert isinstance(args["scalar"], int | float)


class TestNonKwargOnlyFromKwargs:
    """Bug fix: non-kwarg_only args should check parsed.kwargs as fallback."""

    def test_positional_arg_from_kwargs(self):
        from .generate_op_shape_registry import ParsedCacheKey, _build_arg_shapes

        # Simulate a case where a positional arg ended up in kwargs
        parsed = ParsedCacheKey(
            op_name="aten::add.Tensor",
            positional=[[4, 4]],  # only self
            kwargs={"other": [8, 8]},  # other ended up as kwarg
        )
        args = _build_arg_shapes("aten::add.Tensor", parsed)
        assert args["other"] == [8, 8]
