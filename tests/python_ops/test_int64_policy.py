"""Unit tests for torch_neuronx.int64_policy (unified taxonomy).

Modes: native (default, per-op cascade), cpu_bounce, packed, native_strict.
The per-op Int64Policy (decide/should_cpu_fallback/compiler flags) only governs
the native dispatch modes (native / native_strict); packed and cpu_bounce never
force a whole-op CPU fallback and never pass S64 to the compiler.
"""

import logging

import pytest

from torch_neuronx.int64_policy import (
    _NATIVE_INT64_OPS,
    _UNSUPPORTED_INT64_OPS,
    Int64Decision,
    Int64Mode,
    Int64Policy,
    _strip_overload,
    get_int64_mode,
    get_native_int64_op_set,
    get_unsupported_int64_op_set,
    reset_int64_mode_cache,
)


@pytest.fixture(autouse=True)
def reset_policy_cache():
    """Reset the cached mode + policy before and after each test."""
    reset_int64_mode_cache()
    yield
    reset_int64_mode_cache()


def _set_mode(monkeypatch, mode: str) -> None:
    monkeypatch.setenv("TORCH_NEURONX_INT64_MODE", mode)
    reset_int64_mode_cache()


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════


class TestStripOverload:
    def test_no_overload(self):
        assert _strip_overload("aten::add") == "aten::add"

    def test_with_overload(self):
        assert _strip_overload("aten::add.Tensor") == "aten::add"

    def test_with_complex_overload(self):
        assert _strip_overload("aten::index_put.out") == "aten::index_put"

    def test_no_namespace(self):
        assert _strip_overload("add") == "add"

    def test_double_colon_no_overload(self):
        assert _strip_overload("aten::bitwise_and") == "aten::bitwise_and"


# ═══════════════════════════════════════════════════════════════
#  get_int64_mode()
# ═══════════════════════════════════════════════════════════════


class TestGetInt64Mode:
    def test_default_is_native(self, monkeypatch):
        monkeypatch.delenv("TORCH_NEURONX_INT64_MODE", raising=False)
        reset_int64_mode_cache()
        assert get_int64_mode() == Int64Mode.NATIVE

    def test_explicit_packed(self, monkeypatch):
        _set_mode(monkeypatch, "packed")
        assert get_int64_mode() == Int64Mode.PACKED

    def test_cpu_bounce(self, monkeypatch):
        _set_mode(monkeypatch, "cpu_bounce")
        assert get_int64_mode() == Int64Mode.CPU_BOUNCE

    def test_explicit_native(self, monkeypatch):
        _set_mode(monkeypatch, "native")
        assert get_int64_mode() == Int64Mode.NATIVE

    def test_native_strict(self, monkeypatch):
        _set_mode(monkeypatch, "native_strict")
        assert get_int64_mode() == Int64Mode.NATIVE_STRICT

    def test_cpu_fallback(self, monkeypatch):
        _set_mode(monkeypatch, "cpu_fallback")
        assert get_int64_mode() == Int64Mode.CPU_FALLBACK

    def test_case_insensitive(self, monkeypatch):
        _set_mode(monkeypatch, "Native_Strict")
        assert get_int64_mode() == Int64Mode.NATIVE_STRICT

    def test_unknown_raises(self, monkeypatch):
        _set_mode(monkeypatch, "garbage")
        with pytest.raises(ValueError, match="Invalid TORCH_NEURONX_INT64_MODE"):
            get_int64_mode()


# ═══════════════════════════════════════════════════════════════
#  decide()
# ═══════════════════════════════════════════════════════════════


class TestDecide:
    def test_native_mode_native_op(self, monkeypatch):
        _set_mode(monkeypatch, "native")
        p = Int64Policy.get()
        assert p.decide("aten::add") == Int64Decision.NATIVE
        assert p.decide("aten::add.Tensor") == Int64Decision.NATIVE
        assert p.decide("aten::mul.Scalar") == Int64Decision.NATIVE
        assert p.decide("aten::add.out") == Int64Decision.NATIVE
        assert p.decide("aten::prod") == Int64Decision.NATIVE
        assert p.decide("aten::amax") == Int64Decision.NATIVE
        assert p.decide("aten::bitwise_and") == Int64Decision.NATIVE

    def test_native_mode_downcast_op(self, monkeypatch):
        _set_mode(monkeypatch, "native")
        p = Int64Policy.get()
        assert p.decide("aten::pow") == Int64Decision.DOWNCAST
        # sum is intentionally demoted from native (native int64 reduce_add is
        # ~250x slower than the packed path per benchmark).
        assert p.decide("aten::sum") == Int64Decision.DOWNCAST
        assert p.decide("aten::index") == Int64Decision.DOWNCAST
        assert p.decide("aten::scatter") == Int64Decision.DOWNCAST
        assert p.decide("aten::some_unknown_op") == Int64Decision.DOWNCAST

    def test_native_mode_unsupported_op(self, monkeypatch):
        _set_mode(monkeypatch, "native")
        p = Int64Policy.get()
        assert p.decide("aten::mm") == Int64Decision.UNSUPPORTED
        assert p.decide("aten::mm.out") == Int64Decision.UNSUPPORTED
        assert p.decide("aten::matmul") == Int64Decision.UNSUPPORTED
        assert p.decide("aten::conv2d") == Int64Decision.UNSUPPORTED

    def test_native_strict_downcast_op_unsupported(self, monkeypatch):
        _set_mode(monkeypatch, "native_strict")
        p = Int64Policy.get()
        assert p.decide("aten::pow") == Int64Decision.UNSUPPORTED
        assert p.decide("aten::some_unknown_op") == Int64Decision.UNSUPPORTED

    def test_native_strict_native_op_runs(self, monkeypatch):
        _set_mode(monkeypatch, "native_strict")
        assert Int64Policy.get().decide("aten::add") == Int64Decision.NATIVE

    def test_packed_mode_everything_downcasts(self, monkeypatch):
        # packed / cpu_bounce don't pass S64 and don't force CPU fallback, so
        # the policy classifies every op as DOWNCAST (the existing packed /
        # CPU-bounce paths handle the actual int64 → int32 demotion).
        _set_mode(monkeypatch, "packed")
        p = Int64Policy.get()
        assert p.decide("aten::add") == Int64Decision.DOWNCAST
        assert p.decide("aten::pow") == Int64Decision.DOWNCAST
        assert p.decide("aten::mm") == Int64Decision.DOWNCAST

    def test_cpu_bounce_mode_everything_downcasts(self, monkeypatch):
        _set_mode(monkeypatch, "cpu_bounce")
        p = Int64Policy.get()
        assert p.decide("aten::add") == Int64Decision.DOWNCAST
        assert p.decide("aten::pow") == Int64Decision.DOWNCAST
        assert p.decide("aten::mm") == Int64Decision.DOWNCAST

    def test_cpu_fallback_mode_everything_unsupported(self, monkeypatch):
        # cpu_fallback runs every int64 op on CPU → all decisions are UNSUPPORTED.
        _set_mode(monkeypatch, "cpu_fallback")
        p = Int64Policy.get()
        assert p.decide("aten::add") == Int64Decision.UNSUPPORTED
        assert p.decide("aten::pow") == Int64Decision.UNSUPPORTED
        assert p.decide("aten::mm") == Int64Decision.UNSUPPORTED


# ═══════════════════════════════════════════════════════════════
#  should_cpu_fallback()
# ═══════════════════════════════════════════════════════════════


class TestShouldCpuFallback:
    def test_unsupported_falls_back_in_native_modes(self, monkeypatch):
        for mode in ("native", "native_strict"):
            _set_mode(monkeypatch, mode)
            assert Int64Policy.get().should_cpu_fallback("aten::mm") is True

    def test_unsupported_no_fallback_in_packed_or_cpu_bounce(self, monkeypatch):
        # packed / cpu_bounce keep their existing on-device handling; the
        # policy does not force a whole-op CPU fallback for them.
        for mode in ("packed", "cpu_bounce"):
            _set_mode(monkeypatch, mode)
            assert Int64Policy.get().should_cpu_fallback("aten::mm") is False

    def test_native_mode_native_op_no_fallback(self, monkeypatch):
        _set_mode(monkeypatch, "native")
        assert Int64Policy.get().should_cpu_fallback("aten::add") is False

    def test_native_mode_downcast_op_no_fallback(self, monkeypatch):
        _set_mode(monkeypatch, "native")
        assert Int64Policy.get().should_cpu_fallback("aten::pow") is False

    def test_native_strict_downcast_op_falls_back(self, monkeypatch):
        _set_mode(monkeypatch, "native_strict")
        assert Int64Policy.get().should_cpu_fallback("aten::pow") is True

    def test_native_strict_native_op_no_fallback(self, monkeypatch):
        _set_mode(monkeypatch, "native_strict")
        assert Int64Policy.get().should_cpu_fallback("aten::add") is False

    def test_packed_and_cpu_bounce_never_fallback(self, monkeypatch):
        for mode in ("packed", "cpu_bounce"):
            _set_mode(monkeypatch, mode)
            p = Int64Policy.get()
            assert p.should_cpu_fallback("aten::add") is False
            assert p.should_cpu_fallback("aten::pow") is False
            assert p.should_cpu_fallback("aten::mm") is False

    def test_cpu_fallback_mode_every_op_falls_back(self, monkeypatch):
        # cpu_fallback runs every int64 op entirely on CPU — including the
        # natively-supported ones (unlike cpu_bounce, which keeps them on device).
        _set_mode(monkeypatch, "cpu_fallback")
        p = Int64Policy.get()
        assert p.should_cpu_fallback("aten::add") is True
        assert p.should_cpu_fallback("aten::pow") is True
        assert p.should_cpu_fallback("aten::mm") is True


# ═══════════════════════════════════════════════════════════════
#  get_compiler_flags()
# ═══════════════════════════════════════════════════════════════


class TestCompilerFlags:
    def test_native_op_flags(self, monkeypatch):
        _set_mode(monkeypatch, "native")
        flags = Int64Policy.get().get_compiler_flags("aten::add")
        assert flags == ["--native-int64=enabled", "--implicit-integer-downcast=all"]

    def test_downcast_op_flags(self, monkeypatch):
        # In the native cascade, a DOWNCAST op (pow) is demoted to int32 on
        # device via the packed view before lowering, so the compiler sees
        # int32 and needs no int64 flags.
        _set_mode(monkeypatch, "native")
        flags = Int64Policy.get().get_compiler_flags("aten::pow")
        assert flags == []

    def test_unsupported_op_no_flags(self, monkeypatch):
        _set_mode(monkeypatch, "native")
        assert Int64Policy.get().get_compiler_flags("aten::mm") == []

    def test_packed_no_flags(self, monkeypatch):
        _set_mode(monkeypatch, "packed")
        assert Int64Policy.get().get_compiler_flags("aten::add") == []
        assert Int64Policy.get().get_compiler_flags("aten::pow") == []

    def test_cpu_bounce_no_flags(self, monkeypatch):
        _set_mode(monkeypatch, "cpu_bounce")
        assert Int64Policy.get().get_compiler_flags("aten::add") == []


# ═══════════════════════════════════════════════════════════════
#  passes_s64_to_compiler / native_int64_enabled
# ═══════════════════════════════════════════════════════════════


class TestPassesS64ToCompiler:
    def test_native(self, monkeypatch):
        _set_mode(monkeypatch, "native")
        assert Int64Policy.get().passes_s64_to_compiler is True

    def test_native_strict(self, monkeypatch):
        _set_mode(monkeypatch, "native_strict")
        assert Int64Policy.get().passes_s64_to_compiler is True

    def test_packed(self, monkeypatch):
        _set_mode(monkeypatch, "packed")
        assert Int64Policy.get().passes_s64_to_compiler is False

    def test_cpu_bounce(self, monkeypatch):
        _set_mode(monkeypatch, "cpu_bounce")
        assert Int64Policy.get().passes_s64_to_compiler is False

    def test_native_int64_enabled_alias(self, monkeypatch):
        for mode, expected in [
            ("native", True),
            ("native_strict", True),
            ("packed", False),
            ("cpu_bounce", False),
        ]:
            _set_mode(monkeypatch, mode)
            assert Int64Policy.get().native_int64_enabled is expected


# ═══════════════════════════════════════════════════════════════
#  Warnings
# ═══════════════════════════════════════════════════════════════


class TestWarning:
    def test_native_mode_downcast_warns(self, monkeypatch, caplog):
        _set_mode(monkeypatch, "native")
        with caplog.at_level(logging.WARNING):
            Int64Policy.get().warn_if_needed("aten::pow")
        assert "precision" in caplog.text.lower() or "downcast" in caplog.text.lower()

    def test_native_mode_native_op_no_warning(self, monkeypatch, caplog):
        _set_mode(monkeypatch, "native")
        with caplog.at_level(logging.WARNING):
            Int64Policy.get().warn_if_needed("aten::add")
        assert "precision" not in caplog.text.lower()

    def test_native_strict_no_warning(self, monkeypatch, caplog):
        _set_mode(monkeypatch, "native_strict")
        with caplog.at_level(logging.WARNING):
            Int64Policy.get().warn_if_needed("aten::pow")
        assert "precision" not in caplog.text.lower()


# ═══════════════════════════════════════════════════════════════
#  Op tables
# ═══════════════════════════════════════════════════════════════


class TestOpTables:
    def test_native_and_unsupported_are_disjoint(self):
        overlap = _NATIVE_INT64_OPS & _UNSUPPORTED_INT64_OPS
        assert overlap == set(), f"Ops in both native and unsupported: {overlap}"

    def test_native_ops_have_aten_prefix(self):
        for op in _NATIVE_INT64_OPS:
            assert op.startswith("aten::"), f"Op missing aten:: prefix: {op}"

    def test_unsupported_ops_have_aten_prefix(self):
        for op in _UNSUPPORTED_INT64_OPS:
            assert op.startswith("aten::"), f"Op missing aten:: prefix: {op}"

    def test_key_arithmetic_ops_are_native(self):
        for op in ["aten::add", "aten::sub", "aten::mul", "aten::div"]:
            assert op in _NATIVE_INT64_OPS

    def test_comparison_ops_not_native(self):
        # Disabled: native S64 comparison hangs on non-contiguous int64 (output
        # transpose epilogue), which the input-based guard can't gate.
        for op in ["aten::eq", "aten::ne", "aten::gt", "aten::ge", "aten::lt", "aten::le"]:
            assert op not in _NATIVE_INT64_OPS

    def test_matmul_is_unsupported(self):
        assert "aten::mm" in _UNSUPPORTED_INT64_OPS
        assert "aten::matmul" in _UNSUPPORTED_INT64_OPS


# ═══════════════════════════════════════════════════════════════
#  Caching
# ═══════════════════════════════════════════════════════════════


class TestInt64PolicyCaching:
    def test_get_returns_same_instance(self, monkeypatch):
        _set_mode(monkeypatch, "native")
        p1 = Int64Policy.get()
        p2 = Int64Policy.get()
        assert p1 is p2

    def test_reset_cache_creates_new_instance(self, monkeypatch):
        _set_mode(monkeypatch, "native")
        p1 = Int64Policy.get()
        monkeypatch.setenv("TORCH_NEURONX_INT64_MODE", "cpu_bounce")
        reset_int64_mode_cache()
        p2 = Int64Policy.get()
        assert p1 is not p2
        assert p1.mode == Int64Mode.NATIVE
        assert p2.mode == Int64Mode.CPU_BOUNCE


# ═══════════════════════════════════════════════════════════════
#  C++ interop
# ═══════════════════════════════════════════════════════════════


class TestCppInterop:
    def test_get_native_set_returns_frozenset(self):
        result = get_native_int64_op_set()
        assert isinstance(result, frozenset)
        assert "aten::add" in result

    def test_get_unsupported_set_returns_frozenset(self):
        result = get_unsupported_int64_op_set()
        assert isinstance(result, frozenset)
        assert "aten::mm" in result

    def test_init_cpp_policy_does_not_raise(self):
        from torch_neuronx.int64_policy import init_cpp_policy

        init_cpp_policy()
