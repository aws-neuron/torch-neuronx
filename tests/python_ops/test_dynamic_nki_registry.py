"""Unit tests for the dynamic-NKI dispatch infrastructure (PR #1250).

CPU-only — exercises the registry + compile_key contracts with a stub
compile_fn, so no Neuron device or real kernel is needed. Covers the
behavioural guarantees the dispatch path relies on:

  - register / get round-trip, gated by the env flag
  - get_compiled memoizes ONE NEFF per compile_key (compile_fn called once)
  - distinct compile_keys -> distinct NEFFs (the per-H/dtype guarantee)
  - compile_key_fn None -> single NEFF under key None
"""

import pytest

from torch_neuronx.python_ops.torch_mlir.ops.dynamic_kernels.registry import (
    DynamicNkiEntry,
    get_dynamic_nki_entry,
    register_dynamic_nki,
)


def _entry(counter):
    """A DynamicNkiEntry whose compile_fn just records how often it ran and
    returns a distinct fake (hlo_bytes, cache_key) per compile_key."""

    def compile_fn(compile_key):
        counter.append(compile_key)
        return (b"hlo", f"cache_key_for_{compile_key}")

    return DynamicNkiEntry(
        compile_fn=compile_fn,
        can_handle=lambda *a, **k: True,
        prepare_io=lambda *a, **k: ((), (), 0),
    )


class TestDynamicNkiRegistry:
    def test_register_and_get_when_enabled(self, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS", "1")
        e = _entry([])
        register_dynamic_nki("test_op_enabled", e)
        assert get_dynamic_nki_entry("test_op_enabled") is e

    def test_get_returns_none_when_disabled(self, monkeypatch):
        # Even a registered op is invisible when the feature flag is off, so the
        # dispatch fast path stays inert on the default (feature-off) build.
        monkeypatch.setenv("TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS", "0")
        register_dynamic_nki("test_op_disabled", _entry([]))
        assert get_dynamic_nki_entry("test_op_disabled") is None

    def test_get_unregistered_returns_none(self, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS", "1")
        assert get_dynamic_nki_entry("op_never_registered") is None

    def test_compile_once_per_key(self):
        # Same compile_key -> compile_fn runs ONCE; result is memoized and reused.
        calls = []
        e = _entry(calls)
        r1 = e.get_compiled((2048, "bf16"))
        r2 = e.get_compiled((2048, "bf16"))
        assert r1 == r2
        assert calls == [(2048, "bf16")], "compile_fn should run exactly once per key"

    def test_distinct_keys_distinct_neffs(self):
        # Different compile_keys (e.g. different H / dtype) each get their own NEFF.
        calls = []
        e = _entry(calls)
        r_a = e.get_compiled((2048, "bf16"))
        r_b = e.get_compiled((4096, "bf16"))
        r_c = e.get_compiled((2048, "fp32"))
        assert len({r_a[1], r_b[1], r_c[1]}) == 3, "each key -> distinct cache_key"
        assert len(calls) == 3
        # re-fetching any prior key does not recompile
        e.get_compiled((2048, "bf16"))
        assert len(calls) == 3

    def test_compile_key_none(self):
        # compile_key_fn is optional; with key None a single NEFF is compiled.
        calls = []
        e = _entry(calls)
        r1 = e.get_compiled(None)
        r2 = e.get_compiled()  # default arg is None
        assert r1 == r2
        assert calls == [None]

    def test_compile_key_fn_default_none(self):
        # An entry without compile_key_fn set exposes None (dispatch passes None).
        e = _entry([])
        assert e.compile_key_fn is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
