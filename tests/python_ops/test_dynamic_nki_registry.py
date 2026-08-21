"""Unit tests for the dynamic-NKI dispatch infrastructure.

CPU-only — exercises the registry + compile_key contracts and the
try_dynamic_nki_dispatch kwarg handling with stubs, so no Neuron device or real
kernel is needed. Covers:

  - register / get round-trip, gated by the env flag
  - get_compiled memoizes ONE NEFF per compile_key; distinct keys -> distinct NEFFs
  - compile_key_fn None -> single NEFF under key None
  - kwarg handling: out=/grad_input= as dest (offered to the gate as `out`, never in the
    compile_key), framework params dropped, op kwargs forwarded, can_handle as the gate
"""

import pytest
import torch

import torch_neuronx.python_ops.torch_mlir.ops.dynamic_kernels.dispatch_entry as de
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


class FakeTensor:
    """Stub tensor: just .shape and .reshape, enough for the dispatch code to run
    (it logs shapes and reshapes the output). No real data / device."""

    shape = (0,)

    def reshape(self, *_):
        return self


# One output tuple every stub prepare_io can return: (inputs, outputs, actual_M).
_STUB_IO = ((FakeTensor(),), (FakeTensor(),), 0)


def _register_stub_op(op_name, can_handle, prepare_io, compile_key_fn=None, compile_fn=None):
    """Register a fully-stubbed dynamic-NKI op so we can drive try_dynamic_nki_dispatch
    without a real kernel or device."""
    register_dynamic_nki(
        op_name,
        DynamicNkiEntry(
            compile_fn=compile_fn or (lambda key: (b"hlo", f"cache::{key}")),
            can_handle=can_handle,
            prepare_io=prepare_io,
            reshape_output=lambda outputs, args: outputs[0],
            compile_key_fn=compile_key_fn,
        ),
    )


class TestDynamicNkiDispatchKwargs:
    """How try_dynamic_nki_dispatch splits and forwards kwargs. can_handle is the
    sole gate; compile_key_fn / prepare_io run only after it accepts."""

    @pytest.fixture(autouse=True)
    def _enabled(self, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS", "1")
        # execute_compiled_graph is the one device binding — stub it to a no-op.
        monkeypatch.setattr(de._C, "execute_compiled_graph", lambda *a, **k: None)

    def test_op_kwarg_forwarded_to_all_three_fns(self):
        # A real op kwarg (alpha) must reach can_handle, compile_key_fn, and prepare_io.
        alpha_seen = {}

        def can_handle(x, y, alpha=1):
            alpha_seen["can_handle"] = alpha
            return True

        def compile_key_fn(x, y, **kwargs):
            alpha_seen["compile_key_fn"] = kwargs.get("alpha")
            return ("key",)

        def prepare_io(x, y, alpha=1, dest=None):
            alpha_seen["prepare_io"] = alpha
            return _STUB_IO

        _register_stub_op("add_stub", can_handle, prepare_io, compile_key_fn)
        de.try_dynamic_nki_dispatch("add_stub", (FakeTensor(), FakeTensor()), {"alpha": 2})

        assert alpha_seen == {"can_handle": 2, "compile_key_fn": 2, "prepare_io": 2}

    def test_out_kwarg_becomes_dest_and_is_offered_to_the_gate(self):
        # out= reaches prepare_io as `dest` and can_handle as `out` (the gate must be able
        # to reject a dest the kernel can't write in place), never as an op kwarg.
        forwarded = {}

        def can_handle(x, **op_kwargs):
            forwarded["can_handle"] = dict(op_kwargs)
            return True

        def prepare_io(x, dest=None, **op_kwargs):
            forwarded["dest"] = dest
            forwarded["prepare_io"] = dict(op_kwargs)
            return _STUB_IO

        _register_stub_op("out_stub", can_handle, prepare_io)
        de.try_dynamic_nki_dispatch("out_stub", (FakeTensor(),), {"out": "BUFFER"})

        assert forwarded == {"can_handle": {"out": "BUFFER"}, "dest": "BUFFER", "prepare_io": {}}

    def test_grad_input_kwarg_is_offered_to_the_gate_as_out(self):
        # grad_input= is the same role under another aten name; the gate sees one name.
        seen = {}

        def can_handle(x, out=None):
            seen["out"] = out
            return True

        _register_stub_op("grad_input_stub", can_handle, lambda x, dest=None: _STUB_IO)
        de.try_dynamic_nki_dispatch("grad_input_stub", (FakeTensor(),), {"grad_input": "BUFFER"})

        assert seen == {"out": "BUFFER"}

    def test_no_dest_leaves_out_unset(self):
        # No dest -> no `out` kwarg, so a can_handle that doesn't declare one still works.
        _register_stub_op("no_dest_stub", lambda x: True, lambda x, dest=None: _STUB_IO)
        assert de.try_dynamic_nki_dispatch("no_dest_stub", (FakeTensor(),), {}) is not None

    def test_can_handle_declining_on_out_falls_back(self):
        # An adapter that can't serve this dest (e.g. non-contiguous out=) declines ->
        # dispatch returns None -> default lowering honours out=. prepare_io must not run.
        ran = []

        def prepare_io(x, dest=None):
            ran.append(dest)
            return _STUB_IO

        _register_stub_op("picky_dest_stub", lambda x, out=None: out is None, prepare_io)
        assert (
            de.try_dynamic_nki_dispatch("picky_dest_stub", (FakeTensor(),), {"out": "BAD"}) is None
        )
        assert ran == [], "prepare_io ran even though the gate declined"
        assert de.try_dynamic_nki_dispatch("picky_dest_stub", (FakeTensor(),), {}) is not None

    def test_dest_is_not_in_the_compile_key(self):
        # The dest is a runtime buffer: compile_key_fn never sees it, so out= vs no out=
        # share one NEFF.
        compiled_keys = []
        _register_stub_op(
            "dest_key_stub",
            can_handle=lambda x, out=None: True,
            prepare_io=lambda x, dest=None: _STUB_IO,
            compile_key_fn=lambda x: ("H",),
            compile_fn=lambda key: compiled_keys.append(key) or (b"hlo", f"cache::{key}"),
        )
        de.try_dynamic_nki_dispatch("dest_key_stub", (FakeTensor(),), {})
        de.try_dynamic_nki_dispatch("dest_key_stub", (FakeTensor(),), {"out": "BUFFER"})
        assert compiled_keys == [("H",)]

    def test_framework_params_are_dropped(self):
        # device / memory_format etc. aren't op args -> filtered out before dispatch.
        forwarded = {}

        def can_handle(x, **op_kwargs):
            forwarded.update(op_kwargs)
            return True

        _register_stub_op("framework_stub", can_handle, lambda *a, **k: _STUB_IO)
        de.try_dynamic_nki_dispatch(
            "framework_stub", (FakeTensor(),), {"device": "cpu", "memory_format": 1}
        )
        assert forwarded == {}

    def test_undeclared_op_kwarg_raises(self):
        # can_handle doesn't declare `alpha` (and has no **kwargs) -> forwarding it
        # raises (fail-fast: a missing guard), rather than being silently ignored.
        def can_handle_only_x(x):
            return True

        _register_stub_op("strict_stub", can_handle_only_x, lambda *a, **k: _STUB_IO)
        with pytest.raises(TypeError):
            de.try_dynamic_nki_dispatch("strict_stub", (FakeTensor(),), {"alpha": 2})

    def test_no_kwargs_runs_normally(self):
        _register_stub_op("plain_stub", lambda x: True, lambda x, dest=None: _STUB_IO)
        result = de.try_dynamic_nki_dispatch("plain_stub", (FakeTensor(),), None)
        assert isinstance(result, FakeTensor)

    def test_can_handle_false_declines(self):
        # can_handle rejects -> dispatch returns None (caller falls back to default).
        _register_stub_op("reject_stub", lambda *a, **k: False, lambda *a, **k: _STUB_IO)
        assert de.try_dynamic_nki_dispatch("reject_stub", (FakeTensor(),), {}) is None

    def test_kwarg_baked_into_compile_key_gives_one_neff_per_value(self):
        # When compile_key_fn includes the kwarg, each distinct value compiles its
        # own NEFF once and reuses it (recompile-free per value).
        compiled_keys = []

        _register_stub_op(
            "bake_stub",
            can_handle=lambda x, alpha=1: True,
            prepare_io=lambda x, alpha=1, dest=None: _STUB_IO,
            compile_key_fn=lambda x, alpha=1: ("H", alpha),
            compile_fn=lambda key: compiled_keys.append(key) or (b"hlo", f"cache::{key}"),
        )
        for alpha in (1, 3, 1):  # 1 repeats -> should reuse, not recompile
            de.try_dynamic_nki_dispatch("bake_stub", (FakeTensor(),), {"alpha": alpha})

        assert compiled_keys == [("H", 1), ("H", 3)]

    def test_compile_key_fn_missing_kwarg_raises(self):
        # can_handle accepts alpha but compile_key_fn neither bakes nor absorbs it
        # -> raise (the adapter is internally inconsistent).
        _register_stub_op(
            "inconsistent_stub",
            can_handle=lambda x, **kwargs: True,
            prepare_io=lambda x, **kwargs: _STUB_IO,
            compile_key_fn=lambda x: ("H",),
        )
        with pytest.raises(TypeError):
            de.try_dynamic_nki_dispatch("inconsistent_stub", (FakeTensor(),), {"alpha": 2})


class TestToCopyGate:
    """_can_handle_to_copy declines sub-128-M casts to the native path.

    The dynamic _to_copy avoids a pad only for M >= 128 (ragged-tail overlap); a sub-tile M
    would take a fresh [128, H] zeros pad + slice-assign, which is wasteful for a tiny cast
    and has faulted on device (NRT 1004) on some contiguous collective tensors (FSDP
    reduce-scatter chunk_cat). _is_castable is stubbed True here to isolate the M gate from
    the neuron-device / contiguity checks (those need a real device).
    """

    def _can_handle(self, monkeypatch, shape):
        from torch_neuronx.python_ops.torch_mlir.ops.dynamic_kernels.adapters import cast

        monkeypatch.setattr(cast, "_is_castable", lambda t: True)
        t = torch.empty(*shape, dtype=torch.float32)
        return cast._can_handle_to_copy(t, dtype=torch.bfloat16)

    def test_accepts_m_ge_128(self, monkeypatch):
        assert self._can_handle(monkeypatch, (256, 2048)) is True
        assert self._can_handle(monkeypatch, (128, 512)) is True  # exactly one tile

    def test_declines_sub_128_m(self, monkeypatch):
        assert self._can_handle(monkeypatch, (64, 2048)) is False
        assert self._can_handle(monkeypatch, (127, 2048)) is False  # one row short
        assert self._can_handle(monkeypatch, (100,)) is False  # 1-D: M = numel = 100

    def test_no_dtype_change_declines(self, monkeypatch):
        # A layout-only copy (same dtype) is not this kernel's job, regardless of M.
        from torch_neuronx.python_ops.torch_mlir.ops.dynamic_kernels.adapters import cast

        monkeypatch.setattr(cast, "_is_castable", lambda t: True)
        t = torch.empty(256, 2048, dtype=torch.bfloat16)
        assert cast._can_handle_to_copy(t, dtype=torch.bfloat16) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
