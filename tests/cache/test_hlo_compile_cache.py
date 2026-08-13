"""Tests for the compile-path StableHLO cache (keyed by the FX graph).

Covers three layers:
  * the cache key (stability, discrimination, normalization),
  * get / put in compile mode over the C++ PersistentCacheBackend, including
    integrity and key isolation from the eager entry points,
  * the backend hook `_lower_fx_to_stablehlo_cached` (flag gating, store on miss,
    resume on hit).

The MLIR round-trip tests need the compiled torch_mlir bindings but NOT a Neuron
device; they build a tiny StableHLO module from text.
"""

import pytest
import torch
from torch._dynamo.backends.common import aot_autograd

from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import (
    IOMutation,
    MutationKind,
    PassMutations,
)
from torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils import (
    FunctionIO,
    TensorSpec,
)
from torch_neuronx.python_ops.compilation import hlo_cache

# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Point the on-disk cache at a fresh per-test directory and enable the flag.

    The cache persists at the default /tmp/hlo_cache across processes, so without
    isolation an entry from one test (or an earlier CI run on a reused runner)
    could turn a later miss into a spurious hit.
    """
    monkeypatch.setenv("TORCH_NEURONX_HLO_CACHE_DIR", str(tmp_path))
    # The flag is snapshotted at import, so patch the module var, not the env.
    monkeypatch.setattr(hlo_cache, "_compile_enabled", True)
    monkeypatch.delenv("TORCH_NEURONX_NEFF_DISABLE_CACHE", raising=False)
    yield


def _capture_post_aot_gm(model: torch.nn.Module, *inputs) -> torch.fx.GraphModule:
    """Trace a model on CPU and return its post-AOT forward FX GraphModule.

    Mirrors the graph the lowering receives: node.meta['val'] is populated with
    fake-tensor shape/dtype. No Neuron device required.
    """
    captured = {}

    def fw_compiler(gm, example_inputs):
        captured["gm"] = gm
        return gm.forward

    torch._dynamo.reset()
    compiled = torch.compile(model, backend=aot_autograd(fw_compiler=fw_compiler))
    with torch.no_grad():
        compiled(*inputs)
    return captured["gm"]


class _Tiny(torch.nn.Module):
    def __init__(self, act=torch.relu):
        super().__init__()
        self.lin = torch.nn.Linear(8, 4)
        self._act = act

    def forward(self, x):
        return self._act(self.lin(x)) + 1.0


_STABLEHLO_TEXT = """
module {
  func.func @main(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
    return %arg0 : tensor<4x8xf32>
  }
}
"""


def _make_stablehlo_module():
    """Build a live StableHLO Module from text in its own fresh context.

    Requires the compiled torch_mlir bindings (skips the test otherwise).
    """
    mlir_ir = pytest.importorskip("torch_mlir.ir")
    return mlir_ir.Module.parse(_STABLEHLO_TEXT, context=mlir_ir.Context())


def _io_spec():
    return FunctionIO(
        inputs=(TensorSpec(shape=(4, 8), dtype=torch.float32),),
        outputs=(TensorSpec(shape=(4, 8), dtype=torch.float32),),
    )


def _cast_spec():
    return (TensorSpec(shape=(4, 8), dtype=torch.float32),)


def _put(cache_key, module, io_spec, cast_spec, **kwargs):
    """Serialize `module` to MLIR bytecode and store it as a compile-mode entry.

    Mirrors what the backend hook stores via `_serialize_mlir_module`.
    """
    import io as _io

    buf = _io.BytesIO()
    module.operation.write_bytecode(buf)
    hlo_cache.put(
        cache_key,
        hlo_cache.CachedHLO(
            hlo_bytes=buf.getvalue(), io_spec=io_spec, cast_spec=cast_spec, **kwargs
        ),
        compile_mode=True,
    )


def _bytecode(module) -> bytes:
    import io as _io

    buf = _io.BytesIO()
    module.operation.write_bytecode(buf)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Cache key
# --------------------------------------------------------------------------- #


class TestCacheKey:
    def test_stable_across_identical_graphs(self):
        model = _Tiny()
        k1 = hlo_cache.compute_fx_cache_key(_capture_post_aot_gm(model, torch.randn(2, 8)))
        k2 = hlo_cache.compute_fx_cache_key(_capture_post_aot_gm(model, torch.randn(2, 8)))
        assert k1 == k2

    def test_stable_across_weights(self):
        # Weights are lifted to graph inputs, not baked into the graph text, so
        # two models that differ only in weight values must share a key.
        k1 = hlo_cache.compute_fx_cache_key(_capture_post_aot_gm(_Tiny(), torch.randn(2, 8)))
        k2 = hlo_cache.compute_fx_cache_key(_capture_post_aot_gm(_Tiny(), torch.randn(2, 8)))
        assert k1 == k2

    def test_discriminates_input_shape(self):
        model = _Tiny()
        k_small = hlo_cache.compute_fx_cache_key(_capture_post_aot_gm(model, torch.randn(2, 8)))
        k_big = hlo_cache.compute_fx_cache_key(_capture_post_aot_gm(model, torch.randn(5, 8)))
        assert k_small != k_big

    def test_discriminates_op(self):
        k_relu = hlo_cache.compute_fx_cache_key(
            _capture_post_aot_gm(_Tiny(torch.relu), torch.randn(2, 8))
        )
        k_sig = hlo_cache.compute_fx_cache_key(
            _capture_post_aot_gm(_Tiny(torch.sigmoid), torch.randn(2, 8))
        )
        assert k_relu != k_sig

    def test_discriminates_run_fx_passes(self):
        gm = _capture_post_aot_gm(_Tiny(), torch.randn(2, 8))
        k_on = hlo_cache.compute_fx_cache_key(gm, run_fx_passes=True)
        k_off = hlo_cache.compute_fx_cache_key(gm, run_fx_passes=False)
        assert k_on != k_off

    def test_discriminates_platform_target(self, monkeypatch):
        """The FX passes read the target while lowering (flex-attention bakes
        `grid: [lnc]` into the graph), so the StableHLO is target-specific."""
        gm = _capture_post_aot_gm(_Tiny(), torch.randn(2, 8))
        monkeypatch.setattr("torch_neuronx.utils.get_platform_target", lambda: "trn2")
        k_trn2 = hlo_cache.compute_fx_cache_key(gm)
        monkeypatch.setattr("torch_neuronx.utils.get_platform_target", lambda: "trn1")
        k_trn1 = hlo_cache.compute_fx_cache_key(gm)
        assert k_trn2 != k_trn1

    def test_discriminates_logical_neuron_cores(self, monkeypatch):
        gm = _capture_post_aot_gm(_Tiny(), torch.randn(2, 8))
        monkeypatch.setattr("torch_neuronx.utils.get_logical_neuron_cores", lambda: 1)
        k_lnc1 = hlo_cache.compute_fx_cache_key(gm)
        monkeypatch.setattr("torch_neuronx.utils.get_logical_neuron_cores", lambda: 2)
        k_lnc2 = hlo_cache.compute_fx_cache_key(gm)
        assert k_lnc1 != k_lnc2

    def test_fields_cannot_bleed_across_boundaries(self):
        """Field contents must not be able to shift across a field boundary.

        The graph dump and alias_fingerprint carry arbitrary bytes, so a delimiter
        scheme would let content that contains the delimiter produce the same digest
        as a different split of the same stream. Length prefixing prevents that.
        """
        gm = _capture_post_aot_gm(_Tiny(), torch.randn(2, 8))
        # "a" then "" vs "" then "a" is the same concatenated stream, differently
        # split across two adjacent fields.
        k1 = hlo_cache.compute_fx_cache_key(gm, alias_fingerprint="a")
        k2 = hlo_cache.compute_fx_cache_key(gm, alias_fingerprint="")
        assert k1 != k2
        # An embedded NUL must not be interchangeable with a field boundary.
        k3 = hlo_cache.compute_fx_cache_key(gm, alias_fingerprint="x\x00y")
        k4 = hlo_cache.compute_fx_cache_key(gm, alias_fingerprint="x")
        assert k3 != k4

    def test_discriminates_alias_fingerprint(self):
        """Caller-supplied aliasing becomes mhlo.input_output_alias on the module,
        so it must split the key."""
        gm = _capture_post_aot_gm(_Tiny(), torch.randn(2, 8))
        k_none = hlo_cache.compute_fx_cache_key(gm, alias_fingerprint="")
        k_alias = hlo_cache.compute_fx_cache_key(gm, alias_fingerprint="aliasing=[(0, 0)]")
        assert k_none != k_alias

    def test_discriminates_caller_supplied_pass_mutations(self):
        """With run_fx_passes=False a caller's `pass_mutations` is the only source of
        aliasing: it reaches mhlo.input_output_alias via get_alias_output_buffer_map,
        so two different requests must not share a key."""
        from torch_neuronx.neuron_dynamo_backend.backend import _alias_fingerprint

        def _mutations(alias_pairs):
            if alias_pairs is None:
                return []
            return [
                PassMutations(
                    pass_name="caller",
                    output_mutations=[
                        IOMutation(
                            kind=MutationKind.ALIAS,
                            indices=[out for out, _ in alias_pairs],
                            alias_input_indices=[inp for _, inp in alias_pairs],
                            is_mutation=True,
                            pass_name="caller",
                        )
                    ],
                )
            ]

        gm = _capture_post_aot_gm(_Tiny(), torch.randn(2, 8))

        def key(alias_pairs):
            return hlo_cache.compute_fx_cache_key(
                gm,
                run_fx_passes=False,
                alias_fingerprint=_alias_fingerprint({"pass_mutations": _mutations(alias_pairs)}),
            )

        k_empty = key([])
        k_alias = key([(0, 0)])
        k_other = key([(1, 0)])
        assert k_empty != k_alias, "an alias must not share a key with no alias"
        assert k_alias != k_other, "different alias pairs must not share a key"

        # is_mutation distinguishes copy-back from pure buffer reuse; both reach the
        # alias map, so it must split the key too.
        reuse = _mutations([(0, 0)])
        reuse[0].output_mutations[0].is_mutation = False
        k_reuse = hlo_cache.compute_fx_cache_key(
            gm, run_fx_passes=False, alias_fingerprint=_alias_fingerprint({"pass_mutations": reuse})
        )
        assert k_alias != k_reuse

    def test_discriminates_lifted_constant_values(self):
        """The graph hash covers a lifted constant's value, so two graphs that differ
        only in that value must not share a key."""

        class _WithConst(torch.nn.Module):
            def __init__(self, fill):
                super().__init__()
                self.register_buffer("c", torch.full((4,), fill))

            def forward(self, x):
                return x + self.c

        gm_a = _capture_post_aot_gm(_WithConst(1.0), torch.randn(4))
        gm_b = _capture_post_aot_gm(_WithConst(2.0), torch.randn(4))
        # Only meaningful if the constant really was lifted into the graph as a
        # get_attr; a parameter promoted to an input is already covered by shape.
        has_const = any(n.op == "get_attr" for n in gm_a.graph.nodes)
        if not has_const:
            pytest.skip("constant not lifted as get_attr in this torch version")
        assert hlo_cache.compute_fx_cache_key(gm_a) != hlo_cache.compute_fx_cache_key(gm_b)

    def test_keys_bfloat16_lifted_constants(self):
        """A lifted bf16 constant must still discriminate by value. numpy has no
        bfloat16, but the graph hash handles it natively (no numpy conversion)."""

        class _Mul(torch.nn.Module):
            def __init__(self, fill):
                super().__init__()
                self._fill = fill

            def forward(self, x):
                return x * torch.tensor([self._fill, self._fill], dtype=torch.bfloat16)

        gm_a = _capture_post_aot_gm(_Mul(1.5), torch.randn(2, dtype=torch.bfloat16))
        gm_b = _capture_post_aot_gm(_Mul(2.5), torch.randn(2, dtype=torch.bfloat16))
        lifted = [getattr(gm_a, str(n.target)) for n in gm_a.graph.nodes if n.op == "get_attr"]
        if not any(t.dtype is torch.bfloat16 for t in lifted if isinstance(t, torch.Tensor)):
            pytest.skip("bf16 constant not lifted as get_attr in this torch version")
        assert hlo_cache.compute_fx_cache_key(gm_a) != hlo_cache.compute_fx_cache_key(gm_b)

    def test_discriminates_use_tensorizer(self):
        """use_tensorizer legalizes StableHLO to Linalg, changing the lowered bytes, so
        it must split the key."""
        gm = _capture_post_aot_gm(_Tiny(), torch.randn(2, 8))
        k_off = hlo_cache.compute_fx_cache_key(gm, use_tensorizer=False)
        k_on = hlo_cache.compute_fx_cache_key(gm, use_tensorizer=True)
        assert k_off != k_on


# --------------------------------------------------------------------------- #
# Flag gating
# --------------------------------------------------------------------------- #


class TestFlagGating:
    def test_enabled_by_flag(self):
        assert hlo_cache.cache_enabled(compile_mode=True) is True

    def test_off_by_default(self, monkeypatch):
        monkeypatch.setattr(hlo_cache, "_compile_enabled", False)
        assert hlo_cache.cache_enabled(compile_mode=True) is False

    def test_neff_disable_overrides_opt_in(self, monkeypatch):
        # The shared kill switch wins even with the compile flag explicitly on.
        monkeypatch.setenv("TORCH_NEURONX_NEFF_DISABLE_CACHE", "1")
        assert hlo_cache.cache_enabled(compile_mode=True) is False

    def test_disabled_get_and_put_are_noops(self, monkeypatch):
        module = _make_stablehlo_module()
        monkeypatch.setenv("TORCH_NEURONX_NEFF_DISABLE_CACHE", "1")
        _put("k_off", module, _io_spec(), _cast_spec())
        assert hlo_cache.get("k_off", compile_mode=True) is None
        # Re-enabling must not reveal a smuggled entry: nothing was written.
        monkeypatch.delenv("TORCH_NEURONX_NEFF_DISABLE_CACHE")
        assert hlo_cache.get("k_off", compile_mode=True) is None


# --------------------------------------------------------------------------- #
# get / put (compile mode)
# --------------------------------------------------------------------------- #


class TestCompileDiskCache:
    def test_get_miss(self):
        assert hlo_cache.get("no_such_key", compile_mode=True) is None

    def test_put_get_round_trip(self):
        module = _make_stablehlo_module()
        io_spec, cast_spec = _io_spec(), _cast_spec()
        _put("k_rt", module, io_spec, cast_spec, has_collectives=True, alias_buffer_map={0: 0})

        entry = hlo_cache.get("k_rt", compile_mode=True)
        assert entry is not None
        assert entry.io_spec == io_spec
        assert tuple(entry.cast_spec) == tuple(cast_spec)
        assert entry.has_collectives is True
        assert entry.alias_buffer_map == {0: 0}
        # Bytes are returned verbatim, ready to submit without a parse.
        assert entry.hlo_bytes == _bytecode(module)

    def test_alias_map_keys_come_back_as_ints(self):
        # JSON stringifies object keys; the reader must restore int keys.
        module = _make_stablehlo_module()
        _put("k_alias", module, _io_spec(), _cast_spec(), alias_buffer_map={0: 2, 1: 3})
        entry = hlo_cache.get("k_alias", compile_mode=True)
        assert entry.alias_buffer_map == {0: 2, 1: 3}
        assert all(isinstance(k, int) for k in entry.alias_buffer_map)

    def test_pass_mutations_round_trip(self):
        """pass_mutations restore the original call signature, so a resumed lowering
        must reconstruct them exactly -- an empty list silently drops re-inserted
        None outputs and script-object pass-throughs."""
        mutations = [
            PassMutations(
                pass_name="remove_none_outputs",
                input_mutations=[
                    IOMutation(
                        kind=MutationKind.ADD,
                        indices=[1],
                        transform_key="shard_index",
                        transform_params={"shard_index": 3},
                        pass_name="remove_none_outputs",
                    )
                ],
                output_mutations=[
                    IOMutation(
                        kind=MutationKind.REMOVE,
                        indices=[0, 2],
                        transform_key="none",
                        pass_name="remove_none_outputs",
                    ),
                    IOMutation(
                        kind=MutationKind.ALIAS,
                        indices=[1],
                        alias_input_indices=[4],
                        is_mutation=True,
                        pass_name="remove_none_outputs",
                    ),
                ],
            ),
            PassMutations(pass_name="second_pass"),
        ]
        module = _make_stablehlo_module()
        _put("k_pm", module, _io_spec(), _cast_spec(), pass_mutations=mutations)

        entry = hlo_cache.get("k_pm", compile_mode=True)
        assert entry.pass_mutations == mutations, "round trip must be exact"
        # Enum identity, not the serialized string.
        assert entry.pass_mutations[0].output_mutations[0].kind is MutationKind.REMOVE

    def test_transform_params_int_keys_come_back_as_ints(self):
        """`index_to_input` is looked up by an int output index, so the JSON string
        keys must be restored or the pass-through transform misses."""
        module = _make_stablehlo_module()
        mutations = [
            PassMutations(
                pass_name="replace_device_mesh_placeholders",
                output_mutations=[
                    IOMutation(
                        kind=MutationKind.REMOVE,
                        indices=[0, 1],
                        transform_key="input_passthrough",
                        transform_params={"index_to_input": {0: 2, 1: 3}},
                        pass_name="replace_device_mesh_placeholders",
                    )
                ],
            )
        ]
        _put("k_pm_keys", module, _io_spec(), _cast_spec(), pass_mutations=mutations)

        params = (
            hlo_cache.get("k_pm_keys", compile_mode=True)
            .pass_mutations[0]
            .output_mutations[0]
            .transform_params
        )
        assert params == {"index_to_input": {0: 2, 1: 3}}
        assert all(isinstance(k, int) for k in params["index_to_input"])
        # The outer key stays a string; only int-like keys are converted.
        assert set(params) == {"index_to_input"}

    def test_payload_is_submittable_bytes(self):
        """The reader is binding-free: raw serialized bytes, not a live module, so
        a hit can be submitted to the compiler without parsing."""
        module = _make_stablehlo_module()
        _put("k_bytes", module, _io_spec(), _cast_spec())
        entry = hlo_cache.get("k_bytes", compile_mode=True)
        assert isinstance(entry.hlo_bytes, bytes)
        # MLIR bytecode, not text.
        assert entry.hlo_bytes.startswith(b"ML\xefR"), entry.hlo_bytes[:8]

    def test_put_is_first_writer_wins(self):
        module = _make_stablehlo_module()
        _put("k_once", module, _io_spec(), _cast_spec(), has_collectives=False)
        # A second put under the same key must not replace the stored metadata.
        _put("k_once", module, _io_spec(), _cast_spec(), has_collectives=True)
        assert hlo_cache.get("k_once", compile_mode=True).has_collectives is False

    def test_schema_mismatch_is_miss(self, monkeypatch):
        module = _make_stablehlo_module()
        _put("k_schema", module, _io_spec(), _cast_spec())
        monkeypatch.setattr(hlo_cache, "_SCHEMA_VERSION", 999)
        assert hlo_cache.get("k_schema", compile_mode=True) is None

    def test_unserializable_manifest_stores_nothing(self):
        """A manifest that will not serialize must leave no payload behind.

        transform_params is an opaque per-pass payload handed straight to
        json.dumps. Storing the payload first would strand it: _put_hlo_cache
        cannot replace a key, so the next attempt would pair these bytes with a
        later manifest and the digest check would reject the entry forever.
        """
        module = _make_stablehlo_module()
        mutations = [
            PassMutations(
                pass_name="p",
                input_mutations=[],
                output_mutations=[
                    IOMutation(
                        kind=MutationKind.ADD,
                        indices=[0],
                        transform_key="t",
                        transform_params={"not_json": object()},
                        pass_name="p",
                    )
                ],
            )
        ]
        # put swallows the failure, so the observable contract is what is on disk.
        _put("k_unser", module, _io_spec(), _cast_spec(), pass_mutations=mutations)

        backend = hlo_cache._get_backend()
        assert backend._get_hlo_cache(hlo_cache._blob_key("k_unser", "hlo")) is None
        assert backend._get_hlo_cache(hlo_cache._blob_key("k_unser", "meta")) is None

    def test_schema_change_is_storable_not_permanently_stale(self, monkeypatch):
        """A miss caused by a schema change must be re-storable under the same key.

        `_put_hlo_cache` cannot replace a key, so if the old blob occupied the new
        schema's key every later run would miss and then fail to store.
        """
        module = _make_stablehlo_module()
        _put("k_evolve", module, _io_spec(), _cast_spec(), has_collectives=False)

        monkeypatch.setattr(hlo_cache, "_SCHEMA_VERSION", 999)
        assert hlo_cache.get("k_evolve", compile_mode=True) is None

        _put("k_evolve", module, _io_spec(), _cast_spec(), has_collectives=True)
        entry = hlo_cache.get("k_evolve", compile_mode=True)
        assert entry is not None
        assert entry.has_collectives is True

    def test_tampered_payload_is_rejected(self, monkeypatch):
        """A payload that does not match the manifest digest is a miss.

        The digest is written alongside the bytes it describes, so it compares
        stored against stored: a mismatch means the blob was corrupted or torn,
        never that another rank produced different HLO.
        """
        module = _make_stablehlo_module()
        _put("k_tamper", module, _io_spec(), _cast_spec())

        real_get = hlo_cache._get_backend()._get_hlo_cache

        class _Tampering:
            def _get_hlo_cache(self, key):
                # Corrupt only the payload; leave the manifest readable, so the
                # read reaches the digest comparison rather than dying in
                # json.loads.
                if key.endswith("_meta"):
                    return real_get(key)
                return b"tampered-not-the-original-bytecode"

        monkeypatch.setattr(hlo_cache, "_get_backend", lambda: _Tampering())
        assert hlo_cache.get("k_tamper", compile_mode=True) is None

    def test_untampered_payload_passes_the_digest_check(self):
        """The digest a real put writes matches the bytes it stored.

        Guards against the check being satisfied only by the tampering fixture:
        if put and get disagreed on what is hashed, every read would be a miss.
        """
        module = _make_stablehlo_module()
        _put("k_digest_ok", module, _io_spec(), _cast_spec())
        assert hlo_cache.get("k_digest_ok", compile_mode=True) is not None

    def test_mode_selects_manifest_fields_not_key_space(self):
        """Both modes share the {key}_hlo blob layout; only the manifest differs.

        Collisions are avoided by the two callers deriving keys differently (C++
        dispatch key vs compute_fx_cache_key), not by the storage layer. So a read
        in the other mode does find the entry; it just sees that mode's fields at
        their defaults. Pinned here because it is the contract a shared blob layout
        implies, and a future suffix split would silently change it.
        """
        module = _make_stablehlo_module()
        _put("shared_base", module, _io_spec(), _cast_spec(), alias_buffer_map={0: 0})

        compile_entry = hlo_cache.get("shared_base", compile_mode=True)
        assert compile_entry is not None
        assert compile_entry.alias_buffer_map == {0: 0}

        # Same blob, read as eager: the compile-only fields are absent from the
        # manifest, so they come back defaulted rather than raising.
        eager_entry = hlo_cache.get("shared_base")
        assert eager_entry is not None
        assert eager_entry.hlo_bytes == compile_entry.hlo_bytes
        assert eager_entry.kept_input_indices == []
        assert eager_entry.passthrough_mappings == {}


# --------------------------------------------------------------------------- #
# Backend hook
# --------------------------------------------------------------------------- #


class _FakeJob:
    timestamp = "seg_from_job"


class TestBackendHook:
    """The hook wrapping `_lower_fx_to_stablehlo`: gated by the flag, stores on a
    miss, and resumes from the entry on a hit."""

    @pytest.fixture
    def gm(self):
        return _capture_post_aot_gm(_Tiny(), torch.randn(2, 8))

    @staticmethod
    def _lowering(module):
        from torch_neuronx.neuron_dynamo_backend.backend import _Lowering

        return _Lowering(
            gm=None,
            stablehlo_mlir=module,
            io_spec=_io_spec(),
            cast_spec=_cast_spec(),
            cache_key="",
            graph_name="stored_name",
            segment_id="stored_seg",
            has_collectives=False,
            alias_buffer_map={0: 0},
        )

    def _call_hook(self, monkeypatch, gm, *, uncached_returns=None, use_tensorizer=False):
        """Invoke the hook with the uncached front half stubbed out, and report
        whether it was reached."""
        from torch_neuronx.neuron_dynamo_backend import backend as backend_mod

        calls = []

        def fake_uncached(gm_, example_inputs, job, **kwargs):
            calls.append(kwargs)
            return uncached_returns

        monkeypatch.setattr(backend_mod, "_lower_fx_to_stablehlo", fake_uncached)
        monkeypatch.setattr(backend_mod, "_make_graph_name", lambda name: f"{name}_fresh")
        out = backend_mod._lower_fx_to_stablehlo_cached(
            gm,
            [],
            _FakeJob(),
            model_name="m",
            run_fx_passes=True,
            use_tensorizer=use_tensorizer,
        )
        return out, calls

    def test_disabled_delegates_without_touching_cache(self, monkeypatch, gm):
        monkeypatch.setattr(hlo_cache, "_compile_enabled", False)
        sentinel = object()
        out, calls = self._call_hook(monkeypatch, gm, uncached_returns=sentinel)
        assert out is sentinel
        assert len(calls) == 1
        key = hlo_cache.compute_fx_cache_key(gm, run_fx_passes=True)
        monkeypatch.setattr(hlo_cache, "_compile_enabled", True)
        assert hlo_cache.get(key, compile_mode=True) is None

    def test_miss_lowers_and_stores(self, monkeypatch, gm):
        module = _make_stablehlo_module()
        analysis = self._lowering(module)

        out, calls = self._call_hook(monkeypatch, gm, uncached_returns=analysis)
        assert out is analysis
        assert len(calls) == 1, "miss must run the real lowering"

        key = hlo_cache.compute_fx_cache_key(gm, run_fx_passes=True)
        assert hlo_cache.get(key, compile_mode=True) is not None

    def test_hit_skips_lowering_and_resumes(self, monkeypatch, gm):
        module = _make_stablehlo_module()
        key = hlo_cache.compute_fx_cache_key(gm, run_fx_passes=True)
        mutations = [
            PassMutations(
                pass_name="p",
                output_mutations=[
                    IOMutation(kind=MutationKind.REMOVE, indices=[1], transform_key="none")
                ],
            )
        ]
        _put(
            key,
            module,
            _io_spec(),
            _cast_spec(),
            has_collectives=False,
            alias_buffer_map={0: 0},
            pass_mutations=mutations,
        )

        out, calls = self._call_hook(monkeypatch, gm)
        assert calls == [], "hit must not run FX->StableHLO"

        # Resume state comes from the entry...
        assert out.io_spec == _io_spec()
        assert out.alias_buffer_map == {0: 0}
        # pass_mutations must survive the hit: they rebuild the original output
        # signature, so resuming without them changes what the graph returns.
        assert out.pass_mutations == mutations
        # stablehlo_mlir carries the stored bytes, not a live module.
        assert out.stablehlo_mlir == _bytecode(module)
        assert out.cache_key == ""  # NEFF stage has not run yet
        assert out.gm is None
        # ...but per-run identifiers are this job's, not the stored ones.
        assert out.graph_name == "m_fresh"
        assert out.segment_id == "seg_from_job"

    def test_neff_stage_submits_cached_bytes_without_reserializing(self, monkeypatch):
        """CompileGraph passes byte MLIR straight through: a resumed entry submits
        the stored bytes rather than being parsed and re-serialized."""
        from torch_neuronx.neuron_dynamo_backend import backend as backend_mod
        from torch_neuronx.neuron_dynamo_backend import compile as compile_mod

        submitted = {}

        def fake_submit(mlir_bytes, *, has_collectives, use_tensorizer=False):
            submitted.update(bytes=mlir_bytes, has_collectives=has_collectives)
            return "fake_cache_key"

        def exploding_serialize(*a, **k):
            raise AssertionError("stored bytes must not be re-serialized")

        monkeypatch.setattr(compile_mod, "_submit_mlir_bytes", fake_submit)
        monkeypatch.setattr(compile_mod, "_serialize_mlir_module", exploding_serialize)

        module = _make_stablehlo_module()
        low = backend_mod._Lowering(
            gm=None,
            stablehlo_mlir=_bytecode(module),
            io_spec=_io_spec(),
            cast_spec=_cast_spec(),
            cache_key="",
            graph_name="m_0",
            segment_id="seg",
            has_collectives=True,
        )
        out = backend_mod._lower_stablehlo_to_neff(low, model_name="m")
        assert out.cache_key == "fake_cache_key"
        assert submitted["bytes"] == _bytecode(module)
        assert submitted["has_collectives"] is True

    def test_tensorizer_rejects_serialized_mlir(self):
        """The tensorizer pipeline needs the live module, so byte MLIR is refused
        rather than silently compiled without the Linalg legalization."""
        from torch_neuronx.neuron_dynamo_backend import backend as backend_mod

        low = backend_mod._Lowering(
            gm=None,
            stablehlo_mlir=_bytecode(_make_stablehlo_module()),
            io_spec=_io_spec(),
            cast_spec=_cast_spec(),
            cache_key="",
            graph_name="m_0",
            segment_id="seg",
            has_collectives=False,
        )
        with pytest.raises(ValueError, match="tensorizer"):
            backend_mod._lower_stablehlo_to_neff(
                low, model_name="m", options={"use_tensorizer_backend": True}
            )

    def test_tensorizer_is_not_cached(self, monkeypatch, gm):
        """The tensorizer needs the live module, so this path never stores: serializing
        with it legalizes to Linalg in place, and the bytes an entry would carry are
        refused by the NEFF stage."""
        module = _make_stablehlo_module()
        analysis = self._lowering(module)

        out, calls = self._call_hook(
            monkeypatch, gm, uncached_returns=analysis, use_tensorizer=True
        )
        assert out is analysis
        assert len(calls) == 1, "tensorizer must run the real lowering"

        key = hlo_cache.compute_fx_cache_key(gm, run_fx_passes=True)
        assert hlo_cache.get(key, compile_mode=True) is None, "tensorizer must not store"

    def test_tensorizer_does_not_resume_a_stored_entry(self, monkeypatch, gm):
        """An entry written by the default path must not be resumed by the tensorizer:
        the key no longer distinguishes them, so the check precedes the cache read."""
        module = _make_stablehlo_module()
        key = hlo_cache.compute_fx_cache_key(gm, run_fx_passes=True)
        hlo_cache.put(
            key,
            hlo_cache.CachedHLO(
                hlo_bytes=_bytecode(module),
                io_spec=_io_spec(),
                cast_spec=_cast_spec(),
                has_collectives=False,
            ),
            compile_mode=True,
        )
        assert hlo_cache.get(key, compile_mode=True) is not None, "entry must be on disk"

        analysis = self._lowering(module)
        out, calls = self._call_hook(
            monkeypatch, gm, uncached_returns=analysis, use_tensorizer=True
        )
        assert out is analysis, "must lower fresh, not resume the entry"
        assert len(calls) == 1

    def test_key_failure_degrades_to_fresh_lowering(self, monkeypatch, gm):
        def boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(hlo_cache, "compute_fx_cache_key", boom)
        sentinel = object()
        out, calls = self._call_hook(monkeypatch, gm, uncached_returns=sentinel)
        assert out is sentinel
        assert len(calls) == 1

    def test_concurrent_misses_lower_once(self, tmp_path, monkeypatch, gm):
        """The per-key file lock serializes co-launched ranks: exactly one runs
        FX->StableHLO, the rest wait and resume from the entry it stored."""
        import threading

        from torch_neuronx.neuron_dynamo_backend import backend as backend_mod

        # The lock path uses hlo_cache.CACHE_DIR, captured at import; the autouse
        # fixture only repoints the env var, so steer the lock file here too.
        monkeypatch.setattr(hlo_cache, "CACHE_DIR", str(tmp_path))

        module = _make_stablehlo_module()
        lowered = []
        start = threading.Barrier(3)

        def slow_lower(gm_, example_inputs, job, **kwargs):
            lowered.append(1)
            # Hold long enough that the others are already blocked on the lock.
            import time

            time.sleep(0.3)
            return self._lowering(module)

        monkeypatch.setattr(backend_mod, "_lower_fx_to_stablehlo", slow_lower)
        monkeypatch.setattr(backend_mod, "_make_graph_name", lambda name: f"{name}_fresh")

        results = []

        def worker():
            start.wait()
            results.append(
                backend_mod._lower_fx_to_stablehlo_cached(
                    gm, [], _FakeJob(), model_name="m", run_fx_passes=True
                )
            )

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(lowered) == 1, "only one rank may run FX->StableHLO under the lock"
        assert len(results) == 3
        # The winner returns its fresh analysis (a live module); the two losers resume
        # from the stored bytes. So exactly one carries the live module, two the bytes.
        resumed = [r for r in results if isinstance(r.stablehlo_mlir, bytes)]
        assert len(resumed) == 2, "the two losers must resume from the stored entry"
        assert all(r.stablehlo_mlir == _bytecode(module) for r in resumed)

    def test_entry_stored_while_waiting_is_resumed_not_relowered(self, tmp_path, monkeypatch, gm):
        """If another rank stores the entry between our first `get` and acquiring the
        lock, the re-check inside the lock resumes it instead of lowering again."""
        from torch_neuronx.neuron_dynamo_backend import backend as backend_mod

        monkeypatch.setattr(hlo_cache, "CACHE_DIR", str(tmp_path))
        module = _make_stablehlo_module()
        key = hlo_cache.compute_fx_cache_key(gm, run_fx_passes=True)

        real_get = hlo_cache.get
        calls = {"n": 0}

        def get_then_store(k, *, compile_mode=False):
            # First probe misses; simulate a peer writing the entry before the lock.
            calls["n"] += 1
            if calls["n"] == 1:
                return None
            return real_get(k, compile_mode=compile_mode)

        def should_not_lower(*a, **k):
            raise AssertionError("must resume the peer's entry, not lower")

        # Peer writes the entry (as if it won the race).
        _put(key, module, _io_spec(), _cast_spec(), has_collectives=False, alias_buffer_map={0: 0})
        monkeypatch.setattr(hlo_cache, "get", get_then_store)
        monkeypatch.setattr(backend_mod, "_lower_fx_to_stablehlo", should_not_lower)
        monkeypatch.setattr(backend_mod, "_make_graph_name", lambda name: f"{name}_fresh")

        out = backend_mod._lower_fx_to_stablehlo_cached(
            gm, [], _FakeJob(), model_name="m", run_fx_passes=True
        )
        assert out.stablehlo_mlir == _bytecode(module)
        assert calls["n"] == 2, "must re-probe inside the lock after the initial miss"
