"""Persistent HLO cache.

Caches HLO generation results (FX tracing + StableHLO conversion) to disk,
eliminating re-tracing on process restart. One ``get`` / ``put`` pair serves two
callers, selected by ``compile_mode``:

  compile_mode=False  eager ops, keyed by the C++ dispatch key.
                      Enable: TORCH_NEURONX_HLO_CACHE=1 (default on)
  compile_mode=True   torch.compile, keyed by the FX graph
                      (see compute_fx_cache_key).
                      Enable: TORCH_NEURONX_HLO_COMPILE_CACHE=1 (default off)

The mode selects the gate and which manifest fields are written; storage and spec
serialization are identical. Both modes share one blob layout:
  {key}_v{schema}_hlo / {key}_v{schema}_meta   payload bytes + JSON manifest

Cache dir: TORCH_NEURONX_HLO_CACHE_DIR (default: /tmp/hlo_cache).
Both modes are also disabled by TORCH_NEURONX_NEFF_DISABLE_CACHE.
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import torch

from torch_neuronx.kernels.cache_utils import is_caching_disabled

logger = logging.getLogger(__name__)

_enabled = os.environ.get("TORCH_NEURONX_HLO_CACHE", "1") == "1"
_compile_enabled = os.environ.get("TORCH_NEURONX_HLO_COMPILE_CACHE", "0") == "1"
_SCHEMA_VERSION = 1
_FX_KEY_VERSION = 1
CACHE_DIR = os.environ.get("TORCH_NEURONX_HLO_CACHE_DIR", "/tmp/hlo_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _serialize_meta_tensor(t):
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        return {"shape": list(t.shape), "dtype": str(t.dtype)}
    if isinstance(t, tuple | list):
        return [_serialize_meta_tensor(x) for x in t]
    return t


def _deserialize_meta_tensor(d):
    if d is None:
        return None
    if isinstance(d, dict) and "shape" in d and "dtype" in d:
        dtype = getattr(torch, d["dtype"].replace("torch.", ""))
        return torch.empty(d["shape"], dtype=dtype, device="meta")
    if isinstance(d, list):
        return tuple(_deserialize_meta_tensor(x) for x in d)
    return d


def _serialize_specs(specs) -> list:
    """Serialize a TensorSpec sequence (io_spec inputs/outputs, cast_spec)."""
    return [{"shape": list(s.shape), "dtype": str(s.dtype)} for s in specs or ()]


def _deserialize_specs(data) -> tuple:
    """Inverse of _serialize_specs. str(torch.float32) is "torch.float32", so the
    prefix is stripped to get the attribute name."""
    from torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils import TensorSpec

    return tuple(
        TensorSpec(shape=tuple(s["shape"]), dtype=getattr(torch, s["dtype"].replace("torch.", "")))
        for s in data or ()
    )


def _serialize_io_spec(io_spec):
    """Serialize FunctionIO to JSON-compatible dict."""
    if io_spec is None:
        return None
    return {
        "inputs": _serialize_specs(io_spec.inputs),
        "outputs": _serialize_specs(io_spec.outputs),
    }


def _deserialize_io_spec(d):
    """Deserialize FunctionIO from dict."""
    if d is None:
        return None
    from torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils import FunctionIO

    return FunctionIO(
        inputs=_deserialize_specs(d["inputs"]),
        outputs=_deserialize_specs(d["outputs"]),
    )


def _restore_int_keys(value: Any) -> Any:
    """Restore integer dict keys that JSON stringified, recursively.

    `transform_params` is an opaque payload for TRANSFORM_REGISTRY, and its
    `index_to_input` entry is indexed by an int output index, so a string key
    would miss on lookup.
    """
    if isinstance(value, dict):
        return {
            (int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k): _restore_int_keys(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_restore_int_keys(v) for v in value]
    return value


def _serialize_io_mutation(m) -> dict:
    """Serialize one IOMutation to a JSON-compatible dict."""
    return {
        "kind": m.kind.value,
        "indices": list(m.indices or ()),
        "transform_key": m.transform_key,
        "transform_params": m.transform_params,
        "pass_name": m.pass_name,
        "alias_input_indices": (
            list(m.alias_input_indices) if m.alias_input_indices is not None else None
        ),
        "is_mutation": m.is_mutation,
    }


def _serialize_pass_mutations(pass_mutations) -> list:
    """Serialize the ordered per-pass IO mutations.

    List order is significant: each pass's indices are in the IO space before that
    pass ran, so the entries must come back in the order they were emitted.
    """
    return [
        {
            "pass_name": pm.pass_name,
            "input_mutations": [_serialize_io_mutation(m) for m in pm.input_mutations or ()],
            "output_mutations": [_serialize_io_mutation(m) for m in pm.output_mutations or ()],
        }
        for pm in pass_mutations or ()
    ]


def _deserialize_pass_mutations(data) -> list:
    """Inverse of _serialize_pass_mutations.

    Returns early on empty so an eager read, which never carries mutations, does
    not import the dynamo backend.
    """
    if not data:
        return []

    from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import (
        IOMutation,
        MutationKind,
        PassMutations,
    )

    def _mutation(d) -> IOMutation:
        return IOMutation(
            kind=MutationKind(d["kind"]),
            indices=list(d.get("indices") or []),
            transform_key=d.get("transform_key"),
            transform_params=_restore_int_keys(d.get("transform_params")),
            pass_name=d.get("pass_name", ""),
            alias_input_indices=d.get("alias_input_indices"),
            is_mutation=d.get("is_mutation"),
        )

    return [
        PassMutations(
            pass_name=d.get("pass_name", ""),
            input_mutations=[_mutation(m) for m in d.get("input_mutations") or ()],
            output_mutations=[_mutation(m) for m in d.get("output_mutations") or ()],
        )
        for d in data or ()
    ]


def _get_backend():
    """Get C++ PersistentCacheBackend bindings."""
    import torch_neuronx._C as _C

    return _C


def _blob_key(cache_key: str, suffix: str) -> str:
    """Storage key for one blob of an entry.

    The schema version is in the key because ``_put_hlo_cache`` cannot replace an
    existing key, so an entry at an old schema has to be missed rather than
    declined: otherwise nothing could ever overwrite it.
    """
    return f"{cache_key}_v{_SCHEMA_VERSION}_{suffix}"


@dataclass
class CachedHLO:
    """One cache entry: the serialized HLO plus the metadata to resume compilation.

    Shared by both entry points. ``hlo_bytes`` is the serialized StableHLO exactly
    as that mode's compiler consumes it (text for eager, bytecode for compile),
    handed back verbatim, so a reader needs no MLIR bindings.

    The remaining fields are the union of what each path needs; a path leaves the
    ones it does not use at their defaults. Eager populates the output-signature
    fields (``kept_*_indices``, ``downcasted_outputs``, ``original_outputs``,
    ``passthrough_mappings``) used to rebuild the Python calling convention.
    torch.compile populates ``cast_spec``, ``has_collectives``,
    ``alias_buffer_map`` and ``pass_mutations``; ``alias_buffer_map`` is also baked
    into ``hlo_bytes`` as ``mhlo.input_output_alias`` but is surfaced as structured
    data so a third-party runtime need not parse the IR to recover it.
    """

    hlo_bytes: bytes
    io_spec: object = None
    # Eager: restoring the original output signature.
    kept_input_indices: list = field(default_factory=list)
    kept_output_indices: list = field(default_factory=list)
    downcasted_outputs: object = None
    original_outputs: object = None
    passthrough_mappings: dict = field(default_factory=dict)
    # torch.compile: resuming at the NEFF stage.
    cast_spec: tuple = ()
    has_collectives: bool = False
    alias_buffer_map: dict[int, int] = field(default_factory=dict)
    # Ordered per-pass IO mutations; restore the original call signature, so a
    # resumed lowering must carry them or its outputs differ from a fresh one.
    pass_mutations: list = field(default_factory=list)


def cache_enabled(*, compile_mode: bool) -> bool:
    """Whether the cache is on for this mode.

    Both modes honour the shared TORCH_NEURONX_NEFF_DISABLE_CACHE kill switch,
    then their own flag: TORCH_NEURONX_HLO_CACHE (eager, on by default) or
    TORCH_NEURONX_HLO_COMPILE_CACHE (compile, off by default). Both flags are
    snapshotted at import; a test toggles the module-level `_enabled` /
    `_compile_enabled` directly.
    """
    if is_caching_disabled():
        return False
    return _compile_enabled if compile_mode else _enabled


def get(cache_key: str, *, compile_mode: bool = False) -> CachedHLO | None:
    """Load a cache entry from disk. Returns None on miss or any error.

    Populates whichever CachedHLO fields the stored manifest carries, so an eager
    entry comes back with the output-signature metadata and a compile entry with
    the NEFF-resume metadata. ``hlo_bytes`` is returned verbatim; no MLIR bindings
    are needed to read an entry.
    """
    if not cache_enabled(compile_mode=compile_mode):
        return None
    try:
        backend = _get_backend()

        meta_bytes = backend._get_hlo_cache(_blob_key(cache_key, "meta"))
        if meta_bytes is None:
            return None
        manifest = json.loads(meta_bytes)

        hlo_bytes = backend._get_hlo_cache(_blob_key(cache_key, "hlo"))
        if hlo_bytes is None:
            return None
        hlo_bytes = bytes(hlo_bytes)

        if hashlib.sha256(hlo_bytes).hexdigest() != manifest.get("hlo_sha256"):
            logger.warning("HLO cache integrity check failed, ignoring entry")
            return None

        return CachedHLO(
            hlo_bytes=hlo_bytes,
            io_spec=_deserialize_io_spec(manifest.get("io_spec")),
            kept_input_indices=manifest.get("kept_input_indices", []),
            kept_output_indices=manifest.get("kept_output_indices", []),
            downcasted_outputs=_deserialize_meta_tensor(manifest.get("downcasted_outputs")),
            original_outputs=_deserialize_meta_tensor(manifest.get("original_outputs")),
            passthrough_mappings={
                int(k): v for k, v in manifest.get("passthrough_mappings", {}).items()
            },
            cast_spec=_deserialize_specs(manifest.get("cast_spec")),
            has_collectives=manifest.get("has_collectives", False),
            alias_buffer_map={int(k): int(v) for k, v in manifest.get("alias", {}).items()},
            pass_mutations=_deserialize_pass_mutations(manifest.get("pass_mutations")),
        )
    except Exception:
        logger.debug("HLO cache read failed", exc_info=True)
        return None


def put(cache_key: str, entry: CachedHLO, *, compile_mode: bool = False) -> None:
    """Store a cache entry to disk. First-writer-wins.

    Serializes only the fields the given mode populates, so an eager entry and a
    compile entry produce different manifests under the same schema. Failures are
    swallowed: a cache write must not fail a lowering that already succeeded.
    """
    if not cache_enabled(compile_mode=compile_mode):
        return
    try:
        backend = _get_backend()
        if backend._exists_hlo_cache(_blob_key(cache_key, "meta")):
            return

        hlo_bytes = (
            entry.hlo_bytes if isinstance(entry.hlo_bytes, bytes) else bytes(entry.hlo_bytes)
        )
        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "hlo_sha256": hashlib.sha256(hlo_bytes).hexdigest(),
            "io_spec": _serialize_io_spec(entry.io_spec),
        }
        if compile_mode:
            manifest.update(
                {
                    "cast_spec": _serialize_specs(entry.cast_spec),
                    "has_collectives": entry.has_collectives,
                    "alias": {str(k): int(v) for k, v in (entry.alias_buffer_map or {}).items()},
                    "pass_mutations": _serialize_pass_mutations(entry.pass_mutations),
                }
            )
        else:
            manifest.update(
                {
                    "kept_input_indices": list(entry.kept_input_indices),
                    "kept_output_indices": list(entry.kept_output_indices),
                    "downcasted_outputs": _serialize_meta_tensor(entry.downcasted_outputs),
                    "original_outputs": _serialize_meta_tensor(entry.original_outputs),
                    "passthrough_mappings": {
                        str(k): v for k, v in (entry.passthrough_mappings or {}).items()
                    },
                }
            )

        # Serialize before storing anything: a manifest that will not serialize must
        # not leave a payload blob behind
        meta_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        backend._put_hlo_cache(_blob_key(cache_key, "hlo"), hlo_bytes)
        backend._put_hlo_cache(_blob_key(cache_key, "meta"), meta_bytes)
    except Exception:
        logger.debug("HLO cache write failed", exc_info=True)


# ---------------------------------------------------------------------------
# torch.compile path: StableHLO keyed by the FX graph
# ---------------------------------------------------------------------------


def _graph_hash(gm: torch.fx.GraphModule) -> str:
    """Canonical hash of the FX graph and its lifted constants, via torch's
    ``compiled_fx_graph_hash``.

    It reads shapes from ``example_inputs``, which the lowering passes as ``[]``, so
    feed each placeholder's fake tensor instead -- otherwise two shapes collide.
    """
    example_inputs = [
        node.meta["val"]
        for node in gm.graph.nodes
        if node.op == "placeholder"
        and node.meta.get("val") is not None
        and hasattr(node.meta["val"], "shape")
    ]
    from torch._inductor.codecache import compiled_fx_graph_hash

    key, _ = compiled_fx_graph_hash(gm, example_inputs, {}, [])
    return key


def compute_fx_cache_key(
    gm: torch.fx.GraphModule,
    *,
    run_fx_passes: bool = True,
    alias_fingerprint: str = "",
    use_tensorizer: bool = False,
) -> str:
    """Base cache key for the StableHLO that lowering ``gm`` would produce.

    Covers everything that changes the lowered StableHLO:

    * the FX graph and its lifted constants, via ``_graph_hash`` (torch's
      ``compiled_fx_graph_hash``),
    * ``run_fx_passes``,
    * the platform target and logical NC count. The FX passes read these while
      lowering -- flex-attention bakes ``grid: [lnc]`` into the graph -- so the
      StableHLO is target-specific even though the C++ CompilationCacheKey also
      folds them into the downstream NEFF key,
    * ``alias_fingerprint``, the caller's aliasing contribution: it becomes
      ``mhlo.input_output_alias`` on the module, and with ``run_fx_passes=False``
      it is the only source of aliasing,
    * ``use_tensorizer``, which legalizes StableHLO to Linalg and so changes the
      lowered bytes,
    * the source/package version hash.

    Keying on the StableHLO output instead would require the very lowering this
    cache skips.
    """
    from torch_neuronx.utils import (
        get_logical_neuron_cores,
        get_platform_target,
        global_version_hash,
    )

    hasher = hashlib.sha256()

    def _field(data: bytes) -> None:
        # Length-prefixed so arbitrary bytes in a field (e.g. alias_fingerprint)
        # cannot shift across a boundary and collide with a different field split.
        hasher.update(len(data).to_bytes(8, "big"))
        hasher.update(data)

    _field(f"v{_FX_KEY_VERSION}".encode())
    _field(_graph_hash(gm).encode())
    _field(b"fx_passes=1" if run_fx_passes else b"fx_passes=0")
    _field(f"{get_platform_target()}/{get_logical_neuron_cores()}".encode())
    _field(alias_fingerprint.encode())
    _field(b"tensorizer=1" if use_tensorizer else b"tensorizer=0")
    _field(global_version_hash().encode())
    return hasher.hexdigest()
