"""Persistent HLO cache.

Caches HLO generation results (FX tracing + StableHLO conversion) to disk,
eliminating re-tracing on process restart.

Storage format: two files per entry via C++ PersistentCacheBackend:
  {key}_hlo  — raw HLO bytes (StableHLO MLIR text)
  {key}_meta — JSON manifest (io_spec, kept_indices, output specs, etc.)

Enable: TORCH_NEURONX_HLO_CACHE=1
Cache dir: TORCH_NEURONX_HLO_CACHE_DIR (default: /tmp/hlo_cache)
"""

import hashlib
import json
import logging
import os

import torch

logger = logging.getLogger(__name__)

_enabled = os.environ.get("TORCH_NEURONX_HLO_CACHE", "1") == "1"
_SCHEMA_VERSION = 1


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


def _serialize_io_spec(io_spec):
    """Serialize FunctionIO to JSON-compatible dict."""
    if io_spec is None:
        return None
    return {
        "inputs": [{"shape": list(s.shape), "dtype": str(s.dtype)} for s in io_spec.inputs],
        "outputs": [{"shape": list(s.shape), "dtype": str(s.dtype)} for s in io_spec.outputs],
    }


def _deserialize_io_spec(d):
    """Deserialize FunctionIO from dict."""
    if d is None:
        return None
    from torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils import FunctionIO, TensorSpec

    inputs = tuple(
        TensorSpec(
            shape=tuple(s["shape"]),
            dtype=getattr(torch, s["dtype"].replace("torch.", "")),
        )
        for s in d["inputs"]
    )
    outputs = tuple(
        TensorSpec(
            shape=tuple(s["shape"]),
            dtype=getattr(torch, s["dtype"].replace("torch.", "")),
        )
        for s in d["outputs"]
    )
    return FunctionIO(inputs=inputs, outputs=outputs)


def _get_backend():
    """Get C++ PersistentCacheBackend bindings."""
    import torch_neuronx._C as _C

    return _C


def get(cache_key: str):
    """Load HLO cache entry from disk. Returns None on miss."""
    if not _enabled:
        return None
    try:
        backend = _get_backend()
        key = cache_key

        # Load manifest
        meta_bytes = backend._get_hlo_cache(key + "_meta")
        if meta_bytes is None:
            return None
        manifest = json.loads(meta_bytes)

        # Validate schema version
        if manifest.get("schema_version", 0) != _SCHEMA_VERSION:
            logger.debug("HLO cache schema mismatch, ignoring entry")
            return None

        # Load HLO bytes
        hlo_bytes = backend._get_hlo_cache(key + "_hlo")
        if hlo_bytes is None:
            return None

        # Integrity check disabled: NKI HLO bytes differ across ranks due to
        # non-determinism in backend_config. The HLO is functionally identical.
        # Re-enable when NKI backend_config non-determinism is resolved.
        # if hashlib.sha256(hlo_bytes).hexdigest() != manifest.get("hlo_sha256"):
        #     logger.warning("HLO cache integrity check failed, ignoring entry")
        #     return None

        return (
            bytes(hlo_bytes),
            _deserialize_io_spec(manifest["io_spec"]),
            manifest["kept_input_indices"],
            manifest["kept_output_indices"],
            _deserialize_meta_tensor(manifest["downcasted_outputs"]),
            _deserialize_meta_tensor(manifest["original_outputs"]),
            {int(k): v for k, v in manifest["passthrough_mappings"].items()},
        )
    except Exception:
        logger.debug("HLO cache read failed", exc_info=True)
        return None


def put(
    cache_key: str,
    hlo_bytes,
    io_spec,
    kept_input_indices,
    kept_output_indices,
    downcasted_outputs,
    original_outputs,
    passthrough_mappings,
):
    """Store HLO cache entry to disk."""
    if not _enabled:
        return
    try:
        backend = _get_backend()
        key = cache_key

        # Skip if already exists
        if backend._exists_hlo_cache(key + "_meta"):
            return

        # Store raw HLO bytes
        hlo_vec = hlo_bytes if isinstance(hlo_bytes, bytes) else bytes(hlo_bytes)
        backend._put_hlo_cache(key + "_hlo", hlo_vec)

        # Store JSON manifest
        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "hlo_sha256": hashlib.sha256(hlo_vec).hexdigest(),
            "io_spec": _serialize_io_spec(io_spec),
            "kept_input_indices": list(kept_input_indices),
            "kept_output_indices": list(kept_output_indices),
            "downcasted_outputs": _serialize_meta_tensor(downcasted_outputs),
            "original_outputs": _serialize_meta_tensor(original_outputs),
            "passthrough_mappings": {str(k): v for k, v in passthrough_mappings.items()}
            if passthrough_mappings
            else {},
        }
        meta_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        backend._put_hlo_cache(key + "_meta", meta_bytes)
    except Exception:
        logger.debug("HLO cache write failed", exc_info=True)
