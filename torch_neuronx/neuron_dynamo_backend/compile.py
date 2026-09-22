"""
Compilation phase for torch.compile backend.
Dispatches to compilation worker.
"""

import hashlib
import io
import json
import logging
import os
import threading
from typing import Final

from torch_mlir.compiler_utils import run_pipeline_with_repro_report

import torch_neuronx
from torch_neuronx.neuron_dynamo_backend.cache_artifact import ArtifactType

logger = logging.getLogger(__name__)

# Attributes to strip from StableHLO before caching (non-deterministic across ranks/runs)
_NON_DETERMINISTIC_ATTRS = frozenset({"torch.debug_dump_path"})

# Track cache keys by artifact type for MegaCache integration.
# Cleared after save_cache_artifacts() to prevent unbounded growth.
_compiled_cache_keys: dict[ArtifactType, set[str]] = {t: set() for t in ArtifactType}
_cache_keys_lock: Final = threading.Lock()


def get_compiled_cache_keys(artifact_type: ArtifactType) -> frozenset[str]:
    """Return cache keys tracked for a specific MegaCache artifact type.

    Args:
        artifact_type: The artifact type to get keys for (e.g., ArtifactType.NEURON_NEFF).

    Returns:
        Immutable snapshot of cache keys. Thread-safe: returns a copy to avoid
        races with concurrent modifications.
    """
    with _cache_keys_lock:
        return frozenset(_compiled_cache_keys[artifact_type])


def clear_compiled_cache_keys(artifact_type: ArtifactType | None = None) -> None:
    """Clear tracked cache keys.

    Args:
        artifact_type: Specific type to clear, or None to clear all types.
    """
    with _cache_keys_lock:
        if artifact_type is None:
            for keys in _compiled_cache_keys.values():
                keys.clear()
        else:
            _compiled_cache_keys[artifact_type].clear()


def _add_compiled_cache_key(artifact_type: ArtifactType, key: str) -> None:
    """Add a cache key to the tracking set.

    Args:
        artifact_type: The artifact type category.
        key: The cache key to track.
    """
    with _cache_keys_lock:
        _compiled_cache_keys[artifact_type].add(key)


def _serialize_mlir_module(mlir, *, use_tensorizer: bool = False) -> bytes:
    """Serialize MLIR exactly as it will be submitted to the compiler."""
    attrs = mlir.operation.attributes
    for attr_name in _NON_DETERMINISTIC_ATTRS:
        if attr_name in attrs:
            del attrs[attr_name]

    if use_tensorizer:
        run_pipeline_with_repro_report(
            mlir,
            "builtin.module(stablehlo-legalize-to-linalg)",
            "Lowering StableHLO IR -> Linalg IR",
        )

    bytecode_buffer = io.BytesIO()
    mlir.operation.write_bytecode(bytecode_buffer)
    return bytecode_buffer.getvalue()


def _content_key(mlir_bytes: bytes) -> str:
    """Compute the NEFF content key for serialized MLIR.

    Keep the gate identical to the C++ utils::IsHloContentKeyEnabled(). The two must
    agree, or the base key and the persistent key disagree about the version salt.
    """
    if os.environ.get("TORCH_NEURONX_NEFF_KEY_FROM_HLO", "1") != "0":
        return hashlib.sha256(mlir_bytes).hexdigest()

    from torch_neuronx.utils import global_version_hash

    return hashlib.sha256(mlir_bytes).hexdigest() + ":" + global_version_hash()


def _submit_mlir_bytes(
    mlir_bytes: bytes,
    *,
    has_collectives: bool,
    use_tensorizer: bool = False,
) -> str:
    """Submit serialized MLIR and track its NEFF cache key."""
    exec_handle = torch_neuronx._C.compile_graph(
        _content_key(mlir_bytes),
        mlir_bytes,
        has_collectives,
        use_tensorizer=use_tensorizer,
    )
    _add_compiled_cache_key(ArtifactType.NEURON_NEFF, exec_handle)
    return exec_handle


def _submit_mlir_bytes_wlo(mlir_bytes: bytes, *, has_collectives: bool) -> str:
    """Submit serialized MLIR via the async WLO path and return its execution handle.

    The base key carries a "|wlo" suffix: the C++ CompilationCacheKey infers wlo_enabled from it
    in one place, so the persistent key stays consistent across every construction site (compile,
    NeffDirectKernelExecution, put/get_neff_cache).

    The wrapped_neff.hlo sidecar is deliberately NOT fetched here. get_wrapped_neff_hlo blocks until
    the async compile finishes, which would serialize WLO compiles and defeat the async submit.
    Instead the sidecar is fetched and parsed lazily when the executor is first built
    (cache_artifact.NeuronOutputCode._get_executor), letting the compile overlap with the rest of
    the torch.compile pipeline. The "|wlo" suffix on the returned handle is the marker that path
    keys off.
    """
    exec_handle = torch_neuronx._C.compile_graph(
        _content_key(mlir_bytes) + "|wlo",
        mlir_bytes,
        has_collectives,
        wlo=True,
    )
    _add_compiled_cache_key(ArtifactType.NEURON_NEFF, exec_handle)
    return exec_handle


def _compile_stablehlo_bytes(
    stablehlo: bytes,
    *,
    has_collectives: bool,
    use_tensorizer: bool = False,
) -> str:
    """Compile canonical StableHLO bytes without in-process lowering state."""
    if use_tensorizer:
        raise ValueError(
            "Detached StableHLO artifacts do not support use_tensorizer=True; "
            "the tensorizer path requires the in-process MLIR module."
        )
    return _submit_mlir_bytes(stablehlo, has_collectives=has_collectives)


class CompileGraph:
    """
    Takes MLIR, compiles it to NEFF, and returns cache key.
    Supports both the default compilation path and the tensorizer path.
    """

    def __init__(
        self,
        mlir,
        model_name: str,
        segment_id: str,
        graph_name: str,
        has_collectives: bool = False,
        use_tensorizer: bool = False,
    ):
        """Initialize compilation context.

        Args:
            mlir (Module | bytes): StableHLO MLIR module to compile, or already
                serialized MLIR. Bytes are taken as canonical (as produced by
                `serialize()`) and submitted unchanged; the tensorizer path needs
                the in-process module and rejects them.
            model_name (str): Model name for logging.
            segment_id (str): Segment ID for statistics tracking.
            has_collectives (bool): Whether graph contains collective operations.
            use_tensorizer (bool): Whether to use the tensorizer compilation path.
            graph_name (str): Deterministic graph name for cache key.
        """
        self.mlir = mlir
        self.model_name = model_name
        self.segment_id = segment_id
        self.has_collectives = has_collectives
        self.use_tensorizer = use_tensorizer
        self.graph_name = graph_name

    def serialize(self) -> bytes:
        """Return the exact bytes `compile()` would submit to the compiler.

        Lets a caller persist the serialized MLIR (e.g. into the HLO cache) and
        later hand those bytes back to `CompileGraph`, producing the same NEFF key
        as compiling the live module would have.
        """
        if isinstance(self.mlir, bytes | bytearray):
            return bytes(self.mlir)
        return _serialize_mlir_module(self.mlir, use_tensorizer=self.use_tensorizer)

    def compile(self) -> str:
        """Compile MLIR to NEFF.

        Serializes the MLIR module to bytecode and invokes the
        C++ compilation backend.

        Returns:
            str: Execution handle (cache key) for the compiled graph.
        """
        logger.debug(f"CompileGraph.compile() called: model={self.model_name}")

        if isinstance(self.mlir, bytes | bytearray) and self.use_tensorizer:
            raise ValueError(
                "Serialized MLIR does not support use_tensorizer=True; the "
                "tensorizer path requires the in-process MLIR module."
            )

        has_wlo = not isinstance(self.mlir, bytes | bytearray) and (
            "mhlo.frontend_attributes" in self.mlir.operation.attributes
        )
        mlir_bytes = self.serialize()

        if not has_wlo and b"transposable_weight_idx" in mlir_bytes:
            raise RuntimeError(
                "WLO annotation (transposable_weight_idx) found in serialized MLIR that did not "
                "take the WLO path — likely resumed from a cache that must not carry "
                "WLO-annotated modules"
            )

        if not has_wlo:
            return _submit_mlir_bytes(
                mlir_bytes,
                has_collectives=self.has_collectives,
                use_tensorizer=self.use_tensorizer,
            )

        # WLO: submit async and return the handle. The wrapped_neff.hlo sidecar (and the weight
        # transforms parsed from it) is fetched lazily at executor-build time, not here — see
        # _submit_mlir_bytes_wlo. The "|wlo" handle carries the marker the executor keys off.
        exec_handle = _submit_mlir_bytes_wlo(mlir_bytes, has_collectives=self.has_collectives)
        logger.debug(
            f"WLO: async submit complete, sidecar deferred; execution_cache_key={exec_handle}"
        )
        return exec_handle


def _parse_wrapped_neff_hlo(hlo_bytes: bytes) -> list[dict]:
    """Parse wrapped_neff.hlo protobuf to extract weight transforms.

    Runs in a subprocess to avoid protobuf descriptor conflicts.
    Returns list of dicts: {input_index, reshape, transpose}
    """
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    if not hlo_bytes:
        return []

    with tempfile.NamedTemporaryFile(suffix=".hlo", delete=False) as f:
        f.write(hlo_bytes)
        hlo_path = f.name

    try:
        parser_module = Path(__file__).parent / "wlo_hlo_parser.py"
        try:
            result = subprocess.run(
                [sys.executable, str(parser_module), hlo_path],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError("WLO: HLO parsing timed out after 120s") from e
    finally:
        Path(hlo_path).unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(f"WLO: HLO parsing failed: {result.stderr[-300:]}")

    output = result.stdout.strip()
    if not output:
        return []
    return json.loads(output)
