"""
Compilation phase for torch.compile backend.
Dispatches to compilation worker.
"""

import hashlib
import io
import logging
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
            mlir (Module): StableHLO MLIR module to compile.
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

    def compile(self) -> str:
        """Compile MLIR to NEFF.

        Serializes the MLIR module to bytecode and invokes the
        C++ compilation backend.

        Returns:
            str: Execution handle (cache key) for the compiled graph.
        """
        logger.debug(f"CompileGraph.compile() called: model={self.model_name}")

        # Strip non-deterministic attributes before serializing for C++ backend.
        # This ensures persistent cache keys are consistent across ranks.
        # Note: Artifacts are already saved with debug_dump_path intact by this point.
        attrs = self.mlir.operation.attributes
        for attr_name in _NON_DETERMINISTIC_ATTRS:
            if attr_name in attrs:
                del attrs[attr_name]

        if self.use_tensorizer:
            run_pipeline_with_repro_report(
                self.mlir,
                "builtin.module(stablehlo-legalize-to-linalg)",
                "Lowering StableHLO IR -> Linalg IR",
            )

        bytecode_buffer = io.BytesIO()
        self.mlir.operation.write_bytecode(bytecode_buffer)
        mlir_bytes = bytecode_buffer.getvalue()

        # Use content-based cache key for persistent NEFF caching

        from torch_neuronx.utils import global_version_hash

        content_key = hashlib.sha256(mlir_bytes).hexdigest() + ":" + global_version_hash()

        exec_handle = torch_neuronx._C.compile_graph(
            content_key,
            mlir_bytes,
            self.has_collectives,
            use_tensorizer=self.use_tensorizer,
        )

        # Track for MegaCache integration
        _add_compiled_cache_key(ArtifactType.NEURON_NEFF, exec_handle)

        logger.debug(f"Compilation complete: execution_cache_key={exec_handle}")
        return exec_handle
