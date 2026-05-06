"""
MegaCache integration for Neuron compilation caches.

Enables torch.compiler.save_cache_artifacts() / load_cache_artifacts() to include
compiled NEFFs for cross-machine cache sharing, and caching_precompile support to
skip Dynamo tracing on warm starts via AOTAutograd's bundled cache.
"""

import atexit
import dataclasses
import logging
import os
from collections.abc import Sequence
from enum import Enum

import torch
from torch._inductor.output_code import OutputCode
from torch.compiler._cache import CacheArtifact, CacheArtifactFactory, CacheArtifactManager

import torch_neuronx
from torch_neuronx.neuron_dynamo_backend.fx.passes.remove_none_outputs import NoneOutputInfo
from torch_neuronx.neuron_dynamo_backend.utils.alias_info import AliasingInfo
from torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils import FunctionIO, TensorSpec

logger = logging.getLogger(__name__)


class ArtifactType(Enum):
    """MegaCache artifact types for Neuron compilation caches."""

    NEURON_NEFF = "neuron_neff"  # StableHLO → NEFF


@CacheArtifactFactory.register
class NeuronNeffCacheArtifact(CacheArtifact):
    """Cache artifact for Neuron NEFF binaries."""

    @staticmethod
    def type() -> str:
        return ArtifactType.NEURON_NEFF.value

    def populate_cache(self) -> None:
        """Write NEFF to persistent cache during load_cache_artifacts().

        Logs warning on failure (consistent with PyTorch's silent-success API contract).
        """
        if not torch_neuronx._C.put_neff_cache(self.key, self.content):
            logger.warning(f"Failed to write NEFF to persistent cache: {self.key}")


# ---------------------------------------------------------------------------
# NeuronOutputCode: an OutputCode subclass that AOTAutograd's bundled cache
# can serialize.  On warm load, wrap_post_compile() reconstructs the full
# AOTAutograd wrapper chain (arg mapping, subclass dispatch, etc.) around
# this object, so the Executor receives the correct (boxed) AOTAutograd-level
# inputs.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class NeuronOutputCode(OutputCode):
    """Serializable compiled output for the Neuron backend.

    Holds trivially-picklable Executor constructor args plus an ``hlo_cache_key``
    that maps to the StableHLO bytes in the persistent HLO cache.  On warm load
    the HLO bytes are fed back through ``compile_graph`` which hits the NEFF
    persistent cache — no recompilation happens.
    """

    graph_name: str = ""
    cache_key: str = ""
    hlo_cache_key: str = ""
    io_spec: FunctionIO | None = None
    cast_spec: list[TensorSpec] = dataclasses.field(default_factory=list)
    has_collectives: bool = False
    retain_device: bool = False
    mutation_info: AliasingInfo | None = None
    none_output_info: NoneOutputInfo | None = None
    removed_device_mesh_indices: list[int] | None = None
    passthrough_info: AliasingInfo | None = None

    def __post_init__(self):
        self._executor = None
        self._boxed_call = True

    def _ensure_neff_loaded(self):
        """Re-run compile_graph with cached HLO bytes so the NEFF persistent
        cache is consulted and the NEFF is loaded into the in-memory cache."""
        if not self.hlo_cache_key:
            # HLO cache store may have failed
            return
        from torch_neuronx.python_ops.compilation.hlo_cache import get as hlo_cache_get

        entry = hlo_cache_get(self.hlo_cache_key)
        if entry is None:
            raise RuntimeError(
                f"HLO cache miss for key {self.hlo_cache_key}. "
                "The Dynamo precompile cache references HLO artifacts that no longer exist. "
                "Clear the Dynamo cache and recompile."
            )
        hlo_bytes = entry[0]
        # compile_graph is async but will hit the NEFF persistent cache
        torch_neuronx._C.compile_graph(
            self.hlo_cache_key,
            hlo_bytes,
            self.has_collectives,
        )

    def _get_executor(self):
        if self._executor is None:
            self._ensure_neff_loaded()

            from torch_neuronx.neuron_dynamo_backend.executor import Executor

            self._executor = Executor(
                self.graph_name,
                self.cache_key,
                self.io_spec,
                self.cast_spec,
                self.has_collectives,
                retain_device=self.retain_device,
                mutation_info=self.mutation_info,
                none_output_info=self.none_output_info,
                removed_device_mesh_indices=self.removed_device_mesh_indices,
                passthrough_info=self.passthrough_info,
            )
        return self._executor

    def __call__(self, inputs: Sequence) -> tuple[torch.Tensor, ...]:
        return self._get_executor()(*inputs)

    def prepare_for_serialization(self) -> None:
        self._executor = None

    def post_compile(self, example_inputs, constants, graph_kwargs) -> None:
        pass

    def set_triton_bundle(self, triton_bundle) -> None:
        pass


# ---------------------------------------------------------------------------
# NEFF artifact collection for MegaCache
# ---------------------------------------------------------------------------


def _collect_neuron_artifacts() -> None:
    """Record compiled NEFFs as MegaCache artifacts.

    Iterates tracked cache keys and records each NEFF with its persistent key.
    Keys are cleared after collection to prevent unbounded memory growth.
    """
    # Import here to avoid circular dependency: compile.py imports ArtifactType from this module
    from torch_neuronx.neuron_dynamo_backend.compile import (
        clear_compiled_cache_keys,
        get_compiled_cache_keys,
    )

    for cache_key in get_compiled_cache_keys(ArtifactType.NEURON_NEFF):
        result = torch_neuronx._C.get_neff_info(cache_key)
        if result:
            persistent_key, neff_bytes = result
            CacheArtifactManager.record_artifact(
                ArtifactType.NEURON_NEFF.value, persistent_key, neff_bytes
            )
    clear_compiled_cache_keys(ArtifactType.NEURON_NEFF)


# Wrap serialize() to collect Neuron artifacts just-in-time when saving
_original_serialize = CacheArtifactManager.serialize.__func__


def _wrapped_serialize(cls):
    """Sync device and collect Neuron artifacts before serializing."""
    torch.neuron.synchronize()
    _collect_neuron_artifacts()
    return _original_serialize(cls)


CacheArtifactManager.serialize = classmethod(_wrapped_serialize)


def _apply_precompile_patches():
    """Apply monkey-patches and config needed for caching_precompile.

    Called once at import time if caching_precompile is enabled.
    """
    # Enable nonce key fallback so AOTAutograd cache saves regardless of
    # whether the FX graph is hashable (e.g. DTensor fn_with_prim_types).
    torch._dynamo.config.enable_aot_compile = True
    torch._functorch.config.bypass_autograd_cache_key = True  # PT2.11 config

    # Per-rank cache directories — each rank's guard state may contain
    # rank-specific DeviceMesh data, so caches cannot be shared.
    rank = int(os.environ.get("RANK", "0"))
    cache_base = torch._inductor.runtime.cache_dir_utils.cache_dir()
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(cache_base, f"rank_{rank}")

    # Fix dataclass_with_cached_hash.__reduce__ to exclude init=False fields.
    # Without this, DefaultsSource (and other dataclasses with init=False
    # fields) fail to unpickle because __reduce__ includes all fields but
    # __init__ only accepts init=True fields.  Upstream fix pending.
    def _fixed_reduce(self):
        fields = dataclasses.fields(self)
        field_values = tuple(getattr(self, f.name) for f in fields if f.init)
        return (self.__class__, field_values)

    from torch._dynamo.source import DefaultsSource

    DefaultsSource.__reduce__ = _fixed_reduce

    # Save cache artifacts at exit.
    def _save_cache_at_exit():
        try:
            torch.compiler.save_cache_artifacts()
        except Exception:
            logger.debug("Failed to save cache artifacts at exit", exc_info=True)

    atexit.register(_save_cache_at_exit)
