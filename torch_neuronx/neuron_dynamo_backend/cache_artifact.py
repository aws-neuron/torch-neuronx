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
from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import PassMutations
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

    Holds trivially-picklable Executor constructor args plus embedded StableHLO
    bytes.  On warm load the HLO bytes are fed back through ``compile_graph``
    which hits the NEFF persistent cache — no recompilation happens.
    """

    graph_name: str = ""
    cache_key: str = ""
    hlo_cache_key: str = ""  # Deprecated: kept for backward compat with old cache entries
    hlo_bytes: bytes = b""
    io_spec: FunctionIO | None = None
    cast_spec: list[TensorSpec] = dataclasses.field(default_factory=list)
    has_collectives: bool = False
    retain_device: bool = False
    pass_mutations: list[PassMutations] = dataclasses.field(default_factory=list)
    alias_buffer_map: dict[int, int] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        self._executor = None
        self._boxed_call = True

    def _ensure_neff_loaded(self):
        """Re-run compile_graph with cached HLO bytes so the NEFF persistent
        cache is consulted and the NEFF is loaded into the in-memory cache."""
        # Determine the cache key for compile_graph lookup
        key = self.cache_key or self.hlo_cache_key
        if not key:
            return

        # Prefer embedded bytes (self-contained, survives /tmp wipe)
        hlo_bytes = self.hlo_bytes if self.hlo_bytes else None

        # Fall back to HLO cache for backward compat with old cache entries
        if hlo_bytes is None:
            from torch_neuronx.python_ops.compilation.hlo_cache import get as hlo_cache_get

            entry = hlo_cache_get(key)
            if entry is None:
                raise RuntimeError(
                    f"HLO cache miss for key {key}. "
                    "The Dynamo precompile cache references HLO artifacts that no longer exist. "
                    "Clear the Dynamo cache and recompile."
                )
            hlo_bytes = entry.hlo_bytes

        # compile_graph is async but will hit the NEFF persistent cache. For a WLO graph (key ends
        # in "|wlo") re-submit on the WLO path so a persistent cache hit repopulates the
        # wrapped_neff.hlo sidecar (CompilationCache only loads it from the persistent sidecar cache
        # when the kernel RequiresWrappedNeffHlo). Without this, a cross-process warm load would
        # recover an empty sidecar and run the relaid NEFF against untransformed weights.
        torch_neuronx._C.compile_graph(
            key,
            hlo_bytes,
            self.has_collectives,
            wlo=key.endswith("|wlo"),
        )

    def _get_executor(self):
        if self._executor is None:
            self._ensure_neff_loaded()

            # WLO: the wrapped_neff.hlo sidecar was deferred at compile time to keep the submit
            # async. Fetch + parse it now that the NEFF is loaded/compiled so the Executor can
            # relay weights. Non-WLO keys skip this. The "|wlo" suffix is the serialized WLO marker;
            # the sidecar comes from the in-memory entry (cold) or persistent sidecar cache (warm).
            wlo_transforms: list = []
            key = self.cache_key or self.hlo_cache_key
            if key.endswith("|wlo"):
                from torch_neuronx.neuron_dynamo_backend.compile import _parse_wrapped_neff_hlo

                sidecar = torch_neuronx._C.get_wrapped_neff_hlo(key)
                wlo_transforms = _parse_wrapped_neff_hlo(sidecar)
                logger.debug(f"WLO: discovered {len(wlo_transforms)} weight transforms (lazy)")

            from torch_neuronx.neuron_dynamo_backend.executor import Executor

            self._executor = Executor(
                self.graph_name,
                self.cache_key,
                self.io_spec,
                self.cast_spec,
                self.has_collectives,
                retain_device=self.retain_device,
                pass_mutations=self.pass_mutations,
                alias_buffer_map=self.alias_buffer_map,
                wlo_transforms=wlo_transforms,
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

    Records each NEFF under its base cache key; ``put_neff_cache`` /
    ``get_neff_cache`` derive the persistent-cache key internally, so
    callers on both sides only ever deal with the base key.
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
            _persistent_key, neff_bytes = result
            CacheArtifactManager.record_artifact(
                ArtifactType.NEURON_NEFF.value, cache_key, neff_bytes
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

    # Fix precompile guard-state deserialization with no active trace.
    # load_guards_state() unpickles dynamo Sources; ImportSource.__post_init__
    # calls install_guard() which needs an active TracingContext, but cache load
    # runs at torch.compile() time with no trace -> "must be called within an
    # ongoing trace". Wrap the unpickle in a throwaway TracingContext; the guards
    # it installs are discarded (the real set is rebuilt by CheckFunctionManager).
    import torch._dynamo.package as _dynamo_package
    from torch._guards import TracingContext as _TracingContext
    from torch._guards import tracing as _tracing

    _orig_load_guards_state = _dynamo_package.load_guards_state

    def _load_guards_state_safe(guards_state):
        if _TracingContext.try_get() is not None:
            return _orig_load_guards_state(guards_state)
        with _tracing(_TracingContext(fake_mode=None)):
            return _orig_load_guards_state(guards_state)

    _dynamo_package.load_guards_state = _load_guards_state_safe

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
