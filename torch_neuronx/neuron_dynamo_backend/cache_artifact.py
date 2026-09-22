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

# A CPython type accepts an attribute assignment only if it is a heap type.
_PY_TPFLAGS_HEAPTYPE = 1 << 9


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

        ``self.key`` is the persistent cache key that _collect_neuron_artifacts
        recorded. Logs a warning on failure, which keeps PyTorch's silent-success API
        contract.
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

    Records each NEFF under its persistent cache key, which ``get_neff_info``
    returns. The persistent key is content-addressed on the lowered IR, so a base
    cache key cannot derive it. Keys are cleared after collection to prevent
    unbounded memory growth.
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
            if not persistent_key:
                logger.warning(f"No persistent cache key for {cache_key}. Skipped the NEFF.")
                continue
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


# ---------------------------------------------------------------------------
# Per-rank precompile package directory
# ---------------------------------------------------------------------------


def _resolve_package_element() -> str | None:
    """Return a directory element unique to this process, or None.

    Each rank saves rank-specific guard state, so ranks must not share a package
    directory. Prefer NEURON_RT_VISIBLE_CORES: it is unique per host process
    even with data parallelism, and a launcher repeats it across a cold and a
    warm run. Fall back to the torch.distributed rank, then RANK.

    NEURON_RT_VISIBLE_CORES must be disjoint per rank. A launcher that gives each
    rank the same core range makes each rank return the same element. The ranks
    then share one directory with no warning, which is the collision this patch
    prevents.
    """
    cores = os.environ.get("NEURON_RT_VISIBLE_CORES")
    if cores:
        return "cores_" + cores.replace(" ", "").replace(",", "_").replace("-", "_")

    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return f"rank_{dist.get_rank()}"

    env_rank = os.environ.get("RANK")
    if env_rank is not None and env_rank.isdigit():
        return f"rank_{env_rank}"
    return None


def _install_per_rank_package_dir() -> None:
    """Patch DiskDynamoCache.path_prefix so each rank gets its own package dir.

    The rank is unknown at import time (vLLM passes it as an argument, so RANK
    reads 0 for every rank), but path_prefix runs at each save and load. Patch
    path_prefix rather than TORCHINDUCTOR_CACHE_DIR: the env var compounds to
    .../rank_0/rank_0 across the process generations a vLLM server starts. A
    shared directory breaks a warm restore, because a rank loads another rank's
    package whose guard state names an absent process group.
    """
    from torch._dynamo.package import DiskDynamoCache

    if getattr(DiskDynamoCache.path_prefix, "_neuron_patched", False):
        return
    original_path_prefix = DiskDynamoCache.path_prefix
    warned = False

    def path_prefix(self) -> str:
        nonlocal warned
        base = original_path_prefix(self)
        element = _resolve_package_element()
        if element is None:
            if not warned:
                warned = True
                logger.warning(
                    "Kept the shared package directory %s. This process has no "
                    "known rank, so each rank reads the same torch.compile "
                    "package. A warm restore can stop with an error on a process "
                    "group name. Set NEURON_RT_VISIBLE_CORES or RANK, or "
                    "initialize torch.distributed before the first compile.",
                    base,
                )
            return base
        return os.path.join(base, element)

    path_prefix._neuron_patched = True
    DiskDynamoCache.path_prefix = path_prefix
    logger.debug(
        "Patched DiskDynamoCache.path_prefix. Each rank now has its own "
        "torch.compile package directory."
    )


# ---------------------------------------------------------------------------
# Guard state that a distributed runtime owns
# ---------------------------------------------------------------------------


class _UnresolvedObject:
    """A stand-in for an object the guard state cannot hold.

    Each guard that reads it fails to match, so Dynamo recompiles. That is the
    correct result for a package that does not match this process.
    """

    def __init__(self, label: str) -> None:
        self._label = label

    def __getattr__(self, name: str) -> "_UnresolvedObject":
        return _UnresolvedObject(f"{self._label}.{name}")

    def __repr__(self) -> str:
        return f"_UnresolvedObject({self._label!r})"


def _rebuild_unresolved(label: str) -> "_UnresolvedObject":
    """Return a stand-in for ``label``. The unpickle step calls this function."""
    return _UnresolvedObject(label)


def _rebuild_process_group(group_name: str):
    """Return the live process group named group_name, or a stand-in.

    Keep this at module level: pickle saves it by qualified name. Return a
    stand-in instead of raising, because the unpickle runs inside
    CompilePackage.install where no caller catches an exception; the stand-in
    makes the guard fail and Dynamo recompiles.
    """
    try:
        from torch.distributed.distributed_c10d import _resolve_process_group

        return _resolve_process_group(group_name)
    except Exception:
        logger.warning(
            "Replaced the process group %r with a stand-in. This process has no "
            "group with this name, so the torch.compile package belongs to "
            "another rank or to an older run. Dynamo compiles the graph again. "
            "Check that each rank has its own package directory.",
            group_name,
        )
        return _UnresolvedObject(f"process_group_{group_name}")


def _install_process_group_reducer() -> None:
    """Save a ProcessGroup by its name during guard pickling.

    A guard reads the group name through the group, so the group is on the
    guard's path. Dynamo lists ProcessGroup in _get_unsupported_types, so the
    pickler saves the _Missing sentinel and the warm restore fails on
    _Missing.group_name. Give ProcessGroup a __reduce__ that saves the name, and
    drop it from _get_unsupported_types so reducer_override does not shadow
    __reduce__.
    """
    import torch._dynamo.guards as dynamo_guards

    try:
        from torch._C._distributed_c10d import ProcessGroup
    except ImportError:
        return

    # A static extension type refuses a __reduce__ assignment, so gate on the flag.
    if "__reduce__" not in ProcessGroup.__dict__:
        if not ProcessGroup.__flags__ & _PY_TPFLAGS_HEAPTYPE:
            logger.warning(
                "Skipped the process group reducer. This torch build defines "
                "ProcessGroup as a static type that refuses a __reduce__ "
                "assignment. A warm restore recompiles instead of a package "
                "match. Check the torch version."
            )
            return
        ProcessGroup.__reduce__ = lambda self: (
            _rebuild_process_group,
            (self.group_name,),
        )

    original_unsupported_types = getattr(dynamo_guards, "_get_unsupported_types", None)
    if original_unsupported_types is None:
        logger.warning(
            "Skipped the process group reducer. This torch version has no "
            "_get_unsupported_types in torch._dynamo.guards. A warm restore can "
            "stop with an AttributeError on _Missing. Check the torch version."
        )
        return

    if getattr(original_unsupported_types, "_neuron_patched", False):
        return

    def drop_process_group():
        # Call the original each time, so a later addition to the list survives.
        return tuple(t for t in original_unsupported_types() if t is not ProcessGroup)

    drop_process_group._neuron_patched = True
    dynamo_guards._get_unsupported_types = drop_process_group
    logger.debug("Patched _get_unsupported_types to drop ProcessGroup.")


def _transport_types() -> tuple[type, ...]:
    """Return the types that hold a transport and refuse to pickle.

    A traced model can reach one through a collective helper (a vLLM
    GroupCoordinator owns a zmq.Socket via its message queue). Import each
    optional module in a try block, so the tuple holds installed types only.
    """
    import mmap
    import socket
    import threading

    types_found: list[type] = [
        socket.socket,
        mmap.mmap,
        type(threading.Lock()),
        type(threading.RLock()),
    ]

    try:
        import zmq

        types_found += [zmq.Socket, zmq.Poller, zmq.Context]
    except ImportError:
        pass  # zmq is optional; skip its types when it is absent.

    try:
        from multiprocessing.shared_memory import SharedMemory

        types_found.append(SharedMemory)
    except ImportError:
        pass  # SharedMemory is unavailable on some runtimes; skip it.

    return tuple(types_found)


def _install_transport_reducer() -> None:
    """Replace a transport object with a stand-in during guard pickling.

    The guard state holds every object the graph can reach; a socket or lock is
    on that path and refuses to pickle, so the whole package save fails. A
    stand-in makes any guard that reads it fail to match, and Dynamo recompiles.
    """
    from torch._dynamo.guards import GuardsStatePickler

    transport_types = _transport_types()
    if not transport_types:
        return

    original_reducer_override = getattr(GuardsStatePickler, "reducer_override", None)
    if original_reducer_override is None:
        logger.warning(
            "Skipped the transport reducer. This torch version has no "
            "GuardsStatePickler.reducer_override. A package save can stop on an "
            "unpicklable transport object. Check the torch version."
        )
        return
    if getattr(original_reducer_override, "_neuron_patched", False):
        return

    def reducer_override(self, obj):
        if isinstance(obj, transport_types):
            label = f"{type(obj).__module__}.{type(obj).__name__}"
            return _rebuild_unresolved, (label,)
        return original_reducer_override(self, obj)

    reducer_override._neuron_patched = True
    GuardsStatePickler.reducer_override = reducer_override
    logger.debug(
        "Patched GuardsStatePickler.reducer_override. The pickler replaces %d "
        "transport types with a stand-in.",
        len(transport_types),
    )


# ---------------------------------------------------------------------------
# A name that a package restore installs, and a fresh compile then needs
# ---------------------------------------------------------------------------


def _install_builtins_dict_share() -> None:
    """Reuse the builtins-dict name that a package restore installed.

    Dynamo puts builtins.__dict__ into the module globals under a mangled name
    (e.g. __builtins_dict___0). CompilePackage.install writes that name with no
    cleanup, so a later compile in the same module regenerates the same name and
    CleanupHook.create fails on the existing name. Return the existing name
    instead; one module has one builtins dict.
    """
    from torch._dynamo.output_graph import OutputGraph, get_builtins_dict

    original = getattr(OutputGraph, "install_builtins_dict_in_fglobals", None)
    if original is None:
        logger.warning(
            "Skipped the builtins dict patch. This torch version has no "
            "OutputGraph.install_builtins_dict_in_fglobals. A warm restore can "
            "stop with an AssertionError on a name that already exists in the "
            "scope. Check the torch version."
        )
        return
    if getattr(original, "_neuron_patched", False):
        return

    prefix = "__builtins_dict__"

    def install_builtins_dict_in_fglobals(self) -> str:
        f_builtins = get_builtins_dict(self.global_scope)
        for name, value in list(self.global_scope.items()):
            if value is f_builtins and name.startswith(prefix):
                return name
        return original(self)

    install_builtins_dict_in_fglobals._neuron_patched = True
    OutputGraph.install_builtins_dict_in_fglobals = install_builtins_dict_in_fglobals
    logger.debug(
        "Patched OutputGraph.install_builtins_dict_in_fglobals. A fresh compile "
        "now shares the builtins dict entry of a package restore."
    )


def _apply_precompile_patches():
    """Apply monkey-patches and config needed for caching_precompile.

    Called once at import time if caching_precompile is enabled.
    """
    # Enable nonce key fallback so AOTAutograd cache saves regardless of
    # whether the FX graph is hashable (e.g. DTensor fn_with_prim_types).
    torch._dynamo.config.enable_aot_compile = True
    torch._functorch.config.bypass_autograd_cache_key = True  # PT2.11 config

    # Give each rank its own package directory. Rank-specific guard state must
    # not be shared, and the rank is unknown at import time.
    _install_per_rank_package_dir()

    # Upstream torch has no guard-pickle reducer for a process group or a
    # transport object, so the package save or restore fails without these.
    _install_process_group_reducer()
    _install_transport_reducer()

    # A package restore installs a builtins-dict name with no cleanup; let a
    # later compile in the same module reuse it.
    _install_builtins_dict_share()

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
