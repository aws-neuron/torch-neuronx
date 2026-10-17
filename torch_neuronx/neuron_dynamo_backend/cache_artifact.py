"""
MegaCache integration for Neuron compilation caches.

Enables torch.compiler.save_cache_artifacts() / load_cache_artifacts() to include
compiled NEFFs for cross-machine cache sharing, and caching_precompile support to
skip Dynamo tracing on warm starts via AOTAutograd's bundled cache.
"""

import atexit
import copyreg
import dataclasses
import importlib
import inspect
import logging
import os
import sys
import types
from collections.abc import Sequence
from contextvars import ContextVar
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

# Each GuardsStatePickler.reducer_override patch marks itself with one of these.
_REDUCER_FLAGS = ("_neuron_patched", "_neuron_function_patched", "_neuron_mesh_patched")


def _install_reducer_override(reducer_override, original, flag: str) -> None:
    """Install a GuardsStatePickler.reducer_override wrapper.

    Carry the flags of the wrapper it replaces, so a later install of an earlier
    patch sees its flag on the outermost wrapper and does not wrap again.
    """
    from torch._dynamo.guards import GuardsStatePickler

    for name in _REDUCER_FLAGS:
        setattr(reducer_override, name, getattr(original, name, False))
    setattr(reducer_override, flag, True)
    GuardsStatePickler.reducer_override = reducer_override


class ArtifactType(Enum):
    """Enumerate MegaCache artifact types for Neuron compilation caches."""

    NEURON_NEFF = "neuron_neff"  # StableHLO → NEFF


@CacheArtifactFactory.register
class NeuronNeffCacheArtifact(CacheArtifact):
    """Represent a Neuron NEFF binary stored through MegaCache."""

    @staticmethod
    def type() -> str:
        """Return the MegaCache type identifier for Neuron NEFF artifacts.

        Returns:
            str: The ``"neuron_neff"`` artifact type identifier.
        """
        return ArtifactType.NEURON_NEFF.value

    def populate_cache(self) -> None:
        """Synchronously write this NEFF to the persistent cache during artifact loading.

        Note:
            A failed cache write logs a warning and returns without raising.
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
    """Represent serializable compiled output for the Dynamo backend.

    Args:
        graph_name (str, optional): Model or graph name. Defaults to ``""``.
        cache_key (str, optional): NEFF cache key and execution handle. Defaults to ``""``.
        hlo_cache_key (str, optional): Deprecated fallback key. Defaults to ``""``.
        hlo_bytes (bytes, optional): Serialized StableHLO. Defaults to ``b""``.
        io_spec (FunctionIO | None, optional): Flattened I/O specification. Defaults to ``None``.
        cast_spec (list[TensorSpec], optional): Output shape and dtype specifications used for
            final restoration. Defaults to an empty list.
        has_collectives (bool, optional): Whether the graph has collectives. Defaults to ``False``.
        retain_device (bool, optional): Restore outputs to the original input device type.
            Defaults to ``False``.
        pass_mutations (list[PassMutations], optional): Ordered mutations. Defaults to an
            empty list.
        alias_buffer_map (dict[int, int], optional): Output-to-input aliases. Defaults to an
            empty dict.
        dump_dir (str | None, optional): Directory for tensor I/O dumps. Defaults to ``None``.

    Note:
        Warm loads asynchronously resubmit embedded StableHLO through ``compile_graph`` so the
        persistent NEFF cache is consulted. WLO sidecar retrieval waits for compilation.
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
    dump_dir: str | None = None

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
                dump_dir=self.dump_dir,
            )
        return self._executor

    def __call__(self, inputs: Sequence) -> tuple[torch.Tensor, ...]:
        return self._get_executor()(*inputs)

    def prepare_for_serialization(self) -> None:
        """Clear the runtime executor while preserving serializable compilation data."""
        self._executor = None

    def post_compile(self, example_inputs, constants, graph_kwargs) -> None:
        """Accept the ``OutputCode`` post-compilation hook without additional work.

        Args:
            example_inputs (Sequence): Ignored example inputs supplied by Inductor.
            constants (CompiledFxGraphConstants): Ignored compiled FX graph constants supplied by
                Inductor.
            graph_kwargs (_CompileFxKwargs): Ignored FX compilation keyword arguments supplied by
                Inductor.
        """
        pass

    def set_triton_bundle(self, triton_bundle) -> None:
        """Accept a Triton bundle without storing it for Neuron execution.

        Args:
            triton_bundle (Any): Ignored Triton bundle supplied by the ``OutputCode`` interface.
        """
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
        """Return the package directory for the current rank.

        Returns:
            str: Original package directory, optionally suffixed with a rank element.

        Note:
            When no rank can be resolved, retain the shared directory and warn once.
        """
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
        """Return Dynamo's unsupported types without ``ProcessGroup``.

        Returns:
            tuple[type, ...]: Current unsupported types excluding ``ProcessGroup``.
        """
        # Call the original each time, so a later addition to the list survives.
        return tuple(t for t in original_unsupported_types() if t is not ProcessGroup)

    drop_process_group._neuron_patched = True
    dynamo_guards._get_unsupported_types = drop_process_group
    logger.debug("Patched _get_unsupported_types to drop ProcessGroup.")


def _set_device_mesh_state(mesh, state) -> None:
    """Restore a DeviceMesh and every process group name its registry held.

    Keep this at module level: pickle saves it by qualified name.
    """
    mesh_state, group_names = state
    mesh.__setstate__(mesh_state)
    for name in group_names:
        if name not in mesh._pg_registry:
            mesh._pg_registry[name] = _rebuild_process_group(name)


def _install_device_mesh_reducer() -> None:
    """Keep the process group names of a DeviceMesh during guard pickling.

    DeviceMesh.__getstate__ drops _pg_registry, and __setstate__ rebuilds it from
    the mesh's own _dim_group_names. A root mesh's registry also holds the groups
    of meshes derived from it (e.g. a flattened loss mesh), so a guard on
    ``mesh._root_mesh._pg_registry[name]`` fails to rebuild on restore with a
    KeyError. Save the registry's group names next to the state and resolve each
    one by name on restore.
    """
    from torch._dynamo.guards import GuardsStatePickler

    try:
        from torch.distributed.device_mesh import DeviceMesh
    except ImportError:
        return

    original_reducer_override = getattr(GuardsStatePickler, "reducer_override", None)
    if original_reducer_override is None or "__getstate__" not in DeviceMesh.__dict__:
        logger.warning(
            "Skipped the device mesh reducer. This torch version has no "
            "GuardsStatePickler.reducer_override or DeviceMesh.__getstate__. A warm "
            "restore can stop with a KeyError on a process group name. Check the "
            "torch version."
        )
        return
    if getattr(original_reducer_override, "_neuron_mesh_patched", False):
        return

    def reducer_override(self, obj):
        if type(obj) is DeviceMesh:
            state = (obj.__getstate__(), tuple(getattr(obj, "_pg_registry", ())))
            return copyreg.__newobj__, (DeviceMesh,), state, None, None, _set_device_mesh_state
        return original_reducer_override(self, obj)

    _install_reducer_override(reducer_override, original_reducer_override, "_neuron_mesh_patched")
    logger.debug(
        "Patched GuardsStatePickler.reducer_override. A DeviceMesh keeps the "
        "process group names of its registry."
    )


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
        """Reduce transport objects to unresolved stand-ins.

        Args:
            self (GuardsStatePickler): Pickler requesting the reduction.
            obj (object): Object being reduced for guard-state serialization.

        Returns:
            object: Stand-in reduction tuple or the original reducer's result.
        """
        if isinstance(obj, transport_types):
            label = f"{type(obj).__module__}.{type(obj).__name__}"
            return _rebuild_unresolved, (label,)
        return original_reducer_override(self, obj)

    _install_reducer_override(reducer_override, original_reducer_override, "_neuron_patched")
    logger.debug(
        "Patched GuardsStatePickler.reducer_override. The pickler replaces %d "
        "transport types with a stand-in.",
        len(transport_types),
    )


# ---------------------------------------------------------------------------
# A guarded function that pickle cannot find by name
# ---------------------------------------------------------------------------


def _rebuild_function(code, module, name, qualname, closure):
    """Rebuild a function from its code object. The unpickle step calls this.

    Keep this at module level: pickle saves it by qualified name. A guard reads
    attributes off the function and never calls it, so an empty globals dict
    stands in for a module that cannot be imported. _set_function_defaults
    applies the defaults afterwards.
    """
    f_globals = {}
    if isinstance(module, str):
        try:
            f_globals = importlib.import_module(module).__dict__
        except Exception:
            logger.debug("Rebuilt %s with empty globals: no module %r.", qualname, module)
    fn = types.FunctionType(code, f_globals, name, None, closure)
    fn.__qualname__ = qualname
    fn.__module__ = module
    return fn


def _set_function_defaults(fn, state) -> None:
    """Apply a rebuilt function's defaults. The unpickle step calls this.

    Defaults travel as pickle state, which pickle applies after it memoizes the
    function, so a default that refers back to the function does not recurse.
    """
    (fn.__defaults__,) = state


def _new_empty_cell() -> types.CellType:
    """Return an empty closure cell. The unpickle step calls this."""
    return types.CellType()


def _set_cell_contents(cell, state) -> None:
    """Fill a rebuilt closure cell. The unpickle step calls this.

    The contents travel wrapped in a 1-tuple: pickle skips the state step when
    the state is None, and None is a cell value that must not load as an empty
    cell.
    """
    (cell.cell_contents,) = state


def _reduce_cell(cell) -> tuple:
    """Pickle a closure cell as an empty cell plus its contents as state.

    An empty cell (a free variable assigned only on a path that did not run)
    stays empty, and contents that refer back to the cell's function load after
    pickle memoizes the cell.
    """
    try:
        contents = cell.cell_contents
    except ValueError:
        return _new_empty_cell, ()
    return _new_empty_cell, (), (contents,), None, None, _set_cell_contents


def _resolves_by_name(fn) -> bool:
    """Return True if fn.__module__ and fn.__qualname__ lead back to fn."""
    module = fn.__module__
    if not isinstance(module, str) or module not in sys.modules:
        return False
    resolved = sys.modules[module]
    for part in fn.__qualname__.split("."):
        resolved = getattr(resolved, part, None)
    return resolved is fn


def _install_guarded_function_reducer() -> None:
    """Pickle a guarded function by value when its name does not lead back to it.

    A functools.wraps wrapper copies the wrapped function's __module__ and
    __qualname__, so the name resolves to another object and the guard pickler
    saves the _Missing sentinel. A guard whose source walks through the wrapper
    (e.g. ``mod.forward.__closure__[0].cell_contents`` for a forward wrapped by
    torch.fx.traceback.annotate_fn) then fails to rebuild, and the package
    restore stops. Save such a function as its code, names, defaults, and
    closure. A function no guard reads keeps the sentinel.

    Defaults and cell contents travel as pickle state, applied after pickle
    memoizes the function or cell, so a default or captured value that refers
    back to it does not recurse. Every closure cell the guard pickler saves keeps
    an empty cell empty; the cell reducer of torch 2.14 reads cell_contents
    unconditionally and fails the save on one.
    """
    from torch._dynamo.guards import GuardsStatePickler

    original_reducer_override = getattr(GuardsStatePickler, "reducer_override", None)
    if original_reducer_override is None:
        logger.warning(
            "Skipped the guarded function reducer. This torch version has no "
            "GuardsStatePickler.reducer_override. A warm restore can stop with an "
            "AttributeError on _Missing. Check the torch version."
        )
        return
    if getattr(original_reducer_override, "_neuron_function_patched", False):
        return

    def reducer_override(self, obj):
        if isinstance(obj, types.CellType):
            return _reduce_cell(obj)
        if (
            inspect.isfunction(obj)
            and "<locals>" not in obj.__qualname__
            and id(obj) in getattr(self, "guard_tree_values", ())
            and not _resolves_by_name(obj)
        ):
            args = (obj.__code__, obj.__module__, obj.__name__, obj.__qualname__, obj.__closure__)
            return _rebuild_function, args, (obj.__defaults__,), None, None, _set_function_defaults
        return original_reducer_override(self, obj)

    _install_reducer_override(
        reducer_override, original_reducer_override, "_neuron_function_patched"
    )
    logger.debug(
        "Patched GuardsStatePickler.reducer_override. A guarded function that "
        "its name does not resolve to now pickles by value."
    )


# ---------------------------------------------------------------------------
# An FSDP2 module type that fully_shard has not created yet
# ---------------------------------------------------------------------------


def _install_fsdp_module_type_rebuild() -> None:
    """Create a guarded FSDP2 module type on restore if fully_shard has not.

    Guard pickling saves an FSDP2 module type as its original class, and the
    unpickle looks the FSDP subclass up in cls_to_fsdp_cls. TorchTitan calls
    torch.compile() on each block before fully_shard, so the package restore runs
    while the mapping is empty and fails with a KeyError on the original class.
    Create the subclass the way fully_shard does and record it in the mapping;
    fully_shard then reuses the same class, so a type guard on it matches.
    """
    from torch._dynamo.guards import GuardsStatePickler
    from torch.distributed.fsdp._fully_shard import _fully_shard

    original = GuardsStatePickler.__dict__.get("_unpickle_fsdp_module_type")
    fsdp_module = getattr(_fully_shard, "FSDPModule", None)
    deepcopy_fn = getattr(_fully_shard, "_unimplemented_deepcopy", None)
    if not isinstance(original, classmethod) or fsdp_module is None or deepcopy_fn is None:
        logger.warning(
            "Skipped the FSDP module type patch. This torch version has no "
            "GuardsStatePickler._unpickle_fsdp_module_type, FSDPModule, or "
            "_unimplemented_deepcopy. A warm restore can stop with a KeyError on "
            "an FSDP module type. Check the torch version."
        )
        return
    if getattr(original.__func__, "_neuron_patched", False):
        return

    def _unpickle_fsdp_module_type(cls, original_type: type) -> type:
        cls_to_fsdp_cls = _fully_shard.get_cls_to_fsdp_cls()
        fsdp_cls = cls_to_fsdp_cls.get(original_type)
        if fsdp_cls is None:
            # Match fully_shard's _apply_to_module: same name, bases, and dict.
            fsdp_cls = type(
                f"FSDP{original_type.__name__}",
                (fsdp_module, original_type),
                {"__deepcopy__": deepcopy_fn},
            )
            cls_to_fsdp_cls[original_type] = fsdp_cls
        return fsdp_cls

    _unpickle_fsdp_module_type._neuron_patched = True
    GuardsStatePickler._unpickle_fsdp_module_type = classmethod(_unpickle_fsdp_module_type)
    logger.debug(
        "Patched GuardsStatePickler._unpickle_fsdp_module_type. A restore before "
        "fully_shard now creates the FSDP module type."
    )


# Set while load_guards_state runs in this thread or task.
_restoring_guards_state: ContextVar[bool] = ContextVar("_restoring_guards_state", default=False)


def _install_fsdp_module_restore() -> None:
    """Keep the FSDP2 subclass of a module that a guard-state restore unpickles.

    FSDPModule.__new__ builds the original class, not the FSDP subclass it is
    called on, so an unpickled FSDP module comes back as its original class. The
    rebuilt TYPE_MATCH guard then names the original class, never matches the
    FSDP module at runtime, and Dynamo recompiles the restored frame. While
    load_guards_state runs, build the subclass itself. The flag is a ContextVar,
    so an FSDP module another thread constructs during a restore keeps the
    upstream behavior.
    """
    import torch._dynamo.package as dynamo_package
    from torch.distributed.fsdp._fully_shard import _fully_shard

    fsdp_module = getattr(_fully_shard, "FSDPModule", None)
    original_load = getattr(dynamo_package, "load_guards_state", None)
    if fsdp_module is None or "__new__" not in fsdp_module.__dict__ or original_load is None:
        logger.warning(
            "Skipped the FSDP module restore patch. This torch version has no "
            "FSDPModule.__new__ or torch._dynamo.package.load_guards_state. A warm "
            "restore of an FSDP module can recompile. Check the torch version."
        )
        return
    if getattr(original_load, "_neuron_fsdp_patched", False):
        return

    original_new = fsdp_module.__new__

    def fsdp_module_new(cls, *args, **kwargs):
        if _restoring_guards_state.get():
            orig_cls = cls.__mro__[cls._orig_cls_mro_index]
            return orig_cls.__new__(cls, *args, **kwargs)
        return original_new(cls, *args, **kwargs)

    def load_guards_state(guards_state):
        token = _restoring_guards_state.set(True)
        try:
            return original_load(guards_state)
        finally:
            _restoring_guards_state.reset(token)

    fsdp_module.__new__ = staticmethod(fsdp_module_new)
    load_guards_state._neuron_fsdp_patched = True
    dynamo_package.load_guards_state = load_guards_state
    logger.debug(
        "Patched FSDPModule.__new__ and load_guards_state. A restored FSDP module "
        "keeps its FSDP subclass."
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
        """Reuse an existing builtins-dictionary name when available.

        Returns:
            str: Existing name or the name installed by the original method.
        """
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

    # DeviceMesh pickling drops the groups a root mesh holds for derived meshes.
    _install_device_mesh_reducer()

    # Upstream torch saves a guarded functools.wraps wrapper as the _Missing
    # sentinel, so a guard that reads through it cannot rebuild on restore.
    _install_guarded_function_reducer()

    # A package restore can run before fully_shard creates an FSDP module type,
    # and FSDPModule.__new__ drops the FSDP subclass of an unpickled module.
    _install_fsdp_module_type_rebuild()
    _install_fsdp_module_restore()

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
