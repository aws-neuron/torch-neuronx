"""Neuron distributed backend — registration, factory, and runtime setup."""

import contextlib
import os
import socket
import time

import torch.distributed as dist

import torch_neuronx

from .ops.functional_collectives import register_neuron_collectives
from .utils import get_free_port

_WORLD_SIZE = None


def _create_neuron_process_group(dist_backend_opts, pg_options):
    """Factory for neuron backend. Called by c10d with extended_api=True."""
    from torch._C._distributed_c10d import Backend as _CBackend

    from torch_neuronx.distributed.cpp_backend import create_neuron_cpp_backend

    prefix_store = dist_backend_opts.store
    rank = dist_backend_opts.group_rank
    size = dist_backend_opts.group_size
    timeout = dist_backend_opts.timeout
    global_ranks_in_group = dist_backend_opts.global_ranks_in_group

    global _WORLD_SIZE
    if _WORLD_SIZE is None:
        _WORLD_SIZE = size
        _neuron_runtime_setup(rank, size, prefix_store)

    options = _CBackend.Options(backend="neuron", timeout=timeout)
    if global_ranks_in_group:
        options.global_ranks_in_group = list(global_ranks_in_group)
    else:
        options.global_ranks_in_group = list(range(size))
    # For subgroups, `size` is the group size; `_WORLD_SIZE` is the true world size.
    # `prefix_store` is threaded through so the C++ backend can publish/resolve
    # RDMA descriptors via Store.set/get keyed by global rank.
    return create_neuron_cpp_backend(prefix_store, options, rank, size, _WORLD_SIZE)


def _register_neuron_backend():
    dist.Backend.register_backend(
        "neuron", _create_neuron_process_group, extended_api=True, devices=["neuron"]
    )
    dist.Backend.default_device_backend_map["neuron"] = "neuron"


def rendezvous_handler():
    pass


def _graph_break_object_collectives():
    """Force ``torch.compile`` to graph-break on Python object-collectives.

    Collectives such as ``all_gather_object`` pickle arbitrary Python objects and
    operate on the ProcessGroup script object, so they are inherently untraceable.
    Under PT2.12, dynamo otherwise traces into ``_get_object_coll_device`` and
    hard-errors while accessing the ProcessGroup opaque member ``_device_types``
    (``UnsafeScriptObjectError``) instead of breaking cleanly.

    Wrapping them with ``torch._dynamo.disable`` makes dynamo graph-break at the
    call site (running the collective eagerly); eager (non-compiled) behavior is
    unchanged. Mirrors how the untraceable Neuron C extensions are wrapped in
    ``torch_neuronx.__init__``.
    """
    import torch

    for _name in (
        "all_gather_object",
        "gather_object",
        "broadcast_object_list",
        "scatter_object_list",
    ):
        _fn = getattr(dist, _name, None)
        if _fn is not None:
            with contextlib.suppress(TypeError, AttributeError):
                setattr(dist, _name, torch._dynamo.disable(_fn))


_register_neuron_backend()

dist.register_rendezvous_handler("neuron", rendezvous_handler)

register_neuron_collectives()

_graph_break_object_collectives()


# ─────────────────────────────────────────────────────────────────────────
# Runtime setup helpers
# ─────────────────────────────────────────────────────────────────────────


def resolve_visible_cores(visible_cores, local_rank):
    """Resolve NEURON_RT_VISIBLE_CORES for this process."""
    if not visible_cores:
        os.environ["NEURON_RT_VISIBLE_CORES"] = str(local_rank)
        return

    if "," in visible_cores:
        cores = [c.strip() for c in visible_cores.split(",")]
    elif "-" in visible_cores:
        start, end = visible_cores.split("-", 1)
        cores = [str(c) for c in range(int(start), int(end) + 1)]
    else:
        os.environ["NEURON_RT_VISIBLE_CORES"] = visible_cores
        return

    if local_rank >= len(cores):
        raise RuntimeError(
            f"local_rank {local_rank} exceeds available visible cores ({len(cores)})"
        )
    os.environ["NEURON_RT_VISIBLE_CORES"] = cores[local_rank]


def _set_rt_visible_cores(rank, world_size, store):
    torch_neuronx._C._reset_vnc_count()

    local_rank_env = os.environ.get("LOCAL_RANK")
    local_world_size_env = os.environ.get("LOCAL_WORLD_SIZE")

    visible_cores = os.environ.get("NEURON_RT_VISIBLE_CORES")

    if local_rank_env is not None and local_world_size_env is not None:
        local_rank = int(local_rank_env)
        local_world_size = int(local_world_size_env)
        resolve_visible_cores(visible_cores, local_rank)
        return local_rank, local_world_size

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    store.set(f"worker_{rank}", f"{local_ip}:{rank}")

    while True:
        try:
            registered_count = 0
            for i in range(world_size):
                try:
                    store.get(f"worker_{i}")
                    registered_count += 1
                except Exception:
                    break
            if registered_count == world_size:
                break
        except Exception:
            pass
        time.sleep(0.1)

    workers_on_node = []
    for i in range(world_size):
        worker_info = store.get(f"worker_{i}").decode()
        worker_ip, worker_rank = worker_info.split(":")
        if worker_ip == local_ip:
            workers_on_node.append(int(worker_rank))

    workers_on_node.sort()
    local_rank = workers_on_node.index(rank)
    local_world_size = len(workers_on_node)

    resolve_visible_cores(visible_cores, local_rank)
    return local_rank, local_world_size


def _set_root_comm_id(rank, local_rank, world_size, store):
    """Set root communication address/port for NRT barrier."""
    if os.environ.get("NEURON_RT_ROOT_COMM_ID") is None:
        if rank == 0:
            root_port = get_free_port()
            store.set("NEURON_RT_PORT", root_port)
        else:
            root_port = store.get("NEURON_RT_PORT").decode("UTF-8")

        root_addr = os.environ.get("MASTER_ADDR", "localhost")
        os.environ["NEURON_RT_ROOT_COMM_ID"] = f"{root_addr}:{root_port}"


def _neuron_runtime_setup(rank, size, store):
    assert (
        not torch_neuronx.is_neuron_runtime_initialized()
    ), "Neuron runtime should not be initialized before the init_process_group."

    local_rank, local_world_size = _set_rt_visible_cores(rank, size, store)
    _set_root_comm_id(rank, local_rank, size, store)

    torch_neuronx._C._set_world_size(size)
    torch_neuronx._C._set_rank(rank)
    torch_neuronx._C._set_local_world_size(local_world_size)
    torch_neuronx._C._set_local_device_start_index(local_rank)

    torch_neuronx._lazy_init()

    from torch_neuronx.numa import pin_to_numa

    pin_to_numa(local_rank)

    assert (
        torch_neuronx._C._vnc_count() == 1
    ), "Attempted to have more than one device per process in distributed setup"

    torch_neuronx._C._nrt_barrier(0, rank, size)
