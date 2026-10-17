import base64
import os
import pickle
import socket
from collections.abc import Callable
from contextlib import contextmanager

import pytest
import torch

from tests.utils.neuron_test_utils import get_rank_core


def broadcast_object_from_rank0(obj, rank: int, world_size: int, tag: str):
    """SPMD-safe replacement for ``dist.broadcast_object_list([...], src=0)``: return rank 0's
    ``obj`` on every rank via the PG's TCPStore, avoiding the neuron on-device object collective
    (its int64 size broadcast is flaky under parallel CI). ``tag`` must be unique per call site.
    """
    import time

    import torch.distributed as dist

    store = dist.distributed_c10d._get_default_store()
    data_key = f"__spmd_bcast__/data/{tag}"
    done_key = f"__spmd_bcast__/done/{tag}"

    if rank == 0:
        store.set(data_key, base64.b64encode(pickle.dumps(obj)).decode("ascii"))
    # store.get blocks until rank 0 has published the key (bounded by the
    # store's timeout), so it also acts as an arrival gate.
    payload = store.get(data_key)
    result = pickle.loads(base64.b64decode(payload))

    # Rendezvous: every rank reports arrival, then waits for all peers. This
    # prevents rank 0 from destroying the store before slower ranks have read.
    store.add(done_key, 1)
    while int(store.get(done_key).decode()) < world_size:
        time.sleep(0.005)

    # Tight release: a true (nrt) barrier so all ranks leave together and rank 0
    # -- which hosts the TCPStore server -- cannot race ahead into
    # ``destroy_process_group`` (which does store I/O) while a slower rank is
    # still mid-poll above. This uses nrt_barrier, not the int64 allreduce path.
    dist.barrier()

    return result


def get_free_port():
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return str(port)


@contextmanager
def distributed_context(master_addr, master_port, rank: int, world_size: int, process_index):
    """Context manager for distributed setup"""
    core = get_rank_core(rank)
    try:
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["NEURON_RT_VISIBLE_CORES"] = str(core)
        import torch.distributed as dist

        import torch_neuronx

        dist.init_process_group(backend="neuron")
        yield
    finally:
        del os.environ["MASTER_ADDR"]
        del os.environ["MASTER_PORT"]
        del os.environ["WORLD_SIZE"]
        del os.environ["RANK"]
        del os.environ["LOCAL_RANK"]
        del os.environ["NEURON_RT_VISIBLE_CORES"]
        # Destroy only what init_process_group built. An unconditional call raises
        # "Process group cannot be None" and hides the real setup error.
        if dist.is_initialized():
            dist.destroy_process_group()


_BASE_MASTER_PORT = 29500
_PORTS_PER_WORKER = 256


def _get_worker_port(process_index: int) -> str:
    """Find a free port within this worker's dedicated range."""
    start = _BASE_MASTER_PORT + process_index * _PORTS_PER_WORKER
    end = start + _PORTS_PER_WORKER
    for port in range(start, end):
        try:
            sock = socket.socket()
            # Allow binding to ports in TIME_WAIT state, matching PyTorch TCPStore's behavior
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", port))
            sock.close()
            return str(port)
        except OSError:
            continue
    raise RuntimeError(f"No free port in range [{start}, {end}) for worker {process_index}")


class DistributedTester:
    def __init__(
        self,
        world_size: int,
        backend: str = "neuron",
        process_index: int = 0,
    ):
        self.world_size = world_size
        self.backend = backend
        self.process_index = process_index

    def run_test(self, test_fn: Callable, **func_args):
        """Run test across multiple processes"""
        import torch.multiprocessing as mp

        error_queue = mp.get_context("spawn").Queue()
        master_addr = "localhost"
        master_port = _get_worker_port(self.process_index)
        try:
            mp.spawn(
                self._worker,
                args=(test_fn, func_args, master_addr, master_port, error_queue),
                nprocs=self.world_size,
                join=True,
            )
        except Exception as e:
            # Check for exceptions after processes complete
            errors = []
            while not error_queue.empty():
                errors.append(error_queue.get())
            if errors:
                errors.sort(key=lambda e: e[0])  # Sort by rank
                rank, error, tb = errors[0]  # Get lowest rank error
                msg = f"\n=== Error From Rank {rank} ===\n" f" Error: {error}\n Traceback: {tb}"

                # Re-raise the exception in main process so pytest capture it
                # Raise from None to remove unnecessary traceback from main process
                raise RuntimeError(msg) from None
            else:
                raise e

    def _worker(
        self, rank: int, test_fn: Callable, func_args, master_addr, master_port, error_queue
    ):
        """Worker function that runs on each process"""
        try:
            with distributed_context(
                master_addr, master_port, rank, self.world_size, self.process_index
            ):
                test_fn(rank, self.world_size, func_args)
                torch.neuron.synchronize()
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            error_queue.put((rank, f"{type(e).__name__}: {e}", tb))
            raise
