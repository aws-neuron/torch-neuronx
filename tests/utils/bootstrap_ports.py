"""CCOM bootstrap-port assignment for parallel test workers."""

import os

# Base port used when NEURON_RT_PORT is unset. In the dynamic/private range so
# the per-worker offsets are unlikely to collide with well-known services.
_DEFAULT_BASE_PORT = 20000


def configure_worker_bootstrap_port(worker_index: int) -> None:
    """Pin a distinct CCOM root port for this xdist worker.

    Under ``pytest -n<N>`` many process groups bootstrap concurrently on one
    host. Left to ``get_free_port()``, two groups can race onto the same
    just-freed port and deadlock. Instead, derive a deterministic per-worker port
    from the existing ``NEURON_RT_PORT`` base and set ``NEURON_RT_ROOT_COMM_ID``
    directly: the Neuron backend only selects a port when that variable is unset,
    so this keeps concurrent groups on disjoint ports with no new env var.
    """
    base = int(os.environ.get("NEURON_RT_PORT", str(_DEFAULT_BASE_PORT)))
    port = base + worker_index
    if not 1 <= port <= 65535:
        raise RuntimeError(f"No valid CCOM bootstrap port for xdist worker {worker_index}")
    addr = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["NEURON_RT_ROOT_COMM_ID"] = f"{addr}:{port}"
