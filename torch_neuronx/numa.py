"""NUMA topology optimization for Neuron devices.

Provides NUMA-aware CPU affinity and memory binding for multi-rank Neuron
workloads. API design follows :mod:`torch.numa.binding` (PyTorch 2.11+) adapted
for Neuron's PrivateUse1 device model.

NUMA CPU affinity is on by default. Set ``NEURON_NUMA_PIN=0`` to disable.

Topology is discovered via ``neuron-ls --json-output`` (called once, cached).
sysfs helpers mirror :mod:`torch.numa.binding` where applicable.
"""

from __future__ import annotations

__all__ = [
    "get_numa_status",
    "pin_to_numa",
]

import dataclasses
import functools
import json
import logging
import os
import subprocess
import threading
from enum import Enum

logger = logging.getLogger(__name__)

# NeuronCore IDs reported per device by neuron-ls (e.g. trn2: IDs 0-7 per device).
_NEURONCORE_IDS_PER_DEVICE = 8


# ---------------------------------------------------------------------------
# State — guarded by _lock
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _NUMAState:
    """Mutable state for NUMA optimizations, protected by ``_lock``."""

    numa_pinned: bool = False


_lock = threading.Lock()
_state = _NUMAState()


# ---------------------------------------------------------------------------
# AffinityMode — mirrors torch.numa.binding.AffinityMode (subset)
# ---------------------------------------------------------------------------


class AffinityMode(str, Enum):
    """Select a NUMA CPU-affinity strategy.

    Note:
        This subset adapts ``torch.numa.binding.AffinityMode`` from PyTorch 2.11 or
        later. ``NODE`` binds all CPUs on the NUMA node containing the Neuron device.
    """

    NODE = "node"
    """Bind to all CPUs on the NUMA node containing the Neuron device."""


# ---------------------------------------------------------------------------
# Environment variable helpers
# ---------------------------------------------------------------------------


def _is_enabled(var: str, default: str = "1") -> bool:
    """Return whether *var* is enabled."""
    return os.environ.get(var, default) != "0"


# ---------------------------------------------------------------------------
# sysfs helpers (aligned with torch.numa.binding internals)
# ---------------------------------------------------------------------------


def _parse_range_list(range_list_str: str) -> list[int]:
    """Parse a sysfs range string like ``"0-47,96-143"`` into a sorted int list.

    Equivalent to :func:`torch.numa.binding._get_set_of_int_from_ranges_str`
    but returns a list for ordered iteration.
    """
    result: list[int] = []
    for part in range_list_str.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return result


# ---------------------------------------------------------------------------
# Topology discovery (cached)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _discover_topology() -> dict[int, dict] | None:
    """Call ``neuron-ls --json-output`` and return device topology.

    Returns ``{device_index: {"numa_node": int, "bdf": str,
    "cpu_affinity": str, "neuroncore_ids": list[int]}}``
    or ``None`` on failure (non-zero exit, not found, exception).
    Callers treat ``None`` as "no topology" and skip NUMA pinning silently.
    """
    try:
        result = subprocess.run(
            ["neuron-ls", "--json-output"],
            capture_output=True,
            text=True,
            timeout=30,
            start_new_session=True,
        )
        if result.returncode != 0:
            logger.warning(
                "neuron-ls returned %d: %s",
                result.returncode,
                result.stderr.strip(),
            )
            return None
        devices = json.loads(result.stdout)
        return {
            d["neuron_device"]: {
                # sysfs reports -1 when NUMA affinity is unknown
                # (single-NUMA systems, missing ACPI SRAT entry).  Clamp to 0.
                "numa_node": max(int(d["numa_node"]), 0),
                "bdf": d["bdf"],
                "cpu_affinity": d["cpu_affinity"],
                "neuroncore_ids": d["neuroncore_ids"],
            }
            for d in devices
        }
    except FileNotFoundError:
        logger.warning("neuron-ls not found in PATH; NUMA optimizations disabled")
        return None
    except Exception as exc:
        logger.warning("Failed to query neuron-ls: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Visible-core → device mapping
# ---------------------------------------------------------------------------


def _visible_core_to_device_index(visible_core: int) -> int:
    """Map a visible NeuronCore index to a device index."""
    lnc_config = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))
    return visible_core * lnc_config // _NEURONCORE_IDS_PER_DEVICE


def _get_visible_core(visible_core: int | None) -> int | None:
    """Resolve visible core from *visible_core* arg or ``NEURON_RT_VISIBLE_CORES``."""
    if visible_core is not None:
        return visible_core
    vc_str = os.environ.get("NEURON_RT_VISIBLE_CORES")
    if vc_str is None:
        return None
    # May be a single int or a range like "0-3"; take the first value.
    # All cores in a range belong to the same device (and NUMA node),
    # so any one is sufficient for the device → NUMA lookup.
    try:
        return int(vc_str)
    except ValueError:
        try:
            return int(vc_str.split("-")[0].split(",")[0])
        except (ValueError, IndexError):
            logger.warning("Cannot parse NEURON_RT_VISIBLE_CORES=%r", vc_str)
            return None


def _get_device_info(visible_core: int | None) -> dict | None:
    """Return topology dict for the device owning *visible_core*."""
    vc = _get_visible_core(visible_core)
    if vc is None:
        return None
    topo = _discover_topology()
    if topo is None:
        return None
    device_idx = _visible_core_to_device_index(vc)
    info = topo.get(device_idx)
    if info is None:
        logger.warning("No topology for device %d (visible_core=%d)", device_idx, vc)
    return info


# ---------------------------------------------------------------------------
# Thread pinning (mirrors torch.numa.binding._bind_all_threads_in_current_process_to_logical_cpus)
# ---------------------------------------------------------------------------


def _pin_all_threads(cpulist: list[int]) -> int:
    """Set CPU affinity for every thread in this process.

    Enumerates ``/proc/self/task/`` so threads created before this call
    (ZeroMQ I/O, torch thread-pools, NRT threads) are also pinned.

    Returns the number of threads successfully pinned.
    """
    cpu_set = set(cpulist)
    pinned = 0
    try:
        tids = os.listdir("/proc/self/task")
        for tid_str in tids:
            try:
                os.sched_setaffinity(int(tid_str), cpu_set)
                pinned += 1
            except OSError as exc:
                logger.debug("Could not pin tid %s: %s (errno=%s)", tid_str, exc, exc.errno)
    except FileNotFoundError:
        os.sched_setaffinity(0, cpu_set)
        pinned = 1
    return pinned


# ---------------------------------------------------------------------------
# Feature 1: NUMA CPU affinity
# ---------------------------------------------------------------------------


def pin_to_numa(visible_core: int | None = None) -> None:
    """Pin all process threads to CPUs local to a rank's Neuron device.

    Args:
        visible_core (int, optional): Visible NeuronCore index. Defaults to
            ``None``, which resolves the first value in ``NEURON_RT_VISIBLE_CORES``.

    Note:
        ``NEURON_NUMA_PIN=0`` disables pinning. The operation is process-wide and
        idempotent. It binds to the CPU-affinity list reported for the resolved
        device and returns without pinning when topology is unavailable.
    """
    if not _is_enabled("NEURON_NUMA_PIN"):
        logger.debug("NUMA pinning disabled (NEURON_NUMA_PIN=0)")
        return

    with _lock:
        if _state.numa_pinned:
            return

        info = _get_device_info(visible_core)
        if info is None:
            logger.debug("Skipping NUMA pinning: no device topology available")
            return

        cpulist = _parse_range_list(info["cpu_affinity"])
        pinned = _pin_all_threads(cpulist)
        _state.numa_pinned = True

        logger.info(
            "Pinned %d thread(s) to NUMA node %d (CPUs: %s)",
            pinned,
            info["numa_node"],
            info["cpu_affinity"],
        )


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------


def get_numa_status() -> dict:
    """Return diagnostic information about the current NUMA optimization state.

    Returns:
        dict: Values for ``numa_pin_enabled``, ``numa_pinned``,
            ``current_affinity``, ``topology_available``, and ``num_devices``.
    """
    topo = _discover_topology()
    current_affinity = sorted(os.sched_getaffinity(0))
    with _lock:
        numa_pinned = _state.numa_pinned
    return {
        "numa_pin_enabled": _is_enabled("NEURON_NUMA_PIN"),
        "numa_pinned": numa_pinned,
        "current_affinity": current_affinity,
        "topology_available": topo is not None,
        "num_devices": len(topo) if topo else 0,
    }


# ---------------------------------------------------------------------------
# Console-script entry point: neuron-numa-launch
# ---------------------------------------------------------------------------


def numa_launch_main() -> None:
    """Launch Python with NUMA-local binding for ``LOCAL_RANK``.

    Raises:
        ValueError: If ``LOCAL_RANK`` is not an integer.
        OSError: If replacing the current process fails.

    Note:
        This console entry point binds CPU and memory for ``torchrun --no-python``.
        It falls back to unbound Python when ``LOCAL_RANK``, ``numactl``, or device
        topology is unavailable.
    """
    import shutil
    import sys

    rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if rank < 0 or not shutil.which("numactl"):
        logger.debug(
            "neuron-numa-launch: no LOCAL_RANK or numactl not found, "
            "falling back to plain python"
        )
        os.execvp("python", ["python", "-u"] + sys.argv[1:])
        return  # unreachable, but keeps linters happy

    info = _get_device_info(rank)
    if info is None:
        logger.warning(
            "neuron-numa-launch: no topology for rank %d, skipping numactl",
            rank,
        )
        os.execvp("python", ["python", "-u"] + sys.argv[1:])
        return

    numa_node = info["numa_node"]
    try:
        os.execvp(
            "numactl",
            [
                "numactl",
                f"--membind={numa_node}",
                f"--cpunodebind={numa_node}",
                "--",
                "python",
                "-u",
            ]
            + sys.argv[1:],
        )
    except OSError as exc:
        logger.error("neuron-numa-launch: execvp failed: %s", exc)
        raise
