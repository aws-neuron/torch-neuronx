"""
Mesh registry for SPMD collective legalization.

Hooks DeviceMesh._init_process_groups to capture mesh topology, enabling
derivation of full replica groups during collective legalization. Without
this, each rank only sees its own process group, breaking SPMD compilation.

Usage:
    # Automatically applied at torch_neuronx import
    from torch_neuronx.distributed.mesh_registry import get_all_replica_groups

    # Returns full replica groups for any mesh-based PG
    groups = get_all_replica_groups("mesh_tp")  # e.g., [[0,1], [2,3], [4,5], [6,7]]
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

if TYPE_CHECKING:
    from typing import Any

__all__ = ["_apply_mesh_registry_patch", "clear_mesh_registry", "get_all_replica_groups"]

logger = logging.getLogger(__name__)

# Maps PG name -> list of replica groups (pre-computed at registration time).
# Note: entries are never removed on destroy_process_group — this is a known
# limitation. Re-creating a mesh with the same PG name overwrites the entry,
# so correctness is preserved, but stale entries from destroyed PGs remain.
# Not thread-safe; mesh creation and lookup are expected to be single-threaded.
_MESH_REGISTRY: dict[str, list[list[int]]] = {}

_patch_applied = False

_original_init_process_groups = DeviceMesh._init_process_groups


def _patched_init_process_groups(
    layout: Any, rank_map: torch.Tensor, mesh_dim_names: Any, backend_override: Any
) -> list:
    """Wrapper that registers replica groups after PG creation."""
    group_names = _original_init_process_groups(layout, rank_map, mesh_dim_names, backend_override)
    for dim_idx, pg_name in enumerate(group_names):
        try:
            pg_ranks_by_dim = layout[dim_idx].nest().remap_to_tensor(rank_map)
            groups = [row.tolist() for row in pg_ranks_by_dim]
            _MESH_REGISTRY[pg_name] = groups
            logger.debug(f"Registered mesh PG '{pg_name}' -> {len(groups)} groups")
        except Exception as e:
            logger.warning(f"Failed to register mesh PG '{pg_name}': {e}")
    return group_names


def _apply_mesh_registry_patch() -> None:
    """Apply mesh creation patch. Safe to call multiple times."""
    global _patch_applied
    if _patch_applied:
        return

    DeviceMesh._init_process_groups = staticmethod(_patched_init_process_groups)
    _patch_applied = True
    logger.debug("Applied mesh registry patch (_init_process_groups)")


def get_all_replica_groups(pg_name: str) -> list[list[int]]:
    """
    Get all replica groups for a process group.

    For mesh-based PGs, returns pre-computed groups from registration.
    For non-mesh PGs, falls back to current rank's group only.

    Args:
        pg_name: Process group identifier (e.g., "mesh_tp", "mesh_fsdp")

    Returns:
        List of replica groups, each a list of rank IDs.
        For SPMD, all ranks return identical groups.

    Example:
        # 2D mesh (4, 2) with FSDP outer, TP inner
        get_all_replica_groups("mesh_tp")   # [[0,1], [2,3], [4,5], [6,7]]
        get_all_replica_groups("mesh_fsdp") # [[0,2,4,6], [1,3,5,7]]
    """
    # TORCH_NEURONX_SPMD_DISABLE: bypass the registry and fall back to per-rank resolution.
    # Checked at read time (not hook time) so the registry is still populated,
    # allowing A/B comparison without restarting the process.
    if pg_name in _MESH_REGISTRY and os.environ.get("TORCH_NEURONX_SPMD_DISABLE", "") != "1":
        groups = _MESH_REGISTRY[pg_name]
        logger.debug(f"Found {len(groups)} replica groups for PG '{pg_name}'")
        return groups

    # Non-mesh PG: fall back to current rank's group
    pg = dist.distributed_c10d._resolve_process_group(pg_name)
    return [dist.get_process_group_ranks(pg)]


def clear_mesh_registry() -> None:
    """Clear the mesh registry. Useful for testing."""
    _MESH_REGISTRY.clear()
