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

import inspect
import logging
import os
from copy import deepcopy
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
_RAW_PENDING: list[dict[tuple[int, ...], list[str]]] = []

# Resolved once from the installed torch so argument extraction follows parameter
# names rather than positions, which shift between torch versions.
_NEW_GROUP_SIGNATURE = inspect.signature(dist.new_group)

_original_init_process_groups = DeviceMesh._init_process_groups
_original_new_group = dist.new_group


def _register_replica_groups(pg_name: str, groups: list[list[int]]) -> None:
    """Register a complete replica-group table for a process-group name."""
    normalized_name = str(pg_name)
    normalized_groups = deepcopy(groups)
    existing = _MESH_REGISTRY.get(normalized_name)
    if existing is not None and existing != normalized_groups:
        raise RuntimeError(
            f"Process group '{normalized_name}' is already registered with "
            f"different replica groups: {existing} != {normalized_groups}"
        )
    _MESH_REGISTRY[normalized_name] = normalized_groups


def _get_registered_replica_groups(pg_name: str) -> list[list[int]] | None:
    """Return a copy of a registered table, or None when it is unknown."""
    groups = _MESH_REGISTRY.get(str(pg_name))
    return None if groups is None else deepcopy(groups)


def _normalize_new_group_ranks(args: tuple, kwargs: dict, world_size: int) -> tuple[int, ...]:
    """Normalize ``new_group`` rank arguments the way torch itself would.

    Binds the call against the installed ``dist.new_group`` signature so
    ``ranks`` and ``sort_ranks`` resolve by name. Reading them by position
    would silently pick up a neighbouring argument if torch reorders its
    parameters, corrupting the captured rank order instead of failing.
    """
    bound = _NEW_GROUP_SIGNATURE.bind_partial(*args, **kwargs)
    bound.apply_defaults()
    ranks = bound.arguments.get("ranks")
    if ranks is None:
        ranks = range(world_size)
    # Absent when a torch version drops the parameter; sorting matches its old default.
    sort_ranks = bound.arguments.get("sort_ranks", True)
    normalized = tuple(int(rank) for rank in ranks)
    return tuple(sorted(normalized)) if sort_ranks else normalized


def _record_raw_new_group(
    process_group: Any, requested_ranks: tuple[int, ...], world_size: int
) -> None:
    """Match cells to the newest equal-width, disjoint pending candidate.
    Nonmember cells contribute without local names.
    Register local names once the candidate covers the world.
    """
    # A nonmember rank gets a sentinel rather than a real group, so it contributes
    # its cell to the partition without a local name.
    if process_group == dist.GroupMember.NON_GROUP_MEMBER:
        local_name = None
    else:
        group_name = getattr(process_group, "group_name", None)
        if group_name is None:
            raise RuntimeError("Created process group did not expose group_name")
        local_name = str(group_name)

    requested_rank_set = set(requested_ranks)

    candidate = None
    for pending in _RAW_PENDING:
        expected_width = len(next(iter(pending)))
        overlaps = any(requested_rank_set.intersection(cell) for cell in pending)
        if (
            requested_ranks not in pending
            and len(requested_ranks) == expected_width
            and not overlaps
        ):
            candidate = pending
            break

    if candidate is None:
        candidate = {}
        # Prefer newer candidates so a stale partial partition cannot consume
        # a sibling belonging to a later sequence.
        _RAW_PENDING.insert(0, candidate)

    candidate[requested_ranks] = [] if local_name is None else [local_name]

    seen = {rank for cell in candidate for rank in cell}
    if seen != set(range(world_size)):
        return

    groups = [list(cell) for cell in candidate]
    names = [name for cell_names in candidate.values() for name in cell_names]
    for name in names:
        _register_replica_groups(name, groups)

    _RAW_PENDING.remove(candidate)


def _patched_new_group(*args: Any, **kwargs: Any) -> Any:
    """Capture raw new_group topology while preserving PyTorch behavior."""
    try:
        world_size = dist.get_world_size()
        # Materialize a one-shot ``ranks`` iterable (generator/map/filter) once
        # so our capture and the forwarded torch call see identical ranks;
        # otherwise torch would receive an iterator this function drained.
        if args and args[0] is not None:
            args = (list(args[0]), *args[1:])
        elif kwargs.get("ranks") is not None:
            kwargs = {**kwargs, "ranks": list(kwargs["ranks"])}
        requested_ranks = _normalize_new_group_ranks(args, kwargs, world_size)
    except Exception as exc:
        logger.debug("Unable to inspect new_group arguments: %s", exc)
        return _original_new_group(*args, **kwargs)

    process_group = _original_new_group(*args, **kwargs)
    try:
        _record_raw_new_group(process_group, requested_ranks, world_size)
    except Exception as exc:
        logger.debug("Unable to capture raw new_group: %s", exc)
    return process_group


def _patched_init_process_groups(
    layout: Any, rank_map: torch.Tensor, mesh_dim_names: Any, backend_override: Any
) -> list:
    """Wrapper that registers replica groups after PG creation."""
    group_names = _original_init_process_groups(layout, rank_map, mesh_dim_names, backend_override)
    world_size = rank_map.numel()
    for dim_idx, pg_name in enumerate(group_names):
        try:
            groups = layout[dim_idx].global_ranks(world_size)
            _register_replica_groups(pg_name, groups)
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
    # Subgroup helpers call the c10d module alias directly.
    dist.new_group = _patched_new_group
    dist.distributed_c10d.new_group = _patched_new_group
    _patch_applied = True
    logger.debug(
        "Applied mesh registry patch (_init_process_groups, new_group, distributed_c10d.new_group)"
    )


def get_all_replica_groups(pg_name: str) -> list[list[int]]:
    """Return replica groups for a process-group name.

    Args:
        pg_name (str): Process-group identifier.

    Returns:
        list[list[int]]: Complete registered groups, or the resolved local ranks
        when the registry is disabled or incomplete. Registered groups are
        defensive copies and are identical on every rank during SPMD compilation.

    Raises:
        RuntimeError: If the process-group name cannot be resolved.

    Note:
        ``TORCH_NEURONX_SPMD_DISABLE=1`` bypasses the registry. An incomplete
        group logs an error and falls back to local ranks, which violates the
        SPMD identical-replica-groups invariant and may produce wrong collectives.
    """
    # TORCH_NEURONX_SPMD_DISABLE: bypass the registry and fall back to per-rank resolution.
    # Checked at read time (not hook time) so the registry is still populated,
    # allowing A/B comparison without restarting the process.
    spmd_disabled = os.environ.get("TORCH_NEURONX_SPMD_DISABLE", "") == "1"
    groups = _get_registered_replica_groups(pg_name)
    if groups is not None and not spmd_disabled:
        logger.debug(f"Found {len(groups)} replica groups for PG '{pg_name}'")
        return groups

    try:
        pg = dist.distributed_c10d._resolve_process_group(pg_name)
        local_ranks = dist.get_process_group_ranks(pg)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to resolve process group '{pg_name}' for replica-group legalization"
        ) from exc

    if spmd_disabled:
        return [local_ranks]

    world_size = dist.get_world_size() if dist.is_initialized() else len(local_ranks)
    if sorted(local_ranks) == list(range(world_size)):
        return [local_ranks]

    logger.error(
        f"No registered replica groups for PG '{pg_name}' in SPMD mode "
        f"(local ranks {local_ranks}, world {world_size}); falling back to per-rank groups, "
        f"which violates the SPMD identical-replica-groups invariant and may give wrong "
        f"collectives. Register the mesh or set TORCH_NEURONX_SPMD_DISABLE=1 to opt in."
    )
    return [local_ranks]


def clear_mesh_registry() -> None:
    """Clear registered and pending process-group topology.

    Note:
        Use this to reset process-wide registry state between tests.
    """
    _MESH_REGISTRY.clear()
    _RAW_PENDING.clear()
