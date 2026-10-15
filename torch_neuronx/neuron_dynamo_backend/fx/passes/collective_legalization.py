"""
Converts PyTorch's _c10d_functional collective operations
(all_reduce, all_gather, reduce_scatter, all_to_all, broadcast) to a format
compatible with StableHLO.
By re-writing "group name" argument from the process group identifier (e.g '1') to
explicit list of participating ranks (e.g '[[0, 1], [2, 3]]').

For SPMD compilation, all ranks must produce identical replica groups. This module
uses the mesh registry to derive full groups from DeviceMesh topology.
"""

import logging

import torch
import torch.distributed as dist
from torch.fx.passes.infra.pass_base import PassResult

from torch_neuronx.distributed.mesh_registry import get_all_replica_groups
from torch_neuronx.neuron_dynamo_backend.fx.passes.neuron_fx_pass import (
    NeuronFXPass,
)

logger = logging.getLogger(__name__)

# Maps collective op to the index of its "group" argument
# e.g:
#   all_reduce(tensor, reduce_op, group) -> 2
#   reduce_scatter_tensor(tensor, reduce_op, scatter_dim, group) -> 3
#   broadcast(tensor, src, group) -> 2
_COLLECTIVE_OPS_GROUP_INDEX = {
    "_c10d_functional.all_reduce.default": 2,
    "_c10d_functional.all_gather_into_tensor.default": 2,
    "_c10d_functional.reduce_scatter_tensor.default": 3,
    "_c10d_functional.all_to_all_single.default": 3,
    "_dtensor.shard_dim_alltoall.default": 3,
    "_c10d_functional.broadcast.default": 2,
}


def _get_process_group_ranks(group_name: str) -> list[list[int]]:
    """
    Get all replica groups for a process group (SPMD-compatible).

    For mesh-based PGs (registered from DeviceMesh topology) and raw
    dist.new_group PGs (captured at creation), returns the full replica groups.
    If the group cannot be resolved, falls back to the default world group so
    legalization degrades gracefully instead of aborting compilation.

    Args:
        group_name: Process group identifier (e.g., "0", "3", etc.)

    Returns:
        List of replica groups, each a list of rank IDs.

    Example:
        # 2D mesh (4, 2) with FSDP outer, TP inner
        >>> _get_process_group_ranks("tp_pg")
        [[0, 1], [2, 3], [4, 5], [6, 7]]
        >>> _get_process_group_ranks("fsdp_pg")
        [[0, 2, 4, 6], [1, 3, 5, 7]]
    """
    try:
        groups = get_all_replica_groups(group_name)
        logger.debug(f"Process group '{group_name}' has replica groups: {groups}")
        return groups
    except Exception as e:
        logger.error(
            f"Failed to resolve replica groups for process group '{group_name}': {e}. "
            f"Falling back to the full world group; if '{group_name}' does not span the "
            f"whole world this rewrites a sharded collective into a global reduction "
            f"(WRONG NUMERICS). Register the mesh/PG to avoid this."
        )
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        return [list(range(world_size))]


class CollectiveLegalization(NeuronFXPass):
    """
    Rewrite collective operation group names to explicit replica groups.

    Example:
        Transforms the following:

        opcode         name         target                                args
        -------------  -----------  ------------------------------------  --------------------
        call_function  all_reduce   _c10d_functional.all_reduce.default   (arg0_1, 'sum', '0')

        Into:

        opcode         name         target                                args
        -------------  -----------  ------------------------------------  ---------------------------
        call_function  all_reduce   _c10d_functional.all_reduce.default   (arg0_1, 'sum', '[[0, 1]]')

    """  # noqa: E501

    def call(self, gm: torch.fx.GraphModule) -> PassResult:
        modified = False

        for node in gm.graph.nodes:
            if node.op == "call_function" and str(node.target) in _COLLECTIVE_OPS_GROUP_INDEX:
                args = list(node.args)
                group_index = _COLLECTIVE_OPS_GROUP_INDEX[str(node.target)]
                group_arg = args[group_index]
                replica_groups = _get_process_group_ranks(group_arg)
                logger.debug(f"Collective {node.target}: group ({group_arg}) -> ({replica_groups})")
                args[group_index] = str(replica_groups)
                node.args = tuple(args)
                modified = True

        if modified:
            gm.recompile()

        return PassResult(gm, modified=modified)
