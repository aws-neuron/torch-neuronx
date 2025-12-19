"""
Converts PyTorch's _c10d_functional collective operations
(all_reduce, all_gather, reduce_scatter, all_to_all) to a format compatible with StableHLO.
By re-writing "group name" argument from the process group identifier (e.g '1') to
explicit list of participating ranks (e.g '[[0, 2, 4, 6]]').
"""

import logging

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _get_process_group_ranks(group_name: str) -> list[int]:
    """
    Get the list of participating ranks for a process group.

    Args:
        group_name: Process group identifier (e.g., "0", "3", etc.)

    Returns:
        List of rank IDs participating in the process group

    Example:
        >>> get_process_group_ranks("3")
        [0, 1]
        >>> get_process_group_ranks("0")
        [0, 1, 2, 3]
    """
    try:
        # Resolve the process group from its name/identifier
        # e.g: torch_neuronx.distributed.backend.ProcessGroupNeuron
        pg = dist.distributed_c10d._resolve_process_group(group_name)

        # Get the ranks participating in this process group
        # e.g: [0, 1, ...]
        ranks = dist.get_process_group_ranks(pg)

        logger.debug(f"Process group '{group_name}' has ranks: {ranks}")
        return ranks

    except Exception as e:
        logger.warning(f"Failed to resolve process group '{group_name}': {e}")
        logger.warning("Falling back to default world group")
        # Fallback: return all ranks in the default world group
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        return list(range(world_size))


def legalize_replica_groups(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Rewrite collective operation group names mapped to global partial replica groups

    Args:
        gm: Input GraphModule containing collective operations

    Returns:
        Modified GraphModule with rewritten group names

    Example:
        >>> Transforms the following
        opcode         name         target                                args
        -------------  -----------  ------------------------------------  --------------------
        call_function  all_reduce   _c10d_functional.all_reduce.default   (arg0_1, 'sum', '0')

        >>> Into
        opcode         name         target                                args
        -------------  -----------  ------------------------------------  ---------------------------
        call_function  all_reduce   _c10d_functional.all_reduce.default   (arg0_1, 'sum', '[[0, 1]]')
    """  # noqa: E501

    # TODO: Support `broadcast`.`
    # Maps collective op to the index of its "group" argument
    # e.g:
    #   all_reduce(tensor, reduce_op, group) -> 2
    #   reduce_scatter_tensor(tensor, reduce_op, scatter_dim, group) -> 3
    collective_ops_to_group_index = {
        "_c10d_functional.all_reduce.default": 2,
        "_c10d_functional.all_gather_into_tensor.default": 2,
        "_c10d_functional.reduce_scatter_tensor.default": 3,
        "_c10d_functional.all_to_all_single.default": 3,
    }

    for node in gm.graph.nodes:
        if node.op == "call_function" and str(node.target) in collective_ops_to_group_index:
            # Replace the group name with flattened local replica_group
            args = list(node.args)
            group_index = collective_ops_to_group_index[str(node.target)]
            group_arg = args[group_index]
            logger.debug(
                f"Found collective operation: {node.target} group ({group_arg}) -> "
                + f"({_get_process_group_ranks(args[group_index])})"
            )
            args[group_index] = "[" + str(_get_process_group_ranks(group_arg)) + "]"
            node.args = tuple(args)

    gm.recompile()
    return gm
