import socket

import torch.distributed as dist


def get_free_port():
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return str(port)


def get_reduce_type(reduce_op):
    if reduce_op == dist.ReduceOp.SUM:
        return "SUM"
    elif reduce_op == dist.ReduceOp.PRODUCT:
        return "PRODUCT"
    elif reduce_op == dist.ReduceOp.MIN:
        return "MIN"
    elif reduce_op == dist.ReduceOp.MAX:
        return "MAX"
    elif reduce_op == dist.ReduceOp.BAND:
        return "BAND"
    elif reduce_op == dist.ReduceOp.BOR:
        return "BOR"
    elif reduce_op == dist.ReduceOp.BXOR:
        return "BXOR"
    elif reduce_op == dist.ReduceOp.AVG:
        return "AVG"
    else:
        raise ValueError(f"Unsupported reduce operation: {reduce_op}")


def create_global_replica_group(world_size, current_group_ranks):
    """
    This helper function will be deprecated once compiler and collectives
    team fully support partial groups for all to all.
    """
    group_size = len(current_group_ranks)
    unused_ranks = [i for i in range(world_size) if i not in current_group_ranks]
    global_replica_groups = [current_group_ranks] + [
        unused_ranks[j : j + group_size] for j in range(0, world_size - group_size, group_size)
    ]
    return global_replica_groups
