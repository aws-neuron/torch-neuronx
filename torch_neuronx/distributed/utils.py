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
    elif reduce_op == dist.ReduceOp.PREMUL_SUM:
        return "PREMUL_SUM"
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


def parse_and_update_inputs(inputs, reduce_op):
    """
    Preprocess inputs for PREMUL_SUM reduction operations by applying the
    premultiplication factor and converting to a regular SUM operation.

    For non-PREMUL_SUM operations, inputs are returned unchanged with their
    corresponding operation type.
    """
    reduce_type = get_reduce_type(reduce_op)
    if reduce_type == "PREMUL_SUM":
        _, premul_factor = reduce_op.__getstate__()
        if isinstance(inputs, list):
            if isinstance(inputs[0], list):
                scaled_inputs = [
                    [tensor * premul_factor for tensor in inner_list] for inner_list in inputs
                ]
            else:
                scaled_inputs = [tensor * premul_factor for tensor in inputs]
        else:
            scaled_inputs = inputs * premul_factor

        return scaled_inputs, "SUM"

    return inputs, reduce_type
