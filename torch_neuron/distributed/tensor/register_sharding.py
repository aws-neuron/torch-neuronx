"""
This file contains sharding registration for aten ops. Most of these need to be upstreamed
to pytorch
"""

import torch
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpStrategy,
    PlacementList,
    RuntimeSchemaInfo,
)
from torch.distributed.tensor._ops.utils import (
    expand_to_full_mesh_op_strategy,
    register_op_strategy,
)
from torch.distributed.tensor.placement_types import (
    Placement,
    Replicate,
    Shard,
)

aten = torch.ops.aten


@register_op_strategy(
    aten._scaled_dot_product_fused_attention_overrideable.default, schema_info=RuntimeSchemaInfo(4)
)
def scaled_dot_product_fused_attention_overrideable_strategy(op_schema: OpSchema) -> OpStrategy:
    """Sharding strategy for _scaled_dot_product_fused_attention_overrideable.

    Based on the CUDNN attention strategy but adapted for the overrideable signature:
    (query, key, value, attn_bias?, dropout_p, is_causal, return_debug_mask, scale?) ->
    (output, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset,
    debug_attn_mask)
    """
    mesh = op_schema.get_mesh_from_args()

    # Parse arguments based on the signature
    (
        query_strategy,  # query
        _,  # key
        _,  # value
        attn_bias_strategy,  # attn_bias (optional)
        *rest_args,  # dropout_p, is_causal, return_debug_mask, scale
    ) = op_schema.args_schema

    # Extract boolean flags from rest_args
    return_debug_mask = len(rest_args) >= 3 and rest_args[2]  # return_debug_mask
    has_attn_bias = attn_bias_strategy is not None

    debug_attn_mask_sharding: Placement | None = Replicate() if return_debug_mask else None

    assert isinstance(query_strategy, OpStrategy)

    single_mesh_dim_strategies = []

    # Strategy 1: Full replication
    # Outputs: [output, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset,
    #           debug_attn_mask]
    # Inputs: [q, k, v, attn_bias?]
    all_replicate: PlacementList = [
        Replicate(),  # output
        Replicate(),  # logsumexp
        None,  # cum_seq_q (scalar)
        None,  # cum_seq_k (scalar)
        None,  # max_q (scalar)
        None,  # max_k (scalar)
        Replicate(),  # philox_seed
        Replicate(),  # philox_offset
        debug_attn_mask_sharding,  # debug_attn_mask
        Replicate(),  # q
        Replicate(),  # k
        Replicate(),  # v
    ]
    if has_attn_bias:
        all_replicate.append(Replicate())  # attn_bias
    single_mesh_dim_strategies.append(all_replicate)

    # Strategy 2: Tensor parallelism (shard on num_heads dim)
    tp_sharding = Shard(1)  # num_heads dimension
    qkv_sharding = tp_sharding
    output_sharding = tp_sharding
    logsumexp_sharding = tp_sharding
    debug_attn_mask_sharding_tp = tp_sharding if return_debug_mask else None

    num_heads_dim_sharding: PlacementList = [
        output_sharding,  # output
        logsumexp_sharding,  # logsumexp
        None,  # cum_seq_q
        None,  # cum_seq_k
        None,  # max_q
        None,  # max_k
        Replicate(),  # philox_seed
        Replicate(),  # philox_offset
        debug_attn_mask_sharding_tp,  # debug_attn_mask
        qkv_sharding,  # q
        qkv_sharding,  # k
        qkv_sharding,  # v
    ]
    if has_attn_bias:
        num_heads_dim_sharding.append(tp_sharding)  # attn_bias
    single_mesh_dim_strategies.append(num_heads_dim_sharding)

    # Strategy 3: Batch parallelism (shard on batch dim)
    batch_sharding = Shard(0)
    logsumexp_sharding_batch = batch_sharding
    debug_attn_mask_sharding_batch = batch_sharding if return_debug_mask else None

    batch_dim_sharding: PlacementList = [
        batch_sharding,  # output
        logsumexp_sharding_batch,  # logsumexp
        None,  # cum_seq_q
        None,  # cum_seq_k
        None,  # max_q
        None,  # max_k
        Replicate(),  # philox_seed
        Replicate(),  # philox_offset
        debug_attn_mask_sharding_batch,  # debug_attn_mask
        batch_sharding,  # q
        batch_sharding,  # k
        batch_sharding,  # v
    ]
    if has_attn_bias:
        batch_dim_sharding.append(batch_sharding)  # attn_bias
    single_mesh_dim_strategies.append(batch_dim_sharding)

    # Strategy 4: Context parallelism (shard on sequence dim)
    cp_sharding = Shard(2)  # sequence dimension
    logsumexp_sharding_cp = cp_sharding
    debug_attn_mask_sharding_cp = cp_sharding if return_debug_mask else None

    context_parallel_sharding: PlacementList = [
        cp_sharding,  # output
        logsumexp_sharding_cp,  # logsumexp
        None,  # cum_seq_q
        None,  # cum_seq_k
        None,  # max_q
        None,  # max_k
        Replicate(),  # philox_seed
        Replicate(),  # philox_offset
        debug_attn_mask_sharding_cp,  # debug_attn_mask
        cp_sharding,  # q
        cp_sharding,  # k
        cp_sharding,  # v
    ]
    if has_attn_bias:
        context_parallel_sharding.append(cp_sharding)  # attn_bias
    single_mesh_dim_strategies.append(context_parallel_sharding)

    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=9
    )


@register_op_strategy(aten._scaled_dot_product_fused_attention_overrideable_backward.default)
def scaled_dot_product_fused_attention_overrideable_backward_strategy(
    op_schema: OpSchema,
) -> OpStrategy:
    """Sharding strategy for _scaled_dot_product_fused_attention_overrideable_backward.

    Backward signature:
    (grad_out, query, key, value, attn_bias, grad_input_mask, out, logsumexp,
     cum_seq_q, cum_seq_k, max_q, max_k, dropout_p, is_causal, philox_seed, philox_offset,
    scale?) -> (grad_query, grad_key, grad_value, grad_attn_bias)
    """
    # Backward op doesn't need to validate the mesh since forward already did
    mesh = op_schema.get_mesh_from_args(validate=False)

    # Get the query strategy (input 1) to determine sharding patterns
    query_strategy = op_schema.args_schema[1]
    assert isinstance(query_strategy, OpStrategy)

    # Check if attn_bias exists (input 4)
    has_attn_bias = op_schema.args_schema[4] is not None

    single_mesh_dim_strategies = []

    # Strategy 1: Full replication
    # Outputs: [grad_query, grad_key, grad_value, grad_attn_bias]
    all_replicate_out: PlacementList = [
        Replicate(),  # grad_query
        Replicate(),  # grad_key
        Replicate(),  # grad_value
        Replicate() if has_attn_bias else None,  # grad_attn_bias
    ]

    # Inputs: Following the signature order
    all_replicate_inp: PlacementList = [
        Replicate(),  # grad_out
        Replicate(),  # query
        Replicate(),  # key
        Replicate(),  # value
        Replicate() if has_attn_bias else None,  # attn_bias
        None,  # grad_input_mask (bool[4])
        Replicate(),  # out
        Replicate(),  # logsumexp
        None,  # cum_seq_q (tensor)
        None,  # cum_seq_k (tensor)
        None,  # max_q (SymInt)
        None,  # max_k (SymInt)
        None,  # dropout_p (float)
        None,  # is_causal (bool)
        Replicate(),  # philox_seed (tensor)
        Replicate(),  # philox_offset (tensor)
    ]

    # Add scale if present (optional)
    if len(op_schema.args_schema) > 16:
        all_replicate_inp.append(None)  # scale (float?)

    all_replicate: PlacementList = all_replicate_out + all_replicate_inp
    single_mesh_dim_strategies.append(all_replicate)

    # Strategy 2: Tensor parallelism (shard on num_heads dim)
    tp_sharding = Shard(1)

    tp_out: PlacementList = [
        tp_sharding,  # grad_query
        tp_sharding,  # grad_key
        tp_sharding,  # grad_value
        tp_sharding if has_attn_bias else None,  # grad_attn_bias
    ]

    tp_inp: PlacementList = [
        tp_sharding,  # grad_out
        tp_sharding,  # query
        tp_sharding,  # key
        tp_sharding,  # value
        tp_sharding if has_attn_bias else None,  # attn_bias
        None,  # grad_input_mask
        tp_sharding,  # out
        tp_sharding,  # logsumexp
        None,  # cum_seq_q (replicate auxiliary tensors)
        None,  # cum_seq_k
        None,  # max_q
        None,  # max_k
        None,  # dropout_p
        None,  # is_causal
        Replicate(),  # philox_seed
        Replicate(),  # philox_offset
    ]

    if len(op_schema.args_schema) > 16:
        tp_inp.append(None)  # scale

    tp_strategy: PlacementList = tp_out + tp_inp
    single_mesh_dim_strategies.append(tp_strategy)

    # Strategy 3: Batch parallelism (shard on batch dim)
    batch_sharding = Shard(0)

    batch_out: PlacementList = [
        batch_sharding,  # grad_query
        batch_sharding,  # grad_key
        batch_sharding,  # grad_value
        batch_sharding if has_attn_bias else None,  # grad_attn_bias
    ]

    batch_inp: PlacementList = [
        batch_sharding,  # grad_out
        batch_sharding,  # query
        batch_sharding,  # key
        batch_sharding,  # value
        batch_sharding if has_attn_bias else None,  # attn_bias
        None,  # grad_input_mask
        batch_sharding,  # out
        batch_sharding,  # logsumexp
        None,  # cum_seq_q
        None,  # cum_seq_k
        None,  # max_q
        None,  # max_k
        None,  # dropout_p
        None,  # is_causal
        Replicate(),  # philox_seed
        Replicate(),  # philox_offset
    ]

    if len(op_schema.args_schema) > 16:
        batch_inp.append(None)  # scale

    batch_strategy: PlacementList = batch_out + batch_inp
    single_mesh_dim_strategies.append(batch_strategy)

    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=4
    )
