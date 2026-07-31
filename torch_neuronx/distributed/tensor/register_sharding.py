"""
This file contains sharding registration for aten ops. Most of these need to be upstreamed
to pytorch
"""

import torch
from torch.distributed.tensor._dtensor_spec import (
    DTensorSpec,
)
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpStrategy,
    PlacementList,
    RuntimeSchemaInfo,
)
from torch.distributed.tensor._ops import (
    expand_to_full_mesh_op_strategy,
    generate_redistribute_costs,
    infer_broadcast_dims_map,
    is_tensor_shardable,
    map_placements_after_broadcast,
    register_op_strategy,
)
from torch.distributed.tensor._ops._einsum_strategy import gen_einsum_strategies
from torch.distributed.tensor.placement_types import (
    Partial,
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


@register_op_strategy(aten.linear.default)
def linear_out_strategy(op_schema: OpSchema) -> OpStrategy:
    """
    Linear strategy following the einsum-based approach used in addmm.

    linear(input, weight, bias) computes: input @ weight.T + bias
    where weight has shape (out_features, in_features)
    """
    mesh = op_schema.get_mesh_from_args(validate=False)
    input_strategy, weight_strategy = op_schema.args_schema[:2]
    bias_strategy = op_schema.args_schema[2] if len(op_schema.args_schema) > 2 else None

    assert isinstance(input_strategy, OpStrategy)
    assert isinstance(weight_strategy, OpStrategy)

    input_shape = input_strategy.shape
    input_ndim = len(input_shape)

    # Generate einsum equation for input @ weight.T
    # Note: weight is (out_features, in_features), so we transpose it
    # Simple 2D case: mk,nk->mn
    # Batched case: ...mk,nk->...mn
    batch_dims = "".join(chr(ord("a") + j) for j in range(input_ndim - 1))
    mm_equation = f"{batch_dims}k,nk->{batch_dims}n"

    # Generate all possible mm strategies
    mm_strategy = gen_einsum_strategies(mm_equation, mesh)

    # Compute expected output shape from mm
    mm_out_shape = list(input_shape)
    mm_out_shape[-1] = weight_strategy.shape[0]  # out_features
    mm_out_shape = torch.Size(mm_out_shape)

    # Filter strategies and handle bias
    filtered_strategies = []
    for strtg in mm_strategy.strategies:
        assert strtg.input_specs is not None
        input_spec = strtg.input_specs[0]
        weight_spec = strtg.input_specs[1]
        out_spec = strtg.output_spec

        # Check if input and weight are shardable with these specs
        if not (
            is_tensor_shardable(input_strategy.shape, input_spec)
            and is_tensor_shardable(weight_strategy.shape, weight_spec)
        ):
            continue

        redistribute_cost = [
            generate_redistribute_costs(input_strategy, input_spec),
            generate_redistribute_costs(weight_strategy, weight_spec),
        ]
        # Handle bias if present
        if bias_strategy:
            assert isinstance(bias_strategy, OpStrategy)
            bias_shape = bias_strategy.shape

            # Bias is 1D with shape (out_features,)
            # Map its placement from the output's last dimension
            broadcast_dims_map = infer_broadcast_dims_map(mm_out_shape, bias_shape)
            bias_placements = map_placements_after_broadcast(
                out_spec.placements, mm_out_shape, broadcast_dims_map
            )
            bias_spec = DTensorSpec(mesh=mesh, placements=bias_placements)

            # Check if bias is shardable
            if not is_tensor_shardable(bias_shape, bias_spec):
                continue

            # Update input specs to include bias
            strtg.input_specs = (input_spec, weight_spec, bias_spec)

            # Calculate redistribute costs
            redistribute_cost.append(generate_redistribute_costs(bias_strategy, bias_spec))

        strtg.redistribute_cost = redistribute_cost
        filtered_strategies.append(strtg)

    mm_strategy.strategies = filtered_strategies
    return mm_strategy


@register_op_strategy(aten.linear_backward.default, schema_info=RuntimeSchemaInfo(3))
def linear_backward_strategy(op_schema: OpSchema) -> OpStrategy:
    """
    Sharding strategy for linear_backward.

    linear_backward(input[*,k], grad_output[*,n], weight[n,k], output_mask)
    -> (grad_input[*,k], grad_weight[n,k], grad_bias[n])

    Three computations:
      grad_input  = einsum("...j, ji -> ...i", grad_output, weight)   contracts n
      grad_weight = einsum("...j, ...i -> ji", grad_output, input)    contracts batch
      grad_bias   = einsum("...j -> j", grad_output)                  contracts batch

    Dimensions:
      batch dims (0..ndim-2): free in grad_input, contracting in grad_weight/grad_bias
      n (out_features):       contracting in grad_input, free in grad_weight/grad_bias
      k (in_features):        free in grad_input/grad_weight, not in grad_bias
    """
    mesh = op_schema.get_mesh_from_args(validate=False)
    input_strategy, grad_output_strategy, weight_strategy, output_mask = op_schema.args_schema

    assert isinstance(input_strategy, OpStrategy)
    assert isinstance(grad_output_strategy, OpStrategy)
    assert isinstance(weight_strategy, OpStrategy)

    input_ndim = len(input_strategy.shape)

    # Helper: mask output placements based on output_mask, align with meta registration
    def _mask(grad_input_p, grad_weight_p, grad_bias_p):
        return [
            grad_input_p if output_mask[0] else None,
            grad_weight_p if output_mask[1] or output_mask[2] else None,
            grad_bias_p if output_mask[1] or output_mask[2] else None,
        ]

    # PlacementList order: [grad_input, grad_weight, grad_bias, input, grad_output, weight]

    single_mesh_dim_strategies: list[PlacementList] = []

    # Strategy 0: All replicate
    single_mesh_dim_strategies.append(
        [*_mask(Replicate(), Replicate(), Replicate()), Replicate(), Replicate(), Replicate()]
    )

    # Strategy 1: Shard batch dim i (for each i in 0..ndim-2)
    # batch is free in grad_input, contracting in grad_weight and grad_bias
    for i in range(input_ndim - 1):
        single_mesh_dim_strategies.append(
            [*_mask(Shard(i), Partial(), Partial()), Shard(i), Shard(i), Replicate()]
        )

    # Strategy 2: Shard n (out_features)
    # n is contracting in grad_input, free (dim 0) in grad_weight, free (dim 0) in grad_bias
    single_mesh_dim_strategies.append(
        [*_mask(Partial(), Shard(0), Shard(0)), Replicate(), Shard(input_ndim - 1), Shard(0)]
    )

    # Strategy 3: Shard k (in_features)
    # k is free in grad_input (last dim) and grad_weight (dim 1), not in grad_bias
    single_mesh_dim_strategies.append(
        [
            *_mask(Shard(input_ndim - 1), Shard(1), Replicate()),
            Shard(input_ndim - 1),
            Replicate(),
            Shard(1),
        ]
    )

    # Linearity strategies for sum and avg reduce ops.
    # Linearity holds for sum and avg: f(a + b) = f(a) + f(b).
    # min/max are NOT linear and must be redistributed by the planner.
    for reduce_op in ("sum", "avg"):
        p = Partial(reduce_op)

        # Strategy 4: Linearity - grad_output Partial
        # grad_output is used linearly in all three computations
        single_mesh_dim_strategies.append([*_mask(p, p, p), Replicate(), p, Replicate()])

        # Strategy 5: Linearity - input Partial
        # input is used linearly only in grad_weight = grad_output^T @ input
        single_mesh_dim_strategies.append(
            [*_mask(Replicate(), p, Replicate()), p, Replicate(), Replicate()]
        )

        # Strategy 6: Linearity - weight Partial
        # weight is used linearly only in grad_input = grad_output @ weight
        single_mesh_dim_strategies.append(
            [*_mask(p, Replicate(), Replicate()), Replicate(), Replicate(), p]
        )

    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=3
    )
