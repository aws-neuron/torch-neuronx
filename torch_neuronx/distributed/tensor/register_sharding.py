"""
This file contains sharding registration for aten ops. Most of these need to be upstreamed
to pytorch
"""

import torch
from torch.distributed.tensor._dtensor_spec import (
    DTensorSpec,
    TensorMeta,
)
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpSpec,
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
from torch.distributed.tensor._ops._matrix_ops import (
    _mm_like_strategy,
)
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


def derive_bias_spec_from_grad_output(grad_output_spec, mesh):
    """
    grad_bias = grad_output.sum(0) - sum over batch dimensions
    grad_bias shape: [out_features]
    grad_bias placement: same as grad_output's last dimension
    """

    # grad_bias is 1D with shape [out_features]
    # out_features is the last dimension of grad_output
    out_features = grad_output_spec.shape[-1]

    # Create TensorMeta for 1D bias tensor
    bias_tensor_meta = TensorMeta(
        shape=torch.Size([out_features]), stride=(1,), dtype=grad_output_spec.tensor_meta.dtype
    )

    # grad_bias is 1D, so adjust sharding dimensions
    bias_placements = []
    for placement in grad_output_spec.placements:
        if isinstance(placement, Shard):
            # For bias (1D), any sharding becomes Shard(0)
            if placement.dim == -1 or placement.dim == len(grad_output_spec.shape) - 1:
                # grad_output sharded on last dim (features) -> bias sharded on dim 0
                bias_placements.append(Shard(0))
            else:
                # grad_output sharded on batch dims -> bias becomes replicated (sum reduces it)
                bias_placements.append(Replicate())
        else:
            # Replicate, Partial stay the same
            bias_placements.append(placement)

    return DTensorSpec(mesh=mesh, placements=tuple(bias_placements), tensor_meta=bias_tensor_meta)


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


@register_op_strategy(aten.linear_backward.default)
def linear_backward_strategy(op_schema: OpSchema) -> OpStrategy:
    mesh = op_schema.get_mesh_from_args(validate=False)
    input_strategy, grad_output_strategy, weight_strategy, output_mask = op_schema.args_schema

    input_ndim = len(input_strategy.shape)
    if input_ndim <= 2:  # if there're more than 1 batch dim, einsum strategy will break
        batch_dims = "".join(chr(ord("a") + j) for j in range(input_ndim - 1))
        grad_input_eq = f"{batch_dims}k,kn->{batch_dims}n"

        grad_input_schema = OpSchema(
            op=aten.mm.default,
            args_schema=(grad_output_strategy, weight_strategy),
            kwargs_schema={},
        )
        grad_input_mm_strategies = _mm_like_strategy(grad_input_eq, mesh, grad_input_schema)

    strategies = []

    # Iterate through each strategy combination
    num_strategies = len(input_strategy.strategies)

    for i in range(num_strategies):
        input_spec = input_strategy.strategies[i].output_spec
        grad_output_spec = grad_output_strategy.strategies[i].output_spec
        weight_spec = weight_strategy.strategies[i].output_spec

        output_specs = []

        # grad_input = grad_output @ weight.T - use mm_strategy
        if output_mask[0] and input_ndim <= 2:
            # Create single-strategy OpStrategy for this iteration
            grad_input_spec = None
            for mm_strtg in grad_input_mm_strategies.strategies:
                if (
                    mm_strtg.input_specs[0].placements == grad_output_spec.placements
                    and mm_strtg.input_specs[1].placements == weight_spec.placements
                ):
                    grad_input_spec = mm_strtg.output_spec
                    break
            assert grad_input_spec is not None
            output_specs.append(grad_input_spec)
        elif output_mask[0]:
            grad_input_placements = []
            grad_out_ndim = len(grad_output_spec.shape)

            for mesh_dim_idx in range(mesh.ndim):
                grad_out_placement = grad_output_spec.placements[mesh_dim_idx]
                weight_placement = weight_spec.placements[mesh_dim_idx]

                # Check if grad_output is sharded on last dim (output features)
                # and weight is sharded on dim 1 (columns, which become rows when transposed)
                if isinstance(grad_out_placement, Shard):
                    # Normalize dimension to positive index
                    grad_out_shard_dim = grad_out_placement.dim
                    if grad_out_shard_dim < 0:
                        grad_out_shard_dim = grad_out_ndim + grad_out_shard_dim

                    # If grad_output sharded on last dim AND weight sharded on columns
                    # Result is Partial (needs all_reduce)
                    if (
                        grad_out_shard_dim == grad_out_ndim - 1
                        and isinstance(weight_placement, Shard)
                        and weight_placement.dim == 0
                    ):
                        grad_input_placements.append(Partial())
                    else:
                        grad_input_placements.append(input_spec.placements[mesh_dim_idx])
                else:
                    # grad_output not sharded, match input's placement
                    grad_input_placements.append(input_spec.placements[mesh_dim_idx])

                grad_input_spec = DTensorSpec(
                    mesh=mesh,
                    placements=tuple(grad_input_placements),
                    tensor_meta=input_spec.tensor_meta,  # grad_input matches input dtype/shape meta
                )
            output_specs.append(grad_input_spec)
        else:
            output_specs.append(None)

        if output_mask[1] or output_mask[2]:  # same logic as in meta function
            # grad_weight = grad_output.T @ input
            grad_weight_spec = weight_spec  # Match original weight sharding
            output_specs.append(grad_weight_spec)
            # grad_bias = grad_output.sum(0)
            grad_bias_spec = derive_bias_spec_from_grad_output(grad_output_spec, mesh)
            output_specs.append(grad_bias_spec)
        else:
            output_specs.append(None)
            output_specs.append(None)

        strategies.append(
            OpSpec(
                output_specs=tuple(output_specs),
                input_specs=(input_spec, grad_output_spec, weight_spec),
                redistribute_cost=[[0.0], [0.0], [0.0]],
            )
        )

    return OpStrategy(strategies=strategies)
