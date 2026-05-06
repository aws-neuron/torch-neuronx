"""DTensor skip lists — extracted from dtensor_skips_registry.py for readability."""

DTENSOR_COMPLEX_DTYPE_UNSUPPORTED = [
    "test_complex_view_ops",
    "test_conj_complex_dtensor",
    "test_partial_to_replicate_forward_backward_complex64",
    "test_partial_to_shard_complex64",
    "test_redistribute_shard_dim_change_complex64",
    "test_replicate_to_local_partial_grad_complex64",
    "test_rotary_embedding_complex_ops",
    "test_shard_dim_alltoall_complex64",
    "test_shard_to_replicate_forward_backward_complex64",
    "test_linalg_ops",
]

DTENSOR_ALLTOALL_WORLD_SIZE_2_UNSUPPORTED = [
    "test_force_min_cost_redistribution_plan",
    "test_index_put_tensor",
    "test_local_map_redistribute",
    "test_ordered_distribute_all_combination",
    "test_ordered_redistribute",
    "test_ordered_redistribute_with_partial",
    "test_redistribute_shard_dim_change_float32",
    "test_redistribute_shard_dim_multi_dim_mesh",
    "test_redistribute_to_partial",
    "test_shard_dim_alltoall_float32",
    "test_shard_order_same_data_as_strided_shard",
]
