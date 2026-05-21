from tests.pytorch_tests_native.neuron_test_registry import SkipGroup
from tests.pytorch_tests_native.skip_registry import xfail_tests as x

XFAIL_GROUPS: list[SkipGroup] = [
    SkipGroup(
        reason=(
            "For already-executed-but-not-yet-signaled, workwatchdog still sees"
            " isCompleted()=false, which gets aborted either by the per-op timeout"
            " (120s in MultiProcContinuousTest) or by the shutdown cleanup path"
            " terminate_=true aborts all remaining work_list_ entries."
        ),
        tests=[
            "test_StridedShard_to_replicate",
            "test_replicate_to_StridedShard",
            "test_replicate_to_shard_forward_backward",
        ],
    ),
    SkipGroup(
        reason=(
            "Fails because it does an all_reduce with PRODUCT"
            " reduce operation. Neuron only supports"
            " SUM, AVG, MAX, MIN — not PRODUCT"
        ),
        tests=[
            "test_split_on_partial",
            "test_math_ops.py::DistMathOpsTest::test_linear_op_reductions",
        ],
    ),
    SkipGroup(
        reason=(
            "NRT Execution error on reduce_scatter: Invalid"
            " NEFF, instruction, or input — non-contiguous"
            " replica groups in DistributeWithDeviceOrderTest"
        ),
        tests=[
            "test_debug_mode_shows_optimized_trace",
        ],
    ),
    SkipGroup(
        reason=(
            "Debug mode asserts 'c10d::_allgather_base_' in debug string,"
            " but Neuron dispatches through _c10d_functional::all_gather_into_tensor"
            " (functional collectives path) so the op name doesn't match"
        ),
        tests=[
            "test_allgather_base",
            "test_allgather_base_async_op",
        ],
    ),
    SkipGroup(
        reason=(
            "torch.compile traces optimizer step and incorrectly reuses the param's"
            " 2D symbolic context for the grad's 1D view base, causing"
            " AssertionError: 2 != 1. Upstream pytorch bug, should be fixed with latest sync"
        ),
        tests=[
            "test_compile_optimizer_grad_view_base_dim_mismatch",
        ],
    ),
    SkipGroup(
        reason=(
            "No backend type associated with device type neuron,"
            " passes when we set backend as neuron:fake"
        ),
        tests=x.DTENSOR_BACKEND_TYPE_MISSING,
    ),
    SkipGroup(
        reason=(
            "RANK env var not set: all 16 tests pass when DTensorOpTestBase inherits"
            " from MultiThreadedTestCase instead of DTensorTestBase"
        ),
        tests=x.DTENSOR_RANK_ENV_NOT_SET,
    ),
    SkipGroup(
        reason=(
            "DTensor random op on non-CUDA device enters"
            " _distribute_region path, creating real tensor"
            " inside FakeTensorMode. Fixable with adding"
            " dipatch path for privateuse1 backend,"
            " modifying upstream code."
        ),
        tests=[
            "test_compiled_dtensor_rng_op_graph_consistency",
        ],
    ),
    SkipGroup(
        reason="channels_last memory format backward fails on Neuron",
        tests=[
            "test_backward_channels_last",
        ],
    ),
    SkipGroup(
        reason="AssertionError: Tensor-likes are not close!",
        tests=[
            "test_dtensor_op_db_logit_neuron_float32",
        ],
    ),
    SkipGroup(
        reason="TP compile fullgraph backward failures",
        tests=[
            "test_tp_compile_fullgraph_is_seq_parallel_False_use_ca_True",
            "test_tp_compile_fullgraph_is_seq_parallel_True_use_ca_True",
        ],
    ),
    SkipGroup(
        reason="Scalars not equal - comm count mismatch",
        tests=[
            "test_distribute_tensor_rank",
        ],
    ),
    SkipGroup(
        reason="Dynamo redistribute backward - TorchRuntimeError with fake tensors",
        tests=[
            "test_compile_dtensor_redistribute_backward_use_ca_True",
        ],
    ),
    SkipGroup(
        reason="ReduceScatter with empty tensors not supported",
        tests=[
            "test_addmm_empty_operand",
            "test_compute_local_shape_and_global_offset_1D",
            "test_compute_local_shape_and_global_offset_2D",
        ],
    ),
    SkipGroup(
        reason=(
            "Test's subprocess spawned by _attempt_load_from_subprocess sees"
            " only 1 NeuronCore (device_count()=1), so torch.load fails when"
            " deserializing a tensor saved from rank≥1 (e.g. neuron:1) because"
            " _validate_device rejects device indices ≥ device_count"
        ),
        tests=[
            "test_dtensor_save_load_import",
        ],
    ),
    SkipGroup(
        reason="Layer norm backward failures",
        tests=[
            "test_layer_norm_bwd_req_grad",
        ],
    ),
    SkipGroup(
        reason="Device index out of range",
        tests=[
            "test_ring_attention_sdpa",
        ],
    ),
    SkipGroup(
        reason="Non-scalar tensor kwarg lr is on cpu device, expected neuron",
        tests=[
            "test_adam_1d_sharding",
            "test_adamw_1d_sharding",
            "test_admaw_fused_across_meshes",
        ],
    ),
    SkipGroup(
        reason="Operator aten.grid_sampler_3d.default no sharding strategy",
        tests=[
            "test_dtensor_op_db_grid_sampler_3d_neuron_float32",
        ],
    ),
    SkipGroup(
        reason=(
            "NRT reduce_scatter fails with 'Invalid NEFF, instruction, or input' in"
            " StridedShard._split_tensor → torch.cat path: non-contiguous replica groups"
            " created by StridedShard placement are unsupported by Neuron collective impl"
        ),
        tests=[
            "test_strided_shard_to_shard_redistribution",
        ],
    ),
    SkipGroup(
        reason=(
            "NRT all_reduce fails with 'Invalid NEFF' during partial→strided_shard"
            " redistribute: partial sum reduction produces incorrect values"
            " (non-zero actual vs zero expected) before strided sharding is applied"
        ),
        tests=[
            "test_partial_to_strided_shard_redistribution",
        ],
    ),
    SkipGroup(
        reason=(
            "Multi-dim mesh redistribute returns values"
            " 4x too small (actual/expected = 1/world_size²):"
            " reduction across 2D mesh dimensions is applied only once instead of once per dim."
            " NRT all_reduce also fails with 'Invalid NEFF' on the resulting process groups"
        ),
        tests=[
            "test_multi_dim_redistribute",
        ],
    ),
    SkipGroup(
        reason=(
            "Flattened reduction merging returns 1.0 instead of 4.0 (world_size²):"
            " Partial→Replicate all_reduce sum is not executed when reductions are merged"
            " across multiple mesh dims. NRT all_reduce fails with 'Invalid NEFF' on"
            " the non-contiguous process groups created by the flattened reduction path"
        ),
        tests=[
            "test_merging_reductions",
        ],
    ),
    SkipGroup(
        reason=(
            "torch.compile traces FSDP forward into Neuron's Python _allgather_base"
            " method. Dynamo fails tracing dist.get_process_group_ranks(self)"
        ),
        tests=[
            "test_2d_fsdp_tp_ac_compile_use_ca_True",
            "test_2d_fsdp_tp_compile_use_ca_False",
            "test_2d_fsdp_tp_compile_use_ca_True",
        ],
    ),
    SkipGroup(
        reason=(
            "_resolve_process_group() expects string but gets ProcessGroupNeuron"
            " object in torch.compile path. Fixable with _resolve_group handling"
            " ProcessGroupNeuron in functional_collectives.py"
        ),
        tests=[
            "test_dtensor_processgroup_backward",
            "test_dtensor_processgroup_deduplication",
            "test_dtensor_processgroup_extraction",
        ],
    ),
    SkipGroup(
        reason=(
            "Neuron's per-device stream pool is lazily initialized during eager"
            " tensor ops, but the compile async-collective tangent path goes"
            " straight from AOTAutograd's FX graph to NEFF execution setup and"
            " schedules ops before the pool exists, producing"
            " 'Stream pool not initialized for device 0'."
        ),
        tests=[
            "test_dtensor_compile.py::TestDTensorCompile::test_unwrap_async_collective_tensor_tangent",
        ],
    ),
    SkipGroup(
        reason=(
            "Neuron's allreduce returns the output with a preserved dim of size 1,"
            " while upstream _c10d_functional.all_reduce squeezes to scalar. During"
            " output_tp.sum().backward(), DTensor's OpDispatcher.wrap asserts the"
            " gradient is scalar and raises 'output tensor should be scalar!' on all ranks."
        ),
        tests=[
            "test_tp_examples.py::DistTensorParallelExampleTest::test_transformer_req_grad_seq_parallel_float32_thaw_layers_0_attention_wv__layers_0_feed_forward_w1__layers_1_feed_forward_w2__layers_1_ffn_norm__output__tok_embeddings",
        ],
    ),
    SkipGroup(
        reason=(
            "Specific to torch 2.13's LocalTensor simulation path. A 4-process"
            " repro without LocalTensor gives bit-identical dropout masks"
            " across ranks, confirming OffsetBasedRNGTracker sync works. Under"
            " LocalTensor, 1 of 4 simulated ranks diverges at 1-2 of 16"
            " elements because its per-rank dispatch fan-out doesn't keep"
            " neuron RNG state consistent across the per-rank tensor dict."
        ),
        tests=[
            "test_random_ops.py::DistTensorRandomOpTest::test_deterministic_dropout_1d",
        ],
    ),
    SkipGroup(
        reason=(
            "When PYTORCH_TESTING_PREFER_MULTIPROCESS=1 (set by neuron conftest),"
            " DistElementwiseOpsTest uses the multiprocess variant of DTensorOpTestBase."
            " In a recent nightly refactor, assertEqualOnRank was moved from"
            " DTensorTestMixin to MultiThreadedTestCase. Since the multiprocess path"
            " doesn't inherit from MultiThreadedTestCase, assertEqualOnRank is missing."
        ),
        tests=[
            "test_pointwise_ops.py::DistElementwiseOpsTest::test_activations",
            "test_pointwise_ops.py::DistElementwiseOpsTest::test_dropout_backward",
        ],
    ),
    SkipGroup(
        reason=(
            "Test body passes (SUBPASSED logged). xdist's pytest_runtest_logreport"
            " then tries to pickle a Shard placement object and fails."
        ),
        tests=[
            "test_single_dim_strategy.py::TestDijkstraExpandSingleDimStrategy::test_pq_vs_full_expansion_data_driven",
        ],
    ),
]
