from tests.pytorch_tests_native.neuron_test_registry import SkipGroup

XFAIL_GROUPS: list[SkipGroup] = [
    SkipGroup(
        reason=(
            "Neuron SDPA doesn't support float64. " "TODO: Once resolved, xfails will be removed."
        ),
        tests=[
            "DistTensorParallelExampleTest::test_transformer_req_grad_float64_thaw_all",
            "DistTensorParallelExampleTest::test_transformer_training_is_seq_parallel_False_float64",
            "DistTensorParallelExampleTest::test_transformer_training_is_seq_parallel_True_float64",
            "DistTensorParallelExampleTestWithLocalTensor::test_transformer_req_grad_float64_thaw_all",
            "DistTensorParallelExampleTestWithLocalTensor::test_transformer_training_is_seq_parallel_False_float64",
            "DistTensorParallelExampleTestWithLocalTensor::test_transformer_training_is_seq_parallel_True_float64",
        ],
    ),
    SkipGroup(
        reason=(
            "ProcessGroupNeuron inherits from ProcessGroup instead of C10dBackend, "
            "preventing composite backend registration (CPU + Neuron). Async "
            "checkpoint save requires two backends. ProcessGroupNeuron should "
            "subclass C10dBackend."
        ),
        tests=[
            "test_state_dict_stager.py::TestDTensorStateDictStager::test_dtensor",
            "test_state_dict_stager.py::TestReplicationStager::test_replication_dtensors",
            "test_state_dict_stager.py::TestReplicationStager::test_replication_persistence",
            "test_state_dict_stager.py::TestReplicationStager::test_replication_sharded_tensors",
            "test_e2e_save_and_load.py::TestE2ESaveAndLoad::test_e2e_async_cached_cache_staged_state_dict_False_async_checkpointer_type0_zoc_False",
            "test_e2e_save_and_load.py::TestE2ESaveAndLoad::test_e2e_async_cached_cache_staged_state_dict_False_async_checkpointer_type2_zoc_False",
            "test_e2e_save_and_load.py::TestE2ESaveAndLoad::test_e2e_async_cached_cache_staged_state_dict_False_async_checkpointer_type4_zoc_True",
            "test_e2e_save_and_load.py::TestE2ESaveAndLoad::test_e2e_async_cached_cache_staged_state_dict_False_async_checkpointer_type5_zoc_True",
            "test_e2e_save_and_load.py::TestE2ESaveAndLoad::test_e2e_async_cached_cache_staged_state_dict_True_async_checkpointer_type1_zoc_False",
            "test_e2e_save_and_load.py::TestE2ESaveAndLoad::test_e2e_async_cached_cache_staged_state_dict_True_async_checkpointer_type3_zoc_False",
        ],
    ),
    SkipGroup(
        reason=(
            "Distributed cross_entropy with all-reduce "
            "accumulates float32 values in a different "
            "order than local cross_entropy, causing a "
            "~1.7e-05 difference that exceeds the "
            "test's 1e-05 tolerance."
        ),
        tests=[
            "DistTensorParallelExampleTest::test_loss_parallel",
            "DistTensorParallelExampleTestWithLocalTensor::test_loss_parallel",
        ],
    ),
    SkipGroup(
        reason=(
            "Upstream PyTorch tests only support world_size <= 8. "
            "Running with world_size=64 causes failures. "
            "TODO: Remove from xfail after upstreaming the PyTorch test fix. "
            "[Not a test setup issue]"
        ),
        tests=[
            "TensorParallelAPITests::test_parallelize_module_multi_wildcard",
            "TensorParallelAPITests::test_parallelize_module_with_no_match",
            "TensorParallelAPITests::test_parallelize_module_with_root_module",
            "TensorParallelAPITestsWithLocalTensor::test_parallelize_module_multi_wildcard",
            "TensorParallelAPITests::test_linear_row_wise_parallel",
            "TensorParallelAPITestsWithLocalTensor::test_linear_row_wise_parallel",
            "TensorParallelAPITestsWithLocalTensor::test_parallelize_module_with_no_match",
            "TensorParallelAPITestsWithLocalTensor::test_parallelize_module_with_root_module",
        ],
    ),
    SkipGroup(
        reason=(
            "CommDebugMode does not detect Neuron "
            "collective ops because they bypass the "
            "dispatch path that CommDebugMode hooks into"
        ),
        tests=[
            "TensorParallelAPITests::test_parallelize_module_src_data_rank",
        ],
    ),
    SkipGroup(
        reason=(
            "Fake backend test fails because LocalTensor "
            "returns a LocalIntNode, which does not have "
            "guard_int"
        ),
        tests=[
            "DistTensorParallelExampleTestWithLocalTensor::test_weight_tying",
        ],
    ),
    SkipGroup(
        reason="Fake backend test fails; needs handling for LocalTensor philox offset",
        tests=[
            "TensorParallelStyleTestWithLocalTensor::test_sequence_parallel_style",
        ],
    ),
    SkipGroup(
        reason="Neuron RNG tracker doesn't differentiate per-rank",
        tests=[
            "TensorParallelRandomStateTests::test_model_init",
        ],
    ),
    SkipGroup(
        reason="Fake backend test triggers Neuron broadcast backend and fails",
        tests=[
            "TensorParallelRandomStateTestsWithLocalTensor::test_model_init",
        ],
    ),
    SkipGroup(
        reason="pg._register_onCompletion_hook() is not supported on Neuron",
        tests=[
            "test_c10d_nccl.py::WorkHookTest",
        ],
    ),
    SkipGroup(
        reason=(
            "BatchNorm buffer sync in DDP multi-forward does inplace op corrupting autograd graph."
        ),
        tests=[
            "test_c10d_nccl.py::DistributedDataParallelTest::test_ddp_buffer_sync_multi_forward_with_batchnorm",
        ],
    ),
    SkipGroup(
        reason="ProcessGroupNeuron does not support _start_coalescing",
        tests=[
            "test_c10d_nccl.py::CommTest::test_all_gather_into_tensor_coalesced_manager_nccl",
            "test_c10d_nccl.py::CommTest::test_all_reduce_coalesced_manager_nccl",
        ],
    ),
    SkipGroup(
        reason="AttributeError: ProcessGroupNeuron object has no attribute supports_splitting",
        tests=[
            "test_device_mesh.py::DeviceMeshTest::test_2d_mesh_eager_init_subgroup",
            "test_device_mesh.py::InitDeviceMeshTest::test_backend_override_argument_dict_with_idx_and_backend_eager",
            "test_device_mesh.py::TestDeviceMeshGetItem::test_flatten_mesh_4d",
        ],
    ),
    SkipGroup(
        reason="torch.distributed.coalescing_manager is not supported with Neuron backend",
        tests=[
            "test_c10d_nccl.py::CommTest::test_reduce_scatter_tensor_coalesced",
        ],
    ),
    SkipGroup(
        reason="ProcessGroupNeuron does not support sequence numbers",
        tests=[
            "test_c10d_nccl.py::CommTest::test_sequence_num_incremented_nccl_default",
            "test_c10d_nccl.py::CommTest::test_sequence_num_set_default_pg_nccl",
            "test_c10d_nccl.py::CommTest::test_sequence_num_set_nccl_new_group",
        ],
    ),
    SkipGroup(
        reason="ProcessGroupNeuron does not support sort_ranks=False",
        tests=["test_c10d_nccl.py::LargeCommTest::test_new_group_ordered"],
    ),
    SkipGroup(
        reason="Barrier not supported for ProcessGroupNeuron for non-default group",
        tests=[
            "test_c10d_nccl.py::CommTest::test_nccl_warn_not_in_group_debug_info",
            "test_c10d_nccl.py::CommTest::test_nccl_warn_not_in_group_debug_off",
            "test_c10d_nccl.py::CommTest::test_nccl_warn_not_in_group_debug_detail",
            "test_utils.py::TestDistWrapper::test_barrier",
        ],
    ),
    SkipGroup(
        reason=(
            "Neuron tensors with pinned_memory=False report pinned_memory=True "
            "after CPU conversion due to device hook."
        ),
        tests=[
            "test_state_dict_utils.py::TestStateDictUtils::test_create_cpu_state_dict",
        ],
    ),
    SkipGroup(
        reason=("Neuron collectives do not support empty tensor handling."),
        tests=[
            "test_dtensor_resharding.py::TestDTensorReshardMeshChange::test_dtensor_checkpoint_resharding_with_empty_shard",
            "test_hf_safetensor_e2e.py::TestDTensorReshardMeshChange::test_dtensor_checkpoint_resharding_with_empty_shard",
        ],
    ),
    SkipGroup(
        reason=(
            "Fails because it defaults to inductor for compile. After modifying, "
            "FSDP + torch.compile(backend='neuron') works in single-process but "
            "fails in multi-process spawned workers. The spawned process loses the "
            "module wrapper during compilation, resulting in a plain function "
            "instead of an OptimizedModule."
        ),
        tests=[
            "test_e2e_save_and_load.py::TestE2ESaveAndLoad::test_e2e_compile_True_model_type0",
            "test_e2e_save_and_load.py::TestE2ESaveAndLoad::test_e2e_compile_True_model_type1",
            "test_e2e_save_and_load.py::TestE2ESaveAndLoad::test_e2e_compile_True_model_type2",
        ],
    ),
    SkipGroup(
        reason=(
            "Each worker tries to access all 64 cores, causing the tests below "
            "to fail. Each worker needs visibility to only a few cores."
        ),
        tests=[
            "test_state_dict.py::TestStateDict::test_extra_state",
            "test_state_dict.py::TestStateDict::test_flattened_osd",
            "test_state_dict.py::TestStateDict::test_non_persistent_buffers",
            "test_state_dict.py::TestStateDict::test_single_gpu",
            "test_state_dict.py::TestNoComm::test_no_dist",
            "test_state_dict_stager.py::TestDTensorStateDictStager::test_async_save_no_memory_leak",
        ],
    ),
    SkipGroup(
        reason=(
            "Upstream FSDP does not limit the number of devices per process, "
            "causing each worker to claim all 64 cores instead of a subset. "
            "Will be enabled/fixed as part of FSDP onboarding."
        ),
        tests=[
            "test_fsdp_dsd.py::TestFullyShardWithDistributedStateDict::test_1d_fsdp_cpu_offload_full_model_state_dict",
            "test_fsdp_dsd.py::TestFullyShardWithDistributedStateDict::test_1d_fsdp_get_model_state_dict",
            "test_fsdp_dsd.py::TestFullyShardWithDistributedStateDict::test_save_with_fsdp1_and_load_with_fsdp2",
            "test_fsdp_dsd.py::TestFullyShardWithDistributedStateDict::test_save_with_fsdp1_and_load_with_fsdp2_tp",
            "test_fsdp_dsd.py::TestFullyShardWithDistributedStateDict::test_save_with_fsdp2_tp_and_load_with_tp",
            "test_fsdp_dsd.py::TestFullyShardWithDistributedStateDict::test_save_with_tp_and_load_with_fsdp2_tp",
        ],
    ),
]
