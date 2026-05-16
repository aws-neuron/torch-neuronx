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
            "TensorParallelAPITestsWithLocalTensor::test_parallelize_module_multi_wildcard",
            "TensorParallelAPITestsWithLocalTensor::test_linear_row_wise_parallel",
            "TensorParallelAPITestsWithLocalTensor::test_parallelize_module_with_no_match",
            "TensorParallelAPITestsWithLocalTensor::test_parallelize_module_with_root_module",
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
        reason="Fake backend test triggers Neuron broadcast backend and fails",
        tests=[
            "TensorParallelRandomStateTestsWithLocalTensor::test_model_init",
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
        reason="ProcessGroupNeuron missing API: coalescing, sequence numbers, sort_ranks",
        tests=[
            "test_c10d_nccl.py::CommTest::test_all_gather_into_tensor_coalesced_manager_nccl",
            "test_c10d_nccl.py::CommTest::test_all_reduce_coalesced_manager_nccl",
            "test_c10d_nccl.py::CommTest::test_reduce_scatter_tensor_coalesced",
            "test_c10d_nccl.py::CommTest::test_sequence_num_incremented_nccl_default",
            "test_c10d_nccl.py::CommTest::test_sequence_num_set_default_pg_nccl",
            "test_c10d_nccl.py::CommTest::test_sequence_num_set_nccl_new_group",
            "test_c10d_nccl.py::LargeCommTest::test_new_group_ordered",
            "test_device_mesh.py::InitDeviceMeshTest::test_backend_override_argument_dict_with_idx_and_backend_eager",
        ],
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
            "test_c10d_ops_privateuse1.py::ProcessGroupPrivateUse1OpTest::test_empty_tensors",
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
            "test_state_dict.py::TestStateDict::test_non_persistent_buffers",
            "test_state_dict.py::TestStateDict::test_single_gpu",
            "test_state_dict.py::TestNoComm::test_no_dist",
            "test_state_dict_stager.py::TestDTensorStateDictStager::test_async_save_no_memory_leak",
        ],
    ),
    SkipGroup(
        reason="Neuron backend does not support variable-sized inputs to collectives",
        tests=[
            "test_c10d_ops_privateuse1.py::ProcsessGroupPrivateUse1OpTest::test_all_gather_v",
            "test_c10d_ops_privateuse1.py::ProcessGroupPrivateUse1OpTest::test_reduce_scatter_v",
        ],
    ),
    SkipGroup(
        reason="PRODUCT is not supported as an option to collectives on Neuron",
        tests=[
            "test_c10d_ops_privateuse1.py::ProcessGroupPrivateUse1OpTest::test_reduce_scatter_ops",
            "test_c10d_ops_privateuse1.py::ProcessGroupPrivateUse1OpTest::test_allreduce_ops",
        ],
    ),
    SkipGroup(
        reason=(
            "We are pinning CPU tensors to have pinned memory=True always whereas"
            "we should reference ptr_to_block_ for checking if tensor is pinned or not."
        ),
        tests=[
            "test_fsdp_optim_state.py::FsdpOptimStateCheckpoint::test_load_sharded_optimizer_state_dict_pass_planner_False",
            "test_fsdp_optim_state.py::FsdpOptimStateCheckpoint::test_load_sharded_optimizer_state_dict_pass_planner_True",
        ],
    ),
    SkipGroup(
        reason="Tests memory allocated for cuda",
        tests=[
            "test_state_dict.py::TestStateDict::test_setting_meta_device_model_broadcasting_and_memory"
        ],
    ),
    SkipGroup(
        reason=(
            "CommDebugMode does not count Neuron collective ops"
            " - Neuron collectives bypass CommDebugMode dispatch hooks"
        ),
        tests=[
            "TestFullyShardGradientAccumulation::test_gradient_accumulation",
            "TestFullyShardCollectiveOps::test_reduce_scatter_fp16",
            "TestFullyShardCollectiveOps::test_all_gather_fp32",
            "TestFullyShardCollectiveOps::test_reduce_scatter_fp32",
        ],
    ),
    SkipGroup(
        reason="Neuron memory_stats() does not provide active_bytes.all.peak/current keys",
        tests=[
            "TestFullyShardMemory::test_fully_shard_del_memory",
            "TestFullyShardMemory::test_fully_shard_training_memory",
            "TestFullyShardMemory::test_fully_shard_training_memory_no_gc",
        ],
    ),
    SkipGroup(
        reason="NRT execution error on all_gather with int32→int64 dtype cast",
        tests=[
            "TestFullyShardHSDP3DTraining::test_3d_mlp_with_nd_mesh",
        ],
    ),
    SkipGroup(
        reason="NRT Execution error on all_gather with halved world_size or orig_params",
        tests=[
            "TestFSDPOptimState::test_scatter_full_optim_state_dict_nested_halve_world_size",
            "TestFSDPOptimState::test_shard_full_optim_state_dict_nested_halve_world_size",
            "TestFSDPOptimState::test_use_orig_params",
        ],
    ),
    SkipGroup(
        reason="Timing assertion failures in overlap measurement",
        tests=[
            "TestFullyShardPerParamMeshOverlap::test_fully_shard_per_param_mesh_training_overlap",
            "TestFullyShardPerParamMeshOverlap::test_fully_shard_per_param_mesh_no_grad_input_overlap",
        ],
    ),
    SkipGroup(
        reason="Neuron lacks native batch_norm — CPU fallback rejects mixed "
        "precision (bf16/fp16 input, fp32 stats)",
        tests=[
            "TestFullyShardMixedPrecisionCasts::test_norm_modules_fp16",
            "TestFullyShardMixedPrecisionCasts::test_norm_modules_bf16",
            "TestReplicateMixedPrecisionCasts::test_norm_modules_fp16",
            "TestReplicateMixedPrecisionCasts::test_norm_modules_bf16",
        ],
    ),
    SkipGroup(
        reason="Neuron does not support fp16 linear — CPU fallback disabled",
        tests=[
            "TestFSDPMixedPrecisionIgnoredModules::test_mixed_precision_with_ignored_module",
            "TestFSDPDifferentSubmodulePrecision::test_float16_on_one_submodule_skip_inputs_error",
            "TestFSDPDifferentSubmodulePrecision::test_submodules_with_different_precisions_error",
        ],
    ),
    SkipGroup(
        reason="Neuron _to_copy fallback disabled for mixed precision optim state",
        tests=[
            "TestFSDPOptimState::test_scatter_full_optim_state_dict_transformer",
            "TestFSDPOptimState::test_shard_full_optim_state_dict_transformer",
        ],
    ),
    SkipGroup(
        reason="GradScaler found_inf assertion mismatch on Neuron",
        tests=[
            "TestShardedGradScalerParityWithDDP::test_sharded_grad_scaler_found_inf",
        ],
    ),
    SkipGroup(
        reason="Neuron allocator is not a CachingDeviceAllocator — memory_allocated() fails",
        tests=[
            "TestFSDPMemory::test_fsdp_memory_ckpt_ckpt",
            "TestFSDPMemory::test_fsdp_memory_ckpt_no_ckpt",
            "test_embedding_error_msg",
            "test_tensor_hash_redistribute",
        ],
    ),
    SkipGroup(
        reason=(
            "Neuron numerical precision slightly exceeds DDP vs FSDP grad norm"
            " tolerance (rel diff 1.46e-06 vs allowed 1.3e-06)"
        ),
        tests=[
            "TestClipGradNormNEURON::test_ddp_parity_neuron",
        ],
    ),
    SkipGroup(
        reason=(
            "SDPA backward NRT execution failure — invalid NEFF during"
            " all_reduce in _scaled_dot_product_fused_attention_overrideable_backward"
        ),
        tests=[
            "TestFSDPHybridShard::test_fsdp_hybrid_shard_parity",
        ],
    ),
    SkipGroup(
        reason="torch.distributed.record_comm API not supported",
        tests=[
            "RecordCommTest::test_nesting",
            "RecordCommTest::test_set_get",
        ],
    ),
    SkipGroup(
        reason=(
            "NRT all_gather fails with invalid DMA block size for the tensor "
            "shape used in FSDP fine tuning pretrain step"
        ),
        tests=[
            "TestFineTuning::test_fine_tuning",
        ],
    ),
    SkipGroup(
        reason="Neuron clip_grad_norm_ does not support ord=3.0, only 1, 2, and inf",
        tests=[
            "TestClipGradNormWorldSize4::test_clip_grad_norm_2d",
        ],
    ),
    SkipGroup(
        reason="Unwaited collective stress test exceeds default 600s timeout on Neuron",
        tests=[
            "CommTest::test_unwaited",
        ],
    ),
    SkipGroup(
        reason="torch.compile(fullgraph=True) defaults to inductor backend which "
        "cannot lower distributed collectives on Neuron",
        tests=[
            "TestFunctionalDifferentialsWithCompileNEURON::test_all_gather_tensor_compile_neuron",
            "TestFunctionalDifferentialsWithCompileNEURON::test_all_reduce_compile_neuron",
            "TestFunctionalDifferentialsWithCompileNEURON::test_all_to_all_single_compile_neuron",
            "TestFunctionalDifferentialsWithCompileNEURON::test_reduce_scatter_tensor_compile_neuron",
        ],
    ),
    SkipGroup(
        reason=(
            "Eager conv backward hang. Strided/downsampling conv backward dispatches"
            " an all_reduce on a non-contiguous gradient layout."
        ),
        tests=[
            "test_convolution_ops.py::DistConvolutionOpsTest::test_downsampling_convolution",
        ],
    ),
    SkipGroup(
        reason="New test does not have device agnostic changes yet",
        tests=[
            "test_fully_shard_memory.py::TestFullyShardHSDPSyncCorrectness::test_ar_buffer_lifetime_mixed_dtype",
            "test_c10d_functional_native.py::ACTCompileTest::test_act_compile_backward_tangent_mismatch",
        ],
    ),
    SkipGroup(
        reason=(
            "FSDP subprocess exits with non-zero status (exit code 1) without a"
            " captured traceback. Failure is consistent across runs but root"
            " cause is not surfaced through the test log."
        ),
        tests=[
            "test_fully_shard_training.py::TestFullyShardRegisteredParams::test_modules_to_run_forward_empty_for_single_module",
        ],
    ),
    SkipGroup(
        reason=(
            "Distributed subprocess exits with code 1 on Neuron. Test launches"
            " a subprocess that fails without a captured traceback in the log."
        ),
        tests=[
            "test_dtensor_ops.py::DistTensorCppPyTree::test_cpp_cat",
        ],
    ),
    SkipGroup(
        reason=(
            "TestLoadStateDict ref-cycle scalar mismatch — state dict reload"
            " surfaces float precision differences on Neuron."
        ),
        tests=[
            "test_load_state_dict.py::TestLoadStateDict::test_load_state_dict_ref_cycle_swap_False",
        ],
    ),
    SkipGroup(
        reason=(
            "Change needed in upstream PT 2.11 for code path outside"
            " torch/testing/_internal — BF16 compress comm hooks only"
            " allow CUDA/XPU devices."
        ),
        tests=[
            "test_c10d_nccl.py::DistributedDataParallelTest::test_bf16_compress_hook_wrapper",
            "test_c10d_nccl.py::DistributedDataParallelTest::test_bf16_compress_wrapper_is_view",
        ],
    ),
    SkipGroup(
        reason=(
            "C++ PG: BackendType.CUSTOM lacks device awareness. "
            "PyTorch hardcodes BackendType.CUSTOM for all extended_api backends "
            "(distributed_c10d.py:372). ProcessGroup::getBackendName() returns "
            "'custom' instead of delegating to NeuronBackend::getBackendName(). "
            "ProcessGroup::barrier() falls into the else branch (CPU tensor) "
            "because only NCCL->CUDA and XCCL->XPU are handled. "
            "Requires upstream fix: either allow CUSTOM backends to specify "
            "their device type, or make ProcessGroup delegate to the registered "
            "Backend for CUSTOM type."
        ),
        tests=[
            "test_backends.py::TestMiscCollectiveUtilsNEURON::test_create_pg_neuron",
            "test_c10d_ops_privateuse1.py::ProcessGroupPrivateUse1OpTest::test_barrier",
        ],
    ),
    SkipGroup(
        reason=(
            "DTensor + torch.compile: Neuron Dynamo backend fails to compile"
            " DTensor graphs (Torch-MLIR lowering errors and rank-not-set"
            " errors during AOT autograd). DTensor compile path requires"
            " distributed runtime fixtures and Neuron-specific lowering"
            " support that is not yet in place."
        ),
        tests=[
            "test_dtensor_compile.py::TestDTensorCompile::test_compile_redistribute_flattened_mesh",
            "test_dtensor_compile.py::TestDTensorCompile::test_dtensor_basic",
            "test_dtensor_compile.py::TestDTensorCompile::test_dtensor_dynamo_device_mesh_attrs",
            "test_dtensor_compile.py::TestDTensorCompile::test_dtensor_input_mutations",
            "test_dtensor_compile.py::TestDTensorCompile::test_dtensor_noncontiguous_output",
            "test_dtensor_compile.py::TestDTensorCompile::test_dynamo_dtensor",
            "test_dtensor_compile.py::TestDTensorCompile::test_dynamo_dtensor_from_local_redistribute",
            "test_dtensor_compile.py::TestDTensorCompile::test_dynamo_to_local_kwargs_forward_hook",
            "test_dtensor_compile.py::TestDTensorCompile::test_get_local_rank_compile",
        ],
    ),
    SkipGroup(
        reason=(
            "Distributed multi-process test fails with subprocess exit code 10."
            " Worker process exits during test setup/execution without a"
            " surfaced traceback in the parent log."
        ),
        tests=[
            "test_tensor_ops.py::DistArgMaxArgMinTest::test_argmax_argmin_with_placements",
            "test_redistribute.py::DistributeWithStridedShardTest::test_strided_shard_redistribution",
            "test_dtensor_resharding.py::TestDTensorReshardMeshChange::test_dtensor_checkpoint_resharding_with_empty_shard",
            "test_state_dict_stager.py::TestStateDictStager::test_tensor_pinned_and_shared",
            "test_fully_shard_comm.py::TestFullyShardCommunication::test_set_reduce_scatter_divide_factor",
            "test_fully_shard_training.py::TestFullyShardGradientAccumulation::test_1f1b_microbatching",
            "test_fully_shard_training.py::TestFullyShardSharedParams::test_train_parity_with_shared_params",
            "test_fully_shard_training.py::TestFullyShardShardPlacementFnMultiProcess::test_train_parity_shard_placement_fn_shard_largest_dim",
            "test_fully_shard_autograd.py::TestFullyShardPostAccGradHookMultiProcess::test_post_acc_grad_hook_optim_parity",
            "test_replicate_training.py::TestReplicateGradientAccumulation::test_1f1b_microbatching",
            "test_replicate_training.py::TestReplicateSharedParams::test_train_parity_with_shared_params",
        ],
    ),
]
