import os

from tests.pytorch_tests_native.neuron_test_registry import SkipGroup

XFAIL_GROUPS: list[SkipGroup] = [
    SkipGroup(
        reason=(
            "[accuracy] check_sharded_parity compares the FSDP reduce_scatter grad/param "
            "path against an unsharded reference; fp32 reduce-order non-associativity "
            "(compounded over Adam steps) lands ~1e-5 off, just over the strict "
            "1e-5/1.3e-6 tolerance, so one rank's assertEqual fails and raises. That "
            "rank leaves the collective group, so surviving ranks hit a replica-group "
            "signature mismatch. Surfaced by on-device RNG drawing slightly different fp32 "
            "values, not caused by it. strict=False."
        ),
        tests=[
            "test_fully_shard_training.py::TestFullyShardGradientAccumulation::test_gradient_accumulation",
        ],
    ),
    SkipGroup(
        reason=(
            "Neuron SDPA doesn't support float64. TODO: Once resolved, xfails will be removed."
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
            "order than local cross_entropy, causing up "
            "to a ~2.7e-05 difference that exceeds the "
            "test's 1e-05 tolerance."
        ),
        tests=[
            "DistTensorParallelExampleTestWithLocalTensor::test_loss_parallel",
        ],
    ),
    SkipGroup(
        reason=(
            "Support for validating plain cross_entropy targets on multi-dimensional "
            "DTensor meshes needs to be added. Neuron currently converts the target "
            "to a fully replicated DTensor even when another mesh dimension shards "
            "the input batch."
        ),
        tests=[
            "DistTensorParallelExampleTest::test_loss_parallel_plain_tensor_target_rejected_on_multi_dim",
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
        reason="ProcessGroupNeuron missing API: coalescing, sequence numbers, sort_ranks",
        tests=[
            "test_c10d_nccl.py::CommTest::test_reduce_scatter_tensor_coalesced",
            "test_c10d_nccl.py::CommTest::test_sequence_num_incremented_nccl_default",
            "test_c10d_nccl.py::CommTest::test_sequence_num_set_default_pg_nccl",
            "test_c10d_nccl.py::CommTest::test_sequence_num_set_nccl_new_group",
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
        reason="Neuron lacks native batch_norm — CPU fallback rejects mixed "
        "precision (bf16/fp16 input, fp32 stats)",
        tests=[
            "TestFullyShardMixedPrecisionCasts::test_norm_modules_bf16",
            "TestReplicateMixedPrecisionCasts::test_norm_modules_fp16",
            "TestReplicateMixedPrecisionCasts::test_norm_modules_bf16",
        ],
    ),
    SkipGroup(
        reason=(
            "torch-neuronx lacks generator-aware PrivateUse1 support for aten::normal_. "
            "randn with an explicit Neuron generator reaches CPU fallback, which rejects "
            "the generator device before the test reaches torch.func.jvp."
        ),
        tests=[
            "test_fully_shard_mixed_precision.py::TestFullyShardMixedPrecisionJVP::test_no_warmup_jvp_reshard_after_forward_false",
        ],
    ),
    SkipGroup(
        reason=(
            "Each Neuron process can access only its rank-local assigned device. "
            "The test hardcodes device 0 on every rank, causing rank 1 to access "
            "a device outside its visible range."
        ),
        tests=[
            "test_fully_shard_training.py::TestFullyShardForwardInputs::test_root_no_forward_inputs",
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
            "NRT all_gather fails with invalid DMA block size for the tensor "
            "shape used in FSDP fine tuning pretrain step"
        ),
        tests=[
            "TestFineTuning::test_fine_tuning",
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
            "Torch-MLIR cannot legalize _c10d_functional_autograd compatibility "
            "collective operators retained in AOT forward graphs, even after making "
            "the test backend agnostic; xfail until fixed."
        ),
        tests=[
            "test_functional_differentials.py::TestFunctionalDifferentialsWithCompileNEURON::test_all_gather_into_tensor_autograd_compile_neuron",
            "test_functional_differentials.py::TestFunctionalDifferentialsWithCompileNEURON::test_all_to_all_single_autograd_compile_neuron",
            "test_functional_differentials.py::TestFunctionalDifferentialsWithCompileNEURON::test_reduce_scatter_tensor_autograd_compile_neuron",
        ],
    ),
    SkipGroup(
        reason="New test does not have device agnostic changes yet",
        tests=[
            "test_fully_shard_memory.py::TestFullyShardHSDPSyncCorrectness::test_ar_buffer_lifetime_mixed_dtype",
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
            "test_dtensor_compile.py::TestDTensorCompile::test_dynamo_dtensor_from_local_redistribute",
        ],
    ),
    SkipGroup(
        reason=(
            "Distributed multi-process test fails with subprocess exit code 10."
            " Worker process exits during test setup/execution without a"
            " surfaced traceback in the parent log."
        ),
        tests=[
            "test_redistribute.py::DistributeWithStridedShardTest::test_strided_shard_redistribution",
            "test_dtensor_resharding.py::TestDTensorReshardMeshChange::test_dtensor_checkpoint_resharding_with_empty_shard",
        ],
    ),
    SkipGroup(
        reason=(
            "Asserts handler count on _c10d_logger; pytest 9.1+ attaches its"
            " own capture handlers to non-propagating loggers, inflating the count."
        ),
        tests=[
            "test_utils.py::TestMedatadaIndex::test_dcp_logger",
        ],
    ),
    SkipGroup(
        reason=(
            "Under DTensor LocalTensorMode, on-device torch.randn draws per-rank-varying"
            " values that neuron's full_tensor() all-gather doesn't reconstruct coherently,"
            " so assertEqual crashes on a LocalIntNode missing its guard_int attribute."
            " It's the DTensor all-gather, not the RNG: it fails only for on-device per-rank"
            " data (a fixed or CPU-origin input passes), and an RNG-state fix didn't help."
            " Same LocalTensorMode class as the already-xfailed test_weight_tying,"
            " test_sequence_parallel_style, and test_model_init: a neuron DTensor-layer"
            " limitation, not the random ops. Real multi-process distributed is unaffected."
        ),
        tests=[
            "test_tensor_ops.py::DistBucketizeTest::test_bucketize_sharded_input",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT 2.14][unsupported_feature] ProcessGroupNeuron does not support"
            " process group splitting. Pipelining PerDirectionSchedule"
            " tests require backend splitting support."
        ),
        tests=[
            "test_schedule_multiproc.py::PerDirectionScheduleTest::test_creates_distinct_direction_groups",
            "test_schedule_multiproc.py::PerDirectionScheduleTest::test_grad_with_manual_per_direction_ScheduleClass0",
            "test_schedule_multiproc.py::PerDirectionScheduleTest::test_grad_with_manual_per_direction_ScheduleClass1",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT 2.14] New test_addmm_partial_redistribute test creates a CPU"
            " DeviceMesh which calls init_process_group without RANK env var"
            " set in the multi-process test harness."
        ),
        tests=[
            "test_op_strategy.py::TestCostModel::test_addmm_partial_redistribute",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT 2.14] Test uses CompileCounterWithBackend('inductor')"
            " which raises KeyError on Neuron. Test is Inductor-specific."
        ),
        tests=[
            "test_dtensor_compile.py::TestDTensorCompile::test_compile_waits_act_nested_in_dtensor_local_tensor",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT 2.14] Unbacked symbolic strides in DTensor backward"
            " cause Neuron compilation error. Compiler cannot handle"
            " dynamic stride expressions in the backward graph."
        ),
        tests=[
            "test_dtensor_compile.py::TestDTensorCompile::test_to_local_backward_unbacked_symbolic_stride",
        ],
    ),
]

if (
    os.environ.get("TORCH_NEURONX_ENABLE_STREAMS", "1") == "1"
    and os.environ.get("TORCH_NEURONX_ENABLE_HOST_CC", "0") == "0"
):
    XFAIL_GROUPS.append(
        SkipGroup(
            reason=(
                "FSDPv1 NO_SHARD CPU offload can release device parameter storage "
                "before compute finishes. An upstream record_stream fix is required; see "
                "https://github.com/aws-neuron/private-torch-neuronx/issues/1531."
            ),
            tests=[
                "test_fsdp_clip_grad_norm.py::TestClipGradNormNEURON::test_ddp_parity_neuron",
                "test_fsdp_use_orig_params.py::TestFSDPUseOrigParamsUnshardReshard::test_multiple_forward_offload_params_True",
                "test_fsdp_grad_acc.py::TestGradAcc::test_grad_acc_cpu_offload_use_orig_params_True",
                "test_fsdp_use_orig_params.py::TestFSDPUseOrigParamsMultipleParamGroups::test_diff_hyperparams_cpu_offload_sharding_strategy_str_no_shard",
            ],
        )
    )
