from tests.pytorch_tests_native.neuron_test_registry import SkipGroup

SKIP_GROUPS: list[SkipGroup] = [
    SkipGroup(
        reason=(
            "FSDP v1 tests that time out (>600s) on Neuron and fail with"
            " 'Expected exit code 0 but got 1' or 'Scalars are not equal'."
            " Likely collective deadlocks during optim state dict operations."
        ),
        tests=[
            "test_fsdp_comm_hooks.py::TestCommunicationHooks::test_fp16_hook_has_wrapping_True_sharding_strategy1",
            "test_fsdp_core.py::TestHooksNEURON::test_register_functions_called_cuda_first_True_mixed_precision_False_neuron",
            "test_fsdp_optim_state.py::TestFSDPOptimState::test_optim_state_dict_nested_state_dict_type0_use_multiple_param_groups_True_rank0_only_False_use_diff_optim_inputs_False",
            "test_fsdp_optim_state.py::TestFSDPOptimState::test_shard_full_optim_state_dict_nested_use_multiple_param_groups_False_wrap_alt_True_use_diff_optim_inputs_False",
            "test_fsdp_optim_state.py::TestFSDPOptimState::test_shard_full_optim_state_dict_unmanaged_params_state_dict_type0_add_to_fsdp_module_True",
            "test_fsdp_core.py::TestParityWithDDPNEURON::test_mixture_of_experts_with_delay_before_free_offload_true_shard_grad_op_neuron",
            "test_fsdp_use_orig_params.py::TestFSDPUseOrigParamsMultipleParamGroups::test_diff_hyperparams_sharding_strategy_str_full_shard",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT2.11] FSDP v1: Neuron runtime initialized before init_process_group."
            " The test's auto_wrap smoke test expects runtime to be uninitialized"
            " at construction time, but Neuron initializes eagerly."
        ),
        tests=[
            "test_wrap.py::TestAutoWrap::test_auto_wrap_smoke_test_device_init_mode1_cpu_offload0_use_device_id_False",
        ],
    ),
    SkipGroup(
        reason="NCCL-specific tests to not run on Neuron",
        tests=[
            "test_c10d_nccl.py::RendezvousEnvTest",
            "test_c10d_nccl.py::TimeoutTest",
            "test_c10d_nccl.py::ProcessGroupNCCLNoGPUTest",
            "test_c10d_nccl.py::ProcessGroupNCCLInitTest",
            "test_c10d_nccl.py::ProcessGroupNCCLGroupTest",
            "test_c10d_nccl.py::WorkHookTest",
            "test_c10d_nccl.py::NcclErrorHandlingTest",
            "test_c10d_nccl.py::NcclUserBufferRegistrationTest",
            "test_c10d_nccl.py::SetDeviceMethod",
            "test_c10d_nccl.py::NcclProcessGroupWithDispatchedCollectivesTests",
            "test_c10d_nccl.py::ProcessGroupNCCLOneRankTest",
            "test_c10d_nccl.py::NCCLTraceTestBase",
            "test_c10d_nccl.py::NCCLTraceTest",
            "test_c10d_nccl.py::NCCLTraceTestDumpOnTimeoutBase",
            "test_c10d_nccl.py::NCCLTraceTestDumpOnTimeout",
            "test_c10d_nccl.py::NCCLTraceTestTimeoutDumpOnStuckRanks",
            "test_c10d_nccl.py::NcclErrorDumpTest",
            "test_c10d_nccl.py::ProcessGroupNCCLLargerScaleTest",
            "test_c10d_nccl.py::DistributedDataParallelTest::test_nccl_propagate_error_reason",
        ],
    ),
    SkipGroup(
        reason=(
            "Torch dist2 is only supported for NCCL and gloo,"
            " but can be extended to Neuron later upstream"
        ),
        tests=["test_dist2.py"],
    ),
    SkipGroup(
        reason="CPU/Fake PG Group Test",
        tests=[
            "test_c10d_functional_native.py::CompileTest",
            "test_c10d_functional_native.py::CompileTestCPU",
            "test_c10d_functional_native.py::CrossThreadWaitTest",
            "test_c10d_functional_native.py::PyWorkTest",
            "test_run.py::ElasticLaunchTest::test_nproc_launch_auto_configurations",
            "test_store.py",
            "test_fake_collectives.py",
            "test_fake_pg.py",
        ],
    ),
    SkipGroup(
        reason=(
            "Neuron does not support point-to-point send/recv operations. "
            "The test calls transport.send_checkpoint which relies on these "
            "unsupported operations."
        ),
        tests=[
            "test_pg_transport.py::PgTransportGPU::test_pg_transport",
            "test_pg_transport.py::PgTransportGPU::test_pg_transport_with_mixed_content",
            "test_pg_transport.py::PgTransportGPU::test_pg_transport_with_sharded_tensor",
        ],
    ),
    SkipGroup(
        reason=(
            "Functionality Gap: We do not support setting"
            " torch_neuronx.set_device(device) before"
            " dist.init_process_group."
        ),
        tests=[
            "test_c10d_nccl.py::DistributedDataParallelTest::test_backend_multi_device_ids_not_allowed",
            "test_c10d_nccl.py::DistributedDataParallelTest::test_find_unused_parameters_kwarg_grad_is_view_debug_detail",
            "test_c10d_nccl.py::DistributedDataParallelTest::test_find_unused_parameters_kwarg_grad_is_view_debug_info",
            "test_c10d_nccl.py::DistributedDataParallelTest::test_find_unused_parameters_kwarg_grad_is_view_debug_off",
            "test_c10d_nccl.py::DistributedDataParallelTest::test_find_unused_parameters_kwarg_debug_off",
            "test_c10d_nccl.py::DistributedDataParallelTest::test_find_unused_parameters_kwarg_debug_info",
            "test_c10d_nccl.py::DistributedDataParallelTest::test_find_unused_parameters_kwarg_debug_detail",
            "TestAutoWrap::test_auto_wrap_smoke_test_device_init_mode0_cpu_offload0_use_device_id_False",
            "TestAutoWrap::test_auto_wrap_smoke_test_device_init_mode0_cpu_offload1_use_device_id_False",
            "TestAutoWrap::test_auto_wrap_smoke_test_device_init_mode0_cpu_offload1_use_device_id_True",
            "TestAutoWrap::test_auto_wrap_smoke_test_device_init_mode1_cpu_offload0_use_device_id_True",
        ],
    ),
    SkipGroup(
        reason="Functionality Gap: We do not support more than 1 device per process currently.",
        tests=[
            "test_c10d_nccl.py::DistributedDataParallelTest::test_backend_2gpu_module",
            "test_c10d_nccl.py::DistributedDataParallelTest::test_backend_multi_device_module_device_ids_None",
            "test_c10d_nccl.py::DistributedDataParallelTest::test_backend_4gpu_module",
            "test_c10d_nccl.py::DistributedDataParallelTest::test_ddp_multi_device_module_config",
            "test_c10d_nccl.py::DistributedDataParallelTest::test_grad_layout_2devicemodule",
        ],
    ),
    SkipGroup(
        reason="Neuron collectives don't support complex64",
        tests=[
            "test_c10d_nccl.py::DistributedDataParallelTest::test_ddp_complex_params",
            "test_c10d_nccl.py::DistributedDataParallelTest::test_ddp_complex_params_and_grads",
            "test_c10d_nccl.py::DistributedDataParallelTest::test_ddp_mixed_real_and_complex_params",
        ],
    ),
    SkipGroup(
        reason=(
            "Functionality Gap: Test creates tensors on neuron"
            " before PG setup which we do not support currently."
        ),
        tests=[
            "test_c10d_nccl.py::DistributedDataParallelTest::test_dataclass_output_unused_param",
            "test_c10d_nccl.py::DistributedDataParallelTest::test_dataclass_output",
        ],
    ),
    SkipGroup(
        reason=(
            "copy.deepcopy(model).to(device) fails for process with rank=1 "
            "assigned device=32, it should be resovled once we onboard "
            "setup to assign device=1 to process of rank=1"
        ),
        tests=[
            "test_c10d_nccl.py::DistributedDataParallelTest::test_accumulate_gradients_module",
            "test_c10d_nccl.py::DistributedDataParallelTest::test_accumulate_gradients_module_with_grad_is_view",
            "test_c10d_nccl.py::DistributedDataParallelTest::test_backend_1gpu_module_device_ids_integer_list",
            "test_c10d_nccl.py::DistributedDataParallelTest::test_backend_1gpu_module_device_ids_torch_device_list",
        ],
    ),
    SkipGroup(
        reason="NCCL/CUDA-specific test — not applicable to Neuron",
        tests=[
            "test_c10d_nccl.py::CommTest::test_pass_nccl_options_config",
            "test_c10d_nccl.py::CommTest::test_intra_node_comm_all_reduce",
            "test_c10d_nccl.py::CommTest::test_intra_node_comm_all_reduce_custom_group_name_False",
            "test_c10d_nccl.py::CommTest::test_intra_node_comm_all_reduce_custom_group_name_True",
            "test_device_mesh.py::DeviceMeshSetDeviceTest::test_auto_set_device_from_heuristic",
        ],
    ),
    SkipGroup(
        reason="Neuron cannot manually set device indices to a rank",
        tests=["test_device_mesh.py::DeviceMeshSetDeviceTest::test_manual_set_device"],
    ),
    SkipGroup(
        reason="Fake backend test or non Neuron backend test",
        tests=[
            "test_device_mesh.py::InitDeviceMeshTest::test_backend_override_argument_dict_with_idx_and_backend_lazy",
            "test_device_mesh.py::InitDeviceMeshTest::test_backend_override_argument_dict_with_name_and_options",
            "test_c10d_nccl.py::NcclProcessGroupWithDispatchedCollectivesTests::test_init_process_group_for_all_backends",
            "test_op_strategy.py::TestOpSpecMesh::test_mesh_from_input_specs_with_leading_none",
            "test_op_strategy.py::TestOpSpecMesh::test_mesh_raises_when_all_specs_none",
            "test_op_strategy.py::TestOpSpecMesh::test_mesh_from_input_specs_when_tuple_output_all_none",
            "test_op_strategy.py::TestOpSpecMesh::test_mesh_from_tuple_output_specs_with_leading_none",
        ],
    ),
    SkipGroup(
        reason=(
            "Device indexing does not match CUDA specific "
            "behavior, expects specific default indexing "
            "based on CUDA device indexing"
        ),
        tests=[
            "test_device_mesh.py::DeviceMeshSetDeviceTest::test_auto_set_device_from_local_rank"
        ],
    ),
    SkipGroup(
        reason=(
            "torch.dist._time_estimator currently "
            "only supports NCCL backend but it can easily "
            "be extended to other backends."
        ),
        tests=["test_c10d_nccl.py::CommTest::test_time_estimate_nccl"],
    ),
    SkipGroup(
        reason="Send/recv not supported",
        tests=[
            "test_c10d_object_collectives.py::TestObjectCollectives::test_send_recv_object_list",
            "test_c10d_nccl.py::LargeCommTest::test_batch_send_recv_subgroup_group_rank_False",
            "test_c10d_nccl.py::LargeCommTest::test_batch_send_recv_subgroup_group_rank_True",
            "test_c10d_nccl.py::LargeCommTest::test_send_recv_object_list_subgroup_set_device0_group_rank_True",
            "test_c10d_nccl.py::LargeCommTest::test_send_recv_object_list_subgroup_set_device1_group_rank_False",
            "test_c10d_nccl.py::LargeCommTest::test_send_recv_object_list_subgroup_set_device1_group_rank_True",
            "test_c10d_nccl.py::LargeCommTest::test_send_recv_subgroup_group_rank_False_async_op_False",
            "test_c10d_nccl.py::LargeCommTest::test_send_recv_subgroup_group_rank_True_async_op_True",
            "test_c10d_nccl.py::LargeCommTest::test_send_recv_subgroup_group_rank_True_async_op_False",
            "test_c10d_nccl.py::LargeCommTest::test_send_recv_object_list_subgroup_set_device0_group_rank_False",
            "test_c10d_nccl.py::LargeCommTest::test_send_recv_subgroup_group_rank_False_async_op_True",
            "test_c10d_ops_privateuse1.py::ProcessGroupPrivateUse1OpTest::test_send_recv",
            "test_c10d_ops_privateuse1.py::ProcessGroupPrivateUse1OpTest::test_send_recv_object_list",
            "test_c10d_ops_privateuse1.py::ProcessGroupPrivateUse1OpTest::test_send_recv_complex",
        ],
    ),
    SkipGroup(
        reason="Neuron does not support replica_group out of list [4, 8, 16 and 32 divisible]",
        tests=[
            "test_c10d_nccl.py::NcclProcessGroupWithDispatchedCollectivesTests::test_collectives"
        ],
    ),
    SkipGroup(
        reason="No implementation - Neuron not support dtype float8_e4m3fn / float8_e5m2",
        tests=[
            "test_c10d_nccl.py::NcclProcessGroupWithDispatchedCollectivesTests::test_allgather_float8_float8_e5m2",
            "test_c10d_nccl.py::NcclProcessGroupWithDispatchedCollectivesTests::test_allgather_float8_float8_e4m3fn",
            "test_c10d_functional_native.py::TestWithNCCL::test_fixed_striding",
        ],
    ),
    SkipGroup(
        reason=("Neuron does not support all_to_all for world_size 2 and odd split sizes"),
        tests=[
            "test_c10d_spawn_nccl.py::TestDistributedNNFunctionsNccl::test_all_gather",
            "test_c10d_spawn_nccl.py::TestDistributedNNFunctionsNccl::test_all_to_all",
            "test_c10d_spawn_nccl.py::TestDistributedNNFunctionsNccl::test_all_to_all_single",
            "test_c10d_functional_native.py::TestWithNCCL::test_all_to_all_single",
        ],
    ),
    SkipGroup(
        reason="torch.nn.distributed.functional._all_gather_base is only supported for NCCL/XCCL",
        tests=["test_c10d_spawn_nccl.py::TestDistributedNNFunctionsNccl::test_all_gather_base"],
    ),
    SkipGroup(
        reason="Neuron does not support new functional collective: batch_p2p_ops_inplace",
        tests=["test_c10d_functional_native.py::TestWithNCCL::test_functional_collectives_batched"],
    ),
    SkipGroup(
        reason="Test tries to use device index not within the range of the set world_size",
        tests=[
            "tests/pytorch_tests_native/test/distributed/test_c10d_nccl.py::CommTest::test_nccl_barrier",
            "TestFullyShardRegisteredParams::test_param_registration_after_backward",
            "TestFullyShardRegisteredParams::test_param_registration_after_forward",
        ],
    ),
    # FSDP v2
    SkipGroup(
        reason="Expert Parallelism (EP) not supported on Neuron",
        tests=[
            "TestFullyShardNDTraining::test_shard_placement_fn_tp_ep",
        ],
    ),
    # FSDP v2 — Neuron ProcessGroup does not support CPU tensor collectives
    SkipGroup(
        reason=(
            "Neuron ProcessGroup does not support collectives on CPU tensors."
            " Tests broadcast CPU-resident parameters (e.g., meta device init pattern)."
        ),
        tests=[
            "TestFullyShardShardPlacementFn::test_init_1d_transformer_shard_dim_neg1",
            "TestFullyShardShardPlacementFn::test_init_1d_transformer_shard_largest_dim",
            "TestFullyShardShardPlacementFn::test_init_2d_transformer_shard_diff_dim",
            "TestFullyShardMetaDeviceInit::test_rank0_broadcast_meta_device_init",
            "TestFullyShardUnshardMultiThread::test_unshard_without_lazy_init",
            "TestFullyShardDeviceDTensor::test_move_states_to_device_dtensor_invalid",
            "TestFullyShardProcessGroupInit::test_1d_process_group_init",
            "TestFullyShardProcessGroupInit::test_2d_process_group_init",
        ],
    ),
    SkipGroup(
        reason="Timing assertion fails or flaky in overlap measurement",
        tests=[
            "TestFullyShardOverlap::test_fully_shard_training_overlap",
            "TestFullyShardOverlap::test_fully_shard_post_optim_event_overlap",
            "TestFullyShardPerParamMeshOverlap::test_fully_shard_per_param_mesh_training_overlap",
            "TestFullyShardPerParamMeshOverlap::test_fully_shard_per_param_mesh_no_grad_input_overlap",
        ],
    ),
    SkipGroup(
        reason="Requires 2 devices per process, where Neuron only uses 1 device per process",
        tests=[
            "test_zero_redundancy_optimizer.py::TestZeroRedundancyOptimizerDistributed::test_zero_model_parallel",
        ],
    ),
    # FSDP v2
    SkipGroup(
        reason="Needs further investigation - Process SIGABRT in all_gather"
        " extensions - NRT crash on error path",
        tests=[
            "TestFullyShardAllGatherExtensionsMultiProcess::test_all_gather_extensions_train_parity",
        ],
    ),
    # FSDP v1
    SkipGroup(
        reason=(
            "Xdist serialization incompatible: execnet can't serialize"
            " functools.partial in test result"
        ),
        tests=[
            "CheckpointWrapperTest::test_apply_activation_checkpointing",
            "CheckpointWrapperTest::test_checkpoint_wrapper_kwarg_support",
        ],
    ),
    SkipGroup(
        reason=(
            "MultiThreadedTestCase spawns threads sharing one device — "
            "incompatible with Neuron's one-device-per-process runtime model"
        ),
        tests=[
            "test_functional_differentials.py::TestFunctionalDifferentials",
        ],
    ),
    SkipGroup(
        reason=(
            "ReplicateFullyShardInit inherits CPU/gloo tests from"
            " ReplicateTest — hangs until MultiProcContinuousTest"
            " 240s completion queue timeout on Neuron"
        ),
        tests=[
            "ReplicateFullyShardInit::test_replicate_multi_module",
            "ReplicateFullyShardInit::test_replicate_with_kwargs",
            "ReplicateFullyShardInit::test_replicate_single_module",
            "ReplicateFullyShardInit::test_replicate_wrong_device_id_type",
        ],
    ),
    SkipGroup(
        reason="Test uses .cuda(rank) to place params on multiple GPUs — "
        "Neuron has one device per process",
        tests=[
            "TestFSDPMiscMultiThread::test_multigpu_module",
            "TestZeroRedundancyOptimizerDistributed::test_zero_model_parallel_parameters_as_bucket_view_False",
            "TestZeroRedundancyOptimizerDistributed::test_zero_model_parallel_parameters_as_bucket_view_True",
        ],
    ),
    SkipGroup(
        reason=(
            "NRT cannot init collectives for replica groups that are not chip-aligned on trn2."
        ),
        tests=[
            "TestTPFSDPIntegration::test_fsdp_tp_extension_grad",
            "TestTPFSDPIntegration::test_fsdp_tp_integration",
            "TestTPFSDPIntegration::test_fsdp_tp_sync_module_state",
            "ComposabilityTest::test_replicate_pp_ScheduleClass0_bfloat16",
            "ComposabilityTest::test_replicate_pp_ScheduleClass0_float32",
            "ComposabilityTest::test_replicate_pp_ScheduleClass1_bfloat16",
            "ComposabilityTest::test_replicate_pp_ScheduleClass1_float32",
            "ComposabilityTest::test_replicate_pp_ScheduleClass2_bfloat16",
            "ComposabilityTest::test_replicate_pp_ScheduleClass2_float32",
            "ComposabilityTest::test_replicate_pp_ScheduleClass3_bfloat16",
            "ComposabilityTest::test_replicate_pp_ScheduleClass3_float32",
            "ComposabilityTest::test_replicate_pp_ScheduleClass4_bfloat16",
            "ComposabilityTest::test_replicate_pp_ScheduleClass4_float32",
            "ComposabilityTest::test_replicate_pp_grads_ScheduleClass0",
            "ComposabilityTest::test_replicate_pp_grads_ScheduleClass1",
            "ComposabilityTest::test_replicate_pp_grads_ScheduleClass2",
            "ComposabilityTest::test_replicate_pp_grads_ScheduleClass3",
            "ComposabilityTest::test_replicate_pp_grads_ScheduleClass4",
        ],
    ),
    SkipGroup(
        reason=(
            "TestSingleProc and TestMultiProc hardcode MASTER_PORT in setUpClass "
            "which conflicts with xdist parallel workers"
        ),
        tests=[
            "test_dynamo_distributed.py::TestSingleProc",
            "test_dynamo_distributed.py::TestMultiProc",
        ],
    ),
    SkipGroup(
        reason=(
            "C++ source lint test that scans torch/csrc/distributed/c10d/ for "
            "unsafe thread_local types — source tree not present in test overlay"
        ),
        tests=[
            "ThreadLocalSafetyLintTest::test_no_non_trivial_thread_locals",
        ],
    ),
    SkipGroup(
        reason=("Test checks for specific CUDA error message on failures."),
        tests=[
            "TestFSDPMiscMultiThread::test_no_params",
            "TestFSDPMiscWorldSize1::test_training_device_mismatch_errors",
        ],
    ),
    # TODO: Upstream fix needed — tests hardcode MASTER_PORT=6789 which
    # conflicts with Neuron's distributed runtime that already binds ports.
    # Tests exercise the Python PG extension API with a mock "dummy" backend
    # on CPU tensors — no device involvement.
    SkipGroup(
        reason=(
            "Hardcoded MASTER_PORT=6789 conflicts with Neuron runtime. "
            "Tests Python PG extension API with dummy backend on CPU tensors"
        ),
        tests=[
            "PythonProcessGroupExtensionTest::test_abort",
            "PythonProcessGroupExtensionTest::test_canonicalize_helper",
            "PythonProcessGroupExtensionTest::test_collectives",
            "PythonProcessGroupExtensionTest::test_init_process_group_with_multiple_backends",
            "PythonProcessGroupExtensionTest::test_send_recv",
            "PythonProcessGroupExtensionTest::test_shutdown",
        ],
    ),
    # TODO: Upstream investigation — Neuron backend auto-registration
    # overrides the custom backend name. Test expects pg.name() == "custom"
    # but gets "neuron" because torch_neuronx registers on import.
    SkipGroup(
        reason=(
            "Neuron backend auto-registration overrides custom backend name: "
            "pg.name() returns 'neuron' instead of expected 'custom'"
        ),
        tests=[
            "ProcessGroupWithDispatchedCollectivesTests::test_init_process_group_for_all_backends",
        ],
    ),
    SkipGroup(
        reason=(
            "Replicate CPU compile tests — not Neuron-related. CPU-only compile"
            " path (device='cpu', gloo, inductor) that does not involve the"
            " Neuron backend. The _with_compiler variant is additionally flaky:"
            " it compares eager vs inductor-compiled gradients, which accumulate"
            " reductions in a different order, yielding ~3.8e-5-7.3e-5 abs diffs"
            " that intermittently exceed the test's tight rtol=1.3e-6 (~1/5 fail"
            " rate on trn2, worse under parallel CPU contention)."
        ),
        tests=[
            "test_replicate.py::ReplicateTest::test_compile_cpu",
            "test_replicate_with_compiler.py::ReplicateTest::test_compile_cpu",
        ],
    ),
    SkipGroup(
        reason=(
            "HSDP DTensor state-dict offload-to-cpu test intermittently hangs"
            " for 7000+ seconds (2+ hours) before being killed. The test"
            " sometimes passes on retry in ~22s, but the hang risk is too"
            " expensive to leave in the suite. Skip until HSDP cpu-offload"
            " synchronization on Neuron is investigated."
        ),
        tests=[
            "test_hsdp_dtensor_state_dict.py::TestHSDPWithDeviceMeshAndDTensorNEURON::test_dtensor_sharded_tensor_state_dict_identical_offload_to_cpu_True_neuron",
        ],
    ),
    SkipGroup(
        reason=(
            "FSDP autograd test_unshard_params_as_tensors intermittently"
            " hangs and is killed by SIGTERM at the 600s subprocess timeout"
            " (exit code -15). The test passes in some runs but the flaky"
            " hang costs 10 minutes each time it triggers. Skip until the"
            " unshard tensor-handle synchronization is investigated."
        ),
        tests=[
            "test_fsdp_core.py::TestAutogradNEURON::test_unshard_params_as_tensors_neuron",
        ],
    ),
]
