from tests.pytorch_tests_native.neuron_test_registry import SkipGroup

SKIP_GROUPS: list[SkipGroup] = [
    SkipGroup(
        reason="NCCL-specific tests which will be onboarded as part of c10d.",
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
        ],
    ),
    SkipGroup(
        reason="CPU/Fake PG Group Test",
        tests=[
            "test_c10d_functional_native.py::CompileTest",
            "test_c10d_functional_native.py::CompileTestCPU",
            "test_c10d_functional_native.py::CrossThreadWaitTest",
            "test_c10d_functional_native.py::PyWorkTest",
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
        reason="Test uses NCCL specific options",
        tests=["test_c10d_nccl.py::CommTest::test_pass_nccl_options_config"],
    ),
    SkipGroup(
        reason="Looks for CUDA specific warning",
        tests=["test_device_mesh.py::DeviceMeshSetDeviceTest::test_auto_set_device_from_heuristic"],
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
        reason="Testing CUDA specific APIs",
        tests=["test_c10d_nccl.py::CommTest::test_intra_node_comm_all_reduce"],
    ),
    SkipGroup(
        reason=(
            "func: torch.dist._time_estimator currently "
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
        ],
    ),
    SkipGroup(
        reason=(
            "RuntimeError: Neuron does not support "
            "all_to_all for world_size 2 and odd "
            "split sizes"
        ),
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
]
