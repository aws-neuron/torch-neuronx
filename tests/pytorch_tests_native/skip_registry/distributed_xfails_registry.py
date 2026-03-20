from tests.pytorch_tests_native.neuron_test_registry import SkipGroup

XFAIL_GROUPS: list[SkipGroup] = [
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
        reason="Barrier not supported for ProcessGroupNeuron for non default group",
        tests=[
            "test_c10d_nccl.py::CommTest::test_nccl_warn_not_in_group_debug_info",
            "test_c10d_nccl.py::CommTest::test_nccl_warn_not_in_group_debug_off",
            "test_c10d_nccl.py::CommTest::test_nccl_warn_not_in_group_debug_detail",
        ],
    ),
]
