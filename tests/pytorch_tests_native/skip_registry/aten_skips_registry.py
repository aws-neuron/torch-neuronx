from tests.pytorch_tests_native.neuron_test_registry import SkipGroup

SKIP_GROUPS: list[SkipGroup] = [
    SkipGroup(
        reason="Long compilation - causes test suite timeout",
        tests=["test_ReplicationPad2d_large_neuron"],
    ),
    SkipGroup(
        reason="Segfault in view ops - requires investigation",
        patterns=["*conj_imag_view*", "*conj_view*"],
    ),
    SkipGroup(
        reason="Long compilation/execution for conv transpose",
        tests=[
            "test_conv_transpose_with_output_size_and_no_batch_dim",
            "test_conv_transpose_with_output_size_and_no_batch_dim_ConvTranspose2d_neuron",
            "test_conv_transpose_with_output_size_and_no_batch_dim_ConvTranspose3d_neuron",
        ],
    ),
    SkipGroup(
        reason="Functionality Gap: We do not support quantization_per_channel format currently",
        tests=[
            "test_ddp_hooks.py::DistributedDataParallelCommHookTest::test_ddp_comm_hook_quantize_per_channel_hook",
        ],
    ),
    SkipGroup(
        reason=(
            "NotImplementedError: Could not run "
            "aten::_sparse_coo_tensor_with_dims_and_tensors "
            "with arguments from the Sparseneuron backend"
        ),
        tests=[
            "test_c10d_nccl.py::SparseCollective::test_ddp_set_sparse_metadata",
        ],
    ),
]
