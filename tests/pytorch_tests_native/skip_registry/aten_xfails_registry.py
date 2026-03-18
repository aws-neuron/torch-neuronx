from tests.pytorch_tests_native.neuron_test_registry import SkipGroup

XFAIL_GROUPS: list[SkipGroup] = [
    SkipGroup(
        reason=(
            "Neuron only supports contiguous/preserve memory format"
            " Conv2d with channels_last layout fails through forward/backward."
        ),
        tests=[
            "test_c10d_nccl.py::DistributedDataParallelTest::test_grad_layout_1devicemodule_1replicaperprocess",
            "test_c10d_nccl.py::DistributedDataParallelTest::test_param_layout_mismatch_error",
        ],
    ),
    SkipGroup(
        reason=(
            "torch-neuronx registers fused LSTM for PrivateUse1, displacing"
            " CPU registration — test runs reference model on CPU first."
        ),
        tests=[
            "test_c10d_nccl.py::DistributedDataParallelTest::test_ddp_packed_sequence",
        ],
    ),
]
