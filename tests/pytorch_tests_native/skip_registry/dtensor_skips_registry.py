from tests.pytorch_tests_native.neuron_test_registry import SkipGroup
from tests.pytorch_tests_native.skip_registry import skipped_tests as s

SKIP_GROUPS: list[SkipGroup] = [
    SkipGroup(
        reason=(
            "Extreme duration (>15 min): these tests take 20-75 minutes to complete"
            " and block workers for long periods, causing overall suite timeout."
            " Needs investigation to determine if it's a Neuron compiler issue,"
            " test input size, or infrastructure contention."
        ),
        tests=[
            "MultiDimRedistributeTest::test_multi_dim_mesh",
            "RedistributeTest::test_redistribute_uneven_sharding",
            "DistTensorOpsTest::test_index",
        ],
    ),
    SkipGroup(
        reason=(
            "world_size=6 unsupported on Neuron"
            " Child processes crash at nrt_load_collectives, causing parent to hang for 600s."
        ),
        tests=[
            "TestViewOps",
            "UnevenFlattenedReduceScatterTest",
        ],
    ),
    SkipGroup(
        reason=(
            "Test hangs: reduce_scatter with empty tensors (shape [N, 0, N])"
            " falls back to CPU on some ranks, causing"
            " collective rank divergence deadlock"
        ),
        tests=[
            "test_graph_based_redistribute_cost",
        ],
    ),
    SkipGroup(
        reason="Class-level skip: WithLocalTensor not supported on Neuron",
        patterns=["*WithLocalTensor"],
        match_target="test_class",
    ),
    SkipGroup(
        reason="Class-level skip: MultiThreadedDTensorOps not supported on Neuron",
        patterns=["*MultiThreadedDTensorOps*"],
        match_target="test_class",
    ),
    SkipGroup(
        reason="complex dtype unsupported on Neuron",
        tests=s.DTENSOR_COMPLEX_DTYPE_UNSUPPORTED,
    ),
    SkipGroup(
        reason="AllToAll unsupported world size 2",
        tests=s.DTENSOR_ALLTOALL_WORLD_SIZE_2_UNSUPPORTED,
    ),
    SkipGroup(
        reason="Test explicitly uses Inductor backend which has no Neuron codegen support",
        tests=[
            "test_compile_bernoulli",
            "test_compile_bernoulli_float",
            "test_compile_multiple_random_ops",
            "test_compile_native_dropout",
            "test_compile_normal_",
            "test_compile_rand_like",
            "test_compile_randint_like",
            "test_compile_randn_like",
            "test_compile_uniform_",
            "test_flattened_submesh_no_getattr_compile_on_one_rank",
            "test_conv2d_no_bias_compile",
            "test_split_with_symint_split_size",
            "test_dtensor_add_op_dispatch_overhead",
            "test_compile_embedding_redistribute",
        ],
    ),
    SkipGroup(
        reason="Functionality gap: each process is assigned a different core",
        tests=[
            "test_compiled_rowwise_embedding_graph_consistency",
        ],
    ),
    SkipGroup(
        reason="float8 dtype unsupported: scaled_mm requires float8 which Neuron does not support",
        tests=[
            "test_scaled_mm",
        ],
    ),
]
