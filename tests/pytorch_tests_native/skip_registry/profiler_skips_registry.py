from tests.pytorch_tests_native.neuron_test_registry import SkipGroup

SKIP_GROUPS: list[SkipGroup] = [
    SkipGroup(
        reason=(
            "test_memory_profiler expects device caching allocator metrics — "
            "alloc_fn user_scope_alloc reports 0 without GPU memory tracking"
        ),
        tests=[
            "TestProfiler::test_memory_profiler",
        ],
    ),
    SkipGroup(
        reason=(
            "record_function user annotations not correlated with PrivateUse1 device activity — "
            "Kineto does not associate user_function events with PrivateUse1 DeviceType "
            "(same limitation as XPU, which skips these tests upstream)"
        ),
        tests=[
            "CppThreadTestPrivateUse1::test_with_enable_profiler_in_child_thread_privateuse1",
            "CppThreadTestPrivateUse1::test_without_enable_profiler_in_child_thread_privateuse1",
        ],
    ),
]
