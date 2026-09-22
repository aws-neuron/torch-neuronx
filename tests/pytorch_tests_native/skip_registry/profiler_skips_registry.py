from tests.pytorch_tests_native.neuron_test_registry import SkipGroup

SKIP_GROUPS: list[SkipGroup] = [
    SkipGroup(
        reason=(
            "test_memory_profiler expects device caching allocator metrics — "
            "alloc_fn user_scope_alloc reports 0 without GPU memory tracking"
        ),
        tests=[
            "TestProfiler::test_memory_profiler",
            "TestProfilerDeviceNEURON::test_memory_profiler_neuron",
        ],
    ),
    SkipGroup(
        reason=(
            "Neuron's caching allocator does not call c10::reportMemoryUsageToProfiler(), "
            "so PyTorch's memory profiler never sees neuron allocations. The timeline is "
            "missing PREEXISTING+DESTROY events for tensors allocated before the profiler "
            "started (expected 6 events, got 4)."
        ),
        tests=[
            "TestMemoryProfilerTimelineNEURON::test_memory_timeline_no_id_neuron",
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
            "CppThreadTestNEURON::test_with_enable_profiler_in_child_thread_neuron",
            "CppThreadTestNEURON::test_without_enable_profiler_in_child_thread_neuron",
            "CppThreadTestNEURON::test_profile_memory_neuron",
        ],
    ),
    SkipGroup(
        reason=(
            "standard upstream profiler tests do not provide NeuronConfig — "
            "without the neuron_profile custom config, the Neuron activity profiler "
            "intentionally does not create an NRT profiling session"
        ),
        tests=[
            "TestProfilerDeviceNEURON::test_dynamic_toggle_neuron",
            "TestProfilerDeviceNEURON::test_profiler_neuron",
            "TestProfilerDeviceNEURON::test_kineto_kernel_metadata_in_trace_neuron",
        ],
    ),
    SkipGroup(
        reason=(
            "test_basic_profile expects only aten::mul_ in the profiler trace. On a cold "
            "cache, neuron compiles the op inside the profiler scope, and the lowering "
            "function calls torch.mul() internally to produce StableHLO — this leaks "
            "aten::mul into the trace. On a warm cache the test would pass. Same root "
            "cause as the test_user_annotation profiler leak: compilation events are "
            "visible to the profiler when they occur inside a profile() context."
        ),
        tests=[
            "TestProfilerDeviceNEURON::test_basic_profile_neuron",
        ],
    ),
    SkipGroup(
        reason=(
            "test_kineto_profiler_with_environment_variable launches a nested Python "
            "process that cannot share its xdist worker's assigned NeuronCore; the "
            "same profiler path passes when the test runs in isolation"
        ),
        tests=[
            "TestProfilerDeviceNEURON::test_kineto_profiler_with_environment_variable_neuron",
        ],
    ),
]
