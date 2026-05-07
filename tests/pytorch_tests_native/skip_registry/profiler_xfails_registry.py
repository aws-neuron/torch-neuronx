from tests.pytorch_tests_native.neuron_test_registry import SkipGroup

SKIP_GROUPS: list[SkipGroup] = [
    SkipGroup(
        reason=("Expects cpp symbol to be built for profiler which is not automatically there."),
        tests=["TestExperimentalUtils::test_fuzz_symbolize"],
    ),
    SkipGroup(
        reason=(
            "Execution trace tensor_range attribute not populated for PrivateUse1 device — "
            "integral tensor min/max range not recorded in execution trace nodes"
        ),
        tests=[
            "TestExecutionTracePRIVATEUSE1::test_execution_trace_record_integral_tensor_range_neuron"
        ],
    ),
    SkipGroup(
        reason=(
            "test_profile_all_threads event count mismatch — "
            "extra profiler events from device runtime threads"
        ),
        tests=[
            "TestProfiler::test_profile_all_threads",
        ],
    ),
    SkipGroup(
        reason=(
            "test_profiler_synchronized_dataloader_pattern DataLoader worker crashes — "
            "multiprocessing worker exits with code 1 on this device"
        ),
        tests=[
            "TestExperimentalUtils::test_profiler_synchronized_dataloader_pattern",
        ],
    ),
    SkipGroup(
        reason=(
            "Neuron profiler initialization spawns an eager worker thread that the Python "
            "tracer detects as an extra TID (off-by-one in expected thread count) and reports "
            "events with placeholder TID 2^64-1. Root cause: profiler includes PrivateUse1 "
            "activity by default which triggers NRT init and background thread creation."
        ),
        tests=[
            "TestProfiler::test_source_multithreaded_basic_work_in_main_thread_True",
            "TestProfiler::test_source_multithreaded_basic_work_in_main_thread_False",
            "TestProfiler::test_source_multithreaded_multiple_preexisting_work_in_main_thread_True",
            "TestProfiler::test_source_multithreaded_multiple_preexisting_work_in_main_thread_False",
            "TestProfiler::test_source_multithreaded_open_in_scope_work_in_main_thread_True",
            "TestProfiler::test_source_multithreaded_open_in_scope_work_in_main_thread_False",
            "TestProfiler::test_source_multithreaded_close_in_scope_work_in_main_thread_True",
            "TestProfiler::test_source_multithreaded_close_in_scope_work_in_main_thread_False",
            "TestProfiler::test_source_multithreaded_complex_work_in_main_thread_True",
            "TestProfiler::test_source_multithreaded_complex_work_in_main_thread_False",
        ],
    ),
    SkipGroup(
        reason=(
            "Neuron profiler incorrectly marks non-user events as user annotations. "
            "record_function scope boundary detection needs fixing for PrivateUse1 backend."
        ),
        tests=[
            "TestProfiler::test_user_annotation",
        ],
    ),
    SkipGroup(
        reason=(
            "Neuron profiler produces invalid JSON in chrome trace export — "
            "unescaped control character in event list output causes JSONDecodeError."
        ),
        tests=[
            "TestProfiler::test_event_list",
        ],
    ),
]
