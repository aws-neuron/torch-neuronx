"""
Integration test for Neuron Profiler with PyTorch profiler.
Verifies end-to-end profiling: NRT trace output + chrome trace export.
Also tests NeuronConfig and ProfileMode API.

Tests are split into three groups:
  - Non-hardware (cases 1-4): Run without Neuron device, verify Python/Kineto behavior.
  - Per-mode hardware tests: Isolate each profiling mode to verify mode-dependent NRT
    output files (.ntff for DEVICE, ntrace.pb for RUNTIME) are only produced when the
    corresponding mode is enabled. The monitoring files (trace_info.pb, cpu_util.pb,
    host_mem.pb) are produced only when RUNTIME mode is active.
  - Comprehensive hardware test (test_neuron_profiling): All modes enabled with CPU+PU1,
    verifying output structure, chrome trace export, events, flow arrows, and metadata.
"""

import json
import os

import pytest
import torch
from torch._C._profiler import _ExperimentalConfig
from torch.profiler import ProfilerActivity, profile, record_function, schedule

import torch_neuronx
from torch_neuronx.profiling import NeuronConfig, ProfileMode


@pytest.mark.skipif(
    os.environ.get("NEURON_LAUNCH_BLOCKING") == "1",
    reason="Framework mapping tests are incompatible with NEURON_LAUNCH_BLOCKING=1",
)
class TestNeuronProfilerIntegration:
    """Integration tests for Neuron profiler with PyTorch profiler API.

    Systematically tests the matrix of activity combinations x NeuronConfig
    settings and verifies correct traces are produced in each case.

    Key control flow:
    - PyTorch framework traces (RecordFunction callbacks) are registered only
      when CPU is in activities (profiler_kineto.cpp:enableProfiler())
    - NRT profiling happens only when hasNeuronProfileKey() finds "neuron_profile:"
      prefix in the custom config string (Profiler.cpp:29)
    - The session's initNrtConfig() sets all NRT activities OFF, then selectively
      enables only modes from the config (Session.cpp:135-138)
    """

    @pytest.fixture(autouse=True)
    def _test_dir(self, tmp_path):
        """Provide a temporary directory for profiler output."""
        self.test_dir = str(tmp_path)

    def _assert_has_framework_traces(self, prof):
        """Assert profiler captured PyTorch op-level events."""
        events = prof.events()
        assert events is not None
        names = [e.name for e in events]
        assert "model_inference" in names

    def _assert_no_framework_traces(self, prof):
        """Assert profiler did NOT capture PyTorch op-level events."""
        events = prof.events()
        # When CPU not in activities, no RecordFunction callbacks registered
        if events is not None:
            names = [e.name for e in events]
            assert "model_inference" not in names

    def _assert_no_nrt_output(self, output_dir):
        """Assert NRT did NOT create output files."""
        if not os.path.isdir(output_dir):
            return  # Directory doesn't exist, so no output
        subdirs = [
            d
            for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d)) and "_pid_" in d
        ]
        assert len(subdirs) == 0, f"Expected no NRT output, found: {subdirs}"

    def _validate_nrt_output_structure(self, output_dir):
        """Validate that NRT created the expected output files.

        Expected structure:
            {output_dir}/
            └── {instance_id}_pid_{process_id}/
                ├── *_instid_*_vnc_*.ntff    # Trace files per VNC
                ├── cpu_util.pb              # CPU utilization data
                ├── host_mem.pb              # Host memory data
                ├── ntrace.pb                # System trace events
                └── trace_info.pb            # Trace metadata

        Note: .neff files (compiled model binaries) are only dumped on first
        compilation. When NEFFs are already cached by NRT (e.g. from earlier
        profiling sessions in the same process), they won't appear here.

        Returns:
            str: Path to the instance subdirectory
        """
        # Find the instance subdirectory (pattern: *_pid_*)
        subdirs = [
            d
            for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d)) and "_pid_" in d
        ]
        assert len(subdirs) == 1, f"Expected exactly one instance subdirectory, found: {subdirs}"

        instance_dir = os.path.join(output_dir, subdirs[0])
        top_level = os.listdir(instance_dir)

        # NRT places files inside a numeric timestamp subdirectory within the instance dir
        numeric_subdirs = [
            d for d in top_level if os.path.isdir(os.path.join(instance_dir, d)) and d.isdigit()
        ]
        assert (
            len(numeric_subdirs) >= 1
        ), f"Expected at least one numeric subdirectory inside {subdirs[0]}, found: {top_level}"

        trace_dir = os.path.join(instance_dir, numeric_subdirs[0])
        files = os.listdir(trace_dir)

        # Check all required protobuf files
        required_pb_files = ["trace_info.pb", "ntrace.pb", "cpu_util.pb", "host_mem.pb"]
        for pb_file in required_pb_files:
            assert pb_file in files, f"Missing required file: {pb_file}"

        # Check for exactly one .ntff trace file (SESSION mode produces single file)
        ntff_files = [f for f in files if f.endswith(".ntff")]
        assert len(ntff_files) == 1, "Single session NTFF device profile not found"

        # NOTE: We intentionally do NOT assert .neff files here.
        # NRT only dumps neff_*.neff files when a model is first compiled in
        # the process. Once a NEFF is cached in the NRT compilation cache,
        # subsequent profiling sessions in the same process reuse the cached
        # binary and NRT does not copy it into the output directory again.
        # This means .neff presence depends on test ordering: the first
        # hardware test triggers compilation and gets .neff output, but later
        # tests (like test_all_modes running after cases 5-12) do not.
        # The .neff files are not profiling artifacts -- they are compilation
        # artifacts that happen to land in the profile output dir on first run.

        return instance_dir

    def _validate_nrt_output_files(
        self,
        output_dir,
        *,
        expect_ntff=False,
        expect_ntrace=False,
    ):
        """Validate NRT output files for a specific mode combination.

        Mode-dependent files are checked based on the flags:
        - .ntff: produced only when device_profile (DEVICE mode) is enabled
        - ntrace.pb, trace_info.pb, cpu_util.pb, host_mem.pb: produced only when
          system_profile (RUNTIME mode) is enabled

        Args:
            output_dir: Path to the profiler output directory.
            expect_ntff: Whether .ntff device trace files should be present.
            expect_ntrace: Whether ntrace.pb and monitoring files should be present.

        Returns:
            str: Path to the instance subdirectory.
        """
        subdirs = [
            d
            for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d)) and "_pid_" in d
        ]
        assert len(subdirs) == 1, f"Expected exactly one instance subdirectory, found: {subdirs}"

        instance_dir = os.path.join(output_dir, subdirs[0])
        top_level = os.listdir(instance_dir)

        # NRT places files inside a numeric timestamp subdirectory within the instance dir
        numeric_subdirs = [
            d for d in top_level if os.path.isdir(os.path.join(instance_dir, d)) and d.isdigit()
        ]
        assert (
            len(numeric_subdirs) >= 1
        ), f"Expected at least one numeric subdirectory inside {subdirs[0]}, found: {top_level}"

        files = os.listdir(os.path.join(instance_dir, numeric_subdirs[0]))

        # .ntff device trace files (mode-dependent: DEVICE mode)
        ntff_files = [f for f in files if f.endswith(".ntff")]
        if expect_ntff:
            assert len(ntff_files) >= 1, f"Expected .ntff files but found none. Files: {files}"
        else:
            assert len(ntff_files) == 0, f"Unexpected .ntff files: {ntff_files}"

        # System trace / monitoring files (mode-dependent: RUNTIME mode)
        runtime_files = ("ntrace.pb", "trace_info.pb", "cpu_util.pb", "host_mem.pb")
        if expect_ntrace:
            for rf in runtime_files:
                assert rf in files, f"Expected {rf} but not found. Files: {files}"
        else:
            for rf in runtime_files:
                assert rf not in files, f"Unexpected {rf} present. Files: {files}"

        return instance_dir

    def _validate_chrome_trace_basics(self, prof, output_dir):
        """Export and validate basic chrome trace structure.

        Checks that RUNTIME and DRIVER events exist, flow arrows link them,
        and timestamps/durations are valid.

        Args:
            prof: The PyTorch profiler instance (after context exit).
            output_dir: Directory to write the trace JSON file.

        Returns:
            dict: The parsed chrome trace JSON.
        """
        trace_file = os.path.join(output_dir, "neuron_trace.json")
        prof.export_chrome_trace(trace_file)
        assert os.path.exists(trace_file)
        assert os.path.getsize(trace_file) > 0

        with open(trace_file) as f:
            trace = json.load(f)

        events = trace["traceEvents"]
        trace_events = [e for e in events if e.get("ph") == "X"]
        diag = self._trace_diagnostics(trace)

        # RUNTIME and DRIVER events exist
        runtime_evts = [e for e in trace_events if e.get("cat") == "privateuse1_runtime"]
        driver_evts = [e for e in trace_events if e.get("cat") == "privateuse1_driver"]
        assert len(runtime_evts) > 0, f"No RUNTIME events in trace. {diag}"
        assert len(driver_evts) > 0, f"No DRIVER events in trace. {diag}"

        # Flow arrows connect RUNTIME -> DRIVER
        flow_starts = [e for e in events if e.get("ph") == "s" and e.get("cat") == "ac2g"]
        flow_ends = [e for e in events if e.get("ph") == "f" and e.get("cat") == "ac2g"]
        assert len(flow_starts) > 0, f"No flow start events. {diag}"
        start_ids = {e["id"] for e in flow_starts}
        end_ids = {e["id"] for e in flow_ends}
        assert start_ids & end_ids, "No matching flow IDs between start/end"

        # Valid timestamps and non-negative durations
        for evt in runtime_evts + driver_evts:
            assert evt.get("ts", 0) > 0, f"Zero ts: {evt['name']}"
            assert evt.get("dur", -1) >= 0, f"Negative dur: {evt['name']}"

        return trace

    def _run_neuron_workload(self):
        """Run a simple workload on Neuron device."""
        x = torch.randn(10, 5, device="neuron")
        linear = torch.nn.Linear(5, 3).to("neuron")
        _ = linear(x)
        torch.neuron.synchronize()

    @staticmethod
    def _trace_diagnostics(trace):
        """Build diagnostic string for trace assertion failures."""
        events = trace["traceEvents"]
        all_cats = sorted({e.get("cat", "<no cat>") for e in events if e.get("ph") == "X"})
        all_phs = sorted({e.get("ph", "<no ph>") for e in events})
        meta_names = sorted({e.get("name", "") for e in events if e.get("ph") == "M"})
        return (
            f"Total events: {len(events)}, "
            f"X-event categories: {all_cats}, "
            f"Phase types: {all_phs}, "
            f"Metadata names: {meta_names}"
        )

    def test_no_activities(self):
        """Case 1: profile(activities=[]) falls back to supported_activities() defaults.

        In PyTorch, an empty activities list is falsy, so the profile() constructor
        treats it the same as activities=None and uses supported_activities() defaults.
        This means CPU is included and framework traces are captured.
        """
        # activities=[] is falsy in Python, so profile() uses supported_activities()
        with profile(activities=[]) as prof, record_function("model_inference"):
            x = torch.randn(10, 5)
            _ = x + x

        # With defaults (includes CPU), framework traces are captured
        self._assert_has_framework_traces(prof)
        self._assert_no_nrt_output(self.test_dir)

    def test_cpu_only(self):
        """Case 2: activities=[CPU], no NeuronConfig -> framework traces only."""
        with (
            profile(activities=[ProfilerActivity.CPU]) as prof,
            record_function("model_inference"),
        ):
            x = torch.randn(10, 5)
            _ = x + x

        self._assert_has_framework_traces(prof)
        self._assert_no_nrt_output(self.test_dir)

    def test_privateuse1_no_config(self):
        """Case 3: activities=[PrivateUse1], no NeuronConfig -> no traces.

        Without NeuronConfig (no "neuron_profile:" prefix), the Neuron plugin's
        configure() returns nullptr and no NRT session is created. Without CPU
        in activities, no RecordFunction callbacks are registered.
        """
        with (
            profile(activities=[ProfilerActivity.PrivateUse1]) as prof,
            record_function("model_inference"),
        ):
            x = torch.randn(10, 5)
            _ = x + x

        self._assert_no_framework_traces(prof)
        self._assert_no_nrt_output(self.test_dir)

    def test_cpu_privateuse1_no_config(self):
        """Case 4: activities=[CPU, PrivateUse1], no NeuronConfig -> framework traces only.

        CPU enables RecordFunction callbacks for framework traces. Without NeuronConfig,
        the Neuron plugin skips session creation (hasNeuronProfileKey returns false).
        """
        with (
            profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof,
            record_function("model_inference"),
        ):
            x = torch.randn(10, 5)
            _ = x + x

        self._assert_has_framework_traces(prof)
        self._assert_no_nrt_output(self.test_dir)

    def test_runtime_only(self):
        """Per-mode test: RUNTIME only with PrivateUse1 activity.

        ntrace.pb and monitoring files (trace_info.pb, cpu_util.pb, host_mem.pb) should
        be produced (RUNTIME mode). No .ntff files (DEVICE not enabled).
        """
        exp_config = NeuronConfig(
            modes=[ProfileMode.RUNTIME],
            profile_output_dir=self.test_dir,
        )

        with (
            profile(
                activities=[ProfilerActivity.PrivateUse1],
                experimental_config=exp_config,
            ) as prof,
            record_function("model_inference"),
        ):
            self._run_neuron_workload()

        self._assert_no_framework_traces(prof)
        self._validate_nrt_output_files(
            self.test_dir,
            expect_ntrace=True,
        )
        self._validate_chrome_trace_basics(prof, self.test_dir)

    def test_device_only(self):
        """Per-mode test: DEVICE only with PrivateUse1 activity.

        .ntff files should be produced (DEVICE mode). No ntrace.pb or monitoring files
        (RUNTIME not enabled).
        """
        exp_config = NeuronConfig(
            modes=[ProfileMode.DEVICE],
            profile_output_dir=self.test_dir,
        )

        with (
            profile(
                activities=[ProfilerActivity.PrivateUse1],
                experimental_config=exp_config,
            ) as prof,
            record_function("model_inference"),
        ):
            self._run_neuron_workload()

        self._assert_no_framework_traces(prof)
        self._validate_nrt_output_files(
            self.test_dir,
            expect_ntff=True,
        )
        self._validate_chrome_trace_basics(prof, self.test_dir)

    def test_cpu_and_runtime_no_device(self):
        """Per-mode test: CPU + PrivateUse1 with RUNTIME mode only.

        Framework traces captured (CPU in activities). ntrace.pb and monitoring files
        produced (RUNTIME mode), no .ntff files (DEVICE not enabled). The common
        "trace runtime without device overhead" use case.
        """
        exp_config = NeuronConfig(
            modes=[ProfileMode.RUNTIME],
            profile_output_dir=self.test_dir,
        )

        with (
            profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
                experimental_config=exp_config,
            ) as prof,
            record_function("model_inference"),
        ):
            self._run_neuron_workload()

        self._assert_has_framework_traces(prof)
        self._validate_nrt_output_files(
            self.test_dir,
            expect_ntrace=True,
        )
        self._validate_chrome_trace_basics(prof, self.test_dir)

    def test_device_and_runtime(self):
        """Per-mode test: DEVICE + RUNTIME (NeuronConfig defaults) without CPU.

        Both .ntff and ntrace.pb (plus monitoring files) should be produced.
        This is the most common user config (NeuronConfig() defaults).
        """
        exp_config = NeuronConfig(
            modes=[ProfileMode.DEVICE, ProfileMode.RUNTIME],
            profile_output_dir=self.test_dir,
        )

        with (
            profile(
                activities=[ProfilerActivity.PrivateUse1],
                experimental_config=exp_config,
            ) as prof,
            record_function("model_inference"),
        ):
            self._run_neuron_workload()

        self._assert_no_framework_traces(prof)
        self._validate_nrt_output_files(
            self.test_dir,
            expect_ntff=True,
            expect_ntrace=True,
        )
        self._validate_chrome_trace_basics(prof, self.test_dir)

    def test_neuron_profiling(self):
        """Comprehensive hardware profiling test.

        Single NRT session with all modes enabled. Verifies:
        - Framework traces (CPU activity)
        - NRT output directory structure (protobuf + NTFF files)
        - Chrome trace export with RUNTIME/DRIVER events
        - Flow arrows linking RUNTIME -> DRIVER
        - Event metadata (nrt_sequence_id, stream_id, pytorch_logical_tid, neuron_worker_tid)
        - RUNTIME/DRIVER on separate process tracks
        - RUNTIME TID matches CPU op TID (same PyTorch thread)
        - NeuronDevice process metadata
        - Valid timestamps and non-negative durations
        """
        exp_config = NeuronConfig(
            modes=[
                ProfileMode.DEVICE,
                ProfileMode.RUNTIME,
                ProfileMode.CPU_UTIL,
                ProfileMode.HOST_MEMORY,
            ],
            profile_output_dir=self.test_dir,
            max_events_per_nc=100000,
            capture_enabled_for_nc="0,1",
        )

        activities = [ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]
        with (
            profile(activities=activities, experimental_config=exp_config) as prof,
            record_function("model_inference"),
        ):
            self._run_neuron_workload()

        # Framework traces (CPU in activities)
        self._assert_has_framework_traces(prof)

        # NRT output structure
        self._validate_nrt_output_structure(self.test_dir)

        # Chrome trace export
        trace_file = os.path.join(self.test_dir, "neuron_trace.json")
        prof.export_chrome_trace(trace_file)
        assert os.path.exists(trace_file)
        assert os.path.getsize(trace_file) > 0

        with open(trace_file) as f:
            trace = json.load(f)

        events = trace["traceEvents"]
        trace_events = [e for e in events if e.get("ph") == "X"]

        # RUNTIME and DRIVER events exist
        runtime_evts = [e for e in trace_events if e.get("cat") == "privateuse1_runtime"]
        driver_evts = [e for e in trace_events if e.get("cat") == "privateuse1_driver"]
        diag = self._trace_diagnostics(trace)
        assert len(runtime_evts) > 0, f"No RUNTIME events in trace. {diag}"
        assert len(driver_evts) > 0, f"No DRIVER events in trace. {diag}"

        # Flow arrows connect RUNTIME -> DRIVER
        flow_starts = [e for e in events if e.get("ph") == "s" and e.get("cat") == "ac2g"]
        flow_ends = [e for e in events if e.get("ph") == "f" and e.get("cat") == "ac2g"]
        assert len(flow_starts) > 0, "No flow start events"
        start_ids = {e["id"] for e in flow_starts}
        end_ids = {e["id"] for e in flow_ends}
        assert start_ids & end_ids, "No matching flow IDs between start/end"

        # RUNTIME event metadata
        for evt in runtime_evts:
            args = evt.get("args", {})
            assert "nrt_sequence_id" in args
            assert "stream_id" in args
            assert "pytorch_logical_tid" in args
            assert "neuron_worker_tid" in args

        # DRIVER event metadata
        for evt in driver_evts:
            assert "nrt_sequence_id" in evt.get("args", {})

        # RUNTIME/DRIVER on different process tracks
        runtime_pids = {e["pid"] for e in runtime_evts}
        driver_pids = {e["pid"] for e in driver_evts}
        assert runtime_pids.isdisjoint(driver_pids)

        # RUNTIME TID matches CPU op TID
        cpu_ops = [e for e in trace_events if e.get("cat") == "cpu_op"]
        if cpu_ops:
            cpu_tids = {e["tid"] for e in cpu_ops}
            runtime_tids = {e["tid"] for e in runtime_evts}
            assert runtime_tids & cpu_tids

        # NeuronDevice process metadata
        device_meta = [
            e
            for e in events
            if e.get("ph") == "M"
            and e.get("name") == "process_labels"
            and "NeuronDevice" in str(e.get("args", {}).get("labels", ""))
        ]
        assert len(device_meta) > 0, "No NeuronDevice process metadata"

        # Valid timestamps and durations
        for evt in runtime_evts + driver_evts:
            assert evt.get("ts", 0) > 0, f"Zero ts: {evt['name']}"
            assert evt.get("dur", -1) >= 0, f"Negative dur: {evt['name']}"

    def test_schedule(self):
        """Schedule-driven profiling with two cycles.

        Exercises the session lifecycle across multiple cycles:
        schedule(wait=1, warmup=1, active=1, repeat=2) produces 6 steps:

        Cycle 1: step 0 (NONE), step 1 (WARMUP), step 2 (RECORD_AND_SAVE)
          → stop_trace, on_trace_ready fires, session #1 destroyed
        Cycle 2: step 3 (NONE), step 4 (WARMUP), step 5 (RECORD_AND_SAVE)
          → __exit__: stop_trace, on_trace_ready fires, session #2 destroyed

        The critical transition is step 2→3 where session #1 is torn down, then
        step 3→4 where session #2 is created. This validates that the
        s_sessionActive atomic guard is properly released between cycles.
        """
        output_dir = self.test_dir

        cycle_0_dir = os.path.join(output_dir, "cycle_0")
        os.makedirs(cycle_0_dir, exist_ok=True)
        exp_config = NeuronConfig(profile_output_dir=cycle_0_dir)
        activities = [ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]
        traces_captured = []
        num_cycles = 2

        def on_trace_ready(prof):
            cycle_dir = os.path.join(output_dir, f"cycle_{len(traces_captured)}")
            trace_file = os.path.join(cycle_dir, "trace.json")
            prof.export_chrome_trace(trace_file)
            with open(trace_file) as f:
                traces_captured.append(json.load(f))
            # Point next cycle's NRT output at its own subdirectory (skip on last cycle)
            if len(traces_captured) < num_cycles:
                next_dir = os.path.join(output_dir, f"cycle_{len(traces_captured)}")
                os.makedirs(next_dir, exist_ok=True)
                prof.experimental_config = NeuronConfig(profile_output_dir=next_dir)

        step_labels = [
            "cycle1_step0_wait",
            "cycle1_step1_warmup",
            "cycle1_step2_active",
            "cycle2_step3_wait",
            "cycle2_step4_warmup",
            "cycle2_step5_active",
        ]

        with profile(
            activities=activities,
            schedule=schedule(wait=1, warmup=1, active=1, repeat=num_cycles),
            on_trace_ready=on_trace_ready,
            experimental_config=exp_config,
        ) as prof:
            for step_idx in range(6):
                with record_function(step_labels[step_idx]):
                    self._run_neuron_workload()
                if step_idx < 5:
                    prof.step()

        # on_trace_ready fired exactly twice (once per cycle)
        assert len(traces_captured) == 2, f"Expected 2 trace captures, got {len(traces_captured)}"

        # Both cycles produced valid RUNTIME and DRIVER events
        for cycle_idx, trace in enumerate(traces_captured):
            events = trace["traceEvents"]
            trace_events = [e for e in events if e.get("ph") == "X"]
            diag = self._trace_diagnostics(trace)

            runtime_evts = [e for e in trace_events if e.get("cat") == "privateuse1_runtime"]
            driver_evts = [e for e in trace_events if e.get("cat") == "privateuse1_driver"]
            assert len(runtime_evts) > 0, f"Cycle {cycle_idx}: No RUNTIME events. {diag}"
            assert len(driver_evts) > 0, f"Cycle {cycle_idx}: No DRIVER events. {diag}"

            # Valid timestamps and durations
            for evt in runtime_evts + driver_evts:
                assert evt.get("ts", 0) > 0, f"Cycle {cycle_idx}: Zero ts: {evt['name']}"
                assert evt.get("dur", -1) >= 0, f"Cycle {cycle_idx}: Negative dur: {evt['name']}"

        # Helper to collect all event names from a trace
        def event_names(trace):
            return {e.get("name") for e in trace["traceEvents"]}

        # Active-phase labels appear in the corresponding cycle's trace
        assert "cycle1_step2_active" in event_names(
            traces_captured[0]
        ), "Cycle 0 trace missing active-phase label 'cycle1_step2_active'"
        assert "cycle2_step5_active" in event_names(
            traces_captured[1]
        ), "Cycle 1 trace missing active-phase label 'cycle2_step5_active'"

        # Wait-phase labels should NOT appear in either trace
        all_names = event_names(traces_captured[0]) | event_names(traces_captured[1])
        assert (
            "cycle1_step0_wait" not in all_names
        ), "Wait-phase label 'cycle1_step0_wait' should not appear in any trace"
        assert (
            "cycle2_step3_wait" not in all_names
        ), "Wait-phase label 'cycle2_step3_wait' should not appear in any trace"


class TestProfileMode:
    """Test cases for ProfileMode enum."""

    def test_profile_mode_values(self):
        """Test that ProfileMode has expected values."""
        assert ProfileMode.DEVICE.value == "device"
        assert ProfileMode.RUNTIME.value == "runtime"
        assert ProfileMode.CPU_UTIL.value == "cpu_util"
        assert ProfileMode.HOST_MEMORY.value == "host_memory"


class TestNeuronConfig:
    """Test cases for NeuronConfig class."""

    def test_default_modes(self):
        """Test NeuronConfig() defaults to device + runtime."""
        config = NeuronConfig()
        assert config.custom_profiler_config == "neuron_profile:device,runtime"

    def test_device_only(self):
        """Test NeuronConfig with device mode only."""
        config = NeuronConfig(modes=[ProfileMode.DEVICE])
        assert config.custom_profiler_config == "neuron_profile:device"

    def test_runtime_only(self):
        """Test NeuronConfig with runtime mode only."""
        config = NeuronConfig(modes=[ProfileMode.RUNTIME])
        assert config.custom_profiler_config == "neuron_profile:runtime"

    def test_all_modes(self):
        """Test NeuronConfig with all modes."""
        config = NeuronConfig(
            modes=[
                ProfileMode.DEVICE,
                ProfileMode.RUNTIME,
                ProfileMode.CPU_UTIL,
                ProfileMode.HOST_MEMORY,
            ]
        )
        assert config.custom_profiler_config == "neuron_profile:device,runtime,cpu_util,host_memory"

    def test_empty_modes_raises_value_error(self):
        """Test that empty modes list raises ValueError."""
        with pytest.raises(ValueError, match="At least one ProfileMode must be specified"):
            NeuronConfig(modes=[])

    def test_invalid_mode_type_raises_type_error(self):
        """Test that non-ProfileMode values raise TypeError."""
        with pytest.raises(TypeError, match="Expected ProfileMode"):
            NeuronConfig(modes=["invalid"])

    def test_max_events_per_nc(self):
        """Test NeuronConfig with max_events_per_nc parameter."""
        config = NeuronConfig(max_events_per_nc=100000)
        assert "neuron_profile:device,runtime" in config.custom_profiler_config
        assert "max_events_per_nc:100000" in config.custom_profiler_config

    def test_capture_enabled_for_nc(self):
        """Test NeuronConfig with capture_enabled_for_nc parameter."""
        config = NeuronConfig(capture_enabled_for_nc="0,1,2-5")
        assert "neuron_profile:device,runtime" in config.custom_profiler_config
        assert "capture_enabled_for_nc:0,1,2-5" in config.custom_profiler_config

    def test_multiple_nrt_params(self):
        """Test NeuronConfig with multiple NRT config parameters."""
        config = NeuronConfig(
            modes=[ProfileMode.DEVICE],
            max_events_per_nc=100000,
            profile_output_dir="/tmp/traces",
        )
        parts = config.custom_profiler_config.split(";")
        assert parts[0] == "neuron_profile:device"
        assert "max_events_per_nc:100000" in parts
        assert "profile_output_dir:/tmp/traces" in parts

    def test_profile_output_dir(self):
        """Test NeuronConfig with profile_output_dir parameter."""
        config = NeuronConfig(profile_output_dir="/tmp/traces")
        assert "neuron_profile:device,runtime" in config.custom_profiler_config
        assert "profile_output_dir:/tmp/traces" in config.custom_profiler_config

    def test_all_nrt_params(self):
        """Test NeuronConfig with all NRT config parameters specified."""
        config = NeuronConfig(
            modes=[ProfileMode.DEVICE, ProfileMode.RUNTIME],
            max_events_per_nc=50000,
            capture_enabled_for_nc="0,1",
            profile_output_dir="/tmp/output",
        )
        parts = config.custom_profiler_config.split(";")
        assert parts[0] == "neuron_profile:device,runtime"
        assert "max_events_per_nc:50000" in parts
        assert "capture_enabled_for_nc:0,1" in parts
        assert "profile_output_dir:/tmp/output" in parts

    def test_isinstance_experimental_config(self):
        """Test that NeuronConfig is an instance of _ExperimentalConfig."""
        config = NeuronConfig()
        assert isinstance(config, _ExperimentalConfig)
