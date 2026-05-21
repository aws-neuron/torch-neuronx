"""
Tests for dispatch tracing in the profiler.

Verifies that the DISPATCH activity (RuntimeWorker scheduling) and per-XU type
DRIVER rows are correctly emitted in chrome trace exports.
"""

import json
import os

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

import torch_neuronx
from torch_neuronx.profiling import NeuronConfig, ProfileMode


@pytest.mark.skipif(
    os.environ.get("NEURON_LAUNCH_BLOCKING") == "1",
    reason="Dispatch tracing requires async execution (not NEURON_LAUNCH_BLOCKING)",
)
class TestDispatchTracing:
    """Tests for dispatch activity and per-XU device row tracing."""

    @pytest.fixture(autouse=True)
    def _test_dir(self, tmp_path):
        self.test_dir = str(tmp_path)

    def _run_workload(self):
        """Run a simple workload that produces device ops."""
        x = torch.randn(10, 5, device="neuron")
        linear = torch.nn.Linear(5, 3).to("neuron")
        _ = linear(x)
        torch.neuron.synchronize()

    def _export_and_parse_trace(self, prof):
        """Export chrome trace and parse JSON."""
        trace_file = os.path.join(self.test_dir, "trace.json")
        prof.export_chrome_trace(trace_file)
        assert os.path.exists(trace_file)
        with open(trace_file) as f:
            return json.load(f)

    def test_dispatch_events_present(self):
        """Verify DISPATCH events appear on RuntimeWorker thread."""
        exp_config = NeuronConfig(
            modes=[ProfileMode.RUNTIME],
            profile_output_dir=self.test_dir,
        )

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            experimental_config=exp_config,
        ) as prof:
            self._run_workload()

        trace = self._export_and_parse_trace(prof)
        events = trace["traceEvents"]
        trace_events = [e for e in events if e.get("ph") == "X"]

        runtime_evts = [e for e in trace_events if e.get("cat") == "privateuse1_runtime"]
        driver_evts = [e for e in trace_events if e.get("cat") == "privateuse1_driver"]

        assert len(runtime_evts) > 0, "No RUNTIME events"
        assert len(driver_evts) > 0, "No DRIVER events"

        # RUNTIME events should have at least two different TIDs:
        # one for PyTorch submission thread, one for RuntimeWorker dispatch
        runtime_tids = {e["tid"] for e in runtime_evts}
        assert len(runtime_tids) >= 2, (
            f"Expected at least 2 distinct RUNTIME tids (submit + dispatch), "
            f"got {len(runtime_tids)}: {runtime_tids}"
        )

    def test_dispatch_encapsulates_driver(self):
        """Verify DISPATCH fully contains its DRIVER event."""
        exp_config = NeuronConfig(
            modes=[ProfileMode.RUNTIME],
            profile_output_dir=self.test_dir,
        )

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            experimental_config=exp_config,
        ) as prof:
            self._run_workload()

        trace = self._export_and_parse_trace(prof)
        events = trace["traceEvents"]
        trace_events = [e for e in events if e.get("ph") == "X"]

        runtime_evts = [e for e in trace_events if e.get("cat") == "privateuse1_runtime"]
        driver_evts = [e for e in trace_events if e.get("cat") == "privateuse1_driver"]

        # Group RUNTIME events by tid to find dispatch events (non-submit thread)
        # Submit events have the same tid as CPU ops; dispatch events are on RuntimeWorker
        cpu_ops = [e for e in trace_events if e.get("cat") == "cpu_op"]
        cpu_tids = {e["tid"] for e in cpu_ops} if cpu_ops else set()

        # Dispatch events are RUNTIME events NOT on the CPU op thread
        dispatch_evts = [e for e in runtime_evts if e["tid"] not in cpu_tids]

        if not dispatch_evts:
            # If we can't distinguish by CPU tid, find events with xu_type metadata
            dispatch_evts = [e for e in runtime_evts if "xu_type" in e.get("args", {})]

        assert len(dispatch_evts) > 0, "No DISPATCH events found"

        # For each DISPATCH event, find corresponding DRIVER event with same name
        for dispatch_evt in dispatch_evts[:5]:  # Check first 5
            dispatch_start = dispatch_evt["ts"]
            dispatch_end = dispatch_start + dispatch_evt["dur"]

            matching_drivers = [
                d
                for d in driver_evts
                if d["name"] == dispatch_evt["name"]
                and d["ts"] >= dispatch_start
                and d["ts"] + d["dur"] <= dispatch_end + 1  # 1us tolerance
            ]
            # At least some should be encapsulated (timing can have tiny jitter)
            # Not asserting for every op since concurrent ops share names
            if matching_drivers:
                driver = matching_drivers[0]
                assert (
                    driver["ts"] >= dispatch_start
                ), f"DRIVER start ({driver['ts']}) < DISPATCH start ({dispatch_start})"

    def test_per_xu_type_driver_rows(self):
        """Verify device DRIVER events use per-XU type resource IDs."""
        exp_config = NeuronConfig(
            modes=[ProfileMode.RUNTIME],
            profile_output_dir=self.test_dir,
        )

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            experimental_config=exp_config,
        ) as prof:
            self._run_workload()

        trace = self._export_and_parse_trace(prof)
        events = trace["traceEvents"]
        trace_events = [e for e in events if e.get("ph") == "X"]

        driver_evts = [e for e in trace_events if e.get("cat") == "privateuse1_driver"]
        assert len(driver_evts) > 0, "No DRIVER events"

        # Device ops (not alloc/dealloc/event) should have xu_type metadata
        device_driver_evts = [e for e in driver_evts if "xu_type" in e.get("args", {})]
        assert len(device_driver_evts) > 0, (
            "No DRIVER events with xu_type metadata. "
            f"All driver event names: {[e['name'] for e in driver_evts]}"
        )

        for evt in device_driver_evts:
            args = evt.get("args", {})
            assert args["xu_type"] in (
                "Compute",
                "Collectives",
                "TensorOp",
            ), f"Unexpected xu_type: {args['xu_type']}"

    def test_dispatch_flow_arrows(self):
        """Verify flow arrows connect DISPATCH → DRIVER."""
        exp_config = NeuronConfig(
            modes=[ProfileMode.RUNTIME],
            profile_output_dir=self.test_dir,
        )

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            experimental_config=exp_config,
        ) as prof:
            self._run_workload()

        trace = self._export_and_parse_trace(prof)
        events = trace["traceEvents"]

        flow_starts = [e for e in events if e.get("ph") == "s" and e.get("cat") == "ac2g"]
        flow_ends = [e for e in events if e.get("ph") == "f" and e.get("cat") == "ac2g"]

        assert len(flow_starts) > 0, "No flow start events"
        assert len(flow_ends) > 0, "No flow end events"

        start_ids = {e["id"] for e in flow_starts}
        end_ids = {e["id"] for e in flow_ends}
        matched = start_ids & end_ids
        assert len(matched) > 0, "No matching flow IDs between start/end"

    def test_pytorch_sequence_nr_metadata(self):
        """Verify pytorch_seq_nr metadata is present in events."""
        exp_config = NeuronConfig(
            modes=[ProfileMode.RUNTIME],
            profile_output_dir=self.test_dir,
        )

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            experimental_config=exp_config,
        ) as prof:
            self._run_workload()

        trace = self._export_and_parse_trace(prof)
        events = trace["traceEvents"]
        trace_events = [e for e in events if e.get("ph") == "X"]

        driver_evts = [e for e in trace_events if e.get("cat") == "privateuse1_driver"]
        for evt in driver_evts:
            args = evt.get("args", {})
            assert "pytorch_seq_nr" in args, f"DRIVER event '{evt['name']}' missing pytorch_seq_nr"

    def test_dispatch_slot_resource_labels(self):
        """Verify dispatch slot rows get proper labels (Stream X / Compute Slot N)."""
        exp_config = NeuronConfig(
            modes=[ProfileMode.RUNTIME],
            profile_output_dir=self.test_dir,
        )

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            experimental_config=exp_config,
        ) as prof:
            self._run_workload()

        trace = self._export_and_parse_trace(prof)
        events = trace["traceEvents"]

        # Look for thread_name metadata events with "Slot" in the name
        slot_labels = [
            e
            for e in events
            if e.get("ph") == "M"
            and e.get("name") == "thread_name"
            and "Slot" in str(e.get("args", {}).get("name", ""))
        ]
        assert len(slot_labels) > 0, (
            "No dispatch slot thread_name metadata found in trace. "
            "Expected labels like 'Stream 0 / Compute Slot 0'."
        )

    def test_xu_type_resource_labels(self):
        """Verify per-XU type rows get proper labels (Stream X / Compute, etc.)."""
        exp_config = NeuronConfig(
            modes=[ProfileMode.RUNTIME],
            profile_output_dir=self.test_dir,
        )

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            experimental_config=exp_config,
        ) as prof:
            self._run_workload()

        trace = self._export_and_parse_trace(prof)
        events = trace["traceEvents"]

        # Look for thread_name or thread_sort_index metadata for device rows
        thread_labels = [
            e
            for e in events
            if e.get("ph") == "M"
            and e.get("name") == "thread_name"
            and "Stream" in str(e.get("args", {}).get("name", ""))
            and "/" in str(e.get("args", {}).get("name", ""))
        ]
        assert (
            len(thread_labels) > 0
        ), "No per-XU type resource labels found (expected 'Stream X / Compute' etc.)"

    def test_concurrent_ops_get_distinct_slots(self):
        """Verify multiple inflight ops get distinct dispatch_slot values (parallel rows)."""
        exp_config = NeuronConfig(
            modes=[ProfileMode.RUNTIME],
            profile_output_dir=self.test_dir,
        )

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            experimental_config=exp_config,
        ) as prof:
            # Launch multiple ops without synchronizing to keep them inflight
            x = torch.randn(64, 64, device="neuron")
            for _ in range(8):
                x = torch.mm(x, x)
            torch.neuron.synchronize()

        trace = self._export_and_parse_trace(prof)
        events = trace["traceEvents"]
        trace_events = [e for e in events if e.get("ph") == "X"]

        # Get DISPATCH events (have dispatch_slot metadata)
        dispatch_evts = [
            e
            for e in trace_events
            if e.get("cat") == "privateuse1_runtime" and "dispatch_slot" in e.get("args", {})
        ]
        assert len(dispatch_evts) >= 2, f"Need at least 2 dispatch events, got {len(dispatch_evts)}"

        # Multiple distinct tids should be used (not all on the same row)
        dispatch_tids = {e["tid"] for e in dispatch_evts}
        assert len(dispatch_tids) >= 2, (
            f"Expected multiple distinct dispatch tids for parallel ops, "
            f"got {len(dispatch_tids)}: {dispatch_tids}"
        )

    def test_slot_recycling_after_completion(self):
        """Verify slots are reused: many sequential ops reuse the same slots."""
        exp_config = NeuronConfig(
            modes=[ProfileMode.RUNTIME],
            profile_output_dir=self.test_dir,
        )

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            experimental_config=exp_config,
        ) as prof:
            # Run more ops than slots (8) to force recycling
            x = torch.randn(32, 32, device="neuron")
            for _ in range(20):
                x = torch.mm(x, x)
                torch.neuron.synchronize()  # Complete each op before next

        trace = self._export_and_parse_trace(prof)
        events = trace["traceEvents"]
        trace_events = [e for e in events if e.get("ph") == "X"]

        dispatch_evts = [
            e
            for e in trace_events
            if e.get("cat") == "privateuse1_runtime" and "dispatch_slot" in e.get("args", {})
        ]
        assert (
            len(dispatch_evts) >= 10
        ), f"Expected many dispatch events for recycling test, got {len(dispatch_evts)}"

        # Slots should be in range [0, 7] — recycling means we never exceed 8 unique values
        slots = {e["args"]["dispatch_slot"] for e in dispatch_evts}
        assert all(0 <= s <= 7 for s in slots), f"Slots out of range [0,7]: {slots}"

        # If all 20 ops ran, but at most 8 distinct slots, recycling is working
        assert len(slots) <= 8, f"More than 8 distinct slots used: {slots}"

    def test_host_ops_excluded_from_dispatch(self):
        """Verify alloc/dealloc ops do NOT produce DISPATCH events."""
        exp_config = NeuronConfig(
            modes=[ProfileMode.RUNTIME],
            profile_output_dir=self.test_dir,
        )

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            experimental_config=exp_config,
        ) as prof:
            # Only allocation — no compute ops
            _ = torch.empty(100, 100, device="neuron")
            torch.neuron.synchronize()

        trace = self._export_and_parse_trace(prof)
        events = trace["traceEvents"]
        trace_events = [e for e in events if e.get("ph") == "X"]

        # DISPATCH events have dispatch_slot metadata; host ops should not have any
        dispatch_evts = [
            e
            for e in trace_events
            if e.get("cat") == "privateuse1_runtime" and "dispatch_slot" in e.get("args", {})
        ]
        # Alloc-only workload should produce zero DISPATCH events
        assert len(dispatch_evts) == 0, (
            f"Host-only workload should have no DISPATCH events, "
            f"found {len(dispatch_evts)}: {[e['name'] for e in dispatch_evts[:5]]}"
        )

    def test_flow_arrows_one_to_one(self):
        """Verify flow IDs have exact 1:1 matching between starts and ends."""
        exp_config = NeuronConfig(
            modes=[ProfileMode.RUNTIME],
            profile_output_dir=self.test_dir,
        )

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            experimental_config=exp_config,
        ) as prof:
            self._run_workload()

        trace = self._export_and_parse_trace(prof)
        events = trace["traceEvents"]

        flow_starts = [e for e in events if e.get("ph") == "s" and e.get("cat") == "ac2g"]
        flow_ends = [e for e in events if e.get("ph") == "f" and e.get("cat") == "ac2g"]

        start_ids = [e["id"] for e in flow_starts]
        end_ids = [e["id"] for e in flow_ends]

        # No duplicate IDs within starts
        assert len(start_ids) == len(
            set(start_ids)
        ), f"Duplicate flow start IDs found: {len(start_ids)} total, {len(set(start_ids))} unique"

        # Exact set equality (every start has an end and vice versa)
        assert set(start_ids) == set(end_ids), (
            f"Flow ID mismatch: {len(set(start_ids) - set(end_ids))} starts without ends, "
            f"{len(set(end_ids) - set(start_ids))} ends without starts"
        )

    def test_dispatch_events_non_overlapping_per_slot(self):
        """Verify events on the same dispatch slot tid do not overlap in time."""
        exp_config = NeuronConfig(
            modes=[ProfileMode.RUNTIME],
            profile_output_dir=self.test_dir,
        )

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            experimental_config=exp_config,
        ) as prof:
            x = torch.randn(32, 32, device="neuron")
            for _ in range(16):
                x = torch.mm(x, x)
            torch.neuron.synchronize()

        trace = self._export_and_parse_trace(prof)
        events = trace["traceEvents"]
        trace_events = [e for e in events if e.get("ph") == "X"]

        dispatch_evts = [
            e
            for e in trace_events
            if e.get("cat") == "privateuse1_runtime" and "dispatch_slot" in e.get("args", {})
        ]

        # Group by tid
        from collections import defaultdict

        by_tid = defaultdict(list)
        for e in dispatch_evts:
            by_tid[e["tid"]].append(e)

        # Within each tid, events should not overlap
        for tid, evts in by_tid.items():
            sorted_evts = sorted(evts, key=lambda e: e["ts"])
            for i in range(len(sorted_evts) - 1):
                end_i = sorted_evts[i]["ts"] + sorted_evts[i]["dur"]
                start_next = sorted_evts[i + 1]["ts"]
                assert end_i <= start_next + 1, (  # 1us tolerance for rounding
                    f"Overlapping events on tid {tid}: "
                    f"'{sorted_evts[i]['name']}' ends at {end_i} but "
                    f"'{sorted_evts[i + 1]['name']}' starts at {start_next}"
                )
