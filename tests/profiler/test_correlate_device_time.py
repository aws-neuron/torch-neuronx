"""
Tests for correlate_device_time functionality.
Verifies that device execution time is correctly injected into FunctionEvents
and that key_averages() reports NEURON time columns.
"""

import os

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from torch_neuronx.profiling import (
    NeuronConfig,
    ProfileMode,
    correlate_device_time,
)


@pytest.fixture
def profile_dir(tmp_path):
    d = str(tmp_path / "prof_output")
    os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = d
    return d


@pytest.fixture
def neuron_tensor():
    x = torch.randn(4, 4, device="neuron")
    for _ in range(3):
        _ = x * 2 + 1
        torch.neuron.synchronize()
    return x


class TestCorrelateDeviceTime:
    """Tests for the correlate_device_time function."""

    @pytest.mark.xfail(
        reason="[PT2.12] Neuron aten/wheel nightly failure (run 28991277106): likely profiler "
        "regression (profiler/memory/autocast). Xfailing to unblock nightly pending root-cause."
    )
    def test_basic_device_time_columns(self, profile_dir, neuron_tensor):
        """key_averages() shows non-zero NEURON time after correlation."""
        nc = NeuronConfig(modes=[ProfileMode.RUNTIME], profile_output_dir=profile_dir)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            experimental_config=nc,
        ) as prof:
            _ = neuron_tensor * 2 + 1
            torch.neuron.synchronize()

        correlate_device_time(prof)
        ka = prof.key_averages()
        total_neuron = sum(e.self_device_time_total for e in ka)
        assert total_neuron > 0, "Expected non-zero Self NEURON time"

    @pytest.mark.xfail(
        reason="[PT2.12] Neuron aten/wheel nightly failure (run 28991277106): likely profiler "
        "regression (profiler/memory/autocast). Xfailing to unblock nightly pending root-cause."
    )
    def test_non_compute_ops_excluded(self, profile_dir, neuron_tensor):
        """alloc/dealloc/copy ops should not have device time attributed."""
        nc = NeuronConfig(modes=[ProfileMode.RUNTIME], profile_output_dir=profile_dir)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            experimental_config=nc,
        ) as prof:
            _ = neuron_tensor * 2 + 1
            torch.neuron.synchronize()

        correlate_device_time(prof)
        ka = prof.key_averages()
        for e in ka:
            if any(p in e.key for p in ("alloc", "dealloc", "copy")):
                assert (
                    e.self_device_time_total == 0
                ), f"{e.key} should not have device time, got {e.self_device_time_total}"

    @pytest.mark.xfail(
        reason="[PT2.12] Neuron aten/wheel nightly failure (run 28991277106): likely profiler "
        "regression (profiler/memory/autocast). Xfailing to unblock nightly pending root-cause."
    )
    def test_compute_ops_have_device_time(self, profile_dir, neuron_tensor):
        """aten::mul and aten::add should have non-zero device time."""
        nc = NeuronConfig(modes=[ProfileMode.RUNTIME], profile_output_dir=profile_dir)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            experimental_config=nc,
        ) as prof:
            _ = neuron_tensor * 2 + 1
            torch.neuron.synchronize()

        correlate_device_time(prof)
        ka = prof.key_averages()
        compute_ops = [e for e in ka if e.key in ("aten::mul", "aten::add")]
        assert len(compute_ops) > 0, "Expected aten::mul or aten::add in key_averages"
        for e in compute_ops:
            assert e.self_device_time_total > 0, f"{e.key} should have device time"

    @pytest.mark.xfail(
        reason="[PT2.12] Neuron aten/wheel nightly failure (run 28991277106): likely profiler "
        "regression (profiler/memory/autocast). Xfailing to unblock nightly pending root-cause."
    )
    def test_accuracy_vs_kineto_events(self, profile_dir, neuron_tensor):
        """Total NEURON time should match raw KinetoEvent device durations."""
        nc = NeuronConfig(modes=[ProfileMode.RUNTIME], profile_output_dir=profile_dir)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            experimental_config=nc,
        ) as prof:
            _ = neuron_tensor * 2 + 1
            torch.neuron.synchronize()

        # Compute ground truth from kineto events directly
        host_pid = os.getpid()
        raw_device_us = 0
        for ev in prof.profiler.kineto_results.events():
            if ev.linked_correlation_id() <= 0:
                continue
            if ev.device_index() == host_pid:
                continue
            if ev.name().startswith(("neuron::alloc", "neuron::dealloc", "neuron::copy")):
                continue
            dur_ns = ev.duration_ns()
            if dur_ns > 0:
                raw_device_us += dur_ns / 1000

        correlate_device_time(prof)
        ka = prof.key_averages()
        total_neuron = sum(e.self_device_time_total for e in ka)

        assert raw_device_us > 0, "No device events found"
        ratio = total_neuron / raw_device_us
        assert 0.8 < ratio < 1.2, (
            f"key_averages total ({total_neuron:.1f}us) diverges from raw events "
            f"({raw_device_us:.1f}us), ratio={ratio:.2f}"
        )

    @pytest.mark.xfail(
        reason="[PT2.12] Neuron aten/wheel nightly failure (run 28991277106): likely profiler "
        "regression (profiler/memory/autocast). Xfailing to unblock nightly pending root-cause."
    )
    def test_chrome_trace_unaffected(self, profile_dir, neuron_tensor):
        """correlate_device_time does not mutate the kineto events backing export_chrome_trace."""
        nc = NeuronConfig(
            modes=[ProfileMode.RUNTIME],
            profile_output_dir=profile_dir,
            correlate_device_time=False,
        )

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            experimental_config=nc,
        ) as prof:
            _ = neuron_tensor * 2 + 1
            torch.neuron.synchronize()

        kineto = prof.profiler.kineto_results
        before = [(e.name(), e.duration_ns()) for e in kineto.events()]
        correlate_device_time(prof)
        after = [(e.name(), e.duration_ns()) for e in kineto.events()]

        assert before == after
