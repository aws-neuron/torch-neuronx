"""
Unified profiling that provides following three traces:
1. PyTorch trace
2. Neuron Runtime system trace
3. Neuron device trace
This module provides NeuronConfig and NeuronProfiler for use with
torch.profiler.profile(), offering profiling coverage for Neuron workloads.
"""

import logging
import os
from enum import Enum
from typing import ClassVar

from torch._C._profiler import _ExperimentalConfig

import torch_neuronx._C as _C

logger = logging.getLogger(__name__)


class ProfileMode(Enum):
    """Neuron profiling modes for use with NeuronConfig.

    Similar to torch.profiler.ProfilerActivity but for Neuron-specific
    profiling granularity.
    """

    DEVICE = "device"
    RUNTIME = "runtime"
    CPU_UTIL = "cpu_util"
    HOST_MEMORY = "host_memory"


class NeuronConfig(_ExperimentalConfig):
    """Neuron profiler configuration for use with torch.profiler.profile().

    Subclasses _ExperimentalConfig so it can be passed directly to
    experimental_config=. The neuron_profile key in custom_profiler_config
    controls which NRT profiling modes are enabled.

    Args:
        modes: List of ProfileMode values to enable.
            Defaults to [DEVICE, RUNTIME].
        max_events_per_nc: Maximum number of trace events per NeuronCore.
        capture_enabled_for_nc: Comma-separated NeuronCore indices/ranges to capture
            (e.g., "0,1,2-5").
        profile_output_dir: Directory for NRT trace output files.
        precache_node_info: When True, pre-cache expensive per-NEFF metadata
            during wait iterations so the actively profiled iteration has reduced
            overhead. Useful with torch.profiler.schedule(wait=N, warmup=M, active=K).
        neff_cache_dir: Path to the NEFF cache directory. When set, the cached
            NEFFs will be copied into the profile artifact directory when stopping
            the profile instead of creating an additional copy on load.
    """

    _DEFAULT_MODES: ClassVar[list[ProfileMode]] = [ProfileMode.DEVICE, ProfileMode.RUNTIME]

    def __init__(
        self,
        modes: list[ProfileMode] | None = None,
        max_events_per_nc: int | None = None,
        capture_enabled_for_nc: str | None = None,
        profile_output_dir: str | None = None,
        precache_node_info: bool = True,
        neff_cache_dir: str | None = None,
    ):
        if modes is None:
            modes = self._DEFAULT_MODES

        if not modes:
            raise ValueError("At least one ProfileMode must be specified")

        for mode in modes:
            if not isinstance(mode, ProfileMode):
                raise TypeError(f"Expected ProfileMode, got {type(mode).__name__}: {mode}")

        # by default, check if the user explicitly specified the cache,
        # if not neff_cache_dir remains None
        if neff_cache_dir is None:
            neff_cache_dir = os.environ.get("TORCH_NEURONX_NEFF_CACHE_DIR")

        # IMPORTANT: neuron_profile must be the FIRST key -- the C++ hasNeuronProfileKey()
        # in Profiler.cpp uses a prefix check (rfind(kKey, 0) == 0) for fast gating.
        profile_str = ",".join(m.value for m in modes)
        parts = [f"neuron_profile:{profile_str}"]
        if max_events_per_nc is not None:
            parts.append(f"max_events_per_nc:{max_events_per_nc}")
        if capture_enabled_for_nc is not None:
            parts.append(f"capture_enabled_for_nc:{capture_enabled_for_nc}")
        if profile_output_dir is not None:
            parts.append(f"profile_output_dir:{profile_output_dir}")
        if neff_cache_dir is not None:
            parts.append(f"neff_cache_dir:{neff_cache_dir}")
        config_str = ";".join(parts)
        super().__init__(custom_profiler_config=config_str)
        self.custom_profiler_config = config_str
        self.profile_output_dir = profile_output_dir
        self.neff_cache_dir = neff_cache_dir

        # Pre-cache profile node info during warmup iterations so that the
        # actively profiled iteration doesn't pay the cost of collecting
        # instructions, DMA descriptors, and CC info from the device.
        if precache_node_info:
            _C._nrt_inspect_precache_enable()

    def get_session_dir(self) -> str:
        """Get the NRT instance output directory for the current profiling session.

        Must be called after profiling has started (i.e. after the profile context
        manager has entered), as the directory is created by nrt_inspect_begin.

        Returns:
            Path to the instance-specific output directory.
        """
        if self.profile_output_dir is None:
            raise ValueError("profile_output_dir was not set on this NeuronConfig")
        return _C._nrt_inspect_get_instance_output_dir(self.profile_output_dir)


class NeuronProfiler:
    """Exports PyTorch chrome traces into the NRT session output directory.

    Wraps the time-dependent session directory lookup so callers never need
    to reason about when to query it.

    Args:
        config: A NeuronConfig whose profile_output_dir is set.

    Example:
        config = NeuronConfig(profile_output_dir="./output")
        profiler = NeuronProfiler(config)

        # As on_trace_ready callback
        with profile(..., on_trace_ready=lambda p: profiler.export_trace(p)):
            ...

        # Inside a custom hook
        def my_hook(prof):
            # custom logic
            profiler.export_trace(prof)
        with profile(..., on_trace_ready=my_hook):
            ...

        # Or called directly after profiling
        with profile(...) as prof:
            ...
        profiler.export_trace(prof)
    """

    def __init__(self, config: NeuronConfig):
        if config.profile_output_dir is None:
            raise ValueError("profile_output_dir must be set on the NeuronConfig")
        self._config = config

    def export_trace(self, prof, filename: str = "trace.json"):
        """Export a PyTorch chrome trace into the current NRT session directory.

        Args:
            prof: The torch.profiler.profile instance.
            filename: Trace file name within the session directory.
        """
        session_dir = self._config.get_session_dir()
        prof.export_chrome_trace(os.path.join(session_dir, filename))


class NRTProfilerError(Exception):
    """Exception raised for NRT profiling errors."""

    pass
