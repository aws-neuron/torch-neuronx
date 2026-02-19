"""
Unified profiling that provides following three traces:
1. PyTorch trace
2. Neuron Runtime system trace
3. Neuron device trace
This module provides a NeuronProfiler class that coordinates both NRT
profiling and PyTorch's native profiling capabilities, offering profiling
coverage for Neuron workloads.
"""

import logging
import warnings
from enum import Enum
from typing import ClassVar

from torch._C._profiler import _ExperimentalConfig
from torch.profiler import ProfilerActivity, profile

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
    """

    _DEFAULT_MODES: ClassVar[list[ProfileMode]] = [ProfileMode.DEVICE, ProfileMode.RUNTIME]

    def __init__(
        self,
        modes: list[ProfileMode] | None = None,
        max_events_per_nc: int | None = None,
        capture_enabled_for_nc: str | None = None,
        profile_output_dir: str | None = None,
    ):
        if modes is None:
            modes = self._DEFAULT_MODES

        if not modes:
            raise ValueError("At least one ProfileMode must be specified")

        for mode in modes:
            if not isinstance(mode, ProfileMode):
                raise TypeError(f"Expected ProfileMode, got {type(mode).__name__}: {mode}")

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
        config_str = ";".join(parts)
        super().__init__(custom_profiler_config=config_str)
        self.custom_profiler_config = config_str


class NRTProfilerError(Exception):
    """Exception raised for NRT profiling errors."""

    pass


class NeuronProfiler:
    """
    Unified profiler wrapper that coordinates both NRT
    runtime profiling (system + device) and PyTorch profiling.
    NRT profiling is always enabled when NeuronProfiler is used
    (subject to runtime binding availability).

    Args:
        pytorch_activities: List of PyTorch profiler activities to enable
        record_shapes: Record tensor shapes in PyTorch profiling (default: False)
        pytorch_trace_file: File to save PyTorch trace to (optional). If not provided and
                           pytorch_activities are enabled, defaults to 'pytorch_trace.json'
                           in the neuron_output_dir. The trace file is always stored in the
                           neuron_output_dir when PyTorch profiling is active.
        neuron_output_dir: Directory for NRT and PyTorch trace files (default: "./output")
        event_filters: Dict with 'nc' (NeuronCore IDs/ranges) and 'types' (event types) filters
        **pytorch_profiler_kwargs: Additional kwargs passed to torch.profiler.profile()
                                   (e.g., with_stack=True, profile_memory=True, with_flops=True).
                                   its will use recognized kwargs and ignore others.

    Example:
        # NRT profiling only
        with NeuronProfiler(
        neuron_output_dir="./my_profiles",
        event_filters={
                'nc': ['0', '4-7'],
                'types': ['cc_exec_barrier', 'nrt_model_switch']
            }
        ) as prof:
            model(inputs)

        # Combined profiling with stack traces
        with NeuronProfiler(
            pytorch_activities=[ProfilerActivity.CPU],
            record_shapes=True,
            neuron_output_dir="./my_profiles",
            event_filters={
                'nc': ['0', '4-7'],
                'types': ['cc_exec_barrier', 'nrt_model_switch']
            },
            with_stack=True,  # Passed to torch.profiler.profile
        ) as prof:
            model(inputs)
    """

    def __init__(
        self,
        pytorch_activities: list[ProfilerActivity] | None = None,
        record_shapes: bool = False,
        pytorch_trace_file: str | None = None,
        neuron_output_dir: str = "./output",
        event_filters: dict[str, list[str] | list[int]] | None = None,
        **pytorch_profiler_kwargs,
    ):
        self.pytorch_activities = pytorch_activities or []
        self.record_shapes = record_shapes
        self.pytorch_profiler_kwargs = pytorch_profiler_kwargs
        self.neuron_output_dir = neuron_output_dir
        if self.pytorch_activities:
            if pytorch_trace_file is None:
                self.pytorch_trace_file = f"{neuron_output_dir}/pytorch_trace.json"
            else:
                self.pytorch_trace_file = f"{neuron_output_dir}/{pytorch_trace_file}"
        else:
            self.pytorch_trace_file = None

        self.event_filters = event_filters
        self._is_active = False
        self._nrt_active = False
        self._pytorch_active = False
        self._pytorch_profiler = None
        self._nrt_config_capsule = None
        self._framework_mapping_active = False

    def _format_nrt_error(self, base_message: str, status: int) -> str:
        """Format NRT error message with status and documentation link."""
        docs_url = (
            "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/"
            "neuron-runtime/nrt-api-guide.html#the-libnrt-api"
        )
        return f"{base_message}. Status: {status}. See {docs_url}"

    def _parse_neuroncore_range(self, range_str: str) -> list[int]:
        """Parse NeuronCore range string like '4-7' to [4, 5, 6, 7] (inclusive)."""
        range_str = range_str.strip()
        try:
            if "-" in range_str:
                start, end = range_str.split("-", 1)
                start_idx = int(start.strip())
                end_idx = int(end.strip())
                if start_idx > end_idx:
                    logger.warning(f"Invalid range '{range_str}': start > end. Skipping.")
                    return []
                return list(range(start_idx, end_idx + 1))  # +1 for inclusive range
            else:
                return [int(range_str)]
        except ValueError as e:
            logger.warning(f"Failed to parse NeuronCore spec '{range_str}': {e}. Skipping.")
            return []

    def _parse_neuroncore_list(self, nc_list: list[str]) -> list[int]:
        """Parse list of NeuronCore IDs/ranges like ['0', '4-7'] to [0, 4, 5, 6, 7]."""
        return sorted({c for nc_spec in nc_list for c in self._parse_neuroncore_range(nc_spec)})

    def _apply_single_filter(
        self,
        config_capsule,
        filter_key: str,
        filter_data: list,
        process_func,
        set_capture_func,
        item_name: str,
        filter_type: str,
    ):
        """Apply a single filter type to NRT config structure.

        Args:
            config_capsule: NRT config capsule
            filter_key: The filter key name
            filter_data: The raw filter data from event_filters dict
            process_func: Function to process filter_data into final items list
            set_capture_func: C function to call for each item
            item_name: Name for individual items (for logging)
            filter_type: Type description for logging
        """
        try:
            if isinstance(filter_data, list) and filter_data:
                processed_items = process_func(filter_data)
                logger.info(f"Applying {filter_type} filters: {processed_items}")

                for item in processed_items:
                    status = set_capture_func(config_capsule, item, True)
                    if status != 0:
                        logger.warning(f"Failed to enable for {item_name} {item}, status: {status}")
        except Exception as e:
            logger.warning(f"Failed to apply {filter_type} filters: {e}")

    def _apply_event_filters(self, config_capsule, event_filters: dict[str, list[str] | list[int]]):
        """Apply event filters to NRT config structure."""
        if "nc" in event_filters:
            self._apply_single_filter(
                config_capsule=config_capsule,
                filter_key="nc",
                filter_data=event_filters["nc"],
                process_func=lambda nc_list: self._parse_neuroncore_list(
                    [str(nc) for nc in nc_list]
                ),
                set_capture_func=_C._nrt_inspect_config_set_capture_enabled_for_nc,
                item_name="NeuronCore",
                filter_type="NeuronCore",
            )

        if "types" in event_filters:
            self._apply_single_filter(
                config_capsule=config_capsule,
                filter_key="types",
                filter_data=event_filters["types"],
                process_func=lambda event_types: event_types,
                set_capture_func=_C._nrt_inspect_config_set_capture_enabled_for_event_type_string,
                item_name="event type",
                filter_type="event type",
            )

    def start(self):
        """Start profiling with both profilers."""
        if self._is_active:
            raise RuntimeError("Profiler is already active")

        try:
            self._start_nrt_profiling()

            if self.pytorch_activities:
                self._start_pytorch_profiling()

            self._is_active = True

        except Exception:
            self._cleanup()
            raise

    def stop(self):
        """Stop profiling with proper cleanup."""
        if not self._is_active:
            raise RuntimeError("Profiler is not active")

        exceptions = []

        if self._pytorch_active:
            try:
                self._stop_pytorch_profiling()
            except Exception as e:
                exceptions.append(e)

        if self._nrt_active:
            try:
                self._stop_nrt_profiling()
            except Exception as e:
                exceptions.append(e)

        self._is_active = False

        if exceptions:
            error_details = [f"{type(e).__name__}: {e}" for e in exceptions]
            combined_message = "Profiling errors:\n" + "\n".join(
                f"  - {detail}" for detail in error_details
            )
            raise RuntimeError(combined_message)

    def _start_nrt_profiling(self):
        """Start NRT profiling with options."""
        try:
            config_capsule = _C._nrt_inspect_config_allocate()
            self._nrt_config_capsule = config_capsule

            status = _C._nrt_inspect_config_set_defaults(config_capsule)
            if status != 0:
                raise NRTProfilerError(
                    self._format_nrt_error("Failed to set NRT config defaults", status)
                )

            status = _C._nrt_inspect_config_set_output_dir(config_capsule, self.neuron_output_dir)
            if status != 0:
                raise NRTProfilerError(
                    self._format_nrt_error("Failed to set NRT output directory", status)
                )

            status = _C._nrt_inspect_config_set_enable_inspect(config_capsule, True)
            if status != 0:
                raise NRTProfilerError(
                    self._format_nrt_error("Failed to enable NRT inspect", status)
                )

            # Set device profile mode to SESSION (2) for comprehensive profiling
            status = _C._nrt_inspect_config_set_device_profile_mode(config_capsule, 2)
            if status != 0:
                raise NRTProfilerError(
                    self._format_nrt_error("Failed to set device profile mode", status)
                )

            if self.event_filters:
                self._apply_event_filters(config_capsule, self.event_filters)

            status = _C._nrt_inspect_begin_with_options(config_capsule)
            if status != 0:
                raise NRTProfilerError(
                    self._format_nrt_error("Failed to start NRT profiling", status)
                )

            self._nrt_active = True
            self._sync_framework_mapping_state()

        except Exception:
            if self._nrt_config_capsule is not None:
                self._nrt_config_capsule = None
            raise

    def _stop_nrt_profiling(self):
        """Stop NRT profiling."""
        if not self._nrt_active:
            return

        try:
            self._nrt_active = False
            self._sync_framework_mapping_state()
            status = _C._nrt_inspect_stop()
            if status != 0:
                raise NRTProfilerError(
                    self._format_nrt_error("Failed to stop NRT profiling", status)
                )
        finally:
            self._nrt_config_capsule = None

    def _sync_framework_mapping_state(self):
        enabled = self._nrt_active and self._pytorch_active
        _C._set_profiler_mapping_enabled(enabled)
        if not enabled and self._framework_mapping_active:
            try:
                self._dump_profiler_mappings()
            except Exception as e:
                logger.warning(f"Failed to dump profiler mappings: {e}")
        self._framework_mapping_active = enabled

    def _dump_profiler_mappings(self):
        import json
        import os
        from datetime import datetime

        import torch_neuronx

        torch_neuronx.synchronize()
        mappings = _C._get_profiler_mappings()
        if mappings.get("data"):
            os.makedirs(self.neuron_output_dir, exist_ok=True)
            # Match NRT timestamp format: YYYYMMDD_HHMMSS_MMM (milliseconds)
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"
            output_path = os.path.join(
                self.neuron_output_dir, f"framework_mapping_{timestamp}.json"
            )
            with open(output_path, "w") as f:
                json.dump(mappings, f, indent=2)

    def _start_pytorch_profiling(self):
        """Start PyTorch profiling."""
        if not self.pytorch_activities:
            return

        self._pytorch_profiler = profile(
            activities=self.pytorch_activities,
            record_shapes=self.record_shapes,
            **self.pytorch_profiler_kwargs,
        )
        self._pytorch_profiler.__enter__()
        self._pytorch_active = True
        self._sync_framework_mapping_state()

    def _stop_pytorch_profiling(self):
        """Stop PyTorch profiling."""
        if not self._pytorch_active or not self._pytorch_profiler:
            return

        try:
            self._pytorch_profiler.__exit__(None, None, None)
            self._pytorch_profiler.export_chrome_trace(self.pytorch_trace_file)
        finally:
            self._pytorch_active = False
            self._sync_framework_mapping_state()

    def _cleanup(self):
        """Best-effort cleanup of profiling resources."""
        if self._pytorch_active:
            try:
                self._stop_pytorch_profiling()
            except Exception as e:
                warnings.warn(f"Failed to cleanup PyTorch trace: {e}", UserWarning, stacklevel=2)

        if self._nrt_active:
            try:
                self._stop_nrt_profiling()
            except Exception as e:
                warnings.warn(f"Failed to cleanup NRT trace: {e}", UserWarning, stacklevel=2)

        self._is_active = False
        self._pytorch_profiler = None

    def __enter__(self):
        """Enter context manager - start profiling."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - stop profiling."""
        self.stop()
        return False  # Don't suppress exceptions

    # Status and result methods
    def is_active(self) -> bool:
        """Check if profiling is currently active."""
        return self._is_active

    def has_pytorch_profiling(self) -> bool:
        """Check if PyTorch profiling is enabled."""
        return bool(self.pytorch_activities)

    def get_pytorch_trace(self) -> str | None:
        """Get PyTorch trace (if available)."""
        if self._pytorch_profiler and not self._pytorch_active:
            return str(self._pytorch_profiler)
        return None

    def get_nrt_output_directory(self) -> str:
        """Get NRT profiling output directory."""
        return self.neuron_output_dir

    def export_pytorch_trace(self, filename: str):
        """Export PyTorch trace to file."""
        if not self._pytorch_profiler:
            raise ValueError("PyTorch profiling not enabled or not started")
        if self._pytorch_active:
            raise ValueError("Cannot export trace while profiling is active")

        self._pytorch_profiler.export_chrome_trace(filename)
