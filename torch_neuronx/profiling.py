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
from dataclasses import dataclass
from enum import Enum, IntFlag
from typing import ClassVar

from torch._C._profiler import _ExperimentalConfig
from torch.autograd.profiler_util import Kernel

import torch_neuronx._C as _C

logger = logging.getLogger(__name__)


class ProfileMode(Enum):
    """Select Neuron-specific profiling granularity.

    Note:
        This enum parallels ``torch.profiler.ProfilerActivity`` and selects
        the NRT profiling modes configured by ``NeuronConfig``.
    """

    DEVICE = "device"
    RUNTIME = "runtime"
    CPU_UTIL = "cpu_util"
    HOST_MEMORY = "host_memory"


class InspectExportMask(IntFlag):
    """Select data for ``nrt_inspect_export`` to write.

    Note:
        ``SYSTEM_PROFILE`` writes per-core part files, ``DEVICE_PROFILE`` stages HBM
        notifications in host memory, and session stop writes the NTFF. ``ALL`` selects
        both. Keep values synchronized with ``at::neuron::InspectExportMask``.
    """

    SYSTEM_PROFILE = 1
    DEVICE_PROFILE = 2
    ALL = SYSTEM_PROFILE | DEVICE_PROFILE


class ExportStatus(Enum):
    """Report frontend gating outcomes for asynchronous export requests.

    Note:
        These values report immediate frontend submission checks rather than later
        NRT drain statuses; their numeric values may overlap NRT status codes.
        ``request_export`` uses them for immediate submission results, while
        ``export_now`` exposes NRT status through ``NRTProfilerError.status``.
        Keep values synchronized with ``at::neuron::ExportStatus`` in
        ``ContinuousExporter.h``.
    """

    # Successfully submitted and will be executed later. Async API.
    SUBMITTED = 1

    # Error suggesting that no profiling session is active.
    NO_ACTIVE_SESSION = 2

    # The intent to export is not specified at profiling session creation time.
    # Therefore the machinery was not initialized and async export cannot be
    # requested.
    EXPORT_NOT_ENABLED = 3

    # The mask input must be a member of `InspectExportMask`.
    UNSUPPORTED_MASK = 4


_MAX_TIME_TRIGGER_INTERVAL_MS = (2**63 - 1) // 1_000_000


@dataclass(frozen=True)
class TimeTrigger:
    """Immutable interval and mask for automatic exports.

    Each active session owns its timer. This value stores configuration only.

    Args:
        interval_ms: Positive integer in milliseconds, at most (2**63 - 1) // 1_000_000.
            This limit keeps the nanosecond interval within a signed 64-bit integer.
        mask: Explicit, nonzero combination of supported InspectExportMask bits.

    Raises:
        TypeError: If ``interval_ms`` is not an integer or is a ``bool``, or ``mask``
            is not an ``InspectExportMask``.
        ValueError: If ``interval_ms`` is outside the supported range or ``mask`` is
            empty or contains unsupported bits.
    """

    interval_ms: int
    mask: InspectExportMask

    def __post_init__(self) -> None:
        """Validate the interval and the supported mask bits."""
        if isinstance(self.interval_ms, bool) or not isinstance(self.interval_ms, int):
            raise TypeError("interval_ms has an invalid type. Pass an int other than bool.")
        if not 0 < self.interval_ms <= _MAX_TIME_TRIGGER_INTERVAL_MS:
            raise ValueError(
                "interval_ms is outside the supported range. "
                f"Pass a value from 1 to {_MAX_TIME_TRIGGER_INTERVAL_MS}."
            )
        mask_bits = _export_mask_to_int(self.mask)
        supported_bits = 0
        for member in InspectExportMask:
            supported_bits |= int(member)
        if mask_bits == 0 or mask_bits & ~supported_bits:
            raise ValueError(
                "mask is empty or has unsupported bits. Select supported InspectExportMask bits."
            )


class ContinuousExportConfig:
    """Configure asynchronous and optional timed profile export.

    Args:
        policy (TimeTrigger | None, optional): Automatic export interval and mask.
            Defaults to ``None`` (explicit requests only).

    Raises:
        TypeError: If ``policy`` is neither a ``TimeTrigger`` nor ``None``.

    Note:
        Passing this configuration to ``NeuronConfig`` enables the worker used by
        ``request_export`` when the profiler session starts. Without this
        configuration, buffered data is exported only at session end or by
        ``export_now``.
    """

    def __init__(self, policy: TimeTrigger | None = None) -> None:
        """Store an optional policy without runtime state."""
        if policy is not None and not isinstance(policy, TimeTrigger):
            raise TypeError("policy has an invalid type. Pass a TimeTrigger or None.")
        self.policy = policy

    def _to_config_parts(self) -> list[str]:
        """Return export settings for a new profiler session."""
        parts = ["continuous_export:true"]
        if self.policy is not None:
            parts.extend(
                [
                    "export_policy:time",
                    f"export_interval_ms:{self.policy.interval_ms}",
                    f"export_mask:{int(self.policy.mask)}",
                ]
            )
        return parts


def _export_mask_to_int(mask: InspectExportMask) -> int:
    if not isinstance(mask, InspectExportMask):
        raise TypeError(f"Expected InspectExportMask, got {type(mask).__name__}: {mask}")
    return int(mask)


_NRT_SUCCESS = 0


def export_now(mask: InspectExportMask) -> None:
    """Export currently buffered profile events synchronously.

    Args:
        mask (InspectExportMask): Profile data categories to export.

    Raises:
        TypeError: If ``mask`` is not an ``InspectExportMask``.
        NRTProfilerError: If NRT returns a non-success status; ``status`` contains
            the original NRT status code.

    Note:
        This blocking call drains reusable buffers. System export writes part files;
        device export stages HBM notifications for the session-end NTFF. Device export
        requires ``NEURON_RT_INSPECT_ENABLE_PERIODIC_HBM_COLLECTOR=1`` before runtime
        initialization and is otherwise skipped successfully. No active capture is also
        a successful no-op. Use ``request_export`` for asynchronous submission.
    """
    status = _C._nrt_inspect_export_now(_export_mask_to_int(mask))
    if status != _NRT_SUCCESS:
        error = NRTProfilerError(f"nrt_inspect_export failed. StatusCode: {status}")
        error.status = status
        raise error


def request_export(mask: InspectExportMask) -> None:
    """Request asynchronous export of currently buffered profile events.

    Args:
        mask (InspectExportMask): Profile data categories to export.

    Raises:
        TypeError: If ``mask`` is not an ``InspectExportMask``.
        NRTProfilerError: If no session is active, continuous export is disabled,
            the mask is unsupported, or the frontend returns an unknown status.

    Note:
        Accepted requests drain reusable buffers on a background thread, and overlapping
        requests can coalesce. Later failures are written to Neuron logs and not returned;
        ``status`` records the immediate ``ExportStatus`` or unknown integer.
    """
    submit_status = _C._nrt_inspect_request_export(_export_mask_to_int(mask))
    try:
        status = ExportStatus(submit_status)
    except ValueError:
        # A status the C++ ExportStatus enum can return but this (possibly older)
        # Python layer does not know. Surface it as NRTProfilerError -- never let a
        # raw ValueError escape the documented contract.
        error = NRTProfilerError(f"request_export returned unknown status: {submit_status}")
        error.status = submit_status
        raise error from None
    if status is not ExportStatus.SUBMITTED:
        error = NRTProfilerError(f"request_export was not accepted: {status.name}")
        error.status = status
        raise error


class NeuronConfig(_ExperimentalConfig):
    """Configure Neuron profiling for ``torch.profiler.profile``.

    Args:
        modes (list[ProfileMode], optional): Modes to enable. Defaults to device
            and runtime profiling.
        max_events_per_nc (int, optional): Maximum events per NeuronCore. Defaults to ``None``.
        capture_enabled_for_nc (str, optional): NeuronCore indices or ranges. Defaults to ``None``.
        profile_output_dir (str, optional): Directory for NRT trace files. Defaults to ``None``.
        precache_node_info (bool, optional): Pre-cache per-NEFF metadata during profiler wait
            iterations. Defaults to ``True``.
        neff_cache_dir (str, optional): NEFF cache copied into profile artifacts at
            session stop. Defaults to ``TORCH_NEURONX_NEFF_CACHE_DIR``, then ``None``.
        correlate_device_time (bool, optional): Correlate device times during trace export.
            Defaults to ``False``.
        continuous_export (ContinuousExportConfig, optional): Enable the background
            export worker and its optional timed policy. Defaults to ``None``.

    Raises:
        ValueError: If ``modes`` is empty.
        TypeError: If a mode or ``continuous_export`` has an invalid type.

    Note:
        This subclasses ``_ExperimentalConfig`` for use as ``experimental_config``;
        ``neuron_profile`` controls the NRT modes.
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
        correlate_device_time: bool = False,
        continuous_export: ContinuousExportConfig | None = None,
    ):
        if modes is None:
            modes = self._DEFAULT_MODES

        if not modes:
            raise ValueError("At least one ProfileMode must be specified")

        for mode in modes:
            if not isinstance(mode, ProfileMode):
                raise TypeError(f"Expected ProfileMode, got {type(mode).__name__}: {mode}")

        if continuous_export is not None and not isinstance(
            continuous_export, ContinuousExportConfig
        ):
            raise TypeError(
                "continuous_export must be a ContinuousExportConfig, "
                f"got {type(continuous_export).__name__}"
            )

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
        if continuous_export is not None:
            parts.extend(continuous_export._to_config_parts())
        config_str = ";".join(parts)
        super().__init__(custom_profiler_config=config_str)
        self.custom_profiler_config = config_str
        self.profile_output_dir = profile_output_dir
        self.neff_cache_dir = neff_cache_dir
        self.correlate_device_time = correlate_device_time
        self.continuous_export = continuous_export

        # Pre-cache profile node info during warmup iterations so that the
        # actively profiled iteration doesn't pay the cost of collecting
        # instructions, DMA descriptors, and CC info from the device.
        if precache_node_info:
            _C._nrt_inspect_precache_enable()

    def get_session_dir(self) -> str:
        """Return the NRT output directory for the current profiling session.

        Returns:
            str: Instance-specific profile output directory.

        Raises:
            ValueError: If ``profile_output_dir`` was not configured.
            RuntimeError: If NRT cannot resolve the instance output directory.

        Note:
            Call this after entering the profiler context because
            ``nrt_inspect_begin`` creates the directory.
        """
        if self.profile_output_dir is None:
            raise ValueError("profile_output_dir was not set on this NeuronConfig")
        return _C._nrt_inspect_get_instance_output_dir(self.profile_output_dir)


class NeuronProfiler:
    """Export PyTorch Chrome traces into an NRT session directory.

    Args:
        config (NeuronConfig): Configuration with ``profile_output_dir`` set.

    Raises:
        ValueError: If ``config.profile_output_dir`` is not set.

    Note:
        Session-directory lookup is deferred until each export, so ``export_trace``
        can be used as an ``on_trace_ready`` callback, called from a custom hook,
        or called directly after profiling.
    """

    def __init__(self, config: NeuronConfig):
        if config.profile_output_dir is None:
            raise ValueError("profile_output_dir must be set on the NeuronConfig")
        self._config = config

    def export_trace(self, prof, filename: str = "trace.json"):
        """Export a PyTorch Chrome trace into the current NRT session directory.

        Args:
            prof: ``torch.profiler.profile`` instance to export.
            filename (str, optional): File name within the session directory. Defaults to
                ``"trace.json"``.

        Raises:
            ValueError: If ``profile_output_dir`` was not configured.
            RuntimeError: If NRT cannot resolve the instance output directory.

        Note:
            When configured, device-time correlation runs before export so
            ``key_averages`` includes Neuron device time columns.
        """
        if self._config.correlate_device_time:
            correlate_device_time(prof)
        session_dir = self._config.get_session_dir()
        prof.export_chrome_trace(os.path.join(session_dir, filename))

    def export_snapshot(self, filename: str = "memory_snapshot.pickle") -> str:
        """Dump a PyTorch memory snapshot into the current NRT session directory.

        Args:
            filename (str, optional): Snapshot file name within the session directory.
                Defaults to ``"memory_snapshot.pickle"``.

        Returns:
            str: Full path written by the memory snapshot exporter.

        Note:
            This co-locates framework memory data with the system profile for tooling.
            Enable ``torch_neuronx.memory._record_memory_history`` first, then call this
            after the profiled region in the same process and rank that recorded it.
        """
        from torch_neuronx import memory as _memory

        session_dir = self._config.get_session_dir()
        path = os.path.join(session_dir, filename)
        _memory._dump_snapshot(path)
        return path


_NON_COMPUTE_PREFIXES = ("neuron::alloc", "neuron::dealloc", "neuron::copy")


def correlate_device_time(prof) -> None:
    """Inject Neuron device execution time into profiler function events.

    Args:
        prof: Completed ``torch.profiler.profile`` instance to update.

    Note:
        Call this after profiling, or from ``on_trace_ready`` with a schedule.
        Repeated calls are no-ops; allocation, deallocation, copy, host,
        asynchronous, duplicate, and nonpositive-duration events are ignored.
    """
    if getattr(prof, "_neuron_correlated", False):
        return
    kineto_results = prof.profiler.kineto_results
    if kineto_results is None:
        return
    fe_list = prof.profiler.function_events
    if fe_list is None:
        return

    host_pid = os.getpid()
    corr_to_device_us: dict[int, float] = {}

    for ev in kineto_results.events():
        lcid = ev.linked_correlation_id()
        if lcid <= 0:
            continue
        if ev.device_index() == host_pid:
            continue
        if ev.name().startswith(_NON_COMPUTE_PREFIXES):
            continue
        dur_ns = ev.duration_ns()
        if dur_ns <= 0:
            continue
        corr_to_device_us[lcid] = corr_to_device_us.get(lcid, 0) + dur_ns / 1000

    if not corr_to_device_us:
        return

    # is_legacy tells PyTorch to compute device_time_total from sum(kernels)
    # rather than recursing into cpu_children.
    seen_ids: set[int] = set()
    for fe in fe_list:
        # Async events are duplicates of their sync counterparts with no own duration.
        if fe.is_async:
            continue
        # No matching device execution for this op (e.g. pure CPU op).
        if fe.id not in corr_to_device_us:
            continue
        # Multiple FunctionEvents share the same correlation id
        # (e.g. aten::mul and aten::mul.Tensor);
        # attribute device time only to the first to avoid double-counting.
        if fe.id in seen_ids:
            continue
        seen_ids.add(fe.id)
        fe.kernels.append(
            Kernel(
                name=fe.name,
                device=fe.device_index,
                duration=corr_to_device_us[fe.id],
            )
        )
        fe.is_legacy = True

    prof._neuron_correlated = True


class NRTProfilerError(Exception):
    """Represent an NRT profiling failure.

    Note:
        Export APIs attach an NRT status integer, a known frontend
        ``ExportStatus``, or an unknown frontend status integer to the ``status``
        attribute.
    """

    pass
