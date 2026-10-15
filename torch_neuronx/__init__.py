try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("torch-neuronx")
except PackageNotFoundError:
    __version__ = "0.0.0"


import importlib
import os
import sys
import warnings

_MODULE_AUTO_COMPILE_INTEGRATION = "torch_neuronx._module_auto_compile_integration"

# Ensure the embedded torch_mlir is used instead of any standalone
# neuron-torch-mlir wheel. For source builds, prepend the embedded path.
# For wheel installs, torch_mlir is already bundled in the wheel.
_embedded_torch_mlir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "third_party",
    "torch-mlir",
    "python",
)
if os.path.isdir(_embedded_torch_mlir):
    sys.path.insert(0, _embedded_torch_mlir)

import threading

from packaging.version import Version

# Global variable to track if module has been loaded
_is_loaded = False

# Thread-local context for transmitting the C++ dispatch cache key to Python
# lowering code without injecting it into kwargs (which breaks composite ops).
_dispatch_context = threading.local()


def _autoload():
    """Entry point for PyTorch autoload mechanism."""
    global _is_loaded
    if _is_loaded:
        return  # Already loaded

    # actual module import
    import torch_neuronx


import torch

# Store reference to real torch.device before any test framework patches it
_real_torch_device = torch.device

# Import build-time configuration (generated during wheel build)
try:
    from ._build_config import (
        DEFAULT_FALLBACK_ONLY_FOR_UNIMPLEMENTED,
        DEFAULT_MLIR_ATEN_OPS,
        DEFAULT_RETAIN_DEVICE_MODE,
        DEFAULT_SYNC_MODE,
    )
except ImportError:
    # Fallback if _build_config.py doesn't exist (e.g., editable install)
    DEFAULT_SYNC_MODE = "0"  # Dev default
    DEFAULT_FALLBACK_ONLY_FOR_UNIMPLEMENTED = "1"
    DEFAULT_MLIR_ATEN_OPS = "1"
    DEFAULT_RETAIN_DEVICE_MODE = "0"

# Set defaults from build-time configuration
# Users can still override these at runtime
os.environ.setdefault("NEURON_LAUNCH_BLOCKING", DEFAULT_SYNC_MODE)
os.environ.setdefault(
    "TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS", DEFAULT_FALLBACK_ONLY_FOR_UNIMPLEMENTED
)

os.environ.setdefault("TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION", "1")
os.environ.setdefault(
    "TORCH_NEURONX_COMPLEX_DECOMPOSE",
    "1" if Version(torch.__version__) >= Version("2.13.0.dev0") else "0",
)

# Set default for aten ops lowering
os.environ.setdefault("TORCH_NEURONX_MLIR_ATEN_OPS", DEFAULT_MLIR_ATEN_OPS)

# Set default for executor retain device
os.environ.setdefault("TORCH_NEURONX_RETAIN_DEVICE_MODE", DEFAULT_RETAIN_DEVICE_MODE)

# Set default for legacy neff
os.environ.setdefault("NEURON_RT_ALLOW_LEGACY_NEFF", "1")

# Set default for async nrt execution
os.environ.setdefault("NEURON_RT_DBG_ZEROCOPY", "1")

# Set default for HBM mapping
os.environ.setdefault("NEURON_RT_MAP_HBM", "1")


def _read_boolean_environment(name, default):
    value = os.environ.get(name) or default
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1; got {value!r}")
    return value == "1"


_streams_enabled = _read_boolean_environment("TORCH_NEURONX_ENABLE_STREAMS", "1")
_host_cc_enabled = _read_boolean_environment("TORCH_NEURONX_ENABLE_HOST_CC", "0")
if _host_cc_enabled and not _streams_enabled:
    raise RuntimeError("TORCH_NEURONX_ENABLE_HOST_CC=1 requires TORCH_NEURONX_ENABLE_STREAMS=1")

os.environ["TORCH_NEURONX_ENABLE_STREAMS"] = "1" if _streams_enabled else "0"
os.environ["TORCH_NEURONX_ENABLE_HOST_CC"] = "1" if _host_cc_enabled else "0"

_dma_transpose_env = "NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE"
if _host_cc_enabled:
    from torch_neuronx.utils import get_platform_target

    platform_target = get_platform_target()
    if platform_target in ("trn2", "trn3"):
        if os.environ.get(_dma_transpose_env, "").lower() == "false":
            raise RuntimeError(
                "NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE=false is incompatible with "
                f"host collectives on {platform_target}. This "
                "combination can lead to a hang. Remove the "
                "NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE=false setting or select a "
                "configuration with TORCH_NEURONX_ENABLE_HOST_CC=0."
            )
        # HostCC hangs on these platforms without this NKI setting. The setting
        # restricts every kernel to psum_banks=7, which costs kernel performance,
        # so only HostCC turns it on.
        os.environ.setdefault(_dma_transpose_env, "true")
    os.environ.setdefault("NEURON_RT_EXEC_TIMEOUT", "300")
else:
    if os.environ.get(_dma_transpose_env, "").lower() == "true":
        raise RuntimeError(
            "TORCH_NEURONX_ENABLE_HOST_CC=0 does not support "
            "NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE=true. Remove "
            "NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE or set TORCH_NEURONX_ENABLE_HOST_CC=1."
        )

# Default to C++ NeuronBackend process group
os.environ.setdefault("TORCH_NEURONX_CPP_PROCESS_GROUP", "1")

# Disable stack trace by default
os.environ.setdefault("TORCH_NEURONX_ENABLE_STACK_TRACE", "0")

# Set default debug directory
os.environ.setdefault("TORCH_NEURONX_DEBUG_DIR", "/tmp/neuron_backend")

# Surface the neuronx-cc compiler version to C++ via env var so the persistent
# NEFF cache key includes it (a compiler upgrade should bust cached NEFFs).
# Read by utils::GetCompilerVersion() in PlatformUtils.cpp.
if "NEURONXCC_VERSION" not in os.environ:
    try:
        import neuronxcc

        os.environ["NEURONXCC_VERSION"] = getattr(neuronxcc, "__version__", "")
    except Exception:
        os.environ["NEURONXCC_VERSION"] = ""

# Disable artifact preservation by default
os.environ.setdefault("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS", "0")

# Enable NKI colz for intermediate NKI artifacts
os.environ.setdefault("NKI_USE_COLZ", "1")

# Enable NKI persistent cache
os.environ.setdefault("NKI_ENABLE_TRACE_CACHE", "1")

# Scalar device memory cache by default.
os.environ.setdefault("TORCH_NEURONX_ENABLE_SCALAR_CACHE", "1")

# Import and register neuron device
try:
    from . import _C

    _C._register_device()
    _C._register_profiler()

    # Initialize per-op int64 policy tables in C++ (lock-free after this).
    from torch_neuronx.int64_policy import init_cpp_policy

    init_cpp_policy()

    # Wrap neuron C extensions with torch._dynamo.disable since they can't be traced.
    import inspect

    for name in dir(_C):
        obj = getattr(_C, name)
        if callable(obj) and not inspect.isclass(obj):
            try:
                wrapped = torch._dynamo.disable(obj)
                setattr(_C, name, wrapped)
            except (TypeError, AttributeError):
                pass
except ImportError as e:
    import warnings

    warnings.warn(f"Failed to import torch_neuronx C++ extensions: {e}", stacklevel=2)
    raise

# Required for NKI LNC degree - NKI gives a kernel default LNC=1 when this
# variable is not set.
if "NEURON_LOGICAL_NC_CONFIG" not in os.environ:
    try:
        from torch_neuronx.utils import get_lnc_setting

        os.environ["NEURON_LOGICAL_NC_CONFIG"] = str(get_lnc_setting())
    except Exception:
        pass

# Import the neuron submodule
# Import and register Python operations
# Import profiling functionality
from . import distributed, neuron, profiling, python_ops

# Import memory apis
from .memory import (
    _dump_snapshot,
    _record_memory_history,
    _snapshot,
    max_memory_allocated,
    max_memory_reserved,
    memory_allocated,
    memory_reserved,
    memory_stats,
    memory_stats_as_nested_dict,
    memory_summary,
    peak_memory_stats,
    reset_accumulated_memory_stats,
    reset_peak_memory_stats,
    segment_pool_analysis,
    segment_pool_analysis_summary,
)

# Import torch.compile metrics APIs
from .neuron_dynamo_backend.fx.passes.complex_decomposition import (
    apply_complex_decomposition_patches,
)
from .neuron_dynamo_backend.metrics import get_dynamo_metrics, reset_dynamo_metrics

# Register the ComplexTensor new_full rule so the complex-decompose retrace
# completes on Neuron. Idempotent and fail soft.
apply_complex_decomposition_patches()

# Import NKI kernel support
from .nki_hop import nki_op, wrap_nki
from .profiling import (
    ContinuousExportConfig,
    ExportStatus,
    InspectExportMask,
    NeuronConfig,
    NeuronProfiler,
    ProfileMode,
    TimeTrigger,
    export_now,
    request_export,
)

# Import stream and event classes
from .streams import Event, Stream

# Register neuron as a custom backend (Python side of the PrivateUse1 surface).
# Skip if another backend already owns PrivateUse1 (imported first) — re-registering
# would raise or shadow its device module/ops. Uses the frozen C++ snapshot shared
# by both guards. See Coexistence.h.
_should_register_device = not _C._privateuse1_owned_by_other()

if _should_register_device:
    torch.utils.rename_privateuse1_backend("neuron")

    torch._register_device_module("neuron", neuron)

    torch.utils.generate_methods_for_privateuse1_backend(
        for_tensor=True, for_module=True, for_storage=True
    )

# Global state for lazy initialization
_initialized = False
_original_pid = 0
_initialization_lock = threading.RLock()
_is_in_bad_fork = False
_parent_runtime_initialized_before_fork = False


def _lazy_init():
    """Initialize Neuron runtime if not already initialized."""
    global _initialized, _original_pid, _is_in_bad_fork

    if _initialized:
        return

    with _initialization_lock:
        # Double-checked locking
        if _initialized:
            return

        # Check for bad fork
        if _is_in_bad_fork:
            raise RuntimeError(
                "Cannot re-initialize Neuron runtime in forked subprocess. "
                "To use Neuron with multiprocessing, use the 'spawn' start method"
            )

        # Initialize the runtime
        _C._neuron_init()

        _original_pid = os.getpid()
        _initialized = True

        # Set any seed set via manual_seed before runtime was ready
        from torch_neuronx import neuron as _neuron_module

        idx = _neuron_module.current_device()
        gen = _C._get_default_generator(idx)
        _neuron_module._set_device_rng_seed(gen.initial_seed())


def _snapshot_neuron_runtime_before_fork():
    """Record whether the parent initialized Neuron before the OS clones it."""
    global _parent_runtime_initialized_before_fork
    # Direct C++ initialization can leave the Python flag false.
    _parent_runtime_initialized_before_fork = _initialized or _C._is_neuron_runtime_initialized()


def _handle_neuron_child_after_fork():
    """Make inherited Neuron and AutoCompile state safe in a fork child."""
    global _initialized, _is_in_bad_fork
    current_pid = os.getpid()
    # A fork child inherits memory, but it cannot use the parent's Neuron runtime.
    if _parent_runtime_initialized_before_fork and _original_pid != current_pid:
        _initialized = False
        _is_in_bad_fork = True

    # AutoCompile locks and module records are also process-local. Let the
    # integration reset them when the feature was loaded in the parent.
    integration = sys.modules.get(_MODULE_AUTO_COMPILE_INTEGRATION)
    if integration is not None:
        integration.handle_child_after_fork(hard_disable=_is_in_bad_fork)


# Register fork handler
os.register_at_fork(
    before=_snapshot_neuron_runtime_before_fork,
    after_in_child=_handle_neuron_child_after_fork,
)

# HLO persistent cache configuration (enabled by default, disable with TORCH_NEURONX_HLO_CACHE=0)
os.environ.setdefault("TORCH_NEURONX_HLO_CACHE", "1")

# Register Python operations (aten PrivateUse1 op impls). Skip if another backend
# owns the device — ours would shadow its kernels and route to our uninitialized
# runtime. Same guard as the device-module registration above.
if _should_register_device:
    python_ops.register_python_operations()

    _C._set_python_ops_registered(True)

# Register neuron::nki_strided_copy_ custom op for packed int64 support
import torch_neuronx.nki_strided_copy_op

# Apply mesh registry patch for SPMD collective legalization
# TODO: Remove this patch once PyTorch's compile_on_one_rank lands upstream
# (see https://github.com/pytorch/pytorch/pull/178116)
from .distributed.mesh_registry import _apply_mesh_registry_patch

_apply_mesh_registry_patch()

# Make torch.random.fork_rng infer device_type from devices (no-op on torch >=2.13)
from .random import _apply_fork_rng_patch

_apply_fork_rng_patch()

from .neuron_dynamo_backend.graph_break_ops import _apply_dynamo_graph_break_patch

_apply_dynamo_graph_break_patch()


def is_neuron_runtime_initialized():
    """Return whether the Neuron runtime is initialized.

    Returns:
        bool: ``True`` if the Neuron runtime is initialized.
    """
    return _C._is_neuron_runtime_initialized()


def device_count() -> int:
    """Return the number of Neuron devices exposed to PyTorch.

    Returns:
        int: Number of Neuron devices configured for PyTorch.

    Note:
        This query may initialize the Neuron runtime.
    """
    return _C._neuron_getDeviceCount()


# Keep the default-off import path free of AutoCompile imports and mutations.
def _module_auto_compile_requested_without_import() -> bool:
    value = (os.environ.get("TORCH_NEURONX_MODULE_AUTO_COMPILE") or "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def _get_module_auto_compile_integration():
    return sys.modules.get(_MODULE_AUTO_COMPILE_INTEGRATION)


def _load_module_auto_compile_integration():
    integration = _get_module_auto_compile_integration()
    if integration is None:
        integration = importlib.import_module(_MODULE_AUTO_COMPILE_INTEGRATION)
    return integration


def _module_auto_compile_patches_are_active() -> bool:
    integration = _get_module_auto_compile_integration()
    return bool(integration is not None and integration._module_auto_compile_patches_active)


def _install_module_auto_compile_patches() -> None:
    """Install AutoCompile only when the user enables it."""
    if not _module_auto_compile_requested_without_import():
        return
    _load_module_auto_compile_integration().install()


def _uninstall_module_auto_compile_patches() -> None:
    """Remove AutoCompile process mutations for tests."""
    integration = _get_module_auto_compile_integration()
    if integration is not None:
        integration.uninstall()


try:
    _install_module_auto_compile_patches()
except Exception as _module_auto_compile_install_error:
    # Private PyTorch APIs can change. Keep package import usable when setup fails.
    integration = _get_module_auto_compile_integration()
    if integration is None:
        warnings.warn(
            "Module AutoCompile setup failed. "
            "AutoCompile is disabled: "
            f"{type(_module_auto_compile_install_error).__name__}: "
            f"{_module_auto_compile_install_error}",
            stacklevel=2,
        )
    else:
        integration.handle_install_failure(_module_auto_compile_install_error)


def current_device() -> int:
    """Return the current Neuron device index.

    Returns:
        int: Index of the current Neuron device.
    """
    return _C._neuron_getCurrentDevice()


def set_device(device: int) -> None:
    """Set the current Neuron device.

    Args:
        device (int): Device index to select.

    Raises:
        ValueError: If ``device`` is outside the available device range.
    """
    # Validate device ID
    if device < 0 or device >= device_count():
        raise ValueError(f"Invalid device id: {device}. Valid range is 0 to {device_count() - 1}")

    _C._neuron_setDevice(device)


def get_device_properties(device):
    """Return the properties of a Neuron device.

    Args:
        device (int or str): Device index or a ``neuron`` device string.

    Returns:
        object: Device properties with ``name`` and ``total_memory`` attributes;
            ``total_memory`` is measured in bytes and is ``0`` if the VNC memory
            query fails.

    Raises:
        ValueError: If the device type or index is invalid.
        RuntimeError: If the runtime cannot initialize in a forked subprocess.

    Note:
        This initializes the Neuron runtime when needed.
    """
    if isinstance(device, str) and "neuron" in device:
        # Parse device string like 'neuron:0'
        import torch

        device_obj = torch.device(device)
        if device_obj.type != "neuron":
            raise ValueError(f"Expected neuron device, got {device_obj.type}")
        device_id = device_obj.index if device_obj.index is not None else current_device()
    else:
        device_id = int(device)

    # Validate device ID
    if device_id < 0 or device_id >= device_count():
        raise ValueError(f"Invalid device id: {device_id}")

    # Initialize runtime if needed (required for HBM memory queries)
    _lazy_init()

    return _C._neuron_getDeviceProperties(device_id)


def empty_cache() -> None:
    """Release all unused cached memory blocks to the Neuron runtime.

    Note:
        With lazy allocation, this synchronizes every visible Neuron device
        before releasing cache segments. Allocated tensor memory is not released.
    """
    _C.NeuronCachingAllocator.emptyCache()


def get_fallback_ops() -> list:
    """Return operations that used CPU fallback.

    Returns:
        list: Names of operations executed on CPU instead of a Neuron device.
    """
    return _C._get_fallback_ops()


def get_executed_ops() -> list:
    """Return operations executed on a Neuron device.

    Returns:
        list: Names of operations executed on a Neuron device.
    """
    return _C._get_executed_ops()


def clear_op_tracking() -> None:
    """Clear tracked operations and concatenation profiles.

    Note:
        This resets process-wide fallback, executed, and concatenated operation
        tracking together with detailed concatenation profiling data.
    """
    _C._clear_op_tracking()


def get_concatenated_ops() -> dict:
    """Return concatenated operations and their constituent operations.

    Returns:
        dict: Mapping from concatenated operation names to lists of original operation
            names, or an empty dictionary if no concatenation occurred.

    Note:
        When ``TORCH_NEURONX_ENABLE_CONCATENATION=1``, operations may be merged
        into concatenated groups.
    """
    return _C._get_concatenated_ops()


def get_concatenated_op_execution_counts() -> dict:
    """Return recorded creation counts for concatenated operation groups.

    Returns:
        dict: Mapping from concatenated operation names to the number of recorded
            group creations for each name, or an empty dictionary if no
            concatenation occurred.

    Note:
        Counts describe concatenated-group creation and recording, not
        compilation-cache outcomes or cache hits.
    """
    return _C._get_concatenated_op_execution_counts()


def get_concatenation_profiles() -> dict:
    """Return profiling data for each concatenation type.

    Returns:
        dict: Mapping from pipe-separated concatenated operation names to profile
            dictionaries. ``merge_times_ns`` contains per-invocation merge times in
            nanoseconds; ``group_sizes`` contains operation counts per group;
            ``invocation_count`` is the number of groups created;
            ``avg_merge_time_ms`` is the mean merge time in milliseconds; and
            ``avg_group_size`` is the mean operation count. Returns an empty
            dictionary when no profiling data exists.
    """
    raw = _C._get_concatenation_profiles()
    result = {}
    for op_name, profile in raw.items():
        merge_times = profile["merge_times_ns"]
        group_sizes = profile["group_sizes"]
        count = profile["invocation_count"]
        avg_merge_ms = (sum(merge_times) / len(merge_times) / 1e6) if merge_times else 0.0
        avg_group = (sum(group_sizes) / len(group_sizes)) if group_sizes else 0.0
        result[op_name] = {
            "merge_times_ns": merge_times,
            "group_sizes": group_sizes,
            "invocation_count": count,
            "avg_merge_time_ms": avg_merge_ms,
            "avg_group_size": avg_group,
        }
    return result


def get_concatenation_status() -> dict:
    """Report whether operation concatenation is active.

    Returns:
        dict: Diagnostic fields: ``enabled`` reports whether
            ``TORCH_NEURONX_ENABLE_CONCATENATION`` equals ``"1"``;
            ``concatenated_ops_in_executed`` and ``individual_ops_in_executed``
            partition executed names by the ``|`` marker;
            ``concatenated_groups_tracked`` is the tracker group count;
            ``is_active`` reports whether either source found a group; and
            ``diagnosis`` provides a human-readable explanation.

    Note:
        Concatenation is active when executed operation names contain ``|`` or the
        concatenation tracker contains at least one group.
    """
    import re

    enabled = os.environ.get("TORCH_NEURONX_ENABLE_CONCATENATION", "0") == "1"

    executed_ops = get_executed_ops()
    concat_pattern = re.compile(r".*\|.*")
    concatenated_ops_in_executed = [op for op in executed_ops if concat_pattern.match(op)]
    individual_ops_in_executed = [op for op in executed_ops if not concat_pattern.match(op)]

    concatenated_groups = get_concatenated_ops()
    num_groups_tracked = len(concatenated_groups)

    is_active = len(concatenated_ops_in_executed) > 0 or num_groups_tracked > 0

    # Build diagnosis
    if not enabled:
        diagnosis = (
            "Concatenation is DISABLED (TORCH_NEURONX_ENABLE_CONCATENATION != 1). "
            "Set TORCH_NEURONX_ENABLE_CONCATENATION=1 to enable."
        )
    elif is_active:
        diagnosis = (
            f"Concatenation is ACTIVE. "
            f"Found {len(concatenated_ops_in_executed)} concatenated op(s) in executed ops "
            f"and {num_groups_tracked} group(s) in ConcatenatedOpTracker."
        )
    else:
        diagnosis = (
            "Concatenation is ENABLED but NOT ACTIVE — no merged ops detected in executed ops "
            "(no '|' separator found) and ConcatenatedOpTracker has 0 groups. "
            "This may indicate dispatch overhead is preventing concatenation from triggering, "
            "or the workload does not have fusible op sequences."
        )

    return {
        "enabled": enabled,
        "concatenated_ops_in_executed": concatenated_ops_in_executed,
        "individual_ops_in_executed": individual_ops_in_executed,
        "concatenated_groups_tracked": num_groups_tracked,
        "is_active": is_active,
        "diagnosis": diagnosis,
    }


def _get_device_index(device, optional=False, allow_cpu=False):
    """Get device index from device specification."""
    if device is None:
        if optional:
            return None
        else:
            return current_device()

    if isinstance(device, str):
        device = torch.device(device)

    if isinstance(device, _real_torch_device):
        if device.type != "neuron":
            raise ValueError(f"Expected neuron device, got {device.type}")
        return device.index if device.index is not None else current_device()

    if isinstance(device, int):
        return device

    raise ValueError(f"Invalid device specification: {device}")


def synchronize(device=None) -> None:
    """Wait for all kernels in all streams on a Neuron device to complete.

    Args:
        device (int, str, or torch.device, optional): Device specification to
            synchronize. Defaults to ``None`` (the current device).

    Raises:
        ValueError: If ``device`` is not a valid Neuron device specification.
        RuntimeError: If the resolved device index is less than ``-1``.
    """
    device_index = _get_device_index(device, optional=True)
    if device_index is None:
        device_index = -1
    _C._neuron_synchronize(device_index)


def current_stream(device=None) -> Stream:
    """Return the currently selected stream for a Neuron device.

    Args:
        device (int, str, or torch.device, optional): Device specification to query.
            Defaults to ``None`` (the current device).

    Returns:
        Stream: Currently selected stream.

    Raises:
        ValueError: If ``device`` is not a valid Neuron device specification.
        RuntimeError: If the resolved device index is less than ``-1``.
    """
    device_index = _get_device_index(device, optional=True)
    if device_index is None:
        device_index = -1
    streamdata = _C._neuron_getCurrentStream(device_index)
    return Stream(stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2])


def default_stream(device=None) -> Stream:
    """Return the default stream for a Neuron device.

    Args:
        device (int, str, or torch.device, optional): Device specification to query.
            Defaults to ``None`` (the current device).

    Returns:
        Stream: Default stream for the device.

    Raises:
        ValueError: If ``device`` is not a valid Neuron device specification.
        RuntimeError: If the resolved device index is less than ``-1``.
    """
    device_index = _get_device_index(device, optional=True)
    if device_index is None:
        device_index = -1
    streamdata = _C._neuron_getDefaultStream(device_index)
    return Stream(stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2])


def set_stream(stream: Stream) -> None:
    """Set the current Neuron stream.

    Args:
        stream (Stream or None): Stream to select. Passing ``None`` leaves the current
            stream unchanged.

    Note:
        Selecting a stream on another device also makes that device current.
    """
    if stream is None:
        return
    _C._neuron_setStream(
        stream_id=stream.stream_id,
        device_index=stream.device_index,
        device_type=stream.device_type,
    )


class device:  # noqa: N801
    """Temporarily select a Neuron device.

    Args:
        device (int, str, torch.device, or None): Device to select, or ``None`` to use
            the current device.

    Raises:
        ValueError: If ``device`` is invalid, is not a Neuron device, or has an unavailable index.

    Note:
        Exiting the context restores the previously selected device.
    """

    def __init__(self, device):
        self.idx = _get_device_index(device)
        self.prev_idx = -1

    def __enter__(self):
        if self.idx == -1:
            return
        self.prev_idx = current_device()
        if self.prev_idx != self.idx:
            set_device(self.idx)

    def __exit__(self, type, value, traceback):
        if self.prev_idx != self.idx and self.prev_idx != -1:
            set_device(self.prev_idx)


class StreamContext:
    """Temporarily select a Neuron stream.

    Args:
        stream (Stream or None): Stream to select, or ``None`` to leave it unchanged.

    Note:
        Exiting restores the previous source stream and any previous
        destination-device stream used for a cross-device context.
    """

    def __init__(self, stream):
        self.stream = stream
        self.idx = _get_device_index(None, True)
        if not torch.jit.is_scripting() and self.idx is None:
            # If no device index, use the stream's device index
            if stream is not None:
                self.idx = stream.device_index
            else:
                self.idx = -1

        self.src_prev_stream = None
        self.dst_prev_stream = None

    def __enter__(self):
        if self.stream is None:
            return

        # Always save the previous stream and set the new one
        self.src_prev_stream = current_stream(None)

        if self.src_prev_stream.device != self.stream.device:
            with device(self.stream.device):
                self.dst_prev_stream = current_stream(self.stream.device)
        set_stream(self.stream)

    def __exit__(self, type, value, traceback):
        if self.stream is None:
            return

        if self.src_prev_stream is not None:
            if (
                self.dst_prev_stream is not None
                and self.src_prev_stream.device != self.stream.device
            ):
                set_stream(self.dst_prev_stream)
            set_stream(self.src_prev_stream)


def stream(stream) -> StreamContext:
    """Return a context manager that temporarily selects a Neuron stream.

    Args:
        stream (Stream or None): Stream to select, or ``None`` to leave it unchanged.

    Returns:
        StreamContext: Context manager that restores the previous stream on exit.
    """
    return StreamContext(stream)
