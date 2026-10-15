"""Utility functions for torch-neuronx."""

import functools
import hashlib
import os
import typing as _t
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import numpy as np
import torch
from neuronxcc.starfish.support import dtype as _ndt
from packaging.version import Version

from torch_neuronx.packed_storage import (
    packed_int64_execution_view,
    packed_int64_high_word_alias,
    packed_logical_view,
    packed_physical_view,
)


def get_rank() -> int:
    """Get the current process rank.

    Returns the distributed rank if torch.distributed is initialized,
    otherwise returns the current Neuron device ID.

    Returns:
        int: Process rank or device ID.
    """
    return (
        torch.distributed.get_rank()
        if torch.distributed.is_initialized()
        else torch.neuron.current_device()
    )


def deterministic_hash(obj) -> str:
    """Generate a deterministic hash string for an object.

    Args:
        obj (object): Object whose representation is hashed.

    Returns:
        str: First 16 hexadecimal characters of the SHA-256 digest.

    Note:
        The hash is stable across processes only when ``repr(obj)`` is stable
        across processes.
    """
    return hashlib.sha256(repr(obj).encode()).hexdigest()[:16]


_LOWERING_SOURCE_SUFFIXES = frozenset({".py", ".cpp", ".h", ".hpp", ".cc", ".td", ".mlir"})


def _hash_source_tree(hasher, directory: Path, base: Path) -> None:
    """Fold every lowering source file under directory into hasher.

    A missing directory contributes nothing. An installed wheel ships no csrc/ and no
    vendored torch-mlir sources, so those trees are absent there and cost no time.
    """
    if not directory.exists():
        return
    for path in sorted(directory.rglob("*")):
        if path.suffix not in _LOWERING_SOURCE_SUFFIXES or not path.is_file():
            continue
        hasher.update(str(path.relative_to(base)).encode())
        hasher.update(path.read_bytes())


@functools.lru_cache(maxsize=1)
def global_version_hash() -> str:
    """Compute a process-cached version hash for persistent caches.

    Returns:
        str: SHA-256 digest covering lowering sources and package versions.

    Note:
        The digest covers Python files under ``python_ops`` and
        ``neuron_dynamo_backend``, package versions, and
        ``TORCH_NEURONX_INT64_MODE`` so persistent caches are invalidated when
        these inputs change. Missing optional package versions are ignored.
    """
    import torch_neuronx

    hasher = hashlib.sha256()

    root = Path(torch_neuronx.__file__).parent
    for path in sorted(root.glob("*.py")):
        hasher.update(path.name.encode())
        hasher.update(path.read_bytes())
    for d in [
        root / "python_ops",
        root / "neuron_dynamo_backend",
        root / "distributed",
        root / "kernels",
        root / "csrc",
    ]:
        _hash_source_tree(hasher, d, root)

    # torch-mlir is vendored, so its version string never moves. Hash the sources instead.
    _hash_source_tree(hasher, root.parent / "third_party" / "torch-mlir", root.parent)

    # Hash package versions
    hasher.update(torch.__version__.encode())
    hasher.update(getattr(torch_neuronx, "__version__", "unknown").encode())
    try:
        import neuronxcc

        hasher.update(neuronxcc.__version__.encode())
    except Exception:
        pass
    try:
        import nki

        hasher.update(getattr(nki, "__version__", "unknown").encode())
    except Exception:
        pass

    hasher.update(os.environ.get("TORCH_NEURONX_INT64_MODE", "native").encode())

    return hasher.hexdigest()


# ==================== Tensor Utilities ====================


def flatten_tensors(obj) -> list[torch.Tensor]:
    """Flatten tensors from nested lists and tuples into a list.

    Args:
        obj (object): Tensor or nested list or tuple to traverse.

    Returns:
        list[torch.Tensor]: Tensors encountered in traversal order.
    """
    result = []
    if isinstance(obj, torch.Tensor):
        result.append(obj)
    elif isinstance(obj, list | tuple):
        for item in obj:
            result.extend(flatten_tensors(item))
    return result


def _get_first_tensor_device(obj) -> torch.device | None:
    """Get device from first tensor encountered (early return, no full flatten)."""
    if isinstance(obj, torch.Tensor):
        return obj.device
    elif isinstance(obj, list | tuple):
        for item in obj:
            dev = _get_first_tensor_device(item)
            if dev is not None:
                return dev
    return None


def get_device_from_tensors(tensors) -> torch.device:
    """Extract the device of the first tensor in a nested structure.

    Args:
        tensors (object): Tensor or nested list or tuple to inspect.

    Returns:
        torch.device: First tensor device, or the current Neuron device when
        no tensor is found.
    """
    import torch_neuronx

    dev = _get_first_tensor_device(tensors)
    return dev if dev is not None else torch.device("neuron", torch_neuronx.current_device())


@contextmanager
def suppress_specific_warnings(patterns):
    """Temporarily suppress warnings whose messages contain given patterns.

    Args:
        patterns (Iterable[str]): Message substrings to suppress.

    Yields:
        None: Control while the warning filter is installed.

    Note:
        This context manager temporarily replaces the process-wide
        ``warnings.showwarning`` function and restores it on exit. ``patterns``
        is scanned for every warning, so a one-shot iterator is consumed.
    """
    original_showwarning = warnings.showwarning

    def _filtered_showwarning(message, category, filename, lineno, file=None, line=None):
        msg_str = str(message)
        if any(pattern in msg_str for pattern in patterns):
            return
        original_showwarning(message, category, filename, lineno, file, line)

    try:
        warnings.showwarning = _filtered_showwarning
        yield
    finally:
        warnings.showwarning = original_showwarning


def use_mlir_aten_ops():
    """Check whether torch-mlir aten op lowering is enabled.

    Returns:
        bool: ``True`` when ``TORCH_NEURONX_MLIR_ATEN_OPS`` is ``"1"``.
    """
    return os.environ.get("TORCH_NEURONX_MLIR_ATEN_OPS", "0") == "1"


def use_dynamic_nki_kernels():
    """Check whether dynamic-shape NKI kernels are preferred.

    Returns:
        bool: ``True`` when ``TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS`` is
        ``"1"``.

    Note:
        Enabled ops use registered ``dynamic_kernels`` implementations instead
        of default MLIR or PyTorch lowering when compile-time constraints hold.
    """
    return os.environ.get("TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS", "0") == "1"


def is_sync_mode_enabled():
    """Check whether legacy launch blocking is enabled.

    Returns:
        bool: ``True`` when ``NEURON_LAUNCH_BLOCKING`` is ``"1"``.
    """
    return os.environ.get("NEURON_LAUNCH_BLOCKING", "0") == "1"


@functools.lru_cache(maxsize=1)
def is_host_cc_enabled():
    """Check whether host-side collective communication is enabled.

    Returns:
        bool: ``True`` when ``TORCH_NEURONX_ENABLE_HOST_CC`` is ``"1"``.

    Note:
        Host-side communication enables NRT host collectives on non-default
        streams. The result is cached after the first call because the setting
        is process-wide.
    """
    return os.environ["TORCH_NEURONX_ENABLE_HOST_CC"] == "1"


@functools.lru_cache(maxsize=1)
def is_streams_enabled():
    """Check whether distinct Neuron streams are enabled.

    Returns:
        bool: ``True`` when ``TORCH_NEURONX_ENABLE_STREAMS`` is ``"1"``.
    """
    return os.environ["TORCH_NEURONX_ENABLE_STREAMS"] == "1"


def get_gmm_align() -> int:
    """Get the grouped-matmul offset alignment constraint.

    Returns:
        int: Value of ``TORCH_NEURONX_ATEN_GMM_ALIGN``, defaulting to ``128``.

    Raises:
        ValueError: If the environment value is not a valid integer.
    """
    return int(os.environ.get("TORCH_NEURONX_ATEN_GMM_ALIGN", "128"))


def skip_op_preconditions() -> bool:
    """Check whether op precondition validation should be skipped.

    Returns:
        bool: ``True`` when ``TORCH_NEURONX_SKIP_OP_PRECONDITIONS`` is ``"1"``.
    """
    return os.environ.get("TORCH_NEURONX_SKIP_OP_PRECONDITIONS", "0") == "1"


def is_native_int64_enabled() -> bool:
    """Check whether the active int64 mode permits S64 compiler passthrough.

    Returns:
        bool: ``True`` for ``native`` and ``native_strict`` modes and ``False`` for
        ``packed``, ``cpu_bounce``, and ``cpu_fallback`` modes.

    Raises:
        ValueError: If ``TORCH_NEURONX_INT64_MODE`` is not a recognized mode.

    Note:
        This backward-compatibility shim delegates to
        ``torch_neuronx.int64_policy.is_s64_passthrough``.
    """
    from torch_neuronx.int64_policy import is_s64_passthrough

    return is_s64_passthrough()


def get_logical_neuron_cores():
    """Get the logical Neuron core configuration.

    Returns:
        str: ``NEURON_LOGICAL_NC_CONFIG`` when set, otherwise the default
        mapped from the runtime instance family.
    """
    return os.environ.get("NEURON_LOGICAL_NC_CONFIG", get_lnc_setting())


def get_worker_multiplier():
    """Get the worker multiplier for the current platform and LNC setting.

    Returns:
        int: ``2`` for LNC 1 on ``trn2`` and ``1`` otherwise.
    """
    lnc = get_logical_neuron_cores()
    target = get_platform_target()
    if target == "trn2" and lnc == "1":
        # LNC=1 with TRN2 means we can use 2x logical cores
        return 2
    # Otherwise, use the physical cores as is
    return 1


# ==================== Op Logging ====================
# Uses the C++ logger for unified logging between Python and C++


def log_executed_op(op_name: str):
    """Record an operation as executed on a Neuron device.

    Args:
        op_name (str): Name of the executed operation.

    Note:
        The C++ binding adds each name once to the process-wide executed-op
        tracker queried by ``torch_neuronx.get_executed_ops``. Operator-category
        text output is suppressed, and ``clear_op_tracking`` resets the tracker.
    """
    import torch_neuronx._C as _C

    _C._log_executed_op(op_name)


def log_offloaded_op(op_name: str):
    """Record an operation as using CPU fallback.

    Args:
        op_name (str): Name of the offloaded operation.

    Note:
        The C++ binding adds each name once to the process-wide fallback-op
        tracker queried by ``torch_neuronx.get_fallback_ops``. Operator-category
        text output is suppressed, and ``clear_op_tracking`` resets the tracker.
    """
    import torch_neuronx._C as _C

    _C._log_offloaded_op(op_name)


# ==================== NEFF Cache Management ====================


def is_neff_cache_disabled():
    """Check whether NEFF caching is disabled.

    Returns:
        bool: ``True`` when ``TORCH_NEURONX_DISABLE_NEFF_CACHE`` is ``"1"``,
        ``"true"``, or ``"yes"``, ignoring case.
    """
    disable_cache = os.environ.get("TORCH_NEURONX_DISABLE_NEFF_CACHE", "").lower()
    return disable_cache in ["1", "true", "yes"]


# ==================== NEFF Cache Logging ====================


def log_neff_cache_hit(cache_key: str):
    """Log a NEFF cache hit.

    Args:
        cache_key (str): Cache key that was hit.
    """
    import torch_neuronx._C as _C

    _C._log_neff_cache_hit(cache_key)


def log_neff_cache_miss(cache_key: str):
    """Log a NEFF cache miss.

    Args:
        cache_key (str): Cache key that was missed.
    """
    import torch_neuronx._C as _C

    _C._log_neff_cache_miss(cache_key)


def log_neff_cache_store(cache_key: str):
    """Log a NEFF cache store.

    Args:
        cache_key (str): Cache key being stored.
    """
    import torch_neuronx._C as _C

    _C._log_neff_cache_store(cache_key)


# ==================== Hardware Instance Info ====================

_g_instance_type = None


def _get_instance_type():
    """Get the instance type from runtime."""
    global _g_instance_type

    if _g_instance_type is not None:
        return _g_instance_type

    import torch_neuronx._C as _C

    return _C._get_instance_type()


def get_platform_target():
    """Get the platform target for NKI kernels.

    Returns:
        str: ``NEURON_PLATFORM_TARGET_OVERRIDE`` when nonempty, otherwise the
        target mapped from the runtime instance family.

    Raises:
        RuntimeError: If runtime instance information cannot be queried or the
        instance family is unknown or unsupported.
    """
    import torch_neuronx._C as _C

    return _C._get_platform_target()


def get_lnc_setting():
    """Get the logical Neuron core setting.

    Returns:
        str: ``NEURON_LOGICAL_NC_CONFIG`` when set, otherwise the default for
        the runtime instance family.

    Raises:
        RuntimeError: If runtime instance information cannot be queried or the
        instance family is unknown or unsupported.
    """
    import torch_neuronx._C as _C

    return _C._get_logical_neuron_cores()


def move_pytree_to_cpu(obj):
    """Move Neuron tensors in a nested structure to CPU.

    Args:
        obj (object): Tensor or nested list, tuple, or dictionary to transform.

    Returns:
        object: Structure with Neuron tensors moved to CPU and other values
        unchanged.
    """
    if isinstance(obj, torch.Tensor) and obj.device.type == "neuron":
        return obj.to("cpu")
    elif isinstance(obj, list | tuple):
        return type(obj)(move_pytree_to_cpu(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: move_pytree_to_cpu(v) for k, v in obj.items()}
    else:
        return obj


def move_pytree_to_neuron(obj):
    """Move tensors in a nested structure to a Neuron device.

    Args:
        obj (object): Tensor or nested list, tuple, or dictionary to transform.

    Returns:
        object: Structure with every tensor moved to a Neuron device and other
        values unchanged.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to("neuron")
    elif isinstance(obj, list | tuple):
        return type(obj)(move_pytree_to_neuron(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: move_pytree_to_neuron(v) for k, v in obj.items()}
    else:
        return obj


# ==================== Dtype Utilities ====================


def cast_dtype_if_needed(tensor: torch.Tensor, target_dtype: torch.dtype | None) -> torch.Tensor:
    """Cast a tensor when a different target dtype is requested.

    Args:
        tensor (torch.Tensor): Tensor to cast.
        target_dtype (torch.dtype or None): Desired dtype, or ``None`` to
        preserve the input dtype.

    Returns:
        torch.Tensor: Cast tensor, or the original tensor when no cast is needed.

    Note:
        The registered ``aten::_to_copy`` implementation performs the cast.
        On a Neuron device this currently uses CPU fallback when necessary and
        can automatically benefit from future in-device casts.
    """
    if target_dtype is None or tensor.dtype == target_dtype:
        return tensor
    return tensor.to(dtype=target_dtype)


def is_float8_e4m3_supported():
    # NKI casts torch.float8_e4m3fn to nl.float8_e4m3 at the input boundary when
    # UNSAFE_FP8FNCAST is set; allow the same cast on the output boundary so a
    # kernel returning nl.float8_e4m3 maps back to torch.float8_e4m3fn.
    """Check whether unsafe E4M3 boundary casts are enabled.

    Returns:
        bool: ``True`` when ``UNSAFE_FP8FNCAST`` has an enabled value.
    """
    return os.environ.get("UNSAFE_FP8FNCAST", "").lower() in ("1", "true", "yes", "on")


def is_float8_e4m3fn_supported():
    """Check whether the platform supports the E4M3FN dtype.

    Returns:
        bool: ``True`` on platforms other than ``trn2``.

    Raises:
        RuntimeError: If the runtime instance family is unknown or unsupported.
    """
    return get_platform_target() != "trn2"


def is_float8_e8m0fnu_supported():
    """Check whether the platform supports the E8M0FNU dtype.

    Returns:
        bool: ``True`` on platforms other than ``trn2``.
    """
    return get_platform_target() != "trn2"


def is_mxocp_supported():
    """Check whether the platform supports MX OCP tensors.

    Returns:
        bool: ``True`` on platforms other than ``trn2``.

    Raises:
        RuntimeError: If runtime instance information cannot be queried or the
        instance family is unknown or unsupported.
    """
    return get_platform_target() != "trn2"


_MXFP8_value_tensor_dtypes = Literal["float8_e4m3fn", "float8_e5m2"]


# Framework container for MXFP8 data and scale tensors
@dataclass
class MXFP8:
    """Store MXFP8 value data, scales, and logical dtype.

    Args:
        data (torch.Tensor): MXFP8 value tensor.
        scale (torch.Tensor): Scale tensor associated with ``data``.
        dtype (_MXFP8_value_tensor_dtypes): Logical value-tensor dtype.

    Raises:
        RuntimeError: If runtime instance information cannot be queried or mapped.
        Exception: If MX OCP tensors are unsupported on the current platform.
        ValueError: If ``dtype`` is not a supported MXFP8 value dtype.
    """

    data: torch.Tensor
    scale: torch.Tensor
    dtype: _MXFP8_value_tensor_dtypes

    def __post_init__(self):
        if not is_mxocp_supported():
            raise Exception(
                f"MXOCP is not supported on this platform target: {get_platform_target()}"
            )
        if self.dtype not in get_args(_MXFP8_value_tensor_dtypes):
            supported_dtypes = get_args(_MXFP8_value_tensor_dtypes)
            raise ValueError(
                f"Unsupported MXFP8 Value Tensor dtype: {self.dtype}. "
                f"Supported MXFP8 Value Tensor dtypes: {supported_dtypes}"
            )


# ==================== Dtype Conversion Helpers ====================


_NUMPY_TO_TORCH_MAP: dict[object, torch.dtype] = {
    # floats
    getattr(np, "float32", object()): torch.float32,
    getattr(np, "float16", object()): torch.float16,
    getattr(np, "float64", object()): torch.float64,
    # ints
    getattr(np, "int32", object()): torch.int32,
    getattr(np, "int64", object()): torch.int64,
    getattr(np, "int16", object()): torch.int16,
    getattr(np, "int8", object()): torch.int8,
    getattr(np, "uint8", object()): torch.uint8,
    # bool
    getattr(np, "bool_", object()): torch.bool,
}

_STR_TO_TORCH_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "uint16": torch.uint16,
    "uint32": torch.uint32,
    "bool": torch.bool,
    "bool_": torch.bool,
}

# Conditionally add float8 mappings when the torch build supports them.
if hasattr(torch, "float8_e5m2"):
    _STR_TO_TORCH_MAP["float8_e5m2"] = torch.float8_e5m2
if hasattr(torch, "float8_e4m3fn"):
    _STR_TO_TORCH_MAP["float8_e4m3fn"] = torch.float8_e4m3fn
if hasattr(torch, "float8_e4m3fnuz"):
    _STR_TO_TORCH_MAP["float8_e4m3fnuz"] = torch.float8_e4m3fnuz
if hasattr(torch, "float8_e5m2fnuz"):
    _STR_TO_TORCH_MAP["float8_e5m2fnuz"] = torch.float8_e5m2fnuz


def map_external_dtype_to_torch(nki_or_numpy_dtype: _t.Any) -> torch.dtype:
    """Map an external dtype representation to a PyTorch dtype.

    Args:
        nki_or_numpy_dtype (Any): PyTorch, NeuronXCC starfish, NumPy, or string
        dtype representation.

    Returns:
        torch.dtype: Corresponding PyTorch dtype.

    Raises:
        RuntimeError: If runtime platform information cannot be queried or mapped,
        the dtype is unsupported on the current platform, or the dtype is unrecognized.
    """
    # Already a torch dtype
    if isinstance(nki_or_numpy_dtype, torch.dtype):
        return nki_or_numpy_dtype

    # NeuronXCC starfish dtype: explicitly handle bf16 and float8 which lack a NumPy dtype
    if _ndt is not None:
        target = get_platform_target()
        if nki_or_numpy_dtype == getattr(_ndt, "bfloat16", object()):
            return torch.bfloat16
        if nki_or_numpy_dtype == getattr(_ndt, "float8_e5m2", object()):
            return torch.float8_e5m2
        if nki_or_numpy_dtype == getattr(_ndt, "float8_e4m3", object()):
            if is_float8_e4m3_supported():
                return torch.float8_e4m3fn
            raise RuntimeError(f"Unexpected dtype {nki_or_numpy_dtype} for platform {target}")
        if nki_or_numpy_dtype == getattr(_ndt, "float8_e4m3fn", object()):
            if not is_float8_e4m3fn_supported():
                raise RuntimeError(f"Unsupported dtype {nki_or_numpy_dtype} for platform {target}")
            return torch.float8_e4m3fn
        if nki_or_numpy_dtype == getattr(_ndt, "float8_e8m0fnu", object()):
            if not is_float8_e8m0fnu_supported():
                raise RuntimeError(f"Unsupported dtype {nki_or_numpy_dtype} for platform {target}")
            return torch.float8_e8m0fnu

    # NumPy direct matches
    if np is not None:
        # Handle np.dtype instances
        if isinstance(nki_or_numpy_dtype, getattr(np, "dtype", ())):  # type: ignore[arg-type]
            # Normalize to the canonical scalar type (e.g., np.float32)
            kind = getattr(nki_or_numpy_dtype, "type", None)
            if kind in _NUMPY_TO_TORCH_MAP:
                return _NUMPY_TO_TORCH_MAP[kind]
            # Try by name string as a fallback
            name = getattr(nki_or_numpy_dtype, "name", "")
            if name:
                try:
                    return {
                        "float32": torch.float32,
                        "float16": torch.float16,
                        "float64": torch.float64,
                        "int32": torch.int32,
                        "int64": torch.int64,
                        "int16": torch.int16,
                        "int8": torch.int8,
                        "uint8": torch.uint8,
                        "bool": torch.bool,
                        "bool_": torch.bool,
                    }[name]
                except KeyError:
                    pass

        # Handle canonical NumPy scalar types directly (e.g., np.float32)
        if nki_or_numpy_dtype in _NUMPY_TO_TORCH_MAP:
            return _NUMPY_TO_TORCH_MAP[nki_or_numpy_dtype]

    if nki_or_numpy_dtype in _STR_TO_TORCH_MAP:
        return _STR_TO_TORCH_MAP[nki_or_numpy_dtype]

    raise RuntimeError(f"Unrecognized dtype {nki_or_numpy_dtype}")


def map_torch_dtype_to_external(_dtype):
    """Map a PyTorch dtype to a Neuron compiler dtype representation.

    Args:
        _dtype (torch.dtype): PyTorch dtype to map.

    Returns:
        object: NeuronXCC or NumPy dtype representation.

    Raises:
        RuntimeError: If ``_dtype`` is ``torch.float8_e4m3fn``.
    """
    from neuronxcc.starfish.support import dtype as neuron_dtype

    torch_to_neuron_dtype = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.float16: np.float16,
        torch.int64: np.int64,
        torch.int32: np.int32,
        torch.int16: np.int16,
        torch.int8: np.int8,
        torch.uint8: np.uint8,
        torch.uint16: np.uint16,
        torch.uint32: np.uint32,
        torch.uint64: np.uint64,
        torch.bool: np.uint8,
        torch.complex64: np.complex64,
        torch.complex128: np.complex128,
        torch.bfloat16: neuron_dtype.bfloat16,
    }

    if _dtype == getattr(torch, "float8_e4m3fn", None):
        raise RuntimeError("float8_e4m3fn is not supported in neuronxcc. ")

    if hasattr(torch, "float8_e5m2"):
        torch_to_neuron_dtype[torch.float8_e5m2] = neuron_dtype.float8_e5m2

    if hasattr(torch, "float8_e4m3fn"):
        torch_to_neuron_dtype[torch.float8_e4m3fn] = neuron_dtype.float8_e4m3fn

    if hasattr(torch, "float8_e4m3fnuz"):
        torch_to_neuron_dtype[torch.float8_e4m3fnuz] = neuron_dtype.float8_e4m3fn

    if hasattr(torch, "float8_e5m2fnuz"):
        torch_to_neuron_dtype[torch.float8_e5m2fnuz] = neuron_dtype.float8_e5m2

    if _dtype in torch_to_neuron_dtype:
        return torch_to_neuron_dtype[_dtype]

    # For other dtype that is common with numpy, use builtin pytorch to do the translation
    return torch.empty(1, dtype=_dtype, device="cpu").numpy().dtype


def convert_for_neuron(
    tensor: torch.Tensor, non_blocking: bool = False, convert_int64: bool = True
) -> torch.Tensor:
    """Convert a tensor to a Neuron-compatible dtype when needed.

    Args:
        tensor (torch.Tensor): Tensor to convert.
        non_blocking (bool, optional): Allow non-blocking CPU fallback transfers.
            Defaults to ``False``.
        convert_int64 (bool, optional): Convert int64 when required by lowering.
            Defaults to ``True``; when ``False``, MLIR handles S64 HLO natively.

    Returns:
        torch.Tensor: Neuron-compatible tensor: float64 becomes float32; enabled int64
        becomes int32 unless in packed mode on a Neuron device, retaining logical int64.

    Raises:
        OverflowError: If packed int64 overflow checking finds a value outside
        the int32 range.

    Note:
        Packed int64 tensors on a Neuron device retain their logical dtype.
        Other Neuron-device 64-bit conversions use explicit CPU fallback.
    """
    # Determine if conversion is needed
    if tensor.dtype == torch.int64 and convert_int64:
        # Packed mode: extract low words on-device (no CPU bounce)
        from torch_neuronx.int64_policy import is_packed_int64_mode

        if is_packed_int64_mode() and tensor.device.type == "neuron":
            # Leave it *logical int64*. kernel.py reinterprets to the int32[..., 2]
            # physical view at the graph boundary, gathers the low words in-graph,
            # and binds the physical view at execution — so packed int64 is handled
            # in one place and dispatch args keep their logical shape (avoids the
            # shape-leak that broke shape-reading ops like nonzero / cross_entropy).
            if os.environ.get("TORCH_NEURONX_PACKED_INT64_CHECK_OVERFLOW", "0") == "1":
                low = packed_int64_execution_view(tensor, contiguous=True)
                high_alias = packed_int64_high_word_alias(tensor)
                high = torch.empty_like(low)
                if tensor.is_contiguous():
                    torch.ops.neuron.nki_strided_copy_(high_alias, high)
                else:
                    high.copy_(high_alias)
                expected_high = low >> 31
                if not torch.eq(high, expected_high).all().item():
                    raise OverflowError(
                        "int64 tensor contains values outside int32 range "
                        "[-2147483648, 2147483647]. Packed int64 mode only "
                        "supports values representable in 32 bits. Set "
                        "TORCH_NEURONX_INT64_MODE=cpu_bounce for full range."
                    )

            if not tensor.is_contiguous():
                # Mirror the C++ NeuronSafeCast fix for non-contiguous packed
                # int64. The graph boundary binds the int32[..., 2] physical
                # view; a non-contiguous int64 otherwise binds only the low-word
                # half (e.g. NRT "input0 too small: 100 vs 200"). Contiguify the
                # int32 *physical view* (not the int64 tensor — that redispatches
                # through convert_for_neuron and recurses; on int32 it doesn't).
                # The [low, high] pairs are value-preserving, so logical_view of
                # the contiguous result equals tensor.contiguous().
                phys_contig = packed_physical_view(tensor).contiguous()
                return packed_logical_view(
                    phys_contig, tensor.dtype, tuple(int(d) for d in tensor.shape)
                )

            return tensor
        target_dtype = torch.int32
    elif tensor.dtype == torch.float64:
        target_dtype = torch.float32
    else:
        # <=32-bit or convert_int64=False (MLIR S64 HLO): already compatible
        return tensor

    # CPU tensors: safe to use .to(...) without involving Neuron redispatch
    if tensor.device.type == "cpu":
        return tensor.to(target_dtype)

    # Neuron tensors: avoid Tensor.to(...) to prevent redispatch into aten::_to_copy.
    # 64-bit conversions may bounce through CPU explicitly via cast_policy helpers.
    if tensor.device.type == "neuron":
        from torch_neuronx.python_ops import cast_policy  # local import to avoid cycles

        cpu_tmp = cast_policy.copy_neuron_to_cpu(
            tensor, target_dtype=target_dtype, non_blocking=non_blocking
        )
        return cast_policy.copy_cpu_to_neuron(
            cpu_tmp, tensor.device, target_dtype, non_blocking=non_blocking
        )

    # Other devices: fallback to .to(...)
    return tensor.to(target_dtype)


# ==================== Version Info ====================

TORCH_VERSION = Version(torch.__version__)


# ==================== Op Registration Discovery ====================


def get_neuron_registered_ops(
    as_sorted: bool = False, keep_variant: bool = False
) -> set[str] | list[str]:
    """Get op names registered for the ``PrivateUse1`` dispatch key.

    Args:
        as_sorted (bool, optional): Return a sorted list instead of a set.
            Defaults to ``False``.
        keep_variant (bool, optional): Retain variant suffixes such as
            ``"add.Tensor"`` instead of returning only base names. Defaults to ``False``.

    Returns:
        set[str] or list[str]: Registered op names, sorted when requested.
    """
    registrations = torch._C._dispatch_get_registrations_for_dispatch_key("PrivateUse1")

    op_names = set()
    for reg in registrations:
        if "::" in reg:
            _, op = reg.split("::", 1)
            op_names.add(op if keep_variant else op.split(".")[0])

    return sorted(op_names) if as_sorted else op_names
