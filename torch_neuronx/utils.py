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


def deterministic_hash(obj) -> str:
    """Generate a deterministic hash string from an object.

    Uses SHA-256 of the object's repr for cross-process consistency.
    """
    return hashlib.sha256(repr(obj).encode()).hexdigest()[:16]


@functools.lru_cache(maxsize=1)
def global_version_hash() -> str:
    """Version hash combining source code + package versions.

    Hashes all .py files in torch_neuronx lowering directories plus
    package versions into a single SHA-256 digest. Used to invalidate
    persistent caches when code or package versions change.

    """
    import torch_neuronx

    hasher = hashlib.sha256()

    # Hash source files
    root = Path(torch_neuronx.__file__).parent
    for d in [root / "python_ops", root / "neuron_dynamo_backend"]:
        if d.exists():
            for py_file in sorted(d.rglob("*.py")):
                hasher.update(str(py_file.relative_to(root)).encode())
                hasher.update(py_file.read_bytes())

    # Hash package versions
    hasher.update(torch.__version__.encode())
    hasher.update(getattr(torch_neuronx, "__version__", "unknown").encode())
    try:
        import torch_mlir

        hasher.update(getattr(torch_mlir, "__version__", "unknown").encode())
    except Exception:
        pass
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

    return hasher.hexdigest()


# ==================== Tensor Utilities ====================


def flatten_tensors(obj) -> list[torch.Tensor]:
    """
    Recursively flatten nested tensor structures into a flat list.

    Handles: Tensor, List[Tensor], List[List[Tensor]], etc.

    Args:
        obj: Single tensor, list, tuple, or nested structure of tensors

    Returns:
        Flat list of all tensors found in the structure
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
    """
    Extract device from tensor inputs.

    Gets device from first tensor encountered (early return, avoids flattening all).
    Falls back to current_device() when no tensors (e.g., barrier()).

    Args:
        tensors: Single tensor, list, or nested structure of tensors

    Returns:
        torch.device for the tensors, or current device if no tensors
    """
    import torch_neuronx

    dev = _get_first_tensor_device(tensors)
    return dev if dev is not None else torch.device("neuron", torch_neuronx.current_device())


@contextmanager
def suppress_specific_warnings(patterns):
    """Context manager to temporarily suppress specific warnings.

    Args:
        patterns: List of warning message patterns to suppress.
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
    """Check if torch-mlir aten ops lowering is enabled

    Returns:
        bool: True if TORCH_NEURONX_MLIR_ATEN_OPS is set to "1"
    """
    return os.environ.get("TORCH_NEURONX_MLIR_ATEN_OPS", "0") == "1"


def is_sync_mode_enabled():
    """Check if legacy launch blocking is enabled via environment variable.

    Returns:
        bool: True if NEURON_LAUNCH_BLOCKING is set to "1"
    """
    return os.environ.get("NEURON_LAUNCH_BLOCKING", "0") == "1"


@functools.lru_cache(maxsize=1)
def is_host_cc_enabled():
    """Check if host-side collective communication is enabled.

    Host-cc enables collective operations on non-default streams using NRT host collectives API.
    Result is cached after first call since the env var is set once at process start.

    Returns:
        bool: True if TORCH_NEURONX_ENABLE_HOST_CC is "1"
    """
    return os.environ.get("TORCH_NEURONX_ENABLE_HOST_CC", "0") == "1"


def get_gmm_align() -> int:
    """Get the alignment constraint for grouped matmul offsets.

    Returns:
        int: Alignment value from TORCH_NEURONX_ATEN_GMM_ALIGN, default 128
    """
    return int(os.environ.get("TORCH_NEURONX_ATEN_GMM_ALIGN", "128"))


def skip_op_preconditions() -> bool:
    """Check if op precondition validation should be skipped.

    Returns:
        bool: True if TORCH_NEURONX_SKIP_OP_PRECONDITIONS is set to "1"
    """
    return os.environ.get("TORCH_NEURONX_SKIP_OP_PRECONDITIONS", "0") == "1"


def is_native_int64_enabled() -> bool:
    """Check if native int64 passthrough to compiler is enabled.

    .. deprecated::
        Use ``torch_neuronx.int64_policy.get_int64_mode()`` instead.
        This function now delegates to the new policy module which reads
        TORCH_NEURONX_INT64_MODE (preferred) or TORCH_NEURONX_NATIVE_INT64
        (legacy).

    Returns:
        bool: True when mode is NATIVE (TORCH_NEURONX_INT64_MODE=native
              or legacy TORCH_NEURONX_NATIVE_INT64=1).
    """
    from torch_neuronx.int64_policy import is_native_int64_mode

    return is_native_int64_mode()


def get_logical_neuron_cores():
    """Get the Logical Neuron Cores (LNC) setting based on environment.

    Returns:
        str: NEURON_LOGICAL_NC_CONFIG is specified, otherwise the default
             for the instance type.
    """
    return os.environ.get("NEURON_LOGICAL_NC_CONFIG", get_lnc_setting())


def get_worker_multiplier():
    """Get the worker multiplier based on LNC setting.

    Returns:
        int: 2 if LNC=1 (can use 2x logical cores) and TRN2, 1 otherwise
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
    """Log an operation that was executed on Neuron device.

    Args:
        op_name: Name of the operation that was executed
    """
    import torch_neuronx._C as _C

    _C._log_executed_op(op_name)


def log_offloaded_op(op_name: str):
    """Log an operation that fell back to CPU.

    Args:
        op_name: Name of the operation that fell back
    """
    import torch_neuronx._C as _C

    _C._log_offloaded_op(op_name)


# ==================== NEFF Cache Management ====================


def is_neff_cache_disabled():
    """Check if NEFF caching is disabled via environment variable.

    Returns:
        bool: True if TORCH_NEURONX_DISABLE_NEFF_CACHE is set to "1", "true", or "yes"
    """
    disable_cache = os.environ.get("TORCH_NEURONX_DISABLE_NEFF_CACHE", "").lower()
    return disable_cache in ["1", "true", "yes"]


# ==================== NEFF Cache Logging ====================


def log_neff_cache_hit(cache_key: str):
    """Log a NEFF cache hit.

    Args:
        cache_key: The cache key that was hit
    """
    import torch_neuronx._C as _C

    _C._log_neff_cache_hit(cache_key)


def log_neff_cache_miss(cache_key: str):
    """Log a NEFF cache miss.

    Args:
        cache_key: The cache key that was missed
    """
    import torch_neuronx._C as _C

    _C._log_neff_cache_miss(cache_key)


def log_neff_cache_store(cache_key: str):
    """Log storing a NEFF in the cache.

    Args:
        cache_key: The cache key being stored
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
    """Get the platform target for NKI kernels."""
    import torch_neuronx._C as _C

    return _C._get_platform_target()


def get_lnc_setting():
    """Get the LNC setting for the current hardware."""
    import torch_neuronx._C as _C

    return _C._get_logical_neuron_cores()


def move_pytree_to_cpu(obj):
    """Recursively convert tensors to CPU in any nested structure."""
    if isinstance(obj, torch.Tensor) and obj.device.type == "neuron":
        return obj.to("cpu")
    elif isinstance(obj, list | tuple):
        return type(obj)(move_pytree_to_cpu(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: move_pytree_to_cpu(v) for k, v in obj.items()}
    else:
        return obj


def move_pytree_to_neuron(obj):
    """Recursively convert tensors to Neuron device in any nested structure."""
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
    """Cast tensor to ``target_dtype`` if provided and different.

    This wrapper centralizes the common no-op guard before calling
    ``Tensor.to(dtype=...)``. It leverages the device's registered
    implementation for ``aten::_to_copy``; on Neuron, this currently
    performs dtype conversion via a CPU fallback when necessary and will
    automatically benefit from future in-device casts.

    Args:
        tensor: Input tensor
        target_dtype: Desired dtype, or ``None`` to keep as-is

    Returns:
        Tensor with requested dtype, or the original tensor if no change.
    """
    if target_dtype is None or tensor.dtype == target_dtype:
        return tensor
    return tensor.to(dtype=target_dtype)


def is_float8_e4m3_supported():
    return get_platform_target() == "trn2"


def is_float8_e4m3fn_supported():
    return get_platform_target() != "trn2"


def is_float8_e8m0fnu_supported():
    return get_platform_target() != "trn2"


def is_mxocp_supported():
    return get_platform_target() != "trn2"


_MXFP8_value_tensor_dtypes = Literal["float8_e4m3fn", "float8_e5m2"]


# Framework container for MXFP8 data and scale tensors
@dataclass
class MXFP8:
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
    """Map external dtype objects to torch.dtype.

    Supports:
    - NeuronXCC starfish dtype objects (e.g., ndt.bfloat16)
    - NumPy dtypes (np.float32, np.int32, np.dtype("float32"), etc.)
    - Falls back to torch.float32 when unknown

    Args:
        nki_or_numpy_dtype: External dtype object

    Returns:
        torch.dtype corresponding to the input dtype
    """
    # Already a torch dtype
    if isinstance(nki_or_numpy_dtype, torch.dtype):
        return nki_or_numpy_dtype

    # NeuronXCC starfish dtype: explicitly handle bf16 and float8 which lack a NumPy dtype
    if _ndt is not None:
        target = get_platform_target()
        if nki_or_numpy_dtype is getattr(_ndt, "bfloat16", object()):
            return torch.bfloat16
        if nki_or_numpy_dtype is getattr(_ndt, "float8_e5m2", object()):
            return torch.float8_e5m2
        if nki_or_numpy_dtype is getattr(_ndt, "float8_e4m3", object()):
            if is_float8_e4m3_supported():
                return torch.float8_e4m3fn
            raise RuntimeError(f"Unexpected dtype {nki_or_numpy_dtype} for platform {target}")
        if nki_or_numpy_dtype is getattr(_ndt, "float8_e4m3fn", object()):
            if not is_float8_e4m3fn_supported():
                raise RuntimeError(f"Unsupported dtype {nki_or_numpy_dtype} for platform {target}")
            return torch.float8_e4m3fn
        if nki_or_numpy_dtype is getattr(_ndt, "float8_e8m0fnu", object()):
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
    """Convert PyTorch dtype to Neuron dtype."""
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
    """Convert tensor to Neuron-compatible dtype if needed.

    Args:
        tensor: Input tensor
        non_blocking: Whether to use non-blocking transfers
        convert_int64: If True, convert int64 to int32. If False, keep int64 as-is
            (MLIR path handles S64 HLO natively). Defaults to True.

    Returns:
        Tensor with compatible dtype (float64->float32, and int64->int32 if convert_int64=True)
    """
    # Determine if conversion is needed
    if tensor.dtype == torch.int64 and convert_int64:
        # Packed mode: extract low words on-device (no CPU bounce)
        from torch_neuronx.int64_policy import is_packed_int64_mode

        if is_packed_int64_mode() and tensor.device.type == "neuron":
            from torch_neuronx.packed_storage import packed_int64_execution_view

            low = packed_int64_execution_view(tensor, contiguous=True)

            # Optional overflow check (off by default for performance)
            if os.environ.get("TORCH_NEURONX_PACKED_INT64_CHECK_OVERFLOW", "0") == "1":
                from torch_neuronx.packed_storage import packed_int64_high_word_alias

                # Gather the high lane. Contiguous → NKI kernel via the op (same
                # fast path as the low words); non-contiguous → general int32
                # copy of the lane alias.
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

            return low
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
    """Get op names registered for Neuron device (PrivateUse1).

    Args:
        as_sorted: If True, return sorted list; otherwise return set.
        keep_variant: If True, keep variant suffix (e.g., "add.Tensor"),
            otherwise return base name only.

    Returns:
        Set or sorted list of op names (e.g., {"add", "mul"} or ["add.Tensor", "mul"])
    """
    registrations = torch._C._dispatch_get_registrations_for_dispatch_key("PrivateUse1")

    op_names = set()
    for reg in registrations:
        if "::" in reg:
            _, op = reg.split("::", 1)
            op_names.add(op if keep_variant else op.split(".")[0])

    return sorted(op_names) if as_sorted else op_names
