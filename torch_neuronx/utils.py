"""Utility functions for torch-neuronx."""

import os
import typing as _t
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
import torch
from neuronxcc.starfish.support import dtype as _ndt
from packaging.version import Version


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
        bool: True if TORCH_NEURONX_SYNC_MODE is set to "1"
    """
    return os.environ.get("TORCH_NEURONX_SYNC_MODE", "0") == "1"


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

    # Default fallback
    return torch.float32


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


# ==================== Version Info ====================

TORCH_VERSION = Version(torch.__version__)
