"""NKI kernel support for torch-neuronx (CompileKernel-based)."""

import hashlib
import os
from dataclasses import dataclass
from typing import Any

import torch
from nki.compiler.ncc_driver import CompileOptions
from nki.framework.compiled import CompileKernel
from nki.language.buffers import shared_hbm
from nki.language.tensor import NkiTensor
from torch.utils._filelock import FileLock

from .utils import get_platform_target, global_version_hash

# Torch dtype → NKI language dtype string.
# NKI language dtypes are plain strings (e.g. "float32", "bfloat16").
_TORCH_TO_NKI_DTYPE: dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.float64: "float32",  # NKI has no float64; downcast
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int64: "int32",  # NKI has no int64; downcast
    torch.int32: "int32",
    torch.int16: "int16",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.uint16: "uint16",
    torch.uint32: "uint32",
    torch.bool: "uint8",
}

# Conditionally add float8 mappings when the torch build supports them.
if hasattr(torch, "float8_e5m2"):
    _TORCH_TO_NKI_DTYPE[torch.float8_e5m2] = "float8_e5m2"
if hasattr(torch, "float8_e4m3fn"):
    _TORCH_TO_NKI_DTYPE[torch.float8_e4m3fn] = "float8_e4m3fn"
if hasattr(torch, "float8_e4m3fnuz"):
    _TORCH_TO_NKI_DTYPE[torch.float8_e4m3fnuz] = "float8_e4m3fn"
if hasattr(torch, "float8_e5m2fnuz"):
    _TORCH_TO_NKI_DTYPE[torch.float8_e5m2fnuz] = "float8_e5m2"


def _torch_to_nki_dtype(dtype: torch.dtype) -> str:
    """Convert a PyTorch dtype to an NKI language dtype string."""
    nki_dtype = _TORCH_TO_NKI_DTYPE.get(dtype)
    if nki_dtype is None:
        raise TypeError(
            f"Unsupported torch dtype for NKI: {dtype}. "
            f"Supported: {list(_TORCH_TO_NKI_DTYPE.keys())}"
        )
    return nki_dtype


def _convert_torch_input(x: Any, name: str) -> Any:
    """Convert a torch.Tensor to an NkiTensor for CompileKernel tracing.

    Non-tensor arguments are passed through unchanged.
    """
    if isinstance(x, torch.Tensor):
        return NkiTensor(
            name=name,
            shape=x.shape,
            dtype=_torch_to_nki_dtype(x.dtype),
            storage=None,
            buffer=shared_hbm,
        )
    return x


@dataclass
class DumpedConfig:
    """Result of ``TorchNeuronNKIKernelV3.dump_config``.

    Attributes:
        return_types: list of (dtype_str, shape) tuples for each output.
        dumped_config: base64-encoded JSON backend configuration.
        operand_output_aliases: mapping of input_idx → output_idx for aliased tensors.
    """

    return_types: list[tuple[Any, tuple[int, ...]]]
    dumped_config: str
    operand_output_aliases: dict[int, int]


@dataclass(frozen=True)
class TorchNeuronNKIKernelV3(CompileKernel):
    """CompileKernel subclass for torch-neuronx eager / torch.compile execution.

    Overrides input conversion so that the compile pipeline can accept
    ``torch.Tensor`` arguments directly, converting them to ``NkiTensor``
    descriptors before handing off to the NKI compiler frontend.
    """

    def _convert_input(self, x: Any, name: str) -> Any:
        """Convert torch.Tensor → NkiTensor; pass through everything else."""
        return _convert_torch_input(x, name)

    def _platform_target(self, opts):
        """Return the Neuron hardware target (e.g. 'trn1', 'trn2')."""
        return get_platform_target()

    def _get_dump_config_cache(self) -> dict | None:
        """Return (or create) the dump_config cache on self.func"""
        if self._get_compile_cache() is None:
            return None
        if not hasattr(self.func, "_dump_config_cache"):
            self.func._dump_config_cache = {}
        return self.func._dump_config_cache

    def dump_config(self, **kwargs: Any) -> DumpedConfig:
        """Compile the kernel and return a DumpedConfig.

        This is the single entry-point used by the HOP dispatch layer
        and by flex-attention lowering.  Usage::

            dconfig = kernel[grid].dump_config(**meta_args)

        where ``grid`` sets the LNC via ``__getitem__`` and
        ``meta_args`` is a ``{name: value}`` dict of kernel arguments
        (torch.Tensor values are converted to NkiTensor automatically).

        Caching layers (checked in order):
        1. In-memory cache
        2. Disk cache with file lock (cross-rank coordination)
        3. Full NKI compilation (only one rank per unique key)
        """
        converted = {name: _convert_torch_input(v, name) for name, v in kwargs.items()}

        # Lightweight opts for cache key generation
        cache_opts = CompileOptions(
            lnc=self.lnc,
            enable_simulation=self._enable_simulation,
            enable_device_dump=self._enable_device_dump,
        )

        # 1. Check in-memory cache
        dc_cache = self._get_dump_config_cache()
        cache_key = None
        if dc_cache is not None:
            cache_key = self._generate_cache_key(converted, cache_opts)

            if cache_key is not None and cache_key in dc_cache:
                return dc_cache[cache_key]

        # 2. Check persistent cache (cross-rank coordination)
        #    Disabled by default until multi-file kernel source hashing is robust.
        #    Enable with NKI_DUMP_PERSISTENT_CACHE=1.
        if cache_key is not None and os.environ.get("NKI_DUMP_PERSISTENT_CACHE", "0") == "1":
            from torch_neuronx.python_ops.compilation import hlo_cache
            from torch_neuronx.python_ops.compilation.hlo_cache import CACHE_DIR

            # Build persistent key from: kernel source file hash (invalidates on any
            # edit to the file, including constants outside the function body),
            # kernel identity (qualname), input signature (cache_key), and
            # package versions (global_version_hash).
            source_path = self.func.__code__.co_filename
            try:
                with open(source_path, "rb") as f:
                    source_hash = hashlib.sha256(f.read()).hexdigest()[:16]
            except OSError:
                source_hash = hashlib.sha256(self.func.__code__.co_code).hexdigest()[:16]

            persistent_key = hashlib.sha256(
                (
                    self.func.__qualname__
                    + ":"
                    + source_hash
                    + ":"
                    + cache_key
                    + ":"
                    + global_version_hash()
                ).encode()
            ).hexdigest()

            # Fast path: check persistent cache (no lock)
            disk_entry = hlo_cache.get_dump_config(persistent_key)
            if disk_entry is not None:
                if dc_cache is not None:
                    dc_cache[cache_key] = disk_entry
                return disk_entry

            # Slow path: file lock so only one rank compiles per key
            lock_path = os.path.join(CACHE_DIR, persistent_key + "_dump_config.lock")
            with FileLock(lock_path, timeout=1800):
                # Double-check after acquiring lock
                disk_entry = hlo_cache.get_dump_config(persistent_key)
                if disk_entry is not None:
                    if dc_cache is not None:
                        dc_cache[cache_key] = disk_entry
                    return disk_entry

                dconfig = self._compile_dump_config(converted, cache_opts)
                hlo_cache.put_dump_config(persistent_key, dconfig)

            if dc_cache is not None:
                dc_cache[cache_key] = dconfig
            return dconfig

        # No cache key (unhashable args) or persistent cache disabled — compile directly
        dconfig = self._compile_dump_config(converted, cache_opts)
        if dc_cache is not None and cache_key is not None:
            dc_cache[cache_key] = dconfig
        return dconfig

    def _compile_dump_config(self, converted: dict, cache_opts: CompileOptions) -> DumpedConfig:
        """Compile to BIR to build the backend config."""
        compile_opts = self._compile_opts()

        nir = self._cached_compile_to_bir(
            frontend=self._frontend_cls(enable_backend_opt=self._enable_backend_opt),
            inputs=converted,
            compile_opts=compile_opts,
        )

        config = nir.build_config()

        return DumpedConfig(
            return_types=[(spec.dtype, spec.shape) for spec in config.output_specs],
            dumped_config=config.backend_config_b64.decode("ascii"),
            operand_output_aliases=config.operand_output_aliases,
        )
