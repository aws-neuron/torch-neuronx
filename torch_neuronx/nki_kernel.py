"""NKI kernel support for torch-neuronx (CompileKernel-based)."""

import base64
import json
from dataclasses import dataclass, replace
from typing import Any

import torch
from nki.compiler.ncc_driver import CompilationMode, CompileOptions
from nki.framework.compiled import CompileKernel
from nki.language.buffers import shared_hbm
from nki.language.tensor import NkiTensor

from .utils import get_platform_target

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
        """
        converted = {name: _convert_torch_input(v, name) for name, v in kwargs.items()}

        # Lightweight opts for cache key generation
        cache_opts = CompileOptions(
            lnc=self.lnc,
            enable_simulation=self._enable_simulation,
            enable_device_dump=self._enable_device_dump,
        )

        # Check dump_config cache
        dc_cache = self._get_dump_config_cache()
        cache_key = None
        if dc_cache is not None:
            cache_key = self._generate_cache_key(converted, cache_opts)

            if cache_key is not None and cache_key in dc_cache:
                return dc_cache[cache_key]

        frontend = self._frontend_cls(
            enable_backend_opt=self._enable_backend_opt,
        )
        compile_opts = replace(self._compile_opts(), mode=CompilationMode.INTEGRATION)

        # TODO[NKI-1279]: deterministic artifacts directory
        """
        if not compile_opts.artifacts_dir:
            artifacts_dir = str(get_artifacts_directory().resolve()) + '/nki'
            compile_opts = replace(
                compile_opts,
                artifacts_dir=artifacts_dir,
                output_path=os.path.join(artifacts_dir, "kernel.neff"),
            )
        """

        bir, result = self._cached_compile_to_bir(
            frontend=frontend,
            inputs=converted,
            compile_opts=compile_opts,
        )

        backend_config = self._build_backend_config(bir, result)

        # (dtype_str, shape) per output
        return_types = [(spec.dtype, spec.shape) for spec in result.output_specs]

        # input_idx → output_idx  (inverted from CompileKernel's
        # output_idx → input_name mapping)
        aliases = result.input_output_aliases or {}
        input_name_to_idx = {spec.name: idx for idx, spec in enumerate(result.input_specs)}
        operand_output_aliases: dict[int, int] = {}
        for output_idx, input_name in aliases.items():
            input_idx = input_name_to_idx.get(input_name)
            if input_idx is not None:
                operand_output_aliases[input_idx] = output_idx

        dumped = base64.b64encode(json.dumps(backend_config).encode()).decode("ascii")

        result = DumpedConfig(
            return_types=return_types,
            dumped_config=dumped,
            operand_output_aliases=operand_output_aliases,
        )
        if dc_cache is not None and cache_key is not None:
            dc_cache[cache_key] = result
        return result
