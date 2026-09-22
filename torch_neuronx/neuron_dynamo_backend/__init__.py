"""
Neuron Backend Utilities for torch.compile

This package provides a modular implementation of the Neuron backend for torch.compile,
organized into logical components for better maintainability and scalability.

Usage:

    # Set artifacts directory via environment variable (optional)
    # export TORCH_NEURONX_DEBUG_DIR=./my_artifacts

    # Set model name (optional)
    # from torch_neuronx.neuron_dynamo_backend import set_model_name
    # set_model_name("my_model")

    compiled_model = torch.compile(model, backend="neuron")

neuron_dynamo_backend environment variables:
    TORCH_NEURONX_NEURONX_CC_TIMEOUT: int (default=None)
        timeout in seconds when invoking neuronx-cc
    TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS: bool (default=False)
        whether to preserve directory with compilation artifacts after execution is complete
    TORCH_NEURONX_DEBUG_DIR: str (default="/tmp/neuron_backend_<random string>")
        directory in which to save compilation artifacts
    TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION: bool (default=False)
        disables fallback execution on failure for neuron backend.
    TORCH_NEURONX_ENABLE_NKI_SDPA: bool (default=True)
        enables decomposing scaled_dot_product_attention to the NKI kernel if possible.
        currently, the decomposition happens only when seqlen % 2048 == 0 for queries.
    TORCH_NEURONX_ENABLE_NKI_FREE_VAR: bool (default=False)
        enables free-variable (score_mod_other_buffers / mask_mod_other_buffers)
        plumbing through the NKI flex_attention lowering and kernel templates.
    NEURON_CC_FLAGS: str (default="")
        flags passed to neuronx-cc call
"""

import contextlib

# Register MegaCache artifact type for torch.compiler.save/load_cache_artifacts()
import torch

# Eagerly import torch_mlir.extras.fx_importer so its module-level
# LITERAL_CONVERTER_MAP registrations (notably torch.device) run exactly
# once. With dual-namespace torch_mlir installs, deferring this until
# the lazy backend import below can leave a second LITERAL_CONVERTER_MAP
# without those registrations, causing _import_argument to raise
# "Unsupported argument type <class 'torch.device'>".
with contextlib.suppress(ImportError):
    import torch_mlir.extras.fx_importer

from torch_neuronx.neuron_dynamo_backend import cache_artifact as _cache_artifact
from torch_neuronx.neuron_dynamo_backend.config import (
    get_model_name,
    get_rank,
    set_model_name,
)
from torch_neuronx.neuron_dynamo_backend.exceptions import (
    NEFFCompilationError,
    NEFFExecutionError,
)
from torch_neuronx.neuron_dynamo_backend.metrics import (
    get_dynamo_metrics,
    reset_dynamo_metrics,
)

__all__ = [
    "NEFFExecutionError",
    "create_neuron_backend",
    "fx_to_mlir_string",
    "get_dynamo_metrics",
    "get_model_name",
    "get_rank",
    "load_mlir",
    "neuron_backend",
    "reset_dynamo_metrics",
    "save_fx_as_mlir",
    "save_fx_graph_all_formats",
    "save_fx_graph_txt",
    "save_mlir",
    "set_model_name",
]

# Lazy imports for symbols that depend on torch_mlir.
# These are deferred to break a circular import: torch_mlir.compiler_utils imports
# torch, which autoloads torch_neuronx, which would import torch_mlir back before
# it has finished initializing.
_LAZY_BACKEND_IMPORTS = {
    "create_neuron_backend",
    "neuron_backend",
}
_LAZY_IO_IMPORTS = {
    "fx_to_mlir_string",
    "load_mlir",
    "save_fx_as_mlir",
    "save_fx_graph_all_formats",
    "save_fx_graph_txt",
    "save_mlir",
}


def __getattr__(name):
    if name in _LAZY_BACKEND_IMPORTS:
        from torch_neuronx.neuron_dynamo_backend.backend import (
            create_neuron_backend,
            neuron_backend,
        )

        globals().update(
            create_neuron_backend=create_neuron_backend,
            neuron_backend=neuron_backend,
        )
        return globals()[name]

    if name in _LAZY_IO_IMPORTS:
        from torch_neuronx.neuron_dynamo_backend.utils.io_utils import (
            fx_to_mlir_string,
            load_mlir,
            save_fx_as_mlir,
            save_fx_graph_all_formats,
            save_fx_graph_txt,
            save_mlir,
        )

        globals().update(
            fx_to_mlir_string=fx_to_mlir_string,
            load_mlir=load_mlir,
            save_fx_as_mlir=save_fx_as_mlir,
            save_fx_graph_all_formats=save_fx_graph_all_formats,
            save_fx_graph_txt=save_fx_graph_txt,
            save_mlir=save_mlir,
        )
        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Flex attention compatibility patches for Neuron backend
import torch.nn.attention.flex_attention as _flex_attn

# Bypass device validation for Neuron (upstream only allows CUDA/CPU)
_original_validate_device = _flex_attn._validate_device


def _patched_validate_device(query, key, value):
    if query.device.type == "neuron":
        return
    return _original_validate_device(query, key, value)


_flex_attn._validate_device = _patched_validate_device

# Register autocast fallthrough for flex_attention HOPs
torch.ops.higher_order.flex_attention.fallthrough(torch._C.DispatchKey.AutocastPrivateUse1)
torch.ops.higher_order.flex_attention_backward.fallthrough(torch._C.DispatchKey.AutocastPrivateUse1)

if torch._dynamo.config.caching_precompile:
    _cache_artifact._apply_precompile_patches()
