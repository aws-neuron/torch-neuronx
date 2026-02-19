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
    NEURON_CC_FLAGS: str (default="")
        flags passed to neuronx-cc call
"""

# Register MegaCache artifact type for torch.compiler.save/load_cache_artifacts()
from torch_neuronx.neuron_dynamo_backend import cache_artifact as _cache_artifact
from torch_neuronx.neuron_dynamo_backend.backend import (
    create_neuron_backend,
    neuron_backend,
)
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
from torch_neuronx.neuron_dynamo_backend.utils.io_utils import (
    fx_to_mlir_string,
    load_mlir,
    save_fx_as_mlir,
    save_fx_graph_all_formats,
    save_fx_graph_txt,
    save_mlir,
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
