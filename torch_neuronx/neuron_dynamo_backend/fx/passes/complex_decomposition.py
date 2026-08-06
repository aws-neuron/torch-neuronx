"""Decompose complex FX graphs into real arithmetic.

Retracing with ``ComplexTensor`` preserves native real/imag input boundaries
and complex output boundaries.

Requires PyTorch >= 2.13 for ``decompose_complex_in_graph``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch.fx as fx
from packaging.version import Version

from torch_neuronx.neuron_dynamo_backend.settings import _getenv_bool
from torch_neuronx.utils import TORCH_VERSION

logger = logging.getLogger(__name__)

_COMPLEX_DECOMP_AVAILABLE = Version("2.13.0.dev0") <= TORCH_VERSION

if _COMPLEX_DECOMP_AVAILABLE:
    from torch._functorch._aot_autograd.complex_decomposition import (
        decompose_complex_in_graph,
    )
else:
    decompose_complex_in_graph = None


def _prepare_graph_for_lowering(gm: fx.GraphModule) -> None:
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()


def _decompose_complex(
    gm: fx.GraphModule,
    flat_args: list[Any],
    *,
    path_name: str,
) -> fx.GraphModule:
    if not _COMPLEX_DECOMP_AVAILABLE:
        raise RuntimeError(
            f"TORCH_NEURONX_COMPLEX_DECOMPOSE requires PyTorch >= 2.13; found {TORCH_VERSION}."
        )

    try:
        decomposed = decompose_complex_in_graph(gm, flat_args, decompositions={})
        if decomposed is gm:
            return gm
        _prepare_graph_for_lowering(decomposed)
        logger.debug("Applied complex decomposition for the %s path.", path_name)
        return decomposed
    except Exception as error:
        raise RuntimeError(f"Complex decomposition failed for the {path_name} path.") from error


def maybe_decompose_complex(gm: fx.GraphModule, example_inputs: Any) -> fx.GraphModule:
    """Decompose a torch.compile graph when framework decomposition is enabled."""
    if not _getenv_bool("TORCH_NEURONX_COMPLEX_DECOMPOSE", False):
        return gm

    try:
        return _decompose_complex(gm, list(example_inputs), path_name="torch.compile")
    except Exception as error:
        if _getenv_bool("TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION", False):
            raise
        logger.warning("%s Using the original graph for fallback.", error)
        return gm


def eager_decompose_complex(gm: fx.GraphModule, flat_args: list[Any]) -> fx.GraphModule:
    """Decompose an eager op selected by the C++ complex dispatch route."""
    return _decompose_complex(gm, flat_args, path_name="eager")
