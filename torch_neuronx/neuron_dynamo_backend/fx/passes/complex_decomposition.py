"""Decompose complex FX graphs into real arithmetic.

Retracing with ``ComplexTensor`` preserves native real/imag input boundaries
and complex output boundaries.

Requires PyTorch >= 2.13 for ``decompose_complex_in_graph``.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from copy import copy
from typing import Any

import torch
import torch.fx as fx
from packaging.version import Version
from torch._guards import detect_fake_mode

from torch_neuronx.neuron_dynamo_backend.settings import _getenv_bool
from torch_neuronx.utils import TORCH_VERSION

logger = logging.getLogger(__name__)

_COMPLEX_DECOMP_AVAILABLE = Version("2.13.0.dev0") <= TORCH_VERSION

if _COMPLEX_DECOMP_AVAILABLE:
    from torch._functorch._aot_autograd.complex_decomposition import (
        decompose_complex_in_graph,
    )
    from torch._subclasses.complex_tensor import WrapComplexMode
else:
    decompose_complex_in_graph = None


def _prepare_graph_for_lowering(gm: fx.GraphModule) -> None:
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()


def _restore_hierarchy_metadata(original: fx.GraphModule, decomposed: fx.GraphModule) -> None:
    """Keep compiler partition boundaries when retracing backward nodes."""
    # PyTorch torch/fx/proxy.py: TracerBase.create_node copies only _COPY_META_FIELDS,
    # even under preserve_node_meta(). The list keeps from_node but omits the fields below.
    # PyTorch torch/fx/traceback.py: set_current_meta records the source node in from_node.
    # Use that source to restore the layer metadata after the make_fx retrace.
    sources = {node.name: node for node in original.graph.nodes}
    for node in decomposed.graph.nodes:
        for source in node.meta.get("from_node", []):
            if source.graph_id != id(original.graph):
                continue
            original_node = sources[source.name]
            for field in ("fwd_nn_module_stack", "call_hierarchy"):
                if field in original_node.meta:
                    node.meta[field] = copy(original_node.meta[field])


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
        from torch_neuronx.neuron_dynamo_backend.decompositions import (
            get_decomposition_table,
        )

        # Factory ops need the input fake mode to avoid real tensors during retracing.
        fake_mode = detect_fake_mode(flat_args)
        with fake_mode if fake_mode is not None else nullcontext():
            decomposed = decompose_complex_in_graph(
                gm,
                flat_args,
                decompositions=get_decomposition_table(),
            )
        if decomposed is gm:
            return gm
        _restore_hierarchy_metadata(gm, decomposed)
        _prepare_graph_for_lowering(decomposed)
        logger.debug("Applied complex decomposition for the %s path.", path_name)
        return decomposed
    except Exception as error:
        # Cause goes in the message: dynamo re-raises backend failures `from None`.
        logger.error("Complex decomposition failed for the %s path.", path_name, exc_info=True)
        raise RuntimeError(
            f"Complex decomposition failed for the {path_name} path: "
            f"{type(error).__name__}: {error}"
        ) from error


def maybe_decompose_complex(gm: fx.GraphModule, example_inputs: Any) -> fx.GraphModule:
    """Decompose a Dynamo backend graph when complex decomposition is enabled.

    Args:
        gm (fx.GraphModule): FX graph module to decompose.
        example_inputs (Any): Example inputs used to retrace complex operations.

    Returns:
        fx.GraphModule: Decomposed graph, or the original graph when disabled or falling back.

    Raises:
        RuntimeError: If decomposition is requested but unavailable, or decomposition fails,
            while fallback execution is disabled.
    """
    if not _getenv_bool("TORCH_NEURONX_COMPLEX_DECOMPOSE", _COMPLEX_DECOMP_AVAILABLE):
        return gm

    try:
        return _decompose_complex(gm, list(example_inputs), path_name="torch.compile")
    except Exception as error:
        if _getenv_bool("TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION", False):
            raise
        logger.warning("%s Using the original graph for fallback.", error)
        return gm


def eager_decompose_complex(gm: fx.GraphModule, flat_args: list[Any]) -> fx.GraphModule:
    """Decompose an eager op selected by the C++ complex dispatch route.

    Args:
        gm (fx.GraphModule): FX graph module to decompose.
        flat_args (list[Any]): Flattened arguments used to retrace complex operations.

    Returns:
        fx.GraphModule: Decomposed graph module.

    Raises:
        RuntimeError: If PyTorch lacks complex decomposition or decomposition fails.
    """
    return _decompose_complex(gm, flat_args, path_name="eager")


def _register_complex_new_full() -> None:
    """Add an ``aten.new_full`` rule to the upstream ComplexTensor op table.

    ComplexTensor registers ``new_zeros`` and ``full_like`` but not ``new_full``.
    So ``new_full`` on a complex tensor aborts the retrace. Build the result from
    the real and imag parts, the same way the sibling factory rules do.
    """
    from torch._subclasses.complex_tensor._core import ComplexTensor
    from torch._subclasses.complex_tensor._ops import common as ct_common

    aten = torch.ops.aten
    if aten.new_full in ct_common.COMPLEX_OPS_TABLE:
        return

    def _new_full_impl(input, size, fill_value, *args, dtype=None, **kwargs):
        input_r, input_i = ct_common.split_complex_tensor(input)
        if dtype is not None and dtype not in ct_common.COMPLEX_TO_REAL:
            return aten.new_full.default(input_r, size, fill_value, *args, dtype=dtype, **kwargs)
        if dtype is not None:
            kwargs["dtype"] = ct_common.COMPLEX_TO_REAL[dtype]
        fill_r, fill_i = ct_common.split_complex_arg(fill_value)
        ret_r = aten.new_full.default(input_r, size, fill_r, *args, **kwargs)
        ret_i = aten.new_full.default(input_i, size, fill_i, *args, **kwargs)
        return ComplexTensor(ret_r, ret_i)

    ct_common.register_complex(aten.new_full, _new_full_impl)
    logger.info("Registered ComplexTensor new_full rule.")


def apply_complex_decomposition_patches() -> None:
    """Register missing ComplexTensor rules for the retrace on Neuron.

    Call this once at package boot. Existing rules stay unchanged.
    Remove each patch when the minimum supported PyTorch version has its rule.
    """
    if not _COMPLEX_DECOMP_AVAILABLE:
        return
    try:
        _register_complex_new_full()
        WrapComplexMode.supports_higher_order_operators = True
    except Exception:
        logger.exception("Failed to register the ComplexTensor rules.")
