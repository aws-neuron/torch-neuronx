"""
Shared utility functions for FX graph passes.
"""

import copy
import logging

import torch
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensorMode

logger = logging.getLogger(__name__)


def _add_metadata_to_node(node: torch.fx.Node, shape, dtype) -> torch.fx.Node:
    """Add metadata to a node for torch-mlir compatibility."""
    if not hasattr(node, "meta"):
        node.meta = {}
    with FakeTensorMode():
        node.meta["val"] = torch.empty(shape, dtype=dtype)
    return node


def _is_non_cpu_device(value) -> bool:
    """Return True iff ``value`` is a device specifier referring to a non-CPU
    device.
    """
    if isinstance(value, torch.device):
        return value.type != "cpu"
    if isinstance(value, str):
        return "cpu" not in value
    return False


def replace_device_with_cpu(gm: fx.GraphModule) -> fx.GraphModule:
    """Return a copy of ``gm`` with every non-CPU ``device`` value rewritten
    to CPU.

    Used before host-side replay of FX submodules whose trace baked in a
    non-CPU device literal.
    """
    new_graph = copy.deepcopy(gm.graph)
    modified = False
    for node in new_graph.nodes:
        if node.op not in ("call_function", "call_method"):
            continue

        if "device" in node.kwargs and _is_non_cpu_device(node.kwargs["device"]):
            new_kwargs = dict(node.kwargs)
            new_kwargs["device"] = torch.device("cpu")
            node.kwargs = new_kwargs
            modified = True

        new_args = None
        for i, a in enumerate(node.args):
            if _is_non_cpu_device(a):
                if new_args is None:
                    new_args = list(node.args)
                new_args[i] = torch.device("cpu")
        if new_args is not None:
            node.args = tuple(new_args)
            modified = True

    if not modified:
        return gm

    new_graph.lint()
    return fx.GraphModule(gm, new_graph)
