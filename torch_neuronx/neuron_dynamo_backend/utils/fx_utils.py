"""
Shared utility functions for FX graph passes.
"""

import logging

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

logger = logging.getLogger(__name__)


def _add_metadata_to_node(node: torch.fx.Node, shape, dtype) -> torch.fx.Node:
    """Add metadata to a node for torch-mlir compatibility."""
    if not hasattr(node, "meta"):
        node.meta = {}
    with FakeTensorMode():
        node.meta["val"] = torch.empty(shape, dtype=dtype)
    return node
