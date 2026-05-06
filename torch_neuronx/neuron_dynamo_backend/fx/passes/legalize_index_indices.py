"""
Legalize indices for indexing operations.

Handles:
1. Empty index slices (any integer index with a 0-sized dimension) — for
   index_put, replaces the node with its input tensor; for index, replaces
   with an empty tensor matching the expected output shape.
2. Negative indices — normalizes to positive equivalents (idx + dim_size).

Supported ops: aten.index.Tensor, aten.index_put.default
"""

import logging

import torch
from torch.fx import Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult

from torch_neuronx.neuron_dynamo_backend.utils.fx_utils import _add_metadata_to_node

logger = logging.getLogger(__name__)

_INDEX_OPS = {
    torch.ops.aten.index.Tensor,
    torch.ops.aten.index_put.default,
}


class LegalizeIndexIndices(PassBase):
    """Legalize indices in indexing ops (index and index_put).

    - If any integer index has a 0-sized dimension, the op is a no-op:
      index_put is replaced with its input tensor, index is replaced with
      an empty tensor matching the expected output shape.
    - Otherwise, normalizes negative integer indices to positive equivalents
      via: idx = torch.where(idx < 0, idx + dim_size, idx)
    """

    def call(self, gm: torch.fx.GraphModule) -> PassResult:
        graph = gm.graph
        modified = False

        index_nodes = [node for node in graph.nodes if self._is_index_op(node)]
        logger.debug(f"Found {len(index_nodes)} indexing nodes")

        for node in index_nodes:
            input_tensor = node.args[0]
            indices = node.args[1]

            index_metas = []
            for idx in indices:
                if idx is None:
                    index_metas.append(None)
                else:
                    val = idx.meta.get("val", None)
                    if val is None:
                        raise ValueError(f"Index node {idx} is missing 'val' in its metadata")
                    index_metas.append(val)

            has_empty_slice = any(
                meta is not None
                and meta.dtype in (torch.int32, torch.int64)
                and any(s == 0 for s in meta.shape)
                for meta in index_metas
            )
            if has_empty_slice:
                if node.target == torch.ops.aten.index_put.default:
                    logger.debug(
                        f"Node {node.name} (index_put) has empty index slice, "
                        "replacing with input"
                    )
                    node.replace_all_uses_with(input_tensor)
                else:
                    node_meta = node.meta.get("val", None)
                    if node_meta is None:
                        raise ValueError(f"Node {node.name} is missing 'val' in its metadata")
                    with graph.inserting_before(node):
                        empty_node = graph.call_function(
                            torch.ops.aten.empty.memory_format,
                            args=(list(node_meta.shape),),
                            kwargs={"dtype": node_meta.dtype},
                        )
                        _add_metadata_to_node(
                            empty_node, shape=node_meta.shape, dtype=node_meta.dtype
                        )
                    logger.debug(
                        f"Node {node.name} (index) has empty index slice, "
                        f"replacing with empty tensor of shape {list(node_meta.shape)}"
                    )
                    node.replace_all_uses_with(empty_node)
                graph.erase_node(node)
                modified = True
                continue

            new_indices = []
            node_modified = False

            for dim, (idx, idx_meta) in enumerate(zip(indices, index_metas, strict=False)):
                if idx is None:
                    new_indices.append(None)
                    continue

                if idx_meta.dtype not in (torch.int32, torch.int64):
                    logger.debug(
                        f"Node {node.name} dim {dim}: skipping non-integer "
                        f"index (dtype={idx_meta.dtype})"
                    )
                    new_indices.append(idx)
                    continue

                input_meta = input_tensor.meta.get("val", None)
                if input_meta is None:
                    new_indices.append(idx)
                    continue

                dim_size = input_meta.shape[dim]

                with graph.inserting_before(node):
                    zero = graph.call_function(
                        torch.ops.aten.scalar_tensor.default,
                        args=(0,),
                        kwargs={"dtype": idx_meta.dtype},
                    )
                    _add_metadata_to_node(zero, shape=(), dtype=idx_meta.dtype)
                    lt = graph.call_function(torch.ops.aten.lt.Tensor, args=(idx, zero))
                    _add_metadata_to_node(lt, shape=idx_meta.shape, dtype=torch.bool)
                    offset = graph.call_function(
                        torch.ops.aten.scalar_tensor.default,
                        args=(dim_size,),
                        kwargs={"dtype": idx_meta.dtype},
                    )
                    _add_metadata_to_node(offset, shape=(), dtype=idx_meta.dtype)
                    shifted = graph.call_function(torch.ops.aten.add.Tensor, args=(idx, offset))
                    _add_metadata_to_node(shifted, shape=idx_meta.shape, dtype=idx_meta.dtype)
                    normalized = graph.call_function(
                        torch.ops.aten.where.self, args=(lt, shifted, idx)
                    )
                    _add_metadata_to_node(normalized, shape=idx_meta.shape, dtype=idx_meta.dtype)

                new_indices.append(normalized)
                node_modified = True
                logger.debug(
                    f"Node {node.name} dim {dim}: normalized negative indices (dim_size={dim_size})"
                )

            if node_modified:
                node.args = (node.args[0], new_indices, *node.args[2:])
                modified = True

        if modified:
            graph.lint()
            gm.recompile()

        return PassResult(gm, modified=modified)

    @staticmethod
    def _is_index_op(node: Node) -> bool:
        return node.op == "call_function" and node.target in _INDEX_OPS
