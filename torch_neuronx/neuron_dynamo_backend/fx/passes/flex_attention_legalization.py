"""
Flex Attention Legalization Pass for PyTorch FX Graphs.

Legalizes flex_attention higher-order operators by decomposing them into
standard ATen operations that torch-mlir can handle.
"""

import logging
import operator
from dataclasses import dataclass

import torch
import torch.fx as fx
from torch.fx import Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.fx.passes.shape_prop import TensorMetadata

logger = logging.getLogger(__name__)


@dataclass
class AttentionShapes:
    """Container for attention tensor shapes."""

    B: int  # batch size
    H_q: int  # query heads
    H_kv: int  # key/value heads (may differ from H_q in GQA)
    L: int  # query sequence length
    S: int  # key/value sequence length
    E: int  # head dimension


class FlexAttentionLegalization(PassBase):
    """Flex Attention Legalization Pass for PyTorch FX Graphs.

    This pass legalizes flex_attention higher-order operators by decomposing
    them into standard ATen operations that torch-mlir can handle.

    Higher-order operators like flex_attention are not legalized through the
    standard decomposition table mechanism in AOTAutograd, so this pass
    handles them explicitly.
    """

    def __init__(self):
        super().__init__()
        self._forward_score_mod_name: str | None = None
        self._forward_score_mod_module: fx.GraphModule | None = None
        self._forward_mask_mod_module: fx.GraphModule | None = None

    @property
    def name(self) -> str:
        return "flex_attention_legalization"

    # =========================================================================
    # Helper Methods
    # =========================================================================

    @staticmethod
    def _add_metadata_to_node(
        node: fx.Node, example_tensor=None, shape=None, dtype=None
    ) -> fx.Node:
        """Helper to add metadata to a node for torch-mlir compatibility."""
        if not hasattr(node, "meta"):
            node.meta = {}

        if example_tensor is not None:
            node.meta["val"] = example_tensor
        elif shape is not None and dtype is not None:
            node.meta["val"] = torch.empty(shape, dtype=dtype)

        return node

    @staticmethod
    def _is_flex_attention_forward(node: Node) -> bool:
        """Check if node is a flex_attention forward operation."""
        return node.op == "call_function" and node.target == torch.ops.higher_order.flex_attention

    @staticmethod
    def _is_flex_attention_backward(node: Node) -> bool:
        """Check if node is a flex_attention backward operation."""
        return (
            node.op == "call_function"
            and hasattr(node.target, "__name__")
            and node.target.__name__ == "flex_attention_backward"
        )

    # =========================================================================
    # GQA (Grouped Query Attention) Helpers
    # =========================================================================

    def _expand_kv_for_gqa(
        self, graph: fx.Graph, key: fx.Node, value: fx.Node, shapes: AttentionShapes
    ) -> tuple[fx.Node, fx.Node]:
        """
        Expand key/value tensors for Grouped Query Attention (GQA).

        When H_q != H_kv, we need to repeat K/V heads to match Q heads.
        """
        if shapes.H_q == shapes.H_kv:
            return key, value

        if shapes.H_q % shapes.H_kv != 0:
            raise ValueError(f"H_q ({shapes.H_q}) must be multiple of H_kv ({shapes.H_kv}) for GQA")

        logger.debug(f"Handling GQA: expanding H_kv={shapes.H_kv} to H_q={shapes.H_q}")

        repeat_factor = shapes.H_q // shapes.H_kv
        key_dtype = key.meta["val"].dtype
        value_dtype = value.meta["val"].dtype

        key_unsqueezed = graph.call_function(torch.ops.aten.unsqueeze.default, args=(key, 2))
        self._add_metadata_to_node(
            key_unsqueezed,
            shape=[shapes.B, shapes.H_kv, 1, shapes.S, shapes.E],
            dtype=key_dtype,
        )

        expand_shape = [shapes.B, shapes.H_kv, repeat_factor, shapes.S, shapes.E]
        key_expanded = graph.call_function(
            torch.ops.aten.expand.default, args=(key_unsqueezed, expand_shape)
        )
        self._add_metadata_to_node(key_expanded, shape=expand_shape, dtype=key_dtype)

        reshape_shape = [shapes.B, shapes.H_q, shapes.S, shapes.E]
        key_final = graph.call_function(
            torch.ops.aten.reshape.default, args=(key_expanded, reshape_shape)
        )
        self._add_metadata_to_node(key_final, shape=reshape_shape, dtype=key_dtype)

        # Expand value: same transformation
        value_unsqueezed = graph.call_function(torch.ops.aten.unsqueeze.default, args=(value, 2))
        self._add_metadata_to_node(
            value_unsqueezed,
            shape=[shapes.B, shapes.H_kv, 1, shapes.S, shapes.E],
            dtype=value_dtype,
        )

        value_expanded = graph.call_function(
            torch.ops.aten.expand.default, args=(value_unsqueezed, expand_shape)
        )
        self._add_metadata_to_node(value_expanded, shape=expand_shape, dtype=value_dtype)

        value_final = graph.call_function(
            torch.ops.aten.reshape.default, args=(value_expanded, reshape_shape)
        )
        self._add_metadata_to_node(value_final, shape=reshape_shape, dtype=value_dtype)

        return key_final, value_final

    def _reduce_gqa_gradients(
        self,
        graph: fx.Graph,
        grad_key: fx.Node,
        grad_value: fx.Node,
        shapes: AttentionShapes,
        dtype: torch.dtype,
    ) -> tuple[fx.Node, fx.Node]:
        """Reduce gradients for GQA by summing across repeated heads."""
        if shapes.H_q == shapes.H_kv:
            return grad_key, grad_value

        logger.debug(f"Reducing gradients for GQA: H_q={shapes.H_q} to H_kv={shapes.H_kv}")

        repeat_factor = shapes.H_q // shapes.H_kv

        # Reshape and sum grad_key
        grad_key_reshaped = graph.call_function(
            torch.ops.aten.reshape.default,
            args=(
                grad_key,
                [shapes.B, shapes.H_kv, repeat_factor, shapes.S, shapes.E],
            ),
        )
        self._add_metadata_to_node(
            grad_key_reshaped,
            shape=[shapes.B, shapes.H_kv, repeat_factor, shapes.S, shapes.E],
            dtype=dtype,
        )

        grad_key_final = graph.call_function(
            torch.ops.aten.sum.dim_IntList, args=(grad_key_reshaped, [2], False)
        )
        self._add_metadata_to_node(
            grad_key_final,
            shape=[shapes.B, shapes.H_kv, shapes.S, shapes.E],
            dtype=dtype,
        )

        # Reshape and sum grad_value
        grad_value_reshaped = graph.call_function(
            torch.ops.aten.reshape.default,
            args=(
                grad_value,
                [shapes.B, shapes.H_kv, repeat_factor, shapes.S, shapes.E],
            ),
        )
        self._add_metadata_to_node(
            grad_value_reshaped,
            shape=[shapes.B, shapes.H_kv, repeat_factor, shapes.S, shapes.E],
            dtype=dtype,
        )

        grad_value_final = graph.call_function(
            torch.ops.aten.sum.dim_IntList, args=(grad_value_reshaped, [2], False)
        )
        self._add_metadata_to_node(
            grad_value_final,
            shape=[shapes.B, shapes.H_kv, shapes.S, shapes.E],
            dtype=dtype,
        )

        return grad_key_final, grad_value_final

    # =========================================================================
    # Attention Computation Helpers
    # =========================================================================

    def _create_index_tensors(
        self, graph: fx.Graph, shapes: AttentionShapes, device: torch.device | None = None
    ) -> tuple[fx.Node, fx.Node, fx.Node, fx.Node]:
        """Create index tensors for score_mod: b_idx, h_idx, q_idx, kv_idx."""
        kwargs = {"dtype": torch.int32}
        if device is not None:
            kwargs["device"] = device

        idx_dtype = torch.int32

        # Batch index: [B, 1, 1, 1]
        b_idx_arange = graph.call_function(
            torch.ops.aten.arange.default, args=(shapes.B,), kwargs=kwargs
        )
        self._add_metadata_to_node(b_idx_arange, shape=[shapes.B], dtype=idx_dtype)
        b_idx = graph.call_function(
            torch.ops.aten.view.default, args=(b_idx_arange, [shapes.B, 1, 1, 1])
        )
        self._add_metadata_to_node(b_idx, shape=[shapes.B, 1, 1, 1], dtype=idx_dtype)

        # Head index: [1, H, 1, 1]
        h_idx_arange = graph.call_function(
            torch.ops.aten.arange.default, args=(shapes.H_q,), kwargs=kwargs
        )
        self._add_metadata_to_node(h_idx_arange, shape=[shapes.H_q], dtype=idx_dtype)
        h_idx = graph.call_function(
            torch.ops.aten.view.default, args=(h_idx_arange, [1, shapes.H_q, 1, 1])
        )
        self._add_metadata_to_node(h_idx, shape=[1, shapes.H_q, 1, 1], dtype=idx_dtype)

        # Query index: [1, 1, L, 1]
        q_idx_arange = graph.call_function(
            torch.ops.aten.arange.default, args=(shapes.L,), kwargs=kwargs
        )
        self._add_metadata_to_node(q_idx_arange, shape=[shapes.L], dtype=idx_dtype)
        q_idx = graph.call_function(
            torch.ops.aten.view.default, args=(q_idx_arange, [1, 1, shapes.L, 1])
        )
        self._add_metadata_to_node(q_idx, shape=[1, 1, shapes.L, 1], dtype=idx_dtype)

        # Key/Value index: [1, 1, 1, S]
        kv_idx_arange = graph.call_function(
            torch.ops.aten.arange.default, args=(shapes.S,), kwargs=kwargs
        )
        self._add_metadata_to_node(kv_idx_arange, shape=[shapes.S], dtype=idx_dtype)
        kv_idx = graph.call_function(
            torch.ops.aten.view.default, args=(kv_idx_arange, [1, 1, 1, shapes.S])
        )
        self._add_metadata_to_node(kv_idx, shape=[1, 1, 1, shapes.S], dtype=idx_dtype)

        return b_idx, h_idx, q_idx, kv_idx

    def _compute_softmax(
        self, graph: fx.Graph, scores: fx.Node, shapes: AttentionShapes, dtype: torch.dtype
    ) -> tuple[fx.Node, fx.Node, fx.Node]:
        """Compute numerically stable softmax over the last dimension."""
        # max_scores = max(scores, dim=-1, keepdim=True)
        max_result = graph.call_function(torch.ops.aten.max.dim, args=(scores, -1, True))
        max_result.meta["val"] = (
            torch.empty([shapes.B, shapes.H_q, shapes.L, 1], dtype=dtype),
            torch.empty([shapes.B, shapes.H_q, shapes.L, 1], dtype=torch.int64),
        )

        max_scores = graph.call_function(operator.getitem, args=(max_result, 0))
        self._add_metadata_to_node(
            max_scores, shape=[shapes.B, shapes.H_q, shapes.L, 1], dtype=dtype
        )

        # scores_centered = scores - max_scores
        scores_centered = graph.call_function(torch.ops.aten.sub.Tensor, args=(scores, max_scores))
        self._add_metadata_to_node(
            scores_centered, shape=[shapes.B, shapes.H_q, shapes.L, shapes.S], dtype=dtype
        )

        # exp_scores = exp(scores_centered)
        exp_scores = graph.call_function(torch.ops.aten.exp.default, args=(scores_centered,))
        self._add_metadata_to_node(
            exp_scores, shape=[shapes.B, shapes.H_q, shapes.L, shapes.S], dtype=dtype
        )

        # denom = sum(exp_scores, dim=-1, keepdim=True)
        denom = graph.call_function(torch.ops.aten.sum.dim_IntList, args=(exp_scores, [-1], True))
        self._add_metadata_to_node(denom, shape=[shapes.B, shapes.H_q, shapes.L, 1], dtype=dtype)

        # attn_weights = exp_scores / denom
        attn_weights = graph.call_function(torch.ops.aten.div.Tensor, args=(exp_scores, denom))
        self._add_metadata_to_node(
            attn_weights, shape=[shapes.B, shapes.H_q, shapes.L, shapes.S], dtype=dtype
        )

        return attn_weights, exp_scores, denom

    def _compute_attention_scores(
        self,
        graph: fx.Graph,
        query: fx.Node,
        key: fx.Node,
        scale: float,
        shapes: AttentionShapes,
        dtype: torch.dtype,
    ) -> fx.Node:
        """Compute scaled attention scores: Q @ K^T * scale."""
        # K^T: [B, H, S, E] -> [B, H, E, S]
        key_t = graph.call_function(torch.ops.aten.transpose.int, args=(key, -2, -1))
        self._add_metadata_to_node(
            key_t, shape=[shapes.B, shapes.H_q, shapes.E, shapes.S], dtype=dtype
        )

        # scores = Q @ K^T
        scores = graph.call_function(torch.ops.aten.matmul.default, args=(query, key_t))
        self._add_metadata_to_node(
            scores, shape=[shapes.B, shapes.H_q, shapes.L, shapes.S], dtype=dtype
        )

        # scaled_scores = scores * scale
        scaled_scores = graph.call_function(torch.ops.aten.mul.Scalar, args=(scores, scale))
        self._add_metadata_to_node(
            scaled_scores, shape=[shapes.B, shapes.H_q, shapes.L, shapes.S], dtype=dtype
        )

        return scaled_scores

    # =========================================================================
    # Score Mod Inlining
    # =========================================================================

    def _fix_score_mod_metadata(
        self,
        new_node: fx.Node,
        original_node: fx.Node,
        scaled_scores: fx.Node,
        shapes: AttentionShapes,
    ):
        """Fix metadata for score_mod operations that need shape correction."""
        comparison_ops = [
            torch.ops.aten.ge.Tensor,
            torch.ops.aten.le.Tensor,
            torch.ops.aten.gt.Tensor,
            torch.ops.aten.lt.Tensor,
            torch.ops.aten.eq.Tensor,
            torch.ops.aten.ne.Tensor,
        ]

        if original_node.target in comparison_ops:
            # Comparison ops broadcast to (1, 1, L, S)
            bool_shape = torch.Size([1, 1, shapes.L, shapes.S])
            new_node.meta["val"] = torch.empty(bool_shape, dtype=torch.bool)
            new_node.meta["tensor_meta"] = TensorMetadata(
                shape=bool_shape,
                dtype=torch.bool,
                requires_grad=False,
                stride=(shapes.L * shapes.S, shapes.L * shapes.S, shapes.S, 1),
                memory_format=torch.contiguous_format,
                is_quantized=False,
                qparams={},
            )
        elif original_node.target == torch.ops.aten.where.self:
            # Where ops output shape matches scores
            if "val" in scaled_scores.meta:
                score_shape = scaled_scores.meta["val"].shape
                score_dtype = scaled_scores.meta["val"].dtype
                new_node.meta["val"] = torch.empty(score_shape, dtype=score_dtype)
                new_node.meta["tensor_meta"] = TensorMetadata(
                    shape=score_shape,
                    dtype=score_dtype,
                    requires_grad=False,
                    stride=tuple(range(len(score_shape) - 1, -1, -1)),
                    memory_format=torch.contiguous_format,
                    is_quantized=False,
                    qparams={},
                )

    def _map_node_args(self, node_args: tuple, value_map: dict[fx.Node, fx.Node]) -> list:
        """Map node arguments through value_map, handling both Node and non-Node args."""
        return [value_map[arg] if isinstance(arg, fx.Node) else arg for arg in node_args]

    def _map_node_kwargs(self, node_kwargs: dict, value_map: dict[fx.Node, fx.Node]) -> dict:
        """Map node kwargs through value_map, handling both Node and non-Node values."""
        return {k: value_map[v] if isinstance(v, fx.Node) else v for k, v in node_kwargs.items()}

    def _create_node_from_op(
        self,
        graph: fx.Graph,
        gm: fx.GraphModule,
        node: fx.Node,
        new_args: tuple,
        new_kwargs: dict,
        module_prefix: str,
    ) -> fx.Node:
        """Create a new node in the graph based on the operation type."""
        if node.op == "get_attr":
            # Copy the attribute from submodule to main module
            attr_name = node.target
            # Get the parent module by traversing the graph module hierarchy
            parent_module = gm
            for part in module_prefix.split(".")[:-1]:
                if part:
                    parent_module = getattr(parent_module, part)

            attr_value = getattr(parent_module, attr_name)
            new_attr_name = f"_inlined_{module_prefix}_{attr_name}".replace(".", "_")
            setattr(gm, new_attr_name, attr_value)
            new_node = graph.get_attr(new_attr_name)

            # Set metadata based on actual tensor value
            if isinstance(attr_value, torch.Tensor):
                new_node.meta = {
                    "val": torch.empty(
                        attr_value.shape, dtype=attr_value.dtype, device=attr_value.device
                    ),
                    "tensor_meta": TensorMetadata(
                        shape=attr_value.shape,
                        dtype=attr_value.dtype,
                        requires_grad=attr_value.requires_grad,
                        stride=attr_value.stride(),
                        memory_format=(
                            torch.contiguous_format
                            if attr_value.is_contiguous()
                            else torch.preserve_format
                        ),
                        is_quantized=attr_value.is_quantized,
                        qparams={},
                    ),
                }
            return new_node

        # Generic handling for call_function, call_method, call_module
        method = getattr(graph, node.op)
        return method(node.target, args=new_args, kwargs=new_kwargs)

    def _inline_score_mod(
        self,
        graph: fx.Graph,
        gm: fx.GraphModule,
        score_mod_name: str,
        score_mod_module: fx.GraphModule,
        scaled_scores: fx.Node,
        index_tensors: tuple[fx.Node, fx.Node, fx.Node, fx.Node],
        shapes: AttentionShapes,
    ) -> fx.Node:
        """Inline score_mod submodule operations into the main graph."""
        logger.debug(f"Inlining score_mod submodule: {score_mod_name}")

        # Build value map from placeholders to actual args
        value_map = {}
        placeholders = [n for n in score_mod_module.graph.nodes if n.op == "placeholder"]
        actual_args = [scaled_scores, *index_tensors]

        for placeholder, actual_value in zip(placeholders, actual_args, strict=False):
            value_map[placeholder] = actual_value

        final_scores = scaled_scores

        # Copy all non-placeholder, non-output nodes from score_mod graph
        for score_mod_node in score_mod_module.graph.nodes:
            if score_mod_node.op == "placeholder":
                continue

            if score_mod_node.op == "output":
                final_scores = value_map[score_mod_node.args[0]]
                break

            # Map args and kwargs through value_map
            new_args = tuple(self._map_node_args(score_mod_node.args, value_map))
            new_kwargs = self._map_node_kwargs(score_mod_node.kwargs, value_map)

            # Create the new node in our graph
            new_node = self._create_node_from_op(
                graph, gm, score_mod_node, new_args, new_kwargs, score_mod_name
            )

            # Copy and fix metadata
            if score_mod_node.meta:
                new_node.meta = score_mod_node.meta.copy()
                self._fix_score_mod_metadata(new_node, score_mod_node, scaled_scores, shapes)

            value_map[score_mod_node] = new_node

        logger.debug("Successfully inlined score_mod operations")
        return final_scores

    def _apply_combined_mask_score_mod(
        self,
        graph: fx.Graph,
        gm: fx.GraphModule,
        mask_mod_module: fx.GraphModule,
        scores: fx.Node,
        index_tensors: tuple[fx.Node, fx.Node, fx.Node, fx.Node],
        shapes: AttentionShapes,
    ) -> fx.Node:
        """Apply mask_mod by combining it with score using torch.where pattern."""
        logger.debug("Applying combined mask_mod with score_mod pattern")

        # Build value map from placeholders to actual args (mask_mod takes b, h, q_idx, kv_idx)
        value_map = {}
        placeholders = [n for n in mask_mod_module.graph.nodes if n.op == "placeholder"]

        for placeholder, actual_value in zip(placeholders, index_tensors, strict=False):
            value_map[placeholder] = actual_value

        mask_result = None

        # Copy all non-placeholder, non-output nodes from mask_mod graph
        for mask_mod_node in mask_mod_module.graph.nodes:
            if mask_mod_node.op == "placeholder":
                continue

            if mask_mod_node.op == "output":
                mask_result = value_map[mask_mod_node.args[0]]
                break

            # Map args and kwargs through value_map
            new_args = tuple(self._map_node_args(mask_mod_node.args, value_map))
            new_kwargs = self._map_node_kwargs(mask_mod_node.kwargs, value_map)

            # Create the new node in our graph
            new_node = self._create_node_from_op(
                graph, gm, mask_mod_node, new_args, new_kwargs, "mask_mod"
            )

            # Copy metadata
            if mask_mod_node.meta:
                new_node.meta = mask_mod_node.meta.copy()

            value_map[mask_mod_node] = new_node

        # Now apply the mask using torch.where pattern: torch.where(mask, score, -inf)
        if mask_result:
            neg_inf = graph.call_function(
                torch.ops.aten.full_like.default,
                args=(scores,),
                kwargs={"fill_value": float("-inf")},
            )
            self._add_metadata_to_node(
                neg_inf,
                shape=[shapes.B, shapes.H_q, shapes.L, shapes.S],
                dtype=scores.meta["val"].dtype,
            )

            masked_scores = graph.call_function(
                torch.ops.aten.where.self, args=(mask_result, scores, neg_inf)
            )
            self._add_metadata_to_node(
                masked_scores,
                shape=[shapes.B, shapes.H_q, shapes.L, shapes.S],
                dtype=scores.meta["val"].dtype,
            )

            logger.debug("Successfully applied combined mask_mod with score_mod pattern")
            return masked_scores

        logger.warning("mask_mod did not produce a result, returning original scores")
        return scores

    # =========================================================================
    # Sort Replacement
    # =========================================================================

    def _replace_sort_stable_with_identity(self, graph: fx.Graph) -> bool:
        """Replace sort.stable with identity (input passthrough + dummy indices)."""
        modified = False

        for node in list(graph.nodes):
            if node.op == "call_function" and node.target == torch.ops.aten.sort.stable:
                logger.debug(f"Replacing sort.stable node: {node.name} with identity")

                input_node = node.args[0]
                dim = node.kwargs.get("dim", node.args[2] if len(node.args) > 2 else -1)

                if "val" not in input_node.meta:
                    logger.warning(f"No metadata for {input_node.name}, skipping sort replacement")
                    continue

                input_val = input_node.meta["val"]
                shape = list(input_val.shape)
                ndim = len(shape)
                actual_dim = dim if dim >= 0 else ndim + dim
                dim_size = shape[actual_dim]

                with graph.inserting_before(node):
                    # Create indices: arange -> view -> expand
                    indices = graph.call_function(
                        torch.ops.aten.arange.default,
                        args=(dim_size,),
                        kwargs={"dtype": torch.int64},
                    )
                    self._add_metadata_to_node(indices, shape=[dim_size], dtype=torch.int64)

                    view_shape = [1] * ndim
                    view_shape[actual_dim] = dim_size
                    indices_viewed = graph.call_function(
                        torch.ops.aten.view.default, args=(indices, view_shape)
                    )
                    self._add_metadata_to_node(indices_viewed, shape=view_shape, dtype=torch.int64)

                    indices_expanded = graph.call_function(
                        torch.ops.aten.expand.default, args=(indices_viewed, shape)
                    )
                    self._add_metadata_to_node(indices_expanded, shape=shape, dtype=torch.int64)

                # Replace getitem users
                for user in list(node.users):
                    if user.op == "call_function" and user.target == operator.getitem:
                        idx = user.args[1]
                        if idx == 0:  # sorted values -> use input (identity)
                            user.replace_all_uses_with(input_node)
                        else:  # indices -> use dummy arange indices
                            user.replace_all_uses_with(indices_expanded)
                        graph.erase_node(user)

                if len(node.users) == 0:
                    graph.erase_node(node)
                    logger.debug(f"  Successfully replaced {node.name}")
                else:
                    logger.warning(f"{node.name} still has users: {[u.name for u in node.users]}")

                modified = True

        if not modified:
            logger.debug("  No sort.stable operations found to replace")

        return modified

    # =========================================================================
    # Forward Pass Legalization
    # =========================================================================

    def _legalize_flex_forward(self, graph: fx.Graph, gm: fx.GraphModule, node: fx.Node) -> bool:
        """Legalize a flex_attention forward node."""
        logger.debug(f"Legalizing flex_attention forward node: {node.name}")

        # Use local variables to avoid state carryover between nodes
        forward_score_mod_name = None
        forward_score_mod_module = None
        forward_mask_mod_module = None

        if len(node.args) < 9:
            logger.warning(f"Unexpected number of args: {len(node.args)}, skipping")
            return False

        # Extract arguments
        query = node.args[0]
        key = node.args[1]
        value = node.args[2]
        score_mod_node = node.args[3]
        block_mask_node = node.args[4]
        scale = node.args[5]

        # Get score_mod info
        if score_mod_node is not None and hasattr(score_mod_node, "target"):
            forward_score_mod_name = score_mod_node.target
            forward_score_mod_module = getattr(gm, forward_score_mod_name, None)
            logger.debug(f"Found score_mod: {forward_score_mod_name}")

        # Get mask_mod from block_mask if present
        if (
            isinstance(block_mask_node, tuple)
            and len(block_mask_node) > 12
            and hasattr(block_mask_node[12], "target")
        ):
            mask_fn_node = block_mask_node[12]
            forward_mask_mod_module = getattr(gm, mask_fn_node.target, None)
            logger.debug(f"Found mask_mod in block_mask: {mask_fn_node.target}")

        # Get shape information
        if "val" not in query.meta or "val" not in key.meta:
            logger.warning("  Missing shape metadata, cannot legalize flex_attention")
            return False

        q_shape = query.meta["val"].shape
        k_shape = key.meta["val"].shape
        shapes = AttentionShapes(
            B=q_shape[0],
            H_q=q_shape[1],
            H_kv=k_shape[1],
            L=q_shape[2],
            S=k_shape[2],
            E=q_shape[3],
        )
        query_dtype = query.meta["val"].dtype

        logger.debug(
            f"Shape: B={shapes.B}, H_q={shapes.H_q}, L={shapes.L}, " f"S={shapes.S}, E={shapes.E}"
        )

        with graph.inserting_before(node):
            # Handle GQA
            key_expanded, value_expanded = self._expand_kv_for_gqa(graph, key, value, shapes)

            # Compute attention scores
            scaled_scores = self._compute_attention_scores(
                graph, query, key_expanded, scale, shapes, query_dtype
            )

            # Apply score_mod and mask_mod if provided
            query_device = query.meta["val"].device
            index_tensors = self._create_index_tensors(graph, shapes, query_device)

            final_scores = scaled_scores

            # Apply score_mod first if provided
            if forward_score_mod_module is not None:
                final_scores = self._inline_score_mod(
                    graph,
                    gm,
                    forward_score_mod_name,
                    forward_score_mod_module,
                    final_scores,
                    index_tensors,
                    shapes,
                )

            # Apply mask_mod by combining it with score_mod pattern
            if forward_mask_mod_module is not None:
                final_scores = self._apply_combined_mask_score_mod(
                    graph,
                    gm,
                    forward_mask_mod_module,
                    final_scores,
                    index_tensors,
                    shapes,
                )

            # Compute softmax
            attn_weights, _, _ = self._compute_softmax(graph, final_scores, shapes, query_dtype)

            # Compute output: attn_weights @ value
            output = graph.call_function(
                torch.ops.aten.matmul.default, args=(attn_weights, value_expanded)
            )
            self._add_metadata_to_node(
                output,
                shape=[shapes.B, shapes.H_q, shapes.L, shapes.E],
                dtype=query_dtype,
            )

        # Store for backward pass only at the end after successful legalization
        # This ensures we only store the most recent forward pass info
        self._forward_score_mod_name = forward_score_mod_name
        self._forward_score_mod_module = forward_score_mod_module
        self._forward_mask_mod_module = forward_mask_mod_module

        # Replace users of the flex_attention node
        self._replace_flex_forward_users(graph, node, output)
        return True

    def _replace_flex_forward_users(self, graph: fx.Graph, node: fx.Node, output: fx.Node):
        """Replace all users of a flex_attention forward node."""
        users_to_replace = list(node.users)

        for user in users_to_replace:
            if user.op == "call_function" and user.target == operator.getitem:
                index = user.args[1]
                if index == 0 or index == 1:  # output
                    user.replace_all_uses_with(output)
                graph.erase_node(user)
            else:
                user.replace_all_uses_with(output)

        graph.erase_node(node)

    # =========================================================================
    # Backward Pass Legalization
    # =========================================================================

    def _legalize_flex_backward(self, graph: fx.Graph, gm: fx.GraphModule, node: fx.Node) -> bool:
        """Legalize a flex_attention backward node."""
        logger.debug(f"Legalizing flex_attention_backward node: {node.name}")

        if len(node.args) < 11:
            logger.warning(f"Unexpected number of args: {len(node.args)}, skipping")
            return False

        # Extract arguments
        query = node.args[0]
        key = node.args[1]
        value = node.args[2]
        grad_out = node.args[5]
        scale = node.args[10]

        # Get shape information
        if "val" not in query.meta or "val" not in key.meta:
            logger.warning("  Missing shape metadata, cannot legalize flex_attention_backward")
            return False

        q_shape = query.meta["val"].shape
        k_shape = key.meta["val"].shape
        shapes = AttentionShapes(
            B=q_shape[0],
            H_q=q_shape[1],
            H_kv=k_shape[1],
            L=q_shape[2],
            S=k_shape[2],
            E=q_shape[3],
        )
        query_dtype = query.meta["val"].dtype

        if self._forward_score_mod_name:
            logger.debug(f"Using score_mod from forward pass: {self._forward_score_mod_name}")

        with graph.inserting_before(node):
            # Handle GQA
            key_expanded, value_expanded = self._expand_kv_for_gqa(graph, key, value, shapes)

            # Recompute attention scores (needed for backward)
            scaled_scores = self._compute_attention_scores(
                graph, query, key_expanded, scale, shapes, query_dtype
            )

            # Apply score_mod and mask_mod if available
            index_tensors = self._create_index_tensors(graph, shapes)

            final_scores = scaled_scores

            # Apply score_mod first if available
            if self._forward_score_mod_module is not None:
                logger.debug(f"Applying score_mod in backward: {self._forward_score_mod_name}")
                final_scores = self._inline_score_mod(
                    graph,
                    gm,
                    self._forward_score_mod_name,
                    self._forward_score_mod_module,
                    final_scores,
                    index_tensors,
                    shapes,
                )

            # Apply mask_mod if available
            if self._forward_mask_mod_module is not None:
                logger.debug("Applying mask_mod in backward pass")
                final_scores = self._apply_combined_mask_score_mod(
                    graph,
                    gm,
                    self._forward_mask_mod_module,
                    final_scores,
                    index_tensors,
                    shapes,
                )

            # Recompute softmax
            attn_weights, _, _ = self._compute_softmax(graph, final_scores, shapes, query_dtype)

            # Backward through attention
            grad_query, grad_key_expanded, grad_value_expanded = self._compute_attention_backward(
                graph,
                query,
                key_expanded,
                value_expanded,
                attn_weights,
                grad_out,
                scale,
                shapes,
                query_dtype,
            )

            # Handle GQA gradient reduction
            grad_key_final, grad_value_final = self._reduce_gqa_gradients(
                graph, grad_key_expanded, grad_value_expanded, shapes, query_dtype
            )

        # Replace users of the backward node
        self._replace_flex_backward_users(graph, node, grad_query, grad_key_final, grad_value_final)

        logger.debug("Successfully legalized flex_attention_backward")
        return True

    def _compute_attention_backward(
        self,
        graph: fx.Graph,
        query: fx.Node,
        key: fx.Node,
        value: fx.Node,
        attn_weights: fx.Node,
        grad_out: fx.Node,
        scale: float,
        shapes: AttentionShapes,
        dtype: torch.dtype,
    ) -> tuple[fx.Node, fx.Node, fx.Node]:
        """Compute gradients for attention backward pass."""
        # grad_value = attn_weights^T @ grad_out
        attn_weights_t = graph.call_function(
            torch.ops.aten.transpose.int, args=(attn_weights, -2, -1)
        )
        self._add_metadata_to_node(
            attn_weights_t, shape=[shapes.B, shapes.H_q, shapes.S, shapes.L], dtype=dtype
        )

        grad_value = graph.call_function(
            torch.ops.aten.matmul.default, args=(attn_weights_t, grad_out)
        )
        self._add_metadata_to_node(
            grad_value, shape=[shapes.B, shapes.H_q, shapes.S, shapes.E], dtype=dtype
        )

        # grad_attn_weights = grad_out @ value^T
        value_t = graph.call_function(torch.ops.aten.transpose.int, args=(value, -2, -1))
        self._add_metadata_to_node(
            value_t, shape=[shapes.B, shapes.H_q, shapes.E, shapes.S], dtype=dtype
        )

        grad_attn_weights = graph.call_function(
            torch.ops.aten.matmul.default, args=(grad_out, value_t)
        )
        self._add_metadata_to_node(
            grad_attn_weights, shape=[shapes.B, shapes.H_q, shapes.L, shapes.S], dtype=dtype
        )

        # Backward through softmax
        grad_attn_times_attn = graph.call_function(
            torch.ops.aten.mul.Tensor, args=(grad_attn_weights, attn_weights)
        )
        self._add_metadata_to_node(
            grad_attn_times_attn,
            shape=[shapes.B, shapes.H_q, shapes.L, shapes.S],
            dtype=dtype,
        )

        sum_grad_attn_times_attn = graph.call_function(
            torch.ops.aten.sum.dim_IntList, args=(grad_attn_times_attn, [-1], True)
        )
        self._add_metadata_to_node(
            sum_grad_attn_times_attn,
            shape=[shapes.B, shapes.H_q, shapes.L, 1],
            dtype=dtype,
        )

        grad_attn_minus_sum = graph.call_function(
            torch.ops.aten.sub.Tensor, args=(grad_attn_weights, sum_grad_attn_times_attn)
        )
        self._add_metadata_to_node(
            grad_attn_minus_sum,
            shape=[shapes.B, shapes.H_q, shapes.L, shapes.S],
            dtype=dtype,
        )

        grad_scores = graph.call_function(
            torch.ops.aten.mul.Tensor, args=(attn_weights, grad_attn_minus_sum)
        )
        self._add_metadata_to_node(
            grad_scores, shape=[shapes.B, shapes.H_q, shapes.L, shapes.S], dtype=dtype
        )

        # Apply scale
        grad_scaled_scores = graph.call_function(
            torch.ops.aten.mul.Scalar, args=(grad_scores, scale)
        )
        self._add_metadata_to_node(
            grad_scaled_scores,
            shape=[shapes.B, shapes.H_q, shapes.L, shapes.S],
            dtype=dtype,
        )

        # grad_query = grad_scores @ K
        grad_query = graph.call_function(
            torch.ops.aten.matmul.default, args=(grad_scaled_scores, key)
        )
        self._add_metadata_to_node(
            grad_query, shape=[shapes.B, shapes.H_q, shapes.L, shapes.E], dtype=dtype
        )

        # grad_key = grad_scores^T @ Q
        grad_scores_t = graph.call_function(
            torch.ops.aten.transpose.int, args=(grad_scaled_scores, -2, -1)
        )
        self._add_metadata_to_node(
            grad_scores_t, shape=[shapes.B, shapes.H_q, shapes.S, shapes.L], dtype=dtype
        )

        grad_key = graph.call_function(torch.ops.aten.matmul.default, args=(grad_scores_t, query))
        self._add_metadata_to_node(
            grad_key, shape=[shapes.B, shapes.H_q, shapes.S, shapes.E], dtype=dtype
        )

        return grad_query, grad_key, grad_value

    def _replace_flex_backward_users(
        self,
        graph: fx.Graph,
        node: fx.Node,
        grad_query: fx.Node,
        grad_key: fx.Node,
        grad_value: fx.Node,
    ):
        """Replace all users of a flex_attention backward node."""
        users_to_replace = list(node.users)

        for user in users_to_replace:
            if user.op == "call_function" and user.target == operator.getitem:
                index = user.args[1]
                if index == 0:
                    user.replace_all_uses_with(grad_query)
                elif index == 1:
                    user.replace_all_uses_with(grad_key)
                elif index == 2:
                    user.replace_all_uses_with(grad_value)
                elif index == 3:
                    # grad_score_mod_buffers - not supported, use None
                    pass
                graph.erase_node(user)

        graph.erase_node(node)

    # =========================================================================
    # Finalization
    # =========================================================================

    def _finalize_legalization(self, gm: fx.GraphModule, example_inputs=None):
        """Finalize the legalization by cleaning up and running shape propagation."""
        graph = gm.graph

        # Replace sort.stable operations
        logger.debug("Replacing sort.stable operations with identity...")
        self._replace_sort_stable_with_identity(graph)

        # Eliminate dead code and recompile
        graph.eliminate_dead_code()
        gm.recompile()

        # Fix inconsistent metadata
        logger.debug("Fixing inconsistent metadata before shape propagation...")
        for node in graph.nodes:
            if node.meta and "val" in node.meta and "tensor_meta" in node.meta:
                val = node.meta["val"]
                tensor_meta = node.meta["tensor_meta"]
                has_mismatched_shapes = (
                    hasattr(val, "shape")
                    and hasattr(tensor_meta, "shape")
                    and val.shape != tensor_meta.shape
                )
                if has_mismatched_shapes:
                    logger.debug(
                        f"Fixing metadata for {node.name}: "
                        f"val.shape={val.shape}, tensor_meta.shape={tensor_meta.shape}"
                    )
                    node.meta["tensor_meta"] = TensorMetadata(
                        shape=val.shape,
                        dtype=val.dtype if hasattr(val, "dtype") else tensor_meta.dtype,
                        requires_grad=tensor_meta.requires_grad,
                        stride=(
                            val.stride()
                            if hasattr(val, "stride")
                            else tuple(range(len(val.shape) - 1, -1, -1))
                        ),
                        memory_format=torch.contiguous_format,
                        is_quantized=False,
                        qparams={},
                    )

        # Run shape propagation
        if example_inputs is not None:
            from torch.fx.passes.shape_prop import ShapeProp

            ShapeProp(gm).propagate(*example_inputs)

    # =========================================================================
    # Pass Entry Point
    # =========================================================================

    def call(self, gm: fx.GraphModule) -> PassResult:
        """Execute the flex attention legalization pass.

        Args:
            gm: The GraphModule to transform.

        Returns:
            PassResult with modified=True if any transformations were made.
        """
        graph = gm.graph
        modified = False

        logger.debug("Scanning graph for flex_attention nodes...")
        flex_nodes = []
        has_backward = False
        has_sort_stable = False

        for node in graph.nodes:
            if self._is_flex_attention_forward(node):
                logger.debug(f"Found flex_attention forward: {node.name}")
                flex_nodes.append(node)
            elif self._is_flex_attention_backward(node):
                logger.debug(f"Found flex_attention backward: {node.name}")
                flex_nodes.append(node)
                has_backward = True

            if node.op == "call_function" and node.target == torch.ops.aten.sort.stable:
                has_sort_stable = True
                logger.debug(f"Found sort.stable node: {node.name}")

        if not flex_nodes:
            logger.debug("  No flex_attention nodes found in graph")
            if has_sort_stable:
                logger.debug("  Found orphaned sort.stable operations, replacing with identity...")
                self._replace_sort_stable_with_identity(graph)
                graph.eliminate_dead_code()
                gm.recompile()
            return PassResult(gm, modified=False)

        if has_backward:
            logger.debug("  Backward pass detected - will legalize both forward and backward")

        for node in list(graph.nodes):
            if self._is_flex_attention_forward(node):
                if self._legalize_flex_forward(graph, gm, node):
                    modified = True

            elif self._is_flex_attention_backward(node) and self._legalize_flex_backward(
                graph, gm, node
            ):
                modified = True

        if modified:
            self._finalize_legalization(gm)

        return PassResult(gm, modified=modified)
