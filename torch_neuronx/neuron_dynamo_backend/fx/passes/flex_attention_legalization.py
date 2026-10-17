"""
Flex Attention Legalization Pass for PyTorch FX Graphs.

Legalizes flex_attention higher-order operators by decomposing them into
standard ATen operations that torch-mlir can handle.
"""

import logging
import math
import operator
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.fx as fx
from torch.fx import Node
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.shape_prop import TensorMetadata

from torch_neuronx.neuron_dynamo_backend.fx.passes.neuron_fx_pass import NeuronFXPass

logger = logging.getLogger(__name__)

# The flex_attention HOP returns the log-sum-exp in log2 scale, and takes its gradient in
# that same scale.
_LOG2_E = 1.0 / math.log(2.0)


@dataclass
class AttentionShapes:
    """Store dimensions used by flex-attention graph rewrites.

    Args:
        B (int): Batch size.
        H_q (int): Number of query heads.
        H_kv (int): Number of key/value heads.
        L (int): Query sequence length.
        S (int): Key/value sequence length.
        E_qk (int): Query/key head dimension.
        E_v (int): Value head dimension.
    """

    B: int  # batch size
    H_q: int  # query heads
    H_kv: int  # key/value heads (may differ from H_q in GQA)
    L: int  # query sequence length
    S: int  # key/value sequence length
    E_qk: int  # query/key head dimension (d_head_qk)
    E_v: int  # value head dimension (d_head_v), which can be different from E_qk


def decomposition_unsupported_reason(node: Node, backward: bool, has_score_mod: bool) -> str | None:
    """Return why a flex-attention node cannot be decomposed.

    Args:
        node (Node): Flex-attention node to inspect.
        backward (bool): Whether ``node`` is a backward operator.
        has_score_mod (bool): Whether the operator has a score modification.

    Returns:
        str | None: Unsupported reason, or ``None`` when decomposition is valid.

    Note:
        The NKI lowering uses this check before choosing the decomposition path.
    """
    min_args = 11 if backward else 9
    if len(node.args) < min_args:
        return f"the node has {len(node.args)} arguments, and the minimum is {min_args}"
    query, key, value = node.args[:3]
    if any("val" not in arg.meta for arg in (query, key, value)):
        return "the query, the key or the value has no shape metadata"
    q_shape = query.meta["val"].shape
    k_shape = key.meta["val"].shape
    if q_shape[0] != k_shape[0]:
        return (
            f"the batch sizes are not equal: query={q_shape[0]}, key={k_shape[0]}. The "
            "decomposition gives the key and the value gradients the query batch size."
        )
    if k_shape[1] == 0 or q_shape[1] % k_shape[1] != 0:
        return f"the key head count {k_shape[1]} does not divide the query head count {q_shape[1]}"
    if backward and has_score_mod and node.args[8] is None:
        return "the backward operator has a score modification and no joint graph"
    return None


class FlexAttentionLegalization(NeuronFXPass):
    """Decompose flex-attention higher-order operators into ATen operations.

    Args:
        state (NeuronFXPassState | None, optional): Shared state used for pass artifact
            dumping. Defaults to ``None``.

    Note:
        AOTAutograd does not legalize these higher-order operators through its
        standard decomposition table, so this pass handles them explicitly.
    """

    @property
    def name(self) -> str:
        """Return the pass name.

        Returns:
            str: The ``flex_attention_legalization`` identifier.
        """
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
            # Also set tensor_meta for torch-mlir
            if hasattr(example_tensor, "shape"):
                node.meta["tensor_meta"] = TensorMetadata(
                    shape=example_tensor.shape,
                    dtype=example_tensor.dtype,
                    requires_grad=False,
                    stride=example_tensor.stride()
                    if hasattr(example_tensor, "stride")
                    else tuple(range(len(example_tensor.shape) - 1, -1, -1)),
                    memory_format=torch.contiguous_format,
                    is_quantized=False,
                    qparams={},
                )
        elif shape is not None and dtype is not None:
            node.meta["val"] = torch.empty(shape, dtype=dtype)
            # Also set tensor_meta for torch-mlir
            node.meta["tensor_meta"] = TensorMetadata(
                shape=torch.Size(shape) if not isinstance(shape, torch.Size) else shape,
                dtype=dtype,
                requires_grad=False,
                stride=tuple(range(len(shape) - 1, -1, -1)),
                memory_format=torch.contiguous_format,
                is_quantized=False,
                qparams={},
            )

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

    @staticmethod
    def _resolve_subgraph(gm: fx.GraphModule, node) -> tuple[str | None, fx.GraphModule | None]:
        """Resolve a subgraph argument (a get_attr node) to its name and module on ``gm``."""
        if node is None or not hasattr(node, "target"):
            return None, None
        return node.target, getattr(gm, node.target, None)

    @classmethod
    def _resolve_mask_mod(
        cls, gm: fx.GraphModule, block_mask
    ) -> tuple[str | None, fx.GraphModule | None]:
        """Resolve the mask_mod subgraph carried by a block_mask tuple argument.

        mask_mod is always the last element: index 12 in PT2.12, index 16 in PT2.13.
        """
        if not isinstance(block_mask, tuple) or len(block_mask) < 13:
            return None, None
        return cls._resolve_subgraph(gm, block_mask[-1])

    # =========================================================================
    # GQA (Grouped Query Attention) Helpers
    # =========================================================================

    def _expand_heads(
        self, graph: fx.Graph, tensor: fx.Node, head_dim: int, shapes: AttentionShapes
    ) -> fx.Node:
        """Expand the heads of one key or value tensor to the query head count.

        `head_dim` is the last dimension of this tensor. The key uses E_qk and the value
        uses E_v, and the two can be different.
        """
        repeat_factor = shapes.H_q // shapes.H_kv
        dtype = tensor.meta["val"].dtype

        unsqueezed = graph.call_function(torch.ops.aten.unsqueeze.default, args=(tensor, 2))
        self._add_metadata_to_node(
            unsqueezed, shape=[shapes.B, shapes.H_kv, 1, shapes.S, head_dim], dtype=dtype
        )

        expand_shape = [shapes.B, shapes.H_kv, repeat_factor, shapes.S, head_dim]
        expanded = graph.call_function(
            torch.ops.aten.expand.default, args=(unsqueezed, expand_shape)
        )
        self._add_metadata_to_node(expanded, shape=expand_shape, dtype=dtype)

        reshape_shape = [shapes.B, shapes.H_q, shapes.S, head_dim]
        final = graph.call_function(torch.ops.aten.reshape.default, args=(expanded, reshape_shape))
        self._add_metadata_to_node(final, shape=reshape_shape, dtype=dtype)

        return final

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

        return (
            self._expand_heads(graph, key, shapes.E_qk, shapes),
            self._expand_heads(graph, value, shapes.E_v, shapes),
        )

    def _reduce_heads(
        self,
        graph: fx.Graph,
        grad: fx.Node,
        head_dim: int,
        shapes: AttentionShapes,
        dtype: torch.dtype,
    ) -> fx.Node:
        """Add the gradients of the heads that GQA expanded, for one key or value."""
        repeat_factor = shapes.H_q // shapes.H_kv
        grouped_shape = [shapes.B, shapes.H_kv, repeat_factor, shapes.S, head_dim]

        reshaped = graph.call_function(torch.ops.aten.reshape.default, args=(grad, grouped_shape))
        self._add_metadata_to_node(reshaped, shape=grouped_shape, dtype=dtype)

        summed = graph.call_function(torch.ops.aten.sum.dim_IntList, args=(reshaped, [2], False))
        self._add_metadata_to_node(
            summed, shape=[shapes.B, shapes.H_kv, shapes.S, head_dim], dtype=dtype
        )

        return summed

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

        return (
            self._reduce_heads(graph, grad_key, shapes.E_qk, shapes, dtype),
            self._reduce_heads(graph, grad_value, shapes.E_v, shapes, dtype),
        )

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
        self,
        graph: fx.Graph,
        scores: fx.Node,
        shapes: AttentionShapes,
        dtype: torch.dtype,
        defer_normalization: bool = False,
    ) -> tuple[fx.Node, fx.Node, fx.Node | None]:
        """Compute numerically stable softmax over the last dimension.

        Computes in the wider of the scores' actual dtype and `dtype` (a score_mod
        may promote, e.g. bf16 -> f32 for alibi), then casts attn_weights back to
        `dtype` so the caller gets weights matching the original Q/K/V dtype.

        With `defer_normalization` the caller divides the [B, H_q, L, E_v] output instead.
        The denominator is constant along the contracted axis, so the value is the same and
        no score-shaped quotient is built.

        Returns:
            (weights, logsumexp, denom) where weights has the caller's `dtype` and is
            unnormalized when normalization was deferred, logsumexp is always float32 with
            shape [B, H, L] in log2 scale, and denom is the deferred division, else None.
        """
        from torch._subclasses.fake_tensor import FakeTensorMode

        # Compute in the wider dtype so a promoting score_mod doesn't lose precision
        # and the metadata matches the actual tensors.
        scores_dtype = scores.meta["val"].dtype if "val" in scores.meta else dtype
        compute_dtype = max((scores_dtype, dtype), key=lambda d: torch.finfo(d).bits)

        # max_scores = max(scores, dim=-1, keepdim=True)
        max_result = graph.call_function(torch.ops.aten.max.dim, args=(scores, -1, True))

        # max.dim returns a tuple (values, indices), use FakeTensors for torch-mlir compatibility
        # torch-mlir unpacks tuples and needs each element to be a FakeTensor
        max_result_shape = [shapes.B, shapes.H_q, shapes.L, 1]
        with FakeTensorMode():
            fake_values = torch.empty(max_result_shape, dtype=compute_dtype)
            fake_indices = torch.empty(max_result_shape, dtype=torch.int64)

        max_result_stride = fake_values.stride()
        max_result.meta["val"] = (fake_values, fake_indices)
        # For tuple returns, tensor_meta should also be a tuple
        max_result.meta["tensor_meta"] = (
            TensorMetadata(
                shape=torch.Size(max_result_shape),
                dtype=compute_dtype,
                requires_grad=False,
                stride=max_result_stride,
                memory_format=torch.contiguous_format,
                is_quantized=False,
                qparams={},
            ),
            TensorMetadata(
                shape=torch.Size(max_result_shape),
                dtype=torch.int64,
                requires_grad=False,
                stride=max_result_stride,
                memory_format=torch.contiguous_format,
                is_quantized=False,
                qparams={},
            ),
        )

        max_scores = graph.call_function(operator.getitem, args=(max_result, 0))
        self._add_metadata_to_node(
            max_scores, shape=[shapes.B, shapes.H_q, shapes.L, 1], dtype=compute_dtype
        )

        # scores_centered = scores - max_scores
        scores_centered = graph.call_function(torch.ops.aten.sub.Tensor, args=(scores, max_scores))
        self._add_metadata_to_node(
            scores_centered, shape=[shapes.B, shapes.H_q, shapes.L, shapes.S], dtype=compute_dtype
        )

        # exp_scores = exp(scores_centered)
        exp_scores = graph.call_function(torch.ops.aten.exp.default, args=(scores_centered,))
        self._add_metadata_to_node(
            exp_scores, shape=[shapes.B, shapes.H_q, shapes.L, shapes.S], dtype=compute_dtype
        )

        # denom = sum(exp_scores, dim=-1, keepdim=True)
        denom = graph.call_function(torch.ops.aten.sum.dim_IntList, args=(exp_scores, [-1], True))
        self._add_metadata_to_node(
            denom, shape=[shapes.B, shapes.H_q, shapes.L, 1], dtype=compute_dtype
        )

        # A promoting score_mod narrows the quotient afterwards, so deferring past that cast
        # would divide in the narrower dtype. Unnormalized weights also grow the matmul output
        # by up to S, which overflows float16 above finfo.max / S.
        has_exponent_headroom = torch.finfo(dtype).max >= torch.finfo(torch.bfloat16).max
        defer_normalization = (
            defer_normalization and compute_dtype == dtype and has_exponent_headroom
        )
        if defer_normalization:
            attn_weights = exp_scores
        else:
            attn_weights = graph.call_function(torch.ops.aten.div.Tensor, args=(exp_scores, denom))
            self._add_metadata_to_node(
                attn_weights, shape=[shapes.B, shapes.H_q, shapes.L, shapes.S], dtype=compute_dtype
            )

        # Cast attn_weights back to caller dtype if we computed in a wider type
        if compute_dtype != dtype:
            attn_weights = graph.call_function(
                torch.ops.aten._to_copy.default, args=(attn_weights,), kwargs={"dtype": dtype}
            )
            self._add_metadata_to_node(
                attn_weights, shape=[shapes.B, shapes.H_q, shapes.L, shapes.S], dtype=dtype
            )

        # Compute real logsumexp = max + log(sum(exp(scores - max)))
        # log(denom) has shape [B, H, L, 1] (keepdim from sum)
        log_denom = graph.call_function(torch.ops.aten.log.default, args=(denom,))
        self._add_metadata_to_node(
            log_denom, shape=[shapes.B, shapes.H_q, shapes.L, 1], dtype=compute_dtype
        )

        # lse_keepdim = max_scores + log_denom, shape [B, H, L, 1]
        lse_keepdim = graph.call_function(torch.ops.aten.add.Tensor, args=(max_scores, log_denom))
        self._add_metadata_to_node(
            lse_keepdim, shape=[shapes.B, shapes.H_q, shapes.L, 1], dtype=compute_dtype
        )

        # Cast to float32 before squeeze for backward compatibility
        lse_keepdim = graph.call_function(
            torch.ops.aten._to_copy.default, args=(lse_keepdim,), kwargs={"dtype": torch.float32}
        )
        self._add_metadata_to_node(
            lse_keepdim, shape=[shapes.B, shapes.H_q, shapes.L, 1], dtype=torch.float32
        )

        lse_keepdim = graph.call_function(torch.ops.aten.mul.Scalar, args=(lse_keepdim, _LOG2_E))
        self._add_metadata_to_node(
            lse_keepdim, shape=[shapes.B, shapes.H_q, shapes.L, 1], dtype=torch.float32
        )

        # Squeeze to [B, H, L]
        logsumexp = graph.call_function(torch.ops.aten.squeeze.dim, args=(lse_keepdim, -1))
        self._add_metadata_to_node(
            logsumexp, shape=[shapes.B, shapes.H_q, shapes.L], dtype=torch.float32
        )

        return attn_weights, logsumexp, (denom if defer_normalization else None)

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
        # K^T: [B, H, S, E_qk] -> [B, H, E_qk, S]
        key_t = graph.call_function(torch.ops.aten.transpose.int, args=(key, -2, -1))
        self._add_metadata_to_node(
            key_t, shape=[shapes.B, shapes.H_q, shapes.E_qk, shapes.S], dtype=dtype
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
    ):
        """Fix metadata for score_mod operations using generic execution on meta tensors."""
        meta_args = []
        for arg in new_node.args:
            if isinstance(arg, fx.Node) and hasattr(arg, "meta") and "val" in arg.meta:
                meta_args.append(
                    torch.empty(arg.meta["val"].shape, dtype=arg.meta["val"].dtype, device="meta")
                )
            elif isinstance(arg, list | tuple):
                meta_args.append(
                    type(arg)(
                        torch.empty(a.meta["val"].shape, dtype=a.meta["val"].dtype, device="meta")
                        if isinstance(a, fx.Node) and hasattr(a, "meta") and "val" in a.meta
                        else a
                        for a in arg
                    )
                )
            elif isinstance(arg, int | float | bool | str | type(None)):
                meta_args.append(arg)
            else:
                return

        meta_kwargs = {}
        for k, v in original_node.kwargs.items():
            if isinstance(v, fx.Node) and hasattr(v, "meta") and "val" in v.meta:
                meta_kwargs[k] = torch.empty(
                    v.meta["val"].shape, dtype=v.meta["val"].dtype, device="meta"
                )
            else:
                meta_kwargs[k] = v

        if not callable(original_node.target):
            raise TypeError(
                f"Expected callable target for node '{original_node.name}', "
                f"got {type(original_node.target)}: {original_node.target}"
            )

        result = original_node.target(*meta_args, **meta_kwargs)

        if isinstance(result, tuple):
            result = result[0]

        if hasattr(result, "shape") and hasattr(result, "dtype"):
            new_node.meta["val"] = torch.empty(
                result.shape, dtype=result.dtype, device=result.device
            )
            new_node.meta["tensor_meta"] = TensorMetadata(
                shape=torch.Size(result.shape),
                dtype=result.dtype,
                requires_grad=False,
                stride=tuple(range(len(result.shape) - 1, -1, -1)),
                memory_format=torch.contiguous_format,
                is_quantized=False,
                qparams={},
            )
            return

        # Generic fallback: re-execute the op on meta tensors to infer correct shape
        # Handles ops not covered above (e.g. aten.index.Tensor for free variable indexing)
        if original_node.op == "call_function" and callable(original_node.target):
            try:

                def _to_meta(arg):
                    if isinstance(arg, fx.Node) and hasattr(arg, "meta") and "val" in arg.meta:
                        v = arg.meta["val"]
                        return torch.empty(v.shape, dtype=v.dtype, device="meta")
                    elif isinstance(arg, list | tuple):
                        return type(arg)(_to_meta(a) if a is not None else None for a in arg)
                    return arg

                meta_args = tuple(_to_meta(a) for a in new_node.args)
                meta_kwargs = {k: _to_meta(v) for k, v in new_node.kwargs.items()}
                result = original_node.target(*meta_args, **meta_kwargs)
                if isinstance(result, torch.Tensor):
                    new_node.meta["val"] = result
                    new_node.meta["tensor_meta"] = TensorMetadata(
                        shape=result.shape,
                        dtype=result.dtype,
                        requires_grad=False,
                        stride=tuple(range(len(result.shape) - 1, -1, -1)),
                        memory_format=torch.contiguous_format,
                        is_quantized=False,
                        qparams={},
                    )
            except Exception:
                pass

    def _map_node_args(self, node_args: tuple, value_map: dict[fx.Node, fx.Node]) -> list:
        """Map node arguments through value_map, handling both Node and non-Node args."""

        def _map_arg(arg):
            if isinstance(arg, fx.Node):
                return value_map[arg]
            elif isinstance(arg, list | tuple):
                return type(arg)(_map_arg(a) for a in arg)
            return arg

        return [_map_arg(arg) for arg in node_args]

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
        submodule: fx.GraphModule,
    ) -> fx.Node:
        """Create a new node in the graph based on the operation type."""
        if node.op == "get_attr":
            # Copy the attribute from submodule to main module
            attr_name = node.target
            attr_value = getattr(submodule, attr_name)
            new_attr_name = f"_inlined_{module_prefix}_{attr_name}".replace(".", "_")
            if isinstance(attr_value, torch.Tensor):
                gm.register_buffer(new_attr_name, attr_value)
            else:
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

    def _inline_submodule(
        self,
        graph: fx.Graph,
        gm: fx.GraphModule,
        submodule: fx.GraphModule,
        module_prefix: str,
        placeholder_values: list[fx.Node],
    ) -> fx.Node | None:
        """Inline a submodule's operations into the main graph.

        Maps placeholders to the provided values, copies all interior nodes,
        and returns the output node (or None if the submodule has no output).
        """
        value_map = {}
        placeholders = [n for n in submodule.graph.nodes if n.op == "placeholder"]

        for placeholder, actual_value in zip(placeholders, placeholder_values, strict=False):
            value_map[placeholder] = actual_value

        result = None

        for sub_node in submodule.graph.nodes:
            if sub_node.op == "placeholder":
                continue

            if sub_node.op == "output":
                result = self._map_node_args((sub_node.args[0],), value_map)[0]
                break

            new_args = tuple(self._map_node_args(sub_node.args, value_map))
            new_kwargs = self._map_node_kwargs(sub_node.kwargs, value_map)

            new_node = self._create_node_from_op(
                graph, gm, sub_node, new_args, new_kwargs, module_prefix, submodule
            )

            if sub_node.meta and sub_node.op != "get_attr":
                new_node.meta = sub_node.meta.copy()
                self._fix_score_mod_metadata(new_node, sub_node)

            value_map[sub_node] = new_node

        return result

    def _inline_score_mod(
        self,
        graph: fx.Graph,
        gm: fx.GraphModule,
        score_mod_name: str,
        score_mod_module: fx.GraphModule,
        scaled_scores: fx.Node,
        index_tensors: tuple[fx.Node, fx.Node, fx.Node, fx.Node],
        shapes: AttentionShapes,
        other_buffers: tuple = (),
    ) -> fx.Node:
        """Inline score_mod submodule operations into the main graph."""
        result = self._inline_submodule(
            graph,
            gm,
            score_mod_module,
            score_mod_name,
            [scaled_scores, *index_tensors, *other_buffers],
        )
        return result if result is not None else scaled_scores

    def _apply_joint_graph(
        self,
        graph: fx.Graph,
        gm: fx.GraphModule,
        joint_graph_name: str,
        joint_graph_module: fx.GraphModule,
        scaled_scores: fx.Node,
        grad_post_mod_scores: fx.Node,
        index_tensors: tuple[fx.Node, fx.Node, fx.Node, fx.Node],
        shapes: AttentionShapes,
        dtype: torch.dtype,
        other_buffers: tuple = (),
        mask: fx.Node | None = None,
    ) -> fx.Node:
        """Change the gradient of the modified scores into the gradient of the scaled scores.

        ``joint_graph`` gives the derivative of score_mod. It takes the scaled scores before
        the modification, the index tensors, the gradient of the modified scores and the
        captured buffers. It returns a list. The first item of that list is the gradient of
        the scaled scores. The other items are the gradients of the captured buffers, and
        this pass does not return them.

        ``mask`` is the mask_mod result. The derivative reads the scores of a masked position,
        which the mask never made finite, so a modification such as a square root gives a nan
        there. That position contributes no gradient, so the result is zeroed on it.
        """
        result = self._inline_submodule(
            graph,
            gm,
            joint_graph_module,
            joint_graph_name,
            [scaled_scores, *index_tensors, grad_post_mod_scores, *other_buffers],
        )
        if isinstance(result, list | tuple):
            result = result[0]
        if not isinstance(result, fx.Node):
            logger.warning(
                "joint_graph gave no score gradient. The pass keeps the softmax gradient, "
                "which is correct only if the score modification has a unit derivative."
            )
            return grad_post_mod_scores

        result_dtype = result.meta["val"].dtype if "val" in result.meta else dtype
        if result_dtype != dtype:
            result = graph.call_function(
                torch.ops.aten._to_copy.default, args=(result,), kwargs={"dtype": dtype}
            )
            self._add_metadata_to_node(
                result, shape=[shapes.B, shapes.H_q, shapes.L, shapes.S], dtype=dtype
            )

        if mask is not None:
            zero = graph.call_function(
                torch.ops.aten.full_like.default, args=(result,), kwargs={"fill_value": 0.0}
            )
            self._add_metadata_to_node(
                zero, shape=[shapes.B, shapes.H_q, shapes.L, shapes.S], dtype=dtype
            )
            result = graph.call_function(torch.ops.aten.where.self, args=(mask, result, zero))
            self._add_metadata_to_node(
                result, shape=[shapes.B, shapes.H_q, shapes.L, shapes.S], dtype=dtype
            )
        return result

    def _apply_combined_mask_score_mod(
        self,
        graph: fx.Graph,
        gm: fx.GraphModule,
        mask_mod_module: fx.GraphModule,
        scores: fx.Node,
        index_tensors: tuple[fx.Node, fx.Node, fx.Node, fx.Node],
        shapes: AttentionShapes,
        other_buffers: tuple = (),
    ) -> tuple[fx.Node, fx.Node | None]:
        """Apply mask_mod by combining it with score using torch.where pattern."""
        mask_result = self._inline_submodule(
            graph,
            gm,
            mask_mod_module,
            "mask_mod",
            [*index_tensors, *other_buffers],
        )
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

            return masked_scores, mask_result

        logger.warning("mask_mod produced no result, returning original scores")
        return scores, None

    # =========================================================================
    # Forward Pass Legalization
    # =========================================================================

    def _legalize_flex_forward(self, graph: fx.Graph, gm: fx.GraphModule, node: fx.Node) -> bool:
        """Legalize a flex_attention forward node."""
        if len(node.args) < 9:
            logger.warning(f"Unexpected number of args: {len(node.args)}, skipping")
            return False

        # Extract arguments
        query = node.args[0]
        key = node.args[1]
        value = node.args[2]
        scale = node.args[5]

        score_mod_name, score_mod_module = self._resolve_subgraph(gm, node.args[3])
        _, mask_mod_module = self._resolve_mask_mod(gm, node.args[4])

        # external buffers (free variable buffers)
        score_mod_other_buffers = node.args[7] if len(node.args) > 7 else ()
        mask_mod_other_buffers = node.args[8] if len(node.args) > 8 else ()

        # Get shape information
        if "val" not in query.meta or "val" not in key.meta or "val" not in value.meta:
            logger.warning("Missing shape metadata, cannot legalize flex_attention")
            return False

        q_shape = query.meta["val"].shape
        k_shape = key.meta["val"].shape
        v_shape = value.meta["val"].shape
        shapes = AttentionShapes(
            B=q_shape[0],
            H_q=q_shape[1],
            H_kv=k_shape[1],
            L=q_shape[2],
            S=k_shape[2],
            E_qk=q_shape[3],
            E_v=v_shape[3],
        )
        query_dtype = query.meta["val"].dtype

        logger.debug(
            f"Legalizing flex_attention: B={shapes.B}, H_q={shapes.H_q}, "
            f"H_kv={shapes.H_kv}, L={shapes.L}, S={shapes.S}, "
            f"E_qk={shapes.E_qk}, E_v={shapes.E_v}"
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
            if score_mod_module is not None:
                final_scores = self._inline_score_mod(
                    graph,
                    gm,
                    score_mod_name,
                    score_mod_module,
                    final_scores,
                    index_tensors,
                    shapes,
                    other_buffers=score_mod_other_buffers,
                )

            # Apply mask_mod by combining it with score_mod pattern
            if mask_mod_module is not None:
                final_scores, _ = self._apply_combined_mask_score_mod(
                    graph,
                    gm,
                    mask_mod_module,
                    final_scores,
                    index_tensors,
                    shapes,
                    other_buffers=mask_mod_other_buffers,
                )

            # Compute softmax
            attn_weights, logsumexp, denom = self._compute_softmax(
                graph, final_scores, shapes, query_dtype, defer_normalization=True
            )

            # Compute output: attn_weights @ value
            output = graph.call_function(
                torch.ops.aten.matmul.default, args=(attn_weights, value_expanded)
            )
            self._add_metadata_to_node(
                output,
                shape=[shapes.B, shapes.H_q, shapes.L, shapes.E_v],
                dtype=query_dtype,
            )

            if denom is not None:
                output = graph.call_function(torch.ops.aten.div.Tensor, args=(output, denom))
                self._add_metadata_to_node(
                    output,
                    shape=[shapes.B, shapes.H_q, shapes.L, shapes.E_v],
                    dtype=query_dtype,
                )

        # Replace users of the flex_attention node
        self._replace_flex_forward_users(graph, node, output, logsumexp)
        return True

    def _replace_flex_forward_users(
        self, graph: fx.Graph, node: fx.Node, output: fx.Node, logsumexp: fx.Node
    ):
        """Replace all users of a flex_attention forward node.

        flex_attention HOP returns (out, lse, max_scores):
            index 0 -> attention output
            index 1 -> logsumexp
            index 2 -> max_scores
        """
        users_to_replace = list(node.users)

        for user in users_to_replace:
            if user.op == "call_function" and user.target == operator.getitem:
                index = user.args[1]
                if index == 0:  # attention output
                    user.replace_all_uses_with(output)
                elif index == 1:  # logsumexp
                    user.replace_all_uses_with(logsumexp)
                elif (
                    index == 2 and len(user.users) > 0
                ):  # max_scores — unused, create dummy zeros [B, H, L]
                    lse_shape = list(logsumexp.meta["val"].shape)
                    with graph.inserting_before(user):
                        dummy_max = graph.call_function(
                            torch.ops.aten.zeros.default,
                            args=(lse_shape,),
                            kwargs={
                                "dtype": output.meta["val"].dtype,
                                "device": output.meta["val"].device,
                            },
                        )
                        self._add_metadata_to_node(
                            dummy_max, shape=lse_shape, dtype=output.meta["val"].dtype
                        )
                    user.replace_all_uses_with(dummy_max)

                if len(user.users) == 0:
                    graph.erase_node(user)
            else:
                user.replace_all_uses_with(output)

        if len(node.users) == 0:
            graph.erase_node(node)

    # =========================================================================
    # Backward Pass Legalization
    # =========================================================================

    def _legalize_flex_backward(self, graph: fx.Graph, gm: fx.GraphModule, node: fx.Node) -> bool:
        """Legalize a flex_attention backward node."""
        if len(node.args) < 11:
            logger.warning(f"Unexpected number of args: {len(node.args)}, skipping")
            return False

        # Extract arguments
        query = node.args[0]
        key = node.args[1]
        value = node.args[2]
        grad_out = node.args[5]
        grad_logsumexp = node.args[6]
        scale = node.args[10]

        score_mod_name, score_mod_module = self._resolve_subgraph(gm, node.args[7])
        joint_graph_name, joint_graph_module = self._resolve_subgraph(gm, node.args[8])
        _, mask_mod_module = self._resolve_mask_mod(gm, node.args[9])

        score_mod_other_buffers = node.args[12] if len(node.args) > 12 else ()
        mask_mod_other_buffers = node.args[13] if len(node.args) > 13 else ()

        # Get shape information
        if "val" not in query.meta or "val" not in key.meta or "val" not in value.meta:
            logger.warning("Missing shape metadata, cannot legalize flex_attention_backward")
            return False

        q_shape = query.meta["val"].shape
        k_shape = key.meta["val"].shape
        v_shape = value.meta["val"].shape
        shapes = AttentionShapes(
            B=q_shape[0],
            H_q=q_shape[1],
            H_kv=k_shape[1],
            L=q_shape[2],
            S=k_shape[2],
            E_qk=q_shape[3],
            E_v=v_shape[3],
        )
        query_dtype = query.meta["val"].dtype

        with graph.inserting_before(node):
            # Handle GQA
            key_expanded, value_expanded = self._expand_kv_for_gqa(graph, key, value, shapes)

            # Recompute attention scores (needed for backward)
            scaled_scores = self._compute_attention_scores(
                graph, query, key_expanded, scale, shapes, query_dtype
            )

            # Apply score_mod and mask_mod if available
            query_device = query.meta["val"].device if "val" in query.meta else None
            index_tensors = self._create_index_tensors(graph, shapes, query_device)

            final_scores = scaled_scores

            # Apply score_mod first if available
            if score_mod_module is not None:
                final_scores = self._inline_score_mod(
                    graph,
                    gm,
                    score_mod_name,
                    score_mod_module,
                    final_scores,
                    index_tensors,
                    shapes,
                    other_buffers=score_mod_other_buffers,
                )

            # Apply mask_mod if available
            mask = None
            if mask_mod_module is not None:
                final_scores, mask = self._apply_combined_mask_score_mod(
                    graph,
                    gm,
                    mask_mod_module,
                    final_scores,
                    index_tensors,
                    shapes,
                    other_buffers=mask_mod_other_buffers,
                )

            # Recompute softmax
            attn_weights, _, _ = self._compute_softmax(graph, final_scores, shapes, query_dtype)

            apply_score_mod_grad = None
            if joint_graph_module is not None:

                def apply_score_mod_grad(grad_post_mod_scores: fx.Node) -> fx.Node:
                    """Apply the joint score-modification graph to a score gradient.

                    Args:
                        grad_post_mod_scores (fx.Node): Gradient after the score modification.

                    Returns:
                        fx.Node: Score gradient returned by the joint graph, or the unchanged
                            post-modification gradient when no score-gradient node is produced.
                    """
                    return self._apply_joint_graph(
                        graph,
                        gm,
                        joint_graph_name,
                        joint_graph_module,
                        scaled_scores,
                        grad_post_mod_scores,
                        index_tensors,
                        shapes,
                        query_dtype,
                        other_buffers=score_mod_other_buffers,
                        mask=mask,
                    )

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
                grad_logsumexp=grad_logsumexp,
                apply_score_mod_grad=apply_score_mod_grad,
            )

            # Handle GQA gradient reduction
            grad_key_final, grad_value_final = self._reduce_gqa_gradients(
                graph, grad_key_expanded, grad_value_expanded, shapes, query_dtype
            )

        # Replace users of the backward node
        self._replace_flex_backward_users(graph, node, grad_query, grad_key_final, grad_value_final)
        return True

    def _subtract_grad_logsumexp_term(
        self,
        graph: fx.Graph,
        row: fx.Node,
        grad_logsumexp: fx.Node,
        shapes: AttentionShapes,
        dtype: torch.dtype,
    ) -> fx.Node:
        """Fold the log-sum-exp gradient into the row term of the softmax backward.

        The bracket subtracts the row term, so the term enters with a minus sign. This keeps
        the whole conversion on a [B, H, L, 1] tensor instead of the score matrix.
        """
        lse_dtype = (
            grad_logsumexp.meta["val"].dtype if "val" in grad_logsumexp.meta else torch.float32
        )

        term = graph.call_function(torch.ops.aten.mul.Scalar, args=(grad_logsumexp, _LOG2_E))
        self._add_metadata_to_node(term, shape=[shapes.B, shapes.H_q, shapes.L], dtype=lse_dtype)

        term = graph.call_function(torch.ops.aten.unsqueeze.default, args=(term, -1))
        self._add_metadata_to_node(term, shape=[shapes.B, shapes.H_q, shapes.L, 1], dtype=lse_dtype)

        if lse_dtype != dtype:
            term = graph.call_function(
                torch.ops.aten._to_copy.default, args=(term,), kwargs={"dtype": dtype}
            )
            self._add_metadata_to_node(term, shape=[shapes.B, shapes.H_q, shapes.L, 1], dtype=dtype)

        row = graph.call_function(torch.ops.aten.sub.Tensor, args=(row, term))
        self._add_metadata_to_node(row, shape=[shapes.B, shapes.H_q, shapes.L, 1], dtype=dtype)
        return row

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
        grad_logsumexp: fx.Node | None = None,
        apply_score_mod_grad: Callable[[fx.Node], fx.Node] | None = None,
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
            grad_value, shape=[shapes.B, shapes.H_q, shapes.S, shapes.E_v], dtype=dtype
        )

        # grad_attn_weights = grad_out @ value^T
        value_t = graph.call_function(torch.ops.aten.transpose.int, args=(value, -2, -1))
        self._add_metadata_to_node(
            value_t, shape=[shapes.B, shapes.H_q, shapes.E_v, shapes.S], dtype=dtype
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

        row_term = graph.call_function(
            torch.ops.aten.sum.dim_IntList, args=(grad_attn_times_attn, [-1], True)
        )
        self._add_metadata_to_node(
            row_term,
            shape=[shapes.B, shapes.H_q, shapes.L, 1],
            dtype=dtype,
        )
        if isinstance(grad_logsumexp, fx.Node):
            row_term = self._subtract_grad_logsumexp_term(
                graph, row_term, grad_logsumexp, shapes, dtype
            )

        grad_attn_minus_sum = graph.call_function(
            torch.ops.aten.sub.Tensor, args=(grad_attn_weights, row_term)
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

        # The score modification is between the scaled scores and the softmax. Thus its
        # derivative applies after the softmax gradient and before the scale.
        if apply_score_mod_grad is not None:
            grad_scores = apply_score_mod_grad(grad_scores)

        # A scalar commutes with the matmul, so scaling each [B, H_q, *, E_qk] output instead
        # of grad_scores reaches the same gradients without a second score-shaped tensor.
        grad_query_unscaled = graph.call_function(
            torch.ops.aten.matmul.default, args=(grad_scores, key)
        )
        self._add_metadata_to_node(
            grad_query_unscaled, shape=[shapes.B, shapes.H_q, shapes.L, shapes.E_qk], dtype=dtype
        )
        grad_query = graph.call_function(
            torch.ops.aten.mul.Scalar, args=(grad_query_unscaled, scale)
        )
        self._add_metadata_to_node(
            grad_query, shape=[shapes.B, shapes.H_q, shapes.L, shapes.E_qk], dtype=dtype
        )

        grad_scores_t = graph.call_function(
            torch.ops.aten.transpose.int, args=(grad_scores, -2, -1)
        )
        self._add_metadata_to_node(
            grad_scores_t, shape=[shapes.B, shapes.H_q, shapes.S, shapes.L], dtype=dtype
        )

        grad_key_unscaled = graph.call_function(
            torch.ops.aten.matmul.default, args=(grad_scores_t, query)
        )
        self._add_metadata_to_node(
            grad_key_unscaled, shape=[shapes.B, shapes.H_q, shapes.S, shapes.E_qk], dtype=dtype
        )
        grad_key = graph.call_function(torch.ops.aten.mul.Scalar, args=(grad_key_unscaled, scale))
        self._add_metadata_to_node(
            grad_key, shape=[shapes.B, shapes.H_q, shapes.S, shapes.E_qk], dtype=dtype
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

                if len(user.users) == 0:
                    graph.erase_node(user)

        if len(node.users) == 0:
            graph.erase_node(node)

    # =========================================================================
    # Finalization
    # =========================================================================

    def _finalize_legalization(self, gm: fx.GraphModule, example_inputs=None):
        """Finalize the legalization by cleaning up and running shape propagation."""
        graph = gm.graph

        # Remove aten.detach nodes (identity ops from autograd)
        for node in list(graph.nodes):
            if node.op == "call_function" and node.target == torch.ops.aten.detach.default:
                node.replace_all_uses_with(node.args[0])
                graph.erase_node(node)

        graph.eliminate_dead_code()
        gm.recompile()

        # Fix inconsistent metadata between val and tensor_meta
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
        """Legalize flex-attention nodes in an FX graph.

        Args:
            gm: Graph module to transform in place.

        Returns:
            PassResult: The graph module and whether any flex-attention node changed.

        Raises:
            ValueError: If grouped-query attention head counts are incompatible.
            TypeError: If an inlined score-modifier node has a non-callable target.
        """
        graph = gm.graph
        modified = False

        has_flex = False

        for node in graph.nodes:
            if self._is_flex_attention_forward(node) or self._is_flex_attention_backward(node):
                has_flex = True

        if not has_flex:
            return PassResult(gm, modified=False)

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
