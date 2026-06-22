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
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.shape_prop import TensorMetadata

from torch_neuronx.neuron_dynamo_backend.fx.passes.neuron_fx_pass import (
    NeuronFXPass,
    NeuronFXPassState,
)

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


class FlexAttentionLegalization(NeuronFXPass):
    """Flex Attention Legalization Pass for PyTorch FX Graphs.

    This pass legalizes flex_attention higher-order operators by decomposing
    them into standard ATen operations that torch-mlir can handle.

    Higher-order operators like flex_attention are not legalized through the
    standard decomposition table mechanism in AOTAutograd, so this pass
    handles them explicitly.
    """

    def __init__(self, state: NeuronFXPassState | None = None):
        super().__init__(state)
        self._forward_score_mod_name: str | None = None
        self._forward_score_mod_module: fx.GraphModule | None = None
        self._forward_mask_mod_module: fx.GraphModule | None = None
        self._forward_score_mod_other_buffers: tuple = ()
        self._forward_mask_mod_other_buffers: tuple = ()

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
    ) -> tuple[fx.Node, fx.Node]:
        """Compute numerically stable softmax over the last dimension.

        Computes in the wider of the scores' actual dtype and `dtype` (a score_mod
        may promote, e.g. bf16 -> f32 for alibi), then casts attn_weights back to
        `dtype` so the caller gets weights matching the original Q/K/V dtype.

        Returns:
            (attn_weights, logsumexp) where attn_weights has the caller's `dtype`
            and logsumexp is always float32 with shape [B, H, L].
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

        # attn_weights = exp_scores / denom
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

        # Squeeze to [B, H, L]
        logsumexp = graph.call_function(torch.ops.aten.squeeze.dim, args=(lse_keepdim, -1))
        self._add_metadata_to_node(
            logsumexp, shape=[shapes.B, shapes.H_q, shapes.L], dtype=torch.float32
        )

        return attn_weights, logsumexp

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
                result = value_map[sub_node.args[0]]
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

    def _apply_combined_mask_score_mod(
        self,
        graph: fx.Graph,
        gm: fx.GraphModule,
        mask_mod_module: fx.GraphModule,
        scores: fx.Node,
        index_tensors: tuple[fx.Node, fx.Node, fx.Node, fx.Node],
        shapes: AttentionShapes,
        other_buffers: tuple = (),
    ) -> fx.Node:
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

            return masked_scores

        logger.warning("mask_mod produced no result, returning original scores")
        return scores

    # =========================================================================
    # Sort Replacement
    # =========================================================================

    def _replace_sort_stable_with_identity(self, graph: fx.Graph) -> bool:
        """Replace sort.stable with identity (input passthrough + dummy indices)."""
        modified = False

        for node in list(graph.nodes):
            if node.op == "call_function" and node.target == torch.ops.aten.sort.stable:
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
                else:
                    logger.warning(f"sort.stable {node.name} still has users after replacement")

                modified = True

        if not modified:
            logger.debug("No sort.stable operations found")

        return modified

    # =========================================================================
    # Forward Pass Legalization
    # =========================================================================

    def _legalize_flex_forward(self, graph: fx.Graph, gm: fx.GraphModule, node: fx.Node) -> bool:
        """Legalize a flex_attention forward node."""
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

        # external buffers (free variable buffers)
        score_mod_other_buffers = node.args[7] if len(node.args) > 7 else ()
        mask_mod_other_buffers = node.args[8] if len(node.args) > 8 else ()

        # Get score_mod info
        if score_mod_node is not None and hasattr(score_mod_node, "target"):
            forward_score_mod_name = score_mod_node.target
            forward_score_mod_module = getattr(gm, forward_score_mod_name, None)

        # Get mask_mod from block_mask if present
        if (
            isinstance(block_mask_node, tuple)
            and len(block_mask_node) > 12
            and hasattr(block_mask_node[12], "target")
        ):
            mask_fn_node = block_mask_node[12]
            forward_mask_mod_module = getattr(gm, mask_fn_node.target, None)

        # Get shape information
        if "val" not in query.meta or "val" not in key.meta:
            logger.warning("Missing shape metadata, cannot legalize flex_attention")
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
            f"Legalizing flex_attention: B={shapes.B}, H_q={shapes.H_q}, "
            f"H_kv={shapes.H_kv}, L={shapes.L}, S={shapes.S}, E={shapes.E}"
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
                    other_buffers=score_mod_other_buffers,
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
                    other_buffers=mask_mod_other_buffers,
                )

            # Compute softmax
            attn_weights, logsumexp = self._compute_softmax(
                graph, final_scores, shapes, query_dtype
            )

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
        self._forward_score_mod_other_buffers = score_mod_other_buffers
        self._forward_mask_mod_other_buffers = mask_mod_other_buffers

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
        scale = node.args[10]

        # Get shape information
        if "val" not in query.meta or "val" not in key.meta:
            logger.warning("Missing shape metadata, cannot legalize flex_attention_backward")
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
            if self._forward_score_mod_module is not None:
                final_scores = self._inline_score_mod(
                    graph,
                    gm,
                    self._forward_score_mod_name,
                    self._forward_score_mod_module,
                    final_scores,
                    index_tensors,
                    shapes,
                    other_buffers=self._forward_score_mod_other_buffers,
                )

            # Apply mask_mod if available
            if self._forward_mask_mod_module is not None:
                final_scores = self._apply_combined_mask_score_mod(
                    graph,
                    gm,
                    self._forward_mask_mod_module,
                    final_scores,
                    index_tensors,
                    shapes,
                    other_buffers=self._forward_mask_mod_other_buffers,
                )

            # Recompute softmax
            attn_weights, _ = self._compute_softmax(graph, final_scores, shapes, query_dtype)

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

        self._replace_sort_stable_with_identity(graph)

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
        """Execute the flex attention legalization pass.

        Args:
            gm: The GraphModule to transform.

        Returns:
            PassResult with modified=True if any transformations were made.
        """
        graph = gm.graph
        modified = False

        has_flex = False
        has_sort_stable = False

        for node in graph.nodes:
            if self._is_flex_attention_forward(node) or self._is_flex_attention_backward(node):
                has_flex = True
            if node.op == "call_function" and node.target == torch.ops.aten.sort.stable:
                has_sort_stable = True

        if not has_flex:
            if has_sort_stable:
                self._replace_sort_stable_with_identity(graph)
                graph.eliminate_dead_code()
                gm.recompile()
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
