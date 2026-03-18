"""
Flex Attention Legalization Pass for PyTorch FX Graphs.

Legalizes flex_attention higher-order operators by decomposing them into
standard ATen operations that torch-mlir can handle.
"""

import functools
import logging
import operator
import os

import nki
import torch
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.fx.passes.shape_prop import TensorMetadata

from torch_neuronx.neuron_dynamo_backend.flex_attention.kernel_rendering import (
    generate_body_template,
    generate_buffer_allocator,
    generate_buffer_initializer,
    generate_graph_lowering,
    jinja_env,
    save_and_import_kernel,
)
from torch_neuronx.nki_hop import kernel_registry, nki_kernel_wrapper
from torch_neuronx.utils import map_external_dtype_to_torch

logger = logging.getLogger(__name__)


def add_metadata_to_node(node: fx.Node, example_tensor=None, shape=None, dtype=None) -> fx.Node:
    """Helper to add metadata to a node for torch-mlir compatibility."""
    if not hasattr(node, "meta"):
        node.meta = {}

    if example_tensor is not None:
        node.meta["val"] = example_tensor
    elif shape is not None and dtype is not None:
        with FakeTensorMode():
            node.meta["val"] = torch.empty(shape, dtype=dtype)

    return node


def get_fwd_dumped_config(fwd_kernel, q, k, v, scale):
    kernel_idx = fwd_kernel.kernel_idx
    kernel = kernel_registry.get_kernel(kernel_idx)
    q_meta = torch.empty(q.meta["val"].shape, dtype=q.meta["val"].dtype, device="meta")
    k_meta = torch.empty(k.meta["val"].shape, dtype=k.meta["val"].dtype, device="meta")
    v_meta = torch.empty(v.meta["val"].shape, dtype=v.meta["val"].dtype, device="meta")

    dconfig = kernel.dump_config(q_meta, k_meta, v_meta, scale)
    logger.debug(f"FWD Kernel return types: {dconfig.return_types}")
    return str(dconfig.dumped_config), dconfig.operand_output_aliases, dconfig.return_types


def get_bwd_dumped_config(bwd_kernel, q, k, v, o, dy, lse, scale):
    kernel_idx = bwd_kernel.kernel_idx
    kernel = kernel_registry.get_kernel(kernel_idx)
    q_meta = torch.empty(q.meta["val"].shape, dtype=q.meta["val"].dtype, device="meta")
    k_meta = torch.empty(k.meta["val"].shape, dtype=k.meta["val"].dtype, device="meta")
    v_meta = torch.empty(v.meta["val"].shape, dtype=v.meta["val"].dtype, device="meta")
    o_meta = torch.empty(o.meta["val"].shape, dtype=o.meta["val"].dtype, device="meta")
    dy_meta = torch.empty(dy.meta["val"].shape, dtype=dy.meta["val"].dtype, device="meta")
    lse_meta = torch.empty(lse.meta["val"].shape, dtype=lse.meta["val"].dtype, device="meta")

    dconfig = kernel.dump_config(
        q_meta, k_meta, v_meta, o_meta, dy_meta, lse_meta, softmax_scale=scale
    )
    logger.debug(f"BWD Kernel return types: {dconfig.return_types}")
    return str(dconfig.dumped_config), dconfig.operand_output_aliases, dconfig.return_types


class FlexAttentionNKILowering(PassBase):
    """Flex Attention Lowering Pass to a NKI kernel.

    This pass lowers the flex attention HOP to a NKI kernel.
    """

    def __init__(self):
        super().__init__()
        self.tile_shape = [128, 128]

    @property
    def name(self) -> str:
        return "flex_attention_nki_lowering"

    def _render_fwd_kernel_template(
        self, query_dtype, tile_shape, fwd_score_module, fwd_mask_module
    ):
        input_score = [
            ("score", query_dtype),
            ("b", torch.int32),
            ("h", torch.int32),
            ("q_idx", torch.int32),
            ("kv_idx", torch.int32),
        ]
        input_mask = [
            ("b", torch.int32),
            ("h", torch.int32),
            ("q_idx", torch.int32),
            ("kv_idx", torch.int32),
        ]
        score_graph_lowering = generate_graph_lowering(fwd_score_module, input_score, tile_shape)
        mask_graph_lowering = generate_graph_lowering(fwd_mask_module, input_mask, tile_shape)

        def render_score_function(score, b, h, q_idx, kv_idx, prefix):
            body_template = generate_body_template(score_graph_lowering)
            return jinja_env.from_string(body_template).render(
                score=score, b=b, h=h, q_idx=q_idx, kv_idx=kv_idx, prefix=prefix
            )

        score_buffer_initializer = functools.partial(
            generate_buffer_initializer, score_graph_lowering
        )
        score_buffer_allocator = functools.partial(generate_buffer_allocator, score_graph_lowering)

        def render_mask_function(b, h, q_idx, kv_idx, prefix):
            body_template = generate_body_template(mask_graph_lowering)
            return jinja_env.from_string(body_template).render(
                b=b, h=h, q_idx=q_idx, kv_idx=kv_idx, prefix=prefix
            )

        mask_buffer_initializer = functools.partial(
            generate_buffer_initializer, mask_graph_lowering
        )
        mask_buffer_allocator = functools.partial(generate_buffer_allocator, mask_graph_lowering)

        template_path = os.path.join(
            os.path.dirname(__file__), "../../flex_attention/flex_attention.py.jinja"
        )
        with open(template_path) as f:
            template_content = f.read()

        template = jinja_env.from_string(template_content)
        return template.render(
            k_tile_size=self.tile_shape[1],
            render_score_function=render_score_function,
            render_mask_function=render_mask_function,
            score_buffer_allocator=score_buffer_allocator,
            score_buffer_initializer=score_buffer_initializer,
            mask_buffer_allocator=mask_buffer_allocator,
            mask_buffer_initializer=mask_buffer_initializer,
        )

    def _render_bwd_kernel_template(self, query_dtype, fwd_score_module, fwd_mask_module):
        input_score = [
            ("score", query_dtype),
            ("b", torch.int32),
            ("h", torch.int32),
            ("q_idx", torch.int32),
            ("kv_idx", torch.int32),
        ]
        input_mask = [
            ("b", torch.int32),
            ("h", torch.int32),
            ("q_idx", torch.int32),
            ("kv_idx", torch.int32),
        ]
        score_graph_lowering = generate_graph_lowering(
            fwd_score_module, input_score, self.tile_shape
        )
        mask_graph_lowering = generate_graph_lowering(fwd_mask_module, input_mask, self.tile_shape)

        def render_score_function(score, b, h, q_idx, kv_idx, prefix):
            body_template = generate_body_template(score_graph_lowering)
            return jinja_env.from_string(body_template).render(
                score=score, b=b, h=h, q_idx=q_idx, kv_idx=kv_idx, prefix=prefix
            )

        def render_mask_function(b, h, q_idx, kv_idx, prefix):
            body_template = generate_body_template(mask_graph_lowering)
            return jinja_env.from_string(body_template).render(
                b=b, h=h, q_idx=q_idx, kv_idx=kv_idx, prefix=prefix
            )

        score_buffer_allocator = functools.partial(generate_buffer_allocator, score_graph_lowering)
        mask_buffer_allocator = functools.partial(generate_buffer_allocator, mask_graph_lowering)

        template_path = os.path.join(
            os.path.dirname(__file__), "../../flex_attention/flex_attention_bwd.py.jinja"
        )
        with open(template_path) as f:
            template_content = f.read()

        template = jinja_env.from_string(template_content)
        return template.render(
            k_tile_size=self.tile_shape[1],
            render_score_function=render_score_function,
            render_mask_function=render_mask_function,
            score_buffer_allocator=score_buffer_allocator,
            mask_buffer_allocator=mask_buffer_allocator,
        )

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

    def _replace_flex_forward_users(
        self, graph: fx.Graph, node: fx.Node, output: fx.Node, lse: fx.Node
    ):
        """Replace all users of a flex_attention forward node."""
        users_to_replace = list(node.users)

        for user in users_to_replace:
            if user.op == "call_function" and user.target == operator.getitem:
                index = user.args[1]
                if index == 0:
                    user.replace_all_uses_with(output)
                elif index == 1 or index == 2:
                    user.replace_all_uses_with(lse)
                graph.erase_node(user)

    def _legalize_flex_forward(self, graph: fx.Graph, gm: fx.GraphModule, node: fx.Node) -> bool:
        """Legalize a flex_attention forward node."""
        logger.debug(f"Legalizing flex_attention forward node: {node.name}")

        # Extract arguments
        query = node.args[0]
        key = node.args[1]
        value = node.args[2]
        score_mod_node = node.args[3]
        block_mask_node = node.args[4]
        scale = node.args[5]

        (
            _,  # q_length
            _,  # kv_length
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            _,  # q_num_blocks
            _,  # q_indices,
            _,  # full_q_num_blocks,
            _,  # full_q_indices,
            q_block_size,
            kv_block_size,
            mask_fn_node,
        ) = block_mask_node
        if q_block_size != 128 or kv_block_size not in [128, 256, 512]:
            logger.warning(
                f"Unsupported block size: [{q_block_size}, {kv_block_size}]."
                + " Only supported block sizes are [128, 128], [128, 256], [128, 512]."
            )
            return False
        self.tile_shape = [q_block_size, kv_block_size]

        logger.debug(f"Found mask_mod in block_mask: {mask_fn_node}")
        fwd_mask_module = getattr(gm, mask_fn_node.target, None)
        logger.debug(f"{fwd_mask_module.code}")

        query_dtype = query.meta["val"].dtype

        forward_score_mod_name = score_mod_node.target
        fwd_score_module = getattr(gm, forward_score_mod_name, None)
        logger.debug(f"Found score_mod: {forward_score_mod_name}")
        logger.debug(f"{fwd_score_module.code}")

        try:
            rendered_kernel = self._render_fwd_kernel_template(
                query_dtype, self.tile_shape, fwd_score_module, fwd_mask_module
            )
            temp_module = save_and_import_kernel(
                rendered_kernel, "temp_fwd_kernel", dir_name="flex_attention"
            )
            fwd_kernel = nki.jit(mode="torch")(temp_module.attention_cte)
            """
            IO Shapes:
            - q: (batch_size, seqlen_q, d) when tp_q is True
            - k: (batch_size_kv, d, seqlen_kv) when tp_k is False
            - v: (batch_size_kv, seqlen_kv, d)
            - returns output with shape:
                (batch_size, seqlen_q, d) if tp_out is False
            """
            q_shape = query.meta["val"].shape
            q_dtype = query.meta["val"].dtype
            k_shape = key.meta["val"].shape
            k_dtype = key.meta["val"].dtype
            v_dtype = value.meta["val"].dtype
            seq_len, embed_dim = q_shape[-2], q_shape[-1]
            # batch_dim : multiply every other dimensions except the last two
            batch_dim_q = 1
            batch_dim_kv = 1
            for dim in q_shape[:-2]:
                batch_dim_q *= dim
            for dim in k_shape[:-2]:
                batch_dim_kv *= dim
            _, num_heads_kv, kv_seq_len, _ = k_shape
            value_dim = value.meta["val"].shape[-1]

            # Set insertion point before the flex_attention node
            with graph.inserting_before(node):
                q = graph.call_function(
                    torch.ops.aten.view.default,
                    args=(query, [batch_dim_q, seq_len, embed_dim]),
                )
                add_metadata_to_node(q, shape=[batch_dim_q, seq_len, embed_dim], dtype=q_dtype)

                k = graph.call_function(
                    torch.ops.aten.view.default,
                    args=(key, [batch_dim_kv, kv_seq_len, embed_dim]),
                )
                add_metadata_to_node(k, shape=[batch_dim_kv, kv_seq_len, embed_dim], dtype=k_dtype)
                v = graph.call_function(
                    torch.ops.aten.view.default,
                    args=(value, [batch_dim_kv, kv_seq_len, value_dim]),
                )
                add_metadata_to_node(v, shape=[batch_dim_kv, kv_seq_len, value_dim], dtype=v_dtype)
                (backend_config, operand_output_aliases, return_types) = get_fwd_dumped_config(
                    fwd_kernel, q, k, v, scale
                )
                nki_kernel_args = {
                    "kernel_idx": fwd_kernel.kernel_idx,
                    "grid": [],
                    "backend_config": backend_config,
                    "operand_output_aliases": operand_output_aliases,
                    "args": [q, k, v],
                    "arg_names": ("q", "k", "v"),
                    "constant_args_key": -1,
                }
                nki_node = graph.call_function(nki_kernel_wrapper, args=(), kwargs=nki_kernel_args)
                logger.debug(f"Flex attention nki kernel node: {nki_node}")

                # Assign nki_node.meta['val'] using return_types
                with FakeTensorMode():
                    nki_node.meta["val"] = tuple(
                        torch.empty(rtype[1], dtype=map_external_dtype_to_torch(rtype[0]))
                        for rtype in return_types
                    )

                attn_output_raw = graph.call_function(operator.getitem, args=(nki_node, 0))
                lse_raw = graph.call_function(operator.getitem, args=(nki_node, 1))

                add_metadata_to_node(
                    attn_output_raw,
                    shape=return_types[0][1],
                    dtype=v_dtype,
                )
                add_metadata_to_node(
                    lse_raw,
                    shape=return_types[1][1],
                    dtype=torch.float32,
                )

                output_shape = [*list(q_shape[:-1]), value_dim]
                lse_shape = list(q_shape[:-1])  # [b, h, seqlen_q]
                attn_output = graph.call_function(
                    torch.ops.aten.view.default, args=(attn_output_raw, output_shape)
                )
                add_metadata_to_node(
                    attn_output,
                    shape=output_shape,
                    dtype=v_dtype,
                )

                # lse_raw shape: [bs*nheads, tile_size, seqlen_q//tile_size]
                # Transpose to: [bs*nheads, seqlen_q//tile_size, tile_size]
                lse_transposed = graph.call_function(
                    torch.ops.aten.transpose.int, args=(lse_raw, -2, -1)
                )
                lse_raw_shape = return_types[1][1]
                add_metadata_to_node(
                    lse_transposed,
                    shape=[lse_raw_shape[0], lse_raw_shape[2], lse_raw_shape[1]],
                    dtype=torch.float32,
                )
                # Reshape to [b, h, seqlen_q]
                lse = graph.call_function(
                    torch.ops.aten.view.default, args=(lse_transposed, lse_shape)
                )
                add_metadata_to_node(
                    lse,
                    shape=lse_shape,
                    dtype=torch.float32,
                )
            # Store for backward pass is done in _render_fwd_kernel_template

            self._replace_flex_forward_users(graph, node, attn_output, lse)
            # TODO: max_score is currently not available
            tuple_node = graph.call_function(tuple, args=([attn_output, lse, lse],))
            node.replace_all_uses_with(tuple_node)
            graph.erase_node(node)
        except Exception as e:
            logger.warning(
                f"Failed to lower flex_attention forward node to NKI: {e}. "
                "Falling back to Graph-based lowering."
            )
            return False
        return True

    def _legalize_flex_backward(self, graph: fx.Graph, gm: fx.GraphModule, node: fx.Node) -> bool:
        """Legalize a flex_attention backward node."""
        logger.debug(f"Legalizing flex_attention_backward node: {node.name}")
        logger.debug(f"Original graph:\n{gm.code}")
        if len(node.args) < 11:
            logger.warning(f"Unexpected number of args: {len(node.args)}, skipping")
            return False

        # Extract arguments
        query = node.args[0]
        key = node.args[1]
        value = node.args[2]
        out = node.args[3]
        lse = node.args[4]
        grad_out = node.args[5]
        score_mod_node = node.args[7]
        block_mask_node = node.args[9]
        scale = node.args[10]

        q_shape = query.meta["val"].shape  # [bs, nheads, seqlen_q, d_head]
        k_shape = key.meta["val"].shape  # [bs, nheads_kv, seqlen_k, d_head]
        v_shape = value.meta["val"].shape  # [bs, nheads_kv, seqlen_k, d_head]
        query_dtype = query.meta["val"].dtype
        key_dtype = key.meta["val"].dtype
        value_dtype = value.meta["val"].dtype
        out_dtype = out.meta["val"].dtype
        grad_out_dtype = grad_out.meta["val"].dtype

        bs, nheads, seqlen_q, d_head = q_shape
        _, nheads_kv, seqlen_k, _ = k_shape

        (
            _,  # q_length
            _,  # kv_length
            _,  # kv_num_blocks
            _,  # kv_indices
            _,  # full_kv_num_blocks
            _,  # full_kv_indices
            _,  # q_num_blocks
            _,  # q_indices
            _,  # full_q_num_blocks
            _,  # full_q_indices
            q_block_size,
            kv_block_size,
            mask_fn_node,
        ) = block_mask_node
        self.tile_shape = [q_block_size, kv_block_size]
        if q_block_size != 128 or kv_block_size not in [128, 256, 512]:
            logger.warning(
                f"Unsupported block size: [{q_block_size}, {kv_block_size}]."
                + " Only supported block sizes are [128, 128], [128, 256], [128, 512]."
            )
            return False
        fwd_mask_module = getattr(gm, mask_fn_node.target, None)
        fwd_score_module = getattr(gm, score_mod_node.target, None)
        logger.debug(f"BWD: score_mod={fwd_score_module.code}, mask_mod={fwd_mask_module.code}")

        try:
            rendered_kernel = self._render_bwd_kernel_template(
                query_dtype, fwd_score_module, fwd_mask_module
            )
            temp_module = save_and_import_kernel(
                rendered_kernel, "temp_bwd_kernel", dir_name="flex_attention"
            )
            bwd_kernel = nki.jit(mode="torch")(temp_module.attention_bwd)

            with graph.inserting_before(node):
                # Transpose to [bs, nheads, d_head, seqlen]
                q = graph.call_function(torch.ops.aten.transpose.int, args=(query, -2, -1))
                add_metadata_to_node(q, shape=[bs, nheads, d_head, seqlen_q], dtype=query_dtype)

                k = graph.call_function(torch.ops.aten.transpose.int, args=(key, -2, -1))
                add_metadata_to_node(k, shape=[bs, nheads_kv, d_head, seqlen_k], dtype=key_dtype)

                v = graph.call_function(torch.ops.aten.transpose.int, args=(value, -2, -1))
                add_metadata_to_node(v, shape=[bs, nheads_kv, d_head, seqlen_k], dtype=value_dtype)

                o = graph.call_function(torch.ops.aten.transpose.int, args=(out, -2, -1))
                add_metadata_to_node(o, shape=[bs, nheads, d_head, seqlen_q], dtype=out_dtype)

                dy = graph.call_function(torch.ops.aten.transpose.int, args=(grad_out, -2, -1))
                add_metadata_to_node(dy, shape=[bs, nheads, d_head, seqlen_q], dtype=grad_out_dtype)

                # Reshape lse to match lse_ref: [bs, nheads, tile_size, seqlen_q // tile_size]
                tile_size = self.tile_shape[0]

                lse_reshaped = graph.call_function(
                    torch.ops.aten.view.default,
                    args=(lse, [bs, nheads, seqlen_q // tile_size, tile_size]),
                )
                add_metadata_to_node(
                    lse_reshaped,
                    shape=[bs, nheads, seqlen_q // tile_size, tile_size],
                    dtype=torch.float32,
                )

                lse_transposed = graph.call_function(
                    torch.ops.aten.transpose.int, args=(lse_reshaped, -2, -1)
                )
                add_metadata_to_node(
                    lse_transposed,
                    shape=[bs, nheads, tile_size, seqlen_q // tile_size],
                    dtype=torch.float32,
                )
                """
                lse_transposed = graph.call_function(
                    torch.ops.aten.view.default,
                    args=(lse, [bs, nheads, tile_size, seqlen_q // tile_size]),
                )
                add_metadata_to_node(
                    lse_transposed,
                    shape=[bs, nheads, tile_size, seqlen_q // tile_size],
                    dtype=torch.float32,
                )
                """

                (backend_config, operand_output_aliases, return_types) = get_bwd_dumped_config(
                    bwd_kernel, q, k, v, o, dy, lse_transposed, scale
                )

                nki_kernel_args = {
                    "kernel_idx": bwd_kernel.kernel_idx,
                    "grid": [],
                    "backend_config": backend_config,
                    "operand_output_aliases": operand_output_aliases,
                    "args": [q, k, v, o, dy, lse_transposed],
                    "arg_names": ("q", "k", "v", "o", "dy", "lse"),
                    "constant_args_key": -1,
                }
                nki_node = graph.call_function(nki_kernel_wrapper, args=(), kwargs=nki_kernel_args)

                # Set metadata
                with FakeTensorMode():
                    nki_node.meta["val"] = tuple(
                        torch.empty(rtype[1], dtype=map_external_dtype_to_torch(rtype[0]))
                        for rtype in return_types
                    )

                # Extract gradients
                grad_q_raw = graph.call_function(operator.getitem, args=(nki_node, 0))
                grad_k_raw = graph.call_function(operator.getitem, args=(nki_node, 1))
                grad_v_raw = graph.call_function(operator.getitem, args=(nki_node, 2))

                add_metadata_to_node(grad_q_raw, shape=return_types[0][1], dtype=query_dtype)
                add_metadata_to_node(grad_k_raw, shape=return_types[1][1], dtype=key_dtype)
                add_metadata_to_node(grad_v_raw, shape=return_types[2][1], dtype=value_dtype)

                # Transpose gradients back to [bs, nheads, seqlen, d_head]
                grad_q = graph.call_function(
                    torch.ops.aten.transpose.int, args=(grad_q_raw, -2, -1)
                )
                grad_k = graph.call_function(
                    torch.ops.aten.transpose.int, args=(grad_k_raw, -2, -1)
                )
                grad_v = graph.call_function(
                    torch.ops.aten.transpose.int, args=(grad_v_raw, -2, -1)
                )

                add_metadata_to_node(grad_q, shape=list(q_shape), dtype=query_dtype)
                add_metadata_to_node(grad_k, shape=list(k_shape), dtype=key_dtype)
                add_metadata_to_node(grad_v, shape=list(v_shape), dtype=value_dtype)

            # Replace users
            for user in list(node.users):
                if user.op == "call_function" and user.target == operator.getitem:
                    idx = user.args[1]
                    if idx == 0:
                        user.replace_all_uses_with(grad_q)
                    elif idx == 1:
                        user.replace_all_uses_with(grad_k)
                    elif idx == 2:
                        user.replace_all_uses_with(grad_v)
                    graph.erase_node(user)

            graph.erase_node(node)
            logger.debug("Successfully legalized flex_attention_backward")
            return True

        except Exception as e:
            logger.warning(f"Failed to lower flex_attention backward to NKI: {e}")
            return False

    def _finalize_legalization(self, gm: fx.GraphModule, example_inputs=None):
        """Finalize the legalization by cleaning up and running shape propagation."""
        graph = gm.graph

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

        for node in graph.nodes:
            if self._is_flex_attention_forward(node):
                logger.debug(f"Found flex_attention forward: {node.name}")
                if self._legalize_flex_forward(graph, gm, node):
                    modified = True
            elif self._is_flex_attention_backward(node):
                logger.debug(f"Found flex_attention backward: {node.name}")
                if self._legalize_flex_backward(graph, gm, node):
                    modified = True

        if modified:
            self._finalize_legalization(gm)
            logger.debug(f"Flex attention lowered to: {gm.code}")

        return PassResult(gm, modified=modified)
