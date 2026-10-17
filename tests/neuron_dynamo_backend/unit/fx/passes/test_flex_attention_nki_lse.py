"""Unit tests for the log-sum-exp operands at the NKI kernel boundary.

The kernel keeps the log-sum-exp negated and in the natural base, while the HOP uses
log2(sum exp). The pass therefore rescales and relayouts every log-sum-exp operand. These
tests pin that rewrite, and the render flag that adds the gradient operand to the kernel
signature.
"""

import ast
import math
import os

import pytest
import torch
import torch.fx as fx

from torch_neuronx.neuron_dynamo_backend import flex_attention as flex_attention_pkg
from torch_neuronx.neuron_dynamo_backend.flex_attention.kernel_rendering import jinja_env
from torch_neuronx.neuron_dynamo_backend.fx.passes.flex_attention_nki_lowering import (
    _LOG2_E,
    FlexAttentionNKILowering,
    _rescale_lse,
)

_TEMPLATE = os.path.join(
    os.path.dirname(flex_attention_pkg.__file__), "flex_attention_bwd.py.jinja"
)


def _render_bwd_template(has_grad_lse: bool, static_tile_elimination: bool) -> str:
    """Render the backward template with stub subgraph bodies."""

    def stub(*args, **kwargs):
        return "pass"

    with open(_TEMPLATE) as f:
        template = jinja_env.from_string(f.read())
    return template.render(
        k_tile_size=512,
        static_tile_elimination=static_tile_elimination,
        qk_tile_types=[[2]],
        pv_tile_types=[[2]],
        qk_mask_index=[[0]],
        num_unique_masks=0,
        masks_npy_path="",
        masks_int8_npy_path="",
        last_compute_section=0,
        max_large_tiles_needed=0,
        render_score_function=stub,
        render_mask_function=stub,
        render_joint_function=stub,
        score_buffer_allocator=stub,
        mask_buffer_allocator=stub,
        joint_buffer_allocator=stub,
        score_extra_names=[],
        mask_extra_names=[],
        all_extra_names=[],
        has_grad_lse=has_grad_lse,
    )


class TestBuildLseKernelArg:
    """A log-sum-exp operand must reach the kernel scaled and in the kernel layout."""

    @pytest.mark.parametrize("bs,nheads,seqlen_q,tile_size", [(1, 2, 256, 128), (2, 1, 512, 128)])
    def test_layout_and_scale(self, bs, nheads, seqlen_q, tile_size):
        graph = fx.Graph()
        lse_node = graph.placeholder("logsumexp")
        lse_node.meta = {"val": torch.empty(bs, nheads, seqlen_q)}

        scale = 1.0 / math.log(2.0)
        result = FlexAttentionNKILowering._build_lse_kernel_arg(
            graph, lse_node, scale, bs, nheads, seqlen_q, tile_size
        )
        graph.output(result)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        lse = torch.randn(bs, nheads, seqlen_q)
        actual = gm(lse)
        expected = (
            (lse / math.log(2.0))
            .view(bs, nheads, seqlen_q // tile_size, tile_size)
            .transpose(-2, -1)
        )

        assert actual.shape == (bs, nheads, tile_size, seqlen_q // tile_size)
        assert list(result.meta["val"].shape) == [bs, nheads, tile_size, seqlen_q // tile_size]
        torch.testing.assert_close(actual, expected)


class TestLseRescaleBound:
    """A Q row that attends to nothing must not overflow either scale conversion.

    The kernel marks such a row with a value many orders of magnitude past a real
    log-sum-exp, and the forward saves its result for the backward, which turns it into an
    exp bias. An infinite bias makes every gradient nan.
    """

    @staticmethod
    def _run(graph: fx.Graph, result: fx.Node, values: torch.Tensor) -> torch.Tensor:
        graph.output(result)
        return fx.GraphModule(torch.nn.Module(), graph)(values)

    def test_forward_keeps_both_sentinels_finite(self):
        kernel_lse = torch.tensor([-5.0, 12.7, torch.finfo(torch.float32).max, float("-inf")])
        graph = fx.Graph()
        node = graph.placeholder("logsumexp")
        node.meta = {"val": torch.empty(4)}

        hop_lse = self._run(graph, _rescale_lse(graph, node, [4], -_LOG2_E), kernel_lse)

        torch.testing.assert_close(hop_lse[:2], torch.tensor([7.213475, -18.322226]))
        assert torch.isfinite(
            hop_lse[2:]
        ).all(), f"a masked row must stay finite, got {hop_lse[2:]}"

    def test_backward_keeps_the_exp_bias_finite(self):
        hop_lse = torch.full((1, 1, 128), -18.322226)
        hop_lse[0, 0, 64:] = -1.442695e30
        graph = fx.Graph()
        node = graph.placeholder("logsumexp")
        node.meta = {"val": torch.empty(1, 1, 128)}
        result = FlexAttentionNKILowering._build_lse_kernel_arg(
            graph, node, -1.0 / _LOG2_E, 1, 1, 128, 128, bound=True
        )

        kernel_lse = self._run(graph, result, hop_lse).view(128)

        torch.testing.assert_close(kernel_lse[:64], torch.full((64,), 12.699999))
        assert torch.isfinite(kernel_lse[64:]).all(), f"got {kernel_lse[64:]}"


class TestBackwardTemplateGradLse:
    """The kernel takes the gradient operand only when the caller supplies one."""

    @pytest.mark.parametrize("static_tile_elimination", [False, True])
    def test_operand_is_conditional(self, static_tile_elimination):
        with_grad = _render_bwd_template(True, static_tile_elimination)
        without_grad = _render_bwd_template(False, static_tile_elimination)

        param = "    dlse_ref: Optional[nl.ndarray] = None,"
        assert param in with_grad
        assert param not in without_grad
        assert "    dlse_ref = None" in without_grad

        for rendered in (with_grad, without_grad):
            ast.parse(rendered)

    def test_operand_follows_lse_in_the_signature(self):
        """get_bwd_dumped_config maps meta tensors onto parameters by position."""
        rendered = _render_bwd_template(True, False)
        tree = ast.parse(rendered)
        entry = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "attention_bwd"
        )
        names = [arg.arg for arg in entry.args.args]
        assert names[:7] == ["q_ref", "k_ref", "v_ref", "o_ref", "dy_ref", "lse_ref", "dlse_ref"]
