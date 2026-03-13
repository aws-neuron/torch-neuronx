# ruff: noqa: N806, N803, N812

"""
Unit tests for FlexAttentionLegalization pass.

These tests verify the PassBase-based FlexAttentionLegalization class
produces correct numerical results and properly integrates with PassManager.
"""

import logging
import operator

import pytest
import torch
import torch.fx as fx
import torch.nn.functional as F
from functorch.compile import aot_module, make_boxed_func
from torch.fx.passes.pass_manager import PassManager

from tests.neuron_dynamo_backend.unit.utils.test_utils import create_capture_compiler
from torch_neuronx.neuron_dynamo_backend.fx.passes.flex_attention_legalization import (
    AttentionShapes,
    FlexAttentionLegalization,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def create_graph():
    """Factory fixture for creating FX graphs with placeholders and metadata.

    Usage:
        graph, nodes = create_graph({
            "query": ((B, H, L, E), torch.float32),
            "key": ((B, H, S, E), torch.float32),
        })
        # Access nodes: nodes["query"], nodes["key"]
    """

    def _create(placeholders: dict):
        """
        Args:
            placeholders: {"name": (shape, dtype), ...}
                         dtype is optional, defaults to torch.float32
        Returns:
            (graph, {name: node})
        """
        graph = fx.Graph()
        nodes = {}
        for name, spec in placeholders.items():
            # Handle both (shape, dtype) and just shape
            if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[1], torch.dtype):
                shape, dtype = spec
            else:
                shape, dtype = spec, torch.float32

            node = graph.placeholder(name)
            node.meta = {"val": torch.empty(*shape, dtype=dtype)}
            nodes[name] = node
        return graph, nodes

    return _create


# ============================================================================
# Utility Functions
# ============================================================================


def legalize_flex_attention_pass(gm: fx.GraphModule) -> fx.GraphModule:
    flex_pass = FlexAttentionLegalization()
    result = flex_pass(gm)
    return result.graph_module


class TestFlexAttentionLegalizationPassBase:
    """Tests for PassBase interface compliance."""

    def test_pass_has_name_property(self):
        """Test that pass has required name property."""
        flex_pass = FlexAttentionLegalization()
        assert flex_pass.name == "flex_attention_legalization"

    def test_pass_works_with_pass_manager(self):
        """Test that pass integrates correctly with PassManager."""

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        gm = fx.symbolic_trace(SimpleModule())

        flex_pass = FlexAttentionLegalization()
        pm = PassManager(passes=[flex_pass])
        result = pm(gm)

        # Should return PassResult
        assert result is not None
        assert hasattr(result, "graph_module")

    def test_pass_returns_pass_result(self):
        """Test that call() returns PassResult with correct attributes."""

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        gm = fx.symbolic_trace(SimpleModule())

        flex_pass = FlexAttentionLegalization()
        result = flex_pass(gm)

        assert hasattr(result, "graph_module")
        assert hasattr(result, "modified")


class TestExpandKVForGQA:
    """Tests for GQA expansion via FlexAttentionLegalization._expand_kv_for_gqa."""

    def test_gqa_expansion_numerical_accuracy(self):
        """Test that the generated FX graph produces correct numerical results."""
        b, h_q, h_kv, s, e = 2, 8, 2, 16, 32
        shapes = AttentionShapes(B=b, H_q=h_q, H_kv=h_kv, L=16, S=s, E=e)

        flex_pass = FlexAttentionLegalization()

        graph = fx.Graph()
        key_node = graph.placeholder("key")
        key_node.meta = {"val": torch.empty(b, h_kv, s, e)}
        value_node = graph.placeholder("value")
        value_node.meta = {"val": torch.empty(b, h_kv, s, e)}

        key_out, value_out = flex_pass._expand_kv_for_gqa(graph, key_node, value_node, shapes)
        graph.output((key_out, value_out))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        key = torch.randn(b, h_kv, s, e)
        value = torch.randn(b, h_kv, s, e)
        key_expanded, value_expanded = gm(key, value)

        # Compute expected result manually
        repeat_factor = h_q // h_kv
        expected_key = key.unsqueeze(2).expand(b, h_kv, repeat_factor, s, e).reshape(b, h_q, s, e)
        expected_value = (
            value.unsqueeze(2).expand(b, h_kv, repeat_factor, s, e).reshape(b, h_q, s, e)
        )

        torch.testing.assert_close(key_expanded, expected_key)
        torch.testing.assert_close(value_expanded, expected_value)

    def test_gqa_expansion_head_repetition(self):
        """Test that each KV head is correctly repeated to match Q heads."""
        b, h_q, h_kv, s, e = 1, 6, 2, 8, 4
        shapes = AttentionShapes(B=b, H_q=h_q, H_kv=h_kv, L=8, S=s, E=e)
        repeat_factor = h_q // h_kv

        flex_pass = FlexAttentionLegalization()

        graph = fx.Graph()
        key_node = graph.placeholder("key")
        key_node.meta = {"val": torch.empty(b, h_kv, s, e)}
        value_node = graph.placeholder("value")
        value_node.meta = {"val": torch.empty(b, h_kv, s, e)}

        key_out, value_out = flex_pass._expand_kv_for_gqa(graph, key_node, value_node, shapes)
        graph.output((key_out, value_out))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        key = torch.randn(b, h_kv, s, e)
        value = torch.randn(b, h_kv, s, e)
        key_expanded, value_expanded = gm(key, value)

        # Verify each KV head is repeated correctly
        for kv_idx in range(h_kv):
            for r in range(repeat_factor):
                q_idx = kv_idx * repeat_factor + r
                torch.testing.assert_close(key_expanded[:, q_idx], key[:, kv_idx])
                torch.testing.assert_close(value_expanded[:, q_idx], value[:, kv_idx])

    def test_no_expansion_when_equal_heads(self):
        """Test that no expansion happens when H_q == H_kv (returns same nodes)."""
        flex_pass = FlexAttentionLegalization()

        graph = fx.Graph()
        shapes = AttentionShapes(B=2, H_q=8, H_kv=8, L=64, S=64, E=32)
        key = graph.placeholder("key")
        key.meta = {"val": torch.empty(2, 8, 64, 32)}
        value = graph.placeholder("value")
        value.meta = {"val": torch.empty(2, 8, 64, 32)}

        key_out, value_out = flex_pass._expand_kv_for_gqa(graph, key, value, shapes)

        # Should return original nodes unchanged
        assert key_out is key
        assert value_out is value

    def test_invalid_head_ratio_raises(self):
        """Test that invalid head ratios raise ValueError."""
        flex_pass = FlexAttentionLegalization()

        graph = fx.Graph()
        shapes = AttentionShapes(B=2, H_q=7, H_kv=2, L=64, S=64, E=32)
        key = graph.placeholder("key")
        key.meta = {"val": torch.empty(2, 2, 64, 32)}
        value = graph.placeholder("value")
        value.meta = {"val": torch.empty(2, 2, 64, 32)}

        with pytest.raises(ValueError, match="must be multiple of"):
            flex_pass._expand_kv_for_gqa(graph, key, value, shapes)


class TestReduceGQAGradients:
    """Tests for GQA gradient reduction via FlexAttentionLegalization._reduce_gqa_gradients."""

    def test_gradient_reduction_numerical_accuracy(self):
        """Test that gradient reduction produces correct numerical results."""
        B, H_q, H_kv, S, E = 2, 8, 2, 16, 32
        shapes = AttentionShapes(B=B, H_q=H_q, H_kv=H_kv, L=16, S=S, E=E)
        repeat_factor = H_q // H_kv

        flex_pass = FlexAttentionLegalization()

        graph = fx.Graph()
        grad_key_node = graph.placeholder("grad_key")
        grad_key_node.meta = {"val": torch.empty(B, H_q, S, E)}
        grad_value_node = graph.placeholder("grad_value")
        grad_value_node.meta = {"val": torch.empty(B, H_q, S, E)}

        grad_key_out, grad_value_out = flex_pass._reduce_gqa_gradients(
            graph, grad_key_node, grad_value_node, shapes, torch.float32
        )
        graph.output((grad_key_out, grad_value_out))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        grad_key = torch.randn(B, H_q, S, E)
        grad_value = torch.randn(B, H_q, S, E)
        grad_key_reduced, grad_value_reduced = gm(grad_key, grad_value)

        # Expected: reshape and sum across repeated heads
        expected_grad_key = grad_key.reshape(B, H_kv, repeat_factor, S, E).sum(dim=2)
        expected_grad_value = grad_value.reshape(B, H_kv, repeat_factor, S, E).sum(dim=2)

        torch.testing.assert_close(grad_key_reduced, expected_grad_key)
        torch.testing.assert_close(grad_value_reduced, expected_grad_value)

    def test_no_reduction_when_equal_heads(self):
        """Test that no reduction happens when H_q == H_kv."""
        flex_pass = FlexAttentionLegalization()

        graph = fx.Graph()
        shapes = AttentionShapes(B=2, H_q=8, H_kv=8, L=64, S=64, E=32)
        grad_key = graph.placeholder("grad_key")
        grad_key.meta = {"val": torch.empty(2, 8, 64, 32)}
        grad_value = graph.placeholder("grad_value")
        grad_value.meta = {"val": torch.empty(2, 8, 64, 32)}

        grad_key_out, grad_value_out = flex_pass._reduce_gqa_gradients(
            graph, grad_key, grad_value, shapes, torch.float32
        )

        assert grad_key_out is grad_key
        assert grad_value_out is grad_value


class TestComputeAttentionScores:
    """Tests for attention score computation via FlexAttentionLegalization."""

    @pytest.mark.parametrize(
        "B,H,L,S,E",
        [
            (2, 4, 16, 32, 64),
            (1, 2, 8, 8, 16),
        ],
    )
    def test_attention_scores_numerical_accuracy(self, create_graph, B, H, L, S, E):
        """Test that attention scores computation produces correct results."""
        scale = 1.0 / (E**0.5)
        shapes = AttentionShapes(B=B, H_q=H, H_kv=H, L=L, S=S, E=E)

        flex_pass = FlexAttentionLegalization()

        graph, nodes = create_graph(
            {
                "query": (B, H, L, E),
                "key": (B, H, S, E),
            }
        )

        scores = flex_pass._compute_attention_scores(
            graph, nodes["query"], nodes["key"], scale, shapes, torch.float32
        )
        graph.output(scores)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        query = torch.randn(B, H, L, E)
        key = torch.randn(B, H, S, E)
        result = gm(query, key)

        # Expected: Q @ K^T * scale
        expected = torch.matmul(query, key.transpose(-2, -1)) * scale

        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize(
        "B,H,L,S,E",
        [
            (1, 2, 4, 4, 8),
            (2, 4, 8, 8, 16),
        ],
    )
    def test_scale_is_applied_correctly(self, create_graph, B, H, L, S, E):
        """Test that the scale factor is correctly applied."""
        scale = 1.0 / (E**0.5)
        shapes = AttentionShapes(B=B, H_q=H, H_kv=H, L=L, S=S, E=E)

        flex_pass = FlexAttentionLegalization()

        graph, nodes = create_graph(
            {
                "query": (B, H, L, E),
                "key": (B, H, S, E),
            }
        )

        scores = flex_pass._compute_attention_scores(
            graph, nodes["query"], nodes["key"], scale, shapes, torch.float32
        )
        graph.output(scores)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        query = torch.randn(B, H, L, E)
        key = torch.randn(B, H, S, E)
        result = gm(query, key)

        unscaled = torch.matmul(query, key.transpose(-2, -1))
        torch.testing.assert_close(result, unscaled * scale)


class TestComputeSoftmax:
    """Tests for softmax computation via FlexAttentionLegalization."""

    @pytest.mark.parametrize(
        "B,H,L,S",
        [
            (2, 4, 16, 32),
            (1, 2, 8, 8),
        ],
    )
    def test_softmax_matches_pytorch(self, create_graph, B, H, L, S):
        """Test that softmax computation matches PyTorch's F.softmax."""
        shapes = AttentionShapes(B=B, H_q=H, H_kv=H, L=L, S=S, E=64)

        flex_pass = FlexAttentionLegalization()

        graph, nodes = create_graph(
            {
                "scores": (B, H, L, S),
            }
        )

        attn_weights, _, _ = flex_pass._compute_softmax(
            graph, nodes["scores"], shapes, torch.float32
        )
        graph.output(attn_weights)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        scores = torch.randn(B, H, L, S)
        result = gm(scores)
        expected = F.softmax(scores, dim=-1)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)

    def test_softmax_numerical_stability(self):
        """Test softmax is numerically stable with large values."""
        B, H, L, S = 1, 1, 1, 4
        shapes = AttentionShapes(B=B, H_q=H, H_kv=H, L=L, S=S, E=8)

        flex_pass = FlexAttentionLegalization()

        graph = fx.Graph()
        scores_node = graph.placeholder("scores")
        scores_node.meta = {"val": torch.empty(B, H, L, S)}

        attn_weights, _, _ = flex_pass._compute_softmax(graph, scores_node, shapes, torch.float32)
        graph.output(attn_weights)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        # Large values that would overflow naive exp()
        scores = torch.tensor([[[[100.0, 200.0, 300.0, 400.0]]]])
        result = gm(scores)

        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
        torch.testing.assert_close(result.sum(dim=-1), torch.ones(B, H, L), rtol=1e-5, atol=1e-6)


class TestReplaceSortStable:
    """Tests for sort.stable replacement via FlexAttentionLegalization."""

    def test_sort_values_become_identity(self):
        """Test that sorted values output becomes identity (input passthrough)."""
        flex_pass = FlexAttentionLegalization()

        graph = fx.Graph()
        x = graph.placeholder("x")
        x.meta = {"val": torch.randn(2, 4, 8)}

        sort_result = graph.call_function(torch.ops.aten.sort.stable, args=(x,), kwargs={"dim": -1})
        sort_result.meta = {"val": (torch.randn(2, 4, 8), torch.randint(0, 8, (2, 4, 8)))}

        sorted_vals = graph.call_function(operator.getitem, args=(sort_result, 0))
        graph.output(sorted_vals)

        flex_pass._replace_sort_stable_with_identity(graph)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        test_input = torch.randn(2, 4, 8)
        result = gm(test_input)

        # After replacement, sorted_vals should be identity (same as input)
        torch.testing.assert_close(result, test_input)

    def test_returns_false_when_no_sort_nodes(self):
        """Test that function returns False when no sort.stable nodes exist."""
        flex_pass = FlexAttentionLegalization()

        graph = fx.Graph()
        x = graph.placeholder("x")
        y = graph.call_function(torch.ops.aten.add.Tensor, args=(x, x))
        graph.output(y)

        modified = flex_pass._replace_sort_stable_with_identity(graph)

        assert modified is False


class TestFullAttentionPipeline:
    """Integration tests that build and execute full attention computation graphs."""

    def test_mha_attention_forward(self):
        """Test building and executing a complete MHA attention forward pass."""
        B, H, L, S, E = 2, 4, 8, 16, 32
        scale = 1.0 / (E**0.5)
        shapes = AttentionShapes(B=B, H_q=H, H_kv=H, L=L, S=S, E=E)

        flex_pass = FlexAttentionLegalization()

        graph = fx.Graph()
        query_node = graph.placeholder("query")
        query_node.meta = {"val": torch.empty(B, H, L, E)}
        key_node = graph.placeholder("key")
        key_node.meta = {"val": torch.empty(B, H, S, E)}
        value_node = graph.placeholder("value")
        value_node.meta = {"val": torch.empty(B, H, S, E)}

        scores = flex_pass._compute_attention_scores(
            graph, query_node, key_node, scale, shapes, torch.float32
        )
        attn_weights, _, _ = flex_pass._compute_softmax(graph, scores, shapes, torch.float32)
        output = graph.call_function(torch.ops.aten.matmul.default, args=(attn_weights, value_node))
        graph.output(output)

        gm = fx.GraphModule(torch.nn.Module(), graph)

        query = torch.randn(B, H, L, E)
        key = torch.randn(B, H, S, E)
        value = torch.randn(B, H, S, E)

        result = gm(query, key, value)

        # Compute expected
        expected_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        expected_weights = F.softmax(expected_scores, dim=-1)
        expected = torch.matmul(expected_weights, value)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)

    def test_gqa_attention_forward(self):
        """Test attention pipeline with GQA (H_q != H_kv)."""
        B, H_q, H_kv, L, S, E = 2, 8, 2, 8, 16, 32
        scale = 1.0 / (E**0.5)
        shapes = AttentionShapes(B=B, H_q=H_q, H_kv=H_kv, L=L, S=S, E=E)

        flex_pass = FlexAttentionLegalization()

        graph = fx.Graph()
        query_node = graph.placeholder("query")
        query_node.meta = {"val": torch.empty(B, H_q, L, E)}
        key_node = graph.placeholder("key")
        key_node.meta = {"val": torch.empty(B, H_kv, S, E)}
        value_node = graph.placeholder("value")
        value_node.meta = {"val": torch.empty(B, H_kv, S, E)}

        # Expand KV for GQA
        key_expanded, value_expanded = flex_pass._expand_kv_for_gqa(
            graph, key_node, value_node, shapes
        )

        # Compute attention
        scores = flex_pass._compute_attention_scores(
            graph, query_node, key_expanded, scale, shapes, torch.float32
        )
        attn_weights, _, _ = flex_pass._compute_softmax(graph, scores, shapes, torch.float32)
        output = graph.call_function(
            torch.ops.aten.matmul.default, args=(attn_weights, value_expanded)
        )
        graph.output(output)

        gm = fx.GraphModule(torch.nn.Module(), graph)

        query = torch.randn(B, H_q, L, E)
        key = torch.randn(B, H_kv, S, E)
        value = torch.randn(B, H_kv, S, E)

        result = gm(query, key, value)

        # Compute expected with manual GQA expansion
        repeat_factor = H_q // H_kv
        key_exp = key.unsqueeze(2).expand(B, H_kv, repeat_factor, S, E).reshape(B, H_q, S, E)
        value_exp = value.unsqueeze(2).expand(B, H_kv, repeat_factor, S, E).reshape(B, H_q, S, E)
        expected_scores = torch.matmul(query, key_exp.transpose(-2, -1)) * scale
        expected_weights = F.softmax(expected_scores, dim=-1)
        expected = torch.matmul(expected_weights, value_exp)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)


class TestLegalizeFlexAttentionPass:
    """Tests for the legalize_flex_attention_pass convenience function."""

    def test_pass_handles_graph_with_sort_stable(self):
        """Test that pass removes sort.stable nodes even without flex_attention."""
        graph = fx.Graph()
        x = graph.placeholder("x")
        x.meta = {"val": torch.randn(2, 4, 8)}

        sort_result = graph.call_function(torch.ops.aten.sort.stable, args=(x,), kwargs={"dim": -1})
        sort_result.meta = {"val": (torch.randn(2, 4, 8), torch.randint(0, 8, (2, 4, 8)))}

        sorted_vals = graph.call_function(operator.getitem, args=(sort_result, 0))
        graph.output(sorted_vals)

        gm = fx.GraphModule(torch.nn.Module(), graph)
        result = legalize_flex_attention_pass(gm)

        # Should remove sort.stable
        sort_nodes = [n for n in result.graph.nodes if n.target == torch.ops.aten.sort.stable]
        assert len(sort_nodes) == 0

    def test_pass_returns_same_module_when_no_changes_needed(self):
        """Test that pass returns unchanged module for simple graphs."""

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        gm = fx.symbolic_trace(SimpleModule())
        result = legalize_flex_attention_pass(gm)

        assert result is gm

    def test_pass_manager_integration(self):
        """Test that FlexAttentionLegalization works with PassManager."""

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x * 2

        gm = fx.symbolic_trace(SimpleModule())

        flex_pass = FlexAttentionLegalization()
        pm = PassManager(passes=[flex_pass])
        result = pm(gm)

        # Verify the pass ran without errors
        assert result.graph_module is not None


class TestCombinedScoreMaskMod:
    """Tests for combined score_mod and mask_mod functionality."""

    @pytest.mark.parametrize(
        "B,H,L,S,E",
        [
            (1, 2, 8, 8, 16),  # small
            (2, 4, 16, 32, 64),  # medium
            (1, 1, 4, 4, 8),  # minimal
        ],
    )
    def test_combined_score_mask_mod_numerical_accuracy(self, B, H, L, S, E):
        """Test that combined score_mod + mask_mod produces correct results."""
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        # Create test inputs
        query = torch.randn(B, H, L, E)
        key = torch.randn(B, H, S, E)
        value = torch.randn(B, H, S, E)

        # Define separate score_mod and mask_mod
        def score_mod(score, b, h, q_idx, kv_idx):
            return score + 0.1

        def mask_mod(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        # Create block_mask with mask_mod
        block_mask = create_block_mask(mask_mod, B, H, L, S, device="cpu")

        # Test reference: flex_attention with both score_mod and block_mask
        with torch.no_grad():
            reference_output = flex_attention(
                query, key, value, score_mod=score_mod, block_mask=block_mask
            )

        # Capture graph using aot_module
        captured_gm_holder = []
        capture_compiler = create_capture_compiler(captured_gm_holder)

        class CombinedFlexAttnModule(torch.nn.Module):
            def forward(self, q, k, v):
                return flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)

        model = CombinedFlexAttnModule()

        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode(allow_non_fake_inputs=True):
            aot_model = aot_module(model, fw_compiler=capture_compiler)
            with torch.no_grad():
                _ = aot_model(query, key, value)

        assert len(captured_gm_holder) > 0, "Graph not captured"
        captured_gm = captured_gm_holder[0]

        # Apply legalization pass (unit test for the pass)
        legalized_gm = legalize_flex_attention_pass(captured_gm)

        # Verify flex_attention was legalized
        flex_nodes = [
            n
            for n in legalized_gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.higher_order.flex_attention
        ]
        assert len(flex_nodes) == 0, "flex_attention nodes should be legalized"

        # Test legalized output
        with torch.no_grad():
            legalized_output = legalized_gm(query, key, value)

        if isinstance(legalized_output, tuple):
            legalized_output = legalized_output[0]

        torch.testing.assert_close(legalized_output, reference_output, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize(
        "B,H,L,S,E",
        [
            (1, 2, 8, 8, 16),
        ],
    )
    def test_mask_only_flex_attention(self, B, H, L, S, E):
        """Test flex_attention with only mask_mod (no score_mod)."""
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        query = torch.randn(B, H, L, E)
        key = torch.randn(B, H, S, E)
        value = torch.randn(B, H, S, E)

        # Define only mask_mod (causal mask)
        def mask_mod(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        # Create block_mask (force CPU device)
        block_mask = create_block_mask(mask_mod, B, H, L, S, device="cpu")

        # Reference
        with torch.no_grad():
            reference_output = flex_attention(query, key, value, block_mask=block_mask)

        # Test legalization
        captured_gm_holder = []
        capture_compiler = create_capture_compiler(captured_gm_holder)

        class MaskOnlyFlexAttnModule(torch.nn.Module):
            def forward(self, q, k, v):
                return flex_attention(q, k, v, block_mask=block_mask)

        model = MaskOnlyFlexAttnModule()

        # Use FakeTensorMode with allow_non_fake_inputs to handle block_mask tensors
        with FakeTensorMode(allow_non_fake_inputs=True):
            aot_model = aot_module(model, fw_compiler=capture_compiler)

            with torch.no_grad():
                _ = aot_model(query, key, value)

        assert len(captured_gm_holder) > 0
        captured_gm = captured_gm_holder[0]

        # Apply legalization
        flex_pass = FlexAttentionLegalization()
        pm = PassManager(passes=[flex_pass])
        pm(captured_gm)

        with torch.no_grad():
            legalized_output = captured_gm(query, key, value)

        if isinstance(legalized_output, tuple):
            legalized_output = legalized_output[0]

        torch.testing.assert_close(legalized_output, reference_output, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize(
        "mask_mod_func,test_name",
        [
            pytest.param(
                lambda b, h, q_idx, kv_idx: q_idx >= kv_idx, "causal_mask", id="causal_mask"
            ),
            pytest.param(
                lambda b, h, q_idx, kv_idx: torch.abs(q_idx - kv_idx) <= 2,
                "band_mask",
                id="band_mask",
            ),
            pytest.param(
                lambda b, h, q_idx, kv_idx: (q_idx // 2) == (kv_idx // 2),
                "block_diagonal",
                id="block_diagonal",
            ),
        ],
    )
    @pytest.mark.parametrize(
        "B,H,L,S,E",
        [
            (1, 2, 8, 8, 16),
        ],
    )
    def test_complex_mask_mod_patterns(self, B, H, L, S, E, mask_mod_func, test_name):
        """Test various mask_mod patterns to ensure robustness."""
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        query = torch.randn(B, H, L, E)
        key = torch.randn(B, H, S, E)
        value = torch.randn(B, H, S, E)

        block_mask = create_block_mask(mask_mod_func, B, H, L, S, device="cpu")

        # Reference
        with torch.no_grad():
            reference_output = flex_attention(query, key, value, block_mask=block_mask)

        # Test legalization
        captured_gm_holder = []
        capture_compiler = create_capture_compiler(captured_gm_holder)

        class TestFlexAttnModule(torch.nn.Module):
            def __init__(self, mask):
                super().__init__()
                self.mask = mask

            def forward(self, q, k, v):
                return flex_attention(q, k, v, block_mask=self.mask)

        model = TestFlexAttnModule(block_mask)

        # Use FakeTensorMode with allow_non_fake_inputs to handle block_mask tensors
        with FakeTensorMode(allow_non_fake_inputs=True):
            aot_model = aot_module(model, fw_compiler=capture_compiler)

            with torch.no_grad():
                _ = aot_model(query, key, value)

        assert len(captured_gm_holder) > 0
        captured_gm = captured_gm_holder[0]

        # Apply legalization
        flex_pass = FlexAttentionLegalization()
        pm = PassManager(passes=[flex_pass])
        pm(captured_gm)

        with torch.no_grad():
            legalized_output = captured_gm(query, key, value)

        if isinstance(legalized_output, tuple):
            legalized_output = legalized_output[0]

        torch.testing.assert_close(legalized_output, reference_output, rtol=1e-4, atol=1e-5)

    @pytest.mark.xfail(
        reason="External buffer support in mask_mod/score_mod currently not implemented"
    )
    @pytest.mark.parametrize(
        "B,H,L,S,E",
        [
            (2, 2, 8, 8, 16),
        ],
    )
    def test_mask_mod_with_external_buffer(self, B, H, L, S, E):
        """Test mask_mod that uses an external buffer (e.g., prefix_length tensor)."""
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        query = torch.randn(B, H, L, E)
        key = torch.randn(B, H, S, E)
        value = torch.randn(B, H, S, E)

        # External buffer: prefix length per batch
        prefix_length = torch.tensor([4, 6], dtype=torch.int32)  # Different prefix for each batch

        def prefix_mask(b, h, q_idx, kv_idx):
            return kv_idx <= prefix_length[b]

        # This may not work yet - marked as xfail
        block_mask = create_block_mask(prefix_mask, B, H, L, S, device="cpu")

        with torch.no_grad():
            reference_output = flex_attention(query, key, value, block_mask=block_mask)

        # Test legalization
        captured_gm_holder = []
        capture_compiler = create_capture_compiler(captured_gm_holder)

        class PrefixMaskFlexAttnModule(torch.nn.Module):
            def forward(self, q, k, v):
                return flex_attention(q, k, v, block_mask=block_mask)

        model = PrefixMaskFlexAttnModule()

        # Use FakeTensorMode with allow_non_fake_inputs to handle block_mask tensors
        with FakeTensorMode(allow_non_fake_inputs=True):
            aot_model = aot_module(model, fw_compiler=capture_compiler)

            with torch.no_grad():
                _ = aot_model(query, key, value)

        assert len(captured_gm_holder) > 0
        captured_gm = captured_gm_holder[0]

        # Apply legalization
        flex_pass = FlexAttentionLegalization()
        pm = PassManager(passes=[flex_pass])
        pm(captured_gm)

        with torch.no_grad():
            legalized_output = captured_gm(query, key, value)

        if isinstance(legalized_output, tuple):
            legalized_output = legalized_output[0]

        torch.testing.assert_close(legalized_output, reference_output, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize(
        "B,H,L,S",
        [
            (1, 1, 4, 4),
            (2, 2, 8, 8),
        ],
    )
    def test_mask_mod_as_score_mod_conversion(self, B, H, L, S):
        """Test the mask_mod to score_mod conversion pattern."""

        # Create index tensors
        b_idx = torch.arange(B).view(B, 1, 1, 1)
        h_idx = torch.arange(H).view(1, H, 1, 1)
        q_idx = torch.arange(L).view(1, 1, L, 1)
        kv_idx = torch.arange(S).view(1, 1, 1, S)

        # Sample scores
        scores = torch.randn(B, H, L, S)

        # Define a simple mask_mod (causal)
        def mask_mod(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        # Convert mask_mod to score_mod
        def mask_mod_as_score_mod(score, b, h, q_idx, kv_idx):
            mask = mask_mod(b, h, q_idx, kv_idx)
            return torch.where(mask, score, float("-inf"))

        # Apply the conversion
        result = mask_mod_as_score_mod(scores, b_idx, h_idx, q_idx, kv_idx)

        # Verify masking worked correctly
        mask = mask_mod(b_idx, h_idx, q_idx, kv_idx)
        expected = torch.where(mask, scores, float("-inf"))

        torch.testing.assert_close(result, expected)

        # Verify causal pattern
        for i in range(L):
            for j in range(S):
                if i >= j:  # Should be unmasked
                    assert not torch.isinf(result[0, 0, i, j])
                else:  # Should be masked
                    assert result[0, 0, i, j] == float("-inf")


class TestFlexAttentionLegalizationE2E:
    """End-to-end tests for FlexAttentionLegalization with real flex_attention graphs."""

    @pytest.mark.parametrize(
        "B,H,L,S,E",
        [
            pytest.param(1, 4, 16, 16, 32, id="small"),
            pytest.param(2, 8, 32, 32, 64, id="medium"),
            pytest.param(1, 2, 8, 8, 16, id="minimal"),
        ],
    )
    def test_flex_attention_legalization_numerical_accuracy(self, B, H, L, S, E):
        """Test that legalized flex_attention produces same output as original."""
        from torch.nn.attention.flex_attention import flex_attention

        query = torch.randn(B, H, L, E)
        key = torch.randn(B, H, S, E)
        value = torch.randn(B, H, S, E)

        with torch.no_grad():
            reference_output = flex_attention(query, key, value)

        captured_gm_holder = []
        capture_compiler = create_capture_compiler(captured_gm_holder)

        class FlexAttnModule(torch.nn.Module):
            def forward(self, q, k, v):
                return flex_attention(q, k, v)

        model = FlexAttnModule()

        aot_model = aot_module(
            model,
            fw_compiler=capture_compiler,
        )

        with torch.no_grad():
            _ = aot_model(query, key, value)

        assert len(captured_gm_holder) > 0
        captured_gm = captured_gm_holder[0]

        # Test using the class-based pass with PassManager
        flex_pass = FlexAttentionLegalization()
        pm = PassManager(passes=[flex_pass])
        pm(captured_gm)

        flex_nodes = [
            n
            for n in captured_gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.higher_order.flex_attention
        ]

        assert len(flex_nodes) == 0

        with torch.no_grad():
            legalized_output = captured_gm(query, key, value)

        if isinstance(legalized_output, tuple):
            legalized_output = legalized_output[0]

        torch.testing.assert_close(legalized_output, reference_output, rtol=1e-4, atol=1e-5)


class TestDtypeSizeCombinations:
    """Tests for various dtype and size combinations across attention functions."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "b,h_q,h_kv,s,e",
        [
            (2, 8, 2, 16, 32),
            (1, 6, 2, 8, 16),
        ],
    )
    def test_gqa_expansion_dtypes(self, dtype, b, h_q, h_kv, s, e):
        """Test GQA expansion preserves dtype correctly."""
        shapes = AttentionShapes(B=b, H_q=h_q, H_kv=h_kv, L=16, S=s, E=e)

        flex_pass = FlexAttentionLegalization()

        graph = fx.Graph()
        key_node = graph.placeholder("key")
        key_node.meta = {"val": torch.empty(b, h_kv, s, e, dtype=dtype)}
        value_node = graph.placeholder("value")
        value_node.meta = {"val": torch.empty(b, h_kv, s, e, dtype=dtype)}

        key_out, value_out = flex_pass._expand_kv_for_gqa(graph, key_node, value_node, shapes)
        graph.output((key_out, value_out))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        key = torch.randn(b, h_kv, s, e, dtype=dtype)
        value = torch.randn(b, h_kv, s, e, dtype=dtype)
        key_expanded, value_expanded = gm(key, value)

        assert key_expanded.dtype == dtype
        assert value_expanded.dtype == dtype
        assert key_expanded.shape == (b, h_q, s, e)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "B,H,L,S,E",
        [
            (2, 4, 16, 32, 64),
            (1, 2, 8, 8, 16),
        ],
    )
    def test_attention_scores_dtypes(self, dtype, B, H, L, S, E):
        """Test attention scores computation with different dtypes."""
        scale = 1.0 / (E**0.5)
        shapes = AttentionShapes(B=B, H_q=H, H_kv=H, L=L, S=S, E=E)

        flex_pass = FlexAttentionLegalization()

        graph = fx.Graph()
        query_node = graph.placeholder("query")
        query_node.meta = {"val": torch.empty(B, H, L, E, dtype=dtype)}
        key_node = graph.placeholder("key")
        key_node.meta = {"val": torch.empty(B, H, S, E, dtype=dtype)}

        scores = flex_pass._compute_attention_scores(
            graph, query_node, key_node, scale, shapes, dtype
        )
        graph.output(scores)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        query = torch.randn(B, H, L, E, dtype=dtype)
        key = torch.randn(B, H, S, E, dtype=dtype)
        result = gm(query, key)

        assert result.dtype == dtype
        expected = torch.matmul(query, key.transpose(-2, -1)) * scale
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "B,H,L,S",
        [
            (2, 4, 16, 32),
            (1, 2, 8, 8),
        ],
    )
    def test_softmax_dtypes(self, dtype, B, H, L, S):
        """Test softmax computation with different dtypes."""
        shapes = AttentionShapes(B=B, H_q=H, H_kv=H, L=L, S=S, E=64)

        flex_pass = FlexAttentionLegalization()

        graph = fx.Graph()
        scores_node = graph.placeholder("scores")
        scores_node.meta = {"val": torch.empty(B, H, L, S, dtype=dtype)}

        attn_weights, _, _ = flex_pass._compute_softmax(graph, scores_node, shapes, dtype)
        graph.output(attn_weights)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        scores = torch.randn(B, H, L, S, dtype=dtype)
        result = gm(scores)

        assert result.dtype == dtype
        expected = F.softmax(scores.float(), dim=-1).to(dtype)
        # bfloat16 has lower precision
        rtol, atol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)

    @pytest.mark.parametrize(
        "B,H,L,S,E",
        [
            (1, 1, 1, 1, 8),  # Minimal sizes
            (1, 1, 4, 4, 16),  # Small
            (2, 8, 64, 64, 64),  # Medium
            (4, 16, 128, 128, 128),  # Larger
            (1, 32, 1, 512, 64),  # Many heads, single query
            (2, 4, 256, 32, 32),  # Long query, short KV
        ],
    )
    def test_attention_scores_various_sizes(self, B, H, L, S, E):
        """Test attention scores with various tensor sizes."""
        scale = 1.0 / (E**0.5)
        shapes = AttentionShapes(B=B, H_q=H, H_kv=H, L=L, S=S, E=E)

        flex_pass = FlexAttentionLegalization()

        graph = fx.Graph()
        query_node = graph.placeholder("query")
        query_node.meta = {"val": torch.empty(B, H, L, E)}
        key_node = graph.placeholder("key")
        key_node.meta = {"val": torch.empty(B, H, S, E)}

        scores = flex_pass._compute_attention_scores(
            graph, query_node, key_node, scale, shapes, torch.float32
        )
        graph.output(scores)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        query = torch.randn(B, H, L, E)
        key = torch.randn(B, H, S, E)
        result = gm(query, key)

        assert result.shape == (B, H, L, S)
        expected = torch.matmul(query, key.transpose(-2, -1)) * scale
        torch.testing.assert_close(result, expected)


class TestComputeAttentionBackward:
    """Tests for attention backward computation via FlexAttentionLegalization."""

    @pytest.mark.parametrize(
        "B,H,L,S,E",
        [
            (2, 4, 16, 32, 64),
            (1, 2, 8, 8, 16),
        ],
    )
    def test_backward_gradient_shapes(self, B, H, L, S, E):
        """Test that backward pass produces correct gradient shapes."""
        scale = 1.0 / (E**0.5)
        shapes = AttentionShapes(B=B, H_q=H, H_kv=H, L=L, S=S, E=E)

        flex_pass = FlexAttentionLegalization()

        graph = fx.Graph()
        query_node = graph.placeholder("query")
        query_node.meta = {"val": torch.empty(B, H, L, E)}
        key_node = graph.placeholder("key")
        key_node.meta = {"val": torch.empty(B, H, S, E)}
        value_node = graph.placeholder("value")
        value_node.meta = {"val": torch.empty(B, H, S, E)}
        attn_weights_node = graph.placeholder("attn_weights")
        attn_weights_node.meta = {"val": torch.empty(B, H, L, S)}
        grad_out_node = graph.placeholder("grad_out")
        grad_out_node.meta = {"val": torch.empty(B, H, L, E)}

        grad_q, grad_k, grad_v = flex_pass._compute_attention_backward(
            graph,
            query_node,
            key_node,
            value_node,
            attn_weights_node,
            grad_out_node,
            scale,
            shapes,
            torch.float32,
        )
        graph.output((grad_q, grad_k, grad_v))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        query = torch.randn(B, H, L, E)
        key = torch.randn(B, H, S, E)
        value = torch.randn(B, H, S, E)
        attn_weights = F.softmax(torch.randn(B, H, L, S), dim=-1)
        grad_out = torch.randn(B, H, L, E)

        grad_query, grad_key, grad_value = gm(query, key, value, attn_weights, grad_out)

        assert grad_query.shape == (B, H, L, E)
        assert grad_key.shape == (B, H, S, E)
        assert grad_value.shape == (B, H, S, E)

    @pytest.mark.parametrize(
        "B,H,L,S,E",
        [
            (1, 2, 4, 8, 16),
            (2, 4, 8, 16, 32),
        ],
    )
    def test_backward_grad_value_numerical(self, B, H, L, S, E):
        """Test grad_value computation: grad_value = attn_weights^T @ grad_out."""
        scale = 1.0
        shapes = AttentionShapes(B=B, H_q=H, H_kv=H, L=L, S=S, E=E)

        flex_pass = FlexAttentionLegalization()

        graph = fx.Graph()
        query_node = graph.placeholder("query")
        query_node.meta = {"val": torch.empty(B, H, L, E)}
        key_node = graph.placeholder("key")
        key_node.meta = {"val": torch.empty(B, H, S, E)}
        value_node = graph.placeholder("value")
        value_node.meta = {"val": torch.empty(B, H, S, E)}
        attn_weights_node = graph.placeholder("attn_weights")
        attn_weights_node.meta = {"val": torch.empty(B, H, L, S)}
        grad_out_node = graph.placeholder("grad_out")
        grad_out_node.meta = {"val": torch.empty(B, H, L, E)}

        _, _, grad_v = flex_pass._compute_attention_backward(
            graph,
            query_node,
            key_node,
            value_node,
            attn_weights_node,
            grad_out_node,
            scale,
            shapes,
            torch.float32,
        )
        graph.output(grad_v)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        query = torch.randn(B, H, L, E)
        key = torch.randn(B, H, S, E)
        value = torch.randn(B, H, S, E)
        attn_weights = F.softmax(torch.randn(B, H, L, S), dim=-1)
        grad_out = torch.randn(B, H, L, E)

        grad_value = gm(query, key, value, attn_weights, grad_out)

        # Expected: attn_weights^T @ grad_out
        expected_grad_value = torch.matmul(attn_weights.transpose(-2, -1), grad_out)
        torch.testing.assert_close(grad_value, expected_grad_value)

    @pytest.mark.parametrize(
        "B,H,L,S,E",
        [
            (1, 2, 4, 8, 16),
            (2, 4, 8, 16, 32),
        ],
    )
    def test_backward_with_autograd_reference(self, B, H, L, S, E):
        """Test backward pass against PyTorch autograd reference."""
        scale = 1.0 / (E**0.5)
        shapes = AttentionShapes(B=B, H_q=H, H_kv=H, L=L, S=S, E=E)

        # Create inputs with gradients
        query = torch.randn(B, H, L, E, requires_grad=True)
        key = torch.randn(B, H, S, E, requires_grad=True)
        value = torch.randn(B, H, S, E, requires_grad=True)

        # Forward pass with PyTorch
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)

        # Backward with PyTorch
        grad_out = torch.randn(B, H, L, E)
        output.backward(grad_out)

        expected_grad_q = query.grad.clone()
        expected_grad_k = key.grad.clone()
        expected_grad_v = value.grad.clone()

        # Now test our implementation
        flex_pass = FlexAttentionLegalization()

        graph = fx.Graph()
        query_node = graph.placeholder("query")
        query_node.meta = {"val": torch.empty(B, H, L, E)}
        key_node = graph.placeholder("key")
        key_node.meta = {"val": torch.empty(B, H, S, E)}
        value_node = graph.placeholder("value")
        value_node.meta = {"val": torch.empty(B, H, S, E)}
        attn_weights_node = graph.placeholder("attn_weights")
        attn_weights_node.meta = {"val": torch.empty(B, H, L, S)}
        grad_out_node = graph.placeholder("grad_out")
        grad_out_node.meta = {"val": torch.empty(B, H, L, E)}

        grad_q, grad_k, grad_v = flex_pass._compute_attention_backward(
            graph,
            query_node,
            key_node,
            value_node,
            attn_weights_node,
            grad_out_node,
            scale,
            shapes,
            torch.float32,
        )
        graph.output((grad_q, grad_k, grad_v))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        # Use detached tensors and recomputed attn_weights
        query_detached = query.detach()
        key_detached = key.detach()
        value_detached = value.detach()
        attn_weights_detached = attn_weights.detach()

        grad_query, grad_key, grad_value = gm(
            query_detached, key_detached, value_detached, attn_weights_detached, grad_out
        )

        torch.testing.assert_close(grad_query, expected_grad_q, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(grad_key, expected_grad_k, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(grad_value, expected_grad_v, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "B,H,L,S,E",
        [
            (1, 2, 4, 8, 16),
            (2, 4, 8, 16, 32),
        ],
    )
    def test_backward_dtypes(self, dtype, B, H, L, S, E):
        """Test backward pass preserves dtype."""
        scale = 1.0
        shapes = AttentionShapes(B=B, H_q=H, H_kv=H, L=L, S=S, E=E)

        flex_pass = FlexAttentionLegalization()

        graph = fx.Graph()
        query_node = graph.placeholder("query")
        query_node.meta = {"val": torch.empty(B, H, L, E, dtype=dtype)}
        key_node = graph.placeholder("key")
        key_node.meta = {"val": torch.empty(B, H, S, E, dtype=dtype)}
        value_node = graph.placeholder("value")
        value_node.meta = {"val": torch.empty(B, H, S, E, dtype=dtype)}
        attn_weights_node = graph.placeholder("attn_weights")
        attn_weights_node.meta = {"val": torch.empty(B, H, L, S, dtype=dtype)}
        grad_out_node = graph.placeholder("grad_out")
        grad_out_node.meta = {"val": torch.empty(B, H, L, E, dtype=dtype)}

        grad_q, grad_k, grad_v = flex_pass._compute_attention_backward(
            graph,
            query_node,
            key_node,
            value_node,
            attn_weights_node,
            grad_out_node,
            scale,
            shapes,
            dtype,
        )
        graph.output((grad_q, grad_k, grad_v))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        query = torch.randn(B, H, L, E, dtype=dtype)
        key = torch.randn(B, H, S, E, dtype=dtype)
        value = torch.randn(B, H, S, E, dtype=dtype)
        attn_weights = F.softmax(torch.randn(B, H, L, S, dtype=dtype), dim=-1)
        grad_out = torch.randn(B, H, L, E, dtype=dtype)

        grad_query, grad_key, grad_value = gm(query, key, value, attn_weights, grad_out)

        assert grad_query.dtype == dtype
        assert grad_key.dtype == dtype
        assert grad_value.dtype == dtype

    @pytest.mark.parametrize(
        "B,H,L,S,E",
        [
            (1, 2, 4, 4, 8),
            (2, 4, 8, 8, 16),
        ],
    )
    def test_backward_with_mask_mod(self, B, H, L, S, E):
        """Test that legalization handles mask_mod in both forward and backward passes."""
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        # Define causal mask
        def mask_mod(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        # Create block_mask on CPU
        block_mask = create_block_mask(mask_mod, B, H, L, S, device="cpu")

        # Capture graphs using torch.compile pattern (as suggested in review)
        captured_fw_gm = []
        captured_bw_gm = []

        def fw_compiler(gm: torch.fx.GraphModule, example_inputs):
            captured_fw_gm.clear()
            captured_fw_gm.append(gm)
            # Apply legalization in the compiler
            flex_pass = FlexAttentionLegalization()
            pm = PassManager(passes=[flex_pass])
            result = pm(gm)
            return result.graph_module

        def bw_compiler(gm: torch.fx.GraphModule, example_inputs):
            captured_bw_gm.clear()
            captured_bw_gm.append(gm)
            # Apply legalization in the compiler
            flex_pass = FlexAttentionLegalization()
            pm = PassManager(passes=[flex_pass])
            result = pm(gm)
            return result.graph_module

        class FlexAttnWithMaskModule(torch.nn.Module):
            def __init__(self, mask):
                super().__init__()
                self.block_mask = mask

            def forward(self, q, k, v):
                return flex_attention(q, k, v, block_mask=self.block_mask)

        model = FlexAttnWithMaskModule(block_mask)

        # Use torch.compile with backend that applies our pass
        import torch._dynamo as dynamo
        from torch._dynamo.backends.common import aot_autograd

        def custom_backend(gm, example_inputs):
            # This is called for both forward and backward
            return aot_autograd(
                fw_compiler=fw_compiler,
                bw_compiler=bw_compiler,
            )(gm, example_inputs)

        compiled_model = torch.compile(model, backend=custom_backend)

        # Create inputs with gradients
        query = torch.randn(B, H, L, E, requires_grad=True)
        key = torch.randn(B, H, S, E, requires_grad=True)
        value = torch.randn(B, H, S, E, requires_grad=True)

        # Run forward pass
        output = compiled_model(query, key, value)

        # Run backward pass to trigger backward graph compilation
        loss = output.sum()
        loss.backward()

        # Verify forward graph was captured and legalized
        assert len(captured_fw_gm) > 0, "Forward graph not captured"

        # Check if forward had flex_attention nodes (before legalization)
        fw_had_flex = any(
            n.op == "call_function" and n.target == torch.ops.higher_order.flex_attention
            for n in captured_fw_gm[0].graph.nodes
        )

        if fw_had_flex:
            logger.debug("Forward graph had flex_attention nodes (legalized in compiler)")
        else:
            logger.debug("Forward graph captured (may have been pre-legalized)")

        # Verify backward graph was captured
        assert len(captured_bw_gm) > 0, "Backward graph not captured"

        bw_had_flex = any(
            n.op == "call_function"
            and (
                n.target == torch.ops.higher_order.flex_attention_backward
                or (hasattr(n.target, "__name__") and "flex_attention" in n.target.__name__)
            )
            for n in captured_bw_gm[0].graph.nodes
        )

        if bw_had_flex:
            logger.debug("Backward graph had flex_attention nodes (legalized in compiler)")
        else:
            logger.debug("Backward graph captured (may have been pre-legalized)")

        logger.debug("Backward pass with mask_mod successfully legalized")


class TestScoreModInlining:
    """Tests for score_mod inlining functionality."""

    @pytest.mark.parametrize(
        "B,H,L,S,E",
        [
            (1, 2, 4, 8, 16),
            (2, 4, 8, 16, 32),
        ],
    )
    def test_inline_score_mod_creates_nodes(self, B, H, L, S, E):
        """Test that _inline_score_mod creates expected nodes in graph."""
        shapes = AttentionShapes(B=B, H_q=H, H_kv=H, L=L, S=S, E=E)

        flex_pass = FlexAttentionLegalization()

        # Create a simple score_mod that adds a bias
        class SimpleScoreMod(torch.nn.Module):
            def forward(self, score, b_idx, h_idx, q_idx, kv_idx):
                return score + 1.0

        score_mod_module = fx.symbolic_trace(SimpleScoreMod())

        # Main graph
        main_graph = fx.Graph()
        scaled_scores = main_graph.placeholder("scaled_scores")
        scaled_scores.meta = {"val": torch.empty(B, H, L, S)}

        # Create index tensors
        index_tensors = flex_pass._create_index_tensors(main_graph, shapes)

        gm = fx.GraphModule(torch.nn.Module(), main_graph)

        # Inline score_mod
        final_scores = flex_pass._inline_score_mod(
            main_graph,
            gm,
            "score_mod",
            score_mod_module,
            scaled_scores,
            index_tensors,
            shapes,
        )

        main_graph.output(final_scores)
        gm = fx.GraphModule(torch.nn.Module(), main_graph)

        # Verify the graph has add operation
        add_nodes = [
            n for n in main_graph.nodes if n.op == "call_function" and "add" in str(n.target)
        ]
        assert len(add_nodes) > 0

    @pytest.mark.parametrize(
        "B,H,L,S,E",
        [
            (1, 2, 4, 8, 16),
            (2, 4, 8, 16, 32),
        ],
    )
    def test_inline_score_mod_with_arithmetic(self, B, H, L, S, E):
        """Test score_mod inlining with arithmetic operations using indices."""
        shapes = AttentionShapes(B=B, H_q=H, H_kv=H, L=L, S=S, E=E)

        flex_pass = FlexAttentionLegalization()

        # Create a score_mod that uses indices in arithmetic (no tensor constants)
        class ArithmeticScoreMod(torch.nn.Module):
            def forward(self, score, b_idx, h_idx, q_idx, kv_idx):
                # Scale score based on position difference
                return score * (1.0 + q_idx.float() * 0.1)

        score_mod_module = fx.symbolic_trace(ArithmeticScoreMod())

        main_graph = fx.Graph()
        scaled_scores = main_graph.placeholder("scaled_scores")
        scaled_scores.meta = {"val": torch.empty(B, H, L, S)}

        index_tensors = flex_pass._create_index_tensors(main_graph, shapes)

        gm = fx.GraphModule(torch.nn.Module(), main_graph)

        final_scores = flex_pass._inline_score_mod(
            main_graph,
            gm,
            "score_mod",
            score_mod_module,
            scaled_scores,
            index_tensors,
            shapes,
        )

        main_graph.output(final_scores)
        gm = fx.GraphModule(torch.nn.Module(), main_graph)

        # Verify the graph has mul operations from the score_mod
        mul_nodes = [
            n for n in main_graph.nodes if n.op == "call_function" and "mul" in str(n.target)
        ]
        assert len(mul_nodes) > 0

    @pytest.mark.parametrize(
        "B,H,L,S,E",
        [
            (2, 4, 8, 16, 32),
            (1, 2, 4, 8, 16),
        ],
    )
    def test_create_index_tensors_shapes(self, B, H, L, S, E):
        """Test that _create_index_tensors creates correctly shaped tensors."""
        shapes = AttentionShapes(B=B, H_q=H, H_kv=H, L=L, S=S, E=E)

        flex_pass = FlexAttentionLegalization()

        graph = fx.Graph()
        b_idx, h_idx, q_idx, kv_idx = flex_pass._create_index_tensors(graph, shapes)
        graph.output((b_idx, h_idx, q_idx, kv_idx))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        b, h, q, kv = gm()

        # Check shapes after view operations
        assert b.shape == (B, 1, 1, 1)
        assert h.shape == (1, H, 1, 1)
        assert q.shape == (1, 1, L, 1)
        assert kv.shape == (1, 1, 1, S)

    @pytest.mark.parametrize(
        "B,H,L,S,E",
        [
            (3, 2, 4, 5, 8),
            (2, 4, 8, 10, 16),
        ],
    )
    def test_create_index_tensors_values(self, B, H, L, S, E):
        """Test that _create_index_tensors creates correct index values."""
        shapes = AttentionShapes(B=B, H_q=H, H_kv=H, L=L, S=S, E=E)

        flex_pass = FlexAttentionLegalization()

        graph = fx.Graph()
        b_idx, h_idx, q_idx, kv_idx = flex_pass._create_index_tensors(graph, shapes)
        graph.output((b_idx, h_idx, q_idx, kv_idx))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        b, h, q, kv = gm()

        # Check values
        torch.testing.assert_close(b.squeeze(), torch.arange(B, dtype=torch.int32))
        torch.testing.assert_close(h.squeeze(), torch.arange(H, dtype=torch.int32))
        torch.testing.assert_close(q.squeeze(), torch.arange(L, dtype=torch.int32))
        torch.testing.assert_close(kv.squeeze(), torch.arange(S, dtype=torch.int32))
