# ruff: noqa: N806, N803, N812

"""Unit tests for the compile-cost estimate and the compile budget in FlexAttentionNKILowering.

The cost estimate is a pure function of the tile classes and the shapes, and the budget
tests replace the tile classification and the rewrite, so neither needs a device.
"""

import logging
from unittest.mock import patch

import pytest
import torch
import torch.fx as fx
from functorch.compile import aot_module, make_boxed_func
from torch._subclasses.fake_tensor import FakeTensorMode

from tests.neuron_dynamo_backend.unit.utils.test_utils import create_capture_compiler
from torch_neuronx.neuron_dynamo_backend.fx.passes.flex_attention_nki_lowering import (
    _COST_UNITS_PER_SECOND,
    _DEFAULT_COMPILE_BUDGET_S,
    _SCORE_MOD_COST_FACTOR,
    _TILE_FULL,
    _TILE_PARTIAL,
    _TILE_SKIP,
    FlexAttentionNKILowering,
    _compile_budget_seconds,
    _estimate_compile_cost,
)

logger = logging.getLogger(__name__)

_BUDGET_ENV = "TORCH_NEURONX_FLEX_NKI_COMPILE_BUDGET_S"
_LOWERING_LOGGER = "torch_neuronx.neuron_dynamo_backend.fx.passes.flex_attention_nki_lowering"


@pytest.fixture(autouse=True)
def disable_flex_attention_device_check():
    with patch("torch.nn.attention.flex_attention._validate_device", lambda *args, **kwargs: None):
        yield


@pytest.fixture(autouse=True)
def clear_budget_env(monkeypatch):
    monkeypatch.delenv(_BUDGET_ENV, raising=False)


def _full_grid(num_grps, num_k_tiles):
    return [[_TILE_FULL] * num_k_tiles for _ in range(num_grps)]


def _skip_grid(num_grps, num_k_tiles):
    return [[_TILE_SKIP] * num_k_tiles for _ in range(num_grps)]


def _causal_grid(num_tiles):
    """Lower triangular grid. The diagonal tile is partial, the rest is full."""
    grid = []
    for g in range(num_tiles):
        row = []
        for kt in range(num_tiles):
            if kt < g:
                row.append(_TILE_FULL)
            elif kt == g:
                row.append(_TILE_PARTIAL)
            else:
                row.append(_TILE_SKIP)
        grid.append(row)
    return grid


def _band_grid(num_tiles, half_width):
    """Sliding window grid. A tile is live when it is within half_width of the diagonal."""
    return [
        [_TILE_FULL if abs(g - kt) <= half_width else _TILE_SKIP for kt in range(num_tiles)]
        for g in range(num_tiles)
    ]


class TestEstimateCompileCost:
    def test_all_full_grid(self):
        estimate = _estimate_compile_cost(_full_grid(4, 4), batch=1, q_heads=1, has_score_mod=False)
        assert estimate.live_tiles == 16
        assert estimate.total_tiles == 16
        assert estimate.density == 1.0
        assert estimate.cost_units == 16
        assert estimate.est_seconds == 16 / _COST_UNITS_PER_SECOND

    def test_all_skip_grid(self):
        estimate = _estimate_compile_cost(_skip_grid(4, 4), batch=2, q_heads=8, has_score_mod=True)
        assert estimate.live_tiles == 0
        assert estimate.total_tiles == 16
        assert estimate.density == 0.0
        assert estimate.cost_units == 0
        assert estimate.est_seconds == 0.0

    def test_causal_grid(self):
        estimate = _estimate_compile_cost(_causal_grid(4), batch=1, q_heads=2, has_score_mod=False)
        assert estimate.live_tiles == 10
        assert estimate.total_tiles == 16
        assert estimate.density == 10 / 16
        assert estimate.cost_units == 20

    def test_band_grid(self):
        estimate = _estimate_compile_cost(_band_grid(6, 1), batch=1, q_heads=1, has_score_mod=False)
        assert estimate.live_tiles == 16
        assert estimate.total_tiles == 36
        assert estimate.density == 16 / 36
        assert estimate.cost_units == 16

    def test_partial_tiles_count_as_live(self):
        grid = [[_TILE_SKIP, _TILE_PARTIAL], [_TILE_PARTIAL, _TILE_FULL]]
        estimate = _estimate_compile_cost(grid, batch=1, q_heads=1, has_score_mod=False)
        assert estimate.live_tiles == 3

    @pytest.mark.parametrize("batch", [1, 2, 4, 8])
    def test_batch_scaling_is_linear(self, batch):
        base = _estimate_compile_cost(_causal_grid(8), batch=1, q_heads=4, has_score_mod=False)
        scaled = _estimate_compile_cost(
            _causal_grid(8), batch=batch, q_heads=4, has_score_mod=False
        )
        assert scaled.cost_units == base.cost_units * batch
        assert scaled.est_seconds == base.est_seconds * batch
        assert scaled.live_tiles == base.live_tiles

    @pytest.mark.parametrize("q_heads", [1, 2, 8, 16])
    def test_q_heads_scaling_is_linear(self, q_heads):
        base = _estimate_compile_cost(_causal_grid(8), batch=2, q_heads=1, has_score_mod=False)
        scaled = _estimate_compile_cost(
            _causal_grid(8), batch=2, q_heads=q_heads, has_score_mod=False
        )
        assert scaled.cost_units == base.cost_units * q_heads
        assert scaled.est_seconds == base.est_seconds * q_heads

    def test_score_mod_multiplies_cost_by_four(self):
        grid = _causal_grid(8)
        without = _estimate_compile_cost(grid, batch=2, q_heads=8, has_score_mod=False)
        with_score_mod = _estimate_compile_cost(grid, batch=2, q_heads=8, has_score_mod=True)
        assert _SCORE_MOD_COST_FACTOR == 4
        assert with_score_mod.cost_units == without.cost_units * 4
        assert with_score_mod.est_seconds == without.est_seconds * 4
        assert with_score_mod.live_tiles == without.live_tiles
        assert with_score_mod.has_score_mod is True

    def test_empty_grid_gives_no_cost(self):
        estimate = _estimate_compile_cost([], batch=1, q_heads=8, has_score_mod=False)
        assert estimate.total_tiles == 0
        assert estimate.density == 0.0
        assert estimate.cost_units == 0

    @pytest.mark.parametrize(
        "name,batch,has_score_mod,expected_cost_units,measured_seconds",
        [
            ("causal B=1", 1, False, 16640, 131.0),
            ("alibi B=4", 4, True, 266240, 2031.0),
        ],
    )
    def test_matches_calibration_measurements(
        self, name, batch, has_score_mod, expected_cost_units, measured_seconds
    ):
        """Reproduce two measured points. seq_len 8192 with a 128 tile gives 64 groups."""
        estimate = _estimate_compile_cost(
            _causal_grid(64), batch=batch, q_heads=8, has_score_mod=has_score_mod
        )
        assert estimate.live_tiles == 2080
        assert estimate.cost_units == expected_cost_units
        relative_error = abs(estimate.est_seconds - measured_seconds) / measured_seconds
        assert relative_error <= 0.12, f"{name}: {estimate.est_seconds} s vs {measured_seconds} s"


class TestCompileBudgetSeconds:
    def test_the_default_is_250_seconds(self):
        assert _DEFAULT_COMPILE_BUDGET_S == 250.0

    def test_unset_is_the_default(self):
        assert _compile_budget_seconds() == _DEFAULT_COMPILE_BUDGET_S

    @pytest.mark.parametrize(
        "value,expected", [("0", 0.0), ("60", 60.0), ("12.5", 12.5), ("-1", -1.0)]
    )
    def test_parses_a_number(self, monkeypatch, value, expected):
        monkeypatch.setenv(_BUDGET_ENV, value)
        assert _compile_budget_seconds() == expected

    @pytest.mark.parametrize("value", ["abc", "", "60s", "1,5"])
    def test_malformed_value_falls_back_to_the_default(self, monkeypatch, caplog, value):
        monkeypatch.setenv(_BUDGET_ENV, value)
        with caplog.at_level(logging.WARNING, logger=_LOWERING_LOGGER):
            assert _compile_budget_seconds() == _DEFAULT_COMPILE_BUDGET_S
        assert any(record.levelno == logging.WARNING for record in caplog.records)


_B, _H, _L, _S, _E = 1, 2, 8, 8, 16


def _capture_flex_graph(score_mod=None, d_head_qk=_E, d_head_v=_E, b_q=_B, b_kv=_B):
    """Capture a GraphModule that holds one flex_attention forward operator."""
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    query = torch.randn(b_q, _H, _L, d_head_qk)
    key = torch.randn(b_kv, _H, _S, d_head_qk)
    value = torch.randn(b_kv, _H, _S, d_head_v)

    block_mask = create_block_mask(
        lambda b, h, q_idx, kv_idx: q_idx >= kv_idx, b_q, _H, _L, _S, device="cpu"
    )

    class FlexAttnModule(torch.nn.Module):
        def forward(self, q, k, v):
            return flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)

    captured = []
    with FakeTensorMode(allow_non_fake_inputs=True):
        aot_model = aot_module(FlexAttnModule(), fw_compiler=create_capture_compiler(captured))
        with torch.no_grad():
            _ = aot_model(query, key, value)

    assert len(captured) == 1
    return captured[0]


class _BackwardCapturedError(Exception):
    """Stops the backward at the compiler, so the captured graph never runs on CPU."""


def _capture_flex_backward_graph(score_mod=None, d_head_qk=_E, d_head_v=_E):
    """Capture a GraphModule that holds one flex_attention_backward operator."""
    from torch._dynamo.backends.common import aot_autograd
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    inputs = [
        torch.randn(*shape, requires_grad=True)
        for shape in ((_B, _H, _L, d_head_qk), (_B, _H, _S, d_head_qk), (_B, _H, _S, d_head_v))
    ]
    block_mask = create_block_mask(
        lambda b, h, q_idx, kv_idx: q_idx >= kv_idx, _B, _H, _L, _S, device="cpu"
    )

    class FlexAttnModule(torch.nn.Module):
        def forward(self, q, k, v):
            return flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)

    captured = []

    def run_forward(gm, example_inputs):
        return make_boxed_func(gm.forward)

    def capture_backward(gm, example_inputs):
        captured.append(gm)
        raise _BackwardCapturedError

    def backend(gm, example_inputs):
        return aot_autograd(fw_compiler=run_forward, bw_compiler=capture_backward)(
            gm, example_inputs
        )

    torch._dynamo.reset()
    output = torch.compile(FlexAttnModule(), backend=backend)(*inputs)
    if isinstance(output, tuple):
        output = output[0]
    with pytest.raises(_BackwardCapturedError):
        output.sum().backward()

    assert len(captured) == 1
    return captured[0]


def _capture_flex_joint_graph():
    """Capture one GraphModule that holds both flex_attention directions."""
    from torch._functorch.aot_autograd import aot_export_module
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    inputs = [
        torch.randn(*shape, requires_grad=True)
        for shape in ((_B, _H, _L, _E), (_B, _H, _S, _E), (_B, _H, _S, _E))
    ]
    block_mask = create_block_mask(
        lambda b, h, q_idx, kv_idx: q_idx >= kv_idx, _B, _H, _L, _S, device="cpu"
    )

    class FlexAttnLossModule(torch.nn.Module):
        def forward(self, q, k, v):
            return (flex_attention(q, k, v, block_mask=block_mask)[0].sum(),)

    with FakeTensorMode(allow_non_fake_inputs=True):
        gm, _ = aot_export_module(
            FlexAttnLossModule(), tuple(inputs), trace_joint=True, output_loss_index=0
        )
    return gm


def _count_flex_nodes(gm: fx.GraphModule, name: str) -> int:
    """Count the flex_attention or flex_attention_backward operators in a graph."""
    return sum(
        1
        for node in gm.graph.nodes
        if node.op == "call_function" and getattr(node.target, "__name__", "") == name
    )


def _run_pass(gm, tile_info=None):
    """Run the pass with a fixed tile classification and a recording rewrite.

    tile_info=None makes _maybe_build_tile_info report that it has no tile types.
    Returns (PassResult, forward rewrite mock, backward rewrite mock). The rewrites leave
    the graph alone, so the operators survive whether or not the budget fires. The mock
    call counts tell the two cases apart.
    """
    build_result = (False, None) if tile_info is None else (True, tile_info)
    flex_pass = FlexAttentionNKILowering()
    with (
        patch.object(FlexAttentionNKILowering, "_maybe_build_tile_info", return_value=build_result),
        patch.object(
            FlexAttentionNKILowering, "_legalize_flex_forward", return_value=True
        ) as fwd_rewrite,
        patch.object(
            FlexAttentionNKILowering, "_legalize_flex_backward", return_value=True
        ) as bwd_rewrite,
        patch.object(FlexAttentionNKILowering, "_finalize_legalization"),
    ):
        result = flex_pass(gm)
    return result, fwd_rewrite, bwd_rewrite


# One live tile, batch 1 and 2 query heads give 2 cost units, that is 0.015625 s.
_ONE_TILE_INFO = {"qk_tile_types": [[_TILE_PARTIAL]]}
_ONE_TILE_SECONDS = 2 / _COST_UNITS_PER_SECOND

# 25600 live tiles give 400 s, which is above the default budget.
_OVER_DEFAULT_TILE_INFO = {"qk_tile_types": _full_grid(128, 200)}
_OVER_DEFAULT_SECONDS = 2 * 128 * 200 / _COST_UNITS_PER_SECOND


class TestCompileBudget:
    def test_the_over_default_grid_is_above_the_default_budget(self):
        assert _ONE_TILE_SECONDS < _DEFAULT_COMPILE_BUDGET_S < _OVER_DEFAULT_SECONDS

    def test_unset_budget_keeps_an_estimate_below_the_default(self):
        gm = _capture_flex_graph()
        assert _count_flex_nodes(gm, "flex_attention") == 1
        result, rewrite, _ = _run_pass(gm, _ONE_TILE_INFO)
        assert result.modified is True
        assert rewrite.call_count == 1

    def test_unset_budget_bails_on_an_estimate_above_the_default(self):
        gm = _capture_flex_graph()
        result, rewrite, _ = _run_pass(gm, _OVER_DEFAULT_TILE_INFO)
        assert result.modified is False
        assert rewrite.call_count == 0
        assert _count_flex_nodes(result.graph_module, "flex_attention") == 1

    def test_budget_above_the_estimate_keeps_the_nki_lowering(self, monkeypatch):
        monkeypatch.setenv(_BUDGET_ENV, str(_ONE_TILE_SECONDS * 10))
        gm = _capture_flex_graph()
        result, rewrite, _ = _run_pass(gm, _ONE_TILE_INFO)
        assert result.modified is True
        assert rewrite.call_count == 1

    def test_budget_below_the_estimate_bails(self, monkeypatch):
        monkeypatch.setenv(_BUDGET_ENV, str(_ONE_TILE_SECONDS / 10))
        gm = _capture_flex_graph()
        result, rewrite, _ = _run_pass(gm, _ONE_TILE_INFO)
        assert result.modified is False
        assert rewrite.call_count == 0
        assert _count_flex_nodes(result.graph_module, "flex_attention") == 1

    def test_budget_above_the_default_keeps_an_estimate_the_default_would_skip(self, monkeypatch):
        monkeypatch.setenv(_BUDGET_ENV, str(_OVER_DEFAULT_SECONDS * 2))
        gm = _capture_flex_graph()
        result, rewrite, _ = _run_pass(gm, _OVER_DEFAULT_TILE_INFO)
        assert result.modified is True
        assert rewrite.call_count == 1

    def test_malformed_budget_applies_the_default(self, monkeypatch, caplog):
        monkeypatch.setenv(_BUDGET_ENV, "not-a-number")
        gm = _capture_flex_graph()
        with caplog.at_level(logging.WARNING, logger=_LOWERING_LOGGER):
            result, rewrite, _ = _run_pass(gm, _OVER_DEFAULT_TILE_INFO)
        assert result.modified is False
        assert rewrite.call_count == 0
        assert any(record.levelno == logging.WARNING for record in caplog.records)

    def test_zero_budget_keeps_the_nki_lowering(self, monkeypatch):
        monkeypatch.setenv(_BUDGET_ENV, "0")
        gm = _capture_flex_graph()
        result, rewrite, _ = _run_pass(gm, _OVER_DEFAULT_TILE_INFO)
        assert result.modified is True
        assert rewrite.call_count == 1

    def test_negative_budget_keeps_the_nki_lowering(self, monkeypatch):
        """A negative budget is off, the same as 0."""
        monkeypatch.setenv(_BUDGET_ENV, "-1")
        gm = _capture_flex_graph()
        result, rewrite, _ = _run_pass(gm, _OVER_DEFAULT_TILE_INFO)
        assert result.modified is True
        assert rewrite.call_count == 1

    def test_score_mod_below_the_default_keeps_the_nki_lowering(self):
        gm = _capture_flex_graph(score_mod=lambda score, b, h, q_idx, kv_idx: score * 2)
        result, rewrite, _ = _run_pass(gm, _ONE_TILE_INFO)
        assert result.modified is True
        assert rewrite.call_count == 1

    def test_score_mod_multiplies_the_estimate_in_the_pass(self, monkeypatch):
        """A budget between the two estimates bails only with a score modification."""
        budget = _ONE_TILE_SECONDS * 2
        monkeypatch.setenv(_BUDGET_ENV, str(budget))
        plain = _run_pass(_capture_flex_graph(), _ONE_TILE_INFO)[0]
        assert plain.modified is True
        with_score_mod = _run_pass(
            _capture_flex_graph(score_mod=lambda score, b, h, q_idx, kv_idx: score * 2),
            _ONE_TILE_INFO,
        )[0]
        assert with_score_mod.modified is False


class TestCompileBudgetWithoutTileInfo:
    def test_budget_does_not_bail(self, monkeypatch):
        monkeypatch.setenv(_BUDGET_ENV, "0.0001")
        gm = _capture_flex_graph()
        result, rewrite, _ = _run_pass(gm, tile_info=None)
        assert result.modified is True
        assert rewrite.call_count == 1


class TestCompileBudgetOnTheBackward:
    """AOTAutograd compiles the backward as its own GraphModule, with no forward node."""

    def test_unset_budget_keeps_an_estimate_below_the_default(self):
        gm = _capture_flex_backward_graph()
        assert _count_flex_nodes(gm, "flex_attention_backward") == 1
        assert _count_flex_nodes(gm, "flex_attention") == 0
        result, fwd_rewrite, bwd_rewrite = _run_pass(gm, _ONE_TILE_INFO)
        assert result.modified is True
        assert fwd_rewrite.call_count == 0
        assert bwd_rewrite.call_count == 1

    def test_unset_budget_bails_on_an_estimate_above_the_default(self):
        gm = _capture_flex_backward_graph()
        result, _, bwd_rewrite = _run_pass(gm, _OVER_DEFAULT_TILE_INFO)
        assert result.modified is False
        assert bwd_rewrite.call_count == 0
        assert _count_flex_nodes(result.graph_module, "flex_attention_backward") == 1

    def test_zero_budget_keeps_the_nki_lowering(self, monkeypatch):
        monkeypatch.setenv(_BUDGET_ENV, "0")
        gm = _capture_flex_backward_graph()
        result, _, bwd_rewrite = _run_pass(gm, _OVER_DEFAULT_TILE_INFO)
        assert result.modified is True
        assert bwd_rewrite.call_count == 1

    def test_budget_above_the_estimate_keeps_the_nki_lowering(self, monkeypatch):
        monkeypatch.setenv(_BUDGET_ENV, str(_ONE_TILE_SECONDS * 10))
        gm = _capture_flex_backward_graph()
        result, _, bwd_rewrite = _run_pass(gm, _ONE_TILE_INFO)
        assert result.modified is True
        assert bwd_rewrite.call_count == 1

    def test_budget_below_the_estimate_bails(self, monkeypatch):
        monkeypatch.setenv(_BUDGET_ENV, str(_ONE_TILE_SECONDS / 10))
        gm = _capture_flex_backward_graph()
        result, _, bwd_rewrite = _run_pass(gm, _ONE_TILE_INFO)
        assert result.modified is False
        assert bwd_rewrite.call_count == 0
        assert _count_flex_nodes(result.graph_module, "flex_attention_backward") == 1

    def test_budget_does_not_apply_without_tile_info(self, monkeypatch):
        monkeypatch.setenv(_BUDGET_ENV, "0.0001")
        gm = _capture_flex_backward_graph()
        result, _, bwd_rewrite = _run_pass(gm, tile_info=None)
        assert result.modified is True
        assert bwd_rewrite.call_count == 1

    def test_score_mod_multiplies_the_estimate(self, monkeypatch):
        """A budget between the two estimates bails only with a score modification."""
        monkeypatch.setenv(_BUDGET_ENV, str(_ONE_TILE_SECONDS * 2))
        plain = _run_pass(_capture_flex_backward_graph(), _ONE_TILE_INFO)[0]
        assert plain.modified is True
        with_score_mod = _run_pass(
            _capture_flex_backward_graph(score_mod=lambda score, b, h, q_idx, kv_idx: score * 2),
            _ONE_TILE_INFO,
        )[0]
        assert with_score_mod.modified is False


class TestCompileBudgetKeepsAnUndecomposableOperator:
    """An estimate above the budget must not give an operator to a path that cannot
    decompose it.

    FlexAttentionLegalization gives the key and the value gradients the query batch size, so
    it cannot decompose a key that broadcasts over a larger query batch. The budget keeps the
    NKI lowering for that operator and accepts the slow compilation.
    """

    def test_an_undecomposable_shape_keeps_the_nki_lowering(self, caplog):
        gm = _capture_flex_graph(b_q=2, b_kv=1)
        with caplog.at_level(logging.INFO, logger=_LOWERING_LOGGER):
            result, rewrite, _ = _run_pass(gm, _OVER_DEFAULT_TILE_INFO)
        assert result.modified is True
        assert rewrite.call_count == 1
        assert any("cannot decompose" in record.getMessage() for record in caplog.records)

    def test_a_decomposable_shape_still_bails(self):
        gm = _capture_flex_graph(b_q=2, b_kv=2)
        result, rewrite, _ = _run_pass(gm, _OVER_DEFAULT_TILE_INFO)
        assert result.modified is False
        assert rewrite.call_count == 0
        assert _count_flex_nodes(result.graph_module, "flex_attention") == 1

    def test_an_undecomposable_shape_below_the_budget_needs_no_override(self, caplog):
        gm = _capture_flex_graph(b_q=2, b_kv=1)
        with caplog.at_level(logging.INFO, logger=_LOWERING_LOGGER):
            result, rewrite, _ = _run_pass(gm, _ONE_TILE_INFO)
        assert result.modified is True
        assert rewrite.call_count == 1
        assert not any("cannot decompose" in record.getMessage() for record in caplog.records)


class TestCompileBudgetOnBothDirections:
    """A joint graph holds a forward node and a backward node, so one budget covers both."""

    def test_unset_budget_keeps_both_lowerings_below_the_default(self):
        gm = _capture_flex_joint_graph()
        assert _count_flex_nodes(gm, "flex_attention") == 1
        assert _count_flex_nodes(gm, "flex_attention_backward") == 1
        result, fwd_rewrite, bwd_rewrite = _run_pass(gm, _ONE_TILE_INFO)
        assert result.modified is True
        assert fwd_rewrite.call_count == 1
        assert bwd_rewrite.call_count == 1

    def test_unset_budget_bails_on_both_above_the_default(self):
        gm = _capture_flex_joint_graph()
        result, fwd_rewrite, bwd_rewrite = _run_pass(gm, _OVER_DEFAULT_TILE_INFO)
        assert result.modified is False
        assert fwd_rewrite.call_count == 0
        assert bwd_rewrite.call_count == 0
        assert _count_flex_nodes(result.graph_module, "flex_attention") == 1
        assert _count_flex_nodes(result.graph_module, "flex_attention_backward") == 1

    def test_budget_below_the_estimate_bails_on_both(self, monkeypatch):
        monkeypatch.setenv(_BUDGET_ENV, str(_ONE_TILE_SECONDS / 10))
        gm = _capture_flex_joint_graph()
        result, fwd_rewrite, bwd_rewrite = _run_pass(gm, _ONE_TILE_INFO)
        assert result.modified is False
        assert fwd_rewrite.call_count == 0
        assert bwd_rewrite.call_count == 0
        assert _count_flex_nodes(result.graph_module, "flex_attention") == 1
        assert _count_flex_nodes(result.graph_module, "flex_attention_backward") == 1
