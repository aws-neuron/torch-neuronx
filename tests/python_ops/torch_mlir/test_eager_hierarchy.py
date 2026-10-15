"""
Unit tests for eager_hierarchy.py — hierarchy string capture from live call stacks.

These tests exercise the pure-Python logic (backward labels, forward stack walking,
fingerprint caching, counter reset) without requiring a Neuron device.
"""

import torch
import torch.nn as nn

import torch_neuronx.python_ops.torch_mlir.eager_hierarchy as _eh
from torch_neuronx.python_ops.torch_mlir.eager_hierarchy import (
    _backward_hierarchy_from_op,
    _invocation_counters,
    _stack_cache,
    get_hierarchy_from_stack,
    reset_backward_counters,
)


class TestBackwardHierarchy:
    """Tests for _backward_hierarchy_from_op."""

    def setup_method(self):
        reset_backward_counters()

    def test_empty_op_name(self):
        assert _backward_hierarchy_from_op("") == ""

    def test_strips_namespace(self):
        result = _backward_hierarchy_from_op("aten::linear_backward")
        assert result == "linear_backward[backward][1]/"

    def test_no_namespace(self):
        result = _backward_hierarchy_from_op("linear_backward")
        assert result == "linear_backward[backward][1]/"

    def test_counter_increments(self):
        _backward_hierarchy_from_op("aten::mm_backward")
        result = _backward_hierarchy_from_op("aten::mm_backward")
        assert result == "mm_backward[backward][2]/"

    def test_different_ops_independent_counters(self):
        _backward_hierarchy_from_op("aten::mm_backward")
        result = _backward_hierarchy_from_op("aten::add_backward")
        assert result == "add_backward[backward][1]/"


class TestResetBackwardCounters:
    """Tests for reset_backward_counters."""

    def setup_method(self):
        reset_backward_counters()

    def test_reset_clears_current_thread(self):
        _backward_hierarchy_from_op("aten::mm_backward")
        reset_backward_counters()
        result = _backward_hierarchy_from_op("aten::mm_backward")
        assert result == "mm_backward[backward][1]/"

    def test_reset_noop_when_empty(self):
        reset_backward_counters()  # should not raise


class TestForwardHierarchy:
    """Tests for get_hierarchy_from_stack with live nn.Module frames."""

    def setup_method(self):
        _invocation_counters.clear()
        _stack_cache.clear()
        _eh._current_top_level_id = -1
        reset_backward_counters()
        self.captured = []

    def _capture(self, **kwargs):
        """Call get_hierarchy_from_stack and store result in self.captured."""
        self.captured.append(get_hierarchy_from_stack(**kwargs))

    def test_simple_module_hierarchy(self):
        captured = self.captured

        class Inner(nn.Module):
            def forward(self, x):
                captured.append(get_hierarchy_from_stack())
                return x

        class Outer(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = Inner()

            def forward(self, x):
                return self.inner(x)

        Outer()(torch.tensor(1.0))
        assert len(captured) == 1
        assert "Inner[inner]" in captured[0]
        assert captured[0].endswith("/")

    def test_no_module_returns_empty(self):
        """Calling outside any nn.Module should return empty string."""
        result = get_hierarchy_from_stack()
        assert result == ""

    def test_backward_fallback_when_no_modules(self):
        """With op_name but no module frames, should produce backward label."""
        result = get_hierarchy_from_stack(op_name="aten::mm_backward")
        assert "mm_backward[backward]" in result

    def test_backward_results_not_cached(self):
        """Backward labels must not be cached — counters must increment."""
        r1 = get_hierarchy_from_stack(op_name="aten::mm_backward")
        r2 = get_hierarchy_from_stack(op_name="aten::mm_backward")
        assert r1 != r2
        assert "[1]" in r1
        assert "[2]" in r2

    def test_nested_modules(self):
        captured = self.captured

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)

            def forward(self, x):
                captured.append(get_hierarchy_from_stack())
                return x

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = Block()

            def forward(self, x):
                return self.block(x)

        Model()(torch.tensor(1.0))
        assert len(captured) == 1
        assert "Block[block]" in captured[0]

    def test_cache_returns_same_result(self):
        """Second call from same stack location should hit cache."""
        captured = self.captured

        class Inner(nn.Module):
            def forward(self, x):
                captured.append(get_hierarchy_from_stack())
                captured.append(get_hierarchy_from_stack())
                return x

        class Outer(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = Inner()

            def forward(self, x):
                return self.inner(x)

        Outer()(torch.tensor(1.0))
        assert len(captured) == 2
        assert captured[0] == captured[1]
        assert len(_stack_cache) == 1  # first call cached, second was a hit

    def test_distinct_children_get_correct_attr_names(self):
        """Two children of the same class produce distinct hierarchy strings."""
        captured = self.captured

        class Child(nn.Module):
            def forward(self, x):
                captured.append(get_hierarchy_from_stack())
                return x

        class Parent(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = Child()
                self.decoder = Child()

            def forward(self, x):
                x = self.encoder(x)
                return self.decoder(x)

        Parent()(torch.tensor(1.0))
        assert len(captured) == 2
        assert "Child" in captured[0]
        assert "Child" in captured[1]
        assert captured[0] != captured[1], (
            f"Expected distinct hierarchy strings for encoder vs decoder, "
            f"got: {captured[0]!r} and {captured[1]!r}"
        )

    def test_multiple_ops_same_forward_share_count(self):
        """Multiple get_hierarchy_from_stack calls within one forward() get the same [count]."""
        captured = self.captured

        class Multi(nn.Module):
            def forward(self, x):
                captured.append(get_hierarchy_from_stack())
                captured.append(get_hierarchy_from_stack())
                return x

        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.child = Multi()

            def forward(self, x):
                return self.child(x)

        Wrapper()(torch.tensor(1.0))
        assert len(captured) == 2
        assert captured[0] == captured[1]
        assert "[1]" in captured[0]

    def test_counters_reset_on_new_outermost_module(self):
        """Counters reset when a different outermost module instance is used."""
        captured = self.captured

        class Inner(nn.Module):
            def forward(self, x):
                captured.append(get_hierarchy_from_stack())
                return x

        class Outer(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = Inner()

            def forward(self, x):
                return self.inner(x)

        model1 = Outer()
        model2 = Outer()

        model1(torch.tensor(1.0))
        model2(torch.tensor(1.0))

        assert len(captured) == 2
        # Both should have [1] — second model resets counters
        assert "Inner[inner][1]" in captured[0]
        assert "Inner[inner][1]" in captured[1]
