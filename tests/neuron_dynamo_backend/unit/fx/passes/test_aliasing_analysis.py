"""
Unit tests for AliasingAnalysis fx pass.
"""

import logging

import pytest
import torch
import torch.nn as nn
from torch.fx import GraphModule
from torch.fx.passes.pass_manager import PassManager

from tests.neuron_dynamo_backend.unit.utils.test_utils import get_aot_graphs
from torch_neuronx.neuron_dynamo_backend.fx.passes.aliasing_analysis import AliasingAnalysis
from torch_neuronx.neuron_dynamo_backend.utils.alias_info import AliasInfo, AliasingInfo


@pytest.fixture
def run_aliasing_analysis():
    """Fixture providing a function to run AliasingAnalysis pass."""

    def _run(gm: GraphModule):
        aliasing_analysis = AliasingAnalysis()
        pm = PassManager(passes=[aliasing_analysis])
        pm(gm)
        return aliasing_analysis.result

    return _run


def assert_alias_count(aliasing_info: AliasingInfo, expected_count: int):
    """Assert the number of aliases matches expected count."""
    assert (
        len(aliasing_info) == expected_count
    ), f"Expected {expected_count} aliases, found {len(aliasing_info)}"


def assert_alias_exists(
    alias: AliasInfo, param_num: int, param_idx: list[int], output_idx: list[int]
):
    """Assert a specific alias exists in the list."""
    parameter_num = alias.parameter_number
    parameter_idx = alias.parameter_index
    out_idx = alias.output_index

    if parameter_num == param_num and parameter_idx == param_idx and out_idx == output_idx:
        return
    pytest.fail(
        f"Alias not found: input[{parameter_num}]{parameter_idx} -> output{out_idx}, "
        f"expected: input[{param_num}]{param_idx} -> output{output_idx}"
    )


class TestAliasDetection:
    """Unit tests for must-alias detection in FX graphs."""

    def test_view(self, run_aliasing_analysis):
        """Test that view operation creates aliases."""

        class Model(torch.nn.Module):
            def forward(self, x):
                return x.view(-1)  # Output aliases input

        gm = torch.fx.symbolic_trace(Model())
        aliasing_info = run_aliasing_analysis(gm)
        assert_alias_count(aliasing_info.aliases, 1)
        alias = aliasing_info.aliases[0]
        assert_alias_exists(alias, param_num=0, param_idx=[], output_idx=0)

    def test_transpose(self, run_aliasing_analysis):
        """Test that transpose operation creates aliases."""

        class Model(torch.nn.Module):
            def forward(self, x):
                return x.t()  # Output aliases input

        gm = torch.fx.symbolic_trace(Model())
        aliasing_info = run_aliasing_analysis(gm)
        assert_alias_count(aliasing_info.aliases, 1)
        alias = aliasing_info.aliases[0]
        assert_alias_exists(alias, param_num=0, param_idx=[], output_idx=0)

    def test_slice_view(self, run_aliasing_analysis):
        """Slice output changes shape — must not produce an alias.

        x[::2] has a different shape from x and cannot alias the full input
        buffer in HLO.
        """

        class Model(torch.nn.Module):
            def forward(self, x):
                return x[::2]

        gm = torch.fx.symbolic_trace(Model())
        aliasing_info = run_aliasing_analysis(gm)
        assert_alias_count(aliasing_info.aliases, 0)

    def test_transpose_permute(self, run_aliasing_analysis):
        """Test that transpose/permute operations create aliases."""

        class Model(torch.nn.Module):
            def forward(self, x):
                y = x.transpose(0, 1)  # View operation
                z = y.permute(2, 0, 1)  # Another view
                return z  # Aliases input x

        gm = torch.fx.symbolic_trace(Model())
        aliasing_info = run_aliasing_analysis(gm)
        assert_alias_count(aliasing_info.aliases, 1)
        alias = aliasing_info.aliases[0]
        assert_alias_exists(alias, param_num=0, param_idx=[], output_idx=0)

    def test_slicing(self, run_aliasing_analysis):
        """Slice output changes shape — no alias produced for either output.

        x[2:8, :] is shape-mismatched and excluded.  y is a pass-through but
        with no mutation in this graph AOTAutograd does not surface it as an
        alias output — AliasingInfo stays empty.
        """

        class Model(torch.nn.Module):
            def forward(self, x, y):
                sliced = x[2:8, :]
                return sliced, y

        model = Model()
        x = torch.zeros(10, 10)
        y = torch.zeros(10, 10)
        captured_graphs = get_aot_graphs(model, x, y)
        aliasing_info = run_aliasing_analysis(captured_graphs.pre_aot_graph)
        assert_alias_count(aliasing_info.aliases, 0)

    def test_slicing_make_fx(self, run_aliasing_analysis):
        """Slice changes shape — no alias. y pass-through aliased via symbolic_trace."""

        class Model(torch.nn.Module):
            def forward(self, x, y):
                sliced = x[2:8, :]
                return sliced, y

        gm = torch.fx.symbolic_trace(Model())
        aliasing_info = run_aliasing_analysis(gm)
        # symbolic_trace (not AOTAutograd) exposes y as a pass-through alias.
        # sliced (x[2:8,:]) is a slice → excluded; y → alias input[1].
        assert_alias_count(aliasing_info.aliases, 1)
        assert_alias_exists(aliasing_info.aliases[0], param_num=1, param_idx=[], output_idx=1)

    def test_partial(self, run_aliasing_analysis):
        """Test that mix of alias/non-alias."""

        class Model(torch.nn.Module):
            def forward(self, x, y):
                a = x.clone()  # No alias
                b = y.view(-1)  # Aliases y
                return a, b  # Mixed: one aliases, one doesn't

        model = Model()
        x = torch.zeros(10, 10)
        y = torch.zeros(10, 10)
        captured_graphs = get_aot_graphs(model, x, y)
        aliasing_info = run_aliasing_analysis(captured_graphs.pre_aot_graph)
        assert_alias_count(aliasing_info.aliases, 1)
        alias1 = aliasing_info.aliases[0]
        assert_alias_exists(alias1, param_num=1, param_idx=[], output_idx=1)

    def test_in_place_mutation1(self, run_aliasing_analysis):
        """Test in-place mutation."""

        class Model(torch.nn.Module):
            def forward(self, x, y):
                x.add_(y)  # In-place mutation
                return x  # Aliases input x

        model = Model()
        x = torch.zeros(10, 10)
        y = torch.zeros(10, 10)
        captured_graphs = get_aot_graphs(model, x, y)
        aliasing_info = run_aliasing_analysis(captured_graphs.pre_aot_graph)
        assert_alias_count(aliasing_info.aliases, 1)
        alias1 = aliasing_info.aliases[0]
        assert_alias_exists(alias1, param_num=0, param_idx=[], output_idx=0)

    def test_in_place_mutation2(self, run_aliasing_analysis):
        """Test in-place mutation following a view."""

        class Model(torch.nn.Module):
            def forward(self, x, y):
                z = x.view(-1)
                y_t = y.view(-1)
                z.add_(y_t)  # Mutates z, which aliases x
                return x  # Input was mutated via alias

        model = Model()
        x = torch.zeros(10, 10)
        y = torch.zeros(10, 10)
        captured_graphs = get_aot_graphs(model, x, y)
        aliasing_info = run_aliasing_analysis(captured_graphs.pre_aot_graph)
        assert_alias_count(aliasing_info.aliases, 1)
        alias1 = aliasing_info.aliases[0]
        assert_alias_exists(alias1, param_num=0, param_idx=[], output_idx=0)

    def test_in_place_mutation3(self, run_aliasing_analysis):
        """Test in-place mutation following a view and transpose."""

        class Model(torch.nn.Module):
            def forward(self, x):
                y = x.view(-1)
                z = x.t()
                return y, z  # Both outputs alias same input

        model = Model()
        x = torch.zeros(10, 10)
        captured_graphs = get_aot_graphs(model, x)
        aliasing_info = run_aliasing_analysis(captured_graphs.pre_aot_graph)
        assert_alias_count(aliasing_info.aliases, 2)
        alias1 = aliasing_info.aliases[0]
        alias2 = aliasing_info.aliases[1]
        assert_alias_exists(alias1, param_num=0, param_idx=[], output_idx=0)
        assert_alias_exists(alias2, param_num=0, param_idx=[], output_idx=1)

    def test_output_order(self, run_aliasing_analysis):
        """Test output ordering with KV cache pattern."""

        class SimpleKVCacheReplica(nn.Module):
            def __init__(self, hidden_size=64, max_seq_len=128):
                super().__init__()
                self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)

            def forward(self, x, k_cache, v_cache, cache_pos):
                k = self.k_proj(x)
                k_cache[0, cache_pos : cache_pos + 1] = k
                v_cache[0, cache_pos : cache_pos + 1] = k
                return x * k_cache.sum() + cache_pos, k_cache, v_cache

        batch, hidden, max_seq = 1, 64, 128
        model = SimpleKVCacheReplica(hidden, max_seq)
        model.eval()
        # Create example inputs
        x = torch.randn(batch, hidden)
        k_cache = torch.zeros(batch, max_seq, hidden)
        v_cache = torch.zeros(batch, max_seq, hidden)
        captured_gm = get_aot_graphs(model, x, k_cache, v_cache, 0)
        aliasing_info = run_aliasing_analysis(captured_gm.pre_aot_graph)
        assert_alias_count(aliasing_info.aliases, 2)
        alias1 = aliasing_info.aliases[0]
        alias2 = aliasing_info.aliases[1]
        assert_alias_exists(alias1, param_num=2, param_idx=[], output_idx=0)
        assert_alias_exists(alias2, param_num=3, param_idx=[], output_idx=1)

    def test_in_place_mutation4(self, run_aliasing_analysis):
        """Test in-place mutation following a view and transpose."""

        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.register_buffer(
                    "count", torch.zeros((10, 10), dtype=torch.float32, device=device)
                )

            def forward(self, x):
                self.count.add_(x)
                self.count.add_(5)
                self.count.sub_(2)
                return x + self.count

        x = torch.rand(
            (10, 10), dtype=torch.float32, device=f"neuron:{torch.neuron.current_device()}"
        )
        model = Model(device=f"neuron:{torch.neuron.current_device()}")
        captured_graphs = get_aot_graphs(model, x)
        aliasing_info = run_aliasing_analysis(captured_graphs.pre_aot_graph)
        assert_alias_count(aliasing_info.aliases, 1)
        alias1 = aliasing_info.aliases[0]
        assert_alias_exists(alias1, param_num=0, param_idx=[], output_idx=0)

    def test_inplace_scatter1(self, run_aliasing_analysis):
        class KVCache(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros((1, 1, 128, 64)))

            def forward(self, pos, new_kv):
                idx = pos.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(1, 1, 1, 64)
                self.cache.scatter_(2, idx, new_kv.unsqueeze(2))

        model = KVCache().to(f"neuron:{torch.neuron.current_device()}")

        new_kv = torch.randn((1, 1, 64)).to(f"neuron:{torch.neuron.current_device()}")
        pos1 = torch.tensor(124, device=f"neuron:{torch.neuron.current_device()}")

        captured_graphs = get_aot_graphs(model, pos1, new_kv)
        aliasing_info = run_aliasing_analysis(captured_graphs.pre_aot_graph)
        assert_alias_count(aliasing_info.aliases, 1)
        alias1 = aliasing_info.aliases[0]
        assert_alias_exists(alias1, param_num=1, param_idx=[], output_idx=0)

    def test_inplace_scatter2(self, run_aliasing_analysis):
        class KVCache(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("k_cache", torch.zeros((1, 1, 128, 64)))
                self.register_buffer("v_cache", torch.zeros((1, 1, 128, 64)))

            def forward(self, pos, new_k, new_v):
                idx = pos.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(1, 1, 1, 64)
                self.k_cache.scatter_(2, idx, new_k.unsqueeze(2))
                self.v_cache.scatter_(2, idx, new_v.unsqueeze(2))

        model = KVCache().to(f"neuron:{torch.neuron.current_device()}")

        new_k = torch.randn((1, 1, 64)).to(f"neuron:{torch.neuron.current_device()}")
        new_v = torch.randn((1, 1, 64)).to(f"neuron:{torch.neuron.current_device()}")

        pos1 = torch.tensor(124, device=f"neuron:{torch.neuron.current_device()}")
        captured_graphs = get_aot_graphs(model, pos1, new_k, new_v)
        aliasing_info = run_aliasing_analysis(captured_graphs.pre_aot_graph)
        assert_alias_count(aliasing_info.aliases, 2)
        alias1 = aliasing_info.aliases[0]
        alias2 = aliasing_info.aliases[1]
        assert_alias_exists(alias1, param_num=1, param_idx=[], output_idx=0)
        assert_alias_exists(alias2, param_num=3, param_idx=[], output_idx=1)


class TestCustomOpAliasing:
    """Unit tests for custom op with mutable arguments detection."""

    def test_is_custom_op_with_mutations_detection(self):
        """Test that _is_custom_op_with_mutations correctly identifies custom ops."""
        import torch.fx as fx

        # Create a custom op with mutable args
        @torch.library.custom_op("test::mutating_op", mutates_args={"y"})
        def mutating_op(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        @mutating_op.register_fake
        def _(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        graph = fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        op_node = graph.call_function(torch.ops.test.mutating_op.default, args=(x, y))
        graph.output(op_node)

        gm = fx.GraphModule(torch.nn.Module(), graph)

        aliasing_analysis = AliasingAnalysis()

        for node in gm.graph.nodes:
            if node.op == "call_function" and "mutating_op" in str(node.target):
                assert aliasing_analysis._is_custom_op_with_mutations(
                    node
                ), "Should detect custom op with mutations"

    def test_custom_op_aliasing(self, run_aliasing_analysis):
        """Test custom op with mutable argument produces correct aliasing."""
        import torch.fx as fx

        # Create a custom op with mutable args
        @torch.library.custom_op("test::kv_update", mutates_args={"k_cache", "v_cache"})
        def kv_update(
            x: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor
        ) -> torch.Tensor:
            return x

        @kv_update.register_fake
        def _(x: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        graph = fx.Graph()
        x = graph.placeholder("x")
        k_cache = graph.placeholder("k_cache")
        v_cache = graph.placeholder("v_cache")
        op_node = graph.call_function(torch.ops.test.kv_update.default, args=(x, k_cache, v_cache))
        graph.output((op_node,))

        gm = fx.GraphModule(torch.nn.Module(), graph)

        aliasing_info = run_aliasing_analysis(gm)

        # Should detect k_cache (input[1]) and v_cache (input[2]) as mutated
        assert_alias_count(aliasing_info.aliases, 2)
        alias1 = aliasing_info.aliases[0]
        alias2 = aliasing_info.aliases[1]
        assert_alias_exists(alias1, param_num=1, param_idx=[], output_idx=0)
        assert_alias_exists(alias2, param_num=2, param_idx=[], output_idx=1)

    def test_custom_op_with_view_chain(self, run_aliasing_analysis):
        """Test custom op where mutated input comes through a view chain."""
        import torch.fx as fx

        # Create a custom op with mutable args
        @torch.library.custom_op("test::cache_update", mutates_args={"cache"})
        def cache_update(x: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
            return x

        @cache_update.register_fake
        def _(x: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        graph = fx.Graph()
        cache_orig = graph.placeholder("cache_orig")
        x = graph.placeholder("x")

        # View of the original cache
        cache_view = graph.call_method("view", args=(cache_orig, -1))

        # Custom op takes the VIEW as input, should trace back to original
        op_node = graph.call_function(torch.ops.test.cache_update.default, args=(x, cache_view))
        graph.output((op_node,))

        gm = fx.GraphModule(torch.nn.Module(), graph)

        aliasing_info = run_aliasing_analysis(gm)

        # Should trace back through view to detect cache_orig (input[0]) as mutated
        assert_alias_count(aliasing_info.aliases, 1)
        alias = aliasing_info.aliases[0]
        assert_alias_exists(alias, param_num=0, param_idx=[], output_idx=0)

    def test_custom_op_no_mutations(self, run_aliasing_analysis):
        """Test custom op without mutable arguments produces no aliasing."""
        import torch.fx as fx

        # Create a custom op WITHOUT mutable args
        @torch.library.custom_op("test::pure_op", mutates_args={})
        def pure_op(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        @pure_op.register_fake
        def _(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        graph = fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        op_node = graph.call_function(torch.ops.test.pure_op.default, args=(x, y))
        graph.output((op_node,))

        gm = fx.GraphModule(torch.nn.Module(), graph)

        aliasing_info = run_aliasing_analysis(gm)

        # Should have no aliases since no args are mutated
        assert_alias_count(aliasing_info.aliases, 0)

    def test_custom_op_kwargs_mutation(self, run_aliasing_analysis):
        """Test custom op with mutable argument passed via kwargs."""
        import torch.fx as fx

        @torch.library.custom_op("test::kwargs_update", mutates_args={"cache"})
        def kwargs_update(x: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
            return x

        @kwargs_update.register_fake
        def _(x: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        graph = fx.Graph()
        x = graph.placeholder("x")
        cache = graph.placeholder("cache")
        op_node = graph.call_function(
            torch.ops.test.kwargs_update.default, args=(x,), kwargs={"cache": cache}
        )
        graph.output((op_node,))

        gm = fx.GraphModule(torch.nn.Module(), graph)

        aliasing_info = run_aliasing_analysis(gm)

        # Should detect cache (input[1]) as mutated via kwargs
        assert_alias_count(aliasing_info.aliases, 1)
        alias = aliasing_info.aliases[0]
        assert_alias_exists(alias, param_num=1, param_idx=[], output_idx=0)

    def test_custom_op_mixed_positional_and_kwargs_mutation(self, run_aliasing_analysis):
        """Test custom op with mutations in both positional args and kwargs."""
        import torch.fx as fx

        @torch.library.custom_op("test::mixed_update", mutates_args={"a", "c"})
        def mixed_update(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
            return b

        @mixed_update.register_fake
        def _(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(b)

        graph = fx.Graph()
        t_a = graph.placeholder("a")
        t_b = graph.placeholder("b")
        t_c = graph.placeholder("c")
        # a is positional, c is kwarg
        op_node = graph.call_function(
            torch.ops.test.mixed_update.default, args=(t_a, t_b), kwargs={"c": t_c}
        )
        graph.output((op_node,))

        gm = fx.GraphModule(torch.nn.Module(), graph)

        aliasing_info = run_aliasing_analysis(gm)

        # Should detect both a (input[0]) and c (input[2]) as mutated
        assert_alias_count(aliasing_info.aliases, 2)

    def test_custom_op_missing_kwarg(self, run_aliasing_analysis):
        """Test custom op when a mutable kwarg is not provided (uses default or is optional)."""
        import torch.fx as fx

        @torch.library.custom_op("test::optional_cache", mutates_args={"cache"})
        def optional_cache(x: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
            return x

        @optional_cache.register_fake
        def _(x: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        graph = fx.Graph()
        x = graph.placeholder("x")
        # Only pass x, omit cache entirely - simulates missing kwarg scenario
        op_node = graph.call_function(torch.ops.test.optional_cache.default, args=(x,), kwargs={})
        graph.output((op_node,))

        gm = fx.GraphModule(torch.nn.Module(), graph)

        aliasing_info = run_aliasing_analysis(gm)

        # Should handle gracefully - no crash, no alias detected for missing arg
        assert_alias_count(aliasing_info.aliases, 0)


class TestInplaceMethods:
    """Tests for AliasingAnalysis._is_inplace_method (endswith('_') heuristic).

    All inplace ops end with '_', so the endswith('_') check is the sole
    detection gate. These tests confirm that various inplace ops are
    correctly identified as mutations.
    """

    def test_pow_underscore_detected(self, run_aliasing_analysis):
        """pow_ ends with '_' and must be detected as a mutation."""

        class Model(nn.Module):
            def forward(self, x, exp):
                x.pow_(exp)
                return x

        model = Model()
        x = torch.ones(4)
        exp = torch.tensor(2.0)
        captured = get_aot_graphs(model, x, exp)
        aliasing_info = run_aliasing_analysis(captured.pre_aot_graph)
        assert any(
            a.parameter_number == 0 for a in aliasing_info.aliases
        ), "pow_ must be detected as a mutation of input[0]"

    def test_abs_underscore_detected(self, run_aliasing_analysis):
        """abs_ ends with '_' and must be detected as a mutation."""

        class Model(nn.Module):
            def forward(self, x):
                x.abs_()
                return x

        model = Model()
        x = torch.tensor([-1.0, 2.0, -3.0])
        captured = get_aot_graphs(model, x)
        aliasing_info = run_aliasing_analysis(captured.pre_aot_graph)
        assert any(
            a.parameter_number == 0 for a in aliasing_info.aliases
        ), "abs_ must be detected as a mutation of input[0]"

    def test_arbitrary_custom_trailing_underscore_detected(self, run_aliasing_analysis):
        """Any op ending with '_' is detected via the endswith heuristic."""

        class Model(nn.Module):
            def forward(self, x, y):
                # clamp_max_ ends with '_' and must be detected
                x.clamp_max_(y)
                return x

        model = Model()
        x = torch.tensor([1.0, 5.0, 10.0])
        y = torch.tensor(4.0)
        captured = get_aot_graphs(model, x, y)
        aliasing_info = run_aliasing_analysis(captured.pre_aot_graph)
        assert any(
            a.parameter_number == 0 for a in aliasing_info.aliases
        ), "clamp_max_ (ends with '_') must be detected"


class TestMissingViewOps:
    """Regression tests for movedim/swapaxes/swapdims/broadcast_to alias detection.

    These ops create views of the same storage and are now in ALIASING_METHODS.
    Mutations reaching an input through these views must be detected.
    """

    def test_mutation_through_movedim_detected(self, run_aliasing_analysis):
        """add_ on a movedim view must trace back to the original placeholder."""

        class Model(nn.Module):
            def forward(self, x, y):
                v = x.movedim(0, -1)  # view of x
                v.add_(y)
                return x

        model = Model()
        x = torch.zeros(4, 8)
        y = torch.zeros(8, 4)
        captured = get_aot_graphs(model, x, y)
        aliasing_info = run_aliasing_analysis(captured.pre_aot_graph)

        assert any(
            a.parameter_number == 0 for a in aliasing_info.aliases
        ), "x (input[0]) mutated via movedim view must appear in AliasingInfo"

    def test_mutation_through_swapaxes_detected(self, run_aliasing_analysis):
        """add_ on a swapaxes view must trace back to the original placeholder."""

        class Model(nn.Module):
            def forward(self, x, y):
                v = x.swapaxes(0, 1)  # view of x — same storage as transpose
                v.add_(y)
                return x

        model = Model()
        x = torch.zeros(4, 8)
        y = torch.zeros(8, 4)
        captured = get_aot_graphs(model, x, y)
        aliasing_info = run_aliasing_analysis(captured.pre_aot_graph)

        assert any(
            a.parameter_number == 0 for a in aliasing_info.aliases
        ), "x (input[0]) mutated via swapaxes view must appear in AliasingInfo"

    def test_mutation_through_broadcast_to_detected(self, run_aliasing_analysis):
        """add_ on a broadcast_to view must trace back to the original placeholder."""

        class Model(nn.Module):
            def forward(self, x, y):
                v = x.broadcast_to(4, 8)  # view of x — same storage as expand
                v.add_(y)
                return x

        model = Model()
        x = torch.zeros(1, 8)
        y = torch.zeros(4, 8)
        captured = get_aot_graphs(model, x, y)
        aliasing_info = run_aliasing_analysis(captured.pre_aot_graph)

        assert any(
            a.parameter_number == 0 for a in aliasing_info.aliases
        ), "x (input[0]) mutated via broadcast_to view must appear in AliasingInfo"

    def test_mutation_through_transpose_detected(self, run_aliasing_analysis):
        """Sanity: add_ through transpose (covered op) must still be detected."""

        class Model(nn.Module):
            def forward(self, x, y):
                v = x.transpose(0, 1)
                v.add_(y)
                return x

        model = Model()
        x = torch.zeros(4, 8)
        y = torch.zeros(8, 4)
        captured = get_aot_graphs(model, x, y)
        aliasing_info = run_aliasing_analysis(captured.pre_aot_graph)

        assert any(
            a.parameter_number == 0 for a in aliasing_info.aliases
        ), "x (input[0]) mutated via transpose (covered op) must be detected"


class TestNKIMutationDetection:
    """Tests for AliasingAnalysis._is_nki_call_with_mutations / _get_nki_kernel_mutated_inputs.

    NKI HOPs declare mutations via node.kwargs["operand_output_aliases"],
    not via schema is_write=True.  AliasingAnalysis must inspect those
    kwargs rather than target._schema so that NKI mutations are recorded in
    AliasingInfo; otherwise KV cache writes via NKI kernels would not persist
    across calls (computation ordering is handled by py_functionalize_impl;
    only persistence depends on this detection).
    """

    @staticmethod
    def _make_nki_node(
        graph: torch.fx.Graph,
        operand: torch.fx.Node,
        operand_output_aliases: dict,
    ) -> torch.fx.Node:
        """Build a minimal NKI HOP call_function node in-graph."""
        from torch_neuronx.nki_hop import nki_kernel_wrapper

        node = graph.call_function(
            nki_kernel_wrapper,
            args=(),
            kwargs={
                "kernel_idx": 0,
                "grid": (1,),
                "backend_config": "",
                "operand_output_aliases": operand_output_aliases,
                "args": (operand,),
                "arg_names": ["cache"],
                "constant_args_key": -1,
            },
        )
        node.meta["example_value"] = torch.empty(4, 8, dtype=torch.bfloat16)
        return node

    def test_nki_single_write_alias_detected(self, run_aliasing_analysis):
        """Single NKI HOP writing a placeholder must appear in AliasingInfo."""
        graph = torch.fx.Graph()

        cache = graph.placeholder("L_self_buffers_cache_")
        cache.meta["example_value"] = torch.empty(4, 8, dtype=torch.bfloat16)

        kernel = self._make_nki_node(graph, cache, operand_output_aliases={0: 0})
        graph.output((kernel,))

        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        aliasing_info = run_aliasing_analysis(gm)

        # cache (input[0]) must be recorded as mutated
        assert any(
            a.parameter_number == 0 for a in aliasing_info.aliases
        ), "NKI HOP mutation of cache (input[0]) must appear in AliasingInfo"

    def test_nki_dual_write_alias_detected(self, run_aliasing_analysis):
        """Two NKI HOPs writing separate placeholders — both must be detected."""
        graph = torch.fx.Graph()

        k_cache = graph.placeholder("L_self_buffers_k_cache_")
        k_cache.meta["example_value"] = torch.empty(4, 8, dtype=torch.bfloat16)
        v_cache = graph.placeholder("L_self_buffers_v_cache_")
        v_cache.meta["example_value"] = torch.empty(4, 8, dtype=torch.bfloat16)

        k_kernel = self._make_nki_node(graph, k_cache, operand_output_aliases={0: 0})
        v_kernel = self._make_nki_node(graph, v_cache, operand_output_aliases={0: 0})
        graph.output((k_kernel, v_kernel))

        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        aliasing_info = run_aliasing_analysis(gm)

        mutated_inputs = {a.parameter_number for a in aliasing_info.aliases}
        assert 0 in mutated_inputs, "k_cache (input[0]) mutation not detected"
        assert 1 in mutated_inputs, "v_cache (input[1]) mutation not detected"

    def test_nki_mutation_through_view_chain_detected(self, run_aliasing_analysis):
        """NKI HOP operating on a squeezed view must trace back to the original placeholder."""
        graph = torch.fx.Graph()

        # 4D cache placeholder
        cache = graph.placeholder("L_self_buffers_cache_")
        cache.meta["example_value"] = torch.empty(4, 1, 8, 16, dtype=torch.bfloat16)

        # Squeeze to 3D before NKI (common KV-cache pattern)
        squeezed = graph.call_method("squeeze", args=(cache, 1))
        squeezed.meta["example_value"] = torch.empty(4, 8, 16, dtype=torch.bfloat16)

        kernel = self._make_nki_node(graph, squeezed, operand_output_aliases={0: 0})
        graph.output((kernel,))

        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        aliasing_info = run_aliasing_analysis(gm)

        # Should trace through squeeze back to cache (input[0])
        assert any(
            a.parameter_number == 0 for a in aliasing_info.aliases
        ), "NKI HOP mutation through squeeze must trace back to cache (input[0])"

    def test_pure_nki_kernel_no_aliases(self, run_aliasing_analysis):
        """NKI HOP with empty operand_output_aliases must not produce any alias."""
        graph = torch.fx.Graph()

        x = graph.placeholder("L_x_")
        x.meta["example_value"] = torch.empty(4, 8, dtype=torch.bfloat16)

        kernel = self._make_nki_node(graph, x, operand_output_aliases={})
        graph.output((kernel,))

        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        aliasing_info = run_aliasing_analysis(gm)

        assert_alias_count(aliasing_info.aliases, 0)

    def test_nki_alias_with_interleaved_scalar_args(self, run_aliasing_analysis):
        """operand_output_aliases keys index the TENSOR-ONLY operand list.

        Uses the real decode signature: operand_output_aliases = {6: 1, 7: 2}.
        The decode attention kernel aliases tensor operands 6 and 7 (K, V cache).
        nki_hop may populate kwargs["args"] with non-tensor constants interleaved
        (the raw arg tuple, on cached dispatches), so the two KV tensors sit at
        raw positions 9 and 10 but tensor positions 6 and 7. Indexing the raw
        tuple by 6/7 would land on the wrong (non-KV) operands and miss the KV
        mutation -- the OOM bug. Resolving against the tensor-filtered operands
        must recover exactly the two KV placeholders.

        Note operand_output_aliases *values* (1, 2) are the kernel's internal
        output ids; the pass renumbers post-AOT outputs from the sorted mutated
        input indices, so the aliases are (param=6, out=0) and (param=7, out=1).
        """
        from torch_neuronx.nki_hop import nki_kernel_wrapper

        graph = torch.fx.Graph()
        # 8 tensor placeholders; the last two are the K/V caches (operands 6, 7).
        tensors = []
        for i in range(6):
            t = graph.placeholder(f"t{i}")
            t.meta["example_value"] = torch.empty(4, 8, dtype=torch.bfloat16)
            tensors.append(t)
        k_cache = graph.placeholder("L_self_buffers_k_cache_")
        k_cache.meta["example_value"] = torch.empty(4, 8, dtype=torch.bfloat16)
        v_cache = graph.placeholder("L_self_buffers_v_cache_")
        v_cache.meta["example_value"] = torch.empty(4, 8, dtype=torch.bfloat16)

        kernel = graph.call_function(
            nki_kernel_wrapper,
            args=(),
            kwargs={
                "kernel_idx": 0,
                "grid": (1,),
                "backend_config": "",
                "operand_output_aliases": {6: 1, 7: 2},
                # scalars interleaved so raw index != tensor index; K/V are the
                # 7th and 8th *tensor* operands but sit later in the raw tuple.
                "args": (
                    1.0,
                    tensors[0],
                    tensors[1],
                    8,
                    tensors[2],
                    tensors[3],
                    tensors[4],
                    tensors[5],
                    0.5,
                    k_cache,
                    v_cache,
                ),
                "arg_names": [
                    "scale",
                    "t0",
                    "t1",
                    "block_size",
                    "t2",
                    "t3",
                    "t4",
                    "t5",
                    "softmax_scale",
                    "k_cache",
                    "v_cache",
                ],
                "constant_args_key": -1,
            },
        )
        kernel.meta["example_value"] = torch.empty(4, 8, dtype=torch.bfloat16)
        graph.output((kernel,))

        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        aliasing_info = run_aliasing_analysis(gm)

        # Exactly the two KV placeholders (indices 6, 7), renumbered to outputs
        # 0, 1 in mutated-input order.
        assert_alias_count(aliasing_info, 2)
        assert_alias_exists(aliasing_info.aliases[0], param_num=6, param_idx=[], output_idx=0)
        assert_alias_exists(aliasing_info.aliases[1], param_num=7, param_idx=[], output_idx=1)


class TestDunderMethodDetection:
    """Tests that dunder methods are not classified as in-place mutations.

    _is_inplace_method uses target.endswith("_"), which must guard against
    dunder methods (e.g. __iadd__, __isub__) so that they are NOT treated
    as in-place tensor mutations.
    """

    def test_iadd_not_treated_as_mutation(self, run_aliasing_analysis):
        """__iadd__ (+=) must not be classified as an in-place tensor mutation."""

        class Model(nn.Module):
            def forward(self, x, y):
                x.__iadd__(y)
                return x + y

        gm = torch.fx.symbolic_trace(Model())
        aliasing_info = run_aliasing_analysis(gm)
        assert_alias_count(aliasing_info.aliases, 0)

    def test_isub_not_treated_as_mutation(self, run_aliasing_analysis):
        """__isub__ (-=) must not be classified as an in-place tensor mutation."""

        class Model(nn.Module):
            def forward(self, x, y):
                x.__isub__(y)
                return x - y

        gm = torch.fx.symbolic_trace(Model())
        aliasing_info = run_aliasing_analysis(gm)
        assert_alias_count(aliasing_info.aliases, 0)

    def test_imul_not_treated_as_mutation(self, run_aliasing_analysis):
        """__imul__ (*=) must not be classified as an in-place tensor mutation."""

        class Model(nn.Module):
            def forward(self, x, y):
                x.__imul__(y)
                return x * y

        gm = torch.fx.symbolic_trace(Model())
        aliasing_info = run_aliasing_analysis(gm)
        assert_alias_count(aliasing_info.aliases, 0)

    def test_add_underscore_is_mutation(self, run_aliasing_analysis):
        """Sanity check: add_ (non-dunder) must still be detected as a mutation."""

        class Model(nn.Module):
            def forward(self, x, y):
                x.add_(y)
                return x

        model = Model()
        x = torch.zeros(4)
        y = torch.zeros(4)
        captured = get_aot_graphs(model, x, y)
        aliasing_info = run_aliasing_analysis(captured.pre_aot_graph)
        assert any(
            a.parameter_number == 0 for a in aliasing_info.aliases
        ), "add_ on input[0] should produce a mutation alias"


class TestInplaceOperatorDetection:
    """Tests for AliasingAnalysis._is_inplace_operator.

    Augmented assignment on a tensor (buf += x, buf -= x, ...) is traced by
    Dynamo as a call_function targeting operator.iadd / isub / imul / etc.,
    which mutates the buffer in place and must be detected.
    """

    def test_iadd_buffer_is_mutation(self, run_aliasing_analysis):
        """buf += x on a registered buffer must be detected as a mutation."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.zeros(4))

            def forward(self, x):
                self.buf += x
                return self.buf.sum()

        captured = get_aot_graphs(Model(), torch.ones(4))
        aliasing_info = run_aliasing_analysis(captured.pre_aot_graph)
        assert any(
            a.parameter_number == 0 for a in aliasing_info.aliases
        ), "buf += x must produce a mutation alias for buf (input[0])"

    def test_isub_buffer_is_mutation(self, run_aliasing_analysis):
        """buf -= x on a registered buffer must be detected as a mutation."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.zeros(4))

            def forward(self, x):
                self.buf -= x
                return self.buf.sum()

        captured = get_aot_graphs(Model(), torch.ones(4))
        aliasing_info = run_aliasing_analysis(captured.pre_aot_graph)
        assert any(
            a.parameter_number == 0 for a in aliasing_info.aliases
        ), "buf -= x must produce a mutation alias for buf (input[0])"

    def test_imul_buffer_is_mutation(self, run_aliasing_analysis):
        """buf *= x on a registered buffer must be detected as a mutation."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.ones(4))

            def forward(self, x):
                self.buf *= x
                return self.buf.sum()

        captured = get_aot_graphs(Model(), torch.ones(4) * 2)
        aliasing_info = run_aliasing_analysis(captured.pre_aot_graph)
        assert any(
            a.parameter_number == 0 for a in aliasing_info.aliases
        ), "buf *= x must produce a mutation alias for buf (input[0])"

    def test_inplace_operator_detection_unit(self):
        """_is_inplace_operator matches operator.iadd/isub but not operator.add."""
        import operator

        import torch.fx as fx

        p = AliasingAnalysis()
        g = fx.Graph()
        x = g.placeholder("x")
        y = g.placeholder("y")
        iadd = g.call_function(operator.iadd, args=(x, y))
        isub = g.call_function(operator.isub, args=(x, y))
        add = g.call_function(operator.add, args=(x, y))
        getitem = g.call_function(operator.getitem, args=(x, 0))
        assert p._is_inplace_operator(iadd)
        assert p._is_inplace_operator(isub)
        assert not p._is_inplace_operator(add)
        assert not p._is_inplace_operator(getitem)


class TestSliceOutputAliasing:
    """Tests slice-output handling (AliasingAnalysis._is_slice_operation).

    A slice of a mutated input is shape-mismatched with the input buffer.
    It must NOT produce an io_map entry that causes the executor to reuse
    the input buffer for the slice output — doing so would let a subsequent
    in-place op on the returned slice corrupt the original buffer.
    """

    def test_slice_of_mutated_input_not_aliased(self, run_aliasing_analysis):
        """Mutation via add_ followed by returning a slice of the buffer.

        The slice is shape-mismatched ([3] vs [5]) and must NOT be mapped
        to the input in AliasingInfo.  The mutated input itself should
        still appear as a mutation alias at the AOT-convention output index,
        but the returned slice must not produce a separate pass-through entry
        pointing at the same input slot.
        """

        class Model(nn.Module):
            def forward(self, buf, x):
                buf.add_(x)
                return buf[:3]  # shape [3], buf is shape [5]

        model = Model()
        buf = torch.zeros(5)
        x = torch.ones(5)
        captured = get_aot_graphs(model, buf, x)
        aliasing_info = run_aliasing_analysis(captured.pre_aot_graph)

        # The mutation of buf (input[0]) must be recorded.
        assert any(
            a.parameter_number == 0 for a in aliasing_info.aliases
        ), "buf (input[0]) mutation should be detected"

        # The slice output must NOT produce an alias to input[0] because
        # HLO requires matching shapes and the executor must not return
        # the raw input buffer for a shape-mismatched output.
        # There must be exactly ONE alias for input[0] (the mutation writeback),
        # not two (which would mean the slice is also aliased).
        input0_aliases = [a for a in aliasing_info.aliases if a.parameter_number == 0]
        assert len(input0_aliases) == 1, (
            f"Expected exactly one alias for input[0] (the mutation writeback), "
            f"got {len(input0_aliases)}: {input0_aliases}"
        )

    def test_passthrough_slice_not_aliased(self, run_aliasing_analysis):
        """Returning a slice of an un-mutated input must not produce an alias.

        x[:4] has a different shape from x ([4] vs [6]).  The pass-through
        alias detection must not map this slice to input[0].
        """

        class Model(nn.Module):
            def forward(self, x):
                return x[:4]  # shape [4], x is shape [6]

        model = Model()
        x = torch.zeros(6)
        captured = get_aot_graphs(model, x)
        aliasing_info = run_aliasing_analysis(captured.pre_aot_graph)

        # No mutation, no same-shape pass-through → no aliases expected.
        assert_alias_count(aliasing_info.aliases, 0)

    def test_full_slice_passthrough_is_not_aliased(self, run_aliasing_analysis):
        """Even a full-range slice x[:] must not produce an alias.

        getitem-slice nodes are excluded unconditionally: HLO cannot safely
        alias a slice output to the full input buffer even when shapes match,
        because the slice is a distinct object with its own strides.
        """

        class Model(nn.Module):
            def forward(self, x):
                return x[:]

        model = Model()
        x = torch.zeros(6)
        captured = get_aot_graphs(model, x)
        aliasing_info = run_aliasing_analysis(captured.pre_aot_graph)

        assert_alias_count(aliasing_info.aliases, 0)


def cross_entropy_loss(pred, labels):
    """Torchtitan loss function (torchtitan/components/loss.py)."""
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(),
        labels.flatten(0, 1),
        reduction="sum",
    )


class SimpleModel(nn.Module):
    def __init__(self, vocab_size=32, hidden_size=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        return self.head(self.embed(input_ids))


def test_compiled_loss_backward_aliasing_oob():
    device = torch.neuron.current_device()
    model = torch.compile(SimpleModel().to(device), backend="neuron")
    loss_fn = torch.compile(cross_entropy_loss, backend="neuron")

    input_ids = torch.randint(0, 32, (2, 4), device=device)
    labels = torch.randint(0, 32, (2, 4), dtype=torch.int32, device=device)

    # Production runs under torch.autocast (single-device AMP path)
    with torch.autocast("privateuseone", dtype=torch.bfloat16):
        pred = model(input_ids)
        loss = loss_fn(pred, labels)
    loss.backward()
