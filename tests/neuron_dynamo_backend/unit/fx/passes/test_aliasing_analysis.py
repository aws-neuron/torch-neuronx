"""
Unit tests for AliasingAnalysis fx pass.
"""

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
        """Test that subscript operation creates aliases."""

        class Model(torch.nn.Module):
            def forward(self, x):
                return x[::2]  # Output aliases input

        gm = torch.fx.symbolic_trace(Model())
        aliasing_info = run_aliasing_analysis(gm)
        assert_alias_count(aliasing_info.aliases, 1)
        alias = aliasing_info.aliases[0]
        assert_alias_exists(alias, param_num=0, param_idx=[], output_idx=0)

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
        """Test that slice operation creates aliases."""

        class Model(torch.nn.Module):
            def forward(self, x, y):
                sliced = x[2:8, :]  # Slice creates a view
                return sliced, y  # output[0] aliases input[0]

        model = Model()
        x = torch.zeros(10, 10)
        y = torch.zeros(10, 10)
        captured_graphs = get_aot_graphs(model, x, y)
        aliasing_info = run_aliasing_analysis(captured_graphs.pre_aot_graph)
        assert_alias_count(aliasing_info.aliases, 1)
        alias1 = aliasing_info.aliases[0]
        assert_alias_exists(alias1, param_num=0, param_idx=[], output_idx=0)

    def test_slicing_make_fx(self, run_aliasing_analysis):
        """Test that slice operation creates aliases when traced with fx.symbolic_trace."""

        class Model(torch.nn.Module):
            def forward(self, x, y):
                sliced = x[2:8, :]  # Slice creates a view
                return sliced, y  # output[0] aliases input[0]

        gm = torch.fx.symbolic_trace(Model())
        aliasing_info = run_aliasing_analysis(gm)
        assert_alias_count(aliasing_info.aliases, 2)
        alias1 = aliasing_info.aliases[0]
        alias2 = aliasing_info.aliases[1]
        assert_alias_exists(alias1, param_num=0, param_idx=[], output_idx=0)
        assert_alias_exists(alias2, param_num=1, param_idx=[], output_idx=1)

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
