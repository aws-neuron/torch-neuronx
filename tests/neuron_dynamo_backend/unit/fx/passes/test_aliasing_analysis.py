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
