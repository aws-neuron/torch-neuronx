"""Tests for aliasing analysis propagation to StableHLO."""

import pytest
import torch
from torch.fx import GraphModule
from torch.fx.passes.pass_manager import PassManager
from torch_mlir.ir import DenseI64ArrayAttr, IntegerAttr, Module

from tests.neuron_dynamo_backend.unit.utils.test_utils import CapturedGraphs, get_aot_graphs
from torch_neuronx.neuron_dynamo_backend import backend
from torch_neuronx.neuron_dynamo_backend.fx.passes.aliasing_analysis import AliasingAnalysis

# ============== Analysis Helpers ==============


def run_aliasing_analysis(gm: GraphModule):
    """Run AliasingAnalysis FX pass on a GraphModule."""
    analysis = AliasingAnalysis()
    PassManager(passes=[analysis])(gm)
    return analysis.result


def compile_to_stablehlo(captured: CapturedGraphs) -> Module:
    """Convert captured graphs to StableHLO MLIR with aliasing info."""
    pre_aot = captured.pre_aot_graph
    post_aot = captured.post_aot_forward_graph
    aliasing_info = run_aliasing_analysis(pre_aot)

    stablehlo_mlir, *_ = backend._compile_fx_to_stablehlo(
        post_aot,
        [],
        model_name="test_model",
        segment_id="segment_123",
        preserve_artifacts=False,
        aliasing_info=aliasing_info,
    )
    return stablehlo_mlir


def parse_mlir_aliases(module: Module) -> list[dict]:
    """Extract aliasing metadata from StableHLO module."""
    attrs = module.operation.attributes
    if "mhlo.input_output_alias" not in attrs:
        return []

    return [
        {
            "kind": str(entry["alias"]["kind"]).strip('"'),
            "parameter_number": IntegerAttr(entry["alias"]["parameter_number"]).value,
            "parameter_index": list(DenseI64ArrayAttr(entry["alias"]["parameter_index"])),
            "output_index": list(DenseI64ArrayAttr(entry["output_index"])),
        }
        for entry in attrs["mhlo.input_output_alias"]
    ]


# ============== Test Pipeline ==============


def get_aliases(model, *inputs) -> list[dict]:
    """End-to-end: compile model/function, generate StableHLO, extract aliases."""
    captured = get_aot_graphs(model, *inputs)
    stablehlo_ir = compile_to_stablehlo(captured)
    return parse_mlir_aliases(stablehlo_ir)


# ============== Assertion Helpers ==============


def assert_alias_count(aliases: list[dict], expected: int):
    """Verify exact number of aliases."""
    assert len(aliases) == expected, f"Expected {expected} aliases, got {len(aliases)}: {aliases}"


def assert_has_alias(
    aliases: list[dict], param_num: int, output_idx: list[int], kind: str = "must_alias"
):
    """Verify specific alias exists."""
    match = any(
        a["parameter_number"] == param_num and a["output_index"] == output_idx and a["kind"] == kind
        for a in aliases
    )
    if not match:
        pytest.fail(f"Alias not found: param {param_num} -> output {output_idx}\nFound: {aliases}")


# ============== Test Models ==============


class SimpleKVCacheReplica(torch.nn.Module):
    """Model with dual KV cache update pattern."""

    def __init__(self, hidden_size=64, max_seq_len=128):
        super().__init__()
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, k_cache, v_cache, cache_pos):
        k = self.k_proj(x)
        k_cache[0, cache_pos : cache_pos + 1] = k
        v_cache[0, cache_pos : cache_pos + 1] = k
        return x * k_cache.sum() + cache_pos, k_cache, v_cache


class ScatterKVCache(torch.nn.Module):
    """KV cache using scatter_ for in-place updates."""

    def __init__(self, hidden_size=64, max_seq_len=128):
        super().__init__()
        self.register_buffer("k_cache", torch.zeros((1, 1, max_seq_len, hidden_size)))
        self.register_buffer("v_cache", torch.zeros((1, 1, max_seq_len, hidden_size)))

    def forward(self, pos, new_k, new_v):
        idx = pos.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(1, 1, 1, 64)
        self.k_cache.scatter_(2, idx, new_k.unsqueeze(2))
        self.v_cache.scatter_(2, idx, new_v.unsqueeze(2))


class SingleScatterCache(torch.nn.Module):
    """Single cache buffer with scatter_ update."""

    def __init__(self, hidden_size=64, max_seq_len=128):
        super().__init__()
        self.register_buffer("cache", torch.zeros((1, 1, max_seq_len, hidden_size)))

    def forward(self, pos, new_kv):
        idx = pos.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(1, 1, 1, 64)
        self.cache.scatter_(2, idx, new_kv.unsqueeze(2))
        return self.cache


# ============== Tests ==============


class TestAliasingPropagation:
    """Verify aliasing info propagates correctly to StableHLO."""

    def test_reshaped(self):
        """
        Simple alias of input, but reshaped.
        This is not accepted by compiler, so the StableHLO should drop it.
        """

        def reshaped(x):
            return x.reshape(5, 20)

        x = torch.randn(10, 10)
        aliases = get_aliases(reshaped, x)
        assert_alias_count(aliases, 0)

    def test_transpose(self):
        """
        Transpose changes dimension order - shape mismatch.
        """

        def transposed(x):
            return x.transpose(0, 1)

        x = torch.randn(10, 20)
        aliases = get_aliases(transposed, x)
        assert_alias_count(aliases, 0)

    def test_double_transpose(self):
        """
        Transpose changes dimension order - shape mismatch.
        """

        def transposed(x):
            return x.t().t()

        x = torch.randn(10, 20)
        aliases = get_aliases(transposed, x)
        assert_alias_count(aliases, 1)

    def test_permute(self):
        """
        Permute reorders dimensions - shape mismatch.
        """

        def permuted(x):
            return x.permute(2, 0, 1)

        x = torch.randn(4, 8, 16)
        aliases = get_aliases(permuted, x)
        assert_alias_count(aliases, 0)

    def test_unsqueeze(self):
        """
        Unsqueeze adds a dimension - rank mismatch.
        """

        def unsqueezed(x):
            return x.unsqueeze(0)

        x = torch.randn(10, 10)
        aliases = get_aliases(unsqueezed, x)
        assert_alias_count(aliases, 0)

    def test_squeeze(self):
        """
        Squeeze removes a dimension - rank mismatch.
        """

        def squeezed(x):
            return x.squeeze(0)

        x = torch.randn(1, 10, 10)
        aliases = get_aliases(squeezed, x)
        assert_alias_count(aliases, 0)

    def test_flatten(self):
        """
        Flatten reduces to fewer dimensions - shape mismatch.
        """

        def flattened(x):
            return x.flatten()

        x = torch.randn(4, 5, 6)
        aliases = get_aliases(flattened, x)
        assert_alias_count(aliases, 0)

    def test_flatten_partial(self):
        """
        Partial flatten changes shape - shape mismatch.
        """

        def flattened_partial(x):
            return x.flatten(start_dim=1)

        x = torch.randn(4, 5, 6)
        aliases = get_aliases(flattened_partial, x)
        assert_alias_count(aliases, 0)

    def test_slice(self):
        """
        Slicing returns subset - shape mismatch.
        """

        def sliced(x):
            return x[:5, :]

        x = torch.randn(10, 10)
        aliases = get_aliases(sliced, x)
        assert_alias_count(aliases, 0)

    def test_index_select(self):
        """
        Index selection returns subset - shape mismatch.
        """

        class IndexSelected(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("indices", torch.tensor([0, 2, 4]))

            def forward(self, x):
                return torch.index_select(x, 0, self.indices)

        model = IndexSelected()
        x = torch.randn(10, 10)
        aliases = get_aliases(model, x)
        assert_alias_count(aliases, 0)

    def test_expand(self):
        """
        Expand broadcasts tensor - shape mismatch.
        """

        def expanded(x):
            return x.expand(4, 10, 10)

        x = torch.randn(1, 10, 10)
        aliases = get_aliases(expanded, x)
        assert_alias_count(aliases, 0)

    def test_repeat(self):
        """
        Repeat tiles the tensor - shape mismatch.
        """

        def repeated(x):
            return x.repeat(2, 2)

        x = torch.randn(10, 10)
        aliases = get_aliases(repeated, x)
        assert_alias_count(aliases, 0)

    def test_pad(self):
        """
        Padding increases size - shape mismatch.
        """

        def padded(x):
            return torch.nn.functional.pad(x, (1, 1, 1, 1))

        x = torch.randn(10, 10)
        aliases = get_aliases(padded, x)
        assert_alias_count(aliases, 0)

    def test_narrow(self):
        """
        Narrow extracts a subset - shape mismatch.
        """

        def narrowed(x):
            return x.narrow(0, 0, 5)

        x = torch.randn(10, 10)
        aliases = get_aliases(narrowed, x)
        assert_alias_count(aliases, 0)

    def test_sum_reduction(self):
        """
        Sum reduces dimensions - shape mismatch.
        """

        def sum_reduced(x):
            return x.sum(dim=1)

        x = torch.randn(10, 10)
        aliases = get_aliases(sum_reduced, x)
        assert_alias_count(aliases, 0)

    def test_mean_keepdim(self):
        """
        Mean with keepdim still changes shape on reduced dim.
        """

        def mean_reduced(x):
            return x.mean(dim=1, keepdim=True)

        x = torch.randn(10, 10)
        aliases = get_aliases(mean_reduced, x)
        assert_alias_count(aliases, 0)

    def test_concatenate(self):
        """
        Concatenation creates new tensor - shape mismatch.
        """

        def concatenated(x):
            return torch.cat([x, x], dim=0)

        x = torch.randn(10, 10)
        aliases = get_aliases(concatenated, x)
        assert_alias_count(aliases, 0)

    def test_stack(self):
        """
        Stack adds a new dimension - rank mismatch.
        """

        def stacked(x):
            return torch.stack([x, x], dim=0)

        x = torch.randn(10, 10)
        aliases = get_aliases(stacked, x)
        assert_alias_count(aliases, 0)

    def test_matmul(self):
        """
        Matrix multiplication produces different shape.
        """

        class MatMul(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("weight", torch.randn(10, 20))

            def forward(self, x):
                return torch.matmul(x, self.weight)

        model = MatMul()
        x = torch.randn(5, 10)
        aliases = get_aliases(model, x)
        assert_alias_count(aliases, 0)

    def test_linear(self):
        """
        Linear layer changes last dimension.
        """

        class LinearModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 20)

            def forward(self, x):
                return self.linear(x)

        model = LinearModel()
        x = torch.randn(5, 10)
        aliases = get_aliases(model, x)
        assert_alias_count(aliases, 0)

    def test_conv2d(self):
        """
        Convolution changes spatial dimensions.
        """

        class ConvModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, kernel_size=3)

            def forward(self, x):
                return self.conv(x)

        model = ConvModel()
        x = torch.randn(1, 3, 32, 32)
        aliases = get_aliases(model, x)
        assert_alias_count(aliases, 0)

    def test_pooling(self):
        """
        Pooling reduces spatial dimensions.
        """

        class PoolModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = torch.nn.MaxPool2d(2)

            def forward(self, x):
                return self.pool(x)

        model = PoolModel()
        x = torch.randn(1, 3, 32, 32)
        aliases = get_aliases(model, x)
        assert_alias_count(aliases, 0)

    def test_adaptive_pool(self):
        """
        Adaptive pooling changes to fixed output size.
        """

        class AdaptivePoolModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

            def forward(self, x):
                return self.pool(x)

        model = AdaptivePoolModel()
        x = torch.randn(1, 3, 32, 32)
        aliases = get_aliases(model, x)
        assert_alias_count(aliases, 0)

    def test_gather(self):
        """
        Gather selects elements - potential shape mismatch.
        """

        class Gathered(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("indices", torch.tensor([[0, 1], [2, 3]]))

            def forward(self, x):
                return torch.gather(x, 1, self.indices)

        model = Gathered()
        x = torch.randn(2, 10)
        aliases = get_aliases(model, x)
        assert_alias_count(aliases, 0)

    def test_view_as_different_shape(self):
        """
        View with different shape - same as reshape.
        """

        def view_different(x):
            return x.view(25, 4)

        x = torch.randn(10, 10)
        aliases = get_aliases(view_different, x)
        assert_alias_count(aliases, 0)

    def test_diagonal(self):
        """
        Diagonal extracts diagonal elements - shape mismatch.
        """

        def diagonal(x):
            return torch.diagonal(x)

        x = torch.randn(10, 10)
        aliases = get_aliases(diagonal, x)
        assert_alias_count(aliases, 0)

    def test_multiple_outputs_different_shapes(self):
        """
        Multiple outputs with different shapes from input.
        """

        def multi_output(x):
            return x.sum(dim=0), x.mean(dim=1)

        x = torch.randn(10, 10)
        aliases = get_aliases(multi_output, x)
        assert_alias_count(aliases, 0)

    def test_embedding_lookup(self):
        """
        Embedding lookup changes shape based on indices.
        """

        class EmbeddingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(100, 64)

            def forward(self, x):
                return self.embedding(x)

        model = EmbeddingModel()
        x = torch.randint(0, 100, (5, 10))
        aliases = get_aliases(model, x)
        assert_alias_count(aliases, 0)

    def test_dual_kv_cache_output_aliasing(self):
        """Dual cache updates should create two aliases."""
        model = SimpleKVCacheReplica(hidden_size=64, max_seq_len=128)

        x = torch.randn(1, 64)
        k_cache = torch.zeros(1, 128, 64)
        v_cache = torch.zeros(1, 128, 64)

        aliases = get_aliases(model, x, k_cache, v_cache, 0)

        assert_alias_count(aliases, 2)
        assert_has_alias(aliases, param_num=2, output_idx=[0])
        assert_has_alias(aliases, param_num=3, output_idx=[1])

    @pytest.mark.neuron
    def test_scatter_dual_cache_on_device(self):
        """Scatter-based dual cache on Neuron device."""
        device = f"neuron:{torch.neuron.current_device()}"
        model = ScatterKVCache().to(device)

        pos = torch.tensor(124, device=device)
        new_k = torch.randn(1, 1, 64, device=device)
        new_v = torch.randn(1, 1, 64, device=device)

        aliases = get_aliases(model, pos, new_k, new_v)

        assert_alias_count(aliases, 2)
        assert_has_alias(aliases, param_num=1, output_idx=[0])
        assert_has_alias(aliases, param_num=3, output_idx=[1])

    def test_single_cache_scatter_aliasing(self):
        """Single cache with scatter should create one alias."""
        model = SingleScatterCache(hidden_size=64, max_seq_len=128)
        model.eval()

        pos = torch.tensor(0)
        new_kv = torch.randn(1, 1, 64)

        aliases = get_aliases(model, pos, new_kv)
        assert_alias_count(aliases, 1)
        assert_has_alias(aliases, param_num=1, output_idx=[0])
