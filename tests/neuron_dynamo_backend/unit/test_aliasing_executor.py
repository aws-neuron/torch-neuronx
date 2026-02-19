"""
Unit tests for aliasing information propagation.

Tests verify that aliasing information (views, in-place mutations, etc.)
properly propagates through the Neuron compiler backend.
"""

import pytest
import torch
import torch.nn as nn

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility before each test."""
    torch.manual_seed(0)


@pytest.fixture
def device():
    """Default device for tests."""
    return torch.neuron.current_device()


@pytest.fixture
def kv_cache_config():
    """Configuration for KV cache tests."""
    return {
        "batch_size": 1,
        "num_heads": 4,
        "max_seq_len": 128,
        "head_dim": 16,
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def compile_and_run(model, *inputs, backend="neuron", fullgraph=False):
    """Compile model with specified backend and run with given inputs."""
    compiled = torch.compile(model, backend=backend, fullgraph=fullgraph)
    with torch.no_grad():
        return compiled(*inputs)


# ============================================================================
# TEST MODELS - All initialized on CPU, moved with .to(device)
# ============================================================================


class InplaceAddModel(nn.Module):
    """Model with in-place addition on input tensor."""

    def forward(self, x):
        x.add_(10)
        return x * 2


class SimpleKVCache(nn.Module):
    """Simple KV cache with scalar index assignment."""

    def __init__(self, size=10):
        super().__init__()
        self.register_buffer("cache", torch.zeros(size))

    def forward(self, index: int, value: float):
        self.cache[index] = value


class ScatterKVCache(nn.Module):
    """KV cache using scatter operation for position-based updates."""

    def __init__(self, hidden_size=64, max_seq_len=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.register_buffer("cache", torch.zeros(1, 1, max_seq_len, hidden_size))

    def forward(self, pos, new_kv):
        idx = pos.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(1, 1, 1, self.hidden_size)
        self.cache.scatter_(2, idx, new_kv.unsqueeze(2))


class DualScatterKVCache(nn.Module):
    """Dual KV cache with separate key and value caches using scatter."""

    def __init__(self, hidden_size=64, max_seq_len=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.register_buffer("k_cache", torch.zeros(1, 1, max_seq_len, hidden_size))
        self.register_buffer("v_cache", torch.zeros(1, 1, max_seq_len, hidden_size))

    def forward(self, pos, new_k, new_v):
        idx = pos.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(1, 1, 1, self.hidden_size)
        self.k_cache.scatter_(2, idx, new_k.unsqueeze(2))
        self.v_cache.scatter_(2, idx, new_v.unsqueeze(2))


class AttentionWithKVCacheSimple(nn.Module):
    """Simplified attention module with external KV cache updates via slice assignment."""

    def forward(self, x, k_cache, v_cache, cache_pos: int):
        k_cache[1, cache_pos] = 100
        v_cache[5, cache_pos + 5] = 200
        return x * x, k_cache, v_cache


class AttentionWithKVCache(nn.Module):
    """Simplified attention module with external KV cache updates via slice assignment."""

    def __init__(self, hidden_size=64):
        super().__init__()
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, k_cache, v_cache, cache_pos: int):
        k = self.k_proj(x)
        v = self.v_proj(x)
        k_cache[0, cache_pos : cache_pos + 1] = k
        v_cache[0, cache_pos : cache_pos + 1] = v
        return x * k_cache.sum() + cache_pos, k_cache, v_cache


class ResidualBlock(nn.Module):
    """Residual block with in-place addition."""

    def __init__(self, size):
        super().__init__()
        self.linear = nn.Linear(size, size)

    def forward(self, x):
        residual = x.clone()
        out = self.linear(x)
        out.add_(residual)
        return out


# ============================================================================
# TEST CLASS: In-Place Operations
# ============================================================================


class TestInPlaceOperations:
    """Tests for in-place tensor operations and their aliasing behavior."""

    def test_inplace_add(self, device):
        """Test in-place addition modifies input tensor and produces correct output."""
        model = InplaceAddModel()
        model.eval()
        model.to(device)

        x_cpu = torch.randn(10, 10)
        expected_x_modified = x_cpu + 10
        expected_output = expected_x_modified * 2

        x = x_cpu.clone().to(device)
        result = compile_and_run(model, x, fullgraph=True)

        assert result.shape == (10, 10)
        assert result.device.type == "neuron"
        torch.testing.assert_close(
            result.cpu(), expected_output, atol=1e-5, rtol=1e-5, msg="Output mismatch"
        )
        torch.testing.assert_close(
            x.cpu(), expected_x_modified, atol=1e-5, rtol=1e-5, msg="Input not modified in-place"
        )

    def test_inplace_add_with_two_inputs(self, device):
        """Test in-place addition with separate input tensors."""

        class Model(nn.Module):
            def forward(self, x, y):
                x.add_(y)
                return x

        model = Model()
        model.eval()
        model.to(device)

        x_cpu = torch.randn(10, 10)
        y_cpu = torch.randn(10, 10)
        expected_output = x_cpu + y_cpu
        expected_x_modified = expected_output.clone()

        x = x_cpu.clone().to(device)
        y = y_cpu.clone().to(device)

        result = compile_and_run(model, x, y, fullgraph=True)

        assert result.shape == (10, 10)
        assert result.device.type == "neuron"
        torch.testing.assert_close(
            result.cpu(), expected_output, atol=1e-5, rtol=1e-5, msg="Output mismatch"
        )
        torch.testing.assert_close(
            x.cpu(), expected_x_modified, atol=1e-5, rtol=1e-5, msg="x not modified in-place"
        )
        torch.testing.assert_close(
            y.cpu(), y_cpu, atol=1e-5, rtol=1e-5, msg="y was unexpectedly modified"
        )


# ============================================================================
# TEST CLASS: KV Cache Patterns
# ============================================================================


class TestKVCachePatterns:
    """Tests for KV cache update patterns common in transformer models."""

    def test_simple_index_assignment(self, device):
        """Test simple scalar index assignment to cache buffer."""
        model = SimpleKVCache(size=10)
        model.eval()
        model.to(device)

        index = 2
        value = 10.0

        _ = compile_and_run(model, index, value)

        cache_value = model.cache[index].cpu().item()
        assert (
            cache_value == value
        ), f"Cache not updated at index {index}: expected {value}, got {cache_value}"

    def test_scatter_single_position(self, device, kv_cache_config):
        """Test scatter operation for single position update in KV cache."""
        cfg = kv_cache_config
        hidden_size = cfg["num_heads"] * cfg["head_dim"]

        model = ScatterKVCache(hidden_size=hidden_size, max_seq_len=cfg["max_seq_len"])
        model.eval()
        model.to(device)

        new_kv_cpu = torch.randn(1, 1, hidden_size)
        pos = torch.tensor(5, device=device)
        new_kv = new_kv_cpu.clone().to(device)

        _ = compile_and_run(model, pos, new_kv)

        actual = model.cache[0, 0, 5, :].cpu()
        expected = new_kv_cpu[0, 0, :]
        torch.testing.assert_close(
            actual, expected, atol=1e-5, rtol=1e-5, msg="Cache not updated at position 5"
        )

    def test_dual_cache_scatter(self, device, kv_cache_config):
        """Test dual KV cache with simultaneous scatter operations."""
        cfg = kv_cache_config
        hidden_size = cfg["num_heads"] * cfg["head_dim"]

        model = DualScatterKVCache(hidden_size=hidden_size, max_seq_len=cfg["max_seq_len"])
        model.eval()
        model.to(device)

        new_k_cpu = torch.randn(1, 1, hidden_size)
        new_v_cpu = torch.randn(1, 1, hidden_size)

        pos = torch.tensor(10, device=device)
        new_k = new_k_cpu.clone().to(device)
        new_v = new_v_cpu.clone().to(device)

        _ = compile_and_run(model, pos, new_k, new_v, fullgraph=True)

        k_actual = model.k_cache[0, 0, 10, :].cpu()
        k_expected = new_k_cpu[0, 0, :]
        torch.testing.assert_close(
            k_actual, k_expected, atol=1e-5, rtol=1e-5, msg="k_cache not updated at position 10"
        )

        v_actual = model.v_cache[0, 0, 10, :].cpu()
        v_expected = new_v_cpu[0, 0, :]
        torch.testing.assert_close(
            v_actual, v_expected, atol=1e-5, rtol=1e-5, msg="v_cache not updated at position 10"
        )

        zeros = torch.zeros(hidden_size)
        torch.testing.assert_close(
            model.k_cache[0, 0, 0, :].cpu(),
            zeros,
            msg="k_cache position 0 was unexpectedly modified",
        )
        torch.testing.assert_close(
            model.k_cache[0, 0, 11, :].cpu(),
            zeros,
            msg="k_cache position 11 was unexpectedly modified",
        )
        torch.testing.assert_close(
            model.v_cache[0, 0, 0, :].cpu(),
            zeros,
            msg="v_cache position 0 was unexpectedly modified",
        )
        torch.testing.assert_close(
            model.v_cache[0, 0, 11, :].cpu(),
            zeros,
            msg="v_cache position 11 was unexpectedly modified",
        )

    @pytest.mark.parametrize("cache_pos", [0, 1, 63, 127])
    def test_attention_kv_cache_positions(self, device, kv_cache_config, cache_pos):
        """Test KV cache slice assignment at various sequence positions."""
        cfg = kv_cache_config
        hidden_size = cfg["num_heads"] * cfg["head_dim"]

        # Create model on CPU
        model = AttentionWithKVCache(hidden_size=hidden_size)
        model.eval()

        # Generate input and compute expected on CPU
        x_cpu = torch.randn(cfg["batch_size"], hidden_size)
        k_cache_cpu = torch.zeros(cfg["batch_size"], cfg["max_seq_len"], hidden_size)
        v_cache_cpu = torch.zeros(cfg["batch_size"], cfg["max_seq_len"], hidden_size)

        with torch.no_grad():
            expected_k = model.k_proj(x_cpu)
            expected_v = model.v_proj(x_cpu)

        # Move model to device
        model.to(device)

        # Move inputs to device
        x = x_cpu.clone().to(device)
        k_cache = k_cache_cpu.clone().to(device)
        v_cache = v_cache_cpu.clone().to(device)

        result, k_out, v_out = compile_and_run(model, x, k_cache, v_cache, cache_pos)

        assert result.shape == (cfg["batch_size"], hidden_size)
        assert k_out.shape == (cfg["batch_size"], cfg["max_seq_len"], hidden_size)
        assert v_out.shape == (cfg["batch_size"], cfg["max_seq_len"], hidden_size)

        torch.neuron.synchronize()

        torch.testing.assert_close(
            k_out[0, cache_pos].cpu(),
            expected_k[0],
            atol=1e-5,
            rtol=1e-5,
            msg=f"k_cache mismatch at position {cache_pos}",
        )
        torch.testing.assert_close(
            v_out[0, cache_pos].cpu(),
            expected_v[0],
            atol=1e-5,
            rtol=1e-5,
            msg=f"v_cache mismatch at position {cache_pos}",
        )


# ============================================================================
# TEST CLASS: Complex Patterns
# ============================================================================


class TestComplexPatterns:
    """Tests for complex aliasing patterns combining multiple operations."""

    def test_residual_connection_with_inplace(self, device):
        """Test residual connection with in-place addition."""
        size = 64

        # Create and compute expected on CPU
        model = ResidualBlock(size)
        model.eval()

        x_cpu = torch.randn(1, size)

        with torch.no_grad():
            expected = model.linear(x_cpu) + x_cpu

        # Move model to device
        model.to(device)

        x = x_cpu.clone().to(device)
        result = compile_and_run(model, x)

        assert result.shape == (1, size)
        assert result.device.type == "neuron"

        torch.testing.assert_close(
            result.cpu(),
            expected,
            atol=1e-5,
            rtol=1e-5,
            msg="Residual connection output mismatch",
        )
        torch.testing.assert_close(
            x.cpu(), x_cpu, atol=1e-5, rtol=1e-5, msg="Input tensor was unexpectedly modified"
        )

    def test_multiple_inplace_operations(self, device):
        """Test chained in-place operations."""

        class ChainedInplace(nn.Module):
            def forward(self, x):
                x.add_(1)
                x.mul_(2)
                x.sub_(3)
                return x

        model = ChainedInplace()
        model.eval()
        model.to(device)

        x_cpu = torch.randn(10, 10)
        expected = ((x_cpu + 1) * 2) - 3

        x = x_cpu.clone().to(device)

        result = compile_and_run(model, x)

        torch.testing.assert_close(
            result.cpu(), expected, atol=1e-5, rtol=1e-5, msg="Chained in-place output mismatch"
        )

    def test_inplace_on_view(self, device):
        """Test in-place operation on a tensor view."""

        class InplaceOnView(nn.Module):
            def forward(self, x):
                x[:, :5].add_(10)
                return x

        model = InplaceOnView()
        model.eval()
        model.to(device)

        x_cpu = torch.randn(4, 10)
        expected = x_cpu.clone()
        expected[:, :5] = expected[:, :5] + 10

        x = x_cpu.clone().to(device)

        result = compile_and_run(model, x)

        torch.testing.assert_close(
            result[:, :5].cpu(),
            expected[:, :5],
            atol=1e-5,
            rtol=1e-5,
            msg="Modified slice mismatch",
        )
        torch.testing.assert_close(
            result[:, 5:].cpu(),
            expected[:, 5:],
            atol=1e-5,
            rtol=1e-5,
            msg="Unmodified slice was changed",
        )

    def test_multiple_outputs_shared_input(self, device):
        """Test multiple outputs derived from same input with in-place ops."""

        class MultiOutput(nn.Module):
            def forward(self, x):
                a = x.clone()
                b = x.clone()
                a.add_(1)
                b.mul_(2)
                return a, b, x

        model = MultiOutput()
        model.eval()
        model.to(device)

        x_cpu = torch.randn(8, 8)
        expected_a = x_cpu + 1
        expected_b = x_cpu * 2

        x = x_cpu.clone().to(device)

        a, b, x_out = compile_and_run(model, x)

        torch.testing.assert_close(
            a.cpu(), expected_a, atol=1e-5, rtol=1e-5, msg="Output 'a' mismatch"
        )
        torch.testing.assert_close(
            b.cpu(), expected_b, atol=1e-5, rtol=1e-5, msg="Output 'b' mismatch"
        )
        torch.testing.assert_close(
            x_out.cpu(), x_cpu, atol=1e-5, rtol=1e-5, msg="Original 'x' was modified"
        )

    def test_inplace_with_broadcasting(self, device):
        """Test in-place operation with broadcasting."""

        class InplaceBroadcast(nn.Module):
            def forward(self, x, bias):
                x.add_(bias)
                return x

        model = InplaceBroadcast()
        model.eval()
        model.to(device)

        x_cpu = torch.randn(4, 16)
        bias_cpu = torch.randn(1, 16)
        expected = x_cpu + bias_cpu

        x = x_cpu.clone().to(device)
        bias = bias_cpu.clone().to(device)

        result = compile_and_run(model, x, bias)

        torch.testing.assert_close(
            result.cpu(), expected, atol=1e-5, rtol=1e-5, msg="Broadcast in-place mismatch"
        )
        torch.testing.assert_close(
            bias.cpu(), bias_cpu, atol=1e-5, rtol=1e-5, msg="Bias was unexpectedly modified"
        )

    def test_int64_add_dtype_preserved(self, device):
        """Test that int64 addition preserves dtype in output."""

        class Int64AddModel(nn.Module):
            def forward(self, x, y):
                x.add_(y)
                return x

        model = Int64AddModel()
        model.eval()
        model.to(device)

        x_cpu = torch.randint(-1000, 1000, (4, 4), dtype=torch.int64)
        y_cpu = torch.randint(-1000, 1000, (4, 4), dtype=torch.int64)
        expected = x_cpu + y_cpu

        x = x_cpu.clone().to(device)
        y = y_cpu.clone().to(device)

        result = compile_and_run(model, x, y)

        assert result.dtype == torch.int64, f"Expected dtype int64, got {result.dtype}"
        torch.testing.assert_close(result.cpu(), expected, msg="Int64 addition result mismatch")


# ============================================================================
# TEST CLASS: AliasingInfo Data Structure
# ============================================================================


class TestAliasingInfo:
    """Tests for AliasingInfo data structure."""

    def test_empty_aliasing_info(self):
        """Test newly created AliasingInfo is empty."""
        from torch_neuronx.neuron_dynamo_backend.utils.alias_info import AliasingInfo

        aliasing_info = AliasingInfo()

        assert len(aliasing_info) == 0
        assert list(aliasing_info) == []

    def test_add_single_alias(self):
        """Test adding a single aliasing relationship."""
        from torch_neuronx.neuron_dynamo_backend.utils.alias_info import AliasingInfo

        aliasing_info = AliasingInfo()
        aliasing_info.add(parameter_number=0, parameter_index=[], output_index=1)

        assert len(aliasing_info) == 1
        assert aliasing_info.output_to_input == {1: 0}

    def test_add_multiple_aliases(self):
        """Test adding multiple aliasing relationships."""
        from torch_neuronx.neuron_dynamo_backend.utils.alias_info import AliasingInfo

        aliasing_info = AliasingInfo()
        aliasing_info.add(parameter_number=0, parameter_index=[], output_index=0)
        aliasing_info.add(parameter_number=1, parameter_index=[], output_index=1)
        aliasing_info.add(parameter_number=2, parameter_index=[], output_index=2)

        assert len(aliasing_info) == 3
        assert aliasing_info.output_to_input == {0: 0, 1: 1, 2: 2}

    def test_get_input_index_returns_correct_mapping(self):
        """Test get_input_index returns correct input index for aliased output."""
        from torch_neuronx.neuron_dynamo_backend.utils.alias_info import AliasingInfo

        aliasing_info = AliasingInfo()
        aliasing_info.add(parameter_number=5, parameter_index=[], output_index=10)

        assert aliasing_info.get_input_index(10) == 5

    def test_get_input_index_returns_none_for_non_aliased(self):
        """Test get_input_index returns None for non-aliased outputs."""
        from torch_neuronx.neuron_dynamo_backend.utils.alias_info import AliasingInfo

        aliasing_info = AliasingInfo()
        aliasing_info.add(parameter_number=0, parameter_index=[], output_index=2)

        assert aliasing_info.get_input_index(0) is None
        assert aliasing_info.get_input_index(1) is None
        assert aliasing_info.get_input_index(99) is None

    def test_get_input_index_empty_aliasing_info(self):
        """Test get_input_index returns None when no aliases exist."""
        from torch_neuronx.neuron_dynamo_backend.utils.alias_info import AliasingInfo

        aliasing_info = AliasingInfo()

        assert aliasing_info.get_input_index(0) is None

    def test_add_duplicate_raises_error_for_conflicting_input(self):
        """Test adding conflicting alias for same output raises ValueError."""
        from torch_neuronx.neuron_dynamo_backend.utils.alias_info import AliasingInfo

        aliasing_info = AliasingInfo()
        aliasing_info.add(parameter_number=0, parameter_index=[], output_index=0)

        with pytest.raises(ValueError, match="Output 0 already aliases input 0"):
            aliasing_info.add(parameter_number=1, parameter_index=[], output_index=0)

    def test_iteration(self):
        """Test that AliasingInfo can be iterated over."""
        from torch_neuronx.neuron_dynamo_backend.utils.alias_info import AliasingInfo

        aliasing_info = AliasingInfo()
        aliasing_info.add(parameter_number=0, parameter_index=[], output_index=0)
        aliasing_info.add(parameter_number=1, parameter_index=[0], output_index=1)

        aliases_list = list(aliasing_info)

        assert len(aliases_list) == 2
        assert aliases_list[0].parameter_number == 0
        assert aliases_list[0].parameter_index == []
        assert aliases_list[0].output_index == 0
        assert aliases_list[1].parameter_number == 1
        assert aliases_list[1].parameter_index == [0]
        assert aliases_list[1].output_index == 1

    def test_parameter_index_none(self):
        """Test alias with None parameter_index."""
        from torch_neuronx.neuron_dynamo_backend.utils.alias_info import AliasingInfo

        aliasing_info = AliasingInfo()
        aliasing_info.add(parameter_number=0, parameter_index=None, output_index=0)

        alias = next(iter(aliasing_info))
        assert alias.parameter_index is None


# ============================================================================
# TEST CLASS: Retain Device Mode with Mutations
# ============================================================================


class TestRetainDeviceMutation:
    """Test for retain_device mode with mutated outputs."""

    def test_retain_device_copies_mutated_output_to_original_input(self, device, monkeypatch):
        """
        Test that when retain_device=True and an output is aliased to an input,
        the mutated output is copied back to the original input tensor.
        """
        monkeypatch.setenv("TORCH_NEURONX_RETAIN_DEVICE_MODE", "1")

        class InplaceMutationModel(nn.Module):
            def forward(self, x):
                x.add_(10)
                return x

        model = InplaceMutationModel()
        model.eval()
        model.to(device)

        x_cpu = torch.randn(4, 4)
        expected = x_cpu + 10

        x = x_cpu.clone()

        result = compile_and_run(model, x, fullgraph=True)

        # With retain_device + mutation: result should be on CPU
        assert result.device.type == "cpu", f"Expected CPU, got {result.device.type}"

        torch.testing.assert_close(
            x,
            expected,
            atol=1e-5,
            rtol=1e-5,
            msg="Original input tensor was not updated via copy_()",
        )

        torch.testing.assert_close(
            result, expected, atol=1e-5, rtol=1e-5, msg="Mutated output value mismatch"
        )


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=long"])
