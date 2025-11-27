import pytest
import torch

from tests.utils.neuron_test_utils import assert_raises
from torch_neuronx.python_ops.native_multi_head_attn_prefix import NativeMultiHeadAttnPrefixOp

# Create a module-level instance of the operation for testing
_prefix_op = NativeMultiHeadAttnPrefixOp()


class TestNativeMultiHeadAttnPrefix:
    """Test cases for the native multi-head attention prefix operation"""

    @pytest.fixture
    def test_params(self):
        """Common test parameters"""
        return {
            "batch_size": 1,
            "seq_len": 2048,
            "d_model": 1024,  # d_embedding
            "num_heads": 8,
            "d_head": 128,  # d_model // num_heads
        }

    @pytest.fixture
    def create_test_tensors(self, test_params):
        """Create test tensors for the operation"""
        batch = test_params["batch_size"]
        seq_len = test_params["seq_len"]
        d_model = test_params["d_model"]

        # Create deterministic test data
        torch.manual_seed(42)

        query = torch.randn(batch, seq_len, d_model, dtype=torch.float32)
        key = torch.randn(batch, seq_len, d_model, dtype=torch.float32)
        value = torch.randn(batch, seq_len, d_model, dtype=torch.float32)
        qkv_weight = torch.randn(3 * d_model, d_model, dtype=torch.float32)
        qkv_bias = torch.randn(3 * d_model, dtype=torch.float32)

        return query, key, value, qkv_weight, qkv_bias

    def test_basic_operation(self, test_params, create_test_tensors):
        """Test basic operation on Neuron device"""
        query, key, value, qkv_weight, qkv_bias = create_test_tensors
        num_heads = test_params["num_heads"]

        # Move to neuron device
        query_neuron = query.to("neuron")
        key_neuron = key.to("neuron")
        value_neuron = value.to("neuron")
        qkv_weight_neuron = qkv_weight.to("neuron")
        qkv_bias_neuron = qkv_bias.to("neuron")

        # Execute operation
        q_out, k_out, v_out = _prefix_op(
            query_neuron,
            key_neuron,
            value_neuron,
            qkv_weight_neuron,
            qkv_bias_neuron,
            num_heads=num_heads,
        )

        # Verify output shapes
        batch = test_params["batch_size"]
        seq_len = test_params["seq_len"]
        d_head = test_params["d_head"]
        expected_shape = (batch, num_heads, d_head, seq_len)

        assert (
            q_out.shape == expected_shape
        ), f"Q output shape mismatch: {q_out.shape} != {expected_shape}"
        assert (
            k_out.shape == expected_shape
        ), f"K output shape mismatch: {k_out.shape} != {expected_shape}"
        assert (
            v_out.shape == expected_shape
        ), f"V output shape mismatch: {v_out.shape} != {expected_shape}"

        # Verify outputs are on neuron device
        assert q_out.device.type == "neuron"
        assert k_out.device.type == "neuron"
        assert v_out.device.type == "neuron"

    def test_cpu_neuron_comparison(self, test_params, create_test_tensors):
        """Compare Neuron output with CPU reference implementation"""
        query, key, value, qkv_weight, qkv_bias = create_test_tensors
        num_heads = test_params["num_heads"]

        # CPU reference implementation
        def cpu_transform_qkv(query, key, value, qkv_weight, qkv_bias, num_heads):
            b, t, d_model = query.shape
            d_head = d_model // num_heads

            # Fuse inputs
            qkv_in = torch.cat(
                [query.reshape(-1, d_model), key.reshape(-1, d_model), value.reshape(-1, d_model)],
                dim=0,
            )

            # Linear projection
            proj = qkv_in @ qkv_weight.T + qkv_bias

            # Split back
            proj = proj.reshape(3, b * t, 3 * d_model)

            q_all = proj[0]
            k_all = proj[1]
            v_all = proj[2]

            q = q_all[:, :d_model].reshape(b, t, d_model)
            k = k_all[:, d_model : 2 * d_model].reshape(b, t, d_model)
            v = v_all[:, 2 * d_model :].reshape(b, t, d_model)

            def split_heads(x):
                x = x.reshape(b, t, num_heads, d_head)
                x = x.permute(0, 2, 3, 1)
                return x

            return split_heads(q), split_heads(k), split_heads(v)

        # CPU computation
        q_cpu, k_cpu, v_cpu = cpu_transform_qkv(query, key, value, qkv_weight, qkv_bias, num_heads)

        # Neuron computation
        query_neuron = query.to("neuron")
        key_neuron = key.to("neuron")
        value_neuron = value.to("neuron")
        qkv_weight_neuron = qkv_weight.to("neuron")
        qkv_bias_neuron = qkv_bias.to("neuron")

        q_neuron, k_neuron, v_neuron = _prefix_op(
            query_neuron,
            key_neuron,
            value_neuron,
            qkv_weight_neuron,
            qkv_bias_neuron,
            num_heads=num_heads,
        )

        # Move neuron results back to CPU for comparison
        q_neuron_cpu = q_neuron.cpu()
        k_neuron_cpu = k_neuron.cpu()
        v_neuron_cpu = v_neuron.cpu()

        # Compare results - allow slightly higher tolerance for numerical differences
        torch.testing.assert_close(q_neuron_cpu, q_cpu, rtol=1e-3, atol=1e-4)
        torch.testing.assert_close(k_neuron_cpu, k_cpu, rtol=1e-3, atol=1e-4)
        torch.testing.assert_close(v_neuron_cpu, v_cpu, rtol=1e-3, atol=1e-4)

    @pytest.mark.parametrize("num_heads", [1, 2, 4, 8, 16])
    def test_different_head_configurations(self, num_heads):
        """Test with different numbers of attention heads"""
        batch = 1
        seq_len = 512  # Smaller for this test
        d_model = 512

        if d_model % num_heads != 0:
            pytest.skip(f"d_model={d_model} not divisible by num_heads={num_heads}")

        torch.manual_seed(42)
        query = torch.randn(batch, seq_len, d_model, device="neuron")
        key = torch.randn(batch, seq_len, d_model, device="neuron")
        value = torch.randn(batch, seq_len, d_model, device="neuron")
        qkv_weight = torch.randn(3 * d_model, d_model, device="neuron")
        qkv_bias = torch.randn(3 * d_model, device="neuron")

        q_out, k_out, v_out = _prefix_op(
            query, key, value, qkv_weight, qkv_bias, num_heads=num_heads
        )

        d_head = d_model // num_heads
        expected_shape = (batch, num_heads, d_head, seq_len)

        assert q_out.shape == expected_shape
        assert k_out.shape == expected_shape
        assert v_out.shape == expected_shape

    def test_edge_cases(self):
        """Test edge cases"""
        # Test with minimum sequence length
        batch = 1
        seq_len = 1
        d_model = 64
        num_heads = 8

        query = torch.randn(batch, seq_len, d_model, device="neuron")
        key = torch.randn(batch, seq_len, d_model, device="neuron")
        value = torch.randn(batch, seq_len, d_model, device="neuron")
        qkv_weight = torch.randn(3 * d_model, d_model, device="neuron")
        qkv_bias = torch.randn(3 * d_model, device="neuron")

        q_out, k_out, v_out = _prefix_op(
            query, key, value, qkv_weight, qkv_bias, num_heads=num_heads
        )

        d_head = d_model // num_heads
        expected_shape = (batch, num_heads, d_head, seq_len)

        assert q_out.shape == expected_shape
        assert k_out.shape == expected_shape
        assert v_out.shape == expected_shape

    @assert_raises(RuntimeError)
    def test_invalid_inputs(self):
        """Test error handling for invalid inputs"""
        # Test with incompatible d_model and num_heads
        batch = 1
        seq_len = 128
        d_model = 100  # Not divisible by 8
        num_heads = 8

        query = torch.randn(batch, seq_len, d_model, device="neuron")
        key = torch.randn(batch, seq_len, d_model, device="neuron")
        value = torch.randn(batch, seq_len, d_model, device="neuron")
        qkv_weight = torch.randn(3 * d_model, d_model, device="neuron")
        qkv_bias = torch.randn(3 * d_model, device="neuron")

        # This should raise an error
        _prefix_op(query, key, value, qkv_weight, qkv_bias, num_heads=num_heads)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_different_batch_sizes(self, batch_size):
        """Test with different batch sizes"""
        seq_len = 256
        d_model = 256
        num_heads = 8

        torch.manual_seed(42)
        query = torch.randn(batch_size, seq_len, d_model, device="neuron")
        key = torch.randn(batch_size, seq_len, d_model, device="neuron")
        value = torch.randn(batch_size, seq_len, d_model, device="neuron")
        qkv_weight = torch.randn(3 * d_model, d_model, device="neuron")
        qkv_bias = torch.randn(3 * d_model, device="neuron")

        q_out, k_out, v_out = _prefix_op(
            query, key, value, qkv_weight, qkv_bias, num_heads=num_heads
        )

        d_head = d_model // num_heads
        expected_shape = (batch_size, num_heads, d_head, seq_len)

        assert q_out.shape == expected_shape
        assert k_out.shape == expected_shape
        assert v_out.shape == expected_shape

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["float32", "float16"])
    def test_dtype_consistency(self, dtype):
        """Test that the operation maintains dtype consistency"""
        batch = 1
        seq_len = 128
        d_model = 128
        num_heads = 8

        query = torch.randn(batch, seq_len, d_model, dtype=dtype, device="neuron")
        key = torch.randn(batch, seq_len, d_model, dtype=dtype, device="neuron")
        value = torch.randn(batch, seq_len, d_model, dtype=dtype, device="neuron")
        qkv_weight = torch.randn(3 * d_model, d_model, dtype=dtype, device="neuron")
        qkv_bias = torch.randn(3 * d_model, dtype=dtype, device="neuron")

        q_out, k_out, v_out = _prefix_op(
            query, key, value, qkv_weight, qkv_bias, num_heads=num_heads
        )

        assert q_out.dtype == dtype
        assert k_out.dtype == dtype
        assert v_out.dtype == dtype
