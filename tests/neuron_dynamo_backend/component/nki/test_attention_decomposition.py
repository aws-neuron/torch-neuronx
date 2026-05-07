import os

import pytest
import torch
from torch._dynamo.backends.common import aot_autograd
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention


def attention_func(query, key, value):
    """SDPA function to be compiled"""
    with sdpa_kernel(SDPBackend.OVERRIDEABLE):
        result = scaled_dot_product_attention(query, key, value)
    return result


def attention_func_cpu(query, key, value):
    """SDPA function to be compiled"""
    with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
        return scaled_dot_product_attention(query.cpu(), key.cpu(), value.cpu())


class TestAttentionDecomposition:
    @pytest.mark.parametrize(
        "batch_size,num_heads,seq_len,head_dim",
        [
            (4, 8, 128, 128),
            (1, 2, 1024, 128),
            (2, 4, 2048, 64),
        ],
    )
    def test_attention_decomposition(self, batch_size, num_heads, seq_len, head_dim):
        """Test attention with neuron backend"""
        seqlen_q = 128  # Minimum query length to enable sdpa decomposition
        query = torch.rand(batch_size, num_heads, seqlen_q, head_dim, device="neuron")
        key = torch.rand(batch_size, num_heads, seq_len, head_dim, device="neuron")
        value = torch.rand(batch_size, num_heads, seq_len, head_dim, device="neuron")

        # Compile with neuron backend
        compiled_attention = torch.compile(attention_func, backend="neuron", dynamic=False)
        cpu_output = attention_func_cpu(query, key, value)

        # Run compiled attention
        output = compiled_attention(query, key, value)

        # Compare outputs
        torch.testing.assert_close(output.cpu(), cpu_output, rtol=1e-2, atol=1e-3)
