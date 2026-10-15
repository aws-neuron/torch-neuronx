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

    def test_attention_decomposition_asymmetric_head_dimensions(self, monkeypatch):
        """Test compiled SDPA with different query/key and value dimensions."""
        from torch_neuronx.neuron_dynamo_backend import decompositions

        def fail_fallback(*args, **kwargs):
            raise AssertionError("The SDPA decomposition used the non-NKI path.")

        monkeypatch.setattr(
            decompositions, "scaled_dot_product_attention_decomposition", fail_fallback
        )
        monkeypatch.setattr(
            decompositions, "compute_attention_backward_decomposition", fail_fallback
        )

        torch.manual_seed(42)
        shape_qk = (1, 2, 512, 192)
        shape_v = (1, 2, 512, 128)
        query_cpu = torch.randn(shape_qk, dtype=torch.bfloat16, requires_grad=True)
        key_cpu = torch.randn(shape_qk, dtype=torch.bfloat16, requires_grad=True)
        value_cpu = torch.randn(shape_v, dtype=torch.bfloat16, requires_grad=True)

        output_cpu = attention_func_cpu(query_cpu, key_cpu, value_cpu)
        grad_output = torch.randn_like(output_cpu)
        output_cpu.backward(grad_output)

        query_neuron = query_cpu.detach().to("neuron").requires_grad_(True)
        key_neuron = key_cpu.detach().to("neuron").requires_grad_(True)
        value_neuron = value_cpu.detach().to("neuron").requires_grad_(True)
        compiled_attention = torch.compile(attention_func, backend="neuron", dynamic=False)
        output_neuron = compiled_attention(query_neuron, key_neuron, value_neuron)
        output_neuron.backward(grad_output.to("neuron"))

        torch.testing.assert_close(output_neuron.cpu(), output_cpu, rtol=5e-2, atol=1e-2)
        torch.testing.assert_close(query_neuron.grad.cpu(), query_cpu.grad, rtol=5e-2, atol=1e-2)
        torch.testing.assert_close(key_neuron.grad.cpu(), key_cpu.grad, rtol=5e-2, atol=1e-2)
        torch.testing.assert_close(value_neuron.grad.cpu(), value_cpu.grad, rtol=5e-2, atol=1e-2)
