# ruff: noqa: N806

"""
Unit tests for attention decomposition (SDPA forward and backward)
"""

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

# =============================================================================
# Test Configuration
# =============================================================================


@dataclass
class SDPAConfig:
    """Configuration for a single SDPA test case."""

    dtype: torch.dtype
    shape: tuple  # (batch, q_heads, kv_heads, seq_len, head_dim)
    is_causal: bool
    dropout_p: float
    scale: float | None  # None means use default
    attn_bias: bool
    rtol: float
    atol: float


# Curated list of test configurations covering parameter space
SDPA_TEST_CONFIGS = [
    # === Basic dtype coverage ===
    pytest.param(
        SDPAConfig(torch.float32, (1, 2, 2, 4, 8), False, 0.0, None, False, 1e-3, 1e-4),
        id="f32-basic",
    ),
    pytest.param(
        SDPAConfig(torch.float16, (1, 2, 2, 4, 8), False, 0.0, None, False, 5e-2, 5e-2),
        id="f16-basic",
    ),
    pytest.param(
        SDPAConfig(torch.bfloat16, (1, 2, 2, 4, 8), False, 0.0, None, False, 5e-2, 5e-2),
        id="bf16-basic",
    ),
    # === Shape coverage ===
    pytest.param(
        SDPAConfig(torch.float32, (2, 4, 4, 8, 16), False, 0.0, None, False, 1e-3, 1e-4),
        id="f32-standard",
    ),
    pytest.param(
        SDPAConfig(torch.float32, (2, 8, 1, 12, 16), False, 0.0, None, False, 1e-3, 1e-4),
        id="f32-mqa",
    ),
    pytest.param(
        SDPAConfig(torch.float32, (2, 8, 2, 10, 16), False, 0.0, None, False, 1e-3, 1e-4),
        id="f32-gqa",
    ),
    pytest.param(
        SDPAConfig(torch.float32, (1, 4, 4, 6, 12), False, 0.0, None, False, 1e-3, 1e-4),
        id="f32-non_pow2",
    ),
    # === Causal mask coverage ===
    pytest.param(
        SDPAConfig(torch.float32, (1, 2, 2, 6, 8), True, 0.0, None, False, 1e-3, 1e-4),
        id="f32-causal",
    ),
    pytest.param(
        SDPAConfig(torch.float16, (2, 4, 4, 8, 16), True, 0.0, None, False, 5e-2, 5e-2),
        id="f16-causal",
    ),
    # === Dropout coverage ===
    pytest.param(
        SDPAConfig(torch.float32, (1, 2, 2, 4, 8), False, 0.3, None, False, 1e-3, 1e-4),
        id="f32-drop30",
    ),
    pytest.param(
        SDPAConfig(torch.float32, (2, 4, 4, 8, 16), False, 0.5, None, False, 1e-3, 1e-4),
        id="f32-drop50",
    ),
    pytest.param(
        SDPAConfig(torch.float32, (1, 2, 2, 4, 8), False, 1.0, None, False, 1e-3, 1e-4),
        id="f32-drop100",
    ),
    # === Custom scale coverage ===
    pytest.param(
        SDPAConfig(torch.float32, (1, 2, 2, 4, 8), False, 0.0, 0.25, False, 1e-3, 1e-4),
        id="f32-scale25",
    ),
    pytest.param(
        SDPAConfig(torch.float32, (1, 2, 2, 4, 8), False, 0.0, 0.5, False, 1e-3, 1e-4),
        id="f32-scale50",
    ),
    # === Attention bias coverage ===
    pytest.param(
        SDPAConfig(torch.float32, (1, 2, 2, 6, 8), False, 0.0, None, True, 1e-3, 1e-4),
        id="f32-bias",
    ),
    pytest.param(
        SDPAConfig(torch.float16, (1, 2, 2, 6, 8), False, 0.0, None, True, 5e-2, 5e-2),
        id="f16-bias",
    ),
    # === Combined configurations ===
    pytest.param(
        SDPAConfig(torch.float32, (1, 2, 2, 6, 8), True, 0.3, None, False, 1e-3, 1e-4),
        id="f32-causal-drop",
    ),
    pytest.param(
        SDPAConfig(torch.float32, (2, 4, 4, 8, 16), True, 0.0, 0.25, False, 1e-3, 1e-4),
        id="f32-causal-scale",
    ),
    pytest.param(
        SDPAConfig(torch.float32, (1, 2, 2, 6, 8), False, 0.3, 0.25, False, 1e-3, 1e-4),
        id="f32-drop-scale",
    ),
    pytest.param(
        SDPAConfig(torch.float16, (2, 4, 4, 8, 16), True, 0.3, None, False, 5e-2, 5e-2),
        id="f16-causal-drop",
    ),
    pytest.param(
        SDPAConfig(torch.bfloat16, (2, 8, 2, 10, 16), True, 0.0, None, False, 5e-2, 5e-2),
        id="bf16-gqa-causal",
    ),
]


class TestSDPAForward:
    """Forward-only tests for SDPA decomposition (no gradient computation)"""

    @pytest.mark.parametrize("config", SDPA_TEST_CONFIGS)
    def test_sdpa_forward(self, config: SDPAConfig):
        """Test SDPA forward with various configurations."""
        batch_size, q_heads, kv_heads, seq_len, head_dim = config.shape
        torch.manual_seed(42)

        query = torch.randn(batch_size, q_heads, seq_len, head_dim, dtype=config.dtype)
        key = torch.randn(batch_size, kv_heads, seq_len, head_dim, dtype=config.dtype)
        value = torch.randn(batch_size, kv_heads, seq_len, head_dim, dtype=config.dtype)
        attn_bias = (
            torch.randn(batch_size, q_heads, seq_len, seq_len, dtype=config.dtype)
            if config.attn_bias
            else None
        )

        # Build kwargs for SDPA
        sdpa_kwargs = {
            "is_causal": config.is_causal,
            "dropout_p": config.dropout_p,
            "enable_gqa": True,
        }
        if config.scale is not None:
            sdpa_kwargs["scale"] = config.scale
        if attn_bias is not None:
            sdpa_kwargs["attn_mask"] = attn_bias

        with torch.no_grad():
            torch.manual_seed(42)
            output_cpu = F.scaled_dot_product_attention(query, key, value, **sdpa_kwargs)

        with torch.no_grad():
            # Create compiled function with captured kwargs
            if attn_bias is not None:
                compiled_fn = torch.compile(
                    lambda q, k, v, bias: F.scaled_dot_product_attention(
                        q, k, v, **{**sdpa_kwargs, "attn_mask": bias}
                    ),
                    backend="neuron",
                )
                torch.manual_seed(42)
                output_neuron = compiled_fn(
                    query.clone().to("neuron"),
                    key.clone().to("neuron"),
                    value.clone().to("neuron"),
                    attn_bias.clone().to("neuron"),
                )
            else:
                compiled_fn = torch.compile(
                    lambda q, k, v: F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs),
                    backend="neuron",
                )
                torch.manual_seed(42)
                output_neuron = compiled_fn(
                    query.clone().to("neuron"),
                    key.clone().to("neuron"),
                    value.clone().to("neuron"),
                )

        assert output_neuron.cpu().shape == output_cpu.shape
        torch.testing.assert_close(
            output_neuron.cpu(), output_cpu, rtol=config.rtol, atol=config.atol
        )

    def test_sdpa_forward_cross_attention(self):
        """Test SDPA forward for cross-attention (different seq lengths - special case)."""
        torch.manual_seed(42)
        batch_size, num_heads, q_seq_len, kv_seq_len, head_dim = 2, 4, 6, 10, 16

        query = torch.randn(batch_size, num_heads, q_seq_len, head_dim)
        key = torch.randn(batch_size, num_heads, kv_seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, kv_seq_len, head_dim)

        with torch.no_grad():
            torch.manual_seed(42)
            output_cpu = F.scaled_dot_product_attention(query, key, value, enable_gqa=True)

        with torch.no_grad():
            compiled_fn = torch.compile(
                lambda q, k, v: F.scaled_dot_product_attention(q, k, v, enable_gqa=True),
                backend="neuron",
            )
            torch.manual_seed(42)
            output_neuron = compiled_fn(
                query.clone().to("neuron"),
                key.clone().to("neuron"),
                value.clone().to("neuron"),
            )

        torch.testing.assert_close(output_neuron.cpu(), output_cpu, rtol=1e-3, atol=1e-4)


class TestSDPABackward:
    """Forward+backward tests for SDPA decomposition (gradient computation)"""

    @pytest.mark.parametrize("config", SDPA_TEST_CONFIGS)
    def test_sdpa_backward(self, config: SDPAConfig):
        """Test SDPA backward with various configurations."""
        batch_size, q_heads, kv_heads, seq_len, head_dim = config.shape
        torch.manual_seed(42)

        query_cpu = torch.randn(
            batch_size, q_heads, seq_len, head_dim, dtype=config.dtype, requires_grad=True
        )
        key_cpu = torch.randn(
            batch_size, kv_heads, seq_len, head_dim, dtype=config.dtype, requires_grad=True
        )
        value_cpu = torch.randn(
            batch_size, kv_heads, seq_len, head_dim, dtype=config.dtype, requires_grad=True
        )
        grad_out = torch.randn(batch_size, q_heads, seq_len, head_dim, dtype=config.dtype)
        attn_bias = (
            torch.randn(batch_size, q_heads, seq_len, seq_len, dtype=config.dtype)
            if config.attn_bias
            else None
        )

        query_neuron = query_cpu.detach().clone().to("neuron").requires_grad_(True)
        key_neuron = key_cpu.detach().clone().to("neuron").requires_grad_(True)
        value_neuron = value_cpu.detach().clone().to("neuron").requires_grad_(True)

        # Build kwargs for SDPA
        sdpa_kwargs = {
            "is_causal": config.is_causal,
            "dropout_p": config.dropout_p,
            "enable_gqa": True,
        }
        if config.scale is not None:
            sdpa_kwargs["scale"] = config.scale
        if attn_bias is not None:
            sdpa_kwargs["attn_mask"] = attn_bias

        # Reference forward + backward
        torch.manual_seed(42)
        output_cpu = F.scaled_dot_product_attention(query_cpu, key_cpu, value_cpu, **sdpa_kwargs)
        output_cpu.backward(grad_out)

        # Compiled forward + backward
        if attn_bias is not None:
            compiled_fn = torch.compile(
                lambda q, k, v, bias: F.scaled_dot_product_attention(
                    q, k, v, **{**sdpa_kwargs, "attn_mask": bias}
                ),
                backend="neuron",
            )
            torch.manual_seed(42)
            output_neuron = compiled_fn(
                query_neuron, key_neuron, value_neuron, attn_bias.clone().to("neuron")
            )
        else:
            compiled_fn = torch.compile(
                lambda q, k, v: F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs),
                backend="neuron",
            )
            torch.manual_seed(42)
            output_neuron = compiled_fn(query_neuron, key_neuron, value_neuron)

        output_neuron.backward(grad_out.clone().to("neuron"))

        # Verify forward output
        torch.testing.assert_close(
            output_neuron.cpu(), output_cpu, rtol=config.rtol, atol=config.atol
        )

        # Verify gradients
        torch.testing.assert_close(
            query_neuron.grad.cpu(), query_cpu.grad, rtol=config.rtol, atol=config.atol
        )
        torch.testing.assert_close(
            key_neuron.grad.cpu(), key_cpu.grad, rtol=config.rtol, atol=config.atol
        )
        torch.testing.assert_close(
            value_neuron.grad.cpu(), value_cpu.grad, rtol=config.rtol, atol=config.atol
        )

    def test_sdpa_backward_cross_attention(self):
        """Test SDPA backward for cross-attention (different seq lengths - special case)."""
        torch.manual_seed(42)
        batch_size, num_heads, q_seq_len, kv_seq_len, head_dim = 2, 4, 6, 10, 16

        query_cpu = torch.randn(batch_size, num_heads, q_seq_len, head_dim, requires_grad=True)
        key_cpu = torch.randn(batch_size, num_heads, kv_seq_len, head_dim, requires_grad=True)
        value_cpu = torch.randn(batch_size, num_heads, kv_seq_len, head_dim, requires_grad=True)
        grad_out = torch.randn(batch_size, num_heads, q_seq_len, head_dim)

        query_neuron = query_cpu.detach().clone().to("neuron").requires_grad_(True)
        key_neuron = key_cpu.detach().clone().to("neuron").requires_grad_(True)
        value_neuron = value_cpu.detach().clone().to("neuron").requires_grad_(True)

        # Reference forward + backward
        torch.manual_seed(42)
        output_cpu = F.scaled_dot_product_attention(query_cpu, key_cpu, value_cpu, enable_gqa=True)
        output_cpu.backward(grad_out)

        # Compiled forward + backward
        compiled_fn = torch.compile(
            lambda q, k, v: F.scaled_dot_product_attention(q, k, v, enable_gqa=True),
            backend="neuron",
        )
        torch.manual_seed(42)
        output_neuron = compiled_fn(query_neuron, key_neuron, value_neuron)
        output_neuron.backward(grad_out.clone().to("neuron"))

        torch.testing.assert_close(query_neuron.grad.cpu(), query_cpu.grad, rtol=1e-3, atol=1e-4)
        torch.testing.assert_close(key_neuron.grad.cpu(), key_cpu.grad, rtol=1e-3, atol=1e-4)
        torch.testing.assert_close(value_neuron.grad.cpu(), value_cpu.grad, rtol=1e-3, atol=1e-4)
