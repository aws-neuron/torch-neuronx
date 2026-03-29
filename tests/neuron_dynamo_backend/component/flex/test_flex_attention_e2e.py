import logging
import math

import pytest
import torch
from nkilib.core.attention.attention_cte_torch import attention_cte_torch_ref
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import (
    and_masks,
    create_block_mask,
    create_mask,
    flex_attention,
)

from tests.utils.neuron_test_utils import max_all_close

logger = logging.getLogger(__name__)


def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def sliding_window_mod(
    b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
) -> torch.Tensor:
    return q_idx - kv_idx < 128


def alibi_score(score, b, h, q_idx, kv_idx):
    m = 2 ** (-8)
    # Calculate relative position (q_idx - k_idx) and apply bias
    bias = m * (kv_idx - q_idx)
    return score + bias


softcap = 20


def soft_cap(score, b, h, q_idx, kv_idx):
    score = torch.tanh(score * (1 / softcap))
    return score


sliding_window_causal = and_masks(causal, sliding_window_mod)


class TestFlexAttentionE2E:
    """End-to-end tests for flex attention with user-defined score and mask functions."""

    @pytest.mark.parametrize(
        "b,h,seq_len,s,e,score_mod,mask_mod,k_tile_size,dtype",
        [
            pytest.param(
                4, 3, 1024, 512, 128, alibi_score, causal, 128, torch.float32, id="causal_fp32"
            ),
            pytest.param(
                2, 4, 4096, 4096, 128, None, causal, 512, torch.bfloat16, id="causal_bf16"
            ),
            pytest.param(
                2,
                1,
                512,
                4096,
                64,
                None,
                sliding_window_mod,
                512,
                torch.float32,
                id="sliding_window_fp32",
            ),
            pytest.param(
                2,
                4,
                2048,
                2048,
                64,
                soft_cap,
                sliding_window_mod,
                256,
                torch.bfloat16,
                id="sliding_window_bf16",
            ),
            pytest.param(
                2,
                4,
                2048,
                2048,
                64,
                soft_cap,
                sliding_window_causal,
                128,
                torch.float32,
                id="sliding_window_causal_fp32",
            ),
            pytest.param(
                1,
                2,
                4096,
                8192,
                64,
                None,
                sliding_window_causal,
                512,
                torch.bfloat16,
                id="sliding_window_causal_bf16",
            ),
        ],
    )
    def test_flex_attention_forward(
        self, b, h, seq_len, s, e, score_mod, mask_mod, k_tile_size, dtype
    ):
        if score_mod == alibi_score:
            pytest.xfail("TODO: debug accuracy issue with alibi_score")

        # Monkey patch the device validation to allow neuron tensors for testing
        # TODO: find a way to avoid monkey-patching
        import torch.nn.attention.flex_attention as flex_attn_module

        _original_validate_device = flex_attn_module._validate_device

        def _patched_validate_device(query, key, value):
            if query.device.type == "neuron":
                return  # Skip validation for neuron, let it through for tracing
            return _original_validate_device(query, key, value)

        flex_attn_module._validate_device = _patched_validate_device
        """Test flex attention with causal mask end-to-end."""
        torch.manual_seed(42)
        query = torch.randn(b, h, seq_len, e, device="cpu", dtype=dtype)
        key = torch.randn(b, h, s, e, device="cpu", dtype=dtype)
        value = torch.randn(b, h, s, e, device="cpu", dtype=dtype)

        block_mask = create_block_mask(
            mask_mod,
            B=b,
            H=h,
            Q_LEN=seq_len,
            KV_LEN=s,
            BLOCK_SIZE=(128, k_tile_size),
            device="neuron",
        )

        # CPU reference
        cpu_mask = create_block_mask(
            mask_mod,
            B=b,
            H=h,
            Q_LEN=seq_len,
            KV_LEN=s,
            BLOCK_SIZE=(128, k_tile_size),
            device="cpu",
        )
        cpu_output, cpu_lse = flex_attention(
            query, key, value, score_mod=score_mod, block_mask=cpu_mask, return_lse=True
        )

        # Neuron compiled with return_lse=True
        @torch.compile(backend="neuron")
        def flex_attn_neuron(q, k, v, mask):
            output, lse = flex_attention(
                q, k, v, score_mod=score_mod, block_mask=mask, return_lse=True
            )
            return output, lse

        with torch.no_grad():
            neuron_output, neuron_lse = flex_attn_neuron(
                query.to("neuron"), key.to("neuron"), value.to("neuron"), block_mask
            )

        # Compare attention output
        # torch.testing.assert_close(neuron_output.cpu(), cpu_output, rtol=2e-2, atol=1e-3)
        assert max_all_close(
            neuron_output.cpu().float().numpy(),
            cpu_output.float().numpy(),
            rtol=2e-2,
            atol=1e-3,
        )

        # Compare logsumexp
        # CPU LSE uses base 2, while Neuron LSE uses base e and is negated
        neuron_lse_converted = neuron_lse.cpu() / (-math.log(2))
        assert max_all_close(
            neuron_lse_converted.float().numpy(),
            cpu_lse.float().numpy(),
            rtol=2e-2,
            atol=1e-3,
        )

    @pytest.mark.parametrize(
        "b,h,seq_len,s,e,score_mod,mask_mod,k_tile_size,dtype",
        [
            pytest.param(
                4,
                3,
                2048,
                2048,
                128,
                None,
                causal,
                512,
                torch.float32,
                id="causal_fp32",
            ),
            pytest.param(
                2,
                4,
                2048,
                2048,
                64,
                None,
                causal,
                512,
                torch.bfloat16,
                id="causal_bf16",
            ),
            pytest.param(
                2,
                4,
                8192,
                4096,
                64,
                None,
                sliding_window_mod,
                512,
                torch.float32,
                id="sliding_window_fp32",
            ),
            pytest.param(
                2,
                4,
                2048,
                2048,
                64,
                alibi_score,
                sliding_window_mod,
                512,
                torch.bfloat16,
                id="sliding_window_bf16",
            ),
            pytest.param(
                2,
                4,
                2048,
                2048,
                64,
                soft_cap,
                sliding_window_causal,
                512,
                torch.float32,
                id="sliding_window_causal_fp32",
            ),
            pytest.param(
                2,
                4,
                4096,
                8192,
                64,
                None,
                sliding_window_causal,
                512,
                torch.bfloat16,
                id="sliding_window_causal_bf16",
            ),
        ],
    )
    def test_flex_attention_backward(
        self, b, h, seq_len, s, e, score_mod, mask_mod, k_tile_size, dtype
    ):
        """Test flex attention backward pass (grad_q, grad_k, grad_v) end-to-end."""
        if score_mod is not None:
            pytest.xfail("Accuracy mismatch happens with custom score_mod")
        import torch.nn.attention.flex_attention as flex_attn_module

        _original_validate_device = flex_attn_module._validate_device

        def _patched_validate_device(query, key, value):
            if query.device.type == "neuron":
                return
            return _original_validate_device(query, key, value)

        flex_attn_module._validate_device = _patched_validate_device

        torch.manual_seed(42)
        query_cpu = torch.rand(b, h, seq_len, e, device="cpu", dtype=dtype, requires_grad=True)
        key_cpu = torch.rand(b, h, s, e, device="cpu", dtype=dtype, requires_grad=True)
        value_cpu = torch.rand(b, h, s, e, device="cpu", dtype=dtype, requires_grad=True)

        block_mask = create_block_mask(
            mask_mod,
            B=b,
            H=h,
            Q_LEN=seq_len,
            KV_LEN=s,
            BLOCK_SIZE=(128, k_tile_size),
            device="neuron",
        )
        cpu_mask = create_block_mask(
            mask_mod,
            B=b,
            H=h,
            Q_LEN=seq_len,
            KV_LEN=s,
            BLOCK_SIZE=(128, k_tile_size),
            device="cpu",
        )
        cpu_out = flex_attention(
            query_cpu,
            key_cpu,
            value_cpu,
            score_mod=score_mod,
            block_mask=cpu_mask,
        )
        grad_output = torch.ones_like(cpu_out)
        cpu_out.backward(grad_output)
        cpu_grad_q = query_cpu.grad.clone()
        cpu_grad_k = key_cpu.grad.clone()
        cpu_grad_v = value_cpu.grad.clone()

        # Neuron compiled backward
        query_neuron = query_cpu.detach().to("neuron").requires_grad_(True)
        key_neuron = key_cpu.detach().to("neuron").requires_grad_(True)
        value_neuron = value_cpu.detach().to("neuron").requires_grad_(True)

        @torch.compile(backend="neuron")
        def flex_attn_neuron(q, k, v, mask):
            neuron_out = flex_attention(
                q,
                k,
                v,
                score_mod=score_mod,
                block_mask=mask,
            )
            return neuron_out

        neuron_out = flex_attn_neuron(
            query_neuron,
            key_neuron,
            value_neuron,
            block_mask,
        )
        neuron_out.backward(torch.ones_like(neuron_out))

        for neuron_grad, cpu_grad in [
            (query_neuron.grad.cpu(), cpu_grad_q),
            (key_neuron.grad.cpu(), cpu_grad_k),
            (value_neuron.grad.cpu(), cpu_grad_v),
        ]:
            assert max_all_close(
                neuron_grad.float().numpy(),
                cpu_grad.float().numpy(),
                rtol=2e-2,
                atol=1e-2,
            )
