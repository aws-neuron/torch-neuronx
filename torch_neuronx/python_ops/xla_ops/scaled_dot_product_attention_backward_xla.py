import math

import jax.numpy as jnp
import torch

from ...kernels.xla_kernel import TorchNeuronXLAKernel
from ..base import AttentionOpImpl, ExecutionResult
from .scaled_dot_product_attention_xla import generate_dropout_mask


class ScaledDotProductAttentionBackwardXLAImpl(AttentionOpImpl):
    def __init__(self, op_name: str):
        def compute_attention_backward(
            grad_out: jnp.ndarray,
            query: jnp.ndarray,
            key: jnp.ndarray,
            value: jnp.ndarray,
            attn_bias: jnp.ndarray,
            dropout_mask: jnp.ndarray,
            lse: jnp.ndarray,
            scale: float,
            is_causal: bool,
            dropout_p: float,
            mixed_precision: bool,
        ):
            """
            JAX backward pass for scaled dot-product attention

            Args:
                grad_out: [batch, num_heads, seq_len_q, head_dim]
                query: [batch, num_heads, seq_len_q, head_dim]
                key: [batch, num_heads, seq_len_k, head_dim]
                value: [batch, num_heads, seq_len_k, head_dim]
                out: [batch, num_heads, seq_len_q, head_dim] (forward output)
                lse: [batch, num_heads, seq_len_q] (log-sum-exp from forward)
                scale: softmax scale factor
                is_causal: whether causal masking was applied
                dropout_p: dropout probability
                mixed_precision: whether to use mixed precision

            Returns:
                (grad_q, grad_k, grad_v): gradients w.r.t. query, key, value
            """

            # Store original dtype
            orig_dtype = query.dtype
            acc_dtype = jnp.float32 if mixed_precision else orig_dtype

            # Upcast for computation
            grad_out = grad_out.astype(acc_dtype)
            query = query.astype(acc_dtype)
            key = key.astype(acc_dtype)
            value = value.astype(acc_dtype)
            lse = lse.astype(acc_dtype)
            if attn_bias is not None:
                attn_bias = attn_bias.astype(acc_dtype)

            # Compute attention logits: Q @ K^T
            logits = jnp.einsum("bhqd,bhkd->bhqk", query, key) * scale

            # Apply causal mask if needed
            if is_causal:
                seq_len_q, seq_len_k = logits.shape[-2:]
                causal_mask = jnp.tri(seq_len_q, seq_len_k, dtype=bool)
                # Use same mask value as forward pass
                logits = logits + jnp.where(causal_mask, 0.0, -jnp.inf)
            elif attn_bias is not None:
                logits = logits + attn_bias

            # Compute attention probabilities using LSE
            probs = jnp.exp(logits - lse[:, :, :, None])

            # Apply dropout if specified
            if dropout_p > 0.0 and dropout_mask.size > 0:
                if dropout_p == 1.0:
                    probs = probs * dropout_mask
                else:
                    probs = probs * dropout_mask / (1.0 - dropout_p)

            # Compute gradients
            # grad_v = P^T @ grad_out
            # Same as jnp.einsum("bhkq,bhqd->bhkd", probs.transpose(0, 1, 3, 2), grad_out)
            grad_v = jnp.einsum("bhqk,bhqd->bhkd", probs, grad_out)

            # grad_out_v = grad_out @ V^T
            # Same as jnp.einsum("bhqd,bhdk->bhqk", grad_out, value.transpose(0, 1, 3, 2))
            grad_out_v = jnp.einsum("bhqd,bhkd->bhqk", grad_out, value)

            # Compute softmax gradient: ds = P * (grad_out_v - sum_reduction)
            sum_reduction = jnp.sum(grad_out_v * probs, axis=-1, keepdims=True)
            ds = probs * (grad_out_v - sum_reduction) * scale

            # Apply causal mask to gradient if needed
            if is_causal:
                seq_len_q, seq_len_k = ds.shape[-2:]
                causal_mask = jnp.tri(seq_len_q, seq_len_k, dtype=bool)
                ds = jnp.where(causal_mask, ds, 0.0)

            # grad_q = ds @ K
            # No transpose on K since we did Q @ K^T
            grad_q = jnp.einsum("bhqk,bhkd->bhqd", ds, key)

            # grad_k = ds^T @ Q (transpose ds along last two dims)
            # Same as jnp.einsum("bhkq,bhqd->bhkd", ds.transpose(0, 1, 3, 2), query)
            grad_k = jnp.einsum("bhqk,bhqd->bhkd", ds, query)

            # Compute gradient w.r.t. attention bias
            grad_attn_bias = ds.astype(orig_dtype) if attn_bias is not None else None

            # Downcast back to original dtype
            grad_q = grad_q.astype(orig_dtype)
            grad_k = grad_k.astype(orig_dtype)
            grad_v = grad_v.astype(orig_dtype)

            return grad_q, grad_k, grad_v, grad_attn_bias

        # Create XLA kernel
        self.kernel = TorchNeuronXLAKernel(
            compute_attention_backward,
            op_name,
            static_argnums=(7, 8, 9, 10),  # scale, is_causal, dropout_p, mixed_precision
        )

    def can_handle(
        self,
        grad_out: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: torch.Tensor,
        grad_input_mask: list[bool],
        out: torch.Tensor,
        logsumexp: torch.Tensor,
        cum_seq_q: torch.Tensor,
        cum_seq_k: torch.Tensor,
        max_q: int,
        max_k: int,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        philox_seed: torch.Tensor | None = None,
        philox_offset: torch.Tensor | None = None,
        scale: float | None = None,
    ) -> bool:
        """Check if XLA backward implementation can handle the given inputs"""

        # Check query, key, value tensors
        if not super().can_handle(query, key, value):
            return False

        # Must be on Neuron device
        tensors = [grad_out, out, logsumexp]
        if not all(t.device.type == "neuron" for t in tensors):
            return False

        # Check tensors have the same dtype
        if not (query.dtype == grad_out.dtype == out.dtype):
            return False

        # Check shapes - must be 4D tensors
        if grad_out.ndim != 4 or out.ndim != 4:
            return False

        # Extract dimensions
        batch_size, q_heads, seq_len_q, _ = query.shape
        batch_go, go_heads, seq_len_go, _ = grad_out.shape

        # Shape validation
        if seq_len_q != seq_len_go or q_heads != go_heads or batch_size != batch_go:
            return False

        # LSE shape validation
        return logsumexp.ndim == 3 and logsumexp.shape == (batch_size, q_heads, seq_len_q)

    def _execute_impl(
        self,
        grad_out: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: torch.Tensor,
        grad_input_mask: list[bool],
        out: torch.Tensor,
        logsumexp: torch.Tensor,
        cum_seq_q: torch.Tensor,
        cum_seq_k: torch.Tensor,
        max_q: int,
        max_k: int,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        philox_seed: torch.Tensor | None = None,
        philox_offset: torch.Tensor | None = None,
        scale: float | None = None,
    ) -> ExecutionResult:
        """
        Execute the JAX scaled dot-product attention backward operation
        """

        try:
            batch_size, q_heads, seq_len_q, head_dim = query.shape
            _, kv_heads, seq_len_k, _ = key.shape

            # Calculate scale if not provided
            if scale is None:
                scale = 1.0 / math.sqrt(head_dim)

            # Handle GQA by repeating KV heads to match query heads
            if q_heads != kv_heads:
                repeat_factor = q_heads // kv_heads
                key = key.repeat_interleave(repeat_factor, dim=1)
                value = value.repeat_interleave(repeat_factor, dim=1)

            dropout_mask_shape = (batch_size, q_heads, seq_len_q, seq_len_k)
            # Use global RNG state from forward pass to generate identical dropout_mask
            dropout_mask = generate_dropout_mask(
                dropout_mask_shape,
                dropout_p,
                query.device,
                philox_seed,
                philox_offset,
                backward=True,
            )

            # Call the XLA kernel
            grad_q, grad_k, grad_v, grad_attn_bias = self.kernel(
                grad_out,
                query,
                key,
                value,
                attn_bias,
                dropout_mask,
                logsumexp,
                scale,
                is_causal,
                dropout_p,
                True,  # mixed_precision=True
            )

            # Handle GQA for gradients - sum over repeated heads
            if q_heads != kv_heads:
                repeat_factor = q_heads // kv_heads
                # Reshape and sum grad_k and grad_v
                grad_k = grad_k.view(batch_size, kv_heads, repeat_factor, seq_len_k, head_dim)
                grad_k = grad_k.sum(dim=2)

                grad_v = grad_v.view(batch_size, kv_heads, repeat_factor, seq_len_k, head_dim)
                grad_v = grad_v.sum(dim=2)

            return ExecutionResult(success=True, output=(grad_q, grad_k, grad_v, grad_attn_bias))

        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))

    @property
    def priority(self) -> int:
        return super().priority - 10
