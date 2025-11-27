import math

import jax
import jax.numpy as jnp
import torch

from ...kernels.xla_kernel import TorchNeuronXLAKernel
from ..base import AttentionOpImpl, ExecutionResult


def generate_dropout_mask(
    dropout_mask_shape, dropout_p, device, philox_seed, philox_offset, backward=False
):
    if dropout_p < 0.0 or dropout_p > 1.0:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got {dropout_p}")

    if dropout_p == 0.0:
        # Do NOT initialize a dropout_mask with all True values
        # because when dropout is 0.0, Jax will optimized away dropout_mask from the graph
        # but we still pass in the dropout_mask which cause incorrect execution and get inf values
        dropout_mask = None
    elif float(dropout_p) == 1.0:
        dropout_mask = torch.zeros(dropout_mask_shape, dtype=torch.bool, device=device)
    else:
        # Generate this on CPU and bring to neuron
        dropout_mask = torch.empty(dropout_mask_shape, dtype=torch.bool)
        keep_prob = 1.0 - float(dropout_p)
        generator = torch.Generator()
        generator.set_state(philox_seed)
        dropout_mask = torch.bernoulli(dropout_mask, keep_prob, generator=generator).to(device)
        if not backward:
            # set rng state to reproduce same dropout mask as running on CPU
            torch.set_rng_state(generator.get_state())

    return dropout_mask


class ScaledDotProductAttnXLAImpl(AttentionOpImpl):
    def __init__(self, op_name: str):
        def compute_attention(
            query: jnp.ndarray,
            key: jnp.ndarray,
            value: jnp.ndarray,
            attn_bias: jnp.ndarray,
            dropout_mask: jnp.ndarray,
            scale: float,
            is_causal: bool,
            dropout_p: float,
            mixed_precision: bool,
        ):
            """
            Pure JAX attention computation: softmax(Q @ K^T / sqrt(d)) @ V
            query: [batch, num_heads, seq_len_q, head_dim]
            key: [batch, num_heads, seq_len_k, head_dim]
            value: [batch, num_heads, seq_len_k, head_dim]
            dropout_mask: [batch, num_heads, seq_len_q, seq_len_k] or empty array
            """

            # Store original dtype for output
            orig_dtype = query.dtype
            acc_dtype = jnp.float32 if mixed_precision else orig_dtype

            # Compute attention logits: Q @ K^T
            logits = jnp.einsum("bhqd,bhkd->bhqk", query, key)

            # upcast for softmax
            logits = logits.astype(acc_dtype)

            # Apply scale
            logits = logits * scale

            # Apply causal mask if needed
            if is_causal:
                seq_len_q, seq_len_k = logits.shape[-2:]
                causal_mask = jnp.tri(seq_len_q, seq_len_k, dtype=bool)
                # Use a large negative value that won't cause overflow
                logits = logits + jnp.where(causal_mask, 0.0, -jnp.inf)
            elif attn_bias is not None:
                logits = logits + attn_bias

            # Compute log-sum-exp for backward pass
            lse = jax.scipy.special.logsumexp(logits, axis=-1)

            # Compute attention probabilities with numerical stability
            probs = jnp.exp(logits - lse[:, :, :, None])

            # Apply dropout if specified
            if dropout_p > 0.0 and dropout_mask.size > 0:
                if dropout_p == 1.0:
                    probs = probs * dropout_mask
                else:
                    probs = probs * dropout_mask / (1.0 - dropout_p)

            # Downcast and apply attention to values
            probs = probs.astype(orig_dtype)
            output = jnp.einsum("bhqk,bhkd->bhqd", probs, value)

            return output, lse.astype(jnp.float32)

        # Create XLA kernel for backup implementation
        self.kernel = TorchNeuronXLAKernel(
            compute_attention,
            op_name,
            static_argnums=(
                5,
                6,
                7,
                8,
            ),  # scale, is_causal, dropout_p, mixed_precision
        )

    def _execute_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        return_debug_mask: bool = False,
        scale: float | None = None,
    ) -> ExecutionResult:
        """
        Execute the JAX scaled dot-product attention operation
        query: [batch, num_heads, seq_len_q, head_dim]
        key:   [batch, num_heads, seq_len_k, head_dim]
        value: [batch, num_heads, seq_len_k, head_dim]
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
                # Expand KV tensors:
                # [batch, kv_heads, seq_len, head_dim] -> [batch, q_heads, seq_len, head_dim]
                key = key.repeat_interleave(repeat_factor, dim=1)
                value = value.repeat_interleave(repeat_factor, dim=1)

            # Determine if we need LSE for backward pass
            training = torch.is_grad_enabled() and (
                query.requires_grad or key.requires_grad or value.requires_grad
            )

            # Generate philox seed and offset for deterministic dropout
            philox_seed = torch.get_rng_state()
            philox_offset = torch.empty((0,), dtype=torch.int32, device=query.device)

            # Generate dropout mask if needed
            dropout_mask_shape = (batch_size, q_heads, seq_len_q, seq_len_k)
            dropout_mask = generate_dropout_mask(
                dropout_mask_shape, dropout_p, query.device, philox_seed, philox_offset
            )

            # Allocate LSE buffer only when needed for backward pass
            if training:
                lse = torch.empty(
                    batch_size,
                    q_heads,
                    seq_len_q,
                    dtype=torch.float32,
                    device=query.device,
                )
            else:
                lse = torch.empty((0,), dtype=torch.float32, device=query.device)

            # Create cumulative sequence tensors (for nested tensors, using dummy values)
            cum_seq_q = None
            cum_seq_k = None

            # Debug mask (not supported)
            debug_attn_mask = None
            # Call kernel to get attn_output and lse
            attn_output, lse = self.kernel(
                query,
                key,
                value,
                attn_bias,
                dropout_mask,
                scale,
                is_causal,
                dropout_p,
                True,
            )

            # Check if this is the single-tensor version
            if hasattr(self, "_returns_single_tensor"):
                return ExecutionResult(success=True, output=attn_output)
            else:
                # Return tuple matching the overrideable API
                result = (
                    attn_output,
                    lse,
                    cum_seq_q,
                    cum_seq_k,
                    seq_len_q,
                    seq_len_k,
                    philox_seed,
                    philox_offset,
                    debug_attn_mask,
                )
                return ExecutionResult(success=True, output=result)

        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))

    @property
    def priority(self) -> int:
        return super().priority - 10
