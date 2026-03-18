"""
Fused scaled dot-product attention implementation for Neuron devices.

This implements the overrideable fused attention interface which allows
custom backends to provide their own optimized attention implementations.
"""

import math

import torch

from torch_neuronx.neuron_dynamo_backend.decompositions import (
    _can_use_nki_flash_attention,
    _can_use_nki_flash_attention_backward,
)

from .base import AttentionOpImpl, ExecutionResult, Operation


class ScaledDotProductFusedAttentionNKIImpl(AttentionOpImpl):
    """NKI implementation of fused scaled dot-product attention (returns tuple for training)"""

    def can_handle(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        return_debug_mask: bool = False,
        scale: float | None = None,
    ) -> bool:
        """Check if NKI implementation can handle the given inputs"""
        if not super().can_handle(
            query, key, value, attn_bias, dropout_p, is_causal, return_debug_mask, scale
        ):
            return False

        return _can_use_nki_flash_attention(query, key, value, attn_bias, dropout_p=dropout_p)

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
        """Execute the fused scaled dot-product attention operation"""
        try:
            from .nki_kernels.scaled_dot_product_attention import (
                scaled_dot_product_attention_kernel,
            )

            # Extract dimensions
            batch_size, q_heads, seq_len_q, embed_dim = query.shape
            _, kv_heads, seq_len_k, _ = key.shape
            _, _, _, embed_v = value.shape

            # Calculate scale if not provided
            if scale is None:
                scale = 1.0 / math.sqrt(embed_dim)

            # Determine if we're in training mode
            training = dropout_p > 0.0 or torch.is_grad_enabled()

            # Prepare RNG state for dropout
            if dropout_p > 0.0:
                # Generate philox seed and offset for reproducible dropout
                philox_seed = torch.randint(
                    0, 2**31, (1, 1), dtype=torch.int32, device=query.device
                )
                philox_offset = torch.zeros((1, 1), dtype=torch.int32, device=query.device)
            else:
                # Empty tensors when no dropout
                philox_seed = torch.empty((0,), dtype=torch.int32, device=query.device)
                philox_offset = torch.empty((0,), dtype=torch.int32, device=query.device)

            # Allocate LSE buffer for training/backward pass
            if training:
                # LSE shape: (batch, q_heads, B_P_SIZE, n_tile_q) where B_P_SIZE=128
                # and n_tile_q = seq_len_q // B_P_SIZE
                b_p_size = 128
                n_tile_q = seq_len_q // b_p_size
                lse = torch.empty(
                    batch_size,
                    q_heads,
                    b_p_size,
                    n_tile_q,
                    dtype=torch.float32,
                    device=query.device,
                )
            else:
                lse = torch.empty((0,), dtype=torch.float32, device=query.device)

            # Create cumulative sequence tensors, using None since they are not used
            # in backward.
            cum_seq_q = None
            cum_seq_k = None

            # Max sequence lengths
            max_q = seq_len_q
            max_k = seq_len_k

            # Debug mask (not supported)
            debug_attn_mask = None

            # Call the NKI kernel — inputs in ATen format, kernel handles transpose and GQA
            attn_output = scaled_dot_product_attention_kernel(
                query,
                key,
                value,
                is_causal=is_causal,
                dropout_p=dropout_p,
                scale=scale,
                training=training,
                seed=philox_seed if dropout_p > 0.0 else None,
                lse=lse if training else None,
            )

            # Check if this is the single-tensor version
            if hasattr(self, "_returns_single_tensor"):
                # For aten::scaled_dot_product_attention, only return the attention output
                return ExecutionResult(success=True, output=attn_output)
            else:
                # Return tuple matching the overrideable API
                result = (
                    attn_output,
                    lse,
                    cum_seq_q,
                    cum_seq_k,
                    max_q,
                    max_k,
                    philox_seed,
                    philox_offset,
                    debug_attn_mask,
                )
                return ExecutionResult(success=True, output=result)

        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))


class ScaledDotProductFusedAttentionOp(Operation):
    """Fused scaled dot-product attention operation with NKI implementation"""

    def _setup_implementations(self):
        self._implementations.append(ScaledDotProductFusedAttentionNKIImpl())
        from .torch_mlir.ops.scaled_dot_product_attention import ScaledDotProductAttnMLIRImpl

        self._implementations.append(ScaledDotProductAttnMLIRImpl(self.op_name))

    @property
    def op_name(self) -> str:
        return "_scaled_dot_product_fused_attention_overrideable"

    def _unhandled_cpu_fallback(self, *args, **kwargs):
        # remap altered kwargs to original names - still breaks due to incompatible outputs
        kwargs["attn_mask"] = kwargs.pop("attn_bias", None)
        kwargs.pop("return_debug_mask", None)
        super()._unhandled_cpu_fallback(*args, **kwargs)


# Create singleton instance
_fused_sdpa_op = ScaledDotProductFusedAttentionOp()


def scaled_dot_product_fused_attention_overrideable_neuron(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    scale: float | None = None,
) -> tuple[torch.Tensor, ...]:
    """
    Implements the overrideable fused attention interface for Neuron devices.

    This is called by PyTorch when it needs a custom fused attention implementation
    for training or when dropout is enabled.

    Returns:
        Tuple of (output, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k,
                  philox_seed, philox_offset, debug_attn_mask)
    """
    # Validate device
    if query.device.type != "neuron":
        raise ValueError(f"Expected Neuron device, got {query.device}")

    if return_debug_mask:
        raise NotImplementedError("Debug mask not supported in Neuron implementation.")

    # Use the operation
    result = _fused_sdpa_op(
        query,
        key,
        value,
        attn_bias=attn_bias,
        dropout_p=dropout_p,
        is_causal=is_causal,
        return_debug_mask=return_debug_mask,
        scale=scale,
    )

    return result


class ScaledDotProductAttentionBackwardNKIImpl(AttentionOpImpl):
    """NKI implementation of scaled dot-product attention backward"""

    def can_handle(self, *args, **kwargs) -> bool:
        """Check if this implementation can handle the given inputs"""
        query = args[1] if len(args) > 1 else kwargs.get("query")
        key = args[2] if len(args) > 2 else kwargs.get("key")
        value = args[3] if len(args) > 3 else kwargs.get("value")
        attn_bias = args[4] if len(args) > 4 else kwargs.get("attn_bias")
        dropout_p = args[12] if len(args) > 12 else kwargs.get("dropout_p", 0.0)

        if not super().can_handle(query, key, value, attn_bias):
            return False

        return _can_use_nki_flash_attention_backward(query, key, value, attn_bias, dropout_p)

    def _execute_impl(self, *args, **kwargs) -> ExecutionResult:
        """Execute the backward pass"""
        # Extract all arguments
        grad_out = args[0] if len(args) > 0 else kwargs.get("grad_out")
        query = args[1] if len(args) > 1 else kwargs.get("query")
        key = args[2] if len(args) > 2 else kwargs.get("key")
        value = args[3] if len(args) > 3 else kwargs.get("value")
        attn_bias = args[4] if len(args) > 4 else kwargs.get("attn_bias")
        # grad_input_mask = args[5] if len(args) > 5 else kwargs.get("grad_input_mask")  # unused
        out = args[6] if len(args) > 6 else kwargs.get("out")
        logsumexp = args[7] if len(args) > 7 else kwargs.get("logsumexp")
        # cum_seq_q = args[8] if len(args) > 8 else kwargs.get("cum_seq_q")  # unused
        # cum_seq_k = args[9] if len(args) > 9 else kwargs.get("cum_seq_k")  # unused
        # max_q = args[10] if len(args) > 10 else kwargs.get("max_q")  # unused
        # max_k = args[11] if len(args) > 11 else kwargs.get("max_k")  # unused
        dropout_p = args[12] if len(args) > 12 else kwargs.get("dropout_p", 0.0)
        is_causal = args[13] if len(args) > 13 else kwargs.get("is_causal", False)
        philox_seed = args[14] if len(args) > 14 else kwargs.get("philox_seed")
        scale = args[16] if len(args) > 16 else kwargs.get("scale")

        try:
            from .nki_kernels.scaled_dot_product_attention_backward import (
                scaled_dot_product_attention_backward_kernel,
            )

            # Check for unsupported features
            if attn_bias is not None:
                return ExecutionResult(
                    success=False, error_msg="Attention bias gradient not yet supported"
                )

            # Calculate scale if not provided
            if scale is None:
                scale = 1.0 / math.sqrt(query.shape[-1])

            # Convert philox seed for NKI kernel (if dropout was used)
            seed = philox_seed if dropout_p > 0.0 and philox_seed.numel() > 0 else None

            # Call the backward kernel — inputs in ATen format, kernel handles transpose
            grad_q, grad_k, grad_v = scaled_dot_product_attention_backward_kernel(
                grad_out=grad_out,
                q=query,
                k=key,
                v=value,
                out=out,
                lse=logsumexp,
                is_causal=is_causal,
                dropout_p=dropout_p,
                scale=scale,
                seed=seed,
            )

            # grad_attn_bias is None since we don't support it yet
            grad_attn_bias = None

            return ExecutionResult(success=True, output=(grad_q, grad_k, grad_v, grad_attn_bias))
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))

    def _check_and_handle_empty(self, *args, **kwargs) -> ExecutionResult | None:
        """Check for empty tensors and reject them"""
        # Extract the main tensor arguments
        if len(args) >= 4:
            grad_out = args[0]
            query = args[1]
            key = args[2]
            value = args[3]
            if (
                grad_out.numel() == 0
                or query.numel() == 0
                or key.numel() == 0
                or value.numel() == 0
            ):
                return ExecutionResult(
                    success=False,
                    error_msg="Attention backward operations do not support empty tensors. "
                    "Please ensure all input tensors have non-zero elements.",
                )
        return None


class ScaledDotProductAttentionBackwardOp(Operation):
    """Scaled dot-product attention backward operation with NKI implementation"""

    def _setup_implementations(self):
        self._implementations.append(ScaledDotProductAttentionBackwardNKIImpl())
        from .torch_mlir.ops.scaled_dot_product_attention import (
            ScaledDotProductAttnBackwardMLIRImpl,
        )

        self._implementations.append(ScaledDotProductAttnBackwardMLIRImpl(self.op_name))

    @property
    def op_name(self) -> str:
        return "_scaled_dot_product_fused_attention_overrideable_backward"


# Create singleton instance for backward
_sdpa_backward_op = ScaledDotProductAttentionBackwardOp()


def scaled_dot_product_fused_attention_overrideable_backward_neuron(
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
    dropout_p: float,
    is_causal: bool,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    *,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward pass for fused attention using the Operation pattern.

    Returns:
        Tuple of (grad_query, grad_key, grad_value, grad_attn_bias)
    """
    # Validate device
    if query.device.type != "neuron":
        raise ValueError(f"Expected Neuron device, got {query.device}")

    # Use the operation
    return _sdpa_backward_op(
        grad_out,
        query,
        key,
        value,
        attn_bias,
        grad_input_mask,
        out,
        logsumexp,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        philox_seed,
        philox_offset,
        scale=scale,
    )


def scaled_dot_product_fused_attention_overrideable_backward_meta(
    grad_out,
    query,
    key,
    value,
    attn_bias,
    grad_input_mask,
    out,
    logsumexp,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    philox_seed,
    philox_offset,
    scale=None,
):
    """Meta implementation for fake tensor dispatch"""
    grad_query = torch.empty_like(query)
    grad_key = torch.empty_like(key)
    grad_value = torch.empty_like(value)
    grad_attn_bias = torch.empty_like(attn_bias) if attn_bias is not None else None
    return grad_query, grad_key, grad_value, grad_attn_bias
