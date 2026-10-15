"""MLIR implementation of native_multi_head_attn_mid"""

import torch

from ...base import ExecutionResult
from .scaled_dot_product_attention import ScaledDotProductAttnMLIRImpl


class NativeMultiHeadAttnMidMLIRImpl(ScaledDotProductAttnMLIRImpl):
    """MLIR implementation for Mid operation with transposed inputs"""

    def __init__(self, op_name: str):
        super().__init__(op_name)

    def can_handle(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        use_causal_mask: bool = False,
        dropout_p: float = 0.0,
        training: bool = False,
        out: torch.Tensor | None = None,
    ) -> bool:
        """Check if MLIR implementation can handle Mid operation parameters"""
        return super().can_handle(
            query,
            key,
            value,
            is_causal=use_causal_mask,
            dropout_p=dropout_p,
        )

    def execute(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        use_causal_mask: bool = False,
        dropout_p: float = 0.0,
        training: bool = False,
        out: torch.Tensor | None = None,
    ) -> ExecutionResult:
        """Execute with input transposition for Mid format"""
        # Transpose inputs from Mid format (batch, heads, embed_dim, seq_len)
        # to standard format (batch, heads, seq_len, embed_dim)
        q_transposed = query.transpose(-2, -1)
        k_transposed = key.transpose(-2, -1)
        v_transposed = value.transpose(-2, -1)

        # Call base MLIR implementation with mapped parameter names
        result = super()._execute_impl(
            q_transposed,
            k_transposed,
            v_transposed,
            is_causal=use_causal_mask,
            dropout_p=dropout_p,
        )

        if not result.success:
            return result

        # Extract just the attention output (first element of tuple)
        attn_output = result.output[0] if isinstance(result.output, tuple) else result.output

        # Output is already in correct format (batch, heads, seq_len, embed_dim)
        if out is not None:
            out.copy_(attn_output)
            return ExecutionResult(success=True, output=out)

        return ExecutionResult(success=True, output=attn_output)
