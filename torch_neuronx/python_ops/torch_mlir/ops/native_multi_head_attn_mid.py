"""MLIR implementation of native_multi_head_attn_mid"""

import torch

from ...base import ExecutionResult
from .scaled_dot_product_attention import ScaledDotProductAttnMLIRImpl


class NativeMultiHeadAttnMidMLIRImpl(ScaledDotProductAttnMLIRImpl):
    """Implement the middle attention stage with Torch-MLIR.

    Args:
        op_name (str): Operation name passed to the attention kernel.
    """

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
        """Check whether the middle attention stage can handle the inputs.

        Args:
            query (torch.Tensor): Query in transposed middle-stage format.
            key (torch.Tensor): Key in transposed middle-stage format.
            value (torch.Tensor): Value in transposed middle-stage format.
            use_causal_mask (bool, optional): Apply causal masking. Defaults to ``False``.
            dropout_p (float, optional): Dropout probability. Defaults to ``0.0``.
            training (bool, optional): Training mode flag. Defaults to ``False``.
            out (torch.Tensor | None, optional): Output tensor. Defaults to ``None``.

        Returns:
            bool: ``True`` when the base attention implementation accepts the inputs.

        Raises:
            RuntimeError: If key/value heads do not divide query heads.

        Note:
            ``training`` and ``out`` do not affect capability checks.
        """
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
        """Execute middle-stage attention after transposing its inputs.

        Args:
            query (torch.Tensor): Query in ``[B, H_q, D_q, S_q]`` middle-stage layout.
            key (torch.Tensor): Key in ``[B, H_kv, D_q, S_k]`` middle-stage layout.
            value (torch.Tensor): Value in ``[B, H_kv, D_v, S_k]`` middle-stage layout.
            use_causal_mask (bool, optional): Apply causal masking. Defaults to ``False``.
            dropout_p (float, optional): Dropout probability. Defaults to ``0.0``.
            training (bool, optional): Training mode flag. Defaults to ``False``.
            out (torch.Tensor | None, optional): ``[B, H_q, S_q, D_v]`` destination
                copied in place and returned. Defaults to ``None``.

        Returns:
            ExecutionResult: Success with a ``[B, H_q, S_q, D_v]`` attention output,
                or captured execution failure.

        Note:
            ``training`` is accepted but does not alter execution.
        """
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
