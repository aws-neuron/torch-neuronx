"""Native multi-head attention middle operation implementation"""

import torch

from .base import ExecutionResult, Operation, OperationImplementation
from .nki_kernels.scaled_dot_product_attention import scaled_dot_product_attention_kernel


class NativeMultiHeadAttnMidNKIImpl(OperationImplementation):
    """Implement the middle native multi-head attention stage with NKI."""

    def can_handle(self, query, key, value, **kwargs) -> bool:
        """Check whether NKI supports the supplied attention tensors.

        Args:
            query (torch.Tensor): Query tensor in ``[B, H, D, S]`` layout.
            key (torch.Tensor): Key tensor matching ``query``.
            value (torch.Tensor): Value tensor matching ``query``.
            **kwargs (Any): Additional compatibility arguments for the base handler.

        Returns:
            bool: ``True`` for matching four-dimensional Neuron tensors with
            ``S % 512 == 0``.

        Raises:
            RuntimeError: If a non-scalar tensor argument is not on a Neuron device.
        """
        if not super().can_handle(query, key, value, **kwargs):
            return False

        # Must be on Neuron device
        if not all(tensor.device.type == "neuron" for tensor in [query, key, value]):
            return False

        # Check shapes - all must be 4D tensors
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            return False

        # Extract dimensions
        batch_size, num_heads, d_head, seq_len = query.shape

        # Validate key shape matches query
        if key.shape != (batch_size, num_heads, d_head, seq_len):
            return False

        # Validate value shape - we expect V in the same format as Q and K
        if value.shape != (batch_size, num_heads, d_head, seq_len):
            return False

        # Check sequence length constraint - must be multiple of 512
        return seq_len % 512 == 0

    def _check_and_handle_empty(self, *args, **kwargs) -> ExecutionResult | None:
        """Check for empty tensors and reject them"""
        # Extract the main tensor arguments
        if len(args) >= 3:
            query, key, value = args[0], args[1], args[2]
            if query.numel() == 0 or key.numel() == 0 or value.numel() == 0:
                return ExecutionResult(
                    success=False,
                    error_msg="Attention operations do not support empty tensors. "
                    "Please ensure all input tensors have non-zero elements.",
                )
        return None

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
        # Check for empty tensors first
        """Execute attention after rejecting empty input tensors.

        Args:
            query (torch.Tensor): Query tensor in ``[B, H, D, S]`` layout.
            key (torch.Tensor): Key tensor matching ``query``.
            value (torch.Tensor): Value tensor matching ``query``.
            use_causal_mask (bool, optional): Apply a causal attention mask.
                Defaults to ``False``.
            dropout_p (float, optional): Compatibility dropout probability; the
                current NKI wrapper ignores it. Defaults to ``0.0``.
            training (bool, optional): Enable training behavior in the NKI kernel.
                Defaults to ``False``.
            out (torch.Tensor | None, optional): Destination passed to ``copy_`` and
                returned when the copy succeeds. Defaults to ``None``.

        Returns:
            ExecutionResult: Failure details, the computed ``[B, H, S, D]`` output,
            or the supplied ``out`` after a successful copy.
        """
        empty_result = self._check_and_handle_empty(query, key, value)
        if empty_result is not None:
            return empty_result

        # Normal execution for non-empty tensors
        return self._execute_impl(
            query,
            key,
            value,
            use_causal_mask=use_causal_mask,
            dropout_p=dropout_p,
            training=training,
            out=out,
        )

    def _execute_impl(
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
        """Execute the multi-head attention middle operation - only for non-empty tensors"""
        try:
            batch_size, num_heads, d_head, seq_len = query.shape

            # Call the NKI kernel
            attn_output = scaled_dot_product_attention_kernel(
                query.transpose(-2, -1),
                key.transpose(-2, -1),
                value.transpose(-2, -1),
                is_causal=use_causal_mask,
                dropout_p=dropout_p,
                scale=None,
                training=training,
                seed=None,
                lse=None,
            )
            if out is not None:
                # Validate output shape
                if attn_output.shape != (batch_size, num_heads, seq_len, d_head):
                    raise ValueError(
                        f"Output tensor shape {attn_output.shape} doesn't match expected "
                        f"shape ({batch_size}, {num_heads}, {seq_len}, {d_head})"
                    )

                out.copy_(attn_output)
                attn_output = out

            return ExecutionResult(success=True, output=attn_output)

        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))


class NativeMultiHeadAttnMidOp(Operation):
    """Dispatch the middle native multi-head attention operation."""

    def _setup_implementations(self):
        self._implementations.append(NativeMultiHeadAttnMidNKIImpl())
        from .torch_mlir.ops.native_multi_head_attn_mid import NativeMultiHeadAttnMidMLIRImpl

        self._implementations.append(NativeMultiHeadAttnMidMLIRImpl(self.op_name))

    @property
    def op_name(self) -> str:
        """Return the registered operation name.

        Returns:
            str: ``native_multi_head_attn_mid``.
        """
        return "native_multi_head_attn_mid"
