"""Native multi-head attention middle operation implementation"""

import torch

from .base import ExecutionResult, Operation, OperationImplementation
from .nki_kernels.native_multi_head_attn_mid import native_multi_head_attn_mid_kernel_with_grid


class NativeMultiHeadAttnMidNKIImpl(OperationImplementation):
    """NKI implementation of native multi-head attention middle operation"""

    def can_handle(self, query, key, value, **kwargs) -> bool:
        """Check if this implementation can handle the given inputs"""
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

            # Create output tensor if not provided
            if out is None:
                attn_output = torch.empty(
                    batch_size, num_heads, seq_len, d_head, dtype=query.dtype, device=query.device
                )
            else:
                attn_output = out
                # Validate output shape
                if attn_output.shape != (batch_size, num_heads, seq_len, d_head):
                    raise ValueError(
                        f"Output tensor shape {attn_output.shape} doesn't match expected "
                        f"shape ({batch_size}, {num_heads}, {seq_len}, {d_head})"
                    )

            # Call the NKI kernel
            native_multi_head_attn_mid_kernel_with_grid(
                query,
                key,
                value,
                attn_output,
                use_causal_mask=use_causal_mask,
                dropout_p=dropout_p,
                training=training,
            )

            return ExecutionResult(success=True, output=attn_output)

        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))


class NativeMultiHeadAttnMidOp(Operation):
    """Native multi-head attention middle operation with NKI implementation"""

    def _setup_implementations(self):
        self._implementations.append(NativeMultiHeadAttnMidNKIImpl())
        from .torch_mlir.ops.native_multi_head_attn_mid import NativeMultiHeadAttnMidMLIRImpl

        self._implementations.append(NativeMultiHeadAttnMidMLIRImpl(self.op_name))

    @property
    def op_name(self) -> str:
        return "native_multi_head_attn_mid"
