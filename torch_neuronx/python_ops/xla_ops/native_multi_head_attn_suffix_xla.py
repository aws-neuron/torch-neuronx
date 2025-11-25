import jax.numpy as jnp
import torch

from ...kernels.xla_kernel import TorchNeuronXLAKernel
from ..base import ExecutionResult, OperationImplementation


class NativeMultiHeadAttnSuffixXLAImpl(OperationImplementation):
    """XLA implementation for native multi-head attention suffix (output projection)"""

    def __init__(self, op_name: str):
        # Define the JAX computation
        def transform_and_project(
            attn_output: jnp.ndarray,
            proj_weight: jnp.ndarray,
            proj_bias: jnp.ndarray,
        ) -> jnp.ndarray:
            """
            Transform attention output from multi-head format and apply projection.

            Args
            ----
            attn_output : (batch, num_heads, seq_len, d_head)
            proj_weight : (d_model, d_model)
            proj_bias   : (d_model,)

            Returns
            -------
            output      : (batch, seq_len, d_model)
            """
            batch, num_heads, seq_len, d_head = attn_output.shape
            # Compute d_model from the shape information
            d_model = num_heads * d_head

            # Reshape from multi-head to concatenated format
            # [batch, num_heads, seq_len, d_head] -> [batch, seq_len, num_heads, d_head]
            attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))

            # [batch, seq_len, num_heads, d_head] -> [batch, seq_len, d_model]
            attn_output = attn_output.reshape(batch, seq_len, d_model)

            # Apply output projection
            # [batch, seq_len, d_model] @ [d_model, d_model]^T + [d_model]
            output = attn_output @ proj_weight.T + proj_bias

            return output

        # Create the kernel for single-output operation without static arguments
        # Arguments are: attn_output(0), proj_weight(1), proj_bias(2)
        # No static arguments needed since we derive everything from shapes
        self.kernel = TorchNeuronXLAKernel(transform_and_project, op_name)

    def can_handle(self, *args, **kwargs) -> bool:
        """Check if this implementation can handle the given inputs"""
        if not super().can_handle(*args, **kwargs):
            return False

        # Expecting 3 positional args: attn_output, proj_weight, proj_bias
        # And num_heads as keyword argument
        if len(args) != 3:
            return False

        attn_output, proj_weight, proj_bias = args

        # Must be on Neuron device
        if not all(
            tensor.device.type == "neuron" for tensor in [attn_output, proj_weight, proj_bias]
        ):
            return False

        # Check shapes
        if attn_output.ndim != 4:
            return False

        if proj_weight.ndim != 2 or proj_bias.ndim != 1:
            return False

        batch, num_heads_actual, seq_len, d_head = attn_output.shape
        num_heads = kwargs.get("num_heads", 8)
        d_model = num_heads * d_head

        # Check weight/bias dimensions
        if proj_weight.shape != (d_model, d_model):
            return False

        if proj_bias.shape != (d_model,):
            return False

        # Check that num_heads matches the actual heads dimension
        return num_heads == num_heads_actual

    def _execute_impl(
        self,
        attn_output: torch.Tensor,
        proj_weight: torch.Tensor,
        proj_bias: torch.Tensor,
        *,
        num_heads: int = 8,
        out=None,
    ) -> ExecutionResult:
        """Execute the multi-head attention suffix operation"""
        try:
            # We don't need to pass num_heads to the kernel since it can be
            # derived from attn_output shape
            # The kernel only takes 3 arguments: attn_output, proj_weight, proj_bias
            result = self.kernel(attn_output, proj_weight, proj_bias, output=out)

            # Result should be a single tensor
            if isinstance(result, torch.Tensor):
                return ExecutionResult(success=True, output=result)
            else:
                raise ValueError(f"Expected torch.Tensor, got {type(result)}")

        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))

    def _check_and_handle_empty(self, *args, **kwargs) -> ExecutionResult | None:
        """Check for empty tensors and reject them"""
        # Extract the main tensor arguments
        if len(args) >= 3:
            attn_output, proj_weight, proj_bias = args[0], args[1], args[2]
            if attn_output.numel() == 0 or proj_weight.numel() == 0 or proj_bias.numel() == 0:
                return ExecutionResult(
                    success=False,
                    error_msg="Attention operations do not support empty tensors. "
                    "Please ensure all input tensors have non-zero elements.",
                )
        return None
