import jax.numpy as jnp
import torch

from ...kernels.xla_kernel import TorchNeuronXLAKernel
from ..base import ExecutionResult, OperationImplementation


class NativeMultiHeadAttnPrefixXLAImpl(OperationImplementation):
    """XLA implementation for native multi-head attention with prefix projection"""

    def __init__(self, op_name: str):
        # Define the JAX computation
        def transform_qkv(
            query: jnp.ndarray,
            key: jnp.ndarray,
            value: jnp.ndarray,
            qkv_weight: jnp.ndarray,
            qkv_bias: jnp.ndarray,
            num_heads: int,
        ):
            """
            Args
            ----
            query, key, value : (batch, seq_len, d_model)
            qkv_weight        : (3*d_model, d_model)
            qkv_bias          : (3*d_model,)
            num_heads         : attention heads (must divide d_model)

            Returns
            -------
            q, k, v           : (batch, num_heads, d_head, seq_len)
            """
            b, t, d_model = query.shape
            # Note: We cannot use assert with traced values
            # The caller should ensure d_model % num_heads == 0
            d_head = d_model // num_heads

            # ------------------------------------------------------------------
            # 1.  Fuse the three inputs into one tall matrix  [3*b*t, d_model]
            # ------------------------------------------------------------------
            qkv_in = jnp.concatenate(
                (query.reshape(-1, d_model), key.reshape(-1, d_model), value.reshape(-1, d_model)),
                axis=0,
            )

            # ------------------------------------------------------------------
            # 2.  Single linear projection  y = x @ Wᵀ + b
            #     Equivalent to three independent projections.
            # ------------------------------------------------------------------
            proj = qkv_in @ qkv_weight.T + qkv_bias  # [3*b*t, 3*d_model]

            # ------------------------------------------------------------------
            # 3.  Split back into Q, K, V and reshape for heads
            # ------------------------------------------------------------------
            # First reshape to separate the 3 matrices
            proj = proj.reshape(3, b * t, 3 * d_model)  # [3, b*t, 3*d_model]

            # Now split each into q, k, v components
            q_all = proj[0]  # [b*t, 3*d_model] for query
            k_all = proj[1]  # [b*t, 3*d_model] for key
            v_all = proj[2]  # [b*t, 3*d_model] for value

            # Each contains its own q, k, v projection - extract the relevant part
            q = q_all[:, :d_model].reshape(b, t, d_model)  # [b, t, d_model]
            k = k_all[:, d_model : 2 * d_model].reshape(b, t, d_model)  # [b, t, d_model]
            v = v_all[:, 2 * d_model :].reshape(b, t, d_model)  # [b, t, d_model]

            def split_heads(x):
                x = x.reshape(b, t, num_heads, d_head)  # [b, t, h, d_h]
                x = jnp.transpose(x, (0, 2, 3, 1))  # [b, h, d_h, t]
                return x

            return split_heads(q), split_heads(k), split_heads(v)

        # Create the kernel for multi-output operation with num_heads as static argument
        # Arguments are: query(0), key(1), value(2), qkv_weight(3), qkv_bias(4), num_heads(5)
        self.kernel = TorchNeuronXLAKernel(transform_qkv, op_name, static_argnums=(5,))

    def can_handle(self, *args, **kwargs) -> bool:
        """Check if this implementation can handle the given inputs"""
        if not super().can_handle(*args, **kwargs):
            return False

        # Expecting 5 positional args: query, key, value, qkv_weight, qkv_bias
        # And num_heads as keyword argument
        if len(args) != 5:
            return False

        query, key, value, qkv_weight, qkv_bias = args

        # Must be on Neuron device
        if not all(
            tensor.device.type == "neuron" for tensor in [query, key, value, qkv_weight, qkv_bias]
        ):
            return False

        # Check shapes
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            return False

        if qkv_weight.ndim != 2 or qkv_bias.ndim != 1:
            return False

        # Ensure query, key, value have same shape
        if query.shape != key.shape or query.shape != value.shape:
            return False

        batch, seq_len, d_model = query.shape

        # Check weight/bias dimensions
        if qkv_weight.shape != (3 * d_model, d_model):
            return False

        if qkv_bias.shape != (3 * d_model,):
            return False

        # Check num_heads if provided
        num_heads = kwargs.get("num_heads", 8)
        return d_model % num_heads == 0

    def _execute_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        qkv_weight: torch.Tensor,
        qkv_bias: torch.Tensor,
        *,
        num_heads: int = 8,
        out=None,
    ) -> ExecutionResult:
        """Execute the multi-head attention prefix operation"""
        try:
            # Execute the kernel - num_heads is passed as a regular argument
            # The multi-output kernel returns a tuple of (q, k, v)
            result = self.kernel(query, key, value, qkv_weight, qkv_bias, num_heads, output=out)

            # Result should be a tuple of (q_out, k_out, v_out)
            if isinstance(result, tuple) and len(result) == 3:
                return ExecutionResult(success=True, output=result)
            else:
                raise ValueError(f"Expected tuple of 3 tensors, got {type(result)}")

        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))

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
