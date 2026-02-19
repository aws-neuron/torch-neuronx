"""MLIR implementation of native_multi_head_attn_prefix"""

import torch

from ...base import ExecutionResult, OperationImplementation
from ..kernel import TorchMlirKernel


def compute_prefix(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    num_heads: int,
    *,
    out=None,
):
    """
    QKV projection and transformation to multi-head format.

    Args:
        query, key, value: (batch, seq_len, d_model)
        qkv_weight: (3*d_model, d_model)
        qkv_bias: (3*d_model,)
        num_heads: number of attention heads
        out: output tensor (excluded from tracing)

    Returns:
        q, k, v: (batch, num_heads, d_head, seq_len)
    """
    b, t, d_model = query.shape
    d_head = d_model // num_heads

    # Fuse inputs: [3*b*t, d_model]
    qkv_in = torch.cat(
        [query.reshape(-1, d_model), key.reshape(-1, d_model), value.reshape(-1, d_model)], dim=0
    )

    # Single projection: [3*b*t, 3*d_model]
    proj = torch.nn.functional.linear(qkv_in, qkv_weight, qkv_bias)

    # Split back: [3, b*t, 3*d_model]
    proj = proj.reshape(3, b * t, 3 * d_model)

    # Extract q, k, v components
    q = proj[0, :, :d_model].reshape(b, t, d_model)
    k = proj[1, :, d_model : 2 * d_model].reshape(b, t, d_model)
    v = proj[2, :, 2 * d_model :].reshape(b, t, d_model)

    # Split heads and transpose: [b, t, d_model] -> [b, num_heads, d_head, seq_len]
    def split_heads(x):
        x = x.reshape(b, t, num_heads, d_head)  # [b, t, h, d_h]
        x = x.permute(0, 2, 3, 1)  # [b, h, d_h, t]
        return x

    return split_heads(q), split_heads(k), split_heads(v)


class NativeMultiHeadAttnPrefixMLIRImpl(OperationImplementation):
    """MLIR implementation of prefix operation"""

    def __init__(self, op_name: str):
        self.kernel = TorchMlirKernel(
            compute_prefix,
            op_name,
            static_argnums=(5,),  # num_heads
            output_params=("out",),
        )
        self.op_name = op_name

    def can_handle(self, *args, **kwargs) -> bool:
        if not super().can_handle(*args, **kwargs):
            return False

        if len(args) != 5:
            return False

        query, key, value, qkv_weight, qkv_bias = args

        if not all(
            tensor.device.type == "neuron" for tensor in [query, key, value, qkv_weight, qkv_bias]
        ):
            return False

        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            return False

        if qkv_weight.ndim != 2 or qkv_bias.ndim != 1:
            return False

        if query.shape != key.shape or query.shape != value.shape:
            return False

        batch, seq_len, d_model = query.shape

        if qkv_weight.shape != (3 * d_model, d_model):
            return False

        if qkv_bias.shape != (3 * d_model,):
            return False

        num_heads = kwargs.get("num_heads", 8)
        return d_model % num_heads == 0

    def _execute_impl(
        self, query, key, value, qkv_weight, qkv_bias, *, num_heads=8, out=None
    ) -> ExecutionResult:
        try:
            result = self.kernel(query, key, value, qkv_weight, qkv_bias, num_heads, out=out)

            if not isinstance(result, tuple) or len(result) != 3:
                raise ValueError(f"Expected tuple of 3 tensors, got {type(result)}")

            return ExecutionResult(success=True, output=result)

        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))

    def _check_and_handle_empty(self, *args, **kwargs) -> ExecutionResult | None:
        if len(args) >= 3:
            query, key, value = args[0], args[1], args[2]
            if query.numel() == 0 or key.numel() == 0 or value.numel() == 0:
                return ExecutionResult(
                    success=False,
                    error_msg="Attention operations do not support empty tensors. "
                    "Please ensure all input tensors have non-zero elements.",
                )
        return None
