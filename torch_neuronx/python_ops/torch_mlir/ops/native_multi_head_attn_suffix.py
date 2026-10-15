"""MLIR implementation of native_multi_head_attn_suffix"""

import torch

from ...base import ExecutionResult, OperationImplementation
from ..kernel import TorchMlirKernel


def compute_suffix(
    attn_output: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
    *,
    out=None,
):
    """Merge attention heads and apply the output projection.

    Args:
        attn_output (torch.Tensor): Attention output shaped
            ``(batch, num_heads, seq_len, d_head)``.
        proj_weight (torch.Tensor): Projection weight shaped ``(d_model, d_model)``.
        proj_bias (torch.Tensor): Projection bias shaped ``(d_model,)``.
        out (torch.Tensor | None, optional): Accepted for output-parameter compatibility but not
            inspected. Defaults to ``None``.

    Returns:
        torch.Tensor: Projected tensor shaped ``(batch, seq_len, d_model)``.
    """
    batch, num_heads, seq_len, d_head = attn_output.shape
    d_model = num_heads * d_head

    # [batch, num_heads, seq_len, d_head] -> [batch, seq_len, num_heads, d_head]
    attn_output = attn_output.permute(0, 2, 1, 3)

    # [batch, seq_len, num_heads, d_head] -> [batch, seq_len, d_model]
    attn_output = attn_output.reshape(batch, seq_len, d_model)

    # Apply output projection
    output = torch.nn.functional.linear(attn_output, proj_weight, proj_bias)

    return output


class NativeMultiHeadAttnSuffixMLIRImpl(OperationImplementation):
    """Implement the attention-suffix projection with Torch-MLIR.

    Args:
        op_name (str): Operation name used to compile the kernel.
    """

    def __init__(self, op_name: str):
        self.kernel = TorchMlirKernel(
            compute_suffix,
            op_name,
            output_params=("out",),
        )
        self.op_name = op_name

    def can_handle(self, *args, **kwargs) -> bool:
        """Check whether the attention-suffix arguments satisfy kernel constraints.

        Args:
            *args (Any): Attention output, projection weight, and projection bias tensors.
            **kwargs (Any): Options; ``num_heads`` defaults to ``8``.

        Returns:
            bool: ``True`` for compatible ranks, shapes, head counts, and tensors on a
                Neuron device.

        Raises:
            RuntimeError: If a non-scalar tensor is not on a Neuron device.
        """
        if not super().can_handle(*args, **kwargs):
            return False

        if len(args) != 3:
            return False

        attn_output, proj_weight, proj_bias = args

        if not all(
            tensor.device.type == "neuron" for tensor in [attn_output, proj_weight, proj_bias]
        ):
            return False

        if attn_output.ndim != 4:
            return False

        if proj_weight.ndim != 2 or proj_bias.ndim != 1:
            return False

        batch, num_heads_actual, seq_len, d_head = attn_output.shape
        num_heads = kwargs.get("num_heads", 8)
        d_model = num_heads * d_head

        if proj_weight.shape != (d_model, d_model):
            return False

        if proj_bias.shape != (d_model,):
            return False

        return num_heads == num_heads_actual

    def _execute_impl(
        self, attn_output, proj_weight, proj_bias, *, num_heads=8, out=None
    ) -> ExecutionResult:
        try:
            result = self.kernel(attn_output, proj_weight, proj_bias, out=out)
            return ExecutionResult(success=True, output=result)

        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))

    def _check_and_handle_empty(self, *args, **kwargs) -> ExecutionResult | None:
        if len(args) >= 3:
            attn_output, proj_weight, proj_bias = args[0], args[1], args[2]
            if attn_output.numel() == 0 or proj_weight.numel() == 0 or proj_bias.numel() == 0:
                return ExecutionResult(
                    success=False,
                    error_msg="Attention operations do not support empty tensors. "
                    "Please ensure all input tensors have non-zero elements.",
                )
        return None
