"""Linear algebra operations."""

import torch

from ..operation_registry import register_aten


@register_aten(["aten::dot", "aten::dot.out"])
def torch_dot(tensor1, tensor2, out=None):
    return torch.dot(tensor1, tensor2)


@register_aten(["aten::mv", "aten::mv.out"])
def torch_mv(mat, vec, out=None):
    return torch.mv(mat, vec)


@register_aten(["aten::mm", "aten::mm.out"])
def torch_mm(mat1, mat2, out=None):
    return torch.mm(mat1, mat2)


@register_aten(["aten::bmm", "aten::bmm.out"])
def torch_bmm(batch1, batch2, out=None):
    return torch.bmm(batch1, batch2)


@register_aten(["aten::addmm", "aten::addmm.out"], static_argnames=("beta", "alpha"))
def torch_addmm(input, mat1, mat2, beta=1, alpha=1, out=None):
    # Validate that mat1 and mat2 are 2D tensors
    if mat1.dim() != 2:
        raise RuntimeError(f"Expected 2D tensor for mat1, got {mat1.dim()}D")
    if mat2.dim() != 2:
        raise RuntimeError(f"Expected 2D tensor for mat2, got {mat2.dim()}D")

    # Check matrix dimension compatibility
    if mat1.size(1) != mat2.size(0):
        raise RuntimeError(
            f"mat1 and mat2 shapes cannot be multiplied "
            f"({mat1.size(0)}x{mat1.size(1)} and {mat2.size(0)}x{mat2.size(1)})"
        )

    # Check that input is broadcastable to result shape
    result_shape = (mat1.size(0), mat2.size(1))

    # Try PyTorch's broadcasting rules
    torch.broadcast_shapes(input.shape, result_shape)

    return torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha)


@register_aten(
    ["aten::addmv", "aten::addmv_", "aten::addmv.out"],
    static_argnames=(
        "beta",
        "alpha",
    ),
)
def torch_addmv(self, mat, vec, beta=1, alpha=1, out=None):
    """Decomposed implementation of addmv: beta * self + alpha * (mat @ vec)"""
    mv_result = torch.matmul(mat, vec)
    if alpha != 1:
        mv_result = torch.mul(mv_result, alpha)

    self_scaled = torch.mul(self, beta) if beta != 1 else self

    return torch.add(self_scaled, mv_result)


@register_aten(
    ["aten::linalg_vector_norm", "aten::linalg_vector_norm.out"],
    static_argnums=(1, 2, 3),
    static_argnames=("dtype",),
)
def torch_linalg_vector_norm(x, ord=2, dim=None, keepdim=False, dtype=None, out=None):
    return torch.linalg.vector_norm(x, ord, dim, keepdim, dtype=dtype)


@register_aten(["aten::linear"])
def torch_linear(input, weight, bias=None):
    """
    linear layer forward pass.: y = x @ weight.T + bias.

    Args:
        input: Input tensor of shape [..., in_features]
        weight: Weight tensor of shape [out_features, in_features]
        bias: Optional bias tensor of shape [out_features]. Default: None

    Returns:
        Output tensor of shape [..., out_features]
    """
    return torch.nn.functional.linear(input, weight, bias)


@register_aten(["aten::linear_backward"], static_argnums=(3,))
def torch_linear_backward(input, grad_output, weight, output_mask):
    return torch.ops.aten.linear_backward(input, grad_output, weight, output_mask)
