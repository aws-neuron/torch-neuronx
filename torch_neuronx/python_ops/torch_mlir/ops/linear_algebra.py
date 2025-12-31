"""Linear algebra operations."""

import torch

from ..operation_registry import register_aten


@register_aten(["aten::dot", "aten::dot.out"])
def torch_dot(tensor1, tensor2):
    return torch.dot(tensor1, tensor2)


@register_aten(["aten::mv", "aten::mv.out"])
def torch_mv(mat, vec):
    return torch.mv(mat, vec)


@register_aten(["aten::mm", "aten::mm.out"])
def torch_mm(mat1, mat2):
    return torch.mm(mat1, mat2)


@register_aten(["aten::bmm", "aten::bmm.out"])
def torch_bmm(batch1, batch2):
    return torch.bmm(batch1, batch2)


@register_aten(["aten::addmm", "aten::addmm.out"], static_argnames=("beta", "alpha"))
def torch_addmm(input, mat1, mat2, beta=1, alpha=1):
    return torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha)


@register_aten(
    ["aten::addmv", "aten::addmv_", "aten::addmv.out"],
    static_argnames=(
        "beta",
        "alpha",
    ),
)
def torch_addmv(self, mat, vec, beta=1, alpha=1):
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
def torch_linalg_vector_norm(x, ord=2, dim=None, keepdim=False, dtype=None):
    return torch.linalg.vector_norm(x, ord, dim, keepdim, dtype=dtype)


@register_aten(
    ["aten::_foreach_norm.Scalar", "aten::_foreach_norm.Scalar_out"], static_argnames=("ord",)
)
def torch_foreach_norm(tensors, ord=2):
    return [torch.linalg.vector_norm(t, ord=ord) for t in tensors]


@register_aten(["aten::linear"], use_torch_compile=True)
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


@register_aten(["aten::linear_backward"], use_torch_compile=True)
def torch_linear_backward(input, grad_output, weight, output_mask):
    """
    Compute gradients for linear layer backward pass.

    Args:
        input: Input tensor to linear layer, (shape: [..., in_features])
        grad_output: Gradient tensor from the output, (shape: [..., out_features])
        weight: Weight tensor of linear layer (shape: [out_features, in_features])
        output_mask: Boolean mask indicating which gradients to compute

    Returns:
        Tuple of (grad_input, grad_weight, grad_bias)
    """
    grad_input = grad_output @ weight if output_mask[0] else None

    grad_weight = None
    if output_mask[1]:
        batch_dims = "".join(chr(ord("a") + j) for j in range(input.ndim - 1))
        grad_input_eq = f"{batch_dims}i,{batch_dims}j->ij"
        grad_weight = torch.einsum(grad_input_eq, grad_output, input)

    grad_bias = None
    if output_mask[2]:
        if grad_output.ndim > 1:
            grad_bias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))
        else:
            grad_bias = grad_output

    return grad_input, grad_weight, grad_bias
