"""Linear algebra operations."""

import torch

from torch_neuronx.dtype_utils import (
    torch_dtype_to_scalar_type_name,
    torch_dtype_to_type_meta_name,
)

from ..operation_registry import register_aten


def _meta_name(dtype):
    """TypeMeta spelling for an out= dtype message, falling back to the dtype."""
    return torch_dtype_to_type_meta_name(dtype) or dtype


def _scalar_name(dtype):
    """ScalarType spelling for an operand dtype message, falling back to the dtype."""
    return torch_dtype_to_scalar_type_name(dtype) or dtype


def _validate_matmul_out(x, *args, out=None, out_dtype=None, **kwargs):
    """Exact-dtype out= check for the matmul family (CPU structured-kernel rule).

    CPU's structured matmul kernels (mm, bmm, mv, dot, addmm, addmv) require
    the out= tensor to have exactly the result dtype. The result dtype equals
    the input dtype for both float and integer matmul (integer matmul preserves
    the integer type, not promoting to float).

    The dtype_out overload has an explicit out_dtype argument. The C++ dispatcher
    passes it as a positional arg; the Python caller may pass it as a kwarg.
    Either way, if an explicit output dtype is given, skip the check — the result
    is cast to that dtype before writing to out=.
    """
    out_tensor = out[0] if isinstance(out, tuple | list) else out
    if out_tensor is None or not isinstance(out_tensor, torch.Tensor):
        return
    # out_dtype from kwarg path (Python direct call)
    if out_dtype is not None:
        return
    # out_dtype from positional path (C++ dispatcher for .dtype_out overload)
    if any(isinstance(a, torch.dtype) for a in args):
        return
    if out_tensor.dtype != x.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {_meta_name(x.dtype)}, "
            f"but got {_meta_name(out_tensor.dtype)} instead"
        )


@register_aten(
    ["aten::dot", "aten::dot.out"],
    validate_static_inputs=_validate_matmul_out,
)
def torch_dot(tensor1, tensor2, out=None):
    """Compute the dot product of two one-dimensional tensors.

    Args:
        tensor1 (torch.Tensor): First input tensor.
        tensor2 (torch.Tensor): Second input tensor.
        out (torch.Tensor, optional): Dispatch destination validated but unused here. Defaults to
            ``None``.

    Returns:
        torch.Tensor: Scalar dot product.

    Raises:
        RuntimeError: If the inputs are incompatible one-dimensional tensors or ``out`` has the
            wrong dtype.
    """
    return torch.dot(tensor1, tensor2)


@register_aten(
    ["aten::mv", "aten::mv.out"],
    validate_static_inputs=_validate_matmul_out,
)
def torch_mv(mat, vec, out=None):
    """Multiply a matrix by a vector.

    Args:
        mat (torch.Tensor): Matrix input.
        vec (torch.Tensor): Vector input.
        out (torch.Tensor, optional): Dispatch destination validated but unused here. Defaults to
            ``None``.

    Returns:
        torch.Tensor: Matrix-vector product.

    Raises:
        NotImplementedError: If either operand has ``bool`` dtype.
        RuntimeError: If operand ranks, dimensions, or dtypes are incompatible, or ``out`` has the
            wrong dtype.
    """
    _reject_bool_matmul("mv", mat, vec)
    return torch.mv(mat, vec)


@register_aten(
    ["aten::mm", "aten::mm.out", "aten::mm.dtype_out", "aten::mm.dtype"],
    static_argnames=("out_dtype",),
    validate_static_inputs=_validate_matmul_out,
)
def torch_mm(mat1, mat2, out_dtype=None, out=None):
    """Multiply two matrices.

    Args:
        mat1 (torch.Tensor): First matrix.
        mat2 (torch.Tensor): Second matrix.
        out_dtype (torch.dtype, optional): Dtype to cast the matrix product to.
            Defaults to ``None``.
        out (torch.Tensor, optional): Output buffer validated by registration but not
            forwarded by this implementation. Defaults to ``None``.

    Returns:
        torch.Tensor: Matrix product, cast to ``out_dtype`` when provided.

    Raises:
        NotImplementedError: If either operand has ``bool`` dtype.
        RuntimeError: If operand dtypes, ranks, or dimensions are incompatible, or ``out`` has
            the wrong dtype.
    """
    _reject_bool_matmul("mm", mat1, mat2)
    _reject_mm_dtype_mismatch(mat1, mat2)
    result = torch.mm(mat1, mat2)
    return result if out_dtype is None else result.to(out_dtype)


@register_aten(
    ["aten::bmm", "aten::bmm.out", "aten::bmm.dtype_out", "aten::bmm.dtype"],
    static_argnames=("out_dtype",),
    validate_static_inputs=_validate_matmul_out,
)
def torch_bmm(batch1, batch2, out_dtype=None, out=None):
    """Multiply corresponding matrices in two batches.

    Args:
        batch1 (torch.Tensor): First batch of matrices.
        batch2 (torch.Tensor): Second batch of matrices.
        out_dtype (torch.dtype, optional): Dtype to cast the batch product to. Defaults to ``None``.
        out (torch.Tensor, optional): Dispatch destination, unused here. Defaults to ``None``.

    Returns:
        torch.Tensor: Batch matrix product, cast to ``out_dtype`` when provided.

    Raises:
        NotImplementedError: If either operand has ``bool`` dtype.
        RuntimeError: If the operand ranks, batch sizes, or dimensions are incompatible.
    """
    _reject_bool_matmul("bmm", batch1, batch2)
    _reject_bmm_dtype_mismatch(batch1, batch2)
    result = torch.bmm(batch1, batch2)
    return result if out_dtype is None else result.to(out_dtype)


def _reject_bool_matmul(op_name: str, *tensors: object) -> None:
    """Raise if any tensor operand has dtype bool.

    CPU refuses bool for all matmul-family ops. Neuron lowers them without
    error, so without this guard ``test_dtypes_*`` reports bool as a working
    dtype and fails with "dtypes worked in forward but are not listed by the
    OpInfo: {torch.bool}".
    """
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.dtype is torch.bool:
            raise NotImplementedError(f"\"{op_name}\" not implemented for 'Bool'")


def _reject_mm_dtype_mismatch(m1, m2) -> None:
    """Reject mm-family operands with mismatched dtypes, matching CPU.

    Neuron accepts a mixed-precision matmul at the op level and defers it to
    Torch-MLIR, which then fails to legalize with an unclear message. CPU
    rejects it up front. Raise the same message so the cause is clear.
    """
    if m1.dtype != m2.dtype:
        raise RuntimeError(
            f"expected m1 and m2 to have the same dtype, but got: "
            f"{_meta_name(m1.dtype)} != {_meta_name(m2.dtype)}"
        )


def _reject_bmm_dtype_mismatch(batch1, batch2) -> None:
    """Reject bmm operands with mismatched dtypes, matching CPU."""
    if batch1.dtype != batch2.dtype:
        raise RuntimeError(
            f"expected scalar type {_scalar_name(batch1.dtype)} "
            f"but found {_scalar_name(batch2.dtype)}"
        )


def _reject_addmm_dtype_mismatch(self_, mat1, mat2) -> None:
    """Reject addmm operands with mismatched dtypes, matching CPU.

    CPU checks ``self`` against ``mat2`` first, then ``mat1`` against ``mat2``.
    Keep that order so the message names the same operand pair CPU does.
    """
    if self_.dtype != mat2.dtype:
        raise RuntimeError(
            f"self and mat2 must have the same dtype, but got "
            f"{_scalar_name(self_.dtype)} and {_scalar_name(mat2.dtype)}"
        )
    if mat1.dtype != mat2.dtype:
        raise RuntimeError(
            f"mat1 and mat2 must have the same dtype, but got "
            f"{_scalar_name(mat1.dtype)} and {_scalar_name(mat2.dtype)}"
        )


def _check_out_dtype(expected, out):
    """Reject an ``out=`` tensor whose dtype is not exactly the result dtype.

    The Neuron dispatch path copies the result into ``out=`` unconditionally, so
    without this guard an fp32 result silently casts into an int64 ``out=``
    instead of raising — the divergence OpInfo's ``test_out`` case 4 checks.

    The matmul family requires an **exact** dtype match rather than the
    ``at::canCast`` rule that ``TensorIterator`` ops (add/mul/cosh) use: CPU
    accepts ``torch.add(fp32, fp32, out=fp64)`` but rejects
    ``torch.baddbmm(fp32, fp32, fp32, out=fp64)``, because the structured
    kernel's ``resize_out`` compares dtypes for equality. A shared canCast
    guard would therefore be wrong here.
    """
    if isinstance(out, torch.Tensor) and out.dtype is not expected:
        raise RuntimeError(
            f"Expected out tensor to have dtype {_meta_name(expected)}, "
            f"but got {_meta_name(out.dtype)} instead"
        )


def _validate_baddbmm(input, batch1, batch2, *_args, out=None, **_kwargs):
    """Guard bool operands and the ``out=`` dtype for ``baddbmm``/``.out``."""
    _reject_bool_matmul("baddbmm", input, batch1, batch2)
    operands = [t for t in (input, batch1, batch2) if isinstance(t, torch.Tensor)]
    if len({t.dtype for t in operands}) != 1:
        # CPU rejects mismatched operand dtypes ("Input dtypes must be the
        # same, got: ...") *before* it looks at out=, so leave the message to
        # torch.baddbmm rather than reporting a misleading out= mismatch.
        return
    # All operands share one dtype, so that dtype is also the result dtype
    # (torch.result_type over equal dtypes is the identity).
    _check_out_dtype(operands[0].dtype, out)


def _validate_baddbmm_dtype(input, batch1, batch2, out_dtype=None, *_args, out=None, **_kwargs):
    """Same guards for ``.dtype``/``.dtype_out``, whose result dtype is ``out_dtype``."""
    _reject_bool_matmul("baddbmm", input, batch1, batch2)
    if isinstance(out_dtype, torch.dtype):
        _check_out_dtype(out_dtype, out)


@register_aten(
    ["aten::baddbmm", "aten::baddbmm.out"],
    static_argnames=("beta", "alpha"),
    validate_static_inputs=_validate_baddbmm,
)
def torch_baddbmm(input, batch1, batch2, beta=1, alpha=1, out=None):
    """Add a scaled input to a scaled batch matrix product.

    Args:
        input (torch.Tensor): Tensor broadcast over the batch matrix product.
        batch1 (torch.Tensor): First batch of matrices.
        batch2 (torch.Tensor): Second batch of matrices.
        beta (number, optional): Scale applied to ``input``. Defaults to ``1``.
        alpha (number, optional): Scale applied to the batch matrix product. Defaults to ``1``.
        out (torch.Tensor, optional): Output tensor for the ``.out`` overload. Defaults to ``None``.

    Returns:
        torch.Tensor: ``beta * input + alpha * (batch1 @ batch2)``.

    Raises:
        NotImplementedError: If any operand has ``bool`` dtype.
        RuntimeError: If native validation fails or ``out`` has the wrong dtype.

    Note:
        Native validation preserves exact CPU rank, shape, and broadcast errors.
        Static scales avoid scalar extraction; each pair compiles a cached NEFF.
    """
    return torch.baddbmm(input, batch1, batch2, beta=beta, alpha=alpha)


@register_aten(
    ["aten::baddbmm.dtype", "aten::baddbmm.dtype_out"],
    static_argnums=(3,),
    static_argnames=("out_dtype", "beta", "alpha"),
    validate_static_inputs=_validate_baddbmm_dtype,
)
def torch_baddbmm_dtype(input, batch1, batch2, out_dtype, beta=1, alpha=1, out=None):
    """Add a scaled input to a batch matrix product in an explicit dtype.

    Args:
        input (torch.Tensor): Tensor broadcast over the batch matrix product.
        batch1 (torch.Tensor): First batch of matrices.
        batch2 (torch.Tensor): Second batch of matrices.
        out_dtype (torch.dtype): Positional accumulation and output dtype.
        beta (number, optional): Scale applied to ``input``. Defaults to ``1``.
        alpha (number, optional): Scale applied to the batch matrix product. Defaults to ``1``.
        out (torch.Tensor, optional): Output tensor for ``.dtype_out``. Defaults to ``None``.

    Returns:
        torch.Tensor: Scaled addition accumulated in ``out_dtype``.

    Raises:
        NotImplementedError: If any operand has ``bool`` dtype.
        RuntimeError: If native shape validation fails or ``out`` does not have ``out_dtype``.

    Note:
        This overload has no PyTorch 2.12 CPU fallback; operands are cast before matmul.
    """
    # Cast the operands, not the result: the matmul accumulates in out_dtype.
    if out_dtype is not None:
        input = input.to(out_dtype) if input.dtype != out_dtype else input
        batch1 = batch1.to(out_dtype) if batch1.dtype != out_dtype else batch1
        batch2 = batch2.to(out_dtype) if batch2.dtype != out_dtype else batch2
    return torch.baddbmm(input, batch1, batch2, beta=beta, alpha=alpha)


@register_aten(
    ["aten::addmm", "aten::addmm.out", "aten::addmm.dtype_out", "aten::addmm.dtype"],
    static_argnames=("beta", "alpha", "out_dtype"),
    validate_static_inputs=_validate_matmul_out,
)
def torch_addmm(input, mat1, mat2, out_dtype=None, beta=1, alpha=1, out=None):
    """Add a scaled input to a scaled matrix product.

    Args:
        input (torch.Tensor): Tensor broadcast over the matrix product.
        mat1 (torch.Tensor): First matrix.
        mat2 (torch.Tensor): Second matrix.
        out_dtype (torch.dtype, optional): Dtype to cast the result to. Defaults to ``None``.
        beta (number, optional): Scale applied to ``input``. Defaults to ``1``.
        alpha (number, optional): Scale applied to the matrix product. Defaults to ``1``.
        out (torch.Tensor, optional): Dispatch destination, unused here. Defaults to ``None``.

    Returns:
        torch.Tensor: ``beta * input + alpha * (mat1 @ mat2)``.

    Raises:
        NotImplementedError: If any operand has ``bool`` dtype.
        RuntimeError: If matrix ranks or shapes are invalid, or ``input`` cannot broadcast.
    """
    _reject_bool_matmul("addmm", input, mat1, mat2)
    _reject_addmm_dtype_mismatch(input, mat1, mat2)
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

    result = torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha)
    return result if out_dtype is None else result.to(out_dtype)


@register_aten(
    ["aten::addmv", "aten::addmv_", "aten::addmv.out"],
    static_argnames=(
        "beta",
        "alpha",
    ),
    validate_static_inputs=_validate_matmul_out,
)
def torch_addmv(self, mat, vec, beta=1, alpha=1, out=None):
    """Add a scaled tensor to a scaled matrix-vector product.

    Args:
        self (torch.Tensor): Tensor added to the matrix-vector product.
        mat (torch.Tensor): Matrix input.
        vec (torch.Tensor): Vector input.
        beta (number, optional): Scale applied to ``self``. Defaults to ``1``.
        alpha (number, optional): Scale applied to the matrix-vector product. Defaults to ``1``.
        out (torch.Tensor, optional): Output buffer validated by registration but not
            forwarded by this implementation. Defaults to ``None``.

    Returns:
        torch.Tensor: ``beta * self + alpha * (mat @ vec)`` in the dtype of ``self``.

    Raises:
        NotImplementedError: If any operand has ``bool`` dtype.
        RuntimeError: If operands are incompatible or ``out`` does not have ``self.dtype``.

    Note:
        Scalar-induced promotion is reversed before returning the result.
    """
    _reject_bool_matmul("addmv", self, mat, vec)
    orig_dtype = self.dtype
    mv_result = torch.matmul(mat, vec)
    if alpha != 1:
        mv_result = torch.mul(mv_result, alpha)

    self_scaled = torch.mul(self, beta) if beta != 1 else self

    result = torch.add(self_scaled, mv_result)
    # Restore original dtype when float alpha/beta caused promotion
    if result.dtype != orig_dtype:
        result = result.to(orig_dtype)
    return result


@register_aten(
    ["aten::linalg_vector_norm", "aten::linalg_vector_norm.out"],
    static_argnums=(1, 2, 3),
    static_argnames=("dtype",),
    validate_static_inputs=_validate_matmul_out,
)
def torch_linalg_vector_norm(x, ord=2, dim=None, keepdim=False, dtype=None, out=None):
    """Compute a vector norm over selected dimensions.

    Args:
        x (torch.Tensor): Input tensor.
        ord (number, optional): Norm order. Defaults to ``2``.
        dim (int or sequence[int], optional): Dimensions to reduce. Defaults to ``None``.
        keepdim (bool, optional): Preserve reduced dimensions when ``True``. Defaults to ``False``.
        dtype (torch.dtype, optional): Computation and result dtype. Defaults to ``None``.
        out (torch.Tensor, optional): Dispatch destination, unused here. Defaults to ``None``.

    Returns:
        torch.Tensor: Vector norm values.

    Raises:
        RuntimeError: If ``dtype`` is invalid or would narrow ``x``.
        IndexError: If ``dim`` contains an invalid dimension.
    """
    return torch.linalg.vector_norm(x, ord, dim, keepdim, dtype=dtype)


def _validate_solve_triangular(a, b, *args, out=None, **kwargs):
    """Reject an out= tensor whose dtype is not a.dtype (LAPACK result dtype)."""
    out_tensor = out[0] if isinstance(out, tuple | list) else out
    if isinstance(out_tensor, torch.Tensor) and out_tensor.dtype != a.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype "
            f"{_meta_name(a.dtype)}, but got "
            f"{_meta_name(out_tensor.dtype)} instead"
        )


@register_aten(
    ["aten::linalg_solve_triangular", "aten::linalg_solve_triangular.out"],
    static_argnames=("upper", "left", "unitriangular"),
    validate_static_inputs=_validate_solve_triangular,
)
def torch_linalg_solve_triangular(a, b, *, upper, left=True, unitriangular=False, out=None):
    """Solve a triangular linear system.

    Args:
        a (torch.Tensor): Square triangular coefficient matrix or batch of matrices.
        b (torch.Tensor): Right-hand side matrix or batch of matrices.
        upper (bool): Whether ``a`` is upper triangular.
        left (bool, optional): Whether to solve ``a @ x = b`` instead of ``x @ a = b``.
            Defaults to ``True``.
        unitriangular (bool, optional): Whether the diagonal of ``a`` is treated as ones.
            Defaults to ``False``.
        out (torch.Tensor, optional): Dispatch destination, unused here. Defaults to ``None``.

    Returns:
        torch.Tensor: Solution of the triangular system.
    """
    return torch.ops.aten.linalg_solve_triangular(
        a, b, upper=upper, left=left, unitriangular=unitriangular
    )


@register_aten(["aten::linear"])
def torch_linear(input, weight, bias=None):
    """Apply a linear transformation to an input tensor.

    Args:
        input (torch.Tensor): Input whose final dimension is the feature dimension.
        weight (torch.Tensor): One- or two-dimensional transformation weight.
        bias (torch.Tensor, optional): Bias broadcastable to the transformed result.
            Defaults to ``None``.

    Returns:
        torch.Tensor: ``input @ weight.T + bias``; a one-dimensional ``weight`` removes
            the final input dimension.

    Raises:
        RuntimeError: If input, weight, or bias shapes or dtypes are incompatible.
    """
    # Reject a real matmul dtype mismatch (input vs weight) with the CPU
    # message. Do not reject a bias-only mismatch: Neuron promotes a
    # mismatched bias
    if bias is None:
        _reject_mm_dtype_mismatch(input, weight)
    elif input.dtype != weight.dtype:
        raise RuntimeError(
            f"mat1 and mat2 must have the same dtype, but got "
            f"{_scalar_name(input.dtype)} and {_scalar_name(weight.dtype)}"
        )
    return torch.nn.functional.linear(input, weight, bias)


@register_aten(["aten::linear_backward"], static_argnums=(3,))
def torch_linear_backward(input, grad_output, weight, output_mask):
    """Compute selected gradients for a linear transformation.

    Args:
        input (torch.Tensor): Forward input tensor.
        grad_output (torch.Tensor): Gradient of the linear output.
        weight (torch.Tensor): Forward weight tensor.
        output_mask (sequence[bool]): Mask selecting input, weight, and bias gradients.

    Returns:
        tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]: Input,
        weight, and bias gradients; unselected entries are ``None``.
    """
    return torch.ops.aten.linear_backward(input, grad_output, weight, output_mask)
