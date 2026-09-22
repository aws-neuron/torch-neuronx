"""Linear algebra operations."""

import torch

from ..operation_registry import register_aten

# ``caffe2::TypeMeta`` names, spelled exactly as PyTorch's structured-kernel
# out= check prints them.
_TYPE_META_NAMES = {
    torch.float32: "float",
    torch.float64: "double",
    torch.float16: "c10::Half",
    torch.bfloat16: "c10::BFloat16",
    torch.int64: "long int",
    torch.int32: "int",
    torch.int16: "short int",
    torch.int8: "signed char",
    torch.uint8: "unsigned char",
    torch.bool: "bool",
}


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
            f"Expected out tensor to have dtype {_TYPE_META_NAMES.get(x.dtype, x.dtype)}, "
            f"but got {_TYPE_META_NAMES.get(out_tensor.dtype, out_tensor.dtype)} instead"
        )


@register_aten(
    ["aten::dot", "aten::dot.out"],
    validate_static_inputs=_validate_matmul_out,
)
def torch_dot(tensor1, tensor2, out=None):
    return torch.dot(tensor1, tensor2)


@register_aten(
    ["aten::mv", "aten::mv.out"],
    validate_static_inputs=_validate_matmul_out,
)
def torch_mv(mat, vec, out=None):
    _reject_bool_matmul("mv", mat, vec)
    return torch.mv(mat, vec)


@register_aten(
    ["aten::mm", "aten::mm.out", "aten::mm.dtype_out", "aten::mm.dtype"],
    static_argnames=("out_dtype",),
    validate_static_inputs=_validate_matmul_out,
)
def torch_mm(mat1, mat2, out_dtype=None, out=None):
    _reject_bool_matmul("mm", mat1, mat2)
    result = torch.mm(mat1, mat2)
    return result if out_dtype is None else result.to(out_dtype)


@register_aten(
    ["aten::bmm", "aten::bmm.out", "aten::bmm.dtype_out", "aten::bmm.dtype"],
    static_argnames=("out_dtype",),
    validate_static_inputs=_validate_matmul_out,
)
def torch_bmm(batch1, batch2, out_dtype=None, out=None):
    _reject_bool_matmul("bmm", batch1, batch2)
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
            f"Expected out tensor to have dtype {_TYPE_META_NAMES.get(expected, expected)}, "
            f"but got {_TYPE_META_NAMES.get(out.dtype, out.dtype)} instead"
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
    """Batch matrix-matrix product with broadcasted input add.

    Schema: aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2,
                          *, Scalar beta=1, Scalar alpha=1) -> Tensor

    Computes: out = beta * input + alpha * (batch1 @ batch2)

    Shape/dim validation is delegated to ``torch.baddbmm`` rather than
    reimplemented here: it already raises for non-3-D operands, batch-size
    and inner-dim mismatches, and a non-broadcastable ``input``, with the
    exact messages CPU produces. Hand-written checks diverged from those
    messages (e.g. a 2-D ``batch1`` raises an expand error on CPU, not a
    "expected 3D tensor" error), which breaks OpInfo error-regex tests.

    ``beta``/``alpha`` must be static: ``torch.baddbmm`` reads them as
    Python scalars while the graph is traced, so passing them as runtime
    inputs fails with ``aten._local_scalar_dense`` during make_fx. Each
    distinct pair compiles its own NEFF (same as addmm/addmv), which the
    persistent NEFF cache absorbs after the first call.
    """
    return torch.baddbmm(input, batch1, batch2, beta=beta, alpha=alpha)


@register_aten(
    ["aten::baddbmm.dtype", "aten::baddbmm.dtype_out"],
    static_argnums=(3,),
    static_argnames=("out_dtype", "beta", "alpha"),
    validate_static_inputs=_validate_baddbmm_dtype,
)
def torch_baddbmm_dtype(input, batch1, batch2, out_dtype, beta=1, alpha=1, out=None):
    """``baddbmm`` with an explicit accumulate/output dtype.

    Schema: aten::baddbmm.dtype(Tensor self, Tensor batch1, Tensor batch2,
                                ScalarType out_dtype, *, Scalar beta=1,
                                Scalar alpha=1) -> Tensor

    ``out_dtype`` is a *positional* ScalarType (arg 3), not a keyword — so
    these are separate overloads from ``aten::baddbmm``/``.out`` and are
    dispatched independently. Registering them matters even though they are
    rare: an unregistered overload cannot fall back, because upstream
    PyTorch 2.12 ships no CPU kernel for ``baddbmm.dtype``/``.dtype_out``
    at all, so the CPU fallback path raises NotImplementedError instead of
    producing a result.

    Casting the operands up front makes the matmul accumulate in
    ``out_dtype`` (the point of the overload for bf16/fp16 inputs with an
    fp32 output). Each operand is tested against ``out_dtype`` individually
    — gating all three casts on ``batch1`` alone would leave a lower-precision
    ``input`` uncast, and ``torch.baddbmm`` then raises a scalar-type mismatch
    instead of honoring ``out_dtype``.
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
    _reject_bool_matmul("addmm", input, mat1, mat2)
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
    """Decomposed implementation of addmv: beta * self + alpha * (mat @ vec)"""
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
    return torch.linalg.vector_norm(x, ord, dim, keepdim, dtype=dtype)


def _validate_solve_triangular(a, b, *args, out=None, **kwargs):
    """Reject an out= tensor whose dtype is not a.dtype (LAPACK result dtype)."""
    out_tensor = out[0] if isinstance(out, tuple | list) else out
    if isinstance(out_tensor, torch.Tensor) and out_tensor.dtype != a.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype "
            f"{_TYPE_META_NAMES.get(a.dtype, a.dtype)}, but got "
            f"{_TYPE_META_NAMES.get(out_tensor.dtype, out_tensor.dtype)} instead"
        )


@register_aten(
    ["aten::linalg_solve_triangular", "aten::linalg_solve_triangular.out"],
    static_argnames=("upper", "left", "unitriangular"),
    validate_static_inputs=_validate_solve_triangular,
)
def torch_linalg_solve_triangular(a, b, *, upper, left=True, unitriangular=False, out=None):
    return torch.ops.aten.linalg_solve_triangular(
        a, b, upper=upper, left=left, unitriangular=unitriangular
    )


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
