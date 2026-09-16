"""Foreach operations for torch-mlir backend."""

import torch

from torch_neuronx.python_ops.nki_kernels.foreach_norm import _has_nki_norm_kernels

if _has_nki_norm_kernels:
    from torch_neuronx.python_ops.nki_kernels.foreach_norm import foreach_norm_nki

from torch_neuronx.python_ops.nki_kernels.foreach_elementwise import (
    foreach_add_list_nki,
    foreach_add_scalar_nki,
    foreach_add_scalarlist_nki,
    foreach_addcdiv_scalar_nki,
    foreach_addcdiv_scalarlist_nki,
    foreach_addcmul_scalar_nki,
    foreach_addcmul_scalarlist_nki,
    foreach_div_list_nki,
    foreach_div_scalar_nki,
    foreach_div_scalarlist_nki,
    foreach_lerp_list_nki,
    foreach_lerp_scalar_nki,
    foreach_lerp_scalarlist_nki,
    foreach_mul_list_nki,
    foreach_mul_scalar_nki,
    foreach_mul_scalarlist_nki,
    foreach_sqrt_nki,
    foreach_sub_list_nki,
    foreach_sub_scalar_nki,
    foreach_sub_scalarlist_nki,
)

from ..operation_registry import Preprocessing, register_aten
from ..validation import check_bool_sub, check_lerp_dtype

# --- Validation helpers ---


def _check_tensors(tensors):
    if not tensors:
        raise RuntimeError("Tensor list must have at least one tensor.")
    _check_same_device(tensors)


def _check_same_device(*tensor_lists):
    """Raise if tensors span more than one device."""

    # Move 0-d tensors (CPU scalars) to the first tensor's device.
    target = tensor_lists[0][0].device if tensor_lists and tensor_lists[0] else None
    if target is not None:
        for tensors in tensor_lists:
            for i, t in enumerate(tensors):
                if t.dim() == 0 and t.device != target:
                    tensors[i] = t.to(target)

    expected_device = None
    for tensors in tensor_lists:
        for t in tensors:
            if expected_device is None:
                expected_device = t.device
            elif t.device != expected_device:
                raise RuntimeError(
                    "Expected all tensors to be on the same device, but found "
                    f"at least two devices, {expected_device} and {t.device}!"
                )


def _check_tensors_scalars(tensors, scalars):
    _check_tensors(tensors)
    if len(tensors) != len(scalars):
        raise RuntimeError("Tensor list must have same number of elements as scalar list.")


def _check_broadcastable_shapes(t1, t2):
    if t1.shape == t2.shape:
        return
    try:
        torch.broadcast_shapes(t1.shape, t2.shape)
    except RuntimeError:
        for dim in range(max(t1.ndim, t2.ndim)):
            d1 = t1.shape[-(dim + 1)] if dim < t1.ndim else 1
            d2 = t2.shape[-(dim + 1)] if dim < t2.ndim else 1
            if d1 != d2 and d1 != 1 and d2 != 1:
                actual_dim = max(t1.ndim, t2.ndim) - 1 - dim
                raise RuntimeError(
                    f"The size of tensor a ({d1}) must match the size of "
                    f"tensor b ({d2}) at non-singleton dimension {actual_dim}"
                ) from None
        raise


def _check_tensors_tensors(tensors1, tensors2):
    _check_tensors(tensors1)
    _check_tensors(tensors2)
    _check_same_device(tensors1, tensors2)
    if len(tensors1) != len(tensors2):
        raise RuntimeError(
            f"Tensor lists must have the same number of tensors, "
            f"got {len(tensors1)} and {len(tensors2)}"
        )
    for t1, t2 in zip(tensors1, tensors2, strict=False):
        _check_broadcastable_shapes(t1, t2)


def _cast_results(op, results):
    """For div, cast integer results to float32 to match PyTorch semantics."""
    if op != "div":
        return results
    return tuple(r.to(torch.float32) if not r.dtype.is_floating_point else r for r in results)


def _check_div_inplace_dtype(op, tensors):
    """Raise if inplace div would produce a dtype that cannot be stored."""
    if op != "div":
        return
    for t in tensors:
        if not t.dtype.is_floating_point:
            result_dtype = torch.true_divide(
                torch.ones(1, dtype=t.dtype),
                torch.ones(1, dtype=t.dtype),
            ).dtype
            raise RuntimeError(
                f"result type {result_dtype} can't be cast to the desired output type {t.dtype}"
            )


def _make_div_inplace_validator(variant):
    """Preserve schema error ordering before preprocessing changes dtypes."""

    def validate(tensors, other, *args, **kwargs):
        # Keep this preprocessor validation pure: device normalization belongs
        # to the operation factory and must not mutate inputs before an error.
        if variant == "tensor":
            _check_scalar_tensor(other)

        if not tensors:
            raise RuntimeError("Tensor list must have at least one tensor.")

        if variant == "list":
            if not other:
                raise RuntimeError("Tensor list must have at least one tensor.")
            if len(tensors) != len(other):
                raise RuntimeError(
                    f"Tensor lists must have the same number of tensors, "
                    f"got {len(tensors)} and {len(other)}"
                )
            for tensor, other_tensor in zip(tensors, other, strict=False):
                _check_broadcastable_shapes(tensor, other_tensor)
        elif variant == "scalarlist":
            if len(tensors) != len(other):
                raise RuntimeError("Tensor list must have same number of elements as scalar list.")

        _check_div_inplace_dtype("div", tensors)

    return validate


def _check_bool_sub(op, *operands):
    """Reject a bool operand for sub. Each operand is one argument slot."""
    if op == "sub":
        check_bool_sub(*operands)


_DTYPE_NAMES = {}


def _get_dtype_name(dtype):
    """Get C++ type name for a dtype, e.g. torch.int8 -> 'Char'."""
    if dtype not in _DTYPE_NAMES:
        _DTYPE_NAMES[dtype] = (
            torch.empty(0, dtype=dtype).type().split(".")[-1].replace("Tensor", "")
        )
    return _DTYPE_NAMES[dtype]


def _check_inplace_casting(tensors, scalar):
    """Raise if inplace op would require dtype promotion, matching CPU behavior."""
    if not tensors:
        return
    result_dtype = torch.result_type(tensors[0], scalar)
    if result_dtype != tensors[0].dtype:
        raise RuntimeError(
            f"result type {_get_dtype_name(result_dtype)} "
            f"can't be cast to the desired output type "
            f"{_get_dtype_name(tensors[0].dtype)}"
        )


def _check_inplace_dtype_casting(tensors, scalars):
    """For inplace ops, raise if dtype promotion is required, matching CPU behavior."""
    if not tensors:
        return
    if not isinstance(scalars, list | tuple):
        scalars = [scalars]
    for s in scalars:
        _check_inplace_casting(tensors, s)


def _replace_empty_promoted(tensors, dtype=None, scalars=None):
    """Retype empty tensors to the dtype compute_fn will promote them to.

    compute_fn promotes non-float inputs inside the traced graph (bool->int,
    int->float for div), and the compiler rejects that convert on a zero-size
    tensor (NCC_ISPP060). Retyping empties up front keeps the convert off them;
    non-empty tensors are left untouched for the C++ cache-hit path. The target
    is `dtype`, or result_type(t, s) per element when `scalars` is given.
    """
    if scalars is None:
        scalars = [None] * len(tensors)
    elif not isinstance(scalars, list | tuple):
        scalars = [scalars] * len(tensors)

    def _target(t, s):
        d = dtype if s is None else torch.result_type(t, s)
        if d.is_floating_point and t.dtype.is_floating_point:
            return t.dtype
        return d

    return [
        torch.empty(t.shape, dtype=_target(t, s), device=t.device) if t.numel() == 0 else t
        for t, s in zip(tensors, scalars, strict=False)
    ]


def _promote_inside(op, tensors, scalars=None):
    """Promote input dtypes inside compute_fn, traced into the HLO.

    The NEFF itself converts raw inputs — C++ cached dispatch feeds raw args
    on cache hit without re-running preprocessing, so any promotion done
    outside the traced graph would be dropped there (bool fed to an int NEFF
    crashes; int32 fed to a float32 NEFF is read as IEEE-754 bits).

    div: non-float -> float32 (true-division semantics, independent of scalar).
    others: result_type(t, s) per element (bool x int, int x float, ...);
    non-promoting pairs (int x int, float x weak scalar) are no-ops.

    Derived only from live args: the lowering kernel cache reuses compute_fn
    closures by __code__, so capturing dtype decisions from the enclosing
    scope would leak state across dtypes.
    """
    if op == "div":
        return [t.float() if not t.dtype.is_floating_point else t for t in tensors]
    if scalars is None:
        return tensors
    if not isinstance(scalars, list | tuple):
        scalars = [scalars] * len(tensors)
    out = []
    for t, s in zip(tensors, scalars, strict=False):
        rt = torch.result_type(t, s)
        if rt == t.dtype:
            out.append(t)
        elif t.numel() == 0:
            # A dtype convert on a 0-size tensor is rejected by the compiler
            # (NCC_ISPP060). Allocate a fresh empty of the target dtype so the
            # promotion is a no-data allocation, not an input convert — this
            # keeps the graph *input* at its original dtype, which matters when
            # the op is fused (op-concat) with an upstream module that produces
            # the empty in its original dtype (empty i1 -> i32 mismatch).
            out.append(torch.empty(t.shape, dtype=rt, device=t.device))
        else:
            out.append(t.to(rt))
    return out


def _promote_div_pairs(tensors1, tensors2, *, empty_only=False):
    """Promote tensor-list division operands to each pair's result dtype."""
    promoted1 = []
    promoted2 = []
    for t1, t2 in zip(tensors1, tensors2, strict=False):
        result_dtype = torch.result_type(t1, t2)
        if not result_dtype.is_floating_point:
            result_dtype = torch.get_default_dtype()

        pair = []
        for tensor in (t1, t2):
            if tensor.dtype == result_dtype or (empty_only and tensor.numel() != 0):
                pair.append(tensor)
            elif tensor.numel() == 0:
                pair.append(torch.empty(tensor.shape, dtype=result_dtype, device=tensor.device))
            else:
                pair.append(tensor.to(result_dtype))
        promoted1.append(pair[0])
        promoted2.append(pair[1])
    return promoted1, promoted2


def _replace_empty(tensors):
    """Return the list with each empty tensor swapped for a fresh empty of the same
    shape/dtype (a passthrough for dynamo), leaving non-empty tensors untouched.

    ``t.to(t.dtype)`` on a non-empty tensor returns the same object, so object
    identity is preserved — the factory ``out=`` tuple and its positional inputs
    stay the same objects, which the buffer-donation aliasing relies on. Dtype
    promotion for empties (e.g. div's int->float) is handled separately by
    ``_replace_empty_promoted``; this function never changes dtype.
    """
    return [
        torch.empty(t.shape, dtype=t.dtype, device=t.device) if t.numel() == 0 else t
        for t in tensors
    ]


def _check_scalar_tensor(tensor):
    if tensor.dim() != 0:
        raise RuntimeError(
            f"scalar tensor expected to be 0 dim but it has {tensor.dim()}"
            f" dimensions and {tensor.numel()} elements."
        )


# --- NKI kernel dispatch tables ---

NKI_SCALAR_FNS = {
    "add": foreach_add_scalar_nki,
    "sub": foreach_sub_scalar_nki,
    "mul": foreach_mul_scalar_nki,
    "div": foreach_div_scalar_nki,
}

NKI_SCALARLIST_FNS = {
    "add": foreach_add_scalarlist_nki,
    "sub": foreach_sub_scalarlist_nki,
    "mul": foreach_mul_scalarlist_nki,
    "div": foreach_div_scalarlist_nki,
}

NKI_LIST_FNS = {
    "add": foreach_add_list_nki,
    "sub": foreach_sub_list_nki,
    "mul": foreach_mul_list_nki,
    "div": foreach_div_list_nki,
}

HAS_ALPHA = {"add", "sub"}


def _void_postprocess(results):
    """Postprocess for .out/inplace variants — dispatch expects None return."""
    return None


# --- Generic factory functions ---


def _make_scalar_op(op):
    """Create a non-inplace foreach scalar op (also handles .out variant)."""
    nki_fn = NKI_SCALAR_FNS[op]
    is_div = op == "div"

    def fn(tensors, scalar, out=None):
        _check_tensors(tensors)
        _check_bool_sub(op, tensors, scalar)
        # NOTE: do NOT eagerly promote the (non-empty) tensors here. compute_fn
        # promotes in-graph and infer_outputs_from_meta runs compute_fn, so the
        # inferred output dtype is already correct (bool x float -> float, ...).
        # Eagerly promoting the returned tensors would leak the *promoted* dtype
        # into the NEFF cache key (via context.original_inputs), collapsing
        # distinct input dtypes that share a promoted type — e.g. bool x 2.5 and
        # int32 x 2.5 both key as `[Float]` and reuse each other's (wrong) NEFF.
        # Keeping the original dtype here makes the key distinguish them.
        if is_div:
            tensors = _replace_empty_promoted(tensors, torch.float32)
        else:
            # Keep empties at their *original* dtype (not the promoted one): the
            # promoted empty output is allocated in-graph by _promote_inside, and
            # keeping the input original lets a fused upstream module (e.g. a
            # non-contiguous transform emitting the empty as bool) connect by
            # type (avoids the empty i1 -> i32 op-concat dependency mismatch).
            tensors = _replace_empty(tensors)

        def compute_fn(tensors, scalar):
            tensors = _promote_inside(op, tensors, scalar)
            return _cast_results(op, nki_fn(tensors, scalar))

        return compute_fn, (tensors, scalar), {}, None

    return fn


def _make_scalar_inplace_op(op):
    """Create an inplace foreach scalar op."""
    nki_fn = NKI_SCALAR_FNS[op]

    def fn(tensors, scalar, out=None):
        _check_tensors(tensors)
        _check_bool_sub(op, tensors, scalar)
        _check_div_inplace_dtype(op, tensors)
        _check_inplace_dtype_casting(tensors, scalar)
        # Keep the list full-length: replace each empty with a fresh empty
        # placeholder (same dtype/shape) instead of filtering it out. The NKI
        # wrapper returns empties unchanged, so they become FX passthrough nodes
        # that capture_backend drops before the compiler sees them, while the
        # kept output/input indices stay aligned with the raw args (no remap).
        tensors = _replace_empty(tensors)

        def compute_fn(tensors, scalar):
            return nki_fn(tensors, scalar)

        return compute_fn, (tensors, scalar), {"out": tuple(tensors)}, _void_postprocess

    return fn


def _make_list_op(op):
    """Create a non-inplace foreach list op (also handles .out variant)."""
    nki_fn = NKI_LIST_FNS[op]
    has_alpha = op in HAS_ALPHA
    is_div = op == "div"

    def fn(tensors1, tensors2, *args, out=None, alpha=1):
        _check_tensors_tensors(tensors1, tensors2)
        _check_bool_sub(op, tensors1, tensors2)
        # Both operands are tensors, so cross-dtype promotion is traced
        # inside compute_fn. Div empties are promoted here because the compiler
        # rejects dtype conversion on zero-size tensors (NCC_ISPP060).
        if is_div:
            tensors1, tensors2 = _promote_div_pairs(tensors1, tensors2, empty_only=True)
        else:
            tensors1 = _replace_empty(tensors1)

        def compute_fn(tensors1, tensors2, **kwargs):
            if is_div:
                tensors1, tensors2 = _promote_div_pairs(tensors1, tensors2)
            if has_alpha:
                results = nki_fn(tensors1, tensors2, kwargs.get("alpha", 1))
            else:
                results = nki_fn(tensors1, tensors2)
            return _cast_results(op, results)

        if all(t.numel() == 0 for t in tensors1):
            if out is not None:
                return compute_fn, (), {}, _void_postprocess
            return compute_fn, (), {}, None

        kwargs = {"alpha": alpha} if has_alpha else {}
        if out is not None:
            kwargs["out"] = out
            return compute_fn, (tensors1, tensors2), kwargs, _void_postprocess
        return compute_fn, (tensors1, tensors2), kwargs, None

    return fn


def _make_list_inplace_op(op):
    """Create an inplace foreach list op."""
    nki_fn = NKI_LIST_FNS[op]
    has_alpha = op in HAS_ALPHA

    def fn(tensors1, tensors2, *args, out=None, alpha=1):
        _check_tensors_tensors(tensors1, tensors2)
        _check_bool_sub(op, tensors1, tensors2)
        _check_div_inplace_dtype(op, tensors1)
        # Keep both lists full-length (see _make_scalar_inplace_op): empties become
        # FX passthroughs that capture_backend drops, so no filtered-index remap.
        tensors1 = _replace_empty(tensors1)

        def compute_fn(tensors1, tensors2, **kwargs):
            if has_alpha:
                return nki_fn(tensors1, tensors2, kwargs.get("alpha", 1))
            return nki_fn(tensors1, tensors2)

        kwargs = {"out": tuple(tensors1)}
        if has_alpha:
            kwargs["alpha"] = alpha
        return compute_fn, (tensors1, tensors2), kwargs, _void_postprocess

    return fn


def _make_scalarlist_op(op):
    """Create a non-inplace foreach scalarlist op (also handles .out variant)."""
    nki_fn = NKI_SCALARLIST_FNS[op]
    is_div = op == "div"

    def fn(tensors, scalars, out=None):
        _check_tensors_scalars(tensors, scalars)
        _check_bool_sub(op, tensors, scalars)
        # Promotion is per (tensor, scalar) pair. No eager outside branch as in
        # _make_scalar_op: scalars stay Python numbers into compute_fn, so
        # _promote_inside there already yields the CPU-matching output dtype.
        if is_div:
            tensors = _replace_empty_promoted(tensors, torch.float32)
        else:
            tensors = _replace_empty_promoted(tensors, scalars=scalars)

        def compute_fn(tensors, scalars):
            tensors = _promote_inside(op, tensors, scalars)
            return _cast_results(op, nki_fn(tensors, scalars))

        if out is not None:
            return compute_fn, (tensors, scalars), {"out": out}, _void_postprocess
        return compute_fn, (tensors, scalars), {}, None

    return fn


def _make_scalarlist_inplace_op(op):
    """Create an inplace foreach scalarlist op."""
    nki_fn = NKI_SCALARLIST_FNS[op]

    def fn(tensors, scalars, out=None):
        _check_tensors_scalars(tensors, scalars)
        _check_bool_sub(op, tensors, scalars)
        _check_div_inplace_dtype(op, tensors)
        _check_inplace_dtype_casting(tensors, scalars)
        # Keep the list full-length (see _make_scalar_inplace_op). scalars stays
        # paired 1:1 with tensors; the NKI wrapper skips empty slots.
        tensors = _replace_empty(tensors)

        def compute_fn(tensors, scalars):
            return nki_fn(tensors, scalars)

        return (
            compute_fn,
            (tensors, scalars),
            {"out": tuple(tensors)},
            _void_postprocess,
        )

    return fn


def _make_tensor_op(op):
    """Create a non-inplace foreach tensor (0-dim) op (also handles .out variant)."""
    nki_fn = NKI_SCALAR_FNS[op]
    has_alpha = op in HAS_ALPHA
    is_div = op == "div"

    def fn(tensors, tensor, *args, out=None, alpha=1):
        _check_tensors(tensors)
        _check_scalar_tensor(tensor)
        _check_bool_sub(op, tensors, tensor)
        # The 0-dim `tensor` plays the scalar's role for promotion (result_type
        # per list element against it), same as _make_scalar_op.
        if is_div:
            tensors = _replace_empty_promoted(tensors, torch.float32)
        else:
            tensors = _replace_empty_promoted(tensors, scalars=tensor)
        effective_tensor = tensor * alpha if has_alpha else tensor

        def compute_fn(tensors, tensor):
            tensors = _promote_inside(op, tensors, tensor)
            return _cast_results(op, nki_fn(tensors, tensor))

        if out is not None:
            return compute_fn, (tensors, effective_tensor), {"out": out}, _void_postprocess
        return compute_fn, (tensors, effective_tensor), {}, None

    return fn


def _make_tensor_inplace_op(op):
    """Create an inplace foreach tensor (0-dim) op."""
    nki_fn = NKI_SCALAR_FNS[op]
    has_alpha = op in HAS_ALPHA

    def fn(tensors, tensor, *args, out=None, alpha=1):
        _check_tensors(tensors)
        _check_scalar_tensor(tensor)
        _check_bool_sub(op, tensors, tensor)
        _check_div_inplace_dtype(op, tensors)
        # Keep the list full-length (see _make_scalar_inplace_op).
        tensors = _replace_empty(tensors)
        effective_tensor = tensor * alpha if has_alpha else tensor

        def compute_fn(tensors, tensor):
            return nki_fn(tensors, tensor)

        return (
            compute_fn,
            (tensors, effective_tensor),
            {"out": tuple(tensors)},
            _void_postprocess,
        )

    return fn


# --- Norm (special case, not elementwise) ---


@register_aten(
    ["aten::_foreach_norm.Scalar", "aten::_foreach_norm.Scalar_out"],
    static_argnums=(1, 2),
    static_argnames=("dtype",),
)
def torch_foreach_norm(tensors, ord=2, dtype=None, out=None):
    """Foreach norm operation using NKI kernel with MLIR fallback."""
    if _has_nki_norm_kernels:
        return foreach_norm_nki(tensors, ord=ord, dtype=dtype)

    # MLIR fallback
    result = []
    for t in tensors:
        if t.numel() == 0:
            result.append(torch.zeros((), dtype=dtype or t.dtype, device=t.device))
        elif dtype is not None:
            result.append(torch.linalg.vector_norm(t.flatten(), ord=ord, dtype=dtype))
        else:
            result.append(torch.linalg.vector_norm(t.flatten(), ord=ord))
    return tuple(result)


# --- foreach_sqrt (unary) ---


@register_aten(
    ["aten::_foreach_sqrt", "aten::_foreach_sqrt.out"], preprocessing=Preprocessing.SHAPE_ONLY
)
def torch_foreach_sqrt(tensors, out=None):
    _check_tensors(tensors)
    tensors = _replace_empty(tensors)
    return foreach_sqrt_nki, (tensors,), {}, None


@register_aten(["aten::_foreach_sqrt_"], preprocessing=Preprocessing.VALUE_DEPENDENT)
def torch_foreach_sqrt_inplace(tensors, out=None):
    _check_tensors(tensors)
    # Keep the list full-length (see _make_scalar_inplace_op).
    tensors = _replace_empty(tensors)
    return foreach_sqrt_nki, (tensors,), {"out": tuple(tensors)}, _void_postprocess


# --- foreach_addcdiv / foreach_addcmul (ternary) ---


def _check_three_tensor_lists(self, tensor1, tensor2):
    _check_tensors(self)
    _check_tensors(tensor1)
    _check_tensors(tensor2)
    _check_same_device(self, tensor1, tensor2)
    if not (len(self) == len(tensor1) == len(tensor2)):
        raise RuntimeError(
            f"Tensor lists must have the same number of tensors, "
            f"got {len(self)}, {len(tensor1)} and {len(tensor2)}"
        )
    for s, t1, t2 in zip(self, tensor1, tensor2, strict=False):
        if s.numel() == 0 or (s.shape == t1.shape and s.shape == t2.shape):
            continue
        for a, b, _name_a, _name_b in [(s, t1, "self", "tensor1"), (s, t2, "self", "tensor2")]:
            if a.shape == b.shape:
                continue
            try:
                torch.broadcast_shapes(a.shape, b.shape)
            except RuntimeError:
                for dim in range(max(a.ndim, b.ndim)):
                    d1 = a.shape[-(dim + 1)] if dim < a.ndim else 1
                    d2 = b.shape[-(dim + 1)] if dim < b.ndim else 1
                    if d1 != d2 and d1 != 1 and d2 != 1:
                        actual_dim = max(a.ndim, b.ndim) - 1 - dim
                        raise RuntimeError(
                            f"The size of tensor a ({d1}) must match the size of "
                            f"tensor b ({d2}) at non-singleton dimension {actual_dim}"
                        ) from None
                raise


def _check_addcop_dtypes(op, self, tensor1, tensor2):
    """Reject the dtypes that Tensor.addcmul and Tensor.addcdiv also reject.

    addcdiv rejects two integral divisors. See ``torch_addcdiv`` in
    ``ops/binary.py`` for the single-tensor form of the same check.
    """
    if self and self[0].dtype == torch.bool:
        raise RuntimeError("addcmul/addcdiv not supported for Bool dtype")
    if op != "addcdiv":
        return
    if _all_integral(tensor1) and _all_integral(tensor2):
        raise RuntimeError(
            "Integer division with addcdiv is no longer supported, and in a future "
            "release addcdiv will perform a true division of tensor1 and tensor2. "
            "The historic addcdiv behavior can be implemented as "
            "(input + value * torch.trunc(tensor1 / tensor2)).to(input.dtype) for "
            "integer inputs and as (input + value * tensor1 / tensor2) for float inputs. "
            "The future addcdiv behavior is just the latter implementation: "
            "(input + value * tensor1 / tensor2), for all dtypes."
        )


def _all_integral(tensors):
    """True when every tensor has an integral dtype. Bool counts as integral."""
    return bool(tensors) and all(
        not (t.dtype.is_floating_point or t.dtype.is_complex) for t in tensors
    )


def _make_addcop_scalar(nki_fn, op):
    def fn(self, tensor1, tensor2, value=1, out=None):
        _check_three_tensor_lists(self, tensor1, tensor2)
        _check_addcop_dtypes(op, self, tensor1, tensor2)
        if self and not self[0].dtype.is_floating_point:
            out_dtype = torch.result_type(self[0], value)
            self = [t.to(out_dtype) for t in self]
            tensor1 = [t.to(out_dtype) for t in tensor1]
            tensor2 = [t.to(out_dtype) for t in tensor2]
        self = _replace_empty(self)

        def compute_fn(self, tensor1, tensor2, value):
            return nki_fn(self, tensor1, tensor2, value)

        if out is not None:
            return compute_fn, (self, tensor1, tensor2, value), {"out": out}, _void_postprocess
        return compute_fn, (self, tensor1, tensor2, value), {}, None

    return fn


def _make_addcop_scalar_inplace(nki_fn, op):
    def fn(self, tensor1, tensor2, value=1, out=None):
        _check_three_tensor_lists(self, tensor1, tensor2)
        _check_addcop_dtypes(op, self, tensor1, tensor2)
        _check_inplace_dtype_casting(self, value)
        # Keep all three lists full-length (see _make_scalar_inplace_op): empties
        # become FX passthroughs that capture_backend drops, so self[i]/tensor1[i]/
        # tensor2[i] stay paired and no filtered-index remap is needed.
        self = _replace_empty(self)

        def compute_fn(self, tensor1, tensor2, value):
            return nki_fn(self, tensor1, tensor2, value)

        return (
            compute_fn,
            (self, tensor1, tensor2, value),
            {"out": tuple(self)},
            _void_postprocess,
        )

    return fn


def _make_addcop_scalarlist(nki_fn, op):
    def fn(self, tensor1, tensor2, scalars, out=None):
        _check_three_tensor_lists(self, tensor1, tensor2)
        _check_tensors_scalars(self, scalars)
        _check_addcop_dtypes(op, self, tensor1, tensor2)
        if self and scalars and not self[0].dtype.is_floating_point:
            out_dtype = torch.result_type(self[0], scalars[0])
            self = [t.to(out_dtype) for t in self]
            tensor1 = [t.to(out_dtype) for t in tensor1]
            tensor2 = [t.to(out_dtype) for t in tensor2]
        self = _replace_empty(self)

        def compute_fn(self, tensor1, tensor2, scalars):
            return nki_fn(self, tensor1, tensor2, scalars)

        if out is not None:
            return compute_fn, (self, tensor1, tensor2, scalars), {"out": out}, _void_postprocess
        return compute_fn, (self, tensor1, tensor2, scalars), {}, None

    return fn


def _make_addcop_scalarlist_inplace(nki_fn, op):
    def fn(self, tensor1, tensor2, scalars, out=None):
        _check_three_tensor_lists(self, tensor1, tensor2)
        _check_tensors_scalars(self, scalars)
        _check_addcop_dtypes(op, self, tensor1, tensor2)
        _check_inplace_dtype_casting(self, scalars)
        # Keep all lists full-length (see _make_scalar_inplace_op); scalars stays
        # paired 1:1 with self.
        self = _replace_empty(self)

        def compute_fn(self, tensor1, tensor2, scalars):
            return nki_fn(self, tensor1, tensor2, scalars)

        return (
            compute_fn,
            (self, tensor1, tensor2, scalars),
            {"out": tuple(self)},
            _void_postprocess,
        )

    return fn


def _make_addcop_tensor(nki_scalarlist_fn, op):
    """Tensor variant: scalars is a 1-dim packed tensor with one value per tensor."""

    def fn(self, tensor1, tensor2, scalars, out=None):
        _check_three_tensor_lists(self, tensor1, tensor2)
        _check_addcop_dtypes(op, self, tensor1, tensor2)
        self = _replace_empty(self)

        def compute_fn(self, tensor1, tensor2, scalars):
            return nki_scalarlist_fn(self, tensor1, tensor2, scalars)

        if out is not None:
            return (
                compute_fn,
                (self, tensor1, tensor2, scalars),
                {"out": out},
                _void_postprocess,
            )
        return compute_fn, (self, tensor1, tensor2, scalars), {}, None

    return fn


def _make_addcop_tensor_inplace(nki_scalarlist_fn, op):
    """Inplace tensor variant of _make_addcop_tensor (same scalars-as-tensor
    handling). The lists are kept full-length: empties become FX passthroughs
    that capture_backend drops, so list[i] still pairs with scalars[i] and no
    filtered-index remap is needed.
    """

    def fn(self, tensor1, tensor2, scalars, out=None):
        _check_three_tensor_lists(self, tensor1, tensor2)
        _check_addcop_dtypes(op, self, tensor1, tensor2)
        self = _replace_empty(self)

        def compute_fn(self, tensor1, tensor2, scalars):
            return nki_scalarlist_fn(self, tensor1, tensor2, scalars)

        return (
            compute_fn,
            (self, tensor1, tensor2, scalars),
            {"out": tuple(self)},
            _void_postprocess,
        )

    return fn


_ADDCOP_VARIANT_FACTORIES = {
    "scalar": (_make_addcop_scalar, [".Scalar", ".Scalar_out"]),
    "scalar_inplace": (_make_addcop_scalar_inplace, ["_.Scalar"]),
    "scalarlist": (_make_addcop_scalarlist, [".ScalarList", ".ScalarList_out"]),
    "scalarlist_inplace": (_make_addcop_scalarlist_inplace, ["_.ScalarList"]),
    "tensor": (_make_addcop_tensor, [".Tensor", ".Tensor_out"]),
    "tensor_inplace": (_make_addcop_tensor_inplace, ["_.Tensor"]),
}

_ADDCOP_NKI_FNS = {
    "addcdiv": {
        "scalar": foreach_addcdiv_scalar_nki,
        "scalarlist": foreach_addcdiv_scalarlist_nki,
    },
    "addcmul": {
        "scalar": foreach_addcmul_scalar_nki,
        "scalarlist": foreach_addcmul_scalarlist_nki,
    },
}

for _op_name, _nki_fns in _ADDCOP_NKI_FNS.items():
    for _variant_name, (_factory, _suffixes) in _ADDCOP_VARIANT_FACTORIES.items():
        # Tensor variant uses scalarlist nki_fn (packed tensor → list in preprocessing)
        _base = _variant_name.replace("_inplace", "")
        if _base == "tensor":
            _base = "scalarlist"
        _nki_fn = _nki_fns[_base]
        _fn = _factory(_nki_fn, _op_name)
        _fn.__name__ = f"torch_foreach_{_op_name}_{_variant_name}"
        _aten_names = [f"aten::_foreach_{_op_name}{s}" for s in _suffixes]
        # shape_only for non-inplace Scalar variant only.
        _prep = (
            Preprocessing.SHAPE_ONLY if _variant_name == "scalar" else Preprocessing.VALUE_DEPENDENT
        )
        register_aten(_aten_names, preprocessing=_prep)(_fn)


# --- foreach_lerp ---


def _check_lerp_dtype(self, weight_is_tensor):
    """Reject a bool tensor list for lerp, the same as Tensor.lerp."""
    if self:
        check_lerp_dtype(self[0], self[0] if weight_is_tensor else 0.0)


@register_aten(
    ["aten::_foreach_lerp.Scalar", "aten::_foreach_lerp.Scalar_out"],
    preprocessing=Preprocessing.SHAPE_ONLY,
)
def torch_foreach_lerp_scalar(self, tensors1, weight, out=None):
    _check_tensors_tensors(self, tensors1)
    _check_lerp_dtype(self, weight_is_tensor=False)
    self = _replace_empty(self)

    def compute_fn(self, tensors1, weight):
        return foreach_lerp_scalar_nki(self, tensors1, weight)

    if out is not None:
        return compute_fn, (self, tensors1, weight), {"out": out}, _void_postprocess
    return compute_fn, (self, tensors1, weight), {}, None


@register_aten(["aten::_foreach_lerp_.Scalar"], preprocessing=Preprocessing.VALUE_DEPENDENT)
def torch_foreach_lerp_scalar_inplace(self, tensors1, weight, out=None):
    _check_tensors_tensors(self, tensors1)
    _check_lerp_dtype(self, weight_is_tensor=False)
    # Keep both lists full-length (see _make_scalar_inplace_op).
    self = _replace_empty(self)

    def compute_fn(self, tensors1, weight):
        return foreach_lerp_scalar_nki(self, tensors1, weight)

    return (
        compute_fn,
        (self, tensors1, weight),
        {"out": tuple(self)},
        _void_postprocess,
    )


@register_aten(
    ["aten::_foreach_lerp.List", "aten::_foreach_lerp.List_out"],
    preprocessing=Preprocessing.SHAPE_ONLY,
)
def torch_foreach_lerp_list(self, tensors1, weights, out=None):
    _check_tensors_tensors(self, tensors1)
    _check_tensors_tensors(self, weights)
    _check_lerp_dtype(self, weight_is_tensor=True)
    self = _replace_empty(self)

    def compute_fn(self, tensors1, weights):
        return foreach_lerp_list_nki(self, tensors1, weights)

    if out is not None:
        return compute_fn, (self, tensors1, weights), {"out": out}, _void_postprocess
    return compute_fn, (self, tensors1, weights), {}, None


@register_aten(["aten::_foreach_lerp_.List"], preprocessing=Preprocessing.VALUE_DEPENDENT)
def torch_foreach_lerp_list_inplace(self, tensors1, weights, out=None):
    _check_tensors_tensors(self, tensors1)
    _check_tensors_tensors(self, weights)
    _check_lerp_dtype(self, weight_is_tensor=True)
    # Keep all three lists full-length (see _make_scalar_inplace_op).
    self = _replace_empty(self)

    def compute_fn(self, tensors1, weights):
        return foreach_lerp_list_nki(self, tensors1, weights)

    return compute_fn, (self, tensors1, weights), {"out": tuple(self)}, _void_postprocess


@register_aten(
    ["aten::_foreach_lerp.ScalarList", "aten::_foreach_lerp.ScalarList_out"],
    preprocessing=Preprocessing.VALUE_DEPENDENT,
)
def torch_foreach_lerp_scalarlist(self, tensors1, weight, out=None):
    _check_tensors_tensors(self, tensors1)
    _check_tensors_scalars(self, weight)
    _check_lerp_dtype(self, weight_is_tensor=False)
    self = _replace_empty(self)

    def compute_fn(self, tensors1, weight):
        return foreach_lerp_scalarlist_nki(self, tensors1, weight)

    if out is not None:
        return compute_fn, (self, tensors1, weight), {"out": out}, _void_postprocess
    return compute_fn, (self, tensors1, weight), {}, None


@register_aten(["aten::_foreach_lerp_.ScalarList"], preprocessing=Preprocessing.VALUE_DEPENDENT)
def torch_foreach_lerp_scalarlist_inplace(self, tensors1, weight, out=None):
    _check_tensors_tensors(self, tensors1)
    _check_tensors_scalars(self, weight)
    _check_lerp_dtype(self, weight_is_tensor=False)
    # Keep all lists full-length (see _make_scalar_inplace_op); weight (scalar
    # list) stays paired 1:1 with self.
    self = _replace_empty(self)

    def compute_fn(self, tensors1, weight):
        return foreach_lerp_scalarlist_nki(self, tensors1, weight)

    return compute_fn, (self, tensors1, weight), {"out": tuple(self)}, _void_postprocess


# --- Register all binary elementwise foreach ops ---

_OP_REGISTRATIONS = []

_VARIANT_FACTORIES = {
    "scalar": (_make_scalar_op, [".Scalar", ".Scalar_out"]),
    "scalar_inplace": (_make_scalar_inplace_op, ["_.Scalar"]),
    "list": (_make_list_op, [".List", ".List_out"]),
    "list_inplace": (_make_list_inplace_op, ["_.List"]),
    "scalarlist": (_make_scalarlist_op, [".ScalarList", ".ScalarList_out"]),
    "scalarlist_inplace": (_make_scalarlist_inplace_op, ["_.ScalarList"]),
    "tensor": (_make_tensor_op, [".Tensor", ".Tensor_out"]),
    "tensor_inplace": (_make_tensor_inplace_op, ["_.Tensor"]),
}

for _op in ("add", "sub", "mul", "div"):
    for _variant_name, (_factory, _suffixes) in _VARIANT_FACTORIES.items():
        _fn = _factory(_op)
        _fn.__name__ = f"torch_foreach_{_op}_{_variant_name}"
        _aten_names = [f"aten::_foreach_{_op}{s}" for s in _suffixes]
        _OP_REGISTRATIONS.append((_aten_names, _fn))

for _aten_names, _fn in _OP_REGISTRATIONS:
    # shape_only for non-inplace, non-div, non-scalarlist, non-scalar variants.
    _is_inplace = any("_." in n for n in _aten_names)
    _is_div = any("_foreach_div" in n for n in _aten_names)
    _is_scalarlist = any("ScalarList" in n for n in _aten_names)
    _is_scalar = any(n.endswith(".Scalar") or n.endswith(".Scalar_out") for n in _aten_names)
    _validator_variant = (
        "scalarlist"
        if _is_scalarlist
        else "list"
        if any(".List" in n for n in _aten_names)
        else "tensor"
        if any(".Tensor" in n for n in _aten_names)
        else "scalar"
    )
    _prep = (
        Preprocessing.SHAPE_ONLY
        if (not _is_inplace and not _is_div and not _is_scalarlist and not _is_scalar)
        else Preprocessing.VALUE_DEPENDENT
    )
    _validate = (
        _make_div_inplace_validator(_validator_variant) if (_is_inplace and _is_div) else None
    )
    register_aten(
        _aten_names,
        preprocessing=_prep,
        validate_static_inputs=_validate,
    )(_fn)
