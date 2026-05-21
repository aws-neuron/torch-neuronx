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

# --- Validation helpers ---


def _check_tensors(tensors):
    if not tensors:
        raise RuntimeError("Tensor list must have at least one tensor.")


def _check_tensors_scalars(tensors, scalars):
    _check_tensors(tensors)
    if len(tensors) != len(scalars):
        raise RuntimeError("Tensor list must have same number of elements as scalar list.")


def _check_tensors_tensors(tensors1, tensors2):
    _check_tensors(tensors1)
    _check_tensors(tensors2)
    if len(tensors1) != len(tensors2):
        raise RuntimeError(
            f"Tensor lists must have the same number of tensors, "
            f"got {len(tensors1)} and {len(tensors2)}"
        )


def _cast_results(op, results):
    """For div, cast integer results to float32 to match PyTorch semantics."""
    if op != "div":
        return results
    return tuple(r.to(torch.float32) if not r.dtype.is_floating_point else r for r in results)


def _check_div_inplace_dtype(op, tensors):
    """Raise if inplace div on integer tensors, matching CPU behavior."""
    if op != "div":
        return
    for t in tensors:
        if not t.dtype.is_floating_point:
            raise RuntimeError(
                "result type Float can't be cast to the desired output type "
                f"{str(t.dtype).replace('torch.', '').capitalize()}"
            )


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


def _replace_empty(tensors, op=""):
    """Replace empty tensors with new empty tensors (passthrough for dynamo).
    For div, use float32 dtype to match PyTorch int/int -> float semantics.
    """

    def _tensor_dtype(t):
        if op == "div" and not t.dtype.is_floating_point:
            return torch.float32
        return t.dtype

    return [
        torch.empty(t.shape, dtype=_tensor_dtype(t), device=t.device)
        if t.numel() == 0
        else t.to(_tensor_dtype(t))
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


# --- Generic factory functions ---


def _make_scalar_op(op):
    """Create a non-inplace foreach scalar op."""
    nki_fn = NKI_SCALAR_FNS[op]

    def fn(tensors, scalar, out=None):
        _check_tensors(tensors)
        if tensors and not tensors[0].dtype.is_floating_point:
            out_dtype = torch.result_type(tensors[0], scalar)
            tensors = [t.to(out_dtype) for t in tensors]
        tensors = _replace_empty(tensors, op)

        def compute_fn(tensors, scalar):
            return _cast_results(op, nki_fn(tensors, scalar))

        return compute_fn, (tensors, scalar), {}, None

    return fn


def _make_scalar_inplace_op(op):
    """Create an inplace foreach scalar op."""
    nki_fn = NKI_SCALAR_FNS[op]

    def fn(tensors, scalar, out=None):
        _check_tensors(tensors)
        _check_div_inplace_dtype(op, tensors)
        _check_inplace_dtype_casting(tensors, scalar)
        non_empty = [t for t in tensors if t.numel() > 0]

        def compute_fn(tensors, scalar):
            return nki_fn(tensors, scalar)

        def postprocess_fn(results):
            return None

        if not non_empty:
            return compute_fn, (), {}, postprocess_fn
        return compute_fn, (non_empty, scalar), {"out": tuple(non_empty)}, postprocess_fn

    return fn


def _make_list_op(op):
    """Create a non-inplace foreach list op."""
    nki_fn = NKI_LIST_FNS[op]
    has_alpha = op in HAS_ALPHA

    def fn(tensors1, tensors2, *args, out=None, alpha=1):
        _check_tensors_tensors(tensors1, tensors2)
        tensors1 = _replace_empty(tensors1, op)

        def compute_fn(tensors1, tensors2, **kwargs):
            if has_alpha:
                results = nki_fn(tensors1, tensors2, kwargs.get("alpha", 1))
            else:
                results = nki_fn(tensors1, tensors2)
            return _cast_results(op, results)

        if all(t.numel() == 0 for t in tensors1):
            return compute_fn, (), {}, None

        kwargs = {"alpha": alpha} if has_alpha else {}
        return compute_fn, (tensors1, tensors2), kwargs, None

    return fn


def _make_list_inplace_op(op):
    """Create an inplace foreach list op."""
    nki_fn = NKI_LIST_FNS[op]
    has_alpha = op in HAS_ALPHA

    def fn(tensors1, tensors2, *args, out=None, alpha=1):
        _check_tensors_tensors(tensors1, tensors2)
        _check_div_inplace_dtype(op, tensors1)
        non_empty_t1 = [t for t in tensors1 if t.numel() > 0]
        non_empty_t2 = [t2 for t1, t2 in zip(tensors1, tensors2, strict=False) if t1.numel() > 0]

        def compute_fn(tensors1, tensors2, **kwargs):
            if has_alpha:
                return nki_fn(tensors1, tensors2, kwargs.get("alpha", 1))
            return nki_fn(tensors1, tensors2)

        def postprocess_fn(results):
            return None

        if not non_empty_t1:
            return compute_fn, (), {}, postprocess_fn

        kwargs = {"out": tuple(non_empty_t1)}
        if has_alpha:
            kwargs["alpha"] = alpha
        return compute_fn, (non_empty_t1, non_empty_t2), kwargs, postprocess_fn

    return fn


def _make_scalarlist_op(op):
    """Create a non-inplace foreach scalarlist op."""
    nki_fn = NKI_SCALARLIST_FNS[op]

    def fn(tensors, scalars, out=None):
        _check_tensors_scalars(tensors, scalars)
        if tensors and scalars and not tensors[0].dtype.is_floating_point:
            out_dtype = torch.result_type(tensors[0], scalars[0])
            tensors = [t.to(out_dtype) for t in tensors]
        tensors = _replace_empty(tensors, op)

        def compute_fn(tensors, scalars):
            return _cast_results(op, nki_fn(tensors, scalars))

        return compute_fn, (tensors, scalars), {}, None

    return fn


def _make_scalarlist_inplace_op(op):
    """Create an inplace foreach scalarlist op."""
    nki_fn = NKI_SCALARLIST_FNS[op]

    def fn(tensors, scalars, out=None):
        _check_tensors_scalars(tensors, scalars)
        _check_div_inplace_dtype(op, tensors)
        _check_inplace_dtype_casting(tensors, scalars)
        non_empty = [(t, s) for t, s in zip(tensors, scalars, strict=False) if t.numel() > 0]
        non_empty_t = [t for t, s in non_empty]
        non_empty_s = [s for t, s in non_empty]

        def compute_fn(tensors, scalars):
            return nki_fn(tensors, scalars)

        def postprocess_fn(results):
            return None

        if not non_empty_t:
            return compute_fn, (), {}, postprocess_fn

        return compute_fn, (non_empty_t, non_empty_s), {"out": tuple(non_empty_t)}, postprocess_fn

    return fn


def _make_tensor_op(op):
    """Create a non-inplace foreach tensor (0-dim) op."""
    nki_fn = NKI_SCALAR_FNS[op]
    has_alpha = op in HAS_ALPHA

    def fn(tensors, tensor, *args, out=None, alpha=1):
        _check_tensors(tensors)
        _check_scalar_tensor(tensor)
        tensors = _replace_empty(tensors, op)
        effective_tensor = tensor * alpha if has_alpha else tensor

        def compute_fn(tensors, tensor):
            return _cast_results(op, nki_fn(tensors, tensor))

        return compute_fn, (tensors, effective_tensor), {}, None

    return fn


def _make_tensor_inplace_op(op):
    """Create an inplace foreach tensor (0-dim) op."""
    nki_fn = NKI_SCALAR_FNS[op]
    has_alpha = op in HAS_ALPHA

    def fn(tensors, tensor, *args, out=None, alpha=1):
        _check_tensors(tensors)
        _check_scalar_tensor(tensor)
        _check_div_inplace_dtype(op, tensors)
        non_empty = [t for t in tensors if t.numel() > 0]
        effective_tensor = tensor * alpha if has_alpha else tensor

        def compute_fn(tensors, tensor):
            return nki_fn(tensors, tensor)

        def postprocess_fn(results):
            return None

        if not non_empty:
            return compute_fn, (), {}, postprocess_fn
        return (
            compute_fn,
            (non_empty, effective_tensor),
            {"out": tuple(non_empty)},
            postprocess_fn,
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
    non_empty = [t for t in tensors if t.numel() > 0]

    def postprocess_fn(results):
        return None

    if not non_empty:
        return foreach_sqrt_nki, (), {}, postprocess_fn
    return foreach_sqrt_nki, (non_empty,), {"out": tuple(non_empty)}, postprocess_fn


# --- foreach_addcdiv / foreach_addcmul (ternary) ---


def _check_three_tensor_lists(self, tensor1, tensor2):
    _check_tensors(self)
    _check_tensors(tensor1)
    _check_tensors(tensor2)
    if not (len(self) == len(tensor1) == len(tensor2)):
        raise RuntimeError(
            f"Tensor lists must have the same number of tensors, "
            f"got {len(self)}, {len(tensor1)} and {len(tensor2)}"
        )


def _make_addcop_scalar(nki_fn):
    def fn(self, tensor1, tensor2, value=1, out=None):
        _check_three_tensor_lists(self, tensor1, tensor2)
        if self and not self[0].dtype.is_floating_point:
            out_dtype = torch.result_type(self[0], value)
            self = [t.to(out_dtype) for t in self]
            tensor1 = [t.to(out_dtype) for t in tensor1]
            tensor2 = [t.to(out_dtype) for t in tensor2]
        self = _replace_empty(self)

        def compute_fn(self, tensor1, tensor2, value):
            return nki_fn(self, tensor1, tensor2, value)

        return compute_fn, (self, tensor1, tensor2, value), {}, None

    return fn


def _make_addcop_scalar_inplace(nki_fn):
    def fn(self, tensor1, tensor2, value=1, out=None):
        _check_three_tensor_lists(self, tensor1, tensor2)
        _check_inplace_dtype_casting(self, value)
        non_empty_idx = [i for i, t in enumerate(self) if t.numel() > 0]
        ne_self = [self[i] for i in non_empty_idx]
        ne_t1 = [tensor1[i] for i in non_empty_idx]
        ne_t2 = [tensor2[i] for i in non_empty_idx]

        def compute_fn(self, tensor1, tensor2, value):
            return nki_fn(self, tensor1, tensor2, value)

        def postprocess_fn(results):
            return None

        if not ne_self:
            return compute_fn, (), {}, postprocess_fn
        return compute_fn, (ne_self, ne_t1, ne_t2, value), {"out": tuple(ne_self)}, postprocess_fn

    return fn


def _make_addcop_scalarlist(nki_fn):
    def fn(self, tensor1, tensor2, scalars, out=None):
        _check_three_tensor_lists(self, tensor1, tensor2)
        _check_tensors_scalars(self, scalars)
        if self and scalars and not self[0].dtype.is_floating_point:
            out_dtype = torch.result_type(self[0], scalars[0])
            self = [t.to(out_dtype) for t in self]
            tensor1 = [t.to(out_dtype) for t in tensor1]
            tensor2 = [t.to(out_dtype) for t in tensor2]
        self = _replace_empty(self)

        def compute_fn(self, tensor1, tensor2, scalars):
            return nki_fn(self, tensor1, tensor2, scalars)

        return compute_fn, (self, tensor1, tensor2, scalars), {}, None

    return fn


def _make_addcop_scalarlist_inplace(nki_fn):
    def fn(self, tensor1, tensor2, scalars, out=None):
        _check_three_tensor_lists(self, tensor1, tensor2)
        _check_tensors_scalars(self, scalars)
        _check_inplace_dtype_casting(self, scalars)
        non_empty_idx = [i for i, t in enumerate(self) if t.numel() > 0]
        ne_self = [self[i] for i in non_empty_idx]
        ne_t1 = [tensor1[i] for i in non_empty_idx]
        ne_t2 = [tensor2[i] for i in non_empty_idx]
        ne_s = [scalars[i] for i in non_empty_idx]

        def compute_fn(self, tensor1, tensor2, scalars):
            return nki_fn(self, tensor1, tensor2, scalars)

        def postprocess_fn(results):
            return None

        if not ne_self:
            return compute_fn, (), {}, postprocess_fn
        return compute_fn, (ne_self, ne_t1, ne_t2, ne_s), {"out": tuple(ne_self)}, postprocess_fn

    return fn


def _make_addcop_tensor(nki_scalarlist_fn):
    """Tensor variant: scalars is a 1-dim packed tensor with one value per tensor."""

    def fn(self, tensor1, tensor2, scalars, out=None):
        _check_three_tensor_lists(self, tensor1, tensor2)
        self = _replace_empty(self)
        # TODO: need NKI support for passing in multiple scalars as a tensor
        scalar_list = scalars.tolist()

        def compute_fn(self, tensor1, tensor2, scalars):
            return nki_scalarlist_fn(self, tensor1, tensor2, scalars)

        return compute_fn, (self, tensor1, tensor2, scalar_list), {}, None

    return fn


def _make_addcop_tensor_inplace(nki_scalarlist_fn):
    def fn(self, tensor1, tensor2, scalars, out=None):
        _check_three_tensor_lists(self, tensor1, tensor2)
        # TODO: need NKI support for passing in multiple scalars as a tensor
        scalar_list = scalars.tolist()
        non_empty_idx = [i for i, t in enumerate(self) if t.numel() > 0]
        ne_self = [self[i] for i in non_empty_idx]
        ne_t1 = [tensor1[i] for i in non_empty_idx]
        ne_t2 = [tensor2[i] for i in non_empty_idx]
        ne_s = [scalar_list[i] for i in non_empty_idx]

        def compute_fn(self, tensor1, tensor2, scalars):
            return nki_scalarlist_fn(self, tensor1, tensor2, scalars)

        def postprocess_fn(results):
            return None

        if not ne_self:
            return compute_fn, (), {}, postprocess_fn
        return (
            compute_fn,
            (ne_self, ne_t1, ne_t2, ne_s),
            {"out": tuple(ne_self)},
            postprocess_fn,
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
        _fn = _factory(_nki_fn)
        _fn.__name__ = f"torch_foreach_{_op_name}_{_variant_name}"
        _aten_names = [f"aten::_foreach_{_op_name}{s}" for s in _suffixes]
        # shape_only for non-inplace Scalar variant only.
        _prep = (
            Preprocessing.SHAPE_ONLY if _variant_name == "scalar" else Preprocessing.VALUE_DEPENDENT
        )
        register_aten(_aten_names, preprocessing=_prep)(_fn)


# --- foreach_lerp ---


@register_aten(
    ["aten::_foreach_lerp.Scalar", "aten::_foreach_lerp.Scalar_out"],
    preprocessing=Preprocessing.SHAPE_ONLY,
)
def torch_foreach_lerp_scalar(self, tensors1, weight, out=None):
    _check_tensors_tensors(self, tensors1)
    self = _replace_empty(self)

    def compute_fn(self, tensors1, weight):
        return foreach_lerp_scalar_nki(self, tensors1, weight)

    return compute_fn, (self, tensors1, weight), {}, None


@register_aten(["aten::_foreach_lerp_.Scalar"], preprocessing=Preprocessing.VALUE_DEPENDENT)
def torch_foreach_lerp_scalar_inplace(self, tensors1, weight, out=None):
    _check_tensors_tensors(self, tensors1)
    non_empty_s = [s for s in self if s.numel() > 0]
    non_empty_t = [t for s, t in zip(self, tensors1, strict=False) if s.numel() > 0]

    def compute_fn(self, tensors1, weight):
        return foreach_lerp_scalar_nki(self, tensors1, weight)

    def postprocess_fn(results):
        return None

    if not non_empty_s:
        return compute_fn, (), {}, postprocess_fn
    return (
        compute_fn,
        (non_empty_s, non_empty_t, weight),
        {"out": tuple(non_empty_s)},
        postprocess_fn,
    )


@register_aten(
    ["aten::_foreach_lerp.List", "aten::_foreach_lerp.List_out"],
    preprocessing=Preprocessing.SHAPE_ONLY,
)
def torch_foreach_lerp_list(self, tensors1, weights, out=None):
    _check_tensors_tensors(self, tensors1)
    _check_tensors_tensors(self, weights)
    self = _replace_empty(self)

    def compute_fn(self, tensors1, weights):
        return foreach_lerp_list_nki(self, tensors1, weights)

    return compute_fn, (self, tensors1, weights), {}, None


@register_aten(["aten::_foreach_lerp_.List"], preprocessing=Preprocessing.VALUE_DEPENDENT)
def torch_foreach_lerp_list_inplace(self, tensors1, weights, out=None):
    _check_tensors_tensors(self, tensors1)
    _check_tensors_tensors(self, weights)
    non_empty_idx = [i for i, s in enumerate(self) if s.numel() > 0]
    ne_s = [self[i] for i in non_empty_idx]
    ne_t = [tensors1[i] for i in non_empty_idx]
    ne_w = [weights[i] for i in non_empty_idx]

    def compute_fn(self, tensors1, weights):
        return foreach_lerp_list_nki(self, tensors1, weights)

    def postprocess_fn(results):
        return None

    if not ne_s:
        return compute_fn, (), {}, postprocess_fn
    return compute_fn, (ne_s, ne_t, ne_w), {"out": tuple(ne_s)}, postprocess_fn


@register_aten(
    ["aten::_foreach_lerp.ScalarList", "aten::_foreach_lerp.ScalarList_out"],
    preprocessing=Preprocessing.VALUE_DEPENDENT,
)
def torch_foreach_lerp_scalarlist(self, tensors1, weight, out=None):
    _check_tensors_tensors(self, tensors1)
    _check_tensors_scalars(self, weight)
    self = _replace_empty(self)

    def compute_fn(self, tensors1, weight):
        return foreach_lerp_scalarlist_nki(self, tensors1, weight)

    return compute_fn, (self, tensors1, weight), {}, None


@register_aten(["aten::_foreach_lerp_.ScalarList"], preprocessing=Preprocessing.VALUE_DEPENDENT)
def torch_foreach_lerp_scalarlist_inplace(self, tensors1, weight, out=None):
    _check_tensors_tensors(self, tensors1)
    _check_tensors_scalars(self, weight)
    non_empty_idx = [i for i, s in enumerate(self) if s.numel() > 0]
    ne_s = [self[i] for i in non_empty_idx]
    ne_t = [tensors1[i] for i in non_empty_idx]
    ne_w = [weight[i] for i in non_empty_idx]

    def compute_fn(self, tensors1, weight):
        return foreach_lerp_scalarlist_nki(self, tensors1, weight)

    def postprocess_fn(results):
        return None

    if not ne_s:
        return compute_fn, (), {}, postprocess_fn
    return compute_fn, (ne_s, ne_t, ne_w), {"out": tuple(ne_s)}, postprocess_fn


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
    _prep = (
        Preprocessing.SHAPE_ONLY
        if (not _is_inplace and not _is_div and not _is_scalarlist and not _is_scalar)
        else Preprocessing.VALUE_DEPENDENT
    )
    register_aten(_aten_names, preprocessing=_prep)(_fn)
