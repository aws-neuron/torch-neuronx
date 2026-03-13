"""Foreach operations for torch-mlir backend."""

import torch

from torch_neuronx.python_ops.nki_kernels.foreach_norm import _has_nki_norm_kernels

if _has_nki_norm_kernels:
    from torch_neuronx.python_ops.nki_kernels.foreach_norm import foreach_norm_nki

from ..operation_registry import register_aten


def _check_tensors(tensors):
    if not tensors:
        raise ValueError("Tensor list must have at least one tensor")


def _check_tensors_scalars(tensors, scalars):
    _check_tensors(tensors)
    if len(tensors) != len(scalars):
        raise ValueError("Tensor list must have same number of elements as scalar list")


def _check_tensors_tensors(tensors1, tensors2):
    _check_tensors(tensors1)
    _check_tensors(tensors2)
    if len(tensors1) != len(tensors2):
        raise RuntimeError(
            f"Tensor lists must have the same number of tensors, "
            f"got {len(tensors1)} and {len(tensors2)}"
        )


@register_aten(
    ["aten::_foreach_norm.Scalar", "aten::_foreach_norm.Scalar_out"],
    static_argnums=(1, 2),
    static_argnames=("dtype",),
)
def torch_foreach_norm(tensors, ord=2, dtype=None, out=None):
    """Foreach norm operation using NKI kernel with MLIR fallback."""
    # TODO remove once foreach_norm NKI kernel is available
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


@register_aten(
    ["aten::_foreach_add.Scalar", "aten::_foreach_add.Scalar_out"], uses_preprocessing=True
)
def torch_foreach_add_scalar(tensors, scalar, out=None):
    """Foreach add with scalar."""
    _check_tensors(tensors)
    # ensure new objects are returned for empty tensors
    tensors = [
        torch.empty(t.shape, dtype=t.dtype, device=t.device) if t.numel() == 0 else t
        for t in tensors
    ]

    def actual_fn(tensors, scalar):
        result = []
        for t in tensors:
            if t.numel() == 0:
                result.append(t)
            else:
                result.append(torch.add(t, scalar))
        return tuple(result)

    return actual_fn, (tensors, scalar), {}


@register_aten(["aten::_foreach_add_.Scalar"], uses_preprocessing=True)
def torch_foreach_add_scalar_inplace(tensors, scalar, out=None):
    """Foreach add with scalar."""
    _check_tensors(tensors)
    non_empty = [t for t in tensors if t.numel() > 0]

    def compute_fn(tensors, scalar):
        return tuple(torch.add(t, scalar) for t in tensors)

    def postprocess_fn(results):
        return None

    if not non_empty:
        return compute_fn, (), {}, postprocess_fn

    return compute_fn, (non_empty, scalar), {"out": tuple(non_empty)}, postprocess_fn


@register_aten(
    [
        "aten::_foreach_add.List",
        "aten::_foreach_add.List_out",
    ],
    static_argnums=(2,),
    uses_preprocessing=True,
)
def torch_foreach_add_list(tensors1, tensors2, alpha=1, out=None):
    """Foreach add between tensor lists."""
    _check_tensors_tensors(tensors1, tensors2)
    tensors1 = [
        torch.empty(t.shape, dtype=t.dtype, device=t.device) if t.numel() == 0 else t
        for t in tensors1
    ]

    def actual_fn(tensors1, tensors2, alpha):
        result = []
        for t1, t2 in zip(tensors1, tensors2, strict=False):
            if t1.numel() == 0:
                result.append(t1)
            else:
                result.append(torch.add(t1, t2, alpha=alpha))
        return tuple(result)

    return actual_fn, (tensors1, tensors2, alpha), {}


@register_aten(["aten::_foreach_add_.List"], static_argnums=(2,), uses_preprocessing=True)
def torch_foreach_add_list_inplace(tensors1, tensors2, alpha=1, out=None):
    """Foreach add inplace between tensor lists."""
    _check_tensors_tensors(tensors1, tensors2)
    pairs = [(t1, t2) for t1, t2 in zip(tensors1, tensors2, strict=False) if t1.numel() > 0]

    def compute_fn(tensors1, tensors2, alpha):
        return tuple(
            torch.add(t1, t2, alpha=alpha) for t1, t2 in zip(tensors1, tensors2, strict=False)
        )

    def postprocess_fn(results):
        return None

    if not pairs:
        return compute_fn, (), {}, postprocess_fn

    non_empty1, non_empty2 = zip(*pairs, strict=False)
    return (
        compute_fn,
        (list(non_empty1), list(non_empty2), alpha),
        {"out": tuple(non_empty1)},
        postprocess_fn,
    )


@register_aten(
    [
        "aten::_foreach_add.ScalarList",
        "aten::_foreach_add.ScalarList_out",
    ],
    static_argnums=(2,),
    uses_preprocessing=True,
)
def torch_foreach_add_scalarlist(tensors, scalars, alpha=1, out=None):
    """Foreach add with scalar list."""
    _check_tensors_scalars(tensors, scalars)
    tensors = [
        torch.empty(t.shape, dtype=t.dtype, device=t.device) if t.numel() == 0 else t
        for t in tensors
    ]

    def actual_fn(tensors, scalars, alpha):
        result = []
        for t, s in zip(tensors, scalars, strict=False):
            if t.numel() == 0:
                result.append(t)
            else:
                result.append(torch.add(t, s, alpha=alpha))
        return tuple(result)

    return actual_fn, (tensors, scalars, alpha), {}


@register_aten(["aten::_foreach_add_.ScalarList"], uses_preprocessing=True)
def torch_foreach_add_scalarlist_inplace(tensors, scalars, alpha=1, out=None):
    """Foreach add inplace with scalar list."""
    _check_tensors_scalars(tensors, scalars)
    for t, s in zip(tensors, scalars, strict=False):
        if t.numel() > 0:
            t.add_(s, alpha=alpha)

    def noop(tensors):
        return None

    return noop, (torch.empty(0),), {}


@register_aten(
    ["aten::_foreach_add.Tensor", "aten::_foreach_add.Tensor_out"],
    static_argnums=(2,),
    uses_preprocessing=True,
)
def torch_foreach_add_tensor(tensors, tensor, alpha=1, out=None):
    """Foreach add with single tensor."""
    _check_tensors(tensors)
    if tensor.dim() != 0:
        raise RuntimeError(
            f"scalar tensor expected to be 0 dim but it has {tensor.dim()}"
            f" dimensions and {tensor.numel()} elements."
        )
    tensors = [
        torch.empty(t.shape, dtype=t.dtype, device=t.device) if t.numel() == 0 else t
        for t in tensors
    ]

    def actual_fn(tensors, tensor, alpha):
        result = []
        for t in tensors:
            if t.numel() == 0:
                result.append(t)
            else:
                result.append(torch.add(t, tensor, alpha=alpha))
        return tuple(result)

    return actual_fn, (tensors, tensor, alpha), {}


@register_aten(["aten::_foreach_add_.Tensor"], static_argnums=(2,), uses_preprocessing=True)
def torch_foreach_add_tensor_inplace(tensors, tensor, alpha=1, out=None):
    """Foreach add inplace with single tensor."""
    _check_tensors(tensors)
    if tensor.dim() != 0:
        raise RuntimeError(
            f"scalar tensor expected to be 0 dim but it has {tensor.dim()}"
            f" dimensions and {tensor.numel()} elements."
        )
    non_empty = [t for t in tensors if t.numel() > 0]

    def compute_fn(tensors, tensor, alpha):
        return tuple(torch.add(t, tensor, alpha=alpha) for t in tensors)

    def postprocess_fn(results):
        return None

    if not non_empty:
        return compute_fn, (), {}, postprocess_fn

    return compute_fn, (non_empty, tensor, alpha), {"out": tuple(non_empty)}, postprocess_fn


@register_aten(
    ["aten::_foreach_mul.Scalar", "aten::_foreach_mul.Scalar_out"], uses_preprocessing=True
)
def torch_foreach_mul_scalar(tensors, scalar, out=None):
    """Foreach multiply with scalar."""
    _check_tensors(tensors)
    tensors = [
        torch.empty(t.shape, dtype=t.dtype, device=t.device) if t.numel() == 0 else t
        for t in tensors
    ]

    def actual_fn(tensors, scalar):
        result = []
        for t in tensors:
            if t.numel() == 0:
                result.append(t)
            else:
                result.append(torch.mul(t, scalar))
        return tuple(result)

    return actual_fn, (tensors, scalar), {}


@register_aten(["aten::_foreach_mul_.Scalar"], uses_preprocessing=True)
def torch_foreach_mul_scalar_inplace(tensors, scalar, out=None):
    """Foreach multiply inplace with scalar."""
    _check_tensors(tensors)
    non_empty = [t for t in tensors if t.numel() > 0]

    def compute_fn(tensors, scalar):
        return tuple(torch.mul(t, scalar) for t in tensors)

    def postprocess_fn(results):
        return None

    if not non_empty:
        return compute_fn, (), {}, postprocess_fn

    return compute_fn, (non_empty, scalar), {"out": tuple(non_empty)}, postprocess_fn


@register_aten(
    [
        "aten::_foreach_mul.List",
        "aten::_foreach_mul.List_out",
    ],
    uses_preprocessing=True,
)
def torch_foreach_mul_list(tensors1, tensors2, out=None):
    """Foreach multiply between tensor lists."""
    _check_tensors_tensors(tensors1, tensors2)
    tensors1 = [
        torch.empty(t.shape, dtype=t.dtype, device=t.device) if t.numel() == 0 else t
        for t in tensors1
    ]

    def actual_fn(tensors1, tensors2):
        result = []
        for t1, t2 in zip(tensors1, tensors2, strict=False):
            if t1.numel() == 0:
                result.append(t1)
            else:
                result.append(torch.mul(t1, t2))
        return tuple(result)

    return actual_fn, (tensors1, tensors2), {}


@register_aten(
    [
        "aten::_foreach_mul.ScalarList",
        "aten::_foreach_mul.ScalarList_out",
    ],
    uses_preprocessing=True,
)
def torch_foreach_mul_scalarlist(tensors, scalars, out=None):
    """Foreach multiply with scalar list."""
    _check_tensors_scalars(tensors, scalars)
    tensors = [
        torch.empty(t.shape, dtype=t.dtype, device=t.device) if t.numel() == 0 else t
        for t in tensors
    ]

    def actual_fn(tensors, scalars):
        result = []
        for t, s in zip(tensors, scalars, strict=False):
            if t.numel() == 0:
                result.append(t)
            else:
                result.append(torch.mul(t, s))
        return tuple(result)

    return actual_fn, (tensors, scalars), {}


@register_aten(["aten::_foreach_mul_.List"], uses_preprocessing=True)
def torch_foreach_mul_list_inplace(tensors1, tensors2, out=None):
    """Foreach multiply inplace between tensor lists."""
    _check_tensors_tensors(tensors1, tensors2)
    pairs = [(t1, t2) for t1, t2 in zip(tensors1, tensors2, strict=False) if t1.numel() > 0]

    def compute_fn(tensors1, tensors2):
        return tuple(torch.mul(t1, t2) for t1, t2 in zip(tensors1, tensors2, strict=False))

    def postprocess_fn(results):
        return None

    if not pairs:
        return compute_fn, (), {}, postprocess_fn

    non_empty1, non_empty2 = zip(*pairs, strict=False)
    return (
        compute_fn,
        (list(non_empty1), list(non_empty2)),
        {"out": tuple(non_empty1)},
        postprocess_fn,
    )


@register_aten(["aten::_foreach_mul_.ScalarList"], uses_preprocessing=True)
def torch_foreach_mul_scalarlist_inplace(tensors, scalars, out=None):
    """Foreach multiply inplace with scalar list."""
    _check_tensors_scalars(tensors, scalars)
    for t, s in zip(tensors, scalars, strict=False):
        if t.numel() > 0:
            t.mul_(s)

    def noop(tensors):
        return None

    return noop, (torch.empty(0),), {}


@register_aten(
    ["aten::_foreach_mul.Tensor", "aten::_foreach_mul.Tensor_out"], uses_preprocessing=True
)
def torch_foreach_mul_tensor(tensors, tensor, out=None):
    """Foreach multiply with single tensor."""
    _check_tensors(tensors)
    if tensor.dim() != 0:
        raise RuntimeError(
            f"scalar tensor expected to be 0 dim but it has {tensor.dim()}"
            f" dimensions and {tensor.numel()} elements."
        )
    tensors = [
        torch.empty(t.shape, dtype=t.dtype, device=t.device) if t.numel() == 0 else t
        for t in tensors
    ]

    def actual_fn(tensors, tensor):
        result = []
        for t in tensors:
            if t.numel() == 0:
                result.append(t)
            else:
                result.append(torch.mul(t, tensor))
        return tuple(result)

    return actual_fn, (tensors, tensor), {}


@register_aten(["aten::_foreach_mul_.Tensor"], uses_preprocessing=True)
def torch_foreach_mul_tensor_inplace(tensors, tensor, out=None):
    """Foreach multiply inplace with single tensor."""
    _check_tensors(tensors)
    if tensor.dim() != 0:
        raise RuntimeError(
            f"scalar tensor expected to be 0 dim but it has {tensor.dim()}"
            f" dimensions and {tensor.numel()} elements."
        )
    non_empty = [t for t in tensors if t.numel() > 0]

    def compute_fn(tensors, tensor):
        return tuple(torch.mul(t, tensor) for t in tensors)

    def postprocess_fn(results):
        return None

    if not non_empty:
        return compute_fn, (), {}, postprocess_fn

    return compute_fn, (non_empty, tensor), {"out": tuple(non_empty)}, postprocess_fn


@register_aten(
    ["aten::_foreach_sub.Scalar", "aten::_foreach_sub.Scalar_out"],
    static_argnums=(2,),
    uses_preprocessing=True,
)
def torch_foreach_sub_scalar(tensors, scalar, alpha=1, out=None):
    """Foreach subtract with scalar."""
    _check_tensors(tensors)
    tensors = [
        torch.empty(t.shape, dtype=t.dtype, device=t.device) if t.numel() == 0 else t
        for t in tensors
    ]

    def actual_fn(tensors, scalar, alpha):
        result = []
        for t in tensors:
            if t.numel() == 0:
                result.append(t)
            else:
                result.append(torch.sub(t, scalar, alpha=alpha))
        return tuple(result)

    return actual_fn, (tensors, scalar, alpha), {}


@register_aten(["aten::_foreach_sub_.Scalar"], static_argnums=(2,), uses_preprocessing=True)
def torch_foreach_sub_scalar_inplace(tensors, scalar, alpha=1, out=None):
    """Foreach subtract inplace with scalar."""
    _check_tensors(tensors)
    non_empty = [t for t in tensors if t.numel() > 0]

    def compute_fn(tensors, scalar, alpha):
        return tuple(torch.sub(t, scalar, alpha=alpha) for t in tensors)

    def postprocess_fn(results):
        return None

    if not non_empty:
        return compute_fn, (), {}, postprocess_fn

    return compute_fn, (non_empty, scalar, alpha), {"out": tuple(non_empty)}, postprocess_fn


@register_aten(
    [
        "aten::_foreach_sub.List",
        "aten::_foreach_sub.List_out",
    ],
    static_argnums=(2,),
    uses_preprocessing=True,
)
def torch_foreach_sub_list(tensors1, tensors2, alpha=1, out=None):
    """Foreach subtract between tensor lists."""
    _check_tensors_tensors(tensors1, tensors2)
    tensors1 = [
        torch.empty(t.shape, dtype=t.dtype, device=t.device) if t.numel() == 0 else t
        for t in tensors1
    ]

    def actual_fn(tensors1, tensors2, alpha):
        result = []
        for t1, t2 in zip(tensors1, tensors2, strict=False):
            if t1.numel() == 0:
                result.append(t1)
            else:
                result.append(torch.sub(t1, t2, alpha=alpha))
        return tuple(result)

    return actual_fn, (tensors1, tensors2, alpha), {}


@register_aten(
    [
        "aten::_foreach_sub.ScalarList",
        "aten::_foreach_sub.ScalarList_out",
    ],
    static_argnums=(2,),
    uses_preprocessing=True,
)
def torch_foreach_sub_scalarlist(tensors, scalars, alpha=1, out=None):
    """Foreach subtract with scalar list."""
    _check_tensors_scalars(tensors, scalars)
    tensors = [
        torch.empty(t.shape, dtype=t.dtype, device=t.device) if t.numel() == 0 else t
        for t in tensors
    ]

    def actual_fn(tensors, scalars, alpha):
        result = []
        for t, s in zip(tensors, scalars, strict=False):
            if t.numel() == 0:
                result.append(t)
            else:
                result.append(torch.sub(t, s, alpha=alpha))
        return tuple(result)

    return actual_fn, (tensors, scalars, alpha), {}


@register_aten(["aten::_foreach_sub_.List"], static_argnums=(2,), uses_preprocessing=True)
def torch_foreach_sub_list_inplace(tensors1, tensors2, alpha=1, out=None):
    """Foreach subtract inplace between tensor lists."""
    _check_tensors_tensors(tensors1, tensors2)
    pairs = [(t1, t2) for t1, t2 in zip(tensors1, tensors2, strict=False) if t1.numel() > 0]

    def compute_fn(tensors1, tensors2, alpha):
        return tuple(
            torch.sub(t1, t2, alpha=alpha) for t1, t2 in zip(tensors1, tensors2, strict=False)
        )

    def postprocess_fn(results):
        return None

    if not pairs:
        return compute_fn, (), {}, postprocess_fn

    non_empty1, non_empty2 = zip(*pairs, strict=False)
    return (
        compute_fn,
        (list(non_empty1), list(non_empty2), alpha),
        {"out": tuple(non_empty1)},
        postprocess_fn,
    )


@register_aten(["aten::_foreach_sub_.ScalarList"], uses_preprocessing=True)
def torch_foreach_sub_scalarlist_inplace(tensors, scalars, alpha=1, out=None):
    """Foreach subtract inplace with scalar list."""
    _check_tensors_scalars(tensors, scalars)
    for t, s in zip(tensors, scalars, strict=False):
        if t.numel() > 0:
            t.sub_(s, alpha=alpha)

    def noop(tensors):
        return None

    return noop, (torch.empty(0),), {}


@register_aten(
    ["aten::_foreach_sub.Tensor", "aten::_foreach_sub.Tensor_out"],
    static_argnums=(2,),
    uses_preprocessing=True,
)
def torch_foreach_sub_tensor(tensors, tensor, alpha=1, out=None):
    """Foreach subtract with single tensor."""
    _check_tensors(tensors)
    if tensor.dim() != 0:
        raise RuntimeError(
            f"scalar tensor expected to be 0 dim but it has {tensor.dim()}"
            f" dimensions and {tensor.numel()} elements."
        )
    tensors = [
        torch.empty(t.shape, dtype=t.dtype, device=t.device) if t.numel() == 0 else t
        for t in tensors
    ]

    def actual_fn(tensors, tensor, alpha):
        result = []
        for t in tensors:
            if t.numel() == 0:
                result.append(t)
            else:
                result.append(torch.sub(t, tensor, alpha=alpha))
        return tuple(result)

    return actual_fn, (tensors, tensor, alpha), {}


@register_aten(["aten::_foreach_sub_.Tensor"], static_argnums=(2,), uses_preprocessing=True)
def torch_foreach_sub_tensor_inplace(tensors, tensor, alpha=1, out=None):
    """Foreach subtract inplace with single tensor."""
    _check_tensors(tensors)
    if tensor.dim() != 0:
        raise RuntimeError(
            f"scalar tensor expected to be 0 dim but it has {tensor.dim()}"
            f" dimensions and {tensor.numel()} elements."
        )
    non_empty = [t for t in tensors if t.numel() > 0]

    def compute_fn(tensors, tensor, alpha):
        return tuple(torch.sub(t, tensor, alpha=alpha) for t in tensors)

    def postprocess_fn(results):
        return None

    if not non_empty:
        return compute_fn, (), {}, postprocess_fn

    return compute_fn, (non_empty, tensor, alpha), {"out": tuple(non_empty)}, postprocess_fn


@register_aten(
    ["aten::_foreach_div.Scalar", "aten::_foreach_div.Scalar_out"], uses_preprocessing=True
)
def torch_foreach_div_scalar(tensors, scalar, out=None):
    """Foreach divide with scalar."""
    _check_tensors(tensors)
    tensors = [
        torch.empty(t.shape, dtype=t.dtype, device=t.device) if t.numel() == 0 else t
        for t in tensors
    ]

    def actual_fn(tensors, scalar):
        result = []
        for t in tensors:
            if t.numel() == 0:
                result.append(t)
            else:
                result.append(torch.div(t, scalar))
        return tuple(result)

    return actual_fn, (tensors, scalar), {}


@register_aten(["aten::_foreach_div_.Scalar"], uses_preprocessing=True)
def torch_foreach_div_scalar_inplace(tensors, scalar, out=None):
    """Foreach divide inplace with scalar."""
    _check_tensors(tensors)
    non_empty = [t for t in tensors if t.numel() > 0]

    def compute_fn(tensors, scalar):
        return tuple(torch.div(t, scalar) for t in tensors)

    def postprocess_fn(results):
        return None

    if not non_empty:
        return compute_fn, (), {}, postprocess_fn

    return compute_fn, (non_empty, scalar), {"out": tuple(non_empty)}, postprocess_fn


@register_aten(
    [
        "aten::_foreach_div.List",
        "aten::_foreach_div.List_out",
    ],
    uses_preprocessing=True,
)
def torch_foreach_div_list(tensors1, tensors2, out=None):
    """Foreach divide between tensor lists."""
    _check_tensors_tensors(tensors1, tensors2)
    tensors1 = [
        torch.empty(t.shape, dtype=t.dtype, device=t.device) if t.numel() == 0 else t
        for t in tensors1
    ]

    def actual_fn(tensors1, tensors2):
        result = []
        for t1, t2 in zip(tensors1, tensors2, strict=False):
            if t1.numel() == 0:
                result.append(t1)
            else:
                result.append(torch.div(t1, t2))
        return tuple(result)

    return actual_fn, (tensors1, tensors2), {}


@register_aten(
    [
        "aten::_foreach_div.ScalarList",
        "aten::_foreach_div.ScalarList_out",
    ],
    uses_preprocessing=True,
)
def torch_foreach_div_scalarlist(tensors, scalars, out=None):
    """Foreach divide with scalar list."""
    _check_tensors_scalars(tensors, scalars)
    tensors = [
        torch.empty(t.shape, dtype=t.dtype, device=t.device) if t.numel() == 0 else t
        for t in tensors
    ]

    def actual_fn(tensors, scalars):
        result = []
        for t, s in zip(tensors, scalars, strict=False):
            if t.numel() == 0:
                result.append(t)
            else:
                result.append(torch.div(t, s))
        return tuple(result)

    return actual_fn, (tensors, scalars), {}


@register_aten(["aten::_foreach_div_.List"], uses_preprocessing=True)
def torch_foreach_div_list_inplace(tensors1, tensors2, out=None):
    """Foreach divide inplace between tensor lists."""
    _check_tensors_tensors(tensors1, tensors2)
    pairs = [(t1, t2) for t1, t2 in zip(tensors1, tensors2, strict=False) if t1.numel() > 0]

    def compute_fn(tensors1, tensors2):
        return tuple(torch.div(t1, t2) for t1, t2 in zip(tensors1, tensors2, strict=False))

    def postprocess_fn(results):
        return None

    if not pairs:
        return compute_fn, (), {}, postprocess_fn

    non_empty1, non_empty2 = zip(*pairs, strict=False)
    return (
        compute_fn,
        (list(non_empty1), list(non_empty2)),
        {"out": tuple(non_empty1)},
        postprocess_fn,
    )


@register_aten(["aten::_foreach_div_.ScalarList"], uses_preprocessing=True)
def torch_foreach_div_scalarlist_inplace(tensors, scalars, out=None):
    """Foreach divide inplace with scalar list."""
    _check_tensors_scalars(tensors, scalars)
    for t, s in zip(tensors, scalars, strict=False):
        if t.numel() > 0:
            t.div_(s)

    def noop(tensors):
        return None

    return noop, (torch.empty(0),), {}


@register_aten(
    ["aten::_foreach_div.Tensor", "aten::_foreach_div.Tensor_out"], uses_preprocessing=True
)
def torch_foreach_div_tensor(tensors, tensor, out=None):
    """Foreach divide with single tensor."""
    _check_tensors(tensors)
    if tensor.dim() != 0:
        raise RuntimeError(
            f"scalar tensor expected to be 0 dim but it has {tensor.dim()}"
            f" dimensions and {tensor.numel()} elements."
        )
    tensors = [
        torch.empty(t.shape, dtype=t.dtype, device=t.device) if t.numel() == 0 else t
        for t in tensors
    ]

    def actual_fn(tensors, tensor):
        result = []
        for t in tensors:
            if t.numel() == 0:
                result.append(t)
            else:
                result.append(torch.div(t, tensor))
        return tuple(result)

    return actual_fn, (tensors, tensor), {}


@register_aten(["aten::_foreach_div_.Tensor"], uses_preprocessing=True)
def torch_foreach_div_tensor_inplace(tensors, tensor, out=None):
    """Foreach divide inplace with single tensor."""
    _check_tensors(tensors)
    if tensor.dim() != 0:
        raise RuntimeError(
            f"scalar tensor expected to be 0 dim but it has {tensor.dim()}"
            f" dimensions and {tensor.numel()} elements."
        )
    non_empty = [t for t in tensors if t.numel() > 0]

    def compute_fn(tensors, tensor):
        return tuple(torch.div(t, tensor) for t in tensors)

    def postprocess_fn(results):
        return None

    if not non_empty:
        return compute_fn, (), {}, postprocess_fn

    return compute_fn, (non_empty, tensor), {"out": tuple(non_empty)}, postprocess_fn
