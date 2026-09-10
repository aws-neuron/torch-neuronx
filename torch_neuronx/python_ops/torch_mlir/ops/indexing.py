"""Indexing, slicing, and selection operations."""

import torch

from torch_neuronx.dtype_utils import torch_dtype_to_scalar_type_name
from torch_neuronx.python_ops.nki_kernels.bound_check import assert_index_in_bounds
from torch_neuronx.utils import skip_op_preconditions

from ..operation_registry import Preprocessing, register_aten
from ..validation import validate_inplace_broadcast
from .unary import _validate_float_promoting_out


def convert_indices(indices):
    """Convert boolean indices into integer indices."""
    integer_indices = []
    for index in indices:
        if index is None:
            integer_indices.append(index)
        elif index.dtype == torch.bool:
            integer_indices.extend(torch.nonzero(index, as_tuple=True))
        else:
            integer_indices.append(index.to("neuron"))
    return tuple(integer_indices)


def index_checking(tensor, indices):
    """Validate index dtypes for index operations. Bounds checking is fused
    in-graph (see index.py index_fn) to avoid a host sync; an OOB index faults
    on-device and aten::index's error_handler surfaces the exact exception."""
    for _, idx in enumerate(indices):
        if idx is None or idx.dtype == torch.bool:
            continue

        # Index dtype check
        if idx.is_floating_point():
            raise IndexError("tensors used as indices must be long, int, byte or bool tensors")


def _device_bounds_check(data, dim, index, *, wrap) -> torch.Tensor:
    """Device bounds check (no host sync); returns the checked index.

    Int32-exact, any ``dim``, any dim size. The check window is [-n, n) for wrap
    ops (negatives accepted) or [0, n) otherwise, applied directly via the
    kernel's lower bound (no host ``torch.where`` normalization). On an OOB fault
    the runtime invokes the op's registered ``error_handler`` (with the op's
    inputs reconstructed from the OperationContext) to raise the exact
    op-specific exception; no host precondition is needed.

    Issued inside the op's compute_fn: the caller MUST consume the returned
    index (the data dependency keeps the in-graph check alive through DCE — see
    bound_check._mask_bounds_check).
    """
    ndim = data.dim()
    # 0-d (scalar) self has an effective bound of 1 (one addressable element);
    # this also avoids a `dim % 0`. n == 0 (empty dim) is handled by the kernel
    # (every index is out of [0, 0) -> faults).
    n = data.shape[dim % ndim] if ndim >= 1 else 1

    return assert_index_in_bounds(index, n, lo=(-n if wrap else 0))


def _assert_no_internal_overlap(tensor, op_name=""):
    """Raise RuntimeError if tensor has internal memory overlap.

    Matches CPU PyTorch's ``assert_no_internal_overlap`` semantics: a tensor
    whose strides cause multiple elements to alias the same memory location
    cannot be used as an output.
    """
    if tensor is None:
        return
    for stride, size in zip(tensor.stride(), tensor.shape, strict=False):
        if stride == 0 and size > 1:
            raise RuntimeError(
                "unsupported operation: more than one element of the written-to "
                "tensor refers to a single memory location. Please clone() the "
                "tensor before performing the operation."
            )


def _validate_index_select(x, dim, index, out=None):
    """Validate index_select dtypes on raw args."""
    if index.dtype not in [torch.int32, torch.int64]:
        raise TypeError("index_select(): Expected dtype int32 or int64 for index")


def _index_select_error_handler(ctx):
    """Raise index_select's exact out-of-bounds exception on an OOB device fault
    (see op_error_handlers.py). index_select uses a fixed message with no index
    value, so the context is not needed."""
    raise IndexError("index out of range in self")


@register_aten(
    ["aten::index_select", "aten::index_select.out"],
    static_argnums=(1,),
    preprocessing=Preprocessing.SHAPE_ONLY,
    validate_static_inputs=_validate_index_select,
    error_handler=_index_select_error_handler,
)
def torch_index_select(x, dim, index, out=None):
    _assert_no_internal_overlap(out)
    if index.dim() == 0:
        index = index.reshape(1)
    elif index.dim() != 1:
        raise IndexError("index_select(): Index is supposed to be a vector")

    # 0-d (scalar) input: at most one index is allowed. This check is
    # shape-based, so it is safe at trace time (shape is part of the compile
    # cache key -> a different numel re-traces and re-checks). The *value*
    # bounds check must NOT be eager here: this function only runs on a cache
    # miss, so a value-dependent host check is skipped when a same-shape call
    # (e.g. index [0], then [1]) hits the cache. The bounds are enforced
    # in-graph below for every rank: for 0-d self the effective bound is 1
    # (window [-1, 1)), see _device_bounds_check.
    if not skip_op_preconditions() and x.dim() == 0 and index.numel() > 1:
        raise RuntimeError(
            f"index_select(): Index to scalar can have only 1 value, "
            f"got {index.numel()} value(s)"
        )

    def index_select_fn(x, dim, index):
        # In-graph bounds check (no host sync): the gather consumes the checked
        # index, so the check survives DCE and faults on an OOB index;
        # _index_select_error_handler then surfaces the exact exception.
        # Window [-n, n) matches the host check (n == 1 for 0-d self).
        # Skipped under skip_op_preconditions.
        if not skip_op_preconditions():
            index = _device_bounds_check(x, dim, index, wrap=True)
            if x.dim() == 0:
                # A gather from 0-d self folds to a reshape (any in-bounds
                # index selects the single element), dropping the checked
                # index — the backend then dead-stores the whole check (the
                # HLO-level has_side_effect attr does not survive NKI inlining,
                # see neuronx-cc DeadCodeElim/eliminateDeadStore). Consume the
                # checked index through a gather the compiler CANNOT fold: pad
                # self to 2 rows and gather with clamp(index, min=0) — a
                # runtime index into a size-2 dim. Valid checked values
                # {-1, 0} both clamp to row 0 (== x); an OOB value faults in
                # the check before this gather's result is consumed.
                x2 = torch.cat([x.reshape(1), torch.zeros_like(x.reshape(1))])
                idx0 = torch.clamp(index, min=0)
                return torch.index_select(x2, 0, idx0).reshape(x.shape)
        return torch.index_select(x, dim, index)

    return index_select_fn, (x, dim, index), {}


def _validate_index_copy(x, dim, indexes, source, out=None):
    """Validate index_copy dtypes on raw args."""
    if indexes.dtype not in (torch.int32, torch.int64):
        raise RuntimeError("index_copy_(): Expected a long tensor for index, but got Float")


@register_aten(
    ["aten::index_copy", "aten::index_copy_", "aten::index_copy.out"],
    static_argnums=(1,),
    validate_static_inputs=_validate_index_copy,
)
def torch_index_copy(x, dim, indexes, source, out=None):
    # 0-d tensor: index_copy on a scalar is valid when dim=0, index has
    # exactly one element (value 0), and source is also 0-d.  CPU PyTorch
    # handles this by treating the scalar as a 1-element container.
    # Skip the shape validation that assumes at least 1 dimension, but
    # still reject the cases CPU rejects (e.g. multiple indices into a
    # single-element scalar destination when source is also a scalar).
    if x.dim() == 0:
        if source.dim() == 0 and indexes.numel() > 1:
            raise IndexError(
                f"index_copy_(): When source is scalar, index should "
                f"have one element (got {indexes.numel()})"
            )
    else:
        actual_dim = dim if dim >= 0 else x.dim() + dim
        expected_shape = list(x.shape)
        expected_shape[actual_dim] = indexes.numel()
        if list(source.shape) != expected_shape:
            raise ValueError(
                f"Incompatible shapes for broadcasting: {tuple(source.shape)}"
                f" and requested shape {tuple(expected_shape)}"
            )
    return torch.index_copy(x, dim, indexes, source)


@register_aten(
    ["aten::index_add", "aten::index_add_", "aten::index_add.out"],
    static_argnums=(1,),
    static_argnames=("alpha",),
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_index_add(x, dim, index, source, alpha=1, out=None):
    return torch.index_add(x, dim, index, source, alpha=alpha)


@register_aten(["aten::gather", "aten::gather.out"], static_argnums=(1,))
def torch_gather(x, dim, index, *, sparse_grad=False, out=None):
    return torch.gather(x, dim, index)


def _scatter_error_handler(ctx):
    """Raise scatter's exact out-of-bounds exception. Invoked by the C++ error
    path on an OOB device fault (see op_error_handlers.py) with a lazy context
    over scatter's inputs (x, dim, index, value/src) — it does not re-run the
    op. Only the index is copied to CPU; the size comes from metadata. Negatives
    are accepted in [-size, size) to match scatter.
    """
    dim = ctx.scalar(1)
    shape = ctx.shape(0)
    size = shape[dim] if len(shape) > 0 else 1
    index = ctx.tensor(2)
    raise RuntimeError(
        f"index {torch.max(index.abs()).item()} is out of "
        f"bounds for dimension {dim} with size {size}"
    )


@register_aten(
    ["aten::scatter", "aten::scatter.value_out"],
    static_argnums=(1,),
    preprocessing=Preprocessing.SHAPE_ONLY,
    error_handler=_scatter_error_handler,
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_scatter_value(x, dim, index, value, out=None):
    # Convert scalar to tensor with matching dtype
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, dtype=x.dtype, device=x.device)

    def torch_scatter_value_fn(x, dim, index, value, out=None):
        # In-graph bounds check: scatter consumes the checked index (keeps the
        # check alive through DCE, ordered before the scatter).
        if not skip_op_preconditions():
            index = _device_bounds_check(x, dim, index, wrap=True)
        if value.dim() == 0:
            # without expanding, we hit dimension mismatch issues
            value = value.expand_as(index)
        if value.dtype != x.dtype:
            value = value.to(x.dtype)
        return torch.scatter(x, dim, index, value)

    return torch_scatter_value_fn, (x, dim, index, value), {"out": out}


@register_aten(
    [
        "aten::scatter.src",
        "aten::scatter.src_out",
        "aten::scatter.reduce",
        "aten::scatter.value_reduce_out",
        "aten::scatter.reduce_out",
    ],
    static_argnums=(1,),
    static_argnames=("reduce",),
    preprocessing=Preprocessing.SHAPE_ONLY,
    error_handler=_scatter_error_handler,
)
def torch_scatter_src(x, dim, index, src, out=None, reduce=None):
    def torch_scatter_src_fn(x, dim, index, src, out=None, reduce=None):
        # In-graph bounds check: scatter consumes the checked index.
        if not skip_op_preconditions():
            index = _device_bounds_check(x, dim, index, wrap=True)
        if isinstance(src, torch.Tensor) and src.dim() == 0:
            # without expanding, we hit dimension mismatch issues
            src = src.expand_as(index)
        if reduce is None:
            return torch.scatter(x, dim, index, src)

        # Fallback/workaround for unsupported reduce operations
        if reduce == "add":
            # scatter_add requires a tensor src whose dtype matches x.dtype.
            # Upstream scatter(..., reduce='add') tolerates dtype mismatches
            # (e.g. scalar float src with integer self, or a 0-dim src tensor
            # of a different dtype), so replicate that behavior here by
            # casting src to x.dtype before dispatching to scatter_add.
            if not isinstance(src, torch.Tensor):
                src = torch.full_like(index, src, dtype=x.dtype)
            if src.dtype != x.dtype:
                src = src.to(x.dtype)
            return torch.scatter_add(x, dim, index, src)
        elif reduce == "multiply":
            # Use scatter with multiplication: multiplier = ones at positions
            # that aren't scattered, src at positions that are. Upstream
            # scatter(..., reduce='multiply') tolerates dtype mismatches, so
            # normalize src to x.dtype here so the inner scatter doesn't fail
            # the meta-level self.dtype == src.dtype check.
            if not isinstance(src, torch.Tensor):
                src = torch.full_like(index, src, dtype=x.dtype)
            if src.dtype != x.dtype:
                src = src.to(x.dtype)
            ones = torch.ones_like(x)
            multiplier = torch.scatter(ones, dim, index, src)
            return x * multiplier
        else:
            raise RuntimeError(f"reduce argument must be either add or multiply. Got {reduce}")

    # handle out and reduce kwargs
    out_kwargs = {}
    if out is not None:
        out_kwargs["out"] = out
    if reduce is not None:
        if reduce not in ("add", "multiply"):
            raise RuntimeError(f"reduce argument must be either add or multiply. Got {reduce}")
        out_kwargs["reduce"] = reduce

    return torch_scatter_src_fn, (x, dim, index, src), out_kwargs


@register_aten(
    ["aten::masked_select", "aten::masked_select.out", "aten::masked_select.Tensor_out"],
    static_argnums=(2,),
    preprocessing=Preprocessing.VALUE_DEPENDENT,
    # Output size is mask.sum().item(), which the raw cache key cannot capture, so
    # re-lower every call and key on the resolved size. That size usually differs
    # per call, so the NEFF recompiles whenever the count changes.
    relower_every_call=True,
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_masked_select(x, mask, out=None):
    if mask.dtype != torch.bool:
        raise RuntimeError(
            f"masked_select: expected BoolTensor for mask, but got {mask.dtype} tensor"
        )
    size = mask.sum().item()

    # Correct size after broadcast outside kernel function to ensure it is traced correctly
    broadcast_shape = torch.broadcast_shapes(x.shape, mask.shape)
    size = int(size * broadcast_shape.numel() / mask.shape.numel())

    def masked_select_fn(x, mask, size, out=None):
        broadcast_shape = torch.broadcast_shapes(x.shape, mask.shape)

        # A 0-d broadcast shape (both self and mask are scalars) has no
        # coordinate axes, which nonzero_static cannot express: the size==0 path
        # stacks an empty coordinate list ("stack expects a non-empty
        # TensorList") and the size==1 path indexes shape[-1] (IndexError).
        # Viewing both operands as 1-element vectors gives the same result --
        # masked_select always returns a 1-d tensor.
        if not broadcast_shape:
            broadcast_shape = torch.Size([1])

        if x.shape != broadcast_shape:
            x = x.broadcast_to(broadcast_shape)
        if mask.shape != broadcast_shape:
            mask = mask.broadcast_to(broadcast_shape)

        indices = torch.nonzero_static(mask, size=size, fill_value=-1)
        return x[tuple(indices[:, i] for i in range(indices.shape[1]))]

    return masked_select_fn, (x, mask, size), {"out": out}


def _validate_masked_fill(x, mask, value, out=None, **_kwargs):
    """Enforce masked_fill's input contract before the lowering runs.

    Raising here (rather than letting the traced ``torch.masked_fill`` fail and
    the dispatcher retry on CPU) keeps the exact ATen message: with
    ``TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1`` the fallback layer
    replaces the op's own error with "CPU fallback disabled ...".
    """
    if isinstance(mask, torch.Tensor) and mask.dtype != torch.bool:
        raise RuntimeError(
            f"masked_fill_ only supports boolean masks, but got mask with dtype {mask.dtype}"
        )
    if isinstance(value, torch.Tensor) and value.dim() != 0:
        raise RuntimeError(
            "masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
            f"with {value.dim()} dimension(s)."
        )


def _validate_masked_fill_inplace(x, mask, value, out=None, **kwargs):
    """masked_fill_: the input contract plus the no-expanding-destination rule."""
    _validate_masked_fill(x, mask, value, out=out, **kwargs)
    validate_inplace_broadcast(x, mask, **kwargs)


# masked_fill has four overloads plus two in-place ones. Only the ``_`` variants
# have a backend dispatch entry in native_functions.yaml; ``masked_fill.Scalar``,
# ``.Tensor`` and the two ``_out`` forms are CompositeExplicitAutograd. They are
# registered anyway so each lowers as a single Neuron kernel instead of
# decomposing into clone/copy_ + masked_fill_, and so the executed-op log names
# the overload that actually ran.
#
# The result always keeps ``self.dtype`` — masked_fill does not type-promote
# against ``value`` — so the lowering must not run ``result_type``.
#
# The functional and ``_out`` overloads share one registration (the ``_out``
# forms alias their ``out=`` buffer through the standard out-tensor path, so no
# ``output_params`` is needed). Only the in-place ``_`` variants take a separate
# decorator, for the no-expanding-destination validator.
@register_aten(
    [
        "aten::masked_fill.Scalar",
        "aten::masked_fill.Tensor",
        "aten::masked_fill.Scalar_out",
        "aten::masked_fill.Tensor_out",
    ],
    validate_static_inputs=_validate_masked_fill,
)
@register_aten(
    ["aten::masked_fill_.Scalar", "aten::masked_fill_.Tensor"],
    validate_static_inputs=_validate_masked_fill_inplace,
)
def torch_masked_fill(x, mask, value, out=None):
    return torch.masked_fill(x, mask, value)


# NOTE: where.ScalarOther, where.ScalarSelf, and where.Scalar are NOT registered
# because they are CompositeImplicitAutograd ops that decompose to where.self:
# - ScalarOther: converts scalar to tensor, calls where.self
# - ScalarSelf: converts scalar to tensor, calls where.self
# - Scalar: converts both scalars to tensors, calls where.self
# Registering them would break the autograd chain.
# Reference: https://github.com/pytorch/pytorch/blob/8e65dfa6f2c451dc3c71d76d68796d1271c72772/aten/src/ATen/native/TensorCompare.cpp#L642


@register_aten(
    [
        "aten::where.self",
        "aten::where.self_out",
    ],
    validate_static_inputs=lambda condition,
    x=None,
    y=None,
    out=None,
    **kw: _validate_float_promoting_out(x if x is not None else condition, out=out),
)
def torch_where(condition, x=None, y=None, out=None):
    return torch.where(condition, x, y)


def _validate_nonzero_static(
    tensor, size=None, fill_value=-1, out=None, with_count=False, dtype=None
):
    """Validate nonzero_static dtypes on raw (pre-downcast) args."""
    if (dtype is not None and dtype != torch.int64) or (
        out is not None and out.dtype != torch.int64
    ):
        raise RuntimeError(
            "nonzero_static: Expected out tensor to have scalar type Long "
            "but got scalar type " + str(out.dtype if out is not None else dtype)
        )


@register_aten(
    ["aten::nonzero_static", "aten::nonzero_static.out"],
    static_argnames=("size", "fill_value", "dtype"),
    validate_static_inputs=_validate_nonzero_static,
)
def torch_nonzero_static(tensor, size=None, fill_value=-1, out=None, with_count=False, dtype=None):
    return torch.nonzero_static(tensor, size=size, fill_value=fill_value)


@register_aten(
    ["aten::topk", "aten::topk.values"],
    static_argnums=(1, 2, 3, 4),
)
def torch_topk(x, k, dim=-1, largest=True, sorted=True, values=None, indices=None):
    if x.ndim == 0:
        vals, idxs = torch.topk(x.unsqueeze(0), k, dim=0, largest=largest, sorted=sorted)
        return vals.squeeze(0), idxs.squeeze(0)
    return torch.topk(x, k, dim, largest=largest, sorted=sorted)


def _translate_index_put_args(*args, **kwargs):
    """Move CPU index tensors to device before can_handle."""
    if len(args) >= 2 and isinstance(args[1], list | tuple):
        x = args[0]
        indices = args[1]
        moved = [
            idx.to(x.device) if isinstance(idx, torch.Tensor) and idx.device != x.device else idx
            for idx in indices
        ]
        args = (x, moved, *args[2:])
    return args, kwargs


def _index_put_error_handler(ctx):
    """Raise index_put's exact out-of-bounds exception. Invoked by the C++ error
    path on an OOB device fault (see op_error_handlers.py) with a lazy context
    over index_put's inputs (x, indices, values) — it does not re-run the op.
    Only the index tensors are copied to CPU; the sizes come from metadata.
    Iterates the index list to report the offending dim/value; negatives are
    accepted in [-size, size) to match index_put.
    """
    shape = ctx.shape(0)
    indices = ctx.tensor_list(1)
    for dim, idx in enumerate(indices):
        if idx is None or not isinstance(idx, torch.Tensor) or idx.dtype == torch.bool:
            continue
        size = shape[dim]
        if bool(torch.any(idx >= size)) or bool(torch.any(idx < -size)):
            raise IndexError(
                f"index {torch.max(idx).item()} is out of bounds "
                f"for dimension {dim} with size {size}"
            )


def _bool_mask_true_count(idx):
    """True-count of a boolean index, or None if ``idx`` is not one.

    This count is the only tensor *value* index_put's preprocessing reads. It is
    the length of the integer indices the mask expands into, which the raw cache
    key cannot see — hence VALUE_DEPENDENT + relower_every_call.
    """
    if isinstance(idx, torch.Tensor) and idx.dtype == torch.bool:
        return int(idx.sum().item())
    return None


def _expanded_index_specs(indices, mask_counts):
    """Per-slot shapes of the integer indices the graph will use, ``None`` kept.

    A boolean mask of rank ``n`` with ``k`` True entries expands to ``n`` slots of
    shape ``(k,)`` — the shapes ``torch.nonzero(mask, as_tuple=True)`` would
    produce. A rank-0 mask expands to no slots at all. ``None`` slots are kept
    because they say which dimension the later slots index.
    """
    specs = []
    for idx, count in zip(indices, mask_counts, strict=True):
        if idx is None:
            specs.append(None)
        elif count is None:
            specs.append(tuple(idx.shape))
        else:
            specs.extend([(count,)] * idx.dim())
    return specs


def _indexing_result_shape(x, specs):
    """Shape ``values`` has to be expandable to, i.e. ATen's ``info.src.sizes()``.

    Derived by running ``aten::index`` on meta tensors so PyTorch owns the
    advanced-indexing rule rather than this file restating it. With no indexing
    slot the result is ``x`` itself, which is what ATen reports too (``a[True]`` on
    a ``(2, 3)`` tensor reports ``[2, 3]``, and an empty index list on a 0-dim
    ``x`` reports ``[]``).
    """
    if all(s is None for s in specs):
        return tuple(x.shape)
    # ATen raises IndexError here; aten::index's meta kernel raises RuntimeError with
    # the same text, so the check is done before the meta call rather than after.
    if len(specs) > x.dim():
        raise IndexError(f"too many indices for tensor of dimension {x.dim()} (got {len(specs)})")
    meta_x = torch.empty(x.shape, dtype=x.dtype, device="meta")
    meta_indices = [
        None if s is None else torch.empty(s, dtype=torch.int64, device="meta") for s in specs
    ]
    return tuple(torch.ops.aten.index(meta_x, meta_indices).shape)


def _check_index_put_shape(x, values, index_shapes):
    """``values`` must be expandable to the indexing result shape.

    ATen enforces this per backend rather than in a shared layer, so replacing the
    implementation means owning it: CPU in ``make_index_put_iterator``
    (``TensorAdvancedIndexing.cpp``), CUDA again in ``cuda/Indexing.cu``. Both raise
    ``RuntimeError`` synchronously with this message (measured on an H200), and
    without the check an invalid call reaches the compiler instead and surfaces a
    ``TorchMlirCompilerError``, which is not a ``RuntimeError``.
    """
    result_shape = _indexing_result_shape(x, index_shapes)
    # ATen's is_expandable_to (ExpandUtils.h). Deliberately not
    # torch.broadcast_shapes(...) == result_shape, which decides the same question but
    # raises for two non-1 unequal dims instead of returning False, hiding the message
    # below for the most common mismatch.
    value_shape = tuple(values.shape)
    expandable = len(value_shape) <= len(result_shape) and all(
        v in (r, 1) for v, r in zip(reversed(value_shape), reversed(result_shape), strict=False)
    )
    if not expandable:
        raise RuntimeError(
            f"shape mismatch: value tensor of shape {list(values.shape)} "
            f"cannot be broadcast to indexing result of shape {list(result_shape)}"
        )


def _dtype_category(dtype):
    """bool / integral / floating / complex, the coarse half of a dtype."""
    if dtype == torch.bool:
        return "bool"
    if dtype.is_complex:
        return "complex"
    if dtype.is_floating_point:
        return "floating"
    return "integral"


def _check_index_put_dtype(x, values):
    """``values`` and ``x`` must be in the same dtype category.

    ATen requires exact equality here (same sites as the shape check). Categories are
    deliberately looser, for the reason in the TODO below. They still catch an integral
    source into a floating destination or the reverse, at any width.
    """
    # TODO: compare dtypes exactly, like ATen, once 0-dim promotion is fixed. A 0-dim int32
    # times a Python int returns int64 here, int32 on CPU, because PreprocessArgs rebuilds the
    # Python int as a real 0-dim int64 tensor. index_add's decomposition does that multiply and
    # calls index_put, so an exact compare would raise on a valid call. Only errors differ.
    if _dtype_category(values.dtype) != _dtype_category(x.dtype):
        dst = torch_dtype_to_scalar_type_name(x.dtype) or str(x.dtype)
        src = torch_dtype_to_scalar_type_name(values.dtype) or str(values.dtype)
        raise RuntimeError(
            "Index put requires the source and destination dtypes match, got "
            f"{dst} for the destination and {src} for the source."
        )


@register_aten(
    ["aten::_index_put_impl_"],
    # Matched against the preprocessed kwargs by name, so the synthesized ones count.
    static_argnames=("accumulate", "unsafe", "use_masked_fill", "mask_counts"),
    preprocessing=Preprocessing.VALUE_DEPENDENT,
    translate_args_fn=_translate_index_put_args,
    # A boolean mask's True-count is not in the raw cache key, so the dispatch
    # entry must not be replayed.
    relower_every_call=True,
    error_handler=_index_put_error_handler,
)
def torch_index_put_impl(x, indices, values, accumulate=False, unsafe=False, out=None):
    """Torch MLIR implementation of torch._index_put_impl_.

    Takes one of three routes, all of which pass ``indices`` through unchanged:

    - a single boolean mask with a one-element ``values`` becomes a ``where``
    - a mask that selects nothing returns ``x``
    - otherwise the masks expand to integer indices in-graph and ``x`` is scattered into

    ``mask_counts`` carries each mask's True-count, which the expansion needs as a
    compile-time size.
    """

    # Validate index dtypes eagerly (raises before dispatch). The bounds check is
    # fused into _index_put_fn below (in-graph), so an OOB index faults as
    # aten::_index_put_impl_ and its error handler surfaces the exact exception.
    # Both are skipped for unsafe=True.
    if not unsafe:
        for idx in indices:
            if (
                isinstance(idx, torch.Tensor)
                and idx.dtype != torch.bool
                and idx.is_floating_point()
            ):
                raise IndexError("tensors used as indices must be long, int, byte or bool tensors")

    def can_dispatch_to_masked_fill(x, indices, values):
        """Check if index_put can be optimized to masked_fill."""
        if values.numel() != 1 or accumulate:
            return False

        num_defined_indices = 0
        mask = None

        for index in indices:
            if index is not None:
                if hasattr(index, "dtype") and index.dtype == torch.bool:
                    if mask is not None:  # Already found a mask
                        return False
                    mask = index
                    # Check shape compatibility
                    for j in range(index.ndim):
                        if index.shape[j] != x.shape[num_defined_indices + j]:
                            return False
                    num_defined_indices += index.ndim
                else:
                    return False  # Non-boolean index found
            else:
                num_defined_indices += 1

        return mask is not None

    # Check if we can dispatch to masked_fill optimization
    can_dispatch = can_dispatch_to_masked_fill(x, indices, values)

    # The where route rewrites values in-graph, so it would accept a rank ATen
    # rejects. Its result has rank x.dim() - mask.dim() + 1, and every dim of a
    # one-element values is 1, so the rank alone decides expandability here.
    if can_dispatch:
        mask = next(i for i in indices if i is not None)
        if values.dim() > x.dim() - mask.dim() + 1:
            can_dispatch = False  # fall through and be validated with the counts

    # Reading the counts costs a host sync, so only do it on the scatter route —
    # the masked_fill route consumes the mask directly and needs no count.
    mask_counts = None if can_dispatch else tuple(_bool_mask_true_count(i) for i in indices)

    def _index_put_fn(
        x,
        indices,
        values,
        accumulate=False,
        unsafe=False,
        use_masked_fill=False,
        mask_counts=None,
        out=None,
    ):
        if use_masked_fill:
            dim = next(i for i, idx in enumerate(indices) if idx is not None)
            mask = indices[dim]
            # Line the mask up with the dims it indexes — [dim, dim + mask.dim()) —
            # so the select broadcasts over the rest. In-graph, so the mask stays
            # the runtime input the raw arguments declare.
            aligned = (1,) * dim + tuple(mask.shape) + (1,) * (x.dim() - dim - mask.dim())
            # Cannot directly call torch.masked_fill since it only takes float object for value
            return torch.where(mask.reshape(aligned), values.to(x.dtype), x)

        # A mask that selects nothing writes nothing, and index_put returns self.
        # Expanding it would put a zero-size tensor in the graph, which the Neuron
        # compiler cannot lower.
        if any(count == 0 for count in mask_counts):
            return x.clone()

        # Expand boolean masks to integer indices. nonzero_static rather than
        # nonzero because mask_counts makes the size a compile-time constant, so
        # the expansion traces to a static shape. fill_value is unreachable (the
        # size is exact) but 0 keeps any padding in bounds.
        expanded = []
        for idx, count in zip(indices, mask_counts, strict=True):
            if count is None:
                expanded.append(idx)
                continue
            coords = torch.nonzero_static(idx, size=count, fill_value=0)
            expanded.extend(coords[:, c] for c in range(coords.shape[1]))

        # Normalize negative indices and ensure proper tensor format
        normalized_indices = []
        for i, idx in enumerate(expanded):
            if idx is not None and isinstance(idx, torch.Tensor):
                dim_size = x.shape[i]
                idx = torch.where(idx < 0, idx + dim_size, idx)
                # In-graph bounds check (post-normalization -> [0, dim_size)):
                # index_put is a genuine scatter that consumes the index, so the
                # check survives DCE and faults as aten::_index_put_impl_ on an
                # OOB index; its error handler then surfaces the exact exception.
                # Skipped for unsafe.
                if not unsafe and not skip_op_preconditions():
                    idx = assert_index_in_bounds(idx, dim_size, lo=0)
                # Ensure scalar indices are properly shaped for MLIR
                if idx.dim() == 0:
                    idx = idx.unsqueeze(0)  # Convert scalar to [1] shape
            normalized_indices.append(idx)

        return torch.ops.aten.index_put(x, tuple(normalized_indices), values, accumulate=accumulate)

    # Preconditions, in ATen's order: make_info broadcasts the indices together,
    # then make_index_put_iterator checks values against the result.
    if not can_dispatch:
        # Check if the indexing tensors can be broadcast together, on the shapes
        # the in-graph expansion will produce.
        specs = _expanded_index_specs(indices, mask_counts)
        expanded_shapes = [s for s in specs if s is not None]
        if len(expanded_shapes) > 1:
            try:
                torch.broadcast_shapes(*expanded_shapes)
            except RuntimeError:
                # Incompatible shape cause IndexError for index_put
                shapes_str = ", ".join(str(list(s)) for s in expanded_shapes)
                raise IndexError(
                    "shape mismatch: indexing tensors could not be broadcast together "
                    f"with shapes {shapes_str}"
                ) from None
        _check_index_put_shape(x, values, specs)
    _check_index_put_dtype(x, values)

    return (
        _index_put_fn,
        (x, indices, values),
        {
            "accumulate": accumulate,
            "unsafe": unsafe,
            "use_masked_fill": can_dispatch,
            "mask_counts": mask_counts,
            "out": out,
        },
    )


def _validate_index_select_backward(grad, input_sizes, dim, index):
    """Validate index_select_backward dtypes on raw args. The 1-d+ bounds check
    is fused in-graph (see index_select_backward_fn) to avoid a host sync; the
    0-d scalar case is a cheap single-element check kept here."""
    if index.dtype not in [torch.int32, torch.int64]:
        raise TypeError(
            f"index_select_backward(): Expected dtype int32/int64 for index but got: {index.dtype}"
        )

    if skip_op_preconditions():
        return

    # 0-d self (input_sizes empty): the scalar grad holds a single element
    # addressable only at index 0. CPU rejects negatives and multiple indices.
    # Only the shape-based numel check is safe here: this validator runs at
    # lowering (compile-cache miss) only, and shape is part of the cache key.
    # The *value* bounds check is enforced in-graph (index_select_backward_fn)
    # so it runs on every call, including cache hits.
    if len(input_sizes) == 0 and index.numel() > 1:
        raise IndexError(
            f"index_select_backward(): Index to scalar can have only 1 value, "
            f"got {index.numel()} value(s)"
        )


def _index_select_backward_error_handler(ctx):
    """Raise index_select_backward's exact out-of-bounds exception on an OOB
    device fault (see op_error_handlers.py). Args: (grad, input_sizes, dim,
    index); input_sizes/dim are static scalars, index is the copied tensor."""
    input_sizes = ctx.scalar(1)
    dim = ctx.scalar(2)
    index = ctx.tensor(3)
    # 0-d self: input_sizes is empty; the single element gives an effective size 1.
    size = input_sizes[dim] if len(input_sizes) > 0 else 1
    oob = int(torch.min(index)) if bool(torch.any(index < 0)) else int(torch.max(index))
    raise IndexError(f"index {oob} is out of bounds for dimension {dim} with size {size}")


@register_aten(
    ["aten::index_select_backward"],
    static_argnums=(1, 2),
    preprocessing=Preprocessing.SHAPE_ONLY,
    validate_static_inputs=_validate_index_select_backward,
    error_handler=_index_select_backward_error_handler,
)
def torch_index_select_backward(grad, input_sizes, dim, index):
    def index_select_backward_fn(grad, input_sizes, dim, index):
        # 0-d self (input_sizes empty): CPU identity-returns the scalar grad,
        # but only index 0 is valid (CPU rejects negatives). Enforce in-graph
        # (window [0, 1)) and consume the checked index through a gather the
        # compiler cannot fold (pad grad to 2 rows, runtime index) — a gather
        # from the 0-d/size-1 grad folds to a reshape and the backend then
        # dead-stores the check (see torch_index_select).
        if len(input_sizes) == 0:
            if skip_op_preconditions():
                return grad
            index = assert_index_in_bounds(index, 1, lo=0)
            g2 = torch.cat([grad.reshape(1), torch.zeros_like(grad.reshape(1))])
            return torch.index_select(g2, 0, index.reshape(1)).reshape(grad.shape)
        # In-graph bounds check (no host sync); window [0, dim_size) rejects
        # negatives. Consumed by (index == arange) below, so it survives DCE and
        # faults on an OOB index; _index_select_backward_error_handler then
        # surfaces the exact exception.
        if not skip_op_preconditions():
            index = assert_index_in_bounds(index, input_sizes[dim], lo=0)
        arange = torch.arange(input_sizes[dim], dtype=index.dtype, device=index.device).unsqueeze(1)
        one_hot = (index == arange).to(grad.dtype)
        # 1-d grad (forward input was 1-d): movedim(dim, -2) has no second-to-last
        # dim. Note: 0-d grad is already handled by the len(input_sizes) == 0
        # short-circuit above, so this branch only fires for 1-d.
        # dim must be 0 here; [input_size, idx_numel] @ [idx_numel] -> [input_size].
        if grad.dim() == 1:
            return one_hot @ grad
        return (one_hot @ grad.movedim(dim, -2)).movedim(-2, dim)

    return index_select_backward_fn, (grad, input_sizes, dim, index), {}


@register_aten(["aten::slice_backward"], static_argnums=(1, 2, 3, 4, 5))
def torch_slice_backward(grad_output, input_sizes, dim, start, end, step):
    grad_input = torch.zeros(input_sizes, dtype=grad_output.dtype, device=grad_output.device)
    indices = [slice(None)] * len(input_sizes)
    indices[dim] = slice(start, end, step)
    grad_input[tuple(indices)] = grad_output
    return grad_input


@register_aten(["aten::select_backward"], static_argnums=(1, 2, 3))
def torch_select_backward(grad_output, input_sizes, dim, index):
    # Backward of select: place grad_output at position `index` along `dim` in a
    # zero tensor of shape input_sizes, zeros everywhere else. Decompose to
    # unsqueeze + pad so it lowers to stablehlo.pad rather than stablehlo.scatter
    # (which also carries an i64 index tensor).
    ndim = len(input_sizes)
    dim = dim % ndim
    size = input_sizes[dim]
    index = index % size
    grad = grad_output.unsqueeze(dim)  # restore the dim that select dropped (size 1)

    # F.pad reads (low, high) pairs from the last dim backward, so the slot for
    # `dim` is mirrored: index zero-slices before, size-index-1 zero-slices after.
    pad = [0, 0] * ndim
    pos = 2 * (ndim - 1 - dim)
    pad[pos] = index
    pad[pos + 1] = size - index - 1
    return torch.nn.functional.pad(grad, pad)


@register_aten(
    [
        "aten::searchsorted.Tensor",
        "aten::searchsorted.Tensor_out",
    ],
    static_argnames=("out_int32", "right", "side"),
)
def torch_searchsorted_tensor(
    sorted_sequence, values, *, out_int32=False, right=False, side=None, sorter=None, out=None
):
    """Decompose searchsorted into comparison + sum.

    O(N*M) linear scan that avoids data-dependent control flow.
    NOTE: Creates an intermediate [..., N, M] tensor; large inputs may cause OOM.
    """
    if side is not None:
        right = side == "right"
    if sorter is not None:
        sorted_sequence = torch.gather(sorted_sequence, -1, sorter)
    seq = sorted_sequence.unsqueeze(-2)
    val = values.unsqueeze(-1)
    mask = seq <= val if right else seq < val
    result = mask.sum(dim=-1)
    return result.to(torch.int32) if out_int32 else result.to(torch.int64)


@register_aten(
    [
        "aten::searchsorted.Scalar",
        "aten::searchsorted.Scalar_out",
    ],
    static_argnames=("out_int32", "right", "side"),
)
def torch_searchsorted_scalar(
    sorted_sequence, value, *, out_int32=False, right=False, side=None, sorter=None, out=None
):
    """Scalar variant: wrap scalar as tensor and delegate to tensor version."""
    values = (
        torch.as_tensor(value, dtype=sorted_sequence.dtype).reshape(1).to(sorted_sequence.device)
    )
    result = torch_searchsorted_tensor(
        sorted_sequence, values, out_int32=out_int32, right=right, side=side, sorter=sorter
    )
    return result.squeeze(-1)
