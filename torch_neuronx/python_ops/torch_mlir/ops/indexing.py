"""Indexing, slicing, and selection operations."""

import torch

from torch_neuronx.utils import skip_op_preconditions

from ..operation_registry import Preprocessing, register_aten


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
    """Validate indices for index operations."""
    for dim, idx in enumerate(indices):
        if idx is None or idx.dtype == torch.bool:
            continue

        # Index dtype check
        if idx.is_floating_point():
            raise IndexError("tensors used as indices must be long, int, byte or bool tensors")

        # Bounds checking on PyTorch tensors
        if not skip_op_preconditions():
            size = tensor.shape[dim]
            if torch.any(idx >= size) or torch.any(idx < -size):
                raise IndexError(
                    f"index {torch.max(idx).item()} is out of bounds "
                    f"for dimension {dim} with size {size}"
                )


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


@register_aten(
    ["aten::index_select", "aten::index_select.out"],
    static_argnums=(1,),
    preprocessing=Preprocessing.SHAPE_ONLY,
    validate_static_inputs=_validate_index_select,
)
def torch_index_select(x, dim, index, out=None):
    _assert_no_internal_overlap(out)
    if index.dim() == 0:
        index = index.reshape(1)
    elif index.dim() != 1:
        raise IndexError("index_select(): Index is supposed to be a vector")

    # Bounds checking. For 0-d input, dim_size is effectively 1 (a scalar holds
    # exactly one element, addressable only at index 0 or -1) and at most one
    # index is allowed. Matches CPU torch.index_select behavior.
    if not skip_op_preconditions():
        dim_size = x.size(dim) if x.dim() > 0 else 1
        if torch.any(index >= dim_size) or torch.any(index < -dim_size):
            raise IndexError("index out of range in self")
        if x.dim() == 0 and index.numel() > 1:
            raise RuntimeError(
                f"index_select(): Index to scalar can have only 1 value, "
                f"got {index.numel()} value(s)"
            )

    return torch.index_select, (x, dim, index), {}


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
)
def torch_index_add(x, dim, index, source, alpha=1, out=None):
    return torch.index_add(x, dim, index, source, alpha=alpha)


@register_aten(["aten::gather", "aten::gather.out"], static_argnums=(1,))
def torch_gather(x, dim, index, *, sparse_grad=False, out=None):
    return torch.gather(x, dim, index)


@register_aten(
    ["aten::scatter", "aten::scatter.value_out"],
    static_argnums=(1,),
    preprocessing=Preprocessing.SHAPE_ONLY,
)
def torch_scatter_value(x, dim, index, value, out=None):
    # Convert scalar to tensor with matching dtype
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, dtype=x.dtype, device=x.device)

    def torch_scatter_value_fn(x, dim, index, value, out=None):
        if value.dim() == 0:
            # without expanding, we hit dimension mismatch issues
            value = value.expand_as(index)
        if value.dtype != x.dtype:
            value = value.to(x.dtype)
        return torch.scatter(x, dim, index, value)

    if not skip_op_preconditions():
        size = x.shape[dim] if x.dim() > 0 else 1
        if torch.any(index >= size) or torch.any(index < -size):
            raise RuntimeError(
                f"index {torch.max(index.abs()).item()} is out of "
                f"bounds for dimension {dim} with size {size}"
            )

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
)
def torch_scatter_src(x, dim, index, src, out=None, reduce=None):
    def torch_scatter_src_fn(x, dim, index, src, out=None, reduce=None):
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

    # index OOB check
    if not skip_op_preconditions():
        size = x.shape[dim] if x.dim() > 0 else 1
        if torch.any(index >= size) or torch.any(index < -size):
            raise RuntimeError(
                f"index {torch.max(index.abs()).item()} is out of "
                f"bounds for dimension {dim} with size {size}"
            )

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
)
def torch_masked_select(x, mask, out=None):
    if mask.dtype != torch.bool:
        raise RuntimeError(
            "masked_select: expected BoolTensor for mask, but got " f"{mask.dtype} tensor"
        )
    size = mask.sum().item()

    # Correct size after broadcast outside kernel function to ensure it is traced correctly
    broadcast_shape = torch.broadcast_shapes(x.shape, mask.shape)
    size = int(size * broadcast_shape.numel() / mask.shape.numel())

    def masked_select_fn(x, mask, size, out=None):
        broadcast_shape = torch.broadcast_shapes(x.shape, mask.shape)

        if x.shape != broadcast_shape:
            x = x.broadcast_to(broadcast_shape)
        if mask.shape != broadcast_shape:
            mask = mask.broadcast_to(broadcast_shape)

        indices = torch.nonzero_static(mask, size=size, fill_value=-1)
        return x[tuple(indices[:, i] for i in range(indices.shape[1]))]

    return masked_select_fn, (x, mask, size), {"out": out}


@register_aten(["aten::masked_fill", "aten::masked_fill_.Scalar"])
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
def torch_topk(x, k, dim=-1, largest=True, sorted=True):
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


@register_aten(
    ["aten::_index_put_impl_"],
    static_argnames=("accumulate", "unsafe", "use_masked_fill"),
    preprocessing=Preprocessing.VALUE_DEPENDENT,
    translate_args_fn=_translate_index_put_args,
)
def torch_index_put_impl(x, indices, values, accumulate=False, unsafe=False, out=None):
    """Torch MLIR implementation of torch._index_put_impl_."""

    # Perform index checking unless unsafe=True
    if not unsafe:
        index_checking(x, indices)

    def can_dispatch_to_masked_fill(x, indices, values):
        """Check if index_put can be optimized to masked_fill."""
        if values.numel() != 1 or accumulate:
            return False, None

        num_defined_indices = 0
        mask = None

        for index in indices:
            if index is not None:
                if hasattr(index, "dtype") and index.dtype == torch.bool:
                    if mask is not None:  # Already found a mask
                        return False, None
                    mask = index
                    # Check shape compatibility
                    for j in range(index.ndim):
                        if index.shape[j] != x.shape[num_defined_indices + j]:
                            return False, None
                    num_defined_indices += index.ndim
                else:
                    return False, None  # Non-boolean index found
            else:
                num_defined_indices += 1

        if mask is None:
            return False, None

        # Broadcast mask to match x's shape if needed
        if mask.ndim < x.ndim:
            # Add trailing dimensions
            for _ in range(x.ndim - mask.ndim):
                mask = mask.unsqueeze(-1)

        return True, mask

    # Check if we can dispatch to masked_fill optimization
    can_dispatch, mask = can_dispatch_to_masked_fill(x, indices, values)

    def _index_put_fn(
        x, indices, values, accumulate=False, unsafe=False, use_masked_fill=False, out=None
    ):
        if use_masked_fill:
            # Cannot directly call torch.masked_fill since it only takes float object for value
            values = values.to(x.dtype)
            return torch.where(indices, values, x)

        # Normalize negative indices and ensure proper tensor format
        normalized_indices = []
        for i, idx in enumerate(indices):
            if idx is not None and isinstance(idx, torch.Tensor):
                dim_size = x.shape[i]
                idx = torch.where(idx < 0, idx + dim_size, idx)
                # Ensure scalar indices are properly shaped for MLIR
                if idx.dim() == 0:
                    idx = idx.unsqueeze(0)  # Convert scalar to [1] shape
            normalized_indices.append(idx)
        indices = tuple(normalized_indices)

        return torch.ops.aten.index_put(x, indices, values, accumulate=accumulate)

    if can_dispatch:
        processed_indices = mask if mask.device == x.device else mask.to(x.device)
    else:
        processed_indices = convert_indices(indices)

        # Check if indexing tensors can be broadcast together
        non_none_indices = [idx for idx in processed_indices if idx is not None]
        if len(non_none_indices) > 1:
            try:
                torch.broadcast_shapes(*[idx.shape for idx in non_none_indices])
            except RuntimeError:
                # Incompatible shape cause IndexError for index_put
                shapes_str = ", ".join([str(list(idx.shape)) for idx in non_none_indices])
                raise IndexError(
                    "shape mismatch: indexing tensors could not be broadcast together "
                    f"with shapes {shapes_str}"
                ) from None

    return (
        _index_put_fn,
        (x, processed_indices, values),
        {"accumulate": accumulate, "unsafe": unsafe, "use_masked_fill": can_dispatch, "out": out},
    )


def _validate_index_select_backward(grad, input_sizes, dim, index):
    """Validate index_select_backward dtypes and bounds on raw args."""
    if index.dtype not in [torch.int32, torch.int64]:
        raise TypeError(
            f"index_select_backward(): Expected dtype int32/int64 for index but got: {index.dtype}"
        )

    if skip_op_preconditions():
        return

    # For 0-d (input_sizes empty), dim_size is effectively 1 — the scalar
    # grad holds a single element addressable only at index 0. CPU rejects
    # negatives here (unlike the forward path) and rejects multiple indices.
    # Match those semantics. For 1-d+, keep the existing `0 <= index < dim_size`
    # check; negative-index support for 1-d+ backward is a pre-existing gap
    # tracked separately.
    if len(input_sizes) == 0:
        if index.numel() > 1:
            raise IndexError(
                f"index_select_backward(): Index to scalar can have only 1 value, "
                f"got {index.numel()} value(s)"
            )
        if torch.any(index != 0):
            oob_idx = index.flatten()[0].item()
            raise IndexError(f"index {oob_idx} is out of bounds for dimension {dim} with size 1")
    else:
        dim_size = input_sizes[dim]
        if torch.any(index < 0) or torch.any(index >= dim_size):
            oob_idx = index.min().item() if torch.any(index < 0) else index.max().item()
            raise IndexError(
                f"index {oob_idx} is out of bounds for dimension {dim} with size {dim_size}"
            )


@register_aten(
    ["aten::index_select_backward"],
    static_argnums=(1, 2),
    preprocessing=Preprocessing.SHAPE_ONLY,
    validate_static_inputs=_validate_index_select_backward,
)
def torch_index_select_backward(grad, input_sizes, dim, index):
    def index_select_backward_fn(grad, input_sizes, dim, index):
        # 0-d self (input_sizes empty): CPU identity-returns the scalar grad.
        if len(input_sizes) == 0:
            return grad
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
