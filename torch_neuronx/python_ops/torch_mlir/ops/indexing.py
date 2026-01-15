"""Indexing, slicing, and selection operations."""

import torch

from ..operation_registry import register_aten


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
        size = tensor.shape[dim]
        if torch.any(idx >= size) or torch.any(idx < -size):
            raise IndexError(
                f"index {torch.max(idx).item()} is out of bounds "
                f"for dimension {dim} with size {size}"
            )


@register_aten(["aten::index_select", "aten::index_select.out"], static_argnums=(1,))
def torch_index_select(x, dim, index):
    if index.dtype not in [torch.int32, torch.int64]:
        # Actually pytorch raise RuntimeError, but RuntimeError will cause
        # neuron op fallback to CPU. So raise TypeError instead
        raise TypeError("index_select(): Expected dtype int32 or int64 for index")

    if index.dim() == 0:
        index = index.reshape(1)
    elif index.dim() != 1:
        raise IndexError("index_select(): Index is supposed to be a vector")

    return torch.index_select(x, dim, index)


@register_aten(["aten::index_copy", "aten::index_copy_"], static_argnums=(1,))
def torch_index_copy(x, dim, indexes, source):
    return torch.index_copy(x, dim, indexes, source)


@register_aten(["aten::index_add", "aten::index_add.out"], static_argnums=(1,))
def torch_index_add(x, dim, index, source, alpha=1):
    return torch.index_add(x, dim, index, source, alpha=alpha)


@register_aten(["aten::gather", "aten::gather.out"])
def torch_gather(x, dim, index):
    return torch.gather(x, dim, index)


@register_aten(["aten::scatter", "aten::scatter.value_out"])
def torch_scatter_value(x, dim, index, value):
    return torch.scatter(x, dim, index, value)


@register_aten(
    ["aten::scatter.src", "aten::scatter.src_out", "aten::scatter.reduce"],
    static_argnums=(1,),
    static_argnames=("reduce",),
)
def torch_scatter_src(input, dim, index, src, reduce=None):
    if reduce is None:
        return torch.scatter(input, dim, index, src)
    return torch.scatter(input, dim, index, src, reduce=reduce)


@register_aten(
    ["aten::masked_select", "aten::masked_select.out", "aten::masked_select.Tensor_out"],
    static_argnums=(2,),
    uses_preprocessing=True,
)
def torch_masked_select(x, mask, out=None):
    size = mask.sum().item()

    # Correct size after broadcast outside kernel functioin to ensure it is traced correctly
    broadcast_shape = torch.broadcast_shapes(x.shape, mask.shape)
    size = int(size * broadcast_shape.numel() / mask.shape.numel())

    def masked_select_fn(x, mask, size, out=None):
        broadcast_shape = torch.broadcast_shapes(x.shape, mask.shape)

        if x.shape != broadcast_shape:
            x = x.broadcast_to(broadcast_shape)
        if mask.shape != broadcast_shape:
            mask = mask.broadcast_to(broadcast_shape)

        x_flat = x.flatten()
        mask_flat = mask.flatten()

        true_indices = torch.nonzero_static(mask_flat, size=size, fill_value=-1).squeeze(-1)

        return x_flat[true_indices]

    return masked_select_fn, (x, mask, size), {"out": out}


@register_aten(["aten::masked_fill", "aten::masked_fill_.Scalar"])
def torch_masked_fill(x, mask, value):
    return torch.masked_fill(x, mask, value)


@register_aten(
    [
        "aten::where",
        "aten::where.self",
        "aten::where.self_out",
        "aten::where.ScalarSelf",
        "aten::where.ScalarOther",
        "aten::where.Scalar",
    ],
    use_torch_compile=True,
)
def torch_where(condition, x=None, y=None, out=None):
    if x is None:
        return torch.where(condition)
    return torch.where(condition, x, y)


@register_aten(
    ["aten::nonzero_static", "aten::nonzero_static.out"],
    static_argnames=("size", "fill_value", "dtype"),
)
def torch_nonzero_static(tensor, size=None, fill_value=-1, out=None, with_count=False, dtype=None):
    if dtype is not None and dtype != torch.int64:
        raise RuntimeError(
            "nonzero_static: Expected out tensor to have scalar type Long "
            "but got scalar typeFloat"
        )
    return torch.nonzero_static(tensor, size=size, fill_value=fill_value)


@register_aten(
    ["aten::topk", "aten::topk.values"],
    static_argnums=(1, 2, 3, 4),
)
def torch_topk(x, k, dim=-1, largest=True, sorted=True):
    return torch.topk(x, k, dim, largest=largest, sorted=sorted)


@register_aten(
    ["aten::_index_put_impl_"],
    static_argnames=("accumulate", "unsafe", "use_masked_fill"),
    uses_preprocessing=True,
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

        return torch.ops.aten.index_put(x, indices, values, accumulate=accumulate)

    if can_dispatch:
        processed_indices = mask
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


@register_aten(["aten::index_select_backward"], static_argnums=(1, 2))
def torch_index_select_backward(grad, input_sizes, dim, index):
    if index.dtype not in [torch.int32, torch.int64]:
        raise TypeError(
            f"index_select_backward(): Expected dtype int32/int64 for index but got: {index.dtype}"
        )

    if index.dim() == 0:
        index = index.reshape(1)
    return torch.ops.aten.index_select_backward(grad, input_sizes, dim, index)
