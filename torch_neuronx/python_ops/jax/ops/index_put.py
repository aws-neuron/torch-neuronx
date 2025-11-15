"""JAX implementation of index_put operation."""

from torch_neuronx.python_ops.jax.operation_registry import register_aten
from torch_neuronx.python_ops.xla_ops.index_xla import convert_indices, index_checking


@register_aten(
    ["aten::_index_put_impl_"],
    static_argnames=("accumulate", "unsafe"),
    uses_preprocessing=True,
)
def _aten_index_put_impl_(self, indices, values, accumulate=False, unsafe=False, out=None):
    """Jax implementation of torch._index_put_impl_.

    Args:
        self: Input tensor
        indices: Tuple of index tensors
        values: Values to be add/set to self
        accumulate: Execute .at[].add when set to True, else .at[].set
        unsafe: Check index boundary or not

    To handle boolean indices, need to convert them into integer indcies using torch.nonzero.
    """
    if self.numel() == 0:
        return self

    index_checking(self, indices)

    # Convert all indices into integer indices
    integer_indices = convert_indices(indices)

    def _index_put_fn(self, indices, values, accumulate=False, unsafe=False):
        indices = [slice(None, None, None) if i is None else i for i in indices]
        indices = tuple(indices)

        if accumulate:
            return self.at[indices].add(values)
        else:
            return self.at[indices].set(values)

    return (
        _index_put_fn,
        (self, integer_indices, values),
        {"accumulate": accumulate, "unsafe": unsafe},
    )
