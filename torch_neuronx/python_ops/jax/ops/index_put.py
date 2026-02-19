"""JAX implementation of index_put operation."""

import jax.numpy as jnp
import torch

from torch_neuronx.python_ops.jax.operation_registry import register_aten
from torch_neuronx.python_ops.legacy.xla_ops.index_xla import convert_indices, index_checking


@register_aten(
    ["aten::_index_put_impl_"],
    static_argnames=("accumulate", "unsafe", "use_masked_fill"),
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

    def can_dispatch_to_masked_fill(self, indices, values):
        """Check if index_put can be optimized to masked_fill.

        Returns:
            (False, None) if cannot dispatch
            (True, mask) if can dispatch with the processed mask
        """
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
                        if index.shape[j] != self.shape[num_defined_indices + j]:
                            return False, None
                    num_defined_indices += index.ndim
                else:
                    return False, None  # Non-boolean index found
            else:
                num_defined_indices += 1

        if mask is None:
            return False, None

        # Broadcast mask to match self's shape if needed
        if mask.ndim < self.ndim:
            # Add trailing dimensions
            for _ in range(self.ndim - mask.ndim):
                mask = mask.unsqueeze(-1)

        return True, mask

    # Check if we can dispatch to masked_fill optimization
    can_dispatch, mask = can_dispatch_to_masked_fill(self, indices, values)
    indices = mask if can_dispatch else convert_indices(indices)

    def _index_put_fn(self, indices, values, accumulate=False, unsafe=False, use_masked_fill=False):
        if use_masked_fill:
            values = values.astype(self.dtype)
            return jnp.where(indices, values, self)

        indices = [slice(None, None, None) if i is None else i for i in indices]
        indices = tuple(indices)

        if accumulate:
            return self.at[indices].add(values)
        else:
            return self.at[indices].set(values)

    return (
        _index_put_fn,
        (self, indices, values),
        {"accumulate": accumulate, "unsafe": unsafe, "use_masked_fill": can_dispatch},
    )
