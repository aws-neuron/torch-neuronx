"""JAX implementation for torch.repeat_interleave."""

import operator

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax._src import core
from jax._src.numpy import reductions, util
from jax._src.numpy.array import array, asarray
from jax._src.typing import Array, ArrayLike, DimSize
from jax._src.util import canonicalize_axis as _canonicalize_axis


def _custom_repeat(
    a: ArrayLike,
    *,
    repeats: ArrayLike,
    axis: int | None = None,
    total_repeat_length: int | None = None,
) -> Array:
    """Patch _repeat function to avoid Neuron compiler issues.

    Reference: https://github.com/jax-ml/jax/blob/7717ae417b87a5fad292ba914b4f116ec8e30bc2/jax/_src/numpy/lax_numpy.py#L6370
    """
    if core.is_dim(repeats):
        util.check_arraylike("repeat", a)
    else:
        util.check_arraylike("repeat", a, repeats)
    arr = asarray(a)

    if axis is None:
        arr = arr.ravel()
        axis = 0

    axis = core.concrete_or_error(operator.index, axis, "'axis' argument of jnp.repeat()")
    assert isinstance(axis, int)  # to appease mypy

    if core.is_symbolic_dim(repeats) and total_repeat_length is not None:
        raise ValueError(
            "jnp.repeat with a non-constant `repeats` is supported only "
            f"when `total_repeat_length` is None. ({repeats=} {total_repeat_length=})"
        )

    # If total_repeat_length is not given, use a default.
    if total_repeat_length is None:
        repeats = core.concrete_or_error(
            None,
            repeats,
            "When jit-compiling jnp.repeat, the total number of repeats must be static. "
            "To fix this, either specify a static value for `repeats`, or pass a static "
            "value to `total_repeat_length`.",
        )

        # Fast path for when repeats is a scalar.
        if np.ndim(repeats) == 0 and np.ndim(arr) != 0:
            input_shape = arr.shape
            axis = _canonicalize_axis(axis, len(input_shape))
            aux_axis = axis + 1
            aux_shape: list[DimSize] = list(input_shape)
            aux_shape.insert(
                aux_axis, operator.index(repeats) if core.is_constant_dim(repeats) else repeats
            )  # type: ignore
            arr = jax.lax.broadcast_in_dim(
                arr, aux_shape, [i for i in range(len(aux_shape)) if i != aux_axis]
            )
            result_shape: list[DimSize] = list(input_shape)
            result_shape[axis] *= repeats
            return arr.reshape(result_shape)

        repeats = np.ravel(repeats)
        if arr.ndim != 0:
            repeats = np.broadcast_to(repeats, [arr.shape[axis]])
        total_repeat_length = np.sum(repeats)
    else:
        repeats = jnp.ravel(repeats)
        if arr.ndim != 0:
            repeats = jnp.broadcast_to(repeats, [arr.shape[axis]])

    # Special case when a is a scalar.
    if arr.ndim == 0:
        if np.shape(repeats) == (1,):
            return jnp.full([total_repeat_length], arr)
        else:
            raise ValueError(
                "`repeat` with a scalar parameter `a` is only "
                "implemented for scalar values of the parameter `repeats`."
            )

    # Special case if total_repeat_length is zero.
    if total_repeat_length == 0:
        result_shape = list(arr.shape)
        result_shape[axis] = 0
        return jnp.reshape(array([], dtype=arr.dtype), result_shape)

    # If repeats is on a zero sized axis, then return the array.
    if arr.shape[axis] == 0:
        return arr

    # This implementation of repeat avoid having to instantiate a large.
    # intermediate tensor.

    # NOTE Doing roll -> cumsum causes the following compiler error
    # Internal tensorizer error: CommuteConcat:size mismatch!
    # Therefore change to cumsum -> roll to achieve the same result but avoid compiler error
    # Modify repeats from e.g. [1,2,0,5] -> [0,1,2,0] for exclusive repeat.
    # exclusive_repeats = roll(repeats, shift=1).at[0].set(0)
    # Cumsum to get indices of new number in repeated tensor, e.g. [0, 1, 3, 3]
    # scatter_indices = reductions.cumsum(exclusive_repeats)
    scatter_indices = reductions.cumsum(repeats)
    scatter_indices = jnp.roll(scatter_indices, shift=1).at[0].set(0)
    # Scatter these onto a zero buffer, e.g. [1,1,0,2,0,0,0,0]
    block_split_indicators = jnp.zeros([total_repeat_length], dtype="float32")
    # NOTE Neuron requires scatter reduction function add operand to have dtype float
    block_split_indicators = block_split_indicators.at[scatter_indices].add(1.0)
    # NOTE Cumsum results need to be casted back to int for indexing
    # Cumsum again to get scatter indices for repeat, e.g. [0,1,1,3,3,3,3,3]
    gather_indices = reductions.cumsum(block_split_indicators).astype("int32") - 1
    return jnp.take(arr, gather_indices, axis=axis)


jax._src.numpy.lax_numpy._repeat = _custom_repeat


# @register_aten(
#     [
#         "aten::repeat_interleave.Tensor",
#         "aten::repeat_interleave.self_Tensor",
#         "aten::repeat_interleave.self_int",
#     ],
#     static_argnums=(2,),
#     static_argnames=("output_size",),
#     uses_preprocessing=True,
# )
def _aten_repeat_interleave(self, repeats, dim=None, output_size=None):
    """Jax implementation for torch.repeat_interleave operation.

    The behavior matches CPU and CUDA behavior:
        1. calculate output_size
        2. use output_size to make static output and perform actual repeat operation
    """
    if dim is None:
        dim = 0
        self = self.flatten()

    if isinstance(repeats, int):
        calculated_output_size = repeats * self.shape[dim]
    else:
        calculated_output_size = torch.sum(repeats).item()

    if output_size is None:
        output_size = calculated_output_size

    if output_size != calculated_output_size:
        raise RuntimeError("allocated size does not match required size")

    def repeat_interleave_fn(self, repeats, dim, output_size=None):
        """Do NOT rename `output_size` as it is required for meta tensor evaluation."""
        return jnp.repeat(self, repeats, axis=dim, total_repeat_length=output_size)

    return repeat_interleave_fn, (self, repeats, dim), {"output_size": output_size}
