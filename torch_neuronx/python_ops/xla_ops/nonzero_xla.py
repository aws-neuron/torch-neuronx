"""Jax implementation of torch.nonzero."""

import logging

import jax.numpy as jnp
import numpy as np
import torch
from jax._src import core
from jax._src.numpy import reductions, util
from jax._src.typing import Array, ArrayLike
from jax._src.util import safe_zip

from torch_neuronx.kernels.xla_kernel import TorchNeuronXLAKernel
from torch_neuronx.python_ops.base import ExecutionResult, OperationImplementation

logger = logging.getLogger(__name__)


def safe_nonzero(
    a: ArrayLike,
    *,
    size: int | None = None,
    fill_value: None | ArrayLike | tuple[ArrayLike, ...] = None,
) -> tuple[Array, ...]:
    """Reference: https://github.com/jax-ml/jax/blob/5ce49bd22b41d99b9ee741ddc6b162f4d0927819/jax/_src/numpy/lax_numpy.py#L3685"""
    arr = util.ensure_arraylike("nonzero", a)
    del a
    if np.ndim(arr) == 0:
        raise ValueError(
            "Calling nonzero on 0d arrays is not allowed. "
            "Use jnp.atleast_1d(scalar).nonzero() instead."
        )
    mask = arr if arr.dtype == bool else (arr != 0)
    calculated_size_ = mask.sum() if size is None else size
    calculated_size: int = core.concrete_dim_or_error(
        calculated_size_,
        "The size argument of safe_nonzero must be statically specified "
        "to use safe_nonzero within JAX transformations.",
    )
    if arr.size == 0 or calculated_size == 0:
        return tuple(jnp.zeros(calculated_size, int) for dim in arr.shape), mask.sum()

    """
    The following block are customized for Neuron
    1. avoid integer or bool operands in scatter for reduction function add
    2. avoid OOB error by processing indices
    """
    cumsum_counts = reductions.cumsum(mask)
    # Ignore OOB values
    in_range = cumsum_counts < size
    cumsum_counts = jnp.clip(cumsum_counts, 0, size - 1)
    weights = jnp.where(in_range, 1.0, 0.0)
    bin_counts = jnp.bincount(cumsum_counts, length=calculated_size, weights=weights)
    flat_indices = reductions.cumsum(bin_counts)
    flat_indices = flat_indices.astype("int32")

    strides: np.ndarray = (np.cumprod(arr.shape[::-1])[::-1] // arr.shape).astype(
        flat_indices.dtype
    )
    out = tuple(
        (flat_indices // stride) % size for stride, size in zip(strides, arr.shape, strict=False)
    )
    if fill_value is not None:
        fill_value_tup = fill_value if isinstance(fill_value, tuple) else arr.ndim * (fill_value,)
        if any(np.shape(val) != () for val in fill_value_tup):
            raise ValueError(
                f"fill_value must be a scalar or a tuple of length {arr.ndim}; got {fill_value}"
            )
        fill_mask = jnp.arange(calculated_size) >= mask.sum()
        out = tuple(
            jnp.where(fill_mask, fval, entry) for fval, entry in safe_zip(fill_value_tup, out)
        )

    return out, mask.sum()


class NonzeroXLAImpl(OperationImplementation):
    """nonzero implementation using XLA"""

    def __init__(self):
        """Initialize the nonzero kernel with JAX computation"""
        super().__init__()

        def nonzero_fn(tensor):
            """JAX computation for nonzero"""
            return safe_nonzero(tensor, size=tensor.size)

        self.kernel = TorchNeuronXLAKernel(nonzero_fn, "nonzero")

    def can_handle(self, tensor, out=None, as_tuple=False, out_dtype=None):
        return tensor.device.type == "neuron"

    def _execute_impl(
        self, tensor: torch.Tensor, *, out=None, as_tuple=False, out_dtype=None
    ) -> ExecutionResult:
        """Execute nonzero kernel and return result indices.

        Args:
            out: Output tensor
            as_tuple: Return output as a tuple or not. Even though torch.nonzero takes `as_tuple`
                it is handled in PyTorch API, not in the kernel. And PyTorch API will pass in False.
                Supporting here so that we can skip the stack if used during decomposition
            out_type: Output dtype to avoid bouncing between int64 and int32 during decomposition
        """
        # TODO how to handle out tensor with smaller size
        # pytorch will automatically resize the out tensor

        if out_dtype is None:
            out_dtype = torch.int64

        if out is not None and out.dtype != out_dtype:
            # PyTorch requires output dtype to be torch.int64
            raise RuntimeError(
                "nonzero: Expected out tensor to have scalar type Long " "but got scalar typeFloat"
            )

        try:
            result = self.kernel(tensor)
            indices, size = result[:-1], result[-1]
            trimmed_indices = tuple([index[:size].to(out_dtype) for index in indices])

            if as_tuple:
                return ExecutionResult(success=True, output=trimmed_indices)

            # Stack indices to create a single tensor of shape (num_nonzero, ndim)
            if len(trimmed_indices) == 0 or size == 0:
                output = torch.empty((0, tensor.ndim), dtype=out_dtype, device=tensor.device)
            else:
                output = torch.stack(trimmed_indices, dim=1)

            if out is not None:
                out.copy_(output)

            return ExecutionResult(success=True, output=output)

        except Exception as e:
            logger.error(f"Failed to execute nonzero: {e}")
            return ExecutionResult(success=False, error_msg=str(e))


class NonzeroStaticXLAImpl(OperationImplementation):
    """nonzero_static implementation using XLA"""

    def __init__(self):
        """Initialize the nonzero_static kernel with JAX computation"""
        super().__init__()

        def nonzero_static_fn(tensor, size, fill_value):
            """JAX computation for nonzero_static"""
            return safe_nonzero(tensor, size=size, fill_value=fill_value)

        self.kernel = TorchNeuronXLAKernel(nonzero_static_fn, "nonzero_static", static_argnums=(1,))

    def can_handle(self, tensor, size, fill_value=-1, out=None):
        return tensor.device.type == "neuron"

    def _execute_impl(
        self, tensor: torch.Tensor, *, size, fill_value=-1, out=None
    ) -> ExecutionResult:
        """Execute nonzero_static kernel and return result indices."""
        if out is not None and out.dtype != torch.int64:
            raise RuntimeError(
                "nonzero_static: Expected out tensor to have scalar type Long "
                "but got scalar typeFloat"
            )

        try:
            if size == 0:
                # Skip execution if output will be empty tensor
                output = torch.empty((0, tensor.ndim), dtype=torch.int64, device=tensor.device)
            else:
                result = self.kernel(tensor, size, fill_value)
                indices = result[:-1]
                indices = tuple([index[:size].to(torch.int64) for index in indices])
                output = torch.stack(indices, dim=1)

            if out is not None:
                out.copy_(output)

            return ExecutionResult(success=True, output=output)

        except Exception as e:
            logger.error(f"Failed to execute nonzero_static: {e}")
            return ExecutionResult(success=False, error_msg=str(e))
