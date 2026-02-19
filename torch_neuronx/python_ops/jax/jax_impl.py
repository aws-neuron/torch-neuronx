# SPDX-License-Identifier: BSD-3-Clause
# This file includes material from the PyTorch XLA project (pytorch-tpu),
# licensed under the BSD 3-Clause License.
# Source: https://github.com/pytorch/xla/blob/master/torchax/torchax/ops/jaten.py
#
# Original copyright:
#   Copyright (c) 2023, pytorch-tpu
#   All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Modifications by Amazon Web Services, Inc:
#   Copyright (c) 2025, Amazon Web Services, Inc.

"""JAX implementations for all ATen operations.

This file contains the JAX implementations that will be used by the dynamic classes.
All operations should be defined here as _aten_* functions.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import torch

from torch_neuronx.python_ops.legacy import op_base

from . import reimplement as jax_reimplement
from .operation_registry import register_aten
from .type_converter import convert_dtype_with_default


# Error message formatters
def MM_SHAPE_MISMATCH(x, y):
    """Format error for matrix multiplication shape mismatch."""
    return f"Expected size for first two dimensions of batch2 tensor to be: [{x.shape[0]}, {x.shape[2]}] but got: [{y.shape[0]}, {y.shape[1]}]."


def BMM_BATCH_MISMATCH(x, y):
    """Format error for batch dimension mismatch."""
    return f"Expected batch dimensions to match, got {x.shape[0]} and {y.shape[0]}"


def MM_DIMENSION_ERROR(tensor_name):
    """Format error for non-2D tensor in mm."""
    return f"{tensor_name} must be a matrix"


def MM_CANNOT_MULTIPLY(x, y):
    """Format error for incompatible matrix dimensions."""
    return f"shapes cannot be multiplied ({x.shape[1]} != {y.shape[0]})"


def BMM_DIMENSION_ERROR(tensor_name):
    """Format error for non-3D tensor in bmm."""
    return f"{tensor_name} must be a 3D tensor"


def ADDMV_DIMENSION_ERROR(self, mat, vec):
    """Format error for incorrect input ranks in addmv."""
    return f"vector + matrix @ vector expected, got {self.ndim}, {mat.ndim}, {vec.ndim}"


def ADDMV_SHAPE_MISMATCH(self, mat, vec):
    """Format error for incompatible shapes in addmv."""
    self_size = self.shape[0] if self.ndim > 0 else self.size
    return f"size mismatch, got input ({self_size}), mat ({mat.shape[0]}x{mat.shape[1]}), vec ({vec.shape[0]})"


def ADDMM_BROADCAST_ERROR(self, result_shape, dim):
    """Format error for non-broadcastable dimensions in addmm."""
    dim_idx = 0 if dim == "rows" else 1
    return f"The size of tensor a ({self.shape[dim_idx]}) must match the size of tensor b ({result_shape[dim_idx]}) at non-singleton dimension {dim_idx}"


def check_constraint(condition, error_formatter, args):
    """Check a constraint and raise an error if it fails.

    Args:
        condition: Boolean condition to check
        error_formatter: Function that formats the error message
        args: Arguments to pass to the error formatter
    """
    if not condition:
        raise RuntimeError(error_formatter(*args))


# These imports may need to be adjusted based on actual dependencies
# import mappings  # Uncomment and fix import path if needed
# from View import View  # Uncomment and fix import path if needed


# For now, create a simple mappings stub if needed
class mappings:
    @staticmethod
    def t2j_dtype(torch_dtype):
        """Convert torch dtype to JAX dtype."""
        if torch_dtype == torch.float32:
            return jnp.float32
        elif torch_dtype == torch.float64:
            return jnp.float32  # JAX typically uses float32
        elif torch_dtype == torch.float16:
            return jnp.float16
        elif torch_dtype == torch.bfloat16:
            return jnp.bfloat16
        else:
            return jnp.float32


# Stub for View if needed
class View:
    def update(self, value):
        pass


def _aten_add(x, y, *, alpha=1):
    """if isinstance(x, jnp.ndarray) and isinstance(y, jnp.ndarray):

    assert x.dtype == y.dtype, (x.dtype, y.dtype)
    """
    res = x + y * alpha
    if isinstance(x, float) or isinstance(y, float):
        new_dtype = mappings.t2j_dtype(torch.get_default_dtype())
        res = res.astype(new_dtype)
    return res


def _aten_clone(x, memory_format=None):
    return x


# aten.trunc
@register_aten(
    [
        "aten::trunc",
        "aten::trunc.out",
    ],
    operation_type="arithmetic",
)
def _aten_trunc(x):
    res = jnp.trunc(x)
    return res.astype(x.dtype)


@register_aten(
    [
        "aten::index_copy",
        "aten::index_copy_",
    ],
    operation_type="indexing",
    static_argnums=(1,),
    uses_preprocessing=True,
)
def _aten_index_copy(x, dim, indexes, source):
    if x.ndim != 0 and x.ndim != source.ndim:
        raise IndexError(
            f"index_copy_(): When source and destination are not scalars, their dimensionality must match. Source dimensionality ({source.ndim}), destination dimensionality ({x.ndim})"
        )

    if dim >= x.ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [-{x.ndim}, {x.ndim - 1}], but got {dim})"
        )

    def _index_copy_fn(x, dim, indexes, source):
        if x.ndim == 0:
            return source
        if x.ndim == 1:
            source = jnp.squeeze(source)
        if dim < 0:
            dim = dim + x.ndim
        dims = []
        for i in range(len(x.shape)):
            if i == dim:
                dims.append(indexes)
            else:
                dims.append(slice(None, None, None))
        return x.at[tuple(dims)].set(source)

    return _index_copy_fn, (x, dim, indexes, source), {}


@register_aten(
    ["aten::atleast_2d"],
)
def _aten_atleast_2d(inputs):
    return jnp.atleast_2d(inputs)


def _aten_atleast_1d(inputs):
    return jnp.atleast_1d(inputs)


# aten.complex
def _aten_complex(real, imag):
    """
    Constructs a complex array from real and imaginary parts.

    Args:
      real: An array of real values.
      imag: An array of imaginary values.

    Returns:
      A complex array with the specified real and imaginary parts.
    """
    return jnp.array(real, dtype=jnp.float32) + 1j * jnp.array(imag, dtype=jnp.float32)


# aten.linalg_householder_product
def _aten_linalg_householder_product(input, tau):
    return jax.lax.linalg.householder_product(a=input, taus=tau)


@register_aten(
    [
        "aten::index_select",
        "aten::index_select.out",
    ],
    static_argnums=(1,),
    uses_preprocessing=True,
)
def _aten_index_select(x, dim, index):
    # Accept 0-D scalar index (normalize to length-1) and enforce 1-D otherwise
    if hasattr(index, "dim"):
        if index.dim() == 0:
            try:
                index = index.reshape(1)
            except Exception:
                pass
        elif index.dim() != 1:
            raise IndexError(f"index_select(): Expected 1-D index tensor, got dim={index.dim()}")

    def _index_select_fn(x, dim, index):
        if x.shape == ():
            return x
        return jnp.take(x, index, dim)

    return _index_select_fn, (x, dim, index), {}


def _aten_cholesky(input, upper=False):
    return jax.scipy.linalg.cholesky(input, lower=(not upper))


def _aten_linalg_cholesky_ex(input, upper=False, check_errors=False):
    if check_errors:
        raise NotImplementedError(
            "check_errors=True is not supported in this JAX implementation. "
            "Check for positive definiteness using jnp.linalg.eigvalsh before "
            "calling this function."
        )

    L = jax.scipy.linalg.cholesky(input, lower=not upper)
    if len(L.shape) > 2:
        info = jnp.zeros(shape=L.shape[:-2], dtype=jnp.int32)
    else:
        info = jnp.array(0, dtype=jnp.int32)
    return L, info


def _aten_cholesky_solve(input, input2, upper=False):
    # Ensure input2 is lower triangular for cho_solve
    L = input2 if not upper else input2.T
    # Use cho_solve to solve the linear system
    solution = jax.scipy.linalg.cho_solve((L, True), input)
    return solution


def _aten_special_zeta(x, q):
    new_dtype = mappings.t2j_dtype(torch.get_default_dtype())
    res = jax.scipy.special.zeta(x, q)
    if isinstance(x, int) or isinstance(q, int):
        res = res.astype(new_dtype)
    return res  # jax.scipy.special.zeta(x, q)


# aten.igammac
def _aten_igammac(input, other):
    if isinstance(input, jnp.ndarray):
        input = jnp.where(input < 0, jnp.nan, input)
    if isinstance(other, jnp.ndarray):
        other = jnp.where(other < 0, jnp.nan, other)
    else:
        if (input == 0 and other == 0) or (input < 0) or (other < 0):
            other = jnp.nan
    return jnp.array(jax.scipy.special.gammaincc(input, other))


def _aten_mean(x, dim=None, keepdim=False, dtype=None):
    if x.shape == () and dim is not None:
        dim = None  # disable dim for jax array without dim
    result = jnp.mean(x, dim, keepdims=keepdim)
    if dtype is not None:
        result = result.astype(dtype)
    return result


def _torch_binary_scalar_type(scalar, tensor):
    if "float" in str(tensor.dtype) or "complex" in str(tensor.dtype):
        return tensor.dtype

    if isinstance(scalar, int):
        if "int" in str(tensor.dtype):
            return tensor.dtype

    return jnp.float32


def _aten_searchsorted(sorted_sequence, values):
    new_dtype = mappings.t2j_dtype(torch.get_default_dtype())
    res = jnp.searchsorted(sorted_sequence, values)
    if sorted_sequence.dtype == np.dtype(np.int32) or sorted_sequence.dtype == np.dtype(np.int32):
        # res = res.astype(new_dtype)
        res = res.astype(np.dtype(np.int64))
    return res  # jnp.searchsorted(sorted_sequence, values)


def _aten_sub(x, y, alpha=1):
    if isinstance(x, float):
        dtype = _torch_binary_scalar_type(x, y)
        x = jnp.array(x, dtype=dtype)
    if isinstance(y, float):
        dtype = _torch_binary_scalar_type(y, x)
        y = jnp.array(y, dtype=dtype)
    return x - y * alpha


def _aten_numpy_T(input):
    """
    Jax implementation of torch.numpy_T.

    Args:
      input: JAX array.

    Returns:
      Transposed JAX array.
    """
    return jnp.transpose(input)


@register_aten(["aten::mm", "aten::mm.out"])
def _aten_mm(mat1, mat2, *, out=None):
    check_constraint(mat1.ndim == 2, MM_DIMENSION_ERROR, ("mat1",))
    check_constraint(mat2.ndim == 2, MM_DIMENSION_ERROR, ("mat2",))
    check_constraint(mat1.shape[1] == mat2.shape[0], MM_CANNOT_MULTIPLY, (mat1, mat2))
    return _aten_matmul(mat1, mat2)


def _aten_mul(x, y):
    return x * y


def _aten_silu(x):
    return jax.nn.silu(x)


@register_aten(["aten::triu", "aten::triu.out"], static_argnums=(1,))
def _aten_triu(m, diagonal=0):
    return jnp.triu(m, k=diagonal)


@register_aten(
    "aten::isfinite",
)
def _aten_isfinite(x):
    return jnp.isfinite(x)


# @register_aten(["aten::vstack", "aten::vstack.out"])
def _aten_vstack(tensors):
    return jnp.vstack(tensors)


@register_aten(["aten::stack", "aten::stack.out"], static_argnums=(1,))
def _aten_stack(tensors, dim=0):
    return jnp.stack(tensors, dim)


def _aten_softmax(x, dim, dtype=None, halftofloat=False):
    # Convert input to specified dtype if provided
    input_data = x
    if dtype is not None:
        # Handle torch.dtype when passed as static argument
        import torch

        if isinstance(dtype, torch.dtype):
            from torch_neuronx.kernels.type_converter import TypeConverter

            jax_dtype = TypeConverter.torch_to_jax(dtype)
            input_data = x.astype(jax_dtype)
        else:
            input_data = x.astype(dtype)

    if input_data.shape == ():
        result = jax.nn.softmax(input_data.reshape([1]), axis=0).reshape([])
    else:
        result = jax.nn.softmax(input_data, dim)
    return result


def _is_int(x):
    if isinstance(x, int):
        return True
    if isinstance(x, jax.Array) and (
        x.dtype.name.startswith("int") or x.dtype.name.startswith("uint")
    ):
        return True
    return False


def highest_precision_int_dtype(tensor1, tensor2):
    if isinstance(tensor1, int):
        return tensor2.dtype
    if isinstance(tensor2, int):
        return tensor1.dtype

    dtype_hierarchy = {
        "uint8": 8,
        "int8": 8,
        "uint16": 16,
        "int16": 16,
        "uint32": 32,
        "int32": 32,
        "uint64": 64,
        "int64": 64,
    }
    return max(tensor1.dtype, tensor2.dtype, key=lambda dtype: dtype_hierarchy[str(dtype)])


def _aten_pow(x, y):
    y_orig = y
    if isinstance(y, int):
        y = float(y)
    if _is_int(x) and _is_int(y_orig):
        # Do the math in float then cast
        res = jnp.power(jnp.astype(x, jnp.dtype("float")), y)
        return res.astype(highest_precision_int_dtype(x, y_orig))
    res = jnp.power(x, y)
    if isinstance(x, float):
        return res.astype(_torch_binary_scalar_type(x, y_orig))
    if isinstance(y_orig, float):
        return res.astype(_torch_binary_scalar_type(y_orig, x))
    return res


@register_aten(
    [
        "aten::div",
        "aten::div.out",
        "aten::div.out_mode",
        "aten::div_",
    ],
    operation_type="arithmetic",
    static_argnames=("rounding_mode",),
)
def _aten_div(x, y, rounding_mode=None):
    res_dtype = None
    if _is_int(x) and _is_int(y):
        if rounding_mode is None:
            res_dtype = jnp.float32
        else:
            res_dtype = jnp.int32

    if rounding_mode == "floor":
        res = x // y
    else:
        res = x / y

    if rounding_mode == "trunc":
        res = jnp.trunc(res)

    res = jnp.where(y == 0, jnp.sign(x) * jnp.inf, res)

    if res_dtype:
        res = res.astype(res_dtype)
    return res


def _aten_true_divide(x, y):
    return x / y


def _aten_dist(input, other, p=2):
    diff = jnp.abs(jnp.subtract(input, other))
    return _aten_linalg_vector_norm(diff, ord=p)


@register_aten(["aten::bmm", "aten::bmm.out"])
def _aten_bmm(batch1, batch2, *, out=None):
    check_constraint(batch1.ndim == 3, BMM_DIMENSION_ERROR, ("batch1",))
    check_constraint(batch2.ndim == 3, BMM_DIMENSION_ERROR, ("batch2",))
    check_constraint(batch1.shape[0] == batch2.shape[0], BMM_BATCH_MISMATCH, (batch1, batch2))
    check_constraint(batch1.shape[2] == batch2.shape[1], MM_SHAPE_MISMATCH, (batch1, batch2))
    return _aten_matmul(batch1, batch2)


@register_aten(
    ["aten::embedding"],
    static_argnums=(2, 3, 4),
    uses_preprocessing=True,
)
def _aten_embedding(weights, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    def jax_embedding(weights, indices, padding_idx, scale_grad_by_freq, sparse):
        embeddings = jnp.take(weights, indices, axis=0)

        if padding_idx is not None and padding_idx != -1:
            vocab_size = weights.shape[0]
            actual_padding_idx = padding_idx if padding_idx >= 0 else vocab_size + padding_idx

            # Create mask
            is_padding = jnp.equal(indices, actual_padding_idx)
            padding_mask = jnp.expand_dims(is_padding, axis=-1)

            # Zero out padding positions
            embeddings = jnp.where(padding_mask, 0.0, embeddings)

        return embeddings

    # Preprocessing validation
    if sparse:
        raise NotImplementedError("Sparse is currently not supported for embedding.")

    if weights.ndim != 2:
        raise RuntimeError(
            f"Embedding weight must be 2D [vocab_size, embed_dim], got {weights.ndim}D with shape {weights.shape}"
        )

    if 0 not in indices.shape:
        vocab_size = weights.shape[0]
        if (indices < 0).any() or (indices >= vocab_size).any():
            raise IndexError(
                f"Embedding indices out of bounds of {vocab_size=}, "
                f"{weights.shape=}, {indices.shape=}, {indices.min()=}, {indices.max()=}"
            )

        if padding_idx != -1:
            if padding_idx >= vocab_size or padding_idx < -vocab_size:
                raise IndexError(
                    f"padding_idx {padding_idx} out of bounds for vocab_size={vocab_size}. "
                    f"Must be -1 (no padding) or in range [-{vocab_size}, {vocab_size-1}]"
                )

    # Return (actual_jax_fn, processed_args). Keep static args to honor static_argnums.
    return jax_embedding, (weights, indices, padding_idx, scale_grad_by_freq, sparse), {}


def _aten_embedding_renorm_(weight, indices, max_norm, norm_type):
    # Adapted from https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Embedding.cpp
    unique_indices = jnp.unique(indices)

    norm = jnp.linalg.norm(
        _aten_embedding(weight, unique_indices),
        ord=norm_type,
        axis=1,
    )

    indice_idx = jnp.where(norm > max_norm)

    scale = max_norm / (norm[indice_idx] + 1e-7)

    indices_to_update = unique_indices[indice_idx]

    weight = weight.at[indices_to_update].set(weight[indices_to_update] * scale[:, None])
    return weight


# - func: _embedding_bag_forward_only(
# Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False,
# int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1) -> (Tensor, Tensor, Tensor, Tensor)
def _aten__embedding_bag(
    weight,
    indices,
    offsets=None,
    scale_grad_by_freq=False,
    mode=0,
    sparse=False,
    per_sample_weights=None,
    include_last_offset=False,
    padding_idx=-1,
):
    """Jax implementation of the PyTorch _embedding_bag function.

    Args:
        weight: The learnable weights of the module of shape (num_embeddings, embedding_dim).
        indices: A LongTensor containing the indices to extract.
        offsets: A LongTensor containing the starting offset of each bag.
        scale_grad_by_freq: Whether to scale gradients by the inverse of frequency of the words in the mini-batch.
        mode: 0 = "sum", 1 = "mean" or 2 = "max"
        sparse: Whether the gradients with respect to weight should be a sparse tensor.
        per_sample_weights: If given, each embedding vector is weighted by per_sample_weights
        include_last_offset: Whether to include the last offset as a valid bag.
        padding_idx: If specified, the entries at padding_idx do not contribute to the gradient.

    Returns:
        A tuple of (output, offset2bag, bag_size, max_indices).
    """
    embedded = _aten_embedding(weight, indices, padding_idx)

    if offsets is None:
        # offsets is None only when indices.ndim > 1
        if mode == 0:  # sum
            output = jnp.sum(embedded, axis=1)
        elif mode == 1:  # mean
            output = jnp.mean(embedded, axis=1)
        elif mode == 2:  # max
            output = jnp.max(embedded, axis=1)
        return output, None, None, None

    if isinstance(offsets, jax.Array):
        offsets_np = np.array(offsets)
    else:
        offsets_np = offsets
    offset2bag = np.zeros(indices.shape[0], dtype=np.int64)
    bag_size = np.zeros(offsets_np.shape[0], dtype=np.int64)
    max_indices = jnp.full_like(indices, -1)

    for bag in range(offsets_np.shape[0]):
        start = int(offsets_np[bag])

        end = int(indices.shape[0] if bag + 1 == offsets_np.shape[0] else offsets_np[bag + 1])
        bag_size[bag] = end - start
        offset2bag = offset2bag.at[start:end].set(bag)

        if end - start > 0:
            if mode == 0:
                output_bag = jnp.sum(embedded[start:end], axis=0)
            elif mode == 1:
                output_bag = jnp.mean(embedded[start:end], axis=0)
            elif mode == 2:
                output_bag = jnp.max(embedded[start:end], axis=0)
                max_indices = max_indices.at[start:end].set(jnp.argmax(embedded[start:end], axis=0))

    # The original code returned offset2bag, bag_size, and max_indices as numpy arrays.
    # Converting them to JAX arrays for consistency.
    offset2bag = jnp.array(offset2bag)
    bag_size = jnp.array(bag_size)

    return output_bag, offset2bag, bag_size, max_indices


@register_aten(
    ["aten::rsqrt", "aten::rsqrt.out"],
)
def _aten_rsqrt(x):
    if jnp.isdtype(x.dtype, "integral"):
        x = x.astype(jnp.float32)
    return jax.lax.rsqrt(x)


def _aten_dot(x, y):
    return jnp.dot(x, y)


def _ones(size: Sequence[int], dtype=None, **kwargs):
    return jnp.ones(size, dtype)


@register_aten(
    ["aten::zeros", "aten:zeros.out"],
    static_argnums=(0,),
    static_argnames=("dtype",),
    operation_type="creation",
)
def _zeros(size: Sequence[int], dtype=None, **kwargs):
    jdtype = convert_dtype_with_default(dtype)
    return jnp.zeros(size, jdtype)


@register_aten(
    ["aten::full", "aten::full.out"],
    operation_type="creation",
    static_argnums=(0, 1),
    static_argnames=("dtype",),
)
# aten.full
def _full(size: Sequence[int], fill_value, *, dtype=None, **kwargs):
    jdtype = convert_dtype_with_default(dtype)
    return jnp.full(size, fill_value, jdtype)


def _aten_empty_permuted(sizes, physical_layout, dtype=None, **kwargs):
    # Ignore the physical layout,
    # since JAX and torch tensor doesn't share the same memory.
    return jnp.empty(sizes, dtype=dtype)


def _aten_empty_strided(sizes, stride, dtype=None, **kwargs):
    # Ignore stride, since JAX and torch tensor doesn't share the same memory.
    return jnp.empty(sizes, dtype=dtype)


def _aten_ne(x, y):
    return jnp.not_equal(x, y)


# Create indices along a specific axis
#
# For example
# x = jnp.zeros((3,4))
#
# _indices_along_axis(x, axis=0)
# >> [[0], [1], [2]] shape (3, 1)
#
# _indices_along_axis(x, axis=1)
# >> [[0, 1, 2, 3]] shape (1, 4)
def _indices_along_axis(x, axis):
    return jnp.expand_dims(
        jnp.arange(x.shape[axis]), axis=[d for d in range(len(x.shape)) if d != axis]
    )


def _broadcast_indices(indices, shape):
    return jnp.broadcast_to(indices, shape)


def _aten_cummax(x, dim):
    if not x.shape:
        return x, jnp.zeros_like(x, dtype=jnp.int64)

    axis = dim

    indice_along_axis = _indices_along_axis(x, axis)
    indices = _broadcast_indices(indice_along_axis, x.shape)

    def cummax_reduce_func(carry, elem):
        v1, v2 = carry["val"], elem["val"]
        i1, i2 = carry["idx"], elem["idx"]

        v = jnp.maximum(v1, v2)
        i = jnp.where(v1 > v2, i1, i2)
        return {"val": v, "idx": i}

    res = jax.lax.associative_scan(cummax_reduce_func, {"val": x, "idx": indices}, axis=axis)
    return res["val"], res["idx"]


def _aten_cummin(x, dim):
    if not x.shape:
        return x, jnp.zeros_like(x, dtype=jnp.int64)

    axis = dim

    indice_along_axis = _indices_along_axis(x, axis)
    indices = _broadcast_indices(indice_along_axis, x.shape)

    def cummin_reduce_func(carry, elem):
        v1, v2 = carry["val"], elem["val"]
        i1, i2 = carry["idx"], elem["idx"]

        v = jnp.minimum(v1, v2)
        i = jnp.where(v1 < v2, i1, i2)
        return {"val": v, "idx": i}

    res = jax.lax.associative_scan(cummin_reduce_func, {"val": x, "idx": indices}, axis=axis)
    return res["val"], res["idx"]


# aten.cumsum
@register_aten(
    [
        "aten::cumsum",
        "aten::cumsum.out",
    ],
    operation_type="reduction",
    static_argnums=(1,),
    static_argnames=("dtype", "__out_dtype"),
)
def _aten_cumsum(x, y, dtype=None, __out_dtype=None):
    # Handle empty tensors properly by returning an empty tensor with the same shape and specified dtype
    if x.size == 0:
        return jnp.empty(x.shape, dtype=__out_dtype)

    # Handle scalar tensors (0-dimensional)
    if x.ndim == 0:
        return x.astype(__out_dtype)

    res = jnp.cumsum(x, axis=y, dtype=__out_dtype)
    return res


def _aten_cumprod(input, dim, dtype=None, out=None):
    if dtype:
        dtype = mappings.t2j_dtype(dtype)
    if len(input.shape) > 0:
        res = jnp.cumprod(input, axis=dim, dtype=dtype)
    elif dtype:
        res = input.astype(dtype)
    else:
        res = input
    return res


@register_aten(
    ["aten::native_layer_norm", "aten::native_layer_norm.out"],
    static_argnums=(
        1,
        4,
    ),
)
def _aten_native_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    """Implements layer normalization in Jax as defined by `aten::native_layer_norm`.

    Args:
      input: The input tensor.
      normalized_shape: A list of integer dimensions to be normalized over.
      weight: Optional weight tensor for the affine transformation.
      bias: Optional bias tensor for the affine transformation.
      eps: A small epsilon value for numerical stability.

    Returns:
      output: The normalized tensor.
      mean: The calculated mean tensor.
      std: The calculated standard deviation tensor.
    """
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]
    axis = [len(input.shape) - i - 1 for i in range(len(normalized_shape))]

    # Calculate mean and standard deviation
    mean = jnp.mean(input, axis=axis, keepdims=True)
    var = jnp.var(input, axis=axis, keepdims=True)
    rstd = jax.lax.rsqrt(var + eps)

    # Normalize the input
    norm_x = (input - mean) * rstd

    # Apply affine transformation (if provided)
    if weight is not None:
        norm_x *= weight
    if bias is not None:
        norm_x += bias
    return norm_x, mean, rstd


def _aten_matmul(x, y):
    return x @ y


# todo(thangakr): handle common cases more effeciently
@register_aten(
    ["aten::addmv", "aten::addmv_", "aten::addmv.out"],
    static_argnames=("alpha", "beta"),
)
def _aten_addmv(self, mat, vec, *, beta=1, alpha=1, out=None):
    self_arr = jnp.asarray(self)
    mat_arr = jnp.asarray(mat)
    vec_arr = jnp.asarray(vec)

    check_constraint(
        mat_arr.ndim == 2 and vec_arr.ndim == 1 and self_arr.ndim <= 1,
        ADDMV_DIMENSION_ERROR,
        (self_arr, mat_arr, vec_arr),
    )

    mat_vec_match = mat_arr.shape[1] == vec_arr.shape[0]
    self_numel = self_arr.size
    shape_match = mat_arr.shape[0] == self_numel or self_numel == 1
    check_constraint(
        mat_vec_match and shape_match,
        ADDMV_SHAPE_MISMATCH,
        (self_arr, mat_arr, vec_arr),
    )

    common_dtype = jnp.result_type(self_arr, mat_arr, vec_arr, alpha, beta)
    self_arr = self_arr.astype(common_dtype)
    mat_arr = mat_arr.astype(common_dtype)
    vec_arr = vec_arr.astype(common_dtype)

    alpha_val = jnp.array(alpha, dtype=common_dtype)
    beta_val = jnp.array(beta, dtype=common_dtype)

    matvec = jnp.matmul(mat_arr, vec_arr)
    return beta_val * self_arr + alpha_val * matvec


# - func: addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
def _aten_addmm(self, mat1, mat2, *, beta=1.0, alpha=1.0):
    # Validate dimensions for matrix multiplication
    check_constraint(mat1.ndim == 2, MM_DIMENSION_ERROR, ("mat1",))
    check_constraint(mat2.ndim == 2, MM_DIMENSION_ERROR, ("mat2",))
    check_constraint(mat1.shape[1] == mat2.shape[0], MM_CANNOT_MULTIPLY, (mat1, mat2))

    # Check if input is broadcastable to result shape
    result_shape = (mat1.shape[0], mat2.shape[1])
    # JAX will handle scalar and 1D broadcasts, but we need to check 2D compatibility
    if self.ndim == 2:
        check_constraint(
            self.shape[0] == result_shape[0] or self.shape[0] == 1,
            ADDMM_BROADCAST_ERROR,
            (self, result_shape, "rows"),
        )
        check_constraint(
            self.shape[1] == result_shape[1] or self.shape[1] == 1,
            ADDMM_BROADCAST_ERROR,
            (self, result_shape, "cols"),
        )

    alpha = jnp.array(alpha).astype(mat1.dtype)
    beta = jnp.array(beta).astype(mat1.dtype)
    self *= beta
    self += alpha * jnp.matmul(mat1, mat2)
    return self


def _aten_sparse_addmm(self, mat1, mat2, *, beta=1.0, alpha=1.0):
    alpha = jnp.array(alpha).astype(mat1.dtype)
    beta = jnp.array(beta).astype(mat1.dtype)
    self *= beta
    self += alpha * jnp.matmul(mat1, mat2) * (self != 0)
    return self


def _aten_addbmm(input, batch1, batch2, *, beta=1, alpha=1):
    alpha = jnp.array(alpha).astype(batch1.dtype)
    beta = jnp.array(beta).astype(batch1.dtype)
    mm = jnp.einsum("bxy, byz -> xz", batch1, batch2)
    return jax.lax.cond(beta == 0, lambda: alpha * mm, lambda: beta * input + alpha * mm)


def _aten_gelu(self, *, approximate="none"):
    if approximate not in ["none", "tanh"]:
        raise RuntimeError(
            f"approximate argument must be either none or tanh, but got: {approximate}"
        )
    approx = approximate == "tanh"
    return jax.nn.gelu(self, approx)


def _aten_bucketize(input, boundaries, *, out_int32=False, right=False, out=None):
    return_type = jnp.int32 if out_int32 else jnp.int64
    return jnp.digitize(input, boundaries, right=not right).astype(return_type)


@register_aten(
    ["aten::convolution"],
    static_argnums=(3, 4, 5, 6, 7, 8),
)
def _aten_convolution(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
):
    from torch_neuronx.python_ops.jax.ops.convolution_backward import (
        _validate_convolution_inputs,
        expand_param_if_needed,
    )

    num_spatial_dims = weight.ndim - 2
    padding = expand_param_if_needed(padding, num_spatial_dims)

    _validate_convolution_inputs(
        input, weight, stride, padding, dilation, transposed, output_padding, groups
    )

    # Handle empty tensors (zero channels)
    if 0 in input.shape or 0 in weight.shape:
        batch_size = input.shape[0]
        out_channels = weight.shape[0]

        # Calculate spatial output dimensions
        spatial_dims = []
        for i in range(len(stride)):
            in_size = input.shape[i + 2]
            kernel_size = weight.shape[i + 2]
            pad = padding[i] if isinstance(padding, (list, tuple)) else padding
            out_size = (in_size + 2 * pad - dilation[i] * (kernel_size - 1) - 1) // stride[i] + 1
            spatial_dims.append(out_size)

        output_shape = [batch_size, out_channels] + spatial_dims
        return jnp.zeros(output_shape, dtype=input.dtype)

    num_shape_dim = weight.ndim - 1
    batch_dims = input.shape[:-num_shape_dim]

    input = input.reshape((-1, *input.shape[-num_shape_dim:]))

    def make_padding(padding, num_spatial_dims):
        # Convert padding to pairs expected by jax
        if transposed:
            # See https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
            pad_out = []
            for i in range(num_spatial_dims):
                front = dilation[i] * (weight.shape[i + 2] - 1) - padding[i]
                back = front + output_padding[i]
                pad_out.append((front, back))
            return pad_out
        else:
            return ((p, p) for p in padding)

    def create_default_conv_dimension_numbers(num_spatial_dims):
        # Ref: https://github.com/openxla/xla/blob/main/xla/client/xla_builder.cc#L4211
        # (batch dimension, feature dimension, spatial dimensions...)
        lhs_spec = [0, 1]
        # (out feature dimension, in feature dimension, spatial dimensions...)
        # swapped for transposed convolution
        rhs_spec = [1, 0] if transposed else [0, 1]
        # (batch dimension, feature dimension, spatial dimensions...)
        out_spec = [0, 1]
        for i in range(0, num_spatial_dims):
            lhs_spec.append(i + 2)
            rhs_spec.append(i + 2)
            out_spec.append(i + 2)
        return jax.lax.ConvDimensionNumbers(*map(tuple, (lhs_spec, rhs_spec, out_spec)))

    if transposed:  # TODO pending test of transpose
        rhs = jnp.flip(weight, range(2, 1 + num_shape_dim))
        if groups != 1:
            # reshape filters for tranposed depthwise convolution
            assert rhs.shape[0] % groups == 0
            rhs_shape = [rhs.shape[0] // groups, rhs.shape[1] * groups]
            rhs_shape.extend(rhs.shape[2:])
            rhs = jnp.reshape(rhs, rhs_shape)
        res = jax.lax.conv_general_dilated(
            input,
            rhs,
            (1,) * len(stride),
            make_padding(padding, len(stride)),
            lhs_dilation=stride,
            rhs_dilation=dilation,
            dimension_numbers=create_default_conv_dimension_numbers(len(stride)),
            feature_group_count=groups,
            batch_group_count=1,
        )
    else:
        res = jax.lax.conv_general_dilated(
            input,
            weight,
            stride,
            make_padding(padding, len(stride)),
            lhs_dilation=(1,) * len(stride),
            rhs_dilation=dilation,
            dimension_numbers=create_default_conv_dimension_numbers(len(stride)),
            feature_group_count=groups,
            batch_group_count=1,
        )

    if bias is not None:
        # TODO(qihqi): bias always on channel?
        if len(bias.shape) == 1:
            shape = [1] * len(res.shape)
            shape[1] = bias.shape[0]
            bias = bias.reshape(tuple(shape))
        res = res + bias

    res = res.reshape((*batch_dims, *res.shape[-num_shape_dim:]))
    return res


# _native_batch_norm_legit(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, bool training, float momentum, float eps)
def _aten__native_batch_norm_legit(
    input, weight, bias, running_mean, running_var, training, momentum, eps
):
    """JAX implementation of batch normalization with optional parameters.
    Refers to https://github.com/pytorch/pytorch/blob/cd3a71f754a2248bcfe500de7c9860bd7d2002bf/torch/_decomp/decompositions.py#L1713.

    Args:
      input (DeviceArray): Input data (N, C, H, W).
      running_mean ([DeviceArray]): Running mean of input (C,).
      running_var ([DeviceArray]): Running variance of input (C,).
      weight (Optional[DeviceArray]): Scaling factor (gamma) (C,). Can be None.
      bias (Optional[DeviceArray]): Shift factor (beta) (C,). Can be None.
      training (bool): If True, use batch statistics for normalization.
                       If False, use running statistics.
      momentum (float): Momentum factor for updating running statistics.
      eps (float): Small constant for numerical stability.

    Returns:
      DeviceArray: Normalized output
      DeviceArray: Batch mean (C,) or empty if training is False
      DeviceArray: Reversed batch variance (C,) or empty if training is False
    """
    reduction_dims = [0] + list(range(2, input.ndim))
    reshape_dims = [1, -1] + [1] * (input.ndim - 2)
    if training:
        # Calculate batch mean and variance
        mean = jnp.mean(input, axis=reduction_dims, keepdims=True)
        saved_mean = jnp.squeeze(mean, reduction_dims)
        var = jnp.var(input, axis=reduction_dims)
        rstd = jax.lax.rsqrt(var.reshape(reshape_dims) + eps)
        # Update running statistics using momentum
        running_mean = (1 - momentum) * running_mean + momentum * saved_mean
        running_var = (1 - momentum) * running_var + momentum * var
        saved_rstd = jnp.squeeze(rstd, reduction_dims)
    else:
        rstd = jax.lax.rsqrt(running_var.reshape(reshape_dims) + eps)
        saved_mean = jnp.array(
            [], dtype=input.dtype
        )  # No need to calculate batch statistics in inference mode
        saved_rstd = jnp.array([], dtype=input.dtype)

    # Normalize
    if training:
        # use batch statistics if training
        x_hat = (input - mean) * rstd
    else:
        # Use running statistics in inference mode
        x_hat = (input - running_mean.reshape(reshape_dims)) * rstd

    # Scale and shift
    if weight is not None:
        x_hat *= weight.reshape(reshape_dims)  # Reshape weight for broadcasting
    if bias is not None:
        x_hat += bias.reshape(reshape_dims)  # Reshape bias for broadcasting

    return x_hat, saved_mean, saved_rstd


def _aten__native_batch_norm_legit_no_training(
    input, weight, bias, running_mean, running_var, momentum, eps
):
    return _aten__native_batch_norm_legit(
        input, weight, bias, running_mean, running_var, False, momentum, eps
    )


def _aten_relu(self):
    return jax.nn.relu(self)


def _aten_cat(tensors, dims=0):
    # handle empty tensors as a special case.
    # torch.cat will ignore the empty tensor, while jnp.concatenate
    # will error if the dims > 0.
    filtered_tensors = [t for t in tensors if not (t.ndim == 1 and t.shape[0] == 0)]
    if filtered_tensors:
        return jnp.concatenate(filtered_tensors, dims)
    return tensors[0]


def _ceil_mode_padding(
    padding: list[int],
    input_shape: list[int],
    kernel_size: list[int],
    stride: list[int],
    dilation: list[int],
    ceil_mode: bool,
):
    """Creates low and high padding specification for the given padding (which is symmetric) and ceil mode.

    Additional high padding could be required when ceil mode is set.
    """
    ceil_mode_padding = []
    for i in range(len(padding)):
        left_padding = padding[i]
        right_padding = left_padding

        input_size = input_shape[2 + i]
        output_size_rem = (
            input_size + 2 * left_padding - (kernel_size[i] - 1) * dilation[i] - 1
        ) % stride[i]
        if ceil_mode and output_size_rem != 0:
            extra_padding = stride[i] - output_size_rem
            new_output_size = (
                input_size
                + left_padding
                + right_padding
                + extra_padding
                - (kernel_size[i] - 1) * dilation[i]
                - 1
                + stride[i]
                - 1
            ) // stride[i] + 1
            # Ensure that the last pooling starts inside the image.
            size_to_compare = input_size + left_padding

            if (new_output_size - 1) * stride[i] < size_to_compare:
                right_padding += extra_padding

        ceil_mode_padding.append((left_padding, right_padding))
    return ceil_mode_padding


def _aten_max_pool2d_with_indices(
    inputs, kernel_size, strides=None, padding=0, dilation=1, ceil_mode=False
):
    num_batch_dims = len(inputs.shape) - len(kernel_size) - 1
    kernel_size = tuple(kernel_size)
    # Default stride is kernel_size
    strides = tuple(strides) if strides else kernel_size
    if isinstance(padding, int):
        padding = [padding for _ in range(len(kernel_size))]
    if isinstance(dilation, int):
        dilation = tuple(dilation for _ in range(len(kernel_size)))
    elif isinstance(dilation, list):
        dilation = tuple(dilation)

    input_shape = inputs.shape
    if num_batch_dims == 0:
        input_shape = [1, *input_shape]
    padding = _ceil_mode_padding(padding, input_shape, kernel_size, strides, dilation, ceil_mode)

    assert len(kernel_size) == len(strides), f"len({kernel_size=}) must equal len({strides=})"
    assert len(kernel_size) == len(dilation), f"len({kernel_size=}) must equal len({dilation=})"
    strides = (1,) * (1 + num_batch_dims) + strides
    dims = (1,) * (1 + num_batch_dims) + kernel_size
    dilation = (1,) * (1 + num_batch_dims) + dilation

    is_single_input = False
    if num_batch_dims == 0:
        # add singleton batch dimension because lax.reduce_window always
        # needs a batch dimension.
        inputs = inputs[None]
        strides = (1,) + strides
        dims = (1,) + dims
        dilation = (1,) + dilation
        is_single_input = True

    assert inputs.ndim == len(dims), f"len({inputs.shape}) != len({dims})"
    if not isinstance(padding, str):
        padding = tuple(map(tuple, padding))
        assert len(padding) == len(kernel_size), (
            f"padding {padding} must specify pads for same number of dims as "
            f"kernel_size {kernel_size}"
        )
        assert all(
            [len(x) == 2 for x in padding]
        ), f"each entry in padding {padding} must be length 2"
        padding = ((0, 0), (0, 0)) + padding

    indices = jnp.arange(np.prod(inputs.shape[-len(kernel_size) :]))
    indices = indices.reshape(inputs.shape[-len(kernel_size) :])
    indices = jnp.broadcast_to(indices, inputs.shape)

    def reduce_fn(a, b):
        ai, av = a
        bi, bv = b
        which = av >= bv  # torch breaks ties in favor of later indices
        return jnp.where(which, ai, bi), jnp.where(which, av, bv)

    init_val = -jnp.inf
    if inputs.dtype in (jnp.int32, jnp.int64):
        init_val = -(1 << 31)
    init_val = jnp.array(init_val).astype(inputs.dtype)

    # Separate maxpool result and indices into two reduce_window ops. Since
    # the indices tensor is usually unused in inference, separating the two
    # can help DCE computations for argmax.
    y = jax.lax.reduce_window(
        inputs, init_val, jax.lax.max, dims, strides, padding, window_dilation=dilation
    )
    indices, _ = jax.lax.reduce_window(
        (indices, inputs),
        (0, init_val),
        reduce_fn,
        dims,
        strides,
        padding,
        window_dilation=dilation,
    )
    if is_single_input:
        indices = jnp.squeeze(indices, axis=0)
        y = jnp.squeeze(y, axis=0)

    return y, indices


# Aten ops registered under the `xla` library.
try:

    def _xla_max_pool2d_forward(*args, **kwargs):
        return _aten_max_pool2d_with_indices(*args, **kwargs)[0]

    def _xla_aot_mark_sharding(t, mesh: str, partition_spec: str):
        import ast

        import torch_xla.distributed.spmd as xs
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        pmesh = xs.Mesh.from_str(mesh)
        assert pmesh is not None
        partition_spec_eval = ast.literal_eval(partition_spec)
        jmesh = pmesh.get_jax_mesh()
        return jax.lax.with_sharding_constraint(t, NamedSharding(jmesh, P(*partition_spec_eval)))

    def _xla_einsum_linear_forward(input, weight, bias):
        with jax.named_scope("einsum_linear_forward"):
            product = jax.numpy.einsum("...n,mn->...m", input, weight)
            if bias is not None:
                return product + bias
            return product

except AttributeError:
    pass

# TODO add more ops


def _aten_min(x, dim=None, dtype=None, keepdim=False):
    if dim is not None:
        return _with_reduction_scalar(jnp.min, x, dim, keepdim), _with_reduction_scalar(
            jnp.argmin, x, dim, keepdim
        ).astype(jnp.int64)
    else:
        return _with_reduction_scalar(jnp.min, x, dim, keepdim)


def _aten_mode(input, dim=-1, keepdim=False, *, out=None):
    if input.ndim == 0:  # single number
        return input, jnp.array(0)
    dim = (input.ndim + dim) % input.ndim  # jnp.scipy.stats.mode does not accept -1 as dim
    # keepdims must be True for accurate broadcasting
    mode, _ = jax.scipy.stats.mode(input, axis=dim, keepdims=True)
    mode_broadcast = jnp.broadcast_to(mode, input.shape)
    if not keepdim:
        mode = mode.squeeze(axis=dim)
    indices = jnp.argmax(jnp.equal(mode_broadcast, input), axis=dim, keepdims=keepdim)
    return mode, indices


@register_aten(
    [
        "aten::amin",
        "aten::amin.out",
    ],
    operation_type="reduction",
    static_argnums=(1, 2),
    uses_preprocessing=True,
)
def _aten_amin(self, dim=None, keepdim=False, out=None):
    if out is not None and self.dtype != out.dtype:
        # amin CPU requires self and out to have the same dtype
        raise TypeError(
            f"Expected the dtype for input and out to match, but got {self.dtype} for input's dtype and {out.dtype} for out's dtype."
        )

    def _amin_fn(self, dim=None, keepdim=False, out=None):
        return _with_reduction_scalar(jnp.amin, self, dim, keepdim)

    return _amin_fn, (self, dim, keepdim), {"out": out}


def _aten_argmin(self, dim=None, keepdim=False):
    return _with_reduction_scalar(jnp.argmin, self, dim, keepdim)


def _aten_sin(x):
    return jnp.sin(x)


def _aten_sym_size(x, dim):
    return x.shape[dim]


def _aten_var(x, dim=None, *, correction=1, keepdim=False, dtype=None, out=None):
    result = jnp.var(x, axis=dim, ddof=correction, keepdims=keepdim)
    if dtype is not None:
        result = result.astype(dtype)
    return result


def _prims_broadcast_in_dim(t, shape, broadcast_dimensions):
    return jax.lax.broadcast_in_dim(t, shape, broadcast_dimensions=broadcast_dimensions)


# aten.native_group_norm -- should use decomp table
# func: native_group_norm(Tensor input, Tensor? weight, Tensor? bias, SymInt N, SymInt C, SymInt HxW, int group, float eps) -> (Tensor, Tensor, Tensor)


def _aten_native_group_norm(input, weight, bias, N, C, HxW, group, eps=1e-5):
    """Group Normalization implementation in JAX.

    Args:
      input: Input tensor. Expected shape (batch_size, channels, ... spatial dims
        ...)
      weight: Optional scaling (gamma) parameter. Shape (channels,)
      bias: Optional shifting (beta) parameter. Shape (channels,)
      N: Batch size.
      C: Number of channels.
      HxW: Product of spatial dimensions (number of elements per channel after
        flattening).
      group: Number of groups for Group Normalization.
      eps: Small value added for numerical stability.

    Returns:
      A tuple of (normalized_output, mean, rstd)
    """

    input_shape = input.shape

    if 0 in input_shape:
        return input, input, input

    # Reshape for group-wise normalization
    reshaped_input = jnp.reshape(input, (1, N * group, -1))

    # **Core Group Normalization**
    def group_norm_body(x):  # Function to apply within each group
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        rstd = jax.lax.rsqrt(var + eps)  # Reciprocal of std with epsilon
        normalized = (x - mean) * rstd
        return normalized, mean, rstd

    normalized, group_mean, group_rstd = jax.lax.map(group_norm_body, reshaped_input)

    # Reshape back to original input shape
    output = jnp.reshape(normalized, input_shape)

    # **Affine transformation**
    affine_shape = [-1 if i == 1 else 1 for i in range(input.ndim)]  # Shape for broadcasting
    if weight is not None and bias is not None:
        output = bias.reshape(affine_shape) + output * weight.reshape(affine_shape)
    elif weight is not None:
        output = output * weight.reshape(affine_shape)
    elif bias is not None:
        output = output + bias.reshape(affine_shape)

    # Reshape mean and rstd
    mean = jnp.reshape(group_mean, (N, group))
    rstd = jnp.reshape(group_rstd, (N, group))

    return output, mean, rstd


def _aten_linalg_vector_norm(self, ord=2, dim=None, keepdim=False, dtype=None):
    """Calculates the vector norm along specified dimensions.

    Args:
        self: The input tensor.
        ord: The order of the norm. Can be a float or 'inf', '-inf', 'fro'.
          Default is 2 (Euclidean norm).
        dim: Dimensions along which to calculate the norm. If None, the norm is
          calculated over all dimensions.
        keepdim: Whether to keep the reduced dimensions.
        dtype: Optional data type for the output.

    Returns:
        The tensor containing the calculated vector norms.
    """

    if ord not in {2, float("inf"), float("-inf"), "fro"} and not isinstance(ord, (int, float)):
        raise ValueError(
            f"Unsupported ord value: {ord}. Supported values are 2, inf, -inf, and 'fro'."
        )

    # Special cases (for efficiency and clarity)
    if ord == 0:
        if self.shape == ():
            # float sets it to float64. set it back to input type
            result = jnp.astype(jnp.array(float(self != 0)), self.dtype)
        else:
            result = _with_reduction_scalar(jnp.sum, jnp.where(self != 0, 1, 0), dim, keepdim)

    elif ord == 2:  # Euclidean norm
        result = jnp.sqrt(_with_reduction_scalar(jnp.sum, jnp.abs(self) ** 2, dim, keepdim))

    elif ord == float("inf"):
        result = _with_reduction_scalar(jnp.max, jnp.abs(self), dim, keepdim)

    elif ord == float("-inf"):
        result = _with_reduction_scalar(jnp.min, jnp.abs(self), dim, keepdim)

    elif ord == "fro":  # Frobenius norm
        result = jnp.sqrt(_with_reduction_scalar(jnp.sum, jnp.abs(self) ** 2, dim, keepdim))

    else:  # General case (e.g., ord = 1, ord = 3)
        result = _with_reduction_scalar(jnp.sum, jnp.abs(self) ** ord, dim, keepdim) ** (1.0 / ord)

    # (Optional) dtype conversion
    if dtype is not None:
        result = jnp.astype(result, self.dtype)

    new_dtype = mappings.t2j_dtype(torch.get_default_dtype())
    if result.dtype == jax.numpy.int64:
        result = result.astype(new_dtype)
    return result


# aten.reflection_pad1d
def _aten_reflection_pad1d(input, padding):
    rank = len(input.shape)
    pad_size = [(0, 0)] * rank
    pad_size[-1] = padding
    return jnp.pad(input, pad_size, mode="reflect")


# aten.sinh
def _aten_sinh(self):
    return jnp.sinh(self)


# aten.native_layer_norm_backward
@register_aten(
    [
        "aten::native_layer_norm_backward",
        "aten::native_layer_norm_backward.out",
        "aten::layer_norm_backward",
    ],
    static_argnums=(
        2,
        7,
    ),
)
def _aten_native_layer_norm_backward(
    grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask=(True, True, True)
):
    """Implements the backward pass of layer normalization in Jax.

    Based on PyTorch's implementation:
    https://github.com/pytorch/pytorch/blob/2b9ff9953523a2e916234c9197d946f4cff976c7/torch/_decomp/decompositions.py#L1650

    Args:
        grad_out: The gradient of the output tensor.
        input: The input tensor.
        normalized_shape: A list of integer dimensions to be normalized over.
        mean: Mean tensor from forward pass.
        rstd: Reciprocal standard deviation tensor from forward pass.
        weight: Optional weight tensor for the affine transformation.
        bias: Optional bias tensor for the affine transformation.
        output_mask: Tuple of booleans indicating which gradients to compute.

    Returns:
        A tuple of (grad_input, grad_weight, grad_bias).
    """
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]

    input_shape = input.shape
    input_ndim = input.ndim
    input_dtype = input.dtype
    axis = input_ndim - len(normalized_shape)

    inner_dims = input_shape[axis:]
    outer_dims = input_shape[:axis]

    inner_dim_indices = list(range(axis, input_ndim))
    outer_dim_indices = list(range(axis))

    # Compute N (number of elements being normalized over)
    N = 1
    for dim in inner_dims:
        N *= dim

    # Unsqueeze mean and rstd to match input dimensions
    mean_expanded = mean
    rstd_expanded = rstd
    for _ in range(input_ndim - mean.ndim):
        mean_expanded = jnp.expand_dims(mean_expanded, axis=-1)
        rstd_expanded = jnp.expand_dims(rstd_expanded, axis=-1)

    # Compute normalized input
    x_hat = (input - mean_expanded) * rstd_expanded

    # Compute grad_x_hat (gradient w.r.t. normalized input before affine transform)
    if weight is not None:
        grad_x_hat = grad_out * weight
    else:
        grad_x_hat = grad_out

    # Compute input gradient using PyTorch's formula
    grad_input = None
    if output_mask[0]:
        a = grad_x_hat * N
        b = jnp.sum(grad_x_hat, axis=inner_dim_indices, keepdims=True)
        c1 = grad_x_hat * x_hat
        c2 = jnp.sum(c1, axis=inner_dim_indices, keepdims=True)
        c3 = x_hat * c2

        inner = a - b - c3
        grad_input = (rstd_expanded / N) * inner

    # Compute weight gradient
    grad_weight = None
    if output_mask[1] and weight is not None:
        if len(outer_dim_indices) > 0:
            grad_weight = jnp.sum(grad_out * x_hat, axis=outer_dim_indices, keepdims=False)
        else:
            grad_weight = grad_out * x_hat

    # Compute bias gradient
    grad_bias = None
    if output_mask[2] and bias is not None:
        if len(outer_dim_indices) > 0:
            grad_bias = jnp.sum(grad_out, axis=outer_dim_indices, keepdims=False)
        else:
            grad_bias = grad_out

    if grad_input is not None:
        grad_input = grad_input.astype(input_dtype)
    if grad_weight is not None:
        grad_weight = grad_weight.astype(input_dtype)
    if grad_bias is not None:
        grad_bias = grad_bias.astype(input_dtype)

    return grad_input, grad_weight, grad_bias


# aten.reflection_pad3d_backward
# aten.reflection_pad2d


# aten.atanh
def _aten_atanh(self):
    res = jnp.arctanh(self)
    return res


# aten.bincount
def _aten_bincount(input, weights=None, minlength=0):
    return jnp.bincount(input, weights, minlength)


# aten.embedding_dense_backward


# aten.sum
@register_aten(
    [
        "aten::sum",
        "aten::sum.dim_IntList",
        "aten::sum.IntList_out",
    ],
    static_argnums=(1, 2),
    static_argnames=("dtype", "__out_dtype"),
    operation_type="reduction",
)
def _aten_sum(self, dim=None, keepdim=False, dtype=None, __out_dtype=None):
    result = _with_reduction_scalar(jnp.sum, self, dim, keepdim)
    if __out_dtype is not None:
        result = result.astype(__out_dtype)

    return result


# aten.sqrt
@register_aten(
    ["aten::sqrt", "aten::sqrt.out"],
)
def _aten_sqrt(self):
    if jnp.isdtype(self.dtype, "integral"):
        self = self.astype(jnp.float32)
    return jnp.sqrt(self)


# aten.tanh
@register_aten(
    [
        "aten::tanh",
        "aten::tanh.out",
    ],
)
def _aten_tanh(self):
    res = jnp.tanh(self)
    return res


# aten.tanh_backward
@register_aten(
    [
        "aten::tanh_backward",
    ],
)
def _aten_tanh_backward(grad_output, output):
    res = grad_output * (1.0 - output * output)
    return res


# aten.ceil
@register_aten(
    [
        "aten::ceil",
        "aten::ceil.out",
    ],
    operation_type="arithmetic",
)
def _aten_ceil(self):
    return jnp.ceil(self).astype(self.dtype)


# aten.asin
def _aten_asin(self):
    res = jnp.arcsin(self)
    return res


# aten.minimum
def _aten_minimum(self, other):
    return jnp.minimum(self, other)


# aten.max_pool2d_backward


def _scatter_index(dim, index):
    """Returns a tuple of indexes;

    The first is to select in input (to modify),
    the second is to select from the values.
    """
    index_shape = list(index.shape)
    input_indexes = []
    source_indexes = []
    if dim < 0:
        dim += len(index_shape)
    for i in range(len(index_shape)):
        source_indexes.append(slice(0, index_shape[i]))
        if i == dim:
            input_indexes.append(index)
        else:
            target_shape = [1] * len(index_shape)
            target_shape[i] = index_shape[i]
            input_indexes.append(
                jnp.broadcast_to(jnp.arange(index_shape[i]).reshape(target_shape), index_shape)
            )
    return tuple(input_indexes), tuple(source_indexes)


# aten.scatter_add
@register_aten(
    ["aten::scatter_add.out"],
    static_argnums=(1,),
)
def _aten_scatter_add(input, dim, index, src):
    """JAX implementation of scatter, mimicking torch.scatter behavior"""

    input_indexes, source_indexes = _scatter_index(dim, index)
    return input.at[input_indexes].add(src[source_indexes])


# aten.masked_scatter
def _aten_masked_scatter(self, mask, source):
    broadcast_shape = jnp.broadcast_shapes(self.shape, mask.shape)

    if self.shape != broadcast_shape:
        self = jnp.broadcast_to(self, broadcast_shape)
    elif mask.shape != broadcast_shape:
        mask = jnp.broadcast_to(mask, broadcast_shape)

    self_flat = self.flatten()
    mask_flat = mask.flatten()
    source_flat = source.flatten()

    true_indices = jnp.where(mask_flat)[0]
    self_flat = self_flat.at[true_indices].set(source_flat[: len(true_indices)])
    final_arr = self_flat.reshape(self.shape)

    return final_arr


# aten.logical_not


@register_aten(
    ["aten::sign", "aten::sign.out"],
)
def _aten_sign(x):
    return jnp.sign(x)


# aten.signbit
def _aten_signbit(x):
    return jnp.signbit(x)


# aten.sigmoid
@register_aten(
    ["aten::sigmoid", "aten::sigmoid.out"],
)
def _aten_sigmoid(x):
    if jnp.isdtype(x.dtype, "integral"):
        x = x.astype(jnp.float32)
    return jax.nn.sigmoid(x)


# implement aten.asinh in jax
def _aten_asinh(self):
    res = jnp.arcsinh(self)
    return res


# aten.atan
def _aten_atan(self):
    res = jnp.arctan(self)
    return res


def _aten_scatter_reduce(input, dim, index, src, reduce=None, *, include_self=True):
    if not isinstance(src, jnp.ndarray):
        src = jnp.array(src, dtype=input.dtype)
    input_indexes, source_indexes = _scatter_index(dim, index)
    # "Zero out" target elements when not included
    if not include_self:
        if reduce in ["sum", "mean"]:
            base_input = jnp.zeros_like(src)
        elif reduce == "prod":
            base_input = jnp.ones_like(src)
        elif reduce == "amax":
            base_input = jnp.full_like(src, -jnp.inf)
        else:  # amin
            base_input = jnp.full_like(src, jnp.inf)
        input = input.at[input_indexes].set(base_input[source_indexes])

    if reduce == "sum" or reduce == "add":
        return input.at[input_indexes].add(src[source_indexes])
    elif reduce == "prod" or reduce == "multiply":
        return input.at[input_indexes].multiply(src[source_indexes])
    elif reduce == "mean":
        if include_self:
            count = jnp.ones_like(input)
        else:
            count = jnp.zeros_like(input)
        count = count.at[input_indexes].add(jnp.ones_like(src)[source_indexes])
        count = jnp.clip(count, min=1)
        mean = input.at[input_indexes].add(src[source_indexes])
        if _is_int(input):
            return mean // count
        return mean / count
    elif reduce == "amax":
        return input.at[input_indexes].max(src[source_indexes])
    elif reduce == "amin":
        return input.at[input_indexes].min(src[source_indexes])
    else:
        return input.at[input_indexes].set(src[source_indexes])


# aten.acos
def _aten_acos(self):
    return jnp.arccos(self)


# aten.sym_storage_offset
# aten.native_layer_norm_backward
# aten.max_pool3d_with_indices


# aten.gt
def _aten_gt(self, other):
    return self > other


# aten.sym_stride
# aten.lt
def _aten_lt(self, other):
    return self < other


def pool(inputs, init, reduce_fn, window_shape, strides, padding):
    """Helper function to define pooling functions.

    Pooling functions are implemented using the ReduceWindow XLA op.
    NOTE: Be aware that pooling is not generally differentiable.
    That means providing a reduce_fn that is differentiable does not imply that
    pool is differentiable.

    Args:
      inputs: input data with dimensions (batch, window dims..., features).
      init: the initial value for the reduction
      reduce_fn: a reduce function of the form ``(T, T) -> T``.
      window_shape: a shape tuple defining the window to reduce over.
      strides: a sequence of ``n`` integers, representing the inter-window
        strides (default: ``(1, ..., 1)``).
      padding: either the string ``'SAME'``, the string ``'VALID'``, or a sequence
        of ``n`` ``(low, high)`` integer pairs that give the padding to apply before
        and after each spatial dimension.
    Returns:
      The output of the reduction for each window slice.
    """
    num_batch_dims = inputs.ndim - (len(window_shape) + 1)
    strides = strides or (1,) * len(window_shape)
    assert len(window_shape) == len(strides), f"len({window_shape}) must equal len({strides})"
    strides = (1,) * (1 + num_batch_dims) + strides
    dims = (1,) * (1 + num_batch_dims) + window_shape

    is_single_input = False
    if num_batch_dims == 0:
        # add singleton batch dimension because lax.reduce_window always
        # needs a batch dimension.
        inputs = inputs[None]
        strides = (1,) + strides
        dims = (1,) + dims
        is_single_input = True

    assert inputs.ndim == len(dims), f"len({inputs.shape}) != len({dims})"
    if not isinstance(padding, str):
        padding = tuple(map(tuple, padding))
        assert len(padding) == len(window_shape), (
            f"padding {padding} must specify pads for same number of dims as "
            f"window_shape {window_shape}"
        )
        assert all(
            [len(x) == 2 for x in padding]
        ), f"each entry in padding {padding} must be length 2"
        padding = ((0, 0), (0, 0)) + padding
    y = jax.lax.reduce_window(inputs, init, reduce_fn, dims, strides, padding)
    if is_single_input:
        y = jnp.squeeze(y, axis=0)
    return y


def adaptive_avg_pool2or3d(input: jnp.ndarray, output_size: tuple[int, int]) -> jnp.ndarray:
    """
    Applies a 2/3D adaptive average pooling over an input signal composed of several input planes.

    See :class:`~torch.nn.AdaptiveAvgPool2d` for details and output shape.

    Args:
        input: input tensor
        output_size: the target output size (single integer or double-integer tuple)

    Context:
      https://github.com/pytorch/pytorch/blob/main/torch/_decomp/decompositions.py#L2401
    """
    shape = input.shape
    ndim = len(shape)
    out_dim = len(output_size)
    num_spatial_dim = ndim - out_dim

    # Preconditions

    assert (
        ndim in (out_dim + 1, out_dim + 2)
    ), f"adaptive_avg_pool{num_spatial_dim}d(): Expected {num_spatial_dim + 1}D or {num_spatial_dim + 2}D tensor, but got {ndim}"
    for d in input.shape[-2:]:
        assert d != 0, (
            "adaptive_avg_pool{num_spactial_dim}d(): Expected input to have non-zero size for "
            f"non-batch dimensions, but input has shape {tuple(shape)}."
        )

    # Optimisation (we should also do this in the kernel implementation)
    if all(s % o == 0 for o, s in zip(output_size, shape[-out_dim:], strict=False)):
        stride = tuple(i // o for i, o in zip(shape[-out_dim:], output_size, strict=False))
        kernel = tuple(
            i - (o - 1) * s for i, o, s in zip(shape[-out_dim:], output_size, stride, strict=False)
        )
        return _aten_avg_pool(
            input,
            kernel,
            strides=stride,
        )

    def start_index(a, b, c):
        return (a * c) // b

    def end_index(a, b, c):
        return ((a + 1) * c + b - 1) // b

    def compute_idx(in_size, out_size):
        orange = jnp.arange(out_size, dtype=jnp.int64)
        i0 = start_index(orange, out_size, in_size)
        # Let length = end_index - start_index, i.e. the length of the pooling kernels
        # length.max() can be computed analytically as follows:
        maxlength = in_size // out_size + 1
        in_size_mod = in_size % out_size
        # adaptive = True iff there are kernels with different lengths
        adaptive = not (in_size_mod == 0 or out_size % in_size_mod == 0)
        if adaptive:
            maxlength += 1
        elif in_size_mod == 0:
            maxlength -= 1

        range_max = jnp.arange(maxlength, dtype=jnp.int64)
        idx = i0[:, None] + range_max
        if adaptive:
            # Need to clamp to avoid accessing out-of-bounds memory
            idx = jnp.minimum(idx, in_size - 1)

            # Compute the length
            i1 = end_index(orange, out_size, in_size)
            length = i1 - i0
        else:
            length = maxlength
        return idx, length, range_max, adaptive

    idx, length, range_max, adaptive = [[None] * out_dim for _ in range(4)]
    # length is not None if it's constant, otherwise we'll need to compute it
    for i, (s, o) in enumerate(zip(shape[-out_dim:], output_size, strict=False)):
        idx[i], length[i], range_max[i], adaptive[i] = compute_idx(s, o)

    def _unsqueeze_to_dim(x, dim):
        ndim = len(x.shape)
        return jax.lax.expand_dims(x, tuple(range(ndim, dim)))

    if out_dim == 2:
        # NOTE: unsqueeze to insert extra 1 in ranks; so they
        # would broadcast
        vals = input[..., _unsqueeze_to_dim(idx[0], 4), idx[1]]
        reduce_axis = (-3, -1)
    else:
        assert out_dim == 3
        vals = input[..., _unsqueeze_to_dim(idx[0], 6), _unsqueeze_to_dim(idx[1], 4), idx[2]]
        reduce_axis = (-5, -3, -1)

    # Shortcut for the simpler case
    if not any(adaptive):
        return jnp.mean(vals, axis=reduce_axis)

    def maybe_mask(vals, length, range_max, adaptive, dim):
        if isinstance(length, int):
            return vals, length
        else:
            # zero-out the things we didn't really want to select
            assert dim < 0
            # hack
            mask = range_max >= length[:, None]
            if dim == -2:
                mask = _unsqueeze_to_dim(mask, 4)
            elif dim == -3:
                mask = _unsqueeze_to_dim(mask, 6)
            vals = jnp.where(mask, 0.0, vals)
            # Compute the length of each window
            length = _unsqueeze_to_dim(length, -dim)
            return vals, length

    for i in range(len(length)):
        vals, length[i] = maybe_mask(
            vals, length[i], range_max[i], adaptive=adaptive[i], dim=(i - out_dim)
        )

    # We unroll the sum as we assume that the kernels are going to be small
    ret = jnp.sum(vals, axis=reduce_axis)
    # NOTE: math.prod because we want to expand it to length[0] * length[1] * ...
    # this is multiplication with broadcasting, not regular pointwise product
    return ret / math.prod(length)


def _aten_avg_pool(
    inputs,
    kernel_size,
    strides=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    num_batch_dims = len(inputs.shape) - len(kernel_size) - 1
    kernel_size = tuple(kernel_size)
    strides = tuple(strides) if strides else kernel_size
    if isinstance(padding, list) and len(padding) == 1:
        padding = padding[0]
    if isinstance(padding, int):
        padding = [padding for _ in range(len(kernel_size))]

    input_shape = inputs.shape
    if num_batch_dims == 0:
        input_shape = [1, *input_shape]
    padding = _ceil_mode_padding(
        padding, input_shape, kernel_size, strides, [1] * len(kernel_size), ceil_mode
    )

    y = pool(inputs, 0.0, jax.lax.add, kernel_size, strides, padding)
    if divisor_override is not None:
        y = y / jnp.array(divisor_override, y.dtype)
    elif count_include_pad:
        div_shape = list(y.shape)
        div_by = jnp.ones(div_shape, y.dtype) * np.prod(kernel_size)
        unequal_paddings = map(lambda pad: pad[0] != pad[1], padding)
        unequal_padding_indices = np.where(list(unequal_paddings))[0]
        if len(unequal_padding_indices) > 0:
            # indices to update kernel size
            offset = len(div_shape) - len(padding)
            skip_indices = list(map(lambda x: x + offset, unequal_padding_indices))
            indices = _generate_indices(div_shape, skip_dim_indices=skip_indices)
            # updated kernel size accounting for maximum padding
            new_kernel_size = list(kernel_size)
            for j in unequal_padding_indices:
                new_kernel_size[j] = kernel_size[j] - padding[j][1] + padding[j][0]

            for idx in indices:
                for j in unequal_padding_indices:
                    idx[j + offset] = -1
                div_by = div_by.at[tuple(idx)].set(np.prod(new_kernel_size))

        y = y / div_by
    else:
        div_shape = list(inputs.shape)
        div_shape[num_batch_dims] = 1
        div_shape = tuple(div_shape)
        if len(div_shape) - 2 == len(kernel_size):
            div_shape = (1,) + div_shape[1:]
        y = y / pool(
            jnp.ones(div_shape, y.dtype),
            jnp.array(0.0, y.dtype),
            jax.lax.add,
            kernel_size,
            strides,
            padding,
        )
    return y.astype(inputs.dtype)


# helper function to generate all indices to iterate through ndarray
def _generate_indices(dims, skip_dim_indices=[]):
    res = []

    def _helper(curr_dim_idx, sofar):
        if curr_dim_idx in skip_dim_indices:
            _helper(curr_dim_idx + 1, sofar[:])
            return
        if curr_dim_idx >= len(dims):
            res.append(sofar)
            return
        for i in range(dims[curr_dim_idx]):
            sofar[curr_dim_idx] = i
            _helper(curr_dim_idx + 1, sofar[:])

    _helper(0, [0 for _ in dims])
    return res


# aten.sym_numel
# aten.reciprocal
@register_aten(
    [
        "aten::reciprocal",
        "aten::reciprocal_",
        "aten::reciprocal.out",
    ],
    operation_type="arithmetic",
)
def _aten_reciprocal(a):
    res = jnp.reciprocal(a)
    return res


# aten.select_scatter
def _aten_select_scatter(input, src, dim, index):
    input_indexes = []
    if dim < 0:
        dim += len(input.shape)
    for x in range(len(input.shape)):
        if x == dim:
            input_indexes.append(index)
        else:
            input_indexes.append(slice(None, None, None))
    return input.at[tuple(input_indexes)].set(src)


@register_aten(
    [
        "aten::scatter.src",
        "aten::scatter.src_out",
        "aten::scatter.reduce",
        "aten::scatter.reduce_out",
    ],
    operation_type="indexing",
    static_argnums=(1,),
    static_argnames=("reduce",),
    uses_preprocessing=True,
)
def _aten_scatter_src(input, dim, index, src, out=None, reduce=None):
    def jax_scatter(input, dim, index, src, out=None, reduce=None):
        input_index, source_indexes = _scatter_index(dim, index)
        if reduce == "add":
            return input.at[input_index].add(src[source_indexes])
        elif reduce == "multiply":
            return input.at[input_index].multiply(src[source_indexes])
        return input.at[input_index].set(src[source_indexes])

    # index OOB check
    size = input.shape[dim]
    if torch.any(index >= size) or torch.any(index < -size):
        raise RuntimeError(
            f"index {torch.max(index.abs()).item()} is out of bounds for dimension {dim} with size {size}"
        )

    # handle out and non-out variants
    out_kwargs = {}
    if out is not None:
        out_kwargs["out"] = out
    if reduce is not None:
        if reduce not in ("add", "multiply"):
            raise RuntimeError(f"reduce argument must be either add or multiply. Got {reduce}")
        out_kwargs["reduce"] = reduce

    return jax_scatter, (input, dim, index, src), out_kwargs


@register_aten(
    [
        "aten::scatter.value",
        "aten::scatter.value_out",
        "aten::scatter.value_reduce",
        "aten::scatter.value_reduce_out",
    ],
    operation_type="indexing",
    static_argnums=(1,),
    static_argnames=("reduce",),
    uses_preprocessing=True,
)
def _aten_scatter(input, dim, index, src, out=None, reduce=None):
    def jax_scatter_value(input, dim, index, src, out=None, reduce=None):
        input_index, _ = _scatter_index(dim, index)
        if reduce == "add":
            return input.at[input_index].add(src)
        elif reduce == "multiply":
            return input.at[input_index].multiply(src)
        return input.at[input_index].set(src)

    # index OOB check
    size = input.shape[dim]
    if torch.any(index >= size) or torch.any(index < -size):
        raise RuntimeError(
            f"index {torch.max(index.abs()).item()} is out of bounds for dimension {dim} with size {size}"
        )

    # handle out and non-out variants
    out_kwargs = {}
    if out is not None:
        out_kwargs["out"] = out
    if reduce is not None:
        if reduce not in ("add", "multiply"):
            raise RuntimeError(f"reduce argument must be either add or multiply. Got {reduce}")
        out_kwargs["reduce"] = reduce

    return jax_scatter_value, (input, dim, index, src), out_kwargs


# aten.acosh
def _aten_acosh(self):
    return jnp.arccosh(self)


# aten.avg_pool2d_backward
# aten.col2im
# aten.avg_pool3d
# aten.round
def _aten_round(input, decimals=0):
    return jnp.round(input, decimals)


# aten.max - unary version (no dim)
@register_aten(
    [
        "aten::max",
        "aten::max.unary_out",
    ],
    operation_type="reduction",
)
def _aten_max_unary(self, **kwargs):
    return _with_reduction_scalar(jnp.max, self, None, False)


# aten.max.dim - version with dimension
# Schema: aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
@register_aten(
    ["aten::max.dim", "aten::max.dim_max"],
    operation_type="reduction",
    static_argnums=(1, 2),
    output_params=("max", "max_values"),
)
def _aten_max_dim(self, dim, keepdim=False, **kwargs):
    # When called from PyTorch dispatcher, keepdim comes as positional arg
    return _with_reduction_scalar(jnp.max, self, dim, keepdim), _with_reduction_scalar(
        jnp.argmax, self, dim, keepdim
    )


# aten.maximum
def _aten_maximum(self, other):
    return jnp.maximum(self, other)


# aten.abs
@register_aten(
    [
        "aten::abs",
        "aten::abs.out",
    ],
    operation_type="arithmetic",
)
def _aten_abs(self):
    return jnp.abs(self)


@register_aten(
    [
        "aten::amax",
        "aten::amax.out",
    ],
    operation_type="reduction",
    static_argnums=(1, 2),
    uses_preprocessing=True,
)
def _aten_amax(self, dim=None, keepdim=False, out=None):
    if out is not None and self.dtype != out.dtype:
        # amax CPU requires self and out to have the same dtype
        raise TypeError(
            f"Expected the dtype for input and out to match, but got {self.dtype} for input's dtype and {out.dtype} for out's dtype."
        )

    def _amax_fn(self, dim=None, keepdim=False, out=None):
        return _with_reduction_scalar(jnp.amax, self, dim, keepdim)

    return _amax_fn, (self, dim, keepdim), {"out": out}


def _with_reduction_scalar(jax_func, self, dim, keepdim):
    expanded = False
    if self.ndim == 0:
        # for self of rank 0:
        # torch.any(x, 0), torch.any(x, -1) works;
        # torch.any(x, 1) throws out of bounds, so it's
        # behavior is the same as a jnp array of rank 1
        expanded = True
        self = jnp.expand_dims(self, 0)
    res = jax_func(self, axis=dim, keepdims=keepdim)
    if expanded:
        res = res.squeeze()
    return res


# aten.any
@register_aten(
    ["aten::any", "aten::any.out", "aten::any.all_out"],
    operation_type="reduction",
    static_argnums=(1, 2, 3),
)
def _aten_any(self, dim=None, keepdim=False, dtype=None):
    # dtype parameter is passed by PyTorch but not used by JAX's any
    return _with_reduction_scalar(jnp.any, self, dim, keepdim)


# aten.arange
@register_aten(
    ["aten::arange.start_out"],
    static_argnums=(0, 1, 2),
    static_argnames=("__out_dtype",),
)
def _aten_arange_start_out(
    start,
    end,
    step=1,
    *,
    out=None,
    __out_dtype: torch.dtype | None = None,
):
    # __out_dtype is set by the kernel to the JAX/NumPy execution dtype.
    assert __out_dtype is not None, "__out_dtype must be provided by the kernel"

    # Normalize and validate step (Scalar can be bool in ATen schema)
    if isinstance(step, bool):
        step = int(step)
    if step == 0:
        raise RuntimeError("step must be nonzero")

    return jnp.arange(start, end, step, dtype=__out_dtype)


# aten.argmax
def _aten_argmax(self, dim=None, keepdim=False):
    return _with_reduction_scalar(jnp.argmax, self, dim, keepdim)


def _strided_index(sizes, strides, storage_offset=None):
    ind = jnp.zeros(sizes, dtype=jnp.int32)

    for i, (size, stride) in enumerate(zip(sizes, strides, strict=False)):
        result_shape = (1,) * i + (size,) + (1,) * (len(sizes) - i - 1)
        indexes = (jnp.arange(size) * stride).reshape(result_shape)
        ind += indexes

    if storage_offset is not None:
        ind += storage_offset
    return ind


# aten.as_strided


def _aten_as_strided_scatter(x, src, sizes, strides, storage_offset):
    ind = _strided_index(sizes, strides, storage_offset)
    flattened = jnp.ravel(x)
    modified = flattened.at[ind].set(src)
    return modified.reshape(x.shape)


# aten.atan2
def _aten_atan2(input, other):
    return jnp.arctan2(input, other)


# aten.bitwise_not
@register_aten(
    ["aten::bitwise_not", "aten::bitwise_not.out", "aten::bitwise_not.Tensor_out"],
)
def _aten_bitwise_not(self):
    return ~self


# aten.bitwise_left_shift
@register_aten(
    [
        "aten::bitwise_left_shift.Tensor",
        "aten::bitwise_left_shift.Tensor_out",
        "aten::bitwise_left_shift.Tensor_Scalar",
        "aten::bitwise_left_shift.Tensor_Scalar_out",
        "aten::bitwise_left_shift.Scalar_Tensor",
        "aten::bitwise_left_shift.Scalar_Tensor_out",
    ],
)
def _aten_bitwise_left_shift(input, other):
    """Refer to _aten_bitwise_right_shift docstring for more details."""
    if jnp.isscalar(other):
        # Do NOT do type promotion when `other` is scalar tensor
        input_dtype = input.dtype
        input = input.astype(np.int32)
        result = jnp.left_shift(input, other)
        return result.astype(input_dtype)
    else:
        # Allow type promotion when `other` is scalar tensor
        return jnp.left_shift(input, other)


# aten.bitwise_right_shift
@register_aten(
    [
        "aten::bitwise_right_shift.Tensor",
        "aten::bitwise_right_shift.Tensor_out",
        "aten::bitwise_right_shift.Tensor_Scalar",
        "aten::bitwise_right_shift.Tensor_Scalar_out",
        "aten::bitwise_right_shift.Scalar_Tensor",
        "aten::bitwise_right_shift.Scalar_Tensor_out",
    ],
)
def _aten_bitwise_right_shift(input, other):
    """
    Pytorch bitwise_right_shift does not promote dtype for scalar tensor at all,
    >> torch.bitwise_right_shift(torch.tensor([-2, -7, 31], dtype=torch.int8), torch.tensor(2)).dtype
    torch.int8
    >> torch.bitwise_right_shift(torch.tensor([-2, -7, 31], dtype=torch.int8), torch.tensor(2, dtype=torch.int32)).dtype
    torch.int8

    but Jax right_shift will promote dtype if `other` is not weakly typed.
    >> jnp.right_shift(jnp.array([-2, -7, -13], dtype=np.int8), jnp.array(2, dtype=np.int32)).dtype
    dtype('int32')
    For weakly typed `other`, output type remains the same as `input` dtype
    >> jnp.right_shift(jnp.array([-2, -7, -13], dtype=np.int8), jnp.array(2)).dtype
    dtype('int8')

    However, PyTorch tensors are always strongly typed and when we convert them to Jax arrays,
    they will also be strong typed, therefore type promotion will happen and
    the generated HLO will give a different output type than the one we get from meta tensor.
    Also, if `other` is negative or is greater or equal to the number of bits in `input, the behavior is undefined.
    """
    if jnp.isscalar(other):
        # Do NOT do type promotion when `other` is scalar tensor
        input_dtype = input.dtype
        input = input.astype(np.int32)
        result = jnp.right_shift(input, other)
        return result.astype(input_dtype)
    else:
        # Allow type promotion when `other` is scalar tensor
        return jnp.right_shift(input, other)


# aten.bitwise_and
@register_aten(
    ["aten::bitwise_and", "aten::bitwise_and.out", "aten::bitwise_and.Tensor_out"],
)
def _aten_bitwise_and(self, other):
    return self & other


# aten.bitwise_or
@register_aten(
    ["aten::bitwise_or", "aten::bitwise_or.out", "aten::bitwise_or.Tensor_out"],
)
def _aten_bitwise_or(self, other):
    return self | other


# aten.bitwise_xor
@register_aten(
    ["aten::bitwise_xor", "aten::bitwise_xor.out", "aten::bitwise_xor.Tensor_out"],
)
def _aten_bitwise_xor(self, other):
    return self ^ other


# aten.broadcast_tensors
def _aten_broadcast_tensors(*tensors):
    def _get_broadcast_shape(shapes):
        """
        Determines the output shape by broadcasting all input shapes.

        Args:
          shapes: A list of tuples representing the shapes of the input tensors.

        Returns:
          A tuple representing the broadcasted output shape.
        """

        # Find the maximum number of dimensions among all input tensors
        max_dims = max(len(shape) for shape in shapes)
        # Pad shorter shapes with 1s on the left to match the maximum number of dimensions
        padded_shapes = [(1,) * (max_dims - len(shape)) + shape for shape in shapes]

        # Initialize the output shape with 1s
        output_shape = [1] * max_dims
        # Iterate through each dimension and apply broadcasting rules
        for dim in range(max_dims):
            dim_sizes = [shape[dim] for shape in padded_shapes]
            max_size = max(dim_sizes)
            if all(size == 1 or size == max_size for size in dim_sizes):
                output_shape[dim] = max_size
            else:
                raise ValueError("Incompatible shapes for broadcasting")
        return tuple(output_shape)

    def _broadcast_dimensions(input_shape, output_shape):
        """
        Determines the broadcast_dimensions argument for jax.lax.broadcast_in_dim.

        Args:
          input_shape: The shape of the input tensor.
          output_shape: The desired output shape after broadcasting.

        Returns:
          A tuple specifying which dimensions of the input tensor should be broadcasted.
        """

        res = tuple(
            i for i, (in_dim, out_dim) in enumerate(zip(input_shape, output_shape, strict=False))
        )
        return res

    # clean some function's previous wrap
    if len(tensors) == 1 and len(tensors[0]) >= 1 and isinstance(tensors[0][0], jax.Array):
        tensors = tensors[0]

    # Get the shapes of all input tensors
    shapes = [t.shape for t in tensors]
    # Find the output shape by broadcasting all input shapes
    output_shape = _get_broadcast_shape(shapes)
    # Broadcast each tensor to the output shape
    broadcasted_tensors = [
        jax.lax.broadcast_in_dim(t, output_shape, _broadcast_dimensions(t.shape, output_shape))
        for t in tensors
    ]

    return broadcasted_tensors


# aten.broadcast_to
def _aten_broadcast_to(input, shape):
    return jnp.broadcast_to(input, shape)


# aten.clamp
@register_aten(
    ["aten::clamp", "aten::clamp.out", "aten::clamp.Tensor_out"],
)
def _aten_clamp(self, min=None, max=None):
    return jnp.clip(self, min, max)


@register_aten(
    ["aten::clamp_min", "aten::clamp_min.out", "aten::clamp_min.Tensor_out"],
)
def _aten_clamp_min(input, min):
    return jnp.clip(input, min=min)


@register_aten(
    ["aten::clamp_max", "aten::clamp_max.out", "aten::clamp_max.Tensor_out"],
)
def _aten_clamp_max(input, max):
    return jnp.clip(input, max=max)


# aten.constant_pad_nd
def _aten_constant_pad_nd(input, padding, value=0):
    # NOTE: Torch padding is flat and reversed: (1, 1, 2, 2)
    #  means last dim get padded 1 in front and 1 in back;
    #  and second last dim get padded 2 in front and 2 in back.
    # Jax padding tuple of 3-tuple: the same padding is
    # [(0, 0, 0), ..., (2,2,0), (1,1,0)], where the last dimension
    # is the amount of padding added between any two elements in each dimension
    m = len(padding)
    rev_padding = [(padding[i - 1], padding[i], 0) for i in range(m - 1, 0, -2)]
    pad_dim = tuple(([(0, 0, 0)] * (len(input.shape) - m // 2)) + rev_padding)
    value_casted = jax.numpy.array(value, dtype=input.dtype)
    return jax.lax.pad(input, padding_value=value_casted, padding_config=pad_dim)


# aten.convolution_backward
def _aten_lift_fresh_copy(x):
    return jnp.copy(x)


def _aten_cdist_forward(x1, x2, p, compute_mode=""):
    # x1 is B x P x M
    # x2 is B x Q x M
    # res is B x P x Q
    x1 = jnp.expand_dims(x1, len(x1.shape) - 1)
    x2 = jnp.expand_dims(x2, len(x2.shape) - 2)
    return jnp.linalg.norm(x1 - x2, ord=p, axis=-1)


def _aten__pdist_forward(x, p=2):
    pairwise_dists = _aten_cdist_forward(x, x, p)
    condensed_dists = pairwise_dists[jnp.triu_indices(pairwise_dists.shape[0], k=1)]
    return condensed_dists


def _aten_cholesky_inverse(input, upper=False):
    t = jnp.matrix_transpose(input)
    if "complex" in str(input.dtype):
        t = t.conjugate()
    return jnp.linalg.inv(input @ t)


# aten.cos
def _aten_cos(input):
    return jnp.cos(input)


# aten.cosh
def _aten_cosh(input):
    return jnp.cosh(input)


def _aten_diag(input, diagonal=0):
    return jnp.diag(input, diagonal)


# aten.diagonal


def diag_indices_with_offset(input_shape, offset, dim1=0, dim2=1):
    input_len = len(input_shape)
    if dim1 == dim2 or not (0 <= dim1 < input_len and 0 <= dim2 < input_len):
        raise ValueError(
            "dim1 and dim2 must be different and in range [0, " + str(input_len - 1) + "]"
        )

    size1, size2 = input_shape[dim1], input_shape[dim2]
    if offset >= 0:
        indices1 = jnp.arange(min(size1, size2 - offset))
        indices2 = jnp.arange(offset, offset + len(indices1))
    else:
        indices2 = jnp.arange(min(size1 + offset, size2))
        indices1 = jnp.arange(-offset, -offset + len(indices2))
    return [indices1, indices2]


def _aten_diagonal_scatter(input, src, offset=0, dim1=0, dim2=1):
    indexes = diag_indices_with_offset(input.shape, offset, dim1, dim2)

    if input.ndim == 2:
        return input.at[tuple(indexes)].set(src)
    else:
        # src has the same shape as the output of
        # jnp.diagonal(input, offset, dim1, dim2).
        # Last dimension always contains the diagonal elements,
        # while the preceding dimensions represent the "slices"
        # from which these diagonals are extracted. Thus,
        # we alter input axes to match this assumption, write src
        # and then move the axes back to the original state.
        input = jnp.moveaxis(input, (dim1, dim2), (-2, -1))
        multi_indexes = [slice(None)] * (input.ndim - 2) + indexes
        input = input.at[tuple(multi_indexes)].set(src)
        return jnp.moveaxis(input, (-2, -1), (dim1, dim2))


# aten.diagflat
def _aten_diagflat(input, offset=0):
    return jnp.diagflat(jnp.array(input), offset)


# aten.eq
def _aten_eq(input1, input2):
    return input1 == input2


# aten.equal
def _aten_equal(input, other):
    # Note: jnp.array_equal returns a JAX scalar boolean array, not a Python bool
    # The conversion to Python bool happens outside of JAX tracing
    return jnp.array_equal(input, other)


@register_aten(["aten::erf", "aten::erf.out"])
def _aten_erf(x):
    return jax.lax.erf(x)


@register_aten(["aten::erfinv", "aten::erfinv.out"])
def _aten_erfinv(input):
    return jax.lax.erf_inv(input)


# aten.exp
def _aten_exp(input):
    res = jnp.exp(input)
    new_dtype = mappings.t2j_dtype(torch.get_default_dtype())
    if input.dtype == jax.numpy.int64:
        res = res.astype(new_dtype)
    return res


# aten.expm1
def _aten_expm1(input):
    res = jnp.expm1(input)
    new_dtype = mappings.t2j_dtype(torch.get_default_dtype())
    if input.dtype == jax.numpy.int64:
        res = res.astype(new_dtype)
    return res


# aten.exp2
def _aten_exp2(input):
    res = jnp.exp2(input)
    new_dtype = mappings.t2j_dtype(torch.get_default_dtype())
    if input.dtype == jax.numpy.int64:
        res = res.astype(new_dtype)
    return res


# aten.fill_
# Register explicit overloads to match PyTorch dispatcher
# - Scalar overload: value is a Python scalar (static during compilation)
# - Tensor overload: value is a 0-dim tensor
@register_aten(["aten::fill_.Scalar", "aten::fill_.Tensor"], static_argnums=(1,))
def _aten_fill(x, value, dtype=None, pin_memory=None, memory_format=None, device=None):
    if isinstance(value, torch.Tensor):
        if value.dim() != 0:
            raise RuntimeError(
                f"fill_ only supports 0-dimension value tensor but got tensor with {value.dim()} dimensions."
            )
        # Convert 0-d tensor to a Python scalar for JAX
        value = value.item()

    if dtype is None:
        dtype = x.dtype
    else:
        dtype = convert_dtype_with_default(dtype)
    return jnp.full(x.shape, value, dtype)


# aten.flip
@register_aten(["aten::flip"], static_argnums=(1,))
def _aten_flip(input, dims):
    if dims is not None:
        return jnp.flip(input, tuple(dims))
    else:
        return jnp.flip(input)


@register_aten(["aten::zero_"])
def aten_zero_(self):
    return jnp.zeros_like(self)


# aten.floor
def _aten_floor(input):
    return jnp.floor(input).astype(input.dtype)


# aten.fmax
def _aten_fmax(input, other):
    return jnp.fmax(input, other)


# aten.fmin
def _aten_fmin(input, other):
    return jnp.fmin(input, other)


# aten.fmod
def _aten_fmod(input, other):
    return input - other * _aten_div(input, other, "trunc")


# aten.frexp
def _aten_frexp(input):
    return jnp.frexp(input)


# aten.ge
def _aten_ge(self, other):
    return self >= other


def _aten_glu(x, dim=-1):
    return jax.nn.glu(x, dim)


# aten.hardtanh
def _aten_hardtanh(input, min_val=-1, max_val=1, inplace=False):
    if input.dtype == np.int64 and isinstance(max_val, float) and isinstance(min_val, float):
        min_val = int(min_val)
        max_val = int(max_val)
    return jnp.clip(input, min_val, max_val)


def _aten_hypot(input, other):
    return jnp.hypot(input, other)


def _aten_digamma(input, *, out=None):
    res = jax.scipy.special.digamma(input).astype(jnp.float32)
    # replace indices where input == 0 with -inf in res
    return jnp.where(jnp.equal(input, jnp.zeros(input.shape)), -jnp.inf, res)


def _aten_igamma(input, other):
    return jax.scipy.special.gammainc(input, other)


def _aten_lgamma(input, *, out=None):
    return jax.scipy.special.gammaln(input).astype(jnp.float32)


def _aten_mvlgamma(input, p, *, out=None):
    input = input.astype(mappings.t2j_dtype(torch.get_default_dtype()))
    return jax.scipy.special.multigammaln(input, p)


def _aten_linalg_eig(A):
    return jnp.linalg.eig(A)


def _aten_linalg_eigh(A, UPLO="L"):
    return jnp.linalg.eigh(A, UPLO)


def _aten_linalg_lstsq(A, B, rcond=None, driver="gelsy"):
    input_dtype = A.dtype

    m = A.shape[-2]
    n = A.shape[-1]

    is_batched = A.ndim > 2

    if is_batched:
        batch_shape = jnp.broadcast_shapes(A.shape[:-2], B.shape[:-2])
        batch_size = int(np.prod(batch_shape))
        A_reshaped = A.reshape((batch_size,) + A.shape[-2:])
        B_reshaped = B.reshape((batch_size,) + B.shape[-2:])

        X, residuals, rank, singular_values = jax.vmap(jnp.linalg.lstsq, in_axes=(0, 0))(
            A_reshaped, B_reshaped, rcond=rcond
        )

        X = X.reshape(batch_shape + X.shape[-2:])

        if driver in ["gelsd", "gelsy", "gelss"]:
            rank = rank.reshape(batch_shape)
        else:
            rank = jnp.array([], dtype=jnp.int64)

        full_rank = jnp.all(rank == n)
        if driver == "gelsy" or m <= n or (not full_rank):
            residuals = jnp.array([], dtype=input_dtype)
        else:
            residuals = residuals.reshape(batch_shape + residuals.shape[-1:])

        if driver in ["gelsd", "gelss"]:
            singular_values = singular_values.reshape(batch_shape + singular_values.shape[-1:])
        else:
            singular_values = jnp.array([], dtype=input_dtype)

    else:
        X, residuals, rank, singular_values = jnp.linalg.lstsq(A, B, rcond=rcond)

        if driver not in ["gelsd", "gelsy", "gelss"]:
            rank = jnp.array([], dtype=jnp.int64)

        rank_value = None
        if rank.size > 0:
            rank_value = int(rank.item())
            rank = jnp.array(rank_value, dtype=jnp.int64)

        # When driver is ‘gels’, assume that A is full-rank.
        full_rank = driver == "gels" or rank_value == n
        if driver == "gelsy" or m <= n or (not full_rank):
            residuals = jnp.array([], dtype=input_dtype)

        if driver not in ["gelsd", "gelss"]:
            singular_values = jnp.array([], dtype=input_dtype)

    return X, residuals, rank, singular_values


def _aten_linalg_ldl_factor_ex(A, hermitian=False, check_errors=False):
    # TODO: Replace with native LDL when available:
    # https://github.com/jax-ml/jax/issues/12779
    # TODO: Not tested for complex inputs. Does not support hermitian=True
    pivots = jnp.broadcast_to(jnp.arange(1, A.shape[-1] + 1, dtype=jnp.int32), A.shape[:-1])
    info = jnp.zeros(A.shape[:-2], jnp.int32)
    C = jnp.linalg.cholesky(A)
    if C.size == 0:
        return C, pivots, info

    # Fill diagonals of stacked matrices
    @functools.partial(jnp.vectorize, signature="(k,k),(k,k)->(k,k)")
    def fill_diagonal_batch(x, y):
        return jnp.fill_diagonal(x, jnp.diag(y), inplace=False)

    D = C * jnp.eye(C.shape[-1], dtype=A.dtype)
    LD = C @ jnp.linalg.inv(D)
    LD = fill_diagonal_batch(LD, D * D)
    return LD, pivots, info


def _aten_linalg_lu(A, pivot=True, out=None):
    dtype = A.dtype

    *_, m, n = A.shape
    k = jnp.minimum(m, n)

    lu, _, permutation = jax.lax.linalg.lu(A)

    L = jnp.tril(lu[..., :, :k], k=-1)
    eye_L = jnp.eye(m, k, dtype=dtype)
    L = L + eye_L

    U = jnp.triu(lu[..., :k, :])

    def perm_to_P(perm):
        m = perm.shape[-1]
        P = jnp.eye(m, dtype=dtype)[perm].T
        return P

    if permutation.ndim > 1:
        num_batch_dims = permutation.ndim - 1
        for _ in range(num_batch_dims):
            perm_to_P = jax.vmap(perm_to_P, in_axes=0)

    P = perm_to_P(permutation)

    return P, L, U


def _aten_linalg_lu_factor_ex(A, pivot=True, check_errors=False):
    lu, pivots, _ = jax.lax.linalg.lu(A)
    # PT pivots vector is 1-indexed
    pivots = pivots + 1
    info = jnp.zeros(A.shape[:-2], jnp.int32)
    return lu, pivots, info


def _aten_linalg_lu_solve(LU, pivots, B, left=True, adjoint=False):
    # JAX pivots are offset by 1 compared to torch
    pivots = pivots - 1
    if not left:
        # XA = B is same as A'X = B'
        trans = 0 if adjoint else 2
        x = jax.scipy.linalg.lu_solve((LU, pivots), jnp.matrix_transpose(B), trans)
        x = jnp.matrix_transpose(x)
    else:
        trans = 2 if adjoint else 0
        x = jax.scipy.linalg.lu_solve((LU, pivots), B, trans)
    return x


def _aten_gcd(input, other):
    return jnp.gcd(input, other)


# aten.lcm
def _aten_lcm(input, other):
    return jnp.lcm(input, other)


@register_aten(
    "aten::isinf",
)
def _aten_isinf(input):
    return jnp.isinf(input)


@register_aten("aten::isneginf")
def _aten_isneginf(x):
    return jnp.isinf(x) & jnp.signbit(x)


def _aten_isnan(input):
    return jnp.isnan(input)


def _aten_le(self, other):
    return self <= other


# aten.leaky_relu
def _aten_leaky_relu(x, negative_slope=0.01):
    return jax.nn.leaky_relu(x, negative_slope)


# aten.log
def _aten_log(x):
    return jnp.log(x)


# aten.log10
def _aten_log10(x):
    return jnp.log10(x)


# aten.log1p
def _aten_log1p(x):
    return jnp.log1p(x)


# aten.log2
def _aten_log2(x):
    return jnp.log2(x)


# aten.logical_and
@register_aten(
    ["aten::logical_and", "aten::logical_and.out"],
    operation_type="comparison",
    static_argnames=("__out_dtype",),
)
def _aten_logical_and(self, other, __out_dtype=None):
    result = jnp.logical_and(self, other)
    if __out_dtype is not None:
        result = result.astype(__out_dtype)
    return result


# aten.logical_or
@register_aten(
    ["aten::logical_or", "aten::logical_or.out"],
    operation_type="comparison",
    static_argnames=("__out_dtype",),
)
def _aten_logical_or(self, other, __out_dtype=None):
    result = jnp.logical_or(self, other)
    if __out_dtype is not None:
        result = result.astype(__out_dtype)
    return result


@register_aten(
    ["aten::logical_not", "aten::logical_not.out"],
    operation_type="comparison",
    static_argnames=("__out_dtype",),
)
def _aten_logical_not(self, __out_dtype=None):
    result = jnp.logical_not(self)
    if __out_dtype is not None:
        result = result.astype(__out_dtype)
    return result


# aten.log_softmax
def _aten_log_softmax(self, dim=-1, dtype=None, half_to_float=False):
    if self.shape == ():
        return jnp.astype(0.0, self.dtype)

    # Convert input to specified dtype if provided
    input_data = self
    if dtype is not None:
        # Handle torch.dtype when passed as static argument
        import torch

        if isinstance(dtype, torch.dtype):
            from torch_neuronx.kernels.type_converter import TypeConverter

            jax_dtype = TypeConverter.torch_to_jax(dtype)
            input_data = self.astype(jax_dtype)
        else:
            input_data = self.astype(dtype)

    result = jax.nn.log_softmax(input_data, dim)
    return result


# aten.logaddexp
def _aten_logaddexp(self, other):
    return jnp.logaddexp(self, other)


# aten.logaddexp2
def _aten_logaddexp2(self, other):
    return jnp.logaddexp2(self, other)


# aten.logcumsumexp
def _aten_logcumsumexp(self, dim=None):
    if self.shape == ():
        return self
    return jax.lax.cumlogsumexp(self, axis=dim)


# aten.max_pool3d_backward
# aten.logical_xor
@register_aten(
    ["aten::logical_xor", "aten::logical_xor.out"],
    operation_type="comparison",
    static_argnames=("__out_dtype",),
)
def _aten_logical_xor(self, other, __out_dtype=None):
    result = jnp.logical_xor(self, other)
    if __out_dtype is not None:
        result = result.astype(__out_dtype)
    return result


# aten.max_pool2d_with_indices_backward
# aten.native_dropout
# aten.native_group_norm_backward
# aten.neg
def _aten_neg(x):
    return -1 * x


def _aten_nextafter(input, other, *, out=None):
    return jnp.nextafter(input, other)


# aten.prod
def _aten_prod(input, dim=None, keepdim=False, *, dtype=None):
    result = _with_reduction_scalar(jnp.prod, input, dim, keepdim)
    if dtype is not None:
        result = result.astype(mappings.t2j_dtype(dtype))
    return result


def _aten_put(self, index, source, accumulate=False):
    expanded = False
    res = None

    if self.ndim == 0:
        expanded = True
        self = jnp.expand_dims(self, 0)

    if accumulate:
        tmp = jnp.zeros(self.shape)
        tmp = jnp.put(tmp, index, source, inplace=False)
        res = jnp.add(self, tmp).astype(self.dtype)
    else:
        res = jnp.put(self, index, source, inplace=False)

    if expanded:
        res = res.squeeze()

    return res


# aten.reflection_pad3d


@register_aten(
    [
        "aten::remainder.Tensor",
        "aten::remainder.Tensor_out",
        "aten::remainder.Scalar",
        "aten::remainder.Scalar_out",
    ]
)
def _aten_remainder(inputs, other):
    return inputs % other


@register_aten(
    [
        "aten::repeat",
    ],
    static_argnums=(1,),
)
def _aten_repeat(x, repeats):
    return jnp.tile(x, repeats)


# aten.replication_pad2d
# aten.replication_pad3d
# aten.roll
def _aten_roll(input, shifts, dims=None):
    return jnp.roll(input, shifts, dims)


# aten.slice_scatter
def _aten_slice_scatter(input, src, dim=0, start=None, end=None, step=1):
    input_index = []
    for x in range(len(input.shape)):
        if x == dim:
            input_index.append(slice(start, end, step))
        else:
            input_index.append(slice(None, None, None))
    return input.at[tuple(input_index)].set(src)


# aten.sort
# torch.sort(input, dim=-1, descending=False, stable=False, *, out=None)
def _aten_sort(a, dim=-1, descending=False, stable=False):
    if a.shape == ():
        return (a, jnp.astype(0, "int64"))
    return (
        jnp.sort(a, axis=dim, stable=stable, descending=descending),
        jnp.argsort(a, axis=dim, stable=stable, descending=descending),
    )


# aten.sym_size


# aten.topk
def _aten_topk(input, k, dim=None, largest=True, sorted=True, *, out=None):
    """JAX top-k implementation using jax.lax.top_k for improved efficiency.

    Args:
        input: The input JAX array.
        k: The number of top elements to return.
        dim: The dimension along which to find the top-k. If None, operates on the
          flattened array.
        largest: If True, returns the largest k elements. Otherwise, smallest k.
        sorted: If True, returns the elements in sorted order.

    Returns:
        A tuple (values, indices) containing:
            - values: The top k values.
            - indices: The indices of the top k values in the original array.
    """
    if dim is None:
        # last dim is chosen
        dim = input.ndim - 1

    if dim < 0:
        dim = dim + input.ndim

    if not largest:
        input = -input  # Find top-k of negated input if we want the smallest

    if input.ndim == 0:
        return input, jnp.array(0, dtype=jnp.int64.dtype)

    transpose_shape = None
    if dim != -1 and dim != len(input.shape) - 1:
        transpose_shape = list(range(len(input.shape)))
        transpose_shape[dim], transpose_shape[-1] = (
            transpose_shape[-1],
            transpose_shape[dim],
        )
        input = jnp.transpose(input, transpose_shape)

    values, indices = jax.lax.top_k(input, k)

    if sorted:
        values = jnp.sort(values, descending=True)
        indices = jnp.take_along_axis(
            indices, jnp.argsort(values, axis=-1, descending=True), axis=-1
        )

    if not largest:
        values = -values  # Negate values back if we found smallest

    if transpose_shape is not None:
        values = jnp.transpose(values, transpose_shape)
        indices = jnp.transpose(indices, transpose_shape)

    return values, indices


@register_aten(["aten::tril", "aten::tril.out"], static_argnums=(1,))
def _aten_tril(x, diagonal=0):
    return jnp.tril(x, k=diagonal)


# aten.tril_indices
# tril_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None)
def _aten_tril_indices(
    row, col, offset=0, *, dtype=jnp.int64.dtype, layout=None, device=None, pin_memory=None
):
    a, b = jnp.tril_indices(row, offset, col)
    return jnp.stack((a, b))


# aten.tril_indices
# tril_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None)
def _aten_triu_indices(
    row, col, offset=0, *, dtype=jnp.int64.dtype, layout=None, device=None, pin_memory=None
):
    a, b = jnp.triu_indices(row, offset, col)
    return jnp.stack((a, b))


# aten.unique_dim
#
# NOTE: Like the CUDA and CPU implementations, this implementation always sorts
# the tensor regardless of the `sorted` argument passed to `torch.unique`.
def _aten_unique_dim(input_tensor, dim, sort=True, return_inverse=False, return_counts=False):
    result_tensor_or_tuple = jnp.unique(
        input_tensor,
        return_index=False,
        return_inverse=return_inverse,
        return_counts=return_counts,
        axis=dim,
        equal_nan=False,
    )
    result_list = (
        list(result_tensor_or_tuple)
        if isinstance(result_tensor_or_tuple, tuple)
        else [result_tensor_or_tuple]
    )

    if not return_inverse:
        result_list.insert(1, None)
    elif _jax_version < (0, 4, 31) and dim is not None:
        result_list[1] = result_list[1].flatten()

    if not return_counts:
        result_list.insert(2, None)

    # [result, None,    None]    if return_inverse=False and return_counts=False
    # [result, inverse, None]    if return_inverse=True  and return_counts=False
    # [result, None,    counts]  if return_inverse=False and return_counts=True
    # [result, inverse, counts]  if return_inverse=True  and return_counts=True
    return result_list


# aten._unique
#
# NOTE: Like the CUDA and CPU implementations, this implementation always sorts
# the tensor regardless of the `sorted` argument passed to `torch.unique`.
def _aten_unique(input_tensor, sort=True, return_inverse=False):
    result_tensor_or_tuple = jnp.unique(
        input_tensor,
        return_index=False,
        return_inverse=return_inverse,
        return_counts=False,
        axis=None,
        equal_nan=False,
    )
    if return_inverse:
        return result_tensor_or_tuple
    else:
        return (result_tensor_or_tuple, None)


# aten._unique2
#
# NOTE: Like the CUDA and CPU implementations, this implementation always sorts
# the tensor regardless of the `sorted` argument passed to `torch.unique`.
def _aten_unique2(input_tensor, sort=True, return_inverse=False, return_counts=False):
    return _aten_unique_dim(
        input_tensor=input_tensor,
        dim=None,
        sort=sort,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )


# aten.unique_consecutive
def _aten_unique_consecutive(input_tensor, return_inverse=False, return_counts=None, dim=None):
    # Explanation of computations (shown in 1D for simplicity):
    #
    #   Input                                      [a b b c c c d d d d e e e e e]
    #   Slice dropping final element (input[:-1])    [a b b c c c d d d d e e e e]
    #   Slice dropping first element (input[1:])     [b b c c c d d d d e e e e e]
    #   Boolean != operation on shifted slices       [1 0 1 0 0 1 0 0 0 1 0 0 0 0]
    #   Prepend 1 to represent the first element   [1 1 0 1 0 0 1 0 0 0 1 0 0 0 0]
    #   Filter input by the resulting bool array   [a b   c     d       e        ]
    #   Output                                     [a b c d e]

    if dim is None:
        inverse_shape = input_tensor.shape
        input_tensor = input_tensor.flatten()
        ndim = 1
        dim = 0
    else:
        inverse_shape = input_tensor.shape[dim]
        ndim = input_tensor.ndim
        if dim < 0:
            dim += ndim

    nd_slice_0 = tuple(slice(None, -1) if d == dim else slice(None) for d in range(ndim))
    nd_slice_1 = tuple(slice(1, None) if d == dim else slice(None) for d in range(ndim))

    axes_to_reduce = tuple(d for d in range(ndim) if d != dim)

    does_not_equal_prior = jnp.any(
        input_tensor[nd_slice_0] != input_tensor[nd_slice_1], axis=axes_to_reduce, keepdims=False
    )

    if input_tensor.shape[dim] != 0:
        # Prepend `True` to represent the first element of the input.
        does_not_equal_prior = jnp.insert(does_not_equal_prior, 0, True)

    include_indices = jnp.argwhere(does_not_equal_prior)[:, 0]

    output_tensor = input_tensor[
        tuple(include_indices if d == dim else slice(None) for d in range(ndim))
    ]

    if return_inverse or return_counts:
        counts = jnp.append(include_indices[1:], input_tensor.shape[dim]) - include_indices[:]

        inverse = (
            jnp.reshape(jnp.repeat(jnp.arange(len(counts)), counts), inverse_shape)
            if return_inverse
            else None
        )

        return output_tensor, inverse, counts

    return output_tensor, None, None


# NOTE: skip aten.upsample_nearest2d and aten.upsample_bilinear2d
# despite those being core aten ops, they also have decompositions.
# here we are using torch decompositions.


# aten.where
# NOTE: 1 argument case is not jit compilable. So, we
# don't register aten::where and let it fallback to cpu.
@register_aten(
    [
        "aten::where.self",
        "aten::where.self_out",
        "aten::where.ScalarSelf",
        "aten::where.ScalarOther",
        "aten::where.Scalar",
    ],
)
def _aten_where(condition, x=None, y=None):
    return jnp.where(condition, x, y)


@register_aten(
    [
        "aten::masked_fill.Tensor",
        "aten::masked_fill_.Tensor",
        "aten::masked_fill.Scalar",
        "aten::masked_fill_.Scalar",
    ],
)
def _aten_masked_fill(self, mask, value):
    value = value.astype(self.dtype)
    return jnp.where(mask, value, self)


# Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False
def _aten_var_mean_correction(tensor, dim=None, correction=1, keepdim=False):
    # The internal API technically has a default `correction` argument of `None`,
    # but the public API has a default argument of 1. Therefore, we simply set our
    # default argument to 1. However, since the argument is officially supposed to
    # be nullable, we still need to check for `None` per the API contract.
    if correction is None:
        correction = 1
    mean = jnp.mean(tensor, axis=dim, keepdims=keepdim)
    # TODO: Pass in the `mean=mean` argument once `jax.numpy.var` supports it.
    var = jnp.var(tensor, axis=dim, ddof=correction, keepdims=keepdim)
    return var, mean


@register_aten(
    [
        "aten::scalar_tensor",
    ],
    static_argnames=("dtype",),
    uses_preprocessing=True,
)
def _aten_scalar_tensor(s, dtype=None, layout=None, device=None, pin_memory=None):
    """
    Convert scalar value to a tensor. Currently we are using torch.tensor and argument processor is converting scalar to tensor.
    However torch.tensor API and torch.scalar_tensor API will convert the types differently.
    torch.scalar_tensor will convert scalar value to torch.float32 for both int and bool
    torch.tensor will infer type from input type and convert int to torch.int64 and convert bool to torch.bool

    Also, type conversion is different between torch and jax:
    for int, jnp.array will convert to int32, torch.scalar_tensor will convert to float32
    for bool, jnp.array keeps bool, torch.scalar_tensor will convert to float32

    Preprocessing is needed for this function because argument processor will convert the input tensor using torch.tensor
    which uses a different dtype than torch.scalar_tensor
    """
    if dtype is None:
        if type(s) == int or type(s) == bool:
            dtype = torch.float32

    def _scalar_tensor_fn(s, dtype=dtype, **kwargs):
        jdtype = convert_dtype_with_default(dtype)
        return jnp.array(s, dtype=jdtype)

    return _scalar_tensor_fn, (s,), {"dtype": dtype}


def max_pool2d_with_indices_backward_custom(
    grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices
):
    """
    Approximates the gradient calculation of PyTorch's max_pool2d_with_indices_backward.

    Args:
        grad_output: The gradient tensor from the preceding layer.
        self: The input tensor on which the original max pooling was performed.
        kernel_size: The size of the pooling window.
        stride: The stride of the pooling window.
        padding: The padding applied during max pooling.
        dilation: The dilation factor for the pooling operation.
        ceil_mode: Whether to use ceil or floor when calculating output shapes.
        indices: The indices of the maximum values, as produced by max_pool2d_with_indices.

    Returns:
        The calculated gradient with respect to the input (grad_input).
    """

    kH, kW = kernel_size
    dH, dW = stride
    padH, padW = padding
    dilH, dilW = dilation

    # Calculate output shape (may need adjustment based on ceil_mode)
    out_shape = jnp.array(self.shape)
    grad_input = jnp.zeros_like(self)

    # Iterate over the flattened input and output tensors
    for i, idx in enumerate(indices.flatten()):
        # Calculate input coordinates corresponding to the maximum value
        out_y, out_x = i // grad_output.shape[3], i % grad_output.shape[3]
        in_y = out_y * dH - padH + out_y * (dilH - 1)
        in_x = out_x * dW - padW + out_x * (dilW - 1)

        # Scatter the gradient to the appropriate input locations (handling potential overlaps)
        for y in range(in_y, in_y + kH):
            for x in range(in_x, in_x + kW):
                if 0 <= y < grad_input.shape[2] and 0 <= x < grad_input.shape[3]:
                    grad_input = grad_input.at[y, x].add(grad_output.flatten()[i])

    return grad_input


def _aten_local_scalar_dense(x):
    return x.item()


def _aten_tensor_split(ary, indices_or_sections, axis=0):
    return jnp.array_split(ary, indices_or_sections, axis)


def _aten_outer(a, b):
    return jnp.outer(a, b)


@register_aten(["aten::ones_like"], static_argnames=("dtype",))
def _aten_ones_like(
    self,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    memory_format=None,
    pin_memory=False,
):
    # If dtype is not specified, use the input tensor's dtype
    if dtype is None:
        jdtype = self.dtype
    else:
        jdtype = convert_dtype_with_default(dtype)

    # Create a tensor of ones with the same shape as the input
    return jnp.ones(self.shape, dtype=jdtype)


def _aten_allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    return jnp.allclose(input, other, rtol, atol, equal_nan)


def _aten_native_batch_norm(
    input, weight, bias, running_mean, running_var, training=False, momentum=0.1, eps=1e-5
):
    if running_mean is None:
        running_mean = jnp.zeros(
            input.shape[1], dtype=input.dtype
        )  # Initialize running mean if None
    if running_var is None:
        running_var = jnp.ones(
            input.shape[1], dtype=input.dtype
        )  # Initialize running variance if None

    if training:
        return _aten__native_batch_norm_legit(
            input, weight, bias, running_mean, running_var, training, momentum, eps
        )
    else:
        return _aten__native_batch_norm_legit_no_training(
            input, weight, bias, running_mean, running_var, momentum, eps
        )


# TODO: not clear what this function should actually do
# https://github.com/pytorch/pytorch/blob/d96c80649f301129219469d8b4353e52edab3b78/aten/src/ATen/native/native_functions.yaml#L7933-L7940
def _aten_lift_fresh(self):
    return self


def _aten_dim(self):
    return len(self.shape)


def _aten_copysign(input, other, *, out=None):
    result = jnp.copysign(input, other)
    # torch.copysign(x, y) returns float32 for integer x and y,
    # regardless of their exact integer dtype, whereas jax.copysign returns
    # float64 when one or both of them is int64.
    if jnp.issubdtype(input.dtype, jnp.integer) and jnp.issubdtype(other.dtype, jnp.integer):
        result = result.astype(jnp.float32)
    return result


def _aten_i0(self):
    return jax.scipy.special.i0(self)


def _aten_i0e(self):
    return jax.scipy.special.i0e(self)


def _aten_special_i1(self):
    return jax.scipy.special.i1(self)


def _aten_special_i1e(self):
    return jax.scipy.special.i1e(self)


def _aten_special_laguerre_polynomial_l(self, n):
    # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L3106-L3134

    @jnp.vectorize
    def vectorized(x, n_i):
        def negative_n(x):
            return jnp.zeros_like(x)

        def zero_n(x):
            return jnp.ones_like(x)

        def one_n(x):
            return jnp.ones_like(x) - x

        def zero_abs(x):
            return jnp.ones_like(x)

        def default(x):
            def f(k, carry):
                p, q = carry
                return (q, ((k * 2 + (jnp.ones_like(x) - x)) * q - k * p) / (k + 1))

            _, q = jax.lax.fori_loop(1, n_i, f, init_val=(1.0, jnp.ones_like(x) - x))
            return q

        return jnp.piecewise(
            x,
            [n_i == 1, n_i == 0, jnp.abs(n_i) == jnp.zeros_like(x), n_i < 0],
            [one_n, zero_n, zero_abs, negative_n, default],
        )

    return vectorized(self, n.astype(jnp.int64))


def _aten_special_modified_bessel_i0(self):
    # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L3182-L3268

    def small(x):
        A = jnp.array(
            [
                -4.41534164647933937950e-18,
                3.33079451882223809783e-17,
                -2.43127984654795469359e-16,
                1.71539128555513303061e-15,
                -1.16853328779934516808e-14,
                7.67618549860493561688e-14,
                -4.85644678311192946090e-13,
                2.95505266312963983461e-12,
                -1.72682629144155570723e-11,
                9.67580903537323691224e-11,
                -5.18979560163526290666e-10,
                2.65982372468238665035e-09,
                -1.30002500998624804212e-08,
                6.04699502254191894932e-08,
                -2.67079385394061173391e-07,
                1.11738753912010371815e-06,
                -4.41673835845875056359e-06,
                1.64484480707288970893e-05,
                -5.75419501008210370398e-05,
                1.88502885095841655729e-04,
                -5.76375574538582365885e-04,
                1.63947561694133579842e-03,
                -4.32430999505057594430e-03,
                1.05464603945949983183e-02,
                -2.37374148058994688156e-02,
                4.93052842396707084878e-02,
                -9.49010970480476444210e-02,
                1.71620901522208775349e-01,
                -3.04682672343198398683e-01,
                6.76795274409476084995e-01,
            ],
            dtype=self.dtype,
        )

        def f(carry, val):
            p, q, a = carry
            p, q = q, a
            return (p, q, ((x / 2.0) - 2.0) * q - p + val), None

        (p, _, a), _ = jax.lax.scan(f, init=(jnp.zeros_like(x), jnp.zeros_like(x), 0), xs=A)

        return jnp.exp(x) * (0.5 * (a - p))

    def default(x):
        B = jnp.array(
            [
                -7.23318048787475395456e-18,
                -4.83050448594418207126e-18,
                4.46562142029675999901e-17,
                3.46122286769746109310e-17,
                -2.82762398051658348494e-16,
                -3.42548561967721913462e-16,
                1.77256013305652638360e-15,
                3.81168066935262242075e-15,
                -9.55484669882830764870e-15,
                -4.15056934728722208663e-14,
                1.54008621752140982691e-14,
                3.85277838274214270114e-13,
                7.18012445138366623367e-13,
                -1.79417853150680611778e-12,
                -1.32158118404477131188e-11,
                -3.14991652796324136454e-11,
                1.18891471078464383424e-11,
                4.94060238822496958910e-10,
                3.39623202570838634515e-09,
                2.26666899049817806459e-08,
                2.04891858946906374183e-07,
                2.89137052083475648297e-06,
                6.88975834691682398426e-05,
                3.36911647825569408990e-03,
                8.04490411014108831608e-01,
            ],
            dtype=self.dtype,
        )

        def f(carry, val):
            p, q, b = carry
            p, q = q, b
            return (p, q, (32.0 / x - 2.0) * q - p + val), None

        (p, _, b), _ = jax.lax.scan(f, init=(jnp.zeros_like(x), jnp.zeros_like(x), 0), xs=B)

        return jnp.exp(x) * (0.5 * (b - p)) / jnp.sqrt(x)

    self = jnp.abs(self)
    return jnp.piecewise(self, [self <= 8], [small, default])


def _aten_special_modified_bessel_i1(self):
    # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L3271-L3364

    def small(x):
        A = jnp.array(
            [
                2.77791411276104639959e-18,
                -2.11142121435816608115e-17,
                1.55363195773620046921e-16,
                -1.10559694773538630805e-15,
                7.60068429473540693410e-15,
                -5.04218550472791168711e-14,
                3.22379336594557470981e-13,
                -1.98397439776494371520e-12,
                1.17361862988909016308e-11,
                -6.66348972350202774223e-11,
                3.62559028155211703701e-10,
                -1.88724975172282928790e-09,
                9.38153738649577178388e-09,
                -4.44505912879632808065e-08,
                2.00329475355213526229e-07,
                -8.56872026469545474066e-07,
                3.47025130813767847674e-06,
                -1.32731636560394358279e-05,
                4.78156510755005422638e-05,
                -1.61760815825896745588e-04,
                5.12285956168575772895e-04,
                -1.51357245063125314899e-03,
                4.15642294431288815669e-03,
                -1.05640848946261981558e-02,
                2.47264490306265168283e-02,
                -5.29459812080949914269e-02,
                1.02643658689847095384e-01,
                -1.76416518357834055153e-01,
                2.52587186443633654823e-01,
            ],
            dtype=self.dtype,
        )

        def f(carry, val):
            p, q, a = carry
            p, q = q, a
            return (p, q, ((jnp.abs(x) / 2.0) - 2.0) * q - p + val), None

        (p, _, a), _ = jax.lax.scan(f, init=(jnp.zeros_like(x), jnp.zeros_like(x), 0), xs=A)

        return jax.lax.cond(
            x < 0,
            lambda: -(0.5 * (a - p) * jnp.abs(x) * jnp.exp(jnp.abs(x))),
            lambda: 0.5 * (a - p) * jnp.abs(x) * jnp.exp(jnp.abs(x)),
        )

    def default(x):
        B = jnp.array(
            [
                7.51729631084210481353e-18,
                4.41434832307170791151e-18,
                -4.65030536848935832153e-17,
                -3.20952592199342395980e-17,
                2.96262899764595013876e-16,
                3.30820231092092828324e-16,
                -1.88035477551078244854e-15,
                -3.81440307243700780478e-15,
                1.04202769841288027642e-14,
                4.27244001671195135429e-14,
                -2.10154184277266431302e-14,
                -4.08355111109219731823e-13,
                -7.19855177624590851209e-13,
                2.03562854414708950722e-12,
                1.41258074366137813316e-11,
                3.25260358301548823856e-11,
                -1.89749581235054123450e-11,
                -5.58974346219658380687e-10,
                -3.83538038596423702205e-09,
                -2.63146884688951950684e-08,
                -2.51223623787020892529e-07,
                -3.88256480887769039346e-06,
                -1.10588938762623716291e-04,
                -9.76109749136146840777e-03,
                7.78576235018280120474e-01,
            ],
            dtype=self.dtype,
        )

        def f(carry, val):
            p, q, b = carry
            p, q = q, b
            return (p, q, (32.0 / jnp.abs(x) - 2.0) * q - p + val), None

        (p, _, b), _ = jax.lax.scan(f, init=(jnp.zeros_like(x), jnp.zeros_like(x), 0), xs=B)

        return jax.lax.cond(
            x < 0,
            lambda: -(jnp.exp(jnp.abs(x)) * (0.5 * (b - p)) / jnp.sqrt(jnp.abs(x))),
            lambda: jnp.exp(jnp.abs(x)) * (0.5 * (b - p)) / jnp.sqrt(jnp.abs(x)),
        )

    return jnp.piecewise(self, [self <= 8], [small, default])


def _aten_special_modified_bessel_k0(self):
    # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L3367-L3441

    def zero(x):
        return jnp.array(jnp.inf, x.dtype)

    def negative(x):
        return jnp.array(jnp.nan, x.dtype)

    def small(x):
        A = jnp.array(
            [
                1.37446543561352307156e-16,
                4.25981614279661018399e-14,
                1.03496952576338420167e-11,
                1.90451637722020886025e-09,
                2.53479107902614945675e-07,
                2.28621210311945178607e-05,
                1.26461541144692592338e-03,
                3.59799365153615016266e-02,
                3.44289899924628486886e-01,
                -5.35327393233902768720e-01,
            ],
            dtype=self.dtype,
        )

        def f(carry, val):
            p, q, a = carry
            p, q = q, a
            return (p, q, (x * x - 2.0) * q - p + val), None

        (p, _, a), _ = jax.lax.scan(f, init=(jnp.zeros_like(x), jnp.zeros_like(x), 0), xs=A)

        return 0.5 * (a - p) - jnp.log(0.5 * x) * _aten_special_modified_bessel_i0(x)

    def default(x):
        B = jnp.array(
            [
                5.30043377268626276149e-18,
                -1.64758043015242134646e-17,
                5.21039150503902756861e-17,
                -1.67823109680541210385e-16,
                5.51205597852431940784e-16,
                -1.84859337734377901440e-15,
                6.34007647740507060557e-15,
                -2.22751332699166985548e-14,
                8.03289077536357521100e-14,
                -2.98009692317273043925e-13,
                1.14034058820847496303e-12,
                -4.51459788337394416547e-12,
                1.85594911495471785253e-11,
                -7.95748924447710747776e-11,
                3.57739728140030116597e-10,
                -1.69753450938905987466e-09,
                8.57403401741422608519e-09,
                -4.66048989768794782956e-08,
                2.76681363944501510342e-07,
                -1.83175552271911948767e-06,
                1.39498137188764993662e-05,
                -1.28495495816278026384e-04,
                1.56988388573005337491e-03,
                -3.14481013119645005427e-02,
                2.44030308206595545468e00,
            ],
            dtype=self.dtype,
        )

        def f(carry, val):
            p, q, b = carry
            p, q = q, b
            return (p, q, (8.0 / x - 2.0) * q - p + val), None

        (p, _, b), _ = jax.lax.scan(f, init=(jnp.zeros_like(x), jnp.zeros_like(x), 0), xs=B)

        return jnp.exp(-x) * (0.5 * (b - p)) / jnp.sqrt(x)

    return jnp.piecewise(self, [self <= 2, self < 0, self == 0], [small, negative, zero, default])


def _aten_special_modified_bessel_k1(self):
    # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L3444-L3519

    def zero(x):
        return jnp.array(jnp.inf, x.dtype)

    def negative(x):
        return jnp.array(jnp.nan, x.dtype)

    def small(x):
        A = jnp.array(
            [
                -7.02386347938628759343e-18,
                -2.42744985051936593393e-15,
                -6.66690169419932900609e-13,
                -1.41148839263352776110e-10,
                -2.21338763073472585583e-08,
                -2.43340614156596823496e-06,
                -1.73028895751305206302e-04,
                -6.97572385963986435018e-03,
                -1.22611180822657148235e-01,
                -3.53155960776544875667e-01,
                1.52530022733894777053e00,
            ],
            dtype=self.dtype,
        )

        def f(carry, val):
            p, q, a = carry
            p, q = q, a
            a = (x * x - 2.0) * q - p + val
            return (p, q, a), None

        (p, _, a), _ = jax.lax.scan(f, init=(jnp.zeros_like(x), jnp.zeros_like(x), 0), xs=A)

        return jnp.log(0.5 * x) * _aten_special_modified_bessel_i1(x) + 0.5 * (a - p) / x

    def default(x):
        B = jnp.array(
            [
                -5.75674448366501715755e-18,
                1.79405087314755922667e-17,
                -5.68946255844285935196e-17,
                1.83809354436663880070e-16,
                -6.05704724837331885336e-16,
                2.03870316562433424052e-15,
                -7.01983709041831346144e-15,
                2.47715442448130437068e-14,
                -8.97670518232499435011e-14,
                +3.34841966607842919884e-13,
                -1.28917396095102890680e-12,
                5.13963967348173025100e-12,
                -2.12996783842756842877e-11,
                9.21831518760500529508e-11,
                -4.19035475934189648750e-10,
                2.01504975519703286596e-09,
                -1.03457624656780970260e-08,
                5.74108412545004946722e-08,
                -3.50196060308781257119e-07,
                2.40648494783721712015e-06,
                -1.93619797416608296024e-05,
                1.95215518471351631108e-04,
                -2.85781685962277938680e-03,
                1.03923736576817238437e-01,
                2.72062619048444266945e00,
            ],
            dtype=self.dtype,
        )

        def f(carry, val):
            p, q, b = carry
            p, q = q, b
            b = (8.0 / x - 2.0) * q - p + val
            return (p, q, b), None

        (p, _, b), _ = jax.lax.scan(f, init=(jnp.zeros_like(x), jnp.zeros_like(x), 0), xs=B)

        return jnp.exp(-x) * (0.5 * (b - p)) / jnp.sqrt(x)

    return jnp.piecewise(self, [self <= 2, self < 0, self == 0], [small, negative, zero, default])


def _aten_polygamma(n, x):
    if n.dtype in [jnp.int8, jnp.int16, jnp.int32, jnp.int64]:
        n = n.astype(mappings.t2j_dtype(torch.get_default_dtype()))
    return jax.lax.polygamma(jnp.float32(n), x)


def _aten_special_ndtri(self):
    return jax.scipy.special.ndtri(self)


def _aten_special_bessel_j0(self):
    # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L2379-L2489

    def very_small(x):
        return 1.0 - x * x / 4.0

    def small(x):
        RP = jnp.array(
            [
                -4.79443220978201773821e09,
                1.95617491946556577543e12,
                -2.49248344360967716204e14,
                9.70862251047306323952e15,
            ],
            dtype=self.dtype,
        )
        RQ = jnp.array(
            [
                4.99563147152651017219e02,
                1.73785401676374683123e05,
                4.84409658339962045305e07,
                1.11855537045356834862e10,
                2.11277520115489217587e12,
                3.10518229857422583814e14,
                3.18121955943204943306e16,
                1.71086294081043136091e18,
            ],
            dtype=self.dtype,
        )

        rp = op_base.foreach_loop(RP, lambda carry, rp_i: carry * (x * x) + rp_i)
        rq = op_base.foreach_loop(RQ, lambda carry, rq_i: carry * (x * x) + rq_i)

        return (x * x - 5.78318596294678452118e00) * (x * x - 3.04712623436620863991e01) * rp / rq

    def default(x):
        PP = jnp.array(
            [
                7.96936729297347051624e-04,
                8.28352392107440799803e-02,
                1.23953371646414299388e00,
                5.44725003058768775090e00,
                8.74716500199817011941e00,
                5.30324038235394892183e00,
                9.99999999999999997821e-01,
            ],
            dtype=self.dtype,
        )
        PQ = jnp.array(
            [
                9.24408810558863637013e-04,
                8.56288474354474431428e-02,
                1.25352743901058953537e00,
                5.47097740330417105182e00,
                8.76190883237069594232e00,
                5.30605288235394617618e00,
                1.00000000000000000218e00,
            ],
            dtype=self.dtype,
        )
        QP = jnp.array(
            [
                -1.13663838898469149931e-02,
                -1.28252718670509318512e00,
                -1.95539544257735972385e01,
                -9.32060152123768231369e01,
                -1.77681167980488050595e02,
                -1.47077505154951170175e02,
                -5.14105326766599330220e01,
                -6.05014350600728481186e00,
            ],
            dtype=self.dtype,
        )
        QQ = jnp.array(
            [
                6.43178256118178023184e01,
                8.56430025976980587198e02,
                3.88240183605401609683e03,
                7.24046774195652478189e03,
                5.93072701187316984827e03,
                2.06209331660327847417e03,
                2.42005740240291393179e02,
            ],
            dtype=self.dtype,
        )

        pp = op_base.foreach_loop(PP, lambda carry, pp_i: carry * (25.0 / (x * x)) + pp_i)
        pq = op_base.foreach_loop(PQ, lambda carry, pq_i: carry * (25.0 / (x * x)) + pq_i)
        qp = op_base.foreach_loop(QP, lambda carry, qp_i: carry * (25.0 / (x * x)) + qp_i)
        qq = op_base.foreach_loop(QQ, lambda carry, qq_i: carry * (25.0 / (x * x)) + qq_i)

        return (
            (
                pp / pq * jnp.cos(x - 0.785398163397448309615660845819875721)
                - 5.0 / x * (qp / qq) * jnp.sin(x - 0.785398163397448309615660845819875721)
            )
            * 0.797884560802865355879892119868763737
            / jnp.sqrt(x)
        )

    self = jnp.abs(self)
    # Last True condition in  `piecewise` takes priority, but last function is
    # default. See https://github.com/numpy/numpy/issues/16475
    return jnp.piecewise(self, [self <= 5.0, self < 0.00001], [small, very_small, default])


def _aten_special_bessel_j1(self):
    # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L2491-L2597

    def small(x):
        RP = jnp.array(
            [
                -8.99971225705559398224e08,
                4.52228297998194034323e11,
                -7.27494245221818276015e13,
                3.68295732863852883286e15,
            ],
            dtype=self.dtype,
        )
        RQ = jnp.array(
            [
                6.20836478118054335476e02,
                2.56987256757748830383e05,
                8.35146791431949253037e07,
                2.21511595479792499675e10,
                4.74914122079991414898e12,
                7.84369607876235854894e14,
                8.95222336184627338078e16,
                5.32278620332680085395e18,
            ],
            dtype=self.dtype,
        )

        rp = op_base.foreach_loop(RP, lambda carry, rp_i: carry * (x * x) + rp_i)
        rq = op_base.foreach_loop(RQ, lambda carry, rq_i: carry * (x * x) + rq_i)

        return (
            rp / rq * x * (x * x - 1.46819706421238932572e01) * (x * x - 4.92184563216946036703e01)
        )

    def default(x):
        PP = jnp.array(
            [
                7.62125616208173112003e-04,
                7.31397056940917570436e-02,
                1.12719608129684925192e00,
                5.11207951146807644818e00,
                8.42404590141772420927e00,
                5.21451598682361504063e00,
                1.00000000000000000254e00,
            ],
            dtype=self.dtype,
        )
        PQ = jnp.array(
            [
                5.71323128072548699714e-04,
                6.88455908754495404082e-02,
                1.10514232634061696926e00,
                5.07386386128601488557e00,
                8.39985554327604159757e00,
                5.20982848682361821619e00,
                9.99999999999999997461e-01,
            ],
            dtype=self.dtype,
        )
        QP = jnp.array(
            [
                5.10862594750176621635e-02,
                4.98213872951233449420e00,
                7.58238284132545283818e01,
                3.66779609360150777800e02,
                7.10856304998926107277e02,
                5.97489612400613639965e02,
                2.11688757100572135698e02,
                2.52070205858023719784e01,
            ],
            dtype=self.dtype,
        )
        QQ = jnp.array(
            [
                7.42373277035675149943e01,
                1.05644886038262816351e03,
                4.98641058337653607651e03,
                9.56231892404756170795e03,
                7.99704160447350683650e03,
                2.82619278517639096600e03,
                3.36093607810698293419e02,
            ],
            dtype=self.dtype,
        )

        pp = op_base.foreach_loop(PP, lambda carry, pp_i: carry * (25.0 / (x * x)) + pp_i)
        pq = op_base.foreach_loop(PQ, lambda carry, pq_i: carry * (25.0 / (x * x)) + pq_i)
        qp = op_base.foreach_loop(QP, lambda carry, qp_i: carry * (25.0 / (x * x)) + qp_i)
        qq = op_base.foreach_loop(QQ, lambda carry, qq_i: carry * (25.0 / (x * x)) + qq_i)

        return (
            (
                pp / pq * jnp.cos(x - 2.356194490192344928846982537459627163)
                - 5.0 / x * (qp / qq) * jnp.sin(x - 2.356194490192344928846982537459627163)
            )
            * 0.797884560802865355879892119868763737
            / jnp.sqrt(x)
        )

    # If x < 0, bessel_j1(x) = -bessel_j1(-x)
    sign = jnp.sign(self)
    self = jnp.abs(self)
    return sign * jnp.piecewise(
        self,
        [self <= 5.0],
        [small, default],
    )


def _aten_special_bessel_y0(self):
    # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L2599-L2712

    def zero(x):
        return jnp.array(-jnp.inf, x.dtype)

    def negative(x):
        return jnp.array(jnp.nan, x.dtype)

    def small(x):
        YP = jnp.array(
            [
                1.55924367855235737965e04,
                -1.46639295903971606143e07,
                5.43526477051876500413e09,
                -9.82136065717911466409e11,
                8.75906394395366999549e13,
                -3.46628303384729719441e15,
                4.42733268572569800351e16,
                -1.84950800436986690637e16,
            ],
            dtype=self.dtype,
        )
        YQ = jnp.array(
            [
                1.04128353664259848412e03,
                6.26107330137134956842e05,
                2.68919633393814121987e08,
                8.64002487103935000337e10,
                2.02979612750105546709e13,
                3.17157752842975028269e15,
                2.50596256172653059228e17,
            ],
            dtype=self.dtype,
        )

        yp = op_base.foreach_loop(YP, lambda carry, yp_i: carry * (x * x) + yp_i)
        yq = op_base.foreach_loop(YQ, lambda carry, yq_i: carry * (x * x) + yq_i)

        return yp / yq + (
            0.636619772367581343075535053490057448 * jnp.log(x) * _aten_special_bessel_j0(x)
        )

    def default(x):
        PP = jnp.array(
            [
                7.96936729297347051624e-04,
                8.28352392107440799803e-02,
                1.23953371646414299388e00,
                5.44725003058768775090e00,
                8.74716500199817011941e00,
                5.30324038235394892183e00,
                9.99999999999999997821e-01,
            ],
            dtype=self.dtype,
        )
        PQ = jnp.array(
            [
                9.24408810558863637013e-04,
                8.56288474354474431428e-02,
                1.25352743901058953537e00,
                5.47097740330417105182e00,
                8.76190883237069594232e00,
                5.30605288235394617618e00,
                1.00000000000000000218e00,
            ],
            dtype=self.dtype,
        )
        QP = jnp.array(
            [
                -1.13663838898469149931e-02,
                -1.28252718670509318512e00,
                -1.95539544257735972385e01,
                -9.32060152123768231369e01,
                -1.77681167980488050595e02,
                -1.47077505154951170175e02,
                -5.14105326766599330220e01,
                -6.05014350600728481186e00,
            ],
            dtype=self.dtype,
        )
        QQ = jnp.array(
            [
                6.43178256118178023184e01,
                8.56430025976980587198e02,
                3.88240183605401609683e03,
                7.24046774195652478189e03,
                5.93072701187316984827e03,
                2.06209331660327847417e03,
                2.42005740240291393179e02,
            ],
            dtype=self.dtype,
        )

        factor = 25.0 / (x * x)
        pp = op_base.foreach_loop(PP, lambda carry, pp_i: carry * factor + pp_i)
        pq = op_base.foreach_loop(PQ, lambda carry, pq_i: carry * factor + pq_i)
        qp = op_base.foreach_loop(QP, lambda carry, qp_i: carry * factor + qp_i)
        qq = op_base.foreach_loop(QQ, lambda carry, qq_i: carry * factor + qq_i)

        return (
            (
                pp / pq * jnp.sin(x - 0.785398163397448309615660845819875721)
                + 5.0 / x * (qp / qq) * jnp.cos(x - 0.785398163397448309615660845819875721)
            )
            * 0.797884560802865355879892119868763737
            / jnp.sqrt(x)
        )

    return jnp.piecewise(
        self,
        [self <= 5.0, self < 0.0, self == 0.0],
        [small, negative, zero, default],
    )


def _aten_special_bessel_y1(self):
    # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L2714-L2826

    def zero(x):
        return jnp.array(-jnp.inf, x.dtype)

    def negative(x):
        return jnp.array(jnp.nan, x.dtype)

    def small(x):
        YP = jnp.array(
            [
                1.26320474790178026440e09,
                -6.47355876379160291031e11,
                1.14509511541823727583e14,
                -8.12770255501325109621e15,
                2.02439475713594898196e17,
                -7.78877196265950026825e17,
            ],
            dtype=self.dtype,
        )
        YQ = jnp.array(
            [
                5.94301592346128195359e02,
                2.35564092943068577943e05,
                7.34811944459721705660e07,
                1.87601316108706159478e10,
                3.88231277496238566008e12,
                6.20557727146953693363e14,
                6.87141087355300489866e16,
                3.97270608116560655612e18,
            ],
            dtype=self.dtype,
        )

        yp = op_base.foreach_loop(YP, lambda carry, yp_i: carry * (x * x) + yp_i)
        yq = op_base.foreach_loop(YQ, lambda carry, yq_i: carry * (x * x) + yq_i)

        return x * (yp / yq) + (
            0.636619772367581343075535053490057448
            * (_aten_special_bessel_j1(x) * jnp.log(x) - 1.0 / x)
        )

    def default(x):
        PP = jnp.array(
            [
                7.62125616208173112003e-04,
                7.31397056940917570436e-02,
                1.12719608129684925192e00,
                5.11207951146807644818e00,
                8.42404590141772420927e00,
                5.21451598682361504063e00,
                1.00000000000000000254e00,
            ],
            dtype=self.dtype,
        )
        PQ = jnp.array(
            [
                5.71323128072548699714e-04,
                6.88455908754495404082e-02,
                1.10514232634061696926e00,
                5.07386386128601488557e00,
                8.39985554327604159757e00,
                5.20982848682361821619e00,
                9.99999999999999997461e-01,
            ],
            dtype=self.dtype,
        )
        QP = jnp.array(
            [
                5.10862594750176621635e-02,
                4.98213872951233449420e00,
                7.58238284132545283818e01,
                3.66779609360150777800e02,
                7.10856304998926107277e02,
                5.97489612400613639965e02,
                2.11688757100572135698e02,
                2.52070205858023719784e01,
            ],
            dtype=self.dtype,
        )
        QQ = jnp.array(
            [
                7.42373277035675149943e01,
                1.05644886038262816351e03,
                4.98641058337653607651e03,
                9.56231892404756170795e03,
                7.99704160447350683650e03,
                2.82619278517639096600e03,
                3.36093607810698293419e02,
            ],
            dtype=self.dtype,
        )

        factor = 25.0 / (x * x)
        pp = op_base.foreach_loop(PP, lambda carry, pp_i: carry * factor + pp_i)
        pq = op_base.foreach_loop(PQ, lambda carry, pq_i: carry * factor + pq_i)
        qp = op_base.foreach_loop(QP, lambda carry, qp_i: carry * factor + qp_i)
        qq = op_base.foreach_loop(QQ, lambda carry, qq_i: carry * factor + qq_i)

        return (
            (
                pp / pq * jnp.sin(x - 2.356194490192344928846982537459627163)
                + 5.0 / x * (qp / qq) * jnp.cos(x - 2.356194490192344928846982537459627163)
            )
            * 0.797884560802865355879892119868763737
            / jnp.sqrt(x)
        )

    return jnp.piecewise(
        self,
        [self <= 5.0, self < 0.0, self == 0.0],
        [small, negative, zero, default],
    )


def _aten_special_chebyshev_polynomial_t(self, n):
    # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L2828-L2865

    @jnp.vectorize
    def vectorized(x, n_i):
        def negative_n(x):
            return jnp.zeros_like(x)

        def one_x(x):
            return jnp.where((x > 0) | (n_i % 2 == 0), jnp.ones_like(x), -jnp.ones_like(x))

        def large_n_small_x(x):
            return jnp.cos(n_i * jnp.acos(x))

        def zero_n(x):
            return jnp.ones_like(x)

        def one_n(x):
            return x

        def default(x):
            def f(_, carry):
                p, q = carry
                return (q, 2 * x * q - p)

            _, r = jax.lax.fori_loop(0, n_i - 1, f, init_val=(1.0, x))
            return r

        return jnp.piecewise(
            x,
            [n_i == 1, n_i == 0, (n_i == 6) & (jnp.abs(x) < 1), jnp.abs(x) == 1.0, n_i < 0],
            [one_n, zero_n, large_n_small_x, one_x, negative_n, default],
        )

    # Explcicitly vectorize since we must vectorizes over both self and n
    return vectorized(self, n.astype(jnp.int64))


def _aten_special_chebyshev_polynomial_u(self, n):
    # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L2872-L2913

    @jnp.vectorize
    def vectorized(x, n_i):
        def negative_n(x):
            return jnp.zeros_like(x)

        def one_x(x):
            return jnp.where((x > 0) | (n_i % 2 == 0), n_i + 1, -(n_i + 1))

        def large_n_small_x(x):
            sin_acos_x = jnp.sin(jnp.acos(x))
            return jnp.where(
                sin_acos_x != 0,
                jnp.sin((n_i + 1) * jnp.acos(x)) / sin_acos_x,
                (n_i + 1) * jnp.cos((n_i + 1) * jnp.acos(x)) / x,
            )

        def zero_n(x):
            return jnp.ones_like(x)

        def one_n(x):
            return 2 * x

        def default(x):
            def f(_, carry):
                p, q = carry
                return (q, 2 * x * q - p)

            _, r = jax.lax.fori_loop(0, n_i - 1, f, init_val=(1.0, 2 * x))
            return r

        return jnp.piecewise(
            x,
            [
                n_i == 1,
                n_i == 0,
                (n_i > 8) & (jnp.abs(x) < 1),
                jnp.abs(x) == 1.0,
                n_i < 0,
            ],
            [one_n, zero_n, large_n_small_x, one_x, negative_n, default],
        )

    return vectorized(self, n.astype(jnp.int64))


def _aten_special_erfcx(x):
    return jnp.exp(x * x) * jax.lax.erfc(x)


def _aten_erfcx(x):
    return jax.lax.erfc(x)


def _aten_special_hermite_polynomial_h(self, n):
    # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L3036-L3061

    @jnp.vectorize
    def vectorized(x, n_i):
        def negative_n(x):
            return jnp.zeros_like(x)

        def zero_n(x):
            return jnp.ones_like(x)

        def one_n(x):
            return 2 * x

        def default(x):
            def f(k, carry):
                p, q = carry
                return (q, 2 * x * q - 2 * k * p)

            _, r = jax.lax.fori_loop(1, n_i, f, init_val=(1.0, 2 * x))
            return r

        return jnp.piecewise(x, [n_i == 1, n_i == 0, n_i < 0], [one_n, zero_n, negative_n, default])

    return vectorized(self, n.astype(jnp.int64))


def _aten_special_hermite_polynomial_he(self, n):
    # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L3073-L3098

    @jnp.vectorize
    def vectorized(x, n_i):
        def negative_n(x):
            return jnp.zeros_like(x)

        def zero_n(x):
            return jnp.ones_like(x)

        def one_n(x):
            return x

        def default(x):
            def f(k, carry):
                p, q = carry
                return (q, x * q - k * p)

            _, r = jax.lax.fori_loop(1, n_i, f, init_val=(1.0, x))
            return r

        return jnp.piecewise(
            x, [n_i == 1.0, n_i == 0.0, n_i < 0], [one_n, zero_n, negative_n, default]
        )

    return vectorized(self, n.astype(jnp.int64))


def _aten_flatten(x, start_dim=0, end_dim=-1):
    """
    Flattens a JAX array (similar to torch.flatten).

    Args:
        x: The JAX array to be flattened.
        start_dim: The first dimension to include in the flattening.
        end_dim: The last dimension to include in the flattening.

    Returns:
        A flattened JAX array.
    """
    shape = x.shape

    if end_dim < 0:
        end_dim += len(shape)  # Handle negative indexing

    new_shape = (*shape[:start_dim], -1, *shape[end_dim + 1 :])
    return jnp.reshape(x, new_shape)


def _new_empty(self, size, **kwargs):
    dtype = kwargs.get("dtype")
    if dtype is not None:
        dtype = mappings.t2j_dtype(dtype)
    else:
        dtype = self.dtype
    return jnp.empty(size, dtype=dtype)


def _new_empty_strided(self, size, stride, dtype=None, **kwargs):
    # Ignore stride, since JAX and torch tensor doesn't share the same memory.
    if not dtype:
        return jnp.empty(size, dtype=self.dtype)
    else:
        jax_dtype = mappings.t2j_dtype(dtype)
        return jnp.empty(size, dtype=jax_dtype)


def _aten_unsafe_index_put(self, indices, values, accumulate=False):
    return _aten_index_put(self, indices, values, accumulate)


def _aten_tan(self):
    # Decompose tan into sin/cos since neuron doesn't support tan directly
    return jnp.sin(self) / jnp.cos(self)


def _aten_erfc(x):
    return jax.lax.erfc(x)


def _aten__conj_physical(self):
    return jnp.conjugate(self)


def _aten_conj_physical(self):
    return jnp.conjugate(self)


def _aten_log_sigmoid(x):
    return jax.nn.log_sigmoid(x)


# torch.qr
def _aten_qr(input, *args, **kwargs):
    jax_mode = "reduced"
    # torch bool param 'simple=True' corresponds to jax 'reduced' mode,
    # and simple=False corresponds to jax 'complete' mode.
    if kwargs.get("simple") is False:
        jax_mode = "complete"
    return jax.numpy.linalg.qr(input, mode=jax_mode)


# torch.linalg.qr
def _aten_linalg_qr(input, *args, **kwargs):
    mode = kwargs.get("mode", "reduced")
    return jax.numpy.linalg.qr(input, mode=mode)


# torch.linalg.matrix_exp
def _aten_linalg_matrix_exp(input):
    return jax.scipy.linalg.expm(input)


# torch._linalg.slogdet
def _aten__linalg_slogdet(input):
    res = jnp.linalg.slogdet(input)
    return res.sign, res.logabsdet


# torch.linalg.svd
def _aten__linalg_svd(a, full_matrices=False, **kwargs):
    return jnp.linalg.svd(a, full_matrices=full_matrices, **kwargs)


# torch.linalg.pinv
def _aten_linalg_pinv_atol_rtol_tensor(a, rtol=None, **kwargs):
    return jnp.linalg.pinv(a, rtol, hermitian=False)


# torch.linalg.solve
def _aten__linalg_solve_ex(a, b):
    batched = False
    if b.ndim > 1 and b.shape[-1] == a.shape[-1]:
        batched = True
        b = b[..., None]
    res = jnp.linalg.solve(a, b)
    if batched:
        res = res.squeeze(-1)
    info_shape = a.shape[:-2]
    info = jnp.zeros(info_shape, dtype=mappings.t2j_dtype(torch.int32))
    return res, info


# torch.linalg.solve_triangular
def _aten_linalg_solve_triangular(a, b, *, upper=True, left=True, unitriangular=False):
    if left is False:
        a = jnp.matrix_transpose(a)
        b = jnp.matrix_transpose(b)
        upper = not upper
    res = jax.scipy.linalg.solve_triangular(a, b, lower=not upper, unit_diagonal=unitriangular)
    if left is False:
        res = jnp.matrix_transpose(res)
    return res


def _aten_linalg_inv_ex(a):
    ainv = jnp.linalg.inv(a)
    info = jnp.zeros(a.shape[:-2], jnp.int32)
    return ainv, info


def _aten__linalg_check_errors(*args, **kwargs):
    pass


def _aten_median(self, dim=None, keepdim=False):
    output = _with_reduction_scalar(
        functools.partial(jnp.quantile, q=0.5, method="lower"), self, dim=dim, keepdim=keepdim
    ).astype(self.dtype)
    if dim is None:
        return output
    else:
        index = _with_reduction_scalar(_get_median_index, self, dim, keepdim).astype(jnp.int64)
        return output, index


def _aten_nanmedian(input, dim=None, keepdim=False, *, out=None):
    output = _with_reduction_scalar(
        functools.partial(jnp.nanquantile, q=0.5, method="lower"), input, dim=dim, keepdim=keepdim
    ).astype(input.dtype)
    if dim is None:
        return output
    else:
        index = _with_reduction_scalar(_get_median_index, input, dim, keepdim).astype(jnp.int64)
        return output, index


def _get_median_index(x, axis=None, keepdims=False):
    sorted_arg = jnp.argsort(x, axis=axis)
    n = x.shape[axis] if axis is not None else x.size
    if n % 2 == 1:
        index = n // 2
    else:
        index = (n // 2) - 1
    if axis is None:
        median_index = sorted_arg[index]
    else:
        median_index = jnp.take(sorted_arg, index, axis=axis)
    if keepdims and axis is not None:
        median_index = jnp.expand_dims(median_index, axis)
    return median_index


def _aten_triangular_solve(b, a, upper=True, transpose=False, unittriangular=False):
    return (
        jax.lax.linalg.triangular_solve(
            a,
            b,
            left_side=True,
            lower=not upper,
            transpose_a=transpose,
            unit_diagonal=unittriangular,
        ),
        a,
    )


# func: _fft_c2c(Tensor self, SymInt[] dim, int normalization, bool forward) -> Tensor
def _aten__fft_c2c(self, dim, normalization, forward):
    if forward:
        norm = [
            "backward",
            "ortho",
            "forward",
        ][normalization]
        return jnp.fft.fftn(self, axes=dim, norm=norm)
    else:
        norm = [
            "forward",
            "ortho",
            "backward",
        ][normalization]
        return jnp.fft.ifftn(self, axes=dim, norm=norm)


def _aten__fft_r2c(self, dim, normalization, onesided):
    norm = [
        "backward",
        "ortho",
        "forward",
    ][normalization]
    if onesided:
        return jnp.fft.rfftn(self, axes=dim, norm=norm)
    else:
        return jnp.fft.fftn(self, axes=dim, norm=norm)


def _aten__fft_c2r(self, dim, normalization, last_dim_size):
    norm = [
        "forward",
        "ortho",
        "backward",
    ][normalization]
    if len(dim) == 1:
        s = [last_dim_size]
    else:
        s = None
    return jnp.fft.irfftn(self, norm=norm, axes=dim, s=s)


def _aten_trilinear(i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim=1):
    return _aten_sum(
        jnp.expand_dims(i1, expand1) * jnp.expand_dims(i2, expand2) * jnp.expand_dims(i3, expand3),
        sumdim,
    )


def _aten_max_unpoolxd(input, indices, output_size, stride=None, padding=0):
    if output_size is None:
        raise ValueError("output_size value is not set correctly. It cannot be None or empty.")

    output_size = [input.shape[0], input.shape[1]] + output_size
    output = jnp.zeros(output_size, dtype=input.dtype)

    for idx in np.ndindex(input.shape):
        max_index = indices[idx]
        spatial_dims = output_size[2:]  # (D, H, W)
        unpooled_spatial_idx = np.unravel_index(max_index, spatial_dims)
        full_idx = idx[:2] + unpooled_spatial_idx
        output = output.at[full_idx].set(input[idx])

    return output


def _aten_upsample(
    input,
    output_size,
    align_corners,
    antialias,
    method,
    scale_factors=None,
    scales_h=None,
    scales_w=None,
):
    # input: is of type jaxlib.xla_extension.ArrayImpl
    image = input

    # https://jax.readthedocs.io/en/latest/_autosummary/jax.image.resize.html
    # Resize does not distinguish batch, channel size.
    # We need to leave them as is
    # https://pytorch.org/vision/stable/transforms.html#supported-input-types-and-conventions
    # pytorch image shape is (C,H,W) or (N,C,H,W)
    # N - batch size
    # C - no of channels
    # H,W - heigth, width

    shape = list(image.shape)
    # overriding output_size
    if scale_factors:
        shape[-1] = int(math.floor(shape[-1] * scale_factors[-1]))
        shape[-2] = int(math.floor(shape[-2] * scale_factors[-2]))
    if scales_h:
        shape[-2] = int(math.floor(shape[-2] * scales_h))
    if scales_w:
        shape[-1] = int(math.floor(shape[-1] * scales_w))
    # output_size overrides scale_factors, scales_*
    if output_size:
        shape[-1] = output_size[-1]
        shape[-2] = output_size[-2]

    # pytorch upsample_bilinear returns the input as is when the shape is the same as input
    if shape == list(image.shape):
        return image

    spatial_dims = (2, 3)
    if len(shape) == 3:
        spatial_dims = (1, 2)

    scale = list([shape[i] / image.shape[i] for i in spatial_dims])
    if scale_factors:
        scale = scale_factors
    if scales_h:
        scale[0] = scales_h
    if scales_w:
        scale[1] = scales_w
    scale = jnp.array(scale)

    # align_corners is not supported in resize()
    # https://github.com/jax-ml/jax/issues/11206
    if align_corners:
        scale = jnp.array([(shape[i] - 1.0) / (image.shape[i] - 1.0) for i in spatial_dims])

    translation = jnp.array([0 for i in spatial_dims])

    return jax_reimplement.scale_and_translate(
        image,
        shape,
        method=method,
        scale=scale,
        spatial_dims=spatial_dims,
        translation=translation,
        antialias=antialias,
    )


def _aten_upsample_billinear_aa(
    input, output_size, align_corners, scale_factors=None, scales_h=None, scales_w=None
):
    return _aten_upsample(
        input,
        output_size,
        align_corners,
        True,  # antialias
        "bilinear",  # method
        scale_factors,
        scales_h,
        scales_w,
    )


def _aten_upsample_bicubic2d_aa(
    input, output_size, align_corners, scale_factors=None, scales_h=None, scales_w=None
):
    return _aten_upsample(
        input,
        output_size,
        align_corners,
        True,  # antialias
        "bicubic",  # method
        scale_factors,
        scales_h,
        scales_w,
    )


def _aten_polar(abs, angle, *, out=None):
    return jax.lax.complex(abs * jnp.cos(angle), abs * jnp.sin(angle))


def _aten_cdist(x1, x2, p=2.0, compute_mode="use_mm_for_euclid_dist_if_necessary"):
    x1 = x1.astype(jnp.float32)
    x2 = x2.astype(jnp.float32)

    if p == 0.0:
        # For p = 0, use Hamming-like distance multiplied by the number of elements
        return _hamming_distance(x1, x2).astype(jnp.float32)
    elif p == 2.0:
        # Use optimized Euclidean distance calculation
        if (
            compute_mode == "use_mm_for_euclid_dist_if_necessary"
            and (x1.shape[-2] > 25 or x2.shape[-2] > 25)
        ) or compute_mode == "use_mm_for_euclid_dist":
            return _euclidean_mm(x1, x2)
        else:
            return _euclidean_direct(x1, x2)
    else:
        # General p-norm distance calculation
        diff = jnp.abs(jnp.expand_dims(x1, -2) - jnp.expand_dims(x2, -3))
        return jnp.sum(jnp.power(diff, p), axis=-1).astype(jnp.float32) ** (1 / p)


def _hamming_distance(x1, x2):
    """
    Computes the Hamming-like distance for p=0.

    Args:
        x1: JAX array of shape (..., P, M)
        x2: JAX array of shape (..., R, M)

    Returns:
        JAX array of shape (..., P, R) representing pairwise Hamming distances.
    """
    diff = jnp.not_equal(jnp.expand_dims(x1, -2), jnp.expand_dims(x2, -3))

    hamming_dist = jnp.sum(diff, axis=-1).astype(jnp.float32)

    return hamming_dist


def _euclidean_mm(x1, x2):
    """
    Computes the Euclidean distance using matrix multiplication.

    Args:
        x1: JAX array of shape (..., P, M)
        x2: JAX array of shape (..., R, M)

    Returns:
        JAX array of shape (..., P, R) representing pairwise Euclidean distances.
    """
    x1_sq = jnp.sum(x1**2, axis=-1, keepdims=True).astype(jnp.float32)
    x2_sq = jnp.sum(x2**2, axis=-1, keepdims=True).astype(jnp.float32)

    x2_sq = jnp.swapaxes(x2_sq, -2, -1)

    dot_product = jnp.matmul(x1, jnp.swapaxes(x2, -1, -2))

    dist_sq = x1_sq + x2_sq - 2 * dot_product
    dist_sq = jnp.maximum(dist_sq, 0.0)
    dist = jnp.sqrt(dist_sq).astype(jnp.float32)

    return dist


def _euclidean_direct(x1, x2):
    """
    Computes the Euclidean distance directly without matrix multiplication.

    Args:
        x1: JAX array of shape (..., P, M)
        x2: JAX array of shape (..., R, M)

    Returns:
        JAX array of shape (..., P, R) representing pairwise Euclidean distances.
    """
    diff = jnp.expand_dims(x1, -2) - jnp.expand_dims(x2, -3)

    dist_sq = jnp.sum(diff**2, axis=-1).astype(jnp.float32)

    dist_sq = jnp.maximum(dist_sq, 0.0)

    dist = jnp.sqrt(dist_sq).astype(jnp.float32)

    return dist


def _aten_lu_unpack(LU_data, LU_pivots, unpack_data=True, unpack_pivots=True):
    # lu_unpack doesnt exist in jax.
    # Get commonly used data shape variables
    n = LU_data.shape[-2]
    m = LU_data.shape[-1]
    dim = min(n, m)

    ### Compute the Lower and Upper triangle
    if unpack_data:
        # Extract lower triangle
        L = jnp.tril(LU_data, k=-1)

        # emulate pytorch behavior: Add ones to the diagonal of L
        eye = jnp.eye(n, m, dtype=LU_data.dtype)
        L = L + eye

        # emulate pytorch behavior: Reshape lower triangle to match pivot
        start_indices = jnp.zeros(len(LU_data.shape), dtype=int)
        limit_indices = list(LU_data.shape)
        limit_indices[-1] = dim
        L = jax.lax.slice(L, start_indices, limit_indices)

        # Extract upper triangle
        U = jnp.triu(LU_data)

        # emulate pytorch behavior: Reshape upper triangle to match pivot
        start_indices = jnp.zeros(len(LU_data.shape), dtype=int)
        limit_indices = list(LU_data.shape)
        limit_indices[-2] = dim
        U = jax.lax.slice(U, start_indices, limit_indices)
    else:
        # emulate pytroch behavior: return empty tensors
        L = torch.empty(torch.Size([0]))
        U = torch.empty(torch.Size([0]))

    ### Compute the Permutation matrix
    if unpack_pivots:
        # We should return a permutation matrix (2D) for each pivot array (1D)
        # The shape of the final Permutation matrix depends on the shape of the input
        # data and the pivots

        # start with a 2D identity matrix and tile it to the other dims of input data
        identity2d = jnp.identity(n, dtype=jnp.float32)
        tile_shape = list(LU_data.shape)
        tile_shape[-1] = 1
        tile_shape[-2] = 1
        P = jnp.tile(identity2d, tile_shape)

        # closure to be called for each input 2D matrix.
        def _lu_unpack_2d(p, pivot):
            _pivot = pivot - 1  # pivots are offset by 1 in jax
            indices = jnp.array([*range(n)], dtype=jnp.int32)

            def update_indices(i, _indices):
                tmp = _indices[i]
                _indices = _indices.at[i].set(_indices[_pivot[i]])
                _indices = _indices.at[_pivot[i]].set(tmp)
                return _indices

            indices = jax.lax.fori_loop(0, _pivot.size, update_indices, indices)
            p = p[jnp.array(indices)]
            p = jnp.transpose(p)
            return p

        if len(LU_pivots.shape) == 1:
            # if we are dealing with a simple 2D input and 1D pivot, call the closure directly
            P = _lu_unpack_2d(P, LU_pivots)
        else:
            # We are dealing with >=3D inputs. Flatten inputs to 3D and use vmap to call the
            # closure for each 2D matrix. Finally unflatten the result to match the input data
            # shape.

            # reshape permutation matrix to 3d
            dim_size = jnp.prod(jnp.array(P.shape[:-2]))
            newPshape = (dim_size, P.shape[-2], P.shape[-1])
            reshapedP = P.reshape(newPshape)

            # reshape pivots to 3d
            dim_size = jnp.prod(jnp.array(LU_pivots.shape[:-1]))
            newPivotshape = (dim_size, LU_pivots.shape[-1])
            reshapedPivot = LU_pivots.reshape(newPivotshape)

            # vmap the reshaped 3d tensors
            v_lu_unpack_2d = jax.vmap(_lu_unpack_2d, in_axes=(0, 0))
            unpackedP = v_lu_unpack_2d(reshapedP, reshapedPivot)

            # reshape result back to P's shape
            newRetshape = (*P.shape[:-2], unpackedP.shape[-2], unpackedP.shape[-1])
            P = unpackedP.reshape(newRetshape)
    else:
        # emulate pytroch behavior: return empty tensors
        P = torch.empty(torch.Size([0]))

    return P, L, U


@register_aten(["aten::linear", "aten::linear.out"])
def linear(input, weight, bias=None):
    res = input @ jnp.transpose(weight)
    if bias is not None:
        res += bias
    return res


@register_aten(["aten::linear_backward", "aten::linear_backward.out"], static_argnums=(3,))
def _aten_linear_backward(self, grad, weight, mask):
    """
    Compute gradients for linear layer backward pass.

    Args:
        self: Input tensor to linear layer, (shape: [..., in_features])
        grad: Gradient tensor from the output, (shape: [..., out_features])
        weight: Weight tensor of linear layer (shape: [out_features, in_features])
        mask: Boolean mask indicating which gradients to compute [input_grad, weight_grad, bias_grad]

    Returns:
        Tuple of (self_grad, weight_grad, bias_grad)
    """

    # Compute input gradient: grad @ weight
    if mask[0]:
        self_grad = grad @ weight
    else:
        self_grad = None

    # Compute weight gradient: grad.T @ input
    # compute bias gradient: sum(grad)
    # In pytorch, it calculate gradient for both if one of them requires it
    if mask[1] or mask[2]:
        batch_dims = "".join(chr(ord("a") + j) for j in range(self.ndim - 1))
        grad_input_eq = f"{batch_dims}i,{batch_dims}j->ij"
        weight_grad = jnp.einsum(grad_input_eq, grad, self)
        bias_grad = jnp.sum(grad, axis=tuple(range(grad.ndim - 1)), keepdims=False)
    else:
        weight_grad = None
        bias_grad = None

    return (self_grad, weight_grad, bias_grad)


def kthvalue(input, k, dim=None, keepdim=False, *, out=None):
    if input.ndim == 0:
        return input, jnp.array(0)
    dimension = -1
    if dim is not None:
        dimension = dim
    while dimension < 0:
        dimension = dimension + input.ndim
    values = jax.lax.index_in_dim(jnp.partition(input, k - 1, dimension), k - 1, dimension, keepdim)
    indices = jax.lax.index_in_dim(
        jnp.argpartition(input, k - 1, dimension).astype("int64"), k - 1, dimension, keepdim
    )
    return values, indices


def _aten_take(self, index):
    return self.flatten()[index]


# func: pad(Tensor self, SymInt[] pad, str mode="constant", float? value=None) -> Tensor
@register_aten(
    [
        "aten::pad",
        "aten::pad.out",
    ],
    operation_type="arithmetic",
    static_argnums=(1, 2, 3),
)
def _aten_pad(self, pad, mode="constant", value=None):
    if not isinstance(pad, (tuple, list)) or len(pad) % 2 != 0:
        raise RuntimeError("Padding must be a sequence of even length.")

    num_dims = self.ndim
    if len(pad) > 2 * num_dims:
        raise RuntimeError(
            f"Padding sequence length ({len(pad)}) exceeds 2 * number of dimensions ({2 * num_dims})."
        )

    # JAX's pad function expects padding for each dimension as a tuple of (low, high)
    # We need to reverse the pad sequence and group them for JAX.
    # pad = [p_l0, p_r0, p_l1, p_r1, ...]
    # becomes ((..., ..., (p_l1, p_r1), (p_l0, p_r0)))
    jax_pad_width = []
    # Iterate in reverse pairs
    for i in range(len(pad) // 2):
        jax_pad_width.append((pad[(2 * i)], pad[(2 * i + 1)]))

    # Pad any leading dimensions with (0, 0) if the pad sequence is shorter
    # than the number of dimensions.
    for _ in range(num_dims - len(pad) // 2):
        jax_pad_width.append((0, 0))

    # Reverse the jax_pad_width list to match the dimension order
    jax_pad_width.reverse()

    if mode == "constant":
        if value is None:
            value = 0.0
        return jnp.pad(self, pad_width=jax_pad_width, mode="constant", constant_values=value)
    elif mode == "reflect":
        return jnp.pad(self, pad_width=jax_pad_width, mode="reflect")
    elif mode == "replicate" or mode == "edge":
        return jnp.pad(self, pad_width=jax_pad_width, mode="edge")
    elif mode == "circular":
        # manual implementation for circular mode, since its not supported jnp.pad(),
        # whereas supported in torch.nn.functional.pad()
        # torch implementation from _pad_circular_symint -
        # https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/PadNd.cpp
        ndim_padded = len(pad) // 2
        ndim_nonpadded = num_dims - ndim_padded
        # Reverse the pad list to match dimension order (last dim first)
        padding_list = [(pad[2 * i], pad[2 * i + 1]) for i in range(ndim_padded)]
        padding_list.reverse()
        # Prepend the non-padded dimensions with (0, 0)
        pad_config = [(0, 0)] * ndim_nonpadded + padding_list
        padded_x = self
        for dim, (pad_l, pad_r) in enumerate(pad_config):
            if pad_l == 0 and pad_r == 0:
                continue
            # JAX's lax.pad with `mode='wrap'` is more performant than manual slice-and-concat.
            padded_x = jnp.pad(
                padded_x,
                ((0, 0),) * dim + ((pad_l, pad_r),) + ((0, 0),) * (num_dims - dim - 1),
                mode="wrap",
            )
        return padded_x
    else:
        raise RuntimeError(
            f"Unsupported padding mode: {mode}. Expected 'constant', 'reflect', or 'edge'."
        )


def _aten_is_nonzero(a):
    a = jnp.squeeze(a)
    if a.shape == (0,):
        raise RuntimeError("bool value of Tensor with no values is ambiguous")
    if a.ndim != 0:
        raise RuntimeError("bool value of Tensor with more than one value is ambiguous")
    return a.item() != 0


def _aten_logit(self: jax.Array, eps: float | None = None) -> jax.Array:
    """
    Computes the logit function of the input tensor.

    logit(p) = log(p / (1 - p))

    Args:
      self: Input tensor.
      eps: A small value to clip the input tensor to avoid log(0) or division by zero.
           If None, no clipping is performed.

    Returns:
      A tensor with the logit of each element of the input.
    """
    if eps is not None:
        self = jnp.clip(self, eps, 1.0 - eps)
    res = jnp.log(self / (1.0 - self))
    res = res.astype(mappings.t2j_dtype(torch.get_default_dtype()))
    return res


@register_aten(
    ["aten::floor_divide", "aten::floor_divide.out"],
    operation_type="arithmetic",
    output_params=("out",),
)
def _aten_floor_divide(x, y, *, out=None):
    res = jnp.floor_divide(x, y)
    return res


def _aten__assert_tensor_metadata(*args, **kwargs):
    pass
