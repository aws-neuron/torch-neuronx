# Ops Organization

This directory contains torch-neuronx operation implementations organized by functionality, following PyTorch's organization pattern.

## File Structure

### activation.py
Activation functions and their backward passes:
- relu, gelu, silu, sigmoid, tanh
- sigmoid_backward, silu_backward, tanh_backward

### binary.py
Binary arithmetic operations:
- add, sub, mul, div
- pow, remainder, floor_divide
- addcmul, addcdiv

### unary.py
Unary mathematical operations:
- abs, neg, sqrt, rsqrt
- cos, sin, exp, log
- erf, erfinv, ceil, trunc
- sign, reciprocal

### comparison.py
Comparison and clamping operations:
- eq, gt, le
- clamp, clamp_min, clamp_max
- isnan, isinf, isneginf, isfinite

### logical.py
Logical and bitwise operations:
- logical_and, logical_or, logical_not, logical_xor
- bitwise_and, bitwise_or, bitwise_xor, bitwise_not
- bitwise_left_shift, bitwise_right_shift
- signbit, sgn

### reduction.py
Reduction operations:
- sum, mean, max, amax, amin
- all, any
- cumsum

### linear_algebra.py
Linear algebra operations:
- matmul, mm, bmm
- addmm, addmv
- linalg_vector_norm
- _foreach_norm

### indexing.py
Indexing, slicing, and selection operations:
- index, index_select, index_copy, index_add
- gather, scatter, scatter_add
- masked_select, masked_fill
- where, nonzero, topk
- _index_put_impl, index_select_backward

### tensor_ops.py
Tensor manipulation (shape, stacking, etc.):
- cat, stack, vstack
- atleast_2d
- triu, tril
- flip

### creation.py
Tensor creation operations:
- arange, zeros, ones, ones_like
- eye
- fill_, zero_

### convolution.py
Convolution operations:
- convolution
- convolution_backward

### normalization.py
Normalization operations:
- native_layer_norm
- native_layer_norm_backward
- batch_norm_backward_elemt

### optimizer.py
Optimizer operations:
- _fused_adamw

### misc.py
Miscellaneous operations that don't fit other categories:
- embedding, embedding_dense_backward
- softmax, log_softmax and their variants
- lerp, histc
- linear_backward
- reflection_pad2d, replication_pad2d

## How PyTorch Organizes Ops

PyTorch organizes operations by **functionality/category** in files like:
- `Activation.cpp` - activation functions
- `BinaryOps.cpp` - binary operations
- `Blas.cpp` / `BatchLinearAlgebra.cpp` - linear algebra
- `Convolution.cpp` - convolution operations
- `Indexing.cpp` - indexing operations
- `Reduction.cpp` - reduction operations
- `TensorShape.cpp` - shape manipulation
- `UnaryOps.cpp` - unary operations

## Adding New Operations

When adding new operations:
1. Identify the category (activation, binary, unary, etc.)
2. Add the operation to the appropriate file
3. If it doesn't fit existing categories, add to `misc.py` or create a new category file
4. Update `__init__.py` if you create a new category file
