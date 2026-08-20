"""@nki_op wrappers for the dynamic-shape MoE NKI kernels.

Forward path: add, mul, silu, matmul (dynamic-M), gather (index), scatter_add.
Backward: matmul_k (dynamic-K, for grad_weight = x_eᵀ @ dout), silu_bwd.
"""

import torch

from torch_neuronx.nki_hop import nki_op, wrap_nki
from torch_neuronx.utils import get_logical_neuron_cores

from .add_kernel import add_kernel
from .bitwise_or_kernel import bitwise_or_kernel
from .cast_kernel import cast_kernel_bf16, cast_kernel_fp16, cast_kernel_fp32
from .cat_kernel import cat_kernel
from .cos_kernel import cos_kernel
from .dynamic_k_matmul_kernel import dynamic_k_matmul_kernel
from .dynamic_m_matmul_kernel import dynamic_m_matmul_kernel
from .exp_kernel import exp_kernel
from .gather_kernel import gather_kernel
from .mul_kernel import mul_kernel
from .neg_kernel import neg_kernel
from .ones_like_kernel import ones_like_kernel
from .scatter_add_kernel import scatter_add_kernel
from .sigmoid_backward_kernel import sigmoid_backward_kernel
from .sigmoid_kernel import sigmoid_kernel
from .silu_bwd_kernel import silu_bwd_kernel
from .silu_kernel import silu_kernel
from .sin_kernel import sin_kernel
from .sub_kernel import sub_kernel
from .sum_kernel import sum_kernel
from .where_kernel import where_kernel

# Grid must match logical_neuron_cores so the compiler emits identical code on
# every core (NCC_IXGM002 fires on basic-block-count mismatch otherwise). The
# elementwise kernels don't use nl.program_id, so each core redundantly executes
# the full kernel on the full input — correct for idempotent writes.
_GRID = (int(get_logical_neuron_cores()),)

_wrapped_add = wrap_nki(add_kernel)
_wrapped_sub = wrap_nki(sub_kernel)
_wrapped_sigmoid_backward = wrap_nki(sigmoid_backward_kernel)
_wrapped_silu = wrap_nki(silu_kernel)
_wrapped_exp = wrap_nki(exp_kernel)
_wrapped_cos = wrap_nki(cos_kernel)
_wrapped_sin = wrap_nki(sin_kernel)
_wrapped_mul = wrap_nki(mul_kernel)
_wrapped_sum = wrap_nki(sum_kernel)
_wrapped_cat = wrap_nki(cat_kernel)
# One per cast TARGET dtype; the source dtype comes from the input tensor's metadata.
_wrapped_cast_bf16 = wrap_nki(cast_kernel_bf16)
_wrapped_cast_fp16 = wrap_nki(cast_kernel_fp16)
_wrapped_cast_fp32 = wrap_nki(cast_kernel_fp32)
_wrapped_matmul = wrap_nki(dynamic_m_matmul_kernel)
_wrapped_matmul_k = wrap_nki(dynamic_k_matmul_kernel)
_wrapped_silu_bwd = wrap_nki(silu_bwd_kernel)
_wrapped_gather = wrap_nki(gather_kernel)
_wrapped_scatter_add = wrap_nki(scatter_add_kernel)
_wrapped_ones_like = wrap_nki(ones_like_kernel)
_wrapped_where = wrap_nki(where_kernel)
_wrapped_bitwise_or = wrap_nki(bitwise_or_kernel)


@nki_op("dynamic_nki::add", mutates_args={})
def dynamic_add_nki(
    input_a: torch.Tensor, input_b: torch.Tensor, num_m_tiles: torch.Tensor
) -> torch.Tensor:
    return _wrapped_add[_GRID](input_a, input_b, num_m_tiles)


@nki_op("dynamic_nki::sub", mutates_args={})
def dynamic_sub_nki(
    input_a: torch.Tensor, input_b: torch.Tensor, num_m_tiles: torch.Tensor
) -> torch.Tensor:
    return _wrapped_sub[_GRID](input_a, input_b, num_m_tiles)


@nki_op("dynamic_nki::sigmoid_backward", mutates_args={})
def dynamic_sigmoid_backward_nki(
    grad_output: torch.Tensor, output: torch.Tensor, num_m_tiles: torch.Tensor
) -> torch.Tensor:
    return _wrapped_sigmoid_backward[_GRID](grad_output, output, num_m_tiles)


@nki_op("dynamic_nki::silu", mutates_args={})
def dynamic_silu_nki(input_tensor: torch.Tensor, num_m_tiles: torch.Tensor) -> torch.Tensor:
    return _wrapped_silu[_GRID](input_tensor, num_m_tiles)


@nki_op("dynamic_nki::exp", mutates_args={})
def dynamic_exp_nki(input_tensor: torch.Tensor, num_m_tiles: torch.Tensor) -> torch.Tensor:
    return _wrapped_exp[_GRID](input_tensor, num_m_tiles)


@nki_op("dynamic_nki::cos", mutates_args={})
def dynamic_cos_nki(input_tensor: torch.Tensor, num_m_tiles: torch.Tensor) -> torch.Tensor:
    return _wrapped_cos[_GRID](input_tensor, num_m_tiles)


@nki_op("dynamic_nki::sin", mutates_args={})
def dynamic_sin_nki(input_tensor: torch.Tensor, num_m_tiles: torch.Tensor) -> torch.Tensor:
    return _wrapped_sin[_GRID](input_tensor, num_m_tiles)


@nki_op("dynamic_nki::mul", mutates_args={})
def dynamic_mul_nki(
    input_a: torch.Tensor, input_b: torch.Tensor, num_m_tiles: torch.Tensor
) -> torch.Tensor:
    return _wrapped_mul[_GRID](input_a, input_b, num_m_tiles)


@nki_op("dynamic_nki::sum", mutates_args={})
def dynamic_sum_nki(input_tensor: torch.Tensor, num_m_tiles: torch.Tensor) -> torch.Tensor:
    return _wrapped_sum[_GRID](input_tensor, num_m_tiles)


@nki_op("dynamic_nki::cat", mutates_args={})
def dynamic_cat_nki(
    input_a: torch.Tensor, input_b: torch.Tensor, num_m_tiles: torch.Tensor
) -> torch.Tensor:
    return _wrapped_cat[_GRID](input_a, input_b, num_m_tiles)


# Dtype cast (_to_copy / copy_); the adapter picks the wrapper by target dtype.
# These take the real ROW count, not a tile count: the kernel needs the exact M to
# place its ragged tail tile, and pads nothing.
@nki_op("dynamic_nki::cast_bf16", mutates_args={})
def dynamic_cast_bf16_nki(input_tensor: torch.Tensor, num_rows: torch.Tensor) -> torch.Tensor:
    return _wrapped_cast_bf16[_GRID](input_tensor, num_rows)


@nki_op("dynamic_nki::cast_fp16", mutates_args={})
def dynamic_cast_fp16_nki(input_tensor: torch.Tensor, num_rows: torch.Tensor) -> torch.Tensor:
    return _wrapped_cast_fp16[_GRID](input_tensor, num_rows)


@nki_op("dynamic_nki::cast_fp32", mutates_args={})
def dynamic_cast_fp32_nki(input_tensor: torch.Tensor, num_rows: torch.Tensor) -> torch.Tensor:
    return _wrapped_cast_fp32[_GRID](input_tensor, num_rows)


@nki_op("dynamic_nki::matmul", mutates_args={})
def dynamic_matmul_nki(
    lhs: torch.Tensor, rhs: torch.Tensor, num_m_tiles: torch.Tensor
) -> torch.Tensor:
    return _wrapped_matmul[_GRID](lhs, rhs, num_m_tiles)


@nki_op("dynamic_nki::matmul_k", mutates_args={})
def dynamic_matmul_k_nki(
    x_e: torch.Tensor,
    dout: torch.Tensor,
    num_k_tiles: torch.Tensor,
) -> torch.Tensor:
    return _wrapped_matmul_k[_GRID](x_e, dout, num_k_tiles)


@nki_op("dynamic_nki::silu_bwd", mutates_args={})
def dynamic_silu_bwd_nki(
    grad_output: torch.Tensor, input_tensor: torch.Tensor, num_m_tiles: torch.Tensor
) -> torch.Tensor:
    return _wrapped_silu_bwd[_GRID](grad_output, input_tensor, num_m_tiles)


@nki_op("dynamic_nki::gather", mutates_args={})
def dynamic_gather_nki(
    source: torch.Tensor, indices: torch.Tensor, num_m_tiles: torch.Tensor
) -> torch.Tensor:
    return _wrapped_gather[_GRID](source, indices, num_m_tiles)


@nki_op("dynamic_nki::scatter_add", mutates_args={"output_buffer"})
def dynamic_scatter_add_nki(
    data: torch.Tensor,
    indices: torch.Tensor,
    output_buffer: torch.Tensor,
    num_m_tiles: torch.Tensor,
) -> torch.Tensor:
    return _wrapped_scatter_add[_GRID](data, indices, output_buffer, num_m_tiles)


@nki_op("dynamic_nki::ones_like", mutates_args={})
def dynamic_ones_like_nki(input_tensor: torch.Tensor, num_m_tiles: torch.Tensor) -> torch.Tensor:
    return _wrapped_ones_like[_GRID](input_tensor, num_m_tiles)


@nki_op("dynamic_nki::where", mutates_args={})
def dynamic_where_nki(
    condition: torch.Tensor,
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    num_m_tiles: torch.Tensor,
) -> torch.Tensor:
    return _wrapped_where[_GRID](condition, input_a, input_b, num_m_tiles)


@nki_op("dynamic_nki::bitwise_or", mutates_args={})
def dynamic_bitwise_or_nki(
    input_a: torch.Tensor, input_b: torch.Tensor, num_m_tiles: torch.Tensor
) -> torch.Tensor:
    return _wrapped_bitwise_or[_GRID](input_a, input_b, num_m_tiles)
