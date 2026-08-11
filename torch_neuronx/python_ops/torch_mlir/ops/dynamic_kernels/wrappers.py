"""@nki_op wrappers for the dynamic-shape MoE NKI kernels.

Forward path: add, mul, silu, matmul (dynamic-M), gather (index), scatter_add.
Backward: matmul_k (dynamic-K, for grad_weight = x_eᵀ @ dout), silu_bwd.
"""

import torch

from torch_neuronx.nki_hop import nki_op, wrap_nki
from torch_neuronx.utils import get_logical_neuron_cores

from .add_kernel import add_kernel
from .cos_kernel import cos_kernel
from .dynamic_k_matmul_kernel import dynamic_k_matmul_kernel
from .dynamic_m_matmul_kernel import dynamic_m_matmul_kernel
from .exp_kernel import exp_kernel
from .gather_kernel import gather_kernel
from .mul_kernel import mul_kernel
from .scatter_add_kernel import scatter_add_kernel
from .sigmoid_backward_kernel import sigmoid_backward_kernel
from .sigmoid_kernel import sigmoid_kernel
from .silu_bwd_kernel import silu_bwd_kernel
from .silu_kernel import silu_kernel
from .sin_kernel import sin_kernel
from .sub_kernel import sub_kernel

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
_wrapped_matmul = wrap_nki(dynamic_m_matmul_kernel)
_wrapped_matmul_k = wrap_nki(dynamic_k_matmul_kernel)
_wrapped_silu_bwd = wrap_nki(silu_bwd_kernel)
_wrapped_gather = wrap_nki(gather_kernel)
_wrapped_scatter_add = wrap_nki(scatter_add_kernel)


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
