"""@nki_op wrappers for the dynamic-shape MoE-forward NKI kernels.

Covers the MoE forward path: add, mul, silu, matmul (dynamic-M), gather (index),
scatter_add.
"""

import torch

from torch_neuronx.nki_hop import nki_op, wrap_nki
from torch_neuronx.utils import get_logical_neuron_cores

from .add_kernel import add_kernel
from .dynamic_m_matmul_kernel import dynamic_m_matmul_kernel
from .gather_kernel import gather_kernel
from .mul_kernel import mul_kernel
from .scatter_add_kernel import scatter_add_kernel
from .silu_kernel import silu_kernel

# Grid must match logical_neuron_cores so the compiler emits identical code on
# every core (NCC_IXGM002 fires on basic-block-count mismatch otherwise). The
# elementwise kernels don't use nl.program_id, so each core redundantly executes
# the full kernel on the full input — correct for idempotent writes.
_GRID = (int(get_logical_neuron_cores()),)

_wrapped_add = wrap_nki(add_kernel)
_wrapped_silu = wrap_nki(silu_kernel)
_wrapped_mul = wrap_nki(mul_kernel)
_wrapped_matmul = wrap_nki(dynamic_m_matmul_kernel)
_wrapped_gather = wrap_nki(gather_kernel)
_wrapped_scatter_add = wrap_nki(scatter_add_kernel)


@nki_op("dynamic_nki::add", mutates_args={})
def dynamic_add_nki(
    input_a: torch.Tensor, input_b: torch.Tensor, num_m_tiles: torch.Tensor
) -> torch.Tensor:
    return _wrapped_add[_GRID](input_a, input_b, num_m_tiles)


@nki_op("dynamic_nki::silu", mutates_args={})
def dynamic_silu_nki(input_tensor: torch.Tensor, num_m_tiles: torch.Tensor) -> torch.Tensor:
    return _wrapped_silu[_GRID](input_tensor, num_m_tiles)


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
