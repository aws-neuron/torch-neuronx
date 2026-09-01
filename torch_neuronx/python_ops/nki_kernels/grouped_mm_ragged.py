import torch

from torch_neuronx import nki_op, wrap_nki
from torch_neuronx.python_ops.nki_kernels._grouped_mm_ragged import (
    grouped_mm_ragged_2d_2d,
    grouped_mm_ragged_2d_3d,
)
from torch_neuronx.utils import get_logical_neuron_cores


# Not tunable: the kernels shard the moving operand's free dim by nl.num_programs, so
# the grid must match or programs write overlapping output.
def _grid():
    return (int(get_logical_neuron_cores()),)


# Spare output rows for the 2Dx3D kernel's last token tile, sliced off by the caller.
_OUT_SLACK = 128


def grouped_mm_ragged_2d_3d_nki(
    a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    """Ragged grouped matmul: a (r, d1), b (g, d1, d2), offs (g,) -> (r, d2).

    `offs` may be arbitrary -- no alignment requirement, no padding.
    """
    # Returns r + _OUT_SLACK rows. The caller must slice: slicing inside the @nki_op
    # would make its shape inference under-allocate what the kernel writes.
    return wrap_nki(grouped_mm_ragged_2d_3d)[_grid()](a, b, offs)


def grouped_mm_ragged_2d_2d_nki(
    a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    """Ragged grouped matmul: a (d1, t), b (t, d2), offs (g,) -> (g, d1, d2).

    `offs` may be arbitrary -- no alignment requirement, no padding.
    """
    return wrap_nki(grouped_mm_ragged_2d_2d)[_grid()](a, b, offs)


@nki_op("nki_kernels::grouped_mm_ragged_2d_3d")
def grouped_mm_ragged_2d_3d_op(
    a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    return grouped_mm_ragged_2d_3d_nki(a, b, offs)


@nki_op("nki_kernels::grouped_mm_ragged_2d_2d")
def grouped_mm_ragged_2d_2d_op(
    a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    return grouped_mm_ragged_2d_2d_nki(a, b, offs)
