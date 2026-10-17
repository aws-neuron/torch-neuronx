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
    """Run ragged 2D-by-3D grouped matrix multiplication through NKI.

    Args:
        a (torch.Tensor): Token matrix with shape ``(R, D1)``.
        b (torch.Tensor): Expert matrices with shape ``(E, D1, D2)``.
        offs (torch.Tensor): Arbitrary cumulative expert end offsets.

    Returns:
        torch.Tensor: Result with shape ``(R + 128, D2)`` including spare tail rows.

    Raises:
        AssertionError: If ``D2`` is not divisible by the launch program count.

    Note:
        Callers must trim the final 128 rows; offsets require no alignment or padding.
    """
    # Returns r + _OUT_SLACK rows. The caller must slice: slicing inside the @nki_op
    # would make its shape inference under-allocate what the kernel writes.
    return wrap_nki(grouped_mm_ragged_2d_3d)[_grid()](a, b, offs)


def grouped_mm_ragged_2d_2d_nki(
    a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    """Run ragged 2D-by-2D grouped matrix multiplication through NKI.

    Args:
        a (torch.Tensor): Left operand with shape ``(D1, T)``.
        b (torch.Tensor): Right operand with shape ``(T, D2)``.
        offs (torch.Tensor): Arbitrary cumulative expert end offsets.

    Returns:
        torch.Tensor: Per-expert result with shape ``(E, D1, D2)``.

    Raises:
        AssertionError: If ``D2`` is not divisible by the launch program count.

    Note:
        Offsets require no alignment or padding.
    """
    return wrap_nki(grouped_mm_ragged_2d_2d)[_grid()](a, b, offs)


@nki_op("nki_kernels::grouped_mm_ragged_2d_3d")
def grouped_mm_ragged_2d_3d_op(
    a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    """Expose ragged 2D-by-3D grouped multiplication as an NKI op.

    Args:
        a (torch.Tensor): Token matrix with shape ``(R, D1)``.
        b (torch.Tensor): Expert matrices with shape ``(E, D1, D2)``.
        offs (torch.Tensor): Arbitrary cumulative expert end offsets.

    Returns:
        torch.Tensor: Result with shape ``(R + 128, D2)`` including spare tail rows.

    Raises:
        AssertionError: If ``D2`` is not divisible by the launch program count.

    Note:
        Callers must trim the final 128 rows.
    """
    return grouped_mm_ragged_2d_3d_nki(a, b, offs)


@nki_op("nki_kernels::grouped_mm_ragged_2d_2d")
def grouped_mm_ragged_2d_2d_op(
    a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    """Expose ragged 2D-by-2D grouped multiplication as an NKI op.

    Args:
        a (torch.Tensor): Left operand with shape ``(D1, T)``.
        b (torch.Tensor): Right operand with shape ``(T, D2)``.
        offs (torch.Tensor): Arbitrary cumulative expert end offsets.

    Returns:
        torch.Tensor: Per-expert result with shape ``(E, D1, D2)``.

    Raises:
        AssertionError: If ``D2`` is not divisible by the launch program count.

    Note:
        Offsets require no alignment or padding.
    """
    return grouped_mm_ragged_2d_2d_nki(a, b, offs)
