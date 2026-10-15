"""where.self dynamic NKI adapter: out = torch.where(condition, a, b).

`condition` is bool; `a` and `b` share one shape and a supported float dtype. The
[128, H] NEFF is keyed on (H, a.dtype) with M dynamic — the same scheme as the
elementwise ops. Bespoke rather than a ``make_elementwise_entry`` row (like
concat / reduction / gather / scatter) because the factory pads every operand at
arg0's dtype and width, and arg0 here is a half-width int16 view of a bool mask.
Broadcasting is NOT handled: ``can_handle``
requires the three operands to share one shape, so scalar-``other`` / broadcast
cases fall back to the default lowering.

A 1-byte (bool) tensor cannot be DMA'd with the per-tile dynamic offset — only
the offset-0 tile reads correctly, every later tile's byte address is
miscomputed. So the condition is passed to the kernel as an int16 view
([m, H//2], zero-copy); the kernel DMAs it as 2-byte elements (correct dynamic
offset) and reinterprets to uint8 in SBUF. This requires H to be even.
"""

import torch

from torch_neuronx.utils import get_logical_neuron_cores

from ..adapter_utils import (
    _SUPPORTED_DTYPES,
    _TILE_M,
    _compile_or_load_neff,
    _dest_buffer,
    _num_tiles_tensor,
    _pad_or_contig,
)
from ..registry import DynamicNkiEntry, register_dynamic_nki

_GRAPH_NAME = "dynamic_nki_where"

# Row-interleaved LNC sharding, same scheme as the binary elementwise ops: M is
# padded to one row tile per program, and the NEFF is traced at that many rows.
_NUM_PROGRAMS = int(get_logical_neuron_cores())
_ROW_GROUP = _TILE_M * _NUM_PROGRAMS


def _can_handle_where(condition, input_a, input_b, out=None, **_) -> bool:
    """where: bool condition + two float operands of one identical shape/dtype,
    on neuron, M > 0. No broadcast (all three shapes must match); the isinstance
    checks are what decline the where.Scalar* overloads, which land here too.
    """
    if not (
        isinstance(condition, torch.Tensor)
        and isinstance(input_a, torch.Tensor)
        and isinstance(input_b, torch.Tensor)
    ):
        return False
    if condition.device.type != "neuron":
        return False
    if condition.dtype != torch.bool:
        return False
    if input_a.dtype not in _SUPPORTED_DTYPES or input_b.dtype != input_a.dtype:
        return False
    if input_a.dim() < 2:
        return False
    if not (condition.shape == input_a.shape == input_b.shape):
        return False
    H = input_a.shape[-1]  # noqa: N806
    M = input_a.numel() // H  # noqa: N806
    if M == 0 or H == 0:
        return False
    if H % 2 != 0:  # noqa: SIM103
        # condition is viewed as int16 to DMA-load it; needs an even H.
        return False
    return True


def _compile_key(condition, input_a, input_b, **_):
    """compile_key = (H, a.dtype) — both baked into the NEFF (M dynamic).
    Keyed off input_a (the output dtype), NOT the bool condition."""
    return (int(input_a.shape[-1]), input_a.dtype)


def _compile_fn(compile_key):
    """Compile one row tile per LNC program for a (H, dtype): condition bool (int16
    view), a/b that dtype."""
    from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

    from .. import wrappers

    H, dtype = compile_key  # noqa: N806
    # condition arrives int16-viewed ([_ROW_GROUP, H//2]); see module docstring.
    cond = torch.empty(_ROW_GROUP, H // 2, dtype=torch.int16, device="meta")
    a = torch.empty(_ROW_GROUP, H, dtype=dtype, device="meta")
    b = torch.empty(_ROW_GROUP, H, dtype=dtype, device="meta")
    nm = torch.tensor([[1]], dtype=torch.int32, device="meta")
    out = torch.empty(_ROW_GROUP, H, dtype=dtype, device="meta")

    kernel = TorchMlirKernel(torch_fn=wrappers.dynamic_where_nki, op_name=_GRAPH_NAME)
    return _compile_or_load_neff(kernel, _GRAPH_NAME, (cond, a, b, nm), out)


def _prepare_io(condition, input_a, input_b, dest=None, **_):
    """Pad the M rows of all three operands so every LNC program executes the same
    number of row tiles; the output is sliced back in reshape_output."""
    H = input_a.shape[-1]  # noqa: N806
    M = input_a.numel() // H  # noqa: N806
    M_ceil = -(-M // _ROW_GROUP) * _ROW_GROUP  # noqa: N806
    device = input_a.device

    def to_2d(t):
        """Flatten and pad an operand to the shared kernel shape.

        Args:
            t (torch.Tensor): Operand to reshape and pad.

        Returns:
            torch.Tensor: Contiguous or padded tensor with shape ``[M_ceil, H]``.
        """
        return _pad_or_contig(t.reshape(-1, H), M_ceil)

    # Pad/contiguify the bool condition, then view it as int16 ([M_ceil, H//2]) so
    # the kernel's DMA uses a 2-byte element (1-byte dynamic-offset DMA is broken).
    cond2d = to_2d(condition).view(torch.int16)
    a2d = to_2d(input_a)
    b2d = to_2d(input_b)
    nm = _num_tiles_tensor(M_ceil // _ROW_GROUP, device)
    dest_buf = dest if M_ceil == M else None
    outputs = (_dest_buffer(dest_buf, M_ceil, H, input_a.dtype, device),)
    return (cond2d, a2d, b2d, nm), outputs, M


def _reshape_output(outputs, original_args):
    """Slice off the padded rows, restore the operand shape (from input_a)."""
    a = original_args[1]
    H = a.shape[-1]  # noqa: N806
    M = a.numel() // H  # noqa: N806
    return outputs[0][:M].reshape(a.shape)


def register():
    """Register the dynamic element-selection adapter."""
    register_dynamic_nki(
        "where",
        DynamicNkiEntry(
            compile_fn=_compile_fn,
            can_handle=_can_handle_where,
            prepare_io=_prepare_io,
            reshape_output=_reshape_output,
            compile_key_fn=_compile_key,
        ),
    )
