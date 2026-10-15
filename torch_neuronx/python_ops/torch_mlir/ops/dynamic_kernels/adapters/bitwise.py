"""bitwise_or.Tensor dynamic NKI adapter: out = a | b.

Operands share one shape and dtype. The target workload uses it to combine bool
attention masks, so the supported dtype is bool (bitwise OR == logical OR on
0/1). NEFF keyed on (H, dtype), M dynamic.

A 1-byte (bool) tensor cannot be DMA'd with the per-tile dynamic offset (only the
offset-0 tile reads correctly), so both operands and the output are passed as
int16 views (zero-copy). Bitwise OR is bit-parallel, so OR-ing the int16-packed
pairs equals OR-ing the underlying bools. Requires an even H.

That int16 view is why this is bespoke rather than a ``make_elementwise_entry``
row: every buffer — operands and output — is int16 at half the width, and the
result has to be viewed back to bool.
"""

import torch

from torch_neuronx.utils import get_logical_neuron_cores

from ..adapter_utils import (
    _TILE_M,
    _compile_or_load_neff,
    _hidden_compile_key,
    _int16_dest_buffer,
    _num_tiles_tensor,
    _pad_or_contig,
)
from ..registry import DynamicNkiEntry, register_dynamic_nki

_GRAPH_NAME = "dynamic_nki_bitwise_or"

# Row-interleaved LNC sharding, same scheme as the binary elementwise ops: M is
# padded to one row tile per program, and the NEFF is traced at that many rows.
_NUM_PROGRAMS = int(get_logical_neuron_cores())
_ROW_GROUP = _TILE_M * _NUM_PROGRAMS

# Only bool is supported/tested for now (the workload's attention masks). Integer
# dtypes would work in the kernel but are intentionally left out until tested.
_BITWISE_DTYPES = (torch.bool,)


def _can_handle_bitwise_or(input_a, input_b, out=None, **_) -> bool:
    """bitwise_or: two operands of one identical shape/dtype (bool), on neuron,
    M > 0, even H (the operands are viewed as int16 to DMA-load them)."""
    if not (isinstance(input_a, torch.Tensor) and isinstance(input_b, torch.Tensor)):
        return False
    if input_a.device.type != "neuron":
        return False
    if input_a.dtype not in _BITWISE_DTYPES or input_b.dtype != input_a.dtype:
        return False
    if input_a.dim() < 2 or input_a.shape != input_b.shape:
        return False
    H = input_a.shape[-1]  # noqa: N806
    M = input_a.numel() // H  # noqa: N806
    if M == 0 or H == 0:
        return False
    if H % 2 != 0:  # noqa: SIM103
        return False
    return True


def _compile_fn(compile_key):
    """Compile one row tile per LNC program at [_ROW_GROUP, H//2] int16 (the
    operands' int16 view) — H + dtype baked."""
    from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

    from .. import wrappers

    H, _dtype = compile_key  # noqa: N806
    a = torch.empty(_ROW_GROUP, H // 2, dtype=torch.int16, device="meta")
    b = torch.empty(_ROW_GROUP, H // 2, dtype=torch.int16, device="meta")
    nm = torch.tensor([[1]], dtype=torch.int32, device="meta")
    out = torch.empty(_ROW_GROUP, H // 2, dtype=torch.int16, device="meta")

    kernel = TorchMlirKernel(torch_fn=wrappers.dynamic_bitwise_or_nki, op_name=_GRAPH_NAME)
    return _compile_or_load_neff(kernel, _GRAPH_NAME, (a, b, nm), out)


def _prepare_io(input_a, input_b, dest=None, **_):
    """Pad rows so every LNC program executes the same number of row tiles, view
    both operands + the output as int16 ([M_ceil, H//2]); reshape_output views the
    int16 result back to bool."""
    H = input_a.shape[-1]  # noqa: N806
    M = input_a.numel() // H  # noqa: N806
    M_ceil = -(-M // _ROW_GROUP) * _ROW_GROUP  # noqa: N806
    device = input_a.device

    def to_16(t):
        return _pad_or_contig(t.reshape(-1, H), M_ceil).view(torch.int16)

    a16 = to_16(input_a)
    b16 = to_16(input_b)
    nm = _num_tiles_tensor(M_ceil // _ROW_GROUP, device)
    # Output buffer is M_ceil rows; a caller dest= (M*H) can only back it unpadded.
    dest_buf = dest if M_ceil == M else None
    outputs = (_int16_dest_buffer(dest_buf, M_ceil, H, device),)
    return (a16, b16, nm), outputs, M


def _reshape_output(outputs, original_args):
    """View the int16 result back to bool ([M_ceil, H]), slice off padded rows,
    restore the operand shape."""
    x = original_args[0]
    H = x.shape[-1]  # noqa: N806
    M = x.numel() // H  # noqa: N806
    return outputs[0].view(torch.bool)[:M].reshape(x.shape)


def register():
    register_dynamic_nki(
        "bitwise_or",
        DynamicNkiEntry(
            compile_fn=_compile_fn,
            can_handle=_can_handle_bitwise_or,
            prepare_io=_prepare_io,
            reshape_output=_reshape_output,
            compile_key_fn=_hidden_compile_key,  # (H, dtype) off arg0, as elementwise
        ),
    )
