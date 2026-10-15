"""silu_backward dynamic NKI adapter (M dynamic; H + dtype baked).

aten::silu_backward(grad_output, self) -> grad_input, elementwise:
    grad_input = grad_output * silu_dx(self)
Both operands are [M, H]; M (tokens) is the dynamic dim, H + dtype baked. The
kernel uses the same dynamic_range + scalar_offset structure as the fwd silu
kernel, so ONE NEFF per (H, dtype) serves any M that tiles by 128. H is arbitrary
(the kernel tiles the free dim at trace time: 512-wide tiles + a short remainder).
"""

import torch

from ..adapter_utils import (
    _TILE_M,
    _ceil_to_tile,
    _compile_or_load_neff,
    _num_tiles_tensor,
    _pad_or_contig,
)
from ..registry import DynamicNkiEntry, register_dynamic_nki


def _can_handle_silu_bwd(grad_output, self, out=None) -> bool:
    for t in (grad_output, self):
        if not isinstance(t, torch.Tensor) or t.device.type != "neuron":
            return False
        if t.dim() != 2:
            return False
    if grad_output.shape != self.shape or grad_output.dtype != self.dtype:
        return False
    M, H = grad_output.shape  # noqa: N806
    if M == 0 or H == 0:  # noqa: SIM103  (any M > 0 — prepare_io pads rows; H arbitrary)
        return False
    return True


def _silu_bwd_compile_key(grad_output, self, out=None):
    """compile_key = (H, dtype), both baked; M runtime-dynamic."""
    return (int(grad_output.shape[1]), grad_output.dtype)


def _compile_silu_bwd(compile_key) -> tuple[bytes, str]:
    from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

    from ..wrappers import dynamic_silu_bwd_nki

    H, dtype = compile_key  # noqa: N806
    grad = torch.empty(_TILE_M, H, dtype=dtype, device="meta")
    inp = torch.empty(_TILE_M, H, dtype=dtype, device="meta")
    num_m_tiles = torch.tensor([[1]], dtype=torch.int32, device="meta")
    out = torch.empty(_TILE_M, H, dtype=dtype, device="meta")

    kernel = TorchMlirKernel(torch_fn=dynamic_silu_bwd_nki, op_name="dynamic_nki_silu_bwd")
    return _compile_or_load_neff(kernel, "dynamic_nki_silu_bwd", (grad, inp, num_m_tiles), out)


def _prepare_silu_bwd_io(grad_output, self, dest=None) -> tuple[tuple, tuple, int]:
    """Pad both operands' rows up to M_ceil (next 128-multiple) so num_m_tiles
    walks whole tiles; padded rows are sliced off in reshape_output."""
    M, H = grad_output.shape  # noqa: N806
    M_ceil = _ceil_to_tile(M)  # noqa: N806
    device = grad_output.device
    grad_2d = _pad_or_contig(grad_output, M_ceil)
    inp_2d = _pad_or_contig(self, M_ceil)
    num_m_tiles = _num_tiles_tensor(M_ceil // _TILE_M, device)
    outputs = (torch.empty((M_ceil, H), dtype=grad_output.dtype, device=device),)
    return (grad_2d, inp_2d, num_m_tiles), outputs, M


def _reshape_silu_bwd_output(outputs, original_args):
    return outputs[0][: original_args[0].shape[0]].reshape(original_args[0].shape)


def register():
    """Register the dynamic SiLU-backward adapter."""
    register_dynamic_nki(
        "silu_backward",
        DynamicNkiEntry(
            compile_fn=_compile_silu_bwd,
            can_handle=_can_handle_silu_bwd,
            prepare_io=_prepare_silu_bwd_io,
            reshape_output=_reshape_silu_bwd_output,
            compile_key_fn=_silu_bwd_compile_key,
        ),
    )
