"""silu dynamic NKI adapter (M dynamic; H + dtype baked).

Compiles at [128, H] for each (H, dtype); reused for any M that tiles by 128.
"""

import torch

from ..adapter_utils import (
    _TILE_M,
    _can_handle_elementwise_2d,
    _compile_or_load_neff,
    _dest_buffer,
    _hidden_compile_key,
    _num_tiles_tensor,
)
from ..registry import DynamicNkiEntry, register_dynamic_nki


def _can_handle_silu(x, out=None) -> bool:
    return _can_handle_elementwise_2d(x)


def _compile_silu(compile_key) -> tuple[bytes, str]:
    """Compile silu at [128, H] for a dtype (H+dtype baked, M dynamic)."""
    from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

    from ..wrappers import dynamic_silu_nki

    H, dtype = compile_key  # noqa: N806
    inp = torch.empty(_TILE_M, H, dtype=dtype, device="neuron")
    num_m_tiles = torch.tensor([[1]], dtype=torch.int32, device="neuron")
    out = torch.empty(_TILE_M, H, dtype=dtype, device="neuron")

    kernel = TorchMlirKernel(torch_fn=dynamic_silu_nki, op_name="dynamic_nki_silu")
    return _compile_or_load_neff(kernel, "dynamic_nki_silu", (inp, num_m_tiles), out)


def _prepare_silu_io(x, dest=None, **_) -> tuple[tuple, tuple, int]:
    """Actual-sized IO (M is a 128-multiple; no padding)."""
    H = x.shape[-1]  # noqa: N806
    M = x.numel() // H  # noqa: N806
    device = x.device
    x_2d = x.reshape(-1, H).contiguous()
    num_m_tiles = _num_tiles_tensor(M // _TILE_M, device)
    outputs = (_dest_buffer(dest, M, H, x.dtype, device),)
    return (x_2d, num_m_tiles), outputs, M


def register():
    register_dynamic_nki(
        "silu",
        DynamicNkiEntry(
            compile_fn=_compile_silu,
            can_handle=_can_handle_silu,
            prepare_io=_prepare_silu_io,
            compile_key_fn=_hidden_compile_key,
        ),
    )
