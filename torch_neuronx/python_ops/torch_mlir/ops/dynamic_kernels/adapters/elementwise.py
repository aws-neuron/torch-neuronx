"""add / mul dynamic NKI adapters (M dynamic; H + dtype baked).

Both are binary elementwise: same [128, H]-per-(H,dtype) NEFF scheme as silu,
with a second operand of identical shape/dtype.
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


def _can_handle_add(x, y, alpha=1, out=None) -> bool:
    if alpha != 1:
        return False
    if not isinstance(y, torch.Tensor):
        return False
    return _can_handle_elementwise_2d(x) and x.shape == y.shape and y.dtype == x.dtype


def _compile_add(compile_key) -> tuple[bytes, str]:
    from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

    from ..wrappers import dynamic_add_nki

    H, dtype = compile_key  # noqa: N806
    a = torch.empty(_TILE_M, H, dtype=dtype, device="neuron")
    b = torch.empty(_TILE_M, H, dtype=dtype, device="neuron")
    nm = torch.tensor([[1]], dtype=torch.int32, device="neuron")
    out = torch.empty(_TILE_M, H, dtype=dtype, device="neuron")

    kernel = TorchMlirKernel(torch_fn=dynamic_add_nki, op_name="dynamic_nki_add")
    return _compile_or_load_neff(kernel, "dynamic_nki_add", (a, b, nm), out)


def _prepare_add_io(x, y, alpha=1, dest=None) -> tuple[tuple, tuple, int]:
    H = x.shape[-1]  # noqa: N806
    M = x.numel() // H  # noqa: N806
    device = x.device
    x_2d = x.reshape(-1, H).contiguous()
    y_2d = y.reshape(-1, H).contiguous()
    nm = _num_tiles_tensor(M // _TILE_M, device)
    inputs = (x_2d, y_2d, nm)
    outputs = (_dest_buffer(dest, M, H, x.dtype, device),)
    return inputs, outputs, M


def _can_handle_mul(x, y, out=None) -> bool:
    if not isinstance(y, torch.Tensor):
        return False
    return _can_handle_elementwise_2d(x) and x.shape == y.shape and y.dtype == x.dtype


def _compile_mul(compile_key) -> tuple[bytes, str]:
    from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

    from ..wrappers import dynamic_mul_nki

    H, dtype = compile_key  # noqa: N806
    a = torch.empty(_TILE_M, H, dtype=dtype, device="neuron")
    b = torch.empty(_TILE_M, H, dtype=dtype, device="neuron")
    nm = torch.tensor([[1]], dtype=torch.int32, device="neuron")
    out = torch.empty(_TILE_M, H, dtype=dtype, device="neuron")

    kernel = TorchMlirKernel(torch_fn=dynamic_mul_nki, op_name="dynamic_nki_mul")
    return _compile_or_load_neff(kernel, "dynamic_nki_mul", (a, b, nm), out)


def _prepare_mul_io(x, y, dest=None, **_) -> tuple[tuple, tuple, int]:
    H = x.shape[-1]  # noqa: N806
    M = x.numel() // H  # noqa: N806
    device = x.device
    x_2d = x.reshape(-1, H).contiguous()
    y_2d = y.reshape(-1, H).contiguous()
    nm = _num_tiles_tensor(M // _TILE_M, device)
    inputs = (x_2d, y_2d, nm)
    outputs = (_dest_buffer(dest, M, H, x.dtype, device),)
    return inputs, outputs, M


def register():
    register_dynamic_nki(
        "add",
        DynamicNkiEntry(
            compile_fn=_compile_add,
            can_handle=_can_handle_add,
            prepare_io=_prepare_add_io,
            compile_key_fn=_hidden_compile_key,
        ),
    )
    register_dynamic_nki(
        "mul",
        DynamicNkiEntry(
            compile_fn=_compile_mul,
            can_handle=_can_handle_mul,
            prepare_io=_prepare_mul_io,
            compile_key_fn=_hidden_compile_key,
        ),
    )
