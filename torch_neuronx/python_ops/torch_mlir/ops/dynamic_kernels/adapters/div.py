"""div dynamic NKI adapter (plan #38: ``div.out`` / ``div.Tensor`` / ``div.Scalar``).

Three divisor forms — same-shape tensor, ``[*, 1]`` per-row column, Python scalar — one NEFF
each per (H, dtype), since the form is in the compile_key. M is runtime and unpadded, so one
NEFF covers every M and a contiguous ``out=`` is written in place; M >= 128 is required by the
tail tile. A scalar divisor reuses one NEFF whatever its value.
"""

import math

import torch

from torch_neuronx.nki_hop import nki_op, wrap_nki
from torch_neuronx.utils import get_logical_neuron_cores

from ..adapter_utils import (
    _SUPPORTED_DTYPES,
    _TILE_M,
    _as_2d,
    _compile_or_load_neff,
    _ctrl_tensor,
    _dest_buffer,
    _dest_reusable,
    _device_key,
    _dynamic_mh,
    _scalar_operand,
)
from ..div_kernel import div_row_kernel, div_scalar_kernel, div_tensor_kernel
from ..registry import DynamicNkiEntry, register_dynamic_nki

# Free-dim split per program, so the grid is logical_neuron_cores wide.
_GRID = (int(get_logical_neuron_cores()),)

_wrapped_tensor = wrap_nki(div_tensor_kernel)
_wrapped_row = wrap_nki(div_row_kernel)
_wrapped_scalar = wrap_nki(div_scalar_kernel)


@nki_op("dynamic_nki_div::tensor", mutates_args={})
def _div_tensor_nki(
    input_a: torch.Tensor, input_b: torch.Tensor, ctrl: torch.Tensor
) -> torch.Tensor:
    return _wrapped_tensor[_GRID](input_a, input_b, ctrl)


@nki_op("dynamic_nki_div::row", mutates_args={})
def _div_row_nki(input_a: torch.Tensor, divisor: torch.Tensor, ctrl: torch.Tensor) -> torch.Tensor:
    return _wrapped_row[_GRID](input_a, divisor, ctrl)


@nki_op("dynamic_nki_div::scalar", mutates_args={})
def _div_scalar_nki(
    input_a: torch.Tensor, divisor: torch.Tensor, ctrl: torch.Tensor
) -> torch.Tensor:
    return _wrapped_scalar[_GRID](input_a, divisor, ctrl)


_WRAPPERS = {"tensor": _div_tensor_nki, "row": _div_row_nki, "scalar": _div_scalar_nki}

# Memoized: torch.tensor(..., device="neuron") copies host->device on every call.
_DIVISOR_CACHE: dict = {}


def _divisor_column(other: float, device) -> torch.Tensor:
    """[128, 1] fp32 replication of ``other``: tensor_scalar's operand0 is per-partition."""
    key = (other, *_device_key(device))
    t = _DIVISOR_CACHE.get(key)
    if t is None:
        t = torch.full((_TILE_M, 1), other, dtype=torch.float32, device=device)
        _DIVISOR_CACHE[key] = t
    return t


def _mode(x, other):
    """Which kernel serves this divisor, or None if none can."""
    scalar = _scalar_operand(other)
    if scalar is not None:
        return "scalar" if math.isfinite(scalar) else None
    if other.dtype != x.dtype or other.device.type != "neuron":
        return None
    if other.shape == x.shape:
        return "tensor"
    if x.dim() >= 2 and other.shape == (*x.shape[:-1], 1):
        return "row"
    return None


def _can_handle_div(x, other, *, rounding_mode=None, out=None, **_) -> bool:
    """An ``out=`` we can't alias would need a stride-keyed ``copy_`` back, so decline it and
    let the static lowering honour it."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "neuron":
        return False
    if x.dtype not in _SUPPORTED_DTYPES:
        return False
    if rounding_mode is not None:  # floor/trunc division is a different op
        return False
    mode = _mode(x, other)
    if mode is None:
        return False
    if x.dim() == 0:
        return False
    M, H = _dynamic_mh(x)  # noqa: N806
    if M < _TILE_M:
        return False
    # Row mode needs one divisor per row of [M, H], which the [numel, 1] collapse breaks.
    if mode == "row" and other.numel() != M:
        return False
    return out is None or _dest_reusable(out, M, H, x.dtype, x.device)


def _compile_key(x, other, *, rounding_mode=None, out=None, **_):
    """(H, dtype, form). M and the scalar value are runtime, so neither is in the key."""
    _, H = _dynamic_mh(x)  # noqa: N806
    return (H, x.dtype, _mode(x, other))


def _compile_fn(compile_key):
    from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

    H, dtype, mode = compile_key  # noqa: N806
    # Distinct graph names: the NEFF cache_key is graph_name + input specs. Meta inputs keep
    # the cold compile from dispatching to the device.
    graph_name = f"dynamic_nki_div_{mode}"
    x = torch.empty(_TILE_M, H, dtype=dtype, device="meta")
    ctrl = torch.empty(2, 1, dtype=torch.int32, device="meta")
    out = torch.empty(_TILE_M, H, dtype=dtype, device="meta")
    if mode == "tensor":
        divisor = torch.empty(_TILE_M, H, dtype=dtype, device="meta")
    elif mode == "row":
        divisor = torch.empty(_TILE_M, 1, dtype=dtype, device="meta")
    else:
        divisor = torch.empty(_TILE_M, 1, dtype=torch.float32, device="meta")

    kernel = TorchMlirKernel(torch_fn=_WRAPPERS[mode], op_name=graph_name)
    return _compile_or_load_neff(kernel, graph_name, (x, divisor, ctrl), out)


def _prepare_io(x, other, *, dest=None, rounding_mode=None, **_):
    M, H = _dynamic_mh(x)  # noqa: N806
    device = x.device
    mode = _mode(x, other)
    if mode == "tensor":
        divisor = _as_2d(other, M, H)
    elif mode == "row":
        divisor = _as_2d(other, M, 1)
    else:
        divisor = _divisor_column(_scalar_operand(other), device)
    inputs = (_as_2d(x, M, H), divisor, _ctrl_tensor(M, device))
    # can_handle rejected any dest we can't alias, so dispatch's copy-back never fires.
    outputs = (_dest_buffer(dest, M, H, x.dtype, device),)
    return inputs, outputs, M


def register():
    """Register the dynamic tensor, row, and scalar division adapter."""
    register_dynamic_nki(
        "div",
        DynamicNkiEntry(
            compile_fn=_compile_fn,
            can_handle=_can_handle_div,
            prepare_io=_prepare_io,
            compile_key_fn=_compile_key,
        ),
    )
