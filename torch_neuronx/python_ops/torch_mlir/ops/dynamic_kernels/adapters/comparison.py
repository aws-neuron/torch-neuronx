"""Comparison-with-scalar dynamic NKI adapters (ge / lt / ne / le .Scalar_out).

Bool output and a scalar second operand, so these don't ride the unary elementwise
factory. One NEFF per (H, dtype) covers every M and threshold; nothing is padded, so a
contiguous ``out=`` is written in place. Needs M >= 128; an ``out=`` we can't alias
declines to the static lowering.
"""

import math

import torch

from torch_neuronx.nki_hop import nki_op, wrap_nki
from torch_neuronx.utils import get_logical_neuron_cores

from ..adapter_utils import (
    _SUPPORTED_DTYPES,
    _TILE_M,
    _compile_or_load_neff,
    _dest_buffer,
    _dest_reusable,
)
from ..comparison_kernel import _CMP_OPS, make_comparison_kernel
from ..registry import DynamicNkiEntry, register_dynamic_nki

# The kernel splits the free dim per program, so the grid is logical_neuron_cores wide
# (also needed for identical per-core basic-block counts).
_GRID = (int(get_logical_neuron_cores()),)

# One wrapper per op, built at import: custom_op registration is global.
_WRAPPERS = {}
for _name, _op in _CMP_OPS.items():
    _wrapped = wrap_nki(make_comparison_kernel(_op))

    def _make_wrapper(name, wrapped):
        @nki_op(f"dynamic_nki_cmp::{name}", mutates_args={})
        def wrapper(
            input_tensor: torch.Tensor, threshold: torch.Tensor, ctrl: torch.Tensor
        ) -> torch.Tensor:
            return wrapped[_GRID](input_tensor, threshold, ctrl)

        return wrapper

    _WRAPPERS[_name] = _make_wrapper(_name, _wrapped)

# torch.tensor(..., device="neuron") copies host->device every call, but both operands
# depend only on scalars. Read-only in the kernel.
_THRESHOLD_CACHE: dict = {}
_CTRL_CACHE: dict = {}


def _device_key(device):
    return (device.type, device.index if device.index is not None else -1)


def _threshold_column(other: float, device) -> torch.Tensor:
    """[128, 1] fp32 replication of ``other``: nisa.tensor_scalar needs operand0's
    partition count to match the destination tile's, so a [1, 1] scalar won't trace."""
    key = (other, *_device_key(device))
    t = _THRESHOLD_CACHE.get(key)
    if t is None:
        t = torch.full((_TILE_M, 1), other, dtype=torch.float32, device=device)
        _THRESHOLD_CACHE[key] = t
    return t


def _ctrl_tensor(m: int, device) -> torch.Tensor:
    """[2, 1] int32 = [[whole 128-row tiles], [tail tile's row offset = M - 128]]."""
    key = (m, *_device_key(device))
    t = _CTRL_CACHE.get(key)
    if t is None:
        t = torch.tensor([[m // _TILE_M], [m - _TILE_M]], dtype=torch.int32, device=device)
        _CTRL_CACHE[key] = t
    return t


def _mh(x):
    """(M, H) with the runtime-dynamic dim on M; H is baked, M is not. A leading product
    of >= one row tile keeps the last dim as H ([6144, 720]); otherwise the last dim is
    the varying one, so collapse to [numel, 1] ([1, seq] / [seq])."""
    n = x.numel()
    last = x.shape[-1]
    if x.dim() >= 2 and last != 0 and (n // last) >= _TILE_M:
        return n // last, last
    return n, 1


def _can_handle_compare(x, other, out=None, **_) -> bool:
    """Neuron tensor of a supported dtype vs a Python scalar, flattening to at least one
    row tile. An ``out=`` we can't alias would need a stride-keyed ``copy_`` back (one
    compile per row count), so decline it and let the static lowering honour it."""
    if not isinstance(x, torch.Tensor):
        return False
    if isinstance(other, torch.Tensor):  # only the .Scalar / .Scalar_out overloads
        return False
    if x.device.type != "neuron":
        return False
    if x.dtype not in _SUPPORTED_DTYPES:
        return False
    if not math.isfinite(float(other)):  # NaN/inf ALU semantics unverified
        return False
    M, H = _mh(x)  # noqa: N806
    if M < _TILE_M:
        return False
    return out is None or _dest_reusable(out, M, H, torch.bool, x.device)


def _compile_key(x, other, out=None, **_):
    """(H, dtype). M and the threshold are runtime, so neither is in the key."""
    _, H = _mh(x)  # noqa: N806
    return (H, x.dtype)


def _make_compare_entry(name):
    graph_name = f"dynamic_nki_{name}"

    def compile_fn(compile_key):
        from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

        H, dtype = compile_key  # noqa: N806
        # Meta inputs: only shapes/dtypes matter for tracing, and meta avoids a stray
        # dispatch to the neuron device on cold compile.
        x = torch.empty(_TILE_M, H, dtype=dtype, device="meta")
        thr = torch.empty(_TILE_M, 1, dtype=torch.float32, device="meta")
        ctrl = torch.empty(2, 1, dtype=torch.int32, device="meta")
        out = torch.empty(_TILE_M, H, dtype=torch.bool, device="meta")

        kernel = TorchMlirKernel(torch_fn=_WRAPPERS[name], op_name=graph_name)
        return _compile_or_load_neff(kernel, graph_name, (x, thr, ctrl), out)

    def prepare_io(x, other, *, dest=None, **_):
        M, H = _mh(x)  # noqa: N806
        device = x.device
        x_2d = x.reshape(M, H)
        if not x_2d.is_contiguous():
            # The only shape-keyed op left on this path; contiguous inputs skip it.
            x_2d = x_2d.contiguous()
        inputs = (x_2d, _threshold_column(float(other), device), _ctrl_tensor(M, device))
        # Unpadded, and can_handle rejected a dest we can't alias, so dest= always backs
        # the output here and dispatch's copy-back never fires.
        outputs = (_dest_buffer(dest, M, H, torch.bool, device),)
        return inputs, outputs, M

    return DynamicNkiEntry(
        compile_fn=compile_fn,
        can_handle=_can_handle_compare,
        prepare_io=prepare_io,
        compile_key_fn=_compile_key,
    )


def register():
    for name in _CMP_OPS:
        register_dynamic_nki(name, _make_compare_entry(name))
