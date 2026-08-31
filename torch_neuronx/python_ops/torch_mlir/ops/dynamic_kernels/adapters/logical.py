"""logical_and dynamic NKI adapter: out = bool(a) & bool(b)  (plan #56, .out variant).

Two kernels behind one entry, picked by operand dtype (which is in the compile_key): bool
operands go to ``packed_bool_kernel`` as int16 views with ``nl.bitwise_and`` (needs an even H),
float operands to the comparison family's ``binary_mask_kernel`` with ``nl.logical_and``.
Evaluating the float case on device avoids a host ``bool()`` cast, which is shape-keyed and
would recompile per token count. One NEFF per (H, dtype); M >= 128 is required by the tail tile.
"""

import nki.language as nl
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
    _dynamic_mh,
    _int16_dest_buffer,
)
from ..comparison_kernel import make_binary_mask_kernel
from ..packed_bool_kernel import make_packed_bool_kernel
from ..registry import DynamicNkiEntry, register_dynamic_nki

_GRAPH_NAME = "dynamic_nki_logical_and"

# Free-dim split per program, so the grid is logical_neuron_cores wide.
_GRID = (int(get_logical_neuron_cores()),)

_wrapped_packed = wrap_nki(make_packed_bool_kernel(nl.bitwise_and))
_wrapped_float = wrap_nki(make_binary_mask_kernel(nl.logical_and))


@nki_op("dynamic_nki_logical::and_packed", mutates_args={})
def _logical_and_packed_nki(
    input_a: torch.Tensor, input_b: torch.Tensor, ctrl: torch.Tensor
) -> torch.Tensor:
    return _wrapped_packed[_GRID](input_a, input_b, ctrl)


@nki_op("dynamic_nki_logical::and_float", mutates_args={})
def _logical_and_float_nki(
    input_a: torch.Tensor, input_b: torch.Tensor, ctrl: torch.Tensor
) -> torch.Tensor:
    return _wrapped_float[_GRID](input_a, input_b, ctrl)


def _can_handle_logical_and(input_a, input_b, out=None, **_) -> bool:
    """bool needs an even H (the int16 view); an ``out=`` we can't alias would need a
    stride-keyed ``copy_`` back, so decline it and let the static lowering honour it."""
    if not (isinstance(input_a, torch.Tensor) and isinstance(input_b, torch.Tensor)):
        return False
    if input_a.device.type != "neuron" or input_b.device.type != "neuron":
        return False
    if input_a.dtype not in (torch.bool, *_SUPPORTED_DTYPES):
        return False
    # torch promotes mixed dtypes; one access pattern serves both operands here, so decline.
    if input_b.dtype != input_a.dtype or input_b.shape != input_a.shape:
        return False
    if input_a.dim() == 0:
        return False
    M, H = _dynamic_mh(input_a)  # noqa: N806
    if M < _TILE_M:
        return False
    if input_a.dtype is torch.bool and H % 2 != 0:
        return False
    return out is None or _dest_reusable(out, M, H, torch.bool, input_a.device)


def _compile_key(input_a, input_b, out=None, **_):
    """(H, dtype). M is runtime; dtype selects the kernel."""
    _, H = _dynamic_mh(input_a)  # noqa: N806
    return (H, input_a.dtype)


def _compile_fn(compile_key):
    from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

    H, dtype = compile_key  # noqa: N806
    packed = dtype is torch.bool
    # Distinct graph names: the NEFF cache_key is graph_name + input specs. Meta inputs keep
    # the cold compile from dispatching to the device.
    graph_name = f"{_GRAPH_NAME}_packed" if packed else _GRAPH_NAME
    width, in_dtype = (H // 2, torch.int16) if packed else (H, dtype)
    a = torch.empty(_TILE_M, width, dtype=in_dtype, device="meta")
    b = torch.empty(_TILE_M, width, dtype=in_dtype, device="meta")
    ctrl = torch.empty(2, 1, dtype=torch.int32, device="meta")
    out = torch.empty(_TILE_M, width, dtype=in_dtype if packed else torch.bool, device="meta")

    wrapper = _logical_and_packed_nki if packed else _logical_and_float_nki
    kernel = TorchMlirKernel(torch_fn=wrapper, op_name=graph_name)
    return _compile_or_load_neff(kernel, graph_name, (a, b, ctrl), out)


def _prepare_io(input_a, input_b, *, dest=None, **_):
    M, H = _dynamic_mh(input_a)  # noqa: N806
    device = input_a.device
    if input_a.dtype is torch.bool:
        # int16 views: half the width, same bytes.
        inputs = (
            _as_2d(input_a, M, H).view(torch.int16),
            _as_2d(input_b, M, H).view(torch.int16),
            _ctrl_tensor(M, device),
        )
        outputs = (_int16_dest_buffer(dest, M, H, device),)
    else:
        inputs = (_as_2d(input_a, M, H), _as_2d(input_b, M, H), _ctrl_tensor(M, device))
        outputs = (_dest_buffer(dest, M, H, torch.bool, device),)
    return inputs, outputs, M


def _reshape_output(outputs, original_args):
    """Restore the operand shape, viewing a packed int16 result back to bool (same bytes)."""
    x = original_args[0]
    result = outputs[0]
    if result.dtype is torch.int16:
        result = result.view(torch.bool)
    return result.reshape(x.shape)


def register():
    register_dynamic_nki(
        "logical_and",
        DynamicNkiEntry(
            compile_fn=_compile_fn,
            can_handle=_can_handle_logical_and,
            prepare_io=_prepare_io,
            reshape_output=_reshape_output,
            compile_key_fn=_compile_key,
        ),
    )
