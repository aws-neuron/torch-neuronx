"""Elementwise dynamic NKI adapters (silu / exp / sin / cos / add / mul).

All share one shape scheme — a [128, H] NEFF per (H, dtype), M dynamic — differing
only in name and operand count. ``make_elementwise_entry`` builds the shared
``compile_fn`` + ``prepare_io``; each op keeps its own ``can_handle``.
"""

import torch

from ..adapter_utils import (
    _TILE_M,
    _can_handle_elementwise_2d,
    _ceil_to_tile,
    _compile_or_load_neff,
    _dest_buffer,
    _hidden_compile_key,
    _num_tiles_tensor,
    _pad_or_contig,
)
from ..registry import DynamicNkiEntry, register_dynamic_nki


# ── per-op can_handle (kept separate; NOT merged into the factory) ──
def _can_handle_unary(x, out=None) -> bool:
    """silu / exp / sin / cos: one 2-D+ neuron tensor, supported dtype, M % 128 == 0."""
    return _can_handle_elementwise_2d(x)


def _can_handle_add(x, y, alpha=1, out=None) -> bool:
    if alpha != 1:  # the a+b kernel can't honor alpha
        return False
    if not isinstance(y, torch.Tensor):
        return False
    return _can_handle_elementwise_2d(x) and x.shape == y.shape and y.dtype == x.dtype


def _can_handle_sub(x, y, alpha=1, out=None) -> bool:
    if alpha != 1:  # the a-b kernel can't honor alpha
        return False
    if not isinstance(y, torch.Tensor):
        return False
    return _can_handle_elementwise_2d(x) and x.shape == y.shape and y.dtype == x.dtype


def _can_handle_sigmoid_backward(grad_output, output, out=None) -> bool:
    if not isinstance(output, torch.Tensor):
        return False
    return (
        _can_handle_elementwise_2d(grad_output)
        and grad_output.shape == output.shape
        and output.dtype == grad_output.dtype
    )


def _can_handle_mul(x, y, out=None) -> bool:
    if not isinstance(y, torch.Tensor):
        return False
    return _can_handle_elementwise_2d(x) and x.shape == y.shape and y.dtype == x.dtype


# ── factory: the shared compile_fn + prepare_io (the mechanical half) ──
def make_elementwise_entry(name, can_handle, *, arity=1):
    """Build a DynamicNkiEntry for an elementwise op. ``arity`` is the operand
    count (1 unary, 2 binary); ``can_handle`` is supplied per op."""
    graph_name = f"dynamic_nki_{name}"

    def compile_fn(compile_key):
        """Compile at [128, H] for a (H, dtype) — H + dtype baked, M dynamic."""
        from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

        from .. import wrappers

        # Compile-only example inputs on the meta device: the kernel traces on
        # meta for shape inference anyway, so fabricating them on meta avoids a
        # stray aten::empty/zeros dispatch to the neuron device on cold compile.
        # Only shapes/dtypes matter here; the NEFF target comes from the compiler
        # config, not the example-input device.
        H, dtype = compile_key  # noqa: N806
        wrapper = getattr(wrappers, f"dynamic_{name}_nki")
        ins = tuple(torch.empty(_TILE_M, H, dtype=dtype, device="meta") for _ in range(arity))
        nm = torch.tensor([[1]], dtype=torch.int32, device="meta")
        out = torch.empty(_TILE_M, H, dtype=dtype, device="meta")

        kernel = TorchMlirKernel(torch_fn=wrapper, op_name=graph_name)
        return _compile_or_load_neff(kernel, graph_name, (*ins, nm), out)

    def prepare_io(*tensors, dest=None, **_):
        """Pad the M rows up to a 128-multiple (M_ceil); sliced back in reshape_output."""
        x = tensors[0]
        H = x.shape[-1]  # noqa: N806
        M = x.numel() // H  # noqa: N806
        M_ceil = _ceil_to_tile(M)  # noqa: N806
        device = x.device

        def to_2d(t):
            # _pad_or_contig avoids a redundant .contiguous() on the pad path (the
            # pad-assign reads a strided view directly), killing the per-shape
            # aten::contiguous recompiles on strided (e.g. .chunk()) inputs.
            return _pad_or_contig(t.reshape(-1, H), M_ceil)

        ins = tuple(to_2d(tensors[i]) for i in range(arity))
        nm = _num_tiles_tensor(M_ceil // _TILE_M, device)
        # Output buffer is M_ceil rows; a caller dest= (M*H) can only back it unpadded.
        dest_buf = dest if M_ceil == M else None
        outputs = (_dest_buffer(dest_buf, M_ceil, H, x.dtype, device),)
        return (*ins, nm), outputs, M

    def reshape_output(outputs, original_args):
        """Slice off the padded rows, restore the input shape."""
        x = original_args[0]
        H = x.shape[-1]  # noqa: N806
        M = x.numel() // H  # noqa: N806
        return outputs[0][:M].reshape(x.shape)

    return DynamicNkiEntry(
        compile_fn=compile_fn,
        can_handle=can_handle,
        prepare_io=prepare_io,
        reshape_output=reshape_output,
        compile_key_fn=_hidden_compile_key,
    )


# one row per elementwise op: (name, can_handle, arity)
_ELEMENTWISE_OPS = [
    ("silu", _can_handle_unary, 1),
    ("exp", _can_handle_unary, 1),
    ("sin", _can_handle_unary, 1),
    ("cos", _can_handle_unary, 1),
    ("sigmoid", _can_handle_unary, 1),
    ("log", _can_handle_unary, 1),
    ("neg", _can_handle_unary, 1),
    ("add", _can_handle_add, 2),
    ("sub", _can_handle_sub, 2),
    ("mul", _can_handle_mul, 2),
    ("sigmoid_backward", _can_handle_sigmoid_backward, 2),
]


def register():
    for name, can_handle, arity in _ELEMENTWISE_OPS:
        register_dynamic_nki(name, make_elementwise_entry(name, can_handle, arity=arity))
