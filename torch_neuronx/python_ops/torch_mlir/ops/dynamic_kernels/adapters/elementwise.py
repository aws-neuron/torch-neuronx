"""Elementwise dynamic NKI adapters (silu / exp / sin / cos / sigmoid / log / neg /
ones_like / add / sub / mul / sigmoid_backward).

All share one shape scheme with M dynamic, differing only in name and operand
count. ``make_elementwise_entry`` builds the shared ``compile_fn`` +
``prepare_io``; each op keeps its own ``can_handle``. Binary add/sub/mul pad M
to one row tile per LNC program for row-interleaved sharding, as does ones_like —
the one member whose kernel never reads its operand.
"""

import torch

from torch_neuronx.utils import get_logical_neuron_cores

from ..adapter_utils import (
    _TILE_M,
    _can_handle_elementwise_2d,
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


def _can_handle_ones_like(x, dtype=None, **_) -> bool:
    """ones_like: same predicate as unary, but decline a dtype override — the NEFF
    bakes x's dtype. Only dtype is checkable: dispatch strips device/layout/
    memory_format (CompilationConfig.PYTORCH_SPECIFIC_PARAMS) before can_handle.
    """
    if dtype is not None and dtype != x.dtype:
        return False
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
    # Unsharded, every LNC program redundantly runs the whole kernel; ones_like is
    # pure store, so that duplicated traffic is the entire cost of the op.
    lnc_sharded = name in {"add", "sub", "mul", "ones_like"}
    num_programs = int(get_logical_neuron_cores()) if lnc_sharded else 1
    row_group = _TILE_M * num_programs
    # Ops whose kernel never DMAs its operand — it only bakes [M, H] + dtype.
    reads_input = name not in {"ones_like"}

    def compile_fn(compile_key):
        """Compile one row tile per LNC program; H + dtype are baked."""
        from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

        from .. import wrappers

        # Compile-only example inputs on the meta device: the kernel traces on
        # meta for shape inference anyway, so fabricating them on meta avoids a
        # stray aten::empty/zeros dispatch to the neuron device on cold compile.
        # Only shapes/dtypes matter here; the NEFF target comes from the compiler
        # config, not the example-input device.
        H, dtype = compile_key  # noqa: N806
        wrapper = getattr(wrappers, f"dynamic_{name}_nki")
        ins = tuple(torch.empty(row_group, H, dtype=dtype, device="meta") for _ in range(arity))
        nm = torch.tensor([[1]], dtype=torch.int32, device="meta")
        out = torch.empty(row_group, H, dtype=dtype, device="meta")

        kernel = TorchMlirKernel(torch_fn=wrapper, op_name=graph_name)
        return _compile_or_load_neff(kernel, graph_name, (*ins, nm), out)

    def prepare_io(*tensors, dest=None, **_):
        """Pad M so every LNC program executes the same number of row tiles."""
        x = tensors[0]
        H = x.shape[-1]  # noqa: N806
        M = x.numel() // H  # noqa: N806
        M_ceil = -(-M // row_group) * row_group  # noqa: N806
        device = x.device

        def to_2d(t):
            # _pad_or_contig avoids a redundant .contiguous() on the pad path (the
            # pad-assign reads a strided view directly), killing the per-shape
            # aten::contiguous recompiles on strided (e.g. .chunk()) inputs.
            return _pad_or_contig(t.reshape(-1, H), M_ceil)

        nm = _num_tiles_tensor(M_ceil // row_group, device)
        # Output buffer is M_ceil rows; a caller dest= (M*H) can only back it unpadded.
        dest_buf = dest if M_ceil == M else None
        outputs = (_dest_buffer(dest_buf, M_ceil, H, x.dtype, device),)

        # reads_input=False: to_2d(x) would allocate zeros[M_ceil, H] and copy a
        # tensor the kernel never reads. Pass the output buffer as arg0 instead —
        # same shape/dtype, already allocated, unread so the aliasing is safe.
        ins = tuple(to_2d(tensors[i]) for i in range(arity)) if reads_input else (outputs[0],)
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
    ("ones_like", _can_handle_ones_like, 1),
    ("add", _can_handle_add, 2),
    ("sub", _can_handle_sub, 2),
    ("mul", _can_handle_mul, 2),
    ("sigmoid_backward", _can_handle_sigmoid_backward, 2),
]


def register():
    for name, can_handle, arity in _ELEMENTWISE_OPS:
        register_dynamic_nki(name, make_elementwise_entry(name, can_handle, arity=arity))
