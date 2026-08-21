"""Dtype-cast / copy dynamic NKI adapters (``aten::_to_copy``, ``aten::copy_``).

One kernel pass — read [M, H] in the source dtype, write it in the destination
dtype — into a fresh tensor for ``_to_copy`` or into ``self`` for ``copy_``.

**Neither path pads the rows**: the kernel covers a ragged M with an overlapping
tail tile (see ``cast_kernel``), so it needs no pad buffer and no over-allocated
output, which is what makes the in-place ``copy_`` legal at any M >= 128. Both
``_to_copy`` and ``copy_`` require M >= 128 (the tail's ``M - 128`` offset would go
negative for a sub-tile M); sub-128 casts decline to the native path.

compile_key = (H, src_dtype, dst_dtype), all baked; M is dynamic.
"""

import torch

from ..adapter_utils import (
    _SUPPORTED_DTYPES,
    _TILE_M,
    _compile_or_load_neff,
    _dest_buffer,
    _num_tiles_tensor,
)
from ..registry import DynamicNkiEntry, register_dynamic_nki

# Target dtype -> the @nki_op wrapper that bakes it (see wrappers.py).
_WRAPPER_SUFFIX = {
    torch.bfloat16: "bf16",
    torch.float16: "fp16",
    torch.float32: "fp32",
}


def _mh(x):
    """(M, H): leading dims -> dynamic M, last dim -> baked H. Rank 1 becomes
    (numel, 1) — the length itself is dynamic."""
    n = x.numel()
    if x.dim() >= 2:
        last = x.shape[-1]
        return n // last, last
    return n, 1


def _is_castable(t) -> bool:
    """A contiguous, non-empty, non-scalar neuron tensor of a supported dtype.
    Strided operands are beyond a contiguous-tile kernel, so they decline."""
    return (
        isinstance(t, torch.Tensor)
        and t.device.type == "neuron"
        and t.dtype in _SUPPORTED_DTYPES
        and t.dim() >= 1
        and t.numel() > 0
        and t.is_contiguous()
    )


# ── aten::_to_copy ───────────────────────────────────────────────────────────
def _target_dtype(dtype, out_dtype):
    """C++ dispatch forwards the aten ``dtype``, the Python path
    (``ToCopyMLIRImpl.execute``) renames it ``out_dtype``. None = no cast asked for."""
    return dtype if dtype is not None else out_dtype


def _can_handle_to_copy(self, dtype=None, out_dtype=None, non_blocking=False, **_) -> bool:
    """Neuron->neuron dtype cast of a contiguous, M >= 128 tensor. ``device`` /
    ``memory_format`` are filtered out before can_handle: the caller applies memory_format
    itself. No dtype change means a layout-only copy, which this kernel does not implement."""
    target = _target_dtype(dtype, out_dtype)
    if target is None or not isinstance(self, torch.Tensor) or target == self.dtype:
        return False
    if not (target in _SUPPORTED_DTYPES and _is_castable(self)):
        return False
    M, _ = _mh(self)  # noqa: N806
    return M >= _TILE_M


def _to_copy_compile_key(self, dtype=None, out_dtype=None, non_blocking=False, **_):
    """(H, src_dtype, dst_dtype) — all baked into the NEFF; M dynamic."""
    _, H = _mh(self)  # noqa: N806
    return (H, self.dtype, _target_dtype(dtype, out_dtype))


def _prepare_to_copy_io(self, dtype=None, out_dtype=None, non_blocking=False, dest=None, **_):
    """Flatten to [M, H] and hand the kernel the real row count. M >= _TILE_M (can_handle),
    so the [M, H] view goes to the kernel unpadded; reshape_output just restores the shape."""
    target = _target_dtype(dtype, out_dtype)
    M, H = _mh(self)  # noqa: N806
    device = self.device

    # can_handle guarantees M >= _TILE_M, so no pad: the kernel covers the ragged tail with an
    # overlapping last tile and reads the contiguous [M, H] view directly.
    x = self.reshape(M, H)  # contiguous per can_handle: no copy
    # Holds a ROW count here, not a tile count: the kernel needs the exact M for its tail.
    nr = _num_tiles_tensor(M, device)
    outputs = (_dest_buffer(dest, M, H, target, device),)
    return (x, nr), outputs, M


def _reshape_to_copy_output(outputs, original_args):
    """Restore the source shape in the new dtype. M >= _TILE_M so the output is unpadded;
    the [:M] slice is a no-op kept for shape safety."""
    self = original_args[0]
    M, _ = _mh(self)  # noqa: N806
    return outputs[0][:M].reshape(self.shape)


# ── aten::copy_ ──────────────────────────────────────────────────────────────
def _can_handle_copy(self, src=None, non_blocking=False, **_) -> bool:
    """In-place neuron->neuron copy of a contiguous, equally-shaped source (``src``
    arrives already broadcast). Any ragged M >= 128 works; below one tile the tail
    offset would go negative and an in-place output cannot be padded, so it declines.
    C++-dispatch only: the Python path passes the destination as ``out=``."""
    if not _is_castable(self) or not _is_castable(src):
        return False
    if self.shape != src.shape:
        return False
    M, _ = _mh(self)  # noqa: N806
    return M >= _TILE_M


def _copy_compile_key(self, src, non_blocking=False, **_):
    """(H, src_dtype, dst_dtype); the destination dtype is `self`'s."""
    _, H = _mh(self)  # noqa: N806
    return (H, src.dtype, self.dtype)


def _prepare_copy_io(self, src, non_blocking=False, dest=None, **_):
    """Output buffer IS ``self``: the NEFF converts in place, writing only [0, M)."""
    M, H = _mh(self)  # noqa: N806
    x = src.reshape(M, H)
    nr = _num_tiles_tensor(M, self.device)
    outputs = (self.reshape(M, H),)
    return (x, nr), outputs, M


def _reshape_copy_output(outputs, original_args):
    """copy_ returns ``self``, not a view of it."""
    return original_args[0]


# ── shared compile path ──────────────────────────────────────────────────────
def _make_compile_fn(graph_prefix):
    def compile_fn(compile_key) -> tuple[bytes, str]:
        """Compile at [128, H] for one (H, src_dtype, dst_dtype); M dynamic.

        graph_name carries the target dtype because the NEFF cache key is graph_name +
        INPUT specs only: without it bf16->fp16 and bf16->fp32 get identical keys."""
        from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

        from .. import wrappers

        H, src_dtype, dst_dtype = compile_key  # noqa: N806
        wrapper = getattr(wrappers, f"dynamic_cast_{_WRAPPER_SUFFIX[dst_dtype]}_nki")
        graph_name = f"{graph_prefix}_{_WRAPPER_SUFFIX[dst_dtype]}"
        # Meta example inputs: only shapes/dtypes matter, and meta avoids a stray
        # empty/zeros dispatch to the neuron device on cold compile.
        x = torch.empty(_TILE_M, H, dtype=src_dtype, device="meta")
        nr = torch.tensor([[_TILE_M]], dtype=torch.int32, device="meta")
        out = torch.empty(_TILE_M, H, dtype=dst_dtype, device="meta")

        kernel = TorchMlirKernel(torch_fn=wrapper, op_name=graph_name)
        return _compile_or_load_neff(kernel, graph_name, (x, nr), out)

    return compile_fn


def register():
    register_dynamic_nki(
        "_to_copy",
        DynamicNkiEntry(
            compile_fn=_make_compile_fn("dynamic_nki_to_copy"),
            can_handle=_can_handle_to_copy,
            prepare_io=_prepare_to_copy_io,
            reshape_output=_reshape_to_copy_output,
            compile_key_fn=_to_copy_compile_key,
        ),
    )
    register_dynamic_nki(
        "copy_",
        DynamicNkiEntry(
            compile_fn=_make_compile_fn("dynamic_nki_copy_"),
            can_handle=_can_handle_copy,
            prepare_io=_prepare_copy_io,
            reshape_output=_reshape_copy_output,
            compile_key_fn=_copy_compile_key,
        ),
    )
