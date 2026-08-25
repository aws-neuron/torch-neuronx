"""Dtype-cast / copy dynamic NKI adapters (``aten::_to_copy``, ``aten::copy_``).

One kernel pass — read [M, H] in the source dtype, write it in the destination
dtype — into a fresh tensor for ``_to_copy`` or into ``self`` for ``copy_``.

**Neither path pads the rows**: the kernel covers a ragged M with an overlapping
tail tile (see ``cast_kernel``), so it needs no pad buffer and no over-allocated
output, which is what makes the in-place ``copy_`` legal at any M >= 128. Both
``_to_copy`` and ``copy_`` require M >= 128 (the tail's ``M - 128`` offset would go
negative for a sub-tile M). For a same-dtype strided source a sub-tile M is delegated to
``neuron::nki_strided_copy_`` (see ``_block_fallback_view``); a sub-128 cast still
declines to the native path.

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
from ..view_plan import strided_plan

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


_STRIDED_DTYPES = (*_SUPPORTED_DTYPES, torch.int32)

# int32 has no cast wrapper, so even a CONTIGUOUS int32 copy must use the strided
# kernel's passthrough variant.
_NO_CAST_KERNEL_DTYPES = (torch.int32,)

# The strided kernel reads dyn_stride / width / fanout off these tensors' SHAPES; only
# the shape matters, and it touches one element of each so they stay live.
_PROTO_CACHE: dict[tuple[int, str, int], torch.Tensor] = {}
_PROTO_DTYPE = torch.int32


def _proto(n: int, device) -> torch.Tensor:
    key = (int(n), device.type, device.index if device.index is not None else -1)
    t = _PROTO_CACHE.get(key)
    if t is None:
        t = torch.empty(1, int(n), dtype=_PROTO_DTYPE, device=device)
        _PROTO_CACHE[key] = t
    return t


def _is_broadcast_view(t) -> bool:
    """A stride-0 dim with extent > 1, i.e. the contiguous base is smaller than the
    logical shape."""
    if not isinstance(t, torch.Tensor) or t.dim() < 1:
        return False
    return any(s == 0 and d > 1 for s, d in zip(t.stride(), t.shape, strict=True))


def _broadcast_base(t):
    """Narrow every stride-0 dim to extent 1: [M, 1024] stride (1, 0) -> [M, 1]."""
    shape = [1 if (s == 0 and d > 1) else d for d, s in zip(t.shape, t.stride(), strict=True)]
    base = t.as_strided(shape, t.stride(), t.storage_offset())
    return base if base.is_contiguous() else None


def _strided_src_plan(logical, src):
    """A form-1 strided ``src`` whose contiguous run equals ``logical``'s last dim."""
    if not isinstance(src, torch.Tensor) or src.is_contiguous():
        return None
    if src.device.type != "neuron" or src.dtype not in _STRIDED_DTYPES:
        return None
    M, H = _mh(logical)  # noqa: N806
    plan = strided_plan(src, M)
    if plan is None or plan.width * plan.fanout != H:
        return None
    try:
        src.as_strided((plan.span, 1), (1, 1), 0)
    except RuntimeError:
        return None  # span overruns the storage: not our case
    return plan


def _copy_plan(self, src):
    """The strided plan backing ``self.copy_(src)``, or None. A contiguous source of a
    dtype with no cast kernel is the degenerate plan dyn_stride == width."""
    if not isinstance(src, torch.Tensor) or src.device.type != "neuron":
        return None
    if not src.is_contiguous():
        return _strided_src_plan(self, src)
    if src.dtype not in _NO_CAST_KERNEL_DTYPES:
        return None
    M, H = _mh(self)  # noqa: N806
    if M < _TILE_M:
        return None
    return strided_plan(src.reshape(M, H), M)


# _STRIDED_COPY_KERNEL_DTYPES, narrowed to what we can produce here.
_BLOCK_FALLBACK_DTYPES = (torch.float32, torch.float16, torch.bfloat16, torch.int32)


def _block_fallback_view(t) -> bool:
    """A sub-tile strided view for ``neuron::nki_strided_copy_``: its block-strided gather
    loops per row so any M >= 1 works, and its own fallback is a runtime element copy --
    neither compiles per M. Stride-0 is excluded (not a gather; overlapping reads)."""
    if not isinstance(t, torch.Tensor) or t.device.type != "neuron":
        return False
    if t.dtype not in _BLOCK_FALLBACK_DTYPES or t.dim() < 1 or t.numel() == 0:
        return False
    if t.is_contiguous() or any(s <= 0 for s in t.stride()):
        return False
    M, _ = _mh(t)  # noqa: N806
    return M < _TILE_M


def _strided_inputs(src, plan):
    """``meta`` carries rows and both cursors as VALUES, so neither the row count nor the
    storage offset enters the NEFF key."""
    base = src.as_strided((plan.span, 1), (1, 1), 0)
    meta = torch.tensor(
        [
            [plan.rows // _TILE_M],
            [plan.start],
            [plan.start + (plan.rows - _TILE_M) * plan.dyn_stride],
            [plan.rows - _TILE_M],
        ],
        dtype=torch.int32,
    ).to(src.device)
    return (
        base,
        _proto(plan.dyn_stride, src.device),
        _proto(plan.width, src.device),
        _proto(plan.fanout, src.device),
        meta,
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
    if target not in _SUPPORTED_DTYPES:
        return False
    if _is_castable(self):
        M, _ = _mh(self)  # noqa: N806
        return M >= _TILE_M
    # Broadcast view: cast the narrowed base [M, 1] with the existing NEFF, then expand.
    if (
        _is_broadcast_view(self)
        and self.device.type == "neuron"
        and self.dtype in _SUPPORTED_DTYPES
    ):
        base = _broadcast_base(self)
        if base is not None and base.numel() > 0:
            M, _ = _mh(base)  # noqa: N806
            return M >= _TILE_M
    return False


def _to_copy_compile_key(self, dtype=None, out_dtype=None, non_blocking=False, **_):
    """(kind, H, src_dtype, dst_dtype) — all baked into the NEFF; M dynamic."""
    ref = _broadcast_base(self) if _is_broadcast_view(self) else self
    _, H = _mh(ref)  # noqa: N806
    return ("contig", H, self.dtype, _target_dtype(dtype, out_dtype))


def _prepare_to_copy_io(self, dtype=None, out_dtype=None, non_blocking=False, dest=None, **_):
    """Flatten to [M, H] and hand the kernel the real row count. M >= _TILE_M (can_handle),
    so the [M, H] view goes to the kernel unpadded; reshape_output just restores the shape."""
    target = _target_dtype(dtype, out_dtype)
    M, H = _mh(self)  # noqa: N806
    device = self.device

    # can_handle guarantees M >= _TILE_M, so no pad: the kernel covers the ragged tail with an
    # overlapping last tile and reads the contiguous [M, H] view directly.
    x = _broadcast_base(self) if _is_broadcast_view(self) else self
    M, H = _mh(x)  # noqa: N806
    x = x.reshape(M, H)  # contiguous per can_handle: no copy
    # Holds a ROW count here, not a tile count: the kernel needs the exact M for its tail.
    nr = _num_tiles_tensor(M, device)
    outputs = (_dest_buffer(dest, M, H, target, device),)
    return (x, nr), outputs, M


def _reshape_to_copy_output(outputs, original_args):
    """Restore the source shape in the new dtype. M >= _TILE_M so the output is unpadded;
    the [:M] slice is a no-op kept for shape safety."""
    self = original_args[0]
    if _is_broadcast_view(self):
        base = _broadcast_base(self)
        M, _ = _mh(base)  # noqa: N806
        return outputs[0][:M].reshape(base.shape).expand(self.shape)
    M, _ = _mh(self)  # noqa: N806
    return outputs[0][:M].reshape(self.shape)


# ── aten::copy_ ──────────────────────────────────────────────────────────────
def _can_handle_copy(self, src=None, non_blocking=False, **_) -> bool:
    """In-place neuron->neuron copy of a contiguous, equally-shaped source (``src``
    arrives already broadcast). Any ragged M >= 128 works; below one tile the tail
    offset would go negative and an in-place output cannot be padded, so it declines.
    C++-dispatch only: the Python path passes the destination as ``out=``."""
    if not isinstance(self, torch.Tensor) or not isinstance(src, torch.Tensor):
        return False
    if self.shape != src.shape or not self.is_contiguous():
        return False
    if _is_castable(self) and _is_castable(src):
        M, _ = _mh(self)  # noqa: N806
        return M >= _TILE_M
    # Strided source: without this the C++ prologue absorbs the view and bakes M.
    if self.dtype != src.dtype or self.dtype not in _STRIDED_DTYPES:
        return False
    if _copy_plan(self, src) is not None:
        return True
    # Sub-tile M: delegated to nki_strided_copy_ by _run_copy.
    return _block_fallback_view(src)


def _copy_compile_key(self, src, non_blocking=False, **_):
    """``kind`` keeps the contiguous and strided kernels' keys apart; M and the storage
    offset are never in it."""
    plan = _copy_plan(self, src)
    if plan is not None:
        return ("strided", plan.width, plan.fanout, plan.dyn_stride, src.dtype, self.dtype)
    _, H = _mh(self)  # noqa: N806
    return ("contig", H, src.dtype, self.dtype)


def _prepare_copy_io(self, src, non_blocking=False, dest=None, **_):
    """Output buffer IS ``self``: the NEFF converts in place, writing only [0, M)."""
    M, H = _mh(self)  # noqa: N806
    outputs = (self.reshape(M, H),)
    plan = _copy_plan(self, src)
    if plan is not None:
        return _strided_inputs(src, plan), outputs, M
    x = src.reshape(M, H)
    nr = _num_tiles_tensor(M, self.device)
    return (x, nr), outputs, M


def _run_copy(self, src=None, non_blocking=False, dest=None, **_):
    """Sub-tile M: nki_strided_copy_ writes ``self`` in place. None = our kernel takes it."""
    if not _block_fallback_view(src):
        return None
    torch.ops.neuron.nki_strided_copy_(src, self)
    return self, _mh(self)[0]


def _reshape_copy_output(outputs, original_args):
    """copy_ returns ``self``, not a view of it."""
    return original_args[0]


# ── aten::contiguous ─────────────────────────────────────────────────────────
def _can_handle_contiguous(self, memory_format=None, **_) -> bool:
    """Materialize a strided/broadcast view. Same dtype in and out, so this always uses
    the passthrough kernel variant."""
    if not isinstance(self, torch.Tensor) or self.is_contiguous():
        return False
    if self.dtype not in _STRIDED_DTYPES:
        return False
    if _strided_src_plan(self, self) is not None:
        return True
    # Sub-tile M: delegated to nki_strided_copy_ by _run_contiguous.
    return _block_fallback_view(self)


def _contiguous_compile_key(self, memory_format=None, **_):
    plan = _strided_src_plan(self, self)
    return ("strided", plan.width, plan.fanout, plan.dyn_stride, self.dtype, self.dtype)


def _prepare_contiguous_io(self, memory_format=None, dest=None, **_):
    M, H = _mh(self)  # noqa: N806
    plan = _strided_src_plan(self, self)
    outputs = (_dest_buffer(dest, M, H, self.dtype, self.device),)
    return _strided_inputs(self, plan), outputs, M


def _run_contiguous(self, memory_format=None, dest=None, **_):
    """Sub-tile M: materialize through nki_strided_copy_ into a fresh buffer."""
    if not _block_fallback_view(self):
        return None
    out = torch.empty(self.shape, dtype=self.dtype, device=self.device)
    torch.ops.neuron.nki_strided_copy_(self, out)
    return out, _mh(self)[0]


def _reshape_contiguous_output(outputs, original_args):
    self = original_args[0]
    M, _ = _mh(self)  # noqa: N806
    return outputs[0][:M].reshape(self.shape)


# ── shared compile path ──────────────────────────────────────────────────────
def _make_compile_fn(graph_prefix):
    def compile_fn(compile_key) -> tuple[bytes, str]:
        """Compile at [128, H] for one (H, src_dtype, dst_dtype); M dynamic.

        graph_name carries the target dtype because the NEFF cache key is graph_name +
        INPUT specs only: without it bf16->fp16 and bf16->fp32 get identical keys."""
        from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

        from .. import wrappers

        kind = compile_key[0]
        # Meta example inputs: only shapes/dtypes matter, and meta avoids a stray
        # empty/zeros dispatch to the neuron device on cold compile.
        if kind == "strided":
            _, width, fanout, dyn_stride, src_dtype, dst_dtype = compile_key
            sfx = "same" if src_dtype == dst_dtype else _WRAPPER_SUFFIX[dst_dtype]
            wrapper = getattr(wrappers, f"dynamic_strided_copy_{sfx}_nki")
            graph_name = f"{graph_prefix}_strided_{sfx}_s{dyn_stride}_w{width}_f{fanout}"
            ins = (
                torch.empty(_TILE_M * dyn_stride, 1, dtype=src_dtype, device="meta"),
                torch.empty(1, dyn_stride, dtype=torch.int32, device="meta"),
                torch.empty(1, width, dtype=torch.int32, device="meta"),
                torch.empty(1, fanout, dtype=torch.int32, device="meta"),
                torch.zeros(4, 1, dtype=torch.int32, device="meta"),
            )
            out = torch.empty(_TILE_M, width * fanout, dtype=dst_dtype, device="meta")
        else:
            _, H, src_dtype, dst_dtype = compile_key  # noqa: N806
            wrapper = getattr(wrappers, f"dynamic_cast_{_WRAPPER_SUFFIX[dst_dtype]}_nki")
            graph_name = f"{graph_prefix}_{_WRAPPER_SUFFIX[dst_dtype]}"
            ins = (
                torch.empty(_TILE_M, H, dtype=src_dtype, device="meta"),
                torch.tensor([[_TILE_M]], dtype=torch.int32, device="meta"),
            )
            out = torch.empty(_TILE_M, H, dtype=dst_dtype, device="meta")

        kernel = TorchMlirKernel(torch_fn=wrapper, op_name=graph_name)
        return _compile_or_load_neff(kernel, graph_name, ins, out)

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
        "contiguous",
        DynamicNkiEntry(
            compile_fn=_make_compile_fn("dynamic_nki_contiguous"),
            can_handle=_can_handle_contiguous,
            prepare_io=_prepare_contiguous_io,
            reshape_output=_reshape_contiguous_output,
            compile_key_fn=_contiguous_compile_key,
            run=_run_contiguous,
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
            run=_run_copy,
        ),
    )
