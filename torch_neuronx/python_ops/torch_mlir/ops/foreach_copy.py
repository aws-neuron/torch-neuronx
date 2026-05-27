"""MLIR-based aten::_foreach_copy_ for Neuron.

Registered via the optimized Python dispatcher (Preprocessing.VALUE_DEPENDENT)
to avoid the per-call Operation.__call__ overhead of the @neuron_op route.

Targets FSDPv2's all-gather copy-in path: when all srcs and dsts are adjacent
1D views of contiguous flat buffers, replaces N kernel calls with a single
fused MLIR kernel.

Branch decisions read storage identity (via untyped_storage().data_ptr()) and
runtime stream state — neither is in the dispatch cache key — so the op must
be VALUE_DEPENDENT, not SHAPE_ONLY.

Branches:
  - same-dtype, contiguous range:   single fused MLIR identity kernel
  - same-dtype, scattered:          per-tensor MLIR identity kernel
  - cross-dtype, contiguous dst:    cat sources + single fused MLIR cast kernel
  - cross-dtype, scattered:         per-tensor MLIR cast kernel
  - unsupported dtype / mixed device: dst.copy_(src)

On non-default stream + hostCC the MLIR path is gated; same-dtype routes to
raw NRT instead of per-pair dst.copy_(src), cross-dtype falls back to
dst.copy_(src) (no raw-NRT cast available yet).
"""

from __future__ import annotations

import torch

import torch_neuronx._C as _C

from ..operation_registry import Preprocessing, register_aten

unsupported_dtypes = (torch.float64, torch.int64, torch.uint64, torch.bool)


def _is_contiguous_range(tensors: list[torch.Tensor]) -> bool:
    """Check if tensors are adjacent, non-overlapping 1D views of one storage."""
    if not tensors:
        return False

    first = tensors[0]
    if not first.is_contiguous() or first.dim() != 1 or first.numel() == 0:
        return False

    storage_ptr = first.untyped_storage().data_ptr()
    if storage_ptr == 0:
        return False

    elem_size = first.element_size()
    expected_offset = first.storage_offset()

    for t in tensors:
        if (
            not t.is_contiguous()
            or t.dim() != 1
            or t.untyped_storage().data_ptr() != storage_ptr
            or t.storage_offset() != expected_offset
            or t.element_size() != elem_size
        ):
            return False
        expected_offset += t.numel()

    return True


def _is_default_stream() -> bool:
    from torch_neuronx.neuron import current_stream, default_stream
    from torch_neuronx.utils import is_host_cc_enabled

    return not (is_host_cc_enabled() and current_stream() != default_stream())


def _copy_identity_compute(src):
    """Same-dtype identity copy. Returns a tuple so the kernel writes back via out=."""
    return (src.clone(),)


def _copy_cast_compute(src, out_dtype=None):
    """Cross-dtype cast copy."""
    return (src.to(dtype=out_dtype),)


def _copy_scattered_identity_compute(srcs):
    """Per-tensor same-dtype copy."""
    return tuple(s.clone() for s in srcs)


def _copy_scattered_cast_compute(srcs, out_dtypes=None):
    return tuple(s.to(dtype=dt) for s, dt in zip(srcs, out_dtypes, strict=False))


def _fallback_copy(self_tensors, src_tensors):
    """Direct dst.copy_(src) — used when MLIR can't or shouldn't run."""
    for dst, src in zip(self_tensors, src_tensors, strict=False):
        dst.copy_(src)


def _has_lazy_bits(tensors) -> bool:
    return any(t.is_neg() or t.is_conj() for t in tensors)


def _raw_nrt_copy_same_dtype(self_tensors, src_tensors):
    """Contiguous range -> 1 fused NRT copy. Scattered -> per-tensor NRT copy."""
    if _is_contiguous_range(self_tensors) and _is_contiguous_range(src_tensors):
        total_numel = sum(t.numel() for t in src_tensors)
        src_view = torch.as_strided(
            src_tensors[0],
            (total_numel,),
            (1,),
            storage_offset=src_tensors[0].storage_offset(),
        )
        dst_view = torch.as_strided(
            self_tensors[0],
            (total_numel,),
            (1,),
            storage_offset=self_tensors[0].storage_offset(),
        )
        _C._nrt_copy_neuron_to_neuron_tensor(src_view, dst_view)
        return

    for dst, src in zip(self_tensors, src_tensors, strict=False):
        _C._nrt_copy_neuron_to_neuron_tensor(src, dst)


# No cross-dtype raw-NRT helper — device_cast goes through MLIR for cross-dtype.


@register_aten(
    ["aten::_foreach_copy_"],
    preprocessing=Preprocessing.VALUE_DEPENDENT,
    static_argnames=("out_dtype", "out_dtypes"),
)
def torch_foreach_copy_inplace(self_tensors, src_tensors, non_blocking=False):
    """Preprocessing for aten::_foreach_copy_.

    Schema: (Tensor(a!)[] self, Tensor[] src, bool non_blocking=False) -> ().
    Returns (compute_fn, args, kwargs, postprocess_fn). Postprocess returns
    None (inplace op produces no value).
    """
    if not self_tensors or not src_tensors:
        return _copy_identity_compute, (), {}, lambda _r: None
    if len(self_tensors) != len(src_tensors):
        _fallback_copy(self_tensors, src_tensors)
        return _copy_identity_compute, (), {}, lambda _r: None

    # Mixed device — fall back to per-tensor copy_
    for d, s in zip(self_tensors, src_tensors, strict=False):
        if d.device != s.device or d.device.type != "neuron":
            _fallback_copy(self_tensors, src_tensors)
            return _copy_identity_compute, (), {}, lambda _r: None

    same_dtype = all(d.dtype == s.dtype for d, s in zip(self_tensors, src_tensors, strict=False))

    if same_dtype:
        dtype = self_tensors[0].dtype
        if dtype in unsupported_dtypes:
            _fallback_copy(self_tensors, src_tensors)
            return _copy_identity_compute, (), {}, lambda _r: None
        if not _is_default_stream():
            # MLIR gated; raw NRT can't materialize neg/conj bits.
            if _has_lazy_bits(src_tensors) or _has_lazy_bits(self_tensors):
                _fallback_copy(self_tensors, src_tensors)
            else:
                _raw_nrt_copy_same_dtype(self_tensors, src_tensors)
            return _copy_identity_compute, (), {}, lambda _r: None

        if _is_contiguous_range(self_tensors) and _is_contiguous_range(src_tensors):
            total_numel = sum(t.numel() for t in src_tensors)
            src_flat = torch.as_strided(
                src_tensors[0],
                (total_numel,),
                (1,),
                storage_offset=src_tensors[0].storage_offset(),
            )
            dst_flat = torch.as_strided(
                self_tensors[0],
                (total_numel,),
                (1,),
                storage_offset=self_tensors[0].storage_offset(),
            )
            return (
                _copy_identity_compute,
                (src_flat,),
                {"out": (dst_flat,)},
                lambda _r: None,
            )

        # Scattered same-dtype: one kernel for the whole list.
        non_empty_pairs = [
            (d, s) for d, s in zip(self_tensors, src_tensors, strict=False) if d.numel() > 0
        ]
        if not non_empty_pairs:
            return _copy_identity_compute, (), {}, lambda _r: None
        non_empty_dst = [d for d, _s in non_empty_pairs]
        non_empty_src = [s for _d, s in non_empty_pairs]
        return (
            _copy_scattered_identity_compute,
            (non_empty_src,),
            {"out": tuple(non_empty_dst)},
            lambda _r: None,
        )

    # Cross-dtype path
    if any(t.dtype in unsupported_dtypes for t in self_tensors + src_tensors):
        _fallback_copy(self_tensors, src_tensors)
        return _copy_identity_compute, (), {}, lambda _r: None
    if not _is_default_stream():
        _fallback_copy(self_tensors, src_tensors)
        return _copy_identity_compute, (), {}, lambda _r: None

    if _is_contiguous_range(self_tensors):
        total_numel = sum(t.numel() for t in self_tensors)
        dst_flat = torch.as_strided(
            self_tensors[0],
            (total_numel,),
            (1,),
            storage_offset=self_tensors[0].storage_offset(),
        )
        src_flat = torch.cat([s for s in src_tensors if s.numel() > 0])
        return (
            _copy_cast_compute,
            (src_flat,),
            {"out": (dst_flat,), "out_dtype": dst_flat.dtype},
            lambda _r: None,
        )

    # Scattered cross-dtype: one kernel per pair, but bundle into a single kernel call
    # via a list compute_fn so the per-call overhead amortizes.
    non_empty = [(d, s) for d, s in zip(self_tensors, src_tensors, strict=False) if d.numel() > 0]
    if not non_empty:
        return _copy_identity_compute, (), {}, lambda _r: None
    non_empty_dst = [d for d, _s in non_empty]
    non_empty_src = [s for _d, s in non_empty]
    out_dtypes = tuple(d.dtype for d in non_empty_dst)
    return (
        _copy_scattered_cast_compute,
        (non_empty_src,),
        {"out": tuple(non_empty_dst), "out_dtypes": out_dtypes},
        lambda _r: None,
    )
