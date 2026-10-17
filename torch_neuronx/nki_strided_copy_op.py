"""Register neuron::nki_strided_copy_ as a torch custom op.

This allows C++ code to call torch.ops.neuron.nki_strided_copy_(src, dst)
which dispatches to the NKI stride-S gather/scatter kernel on device, so a
regular strided neuron->neuron copy (a uniform stride-S lane view on one side,
contiguous on the other) runs on-device instead of a per-element / host
round-trip. Used for packed int64 (stride-2) and, via copy_strided, for general
strided copies (e.g. MLIR-unsupported dtypes like uint8).
"""

from __future__ import annotations

import torch

_LIB_DEF = torch.library.Library("neuron", "DEF")
_LIB_DEF.define("nki_strided_copy_(Tensor src, Tensor(a!) dst) -> Tensor(a!)")

_LIB_IMPL = torch.library.Library("neuron", "IMPL", "PrivateUse1")

# Dtypes the strided-copy NKI gather/scatter kernel handles correctly —
# fixed-width 1/2/4-byte int and float types.
#
# Deliberately a SEPARATE list from the dispatcher's IsSupportedDtype
# (torch_neuronx/csrc/core/dispatch/DispatchUtils.cpp); it can't be reused here:
#   - IsSupportedDtype includes Bool and ComplexFloat, and
#   - IsSupportedDtype EXCLUDES the narrow ints (int16/int8/uint8) because the
#     compiler mishandles arithmetic on them — but this kernel is pure data
#     movement (byte copy, no compute), so narrow ints are safe here.
# What must NOT go through this kernel:
#   - bool, float64/Double, complex*: fail to lower (TorchMlirCompilerError);
#   - int64: lowers WITHOUT error but the Vector-Engine lane-select copy is not
#     bit-exact for values beyond ~2^24 — it SILENTLY CORRUPTS large int64 (both
#     gather and scatter), so it must stay off the kernel. (int64 in packed mode
#     is gathered as its int32 physical view, which IS in this set.)
# Anything not listed takes the per-element path.
#
# TODO: this is a strided-copy-specific allowlist; remove it once IsSupportedDtype
# (or a shared copy-supported-dtype helper) covers exactly these types.
_STRIDED_COPY_KERNEL_DTYPES = frozenset(
    {
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.int16,
        torch.int8,
        torch.uint8,
    }
)


def _uniform_stride(t: torch.Tensor) -> int | None:
    """Return S if ``t``'s elements sit at a uniform stride S over its storage.

    True when ``t``'s strides equal ``S * contiguous_strides(shape)`` (S =
    innermost stride), i.e. its elements are at offsets base, base+S, base+2S,
    ... — exactly what the flat stride-S kernel treats as a group-stream. Size-1
    dims are skipped (their stride is immaterial). Returns None for contiguous
    (S<2), overlapping, or non-uniform (e.g. odd-row-strided) layouts.
    """
    if t.dim() == 0:
        return None
    sizes = [int(s) for s in t.size()]
    strides = [int(s) for s in t.stride()]
    stride = strides[-1]
    if stride < 2:
        return None
    expected = stride
    for d in range(t.dim() - 1, -1, -1):
        if sizes[d] == 1:
            continue
        if strides[d] != expected:
            return None
        expected *= sizes[d]
    return stride


@torch._dynamo.disable
def _nki_strided_copy_impl(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """NKI-based strided copy between Neuron tensors.

    Routes a uniform stride-S gather (strided src -> contiguous dst) or scatter
    (contiguous src -> strided dst) to the on-device kernel; falls back to the
    element-wise NRT copy for general / overlapping patterns. The kernel handles
    any storage span (incl. a minimal-span slice whose last group is truncated),
    so no full-group guard is needed.
    """
    if src.numel() == 0:
        return dst

    # Only dtypes the strided-copy kernel handles correctly take the kernel path;
    # everything else (bool, float64, int64, complex, ...) uses the per-element
    # NRT copy. See _STRIDED_COPY_KERNEL_DTYPES for why int64 is excluded.
    if src.dtype not in _STRIDED_COPY_KERNEL_DTYPES:
        import torch_neuronx._C as _C

        _C._nrt_copy_strided_neuron_to_neuron(src, dst)
        return dst

    from torch_neuronx.python_ops.nki_kernels.stride2_gather import (
        run_strided_gather,
        run_strided_scatter,
    )

    # Gather: uniform stride-S src -> contiguous dst (stride 2 = packed int64).
    if dst.is_contiguous() and not src.is_contiguous():
        stride = _uniform_stride(src)
        if stride is not None:
            run_strided_gather(src, dst, stride=stride, lane=int(src.storage_offset()) % stride)
            return dst

    # Scatter: contiguous src -> uniform stride-S dst (other lanes preserved).
    if src.is_contiguous() and not dst.is_contiguous():
        stride = _uniform_stride(dst)
        if stride is not None:
            run_strided_scatter(src, dst, stride=stride, lane=int(dst.storage_offset()) % stride)
            return dst

    # Fallback: element-wise NRT copy (general / overlapping / large-stride).
    import torch_neuronx._C as _C

    _C._nrt_copy_strided_neuron_to_neuron(src, dst)
    return dst


_LIB_IMPL.impl("nki_strided_copy_", _nki_strided_copy_impl)
