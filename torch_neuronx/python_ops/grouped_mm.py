"""Grouped matrix multiplication with NKI and MLIR implementations"""

import torch

from torch_neuronx.utils import (
    get_logical_neuron_cores,
    use_dynamic_nki_kernels,
)

from .auto_registration import neuron_op
from .base import ExecutionResult
from .torch_mlir.op_impl import TorchMlirOpImpl

_RAGGED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_TILE = 128
# Moving-operand free-dim limit on the Tensor Engine. Both kernels tile their output
# width against this, so it bounds a resident buffer, not the caller's shape.
_MOVING_FMAX = 512
# SBUF is 192 KiB per partition; leave headroom for the per-iteration scratch.
_SBUF_BUDGET = 160 * 1024
# The 2Dx2D kernel holds four (partition-block, ceil(t/128)) int32 schedule tables in
# SBUF at once, so ceil(t/128) is bounded but g is not.
_SCHED_TABLES = 4
# Largest SBUF free-dimension extent an access pattern may address.
_FREE_MAX = 32767
# Persistent (1, g) int32 schedule vectors kept in SBUF for the whole kernel: five in
# the 2Dx3D kernel, three in the 2Dx2D one. These bound g by bytes and by free dim.
_SCHED_VECS_2D3D = 5
_SCHED_VECS_2D2D = 3


def _translate_grouped_mm_args(*args, **kwargs):
    """Translate ATen _grouped_mm args to (a, b, offs).

    The ATen schema passes additional arguments (e.g. bias, use_fast_accum)
    that the NKI/MLIR implementations don't consume.
    """
    a, b, offs = args[0], args[1], args[2]
    return (a, b, offs), {}


def _grouped_mm_options(*args, **kwargs):
    """``bias`` and ``out_dtype`` from the ATen call, positional or keyword.

    Schema: ``_grouped_mm(self, mat2, offs=None, bias=None, out_dtype=None)``.
    """
    bias = kwargs.get("bias", args[3] if len(args) > 3 else None)
    out_dtype = kwargs.get("out_dtype", args[4] if len(args) > 4 else None)
    return bias, out_dtype


def _ragged_shapes_ok(a, b, offs) -> bool:
    """Shape/dtype gate for the ragged kernels.

    Must not read values in ``offs``: it lives on device, so any predicate over it
    forces the host transfer this path exists to remove. The kernels accept any sorted
    ``offs``, so there is nothing to check.
    """
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        return False
    if a.dtype != b.dtype or a.dtype not in _RAGGED_DTYPES:
        return False
    if not isinstance(offs, torch.Tensor) or offs.dim() != 1:
        return False
    if a.dim() != 2:
        return False
    # Per-group schedule vectors put g on a free dim.
    if offs.shape[0] > _FREE_MAX:
        return False
    itemsize = a.element_size()
    # The kernels shard the output's free dim across one SPMD program per logical
    # NeuronCore, so that dim must divide evenly.
    lnc = int(get_logical_neuron_cores())
    if b.dim() == 3:  # 2Dx3D: a (r, d1) @ b (g, d1, d2) -> (r, d2)
        _r, d1 = a.shape
        g, d1b, d2 = b.shape
        if d1 != d1b or offs.shape[0] != g:
            return False
        if d2 % lnc != 0:
            return False
        # Widest free dims: the token tile spans d1, and the resident weights span
        # (d1 / 128) * (d2 / lnc). Byte budget alone does not imply either fits.
        chunks = (d1 + _TILE - 1) // _TILE
        if d1 > _FREE_MAX or chunks * (d2 // lnc) > _FREE_MAX:
            return False
        # The kernel holds all d1/128 contraction chunks of one expert's weights
        # resident, sized by the full d2, plus the schedule vectors.
        resident = itemsize * (2 * chunks * _TILE + chunks * d2 + min(d2, _MOVING_FMAX))
        resident += _SCHED_VECS_2D3D * g * 4
        return resident <= _SBUF_BUDGET
    if b.dim() == 2:  # 2Dx2D: a (d1, t) @ b (t, d2) -> (g, d1, d2)
        d1, t = a.shape
        tb, d2 = b.shape
        g = offs.shape[0]
        if t != tb:
            return False
        if d2 % lnc != 0:
            return False
        chunks = (t + _TILE - 1) // _TILE
        tiles = (d1 + _TILE - 1) // _TILE
        # Widest free dims: the schedule tables span ceil(t/128), the token block spans
        # ceil(d1 / 128) * 128, and the accumulator spans ceil(d1 / 128) * (d2 / lnc).
        if chunks > _FREE_MAX or tiles * _TILE > _FREE_MAX:
            return False
        if tiles * (d2 // lnc) > _FREE_MAX:
            return False
        # One budget for everything resident at once: the schedule tables and per-group
        # vectors coexist with the fp32 accumulator and the token block, so checking any
        # of them alone lets a large t and g pass separately and overflow together.
        resident = (_SCHED_TABLES * chunks + _SCHED_VECS_2D2D * g) * 4
        resident += tiles * d2 * 4
        resident += itemsize * (tiles * _TILE + d2 + min(d2, _MOVING_FMAX))
        return resident <= _SBUF_BUDGET
    return False


def ragged_grouped_mm_supported(a, b, offs) -> bool:
    """Whether the ragged (pad-free) kernels accept this call's shapes and dtypes.

    Says nothing about whether the dynamic-kernel flag is set.
    """
    return _ragged_shapes_ok(a, b, offs)


_RAGGED_KERNELS: dict[str, object] = {}


def _ragged_kernel(variant: str):
    """Kernel whose traced graph contains the ragged NKI custom call.

    Built lazily so ``import torch_neuronx`` does not pull in NKI.
    """
    if variant not in _RAGGED_KERNELS:
        from .nki_kernels.grouped_mm_ragged import (
            grouped_mm_ragged_2d_2d_op,
            grouped_mm_ragged_2d_3d_op,
        )
        from .torch_mlir.kernel import TorchMlirKernel

        if variant == "2d3d":

            def fn(a, b, offs):
                # Slice outside the @nki_op: its shape inference must see all r + 128
                # rows the kernel writes.
                return grouped_mm_ragged_2d_3d_op(a, b, offs)[: a.shape[0]]

        else:

            def fn(a, b, offs):
                return grouped_mm_ragged_2d_2d_op(a, b, offs)

        _RAGGED_KERNELS[variant] = TorchMlirKernel(fn, op_name="aten::_grouped_mm")
    return _RAGGED_KERNELS[variant]


@neuron_op("aten::_grouped_mm", priority=70)
class GroupedMMRaggedNKIImpl(TorchMlirOpImpl):
    """Ragged grouped_mm: arbitrary ``offs``, no padding, no host transfer.

    Calls the ragged kernels directly; the other implementations reach them through
    ``grouped_mm_decomposition``. Enabled by TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS=1.
    """

    def __init__(self):
        super().__init__(aten_op_name="aten::_grouped_mm", torch_fn=torch._grouped_mm)

    def translate_args(self, *args, **kwargs):
        (a, b, offs), _ = _translate_grouped_mm_args(*args, **kwargs)
        bias, out_dtype = _grouped_mm_options(*args, **kwargs)
        return (a, b, offs), {"bias": bias, "out_dtype": out_dtype}

    def can_handle(self, a, b, offs, bias=None, out_dtype=None) -> bool:
        # Flag first so a declined call costs only one env lookup.
        if not use_dynamic_nki_kernels():
            return False
        # The kernels add no bias and always return the input dtype.
        if bias is not None or out_dtype not in (None, a.dtype):
            return False
        return _ragged_shapes_ok(a, b, offs)

    def get_lower_func(self):
        """Route the C++ cache-miss path to the NKI kernel.

        Lowering, not ``execute()``, is the dispatch hook. Without this the inherited
        TorchMlirOpImpl kernel wraps ``torch._grouped_mm`` and reaches the kernels
        through the decomposition instead.
        """

        def grouped_mm_ragged_lower(*args, **kwargs):
            (a, b, offs), _ = self.translate_args(*args, **kwargs)
            return _ragged_kernel("2d3d" if b.dim() == 3 else "2d2d").lower(a, b, offs)

        return grouped_mm_ragged_lower

    def execute(self, *args, **kwargs) -> ExecutionResult:
        try:
            from .nki_kernels.grouped_mm_ragged import (
                grouped_mm_ragged_2d_2d_op,
                grouped_mm_ragged_2d_3d_op,
            )

            a, b, offs = args[0], args[1], args[2]
            if b.dim() == 3:
                # Slice the kernel's 128 spare rows here, not inside the nki_op:
                # its shape inference would then under-allocate what the kernel writes.
                out = grouped_mm_ragged_2d_3d_op(a, b, offs)[: a.shape[0]]
            else:
                out = grouped_mm_ragged_2d_2d_op(a, b, offs)
            return ExecutionResult(success=True, output=out)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))

    def _execute_impl(self, *args, **kwargs) -> ExecutionResult:
        raise NotImplementedError("GroupedMMRaggedNKIImpl uses execute(), not _execute_impl()")


@neuron_op("aten::_grouped_mm", priority=60)
class GroupedMMNKIImpl(TorchMlirOpImpl):
    """2D x 2D: a (d1, t), b (t, d2) -> (g, d1, d2)

    Uses TorchMlirOpImpl for C++ dispatch lowering/dtype inference,
    but overrides execute to use the NKI kernel for actual execution.
    """

    def __init__(self):
        super().__init__(aten_op_name="aten::_grouped_mm", torch_fn=torch._grouped_mm)

    def translate_args(self, *args, **kwargs):
        return _translate_grouped_mm_args(*args, **kwargs)

    def can_handle(self, a, b, offs) -> bool:
        return a.dim() == 2 and b.dim() == 2

    def execute(self, *args, **kwargs) -> ExecutionResult:
        try:
            from .nki_kernels.grouped_mm import grouped_mm_2d_2d_op

            return ExecutionResult(success=True, output=grouped_mm_2d_2d_op(*args))
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))

    def _execute_impl(self, *args, **kwargs) -> ExecutionResult:
        raise NotImplementedError("GroupedMMNKIImpl uses execute(), not _execute_impl()")


@neuron_op("aten::_grouped_mm", priority=40)
class GroupedMMMLIRImpl(TorchMlirOpImpl):
    """2D x 3D: a (t, d1), b (g, d1, d2) -> (t, d2) - MLIR path only"""

    def __init__(self):
        super().__init__(aten_op_name="aten::_grouped_mm", torch_fn=torch._grouped_mm)

    def translate_args(self, *args, **kwargs):
        return _translate_grouped_mm_args(*args, **kwargs)

    def can_handle(self, a, b, offs) -> bool:
        return a.dim() == 2 and b.dim() == 3
