"""Grouped matrix multiplication with NKI and MLIR implementations.

The aligned path reads ``offs`` on host only to round each group up to whole tiles, then
pads to a row count fixed by ``(t, g)`` and runs padding, product and unpadding as
separate fixed-shape kernels, so a new ``offs`` never recompiles.
"""

import torch

from torch_neuronx.utils import get_logical_neuron_cores, use_dynamic_nki_kernels

from .auto_registration import neuron_op
from .base import ExecutionResult, OperationImplementation
from .shared.context import ExecutionContext
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
    # Both kernels shift ``offs`` down one to get per-group starts, a copy of g - 1
    # elements; at g == 1 that is a zero-length access pattern and fails to compile.
    if offs.shape[0] < 2:
        return False
    # Per-group schedule vectors put g on a free dim.
    if offs.shape[0] > _FREE_MAX:
        return False
    itemsize = a.element_size()
    # The kernels shard the output's free dim across one SPMD program per logical
    # NeuronCore, so that dim must divide evenly.
    lnc = int(get_logical_neuron_cores())
    if b.dim() == 3:  # 2Dx3D: a (r, d1) @ b (g, d1, d2) -> (r, d2)
        r, d1 = a.shape
        g, d1b, d2 = b.shape
        if d1 != d1b or offs.shape[0] != g:
            return False
        if d2 % lnc != 0:
            return False
        # Untested below one tile; decline rather than guess.
        if r < _TILE:
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
        if t != tb or t < _TILE:
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
    """Check whether ragged grouped-matrix-multiplication kernels accept a call.

    Args:
        a (torch.Tensor): Left matrix operand.
        b (torch.Tensor): Right matrix operand.
        offs (torch.Tensor): One-dimensional cumulative group offsets.

    Returns:
        bool: ``True`` when the operand shapes and dtypes satisfy kernel limits.

    Note:
        This check neither reads offset values nor checks whether dynamic kernels
        are enabled.
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
                """Run ragged 2D-by-3D grouped multiplication and trim spare rows.

                Args:
                    a (torch.Tensor): Left operand with shape ``(R, D1)``.
                    b (torch.Tensor): Expert weights with shape ``(G, D1, D2)``.
                    offs (torch.Tensor): Cumulative group end offsets.

                Returns:
                    torch.Tensor: Result trimmed to ``R`` rows.
                """
                # Slice outside the @nki_op: its shape inference must see all r + 128
                # rows the kernel writes.
                return grouped_mm_ragged_2d_3d_op(a, b, offs)[: a.shape[0]]

        else:

            def fn(a, b, offs):
                """Run ragged 2D-by-2D grouped multiplication.

                Args:
                    a (torch.Tensor): Left operand with shape ``(D1, T)``.
                    b (torch.Tensor): Right operand with shape ``(T, D2)``.
                    offs (torch.Tensor): Cumulative group end offsets.

                Returns:
                    torch.Tensor: Per-group result with shape ``(G, D1, D2)``.
                """
                return grouped_mm_ragged_2d_2d_op(a, b, offs)

        _RAGGED_KERNELS[variant] = TorchMlirKernel(fn, op_name="aten::_grouped_mm")
    return _RAGGED_KERNELS[variant]


class GroupedMMRaggedNKIImpl(TorchMlirOpImpl):
    """Provide the dynamic NKI implementation for ragged grouped matrix multiplication.

    Note:
        This pad-free implementation accepts unaligned, sorted ``offs`` without
        host transfer and requires ``TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS=1``.
    """

    def __init__(self):
        super().__init__(aten_op_name="aten::_grouped_mm", torch_fn=torch._grouped_mm)

    def translate_args(self, *args, **kwargs):
        """Translate ATen arguments for ragged implementation selection.

        Args:
            *args (Any): Positional ``_grouped_mm`` arguments.
            **kwargs (Any): Keyword ``_grouped_mm`` arguments.

        Returns:
            tuple[tuple, dict]: ``(a, b, offs)`` and an options mapping containing
            ``bias`` and ``out_dtype``.
        """
        (a, b, offs), _ = _translate_grouped_mm_args(*args, **kwargs)
        bias, out_dtype = _grouped_mm_options(*args, **kwargs)
        return (a, b, offs), {"bias": bias, "out_dtype": out_dtype}

    def can_handle(self, a, b, offs, bias=None, out_dtype=None) -> bool:
        # The kernels add no bias and always return the input dtype.
        """Check whether the dynamic ragged kernels can handle a call.

        Args:
            a (torch.Tensor): Left matrix operand.
            b (torch.Tensor): Right matrix operand.
            offs (torch.Tensor): Cumulative group offsets.
            bias (torch.Tensor | None, optional): Bias tensor. Defaults to ``None``.
            out_dtype (torch.dtype | None, optional): Requested output dtype. Defaults to ``None``.

        Returns:
            bool: ``True`` when the options and operand constraints pass.

        Note:
            Ragged kernels do not support bias or an output dtype differing from ``a``.
            This shape gate does not read offset values; callers must provide sorted offsets.
        """
        if bias is not None or out_dtype not in (None, a.dtype):
            return False
        return _ragged_shapes_ok(a, b, offs)

    def get_lower_func(self):
        """Return the cache-miss lowering function for the ragged NKI kernel.

        Returns:
            Callable: A lowering function that selects the kernel from ``b`` rank.

        Note:
            Lowering, not ``execute()``, is the dispatch hook. Using the inherited
            ``TorchMlirOpImpl`` kernel would route to the aligned-only MLIR lowering.
        """

        def grouped_mm_ragged_lower(*args, **kwargs):
            """Lower a grouped-matrix-multiplication call to a ragged NKI kernel.

            Args:
                *args (Any): Positional ``_grouped_mm`` arguments.
                **kwargs (Any): Keyword ``_grouped_mm`` arguments.

            Returns:
                Any: The selected ragged kernel's lowered representation.
            """
            (a, b, offs), _ = self.translate_args(*args, **kwargs)
            return _ragged_kernel("2d3d" if b.dim() == 3 else "2d2d").lower(a, b, offs)

        return grouped_mm_ragged_lower

    def execute(self, *args, **kwargs) -> ExecutionResult:
        """Execute the selected ragged NKI kernel.

        Args:
            *args (Any): Positional kernel arguments beginning with ``a``, ``b``, and ``offs``.
            **kwargs (Any): Keyword arguments accepted for interface compatibility.

        Returns:
            ExecutionResult: The kernel output, or the caught exception message on failure.
        """
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


def _padded_rows(t, g):
    """Rows of the padded operand for ``t`` rows split into ``g`` groups, for any ``offs``.

    Each group rounds up to whole tiles, adding at most ``_TILE - 1`` rows, and rows past
    ``offs[-1]`` are dropped. Padding every call to this bound keeps the shapes the
    compiled graphs see independent of the routing, so a new ``offs`` never recompiles.
    """
    return -(-(t + (_TILE - 1) * g) // _TILE) * _TILE


def _padded_offs(offs):
    """Group ends after rounding every group up to whole tiles, computed on host.

    Reading ``offs`` costs a sync, but the result is only ``g`` integers; the per-row maps
    are built on device from it.
    """
    padded_offs, prev, total = [], 0, 0
    for end in offs.detach().to("cpu", torch.int64).tolist():
        total += -(-(end - prev) // _TILE) * _TILE
        padded_offs.append(total)
        prev = end
    return torch.tensor(padded_offs, dtype=torch.int32).to(offs.device)


def _row_maps(offs, padded_offs, t):
    """Map rows between the caller's layout and the padded one.

    Returns ``src``, the row of ``a`` that feeds each padded row (-1 for padding), and
    ``dst``, the padded row that holds each of the ``t`` output rows (-1 past ``offs[-1]``).
    """
    g = offs.shape[0]
    padded = _padded_rows(t, g)
    zero = offs.new_zeros(1)
    starts = torch.cat([zero, offs[:-1]])
    padded_starts = torch.cat([zero, padded_offs[:-1]])
    lengths = offs - starts

    rows = torch.arange(padded, dtype=offs.dtype, device=offs.device)
    group = (padded_offs[None, :] <= rows[:, None]).sum(1).to(offs.dtype)
    clamped = group.clamp(max=g - 1)
    local = rows - padded_starts[clamped]
    src = torch.where((group < g) & (local < lengths[clamped]), starts[clamped] + local, -1)

    rows = torch.arange(t, dtype=offs.dtype, device=offs.device)
    group = (offs[None, :] <= rows[:, None]).sum(1).to(offs.dtype)
    clamped = group.clamp(max=g - 1)
    dst = torch.where(group < g, padded_starts[clamped] + rows - starts[clamped], -1)
    return src, dst


def _take_rows(x, index, dim):
    """``x`` gathered along ``dim`` at ``index``, with zeros where ``index`` is -1."""
    picked = x.index_select(dim, index.clamp(min=0))
    shape = [1] * x.dim()
    shape[dim] = -1
    return torch.where((index >= 0).reshape(shape), picked, 0)


def _pad_2d3d(a, offs, padded_offs):
    src, dst = _row_maps(offs, padded_offs, a.shape[0])
    return _take_rows(a, src, 0), dst


def _pad_2d2d(a, b, offs, padded_offs):
    src, _ = _row_maps(offs, padded_offs, a.shape[1])
    return _take_rows(a, src, 1), _take_rows(b, src, 0)


def _grouped_mm_aligned_2d3d_compute(padded_a, b, padded_offs):
    """2Dx3D product over tile-aligned groups. Padding rows of ``padded_a`` are zero."""
    t, d1 = padded_a.shape
    g, _, d2 = b.shape
    b_index = (
        padded_offs[:, None] // _TILE <= torch.arange(t // _TILE, device=padded_a.device)
    ).sum(0)
    b_index = torch.clamp(b_index, 0, g - 1)
    a_batched = padded_a.reshape(t // _TILE, _TILE, d1)
    r_batched = (a_batched.float() @ b[b_index].float()).to(padded_a.dtype)
    return r_batched.reshape(t, d2)


def _grouped_mm_aligned_2d2d_compute(padded_a, padded_b, padded_offs):
    """2Dx2D product over tile-aligned groups. Padding columns are zero."""
    from .nki_kernels.grouped_mm import grouped_mm_2d_2d_op

    return grouped_mm_2d_2d_op(padded_a, padded_b, padded_offs)


def _unpad_2d3d(padded_out, dst):
    return _take_rows(padded_out, dst, 0)


_ALIGNED_KERNELS: dict[str, object] = {}


def _aligned_kernel(fn):
    """Fixed-shape ``TorchMlirKernel`` for ``fn``. Its inputs never encode ``offs`` values,
    so it compiles once per shape."""
    if fn.__name__ not in _ALIGNED_KERNELS:
        from .torch_mlir.kernel import TorchMlirKernel

        _ALIGNED_KERNELS[fn.__name__] = TorchMlirKernel(fn, op_name=f"grouped_mm::{fn.__name__}")
    return _ALIGNED_KERNELS[fn.__name__]


def _run(fn, *args):
    context = ExecutionContext(original_inputs=args, original_kwargs={})
    return _aligned_kernel(fn)(*args, context=context)


def torch_grouped_mm(a, b, offs, bias=None, out_dtype=None):
    """Aligned ``aten::_grouped_mm``.

    Pads every group to whole tiles, then pads the total to ``_padded_rows`` so the
    padding, product and unpadding graphs all see shapes fixed by ``(t, g)`` and never by
    ``offs`` values. Each compiles once per shape. Registered only when
    TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS is off; the ragged handler takes over when it is set.
    """
    if bias is not None:
        raise NotImplementedError("aten::_grouped_mm on Neuron does not support bias")
    if out_dtype is not None and out_dtype != a.dtype:
        raise NotImplementedError(
            f"aten::_grouped_mm on Neuron requires out_dtype == a.dtype "
            f"(got out_dtype={out_dtype}, a.dtype={a.dtype})"
        )
    if b.dim() not in (2, 3):
        raise NotImplementedError(
            f"aten::_grouped_mm on Neuron requires b to be 2D or 3D (got {b.dim()}D)"
        )

    padded_offs = _padded_offs(offs)
    if b.dim() == 3:
        padded_a, dst = _run(_pad_2d3d, a, offs, padded_offs)
        padded_out = _run(_grouped_mm_aligned_2d3d_compute, padded_a, b, padded_offs)
        return _run(_unpad_2d3d, padded_out, dst)
    padded_a, padded_b = _run(_pad_2d2d, a, b, offs, padded_offs)
    return _run(_grouped_mm_aligned_2d2d_compute, padded_a, padded_b, padded_offs)


class GroupedMMAlignedImpl(OperationImplementation):
    """Aligned grouped_mm, run as a sequence of fixed-shape kernels.

    No lowering of its own: the C++ dispatch cache keys on shapes, and this op's padding
    depends on ``offs`` values, so it runs on the Python path and each kernel it calls is
    cached instead.
    """

    def translate_args(self, *args, **kwargs):
        (a, b, offs), _ = _translate_grouped_mm_args(*args, **kwargs)
        bias, out_dtype = _grouped_mm_options(*args, **kwargs)
        return (a, b, offs), {"bias": bias, "out_dtype": out_dtype}

    def _execute_impl(self, a, b, offs, bias=None, out_dtype=None) -> ExecutionResult:
        return ExecutionResult(
            success=True, output=torch_grouped_mm(a, b, offs, bias=bias, out_dtype=out_dtype)
        )


if use_dynamic_nki_kernels():
    GroupedMMRaggedNKIImpl = neuron_op("aten::_grouped_mm", priority=70)(GroupedMMRaggedNKIImpl)
else:
    GroupedMMAlignedImpl = neuron_op("aten::_grouped_mm", disable_empty_shortcircuit=True)(
        GroupedMMAlignedImpl
    )
