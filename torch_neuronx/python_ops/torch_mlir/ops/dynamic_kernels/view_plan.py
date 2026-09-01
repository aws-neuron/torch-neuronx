"""Layout analysis for non-contiguous operands on the dynamic-NKI path.

The base is row-major contiguous, so a dim's stride is the product of every extent inner
to it: any dim OUTER to the runtime-varying one carries `rows` as a factor in its stride
and can never be baked. This module accepts only the case where the varying dim is the
outermost base dim -- what a two-level AP reads directly. Everything else (a permuted
leading dim, or a base whose row length is `rows` itself) returns None to fall back.
"""

from dataclasses import dataclass

import torch

_TILE_M = 128


@dataclass(frozen=True)
class StridedPlan:
    """`rows` rows of `width` contiguous elements, `dyn_stride` apart, from element
    `start`; each element optionally fanned out to `fanout` columns.

    `dyn_stride`/`width`/`fanout`/`tile_m` are model constants and bake into the NEFF.
    `rows`/`start`/`span` are runtime values, which is what lets one NEFF cover every row
    count and every storage offset (the two `.chunk(2)` halves differ only in `start`).
    """

    dyn_stride: int
    width: int
    fanout: int
    tile_m: int
    rows: int
    start: int
    span: int


def strided_plan(t, rows) -> StridedPlan | None:
    """Classify `t` as a strided view with `rows` as the varying extent."""
    if not isinstance(t, torch.Tensor) or t.dim() < 1 or t.numel() == 0:
        return None

    # Rows per tile, baked into the NEFF. 128 fills the partition dim, but the tail tile
    # starts at row `rows - tile_m`, so below one full tile the kernel walks single rows.
    tile_m = _TILE_M if rows >= _TILE_M else 1

    # Extent-1 dims address nothing, whatever their stride -- except the leading one when
    # `rows` is 1, which IS the varying dim: dropping it would lose the plan entirely.
    dims = [
        (int(e), int(s))
        for i, (e, s) in enumerate(zip(t.shape, t.stride(), strict=True))
        if e != 1 or (i == 0 and rows == 1)
    ]

    # Trailing stride-0 dims are a fan-out: one source element feeds `fanout` columns.
    # The kernel replicates in SBUF, so the DMA itself never reads stride 0.
    fanout = 1
    while dims and dims[-1][1] == 0:
        fanout *= dims[-1][0]
        dims.pop()
    if not dims or any(s == 0 for _, s in dims):
        return None  # an interior stride-0 dim is not expressible today

    # The varying dim is the outermost base dim, i.e. the one with the largest stride. If
    # its extent is not `rows`, then `rows` hides under another dim's stride (a permute)
    # or is the base row length itself (a transpose) -- neither has a bakeable stride.
    mi = max(range(len(dims)), key=lambda i: dims[i][1])
    if dims[mi][0] != rows or mi != 0:
        return None

    # Dims inner to the varying one must form one contiguous row-major run.
    width = 1
    for e, s in reversed(dims[mi + 1 :]):
        if s != width:
            return None
        width *= e

    dyn_stride = dims[mi][1]
    if width > dyn_stride:
        return None  # the run would overrun into the next row
    if fanout > 1 and width != 1:
        return None  # the kernel fans out a single column

    start = int(t.storage_offset())
    # NRT rejects a buffer smaller than the NEFF's declared input, and the NEFF declares
    # base as [tile_m * dyn_stride, 1]. A rows == tile_m view of the first half falls one
    # part-row short of that, so floor the span; the storage always holds that much.
    span = max(start + (rows - 1) * dyn_stride + width, tile_m * dyn_stride)
    return StridedPlan(
        dyn_stride=dyn_stride,
        width=width,
        fanout=fanout,
        tile_m=tile_m,
        rows=rows,
        start=start,
        span=span,
    )
