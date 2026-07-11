"""Strided (non-contiguous) tensor coverage for c10d collectives.

Matrix: {strided input, strided output} x {all_reduce, all_gather,
reduce_scatter, all_to_all}.

Non-contiguity is produced via transpose (a permuted view of the *full*
buffer), not slicing — a collective needs all of its input data, so a
transposed full tensor is the right "non-contiguous" representative; a
slice would be a *subset* (a different, heavier SCATTER_IOTA case). Values
come from `arange` so a layout-ignoring bug changes the result and is
caught, rather than being masked by a constant fill.

What each side exercises:
  - strided INPUT  -> the prologue path: ProcessInputTransformations fuses
    a transform into the op MLIR (or CPU-roundtrips if prologue is off).
  - strided OUTPUT -> the epilogue path: ProcessOutputTransformations fuses
    an inverse transform (or stages a contiguous temp + scatter-back).

all_reduce is in-place, so its input and output are the same tensor; its
two cases differ only in tensor rank (2D vs 3D transpose), not in code path.
"""

import math

import pytest
import torch
import torch.distributed as dist

from .base_collective_op import BaseCollectiveOpTest


def make_view(t, transpose_dims):
    return t.t() if transpose_dims is None else t.transpose(*transpose_dims)


# ─────────────────────────────────────────────────────────────────────────
# all_reduce  (in-place: the strided view is both input and output)
# ─────────────────────────────────────────────────────────────────────────


def run_all_reduce_strided(rank, world_size, kwargs):
    """all_reduce reading/writing through a transposed view.

    kwargs:
      base_shape     - shape passed to reshape() before transposing
      transpose_dims - (dim0, dim1) for transpose(); None uses .t()
    """
    base_shape = kwargs["base_shape"]
    transpose_dims = kwargs["transpose_dims"]
    numel = math.prod(base_shape)

    base = torch.arange(numel, dtype=torch.float32).reshape(base_shape) + rank * 100
    expected = torch.zeros_like(make_view(base, transpose_dims).contiguous())
    for r in range(world_size):
        expected += make_view(
            torch.arange(numel, dtype=torch.float32).reshape(base_shape) + r * 100, transpose_dims
        ).contiguous()

    view = make_view(base.to("neuron"), transpose_dims)
    assert not view.is_contiguous()

    dist.all_reduce(view, op=dist.ReduceOp.SUM)
    assert torch.allclose(view.cpu(), expected), f"Rank {rank}: {view.cpu()} != {expected}"


def run_all_reduce_coalesced_strided(rank, world_size, kwargs):
    """Test basic all_reduce_coalesced functionality with multiple tensors"""
    base_shape_list = kwargs["base_shape_list"]
    transpose_dims_list = kwargs["transpose_dims"]
    # Each rank contributes different values to make verification clear
    base_list = [
        torch.arange(math.prod(base_shape), dtype=torch.float32).reshape(base_shape) + rank * 100
        for base_shape in base_shape_list
    ]

    # Expected after SUM reduction across 2 ranks:
    expected_list = [
        torch.zeros_like(make_view(base, transpose_dims).contiguous())
        for (base, transpose_dims) in zip(base_list, transpose_dims_list, strict=False)
    ]
    for r in range(world_size):
        for expected, base_shape, transpose_dims in zip(
            expected_list, base_shape_list, transpose_dims_list, strict=False
        ):
            expected += make_view(
                torch.arange(math.prod(base_shape), dtype=torch.float32).reshape(base_shape)
                + r * 100,
                transpose_dims,
            ).contiguous()
    view_list = [
        make_view(base.to("neuron"), transpose_dims)
        for (base, transpose_dims) in zip(base_list, transpose_dims_list, strict=False)
    ]
    dist.all_reduce_coalesced(view_list, op=dist.ReduceOp.SUM)

    for view, expected in zip(view_list, expected_list, strict=False):
        assert torch.allclose(view.cpu(), expected), f"Rank {rank}: {view.cpu()} != {expected}"


# ─────────────────────────────────────────────────────────────────────────
# all_gather
# ─────────────────────────────────────────────────────────────────────────


def run_all_gather_into_tensor_strided_input(rank, world_size, kwargs):
    """all_gather_into_tensor where each rank contributes a non-contiguous
    (transposed) input; the output buffer is contiguous.

    kwargs:
      base_shape     - shape before transposing
      transpose_dims - (dim0, dim1) for transpose(); None uses .t()
    """
    base_shape = kwargs["base_shape"]
    transpose_dims = kwargs["transpose_dims"]
    numel = math.prod(base_shape)
    base = torch.arange(numel, dtype=torch.float32).reshape(base_shape) + rank * 100
    input_view = make_view(base.to("neuron"), transpose_dims)
    assert not input_view.is_contiguous()

    view_shape = make_view(base, transpose_dims).shape
    output_shape = (world_size * view_shape[0],) + view_shape[1:]
    output = torch.zeros(output_shape, dtype=torch.float32).to("neuron")
    dist.all_gather_into_tensor(output, input_view)

    expected = torch.cat(
        [
            make_view(
                torch.arange(numel, dtype=torch.float32).reshape(base_shape) + r * 100,
                transpose_dims,
            ).contiguous()
            for r in range(world_size)
        ],
        dim=0,
    )
    assert torch.allclose(output.cpu(), expected), f"Rank {rank}: {output.cpu()} != {expected}"


def run_all_gather_into_tensor_strided_output(rank, world_size, kwargs):
    """all_gather_into_tensor scattering into a non-contiguous output view;
    the input is contiguous.

    kwargs:
      base_shape     - shape of each rank's contiguous input
      transpose_dims - (dim0, dim1) for transpose(); None uses .t()
    """
    base_shape = kwargs["base_shape"]
    transpose_dims = kwargs["transpose_dims"]
    numel = math.prod(base_shape)
    input_tensor = (torch.arange(numel, dtype=torch.float32).reshape(base_shape) + rank * 100).to(
        "neuron"
    )

    output_shape = (world_size * base_shape[0],) + base_shape[1:]
    storage_shape = make_view(torch.empty(output_shape), transpose_dims).shape
    out_storage = torch.zeros(storage_shape, dtype=torch.float32).to("neuron")
    output_view = make_view(out_storage, transpose_dims)
    assert not output_view.is_contiguous()

    dist.all_gather_into_tensor(output_view, input_tensor)

    expected = torch.cat(
        [
            torch.arange(numel, dtype=torch.float32).reshape(base_shape) + r * 100
            for r in range(world_size)
        ],
        dim=0,
    )
    assert torch.allclose(
        output_view.cpu(), expected
    ), f"Rank {rank}: {output_view.cpu()} != {expected}"


def run_all_gather_base_strided_input(rank, world_size, kwargs):
    """dist.all_gather where each rank contributes a non-contiguous (transposed)
    input; the output list holds contiguous buffers.

    kwargs:
      base_shape     - shape before transposing
      transpose_dims - (dim0, dim1) for transpose(); None uses .t()
    """
    base_shape = kwargs["base_shape"]
    transpose_dims = kwargs["transpose_dims"]
    numel = math.prod(base_shape)
    base = torch.arange(numel, dtype=torch.float32).reshape(base_shape) + rank * 100
    input_view = make_view(base.to("neuron"), transpose_dims)
    assert not input_view.is_contiguous()

    view_shape = make_view(base, transpose_dims).shape
    output_list = [
        torch.zeros(view_shape, dtype=torch.float32).to("neuron") for _ in range(world_size)
    ]
    dist.all_gather(output_list, input_view)

    for r in range(world_size):
        expected = make_view(
            torch.arange(numel, dtype=torch.float32).reshape(base_shape) + r * 100,
            transpose_dims,
        ).contiguous()
        assert torch.allclose(
            output_list[r].cpu(), expected
        ), f"Rank {rank} slot {r}: {output_list[r].cpu()} != {expected}"


def run_all_gather_base_strided_output(rank, world_size, kwargs):
    """dist.all_gather scattering into non-contiguous output tensors;
    the input is contiguous.

    kwargs:
      base_shape     - shape of each rank's contiguous input
      transpose_dims - (dim0, dim1) for transpose(); None uses .t()
    """
    base_shape = kwargs["base_shape"]
    transpose_dims = kwargs["transpose_dims"]
    numel = math.prod(base_shape)
    input_tensor = (torch.arange(numel, dtype=torch.float32).reshape(base_shape) + rank * 100).to(
        "neuron"
    )

    storage_shape = make_view(torch.empty(base_shape), transpose_dims).shape
    out_storages = [
        torch.zeros(storage_shape, dtype=torch.float32).to("neuron") for _ in range(world_size)
    ]
    output_list = [make_view(s, transpose_dims) for s in out_storages]
    assert not output_list[0].is_contiguous()

    dist.all_gather(output_list, input_tensor)

    for r in range(world_size):
        expected = torch.arange(numel, dtype=torch.float32).reshape(base_shape) + r * 100
        assert torch.allclose(
            output_list[r].cpu(), expected
        ), f"Rank {rank} slot {r}: {output_list[r].cpu()} != {expected}"


def run_all_gather_coalesced_strided_input(rank, world_size, kwargs):
    """dist.all_gather_coalesced where each rank contributes two non-contiguous
    (transposed) inputs; the output lists hold contiguous buffers.

    kwargs:
      base_shape     - shape before transposing
      transpose_dims - (dim0, dim1) for transpose(); None uses .t()
    """
    base_shape = kwargs["base_shape"]
    transpose_dims = kwargs["transpose_dims"]
    numel = math.prod(base_shape)
    base1 = torch.arange(numel, dtype=torch.float32).reshape(base_shape) + rank * 100
    base2 = torch.arange(numel, dtype=torch.float32).reshape(base_shape) + rank * 100 + numel
    input_views = [
        make_view(base1.to("neuron"), transpose_dims),
        make_view(base2.to("neuron"), transpose_dims),
    ]
    assert not input_views[0].is_contiguous()

    view_shape = make_view(base1, transpose_dims).shape
    output_lists = [
        [torch.zeros(view_shape, dtype=torch.float32).to("neuron") for _ in range(world_size)],
        [torch.zeros(view_shape, dtype=torch.float32).to("neuron") for _ in range(world_size)],
    ]
    dist.all_gather_coalesced(output_lists, input_views)

    for r in range(world_size):
        exp0 = make_view(
            torch.arange(numel, dtype=torch.float32).reshape(base_shape) + r * 100, transpose_dims
        ).contiguous()
        exp1 = make_view(
            torch.arange(numel, dtype=torch.float32).reshape(base_shape) + r * 100 + numel,
            transpose_dims,
        ).contiguous()
        assert torch.allclose(
            output_lists[0][r].cpu(), exp0
        ), f"Rank {rank} slot {r} tensor 0: {output_lists[0][r].cpu()} != {exp0}"
        assert torch.allclose(
            output_lists[1][r].cpu(), exp1
        ), f"Rank {rank} slot {r} tensor 1: {output_lists[1][r].cpu()} != {exp1}"


def run_all_gather_coalesced_strided_output(rank, world_size, kwargs):
    """dist.all_gather_coalesced with contiguous inputs scattering into
    non-contiguous output tensors.

    kwargs:
      base_shape     - shape of each rank's contiguous input tensors
      transpose_dims - (dim0, dim1) for transpose(); None uses .t()
    """
    base_shape = kwargs["base_shape"]
    transpose_dims = kwargs["transpose_dims"]
    numel = math.prod(base_shape)
    input_tensors = [
        (torch.arange(numel, dtype=torch.float32).reshape(base_shape) + rank * 100).to("neuron"),
        (torch.arange(numel, dtype=torch.float32).reshape(base_shape) + rank * 100 + numel).to(
            "neuron"
        ),
    ]

    storage_shape = make_view(torch.empty(base_shape), transpose_dims).shape

    def make_strided_slots():
        return [
            make_view(torch.zeros(storage_shape, dtype=torch.float32).to("neuron"), transpose_dims)
            for _ in range(world_size)
        ]

    output_lists = [make_strided_slots(), make_strided_slots()]
    assert not output_lists[0][0].is_contiguous()

    dist.all_gather_coalesced(output_lists, input_tensors)

    for r in range(world_size):
        exp0 = torch.arange(numel, dtype=torch.float32).reshape(base_shape) + r * 100
        exp1 = torch.arange(numel, dtype=torch.float32).reshape(base_shape) + r * 100 + numel
        assert torch.allclose(
            output_lists[0][r].cpu(), exp0
        ), f"Rank {rank} slot {r} tensor 0: {output_lists[0][r].cpu()} != {exp0}"
        assert torch.allclose(
            output_lists[1][r].cpu(), exp1
        ), f"Rank {rank} slot {r} tensor 1: {output_lists[1][r].cpu()} != {exp1}"


# ─────────────────────────────────────────────────────────────────────────
# reduce_scatter
# ─────────────────────────────────────────────────────────────────────────

# All reduce_scatter variants scatter along dim 0 of the input view. The
# input view's dim 0 must therefore be divisible by world_size so each rank
# receives an equal chunk.  base_shape is chosen such that after transposing
# (transpose_dims), the resulting view's dim 0 satisfies this requirement.


def run_reduce_scatter_strided_input(rank, world_size, kwargs):
    """dist.reduce_scatter with a non-contiguous (transposed) input;
    contiguous output. Every rank supplies the same input so the expected
    output is input_view * world_size, sliced along dim 0.

    kwargs:
      base_shape     - shape before transposing; view's dim 0 must be
                       divisible by world_size
      transpose_dims - (dim0, dim1) for transpose(); None uses .t()
    """
    base_shape = kwargs["base_shape"]
    transpose_dims = kwargs["transpose_dims"]
    numel = math.prod(base_shape)
    src = torch.arange(numel, dtype=torch.float32).reshape(base_shape)
    input_view = make_view(src.to("neuron"), transpose_dims)
    assert not input_view.is_contiguous()

    view_shape = make_view(src, transpose_dims).shape
    chunk = view_shape[0] // world_size
    output_shape = (chunk,) + view_shape[1:]
    output_tensor = torch.zeros(output_shape, dtype=torch.float32).to("neuron")

    summed = make_view(src, transpose_dims).contiguous() * world_size
    start = rank * chunk
    expected = summed[start : start + chunk].contiguous()

    dist.reduce_scatter(output=output_tensor, input_list=[input_view], op=dist.ReduceOp.SUM)
    assert torch.allclose(
        output_tensor.cpu(), expected
    ), f"Rank {rank}: {output_tensor.cpu()} != {expected}"


def run_reduce_scatter_strided_output(rank, world_size, kwargs):
    """dist.reduce_scatter with a contiguous input and a non-contiguous
    (transposed) output view. Every rank supplies the same input.

    kwargs:
      base_shape     - shape of the contiguous input; dim 0 must be
                       divisible by world_size
      transpose_dims - (dim0, dim1) for transpose(); None uses .t()
    """
    base_shape = kwargs["base_shape"]
    transpose_dims = kwargs["transpose_dims"]
    numel = math.prod(base_shape)
    src = torch.arange(numel, dtype=torch.float32).reshape(base_shape)
    input_tensor = src.to("neuron")

    chunk = base_shape[0] // world_size
    output_shape = (chunk,) + base_shape[1:]
    storage_shape = make_view(torch.empty(output_shape), transpose_dims).shape
    out_storage = torch.zeros(storage_shape, dtype=torch.float32).to("neuron")
    output_view = make_view(out_storage, transpose_dims)
    assert not output_view.is_contiguous()

    summed = src * world_size
    start = rank * chunk
    expected = summed[start : start + chunk].contiguous()

    dist.reduce_scatter(output=output_view, input_list=[input_tensor], op=dist.ReduceOp.SUM)
    assert torch.allclose(
        output_view.cpu(), expected
    ), f"Rank {rank}: {output_view.cpu()} != {expected}"


def run_reduce_scatter_tensor_strided_input(rank, world_size, kwargs):
    """dist.reduce_scatter_tensor with a non-contiguous (transposed) input;
    contiguous output. Every rank supplies the same input.

    kwargs:
      base_shape     - shape before transposing; view's dim 0 must be
                       divisible by world_size
      transpose_dims - (dim0, dim1) for transpose(); None uses .t()
    """
    base_shape = kwargs["base_shape"]
    transpose_dims = kwargs["transpose_dims"]
    numel = math.prod(base_shape)
    src = torch.arange(numel, dtype=torch.float32).reshape(base_shape)
    input_view = make_view(src.to("neuron"), transpose_dims)
    assert not input_view.is_contiguous()

    view_shape = make_view(src, transpose_dims).shape
    chunk = view_shape[0] // world_size
    output_shape = (chunk,) + view_shape[1:]
    output_tensor = torch.zeros(output_shape, dtype=torch.float32).to("neuron")

    summed = make_view(src, transpose_dims).contiguous() * world_size
    start = rank * chunk
    expected = summed[start : start + chunk].contiguous()

    dist.reduce_scatter_tensor(output_tensor, input_view, op=dist.ReduceOp.SUM)
    assert torch.allclose(
        output_tensor.cpu(), expected
    ), f"Rank {rank}: {output_tensor.cpu()} != {expected}"


def run_reduce_scatter_tensor_strided_output(rank, world_size, kwargs):
    """dist.reduce_scatter_tensor with a contiguous input and a non-contiguous
    (transposed) output view. Every rank supplies the same input.

    kwargs:
      base_shape     - shape of the contiguous input; dim 0 must be
                       divisible by world_size
      transpose_dims - (dim0, dim1) for transpose(); None uses .t()
    """
    base_shape = kwargs["base_shape"]
    transpose_dims = kwargs["transpose_dims"]
    numel = math.prod(base_shape)
    src = torch.arange(numel, dtype=torch.float32).reshape(base_shape)
    input_tensor = src.to("neuron")

    chunk = base_shape[0] // world_size
    output_shape = (chunk,) + base_shape[1:]
    storage_shape = make_view(torch.empty(output_shape), transpose_dims).shape
    out_storage = torch.zeros(storage_shape, dtype=torch.float32).to("neuron")
    output_view = make_view(out_storage, transpose_dims)
    assert not output_view.is_contiguous()

    summed = src * world_size
    start = rank * chunk
    expected = summed[start : start + chunk].contiguous()

    dist.reduce_scatter_tensor(output_view, input_tensor, op=dist.ReduceOp.SUM)
    assert torch.allclose(
        output_view.cpu(), expected
    ), f"Rank {rank}: {output_view.cpu()} != {expected}"


def run_reduce_scatter_tensor_coalesced_strided_input(rank, world_size, kwargs):
    """_c10d_functional.reduce_scatter_tensor_coalesced with two non-contiguous
    (transposed) inputs; outputs are contiguous. Every rank supplies the same
    inputs.

    kwargs:
      base_shape     - shape before transposing; view's dim 0 must be
                       divisible by world_size
      transpose_dims - (dim0, dim1) for transpose(); None uses .t()
    """
    base_shape = kwargs["base_shape"]
    transpose_dims = kwargs["transpose_dims"]
    numel = math.prod(base_shape)
    src1 = torch.arange(numel, dtype=torch.float32).reshape(base_shape)
    src2 = torch.arange(numel, dtype=torch.float32).reshape(base_shape) + numel
    inputs = [
        make_view(src1.to("neuron"), transpose_dims),
        make_view(src2.to("neuron"), transpose_dims),
    ]
    assert not inputs[0].is_contiguous()

    default_group = dist.group.WORLD
    outputs = torch.ops._c10d_functional.reduce_scatter_tensor_coalesced(
        inputs, "sum", world_size, default_group.group_name
    )
    for out in outputs:
        torch.ops._c10d_functional.wait_tensor(out)

    view_shape = make_view(src1, transpose_dims).shape
    chunk = view_shape[0] // world_size
    start = rank * chunk
    exp1 = (make_view(src1, transpose_dims).contiguous() * world_size)[
        start : start + chunk
    ].contiguous()
    exp2 = (make_view(src2, transpose_dims).contiguous() * world_size)[
        start : start + chunk
    ].contiguous()
    assert torch.allclose(
        outputs[0].cpu(), exp1
    ), f"Rank {rank} tensor 0: {outputs[0].cpu()} != {exp1}"
    assert torch.allclose(
        outputs[1].cpu(), exp2
    ), f"Rank {rank} tensor 1: {outputs[1].cpu()} != {exp2}"


def run_reduce_scatter_tensor_coalesced_strided_output(rank, world_size, kwargs):
    """_c10d_functional.reduce_scatter_tensor_coalesced — not applicable for
    strided output since the functional API allocates its own output tensors.
    This variant verifies that contiguous inputs produce correct results when
    the caller immediately stores them through a non-contiguous view.

    kwargs:
      base_shape     - shape of the contiguous inputs; dim 0 must be
                       divisible by world_size
      transpose_dims - (dim0, dim1) for transpose(); None uses .t()
    """
    base_shape = kwargs["base_shape"]
    transpose_dims = kwargs["transpose_dims"]
    numel = math.prod(base_shape)
    src1 = torch.arange(numel, dtype=torch.float32).reshape(base_shape)
    src2 = torch.arange(numel, dtype=torch.float32).reshape(base_shape) + numel
    inputs = [src1.to("neuron"), src2.to("neuron")]

    default_group = dist.group.WORLD
    outputs = torch.ops._c10d_functional.reduce_scatter_tensor_coalesced(
        inputs, "sum", world_size, default_group.group_name
    )
    for out in outputs:
        torch.ops._c10d_functional.wait_tensor(out)

    chunk = base_shape[0] // world_size
    start = rank * chunk
    exp1 = (src1 * world_size)[start : start + chunk].contiguous()
    exp2 = (src2 * world_size)[start : start + chunk].contiguous()

    # Write results through a non-contiguous view to exercise the epilogue path.
    output_shape = (chunk,) + base_shape[1:]
    storage_shape = make_view(torch.empty(output_shape), transpose_dims).shape
    for out, exp in [(outputs[0], exp1), (outputs[1], exp2)]:
        buf = torch.zeros(storage_shape, dtype=torch.float32).to("neuron")
        view = make_view(buf, transpose_dims)
        assert not view.is_contiguous()
        view.copy_(out)
        assert torch.allclose(view.cpu(), exp), f"Rank {rank}: {view.cpu()} != {exp}"


# ─────────────────────────────────────────────────────────────────────────
# all_to_all  (requires world_size == 4)
# ─────────────────────────────────────────────────────────────────────────


def run_all_to_all_single_strided_input(rank, world_size, kwargs):
    """dist.all_to_all_single with a non-contiguous (transposed) input;
    output is contiguous. The input view's dim 0 is world_size chunks,
    one per destination rank.

    kwargs:
      base_shape     - shape before transposing; view's dim 0 must equal
                       world_size * (output dim 0)
      transpose_dims - (dim0, dim1) for transpose(); None uses .t()
    """
    base_shape = kwargs["base_shape"]
    transpose_dims = kwargs["transpose_dims"]
    numel = math.prod(base_shape)

    # Each rank sends the same data so expected = input_view itself (identity all_to_all_single
    # when every chunk has a unique per-(rank,dst) value derived from arange offsets).
    src = torch.arange(numel, dtype=torch.float32).reshape(base_shape) + rank * 100
    input_view = make_view(src.to("neuron"), transpose_dims)
    assert not input_view.is_contiguous()

    view_shape = make_view(src, transpose_dims).shape
    output_tensor = torch.zeros(view_shape, dtype=torch.float32).to("neuron")

    dist.all_to_all_single(output_tensor, input_view)

    # all_to_all_single scatters/gathers equal chunks along dim 0: rank r receives
    # the chunk at index `rank` from each sender.  Since every rank uses the same
    # base arange the received chunk equals the sent chunk (same data everywhere).
    chunk = view_shape[0] // world_size
    expected_chunks = []
    for src_rank in range(world_size):
        full = make_view(
            torch.arange(numel, dtype=torch.float32).reshape(base_shape) + src_rank * 100,
            transpose_dims,
        ).contiguous()
        expected_chunks.append(full[rank * chunk : (rank + 1) * chunk])
    expected = torch.cat(expected_chunks, dim=0)

    assert torch.allclose(
        output_tensor.cpu(), expected
    ), f"Rank {rank}: {output_tensor.cpu()} != {expected}"


def run_all_to_all_single_strided_output(rank, world_size, kwargs):
    """dist.all_to_all_single with a contiguous input and a non-contiguous
    (transposed) output view.

    kwargs:
      base_shape     - shape of the contiguous input; view's dim 0 must equal
                       world_size * (output dim 0)
      transpose_dims - (dim0, dim1) for transpose(); None uses .t()
    """
    base_shape = kwargs["base_shape"]
    transpose_dims = kwargs["transpose_dims"]
    numel = math.prod(base_shape)

    src = torch.arange(numel, dtype=torch.float32).reshape(base_shape) + rank * 100
    input_tensor = src.to("neuron")

    storage_shape = make_view(torch.empty(base_shape), transpose_dims).shape
    out_storage = torch.zeros(storage_shape, dtype=torch.float32).to("neuron")
    output_view = make_view(out_storage, transpose_dims)
    assert not output_view.is_contiguous()

    dist.all_to_all_single(output_view, input_tensor)

    chunk = base_shape[0] // world_size
    expected_chunks = []
    for src_rank in range(world_size):
        full = torch.arange(numel, dtype=torch.float32).reshape(base_shape) + src_rank * 100
        expected_chunks.append(full[rank * chunk : (rank + 1) * chunk])
    expected = torch.cat(expected_chunks, dim=0)

    assert torch.allclose(
        output_view.cpu(), expected
    ), f"Rank {rank}: {output_view.cpu()} != {expected}"


def run_all_to_all_strided_input(rank, world_size, kwargs):
    """dist.all_to_all where each per-destination input is a non-contiguous
    (transposed) view; outputs are contiguous.

    Each slot is unique per (rank, dst) so correctness verifies routing.

    kwargs:
      base_shape     - shape before transposing
      transpose_dims - (dim0, dim1) for transpose(); None uses .t()
    """
    base_shape = kwargs["base_shape"]
    transpose_dims = kwargs["transpose_dims"]
    numel = math.prod(base_shape)

    input_tensor_list = [
        make_view(
            (
                torch.arange(numel, dtype=torch.float32).reshape(base_shape)
                + (rank * 10 + dst) * 100
            ).to("neuron"),
            transpose_dims,
        )
        for dst in range(world_size)
    ]
    assert not input_tensor_list[0].is_contiguous()

    view_shape = make_view(torch.empty(base_shape), transpose_dims).shape
    output_tensor_list = [
        torch.zeros(view_shape, dtype=torch.float32).to("neuron") for _ in range(world_size)
    ]

    dist.all_to_all(output_tensor_list, input_tensor_list)

    for src in range(world_size):
        expected = make_view(
            torch.arange(numel, dtype=torch.float32).reshape(base_shape) + (src * 10 + rank) * 100,
            transpose_dims,
        ).contiguous()
        assert torch.allclose(
            output_tensor_list[src].cpu(), expected
        ), f"Rank {rank} src {src}: {output_tensor_list[src].cpu()} != {expected}"


def run_all_to_all_strided_output(rank, world_size, kwargs):
    """dist.all_to_all with contiguous inputs scattering into non-contiguous
    (transposed) output views.

    kwargs:
      base_shape     - shape of each contiguous input slot
      transpose_dims - (dim0, dim1) for transpose(); None uses .t()
    """
    base_shape = kwargs["base_shape"]
    transpose_dims = kwargs["transpose_dims"]
    numel = math.prod(base_shape)

    input_tensor_list = [
        (torch.arange(numel, dtype=torch.float32).reshape(base_shape) + (rank * 10 + dst) * 100).to(
            "neuron"
        )
        for dst in range(world_size)
    ]

    storage_shape = make_view(torch.empty(base_shape), transpose_dims).shape
    out_storages = [
        torch.zeros(storage_shape, dtype=torch.float32).to("neuron") for _ in range(world_size)
    ]
    output_tensor_list = [make_view(s, transpose_dims) for s in out_storages]
    assert not output_tensor_list[0].is_contiguous()

    dist.all_to_all(output_tensor_list, input_tensor_list)

    for src in range(world_size):
        expected = (
            torch.arange(numel, dtype=torch.float32).reshape(base_shape) + (src * 10 + rank) * 100
        )
        assert torch.allclose(
            output_tensor_list[src].cpu(), expected
        ), f"Rank {rank} src {src}: {output_tensor_list[src].cpu()} != {expected}"


# ─────────────────────────────────────────────────────────────────────────
# Test classes
# ─────────────────────────────────────────────────────────────────────────


class TestStridedCollectives(BaseCollectiveOpTest):
    """Non-contiguous input/output coverage for all_reduce, all_gather, and
    reduce_scatter (world_size 2)."""

    @pytest.mark.parametrize(
        "base_shape,transpose_dims",
        [((3, 4), None), ((2, 3, 4), (0, 2)), ((2, 3, 4, 5), (0, 3))],
        ids=["2d", "3d", "4d"],
    )
    def test_all_reduce_strided(self, base_shape, transpose_dims):
        self.distributed_tester.run_test(
            run_all_reduce_strided, base_shape=base_shape, transpose_dims=transpose_dims
        )

    @pytest.mark.parametrize(
        "base_shape_list,transpose_dims_list",
        [
            ([(3, 4), (4, 6)], [None, None]),
            ([(2, 3, 4), (3, 4, 2)], [(0, 2), (1, 2)]),
            ([(2, 3, 4, 5), (3, 4, 2, 5)], [(0, 3), (0, 3)]),
            ([(3, 4), (2, 3, 4), (2, 3, 4, 5)], [None, (0, 2), (0, 3)]),
        ],
        ids=["2d", "3d", "4d", "mixed"],
    )
    def test_all_reduce_coalesced_strided(self, base_shape_list, transpose_dims_list):
        self.distributed_tester.run_test(
            run_all_reduce_coalesced_strided,
            base_shape_list=base_shape_list,
            transpose_dims=transpose_dims_list,
        )

    @pytest.mark.parametrize(
        "base_shape,transpose_dims",
        [((4, 6), None), ((2, 3, 4), (0, 2)), ((2, 3, 4, 5), (0, 3))],
        ids=["2d", "3d", "4d"],
    )
    def test_all_gather_into_tensor_strided_input(self, base_shape, transpose_dims):
        self.distributed_tester.run_test(
            run_all_gather_into_tensor_strided_input,
            base_shape=base_shape,
            transpose_dims=transpose_dims,
        )

    @pytest.mark.parametrize(
        "base_shape,transpose_dims",
        [((4, 6), None), ((2, 3, 4), (0, 2)), ((2, 3, 4, 5), (0, 3))],
        ids=["2d", "3d", "4d"],
    )
    def test_all_gather_into_tensor_strided_output(self, base_shape, transpose_dims):
        self.distributed_tester.run_test(
            run_all_gather_into_tensor_strided_output,
            base_shape=base_shape,
            transpose_dims=transpose_dims,
        )

    @pytest.mark.parametrize(
        "base_shape,transpose_dims",
        [((4, 6), None), ((2, 3, 4), (0, 2)), ((2, 3, 4, 5), (0, 3))],
        ids=["2d", "3d", "4d"],
    )
    def test_all_gather_base_strided_input(self, base_shape, transpose_dims):
        self.distributed_tester.run_test(
            run_all_gather_base_strided_input, base_shape=base_shape, transpose_dims=transpose_dims
        )

    @pytest.mark.parametrize(
        "base_shape,transpose_dims",
        [((4, 6), None), ((2, 3, 4), (0, 2)), ((2, 3, 4, 5), (0, 3))],
        ids=["2d", "3d", "4d"],
    )
    def test_all_gather_base_strided_output(self, base_shape, transpose_dims):
        self.distributed_tester.run_test(
            run_all_gather_base_strided_output, base_shape=base_shape, transpose_dims=transpose_dims
        )

    @pytest.mark.parametrize(
        "base_shape,transpose_dims",
        [((4, 6), None), ((2, 3, 4), (0, 2)), ((2, 3, 4, 5), (0, 3))],
        ids=["2d", "3d", "4d"],
    )
    def test_all_gather_coalesced_strided_input(self, base_shape, transpose_dims):
        self.distributed_tester.run_test(
            run_all_gather_coalesced_strided_input,
            base_shape=base_shape,
            transpose_dims=transpose_dims,
        )

    @pytest.mark.parametrize(
        "base_shape,transpose_dims",
        [((4, 6), None), ((2, 3, 4), (0, 2)), ((2, 3, 4, 5), (0, 3))],
        ids=["2d", "3d", "4d"],
    )
    def test_all_gather_coalesced_strided_output(self, base_shape, transpose_dims):
        self.distributed_tester.run_test(
            run_all_gather_coalesced_strided_output,
            base_shape=base_shape,
            transpose_dims=transpose_dims,
        )

    @pytest.mark.parametrize(
        "base_shape,transpose_dims",
        [((4, 6), None), ((2, 3, 4), (0, 2)), ((2, 3, 4, 6), (0, 3))],
        ids=["2d", "3d", "4d"],
    )
    def test_reduce_scatter_strided_input(self, base_shape, transpose_dims):
        self.distributed_tester.run_test(
            run_reduce_scatter_strided_input, base_shape=base_shape, transpose_dims=transpose_dims
        )

    @pytest.mark.parametrize(
        "base_shape,transpose_dims",
        [((4, 6), None), ((2, 3, 4), (0, 2)), ((2, 3, 4, 6), (0, 3))],
        ids=["2d", "3d", "4d"],
    )
    def test_reduce_scatter_strided_output(self, base_shape, transpose_dims):
        self.distributed_tester.run_test(
            run_reduce_scatter_strided_output, base_shape=base_shape, transpose_dims=transpose_dims
        )

    @pytest.mark.parametrize(
        "base_shape,transpose_dims",
        [((4, 6), None), ((2, 3, 4), (0, 2)), ((2, 3, 4, 6), (0, 3))],
        ids=["2d", "3d", "4d"],
    )
    def test_reduce_scatter_tensor_strided_input(self, base_shape, transpose_dims):
        self.distributed_tester.run_test(
            run_reduce_scatter_tensor_strided_input,
            base_shape=base_shape,
            transpose_dims=transpose_dims,
        )

    @pytest.mark.parametrize(
        "base_shape,transpose_dims",
        [((4, 6), None), ((2, 3, 4), (0, 2)), ((2, 3, 4, 6), (0, 3))],
        ids=["2d", "3d", "4d"],
    )
    def test_reduce_scatter_tensor_strided_output(self, base_shape, transpose_dims):
        self.distributed_tester.run_test(
            run_reduce_scatter_tensor_strided_output,
            base_shape=base_shape,
            transpose_dims=transpose_dims,
        )

    @pytest.mark.parametrize(
        "base_shape,transpose_dims",
        [((4, 6), None), ((2, 3, 4), (0, 2)), ((2, 3, 4, 6), (0, 3))],
        ids=["2d", "3d", "4d"],
    )
    def test_reduce_scatter_tensor_coalesced_strided_input(self, base_shape, transpose_dims):
        self.distributed_tester.run_test(
            run_reduce_scatter_tensor_coalesced_strided_input,
            base_shape=base_shape,
            transpose_dims=transpose_dims,
        )

    @pytest.mark.parametrize(
        "base_shape,transpose_dims",
        [((4, 6), None), ((2, 3, 4), (0, 2)), ((2, 3, 4, 6), (0, 3))],
        ids=["2d", "3d", "4d"],
    )
    def test_reduce_scatter_tensor_coalesced_strided_output(self, base_shape, transpose_dims):
        self.distributed_tester.run_test(
            run_reduce_scatter_tensor_coalesced_strided_output,
            base_shape=base_shape,
            transpose_dims=transpose_dims,
        )


class TestStridedAllToAll(BaseCollectiveOpTest):
    """Non-contiguous input/output coverage for all_to_all (world_size 4)."""

    @property
    def world_size(self) -> int:
        return 4

    @pytest.mark.parametrize(
        "base_shape,transpose_dims",
        [((2, 3), None), ((2, 3, 4), (0, 2)), ((2, 3, 4, 5), (0, 3))],
        ids=["2d", "3d", "4d"],
    )
    def test_all_to_all_strided_input(self, base_shape, transpose_dims):
        self.distributed_tester.run_test(
            run_all_to_all_strided_input, base_shape=base_shape, transpose_dims=transpose_dims
        )

    @pytest.mark.parametrize(
        "base_shape,transpose_dims",
        [((2, 3), None), ((2, 3, 4), (0, 2)), ((2, 3, 4, 5), (0, 3))],
        ids=["2d", "3d", "4d"],
    )
    def test_all_to_all_strided_output(self, base_shape, transpose_dims):
        self.distributed_tester.run_test(
            run_all_to_all_strided_output, base_shape=base_shape, transpose_dims=transpose_dims
        )

    # all_to_all_single (alltoall_base): splits along dim 0 of the view, so
    # view's dim 0 must be divisible by world_size (4).
    @pytest.mark.parametrize(
        "base_shape,transpose_dims",
        [((8, 4), None), ((4, 3, 4), (0, 2)), ((4, 3, 4, 4), (0, 2))],
        ids=["2d", "3d", "4d"],
    )
    def test_all_to_all_single_strided_input(self, base_shape, transpose_dims):
        self.distributed_tester.run_test(
            run_all_to_all_single_strided_input,
            base_shape=base_shape,
            transpose_dims=transpose_dims,
        )

    @pytest.mark.parametrize(
        "base_shape,transpose_dims",
        [((8, 4), None), ((4, 3, 4), (0, 2)), ((4, 3, 4, 4), (0, 2))],
        ids=["2d", "3d", "4d"],
    )
    def test_all_to_all_single_strided_output(self, base_shape, transpose_dims):
        self.distributed_tester.run_test(
            run_all_to_all_single_strided_output,
            base_shape=base_shape,
            transpose_dims=transpose_dims,
        )
