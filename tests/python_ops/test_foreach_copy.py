"""Tests for MLIR-based aten::_foreach_copy_ implementation."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    requires_nrt_streams_with_host_cc,
    track_neuron_ops,
)


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)


# --- Fused path: contiguous same-dtype splits ---


def test_fused_copy_contiguous_splits():
    """Fused path: both src and dst are adjacent splits of flat buffers."""
    flat_src = torch.randn(1024, dtype=torch.bfloat16, device="neuron")
    flat_dst = torch.zeros(1024, dtype=torch.bfloat16, device="neuron")

    src_splits = list(torch.split(flat_src, [256, 256, 256, 256]))
    dst_splits = list(torch.split(flat_dst, [256, 256, 256, 256]))

    with track_neuron_ops():
        torch._foreach_copy_(dst_splits, src_splits)
        assert_op_runs_on_neuron("_foreach_copy_")

    torch.testing.assert_close(flat_dst.cpu(), flat_src.cpu())


def test_fused_copy_uneven_splits():
    """Fused path with uneven split sizes (like real FSDP param groups)."""
    sizes = [128, 512, 64, 320]
    total = sum(sizes)
    flat_src = torch.randn(total, dtype=torch.float32, device="neuron")
    flat_dst = torch.zeros(total, dtype=torch.float32, device="neuron")

    src_splits = list(torch.split(flat_src, sizes))
    dst_splits = list(torch.split(flat_dst, sizes))

    with track_neuron_ops():
        torch._foreach_copy_(dst_splits, src_splits)
        assert_op_runs_on_neuron("_foreach_copy_")

    torch.testing.assert_close(flat_dst.cpu(), flat_src.cpu())


def test_fused_copy_single_tensor():
    """Fused path with a single tensor (degenerate case)."""
    src = torch.randn(512, dtype=torch.bfloat16, device="neuron")
    dst = torch.zeros(512, dtype=torch.bfloat16, device="neuron")

    with track_neuron_ops():
        torch._foreach_copy_([dst], [src])
        assert_op_runs_on_neuron("_foreach_copy_")

    torch.testing.assert_close(dst.cpu(), src.cpu())


def test_fused_copy_large():
    """Fused path with large tensors (realistic FSDP sizes)."""
    sizes = [4096, 4096, 16384, 4096, 16384, 4096, 4096, 4096, 128, 128, 128]
    total = sum(sizes)
    flat_src = torch.randn(total, dtype=torch.bfloat16, device="neuron")
    flat_dst = torch.zeros(total, dtype=torch.bfloat16, device="neuron")

    src_splits = list(torch.split(flat_src, sizes))
    dst_splits = list(torch.split(flat_dst, sizes))

    with track_neuron_ops():
        torch._foreach_copy_(dst_splits, src_splits)
        assert_op_runs_on_neuron("_foreach_copy_")

    torch.testing.assert_close(flat_dst.cpu(), flat_src.cpu())


# --- Individual path: same dtype, scattered sources ---


def test_individual_copy_scattered_sources():
    """Individual path: src tensors are separate allocations (not contiguous)."""
    srcs = [torch.randn(256, dtype=torch.bfloat16, device="neuron") for _ in range(4)]
    flat_dst = torch.zeros(1024, dtype=torch.bfloat16, device="neuron")
    dst_splits = list(torch.split(flat_dst, [256, 256, 256, 256]))

    with track_neuron_ops():
        torch._foreach_copy_(dst_splits, srcs)
        assert_op_runs_on_neuron("_foreach_copy_")

    for i, src in enumerate(srcs):
        torch.testing.assert_close(dst_splits[i].cpu(), src.cpu())


def test_individual_copy_scattered_both():
    """Individual path: both src and dst are separate allocations."""
    srcs = [torch.randn(128, dtype=torch.float32, device="neuron") for _ in range(3)]
    dsts = [torch.zeros(128, dtype=torch.float32, device="neuron") for _ in range(3)]

    with track_neuron_ops():
        torch._foreach_copy_(dsts, srcs)
        assert_op_runs_on_neuron("_foreach_copy_")

    for dst, src in zip(dsts, srcs, strict=False):
        torch.testing.assert_close(dst.cpu(), src.cpu())


# --- Cross-dtype path ---


def test_cross_dtype_fp32_to_bf16():
    """Cross-dtype: fp32 sources copied into bf16 destinations (FSDP mixed precision)."""
    srcs = [torch.randn(256, dtype=torch.float32, device="neuron") for _ in range(4)]
    flat_dst = torch.zeros(1024, dtype=torch.bfloat16, device="neuron")
    dst_splits = list(torch.split(flat_dst, [256, 256, 256, 256]))

    with track_neuron_ops():
        torch._foreach_copy_(dst_splits, srcs)
        assert_op_runs_on_neuron("_foreach_copy_")

    for i, src in enumerate(srcs):
        torch.testing.assert_close(dst_splits[i].cpu(), src.cpu().to(torch.bfloat16))


def test_cross_dtype_bf16_to_fp32():
    """Cross-dtype: bf16 to fp32."""
    srcs = [torch.randn(128, dtype=torch.bfloat16, device="neuron") for _ in range(3)]
    dsts = [torch.zeros(128, dtype=torch.float32, device="neuron") for _ in range(3)]

    with track_neuron_ops():
        torch._foreach_copy_(dsts, srcs)
        assert_op_runs_on_neuron("_foreach_copy_")

    for dst, src in zip(dsts, srcs, strict=False):
        torch.testing.assert_close(dst.cpu(), src.cpu().to(torch.float32))


# --- Edge cases ---


def test_zero_numel_tensors():
    """Zero-element tensors should be handled."""
    srcs = [torch.empty(0, dtype=torch.float32, device="neuron")]
    dsts = [torch.empty(0, dtype=torch.float32, device="neuron")]

    with track_neuron_ops():
        torch._foreach_copy_(dsts, srcs)
        assert_op_runs_on_neuron("_foreach_copy_")

    assert dsts[0].numel() == 0


def test_mixed_sizes():
    """Tensors with different sizes in the list."""
    srcs = [
        torch.randn(64, dtype=torch.bfloat16, device="neuron"),
        torch.randn(512, dtype=torch.bfloat16, device="neuron"),
        torch.randn(8, dtype=torch.bfloat16, device="neuron"),
    ]
    dsts = [
        torch.zeros(64, dtype=torch.bfloat16, device="neuron"),
        torch.zeros(512, dtype=torch.bfloat16, device="neuron"),
        torch.zeros(8, dtype=torch.bfloat16, device="neuron"),
    ]

    with track_neuron_ops():
        torch._foreach_copy_(dsts, srcs)
        assert_op_runs_on_neuron("_foreach_copy_")

    for dst, src in zip(dsts, srcs, strict=False):
        torch.testing.assert_close(dst.cpu(), src.cpu())


# --- FSDP-realistic scenario ---


def test_fsdp_all_gather_copy_in_pattern():
    """Simulates the exact FSDPv2 all-gather copy-in pattern:
    1st foreach_copy_: scattered params (fp32) -> flat buffer (bf16) [cross-dtype]
    2nd foreach_copy_: flat buffer splits -> AG input splits [fused same-dtype]
    """
    param_sizes = [4096, 4096, 16384, 4096]
    total = sum(param_sizes)

    # 1st foreach_copy_: dtype cast (fp32 params -> bf16 flat buffer)
    params_fp32 = [torch.randn(s, dtype=torch.float32, device="neuron") for s in param_sizes]
    flat_bf16 = torch.zeros(total, dtype=torch.bfloat16, device="neuron")
    bf16_splits = list(torch.split(flat_bf16, param_sizes))

    with track_neuron_ops():
        torch._foreach_copy_(bf16_splits, params_fp32)
        assert_op_runs_on_neuron("_foreach_copy_")

    for split, param in zip(bf16_splits, params_fp32, strict=False):
        torch.testing.assert_close(split.cpu(), param.cpu().to(torch.bfloat16))

    # 2nd foreach_copy_: AG buffer copy (bf16 flat -> bf16 flat, should fuse)
    ag_input = torch.zeros(total, dtype=torch.bfloat16, device="neuron")
    ag_splits = list(torch.split(ag_input, param_sizes))

    with track_neuron_ops():
        torch._foreach_copy_(ag_splits, bf16_splits)
        assert_op_runs_on_neuron("_foreach_copy_")

    torch.testing.assert_close(ag_input.cpu(), flat_bf16.cpu())


# --- Which dtypes reach the fused kernel ---


def test_unsupported_dtype_float64():
    """float64 has no lowering, so the list goes pair by pair through copy_."""
    srcs = [torch.randn(128, dtype=torch.float64, device="neuron") for _ in range(3)]
    dsts = [torch.zeros(128, dtype=torch.float64, device="neuron") for _ in range(3)]

    assert _route_of(dsts, srcs) == "copy_"

    for dst, src in zip(dsts, srcs, strict=False):
        torch.testing.assert_close(dst.cpu(), src.cpu())


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.bool])
def test_narrow_dtypes_are_copied_correctly(dtype):
    """The narrow types the fused kernel takes but copy_'s own lowering does not."""
    hi = 2 if dtype == torch.bool else 100
    srcs = [torch.randint(0, hi, (64,), dtype=dtype, device="neuron") for _ in range(3)]
    dsts = [torch.zeros(64, dtype=dtype, device="neuron") for _ in range(3)]

    # Values only. For uint8, int8 and int16 the fused path issues its own
    # aten::copy_ while staging the dtype, so _route_of cannot tell the two routes
    # apart here even though the handler reports one_kernel=1 for all five.
    torch._foreach_copy_(dsts, srcs)
    torch.neuron.synchronize()

    for dst, src in zip(dsts, srcs, strict=False):
        torch.testing.assert_close(dst.cpu(), src.cpu())


@pytest.mark.needs_process_isolation(
    reason="requires_nrt_streams_with_host_cc sets the stream flags, which the C++ "
    "stream configuration keeps after first use."
)
@requires_nrt_streams_with_host_cc
def test_nondefault_stream_same_dtype_raw_nrt():
    """Non-default stream + hostCC: same-dtype contiguous range routes to raw NRT."""
    flat_src = torch.randn(1024, dtype=torch.bfloat16, device="neuron")
    flat_dst = torch.zeros(1024, dtype=torch.bfloat16, device="neuron")
    src_splits = list(torch.split(flat_src, [256, 256, 256, 256]))
    dst_splits = list(torch.split(flat_dst, [256, 256, 256, 256]))

    stream = torch.neuron.Stream()
    stream.wait_stream(torch.neuron.current_stream())
    with torch.neuron.stream(stream):
        torch._foreach_copy_(dst_splits, src_splits)
    stream.synchronize()
    torch.testing.assert_close(flat_dst.cpu(), flat_src.cpu())


@pytest.mark.needs_process_isolation(
    reason="requires_nrt_streams_with_host_cc sets the stream flags, which the C++ "
    "stream configuration keeps after first use."
)
@requires_nrt_streams_with_host_cc
def test_nondefault_stream_scattered_raw_nrt():
    """Non-default stream + hostCC: scattered same-dtype routes per-tensor to raw NRT."""
    srcs = [torch.randn(128, dtype=torch.bfloat16, device="neuron") for _ in range(3)]
    dsts = [torch.zeros(128, dtype=torch.bfloat16, device="neuron") for _ in range(3)]

    stream = torch.neuron.Stream()
    stream.wait_stream(torch.neuron.current_stream())
    with torch.neuron.stream(stream):
        torch._foreach_copy_(dsts, srcs)
    stream.synchronize()
    for d, s in zip(dsts, srcs, strict=False):
        torch.testing.assert_close(d.cpu(), s.cpu())


@pytest.mark.needs_process_isolation(
    reason="requires_nrt_streams_with_host_cc sets the stream flags, which the C++ "
    "stream configuration keeps after first use."
)
@requires_nrt_streams_with_host_cc
def test_nondefault_stream_neg_src_falls_back():
    """Non-default stream + hostCC: src with .is_neg() falls through to dst.copy_(src)."""
    cpu_srcs = [torch.randn(64, dtype=torch.bfloat16) for _ in range(2)]
    srcs = [s.to("neuron")._neg_view() for s in cpu_srcs]
    dsts = [torch.zeros(64, dtype=torch.bfloat16, device="neuron") for _ in range(2)]

    stream = torch.neuron.Stream()
    stream.wait_stream(torch.neuron.current_stream())
    with torch.neuron.stream(stream):
        torch._foreach_copy_(dsts, srcs)
    stream.synchronize()
    for d, s in zip(dsts, cpu_srcs, strict=False):
        torch.testing.assert_close(d.cpu(), -s)


@pytest.mark.parametrize(
    ("src_dtype", "dst_dtype"),
    [
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float16),
        (torch.float16, torch.float32),
    ],
)
@pytest.mark.needs_process_isolation(reason="HostCC stream flags remain fixed after first use.")
@requires_nrt_streams_with_host_cc
def test_nondefault_stream_cross_dtype_uses_copy_dispatch(src_dtype, dst_dtype):
    """Cross-dtype foreach remains functional through per-pair copy_ dispatch."""
    cpu_srcs = [torch.randn(64, dtype=src_dtype) for _ in range(2)]
    srcs = [src.to("neuron") for src in cpu_srcs]
    dsts = [torch.zeros(64, dtype=dst_dtype, device="neuron") for _ in range(2)]

    stream = torch.neuron.Stream()
    stream.wait_stream(torch.neuron.current_stream())
    with torch.neuron.stream(stream):
        torch._foreach_copy_(dsts, srcs)
    stream.synchronize()
    for dst, src in zip(dsts, cpu_srcs, strict=False):
        torch.testing.assert_close(dst.cpu(), src.to(dst_dtype))


# ── Route pinning ──────────────────────────────────────────────────────────
#
# All three routes report the same executed op name, so a values-only test cannot
# tell them apart: deleting the coalescing block entirely leaves every assertion
# above green. These tests pin the route instead, using the one externally visible
# difference — the coalesced and per-pair routes redispatch aten::copy_, the fused
# kernel does not. The compile count is checked alongside it to confirm the call
# being measured was a cache hit and not a fresh compile.


def _compiled_entries():
    torch.neuron.synchronize()
    return torch_neuronx._C._get_compilation_cache_stats()["total_entries"]


def _route_of(dsts, srcs):
    """Run the op once on a warm cache and report which route handled it."""
    torch._foreach_copy_(dsts, srcs)  # warm: compile whatever this shape needs
    torch.neuron.synchronize()
    before = _compiled_entries()
    torch_neuronx.clear_op_tracking()
    torch._foreach_copy_(dsts, srcs)
    torch.neuron.synchronize()
    ops = torch_neuronx.get_executed_ops()
    assert _compiled_entries() == before, "second call recompiled; not a cache hit"
    return "copy_" if "aten::copy_" in ops else "kernel"


def test_route_adjacent_views_coalesce():
    """Adjacent 1-D slices of one buffer must collapse to a single aten::copy_."""
    flat_src = torch.randn(2048, device="neuron")
    flat_dst = torch.zeros(2048, device="neuron")
    srcs = list(torch.split(flat_src, 512))
    dsts = list(torch.split(flat_dst, 512))
    assert _route_of(dsts, srcs) == "copy_"
    torch.testing.assert_close(flat_dst.cpu(), flat_src.cpu())


def test_route_scattered_uses_the_kernel():
    """Separate allocations cannot coalesce, so the fused kernel must handle them."""
    srcs = [torch.randn(64, 64, device="neuron") for _ in range(3)]
    dsts = [torch.zeros(64, 64, device="neuron") for _ in range(3)]
    assert _route_of(dsts, srcs) == "kernel"
    for d, s in zip(dsts, srcs, strict=False):
        torch.testing.assert_close(d.cpu(), s.cpu())


def test_route_overlapping_spans_go_pair_by_pair():
    """Overlapping src/dst spans must reach neither batch route.

    Coalescing is refused because one copy over the union would read bytes it has
    already written. The fused kernel is refused for the same reason: it is handed
    both buffers with no declared aliasing relationship, so the order of its reads
    and writes is the compiler's choice, while ATen copies pair by pair. Both
    refusals land on aten::copy_, and here that can only be the pair route, since
    the coalesced route provably declined.
    """
    flat = torch.randn(1536, device="neuron")
    srcs = list(torch.split(flat[:1024], 512))
    dsts = list(torch.split(flat[512:1536], 512))
    assert _route_of(dsts, srcs) == "copy_"


def test_shared_buffer_without_overlap_still_uses_the_kernel():
    """Destinations tiling one buffer do not overlap, so the fused route stands.

    This is the FSDP destination shape. The destination guard compares byte ranges
    rather than buffer identity precisely so that a tiled list is not pushed onto
    the pair route.
    """
    dst_buf = torch.zeros(3 * 64, device="neuron")
    dsts = list(torch.split(dst_buf, 64))
    srcs = [torch.randn(64, device="neuron") for _ in range(3)]
    assert _route_of(dsts, srcs) == "kernel"
    torch.testing.assert_close(dst_buf.cpu(), torch.cat([s.cpu() for s in srcs]))


# ── Aliased lists: values, not routes ──────────────────────────────────────
#
# Each of these calls the op exactly ONCE on tensors that have never been written.
# _route_of runs the op twice to warm the cache, which would not do here: the first
# call leaves the buffers in a state where ATen's answer and the fused kernel's
# answer coincide, hiding the divergence on the second.


def test_aliased_dst_and_src_matches_cpu():
    """A tensor used as both a destination and a source follows ATen's order.

    ATen copies pair by pair, so pair 1 reads what pair 0 already wrote and an
    attempted swap collapses to "both hold y". A snapshot of both sources taken up
    front would swap them for real.
    """
    x_cpu, y_cpu = torch.tensor([1.0, 1.0]), torch.tensor([9.0, 9.0])
    torch._foreach_copy_([x_cpu, y_cpu], [y_cpu, x_cpu])

    x = torch.tensor([1.0, 1.0], device="neuron")
    y = torch.tensor([9.0, 9.0], device="neuron")
    torch._foreach_copy_([x, y], [y, x])
    torch.neuron.synchronize()

    torch.testing.assert_close(x.cpu(), x_cpu)
    torch.testing.assert_close(y.cpu(), y_cpu)


def test_repeated_destination_matches_cpu():
    """The same tensor twice as a destination: the last write in list order wins."""
    d_cpu = torch.zeros(2)
    torch._foreach_copy_([d_cpu, d_cpu], [torch.tensor([1.0, 1.0]), torch.tensor([2.0, 2.0])])

    d = torch.zeros(2, device="neuron")
    torch._foreach_copy_(
        [d, d],
        [
            torch.tensor([1.0, 1.0], device="neuron"),
            torch.tensor([2.0, 2.0], device="neuron"),
        ],
    )
    torch.neuron.synchronize()

    torch.testing.assert_close(d.cpu(), d_cpu)


# ── Cache miss and cache hit are different code paths ──────────────────────
#
# On a hit the kernel's inputs are rebuilt in C++ from the raw arguments using
# recorded indices, which is where this op's argument-order bug lived: values were
# correct on the miss that compiled the entry and wrong on every hit after.


@pytest.mark.parametrize(
    "dtype_pair",
    [(torch.float32, torch.float32), (torch.float32, torch.bfloat16)],
    ids=["same-dtype", "cross-dtype"],
)
def test_scattered_correct_on_miss_and_hit(dtype_pair):
    src_dtype, dst_dtype = dtype_pair
    for _ in range(2):  # first iteration compiles, second replays from cache
        srcs = [torch.randn(128, dtype=src_dtype, device="neuron") for _ in range(3)]
        dsts = [torch.zeros(128, dtype=dst_dtype, device="neuron") for _ in range(3)]
        torch._foreach_copy_(dsts, srcs)
        torch.neuron.synchronize()
        for d, s in zip(dsts, srcs, strict=False):
            torch.testing.assert_close(d.cpu(), s.cpu().to(dst_dtype))


# ── Lists the batch routes must not take ───────────────────────────────────


def test_shape_mismatch_broadcasts_like_cpu():
    """copy_ broadcasts src into dst; neither batch route does, so this pair has
    to go one at a time."""
    dst_cpu, src_cpu = torch.zeros(4, 4), torch.ones(1)
    torch._foreach_copy_([dst_cpu], [src_cpu])
    dsts = [torch.zeros(4, 4, device="neuron")]
    torch._foreach_copy_(dsts, [torch.ones(1, device="neuron")])
    torch.neuron.synchronize()
    torch.testing.assert_close(dsts[0].cpu(), dst_cpu)


def test_equal_totals_with_mismatched_pair_shapes_raises_like_cpu():
    """Both sides tile the same 8-element span, so a totals check passes, but the
    pairs are 4+4 against 2+6 and the pairwise copies reject those shapes."""
    with pytest.raises(RuntimeError):
        torch._foreach_copy_(
            list(torch.split(torch.zeros(8), [4, 4])),
            list(torch.split(torch.randn(8), [2, 6])),
        )
    with pytest.raises(RuntimeError):
        torch._foreach_copy_(
            list(torch.split(torch.zeros(8, device="neuron"), [4, 4])),
            list(torch.split(torch.randn(8, device="neuron"), [2, 6])),
        )
        torch.neuron.synchronize()


@pytest.mark.parametrize("device", ["cpu", "neuron"])
def test_incompatible_zero_element_shapes_raise(device):
    """Both sides are empty, but the shapes still have to broadcast.

    (0, 3) does not stretch to (0, 2) -- 3 is neither equal to 2 nor 1 -- so CPU
    raises even though neither tensor holds an element. A handler that skipped a
    pair on numel() == 0 alone would return quietly instead, which is why the pair
    route hands empty pairs to copy_ rather than short-circuiting them itself.
    """
    with pytest.raises(RuntimeError):
        torch._foreach_copy_([torch.zeros(0, 2, device=device)], [torch.zeros(0, 3, device=device)])
        if device == "neuron":
            torch.neuron.synchronize()


@pytest.mark.parametrize("device", ["cpu", "neuron"])
def test_matching_zero_element_shapes_are_a_noop(device):
    """The companion case: empty and broadcast-compatible must stay silent."""
    dsts = [torch.zeros(0, 2, device=device)]
    torch._foreach_copy_(dsts, [torch.zeros(0, 2, device=device)])
    if device == "neuron":
        torch.neuron.synchronize()
    assert dsts[0].numel() == 0
    assert tuple(dsts[0].shape) == (0, 2)


def test_mixed_empty_and_non_empty_list():
    """A zero-size entry is returned unchanged so the tracer drops it, which
    shortens the output list relative to the raw arguments."""
    for _ in range(2):
        srcs = [
            torch.randn(64, device="neuron"),
            torch.empty(0, device="neuron"),
            torch.randn(32, device="neuron"),
        ]
        dsts = [
            torch.zeros(64, device="neuron"),
            torch.empty(0, device="neuron"),
            torch.zeros(32, device="neuron"),
        ]
        torch._foreach_copy_(dsts, srcs)
        torch.neuron.synchronize()
        torch.testing.assert_close(dsts[0].cpu(), srcs[0].cpu())
        assert dsts[1].numel() == 0
        torch.testing.assert_close(dsts[2].cpu(), srcs[2].cpu())


# ── ATen's own argument checks, which the handler now gates ────────────────


@pytest.mark.parametrize(
    "n_dst,n_src,match",
    [(0, 0, "at least one tensor"), (3, 2, "same number of tensors")],
    ids=["empty-lists", "length-mismatch"],
)
def test_argument_checks_raise(n_dst, n_src, match):
    dsts = [torch.zeros(8, device="neuron") for _ in range(n_dst)]
    srcs = [torch.randn(8, device="neuron") for _ in range(n_src)]
    with pytest.raises(RuntimeError, match=match):
        torch._foreach_copy_(dsts, srcs)
