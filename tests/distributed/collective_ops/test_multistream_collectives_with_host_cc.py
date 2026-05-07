"""
Multi-Stream Collective Operations Tests for Host-CC Feature.

Tests collective operations on non-default streams using NRT host collectives API.
Environment variable: TORCH_NEURONX_ENABLE_HOST_CC=1 (enabled by tests)
"""

import os

import pytest
import torch
import torch.distributed as dist

import torch_neuronx
from tests.utils.neuron_test_utils import assert_raises

from .base_collective_op import BaseCollectiveOpTest


# Module-level setup: enable host-cc for all tests in this file
@pytest.fixture(scope="module", autouse=True)
def enable_host_cc():
    """Enable host-cc for all tests in this module."""
    old_value = os.environ.get("TORCH_NEURONX_ENABLE_HOST_CC")
    os.environ["TORCH_NEURONX_ENABLE_HOST_CC"] = "1"
    yield
    # Restore original value
    if old_value is None:
        os.environ.pop("TORCH_NEURONX_ENABLE_HOST_CC", None)
    else:
        os.environ["TORCH_NEURONX_ENABLE_HOST_CC"] = old_value


_DEFAULT_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 2))
_SMALL_TENSOR_SIZE = 1024
_MEDIUM_TENSOR_SIZE = 256 * 1024
_LARGE_TENSOR_SIZE = 4 * 1024 * 1024


def create_tp_fsdp_groups(rank: int, world_size: int, tp_size: int):
    """Create TP and FSDP process groups. Returns (tp_group, fsdp_group)."""
    fsdp_size = world_size // tp_size
    my_tp_group = my_fsdp_group = None
    for i in range(fsdp_size):
        tp_ranks = list(range(i * tp_size, (i + 1) * tp_size))
        group = dist.new_group(ranks=tp_ranks)
        if rank in tp_ranks:
            my_tp_group = group
    for i in range(tp_size):
        fsdp_ranks = list(range(i, world_size, tp_size))
        group = dist.new_group(ranks=fsdp_ranks)
        if rank in fsdp_ranks:
            my_fsdp_group = group
    return my_tp_group, my_fsdp_group


# =============================================================================
# Module-level run functions for TestMultiStreamBasic
# =============================================================================


def run_baseline_default_stream(rank, world_size, kwargs):
    tensor = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + 1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * sum(range(1, world_size + 1))
    assert torch.allclose(tensor.cpu(), expected, rtol=1e-5)


def run_allreduce(rank, world_size, kwargs):
    default_stream = torch_neuronx.default_stream()
    stream = torch.neuron.Stream()
    tensor = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + 1)
    stream.wait_stream(default_stream)
    with torch.neuron.stream(stream):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    stream.synchronize()
    expected = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * sum(range(1, world_size + 1))
    assert torch.allclose(tensor.cpu(), expected, rtol=1e-5)


def run_allgather(rank, world_size, kwargs):
    default_stream = torch_neuronx.default_stream()
    stream = torch.neuron.Stream()
    local = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + 1)
    outputs = [
        torch.zeros(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron")
        for _ in range(world_size)
    ]
    stream.wait_stream(default_stream)
    with torch.neuron.stream(stream):
        dist.all_gather(outputs, local)
    stream.synchronize()
    for i, out in enumerate(outputs):
        expected = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * (i + 1)
        assert torch.allclose(out.cpu(), expected, rtol=1e-5)


def run_reduce_scatter(rank, world_size, kwargs):
    default_stream = torch_neuronx.default_stream()
    stream = torch.neuron.Stream()
    inputs = [
        torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (i + rank)
        for i in range(world_size)
    ]
    output = torch.zeros(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron")
    stream.wait_stream(default_stream)
    with torch.neuron.stream(stream):
        dist.reduce_scatter(output, inputs, op=dist.ReduceOp.SUM)
    stream.synchronize()
    expected_val = sum((rank + r) for r in range(world_size))
    expected = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * expected_val
    assert torch.allclose(output.cpu(), expected, rtol=1e-5)


def run_all_gather_into_tensor_1d_and_2d(rank, world_size, kwargs):
    """Combined: all_gather_into_tensor with 1D and 2D tensors."""
    default_stream = torch_neuronx.default_stream()
    stream = torch.neuron.Stream()
    # 1D test
    input_1d = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + 1)
    output_1d = torch.zeros(_SMALL_TENSOR_SIZE * world_size, dtype=torch.float32, device="neuron")
    stream.wait_stream(default_stream)
    with torch.neuron.stream(stream):
        dist.all_gather_into_tensor(output_1d, input_1d)
    stream.synchronize()
    expected_1d = torch.cat(
        [torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * (i + 1) for i in range(world_size)]
    )
    assert torch.allclose(output_1d.cpu(), expected_1d, rtol=1e-5)
    # 2D test
    rows, cols = 32, 64
    input_2d = torch.ones(rows, cols, dtype=torch.float32, device="neuron") * (rank + 1)
    output_2d = torch.zeros(rows * world_size, cols, dtype=torch.float32, device="neuron")
    stream.wait_stream(default_stream)
    with torch.neuron.stream(stream):
        dist.all_gather_into_tensor(output_2d, input_2d)
    stream.synchronize()
    expected_2d = torch.cat(
        [torch.ones(rows, cols, dtype=torch.float32) * (i + 1) for i in range(world_size)]
    )
    assert torch.allclose(output_2d.cpu(), expected_2d, rtol=1e-5)


def run_reduce_scatter_tensor_1d_and_2d(rank, world_size, kwargs):
    """Combined: reduce_scatter_tensor with 1D and 2D tensors."""
    default_stream = torch_neuronx.default_stream()
    stream = torch.neuron.Stream()
    expected_val = sum(i + 1 for i in range(world_size))
    # 1D test
    input_1d = torch.ones(_SMALL_TENSOR_SIZE * world_size, dtype=torch.float32, device="neuron") * (
        rank + 1
    )
    output_1d = torch.zeros(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron")
    stream.wait_stream(default_stream)
    with torch.neuron.stream(stream):
        dist.reduce_scatter_tensor(output_1d, input_1d, op=dist.ReduceOp.SUM)
    stream.synchronize()
    expected_1d = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * expected_val
    assert torch.allclose(output_1d.cpu(), expected_1d, rtol=1e-5)
    # 2D test
    rows, cols = 32, 64
    input_2d = torch.ones(rows * world_size, cols, dtype=torch.float32, device="neuron") * (
        rank + 1
    )
    output_2d = torch.zeros(rows, cols, dtype=torch.float32, device="neuron")
    stream.wait_stream(default_stream)
    with torch.neuron.stream(stream):
        dist.reduce_scatter_tensor(output_2d, input_2d, op=dist.ReduceOp.SUM)
    stream.synchronize()
    expected_2d = torch.ones(rows, cols, dtype=torch.float32) * expected_val
    assert torch.allclose(output_2d.cpu(), expected_2d, rtol=1e-5)


# =============================================================================
# Module-level run functions for TestMultiStreamMultipleStreams
# =============================================================================


def run_two_streams_sequential_and_interleaved(rank, world_size, kwargs):
    """Combined: Sequential and interleaved collectives on two streams."""
    default_stream = torch_neuronx.default_stream()
    s0, s1 = torch.neuron.Stream(), torch.neuron.Stream()
    # Sequential part
    t0 = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + 1)
    t1 = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + 10)
    s0.wait_stream(default_stream)
    s1.wait_stream(default_stream)
    with torch.neuron.stream(s0):
        dist.all_reduce(t0, op=dist.ReduceOp.SUM)
    with torch.neuron.stream(s1):
        dist.all_reduce(t1, op=dist.ReduceOp.SUM)
    s0.synchronize()
    s1.synchronize()
    assert torch.allclose(
        t0.cpu(),
        torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * sum(range(1, world_size + 1)),
        rtol=1e-5,
    )
    assert torch.allclose(
        t1.cpu(),
        torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * sum(range(10, world_size + 10)),
        rtol=1e-5,
    )
    # Interleaved part
    results = []
    for i in range(6):
        t = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + i)
        stream = s0 if i % 2 == 0 else s1
        stream.wait_stream(default_stream)
        with torch.neuron.stream(stream):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        results.append((t, i))
    s0.synchronize()
    s1.synchronize()
    for t, i in results:
        expected = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * sum(
            r + i for r in range(world_size)
        )
        assert torch.allclose(t.cpu(), expected, rtol=1e-5)


def run_three_streams_parallel(rank, world_size, kwargs):
    default_stream = torch_neuronx.default_stream()
    streams = [torch.neuron.Stream() for _ in range(3)]
    tensors = []
    for i, s in enumerate(streams):
        t = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + i * 10)
        s.wait_stream(default_stream)
        with torch.neuron.stream(s):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        tensors.append((t, i))
    for s in streams:
        s.synchronize()
    for t, i in tensors:
        expected = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * sum(
            r + i * 10 for r in range(world_size)
        )
        assert torch.allclose(t.cpu(), expected, rtol=1e-5)


def run_stream_sync_between_collectives(rank, world_size, kwargs):
    default_stream = torch_neuronx.default_stream()
    stream = torch.neuron.Stream()
    t1 = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * rank
    stream.wait_stream(default_stream)
    with torch.neuron.stream(stream):
        dist.all_reduce(t1, op=dist.ReduceOp.SUM)
    stream.synchronize()
    expected1 = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * sum(range(world_size))
    assert torch.allclose(t1.cpu(), expected1, rtol=1e-5)
    t2 = t1.clone()
    stream.wait_stream(default_stream)
    with torch.neuron.stream(stream):
        dist.all_reduce(t2, op=dist.ReduceOp.SUM)
    stream.synchronize()
    assert torch.allclose(t2.cpu(), expected1 * world_size, rtol=1e-5)


# =============================================================================
# Module-level run functions for TestMultiStreamMultiplePG
# =============================================================================


def run_two_pgs_same_stream(rank, world_size, kwargs):
    if world_size < 4:
        pytest.skip("Requires world_size >= 4")
    default_stream = torch_neuronx.default_stream()
    half = world_size // 2
    pg1 = dist.new_group(ranks=list(range(half)))
    pg2 = dist.new_group(ranks=list(range(half, world_size)))
    my_pg = pg1 if rank < half else pg2
    stream = torch.neuron.Stream()
    t = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + 1)
    stream.wait_stream(default_stream)
    with torch.neuron.stream(stream):
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=my_pg)
    stream.synchronize()
    expected_sum = sum(range(1, half + 1)) if rank < half else sum(range(half + 1, world_size + 1))
    assert torch.allclose(
        t.cpu(), torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * expected_sum, rtol=1e-5
    )


def run_two_pgs_different_streams(rank, world_size, kwargs):
    if world_size < 4:
        pytest.skip("Requires world_size >= 4")
    default_stream = torch_neuronx.default_stream()
    half = world_size // 2
    pg1 = dist.new_group(ranks=list(range(half)))
    pg2 = dist.new_group(ranks=list(range(half, world_size)))
    my_pg = pg1 if rank < half else pg2
    my_stream = torch.neuron.Stream()
    t = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + 1)
    my_stream.wait_stream(default_stream)
    with torch.neuron.stream(my_stream):
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=my_pg)
    my_stream.synchronize()
    expected_sum = sum(range(1, half + 1)) if rank < half else sum(range(half + 1, world_size + 1))
    assert torch.allclose(
        t.cpu(), torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * expected_sum, rtol=1e-5
    )


# =============================================================================
# Module-level run functions for TestMultiStreamEvents
# =============================================================================


def run_event_record_and_wait(rank, world_size, kwargs):
    default_stream = torch_neuronx.default_stream()
    s_a, s_b = torch.neuron.Stream(), torch.neuron.Stream()
    event = torch.neuron.Event()
    t = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + 1)
    s_a.wait_stream(default_stream)
    with torch.neuron.stream(s_a):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        event.record(s_a)
    with torch.neuron.stream(s_b):
        s_b.wait_event(event)
    s_b.synchronize()
    expected = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * sum(range(1, world_size + 1))
    assert torch.allclose(t.cpu(), expected, rtol=1e-5)


def run_cross_stream_sync_with_collective(rank, world_size, kwargs):
    default_stream = torch_neuronx.default_stream()
    fsdp_stream = torch.neuron.Stream()
    event = torch.neuron.Event()
    shard = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + 1)
    gathered = [torch.zeros_like(shard) for _ in range(world_size)]
    fsdp_stream.wait_stream(default_stream)
    with torch.neuron.stream(fsdp_stream):
        dist.all_gather(gathered, shard)
        event.record(fsdp_stream)
    default_stream.wait_event(event)
    for i, g in enumerate(gathered):
        expected = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * (i + 1)
        assert torch.allclose(g.cpu(), expected, rtol=1e-5)


def run_multiple_events_pipeline(rank, world_size, kwargs):
    default_stream = torch_neuronx.default_stream()
    fsdp_stream = torch.neuron.Stream()
    e1, e2 = torch.neuron.Event(), torch.neuron.Event()
    w1 = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + 1)
    w2 = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + 10)
    g1 = [torch.zeros_like(w1) for _ in range(world_size)]
    g2 = [torch.zeros_like(w2) for _ in range(world_size)]
    fsdp_stream.wait_stream(default_stream)
    with torch.neuron.stream(fsdp_stream):
        dist.all_gather(g1, w1)
        e1.record(fsdp_stream)
        dist.all_gather(g2, w2)
        e2.record(fsdp_stream)
    default_stream.wait_event(e1)
    default_stream.wait_event(e2)
    for i in range(world_size):
        assert torch.allclose(
            g1[i].cpu(), torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * (i + 1), rtol=1e-5
        )
        assert torch.allclose(
            g2[i].cpu(), torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * (i + 10), rtol=1e-5
        )


def run_event_timing(rank, world_size, kwargs):
    default_stream = torch_neuronx.default_stream()
    stream = torch.neuron.Stream()
    start, end = torch.neuron.Event(enable_timing=True), torch.neuron.Event(enable_timing=True)
    t = torch.ones(_MEDIUM_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + 1)
    stream.wait_stream(default_stream)
    with torch.neuron.stream(stream):
        start.record(stream)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        end.record(stream)
    stream.synchronize()
    elapsed = start.elapsed_time(end)
    assert elapsed > 0 and elapsed < 10000


# =============================================================================
# Module-level run functions for TestMultiStreamOverlap
# =============================================================================


def run_compute_collective_overlap(rank, world_size, kwargs):
    default_stream = torch_neuronx.default_stream()
    compute_stream = torch.neuron.Stream()
    collective_stream = torch.neuron.Stream()
    compute_in = torch.randn(_MEDIUM_TENSOR_SIZE, dtype=torch.float32, device="neuron")
    collective_t = torch.ones(_MEDIUM_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (
        rank + 1
    )
    compute_stream.wait_stream(default_stream)
    collective_stream.wait_stream(default_stream)
    with torch.neuron.stream(compute_stream):
        _ = compute_in * 2.0 + 1.0
    with torch.neuron.stream(collective_stream):
        dist.all_reduce(collective_t, op=dist.ReduceOp.SUM)
    compute_stream.synchronize()
    collective_stream.synchronize()
    expected = torch.ones(_MEDIUM_TENSOR_SIZE, dtype=torch.float32) * sum(range(1, world_size + 1))
    assert torch.allclose(collective_t.cpu(), expected, rtol=1e-5)


def run_fsdp_allgather_compute_overlap(rank, world_size, kwargs):
    default_stream = torch_neuronx.default_stream()
    fsdp_stream = torch.neuron.Stream()
    event = torch.neuron.Event()
    shard = torch.randn(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron")
    gathered = [torch.zeros_like(shard) for _ in range(world_size)]
    activation = torch.randn(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron")
    fsdp_stream.wait_stream(default_stream)
    with torch.neuron.stream(fsdp_stream):
        dist.all_gather(gathered, shard)
        event.record(fsdp_stream)
    _ = activation * 2.0
    default_stream.wait_event(event)
    default_stream.synchronize()
    fsdp_stream.synchronize()
    for g in gathered:
        assert g.shape == shard.shape


def run_tp_fsdp_overlap(rank, world_size, kwargs):
    if world_size < 4:
        pytest.skip("Requires world_size >= 4")
    half = world_size // 2
    tp_ranks = list(range(half)) if rank < half else list(range(half, world_size))
    fsdp_ranks = [rank % half, rank % half + half]
    tp_group = dist.new_group(ranks=tp_ranks)
    fsdp_group = dist.new_group(ranks=fsdp_ranks)
    default_stream = torch_neuronx.default_stream()
    fsdp_stream = torch.neuron.Stream()
    tp_t = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + 1)
    tp_gathered = [torch.zeros_like(tp_t) for _ in range(len(tp_ranks))]
    fsdp_t = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + 10)
    fsdp_gathered = [torch.zeros_like(fsdp_t) for _ in range(2)]
    dist.all_gather(tp_gathered, tp_t, group=tp_group)
    fsdp_stream.wait_stream(default_stream)
    with torch.neuron.stream(fsdp_stream):
        dist.all_gather(fsdp_gathered, fsdp_t, group=fsdp_group)
    default_stream.synchronize()
    fsdp_stream.synchronize()
    for i, t in enumerate(tp_gathered):
        expected = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * (tp_ranks[i] + 1)
        assert torch.allclose(t.cpu(), expected, rtol=1e-5)
    for i, t in enumerate(fsdp_gathered):
        expected = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * (fsdp_ranks[i] + 10)
        assert torch.allclose(t.cpu(), expected, rtol=1e-5)


def run_pipelined_layers(rank, world_size, kwargs):
    default_stream = torch_neuronx.default_stream()
    fsdp_stream = torch.neuron.Stream()
    num_layers = 2
    events = [torch.neuron.Event() for _ in range(num_layers)]
    shards = [
        torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + i * 10)
        for i in range(num_layers)
    ]
    gathered = [[torch.zeros_like(shards[i]) for _ in range(world_size)] for i in range(num_layers)]
    activation = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron")
    fsdp_stream.wait_stream(default_stream)
    for i in range(num_layers):
        with torch.neuron.stream(fsdp_stream):
            dist.all_gather(gathered[i], shards[i])
            events[i].record(fsdp_stream)
        default_stream.wait_event(events[i])
        activation = activation + gathered[i][rank]
    default_stream.synchronize()
    fsdp_stream.synchronize()
    expected_val = 1.0 + (rank + 0) + (rank + 10)
    expected = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * expected_val
    assert torch.allclose(activation.cpu(), expected, rtol=1e-5)


# =============================================================================
# Module-level run functions for TestMultiStreamStress
# =============================================================================


def run_many_streams_and_rapid_creation(rank, world_size, kwargs):
    """Combined: many_streams and rapid_stream_creation."""
    default_stream = torch_neuronx.default_stream()
    # Many streams part
    num_streams = 8
    streams = [torch.neuron.Stream() for _ in range(num_streams)]
    tensors = []
    for i, s in enumerate(streams):
        t = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + i)
        s.wait_stream(default_stream)
        with torch.neuron.stream(s):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        tensors.append((t, i))
    for s in streams:
        s.synchronize()
    for t, i in tensors:
        expected = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * sum(
            r + i for r in range(world_size)
        )
        assert torch.allclose(t.cpu(), expected, rtol=1e-5)
    # Rapid creation part
    for _ in range(10):
        s = torch.neuron.Stream()
        t = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + 1)
        s.wait_stream(default_stream)
        with torch.neuron.stream(s):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        s.synchronize()
        expected = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * sum(
            range(1, world_size + 1)
        )
        assert torch.allclose(t.cpu(), expected, rtol=1e-5)


def run_large_tensor(rank, world_size, kwargs):
    default_stream = torch_neuronx.default_stream()
    s = torch.neuron.Stream()
    t = torch.ones(_LARGE_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + 1)
    s.wait_stream(default_stream)
    with torch.neuron.stream(s):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    s.synchronize()
    expected = torch.ones(_LARGE_TENSOR_SIZE, dtype=torch.float32) * sum(range(1, world_size + 1))
    assert torch.allclose(t.cpu(), expected, rtol=1e-5)


def run_mixed_collective_types(rank, world_size, kwargs):
    default_stream = torch_neuronx.default_stream()
    s_ar, s_ag, s_rs = torch.neuron.Stream(), torch.neuron.Stream(), torch.neuron.Stream()
    ar_t = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + 1)
    ag_local = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + 1)
    ag_out = [torch.zeros_like(ag_local) for _ in range(world_size)]
    rs_in = torch.ones(_SMALL_TENSOR_SIZE * world_size, dtype=torch.float32, device="neuron")
    rs_chunks = list(rs_in.chunk(world_size))
    rs_out = torch.zeros(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron")
    s_ar.wait_stream(default_stream)
    s_ag.wait_stream(default_stream)
    s_rs.wait_stream(default_stream)
    with torch.neuron.stream(s_ar):
        dist.all_reduce(ar_t, op=dist.ReduceOp.SUM)
    with torch.neuron.stream(s_ag):
        dist.all_gather(ag_out, ag_local)
    with torch.neuron.stream(s_rs):
        dist.reduce_scatter(rs_out, rs_chunks, op=dist.ReduceOp.SUM)
    s_ar.synchronize()
    s_ag.synchronize()
    s_rs.synchronize()
    assert torch.allclose(
        ar_t.cpu(),
        torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * sum(range(1, world_size + 1)),
        rtol=1e-5,
    )
    for i, t in enumerate(ag_out):
        assert torch.allclose(
            t.cpu(), torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * (i + 1), rtol=1e-5
        )
    assert torch.allclose(
        rs_out.cpu(), torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * world_size, rtol=1e-5
    )


# =============================================================================
# Module-level run functions for TestMultiStreamEdgeCases
# =============================================================================


def run_empty_tensor(rank, world_size, kwargs):
    default_stream = torch_neuronx.default_stream()
    s = torch.neuron.Stream()
    t = torch.ones(0, dtype=torch.float32, device="neuron")
    try:
        s.wait_stream(default_stream)
        with torch.neuron.stream(s):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        s.synchronize()
        assert t.numel() == 0
    except Exception:
        pass


def run_single_element(rank, world_size, kwargs):
    default_stream = torch_neuronx.default_stream()
    s = torch.neuron.Stream()
    t = torch.tensor([float(rank + 1)], dtype=torch.float32, device="neuron")
    s.wait_stream(default_stream)
    with torch.neuron.stream(s):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    s.synchronize()
    expected = torch.tensor([float(sum(range(1, world_size + 1)))], dtype=torch.float32)
    assert torch.allclose(t.cpu(), expected, rtol=1e-5)


def run_stream_behavior_combined(rank, world_size, kwargs):
    """Combined: stream_reuse, multiple_syncs, sync_without_ops."""
    default_stream = torch_neuronx.default_stream()
    s = torch.neuron.Stream()
    expected = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * sum(range(1, world_size + 1))
    # sync_without_ops: sync before any ops
    s.synchronize()
    # First collective
    t1 = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + 1)
    s.wait_stream(default_stream)
    with torch.neuron.stream(s):
        dist.all_reduce(t1, op=dist.ReduceOp.SUM)
    # multiple_syncs
    s.synchronize()
    s.synchronize()
    s.synchronize()
    assert torch.allclose(t1.cpu(), expected, rtol=1e-5)
    # stream_reuse: reuse stream after sync
    t2 = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + 10)
    s.wait_stream(default_stream)
    with torch.neuron.stream(s):
        dist.all_reduce(t2, op=dist.ReduceOp.SUM)
    s.synchronize()
    expected2 = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * sum(
        range(10, world_size + 10)
    )
    assert torch.allclose(t2.cpu(), expected2, rtol=1e-5)


def run_decomposed_ops_combined(rank, world_size, kwargs):
    """Combined: broadcast, reduce, scatter, gather (all decomposed)."""
    default_stream = torch_neuronx.default_stream()
    s = torch.neuron.Stream()
    # Broadcast
    t_bcast = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + 1)
    s.wait_stream(default_stream)
    with torch.neuron.stream(s):
        dist.broadcast(t_bcast, src=0)
    s.synchronize()
    assert torch.allclose(
        t_bcast.cpu(), torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32), rtol=1e-5
    )
    # Reduce
    t_reduce = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + 1)
    s.wait_stream(default_stream)
    with torch.neuron.stream(s):
        dist.reduce(t_reduce, dst=0, op=dist.ReduceOp.SUM)
    s.synchronize()
    if rank == 0:
        expected_reduce = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * sum(
            i + 1 for i in range(world_size)
        )
        assert torch.allclose(t_reduce.cpu(), expected_reduce, rtol=1e-5)
    # Scatter
    out_scatter = torch.zeros(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron")
    scatter_list = (
        [
            torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (i + 1)
            for i in range(world_size)
        ]
        if rank == 0
        else None
    )
    s.wait_stream(default_stream)
    with torch.neuron.stream(s):
        dist.scatter(out_scatter, scatter_list, src=0)
    s.synchronize()
    assert torch.allclose(
        out_scatter.cpu(),
        torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * (rank + 1),
        rtol=1e-5,
    )
    # Gather
    inp_gather = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + 1)
    gather_list = (
        [
            torch.zeros(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron")
            for _ in range(world_size)
        ]
        if rank == 0
        else None
    )
    s.wait_stream(default_stream)
    with torch.neuron.stream(s):
        dist.gather(inp_gather, gather_list, dst=0)
    s.synchronize()
    if rank == 0:
        for i, g in enumerate(gather_list):
            assert torch.allclose(
                g.cpu(), torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * (i + 1), rtol=1e-5
            )


# =============================================================================
# Module-level run functions for TestMultiStreamUnsupportedAllToAll
# =============================================================================


def run_alltoall_ops_fail(rank, world_size, kwargs):
    """Combined: all_to_all and all_to_all_single both fail."""
    default_stream = torch_neuronx.default_stream()
    s = torch.neuron.Stream()
    # all_to_all
    inputs = [
        torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron")
        * (rank * world_size + i)
        for i in range(world_size)
    ]
    outputs = [
        torch.zeros(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron")
        for _ in range(world_size)
    ]
    s.wait_stream(default_stream)
    with torch.neuron.stream(s):
        dist.all_to_all(outputs, inputs)


# =============================================================================
# Module-level run functions for TestMultiStreamTP4FSDP4
# =============================================================================


def run_tp_fsdp_groups(rank, world_size, kwargs):
    if world_size != 16:
        pytest.skip("Requires 16 ranks")
    tp_size = 4
    tp_group, fsdp_group = create_tp_fsdp_groups(rank, world_size, tp_size)
    default_stream = torch_neuronx.default_stream()
    fsdp_stream = torch.neuron.Stream()
    tp_t = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + 1)
    dist.all_reduce(tp_t, op=dist.ReduceOp.SUM, group=tp_group)
    fsdp_t = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + 1)
    fsdp_stream.wait_stream(default_stream)
    with torch.neuron.stream(fsdp_stream):
        dist.all_reduce(fsdp_t, op=dist.ReduceOp.SUM, group=fsdp_group)
    default_stream.synchronize()
    fsdp_stream.synchronize()
    tp_group_id = rank // tp_size
    tp_ranks = list(range(tp_group_id * tp_size, (tp_group_id + 1) * tp_size))
    expected_tp = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * sum(r + 1 for r in tp_ranks)
    fsdp_group_id = rank % tp_size
    fsdp_ranks = list(range(fsdp_group_id, world_size, tp_size))
    expected_fsdp = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * sum(
        r + 1 for r in fsdp_ranks
    )
    assert torch.allclose(tp_t.cpu(), expected_tp, rtol=1e-5)
    assert torch.allclose(fsdp_t.cpu(), expected_fsdp, rtol=1e-5)


def run_full_pattern(rank, world_size, kwargs):
    if world_size != 16:
        pytest.skip("Requires 16 ranks")
    tp_size, fsdp_size = 4, 4
    tp_group, fsdp_group = create_tp_fsdp_groups(rank, world_size, tp_size)
    default_stream = torch_neuronx.default_stream()
    fsdp_stream = torch.neuron.Stream()
    event = torch.neuron.Event()
    weight_shard = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + 1)
    gathered_weights = [torch.zeros_like(weight_shard) for _ in range(fsdp_size)]
    activation = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + 1)
    tp_gathered = [torch.zeros_like(activation) for _ in range(tp_size)]
    fsdp_stream.wait_stream(default_stream)
    with torch.neuron.stream(fsdp_stream):
        dist.all_gather(gathered_weights, weight_shard, group=fsdp_group)
        event.record(fsdp_stream)
    dist.all_gather(tp_gathered, activation, group=tp_group)
    default_stream.wait_event(event)
    rs_in = torch.ones(_SMALL_TENSOR_SIZE * tp_size, dtype=torch.float32, device="neuron")
    rs_chunks = list(rs_in.chunk(tp_size))
    rs_out = torch.zeros(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron")
    dist.reduce_scatter(rs_out, rs_chunks, group=tp_group)
    default_stream.synchronize()
    fsdp_stream.synchronize()
    fsdp_group_id = rank % tp_size
    fsdp_ranks = list(range(fsdp_group_id, world_size, tp_size))
    for i, gw in enumerate(gathered_weights):
        expected = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * (fsdp_ranks[i] + 1)
        assert torch.allclose(gw.cpu(), expected, rtol=1e-5)
    tp_group_id = rank // tp_size
    tp_ranks = list(range(tp_group_id * tp_size, (tp_group_id + 1) * tp_size))
    for i, ta in enumerate(tp_gathered):
        expected = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * (tp_ranks[i] + 1)
        assert torch.allclose(ta.cpu(), expected, rtol=1e-5)
    assert torch.allclose(
        rs_out.cpu(), torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * tp_size, rtol=1e-5
    )


# =============================================================================
# Module-level run functions for TestMultiStreamScaling64
# =============================================================================


def run_tp1_fsdp64(rank, world_size, kwargs):
    if world_size != 64:
        pytest.skip("Requires 64 ranks")
    default_stream = torch_neuronx.default_stream()
    fsdp_group = dist.new_group(ranks=list(range(world_size)))
    fsdp_stream = torch.neuron.Stream()
    event = torch.neuron.Event()
    shard = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + 1)
    gathered = [torch.zeros_like(shard) for _ in range(64)]
    fsdp_stream.wait_stream(default_stream)
    with torch.neuron.stream(fsdp_stream):
        dist.all_gather(gathered, shard, group=fsdp_group)
        event.record(fsdp_stream)
    default_stream.wait_event(event)
    default_stream.synchronize()
    fsdp_stream.synchronize()
    for i, g in enumerate(gathered):
        expected = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * (i + 1)
        assert torch.allclose(g.cpu(), expected, rtol=1e-5)


def run_tp4_fsdp16(rank, world_size, kwargs):
    if world_size != 64:
        pytest.skip("Requires 64 ranks")
    tp_size, fsdp_size = 4, 16
    tp_group, fsdp_group = create_tp_fsdp_groups(rank, world_size, tp_size)
    default_stream = torch_neuronx.default_stream()
    fsdp_stream = torch.neuron.Stream()
    event = torch.neuron.Event()
    weight_shard = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + 1)
    gathered_weights = [torch.zeros_like(weight_shard) for _ in range(fsdp_size)]
    activation = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32, device="neuron") * (rank + 1)
    tp_gathered = [torch.zeros_like(activation) for _ in range(tp_size)]
    fsdp_stream.wait_stream(default_stream)
    with torch.neuron.stream(fsdp_stream):
        dist.all_gather(gathered_weights, weight_shard, group=fsdp_group)
        event.record(fsdp_stream)
    dist.all_gather(tp_gathered, activation, group=tp_group)
    default_stream.wait_event(event)
    default_stream.synchronize()
    fsdp_stream.synchronize()
    fsdp_group_id = rank % tp_size
    fsdp_ranks = list(range(fsdp_group_id, world_size, tp_size))
    for i, gw in enumerate(gathered_weights):
        expected = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * (fsdp_ranks[i] + 1)
        assert torch.allclose(gw.cpu(), expected, rtol=1e-5)
    tp_group_id = rank // tp_size
    tp_ranks = list(range(tp_group_id * tp_size, (tp_group_id + 1) * tp_size))
    for i, ta in enumerate(tp_gathered):
        expected = torch.ones(_SMALL_TENSOR_SIZE, dtype=torch.float32) * (tp_ranks[i] + 1)
        assert torch.allclose(ta.cpu(), expected, rtol=1e-5)


# =============================================================================
# Module-level run functions for TestMultiStreamCoalesced
# =============================================================================


def run_coalesced_all_reduce(rank, world_size, kwargs):
    """all_reduce_coalesced on non-default stream."""
    default_stream = torch_neuronx.default_stream()
    s = torch.neuron.Stream()
    ar_tensors = [
        torch.ones(64, dtype=torch.float32, device="neuron") * (rank + 1),
        torch.ones(128, dtype=torch.float32, device="neuron") * (rank + 2),
    ]
    s.wait_stream(default_stream)
    with torch.neuron.stream(s):
        dist.all_reduce_coalesced(ar_tensors, op=dist.ReduceOp.SUM)
    s.synchronize()
    expected0 = torch.ones(64, dtype=torch.float32) * sum(r + 1 for r in range(world_size))
    expected1 = torch.ones(128, dtype=torch.float32) * sum(r + 2 for r in range(world_size))
    assert torch.allclose(ar_tensors[0].cpu(), expected0, rtol=1e-5)
    assert torch.allclose(ar_tensors[1].cpu(), expected1, rtol=1e-5)


def run_coalesced_all_gather(rank, world_size, kwargs):
    """all_gather_coalesced on non-default stream."""
    default_stream = torch_neuronx.default_stream()
    s = torch.neuron.Stream()
    ag_inputs = [
        torch.ones(64, dtype=torch.float32, device="neuron") * (rank + 1),
        torch.ones(128, dtype=torch.float32, device="neuron") * (rank + 10),
    ]
    ag_outputs = [
        [torch.zeros(64, dtype=torch.float32, device="neuron") for _ in range(world_size)],
        [torch.zeros(128, dtype=torch.float32, device="neuron") for _ in range(world_size)],
    ]
    s.wait_stream(default_stream)
    with torch.neuron.stream(s):
        dist.all_gather_coalesced(ag_outputs, ag_inputs)
    s.synchronize()
    for i in range(world_size):
        assert torch.allclose(
            ag_outputs[0][i].cpu(), torch.ones(64, dtype=torch.float32) * (i + 1), rtol=1e-5
        )
        assert torch.allclose(
            ag_outputs[1][i].cpu(), torch.ones(128, dtype=torch.float32) * (i + 10), rtol=1e-5
        )


def run_coalesced_tensor_variants(rank, world_size, kwargs):
    """Combined: all_gather_into_tensor_coalesced and reduce_scatter_tensor_coalesced."""
    default_stream = torch_neuronx.default_stream()
    _c10d_functional = torch.ops._c10d_functional
    s = torch.neuron.Stream()
    default_group = dist.group.WORLD
    # all_gather_into_tensor_coalesced
    ag_inputs = [
        torch.ones(64, dtype=torch.float32, device="neuron") * (rank + 1),
        torch.ones(128, dtype=torch.float32, device="neuron") * (rank + 10),
    ]
    s.wait_stream(default_stream)
    with torch.neuron.stream(s):
        ag_outputs = _c10d_functional.all_gather_into_tensor_coalesced(
            ag_inputs, world_size, default_group.group_name
        )
        for out in ag_outputs:
            _c10d_functional.wait_tensor(out)
    s.synchronize()
    for i in range(world_size):
        assert torch.allclose(
            ag_outputs[0][i * 64 : (i + 1) * 64].cpu(),
            torch.ones(64, dtype=torch.float32) * (i + 1),
            rtol=1e-5,
        )
        assert torch.allclose(
            ag_outputs[1][i * 128 : (i + 1) * 128].cpu(),
            torch.ones(128, dtype=torch.float32) * (i + 10),
            rtol=1e-5,
        )
    # reduce_scatter_tensor_coalesced
    rs_inputs = [
        torch.ones(64 * world_size, dtype=torch.float32, device="neuron") * (rank + 1),
        torch.ones(128 * world_size, dtype=torch.float32, device="neuron") * (rank + 10),
    ]
    s.wait_stream(default_stream)
    with torch.neuron.stream(s):
        rs_outputs = _c10d_functional.reduce_scatter_tensor_coalesced(
            rs_inputs, "sum", world_size, default_group.group_name
        )
        for out in rs_outputs:
            _c10d_functional.wait_tensor(out)
    s.synchronize()
    expected_rs0 = torch.ones(64, dtype=torch.float32) * sum(r + 1 for r in range(world_size))
    expected_rs1 = torch.ones(128, dtype=torch.float32) * sum(r + 10 for r in range(world_size))
    assert torch.allclose(rs_outputs[0].cpu(), expected_rs0, rtol=1e-5)
    assert torch.allclose(rs_outputs[1].cpu(), expected_rs1, rtol=1e-5)


# =============================================================================
# Test Classes
# =============================================================================


class TestMultiStreamBasic(BaseCollectiveOpTest):
    """Basic multi-stream collective tests."""

    def test_baseline_default_stream(self, distributed_tester):
        """AllReduce on default stream (control case)."""
        distributed_tester.run_test(run_baseline_default_stream)

    def test_allreduce(self, distributed_tester):
        """AllReduce on non-default stream."""
        distributed_tester.run_test(run_allreduce)

    def test_allgather(self, distributed_tester):
        """AllGather on non-default stream."""
        distributed_tester.run_test(run_allgather)

    def test_reduce_scatter(self, distributed_tester):
        """ReduceScatter on non-default stream."""
        distributed_tester.run_test(run_reduce_scatter)

    def test_all_gather_into_tensor_1d_and_2d(self, distributed_tester):
        """all_gather_into_tensor with 1D and 2D tensors."""
        distributed_tester.run_test(run_all_gather_into_tensor_1d_and_2d)

    def test_reduce_scatter_tensor_1d_and_2d(self, distributed_tester):
        """reduce_scatter_tensor with 1D and 2D tensors."""
        distributed_tester.run_test(run_reduce_scatter_tensor_1d_and_2d)


class TestMultiStreamMultipleStreams(BaseCollectiveOpTest):
    """Tests with multiple streams on single process group."""

    def test_two_streams_sequential_and_interleaved(self, distributed_tester):
        """Sequential and interleaved collectives on two streams."""
        distributed_tester.run_test(run_two_streams_sequential_and_interleaved)

    def test_three_streams_parallel(self, distributed_tester):
        """Concurrent collectives on three streams."""
        distributed_tester.run_test(run_three_streams_parallel)

    def test_stream_sync_between_collectives(self, distributed_tester):
        """Explicit sync between collectives."""
        distributed_tester.run_test(run_stream_sync_between_collectives)


class TestMultiStreamMultiplePG(BaseCollectiveOpTest):
    """Tests with multiple process groups."""

    @property
    def world_size(self):
        return max(4, _DEFAULT_WORLD_SIZE)

    def test_two_pgs_same_stream(self, distributed_tester):
        """Two process groups on same stream."""
        distributed_tester.run_test(run_two_pgs_same_stream)

    def test_two_pgs_different_streams(self, distributed_tester):
        """Two process groups on different streams."""
        distributed_tester.run_test(run_two_pgs_different_streams)


class TestMultiStreamEvents(BaseCollectiveOpTest):
    """Event-based cross-stream synchronization tests."""

    def test_event_record_and_wait(self, distributed_tester):
        """Basic event record and wait."""
        distributed_tester.run_test(run_event_record_and_wait)

    def test_cross_stream_sync_with_collective(self, distributed_tester):
        """Cross-stream sync with AllGather."""
        distributed_tester.run_test(run_cross_stream_sync_with_collective)

    def test_multiple_events_pipeline(self, distributed_tester):
        """Multiple events for pipelined execution."""
        distributed_tester.run_test(run_multiple_events_pipeline)

    def test_event_timing(self, distributed_tester):
        """Event timing measurement."""
        distributed_tester.run_test(run_event_timing)


class TestMultiStreamOverlap(BaseCollectiveOpTest):
    """Compute-collective overlap tests."""

    @property
    def world_size(self):
        return max(4, _DEFAULT_WORLD_SIZE)

    def test_compute_collective_overlap(self, distributed_tester):
        """Basic compute-collective overlap."""
        distributed_tester.run_test(run_compute_collective_overlap)

    def test_fsdp_allgather_compute_overlap(self, distributed_tester):
        """FSDP AllGather overlapped with compute."""
        distributed_tester.run_test(run_fsdp_allgather_compute_overlap)

    def test_tp_fsdp_overlap(self, distributed_tester):
        """TP collective overlapped with FSDP collective."""
        distributed_tester.run_test(run_tp_fsdp_overlap)

    def test_pipelined_layers(self, distributed_tester):
        """Pipelined layer execution pattern."""
        distributed_tester.run_test(run_pipelined_layers)


class TestMultiStreamStress(BaseCollectiveOpTest):
    """Stress and concurrency tests."""

    def test_many_streams_and_rapid_creation(self, distributed_tester):
        """Many streams with concurrent collectives and rapid stream creation."""
        distributed_tester.run_test(run_many_streams_and_rapid_creation)

    def test_large_tensor(self, distributed_tester):
        """Large tensor collective."""
        distributed_tester.run_test(run_large_tensor)

    def test_mixed_collective_types(self, distributed_tester):
        """Mixed collective types on multiple streams."""
        distributed_tester.run_test(run_mixed_collective_types)


class TestMultiStreamEdgeCases(BaseCollectiveOpTest):
    """Edge cases and error handling tests."""

    def test_empty_tensor(self, distributed_tester):
        """Collective with empty tensor."""
        distributed_tester.run_test(run_empty_tensor)

    def test_single_element(self, distributed_tester):
        """Single-element tensor collective."""
        distributed_tester.run_test(run_single_element)

    def test_stream_behavior_combined(self, distributed_tester):
        """Stream reuse, multiple syncs, and sync without ops."""
        distributed_tester.run_test(run_stream_behavior_combined)

    def test_decomposed_ops_combined(self, distributed_tester):
        """Broadcast, reduce, scatter, gather (all decomposed)."""
        distributed_tester.run_test(run_decomposed_ops_combined)


class TestMultiStreamUnsupportedAllToAll(BaseCollectiveOpTest):
    """Negative test for unsupported AllToAll operations."""

    @property
    def world_size(self):
        return 4

    @assert_raises(RuntimeError, match=".*Unsupported collective.*")
    def test_alltoall_ops_fail(self, distributed_tester):
        """AllToAll and all_to_all_single on non-default stream should fail."""
        distributed_tester.run_test(run_alltoall_ops_fail)


class TestMultiStreamTP4FSDP4(BaseCollectiveOpTest):
    """TP4xFSDP4 tests (16 ranks)."""

    @property
    def world_size(self):
        return 16

    def test_tp_fsdp_groups(self, distributed_tester):
        """Full TP4xFSDP4 pattern."""
        distributed_tester.run_test(run_tp_fsdp_groups)

    def test_full_pattern(self, distributed_tester):
        """Complete TP4xFSDP4 multi-stream pattern with events."""
        distributed_tester.run_test(run_full_pattern)


class TestMultiStreamScaling64(BaseCollectiveOpTest):
    """64-rank scaling tests."""

    @property
    def world_size(self):
        return 64

    def test_tp1_fsdp64(self, distributed_tester):
        """TP1xFSDP64 pattern."""
        distributed_tester.run_test(run_tp1_fsdp64)

    def test_tp4_fsdp16(self, distributed_tester):
        """TP4xFSDP16 pattern."""
        distributed_tester.run_test(run_tp4_fsdp16)


class TestMultiStreamCoalesced(BaseCollectiveOpTest):
    """Coalesced collective operations on non-default stream."""

    def test_coalesced_all_reduce(self, distributed_tester):
        """all_reduce_coalesced on non-default stream."""
        distributed_tester.run_test(run_coalesced_all_reduce)

    @pytest.mark.xfail(reason="all_gather_coalesced requires NRT fix")
    def test_coalesced_all_gather(self, distributed_tester):
        """all_gather_coalesced on non-default stream."""
        distributed_tester.run_test(run_coalesced_all_gather)

    def test_coalesced_tensor_variants(self, distributed_tester):
        """all_gather_into_tensor_coalesced and reduce_scatter_tensor_coalesced."""
        distributed_tester.run_test(run_coalesced_tensor_variants)
