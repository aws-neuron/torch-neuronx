"""
E2E tests for graph break handling in neuron dynamo backend.

Tests that torch.compile(backend="neuron", dynamic=False) works correctly
when graph breaks occur from untraceable ops like dist.all_gather_object.
"""

import os

import pytest
import torch
import torch.distributed as dist

from tests.distributed.utils import DistributedTester

_WORLD_SIZE = 2


def _test_all_gather_object_graph_break(rank, world_size, func_args):
    """Test that all_gather_object inside torch.compile works with graph breaks."""

    def fn(x, group):
        w = dist.get_world_size(group)
        sizes = [None] * w
        dist.all_gather_object(sizes, x.shape, group)
        parts = [torch.empty(sizes[i], dtype=x.dtype, device=x.device) for i in range(w)]
        dist.all_gather(parts, x.contiguous(), group=group)
        return torch.cat(parts, dim=0).sum()

    x = torch.ones(4, 8, device=f"neuron:{rank}", dtype=torch.bfloat16)

    eager_out = fn(x, dist.group.WORLD)
    compiled_fn = torch.compile(fn, backend="neuron", dynamic=False)
    compiled_out = compiled_fn(x, dist.group.WORLD)

    torch.testing.assert_close(compiled_out.cpu(), eager_out.cpu())


def _test_graph_break_with_python_control_flow(rank, world_size, func_args):
    """Test graph break from Python control flow between compiled ops."""

    def fn(x, group):
        w = dist.get_world_size(group)
        meta = [None] * w
        dist.all_gather_object(meta, {"rank": rank, "shape": x.shape}, group)
        total = sum(m["shape"][0] for m in meta)
        return x.sum() * total

    x = torch.ones(4, 8, device=f"neuron:{rank}", dtype=torch.bfloat16)

    eager_out = fn(x, dist.group.WORLD)
    compiled_fn = torch.compile(fn, backend="neuron", dynamic=False)
    compiled_out = compiled_fn(x, dist.group.WORLD)

    torch.testing.assert_close(compiled_out.cpu(), eager_out.cpu())


def _test_trivial_graph_passthrough(rank, world_size, func_args):
    """Test that a function producing only trivial graphs compiles."""

    def fn(x, group):
        sizes = [None] * dist.get_world_size(group)
        dist.all_gather_object(sizes, x.shape, group)
        return x

    x = torch.ones(4, 8, device=f"neuron:{rank}", dtype=torch.bfloat16)

    compiled_fn = torch.compile(fn, backend="neuron", dynamic=False)
    result = compiled_fn(x, dist.group.WORLD)

    torch.testing.assert_close(result.cpu(), x.cpu())


class TestGraphBreakE2E:
    """E2E tests for graph break handling on Neuron."""

    @pytest.fixture(autouse=True)
    def process_index(self, worker_id):
        process_index = int(worker_id[2:]) if worker_id != "master" else 0
        return process_index

    @pytest.fixture(autouse=True)
    def distributed_tester(self, process_index):
        self.distributed_tester = DistributedTester(
            world_size=_WORLD_SIZE, process_index=process_index
        )
        return self.distributed_tester

    def test_all_gather_object_graph_break(self):
        self.distributed_tester.run_test(_test_all_gather_object_graph_break)

    def test_graph_break_with_python_control_flow(self):
        self.distributed_tester.run_test(_test_graph_break_with_python_control_flow)

    def test_trivial_graph_passthrough(self):
        self.distributed_tester.run_test(_test_trivial_graph_passthrough)
