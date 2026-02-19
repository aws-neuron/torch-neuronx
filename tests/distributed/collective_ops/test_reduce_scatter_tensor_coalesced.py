import os

import pytest
import torch
import torch.distributed as dist

from tests.utils.neuron_test_utils import assert_raises

from .base_collective_op import BaseCollectiveOpTest

# Store reference to functional collective ops at module level
_c10d_functional = torch.ops._c10d_functional


def run_basic_reduce_scatter_coalesced_test(rank, world_size, kwargs):
    """Test basic coalesced reduce-scatter with multiple tensors."""
    default_group = torch.distributed.group.WORLD

    # Same shape [*, 3] - only dim 0 differs
    input1 = torch.ones(world_size * 2, 3) * (rank + 1)
    input2 = torch.ones(world_size * 4, 3) * (rank + 2)

    inputs = [input1.to("neuron"), input2.to("neuron")]

    outputs = _c10d_functional.reduce_scatter_tensor_coalesced(
        inputs, "sum", world_size, default_group.group_name
    )

    # Per pytorch documentation + warning: need to call wait_tensor
    for out in outputs:
        _c10d_functional.wait_tensor(out)

    sum_of_ranks = sum(range(1, world_size + 1))

    expected1 = torch.ones(2, 3) * sum_of_ranks
    expected2 = torch.ones(4, 3) * (sum_of_ranks + world_size)

    assert torch.allclose(outputs[0].cpu(), expected1), f"Rank {rank}: Output 0 mismatch"
    assert torch.allclose(outputs[1].cpu(), expected2), f"Rank {rank}: Output 1 mismatch"


def run_different_reduce_ops_test(rank, world_size, kwargs):
    """Test coalesced reduce-scatter with different reduction operations."""
    default_group = torch.distributed.group.WORLD
    reduce_op = kwargs["reduce_op"]
    reduce_op_str = kwargs["reduce_op_str"]

    # Same shape [*, 4]
    input1 = torch.ones(world_size * 2, 4) * (rank + 1)
    input2 = torch.ones(world_size * 3, 4) * (rank + 2)

    inputs = [input1.to("neuron"), input2.to("neuron")]

    outputs = _c10d_functional.reduce_scatter_tensor_coalesced(
        inputs, reduce_op_str, world_size, default_group.group_name
    )

    for out in outputs:
        _c10d_functional.wait_tensor(out)

    # Calculate expected values based on reduce_op
    if reduce_op == dist.ReduceOp.SUM:
        sum_of_ranks = sum(range(1, world_size + 1))
        expected1 = torch.ones(2, 4) * sum_of_ranks
        expected2 = torch.ones(3, 4) * (sum_of_ranks + world_size)
    elif reduce_op == dist.ReduceOp.PRODUCT:
        import math

        product_of_ranks = math.factorial(world_size)
        expected1 = torch.ones(2, 4) * product_of_ranks
        expected2 = torch.ones(3, 4) * (product_of_ranks * math.factorial(world_size))
    elif reduce_op == dist.ReduceOp.MIN:
        expected1 = torch.ones(2, 4) * 1
        expected2 = torch.ones(3, 4) * 2
    elif reduce_op == dist.ReduceOp.MAX:
        expected1 = torch.ones(2, 4) * world_size
        expected2 = torch.ones(3, 4) * (world_size + 1)
    elif reduce_op == dist.ReduceOp.AVG:
        avg_of_ranks = sum(range(1, world_size + 1)) / world_size
        expected1 = torch.ones(2, 4) * avg_of_ranks
        expected2 = torch.ones(3, 4) * (avg_of_ranks + 1)

    assert torch.allclose(outputs[0].cpu(), expected1, atol=1e-5)
    assert torch.allclose(outputs[1].cpu(), expected2, atol=1e-5)


def run_different_dtypes_reduce_scatter_test(rank, world_size, kwargs):
    """Test coalesced reduce-scatter with specific dtype."""
    default_group = torch.distributed.group.WORLD
    dtype = kwargs["dtype"]

    # Same shape [*, 4]
    input1 = torch.ones(world_size * 3, 4, dtype=dtype) * (rank + 1)
    input2 = torch.ones(world_size * 2, 4, dtype=dtype) * (rank + 2)

    inputs = [input1.to("neuron"), input2.to("neuron")]

    outputs = _c10d_functional.reduce_scatter_tensor_coalesced(
        inputs, "sum", world_size, default_group.group_name
    )

    for out in outputs:
        _c10d_functional.wait_tensor(out)

    assert outputs[0].dtype == dtype
    assert outputs[1].dtype == dtype

    sum_of_ranks = sum(range(1, world_size + 1))
    expected1 = torch.ones(3, 4, dtype=dtype) * sum_of_ranks
    expected2 = torch.ones(2, 4, dtype=dtype) * (sum_of_ranks + world_size)

    assert torch.allclose(outputs[0].cpu(), expected1)
    assert torch.allclose(outputs[1].cpu(), expected2)


def run_same_shape_multiple_tensors_test(rank, world_size, kwargs):
    """Test coalesced reduce-scatter with multiple tensors of same shape."""
    default_group = torch.distributed.group.WORLD
    shape = kwargs["shape"]
    num_tensors = kwargs["num_tensors"]

    # All tensors have the same shape
    inputs = [
        torch.ones(world_size * shape[0], *shape[1:]).to("neuron") * (rank + idx + 1)
        for idx in range(num_tensors)
    ]

    outputs = _c10d_functional.reduce_scatter_tensor_coalesced(
        inputs, "sum", world_size, default_group.group_name
    )

    for out in outputs:
        _c10d_functional.wait_tensor(out)

    sum_of_ranks = sum(range(1, world_size + 1))
    for idx in range(num_tensors):
        expected = torch.ones(shape) * (sum_of_ranks + idx * world_size)
        assert torch.allclose(outputs[idx].cpu(), expected), f"Tensor {idx} mismatch"


def run_empty_tensor_mixed_reduce_scatter_test(rank, world_size, kwargs):
    """Test coalesced reduce-scatter with mix of empty and non-empty tensors."""
    default_group = torch.distributed.group.WORLD

    # Same shape [*, 3] for non-empty tensors
    input1 = torch.ones(world_size * 2, 3) * (rank + 1)
    input2 = torch.empty(0, 3)
    input3 = torch.ones(world_size * 4, 3) * (rank + 3)

    inputs = [input1.to("neuron"), input2.to("neuron"), input3.to("neuron")]

    outputs = _c10d_functional.reduce_scatter_tensor_coalesced(
        inputs, "sum", world_size, default_group.group_name
    )

    for out in outputs:
        _c10d_functional.wait_tensor(out)

    sum_of_ranks = sum(range(1, world_size + 1))
    expected1 = torch.ones(2, 3) * sum_of_ranks
    expected3 = torch.ones(4, 3) * (sum_of_ranks + 2 * world_size)

    assert torch.allclose(outputs[0].cpu(), expected1), f"Output 0 mismatch on rank {rank}"
    assert outputs[1].numel() == 0
    assert torch.allclose(outputs[2].cpu(), expected3), f"Output 2 mismatch on rank {rank}"


def run_large_tensors_reduce_scatter_test(rank, world_size, kwargs):
    """Test coalesced reduce-scatter with large tensors."""
    default_group = torch.distributed.group.WORLD

    # Same shape [*, 100]
    input1 = torch.ones(world_size * 1000, 100) * (rank + 1)
    input2 = torch.ones(world_size * 500, 100) * (rank + 2)

    inputs = [input1.to("neuron"), input2.to("neuron")]

    outputs = _c10d_functional.reduce_scatter_tensor_coalesced(
        inputs, "sum", world_size, default_group.group_name
    )

    for out in outputs:
        _c10d_functional.wait_tensor(out)

    assert outputs[0].shape == (1000, 100)
    assert outputs[1].shape == (500, 100)

    sum_of_ranks = sum(range(1, world_size + 1))
    assert torch.allclose(outputs[0][0, 0].cpu(), torch.tensor(float(sum_of_ranks)))
    assert torch.allclose(outputs[1][0, 0].cpu(), torch.tensor(float(sum_of_ranks + world_size)))


def run_three_tensors_different_sizes_test(rank, world_size, kwargs):
    """Test coalesced reduce-scatter with three tensors (same shape except dim 0)."""
    default_group = torch.distributed.group.WORLD

    # Same shape [*, 5]
    input1 = torch.ones(world_size * 2, 5) * (rank + 1)
    input2 = torch.ones(world_size * 4, 5) * (rank + 2)
    input3 = torch.ones(world_size * 6, 5) * (rank + 3)

    inputs = [input1.to("neuron"), input2.to("neuron"), input3.to("neuron")]

    outputs = _c10d_functional.reduce_scatter_tensor_coalesced(
        inputs, "sum", world_size, default_group.group_name
    )

    for out in outputs:
        _c10d_functional.wait_tensor(out)

    sum_of_ranks = sum(range(1, world_size + 1))
    expected1 = torch.ones(2, 5) * sum_of_ranks
    expected2 = torch.ones(4, 5) * (sum_of_ranks + world_size)
    expected3 = torch.ones(6, 5) * (sum_of_ranks + 2 * world_size)

    assert torch.allclose(outputs[0].cpu(), expected1)
    assert torch.allclose(outputs[1].cpu(), expected2)
    assert torch.allclose(outputs[2].cpu(), expected3)


def run_single_tensor_test(rank, world_size, kwargs):
    """Test single tensor functional API."""
    default_group = torch.distributed.group.WORLD

    input1 = torch.ones(world_size * 5, 8) * (rank + 1)
    input_neuron = input1.to("neuron")

    output = _c10d_functional.reduce_scatter_tensor(
        input_neuron, "sum", world_size, default_group.group_name
    )

    _c10d_functional.wait_tensor(output)

    sum_of_ranks = sum(range(1, world_size + 1))
    expected = torch.ones(5, 8) * sum_of_ranks

    assert torch.allclose(output.cpu(), expected)


def run_multi_dimensional_test(rank, world_size, kwargs):
    """Test coalesced reduce-scatter with multi-dimensional tensors."""
    default_group = torch.distributed.group.WORLD

    # Same shape [*, 3, 4, 5]
    input1 = torch.ones(world_size * 2, 3, 4, 5) * (rank + 1)
    input2 = torch.ones(world_size * 4, 3, 4, 5) * (rank + 2)

    inputs = [input1.to("neuron"), input2.to("neuron")]

    outputs = _c10d_functional.reduce_scatter_tensor_coalesced(
        inputs, "sum", world_size, default_group.group_name
    )

    for out in outputs:
        _c10d_functional.wait_tensor(out)

    sum_of_ranks = sum(range(1, world_size + 1))
    expected1 = torch.ones(2, 3, 4, 5) * sum_of_ranks
    expected2 = torch.ones(4, 3, 4, 5) * (sum_of_ranks + world_size)

    assert torch.allclose(outputs[0].cpu(), expected1)
    assert torch.allclose(outputs[1].cpu(), expected2)


def run_single_tensor_coalesced_test(rank, world_size, kwargs):
    """Test coalesced reduce-scatter with a single tensor (special case optimization)."""
    default_group = torch.distributed.group.WORLD

    input_tensor = torch.ones(world_size * 4, 5) * (rank + 1)
    inputs = [input_tensor.to("neuron")]

    outputs = _c10d_functional.reduce_scatter_tensor_coalesced(
        inputs, "sum", world_size, default_group.group_name
    )

    for out in outputs:
        _c10d_functional.wait_tensor(out)

    sum_of_ranks = sum(range(1, world_size + 1))
    expected = torch.ones(4, 5) * sum_of_ranks

    assert len(outputs) == 1, f"Expected 1 output tensor, got {len(outputs)}"
    assert torch.allclose(outputs[0].cpu(), expected), f"Rank {rank}: Single tensor output mismatch"


def run_group_reduce_scatter_coalesced_test(rank, world_size, kwargs):
    """Test coalesced reduce-scatter with specific process group."""
    group = dist.new_group([0, 1])
    group_size = 2
    group_name = group.group_name

    input1 = torch.ones(group_size * 2, 3) * (rank + 1)
    input2 = torch.ones(group_size * 3, 3) * (rank + 2)

    inputs = [input1.to("neuron"), input2.to("neuron")]

    if rank <= 1:
        outputs = _c10d_functional.reduce_scatter_tensor_coalesced(
            inputs, "sum", group_size, group_name
        )

        for out in outputs:
            _c10d_functional.wait_tensor(out)

        expected1 = torch.ones(2, 3) * 3  # 1 + 2
        expected2 = torch.ones(3, 3) * 5  # 2 + 3

        assert torch.allclose(outputs[0].cpu(), expected1)
        assert torch.allclose(outputs[1].cpu(), expected2)


def run_5gb_tensor_test(rank, world_size, kwargs):
    """Test reduce_scatter_tensor_coalesced with 5GB tensor (~1.25 billion float32 elements)."""
    tensor_size = 1250000000  # ~5GB for float32
    default_group = torch.distributed.group.WORLD

    # Same shape [*, 3] - only dim 0 differs
    input1 = torch.ones(world_size * tensor_size, 3, dtype=torch.float32) * (rank + 1)
    input2 = torch.ones(world_size * tensor_size, 3, dtype=torch.float32) * (rank + 2)

    inputs = [input1.to("neuron"), input2.to("neuron")]

    outputs = _c10d_functional.reduce_scatter_tensor_coalesced(
        inputs, "sum", world_size, default_group.group_name
    )

    for out in outputs:
        _c10d_functional.wait_tensor(out)

    sum_of_ranks = sum(range(1, world_size + 1))

    expected1 = torch.ones(tensor_size, 3, dtype=torch.float32) * sum_of_ranks
    expected2 = torch.ones(tensor_size, 3, dtype=torch.float32) * (sum_of_ranks + world_size)

    assert torch.allclose(outputs[0].cpu(), expected1), f"Rank {rank}: Output 0 mismatch"
    assert torch.allclose(outputs[1].cpu(), expected2), f"Rank {rank}: Output 1 mismatch"


def run_shape_mismatch_error(rank, world_size, kwargs):
    """Test error handling when input tensors have different shapes."""
    default_group = torch.distributed.group.WORLD

    # Different shapes (dim 1 differs: 3 vs 5)
    input1 = torch.ones(world_size * 2, 3).to("neuron")
    input2 = torch.ones(world_size * 4, 5).to("neuron")

    inputs = [input1, input2]

    _c10d_functional.reduce_scatter_tensor_coalesced(
        inputs, "sum", world_size, default_group.group_name
    )


def run_bucketing_memory_test(rank, world_size, kwargs):
    """Test reduce_scatter_tensor_coalesced with bucketing to ensure memory stays under 1GB."""
    core_id = os.environ.get("NEURON_RT_VISIBLE_CORES", "0")
    with open(f"./neuron_core_id_{rank}.txt", "w") as f:
        f.write(str(core_id))
    dev_id = int(core_id) // 8
    core_id = int(core_id) - (dev_id * 8)
    os.system(
        f"echo 0 | sudo tee /sys/devices/virtual/neuron_device/neuron{dev_id}/"
        f"neuron_core{core_id}/stats/memory_usage/device_mem/"
        "model_shared_scratchpad/peak"
    )

    default_group = torch.distributed.group.WORLD

    for i in range(2):
        tensor_size = (134217736 + i * 134217736) // world_size // 2

        # Two tensors with same shape except dim 0
        input1 = torch.ones(tensor_size, 3, dtype=torch.float32) * (rank + 1)
        input2 = torch.ones(tensor_size, 3, dtype=torch.float32) * (rank + 2)

        inputs = [input1.to("neuron"), input2.to("neuron")]
        outputs = _c10d_functional.reduce_scatter_tensor_coalesced(
            inputs, "sum", world_size, default_group.group_name
        )

        for out in outputs:
            _c10d_functional.wait_tensor(out)

        sum_of_ranks = sum(range(1, world_size + 1))
        expected1 = torch.ones(tensor_size // world_size, 3, dtype=torch.float32) * sum_of_ranks
        expected2 = torch.ones(tensor_size // world_size, 3, dtype=torch.float32) * (
            sum_of_ranks + world_size
        )

        assert torch.allclose(outputs[0].cpu(), expected1), "Output 0 does not match expected1"
        assert torch.allclose(outputs[1].cpu(), expected2), "Output 1 does not match expected2"

        peak = int(
            os.popen(
                f"cat /sys/devices/virtual/neuron_device/neuron{dev_id}/"
                f"neuron_core{core_id}/stats/memory_usage/device_mem/"
                "model_shared_scratchpad/peak"
            )
            .read()
            .strip()
        )
        assert peak <= 1073741824, f"Memory {peak} exceeds 1GB"


class TestReduceScatterTensorCoalesced(BaseCollectiveOpTest):
    """Test cases for functional reduce_scatter_tensor_coalesced."""

    def test_basic_reduce_scatter_coalesced(self):
        """Test basic coalesced reduce-scatter with multiple tensors."""
        self.distributed_tester.run_test(run_basic_reduce_scatter_coalesced_test)

    @pytest.mark.parametrize(
        "reduce_op,reduce_op_str",
        [
            (dist.ReduceOp.SUM, "sum"),
            pytest.param(
                dist.ReduceOp.PRODUCT,
                "product",
                marks=pytest.mark.skip(reason="PRODUCT not supported by XLA reduce_scatter"),
            ),
            (dist.ReduceOp.MIN, "min"),
            (dist.ReduceOp.MAX, "max"),
            (dist.ReduceOp.AVG, "avg"),
        ],
    )
    def test_different_reduce_ops(self, reduce_op, reduce_op_str):
        """Test coalesced reduce-scatter with different reduction operations."""
        self.distributed_tester.run_test(
            run_different_reduce_ops_test, reduce_op=reduce_op, reduce_op_str=reduce_op_str
        )

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float16,
            torch.int32,
            torch.int64,
            torch.bfloat16,
        ],
    )
    def test_different_dtypes_reduce_scatter(self, dtype):
        """Test coalesced reduce-scatter with different data types."""
        self.distributed_tester.run_test(run_different_dtypes_reduce_scatter_test, dtype=dtype)

    @pytest.mark.parametrize(
        "shape,num_tensors",
        [
            ((2, 3), 2),
            ((10, 5), 3),
            ((5, 5, 5), 2),
        ],
    )
    def test_same_shape_multiple_tensors(self, shape, num_tensors):
        """Test coalesced reduce-scatter with multiple tensors of same shape."""
        self.distributed_tester.run_test(
            run_same_shape_multiple_tensors_test, shape=shape, num_tensors=num_tensors
        )

    def test_empty_tensor_mixed_reduce_scatter(self):
        """Test coalesced reduce-scatter with mix of empty and non-empty tensors."""
        self.distributed_tester.run_test(run_empty_tensor_mixed_reduce_scatter_test)

    def test_large_tensors_reduce_scatter(self):
        """Test coalesced reduce-scatter with large tensors."""
        self.distributed_tester.run_test(run_large_tensors_reduce_scatter_test)

    def test_three_tensors_different_sizes(self):
        """Test coalesced reduce-scatter with three tensors."""
        self.distributed_tester.run_test(run_three_tensors_different_sizes_test)

    def test_single_tensor(self):
        """Test single tensor functional API."""
        self.distributed_tester.run_test(run_single_tensor_test)

    def test_multi_dimensional(self):
        """Test coalesced reduce-scatter with multi-dimensional tensors."""
        self.distributed_tester.run_test(run_multi_dimensional_test)

    def test_single_tensor_coalesced(self):
        """Test coalesced reduce-scatter with a single tensor."""
        self.distributed_tester.run_test(run_single_tensor_coalesced_test)

    def test_group_reduce_scatter_coalesced(self):
        """Test coalesced reduce-scatter with specific process groups."""
        self.distributed_tester.run_test(run_group_reduce_scatter_coalesced_test)

    @assert_raises(
        (RuntimeError, ValueError),
        match=r".*(requires all input tensors to have the same shape|shape).*",
    )
    def test_shape_mismatch_error(self):
        """Test error handling when input tensors have mismatched shapes."""
        self.distributed_tester.run_test(run_shape_mismatch_error)

    @pytest.mark.xfail(reason="5GB tensor test expected to fail")
    def test_5gb_tensor(self):
        """Test reduce_scatter_tensor_coalesced with 5GB tensor."""
        self.distributed_tester.run_test(run_5gb_tensor_test)

    @pytest.mark.xfail(reason="Memory assertions require further investigation, thus xfailing.")
    def test_bucketing_memory(self):
        """Test reduce_scatter_tensor_coalesced with bucketing for memory efficiency."""
        self.distributed_tester.run_test(run_bucketing_memory_test)
