import os

import pytest
import torch
import torch.distributed as dist

from .base_collective_op import BaseCollectiveOpTest

# Store reference to functional collective ops at module level
_c10d_functional = torch.ops._c10d_functional


def run_basic_coalesced_test(rank, world_size, kwargs):
    """Test basic coalesced all-gather with multiple tensors."""
    # Create multiple input tensors with different sizes
    input1 = torch.ones(2, 3) * (rank + 1)
    input2 = torch.ones(4, 5) * (rank + 2)

    inputs = [input1.to("neuron"), input2.to("neuron")]

    default_group = torch.distributed.group.WORLD

    # Use functional collective API
    outputs = _c10d_functional.all_gather_into_tensor_coalesced(
        inputs, world_size, default_group.group_name
    )

    # Per pytorch documentation + warning: need to call wait_tensor
    for out in outputs:
        _c10d_functional.wait_tensor(out)

    # Verify results
    # For input1: each rank contributes [2, 3] tensor, total should be [world_size*2, 3]
    expected1 = torch.cat([torch.ones(2, 3) * (i + 1) for i in range(world_size)], dim=0)
    assert torch.allclose(outputs[0].cpu(), expected1), f"Rank {rank}: Output 0 mismatch"

    # For input2: each rank contributes [4, 5] tensor, total should be [world_size*4, 5]
    expected2 = torch.cat([torch.ones(4, 5) * (i + 2) for i in range(world_size)], dim=0)
    assert torch.allclose(outputs[1].cpu(), expected2), f"Rank {rank}: Output 1 mismatch"


def run_different_dtypes_coalesced_test(rank, world_size, kwargs):
    """Test coalesced all-gather with specific dtype."""
    dtype = kwargs["dtype"]

    input1 = torch.ones(3, 4, dtype=dtype) * (rank + 1)
    input2 = torch.ones(2, 5, dtype=dtype) * (rank + 2)

    inputs = [input1.to("neuron"), input2.to("neuron")]

    default_group = torch.distributed.group.WORLD

    outputs = _c10d_functional.all_gather_into_tensor_coalesced(
        inputs, world_size, default_group.group_name
    )

    for out in outputs:
        _c10d_functional.wait_tensor(out)

    # Verify dtype is preserved
    assert outputs[0].dtype == dtype
    assert outputs[1].dtype == dtype

    # Verify values
    expected1 = torch.cat(
        [torch.ones(3, 4, dtype=dtype) * (i + 1) for i in range(world_size)], dim=0
    )
    expected2 = torch.cat(
        [torch.ones(2, 5, dtype=dtype) * (i + 2) for i in range(world_size)], dim=0
    )

    assert torch.allclose(outputs[0].cpu(), expected1)
    assert torch.allclose(outputs[1].cpu(), expected2)


def run_mixed_shapes_test(rank, world_size, kwargs):
    """Test coalesced all-gather with tensors of different shapes."""
    shapes = kwargs["shapes"]

    inputs = [torch.ones(shape).to("neuron") * (rank + idx + 1) for idx, shape in enumerate(shapes)]

    default_group = torch.distributed.group.WORLD

    outputs = _c10d_functional.all_gather_into_tensor_coalesced(
        inputs, world_size, default_group.group_name
    )

    for out in outputs:
        _c10d_functional.wait_tensor(out)

    # Verify each output
    for idx, shape in enumerate(shapes):
        expected = torch.cat([torch.ones(shape) * (i + idx + 1) for i in range(world_size)], dim=0)
        assert torch.allclose(outputs[idx].cpu(), expected), f"Shape {shape} mismatch"


def run_empty_tensor_mixed_test(rank, world_size, kwargs):
    """Test coalesced all-gather with mix of empty and non-empty tensors."""
    # Mix empty and non-empty tensors
    input1 = torch.ones(2, 3) * (rank + 1)
    input2 = torch.empty(0, 3)  # Empty tensor
    input3 = torch.ones(4, 5) * (rank + 3)

    inputs = [input1.to("neuron"), input2.to("neuron"), input3.to("neuron")]

    default_group = torch.distributed.group.WORLD

    outputs = _c10d_functional.all_gather_into_tensor_coalesced(
        inputs, world_size, default_group.group_name
    )

    for out in outputs:
        _c10d_functional.wait_tensor(out)

    # Verify non-empty tensors
    expected1 = torch.cat([torch.ones(2, 3) * (i + 1) for i in range(world_size)], dim=0)
    expected3 = torch.cat([torch.ones(4, 5) * (i + 3) for i in range(world_size)], dim=0)

    assert torch.allclose(outputs[0].cpu(), expected1)
    assert outputs[1].numel() == 0  # Empty output should remain empty
    assert torch.allclose(outputs[2].cpu(), expected3)


def run_large_tensors_coalesced_test(rank, world_size, kwargs):
    """Test coalesced all-gather with large tensors."""
    # Create large tensors
    input1 = torch.ones(1000, 100) * (rank + 1)
    input2 = torch.ones(500, 200) * (rank + 2)

    inputs = [input1.to("neuron"), input2.to("neuron")]

    default_group = torch.distributed.group.WORLD

    outputs = _c10d_functional.all_gather_into_tensor_coalesced(
        inputs, world_size, default_group.group_name
    )

    for out in outputs:
        _c10d_functional.wait_tensor(out)

    # Verify shapes
    assert outputs[0].shape == (world_size * 1000, 100)
    assert outputs[1].shape == (world_size * 500, 200)

    # Check this rank's chunk
    start_idx_1 = rank * 1000
    end_idx_1 = (rank + 1) * 1000
    assert torch.allclose(outputs[0][start_idx_1:end_idx_1].cpu(), input1.cpu())

    start_idx_2 = rank * 500
    end_idx_2 = (rank + 1) * 500
    assert torch.allclose(outputs[1][start_idx_2:end_idx_2].cpu(), input2.cpu())


def run_with_special_values_test(rank, world_size, kwargs):
    """Test coalesced all-gather with inf, -inf, and nan values."""
    input1 = torch.tensor([[float("inf"), -float("inf")], [1.0, 2.0]]).to("neuron")
    input2 = torch.tensor([[float("nan"), 3.0], [4.0, 5.0]]).to("neuron")

    inputs = [input1, input2]

    default_group = torch.distributed.group.WORLD

    outputs = _c10d_functional.all_gather_into_tensor_coalesced(
        inputs, world_size, default_group.group_name
    )

    for out in outputs:
        _c10d_functional.wait_tensor(out)

    # Verify inf values are preserved
    assert torch.isinf(outputs[0][0, 0])
    assert torch.isinf(outputs[0][0, 1])

    # Verify nan values are preserved
    assert torch.isnan(outputs[1][0, 0])


def run_single_tensor_test(rank, world_size, kwargs):
    """Test functional API with single tensor."""
    input_tensor = torch.ones(3, 4) * (rank + 1)
    input_neuron = input_tensor.to("neuron")

    default_group = torch.distributed.group.WORLD

    # Use single tensor functional API
    output = _c10d_functional.all_gather_into_tensor(
        input_neuron, world_size, default_group.group_name
    )

    _c10d_functional.wait_tensor(output)

    # Verify result
    expected = torch.cat([torch.ones(3, 4) * (i + 1) for i in range(world_size)], dim=0)
    assert torch.allclose(output.cpu(), expected)


def run_group_coalesced_test(rank, world_size, kwargs):
    """Test coalesced all-gather with specific process group."""
    # Create a subgroup
    group = dist.new_group([0, 1])
    group_size = 2

    # Get group name directly from the group object
    group_name = group.group_name

    input1 = torch.ones(2, 3) * (rank + 1)
    input2 = torch.ones(3, 4) * (rank + 2)

    inputs = [input1.to("neuron"), input2.to("neuron")]

    if rank <= 1:
        outputs = _c10d_functional.all_gather_into_tensor_coalesced(inputs, group_size, group_name)

        for out in outputs:
            _c10d_functional.wait_tensor(out)

        # Verify results for ranks in group
        expected1 = torch.cat([torch.ones(2, 3) * (i + 1) for i in range(group_size)], dim=0)
        expected2 = torch.cat([torch.ones(3, 4) * (i + 2) for i in range(group_size)], dim=0)

        assert torch.allclose(outputs[0].cpu(), expected1)
        assert torch.allclose(outputs[1].cpu(), expected2)


def run_5gb_tensor_test(rank, world_size, kwargs):
    """Test all_gather_into_tensor_coalesced with 5GB tensor (~1.25 billion float32 elements)."""
    tensor_size = 1250000000  # ~5GB for float32

    # Create multiple input tensors with different sizes
    input1 = torch.ones(tensor_size, dtype=torch.float32) * (rank + 1)
    input2 = torch.ones(tensor_size, dtype=torch.float32) * (rank + 2)

    inputs = [input1.to("neuron"), input2.to("neuron")]

    default_group = torch.distributed.group.WORLD

    # Use functional collective API
    outputs = _c10d_functional.all_gather_into_tensor_coalesced(
        inputs, world_size, default_group.group_name
    )

    for out in outputs:
        _c10d_functional.wait_tensor(out)

    # Verify results
    expected1 = torch.cat(
        [torch.ones(tensor_size, dtype=torch.float32) * (i + 1) for i in range(world_size)], dim=0
    )
    assert torch.allclose(outputs[0].cpu(), expected1), f"Rank {rank}: Output 0 mismatch"

    expected2 = torch.cat(
        [torch.ones(tensor_size, dtype=torch.float32) * (i + 2) for i in range(world_size)], dim=0
    )
    assert torch.allclose(outputs[1].cpu(), expected2), f"Rank {rank}: Output 1 mismatch"


def run_bucketing_memory_test(rank, world_size, kwargs):
    """Test all_gather_into_tensor_coalesced with bucketing to ensure memory stays under 1GB."""
    core_id = os.environ.get("NEURON_RT_VISIBLE_CORES", "0")
    dev_id = int(core_id) // 8
    core_id = int(core_id) - (dev_id * 8)
    os.system(
        f"echo 0 | sudo tee /sys/devices/virtual/neuron_device/neuron{dev_id}/"
        f"neuron_core{core_id}/stats/memory_usage/device_mem/model_shared_scratchpad/peak"
    )
    for i in range(3):
        tensor_size = 31250000 + i * 31250000
        input1 = torch.ones(tensor_size, dtype=torch.float32)
        input2 = torch.ones(tensor_size, dtype=torch.float32) * 2
        inputs = [input1.to("neuron"), input2.to("neuron")]

        default_group = torch.distributed.group.WORLD
        outputs = _c10d_functional.all_gather_into_tensor_coalesced(
            inputs, world_size, default_group.group_name
        )

        for out in outputs:
            _c10d_functional.wait_tensor(out)

        expected1 = torch.cat([torch.ones(tensor_size) for _ in range(world_size)], dim=0)
        expected2 = torch.cat([torch.ones(tensor_size) * 2 for _ in range(world_size)], dim=0)
        assert torch.allclose(outputs[0].cpu(), expected1)
        assert torch.allclose(outputs[1].cpu(), expected2)

        peak_raw = os.popen(
            f"cat /sys/devices/virtual/neuron_device/neuron{dev_id}/"
            f"neuron_core{core_id}/stats/memory_usage/device_mem/model_shared_scratchpad/peak"
        ).read()
        peak_str = peak_raw.strip()
        peak = int(peak_str) if peak_str else 0
        assert peak < 1073741824, f"Memory {peak} exceeds 1GB"


def run_bucketing_memory_flat_2d_tensor_test(rank, world_size, kwargs):
    """Test all_gather_into_tensor_coalesced with bucketing to ensure memory stays under 1GB."""
    core_id = os.environ.get("NEURON_RT_VISIBLE_CORES", "0")
    dev_id = int(core_id) // 8
    core_id = int(core_id) - (dev_id * 8)
    os.system(
        f"echo 0 | sudo tee /sys/devices/virtual/neuron_device/neuron{dev_id}/"
        f"neuron_core{core_id}/stats/memory_usage/device_mem/model_shared_scratchpad/peak"
    )
    for i in range(3):
        tensor_size = 31250000 + i * 31250000
        input1 = torch.ones(1, tensor_size, dtype=torch.float32)
        input2 = torch.ones(1, tensor_size, dtype=torch.float32) * 2
        inputs = [input1.to("neuron"), input2.to("neuron")]

        default_group = torch.distributed.group.WORLD
        outputs = _c10d_functional.all_gather_into_tensor_coalesced(
            inputs, world_size, default_group.group_name
        )

        for out in outputs:
            _c10d_functional.wait_tensor(out)

        expected1 = torch.cat([torch.ones(1, tensor_size) for _ in range(world_size)], dim=0)
        expected2 = torch.cat([torch.ones(1, tensor_size) * 2 for _ in range(world_size)], dim=0)
        assert torch.allclose(outputs[0].cpu(), expected1)
        assert torch.allclose(outputs[1].cpu(), expected2)

        peak_raw = os.popen(
            f"cat /sys/devices/virtual/neuron_device/neuron{dev_id}/"
            f"neuron_core{core_id}/stats/memory_usage/device_mem/model_shared_scratchpad/peak"
        ).read()
        peak_str = peak_raw.strip()
        peak = int(peak_str) if peak_str else 0
        assert peak < 1073741824, f"Memory {peak} exceeds 1GB"


class TestAllGatherIntoTensorCoalesced(BaseCollectiveOpTest):
    """Test cases for functional all_gather_into_tensor_coalesced."""

    def test_basic_coalesced(self):
        """Test basic coalesced all-gather with multiple tensors."""
        self.distributed_tester.run_test(run_basic_coalesced_test)

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
    def test_different_dtypes_coalesced(self, dtype):
        """Test coalesced all-gather with different data types."""
        self.distributed_tester.run_test(run_different_dtypes_coalesced_test, dtype=dtype)

    @pytest.mark.parametrize(
        "shapes",
        [
            [(2, 3), (4, 5)],
            [(10,), (20,), (30,)],
            [(5, 5, 5), (3, 3, 3)],
            [(2, 3, 4, 5), (1, 2, 3, 4)],
        ],
    )
    def test_mixed_shapes(self, shapes):
        """Test coalesced all-gather with various tensor shapes."""
        self.distributed_tester.run_test(run_mixed_shapes_test, shapes=shapes)

    def test_empty_tensor_mixed(self):
        """Test coalesced all-gather with mix of empty and non-empty tensors."""
        self.distributed_tester.run_test(run_empty_tensor_mixed_test)

    def test_large_tensors_coalesced(self):
        """Test coalesced all-gather with large tensors."""
        self.distributed_tester.run_test(run_large_tensors_coalesced_test)

    def test_with_special_values(self):
        """Test coalesced all-gather with inf and nan values."""
        self.distributed_tester.run_test(run_with_special_values_test)

    def test_single_tensor(self):
        """Test single tensor functional API."""
        self.distributed_tester.run_test(run_single_tensor_test)

    def test_group_coalesced(self):
        """Test coalesced all-gather with specific process groups."""
        self.distributed_tester.run_test(run_group_coalesced_test)

    @pytest.mark.xfail(reason="5GB tensor test expected to fail")
    def test_5gb_tensor(self):
        """Test all_gather_into_tensor_coalesced with 5GB tensor."""
        self.distributed_tester.run_test(run_5gb_tensor_test)

    @pytest.mark.xfail(reason="Memory assertions require further investigation, thus xfailing.")
    def test_bucketing_memory(self):
        """Test all_gather_into_tensor_coalesced with bucketing for memory efficiency."""
        self.distributed_tester.run_test(run_bucketing_memory_test)

    @pytest.mark.xfail(reason="Memory assertions require further investigation, thus xfailing.")
    def test_bucketing_memory_flat_2d_tensor_test(self):
        """Test all_gather_into_tensor with bucketing for memory efficiency with flat 2d tensor"""
        self.distributed_tester.run_test(run_bucketing_memory_flat_2d_tensor_test)
