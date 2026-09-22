import inspect
import os
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from tests.utils.neuron_test_utils import assert_raises

from .base_collective_op import BaseCollectiveOpTest


def verify_alltoall_results(world_size, rank, output_tensor_list, shape):
    # Verify results: output_tensor_list[src_rank] should contain what src_rank sent to us
    dtype = output_tensor_list[0].dtype
    for src_rank in range(world_size):
        expected_value = src_rank * 10 + rank + 1  # What src_rank sent to us (rank)
        expected_tensor = torch.full(shape, expected_value, dtype=dtype)
        assert torch.allclose(output_tensor_list[src_rank].cpu(), expected_tensor), (
            f"Rank {rank}: Expected {expected_tensor} from rank {src_rank}, "
            f"got {output_tensor_list[src_rank].cpu()}"
        )


def run_basic_alltoall_test(rank, world_size, kwargs):
    """Test basic alltoall functionality."""
    # Create input tensors - each rank sends different data to each other rank
    input_tensor_list = []
    for dst_rank in range(world_size):
        # Rank sends tensor with value (rank * 100 + dst_rank) to dst_rank
        tensor_value = rank * 10 + dst_rank + 1
        input_tensor_list.append(torch.full((5,), tensor_value, dtype=torch.float32).to("neuron"))

    # Create output tensors
    output_tensor_list = [
        torch.zeros(5, dtype=torch.float32).to("neuron") for _ in range(world_size)
    ]

    # Perform alltoall
    dist.all_to_all(output_tensor_list, input_tensor_list)

    verify_alltoall_results(world_size, rank, output_tensor_list, (5,))


def run_different_dtype_test(rank, world_size, kwargs):
    """Test alltoall with different data types."""
    dtype = kwargs["dtype"]

    input_tensor_list = []
    for dst_rank in range(world_size):
        tensor_value = rank * 10 + dst_rank + 1
        input_tensor_list.append(torch.full((3,), tensor_value, dtype=dtype).to("neuron"))

    output_tensor_list = [torch.zeros(3, dtype=dtype).to("neuron") for _ in range(world_size)]

    dist.all_to_all(output_tensor_list, input_tensor_list)
    verify_alltoall_results(world_size, rank, output_tensor_list, (3,))


def run_different_shape_test(rank, world_size, kwargs):
    """Test alltoall with different tensor shapes."""
    shape = kwargs["shape"]

    input_tensor_list = []
    for dst_rank in range(world_size):
        tensor_value = rank * 10 + dst_rank + 1
        input_tensor_list.append(torch.full(shape, tensor_value, dtype=torch.float32).to("neuron"))

    output_tensor_list = [
        torch.zeros(shape, dtype=torch.float32).to("neuron") for _ in range(world_size)
    ]

    dist.all_to_all(output_tensor_list, input_tensor_list)
    verify_alltoall_results(world_size, rank, output_tensor_list, shape)


def run_identity_alltoall_test(rank, world_size, kwargs):
    """Test alltoall where each rank sends the same tensor to all ranks."""
    # Each rank sends its rank value to all other ranks
    input_tensor_list = [
        torch.full((4,), rank * 10 + dst_rank + 1, dtype=torch.float32).to("neuron")
        for dst_rank in range(world_size)
    ]
    output_tensor_list = [
        torch.zeros(4, dtype=torch.float32).to("neuron") for _ in range(world_size)
    ]

    dist.all_to_all(output_tensor_list, input_tensor_list)
    verify_alltoall_results(world_size, rank, output_tensor_list, (4,))


def run_async_operation_test(rank, world_size, kwargs):
    """Test asynchronous alltoall operation."""
    input_tensor_list = []
    for dst_rank in range(world_size):
        tensor_value = rank * 10 + dst_rank + 1
        input_tensor_list.append(torch.full((3,), tensor_value, dtype=torch.float32).to("neuron"))

    output_tensor_list = [
        torch.zeros(3, dtype=torch.float32).to("neuron") for _ in range(world_size)
    ]

    # Test async operation
    work = dist.all_to_all(output_tensor_list, input_tensor_list, async_op=True)
    assert work is not None
    work.wait()
    verify_alltoall_results(world_size, rank, output_tensor_list, (3,))


def run_empty_tensor_test(rank, world_size, kwargs):
    """Test alltoall with empty tensors."""
    input_tensor_list = [
        torch.empty(0, dtype=torch.float32).to("neuron") for _ in range(world_size)
    ]
    output_tensor_list = [
        torch.empty(0, dtype=torch.float32).to("neuron") for _ in range(world_size)
    ]

    dist.all_to_all(output_tensor_list, input_tensor_list)

    # Verify all output tensors are empty
    for i, output_tensor in enumerate(output_tensor_list):
        assert output_tensor.numel() == 0, f"Output tensor {i} should be empty"


def run_large_tensor_test(rank, world_size, kwargs):
    """Test alltoall with large tensors."""
    # Create large tensors (100K elements each)
    input_tensor_list = []
    for dst_rank in range(world_size):
        tensor_value = rank * 10 + dst_rank + 1
        input_tensor_list.append(
            torch.full((100000,), tensor_value, dtype=torch.float32).to("neuron")
        )

    output_tensor_list = [
        torch.zeros(100000, dtype=torch.float32).to("neuron") for _ in range(world_size)
    ]

    dist.all_to_all(output_tensor_list, input_tensor_list)
    verify_alltoall_results(world_size, rank, output_tensor_list, (100000,))


def run_with_inf_inputs_test(rank, world_size, kwargs):
    """Test alltoall with tensors containing Inf values."""
    input_tensor_list = []
    for dst_rank in range(world_size):
        if dst_rank == rank:
            # Send inf to ourselves
            input_tensor_list.append(torch.tensor([torch.inf] * 3).to("neuron"))
        else:
            # Send finite values to others
            tensor_value = rank * 10 + dst_rank + 1
            input_tensor_list.append(
                torch.full((3,), tensor_value, dtype=torch.float32).to("neuron")
            )

    output_tensor_list = [
        torch.zeros(3, dtype=torch.float32).to("neuron") for _ in range(world_size)
    ]

    dist.all_to_all(output_tensor_list, input_tensor_list)

    # Verify results
    for src_rank in range(world_size):
        if src_rank == rank:
            # We should receive inf from ourselves
            assert torch.isinf(output_tensor_list[src_rank].cpu()).all()
        else:
            # We should receive finite values from others
            expected_value = src_rank * 10 + rank + 1
            expected_tensor = torch.full((3,), expected_value, dtype=torch.float32)
            assert torch.allclose(output_tensor_list[src_rank].cpu(), expected_tensor)


def run_with_nan_inputs_test(rank, world_size, kwargs):
    """Test alltoall with tensors containing NaN values."""
    input_tensor_list = []
    for dst_rank in range(world_size):
        if dst_rank == 0:
            # Send nan to rank 0
            input_tensor_list.append(torch.tensor([torch.nan] * 3).to("neuron"))
        else:
            # Send finite values to others
            tensor_value = rank * 10 + dst_rank + 1
            input_tensor_list.append(
                torch.full((3,), tensor_value, dtype=torch.float32).to("neuron")
            )

    output_tensor_list = [
        torch.zeros(3, dtype=torch.float32).to("neuron") for _ in range(world_size)
    ]

    dist.all_to_all(output_tensor_list, input_tensor_list)

    # Verify results
    for src_rank in range(world_size):
        if rank == 0:
            # Rank 0 should receive nan from all ranks
            assert torch.isnan(output_tensor_list[src_rank].cpu()).all()
        else:
            # Other ranks should receive finite values
            expected_value = src_rank * 10 + rank + 1
            expected_tensor = torch.full((3,), expected_value, dtype=torch.float32)
            assert torch.allclose(output_tensor_list[src_rank].cpu(), expected_tensor)


def run_wrong_list_size_test(rank, world_size, kwargs):
    """Test alltoall with incorrect list sizes."""
    # Create input list with wrong size
    input_tensor_list = [torch.ones(3).to("neuron") for _ in range(world_size - 1)]  # Wrong size
    output_tensor_list = [torch.zeros(3).to("neuron") for _ in range(world_size)]

    dist.all_to_all(output_tensor_list, input_tensor_list)


def run_trailing_dim_mismatch_test(rank, world_size, kwargs):
    """alltoall where chunks disagree on a dim >= 1 (the per-row feature size).

    Only dim 0 may vary across chunks (variable-split all-to-all); a mismatch in a
    trailing dim is unrepresentable and must raise. Uses 2D chunks so dim 0 is equal
    (rows=2) but dim 1 differs (4 vs 3)."""
    input_tensor_list = []
    for dst_rank in range(world_size):
        if dst_rank == 0:
            input_tensor_list.append(torch.ones(2, 4).to("neuron"))
        else:
            input_tensor_list.append(torch.ones(2, 3).to("neuron"))

    output_tensor_list = [torch.zeros(2, 3).to("neuron") for _ in range(world_size)]

    dist.all_to_all(output_tensor_list, input_tensor_list)


def run_dim0_mismatch_flag_off_test(rank, world_size, kwargs):
    """alltoall with unequal dim-0 chunk shapes and the NKI a2a-v flag OFF.

    Variable-split all-to-all (unequal dim 0) requires the native NKI all_to_all_v
    path. With the flag off it must raise an actionable error telling the user to
    enable NEURON_USE_NKI_ALLTOALLV, NOT the old cryptic 'same shape' assert."""
    input_tensor_list = []
    for dst_rank in range(world_size):
        if dst_rank == 0:
            input_tensor_list.append(torch.ones(5).to("neuron"))
        else:
            input_tensor_list.append(torch.ones(3).to("neuron"))

    output_tensor_list = [torch.zeros(3).to("neuron") for _ in range(world_size)]

    with patch.dict(os.environ, {"NEURON_USE_NKI_ALLTOALLV": "0"}):
        dist.all_to_all(output_tensor_list, input_tensor_list)


def run_dtype_mismatch_test(rank, world_size, kwargs):
    """Test alltoall with mismatched tensor dtypes."""
    input_tensor_list = []
    for dst_rank in range(world_size):
        if dst_rank == 0:
            # Different dtype for first tensor
            input_tensor_list.append(torch.ones(3, dtype=torch.float32).to("neuron"))
        else:
            # Different dtype for other tensors
            input_tensor_list.append(torch.ones(3, dtype=torch.int32).to("neuron"))

    output_tensor_list = [
        torch.zeros(3, dtype=torch.float32).to("neuron") for _ in range(world_size)
    ]

    dist.all_to_all(output_tensor_list, input_tensor_list)


def run_partial_group_alltoall_test(rank, world_size, kwargs):
    """Test alltoall with a partial group.

    Args:
        rank: Current process rank
        world_size: Total number of processes
        kwargs: Dictionary containing:
            - partial_group: List of ranks in the partial group
    """
    partial_group = kwargs.get("partial_group", [0, 1])

    # Create the partial group
    group = dist.new_group(partial_group)
    if rank in partial_group:
        group_size = len(partial_group)

        # Create input tensors - each rank sends different data to each other rank in the group
        input_tensor_list = []
        for i in range(len(partial_group)):
            # Rank sends tensor with value (rank * 10 + position_in_group + 1) to dst_rank
            tensor_value = rank * 10 + i + 1
            input_tensor_list.append(
                torch.full((5,), tensor_value, dtype=torch.float32).to("neuron")
            )

        # Create output tensors for the group
        output_tensor_list = [
            torch.zeros(5, dtype=torch.float32).to("neuron") for _ in range(group_size)
        ]

        # Perform alltoall within the partial group
        dist.all_to_all(output_tensor_list, input_tensor_list, group=group)

        # Verify results: output_tensor_list[src_idx] should contain what
        # partial_group[src_idx] sent to us
        for src_idx, src_rank in enumerate(partial_group):
            rank_idx_in_group = partial_group.index(rank)
            expected_value = (
                src_rank * 10 + rank_idx_in_group + 1
            )  # What src_rank sent to us (rank)
            expected_tensor = torch.full((5,), expected_value, dtype=torch.float32)
            assert torch.allclose(output_tensor_list[src_idx].cpu(), expected_tensor), (
                f"Rank {rank}: Expected {expected_tensor} from rank {src_rank}, "
                f"got {output_tensor_list[src_idx].cpu()}"
            )
    else:
        # Ranks not in partial_group don't participate
        pass


def run_strided_group_alltoall_test(rank, world_size, kwargs):
    """Every rank runs alltoall in its own strided group.

    The groups partition the world: rank i joins the ranks i + k * stride. The runtime needs
    every rank to load the collective. On trn2 a group that spans two chips fails in the
    replica-group validation when another rank stays idle.
    """
    stride = kwargs["stride"]
    groups = [list(range(i, world_size, stride)) for i in range(stride)]
    # dist.new_group is collective. Every rank must create every group.
    process_groups = [dist.new_group(g) for g in groups]
    group_ranks = groups[rank % stride]
    # On trn2 a cross-chip pair needs a chunk of 4*k elements. The runtime asserts otherwise.
    chunk = 8

    input_tensor_list = [
        torch.full((chunk,), rank * 10 + i + 1, dtype=torch.float32).to("neuron")
        for i in range(len(group_ranks))
    ]
    output_tensor_list = [
        torch.zeros(chunk, dtype=torch.float32).to("neuron") for _ in range(len(group_ranks))
    ]

    dist.all_to_all(output_tensor_list, input_tensor_list, group=process_groups[rank % stride])

    rank_idx_in_group = group_ranks.index(rank)
    for src_idx, src_rank in enumerate(group_ranks):
        expected_value = src_rank * 10 + rank_idx_in_group + 1
        expected_tensor = torch.full((chunk,), expected_value, dtype=torch.float32)
        assert torch.allclose(output_tensor_list[src_idx].cpu(), expected_tensor), (
            f"Rank {rank}: Expected {expected_tensor} from rank {src_rank}, "
            f"got {output_tensor_list[src_idx].cpu()}"
        )


def run_5gb_tensor_test(rank, world_size, kwargs):
    """Test alltoall with 5GB tensor (~1.25 billion float32 elements)."""
    tensor_size = 1250000000  # ~5GB for float32

    # Create input tensors - each rank sends different data to each other rank
    input_tensor_list = []
    for dst_rank in range(world_size):
        tensor_value = rank * 10 + dst_rank + 1
        input_tensor_list.append(
            torch.full((tensor_size,), tensor_value, dtype=torch.float32).to("neuron")
        )

    # Create output tensors
    output_tensor_list = [
        torch.zeros(tensor_size, dtype=torch.float32).to("neuron") for _ in range(world_size)
    ]

    # Perform alltoall
    dist.all_to_all(output_tensor_list, input_tensor_list)

    verify_alltoall_results(world_size, rank, output_tensor_list, (tensor_size,))


def run_different_device_test(rank, world_size, kwargs):
    """Test alltoall with tensors on different devices (should fail)."""
    input_tensor_list = [torch.ones(3) for _ in range(world_size)]  # CPU tensors
    output_tensor_list = [torch.zeros(3) for _ in range(world_size)]  # CPU tensors

    dist.all_to_all(output_tensor_list, input_tensor_list)


class TestAllToAll(BaseCollectiveOpTest):
    """Test cases for torch.distributed.all_to_all."""

    @property
    def world_size(self) -> int:
        return 4

    def test_basic_alltoall(self):
        """Test basic alltoall functionality."""
        self.distributed_tester.run_test(run_basic_alltoall_test)

    @pytest.mark.parametrize(
        "dtype",
        [torch.float32, torch.float16, torch.int32, torch.int64, torch.bfloat16],
    )
    def test_different_dtypes(self, dtype):
        """Test alltoall with different data types."""
        self.distributed_tester.run_test(run_different_dtype_test, dtype=dtype)

    @pytest.mark.parametrize("shape", [(10,), (5, 4), (2, 3, 4), (2, 2, 2, 2)])
    def test_different_shapes(self, shape):
        """Test alltoall with different tensor shapes."""
        self.distributed_tester.run_test(run_different_shape_test, shape=shape)

    def test_identity_alltoall(self):
        """Test alltoall where each rank sends the same data to all ranks."""
        self.distributed_tester.run_test(run_identity_alltoall_test)

    def test_async_operation(self):
        """Test asynchronous alltoall operation."""
        self.distributed_tester.run_test(run_async_operation_test)

    def test_empty_tensor(self):
        """Test alltoall with empty tensors."""
        self.distributed_tester.run_test(run_empty_tensor_test)

    def test_large_tensor(self):
        """Test alltoall with large tensors."""
        self.distributed_tester.run_test(run_large_tensor_test)

    def test_with_inf_inputs(self):
        """Test alltoall with tensors containing Inf values."""
        self.distributed_tester.run_test(run_with_inf_inputs_test)

    def test_with_nan_inputs(self):
        """Test alltoall with tensors containing NaN values."""
        self.distributed_tester.run_test(run_with_nan_inputs_test)

    @assert_raises(RuntimeError, match=r".*number of input tensors.*must equal world size.*")
    def test_wrong_list_size_error(self):
        """Test alltoall error handling with wrong list sizes."""
        self.distributed_tester.run_test(run_wrong_list_size_test)

    @assert_raises(
        RuntimeError,
        match=r".*must have identical shape in dims >= 1.*only dim 0 may vary.*",
    )
    def test_trailing_dim_mismatch_error(self):
        """Chunks disagreeing on a dim >= 1 (per-row feature size) must raise; only
        dim 0 may vary for variable-split all-to-all."""
        self.distributed_tester.run_test(run_trailing_dim_mismatch_test)

    @assert_raises(
        RuntimeError,
        match=r".*unequal shapes.*NEURON_USE_NKI_ALLTOALLV.*",
    )
    def test_dim0_mismatch_flag_off_error(self):
        """Unequal dim-0 chunks with the NKI a2a-v flag OFF raise an actionable
        error (enable NEURON_USE_NKI_ALLTOALLV), not the old 'same shape' assert."""
        self.distributed_tester.run_test(run_dim0_mismatch_flag_off_test)

    @assert_raises(
        RuntimeError,
        match=(
            r".*Invalid usage of tensors with different dtypes"
            r"Found torch.float32 and  torch.int32.*"
        ),
    )
    def test_dtype_mismatch_error(self):
        """Test alltoall error handling with dtype mismatch. Error thrown by torch directly"""
        self.distributed_tester.run_test(run_dtype_mismatch_test)

    @assert_raises(
        RuntimeError,
        match=r".*(Expected neuron device, got cpu"
        r"|No backend type associated with device type cpu).*",
    )
    def test_different_devices_error(self):
        """Test alltoall with tensors on different devices."""
        self.distributed_tester.run_test(run_different_device_test)

    @pytest.mark.xfail(reason="5GB tensor test expected to fail")
    def test_5gb_tensor(self):
        """Test alltoall with 5GB tensor."""
        self.distributed_tester.run_test(run_5gb_tensor_test)


class TestAllToAllWorldSize8(BaseCollectiveOpTest):
    """Test cases for alltoall with partial groups in world size 8 (hardware supported)."""

    @property
    def world_size(self) -> int:
        return 8

    @pytest.mark.parametrize(
        "group_config",
        [
            {
                "partial_group": [0, 1, 2, 3],  # Group size 4 - supported
            },
            {
                "partial_group": [4, 5, 6, 7],  # Group size 4 - supported
            },
        ],
    )
    def test_alltoall_partial_group_supported(self, group_config):
        """Test alltoall with partial groups that meet hardware requirements."""
        partial_group = group_config["partial_group"]
        self.distributed_tester.run_test(
            run_partial_group_alltoall_test,
            partial_group=partial_group,
        )

    @pytest.mark.parametrize("partial_group", [[1, 3]])
    def test_alltoall_partial_group_unaligned_start(self, partial_group):
        """Groups whose start rank is not a multiple of the group size still work.

        NeuronBackend::alltoall used to reject these up front. Which groups the device
        interconnect can form an alltoall over is a topology property that a start-rank
        rule cannot express: it excluded these working groups while still admitting
        [0, 2, 4, 6], which the runtime rejects (see below), so the check was removed and
        the runtime is now the only arbiter.
        """
        self.distributed_tester.run_test(
            run_partial_group_alltoall_test,
            partial_group=partial_group,
        )

    def test_alltoall_strided_group(self):
        """Pairs of rank i and rank i + 4 across two chips.

        On trn2 the fabric links rank i to rank i + 4 only. The pair works only when every
        rank in the world takes part in some group, so all four pairs run at once.
        """
        self.distributed_tester.run_test(run_strided_group_alltoall_test, stride=4)

    @pytest.mark.xfail(
        strict=True,
        reason="the device interconnect cannot form an alltoall over these groups; the "
        "runtime rejects them with 'NRT model scheduling failed'",
    )
    @pytest.mark.parametrize("partial_group", [[0, 2, 4, 6], [2, 3, 4, 5]])
    def test_alltoall_partial_group_unsupported_topology(self, partial_group):
        self.distributed_tester.run_test(
            run_partial_group_alltoall_test,
            partial_group=partial_group,
        )
