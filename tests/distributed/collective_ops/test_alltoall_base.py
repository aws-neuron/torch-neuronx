import contextlib

import pytest
import torch
import torch.distributed as dist

from tests.utils.neuron_test_utils import assert_raises

from .base_collective_op import BaseCollectiveOpTest


def verify_alltoall_base_results(world_size, rank, output_tensor, input_tensor, group_ranks=None):
    """Verify results for alltoall_base operation.

    Args:
        world_size: Total world size
        rank: Current rank
        output_tensor: Output tensor from alltoall_base
        input_tensor: Input tensor to alltoall_base
        group_ranks: List of ranks in the group (None for full world)
    """
    if group_ranks is None:
        group_ranks = list(range(world_size))

    group_size = len(group_ranks)
    chunk_size = input_tensor.size(0) // group_size

    # Verify output tensor has correct data from each rank in the group
    for i, src_rank in enumerate(group_ranks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size

        # Expected value: what src_rank sent to us (current rank)
        rank_idx_in_group = group_ranks.index(rank)
        expected_value = src_rank * 10 + rank_idx_in_group + 1
        expected_chunk = torch.full((chunk_size,), expected_value, dtype=output_tensor.dtype)

        actual_chunk = output_tensor[start_idx:end_idx].cpu()
        assert torch.allclose(actual_chunk, expected_chunk), (
            f"Rank {rank}: Expected {expected_chunk} from rank {src_rank} at "
            f"indices {start_idx}:{end_idx}, got {actual_chunk}."
        )


def run_basic_alltoall_base_test(rank, world_size, kwargs):
    """Test basic alltoall_base functionality."""
    # Create input tensor - each chunk goes to a different rank
    input_tensor = torch.zeros(world_size * 2, dtype=torch.float32)
    for dst_rank in range(world_size):
        start_idx = dst_rank * 2
        end_idx = (dst_rank + 1) * 2
        # Fill chunk with value (rank * 10 + dst_rank + 1)
        input_tensor[start_idx:end_idx] = rank * 10 + dst_rank + 1

    input_tensor = input_tensor.to("neuron")

    # Create output tensor
    output_tensor = torch.zeros_like(input_tensor)

    # Perform alltoall_base
    dist.all_to_all_single(output_tensor, input_tensor)

    verify_alltoall_base_results(world_size, rank, output_tensor, input_tensor)


def run_different_dtype_alltoall_base_test(rank, world_size, kwargs):
    """Test alltoall_base with different data types."""
    dtype = kwargs["dtype"]

    # Create input tensor
    input_tensor = torch.zeros(world_size * 3, dtype=dtype)
    for dst_rank in range(world_size):
        start_idx = dst_rank * 3
        end_idx = (dst_rank + 1) * 3
        input_tensor[start_idx:end_idx] = rank * 10 + dst_rank + 1

    input_tensor = input_tensor.to("neuron")
    output_tensor = torch.zeros_like(input_tensor)

    dist.all_to_all_single(output_tensor, input_tensor)
    verify_alltoall_base_results(world_size, rank, output_tensor, input_tensor)


def run_different_shape_alltoall_base_test(rank, world_size, kwargs):
    """Test alltoall_base with different tensor shapes."""
    chunk_size = kwargs["chunk_size"]

    # Create input tensor
    input_tensor = torch.zeros(world_size * chunk_size, dtype=torch.float32)
    for dst_rank in range(world_size):
        start_idx = dst_rank * chunk_size
        end_idx = (dst_rank + 1) * chunk_size
        input_tensor[start_idx:end_idx] = rank * 10 + dst_rank + 1

    input_tensor = input_tensor.to("neuron")
    output_tensor = torch.zeros_like(input_tensor)

    dist.all_to_all_single(output_tensor, input_tensor)
    verify_alltoall_base_results(world_size, rank, output_tensor, input_tensor)


def run_async_alltoall_base_test(rank, world_size, kwargs):
    """Test asynchronous alltoall_base operation."""
    input_tensor = torch.zeros(world_size * 2, dtype=torch.float32)
    for dst_rank in range(world_size):
        start_idx = dst_rank * 2
        end_idx = (dst_rank + 1) * 2
        input_tensor[start_idx:end_idx] = rank * 10 + dst_rank + 1

    input_tensor = input_tensor.to("neuron")
    output_tensor = torch.zeros_like(input_tensor)

    # Test async operation
    work = dist.all_to_all_single(output_tensor, input_tensor, async_op=True)
    assert work is not None
    work.wait()

    verify_alltoall_base_results(world_size, rank, output_tensor, input_tensor)


def run_empty_tensor_alltoall_base_test(rank, world_size, kwargs):
    """Test alltoall_base with empty tensors."""
    input_tensor = torch.empty(0, dtype=torch.float32).to("neuron")
    output_tensor = torch.empty(0, dtype=torch.float32).to("neuron")

    dist.all_to_all_single(output_tensor, input_tensor)

    # Verify output tensor is empty
    assert output_tensor.numel() == 0, "Output tensor should be empty"


def run_split_sizes_alltoall_base_test(rank, world_size, kwargs):
    """Test alltoall_base with split sizes (all 1s - the only supported case)."""
    # Create input tensor with world_size elements (each chunk size 1)
    input_tensor = torch.zeros(world_size, dtype=torch.float32)
    for dst_rank in range(world_size):
        input_tensor[dst_rank] = rank * 10 + dst_rank + 1

    input_tensor = input_tensor.to("neuron")
    output_tensor = torch.zeros_like(input_tensor)

    # Split sizes of all 1s (the only supported non-None case)
    split_sizes = [1] * world_size

    dist.all_to_all_single(
        output_tensor, input_tensor, output_split_sizes=split_sizes, input_split_sizes=split_sizes
    )

    verify_alltoall_base_results(world_size, rank, output_tensor, input_tensor)


def run_partial_group_alltoall_base_test(rank, world_size, kwargs):
    """Test alltoall_base with a partial group.

    Args:
        rank: Current process rank
        world_size: Total number of processes
        kwargs: Dictionary containing:
            - partial_group: List of ranks in the partial group
            - expected_values: List of expected values for verification
    """
    partial_group = kwargs.get("partial_group", [0, 1])

    # Create the partial group
    group = dist.new_group(partial_group)

    if rank in partial_group:
        group_size = len(partial_group)
        chunk_size = 2  # Each rank sends 2 elements to each other rank in group

        # Create input tensor for the group
        input_tensor = torch.zeros(group_size * chunk_size, dtype=torch.float32)

        # Fill input tensor: each chunk goes to a different rank in the group
        for i in range(len(partial_group)):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size
            # Value: rank * 10 + position_in_group + 1
            input_tensor[start_idx:end_idx] = rank * 10 + i + 1

        input_tensor = input_tensor.to("neuron")
        output_tensor = torch.zeros_like(input_tensor)

        # Perform alltoall_base within the partial group
        dist.all_to_all_single(output_tensor, input_tensor, group=group)

        # Verify results
        verify_alltoall_base_results(world_size, rank, output_tensor, input_tensor, partial_group)
    else:
        # Ranks not in partial_group don't participate
        pass

    # Barrier on the default group: participating ranks can finish at
    # different times; the compilation service is single-owner by design,
    # and the first rank to exit tears down its shared socket, starving
    # any in-flight compiles on peers.
    dist.barrier()


def run_strided_group_alltoall_base_test(rank, world_size, kwargs):
    """Every rank runs alltoall_base in its own strided group.

    The groups partition the world: rank i joins the ranks i + k * stride. The runtime needs
    every rank to load the collective. On trn2 a group that spans two chips fails in the
    replica-group validation when another rank stays idle.
    """
    stride = kwargs["stride"]
    groups = [list(range(i, world_size, stride)) for i in range(stride)]
    # dist.new_group is collective. Every rank must create every group.
    process_groups = [dist.new_group(g) for g in groups]
    group_ranks = groups[rank % stride]
    group_size = len(group_ranks)
    # On trn2 a cross-chip pair needs a chunk of 4*k elements. The runtime asserts otherwise.
    chunk_size = 4

    input_tensor = torch.zeros(group_size * chunk_size, dtype=torch.float32)
    for i in range(group_size):
        input_tensor[i * chunk_size : (i + 1) * chunk_size] = rank * 10 + i + 1
    input_tensor = input_tensor.to("neuron")
    output_tensor = torch.zeros_like(input_tensor)

    dist.all_to_all_single(output_tensor, input_tensor, group=process_groups[rank % stride])

    verify_alltoall_base_results(world_size, rank, output_tensor, input_tensor, group_ranks)


def run_5gb_tensor_test(rank, world_size, kwargs):
    """Test alltoall_base with 5GB tensor (~1.25 billion float32 elements)."""
    tensor_size = 1250000000  # ~5GB for float32

    # Create input tensor - each chunk goes to a different rank
    input_tensor = torch.zeros(world_size * tensor_size, dtype=torch.float32)
    for dst_rank in range(world_size):
        start_idx = dst_rank * tensor_size
        end_idx = (dst_rank + 1) * tensor_size
        # Fill chunk with value (rank * 10 + dst_rank + 1)
        input_tensor[start_idx:end_idx] = rank * 10 + dst_rank + 1

    input_tensor = input_tensor.to("neuron")

    # Create output tensor
    output_tensor = torch.zeros_like(input_tensor)

    # Perform alltoall_base
    dist.all_to_all_single(output_tensor, input_tensor)

    verify_alltoall_base_results(world_size, rank, output_tensor, input_tensor)


def run_uneven_split_sizes_alltoall_base_test(rank, world_size, kwargs):
    """Test alltoall_base with uneven split sizes (e.g. permute_tensor pattern).

    Simulates the pattern used by permute_tensor in ring attention backward:
    each rank sends its entire buffer to exactly one destination rank and
    receives from exactly one source rank.  All other split sizes are 0.

    With next_ranks = [1, 2, 3, 0]:
      rank 0 sends to rank 1  -> input_split_sizes  = [0, N, 0, 0]
      rank 0 receives from rank 3 -> output_split_sizes = [0, 0, 0, N]
    """
    N = 4  # elements per rank # noqa N806

    # next_ranks[r] = (r + 1) % world_size
    dst = (rank + 1) % world_size
    src = (rank - 1) % world_size

    input_split_sizes = [0] * world_size
    input_split_sizes[dst] = N

    output_split_sizes = [0] * world_size
    output_split_sizes[src] = N

    # Input: fill with a rank-specific value so we can verify correctness
    input_tensor = torch.full((N,), float(rank + 1), dtype=torch.float32).to("neuron")
    output_tensor = torch.zeros(N, dtype=torch.float32).to("neuron")

    use_nki = kwargs.get("use_nki")
    flag_context = pytest.MonkeyPatch.context() if use_nki is not None else contextlib.nullcontext()
    with flag_context as monkeypatch:
        if use_nki is not None:
            monkeypatch.setenv("NEURON_USE_NKI_ALLTOALLV", use_nki)
        dist.all_to_all_single(
            output_tensor,
            input_tensor,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
        )

    # rank 0 should receive from rank 3 (value = 4.0), etc.
    expected_value = float(src + 1)
    expected = torch.full((N,), expected_value, dtype=torch.float32)
    actual = output_tensor.cpu()
    assert torch.allclose(
        actual, expected
    ), f"Rank {rank}: expected {expected} (from rank {src}), got {actual}"


def run_invalid_split_sizes_test(rank, world_size, kwargs):
    """Test alltoall_base error handling with invalid split sizes (wrong length)."""
    input_tensor = torch.ones(world_size * 2).to("neuron")
    output_tensor = torch.zeros_like(input_tensor)

    # Wrong length — should raise
    bad_split_sizes = [2] * (world_size + 1)
    dist.all_to_all_single(output_tensor, input_tensor, input_split_sizes=bad_split_sizes)


class TestAllToAllBase(BaseCollectiveOpTest):
    """Test cases for torch.distributed.all_to_all_single (alltoall_base)."""

    @property
    def world_size(self) -> int:
        return 4

    def test_basic_alltoall_base(self):
        """Test basic alltoall_base functionality."""
        self.distributed_tester.run_test(run_basic_alltoall_base_test)

    @pytest.mark.parametrize(
        "dtype",
        [torch.float32, torch.float16, torch.int32, torch.int64, torch.bfloat16],
    )
    def test_different_dtypes_alltoall_base(self, dtype):
        """Test alltoall_base with different data types."""
        self.distributed_tester.run_test(run_different_dtype_alltoall_base_test, dtype=dtype)

    @pytest.mark.parametrize("chunk_size", [1, 3, 5, 10])
    def test_different_shapes_alltoall_base(self, chunk_size):
        """Test alltoall_base with different chunk sizes."""
        self.distributed_tester.run_test(
            run_different_shape_alltoall_base_test, chunk_size=chunk_size
        )

    def test_async_operation_alltoall_base(self):
        """Test asynchronous alltoall_base operation."""
        self.distributed_tester.run_test(run_async_alltoall_base_test)

    def test_empty_tensor_alltoall_base(self):
        """Test alltoall_base with empty tensors."""
        self.distributed_tester.run_test(run_empty_tensor_alltoall_base_test)

    def test_split_sizes_alltoall_base(self):
        """Test alltoall_base with split sizes (all 1s)."""
        self.distributed_tester.run_test(run_split_sizes_alltoall_base_test)

    @pytest.mark.xfail(
        reason=(
            "The NKI version in PR CI supports all_to_all_v receive displacements "
            "only on NeuronSwitch-based instances; remove after the Trn2 artifacts are updated"
        )
    )
    def test_uneven_split_sizes_alltoall_base(self):
        """Test alltoall_base with uneven split sizes (permute_tensor / ring-attention pattern)."""
        self.distributed_tester.run_test(run_uneven_split_sizes_alltoall_base_test)

    @assert_raises(RuntimeError, match=r".*split_sizes length.*must match group_size.*")
    def test_invalid_split_sizes_error(self):
        """Test alltoall_base error handling with wrong-length split sizes."""
        self.distributed_tester.run_test(run_invalid_split_sizes_test)

    @pytest.mark.xfail(reason="5GB tensor test expected to fail")
    def test_5gb_tensor(self):
        """Test alltoall_base with 5GB tensor."""
        self.distributed_tester.run_test(run_5gb_tensor_test)


class TestAllToAllBaseWorldSize2(BaseCollectiveOpTest):
    """alltoall_base over a 2-rank group.

    Exercises the smallest collective group size. The runtime historically
    accepted only sizes 4, 8, 16, or multiples of 32; these cases confirm a
    2-rank group works end-to-end for even, async, and uneven (permute-style)
    exchanges.
    """

    @property
    def world_size(self) -> int:
        return 2

    def test_basic_alltoall_base(self):
        """Test basic alltoall_base functionality with world size 2."""
        self.distributed_tester.run_test(run_basic_alltoall_base_test)

    @pytest.mark.parametrize(
        "dtype",
        [torch.float32, torch.float16, torch.int32, torch.int64, torch.bfloat16],
    )
    def test_different_dtypes_alltoall_base(self, dtype):
        """Test alltoall_base with different data types with world size 2."""
        self.distributed_tester.run_test(run_different_dtype_alltoall_base_test, dtype=dtype)

    @pytest.mark.parametrize("chunk_size", [1, 3, 5, 10])
    def test_different_shapes_alltoall_base(self, chunk_size):
        """Test alltoall_base with different chunk sizes with world size 2."""
        self.distributed_tester.run_test(
            run_different_shape_alltoall_base_test, chunk_size=chunk_size
        )

    def test_async_operation_alltoall_base(self):
        """Test asynchronous alltoall_base operation with world size 2."""
        self.distributed_tester.run_test(run_async_alltoall_base_test)

    def test_split_sizes_alltoall_base(self):
        """Test alltoall_base with split sizes (all 1s) with world size 2."""
        self.distributed_tester.run_test(run_split_sizes_alltoall_base_test)

    def test_uneven_split_sizes_alltoall_base(self):
        """The compatibility fallback supports the world-size-2 permute pattern."""
        self.distributed_tester.run_test(
            run_uneven_split_sizes_alltoall_base_test,
            use_nki="0",
        )

    @pytest.mark.parametrize("use_nki", [None, "1"])
    def test_uneven_split_sizes_alltoall_base_routes_around_nki(self, use_nki):
        """A 2-rank group is half a device, which the NKI all_to_all_v kernel cannot serve.

        With the NKI path enabled (default or explicit) the backend must route the uneven
        split to the padded fallback instead of dispatching the kernel, which hangs. This is
        the pattern torch's ring-attention backward emits on a context-parallel group.
        """
        self.distributed_tester.run_test(
            run_uneven_split_sizes_alltoall_base_test,
            use_nki=use_nki,
        )


@pytest.mark.neuron_cores(8)
class TestAllToAllBasePartialGroups(BaseCollectiveOpTest):
    """alltoall_base over partial groups of an 8-rank world.

    alltoall_base delegates to NeuronBackend::alltoall for both the even-split fast path and
    the padded uneven-split path, so it inherits alltoall's group support and needs no group
    validation of its own: groups whose start rank is not a multiple of the group size work
    (they were rejected before that check was removed), and the shapes the device
    interconnect cannot form are rejected by the runtime -- pinned for both entry points by
    test_alltoall.py::TestAllToAllWorldSize8::test_alltoall_partial_group_unsupported_topology.
    """

    @property
    def world_size(self) -> int:
        return 8

    @pytest.mark.parametrize("partial_group", [[0, 1, 2, 3], [1, 3]])
    def test_partial_group_alltoall_base(self, partial_group):
        """Test alltoall_base within a partial group, aligned and unaligned start ranks."""
        self.distributed_tester.run_test(
            run_partial_group_alltoall_base_test,
            partial_group=partial_group,
        )

    def test_alltoall_base_strided_group(self):
        """Pairs of rank i and rank i + 4 across two chips.

        On trn2 the fabric links rank i to rank i + 4 only. The pair works only when every
        rank in the world takes part in some group, so all four pairs run at once.
        """
        self.distributed_tester.run_test(run_strided_group_alltoall_base_test, stride=4)
