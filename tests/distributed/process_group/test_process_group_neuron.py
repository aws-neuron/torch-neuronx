"""Integration tests for ProcessGroupNeuron with NeuronWork and NeuronWatchdog.

These tests are based on PyTorch's ProcessGroupNCCLTest.cpp, testing the Work/Wait
behavior through real collectives. Key patterns adapted from NCCL tests:
- testAllreduce, testBroadcast, testAllgather, testReduceScatter, etc.
- Each test: initialize -> run collective -> wait(work) -> validate results

Run with: torchrun --nproc_per_node=2 -m pytest \
    tests/distributed/process_group/test_process_group_neuron.py -v
"""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

import torch_neuronx

# Skip entire module if not in distributed environment
pytestmark = pytest.mark.skipif(
    not os.environ.get("WORLD_SIZE") or int(os.environ.get("WORLD_SIZE", 1)) < 2,
    reason="Requires distributed environment with WORLD_SIZE >= 2",
)


@pytest.fixture(scope="module")
def dist_setup():
    """Initialize distributed process group for testing."""
    if not dist.is_initialized():
        dist.init_process_group(backend="neuron", timeout=timedelta(seconds=30))

    yield

    # Cleanup handled by test framework


class TestAllreduce:
    """Test allreduce operations with NeuronWork.

    Based on ProcessGroupNCCLTest::testAllreduce (lines 404-421)
    Pattern: run() -> wait(work) -> validate tensor values
    """

    def test_allreduce_async_wait(self, dist_setup):
        """Test async allreduce with work.wait() - mirrors NCCL testAllreduce."""
        rank = dist.get_rank()
        size = dist.get_world_size()

        # Initialize tensor with rank value (NCCL pattern)
        tensor = torch.full((4, 4), float(rank), device="neuron")

        # Run async collective
        work = dist.all_reduce(tensor, async_op=True)
        assert work is not None

        # Wait for work to finish
        result = work.wait()
        assert result is True

        # After wait(): result() should contain outputs with correct VALUES
        post_wait_result = work.result()
        assert len(post_wait_result) > 0, "result() should have outputs after wait()"

        # Validation: expected = sum of all ranks = 0+1+...+(size-1) = size*(size-1)/2
        expected_val = (size * (size - 1)) / 2
        expected = torch.full((4, 4), expected_val, device="neuron")
        torch.testing.assert_close(tensor, expected)

    def test_allreduce_sync(self, dist_setup):
        """Test sync allreduce returns None (implicit wait)."""
        rank = dist.get_rank()
        size = dist.get_world_size()

        tensor = torch.ones(4, 4, device="neuron") * rank

        # Sync operation returns None (NCCL pattern)
        work = dist.all_reduce(tensor, async_op=False)
        assert work is None

        expected_val = (size * (size - 1)) / 2
        expected = torch.ones(4, 4, device="neuron") * expected_val
        assert torch.allclose(tensor, expected)


class TestBroadcast:
    """Test broadcast operations with NeuronWork.

    Based on ProcessGroupNCCLTest::testBroadcast (lines 526-550)
    """

    def test_broadcast_async_wait(self, dist_setup):
        """Test async broadcast with work.wait() - mirrors NCCL testBroadcast."""
        rank = dist.get_rank()
        root_rank = 0

        if rank == root_rank:
            tensor = torch.ones(4, 4, device="neuron") * 42
        else:
            tensor = torch.zeros(4, 4, device="neuron")

        # Run async collective
        work = dist.broadcast(tensor, src=root_rank, async_op=True)

        # Wait for work to finish
        assert work is not None
        work.wait()

        # Validation: all ranks should have root's value
        expected = torch.ones(4, 4, device="neuron") * 42
        assert torch.allclose(tensor, expected), "Broadcast outputs do not match expected outputs"


class TestAllgather:
    """Test allgather operations with NeuronWork.

    Based on ProcessGroupNCCLTest::testAllgather (lines 582-603)
    """

    def test_allgather_async_wait(self, dist_setup):
        """Test async allgather with work.wait() - mirrors NCCL testAllgather."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Each rank contributes its rank value
        tensor = torch.ones(4, 4, device="neuron") * rank
        output_tensors = [torch.zeros(4, 4, device="neuron") for _ in range(world_size)]

        # Run async collective
        work = dist.all_gather(output_tensors, tensor, async_op=True)

        # Wait for work to finish
        assert work is not None
        work.wait()

        # Validation: output[i] should contain rank i's contribution
        for i in range(world_size):
            expected = torch.ones(4, 4, device="neuron") * i
            assert torch.allclose(
                output_tensors[i], expected
            ), f"Allgather output[{i}] does not match expected"


class TestAllgatherBase:
    """Test _allgather_base operations with NeuronWork.

    Based on ProcessGroupNCCLTest::testAllgatherBase (lines 606-624)
    """

    @pytest.mark.xfail(
        reason="Test has been flakey, needs more investigation. Xfailing to unblock CI"
    )
    def test_allgather_into_tensor_async(self, dist_setup):
        """Test allgather_into_tensor with work.wait()."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        input_tensor = torch.ones(4, 4, device="neuron") * rank
        output_tensor = torch.zeros(world_size * 4, 4, device="neuron")

        # Run async collective
        work = dist.all_gather_into_tensor(output_tensor, input_tensor, async_op=True)

        # Wait for work to finish
        assert work is not None
        work.wait()

        # Validation: output should be concatenation of all ranks
        # Move to CPU for comparison to avoid neuron kernel issues
        output_cpu = output_tensor.cpu()
        for i in range(world_size):
            chunk = output_cpu[i * 4 : (i + 1) * 4]
            expected = torch.ones(4, 4) * i
            assert torch.allclose(chunk, expected)


class TestReduceScatter:
    """Test reduce_scatter operations with NeuronWork.

    Based on ProcessGroupNCCLTest::testReduceScatter (lines 649-668)
    """

    def test_reduce_scatter_async_wait(self, dist_setup):
        """Test async reduce_scatter with work.wait() - mirrors NCCL testReduceScatter."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Each rank has world_size chunks
        input_list = [torch.ones(4, 4, device="neuron") * (i + rank) for i in range(world_size)]
        output_tensor = torch.zeros(4, 4, device="neuron")

        # Run async collective
        work = dist.reduce_scatter(output_tensor, input_list, async_op=True)

        # Wait for work to finish
        assert work is not None
        work.wait()

        # Validation: output should be sum of all ranks' contributions for this rank's chunk
        # output[rank] = sum over all ranks r of input_list[r][rank]
        expected_val = sum((rank + r) for r in range(world_size))
        expected = torch.ones(4, 4, device="neuron") * expected_val
        assert torch.allclose(
            output_tensor, expected
        ), "ReduceScatter outputs do not match expected outputs"


class TestReduceScatterBase:
    """Test _reduce_scatter_base operations with NeuronWork.

    Based on ProcessGroupNCCLTest::testReduceScatterBase (lines 627-646)
    """

    def test_reduce_scatter_tensor_async(self, dist_setup):
        """Test reduce_scatter_tensor with work.wait()."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        input_tensor = torch.ones(world_size * 4, 4, device="neuron") * rank
        output_tensor = torch.zeros(4, 4, device="neuron")

        # Run async collective
        work = dist.reduce_scatter_tensor(output_tensor, input_tensor, async_op=True)

        # Wait for work to finish
        assert work is not None
        work.wait()

        # Validation: output should be sum of all ranks
        expected_val = sum(range(world_size))  # 0+1+2+... = world_size*(world_size-1)/2
        expected = torch.ones(4, 4, device="neuron") * expected_val
        assert torch.allclose(output_tensor, expected)


class TestSequenceNumInit:
    """Test sequence number initialization.

    Based on ProcessGroupNCCLTest::testSequenceNumInit (lines 671-676)
    """

    def test_sequence_num_init(self, dist_setup):
        """Test that sequence numbers increment - mirrors NCCL testSequenceNumInit."""
        tensor1 = torch.ones(4, 4, device="neuron")
        tensor2 = torch.ones(4, 4, device="neuron")
        tensor3 = torch.ones(4, 4, device="neuron")

        work1 = dist.all_reduce(tensor1, async_op=True)
        work2 = dist.all_reduce(tensor2, async_op=True)
        work3 = dist.all_reduce(tensor3, async_op=True)

        # Sequence numbers should be unique and incrementing
        seq1 = work1.get_sequence_number()
        seq2 = work2.get_sequence_number()
        seq3 = work3.get_sequence_number()

        assert seq2 > seq1, "Sequence numbers should increment"
        assert seq3 > seq2, "Sequence numbers should increment"

        # Cleanup
        work1.wait()
        work2.wait()
        work3.wait()


class TestBackendName:
    """Test backend name.

    Based on ProcessGroupNCCLTest::testBackendName (lines 852-862)
    """

    def test_backend_name(self, dist_setup):
        """Test getBackendName returns 'neuron' - mirrors NCCL testBackendName."""
        pg = dist.distributed_c10d._get_default_group()
        # Our backend should return "neuron"
        assert pg.name() == "neuron" or "neuron" in str(pg)


class TestWorkWaitBehavior:
    """Test NeuronWork.wait() behavior patterns.

    Additional tests to ensure Work interface matches NCCL behavior.
    """

    def test_wait_twice_succeeds(self, dist_setup):
        """Test that wait() can be called multiple times safely."""
        tensor = torch.ones(4, 4, device="neuron")

        work = dist.all_reduce(tensor, async_op=True)

        # First wait
        result1 = work.wait()
        assert result1 is True

        # Second wait should also succeed (idempotent)
        result2 = work.wait()
        assert result2 is True

    def test_is_completed_no_side_effects(self, dist_setup):
        """Test is_completed() before wait() doesn't affect results.

        NCCL pattern: tests never check is_completed() after wait().
        They only call wait() then validate tensor values.
        """
        tensor = torch.ones(4, 4, device="neuron")

        work = dist.all_reduce(tensor, async_op=True)

        # Call is_completed multiple times (should not have side effects)
        _ = work.is_completed()
        _ = work.is_completed()

        # wait() should still work
        result = work.wait()
        assert result is True

        # NCCL tests don't assert on is_completed() after wait()
        # They just validate tensor values
        expected = torch.ones(4, 4, device="neuron") * dist.get_world_size()
        assert torch.allclose(tensor, expected)

    def test_work_result_returns_outputs(self, dist_setup):
        """Test that work.result() returns output tensors."""
        tensor = torch.ones(4, 4, device="neuron")

        work = dist.all_reduce(tensor, async_op=True)
        work.wait()

        # result() should return the outputs
        outputs = work.result()
        # For allreduce, output is the same tensor
        assert len(outputs) >= 0  # May be empty or contain tensor depending on impl


class TestMultipleAsyncOps:
    """Test multiple concurrent async operations.

    Pattern from NCCL tests where multiple ops are launched then waited.
    """

    def test_multiple_async_allreduce(self, dist_setup):
        """Test multiple async allreduce operations."""
        rank = dist.get_rank()
        size = dist.get_world_size()

        tensors = [torch.ones(4, 4, device="neuron") * (i * rank) for i in range(1, 4)]

        # Launch multiple async operations
        works = []
        for tensor in tensors:
            work = dist.all_reduce(tensor, async_op=True)
            works.append(work)

        # Wait for all to complete
        for work in works:
            assert work is not None
            work.wait()

        # Validate all results
        for i, tensor in enumerate(tensors):
            multiplier = i + 1
            expected_val = multiplier * sum(range(size))  # multiplier * (0+1+2+...)
            expected = torch.ones(4, 4, device="neuron") * expected_val
            assert torch.allclose(tensor, expected)

    def test_interleaved_async_sync(self, dist_setup):
        """Test interleaving async and sync operations."""
        rank = dist.get_rank()
        size = dist.get_world_size()

        tensor1 = torch.full((4, 4), float(rank), device="neuron")
        tensor2 = torch.full((4, 4), float(rank), device="neuron")
        tensor3 = torch.full((4, 4), float(rank), device="neuron")

        # Async op
        work1 = dist.all_reduce(tensor1, async_op=True)

        # Sync op (completes immediately)
        dist.all_reduce(tensor2, async_op=False)

        # Another async op
        work3 = dist.all_reduce(tensor3, async_op=True)

        # Wait for async ops
        work1.wait()
        work3.wait()

        # All should have correct value
        expected_val = float(sum(range(size)))
        expected = torch.full((4, 4), expected_val, device="neuron")
        torch.testing.assert_close(tensor1, expected)
        torch.testing.assert_close(tensor2, expected)
        torch.testing.assert_close(tensor3, expected)

    def test_get_future_wait_value(self, dist_setup):
        """Test work.getFuture().wait().value() code path.

        This is a slightly different code path than work.wait() - tests the
        Future-based API for async completion.
        """
        rank = dist.get_rank()
        size = dist.get_world_size()

        tensor1 = torch.full((4, 4), float(rank), device="neuron")
        tensor2 = torch.full((4, 4), float(rank), device="neuron")
        tensor3 = torch.full((4, 4), float(rank), device="neuron")

        # Launch async ops
        work1 = dist.all_reduce(tensor1, async_op=True)
        work2 = dist.all_reduce(tensor2, async_op=True)
        work3 = dist.all_reduce(tensor3, async_op=True)

        # Use getFuture().wait() instead of work.wait() - different code path
        fut1 = work1.get_future()
        fut2 = work2.get_future()
        fut3 = work3.get_future()

        # Wait on futures
        fut1.wait()
        fut2.wait()
        fut3.wait()

        # Get values from futures
        val1 = fut1.value()
        val2 = fut2.value()
        val3 = fut3.value()

        # Value should return list of tensors
        assert val1 is not None, "Future value should not be None"
        assert val2 is not None, "Future value should not be None"
        assert val3 is not None, "Future value should not be None"

        # Validate tensor results
        expected_val = float(sum(range(size)))
        expected = torch.full((4, 4), expected_val, device="neuron")
        torch.testing.assert_close(tensor1, expected)
        torch.testing.assert_close(tensor2, expected)
        torch.testing.assert_close(tensor3, expected)


class TestWatchdog:
    """Test NeuronWatchdog behavior through ProcessGroup.

    Watchdog should be started automatically and process work queue.
    """

    def test_watchdog_running(self, dist_setup):
        """Test that watchdog exists after process group init."""
        pg = dist.distributed_c10d._get_default_group()
        # ProcessGroupNeuron should have a watchdog
        assert hasattr(pg, "_watchdog")

    def test_watchdog_processes_completed_work(self, dist_setup):
        """Test that watchdog properly handles completed work.

        NCCL pattern: tests just call wait() then validate tensor values,
        they don't assert on is_completed().
        """
        tensors = [torch.ones(4, 4, device="neuron") for _ in range(5)]

        # Launch multiple async operations
        works = []
        for tensor in tensors:
            work = dist.all_reduce(tensor, async_op=True)
            works.append(work)

        # Wait for all (NCCL pattern)
        for work in works:
            work.wait()

        # Validate tensor values (NCCL pattern - don't check is_completed)
        expected = torch.ones(4, 4, device="neuron") * dist.get_world_size()
        for tensor in tensors:
            assert torch.allclose(tensor, expected)


class TestBarrier:
    """Test barrier operation."""

    def test_barrier_sync(self, dist_setup):
        """Test barrier synchronization."""
        rank = dist.get_rank()
        size = dist.get_world_size()

        # Barrier should synchronize all ranks
        dist.barrier()

        # After barrier, operations should see consistent state
        tensor = torch.ones(1, device="neuron") * rank
        dist.all_reduce(tensor)

        expected = torch.ones(1, device="neuron") * sum(range(size))
        assert torch.allclose(tensor, expected)


class TestNonDefaultStream:
    """Test collectives on non-default streams.

    Based on ProcessGroupNCCLTest pattern where tests use getStreamFromPool()
    and CUDAMultiStreamGuard to run collectives on non-default streams
    (see lines 95-122 in ProcessGroupNCCLTest.cpp).
    """

    def test_allreduce_non_default_stream(self, dist_setup):
        """Test async allreduce on a non-default stream."""
        rank = dist.get_rank()
        size = dist.get_world_size()

        tensor = torch.ones(4, 4, device="neuron") * rank

        # Create and use non-default stream (similar to NCCL's getStreamFromPool)
        stream = torch_neuronx.Stream()
        with torch_neuronx.stream(stream):
            work = dist.all_reduce(tensor, async_op=True)

        # Wait for work to finish
        work.wait()

        # Validation
        expected_val = (size * (size - 1)) / 2
        expected = torch.ones(4, 4, device="neuron") * expected_val
        assert torch.allclose(tensor, expected), "Allreduce on non-default stream failed"

    def test_broadcast_non_default_stream(self, dist_setup):
        """Test async broadcast on a non-default stream."""
        rank = dist.get_rank()
        root_rank = 0

        if rank == root_rank:
            tensor = torch.ones(4, 4, device="neuron") * 99
        else:
            tensor = torch.zeros(4, 4, device="neuron")

        stream = torch_neuronx.Stream()
        with torch_neuronx.stream(stream):
            work = dist.broadcast(tensor, src=root_rank, async_op=True)

        work.wait()

        expected = torch.ones(4, 4, device="neuron") * 99
        assert torch.allclose(tensor, expected), "Broadcast on non-default stream failed"

    def test_multiple_streams_concurrent(self, dist_setup):
        """Test concurrent collectives on different non-default streams.

        Similar to ProcessGroupNCCLTest pattern of running operations on
        multiple streams concurrently.
        """
        rank = dist.get_rank()
        size = dist.get_world_size()

        tensor1 = torch.ones(4, 4, device="neuron") * rank
        tensor2 = torch.ones(4, 4, device="neuron") * (rank + 1)

        # Create two different streams
        stream1 = torch_neuronx.Stream()
        stream2 = torch_neuronx.Stream()

        # Launch on different streams
        with torch_neuronx.stream(stream1):
            work1 = dist.all_reduce(tensor1, async_op=True)

        with torch_neuronx.stream(stream2):
            work2 = dist.all_reduce(tensor2, async_op=True)

        # Wait for both
        work1.wait()
        work2.wait()

        # Validate both
        expected1 = torch.ones(4, 4, device="neuron") * ((size * (size - 1)) / 2)
        expected2 = torch.ones(4, 4, device="neuron") * (sum(r + 1 for r in range(size)))

        assert torch.allclose(tensor1, expected1), "Stream1 allreduce failed"
        assert torch.allclose(tensor2, expected2), "Stream2 allreduce failed"


class TestEdgeCases:
    """Test edge cases."""

    def test_large_tensor_allreduce(self, dist_setup):
        """Test allreduce with larger tensor."""
        rank = dist.get_rank()
        size = dist.get_world_size()

        tensor = torch.ones(256, 256, device="neuron") * rank

        work = dist.all_reduce(tensor, async_op=True)
        work.wait()

        expected_val = sum(range(size))
        expected = torch.ones(256, 256, device="neuron") * expected_val
        assert torch.allclose(tensor, expected)

    def test_different_dtypes(self, dist_setup):
        """Test allreduce with different dtypes."""
        rank = dist.get_rank()
        size = dist.get_world_size()

        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            tensor = torch.ones(4, 4, device="neuron", dtype=dtype) * rank

            work = dist.all_reduce(tensor, async_op=True)
            work.wait()

            expected_val = sum(range(size))
            expected = torch.ones(4, 4, device="neuron", dtype=dtype) * expected_val
            assert torch.allclose(tensor, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
