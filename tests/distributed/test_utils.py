"""Unit tests for torch_neuronx.distributed.utils module."""

import pytest
import torch

from torch_neuronx.distributed.utils import (
    create_global_replica_group,
    get_free_port,
    get_reduce_type,
)


class TestGetReduceType:
    """Test cases for get_reduce_type function."""

    def test_sum_operation(self):
        """Test SUM reduce operation."""
        import torch.distributed as dist

        assert get_reduce_type(dist.ReduceOp.SUM) == "SUM"

    def test_product_operation(self):
        """Test PRODUCT reduce operation."""
        import torch.distributed as dist

        assert get_reduce_type(dist.ReduceOp.PRODUCT) == "PRODUCT"

    def test_min_operation(self):
        """Test MIN reduce operation."""
        import torch.distributed as dist

        assert get_reduce_type(dist.ReduceOp.MIN) == "MIN"

    def test_max_operation(self):
        """Test MAX reduce operation."""
        import torch.distributed as dist

        assert get_reduce_type(dist.ReduceOp.MAX) == "MAX"

    def test_avg_operation(self):
        """Test AVG reduce operation."""
        import torch.distributed as dist

        assert get_reduce_type(dist.ReduceOp.AVG) == "AVG"

    def test_band_operation(self):
        """Test BAND reduce operation."""
        import torch.distributed as dist

        assert get_reduce_type(dist.ReduceOp.BAND) == "BAND"

    def test_bor_operation(self):
        """Test BOR reduce operation."""
        import torch.distributed as dist

        assert get_reduce_type(dist.ReduceOp.BOR) == "BOR"

    def test_bxor_operation(self):
        """Test BXOR reduce operation."""
        import torch.distributed as dist

        assert get_reduce_type(dist.ReduceOp.BXOR) == "BXOR"

    def test_unsupported_operation(self):
        """Test unsupported reduce operation raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported reduce operation"):
            get_reduce_type("INVALID_OP")


class TestCreateGlobalReplicaGroup:
    """Test cases for create_global_replica_group function."""

    def test_basic_group_creation(self):
        """Test basic replica group creation."""
        world_size = 8
        current_group_ranks = [0, 1, 2, 3]
        result = create_global_replica_group(world_size, current_group_ranks)

        assert len(result) == 2
        assert result[0] == [0, 1, 2, 3]
        assert result[1] == [4, 5, 6, 7]

    def test_single_group(self):
        """Test when current group equals world size."""
        world_size = 4
        current_group_ranks = [0, 1, 2, 3]
        result = create_global_replica_group(world_size, current_group_ranks)

        assert len(result) == 1
        assert result[0] == [0, 1, 2, 3]

    def test_multiple_groups(self):
        """Test creation of multiple replica groups."""
        world_size = 12
        current_group_ranks = [0, 1, 2]
        result = create_global_replica_group(world_size, current_group_ranks)

        assert len(result) == 4
        assert result[0] == [0, 1, 2]
        assert result[1] == [3, 4, 5]
        assert result[2] == [6, 7, 8]
        assert result[3] == [9, 10, 11]

    def test_non_zero_start_ranks(self):
        """Test with non-zero starting ranks."""
        world_size = 8
        current_group_ranks = [2, 3, 4, 5]
        result = create_global_replica_group(world_size, current_group_ranks)

        assert len(result) == 2
        assert result[0] == [2, 3, 4, 5]
        assert result[1] == [0, 1, 6, 7]


class TestFindScatterDim:
    """Test cases for _find_scatter_dim helper function."""

    def test_same_rank_shapes(self):
        """Test finding scatter dim with same rank shapes."""
        from torch_neuronx.distributed.utils import get_reduce_scatter_inputs_outputs

        input_tensor = torch.zeros(4, 8)
        output_tensor = torch.zeros(2, 8)
        world_size = 2

        input_gen, output_gen = get_reduce_scatter_inputs_outputs(
            input_tensor, output_tensor, world_size
        )
        # Verify generators work by consuming them
        assert list(input_gen) and list(output_gen)

    def test_rank_mismatch_2d_to_1d(self):
        """Test finding scatter dim with rank mismatch: 2D input to 1D output."""
        from torch_neuronx.distributed.utils import get_reduce_scatter_inputs_outputs

        input_tensor = torch.zeros(2, 2)
        output_tensor = torch.zeros(2)
        world_size = 2

        input_gen, output_gen = get_reduce_scatter_inputs_outputs(
            input_tensor, output_tensor, world_size
        )
        assert list(input_gen) and list(output_gen)

    def test_rank_mismatch_3d_to_2d(self):
        """Test finding scatter dim with rank mismatch: 3D input to 2D output."""
        from torch_neuronx.distributed.utils import get_reduce_scatter_inputs_outputs

        input_tensor = torch.zeros(2, 4, 8)
        output_tensor = torch.zeros(4, 8)
        world_size = 2

        input_gen, output_gen = get_reduce_scatter_inputs_outputs(
            input_tensor, output_tensor, world_size
        )
        assert list(input_gen) and list(output_gen)

    def test_identical_shapes(self):
        """Test finding scatter dim with identical shapes."""
        from torch_neuronx.distributed.utils import get_reduce_scatter_inputs_outputs

        input_tensor = torch.zeros(4, 4)
        output_tensor = torch.zeros(4, 4)
        world_size = 1

        input_gen, output_gen = get_reduce_scatter_inputs_outputs(
            input_tensor, output_tensor, world_size
        )
        assert list(input_gen) and list(output_gen)


class TestGetReduceScatterInputsOutputs:
    """Test cases for get_reduce_scatter_inputs_outputs function."""

    def test_small_tensor_no_bucketing(self):
        """Test with small tensor that doesn't require bucketing."""
        from torch_neuronx.distributed.utils import get_reduce_scatter_inputs_outputs

        input_tensor = torch.zeros(4, 8)
        output_tensor = torch.zeros(2, 8)
        world_size = 2

        input_gen, output_gen = get_reduce_scatter_inputs_outputs(
            input_tensor, output_tensor, world_size
        )

        input_list = list(input_gen)
        output_list = list(output_gen)
        assert len(input_list) == len(output_list)
        assert len(input_list) >= 1

    def test_input_breaker_calculation(self):
        """Test that generators produce valid tensors."""
        from torch_neuronx.distributed.utils import get_reduce_scatter_inputs_outputs

        input_tensor = torch.zeros(8, 16)
        output_tensor = torch.zeros(4, 16)
        world_size = 2

        input_gen, output_gen = get_reduce_scatter_inputs_outputs(
            input_tensor, output_tensor, world_size
        )

        input_list = list(input_gen)
        output_list = list(output_gen)
        assert len(input_list) == len(output_list)
        assert all(isinstance(t, torch.Tensor) for t in input_list)
        assert all(isinstance(t, torch.Tensor) for t in output_list)

    def test_invalid_world_size(self):
        """Test that invalid world_size raises ValueError."""
        from torch_neuronx.distributed.utils import get_reduce_scatter_inputs_outputs

        input_tensor = torch.zeros(4, 8)
        output_tensor = torch.zeros(2, 8)

        with pytest.raises(ValueError, match="world_size must be positive"):
            get_reduce_scatter_inputs_outputs(input_tensor, output_tensor, 0)

    def test_empty_input_tensor(self):
        """Test that empty input tensor raises ValueError."""
        from torch_neuronx.distributed.utils import get_reduce_scatter_inputs_outputs

        input_tensor = torch.zeros(0)
        output_tensor = torch.zeros(2, 8)

        with pytest.raises(ValueError, match="input_tensor cannot be empty"):
            get_reduce_scatter_inputs_outputs(input_tensor, output_tensor, 2)

    def test_empty_output_tensor(self):
        """Test that empty output tensor raises ValueError."""
        from torch_neuronx.distributed.utils import get_reduce_scatter_inputs_outputs

        input_tensor = torch.zeros(4, 8)
        output_tensor = torch.zeros(0)

        with pytest.raises(ValueError, match="output tensor cannot be empty"):
            get_reduce_scatter_inputs_outputs(input_tensor, output_tensor, 2)


class TestCollectiveBucketsizeEnvVar:
    """Test cases for COLLECTIVE_BUCKETSIZE_IN_MB environment variable."""

    def test_default_bucketsize(self):
        """Test default bucket size is 512MB."""
        import importlib

        import torch_neuronx.distributed.utils as utils_module

        # Reload module to get default value
        importlib.reload(utils_module)
        assert utils_module._COLLECTIVE_BUCKETSIZE_BYTES == 512 * 1024 * 1024

    def test_custom_bucketsize(self, monkeypatch):
        """Test custom bucket size from environment variable."""
        import importlib

        import torch_neuronx.distributed.utils as utils_module

        # Set custom value
        monkeypatch.setenv("COLLECTIVE_BUCKETSIZE_IN_MB", "1024")
        importlib.reload(utils_module)

        assert utils_module._COLLECTIVE_BUCKETSIZE_BYTES == 1024 * 1024 * 1024

        # Cleanup
        monkeypatch.delenv("COLLECTIVE_BUCKETSIZE_IN_MB")
        importlib.reload(utils_module)


class TestReconstructReduceScatterOutput:
    """Test cases for reconstruct_reduce_scatter_output function."""

    def test_basic_reconstruction(self):
        """Test basic output reconstruction."""
        from torch_neuronx.distributed.utils import reconstruct_reduce_scatter_output

        # Create a simple case: world_size=2, scatter_dim=0
        output_tensor = torch.zeros(4, 8)
        input_breaker = 2
        scatter_dim = 0

        # Simulate concatenated output from bucketed operations
        output_tensor_cat = torch.ones(4, 8).flatten()

        result = reconstruct_reduce_scatter_output(
            output_tensor_cat, output_tensor, input_breaker, scatter_dim
        )

        assert len(result) == 1
        assert result[0].shape == output_tensor.shape

    def test_size_mismatch_raises_error(self):
        """Test that size mismatch raises ValueError."""
        from torch_neuronx.distributed.utils import reconstruct_reduce_scatter_output

        output_tensor = torch.zeros(4, 8)
        input_breaker = 2
        scatter_dim = 0

        # Create wrong size concatenated output
        output_tensor_cat = torch.ones(16)  # Wrong size, should be 32

        with pytest.raises(ValueError, match="Size mismatch"):
            reconstruct_reduce_scatter_output(
                output_tensor_cat, output_tensor, input_breaker, scatter_dim
            )


class TestGetFreePort:
    """Test cases for get_free_port function."""

    def test_returns_valid_port(self):
        """Test that get_free_port returns a valid port number."""

        port = get_free_port()
        assert isinstance(port, str)
        assert int(port) > 0
        assert int(port) < 65536


class TestExecuteWithXlaOpCheck:
    """Test cases for _execute_with_xla_op_check function."""

    def test_can_handle_true_tuple_result(self):
        """Test execution when can_handle returns (True, None)."""
        from torch_neuronx.distributed.utils import _execute_with_xla_op_check

        class MockXlaOp:
            def can_handle(self, *args, **kwargs):
                return True, None

        executed = [False]

        def core_fn():
            executed[0] = True

        def fallback_fn():
            pass

        xla_op = MockXlaOp()
        _execute_with_xla_op_check(xla_op, "test_op", core_fn, fallback_fn, [], {})

        assert executed[0]

    def test_can_handle_true_bool_result(self):
        """Test execution when can_handle returns just True."""
        from torch_neuronx.distributed.utils import _execute_with_xla_op_check

        class MockXlaOp:
            def can_handle(self, *args, **kwargs):
                return True

        executed = [False]

        def core_fn():
            executed[0] = True

        def fallback_fn():
            pass

        xla_op = MockXlaOp()
        _execute_with_xla_op_check(xla_op, "test_op", core_fn, fallback_fn, [], {})

        assert executed[0]

    def test_can_handle_false_raises_error(self):
        """Test that can_handle False raises RuntimeError."""
        from torch_neuronx.distributed.utils import _execute_with_xla_op_check

        class MockXlaOp:
            def can_handle(self, *args, **kwargs):
                return False, "Test error message"

        def core_fn():
            pass

        def fallback_fn():
            pass

        xla_op = MockXlaOp()
        with pytest.raises(RuntimeError, match="No implementation could handle operation"):
            _execute_with_xla_op_check(xla_op, "test_op", core_fn, fallback_fn, [], {})


class TestCalculateChunkingParams:
    """Test cases for _calculate_chunking_params function."""

    def test_evenly_divisible(self):
        """Test when elements divide evenly."""
        from torch_neuronx.distributed.utils import _calculate_chunking_params

        num_chunks, remainder = _calculate_chunking_params(100, 25)
        assert num_chunks == 4
        assert remainder == 0

    def test_with_remainder(self):
        """Test when elements don't divide evenly."""
        from torch_neuronx.distributed.utils import _calculate_chunking_params

        num_chunks, remainder = _calculate_chunking_params(100, 30)
        assert num_chunks == 3
        assert remainder == 10

    def test_zero_pick_per_block(self):
        """Test when num_elem_to_pick_per_block is 0."""
        from torch_neuronx.distributed.utils import _calculate_chunking_params

        num_chunks, remainder = _calculate_chunking_params(100, 0)
        assert num_chunks == 0
        assert remainder == 0


class TestFlattenInputForScatter:
    """Test cases for _flatten_input_for_scatter function."""

    def test_scatter_dim_zero(self):
        """Test flattening when scatter_dim is 0."""
        from torch_neuronx.distributed.utils import _flatten_input_for_scatter

        input_tensor = torch.randn(4, 8)
        result = _flatten_input_for_scatter(input_tensor, 0, 2, 2)

        assert result.dim() == 1
        assert result.numel() == input_tensor.numel()

    def test_scatter_dim_non_zero(self):
        """Test flattening when scatter_dim is not 0."""
        from torch_neuronx.distributed.utils import _flatten_input_for_scatter

        input_tensor = torch.randn(4, 8, 16)
        result = _flatten_input_for_scatter(input_tensor, 1, 2, 4)

        assert result.dim() == 1
        assert result.numel() == input_tensor.numel()
