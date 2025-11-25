import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)


class TestCopyFromAndResize:
    """Test cases for the _copy_from_and_resize operation."""

    def test_copy_from_cpu_empty_dst(self):
        """Test copying from CPU to empty Neuron tensor (should resize dst)."""
        with track_neuron_ops():
            # Create source CPU tensor
            src = torch.randn(10, 20)

            # Create empty destination Neuron tensor
            device = torch.device("neuron")
            dst = torch.empty(0, device=device)

            # Perform copy_from_and_resize
            result = torch._copy_from_and_resize(src, dst)

            # Verify result
            assert result is dst  # Should return dst tensor
            assert dst.shape == src.shape  # dst should be resized to match src
            assert_op_runs_on_neuron("aten::resize_")
            assert_op_runs_on_neuron("aten::empty")

            # Verify data was copied correctly
            cpu_dst = dst.cpu()
            assert torch.allclose(cpu_dst, src)

    def test_copy_from_cpu_same_size(self):
        """Test copying from CPU to Neuron tensor of same size."""
        with track_neuron_ops():
            # Create source CPU tensor
            src = torch.randn(5, 10)

            # Create destination Neuron tensor with same size
            device = torch.device("neuron")
            dst = torch.empty(5, 10, device=device)

            # Perform copy_from_and_resize
            result = torch._copy_from_and_resize(src, dst)

            # Verify result
            assert result is dst
            assert dst.shape == src.shape
            # No resize_ given dst numel is not 0
            assert_op_runs_on_neuron("aten::empty")

            # Verify data was copied correctly
            cpu_dst = dst.cpu()
            assert torch.allclose(cpu_dst, src)

    @assert_raises(RuntimeError, match="same size")
    def test_copy_from_cpu_different_size_error(self):
        """Test that copying to non-empty tensor with different size raises error."""
        # Create source CPU tensor
        src = torch.randn(10, 20)

        # Create destination Neuron tensor with different size
        device = torch.device("neuron")
        dst = torch.empty(5, 5, device=device)

        # Should raise error for size mismatch
        torch._copy_from_and_resize(src, dst)

    @assert_raises(RuntimeError, match="only support copy from cpu tensor to")
    def test_copy_from_neuron_to_neuron_error(self):
        """Test that Neuron to Neuron copy is not supported."""
        device = torch.device("neuron")

        # Create source and destination Neuron tensors
        src = torch.empty(10, 10, device=device)
        dst = torch.empty(0, device=device)

        # Should raise error - only CPU to Neuron is supported
        torch._copy_from_and_resize(src, dst)

    @assert_raises(RuntimeError, match="only support copy from cpu tensor to")
    def test_copy_neuron_to_cpu_error(self):
        """Test that Neuron to CPU copy is not supported."""
        device = torch.device("neuron")

        # Create source Neuron and destination CPU tensors
        src = torch.empty(10, 10, device=device)
        dst = torch.empty(0)  # CPU tensor

        # Should raise error - only CPU to Neuron is supported
        torch._copy_from_and_resize(src, dst)

    @pytest.mark.parametrize(
        "shape",
        [
            (100,),  # 1D
            (50, 50),  # 2D
            (10, 20, 30),  # 3D
            (5, 10, 10, 5),  # 4D
        ],
    )
    def test_various_shapes(self, shape):
        """Test copy_from_and_resize with various tensor shapes."""
        with track_neuron_ops():
            # Create source CPU tensor
            src = torch.randn(*shape)

            # Create empty destination Neuron tensor
            device = torch.device("neuron")
            dst = torch.empty(0, device=device)

            # Perform copy_from_and_resize
            result = torch._copy_from_and_resize(src, dst)

            # Verify result
            assert result is dst
            assert dst.shape == src.shape
            assert_op_runs_on_neuron("aten::resize_")
            assert_op_runs_on_neuron("aten::empty")

            # Verify data was copied correctly
            cpu_dst = dst.cpu()
            assert torch.allclose(cpu_dst, src)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32, torch.int64])
    def test_different_dtypes(self, dtype):
        with track_neuron_ops():
            """Test copy_from_and_resize with different data types."""
            # Create source CPU tensor
            if dtype in [torch.int32, torch.int64]:
                src = torch.randint(0, 100, (10, 20), dtype=dtype)
            else:
                src = torch.randn(10, 20, dtype=dtype)

            # Create empty destination Neuron tensor with same dtype
            device = torch.device("neuron")
            dst = torch.empty(0, device=device, dtype=dtype)

            # Perform copy_from_and_resize
            result = torch._copy_from_and_resize(src, dst)

            # Verify result
            assert result is dst
            assert dst.shape == src.shape
            assert dst.dtype == dtype
            assert_op_runs_on_neuron("aten::resize_")
            assert_op_runs_on_neuron("aten::empty")

            # Verify data was copied correctly
            cpu_dst = dst.cpu()
            if dtype in [torch.float32, torch.float16]:
                assert torch.allclose(cpu_dst, src)
            else:
                assert torch.equal(cpu_dst, src)

    def test_zero_size_tensor(self):
        with track_neuron_ops():
            """Test copy_from_and_resize with zero-size tensors."""
            # Create zero-size source CPU tensor
            src = torch.empty(0, 10)

            # Create empty destination Neuron tensor
            device = torch.device("neuron")
            dst = torch.empty(0, device=device)

            # Perform copy_from_and_resize
            result = torch._copy_from_and_resize(src, dst)

            # Verify result
            assert result is dst
            assert dst.shape == src.shape
            assert dst.numel() == 0
            assert_op_runs_on_neuron("aten::resize_")
            assert_op_runs_on_neuron("aten::empty")
