import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


class TestResize:
    """Test cases for the resize operation on Neuron tensors."""

    def test_basic_resize(self):
        """Test basic resize operation with different sizes."""
        # Create a tensor on neuron device
        with track_neuron_ops():
            device = torch.device("neuron")
            x = torch.empty(10, 20, device=device)

            # Resize to larger size
            x.resize_(20, 30)
            assert x.shape == (20, 30)
            assert_op_runs_on_neuron("aten::resize_")

            # Resize to smaller size
            x.resize_(5, 10)
            assert x.shape == (5, 10)
            assert_op_runs_on_neuron("aten::resize_")

    def test_resize_1d(self):
        """Test resize operation on 1D tensors."""
        with track_neuron_ops():
            device = torch.device("neuron")
            x = torch.empty(100, device=device)

            # Resize to larger
            x.resize_(200)
            assert x.shape == (200,)
            assert x.numel() == 200

            # Resize to smaller
            x.resize_(50)
            assert x.shape == (50,)
            assert x.numel() == 50
            assert_op_runs_on_neuron("aten::resize_")

    def test_resize_multi_dimensional(self):
        """Test resize operation on multi-dimensional tensors."""
        with track_neuron_ops():
            device = torch.device("neuron")

            # 3D tensor
            x = torch.empty(2, 3, 4, device=device)
            x.resize_(4, 5, 6)
            assert x.shape == (4, 5, 6)

            # 4D tensor
            x = torch.empty(2, 3, 4, 5, device=device)
            x.resize_(1, 2, 3, 4)
            assert x.shape == (1, 2, 3, 4)
            assert_op_runs_on_neuron("aten::resize_")

    def test_resize_zero_size(self):
        """Test resize operation with zero-size dimensions."""
        with track_neuron_ops():
            device = torch.device("neuron")

            # Resize to zero size
            x = torch.empty(10, 20, device=device)
            x.resize_(0, 10)
            assert x.shape == (0, 10)
            assert x.numel() == 0

            # Resize from zero size
            x.resize_(5, 5)
            assert x.shape == (5, 5)
            assert x.numel() == 25
            assert_op_runs_on_neuron("aten::resize_")

    def test_resize_empty_tensor(self):
        """Test resize operation on empty tensors."""
        with track_neuron_ops():
            device = torch.device("neuron")

            # Start with empty tensor
            x = torch.empty(0, device=device)
            assert x.numel() == 0

            # Resize to non-empty
            x.resize_(10)
            assert x.shape == (10,)
            assert x.numel() == 10
            assert_op_runs_on_neuron("aten::resize_")

    def test_resize_preserves_device(self):
        """Test that resize operation preserves the device."""
        with track_neuron_ops():
            device = torch.device("neuron")
            x = torch.empty(10, 10, device=device)

            # Multiple resize operations
            x.resize_(20, 20)
            assert_op_runs_on_neuron("aten::resize_")
            x.resize_(5, 5)
            assert_op_runs_on_neuron("aten::resize_")
            x.resize_(0)
            assert_op_runs_on_neuron("aten::resize_")

    def test_resize_same_size(self):
        """Test resize operation with the same size."""
        with track_neuron_ops():
            device = torch.device("neuron")
            x = torch.empty(10, 20, device=device)

            # Resize to same size
            x.resize_(10, 20)
            assert x.shape == (10, 20)
            assert_op_runs_on_neuron("aten::resize_")

    def test_resize_different_num_dimensions(self):
        """Test changing the number of dimensions during resize."""
        with track_neuron_ops():
            device = torch.device("neuron")

            # 2D to 1D
            x = torch.empty(10, 20, device=device)
            x.resize_(200)
            assert x.shape == (200,)

            # 1D to 3D
            x.resize_(4, 5, 10)
            assert x.shape == (4, 5, 10)

            # 3D to 2D
            x.resize_(20, 10)
            assert x.shape == (20, 10)
            assert_op_runs_on_neuron("aten::resize_")

    def test_resize_large_tensor(self):
        """Test resize operation on large tensors."""
        with track_neuron_ops():
            device = torch.device("neuron")

            # Create a large tensor
            x = torch.empty(1000, 1000, device=device)
            assert x.numel() == 1000000

            # Resize to larger
            x.resize_(2000, 1000)
            assert x.numel() == 2000000

            # Resize to smaller
            x.resize_(100, 100)
            assert x.numel() == 10000
            assert_op_runs_on_neuron("aten::resize_")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32, torch.int64])
    def test_resize_different_dtypes(self, dtype):
        """Test resize operation with different data types."""
        with track_neuron_ops():
            device = torch.device("neuron")
            x = torch.empty(10, 10, device=device, dtype=dtype)

            x.resize_(20, 20)
            assert x.shape == (20, 20)
            assert x.dtype == dtype
            assert_op_runs_on_neuron("aten::resize_")

    @pytest.mark.parametrize(
        "resize_case",
        [
            ("increase_size", (50, 50), (100, 100)),
            ("decrease_size", (50, 50), (25, 25)),
            ("same_size", (50, 50), (50, 50)),
            ("different_dims", (50, 50), (10, 10, 10)),
        ],
    )
    def test_resize_preserves_storage_identity(self, resize_case):
        """Test that resize preserves storage identity - noswap data ptr."""
        case_name, initial_shape, target_shape = resize_case
        device = torch.device("neuron")

        # Create tensor
        tensor = torch.ones(initial_shape, device=device, dtype=torch.float32)

        # Get C++ memory address to raw data buffer
        original_storage_ptr = tensor.untyped_storage()._cdata

        # Perform resize operation
        with track_neuron_ops():
            tensor.resize_(*target_shape)
            assert tensor.shape == target_shape

            # Check that underlying storage pointer is preserved
            new_storage_ptr = tensor.untyped_storage()._cdata
            assert new_storage_ptr == original_storage_ptr, (
                f"Storage pointer should be preserved for {case_name} resize: "
                f"{original_storage_ptr} != {new_storage_ptr}"
            )

            assert_op_runs_on_neuron("aten::resize_")

    @pytest.mark.parametrize("memory_format", [torch.channels_last, torch.contiguous_format])
    def test_resize_channels_last_memory_format(self, memory_format):
        """Test resize with ChannelsLast memory format (4D NHWC)"""
        with track_neuron_ops():
            device = torch.device("neuron")
            x = torch.empty(2, 3, 4, 5, device=device)
            x.resize_(2, 4, 6, 8, memory_format=memory_format)

            assert x.shape == (2, 4, 6, 8)
            if memory_format == torch.channels_last:
                assert x.is_contiguous(memory_format=torch.channels_last)
            assert_op_runs_on_neuron("aten::resize_")

    def test_resize_channels_last_3d_memory_format(self):
        """Test resize with ChannelsLast3d memory format (5D NDHWC)"""
        with track_neuron_ops():
            device = torch.device("neuron")
            x = torch.empty(2, 3, 4, 5, 6, device=device)
            x.resize_(2, 4, 3, 5, 7, memory_format=torch.channels_last_3d)

            assert x.shape == (2, 4, 3, 5, 7)
            assert x.is_contiguous(memory_format=torch.channels_last_3d)
            assert_op_runs_on_neuron("aten::resize_")
