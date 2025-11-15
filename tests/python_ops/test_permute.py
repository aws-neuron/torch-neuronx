"""Test that ones operation is properly registered with PyTorch dispatcher."""

import re

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestPermute:
    def setup(self):
        """Set up test environment before each test method."""
        # Set fixed random seed for reproducibility
        torch.manual_seed(42)

    def test_permute_basic(self):
        """Test basic permute functionality."""
        with track_neuron_ops():
            input_arr = torch.randn(2, 3, 5).to("neuron")
            assert input_arr.size() == torch.Size([2, 3, 5])
            permuted_arr = torch.permute(input_arr, (2, 0, 1))
            assert permuted_arr.size() == torch.Size([5, 2, 3])
            assert_op_runs_on_neuron("aten::empty_strided")

    @pytest.mark.parametrize(
        "input_shape, permutation, expected_shape",
        [
            # 1D tensor case
            ((5,), (0,), (5,)),
            # 2D tensor case
            ((3, 4), (1, 0), (4, 3)),
            # 4D tensor case
            ((2, 3, 4, 5), (3, 1, 0, 2), (5, 3, 2, 4)),
        ],
    )
    def test_permute_different_dims(self, input_shape, permutation, expected_shape):
        """Test permute on tensors with different numbers of dimensions."""
        with track_neuron_ops():
            x = torch.randn(*input_shape).to("neuron")
            p = torch.permute(x, permutation)
            assert p.size() == torch.Size(expected_shape)
            assert_op_runs_on_neuron("aten::empty_strided")

    def test_permute_reverse(self):
        """Test permute with reversed dimensions."""
        with track_neuron_ops():
            x = torch.randn(2, 3, 4).to("neuron")
            p = torch.permute(x, (2, 1, 0))
            assert p.size() == torch.Size([4, 3, 2])
            assert_op_runs_on_neuron("aten::empty_strided")

    def test_permute_data_correctness(self):
        """Test that permute correctly rearranges data."""
        # Create and permute tensor on CPU first
        x_cpu = torch.arange(24).reshape(2, 3, 4)
        p_cpu = torch.permute(x_cpu, (2, 0, 1))

        # Now test on neuron and compare with CPU result
        with track_neuron_ops():
            x = x_cpu.to("neuron")
            p = torch.permute(x, (2, 0, 1))
            assert torch.all(p.cpu() == p_cpu)
            assert_op_runs_on_neuron("aten::empty_strided")

    def test_permute_with_backward(self):
        """Test permute with backward pass."""
        with track_neuron_ops():
            # Create the tensor and keep on CPU initially for requires_grad
            x = torch.randn(2, 3, 4, requires_grad=True)

            # Then move to neuron device
            x_neuron = x.to("neuron")
            p = torch.permute(x_neuron, (2, 0, 1))
            loss = p.sum()
            loss.backward()

            # Check the grad_fn instead of grad values
            assert p.grad_fn is not None

            assert_op_runs_on_neuron("aten::empty_strided")

    def test_permute_tensor_method(self):
        """Test tensor.permute() method instead of torch.permute."""
        with track_neuron_ops():
            x = torch.randn(2, 3, 4).to("neuron")
            p = x.permute(2, 0, 1)  # Using tensor method syntax
            assert p.size() == torch.Size([4, 2, 3])
            assert_op_runs_on_neuron("aten::empty_strided")

    def test_permute_non_contiguous(self):
        """Test permute with non-contiguous tensors."""
        with track_neuron_ops():
            x = torch.arange(24).reshape(2, 3, 4).to("neuron")
            # Create a non-contiguous tensor by slicing
            y = x[:, ::2, :]  # Take every other row
            p = torch.permute(y, (2, 0, 1))
            assert p.size() == torch.Size([4, 2, 2])

            # Compare with CPU implementation
            x_cpu = torch.arange(24).reshape(2, 3, 4)
            y_cpu = x_cpu[:, ::2, :]
            p_cpu = torch.permute(y_cpu, (2, 0, 1))
            assert torch.all(p.cpu() == p_cpu)

            assert_op_runs_on_neuron("aten::empty_strided")

    @pytest.mark.parametrize(
        "dtype, input_fn",
        [
            (torch.float32, lambda: torch.randn(2, 3, 4)),
            (torch.float16, lambda: torch.randn(2, 3, 4)),
            (torch.int32, lambda: torch.randint(0, 10, (2, 3, 4))),
            (torch.int64, lambda: torch.randint(0, 10, (2, 3, 4))),
        ],
        ids=["float32", "float16", "int32", "int64"],
    )
    def test_permute_dtypes(self, dtype, input_fn):
        """Test permute with different dtypes."""
        # Create tensor on CPU first
        x_cpu = input_fn().to(dtype)

        with track_neuron_ops():
            x = x_cpu.to("neuron")
            p = torch.permute(x, (2, 0, 1))
            assert p.size() == torch.Size([4, 2, 3])
            assert p.dtype == dtype
            assert_op_runs_on_neuron("aten::empty_strided")

    def test_permute_wrong_dim_count(self):
        """Test permute with wrong number of dimensions (should fail)."""
        # First check CPU behavior to get expected error messages
        x_cpu = torch.randn(2, 3, 4)

        # Get CPU error message for too few dimensions
        try:
            torch.permute(x_cpu, (0, 1))
        except RuntimeError as e:
            few_dims_error = str(e)

        # Get CPU error message for too many dimensions
        try:
            torch.permute(x_cpu, (0, 1, 2, 3))
        except RuntimeError as e:
            many_dims_error = str(e)

        # Now test neuron behavior
        with track_neuron_ops():
            x = x_cpu.to("neuron")

            # Test too few dimensions
            @assert_raises(RuntimeError, match=re.escape(few_dims_error))
            def _test_few_dims():
                torch.permute(x, (0, 1))

            _test_few_dims()

            # Test too many dimensions
            @assert_raises(RuntimeError, match=re.escape(many_dims_error))
            def _test_many_dims():
                torch.permute(x, (0, 1, 2, 3))

            _test_many_dims()

    def test_permute_duplicate_dims(self):
        """Test permute with duplicate dimensions (should fail)."""
        # First check CPU behavior to get expected error message
        x_cpu = torch.randn(2, 3, 4)

        # Get CPU error message for duplicate dimensions
        try:
            torch.permute(x_cpu, (0, 0, 1))
        except RuntimeError as e:
            expected_error = str(e)

        # Now test neuron behavior
        with track_neuron_ops():
            x = x_cpu.to("neuron")

            @assert_raises(RuntimeError, match=re.escape(expected_error))
            def _test_duplicate_dims():
                torch.permute(x, (0, 0, 1))

            _test_duplicate_dims()

    @pytest.mark.xfail(
        reason="JAX doesn't support negative indices in permutation tuples unlike PyTorch"
    )
    def test_permute_negative_dims(self):
        """Test permute with negative dimensions."""
        with track_neuron_ops():
            x_cpu = torch.randn(2, 3, 4)
            expected_size = torch.Size([4, 2, 3])  # -1 refers to last dimension (4)
            x = x_cpu.to("neuron")
            p_neuron = torch.permute(x, (-1, 0, 1))
            assert p_neuron.size() == expected_size
            assert_op_runs_on_neuron("aten::empty_strided")

    def test_permute_out_of_bounds_dims(self):
        """Test permute with out-of-bounds dimensions behaves like CPU."""
        x_cpu = torch.randn(2, 3, 4)

        # Test out of bounds (positive) dimension
        try:
            torch.permute(x_cpu, (0, 1, 3))
            raise AssertionError("Expected IndexError for out of bounds dimension")
        except IndexError as e:
            out_of_bounds_error = str(e)

        # Now test neuron behavior
        with track_neuron_ops():
            x = x_cpu.to("neuron")

            # Test out of bounds dimension
            @assert_raises(IndexError, match=re.escape(out_of_bounds_error))
            def _test_out_of_bounds():
                torch.permute(x, (0, 1, 3))

            _test_out_of_bounds()

    def test_permute_scalar_tensor(self):
        """Test permute on a scalar tensor."""
        with track_neuron_ops():
            # A scalar tensor has 0 dimensions
            x = torch.tensor(5.0).to("neuron")
            # Permuting a scalar should give the same scalar
            p = torch.permute(x, ())
            assert p.item() == 5.0
            assert_op_runs_on_neuron("aten::empty_strided")

    def test_permute_1d_tensor(self):
        """Test permute on 1D tensor."""
        with track_neuron_ops():
            x = torch.arange(5).to("neuron")
            p = torch.permute(x, (0,))  # Only one way to permute a 1D tensor
            assert torch.all(p == x)
            assert_op_runs_on_neuron("aten::empty_strided")

    @pytest.mark.xfail(reason="JAX has limitations with empty tensors during tracing")
    def test_permute_empty_tensor(self):
        """Test permute on tensor with empty dimensions matches CPU behavior."""
        # Check what PyTorch does with empty tensors
        x_cpu = torch.zeros((2, 0, 3))
        p_cpu = torch.permute(x_cpu, (2, 0, 1))
        expected_size = p_cpu.size()

        # Now test on neuron
        with track_neuron_ops():
            x = torch.zeros((2, 0, 3)).to("neuron")
            p = torch.permute(x, (2, 0, 1))
            assert p.size() == expected_size
            assert_op_runs_on_neuron("aten::empty_strided")

    def test_permute_identity(self):
        """Test identity permutation (should be no-op)."""
        with track_neuron_ops():
            x = torch.randn(2, 3, 4).to("neuron")
            p = torch.permute(x, (0, 1, 2))  # Same order as original
            assert p.size() == x.size()
            assert torch.all(p == x)
            assert_op_runs_on_neuron("aten::empty_strided")

    def test_permute_large_dims(self):
        """Test permute with tensor that has many dimensions."""
        with track_neuron_ops():
            x = torch.randn(2, 3, 4, 5, 6, 7).to("neuron")
            p = torch.permute(x, (5, 3, 1, 0, 2, 4))
            assert p.size() == torch.Size([7, 5, 3, 2, 4, 6])
            assert_op_runs_on_neuron("aten::empty_strided")

    def test_permute_singleton_dims(self):
        """Test permute with singleton dimensions."""
        with track_neuron_ops():
            # Tensor with some dimensions of size 1
            x = torch.randn(1, 3, 1, 5).to("neuron")
            p = torch.permute(x, (3, 0, 2, 1))
            assert p.size() == torch.Size([5, 1, 1, 3])
            assert_op_runs_on_neuron("aten::empty_strided")

    def test_permute_non_contiguous_extra(self):
        """Test permute on a non-contiguous tensor."""
        with track_neuron_ops():
            x = torch.randn(2, 3, 4, 5).to("neuron")
            # Create a non-contiguous tensor via slicing
            non_contiguous = x[:, ::2, :, ::2]  # Take every other slice
            assert not non_contiguous.is_contiguous()

            p = torch.permute(non_contiguous, (3, 0, 2, 1))
            assert p.size() == torch.Size([3, 2, 4, 2])  # 3 because we took every other from dim 5
            assert_op_runs_on_neuron("aten::empty_strided")
