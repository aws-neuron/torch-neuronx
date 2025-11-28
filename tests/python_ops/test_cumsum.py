import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestCumsum:
    @pytest.mark.parametrize(
        "input_shape,dim",
        [
            ((4,), 0),  # 1D tensor
            ((2, 3), 0),  # 2D tensor, dim 0
            ((2, 3), 1),  # 2D tensor, dim 1
            ((3, 4, 2), 0),  # 3D tensor, dim 0
            ((3, 4, 2), 1),  # 3D tensor, dim 1
            ((3, 4, 2), 2),  # 3D tensor, dim 2
        ],
        ids=["1d", "2d_dim0", "2d_dim1", "3d_dim0", "3d_dim1", "3d_dim2"],
    )
    def test_cumsum_dimensions(self, input_shape, dim):
        """Test cumsum along different dimensions and tensor shapes"""
        with track_neuron_ops():
            input_cpu = torch.randn(input_shape)
            input_neuron = input_cpu.to("neuron")

            output_cpu = torch.cumsum(input_cpu, dim=dim)
            output_neuron = torch.cumsum(input_neuron, dim=dim)

            torch.testing.assert_close(output_neuron.cpu(), output_cpu)
            assert_op_runs_on_neuron("aten::cumsum")

    @pytest.mark.parametrize(
        "tensor_shape,dim,description",
        [
            ((), 0, "1D empty tensor"),  # Empty 1D tensor
            ((0, 3), 0, "2D empty tensor (0 rows)"),  # 2D empty tensor with 0 rows
            ((3, 0), 1, "2D empty tensor (0 columns)"),  # 2D empty tensor with 0 columns
            ((0, 0), 0, "2D fully empty tensor dim 0"),  # Fully empty 2D tensor
            ((0, 0), 1, "2D fully empty tensor dim 1"),  # Fully empty 2D tensor
        ],
        ids=[
            "empty_1d",
            "empty_2d_rows",
            "empty_2d_cols",
            "empty_2d_full_dim0",
            "empty_2d_full_dim1",
        ],
    )
    def test_cumsum_empty_tensors(self, tensor_shape, dim, description):
        """Test cumsum with empty tensors"""
        with track_neuron_ops():
            input_cpu = torch.tensor([]) if tensor_shape == () else torch.empty(tensor_shape)
            input_neuron = input_cpu.to("neuron")

            output_cpu = torch.cumsum(input_cpu, dim=dim)
            output_neuron = torch.cumsum(input_neuron, dim=dim)

            torch.testing.assert_close(output_neuron.cpu(), output_cpu)
            assert_op_runs_on_neuron("aten::cumsum")

    @pytest.mark.parametrize(
        "tensor_shape,dim",
        [
            ((5,), 0),  # 1D zero tensor
            ((3, 4), 0),  # 2D zero tensor, dim 0
            ((3, 4), 1),  # 2D zero tensor, dim 1
            ((2, 3, 4), 1),  # 3D zero tensor
        ],
        ids=["zero_1d", "zero_2d_dim0", "zero_2d_dim1", "zero_3d"],
    )
    def test_cumsum_zero_tensors(self, tensor_shape, dim):
        """Test cumsum with zero-filled tensors"""
        with track_neuron_ops():
            input_cpu = torch.zeros(tensor_shape)
            input_neuron = input_cpu.to("neuron")

            output_cpu = torch.cumsum(input_cpu, dim=dim)
            output_neuron = torch.cumsum(input_neuron, dim=dim)

            torch.testing.assert_close(output_neuron.cpu(), output_cpu)
            assert_op_runs_on_neuron("aten::cumsum")

    @pytest.mark.parametrize(
        "tensor_shape,dim",
        [
            ((2, 3), 5),  # Invalid dimension (out of bounds)
            ((2, 3), -5),  # Invalid dimension (negative out of bounds)
        ],
        ids=["invalid_dim_positive", "invalid_dim_negative"],
    )
    @assert_raises(IndexError)
    def test_cumsum_invalid_dim_positive(self, tensor_shape, dim):
        """Test cumsum failure case with invalid positive dimension"""
        input_cpu = torch.randn(tensor_shape)
        input_neuron = input_cpu.to("neuron")
        torch.cumsum(input_neuron, dim=dim)

    @pytest.mark.parametrize(
        "tensor_shape,dim",
        [
            ((2, 3), 1.5),  # Float dim parameter
        ],
        ids=["float_dim"],
    )
    @assert_raises(TypeError)
    def test_cumsum_float_dim(self, tensor_shape, dim):
        """Test cumsum failure case with float dim parameter"""
        input_cpu = torch.randn(tensor_shape)
        input_neuron = input_cpu.to("neuron")
        torch.cumsum(input_neuron, dim=dim)

    @pytest.mark.parametrize(
        "tensor_shape,dim",
        [
            ((2, 3), "invalid"),  # String dim parameter
            ((2, 3), None),  # None dim parameter
        ],
    )
    @assert_raises(RuntimeError)
    def test_cumsum_string_dim(self, tensor_shape, dim):
        """Test cumsum failure case with string dim parameter"""
        input_cpu = torch.randn(tensor_shape)
        input_neuron = input_cpu.to("neuron")
        torch.cumsum(input_neuron, dim=dim)

    def test_cumsum_scalar_tensor(self):
        """Test cumsum with scalar tensor (edge case)"""
        with track_neuron_ops():
            scalar_cpu = torch.tensor(5.0)
            scalar_neuron = scalar_cpu.to("neuron")

            output_cpu = torch.cumsum(scalar_cpu, dim=0)
            output_neuron = torch.cumsum(scalar_neuron, dim=0)

            torch.testing.assert_close(output_neuron.cpu(), output_cpu)
            assert_op_runs_on_neuron("aten::cumsum")

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.bfloat16,
        ],
        ids=["float32", "float64", "int32", "int64", "float16", "bfloat16"],
    )
    def test_cumsum_with_dtype(self, dtype):
        """Test cumsum with explicit dtype"""
        with track_neuron_ops():
            input_cpu = torch.tensor([1, 2, 3, 4])
            input_neuron = input_cpu.to("neuron")

            output_cpu = torch.cumsum(input_cpu, dim=0, dtype=dtype)
            output_neuron = torch.cumsum(input_neuron, dim=0, dtype=dtype)

            assert output_neuron.dtype == dtype
            torch.testing.assert_close(output_neuron.cpu(), output_cpu)
            assert_op_runs_on_neuron("aten::cumsum")

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.bfloat16,
        ],
        ids=["float32", "float64", "int32", "int64", "float16", "bfloat16"],
    )
    def test_cumsum_input_dtype(self, dtype):
        """Test cumsum with explicit dtype"""
        with track_neuron_ops():
            input_cpu = torch.tensor([1, 2, 3, 4], dtype=dtype)
            input_neuron = input_cpu.to("neuron")

            output_cpu = torch.cumsum(input_cpu, dim=0)
            output_neuron = torch.cumsum(input_neuron, dim=0)

            torch.testing.assert_close(output_neuron.cpu(), output_cpu)

    def test_cumsum_inplace(self):
        """Test that cumsum_ (in-place) runs on Neuron without CPU fallback"""
        with track_neuron_ops():
            input_cpu = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            input_neuron = input_cpu.to("neuron")

            # Create a copy for CPU inplace operation
            input_cpu_copy = input_cpu.clone()

            output_cpu = input_cpu_copy.cumsum_(dim=1)
            original_id = id(input_neuron)
            output_neuron = input_neuron.cumsum_(dim=1)

            # Verify it's truly in-place (same tensor object)
            assert id(output_neuron) == original_id
            assert output_neuron.device.type == "neuron"
            torch.testing.assert_close(output_neuron.cpu(), output_cpu)
            assert_op_runs_on_neuron("aten::cumsum")

    def test_cumsum_out(self):
        """Test that cumsum with out parameter runs on Neuron without CPU fallback"""
        with track_neuron_ops():
            input_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0])
            input_neuron = input_cpu.to("neuron")

            # Pre-allocate output tensors
            out_cpu = torch.empty_like(input_cpu)
            out_neuron = torch.empty_like(input_neuron, device="neuron")

            output_cpu = torch.cumsum(input_cpu, dim=0, out=out_cpu)
            output_neuron = torch.cumsum(input_neuron, dim=0, out=out_neuron)

            # Verify the result is written to the out tensor
            assert output_neuron is out_neuron
            assert output_neuron.device.type == "neuron"
            torch.testing.assert_close(output_neuron.cpu(), output_cpu)
            torch.testing.assert_close(out_neuron.cpu(), out_cpu)
            assert_op_runs_on_neuron("aten::cumsum.out")
