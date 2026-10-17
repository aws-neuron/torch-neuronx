import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)


class TestStack:
    """Test cases for aten::stack implementation"""

    @pytest.mark.parametrize(
        "tensor_data",
        [
            [[1, 2, 3], [4, 5, 6]],
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            [1, 2, 3],
            [torch.randn(2, 3, 4), torch.randn(2, 3, 4)],
            [torch.randn(3) for _ in range(5)],
        ],
    )
    def test_stack_basic(self, tensor_data):
        """Test basic stacking functionality with
        different tensor shapes and counts"""
        device = "neuron"
        with track_neuron_ops():
            tensors_neuron = [
                t.to(device) if isinstance(t, torch.Tensor) else torch.tensor(t, device=device)
                for t in tensor_data
            ]
            tensors_cpu = [
                t.cpu() if isinstance(t, torch.Tensor) else torch.tensor(t) for t in tensor_data
            ]

            result_neuron = torch.stack(tensors_neuron)
            result_cpu = torch.stack(tensors_cpu)

            torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            assert_op_runs_on_neuron("aten::stack")

    @pytest.mark.parametrize(
        "shape,dims",
        [
            ((2, 3), [1, 2, -1]),
            ((2, 3, 4), [0, 1, 2, 3, -2]),
        ],
    )
    def test_stack_dimensions(self, shape, dims):
        """Test stacking along different dimensions"""
        device = "neuron"
        with track_neuron_ops():
            a_neuron = torch.randn(shape, device=device)
            b_neuron = torch.randn(shape, device=device)

            a_cpu = a_neuron.cpu()
            b_cpu = b_neuron.cpu()

            for dim in dims:
                result_neuron = torch.stack([a_neuron, b_neuron], dim=dim)
                result_cpu = torch.stack([a_cpu, b_cpu], dim=dim)

                torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            assert_op_runs_on_neuron("aten::stack")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32, torch.int64])
    def test_stack_dtypes(self, dtype):
        """Test stack with different dtypes"""
        device = "neuron"
        with track_neuron_ops():
            a_neuron = torch.tensor([1, 2, 3], device=device, dtype=dtype)
            b_neuron = torch.tensor([4, 5, 6], device=device, dtype=dtype)

            result_neuron = torch.stack([a_neuron, b_neuron])
            result_cpu = torch.stack([a_neuron.cpu(), b_neuron.cpu()])

            assert result_neuron.dtype == dtype
            torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            assert_op_runs_on_neuron("aten::stack")

    def test_stack_with_out_parameter(self):
        """Test stack with out parameter"""
        device = "neuron"
        with track_neuron_ops():
            a_neuron = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
            b_neuron = torch.tensor([4, 5, 6], dtype=torch.float32, device=device)
            out_neuron = torch.empty((2, 3), dtype=torch.float32, device=device)

            a_cpu = a_neuron.cpu()
            b_cpu = b_neuron.cpu()
            out_cpu = torch.empty((2, 3), dtype=torch.float32)

            torch.stack([a_neuron, b_neuron], out=out_neuron)
            torch.stack([a_cpu, b_cpu], out=out_cpu)

            torch.testing.assert_close(out_neuron.cpu(), out_cpu)
            assert_op_runs_on_neuron("aten::stack.out")

    @pytest.mark.parametrize(
        "tensor1_shape,tensor2_shape",
        [
            (
                (3,),
                (2,),
            ),
            ((2, 2), (4,)),
            ((2, 2), (2, 2, 1)),
        ],
    )
    @assert_raises(RuntimeError, match="stack expects each tensor to be equal size")
    def test_stack_incompatible_shapes(self, tensor1_shape, tensor2_shape):
        """Test that incompatible shapes raise error"""
        device = "neuron"

        a = torch.randn(tensor1_shape, device=device)
        b = torch.randn(tensor2_shape, device=device)

        torch.stack([a, b])

        a_cpu = a.cpu()
        b_cpu = b.cpu()
        torch.stack([a_cpu, b_cpu])

    @pytest.mark.parametrize("dim", [3, -3])
    @assert_raises(IndexError, match="Dimension out of range")
    def test_stack_invalid_dim(self, dim):
        """Test stack with invalid dimension"""
        device = "neuron"

        a = torch.randn(3, device=device)
        b = torch.randn(3, device=device)

        torch.stack([a, b], dim=dim)
        a_cpu = a.cpu()
        b_cpu = b.cpu()
        torch.stack([a_cpu, b_cpu], dim=dim)

    @pytest.mark.parametrize(
        "shape,dim",
        [
            ((0, 3), 0),
            ((2, 0), 1),
        ],
    )
    def test_stack_empty_tensors(self, shape, dim):
        """Test stacking tensors that have a dimension of size 0"""
        device = "neuron"
        with track_neuron_ops():
            a_neuron = torch.empty(shape, device=device)
            b_neuron = torch.empty(shape, device=device)

            a_cpu = a_neuron.cpu()
            b_cpu = b_neuron.cpu()

            result_neuron = torch.stack([a_neuron, b_neuron], dim=dim)
            result_cpu = torch.stack([a_cpu, b_cpu], dim=dim)

            torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            assert_op_runs_on_neuron("aten::stack")

    @pytest.mark.parametrize("dim", [0, 1])
    def test_stack_empty_tensor_list(self, dim):
        """Test stacking tensors created with torch.tensor([])"""
        device = "neuron"
        with track_neuron_ops():
            a_neuron = torch.tensor([], device=device)
            b_neuron = torch.tensor([], device=device)
            c_neuron = torch.tensor([], device=device)

            a_cpu = a_neuron.cpu()
            b_cpu = b_neuron.cpu()
            c_cpu = c_neuron.cpu()

            result_neuron = torch.stack([a_neuron, b_neuron, c_neuron], dim=dim)
            result_cpu = torch.stack([a_cpu, b_cpu, c_cpu], dim=dim)

            torch.testing.assert_close(result_neuron.cpu(), result_cpu)

            assert_op_runs_on_neuron("aten::stack")


class TestInternalStack:
    """Test cases for aten::_stack and aten::_stack.out."""

    @pytest.mark.parametrize("dim", [0, 1])
    @pytest.mark.parametrize(
        "shape,num_tensors",
        [((256, 4096), 15), ((4, 8), 3), ((32, 64), 5)],
    )
    def test_internal_stack_out_correctness(self, shape, num_tensors, dim):
        """Test aten::_stack.out produces correct shape and values."""
        tensors_cpu = [torch.randn(shape) for _ in range(num_tensors)]
        expected = torch.stack(tensors_cpu, dim=dim)

        tensors_neuron = [t.to("neuron") for t in tensors_cpu]
        out = torch.empty(expected.shape, device="neuron")

        with track_neuron_ops():
            result = torch.ops.aten._stack.out(tensors_neuron, dim, out=out)
            assert_op_runs_on_neuron("aten::_stack")

        assert out.data_ptr() == result.data_ptr()
        assert out.shape == expected.shape
        torch.testing.assert_close(out.cpu(), expected)

    def test_internal_stack_functional(self):
        """Test aten::_stack (functional) produces correct results."""
        tensors_cpu = [torch.randn(4, 8) for _ in range(3)]
        expected = torch.stack(tensors_cpu, dim=0)

        tensors_neuron = [t.to("neuron") for t in tensors_cpu]

        with track_neuron_ops():
            result = torch.ops.aten._stack(tensors_neuron, 0)
            assert_op_runs_on_neuron("aten::_stack")

        assert result.shape == expected.shape
        torch.testing.assert_close(result.cpu(), expected)
