"""Test that scatter operation is properly registered with PyTorch dispatcher."""

from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)
from torch_neuronx.neuron_dynamo_backend.fx.fx_transform import convert_fx_to_stablehlo
from torch_neuronx.utils import use_mlir_aten_ops

# Test cases for both scatter and gather operations
SHAPE_TEST_CASES = [
    [(3, 4), 1, [[3, 2, 1, 0], [0, 1, 2, 3], [3, 2, 1, 0]], (3, 4), torch.float32],
    [(5, 5, 3), 2, [[[0, 1, 2]]], (1, 1, 3), torch.float32],
    [(3, 3), 0, [[0, 2, 1]], (1, 3), torch.float32],
    [(10,), 0, [3, 7, 1], (4,), torch.float32],
    [(10, 5), -1, [], (3, 0), torch.float32],  # Empty index tensor
]
DTYPE_TEST_CASES = []

for dtype in [torch.float16, torch.int32, torch.int64]:
    shape, dim, index, src_shape, float_dtype = SHAPE_TEST_CASES[0]
    DTYPE_TEST_CASES.append((shape, dim, index, src_shape, dtype))

# Additional test cases for scatter operations with scalar values
SCATTER_VALUE_TEST_CASES = [
    [(5, 5, 3), 2, [[[0, 1, 2]]], -2.5, torch.float32],
    [(3, 3), 0, [[0, 2, 1]], 10, torch.int32],
    [(10,), 0, [3, 7, 1], 42, torch.int64],
]

# Combined test cases for scatter (tensor + scalar)
SCATTER_TEST_CASES = SHAPE_TEST_CASES + DTYPE_TEST_CASES + SCATTER_VALUE_TEST_CASES


class TestScatterGatherRegistration:
    """Test scatter and gather operation registration and functionality."""

    @pytest.mark.parametrize("input_shape,dim,index,src_shape_or_value,dtype", SCATTER_TEST_CASES)
    def test_scatter_operation(self, input_shape, dim, index, src_shape_or_value, dtype):
        """Test that torch.scatter works with neuron tensors (both tensor and scalar src)."""
        # Create input tensor on neuron
        input_tensor = torch.zeros(input_shape, dtype=dtype).to("neuron")
        index_tensor = torch.tensor(index, dtype=torch.int64).to("neuron")

        # Handle both tensor and scalar src cases
        if isinstance(src_shape_or_value, tuple):  # Tensor case
            src_shape = src_shape_or_value
            if dtype in (torch.int32, torch.int64):
                src_tensor = torch.randint(-100, 100, src_shape, dtype=dtype).to("neuron")
            else:
                src_tensor = torch.rand(src_shape, dtype=dtype).to("neuron")
            src_cpu = src_tensor.clone().cpu()
        else:  # Scalar case
            src_tensor = src_shape_or_value
            src_cpu = src_shape_or_value

        # Perform scatter operation
        with track_neuron_ops():
            result = torch.scatter(input_tensor, dim, index_tensor, src_tensor)
            if isinstance(src_shape_or_value, tuple):
                assert_op_runs_on_neuron("scatter")
            else:
                assert_op_runs_on_neuron("scatter.value")

        # Verify operation ran by comparing with CPU equivalent
        input_cpu = torch.zeros(input_shape, dtype=dtype)
        index_cpu = torch.tensor(index, dtype=torch.int64)
        expected = torch.scatter(input_cpu, dim, index_cpu, src_cpu)

        torch.testing.assert_close(result.cpu(), expected)

    def test_scatter_inplace_operation(self):
        """Test that Tensor.scatter_ inplace operation works with neuron tensors."""
        # Create input tensor on neuron
        input_tensor = torch.zeros(3, 4, dtype=torch.float32).to("neuron")
        index_tensor = torch.tensor([[0, 1, 2]], dtype=torch.int64).to("neuron")
        src_tensor = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32).to("neuron")

        # Perform inplace scatter operation
        with track_neuron_ops():
            input_tensor.scatter_(0, index_tensor, src_tensor)
            assert_op_runs_on_neuron("aten::scatter")

        # Verify operation ran by comparing with CPU equivalent
        input_cpu = torch.zeros(3, 4, dtype=torch.float32)
        index_cpu = index_tensor.clone().cpu()
        src_cpu = src_tensor.clone().cpu()
        input_cpu.scatter_(0, index_cpu, src_cpu)

        torch.testing.assert_close(input_tensor.cpu(), input_cpu)

    @pytest.mark.parametrize(
        "src_type,reduce_type",
        [
            ("value", "add"),
            ("tensor", "add"),
            ("value", "multiply"),
            ("tensor", "multiply"),
        ],
    )
    def test_scatter_reduce_operation(self, src_type, reduce_type):
        """Test that Tensor.scatter_ inplace add and multiply reduction."""
        # Create input tensor on neuron
        input_tensor = torch.ones(3, 4, dtype=torch.float32).to("neuron")
        index_tensor = torch.tensor(
            [[3, 2, 1, 0], [0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.int64
        ).to("neuron")

        if src_type == "value":
            src = 5.0
            expected_op = "aten::scatter.value_reduce_out"
        else:
            src = torch.tensor(
                [[2.0, 3.0, 4.0, 1.0], [1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 1.0]],
                dtype=torch.float32,
            ).to("neuron")
            expected_op = "aten::scatter.reduce_out"

        # Perform inplace scatter operation with reduce
        with track_neuron_ops():
            input_tensor.scatter_(1, index_tensor, src, reduce=reduce_type)
            assert_op_runs_on_neuron(expected_op)

        # Verify operation ran by comparing with CPU equivalent
        input_cpu = torch.ones(3, 4, dtype=torch.float32)
        index_cpu = index_tensor.clone().cpu()
        src_cpu = src.clone().cpu() if isinstance(src, torch.Tensor) else src
        input_cpu.scatter_(1, index_cpu, src_cpu, reduce=reduce_type)

        torch.testing.assert_close(input_tensor.cpu(), input_cpu)

    @assert_raises(RuntimeError, match=r".*reduce argument must be*")
    def test_unsupported_scatter_reduce(self):
        """Test unsupported reduce arg for scatter."""
        # Create input tensor on neuron
        input_tensor = torch.ones(3, 4, dtype=torch.float32).to("neuron")
        index_tensor = torch.tensor(
            [[3, 2, 1, 0], [0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.int64
        ).to("neuron")
        src = torch.tensor(
            [[2.0, 3.0, 4.0, 1.0], [1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 1.0]], dtype=torch.float32
        ).to("neuron")

        # Perform inplace scatter operation with reduce
        input_tensor.scatter_(1, index_tensor, src, reduce="mean")

    def test_scatter_invalid_index(self):
        """Test scatter with invalid index values to compare CPU vs neuron error handling."""
        print("\n=== Testing scatter with invalid index values ===")

        # Create test tensors with invalid index (out of bounds)
        input_shape = (3, 4)
        input_tensor_cpu = torch.zeros(input_shape, dtype=torch.float32, device="neuron")
        index_tensor_cpu = torch.tensor(
            [[0, 1, 5]], dtype=torch.int64, device="neuron"
        )  # index 5 is out of bounds for dim 1 (size 4)
        src_tensor_cpu = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32, device="neuron")

        # Test neuron behavior
        input_tensor_neuron = input_tensor_cpu.to("neuron")
        index_tensor_neuron = index_tensor_cpu.to("neuron")
        src_tensor_neuron = src_tensor_cpu.to("neuron")

        @assert_raises(
            RuntimeError, match=r".*index 5 is out of bounds for dimension 1 with size 4*"
        )
        def _test_invalid_index():
            torch.scatter(input_tensor_neuron, 1, index_tensor_neuron, src_tensor_neuron)

        _test_invalid_index()

    def test_scatter_dtype_mismatch(self):
        """Test scatter with integer value on float tensor."""
        input_tensor = torch.zeros(1, 2, dtype=torch.float32, device="neuron")
        index_tensor = torch.tensor([[0]], device="neuron")
        result = input_tensor.scatter_(1, index_tensor, 1)
        torch.testing.assert_close(result.cpu(), torch.tensor([[1.0, 0.0]]))

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_scatter_scalar_dtype_conversion(self, dtype):
        """Test scatter with scalar value"""
        input_tensor = torch.zeros(3, 5, dtype=dtype, device="neuron")
        index = torch.tensor([[0, 1, 2, 0, 0]], dtype=torch.long, device="neuron")
        scalar_value = 1.0

        with track_neuron_ops():
            result = torch.scatter(input_tensor, 0, index, scalar_value)
            assert_op_runs_on_neuron("scatter.value")

        input_cpu = torch.zeros(3, 5, dtype=dtype)
        index_cpu = torch.tensor([[0, 1, 2, 0, 0]], dtype=torch.long)
        expected = torch.scatter(input_cpu, 0, index_cpu, scalar_value)

        torch.testing.assert_close(result.cpu(), expected)

    def test_gather_out_variant(self):
        """Test torch.gather with out parameter."""
        # Create input tensor with some values on neuron
        input_tensor = torch.rand(3, 4, dtype=torch.float32).to("neuron")
        index_tensor = torch.tensor([[0, 1, 2]], dtype=torch.int64).to("neuron")

        # Create output tensor
        out_tensor = torch.empty(1, 3, dtype=torch.float32).to("neuron")

        # Perform gather operation with out parameter
        with track_neuron_ops():
            result = torch.gather(input_tensor, 0, index_tensor, out=out_tensor)
            assert_op_runs_on_neuron("aten::gather.out")

        # Verify with CPU equivalent
        input_cpu = input_tensor.cpu()
        index_cpu = index_tensor.cpu()
        out_cpu = torch.empty(1, 3, dtype=torch.float32)
        expected = torch.gather(input_cpu, 0, index_cpu, out=out_cpu)

        torch.testing.assert_close(result.cpu(), expected)
        torch.testing.assert_close(out_tensor.cpu(), out_cpu)

    @pytest.mark.parametrize(
        "input_shape,dim,index,src_shape,dtype", SHAPE_TEST_CASES + DTYPE_TEST_CASES
    )
    def test_gather_operation(self, input_shape, dim, index, src_shape, dtype):
        """Test that torch.gather works with neuron tensors."""
        # Create input tensor with some values on neuron
        if dtype in (torch.int32, torch.int64):
            input_tensor = torch.randint(-100, 100, input_shape, dtype=dtype).to("neuron")
        else:
            input_tensor = torch.rand(input_shape, dtype=dtype).to("neuron")
        index_tensor = torch.tensor(index, dtype=torch.int64).to("neuron")

        # Perform gather operation
        with track_neuron_ops():
            result = torch.gather(input_tensor, dim, index_tensor)
            assert_op_runs_on_neuron("aten::gather")

        # Verify operation ran by comparing with CPU equivalent
        input_cpu = input_tensor.clone().cpu()
        index_cpu = index_tensor.clone().cpu()
        expected = torch.gather(input_cpu, dim, index_cpu)

        torch.testing.assert_close(result.cpu(), expected)

    def test_gather_backward_direct_op_call(self):
        """Test torch.ops.gather_backward direct op call."""
        # Create test tensors
        grad_output = torch.randn(1, 3, dtype=torch.float32).to("neuron")
        input_tensor = torch.randn(3, 3, dtype=torch.float32).to("neuron")
        index_tensor = torch.tensor([[0, 2, 1]], dtype=torch.int32).to("neuron")

        # Direct op call
        with track_neuron_ops():
            result = torch.ops.aten.gather_backward(
                grad_output, input_tensor, 0, index_tensor, False
            )
            assert_op_runs_on_neuron("aten::scatter_add")

        # Verify with CPU equivalent
        grad_output_cpu = grad_output.cpu()
        input_cpu = input_tensor.cpu()
        index_cpu = index_tensor.cpu()
        expected = torch.ops.aten.gather_backward(grad_output_cpu, input_cpu, 0, index_cpu, False)

        torch.testing.assert_close(result.cpu(), expected)

    def test_gather_backward_with_empty_index_tensor(self):
        device = "neuron"
        dtype = torch.float32
        sparse_grad = False
        dim = -1
        input = torch.rand([10, 5], dtype=dtype, device=device, requires_grad=True)
        index = torch.randint(0, 2, [3, 0], dtype=torch.int64, device=device)
        res = torch.gather(input, dim, index, sparse_grad=sparse_grad)
        with track_neuron_ops():
            res.sum().backward()
            assert_op_runs_on_neuron("aten::scatter_add")

        grad = input.grad.to_dense() if sparse_grad else input.grad
        expected_grad = torch.zeros_like(input, requires_grad=False)
        torch.testing.assert_close(grad, expected_grad, atol=0, rtol=0)

    def test_gather_backward_one_dim(self) -> None:
        import random

        device = "neuron"
        m = 2500
        elems = random.randint(10 * m, 20 * m)
        dim = 0
        src = torch.randn(m, device=device, requires_grad=True)
        idx = torch.randint(m, (elems,), device=device)
        res = torch.gather(src, dim, idx)
        weight = torch.rand_like(res, device=device) * 10**6
        res.backward(weight)
        assert src.grad is not None
        grad = src.grad.detach().clone()

        for _ in range(2):
            src.grad.data.zero_()
            res = torch.gather(src, dim, idx)
            res.backward(weight)
            torch.testing.assert_close(src.grad, grad, atol=0, rtol=0)


class TestGatherPriorityImpl:
    """Test priority-based implementation for gather."""

    @patch(
        "torch_neuronx.python_ops.torch_mlir.kernel.convert_fx_to_stablehlo",
        wraps=convert_fx_to_stablehlo,
    )
    def test_gather_nki_kernel_used(self, mock_compiler):
        """Verify NKI kernel is used for 2D tensors with dim=0"""
        initial_count = mock_compiler.call_count

        input_neuron = torch.randn(3, 4, dtype=torch.float32, device="neuron")
        index_tensor = torch.tensor([0, 2], device="neuron")
        index_neuron = index_tensor.unsqueeze(1).expand(-1, 4)

        with torch.no_grad(), track_neuron_ops():
            torch.gather(input_neuron, 0, index_neuron)
            assert_op_runs_on_neuron("aten::gather")

        # index_neuron is not contiguous
        # .contiguous and .gather trigger kernel
        if use_mlir_aten_ops():
            assert (
                mock_compiler.call_count == initial_count + 2
            ), "Compiler was not called as expected"
        else:
            assert (
                mock_compiler.call_count == initial_count + 1
            ), "Compiler was not called as expected"

    @patch("torch_neuronx.python_ops.gather.GatherNKIImpl.can_handle", return_value=False)
    def test_gather_xla_fallback(self, mocked_handle):
        """Verify XLA fallback when NKI can't handle the input"""
        input_cpu = torch.randn(3, 4, dtype=torch.float32)
        index_cpu = torch.tensor([[0, 1, 2, 1], [2, 0, 1, 0]])

        result_cpu = torch.gather(input_cpu, 0, index_cpu)

        input_neuron = input_cpu.clone().to("neuron")
        index_neuron = index_cpu.to("neuron")

        with track_neuron_ops():
            result_neuron = torch.gather(input_neuron, 0, index_neuron)
            assert_op_runs_on_neuron("aten::gather")

        torch.testing.assert_close(result_neuron.cpu(), result_cpu, rtol=1e-4, atol=1e-4)
