"""Tests for unary operations: sqrt, neg, gelu, relu, softmax, and log_softmax."""

import math

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)
from torch_neuronx.utils import use_mlir_aten_ops


class TestUnaryOps:
    """Test suite for unary operations."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.fixture
    def cpu_device(self):
        """Get the CPU device."""
        return torch.device("cpu")

    def _test_unary_op(self, op_name, input_tensor, device, expected_fn=None, **kwargs):
        """Helper to test a unary operation."""
        # Move input to device
        neuron_input = input_tensor.to(device)

        # Get the operation
        op = getattr(neuron_input, op_name)

        # Execute operation
        result = op(**kwargs)

        # Check result is on correct device
        assert result.device.type == "neuron"

        # Compare with expected result
        if expected_fn is not None:
            expected = expected_fn(input_tensor)
            torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)
        else:
            # Compare with CPU result
            cpu_op = getattr(input_tensor, op_name)
            expected = cpu_op(**kwargs)
            torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

        return result

    def _test_unary_op_out(self, op_name, input_tensor, device, **kwargs):
        """Helper to test the out variant of a unary operation."""
        # Move input to device
        neuron_input = input_tensor.to(device)

        # Create output tensor
        out = torch.empty_like(neuron_input)

        # Get the operation
        op = getattr(torch, op_name)

        # Execute operation
        result = op(neuron_input, out=out, **kwargs)

        # Check that result is the same as out
        assert result is out
        assert result.device.type == "neuron"

        # Compare with CPU result
        cpu_op = getattr(torch, op_name)
        cpu_out = torch.empty_like(input_tensor)
        expected = cpu_op(input_tensor, out=cpu_out, **kwargs)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

        return result

    # Test sqrt operation
    def test_sqrt_runs_on_neuron(self, device):
        """Test that sqrt runs on Neuron"""
        if not torch.neuron.is_available():
            pytest.skip("Neuron device not available")

        with track_neuron_ops():
            input_tensor = torch.tensor([1.0, 4.0, 9.0, 16.0], device=device)
            result = torch.sqrt(input_tensor)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::sqrt")

    def test_sqrt_inplace_runs_on_neuron(self, device):
        """Test that sqrt_ runs on Neuron"""
        if not torch.neuron.is_available():
            pytest.skip("Neuron device not available")

        with track_neuron_ops():
            input_tensor = torch.tensor([1.0, 4.0, 9.0, 16.0], device=device)
            original_data_ptr = input_tensor.data_ptr()
            result = input_tensor.sqrt_()
            assert result is input_tensor
            assert result.data_ptr() == original_data_ptr
            assert input_tensor.device.type == "neuron"
            # In-place ops are tracked as regular ops without underscore
            assert_op_runs_on_neuron("sqrt")

    def test_sqrt_basic(self, device):
        """Test basic sqrt operation."""
        input_tensor = torch.tensor([1.0, 4.0, 9.0, 16.0])
        self._test_unary_op("sqrt", input_tensor, device)

    def test_sqrt_2d(self, device):
        """Test sqrt on 2D tensor."""
        input_tensor = torch.tensor([[1.0, 4.0], [9.0, 16.0]])
        self._test_unary_op("sqrt", input_tensor, device)

    def test_sqrt_different_dtypes(self, device):
        """Test sqrt with different data types."""
        for dtype in [torch.float32, torch.float16]:
            input_tensor = torch.tensor([1.0, 4.0, 9.0, 16.0], dtype=dtype)
            result = self._test_unary_op("sqrt", input_tensor, device)
            assert result.dtype == dtype

    def test_sqrt_out(self, device):
        """Test sqrt.out operation."""
        input_tensor = torch.tensor([1.0, 4.0, 9.0, 16.0])
        self._test_unary_op_out("sqrt", input_tensor, device)

    def test_sqrt_inplace(self, device):
        """Test in-place sqrt_ operation."""
        input_tensor = torch.tensor([1.0, 4.0, 9.0, 16.0])
        neuron_input = input_tensor.to(device)
        original_data_ptr = neuron_input.data_ptr()

        result = neuron_input.sqrt_()

        # Check that operation was in-place
        assert result is neuron_input
        assert result.data_ptr() == original_data_ptr

        # Check correctness
        expected = input_tensor.sqrt()
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_sqrt_empty(self, device):
        """Test sqrt on empty tensor."""
        input_tensor = torch.tensor([])
        self._test_unary_op("sqrt", input_tensor, device)

    def test_sqrt_large(self, device):
        """Test sqrt on larger tensor."""
        input_tensor = torch.rand(100, 100) * 100
        self._test_unary_op("sqrt", input_tensor, device)

    # Test neg operation
    def test_neg_runs_on_neuron(self, device):
        """Test that neg runs on Neuron"""
        if not torch.neuron.is_available():
            pytest.skip("Neuron device not available")

        with track_neuron_ops():
            input_tensor = torch.tensor([1.0, -2.0, 3.0, -4.0], device=device)
            result = torch.neg(input_tensor)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::neg")

    def test_neg_inplace_runs_on_neuron(self, device):
        """Test that neg_ runs on Neuron"""
        if not torch.neuron.is_available():
            pytest.skip("Neuron device not available")

        with track_neuron_ops():
            input_tensor = torch.tensor([1.0, -2.0, 3.0, -4.0], device=device)
            original_data_ptr = input_tensor.data_ptr()
            result = input_tensor.neg_()
            assert result is input_tensor
            assert result.data_ptr() == original_data_ptr
            assert input_tensor.device.type == "neuron"
            # In-place ops are tracked as regular ops without underscore
            assert_op_runs_on_neuron("neg")

    def test_neg_basic(self, device):
        """Test basic neg operation."""
        input_tensor = torch.tensor([1.0, -2.0, 3.0, -4.0])
        self._test_unary_op("neg", input_tensor, device)

    def test_neg_2d(self, device):
        """Test neg on 2D tensor."""
        input_tensor = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
        self._test_unary_op("neg", input_tensor, device)

    def test_neg_different_dtypes(self, device):
        """Test neg with different data types."""
        for dtype in [torch.float32, torch.float16, torch.int32]:
            if dtype == torch.int32:
                input_tensor = torch.tensor([1, -2, 3, -4], dtype=dtype)
            else:
                input_tensor = torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=dtype)
            result = self._test_unary_op("neg", input_tensor, device)
            assert result.dtype == dtype

    def test_neg_out(self, device):
        """Test neg.out operation."""
        input_tensor = torch.tensor([1.0, -2.0, 3.0, -4.0])
        self._test_unary_op_out("neg", input_tensor, device)

    def test_neg_inplace(self, device):
        """Test in-place neg_ operation."""
        input_tensor = torch.tensor([1.0, -2.0, 3.0, -4.0])
        neuron_input = input_tensor.to(device)
        original_data_ptr = neuron_input.data_ptr()

        result = neuron_input.neg_()

        # Check that operation was in-place
        assert result is neuron_input
        assert result.data_ptr() == original_data_ptr

        # Check correctness
        expected = input_tensor.neg()
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_neg_empty(self, device):
        """Test neg on empty tensor."""
        input_tensor = torch.tensor([])
        self._test_unary_op("neg", input_tensor, device)

    def test_neg_large(self, device):
        """Test neg on larger tensor."""
        input_tensor = torch.randn(100, 100)
        self._test_unary_op("neg", input_tensor, device)

    # Test gelu operation
    def test_gelu_runs_on_neuron(self, device):
        """Test that gelu runs on Neuron"""
        if not torch.neuron.is_available():
            pytest.skip("Neuron device not available")

        with track_neuron_ops():
            input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device)
            result = torch.nn.functional.gelu(input_tensor)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::gelu")

    def test_gelu_inplace_runs_on_neuron(self, device):
        """Test that gelu_ runs on Neuron"""
        if not torch.neuron.is_available():
            pytest.skip("Neuron device not available")

        with track_neuron_ops():
            input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device)
            original_data_ptr = input_tensor.data_ptr()
            result = torch.ops.aten.gelu_(input_tensor)
            assert result is input_tensor
            assert result.data_ptr() == original_data_ptr
            assert input_tensor.device.type == "neuron"
            # In-place ops are tracked as regular ops without underscore
            assert_op_runs_on_neuron("gelu")

    def test_gelu_basic(self, device):
        """Test basic gelu operation."""
        input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        # GELU is accessed via torch.nn.functional
        neuron_input = input_tensor.to(device)
        result = torch.nn.functional.gelu(neuron_input)
        assert result.device.type == "neuron"

        # Compare with CPU result
        expected = torch.nn.functional.gelu(input_tensor)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_gelu_2d(self, device):
        """Test gelu on 2D tensor."""
        input_tensor = torch.tensor([[-2.0, -1.0], [0.0, 1.0]])
        neuron_input = input_tensor.to(device)
        result = torch.nn.functional.gelu(neuron_input)
        assert result.device.type == "neuron"
        expected = torch.nn.functional.gelu(input_tensor)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_gelu_approximate_none(self, device):
        """Test gelu with approximate='none' (exact)."""
        input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        neuron_input = input_tensor.to(device)
        result = torch.nn.functional.gelu(neuron_input, approximate="none")
        assert result.device.type == "neuron"
        expected = torch.nn.functional.gelu(input_tensor, approximate="none")
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_gelu_approximate_tanh(self, device):
        """Test gelu with approximate='tanh'."""
        input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        neuron_input = input_tensor.to(device)
        result = torch.nn.functional.gelu(neuron_input, approximate="tanh")
        assert result.device.type == "neuron"
        expected = torch.nn.functional.gelu(input_tensor, approximate="tanh")
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_gelu_different_dtypes(self, device):
        """Test gelu with different data types."""
        for dtype in [torch.float32, torch.float16]:
            input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype)
            neuron_input = input_tensor.to(device)
            result = torch.nn.functional.gelu(neuron_input)
            assert result.device.type == "neuron"
            assert result.dtype == dtype
            expected = torch.nn.functional.gelu(input_tensor)
            torch.testing.assert_close(
                result.cpu(),
                expected,
                rtol=1e-3 if dtype == torch.float16 else 1e-4,
                atol=1e-3 if dtype == torch.float16 else 1e-4,
            )

    def test_gelu_out(self, device):
        """Test gelu.out operation."""
        input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        neuron_input = input_tensor.to(device)
        out = torch.empty_like(neuron_input)
        result = torch.nn.functional.gelu(neuron_input, out=out)
        assert result is out
        assert result.device.type == "neuron"
        cpu_out = torch.empty_like(input_tensor)
        expected = torch.nn.functional.gelu(input_tensor, out=cpu_out)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_gelu_out_with_approximate(self, device):
        """Test gelu.out operation with approximate parameter."""
        input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        neuron_input = input_tensor.to(device)
        out = torch.empty_like(neuron_input)
        result = torch.nn.functional.gelu(neuron_input, approximate="tanh", out=out)
        assert result is out
        assert result.device.type == "neuron"
        cpu_out = torch.empty_like(input_tensor)
        expected = torch.nn.functional.gelu(input_tensor, approximate="tanh", out=cpu_out)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_gelu_inplace(self, device):
        """Test in-place gelu_ operation."""
        input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        neuron_input = input_tensor.to(device)
        original_data_ptr = neuron_input.data_ptr()

        result = torch.ops.aten.gelu_(neuron_input)

        # Check that operation was in-place
        assert result is neuron_input
        assert result.data_ptr() == original_data_ptr

        # Check correctness
        expected = torch.nn.functional.gelu(input_tensor)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_gelu_empty(self, device):
        """Test gelu on empty tensor."""
        input_tensor = torch.tensor([])
        neuron_input = input_tensor.to(device)
        result = torch.nn.functional.gelu(neuron_input)
        assert result.device.type == "neuron"
        expected = torch.nn.functional.gelu(input_tensor)
        torch.testing.assert_close(result.cpu(), expected)

    def test_gelu_large(self, device):
        """Test gelu on larger tensor."""
        input_tensor = torch.randn(100, 100)
        neuron_input = input_tensor.to(device)
        result = torch.nn.functional.gelu(neuron_input)
        assert result.device.type == "neuron"
        expected = torch.nn.functional.gelu(input_tensor)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    # Edge cases
    def test_sqrt_negative_values(self, device):
        """Test sqrt with negative values (should produce NaN)."""
        input_tensor = torch.tensor([-1.0, -4.0, -9.0])
        neuron_input = input_tensor.to(device)
        result = neuron_input.sqrt()
        assert result.device.type == "neuron"
        # Both CPU and Neuron should produce NaN for negative sqrt
        assert torch.isnan(result).all()

    def test_sqrt_zero(self, device):
        """Test sqrt of zero."""
        input_tensor = torch.tensor([0.0])
        result = self._test_unary_op("sqrt", input_tensor, device)
        assert result.cpu().item() == 0.0

    def test_neg_zero(self, device):
        """Test neg of zero."""
        input_tensor = torch.tensor([0.0, -0.0])
        result = self._test_unary_op("neg", input_tensor, device)
        expected = torch.tensor([-0.0, 0.0])
        torch.testing.assert_close(result.cpu(), expected)

    def test_gelu_special_values(self, device):
        """Test gelu with special values."""
        input_tensor = torch.tensor([float("inf"), float("-inf"), float("nan")])
        neuron_input = input_tensor.to(device)
        result = torch.nn.functional.gelu(neuron_input)
        assert result.device.type == "neuron"

        # JAX may handle infinities differently than PyTorch
        # For now, just check that the computation doesn't crash
        # and that NaN input produces NaN output
        assert torch.isnan(result.cpu()[2])  # gelu(nan) = nan

    @assert_raises(RuntimeError, match="approximate argument must be either none or tanh")
    def test_invalid_approximate_value(self, device):
        """Test gelu with invalid approximate value."""
        input_tensor = torch.tensor([1.0, 2.0])
        neuron_input = input_tensor.to(device)

        # PyTorch raises RuntimeError, not ValueError
        torch.nn.functional.gelu(neuron_input, approximate="invalid")

    # Test relu operation
    def test_relu_runs_on_neuron(self, device):
        """Test that relu runs on Neuron"""
        if not torch.neuron.is_available():
            pytest.skip("Neuron device not available")

        with track_neuron_ops():
            input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device)
            result = torch.relu(input_tensor)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::relu")

    def test_relu_inplace_runs_on_neuron(self, device):
        """Test that relu_ runs on Neuron"""
        if not torch.neuron.is_available():
            pytest.skip("Neuron device not available")

        with track_neuron_ops():
            input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device)
            original_data_ptr = input_tensor.data_ptr()
            result = torch.nn.functional.relu_(input_tensor)
            assert result is input_tensor
            assert result.data_ptr() == original_data_ptr
            assert input_tensor.device.type == "neuron"
            # In-place ops are tracked as regular ops without underscore
            assert_op_runs_on_neuron("relu")

    def test_relu_basic(self, device):
        """Test basic relu operation."""
        input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        neuron_input = input_tensor.to(device)
        result = torch.relu(neuron_input)
        assert result.device.type == "neuron"

        # Compare with CPU result
        expected = torch.relu(input_tensor)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_relu_2d(self, device):
        """Test relu on 2D tensor."""
        input_tensor = torch.randn(4, 5)
        neuron_input = input_tensor.to(device)
        result = torch.relu(neuron_input)
        assert result.device.type == "neuron"
        expected = torch.relu(input_tensor)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_relu_nd(self, device):
        """Test relu on tensors with different dimensions."""
        test_shapes = [
            (10,),  # 1D
            (5, 5),  # 2D
            (3, 4, 5),  # 3D
            (2, 3, 4, 5),  # 4D
            (2, 2, 2, 2, 2),  # 5D
        ]

        for shape in test_shapes:
            input_tensor = torch.randn(shape)
            neuron_input = input_tensor.to(device)

            result = torch.relu(neuron_input)
            expected = torch.relu(input_tensor)

            assert result.device.type == "neuron"
            assert result.shape == shape
            torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_relu_inplace(self, device):
        """Test in-place relu_ operation."""
        input_tensor = torch.randn(4, 5)
        neuron_input = input_tensor.to(device)
        neuron_input_2 = neuron_input
        original_data_ptr = neuron_input.data_ptr()

        result = torch.nn.functional.relu_(neuron_input)

        # Check that operation was in-place
        assert result is neuron_input
        assert result.data_ptr() == original_data_ptr

        # Check correctness
        expected = torch.nn.functional.relu(neuron_input_2)
        torch.testing.assert_close(result.cpu(), expected.cpu(), rtol=1e-4, atol=1e-4)

    def test_relu_functional_module(self, device):
        """Test F.relu and nn.ReLU module."""
        input_tensor = torch.randn(4, 5)
        neuron_input = input_tensor.to(device)

        # Test torch.nn.functional.relu
        result_f = torch.nn.functional.relu(neuron_input)
        expected_f = torch.nn.functional.relu(input_tensor)
        torch.testing.assert_close(result_f.cpu(), expected_f, rtol=1e-4, atol=1e-4)

        # Test nn.ReLU module
        relu_module = torch.nn.ReLU().to(device)
        result_module = relu_module(neuron_input)

        relu_module_cpu = torch.nn.ReLU()
        expected_module = relu_module_cpu(input_tensor)

        torch.testing.assert_close(result_module.cpu(), expected_module, rtol=1e-4, atol=1e-4)

    def test_relu_inplace_module(self, device):
        """Test nn.ReLU module with inplace=True."""
        input_tensor = torch.randn(4, 5)
        neuron_input = input_tensor.to(device)

        # Create inplace ReLU module
        relu_module = torch.nn.ReLU(inplace=True).to(device)

        # Apply module
        original_data_ptr = neuron_input.data_ptr()
        result = relu_module(neuron_input)

        # Verify it was in-place
        assert result is neuron_input
        assert result.data_ptr() == original_data_ptr

        # Compare with non-in-place operation
        expected = torch.relu(input_tensor)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_relu_different_dtypes(self, device):
        """Test relu with different data types."""
        for dtype in [torch.float32, torch.float16]:
            input_tensor = torch.randn(4, 5).to(dtype=dtype)
            neuron_input = input_tensor.to(device)

            result = torch.relu(neuron_input)
            expected = torch.relu(input_tensor)

            assert result.dtype == dtype
            torch.testing.assert_close(
                result.cpu(),
                expected,
                rtol=1e-3 if dtype == torch.float16 else 1e-4,
                atol=1e-3 if dtype == torch.float16 else 1e-4,
            )

    def test_relu_large(self, device):
        """Test relu on larger tensor."""
        input_tensor = torch.randn(100, 100)
        neuron_input = input_tensor.to(device)

        result = torch.relu(neuron_input)
        expected = torch.relu(input_tensor)

        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_relu_all_negative(self, device):
        """Test relu on tensor with all negative values."""
        input_tensor = -torch.abs(torch.randn(10, 10))
        neuron_input = input_tensor.to(device)

        result = torch.relu(neuron_input)
        expected = torch.relu(input_tensor)

        # All values should be zero
        assert torch.all(result.cpu() == 0)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_relu_all_positive(self, device):
        """Test relu on tensor with all positive values."""
        input_tensor = torch.abs(torch.randn(10, 10))
        neuron_input = input_tensor.to(device)

        result = torch.relu(neuron_input)
        expected = torch.relu(input_tensor)

        # Result should be identical to input
        torch.testing.assert_close(result.cpu(), input_tensor, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_relu_zeros(self, device):
        """Test relu on tensor with all zeros."""
        input_tensor = torch.zeros(10, 10)
        neuron_input = input_tensor.to(device)

        result = torch.relu(neuron_input)
        expected = torch.relu(input_tensor)

        # All values should remain zero
        assert torch.all(result.cpu() == 0)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.xfail(reason="Relu of NAN is 0 on Neuron")
    def test_relu_special_values(self, device):
        """Test relu with special values."""
        input_tensor = torch.tensor([float("inf"), float("-inf"), float("nan")])
        neuron_input = input_tensor.to(device)

        result = torch.relu(neuron_input)

        # ReLU(inf) = inf, ReLU(-inf) = 0, ReLU(nan) = nan
        assert result.cpu()[0] == float("inf")
        assert result.cpu()[1] == 0
        assert torch.isnan(result.cpu()[2])

    def test_relu_empty(self, device):
        """Test relu on empty tensor."""
        input_tensor = torch.tensor([])
        neuron_input = input_tensor.to(device)

        result = torch.relu(neuron_input)
        expected = torch.relu(input_tensor)

        assert result.shape == torch.Size([0])
        torch.testing.assert_close(result.cpu(), expected)

    def test_relu_backward_compatibility(self, device):
        """Test backward compatibility with older PyTorch versions."""
        input_tensor = torch.randn(4, 5)
        neuron_input = input_tensor.to(device)

        # Different ways to call ReLU
        result1 = torch.relu(neuron_input)
        result2 = torch.nn.functional.relu(neuron_input)
        result3 = torch.nn.ReLU()(neuron_input)

        # All should produce the same result
        torch.testing.assert_close(result1.cpu(), result2.cpu(), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(result1.cpu(), result3.cpu(), rtol=1e-4, atol=1e-4)

    def test_relu_crelu_example(self, device):
        """Test the CReLU example from PyTorch documentation."""
        # Create input tensor
        input_tensor = torch.randn(2).unsqueeze(0)
        neuron_input = input_tensor.to(device)

        # Apply CReLU: concatenate ReLU(x) and ReLU(-x)
        result = torch.cat((torch.relu(neuron_input), torch.relu(-neuron_input)))
        expected = torch.cat((torch.relu(input_tensor), torch.relu(-input_tensor)))

        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    # Test softmax operation
    def test_softmax_basic(self, device):
        """Test basic softmax operation."""
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        neuron_input = input_tensor.to(device)

        result = torch.nn.functional.softmax(neuron_input, dim=0)

        assert result.device.type == "neuron"
        expected = torch.nn.functional.softmax(input_tensor, dim=0)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-3, atol=1e-3)

    def test_softmax_2d_dim_0(self, device):
        """Test softmax on 2D tensor."""
        input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        neuron_input = input_tensor.to(device)

        # Test along dim=0 (columns)
        result_dim0 = torch.nn.functional.softmax(neuron_input, dim=0)
        assert result_dim0.device.type == "neuron"
        expected_dim0 = torch.nn.functional.softmax(input_tensor, dim=0)
        torch.testing.assert_close(result_dim0.cpu(), expected_dim0, rtol=1e-3, atol=1e-3)

    def test_softmax_2d(self, device):
        """Test softmax on 2D tensor."""
        input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        neuron_input = input_tensor.to(device)

        # Test along dim=0 (columns)
        result_dim0 = torch.nn.functional.softmax(neuron_input, dim=0)
        assert result_dim0.device.type == "neuron"
        expected_dim0 = torch.nn.functional.softmax(input_tensor, dim=0)
        torch.testing.assert_close(result_dim0.cpu(), expected_dim0, rtol=1e-3, atol=1e-3)

        # Test along dim=1 (rows)
        result_dim1 = torch.nn.functional.softmax(neuron_input, dim=1)
        assert result_dim1.device.type == "neuron"
        expected_dim1 = torch.nn.functional.softmax(input_tensor, dim=1)
        torch.testing.assert_close(result_dim1.cpu(), expected_dim1, rtol=1e-3, atol=1e-3)

    def test_softmax_3d(self, device):
        """Test softmax on 3D tensor."""
        input_tensor = torch.randn(2, 3, 4)
        neuron_input = input_tensor.to(device)

        # Test along different dimensions
        for dim in range(3):
            result = torch.nn.functional.softmax(neuron_input, dim=dim)
            assert result.device.type == "neuron"
            expected = torch.nn.functional.softmax(input_tensor, dim=dim)
            torch.testing.assert_close(result.cpu(), expected, rtol=1e-3, atol=1e-3)

    def test_softmax_half_to_float_bug(self, device):
        """Test softmax half_to_float"""
        input_tensor = torch.randn(2, 3, dtype=torch.float16)
        neuron_input = input_tensor.to(device)
        result = torch.ops.aten._softmax.default(neuron_input, dim=1, half_to_float=True)

        assert result.device.type == "neuron"
        assert result.dtype == torch.float32

        # Compute expected manually given softmax with half to float is not supported on CPU
        input_float32 = input_tensor.float()
        expected = torch.nn.functional.softmax(input_float32, dim=1)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-3, atol=1e-3)

    def test_softmax_module(self, device):
        """Test torch.nn.Softmax module."""
        input_tensor = torch.randn(2, 3, 4)
        neuron_input = input_tensor.to(device)

        # Test with different dimensions
        for dim in range(3):
            softmax_module = torch.nn.Softmax(dim=dim).to(device)
            result = softmax_module(neuron_input)

            softmax_module_cpu = torch.nn.Softmax(dim=dim)
            expected = softmax_module_cpu(input_tensor)

            assert result.device.type == "neuron"
            torch.testing.assert_close(result.cpu(), expected, rtol=1e-3, atol=1e-3)

    def test_softmax_large(self, device):
        """Test softmax on larger tensor."""
        input_tensor = torch.randn(50, 50)
        neuron_input = input_tensor.to(device)

        result = torch.nn.functional.softmax(neuron_input, dim=1)
        expected = torch.nn.functional.softmax(input_tensor, dim=1)

        assert result.device.type == "neuron"
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-3, atol=1e-3)

    def test_softmax_special_values(self, device):
        """Test softmax with special values."""
        # Create tensor with special values
        input_tensor = torch.tensor([float("inf"), 0.0, float("-inf")])
        neuron_input = input_tensor.to(device)

        result = torch.nn.functional.softmax(neuron_input, dim=0)
        expected = torch.nn.functional.softmax(input_tensor, dim=0)

        assert result.device.type == "neuron"
        # Softmax of all special values is NAN
        assert torch.isnan(result).all()
        assert torch.isnan(expected).all()

    def test_softmax_stability(self, device):
        """Test softmax numerical stability with large values."""
        # Large values that could cause overflow in naive implementation
        input_tensor = torch.tensor([1000.0, 0.0, -1000.0])
        neuron_input = input_tensor.to(device)

        result = torch.nn.functional.softmax(neuron_input, dim=0)
        expected = torch.nn.functional.softmax(input_tensor, dim=0)

        assert result.device.type == "neuron"
        # Should be approximately [1, 0, 0] due to numerical stability in implementation
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-3, atol=1e-3)

    # Test log_softmax operation
    def test_log_softmax_basic(self, device):
        """Test basic log_softmax operation."""
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        neuron_input = input_tensor.to(device)

        result = torch.nn.functional.log_softmax(neuron_input, dim=0)

        assert result.device.type == "neuron"
        expected = torch.nn.functional.log_softmax(input_tensor, dim=0)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-3, atol=1e-3)

    def test_log_softmax_2d(self, device):
        """Test log_softmax on 2D tensor."""
        input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        neuron_input = input_tensor.to(device)

        # Test along dim=0 (columns)
        result_dim0 = torch.nn.functional.log_softmax(neuron_input, dim=0)
        assert result_dim0.device.type == "neuron"
        expected_dim0 = torch.nn.functional.log_softmax(input_tensor, dim=0)
        torch.testing.assert_close(result_dim0.cpu(), expected_dim0, rtol=1e-3, atol=1e-3)

        # Test along dim=1 (rows)
        result_dim1 = torch.nn.functional.log_softmax(neuron_input, dim=1)
        assert result_dim1.device.type == "neuron"
        expected_dim1 = torch.nn.functional.log_softmax(input_tensor, dim=1)
        torch.testing.assert_close(result_dim1.cpu(), expected_dim1, rtol=1e-3, atol=1e-3)

    def test_log_softmax_3d(self, device):
        """Test log_softmax on 3D tensor."""
        input_tensor = torch.randn(2, 3, 4)
        neuron_input = input_tensor.to(device)

        # Test along different dimensions
        for dim in range(3):
            result = torch.nn.functional.log_softmax(neuron_input, dim=dim)
            assert result.device.type == "neuron"
            expected = torch.nn.functional.log_softmax(input_tensor, dim=dim)
            torch.testing.assert_close(result.cpu(), expected, rtol=1e-3, atol=1e-3)

    def test_log_softmax_with_dtype(self, device):
        """Test log_softmax with different dtypes."""
        input_tensor = torch.randn(2, 3)
        neuron_input = input_tensor.to(device)

        # Test with float32
        result_float32 = torch.nn.functional.log_softmax(neuron_input, dim=1, dtype=torch.float32)
        assert result_float32.device.type == "neuron"
        assert result_float32.dtype == torch.float32
        expected_float32 = torch.nn.functional.log_softmax(input_tensor, dim=1, dtype=torch.float32)
        torch.testing.assert_close(result_float32.cpu(), expected_float32, rtol=1e-3, atol=1e-3)

        # Test with float16
        if torch.float16 in [torch.float16, torch.float32]:  # Check if float16 is supported
            result_float16 = torch.nn.functional.log_softmax(
                neuron_input, dim=1, dtype=torch.float16
            )
            assert result_float16.device.type == "neuron"
            assert result_float16.dtype == torch.float16
            expected_float16 = torch.nn.functional.log_softmax(
                input_tensor, dim=1, dtype=torch.float16
            )
            torch.testing.assert_close(result_float16.cpu(), expected_float16, rtol=1e-2, atol=1e-2)

    def test_log_softmax_module(self, device):
        """Test torch.nn.LogSoftmax module."""
        input_tensor = torch.randn(2, 3, 4)
        neuron_input = input_tensor.to(device)

        # Test with different dimensions
        for dim in range(3):
            log_softmax_module = torch.nn.LogSoftmax(dim=dim).to(device)
            result = log_softmax_module(neuron_input)

            log_softmax_module_cpu = torch.nn.LogSoftmax(dim=dim)
            expected = log_softmax_module_cpu(input_tensor)

            assert result.device.type == "neuron"
            torch.testing.assert_close(result.cpu(), expected, rtol=1e-3, atol=1e-3)

    def test_log_softmax_large(self, device):
        """Test log_softmax on larger tensor."""
        input_tensor = torch.randn(50, 50)
        neuron_input = input_tensor.to(device)

        result = torch.nn.functional.log_softmax(neuron_input, dim=1)
        expected = torch.nn.functional.log_softmax(input_tensor, dim=1)

        assert result.device.type == "neuron"
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-3, atol=1e-3)

    def test_log_softmax_special_values(self, device):
        """Test log_softmax with special values."""
        # Create tensor with special values
        input_tensor = torch.tensor([float("inf"), 0.0, float("-inf")])
        neuron_input = input_tensor.to(device)

        result = torch.nn.functional.log_softmax(neuron_input, dim=0)
        expected = torch.nn.functional.log_softmax(input_tensor, dim=0)

        assert result.device.type == "neuron"
        # log_softmax of special values should produce NaN
        assert torch.isnan(result).all()
        assert torch.isnan(expected).all()

    def test_log_softmax_stability(self, device):
        """Test log_softmax numerical stability with large values."""
        # Large values that could cause overflow in naive implementation
        input_tensor = torch.tensor([1000.0, 0.0, -1000.0])
        neuron_input = input_tensor.to(device)

        result = torch.nn.functional.log_softmax(neuron_input, dim=0)
        expected = torch.nn.functional.log_softmax(input_tensor, dim=0)

        assert result.device.type == "neuron"
        # Should be approximately [0, -1000, -2000] due to numerical stability in implementation
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-3, atol=1e-3)

    def test_log_softmax_vs_softmax(self, device):
        """Test that log_softmax matches log(softmax)."""
        input_tensor = torch.randn(3, 4)
        neuron_input = input_tensor.to(device)

        # Direct log_softmax
        log_softmax_result = torch.nn.functional.log_softmax(neuron_input, dim=1)

        # log(softmax)
        softmax_result = torch.nn.functional.softmax(neuron_input, dim=1)
        log_of_softmax = torch.log(softmax_result)

        # They should be very close, with log_softmax being more numerically stable
        torch.testing.assert_close(
            log_softmax_result.cpu(), log_of_softmax.cpu(), rtol=1e-3, atol=1e-3
        )

    # Test softplus operation
    @pytest.mark.skipif(
        not use_mlir_aten_ops(), reason="aten::softplus only supported using dynamo decompositions"
    )
    def test_softplus_runs_on_neuron(self, device):
        """Test that softplus runs on Neuron"""
        if not torch.neuron.is_available():
            pytest.skip("Neuron device not available")

        with track_neuron_ops():
            input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device)
            result = torch.nn.functional.softplus(input_tensor)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::softplus")

    @pytest.mark.skipif(
        not use_mlir_aten_ops(), reason="aten::softplus only supported using dynamo decompositions"
    )
    def test_softplus_basic(self, device):
        """Test basic softplus operation."""
        input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        neuron_input = input_tensor.to(device)

        result = torch.nn.functional.softplus(neuron_input)

        assert result.device.type == "neuron"
        expected = torch.nn.functional.softplus(input_tensor)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.skipif(
        not use_mlir_aten_ops(), reason="aten::softplus only supported using dynamo decompositions"
    )
    def test_softplus_2d(self, device):
        """Test softplus on 2D tensor."""
        input_tensor = torch.randn(4, 5)
        neuron_input = input_tensor.to(device)

        result = torch.nn.functional.softplus(neuron_input)

        assert result.device.type == "neuron"
        expected = torch.nn.functional.softplus(input_tensor)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.skipif(
        not use_mlir_aten_ops(), reason="aten::softplus only supported using dynamo decompositions"
    )
    def test_softplus_3d(self, device):
        """Test softplus on 3D tensor."""
        input_tensor = torch.randn(2, 3, 4)
        neuron_input = input_tensor.to(device)

        result = torch.nn.functional.softplus(neuron_input)

        assert result.device.type == "neuron"
        expected = torch.nn.functional.softplus(input_tensor)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.skipif(
        not use_mlir_aten_ops(), reason="aten::softplus only supported using dynamo decompositions"
    )
    def test_softplus_nd(self, device):
        """Test softplus on tensors with different dimensions."""
        test_shapes = [
            (10,),  # 1D
            (5, 5),  # 2D
            (3, 4, 5),  # 3D
            (2, 3, 4, 5),  # 4D
        ]

        for shape in test_shapes:
            input_tensor = torch.randn(shape)
            neuron_input = input_tensor.to(device)

            result = torch.nn.functional.softplus(neuron_input)
            expected = torch.nn.functional.softplus(input_tensor)

            assert result.device.type == "neuron"
            assert result.shape == shape
            torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.skipif(
        not use_mlir_aten_ops(), reason="aten::softplus only supported using dynamo decompositions"
    )
    def test_softplus_with_beta(self, device):
        """Test softplus with different beta values."""
        input_tensor = torch.randn(4, 5)
        neuron_input = input_tensor.to(device)

        for beta in [0.5, 1.0, 2.0, 5.0]:
            result = torch.nn.functional.softplus(neuron_input, beta=beta)

            assert result.device.type == "neuron"
            expected = torch.nn.functional.softplus(input_tensor, beta=beta)
            torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.skipif(
        not use_mlir_aten_ops(), reason="aten::softplus only supported using dynamo decompositions"
    )
    def test_softplus_with_threshold(self, device):
        """Test softplus with different threshold values."""
        input_tensor = torch.randn(4, 5) * 10  # Larger values to test threshold
        neuron_input = input_tensor.to(device)

        for threshold in [10.0, 20.0, 50.0]:
            result = torch.nn.functional.softplus(neuron_input, threshold=threshold)

            assert result.device.type == "neuron"
            expected = torch.nn.functional.softplus(input_tensor, threshold=threshold)
            torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.skipif(
        not use_mlir_aten_ops(), reason="aten::softplus only supported using dynamo decompositions"
    )
    def test_softplus_with_beta_and_threshold(self, device):
        """Test softplus with both beta and threshold parameters."""
        input_tensor = torch.randn(4, 5) * 5
        neuron_input = input_tensor.to(device)

        result = torch.nn.functional.softplus(neuron_input, beta=2.0, threshold=15.0)

        assert result.device.type == "neuron"
        expected = torch.nn.functional.softplus(input_tensor, beta=2.0, threshold=15.0)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.skipif(
        not use_mlir_aten_ops(), reason="aten::softplus only supported using dynamo decompositions"
    )
    def test_softplus_different_dtypes(self, device):
        """Test softplus with different data types."""
        for dtype in [torch.float32, torch.float16]:
            input_tensor = torch.randn(4, 5, dtype=dtype)
            neuron_input = input_tensor.to(device)

            result = torch.nn.functional.softplus(neuron_input)
            expected = torch.nn.functional.softplus(input_tensor)

            assert result.device.type == "neuron"
            assert result.dtype == dtype
            torch.testing.assert_close(
                result.cpu(),
                expected,
                rtol=1e-3 if dtype == torch.float16 else 1e-4,
                atol=1e-3 if dtype == torch.float16 else 1e-4,
            )

    @pytest.mark.skipif(
        not use_mlir_aten_ops(), reason="aten::softplus only supported using dynamo decompositions"
    )
    def test_softplus_module(self, device):
        """Test torch.nn.Softplus module."""
        input_tensor = torch.randn(4, 5)
        neuron_input = input_tensor.to(device)

        # Test with default parameters
        softplus_module = torch.nn.Softplus().to(device)
        result = softplus_module(neuron_input)

        softplus_module_cpu = torch.nn.Softplus()
        expected = softplus_module_cpu(input_tensor)

        assert result.device.type == "neuron"
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.skipif(
        not use_mlir_aten_ops(), reason="aten::softplus only supported using dynamo decompositions"
    )
    def test_softplus_module_with_params(self, device):
        """Test torch.nn.Softplus module with beta and threshold."""
        input_tensor = torch.randn(4, 5) * 5
        neuron_input = input_tensor.to(device)

        # Test with custom parameters
        softplus_module = torch.nn.Softplus(beta=2.0, threshold=15.0).to(device)
        result = softplus_module(neuron_input)

        softplus_module_cpu = torch.nn.Softplus(beta=2.0, threshold=15.0)
        expected = softplus_module_cpu(input_tensor)

        assert result.device.type == "neuron"
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.skipif(
        not use_mlir_aten_ops(), reason="aten::softplus only supported using dynamo decompositions"
    )
    def test_softplus_large(self, device):
        """Test softplus on larger tensor."""
        input_tensor = torch.randn(100, 100)
        neuron_input = input_tensor.to(device)

        result = torch.nn.functional.softplus(neuron_input)
        expected = torch.nn.functional.softplus(input_tensor)

        assert result.device.type == "neuron"
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.skipif(
        not use_mlir_aten_ops(), reason="aten::softplus only supported using dynamo decompositions"
    )
    def test_softplus_all_negative(self, device):
        """Test softplus on tensor with all negative values."""
        input_tensor = -torch.abs(torch.randn(10, 10)) - 1.0
        neuron_input = input_tensor.to(device)

        result = torch.nn.functional.softplus(neuron_input)
        expected = torch.nn.functional.softplus(input_tensor)

        assert result.device.type == "neuron"
        # Softplus of negative values should be small positive numbers
        assert torch.all(result.cpu() > 0)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.skipif(
        not use_mlir_aten_ops(), reason="aten::softplus only supported using dynamo decompositions"
    )
    def test_softplus_all_positive(self, device):
        """Test softplus on tensor with all positive values."""
        input_tensor = torch.abs(torch.randn(10, 10)) + 1.0
        neuron_input = input_tensor.to(device)

        result = torch.nn.functional.softplus(neuron_input)
        expected = torch.nn.functional.softplus(input_tensor)

        assert result.device.type == "neuron"
        # For large positive values, softplus(x) ≈ x
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.skipif(
        not use_mlir_aten_ops(), reason="aten::softplus only supported using dynamo decompositions"
    )
    def test_softplus_zeros(self, device):
        """Test softplus on tensor with all zeros."""
        input_tensor = torch.zeros(10, 10)
        neuron_input = input_tensor.to(device)

        result = torch.nn.functional.softplus(neuron_input)
        expected = torch.nn.functional.softplus(input_tensor)

        assert result.device.type == "neuron"
        # softplus(0) = ln(2) ≈ 0.6931
        expected_value = math.log(2)
        torch.testing.assert_close(
            result.cpu(), torch.full_like(input_tensor, expected_value), rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.skipif(
        not use_mlir_aten_ops(), reason="aten::softplus only supported using dynamo decompositions"
    )
    def test_softplus_empty(self, device):
        """Test softplus on empty tensor."""
        input_tensor = torch.tensor([])
        neuron_input = input_tensor.to(device)

        result = torch.nn.functional.softplus(neuron_input)
        expected = torch.nn.functional.softplus(input_tensor)

        assert result.device.type == "neuron"
        assert result.shape == torch.Size([0])
        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.skipif(
        not use_mlir_aten_ops(), reason="aten::softplus only supported using dynamo decompositions"
    )
    def test_softplus_threshold_behavior(self, device):
        """Test softplus threshold behavior (returns x when beta*x > threshold)."""
        # For default beta=1 and threshold=20, when x > 20, softplus(x) ≈ x
        input_tensor = torch.tensor([25.0, 30.0, 50.0, 100.0])
        neuron_input = input_tensor.to(device)

        result = torch.nn.functional.softplus(neuron_input)
        expected = torch.nn.functional.softplus(input_tensor)

        assert result.device.type == "neuron"
        # For large values, result should be approximately equal to input
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(result.cpu(), input_tensor, rtol=1e-4, atol=1e-4)

    @pytest.mark.skipif(
        not use_mlir_aten_ops(), reason="aten::softplus only supported using dynamo decompositions"
    )
    def test_softplus_mathematical_definition(self, device):
        """Test softplus against mathematical definition: softplus(x) = ln(1 + exp(x))."""
        input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        neuron_input = input_tensor.to(device)

        result = torch.nn.functional.softplus(neuron_input)

        # Manual calculation: ln(1 + exp(x))
        expected_manual = torch.log(1 + torch.exp(input_tensor))

        assert result.device.type == "neuron"
        torch.testing.assert_close(result.cpu(), expected_manual, rtol=1e-4, atol=1e-4)

    @pytest.mark.skipif(
        not use_mlir_aten_ops(), reason="aten::softplus only supported using dynamo decompositions"
    )
    def test_softplus_backward_compatibility(self, device):
        """Test backward compatibility with different ways to call softplus."""
        input_tensor = torch.randn(4, 5)
        neuron_input = input_tensor.to(device)

        # Different ways to call softplus
        result1 = torch.nn.functional.softplus(neuron_input)
        result2 = torch.nn.Softplus()(neuron_input)

        # All should produce the same result
        torch.testing.assert_close(result1.cpu(), result2.cpu(), rtol=1e-4, atol=1e-4)
