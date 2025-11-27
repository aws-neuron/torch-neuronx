import pytest
import torch
import torch.nn.functional as f

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

UNARY_OPS_LIST = [
    torch.log,
    torch.sigmoid,
    torch.sin,
    torch.cos,
    torch.exp,
    torch.abs,
    f.silu,
    torch.ceil,
    torch.trunc,
    torch.sqrt,
    torch.rsqrt,
    torch.logical_not,
    torch.erf,
    torch.erfinv,
]
UNARY_OPS_OUT_LIST = [
    torch.log,
    torch.sigmoid,
    torch.sin,
    torch.cos,
    torch.exp,
    torch.abs,
    torch.ceil,
    torch.trunc,
    torch.sqrt,
    torch.rsqrt,
    torch.logical_not,
    torch.erf,
    torch.erfinv,
]
UNARY_INPLACE_OPS_LIST = [
    "log_",
    "sigmoid_",
    "sin_",
    "cos_",
    "exp_",
    "abs_",
    "ceil_",
    "trunc_",
    "logical_not_",
]

# TODO;Find a better way to register xfail cases for individual operators
UNARY_ZERO_XFAIL_LIST = {
    torch.rsqrt: pytest.param(
        torch.rsqrt,
        marks=pytest.mark.xfail(reason="Neuron return nan while CPU returns inf"),
    ),
}
UNARY_SPECIAL_VALUE_XFAIL_LIST = {
    torch.ceil: pytest.param(
        torch.ceil,
        marks=pytest.mark.xfail(reason="-2.1475e+09 is returned when input is inf, -inf or nan"),
    ),
    torch.trunc: pytest.param(
        torch.trunc,
        marks=pytest.mark.xfail(reason="-2.1475e+09 is returned when input is inf, -inf or nan"),
    ),
}
DTYPES = [torch.float32, torch.float16, torch.int32, torch.int64]
UNSUPPORTED_OP_DTYPE_LIST = [(f.silu, torch.int32), (f.silu, torch.int64)]

UNARY_DTYPE_LIST = {
    "default": [torch.float32, torch.float16],
    torch.logical_not: [torch.bool],  # logical_not returns bool unless specified otherwise
}


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestUnaryElementwiseOp:
    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.mark.parametrize("op", UNARY_OPS_LIST)
    def test_unary_runs_on_neuron(self, op, device):
        """Test unary operator runs on Neuron"""
        with track_neuron_ops():
            input_tensor = torch.tensor([0.0, 1.57, 3.14, -2.27, -4], device=device)
            result = op(input_tensor)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron(f"aten::{op.__name__}")

    @pytest.mark.parametrize("op_name", UNARY_INPLACE_OPS_LIST)
    def test_unary_inplace_runs_on_neuron(self, op_name, device):
        """Test unary inplace operator runs on Neuron"""
        with track_neuron_ops():
            input_tensor = torch.tensor([0.0, 1.57, 3.14, -2.27, -4], device=device)
            original_data_ptr = input_tensor.data_ptr()
            result = getattr(input_tensor, op_name)()
            assert result is input_tensor
            assert result.data_ptr() == original_data_ptr
            assert input_tensor.device.type == "neuron"
            # In-place ops are tracked as regular ops without underscore
            assert_op_runs_on_neuron(op_name.strip("_"))

    @pytest.mark.parametrize("op", UNARY_OPS_LIST)
    def test_unary_basic(self, op):
        """Test unary operator with 1D tensor"""
        with track_neuron_ops():
            input_tensor = torch.tensor([2.0, 3.0, 4.0, 5.0])
            input_tensor_device = input_tensor.to("neuron")

            result = op(input_tensor_device)
            expected = op(input_tensor)
            assert_op_runs_on_neuron(f"aten::{op.__name__}")

        torch.testing.assert_close(
            result.cpu(),
            expected,
            rtol=1e-4,
            atol=1e-3,
            equal_nan=True,
        )

    @pytest.mark.parametrize("op", UNARY_OPS_OUT_LIST)
    def test_unary_out(self, op):
        """Test unary operator with preallocated out"""
        with track_neuron_ops():
            input_tensor = torch.tensor([2.0, 3.0, 4.0, 5.0])
            expected = torch.empty_like(input_tensor)

            input_tensor_device = input_tensor.to("neuron")
            output = torch.empty_like(input_tensor_device)

            op(input_tensor, out=expected)
            op(input_tensor_device, out=output)
            assert_op_runs_on_neuron(f"aten::{op.__name__}")

        torch.testing.assert_close(
            output.cpu(),
            expected,
            rtol=1e-4,
            atol=1e-3,
            equal_nan=True,
        )

    @pytest.mark.parametrize("op_name", UNARY_INPLACE_OPS_LIST)
    def test_unary_inplace(self, op_name):
        """Test unary inplace operator"""
        with track_neuron_ops():
            input_tensor = torch.tensor([2.0, 3.0, 4.0, 5.0])

            input_tensor_device = input_tensor.to("neuron")

            getattr(input_tensor, op_name)()
            getattr(input_tensor_device, op_name)()
            assert_op_runs_on_neuron(op_name.strip("_"))

        torch.testing.assert_close(
            input_tensor_device.cpu(),
            input_tensor,
            rtol=1e-4,
            atol=1e-3,
        )

    @pytest.mark.parametrize("op", UNARY_OPS_LIST)
    def test_unary_2d(self, op):
        """Test unary operator with 2D tensor"""
        with track_neuron_ops():
            input_tensor = torch.randn(3, 3)
            input_tensor_device = input_tensor.to("neuron")

            result = op(input_tensor_device)
            expected = op(input_tensor)
            assert_op_runs_on_neuron(f"aten::{op.__name__}")

        torch.testing.assert_close(
            result.cpu(),
            expected,
            rtol=1e-4,
            atol=1e-3,
            equal_nan=True,
        )

    @pytest.mark.parametrize(
        "op,dtype",
        [
            (op, dtype)
            for op in UNARY_OPS_LIST
            for dtype in DTYPES
            if (op, dtype) not in UNSUPPORTED_OP_DTYPE_LIST
        ],
    )
    def test_unary_different_dtypes(self, op, dtype, device):
        """Test unary operator with different data types."""
        dtypes = UNARY_DTYPE_LIST.get(op, UNARY_DTYPE_LIST["default"])
        for dtype in dtypes:
            input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype)
            input_tensor_device = input_tensor.to(device)

            result = op(input_tensor_device)
            expected = op(input_tensor)
            assert_op_runs_on_neuron(f"aten::{op.__name__}")

        assert result.device.type == "neuron"
        assert result.dtype == expected.dtype
        torch.testing.assert_close(
            result.cpu(),
            expected,
            rtol=1e-4,
            atol=1e-3,
            equal_nan=True,
        )

    @pytest.mark.parametrize("op", UNARY_OPS_LIST)
    def test_unary_large(self, op, device):
        """Test unary operator on larger tensor."""
        with track_neuron_ops():
            input_tensor = torch.randn(100, 100)
            input_tensor_device = input_tensor.to(device)

            result = op(input_tensor_device)
            expected = op(input_tensor)
            assert_op_runs_on_neuron(f"aten::{op.__name__}")

        assert result.device.type == "neuron"
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-3, equal_nan=True)

    @pytest.mark.parametrize("op", UNARY_OPS_LIST)
    def test_unary_empty(self, op, device):
        """Test unary operator on empty tensor."""
        with track_neuron_ops():
            input_tensor = torch.tensor([])
            input_tensor_device = input_tensor.to(device)

            result = op(input_tensor_device)
            expected = op(input_tensor)
            assert_op_runs_on_neuron(f"aten::{op.__name__}")

        assert result.device.type == "neuron"
        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize(
        "op", [UNARY_SPECIAL_VALUE_XFAIL_LIST.get(op, op) for op in UNARY_OPS_LIST]
    )
    def test_unary_special_values(self, op, device):
        """Test unary operator with special values."""
        with track_neuron_ops():
            input_tensor = torch.tensor([float("inf"), float("-inf"), float("nan")])
            input_tensor_device = input_tensor.to(device)

            result = op(input_tensor_device)
            expected = op(input_tensor)
            # Handle parametrized ops that might be wrapped in pytest.param
            op_name = op.__name__ if hasattr(op, "__name__") else op.values[0].__name__
            assert_op_runs_on_neuron(f"aten::{op_name}")

        # Check that the operation doesn't crash with special values
        # For specific behavior, we'd need to check each operation individually
        assert result.device.type == "neuron"

        # Check that NaN inputs produce NaN outputs
        # This is a common behavior for most operations
        if torch.isnan(expected[2]):
            assert torch.isnan(result.cpu()[2])

    @pytest.mark.parametrize("op", UNARY_OPS_LIST)
    def test_unary_nd(self, op, device):
        """Test unary operator on tensors with different dimensions."""
        test_shapes = [
            (10,),  # 1D
            (5, 5),  # 2D
            (3, 4, 5),  # 3D
            (2, 3, 4, 5),  # 4D
        ]

        with track_neuron_ops():
            for shape in test_shapes:
                input_tensor = torch.randn(shape)
                input_tensor_device = input_tensor.to(device)

                result = op(input_tensor_device)
                expected = op(input_tensor)

                assert result.device.type == "neuron"
                assert result.shape == shape
                torch.testing.assert_close(
                    result.cpu(), expected, rtol=1e-4, atol=1e-3, equal_nan=True
                )
            assert_op_runs_on_neuron(f"aten::{op.__name__}")

    @pytest.mark.parametrize("op_name", UNARY_INPLACE_OPS_LIST)
    def test_unary_inplace_data_ptr(self, op_name, device):
        """Test that in-place operations preserve the data pointer."""
        with track_neuron_ops():
            input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
            input_tensor_device = input_tensor.to(device)
            original_data_ptr = input_tensor_device.data_ptr()

            result = getattr(input_tensor_device, op_name)()
            assert_op_runs_on_neuron(op_name.strip("_"))

        # Check that operation was in-place
        assert result is input_tensor_device
        assert result.data_ptr() == original_data_ptr

    @pytest.mark.parametrize("op", [UNARY_ZERO_XFAIL_LIST.get(op, op) for op in UNARY_OPS_LIST])
    def test_unary_zero(self, op, device):
        """Test unary operator with zero input."""
        with track_neuron_ops():
            input_tensor = torch.tensor([0.0])
            input_tensor_device = input_tensor.to(device)

            result = op(input_tensor_device)
            expected = op(input_tensor)
            assert_op_runs_on_neuron(f"aten::{op.__name__}")

        assert result.device.type == "neuron"
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-3, equal_nan=True)

    # Specific tests for torch.exp
    def test_exp_specific(self, device):
        """Test specific properties of exponential function."""
        with track_neuron_ops():
            # Test e^0 = 1
            input_tensor = torch.tensor([0.0])
            input_tensor_device = input_tensor.to(device)
            result = torch.exp(input_tensor_device)
            assert abs(result.cpu().item() - 1.0) < 1e-3

            # Test e^1 ≈ 2.718281828459045
            input_tensor = torch.tensor([1.0])
            input_tensor_device = input_tensor.to(device)
            result = torch.exp(input_tensor_device)
            assert abs(result.cpu().item() - 2.718281828459045) < 1e-3

            # Test that exp is always positive for finite inputs
            input_tensor = torch.randn(10)
            input_tensor_device = input_tensor.to(device)
            result = torch.exp(input_tensor_device)
            assert torch.all(result.cpu() > 0)
            assert_op_runs_on_neuron("aten::exp")

    # Specific tests for torch.abs
    def test_abs_specific(self, device):
        """Test specific properties of absolute value function."""
        with track_neuron_ops():
            # Test abs(-x) = abs(x)
            input_tensor = torch.randn(10)
            input_tensor_device = input_tensor.to(device)
            result_pos = torch.abs(input_tensor_device)
            result_neg = torch.abs(-input_tensor_device)
            torch.testing.assert_close(result_pos.cpu(), result_neg.cpu(), rtol=1e-4, atol=1e-3)

            # Test abs preserves zeros
            input_tensor = torch.tensor([0.0, -0.0])
            input_tensor_device = input_tensor.to(device)
            result = torch.abs(input_tensor_device)
            assert result.cpu()[0].item() == 0.0
            assert result.cpu()[1].item() == 0.0

            # Test abs with mixed positive and negative values
            input_tensor = torch.tensor([-5.0, -3.0, 0.0, 3.0, 5.0])
            input_tensor_device = input_tensor.to(device)
            result = torch.abs(input_tensor_device)
            expected = torch.tensor([5.0, 3.0, 0.0, 3.0, 5.0])
            torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-3)
            assert_op_runs_on_neuron("aten::abs")

    def test_silu_inplace_module(self, device):
        """Test torch.nn.functional.silu module with inplace=True."""
        with track_neuron_ops():
            input_tensor = torch.randn(4, 5)
            neuron_input = input_tensor.to(device)

            # Create inplace Silu module
            result = f.silu(neuron_input, inplace=True).to(device)

            # Apply module
            original_data_ptr = neuron_input.data_ptr()

            # Verify it was in-place
            assert result is neuron_input
            assert result.data_ptr() == original_data_ptr
            assert_op_runs_on_neuron("aten::silu")

        # Compare with non-in-place operation
        expected = f.silu(input_tensor)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-3)

    # Specific tests for torch.silu
    def test_silu_specific(self, device):
        """Test specific properties of SiLU function."""
        with track_neuron_ops():
            # Test silu(0) = 0
            input_tensor = torch.tensor([0.0])
            input_tensor_device = input_tensor.to(device)
            result = f.silu(input_tensor_device)
            assert result.cpu().item() == 0.0

            # Test silu is equivalent to x * sigmoid(x)
            input_tensor = torch.randn(10)
            input_tensor_device = input_tensor.to(device)
            result = f.silu(input_tensor_device)

            # Calculate x * sigmoid(x) manually
            sigmoid_x = torch.sigmoid(input_tensor_device)
            manual_silu = input_tensor_device * sigmoid_x

            # Both should give the same result
            torch.testing.assert_close(result.cpu(), manual_silu.cpu(), rtol=1e-4, atol=1e-3)

            # Test silu with f.silu
            result2 = f.silu(input_tensor_device)
            torch.testing.assert_close(result.cpu(), result2.cpu(), rtol=1e-4, atol=1e-3)
            assert_op_runs_on_neuron("aten::silu")
