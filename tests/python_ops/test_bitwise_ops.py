import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

# Define test data for different operation types
UNARY_BITWISE_OPS = [
    (torch.bitwise_not, "aten::bitwise_not", [5, 10, 15]),
]

BINARY_BITWISE_OPS = [
    (torch.bitwise_and, "aten::bitwise_and", [5, 10, 15], [3, 6, 9]),
    (torch.bitwise_or, "aten::bitwise_or", [5, 10, 15], [3, 6, 9]),
    (torch.bitwise_xor, "aten::bitwise_xor", [5, 10, 15], [3, 6, 9]),
]

SHIFT_OPS = [
    (torch.bitwise_left_shift, "aten::bitwise_left_shift", [1, 2, 4], [1, 2, 3]),
    (torch.bitwise_right_shift, "aten::bitwise_right_shift", [8, 16, 32], [1, 2, 3]),
]

UNARY_BITWISE_INPLACE_OPS = [
    ("bitwise_not_", "aten::bitwise_not", [5, 10, 15]),
]

BINARY_BITWISE_INPLACE_OPS = [
    ("bitwise_and_", "aten::bitwise_and", [5, 10, 15], [3, 6, 9]),
    ("bitwise_or_", "aten::bitwise_or", [5, 10, 15], [3, 6, 9]),
    ("bitwise_xor_", "aten::bitwise_xor", [5, 10, 15], [3, 6, 9]),
]

SHIFT_INPLACE_OPS = [
    ("bitwise_left_shift_", "aten::bitwise_left_shift", [1, 2, 4], [1, 2, 3]),
    ("bitwise_right_shift_", "aten::bitwise_right_shift", [8, 16, 32], [1, 2, 3]),
]


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestBitwiseOps:
    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    ###########################
    # Test basic on device
    ###########################
    @pytest.mark.parametrize("op_func,op_name,test_data", UNARY_BITWISE_OPS)
    def test_unary_bitwise_runs_on_neuron(self, device, op_func, op_name, test_data):
        """Test that unary bitwise operations run on Neuron"""
        with track_neuron_ops():
            input_tensor = torch.tensor(test_data, dtype=torch.int32, device=device)
            result = op_func(input_tensor)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron(op_name)

    @pytest.mark.parametrize("op_func,op_name,input1_data,input2_data", BINARY_BITWISE_OPS)
    def test_binary_bitwise_runs_on_neuron(
        self, device, op_func, op_name, input1_data, input2_data
    ):
        """Test that binary bitwise operations run on Neuron"""
        with track_neuron_ops():
            input1 = torch.tensor(input1_data, dtype=torch.int32, device=device)
            input2 = torch.tensor(input2_data, dtype=torch.int32, device=device)
            result = op_func(input1, input2)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron(op_name)

    @pytest.mark.parametrize("op_func,op_name,input_data,shift_data", SHIFT_OPS)
    def test_shift_ops_run_on_neuron(self, device, op_func, op_name, input_data, shift_data):
        """Test that shift operations run on Neuron"""
        with track_neuron_ops():
            input_tensor = torch.tensor(input_data, dtype=torch.int32, device=device)
            shift_tensor = torch.tensor(shift_data, dtype=torch.int32, device=device)
            result = op_func(input_tensor, shift_tensor)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron(op_name)

    ###########################
    # Test inplace on device
    ###########################
    @pytest.mark.parametrize("op_func,op_name,test_data", UNARY_BITWISE_INPLACE_OPS)
    def test_unary_bitwise_inplace_runs_on_neuron(self, device, op_func, op_name, test_data):
        """Test that unary bitwise inplace operations run on Neuron"""
        with track_neuron_ops():
            input_tensor = torch.tensor(test_data, dtype=torch.int32, device=device)
            result = getattr(input_tensor, op_func)()
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron(op_name)

    @pytest.mark.parametrize("op_func,op_name,input1_data,input2_data", BINARY_BITWISE_INPLACE_OPS)
    def test_binary_bitwise_inplace_runs_on_neuron(
        self, device, op_func, op_name, input1_data, input2_data
    ):
        """Test that binary bitwise inplace operations run on Neuron"""
        with track_neuron_ops():
            input1 = torch.tensor(input1_data, dtype=torch.int32, device=device)
            input2 = torch.tensor(input2_data, dtype=torch.int32, device=device)
            result = getattr(input1, op_func)(input2)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron(op_name)

    @pytest.mark.parametrize("op_func,op_name,input_data,shift_data", SHIFT_INPLACE_OPS)
    def test_shift_ops_inplace_run_on_neuron(
        self, device, op_func, op_name, input_data, shift_data
    ):
        """Test that shift inplace operations run on Neuron"""
        with track_neuron_ops():
            input_tensor = torch.tensor(input_data, dtype=torch.int32, device=device)
            shift_tensor = torch.tensor(shift_data, dtype=torch.int32, device=device)
            result = getattr(input_tensor, op_func)(shift_tensor)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron(op_name)

    ###########################
    # Test out on device
    ###########################
    @pytest.mark.parametrize("op_func,op_name,test_data", UNARY_BITWISE_OPS)
    def test_unary_bitwise_out_runs_on_neuron(self, device, op_func, op_name, test_data):
        """Test that unary bitwise operations run on Neuron"""
        with track_neuron_ops():
            input_tensor = torch.tensor(test_data, dtype=torch.int32, device=device)
            output = torch.empty_like(input_tensor)
            result = op_func(input_tensor, out=output)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron(op_name)

    @pytest.mark.parametrize("op_func,op_name,input1_data,input2_data", BINARY_BITWISE_OPS)
    def test_binary_bitwise_out_runs_on_neuron(
        self, device, op_func, op_name, input1_data, input2_data
    ):
        """Test that binary bitwise operations run on Neuron"""
        with track_neuron_ops():
            input1 = torch.tensor(input1_data, dtype=torch.int32, device=device)
            input2 = torch.tensor(input2_data, dtype=torch.int32, device=device)
            output = torch.empty_like(input1)
            result = op_func(input1, input2, out=output)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron(op_name)

    @pytest.mark.parametrize("op_func,op_name,input_data,shift_data", SHIFT_OPS)
    def test_shift_ops_out_run_on_neuron(self, device, op_func, op_name, input_data, shift_data):
        """Test that shift operations run on Neuron"""
        with track_neuron_ops():
            input_tensor = torch.tensor(input_data, dtype=torch.int32, device=device)
            shift_tensor = torch.tensor(shift_data, dtype=torch.int32, device=device)
            output = torch.empty_like(input_tensor)
            result = op_func(input_tensor, shift_tensor, out=output)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron(op_name)

    ###########################
    # Test basic correctness
    ###########################
    @pytest.mark.parametrize("op_func,op_name,test_data", UNARY_BITWISE_OPS)
    def test_unary_bitwise_correctness(self, device, op_func, op_name, test_data):
        """Test unary bitwise operations produce correct results"""
        with track_neuron_ops():
            input_tensor = torch.tensor(test_data, dtype=torch.int32)
            input_tensor_device = input_tensor.to(device)

            result = op_func(input_tensor_device)
            expected = op_func(input_tensor)

            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron(op_name)

    @pytest.mark.parametrize("op_func,op_name,input1_data,input2_data", BINARY_BITWISE_OPS)
    def test_binary_bitwise_correctness(self, device, op_func, op_name, input1_data, input2_data):
        """Test binary bitwise operations produce correct results"""
        with track_neuron_ops():
            input1 = torch.tensor(input1_data, dtype=torch.int32)
            input2 = torch.tensor(input2_data, dtype=torch.int32)

            input1_device = input1.to(device)
            input2_device = input2.to(device)

            result = op_func(input1_device, input2_device)
            expected = op_func(input1, input2)

            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron(op_name)

    @pytest.mark.parametrize("op_func,op_name,input_data,shift_data", SHIFT_OPS)
    def test_shift_ops_correctness(self, device, op_func, op_name, input_data, shift_data):
        """Test shift operations produce correct results"""
        with track_neuron_ops():
            input_tensor = torch.tensor(input_data, dtype=torch.int32)
            shift_tensor = torch.tensor(shift_data, dtype=torch.int32)

            input_device = input_tensor.to(device)
            shift_device = shift_tensor.to(device)

            result = op_func(input_device, shift_device)
            expected = op_func(input_tensor, shift_tensor)

            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron(op_name)

    ###########################
    # Test inplace correctness
    ###########################
    @pytest.mark.parametrize("op_func,op_name,test_data", UNARY_BITWISE_INPLACE_OPS)
    def test_unary_bitwise_inplace_correctness(self, device, op_func, op_name, test_data):
        """Test unary bitwise operations produce correct results"""
        with track_neuron_ops():
            input_tensor = torch.tensor(test_data, dtype=torch.int32)
            input_tensor_device = input_tensor.to(device)

            getattr(input_tensor_device, op_func)()
            getattr(input_tensor, op_func)()

            torch.testing.assert_close(input_tensor_device.cpu(), input_tensor)
            assert_op_runs_on_neuron(op_name)

    @pytest.mark.parametrize("op_func,op_name,input1_data,input2_data", BINARY_BITWISE_INPLACE_OPS)
    def test_binary_bitwise_inplace_correctness(
        self, device, op_func, op_name, input1_data, input2_data
    ):
        """Test binary bitwise operations produce correct results"""
        with track_neuron_ops():
            input1 = torch.tensor(input1_data, dtype=torch.int32)
            input2 = torch.tensor(input2_data, dtype=torch.int32)

            input1_device = input1.to(device)
            input2_device = input2.to(device)

            getattr(input1_device, op_func)(input2_device)
            getattr(input1, op_func)(input2)

            torch.testing.assert_close(input1_device.cpu(), input1)
            assert_op_runs_on_neuron(op_name)

    @pytest.mark.parametrize("op_func,op_name,input_data,shift_data", SHIFT_INPLACE_OPS)
    def test_shift_ops_inplace_correctness(self, device, op_func, op_name, input_data, shift_data):
        """Test shift operations produce correct results"""
        with track_neuron_ops():
            input_tensor = torch.tensor(input_data, dtype=torch.int32)
            shift_tensor = torch.tensor(shift_data, dtype=torch.int32)

            input_device = input_tensor.to(device)
            shift_device = shift_tensor.to(device)

            getattr(input_device, op_func)(shift_device)
            getattr(input_tensor, op_func)(shift_tensor)

            torch.testing.assert_close(input_device.cpu(), input_tensor)
            assert_op_runs_on_neuron(op_name)

    ###########################
    # Test inplace data_ptr
    ###########################
    @pytest.mark.parametrize("op_func,op_name,test_data", UNARY_BITWISE_INPLACE_OPS)
    def test_unary_bitwise_inplace_data_ptr(self, device, op_func, op_name, test_data):
        """Test unary bitwise operations produce correct results"""
        with track_neuron_ops():
            input_tensor = torch.tensor(test_data, dtype=torch.int32)
            input_tensor_device = input_tensor.to(device)
            original_data_ptr = input_tensor_device.data_ptr()

            result = getattr(input_tensor_device, op_func)()

            assert result is input_tensor_device
            assert result.data_ptr() == original_data_ptr
            assert_op_runs_on_neuron(op_name)

    @pytest.mark.parametrize("op_func,op_name,input1_data,input2_data", BINARY_BITWISE_INPLACE_OPS)
    def test_binary_bitwise_inplace_data_ptr(
        self, device, op_func, op_name, input1_data, input2_data
    ):
        """Test binary bitwise operations produce correct results"""
        with track_neuron_ops():
            input1 = torch.tensor(input1_data, dtype=torch.int32)
            input2 = torch.tensor(input2_data, dtype=torch.int32)

            input1_device = input1.to(device)
            input2_device = input2.to(device)
            original_data_ptr = input1_device.data_ptr()

            result = getattr(input1_device, op_func)(input2_device)

            assert result is input1_device
            assert result.data_ptr() == original_data_ptr
            assert_op_runs_on_neuron(op_name)

    @pytest.mark.parametrize("op_func,op_name,input_data,shift_data", SHIFT_INPLACE_OPS)
    def test_shift_ops_inplace_data_ptr(self, device, op_func, op_name, input_data, shift_data):
        """Test shift operations produce correct results"""
        with track_neuron_ops():
            input_tensor = torch.tensor(input_data, dtype=torch.int32)
            shift_tensor = torch.tensor(shift_data, dtype=torch.int32)

            input_device = input_tensor.to(device)
            shift_device = shift_tensor.to(device)
            original_data_ptr = input_device.data_ptr()

            result = getattr(input_device, op_func)(shift_device)

            assert result is input_device
            assert result.data_ptr() == original_data_ptr
            assert_op_runs_on_neuron(op_name)

    ###########################
    # Test out correctness
    ###########################
    @pytest.mark.parametrize("op_func,op_name,test_data", UNARY_BITWISE_OPS)
    def test_unary_bitwise_out_correctness(self, device, op_func, op_name, test_data):
        """Test unary bitwise operations produce correct results"""
        with track_neuron_ops():
            input_tensor = torch.tensor(test_data, dtype=torch.int32)
            expected = torch.empty_like(input_tensor)
            input_tensor_device = input_tensor.to(device)
            output = torch.empty_like(input_tensor_device)

            op_func(input_tensor_device, out=output)
            op_func(input_tensor, out=expected)

            torch.testing.assert_close(output.cpu(), expected)
            assert_op_runs_on_neuron(op_name)

    @pytest.mark.parametrize("op_func,op_name,input1_data,input2_data", BINARY_BITWISE_OPS)
    def test_binary_bitwise_out_correctness(
        self, device, op_func, op_name, input1_data, input2_data
    ):
        """Test binary bitwise operations produce correct results"""
        with track_neuron_ops():
            input1 = torch.tensor(input1_data, dtype=torch.int32)
            input2 = torch.tensor(input2_data, dtype=torch.int32)
            expected = torch.empty_like(input1)

            input1_device = input1.to(device)
            input2_device = input2.to(device)
            output = torch.empty_like(input1_device)

            op_func(input1_device, input2_device, out=output)
            op_func(input1, input2, out=expected)

            torch.testing.assert_close(output.cpu(), expected)
            assert_op_runs_on_neuron(op_name)

    @pytest.mark.parametrize("op_func,op_name,input_data,shift_data", SHIFT_OPS)
    def test_shift_ops_out_correctness(self, device, op_func, op_name, input_data, shift_data):
        """Test shift operations produce correct results"""
        with track_neuron_ops():
            input_tensor = torch.tensor(input_data, dtype=torch.int32)
            shift_tensor = torch.tensor(shift_data, dtype=torch.int32)
            expected = torch.empty_like(input_tensor)

            input_device = input_tensor.to(device)
            shift_device = shift_tensor.to(device)
            output = torch.empty_like(input_device)

            op_func(input_device, shift_device, out=output)
            op_func(input_tensor, shift_tensor, out=expected)

            torch.testing.assert_close(output.cpu(), expected)
            assert_op_runs_on_neuron(op_name)

    ###########################
    # Test dtypes correctness
    ###########################
    @pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64])
    @pytest.mark.parametrize("op_func,op_name,test_data", UNARY_BITWISE_OPS)
    def test_unary_bitwise_dtypes(self, device, dtype, op_func, op_name, test_data):
        """Test unary bitwise operations with different integer dtypes"""
        with track_neuron_ops():
            input_tensor = torch.tensor(test_data, dtype=dtype)
            input_tensor_device = input_tensor.to(device)

            result = op_func(input_tensor_device)
            expected = op_func(input_tensor)

            assert result.dtype == dtype
            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron(op_name)

    @pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64])
    @pytest.mark.parametrize("op_func,op_name,input1_data,input2_data", BINARY_BITWISE_OPS)
    def test_binary_bitwise_dtypes(self, device, dtype, op_func, op_name, input1_data, input2_data):
        """Test binary bitwise operations with different integer dtypes"""
        with track_neuron_ops():
            input1 = torch.tensor(input1_data, dtype=dtype)
            input2 = torch.tensor(input2_data, dtype=dtype)

            input1_device = input1.to(device)
            input2_device = input2.to(device)

            result = op_func(input1_device, input2_device)
            expected = op_func(input1, input2)

            assert result.dtype == dtype
            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron(op_name)

    @pytest.mark.parametrize(
        "op_func,shift_val,input_data",
        [
            (torch.bitwise_left_shift, 2, [1, 2, 4, 8]),
            (torch.bitwise_right_shift, 2, [16, 32, 64, 127]),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64])
    def test_bitwise_shift_tensor_scalar(self, device, op_func, shift_val, input_data, dtype):
        """Test bitwise shift operations with tensor and scalar inputs"""
        with track_neuron_ops():
            input_tensor = torch.tensor(input_data, dtype=dtype)
            input_tensor_device = input_tensor.to(device)

            result = op_func(input_tensor_device, shift_val)
            expected = op_func(input_tensor, shift_val)
            assert result.dtype == expected.dtype
            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron(f"aten::{op_func.__name__}")

    @pytest.mark.parametrize(
        "op_func,shift_val,input_data",
        [
            (torch.bitwise_left_shift, 2, [1, 2, 4, 8]),
            (torch.bitwise_right_shift, 2, [16, 32, 64, 127]),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64])
    def test_bitwise_shift_tensor_scalar_tensor(
        self, device, op_func, shift_val, input_data, dtype
    ):
        """Test bitwise shift operations with tensor and scalar tensor inputs"""
        with track_neuron_ops():
            input_tensor = torch.tensor(input_data, dtype=dtype)
            input_tensor_device = input_tensor.to(device)
            shift_tensor = torch.tensor(shift_val)
            shift_tensor_device = shift_tensor.to(device)

            result = op_func(input_tensor_device, shift_tensor_device)
            expected = op_func(input_tensor, shift_tensor)
            assert result.dtype == expected.dtype
            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron(f"aten::{op_func.__name__}")

    @pytest.mark.parametrize(
        "op_func,shift_val,input_data",
        [
            (torch.bitwise_left_shift, [2], 1),
            (torch.bitwise_right_shift, [2], 16),
        ],
    )
    def test_bitwise_shift_scalar_tensor(self, device, op_func, shift_val, input_data):
        """Test bitwise shift operations with scalar and tensor inputs"""
        with track_neuron_ops():
            shift_tensor = torch.tensor(shift_val)
            shift_tensor_device = shift_tensor.to(device)

            result = op_func(input_data, shift_tensor_device)
            expected = op_func(input_data, shift_tensor)
            assert result.dtype == expected.dtype
            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron(f"aten::{op_func.__name__}")

    ###########################
    # Test dimension correctness
    ###########################
    @pytest.mark.parametrize("op_func,op_name,input1_data,input2_data", BINARY_BITWISE_OPS)
    def test_binary_bitwise_2d(self, device, op_func, op_name, input1_data, input2_data):
        """Test binary bitwise operations with 2D tensors"""
        with track_neuron_ops():
            # Convert 1D test data to 2D
            input1 = torch.tensor([input1_data[:2], input1_data[1:]], dtype=torch.int32)
            input2 = torch.tensor([input2_data[:2], input2_data[1:]], dtype=torch.int32)

            input1_device = input1.to(device)
            input2_device = input2.to(device)

            result = op_func(input1_device, input2_device)
            expected = op_func(input1, input2)
            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron(op_name)

    @pytest.mark.parametrize("op_func,op_name,test_data", UNARY_BITWISE_OPS)
    def test_unary_bitwise_2d(self, device, op_func, op_name, test_data):
        """Test unary bitwise operations with 2D tensors"""
        with track_neuron_ops():
            # Convert 1D test data to 2D
            input_tensor = torch.tensor([test_data[:2], test_data[1:]], dtype=torch.int32)
            input_tensor_device = input_tensor.to(device)

            result = op_func(input_tensor_device)
            expected = op_func(input_tensor)
            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron(op_name)

    @pytest.mark.parametrize("op_func,op_name,input1_data,input2_data", BINARY_BITWISE_OPS)
    def test_binary_bitwise_broadcasting(self, device, op_func, op_name, input1_data, input2_data):
        """Test binary bitwise operations with broadcasting"""
        with track_neuron_ops():
            input1 = torch.tensor([[input1_data[0]], [input1_data[1]]], dtype=torch.int32)
            input2 = torch.tensor([input2_data[0], input2_data[1]], dtype=torch.int32)

            input1_device = input1.to(device)
            input2_device = input2.to(device)

            result = op_func(input1_device, input2_device)
            expected = op_func(input1, input2)
            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron(op_name)

    @pytest.mark.parametrize("shape", [(10,), (5, 5), (3, 4, 5), (2, 3, 4, 5)])
    @pytest.mark.parametrize("op_func,op_name,input1_data,input2_data", BINARY_BITWISE_OPS)
    def test_binary_bitwise_nd(self, device, shape, op_func, op_name, input1_data, input2_data):
        """Test binary bitwise operations on tensors with different dimensions"""
        with track_neuron_ops():
            input1 = torch.randint(0, 16, shape, dtype=torch.int32)
            input2 = torch.randint(0, 16, shape, dtype=torch.int32)

            input1_device = input1.to(device)
            input2_device = input2.to(device)

            result = op_func(input1_device, input2_device)
            expected = op_func(input1, input2)

            assert result.device.type == "neuron"
            assert result.shape == shape
            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron(op_name)

    @pytest.mark.parametrize("shape", [(10,), (5, 5), (3, 4, 5), (2, 3, 4, 5)])
    @pytest.mark.parametrize("op_func,op_name,test_data", UNARY_BITWISE_OPS)
    def test_unary_bitwise_nd(self, device, shape, op_func, op_name, test_data):
        """Test unary bitwise operations on tensors with different dimensions"""
        with track_neuron_ops():
            input_tensor = torch.randint(0, 16, shape, dtype=torch.int32)
            input_tensor_device = input_tensor.to(device)

            result = op_func(input_tensor_device)
            expected = op_func(input_tensor)

            assert result.device.type == "neuron"
            assert result.shape == shape
            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron(op_name)

    @pytest.mark.parametrize("shape", [(10,), (5, 5), (3, 4, 5), (2, 3, 4, 5)])
    @pytest.mark.parametrize("op_func,op_name,input_data,shift_data", SHIFT_OPS)
    def test_shift_ops_nd(self, device, shape, op_func, op_name, input_data, shift_data):
        """Test shift operations on tensors with different dimensions"""
        with track_neuron_ops():
            input_tensor = torch.randint(1, 16, shape, dtype=torch.int32)
            shift_tensor = torch.randint(1, 4, shape, dtype=torch.int32)

            input_device = input_tensor.to(device)
            shift_device = shift_tensor.to(device)

            result = op_func(input_device, shift_device)
            expected = op_func(input_tensor, shift_tensor)

            assert result.device.type == "neuron"
            assert result.shape == shape
            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron(op_name)
