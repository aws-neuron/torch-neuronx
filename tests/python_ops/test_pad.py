import os

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestPad:
    """Test cases for pad operation"""

    @pytest.mark.parametrize(
        "mode,expected_op",
        [
            ("constant", "aten::constant_pad_nd"),
            ("reflect", "aten::reflection_pad2d"),  # For 4D tensors
            ("replicate", "aten::replication_pad2d"),  # For 4D tensors
            ("circular", "aten::copy_"),  # dispatched to new_empty and copy_
        ],
    )
    def test_pad_decomposition_to_correct_ops(self, mode, expected_op):
        """Test that pad decomposes to the correct specific operations"""
        with track_neuron_ops():
            x_cpu = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
            x_neuron = x_cpu.to("neuron")
            x_cpu_4d = x_cpu.unsqueeze(0).unsqueeze(1)
            x_neuron_4d = x_neuron.unsqueeze(0).unsqueeze(1)

            result_cpu = torch.nn.functional.pad(x_cpu_4d, (1, 1, 1, 1), mode=mode)
            result_neuron = torch.nn.functional.pad(x_neuron_4d, (1, 1, 1, 1), mode=mode)

            assert result_neuron.device.type == "neuron"
            torch.testing.assert_close(result_neuron.cpu(), result_cpu)

            # Check that both have grad_fn (autograd chain preserved)
            assert result_cpu.grad_fn is not None, f"CPU {mode} pad should have grad_fn"
            assert result_neuron.grad_fn is not None, f"Neuron {mode} pad should have grad_fn"

            # Check that grad_fn names match (same autograd operation)
            assert (
                type(result_cpu.grad_fn).__name__ == type(result_neuron.grad_fn).__name__
            ), f"grad_fn mismatch: CPU={type(result_cpu.grad_fn).__name__}, Neuron={type(result_neuron.grad_fn).__name__}"  # noqa E501

            # Check that the specific decomposed operation runs, not aten::pad
            assert_op_runs_on_neuron(expected_op)

    @pytest.mark.parametrize(
        "mode,tensor_shape,padding,expected_op",
        [
            # Constant mode
            ("constant", (2, 3), (1, 1, 1, 1), "aten::constant_pad_nd"),
            ("constant", (3,), (2, 3), "aten::constant_pad_nd"),
            ("constant", (2, 3, 4, 5), (1, 2, 1, 2), "aten::constant_pad_nd"),
            # 1D padding (3D tensors)
            ("reflect", (1, 1, 4), (2, 2), "aten::reflection_pad1d"),
            ("replicate", (1, 1, 4), (2, 2), "aten::replication_pad1d"),
            # 2D padding (4D tensors)
            ("reflect", (1, 1, 3, 3), (1, 1, 1, 1), "aten::reflection_pad2d"),
            ("replicate", (1, 1, 3, 3), (1, 1, 1, 1), "aten::replication_pad2d"),
            # 3D padding (5D tensors)
            ("reflect", (1, 1, 2, 3, 4), (1, 1, 1, 1, 1, 1), "aten::reflection_pad3d"),
            ("replicate", (1, 1, 2, 3, 4), (1, 1, 1, 1, 1, 1), "aten::replication_pad3d"),
        ],
    )
    def test_pad_all_modes_and_dimensions(self, mode, tensor_shape, padding, expected_op):
        """Test all padding modes with different tensor dimensions"""
        with track_neuron_ops():
            x_cpu = torch.randn(*tensor_shape, requires_grad=True)
            x_neuron = x_cpu.to("neuron")

            value = 5.0 if mode == "constant" else None
            result_cpu = torch.nn.functional.pad(x_cpu, padding, mode=mode, value=value)
            result_neuron = torch.nn.functional.pad(x_neuron, padding, mode=mode, value=value)

            assert result_neuron.device.type == "neuron"
            torch.testing.assert_close(result_neuron.cpu(), result_cpu)

            # Check that both have grad_fn (autograd chain preserved)
            assert result_cpu.grad_fn is not None, f"CPU {mode} pad should have grad_fn"
            assert result_neuron.grad_fn is not None, f"Neuron {mode} pad should have grad_fn"

            # Check that grad_fn names match (same autograd operation)
            assert (
                type(result_cpu.grad_fn).__name__ == type(result_neuron.grad_fn).__name__
            ), f"grad_fn mismatch: CPU={type(result_cpu.grad_fn).__name__}, Neuron={type(result_neuron.grad_fn).__name__}"  # noqa E501

            assert_op_runs_on_neuron(expected_op)

    @pytest.mark.parametrize(
        "padding,value",
        [
            ((1, 1, 1, 1), 0),  # Basic symmetric padding with zero fill
            ((1, 2, 1, 2), 5.0),  # Asymmetric padding with custom fill value
        ],
    )
    def test_pad_constant_variations(self, padding, value):
        """Test constant padding with different patterns and values"""
        with track_neuron_ops():
            x_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
            x_neuron = x_cpu.to("neuron")

            result_cpu = torch.nn.functional.pad(x_cpu, padding, mode="constant", value=value)
            result_neuron = torch.nn.functional.pad(x_neuron, padding, mode="constant", value=value)

            assert result_neuron.device.type == "neuron"
            torch.testing.assert_close(result_neuron.cpu(), result_cpu)

            # Check that both have grad_fn (autograd chain preserved)
            assert result_cpu.grad_fn is not None, "CPU constant pad should have grad_fn"
            assert result_neuron.grad_fn is not None, "Neuron constant pad should have grad_fn"

            # Check that grad_fn names match (same autograd operation)
            assert (
                type(result_cpu.grad_fn).__name__ == type(result_neuron.grad_fn).__name__
            ), f"grad_fn mismatch: CPU={type(result_cpu.grad_fn).__name__}, Neuron={type(result_neuron.grad_fn).__name__}"  # noqa E501

            assert_op_runs_on_neuron("aten::constant_pad_nd")

    def test_pad_asymmetric(self):
        """Test asymmetric padding"""
        with track_neuron_ops():
            x_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            x_neuron = x_cpu.to("neuron")

            result_neuron = torch.nn.functional.pad(
                x_neuron, (1, 3, 2, 1), mode="constant", value=0
            )
            expected = torch.nn.functional.pad(x_cpu, (1, 3, 2, 1), mode="constant", value=0)

            assert result_neuron.device.type == "neuron"
            torch.testing.assert_close(result_neuron.cpu(), expected)
            assert_op_runs_on_neuron("aten::pad")

    def test_pad_zero_padding(self):
        """Test zero padding (no-op)"""
        with track_neuron_ops():
            x_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            x_neuron = x_cpu.to("neuron")

            result_neuron = torch.nn.functional.pad(
                x_neuron, (0, 0, 0, 0), mode="constant", value=0
            )
            expected = torch.nn.functional.pad(x_cpu, (0, 0, 0, 0), mode="constant", value=0)

            assert result_neuron.device.type == "neuron"
            torch.testing.assert_close(result_neuron.cpu(), expected)
            assert_op_runs_on_neuron("aten::pad")

    def test_pad_partial_dimensions(self):
        """Test padding only some dimensions"""
        with track_neuron_ops():
            x_cpu = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
            x_neuron = x_cpu.to("neuron")

            # Only pad last dimension
            result_neuron = torch.nn.functional.pad(x_neuron, (1, 2), mode="constant", value=0)
            expected = torch.nn.functional.pad(x_cpu, (1, 2), mode="constant", value=0)

            assert result_neuron.device.type == "neuron"
            torch.testing.assert_close(result_neuron.cpu(), expected)
            assert_op_runs_on_neuron("aten::pad")

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ],
    )
    def test_pad_different_dtypes(self, dtype):
        """Test padding with different data types"""
        with track_neuron_ops():
            x_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
            x_neuron = x_cpu.to("neuron")

            result_neuron = torch.nn.functional.pad(
                x_neuron, (1, 1, 1, 1), mode="constant", value=0
            )
            expected = torch.nn.functional.pad(x_cpu, (1, 1, 1, 1), mode="constant", value=0)

            assert result_neuron.device.type == "neuron"
            torch.testing.assert_close(result_neuron.cpu(), expected)
            assert_op_runs_on_neuron("aten::pad")

    def test_pad_empty_tensor(self):
        """Test padding empty tensor"""
        with track_neuron_ops():
            x_cpu = torch.empty(0, 2)
            x_neuron = x_cpu.to("neuron")

            result_neuron = torch.nn.functional.pad(x_neuron, (1, 1), mode="constant", value=0)
            expected = torch.nn.functional.pad(x_cpu, (1, 1), mode="constant", value=0)

            assert result_neuron.device.type == "neuron"
            torch.testing.assert_close(result_neuron.cpu(), expected)
            assert_op_runs_on_neuron("aten::pad")

    @pytest.mark.parametrize(
        "tensor_data, padding, value",
        [
            (torch.zeros(2, 3), (1, 1), 2),  # Zero-filled tensor
            pytest.param(
                torch.zeros(2, 3),
                (1, -1),
                2,  # Negative padding dimension
                marks=pytest.mark.xfail(
                    os.environ.get("TORCH_NEURONX_MLIR_ATEN_OPS", "0") == "0",
                    reason="JAX does not allow negative padding dimensions",
                ),
            ),
            (
                torch.tensor([[-1.0, -2.0], [-3.0, -4.0]]),
                (1, 1, 1, 1),
                -5.0,
            ),  # Tensor with negative values
            (torch.tensor([[5.0]]), (2, 2, 2, 2), 1.0),  # Single element tensor
            (
                torch.tensor([[1.0, float("inf")], [float("-inf"), 4.0]]),
                (1, 1, 1, 1),
                0,
            ),  # Tensor with infinity values
        ],
    )
    def test_pad_special_cases(self, tensor_data, padding, value):
        """Test padding with special tensor cases"""
        with track_neuron_ops():
            x_cpu = tensor_data
            x_neuron = x_cpu.to("neuron")

            result_neuron = torch.nn.functional.pad(x_neuron, padding, mode="constant", value=value)
            expected = torch.nn.functional.pad(x_cpu, padding, mode="constant", value=value)

            assert result_neuron.device.type == "neuron"
            torch.testing.assert_close(result_neuron.cpu(), expected)
            assert_op_runs_on_neuron("aten::pad")

    @assert_raises(RuntimeError, match="Padding length must be divisible by 2")
    def test_pad_error_invalid_padding_length(self):
        """Test error for invalid padding length"""
        x_neuron = torch.tensor([[1.0, 2.0]], device="neuron")

        torch.nn.functional.pad(x_neuron, (1, 2, 3), mode="constant")

    @assert_raises(
        RuntimeError,
        match="Padding length should be less than or equal to two times the input dimension .*",
    )
    def test_pad_error_padding_too_long(self):
        """Test error for padding sequence too long"""
        x_neuron = torch.tensor([[1.0, 2.0]], device="neuron")

        torch.nn.functional.pad(x_neuron, (1, 1, 1, 1, 1, 1), mode="constant")

    @assert_raises((NotImplementedError, RuntimeError), match="Unrecognised padding mode .*")
    def test_pad_error_unsupported_mode(self):
        """Test error for unsupported padding mode"""
        x_neuron = torch.tensor([[1.0, 2.0]], device="neuron")
        torch.nn.functional.pad(x_neuron, (1, 1), mode="wrap")
