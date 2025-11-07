import os

import pytest
import torch

import torch_neuron
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuron.device_count() == 0, reason="No Neuron devices available")
class TestPad:
    """Test cases for pad operation"""

    @pytest.mark.parametrize(
        "mode",
        [
            "constant",  # Fill with constant value
            "reflect",  # Mirror values at boundaries
            "replicate",  # Repeat edge values
            "circular",  # Wrap around values
        ],
    )
    def test_pad_all_modes_run_on_neuron(self, mode):
        """Test that all pad modes run on Neuron without CPU fallback"""
        with track_neuron_ops():
            x_cpu = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            x_neuron = x_cpu.to("neuron")
            x_cpu_4d = x_cpu.unsqueeze(0).unsqueeze(1)
            x_neuron_4d = x_neuron.unsqueeze(0).unsqueeze(1)
            result_neuron = torch.nn.functional.pad(x_neuron_4d, (1, 1, 1, 1), mode=mode)
            expected = torch.nn.functional.pad(x_cpu_4d, (1, 1, 1, 1), mode=mode)
            assert result_neuron.device.type == "neuron"
            torch.testing.assert_close(result_neuron.cpu(), expected)
            assert_op_runs_on_neuron("aten::pad")

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
            x_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            x_neuron = x_cpu.to("neuron")

            result_neuron = torch.nn.functional.pad(x_neuron, padding, mode="constant", value=value)
            expected = torch.nn.functional.pad(x_cpu, padding, mode="constant", value=value)

            assert result_neuron.device.type == "neuron"
            torch.testing.assert_close(result_neuron.cpu(), expected)
            assert_op_runs_on_neuron("aten::pad")

    @pytest.mark.parametrize(
        "tensor_data, padding",
        [
            (torch.tensor([1.0, 2.0, 3.0]), (2, 3)),  # 1D tensor
            (
                torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
                (1, 1, 1, 1, 1, 1),
            ),  # 3D tensor
            (torch.randn(2, 3, 4, 5), (1, 2, 1, 2)),  # 4D tensor
        ],
    )
    def test_pad_tensor_dimensions(self, tensor_data, padding):
        """Test padding tensors of different dimensions"""
        with track_neuron_ops():
            x_cpu = tensor_data
            x_neuron = x_cpu.to("neuron")

            result_neuron = torch.nn.functional.pad(x_neuron, padding, mode="constant", value=0)
            expected = torch.nn.functional.pad(x_cpu, padding, mode="constant", value=0)

            assert result_neuron.device.type == "neuron"
            torch.testing.assert_close(result_neuron.cpu(), expected)
            assert_op_runs_on_neuron("aten::pad")

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

    def test_pad_error_negative_padding_dimensions(self):
        """Test padding only some dimensions"""
        x_neuron = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], device="neuron")

        with pytest.raises(ValueError, match="index can't contain negative values"):
            torch.nn.functional.pad(x_neuron, (1, -1), mode="constant", value=0)

    def test_pad_error_invalid_padding_length(self):
        """Test error for invalid padding length"""
        x_neuron = torch.tensor([[1.0, 2.0]], device="neuron")

        with pytest.raises(RuntimeError, match="Padding length must be divisible by 2"):
            torch.nn.functional.pad(x_neuron, (1, 2, 3), mode="constant")

    def test_pad_error_padding_too_long(self):
        """Test error for padding sequence too long"""
        x_neuron = torch.tensor([[1.0, 2.0]], device="neuron")

        with pytest.raises(
            RuntimeError,
            match="Padding length should be less than or equal to two times the input dimension .*",
        ):
            torch.nn.functional.pad(x_neuron, (1, 1, 1, 1, 1, 1), mode="constant")

    def test_pad_error_unsupported_mode(self):
        """Test error for unsupported padding mode"""
        x_neuron = torch.tensor([[1.0, 2.0]], device="neuron")
        error_raised = (
            NotImplementedError
            if os.environ.get("TORCH_NEURON_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS") == "0"
            else RuntimeError
        )
        with pytest.raises(error_raised, match="Unrecognised padding mode .*"):
            torch.nn.functional.pad(x_neuron, (1, 1), mode="wrap")
