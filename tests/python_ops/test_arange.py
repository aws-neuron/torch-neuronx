import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)


class TestArange:
    def test_arange_basic(self):
        """Test arange basic"""
        device = "neuron"

        with track_neuron_ops():
            input = torch.arange(0, 5, 1, dtype=torch.float32, device=device)
            input_cpu = torch.arange(0, 5, 1, dtype=torch.float32)
            assert input.device.type == device
            assert input.device.type != input_cpu.device.type
            assert_op_runs_on_neuron("aten::arange.start_out")

    def test_arange_start_only(self):
        """Test arange with only start parameter"""
        device = "neuron"
        with track_neuron_ops():
            input = torch.arange(5, device=device)

            input_cpu = torch.arange(5)
            assert input.device.type == device
            assert input_cpu.device.type == "cpu"

            torch.testing.assert_close(input.cpu(), input_cpu)
            assert_op_runs_on_neuron("aten::arange")

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float64, torch.float16, torch.int32, torch.int64]
    )
    def test_arange_different_dtypes(self, dtype):
        """Test arange with different dtypes"""
        device = "neuron"
        with track_neuron_ops():
            input = torch.arange(0, 5, 1, dtype=dtype, device=device)
            input_cpu = torch.arange(0, 5, 1, dtype=dtype)
            assert input.dtype == dtype
            assert input.device.type == device
            assert input_cpu.device.type == "cpu"
            torch.testing.assert_close(input.cpu(), input_cpu)
            assert_op_runs_on_neuron("aten::arange.start_out")

    def test_arange_negative_step(self):
        """Test arange with negative step"""
        device = "neuron"
        with track_neuron_ops():
            input = torch.arange(5, -1, -1, dtype=torch.int32, device=device)
            input_cpu = torch.arange(5, -1, -1, dtype=torch.int32)

            assert input_cpu.device.type == "cpu"
            assert input.device.type == device
            torch.testing.assert_close(input.cpu(), input_cpu)
            assert_op_runs_on_neuron("aten::arange.start_out")

    def test_arange_float_steps(self):
        """Test arange with float steps"""
        device = "neuron"
        with track_neuron_ops():
            input = torch.arange(0, 2.5, 0.5, dtype=torch.float32, device=device)
            input_cpu = torch.arange(0, 2.5, 0.5, dtype=torch.float32)

            assert input.device.type == device
            torch.testing.assert_close(input.cpu(), input_cpu)
            assert_op_runs_on_neuron("aten::arange.start_out")

    def test_arange_requires_grad(self):
        """Test arange with requires_grad"""
        device = "neuron"
        with track_neuron_ops():
            input = torch.arange(0, 5, 1, dtype=torch.float32, requires_grad=True, device=device)
            assert input.requires_grad
            assert input.device.type == device
            assert_op_runs_on_neuron("aten::arange.start_out")

    def test_arange_equal_bounds(self):
        """Test arange with equal start and stop values"""
        device = "neuron"
        with track_neuron_ops():
            input = torch.arange(start=1, end=1, step=1, dtype=torch.int32, device=device)
            input_cpu = torch.arange(1, 1, 1, dtype=torch.int32)

            assert input.device.type == device
            assert input_cpu.device.type == "cpu"
            assert input.numel() == 0
            torch.testing.assert_close(input.cpu(), input_cpu)
            assert_op_runs_on_neuron("aten::arange")

    @assert_raises(RuntimeError, match="upper bound and lower bound inconsistent with step sign")
    def test_arange_invalid_bounds(self):
        """Test arange with invalid bounds (start > end)"""
        device = "neuron"
        torch.arange(1, 0, 1, dtype=torch.int32, device=device)

    @assert_raises(RuntimeError, match="step must be nonzero")
    def test_arange_invalid_stepsize(self):
        """Test arange with stepsize 0"""
        device = "neuron"
        torch.arange(0, 5, 0, dtype=torch.int32, device=device)

    def test_arange_explicit_layout(self):
        """Test arange with explicitly specified layouts"""
        device = "neuron"
        with track_neuron_ops():
            # Test with explicit strided layout
            input_strided = torch.arange(5, device=device, layout=torch.strided)
            assert input_strided.layout == torch.strided

            # Test default matches strided
            input_default = torch.arange(5, device=device)
            assert input_default.layout == torch.strided

            # Verify both produce same values
            torch.testing.assert_close(input_strided.cpu(), input_default.cpu())

            assert_op_runs_on_neuron("aten::arange")
