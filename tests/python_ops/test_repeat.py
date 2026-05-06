"""Tests for repeat operation."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    track_neuron_ops,
)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestRepeat:
    """Test suite for repeat operation."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
    def test_repeat_basic_shapes(self, device, shape):
        """Test repeat with various tensor shapes."""
        with track_neuron_ops():
            x = torch.randn(shape, device=device)
            repeats = (2,) * len(shape)
            neuron_result = x.repeat(repeats)
            cpu_result = x.cpu().repeat(repeats)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::repeat")

    @pytest.mark.parametrize("repeats", [(3, 2), (2, 3, 4)])
    def test_repeat_different_repeats(self, device, repeats):
        """Test repeat with different repeat patterns."""
        with track_neuron_ops():
            x = torch.randn(2, 3, device=device)
            neuron_result = x.repeat(repeats)
            cpu_result = x.cpu().repeat(repeats)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::repeat")

    def test_repeat_invalid_repeats(self, device):
        """Test repeat with different repeat patterns."""
        x = torch.randn(2, 3, device=device)

        with pytest.raises(
            RuntimeError,
            match="Number of dimensions of repeat dims can not be smaller than "
            "number of dimensions of tensor",
        ):
            x.repeat(2)

        # Verify same behavior on CPU
        with pytest.raises(
            RuntimeError,
            match="Number of dimensions of repeat dims can not be smaller than "
            "number of dimensions of tensor",
        ):
            x.cpu().repeat(2)

    @pytest.mark.parametrize(
        "dtype", [torch.float64, torch.float32, torch.float16, torch.int32, torch.int64]
    )
    def test_repeat_different_dtypes(self, device, dtype):
        """Test repeat with different data types."""
        with track_neuron_ops():
            x = torch.tensor([[1, 2], [3, 4]], dtype=dtype, device=device)
            neuron_result = x.repeat(2, 3)
            cpu_result = x.cpu().repeat(2, 3)

            assert neuron_result.device.type == "neuron"
            assert neuron_result.dtype == dtype
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::repeat")

    def test_repeat_single_element(self, device):
        """Test repeat with single element tensor."""
        with track_neuron_ops():
            x = torch.tensor([5.0], device=device)
            neuron_result = x.repeat(4)
            cpu_result = x.cpu().repeat(4)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::repeat")

    def test_repeat_scalar(self, device):
        """Test repeat with scalar tensor."""
        with track_neuron_ops():
            x = torch.tensor(3.14, device=device)
            neuron_result = x.repeat(2, 3)
            cpu_result = x.cpu().repeat(2, 3)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::repeat")

    def test_repeat_expand_dimensions(self, device):
        """Test repeat that expands dimensions."""
        with track_neuron_ops():
            x = torch.randn(3, device=device)
            neuron_result = x.repeat(2, 4)  # (3,) -> (2, 4, 3)
            cpu_result = x.cpu().repeat(2, 4)

            assert neuron_result.device.type == "neuron"
            assert neuron_result.shape == (2, 12)
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::repeat")

    def test_repeat_no_repeat(self, device):
        """Test repeat with all ones (no actual repetition)."""
        with track_neuron_ops():
            x = torch.randn(2, 3, device=device)
            neuron_result = x.repeat(1, 1)
            cpu_result = x.cpu().repeat(1, 1)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::repeat")
