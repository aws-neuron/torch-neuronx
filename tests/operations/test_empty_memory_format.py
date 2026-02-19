"""Test torch.empty with memory format support on Neuron."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestEmptyMemoryFormat:
    """Test empty operation with different memory formats."""

    def test_empty_channels_last(self):
        """Test empty with ChannelsLast memory format (4D NHWC)."""
        with track_neuron_ops():
            x = torch.empty(2, 3, 4, 5, device="neuron", memory_format=torch.channels_last)
            assert x.shape == (2, 3, 4, 5)
            assert x.is_contiguous(memory_format=torch.channels_last)
            n, c, h, w = x.shape
            assert x.stride() == (h * w * c, 1, w * c, c)
            assert_op_runs_on_neuron("aten::empty")

    def test_empty_channels_last_3d(self):
        """Test empty with ChannelsLast3d memory format (5D NDHWC)."""
        with track_neuron_ops():
            x = torch.empty(2, 3, 4, 5, 6, device="neuron", memory_format=torch.channels_last_3d)
            assert x.shape == (2, 3, 4, 5, 6)
            assert x.is_contiguous(memory_format=torch.channels_last_3d)
            n, c, d, h, w = x.shape
            assert x.stride() == (d * h * w * c, 1, h * w * c, w * c, c)
            assert_op_runs_on_neuron("aten::empty")

    def test_empty_like_channels_last(self):
        """Test empty_like with explicit channels_last memory format."""
        with track_neuron_ops():
            x = torch.randn(2, 3, 4, 5, device="neuron")
            y = torch.empty_like(x, memory_format=torch.channels_last)
            assert y.shape == x.shape
            assert y.is_contiguous(memory_format=torch.channels_last)

    def test_empty_like_channels_last_3d(self):
        """Test empty_like with explicit channels_last_3d memory format."""
        with track_neuron_ops():
            x = torch.randn(2, 3, 4, 5, 6, device="neuron")
            y = torch.empty_like(x, memory_format=torch.channels_last_3d)
            assert y.shape == x.shape
            assert y.is_contiguous(memory_format=torch.channels_last_3d)
