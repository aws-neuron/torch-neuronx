import os

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_did_not_run_on_neuron,
    assert_op_runs_on_neuron,
    track_neuron_ops,
)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestToCopy:
    """Test suite for to_copy operation."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.mark.parametrize("torch_dtype", [torch.float32, torch.bfloat16, torch.float8_e5m2])
    def test_nonzero_static_basic_shapes(self, device, torch_dtype):
        """Test nonzero_static with various tensor shapes."""
        with track_neuron_ops():
            # TESTING CPU -> NEURON
            x = torch.randn(128, dtype=torch.float32).to(torch_dtype)
            neuron_result = x.to(device=device)
            assert neuron_result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::_to_copy")

            # TESTING NEURON -> NEURON
            x = torch.randn(128, dtype=torch.float32, device=device)
            neuron_result = x.to(torch_dtype)
            assert neuron_result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::_to_copy")
            assert_op_runs_on_neuron("aten::copy_")
