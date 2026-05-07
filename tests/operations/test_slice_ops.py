import numpy as np
import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


class TestOpsOnSlice:
    """Test operations performed on slice of tensor."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.device = torch.device("neuron", 0)
        self.cpu_device = torch.device("cpu")

    def test_slice_range(self):
        """Test slice [n:m] operations."""
        # Create test tensor
        cpu_tensor = torch.arange(10, dtype=torch.float32)
        neuron_tensor = cpu_tensor.to(self.device)

        # Test slice [2:7]
        cpu_result = cpu_tensor[2:7]
        neuron_result = neuron_tensor[2:7]

        # Verify results
        assert torch.equal(cpu_result, neuron_result.to("cpu"))

    @pytest.mark.parametrize(
        "op_name,scalar_value", [("ne", float("inf")), ("lt", 0), ("le", 0), ("gt", 0), ("ge", 0)]
    )
    def test_index_value_comparison(self, op_name, scalar_value):
        """Test get correct value by index for comparision."""
        # Create test tensor
        cpu_tensor = torch.tensor([float("inf"), float("-inf"), float("nan")])
        neuron_tensor = cpu_tensor.to(self.device)

        # Perform operation
        op_fn = getattr(torch, op_name)
        cpu_result = op_fn(cpu_tensor[1], scalar_value)
        neuron_result = op_fn(neuron_tensor[1], scalar_value)

        # Verify results
        assert torch.equal(cpu_result.to("cpu"), neuron_result.to("cpu"))

    @pytest.mark.parametrize(
        "op_name,scalar_value",
        [
            ("ne", [float("inf"), float("-inf")]),
            ("lt", [0, 0]),
            ("le", [0, 0]),
            ("gt", [0, 0]),
            ("ge", [0, 0]),
        ],
    )
    def test_slice_value_comparison(self, op_name, scalar_value):
        """Test get correct slice for comparision."""
        # Create test tensor
        cpu_tensor = torch.tensor([float("inf"), float("-inf"), float("inf")])
        neuron_tensor = cpu_tensor.to(self.device)

        # Perform operation
        op_fn = getattr(torch, op_name)
        cpu_result = op_fn(cpu_tensor[1:3], torch.tensor(scalar_value))
        neuron_result = op_fn(neuron_tensor[1:3], torch.tensor(scalar_value, device=self.device))

        # Verify results
        assert torch.equal(cpu_result.to("cpu"), neuron_result.to("cpu"))
