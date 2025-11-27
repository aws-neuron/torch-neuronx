import pytest
import torch

import torch_neuronx


class TestNeuronOpsOnView:
    """Test neuron operations performed on view of tensor."""

    @pytest.mark.parametrize(
        "op_name,op_func",
        [
            ("unfold", lambda x: x.unfold(0, 2, 1)),
            ("view", lambda x: x.view(-1)),
            ("permute", lambda x: x.permute(1, 0)),
        ],
    )
    def test_storage_sharing(self, op_name, op_func):
        """Test storage sharing"""
        original_tensor = torch.randn(4, 3, device="neuron")
        view_tensor = op_func(original_tensor)

        # Check that they share the same storage
        assert (
            original_tensor.data_ptr() == view_tensor.data_ptr()
        ), f"{op_name} operation should share storage with original tensor"
