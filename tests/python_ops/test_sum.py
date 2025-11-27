"""Test that sum operation is properly registered with PyTorch dispatcher."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestSumRegistration:
    """Test sum operation registration and functionality."""

    def test_sum_function_no_dim(self):
        """Test that torch.sum works with neuron tensors (no dimension specified)."""
        # Create neuron tensor
        tensor = torch.ones(16, 16, dtype=torch.float32).to("neuron")

        # Use torch.sum function
        result = torch.sum(tensor)

        # Verify result
        assert result.device.type == "neuron"
        assert result.shape == ()  # Scalar tensor
        expected = torch.tensor(16 * 16, dtype=torch.float32)
        torch.testing.assert_close(result.cpu(), expected)

    def test_sum_method_no_dim(self):
        """Test that tensor.sum() method works with neuron tensors (no dimension specified)."""
        # Create neuron tensor
        tensor = torch.ones(16, 16, dtype=torch.float32).to("neuron") * 2

        # Use tensor.sum() method
        result = tensor.sum()

        # Verify result
        assert result.device.type == "neuron"
        assert result.shape == ()  # Scalar tensor
        expected = torch.tensor(16 * 16 * 2, dtype=torch.float32)
        torch.testing.assert_close(result.cpu(), expected)

    def test_sum_function_with_dim(self):
        """Test that torch.sum works with neuron tensors (with dimension specified)."""
        # Create neuron tensor
        tensor = torch.ones(4, 5, 6, dtype=torch.float32).to("neuron") * 2

        # Sum along dimension 1
        result = torch.sum(tensor, dim=1)

        # Verify result
        assert result.device.type == "neuron"
        assert result.shape == (4, 6)  # Dimension 1 is reduced
        expected = torch.ones(4, 6, dtype=torch.float32) * 10  # 5 * 2
        torch.testing.assert_close(result.cpu(), expected)

    def test_sum_method_with_dim(self):
        """Test that tensor.sum() method works with neuron tensors (with dimension specified)."""
        # Create neuron tensor
        tensor = torch.ones(4, 5, 6, dtype=torch.float32).to("neuron") * 2

        # Sum along dimension 1
        result = tensor.sum(dim=1)

        # Verify result
        assert result.device.type == "neuron"
        assert result.shape == (4, 6)  # Dimension 1 is reduced
        expected = torch.ones(4, 6, dtype=torch.float32) * 10  # 5 * 2
        torch.testing.assert_close(result.cpu(), expected)

    def test_sum_with_keepdim(self):
        """Test sum with keepdim parameter."""
        # Create neuron tensor
        tensor = torch.ones(4, 5, 6, dtype=torch.float32).to("neuron") * 2

        # Sum along dimension 1 with keepdim=True
        result = torch.sum(tensor, dim=1, keepdim=True)

        # Verify result
        assert result.device.type == "neuron"
        assert result.shape == (4, 1, 6)  # Dimension 1 is kept but has size 1
        expected = torch.ones(4, 1, 6, dtype=torch.float32) * 10  # 5 * 2
        torch.testing.assert_close(result.cpu(), expected)

    def test_sum_with_dtype(self):
        """Test sum with dtype parameter."""
        # Create neuron tensor with float16
        tensor = torch.ones(16, 16, dtype=torch.float16).to("neuron") * 2

        # Sum with dtype=torch.float32
        result = torch.sum(tensor, dtype=torch.float32)

        # Verify result
        assert result.device.type == "neuron"
        assert result.dtype == torch.float32
        expected = torch.tensor(16 * 16 * 2, dtype=torch.float32)
        torch.testing.assert_close(result.cpu(), expected)

    def test_sum_dim_out(self):
        """Test sum.dim_out operation."""
        # Create neuron tensors
        tensor = torch.ones(4, 5, 6, dtype=torch.float32).to("neuron") * 2
        out = torch.empty(4, 6, dtype=torch.float32).to("neuron")

        # Use torch.sum with dim and out parameters
        result = torch.sum(tensor, dim=1, out=out)

        # Verify result and that out was modified in-place
        assert result is out
        assert out.device.type == "neuron"
        expected = torch.ones(4, 6, dtype=torch.float32) * 10  # 5 * 2
        torch.testing.assert_close(out.cpu(), expected)

    @pytest.mark.xfail(
        reason="Bug with memory management, NRT errors out "
        "expecting a buffer size of test run before i.e. "
        "test_sum_dim_out PS: occurs in serial run"
    )
    def test_sum_multiple_dims(self):
        """Test sum with multiple dimensions."""
        # Create neuron tensor
        tensor = torch.ones(4, 5, 6, dtype=torch.float32).to("neuron") * 2

        # Sum along dimensions 1 and 2
        result = torch.sum(tensor, dim=(1, 2))

        # Verify result
        assert result.device.type == "neuron"
        assert result.shape == (4,)  # Dimensions 1 and 2 are reduced
        expected = torch.ones(4, dtype=torch.float32) * (5 * 6 * 2)  # 5 * 6 * 2
        torch.testing.assert_close(result.cpu(), expected)

    def test_sum_boolean_input(self):
        """Test sum with boolean input tensor."""
        # Create boolean neuron tensor
        tensor = torch.tensor([[True, False, True], [False, True, True]], dtype=torch.bool).to(
            "neuron"
        )

        # Sum boolean tensor
        result = torch.sum(tensor)

        # Verify result
        assert result.device.type == "neuron"
        expected = torch.tensor(4, dtype=torch.int64)  # 4 True values
        torch.testing.assert_close(result.cpu(), expected)

    def test_sum_runs_on_neuron(self):
        """Test that sum operations run on Neuron."""

        with track_neuron_ops():
            tensor = torch.ones(16, 16, dtype=torch.float32, device="neuron")
            result = torch.sum(tensor)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::sum")

    def test_sum_dim_runs_on_neuron(self):
        """Test that sum with dimension runs on Neuron."""

        with track_neuron_ops():
            tensor = torch.ones(4, 5, 6, dtype=torch.float32, device="neuron")
            result = torch.sum(tensor, dim=1)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::sum.dim_IntList")

    def test_sum_out_runs_on_neuron(self):
        """Test that sum with out parameter runs on Neuron."""

        with track_neuron_ops():
            tensor = torch.ones(4, 5, dtype=torch.float32, device="neuron")
            out = torch.empty(4, dtype=torch.float32, device="neuron")
            result = torch.sum(tensor, dim=1, out=out)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::sum.IntList_out")

    def test_sum_method_runs_on_neuron(self):
        """Test that tensor.sum() method runs on Neuron."""

        with track_neuron_ops():
            tensor = torch.ones(16, 16, dtype=torch.float32, device="neuron")
            result = tensor.sum()
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::sum")

    def test_sum_method_dim_runs_on_neuron(self):
        """Test that tensor.sum(dim) method runs on Neuron."""

        with track_neuron_ops():
            tensor = torch.ones(4, 5, 6, dtype=torch.float32, device="neuron")
            result = tensor.sum(dim=1)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::sum.dim_IntList")
