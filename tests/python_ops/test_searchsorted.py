import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestSearchsortedEager:
    """Eager mode tests for searchsorted on Neuron device."""

    def test_tensor_basic(self):
        """Test basic searchsorted with left (default) insertion."""
        sorted_seq = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0], device="neuron")
        values = torch.tensor([2.0, 4.0, 6.0], device="neuron")

        with track_neuron_ops():
            result = torch.searchsorted(sorted_seq, values)
            assert_op_runs_on_neuron("aten::searchsorted")

        expected = torch.searchsorted(sorted_seq.cpu(), values.cpu())
        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize("right", [False, True])
    def test_tensor_right(self, right):
        """Test searchsorted with left/right insertion on duplicates."""
        sorted_seq = torch.tensor([1.0, 2.0, 2.0, 3.0, 4.0], device="neuron")
        values = torch.tensor([2.0], device="neuron")

        result = torch.searchsorted(sorted_seq, values, right=right)
        expected = torch.searchsorted(sorted_seq.cpu(), values.cpu(), right=right)
        torch.testing.assert_close(result.cpu(), expected)

    def test_tensor_side_param(self):
        """Test that side='right' behaves like right=True."""
        sorted_seq = torch.tensor([1.0, 2.0, 2.0, 3.0], device="neuron")
        values = torch.tensor([2.0], device="neuron")

        result = torch.searchsorted(sorted_seq, values, side="right")
        expected = torch.searchsorted(sorted_seq.cpu(), values.cpu(), right=True)
        torch.testing.assert_close(result.cpu(), expected)

    def test_tensor_2d(self):
        """Test searchsorted with batched 2D inputs."""
        sorted_seq = torch.tensor([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], device="neuron")
        values = torch.tensor([[2.0, 4.0], [3.0, 5.0]], device="neuron")

        result = torch.searchsorted(sorted_seq, values)
        expected = torch.searchsorted(sorted_seq.cpu(), values.cpu())
        torch.testing.assert_close(result.cpu(), expected)

    def test_tensor_out_int32(self):
        """Test out_int32 flag returns int32 dtype."""
        sorted_seq = torch.tensor([1.0, 3.0, 5.0], device="neuron")
        values = torch.tensor([2.0], device="neuron")

        result = torch.searchsorted(sorted_seq, values, out_int32=True)
        assert result.dtype == torch.int32

    def test_tensor_sorter(self):
        """Test searchsorted with a sorter index."""
        unsorted_seq = torch.tensor([3.0, 1.0, 4.0, 2.0], device="neuron")
        sorter = torch.argsort(unsorted_seq.cpu()).to("neuron")
        values = torch.tensor([2.5], device="neuron")

        result = torch.searchsorted(unsorted_seq, values, sorter=sorter)
        expected = torch.searchsorted(unsorted_seq.cpu(), values.cpu(), sorter=sorter.cpu())
        torch.testing.assert_close(result.cpu(), expected)

    def test_scalar(self):
        """Test scalar variant."""
        sorted_seq = torch.tensor([1.0, 3.0, 5.0, 7.0], device="neuron")

        with track_neuron_ops():
            result = torch.searchsorted(sorted_seq, 4.0)
            assert_op_runs_on_neuron("aten::searchsorted")

        expected = torch.searchsorted(sorted_seq.cpu(), 4.0)
        torch.testing.assert_close(result.cpu(), expected)

    def test_scalar_right(self):
        """Test scalar variant with right insertion."""
        sorted_seq = torch.tensor([1.0, 2.0, 2.0, 3.0], device="neuron")

        result = torch.searchsorted(sorted_seq, 2.0, right=True)
        expected = torch.searchsorted(sorted_seq.cpu(), 2.0, right=True)
        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_tensor_dtypes(self, dtype):
        """Test searchsorted with various dtypes."""
        torch.manual_seed(42)
        sorted_seq = torch.sort(torch.randn(20, dtype=dtype))[0].to("neuron")
        values = torch.randn(5, dtype=dtype).to("neuron")

        result = torch.searchsorted(sorted_seq, values)
        expected = torch.searchsorted(sorted_seq.cpu(), values.cpu())
        torch.testing.assert_close(result.cpu(), expected)
