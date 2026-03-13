"""Tests for repeat_interleave operation.

Note: repeat_interleave.self_int and repeat_interleave.self_Tensor are NOT registered
because they are CompositeImplicitAutograd ops that decompose naturally:
- self_int decomposes to: unsqueeze -> expand -> clone -> flatten
- self_Tensor decomposes to: index_select
This preserves the autograd chain.
"""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_does_not_run,
    assert_op_runs_on_neuron,
    track_neuron_ops,
)
from torch_neuronx.utils import use_mlir_aten_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
@pytest.mark.skipif(not use_mlir_aten_ops(), reason="MLIR ATen ops not enabled")
class TestRepeatInterleave:
    """Test suite for repeat_interleave operation."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
    def test_repeat_interleave_basic_shapes(self, device, shape):
        """Test repeat_interleave with various tensor shapes."""
        with track_neuron_ops():
            x = torch.randn(shape, device=device)
            neuron_result = torch.repeat_interleave(x, 3)
            cpu_result = torch.repeat_interleave(x.cpu(), 3)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            # self_int decomposes to clone
            assert_op_runs_on_neuron("aten::clone")

    def test_repeat_interleave_autograd_self_int(self, device):
        """Test that repeat_interleave.self_int preserves autograd."""
        x = torch.randn(3, 4, device=device, requires_grad=True)
        result = torch.repeat_interleave(x, 2, dim=0)

        # Should have grad_fn from decomposed ops
        assert result.grad_fn is not None
        assert "Backward" in str(type(result.grad_fn))

    def test_repeat_interleave_autograd_self_tensor(self, device):
        """Test that repeat_interleave.self_Tensor preserves autograd."""
        x = torch.randn(3, device=device, requires_grad=True)
        repeats = torch.tensor([1, 2, 3], device=device)
        result = torch.repeat_interleave(x, repeats)

        # Should have grad_fn from index_select (IndexSelectBackward0)
        assert result.grad_fn is not None
        assert "Backward" in str(type(result.grad_fn))

    @pytest.mark.parametrize("repeats", [1, 2, 3, 5])
    def test_repeat_interleave_different_repeats(self, device, repeats):
        """Test repeat_interleave with different repeat counts."""
        with track_neuron_ops():
            x = torch.randn(4, device=device)
            neuron_result = torch.repeat_interleave(x, repeats)
            cpu_result = torch.repeat_interleave(x.cpu(), repeats)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            # self_int decomposes to clone
            assert_op_runs_on_neuron("aten::clone")

    @pytest.mark.parametrize("dim", [0, 1, -1])
    def test_repeat_interleave_with_dim(self, device, dim):
        """Test repeat_interleave with specified dimension."""
        with track_neuron_ops():
            x = torch.randn(3, 4, device=device)
            neuron_result = torch.repeat_interleave(x, 2, dim=dim)
            cpu_result = torch.repeat_interleave(x.cpu(), 2, dim=dim)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            # self_int decomposes to clone
            assert_op_runs_on_neuron("aten::clone")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32, torch.int64])
    def test_repeat_interleave_different_dtypes(self, device, dtype):
        """Test repeat_interleave with different data types."""
        with track_neuron_ops():
            x = torch.tensor([1, 2, 3], dtype=dtype, device=device)
            neuron_result = torch.repeat_interleave(x, 2)
            cpu_result = torch.repeat_interleave(x.cpu(), 2)

            assert neuron_result.device.type == "neuron"
            assert neuron_result.dtype == dtype
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            # self_int decomposes to clone
            assert_op_runs_on_neuron("aten::clone")

    def test_repeat_interleave_tensor_repeats(self, device):
        """Test repeat_interleave with tensor repeats (self_Tensor variant)."""
        with track_neuron_ops():
            x = torch.randn(3, device=device)
            repeats = torch.tensor([1, 2, 3], device=device)
            neuron_result = torch.repeat_interleave(x, repeats)
            cpu_result = torch.repeat_interleave(x.cpu(), repeats.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            # self_Tensor decomposes to: index_select(input, repeat_interleave.Tensor(repeats))
            assert_op_runs_on_neuron("aten::index_select")
            assert_op_runs_on_neuron("aten::repeat_interleave.Tensor")

    def test_repeat_interleave_empty_tensor(self, device):
        """Test repeat_interleave with empty tensor."""
        with track_neuron_ops():
            x = torch.tensor([], device=device)
            neuron_result = torch.repeat_interleave(x, 3)
            cpu_result = torch.repeat_interleave(x.cpu(), 3)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            # self_int decomposes to clone
            assert_op_runs_on_neuron("aten::clone")

    def test_repeat_interleave_single_element(self, device):
        """Test repeat_interleave with single element tensor."""
        with track_neuron_ops():
            x = torch.tensor([5.0], device=device)
            neuron_result = torch.repeat_interleave(x, 4)
            cpu_result = torch.repeat_interleave(x.cpu(), 4)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            # self_int decomposes to clone
            assert_op_runs_on_neuron("aten::clone")

    def test_repeat_interleave_with_output_size(self, device):
        """Test repeat_interleave.self_Tensor with output_size parameter."""
        with track_neuron_ops():
            x = torch.randn(3, device=device)
            repeats = torch.tensor([2, 3, 4], device=device)  # sum = 9
            neuron_result = torch.repeat_interleave(x, repeats, output_size=9)
            cpu_result = torch.repeat_interleave(x.cpu(), repeats.cpu(), output_size=9)

            assert neuron_result.device.type == "neuron"
            assert neuron_result.shape[0] == 9
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            # self_Tensor decomposes to: index_select(input, repeat_interleave.Tensor(repeats))
            assert_op_runs_on_neuron("aten::index_select")
            assert_op_runs_on_neuron("aten::repeat_interleave.Tensor")

    def test_repeat_interleave_method(self, device):
        """Test x.repeat_interleave method."""
        with track_neuron_ops():
            x = torch.randn(4, device=device)
            neuron_result = x.repeat_interleave(3)
            cpu_result = x.cpu().repeat_interleave(3)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            # self_int decomposes to clone
            assert_op_runs_on_neuron("aten::clone")

    def test_repeat_interleave_method_with_dim(self, device):
        """Test x.repeat_interleave method with dimension."""
        with track_neuron_ops():
            x = torch.randn(3, 4, device=device)
            neuron_result = x.repeat_interleave(2, dim=1)
            cpu_result = x.cpu().repeat_interleave(2, dim=1)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            # self_int decomposes to clone
            assert_op_runs_on_neuron("aten::clone")

    def test_repeat_interleave_method_tensor_repeats(self, device):
        """Test x.repeat_interleave method with tensor repeats (self_Tensor variant)."""
        with track_neuron_ops():
            x = torch.randn(3, device=device)
            repeats = torch.tensor([1, 3, 2], device=device)
            neuron_result = x.repeat_interleave(repeats)
            cpu_result = x.cpu().repeat_interleave(repeats.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            # self_Tensor decomposes to: index_select(input, repeat_interleave.Tensor(repeats))
            assert_op_runs_on_neuron("aten::index_select")
            assert_op_runs_on_neuron("aten::repeat_interleave.Tensor")

    def test_repeat_interleave_tensor_variant(self, device):
        """Test repeat_interleave.Tensor variant (standalone repeats tensor)."""
        with track_neuron_ops():
            # This calls the .Tensor variant: repeat_interleave(repeats, output_size=None)
            repeats = torch.tensor([2, 3, 1, 4], device=device)
            neuron_result = torch.repeat_interleave(repeats)
            cpu_result = torch.repeat_interleave(repeats.cpu())

            assert neuron_result.device.type == "neuron"
            # Expected result: [0, 0, 1, 1, 1, 2, 3, 3, 3, 3]
            expected = torch.tensor([0, 0, 1, 1, 1, 2, 3, 3, 3, 3], device="cpu")
            torch.testing.assert_close(neuron_result.cpu(), expected)
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::repeat_interleave.Tensor")

    def test_repeat_interleave_tensor_variant_with_output_size(self, device):
        """Test repeat_interleave.Tensor variant with output_size."""
        with track_neuron_ops():
            repeats = torch.tensor([2, 3, 1, 4], device=device)
            output_size = torch.sum(repeats).item()  # 10
            neuron_result = torch.repeat_interleave(repeats, output_size=output_size)
            cpu_result = torch.repeat_interleave(repeats.cpu(), output_size=output_size)

            assert neuron_result.device.type == "neuron"
            assert neuron_result.shape[0] == output_size
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::repeat_interleave.Tensor")

    def test_repeat_interleave_incorrect_output_size(self, device):
        """Test repeat_interleave with incorrect output_size raises RuntimeError."""
        x = torch.randn(3, device=device)
        repeats = torch.tensor([2, 3, 4], device=device)  # sum = 9

        with pytest.raises(RuntimeError, match="allocated size does not match required size"):
            torch.repeat_interleave(x, repeats, output_size=5)  # incorrect size

        # Verify same behavior on CPU
        with pytest.raises(RuntimeError, match="allocated size does not match required size"):
            torch.repeat_interleave(x.cpu(), repeats.cpu(), output_size=5)

    def test_repeat_interleave_tensor_repeats_with_trailing_zeros(self, device):
        """Test repeat_interleave with trailing zeros in repeats tensor."""
        x = torch.randn(3, device=device)
        repeats = torch.tensor([0, 2, 0], device=device)
        neuron_result = torch.repeat_interleave(x, repeats)
        cpu_result = torch.repeat_interleave(x.cpu(), repeats.cpu())

        assert neuron_result.device.type == "neuron"
        torch.testing.assert_close(neuron_result.cpu(), cpu_result)
