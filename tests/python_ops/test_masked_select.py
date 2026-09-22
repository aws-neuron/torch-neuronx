"""Tests for masked_select operation."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, assert_raises, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestMaskedSelect:
    """Test suite for masked_select operation."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
    def test_masked_select_basic_shapes(self, device, shape):
        """Test masked_select with various tensor shapes."""
        with track_neuron_ops():
            x = torch.randn(shape, device=device)
            mask = torch.randint(0, 2, shape, device=device).bool()
            neuron_result = torch.masked_select(x, mask)
            cpu_result = torch.masked_select(x.cpu(), mask.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_select")

    def test_masked_select_all_false_mask(self, device):
        """Test masked_select with all False mask."""
        with track_neuron_ops():
            x = torch.randn(3, 4, device=device)
            mask = torch.zeros(3, 4, device=device).bool()
            neuron_result = torch.masked_select(x, mask)
            cpu_result = torch.masked_select(x.cpu(), mask.cpu())

            assert neuron_result.device.type == "neuron"
            assert neuron_result.shape == cpu_result.shape
            # Op will be short-circuited with all false mask
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)

    def test_masked_select_stale_neff_regression(self, device):
        """Two masked_select calls with the same mask shape but different
        true-counts must not share a cached NEFF.

        The output length is mask.sum(), baked in as a static arg. The raw
        dispatch args are identical for both calls (same-shape float32 input and
        bool mask), so the lowering must key on the preprocessed size; otherwise
        the second call returns the first call's length.
        """
        with track_neuron_ops():
            x = torch.arange(8, dtype=torch.float32, device=device)

            # First call: 3 True -> output length 3.
            mask_a = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0], device=device).bool()
            result_a = torch.masked_select(x, mask_a)
            assert result_a.shape == torch.Size([3])

            # Second call: same mask shape, 5 True -> output length 5. A stale
            # size would return length 3.
            mask_b = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0], device=device).bool()
            result_b = torch.masked_select(x, mask_b)
            assert result_b.shape == torch.Size([5])
            torch.testing.assert_close(result_b.cpu(), torch.masked_select(x.cpu(), mask_b.cpu()))

    def test_masked_select_all_true_mask(self, device):
        """Test masked_select with all True mask."""
        with track_neuron_ops():
            x = torch.randn(2, 3, device=device)
            mask = torch.ones(2, 3, device=device).bool()
            neuron_result = torch.masked_select(x, mask)
            cpu_result = torch.masked_select(x.cpu(), mask.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_select")

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float64,
            torch.float16,
            torch.bfloat16,
            torch.int32,
            torch.int64,
            torch.bool,
        ],
    )
    def test_masked_select_different_dtypes(self, device, dtype):
        """Test masked_select with different data types."""
        with track_neuron_ops():
            if dtype == torch.bool:
                x = torch.randint(0, 2, (3, 4), device=device).bool()
            elif dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
                x = torch.randn(3, 4, dtype=dtype, device=device)
            else:
                x = torch.randint(0, 10, (3, 4), dtype=dtype, device=device)
            mask = torch.randint(0, 2, (3, 4), device=device).bool()
            neuron_result = torch.masked_select(x, mask)
            cpu_result = torch.masked_select(x.cpu(), mask.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_select")

    def test_masked_select_broadcast_mask(self, device):
        """Test masked_select with broadcastable mask."""
        with track_neuron_ops():
            x = torch.randn(3, 4, device=device)
            mask = torch.tensor([True, False, True, False], device=device)
            neuron_result = torch.masked_select(x, mask)
            cpu_result = torch.masked_select(x.cpu(), mask.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_select")

    def test_masked_select_broadcast_input(self, device):
        """Test masked_select with broadcastable input tensor."""
        with track_neuron_ops():
            x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
            mask = torch.randint(0, 2, (3, 4), device=device).bool()
            neuron_result = torch.masked_select(x, mask)
            cpu_result = torch.masked_select(x.cpu(), mask.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_select")

    def test_masked_select_single_element(self, device):
        """Test masked_select with single element tensor."""
        with track_neuron_ops():
            x = torch.tensor([1.0], device=device)
            mask = torch.tensor([True], device=device)
            neuron_result = torch.masked_select(x, mask)
            cpu_result = torch.masked_select(x.cpu(), mask.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_select")

    @pytest.mark.parametrize("mask_value", [True, False])
    def test_masked_select_scalar_self_and_mask(self, device, mask_value):
        """Test masked_select when both self and mask are 0-dimensional.

        The broadcast shape is () -- with no coordinate axis, the nonzero_static
        decomposition used by the lowering has nothing to stack (mask False) and
        no last dim to size (mask True), so the lowering views both operands as
        1-element vectors. Result is 1-d either way: shape (1,) or (0,).
        """
        with track_neuron_ops():
            x = torch.randn((), device=device)
            mask = torch.tensor(mask_value, device=device)
            neuron_result = torch.masked_select(x, mask)
            cpu_result = torch.masked_select(x.cpu(), mask.cpu())

            assert neuron_result.device.type == "neuron"
            assert neuron_result.shape == cpu_result.shape
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_select")

    def test_masked_select_scalar_self_scalar_mask_out(self, device):
        """Test the out= variant with 0-dimensional self and mask."""
        with track_neuron_ops():
            x = torch.randn((), device=device)
            mask = torch.tensor(True, device=device)
            out = torch.empty(1, dtype=x.dtype, device=device)
            result = torch.masked_select(x, mask, out=out)
            cpu_result = torch.masked_select(x.cpu(), mask.cpu())

            assert result is out
            torch.testing.assert_close(out.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_select")

    def test_masked_select_scalar_backward(self, device):
        """masked_select backward with 0-dimensional self and mask.

        This is the path masked_fill's own backward takes for a 0-d tensor
        ``value`` (grad_value = grad.masked_select(mask).sum()).
        """
        x = torch.randn((), device=device, requires_grad=True)
        mask = torch.tensor(True, device=device)
        torch.masked_select(x, mask).sum().backward()

        x_cpu = x.detach().cpu().requires_grad_(True)
        torch.masked_select(x_cpu, mask.cpu()).sum().backward()
        torch.testing.assert_close(x.grad.cpu(), x_cpu.grad)

    def test_masked_select_large_3d_shape(self, device):
        """Test masked_select with large 3D tensor (1, 300, 2048)."""
        with track_neuron_ops():
            shape = (1, 300, 2048)
            x = torch.randn(shape, device=device)
            mask = torch.randint(0, 2, shape, device=device).bool()
            neuron_result = torch.masked_select(x, mask)
            cpu_result = torch.masked_select(x.cpu(), mask.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_select")

    def test_masked_select_with_out(self, device):
        """Test masked_select with out parameter."""
        with track_neuron_ops():
            x = torch.randn(3, 4, device=device)
            mask = torch.randint(0, 2, (3, 4), device=device).bool()
            # Get expected shape first
            expected_shape = torch.masked_select(x.cpu(), mask.cpu()).shape
            out = torch.empty(expected_shape, dtype=x.dtype, device=device)
            result = torch.masked_select(x, mask, out=out)
            cpu_result = torch.masked_select(x.cpu(), mask.cpu())

            assert result is out
            assert out.device.type == "neuron"
            torch.testing.assert_close(out.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_select")

    @pytest.mark.parametrize(
        "mask_dtype", [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]
    )
    @assert_raises(RuntimeError, match=r"expected BoolTensor for mask")
    def test_masked_select_non_bool_mask_raises(self, device, mask_dtype):
        """Test masked_select raises RuntimeError when mask is not a bool tensor.

        CPU raises: 'expected BoolTensor for mask'
        Neuron must match.
        """
        x = torch.randn(10, device=device)
        mask = torch.randint(0, 2, (10,), device=device, dtype=mask_dtype)
        torch.masked_select(x, mask)
