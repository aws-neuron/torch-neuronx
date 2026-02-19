"""Tests for nonzero operation."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestNonzero:
    """Test suite for nonzero operation."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
    def test_nonzero_basic_shapes(self, device, shape):
        """Test nonzero with various tensor shapes."""
        with track_neuron_ops():
            x = torch.randint(0, 2, shape, device=device)
            neuron_result = torch.nonzero(x)
            cpu_result = torch.nonzero(x.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::nonzero")

    def test_nonzero_all_zeros(self, device):
        """Test nonzero with all zeros tensor."""
        with track_neuron_ops():
            x = torch.zeros(3, 4, device=device)
            neuron_result = torch.nonzero(x)
            cpu_result = torch.nonzero(x.cpu())

            assert neuron_result.device.type == "neuron"
            assert neuron_result.shape == cpu_result.shape
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::nonzero")

    @pytest.mark.parametrize("shape", [(0,), (0, 3), (2, 0), (0, 0)])
    def test_nonzero_empty_tensor(self, device, shape):
        """Test nonzero with empty tensor (numel == 0)."""
        x = torch.empty(shape, device=device)
        neuron_result = torch.nonzero(x)
        cpu_result = torch.nonzero(x.cpu())

        assert neuron_result.device.type == "neuron"
        assert neuron_result.shape == cpu_result.shape
        assert neuron_result.shape == (0, len(shape))
        torch.testing.assert_close(neuron_result.cpu(), cpu_result)

    def test_nonzero_all_ones(self, device):
        """Test nonzero with all ones tensor."""
        with track_neuron_ops():
            x = torch.ones(2, 3, device=device)
            neuron_result = torch.nonzero(x)
            cpu_result = torch.nonzero(x.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::nonzero")

    @pytest.mark.parametrize("as_tuple", [True, False])
    def test_nonzero_as_tuple(self, device, as_tuple):
        """Test nonzero with as_tuple parameter."""
        with track_neuron_ops():
            x = torch.randint(0, 2, (3, 4), device=device)
            neuron_result = torch.nonzero(x, as_tuple=as_tuple)
            cpu_result = torch.nonzero(x.cpu(), as_tuple=as_tuple)

            if as_tuple:
                assert isinstance(neuron_result, tuple)
                assert len(neuron_result) == len(cpu_result)
                for nr, cr in zip(neuron_result, cpu_result, strict=False):
                    assert nr.device.type == "neuron"
                    torch.testing.assert_close(nr.cpu(), cr)
            else:
                assert neuron_result.device.type == "neuron"
                torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::nonzero")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.bool])
    def test_nonzero_different_dtypes(self, device, dtype):
        """Test nonzero with different data types."""
        with track_neuron_ops():
            if dtype == torch.bool:
                x = torch.randint(0, 2, (3, 4), device=device).bool()
            else:
                x = torch.randint(0, 2, (3, 4), dtype=dtype, device=device)
            neuron_result = torch.nonzero(x)
            cpu_result = torch.nonzero(x.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::nonzero")

    def test_nonzero_single_element(self, device):
        """Test nonzero with single element tensor."""
        with track_neuron_ops():
            x = torch.tensor([1.0], device=device)
            neuron_result = torch.nonzero(x)
            cpu_result = torch.nonzero(x.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::nonzero")

    def test_nonzero_with_out(self, device):
        """Test nonzero with out parameter."""
        with track_neuron_ops():
            x = torch.randint(0, 2, (3, 4), device=device)
            # Get expected shape first
            expected_shape = torch.nonzero(x.cpu()).shape
            out = torch.empty(expected_shape, dtype=torch.long, device=device)
            result = torch.nonzero(x, out=out)
            cpu_result = torch.nonzero(x.cpu())

            assert result is out
            assert out.device.type == "neuron"
            torch.testing.assert_close(out.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::nonzero")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.bool])
    def test_nonzero_with_out_wrong_dtype(self, device, dtype):
        """Test nonzero with out parameter of wrong dtype."""
        with pytest.raises(RuntimeError):
            x = torch.randint(0, 2, (3, 4), device=device)
            out = torch.empty((1, 2), dtype=dtype, device=device)
            torch.nonzero(x, out=out)

    def test_nonzero_as_tuple_with_out(self, device):
        """Test nonzero with out parameter of wrong dtype."""
        with pytest.raises(RuntimeError):
            x = torch.randint(0, 2, (3, 4), device=device)
            out = torch.empty((1, 2), dtype=torch.int64, device=device)
            torch.nonzero(x, out=out, as_tuple=True)
