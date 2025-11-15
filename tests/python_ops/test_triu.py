import pytest
import torch

from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)


class TestTriu:
    def test_triu_basic(self):
        """Test triu basic functionality"""
        device = "neuron"
        with track_neuron_ops():
            x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32, device=device)
            x_cpu = x.cpu()

            result = torch.triu(x)
            result_cpu = torch.triu(x_cpu)

            assert result.device.type == device
            torch.testing.assert_close(result.cpu(), result_cpu)
            assert_op_runs_on_neuron("aten::triu")

    def test_triu_inplace(self):
        """Test triu_ (in-place operation)"""
        device = "neuron"
        with track_neuron_ops():
            x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32, device=device)
            x_cpu = torch.tensor(
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32, device="cpu"
            )

            x.triu_()
            x_cpu.triu_()

            assert x.device.type == device
            assert_op_runs_on_neuron("aten::triu.out")
            torch.testing.assert_close(x.cpu(), x_cpu)

    @pytest.mark.parametrize("diagonal", [-1, 0, 1])
    def test_triu_different_diagonals(self, diagonal):
        """Test triu with different diagonal values"""
        device = "neuron"

        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.int32, device=device)

        x_cpu = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.int32, device="cpu")

        with track_neuron_ops():
            result = torch.triu(x, diagonal=diagonal)
            result_cpu = torch.triu(x_cpu, diagonal=diagonal)

            assert result.device.type == device
            torch.testing.assert_close(result.cpu(), result_cpu)
            assert_op_runs_on_neuron("aten::triu")

    def test_triu_zero_size(self):
        """Test triu with empty tensor"""
        device = "neuron"
        with track_neuron_ops():
            x = torch.empty(0, 0, dtype=torch.float32, device=device)
            x_cpu = torch.empty(0, 0, dtype=torch.float32)

            result = torch.triu(x)
            result_cpu = torch.triu(x_cpu)

            assert result.device.type == device
            assert result.numel() == 0
            torch.testing.assert_close(result.cpu(), result_cpu)

    @assert_raises(RuntimeError)
    def test_triu_1d_input(self):
        """Test triu with 1D input (should fail)"""
        device = "neuron"
        x = torch.tensor([1, 2, 3], device=device)
        torch.triu(x)
