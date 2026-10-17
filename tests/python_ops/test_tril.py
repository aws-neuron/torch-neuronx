import pytest
import torch

from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)


class TestTril:
    def test_tril_basic(self):
        """Test tril basic functionality"""
        device = "neuron"
        with track_neuron_ops():
            x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32, device=device)
            x_cpu = x.cpu()

            result = torch.tril(x)
            result_cpu = torch.tril(x_cpu)

            assert result.device.type == device
            torch.testing.assert_close(result.cpu(), result_cpu)
            assert_op_runs_on_neuron("aten::tril")

    def test_tril_inplace(self):
        """Test tril_ (in-place operation)"""
        device = "neuron"
        with track_neuron_ops():
            x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32, device=device)
            x_cpu = torch.tensor(
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32, device="cpu"
            )

            x.tril_()
            x_cpu.tril_()

            assert x.device.type == device
            assert_op_runs_on_neuron("tril.out")
            torch.testing.assert_close(x.cpu(), x_cpu)

    def test_tril_different_diagonals(self):
        """Test tril with different diagonal values"""
        device = "neuron"

        for diagonal in [-1, 0, 1]:
            x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.int32, device=device)

            x_cpu = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.int32, device="cpu")

            with track_neuron_ops():
                result = torch.tril(x, diagonal=diagonal)
                result_cpu = torch.tril(x_cpu, diagonal=diagonal)

                assert result.device.type == device
                torch.testing.assert_close(result.cpu(), result_cpu)
                assert_op_runs_on_neuron("aten::tril")

    def test_tril_zero_size(self):
        """Test tril with empty tensor"""
        device = "neuron"
        with track_neuron_ops():
            x = torch.empty(0, 0, dtype=torch.float32, device=device)
            x_cpu = torch.empty(0, 0, dtype=torch.float32)

            result = torch.tril(x)
            result_cpu = torch.tril(x_cpu)

            assert result.device.type == device
            assert result.numel() == 0
            torch.testing.assert_close(result.cpu(), result_cpu)

    @assert_raises(RuntimeError)
    def test_tril_1d_input(self):
        """Test tril with 1D input (should fail)"""
        device = "neuron"
        x = torch.tensor([1, 2, 3], device=device)
        torch.tril(x)
