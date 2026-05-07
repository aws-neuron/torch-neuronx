import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)


class TestFill:
    def test_fill_basic(self):
        """Test basic fill operation"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.ones(3, 4, device=device)
            x_cpu = torch.ones(3, 4)

            x_neuron.fill_(5)
            x_cpu.fill_(5)

            torch.testing.assert_close(x_neuron.cpu(), x_cpu)
            assert_op_runs_on_neuron("aten::fill_")

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float16, torch.int32, torch.int64, torch.float64]
    )
    def test_fill_different_dtypes(self, dtype):
        """Test fill with different dtypes"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.ones(2, 3, dtype=dtype, device=device)
            x_cpu = torch.ones(2, 3, dtype=dtype)

            x_neuron.fill_(7)
            x_cpu.fill_(7)

            torch.testing.assert_close(x_neuron.cpu(), x_cpu)
            assert_op_runs_on_neuron("aten::fill_")

    @pytest.mark.parametrize(
        "scalar_types",
        [
            # Integer fill
            42,
            # Float fill
            3.14,
            # Boolean fill
            False,
        ],
    )
    def test_fill_scalar_types(self, scalar_types):
        """Test fill with different scalar types"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.ones(2, 2, dtype=torch.float32, device=device)
            x_cpu = torch.ones(2, 2, dtype=torch.float32)
            x_neuron.fill_(scalar_types)
            x_cpu.fill_(scalar_types)
            torch.testing.assert_close(x_neuron.cpu(), x_cpu)
            assert_op_runs_on_neuron("aten::fill_")

    def test_fill_empty_tensor(self):
        """Test fill with empty tensor"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.empty(0, 5, device=device)
            x_cpu = torch.empty(0, 5)

            x_neuron.fill_(-1)
            x_cpu.fill_(-1)

            torch.testing.assert_close(x_neuron.cpu(), x_cpu)
            assert_op_runs_on_neuron("aten::fill_")

    def test_fill_scalar_tensor(self):
        """Test fill with scalar tensor"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.tensor(5, device=device)
            x_cpu = torch.tensor(3)

            x_neuron.fill_(10)
            x_cpu.fill_(10)

            torch.testing.assert_close(x_neuron.cpu(), x_cpu)
            assert_op_runs_on_neuron("aten::fill_")

    def test_fill_with_nan_inf(self):
        """Test fill with NaN and Inf"""
        device = "neuron"
        # Test with NaN
        x = torch.zeros(3, 4, dtype=torch.float32, device=device)
        x.fill_(float("nan"))
        assert torch.isnan(x).all()

        # Test with Inf
        y = torch.zeros(3, 4, dtype=torch.float32, device=device)
        y.fill_(float("inf"))
        assert torch.isinf(y).all()
        assert_op_runs_on_neuron("aten::fill_")

    @assert_raises(TypeError, match=r"fill_\(\) received an invalid combination of arguments")
    def test_fill_with_list_value(self):
        """Test fill with list value"""
        x = torch.zeros(3, 4, device="neuron")
        x.fill_([1, 2, 3])

    @assert_raises(RuntimeError)
    def test_fill_with_1d_tensor_value(self):
        """Test fill with 1-D tensor value"""
        device = "neuron"
        x = torch.zeros(3, 4, device=device)
        # 1-D tensor value is invalid for fill_.Tensor (expects 0-D); PyTorch raises
        # a dimension-specific RuntimeError on CPU. Mirror that behavior.
        x.fill_(torch.tensor([1, 2], device=device))

    @assert_raises(TypeError, match=r"fill_\(\) received an invalid combination of arguments")
    def test_fill_with_dict_value(self):
        """Test fill with dict value"""
        x = torch.zeros(3, 4, device="neuron")
        x.fill_({"value": 5})

    @assert_raises(TypeError)
    def test_fill_with_none(self):
        """Test fill with None value"""
        device = "neuron"
        x = torch.zeros(3, 4, device=device)
        x.fill_(None)
