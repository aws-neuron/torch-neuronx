import pytest
import torch

import torch_neuronx  # Registers the neuron device and operations
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


def test_ones_basic_shape_and_device():
    with track_neuron_ops():
        x = torch.ones((2, 3), device="neuron:0")
        assert x.device.type == "neuron"
        assert x.device.index == 0
        assert tuple(x.shape) == (2, 3)
        assert_op_runs_on_neuron("aten::ones")


def test_ones_scalar_value():
    with track_neuron_ops():
        x = torch.ones((), device="neuron:0")
        assert x.device.type == "neuron"
        assert x.dim() == 0
        assert x.to("cpu").item() == 1
        assert_op_runs_on_neuron("aten::ones")


def test_ones_default_dtype():
    expected = torch.get_default_dtype()
    with track_neuron_ops():
        x = torch.ones((4, 5), device="neuron:0")
        assert x.dtype == expected
        assert_op_runs_on_neuron("aten::ones")


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.int32, torch.bool],
)
def test_ones_explicit_dtype(dtype):
    with track_neuron_ops():
        x = torch.ones((3, 2), dtype=dtype, device="neuron:0")
        assert x.device.type == "neuron"
        assert x.dtype == dtype
        # Validate values on CPU for a couple of elements
        cpu = x.to("cpu")
        assert cpu[0, 0].item() == 1
        assert cpu[-1, -1].item() == 1
        assert_op_runs_on_neuron("aten::ones")


def test_ones_out_parameter_resize_and_fill():
    with track_neuron_ops():
        # Prepare an out tensor with mismatched shape and dtype
        out = torch.empty((1,), dtype=torch.float32, device="neuron:0")
        # For .out overload, do not pass device kwarg
        y = torch.ones((2, 3), out=out)
        # Should return the same tensor object and resize it to (2, 3)
        assert y is out
        assert tuple(out.shape) == (2, 3)
        # All values should be 1
        cpu = out.to("cpu")
        assert torch.all(cpu == 1)
        assert_op_runs_on_neuron("aten::ones")


def test_ones_matches_cpu_result_after_move():
    shape = (2, 4)
    with track_neuron_ops():
        x_neuron = torch.ones(shape, dtype=torch.float32, device="neuron:0")
        x_cpu = torch.ones(shape, dtype=torch.float32, device="cpu")
    torch.testing.assert_close(x_neuron.to("cpu"), x_cpu)
    assert_op_runs_on_neuron("aten::ones")


def test_ones_zero_size_runs_on_neuron():
    with track_neuron_ops():
        x = torch.ones(0, device="neuron:0")
        assert x.device.type == "neuron"
        assert x.numel() == 0
        assert tuple(x.shape) == (0,)
        assert x.dtype == torch.get_default_dtype()
        assert_op_runs_on_neuron("aten::ones")


def test_new_empty_explicit_cpu_device():
    """Verify new_empty with device='cpu' on a neuron tensor returns a CPU tensor."""
    x = torch.randn(4, device="neuron:0")
    y = x.new_empty((2, 3), device="cpu")
    assert y.device.type == "cpu"
    assert y.shape == (2, 3)


def test_empty_explicit_cpu_device():
    """Verify torch.empty with device='cpu' doesn't break when neuron is loaded."""
    y = torch.empty((2, 3), device="cpu")
    assert y.device.type == "cpu"
    assert y.shape == (2, 3)
