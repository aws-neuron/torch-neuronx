"""Tests for convolution output memory-format contracts."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

pytestmark = pytest.mark.skipif(
    torch_neuronx.device_count() == 0, reason="No Neuron devices available"
)


def _assert_close(actual, expected, *, rtol=1.3e-6, atol=1e-5):
    torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)


def test_convolution_separates_contiguous_and_channels_last_cache_entries():
    torch.manual_seed(42)
    cpu_module = torch.nn.Conv2d(3, 4, 3).eval()
    neuron_module = torch.nn.Conv2d(3, 4, 3).eval()
    neuron_module.load_state_dict(cpu_module.state_dict())
    neuron_module.to("neuron")
    input = torch.randn(2, 3, 8, 8)

    for memory_format in (
        torch.contiguous_format,
        torch.channels_last,
        torch.channels_last,
        torch.contiguous_format,
    ):
        logical_input = input.contiguous(memory_format=memory_format)
        with track_neuron_ops():
            actual = neuron_module(logical_input.to("neuron"))
        assert_op_runs_on_neuron("aten::convolution")

        assert actual.is_contiguous(memory_format=memory_format)
        _assert_close(actual, cpu_module(logical_input))


@pytest.mark.parametrize(
    "module_cls,memory_format,input_shape",
    [
        (torch.nn.Conv2d, torch.channels_last, (2, 3, 8, 8)),
        (torch.nn.ConvTranspose2d, torch.channels_last, (2, 3, 8, 8)),
        (torch.nn.Conv3d, torch.channels_last_3d, (2, 3, 5, 6, 7)),
        (torch.nn.ConvTranspose3d, torch.channels_last_3d, (2, 3, 5, 6, 7)),
    ],
)
def test_formatted_weight_preserves_convolution_output_layout(
    module_cls, memory_format, input_shape
):
    torch.manual_seed(42)
    cpu_module = module_cls(3, 4, 3).eval()
    neuron_module = module_cls(3, 4, 3).eval()
    neuron_module.load_state_dict(cpu_module.state_dict())
    neuron_module.to("neuron")
    neuron_module.weight.data = neuron_module.weight.detach().contiguous(
        memory_format=memory_format
    )
    input = torch.randn(input_shape)

    actual = neuron_module(input.to("neuron"))

    assert actual.is_contiguous(memory_format=memory_format)
    _assert_close(actual, cpu_module(input))


@pytest.mark.parametrize(
    "module_cls,memory_format,input_shape",
    [
        (torch.nn.Conv2d, torch.channels_last, (0, 3, 8, 8)),
        (torch.nn.Conv3d, torch.channels_last_3d, (0, 3, 5, 6, 7)),
    ],
)
def test_empty_convolution_preserves_output_layout(module_cls, memory_format, input_shape):
    module = module_cls(3, 4, 3).eval().to("neuron")
    input = torch.empty(input_shape, device="neuron", memory_format=memory_format)

    actual = module(input)

    assert actual.numel() == 0
    assert actual.is_contiguous(memory_format=memory_format)


def test_float64_convolution_preserves_channels_last_after_upcast():
    torch.manual_seed(42)
    cpu_module = torch.nn.Conv2d(3, 4, 3).double().eval()
    neuron_module = torch.nn.Conv2d(3, 4, 3).double().eval()
    neuron_module.load_state_dict(cpu_module.state_dict())
    neuron_module.to("neuron")
    input = torch.randn(2, 3, 8, 8, dtype=torch.float64).contiguous(
        memory_format=torch.channels_last
    )

    actual = neuron_module(input.to("neuron"))

    assert actual.is_contiguous(memory_format=torch.channels_last)
    _assert_close(actual, cpu_module(input), rtol=1e-5, atol=1e-5)
