import pytest
import torch

from torch_neuronx.python_ops.lowering_result import LoweringResult
from torch_neuronx.python_ops.torch_mlir.ops.convolution import (
    _set_convolution_output_strides,
)
from torch_neuronx.python_ops.torch_mlir.ops.normalization import (
    _set_batch_norm_output_strides,
)


@pytest.mark.parametrize(
    "shape,memory_format,expected_strides",
    [
        ((2, 4, 6, 7), torch.channels_last, (168, 1, 28, 4)),
        ((2, 4, 3, 6, 7), torch.channels_last_3d, (504, 1, 168, 28, 4)),
    ],
)
@pytest.mark.parametrize("formatted_arg", [0, 1], ids=["input", "weight"])
def test_convolution_requests_fused_channels_last_output(
    shape, memory_format, expected_strides, formatted_arg
):
    input = torch.empty((2, 3, *shape[2:]))
    weight = torch.empty((shape[1], 3, *(3 for _ in shape[2:])))
    args = [input, weight]
    args[formatted_arg] = args[formatted_arg].contiguous(memory_format=memory_format)
    result = LoweringResult(output_specs=[(shape, torch.float32)])

    _set_convolution_output_strides(result, tuple(args), {})

    assert result.output_specs == [(shape, torch.float32, expected_strides)]


def test_convolution_leaves_contiguous_output_spec_unchanged():
    shape = (2, 4, 6, 7)
    result = LoweringResult(output_specs=[(shape, torch.float32)])

    _set_convolution_output_strides(
        result,
        (torch.empty((2, 3, 6, 7)), torch.empty((4, 3, 1, 1))),
        {},
    )

    assert result.output_specs == [(shape, torch.float32)]


def test_convolution_ignores_ambiguous_channels_last_weight():
    shape = (2, 4, 6, 7)
    weight = torch.empty((4, 1, 1, 1)).contiguous(memory_format=torch.channels_last)
    result = LoweringResult(output_specs=[(shape, torch.float32)])

    _set_convolution_output_strides(result, (torch.empty((2, 1, 6, 7)), weight), {})

    assert result.output_specs == [(shape, torch.float32)]


@pytest.mark.parametrize(
    "input_shape,output_shape,memory_format,expected_strides",
    [
        ((0, 3, 8, 8), (0, 4, 6, 6), torch.channels_last, (144, 1, 24, 4)),
        (
            (0, 3, 5, 6, 7),
            (0, 4, 3, 4, 5),
            torch.channels_last_3d,
            (240, 1, 80, 20, 4),
        ),
    ],
)
def test_empty_convolution_requests_channels_last_output(
    input_shape, output_shape, memory_format, expected_strides
):
    input = torch.empty(input_shape, memory_format=memory_format)
    weight = torch.empty((output_shape[1], input_shape[1], *(3 for _ in input_shape[2:])))
    result = LoweringResult(output_specs=[(output_shape, torch.float32)])

    _set_convolution_output_strides(result, (input, weight), {})

    assert result.output_specs == [(output_shape, torch.float32, expected_strides)]


@pytest.mark.parametrize(
    "shape,memory_format,expected_strides",
    [
        ((2, 4, 6, 7), torch.channels_last, (168, 1, 28, 4)),
        ((2, 4, 3, 6, 7), torch.channels_last_3d, (504, 1, 168, 28, 4)),
    ],
)
def test_batch_norm_preserves_channels_last_output(shape, memory_format, expected_strides):
    input = torch.empty(shape).contiguous(memory_format=memory_format)
    result = LoweringResult(
        output_specs=[
            (shape, torch.float32),
            ((shape[1],), torch.float32),
            ((shape[1],), torch.float32),
        ]
    )

    _set_batch_norm_output_strides(result, (input,), {})

    assert result.output_specs == [
        (shape, torch.float32, expected_strides),
        ((shape[1],), torch.float32),
        ((shape[1],), torch.float32),
    ]


@pytest.mark.parametrize(
    "shape,memory_format,expected_strides",
    [
        ((0, 4, 6, 7), torch.channels_last, (168, 1, 28, 4)),
        ((0, 4, 3, 6, 7), torch.channels_last_3d, (504, 1, 168, 28, 4)),
    ],
)
def test_empty_batch_norm_preserves_channels_last_output(shape, memory_format, expected_strides):
    input = torch.empty(shape, memory_format=memory_format)
    result = LoweringResult(
        output_specs=[
            (shape, torch.float32),
            ((shape[1],), torch.float32),
            ((shape[1],), torch.float32),
        ]
    )

    _set_batch_norm_output_strides(result, (input,), {})

    assert result.output_specs == [
        (shape, torch.float32, expected_strides),
        ((shape[1],), torch.float32),
        ((shape[1],), torch.float32),
    ]


@pytest.mark.parametrize("out_name", ["out", "out0", "save_mean", "save_invstd"])
def test_batch_norm_preserves_preallocated_output_layout(out_name):
    shape = (2, 4, 6, 7)
    result = LoweringResult(
        output_specs=[
            (shape, torch.float32),
            ((shape[1],), torch.float32),
            ((shape[1],), torch.float32),
        ]
    )

    _set_batch_norm_output_strides(
        result,
        (torch.empty(shape).contiguous(memory_format=torch.channels_last),),
        {out_name: torch.empty(shape)},
    )

    assert result.output_specs == [
        (shape, torch.float32),
        ((shape[1],), torch.float32),
        ((shape[1],), torch.float32),
    ]
