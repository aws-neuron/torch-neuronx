"""
E2E tests for convolution operations under torch.compile(backend='neuron').

Covers Conv1d, Conv2d, and Conv3d with various configurations.
"""

import pytest
import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

assert torch.neuron.device_count() > 0, "No neuron devices were discovered."


class FunctionalConv2dScalarStride(nn.Module):
    """Uses F.conv2d with single-element list stride on 4D input."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(16, 3, 3, 3))
        self.bias = nn.Parameter(torch.randn(16))

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, stride=[2], padding=[1], dilation=[1])


class FunctionalConv2dScalarDilation(nn.Module):
    """Uses F.conv2d with single-element list dilation on 4D input."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(16, 3, 3, 3))
        self.bias = nn.Parameter(torch.randn(16))

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, stride=[1], padding=[2], dilation=[2])


class FunctionalConv2dAllScalar(nn.Module):
    """Uses F.conv2d with all single-element list params on 4D input."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(32, 3, 5, 5))
        self.bias = nn.Parameter(torch.randn(32))

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, stride=[2], padding=[2], dilation=[1])


class FunctionalConv3dScalarStride(nn.Module):
    """Uses F.conv3d with single-element list stride on 5D input."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 3, 3, 3, 3))
        self.bias = nn.Parameter(torch.randn(8))

    def forward(self, x):
        return F.conv3d(x, self.weight, self.bias, stride=[2], padding=[1], dilation=[1])


class FunctionalConv3dAllScalar(nn.Module):
    """Uses F.conv3d with all single-element list params on 5D input."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 3, 3, 3, 3))
        self.bias = nn.Parameter(torch.randn(8))

    def forward(self, x):
        return F.conv3d(x, self.weight, self.bias, stride=[2], padding=[1], dilation=[2])


CONV_CONFIGS = [
    # --- Standard nn.ConvNd (params expanded by module __init__) ---
    pytest.param(
        lambda: nn.Conv1d(3, 8, 3, stride=1, padding=1),
        (2, 3, 16),
        id="conv1d_basic",
    ),
    pytest.param(
        lambda: nn.Conv1d(3, 8, 3, stride=2, padding=1),
        (2, 3, 16),
        id="conv1d_stride2",
    ),
    pytest.param(
        lambda: nn.Conv2d(3, 16, 3, stride=(2, 2), padding=(1, 1)),
        (1, 3, 32, 32),
        id="conv2d_tuple_params",
    ),
    pytest.param(
        lambda: nn.Conv2d(3, 16, 3, stride=1, padding=1, dilation=(2, 2)),
        (1, 3, 32, 32),
        id="conv2d_tuple_dilation",
    ),
    pytest.param(
        lambda: nn.Conv2d(3, 16, 3, stride=2, padding=1),
        (1, 3, 32, 32),
        id="conv2d_stride2",
    ),
    pytest.param(
        lambda: nn.Conv2d(4, 8, 3, stride=2, padding=1, groups=2),
        (1, 4, 32, 32),
        id="conv2d_groups",
    ),
    pytest.param(
        lambda: nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
        (1, 3, 32, 32),
        id="conv2d_no_bias",
    ),
    pytest.param(
        lambda: nn.Conv3d(3, 8, 3, stride=2, padding=1),
        (1, 3, 8, 8, 8),
        id="conv3d_stride2",
    ),
    pytest.param(
        lambda: nn.Conv3d(3, 8, 3, stride=1, padding=2, dilation=2),
        (1, 3, 16, 16, 16),
        id="conv3d_dilation2",
    ),
    # --- Single-element list params via F.conv2d/F.conv3d ---
    # These can bypass normalization and hit torch-mlir with [N] instead
    # of [N, N], exercising the broadcast fix.
    pytest.param(
        FunctionalConv2dScalarStride,
        (1, 3, 32, 32),
        id="conv2d_scalar_stride_functional",
    ),
    pytest.param(
        FunctionalConv2dScalarDilation,
        (1, 3, 32, 32),
        id="conv2d_scalar_dilation_functional",
    ),
    pytest.param(
        FunctionalConv2dAllScalar,
        (1, 3, 64, 64),
        id="conv2d_scalar_all_functional",
    ),
    pytest.param(
        FunctionalConv3dScalarStride,
        (1, 3, 8, 8, 8),
        id="conv3d_scalar_stride_functional",
    ),
    pytest.param(
        FunctionalConv3dAllScalar,
        (1, 3, 16, 16, 16),
        id="conv3d_scalar_all_functional",
    ),
]


class TestConvE2E:
    """E2E tests for convolution under torch.compile(backend='neuron')."""

    @pytest.mark.parametrize("model_factory,input_shape", CONV_CONFIGS)
    def test_conv_forward(self, model_factory, input_shape):
        torch._dynamo.reset()
        torch.manual_seed(42)

        model = model_factory().eval()
        x = torch.randn(*input_shape)

        with torch.no_grad():
            cpu_out = model(x)

        compiled_model = torch.compile(model.to("neuron"), backend="neuron")
        with torch.no_grad():
            neuron_out = compiled_model(x.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=1e-2, atol=1e-2)
