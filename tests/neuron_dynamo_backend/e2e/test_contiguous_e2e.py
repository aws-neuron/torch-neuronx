"""
E2E tests for contiguous operations verifying correct forward and backward behavior.
"""

import pytest
import torch
import torch.nn as nn

# Check if neuron device is available
assert torch.neuron.device_count() > 0, "No neuron devices were discovered."


class ContiguousReshapeMatmul(nn.Module):
    """Model with transpose -> contiguous -> reshape -> matmul."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 4))

    def forward(self, x):
        y = x.transpose(0, 1)
        z = y.contiguous()
        r = z.reshape(4, -1)
        return r


class MultipleContiguous(nn.Module):
    """Model with multiple contiguous ops and reshapes."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(16, 8))

    def forward(self, x):
        y = x.transpose(0, 1).contiguous()
        r1 = y.reshape(4, 4)
        z = r1.permute(1, 0).contiguous()
        r2 = z.reshape(16, 1)
        return torch.matmul(r2.T, self.weight)


class AtenContiguousReshapeMatmul(nn.Module):
    """Model using torch.ops.aten.contiguous.default."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 4))

    def forward(self, x):
        y = x.transpose(0, 1)
        z = torch.ops.aten.contiguous.default(y)
        r = z.reshape(4, -1)
        return torch.matmul(r, self.weight)


class MixedContiguousStyles(nn.Module):
    """Model with both tensor.contiguous() and torch.ops.aten.contiguous."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 4))

    def forward(self, x):
        y = x.transpose(0, 1)
        z = y.contiguous()
        w = torch.ops.aten.contiguous.default(z)
        r = w.reshape(4, -1)
        return torch.matmul(r, self.weight)


class ChannelsLastContiguous(nn.Module):
    """Model with channels_last memory format and contiguous."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x):
        y = x.to(memory_format=torch.preserve_format).contiguous()
        z = self.conv(y)
        return z.reshape(z.size(0), -1).sum(dim=1)


class TestContiguousE2E:
    """E2E tests verifying contiguous operations work correctly with real compilation."""

    @pytest.mark.parametrize(
        "model_cls,input_shape,tol",
        [
            pytest.param(ContiguousReshapeMatmul, (4, 8), 1e-4, id="contiguous_reshape_matmul"),
            pytest.param(MultipleContiguous, (4, 4), 1e-4, id="multiple_contiguous"),
            pytest.param(ChannelsLastContiguous, (2, 3, 8, 8), 1e-3, id="channels_last_contiguous"),
            pytest.param(
                AtenContiguousReshapeMatmul, (4, 8), 1e-4, id="aten_contiguous_reshape_matmul"
            ),
            pytest.param(MixedContiguousStyles, (4, 8), 1e-4, id="mixed_contiguous_styles"),
        ],
    )
    def test_forward(self, model_cls, input_shape, tol):
        """Test forward pass produces correct output with contiguous operations."""
        cpu_model = model_cls()
        x = torch.randn(*input_shape)
        with torch.no_grad():
            cpu_out = cpu_model(x)
        with torch.no_grad():
            compiled_model = torch.compile(cpu_model.to("neuron"), backend="neuron")
            neuron_out = compiled_model(x.to("neuron"))
        torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=tol, atol=tol)

    @pytest.mark.parametrize(
        "model_cls,input_shape,tol",
        [
            pytest.param(ContiguousReshapeMatmul, (4, 8), 1e-4, id="contiguous_reshape_matmul"),
            pytest.param(MultipleContiguous, (4, 4), 1e-4, id="multiple_contiguous"),
            pytest.param(
                AtenContiguousReshapeMatmul, (4, 8), 1e-4, id="aten_contiguous_reshape_matmul"
            ),
            pytest.param(MixedContiguousStyles, (4, 8), 1e-4, id="mixed_contiguous_styles"),
        ],
    )
    def test_backward(self, model_cls, input_shape, tol):
        """Test backward pass computes correct gradients with contiguous operations."""
        torch._dynamo.reset()
        cpu_model = model_cls()
        cpu_x = torch.randn(*input_shape, requires_grad=True)
        cpu_out = cpu_model(cpu_x)
        cpu_out.sum().backward()
        expected_grad_x = cpu_x.grad.clone()

        cpu_model.zero_grad()
        neuron_x = cpu_x.detach().clone().to("neuron").requires_grad_(True)
        compiled_model = torch.compile(cpu_model.to("neuron"), backend="neuron")
        neuron_out = compiled_model(neuron_x)
        neuron_out.sum().backward()

        torch.testing.assert_close(neuron_x.grad.cpu(), expected_grad_x, rtol=tol, atol=tol)
