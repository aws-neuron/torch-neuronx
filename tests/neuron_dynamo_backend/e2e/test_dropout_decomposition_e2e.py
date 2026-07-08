"""
E2E tests for native_dropout decomposition through torch.compile.

Verifies that models with dropout layers compile and produce correct
output shapes and statistical properties when using the neuron backend.
"""

import pytest
import torch
import torch.nn as nn

assert torch.neuron.device_count() > 0, "No neuron devices were discovered."


class PureDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        return self.dropout(x)


class PureDualDropout(nn.Module):
    def __init__(self, p1=0.3, p2=0.7):
        super().__init__()
        self.dropout1 = nn.Dropout(p=p1)
        self.dropout2 = nn.Dropout(p=p2)

    def forward(self, x):
        return self.dropout2(self.dropout1(x))


class SingleDropoutModel(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.linear = nn.Linear(16, 16)
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        return self.dropout(self.linear(x))


class MultiDropoutModel(nn.Module):
    def __init__(self, p1=0.3, p2=0.7):
        super().__init__()
        self.linear1 = nn.Linear(16, 16)
        self.dropout1 = nn.Dropout(p=p1)
        self.linear2 = nn.Linear(16, 16)
        self.dropout2 = nn.Dropout(p=p2)

    def forward(self, x):
        x = self.dropout1(self.linear1(x))
        return self.dropout2(self.linear2(x))


class TestDropoutDecompositionE2E:
    """E2E tests for dropout decomposition via torch.compile on neuron."""

    @pytest.mark.parametrize(
        "p",
        [
            pytest.param(
                0.0,
                id="p0.0",
                marks=pytest.mark.xfail(
                    reason="pre-existing PT 2.11->2.12 bump failure; tracked for fix"
                ),
            ),
            pytest.param(0.3, id="p0.3"),
            pytest.param(0.5, id="p0.5"),
            pytest.param(0.7, id="p0.7"),
            pytest.param(1.0, id="p1.0"),
        ],
    )
    def test_single_dropout_zero_rate(self, p):
        """Pure dropout produces zero fraction matching p."""
        torch._dynamo.reset()
        model = PureDropout(p=p).to("neuron").train()
        compiled = torch.compile(model, backend="neuron")
        x = torch.ones(256, 16, device="neuron")
        out = compiled(x)
        assert out.shape == (256, 16)
        zero_frac = (out == 0).float().mean().item()
        if p == 0.0:
            assert zero_frac == 0.0
        elif p == 1.0:
            assert zero_frac == 1.0
        else:
            assert abs(zero_frac - p) < 0.1, f"Expected ~{p} zeros, got {zero_frac}"

    @pytest.mark.parametrize(
        "p1,p2",
        [
            pytest.param(0.3, 0.7, id="p0.3_p0.7"),
            pytest.param(0.1, 0.9, id="p0.1_p0.9"),
            pytest.param(0.5, 0.5, id="p0.5_p0.5"),
        ],
    )
    def test_dual_dropout_compounded_zero_rate(self, p1, p2):
        """Two sequential dropouts produce compounded zero fraction: 1 - (1-p1)*(1-p2)."""
        torch._dynamo.reset()
        model = PureDualDropout(p1=p1, p2=p2).to("neuron").train()
        compiled = torch.compile(model, backend="neuron")
        x = torch.ones(256, 16, device="neuron")
        out = compiled(x)
        assert out.shape == (256, 16)
        zero_frac = (out == 0).float().mean().item()
        expected = 1.0 - (1.0 - p1) * (1.0 - p2)
        assert abs(zero_frac - expected) < 0.1, f"Expected ~{expected} zeros, got {zero_frac}"

    @pytest.mark.parametrize(
        "p",
        [
            pytest.param(0.0, id="p0.0"),
            pytest.param(0.5, id="p0.5"),
        ],
    )
    def test_dropout_eval_deterministic(self, p):
        """Dropout is identity in eval mode; output should match across calls."""
        torch._dynamo.reset()
        model = SingleDropoutModel(p=p).to("neuron").eval()
        compiled = torch.compile(model, backend="neuron")
        x = torch.randn(4, 16, device="neuron")
        with torch.no_grad():
            out1 = compiled(x)
            out2 = compiled(x)
        torch.testing.assert_close(out1, out2)

    def test_dropout_mask_stochasticity(self):
        """Two forward passes in train mode should produce different outputs."""
        torch._dynamo.reset()
        model = PureDropout(p=0.5).to("neuron").train()
        compiled = torch.compile(model, backend="neuron")
        x = torch.ones(32, 16, device="neuron")
        out1 = compiled(x)
        out2 = compiled(x)
        assert not torch.equal(out1, out2), "Dropout should produce different masks across calls"

    @pytest.mark.parametrize(
        "p",
        [
            pytest.param(0.3, id="p0.3"),
            pytest.param(0.5, id="p0.5"),
            pytest.param(0.7, id="p0.7"),
        ],
    )
    def test_single_dropout_backward(self, p):
        """Backward pass through compiled dropout produces gradients."""
        torch._dynamo.reset()
        model = SingleDropoutModel(p=p).to("neuron").train()
        compiled = torch.compile(model, backend="neuron")
        x = torch.randn(8, 16, device="neuron", requires_grad=True)
        out = compiled(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == (8, 16)

    @pytest.mark.parametrize(
        "p1,p2",
        [
            pytest.param(0.3, 0.7, id="p0.3_p0.7"),
            pytest.param(0.5, 0.5, id="p0.5_p0.5"),
        ],
    )
    def test_multi_dropout_backward(self, p1, p2):
        """Backward pass through model with multiple dropouts produces gradients."""
        torch._dynamo.reset()
        model = MultiDropoutModel(p1=p1, p2=p2).to("neuron").train()
        compiled = torch.compile(model, backend="neuron")
        x = torch.randn(8, 16, device="neuron", requires_grad=True)
        out = compiled(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == (8, 16)
        for param in model.parameters():
            assert param.grad is not None
