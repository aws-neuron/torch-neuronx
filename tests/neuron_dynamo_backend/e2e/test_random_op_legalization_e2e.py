"""
End-to-end tests for random_op_legalization FX pass.

Tests the complete compilation pipeline with torch.compile and the neuron backend
for models containing dropout operations, with accuracy comparison against reference.
"""

from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

# =============================================================================
# Model Definitions
# =============================================================================


class LinearDropoutModel(torch.nn.Module):
    """Linear layer followed by dropout with 2D input."""

    input_dim = 32
    output_dim = 16

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=True)
        return x

    @staticmethod
    def get_inputs():
        return (torch.randn(4, LinearDropoutModel.input_dim),)


class AttentionWithDropout(torch.nn.Module):
    """Scaled dot-product attention with dropout."""

    batch_size = 2
    seq_len = 8
    num_heads = 4
    head_dim = 16

    def __init__(self, dropout_p=0.1):
        super().__init__()
        self.dropout_p = dropout_p
        self.scale = self.head_dim**-0.5

    def forward(self, query, key, value):
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)
        output = torch.matmul(attn_weights, value)
        return output

    @staticmethod
    def get_inputs():
        shape = (
            AttentionWithDropout.batch_size,
            AttentionWithDropout.num_heads,
            AttentionWithDropout.seq_len,
            AttentionWithDropout.head_dim,
        )
        return (torch.randn(*shape), torch.randn(*shape), torch.randn(*shape))


class MLPWithDropout(torch.nn.Module):
    """MLP with multiple dropout layers."""

    input_dim = 64
    hidden_dim = 128
    output_dim = 32

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = torch.nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout1 = torch.nn.Dropout(p=0.1)
        self.dropout2 = torch.nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

    @staticmethod
    def get_inputs():
        return (torch.randn(4, MLPWithDropout.input_dim),)


class MLPWithMixedDropout(torch.nn.Module):
    """MLP mixing nn.Dropout and F.dropout."""

    input_dim = 64
    hidden_dim = 128
    output_dim = 32

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc4 = torch.nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout_module1 = torch.nn.Dropout(p=0.1)
        self.dropout_module2 = torch.nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout_module1(x)
        x = self.fc2(x)
        x = F.gelu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc3(x)
        x = F.gelu(x)
        x = self.dropout_module2(x)
        x = self.fc4(x)
        x = F.dropout(x, p=0.15, training=self.training)
        return x

    @staticmethod
    def get_inputs():
        return (torch.randn(4, MLPWithMixedDropout.input_dim),)


# Backward-compatible models (with training=False or deterministic dropout)
class LinearDropoutModelBackward(torch.nn.Module):
    """Linear layer followed by dropout for backward testing."""

    input_dim = 32
    output_dim = 16

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=False)
        return x

    @staticmethod
    def get_inputs():
        return (torch.randn(4, LinearDropoutModelBackward.input_dim),)


class MLPWithDropoutBackward(torch.nn.Module):
    """MLP with dropout for backward testing."""

    input_dim = 32
    hidden_dim = 64
    output_dim = 16

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = F.dropout(x, p=0.32)
        x = self.fc2(x)
        return x

    @staticmethod
    def get_inputs():
        return (torch.randn(4, MLPWithDropoutBackward.input_dim),)


class AttentionWithDropoutBackward(torch.nn.Module):
    """Attention with dropout for backward testing."""

    batch_size = 2
    seq_len = 2
    num_heads = 2
    head_dim = 2

    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.dropout_p = dropout_p
        self.scale = self.head_dim**-0.5

    def forward(self, query, key, value):
        output = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        output = torch.softmax(output, dim=-1)
        output = F.dropout(output, p=self.dropout_p, training=True)
        output = torch.matmul(output, value)
        return output

    @staticmethod
    def get_inputs():
        shape = (
            AttentionWithDropoutBackward.batch_size,
            AttentionWithDropoutBackward.num_heads,
            AttentionWithDropoutBackward.seq_len,
            AttentionWithDropoutBackward.head_dim,
        )
        return (torch.randn(*shape), torch.randn(*shape), torch.randn(*shape))


# =============================================================================
# Test Parameter Sets
# =============================================================================

FORWARD_MODEL_CONFIGS = [
    pytest.param(LinearDropoutModel, id="linear_dropout"),
    pytest.param(AttentionWithDropout, id="attention_dropout"),
    pytest.param(MLPWithDropout, id="mlp_dropout"),
    pytest.param(MLPWithMixedDropout, id="mlp_mixed_dropout"),
]

BACKWARD_MODEL_CONFIGS = [
    pytest.param(LinearDropoutModelBackward, id="linear_dropout"),
    pytest.param(MLPWithDropoutBackward, id="mlp_dropout"),
    pytest.param(AttentionWithDropoutBackward, id="attention_dropout"),
]


# =============================================================================
# Test Classes
# =============================================================================


class TestRandomOpLegalizationE2E:
    """End-to-end forward tests using torch.compile with neuron backend."""

    @pytest.mark.parametrize("model_cls", FORWARD_MODEL_CONFIGS)
    def test_e2e_forward(self, model_cls):
        """Test forward pass compilation with various dropout models."""
        torch.manual_seed(42)
        inputs = model_cls.get_inputs()

        with torch.no_grad():
            model = model_cls()
            torch.manual_seed(42)
            inputs_clone = tuple(inp.clone() for inp in inputs)
            reference_output = model(*inputs_clone)

        with torch.no_grad():
            compiled_model = torch.compile(model, backend="neuron")
            torch.manual_seed(42)
            inputs_clone = tuple(inp.clone() for inp in inputs)
            compiled_output = compiled_model(*inputs_clone)

        assert compiled_output.shape == reference_output.shape
        assert torch.allclose(compiled_output.cpu(), reference_output, rtol=1e-4, atol=1e-4)


class TestRandomOpLegalizationE2EBackward:
    """End-to-end backward tests using torch.compile with neuron backend."""

    @pytest.mark.parametrize("model_cls", BACKWARD_MODEL_CONFIGS)
    def test_e2e_backward(self, model_cls):
        """Test backward pass compilation with various dropout models."""
        torch.manual_seed(42)
        inputs_cpu = tuple(inp.requires_grad_(True) for inp in model_cls.get_inputs())
        inputs_neuron = tuple(
            inp.detach().clone().to("neuron").requires_grad_(True) for inp in inputs_cpu
        )

        # Reference forward + backward
        model_cpu = model_cls()
        torch.manual_seed(42)
        output_cpu = model_cpu(*inputs_cpu)
        grad_cpu = torch.randn(output_cpu.shape, device="cpu")
        output_cpu.backward(grad_cpu)

        # Compiled forward + backward
        compiled_model = torch.compile(model_cpu.to("neuron"), backend="neuron")
        torch.manual_seed(42)
        output_neuron = compiled_model(*inputs_neuron)
        grad_neuron = grad_cpu.detach().clone().to("neuron")
        output_neuron.backward(grad_neuron)

        # Verify forward output
        assert output_neuron.shape == output_cpu.shape
        assert torch.allclose(
            output_neuron.cpu().detach(), output_cpu.detach(), rtol=1e-4, atol=1e-4
        )

        # Verify input gradients
        for inp_neuron, inp_cpu in zip(inputs_neuron, inputs_cpu, strict=False):
            assert inp_neuron.grad is not None
            assert inp_cpu.grad is not None
            assert torch.allclose(inp_neuron.grad.cpu(), inp_cpu.grad, rtol=1e-3, atol=1e-4)
