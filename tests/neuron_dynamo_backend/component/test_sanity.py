"""
Trivial 1 core sanity tests for torch.compile functionality on neuron backend.
"""

import os

import pytest
import torch
import torch.nn as nn

# Check if neuron device is available
assert torch.neuron.device_count() > 0, "No neuron devices were discovered."


class TestTorchCompileBackend:
    """Test trivial torch.compile tests with neuron backend."""

    def test_backend_available(self):
        """Test `neuron` backend is available under dynamo."""
        assert "neuron" in torch._dynamo.list_backends()

    def test_simple_model(self):
        """Test simple model."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        with torch.inference_mode():
            model = SimpleModel().to("neuron")
            compiled_model = torch.compile(model, backend="neuron")

            # Create CPU model with same parameters
            cpu_model = SimpleModel()
            cpu_model.load_state_dict(model.cpu().state_dict())
            cpu_model = cpu_model.to("cpu")
            model = model.to("neuron")  # Move back to neuron

            x = torch.randn(2, 10, device="neuron")
            y = compiled_model(x)
            expected = cpu_model(x.cpu())

        assert y.device.type == "neuron"
        assert y.shape == (2, 5)
        torch.testing.assert_close(y.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_linear_model_w_eager(self):
        """Test linear model against eager"""

        class SimpleLinear(nn.Module):
            """
            Input: [batch, in_features]
            Weight: [in_features, out_features]
            Output: [batch, out_features]
            """

            def __init__(self, in_features, out_features):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features

                self.weight = nn.Parameter(torch.randn(in_features, out_features))
                self.bias = nn.Parameter(torch.randn(out_features))

            def forward(self, x):
                """
                x: [batch, in_features]
                output: [batch, out_features]
                """
                output = torch.matmul(x, self.weight) + self.bias
                return output

        # Model config
        batch_size = 4
        in_features = 8
        out_features = 16

        model = SimpleLinear(in_features, out_features).to("neuron")
        model.eval()

        with torch.inference_mode():
            x = torch.randn(batch_size, in_features).to("neuron")
            eager_output = model(x)
            compiled_model = torch.compile(model, backend="neuron")
            compiled_output = compiled_model(x)

        torch.testing.assert_close(compiled_output, eager_output, rtol=1e-4, atol=1e-4)

    def test_cpu_tensor_handling(self):
        """Test CPU tensors are handled on neuron backend during execution."""

        @torch.compile(backend="neuron")
        def fn(x):
            return x * 2

        x_cpu = torch.randn(10, device="cpu")
        y = fn(x_cpu)

        assert y.device.type == "neuron"
        assert y.shape == (10,)

        expected = x_cpu * 2
        torch.testing.assert_close(y.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_multiple_outputs(self):
        """Test function with multiple outputs"""

        @torch.compile(backend="neuron")
        def fn(x):
            return x + 1, x * 2

        x = torch.randn(5, device="neuron")
        y1, y2 = fn(x)

        assert y1.device.type == "neuron"
        assert y2.device.type == "neuron"
        assert y1.shape == (5,)
        assert y2.shape == (5,)

    def test_where_with_scalar(self):
        """Test torch.where with scalar broadcast"""

        @torch.compile(backend="neuron")
        def fn(x, y, z):
            return torch.where(x, y, z)

        x = torch.tensor([True, False], device="neuron")
        y = torch.tensor([0.0, 0.0], device="neuron")
        z = torch.tensor(1.0, device="neuron")

        result = fn(x, y, z)

        assert result.device.type == "neuron"
        expected = torch.where(x.cpu(), y.cpu(), z.cpu())
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_add_default_dtype(self):
        """Test torch.add with default 64-bit dtypes"""

        @torch.compile(backend="neuron")
        def fn(x, y):
            return torch.add(x, y)

        x = torch.tensor([3, 4], dtype=torch.int64, device="neuron")
        y = torch.tensor([3, 4], dtype=torch.int64, device="neuron")

        result = fn(x, y)

        assert result.device.type == "neuron"
        assert result.dtype == torch.int64
        expected = torch.add(x.cpu(), y.cpu())
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_add_float64(self):
        """Test torch.add with float 64-bit dtypes"""

        @torch.compile(backend="neuron")
        def fn(x, y):
            return torch.add(x, y)

        x = torch.tensor([3, 4], dtype=torch.float64, device="neuron")
        y = torch.tensor([3, 4], dtype=torch.float64, device="neuron")

        result = fn(x, y)

        assert result.device.type == "neuron"
        assert result.dtype == torch.float64
        expected = torch.add(x.cpu(), y.cpu())
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)
