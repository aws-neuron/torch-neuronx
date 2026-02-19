"""
E2E tests for torch.float8_e5m2 dtype operations.
"""

import pytest
import torch
import torch.nn as nn

# Check if neuron device is available
assert torch.neuron.device_count() > 0, "No neuron devices were discovered."


class Float8Concat(nn.Module):
    """Test concat with float8_e5m2."""

    def forward(self, x, y):
        return torch.cat([x, y])


class Float8Mul(nn.Module):
    """Test basic multiplication with float8_e5m2."""

    def forward(self, x, y):
        return x * y


class Float8MatMul(nn.Module):
    """Test matrix multiplication with float8_e5m2."""

    def forward(self, x, y):
        return torch.matmul(x, y)


class Float8TypeConversion(nn.Module):
    """Test type conversion to/from float8_e5m2."""

    def forward(self, x):
        # Convert to float32 and back to fp8
        x_f32 = x.to(torch.float32)
        return x_f32.to(torch.float8_e5m2)


class Float8Reshape(nn.Module):
    """Test reshape operations with float8_e5m2."""

    def forward(self, x):
        return x.reshape(-1, 4)


class TestFloat8E5M2:
    """E2E tests for torch.float8_e5m2 dtype operations."""

    @pytest.mark.parametrize(
        "model_cls,input_shapes,tol",
        [
            pytest.param(Float8Concat, [(4, 8), (4, 8)], 1e-2, id="float8_cat"),
            pytest.param(Float8Mul, [(4, 8), (4, 8)], 1e-2, id="float8_mul"),
            pytest.param(Float8MatMul, [(4, 8), (8, 4)], 1e-2, id="float8_matmul"),
            pytest.param(Float8TypeConversion, [(4, 8)], 1e-2, id="float8_type_conversion"),
            pytest.param(Float8Reshape, [(2, 8)], 1e-2, id="float8_reshape"),
        ],
    )
    def test_forward(self, model_cls, input_shapes, tol):
        """Test basic ops with float8_e5m2 dtype."""
        model = model_cls()

        # Create inputs in float32, convert to float8_e5m2
        inputs = [torch.randn(shape).to(torch.float8_e5m2) for shape in input_shapes]

        with torch.no_grad():
            # CPU reference (convert back to float32 for computation)
            cpu_inputs = list(inputs)
            cpu_out = model(*cpu_inputs)

        with torch.no_grad():
            # Neuron execution
            compiled_model = torch.compile(model.to("neuron"), backend="neuron")
            neuron_inputs = [x.to("neuron") for x in inputs]
            neuron_out = compiled_model(*neuron_inputs)

        torch.testing.assert_close(
            neuron_out.cpu().to(torch.float32), cpu_out.to(torch.float32), rtol=tol, atol=tol
        )
