"""
E2E tests for native_batch_norm decomposition through torch.compile.

Verifies BatchNorm2d models compile and match CPU output on the neuron
backend across uniform and mixed param/buffer dtype configurations.
"""

import pytest
import torch
import torch.nn as nn

assert torch.neuron.device_count() > 0, "No neuron devices were discovered."


# (input_shape, num_features, param_dtype, buffer_dtype, tol)
# Input tensor dtype matches param_dtype
BN_CONFIGS = [
    pytest.param(
        (2, 8, 16, 16),
        8,
        torch.float32,
        torch.float32,
        1e-4,
        id="bn2d-all-f32",
    ),
    pytest.param(
        (2, 8, 16, 16),
        8,
        torch.bfloat16,
        torch.bfloat16,
        2e-2,
        id="bn2d-all-bf16",
    ),
    pytest.param(
        (2, 8, 16, 16),
        8,
        torch.bfloat16,
        torch.float32,
        2e-2,
        id="bn2d-bf16-stats-f32",
    ),
    pytest.param(
        (2, 64, 56, 56),
        64,
        torch.bfloat16,
        torch.float32,
        2e-2,
        id="bn2d-resnet-bf16-stats-f32",
    ),
]


class SingleBNModel(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        return self.bn(x)


def _make_model(num_features, param_dtype, buffer_dtype, training):
    """BatchNorm2d with params in ``param_dtype`` and running stats in ``buffer_dtype``."""
    torch.manual_seed(0)
    model = SingleBNModel(num_features).to(param_dtype)
    if buffer_dtype != param_dtype:
        # Update buffers in place to preserve their registered identity.
        model.bn.running_mean.data = model.bn.running_mean.data.to(buffer_dtype)
        model.bn.running_var.data = model.bn.running_var.data.to(buffer_dtype)
    return model.train(training)


def _cpu_reference_out(num_features, param_dtype, buffer_dtype, x, training):
    """CPU reference output. PyTorch rejects mixed (param, buffer) dtypes, so fall back to fp32."""
    if param_dtype == buffer_dtype:
        model = _make_model(num_features, param_dtype, buffer_dtype, training)
        return model(x)
    model = _make_model(num_features, torch.float32, torch.float32, training)
    return model(x.to(torch.float32))


class TestBatchNormDecompositionE2E:
    @pytest.mark.parametrize("shape,num_features,param_dtype,buffer_dtype,tol", BN_CONFIGS)
    def test_batch_norm_train(self, shape, num_features, param_dtype, buffer_dtype, tol):
        """Training mode hits aten._native_batch_norm_legit_functional."""
        torch.manual_seed(0)
        x = torch.randn(shape, dtype=param_dtype)

        cpu_out = _cpu_reference_out(num_features, param_dtype, buffer_dtype, x, training=True)

        neuron_model = _make_model(num_features, param_dtype, buffer_dtype, training=True).to(
            "neuron"
        )
        compiled_model = torch.compile(neuron_model, backend="neuron", dynamic=False)
        neuron_out = compiled_model(x.to("neuron"))

        # Compare in fp32 since the mixed-dtype reference is fp32.
        torch.testing.assert_close(neuron_out.cpu().float(), cpu_out.float(), rtol=tol, atol=tol)

    @pytest.mark.parametrize("shape,num_features,param_dtype,buffer_dtype,tol", BN_CONFIGS)
    def test_batch_norm_eval(self, shape, num_features, param_dtype, buffer_dtype, tol):
        """Eval mode hits aten._native_batch_norm_legit_no_training."""
        torch.manual_seed(0)
        x = torch.randn(shape, dtype=param_dtype)

        with torch.no_grad():
            cpu_out = _cpu_reference_out(num_features, param_dtype, buffer_dtype, x, training=False)

            neuron_model = _make_model(num_features, param_dtype, buffer_dtype, training=False).to(
                "neuron"
            )
            compiled_model = torch.compile(neuron_model, backend="neuron", dynamic=False)
            neuron_out = compiled_model(x.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu().float(), cpu_out.float(), rtol=tol, atol=tol)
