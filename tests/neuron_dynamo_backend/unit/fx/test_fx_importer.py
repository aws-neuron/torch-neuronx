"""
Unit test for FX importer handling of neuron device tensor constants.
"""

import torch
import torch.nn as nn


class TestFxImporterTensorConstants:
    def test_neuron_device_scalar_constant(self):
        """Scalar constant created on neuron device via torch.tensor(0.0, device=x.device)."""

        class Model(nn.Module):
            def forward(self, x):
                mask = x > 0
                min_dtype = torch.tensor(-3.389e38, device=x.device, dtype=x.dtype)
                result = torch.where(
                    mask, torch.tensor(0.0, device=mask.device, dtype=x.dtype), min_dtype
                )
                return result

        model = Model().to(torch.bfloat16).to("neuron")
        x = torch.randn(1, 512, dtype=torch.bfloat16, device="neuron")

        compiled = torch.compile(model, fullgraph=True, backend="neuron")
        out = compiled(x)
        assert out.shape == (1, 512)
