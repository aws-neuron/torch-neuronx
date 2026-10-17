"""E2E test for aten.clone lowering through torch.compile(backend='neuron')."""

import pytest
import torch


class TestCloneE2E:
    """Verify clone produces a correct copy through the neuron backend."""

    @pytest.mark.parametrize("shape", [(4, 8), (2, 3, 16), (1,)])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_clone(self, shape, dtype):
        torch.manual_seed(42)
        x = torch.randn(shape, dtype=dtype)

        def fn(a):
            return a.clone()

        with torch.no_grad():
            cpu_out = fn(x)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out)

    def test_clone_preserves_values_after_mutation(self):
        """Clone must produce an independent copy — mutating original doesn't affect clone."""
        torch.manual_seed(42)
        x = torch.randn(4, 8)

        def fn(a):
            b = a.clone()
            return b

        with torch.no_grad():
            cpu_out = fn(x)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out)
