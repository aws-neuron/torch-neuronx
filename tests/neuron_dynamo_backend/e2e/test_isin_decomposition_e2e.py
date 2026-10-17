"""E2E tests for aten::isin under torch.compile(backend="neuron").

Eager isin is lowered by ``@register_aten`` (see python_ops test_isin.py). Under
torch.compile the op is lowered by PyTorch's built-in ``aten.isin`` decomposition,
opted into in ``neuron_dynamo_backend/decompositions.py``. ``fullgraph=True`` is
the routing assertion: isin has no StableHLO membership primitive, so without the
decomposition compilation would graph-break and raise.
"""

import pytest
import torch

# (elements_shape, test_shape, dtype)
ISIN_CONFIGS = [
    pytest.param((2, 3), (3,), torch.int64, id="2x3-int64"),
    pytest.param((2, 3), (3,), torch.int32, id="2x3-int32"),
    pytest.param((2, 3), (3,), torch.float32, id="2x3-float32"),
    pytest.param((8,), (4,), torch.int64, id="1d-int64"),
    pytest.param((4, 5), (10,), torch.float32, id="4x5-float32"),
]


class TestIsinDecompositionE2E:
    @pytest.mark.parametrize("elements_shape,test_shape,dtype", ISIN_CONFIGS)
    @pytest.mark.parametrize("invert", [False, True])
    def test_forward(self, elements_shape, test_shape, dtype, invert):
        torch.manual_seed(0)
        if dtype.is_floating_point:
            elements = torch.randint(0, 8, elements_shape).to(dtype)
            test_elements = torch.randint(0, 8, test_shape).to(dtype)
        else:
            elements = torch.randint(0, 8, elements_shape, dtype=dtype)
            test_elements = torch.randint(0, 8, test_shape, dtype=dtype)

        def fn(x, t):
            return torch.isin(x, t, invert=invert)

        with torch.no_grad():
            cpu_out = fn(elements, test_elements)
            compiled = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled(elements.to("neuron"), test_elements.to("neuron"))

        assert neuron_out.dtype == torch.bool
        torch.testing.assert_close(neuron_out.cpu(), cpu_out)

    @pytest.mark.parametrize("invert", [False, True])
    def test_forward_empty_test_elements(self, invert):
        """Empty test_elements under compile: all-`invert`, via the decomposition."""
        torch.manual_seed(0)
        elements = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
        empty = torch.tensor([], dtype=torch.int64)

        def fn(x, t):
            return torch.isin(x, t, invert=invert)

        with torch.no_grad():
            cpu_out = fn(elements, empty)
            compiled = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled(elements.to("neuron"), empty.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out)
