"""E2E tests for the aten::expm1 torch.compile decomposition.

torch-mlir lowers expm1 through the hardware exp (~1e-05 relative error), which
misses the fp32 tolerance. A Neuron decomposition routes the compile path to the
same range-reduced math the eager op uses. These tests assert the routing and
the numerics.
"""

import pytest
import torch
from torch._C import DispatchKey

import torch_neuronx

# (shape, dtype, tol)
CONFIGS = [
    pytest.param((64,), torch.float32, 1e-5, id="1d-f32"),
    pytest.param((8, 16), torch.float32, 1e-5, id="2d-f32"),
    pytest.param((2, 3, 8), torch.float32, 1e-5, id="3d-f32"),
    pytest.param((64,), torch.float16, 5e-3, id="1d-f16"),
    pytest.param((64,), torch.bfloat16, 5e-2, id="1d-bf16"),
    pytest.param((4, 6), torch.int32, 1e-5, id="2d-i32"),
]


class TestExpm1DecompositionE2E:
    """Verify the expm1 decomposition routes correctly and matches CPU."""

    def test_expm1_decomposition_routing(self):
        """Test no Autograd/CompositeImplicitAutograd shadows the decomposition.

        A decomposition table is only read below __torch_dispatch__. If PyTorch
        published expm1 as py_impl(Autograd) or py_impl(CompositeImplicitAutograd),
        the Neuron decomposition would be dead code (gotcha #5).
        """
        import torch_neuronx.neuron_dynamo_backend.decompositions as d

        assert torch.ops.aten.expm1.default in d.neuron_decompositions

        op = torch.ops.aten.expm1.default
        assert DispatchKey.Autograd not in op.py_kernels
        assert DispatchKey.CompositeImplicitAutograd not in op.py_kernels

    @pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
    @pytest.mark.parametrize("shape,dtype,tol", CONFIGS)
    def test_expm1_compile_matches_cpu(self, shape, dtype, tol):
        """Test the compiled expm1 matches CPU across dtypes and shapes."""
        torch.manual_seed(0)
        if dtype == torch.int32:
            x = torch.randint(-4, 5, shape, dtype=dtype)
        else:
            x = (torch.rand(shape, dtype=dtype) * 4 - 2).to(dtype)

        def fn(t):
            return torch.expm1(t)

        torch._dynamo.reset()
        with torch.no_grad():
            cpu_out = fn(x)
            compiled_fn = torch.compile(fn, backend="neuron")
            neuron_out = compiled_fn(x.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=tol, atol=tol)
