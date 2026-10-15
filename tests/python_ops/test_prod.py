"""Test that the prod full-reduction is properly registered and correct."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    track_neuron_ops,
)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestProd:
    """Test suite for the aten::prod base (full-reduction) overload."""

    @pytest.fixture
    def device(self):
        return torch.device("neuron")

    @pytest.mark.parametrize("shape", [(8,), (32, 32), (4, 8, 16), (2, 3, 4, 5)])
    def test_prod_basic_shapes(self, device, shape):
        """Full reduction over various shapes matches CPU."""
        x_cpu = torch.rand(shape) + 0.5  # keep values away from 0 for stable products
        with track_neuron_ops():
            neuron_result = torch.prod(x_cpu.to(device))
            assert_op_runs_on_neuron("aten::prod")
        torch.testing.assert_close(neuron_result.cpu(), torch.prod(x_cpu))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_prod_float_dtypes(self, device, dtype):
        """Floating dtypes reduce correctly (fp32 accumulation for low precision)."""
        torch.manual_seed(0)
        x_cpu = (torch.rand(64) + 0.5).to(dtype)
        with track_neuron_ops():
            neuron_result = torch.prod(x_cpu.to(device))
            assert_op_runs_on_neuron("aten::prod")
        assert neuron_result.dtype == dtype
        # Low-precision floats accumulate ordering-dependent rounding over a
        # 64-element product beyond their default tolerance; fp32 stays default.
        tol = {"rtol": 1.6e-2, "atol": 1e-3} if dtype != torch.float32 else {}
        torch.testing.assert_close(neuron_result.cpu(), torch.prod(x_cpu), **tol)

    @pytest.mark.parametrize(
        "dtype",
        [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.bool],
    )
    def test_prod_integer_promotes_to_int64(self, device, dtype):
        """Integer/bool inputs promote the output to int64, matching CPU."""
        if dtype == torch.bool:
            x_cpu = torch.tensor([True, True, False, True], dtype=torch.bool)
        else:
            x_cpu = torch.arange(1, 5, dtype=dtype)
        expected = torch.prod(x_cpu)
        with track_neuron_ops():
            neuron_result = torch.prod(x_cpu.to(device))
            assert_op_runs_on_neuron("aten::prod")
        assert neuron_result.dtype == expected.dtype == torch.int64
        torch.testing.assert_close(neuron_result.cpu(), expected)

    def test_prod_dtype_kwarg(self, device):
        """The dtype= kwarg overrides accumulation and output dtype."""
        x_cpu = (torch.rand(16, 16) + 0.5).to(torch.float16)
        with track_neuron_ops():
            neuron_result = torch.prod(x_cpu.to(device), dtype=torch.float32)
            assert_op_runs_on_neuron("aten::prod")
        assert neuron_result.dtype == torch.float32
        torch.testing.assert_close(neuron_result.cpu(), torch.prod(x_cpu, dtype=torch.float32))

    @pytest.mark.parametrize("shape", [(0,), (4, 0, 8)])
    def test_prod_empty_is_one(self, device, shape):
        """Product of an empty tensor is the identity value 1."""
        x_cpu = torch.rand(shape)
        with track_neuron_ops():
            neuron_result = torch.prod(x_cpu.to(device))
        torch.testing.assert_close(neuron_result.cpu(), torch.prod(x_cpu))
        assert neuron_result.item() == 1.0

    def test_prod_single_element(self, device):
        """Single-element and scalar (0-d) inputs reduce to the element itself."""
        x_cpu = torch.tensor([3.5])
        s_cpu = torch.tensor(2.5)
        with track_neuron_ops():
            neuron_vec = torch.prod(x_cpu.to(device))
            neuron_scalar = torch.prod(s_cpu.to(device))
            assert_op_runs_on_neuron("aten::prod")
        torch.testing.assert_close(neuron_vec.cpu(), torch.prod(x_cpu))
        torch.testing.assert_close(neuron_scalar.cpu(), torch.prod(s_cpu))

    @pytest.mark.parametrize(
        "values",
        [
            [1.0, 2.0, 0.0, 4.0],
            [-1.0, 2.0, -3.0],
            [float("inf"), 2.0],
            [float("nan"), 2.0],
        ],
    )
    def test_prod_special_values(self, device, values):
        """inf/nan/zero/negative values behave like CPU."""
        x_cpu = torch.tensor(values, dtype=torch.float32)
        with track_neuron_ops():
            neuron_result = torch.prod(x_cpu.to(device))
            assert_op_runs_on_neuron("aten::prod")
        torch.testing.assert_close(neuron_result.cpu(), torch.prod(x_cpu), equal_nan=True)

    def test_prod_large(self, device):
        """A large reduction stays close to CPU (values near 1 to avoid over/underflow)."""
        x_cpu = 1.0 + (torch.rand(1024, 1024) - 0.5) * 1e-3
        with track_neuron_ops():
            neuron_result = torch.prod(x_cpu.to(device))
            assert_op_runs_on_neuron("aten::prod")
        # Device tree-reduction vs CPU sequential ordering diverges beyond the
        # fp32 default rtol over a 2^20-element product.
        torch.testing.assert_close(neuron_result.cpu(), torch.prod(x_cpu), rtol=5e-4, atol=0.0)

    def test_prod_backward(self, device):
        """Backward keeps the autograd chain and matches CPU gradients."""
        x_cpu = (torch.rand(5, 4) + 0.5).requires_grad_(True)
        x_neuron = x_cpu.detach().clone().to(device).requires_grad_(True)

        out_cpu = torch.prod(x_cpu)
        with track_neuron_ops():
            out_neuron = torch.prod(x_neuron)
            assert_op_runs_on_neuron("aten::prod")

        assert out_neuron.grad_fn is not None, "prod broke the autograd chain"
        torch.testing.assert_close(out_neuron.cpu(), out_cpu)

        out_cpu.backward()
        out_neuron.backward()
        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad)

    def test_prod_method(self, device):
        """The x.prod() tensor method dispatches to the base overload."""
        x_cpu = torch.rand(16, 16) + 0.5
        with track_neuron_ops():
            neuron_result = x_cpu.to(device).prod()
            assert_op_runs_on_neuron("aten::prod")
        torch.testing.assert_close(neuron_result.cpu(), x_cpu.prod())

    @pytest.mark.parametrize("dim", [0, -1])
    @pytest.mark.parametrize("keepdim", [False, True])
    def test_prod_dim(self, device, dim, keepdim):
        """prod.dim_int over a chosen dim, with and without keepdim."""
        x_cpu = torch.rand(6, 4) + 0.5
        expected = torch.prod(x_cpu, dim=dim, keepdim=keepdim)
        with track_neuron_ops():
            neuron_result = torch.prod(x_cpu.to(device), dim=dim, keepdim=keepdim)
            assert_op_runs_on_neuron("aten::prod")
        assert neuron_result.shape == expected.shape
        torch.testing.assert_close(neuron_result.cpu(), expected)

    def test_prod_dim_dtype_kwarg(self, device):
        """dtype= on the dim variant overrides accumulation and output dtype."""
        x_cpu = (torch.rand(4, 5) + 0.5).to(torch.bfloat16)
        expected = torch.prod(x_cpu, dim=1, dtype=torch.float32)
        with track_neuron_ops():
            neuron_result = torch.prod(x_cpu.to(device), dim=1, dtype=torch.float32)
            assert_op_runs_on_neuron("aten::prod")
        assert neuron_result.dtype == torch.float32
        torch.testing.assert_close(neuron_result.cpu(), expected)

    def test_prod_out_variant(self, device):
        """prod.out writes the full-reduction result into the pre-allocated tensor."""
        x_cpu = torch.rand(3, 4) + 0.5
        expected = torch.prod(x_cpu)
        out = torch.empty((), device=device)
        # torch.prod's Python signature only accepts out= on the dim overloads;
        # the full-reduce out variant is the ATen overload aten::prod.out.
        with track_neuron_ops():
            result = torch.ops.aten.prod.out(x_cpu.to(device), out=out)
            assert_op_runs_on_neuron("aten::prod")
        assert result is out
        torch.testing.assert_close(out.cpu(), expected)

    @pytest.mark.parametrize("keepdim", [False, True])
    def test_prod_int_out_variant(self, device, keepdim):
        """prod.int_out writes the dim-reduced result into the out tensor."""
        x_cpu = torch.rand(3, 4) + 0.5
        expected = torch.prod(x_cpu, dim=1, keepdim=keepdim)
        out = torch.empty(expected.shape, device=device)
        with track_neuron_ops():
            result = torch.prod(x_cpu.to(device), dim=1, keepdim=keepdim, out=out)
            assert_op_runs_on_neuron("aten::prod")
        assert result is out
        assert out.shape == expected.shape
        torch.testing.assert_close(out.cpu(), expected)

    def test_prod_dim_backward(self, device):
        """Dim-reduce backward keeps the autograd chain and matches CPU."""
        x_cpu = (torch.rand(4, 5) + 0.5).requires_grad_(True)
        x_neuron = x_cpu.detach().clone().to(device).requires_grad_(True)

        out_cpu = torch.prod(x_cpu, dim=1)
        with track_neuron_ops():
            out_neuron = torch.prod(x_neuron, dim=1)
        assert out_neuron.grad_fn is not None, "prod.dim_int broke the autograd chain"

        grad = torch.rand_like(out_cpu)
        out_cpu.backward(grad)
        out_neuron.backward(grad.to(device))
        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad)
