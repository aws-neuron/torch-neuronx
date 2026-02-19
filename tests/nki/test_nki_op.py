"""Tests for NKI op registration."""

import neuronxcc.nki.typing as nt
import pytest
import torch
import torch.fx
from neuronxcc import nki

from torch_neuronx import nki_op, wrap_nki


class TestNKIJitDecorator:
    """Test @nki.jit decorator functionality."""

    def test_simple_add_kernel(self):
        """Test a simple addition kernel with @nki.jit."""

        @nki.jit
        def add_kernel(x1, x2, y: nt.mutable_tensor):
            import neuronxcc.nki.language as nl

            x1_tile = nl.load(x1[0:128])
            x2_tile = nl.load(x2[0:128])
            y_tile = x1_tile + x2_tile
            nl.store(y[0:128], value=y_tile)
            return y

        @nki_op("test::nki_add", mutates_args={"y"})
        def nki_add(x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            wrap_nki(add_kernel)(x1, x2, y)

        x1 = torch.randn(128, device="neuron")
        x2 = torch.randn(128, device="neuron")
        y = torch.empty(128, device="neuron")

        nki_add(x1, x2, y)

        expected = x1 + x2
        assert torch.allclose(y.cpu(), expected.cpu())

    def test_scale_kernel(self):
        """Test scaling kernel with @nki.jit."""

        @nki.jit
        def scale_kernel(x, y: nt.mutable_tensor, scale):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x[0:128])
            y_tile = x_tile * scale
            nl.store(y[0:128], value=y_tile)
            return y

        @nki_op("test::nki_scale", mutates_args={"y"})
        def nki_scale(x: torch.Tensor, y: torch.Tensor, scale: float) -> torch.Tensor:
            wrap_nki(scale_kernel)(x, y, scale)

        x = torch.randn(128, device="neuron")
        y = torch.empty(128, device="neuron")
        scale = 3.0

        nki_scale(x, y, scale)

        expected = x * scale
        assert torch.allclose(y.cpu(), expected.cpu())


class TestNKIOpRegistration:
    """Test nki_op registration and execution."""

    def test_basic_op_registration(self):
        """Test basic op registration with in-place mutation."""

        @nki.jit
        def add_kernel(x1, x2, y: nt.mutable_tensor):
            import neuronxcc.nki.language as nl

            x1_tile = nl.load(x1[0:128])
            x2_tile = nl.load(x2[0:128])
            y_tile = x1_tile + x2_tile
            nl.store(y[0:128], value=y_tile)
            return y

        @nki_op("test::nki_add", mutates_args={"y"})
        def nki_add(x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            wrap_nki(add_kernel)(x1, x2, y)

        x1 = torch.randn(128, device="neuron")
        x2 = torch.randn(128, device="neuron")
        y = torch.empty(128, device="neuron")

        nki_add(x1, x2, y)

        expected = x1 + x2
        assert torch.allclose(y.cpu(), expected.cpu())

    def test_op_appears_in_torch_ops(self):
        """Test that registered op appears in torch.ops namespace."""

        @nki_op("test::registry_test", mutates_args={})
        def registry_test(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        # Check op is registered in torch.ops
        assert hasattr(torch.ops.test, "registry_test")

        x = torch.randn(10, device="neuron")
        result = torch.ops.test.registry_test.default(x)
        expected = x * 2

        assert torch.allclose(result.cpu(), expected.cpu())

    def test_different_dtypes(self):
        """Test NKI ops with different data types."""

        @nki.jit
        def copy_kernel(x, y: nt.mutable_tensor):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x)
            nl.store(y, value=x_tile)
            return y

        @nki_op("test::dtype_test", mutates_args={})
        def dtype_test(x: torch.Tensor) -> torch.Tensor:
            y = torch.empty(64, device="neuron", dtype=dtype)
            wrap_nki(copy_kernel)(x, y)
            return y

        dtypes = [torch.float32, torch.float16]

        for dtype in dtypes:
            x = torch.randn(64, device="neuron", dtype=dtype)

            result = dtype_test(x)
            assert torch.equal(result.cpu(), x.cpu())

    def test_multiple_ops_in_graph(self):
        """Test multiple NKI ops in the same graph."""

        @nki.jit
        def add_kernel(x1, x2, y: nt.mutable_tensor):
            import neuronxcc.nki.language as nl

            x1_tile = nl.load(x1[0:128])
            x2_tile = nl.load(x2[0:128])
            y_tile = x1_tile + x2_tile
            nl.store(y[0:128], value=y_tile)
            return y

        @nki.jit
        def mul_kernel(x, y: nt.mutable_tensor, scale):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x[0:128])
            y_tile = x_tile * scale
            nl.store(y[0:128], value=y_tile)
            return y

        @nki_op("test::multi_add", mutates_args={})
        def multi_add(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
            y = torch.empty_like(x1)
            wrap_nki(add_kernel)(x1, x2, y)
            return y

        @nki_op("test::multi_mul", mutates_args={})
        def multi_mul(x: torch.Tensor, scale: float) -> torch.Tensor:
            y = torch.empty_like(x)
            wrap_nki(mul_kernel)(x, y, scale)
            return y

        def combined_fn(x1, x2, scale):
            temp = multi_add(x1, x2)
            return multi_mul(temp, scale)

        x1 = torch.randn(128, device="neuron")
        x2 = torch.randn(128, device="neuron")
        scale = 1.5

        result = combined_fn(x1, x2, scale)
        expected = (x1 + x2) * scale

        assert torch.allclose(result.cpu(), expected.cpu())

    def test_nki_op_forward_backward(self):
        """Test NKI op with forward and backward pass."""

        @nki.jit
        def add_kernel(x, y, output: nt.mutable_tensor):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x[0:128])
            y_tile = nl.load(y[0:128])
            result = x_tile + y_tile
            nl.store(output[0:128], value=result)
            return output

        @nki_op("test::add_fwd", mutates_args={})
        def add_fwd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            output = torch.empty_like(x)
            wrap_nki(add_kernel)(x, y, output)
            return output

        @add_fwd.register_autograd
        def add_fwd_backward(ctx, grad_output):
            # For z = x + y: dz/dx = 1, dz/dy = 1
            return grad_output.clone(), grad_output.clone()

        # Test forward and backward
        x = torch.randn(128, device="neuron", requires_grad=True)
        y = torch.randn(128, device="neuron", requires_grad=True)

        # Forward pass
        result = add_fwd(x, y)
        expected = x + y
        assert torch.allclose(result, expected)

        # Backward pass
        loss = result.sum()
        loss.backward()

        # Check gradients
        assert x.grad is not None
        assert y.grad is not None
        assert torch.allclose(x.grad, torch.ones_like(x))
        assert torch.allclose(y.grad, torch.ones_like(y))

    def test_nki_op_with_autocast(self):
        """Test NKI op works with autocast enabled."""

        @nki.jit
        def add_kernel(x1, x2, y: nt.mutable_tensor):
            import neuronxcc.nki.language as nl

            x1_tile = nl.load(x1[0:128])
            x2_tile = nl.load(x2[0:128])
            y_tile = x1_tile + x2_tile
            nl.store(y[0:128], value=y_tile)
            return y

        @nki_op("test::autocast_add", mutates_args={"y"})
        def autocast_add(x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            wrap_nki(add_kernel)(x1, x2, y)

        x1 = torch.randn(128, device="neuron")
        x2 = torch.randn(128, device="neuron")
        y = torch.empty(128, device="neuron")

        # Test with autocast enabled
        with torch.autocast(device_type="neuron", dtype=torch.bfloat16):
            autocast_add(x1, x2, y)

        expected = x1 + x2
        assert torch.allclose(y.cpu(), expected.cpu())


class TestTorchCompileIntegration:
    """Test integration with torch.compile and FX graphs."""

    @pytest.mark.xfail(reason="fake tensors not supported well by baremetal")
    def test_compile_with_nki_op(self):
        """Test that NKI ops work with torch.compile."""

        @nki.jit
        def add_one_kernel(x, y: nt.mutable_tensor):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x[0:128])
            y_tile = x_tile + 1.0
            nl.store(y[0:128], value=y_tile)
            return y

        @nki_op("test::add_one", mutates_args={})
        def add_one(x: torch.Tensor) -> torch.Tensor:
            y = torch.empty_like(x)
            wrap_nki(add_one_kernel)(x, y)
            return y

        @torch.compile
        def compiled_fn(x):
            return add_one(x)

        x = torch.randn(128, device="neuron")
        result = compiled_fn(x)
        expected = x + 1.0

        assert torch.allclose(result, expected)

    def test_fx_graph_contains_custom_op(self):
        """Test that FX graph contains the custom NKI op."""

        @nki_op("test::fx_graph_test", mutates_args={})
        def fx_graph_test(x: torch.Tensor) -> torch.Tensor:
            return x + 2.0

        def test_fn(x):
            return fx_graph_test(x)

        # Trace with FX
        traced = torch.fx.symbolic_trace(test_fn)

        # Check that custom op appears in graph
        found_custom_op = False
        for node in traced.graph.nodes:
            if (
                node.op == "call_function"
                and hasattr(node.target, "_name")
                and "test::fx_graph_test" in node.target._name
            ):
                found_custom_op = True
                break

        assert found_custom_op, "Custom NKI op not found in FX graph"
