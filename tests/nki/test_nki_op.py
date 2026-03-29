"""Tests for NKI op registration."""

import nki
import nki.isa as nisa
import nki.language as nl
import pytest
import torch
import torch.fx

from torch_neuronx import nki_op


class TestNKIJitDecorator:
    """Test @nki.jit decorator functionality."""

    def test_simple_add_kernel(self):
        """Test a simple addition kernel with @nki.jit."""

        @nki.jit
        def add_kernel(x1, x2):
            s1 = nl.ndarray(x1.shape, dtype=x1.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=s1, src=x1)

            s2 = nl.ndarray(x2.shape, dtype=x2.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=s2, src=x2)

            s3 = nl.ndarray(x1.shape, dtype=x1.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=s3, data1=s1, data2=s2, op=nl.add)

            out = nl.ndarray(x1.shape, dtype=x1.dtype, buffer=nl.shared_hbm)
            nisa.dma_copy(dst=out, src=s3)
            return out

        @nki_op("test::nki_add")
        def nki_add(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
            return add_kernel(x1, x2)

        x1 = torch.randn(1, 128, device="neuron")
        x2 = torch.randn(1, 128, device="neuron")

        result = nki_add(x1, x2)

        expected = x1 + x2
        assert torch.allclose(result.cpu(), expected.cpu())

    def test_scale_kernel(self):
        """Test scaling kernel with @nki.jit."""

        @nki.jit
        def scale_kernel(x, scale):
            sx = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=sx, src=x)

            sy = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=sy, data=sx, op0=nl.multiply, operand0=scale)

            out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
            nisa.dma_copy(dst=out, src=sy)
            return out

        @nki_op("test::nki_scale")
        def nki_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
            return scale_kernel(x, scale)

        x = torch.randn(1, 128, device="neuron")
        scale = 3.0

        result = nki_scale(x, scale)

        expected = x * scale
        assert torch.allclose(result.cpu(), expected.cpu())


class TestNKIOpRegistration:
    """Test nki_op registration and execution."""

    def test_basic_op_registration(self):
        """Test basic op registration."""

        @nki.jit
        def add_kernel(x1, x2):
            s1 = nl.ndarray(x1.shape, dtype=x1.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=s1, src=x1)

            s2 = nl.ndarray(x2.shape, dtype=x2.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=s2, src=x2)

            s3 = nl.ndarray(x1.shape, dtype=x1.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=s3, data1=s1, data2=s2, op=nl.add)

            out = nl.ndarray(x1.shape, dtype=x1.dtype, buffer=nl.shared_hbm)
            nisa.dma_copy(dst=out, src=s3)
            return out

        @nki_op("test::nki_add_reg")
        def nki_add(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
            return add_kernel(x1, x2)

        x1 = torch.randn(1, 128, device="neuron")
        x2 = torch.randn(1, 128, device="neuron")

        result = nki_add(x1, x2)

        expected = x1 + x2
        assert torch.allclose(result.cpu(), expected.cpu())

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
        """Test NKI ops with float32 dtype."""

        @nki.jit
        def copy_kernel(x):
            sx = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=sx, src=x)

            out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
            nisa.dma_copy(dst=out, src=sx)
            return out

        @nki_op("test::dtype_test_v3")
        def dtype_test(x: torch.Tensor) -> torch.Tensor:
            return copy_kernel(x)

        x = torch.randn(1, 64, device="neuron", dtype=torch.float32)
        result = dtype_test(x)
        assert torch.equal(result.cpu(), x.cpu())

    def test_multiple_ops_in_graph(self):
        """Test multiple NKI ops in the same graph."""

        @nki.jit
        def add_kernel(x1, x2):
            s1 = nl.ndarray(x1.shape, dtype=x1.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=s1, src=x1)

            s2 = nl.ndarray(x2.shape, dtype=x2.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=s2, src=x2)

            s3 = nl.ndarray(x1.shape, dtype=x1.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=s3, data1=s1, data2=s2, op=nl.add)

            out = nl.ndarray(x1.shape, dtype=x1.dtype, buffer=nl.shared_hbm)
            nisa.dma_copy(dst=out, src=s3)
            return out

        @nki.jit
        def mul_kernel(x, scale):
            sx = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=sx, src=x)

            sy = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=sy, data=sx, op0=nl.multiply, operand0=scale)

            out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
            nisa.dma_copy(dst=out, src=sy)
            return out

        @nki_op("test::multi_add_v3")
        def multi_add(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
            return add_kernel(x1, x2)

        @nki_op("test::multi_mul_v3")
        def multi_mul(x: torch.Tensor, scale: float) -> torch.Tensor:
            return mul_kernel(x, scale)

        def combined_fn(x1, x2, scale):
            temp = multi_add(x1, x2)
            return multi_mul(temp, scale)

        x1 = torch.randn(1, 128, device="neuron")
        x2 = torch.randn(1, 128, device="neuron")
        scale = 1.5

        result = combined_fn(x1, x2, scale)
        expected = (x1 + x2) * scale

        assert torch.allclose(result.cpu(), expected.cpu())

    def test_nki_op_forward_backward(self):
        """Test NKI op with forward and backward pass."""

        @nki.jit
        def add_kernel(x, y):
            sx = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=sx, src=x)

            sy = nl.ndarray(y.shape, dtype=y.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=sy, src=y)

            sr = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sr, data1=sx, data2=sy, op=nl.add)

            out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
            nisa.dma_copy(dst=out, src=sr)
            return out

        @nki_op("test::add_fwd_v3")
        def add_fwd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return add_kernel(x, y)

        @add_fwd.register_autograd
        def add_fwd_backward(ctx, grad_output):
            # For z = x + y: dz/dx = 1, dz/dy = 1
            return grad_output.clone(), grad_output.clone()

        # Test forward and backward
        x = torch.randn(1, 128, device="neuron", requires_grad=True)
        y = torch.randn(1, 128, device="neuron", requires_grad=True)

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
        def add_kernel(x1, x2):
            s1 = nl.ndarray(x1.shape, dtype=x1.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=s1, src=x1)

            s2 = nl.ndarray(x2.shape, dtype=x2.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=s2, src=x2)

            s3 = nl.ndarray(x1.shape, dtype=x1.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=s3, data1=s1, data2=s2, op=nl.add)

            out = nl.ndarray(x1.shape, dtype=x1.dtype, buffer=nl.shared_hbm)
            nisa.dma_copy(dst=out, src=s3)
            return out

        @nki_op("test::autocast_add_v3")
        def autocast_add(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
            return add_kernel(x1, x2)

        x1 = torch.randn(1, 128, device="neuron")
        x2 = torch.randn(1, 128, device="neuron")

        # Test with autocast enabled
        with torch.autocast(device_type="neuron", dtype=torch.bfloat16):
            result = autocast_add(x1, x2)

        expected = x1 + x2
        assert torch.allclose(result.cpu(), expected.cpu())


class TestTorchCompileIntegration:
    """Test integration with torch.compile and FX graphs."""

    @pytest.mark.xfail(reason="fake tensors not supported well by baremetal")
    def test_compile_with_nki_op(self):
        """Test that NKI ops work with torch.compile."""

        @nki.jit
        def add_one_kernel(x, y):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x[0:128])
            y_tile = x_tile + 1.0
            nl.store(y[0:128], value=y_tile)
            return y

        @nki_op("test::add_one", mutates_args={})
        def add_one(x: torch.Tensor) -> torch.Tensor:
            y = torch.empty_like(x)
            add_one_kernel(x, y)
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
