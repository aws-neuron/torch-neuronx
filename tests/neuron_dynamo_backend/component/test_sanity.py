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

    def test_cpu_tensor_raises_when_autocopy_disabled(self, monkeypatch):
        """Test CPU tensors raise error when TORCH_NEURONX_DYNAMO_DISABLE_CPU_AUTOCOPY=1"""
        monkeypatch.setenv("TORCH_NEURONX_DYNAMO_DISABLE_CPU_AUTOCOPY", "1")
        monkeypatch.setenv("TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION", "1")

        @torch.compile(backend="neuron")
        def fn(x):
            return x * 2

        x_cpu = torch.randn(10, device="cpu")

        with pytest.raises(RuntimeError, match=r"Input tensor at index 0 is on cpu device"):
            fn(x_cpu)

    def test_neuron_tensor_works_when_autocopy_disabled(self, monkeypatch):
        """Test neuron tensors work normally when autocopy is disabled"""
        monkeypatch.setenv("TORCH_NEURONX_DYNAMO_DISABLE_CPU_AUTOCOPY", "1")

        @torch.compile(backend="neuron")
        def fn(x):
            return x * 2

        x_neuron = torch.randn(10, device="neuron")
        y = fn(x_neuron)

        assert y.device.type == "neuron"
        torch.testing.assert_close(y.cpu(), x_neuron.cpu() * 2, rtol=1e-4, atol=1e-4)

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

    @pytest.mark.skipif(
        os.environ.get("TORCH_NEURONX_MLIR_ATEN_OPS") != "1",
        reason="Requires TORCH_NEURONX_MLIR_ATEN_OPS=1",
    )
    def test_print_with_cpu_conversion(self, capsys):
        """Test that printing tensors with .cpu() works inside compiled functions."""

        @torch.compile(backend="neuron")
        def fn(x, y):
            tmp = torch.add(x, y)
            print(f"Output from print: {tmp.cpu()}")
            return tmp * 2

        x = torch.tensor([3.0, 4.0], dtype=torch.float32, device="neuron")
        y = torch.tensor([1.0, 2.0], dtype=torch.float32, device="neuron")

        result = fn(x, y)

        # Verify output is correct
        assert result.device.type == "neuron"
        expected = (x + y) * 2
        torch.testing.assert_close(result.cpu(), expected.cpu(), rtol=1e-4, atol=1e-4)

        # Verify print output contains our message and tensor values
        captured = capsys.readouterr()
        assert "Output from print:" in captured.out
        assert "4." in captured.out and "6." in captured.out

    @pytest.mark.skipif(
        os.environ.get("TORCH_NEURONX_MLIR_ATEN_OPS") != "1",
        reason="Requires TORCH_NEURONX_MLIR_ATEN_OPS=1",
    )
    def test_print_cpu_tensor_operations(self, capsys):
        """Test various operations on CPU tensors inside compiled functions."""

        @torch.compile(backend="neuron")
        def fn(x):
            # Convert to CPU
            cpu_x = x.cpu()

            # Test basic arithmetic on CPU tensor
            result = cpu_x + 1
            print(f"CPU arithmetic: {result}")

            # Test comparison operations
            mask = cpu_x > 2.0
            print(f"CPU comparison: {mask}")

            # Test reduction operations
            sum_val = cpu_x.sum()
            print(f"CPU sum: {sum_val}")

            return x * 2

        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device="neuron")
        result = fn(x)

        assert result.device.type == "neuron"
        captured = capsys.readouterr()
        assert "CPU arithmetic:" in captured.out
        assert "CPU comparison:" in captured.out
        assert "CPU sum:" in captured.out

    @pytest.mark.skipif(
        os.environ.get("TORCH_NEURONX_MLIR_ATEN_OPS") != "1",
        reason="Requires TORCH_NEURONX_MLIR_ATEN_OPS=1",
    )
    def test_print_multiple_cpu_conversions(self, capsys):
        """Test multiple .cpu() calls in compiled function."""

        @torch.compile(backend="neuron")
        def fn(x, y):
            tmp1 = x + y
            print(f"First: {tmp1.cpu()}")

            tmp2 = tmp1 * 2
            print(f"Second: {tmp2.cpu()}")

            return tmp2

        x = torch.tensor([1.0, 2.0], dtype=torch.float32, device="neuron")
        y = torch.tensor([3.0, 4.0], dtype=torch.float32, device="neuron")

        result = fn(x, y)

        assert result.device.type == "neuron"
        captured = capsys.readouterr()
        assert "First:" in captured.out
        assert "Second:" in captured.out

    @pytest.mark.skipif(
        os.environ.get("TORCH_NEURONX_MLIR_ATEN_OPS") != "1",
        reason="Requires TORCH_NEURONX_MLIR_ATEN_OPS=1",
    )
    def test_print_cpu_tensor_indexing(self, capsys):
        """Test indexing CPU tensors inside compiled functions."""

        @torch.compile(backend="neuron")
        def fn(x):
            cpu_x = x.cpu()

            # Test indexing
            first = cpu_x[0]
            print(f"First element: {first}")

            # Test slicing
            subset = cpu_x[1:]
            print(f"Slice: {subset}")

            return x + 1

        x = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32, device="neuron")
        result = fn(x)

        assert result.device.type == "neuron"
        captured = capsys.readouterr()
        assert "First element:" in captured.out
        assert "Slice:" in captured.out

    @pytest.mark.skipif(
        os.environ.get("TORCH_NEURONX_MLIR_ATEN_OPS") != "1",
        reason="Requires TORCH_NEURONX_MLIR_ATEN_OPS=1",
    )
    def test_print_cpu_tensor_with_item(self, capsys):
        """Test .item() on CPU tensors inside compiled functions."""

        @torch.compile(backend="neuron")
        def fn(x):
            cpu_x = x.cpu()

            # Test .item() for scalar extraction
            val = cpu_x[0].item()
            print(f"Scalar value: {val}")

            return x * 2

        x = torch.tensor([5.0], dtype=torch.float32, device="neuron")
        result = fn(x)

        assert result.device.type == "neuron"
        captured = capsys.readouterr()
        assert "Scalar value: 5" in captured.out


class TestEmptyTensorConsistency:
    """Test no diverging behavior b/w eager and compile for empty tensor handling."""

    @pytest.mark.xfail(reason="Empty tensor handling not yet implemented in compile path")
    @pytest.mark.parametrize(
        "op,expected",
        [
            (torch.all, torch.tensor(True)),
            (torch.prod, torch.tensor(1)),
        ],
    )
    def test_reduction_empty_consistency(self, op, expected):
        """Reduction ops on empty tensors should return identity and match between compile/eager."""
        x_cpu = torch.empty(0, 3, dtype=torch.float32)
        x = x_cpu.to("neuron")

        eager_result = op(x).cpu()
        compile_result = torch.compile(op, backend="neuron", fullgraph=True)(x).cpu()

        torch.testing.assert_close(eager_result, expected)
        torch.testing.assert_close(compile_result, expected)

    @pytest.mark.skip(reason="Empty tensor handling not yet implemented in compile path")
    def test_index_select_empty_index_consistency(self):
        """Index select with empty index should match between compile/eager."""
        x_cpu = torch.randn(5, 3)
        idx_cpu = torch.empty(0, dtype=torch.long)

        x = x_cpu.to("neuron")
        idx = idx_cpu.to("neuron")

        def index_op(t):
            return torch.index_select(t, 0, idx)

        expected = torch.index_select(x_cpu, 0, idx_cpu)
        eager_result = index_op(x).cpu()
        compile_result = torch.compile(index_op, backend="neuron", fullgraph=True)(x).cpu()

        assert eager_result.shape == compile_result.shape == expected.shape
        assert torch.equal(eager_result, compile_result)
