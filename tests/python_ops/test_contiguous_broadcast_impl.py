"""Tests ensuring broadcast-based contiguous path is used when applicable."""

import math

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.usefixtures("enable_metrics_for_class")
class TestContiguousBroadcast:
    """Test class for contiguous broadcast operations"""

    # ── New tests: work with both Python and C++ dispatch paths ──

    def test_contiguous_broadcast_runs_on_neuron(self):
        """Broadcast contiguous should run on Neuron, not CPU fallback."""
        with track_neuron_ops():
            scalar = torch.tensor(42.0, device="neuron")
            broadcasted_view = scalar.expand(128, 256)
            assert not broadcasted_view.is_contiguous()

            y = broadcasted_view.contiguous()

            assert y.is_contiguous()
            assert y.shape == (128, 256)
            assert torch.allclose(y.cpu(), broadcasted_view.cpu())
            assert torch.all(y == 42.0)
            assert_op_runs_on_neuron("aten::contiguous")

    def test_repeat_kv_runs_on_neuron(self):
        """repeat_kv pattern should run on Neuron, not CPU fallback."""

        def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
            bs, slen, n_kv_heads, head_dim = x.shape
            if n_rep == 1:
                return x
            return (
                torch.unsqueeze(x, dim=3)
                .expand(bs, slen, n_kv_heads, n_rep, head_dim)
                .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
            )

        with track_neuron_ops():
            x = torch.zeros(1, 4096, 8, 128, dtype=torch.bfloat16, device="neuron")
            y = _repeat_kv(x, 4)
            y_cpu = _repeat_kv(x.cpu(), 4)

            assert torch.allclose(y.cpu(), y_cpu)
            assert_op_runs_on_neuron("aten::contiguous")

    def test_broadcast_different_shapes_produce_correct_results(self):
        """Different broadcast shapes should each produce correct results."""
        with track_neuron_ops():
            scalar = torch.tensor(123.0, dtype=torch.bfloat16, device="neuron")

            result_1 = scalar.expand(4096, 2048).contiguous()
            assert result_1.is_contiguous()
            assert torch.all(result_1 == 123.0)

            result_2 = scalar.expand(2048, 4096).contiguous()
            assert result_2.is_contiguous()
            assert torch.all(result_2 == 123.0)

            assert_op_runs_on_neuron("aten::contiguous")

    def test_contiguous_broadcast_autograd(self):
        """Broadcast path should preserve gradients."""

        scalar_grad = torch.tensor(2.71, device="neuron", requires_grad=True)
        broadcasted_grad = scalar_grad.expand(128, 256)
        yg = broadcasted_grad.contiguous()
        yg.retain_grad()
        yg.sum().backward()

        scalar_grad_cpu = torch.tensor(2.71, device="cpu", requires_grad=True)
        broadcasted_grad_cpu = scalar_grad_cpu.expand(128, 256)
        yg_cpu = broadcasted_grad_cpu.contiguous()
        yg_cpu.retain_grad()
        yg_cpu.sum().backward()

        assert torch.allclose(
            scalar_grad.grad.cpu(), scalar_grad_cpu.grad
        ), "Neuron and CPU gradients should match"

        assert torch.allclose(
            yg.grad.cpu(), yg_cpu.grad
        ), "Neuron and CPU broadcasted gradients should match"

    @pytest.mark.parametrize(
        "shape",
        [
            (2,),
            (2, 3),
            (2, 3, 4),
        ],
    )
    def test_contiguous_broadcast_scalar(self, shape):
        """Making scalar broadcasted tensor contiguous match cpu"""
        scalar_val = 42.0
        scalar = torch.tensor(scalar_val, device="neuron")
        broadcasted_view = scalar.expand(*shape)
        assert not broadcasted_view.is_contiguous()
        assert all(s == 0 for s in broadcasted_view.stride())

        y = broadcasted_view.contiguous()

        assert y.is_contiguous()
        assert y.shape == broadcasted_view.shape
        assert torch.allclose(y.cpu(), broadcasted_view.cpu())
        assert torch.all(y == scalar_val)

    @pytest.mark.parametrize(
        "source_shape,target_shape",
        [
            ((1,), (2, 3, 4)),
            ((2, 1), (2, 3)),
            ((2, 3), (5, 2, 3)),
            ((2, 1, 4), (2, 3, 4)),
            ((1, 3, 1, 4), (2, 3, 5, 4)),
        ],
    )
    def test_contiguous_broadcast_various_dimensions(self, source_shape, target_shape):
        """Making broadcasted tensor contiguous match cpu"""
        numel = math.prod(source_shape)
        source_neuron = torch.arange(numel, dtype=torch.float32, device="neuron").reshape(
            source_shape
        )
        source_cpu = torch.arange(numel, dtype=torch.float32, device="cpu").reshape(source_shape)

        # Perform broadcast and contiguous operations on neuron
        broadcasted_neuron = source_neuron.expand(*target_shape)
        assert not broadcasted_neuron.is_contiguous()
        contiguous_neuron = broadcasted_neuron.contiguous()
        assert contiguous_neuron.is_contiguous()

        # Perform same operations on CPU
        broadcasted_cpu = source_cpu.expand(*target_shape)
        contiguous_cpu = broadcasted_cpu.contiguous()

        # Verify neuron tensor properties
        assert torch.allclose(contiguous_neuron.cpu(), contiguous_cpu)
