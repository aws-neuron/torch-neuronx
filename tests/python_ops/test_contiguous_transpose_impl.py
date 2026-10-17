"""Tests ensuring transpose-based contiguous path is used when applicable."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    track_neuron_ops,
)


@pytest.mark.usefixtures("enable_metrics_for_class")
class TestContiguousTranspose:
    """Test class for contiguous transpose operations"""

    # ── New tests: work with both Python and C++ dispatch paths ──

    def test_contiguous_transpose_runs_on_neuron(self):
        """Transpose contiguous should run on Neuron, not CPU fallback."""
        with track_neuron_ops():
            x = torch.randn(128, 256, device="neuron")
            y_view = x.transpose(0, 1)
            y = y_view.contiguous()

            assert y.is_contiguous()
            assert y.shape == (256, 128)
            assert torch.allclose(y, y_view)
            assert_op_runs_on_neuron("aten::contiguous")

    @pytest.mark.xfail(
        reason="stride-2 slices use contiguous_generic_kernel (CPU fallback) — not a Neuron op",
    )
    def test_contiguous_non_permutation_runs_on_neuron(self):
        """Non-permutation contiguous should run on Neuron, not CPU fallback."""
        with track_neuron_ops():
            x_cpu = torch.arange(24, dtype=torch.float32).reshape(4, 6)
            x = x_cpu.to("neuron")
            y = x[:, ::2]
            assert not y.is_contiguous()

            z = y.contiguous()
            assert z.is_contiguous()
            assert torch.allclose(z, y)
            assert_op_runs_on_neuron("aten::contiguous")

    def test_different_permutations_produce_correct_results(self):
        """Different permutations should each produce correct results."""
        with track_neuron_ops():
            base_tensor = torch.arange(
                1 * 64 * 32 * 128, dtype=torch.bfloat16, device="neuron"
            ).reshape(1, 64, 32, 128)

            view_1 = base_tensor.permute(0, 2, 3, 1)
            result_1 = view_1.contiguous()
            expected_1 = base_tensor.cpu().permute(0, 2, 3, 1).contiguous()
            assert torch.allclose(result_1.cpu(), expected_1)

            view_2 = base_tensor.permute(0, 2, 1, 3)
            result_2 = view_2.contiguous()
            expected_2 = base_tensor.cpu().permute(0, 2, 1, 3).contiguous()
            assert torch.allclose(result_2.cpu(), expected_2)

            assert_op_runs_on_neuron("aten::contiguous")

    def test_contiguous_transpose_2d_and_autograd(self):
        """2D transpose path should preserve values and gradients."""
        x = torch.randn(128, 256, device="neuron")
        y_view = x.transpose(0, 1)
        y = y_view.contiguous()

        assert y.is_contiguous()
        assert y.shape == y_view.shape
        assert torch.allclose(y, y_view)

        xg = torch.randn(128, 256, device="neuron", requires_grad=True)
        t_view = xg.transpose(0, 1)
        yg = t_view.contiguous()

        yg.retain_grad()
        yg.sum().backward()

        assert xg.grad is not None and torch.allclose(xg.grad, torch.ones_like(xg))
        assert yg.grad is not None and torch.allclose(yg.grad, torch.ones_like(yg))

    def test_contiguous_transpose_3d_permute_201(self):
        """3D permutation (2,0,1) uses transpose path; values preserved."""
        x = torch.randn(4, 5, 6, device="neuron")
        y_view = x.permute(2, 0, 1)
        y = y_view.contiguous()

        assert y.is_contiguous()
        assert y.shape == y_view.shape
        assert torch.allclose(y, y_view)

    @pytest.mark.parametrize(
        "rank,shape",
        [
            (4, (2, 3, 4, 5)),
            (5, (2, 3, 4, 5, 2)),
            (6, (2, 3, 4, 5, 2, 3)),
        ],
    )
    def test_contiguous_transpose_random_perms_rank_4_to_6(self, rank, shape):
        """Random non-identity perms on ranks 4-6 preserve values; gradients flow."""
        import random

        rng = random.Random(0)

        # Try a few random permutations per rank
        for _ in range(3):
            perm = list(range(rank))
            rng.shuffle(perm)
            if perm == list(range(rank)):
                rng.shuffle(perm)

            x = torch.randn(*shape, device="neuron")
            y_view = x.permute(*perm)
            y = y_view.contiguous()

            assert y.is_contiguous()
            assert y.shape == y_view.shape
            assert torch.allclose(y.cpu(), y_view.cpu())

        # Deterministic autograd check: rotate axes by 1
        perm0 = tuple(range(rank))
        perm0 = perm0[1:] + perm0[:1]
        xg = torch.randn(*shape, device="neuron", requires_grad=True)
        yg = xg.permute(*perm0).contiguous()
        yg.sum().backward()
        assert xg.grad is not None and torch.allclose(xg.grad.cpu(), torch.ones_like(xg).cpu())
