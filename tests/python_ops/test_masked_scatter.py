"""Tests for the masked_scatter op family.

masked_scatter copies elements from ``source`` into ``self`` at the positions
where ``mask`` is True. It reads ``source`` in row-major (flattened) order and
broadcasts ``mask`` to ``self``. The result keeps ``self.dtype``; masked_scatter
does not type-promote against ``source``.

Overloads under test:
* ``aten::masked_scatter`` — functional.
* ``aten::masked_scatter.out`` — writes into ``out`` and leaves ``self`` alone.
* ``aten::masked_scatter_`` — in-place; mutates ``self``.
* ``aten::masked_scatter_backward`` — gradient w.r.t. ``source`` (eager autograd
  dispatches it as a separate op, so it must run on Neuron, not fall back).
"""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


def _bool_mask(shape, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(shape, generator=g) > 0.5).to(device)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestMaskedScatter:
    """Functional and .out overloads."""

    @pytest.fixture
    def device(self):
        return torch.device("neuron")

    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 5)])
    def test_basic_shapes(self, device, shape):
        with track_neuron_ops():
            x = torch.randn(shape, device=device)
            mask = _bool_mask(shape, device)
            source = torch.randn(mask.sum().item() + 5, device=device)

            neuron_result = x.masked_scatter(mask, source)
            cpu_result = x.cpu().masked_scatter(mask.cpu(), source.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_scatter")

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float64,
            torch.float16,
            torch.bfloat16,
            torch.int32,
            torch.int64,
            torch.bool,
        ],
    )
    def test_dtypes(self, device, dtype):
        shape = (4, 6)
        with track_neuron_ops():
            if dtype == torch.bool:
                x = torch.randint(0, 2, shape, dtype=dtype, device=device)
                source = torch.randint(0, 2, (shape[0] * shape[1],), dtype=dtype, device=device)
            elif dtype.is_floating_point:
                x = torch.randn(shape, device=device).to(dtype)
                source = torch.randn(shape[0] * shape[1], device=device).to(dtype)
            else:
                x = torch.randint(-8, 9, shape, dtype=dtype, device=device)
                source = torch.randint(-8, 9, (shape[0] * shape[1],), dtype=dtype, device=device)
            mask = _bool_mask(shape, device)

            neuron_result = x.masked_scatter(mask, source)
            cpu_result = x.cpu().masked_scatter(mask.cpu(), source.cpu())

            assert neuron_result.dtype == cpu_result.dtype
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_scatter")

    def test_row_major_source_order_bit_exact(self, device):
        """source is consumed in row-major order; a data-move op must be exact."""
        with track_neuron_ops():
            x = torch.zeros(3, 4, device=device)
            mask = torch.tensor(
                [
                    [True, False, True, False],
                    [False, True, True, False],
                    [True, False, False, True],
                ],
                device=device,
            )
            source = torch.arange(1, 20, dtype=torch.float32, device=device)

            neuron_result = x.masked_scatter(mask, source)
            cpu_result = x.cpu().masked_scatter(mask.cpu(), source.cpu())

            assert torch.equal(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_scatter")

    @pytest.mark.parametrize(
        "self_shape,mask_shape",
        [
            ((4, 3), (1, 3)),  # broadcast leading dim of mask up to self
            ((4, 3), (4, 1)),
            ((2, 3, 4), (1, 3, 4)),
            ((2, 3, 4), (4,)),  # fewer dims
            ((3,), (2, 3)),  # mask has more dims than self: result grows
        ],
    )
    def test_broadcast_mask(self, device, self_shape, mask_shape):
        with track_neuron_ops():
            x = torch.randn(self_shape, device=device)
            mask = _bool_mask(mask_shape, device)
            out_numel = torch.broadcast_shapes(self_shape, mask_shape).numel()
            source = torch.randn(out_numel, device=device)

            neuron_result = x.masked_scatter(mask, source)
            cpu_result = x.cpu().masked_scatter(mask.cpu(), source.cpu())

            assert neuron_result.shape == cpu_result.shape
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_scatter")

    def test_all_true_and_all_false_masks(self, device):
        with track_neuron_ops():
            x = torch.randn(3, 4, device=device)
            source = torch.arange(12, dtype=torch.float32, device=device)

            all_true = x.masked_scatter(torch.ones(3, 4, dtype=torch.bool, device=device), source)
            all_false = x.masked_scatter(torch.zeros(3, 4, dtype=torch.bool, device=device), source)

            torch.testing.assert_close(all_true.cpu(), source.cpu().reshape(3, 4))
            torch.testing.assert_close(all_false.cpu(), x.cpu())
            assert_op_runs_on_neuron("aten::masked_scatter")

    def test_empty(self, device):
        with track_neuron_ops():
            x = torch.empty(0, device=device)
            mask = torch.empty(0, dtype=torch.bool, device=device)
            source = torch.empty(0, device=device)

            neuron_result = x.masked_scatter(mask, source)
            assert neuron_result.numel() == 0
            assert neuron_result.shape == torch.Size([0])
            assert_op_runs_on_neuron("aten::masked_scatter")

    def test_single_element(self, device):
        with track_neuron_ops():
            x = torch.tensor([3.0], device=device)
            mask = torch.tensor([True], device=device)
            source = torch.tensor([-1.0], device=device)

            neuron_result = x.masked_scatter(mask, source)
            cpu_result = x.cpu().masked_scatter(mask.cpu(), source.cpu())

            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_scatter")

    @pytest.mark.parametrize("mask_value", [True, False])
    def test_zero_dim_self(self, device, mask_value):
        with track_neuron_ops():
            x = torch.tensor(5.0, device=device)
            mask = torch.tensor(mask_value, device=device)
            source = torch.tensor([9.0], device=device)

            neuron_result = x.masked_scatter(mask, source)
            cpu_result = x.cpu().masked_scatter(mask.cpu(), source.cpu())

            assert neuron_result.shape == torch.Size([])
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_scatter")

    def test_does_not_mutate_self(self, device):
        with track_neuron_ops():
            x = torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3)
            before = x.cpu().clone()
            mask = torch.tensor([[True, False, True], [False, True, False]], device=device)
            source = torch.arange(10, dtype=torch.float32, device=device)

            x.masked_scatter(mask, source)

            torch.testing.assert_close(x.cpu(), before)
            assert_op_runs_on_neuron("aten::masked_scatter")

    # ── out= : aten::masked_scatter.out ──────────────────────────────────────

    def test_out_variant(self, device):
        with track_neuron_ops():
            x = torch.zeros(3, 4, device=device)
            self_before = x.cpu().clone()
            mask = torch.tensor([[True, False, True, False]] * 3, device=device)
            source = torch.arange(1, 13, dtype=torch.float32, device=device)
            out = torch.empty(3, 4, device=device)

            result = torch.ops.aten.masked_scatter.out(x, mask, source, out=out)
            expected = x.cpu().masked_scatter(mask.cpu(), source.cpu())

            assert result.data_ptr() == out.data_ptr()
            torch.testing.assert_close(out.cpu(), expected)
            # out= must not corrupt self.
            torch.testing.assert_close(x.cpu(), self_before)
            assert_op_runs_on_neuron("aten::masked_scatter.out")

    def test_out_noncontiguous(self, device):
        with track_neuron_ops():
            x = torch.zeros(3, 4, device=device)
            mask = torch.tensor([[True, False, True, False]] * 3, device=device)
            source = torch.arange(1, 13, dtype=torch.float32, device=device)
            buffer = torch.zeros(3, 8, device=device)
            out = buffer[:, ::2]
            assert not out.is_contiguous()

            result = torch.ops.aten.masked_scatter.out(x, mask, source, out=out)
            expected = x.cpu().masked_scatter(mask.cpu(), source.cpu())

            assert result.data_ptr() == out.data_ptr()
            torch.testing.assert_close(out.cpu(), expected)
            torch.testing.assert_close(buffer[:, 1::2].cpu(), torch.zeros(3, 4))
            assert_op_runs_on_neuron("aten::masked_scatter.out")

    # ── in-place : aten::masked_scatter_ ─────────────────────────────────────

    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
    def test_inplace(self, device, shape):
        with track_neuron_ops():
            x = torch.randn(shape, device=device)
            expected = x.cpu().clone()
            mask = _bool_mask(shape, device)
            source = torch.randn(mask.sum().item() + 3, device=device)

            result = x.masked_scatter_(mask, source)
            expected.masked_scatter_(mask.cpu(), source.cpu())

            assert result.data_ptr() == x.data_ptr()
            torch.testing.assert_close(x.cpu(), expected)
            assert_op_runs_on_neuron("aten::masked_scatter_")

    @pytest.mark.parametrize("mask_shape", [(1, 3), (3,)])
    def test_inplace_broadcast_mask(self, device, mask_shape):
        with track_neuron_ops():
            x = torch.randn(2, 3, device=device)
            expected = x.cpu().clone()
            mask = _bool_mask(mask_shape, device)
            source = torch.arange(1, 13, dtype=torch.float32, device=device)

            x.masked_scatter_(mask, source)
            expected.masked_scatter_(mask.cpu(), source.cpu())

            torch.testing.assert_close(x.cpu(), expected)
            assert_op_runs_on_neuron("aten::masked_scatter_")

    def test_inplace_expanding_mask_raises(self, device):
        x = torch.zeros(1, 3, device=device)
        mask = torch.ones(2, 3, dtype=torch.bool, device=device)
        source = torch.arange(6, dtype=torch.float32, device=device)

        with pytest.raises(RuntimeError, match=r"doesn't match the broadcast shape"):
            x.masked_scatter_(mask, source)

    # ── error contract ───────────────────────────────────────────────────────

    @pytest.mark.parametrize("mask_dtype", [torch.uint8, torch.int32, torch.float32])
    def test_non_bool_mask_raises(self, device, mask_dtype):
        x = torch.randn(2, 3, device=device)
        mask = torch.ones(2, 3, dtype=mask_dtype, device=device)
        source = torch.arange(6, dtype=torch.float32, device=device)

        with pytest.raises(RuntimeError, match=r"only supports boolean masks"):
            x.masked_scatter(mask, source)

    def test_source_too_small_raises(self, device):
        """source must hold at least as many elements as the True mask count.

        CPU raises here. The Neuron gather clamps an out-of-range rank instead of
        a fault, so the eager path checks the count on the host and raises.
        """
        x = torch.arange(10, dtype=torch.float32, device=device)
        mask = torch.tensor(
            [0, 0, 0, 0, 1, 0, 1, 0, 1, 0], dtype=torch.bool, device=device
        )  # 3 True positions
        short = torch.zeros(2, dtype=torch.float32, device=device)

        with pytest.raises(RuntimeError, match=r"needs 3"):
            x.masked_scatter(mask, short)
        with pytest.raises(RuntimeError, match=r"needs 3"):
            x.clone().masked_scatter_(mask, short)

        # An exact-size source must not raise.
        exact = torch.zeros(3, dtype=torch.float32, device=device)
        x.masked_scatter(mask, exact)

    # ── autograd : aten::masked_scatter_backward ─────────────────────────────

    def test_backward(self, device):
        """Gradients flow to source (masked_scatter_backward) and self (masked_fill).

        Assert the backward op runs on Neuron; a forward-only assert would miss a
        silent CPU fallback of the separate backward op.
        """
        mask_cpu = torch.tensor([[True, False, True], [False, True, False]])

        def run(dev):
            x = torch.arange(6, dtype=torch.float32, device=dev).reshape(2, 3)
            x = x.detach().requires_grad_(True)
            s = torch.arange(1, 5, dtype=torch.float32, device=dev).requires_grad_(True)
            out = x.masked_scatter(mask_cpu.to(dev), s)
            (out * 2).sum().backward()
            return x.grad, s.grad

        with track_neuron_ops():
            gx, gs = run(device)
            assert_op_runs_on_neuron("aten::masked_scatter")
            assert_op_runs_on_neuron("aten::masked_scatter_backward")

        gx_cpu, gs_cpu = run(torch.device("cpu"))
        torch.testing.assert_close(gx.cpu(), gx_cpu)
        torch.testing.assert_close(gs.cpu(), gs_cpu)
