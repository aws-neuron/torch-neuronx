"""Tests for aten::_adaptive_avg_pool2d and its backward on Neuron."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestAdaptiveAvgPool2d:
    @pytest.mark.parametrize(
        "in_hw,out_hw",
        [
            ((8, 8), (1, 1)),  # global average pool
            ((8, 8), (4, 4)),  # divisible, 2x2 kernel
            ((8, 8), (2, 2)),  # divisible, 4x4 kernel
            ((12, 18), (4, 6)),  # divisible, non-square
            ((6, 6), (4, 4)),  # non-divisible
            ((10, 10), (3, 3)),  # non-divisible
        ],
    )
    def test_forward(self, in_hw, out_hw):
        x_cpu = torch.randn(2, 3, *in_hw)
        with track_neuron_ops():
            result = torch.ops.aten._adaptive_avg_pool2d(x_cpu.to("neuron"), list(out_hw))
            assert_op_runs_on_neuron("aten::_adaptive_avg_pool2d")
        expected = torch.ops.aten._adaptive_avg_pool2d(x_cpu, list(out_hw))
        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_dtypes(self, dtype):
        x_cpu = torch.randn(1, 3, 8, 8, dtype=dtype)
        with track_neuron_ops():
            result = torch.ops.aten._adaptive_avg_pool2d(x_cpu.to("neuron"), [4, 4])
            assert_op_runs_on_neuron("aten::_adaptive_avg_pool2d")
        expected = torch.ops.aten._adaptive_avg_pool2d(x_cpu, [4, 4])
        assert result.dtype == dtype
        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize(
        "in_hw,out_hw",
        [
            ((8, 8), (4, 4)),  # divisible
            ((12, 18), (4, 6)),  # divisible, non-square
            ((6, 6), (4, 4)),  # non-divisible
        ],
    )
    def test_backward(self, in_hw, out_hw):
        x_cpu = torch.randn(2, 3, *in_hw, requires_grad=True)
        x_n = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        out_cpu = torch.nn.functional.adaptive_avg_pool2d(x_cpu, out_hw)
        grad = torch.randn_like(out_cpu)

        with track_neuron_ops():
            out_n = torch.nn.functional.adaptive_avg_pool2d(x_n, out_hw)
            out_n.backward(grad.to("neuron"))
            assert_op_runs_on_neuron("aten::_adaptive_avg_pool2d_backward")

        out_cpu.backward(grad)
        torch.testing.assert_close(x_n.grad.cpu(), x_cpu.grad)

    def test_forward_out_variant(self):
        x_cpu = torch.randn(2, 3, 8, 8)
        expected = torch.ops.aten._adaptive_avg_pool2d(x_cpu, [4, 4])
        out_n = torch.empty_like(expected).to("neuron")
        with track_neuron_ops():
            result = torch.ops.aten._adaptive_avg_pool2d.out(x_cpu.to("neuron"), [4, 4], out=out_n)
            assert_op_runs_on_neuron("aten::_adaptive_avg_pool2d.out")
        torch.testing.assert_close(result.cpu(), expected)
        torch.testing.assert_close(out_n.cpu(), expected)

    def test_backward_out_variant(self):
        x_cpu = torch.randn(2, 3, 8, 8)
        grad_cpu = torch.randn(2, 3, 4, 4)
        expected = torch.ops.aten._adaptive_avg_pool2d_backward(grad_cpu, x_cpu)
        out_n = torch.empty_like(expected).to("neuron")
        with track_neuron_ops():
            result = torch.ops.aten._adaptive_avg_pool2d_backward.out(
                grad_cpu.to("neuron"), x_cpu.to("neuron"), out=out_n
            )
            assert_op_runs_on_neuron("aten::_adaptive_avg_pool2d_backward.out")
        torch.testing.assert_close(result.cpu(), expected)
        torch.testing.assert_close(out_n.cpu(), expected)
