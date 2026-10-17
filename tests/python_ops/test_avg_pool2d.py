"""Tests for aten::avg_pool2d / avg_pool2d_backward and their out variants on Neuron."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestAvgPool2d:
    """Test suite for aten::avg_pool2d and aten::avg_pool2d.out."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.mark.parametrize(
        "shape,kernel,stride",
        [
            ((2, 3, 8, 8), 2, 2),
            ((2, 3, 8, 8), 2, None),  # stride defaults to kernel_size
            ((1, 3, 8, 8), (2, 2), (2, 2)),
            ((2, 4, 12, 18), (3, 3), (2, 2)),
            ((3, 8, 8), 4, 4),  # 3D (unbatched)
        ],
    )
    def test_forward_shapes(self, shape, kernel, stride):
        x_cpu = torch.randn(*shape)
        with track_neuron_ops():
            result = torch.nn.functional.avg_pool2d(x_cpu.to("neuron"), kernel, stride)
            assert_op_runs_on_neuron("aten::avg_pool2d")
        expected = torch.nn.functional.avg_pool2d(x_cpu, kernel, stride)
        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize("padding", [0, 1, (1, 1)])
    def test_forward_padding(self, padding):
        x_cpu = torch.randn(2, 3, 8, 8)
        with track_neuron_ops():
            result = torch.nn.functional.avg_pool2d(
                x_cpu.to("neuron"), kernel_size=3, stride=2, padding=padding
            )
            assert_op_runs_on_neuron("aten::avg_pool2d")
        expected = torch.nn.functional.avg_pool2d(x_cpu, kernel_size=3, stride=2, padding=padding)
        torch.testing.assert_close(result.cpu(), expected)

    def test_forward_count_include_pad_false(self):
        x_cpu = torch.randn(2, 3, 8, 8)
        with track_neuron_ops():
            result = torch.nn.functional.avg_pool2d(
                x_cpu.to("neuron"), kernel_size=3, stride=2, padding=1, count_include_pad=False
            )
            assert_op_runs_on_neuron("aten::avg_pool2d")
        expected = torch.nn.functional.avg_pool2d(
            x_cpu, kernel_size=3, stride=2, padding=1, count_include_pad=False
        )
        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize(
        "hw,kernel,stride,padding,count_include_pad",
        [
            # Odd input, no pad: ceil creates clipped edge windows (1 or 2 real
            # elements). CPU divides those by the real count, not kernel_prod.
            ((9, 9), 2, 2, 0, True),
            ((9, 9), 2, 2, 0, False),
            ((9, 9), 3, 2, 0, True),
            # Even input: ceil may not add padding (equals floor) — sanity.
            ((10, 10), 2, 2, 0, True),
            # ceil + explicit padding: explicit pad counts, ceil pad does not.
            ((10, 10), 2, 2, 1, True),
            ((10, 10), 2, 2, 1, False),
            ((9, 9), 3, 2, 1, True),
            # Non-square.
            ((9, 12), (2, 3), 2, 0, True),
        ],
    )
    def test_forward_ceil_mode(self, hw, kernel, stride, padding, count_include_pad):
        """ceil_mode=True must match CPU across clipped-edge and padding configs.

        The clipped edge windows (H=9,p=0 style) are the ones that exposed the
        divisor bug: ceil-added cells must never count toward the mean, even
        under count_include_pad=True.
        """
        x_cpu = torch.randn(2, 3, *hw)
        with track_neuron_ops():
            result = torch.nn.functional.avg_pool2d(
                x_cpu.to("neuron"),
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                ceil_mode=True,
                count_include_pad=count_include_pad,
            )
            assert_op_runs_on_neuron("aten::avg_pool2d")
        expected = torch.nn.functional.avg_pool2d(
            x_cpu,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            ceil_mode=True,
            count_include_pad=count_include_pad,
        )
        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize("divisor", [5, 9, 10, -4])
    def test_forward_divisor_override(self, divisor):
        """divisor_override forces a fixed divisor; CPU allows negatives too."""
        x_cpu = torch.randn(2, 3, 8, 8)
        with track_neuron_ops():
            result = torch.nn.functional.avg_pool2d(
                x_cpu.to("neuron"), kernel_size=3, stride=3, divisor_override=divisor
            )
            assert_op_runs_on_neuron("aten::avg_pool2d")
        expected = torch.nn.functional.avg_pool2d(
            x_cpu, kernel_size=3, stride=3, divisor_override=divisor
        )
        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_forward_dtypes(self, dtype):
        x_cpu = torch.randn(1, 3, 8, 8, dtype=dtype)
        with track_neuron_ops():
            result = torch.nn.functional.avg_pool2d(x_cpu.to("neuron"), 2, 2)
            assert_op_runs_on_neuron("aten::avg_pool2d")
        expected = torch.nn.functional.avg_pool2d(x_cpu, 2, 2)
        assert result.dtype == dtype
        torch.testing.assert_close(result.cpu(), expected)

    def test_forward_out_variant(self):
        x_cpu = torch.randn(2, 3, 8, 8)
        expected = torch.ops.aten.avg_pool2d(x_cpu, [2, 2], [2, 2], [0, 0], False, True, None)
        out_n = torch.empty_like(expected).to("neuron")
        with track_neuron_ops():
            result = torch.ops.aten.avg_pool2d.out(
                x_cpu.to("neuron"), [2, 2], [2, 2], [0, 0], False, True, None, out=out_n
            )
            assert_op_runs_on_neuron("aten::avg_pool2d.out")
        assert out_n.data_ptr() == result.data_ptr()
        torch.testing.assert_close(out_n.cpu(), expected)

    # ── backward ──────────────────────────────────────────────────────
    @pytest.mark.parametrize(
        "shape,kernel,stride,padding",
        [
            ((2, 3, 8, 8), 2, 2, 0),
            ((2, 3, 8, 8), 3, 2, 1),
            ((2, 4, 12, 18), (3, 3), (2, 2), 0),
        ],
    )
    def test_backward(self, shape, kernel, stride, padding):
        x_cpu = torch.randn(*shape, requires_grad=True)
        x_n = x_cpu.detach().clone().to("neuron").requires_grad_(True)
        out_cpu = torch.nn.functional.avg_pool2d(x_cpu, kernel, stride, padding)
        grad = torch.randn_like(out_cpu)
        with track_neuron_ops():
            out_n = torch.nn.functional.avg_pool2d(x_n, kernel, stride, padding)
            out_n.backward(grad.to("neuron"))
            assert_op_runs_on_neuron("aten::avg_pool2d_backward")
        out_cpu.backward(grad)
        torch.testing.assert_close(x_n.grad.cpu(), x_cpu.grad)

    def test_backward_count_include_pad_false(self):
        x_cpu = torch.randn(2, 3, 8, 8, requires_grad=True)
        x_n = x_cpu.detach().clone().to("neuron").requires_grad_(True)
        pool = lambda t: torch.nn.functional.avg_pool2d(  # noqa: E731
            t, 3, 2, 1, count_include_pad=False
        )
        out_cpu = pool(x_cpu)
        grad = torch.randn_like(out_cpu)
        with track_neuron_ops():
            out_n = pool(x_n)
            out_n.backward(grad.to("neuron"))
            assert_op_runs_on_neuron("aten::avg_pool2d_backward")
        out_cpu.backward(grad)
        torch.testing.assert_close(x_n.grad.cpu(), x_cpu.grad)

    def test_backward_out_variant(self):
        x_cpu = torch.randn(2, 3, 8, 8)
        grad_cpu = torch.ops.aten.avg_pool2d(x_cpu, [2, 2], [2, 2], [0, 0], False, True, None)
        expected = torch.ops.aten.avg_pool2d_backward(
            grad_cpu, x_cpu, [2, 2], [2, 2], [0, 0], False, True, None
        )
        gi_n = torch.empty_like(expected).to("neuron")
        with track_neuron_ops():
            result = torch.ops.aten.avg_pool2d_backward.grad_input(
                grad_cpu.to("neuron"),
                x_cpu.to("neuron"),
                [2, 2],
                [2, 2],
                [0, 0],
                False,
                True,
                None,
                grad_input=gi_n,
            )
            assert_op_runs_on_neuron("aten::avg_pool2d_backward.grad_input")
        assert gi_n.data_ptr() == result.data_ptr()
        torch.testing.assert_close(gi_n.cpu(), expected)

    def test_no_recompilation(self):
        import time

        with track_neuron_ops():
            x = torch.randn(2, 3, 16, 16, device="neuron")
            torch.nn.functional.avg_pool2d(x, 2, 2)
            torch.neuron.synchronize()
            start = time.perf_counter()
            torch.nn.functional.avg_pool2d(x, 2, 2)
            torch.neuron.synchronize()
            elapsed = time.perf_counter() - start
            assert elapsed < 0.1, f"Recompiled: {elapsed:.3f}s"
            assert_op_runs_on_neuron("aten::avg_pool2d")

    # ── error semantics (must match CPU messages verbatim) ─────────────
    def test_error_divisor_override_zero(self, device):
        x = torch.randn(2, 3, 8, 8, device=device)
        with pytest.raises(RuntimeError, match="divisor must be not zero"):
            torch.nn.functional.avg_pool2d(x, 3, 3, divisor_override=0)

    def test_error_stride_zero(self, device):
        x = torch.randn(2, 3, 8, 8, device=device)
        with pytest.raises(RuntimeError, match="stride should not be zero"):
            torch.nn.functional.avg_pool2d(x, 3, 0)

    def test_error_negative_padding(self, device):
        x = torch.randn(2, 3, 8, 8, device=device)
        with pytest.raises(RuntimeError, match="pad must be non-negative"):
            torch.nn.functional.avg_pool2d(x, 3, 2, -1)

    def test_error_padding_too_large(self, device):
        x = torch.randn(2, 3, 8, 8, device=device)
        with pytest.raises(
            RuntimeError, match="pad should be at most half of effective kernel size"
        ):
            torch.nn.functional.avg_pool2d(x, 3, 2, 2)

    def test_error_kernel_zero(self, device):
        x = torch.randn(2, 3, 8, 8, device=device)
        with pytest.raises(RuntimeError, match="integer out of range"):
            torch.nn.functional.avg_pool2d(x, 0)

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int16, torch.uint8, torch.bool])
    def test_error_non_float_input(self, device, dtype):
        """CPU supports float dtypes + int64; smaller int/bool dtypes are rejected."""
        x = torch.ones(2, 3, 8, 8, dtype=dtype, device=device)
        with pytest.raises(NotImplementedError, match='"avg_pool2d" not implemented'):
            torch.nn.functional.avg_pool2d(x, 2, 2)

    def test_int64_matches_cpu(self, device):
        """int64 is supported (integer average, truncated toward zero)."""
        x = torch.randint(-20, 20, (2, 3, 8, 8), dtype=torch.int64, device=device)
        with track_neuron_ops():
            neuron_result = torch.nn.functional.avg_pool2d(x, 2, 2)
            assert_op_runs_on_neuron("aten::avg_pool2d")
        cpu_result = torch.nn.functional.avg_pool2d(x.cpu(), 2, 2)
        assert neuron_result.dtype == torch.int64
        torch.testing.assert_close(neuron_result.cpu(), cpu_result)

    def test_error_backward_divisor_zero(self, device):
        grad = torch.randn(2, 3, 4, 4, device=device)
        x = torch.randn(2, 3, 8, 8, device=device)
        with pytest.raises(RuntimeError, match="divisor must be not zero"):
            torch.ops.aten.avg_pool2d_backward(grad, x, [2, 2], [2, 2], [0, 0], False, True, 0)
