"""Tests for the index_fill op family.

index_fill writes ``value`` into the slices of ``self`` that ``index`` selects
along ``dim``. The result keeps ``self.dtype``; index_fill casts ``value`` to
``self.dtype`` and does not type-promote.

Overloads under test:
* ``aten::index_fill.int_Scalar`` / ``aten::index_fill.int_Tensor`` - functional.
* ``aten::index_fill.int_Scalar_out`` / ``aten::index_fill.int_Tensor_out`` -
  write into ``out`` and leave ``self`` alone.
* ``aten::index_fill_.int_Scalar`` / ``aten::index_fill_.int_Tensor`` - in-place.
* backward w.r.t. ``self`` is ``grad.index_fill(dim, index, 0)``. Eager autograd
  dispatches index_fill again, so the backward must run on Neuron.
"""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestIndexFill:
    @pytest.fixture
    def device(self):
        return torch.device("neuron")

    @pytest.mark.parametrize(
        "shape,dim", [((5,), 0), ((3, 4), 1), ((2, 3, 4), 0), ((2, 3, 4, 5), 2)]
    )
    def test_basic_shapes(self, device, shape, dim):
        with track_neuron_ops():
            x = torch.randn(shape, device=device)
            index = torch.tensor([0, 1], device=device)

            neuron_result = x.index_fill(dim, index, 3.5)
            cpu_result = x.cpu().index_fill(dim, index.cpu(), 3.5)

            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::index_fill")

    @pytest.mark.parametrize(
        "dtype",
        [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64, torch.bool],
    )
    def test_dtypes(self, device, dtype):
        shape = (4, 6)
        with track_neuron_ops():
            if dtype == torch.bool:
                x = torch.randint(0, 2, shape, dtype=dtype, device=device)
                value = True
            elif dtype.is_floating_point:
                x = torch.randn(shape, device=device).to(dtype)
                value = 2.5
            else:
                x = torch.randint(-8, 9, shape, dtype=dtype, device=device)
                value = 7
            index = torch.tensor([0, 2], device=device)

            neuron_result = x.index_fill(1, index, value)
            cpu_result = x.cpu().index_fill(1, index.cpu(), value)

            assert neuron_result.dtype == cpu_result.dtype
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::index_fill")

    def test_int_value_into_float_self(self, device):
        """index_fill casts value to self.dtype. An int value into a float self
        must not fail the dtype check on the index_copy scatter."""
        with track_neuron_ops():
            x = torch.randn(6, 5, device=device)
            index = torch.tensor([0, 2, 4], device=device)

            neuron_result = x.index_fill(0, index, 0)
            cpu_result = x.cpu().index_fill(0, index.cpu(), 0)

            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::index_fill")

    def test_int_tensor_value_into_float_self(self, device):
        """A 0-d int tensor value into a float self exercises the value.to(self.dtype)
        cast on device (an int64 value arrives widened to int32, not self's float32)."""
        with track_neuron_ops():
            x = torch.randn(6, 5, device=device)
            index = torch.tensor([0, 2, 4], device=device)
            value = torch.tensor(7, device=device)

            neuron_result = x.index_fill(0, index, value)
            cpu_result = x.cpu().index_fill(0, index.cpu(), value.cpu())

            assert neuron_result.dtype == x.dtype
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::index_fill")

    def test_data_move_bit_exact(self, device):
        """index_fill only moves a constant into position, so compare bit-exact."""
        with track_neuron_ops():
            x = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
            index = torch.tensor([0, 3], device=device)

            neuron_result = x.index_fill(1, index, -1.0)
            cpu_result = x.cpu().index_fill(1, index.cpu(), -1.0)

            assert torch.equal(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::index_fill")

    @pytest.mark.parametrize("dim", [-1, -2])
    def test_negative_dim(self, device, dim):
        # A negative index is a family-wide Neuron limitation (index_copy and
        # index_select fault the same way), so cover a negative dim only.
        with track_neuron_ops():
            x = torch.randn(5, 4, device=device)
            idx = torch.tensor([0, 2], device=device)

            neuron_result = x.index_fill(dim, idx, 9.0)
            cpu_result = x.cpu().index_fill(dim, idx.cpu(), 9.0)

            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::index_fill")

    def test_int_tensor_value(self, device):
        """aten::index_fill.int_Tensor takes a 0-d tensor value."""
        with track_neuron_ops():
            x = torch.randn(6, 5, device=device)
            index = torch.tensor([1, 3], device=device)
            value = torch.tensor(2.5, device=device)

            neuron_result = x.index_fill(0, index, value)
            cpu_result = x.cpu().index_fill(0, index.cpu(), value.cpu())

            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::index_fill")

    def test_zero_dim_self(self, device):
        with track_neuron_ops():
            x = torch.tensor(5.0, device=device)
            index = torch.tensor([0], device=device)

            neuron_result = x.index_fill(0, index, 8.0)
            cpu_result = x.cpu().index_fill(0, index.cpu(), 8.0)

            assert neuron_result.shape == torch.Size([])
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::index_fill")

    def test_does_not_mutate_self(self, device):
        with track_neuron_ops():
            x = torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3)
            before = x.cpu().clone()
            index = torch.tensor([0], device=device)

            x.index_fill(1, index, 99.0)

            torch.testing.assert_close(x.cpu(), before)
            assert_op_runs_on_neuron("aten::index_fill")

    # -- out= : aten::index_fill.int_Scalar_out --------------------------------

    def test_out_variant(self, device):
        with track_neuron_ops():
            x = torch.randn(3, 4, device=device)
            self_before = x.cpu().clone()
            index = torch.tensor([0, 2], device=device)
            out = torch.empty(3, 4, device=device)

            result = torch.ops.aten.index_fill.int_Scalar_out(x, 1, index, 9.0, out=out)
            expected = x.cpu().index_fill(1, index.cpu(), 9.0)

            assert result.data_ptr() == out.data_ptr()
            torch.testing.assert_close(out.cpu(), expected)
            # out= must not corrupt self.
            torch.testing.assert_close(x.cpu(), self_before)
            assert_op_runs_on_neuron("aten::index_fill")

    def test_out_noncontiguous(self, device):
        with track_neuron_ops():
            x = torch.randn(3, 4, device=device)
            index = torch.tensor([0, 2], device=device)
            buffer = torch.zeros(3, 8, device=device)
            out = buffer[:, ::2]
            assert not out.is_contiguous()

            result = torch.ops.aten.index_fill.int_Scalar_out(x, 1, index, 9.0, out=out)
            expected = x.cpu().index_fill(1, index.cpu(), 9.0)

            assert result.data_ptr() == out.data_ptr()
            torch.testing.assert_close(out.cpu(), expected)
            torch.testing.assert_close(buffer[:, 1::2].cpu(), torch.zeros(3, 4))
            assert_op_runs_on_neuron("aten::index_fill")

    def test_out_int_tensor(self, device):
        """aten::index_fill.int_Tensor_out takes a 0-d tensor value."""
        with track_neuron_ops():
            x = torch.randn(3, 4, device=device)
            self_before = x.cpu().clone()
            index = torch.tensor([0, 2], device=device)
            value = torch.tensor(9.0, device=device)
            out = torch.empty(3, 4, device=device)

            result = torch.ops.aten.index_fill.int_Tensor_out(x, 1, index, value, out=out)
            expected = x.cpu().index_fill(1, index.cpu(), value.cpu())

            assert result.data_ptr() == out.data_ptr()
            torch.testing.assert_close(out.cpu(), expected)
            torch.testing.assert_close(x.cpu(), self_before)
            assert_op_runs_on_neuron("aten::index_fill")

    # -- in-place : aten::index_fill_ ------------------------------------------

    @pytest.mark.parametrize("shape,dim", [((5,), 0), ((3, 4), 1), ((2, 3, 4), 0)])
    def test_inplace(self, device, shape, dim):
        with track_neuron_ops():
            x = torch.randn(shape, device=device)
            expected = x.cpu().clone()
            index = torch.tensor([0, 1], device=device)

            result = x.index_fill_(dim, index, 4.0)
            expected.index_fill_(dim, index.cpu(), 4.0)

            assert result.data_ptr() == x.data_ptr()
            torch.testing.assert_close(x.cpu(), expected)
            assert_op_runs_on_neuron("aten::index_fill_")

    def test_inplace_int_tensor_value(self, device):
        with track_neuron_ops():
            x = torch.randn(4, 5, device=device)
            expected = x.cpu().clone()
            index = torch.tensor([1, 3], device=device)
            value = torch.tensor(6.0, device=device)

            x.index_fill_(0, index, value)
            expected.index_fill_(0, index.cpu(), value.cpu())

            torch.testing.assert_close(x.cpu(), expected)
            assert_op_runs_on_neuron("aten::index_fill_")

    # -- edge case : an empty index leaves self unchanged ----------------------

    @pytest.mark.parametrize("inplace", [False, True])
    def test_empty_index_returns_self(self, device, inplace):
        # An empty index selects no slice, so index_fill returns self unchanged.
        # The C++ empty-tensor path short-circuits this and does not dispatch the
        # index_fill handler, so check the result and the device, not the op trace.
        x = torch.randn(3, 4, device=device)
        index = torch.tensor([], dtype=torch.int64, device=device)

        cpu_result = x.cpu().index_fill(0, index.cpu(), 9.0)
        neuron_result = x.index_fill_(0, index, 9.0) if inplace else x.index_fill(0, index, 9.0)

        assert neuron_result.device.type == "neuron"
        torch.testing.assert_close(neuron_result.cpu(), cpu_result)

    # -- error contract --------------------------------------------------------

    def test_non_int64_index_raises(self, device):
        x = torch.randn(4, 5, device=device)
        index = torch.tensor([0, 2], dtype=torch.int32, device=device)

        # aten raises IndexError (not RuntimeError) for a non-int64 index.
        with pytest.raises(IndexError, match=r"Expected dtype int64 for index"):
            x.index_fill(0, index, 1.0)

    def test_non_scalar_value_tensor_raises(self, device):
        x = torch.randn(4, 5, device=device)
        index = torch.tensor([0, 2], device=device)
        value = torch.tensor([1.0, 2.0], device=device)

        with pytest.raises(RuntimeError, match=r"0-dimensional value tensor"):
            x.index_fill(0, index, value)

    def test_complex_scalar_into_non_complex_raises(self, device):
        # Neuron has no complex dtypes. The op rejects a complex value with aten's
        # message. This needs the PreprocessArgs complex-Scalar conversion on main.
        x = torch.randn(4, 5, device=device)
        index = torch.tensor([0, 2], device=device)

        with pytest.raises(RuntimeError, match=r"Scalar"):
            x.index_fill(1, index, 1 + 1j)

    # -- autograd : backward is index_fill(grad, dim, index, 0) ----------------

    def test_backward(self, device):
        """Gradient w.r.t. self is grad.index_fill(dim, index, 0). Assert the
        backward op runs on Neuron; a forward-only assert would miss a fallback."""

        def run(dev):
            torch.manual_seed(0)
            x = torch.randn(6, 5, device=dev).detach().requires_grad_(True)
            index = torch.tensor([0, 2, 4], device=dev)
            out = x.index_fill(0, index, 3.5)
            (out * 2).sum().backward()
            return x.grad

        with track_neuron_ops():
            gx = run(device)
            assert_op_runs_on_neuron("aten::index_fill")

        gx_cpu = run(torch.device("cpu"))
        torch.testing.assert_close(gx.cpu(), gx_cpu)

    def test_int_tensor_value_backward(self, device):
        """int_Tensor value is differentiable. The value gradient uses
        aten._unique; confirm the gradients still match CPU under eager."""

        def run(dev):
            torch.manual_seed(0)
            x = torch.randn(6, 5, device=dev).detach().requires_grad_(True)
            value = torch.tensor(2.5, device=dev).detach().requires_grad_(True)
            index = torch.tensor([1, 3], device=dev)
            out = x.index_fill(0, index, value)
            (out * 2).sum().backward()
            return x.grad, value.grad

        with track_neuron_ops():
            gx, gv = run(device)
            # Forward and self grad run index_fill on Neuron. The value grad uses
            # aten._unique (dynamic output), which may fall back; do not assert it.
            assert_op_runs_on_neuron("aten::index_fill")

        gx_cpu, gv_cpu = run(torch.device("cpu"))
        torch.testing.assert_close(gx.cpu(), gx_cpu)
        torch.testing.assert_close(gv.cpu(), gv_cpu)
