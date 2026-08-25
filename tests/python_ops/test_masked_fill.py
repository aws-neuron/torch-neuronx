"""Tests for masked_fill operation.

Covers both value flavours of the op:

* ``Scalar`` — ``value`` is a Python number (``aten::masked_fill.Scalar``,
  ``aten::masked_fill.Scalar_out``, ``aten::masked_fill_.Scalar``).
* ``Tensor`` — ``value`` is a 0-dim tensor (``aten::masked_fill.Tensor``,
  ``aten::masked_fill.Tensor_out``, ``aten::masked_fill_.Tensor``).

The result always keeps ``self.dtype``: masked_fill does *not* type-promote
against ``value``, so a float ``value`` into an integer ``self`` truncates.
"""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestMaskedFill:
    """Test suite for masked_fill operation."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 5)])
    def test_masked_fill_basic_shapes(self, device, shape):
        """Test masked_fill with various tensor shapes."""
        with track_neuron_ops():
            x = torch.randint(0, 10, shape, device=device)
            mask = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
            value = 42.1
            neuron_result = x.masked_fill(mask, value)
            cpu_result = x.cpu().masked_fill(mask.cpu(), value)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_fill.Scalar")

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.bool,
            torch.float32,
            torch.float64,
            torch.float16,
            torch.bfloat16,
            torch.int32,
            torch.int64,
        ],
    )
    @pytest.mark.parametrize("value", [42.1, -42.1, 0, 4, -4])
    def test_masked_fill_different_dtypes(self, device, dtype, value):
        """Test masked_fill with various tensor shapes."""
        shape = (3, 4)
        with track_neuron_ops():
            if dtype == torch.bool:
                x = torch.randint(0, 2, shape, dtype=dtype, device=device)
            else:
                x = torch.randint(0, 10, shape, dtype=dtype, device=device)
            mask = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
            neuron_result = x.masked_fill(mask, value)
            cpu_result = x.cpu().masked_fill(mask.cpu(), value)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_fill.Scalar")

    @pytest.mark.parametrize(
        "value,equal_nan",
        [
            (float("inf"), False),
            (float("-inf"), False),
            (float("nan"), True),
        ],
    )
    def test_masked_fill_special_values(self, device, value, equal_nan):
        """Test masked_fill with special values (inf, -inf, nan)."""
        with track_neuron_ops():
            x = torch.randn(3, device=device)
            mask = torch.tensor([True, False, True], device=device)

            neuron_result = x.masked_fill(mask, value)
            cpu_result = x.cpu().masked_fill(mask.cpu(), value)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result, equal_nan=equal_nan)
            assert_op_runs_on_neuron("aten::masked_fill.Scalar")

    @pytest.mark.parametrize(
        "tensor_shape,mask_shape",
        [
            ((4, 3), (1, 3)),  # Broadcast first dimension
            ((4, 3), (4, 1)),  # Broadcast second dimension
            ((2, 3, 4), (1, 3, 4)),  # Broadcast first dimension
            ((2, 3, 4), (2, 1, 4)),  # Broadcast middle dimension
            ((2, 3, 4), (2, 3, 1)),  # Broadcast last dimension
            ((2, 3, 4), (1, 1, 4)),  # Broadcast multiple dimensions
            ((2, 3, 4), (3, 4)),  # Broadcast with fewer dimensions
            ((2, 3, 4), (4,)),  # Broadcast with single dimension
        ],
    )
    def test_masked_fill_broadcast(self, device, tensor_shape, mask_shape):
        """Test masked_fill with broadcasting masks."""
        with track_neuron_ops():
            x = torch.randn(tensor_shape, device=device)
            mask = torch.randint(0, 2, mask_shape, dtype=torch.bool, device=device)
            value = 42.1

            neuron_result = x.masked_fill(mask, value)
            cpu_result = x.cpu().masked_fill(mask.cpu(), value)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_fill.Scalar")


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestMaskedFillTensor:
    """Test suite for the ``Tensor``-value overloads of masked_fill.

    ``aten::masked_fill_.Tensor`` is the only kernel PyTorch dispatches to the
    backend for this flavour: ``masked_fill.Tensor`` and
    ``masked_fill.Tensor_out`` are CompositeExplicitAutograd and decompose into
    it. Each test therefore asserts on the *exact* overload name, so a silent
    CPU fallback (which still produces correct numbers) fails the test.
    """

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 5)])
    def test_masked_fill_tensor_basic_shapes(self, device, shape):
        """Test masked_fill with a 0-dim tensor value across tensor shapes."""
        with track_neuron_ops():
            x = torch.randn(shape, device=device)
            mask = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
            value = torch.tensor(42.5, device=device)

            neuron_result = x.masked_fill(mask, value)
            cpu_result = x.cpu().masked_fill(mask.cpu(), value.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_fill.Tensor")

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.bool,
            torch.float32,
            torch.float64,
            torch.float16,
            torch.bfloat16,
            torch.int32,
            torch.int64,
        ],
    )
    def test_masked_fill_tensor_dtypes(self, device, dtype):
        """Test masked_fill with a same-dtype 0-dim tensor value, per dtype."""
        shape = (3, 4)
        with track_neuron_ops():
            hi = 2 if dtype == torch.bool else 10
            x = torch.randint(0, hi, shape, dtype=dtype, device=device)
            mask = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
            value = torch.tensor(1 if dtype == torch.bool else 4, dtype=dtype, device=device)

            neuron_result = x.masked_fill(mask, value)
            cpu_result = x.cpu().masked_fill(mask.cpu(), value.cpu())

            assert neuron_result.device.type == "neuron"
            assert neuron_result.dtype == cpu_result.dtype
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_fill.Tensor")

    @pytest.mark.parametrize("value_dtype", [torch.float32, torch.float64, torch.int64, torch.bool])
    def test_masked_fill_tensor_value_dtype_is_not_promoted(self, device, value_dtype):
        """The result keeps ``self.dtype`` regardless of the value's dtype.

        masked_fill does not type-promote against ``value`` — unlike most binary
        ops — so this guards against a lowering that runs ``result_type`` and
        widens the output.
        """
        with track_neuron_ops():
            x = torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3)
            mask = torch.tensor([[True, False, True], [False, True, False]], device=device)
            value = torch.tensor(1, dtype=value_dtype, device=device)

            neuron_result = x.masked_fill(mask, value)
            cpu_result = x.cpu().masked_fill(mask.cpu(), value.cpu())

            assert neuron_result.dtype == torch.float32
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_fill.Tensor")

    def test_masked_fill_tensor_float_value_into_int_truncates(self, device):
        """A float ``value`` filled into an integer ``self`` truncates, as on CPU."""
        with track_neuron_ops():
            x = torch.arange(6, dtype=torch.int32, device=device).reshape(2, 3)
            mask = torch.tensor([[True, False, True], [False, True, False]], device=device)
            value = torch.tensor(2.7, device=device)

            neuron_result = x.masked_fill(mask, value)
            cpu_result = x.cpu().masked_fill(mask.cpu(), value.cpu())

            assert neuron_result.dtype == torch.int32
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_fill.Tensor")

    @pytest.mark.parametrize(
        "value,equal_nan",
        [
            (float("inf"), False),
            (float("-inf"), False),
            (float("nan"), True),
        ],
    )
    def test_masked_fill_tensor_special_values(self, device, value, equal_nan):
        """Test masked_fill with special 0-dim tensor values (inf, -inf, nan)."""
        with track_neuron_ops():
            x = torch.randn(3, device=device)
            mask = torch.tensor([True, False, True], device=device)
            value_t = torch.tensor(value, device=device)

            neuron_result = x.masked_fill(mask, value_t)
            cpu_result = x.cpu().masked_fill(mask.cpu(), value_t.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result, equal_nan=equal_nan)
            assert_op_runs_on_neuron("aten::masked_fill.Tensor")

    @pytest.mark.parametrize(
        "tensor_shape,mask_shape",
        [
            ((4, 3), (1, 3)),  # Broadcast first dimension
            ((4, 3), (4, 1)),  # Broadcast second dimension
            ((2, 3, 4), (2, 1, 4)),  # Broadcast middle dimension
            ((2, 3, 4), (1, 1, 4)),  # Broadcast multiple dimensions
            ((2, 3, 4), (4,)),  # Broadcast with single dimension
            ((3,), (2, 3)),  # Mask has more dims than self: output grows
        ],
    )
    def test_masked_fill_tensor_broadcast(self, device, tensor_shape, mask_shape):
        """Test masked_fill with broadcasting masks and a tensor value."""
        with track_neuron_ops():
            x = torch.randn(tensor_shape, device=device)
            mask = torch.randint(0, 2, mask_shape, dtype=torch.bool, device=device)
            value = torch.tensor(42.5, device=device)

            neuron_result = x.masked_fill(mask, value)
            cpu_result = x.cpu().masked_fill(mask.cpu(), value.cpu())

            assert neuron_result.shape == cpu_result.shape
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_fill.Tensor")

    def test_masked_fill_tensor_all_true_and_all_false_masks(self, device):
        """An all-True mask replaces everything; an all-False mask is a no-op."""
        with track_neuron_ops():
            x = torch.randn(3, 4, device=device)
            value = torch.tensor(-1.5, device=device)

            all_true = x.masked_fill(torch.ones(3, 4, dtype=torch.bool, device=device), value)
            all_false = x.masked_fill(torch.zeros(3, 4, dtype=torch.bool, device=device), value)

            torch.testing.assert_close(all_true.cpu(), torch.full((3, 4), -1.5))
            torch.testing.assert_close(all_false.cpu(), x.cpu())
            assert_op_runs_on_neuron("aten::masked_fill.Tensor")

    def test_masked_fill_tensor_empty(self, device):
        """Test masked_fill with an empty tensor (numel=0)."""
        with track_neuron_ops():
            x = torch.empty(0, device=device)
            mask = torch.empty(0, dtype=torch.bool, device=device)
            value = torch.tensor(1.0, device=device)

            neuron_result = x.masked_fill(mask, value)
            assert neuron_result.numel() == 0
            assert neuron_result.shape == torch.Size([0])
            assert_op_runs_on_neuron("aten::masked_fill.Tensor")

    @pytest.mark.parametrize("mask_value", [True, False])
    def test_masked_fill_tensor_zero_dim_self(self, device, mask_value):
        """Test masked_fill on a 0-dim self with a 0-dim mask and value."""
        with track_neuron_ops():
            x = torch.tensor(5.0, device=device)
            mask = torch.tensor(mask_value, device=device)
            value = torch.tensor(1.0, device=device)

            neuron_result = x.masked_fill(mask, value)
            cpu_result = x.cpu().masked_fill(mask.cpu(), value.cpu())

            assert neuron_result.shape == torch.Size([])
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_fill.Tensor")

    def test_masked_fill_tensor_single_element(self, device):
        """Test masked_fill with a single-element tensor."""
        with track_neuron_ops():
            x = torch.tensor([3.0], device=device)
            mask = torch.tensor([True], device=device)
            value = torch.tensor(-1.0, device=device)

            neuron_result = x.masked_fill(mask, value)
            cpu_result = x.cpu().masked_fill(mask.cpu(), value.cpu())

            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_fill.Tensor")

    def test_masked_fill_tensor_does_not_mutate_self(self, device):
        """The non-in-place overload must leave ``self`` untouched."""
        with track_neuron_ops():
            x = torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3)
            before = x.cpu().clone()
            mask = torch.tensor([[True, False, True], [False, True, False]], device=device)

            x.masked_fill(mask, torch.tensor(99.0, device=device))

            torch.testing.assert_close(x.cpu(), before)
            assert_op_runs_on_neuron("aten::masked_fill.Tensor")

    # ── in-place: aten::masked_fill_.Tensor ──────────────────────────────────

    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
    def test_masked_fill__tensor_inplace(self, device, shape):
        """Test in-place masked_fill_ with a 0-dim tensor value."""
        with track_neuron_ops():
            x = torch.randn(shape, device=device)
            expected = x.cpu().clone()
            mask = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
            value = torch.tensor(42.5, device=device)

            result = x.masked_fill_(mask, value)
            expected.masked_fill_(mask.cpu(), value.cpu())

            assert result.data_ptr() == x.data_ptr()
            torch.testing.assert_close(x.cpu(), expected)
            assert_op_runs_on_neuron("aten::masked_fill_.Tensor")

    @pytest.mark.parametrize(
        "dtype",
        [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64, torch.bool],
    )
    def test_masked_fill__tensor_inplace_dtypes(self, device, dtype):
        """Test in-place masked_fill_ with a tensor value, per dtype."""
        shape = (3, 4)
        with track_neuron_ops():
            hi = 2 if dtype == torch.bool else 10
            x = torch.randint(0, hi, shape, dtype=dtype, device=device)
            expected = x.cpu().clone()
            mask = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
            value = torch.tensor(1 if dtype == torch.bool else 4, dtype=dtype, device=device)

            x.masked_fill_(mask, value)
            expected.masked_fill_(mask.cpu(), value.cpu())

            torch.testing.assert_close(x.cpu(), expected)
            assert_op_runs_on_neuron("aten::masked_fill_.Tensor")

    @pytest.mark.parametrize("mask_shape", [(1, 3), (3,), (2, 3)])
    def test_masked_fill__tensor_inplace_broadcast_mask(self, device, mask_shape):
        """In-place masked_fill_ broadcasts the mask up to ``self``'s shape."""
        with track_neuron_ops():
            x = torch.randn(2, 3, device=device)
            expected = x.cpu().clone()
            mask = torch.randint(0, 2, mask_shape, dtype=torch.bool, device=device)
            value = torch.tensor(7.0, device=device)

            x.masked_fill_(mask, value)
            expected.masked_fill_(mask.cpu(), value.cpu())

            torch.testing.assert_close(x.cpu(), expected)
            assert_op_runs_on_neuron("aten::masked_fill_.Tensor")

    def test_masked_fill__tensor_inplace_expanding_mask_raises(self, device):
        """An in-place mask that would expand ``self`` must raise, as on CPU."""
        x = torch.zeros(1, 3, device=device)
        mask = torch.ones(2, 3, dtype=torch.bool, device=device)
        value = torch.tensor(1.0, device=device)

        with pytest.raises(RuntimeError, match=r"doesn't match the broadcast shape"):
            x.masked_fill_(mask, value)

    # ── out=: aten::masked_fill.Tensor_out / .Scalar_out ─────────────────────

    def test_masked_fill_tensor_out_variant(self, device):
        """The .Tensor_out overload writes into ``out`` and aliases it."""
        with track_neuron_ops():
            x = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
            self_before = x.cpu().clone()
            mask = torch.tensor([[True, False, True, False]] * 3, device=device)
            value = torch.tensor(7.0, device=device)
            out = torch.empty(3, 4, device=device)

            result = torch.ops.aten.masked_fill.Tensor_out(x, mask, value, out=out)
            expected = x.cpu().masked_fill(mask.cpu(), value.cpu())

            assert result.data_ptr() == out.data_ptr()
            torch.testing.assert_close(out.cpu(), expected)
            # out= must not corrupt self — PyTorch's fallback path can alias them.
            torch.testing.assert_close(x.cpu(), self_before)
            assert_op_runs_on_neuron("aten::masked_fill.Tensor_out")

    def test_masked_fill_tensor_out_noncontiguous(self, device):
        """The .Tensor_out overload handles a non-contiguous ``out`` buffer."""
        with track_neuron_ops():
            x = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
            mask = torch.tensor([[True, False, True, False]] * 3, device=device)
            value = torch.tensor(7.0, device=device)
            buffer = torch.zeros(3, 8, device=device)
            out = buffer[:, ::2]
            assert not out.is_contiguous()

            result = torch.ops.aten.masked_fill.Tensor_out(x, mask, value, out=out)
            expected = x.cpu().masked_fill(mask.cpu(), value.cpu())

            assert result.data_ptr() == out.data_ptr()
            torch.testing.assert_close(out.cpu(), expected)
            # The interleaved columns of the backing buffer must stay zero.
            torch.testing.assert_close(buffer[:, 1::2].cpu(), torch.zeros(3, 4))
            assert_op_runs_on_neuron("aten::masked_fill.Tensor_out")

    def test_masked_fill_tensor_out_broadcast_mask(self, device):
        """.Tensor_out sizes ``out`` from the broadcast shape, not from self."""
        with track_neuron_ops():
            x = torch.randn(3, device=device)
            mask = torch.randint(0, 2, (2, 3), dtype=torch.bool, device=device)
            value = torch.tensor(7.0, device=device)
            out = torch.empty(2, 3, device=device)

            torch.ops.aten.masked_fill.Tensor_out(x, mask, value, out=out)
            expected = x.cpu().masked_fill(mask.cpu(), value.cpu())

            assert out.shape == torch.Size([2, 3])
            torch.testing.assert_close(out.cpu(), expected)
            assert_op_runs_on_neuron("aten::masked_fill.Tensor_out")

    def test_masked_fill_scalar_out_variant(self, device):
        """The .Scalar_out overload writes into ``out`` and aliases it."""
        with track_neuron_ops():
            x = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
            self_before = x.cpu().clone()
            mask = torch.tensor([[True, False, True, False]] * 3, device=device)
            out = torch.empty(3, 4, device=device)

            result = torch.ops.aten.masked_fill.Scalar_out(x, mask, 7.0, out=out)
            expected = x.cpu().masked_fill(mask.cpu(), 7.0)

            assert result.data_ptr() == out.data_ptr()
            torch.testing.assert_close(out.cpu(), expected)
            torch.testing.assert_close(x.cpu(), self_before)
            assert_op_runs_on_neuron("aten::masked_fill.Scalar_out")

    # ── error contracts ──────────────────────────────────────────────────────

    @pytest.mark.parametrize("value_shape", [(1,), (3,), (1, 1)])
    def test_masked_fill_tensor_nonzero_dim_value_raises(self, device, value_shape):
        """A value tensor with more than 0 dimensions must raise, as on CPU."""
        x = torch.randn(2, 3, device=device)
        mask = torch.ones(2, 3, dtype=torch.bool, device=device)
        value = torch.ones(value_shape, device=device)

        with pytest.raises(RuntimeError, match=r"0-dimensional value tensor"):
            x.masked_fill(mask, value)

    @pytest.mark.parametrize("value_shape", [(1,), (3,)])
    def test_masked_fill__tensor_nonzero_dim_value_raises(self, device, value_shape):
        """In-place masked_fill_ also rejects a non-0-dim value tensor."""
        x = torch.randn(2, 3, device=device)
        mask = torch.ones(2, 3, dtype=torch.bool, device=device)
        value = torch.ones(value_shape, device=device)

        with pytest.raises(RuntimeError, match=r"0-dimensional value tensor"):
            x.masked_fill_(mask, value)

    @pytest.mark.parametrize("mask_dtype", [torch.uint8, torch.int32, torch.float32])
    def test_masked_fill_tensor_non_bool_mask_raises(self, device, mask_dtype):
        """A non-boolean mask must raise, as on CPU."""
        x = torch.randn(2, 3, device=device)
        mask = torch.ones(2, 3, dtype=mask_dtype, device=device)
        value = torch.tensor(1.0, device=device)

        with pytest.raises(RuntimeError, match=r"only supports boolean masks"):
            x.masked_fill(mask, value)

    # ── autograd ─────────────────────────────────────────────────────────────

    def test_masked_fill_tensor_backward(self, device):
        """Gradients flow to both ``self`` and ``value``, matching CPU."""
        mask_cpu = torch.tensor([[True, False, True], [False, True, False]])

        def run(dev):
            x = torch.arange(6, dtype=torch.float32, device=dev).reshape(2, 3)
            x = x.detach().requires_grad_(True)
            v = torch.tensor(3.0, device=dev, requires_grad=True)
            out = x.masked_fill(mask_cpu.to(dev), v)
            (out * 2).sum().backward()
            return x.grad, v.grad

        with track_neuron_ops():
            gx, gv = run(device)
            assert_op_runs_on_neuron("aten::masked_fill.Tensor")

        gx_cpu, gv_cpu = run(torch.device("cpu"))
        torch.testing.assert_close(gx.cpu(), gx_cpu)
        torch.testing.assert_close(gv.cpu(), gv_cpu)

    # ── torch.compile parity ─────────────────────────────────────────────────

    def test_masked_fill_tensor_compile(self, device):
        """masked_fill with a tensor value works under torch.compile.

        Eager (``@register_aten``) and torch.compile (StableHLO conversion of
        the decomposed graph) are different lowering paths; this guards parity.
        """
        x = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
        mask = torch.tensor([[True, False, True, False]] * 3, device=device)
        value = torch.tensor(7.0, device=device)

        def fn(a, m, v):
            return a.masked_fill(m, v)

        compiled = torch.compile(fn, backend="neuron", fullgraph=True)
        neuron_result = compiled(x, mask, value)
        cpu_result = fn(x.cpu(), mask.cpu(), value.cpu())
        torch.testing.assert_close(neuron_result.cpu(), cpu_result)

    def test_masked_fill__tensor_inplace_compile(self, device):
        """In-place masked_fill_ with a tensor value works under torch.compile."""
        x = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
        mask = torch.tensor([[True, False, True, False]] * 3, device=device)
        value = torch.tensor(7.0, device=device)

        def fn(a, m, v):
            b = a.clone()
            b.masked_fill_(m, v)
            return b

        compiled = torch.compile(fn, backend="neuron", fullgraph=True)
        neuron_result = compiled(x, mask, value)
        cpu_result = fn(x.cpu(), mask.cpu(), value.cpu())
        torch.testing.assert_close(neuron_result.cpu(), cpu_result)
