"""Test that add operation is properly registered with PyTorch dispatcher."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)


class TestAddRegistration:
    """Test add operation registration and functionality."""

    def test_add_runs_on_neuron(self):
        """Test that add runs on Neuron without CPU fallback"""
        with track_neuron_ops():
            a = torch.tensor([1.0, 2.0, 3.0], device="neuron")
            b = torch.tensor([4.0, 5.0, 6.0], device="neuron")
            result = torch.add(a, b)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::add")

    def test_add_inplace_runs_on_neuron(self):
        """Test that add_ runs on Neuron without CPU fallback"""
        with track_neuron_ops():
            a = torch.tensor([1.0, 2.0, 3.0], device="neuron")
            b = torch.tensor([4.0, 5.0, 6.0], device="neuron")
            original_data_ptr = a.data_ptr()
            result = a.add_(b)
            assert result is a
            assert result.data_ptr() == original_data_ptr
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("add")

    def test_add_inplace_non_contiguous_output(self):
        """Test in-place add operations on tensor slices"""
        m1 = torch.ones(10, 10, device="neuron")
        m1[:, 3].add_(2)
        expected = torch.ones(10, 10)
        expected[:, 3].add_(2)
        torch.testing.assert_close(m1.cpu(), expected)

    def test_add_inplace_wider_int_operand_no_promotion(self):
        """int32.add_(int64) keeps self's shape+dtype — in-place does not promote."""
        with track_neuron_ops():
            a = torch.ones([4, 4, 4], dtype=torch.int32, device="neuron")
            b = torch.ones([4, 4, 4], dtype=torch.int64, device="neuron")
            a.add_(b)

            expected = torch.ones([4, 4, 4], dtype=torch.int32)
            expected.add_(torch.ones([4, 4, 4], dtype=torch.int64))

            assert a.device.type == "neuron"
            assert a.dtype == torch.int32
            assert tuple(a.shape) == (4, 4, 4)
            torch.testing.assert_close(a.cpu(), expected)

    @assert_raises(RuntimeError, match="can't be cast")
    def test_add_inplace_float_into_int_raises(self):
        """int.add_(float) is rejected — a float result can't be cast to int."""
        a = torch.ones([4], dtype=torch.int32, device="neuron")
        a.add_(1.5)

    def test_add_operator_syntax(self):
        """Test that a + b syntax works with neuron tensors."""
        # Create neuron tensors
        a = torch.ones(16, 16, dtype=torch.float32).to("neuron")
        b = torch.ones(16, 16, dtype=torch.float32).to("neuron")

        # Use + operator (calls add.Tensor)
        c = a + b

        # Verify result
        assert c.device.type == "neuron"
        assert c.shape == (16, 16)
        expected = torch.ones(16, 16, dtype=torch.float32) * 2
        torch.testing.assert_close(c.cpu(), expected)

    def test_add_function_syntax(self):
        """Test that torch.add works with neuron tensors."""
        # Create neuron tensors
        a = torch.ones(16, 16, dtype=torch.float32).to("neuron")
        b = torch.ones(16, 16, dtype=torch.float32).to("neuron")

        # Use torch.add function
        c = torch.add(a, b)

        # Verify result
        assert c.device.type == "neuron"
        assert c.shape == (16, 16)
        expected = torch.ones(16, 16, dtype=torch.float32) * 2
        torch.testing.assert_close(c.cpu(), expected)

    def test_add_with_alpha(self):
        """Test add with alpha parameter."""
        # Create neuron tensors
        a = torch.ones(16, 16, dtype=torch.float32).to("neuron")
        b = torch.ones(16, 16, dtype=torch.float32).to("neuron")

        # Use torch.add with alpha
        c = torch.add(a, b, alpha=2)

        # Verify result: a + alpha * b = 1 + 2 * 1 = 3
        assert c.device.type == "neuron"
        expected = torch.ones(16, 16, dtype=torch.float32) * 3
        torch.testing.assert_close(c.cpu(), expected)

    def test_add_out(self):
        """Test add.out operation."""
        # Create neuron tensors
        a = torch.ones(16, 16, dtype=torch.float32).to("neuron")
        b = torch.ones(16, 16, dtype=torch.float32).to("neuron")
        out = torch.empty(16, 16, dtype=torch.float32).to("neuron")

        # Use torch.add with out parameter
        result = torch.add(a, b, out=out)

        # Verify result and that out was modified in-place
        assert result is out
        assert out.device.type == "neuron"
        expected = torch.ones(16, 16, dtype=torch.float32) * 2
        torch.testing.assert_close(out.cpu(), expected)

    def test_add_inplace(self):
        """Test in-place add operation."""
        # Create neuron tensors
        a = torch.ones(16, 16, dtype=torch.float32).to("neuron")
        b = torch.ones(16, 16, dtype=torch.float32).to("neuron")

        # Store original tensor id
        original_id = id(a)

        # Use += operator (calls add_ which uses add.out internally)
        a += b

        # Verify it was in-place
        assert id(a) == original_id
        assert a.device.type == "neuron"
        expected = torch.ones(16, 16, dtype=torch.float32) * 2
        torch.testing.assert_close(a.cpu(), expected)

    def test_add_mixed_devices(self):
        """Test that mixing CPU and neuron tensors promotes cpu tensor to device."""
        # One tensor on neuron, one on CPU
        a = torch.ones(16, 16, dtype=torch.float32).to("neuron")
        b = torch.ones(16, 16, dtype=torch.float32)  # CPU

        # Should raise RuntimeError for non-scalar cross-device operation
        with pytest.raises(RuntimeError, match="is on cpu device, expected neuron"):
            _ = a + b

    def test_add_small_tensor(self):
        """Test that small tensors work with XLA implementation."""
        # Create small neuron tensors (< 256 elements)
        a = torch.ones(4, 4, dtype=torch.float32).to("neuron")
        b = torch.ones(4, 4, dtype=torch.float32).to("neuron")

        # Since we removed can_handle, XLA will handle all cases
        c = a + b

        # Verify result
        assert c.device.type == "neuron"
        expected = torch.ones(4, 4, dtype=torch.float32) * 2
        torch.testing.assert_close(c.cpu(), expected)

    def test_add_first_operand_python_scalar(self):
        """Test addition with scalar"""
        # Create CPU tensors
        a_cpu = torch.tensor([4.0, 6.0, 8.0, 10.0])
        b_cpu = 2.0
        expected = b_cpu + a_cpu

        # Move to Neuron device
        a_neuron = a_cpu.to("neuron")

        # Perform addition
        c = b_cpu + a_neuron

        torch.testing.assert_close(c.cpu(), expected)

    def test_add_second_operand_python_scalar(self):
        """Test addition with scalar"""
        # Create CPU tensors
        a_cpu = torch.tensor([4.0, 6.0, 8.0, 10.0])
        b_cpu = 2.0
        expected = a_cpu + b_cpu

        # Move to Neuron device
        a_neuron = a_cpu.to("neuron")

        # Perform addition
        c = a_neuron + b_cpu

        torch.testing.assert_close(c.cpu(), expected)

    def test_add_type_promotions(self):
        """Test addition with scalar-like tensor"""
        # Create CPU tensors
        a_cpu = torch.tensor([4.0, 6.0, 8.0, 10.0], dtype=torch.float32)
        b_cpu = torch.tensor(2, dtype=torch.int32)
        expected = a_cpu + b_cpu

        # Move to Neuron device
        a_neuron = a_cpu.to("neuron")
        b_neuron = b_cpu.to("neuron")

        # Perform addition
        c = a_neuron + b_neuron

        torch.testing.assert_close(c.cpu(), expected)

    def test_add_with_int64(self):
        """Test addition with int64 tensor"""
        # Create CPU tensors
        a_cpu = torch.tensor([4, 6, 10], dtype=torch.int64)
        b_cpu = 0
        expected = a_cpu + b_cpu

        # Move to Neuron device
        a_neuron = a_cpu.to("neuron")

        # Perform addition
        c = a_neuron + b_cpu

        torch.testing.assert_close(c.cpu(), expected)

    @pytest.mark.parametrize("alpha", [1, 2, -1, 0])
    def test_add_bool_with_alpha(self, alpha):
        """torch.add(bool, bool, alpha=N) should match CPU dtype and values."""
        x_cpu = torch.tensor([False, True, False, True])
        y_cpu = torch.tensor([True, False, True, False])
        x = x_cpu.to("neuron")
        y = y_cpu.to("neuron")

        result = torch.add(x, y, alpha=alpha)
        expected = torch.add(x_cpu, y_cpu, alpha=alpha)

        assert (
            result.dtype == expected.dtype
        ), f"alpha={alpha}: expected {expected.dtype} but got {result.dtype}"
        torch.testing.assert_close(result.cpu(), expected)


class TestAddDtypePromotion:
    """Test that Tensor.add_ raises for int+float inplace, matching CPU."""

    @pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.uint8])
    def test_add_inplace_int_float_scalar_raises(self, dtype):
        """Tensor.add_(int_tensor, float_scalar) raises matching CPU."""
        t = torch.tensor([-3, -1, 0, 1, 5], dtype=dtype, device="neuron")
        with pytest.raises(RuntimeError, match="can't be cast"):
            t.add_(0.883)

    @pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.uint8])
    def test_add_inplace_int_int_scalar_succeeds(self, dtype):
        """Tensor.add_(int_tensor, int_scalar) succeeds — no dtype promotion."""
        t = torch.tensor([1, 2, 3], dtype=dtype, device="neuron")
        t.add_(5)
        expected = torch.tensor([6, 7, 8], dtype=dtype)
        torch.testing.assert_close(t.cpu(), expected)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
    def test_add_inplace_float_float_scalar_succeeds(self, dtype):
        """Tensor.add_(float_tensor, float_scalar) succeeds — same dtype class."""
        t = torch.tensor([1.0, 2.0, 3.0], dtype=dtype, device="neuron")
        t.add_(0.5)
        expected = torch.tensor([1.5, 2.5, 3.5], dtype=dtype)
        torch.testing.assert_close(t.cpu(), expected)

    @pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32])
    def test_add_outplace_int_float_scalar_does_not_raise(self, dtype):
        """Outplace torch.add(int, float_scalar) succeeds (no raise for outplace)."""
        t = torch.tensor([1, 2, 3], dtype=dtype, device="neuron")
        torch.add(t, 0.5)  # should not raise

    @pytest.mark.parametrize(
        "dtype", [torch.int32, torch.bool, torch.float32, torch.bfloat16, torch.float16]
    )
    def test_add_float_scalar_weak_promotion(self, dtype):
        """torch.add(tensor, <python float>) follows weak-scalar typing: a python
        float never upcasts a tensor operand. result_type is float32 for an int/bool
        tensor and the float tensor's own dtype for a float tensor -- never float64
        (which Neuron drops). Fences the wrapped-float64-scalar handling in
        DispatchImpl (csrc/core/dispatch/NeuronDispatcher.cpp)."""
        x = torch.tensor([1, 0, 1], dtype=dtype, device="neuron")
        got = (x + 1.0).dtype
        exp = torch.result_type(x, 1.0)
        assert got == exp, f"{dtype} + 1.0 -> {got}, expected {exp}"

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_add_float_scalar_0dim_weak_promotion(self, dtype):
        """0-dim float tensor + python float keeps the tensor's dtype (not
        float64). Neuron only anchors output dtype on dim>0 float operands, so the
        weak-scalar rewrite in DispatchImpl must still fire for a 0-dim float
        tensor. Regression fence for the merge-queue dtype-inference fix."""
        x = torch.tensor(2.0, dtype=dtype, device="neuron")  # 0-dim
        got = (x + 1.0).dtype
        exp = torch.result_type(x, 1.0)
        assert got == exp == dtype, f"0-dim {dtype} + 1.0 -> {got}, expected {dtype}"

    def test_add_half_scalar_out_of_range_no_overflow(self):
        """A python float outside half's range must not be narrowed to half before
        the op (which would overflow to inf); computed in float opmath the result
        is finite. Fences 'do not narrow the scalar to a dim>0 float tensor's
        dtype'."""
        x = torch.tensor([60000.0], dtype=torch.half, device="neuron")
        r = torch.add(x, -70000.0).cpu()
        assert not (r.isnan().any() or r.isinf().any()), f"half + (-70000.0) -> {r}"

    @pytest.mark.parametrize("dtype", [torch.uint16, torch.uint32, torch.uint64])
    def test_dispatch_no_promotetypes_crash_on_uint(self, dtype):
        """Ops over uint16/uint32/uint64 must not trip 'Unsupported ScalarType'
        from the weak-scalar block, which feeds only standard floats to
        c10::promoteTypes."""
        x = torch.tensor([5, 5, 5], dtype=dtype, device="neuron")
        torch.testing.assert_close(x, x.clone(), rtol=0, atol=0)
