#!/usr/bin/env python3
"""
Tests for the dtype autocast context manager.

This module tests the automatic casting of unsupported dtypes (float64, int64)
to supported dtypes (float32, int32) for Neuron hardware execution.
"""

import pytest
import torch

from torch_neuronx.python_ops.dtype_autocast import DtypeAutocastContext, autocast_neuron


class TestDtypeAutocastBasic:
    """Basic functionality tests for dtype autocast."""

    def test_float64_to_float32_casting(self):
        """Test that float64 tensors are cast to float32."""
        with autocast_neuron() as context:
            # Single float64 tensor
            args = (torch.tensor([1.0, 2.0], dtype=torch.float64),)
            kwargs = {}

            new_args, new_kwargs, dtype_info = context.process_args(args, kwargs)

            assert new_args[0].dtype == torch.float32
            assert dtype_info["had_float64"] is True
            assert dtype_info["had_int64"] is False

    def test_int64_to_int32_casting(self):
        """Test that int64 tensors are cast to int32."""
        with autocast_neuron() as context:
            args = (torch.tensor([1, 2, 3], dtype=torch.int64),)
            kwargs = {}

            new_args, new_kwargs, dtype_info = context.process_args(args, kwargs)

            assert new_args[0].dtype == torch.int32
            assert dtype_info["had_int64"] is True
            assert dtype_info["had_float64"] is False

    def test_mixed_dtype_casting(self):
        """Test casting with mixed float64 and int64 inputs."""
        with autocast_neuron() as context:
            args = (
                torch.tensor([1.0, 2.0], dtype=torch.float64),
                torch.tensor([1, 2], dtype=torch.int64),
                torch.tensor([3.0, 4.0], dtype=torch.float32),
                torch.tensor([3, 4], dtype=torch.int32),
            )
            kwargs = {}

            new_args, new_kwargs, dtype_info = context.process_args(args, kwargs)

            assert new_args[0].dtype == torch.float32  # float64 -> float32
            assert new_args[1].dtype == torch.int32  # int64 -> int32
            assert new_args[2].dtype == torch.float32  # unchanged
            assert new_args[3].dtype == torch.int32  # unchanged
            assert dtype_info["had_float64"] is True
            assert dtype_info["had_int64"] is True

    def test_kwargs_casting(self):
        """Test that kwargs are properly cast."""
        with autocast_neuron() as context:
            args = ()
            kwargs = {
                "input": torch.tensor([1.0, 2.0], dtype=torch.float64),
                "weight": torch.tensor([[1.0, 2.0]], dtype=torch.float64),
                "indices": torch.tensor([0, 1], dtype=torch.int64),
            }

            new_args, new_kwargs, dtype_info = context.process_args(args, kwargs)

            assert new_kwargs["input"].dtype == torch.float32
            assert new_kwargs["weight"].dtype == torch.float32
            assert new_kwargs["indices"].dtype == torch.int32
            assert dtype_info["had_float64"] is True
            assert dtype_info["had_int64"] is True

    def test_no_casting_needed(self):
        """Test that tensors with supported dtypes are not changed."""
        with autocast_neuron() as context:
            args = (
                torch.tensor([1.0, 2.0], dtype=torch.float32),
                torch.tensor([1, 2], dtype=torch.int32),
            )
            kwargs = {"scale": 2.0}  # scalar

            new_args, new_kwargs, dtype_info = context.process_args(args, kwargs)

            assert new_args[0].dtype == torch.float32
            assert new_args[1].dtype == torch.int32
            assert new_kwargs["scale"] == 2.0
            assert dtype_info["had_float64"] is False
            assert dtype_info["had_int64"] is False


class TestDtypeRestoration:
    """Tests for restoring dtypes after operation execution."""

    def test_float64_restoration(self):
        """Test that float32 outputs are restored to float64 when inputs were float64."""
        with autocast_neuron() as context:
            # Process float64 input
            args = (torch.tensor([1.0, 2.0], dtype=torch.float64),)
            kwargs = {}
            context.process_args(args, kwargs)

            # Simulate float32 output from operation
            output = torch.tensor([3.0, 4.0], dtype=torch.float32)
            restored = context.restore_dtypes(output)

            assert restored.dtype == torch.float64

    def test_int64_restoration(self):
        """Test that int32 outputs are restored to int64 when inputs were int64."""
        with autocast_neuron() as context:
            # Process int64 input
            args = (torch.tensor([1, 2], dtype=torch.int64),)
            kwargs = {}
            context.process_args(args, kwargs)

            # Simulate int32 output from operation
            output = torch.tensor([3, 4], dtype=torch.int32)
            restored = context.restore_dtypes(output)

            assert restored.dtype == torch.int64

    def test_no_restoration_without_matching_input(self):
        """Test that outputs are not restored without matching input dtypes."""
        with autocast_neuron() as context:
            # Only float32 inputs
            args = (torch.tensor([1.0, 2.0], dtype=torch.float32),)
            kwargs = {}
            context.process_args(args, kwargs)

            # Float32 output should remain float32
            output_f32 = torch.tensor([3.0, 4.0], dtype=torch.float32)
            restored_f32 = context.restore_dtypes(output_f32)
            assert restored_f32.dtype == torch.float32

            # Int32 output should remain int32
            output_i32 = torch.tensor([3, 4], dtype=torch.int32)
            restored_i32 = context.restore_dtypes(output_i32)
            assert restored_i32.dtype == torch.int32

    def test_tuple_output_restoration(self):
        """Test restoration of tuple outputs (e.g., from torch.max)."""
        with autocast_neuron() as context:
            # Mixed inputs
            args = (
                torch.tensor([1.0, 2.0], dtype=torch.float64),
                torch.tensor([0, 1], dtype=torch.int64),
            )
            kwargs = {}
            context.process_args(args, kwargs)

            # Simulate tuple output (values, indices)
            output = (
                torch.tensor([2.0], dtype=torch.float32),
                torch.tensor([1], dtype=torch.int32),
            )
            restored = context.restore_dtypes(output)

            assert isinstance(restored, tuple)
            assert restored[0].dtype == torch.float64  # values restored
            assert restored[1].dtype == torch.int64  # indices restored

    def test_list_output_restoration(self):
        """Test restoration of list outputs."""
        with autocast_neuron() as context:
            args = (torch.tensor([1.0, 2.0], dtype=torch.float64),)
            kwargs = {}
            context.process_args(args, kwargs)

            # List output
            output = [
                torch.tensor([1.0], dtype=torch.float32),
                torch.tensor([2.0], dtype=torch.float32),
            ]
            restored = context.restore_dtypes(output)

            assert isinstance(restored, list)
            assert all(t.dtype == torch.float64 for t in restored)

    def test_mixed_dtype_tuple_restoration(self):
        """Test restoration with mixed dtype outputs."""
        with autocast_neuron() as context:
            args = (torch.tensor([1.0, 2.0], dtype=torch.float64),)
            kwargs = {}
            context.process_args(args, kwargs)

            # Mixed dtype output (float32 and int64)
            output = (
                torch.tensor([2.0], dtype=torch.float32),
                torch.tensor([1], dtype=torch.int64),  # Already int64
            )
            restored = context.restore_dtypes(output)

            assert restored[0].dtype == torch.float64  # Restored
            assert restored[1].dtype == torch.int64  # Unchanged


class TestSelectiveCasting:
    """Tests for selective dtype casting."""

    def test_disable_float64_casting(self):
        """Test disabling float64 casting."""
        with autocast_neuron(cast_float64=False) as context:
            args = (
                torch.tensor([1.0, 2.0], dtype=torch.float64),
                torch.tensor([1, 2], dtype=torch.int64),
            )
            kwargs = {}

            new_args, new_kwargs, dtype_info = context.process_args(args, kwargs)

            assert new_args[0].dtype == torch.float64  # Not cast
            assert new_args[1].dtype == torch.int32  # Cast
            assert dtype_info["had_float64"] is False  # Not tracked
            assert dtype_info["had_int64"] is True

    def test_disable_int64_casting(self):
        """Test disabling int64 casting."""
        with autocast_neuron(cast_int64=False) as context:
            args = (
                torch.tensor([1.0, 2.0], dtype=torch.float64),
                torch.tensor([1, 2], dtype=torch.int64),
            )
            kwargs = {}

            new_args, new_kwargs, dtype_info = context.process_args(args, kwargs)

            assert new_args[0].dtype == torch.float32  # Cast
            assert new_args[1].dtype == torch.int64  # Not cast
            assert dtype_info["had_float64"] is True
            assert dtype_info["had_int64"] is False  # Not tracked

    def test_disable_all_casting(self):
        """Test disabling all casting."""
        with autocast_neuron(cast_float64=False, cast_int64=False) as context:
            args = (
                torch.tensor([1.0, 2.0], dtype=torch.float64),
                torch.tensor([1, 2], dtype=torch.int64),
            )
            kwargs = {}

            new_args, new_kwargs, dtype_info = context.process_args(args, kwargs)

            assert new_args[0].dtype == torch.float64  # Not cast
            assert new_args[1].dtype == torch.int64  # Not cast
            assert dtype_info["had_float64"] is False
            assert dtype_info["had_int64"] is False


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_nested_tuples_and_lists(self):
        """Test handling of nested tuples and lists in arguments."""
        with autocast_neuron() as context:
            args = (
                [
                    torch.tensor([1.0], dtype=torch.float64),
                    torch.tensor([2.0], dtype=torch.float32),
                ],
                (torch.tensor([1], dtype=torch.int64), torch.tensor([2], dtype=torch.int32)),
            )
            kwargs = {}

            new_args, new_kwargs, dtype_info = context.process_args(args, kwargs)

            # Check list
            assert isinstance(new_args[0], list)
            assert new_args[0][0].dtype == torch.float32  # Cast
            assert new_args[0][1].dtype == torch.float32  # Unchanged

            # Check tuple
            assert isinstance(new_args[1], tuple)
            assert new_args[1][0].dtype == torch.int32  # Cast
            assert new_args[1][1].dtype == torch.int32  # Unchanged

            assert dtype_info["had_float64"] is True
            assert dtype_info["had_int64"] is True

    def test_scalar_arguments(self):
        """Test that scalar arguments are handled correctly."""
        with autocast_neuron() as context:
            args = (
                torch.tensor([1.0], dtype=torch.float64),
                2.0,  # scalar float
                3,  # scalar int
                True,  # scalar bool
            )
            kwargs = {"alpha": 0.5, "beta": 2}

            new_args, new_kwargs, dtype_info = context.process_args(args, kwargs)

            assert new_args[0].dtype == torch.float32  # Tensor cast
            assert new_args[1] == 2.0  # Scalar unchanged
            assert new_args[2] == 3  # Scalar unchanged
            assert new_args[3] is True  # Scalar unchanged
            assert new_kwargs["alpha"] == 0.5
            assert new_kwargs["beta"] == 2

    def test_none_values(self):
        """Test handling of None values in arguments."""
        with autocast_neuron() as context:
            args = (
                torch.tensor([1.0], dtype=torch.float64),
                None,
            )
            kwargs = {"out": None, "mask": None}

            new_args, new_kwargs, dtype_info = context.process_args(args, kwargs)

            assert new_args[0].dtype == torch.float32
            assert new_args[1] is None
            assert new_kwargs["out"] is None
            assert new_kwargs["mask"] is None

    def test_empty_tensors(self):
        """Test handling of empty tensors."""
        with autocast_neuron() as context:
            args = (
                torch.empty(0, dtype=torch.float64),
                torch.empty((2, 0, 3), dtype=torch.int64),
            )
            kwargs = {}

            new_args, new_kwargs, dtype_info = context.process_args(args, kwargs)

            assert new_args[0].dtype == torch.float32
            assert new_args[0].numel() == 0
            assert new_args[1].dtype == torch.int32
            assert new_args[1].shape == (2, 0, 3)

    def test_output_none_restoration(self):
        """Test that None outputs are handled correctly."""
        with autocast_neuron() as context:
            args = (torch.tensor([1.0], dtype=torch.float64),)
            kwargs = {}
            context.process_args(args, kwargs)

            assert context.restore_dtypes(None) is None

    def test_preserve_tensor_properties(self):
        """Test that tensor properties other than dtype are preserved."""
        with autocast_neuron() as context:
            # Create tensor with specific properties
            tensor = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
            args = (tensor,)
            kwargs = {}

            new_args, new_kwargs, dtype_info = context.process_args(args, kwargs)

            # Check dtype is cast but other properties preserved
            assert new_args[0].dtype == torch.float32
            assert new_args[0].requires_grad is True
            assert new_args[0].shape == tensor.shape
            assert torch.allclose(new_args[0], tensor.to(torch.float32))


class TestContextManagerSemantics:
    """Tests for context manager behavior."""

    def test_context_manager_basic(self):
        """Test basic context manager usage."""
        with autocast_neuron() as context:
            assert isinstance(context, DtypeAutocastContext)
            assert context.cast_float64 is True
            assert context.cast_int64 is True

    def test_context_manager_with_params(self):
        """Test context manager with parameters."""
        with autocast_neuron(cast_float64=False, cast_int64=True) as context:
            assert context.cast_float64 is False
            assert context.cast_int64 is True

    def test_nested_contexts(self):
        """Test that nested contexts work independently."""
        with autocast_neuron() as context1:
            args1 = (torch.tensor([1.0], dtype=torch.float64),)
            new_args1, _, info1 = context1.process_args(args1, {})
            assert info1["had_float64"] is True

            with autocast_neuron(cast_float64=False) as context2:
                args2 = (torch.tensor([2.0], dtype=torch.float64),)
                new_args2, _, info2 = context2.process_args(args2, {})
                assert new_args2[0].dtype == torch.float64  # Not cast in context2

            # context1 should still work after context2 exits
            assert new_args1[0].dtype == torch.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
