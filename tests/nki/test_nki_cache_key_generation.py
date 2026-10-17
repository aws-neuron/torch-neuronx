from unittest.mock import Mock, patch

import pytest
import torch

from torch_neuronx.python_ops.torch_mlir.nki_op_impl import NKITorchMlirOpImpl


class TestNKICacheKey:
    """Test cache key generation for NKI kernels with different static arguments."""

    def setup_method(self):
        self.op_impl = NKITorchMlirOpImpl()

    def test_cache_key_different_kernel_idx(self):
        """Test that different kernel_idx values produce different cache keys."""
        inputs = (torch.randn(4, 4),)

        kwargs1 = {
            "kernel_idx": 0,
            "constant_args_key": 123,
        }

        kwargs2 = {
            "kernel_idx": 1,
            "constant_args_key": 123,
        }

        key1 = self.op_impl.kernel._generate_unified_cache_key(inputs, kwargs1)
        key2 = self.op_impl.kernel._generate_unified_cache_key(inputs, kwargs2)

        assert key1 != key2, "Cache keys should differ for different kernel_idx values"

    def test_cache_key_different_constant_args_key(self):
        """Test that different constant_args_key values produce different cache keys."""
        inputs = (torch.randn(4, 4),)

        kwargs1 = {
            "kernel_idx": 0,
            "constant_args_key": 123,
        }

        kwargs2 = {
            "kernel_idx": 0,
            "constant_args_key": 456,
        }

        key1 = self.op_impl.kernel._generate_unified_cache_key(inputs, kwargs1)
        key2 = self.op_impl.kernel._generate_unified_cache_key(inputs, kwargs2)

        assert key1 != key2, "Cache keys should differ for different constant_args_key values"

    def test_cache_key_different_tensor_shapes(self):
        """Test that different tensor shapes produce different cache keys."""
        inputs1 = (torch.randn(4, 4),)
        inputs2 = (torch.randn(8, 8),)

        kwargs = {
            "kernel_idx": 0,
            "constant_args_key": 123,
        }

        key1 = self.op_impl.kernel._generate_unified_cache_key(inputs1, kwargs)
        key2 = self.op_impl.kernel._generate_unified_cache_key(inputs2, kwargs)

        assert key1 != key2, "Cache keys should differ for different tensor shapes"

    def test_cache_key_different_tensor_dtypes(self):
        """Test that different tensor dtypes produce different cache keys."""
        inputs1 = (torch.randn(4, 4, dtype=torch.float32),)
        inputs2 = (torch.randn(4, 4, dtype=torch.float16),)

        kwargs = {
            "kernel_idx": 0,
            "constant_args_key": 123,
        }

        key1 = self.op_impl.kernel._generate_unified_cache_key(inputs1, kwargs)
        key2 = self.op_impl.kernel._generate_unified_cache_key(inputs2, kwargs)

        assert key1 != key2, "Cache keys should differ for different tensor dtypes"

    def test_cache_key_same_args(self):
        """Test that identical arguments produce the same cache key."""
        inputs = (torch.randn(4, 4),)

        kwargs = {
            "kernel_idx": 0,
            "constant_args_key": 123,
        }

        key1 = self.op_impl.kernel._generate_unified_cache_key(inputs, kwargs)
        key2 = self.op_impl.kernel._generate_unified_cache_key(inputs, kwargs)

        assert key1 == key2, "Cache keys should be identical for same arguments"

    def test_static_argnames_configuration(self):
        """Test that the static_argnames are correctly configured."""
        expected_static_argnames = (
            "kernel_idx",
            "grid",
            "backend_config",
            "operand_output_aliases",
            "arg_names",
            "constant_args_key",
        )

        expected_msg = (
            f"Expected static_argnames {expected_static_argnames}, "
            f"got {self.op_impl.static_argnames}"
        )
        assert self.op_impl.static_argnames == expected_static_argnames, expected_msg
