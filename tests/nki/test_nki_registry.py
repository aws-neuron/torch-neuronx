"""Tests for NKIRegistry kernel registration and lookup."""

from unittest.mock import MagicMock

import pytest

from torch_neuronx.nki_hop import NKIRegistry
from torch_neuronx.nki_kernel import TorchNeuronNKIKernelV3


@pytest.fixture
def registry():
    registry = NKIRegistry()
    registry.reset_table()
    return registry


def _make_mock_kernel():
    return MagicMock(spec=TorchNeuronNKIKernelV3)


class TestNKIRegistryAddKernel:
    def test_add_kernel_returns_index(self, registry):
        kernel = _make_mock_kernel()
        idx = registry.add_kernel(kernel, ["x", "y"])
        assert idx == 0

    def test_add_multiple_kernels_returns_sequential_indices(self, registry):
        k1 = _make_mock_kernel()
        k2 = _make_mock_kernel()
        assert registry.add_kernel(k1, ["x"]) == 0
        assert registry.add_kernel(k2, ["x"]) == 1

    def test_add_same_kernel_twice_returns_same_index(self, registry):
        kernel = _make_mock_kernel()
        idx1 = registry.add_kernel(kernel, ["x"])
        idx2 = registry.add_kernel(kernel, ["x"])
        assert idx1 == idx2
        assert len(registry.id_to_kernel) == 1


class TestNKIRegistryGetKernel:
    def test_get_kernel_returns_registered_kernel(self, registry):
        kernel = _make_mock_kernel()
        idx = registry.add_kernel(kernel, ["a", "b"])
        assert registry.get_kernel(idx) is kernel

    def test_get_kernel_invalid_index_raises(self, registry):
        with pytest.raises(AssertionError):
            registry.get_kernel(999)

    def test_get_kernel_arg_names(self, registry):
        kernel = _make_mock_kernel()
        idx = registry.add_kernel(kernel, ["a", "b", "c"])
        assert registry.get_kernel_arg_names(idx) == ["a", "b", "c"]

    def test_get_kernel_default_args(self, registry):
        kernel = _make_mock_kernel()
        defaults = {"scale": 1.0}
        idx = registry.add_kernel(kernel, ["x", "scale"], defaults)
        assert registry.get_kernel_default_args(idx) == {"scale": 1.0}

    def test_get_kernel_default_args_none_becomes_empty(self, registry):
        kernel = _make_mock_kernel()
        idx = registry.add_kernel(kernel, ["x"], None)
        assert registry.get_kernel_default_args(idx) == {}


class TestNKIRegistryResetTable:
    def test_reset_clears_all_state(self, registry):
        kernel = _make_mock_kernel()
        registry.add_kernel(kernel, ["x"])
        registry.add_constant_args({"c": 1}, "key1")

        registry.reset_table()

        assert registry.id_to_kernel == {}
        assert registry.kernel_to_id == {}
        assert registry.kernel_arg_names == {}
        assert registry.kernel_default_args == {}
        assert registry.constant_args == {}


class TestNKIRegistryConstantArgs:
    def test_add_and_get_constant_args(self, registry):
        registry.add_constant_args({"scale": 2.0}, "hash1")
        assert registry.get_constant_args("hash1") == {"scale": 2.0}

    def test_get_constant_args_missing_key_raises(self, registry):
        with pytest.raises(AssertionError):
            registry.get_constant_args("nonexistent")
