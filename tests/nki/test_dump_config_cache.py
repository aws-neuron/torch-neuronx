"""Tests for dump_config cross-rank disk cache coordination in nki_kernel.py."""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from torch_neuronx.nki_kernel import DumpedConfig, TorchNeuronNKIKernelV3
from torch_neuronx.python_ops.compilation import hlo_cache as _hlo_cache_mod


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path):
    """Isolate each test with a fresh cache dir and enable hlo_cache."""
    orig_enabled = _hlo_cache_mod._enabled
    _hlo_cache_mod._enabled = True
    with patch.dict(
        os.environ, {"TORCH_NEURONX_HLO_CACHE_DIR": str(tmp_path), "NKI_DUMP_PERSISTENT_CACHE": "1"}
    ):
        yield tmp_path
    _hlo_cache_mod._enabled = orig_enabled


def _make_dconfig(tag="test"):
    return DumpedConfig(
        return_types=[("float32", (1, 1))],
        dumped_config=tag,
        operand_output_aliases={},
    )


def _make_kernel(expected=None):
    if expected is None:
        expected = _make_dconfig("compiled")
    kernel = MagicMock(spec=TorchNeuronNKIKernelV3)
    kernel.func = MagicMock()
    kernel.func.__qualname__ = "test_kernel"
    kernel.func.__code__ = MagicMock(co_filename="test_kernel.py", co_code=b"mock_bytecode")
    if hasattr(kernel.func, "_dump_config_cache"):
        del kernel.func._dump_config_cache
    kernel._compile_dump_config = MagicMock(return_value=expected)
    kernel.lnc = 1
    kernel._enable_simulation = False
    kernel._enable_device_dump = False
    kernel._get_compile_cache = MagicMock(return_value={})
    kernel._generate_cache_key = MagicMock(return_value="test_cache_key")
    kernel._get_dump_config_cache = TorchNeuronNKIKernelV3._get_dump_config_cache.__get__(kernel)
    real_dump_config = TorchNeuronNKIKernelV3.dump_config.__get__(kernel)
    kernel.dump_config = real_dump_config
    return kernel


class TestDumpConfigCrossRankCache:
    def test_first_call_compiles_and_caches(self):
        kernel = _make_kernel()
        result = kernel.dump_config(x=torch.randn(1, 128))

        assert result.dumped_config == "compiled"
        kernel._compile_dump_config.assert_called_once()

    def test_second_call_hits_in_memory_cache(self):
        kernel = _make_kernel()
        result1 = kernel.dump_config(x=torch.randn(1, 128))
        result2 = kernel.dump_config(x=torch.randn(1, 128))

        assert result1 is result2
        kernel._compile_dump_config.assert_called_once()

    def test_persistent_cache_hit_skips_compilation(self):
        """Simulate rank 2 finding result written by rank 1."""
        expected = _make_dconfig("from_rank1")

        # Rank 1: compile and write to persistent cache
        kernel1 = _make_kernel(expected)
        kernel1.dump_config(x=torch.randn(1, 128))
        kernel1._compile_dump_config.assert_called_once()

        # Rank 2: fresh kernel (no in-memory cache), should hit persistent
        kernel2 = _make_kernel(_make_dconfig("should_not_compile"))
        kernel2._generate_cache_key.return_value = "test_cache_key"  # same key
        kernel2.func.__qualname__ = "test_kernel"  # same kernel identity
        result = kernel2.dump_config(x=torch.randn(1, 128))

        assert result.dumped_config == "from_rank1"
        kernel2._compile_dump_config.assert_not_called()

    def test_different_keys_compile_separately(self):
        kernel = _make_kernel()
        kernel._generate_cache_key.side_effect = ["key_a", "key_b"]

        kernel.dump_config(x=torch.randn(1, 128))
        kernel.dump_config(x=torch.randn(2, 128))

        assert kernel._compile_dump_config.call_count == 2

    def test_different_kernels_dont_collide(self):
        """Two different kernel functions with same input shapes get separate cache entries."""
        dconfig_a = _make_dconfig("kernel_a")
        dconfig_b = _make_dconfig("kernel_b")

        kernel_a = _make_kernel(dconfig_a)
        kernel_a.func.__qualname__ = "kernel_a_func"
        kernel_a.dump_config(x=torch.randn(1, 128))

        kernel_b = _make_kernel(dconfig_b)
        kernel_b.func.__qualname__ = "kernel_b_func"
        result_b = kernel_b.dump_config(x=torch.randn(1, 128))

        # kernel_b should compile (not reuse kernel_a's cache)
        assert result_b.dumped_config == "kernel_b"
        kernel_b._compile_dump_config.assert_called_once()

    def test_no_cache_when_key_is_none(self):
        kernel = _make_kernel()
        kernel._generate_cache_key.return_value = None

        kernel.dump_config(x=torch.randn(1, 128))
        kernel.dump_config(x=torch.randn(1, 128))

        assert kernel._compile_dump_config.call_count == 2

    def test_no_cache_when_compile_cache_disabled(self):
        kernel = _make_kernel()
        kernel._get_compile_cache.return_value = None

        kernel.dump_config(x=torch.randn(1, 128))
        kernel.dump_config(x=torch.randn(1, 128))

        assert kernel._compile_dump_config.call_count == 2
