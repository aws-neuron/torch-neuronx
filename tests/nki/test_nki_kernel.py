"""Tests for nki kernel integration."""

from unittest.mock import MagicMock, patch

import nki
import nki.isa as nisa
import nki.language as nl
import pytest
import torch

from tests.utils.neuron_test_utils import assert_raises
from torch_neuronx.python_ops.compilation import hlo_cache as _hlo_cache_mod


@pytest.fixture(autouse=True)
def _disable_hlo_disk_cache():
    """Disable HLO persistent cache for inline NKI kernel tests.

    Inline @nki.jit kernels get kernel_idx assigned at runtime, which
    can differ across forked test workers sharing the same disk cache.
    """
    orig = _hlo_cache_mod._enabled
    _hlo_cache_mod._enabled = False
    yield
    _hlo_cache_mod._enabled = orig


class TestBasicExecution:
    """Test basic kernel execution functionality."""

    def test_simple_dma_copy(self):
        """Test a simple DMA copy kernel."""

        @nki.jit
        def dma_copy(x):
            x_sbuf = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_sbuf, src=x)

            out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
            nisa.dma_copy(dst=out, src=x_sbuf)
            return out

        x = torch.randn(1, 128, device="neuron")

        result = dma_copy(x)

        assert torch.allclose(result, x)

    def test_tensor_scale(self):
        """Test scaling a tensor by a constant."""

        @nki.jit
        def tensor_scale(x, scale):
            x_sbuf = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_sbuf, src=x)

            y_sbuf = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=y_sbuf, data=x_sbuf, op0=nl.multiply, operand0=scale)

            out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
            nisa.dma_copy(dst=out, src=y_sbuf)
            return out

        x = torch.randn(1, 128, device="neuron")
        scale = 2.5

        result = tensor_scale(x, scale)

        expected = x * scale
        assert torch.allclose(result, expected)

    def test_element_wise_add(self):
        """Test element-wise addition of two tensors."""

        @nki.jit
        def tensor_add(x1, x2):
            x1_sbuf = nl.ndarray(x1.shape, dtype=x1.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x1_sbuf, src=x1)

            x2_sbuf = nl.ndarray(x2.shape, dtype=x2.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x2_sbuf, src=x2)

            y_sbuf = nl.ndarray(x1.shape, dtype=x1.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=y_sbuf, data1=x1_sbuf, data2=x2_sbuf, op=nl.add)

            out = nl.ndarray(x1.shape, dtype=x1.dtype, buffer=nl.shared_hbm)
            nisa.dma_copy(dst=out, src=y_sbuf)
            return out

        x1 = torch.randn(1, 128, device="neuron")
        x2 = torch.randn(1, 128, device="neuron")

        result = tensor_add(x1, x2)

        expected = x1 + x2
        assert torch.allclose(result, expected)

    def test_different_shapes(self):
        """Test kernels with different tensor shapes."""

        @nki.jit
        def copy_2d(x):
            x_sbuf = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_sbuf, src=x)

            out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
            nisa.dma_copy(dst=out, src=x_sbuf)
            return out

        shapes = [(10, 10), (16, 32), (128, 64)]

        for shape in shapes:
            x = torch.randn(*shape, device="neuron")
            result = copy_2d(x)
            assert torch.allclose(result, x)

    def test_different_dtypes(self):
        """Test kernels with different data types."""

        @nki.jit
        def copy_kernel(x):
            x_sbuf = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_sbuf, src=x)

            out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
            nisa.dma_copy(dst=out, src=x_sbuf)
            return out

        dtypes = [torch.float32, torch.float16, torch.int32]

        for dtype in dtypes:
            if dtype in [torch.int32]:
                x = torch.randint(0, 100, (1, 64), device="neuron", dtype=dtype)
            else:
                x = torch.randn(1, 64, device="neuron", dtype=dtype)

            result = copy_kernel(x)
            assert torch.equal(result, x)


class TestDumpConfigCache:
    """Test DumpedConfig caching in dump_config."""

    def _make_kernel(self):
        """Create a TorchNeuronNKIKernelV3 with mocked internals."""
        from torch_neuronx.nki_kernel import DumpedConfig, TorchNeuronNKIKernelV3

        kernel = MagicMock(spec=TorchNeuronNKIKernelV3)
        kernel.func = MagicMock()
        # Ensure _dump_config_cache doesn't exist yet
        if hasattr(kernel.func, "_dump_config_cache"):
            del kernel.func._dump_config_cache

        expected = DumpedConfig(
            return_types=[("float32", (1, 128))],
            dumped_config="dGVzdA==",
            operand_output_aliases={},
        )
        return kernel, expected

    def _setup_compile_mocks(self, kernel, expected):
        """Wire up the mocked compile pipeline to return expected DumpedConfig fields."""
        from torch_neuronx.nki_kernel import TorchNeuronNKIKernelV3

        output_spec = MagicMock()
        output_spec.dtype = expected.return_types[0][0]
        output_spec.shape = expected.return_types[0][1]

        result = MagicMock()
        result.output_specs = [output_spec]
        result.input_specs = []
        result.input_output_aliases = {}

        kernel._cached_compile_to_bir.return_value = (MagicMock(), result)
        kernel._build_backend_config.return_value = {"test": True}
        kernel._frontend_cls.return_value = MagicMock()
        kernel._enable_backend_opt = False
        kernel._compile_opts.return_value = MagicMock()
        kernel.lnc = 1
        kernel._enable_simulation = False
        kernel._enable_device_dump = False

        # Bind the real methods
        kernel._get_dump_config_cache = TorchNeuronNKIKernelV3._get_dump_config_cache.__get__(
            kernel
        )
        # Wrap dump_config so that dataclasses.replace() on the mock compile_opts doesn't fail
        real_dump_config = TorchNeuronNKIKernelV3.dump_config.__get__(kernel)

        def patched_dump_config(**kwargs):
            with patch("torch_neuronx.nki_kernel.replace", side_effect=lambda obj, **kw: obj):
                return real_dump_config(**kwargs)

        kernel.dump_config = patched_dump_config

    def test_cache_hit_skips_recompilation(self):
        """Second call with same key returns cached result without recompiling."""
        kernel, expected = self._make_kernel()
        self._setup_compile_mocks(kernel, expected)

        kernel._get_compile_cache.return_value = {}
        kernel._generate_cache_key.return_value = "key1"

        result1 = kernel.dump_config(x=torch.randn(1, 128))
        result2 = kernel.dump_config(x=torch.randn(1, 128))

        assert result1 is result2
        assert kernel._cached_compile_to_bir.call_count == 1

    def test_cache_miss_on_different_key(self):
        """Different cache keys trigger separate compilations."""
        kernel, expected = self._make_kernel()
        self._setup_compile_mocks(kernel, expected)

        kernel._get_compile_cache.return_value = {}
        kernel._generate_cache_key.side_effect = ["key1", "key2"]

        kernel.dump_config(x=torch.randn(1, 128))
        kernel.dump_config(x=torch.randn(2, 128))

        assert kernel._cached_compile_to_bir.call_count == 2

    def test_no_cache_when_compile_cache_disabled(self):
        """When _get_compile_cache returns None, every call compiles."""
        kernel, expected = self._make_kernel()
        self._setup_compile_mocks(kernel, expected)

        kernel._get_compile_cache.return_value = None

        kernel.dump_config(x=torch.randn(1, 128))
        kernel.dump_config(x=torch.randn(1, 128))

        assert kernel._cached_compile_to_bir.call_count == 2
        assert not hasattr(kernel.func, "_dump_config_cache")

    def test_no_cache_when_key_is_none(self):
        """When _generate_cache_key returns None (unhashable inputs), every call compiles."""
        kernel, expected = self._make_kernel()
        self._setup_compile_mocks(kernel, expected)

        kernel._get_compile_cache.return_value = {}
        kernel._generate_cache_key.return_value = None

        kernel.dump_config(x=torch.randn(1, 128))
        kernel.dump_config(x=torch.randn(1, 128))

        assert kernel._cached_compile_to_bir.call_count == 2

    def test_cache_stored_on_func(self):
        """Cache dict is stored as self.func._dump_config_cache."""
        kernel, expected = self._make_kernel()
        self._setup_compile_mocks(kernel, expected)

        kernel._get_compile_cache.return_value = {}
        kernel._generate_cache_key.return_value = "key1"

        result = kernel.dump_config(x=torch.randn(1, 128))

        assert hasattr(kernel.func, "_dump_config_cache")
        assert kernel.func._dump_config_cache["key1"] is result
