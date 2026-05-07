"""Tests for nki kernel integration."""

import os
from unittest.mock import MagicMock

import nki
import nki.isa as nisa
import nki.language as nl
import pytest
import torch

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

    def setup_method(self):
        import tempfile

        self._tmpdir = tempfile.mkdtemp()
        os.environ["TORCH_NEURONX_HLO_CACHE_DIR"] = self._tmpdir

    def teardown_method(self):
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)
        os.environ.pop("TORCH_NEURONX_HLO_CACHE_DIR", None)

    def _make_kernel(self):
        """Create a TorchNeuronNKIKernelV3 with mocked internals."""
        from torch_neuronx.nki_kernel import DumpedConfig, TorchNeuronNKIKernelV3

        kernel = MagicMock(spec=TorchNeuronNKIKernelV3)
        kernel.func = MagicMock()
        kernel.func.__qualname__ = "mock_nki_kernel"
        kernel.func.__code__ = MagicMock(co_filename="mock_nki_kernel.py", co_code=b"mock_bytecode")
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

        kernel._compile_dump_config = MagicMock(return_value=expected)
        kernel.lnc = 1
        kernel._enable_simulation = False
        kernel._enable_device_dump = False

        # Bind the real methods
        kernel._get_dump_config_cache = TorchNeuronNKIKernelV3._get_dump_config_cache.__get__(
            kernel
        )
        kernel.dump_config = TorchNeuronNKIKernelV3.dump_config.__get__(kernel)

    def test_cache_hit_skips_recompilation(self):
        """Second call with same key returns cached result without recompiling."""
        kernel, expected = self._make_kernel()
        self._setup_compile_mocks(kernel, expected)

        kernel._get_compile_cache.return_value = {}
        kernel._generate_cache_key.return_value = "key1"

        result1 = kernel.dump_config(x=torch.randn(1, 128))
        result2 = kernel.dump_config(x=torch.randn(1, 128))

        assert result1 is result2
        assert kernel._compile_dump_config.call_count == 1

    def test_cache_miss_on_different_key(self):
        """Different cache keys trigger separate compilations."""
        kernel, expected = self._make_kernel()
        self._setup_compile_mocks(kernel, expected)

        kernel._get_compile_cache.return_value = {}
        kernel._generate_cache_key.side_effect = ["key1", "key2"]

        kernel.dump_config(x=torch.randn(1, 128))
        kernel.dump_config(x=torch.randn(2, 128))

        assert kernel._compile_dump_config.call_count == 2

    def test_no_cache_when_compile_cache_disabled(self):
        """When _get_compile_cache returns None, every call compiles."""
        kernel, expected = self._make_kernel()
        self._setup_compile_mocks(kernel, expected)

        kernel._get_compile_cache.return_value = None

        kernel.dump_config(x=torch.randn(1, 128))
        kernel.dump_config(x=torch.randn(1, 128))

        assert kernel._compile_dump_config.call_count == 2

    def test_no_cache_when_key_is_none(self):
        """When _generate_cache_key returns None (unhashable inputs), every call compiles."""
        kernel, expected = self._make_kernel()
        self._setup_compile_mocks(kernel, expected)

        kernel._get_compile_cache.return_value = {}
        kernel._generate_cache_key.return_value = None

        kernel.dump_config(x=torch.randn(1, 128))
        kernel.dump_config(x=torch.randn(1, 128))

        assert kernel._compile_dump_config.call_count == 2

    def test_cache_stored_on_func(self):
        """Cache dict is stored as self.func._dump_config_cache."""
        kernel, expected = self._make_kernel()
        self._setup_compile_mocks(kernel, expected)

        kernel._get_compile_cache.return_value = {}
        kernel._generate_cache_key.return_value = "key1"

        result = kernel.dump_config(x=torch.randn(1, 128))

        assert hasattr(kernel.func, "_dump_config_cache")
        assert kernel.func._dump_config_cache["key1"] is result


class TestNkiHloCacheSourceInvalidation:
    """Test that NKI HLO cache key includes backend_config for invalidation."""

    def test_different_backend_config_produces_different_cache_key(self):
        """_generate_unified_cache_key must produce different keys when
        backend_config differs, even with same kernel_idx and qualname.

        backend_config changes when the kernel source changes, so including it
        in the cache key ensures stale HLO is not reused.
        """
        from unittest.mock import MagicMock

        from torch_neuronx.nki_hop import kernel_registry
        from torch_neuronx.nki_kernel import TorchNeuronNKIKernelV3
        from torch_neuronx.python_ops.torch_mlir.nki_op_impl import NkiKernel

        mock_k = MagicMock(spec=TorchNeuronNKIKernelV3)
        mock_k.func = MagicMock()
        mock_k.func.__qualname__ = "scale_kernel"
        idx = kernel_registry.add_kernel(mock_k, ["x"])

        nki_kernel = NkiKernel(
            op_name="nki_test",
            torch_fn=lambda *a, **kw: None,
            static_argnums=(),
            static_argnames=("kernel_idx", "constant_args_key", "grid", "backend_config"),
        )

        dummy_input = torch.randn(1, 128, device="neuron")
        inputs = (dummy_input,)
        base_kwargs = {"kernel_idx": idx, "constant_args_key": 0, "grid": ()}

        key_v1 = nki_kernel._generate_unified_cache_key(
            inputs, {**base_kwargs, "backend_config": "compiled_bir_v1_multiply_by_1"}
        )
        key_v2 = nki_kernel._generate_unified_cache_key(
            inputs, {**base_kwargs, "backend_config": "compiled_bir_v2_multiply_by_2"}
        )

        assert key_v1 != key_v2, (
            f"HLO cache keys must differ when backend_config changes. " f"Both produced: {key_v1}"
        )
