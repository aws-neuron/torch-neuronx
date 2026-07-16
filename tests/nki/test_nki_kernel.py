"""Tests for nki kernel integration."""

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


class TestGetConfigCache:
    """Test in-memory FrameworkConfig caching in get_config."""

    def _make_kernel(self):
        """Create a mock TorchNativeKernel with mocked compile() and get_cache_key()."""
        from nki.framework.torch_native import TorchNativeKernel

        kernel = MagicMock(spec=TorchNativeKernel)
        # Use a plain object for func so hasattr/_config_cache behaves like a real attribute.
        kernel.func = type("FakeFunc", (), {})()

        mock_config = MagicMock()
        mock_config.output_specs = [MagicMock(dtype="float32", shape=(1, 128))]
        mock_config.backend_config_b64 = b"dGVzdA=="
        mock_config.operand_output_aliases = {}
        kernel.compile = MagicMock(return_value=(mock_config, "key1"))
        kernel.get_cache_key = MagicMock(return_value="key1")
        kernel._get_compile_cache = MagicMock(return_value={})

        return kernel

    def test_cache_hit_skips_recompilation(self):
        """Second call with same key returns cached result without recompiling."""
        from torch_neuronx.nki_kernel import get_config

        kernel = self._make_kernel()

        result1 = get_config(kernel, x=torch.randn(1, 128))
        result2 = get_config(kernel, x=torch.randn(1, 128))

        assert result1 is result2
        assert kernel.compile.call_count == 1

    def test_cache_miss_on_different_key(self):
        """Different cache key triggers a new compile despite existing cache entry."""
        from torch_neuronx.nki_kernel import get_config

        kernel = self._make_kernel()

        # Populate cache for key1
        get_config(kernel, x=torch.randn(1, 128))
        kernel.compile.reset_mock()

        # Same key → cache hit, no compile
        get_config(kernel, x=torch.randn(1, 128))
        kernel.compile.assert_not_called()

        # Different key → cache miss, must compile
        kernel.get_cache_key.return_value = "key2"
        get_config(kernel, x=torch.randn(2, 128))
        kernel.compile.assert_called_once()

    def test_no_cache_when_compile_cache_disabled(self):
        """When _get_compile_cache returns None, every call compiles."""
        from torch_neuronx.nki_kernel import get_config

        kernel = self._make_kernel()
        kernel._get_compile_cache.return_value = None

        get_config(kernel, x=torch.randn(1, 128))
        get_config(kernel, x=torch.randn(1, 128))

        assert kernel.compile.call_count == 2

    def test_cache_stored_on_func(self):
        """Cache dict is stored as kernel.func._config_cache."""
        from torch_neuronx.nki_kernel import get_config

        kernel = self._make_kernel()

        result = get_config(kernel, x=torch.randn(1, 128))

        assert hasattr(kernel.func, "_config_cache")
        assert kernel.func._config_cache["key1"] is result

    def test_none_cache_key_bypasses_cache(self):
        """When get_cache_key returns None, cache is bypassed and compile runs every time."""
        from torch_neuronx.nki_kernel import get_config

        kernel = self._make_kernel()
        kernel.get_cache_key.return_value = None

        get_config(kernel, x=torch.randn(1, 128))
        get_config(kernel, x=torch.randn(1, 128))

        assert kernel.compile.call_count == 2
        assert not hasattr(kernel.func, "_config_cache") or None not in kernel.func._config_cache


class TestCacheKeyInvalidation:
    """Test that get_cache_key changes when kernel source is modified."""

    _KERNEL_SOURCE = """\
import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def scale_kernel(x):
    x_sbuf = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sbuf, src=x)
    y_sbuf = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=y_sbuf, data=x_sbuf, op0=nl.multiply, operand0={operand})
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=y_sbuf)
    return out
"""

    def _load_kernel(self, tmp_path, operand):
        """Write kernel source to file, import it, return the @nki.jit kernel."""
        import importlib.util

        source = self._KERNEL_SOURCE.format(operand=operand)
        module_file = tmp_path / "scale_kernel_mod.py"
        module_file.write_text(source)

        spec = importlib.util.spec_from_file_location("scale_kernel_mod", str(module_file))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.scale_kernel

    def test_cache_key_changes_on_source_modification(self, tmp_path):
        """Editing a kernel source file and reloading produces a different cache key."""
        from nki.framework.torch_native import TorchNativeKernel

        inp = torch.randn(1, 128, device="neuron")

        # Write kernel with operand0=2.0, import, get cache key
        kernel_jit = self._load_kernel(tmp_path, "2.0")
        key_v1 = TorchNativeKernel(kernel_jit.func).get_cache_key(x=inp)

        # Overwrite same file with operand0=3.0, reimport, get cache key
        kernel_jit = self._load_kernel(tmp_path, "3.0")
        key_v2 = TorchNativeKernel(kernel_jit.func).get_cache_key(x=inp)

        assert key_v1 != key_v2, (
            "get_cache_key must differ when kernel source is modified "
            f"(operand0=2.0 vs 3.0). Both produced: {key_v1}"
        )


class TestNkiHloCacheSourceInvalidation:
    """Test that NKI HLO cache key includes backend_config for invalidation."""

    def test_different_backend_config_produces_different_cache_key(self):
        """_generate_unified_cache_key must produce different keys when
        backend_config differs, even with same kernel_idx and qualname.

        backend_config changes when the kernel source changes, so including it
        in the cache key ensures stale HLO is not reused.
        """

        from nki.framework.torch_native import TorchNativeKernel

        from torch_neuronx.nki_hop import kernel_registry
        from torch_neuronx.python_ops.torch_mlir.nki_op_impl import NkiKernel

        mock_k = MagicMock(spec=TorchNativeKernel)
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
