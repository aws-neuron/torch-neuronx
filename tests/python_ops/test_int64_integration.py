"""
Integration tests for the int64 dispatch policy across all modes.

Each test class pins its int64 mode via ``@pytest.mark.int64_mode("...")``; the
autouse fixture in ``conftest.py`` syncs that mode across the env var, the
Python lru_caches (``get_int64_mode`` / ``Int64Policy``), ``global_version_hash``,
and the C++ mode storage (``_C._set_int64_mode_for_testing``) — so the whole
suite runs in a single pytest process.

Modes (unified taxonomy):
  - native        — per-op cascade: native ops S64; non-native packed; mm/conv CPU
  - packed        — every int64 op via the packed view (low word)
  - cpu_bounce    — int64→int32 via CPU roundtrip; op runs on device in int32
  - native_strict — S64 for native ops only; everything else CPU-falls-back

Requires a Neuron device.
"""

import os

import pytest
import torch

import torch_neuronx

# TORCH_NEURONX_CPP_DISPATCH is read once at module init on both the C++ and
# Python sides, so it can't be toggled per-test. The python-dispatch
# regression tests below must be skipped when the process runs with cpp dispatch.
_CPP_DISPATCH = os.environ.get("TORCH_NEURONX_CPP_DISPATCH", "1") == "1"

_NO_NEURON = not hasattr(torch, "neuron") or torch.neuron.device_count() == 0
requires_neuron = pytest.mark.skipif(_NO_NEURON, reason="No Neuron device available")


@pytest.fixture(params=["0", "1"], ids=["python_dispatch", "cpp_dispatch"])
def dispatch_mode(request, monkeypatch):
    monkeypatch.setenv("TORCH_NEURONX_CPP_DISPATCH", request.param)
    return request.param


# ═══════════════════════════════════════════════════════════════
#  native cascade
# ═══════════════════════════════════════════════════════════════


@requires_neuron
@pytest.mark.int64_mode("native")
class TestNativeOpsInt64:
    """NATIVE ops produce correct int64 results on device (S64 / GPSIMD)."""

    def test_add_int64(self, dispatch_mode):
        a = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device="neuron")
        b = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="neuron")
        result = a + b
        torch.neuron.synchronize()
        torch.testing.assert_close(result.cpu(), torch.tensor([11, 22, 33, 44], dtype=torch.int64))

    def test_mul_int64(self, dispatch_mode):
        a = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device="neuron")
        b = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="neuron")
        result = a * b
        torch.neuron.synchronize()
        torch.testing.assert_close(result.cpu(), torch.tensor([10, 40, 90, 160], dtype=torch.int64))

    def test_sub_int64(self, dispatch_mode):
        a = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device="neuron")
        b = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="neuron")
        result = a - b
        torch.neuron.synchronize()
        torch.testing.assert_close(result.cpu(), torch.tensor([9, 18, 27, 36], dtype=torch.int64))

    def test_comparison_int64(self, dispatch_mode):
        a = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device="neuron")
        b = torch.tensor([10, 25, 25, 45], dtype=torch.int64, device="neuron")
        result = a == b
        torch.neuron.synchronize()
        torch.testing.assert_close(result.cpu(), torch.tensor([True, False, False, False]))

    def test_sum_int64(self, dispatch_mode):
        a = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device="neuron")
        result = a.sum()
        torch.neuron.synchronize()
        assert result.cpu().item() == 100

    def test_int64_above_int32_max_is_exact(self, dispatch_mode):
        """Native mode runs int64 on GPSIMD, so values beyond int32 are exact."""
        big = (1 << 40) + 7  # overflows int32
        a = torch.tensor([big, big, big], dtype=torch.int64, device="neuron")
        b = torch.tensor([1, 2, 3], dtype=torch.int64, device="neuron")
        result = a + b
        torch.neuron.synchronize()
        expected = torch.tensor([big + 1, big + 2, big + 3], dtype=torch.int64)
        torch.testing.assert_close(result.cpu(), expected)


@requires_neuron
@pytest.mark.int64_mode("native")
class TestDowncastOpsInt64:
    """Non-native ops are demoted to int32 via the packed view; correct within
    int32 range."""

    def test_pow_int64(self, dispatch_mode):
        a = torch.tensor([2, 3, 4, 5], dtype=torch.int64, device="neuron")
        result = torch.pow(a, 2)
        torch.neuron.synchronize()
        torch.testing.assert_close(result.cpu(), torch.tensor([4, 9, 16, 25], dtype=torch.int64))

    def test_pow_int64_larger_values(self, dispatch_mode):
        a = torch.tensor([100, 200, 300, 400], dtype=torch.int64, device="neuron")
        result = torch.pow(a, 2)
        torch.neuron.synchronize()
        expected = torch.tensor([10000, 40000, 90000, 160000], dtype=torch.int64)
        torch.testing.assert_close(result.cpu(), expected)


@requires_neuron
@pytest.mark.int64_mode("native")
class TestUnsupportedOpsInt64:
    """matmul/conv int64 fall back to CPU and produce correct results."""

    def test_matmul_int64_cpu_fallback(self, dispatch_mode):
        a = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64, device="neuron")
        b = torch.tensor([[5, 6], [7, 8]], dtype=torch.int64, device="neuron")
        result = torch.matmul(a, b)
        torch.neuron.synchronize()
        expected = torch.tensor([[19, 22], [43, 50]], dtype=torch.int64)
        torch.testing.assert_close(result.cpu(), expected)


# ═══════════════════════════════════════════════════════════════
#  packed mode — every int64 op via the packed view
# ═══════════════════════════════════════════════════════════════


@requires_neuron
@pytest.mark.int64_mode("packed")
class TestPackedMode:
    """In packed mode all int64 ops run on device via the low-word packed view;
    correct within int32 range."""

    def test_add_int64(self, dispatch_mode):
        a = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device="neuron")
        b = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="neuron")
        result = a + b
        torch.neuron.synchronize()
        torch.testing.assert_close(result.cpu(), torch.tensor([11, 22, 33, 44], dtype=torch.int64))

    def test_pow_int64(self, dispatch_mode):
        a = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device="neuron")
        result = torch.pow(a, 2)
        torch.neuron.synchronize()
        expected = torch.tensor([100, 400, 900, 1600], dtype=torch.int64)
        torch.testing.assert_close(result.cpu(), expected)


# ═══════════════════════════════════════════════════════════════
#  cpu_bounce mode — int64→int32 via CPU roundtrip, op runs on device
# ═══════════════════════════════════════════════════════════════


@requires_neuron
@pytest.mark.int64_mode("cpu_bounce")
class TestCpuBounceMode:
    """In cpu_bounce mode the framework casts int64→int32 via a CPU roundtrip;
    results stay correct within int32 range."""

    def test_add_int64(self, dispatch_mode):
        a = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device="neuron")
        b = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="neuron")
        result = a + b
        torch.neuron.synchronize()
        torch.testing.assert_close(result.cpu(), torch.tensor([11, 22, 33, 44], dtype=torch.int64))

    def test_pow_int64(self, dispatch_mode):
        """The dispatcher must downcast int64 before lowering; otherwise
        neuronx-cc sees S64 pow with no int64 flag and fails compilation."""
        x = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device="neuron")
        result = torch.pow(x, 2)
        torch.neuron.synchronize()
        expected = torch.tensor([100, 400, 900, 1600], dtype=torch.int64)
        torch.testing.assert_close(result.cpu(), expected)

    def test_pow_int64_out(self, dispatch_mode):
        """out= variant — exercises the write-aliased branch in DowncastArgs."""
        x = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device="neuron")
        out = torch.zeros_like(x)
        torch.pow(x, 2, out=out)
        torch.neuron.synchronize()
        expected = torch.tensor([100, 400, 900, 1600], dtype=torch.int64)
        torch.testing.assert_close(out.cpu(), expected)


# ═══════════════════════════════════════════════════════════════
#  native_strict — S64 for native ops; everything else CPU-falls-back
# ═══════════════════════════════════════════════════════════════


@requires_neuron
@pytest.mark.int64_mode("native_strict")
class TestNativeStrictMode:
    """In native_strict mode, non-native ops fall back to CPU (no truncation)."""

    def test_pow_cpu_fallback(self, dispatch_mode):
        a = torch.tensor([2, 3, 4, 5], dtype=torch.int64, device="neuron")
        result = torch.pow(a, 2)
        torch.neuron.synchronize()
        torch.testing.assert_close(result.cpu(), torch.tensor([4, 9, 16, 25], dtype=torch.int64))

    def test_native_op_runs_on_neuron(self, dispatch_mode):
        a = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device="neuron")
        b = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="neuron")
        result = a + b
        torch.neuron.synchronize()
        torch.testing.assert_close(result.cpu(), torch.tensor([11, 22, 33, 44], dtype=torch.int64))


# ═══════════════════════════════════════════════════════════════
#  cpu_fallback — every int64 op runs entirely on CPU
# ═══════════════════════════════════════════════════════════════


@requires_neuron
@pytest.mark.int64_mode("cpu_fallback")
class TestCpuFallbackMode:
    """In cpu_fallback mode the whole int64 op runs on CPU (no Neuron exec).

    This differs from cpu_bounce, where the int64→int32 cast happens on CPU but
    the op still runs on device. Here even a natively-supported op like add
    reports a CPU fallback.
    """

    def test_add_runs_on_cpu(self, dispatch_mode):
        a = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device="neuron")
        b = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="neuron")
        torch_neuronx.clear_op_tracking()
        result = a + b
        torch.neuron.synchronize()
        fb = set(torch_neuronx.get_fallback_ops())
        assert any("add" in op for op in fb), f"add should CPU-fall-back in cpu_fallback; fb={fb}"
        torch.testing.assert_close(result.cpu(), torch.tensor([11, 22, 33, 44], dtype=torch.int64))

    def test_int64_above_int32_max_is_exact(self, dispatch_mode):
        # Running on CPU, full int64 precision is preserved regardless of compiler.
        big = (1 << 40) + 7
        a = torch.tensor([big, big, big], dtype=torch.int64, device="neuron")
        b = torch.tensor([1, 2, 3], dtype=torch.int64, device="neuron")
        result = a + b
        torch.neuron.synchronize()
        expected = torch.tensor([big + 1, big + 2, big + 3], dtype=torch.int64)
        torch.testing.assert_close(result.cpu(), expected)


# ═══════════════════════════════════════════════════════════════════════════
#  Python-dispatch CPU fallback regressions
#
#  These pin observable bugs in
#  torch_neuronx/python_ops/torch_mlir/op_impl.py::TorchMlirOpImpl.__call__
#  CPU-fallback branch (out= dropped, inplace not mutating self, multi-return
#  tensors left on CPU). The C++ dispatcher (TORCH_NEURONX_CPP_DISPATCH=1) uses
#  at::native::cpu_fallback, which handles all three correctly, so these are
#  skipped on that path. They use ops that actually CPU-fall-back:
#    * aten::mm (UNSUPPORTED) in native mode, and
#    * non-native ops (pow, sort, max.dim) in native_strict mode.
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(
    _CPP_DISPATCH,
    reason="targets python-dispatch op_impl.py CPU fallback; "
    "run with TORCH_NEURONX_CPP_DISPATCH=0 to exercise",
)
@requires_neuron
@pytest.mark.int64_mode("native")
class TestPythonCpuFallbackOutKwarg:
    """CPU fallback must honor user-provided ``out=`` tensors. Uses ``mm.out``
    (mm is UNSUPPORTED), so the CPU fallback fires in native mode."""

    def test_mm_out_writes_user_tensor(self):
        a = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64, device="neuron")
        b = torch.tensor([[5, 6], [7, 8]], dtype=torch.int64, device="neuron")
        out = torch.zeros(2, 2, dtype=torch.int64, device="neuron")
        out_storage_ptr_before = out.untyped_storage().data_ptr()

        ret = torch.mm(a, b, out=out)
        torch.neuron.synchronize()

        assert ret.data_ptr() == out.data_ptr(), (
            "Return value of out= op must alias the user's out tensor; got a "
            "fresh tensor instead. CPU fallback dropped out= kwarg."
        )
        assert (
            out.untyped_storage().data_ptr() == out_storage_ptr_before
        ), "User's out tensor storage was replaced; CPU fallback ignored out=."
        expected = torch.tensor([[19, 22], [43, 50]], dtype=torch.int64)
        torch.testing.assert_close(out.cpu(), expected)


@pytest.mark.skipif(
    _CPP_DISPATCH,
    reason="targets python-dispatch op_impl.py CPU fallback; "
    "run with TORCH_NEURONX_CPP_DISPATCH=0 to exercise",
)
@requires_neuron
@pytest.mark.int64_mode("native_strict")
class TestPythonCpuFallbackOutKwargNativeStrict:
    """In native_strict, non-native ops fall back to CPU. The out= contract
    still applies (uses pow.out)."""

    def test_pow_out_writes_user_tensor(self):
        x = torch.tensor([2, 3, 4, 5], dtype=torch.int64, device="neuron")
        out = torch.zeros_like(x)
        out_storage_ptr_before = out.untyped_storage().data_ptr()

        ret = torch.pow(x, 2, out=out)
        torch.neuron.synchronize()

        assert ret.data_ptr() == out.data_ptr(), "pow(out=) must return the user's out tensor."
        assert out.untyped_storage().data_ptr() == out_storage_ptr_before
        torch.testing.assert_close(out.cpu(), torch.tensor([4, 9, 16, 25], dtype=torch.int64))


@pytest.mark.skipif(
    _CPP_DISPATCH,
    reason="targets python-dispatch op_impl.py CPU fallback; "
    "run with TORCH_NEURONX_CPP_DISPATCH=0 to exercise",
)
@requires_neuron
@pytest.mark.int64_mode("native_strict")
class TestPythonCpuFallbackInplace:
    """In native_strict, a non-native inplace op (pow_) CPU-falls-back. It must
    still mutate ``self`` rather than returning a fresh tensor."""

    def test_pow_inplace_mutates_self(self):
        a = torch.tensor([2, 3, 4, 5], dtype=torch.int64, device="neuron")
        a_storage_ptr_before = a.untyped_storage().data_ptr()

        ret = a.pow_(2)
        torch.neuron.synchronize()

        assert ret.data_ptr() == a.data_ptr(), (
            "Inplace pow_ must return self; got a fresh tensor. CPU fallback "
            "constructed a new tensor and discarded the inplace alias."
        )
        assert (
            a.untyped_storage().data_ptr() == a_storage_ptr_before
        ), "Inplace pow_ rebound self's storage; the original tensor was not mutated."
        torch.testing.assert_close(a.cpu(), torch.tensor([4, 9, 16, 25], dtype=torch.int64))


@pytest.mark.skipif(
    _CPP_DISPATCH,
    reason="targets python-dispatch op_impl.py CPU fallback; "
    "run with TORCH_NEURONX_CPP_DISPATCH=0 to exercise",
)
@requires_neuron
@pytest.mark.int64_mode("native_strict")
class TestPythonCpuFallbackMultiReturn:
    """Multi-tensor-return ops (sort, max.dim) are non-native, so they CPU-fall-
    back in native_strict. Each output must land on the neuron device."""

    def test_sort_returns_neuron_tensors(self):
        x = torch.tensor([3, 1, 4, 1, 5], dtype=torch.int64, device="neuron")
        values, indices = torch.sort(x)
        torch.neuron.synchronize()

        assert (
            values.device.type == "neuron"
        ), f"sort values should be on neuron; got {values.device}."
        assert (
            indices.device.type == "neuron"
        ), f"sort indices should be on neuron; got {indices.device}."
        torch.testing.assert_close(values.cpu(), torch.tensor([1, 1, 3, 4, 5], dtype=torch.int64))

    def test_max_dim_returns_neuron_tensors(self):
        x = torch.tensor([[3, 1, 4], [1, 5, 9]], dtype=torch.int64, device="neuron")
        values, indices = torch.max(x, dim=1)
        torch.neuron.synchronize()

        assert values.device.type == "neuron"
        assert indices.device.type == "neuron"
        torch.testing.assert_close(values.cpu(), torch.tensor([4, 9], dtype=torch.int64))


# ═══════════════════════════════════════════════════════════════════════════
#  Native-mode inplace small-int add.
#
#  ``t.add_(5)`` on a small-int tensor reroutes to ``aten::add.out`` with the
#  Python int wrapped as a 0-d Long (a mixed-dtype inplace on aliased self).
#  This previously failed to compile on Neuron (NCC_IBIR098 "Unsupported DGE
#  Type: HWDGE"); the compiler's native-int64 support + the 64-bit→32-bit
#  integer-cast-in-DMA correctness fix resolved it, so it now runs on device.
# ═══════════════════════════════════════════════════════════════════════════


@requires_neuron
@pytest.mark.int64_mode("native")
class TestNativeInplaceAddSmallInt:
    @pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.uint8])
    def test_inplace_add_small_int_native(self, dtype):
        t = torch.tensor([1, 2, 3], dtype=dtype, device="neuron")
        t.add_(5)
        torch.neuron.synchronize()
        torch.testing.assert_close(t.cpu(), torch.tensor([6, 7, 8], dtype=dtype))

    def test_inplace_add_small_int_cpu_bounce_works(self):
        """Cross-mode sanity: the same op in cpu_bounce mode is also correct."""
        from torch_neuronx.int64_policy import set_int64_mode_for_testing

        try:
            set_int64_mode_for_testing("cpu_bounce")
            t = torch.tensor([1, 2, 3], dtype=torch.int8, device="neuron")
            t.add_(5)
            torch.neuron.synchronize()
            torch.testing.assert_close(t.cpu(), torch.tensor([6, 7, 8], dtype=torch.int8))
        finally:
            set_int64_mode_for_testing("native")
