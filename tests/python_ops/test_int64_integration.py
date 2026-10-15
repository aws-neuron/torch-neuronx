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

import pytest
import torch

import torch_neuronx

_NO_NEURON = not hasattr(torch, "neuron") or torch.neuron.device_count() == 0
requires_neuron = pytest.mark.skipif(_NO_NEURON, reason="No Neuron device available")


# ═══════════════════════════════════════════════════════════════
#  native cascade
# ═══════════════════════════════════════════════════════════════


@requires_neuron
@pytest.mark.int64_mode("native")
class TestNativeOpsInt64:
    """NATIVE ops produce correct int64 results on device (S64 / GPSIMD)."""

    def test_add_int64(self):
        a = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device="neuron")
        b = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="neuron")
        result = a + b
        torch.neuron.synchronize()
        torch.testing.assert_close(result.cpu(), torch.tensor([11, 22, 33, 44], dtype=torch.int64))

    def test_mul_int64(self):
        a = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device="neuron")
        b = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="neuron")
        result = a * b
        torch.neuron.synchronize()
        torch.testing.assert_close(result.cpu(), torch.tensor([10, 40, 90, 160], dtype=torch.int64))

    def test_sub_int64(self):
        a = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device="neuron")
        b = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="neuron")
        result = a - b
        torch.neuron.synchronize()
        torch.testing.assert_close(result.cpu(), torch.tensor([9, 18, 27, 36], dtype=torch.int64))

    def test_comparison_int64(self):
        a = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device="neuron")
        b = torch.tensor([10, 25, 25, 45], dtype=torch.int64, device="neuron")
        result = a == b
        torch.neuron.synchronize()
        torch.testing.assert_close(result.cpu(), torch.tensor([True, False, False, False]))

    def test_sum_int64(self):
        a = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device="neuron")
        result = a.sum()
        torch.neuron.synchronize()
        assert result.cpu().item() == 100

    def test_int64_above_int32_max_is_exact(self):
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

    def test_pow_int64(self):
        a = torch.tensor([2, 3, 4, 5], dtype=torch.int64, device="neuron")
        result = torch.pow(a, 2)
        torch.neuron.synchronize()
        torch.testing.assert_close(result.cpu(), torch.tensor([4, 9, 16, 25], dtype=torch.int64))

    def test_pow_int64_larger_values(self):
        a = torch.tensor([100, 200, 300, 400], dtype=torch.int64, device="neuron")
        result = torch.pow(a, 2)
        torch.neuron.synchronize()
        expected = torch.tensor([10000, 40000, 90000, 160000], dtype=torch.int64)
        torch.testing.assert_close(result.cpu(), expected)


@requires_neuron
@pytest.mark.int64_mode("native")
class TestUnsupportedOpsInt64:
    """matmul/conv int64 fall back to CPU and produce correct results."""

    def test_matmul_int64_cpu_fallback(self):
        a = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64, device="neuron")
        b = torch.tensor([[5, 6], [7, 8]], dtype=torch.int64, device="neuron")
        result = torch.matmul(a, b)
        torch.neuron.synchronize()
        expected = torch.tensor([[19, 22], [43, 50]], dtype=torch.int64)
        torch.testing.assert_close(result.cpu(), expected)


@requires_neuron
@pytest.mark.int64_mode("native")
class TestMixedDtypePromotionInt64:
    """A mixed int32/int64 op widens its int32 operand on device.

    This is a regression test for a fixed bug. NeuronSafeCast once gated the
    on-device widening (materialize_from_i32) on the packed mode alone. Native
    mode then took the generic 64-bit CPU round-trip, and the operand went to
    the host and back only to be promoted. The results stayed correct, so these
    tests check where the promotion runs, not only the values.
    """

    @staticmethod
    def _track(fn):
        """Run fn and report the ops it executed. Drains lazy work on both sides."""
        torch.neuron.synchronize()
        torch_neuronx.clear_op_tracking()
        result = fn()
        torch.neuron.synchronize()
        return (
            result,
            set(torch_neuronx.get_executed_ops()),
            set(torch_neuronx.get_fallback_ops()),
        )

    def test_promotion_does_not_round_trip_through_host(self):
        a = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device="neuron")
        b = torch.tensor([10, 21, 30, 41], dtype=torch.int32, device="neuron")

        result, executed, fallback = self._track(lambda: torch.eq(a, b))

        # A host round-trip shows up in two ways. The widening records an
        # aten::_to_copy fallback, and it reads the operand back to the host.
        # The on-device path only uploads the 8-byte shift constant, so a
        # device-to-host copy is the exact marker of the round-trip.
        assert "aten::_to_copy" not in fallback, f"promotion fell back to CPU; fallback={fallback}"
        assert (
            "neuron::copy::neuron_to_cpu" not in executed
        ), f"promotion copied the operand to the host; executed={executed}"
        expected = torch.tensor([True, False, True, False])
        torch.testing.assert_close(result.cpu(), expected)

    def test_promotion_keeps_the_int64_high_word(self):
        # 2**40 + 7 and 7 share their low 32 bits, so a low-word-only promotion
        # reports the pair as equal. Only a full 64-bit widening says False.
        a = torch.tensor([2**40 + 7, -(2**40) - 7, 7], dtype=torch.int64, device="neuron")
        b = torch.tensor([7, -7, 7], dtype=torch.int32, device="neuron")

        result, _, _ = self._track(lambda: torch.eq(a, b))

        torch.testing.assert_close(result.cpu(), torch.tensor([False, False, True]))

    def test_promotion_is_exact_in_arithmetic(self):
        a = torch.tensor([2**40, -(2**40)], dtype=torch.int64, device="neuron")
        b = torch.tensor([7, -7], dtype=torch.int32, device="neuron")

        result, _, _ = self._track(lambda: a + b)

        expected = torch.tensor([2**40 + 7, -(2**40) - 7], dtype=torch.int64)
        assert result.dtype == torch.int64
        torch.testing.assert_close(result.cpu(), expected)


@requires_neuron
@pytest.mark.int64_mode("native")
class TestInt64CopyStaysOnDevice:
    """A device-to-device int64 copy does not go through the host.

    This is a regression test for a fixed bug. aten::copy_ had no on-device tier
    for int64. IsCopyMlirDtypeSupported() rejects Long, so the copy reached the
    last tier, which is the NRT CPU bounce. A same-dtype copy only moves bytes,
    so it needs no dtype support in the compiler.
    """

    @staticmethod
    def _host_read_backs(run):
        """Run the copy and report the host read-backs.

        A read back to the host marks the round-trip. An upload alone does not,
        because a helper op can upload a scalar operand. The first synchronize
        drains the lazy tensor creation, so only the copy is measured.
        """
        torch.neuron.synchronize()
        torch_neuronx.clear_op_tracking()
        run()
        torch.neuron.synchronize()
        return [op for op in torch_neuronx.get_executed_ops() if "neuron_to_cpu" in op]

    def test_slice_assign_stays_on_device(self):
        src = torch.tensor([11, 22, 33, 44], dtype=torch.int64, device="neuron")
        dst = torch.zeros(8, dtype=torch.int64, device="neuron")

        def run():
            dst[: src.numel()] = src

        host = self._host_read_backs(run)

        assert not host, f"the copy went through the host; ops={host}"
        expected = torch.tensor([11, 22, 33, 44, 0, 0, 0, 0], dtype=torch.int64)
        torch.testing.assert_close(dst.cpu(), expected)

    def test_copy_stays_on_device(self):
        src = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="neuron")
        dst = torch.zeros(4, dtype=torch.int64, device="neuron")

        host = self._host_read_backs(lambda: dst.copy_(src))

        assert not host, f"the copy went through the host; ops={host}"
        torch.testing.assert_close(dst.cpu(), src.cpu())

    def test_copy_keeps_the_high_word(self):
        # A byte copy must hold values that need more than 32 bits.
        values = [2**62, -(2**62), 2**40 + 7, -(2**40) - 7]
        src = torch.tensor(values, dtype=torch.int64, device="neuron")
        dst = torch.zeros(len(values), dtype=torch.int64, device="neuron")

        dst.copy_(src)
        torch.neuron.synchronize()

        torch.testing.assert_close(dst.cpu(), torch.tensor(values, dtype=torch.int64))


# ═══════════════════════════════════════════════════════════════
#  packed mode — every int64 op via the packed view
# ═══════════════════════════════════════════════════════════════


@requires_neuron
@pytest.mark.int64_mode("packed")
class TestPackedMode:
    """In packed mode all int64 ops run on device via the low-word packed view;
    correct within int32 range."""

    def test_add_int64(self):
        a = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device="neuron")
        b = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="neuron")
        result = a + b
        torch.neuron.synchronize()
        torch.testing.assert_close(result.cpu(), torch.tensor([11, 22, 33, 44], dtype=torch.int64))

    def test_pow_int64(self):
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

    def test_add_int64(self):
        a = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device="neuron")
        b = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="neuron")
        result = a + b
        torch.neuron.synchronize()
        torch.testing.assert_close(result.cpu(), torch.tensor([11, 22, 33, 44], dtype=torch.int64))

    def test_pow_int64(self):
        """The dispatcher must downcast int64 before lowering; otherwise
        neuronx-cc sees S64 pow with no int64 flag and fails compilation."""
        x = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device="neuron")
        result = torch.pow(x, 2)
        torch.neuron.synchronize()
        expected = torch.tensor([100, 400, 900, 1600], dtype=torch.int64)
        torch.testing.assert_close(result.cpu(), expected)

    def test_pow_int64_out(self):
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

    def test_pow_cpu_fallback(self):
        a = torch.tensor([2, 3, 4, 5], dtype=torch.int64, device="neuron")
        result = torch.pow(a, 2)
        torch.neuron.synchronize()
        torch.testing.assert_close(result.cpu(), torch.tensor([4, 9, 16, 25], dtype=torch.int64))

    def test_native_op_runs_on_neuron(self):
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

    def test_add_runs_on_cpu(self):
        a = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device="neuron")
        b = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="neuron")
        torch_neuronx.clear_op_tracking()
        result = a + b
        torch.neuron.synchronize()
        fb = set(torch_neuronx.get_fallback_ops())
        assert any("add" in op for op in fb), f"add should CPU-fall-back in cpu_fallback; fb={fb}"
        torch.testing.assert_close(result.cpu(), torch.tensor([11, 22, 33, 44], dtype=torch.int64))

    def test_int64_above_int32_max_is_exact(self):
        # Running on CPU, full int64 precision is preserved regardless of compiler.
        big = (1 << 40) + 7
        a = torch.tensor([big, big, big], dtype=torch.int64, device="neuron")
        b = torch.tensor([1, 2, 3], dtype=torch.int64, device="neuron")
        result = a + b
        torch.neuron.synchronize()
        expected = torch.tensor([big + 1, big + 2, big + 3], dtype=torch.int64)
        torch.testing.assert_close(result.cpu(), expected)


# ═══════════════════════════════════════════════════════════════════════════
#  CPU fallback aliasing contract
#
#  A CPU fallback must honor a caller's out= tensor, must mutate self for an
#  inplace op, and must return multi-output results on the neuron device.
#  at::native::cpu_fallback in NeuronDispatcher.cpp does all three; these pin
#  it. They use ops that actually CPU-fall-back:
#    * aten::mm (UNSUPPORTED) in native mode, and
#    * non-native ops (pow, sort, max.dim) in native_strict mode.
# ═══════════════════════════════════════════════════════════════════════════


@requires_neuron
@pytest.mark.int64_mode("native")
class TestCpuFallbackOutKwarg:
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


@requires_neuron
@pytest.mark.int64_mode("native_strict")
class TestCpuFallbackOutKwargNativeStrict:
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


@requires_neuron
@pytest.mark.int64_mode("native_strict")
class TestCpuFallbackInplace:
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


@requires_neuron
@pytest.mark.int64_mode("native_strict")
class TestCpuFallbackMultiReturn:
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
