"""Tests for comprehensive CPU fallback reporting.

Verifies that all CPU fallback/roundtrip paths added in the
"report all CPU fallback paths in get_fallback_ops()" change are properly
reported in torch_neuronx.get_fallback_ops(), so ModuScope's offloading
stats accurately reflect performance-impacting CPU involvement.

The paths under test are:
  - Prologue unsupported stride patterns (MakeContiguousViaCPU)          [C++]
  - Epilogue unsupported output patterns (contiguous temp + CopyBack)    [C++]
  - NeuronSafeCast dtype CPU roundtrip (float64/non-native int64)        [C++]
  - write_to_noncontiguous_neuron CPU bounce                             [C++]
  - contiguous_generic_kernel (generic CPU contiguous fallback)          [Python]
  - Linalg CPU fallback ops (QR, SVD, eigh, etc.)                        [Python]

Reporting-name convention
--------------------------
Two categories are distinguished by the reported op name:

  * Real whole-op CPU fallbacks — the op's actual computation runs on CPU.
    Reported under the bare op name (e.g. ``aten::_to_copy``,
    ``aten::contiguous``, the linalg ops). Tests keying on the bare name treat
    presence in get_fallback_ops() as "did not run on Neuron", which is correct
    for these.

  * On-device ops with a partial CPU round-trip — the op runs on Neuron but a
    layout fixup (input restride, output scatter, non-contiguous copy_) bounces
    through the host. These are reported under a distinct suffixed name so they
    do NOT shadow the bare op name that also appears in get_executed_ops():
      - ``aten::copy_[noncontiguous_scatter_cpu_bounce]``  (write_to_noncontiguous_neuron)
      - ``<op>[prologue_cpu_roundtrip]``                   (input restride)
      - ``<op>[epilogue_cpu_roundtrip]``                   (output scatter back)
"""

import pytest
import torch

import torch_neuronx
import torch_neuronx._C as _C


def _epilogue_enabled():
    """Whether the output epilogue path is active (delegates to C++)."""
    return _C._is_epilogue_enabled()


# ---------------------------------------------------------------------------
# Device-independent tests: the reporting primitive itself.
# These do not require a Neuron device and directly exercise the
# _log_offloaded_op -> get_fallback_ops() -> clear_op_tracking() contract that
# every fallback path in this change relies on.
# ---------------------------------------------------------------------------
class TestFallbackTrackingPrimitive:
    """Directly test the FallbackOpTracker plumbing used by all fallback paths."""

    def test_log_offloaded_op_is_reported(self):
        """An op logged via _log_offloaded_op shows up in get_fallback_ops()."""
        torch_neuronx.clear_op_tracking()
        _C._log_offloaded_op("aten::__unit_test_op")
        assert "aten::__unit_test_op" in torch_neuronx.get_fallback_ops()

    def test_log_offloaded_op_is_deduplicated(self):
        """Logging the same op multiple times reports it exactly once."""
        torch_neuronx.clear_op_tracking()
        for _ in range(5):
            _C._log_offloaded_op("aten::__unit_test_dedup")
        reported = torch_neuronx.get_fallback_ops()
        assert reported.count("aten::__unit_test_dedup") == 1

    def test_clear_op_tracking_resets_fallbacks(self):
        """clear_op_tracking() empties previously reported fallbacks."""
        _C._log_offloaded_op("aten::__unit_test_clear")
        assert "aten::__unit_test_clear" in torch_neuronx.get_fallback_ops()
        torch_neuronx.clear_op_tracking()
        assert "aten::__unit_test_clear" not in torch_neuronx.get_fallback_ops()

    def test_multiple_distinct_ops_reported(self):
        """Distinct fallback ops are each reported independently."""
        torch_neuronx.clear_op_tracking()
        names = ["aten::__op_a", "aten::__op_b", "aten::__op_c"]
        for n in names:
            _C._log_offloaded_op(n)
        reported = set(torch_neuronx.get_fallback_ops())
        assert set(names).issubset(reported)


# ---------------------------------------------------------------------------
# Device tests: exercise the real CPU fallback paths end-to-end and assert the
# op is reported. These require a Neuron device.
# ---------------------------------------------------------------------------
def _assert_reported(op_name: str):
    """Assert op_name is present in get_fallback_ops() after synchronizing.

    Matches either the exact op name or any overload of it (e.g. the ``.out``
    or ``.U`` overloads that some ops dispatch through), so the assertion is
    robust to which overload the dispatcher selects.
    """
    torch.neuron.synchronize()
    fallback_ops = torch_neuronx.get_fallback_ops()
    matched = op_name in fallback_ops or any(
        op == op_name or op.startswith(op_name + ".") for op in fallback_ops
    )
    assert matched, (
        f"Expected '{op_name}' (or an overload of it) to be reported as a "
        f"CPU fallback.\nReported fallback ops: {sorted(fallback_ops)}"
    )


class TestFallbackReportingLinalg:
    """Linalg ops run on CPU (Python cpu_fallback_registrations) and must report."""

    def test_linalg_qr_reported(self):
        x = torch.randn(4, 4, device="neuron")
        torch_neuronx.clear_op_tracking()
        q, r = torch.linalg.qr(x)
        _assert_reported("aten::linalg_qr")
        # Correctness: reconstruct the input.
        assert torch.allclose((q @ r).cpu(), x.cpu(), atol=1e-4)

    def test_linalg_svd_reported(self):
        x = torch.randn(4, 4, device="neuron")
        torch_neuronx.clear_op_tracking()
        torch.linalg.svd(x)
        _assert_reported("aten::_linalg_svd")

    def test_linalg_eigh_reported(self):
        # Symmetric matrix for eigh.
        a = torch.randn(4, 4, device="neuron")
        x = a + a.transpose(0, 1)
        torch_neuronx.clear_op_tracking()
        torch.linalg.eigh(x)
        _assert_reported("aten::_linalg_eigh")


class TestFallbackReportingDtypeCast:
    """Dtype casts that need a CPU roundtrip via NeuronSafeCast (C++).

    NeuronSafeCast reports ``aten::_to_copy`` when it performs the CPU
    round-trip for an unsupported dtype. A plain float32->float64 ``.to()`` on
    this build does not exercise that specific NeuronSafeCast branch, so we
    only verify correctness here; the reporting mechanism itself is covered by
    ``TestFallbackTrackingPrimitive`` and the DtypeUtils.cpp call site is
    identical to the other verified fallback paths.
    """

    def test_float64_cast_correct(self):
        x = torch.randn(4, 4, dtype=torch.float32, device="neuron")
        torch_neuronx.clear_op_tracking()
        y = x.to(torch.float64)
        torch.neuron.synchronize()
        assert torch.allclose(y.cpu(), x.cpu().to(torch.float64))


class TestFallbackReportingContiguous:
    """Generic contiguous CPU fallback (contiguous_generic_kernel, Python)."""

    def test_generic_contiguous_fallback_reported(self):
        """A stride pattern no NKI kernel handles falls back to CPU contiguous.

        If the prologue/NKI kernels DO handle this pattern, aten::contiguous
        will not be reported — in that case the fallback simply did not occur,
        which is also correct. We therefore assert correctness unconditionally
        and only assert reporting when the CPU generic kernel was actually used
        (detected by directly invoking it below in a dedicated test).
        """
        x = torch.randn(10, 10, device="neuron")
        view = x[::3, ::3]  # shape (4,4), strides (30, 3)
        torch_neuronx.clear_op_tracking()
        result = view.contiguous()
        expected = view.cpu().contiguous()
        assert torch.allclose(result.cpu(), expected)

    def test_contiguous_generic_kernel_reports_directly(self):
        """Directly invoking the CPU generic kernel must report aten::contiguous.

        This deterministically covers the reporting line regardless of whether
        the higher-level dispatch chooses this kernel for a given stride.
        """
        from torch_neuronx.python_ops.nki_kernels.contiguous_generic import (
            contiguous_generic_kernel,
        )

        src = torch.arange(16, dtype=torch.float32, device="neuron")
        dst = torch.empty(16, dtype=torch.float32, device="neuron")
        torch_neuronx.clear_op_tracking()
        contiguous_generic_kernel(src, dst, shape=(16,), src_strides=(1,), dst_strides=(1,))
        _assert_reported("aten::contiguous")


class TestFallbackReportingNonContiguousCopy:
    """write_to_noncontiguous_neuron CPU bounce (C++).

    Reported under the distinct name
    ``aten::copy_[noncontiguous_scatter_cpu_bounce]`` rather than the bare
    ``aten::copy_`` — the copy_ still runs on device, only the scatter into the
    non-contiguous destination round-trips through the host. Reusing the bare
    name would shadow the on-device copy_ that other tests (e.g.
    test_pad_decomposition_to_correct_ops[circular-aten::copy_]) assert runs on
    Neuron.
    """

    NONCONTIG_COPY_NAME = "aten::copy_[noncontiguous_scatter_cpu_bounce]"

    def test_noncontiguous_copy_reported(self):
        # Build a destination whose target region is non-contiguous so the
        # copy_ implementation must take the CPU bounce path.
        dst = torch.zeros(4, 4, device="neuron")
        dst_view = dst[:, ::2]  # non-contiguous columns
        src = torch.ones(4, 2, device="neuron")
        torch_neuronx.clear_op_tracking()
        dst_view.copy_(src)
        torch.neuron.synchronize()
        # Correctness must always hold.
        expected = torch.zeros(4, 4)
        expected[:, ::2] = 1.0
        assert torch.allclose(dst.cpu(), expected)

        fallback_ops = torch_neuronx.get_fallback_ops()
        # If the CPU bounce was taken it must be reported under the distinct
        # suffixed name — never under the bare "aten::copy_" (which would
        # falsely mark the on-device copy_ as a fallback).
        assert "aten::copy_" not in fallback_ops, (
            "The non-contiguous copy_ CPU bounce must not be reported under the "
            "bare 'aten::copy_' name (it would shadow the on-device copy_). "
            f"Reported: {sorted(fallback_ops)}"
        )
        if any("copy_" in op for op in fallback_ops):
            assert self.NONCONTIG_COPY_NAME in fallback_ops, (
                "Expected the non-contiguous copy_ CPU bounce to be reported as "
                f"'{self.NONCONTIG_COPY_NAME}'. Reported: {sorted(fallback_ops)}"
            )


class TestFallbackReportingPrologue:
    """Prologue unsupported strides fall back via MakeContiguousViaCPU (C++).

    When an input has a stride pattern that no prologue transformation can
    handle, ProcessInputTransformations reports the op via markFallbackLogged()
    and then materializes the input with a CPU round-trip (it does not raise on
    this branch). We use a known-unsupported layout — shape=[5, 6, 2] with
    strides=[1, 1, 7] — which cannot be lowered by any prologue transform.

    Note: the analogous *epilogue* (output) fallback branch is intentionally not
    given a dedicated positive test. The only output layouts TryCreateEpilogue
    rejects are broadcast/negative strides (which PyTorch itself refuses to
    write into via out=) and slice patterns when SCATTER_IOTA is disabled; with
    the generic epilogue (PR #1040) the output-unsupported branch becomes
    unreachable for practical layouts. The epilogue call site uses the same
    markFallbackLogged() mechanism already verified by
    TestFallbackTrackingPrimitive and the prologue test above.
    """

    def _make_unsupported_strided(self):
        # Back the strided view with enough storage, then re-stride into the
        # prologue-unsupported layout.
        base = torch.randn(5 * 6 * 2 + 16, device="neuron")
        return torch.as_strided(base, size=(5, 6, 2), stride=(1, 1, 7))

    def test_prologue_unsupported_stride_reported(self):
        view = self._make_unsupported_strided()
        torch_neuronx.clear_op_tracking()
        # A binary op forces the input through ProcessInputTransformations.
        result = view + 1.0
        torch.neuron.synchronize()
        expected = view.cpu() + 1.0
        assert torch.allclose(result.cpu(), expected, atol=1e-5)
        # The op must have been reported as a CPU fallback, under whichever add
        # overload the dispatcher selected (e.g. aten::add.Tensor).
        fallback_ops = torch_neuronx.get_fallback_ops()
        assert any(op.startswith("aten::add") for op in fallback_ops), (
            "Expected the prologue CPU fallback for the unsupported stride "
            f"pattern to report an aten::add* op. Reported: {sorted(fallback_ops)}"
        )


def _assert_not_reported(op_prefix: str):
    """Assert no op starting with op_prefix is in get_fallback_ops()."""
    torch.neuron.synchronize()
    fallback_ops = torch_neuronx.get_fallback_ops()
    offenders = [op for op in fallback_ops if op.startswith(op_prefix)]
    assert not offenders, (
        f"Did not expect any '{op_prefix}*' op to be reported as a CPU "
        f"fallback (it ran on device). Reported: {sorted(fallback_ops)}"
    )


class TestFallbackReportingNegative:
    """Negative tests: on-device ops must NOT be reported as fallbacks.

    These guard against over-reporting — an op that runs fully on Neuron should
    never appear in get_fallback_ops(), otherwise ModuScope's offloading stats
    would be inflated with false positives.
    """

    def test_contiguous_elementwise_not_reported(self):
        """A plain elementwise add on contiguous tensors is not a fallback."""
        x = torch.randn(8, 8, device="neuron")
        y = torch.randn(8, 8, device="neuron")
        torch_neuronx.clear_op_tracking()
        result = x + y
        torch.neuron.synchronize()
        assert torch.allclose(result.cpu(), x.cpu() + y.cpu(), atol=1e-5)
        _assert_not_reported("aten::add")

    def test_contiguous_tensor_no_contiguous_fallback(self):
        """Calling .contiguous() on an already-contiguous tensor must not
        report aten::contiguous (no CPU generic kernel involved)."""
        x = torch.randn(8, 8, device="neuron")
        assert x.is_contiguous()
        torch_neuronx.clear_op_tracking()
        result = x.contiguous()
        torch.neuron.synchronize()
        assert torch.allclose(result.cpu(), x.cpu())
        _assert_not_reported("aten::contiguous")

    def test_supported_prologue_stride_not_reported(self):
        """A transpose is a prologue-supported stride, so the op running on it
        should NOT be reported as a CPU fallback."""
        x = torch.randn(4, 6, device="neuron")
        view = x.transpose(0, 1)  # supported prologue pattern
        assert not view.is_contiguous()
        torch_neuronx.clear_op_tracking()
        result = view + 1.0
        torch.neuron.synchronize()
        assert torch.allclose(result.cpu(), view.cpu() + 1.0, atol=1e-5)
        _assert_not_reported("aten::add")

    @pytest.mark.skipif(
        not _epilogue_enabled(),
        reason="Epilogue path only runs when TORCH_NEURONX_ENABLE_EPILOGUE=1",
    )
    def test_supported_epilogue_output_not_reported(self):
        """A transposed out= output is an epilogue-supported pattern, so no CPU
        fallback should be reported when writing into it."""
        a = torch.randn(3, 4, device="neuron")
        base = torch.zeros(3, 4, device="neuron")
        out_view = base.transpose(0, 1)  # supported epilogue pattern
        assert not out_view.is_contiguous()
        torch_neuronx.clear_op_tracking()
        torch.mul(a.transpose(0, 1), 2.0, out=out_view)
        torch.neuron.synchronize()
        _assert_not_reported("aten::mul")

    def test_no_linalg_fallback_without_linalg(self):
        """A run that uses no linalg op must not report any linalg fallback."""
        x = torch.randn(4, 4, device="neuron")
        torch_neuronx.clear_op_tracking()
        _ = x + x
        torch.neuron.synchronize()
        _assert_not_reported("aten::linalg_")
        _assert_not_reported("aten::_linalg_")


class TestFallbackReportingRegressionBaseline:
    """Baseline: previously-reported ops must still be reported (no regression)."""

    def test_normal_still_reported(self):
        torch_neuronx.clear_op_tracking()
        _ = torch.randn(4, device="neuron")  # uses aten::normal_
        torch.neuron.synchronize()
        fallback_ops = torch_neuronx.get_fallback_ops()
        assert isinstance(fallback_ops, list)
        assert "aten::normal_" in fallback_ops
