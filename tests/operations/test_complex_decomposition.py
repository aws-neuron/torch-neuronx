"""Integration tests for framework complex decomposition path."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest
import torch

import torch_neuronx  # registers backend
from torch_neuronx.neuron_dynamo_backend.fx.passes.complex_decomposition import (
    _COMPLEX_DECOMP_AVAILABLE,
)

pytestmark = pytest.mark.skipif(
    not _COMPLEX_DECOMP_AVAILABLE,
    reason="Complex decomposition requires PyTorch >= 2.13",
)


# Helpers


def _run_decomp_subprocess(snippet: str) -> subprocess.CompletedProcess:
    """Run snippet with framework decomposition enabled in a subprocess."""
    env = os.environ.copy()
    env["TORCH_NEURONX_COMPLEX_DECOMPOSE"] = "1"
    env.pop("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS", None)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(snippet)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _assert_subprocess_ok(result: subprocess.CompletedProcess, op_name: str) -> None:
    """Assert subprocess completed successfully with PASS marker."""
    if result.returncode != 0:
        pytest.fail(
            f"{op_name} decomposition test failed (rc={result.returncode}):\n"
            f"STDOUT:\n{result.stdout[-1000:]}\n"
            f"STDERR:\n{result.stderr[-1000:]}"
        )
    assert (
        "PASS" in result.stdout
    ), f"{op_name}: subprocess completed but PASS marker missing.\nSTDOUT:\n{result.stdout[-500:]}"


def _reset_op_tracking() -> None:
    """Drain queued ops, then reset tracking so only the next op is observed."""
    torch.neuron.synchronize()
    torch_neuronx.clear_op_tracking()


def _assert_not_cpu_fallback(op_prefix: str) -> None:
    """Assert no op named ``op_prefix*`` was reported as a CPU fallback.

    Each relevant C++ fallback branch calls ``markFallbackLogged`` immediately
    before ``at::native::cpu_fallback``, so an empty result means that branch
    was never reached.
    """
    torch.neuron.synchronize()
    fallback_ops = torch_neuronx.get_fallback_ops()
    assert not any(
        op.startswith(op_prefix) for op in fallback_ops
    ), f"{op_prefix} fell back to CPU; fallback ops: {sorted(fallback_ops)}"


# Eager dispatch path tests


class TestComplexDecompEagerAdd:
    """complex add via framework decomposition (eager path)."""

    def test_add_tensor(self):
        snippet = """
            import torch, torch_neuronx
            x = torch.randn(4, 4, dtype=torch.complex64)
            y = torch.randn(4, 4, dtype=torch.complex64)
            result = (x.to("neuron") + y.to("neuron")).cpu()
            expected = x + y
            assert torch.allclose(result, expected, atol=1e-5, rtol=1e-4), (
                f"Max diff: {(result - expected).abs().max()}"
            )
            print("PASS")
        """
        result = _run_decomp_subprocess(snippet)
        _assert_subprocess_ok(result, "add.Tensor")


class TestComplexDecompEagerSub:
    """complex sub via framework decomposition (eager path)."""

    def test_sub_tensor(self):
        snippet = """
            import torch, torch_neuronx
            x = torch.randn(4, 4, dtype=torch.complex64)
            y = torch.randn(4, 4, dtype=torch.complex64)
            result = (x.to("neuron") - y.to("neuron")).cpu()
            expected = x - y
            assert torch.allclose(result, expected, atol=1e-5, rtol=1e-4), (
                f"Max diff: {(result - expected).abs().max()}"
            )
            print("PASS")
        """
        result = _run_decomp_subprocess(snippet)
        _assert_subprocess_ok(result, "sub.Tensor")


# torch.compile path tests


class TestComplexDecompCompileAdd:
    """complex add via framework decomposition (torch.compile path)."""

    def test_compile_add(self):
        snippet = """
            import torch, torch_neuronx
            @torch.compile(backend="neuron")
            def fn(x, y):
                return x + y

            x = torch.randn(4, 4, dtype=torch.complex64, device="neuron")
            y = torch.randn(4, 4, dtype=torch.complex64, device="neuron")
            result = fn(x, y).cpu()
            expected = x.cpu() + y.cpu()
            assert torch.allclose(result, expected, atol=1e-5, rtol=1e-4), (
                f"Max diff: {(result - expected).abs().max()}"
            )
            print("PASS")
        """
        result = _run_decomp_subprocess(snippet)
        _assert_subprocess_ok(result, "compile:add")


class TestComplexDecompCompileSub:
    """complex sub via framework decomposition (torch.compile path)."""

    def test_compile_sub(self):
        snippet = """
            import torch, torch_neuronx
            @torch.compile(backend="neuron")
            def fn(x, y):
                return x - y

            x = torch.randn(4, 4, dtype=torch.complex64, device="neuron")
            y = torch.randn(4, 4, dtype=torch.complex64, device="neuron")
            result = fn(x, y).cpu()
            expected = x.cpu() - y.cpu()
            assert torch.allclose(result, expected, atol=1e-5, rtol=1e-4), (
                f"Max diff: {(result - expected).abs().max()}"
            )
            print("PASS")
        """
        result = _run_decomp_subprocess(snippet)
        _assert_subprocess_ok(result, "compile:sub")


class TestComplexDecompCompileWithNKIKernel:
    """Complex ops and a NKI kernel in the same compiled graph (llama3's shape).

    Regression test for the missing `WrapComplexMode` rule on the `nki_kernel_wrapper` HOP.
    The all-complex graphs above do not cover it.
    """

    def test_compile_complex_rope_with_nki_attention(self):
        snippet = """
            import torch, torch_neuronx
            import torch.nn.functional as F

            BS, SEQ, HEADS, DIM = 1, 512, 4, 64  # SEQ >= 512 keeps SDPA on the NKI kernel path

            def rope_then_attention(xq, xk, xv, freqs_cis):
                # llama3 RoPE: real -> complex -> multiply -> back to real, then NKI attention.
                xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
                xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
                freqs = freqs_cis[: xq_.shape[1]].view(1, xq_.shape[1], 1, xq_.shape[-1])
                q = torch.view_as_real(xq_ * freqs).flatten(3).type_as(xq)
                k = torch.view_as_real(xk_ * freqs).flatten(3).type_as(xk)
                return F.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), xv.transpose(1, 2), is_causal=True
                )

            freqs = torch.outer(
                torch.arange(SEQ, dtype=torch.float32),
                1.0 / (10000 ** (torch.arange(0, DIM, 2).float() / DIM)),
            )
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs).to("neuron")
            xq, xk, xv = (
                torch.randn(
                    BS, SEQ, HEADS, DIM, dtype=torch.bfloat16, device="neuron", requires_grad=True
                )
                for _ in range(3)
            )

            out = torch.compile(rope_then_attention, backend="neuron")(xq, xk, xv, freqs_cis)
            assert out.shape == (BS, HEADS, SEQ, DIM), out.shape
            out.float().sum().backward()
            assert xq.grad is not None and torch.isfinite(xq.grad.float()).all()
            print("PASS")
        """
        result = _run_decomp_subprocess(snippet)
        _assert_subprocess_ok(result, "compile:complex_rope+nki_attention")


# Version guard tests (no hardware needed)


class TestComplexDecompVersionGuard:
    """Verify version guard logic (does not require Neuron hardware)."""

    def test_availability_flag_matches_torch_version(self):
        """_COMPLEX_DECOMP_AVAILABLE is True iff torch >= 2.13."""
        from packaging.version import Version

        expected = Version(torch.__version__) >= Version("2.13.0.dev0")
        assert expected == _COMPLEX_DECOMP_AVAILABLE

    def test_maybe_decompose_returns_graph_unchanged_when_disabled(self):
        """When env var is not set, the graph is returned as-is."""
        from unittest.mock import MagicMock

        from torch_neuronx.neuron_dynamo_backend.fx.passes.complex_decomposition import (
            maybe_decompose_complex,
        )

        gm = MagicMock()
        # Ensure the env var is not set
        env_backup = os.environ.pop("TORCH_NEURONX_COMPLEX_DECOMPOSE", None)
        try:
            result = maybe_decompose_complex(gm, [])
            assert result is gm
        finally:
            if env_backup is not None:
                os.environ["TORCH_NEURONX_COMPLEX_DECOMPOSE"] = env_backup


class TestStructuralComplexDecomposition:
    """Structural complex ops execute on Neuron through the interleaved-real path.

    These run in-process rather than in a subprocess like the rest of the file:
    ComplexDecompositionEnabled() re-reads the env var on every call, so
    monkeypatch is enough.
    """

    @pytest.fixture(autouse=True)
    def _enable_complex_decomposition(self, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_COMPLEX_DECOMPOSE", "1")

    def test_contiguous_stays_on_device(self):
        base_cpu = torch.randn(4, 6, dtype=torch.complex64)
        x_cpu = base_cpu.transpose(0, 1)
        x = base_cpu.to("neuron").transpose(0, 1)
        assert not x.is_contiguous()

        _reset_op_tracking()
        result = x.contiguous()

        _assert_not_cpu_fallback("aten::contiguous")
        assert result.is_contiguous()
        torch.testing.assert_close(result.cpu(), x_cpu.contiguous())

    @pytest.mark.skipif(
        os.environ.get("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS") != "1",
        reason="complex CPU fallback segfaults for conj/neg views, independently of this path",
    )
    @pytest.mark.parametrize("flag", ["conj", "neg"])
    def test_conjugate_or_negative_view_is_declined(self, flag):
        """A flagged complex view must not reach view_as_real.

        Whatever the decline lands on is the ordinary complex route, identical
        with TORCH_NEURONX_COMPLEX_DECOMPOSE=0, so this asserts only that the
        structural path is not the thing that fails and that nothing silently
        returns a wrong value.
        """
        make_flagged = torch.Tensor.conj if flag == "conj" else torch._neg_view
        x_cpu = torch.randn(4, 5, dtype=torch.complex64)
        index_cpu = torch.tensor([3, 1], dtype=torch.int64)
        expected = torch.index_select(make_flagged(x_cpu.transpose(0, 1)), 0, index_cpu)

        flagged = make_flagged(x_cpu.to("neuron").transpose(0, 1))
        assert flagged.is_conj() or flagged.is_neg()
        assert not flagged.is_contiguous()

        try:
            result = torch.index_select(flagged, 0, index_cpu.to("neuron"))
        except RuntimeError as error:
            assert "view_as_real" not in str(
                error
            ), f"structural path did not decline the {flag} view: {error}"
        else:
            torch.testing.assert_close(result.cpu(), expected)

    def test_gather_stays_on_device(self):
        x_cpu = torch.randn(4, 8, dtype=torch.complex64)
        index_cpu = torch.tensor([[2, 0, 1], [1, 3, 0]], dtype=torch.int64)
        x = x_cpu.to("neuron")
        index = index_cpu.to("neuron")

        _reset_op_tracking()
        result = torch.gather(x, 1, index)

        _assert_not_cpu_fallback("aten::gather")
        torch.testing.assert_close(result.cpu(), torch.gather(x_cpu, 1, index_cpu))

    def test_index_select_stays_on_device(self):
        x_cpu = torch.randn(8, 4, dtype=torch.complex64)
        index_cpu = torch.tensor([2, 0], dtype=torch.int64)
        x = x_cpu.to("neuron")
        index = index_cpu.to("neuron")

        _reset_op_tracking()
        result = torch.index_select(x, 0, index)

        _assert_not_cpu_fallback("aten::index_select")
        torch.testing.assert_close(result.cpu(), torch.index_select(x_cpu, 0, index_cpu))

    def test_index_tensor_stays_on_device(self):
        x_cpu = torch.randn(8, 4, dtype=torch.complex64)
        index_cpu = torch.tensor([2, 0], dtype=torch.int64)
        x = x_cpu.to("neuron")
        index = index_cpu.to("neuron")

        _reset_op_tracking()
        result = x[index]

        _assert_not_cpu_fallback("aten::index")
        torch.testing.assert_close(result.cpu(), x_cpu[index_cpu])

    # ── dim / rank guards ──
    #
    # view_as_real appends a trailing [real, imag] axis, so a dim out of range for
    # the complex tensor is in range for the real view, where it would address that
    # component axis. What matters below is that nothing silently returns a value
    # built from it. The exception type varies by op because each raises wherever
    # the generic path first validates dim.

    @pytest.mark.parametrize("dim", [2, -3])
    def test_gather_out_of_range_dim_raises(self, dim):
        x_cpu = torch.randn(1, 2, dtype=torch.complex64)
        index_cpu = torch.zeros(1, 2, dtype=torch.int64)
        with pytest.raises(IndexError):
            torch.gather(x_cpu, dim, index_cpu)

        x = x_cpu.to("neuron")
        index = index_cpu.to("neuron")
        # AssertionError comes from maybe_wrap_dim inside meta_gather.
        with pytest.raises((IndexError, RuntimeError, AssertionError)):
            torch.gather(x, dim, index)

    @pytest.mark.parametrize("dim", [2, -3])
    def test_index_select_out_of_range_dim_raises(self, dim):
        x_cpu = torch.randn(1, 2, dtype=torch.complex64)
        index_cpu = torch.tensor([0], dtype=torch.int64)
        with pytest.raises(IndexError):
            torch.index_select(x_cpu, dim, index_cpu)

        x = x_cpu.to("neuron")
        index = index_cpu.to("neuron")
        with pytest.raises((IndexError, RuntimeError)):
            torch.index_select(x, dim, index)

    @pytest.mark.parametrize("dim", [0, -1])
    def test_gather_scalar_matches_cpu(self, dim):
        """A 0-dim input has no axis besides [real, imag], so the structural path
        declines it. With TORCH_NEURONX_COMPLEX_DECOMPOSE=1 the generic path
        decomposes the op into one real gather per component, and the result is
        the complete complex value. The wrong structural rewrite returned 1+1j.
        """
        x_cpu = torch.tensor(1 + 2j, dtype=torch.complex64)
        index_cpu = torch.tensor(0, dtype=torch.int64)
        expected = torch.gather(x_cpu, dim, index_cpu)
        torch.testing.assert_close(expected, x_cpu)

        result = torch.gather(x_cpu.to("neuron"), dim, index_cpu.to("neuron"))
        torch_neuronx.synchronize()
        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.xfail(
        reason="Needs the nki_kernel_wrapper HOP rule from the NKI-HOP fix "
        "(fix/nki-hop-complex-decompose, PR #1605), which registers "
        "@nki_kernel_wrapper.py_impl(WrapComplexMode) and therefore needs PyTorch >= 2.13 "
        "where WrapComplexMode exists. A 0-dim input declines the structural path, so the "
        "generic path retraces under WrapComplexMode; the lowered graph carries the "
        "nki_kernel_wrapper HOP, which has no mode rule. Only the second lowering in a "
        "process emits the HOP, so whichever dim runs first XPASSes.",
    )
    @pytest.mark.parametrize("dim", [0, -1])
    def test_index_select_scalar_matches_cpu(self, dim):
        x_cpu = torch.tensor(1 + 2j, dtype=torch.complex64)
        index_cpu = torch.tensor([0], dtype=torch.int64)
        expected = torch.index_select(x_cpu, dim, index_cpu)

        result = torch.index_select(x_cpu.to("neuron"), dim, index_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), expected)

    def test_index_too_many_indices_raises(self):
        """More index entries than dimensions must raise, not index [real, imag]."""
        x_cpu = torch.randn(3, dtype=torch.complex64)
        index_cpu = torch.tensor([0], dtype=torch.int64)
        with pytest.raises(IndexError):
            _ = x_cpu[index_cpu, index_cpu]

        x = x_cpu.to("neuron")
        index = index_cpu.to("neuron")
        with pytest.raises((IndexError, RuntimeError)):
            _ = x[index, index]

    def test_index_scalar_raises(self):
        x_cpu = torch.tensor(1 + 2j, dtype=torch.complex64)
        index_cpu = torch.tensor([0], dtype=torch.int64)
        with pytest.raises(IndexError):
            _ = x_cpu[index_cpu]

        x = x_cpu.to("neuron")
        index = index_cpu.to("neuron")
        with pytest.raises((IndexError, RuntimeError)):
            _ = x[index]


class TestComplexIndexEmptyResultShape:
    """Zero-size advanced indexing keeps PyTorch's result shape.

    DispatchIndexImpl short-circuits an all-false mask without executing the
    kernel, so it has to synthesize the shape rather than observe it.
    """

    @pytest.fixture(autouse=True)
    def _enable_complex_decomposition(self, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_COMPLEX_DECOMPOSE", "1")

    def test_all_false_mask_keeps_trailing_dims(self):
        """(8, 4) indexed by an all-false (8,) mask is (0, 4), not (0,)."""
        x_cpu = torch.randn(8, 4, dtype=torch.complex64)
        mask_cpu = torch.zeros(8, dtype=torch.bool)
        expected = x_cpu[mask_cpu]

        result = x_cpu.to("neuron")[mask_cpu.to("neuron")]

        assert result.shape == expected.shape
        assert result.dtype == expected.dtype
        torch.testing.assert_close(result.cpu(), expected)

    def test_all_false_mask_on_middle_dim(self):
        """A mask on dim 1 of (2, 3, 4) keeps both un-indexed dims: (2, 0, 4)."""
        x_cpu = torch.randn(2, 3, 4, dtype=torch.complex64)
        mask_cpu = torch.zeros(3, dtype=torch.bool)
        expected = x_cpu[:, mask_cpu]

        result = x_cpu.to("neuron")[:, mask_cpu.to("neuron")]

        assert result.shape == expected.shape
        torch.testing.assert_close(result.cpu(), expected)

    def test_all_false_full_rank_mask_is_flat(self):
        """A full-rank mask consumes every dimension, so the result stays 1-D."""
        x_cpu = torch.randn(2, 3, dtype=torch.complex64)
        mask_cpu = torch.zeros(2, 3, dtype=torch.bool)
        expected = x_cpu[mask_cpu]

        result = x_cpu.to("neuron")[mask_cpu.to("neuron")]

        assert result.shape == expected.shape
        torch.testing.assert_close(result.cpu(), expected)


class TestComplexIndexNoFallbackPolicy:
    """Complex index.Tensor honors the no-fallback policy when decomposition is off.

    This one needs a subprocess: kFallbackOnlyForUnimplemented in NeuronDispatcher.cpp
    is a load-time static, so TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS cannot
    be set in the running process.
    """

    def test_index_raises_without_decomposition(self):
        snippet = """
            import torch, torch_neuronx
            x = torch.randn(8, 4, dtype=torch.complex64, device="neuron")
            index = torch.tensor([2, 0], dtype=torch.int64, device="neuron")
            x[index]
        """
        env = os.environ.copy()
        env["TORCH_NEURONX_COMPLEX_DECOMPOSE"] = "0"
        env["TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(snippet)],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        assert result.returncode != 0
        assert "CPU fallback disabled" in result.stderr
        assert "aten::index" in result.stderr
