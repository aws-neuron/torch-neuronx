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
    """Run the snippet with the default complex decomposition setting."""
    env = os.environ.copy()
    env.pop("TORCH_NEURONX_COMPLEX_DECOMPOSE", None)
    env.pop("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS", None)
    # A native crash in the child would otherwise be a bare rc; print a C traceback.
    env["PYTHONFAULTHANDLER"] = "1"
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(snippet)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _run_eager_subprocess(snippet: str) -> subprocess.CompletedProcess:
    """Run snippet in a subprocess with the default env (no decomposition flag).

    The eager _fft_r2c path is a registered kernel, so it must work without
    TORCH_NEURONX_COMPLEX_DECOMPOSE. Run with a clean env to prove that.
    """
    env = os.environ.copy()
    # A native crash in the child would otherwise be a bare rc; print a C traceback.
    env["PYTHONFAULTHANDLER"] = "1"
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


# Each test below starts a Neuron subprocess, which initializes the runtime and
# takes a core. To keep that cost low under the shared parallel run, each class
# packs its variants into ONE subprocess and prints the variant name as it
# passes; the last line is "PASS". A failing variant prints its name in the
# assertion, so the single test still points at the broken case.


# Eager dispatch path tests


@pytest.mark.needs_process_isolation(reason="Starts a Neuron subprocess that needs its own core.")
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


@pytest.mark.needs_process_isolation(reason="Starts a Neuron subprocess that needs its own core.")
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


class TestComplexDecompEagerToCopy:
    """Test real-to-complex conversion on the eager path."""

    def test_f32_to_complex64(self):
        snippet = """
            import torch, torch_neuronx
            x = torch.tensor([-3.5, 0.0, 2.25], dtype=torch.float32)
            result_neuron = torch.ops.aten._to_copy.default(
                x.to("neuron"), dtype=torch.complex64
            )
            assert result_neuron.device.type == "neuron"
            result = result_neuron.cpu()
            expected = torch.complex(x, torch.zeros_like(x))
            assert result.dtype == torch.complex64
            torch.testing.assert_close(result, expected)
            print("PASS")
        """
        result = _run_decomp_subprocess(snippet)
        _assert_subprocess_ok(result, "_to_copy:f32_to_complex64")


# torch.compile path tests


@pytest.mark.needs_process_isolation(reason="Starts a Neuron subprocess that needs its own core.")
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


@pytest.mark.needs_process_isolation(reason="Starts a Neuron subprocess that needs its own core.")
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


# FFT family tests (real DFT matmul decomposition)
#
# The FFT primitives decompose to real cosine/sine matmuls (see
# decompositions.py). The complex output boundary still needs the framework
# complex pass, so the compile variants run with TORCH_NEURONX_COMPLEX_DECOMPOSE=1.
# Use torch.testing.assert_close with its default per-dtype tolerance. The
# transform length stays small so the fp32 matmul meets the float32 default; a
# larger length accumulates more rounding.


@pytest.mark.needs_process_isolation(reason="Starts a Neuron subprocess that needs its own core.")
class TestFftR2cCompile:
    """real-to-complex FFT via the DFT matmul decomposition (torch.compile).

    Every variant runs in one subprocess. Cases: onesided, full spectrum, a
    non-last axis (movedim round-trip), a negative dim, both normalization
    modes, and a bfloat16 input.
    """

    def test_compile_fft_r2c_variants(self):
        snippet = """
            import torch, torch_neuronx
            torch.manual_seed(0)

            def compile_r2c(x, dim, norm, onesided):
                @torch.compile(backend="neuron", fullgraph=True, dynamic=False)
                def fn(t):
                    return torch.ops.aten._fft_r2c.default(t, dim, norm, onesided)
                r = fn(x); torch_neuronx.synchronize()
                return r

            def check(name, x_cpu, dim, norm, onesided, exp_shape, ref_cpu=None,
                      atol=None, rtol=None):
                r = compile_r2c(x_cpu.to("neuron"), dim, norm, onesided)
                assert r.device.type == "neuron", (name, r.device)
                assert r.dtype == torch.complex64, (name, r.dtype)
                assert tuple(r.shape) == exp_shape, (name, r.shape)
                ref = ref_cpu if ref_cpu is not None else x_cpu
                exp = torch.ops.aten._fft_r2c.default(ref, dim, norm, onesided)
                kw = {} if atol is None else {"atol": atol, "rtol": rtol}
                torch.testing.assert_close(r.cpu(), exp, **kw)
                print(f"  {name} OK", flush=True)

            check("onesided",      torch.randn(8, 32), [1], 0, True,  (8, 17))
            check("full_spectrum", torch.randn(8, 32), [1], 0, False, (8, 32))
            check("nonlast_axis",  torch.randn(32, 8), [0], 0, True,  (17, 8))
            # negative dim: validate through the public torch.fft.rfft, which
            # wraps the dim before the private _fft_r2c op. A raw negative dim
            # into the private op is unsupported and segfaults (including the
            # CPU reference), so both sides use the public API.
            xn = torch.randn(32, 8)
            rn = torch.fft.rfft(xn.to("neuron"), dim=-2); torch_neuronx.synchronize()
            assert tuple(rn.shape) == (17, 8), ("negative_dim", rn.shape)
            torch.testing.assert_close(rn.cpu(), torch.fft.rfft(xn, dim=-2), atol=1e-4, rtol=1e-5)
            print("  negative_dim OK", flush=True)
            check("norm_sqrt",     torch.randn(8, 32), [1], 1, True,  (8, 17))
            check("norm_full",     torch.randn(8, 32), [1], 2, True,  (8, 17))
            # bfloat16 runs the matmul at bfloat16 precision, so relax the tolerance.
            bf = torch.randn(8, 32).to(torch.bfloat16)
            check("bfloat16", bf, [1], 0, True, (8, 17), ref_cpu=bf.float(),
                  atol=2e-2, rtol=1.6e-2)
            print("PASS")
        """
        result = _run_decomp_subprocess(snippet)
        _assert_subprocess_ok(result, "compile:_fft_r2c.variants")


@pytest.mark.needs_process_isolation(reason="Starts a Neuron subprocess that needs its own core.")
class TestFftR2cEager:
    """real-to-complex FFT through eager dispatch.

    _fft_r2c is a registered eager kernel (see ops/fft.py). The kernel retraces
    the shared _fft_r2c decomposition with make_fx, so eager and compile use one
    implementation. These run without TORCH_NEURONX_COMPLEX_DECOMPOSE. Every
    variant runs in one subprocess, including the high-level torch.fft.rfft path.
    """

    def test_eager_fft_r2c_variants(self):
        snippet = """
            import torch, torch_neuronx
            torch.manual_seed(0)

            def eager_r2c(x, dim, norm, onesided):
                r = torch.ops.aten._fft_r2c.default(x, dim, norm, onesided)
                torch_neuronx.synchronize()
                return r

            def check(name, x_cpu, dim, norm, onesided, exp_shape, ref_cpu=None,
                      atol=None, rtol=None):
                r = eager_r2c(x_cpu.to("neuron"), dim, norm, onesided)
                assert r.dtype == torch.complex64, (name, r.dtype)
                assert tuple(r.shape) == exp_shape, (name, r.shape)
                ref = ref_cpu if ref_cpu is not None else x_cpu
                exp = torch.ops.aten._fft_r2c.default(ref, dim, norm, onesided)
                kw = {} if atol is None else {"atol": atol, "rtol": rtol}
                torch.testing.assert_close(r.cpu(), exp, **kw)
                print(f"  {name} OK", flush=True)

            # The high-level torch.fft.rfft path lowers to _fft_r2c eagerly.
            xr = torch.randn(8, 32)
            r = torch.fft.rfft(xr.to("neuron")); torch_neuronx.synchronize()
            assert r.device.type == "neuron", ("rfft", r.device)
            assert r.dtype == torch.complex64, ("rfft", r.dtype)
            assert tuple(r.shape) == (8, 17), ("rfft", r.shape)
            torch.testing.assert_close(r.cpu(), torch.fft.rfft(xr), atol=1e-4, rtol=1e-5)
            print("  rfft OK", flush=True)

            check("onesided",      torch.randn(8, 32), [1], 0, True,  (8, 17))
            check("full_spectrum", torch.randn(8, 32), [1], 0, False, (8, 32))
            check("nonlast_axis",  torch.randn(32, 8), [0], 0, True,  (17, 8))
            # negative dim: validate through the public torch.fft.rfft, which
            # wraps the dim before the private _fft_r2c op. A raw negative dim
            # into the private op is unsupported and segfaults (including the
            # CPU reference), so both sides use the public API.
            xn = torch.randn(32, 8)
            rn = torch.fft.rfft(xn.to("neuron"), dim=-2); torch_neuronx.synchronize()
            assert tuple(rn.shape) == (17, 8), ("negative_dim", rn.shape)
            torch.testing.assert_close(rn.cpu(), torch.fft.rfft(xn, dim=-2), atol=1e-4, rtol=1e-5)
            print("  negative_dim OK", flush=True)
            check("norm_sqrt",     torch.randn(8, 32), [1], 1, True,  (8, 17))
            check("norm_full",     torch.randn(8, 32), [1], 2, True,  (8, 17))
            bf = torch.randn(8, 32).to(torch.bfloat16)
            check("bfloat16", bf, [1], 0, True, (8, 17), ref_cpu=bf.float(),
                  atol=2e-2, rtol=1.6e-2)
            print("PASS")
        """
        result = _run_eager_subprocess(snippet)
        _assert_subprocess_ok(result, "_fft_r2c.variants")


# Complex abs tests
#
# abs of a complex tensor is not an explicit coverage op. It decomposes to
# hypot(real, imag) (see decompositions.py) so that torch.stft(...).abs()
# lowers end to end. Real abs is unaffected (the decomposition defers it).


@pytest.mark.needs_process_isolation(reason="Starts a Neuron subprocess that needs its own core.")
class TestComplexAbs:
    """complex abs via the hypot(real, imag) decomposition (eager and compile)."""

    def test_complex_abs(self):
        # Neuron sqrt has a ~7e-6 relative floor. Relax like TestStftEndToEnd.
        snippet = """
            import torch, torch_neuronx
            x = torch.randn(8, 257, dtype=torch.complex64)
            expected = x.abs()

            # eager
            r_eager = x.to("neuron").abs().cpu()
            assert r_eager.dtype == torch.float32, r_eager.dtype
            torch.testing.assert_close(r_eager, expected, atol=1e-4, rtol=1e-5)
            print("  eager OK", flush=True)

            # torch.compile
            @torch.compile(backend="neuron", fullgraph=True, dynamic=False)
            def fn(t):
                return t.abs()
            r_comp = fn(x.to("neuron")); torch_neuronx.synchronize()
            assert r_comp.device.type == "neuron", r_comp.device
            assert r_comp.dtype == torch.float32, r_comp.dtype
            torch.testing.assert_close(r_comp.cpu(), expected, atol=1e-4, rtol=1e-5)
            print("  compile OK", flush=True)
            print("PASS")
        """
        result = _run_decomp_subprocess(snippet)
        _assert_subprocess_ok(result, "complex_abs")


@pytest.mark.needs_process_isolation(reason="Starts a Neuron subprocess that needs its own core.")
class TestStftEndToEnd:
    """torch.stft(...).abs() end to end. This motivates the FFT and complex
    abs decompositions.

    The primitive tests above run at the float32 default tolerance. This
    composite runs the window multiply, the DFT matmul, and the magnitude in
    fp32, then compares against the CPU FFT (a different algorithm). That
    pipeline accumulates a slightly larger error (measured max abs 4.8e-5,
    rel 6.6e-6), so relax to atol 1e-4 / rtol 1e-5. Both stay negligible on a
    magnitude spectrum.
    """

    def test_compile_stft_abs(self):
        snippet = """
            import torch, torch_neuronx
            n_fft, hop = 64, 16
            def stft_abs(x, window):
                return torch.stft(x, n_fft, hop, window=window, center=True,
                                  return_complex=True).abs()

            compiled = torch.compile(stft_abs, backend="neuron", fullgraph=True,
                                     dynamic=False)
            x = torch.randn(1, 512)
            window = torch.hann_window(n_fft)
            result = compiled(x.to("neuron"), window.to("neuron")); torch_neuronx.synchronize()
            assert result.device.type == "neuron", result.device
            expected = stft_abs(x, window)
            torch.testing.assert_close(result.cpu(), expected, atol=1e-4, rtol=1e-5)
            print("PASS")
        """
        result = _run_decomp_subprocess(snippet)
        _assert_subprocess_ok(result, "compile:stft.abs")


class TestComplexDecompCompileAbs:
    """Complex absolute value through the torch.compile path."""

    def test_compile_abs(self):
        snippet = """
            import torch, torch_neuronx
            @torch.compile(backend="neuron")
            def fn(x):
                return torch.abs(x)

            x = torch.tensor([3 + 4j, 5 + 12j, 0j], dtype=torch.complex64, device="neuron")
            result = fn(x).cpu()
            expected = torch.abs(x.cpu())
            torch.testing.assert_close(result, expected)
            print("PASS")
        """
        result = _run_decomp_subprocess(snippet)
        _assert_subprocess_ok(result, "compile:abs")


@pytest.mark.needs_process_isolation(reason="Starts a Neuron subprocess that needs its own core.")
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

    def test_maybe_decompose_returns_graph_unchanged_when_disabled(self, monkeypatch):
        """Return the graph when the environment variable disables decomposition."""
        from unittest.mock import MagicMock

        from torch_neuronx.neuron_dynamo_backend.fx.passes.complex_decomposition import (
            maybe_decompose_complex,
        )

        gm = MagicMock()
        monkeypatch.setenv("TORCH_NEURONX_COMPLEX_DECOMPOSE", "0")

        result = maybe_decompose_complex(gm, [])

        assert result is gm


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
        """A scalar complex gather must preserve both components."""
        index_cpu = torch.tensor(0, dtype=torch.int64)
        index = index_cpu.to("neuron")

        for value in (1 + 2j, 3 + 4j):
            x_cpu = torch.tensor(value, dtype=torch.complex64)
            expected = torch.gather(x_cpu, dim, index_cpu)
            torch.testing.assert_close(expected, x_cpu)
            result = torch.gather(x_cpu.to("neuron"), dim, index)
            torch.testing.assert_close(result.cpu(), expected)

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


# ComplexTensor new_full rule tests (no hardware needed).
# torch_neuronx/__init__.py registers the rule at import (via
# apply_complex_decomposition_patches), so importing torch_neuronx -- done at the
# top of this module -- registers it before any test runs.


class TestComplexNewFullRegisteredOnImport:
    """Importing torch_neuronx registers the new_full rule (the __init__ wiring)."""

    def test_new_full_registered_on_import(self):
        from torch._subclasses.complex_tensor._ops.common import COMPLEX_OPS_TABLE

        assert torch.ops.aten.new_full in COMPLEX_OPS_TABLE


class TestComplexTensorNewFull:
    """The registered new_full rule matches torch.full across dtype and fill."""

    @pytest.mark.parametrize("fill", [2.0 + 3.0j, 5.0])
    @pytest.mark.parametrize("dtype", [None, torch.complex64])
    def test_new_full_complex_result(self, fill, dtype):
        from torch._subclasses.complex_tensor._ops.common import ComplexTensorMode

        x = torch.randn(2, dtype=torch.complex64)
        with ComplexTensorMode():
            out = torch.ops.aten.new_full.default(x, [3], fill, dtype=dtype)
        out = torch.complex(out.real.cpu(), out.imag.cpu())
        expected = torch.full([3], fill, dtype=torch.complex64)
        assert out.dtype == torch.complex64
        assert torch.allclose(out, expected)

    def test_new_full_real_dtype_real_fill(self):
        from torch._subclasses.complex_tensor._ops.common import ComplexTensorMode

        x = torch.randn(2, dtype=torch.complex64)
        with ComplexTensorMode():
            out = torch.ops.aten.new_full.default(x, [3], 5.0, dtype=torch.float32)
        out = out.cpu()
        assert out.dtype == torch.float32
        assert torch.allclose(out, torch.full([3], 5.0))
