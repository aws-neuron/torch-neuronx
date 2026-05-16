"""Tests for complex-dtype CPU fallback behavior.

The global PrivateUse1 fallback (`neuron_cpu_fallback`) routes ops with no
native Neuron kernel to CPU. Complex dtypes are never supported natively,
so they always fall back. When TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1
the fallback handler raises instead — but only for complex *inputs*, not for
ops that merely produce a complex result from real inputs (e.g. aten::complex.out).

See torch_neuronx/csrc/core/NeuronBindings.cpp::neuron_cpu_fallback and
torch_neuronx/csrc/core/dispatch/DispatchUtils.h::HasComplexNonOutputInput.
"""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

import torch_neuronx  # Registers neuron backend and ops

FLAG_ENV = "TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS"
FLAG_ENABLED = os.environ.get(FLAG_ENV) == "1"


def _run_in_subprocess(snippet: str, flag_value: str) -> subprocess.CompletedProcess:
    """Run a Python snippet with TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS set.

    The flag is cached in C++ at module load, so we need a fresh process to
    exercise the "flag unset" path in an environment where the default build
    sets the flag to "1".
    """
    env = os.environ.copy()
    env[FLAG_ENV] = flag_value
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(snippet)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


class TestComplexFallback:
    """Complex fallback behavior in the currently-running process.

    Runs against whatever value of TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS
    the C++ module was loaded with. DEV builds default to "1".
    """

    @pytest.mark.skipif(
        not FLAG_ENABLED, reason="Requires TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1"
    )
    def test_complex_input_raises_when_flag_set(self):
        """Complex input to an op without native Neuron support raises."""
        z = torch.randn(4, 4, dtype=torch.complex64).to("neuron")
        with pytest.raises(RuntimeError) as excinfo:
            torch.asinh(z)
        msg = str(excinfo.value)
        # The raise message is produced by RaiseComplexFallbackDisabled in
        # DispatchUtils.h — assert its key components.
        assert "CPU fallback disabled" in msg
        assert "TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1" in msg
        assert "complex-dtype input" in msg
        assert "aten::asinh" in msg  # op name interpolated into message

    @pytest.mark.skipif(
        not FLAG_ENABLED, reason="Requires TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1"
    )
    def test_complex_output_from_real_inputs_does_not_raise(self):
        """aten::complex.out produces complex from real inputs — should fall back silently.

        The guard skips out= params via HasComplexNonOutputInput, so this path
        is not blocked even when the flag is set.
        """
        real = torch.randn(4, 4, device="neuron")
        imag = torch.randn(4, 4, device="neuron")
        result = torch.complex(real, imag)
        assert result.dtype == torch.complex64
        assert result.shape == real.shape
        # Result is CPU-backed from the fallback; dtype + numerical correctness is
        # enough — matching against torch.complex(real.cpu(), imag.cpu()) confirms
        # the fallback executed successfully.
        expected = torch.complex(real.cpu(), imag.cpu())
        assert torch.allclose(result.cpu(), expected)

    @pytest.mark.skipif(
        not FLAG_ENABLED, reason="Requires TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1"
    )
    def test_whitelisted_view_on_complex_does_not_raise(self):
        """View ops in VIEW_OPS_WHITELIST are exempt from the guard.

        view_as_complex reshapes a real tensor into a complex view without
        computing on complex values — backend capability is irrelevant.
        """
        real = torch.randn(4, 2, device="neuron")  # last dim = 2 for complex pairing
        result = torch.view_as_complex(real)
        assert result.dtype == torch.complex64

    @pytest.mark.skipif(
        not FLAG_ENABLED, reason="Requires TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1"
    )
    def test_unimplemented_non_complex_op_still_falls_back(self):
        """Ops without native Neuron kernels but no complex input still silent-fall-back.

        The guard is narrowly scoped to complex-input ops; ops like aten::uniform_
        (called by torch.randn) and other infrastructure ops must keep working.
        """
        # torch.randn internally calls aten::uniform_ which has no Neuron kernel.
        x = torch.randn(4, 4, device="neuron")
        assert x.shape == (4, 4)
        assert x.device.type == "neuron"


class TestComplexFallbackFlagUnset:
    """Behavior when TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS is unset.

    The flag is cached at C++ module load so we use subprocesses to isolate.
    """

    def test_complex_input_silent_fallback_when_flag_unset(self):
        """With the flag unset, complex inputs silently fall back to CPU — no raise."""
        snippet = """
            import torch
            import torch_neuronx
            z = torch.randn(4, 4, dtype=torch.complex64).to("neuron")
            result = torch.asinh(z)
            assert result.dtype == torch.complex64, f"got {result.dtype}"
            # Correctness vs pure CPU reference.
            expected = torch.asinh(z.cpu())
            assert torch.allclose(result.cpu(), expected, atol=1e-5), "mismatch vs CPU"
            print("OK")
        """
        result = _run_in_subprocess(snippet, flag_value="0")
        assert (
            result.returncode == 0
        ), f"subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
        assert "OK" in result.stdout

    def test_complex_output_from_real_inputs_works_when_flag_unset(self):
        """Regression: complex-output ops must also work with flag unset."""
        snippet = """
            import torch
            import torch_neuronx
            real = torch.randn(4, 4, device="neuron")
            imag = torch.randn(4, 4, device="neuron")
            result = torch.complex(real, imag)
            assert result.dtype == torch.complex64
            print("OK")
        """
        result = _run_in_subprocess(snippet, flag_value="0")
        assert (
            result.returncode == 0
        ), f"subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
        assert "OK" in result.stdout


class TestComplexCPUFallbackOps:
    """Verify non-allowlisted complex ops fall back to CPU correctly.

    Uses subprocesses with FALLBACK_ONLY=0 since the default (=1) raises
    for complex inputs.
    """

    @pytest.mark.parametrize(
        "op_snippet,expected_dtype",
        [
            ("x + y", "torch.complex64"),
            ("x - y", "torch.complex64"),
            ("torch.sqrt(x)", "torch.complex64"),
            ("torch.abs(x)", "torch.float32"),
        ],
        ids=["add", "sub", "sqrt", "abs"],
    )
    def test_complex_op_falls_back_correctly(self, op_snippet, expected_dtype):
        snippet = f"""
            import torch
            import torch_neuronx
            x_cpu = torch.randn(4, 4, dtype=torch.complex64)
            y_cpu = torch.randn(4, 4, dtype=torch.complex64)
            x = x_cpu.to("neuron")
            y = y_cpu.to("neuron")
            result = {op_snippet}
            assert result.dtype == {expected_dtype}, f"got {{result.dtype}}"
            print("OK")
        """
        result = _run_in_subprocess(snippet, flag_value="0")
        assert (
            result.returncode == 0
        ), f"subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
        assert "OK" in result.stdout

    @pytest.mark.skipif(
        not FLAG_ENABLED, reason="Requires TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1"
    )
    @pytest.mark.parametrize(
        "make_noncontig",
        [
            lambda: torch.randn(8, 8, dtype=torch.complex64, device="neuron").transpose(0, 1),
            lambda: torch.randn(8, 8, dtype=torch.complex64, device="neuron")[::2, :],
            lambda: torch.randn(2, 3, 4, dtype=torch.complex64, device="neuron").permute(2, 0, 1),
        ],
        ids=["transpose", "slice", "permute"],
    )
    def test_complex_contiguous_raises_when_flag_set(self, make_noncontig):
        """Non-contiguous complex tensors raise with FALLBACK_ONLY=1."""
        x_view = make_noncontig()
        with pytest.raises(RuntimeError, match="complex"):
            x_view.contiguous()

    def test_complex_contiguous_noop_when_already_contiguous(self):
        """Already-contiguous returns as-is — no fallback, no raise."""
        x_cpu = torch.randn(4, 4, dtype=torch.complex64)
        x_neuron = x_cpu.to("neuron")
        assert x_neuron.is_contiguous()
        result = x_neuron.contiguous()
        assert result.data_ptr() == x_neuron.data_ptr()
        torch.testing.assert_close(result.cpu(), x_cpu)

    def test_complex_transpose_then_mul_falls_back(self):
        """Transpose + mul: non-contiguous complex input falls back to CPU gracefully."""
        snippet = """
            import torch
            import torch_neuronx
            x = torch.randn(4, 8, dtype=torch.complex64, device="neuron")
            y = torch.randn(8, 4, dtype=torch.complex64, device="neuron")
            x_t = x.transpose(0, 1)
            result = x_t * y
            expected = x.cpu().transpose(0, 1) * y.cpu()
            assert torch.allclose(result.cpu(), expected, atol=1e-5), "mismatch"
            print("OK")
        """
        result = _run_in_subprocess(snippet, flag_value="0")
        assert (
            result.returncode == 0
        ), f"subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
        assert "OK" in result.stdout

    def test_complex_permute_then_div_falls_back(self):
        """Permute + div: non-contiguous complex input falls back to CPU gracefully."""
        snippet = """
            import torch
            import torch_neuronx
            x = torch.randn(2, 3, 4, dtype=torch.complex64, device="neuron")
            y = torch.randn(4, 2, 3, dtype=torch.complex64, device="neuron")
            x_p = x.permute(2, 0, 1)
            result = x_p / y
            expected = x.cpu().permute(2, 0, 1) / y.cpu()
            assert torch.allclose(result.cpu(), expected, atol=1e-5), "mismatch"
            print("OK")
        """
        result = _run_in_subprocess(snippet, flag_value="0")
        assert (
            result.returncode == 0
        ), f"subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
        assert "OK" in result.stdout

    def test_complex_noncontiguous_out_falls_back(self):
        """mul.out with non-contiguous out= tensor falls back to CPU gracefully."""
        snippet = """
            import torch
            import torch_neuronx
            x = torch.randn(4, 4, dtype=torch.complex64, device="neuron")
            y = torch.randn(4, 4, dtype=torch.complex64, device="neuron")
            out = torch.empty(8, 4, dtype=torch.complex64, device="neuron").transpose(0, 1)[:4, :]
            torch.mul(x, y, out=out)
            expected = x.cpu() * y.cpu()
            assert torch.allclose(out.cpu(), expected, atol=1e-5), "mismatch"
            print("OK")
        """
        result = _run_in_subprocess(snippet, flag_value="0")
        assert (
            result.returncode == 0
        ), f"subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
        assert "OK" in result.stdout

    @pytest.mark.xfail(
        reason="cpu_fallback can't make non-contiguous neuron tensor contiguous"
        " (view op can't cross devices), produces silent wrong results",
        strict=False,
    )
    def test_noncontiguous_inplace_complex_mul(self):
        """mul_ on non-contiguous self: cpu_fallback produces incorrect results."""
        snippet = """
            import torch
            import torch_neuronx
            x = torch.randn(4, 8, dtype=torch.complex64, device="neuron")
            y = torch.randn(8, 4, dtype=torch.complex64, device="neuron")
            x_t = x.transpose(0, 1)  # non-contiguous
            x_t.multiply_(y)
            expected = x.cpu().transpose(0, 1) * y.cpu()
            assert torch.allclose(x_t.cpu(), expected, atol=1e-5), "mismatch"
            print("OK")
        """
        result = _run_in_subprocess(snippet, flag_value="0")
        assert (
            result.returncode == 0
        ), f"subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
        assert "OK" in result.stdout


class TestConjNegNonContiguousCopy:
    """Non-contiguous complex copy with conj/neg bits requires CPU fallback.

    These tests exercise copy_ paths where the source or destination is
    non-contiguous AND has conj/neg metadata. The contiguous() call on a
    complex Neuron tensor falls back to CPU, so these need fallback enabled.
    """

    def test_copy_conj_non_contiguous_src_to_neuron(self):
        """Transposed + conj'd source into contiguous Neuron dst."""
        snippet = """
            import torch
            import torch_neuronx
            t = torch.randn(4, 4, dtype=torch.complex64)
            src_neuron = t.to("neuron").transpose(0, 1).conj()
            dst = torch.empty(4, 4, dtype=torch.complex64, device="neuron")
            dst.copy_(src_neuron)
            expected = t.transpose(0, 1).conj().resolve_conj()
            assert torch.allclose(dst.cpu(), expected, atol=1e-5), "mismatch"
            print("OK")
        """
        result = _run_in_subprocess(snippet, flag_value="0")
        assert (
            result.returncode == 0
        ), f"subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
        assert "OK" in result.stdout

    def test_copy_conj_into_non_contiguous_dst(self):
        """Conj'd source into transposed (non-contiguous) Neuron destination."""
        snippet = """
            import torch
            import torch_neuronx
            src = torch.randn(4, 4, dtype=torch.complex64)
            src_neuron = src.to("neuron").conj()
            dst = torch.empty(4, 4, dtype=torch.complex64, device="neuron").t()
            dst.copy_(src_neuron)
            expected = src.conj().resolve_conj()
            assert torch.allclose(dst.cpu(), expected, atol=1e-5), "mismatch"
            print("OK")
        """
        result = _run_in_subprocess(snippet, flag_value="0")
        assert (
            result.returncode == 0
        ), f"subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
        assert "OK" in result.stdout

    @pytest.mark.xfail(reason="Segfaults — non-contiguous complex conj copy not yet supported")
    def test_copy_conj_non_contiguous_src_to_cpu(self):
        """Non-contiguous conj'd Neuron src copied directly to CPU dst."""
        snippet = """
            import torch
            import torch_neuronx
            t = torch.randn(4, 4, dtype=torch.complex64)
            t_neuron = t.to("neuron").transpose(0, 1).conj()
            dst = torch.empty(4, 4, dtype=torch.complex64)
            dst.copy_(t_neuron)
            expected = t.transpose(0, 1).conj().resolve_conj()
            assert torch.allclose(dst, expected, atol=1e-5), "mismatch"
            print("OK")
        """
        result = _run_in_subprocess(snippet, flag_value="0")
        assert (
            result.returncode == 0
        ), f"subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
        assert "OK" in result.stdout

    @pytest.mark.xfail(reason="Segfaults — non-contiguous complex conj clone not yet supported")
    def test_clone_conj_non_contiguous(self):
        """clone() on a non-contiguous conj'd Neuron tensor."""
        snippet = """
            import torch
            import torch_neuronx
            t = torch.randn(4, 4, dtype=torch.complex64)
            t_neuron = t.to("neuron").transpose(0, 1).conj()
            result = t_neuron.clone().cpu()
            expected = t.transpose(0, 1).conj().resolve_conj()
            assert torch.allclose(result, expected, atol=1e-5), "mismatch"
            print("OK")
        """
        result = _run_in_subprocess(snippet, flag_value="0")
        assert (
            result.returncode == 0
        ), f"subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
        assert "OK" in result.stdout
