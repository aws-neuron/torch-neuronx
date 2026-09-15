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
