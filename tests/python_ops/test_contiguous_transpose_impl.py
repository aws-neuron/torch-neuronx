"""Tests ensuring transpose-based contiguous path is used when applicable."""

from unittest.mock import patch

import pytest
import torch

import torch_neuron


def test_contiguous_transpose_uses_jax_kernel(monkeypatch):
    """Transpose case should use the JAX transpose kernel, not the generic copy."""
    # Patch the generic contiguous symbol used by the fallback impl
    from torch_neuron.python_ops import contiguous as contiguous_mod
    from torch_neuron.python_ops.contiguous_transpose import ContiguousTransposeImpl

    jax_called = {"count": 0}
    generic_called = {"count": 0}

    # Wrap _get_kernel to intercept JAX kernel calls
    orig_get_kernel = ContiguousTransposeImpl._get_kernel

    def wrapped_get_kernel(self, permutation):
        k = orig_get_kernel(self, permutation)

        def wrapper(*args, **kwargs):
            jax_called["count"] += 1
            return k(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(ContiguousTransposeImpl, "_get_kernel", wrapped_get_kernel, raising=True)

    # Wrap the generic fallback to ensure it's not hit in the transpose case
    orig_generic = contiguous_mod.contiguous_generic_kernel

    def wrapped_generic(*args, **kwargs):
        generic_called["count"] += 1
        return orig_generic(*args, **kwargs)

    monkeypatch.setattr(contiguous_mod, "contiguous_generic_kernel", wrapped_generic, raising=True)

    # Execute: transpose should be handled by the transpose-based path
    x = torch.randn(128, 256, device="neuron")
    y_view = x.transpose(0, 1)
    y = y_view.contiguous()

    assert y.is_contiguous()
    assert y.shape == (256, 128)
    assert torch.allclose(y, y_view)

    # Validate paths taken
    assert jax_called["count"] >= 1, "Expected JAX transpose kernel to be invoked"
    assert generic_called["count"] == 0, "Generic fallback should not be used for transpose"


def test_contiguous_non_permutation_uses_generic(monkeypatch):
    """Non-permutation view should NOT use JAX transpose kernel and should use generic path."""
    # Patch the generic contiguous symbol used by the fallback impl
    from torch_neuron.python_ops import contiguous as contiguous_mod
    from torch_neuron.python_ops.contiguous_transpose import ContiguousTransposeImpl

    jax_called = {"count": 0}
    generic_called = {"count": 0}

    # Count JAX kernel usage (should not be called)
    orig_get_kernel = ContiguousTransposeImpl._get_kernel

    def wrapped_get_kernel(self, permutation):
        k = orig_get_kernel(self, permutation)

        def wrapper(*args, **kwargs):
            jax_called["count"] += 1
            return k(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(ContiguousTransposeImpl, "_get_kernel", wrapped_get_kernel, raising=True)

    # Count generic fallback usage (should be called)
    orig_generic = contiguous_mod.contiguous_generic_kernel

    def wrapped_generic(*args, **kwargs):
        generic_called["count"] += 1
        return orig_generic(*args, **kwargs)

    monkeypatch.setattr(contiguous_mod, "contiguous_generic_kernel", wrapped_generic, raising=True)

    # Execute: strided slice is not a pure permutation
    x_cpu = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    x = x_cpu.to("neuron")
    y = x[:, ::2]
    assert not y.is_contiguous()

    z = y.contiguous()
    assert z.is_contiguous()
    assert torch.allclose(z, y)

    # Validate paths taken
    assert jax_called["count"] == 0, "JAX transpose kernel should not be used for non-permutation"
    assert generic_called["count"] >= 1, "Generic fallback should handle non-permutation views"


def test_contiguous_transpose_2d_and_autograd():
    """2D transpose path should preserve values and gradients."""
    x = torch.randn(128, 256, device="neuron")
    y_view = x.transpose(0, 1)
    y = y_view.contiguous()

    assert y.is_contiguous()
    assert y.shape == y_view.shape
    assert torch.allclose(y, y_view)

    xg = torch.randn(128, 256, device="neuron", requires_grad=True)
    yg = xg.transpose(0, 1).contiguous()
    yg.retain_grad()
    yg.sum().backward()

    assert xg.grad is not None and torch.allclose(xg.grad, torch.ones_like(xg))
    assert yg.grad is not None and torch.allclose(yg.grad, torch.ones_like(yg))


def test_contiguous_transpose_3d_permute_201():
    """3D permutation (2,0,1) uses transpose path; values preserved."""
    x = torch.randn(4, 5, 6, device="neuron")
    y_view = x.permute(2, 0, 1)
    y = y_view.contiguous()

    assert y.is_contiguous()
    assert y.shape == y_view.shape
    assert torch.allclose(y, y_view)


@pytest.mark.parametrize(
    "rank,shape",
    [
        (4, (2, 3, 4, 5)),
        (5, (2, 3, 4, 5, 2)),
        (6, (2, 3, 4, 5, 2, 3)),
    ],
)
def test_contiguous_transpose_random_perms_rank_4_to_6(rank, shape):
    """Random non-identity perms on ranks 4-6 preserve values; gradients flow."""
    import random

    rng = random.Random(0)

    # Try a few random permutations per rank
    for _ in range(3):
        perm = list(range(rank))
        rng.shuffle(perm)
        if perm == list(range(rank)):
            rng.shuffle(perm)

        x = torch.randn(*shape, device="neuron")
        y_view = x.permute(*perm)
        y = y_view.contiguous()

        assert y.is_contiguous()
        assert y.shape == y_view.shape
        assert torch.allclose(y, y_view)

    # Deterministic autograd check: rotate axes by 1
    perm0 = tuple(range(rank))
    perm0 = perm0[1:] + perm0[:1]
    xg = torch.randn(*shape, device="neuron", requires_grad=True)
    yg = xg.permute(*perm0).contiguous()
    yg.sum().backward()
    assert xg.grad is not None and torch.allclose(xg.grad, torch.ones_like(xg))


def test_incorrect_fallback_to_generic_kernel(monkeypatch):
    """Test incorrect fallback to generic kernel"""

    from torch_neuron.python_ops.compilation import HloCompiler

    compile_called = {"count": 0}

    # Wrap compile_to_neff to track compilation calls
    orig_compile_to_neff = HloCompiler.compile_to_neff

    def wrapped_compile_to_neff(self, hlo_protobuf: bytes) -> bytes:
        compile_called["count"] += 1
        return orig_compile_to_neff(self, hlo_protobuf)

    monkeypatch.setattr(HloCompiler, "compile_to_neff", wrapped_compile_to_neff, raising=True)

    def mock_contiguous_generic_kernel(
        src, dst, shape, src_strides, dst_strides, src_storage_offset=0
    ):
        """Mock that returns corrupted output to detect generic kernel was used"""
        dst.cpu().zero_()  # Fill with zeros instead of correct values
        return dst

    with patch(
        "torch_neuron.python_ops.contiguous.contiguous_generic_kernel",
        side_effect=mock_contiguous_generic_kernel,
    ):
        base_tensor = torch.arange(
            1 * 2048 * 32 * 128, dtype=torch.bfloat16, device="neuron"
        ).reshape(1, 2048, 32, 128)

        # Step 1: Cache the first permutation
        view_1 = base_tensor.permute(0, 2, 3, 1)
        compile_called["count"] = 0
        view_1.contiguous()
        assert (
            compile_called["count"] == 1
        ), f"Expected 1 compilation, got {compile_called['count']}"

        # Step 2: This should compile a new neff and successfully execute
        view_2 = base_tensor.permute(0, 2, 1, 3)
        compile_called["count"] = 0
        result_2 = view_2.contiguous()
        assert (
            compile_called["count"] == 1
        ), f"Expected 1 compilation, got {compile_called['count']}"

        expected_2 = base_tensor.cpu().permute(0, 2, 1, 3).contiguous()
        assert torch.allclose(result_2.cpu(), expected_2)
