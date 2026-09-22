# ruff: noqa: N806

"""
Unit tests for utility decompositions (frexp).
"""

import pytest
import torch

from torch_neuronx.neuron_dynamo_backend import decompositions

# ---------------------------------------------------------------------------
# frexp
# ---------------------------------------------------------------------------


class TestFrexpDecomposition:
    """Tests for the custom frexp decomposition (aten.frexp.Tensor).

    ``torch.frexp(x)`` returns ``(m, e)`` such that ``x = m * 2**e`` with
    ``|m| in [0.5, 1)`` for finite non-zero ``x``. For ``x in {0, +/-inf, NaN}``
    PyTorch returns ``m = x`` and ``e = 0``. The custom decomposition mirrors
    both branches without touching a backend-native ``frexp`` primitive.
    """

    def test_registered_in_decomposition_table(self):
        assert (
            torch.ops.aten.frexp.Tensor in decompositions.neuron_decompositions
        ), "aten.frexp.Tensor not found in neuron_decompositions"

    def test_matches_cpu_for_finite_nonzero(self):
        """Finite non-zero values reconstruct exactly as ``m * 2**e``."""
        x = torch.tensor(
            [0.5, 1.0, 2.0, 4.0, -3.5, 0.7, 1e-30, 1e30, -1e-5],
            dtype=torch.float32,
        )
        m_ref, e_ref = torch.frexp(x)
        m, e = decompositions.frexp_decomposition(x)
        torch.testing.assert_close(m, m_ref, rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(e, e_ref)
        assert e.dtype == torch.int32
        # Reconstruction identity.
        torch.testing.assert_close(m * torch.exp2(e.float()), x, rtol=1e-6, atol=1e-8)

    def test_matches_cpu_for_specials(self):
        """Zero, +/-inf and NaN follow the ``(m=self, e=0)`` convention."""
        x = torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan")])
        m_ref, e_ref = torch.frexp(x)
        m, e = decompositions.frexp_decomposition(x)
        # Compare with NaN-tolerance since equality on NaN is False.
        torch.testing.assert_close(m, m_ref, equal_nan=True)
        torch.testing.assert_close(e, e_ref)

    @pytest.mark.parametrize("shape", [(4,), (2, 8), (2, 3, 4)])
    def test_multidim_shapes(self, shape):
        torch.manual_seed(0)
        x = torch.randn(shape)
        m_ref, e_ref = torch.frexp(x)
        m, e = decompositions.frexp_decomposition(x)
        torch.testing.assert_close(m, m_ref, rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(e, e_ref)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestFrexpDecompositionE2E:
    """E2E: torch.compile(backend='neuron') traces frexp through the decomp."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("shape", [(8,), (2, 8), (2, 4, 8)])
    def test_accuracy_e2e(self, shape, dtype):
        torch.manual_seed(42)

        def fn(x):
            return torch.frexp(x)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)

        x_cpu = torch.randn(shape, dtype=dtype)
        x_neuron = x_cpu.detach().clone().to("neuron")

        m_ref, e_ref = torch.frexp(x_cpu)
        m_out, e_out = compiled_fn(x_neuron)

        torch.testing.assert_close(m_out.cpu(), m_ref)
        torch.testing.assert_close(e_out.cpu(), e_ref)
