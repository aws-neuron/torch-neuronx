"""Unit tests for RoPE-style complex computation using view_as_complex/view_as_real.

Validates that Neuron supports the full RoPE pipeline:
- Complex tensor views (view_as_complex, view_as_real)
- Complex number multiplication for rotary embeddings
- End-to-end apply_rotary_emb_complex matching CPU
"""

import unittest

import torch

import torch_neuronx


def _reshape_for_broadcast_complex(
    freqs_cis: torch.Tensor,
    x: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reshape frequency tensor for broadcasting (Llama3/4 style)."""
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb_complex(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply complex-format RoPE to query and key tensors (Llama3/4 style)."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = _reshape_for_broadcast_complex(freqs_cis, xq_, positions)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class TestRoPEComplexComputation(unittest.TestCase):
    """Test RoPE-style complex number computation on Neuron device."""

    def _precompute_freqs_cis(self, dim: int, end: int, theta: float = 10000.0, device="cpu"):
        """Precompute the frequency tensor for complex exponentials (cis)."""
        freqs = 1.0 / (
            theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim)
        )
        t = torch.arange(end, device=device)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

    def test_rope_apply_rotary_emb_complex(self):
        """End-to-end RoPE with complex multiplication matches CPU."""
        batch_size, seq_len, n_heads, head_dim = 2, 128, 8, 64

        torch.manual_seed(42)
        xq_cpu = torch.randn(batch_size, seq_len, n_heads, head_dim, dtype=torch.float32)
        xk_cpu = torch.randn(batch_size, seq_len, n_heads, head_dim, dtype=torch.float32)
        freqs_cis_cpu = self._precompute_freqs_cis(head_dim, seq_len, device="cpu")

        xq_out_cpu, xk_out_cpu = apply_rotary_emb_complex(xq_cpu, xk_cpu, freqs_cis_cpu)

        xq_neuron = xq_cpu.to("neuron")
        xk_neuron = xk_cpu.to("neuron")
        freqs_cis_neuron = self._precompute_freqs_cis(head_dim, seq_len, device="neuron")

        xq_out_neuron, xk_out_neuron = apply_rotary_emb_complex(
            xq_neuron, xk_neuron, freqs_cis_neuron
        )

        torch.testing.assert_close(xq_out_neuron.cpu(), xq_out_cpu, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(xk_out_neuron.cpu(), xk_out_cpu, atol=1e-5, rtol=1e-4)

    def test_rope_complex_multiplication_pattern(self):
        """Complex mul with broadcast freqs_cis matches CPU."""
        batch_size, seq_len, n_heads, head_dim = 2, 64, 4, 32

        torch.manual_seed(42)
        xq_cpu = torch.randn(batch_size, seq_len, n_heads, head_dim, dtype=torch.float32)
        xq_neuron = xq_cpu.to("neuron")

        xq_complex_cpu = torch.view_as_complex(xq_cpu.reshape(*xq_cpu.shape[:-1], -1, 2))
        xq_complex_neuron = torch.view_as_complex(xq_neuron.reshape(*xq_neuron.shape[:-1], -1, 2))

        freqs_cis_cpu = self._precompute_freqs_cis(head_dim, seq_len, device="cpu")
        freqs_cis_neuron = self._precompute_freqs_cis(head_dim, seq_len, device="neuron")

        result_cpu = xq_complex_cpu * freqs_cis_cpu[None, :, None, :]
        result_neuron = xq_complex_neuron * freqs_cis_neuron[None, :, None, :]

        torch.testing.assert_close(result_neuron.cpu(), result_cpu, atol=1e-5, rtol=1e-4)

    def test_rope_frequency_tensor_creation(self):
        """freqs_cis computed on Neuron matches CPU."""
        dim, end = 64, 128

        freqs_cis_cpu = self._precompute_freqs_cis(dim, end, device="cpu")
        freqs_cis_neuron = self._precompute_freqs_cis(dim, end, device="neuron")

        self.assertEqual(freqs_cis_neuron.dtype, torch.complex64)
        self.assertEqual(freqs_cis_neuron.shape, (end, dim // 2))
        torch.testing.assert_close(freqs_cis_neuron.cpu(), freqs_cis_cpu, atol=1e-6, rtol=1e-6)

    def test_rope_frequency_tensor_different_thetas(self):
        """freqs_cis with different theta values matches CPU."""
        dim, end = 64, 128

        for theta in [10000.0, 100000.0, 500000.0]:
            with self.subTest(theta=theta):
                freqs_cis_cpu = self._precompute_freqs_cis(dim, end, theta=theta, device="cpu")
                freqs_cis_neuron = self._precompute_freqs_cis(
                    dim, end, theta=theta, device="neuron"
                )
                torch.testing.assert_close(
                    freqs_cis_neuron.cpu(), freqs_cis_cpu, atol=1e-6, rtol=1e-6
                )

    def test_rope_pattern_end_to_end(self):
        """Full reshape → complex → view_as_real → flatten round-trip matches CPU."""
        batch_size, seq_len, n_heads, head_dim = 2, 128, 8, 64

        xq_cpu = torch.randn(batch_size, seq_len, n_heads, head_dim, dtype=torch.float32)
        xq_neuron = xq_cpu.to("neuron")

        def rope_roundtrip(xq):
            return torch.view_as_real(
                torch.view_as_complex(xq.reshape(*xq.shape[:-1], -1, 2))
            ).flatten(-2)

        torch.testing.assert_close(
            rope_roundtrip(xq_neuron).cpu(), rope_roundtrip(xq_cpu), atol=1e-7, rtol=1e-6
        )

    def test_rope_with_different_head_dimensions(self):
        """RoPE round-trip works with various head dimensions."""
        batch_size, seq_len, n_heads = 2, 64, 4

        for head_dim in [32, 64, 128, 256]:
            with self.subTest(head_dim=head_dim):
                xq = torch.randn(
                    batch_size, seq_len, n_heads, head_dim, dtype=torch.float32, device="neuron"
                )
                xq_complex = torch.view_as_complex(xq.reshape(*xq.shape[:-1], -1, 2))
                xq_recovered = torch.view_as_real(xq_complex).flatten(-2)

                self.assertEqual(xq_complex.shape, (batch_size, seq_len, n_heads, head_dim // 2))
                torch.testing.assert_close(xq.cpu(), xq_recovered.cpu(), atol=1e-6, rtol=1e-5)

    def test_rope_batch_and_seqlen_variations(self):
        """RoPE round-trip works with various batch sizes and sequence lengths."""
        n_heads, head_dim = 8, 64

        for batch_size, seq_len in [(1, 32), (2, 128), (4, 256), (8, 512)]:
            with self.subTest(batch_size=batch_size, seq_len=seq_len):
                xq = torch.randn(
                    batch_size, seq_len, n_heads, head_dim, dtype=torch.float32, device="neuron"
                )
                xq_recovered = torch.view_as_real(
                    torch.view_as_complex(xq.reshape(*xq.shape[:-1], -1, 2))
                ).flatten(-2)
                torch.testing.assert_close(xq.cpu(), xq_recovered.cpu(), atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
