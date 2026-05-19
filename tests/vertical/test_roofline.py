"""Unit tests for the roofline classifier and per-op handlers.

Covers:
- Classification: matmul family ops, SDPA, reductions, norms, elementwise.
- Matmul layout correctness: linear's transposed weight, addmm's bias skip.
- Broadcasting matmul: 2D x 2D, 3D x 3D, mixed with leading-dim broadcasts.
- SDPA forward and backward shape dispatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from internal.roofline import (
    HBM_BW,
    TE_FLOPS,
    _classify,
    compute_roofline,
)


class TestClassify:
    def test_mm(self):
        assert _classify("aten::mm") == "mm"
        assert _classify("aten::mm.out") == "mm"

    def test_matmul(self):
        assert _classify("aten::matmul") == "matmul"
        assert _classify("aten::matmul.out") == "matmul"

    def test_linear(self):
        assert _classify("aten::linear") == "linear"
        assert _classify("aten::linear.default") == "linear"

    def test_addmm(self):
        assert _classify("aten::addmm") == "addmm"
        assert _classify("aten::addmm.out") == "addmm"

    def test_bmm(self):
        assert _classify("aten::bmm") == "bmm"
        assert _classify("aten::bmm.out") == "bmm"

    def test_baddbmm(self):
        assert _classify("aten::baddbmm") == "baddbmm"

    def test_sdpa(self):
        assert _classify("aten::_scaled_dot_product_fused_attention_overrideable") == "sdpa"
        assert _classify("aten::scaled_dot_product_attention") == "sdpa"

    def test_sdpa_backward(self):
        assert (
            _classify("aten::_scaled_dot_product_fused_attention_overrideable_backward")
            == "sdpa_backward"
        )

    def test_reduction(self):
        assert _classify("aten::sum") == "reduction"
        assert _classify("aten::_foreach_norm.Scalar") == "reduction"

    def test_softmax(self):
        assert _classify("aten::softmax.int") == "softmax"

    def test_elementwise_default(self):
        assert _classify("aten::add.Tensor") == "elementwise"


class TestMatmulFamily:
    def test_mm_flops(self):
        m, k, n = 128, 256, 512
        r = compute_roofline("aten::mm", [torch.randn(m, k), torch.randn(k, n)])
        assert r is not None
        # FLOPs = 2·M·K·N
        expected_compute_s = (2 * m * k * n) / TE_FLOPS
        assert abs(r["t_compute_ms"] / 1000 - expected_compute_s) < 1e-9

    def test_addmm_uses_mat1_mat2_not_self(self):
        # addmm(self=(N,), mat1=(M,K), mat2=(K,N))
        # Bug in old code: read inputs[0]=self as A. For 1D self, returned None.
        m, k, n = 64, 128, 256
        bias = torch.randn(n)
        mat1 = torch.randn(m, k)
        mat2 = torch.randn(k, n)
        r = compute_roofline("aten::addmm", [bias, mat1, mat2])
        assert r is not None
        # FLOPs should be 2·M·K·N, same as the equivalent mm.
        r_mm = compute_roofline("aten::mm", [mat1, mat2])
        assert r["t_compute_ms"] == pytest.approx(r_mm["t_compute_ms"])

    def test_linear_weight_is_transposed(self):
        # linear(input=(B*S, K), weight=(N, K), bias=(N,)) -> (B*S, N)
        # Bug in old code: n = weight.shape[-1] = K. For N != K, this is wrong.
        m, k, n = 2374, 4096, 11008  # llama gate_proj: K != N
        inp = torch.randn(m, k)
        weight = torch.randn(n, k)
        bias = torch.randn(n)
        r = compute_roofline("aten::linear", [inp, weight, bias])
        assert r is not None
        # FLOPs should be 2·M·K·N (gate_proj on a big intermediate dim).
        expected_compute_s = (2 * m * k * n) / TE_FLOPS
        assert r["t_compute_ms"] / 1000 == pytest.approx(expected_compute_s, rel=1e-6)

    def test_linear_3d_input_flattens_batch(self):
        # linear on (B, S, K) with weight (N, K) is a mm with M = B*S.
        b, s, k, n = 2, 64, 4096, 4096
        inp = torch.randn(b, s, k)
        weight = torch.randn(n, k)
        r = compute_roofline("aten::linear", [inp, weight])
        assert r is not None
        expected_flops = 2 * b * s * k * n
        assert r["t_compute_ms"] / 1000 == pytest.approx(expected_flops / TE_FLOPS, rel=1e-6)

    def test_bmm(self):
        b, m, k, n = 8, 64, 128, 256
        r = compute_roofline("aten::bmm", [torch.randn(b, m, k), torch.randn(b, k, n)])
        assert r is not None
        assert r["t_compute_ms"] / 1000 == pytest.approx(b * 2 * m * k * n / TE_FLOPS, rel=1e-6)

    def test_baddbmm_skips_bias(self):
        b, m, k, n = 4, 32, 64, 128
        bias = torch.randn(b, m, n)
        mat1 = torch.randn(b, m, k)
        mat2 = torch.randn(b, k, n)
        r = compute_roofline("aten::baddbmm", [bias, mat1, mat2])
        r_bmm = compute_roofline("aten::bmm", [mat1, mat2])
        assert r["t_compute_ms"] == pytest.approx(r_bmm["t_compute_ms"])


class TestMatmulGeneric:
    def test_2d_by_2d(self):
        m, k, n = 64, 128, 256
        r = compute_roofline("aten::matmul", [torch.randn(m, k), torch.randn(k, n)])
        # For 2D x 2D, matmul should match mm.
        r_mm = compute_roofline("aten::mm", [torch.randn(m, k), torch.randn(k, n)])
        assert r["t_compute_ms"] == pytest.approx(r_mm["t_compute_ms"])

    def test_3d_by_3d(self):
        # Attention-like: Q·K^T with Q=(B,H,S,D), K=(B,H,D,S) via matmul
        b, h, s, d = 1, 32, 1024, 128
        a = torch.randn(b, h, s, d)
        bb = torch.randn(b, h, d, s)
        r = compute_roofline("aten::matmul", [a, bb])
        assert r is not None
        expected_flops = b * h * 2 * s * d * s
        assert r["t_compute_ms"] / 1000 == pytest.approx(expected_flops / TE_FLOPS, rel=1e-6)

    def test_1d_dot_product(self):
        k = 1024
        r = compute_roofline("aten::matmul", [torch.randn(k), torch.randn(k)])
        assert r is not None
        assert r["t_compute_ms"] / 1000 == pytest.approx(2 * k / TE_FLOPS, rel=1e-6)

    def test_broadcasting_leading_dims(self):
        # (B, M, K) x (K, N) — b has 2D, a has 3D. Batch = B.
        b, m, k, n = 4, 32, 64, 128
        r = compute_roofline("aten::matmul", [torch.randn(b, m, k), torch.randn(k, n)])
        assert r is not None
        assert r["t_compute_ms"] / 1000 == pytest.approx(b * 2 * m * k * n / TE_FLOPS, rel=1e-6)


class TestSDPA:
    def test_sdpa_forward(self):
        # LLaMA-7b: (B=1, H=32, S=2048, D=128), fp32 for deterministic math
        b, h, s, d = 1, 32, 2048, 128
        q = torch.randn(b, h, s, d)
        k = torch.randn(b, h, s, d)
        v = torch.randn(b, h, s, d)
        r = compute_roofline(
            "aten::_scaled_dot_product_fused_attention_overrideable",
            [q, k, v],
            dtype=torch.float32,
        )
        assert r is not None
        # FLOPs ≈ 4·B·H·S·S·D (two tensor-engine matmuls)
        expected_flops = 4 * b * h * s * s * d
        assert r["t_compute_ms"] / 1000 == pytest.approx(expected_flops / TE_FLOPS, rel=1e-6)
        # Memory ≈ 4·B·H·S·D·eb (Q + K + V + O, no attention matrix spill)
        eb = 4  # fp32
        expected_bytes = 4 * b * h * s * d * eb
        assert r["t_memory_ms"] / 1000 == pytest.approx(expected_bytes / HBM_BW, rel=1e-6)

    def test_sdpa_asymmetric_seq_lens(self):
        # Decoding: S_q=1, S_kv=2048 (typical generation step)
        b, h, s_q, s_kv, d = 1, 32, 1, 2048, 128
        q = torch.randn(b, h, s_q, d)
        k = torch.randn(b, h, s_kv, d)
        v = torch.randn(b, h, s_kv, d)
        r = compute_roofline("aten::_scaled_dot_product_fused_attention_overrideable", [q, k, v])
        assert r is not None
        expected_flops = 4 * b * h * s_q * s_kv * d
        assert r["t_compute_ms"] / 1000 == pytest.approx(expected_flops / TE_FLOPS, rel=1e-6)

    def test_sdpa_backward_heavier_than_forward(self):
        b, h, s, d = 1, 32, 2048, 128
        q = torch.randn(b, h, s, d)
        k = torch.randn(b, h, s, d)
        v = torch.randn(b, h, s, d)
        grad_out = torch.randn(b, h, s, d)
        r_fwd = compute_roofline(
            "aten::_scaled_dot_product_fused_attention_overrideable", [q, k, v]
        )
        r_bwd = compute_roofline(
            "aten::_scaled_dot_product_fused_attention_overrideable_backward",
            [grad_out, q, k, v],
        )
        # Backward is ~2.5x forward in compute, so roofline should be larger.
        assert r_bwd["t_compute_ms"] > r_fwd["t_compute_ms"]


class TestReductionNormSoftmaxCopy:
    def test_reduction_smaller_memory_than_elementwise(self):
        # aten::sum reads N elements, writes ~1. Memory = N·eb, not 2N·eb.
        t = torch.randn(1024, 1024)
        r = compute_roofline("aten::sum", [t])
        r_add = compute_roofline("aten::add.Tensor", [t, t])
        # Reduction should have strictly less memory traffic than a binary add.
        assert r["t_memory_ms"] < r_add["t_memory_ms"]

    def test_softmax_compute(self):
        n = 4 * 1024
        t = torch.randn(n)
        r = compute_roofline("aten::softmax.int", [t])
        # 5 ops per element on the vector engine.
        # Just sanity-check the result is populated and positive.
        assert r["t_compute_ms"] > 0
        assert r["t_memory_ms"] > 0

    def test_copy_zero_compute(self):
        t = torch.randn(1024)
        r = compute_roofline("aten::clone", [t])
        assert r["t_compute_ms"] == 0.0
        assert r["t_memory_ms"] > 0


class TestDynamicOutputSkip:
    """Roofline is skipped (returns None) for ops whose output size is
    data-dependent; a "pseudo-roofline" from input-size estimates would be
    misleading because actual execution cost scales with output cardinality,
    not input shape.
    """

    def test_masked_select_skipped(self):
        mask = torch.randn(1024) > 0
        self_t = torch.randn(1024)
        r = compute_roofline("aten::masked_select", [self_t, mask])
        assert r is None

    def test_masked_select_out_skipped(self):
        mask = torch.randn(1024) > 0
        self_t = torch.randn(1024)
        r = compute_roofline("aten::masked_select.out", [self_t, mask])
        assert r is None

    def test_nonzero_skipped(self):
        r = compute_roofline("aten::nonzero", [torch.randn(1024)])
        assert r is None

    def test_bincount_skipped(self):
        r = compute_roofline("aten::bincount", [torch.randint(0, 100, (1024,))])
        assert r is None

    def test_unique_skipped(self):
        r = compute_roofline("aten::_unique2", [torch.randn(1024)])
        assert r is None

    def test_repeat_interleave_tensor_skipped(self):
        r = compute_roofline("aten::repeat_interleave.Tensor", [torch.tensor([1, 2, 3])])
        assert r is None

    # ── Intentionally NOT skipped: output shape is determinable ──────────

    def test_count_nonzero_not_skipped(self):
        # aten::count_nonzero always returns a scalar — shape is static.
        r = compute_roofline("aten::count_nonzero", [torch.randn(1024)])
        assert r is not None

    def test_nonzero_static_not_skipped(self):
        # aten::nonzero_static takes an explicit `size` kwarg → static shape.
        r = compute_roofline("aten::nonzero_static", [torch.randn(1024)])
        assert r is not None

    def test_repeat_interleave_self_int_not_skipped(self):
        # Integer repeats give a statically-determined output size.
        r = compute_roofline("aten::repeat_interleave.self_int", [torch.randn(32)])
        assert r is not None

    def test_where_not_skipped(self):
        # where returns a broadcast of inputs; shape is static.
        cond = torch.randn(32) > 0
        r = compute_roofline("aten::where.self", [cond, torch.randn(32), torch.randn(32)])
        assert r is not None
