"""Tests for dynamic-shape NKI kernel dispatch.

When TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS=1 and the input shape meets the
kernel's constraints, the op routes to the dynamic NKI kernel instead of the
default MLIR/PyTorch lowering.
Each op test asserts TWO things:

  1. the dynamic kernel actually executed
  2. the result matches the CPU reference.

Plus shape-invariance tests: compile-for-one-shape / execute-on-others must NOT
recompile (single op, and the full MoE-forward loop).

Covers the MoE-forward ops: silu, add, mul, mm (dynamic-M), index (row gather),
scatter_add.

The whole module is skipped unless the feature flag is EXPORTED before the
process starts.
"""

import contextlib
import os
from typing import ClassVar

import pytest
import torch

import torch_neuronx

_DTYPES = [torch.bfloat16, torch.float32, torch.float16]
_TOL = {torch.float32: (1e-4, 1e-4), torch.bfloat16: (1e-2, 1e-2), torch.float16: (5e-3, 5e-3)}

_FLAG_ON = os.environ.get("TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS") == "1"


def _entry(op_name):
    from torch_neuronx.python_ops.torch_mlir.ops.dynamic_kernels.registry import (
        get_dynamic_nki_entry,
    )

    e = get_dynamic_nki_entry(op_name)
    assert e is not None, f"{op_name} not registered (is the flag exported?)"
    return e


@contextlib.contextmanager
def _assert_dynamic_ran(op_name, min_execs=1):
    """Assert the dynamic NKI kernel for `op_name` executed inside the block.

    Guards against the silent-fallback trap: a numerically-correct result can
    come from the DEFAULT lowering without ever touching the dynamic kernel.
    exec_count only advances in the dynamic dispatch entry point, so a jump of
    >= min_execs proves the dynamic path ran.
    """
    e = _entry(op_name)
    before = e._exec_count
    yield e
    got = e._exec_count - before
    assert got >= min_execs, (
        f"{op_name}: dynamic kernel did NOT run (exec_count +{got}, expected "
        f">= {min_execs}) — the op fell back to the default lowering"
    )


@contextlib.contextmanager
def _assert_dynamic_skipped(op_name):
    """Assert the dynamic kernel did NOT run (op declined -> default lowering)."""
    e = _entry(op_name)
    before = e._exec_count
    yield e
    assert e._exec_count == before, f"{op_name}: dynamic kernel ran but should have declined"


@pytest.mark.skipif(
    not _FLAG_ON,
    reason="dynamic NKI disabled — export TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS=1 to run",
)
@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestDynamicNkiKernels:
    SHAPE = (1, 512, 12288)

    def _bf16_neuron(self, *shapes):
        return [torch.randn(s, dtype=torch.bfloat16).to("neuron") for s in shapes]

    def _neuron(self, dtype, *shapes):
        return [(torch.randn(s, dtype=dtype) / 8).to("neuron") for s in shapes]

    def test_dynamic_nki_silu(self):
        (x,) = self._bf16_neuron(self.SHAPE)
        with _assert_dynamic_ran("silu"):
            out = torch.nn.functional.silu(x).cpu()
        torch.testing.assert_close(out, torch.nn.functional.silu(x.cpu()), rtol=1e-2, atol=1e-2)

    def test_dynamic_nki_add(self):
        x, y = self._bf16_neuron(self.SHAPE, self.SHAPE)
        with _assert_dynamic_ran("add"):
            out = torch.add(x, y).cpu()
        torch.testing.assert_close(out, x.cpu() + y.cpu(), rtol=1e-2, atol=1e-2)

    def test_dynamic_nki_add_alpha_declines(self):
        # alpha != 1 (kwarg-only) can't be honored by the a+b kernel -> can_handle
        # rejects -> falls back to the default lowering -> still numerically correct.
        x, y = self._bf16_neuron(self.SHAPE, self.SHAPE)
        with _assert_dynamic_skipped("add"):
            out = torch.add(x, y, alpha=2).cpu()
        torch.testing.assert_close(out, x.cpu() + 2 * y.cpu(), rtol=1e-2, atol=1e-2)

    def test_dynamic_nki_add_alpha_one_routes(self):
        # alpha == 1 is the default -> dynamic path still handles it.
        x, y = self._bf16_neuron(self.SHAPE, self.SHAPE)
        with _assert_dynamic_ran("add"):
            out = torch.add(x, y, alpha=1).cpu()
        torch.testing.assert_close(out, x.cpu() + y.cpu(), rtol=1e-2, atol=1e-2)

    def test_dynamic_nki_add_out_writes_in_place(self):
        # out= is the destination buffer (not an op kwarg); dynamic path runs and
        # the caller's buffer receives the result.
        x, y = self._bf16_neuron(self.SHAPE, self.SHAPE)
        buf = torch.empty_like(x)
        with _assert_dynamic_ran("add"):
            torch.add(x, y, out=buf)
        torch.testing.assert_close(buf.cpu(), x.cpu() + y.cpu(), rtol=1e-2, atol=1e-2)

    def test_dynamic_nki_mul(self):
        x, y = self._bf16_neuron(self.SHAPE, self.SHAPE)
        with _assert_dynamic_ran("mul"):
            out = torch.mul(x, y).cpu()
        torch.testing.assert_close(out, x.cpu() * y.cpu(), rtol=1e-2, atol=1e-2)

    # ── dtype coverage for the elementwise / gather / scatter kernels ──
    # 2-D [M, H]: M % 128 == 0 (H arbitrary).
    ELEM_SHAPE = (256, 2048)
    T, N = 2048, 512  # gather/scatter source rows, gathered/scattered rows

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_dynamic_nki_silu_dtypes(self, dtype):
        rtol, atol = _TOL[dtype]
        (x,) = self._neuron(dtype, self.ELEM_SHAPE)
        with _assert_dynamic_ran("silu"):
            out = torch.nn.functional.silu(x).cpu()
        torch.testing.assert_close(out, torch.nn.functional.silu(x.cpu()), rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_dynamic_nki_mul_dtypes(self, dtype):
        rtol, atol = _TOL[dtype]
        x, y = self._neuron(dtype, self.ELEM_SHAPE, self.ELEM_SHAPE)
        with _assert_dynamic_ran("mul"):
            out = torch.mul(x, y).cpu()
        torch.testing.assert_close(out, x.cpu() * y.cpu(), rtol=rtol, atol=atol)

    def test_dynamic_nki_mm(self):
        # mm is bf16-only. K=2048, N=4096 (K,N % 1024 == 0), M % 128 == 0.
        a, w = self._neuron(torch.bfloat16, (256, 2048), (2048, 4096))
        with _assert_dynamic_ran("mm"):
            out = (a @ w).cpu()
        torch.testing.assert_close(out, a.cpu() @ w.cpu(), rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_dynamic_nki_index(self, dtype):
        # x[idx] row-gather: source [T, H], 1-D index of n rows (n % 128 == 0, n <= T).
        rtol, atol = _TOL[dtype]
        (src,) = self._neuron(dtype, (self.T, 2048))
        idx = torch.randperm(self.T)[: self.N]
        with _assert_dynamic_ran("index"):
            out = src[idx.to("neuron")].cpu()
        torch.testing.assert_close(out, src.cpu()[idx], rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_dynamic_nki_scatter_add(self, dtype):
        # in-place scatter_add_ into [T, H] at n unique rows (MoE token->row unique).
        rtol, atol = _TOL[dtype]
        n, H = self.N, 2048  # noqa: N806
        (src,) = self._neuron(dtype, (n, H))
        idx = torch.randperm(self.T)[:n]

        ref = torch.zeros(self.T, H, dtype=dtype)
        ref.scatter_add_(0, idx.unsqueeze(1).expand(n, H).to(torch.int64), src.cpu())

        out = torch.zeros(self.T, H, dtype=dtype).to("neuron")
        # in-place variant routes to the "scatter_add_" registry entry.
        with _assert_dynamic_ran("scatter_add_"):
            out.scatter_add_(0, idx.to("neuron").unsqueeze(1).expand(n, H), src)
        torch.testing.assert_close(out.cpu(), ref, rtol=rtol, atol=atol)

    def test_dynamic_nki_compile_once(self):
        """Compile for one shape, execute on OTHERS, without recompiling.

        Sweep M over distinct 128-multiples (per-expert token counts). The
        dynamic kernel must run at every M (exec_count advances, >=2 distinct Ms
        seen) while reusing ONE NEFF per compile_key (compiled count grows <=1).
        """
        entry = _entry("silu")
        compiled_before = len(entry._compiled)
        exec_before = entry._exec_count
        entry._exec_ms.clear()

        ms = (128, 256, 512)
        for m in ms:
            x = (torch.randn(m, 2048, dtype=torch.bfloat16) / 8).to("neuron")
            torch.nn.functional.silu(x).cpu()

        # dynamic kernel ran for each M ...
        assert entry._exec_count - exec_before == len(ms), "dynamic silu did not run for every M"
        assert entry._exec_ms.issuperset(
            set(ms)
        ), f"expected to execute at Ms {set(ms)}, saw {entry._exec_ms}"
        # ... but bf16 H=2048 is ONE compile_key -> at most one NEFF added.
        assert len(entry._compiled) - compiled_before <= 1, "recompiled across M values"

    # H values that exercise the free-dim remainder tile: 640=512+128,
    # 576=512+64, 384<512 (single short tile), 1000=512+488 (non-128 remainder).
    # H is a compile-time constant, so the kernel tiles the free dim at trace
    # time (512-wide tiles + a final short tile) — arbitrary H, no padding.
    @pytest.mark.parametrize("H", [640, 576, 384, 1000])
    def test_dynamic_nki_arbitrary_h_silu(self, H):  # noqa: N803
        with _assert_dynamic_ran("silu"):
            x = (torch.randn(256, H, dtype=torch.bfloat16) / 8).to("neuron")
            out = torch.nn.functional.silu(x).cpu()
        torch.testing.assert_close(out, torch.nn.functional.silu(x.cpu()), rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("H", [640, 576, 384, 1000])
    def test_dynamic_nki_arbitrary_h_gather(self, H):  # noqa: N803
        T, n = 512, 256  # noqa: N806
        src = (torch.randn(T, H, dtype=torch.bfloat16) / 8).to("neuron")
        idx = torch.randperm(T)[:n]
        with _assert_dynamic_ran("index"):
            out = src[idx.to("neuron")].cpu()
        torch.testing.assert_close(out, src.cpu()[idx], rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("H", [640, 576, 384, 1000])
    def test_dynamic_nki_arbitrary_h_scatter_add(self, H):  # noqa: N803
        T, n = 512, 256  # noqa: N806
        src = (torch.randn(n, H, dtype=torch.bfloat16) / 8).to("neuron")
        idx = torch.randperm(T)[:n]
        ref = torch.zeros(T, H, dtype=torch.bfloat16)
        ref.scatter_add_(0, idx.unsqueeze(1).expand(n, H).to(torch.int64), src.cpu())
        out = torch.zeros(T, H, dtype=torch.bfloat16).to("neuron")
        with _assert_dynamic_ran("scatter_add_"):
            out.scatter_add_(0, idx.to("neuron").unsqueeze(1).expand(n, H), src)
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)

    # ── end-to-end MoE forward: no recompiles as routing (token counts) change ──
    # per-expert SwiGLU FFN over a routed token slice
    # — index -> mm -> silu -> mul -> mm -> scatter_add. Real routing
    # changes every step, so per-expert token counts (the M dim) change; the point
    # of dynamic shapes is that each op reuses ONE NEFF per compile_key across all
    # those M values instead of recompiling.
    MOE_T = 1024  # tokens (multiple of 128)
    MOE_E = 4  # experts
    MOE_H = 2048  # hidden (multiple of mm N-block 1024)
    MOE_H4 = 4096  # FFN inner (multiple of mm N-block 1024)
    # ops the loop exercises. scatter_add_ (in-place) is the variant the loop calls.
    MOE_OPS: ClassVar = ["index", "mm", "silu", "mul", "scatter_add_"]

    def _moe_routing_128(self, route_seed):
        """Partition MOE_T tokens across MOE_E experts; each count a 128-multiple.

        Different route_seed -> different per-expert counts -> different M shapes.
        Returns a list of 1-D int32 index tensors (a partition of [0, MOE_T)).
        """
        T, E, tile = self.MOE_T, self.MOE_E, 128  # noqa: N806
        n_tiles = T // tile
        g = torch.Generator().manual_seed(route_seed)
        tiles_per_expert = [1] * E
        for _ in range(n_tiles - E):
            tiles_per_expert[int(torch.randint(0, E, (1,), generator=g).item())] += 1
        perm = torch.randperm(T, generator=g)
        idx, start = [], 0
        for e in range(E):
            n_e = tiles_per_expert[e] * tile
            idx.append(perm[start : start + n_e].to(torch.int32))
            start += n_e
        return idx

    @staticmethod
    def _moe_expert_loop(x, w1, w2, w3, expert_indices, output, num_experts):
        """One MoE forward pass (SwiGLU per expert), accumulating into output."""
        for e in range(num_experts):
            idx = expert_indices[e]
            if idx.numel() == 0:
                continue
            x_expert = x[idx]  # index (gather)
            up = x_expert @ w1[e]  # mm
            gate = x_expert @ w2[e]  # mm
            gated = torch.nn.functional.silu(up) * gate  # silu, mul
            out = gated @ w3[e]  # mm
            # scatter_add_ index must be int64 on CPU
            idx_expanded = idx.to(torch.int64).unsqueeze(1).expand_as(out)
            output.scatter_add_(0, idx_expanded, out)  # scatter_add_

    def test_dynamic_nki_moe_forward_no_recompile(self):
        """E2E MoE forward across changing routing: routes dynamically, is
        correct, and does NOT recompile.

        Step 0 warms the NEFF cache. Steps 1..n use DIFFERENT routing (new
        per-expert token counts = new M shapes). Asserts, across the warm steps:
          - every MoE op executed on the dynamic path (exec_count advanced);
          - the ops saw MULTIPLE distinct M values (proving shape variation);
          - NO new NEFF was compiled (compiled count flat) — recompile-free.
        """
        T, E, H, H4 = self.MOE_T, self.MOE_E, self.MOE_H, self.MOE_H4  # noqa: N806
        torch.manual_seed(0)
        w1 = [(torch.randn(H, H4, dtype=torch.bfloat16) / 64) for _ in range(E)]
        w2 = [(torch.randn(H, H4, dtype=torch.bfloat16) / 64) for _ in range(E)]
        w3 = [(torch.randn(H4, H, dtype=torch.bfloat16) / 64) for _ in range(E)]
        x = torch.randn(T, H, dtype=torch.bfloat16) / 8

        w1_n = [w.to("neuron") for w in w1]
        w2_n = [w.to("neuron") for w in w2]
        w3_n = [w.to("neuron") for w in w3]
        x_n = x.to("neuron")

        entries = {name: _entry(name) for name in self.MOE_OPS}

        def _step(route_seed):
            idx = self._moe_routing_128(route_seed)
            idx_n = [i.to("neuron") for i in idx]
            out_n = torch.zeros(T, H, dtype=torch.bfloat16).to("neuron")
            self._moe_expert_loop(x_n, w1_n, w2_n, w3_n, idx_n, out_n, E)
            ref = torch.zeros(T, H, dtype=torch.bfloat16)
            self._moe_expert_loop(x, w1, w2, w3, idx, ref, E)
            torch.testing.assert_close(out_n.cpu(), ref, rtol=2e-2, atol=2e-2)

        # Warmup step compiles the NEFFs for the baked dims (H, H4, T, dtype).
        _step(route_seed=1)
        compiled_after_warmup = {n: len(e._compiled) for n, e in entries.items()}
        exec_after_warmup = {n: e._exec_count for n, e in entries.items()}
        for e in entries.values():
            e._exec_ms.clear()

        # Warm steps: different routing => different per-expert token counts (M).
        # Use seeds spaced apart so the partitions genuinely differ.
        warm_seeds = (5, 9, 13)
        for route_seed in warm_seeds:
            _step(route_seed)

        for name, e in entries.items():
            # 1. dynamic kernel ran during the warm steps (>= one exec per step).
            execs = e._exec_count - exec_after_warmup[name]
            assert execs >= len(warm_seeds), (
                f"{name}: dynamic kernel ran only {execs} times across "
                f"{len(warm_seeds)} warm steps — likely fell back to default lowering"
            )
            # 2. it executed on MULTIPLE distinct M shapes (shape variation).
            assert len(e._exec_ms) >= 2, (
                f"{name}: executed at only Ms={e._exec_ms}; expected multiple "
                f"distinct token counts across routing changes"
            )
            # 3. NO recompile: compiled-NEFF count did not grow after warmup.
            assert len(e._compiled) == compiled_after_warmup[name], (
                f"{name}: recompiled on routing change — compiled NEFFs grew from "
                f"{compiled_after_warmup[name]} to {len(e._compiled)} "
                f"(dynamic shapes not reused)"
            )
