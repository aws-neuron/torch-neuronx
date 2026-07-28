"""Tests for dynamic-shape NKI kernel dispatch.

With TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS=1 and a supported shape, an op routes
to its dynamic NKI kernel instead of the default lowering.

Each op is one ``OpSpec`` row (registry name, input+reference builder, invoke fn);
generic tests fan out over ``_OP_SPECS`` via parametrize, so a new op is one table
entry. Every op is checked for both: the dynamic kernel ran (not a silent
fallback), and the result matches the CPU reference for each supported dtype.
Plus sweeps: arbitrary-H (free-dim remainder tiling) and compile-once (one NEFF
reused across M / the MoE-forward loop).

Skipped unless the flag is EXPORTED before the process starts.
"""

import contextlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import pytest
import torch

import torch_neuronx

_DTYPES = [torch.bfloat16, torch.float32, torch.float16]
_DTYPE_ID = {torch.bfloat16: "bf16", torch.float32: "fp32", torch.float16: "fp16"}
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
    """Assert the dynamic kernel ran (exec_count advanced). Guards the
    silent-fallback trap: a correct result can come from the default lowering."""
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


# ── per-op specification table ───────────────────────────────────────────────
# One row per dynamic op. build(dtype, *, rows, h) -> (run_args, cpu_ref), where
# run_args are neuron operands; run(*run_args) invokes the op on neuron. `rows` is
# the dynamic dim (M / gather-scatter n), `h` the free dim; sweeps vary both.


@dataclass
class OpSpec:
    name: str  # registry entry name, e.g. "exp", "mm", "scatter_add_"
    build: Callable  # (dtype, *, rows, h) -> (run_args, cpu_ref)
    run: Callable  # (*run_args) -> neuron result tensor
    dtypes: tuple = tuple(_DTYPES)
    arbitrary_h: bool = False  # in the free-dim remainder sweep
    compile_once: bool = False  # in the compile-once-per-shape sweep


def _unary_spec(name, fn, **flags):
    """Unary elementwise (silu/exp/sin/cos); 3-D input exercises the flatten to [M, H]."""

    def build(dtype, *, rows=512, h=2048):
        x = torch.randn(1, rows, h, dtype=dtype) / 8
        return (x.to("neuron"),), fn(x)

    return OpSpec(name, build, lambda x: fn(x), **flags)


def _binary_spec(name, fn, **flags):
    """Binary elementwise (add/mul): two operands of identical shape/dtype."""

    def build(dtype, *, rows=512, h=2048):
        x = torch.randn(1, rows, h, dtype=dtype) / 8
        y = torch.randn(1, rows, h, dtype=dtype) / 8
        return (x.to("neuron"), y.to("neuron")), fn(x, y)

    return OpSpec(name, build, lambda x, y: fn(x, y), **flags)


def _mm_spec():
    """mm is bf16-only; K=2048, N=4096 (both % 1024 == 0), M % 128 == 0."""

    def build(dtype, *, rows=256, h=None):
        k, n = 2048, 4096
        a = torch.randn(rows, k, dtype=dtype) / 8
        w = torch.randn(k, n, dtype=dtype) / 8
        return (a.to("neuron"), w.to("neuron")), a @ w

    return OpSpec("mm", build, lambda a, w: a @ w, dtypes=(torch.bfloat16,))


def _index_spec(**flags):
    """x[idx] row-gather: source [T, H], 1-D index of n rows (n % 128 == 0, n <= T)."""

    def build(dtype, *, rows=512, h=2048):
        t = 2 * rows  # n < T exercises the index-padding path
        src = torch.randn(t, h, dtype=dtype) / 8
        idx = torch.randperm(t)[:rows]
        return (src.to("neuron"), idx.to("neuron")), src[idx]

    return OpSpec("index", build, lambda src, idx: src[idx], **flags)


def _scatter_add_spec(**flags):
    """in-place scatter_add_ into [T, H] at n unique rows (MoE token->row unique)."""

    def build(dtype, *, rows=512, h=2048):
        t, n = 2 * rows, rows
        src = torch.randn(n, h, dtype=dtype) / 8
        idx = torch.randperm(t)[:n]
        ref = torch.zeros(t, h, dtype=dtype)
        ref.scatter_add_(0, idx.unsqueeze(1).expand(n, h).to(torch.int64), src)
        out = torch.zeros(t, h, dtype=dtype).to("neuron")
        idx_n = idx.to("neuron").unsqueeze(1).expand(n, h)
        return (out, idx_n, src.to("neuron")), ref

    def run(out, idx_n, src):
        out.scatter_add_(0, idx_n, src)  # in-place -> "scatter_add_" entry
        return out

    return OpSpec("scatter_add_", build, run, **flags)


_OP_SPECS = [
    _unary_spec("silu", torch.nn.functional.silu, arbitrary_h=True, compile_once=True),
    _unary_spec("exp", torch.exp, arbitrary_h=True, compile_once=True),
    _unary_spec("sin", torch.sin, arbitrary_h=True, compile_once=True),
    _unary_spec("cos", torch.cos, arbitrary_h=True, compile_once=True),
    _binary_spec("add", torch.add),
    _binary_spec("mul", torch.mul),
    _mm_spec(),
    _index_spec(arbitrary_h=True),
    _scatter_add_spec(arbitrary_h=True),
]

# one correctness case per op per supported dtype
_OP_DTYPE_CASES = [(s, dt) for s in _OP_SPECS for dt in s.dtypes]
_OP_DTYPE_IDS = [f"{s.name}-{_DTYPE_ID[dt]}" for s, dt in _OP_DTYPE_CASES]
_ARBITRARY_H_SPECS = [s for s in _OP_SPECS if s.arbitrary_h]
_COMPILE_ONCE_SPECS = [s for s in _OP_SPECS if s.compile_once]


@pytest.mark.skipif(
    not _FLAG_ON,
    reason="dynamic NKI disabled — export TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS=1 to run",
)
@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestDynamicNkiKernels:
    SHAPE = (1, 512, 12288)

    def _bf16_neuron(self, *shapes):
        return [torch.randn(s, dtype=torch.bfloat16).to("neuron") for s in shapes]

    # ── per-op correctness: routes to dynamic kernel AND matches CPU ──
    @pytest.mark.parametrize("spec,dtype", _OP_DTYPE_CASES, ids=_OP_DTYPE_IDS)
    def test_dynamic_nki_op(self, spec, dtype):
        rtol, atol = _TOL[dtype]
        run_args, ref = spec.build(dtype)
        with _assert_dynamic_ran(spec.name):
            got = spec.run(*run_args).cpu()
        torch.testing.assert_close(got, ref, rtol=rtol, atol=atol)

    # ── free-dim remainder sweep ──
    # H values hitting the remainder tile: 640=512+128, 576=512+64, 384 (single
    # short tile), 1000=512+488 (non-128 remainder). H is baked, so the kernel
    # tiles the free dim at trace time (512-wide + final short tile), no padding.
    @pytest.mark.parametrize("H", [640, 576, 384, 1000])
    @pytest.mark.parametrize("spec", _ARBITRARY_H_SPECS, ids=lambda s: s.name)
    def test_dynamic_nki_arbitrary_h(self, spec, H):  # noqa: N803
        run_args, ref = spec.build(torch.bfloat16, h=H)
        with _assert_dynamic_ran(spec.name):
            got = spec.run(*run_args).cpu()
        torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2)

    # ── compile-once-per-shape sweep ──
    @pytest.mark.parametrize("spec", _COMPILE_ONCE_SPECS, ids=lambda s: s.name)
    def test_dynamic_nki_compile_once(self, spec):
        """Compile for one M, run on other Ms without recompiling — the
        warm-restart / MoE-routing scenario. Runs at every M while one (H, dtype)
        stays one NEFF (compiled count grows <=1)."""
        entry = _entry(spec.name)
        compiled_before = len(entry._compiled)
        exec_before = entry._exec_count
        entry._exec_ms.clear()

        ms = (128, 256, 512)
        for m in ms:
            run_args, _ = spec.build(torch.bfloat16, rows=m, h=2048)
            spec.run(*run_args).cpu()

        # ran at every M ...
        assert entry._exec_count - exec_before == len(
            ms
        ), f"dynamic {spec.name} did not run for every M"
        assert entry._exec_ms.issuperset(
            set(ms)
        ), f"expected to execute at Ms {set(ms)}, saw {entry._exec_ms}"
        # ... but one (H, dtype) is one compile_key -> <=1 NEFF added.
        assert len(entry._compiled) - compiled_before <= 1, "recompiled across M values"

    # ── add-specific declining / routing (not in the op table) ──
    def test_dynamic_nki_add_alpha_declines(self):
        # alpha != 1 can't be honored by the a+b kernel -> can_handle rejects -> fallback.
        x, y = self._bf16_neuron(self.SHAPE, self.SHAPE)
        with _assert_dynamic_skipped("add"):
            out = torch.add(x, y, alpha=2).cpu()
        torch.testing.assert_close(out, x.cpu() + 2 * y.cpu(), rtol=1e-2, atol=1e-2)

    def test_dynamic_nki_add_alpha_one_routes(self):
        # alpha == 1 (default) -> dynamic path handles it.
        x, y = self._bf16_neuron(self.SHAPE, self.SHAPE)
        with _assert_dynamic_ran("add"):
            out = torch.add(x, y, alpha=1).cpu()
        torch.testing.assert_close(out, x.cpu() + y.cpu(), rtol=1e-2, atol=1e-2)

    def test_dynamic_nki_add_out_writes_in_place(self):
        # out= is the destination buffer; dynamic path writes into it.
        x, y = self._bf16_neuron(self.SHAPE, self.SHAPE)
        buf = torch.empty_like(x)
        with _assert_dynamic_ran("add"):
            torch.add(x, y, out=buf)
        torch.testing.assert_close(buf.cpu(), x.cpu() + y.cpu(), rtol=1e-2, atol=1e-2)

    # ── E2E MoE forward: no recompiles as routing (token counts) change ──
    # per-expert SwiGLU FFN: index -> mm -> silu -> mul -> mm -> scatter_add.
    # Routing changes per-expert token counts (M) each step; each op must reuse
    # one NEFF across all M instead of recompiling.
    MOE_T = 1024  # tokens (multiple of 128)
    MOE_E = 4  # experts
    MOE_H = 2048  # hidden (multiple of mm N-block 1024)
    MOE_H4 = 4096  # FFN inner (multiple of mm N-block 1024)
    # ops the loop exercises. scatter_add_ (in-place) is the variant the loop calls.
    MOE_OPS: ClassVar = ["index", "mm", "silu", "mul", "scatter_add_"]

    def _moe_routing_128(self, route_seed):
        """Partition MOE_T tokens across MOE_E experts (each count a 128-multiple)
        into 1-D int32 index tensors. Different seed -> different M shapes."""
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

    @pytest.mark.parametrize("H", [640, 576, 384, 1000])
    def test_dynamic_nki_arbitrary_h_silu_backward(self, H):  # noqa: N803
        M = 256  # noqa: N806
        x = (torch.randn(M, H, dtype=torch.bfloat16) / 4).to("neuron").requires_grad_(True)
        grad_out = (torch.randn(M, H, dtype=torch.bfloat16) / 4).to("neuron")
        with _assert_dynamic_ran("silu_backward"):
            (dx,) = torch.autograd.grad(torch.nn.functional.silu(x), x, grad_out)
        x_c = x.detach().cpu().float().requires_grad_(True)
        (dx_ref,) = torch.autograd.grad(torch.nn.functional.silu(x_c), x_c, grad_out.cpu().float())
        torch.testing.assert_close(dx.cpu().float(), dx_ref, rtol=1e-2, atol=1e-2)

    # backward mm disambiguation: grad_weight (transposed mat1) -> "mm_k",
    # grad_input/fwd (contiguous mat1) -> "mm". See dispatch_entry._resolve_entry.
    def test_dynamic_nki_mm_backward_routing_no_recompile(self):
        """grad_weight routes to mm_k, everything else to mm; no recompile on a
        token-count change."""
        H, H4 = 2048, 4096  # noqa: N806
        mm = _entry("mm")
        mm_k = _entry("mm_k")

        def _fwd_bwd(n):
            dt = torch.bfloat16
            x = (torch.randn(n, H, dtype=dt) / 64).to("neuron").requires_grad_(True)
            w1 = (torch.randn(H, H4, dtype=dt) / 64).to("neuron").requires_grad_(True)
            w3 = (torch.randn(H4, H, dtype=dt) / 64).to("neuron").requires_grad_(True)
            out = torch.nn.functional.silu(x @ w1) @ w3
            (out * out).sum().backward()  # non-degenerate loss (no stride-0 grad seed)
            torch_neuronx.synchronize()

        _fwd_bwd(384)  # warmup
        mm_neffs, mmk_neffs = len(mm._compiled), len(mm_k._compiled)
        mm_execs, mmk_execs = mm._exec_count, mm_k._exec_count

        _fwd_bwd(512)  # different token count -> must not recompile

        assert mm_k._exec_count > mmk_execs, "grad_weight not routed to mm_k"
        assert mm._exec_count > mm_execs, "fwd/grad_input not routed to mm"
        assert len(mm._compiled) == mm_neffs, "mm recompiled on token change"
        assert len(mm_k._compiled) == mmk_neffs, "mm_k recompiled on token change"

    # backward ops asserted on (mm/mm_k = grad_input/grad_weight).
    MOE_BWD_OPS: ClassVar = ["mm", "mm_k", "silu_backward", "index", "mul"]

    @staticmethod
    def _moe_expert_loop_autograd(x, w1, w2, w3, expert_indices, T, H, num_experts):  # noqa: N803
        """Functional MoE forward (index_add, autograd-friendly)."""
        output = torch.zeros(T, H, dtype=x.dtype, device=x.device)
        for e in range(num_experts):
            idx = expert_indices[e]
            if idx.numel() == 0:
                continue
            x_expert = x[idx]
            gated = torch.nn.functional.silu(x_expert @ w1[e]) * (x_expert @ w2[e])
            out = gated @ w3[e]
            output = output.index_add(0, idx.to(torch.int64), out)
        return output

    def test_dynamic_nki_moe_backward_no_recompile(self):
        """E2E MoE fwd+bwd across changing routing: every backward op runs dynamic,
        grads match CPU, and no NEFF recompiles after warmup."""
        T, E, H, H4 = self.MOE_T, self.MOE_E, self.MOE_H, self.MOE_H4  # noqa: N806
        torch.manual_seed(0)

        def _mk_weights(device):
            g = torch.Generator().manual_seed(1)
            w1 = [(torch.randn(H, H4, dtype=torch.bfloat16, generator=g) / 64) for _ in range(E)]
            w2 = [(torch.randn(H, H4, dtype=torch.bfloat16, generator=g) / 64) for _ in range(E)]
            w3 = [(torch.randn(H4, H, dtype=torch.bfloat16, generator=g) / 64) for _ in range(E)]
            xv = torch.randn(T, H, dtype=torch.bfloat16, generator=g) / 8

            def move(t):
                return t.to(device).detach().requires_grad_(True)

            return [move(w) for w in w1], [move(w) for w in w2], [move(w) for w in w3], move(xv)

        entries = {name: _entry(name) for name in self.MOE_BWD_OPS}

        def _step(route_seed, on_neuron):
            device = "neuron" if on_neuron else "cpu"
            w1, w2, w3, x = _mk_weights(device)
            idx = self._moe_routing_128(route_seed)
            idx_d = [i.to(device) for i in idx]
            out = self._moe_expert_loop_autograd(x, w1, w2, w3, idx_d, T, H, E)
            (out * out).sum().backward()
            if on_neuron:
                torch_neuronx.synchronize()
            grads = [x.grad, *[w.grad for w in w1], *[w.grad for w in w2], *[w.grad for w in w3]]
            return [g.cpu() for g in grads]

        _step(route_seed=1, on_neuron=True)  # warmup
        compiled_after_warmup = {n: len(e._compiled) for n, e in entries.items()}
        exec_after_warmup = {n: e._exec_count for n, e in entries.items()}
        for e in entries.values():
            e._exec_ms.clear()

        warm_seeds = (5, 9, 13)  # different routing -> different token counts
        for route_seed in warm_seeds:
            neuron_grads = _step(route_seed, on_neuron=True)
            cpu_grads = _step(route_seed, on_neuron=False)
            for gi, (gn, gc) in enumerate(zip(neuron_grads, cpu_grads, strict=True)):
                torch.testing.assert_close(gn, gc, rtol=3e-2, atol=3e-2, msg=f"grad[{gi}] mismatch")

        for name, e in entries.items():
            execs = e._exec_count - exec_after_warmup[name]
            assert execs >= len(warm_seeds), f"{name}: ran only {execs}x (fell back?)"
            assert len(e._exec_ms) >= 2, f"{name}: only saw Ms={e._exec_ms}"
            assert (
                len(e._compiled) == compiled_after_warmup[name]
            ), f"{name}: recompiled on routing change"
