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
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

_DTYPES = [torch.bfloat16, torch.float32, torch.float16]
_DTYPE_ID = {
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bool: "bool",
}
_TOL = {
    torch.float32: (1e-4, 1e-4),
    torch.bfloat16: (1e-2, 1e-2),
    torch.float16: (5e-3, 5e-3),
    torch.bool: (0, 0),
}


def _cast_tol(src_dtype, dst_dtype):
    """Exact when the cast is lossless (same dtype, or widening to fp32); the
    destination's tolerance when it rounds."""
    if src_dtype is dst_dtype or dst_dtype is torch.float32:
        return (0, 0)
    return _TOL[dst_dtype]


_FLAG_ON = os.environ.get("TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS") == "1"


def _exact_src(rows, cols, dtype):
    """Bit-exact source values; int dtypes can't use randn."""
    if dtype.is_floating_point:
        return torch.randn(rows, cols, dtype=dtype) / 8
    return torch.arange(rows * cols, dtype=dtype).reshape(rows, cols)


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
def _static_lowering():
    """Run the block on the default lowering. ``get_dynamic_nki_entry`` re-reads the env var per
    call, so flipping it is enough -- the C++ pre-check just gets None back."""
    prev = os.environ.get("TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS")
    os.environ["TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS"] = "0"
    try:
        yield
    finally:
        if prev is None:
            del os.environ["TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS"]
        else:
            os.environ["TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS"] = prev


@contextlib.contextmanager
def _assert_dynamic_skipped(op_name):
    """Assert the dynamic kernel did NOT run (op declined -> default lowering)."""
    e = _entry(op_name)
    before = e._exec_count
    yield e
    assert e._exec_count == before, f"{op_name}: dynamic kernel ran but should have declined"


def _op_of_cache_key(key):
    """Op name from a compilation-cache key."""
    import re

    head = re.split(r"__[0-9a-f]{6,}", key.split("--")[0])[0].split(":")[0].replace("aten::", "")
    m = re.search(r"([a-z_]+_kernel)", head)
    return m.group(1) if m else head.split(".")[0]


def _recompiling_ops(step_fn, warm_seed, probe_seeds):
    """Set of ops that compile a new NEFF as routing changes, via a compilation-cache
    entry diff — path-complete (sees C++-dispatched and glue ops, not just registry
    kernels). step_fn(seed) runs one MoE step at that routing."""
    import torch_neuronx._C as _C

    def keys():
        return {e["cache_key"] for e in _C._get_compilation_cache_entries()}

    step_fn(warm_seed)
    recompiled = set()
    prev = keys()
    for s in probe_seeds:
        step_fn(s)
        now = keys()
        recompiled |= {_op_of_cache_key(k) for k in now - prev}
        prev = now
    return recompiled


# ── per-op specification table ───────────────────────────────────────────────
# One row per dynamic op. build(dtype, *, rows, h) -> (run_args, cpu_ref), where
# run_args are neuron operands; run(*run_args) invokes the op on neuron. `rows` is
# the dynamic dim (M / gather-scatter n), `h` the free dim; sweeps vary both.


def _tol(spec, dtype):
    """The spec's own tolerance if it declares one, else the per-dtype default."""
    return spec.tol if spec.tol is not None else _TOL[dtype]


@dataclass
class OpSpec:
    name: str  # registry entry name, e.g. "exp", "mm", "scatter_add_"
    build: Callable  # (dtype, *, rows, h) -> (run_args, cpu_ref)
    run: Callable  # (*run_args) -> neuron result tensor
    dtypes: tuple = tuple(_DTYPES)
    arbitrary_h: bool = False  # in the free-dim remainder sweep
    compile_once: bool = False  # in the compile-once-per-shape sweep
    non_128: bool = True  # in the non-128-count sweeps (False: op requires whole tiles)
    min_rows: int = 1  # smallest M the op accepts (in-place copy_ needs a whole tile)
    label: str | None = None  # test-id stem; defaults to name (set when name repeats)
    # (rtol, atol) override; None -> the per-dtype _TOL. Ops that are bit-exact by
    # construction must set (0, 0) — a tolerance there can only hide a mis-tiled or
    # partially written result, never a real numeric difference.
    tol: tuple | None = None

    @property
    def id(self) -> str:
        return self.label or self.name


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


def _sigmoid_backward_spec(**flags):
    """sigmoid_backward(grad_output, output): grad_input = grad_output * output * (1 - output).
    output is a valid sigmoid activation (in (0, 1)); reference is the aten op."""

    def build(dtype, *, rows=512, h=2048):
        go = torch.randn(1, rows, h, dtype=dtype) / 8
        out = torch.sigmoid(torch.randn(1, rows, h, dtype=dtype) / 8)
        ref = torch.ops.aten.sigmoid_backward(go, out)
        return (go.to("neuron"), out.to("neuron")), ref

    return OpSpec(
        "sigmoid_backward",
        build,
        lambda go, out: torch.ops.aten.sigmoid_backward(go, out),
        **flags,
    )


def _sum_spec(**flags):
    """Free-dim sum over a 2-D [rows, h] input -> [rows, 1] (keepdim). Reference
    accumulates in fp32 then casts back, matching the kernel's fp32 accumulator +
    single output cast (a bf16 running sum would drift far more than the kernel)."""

    def build(dtype, *, rows=512, h=2048):
        x = torch.randn(rows, h, dtype=dtype) / 8
        ref = x.float().sum(dim=1, keepdim=True).to(dtype)
        return (x.to("neuron"),), ref

    return OpSpec("sum", build, lambda x: x.sum(dim=1, keepdim=True), **flags)


def _compare_spec(name, fn, other=0.0, plant=False, **flags):
    """Compare-with-scalar. The reference is fp32 because both neuron paths compare in fp32 -- a
    bf16 one disagrees at boundaries. ``plant`` writes ``other`` into a third of the columns so
    eq/ne see a mixed mask."""

    def build(dtype, *, rows=512, h=2048):
        x = torch.randn(1, rows, h, dtype=dtype) / 8
        if plant:
            x[..., ::3] = other
        return (x.to("neuron"),), fn(x.float(), other)

    return OpSpec(name, build, lambda x: fn(x, other), tol=(0, 0), non_128=False, **flags)


def _compare_tensor_spec(name, fn, **flags):
    """The .Tensor overload: two same-shape operands, a third of the columns copied across so
    eq/ne see equal values."""

    def build(dtype, *, rows=512, h=2048):
        x = torch.randn(1, rows, h, dtype=dtype) / 8
        y = torch.randn(1, rows, h, dtype=dtype) / 8
        y[..., ::3] = x[..., ::3]
        return (x.to("neuron"), y.to("neuron")), fn(x.float(), y.float())

    return OpSpec(
        name,
        build,
        lambda x, y: fn(x, y),
        tol=(0, 0),
        non_128=False,
        min_rows=128,
        label=f"{name}_tensor",
        **flags,
    )


def _logical_and_spec(bool_operands, **flags):
    """bool operands take the packed kernel, float operands the mask kernel -- the ALU evaluates
    ``(a != 0) & (b != 0)``, hence the planted zeros. tol=(0, 0): no numeric error term."""

    def build(dtype, *, rows=512, h=2048):
        if bool_operands:
            a = torch.rand(1, rows, h) > 0.5
            b = torch.rand(1, rows, h) > 0.5
        else:
            a = torch.randn(1, rows, h, dtype=dtype) / 8
            b = torch.randn(1, rows, h, dtype=dtype) / 8
            a[..., ::4] = 0.0
            b[..., ::3] = 0.0
        return (a.to("neuron"), b.to("neuron")), torch.logical_and(a, b)

    return OpSpec(
        "logical_and",
        build,
        torch.logical_and,
        dtypes=(torch.bool,) if bool_operands else tuple(_DTYPES),
        tol=(0, 0),
        min_rows=128,
        label="logical_and_bool" if bool_operands else "logical_and_float",
        **flags,
    )


def _div_spec(form, **flags):
    """div by a same-shape tensor, a [*, 1] column, or a scalar; divisors stay away from zero.

    The reference mirrors the hardware (``a * reciprocal(b)`` in fp32, which is also what the
    static lowering does), so tol=(0, 0) holds and no 1-ULP slack can hide a mis-tiled result.
    Accuracy of recip-then-multiply is asserted separately in test_dynamic_nki_div_accuracy."""
    divisor_scalar = 3.5

    def build(dtype, *, rows=512, h=2048):
        x = torch.randn(1, rows, h, dtype=dtype) / 8
        if form == "scalar":
            return (x.to("neuron"),), (x.float() * (1.0 / divisor_scalar)).to(dtype)
        shape = (1, rows, h) if form == "tensor" else (1, rows, 1)
        y = (torch.randn(*shape, dtype=dtype) / 8).abs() + 0.05
        return (x.to("neuron"), y.to("neuron")), (x.float() * y.float().reciprocal()).to(dtype)

    run = (lambda x: torch.div(x, divisor_scalar)) if form == "scalar" else torch.div
    return OpSpec("div", build, run, tol=(0, 0), min_rows=128, label=f"div_{form}", **flags)


def _to_copy_spec(dst_dtype, **flags):
    """``x.to(dst_dtype)`` (aten::_to_copy): dtype cast, fresh output.

    The sweep's ``dtype`` is the SOURCE dtype — one spec per target, sources being
    the other two (same-dtype declines). Tolerance is the destination dtype's,
    since the cast rounds to that precision. min_rows matches ``copy_``: since #1687
    the cast adapter declines sub-128 M instead of padding it."""

    def build(dtype, *, rows=512, h=2048):
        x = torch.randn(rows, h, dtype=dtype) / 8
        return (x.to("neuron"),), x.to(dst_dtype)

    return OpSpec(
        "_to_copy",
        build,
        lambda x: x.to(dst_dtype),
        dtypes=tuple(d for d in _DTYPES if d != dst_dtype),
        label=f"_to_copy_{_DTYPE_ID[dst_dtype]}",
        min_rows=128,
        # Widening to fp32 is lossless (bf16 and fp16 are both exactly
        # representable) -> exact; narrowing rounds to the destination's precision.
        tol=(0, 0) if dst_dtype is torch.float32 else _TOL[dst_dtype],
        **flags,
    )


def _copy_spec(**flags):
    """``dst.copy_(src)`` (aten::copy_): in-place, same dtype — a pure data move, so
    the reference is `src` itself and the comparison is exact. min_rows keeps the
    sub-tile counts, which decline, out of the shared sweeps."""

    def build(dtype, *, rows=512, h=2048):
        src = torch.randn(rows, h, dtype=dtype) / 8
        dst = torch.zeros(rows, h, dtype=dtype)
        return (dst.to("neuron"), src.to("neuron")), src

    return OpSpec(
        "copy_",
        build,
        lambda dst, src: dst.copy_(src),
        min_rows=128,
        tol=(0, 0),
        **flags,
    )


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


def _gather_spec(**flags):
    """gather(0, idx) with the index constant along the free dim — a row gather in
    disguise, as produced by the backward of a row scatter_add_. Same [T, H] source as
    _index_spec. The expand must come AFTER the device copy: ``.to()`` on an expanded
    view materializes it contiguous, and a contiguous index is not a row gather."""

    def build(dtype, *, rows=512, h=2048):
        t = 2 * rows  # n < T exercises the index-padding path
        src = torch.randn(t, h, dtype=dtype) / 8
        idx = torch.randperm(t)[:rows]
        idx_n = idx.to("neuron").unsqueeze(1).expand(rows, h)
        return (src.to("neuron"), idx_n), src[idx]

    # tol=(0, 0): every output element is a verbatim copy of a source row element.
    return OpSpec("gather", build, lambda src, idx: src.gather(0, idx), tol=(0, 0), **flags)


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


def _index_put_spec(**flags):
    """in-place index_put_ (accumulate=True) into [T, H] at n unique rows.

    Lowers to aten::_index_put_impl_ — the same row-accumulate as scatter_add_,
    but reached via the index_put_ op. This is the op the MoE combine and the
    loss's masked assignment emit; a single 1-D integer row index on dim 0 with
    accumulate=True routes to the dynamic row-scatter kernel.
    """

    def build(dtype, *, rows=512, h=2048):
        t, n = 2 * rows, rows  # n < T, unique target rows
        src = torch.randn(n, h, dtype=dtype) / 8
        idx = torch.randperm(t)[:n]
        ref = torch.zeros(t, h, dtype=dtype)
        ref.index_put_((idx.to(torch.int64),), src, accumulate=True)
        out = torch.zeros(t, h, dtype=dtype).to("neuron")
        return (out, idx.to("neuron"), src.to("neuron")), ref

    def run(out, idx_n, src):
        out.index_put_((idx_n,), src, accumulate=True)  # -> aten::_index_put_impl_
        return out

    return OpSpec("_index_put_impl_", build, run, label="index_put", **flags)


def _where_spec(**flags):
    """where.self: bool condition selects between two float operands of one shape.
    tol=(0, 0): every output element is a verbatim copy of an input element."""

    def build(dtype, *, rows=512, h=2048):
        cond = torch.rand(1, rows, h) > 0.5
        a = torch.randn(1, rows, h, dtype=dtype) / 8
        b = torch.randn(1, rows, h, dtype=dtype) / 8
        return (cond.to("neuron"), a.to("neuron"), b.to("neuron")), torch.where(cond, a, b)

    return OpSpec("where", build, torch.where, tol=(0, 0), **flags)


def _bitwise_or_spec(**flags):
    """bitwise_or.Tensor: two bool masks (bitwise OR == logical OR on 0/1).
    tol=(0, 0): a bitwise op on bool has no numeric error term."""

    def build(dtype, *, rows=512, h=2048):
        a = torch.rand(1, rows, h) > 0.5
        b = torch.rand(1, rows, h) > 0.5
        return (a.to("neuron"), b.to("neuron")), torch.bitwise_or(a, b)

    return OpSpec("bitwise_or", build, torch.bitwise_or, dtypes=(torch.bool,), tol=(0, 0), **flags)


_CMP_NAMES = ["ge", "lt", "ne", "le", "eq"]

_OP_SPECS = [
    _unary_spec("silu", torch.nn.functional.silu, arbitrary_h=True, compile_once=True),
    _unary_spec("exp", torch.exp, arbitrary_h=True, compile_once=True),
    _unary_spec("sin", torch.sin, arbitrary_h=True, compile_once=True),
    _unary_spec("cos", torch.cos, arbitrary_h=True, compile_once=True),
    _unary_spec("sigmoid", torch.sigmoid, arbitrary_h=True, compile_once=True),
    _log_spec(arbitrary_h=True, compile_once=True),
    _unary_spec("neg", torch.neg, arbitrary_h=True, compile_once=True),
    # tol=(0, 0): every element is exactly 1.0, representable in every dtype.
    _unary_spec("ones_like", torch.ones_like, arbitrary_h=True, compile_once=True, tol=(0, 0)),
    _where_spec(arbitrary_h=True, compile_once=True),
    _bitwise_or_spec(arbitrary_h=True, compile_once=True),
    _compare_spec("ge", torch.ge, arbitrary_h=True, compile_once=True),
    _compare_spec("lt", torch.lt, arbitrary_h=True, compile_once=True),
    _compare_spec("ne", torch.ne, arbitrary_h=True, compile_once=True, plant=True),
    _compare_spec("le", torch.le, arbitrary_h=True, compile_once=True),
    _compare_spec("eq", torch.eq, arbitrary_h=True, compile_once=True, plant=True),
    # the .Tensor overload of each: same kernel family, second full-width operand
    _compare_tensor_spec("ge", torch.ge, arbitrary_h=True, compile_once=True),
    _compare_tensor_spec("lt", torch.lt),
    _compare_tensor_spec("ne", torch.ne),
    _compare_tensor_spec("le", torch.le),
    _compare_tensor_spec("eq", torch.eq, arbitrary_h=True, compile_once=True),
    _logical_and_spec(True, arbitrary_h=True, compile_once=True),
    _logical_and_spec(False, arbitrary_h=True, compile_once=True),
    _div_spec("tensor", arbitrary_h=True, compile_once=True),
    _div_spec("row", arbitrary_h=True, compile_once=True),
    _div_spec("scalar", arbitrary_h=True, compile_once=True),
    _binary_spec("add", torch.add, compile_once=True),
    _binary_spec("sub", torch.sub, compile_once=True),
    _binary_spec("mul", torch.mul),
    _sigmoid_backward_spec(compile_once=True),
    _sum_spec(arbitrary_h=True, compile_once=True),
    _to_copy_spec(torch.float32, arbitrary_h=True, compile_once=True),
    # No arbitrary_h / compile_once for the bf16 target: those sweeps hardcode a
    # bf16 source, which would be a same-dtype (declined) _to_copy.
    _to_copy_spec(torch.bfloat16),
    _copy_spec(arbitrary_h=True, compile_once=True),
    _mm_spec(),
    _index_spec(arbitrary_h=True),
    _gather_spec(arbitrary_h=True),
    _scatter_add_spec(arbitrary_h=True),
    _index_put_spec(arbitrary_h=True),
]

# one correctness case per op per supported dtype
_OP_DTYPE_CASES = [(s, dt) for s in _OP_SPECS for dt in s.dtypes]
_OP_DTYPE_IDS = [f"{s.id}-{_DTYPE_ID[dt]}" for s, dt in _OP_DTYPE_CASES]
_ARBITRARY_H_SPECS = [s for s in _OP_SPECS if s.arbitrary_h]
_COMPILE_ONCE_SPECS = [s for s in _OP_SPECS if s.compile_once]
# Non-128 sweeps skip the ops that require whole 128-row tiles; spec.min_rows drops
# the sub-tile counts per op.
_NON_128_CASES = [(s, dt) for s, dt in _OP_DTYPE_CASES if s.non_128]
_NON_128_IDS = [f"{s.id}-{_DTYPE_ID[dt]}" for s, dt in _NON_128_CASES]
_NON_128_ONCE_SPECS = [s for s in _COMPILE_ONCE_SPECS if s.non_128]


@pytest.mark.skipif(
    not _FLAG_ON,
    reason="dynamic NKI disabled — export TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS=1 to run",
)
@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestDynamicNkiKernels:
    SHAPE = (1, 512, 12288)

    @pytest.fixture(autouse=True)
    def _seed_rng(self):
        # Specs' build() draw inputs from global RNG (randn/randperm). Seed per test
        # so a case is reproducible regardless of suite ordering — otherwise a draw
        # near the fp16 tolerance edge can pass alone but fail in the full suite.
        torch.manual_seed(0)

    def _bf16_neuron(self, *shapes):
        return [torch.randn(s, dtype=torch.bfloat16).to("neuron") for s in shapes]

    def test_dynamic_packed_int64_gather_compile_once(self):
        from torch_neuronx.packed_storage import packed_int64_execution_view

        def run(numel):
            values = torch.arange(numel, dtype=torch.int64)
            low = packed_int64_execution_view(values.to("neuron"))
            torch.testing.assert_close(low.cpu(), values.to(torch.int32))

        run(63)
        run(128)
        before = {entry["cache_key"] for entry in torch_neuronx._C._get_compilation_cache_entries()}
        assert any("dynamic_packed_int64_gather_small--" in key for key in before)
        assert any("dynamic_packed_int64_gather--" in key for key in before)

        for numel in (1, 127, 129, 190, 218, 289, 327, 654):
            run(numel)

        after = {entry["cache_key"] for entry in torch_neuronx._C._get_compilation_cache_entries()}
        assert after == before, f"dynamic numel sweep compiled new NEFFs: {after - before}"

    # ── per-op correctness: routes to dynamic kernel AND matches CPU ──
    @pytest.mark.parametrize("spec,dtype", _OP_DTYPE_CASES, ids=_OP_DTYPE_IDS)
    def test_dynamic_nki_op(self, spec, dtype):
        rtol, atol = _tol(spec, dtype)
        run_args, ref = spec.build(dtype)
        with _assert_dynamic_ran(spec.name):
            got = spec.run(*run_args).cpu()
        torch.testing.assert_close(got, ref, rtol=rtol, atol=atol)

    # ── free-dim remainder sweep ──
    # H values hitting the remainder tile: 640=512+128, 576=512+64, 384 (single
    # short tile), 1000=512+488 (non-128 remainder). H is baked, so the kernel
    # tiles the free dim at trace time (512-wide + final short tile), no padding.
    @pytest.mark.parametrize("H", [640, 576, 384, 1000])
    @pytest.mark.parametrize("spec", _ARBITRARY_H_SPECS, ids=lambda s: s.id)
    def test_dynamic_nki_arbitrary_h(self, spec, H):  # noqa: N803
        rtol, atol = _tol(spec, torch.bfloat16)
        run_args, ref = spec.build(torch.bfloat16, h=H)
        with _assert_dynamic_ran(spec.name):
            got = spec.run(*run_args).cpu()
        torch.testing.assert_close(got, ref, rtol=rtol, atol=atol)

    # ── single-tile sweep: rows=128 -> num_m_tiles == 1 -> the runtime loop
    # (nl.fori_loop) runs exactly once. Guards the count==1 boundary: body must
    # execute once and cross-iteration state (m_tile_start / k_off) must be
    # initialized correctly. rows=128 keeps mm (M % 128 == 0) and gather/scatter
    # (n % 128 == 0) valid, so one case covers every dynamic kernel. ──
    @pytest.mark.parametrize("spec,dtype", _OP_DTYPE_CASES, ids=_OP_DTYPE_IDS)
    def test_dynamic_nki_single_tile(self, spec, dtype):
        rtol, atol = _tol(spec, dtype)
        run_args, ref = spec.build(dtype, rows=128)
        with _assert_dynamic_ran(spec.name):
            got = spec.run(*run_args).cpu()
        torch.testing.assert_close(got, ref, rtol=rtol, atol=atol)

    # ── non-128 count sweep: the dynamic dim (M / gather-scatter n) is padded up
    # to the next 128-multiple inside prepare_io; the op still routes dynamically
    # and stays correct after the padded rows are sliced off. 200->256, 129->256,
    # 383->384 exercise the pad path (and 383 is just below a tile boundary). ──
    @pytest.mark.parametrize("rows", [200, 129, 383])
    @pytest.mark.parametrize("spec,dtype", _NON_128_CASES, ids=_NON_128_IDS)
    def test_dynamic_nki_non_128_count(self, spec, dtype, rows):
        rtol, atol = _tol(spec, dtype)
        run_args, ref = spec.build(dtype, rows=rows)
        with _assert_dynamic_ran(spec.name):
            got = spec.run(*run_args).cpu()
        torch.testing.assert_close(got, ref, rtol=rtol, atol=atol)

    # ── compile-once-per-shape sweep ──
    @pytest.mark.parametrize("spec", _COMPILE_ONCE_SPECS, ids=lambda s: s.id)
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

    # ── non-128 counts don't recompile: the padded dim is not in the compile_key,
    # so distinct non-128 counts (padding to distinct tiles) reuse one NEFF. This
    # is the recompile-avoidance guarantee for realistic (unfloored) routing. ──
    @pytest.mark.parametrize("spec", _NON_128_ONCE_SPECS, ids=lambda s: s.id)
    def test_dynamic_nki_non_128_no_recompile(self, spec):
        entry = _entry(spec.name)
        compiled_before = len(entry._compiled)
        exec_before = entry._exec_count
        entry._exec_ms.clear()

        # 100/200/300 -> three distinct tile counts; min_rows drops the sub-tile one.
        rows = tuple(m for m in (100, 200, 300) if m >= spec.min_rows)
        for m in rows:
            run_args, _ = spec.build(torch.bfloat16, rows=m, h=2048)
            spec.run(*run_args).cpu()

        assert entry._exec_count - exec_before == len(
            rows
        ), f"dynamic {spec.name} did not run for every non-128 count"
        assert entry._exec_ms.issuperset(
            set(rows)
        ), f"expected exec at actual (unpadded) counts {set(rows)}, saw {entry._exec_ms}"
        # padded dim not in compile_key -> one (H, dtype) is one NEFF, no recompile.
        assert (
            len(entry._compiled) - compiled_before <= 1
        ), "recompiled across non-128 counts (padded dim leaked into compile_key?)"

    # ── comparison-specific: plan shapes, arbitrary M, NEFF reuse, overloads ──
    _CMP_FNS: ClassVar = {
        "ge": torch.ge,
        "lt": torch.lt,
        "ne": torch.ne,
        "le": torch.le,
        "eq": torch.eq,
    }

    @pytest.mark.parametrize("name", _CMP_NAMES)
    def test_dynamic_nki_compare_plan_shapes(self, name):
        """The plan's shapes: [1, seq] and 1-D collapse to [numel, 1], [rows, H]
        keeps the last dim as H."""
        fn = self._CMP_FNS[name]
        for shape in ((1, 512), (512,), (6144, 720)):
            x = (torch.randn(*shape, dtype=torch.bfloat16) / 8).to("neuron")
            with _assert_dynamic_ran(name):
                got = fn(x, 0.0).cpu()
            torch.testing.assert_close(got, fn(x.cpu().float(), 0.0), rtol=0, atol=0)

    @pytest.mark.parametrize("name", _CMP_NAMES)
    def test_dynamic_nki_compare_arbitrary_m_no_pad(self, name):
        """Any M >= 128, no padding: the tail tile covers the remainder. rem=1 and
        rem=127 included; the whole sweep must stay on one NEFF."""
        fn = self._CMP_FNS[name]
        entry = _entry(name)
        compiled_before = len(entry._compiled)
        for m in (128, 129, 200, 255, 383, 1001):
            x = (torch.randn(m, 2048, dtype=torch.bfloat16) / 8).to("neuron")
            with _assert_dynamic_ran(name):
                got = fn(x, 0.0).cpu()
            torch.testing.assert_close(got, fn(x.cpu().float(), 0.0), rtol=0, atol=0, msg=f"M={m}")
        assert len(entry._compiled) - compiled_before <= 1, "recompiled across M values"

    @pytest.mark.parametrize("name", _CMP_NAMES)
    def test_dynamic_nki_compare_m_below_one_tile_declines(self, name):
        """M < 128 leaves no room for the tail tile -> declined."""
        fn = self._CMP_FNS[name]
        x = (torch.randn(4, 8, dtype=torch.bfloat16) / 8).to("neuron")
        with _assert_dynamic_skipped(name):
            got = fn(x, 0.0).cpu()
        torch.testing.assert_close(got, fn(x.cpu().float(), 0.0), rtol=0, atol=0)

    @pytest.mark.parametrize("name", _CMP_NAMES)
    def test_dynamic_nki_compare_no_recompile_on_seqlen(self, name):
        """Warm-restart scenario: same op, different seq length (plan #8/#10 go
        512 -> 4096). One NEFF, including seq lengths that are not 128-multiples."""
        fn = self._CMP_FNS[name]
        entry = _entry(name)
        compiled_before = len(entry._compiled)
        exec_before = entry._exec_count
        seqs = (512, 4096, 777, 4097, 256)
        for seq in seqs:
            x = (torch.randn(1, seq, dtype=torch.bfloat16) / 8).to("neuron")
            got = fn(x, 0.0).cpu()
            torch.testing.assert_close(
                got, fn(x.cpu().float(), 0.0), rtol=0, atol=0, msg=f"seq={seq}"
            )
        assert entry._exec_count - exec_before == len(seqs), f"{name} skipped a seq"
        assert len(entry._compiled) - compiled_before <= 1, f"{name}: recompiled across seq lengths"

    @pytest.mark.parametrize("name", _CMP_NAMES)
    def test_dynamic_nki_compare_scalar_no_recompile(self, name):
        """The threshold is a runtime operand, not baked: distinct scalars must NOT
        add NEFFs. (H is unique to this test so sibling tests can't mask a miss.)"""
        fn = self._CMP_FNS[name]
        entry = _entry(name)
        h = 1234  # no sibling spec uses this H
        x = (torch.randn(256, h, dtype=torch.bfloat16) / 8).to("neuron")
        fn(x, 0.0).cpu()  # compile the (h, bf16) NEFF
        compiled_before = len(entry._compiled)
        for other in (0.3, -1.5, 7.25, 0.0):
            got = fn(x, other).cpu()
            torch.testing.assert_close(
                got, fn(x.cpu().float(), other), rtol=0, atol=0, msg=f"other={other}"
            )
        assert len(entry._compiled) == compiled_before, (
            f"{name}: a distinct threshold compiled a new NEFF -- the scalar leaked "
            f"into the compile_key"
        )

    @pytest.mark.parametrize("name", _CMP_NAMES)
    def test_dynamic_nki_compare_tensor_plan_shapes(self, name):
        """The .Tensor overload on the plan's shapes, including the 1-D / [1, seq] cases that
        collapse to [numel, 1]."""
        fn = self._CMP_FNS[name]
        for shape in ((1, 512), (512,), (6144, 720), (1001, 313)):
            x = (torch.randn(*shape, dtype=torch.bfloat16) / 8).to("neuron")
            y = (torch.randn(*shape, dtype=torch.bfloat16) / 8).to("neuron")
            with _assert_dynamic_ran(name):
                got = fn(x, y).cpu()
            expected = fn(x.cpu().float(), y.cpu().float())
            torch.testing.assert_close(got, expected, rtol=0, atol=0, msg=f"shape={shape}")

    @pytest.mark.parametrize("name", _CMP_NAMES)
    def test_dynamic_nki_compare_tensor_no_recompile_on_seqlen(self, name):
        """Warm restart: one NEFF across seq lengths, 128-multiples or not."""
        fn = self._CMP_FNS[name]
        entry = _entry(name)
        compiled_before = len(entry._compiled)
        exec_before = entry._exec_count
        seqs = (512, 4096, 777, 4097, 256)
        for seq in seqs:
            x = (torch.randn(seq, 128, dtype=torch.bfloat16) / 8).to("neuron")
            y = (torch.randn(seq, 128, dtype=torch.bfloat16) / 8).to("neuron")
            got = fn(x, y).cpu()
            expected = fn(x.cpu().float(), y.cpu().float())
            torch.testing.assert_close(got, expected, rtol=0, atol=0, msg=f"seq={seq}")
        assert entry._exec_count - exec_before == len(seqs), f"{name} skipped a seq"
        assert len(entry._compiled) - compiled_before <= 1, f"{name}: recompiled across seqs"

    @pytest.mark.parametrize("name", _CMP_NAMES)
    def test_dynamic_nki_compare_overloads_are_separate_neffs(self, name):
        """Scalar and tensor `other` are different kernels, so the overload must be in the
        compile_key -- one NEFF each."""
        fn = self._CMP_FNS[name]
        entry = _entry(name)
        h = 1178  # unique to this test, so sibling specs can't mask a miss
        x = (torch.randn(256, h, dtype=torch.bfloat16) / 8).to("neuron")
        y = (torch.randn(256, h, dtype=torch.bfloat16) / 8).to("neuron")
        keys_before = set(entry._compiled)
        fn(x, 0.0).cpu()
        fn(x, y).cpu()
        added = set(entry._compiled) - keys_before
        assert added == {(h, torch.bfloat16, False), (h, torch.bfloat16, True)}, added

    @pytest.mark.parametrize("name", _CMP_NAMES)
    def test_dynamic_nki_compare_tensor_operand_declines(self, name):
        """Operands one access pattern can't read: broadcast shape, dtype mismatch, or a 0-d
        device tensor (reading its value would sync)."""
        fn = self._CMP_FNS[name]
        x = (torch.randn(256, 2048, dtype=torch.bfloat16) / 8).to("neuron")
        others = (
            torch.randn(1, 2048, dtype=torch.bfloat16).to("neuron"),  # broadcast
            torch.randn(256, 2048, dtype=torch.float32).to("neuron"),  # dtype mismatch
            torch.tensor(0.5, dtype=torch.bfloat16).to("neuron"),  # 0-d on device
        )
        for other in others:
            assert not _entry(name).can_handle(x, other), f"took {tuple(other.shape)}"
            with _assert_dynamic_skipped(name):
                got = fn(x, other).cpu()
            expected = fn(x.cpu().float(), other.cpu().float())
            torch.testing.assert_close(got, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("name", _CMP_NAMES)
    def test_dynamic_nki_compare_nonfinite_declines(self, name):
        """NaN/inf compare semantics on the ALU are unverified -> declined."""
        fn = self._CMP_FNS[name]
        x = (torch.randn(256, 2048, dtype=torch.bfloat16) / 8).to("neuron")
        for other in (float("nan"), float("inf")):
            with _assert_dynamic_skipped(name):
                fn(x, other).cpu()

    @pytest.mark.parametrize("name", _CMP_NAMES)
    def test_dynamic_nki_compare_out_writes_in_place(self, name):
        """Unpadded, so the kernel writes into out= directly -- result aliases it."""
        fn = self._CMP_FNS[name]
        x = (torch.randn(256, 2048, dtype=torch.bfloat16) / 8).to("neuron")
        buf = torch.empty(256, 2048, dtype=torch.bool).to("neuron")
        with _assert_dynamic_ran(name):
            res = fn(x, 0.0, out=buf)
        assert res.data_ptr() == buf.data_ptr(), "out= was not written in place"
        torch.testing.assert_close(buf.cpu(), fn(x.cpu().float(), 0.0), rtol=0, atol=0)

    @pytest.mark.parametrize("name", _CMP_NAMES)
    def test_dynamic_nki_compare_noncontiguous_out_is_honored(self, name):
        """out= must hold the result and be the returned tensor (the `Tensor(a!) out`
        contract), non-contiguous included. can_handle declines a dest it can't alias --
        serving it means a stride-keyed copy_ back -- so the static lowering honours it."""
        fn = self._CMP_FNS[name]
        entry = _entry(name)
        x = (torch.randn(256, 2048, dtype=torch.bfloat16) / 8).to("neuron")

        # transposed view, pre-zeroed so a missed write is deterministically detectable
        buf = torch.zeros(2048, 256, dtype=torch.bool).to("neuron").t()
        assert buf.shape == (256, 2048) and not buf.is_contiguous()

        assert not entry.can_handle(x, 0.0, out=buf), "can_handle took a dest it can't alias"
        assert entry.can_handle(x, 0.0, out=buf.contiguous())

        with _assert_dynamic_skipped(name):
            res = fn(x, 0.0, out=buf)
        expected = fn(x.cpu().float(), 0.0)
        torch.testing.assert_close(res.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(buf.cpu(), expected, rtol=0, atol=0)
        assert res.data_ptr() == buf.data_ptr(), "out= was not written in place"

    # ── add-specific declining / routing (not in the op table) ──
    # ── logical_and: two kernels behind one entry (bool packed / float mask) ──
    def test_dynamic_nki_logical_and_plan_shape(self):
        """Plan #56: [6144, 720] bool masks, warm -> cold row count on one NEFF. A 1-byte operand
        load with a dynamic offset is exactly what the int16 view exists to fix."""
        entry = _entry("logical_and")
        compiled_before = len(entry._compiled)
        for rows in (6144, 12288, 6145):
            a = torch.rand(rows, 720) > 0.5
            b = torch.rand(rows, 720) > 0.5
            with _assert_dynamic_ran("logical_and"):
                got = torch.logical_and(a.to("neuron"), b.to("neuron")).cpu()
            ref = torch.logical_and(a, b)
            torch.testing.assert_close(got, ref, rtol=0, atol=0, msg=f"rows={rows}")
        assert len(entry._compiled) - compiled_before <= 1, "recompiled across row counts"

    def test_dynamic_nki_logical_and_odd_h_declines_for_bool_only(self):
        """The int16 view needs an even H; float operands load at full width, so odd H is fine."""
        entry = _entry("logical_and")
        a_bool = (torch.rand(256, 721) > 0.5).to("neuron")
        b_bool = (torch.rand(256, 721) > 0.5).to("neuron")
        assert not entry.can_handle(a_bool, b_bool)
        with _assert_dynamic_skipped("logical_and"):
            got = torch.logical_and(a_bool, b_bool).cpu()
        torch.testing.assert_close(got, torch.logical_and(a_bool.cpu(), b_bool.cpu()))

        a = (torch.randn(256, 721, dtype=torch.bfloat16) / 8).to("neuron")
        b = (torch.randn(256, 721, dtype=torch.bfloat16) / 8).to("neuron")
        with _assert_dynamic_ran("logical_and"):
            got = torch.logical_and(a, b).cpu()
        torch.testing.assert_close(got, torch.logical_and(a.cpu(), b.cpu()), rtol=0, atol=0)

    def test_dynamic_nki_logical_and_mixed_dtype_declines(self):
        """One access pattern serves both operands, so a promoted bool/float pair declines."""
        entry = _entry("logical_and")
        a = (torch.rand(256, 720) > 0.5).to("neuron")
        b = (torch.randn(256, 720, dtype=torch.bfloat16) / 8).to("neuron")
        assert not entry.can_handle(a, b)
        with _assert_dynamic_skipped("logical_and"):
            got = torch.logical_and(a, b).cpu()
        torch.testing.assert_close(got, torch.logical_and(a.cpu(), b.cpu()), rtol=0, atol=0)

    @pytest.mark.parametrize("bool_operands", [True, False], ids=["bool", "float"])
    def test_dynamic_nki_logical_and_out_writes_in_place(self, bool_operands):
        """Nothing is padded, so out= backs the output directly -- including the bool dest the
        packed kernel writes through an int16 view."""
        if bool_operands:
            a = (torch.rand(256, 720) > 0.5).to("neuron")
            b = (torch.rand(256, 720) > 0.5).to("neuron")
        else:
            a = (torch.randn(256, 720, dtype=torch.bfloat16) / 8).to("neuron")
            b = (torch.randn(256, 720, dtype=torch.bfloat16) / 8).to("neuron")
        buf = torch.empty(256, 720, dtype=torch.bool).to("neuron")
        with _assert_dynamic_ran("logical_and"):
            res = torch.logical_and(a, b, out=buf)
        assert res.data_ptr() == buf.data_ptr(), "out= was not written in place"
        torch.testing.assert_close(buf.cpu(), torch.logical_and(a.cpu(), b.cpu()), rtol=0, atol=0)

    # ── div: three divisor forms, one NEFF each ──
    def test_dynamic_nki_div_plan_shapes(self):
        """Plan #38's [512,4] / [512,1] and larger reruns: one NEFF for every row count."""
        entry = _entry("div")
        compiled_before = len(entry._compiled)
        for rows in (512, 4096, 513):
            x = torch.randn(rows, 4, dtype=torch.bfloat16) / 8
            d = (torch.randn(rows, 1, dtype=torch.bfloat16) / 8).abs() + 0.05
            with _assert_dynamic_ran("div"):
                got = torch.div(x.to("neuron"), d.to("neuron")).cpu()
            ref = (x.float() * d.float().reciprocal()).to(torch.bfloat16)
            torch.testing.assert_close(got, ref, rtol=0, atol=0, msg=f"rows={rows}")
        assert len(entry._compiled) - compiled_before <= 1, "recompiled across row counts"

    def test_dynamic_nki_div_scalar_no_recompile(self):
        """The divisor column is a runtime operand, so distinct scalars must not add NEFFs. H is
        unique to this test so sibling specs can't mask a miss."""
        entry = _entry("div")
        h = 1102
        x = (torch.randn(256, h, dtype=torch.bfloat16) / 8).to("neuron")
        torch.div(x, 2.0).cpu()  # compile the (h, bf16, scalar) NEFF
        compiled_before = len(entry._compiled)
        for other in (0.3, -1.5, 7.25, 2.0):
            got = torch.div(x, other).cpu()
            ref = (x.cpu().float() * (1.0 / other)).to(torch.bfloat16)
            torch.testing.assert_close(got, ref, rtol=0, atol=0, msg=f"other={other}")
        assert len(entry._compiled) == compiled_before, "a distinct divisor compiled a NEFF"

    @pytest.mark.parametrize("dtype", _DTYPES, ids=[_DTYPE_ID[d] for d in _DTYPES])
    def test_dynamic_nki_div_accuracy(self, dtype):
        """a * reciprocal(b) in fp32 lands on the correctly rounded quotient exactly for
        bf16/fp16, and within one ULP for fp32 (the reciprocal's own rounding)."""
        x = torch.randn(256, 512, dtype=dtype) / 8
        y = (torch.randn(256, 512, dtype=dtype) / 8).abs() + 0.05
        with _assert_dynamic_ran("div"):
            got = torch.div(x.to("neuron"), y.to("neuron")).cpu()
        exact = (x.double() / y.double()).to(dtype)
        if dtype is torch.float32:
            torch.testing.assert_close(got, exact, rtol=2**-23, atol=0)
        else:
            torch.testing.assert_close(got, exact, rtol=0, atol=0)

    @pytest.mark.parametrize("form", ["tensor", "row"])
    def test_dynamic_nki_div_matches_static_lowering(self, form):
        """Bit-for-bit against the default lowering, which does the same recip-then-multiply.
        Not the scalar form: static rounds the divisor to the operand dtype first, the dynamic
        path keeps the fp32 value (and is the closer of the two)."""
        x = torch.randn(300, 256, dtype=torch.bfloat16) / 8
        shape = (300, 256) if form == "tensor" else (300, 1)
        y = (torch.randn(*shape, dtype=torch.bfloat16) / 8).abs() + 0.05
        xn, yn = x.to("neuron"), y.to("neuron")
        entry = _entry("div")
        with _assert_dynamic_ran("div"):
            dynamic = torch.div(xn, yn).cpu()
        # Look the entry up before the toggle: get_dynamic_nki_entry is itself flag-gated.
        exec_before = entry._exec_count
        with _static_lowering():
            static = torch.div(xn, yn).cpu()
        assert entry._exec_count == exec_before, "flag off but the dynamic kernel still ran"
        torch.testing.assert_close(dynamic, static, rtol=0, atol=0)

    def test_dynamic_nki_div_by_zero_matches_torch(self):
        """The ALU produces inf / -inf / nan just as torch does, so a zero divisor needs no gate."""
        x = torch.tensor([[1.0, -2.0, 0.0, 3.0]] * 256, dtype=torch.float32)
        z = torch.zeros(256, 4, dtype=torch.float32)
        with _assert_dynamic_ran("div"):
            got = torch.div(x.to("neuron"), z.to("neuron")).cpu()
        torch.testing.assert_close(got, torch.div(x, z), rtol=0, atol=0, equal_nan=True)

    def test_dynamic_nki_div_declines(self):
        """rounding_mode, a 0-d device tensor divisor (reading it would sync), and a broadcast
        the kernel can't tile."""
        entry = _entry("div")
        x = (torch.randn(256, 64, dtype=torch.bfloat16) / 8).to("neuron")
        assert not entry.can_handle(x, 2.0, rounding_mode="floor")
        assert not entry.can_handle(x, torch.tensor(2.0, dtype=torch.bfloat16).to("neuron"))
        assert not entry.can_handle(x, (torch.randn(1, 64, dtype=torch.bfloat16)).to("neuron"))
        with _assert_dynamic_skipped("div"):
            got = torch.div(x, 2.0, rounding_mode="floor").cpu()
        torch.testing.assert_close(got, torch.div(x.cpu(), 2.0, rounding_mode="floor"))

    def test_dynamic_nki_div_out_writes_in_place(self):
        """Unpadded, so the kernel writes into out= directly -- result aliases it."""
        x = (torch.randn(256, 720, dtype=torch.bfloat16) / 8).to("neuron")
        buf = torch.empty(256, 720, dtype=torch.bfloat16).to("neuron")
        with _assert_dynamic_ran("div"):
            res = torch.div(x, 3.5, out=buf)
        assert res.data_ptr() == buf.data_ptr(), "out= was not written in place"
        ref = (x.cpu().float() * (1.0 / 3.5)).to(torch.bfloat16)
        torch.testing.assert_close(buf.cpu(), ref, rtol=0, atol=0)

    def test_dynamic_nki_div_scalar_tensor_dividend_declines(self):
        """A 0-d scalar dividend has no last dim; can_handle must decline it (not raise)
        and fall back to the static lowering."""
        entry = _entry("div")
        x = torch.tensor(7.5, dtype=torch.bfloat16).to("neuron")  # 0-dim, dim()==0
        assert x.dim() == 0
        # can_handle must not raise and must decline (scalar -> M=1 < _TILE_M).
        assert not entry.can_handle(x, 2.0)
        with _assert_dynamic_skipped("div"):
            got = torch.div(x, 2.0).cpu()
        torch.testing.assert_close(got, torch.div(x.cpu(), 2.0))

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

    # These two assert bit-exactness (rtol=atol=0): a single bf16 subtract of the
    # same two bf16 inputs is one correctly-rounded op on both CPU and Neuron, so
    # there is no accumulation to accrue error. A tolerance here would only mask a
    # real dispatch/indexing bug (e.g. a mis-tiled or partially-written result).
    def test_dynamic_nki_sub_alpha_declines(self):
        # alpha != 1 can't be honored by the a-b kernel -> can_handle rejects -> fallback.
        x, y = self._bf16_neuron(self.SHAPE, self.SHAPE)
        with _assert_dynamic_skipped("sub"):
            out = torch.sub(x, y, alpha=2).cpu()
        torch.testing.assert_close(out, x.cpu() - 2 * y.cpu(), rtol=0, atol=0)

    def test_dynamic_nki_sub_out_writes_in_place(self):
        # out= is the destination buffer; dynamic path writes into it.
        x, y = self._bf16_neuron(self.SHAPE, self.SHAPE)
        buf = torch.empty_like(x)
        with _assert_dynamic_ran("sub"):
            torch.sub(x, y, out=buf)
        torch.testing.assert_close(buf.cpu(), x.cpu() - y.cpu(), rtol=0, atol=0)

    # ── sum-specific: N-D last-dim reduce flattens leading dims; non-last / dtype
    # / all-reduce decline to the default lowering ──
    @pytest.mark.parametrize("shape", [(256, 2048), (2, 256, 2048), (1, 128, 2048)])
    @pytest.mark.parametrize("keepdim", [True, False])
    def test_dynamic_nki_sum_last_dim_nd(self, shape, keepdim):
        # last-dim reduce of a 2-D+ input: leading dims flatten to M, kernel runs,
        # output shape matches torch (fp32-accumulated ref to match the kernel).
        x = torch.randn(*shape, dtype=torch.bfloat16) / 8
        ref = x.float().sum(dim=-1, keepdim=keepdim).to(torch.bfloat16)
        with _assert_dynamic_ran("sum"):
            got = x.to("neuron").sum(dim=-1, keepdim=keepdim).cpu()
        assert got.shape == ref.shape, f"{got.shape} != {ref.shape}"
        torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2)

    def test_dynamic_nki_sum_non_last_dim_declines(self):
        # reducing a non-last dim isn't the [M, H] free-dim reduce -> fallback.
        x = (torch.randn(256, 2048, dtype=torch.bfloat16) / 8).to("neuron")
        with _assert_dynamic_skipped("sum"):
            got = x.sum(dim=0).cpu()
        torch.testing.assert_close(got, x.cpu().float().sum(dim=0).bfloat16(), rtol=1e-2, atol=1e-2)

    def test_dynamic_nki_sum_all_reduce_declines(self):
        # dim=None (reduce every dim) isn't a single free-dim reduce -> fallback.
        x = (torch.randn(256, 2048, dtype=torch.bfloat16) / 8).to("neuron")
        with _assert_dynamic_skipped("sum"):
            x.sum()

    # ── cat-specific, two shapes. FREE DIM: 2-input last-dim concat (backward of
    # .chunk(2, -1)) on one NEFF; N-D flattens leading dims; unequal widths OK; >2
    # tensors and dtype-or-lead mismatch decline. ROW DIM: K-input dim-0 concat
    # (TorchTitan's MoE for-loop) as K composite copies, so no row count is baked;
    # a strided input or any middle dim declines. ──
    @pytest.mark.parametrize(
        "shape_a,shape_b",
        [
            ((256, 2048), (256, 2048)),  # equal widths (the .chunk(2) case)
            ((256, 2048), (256, 1024)),  # unequal widths
            ((2, 256, 1024), (2, 256, 1024)),  # 3-D: leading dims flatten
        ],
    )
    def test_dynamic_nki_cat_last_dim(self, shape_a, shape_b):
        a = torch.randn(*shape_a, dtype=torch.bfloat16) / 8
        b = torch.randn(*shape_b, dtype=torch.bfloat16) / 8
        ref = torch.cat([a, b], dim=-1)
        with _assert_dynamic_ran("cat"):
            got = torch.cat([a.to("neuron"), b.to("neuron")], dim=-1).cpu()
        assert got.shape == ref.shape, f"{got.shape} != {ref.shape}"
        # concat is pure data movement — bit-exact, no accumulation.
        torch.testing.assert_close(got, ref, rtol=0, atol=0)

    @pytest.mark.parametrize("rows", [256, 129, 383])
    def test_dynamic_nki_cat_non_128_count(self, rows):
        # M padded to a 128-multiple inside prepare_io, sliced back after.
        a = torch.randn(rows, 1024, dtype=torch.bfloat16) / 8
        b = torch.randn(rows, 1024, dtype=torch.bfloat16) / 8
        ref = torch.cat([a, b], dim=-1)
        with _assert_dynamic_ran("cat"):
            got = torch.cat([a.to("neuron"), b.to("neuron")], dim=-1).cpu()
        torch.testing.assert_close(got, ref, rtol=0, atol=0)

    def test_dynamic_nki_cat_three_tensors_declines(self):
        # kernel concatenates exactly two operands -> 3-tensor cat falls back.
        ts = [torch.randn(256, 1024, dtype=torch.bfloat16).to("neuron") for _ in range(3)]
        with _assert_dynamic_skipped("cat"):
            got = torch.cat(ts, dim=-1).cpu()
        torch.testing.assert_close(got, torch.cat([t.cpu() for t in ts], dim=-1), rtol=0, atol=0)

    @pytest.mark.parametrize(
        "counts,width",
        [
            ([400, 500, 383], 352),  # DSv3 per-expert rows, w1/w3 width (1408/TP4)
            ([128, 129, 384, 1024], 2048),  # DSv3 w2 width
            ([37, 400, 60, 511], 352),  # sub-tile blocks mixed with full ones
            ([8, 17, 120], 352),  # every block below one tile
            ([0, 400, 0, 250], 2048),  # an empty expert contributes no rows
            ([1, 400, 129], 352),  # a 1-row block: one token routed to an expert
        ],
    )
    def test_dynamic_nki_cat_row_dim(self, counts, width):
        """dim-0 concat of K same-width blocks — TorchTitan's MoE for-loop
        ``cat([chunk_e @ w_e ...], dim=0)``. Bit-exact: a concat does no arithmetic."""
        parts = [_exact_src(n, width, torch.bfloat16) for n in counts]
        ref = torch.cat(parts, dim=0)
        with _assert_dynamic_ran("cat"):
            got = torch.cat([p.to("neuron") for p in parts], dim=0).cpu()
        assert got.shape == ref.shape, f"{got.shape} != {ref.shape}"
        torch.testing.assert_close(got, ref, rtol=0, atol=0)

    def test_dynamic_nki_cat_row_dim_3d(self):
        """Dims after the first flatten into the row width."""
        parts = [_exact_src(n, 8 * 256, torch.bfloat16).reshape(n, 8, 256) for n in (300, 411)]
        ref = torch.cat(parts, dim=0)
        with _assert_dynamic_ran("cat"):
            got = torch.cat([p.to("neuron") for p in parts], dim=0).cpu()
        assert got.shape == ref.shape, f"{got.shape} != {ref.shape}"
        torch.testing.assert_close(got, ref, rtol=0, atol=0)

    def test_dynamic_nki_cat_row_dim_no_recompile(self):
        """The point of the composite path: 64 blocks with fresh row counts compile
        nothing. The static lowering keys on ALL K counts -> 2 NEFFs per new routing."""
        import random

        import torch_neuronx._C as _C

        rng = random.Random(0)

        def step():
            for width in (352, 2048):
                counts = [rng.randint(1, 480) for _ in range(64)]
                parts = [
                    (torch.randn(n, width, dtype=torch.bfloat16) / 8).to("neuron") for n in counts
                ]
                assert torch.cat(parts, dim=0).shape == (sum(counts), width)

        step()  # warm: one copy NEFF per width
        for width in (352, 2048):  # plus the 1-row block's fixed [1, H] fallback
            blocks = [
                (torch.randn(n, width, dtype=torch.bfloat16) / 8).to("neuron") for n in (1, 200)
            ]
            torch.cat(blocks, dim=0)
        before = {e["cache_key"] for e in _C._get_compilation_cache_entries()}
        for _ in range(3):
            step()
        new = {e["cache_key"] for e in _C._get_compilation_cache_entries()} - before
        assert not new, f"recompiled for new per-expert counts: {sorted(new)}"

    def test_dynamic_nki_cat_row_dim_backward(self):
        """The composite copies run below autograd, so cat's backward still works."""
        counts = (400, 90, 511)
        parts = [(torch.randn(n, 352, dtype=torch.bfloat16) / 8).float() for n in counts]
        w = torch.randn(sum(counts), 352)
        cpu_leaves = [p.clone().requires_grad_(True) for p in parts]
        (torch.cat(cpu_leaves, dim=0) * w).sum().backward()

        nrn_leaves = [p.to(torch.bfloat16).to("neuron").requires_grad_(True) for p in parts]
        with _assert_dynamic_ran("cat"):
            out = torch.cat(nrn_leaves, dim=0)
        (out.float() * w.to("neuron")).sum().backward()
        for got, ref in zip(nrn_leaves, cpu_leaves, strict=True):
            torch.testing.assert_close(got.grad.float().cpu(), ref.grad, rtol=2e-2, atol=2e-2)

    def test_dynamic_nki_cat_row_dim_non_contiguous_declines(self):
        """A strided block would bake its stride into the copy NEFF -> fall back."""
        wide = (torch.randn(400, 704, dtype=torch.bfloat16) / 8).to("neuron")
        parts = [wide[:, :352], wide[:, 352:]]
        assert not parts[0].is_contiguous()
        with _assert_dynamic_skipped("cat"):
            got = torch.cat(parts, dim=0).cpu()
        torch.testing.assert_close(got, torch.cat([p.cpu() for p in parts], dim=0), rtol=0, atol=0)

    def test_dynamic_nki_cat_middle_dim_declines(self):
        """Neither the row dim nor the free dim -> fallback."""
        a = (torch.randn(2, 256, 1024, dtype=torch.bfloat16) / 8).to("neuron")
        b = (torch.randn(2, 128, 1024, dtype=torch.bfloat16) / 8).to("neuron")
        with _assert_dynamic_skipped("cat"):
            torch.cat([a, b], dim=1)

    # ── where / bitwise_or / ones_like: the decline guards (the generic sweeps only
    # build routable cases, and the arbitrary-H sweep only uses even H) plus the out=
    # destinations. All three are bit-exact, so every comparison here is rtol=atol=0. ──
    def test_dynamic_nki_where_odd_h_declines(self):
        # The condition is DMA'd as an int16 view, which needs an even H.
        cond = (torch.rand(256, 1025) > 0.5).to("neuron")
        a = (torch.randn(256, 1025, dtype=torch.bfloat16) / 8).to("neuron")
        b = (torch.randn(256, 1025, dtype=torch.bfloat16) / 8).to("neuron")
        with _assert_dynamic_skipped("where"):
            got = torch.where(cond, a, b).cpu()
        torch.testing.assert_close(got, torch.where(cond.cpu(), a.cpu(), b.cpu()), rtol=0, atol=0)

    def test_dynamic_nki_bitwise_or_odd_h_declines(self):
        # Same int16-view constraint, on both operands and the output.
        a = (torch.rand(256, 1025) > 0.5).to("neuron")
        b = (torch.rand(256, 1025) > 0.5).to("neuron")
        with _assert_dynamic_skipped("bitwise_or"):
            got = torch.bitwise_or(a, b).cpu()
        torch.testing.assert_close(got, a.cpu() | b.cpu(), rtol=0, atol=0)

    def test_dynamic_nki_where_scalar_other_declines(self):
        # where.ScalarOther lands here too; a scalar operand can't ride the kernel.
        cond = (torch.rand(256, 2048) > 0.5).to("neuron")
        a = (torch.randn(256, 2048, dtype=torch.bfloat16) / 8).to("neuron")
        with _assert_dynamic_skipped("where"):
            got = torch.where(cond, a, 0.0).cpu()
        torch.testing.assert_close(got, torch.where(cond.cpu(), a.cpu(), 0.0), rtol=0, atol=0)

    def test_dynamic_nki_where_broadcast_declines(self):
        # All three shapes must match; a broadcast condition must fall back.
        cond = (torch.rand(256, 1) > 0.5).to("neuron")
        a = (torch.randn(256, 2048, dtype=torch.bfloat16) / 8).to("neuron")
        b = (torch.randn(256, 2048, dtype=torch.bfloat16) / 8).to("neuron")
        with _assert_dynamic_skipped("where"):
            got = torch.where(cond, a, b).cpu()
        torch.testing.assert_close(got, torch.where(cond.cpu(), a.cpu(), b.cpu()), rtol=0, atol=0)

    def test_dynamic_nki_bitwise_or_int_dtype_declines(self):
        # Only bool is enabled (_BITWISE_DTYPES); integer operands fall back.
        a = torch.randint(0, 255, (256, 2048), dtype=torch.int32).to("neuron")
        b = torch.randint(0, 255, (256, 2048), dtype=torch.int32).to("neuron")
        with _assert_dynamic_skipped("bitwise_or"):
            got = torch.bitwise_or(a, b).cpu()
        torch.testing.assert_close(got, a.cpu() | b.cpu(), rtol=0, atol=0)

    def test_dynamic_nki_ones_like_dtype_override_declines(self):
        # The NEFF bakes x's dtype, so a dtype override must fall back.
        x = (torch.randn(256, 2048, dtype=torch.bfloat16) / 8).to("neuron")
        with _assert_dynamic_skipped("ones_like"):
            got = torch.ones_like(x, dtype=torch.float32).cpu()
        assert got.dtype == torch.float32
        torch.testing.assert_close(got, torch.ones(256, 2048), rtol=0, atol=0)

    def test_dynamic_nki_ones_like_same_dtype_routes(self):
        # ... but the guard must not over-decline an explicit dtype equal to x's.
        x = (torch.randn(256, 2048, dtype=torch.bfloat16) / 8).to("neuron")
        with _assert_dynamic_ran("ones_like"):
            got = torch.ones_like(x, dtype=torch.bfloat16).cpu()
        torch.testing.assert_close(got, torch.ones(256, 2048, dtype=torch.bfloat16), rtol=0, atol=0)

    @pytest.mark.parametrize("rows", [256, 200])
    def test_dynamic_nki_ones_like_strided_input_untouched(self, rows):
        # reads_input=False: prepare_io hands the output buffer to arg0 instead of
        # padding/contiguifying the caller's tensor, so a strided input at an aligned
        # and an unaligned M must still route and leave the source untouched.
        base = (torch.randn(rows, 4096, dtype=torch.bfloat16) / 8).to("neuron")
        x = base.chunk(2, dim=-1)[1]  # strided view, H=2048
        before = x.cpu().clone()
        with _assert_dynamic_ran("ones_like"):
            got = torch.ones_like(x).cpu()
        torch.testing.assert_close(
            got, torch.ones(rows, 2048, dtype=torch.bfloat16), rtol=0, atol=0
        )
        torch.testing.assert_close(x.cpu(), before, rtol=0, atol=0)

    def test_dynamic_nki_where_out_writes_in_place(self):
        # out= is the destination buffer; dynamic path writes into it.
        cond = (torch.rand(256, 2048) > 0.5).to("neuron")
        a = (torch.randn(256, 2048, dtype=torch.bfloat16) / 8).to("neuron")
        b = (torch.randn(256, 2048, dtype=torch.bfloat16) / 8).to("neuron")
        buf = torch.empty_like(a)
        with _assert_dynamic_ran("where"):
            torch.where(cond, a, b, out=buf)
        torch.testing.assert_close(
            buf.cpu(), torch.where(cond.cpu(), a.cpu(), b.cpu()), rtol=0, atol=0
        )

    def test_dynamic_nki_bitwise_or_out_writes_in_place(self):
        # The bool out= buffer is handed to the NEFF as an int16 view — the one dest
        # reuse in the registry that crosses a dtype view.
        a = (torch.rand(256, 2048) > 0.5).to("neuron")
        b = (torch.rand(256, 2048) > 0.5).to("neuron")
        buf = torch.empty_like(a)
        with _assert_dynamic_ran("bitwise_or"):
            torch.bitwise_or(a, b, out=buf)
        torch.testing.assert_close(buf.cpu(), a.cpu() | b.cpu(), rtol=0, atol=0)

    # ── _to_copy-specific: every dtype pair, the plan's warm-restart shapes, the
    # dim1-dynamic ([1, seq]) collapse, and the declines (same dtype / unsupported
    # dtype / non-contiguous source) ──
    @pytest.mark.parametrize("src_dtype", _DTYPES, ids=lambda d: _DTYPE_ID[d])
    @pytest.mark.parametrize("dst_dtype", _DTYPES, ids=lambda d: _DTYPE_ID[d])
    def test_dynamic_nki_to_copy_dtype_pairs(self, src_dtype, dst_dtype):
        """All 9 (src, dst) pairs: the 6 casts run dynamically and match CPU; the 3
        same-dtype ones are short-circuited before the dynamic path."""
        x = torch.randn(512, 256, dtype=src_dtype) / 8
        x_n = x.to("neuron")
        ref = x.to(dst_dtype)
        expect_dynamic = src_dtype != dst_dtype
        ctx = _assert_dynamic_ran if expect_dynamic else _assert_dynamic_skipped
        with ctx("_to_copy"):
            got = x_n.to(dst_dtype).cpu()
        assert got.dtype == dst_dtype
        rtol, atol = _cast_tol(src_dtype, dst_dtype)
        torch.testing.assert_close(got, ref, rtol=rtol, atol=atol)

    def test_dynamic_nki_to_copy_plan_shapes_no_recompile(self):
        """Plan's _to_copy row: [512, 256] (warm) and [4096, 256] (cold) bf16 -> fp32
        on ONE NEFF."""
        entry = _entry("_to_copy")
        compiled_before = len(entry._compiled)
        exec_before = entry._exec_count
        entry._exec_ms.clear()

        for rows in (512, 4096):
            x = (torch.randn(rows, 256, dtype=torch.bfloat16) / 8).to("neuron")
            got = x.to(torch.float32).cpu()
            torch.testing.assert_close(got, x.cpu().to(torch.float32), rtol=1e-4, atol=1e-4)

        assert entry._exec_count - exec_before == 2, "_to_copy did not run at both shapes"
        assert entry._exec_ms.issuperset({512, 4096}), f"saw Ms {entry._exec_ms}"
        assert (
            len(entry._compiled) - compiled_before <= 1
        ), "_to_copy recompiled between [512,256] and [4096,256]"

    def test_dynamic_nki_to_copy_1d_no_recompile(self):
        """1-D [seq] flattens to [numel, 1] — H = 1 baked, length dynamic -> one NEFF
        across a seq change."""
        entry = _entry("_to_copy")
        compiled_before = len(entry._compiled)
        exec_before = entry._exec_count

        for seq in (512, 4096):
            x = (torch.randn(seq, dtype=torch.bfloat16) / 8).to("neuron")
            with _assert_dynamic_ran("_to_copy"):
                got = x.to(torch.float32).cpu()
            assert got.shape == (seq,)
            torch.testing.assert_close(got, x.cpu().to(torch.float32), rtol=1e-4, atol=1e-4)

        assert entry._exec_count - exec_before == 2
        assert (
            len(entry._compiled) - compiled_before <= 1
        ), "_to_copy recompiled across the 1-D length change"

    def test_dynamic_nki_to_copy_unsupported_dtype_declines(self):
        # fp64 is not a kernel dtype -> fallback (NRT/CPU-bounce path), still correct.
        x = torch.randn(512, 256, dtype=torch.float64, device="neuron")
        with _assert_dynamic_skipped("_to_copy"):
            got = x.to(torch.float32).cpu()
        torch.testing.assert_close(got, x.cpu().to(torch.float32))

    def test_dynamic_nki_to_copy_non_contiguous_declines(self):
        # A transposed source must keep its dense strides -> decline to the
        # framework's strided path.
        x = (torch.randn(256, 512, dtype=torch.float32) / 8).to("neuron").t()
        assert not x.is_contiguous()
        with _assert_dynamic_skipped("_to_copy"):
            got = x.to(torch.bfloat16).cpu()
        torch.testing.assert_close(got, x.cpu().to(torch.bfloat16), rtol=1e-2, atol=1e-2)

    # ── copy_-specific: in-place (the destination buffer IS `self`) — casts, the
    # plan's [512, 1] shapes, broadcast, row slices, ragged counts, declines ──
    @pytest.mark.parametrize("src_dtype", _DTYPES, ids=lambda d: _DTYPE_ID[d])
    @pytest.mark.parametrize("dst_dtype", _DTYPES, ids=lambda d: _DTYPE_ID[d])
    def test_dynamic_nki_copy_dtype_pairs(self, src_dtype, dst_dtype):
        """dst.copy_(src), all 9 dtype pairs: casts src into dst's dtype in place
        (same-dtype included — copy_ is a real data move)."""
        src = torch.randn(512, 256, dtype=src_dtype) / 8
        dst_n = torch.zeros(512, 256, dtype=dst_dtype).to("neuron")
        with _assert_dynamic_ran("copy_"):
            out = dst_n.copy_(src.to("neuron"))
        assert out.data_ptr() == dst_n.data_ptr(), "copy_ must return the mutated self"
        rtol, atol = _cast_tol(src_dtype, dst_dtype)
        torch.testing.assert_close(dst_n.cpu(), src.to(dst_dtype), rtol=rtol, atol=atol)

    def test_dynamic_nki_copy_plan_shapes_no_recompile(self):
        """Plan's copy_ row: [512, 1] (warm) and [4096, 1] (cold) on ONE NEFF."""
        entry = _entry("copy_")
        compiled_before = len(entry._compiled)
        exec_before = entry._exec_count
        entry._exec_ms.clear()

        for rows in (512, 4096):
            src = torch.randn(rows, 1, dtype=torch.bfloat16) / 8
            dst = torch.zeros(rows, 1, dtype=torch.bfloat16).to("neuron")
            dst.copy_(src.to("neuron"))
            torch.testing.assert_close(dst.cpu(), src, rtol=0, atol=0)

        assert entry._exec_count - exec_before == 2, "copy_ did not run at both shapes"
        assert entry._exec_ms.issuperset({512, 4096}), f"saw Ms {entry._exec_ms}"
        assert (
            len(entry._compiled) - compiled_before <= 1
        ), "copy_ recompiled between [512,1] and [4096,1]"

    def test_dynamic_nki_strided_large_width_compiles(self):
        # Wide contiguous runs within the unroll budget compile (was an unroll failure).
        entry = _entry("copy_")
        f32 = torch.float32
        for width in (1024, 5_000_000, 8_388_608):
            entry.compile_fn(("strided", width, 1, width, 1, f32, f32))

    def test_dynamic_nki_strided_over_budget_declines(self):
        # ceil(width / knee) beyond _MAX_H_SEGMENTS -> can_handle declines (falls back).
        from torch_neuronx.python_ops.torch_mlir.ops.dynamic_kernels.strided_copy_kernel import (
            width_within_unroll_budget,
        )

        assert width_within_unroll_budget(8_388_608, 4)  # 8192 segs
        assert not width_within_unroll_budget(8_388_609, 4)  # 8193 segs

    def test_dynamic_nki_copy_broadcast_src(self):
        """A broadcast source is expanded to self's shape before this path, so the
        dynamic kernel handles it."""
        src = torch.randn(1, 2048, dtype=torch.bfloat16) / 8
        dst = torch.zeros(512, 2048, dtype=torch.bfloat16).to("neuron")
        with _assert_dynamic_ran("copy_"):
            dst.copy_(src.to("neuron"))
        torch.testing.assert_close(dst.cpu(), src.expand(512, 2048), rtol=0, atol=0)

    @pytest.mark.parametrize("rows", [129, 200, 255, 383, 1001])
    def test_dynamic_nki_copy_non_128_runs(self, rows):
        """A ragged row count runs in place: the tail tile is backward-aligned at row
        M-128, so `self` is written exactly on [0, M)."""
        src = torch.randn(rows, 2048, dtype=torch.bfloat16) / 8
        dst = torch.zeros(rows, 2048, dtype=torch.bfloat16).to("neuron")
        with _assert_dynamic_ran("copy_"):
            dst.copy_(src.to("neuron"))
        torch.testing.assert_close(dst.cpu(), src, rtol=0, atol=0)

    @pytest.mark.parametrize("op", ["copy_", "_to_copy"])
    def test_dynamic_nki_cast_ragged_no_recompile(self, op):
        """The point of dropping the padding: ragged counts stay on ONE NEFF — M is not
        in the compile_key and there is no pad buffer whose shape could recompile."""
        entry = _entry(op)
        compiled_before = len(entry._compiled)
        exec_before = entry._exec_count
        entry._exec_ms.clear()

        rows = (129, 200, 255, 383, 1001)
        for m in rows:
            src = torch.randn(m, 2048, dtype=torch.bfloat16) / 8
            if op == "copy_":
                dst = torch.zeros(m, 2048, dtype=torch.bfloat16).to("neuron")
                dst.copy_(src.to("neuron"))
                torch.testing.assert_close(dst.cpu(), src, rtol=0, atol=0)
            else:
                got = src.to("neuron").to(torch.float32).cpu()
                torch.testing.assert_close(got, src.to(torch.float32), rtol=0, atol=0)

        assert entry._exec_count - exec_before == len(rows), f"{op} did not run at every count"
        assert entry._exec_ms.issuperset(
            set(rows)
        ), f"expected exec at {set(rows)}, saw {entry._exec_ms}"
        assert (
            len(entry._compiled) - compiled_before <= 1
        ), f"{op} recompiled across ragged counts {rows}"

    # ── sub-tile M: the same strided kernel, built with tile_m == 1 ──
    @pytest.mark.parametrize(
        "dtype",
        [torch.bfloat16, torch.float16, torch.float32, torch.int32],
        ids=["bf16", "fp16", "fp32", "int32"],
    )
    # rows=1 excluded: a [1, H] slice is contiguous, so it is the contiguous test's case.
    @pytest.mark.parametrize("rows", [2, 16, 64, 100, 119, 127])
    def test_dynamic_nki_copy_sub_tile_strided_src(self, rows, dtype):
        """A sub-tile M takes the strided kernel's tile_m == 1 variant, which walks single
        rows, instead of declining to the shape-baking prologue. Byte-exact: a copy does no
        arithmetic, so any tolerance would hide a wrong access-pattern offset."""
        wide = _exact_src(rows, 2048, dtype)
        src_n = wide.to("neuron")[:, :1024]
        assert not src_n.is_contiguous()
        dst_n = torch.zeros(rows, 1024, dtype=dtype).to("neuron")
        with _assert_dynamic_ran("copy_"):
            dst_n.copy_(src_n)
        torch.testing.assert_close(dst_n.cpu(), wide[:, :1024].contiguous(), rtol=0, atol=0)

    @pytest.mark.parametrize("rows", [1, 2, 64, 127])
    def test_dynamic_nki_copy_sub_tile_contiguous_src(self, rows):
        """A CONTIGUOUS sub-tile copy takes the row-walking variant too: the cast kernel's
        tail would start at row M - 128. Declining would compile per row count."""
        src = _exact_src(rows, 1024, torch.bfloat16)
        dst = torch.zeros(rows, 1024, dtype=torch.bfloat16).to("neuron")
        with _assert_dynamic_ran("copy_"):
            dst.copy_(src.to("neuron"))
        torch.testing.assert_close(dst.cpu(), src, rtol=0, atol=0)

    def test_dynamic_nki_copy_into_row_slice_no_recompile(self):
        """The dst is one row block of a bigger buffer (how the dim-0 concat fills its
        output): neither its storage offset nor the row count may reach the NEFF key."""
        import torch_neuronx._C as _C

        def fill(counts):
            buf = torch.empty(sum(counts), 352, dtype=torch.bfloat16).to("neuron")
            off = 0
            for n in counts:
                src = _exact_src(n, 352, torch.bfloat16)
                buf[off : off + n].copy_(src.to("neuron"))
                torch.testing.assert_close(buf[off : off + n].cpu(), src, rtol=0, atol=0)
                off += n

        fill([200, 64])  # warm: full-tile copy NEFF + row-walk NEFF
        before = {e["cache_key"] for e in _C._get_compilation_cache_entries()}
        for counts in ([383, 129], [1001, 7], [17, 500]):
            fill(counts)
        new = {e["cache_key"] for e in _C._get_compilation_cache_entries()} - before
        assert not new, f"recompiled for a new block size / offset: {sorted(new)}"

    def test_dynamic_nki_copy_single_row_shares_the_sub_tile_neff(self):
        """A 1-row block is what a 1-token MoE expert hands the dim-0 concat. It must reuse
        the row-walking NEFF; a fixed [1, H] one was DSv3's residual per-run copy_ pair."""
        import torch_neuronx._C as _C

        for width in (352, 2048):
            buf = torch.empty(200, width, dtype=torch.bfloat16).to("neuron")
            warm = _exact_src(64, width, torch.bfloat16)
            buf[:64].copy_(warm.to("neuron"))
            before = {e["cache_key"] for e in _C._get_compilation_cache_entries()}
            one = _exact_src(1, width, torch.bfloat16)
            with _assert_dynamic_ran("copy_"):
                buf[70:71].copy_(one.to("neuron"))
            torch.testing.assert_close(buf[70:71].cpu(), one, rtol=0, atol=0)
            new = {e["cache_key"] for e in _C._get_compilation_cache_entries()} - before
            assert not new, f"the 1-row copy compiled its own NEFF: {sorted(new)}"

    def test_dynamic_nki_copy_sub_tile_strided_src_no_recompile(self):
        """One NEFF per (width, dyn_stride, dtype) covers every sub-tile row count."""
        import torch_neuronx._C as _C

        def move(rows):
            wide = _exact_src(rows, 2048, torch.bfloat16)
            dst = torch.zeros(rows, 1024, dtype=torch.bfloat16).to("neuron")
            dst.copy_(wide.to("neuron")[:, :1024])
            torch.neuron.synchronize()
            torch.testing.assert_close(dst.cpu(), wide[:, :1024].contiguous(), rtol=0, atol=0)

        move(64)  # warm
        before = {e["cache_key"] for e in _C._get_compilation_cache_entries()}
        for rows in [2, 16, 100, 127]:
            move(rows)
        new = {e["cache_key"] for e in _C._get_compilation_cache_entries()} - before
        assert not new, f"recompiled for a new sub-tile M: {sorted(new)}"

    @pytest.mark.parametrize("rows", [2, 64, 127])
    def test_dynamic_nki_copy_sub_tile_broadcast_src(self, rows):
        """Stride-0 fan-out works at tile_m == 1 as well: the kernel replicates in SBUF, so
        the DMA never reads stride 0."""
        col = torch.randn(rows, 1, dtype=torch.bfloat16) / 8
        src_n = col.to("neuron").expand(rows, 1024)
        dst_n = torch.zeros(rows, 1024, dtype=torch.bfloat16).to("neuron")
        with _assert_dynamic_ran("copy_"):
            dst_n.copy_(src_n)
        torch.testing.assert_close(dst_n.cpu(), col.expand(rows, 1024), rtol=0, atol=0)

    @pytest.mark.parametrize("rows", [129, 200, 383])
    def test_dynamic_nki_copy_ragged_row_slice_no_clobber(self, rows):
        """Tightest guard on the tail: a ragged in-place destination that is a row slice
        in the MIDDLE of a bigger tensor — padding would corrupt the rows after it."""
        big = torch.zeros(1024, 2048, dtype=torch.bfloat16).to("neuron")
        dst = big[128 : 128 + rows]
        assert dst.is_contiguous() and dst.storage_offset() != 0
        src = torch.randn(rows, 2048, dtype=torch.bfloat16) / 8
        with _assert_dynamic_ran("copy_"):
            dst.copy_(src.to("neuron"))
        out = big.cpu()
        torch.testing.assert_close(out[128 : 128 + rows], src, rtol=0, atol=0)
        assert out[:128].eq(0).all(), "wrote before the slice"
        assert out[128 + rows :].eq(0).all(), "wrote past the ragged slice end"

    def test_dynamic_nki_copy_into_row_slice(self):
        """A row-slice destination is contiguous but starts at a non-zero storage
        offset — the NEFF must write at that offset and leave the rest untouched."""
        big = torch.zeros(512, 2048, dtype=torch.bfloat16).to("neuron")
        dst = big[128:384]  # contiguous view, storage_offset = 128 * 2048
        assert dst.is_contiguous() and dst.storage_offset() != 0
        src = torch.randn(256, 2048, dtype=torch.bfloat16) / 8
        with _assert_dynamic_ran("copy_"):
            dst.copy_(src.to("neuron"))
        out = big.cpu()
        torch.testing.assert_close(out[128:384], src, rtol=0, atol=0)
        # rows outside the slice must be untouched (a base-pointer write would clobber them)
        assert out[:128].eq(0).all() and out[384:].eq(0).all(), "wrote outside the slice"

    def test_dynamic_nki_copy_non_contiguous_dst_declines(self):
        # A strided destination can't be written by the contiguous-tile kernel.
        base = torch.zeros(2048, 512, dtype=torch.bfloat16).to("neuron")
        dst = base.t()  # [512, 2048], non-contiguous
        src = torch.randn(512, 2048, dtype=torch.bfloat16) / 8
        with _assert_dynamic_skipped("copy_"):
            dst.copy_(src.to("neuron"))
        torch.testing.assert_close(dst.cpu(), src, rtol=0, atol=0)

    # ── row-gather via aten::gather (scatter_add_'s backward) ──

    def test_dynamic_nki_gather_no_recompile_across_n(self):
        """One NEFF per (T, H, dtype) while the gathered row count n varies — the
        MoE-backward guarantee, where n is a per-expert token count."""
        T, H = 512, 2048  # noqa: N806
        entry = _entry("gather")
        compiled_before = len(entry._compiled)
        exec_before = entry._exec_count
        entry._exec_ms.clear()

        src = torch.randn(T, H, dtype=torch.bfloat16) / 8
        src_n = src.to("neuron")
        ns = (128, 200, 383, 512)
        for n in ns:
            idx = torch.randperm(T)[:n]
            idx_n = idx.to("neuron").unsqueeze(1).expand(n, H)
            torch.testing.assert_close(src_n.gather(0, idx_n).cpu(), src[idx], rtol=0, atol=0)

        assert entry._exec_count - exec_before == len(ns), "gather did not run for every n"
        assert entry._exec_ms.issuperset(
            set(ns)
        ), f"expected exec at n={set(ns)}, saw {entry._exec_ms}"
        assert (
            len(entry._compiled) - compiled_before <= 1
        ), "recompiled across n (did n leak into the compile_key?)"

    def test_dynamic_nki_index_put_no_recompile_across_n(self):
        """One NEFF per (T, H, dtype) while the scattered row count n varies — the
        MoE-combine guarantee for aten::_index_put_impl_ (accumulate=True). n is the
        per-expert token count; it is padded to the baked T in prepare_io, so it must
        not enter the compile_key. Shares the scatter_add kernel."""
        T, H = 512, 2048  # noqa: N806
        entry = _entry("_index_put_impl_")
        compiled_before = len(entry._compiled)
        exec_before = entry._exec_count
        entry._exec_ms.clear()

        ns = (128, 200, 383, 512)  # non-128 and 128-multiple counts, all n <= T
        for n in ns:
            idx = torch.randperm(T)[:n]  # unique target rows
            src = torch.randn(n, H, dtype=torch.bfloat16) / 8
            ref = torch.zeros(T, H, dtype=torch.bfloat16)
            ref.index_put_((idx.to(torch.int64),), src, accumulate=True)
            out = torch.zeros(T, H, dtype=torch.bfloat16).to("neuron")
            out.index_put_((idx.to("neuron"),), src.to("neuron"), accumulate=True)
            torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2, msg=f"n={n}")

        assert entry._exec_count - exec_before == len(ns), "index_put did not run for every n"
        assert entry._exec_ms.issuperset(
            set(ns)
        ), f"expected exec at n={set(ns)}, saw {entry._exec_ms}"
        assert (
            len(entry._compiled) - compiled_before <= 1
        ), "recompiled across n (did n leak into the compile_key?)"

    # ── MoE combine: many-to-one scatter_add (N = T*top_k) via gather-reduce ──

    @staticmethod
    def _combine_case(T, top_k, H, *, seed=0):  # noqa: N803
        """Expert-sorted combine inputs: routed[N,H], expanded index[N,H], and the
        torch.scatter_add reference into a zeros [T,H] buffer. N = T*top_k."""
        g = torch.Generator().manual_seed(seed)
        n = T * top_k
        # each token appears exactly top_k times, shuffled into expert-sorted order
        token_ids = torch.arange(T).repeat(top_k)[torch.randperm(n, generator=g)]
        src = torch.randn(n, H, generator=g, dtype=torch.bfloat16) / 8
        index = token_ids.reshape(n, 1).expand(n, H)
        ref = torch.zeros(T, H, dtype=torch.float32)
        ref.scatter_add_(0, index.to(torch.int64), src.float())
        return src, index, ref

    def test_dynamic_nki_combine_scatter_add_correct(self):
        """Many-to-one combine (N = T*top_k, top_k=6 rows per token) matches
        torch.scatter_add and runs the gather-reduce kernel (not a fallback)."""
        T, top_k, H = 512, 6, 2048  # noqa: N806
        src, index, ref = self._combine_case(T, top_k, H)
        out = torch.zeros(T, H, dtype=torch.bfloat16).to("neuron")
        with _assert_dynamic_ran("scatter_add_"):
            out.scatter_add_(0, index.to("neuron"), src.to("neuron"))
        torch.testing.assert_close(out.cpu().float(), ref, rtol=2e-2, atol=2e-2)

    def test_dynamic_nki_combine_no_recompile_across_routing(self):
        """One NEFF per (T, top_k, H, dtype): at fixed T the combine reuses it as
        routing (the index values) changes. T is baked (kernel output/APs are
        traced at T), so distinct T compile distinct NEFFs — but T is a static
        model dim, so a real run has one combine shape."""
        T, top_k, H = 512, 6, 2048  # noqa: N806
        entry = _entry("scatter_add_")
        compiled_before = len(entry._compiled)
        exec_before = entry._exec_count

        seeds = (0, 1, 2, 3)  # same T, different routing (index values) each time
        for seed in seeds:
            src, index, ref = self._combine_case(T, top_k, H, seed=seed)
            out = torch.zeros(T, H, dtype=torch.bfloat16).to("neuron")
            out.scatter_add_(0, index.to("neuron"), src.to("neuron"))
            torch.testing.assert_close(out.cpu().float(), ref, rtol=2e-2, atol=2e-2)

        assert entry._exec_count - exec_before == len(seeds), "combine did not run every case"
        assert (
            len(entry._compiled) - compiled_before <= 1
        ), "combine recompiled across routing (did the index values leak into the key?)"

    def test_dynamic_nki_gather_non_row_index_declines(self):
        """Only a free-dim-constant index on dim 0 is a row gather. A per-element
        index falls back to the default lowering and stays correct."""
        T, H = 512, 256  # noqa: N806
        src = torch.randn(T, H, dtype=torch.bfloat16) / 8
        elem = torch.randint(0, T, (256, H))  # a genuine element gather, not an expand
        with _assert_dynamic_skipped("gather"):
            got = src.to("neuron").gather(0, elem.to("neuron")).cpu()
        torch.testing.assert_close(got, src.gather(0, elem), rtol=0, atol=0)

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

    @pytest.mark.parametrize("n", [200, 129, 383])
    def test_dynamic_nki_mm_backward_non_128_count(self, n):
        """fwd+bwd with a non-128 token count: grad_input (mm, M-pad), grad_weight
        (mm_k, K-pad), and silu_backward (M-pad) all route dynamically and grads
        match CPU after the padded rows are sliced/zero-contributed."""
        H, H4 = 2048, 4096  # noqa: N806
        mm, mm_k, sb = _entry("mm"), _entry("mm_k"), _entry("silu_backward")

        def _fwd_bwd(device):
            g = torch.Generator().manual_seed(3)
            x = (torch.randn(n, H, dtype=torch.bfloat16, generator=g) / 64).to(device)
            w1 = (torch.randn(H, H4, dtype=torch.bfloat16, generator=g) / 64).to(device)
            w3 = (torch.randn(H4, H, dtype=torch.bfloat16, generator=g) / 64).to(device)
            x, w1, w3 = (t.detach().requires_grad_(True) for t in (x, w1, w3))
            out = torch.nn.functional.silu(x @ w1) @ w3
            (out * out).sum().backward()
            if device == "neuron":
                torch_neuronx.synchronize()
            return [t.grad.cpu() for t in (x, w1, w3)]

        mm_e, mmk_e, sb_e = mm._exec_count, mm_k._exec_count, sb._exec_count
        neuron_grads = _fwd_bwd("neuron")
        assert mm._exec_count > mm_e, "grad_input not routed to mm at non-128 count"
        assert mm_k._exec_count > mmk_e, "grad_weight not routed to mm_k at non-128 count"
        assert sb._exec_count > sb_e, "silu_backward not routed at non-128 count"
        for gi, (gn, gc) in enumerate(zip(neuron_grads, _fwd_bwd("cpu"), strict=True)):
            torch.testing.assert_close(gn, gc, rtol=3e-2, atol=3e-2, msg=f"grad[{gi}] mismatch")

    # generalized mm: any positive K, N a multiple of num_programs (ragged k-tile /
    # n-tile), sharded.
    @pytest.mark.parametrize(
        "K, N",
        [
            (2048, 4096),  # both %1024 (regression)
            (2048, 1408),  # N = 512+512+384 (non-%512 remainder)
            (1408, 2048),  # K = 11 tiles (partial final k-block)
            (256, 256),  # tiny K, N < 512
            (128, 128),  # smallest single tile
            (2048, 352),  # dsv3 16B TP4 w1/w3: N=352 not %128 (ragged n-tile)
            (352, 2048),  # dsv3 16B TP4 w2: K=352 not %128 (partial final k-tile, width 96)
            (352, 352),  # both K and N ragged (non-128)
            (129, 256),  # K=129: final k-tile width 1 (smallest remainder)
            (255, 256),  # K=255: final k-tile width 127 (largest remainder)
            (256, 130),  # N=130: ragged n-tile, even so LNC-shardable
            (16384, 4096),  # adaptive block_size_n -> 512
            (32768, 512),  # adaptive block_size_n -> 256
            (65536, 256),  # adaptive block_size_n -> 128
        ],
    )
    def test_dynamic_nki_mm_generalized_shapes(self, K, N):  # noqa: N803
        mm = _entry("mm")
        execs = mm._exec_count
        a = torch.randn(384, K, dtype=torch.bfloat16) / 8
        w = torch.randn(K, N, dtype=torch.bfloat16) / 8
        ref = a.float() @ w.float()
        with _assert_dynamic_ran("mm"):
            got = torch.mm(a.to("neuron"), w.to("neuron")).cpu().float()
        assert mm._exec_count > execs
        torch.testing.assert_close(got, ref, rtol=3e-2, atol=3e-2)

    def test_dynamic_nki_mm_over_capacity_declines(self):
        # Weight-resident rhs is (K/128)*block_width*2 bytes/partition; SBUF is 192 KiB.
        # The kernel picks an adaptive block_width (1024/512/256/128 by K), so it declines
        # only when even width-128 over-subscribes: (K/128)*min(128,shard_n)*2 > budget.
        from torch_neuronx.python_ops.torch_mlir.ops.dynamic_kernels.adapters.matmul import (
            _can_handle_mm,
        )

        # K=16384,N=4096: width-1024 = 256 KiB (over), but width-512 = 128 KiB fits ->
        # ACCEPTED (runs dynamically at the shrunk block width).
        big_a = torch.empty(128, 16384, dtype=torch.bfloat16, device="neuron")
        big_w = torch.empty(16384, 4096, dtype=torch.bfloat16, device="neuron")
        assert _can_handle_mm(big_a, big_w) is True
        # K=131072,N=256: even width-128 = 256 KiB/partition over budget -> DECLINES.
        # (N=256 keeps the rhs alloc ~67 MB — can_handle only reads shape metadata.)
        huge_a = torch.empty(128, 131072, dtype=torch.bfloat16, device="neuron")
        huge_w = torch.empty(131072, 256, dtype=torch.bfloat16, device="neuron")
        assert _can_handle_mm(huge_a, huge_w) is False
        ok_a = torch.empty(128, 2048, dtype=torch.bfloat16, device="neuron")
        ok_w = torch.empty(2048, 4096, dtype=torch.bfloat16, device="neuron")
        assert _can_handle_mm(ok_a, ok_w) is True
        # N not divisible by num_programs (e.g. odd N under LNC>=2) declines -> native.
        from torch_neuronx.utils import get_logical_neuron_cores

        lnc = int(get_logical_neuron_cores())
        if lnc >= 2:
            odd_a = torch.empty(128, 256, dtype=torch.bfloat16, device="neuron")
            odd_w = torch.empty(256, lnc * 4 + 1, dtype=torch.bfloat16, device="neuron")
            assert _can_handle_mm(odd_a, odd_w) is False

    # Generalized mm_k: M and N are baked, so the kernel peels a short final tile.
    # A decline is not free here. This op is the MoE backward dW, so a decline becomes a
    # `prologue` recompile. DSv3-16B at TP4 has M or N = 1408/4 = 352 on every expert.
    @pytest.mark.parametrize(
        "M, N",
        [
            (2048, 4096),  # both aligned (regression)
            (2048, 512),  # smallest aligned N
            (2048, 352),  # dsv3 16B TP4 w1/w3 dW: N below one n-tile
            (352, 2048),  # dsv3 16B TP4 w2 dW: M = 128 + 128 + 96
            (352, 352),  # both dims ragged
            (96, 130),  # no full tile in either dim
            (128, 4096),  # single m-tile, 8 full n-tiles
            (129, 513),  # 1-wide m-tail and n-tail (smallest remainders)
        ],
    )
    def test_dynamic_nki_mm_k_generalized_shapes(self, M, N):  # noqa: N803
        """grad_weight = x_e.T @ dout for a ragged M and N, at a K of 2 contraction tiles."""
        K = 200  # noqa: N806  (K_ceil = 256)
        g = torch.Generator().manual_seed(M + N)
        x = torch.randn(K, M, dtype=torch.bfloat16, generator=g) / 8
        d = torch.randn(K, N, dtype=torch.bfloat16, generator=g) / 8
        ref = x.float().t() @ d.float()
        mat1 = x.to("neuron").t()  # stride (1, M) -> _resolve_entry routes to mm_k
        with _assert_dynamic_ran("mm_k"):
            got = torch.mm(mat1, d.to("neuron")).cpu().float()
        torch.testing.assert_close(got, ref, rtol=3e-2, atol=3e-2)

    def test_dynamic_nki_mm_k_no_recompile_across_k(self):
        """One NEFF for each (M, N) at the DSv3 dW shapes. Every per-expert token count
        reuses it, also a count below one contraction tile."""
        mm_k = _entry("mm_k")
        for M, N in ((2048, 352), (352, 2048)):  # noqa: N806
            compiled = None
            for K in (46, 130, 200, 384, 500):  # noqa: N806
                x = (torch.randn(K, M, dtype=torch.bfloat16) / 8).to("neuron")
                d = (torch.randn(K, N, dtype=torch.bfloat16) / 8).to("neuron")
                with _assert_dynamic_ran("mm_k"):
                    torch.mm(x.t(), d)
                torch_neuronx.synchronize()
                if compiled is None:
                    compiled = len(mm_k._compiled)
                assert len(mm_k._compiled) == compiled, (
                    f"mm_k recompiled at M={M} N={N} K={K}: NEFFs "
                    f"{compiled} -> {len(mm_k._compiled)}"
                )

    def test_dynamic_nki_pad_rows_compiles_nothing_per_count(self):
        """`_pad_rows_to` must not key a NEFF on target_rows. A 128-multiple ceiling puts
        every padded row in the last tile, so only that fixed block needs zeroing."""
        import torch_neuronx._C as _C
        from torch_neuronx.python_ops.torch_mlir.ops.dynamic_kernels.adapter_utils import (
            _ceil_to_tile,
            _pad_rows_to,
        )

        def keys():
            return {e["cache_key"] for e in _C._get_compilation_cache_entries()}

        C = 2048  # noqa: N806
        warm = (torch.randn(46, C, dtype=torch.bfloat16) / 8).to("neuron")
        _pad_rows_to(warm, _ceil_to_tile(46)).sum().to("cpu")
        before = keys()
        for K in (200, 384, 500, 700):  # noqa: N806
            src = (torch.randn(K, C, dtype=torch.bfloat16) / 8).to("neuron")
            out = _pad_rows_to(src, _ceil_to_tile(K))
            torch_neuronx.synchronize()
            assert tuple(out.shape) == (_ceil_to_tile(K), C)
            torch.testing.assert_close(out[:K].cpu(), src.cpu(), rtol=0, atol=0)
            assert bool((out[K:].cpu() == 0).all()), f"pad rows not zero at K={K}"
        new = sorted(keys() - before)
        assert not new, f"pad compiled {len(new)} new NEFF(s): {new}"

    def test_dynamic_nki_mm_k_over_capacity_declines(self):
        """Decline an N that is wider than the [128, N] SBUF buffer. Also decline a zero
        extent."""
        from torch_neuronx.python_ops.torch_mlir.ops.dynamic_kernels.adapter_utils import (
            _MM_RESIDENT_BUDGET,
        )
        from torch_neuronx.python_ops.torch_mlir.ops.dynamic_kernels.adapters.matmul_k import (
            _can_handle_mm_k,
        )

        nb = {"dtype": torch.bfloat16, "device": "neuron"}
        n_max = _MM_RESIDENT_BUDGET // 2  # bf16: 2 bytes/element/partition
        mat1 = torch.empty(2048, 200, **nb)
        assert _can_handle_mm_k(mat1, torch.empty(200, 4096, **nb)) is True
        assert _can_handle_mm_k(mat1, torch.empty(200, n_max, **nb)) is True
        assert _can_handle_mm_k(mat1, torch.empty(200, n_max + 1, **nb)) is False
        # K == 0 has nothing to contract. N == 0 has no output column.
        assert _can_handle_mm_k(torch.empty(2048, 0, **nb), torch.empty(0, 352, **nb)) is False
        assert _can_handle_mm_k(mat1, torch.empty(200, 0, **nb)) is False

    # backward ops asserted on (mm/mm_k = grad_input/grad_weight).
    MOE_BWD_OPS: ClassVar = [
        "mm",
        "mm_k",
        "silu_backward",
        "index",
        "mul",
        "scatter_add_",
        "gather",
    ]

    @staticmethod
    def _moe_expert_loop_autograd(x, w1, w2, w3, expert_indices, T, H, num_experts):  # noqa: N803
        """MoE forward whose expert-combine is the real scatter_add_ over an expanded
        row index, so the backward goes through aten::gather. index_add's backward is
        index_select and never reaches the row-gather path the MoE actually hits."""
        output = torch.zeros(T, H, dtype=x.dtype, device=x.device)
        for e in range(num_experts):
            idx = expert_indices[e]
            if idx.numel() == 0:
                continue
            x_expert = x[idx]
            gated = torch.nn.functional.silu(x_expert @ w1[e]) * (x_expert @ w2[e])
            out = gated @ w3[e]
            # scatter_add_ index must be int64 on CPU
            output.scatter_add_(0, idx.to(torch.int64).unsqueeze(1).expand_as(out), out)
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

    # Ops still expected to recompile per token count (not yet covered). The test fails
    # if the real set diverges either way: a NEW op recompiles, or a listed op stops
    # (cover it, then remove it here).
    MOE_BWD_EXPECTED_RECOMPILES: ClassVar = set()

    def test_dynamic_nki_moe_backward_recompile_set(self):
        """Path-complete recompile check over the fwd+bwd MoE loop (cache-key diff): sees
        every op, not just the ported registry kernels, so it catches recompiling glue
        (cat/sum/index_put) the _compiled-based assertion is blind to."""
        T, E, H, H4 = self.MOE_T, self.MOE_E, self.MOE_H, self.MOE_H4  # noqa: N806

        def _mk(device, g):
            def mk(*s):
                return (
                    (torch.randn(*s, dtype=torch.bfloat16, generator=g) / 64)
                    .to(device)
                    .detach()
                    .requires_grad_(True)
                )

            return (
                [mk(H, H4) for _ in range(E)],
                [mk(H, H4) for _ in range(E)],
                [mk(H4, H) for _ in range(E)],
            )

        def _step(route_seed):
            g = torch.Generator().manual_seed(0)
            w1, w2, w3 = _mk("neuron", g)
            x = (
                (torch.randn(T, H, dtype=torch.bfloat16, generator=g) / 8)
                .to("neuron")
                .detach()
                .requires_grad_(True)
            )
            idx = [i.to("neuron") for i in self._moe_routing_128(route_seed)]
            out = self._moe_expert_loop_autograd(x, w1, w2, w3, idx, T, H, E)
            (out * out).sum().backward()
            torch_neuronx.synchronize()

        got = _recompiling_ops(_step, warm_seed=1, probe_seeds=(5, 9, 13))
        exp = self.MOE_BWD_EXPECTED_RECOMPILES
        assert got == exp, (
            f"recompiling-op set changed: got {sorted(got)}, expected {sorted(exp)}. "
            f"New op -> cover it or add here; missing op -> a gap closed, remove it."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Dynamic block-strided gather (column-slice copy without recompilation)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not _FLAG_ON, reason="TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS != 1")
class TestDynamicBlockStridedGather:
    """Tests for the dynamic block-strided gather kernel.

    The kernel copies contiguous blocks at regular offsets — the "row-gap" pattern
    from column-slice views (e.g., ``gate_up[:, :1024]`` where ``gate_up`` is
    ``[M, 2048]``). Compiled once per (block_size, row_stride, dtype); M is dynamic.
    """

    @pytest.mark.parametrize("num_rows", [128, 200, 211, 256, 300, 512])
    def test_correctness_varying_m(self, num_rows):
        """Correct results for varying M with H=1024, row_stride=2048."""
        wide = torch.randn(num_rows, 2048, dtype=torch.bfloat16, device="neuron")
        src = wide[:, :1024]
        dst = torch.empty(num_rows, 1024, dtype=torch.bfloat16, device="neuron")
        torch.ops.neuron.nki_strided_copy_(src, dst)
        torch.neuron.synchronize()
        torch.testing.assert_close(dst.cpu(), wide.cpu()[:, :1024].contiguous(), rtol=0, atol=0)

    def test_second_half_offset(self):
        """Handles non-zero storage offset (second half of buffer)."""
        for num_r in [211, 300]:
            wide = torch.randn(num_r, 2048, dtype=torch.bfloat16, device="neuron")
            src = wide[:, 1024:]
            dst = torch.empty(num_r, 1024, dtype=torch.bfloat16, device="neuron")
            torch.ops.neuron.nki_strided_copy_(src, dst)
            torch.neuron.synchronize()
            torch.testing.assert_close(dst.cpu(), wide.cpu()[:, 1024:].contiguous(), rtol=0, atol=0)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_dtypes(self, dtype):
        """Works for all supported float dtypes."""
        wide = torch.randn(256, 2048, dtype=dtype, device="neuron")
        dst = torch.empty(256, 1024, dtype=dtype, device="neuron")
        torch.ops.neuron.nki_strided_copy_(wide[:, :1024], dst)
        torch.neuron.synchronize()
        torch.testing.assert_close(dst.cpu(), wide.cpu()[:, :1024].contiguous(), rtol=0, atol=0)

    def test_compile_once_varying_m(self):
        """Compiles once and handles multiple M values without recompiling."""
        import torch_neuronx._C as _C

        # Warmup: compiles the kernel for (H=1024, row_stride=2048, bf16)
        wide0 = torch.randn(200, 2048, dtype=torch.bfloat16, device="neuron")
        dst0 = torch.empty(200, 1024, dtype=torch.bfloat16, device="neuron")
        torch.ops.neuron.nki_strided_copy_(wide0[:, :1024], dst0)
        torch.neuron.synchronize()

        before = {e["cache_key"] for e in _C._get_compilation_cache_entries()}

        # Multiple M values — all reuse the same NEFF
        for num_r in [211, 256, 275, 300, 400]:
            wide = torch.randn(num_r, 2048, dtype=torch.bfloat16, device="neuron")
            dst = torch.empty(num_r, 1024, dtype=torch.bfloat16, device="neuron")
            torch.ops.neuron.nki_strided_copy_(wide[:, :1024], dst)
            torch.neuron.synchronize()
            torch.testing.assert_close(dst.cpu(), wide.cpu()[:, :1024].contiguous(), rtol=0, atol=0)

        after = {e["cache_key"] for e in _C._get_compilation_cache_entries()}
        new_keys = after - before
        assert len(new_keys) == 0, (
            f"Dynamic kernel recompiled! {len(new_keys)} new entries: "
            f"{sorted(k[:80] for k in new_keys)}"
        )

    @pytest.mark.parametrize(
        "block_size,row_stride",
        [(512, 1024), (1024, 2048), (256, 1024), (1024, 4096)],
        ids=["H512_S1024", "H1024_S2048", "H256_S1024", "H1024_S4096"],
    )
    def test_different_geometries(self, block_size, row_stride):
        """Different (block_size, row_stride) pairs each compile their own NEFF."""
        num_r = 256
        wide = torch.randn(num_r, row_stride, dtype=torch.bfloat16, device="neuron")
        src = wide[:, :block_size]
        dst = torch.empty(num_r, block_size, dtype=torch.bfloat16, device="neuron")
        torch.ops.neuron.nki_strided_copy_(src, dst)
        torch.neuron.synchronize()
        torch.testing.assert_close(
            dst.cpu(), wide.cpu()[:, :block_size].contiguous(), rtol=0, atol=0
        )


# grouped_mm ragged (pad-free) path. Unaligned offs by itself proves the ragged impl
# ran: the aligned impls' validate_offs raises on unaligned rather than declining.
_GMM_RAGGED_CASES = [
    # (r, d1, d2, g) -- r >= 128 is a kernel constraint
    (384, 256, 128, 4),
    (1024, 512, 352, 8),
    (2048, 256, 128, 16),
    # A last group with many tiles alongside earlier single-tile groups at unaligned
    # bases. Caught a shared-weight-buffer hazard that the cases above do not reach.
    (2048, 512, 352, 8),
]

_GMM_RAGGED_KINDS = [
    "even",
    "first_takes_all",
    "last_takes_all",
    "one_row_each",
    "just_under_128",
    "aligned",  # the ragged kernel must also handle the easy case
    "random",
]


def _gmm_ragged_offs(total, g, kind, seed=0):
    """Group boundaries with no alignment. `kind` names the routing pattern."""
    if kind == "even":
        offs = [min(total, (i + 1) * (total // g)) for i in range(g)]
    elif kind == "first_takes_all":
        offs = [total] * g
    elif kind == "last_takes_all":
        offs = [0] * (g - 1) + [total]
    elif kind == "one_row_each":
        offs = [min(i + 1, total) for i in range(g)]
    elif kind == "just_under_128":
        offs = [min(total, (i + 1) * 128 - 1) for i in range(g)]
    elif kind == "aligned":
        offs = [min(total, (i + 1) * 128) for i in range(g)]
    else:  # random, unaligned
        gen = torch.Generator().manual_seed(seed)
        cuts = torch.randint(0, total + 1, (g - 1,), generator=gen)
        offs = torch.sort(cuts).values.tolist()
    offs = [*offs[: g - 1], total]
    return torch.tensor(offs, dtype=torch.int32)


@pytest.mark.skipif(
    not _FLAG_ON,
    reason="dynamic NKI disabled — export TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS=1 to run",
)
@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestGroupedMMRagged:
    """Pad-free grouped_mm: arbitrary offs, no alignment requirement.

    The aligned path is covered by test_grouped_mm.py, which runs with the flag off.
    """

    @pytest.mark.parametrize("kind", _GMM_RAGGED_KINDS)
    @pytest.mark.parametrize("r, d1, d2, g", _GMM_RAGGED_CASES)
    def test_ragged_2d_3d(self, r, d1, d2, g, kind):
        """a (r, d1) @ b (g, d1, d2) -> (r, d2) with unaligned offs."""
        dtype = torch.bfloat16
        torch.manual_seed(0)
        a = torch.randn((r, d1), dtype=dtype) / d1**0.5
        b = torch.randn((g, d1, d2), dtype=dtype) / d1**0.5
        offs = _gmm_ragged_offs(r, g, kind)

        # fp32 oracle: the kernel accumulates wider than bf16.
        result_ref = torch._grouped_mm(a.float(), b.float(), offs)

        with track_neuron_ops():
            result = torch._grouped_mm(a.to("neuron"), b.to("neuron"), offs.to("neuron"))
            assert_op_runs_on_neuron("aten::_grouped_mm")

        assert result.dtype == dtype, "ragged 2Dx3D must return the input dtype"
        assert result.shape == (r, d2)
        torch.testing.assert_close(result.float().cpu(), result_ref, rtol=1.6e-2, atol=1e-2)

    @pytest.mark.parametrize("kind", _GMM_RAGGED_KINDS)
    @pytest.mark.parametrize("t, d1, d2, g", _GMM_RAGGED_CASES)
    def test_ragged_2d_2d(self, t, d1, d2, g, kind):
        """a (d1, t) @ b (t, d2) -> (g, d1, d2) with unaligned offs."""
        dtype = torch.bfloat16
        torch.manual_seed(0)
        a = torch.randn((d1, t), dtype=dtype) / t**0.5
        b = torch.randn((t, d2), dtype=dtype) / t**0.5
        offs = _gmm_ragged_offs(t, g, kind)

        result_ref = torch._grouped_mm(a.float(), b.float(), offs)

        with track_neuron_ops():
            result = torch._grouped_mm(a.to("neuron"), b.to("neuron"), offs.to("neuron"))
            assert_op_runs_on_neuron("aten::_grouped_mm")

        assert result.dtype == dtype, "ragged 2Dx2D must return the input dtype"
        assert result.shape == (g, d1, d2)
        torch.testing.assert_close(result.float().cpu(), result_ref, rtol=1.6e-2, atol=1e-2)

    @pytest.mark.parametrize("d1", [352, 200, 96])
    def test_ragged_2d_3d_short_contraction(self, d1):
        """Contraction dim that is not a multiple of 128, so the last chunk is short.

        352 is the MoE hidden dim per shard at TP4, i.e. the down projection's
        contraction; 96 is a single short chunk. Only the 2Dx3D form allows this; the
        2Dx2D d1 is a different axis. Neuron needs a row stride that is a multiple of
        16 bytes, so d1 must stay a multiple of 8 for bfloat16.
        """
        r, d2, g = 1024, 256, 8
        dtype = torch.bfloat16
        torch.manual_seed(0)
        a = torch.randn((r, d1), dtype=dtype) / d1**0.5
        b = torch.randn((g, d1, d2), dtype=dtype) / d1**0.5
        offs = _gmm_ragged_offs(r, g, "random")
        result_ref = torch._grouped_mm(a.float(), b.float(), offs)

        with track_neuron_ops():
            result = torch._grouped_mm(a.to("neuron"), b.to("neuron"), offs.to("neuron"))
            assert_op_runs_on_neuron("aten::_grouped_mm")

        assert result.shape == (r, d2)
        torch.testing.assert_close(result.float().cpu(), result_ref, rtol=1.6e-2, atol=1e-2)

    @pytest.mark.parametrize("d2", [1280, 2048])
    def test_ragged_2d_3d_wide_output(self, d2):
        """Per-core output share above the 512 moving-operand limit, so it tiles.

        Every other case has a share <= 512 and one tile, leaving the final tile's
        width uncomputed.
        """
        r, d1, g = 384, 256, 4
        dtype = torch.bfloat16
        torch.manual_seed(0)
        a = torch.randn((r, d1), dtype=dtype) / d1**0.5
        b = torch.randn((g, d1, d2), dtype=dtype) / d1**0.5
        offs = _gmm_ragged_offs(r, g, "just_under_128")
        result_ref = torch._grouped_mm(a.float(), b.float(), offs)

        result = torch._grouped_mm(a.to("neuron"), b.to("neuron"), offs.to("neuron"))

        assert result.shape == (r, d2)
        torch.testing.assert_close(result.float().cpu(), result_ref, rtol=1.6e-2, atol=1e-2)

    @pytest.mark.parametrize("g", [128, 129, 256])
    def test_ragged_2d_2d_many_groups(self, g):
        """g at and past one partition block: the 2Dx2D schedule is computed in blocks
        of 128 experts, so a boundary crossing must not shift the table."""
        d1, t, d2 = 256, 2048, 128
        dtype = torch.bfloat16
        torch.manual_seed(0)
        a = torch.randn((d1, t), dtype=dtype) / t**0.5
        b = torch.randn((t, d2), dtype=dtype) / t**0.5
        offs = _gmm_ragged_offs(t, g, "random")
        result_ref = torch._grouped_mm(a.float(), b.float(), offs)

        result = torch._grouped_mm(a.to("neuron"), b.to("neuron"), offs.to("neuron"))

        assert result.shape == (g, d1, d2)
        torch.testing.assert_close(result.float().cpu(), result_ref, rtol=1.6e-2, atol=1e-2)

    @pytest.mark.parametrize("kind", ["random", "one_row_each", "aligned"])
    @pytest.mark.parametrize(
        "d1, d2",
        [
            (352, 128),  # short final D1 tile only
            (256, 2048),  # d2 past the moving free-dim limit only
            (352, 2048),  # both, as a DeepSeek-V3 w1/w3 weight gradient does
        ],
    )
    def test_ragged_2d_2d_wgrad_shapes(self, d1, d2, kind):
        """The 2Dx2D shapes a MoE backward pass asks for.

        A weight gradient has a (d_out, tokens) and b (tokens, d_in), so d1 and d2 are
        free of the 128 and 512 limits.
        """
        t, g = 2048, 8
        dtype = torch.bfloat16
        torch.manual_seed(0)
        a = torch.randn((d1, t), dtype=dtype) / t**0.5
        b = torch.randn((t, d2), dtype=dtype) / t**0.5
        offs = _gmm_ragged_offs(t, g, kind)
        result_ref = torch._grouped_mm(a.float(), b.float(), offs)

        with track_neuron_ops():
            result = torch._grouped_mm(a.to("neuron"), b.to("neuron"), offs.to("neuron"))
            assert_op_runs_on_neuron("aten::_grouped_mm")

        assert result.shape == (g, d1, d2)
        torch.testing.assert_close(result.float().cpu(), result_ref, rtol=1.6e-2, atol=1e-2)

    @pytest.mark.parametrize("form", ["2d3d", "2d2d"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_ragged_dtypes(self, dtype, form):
        """_RAGGED_DTYPES admits fp16 and fp32 for both forms, so cover both."""
        n, d1, d2, g = 384, 256, 128, 4
        torch.manual_seed(0)
        if form == "2d3d":
            a = (torch.randn((n, d1), dtype=torch.float32) / d1**0.5).to(dtype)
            b = (torch.randn((g, d1, d2), dtype=torch.float32) / d1**0.5).to(dtype)
            expected_shape = (n, d2)
        else:
            a = (torch.randn((d1, n), dtype=torch.float32) / n**0.5).to(dtype)
            b = (torch.randn((n, d2), dtype=torch.float32) / n**0.5).to(dtype)
            expected_shape = (g, d1, d2)
        offs = _gmm_ragged_offs(n, g, "random")
        result_ref = torch._grouped_mm(a.float(), b.float(), offs)
        rtol, atol = _TOL[dtype]

        result = torch._grouped_mm(a.to("neuron"), b.to("neuron"), offs.to("neuron"))

        assert result.dtype == dtype
        assert result.shape == expected_shape
        torch.testing.assert_close(result.float().cpu(), result_ref, rtol=rtol, atol=atol)

    @pytest.mark.parametrize(
        "opts",
        [{"bias": True}, {"out_dtype": torch.float32}],
        ids=["bias", "out_dtype"],
    )
    def test_ragged_declines_unsupported_options(self, opts):
        """The kernels add no bias and always return the input dtype, so a call asking
        for either must fall through rather than silently ignoring it."""
        from torch_neuronx.python_ops.grouped_mm import GroupedMMRaggedNKIImpl

        r, d1, d2, g = 384, 256, 128, 4
        a = torch.empty((r, d1), dtype=torch.bfloat16, device="meta")
        b = torch.empty((g, d1, d2), dtype=torch.bfloat16, device="meta")
        offs = torch.empty((g,), dtype=torch.int32, device="meta")
        impl = GroupedMMRaggedNKIImpl()

        assert impl.can_handle(a, b, offs), "baseline call must be accepted"

        kwargs = dict(opts)
        if kwargs.pop("bias", None):
            kwargs["bias"] = torch.empty((d2,), dtype=torch.bfloat16, device="meta")
        assert not impl.can_handle(a, b, offs, **kwargs)

    def test_ragged_no_recompile_across_routings(self):
        """Routing is data, not shape: offs values change every step at fixed shapes."""
        r, d1, d2, g = 2048, 512, 352, 8
        dtype = torch.bfloat16
        torch.manual_seed(0)
        a = (torch.randn((r, d1), dtype=dtype) / d1**0.5).to("neuron")
        b = (torch.randn((g, d1, d2), dtype=dtype) / d1**0.5).to("neuron")

        def step(seed):
            offs = _gmm_ragged_offs(r, g, "random", seed=seed).to("neuron")
            torch._grouped_mm(a, b, offs)
            torch_neuronx.synchronize()

        got = _recompiling_ops(step, warm_seed=1, probe_seeds=(5, 9, 13))
        assert not got, f"routing change must not recompile, got {sorted(got)}"
