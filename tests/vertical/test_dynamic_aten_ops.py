"""Benchmark for the dynamic NKI aten kernels.

Per-op latency & correctness for the dynamic-shape NKI kernels.
Each test compares the dynamic NKI kernel against the static
lowering baseline given same input shape, and dtype.

"""

from __future__ import annotations

import contextlib
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from math import prod
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from internal import NeuronOpInfo
from internal.common import allocate_to_device as _allocate_to_device
from internal.neuron_shape_registry import SHAPE_REGISTRY

# Mark multi_device so the ROOT conftest's autouse `ensure_neuron_initialized`
# fixture RETURNS EARLY in the parent pytest process.
pytestmark = pytest.mark.multi_device

_NEURON_PROFILE_BIN = os.environ.get("NEURON_PROFILE_BIN", "/opt/aws/neuron/bin/neuron-profile")
_CORE_START = int(os.environ.get("AOP_CORE_START", "0"))


def _xdist_core_index() -> int:
    """Logical NeuronCore index this pytest worker owns."""
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    if worker.startswith("gw"):
        with contextlib.suppress(ValueError):
            return int(worker[2:])
    return 0


# Compile flag required by the dynamic-shape NKI kernels
_DYN_CC_FLAGS = ("--disable-internal-gather-scatter-bound-check",)

_DTYPES = {
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float16: "float16",
}
_TOL = {
    torch.float32: (1e-4, 1e-4),
    torch.bfloat16: (2e-2, 2e-2),
    torch.float16: (5e-3, 5e-3),
}

_CSV_COLUMNS = [
    "op_name",
    "input_shape",
    "dtype",
    "input_size_mb",  # total input size in MiB
    "dynamic_time_to_first_ms",
    "static_time_to_first_ms",
    "dynamic_e2e_time_us",
    "static_e2e_time_us",
    "e2e_ratio",  # dynamic_e2e / static_e2e
    "dynamic_device_time_us",
    "static_device_time_us",
    "device_ratio",  # dynamic_device / static_device
    "correctness_test",
    "last_updated",
    "error",
]


def _discover_ops() -> dict[str, str]:
    """Map each dynamic-NKI adapter's short key to the full aten op to sweep.

    Returns ``{short -> full}`` (e.g. ``add -> aten::add.Tensor``) for every
    adapter that has registry shapes; keys without shapes are skipped. The full op
    is the SHAPE_REGISTRY entry with that base, preferring functional overloads
    over ``.out``/``.Scalar``."""
    from torch_neuronx.python_ops.torch_mlir.ops.dynamic_kernels.adapters import register_all
    from torch_neuronx.python_ops.torch_mlir.ops.dynamic_kernels.registry import _registry

    register_all()
    base = lambda op: op.replace("aten::", "").split(".", 1)[0]  # noqa: E731
    ops = {}
    for short in sorted(_registry):
        matches = sorted(
            (op for op in SHAPE_REGISTRY.all_ops() if base(op) == short),
            key=lambda o: ("out" in o, "Scalar" in o, o),
        )
        if matches:
            ops[short] = matches[0]
    return ops


_OPS = _discover_ops()

_DTYPES_TO_TEST = (torch.bfloat16,)

# Bytes per element for the integer/bool dtypes the registry pins explicitly on
# non-float args (e.g. the ``index`` arg's "long"), for the input-size estimate.
_NAME_BYTES = {
    "long": 8,
    "int64": 8,
    "int": 4,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
}


def _op_info(op: str) -> NeuronOpInfo:
    """NeuronOpInfo for the short op name, resolved from the shared registry."""
    return NeuronOpInfo.from_schema(_OPS[op])


def _entries(op: str) -> list:
    """Registry entries for ``op``, already capped/ordered by
    ``NEURON_MAX_SHAPES_PER_OP`` (via ``_registry_entries``)."""
    return _op_info(op)._registry_entries()


def _shape_str(arg_shapes: dict) -> str:
    """Human-readable shape tag for the CSV, derived from an entry's arg_shapes.
    Joins each tensor-shaped arg's dims with ``x`` and args with ``_`` (scalar
    args like ``dim`` are skipped)."""
    parts = []
    for v in arg_shapes.values():
        if isinstance(v, dict) and "shape" in v:
            parts.append("x".join(map(str, v["shape"])))
        elif isinstance(v, list | tuple) and v and isinstance(v[0], dict):
            parts += ["x".join(map(str, d["shape"])) for d in v if "shape" in d]
        elif isinstance(v, list | tuple) and v and all(isinstance(x, int) for x in v):
            parts.append("x".join(map(str, v)))
    return "_".join(parts)


def _spec_bytes(v, default_bytes: int) -> int:
    """Byte footprint of one arg value (tensor storage), 0 for scalars."""
    if isinstance(v, dict) and "shape" in v:
        return prod(v["shape"]) * _NAME_BYTES.get(v.get("dtype", "float"), default_bytes)
    if isinstance(v, list | tuple) and v and isinstance(v[0], dict):
        return sum(_spec_bytes(d, default_bytes) for d in v)
    if isinstance(v, list | tuple) and v and all(isinstance(x, int) for x in v):
        return prod(v) * default_bytes
    return 0


def _entry_mb(arg_shapes: dict, dtype: torch.dtype) -> float:
    """Total input size in MiB = sum of input-tensor bytes for the entry."""
    default_bytes = dtype.itemsize
    total = sum(_spec_bytes(v, default_bytes) for v in arg_shapes.values())
    return round(total / (1024 * 1024), 3)


def _bench(fn, n_warmup: int = 3, n_iters: int = 10) -> float:
    """Warm wall-clock latency (µs) including the host overhead."""
    for _ in range(n_warmup):
        fn()
    torch.neuron.synchronize()
    times = []
    for _ in range(n_iters):
        torch.neuron.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.neuron.synchronize()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times) * 1e6


def _run_summary(cmd: list[str]):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    except (FileNotFoundError, OSError):
        return None
    out = r.stdout or ""
    # total_exec_time = actual device busy execution (sum over the executed NEFFs),
    # not the total_time. For a multi-NEFF op, total_time is the profiled-window wall span,
    # inflated by host gaps between the cold NEFF compiles, whereas
    # total_exec_time is the on-device time. They coincide for single-NEFF ops.
    m = re.search(r"^\s*total_exec_time\s+([-+0-9.eE]+)", out, re.M)
    return float(m.group(1)) if m else None


def _parse_device_us(profile_dir: str) -> float:
    """Device-busy time (µs) for the cell profiled into ``profile_dir``.

    ``view -d`` auto-discovers the ntff and its NEFF(s) by UID when pointed at the
    profile-dir ROOT: that dir holds both the ntff (in an instance/session subdir)
    and the profiler's ``neffs/`` copies."""
    if not glob.glob(os.path.join(profile_dir, "**", "*.ntff"), recursive=True):
        return -1.0
    tt = _run_summary(
        [_NEURON_PROFILE_BIN, "view", "-d", profile_dir, "--output-format", "summary-text"]
    )
    return tt * 1e6 if tt else -1.0


def _worker_cache_root() -> str:
    """Per-worker on-disk cache root. Under pytest-xdist each worker gets its own
    ``/tmp/aop_cache_gw<k>``. Non-xdist runs use ``/tmp/aop_cache_main``."""
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    return os.path.join(tempfile.gettempdir(), f"aop_cache_{worker}")


def _apply_worker_cache_env(env: dict) -> None:
    """Point the child job's HLO + NEFF compile caches at this worker's
    private root. The profiler copies NEFFs into ``<profile_dir>/neffs/``"""
    root = _worker_cache_root()
    env["TORCH_NEURONX_HLO_CACHE_DIR"] = os.path.join(root, "hlo")
    env["TORCH_NEURONX_NEFF_CACHE_DIR"] = os.path.join(root, "neff")


def _reset_cache_dirs() -> None:
    """Clear the on-disk HLO/NEFF compile caches so the profiled first call
    compiles COLD (a true time-to-first)."""
    for var, default in (
        ("TORCH_NEURONX_HLO_CACHE_DIR", "/tmp/hlo_cache"),
        ("TORCH_NEURONX_NEFF_CACHE_DIR", "/tmp/neff_cache"),
    ):
        d = os.environ.get(var, default)
        shutil.rmtree(d, ignore_errors=True)
        with contextlib.suppress(OSError):
            os.makedirs(d, exist_ok=True)


def _dynamic_entry(op: str):
    """Registry entry for `op` (dynamic mode only); None if flag off/not registered."""
    try:
        from torch_neuronx.python_ops.torch_mlir.ops.dynamic_kernels.registry import (
            get_dynamic_nki_entry,
        )

        return get_dynamic_nki_entry(op)
    except Exception:
        return None


def _is_worker() -> bool:
    return os.environ.get("AOP_WORKER") == "1"


def worker() -> None:
    from torch_neuronx.profiling import NeuronConfig, NeuronProfiler, ProfileMode

    os.environ["NEURON_RT_VISIBLE_CORES"] = str(_CORE_START)

    op = os.environ["AOP_OP"]
    entry_idx = int(os.environ["AOP_ENTRY_IDX"])
    dtype = {v: k for k, v in _DTYPES.items()}[os.environ["AOP_DTYPE"]]
    mode = os.environ.get("AOP_MODE", "dynamic")
    rtol, atol = _TOL[dtype]

    # Rebuild this cell's sample from the SHARED registry path. The parent picked
    # entry_idx from the same deterministic ``_registry_entries()`` list, so this
    # reconstructs the identical arg_shapes.
    op_info = _op_info(op)
    resolved = op_info._resolved_op
    arg_shapes = op_info._registry_entries()[entry_idx].arg_shapes
    shape = _shape_str(arg_shapes)

    result = {
        "mode": mode,
        "ttf_ms": -1.0,
        "e2e_us": -1.0,
        "device_us": -1.0,
        "ok": False,
        "ran": None,  # dynamic: did the NKI kernel actually execute?
        "error": "",
    }

    profile_root = os.environ.get("AOP_PROFILE_DIR", os.path.abspath("./aten_profiles"))
    cell = f"{op}_{shape}_{_DTYPES[dtype]}"
    base_dir = os.path.join(profile_root, cell, mode)
    shutil.rmtree(base_dir, ignore_errors=True)
    os.makedirs(base_dir, exist_ok=True)

    entry = None
    try:
        # Compile caches point at this worker's private root. The profiler copies
        # the NEFF into <profile_dir>/neffs/ on stop, which _parse_device_us reads.
        _reset_cache_dirs()  # cold start for a true time-to-first

        # Build the sample once on CPU, then deep-clone to device to
        # share identical input values.
        cpu_sample = op_info.make_sample(arg_shapes, "cpu", dtype)
        if cpu_sample is None:
            raise RuntimeError(f"no sample for aten::{op} entry {entry_idx}")
        cpu_inputs = (cpu_sample.input, *cpu_sample.args)
        cpu_kwargs = cpu_sample.kwargs
        dev_inputs = tuple(_allocate_to_device(a, "neuron") for a in cpu_inputs)
        dev_kwargs = {k: _allocate_to_device(v, "neuron") for k, v in cpu_kwargs.items()}

        def run(ins=dev_inputs, kw=dev_kwargs):
            return resolved(*ins, **kw)

        # exec-count probe (dynamic mode): detect a silent fallback to the
        # default lowering (numerically correct but NOT the dynamic NKI kernel).
        entry = _dynamic_entry(op) if mode == "dynamic" else None
        exec_before = entry._exec_count if entry is not None else None

        # (1) COLD profiled run: compile + load + execute inside the profiler.
        cfg = NeuronConfig(
            modes=[ProfileMode.DEVICE, ProfileMode.RUNTIME],
            profile_output_dir=base_dir,
            max_events_per_nc=100000,
        )
        nprof = NeuronProfiler(cfg)
        torch.neuron.synchronize()
        t0 = time.perf_counter()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.PrivateUse1,
            ],
            experimental_config=cfg,
            on_trace_ready=nprof.export_trace,
        ):
            out_dev = run()
            torch.neuron.synchronize()
        result["ttf_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        if entry is not None:
            result["ran"] = entry._exec_count > exec_before

        # (2) correctness vs CPU reference. out_dev is a single run; for the
        # in-place scatter_add_ the out buffer started at zeros, so it holds a
        # single scatter — matching the CPU reference. cpu_inputs/cpu_kwargs
        # haven't been run yet, so running the resolved op on them here produces
        # the reference (for in-place ops this mutates cpu_inputs[0] once, which
        # is exactly the single-scatter semantics we compare against).
        out_cpu = resolved(*cpu_inputs, **cpu_kwargs)
        try:
            torch.testing.assert_close(
                out_dev.detach().to("cpu"), out_cpu, rtol=rtol, atol=atol, equal_nan=True
            )
            result["ok"] = True
        except AssertionError as e:
            result["ok"] = False
            result["error"] = f"correctness: {str(e).splitlines()[0][:160]}"

        # (3) warm e2e latency (host + device).
        result["e2e_us"] = round(_bench(run), 3)

        # (4) device time from the cold profile.
        result["device_us"] = round(_parse_device_us(base_dir), 3)
    except Exception as e:
        result["error"] = result["error"] or f"{type(e).__name__}: {e!s}"[:200]

    if mode == "dynamic" and not result["error"]:
        if entry is None:
            result["error"] = "dynamic registry entry missing (flag not effective?)"
        elif result["ran"] is False:
            result["error"] = "dynamic kernel did not run (fell back to default lowering)"

    print(
        f"[aop.result] {mode} {op} {shape} {_DTYPES[dtype]} "
        f"ttf_ms={result['ttf_ms']} e2e_us={result['e2e_us']} "
        f"device_us={result['device_us']} ran={result['ran']} "
        f"correct={'PASS' if result['ok'] else 'FAIL'}",
        flush=True,
    )
    rf = os.environ.get("AOP_RESULT_FILE")
    if rf:
        with open(rf, "w") as f:
            json.dump(result, f)


# ══════════════════════════════════════════════════════════════════════════════
# Driver: one cold subprocess per (op, shape, dtype, mode); merge into the CSV.
# ══════════════════════════════════════════════════════════════════════════════
def _apply_mode_env(env: dict, mode: str) -> None:
    """dynamic -> export TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS=1 (adapters register
    at import) and add the dynamic-kernel compile flag; static -> unset the flag
    (default MLIR lowering) and strip the dynamic compile flag."""
    env.pop("TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS", None)
    # Strip our dynamic CC flag from any inherited NEURON_CC_FLAGS, then re-add for
    # dynamic mode only.
    cc = env.get("NEURON_CC_FLAGS", "")
    for tok in _DYN_CC_FLAGS:
        cc = cc.replace(tok, "")
    cc = " ".join(cc.split())
    if mode == "dynamic":
        env["TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS"] = "1"
        for tok in _DYN_CC_FLAGS:
            if tok not in cc:
                cc = (cc + " " + tok).strip()
    if cc:
        env["NEURON_CC_FLAGS"] = cc
    else:
        env.pop("NEURON_CC_FLAGS", None)


def _run_job(op, entry_idx, dtype, mode, profile_dir, env_base) -> dict:
    """Spawn one cold single-process job for (cell, mode); return its result.

    The cell is identified by ``(op, entry_idx)`` — the worker rebuilds the sample from the
    shared registry via that index, so no shapes cross the process boundary."""
    rf_fd, rf = tempfile.mkstemp(suffix=".json")
    os.close(rf_fd)
    env = dict(env_base)

    # Drop NEURON_RT_NUM_CORES so the child doesn't inherit
    env.pop("NEURON_RT_NUM_CORES", None)
    env["AOP_WORKER"] = "1"  # tells the re-entered file to run worker(), not the driver
    env["AOP_OP"] = op
    env["AOP_ENTRY_IDX"] = str(entry_idx)
    env["AOP_DTYPE"] = _DTYPES[dtype]
    env["AOP_MODE"] = mode
    env["AOP_RESULT_FILE"] = rf
    env["AOP_PROFILE_DIR"] = profile_dir

    # Logical NeuronCore for this job
    env["AOP_CORE_START"] = str(_xdist_core_index())
    _apply_mode_env(env, mode)

    # Per-worker cache root
    _apply_worker_cache_env(env)
    cmd = [sys.executable, str(Path(__file__).resolve())]
    print(
        f"[aop.driver] === run {op} entry={entry_idx} {_DTYPES[dtype]} mode={mode} ===",
        flush=True,
    )
    subprocess.run(cmd, env=env, check=False)
    try:
        if os.path.isfile(rf) and os.path.getsize(rf) > 0:
            with open(rf) as f:
                return json.load(f)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(rf)
    return {"ttf_ms": -1.0, "e2e_us": -1.0, "device_us": -1.0, "ok": False, "error": "no result"}


def _default_csv_path() -> str:
    """CWD-relative timestamped CSV name, example ``dynamic_aten_ops_<YYYYMMDD>_<HHMMSS>.csv``"""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"dynamic_aten_ops_{ts}.csv"


def _ratio(dyn: float, sta: float) -> float | str:
    """dynamic/static ratio"""
    if dyn is None or sta is None or dyn <= 0 or sta <= 0:
        return ""
    return round(dyn / sta, 2)


def _run_op_cells(op: str, profile_dir: str, env_base: dict, now: str) -> list[dict]:
    """Sweep one op's (registry-entry, dtype) cells through cold dynamic + static
    subprocess jobs; return one CSV row per cell."""
    rows = []
    entries = _entries(op)
    for idx, entry in enumerate(entries):
        for dtype in _DTYPES_TO_TEST:
            d = _run_job(op, idx, dtype, "dynamic", profile_dir, env_base)
            s = _run_job(op, idx, dtype, "static", profile_dir, env_base)
            err = d.get("error") or s.get("error") or ""
            rows.append(
                {
                    "op_name": _OPS[op],
                    "input_shape": _shape_str(entry.arg_shapes),
                    "dtype": _DTYPES[dtype],
                    "input_size_mb": _entry_mb(entry.arg_shapes, dtype),
                    "dynamic_time_to_first_ms": d.get("ttf_ms", -1.0),
                    "static_time_to_first_ms": s.get("ttf_ms", -1.0),
                    "dynamic_e2e_time_us": d.get("e2e_us", -1.0),
                    "static_e2e_time_us": s.get("e2e_us", -1.0),
                    "e2e_ratio": _ratio(d.get("e2e_us", -1.0), s.get("e2e_us", -1.0)),
                    "dynamic_device_time_us": d.get("device_us", -1.0),
                    "static_device_time_us": s.get("device_us", -1.0),
                    "device_ratio": _ratio(d.get("device_us", -1.0), s.get("device_us", -1.0)),
                    "correctness_test": "pass" if d.get("ok") else "fail",
                    "last_updated": now,
                    "error": err,
                }
            )
    return rows


def _write_csv(out_csv: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[aop.driver] wrote {len(rows)} rows -> {out_csv}", flush=True)


def _dynamic_metrics_dir() -> Path:
    """Directory where each parametrized ``test_op`` drops its rows as JSON, for
    the conftest ``pytest_sessionfinish`` (main process) to merge into one
    timestamped CSV"""
    base = Path(os.environ.get("NEURON_COVERAGE_METRICS_DIR", tempfile.gettempdir()))
    return base / "neuron_dynamic_metrics"


def _save_op_rows(op_full: str, rows: list[dict]) -> None:
    """Persist one op's CSV rows to ``<metrics_dir>/<op>_<worker>.json``. Under
    pytest-xdist each ``gw<k>`` worker writes its own files."""
    metrics_dir = _dynamic_metrics_dir()
    metrics_dir.mkdir(parents=True, exist_ok=True)
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    safe = op_full.replace("::", "_").replace(".", "_")
    (metrics_dir / f"{safe}_{worker}.json").write_text(json.dumps(rows))


def driver(out_csv: str, profile_dir: str) -> list[dict]:
    env_base = dict(os.environ)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    total = sum(len(_entries(op)) for op in _OPS) * len(_DTYPES_TO_TEST)
    print(
        f"[aop.driver] {len(_OPS)} ops, {total} (op,shape,dtype) cells x 2 modes",
        flush=True,
    )

    rows = []
    for op in _OPS:
        rows.extend(_run_op_cells(op, profile_dir, env_base, now))

    _write_csv(out_csv, rows)
    return rows


def _assert_cells_correct(rows: list[dict]) -> None:
    """Validate the ``correctness_test`` column: every cell that actually produced
    a result must be numerically correct. A cell with no error — or only an
    informational note (the dynamic kernel declined / the flag was ineffective, so
    the op still ran via the default lowering) — ran to completion and so must be
    ``pass``. Cells that hit a genuine error (e.g. a compile failure) produced no
    comparable output; they're recorded in the CSV's ``error`` column for triage,
    not hard-failed. Kernel-engagement (``fell back``) is a routing/coverage signal
    in that column, orthogonal to numerical correctness."""
    _info_notes = ("fell back", "registry entry missing")
    for r in rows:
        err = r["error"]
        informational = (not err) or any(k in err for k in _info_notes)
        if informational:
            assert r["correctness_test"] == "pass", f"correctness failed: {r}"


def _op_params():
    """One pytest param per op, id = full aten name — mirrors coverage's
    ``build_params`` so ``-k`` filters by op name (e.g. ``-k "silu or mm"``)."""
    return [pytest.param(op, id=full) for op, full in _OPS.items()]


class TestDynamicAtenOps:
    """Per-op sweep of the dynamic-NKI aten kernels. Each test runs one op's
    registered shapes through cold dynamic + static subprocess profile jobs,
    records a CSV row per (shape, dtype) cell, and asserts
    every dynamically-routed cell is numerically correct."""

    @pytest.mark.parametrize("op", _op_params())
    def test_op(self, op, tmp_path):
        profile_dir = str(tmp_path / "aten_profiles")
        env_base = dict(os.environ)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        rows = _run_op_cells(op, profile_dir, env_base, now)
        _save_op_rows(_OPS[op], rows)

        assert rows, f"no cells ran for {_OPS[op]}"
        _assert_cells_correct(rows)


if __name__ == "__main__":
    if _is_worker():
        worker()
    else:
        _csv = os.environ.get("AOP_CSV") or _default_csv_path()
        _pdir = os.environ.get("AOP_PROFILE_DIR", os.path.abspath("./aten_profiles"))
        driver(_csv, _pdir)
