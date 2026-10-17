"""E2E + device-profile benchmark for the dynamic NKI collectives (AllGather + AllReduce).

Sweeps every (op, reduce_op, config, shape, dtype); for each cell x {dynamic, torch},
profiles the op, checks bit-exact correctness, and measures device execution as well as
the e2e latency.

One torchrun per (cell, mode): a cold, isolated profile is required so the kernel's NEFF
co-locates with its ntff.

Dynamic and torch are separate jobs because ``NEURON_DYN_NKI_COLLECTIVES`` is read once
(process-static) in the C++ backend/

Correctness: inputs are seeded per global rank, so any rank reconstructs the expected result
locally and compares bit-exactly (all-reduce uses small-integer shards so its SUM is exact).
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
from pathlib import Path

import pytest
import torch

# Mark multi_device so the ROOT conftest's autouse `ensure_neuron_initialized`
# fixture returns early in the parent pytest process (it otherwise reserves lnc0 for
# the whole session, starving the torchrun workers).
pytestmark = pytest.mark.multi_device

# ── target shapes (b, H, W); numel = H*W (b assumed 1) ──
_ALL_GATHER_SHAPES_FSDP = [
    (1, 1, 311568),
    (1, 1, 421103),
    (1, 1, 549152),
    (1, 1, 1321248),
    (1, 1, 3015184),
    (1, 1, 3276800),
    (1, 1, 3408384),
    (1, 1, 4861952),
    (1, 1, 4862080),
    (1, 1, 8208384),
    (1, 1, 8208640),
    (1, 1, 8650752),
    (1, 1, 9049140),
    (1, 1, 9724160),
    (1, 1, 12450240),
]

_ALL_GATHER_SHAPES_TP = [
    (1, 8192, 4096),
    (1, 4096, 4096),
    (1, 2048, 4096),
    (1, 1024, 4096),
    (1, 1024, 2048),
    (1, 512, 2880),
    (1, 128, 8192),
    (1, 128, 4096),
    (1, 128, 2048),
    (1, 128, 2880),
]

_DTYPE = torch.bfloat16  # default / all-gather dtype
_FP32 = torch.float32

# Sweep configs; the config bucket fixes the launch.
#  * tp:   contiguous group [0..nproc-1], one collective over all nproc ranks (group=4).
#  * fsdp: stride-TP group -- nproc = tp * world ranks, each rank's collective is its stride-tp
#          subgroup [r%tp, r%tp+tp, ...] (e.g. [0,4,...,60] for tp=4), i.e. group=16.
_FSDP_TP = 4
_FSDP_WORLD = 16
_CONFIG_LAUNCH = {
    "tp": {"nproc": 4, "tp": 1},
    "fsdp": {"nproc": _FSDP_TP * _FSDP_WORLD, "tp": _FSDP_TP},
}

# Each sweep cell is (op, reduce_op, config, H, W, dtype).
_ALL_GATHER_CELLS = [
    ("all_gather", "", "tp", h, w, _DTYPE) for _b, h, w in _ALL_GATHER_SHAPES_TP
] + [("all_gather", "", "fsdp", h, w, _DTYPE) for _b, h, w in _ALL_GATHER_SHAPES_FSDP]

_ALL_REDUCE_CELLS = [
    # bf16 SUM activations / MoE combine (dynamic-eligible), one shape per distinct numel bucket.
    ("all_reduce", "sum", "tp", 4096, 2048, _DTYPE),  # 8388608
    ("all_reduce", "sum", "tp", 2048, 2880, _DTYPE),  # 5898240
    ("all_reduce", "sum", "tp", 4096, 512, _DTYPE),  # 2097152 (MLA activation [1,seq,kv_lora])
    ("all_reduce", "sum", "tp", 512, 2880, _DTYPE),  # 1474560
    ("all_reduce", "sum", "tp", 576, 2048, _DTYPE),  # 1179648 (MoE expert-intermediate x hidden)
    ("all_reduce", "sum", "tp", 512, 2048, _DTYPE),  # 1048576
    ("all_reduce", "sum", "tp", 128, 2048, _DTYPE),  # 262144
    ("all_reduce", "sum", "tp", 64, 2048, _DTYPE),  # 131072
    ("all_reduce", "sum", "tp", 32, 2880, _DTYPE),  # 92160
    # fp32 (seq,1) seq-stats: MAX is the captured production dynamic bucket; SUM is a distinct NEFF.
    ("all_reduce", "max", "tp", 8192, 1, _FP32),  # dynamic (fp32 floor 4096)
    ("all_reduce", "max", "tp", 4096, 1, _FP32),  # dynamic (numel == small tile)
    ("all_reduce", "sum", "tp", 8192, 1, _FP32),
    ("all_reduce", "sum", "tp", 4096, 1, _FP32),
]

_CELLS = _ALL_GATHER_CELLS + _ALL_REDUCE_CELLS


def _plan_for(op, numel, world, lnc, elem_size, platform="trn2"):
    """Dynamic tiling plan for `op` (None => native fallback)."""
    from torch_neuronx.python_ops.torch_mlir.ops.dynamic_kernels.dynamic_collectives import (
        plan_dynamic_allgather,
        plan_dynamic_allreduce,
    )

    if op == "all_gather":
        return plan_dynamic_allgather(numel, world, lnc, elem_size, platform)
    return plan_dynamic_allreduce(numel, world, lnc, elem_size, platform)


def _op_total_bytes(op, group, numel, elem):
    """Bytes for the size/busbw columns: all-gather = gathered output (group x per-rank);
    all-reduce = per-rank message (output == input size)."""
    return group * numel * elem if op == "all_gather" else numel * elem


def _op_busbw(op, total_bytes, device_us, world):
    """Bus bandwidth (GiB/s). all-gather busbw = algbw*(n-1)/n; all-reduce = algbw*2*(n-1)/n."""
    if device_us <= 0 or world <= 1:
        return -1.0
    algbw = (total_bytes / device_us) * 1e6 / (1024**3)
    factor = (world - 1) / world if op == "all_gather" else 2 * (world - 1) / world
    return algbw * factor


_GROW_CC_FLAGS = ("--disable-internal-gather-scatter-bound-check",)
_NEURON_PROFILE_BIN = os.environ.get("NEURON_PROFILE_BIN", "/opt/aws/neuron/bin/neuron-explorer")
_BASE_RDZV_PORT = int(os.environ.get("AG_BASE_RDZV_PORT", "29500"))

# Session timestamp
if "NEURON_COVERAGE_TIMESTAMP" not in os.environ:
    os.environ["NEURON_COVERAGE_TIMESTAMP"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
_SESSION_TIMESTAMP = os.environ["NEURON_COVERAGE_TIMESTAMP"]


_CSV_COLUMNS = [
    "op_name",
    "reduce_op",  # sum|min|max for all_reduce; empty for non-reducing ops (all_gather)
    "platform",
    "dtype",  # element dtype of the collective (e.g. bfloat16)
    "total_mib",  # all_gather: gathered bytes (group*h*w*elem); all_reduce: per-rank bytes, in MiB
    "input_h",
    "input_w",
    "replica_group",  # ranks in this cell's collective group (== replica_groups[0])
    "world_size",  # len(replica_group)
    "regime",
    "large_tile_h",
    "medium_tile_h",
    "small_tile_h",
    "n_large",
    "n_medium",
    "n_small",
    "n_rem",
    "remainder_row_count",
    "unroll",  # flat unroll variant (1/2/4) auto-picked from tile count; 1=2-D, 0=native
    "dynamic_e2e_us",  # warm wall-clock, INCLUDES host overhead
    "torch_e2e_us",
    "e2e_ratio",  # dynamic_e2e_us / torch_e2e_us
    "dynamic_device_us",  # device total_time from neuron-profile
    "torch_device_us",
    "device_ratio",  # dynamic_device_us / torch_device_us
    "dynamic_cc_us",  # device cc_op_time from neuron-profile
    "torch_cc_us",
    "dynamic_bus_bw_gbps",  # derived from device total_time (busbw = algbw*(n-1)/n)
    "torch_bus_bw_gbps",
    "correctness_test",
    "last_updated",
    "error",  # failure detail, or empty when the cell ran clean
]


def tiling_row(
    op: str, h: int, w: int, world: int, lnc: int, elem_size: int, platform: str = "trn2"
) -> dict:
    """Tiling breakdown for the CSV.

    The kernel is shape-agnostic: it tiles the flattened ``numel = h*w`` in pow2 ELEMENT
    chunks (plan from ``_plan_for(op, numel, ...)``), so ``large_tile_h``/``small_tile_h`` carry
    ELEMENT tile sizes and ``remainder_row_count`` is leftover ELEMENTS. ``unroll`` is the
    auto-picked unroll variant (1/2/4) the dispatch uses.
    """
    total_bytes = _op_total_bytes(op, world, h * w, elem_size)
    numel = h * w
    # Flat/element tiling: plan takes numel directly; large/small/trace are pow2 ELEMENT counts.
    plan = _plan_for(op, numel, world, lnc, elem_size, platform)
    if plan is None:
        return {
            "total_bytes": total_bytes,
            "regime": "native-fallback",
            "large_tile_h": 0,
            "medium_tile_h": 0,
            "small_tile_h": 0,
            "n_large": 0,
            "n_medium": 0,
            "n_small": 0,
            "n_rem": 0,
            "remainder_row_count": 0,
            "unroll": 0,
        }

    large, medium, small = plan["large"], plan["medium"], plan["small"]
    n_large = numel // large
    r0 = numel % large
    n_medium = r0 // medium
    r1 = r0 % medium
    n_small = r1 // small
    remainder = r1 % small
    return {
        "total_bytes": total_bytes,
        "regime": plan["regime"],
        "large_tile_h": large,
        "medium_tile_h": medium,
        "small_tile_h": small,
        "n_large": n_large,
        "n_medium": n_medium,
        "n_small": n_small,
        "n_rem": 1 if remainder else 0,
        "remainder_row_count": remainder,
        "unroll": plan["unroll"],
    }


def _make_input(
    global_rank: int, h: int, w: int, op: str = "all_gather", dtype: torch.dtype = _DTYPE
) -> torch.Tensor:
    torch.manual_seed(7919 * (global_rank + 1) + h)
    if op == "all_reduce":
        # Small integers are exact in bf16 and their SUM across the group (<= group*4 <= 256)
        # stays exact, so all-reduce correctness is bit-exact (tests plumbing, not CCE precision).
        return torch.randint(-4, 5, (1, h, w)).to(dtype)
    return torch.randn(1, h, w, dtype=dtype)


def _check_result(
    op: str,
    out_cpu: torch.Tensor,
    group_ranks: list[int],
    h: int,
    w: int,
    reduce_name: str = "sum",
    dtype: torch.dtype = _DTYPE,
) -> bool:
    """all-gather: out_cpu[i] is the shard of group_ranks[i] (group-local order). all-reduce:
    out_cpu is the in-place reduced tensor == reduce over the group (bit-exact for integer
    shards; reduce_name in sum|min|max)."""
    if op == "all_gather":
        return all(
            torch.equal(out_cpu[i], _make_input(gr, h, w, op, dtype).squeeze(0))
            for i, gr in enumerate(group_ranks)
        )
    refs = torch.stack(
        [_make_input(gr, h, w, op, dtype).squeeze(0).to(torch.float32) for gr in group_ranks]
    )
    if reduce_name == "min":
        expected = refs.amin(0)
    elif reduce_name == "max":
        expected = refs.amax(0)
    else:
        expected = refs.sum(0)
    return torch.equal(out_cpu.squeeze(0), expected.to(dtype))


def _bench(fn, n_warmup: int = 3, n_iters: int = 10) -> float:
    """Warm wall-clock latency (us) including the host overhead."""
    import torch.distributed as dist

    for _ in range(n_warmup):
        fn()
    torch.neuron.synchronize()
    times = []
    for _ in range(n_iters):
        dist.barrier()
        torch.neuron.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.neuron.synchronize()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times) * 1e6


def _run_summary(cmd: list[str]):
    """Run a neuron-profile summary command; grab total_time / cc_op_time / mbu.
    Robust to a missing neuron-profile binary (returns all-None)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    except (FileNotFoundError, OSError) as e:
        return {"total_time": None, "cc_op_time": None, "mbu": None}, e
    out = r.stdout or ""

    def grab_max(key: str):
        # `view -d <root>` emits one section per neuron core (per rank), each with its own
        # metrics. A collective's latency is the SLOWEST rank, so take the max across sections
        # (the old first-match grabbed an arbitrary/first rank and under-reported).
        vals = [float(x) for x in re.findall(rf"^\s*{re.escape(key)}\s+([-+0-9.eE]+)", out, re.M)]
        return max(vals) if vals else None

    metrics = {
        "total_time": grab_max("total_time"),
        "cc_op_time": grab_max("cc_op_time"),
        "mbu": grab_max("mbu_estimated_percent"),
    }
    return metrics, r


def _parse_profile(profile_dir: str) -> dict:
    """Device metrics for the cell profiled into ``profile_dir``.

    ``view -d`` auto-discovers the ntff(s) and their NEFF(s) by UID when pointed at
    the profile-dir root. Prints a one-line reason whenever it falls back to the -1
    sentinel, to distinguish a missing/mis-located ntff from a profiler-binary/parse issue."""
    ntffs = glob.glob(os.path.join(profile_dir, "**", "*.ntff"), recursive=True)
    if not ntffs:
        print(f"[cc.profile] device_us=-1: no *.ntff under {profile_dir}", flush=True)
        return {"device_us": -1.0, "cc_us": -1.0}
    m, r = _run_summary(
        [_NEURON_PROFILE_BIN, "view", "-d", profile_dir, "--output-format", "summary-text"]
    )
    tt, cc = m.get("total_time"), m.get("cc_op_time")
    if tt is None:
        rc = getattr(r, "returncode", "n/a")
        err = (getattr(r, "stderr", "") or "")[:300]
        print(
            f"[cc.profile] device_us=-1: {_NEURON_PROFILE_BIN} parsed no total_time "
            f"(rc={rc}, ntff={ntffs[0]}); stderr={err!r}",
            flush=True,
        )
    return {
        "device_us": (tt * 1e6 if tt else -1.0),
        "cc_us": (cc * 1e6 if cc else -1.0),
    }


def _ratio(dyn: float, tor: float) -> float | str:
    """dynamic/torch timing ratio to 2 dp; '' when either measurement is missing
    (<=0, e.g. an unparsed device profile)."""
    if dyn is None or tor is None or dyn <= 0 or tor <= 0:
        return ""
    return round(dyn / tor, 2)


def _core_start_index() -> int:
    """Base logical-core index for this pytest worker's rank 0. Each xdist worker
    ``gw<k>`` owns a disjoint block of ``AG_CORE_BLOCK`` cores."""
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    worker_idx = int(worker[2:]) if worker.startswith("gw") else 0
    block = int(os.environ.get("AG_CORE_BLOCK", "4"))
    return worker_idx * block


def _is_worker() -> bool:
    return "LOCAL_RANK" in os.environ or "RANK" in os.environ


def worker() -> None:
    import torch.distributed as dist

    from torch_neuronx.profiling import NeuronConfig, NeuronProfiler, ProfileMode
    from torch_neuronx.utils import get_logical_neuron_cores, get_platform_target

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # Pin this rank to its own logical NeuronCore before nrt_init
    core_start = int(os.environ.get("AG_CORE_START", "0"))
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(core_start + local_rank)
    dist.init_process_group("neuron")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.neuron.set_device(rank)
    device = f"neuron:{rank}"
    lnc = int(get_logical_neuron_cores())

    mode = os.environ.get("AG_MODE", "dynamic")
    op = os.environ.get("AG_OP", "all_gather")
    reduce_name = os.environ.get("AG_REDUCE", "")  # sum|min|max for all_reduce; "" otherwise
    dtype_name = os.environ.get("AG_DTYPE", "bfloat16")  # per-cell dtype (bf16 default)
    dtype = getattr(torch, dtype_name)
    elem_size = torch.empty((), dtype=dtype).element_size()
    h = int(os.environ["AG_H"])
    w = int(os.environ["AG_W"])
    profile_root = os.environ.get("AG_PROFILE_DIR", os.path.abspath("./c10d_profiles"))
    profile_parts = [profile_root, op]
    if op == "all_reduce":
        profile_parts.append(reduce_name)
    profile_parts.extend((f"{h}x{w}", mode))
    base_dir = os.path.join(*profile_parts)
    if rank == 0:
        shutil.rmtree(base_dir, ignore_errors=True)
        os.makedirs(base_dir, exist_ok=True)
    dist.barrier()

    inp = _make_input(rank, h, w, op, dtype).to(device)
    tp = int(os.environ.get("AG_TP", "1"))
    if tp > 1:
        # Stride-TP FSDP replica group: rank r joins subgroup [r%tp, r%tp+tp, ...] -- one core
        # per device across all FSDP devices (e.g. [0,4,...,60] for tp=4), matching production.
        # All ranks build all tp subgroups (dist.new_group is collective); this rank uses its own.
        assert world % tp == 0, f"world {world} not divisible by AG_TP {tp}"
        t = rank % tp
        subgroups = [dist.new_group(ranks=list(range(gt, world, tp))) for gt in range(tp)]
        cc_group = subgroups[t]
        group_ranks = list(range(t, world, tp))
    else:
        cc_group = None  # default (world) group
        group_ranks = list(range(world))

    if op == "all_gather":
        out = torch.empty(len(group_ranks), h, w, dtype=dtype, device=device)

        def run(o=out, i=inp, g=cc_group):
            dist.all_gather_into_tensor(o, i, group=g)

        profiled = run  # single device op already; profile it as-is
        result_tensor = out
    else:
        # all_reduce is in-place; `run` (warm bench + correctness) resets a scratch from the
        # unchanged input each call so repeats don't re-reduce an already-reduced buffer.
        _reduce_ops = {"sum": dist.ReduceOp.SUM, "min": dist.ReduceOp.MIN, "max": dist.ReduceOp.MAX}
        rop = _reduce_ops.get(reduce_name, dist.ReduceOp.SUM)
        buf = inp.clone()

        def run(b=buf, i=inp, g=cc_group, r=rop):
            b.copy_(i)
            dist.all_reduce(b, op=r, group=g)

        def profiled(b=buf, g=cc_group, r=rop):
            dist.all_reduce(b, op=r, group=g)

        result_tensor = buf

    # (1) COLD device profile: exactly one execution inside the profiler so the
    # NEFF dumps co-located with the ntff.
    cfg = NeuronConfig(
        modes=[
            ProfileMode.DEVICE,
            ProfileMode.RUNTIME,
            ProfileMode.CPU_UTIL,
            ProfileMode.HOST_MEMORY,
        ],
        profile_output_dir=base_dir,
        max_events_per_nc=100000,
    )
    nprof = NeuronProfiler(cfg)
    torch.neuron.synchronize()
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.PrivateUse1,
    ]
    with torch.profiler.profile(
        activities=activities,
        experimental_config=cfg,
        on_trace_ready=nprof.export_trace,
    ):
        profiled()
        torch.neuron.synchronize()

    # (2) correctness (after profiling; bit-exact vs reconstructed reference)
    run()
    torch.neuron.synchronize()
    ok = _check_result(op, result_tensor.cpu(), group_ranks, h, w, reduce_name, dtype)

    # (3) e2e latency: warm wall-clock INCLUDING host overhead (unprofiled)
    e2e_us = _bench(run)
    dist.barrier()

    if rank == 0:
        # The collective spans only the group's members (FSDP subgroup, not the full 64-rank
        # world), so size/tiling/busbw use the group size to match the kernel the backend
        # compiles (world = len(replica_groups[0])).
        group_size = len(group_ranks)
        total_bytes = _op_total_bytes(op, group_size, h * w, elem_size)
        prof_m = _parse_profile(base_dir)
        row = tiling_row(op, h, w, group_size, lnc, elem_size, get_platform_target())
        result = {
            "op": op,
            "reduce_op": reduce_name,
            "mode": mode,
            "platform": get_platform_target(),
            "dtype": dtype_name,
            "input_h": h,
            "input_w": w,
            "replica_group": str(group_ranks),
            "world_size": group_size,
            "e2e_us": round(e2e_us, 3),
            "device_us": round(prof_m["device_us"], 3),
            "cc_us": round(prof_m["cc_us"], 3),
            "bus_bw_gbps": round(_op_busbw(op, total_bytes, prof_m["device_us"], group_size), 4),
            "ok": ok,
            "error": "" if ok else "correctness mismatch vs reconstructed reference",
            **row,
        }
        print(
            f"[cc.result] {op}{('/' + reduce_name) if reduce_name else ''} {mode} [{h},{w}] "
            f"e2e_us={result['e2e_us']:.2f} "
            f"device_us={result['device_us']:.2f} cc_us={result['cc_us']:.2f} "
            f"bus_bw={result['bus_bw_gbps']:.2f} GiB/s correct={'PASS' if ok else 'FAIL'}",
            flush=True,
        )
        rf = os.environ.get("AG_RESULT_FILE")
        if rf:
            with open(rf, "w") as f:
                json.dump(result, f)
    if dist.is_initialized():
        dist.destroy_process_group()


# ── driver: one cold torchrun per (shape, mode); merge into the CSV ──
# Env the child torchrun must not inherit from the parent pytest process: rendezvous
# vars (fresh process group per job); NEURON_RT_VISIBLE_CORES / NEURON_RT_NUM_CORES
# (the conftest / dev pods pin a single core, but a collective torchrun needs
# nproc x LNC cores); NEURON_RT_ROOT_COMM_ID (fresh comm per job); and the NEFF/HLO
# cache dirs (when set, the profiler copies the NEFF to the cache dir instead of
# co-locating it with the ntff, so the profile parse finds nothing). The dynamic-
# feature env (gate + CC control-flow) is not listed here -- _apply_mode_env clears
# and re-sets it per-mode.
_STRIP_CHILD_ENV = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RUN_ID",
    "NEURON_RT_VISIBLE_CORES",
    "NEURON_RT_NUM_CORES",
    "NEURON_RT_ROOT_COMM_ID",
    "TORCH_NEURONX_NEFF_CACHE_DIR",
    "TORCH_NEURONX_HLO_CACHE_DIR",
)


def _apply_mode_env(env: dict, mode: str) -> None:
    """Apply the dynamic-feature env for mode=='dynamic' only; for the 'torch'
    baseline the gate + CC control-flow are UNSET and the grow flags stripped, so
    torch is a clean static/native measurement that can never route to the dynamic
    kernel."""
    # Start clean: drop gate, CC control-flow, and any grow flags inherited via
    # the parent's NEURON_CC_FLAGS.
    env.pop("NEURON_DYN_NKI_COLLECTIVES", None)
    env.pop("NEURON_RT_EXPERIMENTAL_CC_CONTROL_FLOW_ENABLE", None)
    cc = env.get("NEURON_CC_FLAGS", "")
    for tok in _GROW_CC_FLAGS:
        cc = cc.replace(tok, "")
    cc = " ".join(cc.split())  # normalize whitespace after removals
    if mode == "dynamic":
        env["NEURON_DYN_NKI_COLLECTIVES"] = "1"
        env["NEURON_RT_EXPERIMENTAL_CC_CONTROL_FLOW_ENABLE"] = "1"
        for tok in _GROW_CC_FLAGS:
            if tok not in cc:
                cc = (cc + " " + tok).strip()

    else:
        env["NEURON_DYN_NKI_COLLECTIVES"] = "0"

    if cc:
        env["NEURON_CC_FLAGS"] = cc
    else:
        env.pop("NEURON_CC_FLAGS", None)


def _run_job(
    op: str,
    reduce_op: str,
    h: int,
    w: int,
    mode: str,
    nproc: int,
    tp: int,
    profile_dir: str,
    env_base: dict,
    dtype: torch.dtype = _DTYPE,
) -> dict:
    """Spawn one cold torchrun for (op, reduce_op, shape, mode); return the worker's result dict."""
    rf_fd, rf = tempfile.mkstemp(suffix=".json")
    os.close(rf_fd)
    env = dict(env_base)
    for k in _STRIP_CHILD_ENV:  # fresh rendezvous + full core visibility per job
        env.pop(k, None)
    env["AG_H"], env["AG_W"], env["AG_MODE"], env["AG_RESULT_FILE"] = str(h), str(w), mode, rf
    env["AG_OP"] = op  # all_gather | all_reduce
    env["AG_REDUCE"] = reduce_op  # sum|min|max for all_reduce; "" otherwise
    env["AG_DTYPE"] = str(dtype).removeprefix("torch.")  # per-cell dtype (bf16 / float32)
    env["AG_TP"] = str(tp)  # >1 => worker runs the stride-TP FSDP subgroup collective
    env["AG_PROFILE_DIR"] = profile_dir
    # Also pass this worker's rank-0 core base to the child so its per-rank pinning
    # (NEURON_RT_VISIBLE_CORES = start + rank) starts from this xdist worker's slice.
    env["AG_CORE_START"] = str(_core_start_index())
    _apply_mode_env(env, mode)
    # Unique torchrun rendezvous port per pytest worker.
    port = _BASE_RDZV_PORT + _core_start_index()
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        "--master-addr=127.0.0.1",
        f"--master-port={port}",
        str(Path(__file__).resolve()),
    ]
    print(
        f"\n[cc.driver] === torchrun {op}{('/' + reduce_op) if reduce_op else ''} [{h},{w}] "
        f"mode={mode} nproc={nproc} port={port} ===",
        flush=True,
    )
    proc = subprocess.run(cmd, env=env, check=False)
    try:
        if os.path.isfile(rf) and os.path.getsize(rf) > 0:
            with open(rf) as f:
                return json.load(f)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(rf)
    # No result file => the torchrun job crashed / timed out / OOM'd before rank 0 wrote its
    # result. Surface it in the `error` column (blank error would hide the failure).
    return {
        "e2e_us": -1.0,
        "device_us": -1.0,
        "cc_us": -1.0,
        "bus_bw_gbps": -1.0,
        "ok": False,
        "error": f"{mode} torchrun produced no result (exit={proc.returncode})",
    }


def _base_env() -> dict:
    """Env shared by every torchrun job (mode-neutral); the dynamic-feature env is
    applied PER-MODE in _apply_mode_env."""
    env_base = dict(os.environ)
    env_base["NEURON_LOGICAL_NC_CONFIG"] = env_base.get("NEURON_LOGICAL_NC_CONFIG", "2")
    env_base["NEURON_RT_ENABLE_DGE_NOTIFICATIONS"] = "1"  # DGE profiling
    return env_base


def _run_shape(
    op: str,
    reduce_op: str,
    h: int,
    w: int,
    nproc: int,
    tp: int,
    profile_dir: str,
    env_base: dict,
    dtype: torch.dtype = _DTYPE,
) -> dict:
    """Run one (op, reduce_op, shape) through cold dynamic + torch torchrun jobs; return its CSV
    row. ``nproc``/``tp`` come from the sweep config (see _CONFIG_LAUNCH): tp>1 runs the stride-TP
    FSDP replica group, tp==1 the contiguous group over all ``nproc`` ranks."""
    d = _run_job(op, reduce_op, h, w, "dynamic", nproc, tp, profile_dir, env_base, dtype)
    t = _run_job(op, reduce_op, h, w, "torch", nproc, tp, profile_dir, env_base, dtype)
    # tiling columns are mode-independent; prefer the dynamic worker's values
    # (computed on-device), fall back to torch's, then to sentinels.
    src = d if "regime" in d else (t if "regime" in t else {})
    return {
        "op_name": op,
        "reduce_op": reduce_op,
        "platform": d.get("platform") or t.get("platform", ""),
        "dtype": d.get("dtype") or t.get("dtype", ""),
        "total_mib": (
            round(src.get("total_bytes", 0) / (1024 * 1024), 3) if src.get("total_bytes") else ""
        ),
        "input_h": h,
        "input_w": w,
        "replica_group": d.get("replica_group") or t.get("replica_group", ""),
        "world_size": d.get("world_size") or t.get("world_size", ""),
        "regime": src.get("regime", "?"),
        "large_tile_h": src.get("large_tile_h", ""),
        "medium_tile_h": src.get("medium_tile_h", ""),
        "small_tile_h": src.get("small_tile_h", ""),
        "n_large": src.get("n_large", ""),
        "n_medium": src.get("n_medium", ""),
        "n_small": src.get("n_small", ""),
        "n_rem": src.get("n_rem", ""),
        "remainder_row_count": src.get("remainder_row_count", ""),
        "unroll": src.get("unroll", ""),
        "dynamic_e2e_us": d.get("e2e_us", -1.0),
        "torch_e2e_us": t.get("e2e_us", -1.0),
        "e2e_ratio": _ratio(d.get("e2e_us", -1.0), t.get("e2e_us", -1.0)),
        "dynamic_device_us": d.get("device_us", -1.0),
        "torch_device_us": t.get("device_us", -1.0),
        "device_ratio": _ratio(d.get("device_us", -1.0), t.get("device_us", -1.0)),
        "dynamic_cc_us": d.get("cc_us", -1.0),
        "torch_cc_us": t.get("cc_us", -1.0),
        "dynamic_bus_bw_gbps": d.get("bus_bw_gbps", -1.0),
        "torch_bus_bw_gbps": t.get("bus_bw_gbps", -1.0),
        "correctness_test": "pass" if d.get("ok") else "fail",
        "last_updated": _SESSION_TIMESTAMP,
        "error": d.get("error") or t.get("error", ""),
    }


def driver(out_csv: str, profile_dir: str) -> list[dict]:
    env_base = _base_env()
    rows = []
    for op, reduce_op, config, h, w, dtype in _CELLS:
        launch = _CONFIG_LAUNCH[config]
        rows.append(
            _run_shape(
                op, reduce_op, h, w, launch["nproc"], launch["tp"], profile_dir, env_base, dtype
            )
        )

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ag.driver] wrote {len(rows)} rows -> {out_csv}", flush=True)
    return rows


def _neuron_devices_present() -> bool:
    """Detect Neuron devices WITHOUT initializing NRT. Calling
    torch_neuronx.device_count() here would nrt_init() in the long-lived pytest
    parent and reserve lnc0 for the whole session, starving the spawned torchrun
    workers (rank 0 then hits 'Requested:lnc0 Available:0 ... cores busy')."""
    return bool(glob.glob("/dev/neuron*"))


def _collectives_metrics_dir() -> Path:
    """Directory where each parametrized shape drops its CSV row as JSON, for the
    conftest ``pytest_sessionfinish`` (main process) to merge into one timestamped
    CSV. Shared with conftest via ``NEURON_COVERAGE_METRICS_DIR`` so xdist workers
    and the controller agree on the path."""
    base = Path(os.environ.get("NEURON_COVERAGE_METRICS_DIR", tempfile.gettempdir()))
    return base / "neuron_collectives_metrics"


def _save_shape_row(
    op: str, reduce_op: str, config: str, h: int, w: int, dtype_label: str, row: dict
) -> None:
    """Persist one cell's CSV row to
    ``<metrics_dir>/<op>[_<reduce_op>]_<config>_<h>x<w>_<dtype>_<worker>.json``. Under pytest-xdist
    each ``gw<k>`` worker writes its own files; the controller merges them on session finish (see
    conftest). The op + reduce_op + config + dtype prefix keeps distinct cells for the same
    (h, w) from clobbering each other."""
    metrics_dir = _collectives_metrics_dir()
    metrics_dir.mkdir(parents=True, exist_ok=True)
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    rlabel = f"_{reduce_op}" if reduce_op else ""
    (metrics_dir / f"{op}{rlabel}_{config}_{h}x{w}_{dtype_label}_{worker}.json").write_text(
        json.dumps(row)
    )


def _shape_params():
    """One pytest param per (op, reduce_op, config, h, w, dtype) cell, id
    = ``<op>[-<reduce_op>]-<config>-<h>x<w>-<dtype>`` so ``-k`` filters by op/reduce/config/shape/
    dtype and pytest-xdist distributes cells across workers."""
    params = []
    for op, reduce_op, config, h, w, dtype in _CELLS:
        rlabel = f"-{reduce_op}" if reduce_op else ""
        dlabel = str(dtype).removeprefix("torch.")
        pid = f"{op}{rlabel}-{config}-{h}x{w}-{dlabel}"
        params.append(pytest.param(op, reduce_op, config, h, w, dtype, id=pid))
    return params


# ── pytest entry: one parametrized case per (op, reduce_op, config, shape, dtype) cell ──
@pytest.mark.skipif(not _neuron_devices_present(), reason="no Neuron devices (/dev/neuron*)")
@pytest.mark.multi_device
@pytest.mark.parametrize(("op", "reduce_op", "config", "h", "w", "dtype"), _shape_params())
def test_dynamic_collectives(op, reduce_op, config, h, w, dtype):
    """Run one (op, reduce_op, config, shape, dtype) cell through cold dynamic + torch torchrun
    profile jobs, assert bit-exactness (for dynamically-routed cells) and that both e2e latencies
    were recorded."""
    launch = _CONFIG_LAUNCH[config]
    profile_dir = os.environ.get("AG_PROFILE_DIR", os.path.abspath("c10d_profiles"))
    row = _run_shape(
        op, reduce_op, h, w, launch["nproc"], launch["tp"], profile_dir, _base_env(), dtype
    )
    _save_shape_row(op, reduce_op, config, h, w, str(dtype).removeprefix("torch."), row)

    # both modes must have recorded a warm e2e latency
    assert row["dynamic_e2e_us"] > 0, f"missing dynamic e2e for {row}"
    assert row["torch_e2e_us"] > 0, f"missing torch e2e for {row}"
    # dynamically-routed cells must be bit-exact
    if row["regime"] != "native-fallback":
        assert row["correctness_test"] == "pass", f"correctness failed: {row}"


if __name__ == "__main__":
    if _is_worker():
        worker()
    else:
        _csv = os.environ.get("AG_CSV", "/tmp/dynamic_collectives.csv")
        _pdir = os.environ.get("AG_PROFILE_DIR", os.path.abspath("./c10d_profiles"))
        driver(_csv, _pdir)
