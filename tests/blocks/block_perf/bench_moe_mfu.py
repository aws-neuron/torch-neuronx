"""MoE MFU: static vs dynamic-NKI kernels, single device.

Runs {fixed, varying}-shape x {static, dynamic} in cold subprocesses (dynamic adapters
register at import under TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS=1) and reports median/mean
step time + MFU.

- fixed: identical routing every step -> both paths cache-hit -> the median gap is the
  dynamic per-op overhead.
- varying: new routing each step -> the dynamic kernels (mm/silu/mul/gather/scatter)
  reuse one NEFF per op while static recompiles them per shape, so dynamic is faster.
  The dynamic_recompiling_ops column lists the ops that still recompile.
  Use mixtral-tiny to keep static's per-shape compiles tractable.

  python bench_moe_mfu.py --preset mixtral-tiny --seq_len 512 --steps 6 --warmup 2
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

_DYN_CC_FLAG = "--disable-internal-gather-scatter-bound-check"

# One row per shape_mode, static + dynamic paired, for the dashboard parquet.
# fixed row answers "dynamic overhead at static shapes"; varying row answers
# "dynamic vs static incl. static's recompile cost" (dynamic_recompiling_ops names the
# ops that still recompile on the dynamic path).
CSV_COLUMNS = [
    "preset",
    "seq_len",
    "mode",  # fwd | fwd+bwd
    "shape_mode",  # fixed | varying
    "static_median_ms",
    "dynamic_median_ms",
    "static_mfu",
    "dynamic_mfu",
    "mfu_delta_pts",  # dynamic_mfu - static_mfu
    "overhead_ratio",  # dynamic_median_ms / static_median_ms
    "dynamic_recompiling_ops",  # "op:files;..." ops that still recompile on the dynamic path
    "last_updated",
]


def _worker(preset, seq_len, batch, steps, warmup, out_json, shape_mode="varying", fwd_only=True):
    import torch
    from block_perf import moe_dynamic as m
    from run_benchmarks import calculate_mfu_trn2

    import torch_neuronx

    dev = "neuron"
    cfg = m.create_config(preset, batch, seq_len)
    block = m.create_block(cfg, torch.device(dev), dtype=torch.bfloat16)
    T = batch * seq_len  # noqa: N806

    def make_input(step):
        # fixed: same seed -> identical routing every step. varying: per-step seed ->
        # different per-expert token counts.
        seed = 1000 if shape_mode == "fixed" else 1000 + step
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(batch, seq_len, cfg["hidden_size"], dtype=torch.bfloat16, generator=g) / 8
        return x.to(dev)

    def one_step(step):
        x = make_input(step)
        torch.neuron.synchronize()  # finish input H2D copy so the timer excludes it (not in FLOPs)
        t0 = time.perf_counter()
        if fwd_only:
            with torch.no_grad():
                _ = m.run_block(block, x)
        else:
            block.zero_grad(set_to_none=True)
            torch.tanh(m.run_block(block, x).sum()).backward()
        torch.neuron.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    # Warmup one shape (fixed) or distinct shapes (varying) so timed steps cache-hit.
    for s in range(warmup):
        one_step(1000 if shape_mode == "fixed" else s)

    times = sorted(one_step(1000 + s) for s in range(steps))

    # Un-timed pass: which ops recompile per new shape (0 growth = NEFF reused).
    recompiling_ops = _attribute_recompiles(_neff_cache_dir(), one_step, shape_mode)

    fwd_flops = m.count_flops(cfg, active_tokens=cfg["num_experts_per_tok"] * T)
    flops = fwd_flops if fwd_only else fwd_flops * 3  # bwd ~= 2x fwd
    median_ms = times[len(times) // 2]
    tflops = flops / (median_ms / 1000) / 1e12
    with open(out_json, "w") as f:
        json.dump(
            {
                "median_ms": median_ms,
                "mean_ms": sum(times) / len(times),
                "tflops": tflops,
                "mfu": calculate_mfu_trn2(tflops)["mfu_percent"],
                "times": times,
                "recompiling_ops": recompiling_ops,
            },
            f,
        )


def _neff_cache_dir():
    import tempfile as _t

    default = os.path.join(_t.gettempdir(), "neff_cache")
    return os.environ.get("TORCH_NEURONX_NEFF_CACHE_DIR", default)


def _attribute_recompiles(cache_dir, one_step, shape_mode):
    """Run two more steps with TorchMlirKernel.lower hooked, attributing NEFF-cache-file
    growth to each op_name. Returns "op:files;..." (most-recompiled first), or "" if none.

    Syncs after each lower() so the compile's cache write completes before diffing —
    otherwise the write is still in flight and growth reads as 0 (false recompile-free).
    Uses fresh seeds (2000+) so the ops face shapes not compiled in warmup/timed. For
    fixed mode both extra steps are the same shape, so nothing should recompile."""
    import glob
    from collections import Counter

    import torch

    from torch_neuronx.python_ops.torch_mlir import kernel as kmod

    def nfiles():
        return len(glob.glob(os.path.join(cache_dir, "**", "*"), recursive=True))

    grow = Counter()
    orig = kmod.TorchMlirKernel.lower

    def traced(kernel_self, *a, **k):
        before = nfiles()
        out = orig(kernel_self, *a, **k)
        torch.neuron.synchronize()  # flush the compile's cache write before diffing
        d = nfiles() - before
        if d > 0:
            grow[getattr(kernel_self, "op_name", "?")] += d
        return out

    # Prime one fresh shape unhooked (so first-time compiles here aren't miscounted),
    # then hook and run another fresh shape — its growth is per-shape recompilation.
    one_step(2000 if shape_mode == "varying" else 1000)
    kmod.TorchMlirKernel.lower = traced
    try:
        one_step(2001 if shape_mode == "varying" else 1000)
    finally:
        kmod.TorchMlirKernel.lower = orig
    return ";".join(f"{op}:{n}" for op, n in grow.most_common())


def _run(mode, shape_mode, args):
    out_fd, out_json = tempfile.mkstemp(suffix=f"_{shape_mode}_{mode}.json")
    os.close(out_fd)
    env = dict(os.environ)
    env.pop("TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS", None)
    cc = env.get("NEURON_CC_FLAGS", "").replace(_DYN_CC_FLAG, "")
    cc = " ".join(cc.split())
    if mode == "dynamic":
        env["TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS"] = "1"
        cc = (cc + " " + _DYN_CC_FLAG).strip()
    if cc:
        env["NEURON_CC_FLAGS"] = cc
    # Isolated NEFF cache per subprocess so runs don't share compiles; each run's own
    # warmup populates it, keeping the timed steps warm without cross-run leakage.
    neff_cache = tempfile.mkdtemp(prefix=f"neff_{shape_mode}_{mode}_")
    env["TORCH_NEURONX_NEFF_CACHE_DIR"] = neff_cache
    env["MOE_MFU_WORKER"] = "1"
    env["MOE_MFU_ARGS"] = json.dumps(
        {
            "preset": args.preset,
            "seq_len": args.seq_len,
            "batch": args.batch_size,
            "steps": args.steps,
            "warmup": args.warmup,
            "out_json": out_json,
            "shape_mode": shape_mode,
            "fwd_only": args.fwd_only,
        }
    )
    print(f"\n[{shape_mode}/{mode}] running {args.steps} steps...", flush=True)
    try:
        r = subprocess.run([sys.executable, os.path.abspath(__file__)], env=env)
    finally:
        shutil.rmtree(neff_cache, ignore_errors=True)  # drop this run's isolated cache
    if r.returncode != 0:
        print(f"[{shape_mode}/{mode}] FAILED rc={r.returncode}")
        os.unlink(out_json)
        return None
    with open(out_json) as f:
        result = json.load(f)
    os.unlink(out_json)
    return result


def main():
    p = argparse.ArgumentParser(description="MoE static-vs-dynamic MFU")
    p.add_argument("--preset", default="mixtral-8x7b")
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--shapes", default="both", choices=["fixed", "varying", "both"])
    p.add_argument("--fwd-only", action="store_true", help="forward-only (default: fwd+loss+bwd)")
    p.add_argument("--csv", default=None, help="write one paired row per shape_mode to this CSV")
    args = p.parse_args()
    mode_str = "fwd" if args.fwd_only else "fwd+bwd"

    shape_modes = ["fixed", "varying"] if args.shapes == "both" else [args.shapes]
    res = {(sm, m): _run(m, sm, args) for sm in shape_modes for m in ("static", "dynamic")}

    print("\n" + "=" * 74)
    print(f"MoE MFU  preset={args.preset} seq={args.seq_len} steps={args.steps} mode={mode_str}")
    print("=" * 74)
    for sm in shape_modes:
        print(f"\n-- {sm} shapes --")
        for mode in ("static", "dynamic"):
            r = res.get((sm, mode))
            if not r:
                print(f"  {mode:<8}: FAILED")
                continue
            print(
                f"  {mode:<8}: median {r['median_ms']:.1f} ms  mean {r['mean_ms']:.1f} ms  "
                f"MFU {r['mfu']:.2f}%"
            )
        s, d = res.get((sm, "static")), res.get((sm, "dynamic"))
        if s and d and sm == "fixed":
            # both cache-hit -> the gap is the dynamic per-op overhead.
            print(
                f"  => dynamic overhead: {d['median_ms'] - s['median_ms']:+.1f} ms/step "
                f"({d['median_ms']/s['median_ms']:.2f}x), MFU {d['mfu'] - s['mfu']:+.2f} pts"
            )
        if s and d and sm == "varying":
            # static recompiles per shape, dynamic reuses one NEFF -> the gap is the
            # recompile cost dynamic avoids on its covered kernels (glue ops still recompile).
            print(
                f"  => dynamic recompile benefit: {s['median_ms'] - d['median_ms']:+.1f} "
                f"ms/step faster ({d['median_ms']/s['median_ms']:.2f}x)"
            )
            ops = d.get("recompiling_ops", "")
            print(
                f"     ops still recompiling on the dynamic path: {ops or 'NONE (recompile-free)'}"
            )

    if args.csv:
        _write_csv(args.csv, args, mode_str, shape_modes, res)


def _write_csv(path, args, mode_str, shape_modes, res):
    """One paired row per shape_mode (static+dynamic side-by-side). A shape_mode
    with either side failed is skipped."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for sm in shape_modes:
        s, d = res.get((sm, "static")), res.get((sm, "dynamic"))
        if not (s and d):
            continue
        rows.append(
            {
                "preset": args.preset,
                "seq_len": args.seq_len,
                "mode": mode_str,
                "shape_mode": sm,
                "static_median_ms": round(s["median_ms"], 3),
                "dynamic_median_ms": round(d["median_ms"], 3),
                "static_mfu": round(s["mfu"], 3),
                "dynamic_mfu": round(d["mfu"], 3),
                "mfu_delta_pts": round(d["mfu"] - s["mfu"], 3),
                "overhead_ratio": round(d["median_ms"] / s["median_ms"], 3)
                if s["median_ms"]
                else "",
                "dynamic_recompiling_ops": d.get("recompiling_ops", ""),
                "last_updated": now,
            }
        )
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[csv] wrote {len(rows)} rows -> {path}", flush=True)


if __name__ == "__main__":
    if os.environ.get("MOE_MFU_WORKER") == "1":
        _worker(**json.loads(os.environ["MOE_MFU_ARGS"]))
    else:
        main()
