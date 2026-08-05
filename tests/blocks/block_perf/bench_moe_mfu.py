"""MoE MFU: static vs dynamic-NKI kernels, single device.

Runs {fixed, varying}-shape x {static, dynamic} in cold subprocesses (dynamic adapters
register at import under TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS=1) and reports median/mean
step time + MFU.

- fixed: identical routing every step -> both paths cache-hit -> the median gap is the
  dynamic per-op overhead.
- varying: new routing each step -> the dynamic kernels reuse one NEFF (recompile-free)
  while static recompiles per shape, so the gap is the recompile cost. Use mixtral-tiny
  to keep static's per-shape compiles tractable.

  python bench_moe_mfu.py --preset mixtral-tiny --seq_len 512 --steps 6 --warmup 2
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

_DYN_CC_FLAG = "--disable-internal-gather-scatter-bound-check"


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

    # Compiled-NEFF count after warmup; if it doesn't grow across timed steps the
    # dynamic kernels are recompile-free (0 for the static run — registry unused).
    dyn_neffs_after_warmup = _dynamic_neff_count()

    times = sorted(one_step(1000 + s) for s in range(steps))
    dyn_recompiles = _dynamic_neff_count() - dyn_neffs_after_warmup

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
                "dyn_recompiles": dyn_recompiles,
            },
            f,
        )


def _dynamic_neff_count():
    """Total compiled-NEFF count across all registered dynamic-kernel entries. 0 when
    the dynamic path is off (static run) — the registry is empty/unused there."""
    from torch_neuronx.python_ops.torch_mlir.ops.dynamic_kernels.registry import _registry

    return sum(len(e._compiled) for e in _registry.values())


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
            # recompile cost dynamic avoids; dyn_recompiles==0 confirms it stayed warm.
            print(
                f"  => dynamic recompile benefit: {s['median_ms'] - d['median_ms']:+.1f} "
                f"ms/step faster, MFU {d['mfu'] - s['mfu']:+.2f} pts"
            )
            print(
                f"     dynamic recompiles across {args.steps} varying steps: "
                f"{d.get('dyn_recompiles', '?')} (0 = recompile-free)"
            )


if __name__ == "__main__":
    if os.environ.get("MOE_MFU_WORKER") == "1":
        _worker(**json.loads(os.environ["MOE_MFU_ARGS"]))
    else:
        main()
