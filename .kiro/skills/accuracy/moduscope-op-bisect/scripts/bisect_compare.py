#!/usr/bin/env python3
"""bisect_compare.py — localize the FIRST op/line where two per-op dump sequences diverge.

Loads a REFERENCE dump dir (e.g. CPU eager) and a TARGET dump dir (e.g. Neuron), matches
per-op artifacts by stage tag, computes a per-op divergence metric in EXECUTION ORDER, and
reports the FIRST stage whose target diverges beyond threshold — i.e. the op that introduced
the divergence (the bug lives between the last-OK stage and that stage).

Dump file convention (produced by instrument_dump.py / _dump_and_hook):
    <phase>-<stage>__value.pt   # forward value   (phase = "fwd")
    <phase>-<stage>__grad.pt    # backward grad   (phase = "bwd")
Execution order: read <ref>/ops_order.json (a JSON list of stage names) if present; else
sort stages by a numeric-aware natural key (so op00.. op10 order correctly).

Two metrics:
  relerr (default, 2 dirs): ||tgt - ref||_F / ||ref||_F
  ratio  (3 dirs: --fp32-dir --bf16-dir --tgt-dir): ||tgt-fp32|| / ||bf16-fp32|| (verdict)

NaN rule: if target is non-finite where reference is finite -> that stage is a [targetNaN]
divergence (reported as the origin if it is the first).

Usage:
  python bisect_compare.py --ref-dir CPU_DUMP --tgt-dir NEURON_DUMP \
      [--phase fwd|bwd] [--threshold 1.1]
  python bisect_compare.py --fp32-dir F --bf16-dir B --tgt-dir T [--phase fwd] [--threshold 1.1]
Exit code: 0 if no divergence beyond threshold, 1 if a divergence origin is found.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch


def _natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _suffix(phase):
    return "__grad.pt" if phase == "bwd" else "__value.pt"


def _stages(d: Path, phase):
    suf = _suffix(phase)
    pref = f"{phase}-"
    out = {}
    for p in d.glob(f"*{suf}"):
        name = p.name
        if name.startswith(pref):
            stage = name[len(pref) : -len(suf)]
        else:  # tolerate files without the phase prefix
            stage = name[: -len(suf)] if name.endswith(suf) else p.stem
        out[stage] = p
    return out


def _order(ref_dir: Path, stages):
    man = ref_dir / "ops_order.json"
    if man.exists():
        try:
            ordered = json.loads(man.read_text())
            return [s for s in ordered if s in stages] + [s for s in stages if s not in ordered]
        except Exception:
            pass
    return sorted(stages, key=_natural_key)


def _load(p):
    t = torch.load(p, weights_only=True)
    return t.detach().float() if isinstance(t, torch.Tensor) else torch.as_tensor(t).float()


def _fro(a, b):
    return (a - b).norm().item()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ref-dir")
    ap.add_argument("--tgt-dir", required=True)
    ap.add_argument("--fp32-dir")
    ap.add_argument("--bf16-dir")
    ap.add_argument("--phase", default="fwd", choices=["fwd", "bwd"])
    ap.add_argument("--threshold", type=float, default=1.1)
    args = ap.parse_args()

    tgt = _stages(Path(args.tgt_dir), args.phase)
    three = bool(args.fp32_dir and args.bf16_dir)
    if three:
        fp32 = _stages(Path(args.fp32_dir), args.phase)
        bf16 = _stages(Path(args.bf16_dir), args.phase)
        base_dir = Path(args.fp32_dir)
        common = list(_order(base_dir, set(fp32) & set(bf16) & set(tgt)))
        mode = "ratio (target_err/baseline_err)"
    else:
        if not args.ref_dir:
            ap.error("provide --ref-dir (2-dir relerr) or --fp32-dir+--bf16-dir (3-dir ratio)")
        ref = _stages(Path(args.ref_dir), args.phase)
        base_dir = Path(args.ref_dir)
        common = list(_order(base_dir, set(ref) & set(tgt)))
        mode = "relerr (||tgt-ref||/||ref||)"

    if not common:
        print("NO COMMON STAGES between dumps — check dirs/phase/naming.", file=sys.stderr)
        return 1

    print(f"phase={args.phase}  metric={mode}  threshold={args.threshold}  stages={len(common)}\n")
    print(f"{'#':>3} {'stage':<28} {'metric':>12} {'verdict':>10}")
    print("-" * 58)

    origin = None
    for i, s in enumerate(common):
        t = _load(tgt[s])
        tgt_nan = not torch.isfinite(t).all()
        if three:
            f = _load(fp32[s])
            b = _load(bf16[s])
            if not torch.isfinite(f).all():
                metric, verdict = float("nan"), "PASS-refNaN"  # bad reference, skip
            elif tgt_nan:
                metric, verdict = float("inf"), "FAIL-targetNaN"
            else:
                denom = _fro(b, f)
                metric = (
                    (_fro(t, f) / denom)
                    if denom > 0
                    else (0.0 if _fro(t, f) == 0 else float("inf"))
                )
                verdict = "PASS" if metric < args.threshold else "FAIL"
        else:
            r = _load(ref[s])
            if tgt_nan and torch.isfinite(r).all():
                metric, verdict = float("inf"), "FAIL-targetNaN"
            else:
                rn = r.norm().item()
                metric = (_fro(t, r) / rn) if rn > 0 else (0.0 if _fro(t, r) == 0 else float("inf"))
                verdict = "PASS" if metric < args.threshold else "FAIL"

        flag = "  <== FIRST DIVERGENCE" if (origin is None and verdict.startswith("FAIL")) else ""
        if origin is None and verdict.startswith("FAIL"):
            origin = (i, s, metric, verdict)
        mstr = (
            "nan" if math.isnan(metric) else (f"{metric:.4e}" if math.isfinite(metric) else "inf")
        )
        print(f"{i:>3} {s:<28} {mstr:>12} {verdict:>10}{flag}")

    print()
    if origin is None:
        print(
            f"RESULT: no divergence > {args.threshold} across {len(common)} stages "
            "— sequences agree."
        )
        return 0
    i, s, m, v = origin
    prev = common[i - 1] if i > 0 else "(input)"
    print(f"DIVERGENCE ORIGIN: stage #{i} '{s}' ({v}, metric={m}).")
    print(
        f"  -> The op producing '{s}' (between '{prev}' and '{s}') is where target first diverges."
    )
    print(
        f"  -> Inspect/instrument that single op next; upstream stages (< #{i}) "
        "match within tolerance."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
