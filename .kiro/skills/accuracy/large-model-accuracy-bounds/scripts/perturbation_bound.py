#!/usr/bin/env python3
"""perturbation_bound.py — accept/reject a TRN bf16 run vs GPU bf16, no fp32 needed.

Envelope mode (default): given per-step observables for the unperturbed GPU run (ref), the
GPU run perturbed by a bf16-ULP error at step 0 (pert), and the TRN run (trn), accept when
|trn - ref| <= |pert - ref| (the self-perturbation envelope) at each step.

Grad-norm mode (--gradnorm): given grad_norm(GPU_bf16) and grad_norm(TRN_bf16) per step,
report the fraction of steps with |diff| <= tol (should be ~90%) and mean/median diff, split
early vs late.

Inputs are plain lists (or JSON files of lists). This is metric-agnostic: the observable can be
loss, a layer activation Frobenius norm, or grad-norm.
"""

import argparse
import json
import statistics as st


def envelope_verdict(ref, pert, trn, slack=0.0):
    """Return (accept_all, per_step) where per_step[t]=True if TRN within the bf16-ULP envelope."""
    n = min(len(ref), len(pert), len(trn))
    per = []
    for t in range(n):
        env = abs(pert[t] - ref[t])
        dev = abs(trn[t] - ref[t])
        per.append(dev <= env + slack)
    return (all(per), per)


def gradnorm_health(gpu_gn, trn_gn, tol=0.1):
    n = min(len(gpu_gn), len(trn_gn))
    diffs = [gpu_gn[t] - trn_gn[t] for t in range(n)]
    within = [d for d in diffs if abs(d) <= tol]
    half = n // 2

    def frac(xs):
        return (sum(1 for d in xs if abs(d) <= tol) / len(xs)) if xs else float("nan")

    return {
        "n": n,
        "frac_within_tol": len(within) / n if n else float("nan"),
        "mean_diff": st.fmean(diffs) if diffs else float("nan"),
        "median_diff": st.median(diffs) if diffs else float("nan"),
        "frac_within_early": frac(diffs[:half]),
        "frac_within_late": frac(diffs[half:]),
        "tol": tol,
    }


def _load(x):
    try:
        return json.load(open(x))
    except (OSError, ValueError):
        return json.loads(x)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gradnorm", action="store_true")
    ap.add_argument("--ref")
    ap.add_argument("--pert")
    ap.add_argument("--trn")
    ap.add_argument("--gpu-gn")
    ap.add_argument("--trn-gn")
    ap.add_argument("--tol", type=float, default=0.1)
    a = ap.parse_args()
    if a.gradnorm:
        print(gradnorm_health(_load(a.gpu_gn), _load(a.trn_gn), a.tol))
    else:
        ok, per = envelope_verdict(_load(a.ref), _load(a.pert), _load(a.trn))
        exceed = [t for t, p in enumerate(per) if not p]
        print(f"ACCEPT={ok}  steps_within={sum(per)}/{len(per)}  exceed_at={exceed[:20]}")
