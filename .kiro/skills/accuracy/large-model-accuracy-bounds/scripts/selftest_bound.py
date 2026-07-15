#!/usr/bin/env python3
"""selftest_bound.py — within-envelope accepted, outside rejected, grad-norm hover healthy."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perturbation_bound import envelope_verdict, gradnorm_health

# unperturbed GPU baseline observable per step, and a bf16-ULP perturbed GPU run
ref = [10.0 - 0.1 * t for t in range(20)]
pert = [r + 0.05 for r in ref]  # envelope ~0.05 per step

trn_good = [r + 0.03 for r in ref]  # inside envelope
trn_bad = [r + (0.03 if t < 15 else 0.20) for t, r in enumerate(ref)]  # exceeds late

ok_good, _ = envelope_verdict(ref, pert, trn_good)
ok_bad, per_bad = envelope_verdict(ref, pert, trn_bad)

# grad-norm diff hovering ~0 (healthy) vs drifting
gpu_gn = [2.0 + 0.01 * ((t * 7) % 5 - 2) for t in range(40)]
trn_gn_ok = [g + 0.02 * ((t * 3) % 5 - 2) for t, g in enumerate(gpu_gn)]  # small diffs
health = gradnorm_health(gpu_gn, trn_gn_ok, tol=0.1)

good_pass = ok_good is True
bad_reject = ok_bad is False and any(t >= 15 for t, p in enumerate(per_bad) if not p)
gn_healthy = health["frac_within_tol"] >= 0.9

if good_pass and bad_reject and gn_healthy:
    print(
        f"SELFTEST_BOUND OK — within-envelope accepted, outside rejected (exceeds late), "
        f"grad-norm within-tol={health['frac_within_tol']:.2f}"
    )
    sys.exit(0)
print(
    f"SELFTEST_BOUND FAIL — good={good_pass} bad_reject={bad_reject} "
    f"gn_healthy={gn_healthy} ({health})"
)
sys.exit(1)
