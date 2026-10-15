#!/usr/bin/env python3
"""selftest_bisect.py — deterministic acceptance test for the op-bisect skill.

Fakes a 10-op forward chain on "cpu" (reference) and the SAME 10 ops on "neuron" (target)
with exactly ONE op perturbed (op05). Because ops chain, the perturbation propagates to
op06..op09 too — so a naive diff shows 5 differing ops. The bisector must report the FIRST
one (op05) as the divergence origin. No cluster, no real Neuron — pure tensor artifacts.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PERTURB_AT = 5
N = 10


def _write_chain(d: Path, perturb):
    d.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    h = torch.randn(64, 128)  # identical seed => identical starting point on both sides
    order = []
    for i in range(N):
        h = torch.tanh(h @ torch.eye(128) * 1.01 + 0.1)  # a deterministic per-op transform
        if perturb and i == PERTURB_AT:
            h = h + 0.05 * torch.ones_like(h)  # inject the bug at op05 (propagates)
        stage = f"op{i:02d}"
        order.append(stage)
        torch.save(h.clone(), d / f"fwd-{stage}__value.pt")
    (d / "ops_order.json").write_text(json.dumps(order))


def main():
    tmp = Path(tempfile.mkdtemp(prefix="opbisect_selftest_"))
    ref, tgt = tmp / "cpu", tmp / "neuron"
    _write_chain(ref, perturb=False)  # reference: clean 10-op chain
    _write_chain(tgt, perturb=True)  # target: identical except op05 perturbed
    print(f"[selftest] ref={ref}  tgt={tgt}  (perturbed op{PERTURB_AT:02d})")

    out = subprocess.run(
        [
            sys.executable,
            str(HERE / "bisect_compare.py"),
            "--ref-dir",
            str(ref),
            "--tgt-dir",
            str(tgt),
            "--phase",
            "fwd",
            "--threshold",
            "1e-3",
        ],
        capture_output=True,
        text=True,
    )
    print(out.stdout)
    if out.returncode != 1:
        print("SELFTEST FAIL — expected a divergence (exit 1), got", out.returncode)
        return 2

    want = f"op{PERTURB_AT:02d}"
    # origin line: "DIVERGENCE ORIGIN: stage #5 'op05' ..."
    origin_ok = f"ORIGIN: stage #{PERTURB_AT} '{want}'" in out.stdout
    # ops before the perturb must be PASS (identical), op05.. must FAIL
    upstream_pass = all(
        ("PASS" in out.stdout.split(f"op{i:02d}")[1].splitlines()[0]) for i in range(PERTURB_AT)
    )
    if origin_ok and upstream_pass:
        print(
            f"SELFTEST OK — origin = {want} (upstream op00..op0{PERTURB_AT - 1} match; "
            f"{want} is the first divergence)"
        )
        return 0
    print(
        "SELFTEST FAIL — origin not correctly localized. origin_ok=",
        origin_ok,
        "upstream_pass=",
        upstream_pass,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
