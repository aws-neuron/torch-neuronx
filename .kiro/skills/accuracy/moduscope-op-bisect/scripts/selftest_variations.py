#!/usr/bin/env python3
"""selftest_variations.py — validate the remaining bisect variations (torch only, no cluster):
V1: dump() LINE-LEVEL wrapper (fwd)      -> origin op2_add
V2: --phase bwd (grad dumps)             -> origin op04
V3: 3-tensor RATIO mode (fp32/bf16/tgt)  -> origin op03
"""

import json
import os
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import instrument_dump as ins

BC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bisect_compare.py")
tmp = tempfile.mkdtemp(prefix="variations_")


def rb(a):
    return subprocess.run([sys.executable, BC, *a], capture_output=True, text=True)


def f(x, p, d):
    ins._ORDER.clear()
    ins._SEEN.clear()
    os.environ["OPDUMP_DIR"] = d
    a = ins.dump("op0_scale", x * 2.0)
    b = ins.dump("op1_relu", torch.relu(a))
    c = ins.dump("op2_add", b + (0.05 if p else 0.0))
    e = ins.dump("op3_mul", c * 1.5)
    return ins.dump("op4_sum", e.sum(-1, keepdim=True))


x = torch.randn(4, 8)
f(x, False, tmp + "/v1r")
f(x, True, tmp + "/v1t")
v1 = (
    "ORIGIN: stage #2 'op2_add'"
    in rb(
        [
            "--ref-dir",
            tmp + "/v1r",
            "--tgt-dir",
            tmp + "/v1t",
            "--phase",
            "fwd",
            "--threshold",
            "1e-4",
        ]
    ).stdout
)

order = [f"op{i:02d}" for i in range(8)]
r2 = tmp + "/v2r"
t2 = tmp + "/v2t"
os.makedirs(r2)
os.makedirs(t2)
torch.manual_seed(1)
for i, s in enumerate(order):
    g = torch.randn(16, 16)
    torch.save(g, f"{r2}/bwd-{s}__grad.pt")
    torch.save(g + (0.3 if i >= 4 else 0.0), f"{t2}/bwd-{s}__grad.pt")
with open(f"{r2}/ops_order.json", "w") as _of:
    json.dump(order, _of)
v2 = (
    "ORIGIN: stage #4 'op04'"
    in rb(["--ref-dir", r2, "--tgt-dir", t2, "--phase", "bwd", "--threshold", "1e-3"]).stdout
)

f32, b16, tg = tmp + "/v3f", tmp + "/v3b", tmp + "/v3t"
for d in (f32, b16, tg):
    os.makedirs(d)
torch.manual_seed(2)
for i, s in enumerate(order):
    base = torch.randn(16, 16)
    torch.save(base, f"{f32}/fwd-{s}__value.pt")
    torch.save(base + 1e-3 * torch.randn(16, 16), f"{b16}/fwd-{s}__value.pt")
    torch.save(base + (0.5 if i >= 3 else 1e-3) * torch.randn(16, 16), f"{tg}/fwd-{s}__value.pt")
with open(f"{f32}/ops_order.json", "w") as _of:
    json.dump(order, _of)
v3 = (
    "ORIGIN: stage #3 'op03'"
    in rb(
        [
            "--fp32-dir",
            f32,
            "--bf16-dir",
            b16,
            "--tgt-dir",
            tg,
            "--phase",
            "fwd",
            "--threshold",
            "1.1",
        ]
    ).stdout
)

print(f"V1 dump() line-level: {'PASS' if v1 else 'FAIL'}")
print(f"V2 --phase bwd:       {'PASS' if v2 else 'FAIL'}")
print(f"V3 3-tensor ratio:    {'PASS' if v3 else 'FAIL'}")
print("SELFTEST_VARIATIONS", "OK" if (v1 and v2 and v3) else "FAIL")
sys.exit(0 if (v1 and v2 and v3) else 1)
