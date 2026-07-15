#!/usr/bin/env python3
"""selftest_realmodule.py — REAL end-to-end test of the op-bisect instrumentation.

Unlike selftest_bisect.py (which fakes the dump files), this exercises the ACTUAL
instrument_dump.register_all_hooks on a real nn.Module: it dumps per-submodule fwd values
+ bwd grads for a reference run, perturbs ONE layer's weight for the "target" run, dumps
again, and confirms bisect_compare.py localizes that layer as the first divergence.
No Neuron needed (CPU only). NKI nl.device_print mode is NOT covered here — that requires
a real NKI kernel on a Neuron node.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import instrument_dump as ins  # noqa: E402

PERTURB_LAYER = "l2"  # -> stage 004_l2


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(32, 32)
        self.a0 = nn.ReLU()
        self.l1 = nn.Linear(32, 32)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(32, 32)
        self.a2 = nn.ReLU()
        self.l3 = nn.Linear(32, 32)

    def forward(self, x):
        x = self.a0(self.l0(x))
        x = self.a1(self.l1(x))
        x = self.a2(self.l2(x))
        return self.l3(x)


def _run(net, x, d):
    ins._ORDER.clear()
    ins._SEEN.clear()
    os.environ["OPDUMP_DIR"] = d
    os.environ["OPDUMP_TAG"] = ""
    handles = ins.register_all_hooks(net)
    net(x).sum().backward()
    for h in handles:
        h.remove()


def main():
    tmp = Path(tempfile.mkdtemp(prefix="opbisect_real_"))
    ref, tgt = str(tmp / "cpu"), str(tmp / "neuron")
    torch.manual_seed(0)
    net_ref = Net()
    torch.manual_seed(0)
    net_tgt = Net()  # identical init
    x = torch.randn(8, 32)
    _run(net_ref, x.clone().requires_grad_(True), ref)  # reference
    getattr(net_tgt, PERTURB_LAYER).weight.data += 0.05  # inject a divergent op
    _run(net_tgt, x.clone().requires_grad_(True), tgt)  # target

    out = subprocess.run(
        [
            sys.executable,
            str(HERE / "bisect_compare.py"),
            "--ref-dir",
            ref,
            "--tgt-dir",
            tgt,
            "--phase",
            "fwd",
            "--threshold",
            "1e-4",
        ],
        capture_output=True,
        text=True,
    )
    print(out.stdout)
    want = f"'004_{PERTURB_LAYER}'"
    if out.returncode == 1 and f"ORIGIN: stage #4 {want}" in out.stdout:
        print(
            f"SELFTEST OK — real hooks localized the perturbed layer as origin 004_{PERTURB_LAYER}"
        )
        return 0
    print("SELFTEST FAIL — expected origin 004_" + PERTURB_LAYER)
    return 1


if __name__ == "__main__":
    sys.exit(main())
