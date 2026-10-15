"""Compiled-mode capture: torch.compile(backend='neuron') can't hook mid-graph, so the module
RETURNS its intermediates as extra outputs; we dump_returned() them after the call. Compare a
compiled-neuron run (one layer perturbed) vs eager-cpu ref and localize the divergent op."""

import os
import subprocess
import sys
import tempfile

import torch
import torch.nn as nn

sys.path.insert(0, "$SHARED/opbisect")
import torch_neuronx  # noqa
import instrument_dump as ins

STAGES = ["op0_l0", "op1_l1", "op2_l2", "op3_out"]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(64, 64)
        self.l1 = nn.Linear(64, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 64)

    def forward(self, x):
        i0 = torch.relu(self.l0(x))
        i1 = torch.relu(self.l1(i0))
        i2 = torch.relu(self.l2(i1))
        out = self.l3(i2)
        return out, (i0, i1, i2, out)  # <-- intermediates returned as graph outputs


def capture(net, x, d, compiled):
    ins._ORDER.clear()
    ins._SEEN.clear()
    os.environ["OPDUMP_DIR"] = d
    m = torch.compile(net, backend="neuron", fullgraph=True) if compiled else net
    out, inter = m(x)
    ins.dump_returned({STAGES[i]: inter[i] for i in range(len(STAGES))})
    return out


tmp = tempfile.mkdtemp(prefix="compiled_")
ref = tmp + "/cpu"
tgt = tmp + "/neuron"
torch.manual_seed(0)
net_ref = Net().bfloat16()
torch.manual_seed(0)
net_tgt = Net().bfloat16()
x = torch.randn(8, 64).bfloat16()
capture(net_ref, x.clone(), ref, compiled=False)  # eager cpu reference
net_tgt.l2.weight.data += 0.05  # inject divergence at op2 (l2)
capture(net_tgt.to("neuron"), x.clone().to("neuron"), tgt, compiled=True)  # COMPILED on neuron
print("=== BISECT (compiled neuron vs eager cpu, threshold 0.05) ===", flush=True)
r = subprocess.run(
    [
        sys.executable,
        "$SHARED/opbisect/bisect_compare.py",
        "--ref-dir",
        ref,
        "--tgt-dir",
        tgt,
        "--phase",
        "fwd",
        "--threshold",
        "0.05",
    ],
    capture_output=True,
    text=True,
)
print(r.stdout)
print("EXIT", r.returncode, "expect origin op2_l2")
