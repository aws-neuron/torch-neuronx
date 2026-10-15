"""Validate op-bisect NKI capture mode with REAL device prints (no simulator):
capture two device_print tiles (ref a+b, target a+b_perturbed) via wrap_nki +
NEURON_RT_DEBUG_OUTPUT_DIR, load them with a real-format loader, and localize via bisect_compare.
Note: the nrtpy DebugHandler increments the sample index per device_print call across the
process, so we select the LATEST index dir rather than assuming 0."""

import glob
import json
import os
import subprocess
import sys
import tempfile

DBG = "/tmp/dp_full_out"
os.environ["NEURON_RT_DEBUG_OUTPUT_DIR"] = DBG
os.system(f"rm -rf {DBG}; mkdir -p {DBG}")
import nki  # noqa: E402
import nki.language as nl  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from torch_neuronx import wrap_nki  # noqa: E402


def load_device_print(base, tag, core=0, idx="latest"):
    """Load an nl.device_print dump -> torch.Tensor. Files: metadata.txt + tensor_data.bin."""
    cdir = os.path.join(base, tag, f"core_{core}")
    if idx == "latest":
        idxs = sorted(
            int(os.path.basename(p))
            for p in glob.glob(cdir + "/*")
            if os.path.basename(p).isdigit()
        )
        idx = idxs[-1]
    d = os.path.join(cdir, str(idx))
    with open(os.path.join(d, "metadata.txt")) as _mf:
        meta = {k.strip(): v.strip() for k, v in (ln.split("=", 1) for ln in _mf if "=" in ln)}
    npdt = {
        "float32": np.float32,
        "bfloat16": np.float32,
        "float16": np.float16,
        "int32": np.int32,
    }[meta["dtype"]]
    with open(os.path.join(d, "tensor_data.bin"), "rb") as _bf:
        raw = _bf.read()
    return torch.from_numpy(np.frombuffer(raw, dtype=npdt).reshape(eval(meta["shape"])).copy())


@nki.jit
def add_dp(a, b):
    sa = nl.load(a)
    sb = nl.load(b)
    sc = nl.add(sa, sb)
    nl.device_print("moe_stage", sc)
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    nl.store(out, value=sc)
    return out


wrapped = wrap_nki(add_dp)
a = torch.ones(16, 128, dtype=torch.float32).to("neuron")

wrapped(a, (torch.ones(16, 128) * 2).to("neuron")).to("cpu")  # REF: 1+2=3.0 -> idx 0
ref_tile = load_device_print(DBG, "moe_stage")
print("REF tile:", tuple(ref_tile.shape), "mean", float(ref_tile.mean()))

wrapped(a, (torch.ones(16, 128) * 2.5).to("neuron")).to("cpu")  # TARGET: 1+2.5=3.5 -> idx 1
tgt_tile = load_device_print(DBG, "moe_stage")
print("TGT tile:", tuple(tgt_tile.shape), "mean", float(tgt_tile.mean()))

assert (
    abs(float(ref_tile.mean()) - 3.0) < 1e-3 and abs(float(tgt_tile.mean()) - 3.5) < 1e-3
), "loader mismatch"
print("LOADER OK — real device_print tiles captured+loaded (3.0 / 3.5)")

tmp = tempfile.mkdtemp()
r = tmp + "/ref"
t = tmp + "/tgt"
os.makedirs(r)
os.makedirs(t)
torch.save(ref_tile, f"{r}/fwd-000_moe_stage__value.pt")
torch.save(tgt_tile, f"{t}/fwd-000_moe_stage__value.pt")
with open(f"{r}/ops_order.json", "w") as _of:
    json.dump(["000_moe_stage"], _of)
o = subprocess.run(
    [
        sys.executable,
        "$SHARED/opbisect/bisect_compare.py",
        "--ref-dir",
        r,
        "--tgt-dir",
        t,
        "--phase",
        "fwd",
        "--threshold",
        "0.05",
    ],
    capture_output=True,
    text=True,
)
print(o.stdout)
print("NKI->BISECT VALIDATED:", "YES" if "000_moe_stage' (FAIL" in o.stdout else "NO")
