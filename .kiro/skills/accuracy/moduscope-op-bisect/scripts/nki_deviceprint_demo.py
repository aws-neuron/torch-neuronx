#!/usr/bin/env python3
"""nki_deviceprint_demo.py — proof that nl.device_print emits per-op tile values from a NKI kernel.

Runs a trivial add kernel via nki.simulate_kernel (CPU simulator — no Neuron device, no torch_xla)
and shows device_print dumping the intermediate tile. This is the NKI capture mode of op-bisect:
add nl.device_print("tag", tile) inside the kernel and read the values.

IMPORTANT — two different sinks:
  * SIMULATOR (nki.simulate_kernel, this demo): device_print goes to STDOUT.
  * REAL NEURON DEVICE: the runtime ABSORBS device_print and writes it to FILES at
    $NEURON_RT_DEBUG_OUTPUT_DIR/<tag>/core_0/<i>/tensor.pt (torch.load them) — NOT stdout.
    Standalone @nki.jit / @nki_op won't run in the eager accuracy venv (missing
    torch_neuronx.pyhlo and/or fake impl); dump from inside a framework-invoked kernel.
    See SKILL.md "NKI kernel internals".

Requires the neuronxcc/nki venv (e.g. $SHARED/<neuron-env>/private-torch-neuronx/.venv);
does NOT run in a plain-torch venv.

Verified simulator output includes:  NKI_DUMP c_tile [[3. 3. ...]]   and   out[0,0]=3.0
"""

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import numpy as np


def add_dp(a_in, b_in, out):
    a = nl.load(a_in)
    b = nl.load(b_in)
    c = a + b
    nl.device_print("NKI_DUMP c_tile", c)  # <-- on-device / simulated per-op dump
    nl.store(out, value=c)


if __name__ == "__main__":
    a = np.ones((16, 128), dtype=np.float32)
    b = np.full((16, 128), 2.0, dtype=np.float32)
    out = np.zeros((16, 128), dtype=np.float32)
    nki.simulate_kernel(add_dp, a, b, out)  # runs the kernel logic + fires device_print
    assert abs(float(out[0, 0]) - 3.0) < 1e-5, out[0, 0]
    print("NKI DEVICE_PRINT DEMO OK — device_print emitted c_tile; out[0,0]=", float(out[0, 0]))
