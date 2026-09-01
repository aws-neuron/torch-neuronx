#!/usr/bin/env python3
"""Check that no process holds a Neuron device. Gate the upstream runner on it.

The upstream runner is single-tenant. It claims all host cores. Run this gate
first. It confirms the host is idle, so the runner does not collide with an
Implementer that still holds a core.

Exit codes:
  0  FREE     - no process holds a device. The runner can claim the host.
  1  BUSY     - a process still holds a device. Wait, then retry.
  2  INFRA     - neuron-ls failed. Check the driver and the host.
"""

import json
import subprocess
import sys

try:
    out = subprocess.run(["neuron-ls", "-j"], capture_output=True, text=True, timeout=60)
except (OSError, subprocess.TimeoutExpired) as exc:
    print(f"neuron-ls did not run: {exc}", file=sys.stderr)
    sys.exit(2)
if out.returncode != 0:
    print(f"neuron-ls failed rc={out.returncode}: {out.stderr.strip()}", file=sys.stderr)
    sys.exit(2)

devs = json.loads(out.stdout)
busy = [(d.get("neuron_device"), d["neuron_processes"]) for d in devs if d.get("neuron_processes")]
if busy:
    print("BUSY: a process still holds a Neuron device. Wait for it to finish.")
    for dev, procs in busy:
        pids = ",".join(str(p.get("pid")) for p in procs)
        print(f"  device {dev}: pids={pids}")
    sys.exit(1)
print(f"FREE: all {len(devs)} Neuron devices are idle.")
sys.exit(0)
