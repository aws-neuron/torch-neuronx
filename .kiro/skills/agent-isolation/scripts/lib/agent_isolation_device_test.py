"""On-device smoke tests for the agent-isolation kit.

These tests need a Neuron host. They skip when fewer than two devices are
present, so the file is safe to collect anywhere. They cover the claims the
hardware-free tests cannot:
  * agent-run exports the slice, the port window, and the per-slice cache dirs;
  * the agent_core_isolation plugin pins each xdist worker to one distinct core,
    so parallel workers each run a device op without a runtime collision.

Run on a trn host from a checkout with a working torch_neuronx build:
    python -m pytest \\
      .kiro/skills/agent-isolation/scripts/lib/agent_isolation_device_test.py -q
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

AGENT_RUN = Path(__file__).resolve().parents[1] / "agent-run"


def _neuron_devices():
    """Number of Neuron devices, or 0 when neuron-ls is missing or fails."""
    if shutil.which("neuron-ls") is None:
        return 0
    try:
        out = subprocess.run(["neuron-ls", "-j"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return 0
    if out.returncode != 0:
        return 0
    try:
        return len(json.loads(out.stdout))
    except ValueError:
        return 0


pytestmark = pytest.mark.skipif(
    _neuron_devices() < 2, reason="needs a Neuron host with >= 2 devices"
)


def _agent_run(base, size, *cmd, timeout=600):
    return subprocess.run(
        [str(AGENT_RUN), str(base), str(size), "--", *cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_agent_run_exports_slice():
    # agent-run hands the command a slice env, a port window, and per-slice
    # cache dirs derived from BASE.
    r = _agent_run(
        3,
        2,
        "bash",
        "-c",
        "echo CORES=$NEURON_RT_VISIBLE_CORES PORT=$AGENT_PORT_OFFSET "
        "NEFF=$TORCH_NEURONX_NEFF_CACHE_DIR",
    )
    assert r.returncode == 0, r.stderr
    assert "CORES=3-4" in r.stdout
    assert "PORT=1096" in r.stdout  # 1000 + 3*32
    assert "/c3/" in r.stdout  # per-slice cache dir


# Inner test: each xdist worker asserts it owns ONE core and runs a device op.
# Without the plugin both workers keep the whole slice range (start != end).
_INNER_TEST = """
import os
import torch
import torch_neuronx


def test_worker_owns_one_core():
    cores = os.environ["NEURON_RT_VISIBLE_CORES"]
    start, _, end = cores.partition("-")
    assert start == end, f"worker kept a range, not one core: {cores}"
    a = torch.tensor([1.0, 2.0, 3.0], device="neuron")
    torch.testing.assert_close((a + a).cpu(), torch.tensor([2.0, 4.0, 6.0]))
"""


def test_parallel_workers_isolate_cores(tmp_path):
    # Two xdist workers under one agent-run slice must each grab a distinct
    # core and run a device op. `--dist each` runs the test on BOTH workers, so
    # both init the runtime; the plugin pins gw0 -> core 0 and gw1 -> core 1.
    # Without the plugin both keep NEURON_RT_VISIBLE_CORES=0-1 and the inner
    # single-core assertion fails.
    inner = tmp_path / "inner_device_test.py"
    inner.write_text(_INNER_TEST)
    r = _agent_run(
        0,
        2,
        sys.executable,
        "-m",
        "pytest",
        str(inner),
        "-n",
        "2",
        "--dist",
        "each",
        "-p",
        "no:cacheprovider",
        "-q",
    )
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "2 passed" in r.stdout
