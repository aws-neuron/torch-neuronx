"""Hardware-free tests for agent-run: slice, port, cache, and exit-code handling."""

import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "agent-run"
DUMP = "import json, os; print(json.dumps({k: os.environ.get(k) for k in %r}))"
KEYS = [
    "NEURON_RT_VISIBLE_CORES",
    "PYTEST_XDIST_WORKER",
    "AGENT_PORT_OFFSET",
    "TORCH_NEURONX_HLO_CACHE_DIR",
    "TORCH_NEURONX_NEFF_CACHE_DIR",
    "TMPDIR",
    "PYTEST_PLUGINS",
    "PYTHONPATH",
]


def run(*args, env_root):
    env = dict(os.environ, AGENT_CACHE_ROOT=str(env_root))
    env.pop("PYTEST_PLUGINS", None)
    return subprocess.run([str(SCRIPT), *args], capture_output=True, text=True, env=env, timeout=60)


def test_help_exits_zero():
    result = subprocess.run([str(SCRIPT), "--help"], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0
    assert "BASE_CORE NUM_CORES" in result.stdout


def test_missing_arguments_exit_two(tmp_path):
    assert run("4", env_root=tmp_path).returncode == 2
    assert run("4", "4", "--", env_root=tmp_path).returncode == 2


def test_exports_the_slice_ports_and_caches(tmp_path):
    import json

    result = run("8", "4", "--", sys.executable, "-c", DUMP % KEYS, env_root=tmp_path)
    assert result.returncode == 0, result.stderr
    env = json.loads(result.stdout)
    assert env["NEURON_RT_VISIBLE_CORES"] == "8-11"
    assert env["PYTEST_XDIST_WORKER"] == "gw8"
    assert env["AGENT_PORT_OFFSET"] == str(1000 + 8 * 32)
    root = tmp_path / "c8"
    assert env["TORCH_NEURONX_HLO_CACHE_DIR"] == str(root / "hlo")
    assert env["TORCH_NEURONX_NEFF_CACHE_DIR"] == str(root / "neff")
    assert env["TMPDIR"] == str(root / "tmp")
    assert (root / "tmp").is_dir()
    assert env["PYTEST_PLUGINS"] == "agent_port_isolation,agent_core_isolation"
    assert env["PYTHONPATH"].split(":")[0] == str(SCRIPT.parent)


def test_slices_get_disjoint_port_windows(tmp_path):
    offsets = []
    for base in ("0", "4"):
        result = run(base, "4", "--", "sh", "-c", "echo $AGENT_PORT_OFFSET", env_root=tmp_path)
        offsets.append(int(result.stdout))
    assert offsets[1] - offsets[0] == 4 * 32


def test_passes_through_the_command_exit_code(tmp_path):
    assert run("0", "4", "--", "sh", "-c", "exit 7", env_root=tmp_path).returncode == 7
