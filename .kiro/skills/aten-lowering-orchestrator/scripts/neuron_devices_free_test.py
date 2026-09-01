"""Hardware-free tests for the neuron_devices_free.py device gate.

The script runs `neuron-ls -j` at import, so it is not importable. Test it
through a subprocess with a fake `neuron-ls` on PATH. No Neuron device is
needed. Exit codes: 0 FREE, 1 BUSY, 2 INFRA.
"""

import os
import stat
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).with_name("neuron_devices_free.py")


def _fake_neuron_ls(tmp_path, body):
    """Write an executable fake `neuron-ls` and return a PATH with it first."""
    p = tmp_path / "neuron-ls"
    p.write_text(body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}"


def _run(path):
    env = dict(os.environ, PATH=path)
    return subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, env=env)


def test_free_exits_0(tmp_path):
    body = (
        '#!/bin/sh\necho \'[{"neuron_device":0,"neuron_processes":[]},'
        '{"neuron_device":1,"neuron_processes":[]}]\'\n'
    )
    r = _run(_fake_neuron_ls(tmp_path, body))
    assert r.returncode == 0, r.stderr
    assert "FREE" in r.stdout


def test_busy_exits_1(tmp_path):
    body = (
        '#!/bin/sh\necho \'[{"neuron_device":0,"neuron_processes":[{"pid":123}]},'
        '{"neuron_device":1,"neuron_processes":[]}]\'\n'
    )
    r = _run(_fake_neuron_ls(tmp_path, body))
    assert r.returncode == 1, r.stderr
    assert "BUSY" in r.stdout
    assert "123" in r.stdout


def test_neuron_ls_error_exits_2(tmp_path):
    body = "#!/bin/sh\necho boom >&2\nexit 3\n"
    r = _run(_fake_neuron_ls(tmp_path, body))
    assert r.returncode == 2, r.stdout


def test_neuron_ls_missing_exits_2(tmp_path):
    # An empty PATH makes the `neuron-ls` lookup fail (OSError) -> INFRA.
    r = _run(str(tmp_path))
    assert r.returncode == 2, r.stdout
