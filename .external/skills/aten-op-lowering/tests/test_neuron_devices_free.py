import importlib.util
import json
import math
import stat
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "neuron_devices_free.py"
SPEC = importlib.util.spec_from_file_location("neuron_devices_free", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def executable(tmp_path, value):
    path = tmp_path / "device-query"
    body = json.dumps(value)
    path.write_text(f"#!/bin/sh\nprintf '%s\\n' '{body}'\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(path)


def test_free_devices_return_zero_with_count(tmp_path):
    command = executable(tmp_path, [{"neuron_device": 0, "neuron_processes": []}])
    code, message = MODULE.inspect_devices(command, 2)
    assert code == 0
    assert message == "All 1 Neuron devices are free."


def test_busy_device_returns_one_with_pid(tmp_path):
    command = executable(tmp_path, [{"neuron_device": 0, "neuron_processes": [{"pid": 7}]}])
    code, message = MODULE.inspect_devices(command, 2)
    assert code == 1
    assert "device 0: pids=7" in message


@pytest.mark.parametrize(
    ("value", "evidence"),
    [
        ([], "No Neuron devices found"),
        ({}, "root must be a list"),
        (None, "root must be a list"),
        ([None], "Device 0 must be an object"),
        ([{}], "neuron_processes must be a list"),
        ([{"neuron_processes": {}}], "neuron_processes must be a list"),
        ([{"neuron_processes": [None]}], "process 0 must be an object"),
    ],
)
def test_malformed_device_data_returns_two(tmp_path, value, evidence):
    command = executable(tmp_path, value)
    code, message = MODULE.inspect_devices(command, 2)
    assert code == 2
    assert evidence in message


@pytest.mark.parametrize("timeout", [0, -1, math.nan, math.inf, -math.inf])
def test_nonpositive_or_nonfinite_timeout_is_error(timeout):
    code, message = MODULE.inspect_devices("unused", timeout)
    assert code == 2
    assert "finite and greater than zero" in message


def test_query_failure_returns_two(tmp_path):
    path = tmp_path / "query"
    path.write_text("#!/bin/sh\necho failed >&2\nexit 3\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    code, message = MODULE.inspect_devices(str(path), 2)
    assert code == 2
    assert "code 3" in message


def test_invalid_json_returns_two(tmp_path):
    path = tmp_path / "query"
    path.write_text("#!/bin/sh\necho invalid\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    code, message = MODULE.inspect_devices(str(path), 2)
    assert code == 2
    assert "invalid data" in message


def test_cli_help(capsys):
    script = Path(__file__).resolve().parents[1] / "scripts" / "neuron_devices_free.py"
    assert script == SCRIPT
    with pytest.raises(SystemExit) as result:
        MODULE.main(["--help"])
    assert result.value.code == 0
    assert "Check whether" in capsys.readouterr().out


def test_neuron_devices_free_script_help_subprocess():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout
