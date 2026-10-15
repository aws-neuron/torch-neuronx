import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "report_signatures.py"
SPEC = importlib.util.spec_from_file_location("report_signatures", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_call_stage_wins_over_setup():
    test = {
        "setup": {"longrepr": "setup text"},
        "call": {"crash": {"message": "call message"}, "longrepr": "call trace"},
    }
    assert MODULE.extract_failure_fields(test) == ("call message", "call trace", "call")


def test_worker_crash_stage_is_used_when_call_is_empty():
    test = {"call": {}, "???": {"longrepr": "worker 'gw1' crashed"}}
    message, longrepr, stage = MODULE.extract_failure_fields(test)
    assert (message, stage) == ("", "???")
    assert "gw1" in longrepr


def test_distinct_worker_crashes_give_distinct_text():
    first = {"???": {"longrepr": "worker 'gw0' crashed: signal 11"}}
    second = {"???": {"longrepr": "worker 'gw0' crashed: signal 6"}}
    assert MODULE.extract_failure_text(first) != MODULE.extract_failure_text(second)


def test_no_failure_text_returns_empty_fields():
    assert MODULE.extract_failure_fields({"call": {"outcome": "passed"}}) == ("", "", "")


@pytest.mark.parametrize(
    "test",
    [{"call": "not an object"}, {"call": {"crash": "bad"}}, {"call": {"longrepr": 3}}],
)
def test_malformed_phase_raises(test):
    with pytest.raises(ValueError):
        MODULE.extract_failure_fields(test)


@pytest.mark.parametrize(
    "nodeid",
    [
        "",
        "-k",
        "a\nb",
        "x" * (MODULE.MAX_NODEID_CHARS + 1),
        3,
        "/tmp/x.py::test_x",
        "../x.py::test_x",
        r"a\b.py::test_x",
        "a//b.py::test_x",
        "a/./b.py::test_x",
        "a/../b.py::test_x",
        "a.txt::test_x",
    ],
)
def test_invalid_nodeid_raises(nodeid):
    with pytest.raises(ValueError):
        MODULE.validate_nodeid(nodeid)


def test_valid_nodeid_with_parameters_is_kept():
    nodeid = r"test/test_ops.py::TestX::test_y[a::b c\d/../e]"
    assert MODULE.validate_nodeid(nodeid) == nodeid


def test_tests_array_validation():
    assert MODULE.validate_tests_array({"tests": [{"nodeid": "a"}]}) == [{"nodeid": "a"}]
    with pytest.raises(ValueError):
        MODULE.validate_tests_array([])
    with pytest.raises(ValueError):
        MODULE.validate_tests_array({"tests": ["x"]})
    with pytest.raises(ValueError):
        MODULE.validate_tests_array({"tests": [{}] * (MODULE.MAX_RECORDS + 1)})


def test_help_exits_zero():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0
