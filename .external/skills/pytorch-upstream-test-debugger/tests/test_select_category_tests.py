import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SCRIPT = SCRIPTS / "select_category_tests.py"


def run_selector(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        text=True,
        capture_output=True,
        check=False,
    )


def test_malformed_json_fails_instead_of_passing_silently(tmp_path):
    report = tmp_path / "report.json"
    report.write_text("{not-json")
    result = run_selector(report, "--pin-error", "anything")
    assert result.returncode == 3
    assert "invalid report" in result.stderr


def test_empty_failure_result_does_not_claim_success(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"tests": [{"nodeid": "tests/test_ok.py::test_ok", "outcome": "passed"}]})
    )
    output = tmp_path / "selected.txt"
    result = run_selector(report, "--pin-nodeid", "tests/test_ok.py::test_ok", "--out", output)
    assert result.returncode == 2
    assert not output.exists()
    assert "matched no failing" in result.stderr


def test_json_selection_expands_exact_signature_and_does_not_print_absolute_paths(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "tests": [
                    {
                        "nodeid": "tests/test_ops.py::test_a",
                        "outcome": "failed",
                        "call": {"crash": {"message": "ValueError: shape 128 is invalid"}},
                    },
                    {
                        "nodeid": "tests/test_ops.py::test_b",
                        "outcome": "xfailed",
                        "call": {"crash": {"message": "ValueError: shape 256 is invalid"}},
                    },
                    {
                        "nodeid": "tests/test_ops.py::test_c",
                        "outcome": "failed",
                        "call": {"crash": {"message": "RuntimeError: unrelated"}},
                    },
                ]
            }
        )
    )
    output = tmp_path / "selected.txt"
    result = run_selector(report, "--pin-nodeid", "tests/test_ops.py::test_a", "--out", output)
    assert result.returncode == 0, result.stderr
    assert output.read_text().splitlines() == [
        "tests/test_ops.py::test_a",
        "tests/test_ops.py::test_b",
    ]
    assert "shape <N> is invalid" in output.with_suffix(".txt.signatures.txt").read_text()
    assert str(tmp_path) not in result.stdout + result.stderr


def test_csv_and_log_parsers_are_supported(tmp_path):
    csv_report = tmp_path / "report.csv"
    csv_report.write_text("nodeid,outcome,message\ntests/test_a.py::test_a,failed,boom\n")
    csv_result = run_selector(csv_report, "--pin-error", "boom")
    assert csv_result.returncode == 0
    assert "tests/test_a.py::test_a" in csv_result.stdout

    log_report = tmp_path / "report.log"
    log_report.write_text("FAILED tests/test_b.py::test_b - RuntimeError: boom\n")
    log_result = run_selector(log_report, "--pin-nodeid", "tests/test_b.py::test_b")
    assert log_result.returncode == 0
    assert "tests/test_b.py::test_b" in log_result.stdout


def test_help_exits_zero():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()


def test_truly_malformed_json_fixture_exits_three_without_traceback(tmp_path):
    report = tmp_path / "broken.json"
    report.write_text('{"tests": [}')
    result = run_selector(report, "--pin-error", "anything")
    assert result.returncode == 3
    assert "Traceback" not in result.stderr


def test_worker_crash_stage_diagnostics_form_distinct_signatures(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "tests": [
                    {
                        "nodeid": "tests/test_ops.py::test_worker_a",
                        "outcome": "failed",
                        "???": {"longrepr": "worker exited: device runtime crash"},
                    },
                    {
                        "nodeid": "tests/test_ops.py::test_worker_b",
                        "outcome": "failed",
                        "???": {"longrepr": "worker exited: host allocation crash"},
                    },
                ]
            }
        )
    )
    result = run_selector(report, "--pin-error", "device runtime")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == "tests/test_ops.py::test_worker_a"
    assert "test_worker_b" not in result.stdout


def test_real_short_summary_line_preserves_parameter_and_message_separators(tmp_path):
    nodeid = "tests/test_ops.py::TestOps::test_add[dtype float32 - layout strided]"
    report = tmp_path / "report.log"
    report.write_text(f"FAILED {nodeid} - RuntimeError: left - right\n")
    result = run_selector(report, "--pin-error", r"left - right")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == nodeid


def test_report_record_limit_is_enforced(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "tests": [
                    {"nodeid": f"tests/test_many.py::test_{index}", "outcome": "failed"}
                    for index in range(5001)
                ]
            }
        )
    )
    result = run_selector(report, "--pin-error", "anything")
    assert result.returncode == 3
    assert "more than 5000" in result.stderr


def test_pair_write_rolls_back_both_outputs_on_second_install_failure(tmp_path, monkeypatch):
    sys.path.insert(0, str(SCRIPTS))
    try:
        import select_category_tests as selector

        first = tmp_path / "selected.txt"
        second = tmp_path / "selected.txt.signatures.txt"
        first.write_text("old first\n")
        second.write_text("old second\n")
        monkeypatch.setenv("SAFE_OUTPUT_TEST_FAIL_INSTALL", "2")

        with pytest.raises(OSError, match="injected late"):
            selector._write_pair(first, "new first\n", second, "new second\n")
        assert first.read_text() == "old first\n"
        assert second.read_text() == "old second\n"
        assert sorted(path.name for path in tmp_path.iterdir()) == [
            "selected.txt",
            "selected.txt.signatures.txt",
        ]
    finally:
        sys.path.remove(str(SCRIPTS))


@pytest.mark.parametrize("destination_name", ["selected.txt", "selected.txt.signatures.txt"])
@pytest.mark.parametrize("kind", ["directory", "symlink", "fifo"])
def test_pair_preflights_both_nonregular_destinations_without_mutation(
    tmp_path, destination_name, kind
):
    report = tmp_path / "report.json"
    nodeid = "tests/test_ops.py::test_a"
    report.write_text(json.dumps({"tests": [{"nodeid": nodeid, "outcome": "failed"}]}))
    output = tmp_path / "selected.txt"
    destination = tmp_path / destination_name
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("keep\n")

    if kind == "directory":
        destination.mkdir()
    elif kind == "symlink":
        destination.symlink_to(sentinel)
    else:
        os.mkfifo(destination)

    result = run_selector(report, "--pin-nodeid", nodeid, "--out", output)
    assert result.returncode == 4
    assert "cannot write selection output" in result.stderr
    assert sentinel.read_text() == "keep\n"
    if kind == "directory":
        assert destination.is_dir()
    elif kind == "symlink":
        assert destination.is_symlink()
    else:
        assert stat.S_ISFIFO(destination.lstat().st_mode)

    other = tmp_path / "selected.txt.signatures.txt" if destination == output else output
    assert not other.exists()
    assert not other.is_symlink()


def test_pair_rejects_symlink_output_directory_without_writing_through_it(tmp_path):
    report = tmp_path / "report.json"
    nodeid = "tests/test_ops.py::test_a"
    report.write_text(json.dumps({"tests": [{"nodeid": nodeid, "outcome": "failed"}]}))
    outside = tmp_path / "outside"
    outside.mkdir()
    output_dir = tmp_path / "output"
    output_dir.symlink_to(outside, target_is_directory=True)

    result = run_selector(
        report,
        "--pin-nodeid",
        nodeid,
        "--out",
        output_dir / "selected.txt",
    )
    assert result.returncode == 4
    assert "output directory must not be a symlink" in result.stderr
    assert list(outside.iterdir()) == []


def test_signature_sidecar_redacts_sensitive_message_values(tmp_path):
    home = "/" + "home/person/project.py"
    account = "123456" * 2
    key_id = "AKIA" + "B" * 16
    import socket

    hostname = socket.gethostname()
    message = f"failure at {home} account {account} key {key_id} token=hidden host {hostname}"
    nodeid = "tests/test_ops.py::test_sensitive"
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "tests": [
                    {
                        "nodeid": nodeid,
                        "outcome": "failed",
                        "call": {"crash": {"message": message}},
                    }
                ]
            }
        )
    )
    output = tmp_path / "selected.txt"
    result = run_selector(report, "--pin-nodeid", nodeid, "--out", output)
    assert result.returncode == 0, result.stderr
    sidecar = output.with_suffix(".txt.signatures.txt").read_text()
    for sensitive in (home, account, key_id, hostname, "hidden"):
        assert sensitive not in sidecar
    assert "<redacted-home>" in sidecar
    assert "<redacted-account>" in sidecar
    assert "<redacted-key-id>" in sidecar
    assert "<redacted-secret>" in sidecar
    assert "<redacted-hostname>" in sidecar


@pytest.mark.parametrize("report_format", ["json", "csv", "log"])
def test_duplicate_report_nodeids_are_rejected(tmp_path, report_format):
    nodeid = "tests/test_duplicate.py::test_duplicate"
    report = tmp_path / f"report.{report_format}"
    if report_format == "json":
        report.write_text(
            json.dumps(
                {
                    "tests": [
                        {"nodeid": nodeid, "outcome": "failed"},
                        {"nodeid": nodeid, "outcome": "failed"},
                    ]
                }
            )
        )
    elif report_format == "csv":
        report.write_text(
            f"nodeid,outcome,message\n{nodeid},failed,first\n{nodeid},failed,second\n"
        )
    else:
        report.write_text(f"FAILED {nodeid} - first\nFAILED {nodeid} - second\n")

    result = run_selector(report, "--pin-nodeid", nodeid)
    assert result.returncode == 3
    assert "duplicate report node ID" in result.stderr
