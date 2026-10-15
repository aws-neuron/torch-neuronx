import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SCRIPT = SCRIPTS / "rerun_category.sh"


def fake_environment(tmp_path, report, pytest_exit=0, missing_module="", log_text=""):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python = bin_dir / "python"
    fake_python.write_text(
        """#!/usr/bin/env python3
import json, os, pathlib, sys
if sys.argv[1:2] == ['-c']:
    missing = os.environ.get('FAKE_MISSING_MODULE', '')
    sys.exit(1 if missing and f'import {missing}' in sys.argv[2] else 0)
if sys.argv[1:3] == ['-m', 'pytest']:
    report_arg = next(a for a in sys.argv if a.startswith('--json-report-file='))
    pathlib.Path(report_arg.split('=', 1)[1]).write_text(os.environ['FAKE_REPORT'])
    pathlib.Path(os.environ['FAKE_ARGS']).write_text(json.dumps(sys.argv[3:]))
    pathlib.Path(os.environ['FAKE_PRELOAD']).write_text(os.environ.get('LD_PRELOAD', ''))
    print(os.environ.get('FAKE_LOG_TEXT', 'synthetic pytest execution'))
    sys.exit(int(os.environ.get('FAKE_PYTEST_EXIT', '0')))
if sys.argv[1:2] == ['-']:
    os.execv(os.environ['REAL_PYTHON'], [os.environ['REAL_PYTHON'], *sys.argv[1:]])
sys.exit(99)
"""
    )
    fake_python.chmod(0o755)
    args_file = tmp_path / "pytest-args.json"
    preload_file = tmp_path / "pytest-preload.txt"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "REAL_PYTHON": sys.executable,
            "FAKE_REPORT": json.dumps(report),
            "FAKE_PYTEST_EXIT": str(pytest_exit),
            "FAKE_ARGS": str(args_file),
            "FAKE_PRELOAD": str(preload_file),
            "FAKE_MISSING_MODULE": missing_module,
            "FAKE_LOG_TEXT": log_text,
        }
    )
    return env, args_file, preload_file


def invoke(
    tmp_path,
    report,
    pytest_exit=0,
    extra=(),
    tests_text=None,
    stale_report=None,
    missing_module="",
    configured_cc="",
    output_kind="directory",
    destination_name="",
    destination_kind="symlink",
    repo_file_symlink=False,
    swap_component="",
    fail_mkdir=False,
    log_text="",
    temp_root=None,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    runner = repo / "tests" / "pytorch_tests_native" / "runner"
    runner.mkdir(parents=True)
    (runner / "env_lock.c").write_text("void env_lock_placeholder(void) {}\n")
    if repo_file_symlink:
        outside_test = tmp_path / "outside.py"
        outside_test.write_text("def test_failure(): pass\n")
        (repo / "tests" / "test_synthetic.py").symlink_to(outside_test)
    tests_file = tmp_path / "tests.txt"
    tests_file.write_text(tests_text or "tests/test_synthetic.py::test_failure\n")
    output_parent = tmp_path
    if swap_component == "ancestor":
        output_parent = tmp_path / "output-parent"
        output_parent.mkdir()
    output = output_parent / "output"
    if output_kind == "symlink":
        outside_output = tmp_path / "outside-output"
        outside_output.mkdir()
        output.symlink_to(outside_output, target_is_directory=True)
    elif output_kind == "regular":
        output.write_text("keep\n")
    elif output_kind == "fifo":
        os.mkfifo(output)
    else:
        output.mkdir()
    if stale_report is not None:
        (output / "report.json").write_text(json.dumps(stale_report))
    if destination_name:
        destination = output / destination_name
        if destination_kind == "symlink":
            sentinel = tmp_path / "sentinel.txt"
            sentinel.write_text("keep\n")
            destination.symlink_to(sentinel)
        elif destination_kind == "directory":
            destination.mkdir()
        else:
            os.mkfifo(destination)
    env, args_file, preload_file = fake_environment(
        tmp_path,
        report,
        pytest_exit,
        missing_module=missing_module,
        log_text=log_text,
    )
    if swap_component:
        outside = tmp_path / "outside-output"
        outside.mkdir()
        swap_path = output_parent if swap_component == "ancestor" else output
        env["SAFE_OUTPUT_TEST_SWAP_ANCESTOR"] = str(swap_path)
        env["SAFE_OUTPUT_TEST_SWAP_TARGET"] = str(outside)
    if temp_root is not None:
        env["TMPDIR"] = str(temp_root)
    if fail_mkdir:
        env["FAKE_MKDIR_PARENT"] = str(tmp_path / "mkdir-parent.txt")
        fake_mkdir = Path(env["PATH"].split(":", maxsplit=1)[0]) / "mkdir"
        fake_mkdir.write_text(
            '#!/bin/sh\n[ "$1" = "--" ] && shift\n'
            'printf \'%s\n\' "${1%/*}" > "$FAKE_MKDIR_PARENT"\n'
            "exit 1\n"
        )
        fake_mkdir.chmod(0o755)
    if configured_cc:
        env["CC"] = configured_cc
    result = subprocess.run(
        ["bash", str(SCRIPT), str(repo), str(tests_file), str(output), *extra],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    return result, output, args_file, preload_file


def failed_report(message="AssertionError: synthetic failure"):
    return {
        "tests": [
            {
                "nodeid": "tests/test_synthetic.py::test_failure",
                "outcome": "failed",
                "call": {"crash": {"message": message}},
            }
        ]
    }


def test_failing_pytest_status_is_propagated_after_summary(tmp_path):
    result, output, _, _ = invoke(tmp_path, failed_report(), pytest_exit=1)
    assert result.returncode == 1
    assert "# pytest exit=1" in result.stderr
    assert "failed" in (output / "summary.txt").read_text()


def test_malformed_report_fails_even_when_pytest_returns_zero(tmp_path):
    result, output, _, _ = invoke(tmp_path, "not a report", pytest_exit=0)
    assert result.returncode == 11
    assert not (output / "summary.txt").exists()
    assert "cannot validate pytest JSON report" in result.stderr


def test_expected_flags_nodeid_and_worker_clamp_reach_pytest(tmp_path):
    report = {"tests": [{"nodeid": "tests/test_synthetic.py::test_failure", "outcome": "passed"}]}
    result, _, args_file, preload_file = invoke(
        tmp_path, report, extra=("-n", "8", "--timeout", "17")
    )
    assert result.returncode == 0, result.stderr
    args = json.loads(args_file.read_text())
    assert "--runxfail" in args
    assert "--timeout=17" in args
    assert "-n" in args and args[args.index("-n") + 1] == "1"
    separator = args.index("--")
    assert args[separator + 1 :] == ["tests/test_synthetic.py::test_failure"]
    assert "env_lock.so" in preload_file.read_text()


def test_help_exits_zero_without_device_environment():
    result = subprocess.run(
        ["bash", str(SCRIPT), "--help"], text=True, capture_output=True, check=False
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout


def test_stale_report_cannot_make_pytest_exit_one_look_fixed(tmp_path):
    fresh = {"tests": [{"nodeid": "tests/test_synthetic.py::test_failure", "outcome": "passed"}]}
    stale = {"tests": [{"nodeid": "tests/test_synthetic.py::test_failure", "outcome": "passed"}]}
    result, output, _, _ = invoke(
        tmp_path,
        fresh,
        pytest_exit=1,
        stale_report=stale,
    )
    assert result.returncode == 11
    assert "fresh report contains no failures" in result.stderr
    assert json.loads((output / "report.json").read_text()) == stale
    assert not (output / "summary.txt").exists()


def test_report_must_contain_exact_requested_node_set(tmp_path):
    report = {"tests": [{"nodeid": "tests/test_other.py::test_other", "outcome": "passed"}]}
    result, output, _, _ = invoke(tmp_path, report)
    assert result.returncode == 11
    assert "set differs" in result.stderr
    assert not (output / "report.json").exists()


def test_unsafe_tests_file_lines_exit_two_without_traceback(tmp_path):
    report = {"tests": [{"nodeid": "tests/test_synthetic.py::test_failure", "outcome": "passed"}]}
    unsafe_lines = (
        "-k expression\n",
        "tests/test_synthetic.py::test_failure\n\nother.py::test_other\n",
        "tests/test_synthetic.py::test_\x01bad\n",
        "not-a-node-id\n",
        "/tmp/x.py::test_x\n",
        "../x.py::test_x\n",
        "a\\b.py::test_x\n",
    )
    for index, tests_text in enumerate(unsafe_lines):
        case_dir = tmp_path / str(index)
        case_dir.mkdir()
        result, output, _, _ = invoke(case_dir, report, tests_text=tests_text)
        assert result.returncode == 2, (tests_text, result.stderr)
        assert "Traceback" not in result.stderr
        assert not (output / "report.json").exists()


def test_dependency_preflight_names_missing_package_and_exits_ten(tmp_path):
    report = {"tests": [{"nodeid": "tests/test_synthetic.py::test_failure", "outcome": "passed"}]}
    for module, package, extra in (
        ("pytest", "pytest", ()),
        ("pytest_timeout", "pytest-timeout", ()),
        ("xdist", "pytest-xdist", ("-n", "2")),
        (
            "pytest_forked",
            "pytest-forked",
            (),
        ),
    ):
        case_dir = tmp_path / module
        case_dir.mkdir()
        tests_text = None
        if module == "pytest_forked":
            nodeid = "tests/distributed/test_synthetic.py::test_failure"
            tests_text = nodeid + "\n"
            report = {"tests": [{"nodeid": nodeid, "outcome": "passed"}]}
        result, _, _, _ = invoke(
            case_dir,
            report,
            extra=extra,
            tests_text=tests_text,
            missing_module=module,
        )
        assert result.returncode == 10
        assert package in result.stderr
        assert "Traceback" not in result.stderr


def test_distributed_target_uses_forked_without_timeout(tmp_path):
    nodeid = "tests/pytorch_tests_native/test/distributed/test_c10d.py::test_rank"
    report = {"tests": [{"nodeid": nodeid, "outcome": "passed"}]}
    result, _, args_file, _ = invoke(
        tmp_path,
        report,
        tests_text=nodeid + "\n",
        extra=("--timeout", "17"),
    )
    assert result.returncode == 0, result.stderr
    args = json.loads(args_file.read_text())
    assert "--forked" in args
    assert not any(argument.startswith("--timeout=") for argument in args)


def test_nodeid_length_limit_exits_two(tmp_path):
    nodeid = "tests/test_synthetic.py::test_" + "x" * 1000
    report = {"tests": [{"nodeid": nodeid, "outcome": "passed"}]}
    result, _, _, _ = invoke(tmp_path, report, tests_text=nodeid + "\n")
    assert result.returncode == 2
    assert "exceeds 1000" in result.stderr
    assert "Traceback" not in result.stderr


def test_pytest_exit_zero_with_failure_report_is_rejected(tmp_path):
    result, output, _, _ = invoke(tmp_path, failed_report(), pytest_exit=0)
    assert result.returncode == 11
    assert "pytest exited 0 but the fresh report contains failures" in result.stderr
    assert not (output / "report.json").exists()
    assert not (output / "pytest.log").exists()
    assert not (output / "summary.txt").exists()


@pytest.mark.parametrize("kind", ["regular", "symlink", "fifo"])
def test_non_directory_output_path_is_rejected_without_mutation(tmp_path, kind):
    report = {"tests": [{"nodeid": "tests/test_synthetic.py::test_failure", "outcome": "passed"}]}
    result, output, _, _ = invoke(tmp_path, report, output_kind=kind)
    assert result.returncode == 11
    if kind == "regular":
        assert output.read_text() == "keep\n"
    elif kind == "symlink":
        assert "output directory must not be a symlink" in result.stderr
        assert list(output.iterdir()) == []
    else:
        assert stat.S_ISFIFO(output.lstat().st_mode)


@pytest.mark.parametrize("destination_name", ["pytest.log", "report.json", "summary.txt"])
@pytest.mark.parametrize("kind", ["directory", "symlink", "fifo"])
def test_nonregular_final_output_is_rejected_without_mutation(tmp_path, destination_name, kind):
    report = {"tests": [{"nodeid": "tests/test_synthetic.py::test_failure", "outcome": "passed"}]}
    result, output, _, _ = invoke(
        tmp_path,
        report,
        destination_name=destination_name,
        destination_kind=kind,
    )
    destination = output / destination_name
    assert result.returncode == 11
    if kind == "directory":
        assert destination.is_dir()
    elif kind == "symlink":
        assert (tmp_path / "sentinel.txt").read_text() == "keep\n"
        assert destination.is_symlink()
    else:
        assert stat.S_ISFIFO(destination.lstat().st_mode)
    assert {path.name for path in output.iterdir()} == {destination_name}


def test_missing_configured_c_compiler_exits_ten(tmp_path):
    report = {"tests": [{"nodeid": "tests/test_synthetic.py::test_failure", "outcome": "passed"}]}
    result, _, _, _ = invoke(tmp_path, report, configured_cc="missing-compiler")
    assert result.returncode == 10
    assert "C compiler" in result.stderr


def test_node_file_must_resolve_under_repository_root(tmp_path):
    report = {"tests": [{"nodeid": "tests/test_synthetic.py::test_failure", "outcome": "passed"}]}
    result, output, args_file, _ = invoke(tmp_path, report, repo_file_symlink=True)
    assert result.returncode == 2
    assert "resolves outside the repository" in result.stderr
    assert not args_file.exists()
    assert not (output / "report.json").exists()


@pytest.mark.parametrize("argc", range(3))
def test_missing_positional_arguments_print_stable_usage_and_exit_two(argc):
    arguments = ["repo", "tests.txt"][:argc]
    result = subprocess.run(
        ["bash", str(SCRIPT), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith("Usage: rerun_category.sh ")
    assert "unbound variable" not in result.stderr


def test_summary_redacts_sensitive_failure_message(tmp_path):
    home = "/" + "Users/person/project.py"
    account = "123456" * 2
    key_id = "ASIA" + "D" * 16
    import socket

    hostname = socket.gethostname()
    message = f"at {home} account={account} key={key_id} password=hidden host={hostname}"
    result, output, _, _ = invoke(tmp_path, failed_report(message), pytest_exit=1)
    assert result.returncode == 1, result.stderr
    summary = (output / "summary.txt").read_text()
    for sensitive in (home, account, key_id, hostname, "hidden"):
        assert sensitive not in summary
    assert "<redacted-home>" in summary
    assert "<redacted-account>" in summary
    assert "<redacted-key-id>" in summary
    assert "<redacted-secret>" in summary
    assert "<redacted-hostname>" in summary
    assert sorted(path.name for path in output.iterdir()) == [
        "pytest.log",
        "report.json",
        "summary.txt",
    ]


@pytest.mark.parametrize("swap_component", ["ancestor", "final"])
def test_path_swap_fails_closed_without_external_writes(tmp_path, swap_component):
    report = {"tests": [{"nodeid": "tests/test_synthetic.py::test_failure", "outcome": "passed"}]}
    result, _, _, _ = invoke(tmp_path, report, swap_component=swap_component)
    assert result.returncode == 11
    assert "output-directory" in result.stderr
    assert list((tmp_path / "outside-output").iterdir()) == []
    if swap_component == "ancestor":
        displaced_output = tmp_path / "output-parent.displaced" / "output"
    else:
        displaced_output = tmp_path / "output.displaced"
    assert list(displaced_output.iterdir()) == []


def test_every_installed_rerun_artifact_is_redacted(tmp_path):
    import socket

    home = "/local/" + "home/person/project.py"
    account = "123456" * 2
    key_id = "AKIA" + "G" * 16
    sensitive_value = "very-sensitive"
    hostname = socket.gethostname()
    secret_assignment = "to" + "ken=" + sensitive_value
    message = f"at {home} {account} {key_id} {secret_assignment} host={hostname}"
    report = failed_report(message)
    report["metadata"] = {"nested": [message, {"value": message}]}
    result, output, _, _ = invoke(
        tmp_path,
        report,
        pytest_exit=1,
        log_text=message,
    )
    assert result.returncode == 1, result.stderr
    json.loads((output / "report.json").read_text())
    installed = "\n".join(path.read_text() for path in output.iterdir())
    for sensitive in (home, account, key_id, sensitive_value, hostname):
        assert sensitive not in installed
    for token in (
        "<redacted-home>",
        "<redacted-account>",
        "<redacted-key-id>",
        "<redacted-secret>",
        "<redacted-hostname>",
    ):
        assert token in installed


def test_mkdir_failure_removes_exact_rerun_temp_directory(tmp_path):
    report = {"tests": [{"nodeid": "tests/test_synthetic.py::test_failure", "outcome": "passed"}]}
    temp_root = tmp_path / "temp-root"
    temp_root.mkdir()
    result, _, _, _ = invoke(tmp_path, report, fail_mkdir=True, temp_root=temp_root)
    created = Path((tmp_path / "mkdir-parent.txt").read_text().strip())
    assert result.returncode == 11
    assert created.parent == temp_root
    assert created.name.startswith("pytorch-upstream-rerun.")
    assert not created.exists()
    assert list(temp_root.iterdir()) == []


def test_tmpdir_is_honored(tmp_path):
    report = {"tests": [{"nodeid": "tests/test_synthetic.py::test_failure", "outcome": "passed"}]}
    temp_root = tmp_path / "custom-temp"
    temp_root.mkdir()
    result, _, _, _ = invoke(tmp_path, report, fail_mkdir=True, temp_root=temp_root)
    created = Path((tmp_path / "mkdir-parent.txt").read_text().strip())
    assert result.returncode == 11
    assert created.parent == temp_root
    assert not created.exists()


def test_pytest_exit_three_is_not_confused_with_prerequisite_failure(tmp_path):
    report = {"tests": [{"nodeid": "tests/test_synthetic.py::test_failure", "outcome": "passed"}]}
    result, output, _, _ = invoke(tmp_path, report, pytest_exit=3)
    assert result.returncode == 3
    assert (output / "report.json").exists()
    assert "# pytest exit=3" in result.stderr


@pytest.mark.parametrize("kind", ["relative", "missing", "file", "symlink"])
def test_tmpdir_must_be_an_absolute_existing_real_directory(tmp_path, kind):
    case_dir = tmp_path / kind
    case_dir.mkdir()
    if kind == "relative":
        temp_root = Path("relative-temp")
    elif kind == "missing":
        temp_root = case_dir / "missing"
    elif kind == "file":
        temp_root = case_dir / "temp-file"
        temp_root.write_text("not a directory\n")
    else:
        real_temp = case_dir / "real-temp"
        real_temp.mkdir()
        temp_root = case_dir / "temp-link"
        temp_root.symlink_to(real_temp, target_is_directory=True)
    report = {"tests": [{"nodeid": "tests/test_synthetic.py::test_failure", "outcome": "passed"}]}
    result, output, _, _ = invoke(case_dir, report, temp_root=temp_root)
    assert result.returncode == 10
    assert "TMPDIR must be an absolute path to an existing real directory" in result.stderr
    assert not (output / "report.json").exists()
