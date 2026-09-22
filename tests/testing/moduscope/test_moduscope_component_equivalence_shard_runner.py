"""Pytest calls this module to test the component-equivalence shard runner."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import moduscope_component_equivalence_shard_runner as component_runner
import moduscope_process as entry_results
import moduscope_runner_support as runner_support
import pytest
from moduscope_shard_planner import Shard
from moduscope_targets import Target

from conftest import RUN_ID, run_test_shard, write_device_results

TARGET = Target(
    name="component-equiv-dense",
    timeout_minutes=330,
    adapter="component-equivalence",
    shards=(Shard(0, ("qwen3_8b", "qwen3_14b")), Shard(1, ("llama3_8b", "llama3_70b"))),
    torch_compile_enabled=True,
)
SINGLE_MEMBER_SHARD = Shard(0, ("qwen3_8b",))
FAKE_SUITE_SCRIPT = """#!/bin/bash
here="$(cd "$(dirname "$0")" && pwd)"
printf '%s\\n' "args=$*" "MODEL_CONFIG=$MODEL_CONFIG" "TORCH_COMPILE=$TORCH_COMPILE" \\
  "NUM_WORKERS=$NUM_WORKERS" > seen.txt
for device in cpu neuron; do mkdir -p "logs/$MODEL_CONFIG/$device"; done
for device in cpu neuron; do cp "{results}"/* "logs/$MODEL_CONFIG/$device/"; done
exit 3
"""
PASSING_COUNTS = {"passed": 19, "failed": 1, "error": 0}
UNDER_THRESHOLD_COUNTS = {"passed": 18, "failed": 2, "error": 0}
DEFAULT_VALUE = object()
OMIT_FIELD = object()


def read_grid_rows(output_dir) -> list[dict]:
    return json.loads(Path(output_dir, "grid_metrics.json").read_text())


def component_metrics_row(
    *, threshold_result="PASS", threshold=95.0, by_model=DEFAULT_VALUE, by_module=DEFAULT_VALUE
) -> dict:
    row = {"threshold_result": threshold_result, "threshold": threshold}
    if by_model is not OMIT_FIELD:
        row["by_model"] = {"m": dict(PASSING_COUNTS)} if by_model is DEFAULT_VALUE else by_model
    if by_module is not OMIT_FIELD:
        row["by_module"] = {"x": dict(PASSING_COUNTS)} if by_module is DEFAULT_VALUE else by_module
    return row


def write_counted_device_results(directory, verdict="PASS"):
    write_device_results(directory, verdict)
    path = directory / "grid_metrics.json"
    (row,) = json.loads(path.read_text())
    row.update(component_metrics_row(threshold_result=verdict))
    path.write_text(json.dumps([row]))


def write_configuration_result(suite_dir, model_config, verdict="PASS", **row_fields):
    results_dir = Path(suite_dir, "logs", model_config, "neuron")
    write_counted_device_results(results_dir, verdict)
    if row_fields:
        grid_path = results_dir / "grid_metrics.json"
        (row,) = json.loads(grid_path.read_text())
        row.update(row_fields)
        grid_path.write_text(json.dumps([row]))
    return results_dir


def write_process_tree(root, processes):
    """Create device-file links for each process ID."""
    for process_id, targets in processes.items():
        (root / str(process_id) / "fd").mkdir(parents=True)
        for file_descriptor, target in enumerate(targets, start=3):
            (root / str(process_id) / "fd" / str(file_descriptor)).symlink_to(target)


def process_result(returncode=0):
    return entry_results.InvocationResult(returncode, f"ModuScope exited with code {returncode}", 0)


@pytest.fixture(autouse=True)
def empty_device_state(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_support, "PROC_ROOT", str(tmp_path / "proc"))


@pytest.fixture
def component_run_dirs(tmp_path, monkeypatch):
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    monkeypatch.setattr(component_runner, "SUITE_DIR", str(suite_dir))
    output_dir = tmp_path / "outputs" / "moduscope"
    output_dir.mkdir(parents=True)
    return suite_dir, output_dir


@pytest.mark.parametrize(("compile_enabled", "expected_value"), [(False, "0"), (True, "1")])
def test_build_test_invocation_uses_suite_command_directory_and_environment(
    compile_enabled, expected_value
):
    target = TARGET._replace(torch_compile_enabled=compile_enabled)
    assert component_runner.build_test_invocation(
        target, "llama3_70b"
    ) == entry_results.TestInvocation(
        argv=("bash", "run_all_matrices.sh", "1"),
        cwd=component_runner.SUITE_DIR,
        env={"MODEL_CONFIG": "llama3_70b", "TORCH_COMPILE": expected_value, "NUM_WORKERS": "64"},
        unset_env=("DEVICES", "TEST_FILES"),
    )


@pytest.mark.parametrize(
    ("compile_enabled", "compile_value"), [(False, "0"), (True, "1")], ids=["eager", "compile"]
)
def test_run_shard_collects_each_suite_device_and_emits_three_token_lines(
    run_identity_env, component_run_dirs, compile_enabled, compile_value
):
    suite_dir, output_dir = component_run_dirs
    result_template = suite_dir.parent / "template" / "neuron"
    write_counted_device_results(result_template)
    (suite_dir / "run_all_matrices.sh").write_text(
        FAKE_SUITE_SCRIPT.format(results=result_template)
    )
    target = TARGET._replace(torch_compile_enabled=compile_enabled)
    assert run_test_shard(component_runner, target, SINGLE_MEMBER_SHARD, output_dir, 900) == 3
    assert (suite_dir / "run_all_matrices.sh").stat().st_mode & stat.S_IXUSR
    assert (suite_dir / "seen.txt").read_text().splitlines() == [
        "args=1",
        "MODEL_CONFIG=qwen3_8b",
        f"TORCH_COMPILE={compile_value}",
        "NUM_WORKERS=64",
    ]
    collected_results = output_dir / "component-equiv" / "qwen3_8b"
    assert sorted(path.name for path in collected_results.iterdir()) == ["cpu", "neuron"]
    assert (collected_results / "cpu" / "summary.txt").read_text() == (
        result_template / "summary.txt"
    ).read_text()
    fields = ("model_config", "threshold_result", "status", "workload_id")
    expected = ("qwen3_8b", "PASS", "pass", RUN_ID)
    assert [tuple(row[field] for field in fields) for row in read_grid_rows(output_dir)] == [
        expected,
        expected,
    ]
    assert entry_results._token_lines == [
        "GUARDS entry=0 model=qwen3_8b caches_cleared=0 device_pids=none foreign=none",
        "COMPONENT_EQUIV_RC=3",
        "COLLECTED_REPORTS=2",
        "AGGREGATE_GRID_ROWS=2",
    ]
    assert not (
        output_dir / "tokens.txt"
    ).exists(), "moduscope_pod_runner.py writes the file, not the runner"


def test_run_shard_runs_configurations_in_order_and_combines_result_trees(
    monkeypatch, component_run_dirs
):
    suite_dir, output_dir = component_run_dirs
    now = [0.0]
    monkeypatch.setattr(runner_support.time, "monotonic", lambda: now[0])
    calls = []

    def run_configuration(invocation, timeout_seconds, *_args):
        actual_suite_dir = invocation.cwd
        model_config = invocation.env["MODEL_CONFIG"]
        if calls:
            previous_member = calls[-1][1]
            assert (
                output_dir / "component-equiv" / previous_member / "neuron" / "grid_metrics.json"
            ).is_file()
        calls.append((actual_suite_dir, model_config, timeout_seconds))
        write_configuration_result(
            actual_suite_dir, model_config, verdict="FAIL", error=f"{model_config} metrics failed"
        )
        now[0] += 100.0
        return process_result()

    monkeypatch.setattr(component_runner, "run_member_process", run_configuration)
    members = TARGET.shards[0].members

    assert run_test_shard(component_runner, TARGET, TARGET.shards[0], output_dir, 900) == 1
    assert [model_config for _, model_config, _ in calls] == list(members)
    assert all(actual_suite_dir == str(suite_dir) for actual_suite_dir, _, _ in calls)
    assert [timeout_seconds for _, _, timeout_seconds in calls] == [900.0, 800.0]
    results_root = output_dir / "component-equiv"
    assert sorted(path.name for path in results_root.iterdir()) == sorted(members)
    assert [
        (row["model_config"], row["status"], row["error"]) for row in read_grid_rows(output_dir)
    ] == [
        ("qwen3_8b", "fail", "qwen3_8b metrics failed"),
        ("qwen3_14b", "fail", "qwen3_14b metrics failed"),
    ]
    assert entry_results._token_lines == [
        "GUARDS entry=0 model=qwen3_8b caches_cleared=0 device_pids=none foreign=none",
        "GUARDS entry=1 model=qwen3_14b caches_cleared=0 device_pids=none foreign=none",
        "COMPONENT_EQUIV_RC=1",
        "COLLECTED_REPORTS=2",
        "AGGREGATE_GRID_ROWS=2",
    ]


def test_run_shard_keeps_first_failure_and_runs_later_configuration(
    monkeypatch, component_run_dirs
):
    _, output_dir = component_run_dirs
    calls = []
    members = TARGET.shards[0].members

    def run_configuration(invocation, _timeout_seconds, *_args):
        model_config = invocation.env["MODEL_CONFIG"]
        calls.append(model_config)
        if model_config == members[0]:
            return process_result(3)
        write_configuration_result(invocation.cwd, model_config)
        return process_result()

    monkeypatch.setattr(component_runner, "run_member_process", run_configuration)

    assert run_test_shard(component_runner, TARGET, TARGET.shards[0], output_dir, 900) == 3
    assert calls == list(members)
    assert [
        (row["model_config"], row["status"], row.get("error")) for row in read_grid_rows(output_dir)
    ] == [
        ("qwen3_8b", "infra_fail", "component suite exited with code 3. Check the suite log."),
        ("qwen3_14b", "pass", None),
    ]


@pytest.mark.parametrize(
    (
        "listed_verdict",
        "listed_exit_code",
        "other_verdict",
        "other_exit_code",
        "expected_exit_code",
        "statuses",
    ),
    [
        ("FAIL", 0, "PASS", 0, 0, ["xfail", "pass"]),
        ("PASS", 0, "PASS", 0, 0, ["xpass", "pass"]),
        ("FAIL", 0, "FAIL", 0, 1, ["xfail", "fail"]),
        ("FAIL", 1, "PASS", 0, 0, ["xfail", "pass"]),
        ("FAIL", 124, "PASS", 0, 124, ["xfail", "pass"]),
        ("FAIL", 137, "PASS", 0, 137, ["xfail", "pass"]),
        ("PASS", 3, "PASS", 0, 3, ["pass", "pass"]),
        (None, 3, "PASS", 0, 3, ["infra_fail", "pass"]),
        (None, 1, "PASS", 0, 1, ["infra_fail", "pass"]),
        ("PASS", 0, "FAIL", 1, 1, ["xpass", "fail"]),
    ],
    ids=[
        "listed-failure-is-xfail",
        "listed-pass-is-xpass",
        "unlisted-failure-still-fails",
        "listed-failure-absorbs-exit",
        "listed-failure-keeps-timeout-exit",
        "listed-failure-keeps-signal-exit",
        "listed-pass-keeps-other-exit",
        "listed-no-rows-keeps-exit",
        "listed-no-rows-keeps-fail-verdict-exit",
        "unlisted-failure-keeps-exit",
    ],
)
def test_run_shard_applies_expected_failures_to_listed_members_only(
    monkeypatch,
    component_run_dirs,
    listed_verdict,
    listed_exit_code,
    other_verdict,
    other_exit_code,
    expected_exit_code,
    statuses,
):
    _, output_dir = component_run_dirs
    listed_member, other_member = TARGET.shards[0].members
    target = TARGET._replace(xfail={listed_member: "known device finding"})

    def run_configuration(invocation, _timeout_seconds, *_args):
        model_config = invocation.env["MODEL_CONFIG"]
        if model_config == other_member:
            write_configuration_result(invocation.cwd, model_config, other_verdict)
            return process_result(other_exit_code)
        if listed_verdict is not None:
            write_configuration_result(invocation.cwd, model_config, listed_verdict)
        return process_result(listed_exit_code)

    monkeypatch.setattr(component_runner, "run_member_process", run_configuration)

    assert (
        run_test_shard(component_runner, target, TARGET.shards[0], output_dir, 900)
        == expected_exit_code
    )
    assert [(row["model_config"], row["status"]) for row in read_grid_rows(output_dir)] == [
        (listed_member, statuses[0]),
        (other_member, statuses[1]),
    ]
    assert entry_results._token_lines[-3] == f"COMPONENT_EQUIV_RC={expected_exit_code}"


@pytest.mark.parametrize(
    ("row_fields", "expected_exit_code", "status", "error"),
    [
        ({}, 0, "xfail", None),
        (
            {"by_model": None},
            1,
            "infra_fail",
            "invalid component-equivalence metrics: by_model must be a nonempty object",
        ),
    ],
    ids=["threshold-failure-is-xfail", "malformed-metrics-are-infra-fail"],
)
def test_listed_member_with_malformed_metrics_fails_the_shard(
    monkeypatch, component_run_dirs, row_fields, expected_exit_code, status, error
):
    _, output_dir = component_run_dirs
    (member,) = SINGLE_MEMBER_SHARD.members
    target = TARGET._replace(xfail={member: "known device finding"})

    def run_configuration(invocation, _timeout_seconds, *_args):
        write_configuration_result(invocation.cwd, member, "FAIL", **row_fields)
        return process_result(1)

    monkeypatch.setattr(component_runner, "run_member_process", run_configuration)

    assert (
        run_test_shard(component_runner, target, SINGLE_MEMBER_SHARD, output_dir, 900)
        == expected_exit_code
    )
    assert [
        (row["model_config"], row["status"], row.get("error")) for row in read_grid_rows(output_dir)
    ] == [(member, status, error)]
    assert entry_results._token_lines[-3] == f"COMPONENT_EQUIV_RC={expected_exit_code}"


def test_held_device_refuses_the_next_configuration(monkeypatch, component_run_dirs, capsys):
    suite_dir, output_dir = component_run_dirs
    process_root = Path(runner_support.PROC_ROOT)
    calls = []

    def run_configuration(invocation, _timeout_seconds, *_args):
        model_config = invocation.env["MODEL_CONFIG"]
        calls.append(model_config)
        write_configuration_result(invocation.cwd, model_config)
        write_process_tree(process_root, {4242: ["/dev/neuron0"]})
        return process_result()

    monkeypatch.setattr(component_runner, "run_member_process", run_configuration)
    monkeypatch.setattr(runner_support, "DEVICE_GRACE_SECONDS", 0.0)
    monkeypatch.setattr(runner_support, "send_sigkill", lambda _process_id: None)

    assert run_test_shard(component_runner, TARGET, TARGET.shards[0], output_dir, 900) == 1
    assert calls == ["qwen3_8b"]
    assert read_grid_rows(output_dir) == [
        {
            **component_metrics_row(),
            "device": "neuron",
            "compile": False,
            "model_config": "qwen3_8b",
            "status": "pass",
        },
        {
            "model_config": "qwen3_14b",
            "status": "infra_fail",
            "error": "device held by pids [4242] after qwen3_8b exited",
            "never_started": True,
        },
    ]
    assert entry_results._token_lines == [
        "GUARDS entry=0 model=qwen3_8b caches_cleared=0 device_pids=4242 foreign=none",
        "DEVICE_HELD entry=0 pids=4242",
        "COMPONENT_EQUIV_RC=1",
        "COLLECTED_REPORTS=1",
        "AGGREGATE_GRID_ROWS=2",
    ]
    assert "DEVICE_HELD entry=0 pids=4242" in capsys.readouterr().out


def test_run_shard_records_configurations_that_miss_the_deadline(
    run_identity_env, monkeypatch, component_run_dirs
):
    _, output_dir = component_run_dirs
    now = [0.0]
    monkeypatch.setattr(runner_support.time, "monotonic", lambda: now[0])
    calls = []
    shard = Shard(0, ("qwen3_8b", "qwen3_14b", "qwen3_32b"))

    def run_configuration(invocation, timeout_seconds, *_args):
        calls.append((invocation.env["MODEL_CONFIG"], timeout_seconds))
        write_configuration_result(invocation.cwd, invocation.env["MODEL_CONFIG"])
        now[0] = 10.0
        return process_result()

    monkeypatch.setattr(component_runner, "run_member_process", run_configuration)

    assert run_test_shard(component_runner, TARGET, shard, output_dir, 10) == 124
    assert calls == [("qwen3_8b", 10.0)]
    rows = read_grid_rows(output_dir)
    assert (rows[0]["model_config"], rows[0]["status"]) == ("qwen3_8b", "pass")
    for row, member_name in zip(rows[1:], shard.members[1:], strict=True):
        assert row == {
            "model_config": member_name,
            "status": "infra_fail",
            "error": "never started: shard deadline reached after 1 of 3 members",
            "never_started": True,
            "workload_id": RUN_ID,
            "workload_link": (
                f"https://github.com/aws-neuron/private-torch-neuronx/actions/runs/{RUN_ID}"
            ),
        }
    assert entry_results._token_lines == [
        "GUARDS entry=0 model=qwen3_8b caches_cleared=0 device_pids=none foreign=none",
        "COMPONENT_EQUIV_RC=124",
        "COLLECTED_REPORTS=1",
        "AGGREGATE_GRID_ROWS=3",
    ]


def test_shard_grid_keeps_readable_devices_and_applies_thresholds(tmp_path, capsys):
    results = tmp_path / "component-equiv" / "qwen3_8b"
    for device, verdict in (("neuron", "PASS"), ("cpu", "FAIL")):
        write_counted_device_results(results / device, verdict)
    broken = results / "neuron-compiled" / "grid_metrics.json"
    broken.parent.mkdir()
    broken.write_text("{ truncated")
    write_device_results(results.parent / "llama3_8b" / "neuron")
    assert component_runner.write_component_shard_grid(
        str(tmp_path), str(results.parent), ("qwen3_8b",), {}, None, {}, {}
    ) == (3, True)
    assert [
        (
            row["model_config"],
            row["device"],
            row["compile"],
            row.get("threshold_result"),
            row["status"],
        )
        for row in read_grid_rows(tmp_path)
    ] == [
        ("qwen3_8b", "cpu", False, "FAIL", "fail"),
        ("qwen3_8b", "neuron", False, "PASS", "pass"),
        ("qwen3_8b", "neuron", True, None, "infra_fail"),
    ]
    assert capsys.readouterr().out.splitlines() == [f"UNREADABLE_GRID path={broken}"]


def test_row_status_requires_model_and_module_thresholds(tmp_path):
    results = tmp_path / "component-equiv" / "qwen3_8b" / "neuron"
    write_device_results(results)
    cases = [
        (component_metrics_row(), "pass"),
        (component_metrics_row(threshold="95.0"), "pass"),
        (component_metrics_row(by_module=OMIT_FIELD), "pass"),
        (component_metrics_row(threshold_result="FAIL"), "fail"),
        (component_metrics_row(by_model={"m": UNDER_THRESHOLD_COUNTS}), "fail"),
        (component_metrics_row(by_module={"x": UNDER_THRESHOLD_COUNTS}), "fail"),
        (component_metrics_row(by_model={"m": {"skipped": 20}}), "fail"),
        (component_metrics_row(threshold=None), "fail"),
        (component_metrics_row(by_model=OMIT_FIELD), "infra_fail"),
        (component_metrics_row(by_model={}), "infra_fail"),
        (component_metrics_row(by_model={"m": {"passed": 18, "error": 2}}), "fail"),
    ]
    rows = [row for row, _ in cases]
    (results / "grid_metrics.json").write_text(json.dumps(rows))
    assert component_runner.write_component_shard_grid(
        str(tmp_path), str(results.parents[1]), ("qwen3_8b",), {}, None, {}, {}
    ) == (len(cases), True)
    assert [row["status"] for row in read_grid_rows(tmp_path)] == [
        expected for _, expected in cases
    ]


def test_malformed_metrics_create_infra_fail_rows_with_errors(tmp_path):
    results = tmp_path / "component-equiv" / "qwen3_8b" / "neuron"
    results.mkdir(parents=True)
    rows = [
        component_metrics_row(by_model={"m": None}),
        component_metrics_row(by_module={"x": None}),
        component_metrics_row(threshold="not-a-number"),
        component_metrics_row(by_model={"m": UNDER_THRESHOLD_COUNTS}, by_module={"x": None}),
        component_metrics_row(by_model={"first": UNDER_THRESHOLD_COUNTS, "bad": None}),
        component_metrics_row(threshold="not-a-number", by_model={"m": {"skipped": 20}}),
        component_metrics_row(threshold=[]),
        component_metrics_row(threshold=10**400),
        component_metrics_row(by_model={"m": {"passed": "20"}}),
        component_metrics_row(by_model={"m": {"passed": 20, "failed": -1}}),
        component_metrics_row(by_model={"m": {"passed": True}}),
        component_metrics_row(threshold="-inf"),
        component_metrics_row(threshold=True),
        component_metrics_row(by_module="not-an-object"),
        component_metrics_row(by_model=None),
    ]
    (results / "grid_metrics.json").write_text(json.dumps(rows))
    assert component_runner.write_component_shard_grid(
        str(tmp_path), str(results.parents[1]), ("qwen3_8b",), {}, None, {}, {}
    ) == (len(rows), True)
    written = read_grid_rows(tmp_path)
    assert [row["status"] for row in written] == ["infra_fail"] * len(rows)
    assert all(
        row["error"].startswith("invalid component-equivalence metrics: ") for row in written
    )


def test_make_executable_updates_both_scripts_and_ignores_missing_paths(tmp_path):
    suite = tmp_path / "suite"
    scripts = [suite / "run_all_matrices.sh", suite / "bf16" / "tests" / "run_all.sh"]
    for script in scripts:
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("#!/bin/bash\n")
    assert [bool(script.stat().st_mode & stat.S_IXUSR) for script in scripts] == [False, False]
    component_runner.make_suite_scripts_executable(str(suite))
    assert [bool(script.stat().st_mode & stat.S_IXUSR) for script in scripts] == [True, True]
    component_runner.make_suite_scripts_executable(str(tmp_path / "nowhere"))


@pytest.mark.parametrize(
    ("suite_exit_code", "expected_exit_code", "expected_error"),
    [
        (0, 1, "component suite wrote no readable result rows. Check the suite log."),
        (1, 1, "component suite exited with code 1. Check the suite log."),
    ],
)
def test_missing_logs_create_a_configuration_error_row(
    monkeypatch, tmp_path, suite_exit_code, expected_exit_code, expected_error
):
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "run_all_matrices.sh").write_text(f"#!/bin/bash\nexit {suite_exit_code}\n")
    monkeypatch.setattr(component_runner, "SUITE_DIR", str(suite))
    out = tmp_path / "outputs" / "moduscope"
    out.mkdir(parents=True)
    assert (
        run_test_shard(component_runner, TARGET, SINGLE_MEMBER_SHARD, out, 900)
        == expected_exit_code
    )
    results = out / "component-equiv"
    assert results.is_dir() and not any(results.iterdir())
    assert entry_results._token_lines == [
        "GUARDS entry=0 model=qwen3_8b caches_cleared=0 device_pids=none foreign=none",
        f"COMPONENT_EQUIV_RC={expected_exit_code}",
        "COLLECTED_REPORTS=0",
        "AGGREGATE_GRID_ROWS=1",
    ]
    assert read_grid_rows(out) == [
        {"model_config": "qwen3_8b", "status": "infra_fail", "error": expected_error}
    ]
