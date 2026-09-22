"""Pytest calls this module to test the shard runner with fake model processes and real process
groups.
"""

from __future__ import annotations

import errno
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import moduscope_artifact_layout as entry_results
import moduscope_hugging_face_shard_runner as shard_runner
import moduscope_process as process
import moduscope_runner_support as runner_support
import pytest
from moduscope_shard_planner import Shard, build_shards
from moduscope_targets import Target

from conftest import RUN_IDENTITY_FIELDS, run_test_shard
from conftest import SAMPLE_HUGGING_FACE_MODEL_IDS as MODEL_IDS

GRID_FILE_NAME = "grid_metrics.json"
BENCHMARK_FILE_NAME = "benchmark_results.jsonl"
ENTRIES_SUBDIR = shard_runner.ENTRIES_SUBDIR
TARGET_CONFIG = "full_dense_decoder"
TARGET = Target(
    name="test_hf",
    timeout_minutes=90,
    adapter="hugging-face",
    config=TARGET_CONFIG,
    shards=build_shards((MODEL_IDS,)),
)
EXPECTED_FAILURE_TARGET = TARGET._replace(xfail={MODEL_IDS[1]: "known device finding"})
MODEL_CONFIG_TOKEN = "smeoni_tiny-random-LlamaForCausalLM"
ABSORBED = f"XFAIL_ABSORBED=1 model={MODEL_IDS[1]}"
KILLED_REASON = "killed by the runner after {}s (exit code 124)"
NEVER_STARTED_REASON = "never started: shard deadline reached after 2 of 3 members"
SHARD_ORDER = (2, 1, 0)
ALL_MODEL_INDEXES = (0, 1, 2)
REMOVAL_TOKEN_KEYS = ("INDIVIDUAL_REMOVED", "XLA_DUMP_REMOVED", "MANIFESTS_REMOVED")
REMOVED_PATHS = ("individual/x/outputs/x", "xla_dump/module_0000.hlo.pb", "manifests/x.json")
TOKEN_KIND_PATTERN = re.compile(r"::\w+::|[A-Z_]+(?=[= ]|$)")
# Use a process ID above the system limit so no real process group matches it.
FAKE_PID = 2**31 - 1


def read_rows(root, *parts):
    return json.loads(Path(root).joinpath(*parts).read_text())


def make_shard(*indexes):
    return Shard(0, tuple(MODEL_IDS[index] for index in indexes))


def command_model_id(command_argv):
    return (
        next(argument for argument in command_argv if "hf_cfg_wishlist" in argument)
        .split("['")[1]
        .rstrip("']")
    )


def fake_clock():
    """Advance the fake clock during sleep.

    Fail if a loop reads the clock 10,000 times without sleeping.
    """
    fake = SimpleNamespace(now=1000.0, reads=0)

    def monotonic():
        fake.reads += 1
        assert fake.reads <= 10_000, "10,000 clock reads without a sleep: a loop that never sleeps"
        return fake.now

    def sleep(seconds):
        fake.now, fake.reads = fake.now + seconds, 0

    fake.monotonic, fake.sleep = monotonic, sleep
    return fake


def build_fake_child_process(calls, plan, fake_time, on_start):
    """Return a fake child process class.

    A timeout case advances the fake clock and raises TimeoutExpired.
    """

    class Child:
        pid = FAKE_PID

        def __init__(self, command_argv, **kwargs):
            self.command_argv, self.outcome, self.timed_out = (
                command_argv,
                plan.get(command_model_id(command_argv), 0),
                False,
            )
            calls.append((command_argv, kwargs))
            target = Path(
                next(
                    argument for argument in command_argv if "artifact_directory" in argument
                ).split("=", 1)[1]
            )
            model_id = command_model_id(command_argv)
            for relative_path in (*REMOVED_PATHS, "torchrun_logs/rank0.log"):
                (target / relative_path).parent.mkdir(parents=True, exist_ok=True)
                (target / relative_path).write_bytes(b"x")
            if self.outcome != "timeout":
                token = model_id.replace("/", "_") + "_config"
                row = {"module_configs_str": token, "status": "pass"}
                if self.outcome:
                    row.update(status="fail", error="loss mismatch")
                grid_path = target / GRID_FILE_NAME
                existing_rows = json.loads(grid_path.read_text()) if grid_path.exists() else []
                grid_path.write_text(json.dumps([*existing_rows, row]))
                (target / BENCHMARK_FILE_NAME).write_text(json.dumps({"model": model_id}) + "\n")
            if on_start is not None:
                on_start()

        def wait(self, timeout=None):
            if self.outcome == "timeout" and not self.timed_out:
                self.timed_out = True
                fake_time.sleep(timeout)
                raise subprocess.TimeoutExpired(self.command_argv, timeout)
            return -9 if self.timed_out else self.outcome

        def poll(self):
            return None if self.outcome == "timeout" else self.outcome

    return Child


def run_shard_with_fakes(
    monkeypatch,
    tmp_path,
    shard,
    calls=None,
    plan=None,
    fake_time=None,
    on_start=None,
    *,
    target=TARGET,
    timeout=900,
):
    """Run one shard with fake child processes.

    Return its exit code and output directory.
    """
    output_dir = tmp_path / entry_results.OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    fake_time, calls = (fake_time or fake_clock(), [] if calls is None else calls)
    monkeypatch.setattr(runner_support, "time", fake_time)
    monkeypatch.setattr(process, "time", fake_time)
    monkeypatch.setattr(
        subprocess, "Popen", build_fake_child_process(calls, plan or {}, fake_time, on_start)
    )
    return (run_test_shard(shard_runner, target, shard, output_dir, timeout, fake_time), output_dir)


def write_process_tree(root, processes):
    """Create a process tree with device-file links for each process ID."""
    for process_id, targets in processes.items():
        (root / str(process_id) / "fd").mkdir(parents=True)
        for file_descriptor, target in enumerate(targets, start=3):
            (root / str(process_id) / "fd" / str(file_descriptor)).symlink_to(target)


def never_started_output(index, reason):
    """Build the NEVER_STARTED line and its following warning."""
    model = MODEL_IDS[index]
    return (
        f"NEVER_STARTED entry={index} model={model} reason={reason}\n"
        + f"::warning::model {model.replace('/', '_')} {model} status=infra_fail error={reason}\n"
    )


@pytest.fixture(autouse=True)
def empty_device_state(monkeypatch, tmp_path):
    monkeypatch.setattr(runner_support, "PROC_ROOT", str(tmp_path / "proc"))
    monkeypatch.setattr(shard_runner, "clear_compile_caches", lambda: ())


@pytest.mark.parametrize(
    ("torch_compile_enabled", "compile_override"),
    [(False, ()), (True, ("++task_config.apply_torch_compile=true",))],
    ids=["eager", "compile"],
)
def test_build_test_invocation_has_the_exact_process_inputs(
    tmp_path, torch_compile_enabled, compile_override
):
    entry_output_dir = tmp_path / "entry-1"
    target = TARGET._replace(torch_compile_enabled=torch_compile_enabled)

    invocation = shard_runner.build_test_invocation(target, MODEL_IDS[1], str(entry_output_dir))

    assert invocation == process.TestInvocation(
        argv=(
            sys.executable,
            "-m",
            "cli_scripts.task_runner",
            f"--config-name={TARGET_CONFIG}",
            f"++task_config.hf_cfg_wishlist=['{MODEL_IDS[1]}']",
            f"++task_config.artifact_directory={entry_output_dir}",
            "++task_config.always_clean_artifacts=false",
            "++task_config.clean_artifacts_on_success=false",
            *compile_override,
        )
    )
    assert not entry_output_dir.exists()


SINGLE = [
    (0, False, 0, "pass", "ENTRY_GRID entry=1 rows=1 marked_failed=0 profile_s3_url=none"),
    (2, False, 2, "fail",
     f"::warning::model {MODEL_CONFIG_TOKEN} status=fail error=loss mismatch"),
    ("timeout", False, 124, "infra_fail",
     f"::warning::model {MODEL_CONFIG_TOKEN} {MODEL_IDS[1]} status=infra_fail " +
     f"error={KILLED_REASON.format(900)}"),
    (0, True, 0, "xpass",
     f"::warning::model {MODEL_CONFIG_TOKEN} xpass: prune its xfail entry"),
    (2, True, 0, "xfail",
     f"::notice::model {MODEL_CONFIG_TOKEN} status=xfail reason=known device finding " +
     "error=loss mismatch")
]  # fmt: skip


@pytest.mark.parametrize(("outcome", "listed", "exit_code", "status", "note"), SINGLE)
def test_one_member_shard_uses_the_single_model_path(
    tmp_path, monkeypatch, capsys, outcome, listed, exit_code, status, note
):
    target = EXPECTED_FAILURE_TARGET if listed else TARGET
    seen, out = run_shard_with_fakes(
        monkeypatch, tmp_path, make_shard(1), plan={MODEL_IDS[1]: outcome}, target=target
    )
    assert seen == exit_code
    rows = read_rows(out, GRID_FILE_NAME)
    assert [row["status"] for row in rows] == [status]
    if status == "infra_fail":
        assert rows[0]["mfu(%)"] == -1
    assert read_rows(out, ENTRIES_SUBDIR, "entry-1", GRID_FILE_NAME) == read_rows(
        out, GRID_FILE_NAME
    )
    printed = capsys.readouterr().out
    assert note in printed
    assert (ABSORBED in printed) == (listed and outcome != 0)
    # Keep this token order because hosted jobs parse it.
    lines = printed.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("MODUSCOPE_PROCESS_"))
    end = next(i for i, line in enumerate(lines) if line.startswith("MANIFESTS_REMOVED="))
    block = lines[start : end + 1]
    middle = {"pass": [], "xfail": ["::notice::", "XFAIL_ABSORBED"]}.get(status, ["::warning::"])
    kinds = [match.group() if (match := TOKEN_KIND_PATTERN.match(line)) else line for line in block]
    assert kinds == ["MODUSCOPE_PROCESS_EXIT_CODE", "ENTRY_GRID", *middle, *REMOVAL_TOKEN_KEYS]
    model_exit_code = 124 if outcome == "timeout" else outcome
    assert block[0] == (
        f"MODUSCOPE_PROCESS_EXIT_CODE={model_exit_code} entry=1 model={MODEL_IDS[1]}"
    )
    assert block[-3:] == [f"{token}=1" for token in REMOVAL_TOKEN_KEYS]
    assert f"SHARD_DONE index=0 started=1 never_started=0 rc={exit_code}" in printed
    assert not (out / "tokens.txt").exists(), "moduscope_pod_runner.py writes it, not the runner"


@pytest.mark.parametrize("listed", [False, True], ids=["unlisted", "listed"])
def test_failure_without_wheel_grid_is_infrastructure_failure_even_when_listed(
    tmp_path, monkeypatch, capsys, listed
):
    reason = "ModuScope exited with code 3"
    monkeypatch.setattr(
        runner_support.process,
        "run_test_invocation",
        lambda _invocation, _timeout_seconds: process.InvocationResult(3, reason, 15.0),
    )

    seen, out = run_shard_with_fakes(
        monkeypatch, tmp_path, make_shard(1), target=EXPECTED_FAILURE_TARGET if listed else TARGET
    )

    (row,) = read_rows(out, GRID_FILE_NAME)
    assert seen == 3
    assert (row["status"], row["mfu(%)"], row["error"]) == ("infra_fail", -1, reason)
    assert ABSORBED not in capsys.readouterr().out


def test_placeholder_removal_matches_status_name_and_index(tmp_path):
    grid_path = tmp_path / GRID_FILE_NAME
    running_row = {"grid_entry_name": MODEL_IDS[1], "grid_entry_index": 1, "status": "running"}
    other_rows = [
        {**running_row, "status": "pass"},
        {**running_row, "grid_entry_name": "other"},
        {**running_row, "grid_entry_index": 2},
    ]
    shard_runner.write_grid_rows(str(grid_path), [running_row, *other_rows])

    assert not shard_runner.finalize_running_placeholder(
        str(grid_path), running_row, 0, "ModuScope exited with code 0"
    )
    assert read_rows(tmp_path, GRID_FILE_NAME) == other_rows


@pytest.mark.parametrize("grid_state", ["placeholder", "missing", "empty"])
def test_zero_exit_without_wheel_rows_is_infrastructure_failure(
    tmp_path, monkeypatch, capsys, grid_state
):
    grid_path = tmp_path / entry_results.OUTPUT_SUBDIR / ENTRIES_SUBDIR / "entry-1" / GRID_FILE_NAME

    def finish_model(_argv, _timeout_seconds):
        if grid_state == "missing":
            grid_path.unlink()
        elif grid_state == "empty":
            grid_path.write_text("[]")
        return process.InvocationResult(0, "ModuScope exited with code 0", 15.0)

    monkeypatch.setattr(runner_support.process, "run_test_invocation", finish_model)
    seen, out = run_shard_with_fakes(monkeypatch, tmp_path, make_shard(1))

    (row,) = read_rows(out, GRID_FILE_NAME)
    assert seen == 1
    assert (row["status"], row["mfu(%)"], row["error"]) == (
        "infra_fail",
        -1,
        "ModuScope exited with code 0 but wrote no result rows. Check the wheel output.",
    )
    assert read_rows(out, ENTRIES_SUBDIR, "entry-1", GRID_FILE_NAME) == [row]
    printed = capsys.readouterr().out
    assert f"MODUSCOPE_PROCESS_EXIT_CODE=0 entry=1 model={MODEL_IDS[1]}" in printed
    assert "SHARD_DONE index=0 started=1 never_started=0 rc=1" in printed


@pytest.mark.parametrize(
    ("listed", "exit_code", "status"),
    [(False, 1, "fail"), (True, 0, "xfail")],
    ids=["unlisted", "listed"],
)
def test_zero_exit_with_a_failed_row_fails_the_shard_unless_listed(
    tmp_path, monkeypatch, capsys, listed, exit_code, status
):
    grid_path = tmp_path / entry_results.OUTPUT_SUBDIR / ENTRIES_SUBDIR / "entry-1" / GRID_FILE_NAME
    failed_row = {
        "module_configs_str": f"{MODEL_CONFIG_TOKEN}_config",
        "status": "fail",
        "error": "loss mismatch",
    }

    def finish_model(_argv, _timeout_seconds):
        grid_path.write_text(json.dumps([failed_row]))
        return process.InvocationResult(0, "ModuScope exited with code 0", 15.0)

    monkeypatch.setattr(runner_support.process, "run_test_invocation", finish_model)
    seen, out = run_shard_with_fakes(
        monkeypatch, tmp_path, make_shard(1), target=EXPECTED_FAILURE_TARGET if listed else TARGET
    )

    (row,) = read_rows(out, GRID_FILE_NAME)
    assert seen == exit_code
    assert (row["status"], row["error"]) == (status, "loss mismatch")
    printed = capsys.readouterr().out
    assert f"MODUSCOPE_PROCESS_EXIT_CODE=0 entry=1 model={MODEL_IDS[1]}" in printed
    assert ABSORBED not in printed
    assert f"SHARD_DONE index=0 started=1 never_started=0 rc={exit_code}" in printed


@pytest.mark.parametrize("listed", [False, True], ids=["unlisted", "listed"])
def test_zero_exit_with_a_row_without_status_fails_the_shard_even_when_listed(
    tmp_path, monkeypatch, capsys, listed
):
    grid_path = tmp_path / entry_results.OUTPUT_SUBDIR / ENTRIES_SUBDIR / "entry-1" / GRID_FILE_NAME
    row_without_status = {"module_configs_str": f"{MODEL_CONFIG_TOKEN}_config"}

    def finish_model(_argv, _timeout_seconds):
        grid_path.write_text(json.dumps([row_without_status]))
        return process.InvocationResult(0, "ModuScope exited with code 0", 15.0)

    monkeypatch.setattr(runner_support.process, "run_test_invocation", finish_model)
    seen, out = run_shard_with_fakes(
        monkeypatch, tmp_path, make_shard(1), target=EXPECTED_FAILURE_TARGET if listed else TARGET
    )

    (row,) = read_rows(out, GRID_FILE_NAME)
    assert seen == 1
    assert "status" not in row
    printed = capsys.readouterr().out
    assert f"MODUSCOPE_PROCESS_EXIT_CODE=0 entry=1 model={MODEL_IDS[1]}" in printed
    assert ABSORBED not in printed
    assert "SHARD_DONE index=0 started=1 never_started=0 rc=1" in printed


def test_running_placeholder_is_replaced_by_the_final_grid(tmp_path, monkeypatch, run_identity_env):
    entry_dir = tmp_path / entry_results.OUTPUT_SUBDIR / ENTRIES_SUBDIR / "entry-1"
    running_row = {
        "grid_entry_name": MODEL_IDS[1],
        "grid_entry_index": 1,
        "module_configs_str": f"{MODEL_CONFIG_TOKEN}_config",
        "status": "running",
        **RUN_IDENTITY_FIELDS,
    }
    wheel_rows = [
        {
            "grid_entry_name": f"{MODEL_IDS[1]} wheel row {wheel_row_index}",
            "module_configs_str": f"{MODEL_CONFIG_TOKEN}_config",
            "status": "pass",
        }
        for wheel_row_index in range(2)
    ]
    observed_rows = []

    def finish_model(_argv, _timeout_seconds):
        observed_rows.append(read_rows(entry_dir, GRID_FILE_NAME))
        observed_rows.append(read_rows(tmp_path / entry_results.OUTPUT_SUBDIR, GRID_FILE_NAME))
        (entry_dir / GRID_FILE_NAME).write_text(json.dumps([running_row, *wheel_rows]))
        return process.InvocationResult(0, "ModuScope exited with code 0", 75.0)

    monkeypatch.setattr(runner_support.process, "run_test_invocation", finish_model)
    seen, out = run_shard_with_fakes(monkeypatch, tmp_path, make_shard(1))

    final_rows = read_rows(out, ENTRIES_SUBDIR, "entry-1", GRID_FILE_NAME)
    assert seen == 0 and observed_rows == [[running_row], [running_row]]
    assert final_rows == [
        {**wheel_row, **RUN_IDENTITY_FIELDS, "wall_minutes": 1.25} for wheel_row in wheel_rows
    ]
    assert all(row["status"] != "running" for row in final_rows)
    assert read_rows(out, GRID_FILE_NAME) == final_rows


@pytest.mark.parametrize(
    ("target", "plan", "exit_code", "codes", "statuses"),
    [
        (TARGET, {}, 0, (0, 0, 0), ["pass"] * 3),
        (TARGET, {MODEL_IDS[1]: -9, MODEL_IDS[0]: 3}, 137, (0, 137, 3), ["pass", "fail", "fail"]),
        (
            EXPECTED_FAILURE_TARGET,
            {MODEL_IDS[1]: 2, MODEL_IDS[0]: 3},
            3,
            (0, 2, 3),
            ["pass", "xfail", "fail"],
        ),
    ],
    ids=["all-pass", "first-nonzero-wins", "absorbed-member-masks-nothing"],
)
def test_shard_runs_members_in_order_and_keeps_first_nonzero_exit_code(
    tmp_path, monkeypatch, capsys, target, plan, exit_code, codes, statuses
):
    calls = []
    seen, out = run_shard_with_fakes(
        monkeypatch, tmp_path, make_shard(*SHARD_ORDER), calls, plan, target=target
    )
    assert seen == exit_code
    assert [command_model_id(command_argv) for command_argv, _ in calls] == [
        MODEL_IDS[index] for index in SHARD_ORDER
    ]
    rows = read_rows(out, GRID_FILE_NAME)
    assert [row["status"] for row in rows] == statuses
    assert [row.get("peak_memory_unreliable") for row in rows] == [None, True, True]
    assert [
        read_rows(out, ENTRIES_SUBDIR, f"entry-{index}", GRID_FILE_NAME) for index in SHARD_ORDER
    ] == [[row] for row in rows]
    printed = capsys.readouterr().out
    members = ", ".join(MODEL_IDS[index] for index in SHARD_ORDER)
    assert f"SHARD index=0 members=3 config={TARGET_CONFIG}: {members}" in printed
    assert re.findall(r"^MODUSCOPE_PROCESS_EXIT_CODE=(-?\d+)", printed, re.MULTILINE) == [
        str(exit_code) for exit_code in codes
    ]
    assert (ABSORBED in printed) == (target is EXPECTED_FAILURE_TARGET)
    assert "AGGREGATE_GRID_ROWS=3 AGGREGATE_BENCH_LINES=3 UNREADABLE_GRIDS=0" in printed
    assert [
        json.loads(line)["model"] for line in (out / BENCHMARK_FILE_NAME).read_text().splitlines()
    ] == [MODEL_IDS[index] for index in SHARD_ORDER]
    assert f"SHARD_DONE index=0 started=3 never_started=0 rc={exit_code}" in printed


@pytest.mark.parametrize("foreign", [False, True], ids=["ours", "from-before-the-shard"])
def test_held_device_refuses_remaining_members_and_preserves_baseline_holder(
    tmp_path, monkeypatch, capsys, foreign
):
    proc, killed, calls = tmp_path / "proc", [], []
    if foreign:
        write_process_tree(proc, {4243: ["/dev/neuron1", "/dev/null"], 4244: ["/dev/null"]})

    def record_and_reap(pid):
        killed.append(pid)
        if foreign:
            shutil.rmtree(proc / str(pid))

    monkeypatch.setattr(runner_support, "send_sigkill", record_and_reap)
    appears = lambda: write_process_tree(  # noqa: E731
        proc, {4242: ["/dev/neuron0"]}
    )
    seen, out = run_shard_with_fakes(
        monkeypatch, tmp_path, make_shard(*ALL_MODEL_INDEXES), calls, on_start=appears
    )
    held = 4243 if foreign else 4242
    reason = f"device held by pids [{held}] after {MODEL_IDS[0]} exited"
    assert (seen, killed, len(calls)) == (1, [4242], 1)
    rows = read_rows(out, GRID_FILE_NAME)
    assert [row["status"] for row in rows] == ["pass", "infra_fail", "infra_fail"]
    assert [(row["error"], row["never_started"]) for row in rows[1:]] == [(reason, True)] * 2
    printed = capsys.readouterr().out
    assert f"device_pids=4242 foreign={4243 if foreign else 'none'}" in printed
    assert f"DEVICE_HELD entry=0 pids={held}" in printed
    assert never_started_output(1, reason) in printed
    assert "SHARD_DONE index=0 started=1 never_started=2 rc=1" in printed


@pytest.mark.parametrize("outcome", [0, 2])
def test_held_device_after_last_member_does_not_change_model_exit_code(
    tmp_path, monkeypatch, capsys, outcome
):
    monkeypatch.setattr(runner_support, "send_sigkill", lambda pid: None)
    appears = lambda: write_process_tree(  # noqa: E731
        tmp_path / "proc", {4242: ["/dev/neuron0"]}
    )
    seen, _ = run_shard_with_fakes(
        monkeypatch, tmp_path, make_shard(1), plan={MODEL_IDS[1]: outcome}, on_start=appears
    )
    printed = capsys.readouterr().out
    assert seen == outcome
    assert "DEVICE_HELD entry=1 pids=4242" in printed and "NEVER_STARTED" not in printed
    assert f"SHARD_DONE index=0 started=1 never_started=0 rc={outcome}" in printed


@pytest.mark.parametrize(
    ("site", "plan", "statuses", "model_lines"),
    [
        ("bound", {MODEL_IDS[1]: "timeout"}, ["pass", "infra_fail", "infra_fail"], 2),
        ("guards", {}, ["pass", "xpass", "infra_fail"], 3),
    ],
    ids=["bound", "guards"],
)
def test_deadline_refuses_remaining_members_without_absorbing_expected_failures(
    tmp_path, monkeypatch, capsys, run_identity_env, site, plan, statuses, model_lines
):
    # Use the model timeout or a slow prelaunch check to consume the shard deadline.
    fake_time, cleared = fake_clock(), []
    listed = TARGET._replace(xfail=dict.fromkeys(MODEL_IDS[1:], "known device finding"))

    def slow_third_clear():
        cleared.append(1)
        if site == "guards" and len(cleared) == 3:
            fake_time.sleep(100)
        return ()

    monkeypatch.setattr(shard_runner, "clear_compile_caches", slow_third_clear)
    shard = make_shard(*ALL_MODEL_INDEXES)
    seen, out = run_shard_with_fakes(
        monkeypatch, tmp_path, shard, None, plan, fake_time, timeout=100, target=listed
    )
    rows = read_rows(out, GRID_FILE_NAME)
    assert seen == 124
    assert [row["status"] for row in rows] == statuses
    assert (rows[2]["error"], rows[2]["never_started"]) == (NEVER_STARTED_REASON, True)
    assert "never_started" not in rows[1]
    assert [{key: row.get(key) for key in RUN_IDENTITY_FIELDS} for row in rows] == [
        RUN_IDENTITY_FIELDS
    ] * 3
    assert [
        read_rows(out, ENTRIES_SUBDIR, f"entry-{index}", GRID_FILE_NAME)
        for index in ALL_MODEL_INDEXES
    ] == [[row] for row in rows]
    printed = capsys.readouterr().out
    assert printed.count("SHARD_MODEL entry=") == model_lines
    assert ABSORBED not in printed
    assert never_started_output(2, NEVER_STARTED_REASON) in printed
    assert "SHARD_DONE index=0 started=2 never_started=1 rc=124" in printed
    kept = sorted(path.name for path in (out / ENTRIES_SUBDIR / "entry-0").iterdir())
    assert kept == [BENCHMARK_FILE_NAME, GRID_FILE_NAME, "torchrun_logs"]
    assert [path.name for path in (out / ENTRIES_SUBDIR / "entry-2").iterdir()] == [GRID_FILE_NAME]


def test_prelaunch_cache_clear_can_prevent_every_model_from_starting(tmp_path, monkeypatch, capsys):
    fake_time, calls = fake_clock(), []

    def slow_guard():
        fake_time.sleep(1000)
        return ()

    monkeypatch.setattr(shard_runner, "clear_compile_caches", slow_guard)
    seen, out = run_shard_with_fakes(
        monkeypatch, tmp_path, make_shard(*ALL_MODEL_INDEXES), calls, fake_time=fake_time
    )
    assert (seen, calls) == (124, [])
    printed = capsys.readouterr().out
    assert (
        printed.count("SHARD_MODEL entry=") == 1 and "MODUSCOPE_PROCESS_EXIT_CODE=" not in printed
    )
    reason = "never started: shard deadline reached after 0 of 3 members"
    assert [line for line in printed.splitlines() if line.startswith("NEVER_STARTED")] == [
        f"NEVER_STARTED entry={index} model={MODEL_IDS[index]} reason={reason}"
        for index in ALL_MODEL_INDEXES
    ]
    assert "SHARD_DONE index=0 started=0 never_started=3 rc=124" in printed
    assert [(row["status"], row["error"]) for row in read_rows(out, GRID_FILE_NAME)] == [
        ("infra_fail", reason)
    ] * 3


def test_prelaunch_guards_reduce_the_timeout_but_not_recorded_wall_minutes(
    tmp_path, monkeypatch, capsys
):
    fake_time = fake_clock()

    def slow_clear():
        fake_time.sleep(100.375)
        return ("/tmp/hlo_cache", "/tmp/neff_cache")

    monkeypatch.setattr(shard_runner, "clear_compile_caches", slow_clear)
    plan = {MODEL_IDS[1]: "timeout"}
    seen, out = run_shard_with_fakes(
        monkeypatch, tmp_path, make_shard(*ALL_MODEL_INDEXES), plan=plan, fake_time=fake_time
    )
    printed = capsys.readouterr().out
    assert seen == 124
    assert re.findall(r"^SHARD_MODEL .* remaining=(\d+)s$", printed, re.MULTILINE) == ["900", "800"]
    assert re.findall(r"^Entry \d .*, (\d+)s bound: ", printed, re.MULTILINE) == ["800", "699"]
    guards = "caches_cleared=2 device_pids=none foreign=none"
    assert re.findall(rf"^GUARDS entry=(\d) .* {guards}$", printed, re.MULTILINE) == ["0", "1"]
    rows = read_rows(out, GRID_FILE_NAME)
    assert [row.get("peak_memory_unreliable") for row in rows] == [None, True, None]
    # The second timeout excludes its prelaunch checks and rounds down to whole seconds.
    killed = KILLED_REASON.format("699")
    assert rows[1]["error"] == killed
    assert (
        f"::warning::model {MODEL_CONFIG_TOKEN} {MODEL_IDS[1]} "
        + f"status=infra_fail error={killed}"
        in printed
    )
    assert [row.get("wall_minutes") for row in rows] == [0.0, round(699.25 / 60, 3), None]
    assert rows[2]["never_started"] is True


def test_failed_row_update_emits_one_token_and_continues_the_shard(tmp_path, monkeypatch, capsys):
    write_rows = shard_runner.write_grid_rows

    def full_disk(path, rows):
        if any("wall_minutes" in row for row in rows):
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), path)
        write_rows(path, rows)

    monkeypatch.setattr(shard_runner, "write_grid_rows", full_disk)
    seen, out = run_shard_with_fakes(monkeypatch, tmp_path, make_shard(*ALL_MODEL_INDEXES))
    printed = capsys.readouterr().out
    assert seen == 0
    grids = [
        out / ENTRIES_SUBDIR / f"entry-{index}" / GRID_FILE_NAME for index in ALL_MODEL_INDEXES
    ]
    field_names = ["wall_minutes", *["wall_minutes,peak_memory_unreliable"] * 2]
    assert re.findall(r"^STAMP_FAILED (.*)$", printed, re.MULTILINE) == [
        f"entry={entry_index} fields={fields} reason=OSError: [Errno {errno.ENOSPC}] "
        + f"{os.strerror(errno.ENOSPC)}: '{grid}'"
        for entry_index, grid, fields in zip(ALL_MODEL_INDEXES, grids, field_names, strict=True)
    ]
    assert "SHARD_DONE index=0 started=3 never_started=0 rc=0" in printed
    assert not any("wall_minutes" in row for row in read_rows(out, GRID_FILE_NAME))


def test_aggregate_grids_marks_a_missing_entry_as_infra_fail(tmp_path, capsys):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    missing_entry = tmp_path / "entry-1"

    counts = shard_runner.aggregate_grids(
        str(output_dir), [(1, MODEL_IDS[1], str(missing_entry))], None
    )

    assert counts == (1, 1)
    (row,) = read_rows(output_dir, GRID_FILE_NAME)
    assert (row["grid_entry_name"], row["status"], row["mfu(%)"]) == (
        MODEL_IDS[1],
        "infra_fail",
        -1,
    )
    unreadable_grid = f"UNREADABLE_GRID entry=1 path={missing_entry / GRID_FILE_NAME}"
    assert unreadable_grid in capsys.readouterr().out


STARTED = ("SHARD_MODEL", "MODUSCOPE_PROCESS_EXIT_CODE", "ENTRY_GRID")
REFUSED = ("NEVER_STARTED", "::warning::")


@pytest.mark.parametrize(
    ("held", "indexes", "exit_code", "kinds", "plain"),
    [
        (
            False,
            ALL_MODEL_INDEXES,
            124,
            [
                *STARTED,
                *REMOVAL_TOKEN_KEYS,
                "GUARDS",
                *STARTED,
                "::warning::",
                *REMOVAL_TOKEN_KEYS,
                "GUARDS",
                *REFUSED,
            ],
            ["Entry", "Entry"],
        ),
        (
            True,
            (0, 1),
            1,
            [*STARTED, *REMOVAL_TOKEN_KEYS, "GUARDS", "DEVICE_HELD", *REFUSED],
            ["Entry"],
        ),
    ],
    ids=["deadline-after-a-timeout", "held-device"],
)
def test_recorded_token_lines_keep_print_order(
    tmp_path, monkeypatch, capsys, held, indexes, exit_code, kinds, plain
):
    plan, hook = {MODEL_IDS[1]: "timeout"}, None
    if held:
        plan = {}
        monkeypatch.setattr(runner_support, "send_sigkill", lambda pid: None)
        hook = lambda: write_process_tree(  # noqa: E731
            tmp_path / "proc", {4242: ["/dev/neuron0"]}
        )
    seen, _ = run_shard_with_fakes(
        monkeypatch, tmp_path, make_shard(*indexes), plan=plan, on_start=hook, timeout=100
    )
    assert seen == exit_code
    printed = capsys.readouterr().out.splitlines()
    tokens = list(process._token_lines)
    assert [TOKEN_KIND_PATTERN.match(token).group() for token in tokens] == [
        "SHARD",
        "TORCH_COMPILE_ENABLED",
        *kinds,
        "AGGREGATE_GRID_ROWS",
        "SHARD_DONE",
    ]
    assert [line for line in printed if line in set(tokens)] == tokens
    assert [line.split(" ", 1)[0] for line in printed if line not in tokens] == [
        "RUN_IDENTITY=absent",
        *plain,
    ]
