"""Pytest calls this module to test the Titan shard runner."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import moduscope_entry_results as grid_results
import moduscope_process as entry_results
import moduscope_runner_support as runner_support
import moduscope_titan_shard_runner as titan_runner
import pytest
from moduscope_shard_planner import Shard
from moduscope_targets import Target

from conftest import RUN_IDENTITY_FIELDS, read_rows_at, run_test_shard, write_bytes, write_rows

pytestmark = pytest.mark.usefixtures("compile_cache_directories")

TARGET_CONFIG = "e_titan_multicore_async"
TITAN_ENTRY_NAMES = ("Titan_Model_0", "Titan_Model_1")
INSTALLED_TITAN_ENTRY_NAMES = ("Titan_Unselected", *TITAN_ENTRY_NAMES)
TARGET = Target(
    name="test_titan",
    timeout_minutes=70,
    adapter="titan",
    config=TARGET_CONFIG,
    shards=tuple(Shard(index, (entry_name,)) for index, entry_name in enumerate(TITAN_ENTRY_NAMES)),
)
EXPECTED_FAILURE_TARGET = TARGET._replace(
    xfail={"Titan_Model_0": "known finding", "Titan_Model_1": "slow"}
)
EXIT_CODE_3_REASON = "ModuScope exited with code 3"
EXIT_CODE_124_REASON = "ModuScope exited with code 124"
KILLED_900_REASON = "killed by the runner after 900s (exit code 124)"
PASSING_ROW = {"grid_entry_name": "Titan_b_Async", "grid_entry_index": 1, "status": "pass"}
RUNNING_ROW = {**PASSING_ROW, "status": "running"}
ERROR_ROW = {**PASSING_ROW, "status": "error"}
FALLBACK_ROW = {"grid_entry_name": "Titan_Model_1", "grid_entry_index": 1, "mfu(%)": -1}
FALLBACK_ROW.update(status="infra_fail", error=EXIT_CODE_3_REASON)
ZERO_EXIT_FALLBACK_ROW = {
    **FALLBACK_ROW,
    "error": "ModuScope produced no usable grid rows and exited with code 0.",
}
PRUNED_DIR_PATHS = (
    "manifests",
    "xla_dump",
    "individual/Cfg.0/xla_dump",
    "Titan_x/_profile_upload_root",
)
KEPT_LOOKALIKE_PATHS = (
    "Titan_x/_profile_upload_root_keep",
    "manifests_keep",
    "torchrun_logs/xla_dump",
)
STDOUT_ONLY_PREFIXES = ("PRUNED_DIRS=", "TB_COPIED_FILES=")


@pytest.fixture(autouse=True)
def fake_runner_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(
        titan_runner, "installed_config_field", lambda _config, _field_name: TITAN_ENTRY_NAMES
    )
    monkeypatch.setattr(runner_support.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(runner_support, "PROC_ROOT", str(tmp_path / "proc"))


def run_titan_entry(tmp_path, monkeypatch, index=0, exit_code=0, target=TARGET, timeout=2100):
    monkeypatch.setattr(entry_results, "run_in_process_group", lambda *args, **kwargs: exit_code)
    return run_test_shard(titan_runner, target, target.shards[index], tmp_path, timeout)


def lines_with_prefix(output_text, prefix):
    return [
        output_line for output_line in output_text.splitlines() if output_line.startswith(prefix)
    ]


def command_entry_index(argv):
    argument = next(argument for argument in argv if "grid_entry_list" in argument)
    return int(argument.split("[", 1)[1].rstrip("]"))


def write_passing_entry(output_dir, entry_index):
    write_rows(
        Path(output_dir, f"e{entry_index}", "grid_metrics.json"),
        [
            {
                **PASSING_ROW,
                "grid_entry_name": TITAN_ENTRY_NAMES[entry_index],
                "grid_entry_index": entry_index,
            }
        ],
    )


def raise_no_space(*args, **kwargs):
    raise OSError(28, "No space left on device")


def test_build_test_invocation_matches_the_entry_command_contract():
    assert titan_runner.build_test_invocation(TARGET, "/out", 2) == entry_results.TestInvocation(
        argv=(
            sys.executable,
            "-m",
            "cli_scripts.task_runner",
            f"--config-name={TARGET_CONFIG}",
            "hydra.run.dir=/out/e2",
            "++task_config.artifact_directory=/out/e2",
            "+hydra.overwrites.grid_entry_list=[2]",
            "++task_config.clean_artifacts_on_success=false",
        ),
        cwd=None,
        env=None,
    )


def test_build_test_invocation_rejects_torch_compile():
    compile_target = TARGET._replace(torch_compile_enabled=True)

    with pytest.raises(ValueError, match="Titan does not support torch_compile_enabled=True"):
        titan_runner.build_test_invocation(compile_target, "/out", 0)


def test_member_name_selects_the_canonical_wheel_index(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(
        titan_runner,
        "installed_config_field",
        lambda _config, _field_name: INSTALLED_TITAN_ENTRY_NAMES,
    )
    monkeypatch.setattr(
        entry_results,
        "run_in_process_group",
        lambda argv, *args, **kwargs: seen.append((argv, args, kwargs)) or 0,
    )

    assert run_test_shard(titan_runner, TARGET, TARGET.shards[1], tmp_path, 900) == 1
    invocation = titan_runner.build_test_invocation(TARGET, str(tmp_path), 2)
    assert seen == [(invocation.argv, (900,), {"cwd": None, "env": os.environ})]


def test_missing_member_is_rejected_before_the_child_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(
        entry_results, "run_in_process_group", lambda *args, **kwargs: pytest.fail("the child ran")
    )
    missing_shard = Shard(7, ("Titan_Model_0", "Titan_Missing"))

    with pytest.raises(
        ValueError, match="Titan member 'Titan_Missing' is absent from installed config"
    ):
        run_test_shard(titan_runner, TARGET, missing_shard, tmp_path, 900)


def test_grouped_shard_runs_each_member_and_keeps_the_first_failure(monkeypatch, tmp_path):
    grouped_shard = Shard(0, TITAN_ENTRY_NAMES)
    calls = []

    def run_entry(argv, timeout_seconds, **kwargs):
        entry_index = command_entry_index(argv)
        calls.append((entry_index, timeout_seconds, kwargs))
        write_passing_entry(tmp_path, entry_index)
        return 3 if entry_index == 0 else 0

    monkeypatch.setattr(entry_results, "run_in_process_group", run_entry)

    assert run_test_shard(titan_runner, TARGET, grouped_shard, tmp_path, 900) == 3
    assert calls == [
        (0, 900.0, {"cwd": None, "env": os.environ}),
        (1, 900.0, {"cwd": None, "env": os.environ}),
    ]
    assert [row["grid_entry_name"] for row in read_rows_at(tmp_path / "grid_metrics.json")] == list(
        TITAN_ENTRY_NAMES
    )


def test_grouped_shard_records_members_that_miss_the_deadline(monkeypatch, tmp_path, capsys):
    now = [0.0]
    calls = []
    monkeypatch.setattr(runner_support.time, "monotonic", lambda: now[0])

    def clear_caches():
        if calls:
            now[0] = 10.0
        return ()

    def run_entry(argv, _timeout_seconds, **_kwargs):
        entry_index = command_entry_index(argv)
        calls.append(entry_index)
        write_passing_entry(tmp_path, entry_index)
        now[0] = 9.0
        return 0

    monkeypatch.setattr(titan_runner, "clear_compile_caches", clear_caches)
    monkeypatch.setattr(entry_results, "run_in_process_group", run_entry)

    shard = Shard(0, TITAN_ENTRY_NAMES)
    assert run_test_shard(titan_runner, EXPECTED_FAILURE_TARGET, shard, tmp_path, 10) == 124
    assert calls == [0]
    rows = read_rows_at(tmp_path / "grid_metrics.json")
    assert rows[0]["status"] == "xpass"
    assert rows[1] == {
        "grid_entry_name": "Titan_Model_1",
        "grid_entry_index": 1,
        "mfu(%)": -1,
        "status": "infra_fail",
        "error": "never started: shard deadline reached after 1 of 2 members",
        "never_started": True,
    }
    assert (
        "NEVER_STARTED entry=1 model=Titan_Model_1 "
        + "reason=never started: shard deadline reached after 1 of 2 members"
        in capsys.readouterr().out
    )


def test_grouped_shard_refuses_members_after_a_held_device(monkeypatch, tmp_path, capsys):
    process_root = Path(runner_support.PROC_ROOT)
    calls = []

    def run_entry(argv, _timeout_seconds, **_kwargs):
        entry_index = command_entry_index(argv)
        calls.append(entry_index)
        write_passing_entry(tmp_path, entry_index)
        holder_dir = process_root / "4242" / "fd"
        holder_dir.mkdir(parents=True)
        (holder_dir / "3").symlink_to("/dev/neuron0")
        return 0

    monkeypatch.setattr(entry_results, "run_in_process_group", run_entry)
    monkeypatch.setattr(runner_support, "DEVICE_GRACE_SECONDS", 0.0)
    monkeypatch.setattr(runner_support, "send_sigkill", lambda _process_id: None)

    assert run_test_shard(titan_runner, TARGET, Shard(0, TITAN_ENTRY_NAMES), tmp_path, 900) == 1
    assert calls == [0]
    rows = read_rows_at(tmp_path / "grid_metrics.json")
    assert rows[0]["status"] == "pass"
    assert rows[1]["status"] == "infra_fail"
    assert rows[1]["error"] == "device held by pids [4242] after Titan_Model_0 exited"
    assert rows[1]["never_started"] is True
    assert "DEVICE_HELD entry=0 pids=4242" in capsys.readouterr().out


def test_compile_request_is_rejected_before_entry_resolution(monkeypatch, tmp_path):
    monkeypatch.setattr(
        titan_runner,
        "installed_config_field",
        lambda _config, _field_name: pytest.fail("the installed config was read"),
    )
    compile_target = TARGET._replace(torch_compile_enabled=True)

    with pytest.raises(ValueError, match="Titan does not support torch_compile_enabled=True"):
        run_test_shard(titan_runner, compile_target, compile_target.shards[0], tmp_path, 900)


def test_prune_removes_only_compiler_and_profile_directories(monkeypatch, tmp_path, capsys):
    e0 = tmp_path / "e0"
    row = {"grid_entry_name": "Titan_x_Async", "grid_entry_index": 0, "status": "pass"}
    write_rows(e0 / "grid_metrics.json", [row])
    for relative_path, size in zip(
        PRUNED_DIR_PATHS, (1_000_000, 2_000_000, 500_000, 1_000_000), strict=True
    ):
        write_bytes(e0 / relative_path / "TitanTask" / "payload.bin", size)
    for relative_path in KEPT_LOOKALIKE_PATHS:
        write_bytes(e0 / relative_path, 10)
    assert run_titan_entry(tmp_path, monkeypatch, exit_code=3) == 3
    assert [
        (e0 / relative_path).exists() for relative_path in PRUNED_DIR_PATHS + KEPT_LOOKALIKE_PATHS
    ] == [False] * 4 + [True] * 3
    assert (e0 / "individual" / "Cfg.0").is_dir()
    assert lines_with_prefix(capsys.readouterr().out, "PRUNED_DIRS=") == ["PRUNED_DIRS=4"]
    write_bytes(e0 / "xla_dump" / "again.txt", 10)
    monkeypatch.setattr(entry_results, "run_in_process_group", lambda *args, **kwargs: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        run_test_shard(titan_runner, TARGET, TARGET.shards[0], tmp_path, 900)
    assert not (e0 / "xla_dump").exists()


@pytest.mark.parametrize(
    ("initial", "codes", "target", "expected", "marked", "annotation"),
    [
        ([RUNNING_ROW], ("timeout", 124), TARGET,
         {**RUNNING_ROW, "status": "fail", "error": KILLED_900_REASON}, 1,
         f"::warning::model Titan_b_Async status=fail error={KILLED_900_REASON}"),
        ([RUNNING_ROW], (124, 124), TARGET,
         {**RUNNING_ROW, "status": "fail", "error": EXIT_CODE_124_REASON}, 1,
         f"::warning::model Titan_b_Async status=fail error={EXIT_CODE_124_REASON}"),
        (None, (3, 3), EXPECTED_FAILURE_TARGET, FALLBACK_ROW, 1,
         f"::warning::model Titan_Model_1 status=infra_fail error={EXIT_CODE_3_REASON}"),
        ([ERROR_ROW], (3, 3), EXPECTED_FAILURE_TARGET, ERROR_ROW, 0,
         "::warning::model Titan_b_Async status=error error="),
        ([PASSING_ROW], (0, 0), EXPECTED_FAILURE_TARGET, {**PASSING_ROW, "status": "xpass"}, 0,
         "::warning::model Titan_b_Async xpass: prune its xfail entry")
    ],
    ids=["skeleton-killed", "self-exit-124", "listed-no-grid", "listed-error", "listed-pass"]
)  # fmt: skip
def test_titan_entry_outcomes_update_rows_and_emit_matching_tokens(
    monkeypatch, tmp_path, capsys, initial, codes, target, expected, marked, annotation
):
    process_exit_code, runner_exit_code = codes
    grid = tmp_path / "e1" / "grid_metrics.json"
    if initial is not None:
        write_rows(grid, initial)
    seen = []

    def fake_run(argv, bound, **kwargs):
        seen.append((argv, bound, kwargs))
        if process_exit_code == "timeout":
            raise subprocess.TimeoutExpired(argv, bound)
        return process_exit_code

    monkeypatch.setattr(entry_results, "run_in_process_group", fake_run)
    assert run_test_shard(titan_runner, target, target.shards[1], tmp_path, 900) == runner_exit_code
    invocation = titan_runner.build_test_invocation(target, str(tmp_path), 1)
    assert seen == [(invocation.argv, 900, {"cwd": None, "env": os.environ})]
    assert read_rows_at(grid) == [expected]
    assert entry_results._token_lines == [
        f"ENTRY_GRID entry=1 rows=1 marked_failed={marked} profile_s3_url=none",
        annotation,
        "GUARDS entry=1 model=Titan_Model_1 caches_cleared=0 " + "device_pids=none foreign=none",
        "AGGREGATE_GRID_ROWS=1",
    ]
    out = capsys.readouterr().out
    assert [
        line for line in out.splitlines() if line in entry_results._token_lines
    ] == entry_results._token_lines
    reason = (
        KILLED_900_REASON
        if process_exit_code == "timeout"
        else f"ModuScope exited with code {process_exit_code}"
    )
    assert lines_with_prefix(out, "Entry 1: ") == [
        "Entry 1: cleared 0 cache dir(s): none",
        f"Entry 1: {reason}",
    ]
    for stdout_only in STDOUT_ONLY_PREFIXES:
        assert lines_with_prefix(out, stdout_only) and not lines_with_prefix(
            "\n".join(entry_results._token_lines), stdout_only
        )
    assert not (
        tmp_path / "tokens.txt"
    ).exists(), "moduscope_pod_runner.py writes it, not the runner"


def test_shard_grid_copies_only_the_selected_entry_grid(monkeypatch, tmp_path, capsys):
    write_rows(tmp_path / "e1" / "grid_metrics.json", [PASSING_ROW])
    write_rows(tmp_path / "other" / "grid_metrics.json", [{"grid_entry_name": "not-an-entry"}])
    assert run_titan_entry(tmp_path, monkeypatch, index=1) == 0
    assert read_rows_at(tmp_path / "grid_metrics.json") == [PASSING_ROW]
    assert "AGGREGATE_GRID_ROWS=1" in capsys.readouterr().out
    (tmp_path / "e1" / "grid_metrics.json").write_text("{ truncated")
    assert run_titan_entry(tmp_path, monkeypatch, index=1) == 1
    assert read_rows_at(tmp_path / "grid_metrics.json") == [ZERO_EXIT_FALLBACK_ROW]
    assert "AGGREGATE_GRID_ROWS=1" in capsys.readouterr().out


def test_zero_child_exit_rejects_a_missing_grid(monkeypatch, tmp_path):
    assert run_titan_entry(tmp_path, monkeypatch, index=1) == 1
    assert read_rows_at(tmp_path / "grid_metrics.json") == [ZERO_EXIT_FALLBACK_ROW]


@pytest.mark.parametrize("status", [None, "", "running", "fail", "error", "infra_fail"])
def test_zero_child_exit_rejects_nonpassing_status(monkeypatch, tmp_path, status):
    row = {**PASSING_ROW, "status": status}
    write_rows(tmp_path / "e1" / "grid_metrics.json", [row])
    assert run_titan_entry(tmp_path, monkeypatch, index=1) == 1
    assert read_rows_at(tmp_path / "grid_metrics.json") == [row]


@pytest.mark.parametrize("status", ["pass", "xfail", "xpass"])
def test_zero_child_exit_accepts_passing_status(monkeypatch, tmp_path, status):
    row = {**PASSING_ROW, "status": status}
    write_rows(tmp_path / "e1" / "grid_metrics.json", [row])
    assert run_titan_entry(tmp_path, monkeypatch, index=1) == 0
    assert read_rows_at(tmp_path / "grid_metrics.json") == [row]


def test_titan_rows_add_run_identity_without_overwriting_profile_urls(
    run_identity_env, monkeypatch, tmp_path, capsys
):
    e1 = tmp_path / "e1"
    write_bytes(e1 / "Titan_b_Async" / "_profile_upload_root" / "neuron_profile" / "trace.json", 2)
    explorer = "https://profile-viewer.example/profile/x_2026"
    rows = [PASSING_ROW, {**PASSING_ROW, "profile_s3_url": explorer}]
    write_rows(e1 / "grid_metrics.json", rows)
    assert run_titan_entry(tmp_path, monkeypatch, index=1) == 0
    assert read_rows_at(e1 / "grid_metrics.json") == [
        {**row, **RUN_IDENTITY_FIELDS} for row in rows
    ]
    assert lines_with_prefix(capsys.readouterr().out, "ENTRY_GRID ") == [
        "ENTRY_GRID entry=1 rows=2 marked_failed=0 profile_s3_url=none"
    ]


def test_finalization_write_failure_is_not_absorbed_by_xfail(
    run_identity_env, monkeypatch, tmp_path, capsys
):
    grid = write_rows(tmp_path / "e1" / "grid_metrics.json", [PASSING_ROW])
    monkeypatch.setattr(grid_results.artifact_layout, "write_grid_rows", raise_no_space)
    assert (
        run_titan_entry(tmp_path, monkeypatch, index=1, exit_code=3, target=EXPECTED_FAILURE_TARGET)
        == 1
    )
    assert read_rows_at(grid) == [PASSING_ROW]
    assert read_rows_at(tmp_path / "grid_metrics.json") == [PASSING_ROW]
    assert lines_with_prefix(capsys.readouterr().out, "ENTRY_GRID ") == [
        "ENTRY_GRID entry=1 write_failed reason=OSError: [Errno 28] No space left on device"
    ]


def test_expected_failure_rows_become_xfail_or_xpass_and_can_absorb_exit_code(
    monkeypatch, tmp_path, capsys
):
    grid = tmp_path / "e1" / "grid_metrics.json"
    failed = {**PASSING_ROW, "status": "fail", "error": "loss is nan"}
    write_rows(grid, [PASSING_ROW, failed])
    assert run_titan_entry(tmp_path, monkeypatch, index=1, target=EXPECTED_FAILURE_TARGET) == 0
    assert [row["status"] for row in read_rows_at(grid)] == ["xpass", "xfail"]
    assert lines_with_prefix(capsys.readouterr().out, "::") == [
        "::warning::model Titan_b_Async xpass: prune its xfail entry",
        "::notice::model Titan_b_Async status=xfail reason=slow error=loss is nan",
    ]
    write_rows(grid, [PASSING_ROW, failed])
    assert (
        run_titan_entry(tmp_path, monkeypatch, index=1, exit_code=3, target=EXPECTED_FAILURE_TARGET)
        == 0
    )
    assert [row["status"] for row in read_rows_at(grid)] == ["pass", "xfail"]
    write_rows(grid, [PASSING_ROW, failed])
    assert run_titan_entry(tmp_path, monkeypatch, index=1, exit_code=3) == 3
    assert [row["status"] for row in read_rows_at(grid)] == ["pass", "fail"]


def test_tensorboard_copy_failure_preserves_exit_code_and_grid(monkeypatch, tmp_path, capsys):
    session = tmp_path / "outputs" / "m" / "c" / "tb" / "20260902-1200"
    monkeypatch.setattr(titan_runner, "torchtitan_outputs_dir", lambda: str(tmp_path / "outputs"))
    monkeypatch.setattr(
        entry_results,
        "run_in_process_group",
        lambda *args, **kwargs: write_bytes(session / "events", 1) or 3,
    )
    monkeypatch.setattr(titan_runner.shutil, "copytree", raise_no_space)
    assert run_test_shard(titan_runner, TARGET, TARGET.shards[1], tmp_path, 900) == 3
    assert read_rows_at(tmp_path / "e1" / "grid_metrics.json") == [FALLBACK_ROW]
    out = capsys.readouterr().out
    assert lines_with_prefix(out, "TB_COPIED_FILES=") == [
        "TB_COPIED_FILES=0 reason=OSError: [Errno 28] No space left on device"
    ]
    assert lines_with_prefix(out, "ENTRY_GRID ") == [
        "ENTRY_GRID entry=1 rows=1 marked_failed=1 profile_s3_url=none"
    ]


def test_run_shard_clears_entry_caches_before_start(
    compile_cache_directories, monkeypatch, tmp_path, capsys
):
    hlo = compile_cache_directories / "hlo_cache"
    hlo.mkdir()
    seen = []
    monkeypatch.setattr(
        entry_results,
        "run_in_process_group",
        lambda *args, **kwargs: seen.append(hlo.exists()) or 0,
    )
    assert run_test_shard(titan_runner, TARGET, TARGET.shards[0], tmp_path, 900) == 1
    assert seen == [False]
    assert lines_with_prefix(capsys.readouterr().out, "Entry 0: cleared ") == [
        f"Entry 0: cleared 1 cache dir(s): {hlo}"
    ]


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        (None, "torchtitan is not importable, so the TorchTitan output directory is unknown"),
        ("nowhere", "no outputs tree at {path}"),
    ],
)
def test_missing_torchtitan_outputs_tree_reports_one_tensorboard_line(
    monkeypatch, tmp_path, capsys, source, reason
):
    path = None if source is None else str(tmp_path / source)
    monkeypatch.setattr(titan_runner, "torchtitan_outputs_dir", lambda: path)
    assert run_titan_entry(tmp_path, monkeypatch) == 1
    (line,) = lines_with_prefix(capsys.readouterr().out, "TB_COPIED_FILES=")
    assert line == f"TB_COPIED_FILES=0 reason={reason.format(path=path)}"
    assert not (tmp_path / "e0" / "tb").exists()


def test_torchtitan_outputs_dir_uses_package_location_without_import(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torchtitan", None)
    assert titan_runner.torchtitan_outputs_dir() is None
    monkeypatch.delitem(sys.modules, "torchtitan")
    (tmp_path / "torchtitan").mkdir()
    (tmp_path / "torchtitan" / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert titan_runner.torchtitan_outputs_dir() == str(tmp_path / "outputs")
    assert "torchtitan" not in sys.modules


def test_packed_shard_keeps_each_members_tensorboard_sessions_separate(
    monkeypatch, tmp_path, capsys
):
    outputs = tmp_path / "NeuronTorchTitan" / "outputs"
    stale = outputs / "qwen3" / "qwen3_0_6b" / "tb" / "20260901-0900"
    write_bytes(stale / "events.out.tfevents.0", 3)
    session_name = "20260902-1200"
    session = outputs / "llama3" / "shared" / "tb" / session_name
    write_bytes(session.parent / "stray.txt", 1)

    def run_entry(argv, _timeout_seconds, **_kwargs):
        entry_index = command_entry_index(argv)
        event_path = session / f"events.out.tfevents.{entry_index}"
        write_bytes(event_path, 1)
        if entry_index:
            later_time = (session / "events.out.tfevents.0").stat().st_mtime + 10
            os.utime(event_path, (later_time, later_time))
        write_passing_entry(tmp_path, entry_index)
        return 0

    monkeypatch.setattr(titan_runner, "torchtitan_outputs_dir", lambda: str(outputs))
    monkeypatch.setattr(entry_results, "run_in_process_group", run_entry)

    assert run_test_shard(titan_runner, TARGET, Shard(0, TITAN_ENTRY_NAMES), tmp_path, 900) == 0
    for entry_index in range(2):
        tensorboard_dir = tmp_path / f"e{entry_index}" / "tb"
        assert sorted(
            path.relative_to(tensorboard_dir).as_posix() for path in tensorboard_dir.rglob("*")
        ) == [
            session_name,
            *(
                f"{session_name}/events.out.tfevents.{event_index}"
                for event_index in range(entry_index + 1)
            ),
        ]
        assert not (tensorboard_dir / "20260901-0900").exists()
        assert not (tensorboard_dir / "stray.txt").exists()
    out = capsys.readouterr().out
    assert lines_with_prefix(out, "TB_COPIED_FILES=") == ["TB_COPIED_FILES=1", "TB_COPIED_FILES=2"]
