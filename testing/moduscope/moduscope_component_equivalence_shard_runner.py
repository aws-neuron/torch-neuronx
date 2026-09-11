"""Run one component-equivalence shard.

The pod entry point calls this module for a target from a pod.
"""

from __future__ import annotations

import math
import os
import shutil
import stat
from collections.abc import Mapping
from pathlib import Path

from moduscope_artifact_layout import (
    GRID_FILE_NAME,
    add_run_identity,
    load_grid_rows,
    write_grid_rows,
)
from moduscope_entry_results import (
    PASSING_STATUSES,
    all_rows_passing,
    infra_fail_row,
    status_after_expected_failure,
)
from moduscope_process import TestInvocation, emit_token_line
from moduscope_runner_support import ShardRunState, aggregate_member_rows, run_member_process
from moduscope_targets import ShardRunContext, Target

# The pod suite uses this fixed path. Its entry point requires "1" as its only argument.
SUITE_DIR = "/workspace/ModuScope/test/component_tests_torchtitan"
RUN_ALL_MATRICES_ARGUMENT = "1"
SCRIPTS = ("run_all_matrices.sh", os.path.join("bf16", "tests", "run_all.sh"))
RESULTS_SUBDIR = "component-equiv"
REPORT_FILE_NAME = "report.json"
NUM_WORKERS = "64"
# run_all_matrices.sh exits with this code when a device summary is missing or not PASS.
FAIL_VERDICT_EXIT_CODE = 1


def _counts_meet_threshold(counts: dict, threshold: float | None) -> bool:
    count_values = []
    for count_name in ("passed", "failed", "error"):
        count = counts.get(count_name, 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"component counts must be nonnegative integers, got {counts!r}")
        count_values.append(count)
    passed, failed, errors = count_values
    graded_count = passed + failed + errors
    if graded_count == 0:
        return False
    return (
        passed / graded_count * 100 >= threshold
        if threshold is not None
        else passed == graded_count
    )


def _component_row_status(row: dict) -> str:
    threshold = row.get("threshold")
    by_model = row.get("by_model")
    by_module = row.get("by_module")
    try:
        if not isinstance(by_model, dict) or not by_model:
            raise ValueError("by_model must be a nonempty object")
        if "by_module" in row and not isinstance(by_module, dict):
            raise ValueError("by_module must be an object when present")
        if isinstance(threshold, bool):
            raise ValueError(f"threshold must be a percentage, got {threshold!r}")
        normalized_threshold = float(threshold) if threshold is not None else None
        if normalized_threshold is not None and (
            not math.isfinite(normalized_threshold) or not 0 <= normalized_threshold <= 100
        ):
            raise ValueError(
                f"threshold must be a finite percentage from 0 to 100, got {threshold!r}"
            )
        threshold_results = [
            _counts_meet_threshold(counts, normalized_threshold)
            for result_group in (by_model, by_module or {})
            for counts in result_group.values()
        ]
    except (AttributeError, TypeError, ValueError, OverflowError) as metrics_error:
        row["error"] = f"invalid component-equivalence metrics: {metrics_error}"
        return "infra_fail"
    passes_thresholds = row.get("threshold_result") == "PASS" and all(threshold_results)
    return "pass" if passes_thresholds else "fail"


def build_test_invocation(target: Target, member_name: str) -> TestInvocation:
    """Build the component-suite process launch."""
    return TestInvocation(
        argv=("bash", "run_all_matrices.sh", RUN_ALL_MATRICES_ARGUMENT),
        cwd=SUITE_DIR,
        env={
            "MODEL_CONFIG": member_name,
            "TORCH_COMPILE": str(int(target.torch_compile_enabled)),
            "NUM_WORKERS": NUM_WORKERS,
        },
        unset_env=("DEVICES", "TEST_FILES"),
    )


def make_suite_scripts_executable(suite_dir: str) -> None:
    """Make each existing suite script executable."""
    for relative_script_path in SCRIPTS:
        script_path = os.path.join(suite_dir, relative_script_path)
        if os.path.isfile(script_path):
            os.chmod(
                script_path,
                os.stat(script_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
            )


def collect_suite_results(suite_dir: str, output_dir: str) -> str:
    """Collect the component suite result tree under the pod output directory."""
    results_destination = os.path.join(output_dir, RESULTS_SUBDIR)
    suite_logs_dir = os.path.join(suite_dir, "logs")
    if os.path.isdir(suite_logs_dir):
        shutil.copytree(suite_logs_dir, results_destination, dirs_exist_ok=True)
    else:
        os.makedirs(results_destination, exist_ok=True)
    return results_destination


def member_rows_from_grids(
    results_root: str, member_name: str, expected_to_fail: bool, process_passed: bool, run_identity
) -> list[dict]:
    """Return one member's rows from its collected device grids with their final statuses."""
    member_rows = []
    for grid_path in sorted(Path(results_root, member_name).rglob(GRID_FILE_NAME)):
        grid_rows = load_grid_rows(str(grid_path))
        if grid_rows is None:
            print(f"UNREADABLE_GRID path={grid_path}")
            device_name, _, compiled_marker = grid_path.parent.name.partition("-")
            failure_row = infra_fail_row(
                "The component suite result grid is unreadable. Check the suite log.",
                identity=run_identity,
                model_config=member_name,
                device=device_name,
                compile=bool(compiled_marker),
            )
            member_rows.append(failure_row)
            continue
        for row in grid_rows:
            row["model_config"] = member_name
            row["status"] = status_after_expected_failure(
                _component_row_status(row), expected_to_fail, process_passed
            )
            add_run_identity(row, run_identity)
            member_rows.append(row)
    return member_rows


def write_component_shard_grid(
    output_dir: str,
    results_root: str,
    members: tuple[str, ...],
    member_errors: dict[str, tuple[str, bool]],
    run_identity,
    expected_failures: Mapping[str, str],
    member_exit_codes: Mapping[str, int],
) -> tuple[int, bool]:
    """Combine each member's readable device grids.

    Add the model configuration, status, and run identity. Apply the expected-failure
    rules to the rows of each member listed in ``expected_failures``.
    """
    member_row_batches = []
    for member_name in members:
        member_rows = member_rows_from_grids(
            results_root,
            member_name,
            member_name in expected_failures,
            member_exit_codes.get(member_name) == 0,
            run_identity,
        )
        if not member_rows:
            error, never_started = member_errors.get(
                member_name,
                ("component suite wrote no readable result rows. Check the suite log.", False),
            )
            failure_row = infra_fail_row(
                error, identity=run_identity, never_started=never_started, model_config=member_name
            )
            member_rows.append(failure_row)
        member_row_batches.append(member_rows)

    combined_rows = aggregate_member_rows(member_row_batches)
    write_grid_rows(os.path.join(output_dir, GRID_FILE_NAME), combined_rows)
    has_failed_rows = any(row["status"] not in PASSING_STATUSES for row in combined_rows)
    return len(combined_rows), has_failed_rows


def run_shard(context: ShardRunContext) -> int:
    """Run one component-equivalence shard and return its exit code."""
    target, shard, output_dir, run_identity, deadline = context
    make_suite_scripts_executable(SUITE_DIR)
    results_root = os.path.join(output_dir, RESULTS_SUBDIR)
    os.makedirs(results_root, exist_ok=True)
    member_errors = {}
    member_exit_codes = {}
    run_state = ShardRunState(len(shard.members))
    for member_index, member_name in enumerate(shard.members):
        remaining_seconds, failure_reason = run_state.check_member_start(deadline)
        if failure_reason:
            member_errors[member_name] = (failure_reason, True)
            continue
        config_exit_code = run_member_process(
            build_test_invocation(target, member_name),
            remaining_seconds,
            f"Running in {SUITE_DIR}: ",
            f"Component suite exceeded {remaining_seconds}s and was killed.",
        ).returncode
        collect_suite_results(SUITE_DIR, output_dir)
        member_exit_codes[member_name] = config_exit_code
        listed = member_name in target.xfail
        absorbed = (
            listed
            and config_exit_code == FAIL_VERDICT_EXIT_CODE
            and all_rows_passing(
                member_rows_from_grids(
                    results_root, member_name, listed, config_exit_code == 0, run_identity
                )
            )
        )
        run_state.finish_member(member_index, member_name, 0 if absorbed else config_exit_code, 0)
        if config_exit_code:
            member_errors[member_name] = (
                f"component suite exited with code {config_exit_code}. Check the suite log.",
                False,
            )
    collected_report_count = sum(1 for _ in Path(results_root).rglob(REPORT_FILE_NAME))
    grid_row_count, has_failed_rows = write_component_shard_grid(
        output_dir,
        results_root,
        shard.members,
        member_errors,
        run_identity,
        target.xfail,
        member_exit_codes,
    )
    if has_failed_rows:
        run_state.returncode = run_state.returncode or 1
    emit_token_line(f"COMPONENT_EQUIV_RC={run_state.returncode}")
    emit_token_line(f"COLLECTED_REPORTS={collected_report_count}")
    emit_token_line(f"AGGREGATE_GRID_ROWS={grid_row_count}")
    return run_state.returncode
