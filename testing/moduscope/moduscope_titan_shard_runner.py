"""Run one TorchTitan shard and collect its result files.

The pod entry point calls this module for a target from a pod.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import shutil
from pathlib import Path

import moduscope_runner_support as runner_support
from moduscope_artifact_layout import (
    GRID_FILE_NAME,
    MANIFESTS_DIR_NAME,
    XLA_DUMP_DIR_NAME,
    remove_named_directories,
    task_runner_argv,
    write_grid_rows,
)
from moduscope_entry_results import all_rows_passing, finalize_entry, record_never_started_entry
from moduscope_process import TestInvocation, emit_token_line
from moduscope_targets import ShardRunContext, Target

TITAN_CONFIG_FIELD = "model_configurations"
PRUNE_DIR_NAMES = (MANIFESTS_DIR_NAME, XLA_DUMP_DIR_NAME, "_profile_upload_root")
# Keep profiler tables because the Titan configuration deletes them by default.
KEEP_ARTIFACTS_OVERRIDE = "++task_config.clean_artifacts_on_success=false"
TB_DIR_NAME = "tb"
# Preserve the runner-local patch point while sharing the implementation.
clear_compile_caches = runner_support.clear_compile_caches
installed_config_field = runner_support.installed_config_field


def titan_entry_output_dir(output_dir: str, index: int) -> str:
    """Return the per-entry artifact directory, ``e<index>``."""
    return os.path.join(output_dir, f"e{index}")


def _reject_torch_compile(target: Target) -> None:
    if target.torch_compile_enabled:
        raise ValueError(
            "Titan does not support torch_compile_enabled=True. "
            + f"Target {target.name!r} requests this setting. "
            + "Select a Titan compile config instead."
        )


def build_test_invocation(target: Target, output_dir: str, index: int) -> TestInvocation:
    """Build the entry process launch for one Titan member."""
    _reject_torch_compile(target)
    entry_dir = titan_entry_output_dir(output_dir, index)
    return TestInvocation(
        argv=(
            *task_runner_argv(target.config),
            f"hydra.run.dir={entry_dir}",
            f"++task_config.artifact_directory={entry_dir}",
            f"+hydra.overwrites.grid_entry_list=[{index}]",
            KEEP_ARTIFACTS_OVERRIDE,
        ),
    )


def resolve_titan_entry_index(
    member_name: str, installed_member_names: tuple[str, ...], config: str
) -> int:
    """Return the installed config index for one Titan member."""
    try:
        return installed_member_names.index(member_name)
    except ValueError:
        raise ValueError(
            f"Titan member {member_name!r} is absent from installed config {config!r}. "
            + "Select a member that the config contains."
        ) from None


def torchtitan_outputs_dir() -> str | None:
    """Return the TorchTitan outputs directory without importing TorchTitan."""
    package_spec = importlib.util.find_spec("torchtitan")
    if package_spec is None or not package_spec.origin:
        return None
    return os.path.join(os.path.dirname(os.path.dirname(package_spec.origin)), "outputs")


def _latest_modification_time(root: Path) -> float:
    latest_time = root.stat().st_mtime
    for child_path in root.rglob("*"):
        try:
            latest_time = max(latest_time, child_path.stat().st_mtime)
        except FileNotFoundError:
            continue
    return latest_time


def _tensorboard_session_dirs(outputs_dir: str) -> tuple[Path, ...]:
    """Return the TensorBoard session directories under the output tree."""
    tensorboard_dirs = sorted(
        path for path in Path(outputs_dir).rglob(TB_DIR_NAME) if path.is_dir()
    )
    return tuple(
        sorted(
            child
            for tensorboard_dir in tensorboard_dirs
            for child in tensorboard_dir.iterdir()
            if child.is_dir()
        )
    )


def copy_tensorboard_dirs(
    outputs_dir: str | None, entry_dir: str, seen_session_mtimes: dict[Path, float]
) -> str:
    """Copy TensorBoard sessions created or changed during this entry."""
    if outputs_dir is None:
        return (
            "TB_COPIED_FILES=0 reason=torchtitan is not importable, so the TorchTitan output "
            + "directory is unknown"
        )
    if not os.path.isdir(outputs_dir):
        return f"TB_COPIED_FILES=0 reason=no outputs tree at {outputs_dir}"
    copied_file_count = 0
    try:
        for session_dir in _tensorboard_session_dirs(outputs_dir):
            modification_time = _latest_modification_time(session_dir)
            if modification_time <= seen_session_mtimes.get(session_dir, float("-inf")):
                continue
            seen_session_mtimes[session_dir] = modification_time
            destination = Path(entry_dir, TB_DIR_NAME, session_dir.name)
            shutil.copytree(session_dir, destination, dirs_exist_ok=True)
            copied_file_count += sum(
                1 for child_path in session_dir.rglob("*") if child_path.is_file()
            )
    except OSError as exc:
        return f"TB_COPIED_FILES={copied_file_count} reason={type(exc).__name__}: {exc}"
    return f"TB_COPIED_FILES={copied_file_count}"


def prune_entry_dirs(output_root: str) -> None:
    """Remove compiler and profile output directories.

    Print the removed count.
    """
    removed_dir_count = sum(
        remove_named_directories(output_root, dir_name) for dir_name in PRUNE_DIR_NAMES
    )
    print(f"PRUNED_DIRS={removed_dir_count}")


def run_shard(context: ShardRunContext) -> int:
    """Run the Titan members in ``shard`` under one deadline. Return the shard exit code."""
    target, shard, output_dir, run_identity, deadline = context
    _reject_torch_compile(target)
    installed_member_names = installed_config_field(target.config, TITAN_CONFIG_FIELD)
    indexed_members = [
        (resolve_titan_entry_index(member_name, installed_member_names, target.config), member_name)
        for member_name in shard.members
    ]
    tensorboard_source_dir = torchtitan_outputs_dir()
    run_state = runner_support.ShardRunState(len(indexed_members))
    seen_tensorboard_sessions = {}
    if tensorboard_source_dir and os.path.isdir(tensorboard_source_dir):
        with contextlib.suppress(OSError):
            seen_tensorboard_sessions = {
                session_dir: _latest_modification_time(session_dir)
                for session_dir in _tensorboard_session_dirs(tensorboard_source_dir)
            }
    aggregate_grid_path = os.path.join(output_dir, GRID_FILE_NAME)

    def member_row_batches():
        for index, entry_name in indexed_members:
            remaining_seconds, failure_reason = run_state.check_member_start(deadline)
            if not failure_reason:
                removed_cache_dirs = clear_compile_caches()
                print(
                    f"Entry {index}: cleared {len(removed_cache_dirs)} cache dir(s): "
                    + f"{', '.join(removed_cache_dirs) or 'none'}"
                )
                remaining_seconds, failure_reason = run_state.check_member_start(deadline)
            if failure_reason:
                yield [
                    record_never_started_entry(
                        os.path.join(titan_entry_output_dir(output_dir, index), GRID_FILE_NAME),
                        index=index,
                        name=entry_name,
                        reason=failure_reason,
                        identity=run_identity,
                    )
                ]
                continue
            entry_dir = titan_entry_output_dir(output_dir, index)
            invocation = build_test_invocation(target, output_dir, index)
            invocation_result = runner_support.run_member_process(
                invocation, remaining_seconds, "Running: "
            )
            print(f"Entry {index}: {invocation_result.reason}")
            print(
                copy_tensorboard_dirs(tensorboard_source_dir, entry_dir, seen_tensorboard_sessions)
            )
            grid_path = os.path.join(entry_dir, GRID_FILE_NAME)
            # Do not add profile_s3_url. Cleanup removes it before the artifact copy.
            member_returncode, grid_rows, finalization_failed = finalize_entry(
                grid_path,
                index=index,
                name=entry_name,
                returncode=invocation_result.returncode,
                reason=invocation_result.reason,
                identity=run_identity,
                expected_failures=target.xfail,
            )
            if member_returncode == 0 and (not all_rows_passing(grid_rows) or finalization_failed):
                member_returncode = 1
            yield grid_rows
            run_state.finish_member(index, entry_name, member_returncode, len(removed_cache_dirs))

    try:
        aggregate_rows = runner_support.aggregate_member_rows(
            member_row_batches(), lambda rows: write_grid_rows(aggregate_grid_path, rows)
        )
        emit_token_line(f"AGGREGATE_GRID_ROWS={len(aggregate_rows)}")
    finally:
        prune_entry_dirs(output_dir)
    print(f"Titan runner exiting {run_state.returncode}.")
    return run_state.returncode
