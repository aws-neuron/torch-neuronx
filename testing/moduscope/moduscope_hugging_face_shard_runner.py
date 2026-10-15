"""Run the Hugging Face members of one shard one after another inside one pod.

The pod entry point calls this module for a target from a pod.
"""

from __future__ import annotations

import os

import moduscope_runner_support as runner_support
from moduscope_artifact_layout import (
    BENCHMARK_FILE_NAME,
    GRID_FILE_NAME,
    XLA_DUMP_DIR_NAME,
    add_run_identity,
    grid_model_id,
    load_grid_rows,
    read_benchmark_lines,
    read_grid_rows,
    remove_compiler_manifests,
    remove_named_directories,
    task_runner_argv,
    write_grid_rows,
)
from moduscope_entry_results import (
    all_rows_passing,
    finalize_entry,
    grid_entry_fields,
    infra_fail_row,
    record_never_started_entry,
)
from moduscope_process import InvocationResult, TestInvocation, emit_token_line
from moduscope_targets import ShardRunContext, Target

HUGGING_FACE_MEMBER_FIELD = "hf_cfg_wishlist"
ENTRIES_SUBDIR = "entries"
# Preserve the runner-local patch point while sharing the implementation.
clear_compile_caches = runner_support.clear_compile_caches


def aggregate_grids(
    output_dir: str, entry_results: list[tuple[int, str, str]], run_identity: dict[str, str] | None
) -> tuple[int, int]:
    """Combine all entry rows into one grid. Return the row and unreadable-entry counts."""
    grid_row_batches = []
    unreadable_entry_count = 0
    for entry_index, model_id, entry_dir in entry_results:
        grid_path = os.path.join(entry_dir, GRID_FILE_NAME)
        try:
            grid_row_batches.append(read_grid_rows(grid_path))
        except (FileNotFoundError, ValueError) as error:
            print(f"UNREADABLE_GRID entry={entry_index} path={grid_path} reason={error}")
            model_config = f"{grid_model_id(model_id)}_config"
            row_fields = grid_entry_fields(entry_index, model_id, model_config)
            failure_row = infra_fail_row(
                f"unreadable grid at {grid_path}: {error}", identity=run_identity, **row_fields
            )
            unreadable_entry_count += 1
            grid_row_batches.append([failure_row])

    grid_rows = runner_support.aggregate_member_rows(grid_row_batches)
    write_grid_rows(os.path.join(output_dir, GRID_FILE_NAME), grid_rows)
    return len(grid_rows), unreadable_entry_count


def aggregate_benchmarks(output_dir: str, entry_results: list[tuple[int, str, str]]) -> int:
    """Combine each entry's benchmark records into one file. Return the written line count."""
    benchmark_lines: list[str] = []
    for _, _, entry_result_dir in entry_results:
        benchmark_path = os.path.join(entry_result_dir, BENCHMARK_FILE_NAME)
        if os.path.exists(benchmark_path):
            benchmark_lines.extend(read_benchmark_lines(benchmark_path))
    if not benchmark_lines:
        return 0
    with open(os.path.join(output_dir, BENCHMARK_FILE_NAME), "w", encoding="utf-8") as handle:
        handle.writelines(line + "\n" for line in benchmark_lines)
    return len(benchmark_lines)


def add_row_fields(grid_path: str, index: int, **row_fields) -> None:
    """Add fields to each grid row."""
    rows = load_grid_rows(grid_path)
    if not rows:
        return
    for row in rows:
        row.update(row_fields)
    try:
        write_grid_rows(grid_path, rows)
    except OSError as exc:
        emit_token_line(
            f"STAMP_FAILED entry={index} fields={','.join(row_fields)} "
            + f"reason={type(exc).__name__}: {exc}"
        )


def build_test_invocation(target: Target, member_name: str, output_dir: str) -> TestInvocation:
    """Build the process inputs for one Hugging Face member."""
    argv = [
        *task_runner_argv(target.config),
        f"++task_config.hf_cfg_wishlist=['{member_name}']",
        f"++task_config.artifact_directory={output_dir}",
        "++task_config.always_clean_artifacts=false",
        "++task_config.clean_artifacts_on_success=false",
    ]
    if target.torch_compile_enabled:
        argv.append("++task_config.apply_torch_compile=true")
    return TestInvocation(tuple(argv))


def finalize_hugging_face_entry(
    target: Target,
    entry_index: int,
    model_id: str,
    entry_dir: str,
    invocation_result: InvocationResult,
    identity: dict[str, str] | None,
) -> int:
    """Finalize one entry, remove temporary data, and return its effective exit code."""
    returncode = invocation_result.returncode
    emit_token_line(
        f"MODUSCOPE_PROCESS_EXIT_CODE={returncode} entry={entry_index} model={model_id}"
    )
    effective_returncode, _, _ = finalize_entry(
        os.path.join(entry_dir, GRID_FILE_NAME),
        index=entry_index,
        name=model_id,
        returncode=returncode,
        reason=invocation_result.reason,
        identity=identity,
        expected_failures=target.xfail,
        module_configs_str=f"{grid_model_id(model_id)}_config",
    )
    if returncode != 0 and effective_returncode == 0:
        emit_token_line(f"XFAIL_ABSORBED=1 model={model_id}")
    emit_token_line(f"INDIVIDUAL_REMOVED={remove_named_directories(entry_dir, 'individual')}")
    emit_token_line(f"XLA_DUMP_REMOVED={remove_named_directories(entry_dir, XLA_DUMP_DIR_NAME)}")
    emit_token_line(f"MANIFESTS_REMOVED={remove_compiler_manifests(entry_dir)}")
    return effective_returncode


def finalize_running_placeholder(
    grid_path: str, running_row: dict, model_exit_code: int, reason: str
) -> bool:
    """Finalize the runner placeholder after the model process exits.

    Return true when a zero exit has no wheel rows.
    """
    rows = load_grid_rows(grid_path)
    if rows is None:
        if os.path.exists(grid_path):
            return model_exit_code == 0
        rows = []
    final_rows = [
        row
        for row in rows
        if row.get("status") != "running"
        or row.get("grid_entry_name") != running_row["grid_entry_name"]
        or row.get("grid_entry_index") != running_row["grid_entry_index"]
    ]
    if rows and len(final_rows) == len(rows):
        return False
    if final_rows:
        write_grid_rows(grid_path, final_rows)
        return False
    failure_reason = (
        reason if model_exit_code else f"{reason} but wrote no result rows. Check the wheel output."
    )
    failure_row = infra_fail_row(failure_reason, **running_row, **{"mfu(%)": -1})
    write_grid_rows(grid_path, [failure_row])
    return model_exit_code == 0


def run_shard(context: ShardRunContext) -> int:
    """Run ``shard`` of ``target`` one model at a time under one deadline; return its exit code."""
    target, shard, output_dir, run_identity, deadline = context
    shard_members = shard.members
    emit_token_line(
        f"SHARD index={shard.index} members={len(shard_members)} config={target.config}: "
        + ", ".join(shard_members)
    )
    emit_token_line(f"TORCH_COMPILE_ENABLED={target.torch_compile_enabled}")
    run_state = runner_support.ShardRunState(len(shard_members))
    entries_root = os.path.join(output_dir, ENTRIES_SUBDIR)
    entry_results: list[tuple[int, str, str]] = []
    for model_id in shard_members:
        index = target.members.index(model_id)
        entry_output_dir = os.path.join(entries_root, f"entry-{index}")
        entry_results.append((index, model_id, entry_output_dir))
        grid_path = os.path.join(entry_output_dir, GRID_FILE_NAME)
        remaining_seconds, failure_reason = run_state.check_member_start(deadline)
        if not failure_reason:
            os.makedirs(entry_output_dir, exist_ok=True)
            running_row = {
                "grid_entry_name": model_id,
                "grid_entry_index": index,
                "module_configs_str": f"{grid_model_id(model_id)}_config",
                "status": "running",
            }
            add_run_identity(running_row, run_identity)
            write_grid_rows(grid_path, [running_row])
            aggregate_grids(output_dir, entry_results, run_identity)
            emit_token_line(
                f"SHARD_MODEL entry={index} model={model_id} remaining={remaining_seconds:.0f}s"
            )
            cleared_cache_count = len(clear_compile_caches())
            model_timeout_seconds, failure_reason = run_state.check_member_start(deadline)
        if failure_reason:
            record_never_started_entry(
                grid_path,
                index=index,
                name=model_id,
                reason=failure_reason,
                identity=run_identity,
                module_configs_str=f"{grid_model_id(model_id)}_config",
            )
            continue
        invocation_result = runner_support.run_member_process(
            build_test_invocation(target, model_id, entry_output_dir),
            model_timeout_seconds,
            f"Entry {index} ({model_id}), {model_timeout_seconds:.0f}s bound: ",
        )
        finalize_running_placeholder(
            grid_path, running_row, invocation_result.returncode, invocation_result.reason
        )
        model_exit_code = finalize_hugging_face_entry(
            target, index, model_id, entry_output_dir, invocation_result, run_identity
        )
        if model_exit_code == 0 and not all_rows_passing(load_grid_rows(grid_path)):
            model_exit_code = 1
        entry_row_fields = {"wall_minutes": round(invocation_result.seconds / 60, 3)}
        if run_state.started_count > 0:
            entry_row_fields["peak_memory_unreliable"] = True
        add_row_fields(grid_path, index, **entry_row_fields)
        run_state.finish_member(index, model_id, model_exit_code, cleared_cache_count)
        aggregate_grids(output_dir, entry_results, run_identity)
    grid_row_count, unreadable_grid_count = aggregate_grids(output_dir, entry_results, run_identity)
    benchmark_record_count = aggregate_benchmarks(output_dir, entry_results)
    emit_token_line(
        f"AGGREGATE_GRID_ROWS={grid_row_count} "
        + f"AGGREGATE_BENCH_LINES={benchmark_record_count} "
        + f"UNREADABLE_GRIDS={unreadable_grid_count}"
    )
    if unreadable_grid_count:
        run_state.returncode = run_state.returncode or 1
    emit_token_line(
        f"SHARD_DONE index={shard.index} started={run_state.started_count} "
        + f"never_started={len(shard_members) - run_state.started_count} rc={run_state.returncode}"
    )
    return run_state.returncode
