"""Provide command, result, and cleanup helpers for one Hugging Face entry.

Later pod-runner and grader changes can import these helpers.
"""

from __future__ import annotations

import os
import re
import sys
from typing import TYPE_CHECKING

from moduscope_artifact_layout import (
    BENCHMARK_FILE_NAME,
    GRID_FILE_NAME,
    MODUSCOPE_TASK_RUNNER_MODULE,
    XLA_DUMP_DIR_NAME,
    add_run_identity,
    load_grid_rows,
    read_benchmark_lines,
    read_grid_rows,
    remove_compiler_manifests,
    remove_named_directories,
    write_grid_rows,
)
from moduscope_entry_results import (
    expected_failure_absorbs_exit,
    finalize_entry_grid,
    github_failure_annotations,
    infra_fail_row,
)
from moduscope_process import emit_token_line

if TYPE_CHECKING:
    from moduscope_targets import Target

# clean_up_artifacts removes individual/**/*_metrics*.json, including the profiler JSON.
# Keep these settings disabled so the artifact retains that data.
KEEP_ARTIFACTS_OVERRIDES = (
    "++task_config.always_clean_artifacts=false",
    "++task_config.clean_artifacts_on_success=false",
)
MODEL_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+")
ENTRIES_SUBDIR = "entries"


def grid_model_id(model_id: str) -> str:
    """Return the model identifier encoded in a grid row."""
    return model_id.replace("/", "_")


def model_config_token(model_id: str) -> str:
    """Return the exact `module_configs_str` value for the model."""
    return f"{grid_model_id(model_id)}_config"


def aggregate_grids(
    output_dir: str,
    entry_results: list[tuple[int, str, str]],
    run_identity: dict[str, str] | None,
) -> tuple[int, int]:
    """Combine all entry rows into one grid. Return the row and unreadable-entry counts."""
    grid_rows: list = []
    unreadable_entry_count = 0
    for entry_index, model_id, entry_dir in entry_results:
        grid_path = os.path.join(entry_dir, GRID_FILE_NAME)
        try:
            grid_rows.extend(read_grid_rows(grid_path))
        except (FileNotFoundError, ValueError) as error:
            print(f"UNREADABLE_GRID entry={entry_index} path={grid_path} reason={error}")
            failure_row = infra_fail_row(
                entry_index,
                model_id,
                f"unreadable grid at {grid_path}: {error}",
                model_config_token(model_id),
            )
            add_run_identity(failure_row, run_identity)
            grid_rows.append(failure_row)
            unreadable_entry_count += 1
    write_grid_rows(os.path.join(output_dir, GRID_FILE_NAME), grid_rows)
    return len(grid_rows), unreadable_entry_count


def aggregate_benchmarks(output_dir: str, entry_result_dirs: list[str]) -> int:
    """Combine each entry's benchmark records into one file. Return the written line count."""
    benchmark_lines: list[str] = []
    for entry_result_dir in entry_result_dirs:
        benchmark_path = os.path.join(entry_result_dir, BENCHMARK_FILE_NAME)
        if os.path.exists(benchmark_path):
            benchmark_lines.extend(read_benchmark_lines(benchmark_path))
    if not benchmark_lines:
        return 0
    with open(os.path.join(output_dir, BENCHMARK_FILE_NAME), "w", encoding="utf-8") as handle:
        handle.writelines(line + "\n" for line in benchmark_lines)
    return len(benchmark_lines)


def build_moduscope_argv(model_id: str, output_dir: str, config_name: str) -> list[str]:
    """Build the ModuScope wheel arguments for ``model_id`` under ``config_name``."""
    return [
        sys.executable,
        "-m",
        MODUSCOPE_TASK_RUNNER_MODULE,
        f"--config-name={config_name}",
        f"++task_config.hf_cfg_wishlist=['{model_id}']",
        f"++task_config.artifact_directory={output_dir}",
        *KEEP_ARTIFACTS_OVERRIDES,
    ]


def finalize_hugging_face_entry(
    target: Target,
    entry_index: int,
    model_id: str,
    entry_dir: str,
    *,
    returncode: int,
    reason: str,
    identity: dict[str, str] | None,
) -> int:
    """Finalize and annotate the entry grid while printing token lines in their required order.

    Remove temporary data and return zero when an xfail absorbs the exit.
    """
    emit_token_line(
        f"MODUSCOPE_PROCESS_EXIT_CODE={returncode} entry={entry_index} model={model_id}"
    )
    grid_path = os.path.join(entry_dir, GRID_FILE_NAME)
    # Do not add profile_url. Cleanup removes the profile before the artifact copy.
    emit_token_line(
        finalize_entry_grid(
            grid_path,
            index=entry_index,
            name=model_id,
            returncode=returncode,
            reason=reason,
            identity=identity,
            module_configs_str=model_config_token(model_id),
            expected_to_fail=model_id in target.xfail,
        )
    )
    grid_rows = load_grid_rows(grid_path)
    for annotation_line in github_failure_annotations(grid_rows, target.xfail.get(model_id)):
        emit_token_line(annotation_line)
    if returncode and expected_failure_absorbs_exit(grid_rows, listed=model_id in target.xfail):
        emit_token_line(f"XFAIL_ABSORBED=1 model={model_id}")
        returncode = 0
    emit_token_line(f"INDIVIDUAL_REMOVED={remove_named_directories(entry_dir, 'individual')}")
    emit_token_line(f"XLA_DUMP_REMOVED={remove_named_directories(entry_dir, XLA_DUMP_DIR_NAME)}")
    emit_token_line(f"MANIFESTS_REMOVED={remove_compiler_manifests(entry_dir)}")
    return returncode
