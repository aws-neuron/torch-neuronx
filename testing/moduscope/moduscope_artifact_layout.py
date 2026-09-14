"""Where a run's files live and how they are named, read and written."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

MODUSCOPE_TASK_RUNNER_MODULE = "cli_scripts.task_runner"
ARTIFACT_ROOT = "outputs"
OUTPUT_SUBDIR = f"{ARTIFACT_ROOT}/moduscope"
# Downloaded shard files start at moduscope/ because the upload root is test-results/outputs/.
SHARD_SUBDIR = Path(OUTPUT_SUBDIR).relative_to(ARTIFACT_ROOT).as_posix()
SHARD_ARTIFACT_PREFIX = "moduscope-results-"
RELAY_ENTRIES_SUBDIR = "moduscope-entries"
RUN_ID_ENV_VAR = "MODUSCOPE_GITHUB_RUN_ID"
REPOSITORY_ENV_VAR = "MODUSCOPE_GITHUB_REPOSITORY"
IDENTITY_ENV_VARS = (RUN_ID_ENV_VAR, REPOSITORY_ENV_VAR)
XLA_DUMP_DIR_NAME = "xla_dump"
MANIFESTS_DIR_NAME = "manifests"
GRID_FILE_NAME = "grid_metrics.json"
BENCHMARK_FILE_NAME = "benchmark_results.jsonl"
GRID_HTML_RENDERER_PATH = os.path.join("tools", "grid_report_html.py")
STAGED_GRID_RENDERER_RELATIVE_PATH = Path("renderer/grid_report_html.py")


def grid_model_id(model_id: str) -> str:
    """Return the model identifier stored in a grid row."""
    return model_id.replace("/", "_")


def task_runner_argv(config_name: str) -> list[str]:
    """Return the common ModuScope task-runner command prefix."""
    return [sys.executable, "-m", MODUSCOPE_TASK_RUNNER_MODULE, f"--config-name={config_name}"]


def run_id_filename(run_id: str, basename: str) -> str:
    """Prefix a basename with the run ID."""
    return f"{run_id}-{basename}"


def run_id_file_pattern(basename: str) -> str:
    """Return the glob pattern for one run-stamped basename."""
    return f"*-{basename}"


def find_run_id_file(directory: str | Path, basename: str) -> Path | None:
    """Return one run-stamped file, or ``None`` when it is absent."""
    matches = sorted(
        path for path in Path(directory).glob(run_id_file_pattern(basename)) if path.is_file()
    )
    if len(matches) > 1:
        raise ValueError(f"{directory} has multiple run-ID files for {basename}: {matches}")
    return matches[0] if matches else None


def require_run_id_file(directory: str | Path, basename: str) -> Path:
    """Return one required run-stamped file."""
    matched_path = find_run_id_file(directory, basename)
    if matched_path is None:
        expected_path = Path(directory, run_id_file_pattern(basename))
        raise FileNotFoundError(
            f"No run-ID file matches {expected_path}. Check the uploaded artifact."
        )
    return matched_path


def stripped_env(name: str) -> str:
    """Return an environment variable without surrounding whitespace.

    Return an empty string when it is unset.
    """
    return os.environ.get(name, "").strip()


def require_env(names: tuple[str, ...]) -> dict[str, str]:
    """Return required environment variables without surrounding whitespace.

    Raise ``ValueError`` with every missing or blank name.
    """
    environment_values = {name: stripped_env(name) for name in names}
    missing_names = [name for name in names if not environment_values[name]]
    if missing_names:
        raise ValueError(
            f"required environment variable(s) unset or blank: {missing_names}. The workflow "
            + "exports every one of them before invoking this module."
        )
    return environment_values


def read_run_identity() -> dict[str, str] | None:
    """Return the run identity, the GitHub run fields added to each result row.

    Return ``None`` when a required field is missing or blank.
    """
    identity_values = {name: stripped_env(name) for name in IDENTITY_ENV_VARS}
    if all(identity_values.values()):
        run_id, repository = identity_values.values()
        return {
            "workload_id": run_id,
            "workload_link": f"https://github.com/{repository}/actions/runs/{run_id}",
        }
    missing_identity_variables = [name for name, value in identity_values.items() if not value]
    print(
        f"RUN_IDENTITY=absent missing={','.join(missing_identity_variables)}; "
        + "rows get no workload_id or workload_link"
    )
    return None


def read_grid_rows(path: str) -> list[dict]:
    """Read the grid from ``grid_metrics.json``.

    Raise ``FileNotFoundError`` for a missing file or ``ValueError`` for invalid rows.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no grid at {path}: the file does not exist.")
    with open(path, encoding="utf-8") as grid_file:
        rows = json.load(grid_file)
    if not isinstance(rows, list):
        raise ValueError(f"{path} is not a list of rows: top level is {type(rows).__name__}.")
    non_row_types = [type(row).__name__ for row in rows if not isinstance(row, dict)]
    if non_row_types:
        raise ValueError(
            f"{path} is a list but {len(non_row_types)} of its entries are not rows: "
            + f"{sorted(set(non_row_types))}."
        )
    return rows


def load_grid_rows(path: str) -> list[dict] | None:
    """Return the grid rows at ``path``.

    Return ``None`` when the file is missing, unreadable, or invalid.
    """
    try:
        return read_grid_rows(path)
    except (OSError, ValueError):
        return None


def read_benchmark_lines(path: str) -> list[str]:
    """A benchmark record is one JSON line in ``benchmark_results.jsonl``.

    Return each non-blank record without its final newline.
    """
    with open(path, encoding="utf-8") as benchmark_file:
        return [
            benchmark_line.rstrip("\n")
            for benchmark_line in benchmark_file
            if benchmark_line.strip()
        ]


def write_grid_rows(path: str, rows: list[dict]) -> None:
    """Write grid rows as JSON with an indent of 2.

    Replace the grid atomically so readers do not see partial JSON.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    staging_path = f"{path}.tmp"
    try:
        with open(staging_path, "w", encoding="utf-8") as staging_file:
            json.dump(rows, staging_file, indent=2)
        os.replace(staging_path, path)
    finally:
        if os.path.exists(staging_path):
            os.unlink(staging_path)


def remove_directory_tree(path: Path) -> bool:
    """Remove one directory tree.

    Return ``False`` and print a warning when the directory survives.
    """
    # A live child can recreate this directory after timeout.
    # Report a survivor so the runner can still write the grid.
    shutil.rmtree(path, ignore_errors=True)
    if path.exists():
        print(f"::warning::{path.name} survived removal: {path}")
        return False
    return True


def remove_named_directories(root: str, name: str) -> int:
    """Find all matching directories before removal so tree changes cannot skip one.

    Return the number removed.
    """
    return sum(
        remove_directory_tree(directory_path)
        for directory_path in list(Path(root).rglob(name))
        if directory_path.is_dir()
    )


def remove_compiler_manifests(entry_dir: str) -> int:
    """Remove ``<entry_dir>/manifests``."""
    manifests_path = Path(entry_dir, MANIFESTS_DIR_NAME)
    if not manifests_path.is_dir():
        return 0
    return int(remove_directory_tree(manifests_path))


def add_run_identity(row: dict, identity: dict[str, str] | None) -> None:
    """Add the run identity to ``row`` when it is available."""
    if identity is not None:
        row.update(identity)
