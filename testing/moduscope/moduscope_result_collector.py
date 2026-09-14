"""Collects every pod's results into the one artifact the hosted jobs read and builds the combined
reports.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
from pathlib import Path

from moduscope_artifact_layout import (
    BENCHMARK_FILE_NAME,
    GRID_FILE_NAME,
    RELAY_ENTRIES_SUBDIR,
    RUN_ID_ENV_VAR,
    SHARD_ARTIFACT_PREFIX,
    SHARD_SUBDIR,
    STAGED_GRID_RENDERER_RELATIVE_PATH,
    find_run_id_file,
    read_benchmark_lines,
    read_grid_rows,
    require_env,
    run_id_file_pattern,
    run_id_filename,
)

CACHE_KEYS_FILE_NAME = "cache_keys.txt"
TOKENS_FILE_NAME = "tokens.txt"
MODUSCOPE_WHEEL_TOKEN_PATTERN = re.compile(
    r"MODUSCOPE_WHEEL version=(?P<version>\S+) sha256=(?P<sha256>\S+)"
)
AGGREGATE_GRID_HTML_FILE_NAME = "aggregate_grid_metrics.html"
GRID_HTML_TITLE = "ModuScope Grid Metrics"
STEP_SUMMARY_COLUMNS = ("grid_entry_name", "status", "mfu(%)")
SUMMARY_FAILURE_STATUSES = frozenset({"fail", "infra_fail", "xfail"})


def gather_shard_outputs(results_root: Path, destination_root: Path) -> int:
    """Copy each result tree to ``<destination>/moduscope-entries/<shard>``. Return the file
    count.
    """
    collected_entries_dir = destination_root / RELAY_ENTRIES_SUBDIR
    collected_entries_dir.mkdir(parents=True, exist_ok=True)
    for shard_artifact_dir in sorted(path for path in results_root.glob("*") if path.is_dir()):
        pod_output_dir = shard_artifact_dir / SHARD_SUBDIR
        if not pod_output_dir.is_dir():
            print(f"SHARD_WITHOUT_OUTPUTS={shard_artifact_dir.name}")
            continue
        shutil.copytree(pod_output_dir, collected_entries_dir / shard_artifact_dir.name)
    return sum(1 for file_path in collected_entries_dir.rglob("*") if file_path.is_file())


def shard_output_dirs(results_root: Path) -> list[Path]:
    """Return each downloaded ``moduscope`` output directory in shard order."""
    return sorted(
        (path for path in results_root.glob(f"*/{SHARD_SUBDIR}") if path.is_dir()),
        key=lambda path: int(path.parent.name.removeprefix(SHARD_ARTIFACT_PREFIX)),
    )


def read_moduscope_wheel(shard_dir: Path) -> dict[str, str] | None:
    """Return the ModuScope wheel identity from one shard token file."""
    tokens_path = find_run_id_file(shard_dir, TOKENS_FILE_NAME)
    if tokens_path is None:
        return None
    for token_line in tokens_path.read_text(encoding="utf-8").splitlines():
        token_match = MODUSCOPE_WHEEL_TOKEN_PATTERN.fullmatch(token_line)
        if token_match:
            return token_match.groupdict()
    return None


def aggregate_grids(shard_dirs: list[Path]) -> list[dict]:
    """Return all rows in shard order."""
    grid_rows = []
    for shard_dir in shard_dirs:
        try:
            grid_path = find_run_id_file(shard_dir, GRID_FILE_NAME)
        except ValueError as error:
            print(f"GRID_AGGREGATED=0 reason=ValueError: {error}")
            return []
        if grid_path is not None:
            shard_rows = read_grid_rows(str(grid_path))
            try:
                moduscope_wheel = read_moduscope_wheel(shard_dir)
            except ValueError as error:
                print(
                    f"::warning::Wheel metadata is ambiguous: {error}. "
                    + "The collector omits the wheel metadata."
                )
                moduscope_wheel = None
            if moduscope_wheel is not None:
                for row in shard_rows:
                    row["moduscope_wheel"] = moduscope_wheel
            grid_rows.extend(shard_rows)
    return grid_rows


def aggregate_benchmarks(shard_dirs: list[Path]) -> list[str]:
    """Return each benchmark record without parsing it and preserve shard order."""
    benchmark_lines = []
    for shard_dir in shard_dirs:
        try:
            benchmark_path = find_run_id_file(shard_dir, BENCHMARK_FILE_NAME)
        except ValueError as error:
            print(f"BENCHMARKS_AGGREGATED=0 reason=ValueError: {error}")
            return []
        if benchmark_path is not None:
            benchmark_lines.extend(read_benchmark_lines(str(benchmark_path)))
    return benchmark_lines


def aggregate_cache_keys(shard_dirs: list[Path]) -> set[str]:
    """Return each unique cache key. A cache key identifies reusable compiled output."""
    cache_keys = set()
    for shard_dir in shard_dirs:
        for cache_keys_path in shard_dir.rglob(run_id_file_pattern(CACHE_KEYS_FILE_NAME)):
            cache_key_lines = cache_keys_path.read_text(encoding="utf-8").split("\n")
            cache_keys.update(line.strip() for line in cache_key_lines if line.strip())
    return cache_keys


def _select_grid_renderer(shard_dirs: list[Path]) -> Path | None:
    copied_renderers = []
    for shard_dir in shard_dirs:
        renderer_path = find_run_id_file(
            shard_dir / STAGED_GRID_RENDERER_RELATIVE_PATH.parent,
            STAGED_GRID_RENDERER_RELATIVE_PATH.name,
        )
        if renderer_path is not None:
            renderer_sha256 = hashlib.sha256(renderer_path.read_bytes()).hexdigest()
            copied_renderers.append((renderer_path, renderer_sha256))
    if not copied_renderers:
        return None
    first_path, first_sha256 = copied_renderers[0]
    for renderer_path, renderer_sha256 in copied_renderers[1:]:
        if renderer_sha256 != first_sha256:
            print(
                f"::warning::renderer SHA-256 mismatch first={first_sha256} "
                + f"other={renderer_sha256} path={renderer_path}. "
                + "Use one renderer SHA-256 in all shards. "
                + f"The collector uses {first_path}."
            )
    return first_path


def _render_grid_html(shard_dirs: list[Path], grid_rows: list[dict], output_path: Path) -> None:
    # HTML rendering is optional. Report a failure without stopping aggregation.
    if not grid_rows:
        print("HTML_RENDERED=0 reason=no grid rows")
        return
    try:
        renderer_path = _select_grid_renderer(shard_dirs)
        if renderer_path is None:
            print("::warning::renderer unavailable. Add the copied renderer to a shard artifact.")
            return
        module_spec = importlib.util.spec_from_file_location("grid_report_html", renderer_path)
        renderer_module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(renderer_module)
        html_document = renderer_module.generate_report_html(grid_rows, title=GRID_HTML_TITLE)
        if not isinstance(html_document, str):
            raise TypeError(f"renderer returned {type(html_document).__name__}, not str")
        output_path.write_text(html_document, encoding="utf-8")
        print(f"HTML_RENDERED=1 renderer={renderer_path}")
    # A downloaded renderer can call sys.exit() at import time. Keep that exit local.
    except (Exception, SystemExit) as error:
        print(f"::warning::renderer {type(error).__name__}: {error}. Check the copied renderer.")


def build_step_summary(grid_rows: list[dict], aggregate_counts: str) -> str:
    """Build the hosted job's Markdown summary. Include counts, model rows, and failure details."""

    def format_markdown_cell(cell_value) -> str:
        return "" if cell_value is None else str(cell_value).replace("|", "\\|")

    summary_lines = [f"## {GRID_HTML_TITLE}", "", f"`{aggregate_counts}`"]
    copied_file_count = os.environ.get("RELAY_ENTRIES_FILES")
    if copied_file_count:
        summary_lines += ["", f"`RELAY_ENTRIES_FILES={copied_file_count}`"]
    summary_lines += [
        "",
        "| " + " | ".join(STEP_SUMMARY_COLUMNS) + " |",
        "|" + " --- |" * len(STEP_SUMMARY_COLUMNS),
    ]
    failed_model_labels = []
    for row in grid_rows:
        cells = " | ".join(format_markdown_cell(row.get(column)) for column in STEP_SUMMARY_COLUMNS)
        summary_lines.append(f"| {cells} |")
        if row.get("status") in SUMMARY_FAILURE_STATUSES:
            failed_model_labels.append(f"{row.get('grid_entry_name')} ({row.get('status')})")
    if failed_model_labels:
        summary_lines += ["", "Model statuses: " + ", ".join(failed_model_labels)]
    return "\n".join(summary_lines) + "\n"


def aggregate_shard_results(results_root: Path, output_dir: Path) -> None:
    """Write the combined result, benchmark, cache key, and HTML files. Append the step
    summary when ``GITHUB_STEP_SUMMARY`` is set.
    """
    run_id = require_env((RUN_ID_ENV_VAR,))[RUN_ID_ENV_VAR]
    shard_dirs = shard_output_dirs(results_root)
    grid_rows = aggregate_grids(shard_dirs)
    benchmark_lines = aggregate_benchmarks(shard_dirs)
    cache_keys = aggregate_cache_keys(shard_dirs)
    aggregate_counts = (
        f"shards_read={len(shard_dirs)} grid_rows={len(grid_rows)} "
        + f"benchmark_records={len(benchmark_lines)} cache_keys={len(cache_keys)}"
    )
    print(aggregate_counts)
    for filename, content, aggregate_items, aggregate_label in (
        (GRID_FILE_NAME, json.dumps(grid_rows, indent=2), grid_rows, "grid"),
        (BENCHMARK_FILE_NAME, "\n".join(benchmark_lines) + "\n", benchmark_lines, "benchmarks"),
        (CACHE_KEYS_FILE_NAME, "\n".join(sorted(cache_keys)) + "\n", cache_keys, "cache keys"),
    ):
        output_path = output_dir / run_id_filename(run_id, f"aggregate_{filename}")
        if not aggregate_items:
            print(
                f"{aggregate_label}: nothing to aggregate; leaving any previous aggregate in place"
            )
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        print(f"{aggregate_label}: wrote {len(aggregate_items)} to {output_path}")
    _render_grid_html(
        shard_dirs, grid_rows, output_dir / run_id_filename(run_id, AGGREGATE_GRID_HTML_FILE_NAME)
    )
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as summary_file:
            summary_file.write(build_step_summary(grid_rows, aggregate_counts))


def main(argv: list[str] | None = None) -> int:
    """Run the gather or aggregate subcommand named in ``argv``."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    gather_parser = subparsers.add_parser("gather")
    gather_parser.add_argument("results_root", type=Path)
    gather_parser.add_argument("dest", type=Path)
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("root", type=Path)
    aggregate_parser.add_argument("-d", "--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "gather":
        print(f"RELAY_ENTRIES_FILES={gather_shard_outputs(arguments.results_root, arguments.dest)}")
    else:
        aggregate_shard_results(arguments.root, arguments.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
