"""Grid assertion for a ModuScope run: does the result grid prove what ran?

ModuScope's own exit code is a conjunction over the validators that ran, so it cannot encode
WHICH models ran: a run that executed two of three requested ids exits 0. The
_reusable-moduscope-tests.yml grade-grid job runs this module as a script. The module checks the
downloaded artifact. Unit tests can run without a pod.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

from moduscope_artifact_layout import GRID_FILE_NAME, read_grid_rows
from moduscope_artifact_layout import grid_model_id as normalize_model_id
from moduscope_job_runner import resolve_model_ids

#: ModuScope writes the fixed grid name on the device pod. The hosted grader deliberately does
#: not read the writer's GRID_SKELETON_FILENAME environment override.

#: No trailing anchor, matching ModuScope: one would make ``llama3_1_8B_config_cpu`` yield no id,
#: reporting a model that really ran as missing. The lookahead skips dict keys instead: full-model
#: rows carry ``'quantization_config': {}``, which would otherwise read as a model nobody
#: requested. A key is followed by a colon and the config token never is.
_CONFIG_TOKEN_RE = re.compile(r"""(?<![\w.-])([\w.-]+)_config(?!['"]?\s*:)""")


def read_grid(output_dir: str) -> list[dict]:
    """Load the result grid as a list of rows.

    Raises rather than returning ``[]`` when the grid is absent: no grid would otherwise look like
    an empty one.
    """
    path = os.path.join(output_dir, GRID_FILE_NAME)
    try:
        return read_grid_rows(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"ModuScope produced no {GRID_FILE_NAME}: {path} does not exist."
        ) from None


def grid_model_ids(rows: list[dict]) -> set[str]:
    """Return every normalized model id the grid carries, across all rows."""
    return {
        token
        for row in rows
        for token in _CONFIG_TOKEN_RE.findall(row.get("module_configs_str") or "")
    }


def assert_grid(output_dir: str, requested: tuple[str, ...]) -> None:
    """Assert the run produced a grid covering exactly ``requested``, all passing."""
    rows = read_grid(output_dir)
    found = grid_model_ids(rows)
    wanted = {normalize_model_id(model_id) for model_id in requested}
    missing = sorted(wanted - found)
    unexpected = sorted(found - wanted)
    failing = sorted(
        f"{row.get('module_configs_str') or '<no module_configs_str>'} "
        f"(entry {row.get('grid_entry_identifier')}) status={row.get('status')!r}"
        for row in rows
        if row.get("status") != "pass"
    )

    problems = []
    if missing:
        problems.append(f"requested but missing from the grid: {missing}")
    if unexpected:
        problems.append(f"in the grid but not requested: {unexpected}")
    if failing:
        problems.append(f"rows whose status is not 'pass': {failing}")
    if problems:
        raise AssertionError(
            f"{GRID_FILE_NAME} in {output_dir} does not prove the run ({len(rows)} row(s)): "
            + "; ".join(problems)
        )


#: Where GitHub reads the job summary from. The pod cannot write it -- the dispatcher does not
#: pass this variable into the workload -- so the summary is built here from the artifact.
STEP_SUMMARY_ENV_VAR = "GITHUB_STEP_SUMMARY"


def error_files(output_dir: str) -> list[str]:
    """Return every ModuScope error file under ``output_dir``, sorted, relative to it.

    On-call reads these next, so the summary names them rather than making someone glob a
    3 MB log for a path.
    """
    hits = glob.glob(os.path.join(output_dir, "**", "*error*.json"), recursive=True)
    return sorted(os.path.relpath(path, output_dir) for path in hits)


def summarize(output_dir: str, requested: tuple[str, ...]) -> str:
    """Return a markdown summary of one grid: the verdict first, then where to look.

    Verdict first because an on-call opening a failed run wants the one line that says which
    model to investigate, not a table to read top to bottom.
    """
    try:
        rows = read_grid(output_dir)
    except (FileNotFoundError, ValueError) as exc:
        # Loudly, not emptily. A missing grid is the case most likely to be misread as
        # "nothing to see", and it is the one where the run proved nothing at all.
        return (
            f"### ModuScope: NO RESULT GRID\n\n"
            f"`{exc}`\n\n"
            f"Nothing proves which models ran. The device job died before writing a grid, so "
            f"start from the device job log rather than from here.\n"
        )

    wanted = {normalize_model_id(model_id): model_id for model_id in requested}
    status_of: dict[str, str] = {}
    for row in rows:
        for token in _CONFIG_TOKEN_RE.findall(row.get("module_configs_str") or ""):
            status_of[token] = str(row.get("status"))
    found = set(status_of)

    lines = []
    # Failed and never-ran are split on purpose: they send an on-call to different places. A
    # failure means read that model error file; a no-show means the run stopped early and the
    # device log is where the story is.
    failed = sorted(wanted[t] for t in wanted if t in status_of and status_of[t] != "pass")
    absent = sorted(wanted[t] for t in wanted if t not in status_of)
    unexpected = sorted(found - set(wanted))
    if failed or absent or unexpected:
        parts = []
        if failed:
            parts.append(f"{len(failed)} failed ({', '.join(failed)})")
        if absent:
            parts.append(f"{len(absent)} never ran ({', '.join(absent)})")
        if unexpected:
            parts.append(f"{len(unexpected)} not requested ({', '.join(unexpected)})")
        lines.append(f"### ModuScope: FAILED - {'; '.join(parts)}")
    else:
        lines.append(f"### ModuScope: {len(wanted)}/{len(wanted)} models passed")
    lines.append("")
    lines.append("| model | status |")
    lines.append("| --- | --- |")
    for token, model_id in sorted(wanted.items(), key=lambda kv: kv[1]):
        state = status_of.get(token) or "DID NOT RUN"
        lines.append(f"| `{model_id}` | {state} |")
    for token in unexpected:
        lines.append(f"| `{token}` | ran but was not requested |")
    lines.append("")

    errors = error_files(output_dir)
    if errors:
        lines.append("Error files in the artifact, which is where to look next:")
        lines.extend(f"- `{name}`" for name in errors)
    else:
        lines.append("No error files in the artifact.")
    lines.append("")
    lines.append(f"Grid read from `{output_dir}`, {len(rows)} row(s).")
    return "\n".join(lines) + "\n"


def write_step_summary(text: str) -> bool:
    """Append ``text`` to the job summary. False when GitHub is not the caller."""
    path = os.environ.get(STEP_SUMMARY_ENV_VAR)
    if not path:
        return False
    with open(path, "a") as handle:
        handle.write(text)
    return True


def find_grid_dirs(artifact_root: str) -> list[str]:
    """Return every directory under ``artifact_root`` holding a grid, sorted.

    Globbed rather than joined to a fixed path: internal compute can deliver the same tree twice,
    once under ``outputs/`` and once under a ``test-results/`` mirror, so a hardcoded join reads
    one layout as no grid at all.
    """
    pattern = os.path.join(artifact_root, "**", GRID_FILE_NAME)
    return sorted({os.path.dirname(path) for path in glob.glob(pattern, recursive=True)})


def main(argv: list[str] | None = None) -> int:
    """Grade every grid in a downloaded artifact tree. Non-zero means the run is not proven.

    Every grid found must pass, not just one: a mirrored copy that disagrees with its twin is
    itself a finding, and an extra stale tree must not be the one that happens to get graded.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", help="Directory the ModuScope artifact was downloaded to.")
    parser.add_argument(
        "--model-ids",
        required=True,
        help="The model ids that were dispatched, in the workflow's comma-separated form.",
    )
    args = parser.parse_args(argv)

    # Resolved with the runner's own parser, so the list graded here is the list that ran.
    requested = resolve_model_ids(args.model_ids)
    grid_dirs = find_grid_dirs(args.artifact_root)
    if not grid_dirs:
        message = (
            f"no {GRID_FILE_NAME} anywhere under {args.artifact_root}. The device job "
            f"produced no grid, so nothing proves which models ran."
        )
        print(f"FAIL: {message}", file=sys.stderr)
        write_step_summary(
            f"### ModuScope: NO RESULT GRID\n\n`{message}`\n\n"
            f"Start from the device job log; this job had nothing to grade.\n"
        )
        return 1

    failures = []
    for grid_dir in grid_dirs:
        # The summary is written for every grid found, pass or fail, so the job page always
        # answers "which models ran" without opening the device log.
        write_step_summary(summarize(grid_dir, requested))
        try:
            assert_grid(grid_dir, requested)
        except (AssertionError, FileNotFoundError, ValueError) as exc:
            failures.append(f"{grid_dir}: {exc}")
        else:
            print(f"PASS: {grid_dir} proves {len(requested)} requested model(s).")

    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
        # ::error:: shows on the run page without opening the job.
        print(f"::error title=ModuScope grid::{failure}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
