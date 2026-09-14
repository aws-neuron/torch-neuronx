"""Generate a GitHub Actions step summary from pytest JSON reports.

``run-test-parallel`` writes one pytest-json report per suite into
``<reports-dir>/json_reports`` (``report.json``, ``isolated-report.json``,
...). It writes a ``<suite>-retry.json`` when a suite retries its failures.
This tool merges the retry outcome into each suite, sums the suites, and
writes a markdown panel to two places:

  - ``$GITHUB_STEP_SUMMARY``, when the variable is set. Internal compute
    does not propagate the variable into the test workload, so the write
    is a no-op there.
  - ``<reports-dir>/step_summary.md``. A runner-host post-test step appends
    this file to the real job summary later.

This mirrors the per-node summary of the upstream test runner. See
``tests/pytorch_tests_native/runner/step_summary.py``.

Usage:
    python generate_unit_step_summary.py --reports-dir DIR [--title TITLE]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# convert_reports and step_summary live in the upstream runner dir. Reuse
# their pytest-json parsing and retry-merge logic instead of a second copy.
_RUNNER_DIR = Path(__file__).resolve().parent.parent / "pytorch_tests_native" / "runner"
sys.path.insert(0, str(_RUNNER_DIR))

# Cap on failure rows. GitHub truncates a step summary above ~1MB. The full
# set stays in the uploaded JSON/HTML report artifact.
_MAX_FAILURES_IN_SUMMARY = 50


def _suite_reports(json_dir: Path) -> list[tuple[str, Path, Path | None]]:
    """Return one ``(suite, main_report, retry_report)`` for each base report.

    A base report is any ``*.json`` that is not the coverage report and is
    not a ``*-retry.json`` file. The retry report is the matching
    ``<suite>-retry.json`` when it exists.
    """
    out: list[tuple[str, Path, Path | None]] = []
    for path in sorted(json_dir.glob("*.json")):
        name = path.name
        if name == "coverage.json" or name.endswith("-retry.json"):
            continue
        suite = path.stem
        retry = json_dir / f"{suite}-retry.json"
        out.append((suite, path, retry if retry.exists() else None))
    return out


def aggregate_reports(json_dir: Path) -> dict | None:
    """Merge every suite report in ``json_dir`` into one summary.

    Return ``None`` when no suite produced a usable report. Each suite goes
    through ``compute_node_summary``, so a retry outcome overrides a
    main-run failure before the sum. The returned dict has this shape:

        {
            "counts":    {status: n, ...},   # base statuses, summed
            "total":     int,
            "pass_rate": str,
            "duration":  float,              # seconds, summed
            "failures":  [{suite, test_id, status, failure_reason}, ...],
            "suites":    [{name, counts, total}, ...],
        }
    """
    from step_summary import compute_node_summary

    counts: dict[str, int] = {}
    duration = 0.0
    failures: list[dict] = []
    suites: list[dict] = []

    for suite, main_report, retry_report in _suite_reports(json_dir):
        summary = compute_node_summary(main_report, retry_report)
        if summary is None:
            continue
        for status, n in summary["counts"].items():
            counts[status] = counts.get(status, 0) + n
        duration += summary["duration"]
        for failure in summary["failures"]:
            failures.append({"suite": suite, **failure})
        suites.append({"name": suite, "counts": summary["counts"], "total": summary["total"]})

    if not suites:
        return None

    passed = counts.get("passed", 0) + counts.get("xpassed", 0)
    denom = passed + counts.get("failed", 0) + counts.get("xfailed", 0) + counts.get("error", 0)
    pass_rate = f"{passed / denom * 100:.1f}%" if denom > 0 else "N/A"

    return {
        "counts": counts,
        "total": sum(counts.values()),
        "pass_rate": pass_rate,
        "duration": duration,
        "failures": failures,
        "suites": suites,
    }


def _status_row(counts: dict[str, int]) -> str:
    def n(key: str) -> int:
        return counts.get(key, 0)

    return (
        f"| {n('passed')} | {n('failed')} | {n('skipped')} | "
        f"{n('xfailed')} | {n('xpassed')} | {n('error')} |"
    )


def render_markdown(title: str, summary: dict | None) -> str:
    """Render the aggregated summary as markdown for the job summary.

    Fall back to a "no report" message when ``summary`` is ``None``. The
    reader then has a next step instead of a blank panel.
    """
    if summary is None:
        return (
            f"## {title}\n\n"
            "_No pytest JSON report was produced. Check the job log and the "
            "uploaded test-results artifact for a crash before test discovery._\n"
        )

    lines = [
        f"## {title}",
        "",
        f"**Total:** {summary['total']}  •  "
        f"**Pass rate:** {summary['pass_rate']}  •  "
        f"**Duration:** {summary['duration']:.1f}s",
        "",
        "| passed | failed | skipped | xfailed | xpassed | error |",
        "|---|---|---|---|---|---|",
        _status_row(summary["counts"]),
        "",
    ]

    # Show the worst suites first so a failing suite is at the top.
    suites = sorted(
        summary["suites"],
        key=lambda s: (
            -(s["counts"].get("failed", 0) + s["counts"].get("error", 0)),
            s["name"],
        ),
    )
    lines.append("### Suites")
    lines.append("")
    lines.append("| suite | total | passed | failed | skipped | xfailed | xpassed | error |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in suites:
        c = s["counts"]

        def n(key: str, c: dict = c) -> int:
            return c.get(key, 0)

        lines.append(
            f"| {s['name']} | {s['total']} | {n('passed')} | {n('failed')} | "
            f"{n('skipped')} | {n('xfailed')} | {n('xpassed')} | {n('error')} |"
        )
    lines.append("")

    failures = summary["failures"]
    if failures:
        lines.append(f"### Failures ({len(failures)} total)")
        lines.append("")
        shown = failures
        if len(failures) > _MAX_FAILURES_IN_SUMMARY:
            lines.append(
                f"_Showing first {_MAX_FAILURES_IN_SUMMARY}; the full list is "
                "in the uploaded test-results artifact._"
            )
            lines.append("")
            shown = failures[:_MAX_FAILURES_IN_SUMMARY]
        lines.append("| suite | test_id | status | error |")
        lines.append("|---|---|---|---|")
        for f in shown:
            # Keep the first line only. A markdown table cell cannot span
            # newlines. Escape pipes so they do not end the cell.
            first_line = (f.get("failure_reason") or "").splitlines()
            err = first_line[0] if first_line else ""
            err = err.replace("|", "\\|")
            tid = str(f.get("test_id", "<unknown>")).replace("|", "\\|")
            lines.append(f"| {f['suite']} | `{tid}` | {f['status']} | {err} |")
        lines.append("")

    return "\n".join(lines) + "\n"


def write_summary(markdown: str, reports_dir: Path) -> None:
    """Write the markdown to ``$GITHUB_STEP_SUMMARY`` and to a file.

    Both writes are best-effort. A write failure must not mask the test
    exit code.
    """
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        try:
            with open(summary_file, "a") as f:
                f.write(markdown)
        except Exception as e:
            print(f"WARNING: could not write GITHUB_STEP_SUMMARY directly: {e}")

    try:
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "step_summary.md").write_text(markdown)
    except Exception as e:
        print(f"WARNING: could not persist step_summary.md: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a pytest step summary.")
    parser.add_argument("--reports-dir", required=True, type=Path)
    parser.add_argument("--title", default="Unit Test Results")
    args = parser.parse_args()

    json_dir = args.reports_dir / "json_reports"
    try:
        summary = aggregate_reports(json_dir) if json_dir.is_dir() else None
        markdown = render_markdown(args.title, summary)
    except Exception as e:
        print(f"WARNING: failed to build step summary: {e}")
        return

    write_summary(markdown, args.reports_dir)
    print(f"Wrote step summary to {args.reports_dir / 'step_summary.md'}")


if __name__ == "__main__":
    main()
