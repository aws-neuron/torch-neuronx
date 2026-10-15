"""Per-node GitHub Actions step summary for the upstream test runner.

Renders a markdown block (totals, pass rate, failures) and writes it to
both ``$GITHUB_STEP_SUMMARY`` (when available) and a ``step_summary.md``
file in the artifact dir for the runner-host post-test step to pick up.

Retry semantics: ``*_on_retry`` rows are collapsed to their base status
so the panel reflects final state. The aggregator's CSVs track the
``*_on_retry`` variants separately for flakiness analysis.

Failure list is truncated to ``_MAX_FAILURES_IN_SUMMARY`` to stay under
GitHub's ~1MB step-summary limit.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Sequence
from pathlib import Path

# convert_reports is imported lazily inside compute_node_summary.


# *_on_retry → base status. The per-node panel shows final state; the
# aggregator's CSVs separately track the *_on_retry variants.
_RETRY_STATUS_TO_BASE = {
    "passed_on_retry": "passed",
    "xpassed_on_retry": "xpassed",
    "skipped_on_retry": "skipped",
    "xfailed_on_retry": "xfailed",
}

# Statuses considered failures (post-retry). Match aggregate_upstream_results.
_FAILURE_STATUSES = ("failed", "error")

# Cap on rows in the failures table. GitHub Actions truncates step
# summaries above ~1MB; this keeps us well under that for any realistic
# run. Full failures are still in the per-node CSV and the aggregator.
_MAX_FAILURES_IN_SUMMARY = 50


def compute_node_summary(main_report: Path, retry_report: Path | None) -> dict | None:
    """Build the summary payload from the main + (optional) retry JSON.

    Returns ``None`` if the main report doesn't exist or has no rows
    (the caller renders a "no report" placeholder for that case).

    Returned dict shape:
        {
            "counts":   {"passed": 100, "failed": 3, ...},  # base statuses only
            "total":    103,                                # sum of counts
            "pass_rate": "97.1%",                           # passed / non-skipped
            "duration":  1842.3,                             # wall-clock seconds
            "failures": [{"test_id": ..., "status": ...,
                          "failure_reason": ...}, ...],
        }
    """
    from convert_reports import apply_retry_outcomes, extract_test_data_from_json

    if not main_report.exists():
        return None

    main_rows = extract_test_data_from_json(str(main_report))
    if not main_rows:
        return None

    if retry_report and retry_report.exists():
        retry_rows = extract_test_data_from_json(str(retry_report))
        if retry_rows:
            main_rows = apply_retry_outcomes(main_rows, retry_rows)

    counts: dict[str, int] = {}
    duration = 0.0
    failures: list[dict] = []
    for row in main_rows:
        status = row.get("status") or "unknown"
        # Collapse *_on_retry to base. The summary shows final state.
        base = _RETRY_STATUS_TO_BASE.get(status, status)
        counts[base] = counts.get(base, 0) + 1
        with contextlib.suppress(TypeError, ValueError):
            duration += float(row.get("duration(sec)", 0))
        # Note: this checks the (possibly retry-merged) status. If retry
        # turned a main-run failure into passed_on_retry, base becomes
        # "passed" and we don't include it in failures. Correct.
        if status in _FAILURE_STATUSES:
            failures.append(
                {
                    "test_id": row.get("test_id") or row.get("nodeid", "<unknown>"),
                    "status": status,
                    "failure_reason": (row.get("failure_reason") or "").strip(),
                }
            )

    passed = counts.get("passed", 0) + counts.get("xpassed", 0)
    denom = passed + counts.get("failed", 0) + counts.get("xfailed", 0) + counts.get("error", 0)
    pass_rate = f"{passed / denom * 100:.1f}%" if denom > 0 else "N/A"

    return {
        "counts": counts,
        "total": sum(counts.values()),
        "pass_rate": pass_rate,
        "duration": duration,
        "failures": failures,
    }


def render_step_summary_markdown(items: Sequence[str], summary: dict | None) -> str:
    """Render the per-node summary as markdown for ``$GITHUB_STEP_SUMMARY``.

    Falls back to a "no report" message when ``summary`` is None so the
    panel isn't blank — gives the next person something to act on.
    """
    # A node can run several files. Show the first and a count so the heading
    # stays short.
    if not items:
        label = "(no items)"
    elif len(items) == 1:
        label = items[0]
    else:
        label = f"{items[0]} (+{len(items) - 1} more)"
    if summary is None:
        return (
            f"## Test results: {label}\n\n"
            "_No pytest JSON report was produced. Check the per-node `.log` "
            "artifact for evidence of a crash before test discovery._\n"
        )

    counts = summary["counts"]

    def n(key: str) -> int:
        return counts.get(key, 0)

    lines = [
        f"## Test results: {label}",
        "",
        f"**Total:** {summary['total']}  •  "
        f"**Pass rate:** {summary['pass_rate']}  •  "
        f"**Duration:** {summary['duration']:.1f}s",
        "",
        "| passed | failed | skipped | xfailed | xpassed | error |",
        "|---|---|---|---|---|---|",
        f"| {n('passed')} | {n('failed')} | {n('skipped')} | "
        f"{n('xfailed')} | {n('xpassed')} | {n('error')} |",
        "",
    ]

    failures = summary["failures"]
    if failures:
        lines.append(f"### Failures ({len(failures)} total)")
        lines.append("")
        shown = failures
        if len(failures) > _MAX_FAILURES_IN_SUMMARY:
            lines.append(
                f"_Showing first {_MAX_FAILURES_IN_SUMMARY}; full list is "
                "in the per-node CSV / failures aggregator._"
            )
            lines.append("")
            shown = failures[:_MAX_FAILURES_IN_SUMMARY]
        lines.append("| test_id | status | error |")
        lines.append("|---|---|---|")
        for f in shown:
            # Trim error to first line — markdown tables can't span newlines
            # without breaking the grid. Escape pipes so they don't terminate
            # the cell.
            first_line = (f["failure_reason"] or "").splitlines()
            err = first_line[0] if first_line else ""
            err = err.replace("|", "\\|")
            tid = f["test_id"].replace("|", "\\|")
            lines.append(f"| `{tid}` | {f['status']} | {err} |")
        lines.append("")

    return "\n".join(lines) + "\n"


def write_node_step_summary(
    items: Sequence[str],
    main_report: Path,
    retry_report: Path | None,
    output_dir: Path | None = None,
) -> None:
    """Render the per-node summary; write to GITHUB_STEP_SUMMARY and/or output_dir.

    Internal compute does not propagate ``$GITHUB_STEP_SUMMARY`` into the
    test workload, so the in-process write is a no-op there. Persisting
    to ``output_dir/step_summary.md`` lets a runner-host post-test step
    append the file to the real summary later. Both writes are best-effort.
    """
    try:
        summary = compute_node_summary(main_report, retry_report)
        markdown = render_step_summary_markdown(items, summary)
    except Exception as e:
        print(f"WARNING: failed to compute GitHub step summary: {e}")
        return

    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        try:
            with open(summary_file, "a") as f:
                f.write(markdown)
        except Exception as e:
            print(f"WARNING: failed to write GITHUB_STEP_SUMMARY directly: {e}")

    if output_dir is not None:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "step_summary.md", "a") as f:
                f.write(markdown)
        except Exception as e:
            print(f"WARNING: failed to persist step_summary.md to artifact: {e}")
