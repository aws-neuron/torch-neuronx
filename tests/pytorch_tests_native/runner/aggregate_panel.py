"""GitHub Actions step-summary panel for the aggregator.

Renders a three-section markdown panel onto ``$GITHUB_STEP_SUMMARY``:

  1. Headline + status breakdown — the high-level health number plus a
     semantic grouping (Passed / Passed-on-retry / Failed / Known issues
     (xfailed) / Skipped) so the wide raw-status set isn't dumped at the
     reader.
  2. Failures — failed/error rows from ``test_rows``, capped at
     :data:`_MAX_AGGREGATED_FAILURES_IN_PANEL`. Full failures live in the
     failures CSV.
  3. Per-group breakdown — file-level stats collapsed by top-level
     directory (``dynamo/``, ``distributed/``, etc.) so the table stays
     readable on the full suite.

Lives as its own module (alongside ``step_summary.py`` for the per-node
panel) so the aggregator file stays focused on parsing and CSV output.
The canonical per-class summary CSV is unchanged — this is panel-only.
"""

from __future__ import annotations

import os
from collections import defaultdict

from aggregate_upstream_results import (
    FAILURE_STATUSES,
    SKIPPED_STATUSES,
    _calc_pass_rate,
    _file_from_row,
    _get_run_metadata,
)

# Semantic groupings used by the panel's "Status breakdown" table. Each row
# of the breakdown collapses related raw statuses (e.g. ``passed`` +
# ``xpassed`` + ``subtests passed`` are all "this test ran and was good").
# Order here = display order.
_STATUS_BREAKDOWN_GROUPS = (
    ("Passed", ("passed", "xpassed", "subtests passed")),
    ("Passed on retry (flaky)", ("passed_on_retry", "xpassed_on_retry")),
    ("Failed", ("failed", "error")),
    ("Known issues (xfailed)", ("xfailed", "xfailed_on_retry")),
    ("Skipped", ("skipped", "skipped_on_retry")),
)

# Cap the failures table in the aggregator panel — full failures are in
# the failures CSV, which the panel links to implicitly via the bundle.
_MAX_AGGREGATED_FAILURES_IN_PANEL = 100


def _is_all_skipped(unit_stats: dict) -> bool:
    """A grouping (class or file) with no executed tests — only skips.

    These rows are pure noise in the GH summary panel: pass_rate is N/A and
    they outnumber the rows that actually ran. We hide them in the
    rendered markdown but keep them in the canonical summary CSV so
    dashboards and historical comparisons aren't affected.
    """
    return all(status in SKIPPED_STATUSES for status in unit_stats)


# Back-compat alias for tests that imported the old class-specific name.
_is_all_skipped_class = _is_all_skipped


def _sum_stats(per_class_stats):
    """Sum a sequence of per-class stat dicts into a single status→count dict."""
    out: dict[str, int] = defaultdict(int)
    for cls_stats in per_class_stats:
        for status, count in cls_stats.items():
            out[status] += count
    return out


def _build_status_breakdown_rows(total_stats: dict) -> list[tuple[str, int, str]]:
    """Return semantically-grouped (label, count, breakdown) tuples.

    Skips groups whose count is 0 so the panel doesn't show empty rows.
    """
    rows: list[tuple[str, int, str]] = []
    for label, statuses in _STATUS_BREAKDOWN_GROUPS:
        present = [(s, total_stats.get(s, 0)) for s in statuses]
        total = sum(c for _, c in present)
        if total == 0:
            continue
        breakdown = " • ".join(f"{s}: {c}" for s, c in present if c > 0)
        rows.append((label, total, breakdown))
    return rows


def _aggregated_failures(test_rows: list[dict]) -> list[dict]:
    """Return failure rows (failed/error) sorted by file + test_id.

    Stable ordering means two runs with the same failures produce the
    same panel — diffable across runs without spurious reorder noise.
    """
    failures = [row for row in test_rows if row.get("status") in FAILURE_STATUSES]
    failures.sort(
        key=lambda r: (
            _file_from_row(r),
            r.get("test_id") or r.get("nodeid", ""),
        )
    )
    return failures


def write_github_summary(result: dict) -> None:
    """Write the aggregator's GitHub Actions step summary panel.

    Three sections, top-to-bottom: headline + status breakdown, failures
    table (capped), per-group breakdown. See module docstring.
    """
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_file:
        return

    # Group-level stats for the per-group section; fall back to file-
    # then class-level for older callers that didn't populate group_stats.
    panel_stats = result.get("group_stats") or result.get("file_stats") or result.get("stats", {})
    panel_durations = (
        result.get("group_durations") or result.get("file_durations") or result.get("durations", {})
    )
    sorted_statuses = result["sorted_statuses"]
    total_stats = result["total_stats"]
    test_rows = result.get("test_rows", [])
    metadata = _get_run_metadata()

    with open(summary_file, "a") as f:
        # ---------- Section 1: headline + status breakdown ----------
        f.write("## PT Upstream Test Results\n\n")
        f.write(
            f"**Commit:** `{metadata['commit_sha']}`  •  "
            f"**Branch:** `{metadata['branch']}`  •  "
            f"**Run:** `{metadata['run_id']}`\n\n"
        )

        total_count = sum(total_stats.values())
        pass_rate = _calc_pass_rate(total_stats)
        duration = sum(panel_durations.values())
        # Failure count surfaces in the headline so reviewers don't have
        # to scroll to find it.
        failure_count = sum(total_stats.get(s, 0) for s in FAILURE_STATUSES)
        f.write(
            f"**Total:** {total_count}  •  "
            f"**Pass rate:** {pass_rate}%  •  "
            f"**Duration:** {round(duration, 1)}s  •  "
            f"**Failures:** {failure_count}\n\n"
        )

        # Semantic groups — collapses 11 raw status columns down to 5
        # meaningful rows. Empty groups are skipped.
        breakdown_rows = _build_status_breakdown_rows(total_stats)
        if breakdown_rows:
            f.write("| group | count | breakdown |\n")
            f.write("|---|---|---|\n")
            for label, count, breakdown in breakdown_rows:
                f.write(f"| **{label}** | {count} | {breakdown} |\n")
            f.write("\n")

        # ---------- Section 2: failures ----------
        failures = _aggregated_failures(test_rows)
        if failures:
            f.write(f"## Failures ({len(failures)} total)\n\n")
            shown = failures
            if len(failures) > _MAX_AGGREGATED_FAILURES_IN_PANEL:
                f.write(
                    f"_Showing first {_MAX_AGGREGATED_FAILURES_IN_PANEL}; "
                    "full list in the failures CSV._\n\n"
                )
                shown = failures[:_MAX_AGGREGATED_FAILURES_IN_PANEL]
            f.write("| test_id | status | error |\n")
            f.write("|---|---|---|\n")
            for row in shown:
                # First line of the error only — markdown tables can't
                # span newlines without breaking the grid.
                reason = (row.get("failure_reason") or "").strip()
                err = reason.splitlines()[0] if reason else ""
                err = err.replace("|", "\\|")
                test_id = (row.get("test_id") or row.get("nodeid", "")).replace("|", "\\|")
                status = row.get("status", "")
                f.write(f"| `{test_id}` | {status} | {err} |\n")
            f.write("\n")

        # ---------- Section 3: per-group breakdown ----------
        visible_units = [u for u in sorted(panel_stats) if not _is_all_skipped(panel_stats[u])]
        hidden_count = len(panel_stats) - len(visible_units)

        f.write("## Per-group breakdown\n\n")
        if not visible_units:
            f.write("_All groups have only skipped tests — nothing to show._\n")
            return

        f.write(
            f"| group | {' | '.join(sorted_statuses)} " "| total | pass_rate(%) | duration(sec) |\n"
        )
        f.write("|---" * (len(sorted_statuses) + 4) + "|\n")
        totals = [0] * len(sorted_statuses)
        total_duration = 0.0
        for unit in visible_units:
            counts = [panel_stats[unit].get(s, 0) for s in sorted_statuses]
            unit_pass_rate = _calc_pass_rate(panel_stats[unit])
            unit_duration = round(panel_durations.get(unit, 0), 3)
            f.write(
                f"| {unit} | {' | '.join(str(c) for c in counts)} | "
                f"{sum(counts)} | {unit_pass_rate} | {unit_duration} |\n"
            )
            totals = [t + c for t, c in zip(totals, counts, strict=False)]
            total_duration += panel_durations.get(unit, 0)

        visible_total_pass_rate = _calc_pass_rate(_sum_stats(panel_stats[u] for u in visible_units))
        f.write(
            f"| **TOTAL (visible)** | {' | '.join(str(t) for t in totals)} | "
            f"**{sum(totals)}** | **{visible_total_pass_rate}** | "
            f"**{round(total_duration, 3)}** |\n"
        )

        if hidden_count:
            f.write(f"\n_Hiding {hidden_count} all-skipped group(s) (pass_rate N/A)._\n")
