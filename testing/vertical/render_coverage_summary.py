#!/usr/bin/env python3
"""Render a GitHub job summary from the vertical coverage CSVs.

The vertical suites never fail: they skip on fallback and record an op status.
So the panel is an op-status roll-up, not a pass/fail table. It reads the
aggregated CSVs that ``conftest`` writes (``target_ops_coverage_*.csv`` for the
eager suite, ``target_ops_compile_coverage_*.csv`` for the torch.compile suite)
and renders a markdown panel:

  - a per-suite headline (op count and "on Neuron" rate),
  - a per-suite status table,
  - an "Ops not fully on Neuron" table (the actionable gaps), and
  - for a perf run, the slowest ops by compile and by exec time.

Like the accuracy panel, it writes the markdown to BOTH ``step_summary.md`` in
the output directory (the append-step-summary action posts it on the runner)
and, when set, ``$GITHUB_STEP_SUMMARY`` directly. Neither write can fail the run.

The tool is credential-free and needs no third-party package. ``conftest`` calls
``render`` in ``pytest_sessionfinish``; the CLI is for a local preview.

Usage:
    python testing/vertical/render_coverage_summary.py \
        --csv-dir outputs/vertical_coverage --kind coverage
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

# Statuses where the op runs fully on Neuron. Eager writes runs_on_neuron; the
# torch.compile suite writes supported. Every other status is a gap.
SUPPORTED_STATUSES = ("runs_on_neuron", "supported")

# Gap statuses, worst first. Drives both the status-table order and the sort of
# the "Ops not fully on Neuron" table. An unlisted status sorts last.
_GAP_ORDER = (
    "failed_lowering",
    "compile_timeout",
    "cpu_fallback",
    "graph_break",
    "invalid_sample",
    "unresolved",
    "no_input_sample",
)

# Each suite writes its own CSV prefix. The label is the panel heading.
_SUITES = (
    ("eager", "target_ops_coverage_*.csv"),
    ("torch.compile", "target_ops_compile_coverage_*.csv"),
)

_MAX_GAPS_IN_SUMMARY = 50
_MAX_SLOWEST_IN_SUMMARY = 15


def _status_rank(status: str) -> int:
    try:
        return _GAP_ORDER.index(status)
    except ValueError:
        return len(_GAP_ORDER)


def _find_csv(csv_dir: Path, pattern: str) -> Path | None:
    """Return the newest CSV that matches ``pattern`` in ``csv_dir``, or None."""
    matches = sorted(csv_dir.glob(pattern))
    return matches[-1] if matches else None


def _read_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _truncate(text: str, limit: int = 120) -> str:
    text = (text or "").replace("\n", " ").replace("\r", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def summarize_suite(rows: list[dict]) -> dict:
    """Roll up one suite's rows: total, on-Neuron count/rate, status counts, gaps."""
    counts = Counter(r.get("status") or "unknown" for r in rows)
    total = sum(counts.values())
    on_neuron = sum(counts.get(s, 0) for s in SUPPORTED_STATUSES)
    gaps = [r for r in rows if (r.get("status") or "unknown") not in SUPPORTED_STATUSES]
    return {
        "total": total,
        "on_neuron": on_neuron,
        "rate": round(on_neuron / total * 100, 1) if total else None,
        "counts": counts,
        "gaps": gaps,
    }


def _status_table(counts: Counter) -> list[str]:
    ordered = sorted(
        counts.items(), key=lambda kv: (kv[0] not in SUPPORTED_STATUSES, _status_rank(kv[0]), kv[0])
    )
    lines = ["| status | count |", "|---|--:|"]
    lines += [f"| {status} | {count} |" for status, count in ordered]
    return lines


def _gap_table(gaps: list[dict]) -> list[str]:
    gaps = sorted(
        gaps,
        key=lambda r: (
            _status_rank(r.get("status") or ""),
            r.get("_suite", ""),
            r.get("op_name") or "",
        ),
    )
    lines = [
        f"### Ops not fully on Neuron ({len(gaps)})",
        "",
        "| op | suite | status | detail |",
        "|---|---|---|---|",
    ]
    for r in gaps[:_MAX_GAPS_IN_SUMMARY]:
        detail = _truncate(r.get("error") or r.get("supported_by") or "")
        lines.append(f"| {r.get('op_name')} | {r.get('_suite')} | {r.get('status')} | {detail} |")
    if len(gaps) > _MAX_GAPS_IN_SUMMARY:
        lines.append(f"| … and {len(gaps) - _MAX_GAPS_IN_SUMMARY} more | | | |")
    lines.append("")
    return lines


def _slowest_table(rows: list[dict], field: str, caption: str) -> list[str]:
    def _val(r: dict) -> float | None:
        try:
            return float(r.get(field) or "")
        except (TypeError, ValueError):
            return None

    timed = [(r, _val(r)) for r in rows]
    timed = sorted((t for t in timed if t[1] is not None), key=lambda t: t[1], reverse=True)
    if not timed:
        return []
    lines = [
        f"### {caption} (top {min(_MAX_SLOWEST_IN_SUMMARY, len(timed))})",
        "",
        f"| op | suite | {field} |",
        "|---|---|--:|",
    ]
    lines += [
        f"| {r.get('op_name')} | {r.get('_suite')} | {v:.1f} |"
        for r, v in timed[:_MAX_SLOWEST_IN_SUMMARY]
    ]
    lines.append("")
    return lines


def render_markdown(suites: list[tuple[str, list[dict]]], kind: str) -> str:
    """Build the markdown panel. ``suites`` is a list of (label, rows)."""
    title = "Vertical Op Perf" if kind == "perf" else "Vertical Op Coverage"
    lines = [f"# {title} Results", ""]
    if not any(rows for _, rows in suites):
        lines += ["No coverage CSV was found. The run produced no op results.", ""]
        return "\n".join(lines) + "\n"

    lines += [
        (
            'One row per op. "On Neuron" means the op runs fully on Neuron; '
            "every other status is a gap."
        ),
        "",
    ]

    all_gaps: list[dict] = []
    all_rows: list[dict] = []
    for label, rows in suites:
        if not rows:
            continue
        s = summarize_suite(rows)
        rate = f"{s['rate']}%" if s["rate"] is not None else "n/a"
        lines += [
            f"## {label}",
            "",
            f"**Ops:** {s['total']}  •  **On Neuron:** {s['on_neuron']} ({rate})",
            "",
        ]
        lines += _status_table(s["counts"])
        lines.append("")
        all_gaps += s["gaps"]
        all_rows += rows

    if kind == "perf":
        lines += _slowest_table(all_rows, "time_to_first_ms", "Slowest compiles")
        lines += _slowest_table(all_rows, "exec_avg_ms", "Slowest exec")
    if all_gaps:
        lines += _gap_table(all_gaps)

    return "\n".join(lines) + "\n"


def _write(markdown: str, out_dir: Path) -> None:
    """Persist the panel and, under Actions, append it to the job summary.

    Neither write can fail the run: the panel is informational.
    """
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "step_summary.md").write_text(markdown)
    except Exception as e:
        logger.warning("could not write step_summary.md: %s", e)
    gh = os.environ.get("GITHUB_STEP_SUMMARY")
    if gh:
        try:
            with open(gh, "a") as f:
                f.write(markdown)
        except Exception as e:
            logger.warning("could not append to GITHUB_STEP_SUMMARY: %s", e)


def render(csv_dir: Path, out_dir: Path, kind: str = "coverage") -> str:
    """Read the suite CSVs under ``csv_dir``, render the panel, write it, return it."""
    suites: list[tuple[str, list[dict]]] = []
    for label, pattern in _SUITES:
        path = _find_csv(csv_dir, pattern)
        rows = _read_rows(path) if path else []
        for r in rows:
            r["_suite"] = label
        suites.append((label, rows))
    markdown = render_markdown(suites, kind)
    _write(markdown, out_dir)
    return markdown


def _parse_args(argv: list | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--csv-dir", required=True, help="Directory that holds the coverage CSVs.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for step_summary.md. Defaults to --csv-dir.",
    )
    parser.add_argument("--kind", choices=("coverage", "perf"), default="coverage")
    return parser.parse_args(argv)


def main(argv: list | None = None) -> int:
    args = _parse_args(argv)
    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir) if args.out_dir else csv_dir
    render(csv_dir, out_dir, args.kind)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
