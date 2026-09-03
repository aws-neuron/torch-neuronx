"""Pytest config for the op-accuracy suite.

Pytest merges conftests down the directory chain, so this file
inherits everything ``tests/vertical/conftest.py`` already sets up
and adds only what accuracy needs on top.

Does **not** use ``COVERAGE_FIELDS``, added ``ROW_FIELDS`` instead.

Aggregates: test_accuracy.py -> op_accuracy_<ts>.csv
"""

from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from .xfail_registry import Action, Metric
from .xfail_registry import lookup as xfail_disposition

# The CSV metric column each panel/gate metric reads, mapped to its registry Metric.
_FIELD_METRIC = {
    "relative_norm_verdict": Metric.RELATIVE,
    "frobenius_ratio_verdict": Metric.FROBENIUS,
}

logger = logging.getLogger(__name__)

_WORKER_ID = os.environ.get("PYTEST_XDIST_WORKER", "main")

_ROWS_DIR = (
    Path(os.environ.get("NEURON_COVERAGE_METRICS_DIR", tempfile.gettempdir()))
    / "neuron_accuracy_rows"
)

# Where the aggregated CSV(s) and step-summary land. Default CWD keeps local runs
# unchanged; CI sets it to ``outputs/accuracy`` so run_test.sh ships them as the
# downloadable artifact and the step summary reaches the GitHub job summary.
_OUTPUT_DIR = Path(os.environ.get("NEURON_ACCURACY_OUTPUT_DIR", "."))

# accuracy output csv columns.
ROW_FIELDS = [
    "cell_id",
    "op",
    "op_group",
    "op_mode",
    "dtype",
    "path",
    "arg_shapes",
    "unit",
    "max_rel_err_units",
    "budget_units",
    "relative_norm_verdict",
    "max_abs_err",
    "max_abs_ref",
    "frobenius_abs_err_base",
    "frobenius_ratio",
    "frobenius_ratio_verdict",
    "seed",
    "gates",
    "status",
    "error_cause_raw",
]


def pytest_addoption(parser):
    group = parser.getgroup("accuracy")
    group.addoption(
        "--seed-base",
        default=0,
        type=int,
        help="Re-roll every cell's input draw while keeping cells independent.",
    )
    group.addoption(
        "--dtypes",
        default="bfloat16,float32",
        help=(
            "Comma-separated dtypes to sweep (default: both bf16 and fp32). This is the "
            "dtype-scalability seam: adding a format means implementing its NumericFormat and "
            "naming it here, not touching the driver. float32 and bfloat16 are the only two "
            "Neuron has; anything else raises KeyError rather than silently reporting a "
            "wrong unit."
        ),
    )


def pytest_configure(config):
    # Create the output dir up front so the json-report plugin's --json-report-file (and
    # our CSV/step-summary writers) can write into it regardless of hook ordering.
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def pytest_generate_tests(metafunc):
    """Parametrize the sweep's dtype axis from ``--dtypes`` so each ``(op, dtype)`` is its
    own item -- that is what lets the skip/xfail registry act at op+dtype granularity."""
    if "dtype_name" in metafunc.fixturenames:
        raw = metafunc.config.getoption("--dtypes")
        metafunc.parametrize("dtype_name", [s.strip() for s in raw.split(",") if s.strip()])


@pytest.fixture(scope="session")
def accuracy_config(request) -> dict:
    """Resolved knobs, plus the gate values that decided which shapes are eligible."""
    opt = request.config.getoption
    return {
        "seed_base": opt("--seed-base"),
        "gates": {
            "max_shapes_per_op": int(os.environ.get("NEURON_MAX_SHAPES_PER_OP", "10")),
            "max_tensor_elems": int(
                os.environ.get("NEURON_MAX_TENSOR_ELEMS", str(512 * 1024 * 1024))
            ),
            "model_shapes_only": bool(os.environ.get("NEURON_MODEL_SHAPES_ONLY")),
            "shape_registry": os.environ.get("NEURON_SHAPE_REGISTRY", ""),
            "traced_model": os.environ.get("NEURON_TRACED_MODEL", ""),
        },
    }


@pytest.fixture
def emit_row():
    """Append one cell's row. One file per cell so xdist workers never contend."""

    def _emit(row: dict) -> None:
        _ROWS_DIR.mkdir(parents=True, exist_ok=True)
        # Include path so a shape measured for both capture paths writes two files, not one.
        name = f"{row.get('cell_id', 'unknown')}_{row.get('path') or 'all'}_{_WORKER_ID}.json"
        (_ROWS_DIR / name).write_text(json.dumps(row, default=str))

    return _emit


def pytest_sessionstart(session):
    if hasattr(session.config, "workerinput"):
        return
    if _ROWS_DIR.exists():
        for f in _ROWS_DIR.glob("*.json"):
            f.unlink()


def pytest_sessionfinish(session, exitstatus):
    if hasattr(session.config, "workerinput"):
        return
    _aggregate(_ROWS_DIR, "op_accuracy")


def _aggregate(rows_dir: Path, prefix: str):
    """Merge per-cell JSON rows into one timestamped CSV."""
    if not rows_dir.exists():
        return

    rows = []
    for fp in sorted(rows_dir.glob("*.json")):
        try:
            rows.append(_flatten(json.loads(fp.read_text())))
        except Exception as e:
            logger.warning(f"Failed to read {fp}: {e}")

    if not rows:
        return

    rows.sort(
        key=lambda r: (str(r.get("op", "")), str(r.get("dtype", "")), str(r.get("cell_id", "")))
    )
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # One CSV per capture path (``row["path"]``); an unsplit run (path=None) writes one CSV.
    by_path: dict = {}
    for row in rows:
        by_path.setdefault(row.get("path"), []).append(row)
    for path, prows in sorted(by_path.items(), key=lambda kv: str(kv[0])):
        out_prefix = f"{prefix}_{path}" if path else prefix
        output = _OUTPUT_DIR / f"{out_prefix}_{ts}.csv"
        with open(output, "w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=ROW_FIELDS, quoting=csv.QUOTE_MINIMAL, extrasaction="ignore"
            )
            w.writeheader()
            for row in prows:
                w.writerow({k: _cell(row.get(k)) for k in ROW_FIELDS})
        logger.info(f"wrote {len(prows)} rows -> {output}")

    _summarize(rows)
    _write_step_summary(rows, _OUTPUT_DIR)
    for fp in rows_dir.glob("*.json"):
        fp.unlink()


def _cell(value) -> str:
    """One CSV cell. ``None`` becomes empty, not the string ``"None"``.

    Every nullable column depends on this: a ``no_gate`` row's ``budget_units``, or the
    ``max_rel_err_units`` of a cell that never ran, would otherwise be written as a non-empty
    ``"None"`` that reads as a value and breaks a numeric parse downstream.
    """
    if value is None:
        return ""
    return str(value).replace("\n", " ").replace("\r", "")


def _flatten(row: dict) -> dict:
    """JSON-encode the dict-valued fields so each stays one CSV cell."""
    out = dict(row)
    for k in ("arg_shapes", "gates"):
        if isinstance(out.get(k), dict | list):
            out[k] = json.dumps(out[k], sort_keys=True)
    return out


def _summarize(rows: list[dict]):
    """Print the disposition tally.

    Non-verdicts are named explicitly rather than folded into a pass count: an op that was
    not measurable must stay visible, which is the failure mode where a "skipped" status
    quietly counts as success.
    """
    tally: dict[str, int] = {}
    for r in rows:
        key = (
            r.get("relative_norm_verdict")
            if r.get("status") == "ok"
            else r.get("status", "unknown")
        )
        tally[str(key)] = tally.get(str(key), 0) + 1
    logger.info("accuracy disposition: " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))


# ── step summary (metrics panel) ─────────────────────────────────────────────
_KINDS = ["pass", "fail", "xfail", "skip", "xpass", "other"]
# An op rolls up to the most notable disposition among its shape-cells, worst first.
_OP_PRECEDENCE = ["fail", "xfail", "xpass", "pass", "other", "skip"]
_MAX_FAILURES_IN_SUMMARY = 50


def _reduce_op(kinds: list[str]) -> str:
    """Collapse an op's per-shape cell dispositions to one, by ``_OP_PRECEDENCE``."""
    present = set(kinds)
    return next((k for k in _OP_PRECEDENCE if k in present), "other")


def _gated(dtype: str) -> bool:
    from .formats import get_format

    try:
        return get_format(dtype).gated
    except KeyError:
        return False


def _classify(row: dict, field: str) -> str:
    """One cell's disposition for a metric ``field`` (``"relative_norm_verdict"`` = rel-err,
    ``"frobenius_ratio_verdict"`` = Frobenius). ``other`` = ran but produced no verdict for this
    metric (device error, not-measured, int/bool output, no baseline) -- kept distinct so it is
    never silently counted as a pass. Module-level (not a closure) so it is unit-testable."""
    if row.get("status") == "skipped":
        return "skip"
    verdict = row.get(field)
    if verdict == "within":
        ok = True
    elif verdict in ("over", "invalid"):
        ok = False
    else:
        return "other"
    disposition = xfail_disposition(
        row.get("op"), row.get("dtype"), row.get("arg_shapes"), _FIELD_METRIC.get(field)
    )
    if disposition is not None and disposition.action is Action.XFAIL:
        return "xpass" if ok else "xfail"
    return "pass" if ok else "fail"


def _write_step_summary(rows: list[dict], out_dir: Path) -> None:
    """Render a markdown metrics panel and both persist it (``step_summary.md``, picked up
    by the append-step-summary action on the runner) and, when running under Actions,
    append it to ``$GITHUB_STEP_SUMMARY`` directly.

    Two head-level metrics, both of which gate CI -- **relative error** (bf16 only; fp32 has
    no rel-err verdict) and **Frobenius ratio** (every dtype) -- each with a section per dtype
    and one table row per capture mode (eager/compiled). Each cell is one test case
    (op + shape); it is classified by that metric's own verdict (see ``_classify``).
    """

    def _mode(row: dict) -> str:
        return row.get("path") or "all"

    def _line(label: str, rs: list[dict], field: str, per_ops: bool) -> str:
        if per_ops:
            by_op: dict = {}
            for r in rs:
                by_op.setdefault(r["op"], []).append(_classify(r, field))
            kinds = [_reduce_op(v) for v in by_op.values()]
            total = len(by_op)
        else:
            kinds = [_classify(r, field) for r in rs]
            total = len(rs)
        c = dict.fromkeys(_KINDS, 0)
        for k in kinds:
            c[k] += 1
        return f"| {label} | " + " | ".join(str(c[k]) for k in _KINDS) + f" | {total} |"

    # (heading, row-field, bf16_only). rel-err is gated on bf16 only
    metrics = [
        ("Relative error — gates bf16", "relative_norm_verdict", True),
        ("Frobenius ratio — gates every dtype", "frobenius_ratio_verdict", False),
    ]
    model = os.environ.get("NEURON_TRACED_MODEL", "") or "(none)"
    dtypes = sorted({r["dtype"] for r in rows})
    header = "| mode | " + " | ".join(_KINDS) + " | total |"
    sep = "|---|" + "--:|" * (len(_KINDS) + 1)

    lines = [f"# Op-Accuracy Results — `{model}`", ""]
    lines += [
        "Each cell is one test case (op + shape). Both metrics gate CI. Column meanings:",
        "",
        "- **pass** — within the metric threshold.",
        "- **fail** — over the threshold; blocks CI unless the (op, dtype) is xfail-listed.",
        "- **xfail** — a known gap expected to fail; does **not** block CI.",
        "- **xpass** — an xfail that unexpectedly passed (ready to drop from `xfail_registry.py`).",
        "- **skip** — not run.",
        "- **other** — no verdict: int/bool out, no baseline, or a device error. Not gated.",
        "",
        "Two tables per dtype:",
        "",
        "**Tests (ops + shapes)** — one test case per op + shape + mode; each runs the op on "
        + "the device and compares its output to the reference. Rows:",
        "",
        "- `eager` / `compiled` — one capture mode each. An op+shape traced in both modes is "
        + "counted in both rows.",
        "- `total tests` — the sum of all test cases; a cell traced in both modes counts as two.",
        "",
        "**Ops** — the same test cases grouped to one row per op (an op's worst disposition "
        + "wins). Rows:",
        "",
        "- `eager` / `compiled` — one capture mode each.",
        "- `total distinct ops` — ops deduplicated; each op counted once.",
        "",
    ]
    for title, field, bf16_only in metrics:
        lines += [f"## {title}", ""]
        for dtype in dtypes:
            if bf16_only and not _gated(dtype):
                continue
            d = [r for r in rows if r["dtype"] == dtype]
            if not any(r.get(field) in ("within", "over", "invalid") for r in d):
                continue
            modes = sorted({_mode(r) for r in d})
            lines += [f"### dtype = {dtype}", ""]
            for caption, per_ops in [("Tests (ops + shapes)", False), ("Ops", True)]:
                lines += [f"**{caption}**", "", header, sep]
                for mode in modes:
                    lines.append(_line(mode, [r for r in d if _mode(r) == mode], field, per_ops))
                if len(modes) > 1:
                    total_label = "total distinct ops" if per_ops else "total tests"
                    lines.append(_line(total_label, d, field, per_ops))
                lines.append("")

    # Genuine failures (a metric's cell over/invalid and NOT excused by the xfail registry):
    # one table per (metric, dtype), each printing the offending rows. Empty when every known
    # gap is xfail-listed.
    fail_blocks = [
        (title, field, dtype, fails)
        for title, field, bf16_only in metrics
        for dtype in dtypes
        if not (bf16_only and not _gated(dtype))
        for fails in [[r for r in rows if r["dtype"] == dtype and _classify(r, field) == "fail"]]
        if fails
    ]
    if fail_blocks:
        lines += ["## Failing cells", ""]
        for title, field, dtype, fails in fail_blocks:
            lines += [f"### {title} — {dtype} ({len(fails)})", ""]
            if field == "relative_norm_verdict":
                lines += [
                    "| op | mode | arg_shapes | max_rel_err_units | budget |",
                    "|---|---|---|--:|--:|",
                ]
                body = [
                    f"| {r['op']} | {_mode(r)} | {r.get('arg_shapes')} "
                    f"| {r.get('max_rel_err_units')} | {r.get('budget_units')} |"
                    for r in fails[:_MAX_FAILURES_IN_SUMMARY]
                ]
                cols = 5
            else:
                lines += ["| op | mode | arg_shapes | frobenius_ratio |", "|---|---|---|--:|"]
                body = [
                    f"| {r['op']} | {_mode(r)} | {r.get('arg_shapes')} "
                    f"| {r.get('frobenius_ratio')} |"
                    for r in fails[:_MAX_FAILURES_IN_SUMMARY]
                ]
                cols = 4
            lines += body
            if len(fails) > _MAX_FAILURES_IN_SUMMARY:
                lines.append("| … and more |" + " |" * (cols - 1))
            lines.append("")

    md = "\n".join(lines) + "\n"
    try:
        (out_dir / "step_summary.md").write_text(md)
    except Exception as e:  # persisting the panel must never fail the run
        logger.warning(f"could not write step_summary.md: {e}")
    gh = os.environ.get("GITHUB_STEP_SUMMARY")
    if gh:
        try:
            with open(gh, "a") as f:
                f.write(md)
        except Exception as e:
            logger.warning(f"could not append to GITHUB_STEP_SUMMARY: {e}")
