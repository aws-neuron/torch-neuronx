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

logger = logging.getLogger(__name__)

_WORKER_ID = os.environ.get("PYTEST_XDIST_WORKER", "main")

_ROWS_DIR = (
    Path(os.environ.get("NEURON_COVERAGE_METRICS_DIR", tempfile.gettempdir()))
    / "neuron_accuracy_rows"
)

# accuracy output csv columns.
ROW_FIELDS = [
    "cell_id",
    "op",
    "op_group",
    "dtype",
    "path",
    "arg_shapes",
    "unit",
    "max_rel_err_units",
    "budget_units",
    "verdict",
    "max_abs_err",
    "max_abs_ref",
    "fro_abs_err_base",
    "fro_ratio",
    "fro_ratio_verdict",
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
        default="bfloat16",
        help=(
            "Comma-separated dtypes to sweep. This is the dtype-scalability seam: adding a "
            "format means implementing its NumericFormat and naming it here, not touching "
            "the driver. float32 and bfloat16 are the only two Neuron has; anything else "
            "raises KeyError rather than silently reporting a wrong unit."
        ),
    )


@pytest.fixture(scope="session")
def accuracy_config(request) -> dict:
    """Resolved knobs, plus the gate values that decided which shapes are eligible."""
    opt = request.config.getoption
    return {
        "seed_base": opt("--seed-base"),
        "dtypes": [s.strip() for s in opt("--dtypes").split(",") if s.strip()],
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

    # One CSV per capture path (``row["path"]``); an unsplit run (path=None) writes one CSV.
    by_path: dict = {}
    for row in rows:
        by_path.setdefault(row.get("path"), []).append(row)
    for path, prows in sorted(by_path.items(), key=lambda kv: str(kv[0])):
        out_prefix = f"{prefix}_{path}" if path else prefix
        output = Path(f"{out_prefix}_{ts}.csv")
        with open(output, "w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=ROW_FIELDS, quoting=csv.QUOTE_MINIMAL, extrasaction="ignore"
            )
            w.writeheader()
            for row in prows:
                w.writerow({k: _cell(row.get(k)) for k in ROW_FIELDS})
        logger.info(f"wrote {len(prows)} rows -> {output}")

    _summarize(rows)
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
        key = r.get("verdict") if r.get("status") == "ok" else r.get("status", "unknown")
        tally[str(key)] = tally.get(str(key), 0) + 1
    logger.info("accuracy disposition: " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))
