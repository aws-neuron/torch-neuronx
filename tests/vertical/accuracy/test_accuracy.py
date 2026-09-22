"""Phase 2 driver: sweeps every registered ``(op, shape)`` across all configured dtypes.

Runs the reference and the device path **in one process**.

``--traced-model`` selects the traced op+shape set; with no model the sweep fails loudly.
The dtype sweep defaults to both bf16 and fp32; ``--max-shapes-per-op 0`` sweeps every
registered shape::

    pytest tests/vertical/accuracy/test_accuracy.py \\
        --traced-model qwen3_235b_ep64_tp4 --max-shapes-per-op 0 \\
        -n 32 --tb=no -q 2>&1 | tee accuracy_$(date +%Y%m%d_%H%M%S).log

**This gates on both metrics.** Every cell emits a CSV row, and the item fails if any cell is
over/invalid on the rel-err budget (bf16 only -- fp32 has no rel-err verdict) OR on the
Frobenius ratio (every dtype). Known gaps are excused per ``(op, dtype, shape)`` by
``xfail_registry.py`` (consulted in-body: an excused failing cell does not fail the item, and a
skip-listed op+dtype is not measured).
"""

from __future__ import annotations

import functools
import logging
import math
import os
import sys
from pathlib import Path

import pytest
import torch

import torch_neuronx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from internal import NeuronOpInfo
from internal.common import allocate_to_device as _allocate_to_device
from internal.common import classify_error as _classify_error
from internal.common import flatten_tensors as _flatten_tensors
from internal.neuron_op_info import _resolve_op
from internal.neuron_shape_defaults_generated import (
    SMOKE_SUBSET_OP_MODE,
    SMOKE_SUBSET_OPS,
)
from internal.target_ops import TARGET_ATEN_OP_GROUP

from .budgets import EXCLUDED_GROUPS, error_budget
from .formats import get_format
from .metrics import (
    FROBENIUS_RATIO_THRESHOLD,
    FrobeniusVerdict,
    build_sample,
    cell_id,
    frobenius_ratio,
    norms,
    rel_err_units,
    seed_for,
)
from .xfail_registry import Action, Metric
from .xfail_registry import lookup as xfail_disposition

logger = logging.getLogger(__name__)


# ── op selection ─────────────────────────────────────────────────────────────
_NO_MODEL = None


@functools.cache
def _selected_model() -> str | None:
    """The ``--traced-model`` name that actually selected an op set, or ``None``."""
    for name in (os.environ.get("NEURON_TRACED_MODEL") or "").split(","):
        name = name.strip()
        if SMOKE_SUBSET_OPS.get(name):
            return name
    return None


def _registry_ops() -> list[str]:
    """Ops to sweep: the declared op list by ``--traced-model``.
    Selecting nothing is a loud failure, in ``_op_params``.
    """
    model = _selected_model()
    return sorted(SMOKE_SUBSET_OPS[model]) if model else []


def _op_mode(op_name: str) -> str:
    """``eager_only`` / ``compile_only`` / ``both_modes`` for this op in the selected subset's
    capture, or ``""`` when the subset declares no mode partition.
    """
    return (SMOKE_SUBSET_OP_MODE.get(_selected_model()) or {}).get(op_name, "")


def _op_params():
    """
    ops params helper
    """
    ops = _registry_ops()
    if not ops:
        return [pytest.param(_NO_MODEL, id="no-traced-model")]
    return [pytest.param(op, id=op) for op in ops]


def _is_xfail(op: str, dtype: str, arg_shapes: dict, metric: Metric) -> bool:
    """Whether this (op, dtype, shape) cell is an expected failure of ``metric`` in the registry."""
    disp = xfail_disposition(op, dtype, arg_shapes, metric)
    return disp is not None and disp.action is Action.XFAIL


# ── one cell ─────────────────────────────────────────────────────────────────


def _blank_row(op_name: str, op_group: str, arg_shapes: dict, dtype_name: str, cfg: dict) -> dict:
    cell = cell_id(op_name, arg_shapes, dtype_name)
    return {
        "cell_id": cell,
        "op": op_name,
        "op_group": op_group,
        "op_mode": _op_mode(op_name),
        "dtype": dtype_name,
        "path": None,
        "arg_shapes": arg_shapes,
        "unit": None,
        "max_rel_err_units": None,
        "budget_units": None,
        "relative_norm_verdict": "",
        "max_abs_err": None,
        "max_abs_ref": None,
        "frobenius_abs_err_base": None,
        "frobenius_ratio": None,
        "frobenius_ratio_verdict": "",
        "seed": seed_for(cell, cfg["seed_base"]),
        "gates": cfg["gates"],
        "status": "ok",
        "error_cause_raw": None,
    }


def _worst_pair(dev_outs: list[torch.Tensor], ref_outs: list[torch.Tensor], unit: float):
    """
    when outputs are from multiple arithmic ops, take the largest relative error as the worst
    """
    worst = None
    for dev, ref in zip(dev_outs, ref_outs, strict=True):
        n = norms(dev, ref)
        u = rel_err_units(n, unit)
        if worst is None or u["max_rel_err_units"] > worst[1]["max_rel_err_units"]:
            worst = (n, u)
    return worst


def _exact_match_units(dev_outs, ref_outs) -> tuple[dict, dict]:
    """Integer/bool outputs (argmax, comparisons, index results) have no rounding error.
    A correct kernel reproduces them exactly
    """
    exact = all(
        torch.equal(d.detach().cpu(), r.detach().cpu())
        for d, r in zip(dev_outs, ref_outs, strict=True)
    )
    err = 0.0 if exact else float("inf")
    return ({"max_abs_err": err, "max_abs_ref": 0.0}, {"max_rel_err_units": err})


def _cell_paths(entry) -> list:
    """Capture paths an entry was traced on: e.g. ``["eager"]``,``["compiled"]``, or both.
    ``[None]`` when the entry carries no capture path, e.g. builtin shapes."""
    return sorted(entry.paths) or [None]


def _worst_frobenius_ratio(dev_outs, base_outs, ref_outs):
    """Frobenius ratio ||trn_low - ref_high||_F / ||cpu_low - ref_high||_F per output *tensor*
    (a tensor-wise norm ratio, not element-wise), worst across an op's outputs (single-output
    ops yield the one ratio). ``frobenius_trn`` = ||trn_low - ref_high|| (numerator),
    ``frobenius_cpu`` = ||cpu_low - ref_high|| (denominator); the zero-denominator case
    (bit-exact CPU baseline)
    is guarded inside ``frobenius_ratio`` -- it returns 0.0 if the device is also exact, else inf,
    and never divides by zero. Returns (ratio, frobenius_trn, frobenius_cpu)."""
    worst = None
    for dev, base, ref in zip(dev_outs, base_outs, ref_outs, strict=True):
        frobenius_trn = norms(dev, ref)["frobenius_abs_err"]
        frobenius_cpu = norms(base, ref)["frobenius_abs_err"]
        if frobenius_cpu == 0.0:
            r = 0.0 if frobenius_trn == 0.0 else float("inf")
        else:
            r = frobenius_ratio(frobenius_trn, frobenius_cpu)
        if worst is None or r > worst[0]:
            worst = (r, frobenius_trn, frobenius_cpu)
    return worst  # (ratio, frobenius_trn, frobenius_cpu)


def _measure_cell(op_info: NeuronOpInfo, arg_shapes: dict, dtype_name: str, cfg: dict) -> dict:
    """Measure one (op, shape, dtype) cell. Always returns a complete row."""
    op_name = op_info.op_name
    row = _blank_row(op_name, op_info.op_group, arg_shapes, dtype_name, cfg)

    fmt = get_format(dtype_name)

    # Build initial sample once on CPU **at the target dtype**.
    try:
        sample = build_sample(op_info, arg_shapes, fmt.dtype, row["seed"])
    except Exception as e:
        row["status"] = "no_input_sample"
        row["error_cause_raw"] = f"{type(e).__name__}: {e}"[:200]
        return row
    if sample is None:
        row["status"] = "no_input_sample"
        return row
    sample.kwargs.pop("requires_grad", None)

    resolved = _resolve_op(op_name)
    if resolved is None:
        row["status"] = "unresolved"
        return row

    # Allocate to device *before* running the reference. For in-place ops the reference run
    # mutates its own (upcast) copy, and the device tensors must be snapshots of the
    # pre-mutation input, not of whatever the reference left behind.
    try:
        dev_args = tuple(_allocate_to_device(a, "neuron") for a in (sample.input, *sample.args))
        dev_kwargs = {k: _allocate_to_device(v, "neuron") for k, v in sample.kwargs.items()}
    except Exception as e:
        row["status"] = "failed_lowering"
        row["error_cause_raw"] = f"allocation failed: {e}"[:200]
        return row

    try:
        with torch.no_grad():
            out_dev = resolved(*dev_args, **dev_kwargs)
        torch_neuronx.synchronize()
    except Exception as e:
        exc = str(e).replace("\n", " ")
        row["status"] = _classify_error(exc)
        row["error_cause_raw"] = exc[:200]
        return row

    try:
        with torch.no_grad():
            out_ref = fmt.reference(resolved, sample)
    except Exception as e:
        row["status"] = "no_reference"
        row["error_cause_raw"] = f"{type(e).__name__}: {e}"[:200]
        return row

    dev_outs = [t.detach().to("cpu") for t in _flatten_tensors(out_dev)]
    ref_outs = _flatten_tensors(out_ref)
    if not dev_outs or len(dev_outs) != len(ref_outs):
        row["status"] = "no_reference"
        row["error_cause_raw"] = f"output arity {len(dev_outs)} vs ref {len(ref_outs)}"
        return row

    # Fro-ratio denominator baseline: the op on CPU at the *candidate* dtype (the "ideal"
    # rounding a reference device would show) -- i.e. the reference, just without the upcast.
    # This is the last read of ``sample``, so an in-place op mutating it here is harmless.
    try:
        with torch.no_grad():
            base_out = fmt.reference(resolved, sample, dtype=fmt.dtype)
        base_outs = [t.detach().cpu() for t in _flatten_tensors(base_out)]
    except Exception:
        base_outs = None

    unit = fmt.unit
    row["unit"] = unit

    try:
        if ref_outs[0].is_floating_point():
            n, u = _worst_pair(dev_outs, ref_outs, unit)
        else:
            n, u = _exact_match_units(dev_outs, ref_outs)
    except Exception as e:
        row["status"] = "no_reference"
        row["error_cause_raw"] = f"compare failed: {e}"[:200]
        return row
    row.update(n)
    row.update(u)

    # Budget and verdict. ``None`` is a reported disposition, not a pass: this keeps it countable.
    budget = error_budget(op_info.op_group, gated=fmt.gated)
    row["budget_units"] = budget

    # ``nan`` is the one measured value that is not a comparison: it means the output held a nan,
    # or that both sides overflowed to inf and the ratio was inf/inf. ``inf`` is
    # *not* included -- ``_exact_match_units`` and ``to_units``'s zero-reference branch both
    # return it deliberately to mean "wrong", and those must stay countable as "over".
    if math.isnan(row["max_rel_err_units"]):
        row["relative_norm_verdict"] = "invalid"
    elif budget is None:
        row["status"] = "excluded" if op_info.op_group in EXCLUDED_GROUPS else "no_gate"
    else:
        row["relative_norm_verdict"] = "within" if row["max_rel_err_units"] <= budget else "over"

    # Frobenius ratio = ||trn_low - ref_high|| / ||cpu_low - ref_high||: device vs the CPU
    # baseline at the candidate dtype, both against the higher-precision reference (fp32 for
    # bf16, fp64 for fp32). Floating outputs only (int/bool outputs have no rounding scale).
    if ref_outs[0].is_floating_point() and base_outs and len(base_outs) == len(ref_outs):
        ratio, _frobenius_trn, frobenius_cpu = _worst_frobenius_ratio(dev_outs, base_outs, ref_outs)
        row["frobenius_abs_err_base"] = frobenius_cpu
        row["frobenius_ratio"] = ratio
        if math.isnan(ratio):
            row["frobenius_ratio_verdict"] = FrobeniusVerdict.INVALID.value
        elif ratio <= FROBENIUS_RATIO_THRESHOLD:
            row["frobenius_ratio_verdict"] = FrobeniusVerdict.WITHIN.value
        else:
            row["frobenius_ratio_verdict"] = FrobeniusVerdict.OVER.value
    return row


# ── the test ─────────────────────────────────────────────────────────────────


class TestOpAccuracy:
    """Per-op accuracy sweep: measured relative error vs. the flat budget, one row per cell."""

    @pytest.mark.parametrize("op_name", _op_params())
    def test_op(self, op_name: str, dtype_name: str, accuracy_config: dict, emit_row):
        if op_name is _NO_MODEL:
            pytest.fail(
                "no op set selected: pass --traced-model with one of "
                f"{sorted(SMOKE_SUBSET_OPS)}."
            )

        op_group = TARGET_ATEN_OP_GROUP.get(op_name, "unknown")
        op_info = NeuronOpInfo.from_schema(op_name, op_group=op_group)

        if not op_info.op_resolved:
            emit_row(
                {
                    **_blank_row(op_name, op_group, {}, "n/a", accuracy_config),
                    "status": "unresolved",
                }
            )
            pytest.skip(f"{op_name}: op not found")

        # Same gate path as every other suite here: registry order, size cap, per-op
        # top-N, and the value-level entry validators.
        entries = op_info._registry_entries()
        if not entries:
            emit_row(
                {
                    **_blank_row(op_name, op_group, {}, "n/a", accuracy_config),
                    "status": "no_input_sample",
                }
            )
            pytest.skip(f"{op_name}: no registry shapes")

        # SKIP registry (op+dtype): don't measure, but still emit a ``skipped`` row per
        # (shape, mode) so the op shows up in the per-(dtype, mode) tables.
        # (lookup can also return an op-level XFAIL disposition here; only SKIP is acted on.)
        disposition = xfail_disposition(op_name, dtype_name)
        if disposition is not None and disposition.action is Action.SKIP:
            for entry in entries:
                for path in _cell_paths(entry):
                    emit_row(
                        {
                            **_blank_row(
                                op_name, op_group, entry.arg_shapes, dtype_name, accuracy_config
                            ),
                            "status": "skipped",
                            "path": path,
                        }
                    )
            pytest.skip(disposition.reason)

        # Both metrics gate: the rel-err budget (only gated formats, i.e. bf16, carry one --
        # fp32 rel-err verdict is "") and the Frobenius ratio (every dtype). A cell fails the
        # gate if either is over/invalid AND it is not excused by the xfail registry (matched
        # per op+dtype+shape).
        #
        # Scope: only *measured* accuracy verdicts gate. A cell with an empty verdict -- a
        # device/lowering error (``failed_lowering``, a classified runtime error), ``no_input_
        # sample``, ``unresolved``, ``no_reference`` -- becomes ``other`` in the panel (never
        # ``pass``) and does NOT fail the item. Op runnability/lowering is owned by the vertical
        # coverage suites (test_target_ops_*); this accuracy suite only judges onboarded ops
        # that actually run.
        gated = get_format(dtype_name).gated
        rows = []
        failures = []
        for entry in entries:
            paths = _cell_paths(entry)
            row = _measure_cell(op_info, entry.arg_shapes, dtype_name, accuracy_config)
            # A shape traced in both paths is measured once and reported under each.
            for path in paths:
                prow = row if path is None else {**row, "path": path}
                emit_row(prow)
                rows.append(prow)
            rel_fail = gated and row["relative_norm_verdict"] in ("over", "invalid")
            frobenius_fail = row["frobenius_ratio_verdict"] in ("over", "invalid")
            # Excuse each metric independently: a Frobenius gap must not mask a new rel-err
            # regression on the same cell (and vice versa).
            if rel_fail and not _is_xfail(op_name, dtype_name, entry.arg_shapes, Metric.RELATIVE):
                failures.append(("rel-err", row))
            if frobenius_fail and not _is_xfail(
                op_name, dtype_name, entry.arg_shapes, Metric.FROBENIUS
            ):
                failures.append(("frobenius-ratio", row))
            logger.info(
                "%s %s max_rel_err=%s budget=%s frobenius_ratio=%s -> %s",
                op_name,
                dtype_name,
                row["max_rel_err_units"],
                row["budget_units"],
                row["frobenius_ratio"],
                row["relative_norm_verdict"] or row["status"],
            )

        assert rows, f"no cells ran for {op_name} [{dtype_name}]"

        if failures:
            detail = "; ".join(
                f"[{metric}] {r['arg_shapes']} rel_err={r['max_rel_err_units']} "
                f"frobenius_ratio={r['frobenius_ratio']}"
                for metric, r in failures[:10]
            )
            more = "" if len(failures) <= 10 else f" (+{len(failures) - 10} more)"
            pytest.fail(
                f"{op_name} [{dtype_name}]: {len(failures)} cell(s) over the accuracy budget: "
                f"{detail}{more}",
                pytrace=False,
            )
