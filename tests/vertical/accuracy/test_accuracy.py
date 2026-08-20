"""Phase 2 driver: one measured cell per (op, shape, dtype).

Runs the reference and the device path **in one process**.

Usage::
    pytest tests/vertical/accuracy/test_accuracy.py \\
        --traced-model qwen3_235b_a22b --max-shapes-per-op 0 \\
        -n 32 --tb=no -q 2>&1 | tee accuracy_$(date +%Y%m%d_%H%M%S).log

**Not a gate.** Every cell emits a row and the test passes as long as cells ran.
"""

from __future__ import annotations

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
from internal.neuron_shape_defaults_generated import TRACED_MODEL_OPS
from internal.neuron_shape_registry import SHAPE_REGISTRY
from internal.target_ops import TARGET_ATEN_OP_GROUP

from .budgets import error_budget, has_model, width
from .formats import get_format
from .metrics import build_sample, cell_id, norms, seed_for, ulps

logger = logging.getLogger(__name__)


# ── op selection ─────────────────────────────────────────────────────────────


def _registry_ops() -> list[str]:
    """Ops to sweep. With ``--traced-model``, the set's own declared op list in the declared
    frozenset; otherwise the whole registry, kept to ops with a known error-budget formula."""
    for name in (os.environ.get("NEURON_TRACED_MODEL") or "").split(","):
        declared = TRACED_MODEL_OPS.get(name.strip())
        if declared:
            return sorted(declared)

    return [op for op in SHAPE_REGISTRY.all_ops() if has_model(op)]


def _op_params():
    """
    Per-op rather than per-cell so one worker compiles one op's shapes back to back and hits
    a warm NEFF cache.
    """
    ops = _registry_ops()
    return [pytest.param(op, id=op) for op in ops]


# ── one cell ─────────────────────────────────────────────────────────────────


def _blank_row(op_name: str, op_group: str, arg_shapes: dict, dtype_name: str, cfg: dict) -> dict:
    cell = cell_id(op_name, arg_shapes, dtype_name)
    return {
        "cell_id": cell,
        "op": op_name,
        "op_group": op_group,
        "dtype": dtype_name,
        "arg_shapes": arg_shapes,
        "unit": None,
        "n": None,
        "u_acc": None,
        "order": cfg["order"],
        "max_err_ulp": None,
        "fro_err_ulp": None,
        "budget_ulp": None,
        "verdict": "",
        "max_abs_err": None,
        "max_abs_ref": None,
        "fro_abs_err": None,
        "fro_abs_ref": None,
        "seed": seed_for(cell, cfg["seed_base"]),
        "gates": cfg["gates"],
        "status": "ok",
        "error_cause_raw": None,
    }


def _worst_pair(dev_outs: list[torch.Tensor], ref_outs: list[torch.Tensor], unit: float):
    """
    when outputs are from multiple arithmic ops, take the largest ulp divergence as the worst
    """
    worst = None
    for dev, ref in zip(dev_outs, ref_outs, strict=True):
        n = norms(dev, ref)
        u = ulps(n, unit)
        if worst is None or u["max_err_ulp"] > worst[1]["max_err_ulp"]:
            worst = (n, u)
    return worst


def _exact_match_ulps(dev_outs, ref_outs) -> tuple[dict, dict]:
    """Integer/bool outputs (argmax, comparisons, index results) have no ULP.
    A correct kernel reproduces them exactly
    """
    exact = all(
        torch.equal(d.detach().cpu(), r.detach().cpu())
        for d, r in zip(dev_outs, ref_outs, strict=True)
    )
    err = 0.0 if exact else float("inf")
    return (
        {"max_abs_err": err, "max_abs_ref": 0.0, "fro_abs_err": err, "fro_abs_ref": 0.0},
        {"max_err_ulp": err, "fro_err_ulp": err},
    )


def _measure_cell(op_info: NeuronOpInfo, arg_shapes: dict, dtype_name: str, cfg: dict) -> dict:
    """Measure one (op, shape, dtype) cell. Always returns a complete row."""
    op_name = op_info.op_name
    row = _blank_row(op_name, op_info.op_group, arg_shapes, dtype_name, cfg)

    fmt = get_format(dtype_name)
    row["u_acc"] = get_format(cfg["accumulator"]).unit

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

    unit = fmt.unit
    row["unit"] = unit

    try:
        if ref_outs[0].is_floating_point():
            n, u = _worst_pair(dev_outs, ref_outs, unit)
        else:
            n, u = _exact_match_ulps(dev_outs, ref_outs)
    except Exception as e:
        row["status"] = "no_reference"
        row["error_cause_raw"] = f"compare failed: {e}"[:200]
        return row
    row.update(n)
    row.update(u)

    # Budget and verdict. ``None`` from either layer is a reported disposition, not a pass:
    # an op nobody has modelled has to stay countable
    row["n"] = width(op_name, op_info.op_group, arg_shapes)
    budget = error_budget(
        op_name, op_info.op_group, arg_shapes, unit, row["u_acc"], order=cfg["order"]
    )
    row["budget_ulp"] = budget

    # ``nan`` is the one ULP value that is not a comparison: it means the output held a nan,
    # or that both sides overflowed to inf and the ratio was inf/inf. ``inf`` is
    # *not* included -- ``_exact_match_ulps`` and ``to_ulp``'s zero-reference branch both
    # return it deliberately to mean "wrong", and those must stay countable as "over".
    if math.isnan(row["max_err_ulp"]):
        row["verdict"] = "invalid"
    elif budget is None:
        row["status"] = "excluded" if op_info.op_group == "random" else "no_budget"
    else:
        row["verdict"] = "within" if row["max_err_ulp"] <= budget else "over"
    return row


# ── the test ─────────────────────────────────────────────────────────────────


class TestOpAccuracy:
    """Per-op accuracy sweep: measured ULP vs. derived budget, one row per cell."""

    @pytest.mark.parametrize("op_name", _op_params())
    def test_op(self, op_name: str, accuracy_config: dict, emit_row):
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

        rows = []
        for entry in entries:
            for dtype_name in accuracy_config["dtypes"]:
                row = _measure_cell(op_info, entry.arg_shapes, dtype_name, accuracy_config)
                emit_row(row)
                rows.append(row)
                logger.info(
                    "%s %s n=%s max_ulp=%s budget=%s -> %s",
                    op_name,
                    dtype_name,
                    row["n"],
                    row["max_err_ulp"],
                    row["budget_ulp"],
                    row["verdict"] or row["status"],
                )

        assert rows, f"no cells ran for {op_name}"
