"""Skip/xfail registry for the op-accuracy gate (op + dtype + shape granularity).

Single source of truth for the cells the accuracy gate should not block on. The gate and the
step-summary panel consult ``lookup(op, dtype, arg_shapes)`` per cell, so an entry can excuse a
whole op+dtype or just one traced shape of it -- the ``arg_shapes`` matched here is the same
dict the traced-shape registry keys on.

Organization mirrors the upstream ``neuron_test_registry`` idea: each ``Group`` is ONE root
cause. Fix the cause, delete the group, and its cells re-enter the gate automatically.

  - ``Action.SKIP``  -- the op is not measured for that dtype (it crashes the node, or its
                        reference is meaningless). No CSV row. Skip is op+dtype only.
  - ``Action.XFAIL`` -- the cell IS measured and its row still lands in the CSV, but a gate
                        failure is *expected* (a known accuracy gap under RCA). It flips back
                        to a normal pass the moment the gap closes; a listed cell that starts
                        passing shows up as ``xpass`` in the panel (drop its entry).

Entry forms (in ``Group.ops``):
  - ``(op, dtype)``                      -- every traced shape of that op+dtype. ``dtype=None``
                                            = all dtypes. Excuses both metrics.
  - ``(op, dtype, arg_shapes)``          -- exactly one traced shape (``arg_shapes`` must equal
                                            the cell's, compared canonically). Excuses both.
  - ``(op, dtype, arg_shapes, metric)``  -- that shape, but only the named metric
                                            (``Metric.RELATIVE`` or ``Metric.FROBENIUS``). The other
                                            metric still gates for real.

Two metrics gate: rel-err (bf16 only) and the Frobenius ratio (every dtype). An entry with no
``metric`` excuses whichever one the cell fails; a ``metric``-tagged entry excuses only that
one, so a gap in the Frobenius ratio does not mask a new rel-err regression on the same cell.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    """What to do with a matched cell. ``str`` subclass so the value is a
    human-readable marker reason prefix while callers compare against the enum."""

    SKIP = "skip"
    XFAIL = "xfail"


class Metric(str, Enum):
    """A gated metric, used as the optional 4th element of an entry to excuse only that one.
    ``str`` subclass so an entry reads cleanly and compares to the CSV's metric name."""

    RELATIVE = "relative"  # relative error (bf16-gated)
    FROBENIUS = "frobenius"  # Frobenius ratio (every dtype)


@dataclass(frozen=True)
class Group:
    """One root cause and the cells it covers. The list a group lives in
    (``XFAIL_GROUPS`` / ``SKIP_GROUPS``) is its action.

    Each entry is ``(op, dtype)`` (every shape; ``dtype=None`` = every dtype),
    ``(op, dtype, arg_shapes)`` (exactly that traced shape), or
    ``(op, dtype, arg_shapes, metric)`` (that shape, only ``Metric.RELATIVE`` or
    ``Metric.FROBENIUS``).
    """

    reason: str
    ops: tuple[tuple, ...] = ()


# Known accuracy gaps the gate expects to fail today. The cell IS measured (its row still lands
# in the CSV) but a gate failure is expected; it flips back to a pass the moment the gap closes.
# Seeded from the 2026-08-27 trn2 run -- bf16 is clean apart from ``full``; the fp32 Frobenius
# gaps are listed per shape where an op fails only some of its shapes (mm / bmm / div / mul),
# and per op where every shape fails (exp / softmax / sigmoid / silu / silu_backward).
XFAIL_GROUPS: list[Group] = [
    Group(
        "bf16 gaps on the 2026-08-31 trn2 sweep (qwen3_235b_ep64_tp4): the exact (op, shape) "
        "cells over the rel-err budget or the Frobenius ratio, tagged with the metric each "
        "fails so the other still gates. A cell flips to xpass and leaves this list when its "
        "gap closes. Pending per-op RCA.",
        (
            (
                "aten::_foreach_norm.Scalar",
                "bfloat16",
                {
                    "ord": 2.0,
                    "self": [
                        [64, 1536, 4096],
                        [64, 4096, 1536],
                        [64, 1536, 4096],
                        [64, 1536, 4096],
                        [64, 4096, 1536],
                        [64, 1536, 4096],
                        [64, 1536, 4096],
                        [64, 4096, 1536],
                        [64, 1536, 4096],
                        [64, 1536, 4096],
                        [64, 4096, 1536],
                        [64, 1536, 4096],
                        [64, 1536, 4096],
                        [64, 4096, 1536],
                        [64, 1536, 4096],
                        [64, 1536, 4096],
                        [64, 4096, 1536],
                        [64, 1536, 4096],
                    ],
                },
            ),
            (
                "aten::_fused_rms_norm",
                "bfloat16",
                {
                    "eps": 1e-06,
                    "input": [1, 2048, 1, 128],
                    "normalized_shape": [128],
                    "weight": [128],
                },
                Metric.FROBENIUS,
            ),
            (
                "aten::_fused_rms_norm",
                "bfloat16",
                {
                    "eps": 1e-06,
                    "input": [1, 2048, 16, 128],
                    "normalized_shape": [128],
                    "weight": [128],
                },
                Metric.FROBENIUS,
            ),
            (
                "aten::_fused_rms_norm",
                "bfloat16",
                {
                    "eps": 1e-06,
                    "input": [1, 2048, 4, 128],
                    "normalized_shape": [128],
                    "weight": [128],
                },
                Metric.FROBENIUS,
            ),
            (
                "aten::_fused_rms_norm",
                "bfloat16",
                {
                    "eps": 1e-06,
                    "input": [1, 2048, 4096],
                    "normalized_shape": [4096],
                    "weight": [4096],
                },
                Metric.FROBENIUS,
            ),
            (
                "aten::_fused_rms_norm",
                "bfloat16",
                {
                    "eps": 1e-06,
                    "input": [1, 2048, 64, 128],
                    "normalized_shape": [128],
                    "weight": [128],
                },
                Metric.FROBENIUS,
            ),
            (
                "aten::_fused_rms_norm",
                "bfloat16",
                {
                    "eps": 1e-06,
                    "input": [1, 512, 4096],
                    "normalized_shape": [4096],
                    "weight": [4096],
                },
                Metric.FROBENIUS,
            ),
            (
                "aten::_grouped_mm",
                "bfloat16",
                {
                    "mat2": [32768, 1536],
                    "offs": {
                        "dtype": "int32",
                        "high": 2,
                        "low": 1,
                        "multiple_of": 4096,
                        "shape": [1],
                    },
                    "self": [4096, 32768],
                },
            ),
            (
                "aten::_grouped_mm",
                "bfloat16",
                {
                    "mat2": [32768, 4096],
                    "offs": {
                        "dtype": "int32",
                        "high": 2,
                        "low": 1,
                        "multiple_of": 1536,
                        "shape": [1],
                    },
                    "self": [1536, 32768],
                },
            ),
            (
                "aten::add_.Tensor",
                "bfloat16",
                {"other": {"dtype": "int64", "shape": []}, "self": {"dtype": "int64", "shape": []}},
                Metric.RELATIVE,
            ),
            (
                "aten::embedding_dense_backward",
                "bfloat16",
                {
                    "grad_output": [1, 2048, 4096],
                    "indices": {"dtype": "int64", "shape": [1, 2048]},
                    "num_weights": 37984,
                    "padding_idx": -1,
                    "scale_grad_by_freq": False,
                },
                Metric.RELATIVE,
            ),
            (
                "aten::index_put",
                "bfloat16",
                {
                    "indices": [
                        {"dtype": "int64", "shape": [256]},
                        {"dtype": "int64", "shape": [256]},
                    ],
                    "self": [256, 37984],
                    "values": [256],
                },
            ),
            (
                "aten::index_put",
                "bfloat16",
                {
                    "indices": [
                        {"dtype": "int64", "shape": [4096]},
                        {"dtype": "int64", "shape": [4096]},
                    ],
                    "self": {"dtype": "int32", "shape": [4096, 64]},
                    "values": {"dtype": "int32", "shape": [4096]},
                },
                Metric.RELATIVE,
            ),
            (
                "aten::index_put_",
                "bfloat16",
                {
                    "indices": [
                        {"dtype": "int64", "shape": [256]},
                        {"dtype": "int64", "shape": [256]},
                    ],
                    "self": [256, 37984],
                    "values": [256],
                },
            ),
            (
                "aten::index_put_",
                "bfloat16",
                {
                    "indices": [
                        {"dtype": "int64", "shape": [4096]},
                        {"dtype": "int64", "shape": [4096]},
                    ],
                    "self": {"dtype": "int32", "shape": [4096, 64]},
                    "values": {"dtype": "int32", "shape": [4096]},
                },
                Metric.RELATIVE,
            ),
            ("aten::log", "bfloat16", {"self": [16, 128, 16]}),
            ("aten::log", "bfloat16", {"self": [256, 1]}),
            (
                "aten::scatter_add",
                "bfloat16",
                {
                    "dim": -1,
                    "index": {"dtype": "int64", "shape": [1, 2048, 8]},
                    "self": [1, 2048, 64],
                    "src": [1, 2048, 8],
                },
            ),
            (
                "aten::scatter_add",
                "bfloat16",
                {
                    "dim": -1,
                    "index": {"dtype": "int64", "shape": [1, 512, 8]},
                    "self": [1, 512, 64],
                    "src": [1, 512, 8],
                },
            ),
            (
                "aten::scatter_add",
                "bfloat16",
                {
                    "dim": 0,
                    "index": {"dtype": "int64", "shape": [4096, 4096]},
                    "self": [2048, 4096],
                    "src": [4096, 4096],
                },
            ),
            (
                "aten::sum.dim_IntList",
                "bfloat16",
                {"dim": [-1], "keepdim": True, "self": [256, 37984]},
            ),
            (
                "aten::sum.dim_IntList",
                "bfloat16",
                {"dim": [0, 1, 2], "self": [1, 2048, 16, 128]},
                Metric.FROBENIUS,
            ),
            (
                "aten::sum.dim_IntList",
                "bfloat16",
                {"dim": [1], "keepdim": True, "self": [512, 8, 4096]},
            ),
            (
                "aten::topk",
                "bfloat16",
                {"dim": -1, "k": 8, "largest": True, "self": [1, 2048, 64], "sorted": False},
            ),
            ("aten::topk", "bfloat16", {"k": 8, "self": [1, 512, 64]}),
        ),
    ),
    Group(
        "fp32 gaps on the 2026-08-31 trn2 sweep (qwen3_235b_ep64_tp4): Neuron diverges from the "
        "fp64 reference more than CPU fp32 does, pushing the Frobenius ratio over threshold "
        "(fp32 has no rel-err verdict). Exact (op, shape) cells. Pending RCA.",
        (
            (
                "aten::_fused_rms_norm",
                "float32",
                {
                    "eps": 1e-06,
                    "input": [1, 2048, 1, 128],
                    "normalized_shape": [128],
                    "weight": [128],
                },
                Metric.FROBENIUS,
            ),
            (
                "aten::_fused_rms_norm",
                "float32",
                {
                    "eps": 1e-06,
                    "input": [1, 2048, 16, 128],
                    "normalized_shape": [128],
                    "weight": [128],
                },
                Metric.FROBENIUS,
            ),
            (
                "aten::_fused_rms_norm",
                "float32",
                {
                    "eps": 1e-06,
                    "input": [1, 2048, 4, 128],
                    "normalized_shape": [128],
                    "weight": [128],
                },
                Metric.FROBENIUS,
            ),
            (
                "aten::_fused_rms_norm",
                "float32",
                {
                    "eps": 1e-06,
                    "input": [1, 2048, 4096],
                    "normalized_shape": [4096],
                    "weight": [4096],
                },
                Metric.FROBENIUS,
            ),
            (
                "aten::_fused_rms_norm",
                "float32",
                {
                    "eps": 1e-06,
                    "input": [1, 2048, 64, 128],
                    "normalized_shape": [128],
                    "weight": [128],
                },
                Metric.FROBENIUS,
            ),
            (
                "aten::_fused_rms_norm",
                "float32",
                {
                    "eps": 1e-06,
                    "input": [1, 512, 4096],
                    "normalized_shape": [4096],
                    "weight": [4096],
                },
                Metric.FROBENIUS,
            ),
            (
                "aten::_softmax",
                "float32",
                {"dim": -1, "half_to_float": False, "self": [1, 2048, 64]},
                Metric.FROBENIUS,
            ),
            (
                "aten::_softmax",
                "float32",
                {"dim": -1, "half_to_float": False, "self": [1, 512, 64]},
                Metric.FROBENIUS,
            ),
            (
                "aten::_softmax_backward_data",
                "float32",
                {
                    "dim": -1,
                    "grad_output": [1, 2048, 64],
                    "input_dtype": 6,
                    "output": [1, 2048, 64],
                },
                Metric.FROBENIUS,
            ),
            (
                "aten::div.Tensor",
                "float32",
                {"other": [1, 2048, 1], "self": [1, 2048, 8]},
                Metric.FROBENIUS,
            ),
            (
                "aten::div.Tensor",
                "float32",
                {"other": [1, 512, 1], "self": [1, 512, 8]},
                Metric.FROBENIUS,
            ),
            (
                "aten::div.Tensor",
                "float32",
                {"other": [], "self": [1, 2048, 1, 128]},
                Metric.FROBENIUS,
            ),
            ("aten::exp", "float32", {"self": [256, 37984]}, Metric.FROBENIUS),
            (
                "aten::index_put",
                "float32",
                {
                    "indices": [
                        {"dtype": "int64", "shape": [256]},
                        {"dtype": "int64", "shape": [256]},
                    ],
                    "self": [256, 37984],
                    "values": [256],
                },
                Metric.FROBENIUS,
            ),
            (
                "aten::index_put_",
                "float32",
                {
                    "indices": [
                        {"dtype": "int64", "shape": [256]},
                        {"dtype": "int64", "shape": [256]},
                    ],
                    "self": [256, 37984],
                    "values": [256],
                },
                Metric.FROBENIUS,
            ),
            ("aten::linalg_vector_norm", "float32", {"ord": 2.0, "self": [18]}, Metric.FROBENIUS),
            ("aten::log", "float32", {"self": [16, 128, 16]}, Metric.FROBENIUS),
            ("aten::log", "float32", {"self": [256, 1]}, Metric.FROBENIUS),
            ("aten::pow_.Scalar", "float32", {"exponent": 0.5, "self": []}, Metric.FROBENIUS),
            ("aten::sigmoid", "float32", {"self": [32768, 1536]}, Metric.FROBENIUS),
            ("aten::silu", "float32", {"self": [32768, 1536]}, Metric.FROBENIUS),
            (
                "aten::silu_backward",
                "float32",
                {"grad_output": [32768, 1536], "self": [32768, 1536]},
                Metric.FROBENIUS,
            ),
            ("aten::sum", "float32", {"self": [256]}, Metric.FROBENIUS),
            (
                "aten::sum.dim_IntList",
                "float32",
                {"dim": [-1], "keepdim": True, "self": [256, 37984]},
                Metric.FROBENIUS,
            ),
            (
                "aten::sum.dim_IntList",
                "float32",
                {"dim": [0, 1], "self": [1, 512, 4096]},
                Metric.FROBENIUS,
            ),
            (
                "aten::sum.dim_IntList",
                "float32",
                {"dim": [1], "keepdim": True, "self": [4096, 4096]},
                Metric.FROBENIUS,
            ),
            (
                "aten::sum.dim_IntList",
                "float32",
                {"dim": [2], "keepdim": True, "self": [1, 512, 4096]},
                Metric.FROBENIUS,
            ),
            (
                "aten::sum.dim_IntList",
                "float32",
                {"dim": [3], "keepdim": True, "self": [1, 2048, 1, 128]},
                Metric.FROBENIUS,
            ),
            (
                "aten::sum.dim_IntList",
                "float32",
                {"dim": [3], "keepdim": True, "self": [1, 2048, 16, 128]},
                Metric.FROBENIUS,
            ),
            (
                "aten::topk",
                "float32",
                {"dim": -1, "k": 8, "largest": True, "self": [1, 2048, 64], "sorted": False},
                Metric.FROBENIUS,
            ),
        ),
    ),
]

# Ops that must not run at all (crash the node, or have no meaningful reference). Skip is
# op+dtype only -- no CSV row is emitted for a skipped cell.
SKIP_GROUPS: list[Group] = [
    Group(
        "Moving the traced foreach input list to the device aborts the NRT runtime (a hard "
        "SIGABRT in allocate_to_device), so the cell cannot be measured and would crash the "
        "worker. Observed for fp32 on the 2026-08-31 trn2 sweep. Pending RCA.",
        (
            ("aten::_foreach_norm.Scalar", "float32"),
            ("aten::_foreach_mul_.Tensor", "float32"),
        ),
    ),
]


@dataclass(frozen=True)
class Disposition:
    """The resolved action + reason for one cell."""

    action: Action
    reason: str


def _shape_key(shapes) -> str | None:
    """Canonical form of an ``arg_shapes`` dict or JSON string; ``None`` matches every shape."""
    if shapes is None:
        return None
    if isinstance(shapes, str):
        try:
            shapes = json.loads(shapes)
        except (ValueError, TypeError):
            return shapes
    return json.dumps(shapes, sort_keys=True, default=str)


def _match(
    groups: list[Group], op: str, dtype: str, shape_key: str | None, metric: str | None
) -> str | None:
    for g in groups:
        for entry in g.ops:
            e_op = entry[0]
            e_dtype = entry[1] if len(entry) > 1 else None
            e_shape = _shape_key(entry[2]) if len(entry) > 2 else None
            e_metric = entry[3] if len(entry) > 3 else None
            if e_op != op or e_dtype not in (None, dtype):
                continue
            if e_shape is not None and e_shape != shape_key:
                # a shape-specific entry only matches when the caller gave that exact shape
                continue
            if e_metric is not None and metric is not None and e_metric != metric:
                # a metric-specific entry only excuses that metric
                continue
            return g.reason
    return None


def lookup(op: str, dtype: str, arg_shapes=None, metric: str | None = None) -> Disposition | None:
    """Disposition for one cell, or ``None`` to run and gate it normally.

    ``arg_shapes`` (dict or the CSV's JSON string) selects the shape; omit it to test only the
    op+dtype-level entries (shape-specific entries won't match). ``metric`` (``Metric.RELATIVE`` /
    ``Metric.FROBENIUS``) asks whether that one metric is excused; omit it to match either metric.
    ``Action.SKIP`` wins over ``Action.XFAIL`` and ignores ``metric`` (a skip is never measured).
    """
    sk = _shape_key(arg_shapes)
    reason = _match(SKIP_GROUPS, op, dtype, sk, None)
    if reason is not None:
        return Disposition(Action.SKIP, reason)
    reason = _match(XFAIL_GROUPS, op, dtype, sk, metric)
    if reason is not None:
        return Disposition(Action.XFAIL, reason)
    return None
