"""Error budgets: the error a *correct* kernel should produce in theory.

A budget is a prediction derived from the op's arithmetic. The check is
``max_err_ulp <= budget``, where both sides are counts of ``u_in``.

1 rounding = 1/2 resolution spacing (between adjacent representables) = 1/2 ULP = 1 u_in

``max|T-R| / max|R|``: a dimensionless relative error, so ``1.0`` means "the worst element is
off by one rounding's worth of relative error."

Two contributors, added::

    budget = 1.0                        <- the final cast to the output dtype:
                                           one rounding, one u_in
           + k * g(n) * (u_acc / u_in)  <- accumulating n values

The accumulation happens in the *accumulator's* format (e.g. fp32), so its roundings costs
``u_acc``, not ``u_in``, and g(n) describes the aggregate behavior over n ops, and it is a
modelling choice:

- ``sqrt(n)``       -- probabilistic, sequential summation. What we use. The n-1 roundings
                       treated as independent zero-mean variables accumulating as a random
                       walk: "A New Approach to Probabilistic
                       Rounding Error Analysis", SIAM J. Sci. Comput. 41(5), 2019.
- ``sqrt(log2 n)``  -- probabilistic, pairwise/tree summation. A tiled reducer very likely
                       does this, and at n=8192 it is 25x tighter than sqrt(n). The
                       *deterministic* pairwise bound is ``u * log2(n)`` ("The accuracy of
                       floating point summation", SIAM J. Sci. Comput. 14(4), 1993). The
                       sqrt is the same probabilistic substitution like above sequential
                       case, applied by analogy.
"""

from __future__ import annotations

import math
from collections.abc import Callable

# ── growth models ────────────────────────────────────────────────────────────

GROWTH: dict[str, Callable[[int], float]] = {
    "sequential": math.sqrt,
    "pairwise": lambda n: math.sqrt(max(1.0, math.log2(n))),
}

# Groups whose ops are pure data movement: the output is a permutation/selection of input
# bits, so a correct kernel is bit-exact and any nonzero error is a bug.
BITEXACT_GROUPS = frozenset({"shape", "indexing"})

# No elementwise comparison is meaningful -- two correct RNG implementations disagree.
EXCLUDED_GROUPS = frozenset({"random"})

# One rounding, no accumulation, so width is irrelevant.
FLAT_GROUPS = frozenset({"elementwise", "foreach"})


# ── shape helpers ────────────────────────────────────────────────────────────


def _shape_of(spec) -> tuple[int, ...] | None:
    """Extract a shape from a registry arg spec.

    Entries are either a bare shape tuple or the ``_tensor_from_spec`` dict form
    (``{"shape": ..., "dtype": ..., ...}``), so both have to be handled.
    """
    if isinstance(spec, dict):
        spec = spec.get("shape")
    if isinstance(spec, list | tuple) and all(isinstance(d, int) for d in spec):
        return tuple(spec)
    return None


def _dims(arg_shapes: dict, rank: int) -> list[int]:
    """Normalize the ``dim`` argument to a list of non-negative axes.

    ``dim`` must be **read from the entry**, never assumed. When absent, the reduction is
    over every axis -- which mirrors ``_schema_make_sample``, where a missing optional
    ``dim`` becomes ``None`` (reduce-all) rather than axis 0. Getting this wrong silently
    changes ``n`` by orders of magnitude.
    """
    dim = arg_shapes.get("dim")
    if dim is None:
        return list(range(rank))
    if isinstance(dim, int):
        dim = [dim]
    return [d % rank for d in dim]


# ── width extractors ─────────────────────────────────────────────────────────


def _width_reduction(arg_shapes: dict) -> int | None:
    """Product of the reduced dimensions of ``self``."""
    shape = _shape_of(arg_shapes.get("self"))
    if not shape:
        return None
    return math.prod(shape[d] for d in _dims(arg_shapes, len(shape))) or None


def _width_contract(arg_shapes: dict) -> int | None:
    """
    contract dimension e.g.``K``, the inner dimension of a matmul.
    """
    for name in ("mat2", "batch2", "mat", "other", "weight"):
        shape = _shape_of(arg_shapes.get(name))
        if shape and len(shape) >= 2:
            # ``linear`` stores weight as (out, in), so its contraction is the last axis;
            # mm/bmm/baddbmm store (K, N), so theirs is the second-to-last.
            return shape[-1] if name == "weight" else shape[-2]
    shape = _shape_of(arg_shapes.get("self"))
    return shape[-1] if shape else None


def _width_axis(arg_shapes: dict) -> int | None:
    """Size of the single axis a softmax-like op reduces over."""
    shape = _shape_of(arg_shapes.get("self"))
    dim = arg_shapes.get("dim")
    if isinstance(dim, list | tuple):
        dim = dim[0] if dim else None
    if not shape or not isinstance(dim, int):
        return None
    return shape[dim % len(shape)]


def _width_normalized(arg_shapes: dict) -> int | None:
    """Product of ``normalized_shape`` for layernorm-family ops."""
    ns = arg_shapes.get("normalized_shape")
    if isinstance(ns, list | tuple) and all(isinstance(d, int) for d in ns):
        return math.prod(ns) or None
    shape = _shape_of(arg_shapes.get("self"))
    return shape[-1] if shape else None


# ── the model table ──────────────────────────────────────────────────────────
#
# base op name -> (const, coeff, width_fn). Budget is
# ``const + coeff * g(width) * (u_acc / u_in)``.
#
# ``const`` counts the roundings that do not scale with n, one unit each: 1 for the output
# cast alone, 2 for a norm layer (cast + scale-and-shift), 3 for softmax (cast + reciprocal +
# exp's own relative error). Everything else rounds in the accumulator and is already carried
# by the ``u_acc / u_in`` factor.

_MODELS: dict[str, tuple[float, float, Callable[[dict], int | None]]] = {
    # reductions
    "sum": (1.0, 1.0, _width_reduction),
    "nansum": (1.0, 1.0, _width_reduction),
    "mean": (1.0, 1.0, _width_reduction),
    "prod": (1.0, 1.0, _width_reduction),
    "norm": (1.0, 1.0, _width_reduction),
    "linalg_vector_norm": (1.0, 1.0, _width_reduction),
    "cumsum": (1.0, 1.0, _width_reduction),
    # contractions
    "mm": (1.0, 1.0, _width_contract),
    "bmm": (1.0, 1.0, _width_contract),
    "matmul": (1.0, 1.0, _width_contract),
    "addmm": (1.0, 1.0, _width_contract),
    "baddbmm": (1.0, 1.0, _width_contract),
    "linear": (1.0, 1.0, _width_contract),
    # normalized exponentials -- const covers exp + reciprocal on top of the reduction
    "_softmax": (3.0, 1.0, _width_axis),
    "softmax": (3.0, 1.0, _width_axis),
    "_log_softmax": (3.0, 1.0, _width_axis),
    "log_softmax": (3.0, 1.0, _width_axis),
    # Norm layers: two reductions over the axis (mean, then variance) -> coeff 2.
    "native_layer_norm": (2.0, 2.0, _width_normalized),
    "layer_norm": (2.0, 2.0, _width_normalized),
    # rms_norm is x / sqrt(mean(x^2) + eps) * w -- a *single* reduction.
    "rms_norm": (2.0, 1.0, _width_normalized),
}


def _base(op_name: str) -> str:
    """``aten::sum.dim_IntList`` -> ``sum``."""
    return op_name.removeprefix("aten::").split(".")[0].rstrip("_")


# ── public API ───────────────────────────────────────────────────────────────


def has_model(op_name: str) -> bool:
    """Whether this op has a budget model, independent of any particular shape."""
    return _base(op_name) in _MODELS


def width(op_name: str, op_group: str, arg_shapes: dict) -> int | None:
    """Reduction width ``n`` for this op and these shapes, or ``None`` if unmodelled.

    ``None`` is a real answer, not a failure: the caller reports ``status = no_budget`` so
    the measurement stays visible instead of vanishing from the totals.

    An explicit ``_MODELS`` entry outranks ``op_group``. The group comes from
    ``TARGET_ATEN_OP_GROUP``, a coarse taxonomy shared with the perf suites, and it puts
    ``linear`` and ``baddbmm`` in ``elementwise`` -- so consulting the group first silently
    reported ``n = 1`` for a contraction over 4096 and hid the entire width analysis behind a
    flat allowance.
    """
    if op_group in BITEXACT_GROUPS:
        return 0
    model = _MODELS.get(_base(op_name))
    if model is not None:
        return model[2](arg_shapes)
    return 1 if op_group in FLAT_GROUPS else None


def error_budget(
    op_name: str,
    op_group: str,
    arg_shapes: dict,
    u_in: float,
    u_acc: float,
    order: str = "sequential",
) -> float | None:
    """Units of ``u_in`` a correct kernel is permitted. ``None`` means unmodelled.

    ``u_acc`` is the *accumulator's* unit roundoff, not the input's -- see the module
    docstring for why that is the parameter the whole budget turns on.

    Precedence mirrors ``width()``: an explicit ``_MODELS`` entry beats ``op_group``, because
    the group taxonomy is shared with the perf suites and classifies ``linear`` and
    ``baddbmm`` as ``elementwise``.
    """
    if op_group in EXCLUDED_GROUPS:
        return None
    if op_group in BITEXACT_GROUPS:
        return 0.0

    model = _MODELS.get(_base(op_name))
    if model is None:
        return 1.0 if op_group in FLAT_GROUPS else None
    const, coeff, width_fn = model
    n = width_fn(arg_shapes)
    if not n:
        return None
    if order not in GROWTH:
        raise ValueError(f"unknown order {order!r}; known: {sorted(GROWTH)}")
    return const + coeff * GROWTH[order](n) * (u_acc / u_in)
