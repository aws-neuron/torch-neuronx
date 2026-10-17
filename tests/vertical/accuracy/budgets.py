"""Error budgets: the normwise relative error a *correct* kernel is allowed.

The check is ``max_rel_err_units <= budget_units``, where

    max_rel_err_units = (max|T-R| / max|R|) / unit_in

is a **normwise relative error** -- one dimensionless ratio for the whole tensor, normalized
by L_infinity -- expressed in the format's unit roundoff ``unit = 2**-P``.

1 rounding = 1/2 the spacing between adjacent representables = 1/2 ULP
, so ``1.0`` unit means "the worst element is off by one rounding's worth of relative error".
"""

from __future__ import annotations

# ── policy ───────────────────────────────────────────────────────────────────
# 2.0 units = 1 full ULP = 1 input format's machine epsilon.
#
ARITHMETIC_BUDGET_UNITS = 2.0

GROUP_BUDGET_UNITS: dict[str, float | None] = {
    "shape": 0.0,
    "indexing": 0.0,
    # Two RNG implementations disagree elementwise, None type exclude this comparison
    "random": None,
}

EXCLUDED_GROUPS = frozenset(g for g, b in GROUP_BUDGET_UNITS.items() if b is None)


def error_budget(op_group: str, *, gated: bool = True) -> float | None:
    """Relative error tolerance. ``None`` means no verdict.``gated`` is the cell dtype's
     ``NumericFormat.gated``: ``False`` at fp32 and finer, no downcast rounding error.

    Bit-exactness is a property for ``view`` type op, which must reproduce its input
    regardless of dtype.
    """
    if op_group in GROUP_BUDGET_UNITS:
        return GROUP_BUDGET_UNITS[op_group]
    return ARITHMETIC_BUDGET_UNITS if gated else None
