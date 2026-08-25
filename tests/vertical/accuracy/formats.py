"""Per-dtype error models.

A ``NumericFormat`` answers three questions about a dtype:

1. ``unit``            — 1 "unit of error" value = 0.5 ULP = 0.5 adjacent representable spacing
2. ``gated``           — whether this input format needs a normalized relative error gating
3. ``reference(...)``  — upcast tensor to fp32, which is used to rerun and generate reference value

This covers bf16 + fp32 format, mxfp8 to be added
"""

from __future__ import annotations

import torch

# No downcast rounding error check for input w/ precision 2^-24 (fp32) and above
_UNGATED_FROM_PRECISION = 24

# ── the model ────────────────────────────────────────────────────────────────


class NumericFormat:
    """Relative error model for a float dtype: ``fl(x) = x(1 + d)``, ``|d| <= unit``.

    ``precision`` counts mantissa bits *including* the implicit leading bit, so bf16 is 8
    (7 stored) and fp32 is 24.
    """

    def __init__(self, name: str, dtype: torch.dtype, precision: int):
        self.name = name
        self.dtype = dtype
        self.unit = 2.0**-precision
        self.gated = precision < _UNGATED_FROM_PRECISION

    def reference(self, op, sample) -> torch.Tensor:
        """
        Run ``op`` on fp32-upcast inputs.
        The caller must have generated the sample *at* ``self.dtype`` already, e.g. bf16 input,
        we generate bf16 tensors, rerun ops and output downcast to bf16. The diff is arithmatic.
        """
        args = _upcast((sample.input, *sample.args))
        kwargs = {k: _upcast_one(v) for k, v in sample.kwargs.items()}
        return op(*args, **kwargs)


# ── registry ─────────────────────────────────────────────────────────────────

FP32 = NumericFormat("float32", torch.float32, 24)
BF16 = NumericFormat("bfloat16", torch.bfloat16, 8)

FORMATS: dict[str, NumericFormat] = {f.name: f for f in (FP32, BF16)}


def get_format(dtype: torch.dtype | str) -> NumericFormat:
    """Look up a format by ``torch.dtype`` or by its string name."""
    key = str(dtype).removeprefix("torch.") if isinstance(dtype, torch.dtype) else dtype
    if key not in FORMATS:
        raise KeyError(f"no NumericFormat for {dtype!r}; known: {sorted(FORMATS)}")
    return FORMATS[key]


# ── helpers ──────────────────────────────────────────────────────────────────


def _upcast_one(v):
    """fp32-upcast a tensor, or each tensor in a list/tuple (``_foreach_*``, ``cat``).
    Integer and bool tensors pass through: they are exact.
    """
    if isinstance(v, torch.Tensor):
        return v.to(torch.float32) if v.is_floating_point() else v
    if isinstance(v, list | tuple):
        return type(v)(_upcast_one(x) for x in v)
    return v


def _upcast(values):
    return tuple(_upcast_one(v) for v in values)
