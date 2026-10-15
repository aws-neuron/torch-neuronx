"""Per-dtype error models.

A ``NumericFormat`` answers three questions about a dtype:

1. ``unit``            — 1 "unit of error" value = 0.5 ULP = 0.5 adjacent representable spacing
2. ``gated``           — whether this input format needs a normalized relative error gating
3. ``reference(...)``  — re-run the op at a higher-precision dtype (bf16 -> fp32, fp32 -> fp64)

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

    def __init__(
        self,
        name: str,
        dtype: torch.dtype,
        precision: int,
        reference_dtype: torch.dtype = torch.float32,
    ):
        self.name = name
        self.dtype = dtype
        self.unit = 2.0**-precision
        self.gated = precision < _UNGATED_FROM_PRECISION
        # Precision the op is re-run at for the reference: the next tier up, so the
        # reference's own rounding is negligible (bf16 -> fp32, fp32 -> fp64).
        self.reference_dtype = reference_dtype

    def reference(self, op, sample, dtype: torch.dtype | None = None) -> torch.Tensor:
        """
        Run ``op`` on inputs cast to ``dtype`` (defaults to ``self.reference_dtype`` -- fp32 for
        bf16 inputs, fp64 for fp32 inputs). Callers can pass an explicit dtype to override, e.g.
        the frobenius-ratio baseline reruns at ``self.dtype`` for the same-precision CPU baseline.
        The caller must have generated the sample *at* ``self.dtype`` already.
        """
        target = self.reference_dtype if dtype is None else dtype
        args = _cast((sample.input, *sample.args), target)
        kwargs = {k: _cast_one(v, target) for k, v in sample.kwargs.items()}
        return op(*args, **kwargs)


# ── registry ─────────────────────────────────────────────────────────────────

FP32 = NumericFormat("float32", torch.float32, 24, reference_dtype=torch.float64)
BF16 = NumericFormat("bfloat16", torch.bfloat16, 8)

FORMATS: dict[str, NumericFormat] = {f.name: f for f in (FP32, BF16)}


def get_format(dtype: torch.dtype | str) -> NumericFormat:
    """Look up a format by ``torch.dtype`` or by its string name."""
    key = str(dtype).removeprefix("torch.") if isinstance(dtype, torch.dtype) else dtype
    if key not in FORMATS:
        raise KeyError(f"no NumericFormat for {dtype!r}; known: {sorted(FORMATS)}")
    return FORMATS[key]


# ── helpers ──────────────────────────────────────────────────────────────────


def _cast_one(v, target: torch.dtype = torch.float32):
    """Cast a floating tensor to ``target``, or each tensor in a list/tuple
    (``_foreach_*``, ``cat``). Integer and bool tensors pass through: exact.
    """
    if isinstance(v, torch.Tensor):
        return v.to(target) if v.is_floating_point() else v
    if isinstance(v, list | tuple):
        return type(v)(_cast_one(x, target) for x in v)
    return v


def _cast(values, target: torch.dtype = torch.float32):
    return tuple(_cast_one(v, target) for v in values)
