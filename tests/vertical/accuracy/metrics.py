"""
Compares two tensors down to four scalars, derives cell identity, and builds seeded inputs.

Everything here runs on CPU so the metric layer is testable without hardware.

Normalization is deliberately **global**, not per-element: ``max|T-R| / max|R|`` rather than
per-element ``|T_i - R_i| / |R_i|``. A per-element relative error is unbounded wherever the
reference is near zero.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager

import torch

# ── identity ─────────────────────────────────────────────────────────────────


def cell_id(op_name: str, arg_shapes: dict, dtype: str) -> str:
    """Stable 16-hex-char id for one ``(op, shapes, dtype)`` measurement.

    Deliberately excludes the seed: the seed is *derived from* the cell id (see
    ``seed_for``), so including it would be circular. The row records both.
    """
    payload = json.dumps(
        {"op": op_name, "arg_shapes": arg_shapes, "dtype": dtype}, sort_keys=True, default=str
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def seed_for(cell: str, base: int = 0) -> int:
    """
    Derive this cell's RNG seed from its id.
    ``cell`` is fixed, ``base`` is knob for different seed.
    """
    h = hashlib.sha256(f"{cell}:{base}".encode()).digest()
    return int.from_bytes(h[:4], "big")


# ── input construction ───────────────────────────────────────────────────────
@contextmanager
def seeded(seed: int):
    """Save the initial RNG state, restore it before test exit"""
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        yield
    finally:
        torch.random.set_rng_state(state)


def build_sample(op_info, arg_shapes: dict, dtype: torch.dtype, seed: int):
    """Build a ``SampleInput`` on CPU **at** ``dtype``, deterministically.

    Generating at the target dtype and upcasting for the reference.
    ``_with_out_kwargs`` writes the out buffer w/ sample input.

    Explicitly returns ``None`` when the op's schema cannot be satisfied from ``arg_shapes``.
    """
    with seeded(seed):
        sample = op_info.make_sample(arg_shapes, "cpu", dtype)
        return sample if sample is None else op_info._with_out_kwargs(sample)


# ── the scalars ──────────────────────────────────────────────────────────────
def norms(actual: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    """The four absolute scalars a comparison reduces to.

    FP64 is used so no additional error introduced at comparison time.

    Both L-inf and Frobenius are returned because their blind spots are complementary --
    L-inf catches a spike and ignores element drift, Frobenius catches drift, but dilutes
    a single catastrophical spike.
    """
    a = actual.detach().to("cpu", torch.float64)
    r = ref.detach().to("cpu", torch.float64)
    if a.shape != r.shape:
        raise ValueError(f"shape mismatch: actual {tuple(a.shape)} vs ref {tuple(r.shape)}")
    diff = a - r
    return {
        "max_abs_err": float(diff.abs().max()) if diff.numel() else 0.0,
        "max_abs_ref": float(r.abs().max()) if r.numel() else 0.0,
        "fro_abs_err": float(torch.linalg.vector_norm(diff)),
        "fro_abs_ref": float(torch.linalg.vector_norm(r)),
    }


def to_ulp(abs_err: float, abs_ref: float, unit: float) -> float:
    """Convert an absolute error to a count of error units.

    ``abs_ref == 0`` means the reference is identically zero, so there is no scale to be
    relative to. Returns 0.0 when the error is also zero (a correct all-zero result) and
    ``inf`` otherwise (any error against an exact zero is infinitely wrong in relative
    terms) -- both honest, neither silently passing.
    """
    if abs_ref == 0.0:
        return 0.0 if abs_err == 0.0 else float("inf")
    return (abs_err / abs_ref) / unit


def ulps(n: dict[str, float], unit: float) -> dict[str, float]:
    """``norms()`` output plus ``unit`` -> the two reported ULP figures."""
    return {
        "max_err_ulp": to_ulp(n["max_abs_err"], n["max_abs_ref"], unit),
        "fro_err_ulp": to_ulp(n["fro_abs_err"], n["fro_abs_ref"], unit),
    }
