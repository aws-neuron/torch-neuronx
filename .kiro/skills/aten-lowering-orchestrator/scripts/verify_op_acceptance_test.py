"""Regression tests for the _coverage_row / check_roofline contract and the
platform-limitation allowlist gate.

_coverage_row returns a (row, exact) tuple. A prior bug unpacked it as a single
value, so check_compile / check_backward / check_roofline crashed. These tests
pin the 2-tuple shape and prove check_roofline consumes it without a crash.

The allowlist tests parse the real platform-limitations.md and prove the
registry gate accepts a detailed per-op reason but rejects a vague bucket. No
Neuron device is needed; verify_op_acceptance imports cleanly (main is guarded).
"""

from pathlib import Path
from types import SimpleNamespace

import verify_op_acceptance as v

ALLOWLIST = str(Path(__file__).resolve().parents[1] / "references" / "platform-limitations.md")


def _write_csv(tmp_path, rows):
    header = "op_name,roofline_ratio,roofline_category\n"
    body = "".join(f"{r[0]},{r[1]},{r[2]}\n" for r in rows)
    p = tmp_path / "coverage.csv"
    p.write_text(header + body)
    return str(p)


# --- _coverage_row --------------------------------------------------------


def test_coverage_row_exact_match(tmp_path):
    csv_path = _write_csv(tmp_path, [("aten::mm", "1.5", "mm")])
    row, exact = v._coverage_row(csv_path, "aten::mm")
    assert exact is True
    assert row["roofline_ratio"] == "1.5"


def test_coverage_row_base_name_fallback(tmp_path):
    # No exact `.Scalar` overload; fall back to the base name, exact=False.
    csv_path = _write_csv(tmp_path, [("aten::fmod", "0.9", "elementwise")])
    row, exact = v._coverage_row(csv_path, "aten::fmod.Scalar")
    assert exact is False
    assert row["roofline_category"] == "elementwise"


def test_coverage_row_missing_returns_none(tmp_path):
    csv_path = _write_csv(tmp_path, [("aten::mm", "1.5", "mm")])
    row, exact = v._coverage_row(csv_path, "aten::expm1")
    assert row is None
    assert exact is False


def test_coverage_row_is_two_tuple(tmp_path):
    csv_path = _write_csv(tmp_path, [("aten::mm", "1.5", "mm")])
    result = v._coverage_row(csv_path, "aten::mm")
    assert isinstance(result, tuple) and len(result) == 2


# --- check_roofline (the previously-crashing caller) ----------------------


def _args(csv_path, require_roofline=False, threshold=2.0):
    return SimpleNamespace(
        coverage_csv=csv_path,
        roofline_threshold=threshold,
        require_roofline=require_roofline,
    )


def test_check_roofline_records_ratio(tmp_path):
    csv_path = _write_csv(tmp_path, [("aten::mm", "1.5", "mm")])
    verdict = v.Verdict("aten::mm")
    v.check_roofline(verdict, "aten::mm", _args(csv_path))
    roofline = next(c for c in verdict.checks if c.name == "roofline")
    assert roofline.gating is False  # roofline is advisory only
    assert verdict.advisory["roofline_ratio"] == "1.5"


def test_check_roofline_over_threshold_is_advisory(tmp_path):
    csv_path = _write_csv(tmp_path, [("aten::expm1", "3.0", "unknown")])
    verdict = v.Verdict("aten::expm1")
    v.check_roofline(verdict, "aten::expm1", _args(csv_path))
    roofline = next(c for c in verdict.checks if c.name == "roofline")
    assert roofline.gating is False
    assert roofline.status != v.FAILED  # never a gating failure


def test_check_roofline_never_gates_even_with_require(tmp_path):
    # --require-roofline is inert: even a trusted category over threshold stays
    # advisory. Roofline never contributes a FAIL.
    csv_path = _write_csv(tmp_path, [("aten::mm", "5.0", "mm")])
    verdict = v.Verdict("aten::mm")
    v.check_roofline(verdict, "aten::mm", _args(csv_path, require_roofline=True))
    roofline = next(c for c in verdict.checks if c.name == "roofline")
    assert roofline.gating is False
    assert roofline.status != v.FAILED


def test_check_roofline_no_csv(tmp_path):
    verdict = v.Verdict("aten::mm")
    v.check_roofline(verdict, "aten::mm", _args(None))
    roofline = next(c for c in verdict.checks if c.name == "roofline")
    assert roofline.gating is False
    assert roofline.status == v.NOT_PROVIDED


# --- platform-limitation allowlist gate -----------------------------------


def test_dtype_superset_reason_is_allowed():
    # A detailed per-op reason for a proven dtype superset PASSES the gate.
    patterns = v._load_allowlist(ALLOWLIST)
    reason = (
        "aten reference kernel does not support bf16/fp16 for "
        "_upsample_bilinear2d_aa; Neuron proves both bit-close vs an fp32 oracle"
    )
    assert v._reason_allowed(reason, patterns)


def test_vague_opinfo_reason_is_rejected():
    # A generic bucket must still FAIL; a broken OpInfo needs a per-op reason.
    patterns = v._load_allowlist(ALLOWLIST)
    assert not v._reason_allowed("opinfo broken", patterns)


def test_fewer_dtypes_reason_is_rejected():
    # Neuron supporting FEWER dtypes than aten is a real gap, not a superset.
    patterns = v._load_allowlist(ALLOWLIST)
    assert not v._reason_allowed("neuron does not support bf16 for this op", patterns)
