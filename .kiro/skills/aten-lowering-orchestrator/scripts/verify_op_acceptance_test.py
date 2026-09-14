"""Regression tests for the _coverage_row / check_roofline contract and the
platform-limitation allowlist gate.

_coverage_row returns a (row, exact) tuple. A prior bug unpacked it as a single
value, so check_compile / check_backward / check_roofline crashed. These tests
pin the 2-tuple shape and prove check_roofline consumes it without a crash.

The allowlist tests parse the real platform-limitations.md and prove the
registry gate accepts a detailed per-op reason but rejects a vague bucket. No
Neuron device is needed; verify_op_acceptance imports cleanly (main is guarded).
"""

import json
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


# --- REVIEW verdict: precedent-cited, non-allowlisted skip/xfail -----------

# A baseline group that a candidate can cite as a real precedent.
_BASE = [
    {
        "kind": "xfail",
        "reason": "reference numerics ulp for exp/erf family",
        "tests": ["test_reference_numerics_normal_erf"],
        "patterns": [],
        "folders": [],
    }
]


def _registry_args(tmp_path, cand_groups):
    (tmp_path / "base.json").write_text(json.dumps(_BASE))
    (tmp_path / "cand.json").write_text(json.dumps(_BASE + cand_groups))
    return SimpleNamespace(
        candidate_registry=str(tmp_path / "cand.json"),
        allowlist=ALLOWLIST,
        baseline_registry=str(tmp_path / "base.json"),
        op_nodeids=None,
    )


def _registry_status(tmp_path, cand_groups):
    vd = v.Verdict("aten::foo")
    v.check_registry(vd, "aten::foo", _registry_args(tmp_path, cand_groups))
    return vd.checks[-1].status


def _xfail(reason):
    return {
        "kind": "xfail",
        "reason": reason,
        "tests": ["test_ref_foo"],
        "patterns": [],
        "folders": [],
    }


def test_allowlisted_reason_passes_registry(tmp_path):
    assert _registry_status(tmp_path, [_xfail("complex dtypes not supported on Neuron")]) == v.OK


def test_non_allowlisted_no_precedent_fails(tmp_path):
    assert _registry_status(tmp_path, [_xfail("reference numerics ulp for foo")]) == v.FAILED


def test_non_allowlisted_valid_precedent_is_review(tmp_path):
    reason = "reference numerics ulp for foo; precedent: test_reference_numerics_normal_erf"
    assert _registry_status(tmp_path, [_xfail(reason)]) == v.NEEDS_REVIEW


def test_bogus_precedent_fails(tmp_path):
    reason = "foo bar; precedent: test_does_not_exist_xyz"
    assert _registry_status(tmp_path, [_xfail(reason)]) == v.FAILED


def test_deny_reason_with_precedent_still_fails(tmp_path):
    # A deny pattern is a hard FAIL; a precedent citation cannot rescue it.
    reason = "tensor-likes are not close on float32; precedent: test_reference_numerics_normal_erf"
    assert _registry_status(tmp_path, [_xfail(reason)]) == v.FAILED


def test_finalize_review_below_fail_above_pass():
    # REVIEW wins only when every hard check is OK; a FAIL still beats it.
    vd = v.Verdict("aten::foo")
    vd.add(v.Check("upstream", True, v.OK))
    vd.add(v.Check("registry", True, v.NEEDS_REVIEW))
    assert vd.finalize(strict_missing=False) == v.REVIEW

    vd = v.Verdict("aten::foo")
    vd.add(v.Check("upstream", True, v.FAILED))
    vd.add(v.Check("registry", True, v.NEEDS_REVIEW))
    assert vd.finalize(strict_missing=False) == v.FAIL


# --- check_upstream: an op left skipped is not "passing" ------------------


def _upstream_report(tmp_path, records):
    p = tmp_path / "upstream.json"
    p.write_text(json.dumps({"tests": records}))
    return str(p)


def _upstream_args(report_path):
    return SimpleNamespace(
        upstream_report=report_path,
        upstream_match_nodeids=None,
        upstream_report_runxfail=None,
        baseline_upstream_report=None,
    )


def _nid(name):
    return f"test/test_ops.py::TestCommonNEURON::test_dtypes_{name}_neuron_float32"


def test_upstream_all_skipped_op_fails(tmp_path):
    # The pod never removed the op from the skip registry: its tests are skipped,
    # 0 passed. Skipped is not passing -- this must FAIL, not silently PASS.
    report = _upstream_report(
        tmp_path,
        [
            {"nodeid": _nid("nansum"), "outcome": "skipped"},
            {"nodeid": _nid("nansum").replace("float32", "float16"), "outcome": "skipped"},
        ],
    )
    vd = v.Verdict("aten::nansum")
    v.check_upstream(vd, "aten::nansum", _upstream_args(report))
    up = next(c for c in vd.checks if c.name == "upstream")
    assert up.status == v.FAILED
    assert "skip registry" in up.detail


def test_upstream_one_pass_among_skips_is_ok(tmp_path):
    # A real lowering has a passing witness even if some samples skip for a
    # legitimate platform limit. One PASS clears the no-witness gate.
    report = _upstream_report(
        tmp_path,
        [
            {"nodeid": _nid("nansum"), "outcome": "passed"},
            {"nodeid": _nid("nansum").replace("float32", "float64"), "outcome": "skipped"},
        ],
    )
    vd = v.Verdict("aten::nansum")
    v.check_upstream(vd, "aten::nansum", _upstream_args(report))
    up = next(c for c in vd.checks if c.name == "upstream")
    assert up.status == v.OK


def test_upstream_all_xfail_not_flagged_by_skip_gate(tmp_path):
    # All samples xfail (e.g. fp64-only), 0 skipped: the skip gate must NOT fire.
    # The registry / --runxfail legs handle a genuine all-xfail op.
    report = _upstream_report(
        tmp_path,
        [{"nodeid": _nid("nansum"), "outcome": "xfailed"}],
    )
    vd = v.Verdict("aten::nansum")
    v.check_upstream(vd, "aten::nansum", _upstream_args(report))
    up = next(c for c in vd.checks if c.name == "upstream")
    assert up.status == v.OK


# --- _upstream_selector ---------------------------------------------------


def test_upstream_selector_default_uses_keyword():
    # No --upstream-match-nodeids: fall back to the keyword heuristic.
    args = SimpleNamespace(upstream_match_nodeids=None)
    select = v._upstream_selector(args, "aten::fmod")
    assert select("test/test_ops.py::TestCommonNEURON::test_dtypes_fmod_neuron_float32")
    assert not select("test/test_ops.py::TestCommonNEURON::test_dtypes_add_neuron_float32")


def test_upstream_selector_explicit_nodeids_match_by_id(tmp_path):
    # An op whose test name differs from the aten name: keyword misses it, the
    # explicit set matches it exactly.
    nid = (
        "test/test_ops.py::TestCommonNEURON::"
        "test_compare_cpu_nn_functional_interpolate_trilinear_neuron_float32"
    )
    p = tmp_path / "op_nodeids.txt"
    p.write_text("# collected\n" + nid + "\n\n")
    args = SimpleNamespace(upstream_match_nodeids=str(p))
    select = v._upstream_selector(args, "aten::upsample_trilinear3d")
    assert select(nid)
    assert select("  " + nid + "  ")  # whitespace tolerant
    assert not select(nid.replace("float32", "float64"))
    # keyword alone would never match this op name
    assert not v._op_keyword_matches(nid, "aten::upsample_trilinear3d")
