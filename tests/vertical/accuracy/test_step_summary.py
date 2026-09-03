"""CPU coverage for the step-summary cell classification and op rollup (``conftest``).

Exercises ``_classify`` (per-cell disposition, incl. shape-level xfail) and ``_reduce_op``
(op-level rollup precedence) directly, plus a smoke check that ``_write_step_summary`` renders
both metrics. No device needed.
"""

from __future__ import annotations

import json

from . import conftest as c
from . import xfail_registry as reg
from .xfail_registry import Group, Metric

# Synthetic shapes for shape-level matching -- independent of the live registry content, which
# is regenerated per accuracy run. These tests cover the classification logic, not live entries.
_SHAPE = {
    "mat2": {"dtype": "float32", "shape": [8, 4096]},
    "self": {"dtype": "float32", "shape": [128, 8]},
}
_OTHER_SHAPE = {
    "mat2": {"dtype": "float32", "shape": [8, 8]},
    "self": {"dtype": "float32", "shape": [8, 8]},
}


def _row(op, dtype, *, verdict="", frobenius="", status="ok", arg_shapes=None, path="eager"):
    return {
        "op": op,
        "dtype": dtype,
        "relative_norm_verdict": verdict,
        "frobenius_ratio_verdict": frobenius,
        "status": status,
        "arg_shapes": json.dumps(arg_shapes or {}, sort_keys=True),
        "path": path,
    }


class TestClassify:
    def test_skipped_status(self):
        assert (
            c._classify(_row("aten::relu", "bfloat16", status="skipped"), "relative_norm_verdict")
            == "skip"
        )

    def test_pass_and_fail_for_unlisted_op(self):
        assert (
            c._classify(_row("aten::relu", "bfloat16", verdict="within"), "relative_norm_verdict")
            == "pass"
        )
        assert (
            c._classify(_row("aten::relu", "bfloat16", verdict="over"), "relative_norm_verdict")
            == "fail"
        )
        assert (
            c._classify(_row("aten::relu", "bfloat16", verdict="invalid"), "relative_norm_verdict")
            == "fail"
        )

    def test_empty_verdict_and_errors_are_other_not_pass(self):
        # ungated fp32 rel-err verdict is "" -> other; a device error likewise (verdict "").
        assert (
            c._classify(_row("aten::relu", "float32", verdict=""), "relative_norm_verdict")
            == "other"
        )
        assert (
            c._classify(
                _row("aten::relu", "bfloat16", status="failed_lowering"), "relative_norm_verdict"
            )
            == "other"
        )

    def test_op_level_xfail_maps_fail_to_xfail_and_pass_to_xpass(self, monkeypatch):
        # An op+dtype xfail entry excuses every shape: a fail reads xfail, a pass reads xpass.
        monkeypatch.setattr(reg, "XFAIL_GROUPS", [Group("gap", (("aten::foo", "float32"),))])
        monkeypatch.setattr(reg, "SKIP_GROUPS", [])
        assert (
            c._classify(_row("aten::foo", "float32", frobenius="over"), "frobenius_ratio_verdict")
            == "xfail"
        )
        assert (
            c._classify(_row("aten::foo", "float32", frobenius="within"), "frobenius_ratio_verdict")
            == "xpass"
        )

    def test_shape_level_entry_matches_only_that_shape(self, monkeypatch):
        monkeypatch.setattr(
            reg, "XFAIL_GROUPS", [Group("gap", (("aten::foo", "float32", _SHAPE),))]
        )
        monkeypatch.setattr(reg, "SKIP_GROUPS", [])
        listed = _row("aten::foo", "float32", frobenius="over", arg_shapes=_SHAPE)
        other = _row("aten::foo", "float32", frobenius="over", arg_shapes=_OTHER_SHAPE)
        assert c._classify(listed, "frobenius_ratio_verdict") == "xfail"
        assert (
            c._classify(other, "frobenius_ratio_verdict") == "fail"
        )  # unlisted shape stays a fail

    def test_listed_shape_that_passes_reads_xpass(self, monkeypatch):
        monkeypatch.setattr(
            reg, "XFAIL_GROUPS", [Group("gap", (("aten::foo", "float32", _SHAPE),))]
        )
        monkeypatch.setattr(reg, "SKIP_GROUPS", [])
        row = _row("aten::foo", "float32", frobenius="within", arg_shapes=_SHAPE)
        assert c._classify(row, "frobenius_ratio_verdict") == "xpass"

    def test_metric_tagged_entry_has_no_crossover_xpass(self, monkeypatch):
        # A FROBENIUS-only entry excuses the Frobenius fail (xfail) but leaves rel-err as a normal
        # pass -- not xpass -- so the panel never shows an undroppable xpass.
        monkeypatch.setattr(
            reg,
            "XFAIL_GROUPS",
            [Group("frobenius gap", (("aten::foo", "bfloat16", _SHAPE, Metric.FROBENIUS),))],
        )
        monkeypatch.setattr(reg, "SKIP_GROUPS", [])
        frobenius_row = _row("aten::foo", "bfloat16", frobenius="over", arg_shapes=_SHAPE)
        rel_row = _row("aten::foo", "bfloat16", verdict="within", arg_shapes=_SHAPE)
        assert c._classify(frobenius_row, "frobenius_ratio_verdict") == "xfail"
        assert c._classify(rel_row, "relative_norm_verdict") == "pass"


class TestReduceOp:
    def test_fail_beats_everything(self):
        assert c._reduce_op(["pass", "xfail", "xpass", "fail"]) == "fail"

    def test_xfail_beats_xpass_and_pass(self):
        assert c._reduce_op(["xpass", "pass", "xfail"]) == "xfail"

    def test_all_pass(self):
        assert c._reduce_op(["pass", "pass"]) == "pass"

    def test_only_other_and_skip(self):
        assert c._reduce_op(["other", "skip"]) == "other"


class TestWriteStepSummary:
    def test_renders_both_metrics_and_failing_table(self, tmp_path, monkeypatch):
        # foo is xfail-listed; bar's bf16 fail is not, so only bar reaches the failing table.
        monkeypatch.setattr(reg, "XFAIL_GROUPS", [Group("gap", (("aten::foo", "float32"),))])
        monkeypatch.setattr(reg, "SKIP_GROUPS", [])
        rows = [
            _row("aten::add", "bfloat16", verdict="within", frobenius="within"),
            _row("aten::bar", "bfloat16", verdict="over"),  # unlisted bf16 -> real fail
            _row("aten::foo", "float32", frobenius="over"),  # xfail-listed
        ]
        c._write_step_summary(rows, tmp_path)
        md = (tmp_path / "step_summary.md").read_text()
        assert "## Relative error" in md and "## Frobenius ratio" in md
        assert "## Failing cells" in md
        assert "aten::bar" in md  # the unlisted bf16 fail surfaces
        assert "aten::foo" not in md  # xfail-listed, excluded from the failing table
