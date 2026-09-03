"""Unit coverage for the accuracy skip/xfail registry lookup.

Runs on CPU with no device -- it only exercises the pure ``(op, dtype, shape) -> Disposition``
matching, which the gate (``test_accuracy``) and the step-summary panel both rely on.
"""

from __future__ import annotations

import json

from . import xfail_registry as reg
from .xfail_registry import Action, Group, Metric, lookup

# Synthetic shapes to exercise matching. Kept independent of the live registry content, which
# is regenerated per accuracy run -- these tests cover the lookup logic, not specific entries.
_SHAPE = {
    "mat2": {"dtype": "float32", "shape": [8, 4096]},
    "self": {"dtype": "float32", "shape": [128, 8]},
}
_OTHER_SHAPE = {
    "mat2": {"dtype": "float32", "shape": [8, 8]},
    "self": {"dtype": "float32", "shape": [8, 8]},
}


class TestLookup:
    def test_no_match_returns_none(self, monkeypatch):
        monkeypatch.setattr(reg, "XFAIL_GROUPS", [])
        monkeypatch.setattr(reg, "SKIP_GROUPS", [])
        assert lookup("aten::relu", "bfloat16") is None

    def test_dtype_specific_entry_does_not_leak_to_other_dtype(self, monkeypatch):
        # A float32-only entry must not excuse the same op at bf16.
        monkeypatch.setattr(reg, "XFAIL_GROUPS", [Group("fp32 gap", (("aten::foo", "float32"),))])
        monkeypatch.setattr(reg, "SKIP_GROUPS", [])
        assert lookup("aten::foo", "float32").action is Action.XFAIL
        assert lookup("aten::foo", "bfloat16") is None


class TestShapeMatch:
    def _shape_entry(self, monkeypatch):
        monkeypatch.setattr(
            reg, "XFAIL_GROUPS", [Group("shape gap", (("aten::foo", "float32", _SHAPE),))]
        )
        monkeypatch.setattr(reg, "SKIP_GROUPS", [])

    def test_shape_specific_entry_matches_only_that_shape(self, monkeypatch):
        self._shape_entry(monkeypatch)
        assert lookup("aten::foo", "float32", _SHAPE).action is Action.XFAIL
        # a different shape of the same op+dtype is NOT excused
        assert lookup("aten::foo", "float32", _OTHER_SHAPE) is None

    def test_shape_specific_entry_needs_a_shape(self, monkeypatch):
        self._shape_entry(monkeypatch)
        # Querying op+dtype with no shape doesn't match a shape-specific entry.
        assert lookup("aten::foo", "float32") is None

    def test_arg_shapes_as_json_string_matches(self, monkeypatch):
        self._shape_entry(monkeypatch)
        # The panel passes the CSV's JSON string; the gate passes the dict -- both canonicalize.
        assert lookup("aten::foo", "float32", json.dumps(_SHAPE)).action is Action.XFAIL

    def test_op_level_entry_matches_any_shape(self, monkeypatch):
        monkeypatch.setattr(reg, "XFAIL_GROUPS", [Group("op gap", (("aten::foo", "float32"),))])
        monkeypatch.setattr(reg, "SKIP_GROUPS", [])
        any_shape = {"self": {"dtype": "float32", "shape": [3, 3]}}
        assert lookup("aten::foo", "float32", any_shape).action is Action.XFAIL


class TestMetric:
    def test_tagged_entry_excuses_only_that_metric(self, monkeypatch):
        monkeypatch.setattr(
            reg,
            "XFAIL_GROUPS",
            [Group("frobenius gap", (("aten::foo", "float32", _SHAPE, Metric.FROBENIUS),))],
        )
        monkeypatch.setattr(reg, "SKIP_GROUPS", [])
        assert lookup("aten::foo", "float32", _SHAPE, Metric.FROBENIUS).action is Action.XFAIL
        assert lookup("aten::foo", "float32", _SHAPE, Metric.RELATIVE) is None
        # a metric-agnostic query still matches a tagged entry
        assert lookup("aten::foo", "float32", _SHAPE).action is Action.XFAIL

    def test_untagged_entry_excuses_both_metrics(self, monkeypatch):
        monkeypatch.setattr(
            reg, "XFAIL_GROUPS", [Group("both", (("aten::foo", "float32", _SHAPE),))]
        )
        monkeypatch.setattr(reg, "SKIP_GROUPS", [])
        assert lookup("aten::foo", "float32", _SHAPE, Metric.RELATIVE).action is Action.XFAIL
        assert lookup("aten::foo", "float32", _SHAPE, Metric.FROBENIUS).action is Action.XFAIL


class TestLiveRegistry:
    def test_entries_are_well_formed(self):
        # Content-independent sanity: every live entry is a valid (op[, dtype[, shape]]) tuple.
        for g in (*reg.XFAIL_GROUPS, *reg.SKIP_GROUPS):
            assert g.reason
            for entry in g.ops:
                assert 1 <= len(entry) <= 4
                assert isinstance(entry[0], str) and entry[0].startswith("aten::")
                if len(entry) > 3:
                    assert entry[3] in (Metric.RELATIVE, Metric.FROBENIUS)


class TestMatchSemantics:
    def test_dtype_none_matches_every_dtype(self, monkeypatch):
        monkeypatch.setattr(reg, "XFAIL_GROUPS", [Group("all dtypes", (("aten::foo", None),))])
        monkeypatch.setattr(reg, "SKIP_GROUPS", [])
        assert lookup("aten::foo", "bfloat16").action is Action.XFAIL
        assert lookup("aten::foo", "float32").action is Action.XFAIL

    def test_skip_wins_over_xfail_for_the_same_cell(self, monkeypatch):
        monkeypatch.setattr(reg, "XFAIL_GROUPS", [Group("x", (("aten::foo", "bfloat16"),))])
        monkeypatch.setattr(reg, "SKIP_GROUPS", [Group("s", (("aten::foo", "bfloat16"),))])
        assert lookup("aten::foo", "bfloat16").action is Action.SKIP

    def test_reason_is_carried_through(self, monkeypatch):
        monkeypatch.setattr(reg, "SKIP_GROUPS", [Group("crashes the node", (("aten::foo", None),))])
        monkeypatch.setattr(reg, "XFAIL_GROUPS", [])
        assert lookup("aten::foo", "bfloat16").reason == "crashes the node"


class TestActionEnum:
    def test_values_are_human_readable_marker_prefixes(self):
        assert Action.SKIP.value == "skip"
        assert Action.XFAIL.value == "xfail"
        assert Action.SKIP == "skip"  # str subclass: usable directly as a marker-reason prefix
