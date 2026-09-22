"""Tests for ``label_selection`` — the label → dist config subset filter.

Pure-function module, so tests pass an in-memory mapping rather than
loading the real ``label_test_selection.json`` (keeps tests independent
of mapping changes).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pytorch_tests_native" / "runner"))

from label_selection import (
    RULE_FORCE_FULL_SUITE,
    RULE_ITEMS_MATCH,
    RULE_NO_LABELS,
    RULE_NO_OVERLAP,
    RULE_TERMINATOR,
    RULE_UNKNOWN_LABELS,
    load_mapping,
    parse_labels_csv,
    select_nodes,
)


# Mapping fixture — small but covers all three "kinds" we care about.
# Keeps tests independent of the real label_test_selection.json so
# mapping changes don't churn this file.
@pytest.fixture
def mapping():
    return {
        "module:dynamo": {"kind": "items", "items": ["dynamo"]},
        "module:profiler": {"kind": "items", "items": ["profiler"]},
        "module:aten-ops": {"kind": "items", "items": ["test_ops.py", "test_meta.py"]},
        "module:distributed": {"kind": "items", "items": ["distributed", "distributed/fsdp"]},
        "module:runtime-core": {"kind": "full_suite"},
        "module:build": {"kind": "full_suite"},
    }


@pytest.fixture
def dist_config():
    """Realistic-ish dist config covering each kind of node we filter."""
    return {
        "description": "test",
        "nodes": [
            {"items": ["dynamo"]},
            {"items": ["profiler"]},
            {"items": ["test_ops.py"], "num_shards": 5},
            {"items": ["test_meta.py"]},
            {"items": ["test_autograd.py"]},  # aten_core — not in any test fixture label
            {"items": ["distributed"]},
            {"items": ["distributed/fsdp"]},
            {"items": ["distributed/tensor"]},
            {"items": ["test_linalg.py", "nn"]},  # multi-item bundle
        ],
    }


# ---------------------------------------------------------------------------
# parse_labels_csv
# ---------------------------------------------------------------------------
def test_parse_labels_empty_returns_empty():
    assert parse_labels_csv("") == []
    assert parse_labels_csv(None) == []


def test_parse_labels_single():
    assert parse_labels_csv("module:dynamo") == ["module:dynamo"]


def test_parse_labels_csv_strips_whitespace_and_dedupes():
    assert parse_labels_csv("module:dynamo, module:dynamo, module:profiler") == [
        "module:dynamo",
        "module:profiler",
    ]


def test_parse_labels_drops_empty_tokens():
    """A trailing comma or double-comma shouldn't produce empty entries."""
    assert parse_labels_csv("module:dynamo,,module:profiler,") == [
        "module:dynamo",
        "module:profiler",
    ]


# ---------------------------------------------------------------------------
# select_nodes — selection precedence
# ---------------------------------------------------------------------------
def test_force_full_suite_overrides_everything(dist_config, mapping):
    """``ciflow:run-full-suite`` always wins, even with restrictive labels."""
    result, decision = select_nodes(
        dist_config, ["module:dynamo"], force_full_suite=True, mapping=mapping
    )
    assert result == dist_config
    assert decision.rule == RULE_FORCE_FULL_SUITE
    assert decision.is_full_suite


def test_no_labels_returns_full_suite(dist_config, mapping):
    """Empty labels (direct push, workflow_dispatch) → safe default."""
    result, decision = select_nodes(dist_config, [], mapping=mapping)
    assert result == dist_config
    assert decision.rule == RULE_NO_LABELS
    assert decision.is_full_suite


def test_full_suite_terminator_label_returns_full_suite(dist_config, mapping):
    """``module:runtime-core`` is a terminator — returns full suite."""
    result, decision = select_nodes(
        dist_config, ["module:runtime-core", "module:dynamo"], mapping=mapping
    )
    assert result == dist_config
    assert decision.rule == RULE_TERMINATOR
    assert decision.terminator_labels == ("module:runtime-core",)


def test_unknown_labels_only_returns_full_suite(dist_config, mapping):
    """Unrecognized labels → safe default (full suite, not empty matrix)."""
    result, decision = select_nodes(
        dist_config, ["something:weird", "another:unknown"], mapping=mapping
    )
    assert result == dist_config
    assert decision.rule == RULE_UNKNOWN_LABELS


def test_single_module_label_filters_to_matching_node(dist_config, mapping):
    result, decision = select_nodes(dist_config, ["module:dynamo"], mapping=mapping)
    assert len(result["nodes"]) == 1
    assert result["nodes"][0]["items"] == ["dynamo"]
    assert decision.rule == RULE_ITEMS_MATCH
    assert decision.matched_labels == ("module:dynamo",)
    assert decision.kept_count == 1
    assert decision.total_count == len(dist_config["nodes"])


def test_multiple_module_labels_union_their_items(dist_config, mapping):
    """``module:dynamo`` + ``module:profiler`` → both nodes kept."""
    result, decision = select_nodes(
        dist_config, ["module:dynamo", "module:profiler"], mapping=mapping
    )
    items = [node["items"] for node in result["nodes"]]
    assert ["dynamo"] in items
    assert ["profiler"] in items
    assert len(result["nodes"]) == 2
    assert set(decision.matched_labels) == {"module:dynamo", "module:profiler"}
    assert set(decision.implied_items) == {"dynamo", "profiler"}


def test_aten_label_keeps_all_aten_ops_nodes(dist_config, mapping):
    """``module:aten-ops`` matches multiple test files → all kept."""
    result, _ = select_nodes(dist_config, ["module:aten-ops"], mapping=mapping)
    flat = {item for node in result["nodes"] for item in node["items"]}
    # Both test_ops.py and test_meta.py nodes kept.
    assert "test_ops.py" in flat
    assert "test_meta.py" in flat
    # test_autograd.py is NOT in aten-ops — must not be present.
    assert "test_autograd.py" not in flat


def test_multi_item_bundle_kept_when_any_item_matches(dist_config, mapping):
    """A node with items ``[test_linalg.py, nn]`` should be kept if a
    label implies either test_linalg.py OR nn — bundles aren't split."""
    bundle_label_mapping = {"module:linalg-only": {"kind": "items", "items": ["test_linalg.py"]}}
    result, _ = select_nodes(dist_config, ["module:linalg-only"], mapping=bundle_label_mapping)
    assert len(result["nodes"]) == 1
    assert result["nodes"][0]["items"] == ["test_linalg.py", "nn"]


def test_distributed_label_keeps_only_matching_distributed_nodes(dist_config, mapping):
    """``module:distributed`` lists ``distributed`` and ``distributed/fsdp``
    but NOT ``distributed/tensor`` (per the fixture); tensor must be dropped."""
    result, _ = select_nodes(dist_config, ["module:distributed"], mapping=mapping)
    items = [node["items"][0] for node in result["nodes"]]
    assert "distributed" in items
    assert "distributed/fsdp" in items
    assert "distributed/tensor" not in items


def test_no_overlap_falls_back_to_full_suite(dist_config, mapping):
    """If labels imply items that don't appear in any dist config node,
    fall back to full suite rather than emit an empty matrix."""
    # No node has items=["nonexistent.py"], so this can never match.
    impossible_mapping = {"module:nonexistent": {"kind": "items", "items": ["nonexistent.py"]}}
    result, decision = select_nodes(dist_config, ["module:nonexistent"], mapping=impossible_mapping)
    assert result == dist_config
    assert decision.rule == RULE_NO_OVERLAP
    # Decision still records what was attempted, useful for the summary panel.
    assert decision.matched_labels == ("module:nonexistent",)
    assert decision.implied_items == ("nonexistent.py",)


def test_full_suite_terminator_with_other_labels_still_wins(dist_config, mapping):
    """Terminator label wins over module-specific labels — order shouldn't matter."""
    result, decision = select_nodes(
        dist_config,
        ["module:dynamo", "module:build"],  # build is a terminator
        mapping=mapping,
    )
    assert result == dist_config
    assert decision.rule == RULE_TERMINATOR


def test_select_nodes_does_not_mutate_input(dist_config, mapping):
    """Caller shouldn't see its dist_config modified by select_nodes."""
    original_node_count = len(dist_config["nodes"])
    select_nodes(dist_config, ["module:dynamo"], mapping=mapping)
    assert len(dist_config["nodes"]) == original_node_count


def test_unknown_labels_alongside_known_labels_are_ignored(dist_config, mapping):
    """Mixing recognized + unrecognized labels → only the recognized ones filter."""
    result, decision = select_nodes(
        dist_config,
        ["module:dynamo", "experimental:something"],
        mapping=mapping,
    )
    assert len(result["nodes"]) == 1
    assert result["nodes"][0]["items"] == ["dynamo"]
    assert decision.matched_labels == ("module:dynamo",)
    # Unknown labels still appear in `labels` (they were inputs), just not matched.
    assert "experimental:something" in decision.labels


# ---------------------------------------------------------------------------
# load_mapping — integration-y test against the real JSON
# ---------------------------------------------------------------------------
def test_load_mapping_default_path_works():
    """The shipped label_test_selection.json must load and parse."""
    mapping = load_mapping()
    # All entries must declare a 'kind' field — schema invariant.
    for label, entry in mapping.items():
        assert "kind" in entry, f"Label {label!r} missing 'kind'"
        assert entry["kind"] in {
            "items",
            "full_suite",
        }, f"Label {label!r} has invalid kind: {entry['kind']!r}"
        if entry["kind"] == "items":
            assert entry.get("items"), f"Label {label!r} kind=items must list non-empty 'items'"


def test_real_mapping_covers_known_labels():
    """Sanity check: the shipped mapping covers labels listed in
    .github/labeler.yml. Catches drift between the two files."""
    mapping = load_mapping()
    expected_labels = {
        "module:aten-ops",
        "module:torch-mlir",
        "module:distributed",
        "module:dynamo",
        "module:profiler",
        "module:memory",
        "module:runtime-core",
        "module:upstream-test-infra",
        "module:build",
    }
    missing = expected_labels - set(mapping.keys())
    assert not missing, f"label_test_selection.json missing entries for: {missing}"


def test_module_ci_is_intentionally_unmapped():
    """``module:ci`` is auto-applied by labeler.yml for non-functional
    plumbing (.github/, tools/, tests/test_infra/, skip_registry/) and
    is intentionally NOT mapped here. A pure-plumbing PR is paired with
    ``ciflow:skip-upstream`` so upstream is skipped entirely; a mixed
    PR (plumbing + real code) gets only the real-code label's items.
    Adding ``module:ci`` to the mapping would re-introduce profiler
    runs for plumbing changes — defeating the design.
    """
    mapping = load_mapping()
    assert "module:ci" not in mapping, (
        "module:ci must NOT be mapped — see docstring. "
        "If you want plumbing PRs to run something, change labeler.yml "
        "to apply a different module label, don't add module:ci here."
    )


def test_load_mapping_custom_path(tmp_path):
    """Caller can pass an explicit mapping path."""
    custom = tmp_path / "custom_mapping.json"
    custom.write_text(
        json.dumps(
            {
                "labels": {
                    "module:foo": {"kind": "items", "items": ["foo.py"]},
                }
            }
        )
    )
    mapping = load_mapping(custom)
    assert "module:foo" in mapping
