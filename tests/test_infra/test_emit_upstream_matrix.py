"""Tests for ``emit_upstream_matrix.py`` — picks the distribution config
and emits the matrix expansion.

The interesting unit here is ``_resolve_config_path``: which file gets
loaded based on the ``DIST_CONFIG`` env var. The smoke-vs-full config
toggle for CI reruns hangs off this.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pytorch_tests_native" / "runner"))

from distribution_config import expand_distribution_config
from emit_upstream_matrix import (
    CONFIG_DIR,
    DEFAULT_DIST_CONFIG,
    _render_step_summary,
    _resolve_config_path,
)
from label_selection import (
    RULE_FORCE_FULL_SUITE,
    RULE_ITEMS_MATCH,
    RULE_NO_LABELS,
    RULE_NO_OVERLAP,
    RULE_TERMINATOR,
    RULE_UNKNOWN_LABELS,
    SelectionDecision,
)


# ---------------------------------------------------------------------------
# _resolve_config_path
# ---------------------------------------------------------------------------
def test_resolve_config_path_default_when_env_unset(monkeypatch):
    monkeypatch.delenv("DIST_CONFIG", raising=False)
    assert _resolve_config_path() == CONFIG_DIR / DEFAULT_DIST_CONFIG


def test_resolve_config_path_env_overrides_default(monkeypatch):
    monkeypatch.setenv("DIST_CONFIG", "upstream_pytorch_test_distribution_config.smoke.json")
    assert (
        _resolve_config_path()
        == CONFIG_DIR / "upstream_pytorch_test_distribution_config.smoke.json"
    )


def test_resolve_config_path_empty_env_treated_as_unset(monkeypatch):
    # Empty string env var should fall back to the default — guards
    # against shells that pass DIST_CONFIG="" when the workflow input
    # was left blank.
    monkeypatch.setenv("DIST_CONFIG", "")
    assert _resolve_config_path() == CONFIG_DIR / DEFAULT_DIST_CONFIG


def test_resolve_config_path_strips_whitespace(monkeypatch):
    # Some YAML expressions emit a trailing newline; .strip() handles it.
    monkeypatch.setenv("DIST_CONFIG", "  upstream_pytorch_test_distribution_config.smoke.json  ")
    assert (
        _resolve_config_path()
        == CONFIG_DIR / "upstream_pytorch_test_distribution_config.smoke.json"
    )


@pytest.mark.parametrize(
    "bad_value",
    [
        "../something.json",
        "/etc/passwd",
        "a/b.json",
        "evil\\path.json",
    ],
)
def test_resolve_config_path_rejects_path_traversal(monkeypatch, bad_value):
    # The env var must be a basename only — defensive against an attacker
    # or accidental misuse trying to load a config from outside the tests
    # directory. Raises ValueError, never silently loads a different file.
    monkeypatch.setenv("DIST_CONFIG", bad_value)
    with pytest.raises(ValueError, match="basename"):
        _resolve_config_path()


# ---------------------------------------------------------------------------
# Smoke config wiring — verify both configs exist and parse cleanly.
# ---------------------------------------------------------------------------
def test_default_config_exists_and_has_full_coverage():
    # Sanity: the default config covers the full upstream suite. If
    # someone shrinks it accidentally (like we did during a previous
    # debugging pass), this fails fast.
    path = CONFIG_DIR / DEFAULT_DIST_CONFIG
    assert path.is_file(), f"Default distribution config not found: {path}"
    with open(path) as f:
        dist = json.load(f)
    # 18+ logical nodes is the historical floor; main has had this many
    # for a while. Going much lower means someone forgot to restore the
    # full config after smoke-testing.
    assert len(dist["nodes"]) >= 10, (
        f"Default config has only {len(dist['nodes'])} nodes — "
        "did someone leave the smoke config in place?"
    )


def test_smoke_config_exists_and_is_minimal():
    path = CONFIG_DIR / "upstream_pytorch_test_distribution_config.smoke.json"
    assert path.is_file(), f"Smoke config missing: {path}"
    with open(path) as f:
        dist = json.load(f)
    # Smoke config is intentionally tiny — verify it stays that way.
    assert (
        len(dist["nodes"]) <= 5
    ), f"Smoke config has {len(dist['nodes'])} nodes — that's not 'smoke' anymore."


# ---------------------------------------------------------------------------
# _render_step_summary — markdown panel for the matrix-compute job
# ---------------------------------------------------------------------------
def test_render_summary_no_labels_full_suite():
    """No-labels branch renders the safe-default headline and node count."""
    decision = SelectionDecision(rule=RULE_NO_LABELS, kept_count=28, total_count=28)
    text = _render_step_summary(decision, "upstream_pytorch_test_distribution_config.json", 28)
    assert "## PT Upstream — Matrix Selection" in text
    assert "no PR labels" in text
    assert "28 / 28" in text
    # The "| PR labels |" table row only renders when labels are present.
    assert "| PR labels |" not in text


def test_render_summary_force_full_suite_shows_override():
    decision = SelectionDecision(
        rule=RULE_FORCE_FULL_SUITE,
        labels=("ciflow:run-full-suite", "module:dynamo"),
        kept_count=28,
        total_count=28,
    )
    text = _render_step_summary(decision, "config.json", 28)
    assert "ciflow:run-full-suite" in text
    assert "Full suite" in text


def test_render_summary_items_match_lists_matched_labels_and_items():
    """Filtered runs surface the matched labels + implied items so reviewers
    can see *why* the matrix was reduced."""
    decision = SelectionDecision(
        rule=RULE_ITEMS_MATCH,
        labels=("module:dynamo", "module:profiler"),
        matched_labels=("module:dynamo", "module:profiler"),
        implied_items=("dynamo", "profiler"),
        kept_count=2,
        total_count=28,
    )
    text = _render_step_summary(decision, "config.json", 2)
    assert "Filtered" in text
    assert "2 / 28" in text
    assert "`module:dynamo`" in text
    assert "`module:profiler`" in text
    assert "`dynamo`" in text
    assert "`profiler`" in text


def test_render_summary_terminator_shows_terminator_label():
    decision = SelectionDecision(
        rule=RULE_TERMINATOR,
        labels=("module:dynamo", "module:runtime-core"),
        terminator_labels=("module:runtime-core",),
        kept_count=28,
        total_count=28,
    )
    text = _render_step_summary(decision, "config.json", 28)
    assert "Terminator labels" in text
    assert "`module:runtime-core`" in text


def test_render_summary_unknown_labels():
    decision = SelectionDecision(
        rule=RULE_UNKNOWN_LABELS,
        labels=("something:weird",),
        kept_count=28,
        total_count=28,
    )
    text = _render_step_summary(decision, "config.json", 28)
    assert "no recognized" in text
    assert "`something:weird`" in text


def test_render_summary_no_overlap_warns():
    """Fallback branch — labels matched but nothing in dist config — should
    visibly differ from a clean full-suite headline so reviewers notice."""
    decision = SelectionDecision(
        rule=RULE_NO_OVERLAP,
        labels=("module:nonexistent",),
        matched_labels=("module:nonexistent",),
        implied_items=("nonexistent.py",),
        kept_count=28,
        total_count=28,
    )
    text = _render_step_summary(decision, "config.json", 28)
    assert "fallback" in text.lower() or "⚠️" in text
    assert "`module:nonexistent`" in text


def test_render_summary_includes_matrix_node_count():
    """Sharding can expand a single dist-config node into multiple matrix
    nodes (e.g. test_ops.py with num_shards=5 → 5 matrix nodes). The
    panel surfaces the post-shard count too so reviewers know what
    actually ran in parallel."""
    decision = SelectionDecision(
        rule=RULE_ITEMS_MATCH,
        labels=("module:aten-ops",),
        matched_labels=("module:aten-ops",),
        implied_items=("test_ops.py",),
        kept_count=1,
        total_count=28,
    )
    text = _render_step_summary(decision, "config.json", 5)  # 5 shards
    assert "1 / 28" in text  # nodes kept (pre-shard)
    assert "**5**" in text  # matrix nodes (post-shard)


def test_main_groups_the_entries_of_a_node(tmp_path, monkeypatch):
    from emit_upstream_matrix import main

    output = tmp_path / "github_output"
    monkeypatch.setenv("DIST_CONFIG", "internal_test_distribution_config.json")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.delenv("PR_LABELS", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    main()
    matrix = json.loads(output.read_text().removeprefix("matrix="))
    by_node = {entry["node"]: entry for entry in matrix["include"]}
    assert list(by_node) == [
        "aten-unit-tests",
        "aten-tail-tests",
        "dist-unit-tests",
        "dynamic-nki-unit-tests",
    ]
    assert by_node["aten-tail-tests"]["neff_cache"] is True
    assert by_node["dist-unit-tests"]["neff_cache"] is False
    # The matrix label is the display name, so the job title reads cleanly.
    assert by_node["aten-unit-tests"]["label"] == "ATen Unit Tests"
    assert by_node["dynamic-nki-unit-tests"]["label"] == "Dynamic NKI Unit Tests"


def test_render_summary_title_matches_the_suite():
    decision = SelectionDecision(rule=RULE_NO_LABELS, kept_count=14, total_count=14)
    internal = _render_step_summary(decision, "internal_test_distribution_config.json", 4)
    assert "## Internal — Matrix Selection" in internal
    upstream = _render_step_summary(decision, "upstream_pytorch_test_distribution_config.json", 28)
    assert "## PT Upstream — Matrix Selection" in upstream
