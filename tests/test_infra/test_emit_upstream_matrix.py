"""Tests for ``emit_upstream_matrix.py`` — picks the distribution config
and emits the matrix expansion.

The interesting unit here is ``_resolve_config_path``: which file gets
loaded based on the ``DIST_CONFIG`` env var. The smoke-vs-full config
toggle for CI reruns hangs off this.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pytorch_tests_native" / "runner"))

from emit_upstream_matrix import (
    CONFIG_DIR,
    DEFAULT_DIST_CONFIG,
    _resolve_config_path,
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
    import json

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
    import json

    path = CONFIG_DIR / "upstream_pytorch_test_distribution_config.smoke.json"
    assert path.is_file(), f"Smoke config missing: {path}"
    with open(path) as f:
        dist = json.load(f)
    # Smoke config is intentionally tiny — verify it stays that way.
    assert (
        len(dist["nodes"]) <= 5
    ), f"Smoke config has {len(dist['nodes'])} nodes — that's not 'smoke' anymore."
