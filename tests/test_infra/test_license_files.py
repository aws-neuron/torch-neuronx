"""Check the license files and license metadata that the repository and wheel declare."""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
COPYRIGHT = "Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved."


def _declared_license_files():
    tree = ast.parse((REPO_ROOT / "setup.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "license_files":
            return ast.literal_eval(node.value)
    raise AssertionError("setup.py does not pass license_files to setup()")


def _pyproject():
    tomllib = pytest.importorskip("tomllib")  # Python 3.11 and later
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_every_declared_license_file_exists():
    for name in _declared_license_files():
        assert (REPO_ROOT / name).is_file(), f"{name} is declared in setup.py but is missing"


def test_notice_is_declared_and_holds_the_amazon_copyright():
    assert "NOTICE" in _declared_license_files()
    lines = (REPO_ROOT / "NOTICE").read_text(encoding="utf-8").splitlines()
    assert lines == [COPYRIGHT]


def test_pyproject_declares_the_license_expression():
    assert _pyproject()["project"]["license"] == "Apache-2.0"


def test_build_requires_a_setuptools_that_supports_license_expressions():
    requires = _pyproject()["build-system"]["requires"]
    setuptools = [r for r in requires if r.replace(" ", "").startswith("setuptools")]
    assert setuptools == ["setuptools>=77.0.3"]


def test_readme_states_the_license_and_links_the_files():
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "## License\n\nApache-2.0." in text
    assert "](LICENSE)" in text
    assert "](NOTICE)" in text


def test_contributing_asks_contributors_to_confirm_licensing():
    text = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "## Licensing" in text
    # Compare with whitespace collapsed: the sentence may wrap across lines.
    flat = " ".join(text.split())
    assert "We will ask you to confirm the licensing of your contribution." in flat


def test_setup_py_minimum_setuptools_matches_the_build_pin():
    tree = ast.parse((REPO_ROOT / "setup.py").read_text(encoding="utf-8"))
    minimum = next(
        ast.literal_eval(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "_MIN_SETUPTOOLS" for t in node.targets)
    )
    assert ".".join(map(str, minimum)) == "77.0.3"
