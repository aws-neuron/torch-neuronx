"""Hardware-free tests for provision-worktree."""

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "provision-worktree"


def test_script_parses():
    assert subprocess.run(["bash", "-n", str(SCRIPT)], timeout=30).returncode == 0


def test_documents_its_outputs():
    text = SCRIPT.read_text(encoding="utf-8")
    for name in ("WORKTREE=", "BRANCH=", "VENV="):
        assert name in text


def test_missing_slug_fails(tmp_path):
    result = subprocess.run(
        ["bash", str(SCRIPT)], cwd=tmp_path, capture_output=True, text=True, timeout=60
    )
    assert result.returncode != 0
