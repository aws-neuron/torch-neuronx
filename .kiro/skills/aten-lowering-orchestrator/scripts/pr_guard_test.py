"""Hardware-free tests for the pr-guard PreToolUse hook.

pr-guard.py runs at import (reads stdin, then `sys.exit`), so it is not a
normal module. Test it two ways, both without a Neuron device:
  * exec the source with the guard active and a benign command, then call the
    pure `_effective_cwd` helper from the resulting namespace;
  * run the whole guard and assert the block exit codes (0 allow, 2 block).
"""

import io
import json
from pathlib import Path

SOURCE = Path(__file__).with_name("pr-guard.py").read_text()


def _exec_guard(command, env, cwd="/tmp"):
    """Exec pr-guard.py with a fake stdin and env. Return (exit_code, namespace).

    The guard calls `sys.exit` at the end (or on a block), so catch SystemExit.
    An exit_code of 0 means allow; 2 means block.
    """
    import os
    import sys

    ns = {"__name__": "pr_guard_under_test"}
    payload = {"tool_input": {"command": command}, "cwd": cwd}
    old_environ = dict(os.environ)
    old_stdin = sys.stdin
    os.environ.clear()
    os.environ.update(env)
    sys.stdin = io.StringIO(json.dumps(payload))
    try:
        exec(compile(SOURCE, "pr-guard.py", "exec"), ns)
        code = 0
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 0
    finally:
        sys.stdin = old_stdin
        os.environ.clear()
        os.environ.update(old_environ)
    return code, ns


# --- _effective_cwd (pure) ------------------------------------------------


def _load_ns():
    _, ns = _exec_guard("ls", env={"ATEN_AGENT_GUARD": "1"})
    return ns


def test_effective_cwd_git_dash_c():
    ns = _load_ns()
    assert ns["_effective_cwd"]("git -C /tmp/wt-1769 status") == "/tmp/wt-1769"


def test_effective_cwd_git_dash_c_quoted():
    ns = _load_ns()
    assert ns["_effective_cwd"]("git -C '/tmp/wt-1769' commit") == "/tmp/wt-1769"


def test_effective_cwd_space_path_falls_open():
    # A path with a space is not resolved; the guard returns None and falls
    # open (never a false-positive block).
    ns = _load_ns()
    assert ns["_effective_cwd"]("git -C '/tmp/my wt' commit") is None


def test_effective_cwd_cd_prefix():
    ns = _load_ns()
    assert ns["_effective_cwd"]("cd /tmp/wt && git commit -m x") == "/tmp/wt"


def test_effective_cwd_none_when_absent():
    ns = _load_ns()
    assert ns["_effective_cwd"]("git commit -m x") is None


# --- end-to-end block behavior -------------------------------------------


def test_noop_when_guard_inactive():
    # With no opt-in the hook must never block, even a non-draft PR create.
    code, _ = _exec_guard("gh pr create", env={})
    assert code == 0


def test_blocks_non_draft_pr_create():
    code, _ = _exec_guard("gh pr create --title x", env={"ATEN_AGENT_GUARD": "1"})
    assert code == 2


def test_allows_draft_pr_create():
    code, _ = _exec_guard("gh pr create --draft --title x", env={"ATEN_AGENT_GUARD": "1"})
    assert code == 0


def test_blocks_pr_merge():
    code, _ = _exec_guard("gh pr merge 123", env={"ATEN_AGENT_GUARD": "1"})
    assert code == 2


def test_blocks_force_push():
    code, _ = _exec_guard("git push --force origin lower/x", env={"ATEN_AGENT_GUARD": "1"})
    assert code == 2


def test_blocks_push_to_protected_branch():
    code, _ = _exec_guard("git push origin main", env={"ATEN_AGENT_GUARD": "1"})
    assert code == 2


def test_blocks_push_outside_namespace():
    code, _ = _exec_guard("git push origin feature/x", env={"ATEN_AGENT_GUARD": "1"})
    assert code == 2


def test_allows_push_to_lower_namespace():
    # cwd /tmp is not a repo, so the branch guard fails open; the named target
    # is in an allowed namespace, so the push guard allows it.
    code, _ = _exec_guard("git push origin lower/expm1", env={"ATEN_AGENT_GUARD": "1"})
    assert code == 0
