#!/usr/bin/env python3

"""
PreToolUse guard for the aten-lowering agent workflow.

Blocks the git/gh actions an autonomous agent must never take, for any user:
force-push, marking a PR ready, merging, non-draft PR creation, pushes outside
the op-lowering branch namespaces, and -- the branch guard -- committing or
pushing from the WRONG branch. The last one exists because the coordinator and
the implementer can share a single working tree: if the checkout is moved
between turns, an unqualified `git commit --amend` lands on whatever branch HEAD
happens to point at (a `lower/*` PR branch, say) instead of the branch the agent
was told to work on. That is a silent, hard-to-notice mistake, so we refuse it.

Exit 2 blocks the call (stderr is shown to the agent); exit 0 lets it proceed.

Opt-in: this hook is registered project-wide as a PreToolUse hook (matcher
`Bash`), so it runs on every Bash call in every session in this repo. To avoid
blocking normal human git/gh work and other skills, it is a no-op UNLESS the
aten-lowering agent workflow is active -- signalled by `ATEN_AGENT_GUARD` (or
the orchestrator's per-Implementer `ATEN_AGENT_EXPECTED_BRANCH`) being set in the
environment `claude` was launched with. Absent either, `_guard_active()` returns
False and the guard exits 0 immediately. Launch the workflow with
`ATEN_AGENT_GUARD=1 claude ...` to enforce.

Env:
  ATEN_AGENT_GUARD -- master opt-in. Set (to any non-empty value) to enable the
      guard; unset disables it entirely (default for teammates / other work).
  ATEN_AGENT_EXPECTED_BRANCH -- when the orchestrator sets this to the branch it
      told the agent to work on, any commit/amend/push from a different branch is
      blocked. Its presence also implies the guard is active. Absent, the guard
      still blocks commits from a detached HEAD or from main/master (never correct
      for this workflow) -- but only once the guard itself is active.
"""

import json
import os
import re
import subprocess
import sys


def block(msg):
    sys.stderr.write(f"pr-guard: {msg}\n")
    sys.exit(2)


def _guard_active():
    """The guard enforces only inside the aten-lowering agent workflow.

    Fail-open by default: with no opt-in signal in the environment, this hook is
    a no-op so it never blocks normal human git/gh work (or any other skill) in a
    repo that ships this project-wide `Bash` hook.
    """
    return bool(
        (os.environ.get("ATEN_AGENT_GUARD") or "").strip()
        or (os.environ.get("ATEN_AGENT_EXPECTED_BRANCH") or "").strip()
    )


if not _guard_active():
    sys.exit(0)


try:
    payload = json.load(sys.stdin)
    cmd = (payload.get("tool_input") or {}).get("command", "")
except Exception:
    sys.exit(0)  # unparseable -> don't interfere


def _effective_cwd(command):
    """Directory the git command actually runs in.

    The harness reports the tool cwd as the project root, even when the command
    is `cd <worktree> && git ...`. Resolving the branch from the project root
    then reads a worktree commit as a commit on the project branch (main) and
    blocks it. Honor an explicit `git -C <dir>` or a leading `cd <dir> &&`, so
    the guard reads the branch where the commit lands. A path this cannot
    resolve (for example an unexpanded variable) returns None and the caller
    falls back, so the guard fails open, never on a false positive.
    """
    m = re.search(r"\bgit\s+-C\s+(['\"]?)([^'\"\s]+)\1", command)
    if m:
        return m.group(2)
    m = re.match(r"\s*cd\s+(['\"]?)([^'\"&]+?)\1\s*&&", command)
    if m:
        return m.group(2).strip()
    return None


def current_branch():
    """Branch HEAD points at, or None if detached / not a repo / git error.

    Fail-open: any problem resolving the branch returns None and the branch guard
    simply doesn't fire -- a guard bug must never block legitimate work.
    """
    cwd = _effective_cwd(cmd) or payload.get("cwd") or os.getcwd()

    def git(*a):
        return subprocess.run(["git", "-C", cwd, *a], capture_output=True, text=True, timeout=5)

    try:
        # Confirm we're actually in a work tree first, so "not a repo" fails open
        # (returns None) rather than being mistaken for a detached HEAD.
        inside = git("rev-parse", "--is-inside-work-tree")
        if inside.returncode != 0 or inside.stdout.strip() != "true":
            return None
        r = git("symbolic-ref", "--quiet", "--short", "HEAD")
    except Exception:
        return None
    if r.returncode != 0:
        return "DETACHED"  # symbolic-ref --quiet exits non-zero on a detached HEAD
    return r.stdout.strip() or None


# --- gh guards (unchanged) ------------------------------------------------
if "gh pr create" in cmd and "--draft" not in cmd:
    block("PRs must be opened with --draft")

if re.search(r"gh pr (ready|merge)\b", cmd):
    block("the agent may not mark a PR ready or merge it")

# --- branch guard: commits/amends/pushes must be on the intended branch ---
mutates = re.search(r"\bgit\b.*\bcommit\b", cmd) or re.search(r"\bgit push\b", cmd)
if mutates:
    br = current_branch()
    expected = (os.environ.get("ATEN_AGENT_EXPECTED_BRANCH") or "").strip()
    if br == "DETACHED":
        block(
            "HEAD is detached -- check out your working branch before committing "
            "(the checkout may have been moved between turns)"
        )
    elif br in ("main", "master"):
        block(f"refusing to commit/push on {br}; check out your working branch first")
    elif expected and br and br != expected:
        block(
            f"current branch is {br!r} but you were told to work on {expected!r}. "
            "The working tree was likely moved between turns -- `git checkout "
            f"{expected}` (and re-apply your change there) before committing. "
            "Do NOT commit/amend onto a PR branch you did not create this turn."
        )

# --- push guards ----------------------------------------------------------
# A bare `git push` (no target args) is already covered by the branch guard
# above (it blocks pushing from main/master/detached/the wrong branch). These
# checks cover a push that NAMES a target -- and must not hinge on the remote
# being literally `origin`, since the remote can be `upstream`, a fork, or a URL.
if re.search(r"\bgit push\b", cmd):
    if re.search(r"(--force\b|--force-with-lease\b|-f\b|\s\+[\w./-]+:)", cmd):
        block("force-push is not allowed")
    names_target = re.search(r"\bgit push\b[^|&;]*\s\S", cmd)
    if names_target:
        # never push to a protected branch, whatever the remote or refspec form.
        # mainline is a real protected branch in this repo (origin/mainline), and
        # must be caught even when an allowed namespace also appears in the cmd
        # (e.g. `git push origin lower/x:mainline`), so it is checked before the
        # namespace allowlist below.
        if re.search(r"(^|[\s:])(main|master|mainline)(\s|$)", cmd):
            block("refusing to push to a protected branch (main/master/mainline)")
        # any named destination must land in an allowed namespace
        if not re.search(r"(lower/|fix/|feat/aten-lowering-orchestrator)", cmd):
            block("push only to lower/*, fix/*, or the batch branch (any remote)")

sys.exit(0)
