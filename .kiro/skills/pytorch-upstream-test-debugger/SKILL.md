---
name: pytorch-upstream-test-debugger
description: Drive an 8-step debug loop (ingest → pin → reproduce → branch → test+commit → fix+commit → verify+commit → draft PR) for a category of failing or xfailing PyTorch upstream tests on Neuron. Starts from a pytest report (JSON / CSV / plain log) plus a user pin (test nodeid or error-message regex), expands the pin to every test sharing the same error signature, scaffolds an RCA workspace with a unit-test template and a pytest-rerun script, creates the feature branch before any source edit, iterates a fix inside `torch_neuronx/` and commits each gate the moment its verification passes (failing fence test → fix passes → registry cleanup; 1 to 3 commits — the fence-test commit is skipped when the pinned upstream test already serves as the fence), re-verifies against a baseline, and opens a draft PR with a verify-failing shell script uploaded as a secret gist linked from the PR body (not committed to the repo — reviewers re-run the failing nodeids themselves). Respects Neuron platform limitations (no float64, no complex dtypes) — signatures caused purely by those get xfailed with a platform-limitation reason rather than a source fix. Also supports a second entry mode — given an existing PR number, fetches its context via `gh` / `gh api`, classifies review comments (actionable / discussion / nit / out-of-scope), addresses them with new commits and in-thread replies, reruns verification, and updates the PR body. The PR body is the canonical context-carrier; PRs stay draft. xfailed tests are tracked failures and are in scope — they must be fixed, not tolerated. Fixes land in `torch_neuronx` source (not pytest patches). One bug per PR. Neuron hardware access is delegated to the `agent-isolation` skill — this skill emits the commands; that skill runs them.
---

# Fix a category of failing (or xfailing) upstream tests

This skill attacks **one failure category at a time** with a
reproduce → RCA → fix → verify → PR loop. It has two entry points:

- **Mode A — new fix from a pytest report.** Start from a pytest
  report and a pin (a test nodeid or error-message regex), expand
  the pin to everything failing for the same reason, iterate a fix,
  and open a draft PR. This is the main 8-step loop.
- **Mode B — resume an existing PR.** Start from a PR number. Fetch
  the PR's context via the GitHub API, classify review comments,
  address actionable ones, reply to discussion ones, rerun
  verification, push updates. Shorter loop. See "Mode B" below.

**`xfailed` tests are in scope.** An xfail is a tracked, known failure.
Fixing one is a win; ignoring one is technical debt. Everywhere this
document says "failing test" it means "failed OR xfailed".

## Modes of operation

This skill has two runtime modes:

- **Interactive (default):** a human is in the loop at ambiguous
  points. Any "stop and ask" condition (unclear scope, unexpected
  error class, git conflicts, merge decisions) pauses and waits for
  a reply.
- **Headless / fleet:** no human available. On the same "stop and
  ask" conditions, the skill writes a **blocked** completion
  manifest (see below) and exits non-zero. A fleet orchestrator
  polls the manifest to learn the session ended and why.

**Both modes always write a completion manifest** at
`agent_artifacts/<slug>/completion.json`. Fleet runners consume this
file — it's the skill's only machine-readable output.

**Write the manifest twice: once at session start, once at terminal
status.** The first write goes out before any heavy work begins (right
after Step 1 picks the slug) with `status: "in_progress"`. The second
write replaces it on the way out the door — `done`, `blocked`,
`no-repro`, or `regressed`. Without the early write, a cancelled
subagent (the harness sends SIGTERM in batch runs) leaves no manifest
behind, and the orchestrator can't distinguish "agent was killed mid-
flight" from "agent decided not to write". Both writes go to the same
path; the second overwrites.

Schema:

```json
{
  "status":          "in_progress | done | blocked | no-repro | regressed",
  "mode":            "A | B",
  "category":        "<slug>",
  "branch":          "fix/<slug>" | null,
  "pr":              { "url": "...", "number": N, "draft": true } | null,
  "gist":            { "url": "...", "id": "..." } | null,
  "verification": {
    "fixed":                    N,
    "still_bad_same_signature": 0,
    "still_bad_different_sig":  N,
    "regressed":                0,
    "baseline_report":          "agent_artifacts/<slug>/rerun-baseline/report.json",
    "verify_report":            "agent_artifacts/<slug>/rerun-verify/verify.md"
  },
  "blocked_reason":  null | "<human-readable reason>",
  "timestamp":       "<ISO-8601 UTC>"
}
```

Status semantics for fleet orchestrators:

| `status` | Meaning | Fleet action |
|---|---|---|
| `in_progress` | Session running. Written at session start to claim the slug; replaced when the loop exits | Wait. If `timestamp` is older than the loop's expected duration AND the subagent is no longer alive, treat as crashed and rerun. |
| `done` | PR is open, verify clean, no regressions, no same-signature stragglers | Record PR URL, close the job |
| `blocked` | Human input required. `blocked_reason` names the gate | Route to on-call / triage queue |
| `no-repro` | Step 3 baseline didn't reproduce the input report's signatures. Usually a version/overlay drift | Re-pin to the current report and retry |
| `regressed` | Fix attempt caused regressions outside the category | Discard branch, rerun with narrower pin |

Mode B (resume PR) writes the manifest at the end of Step B6 in the
**same** `agent_artifacts/<slug>/` directory Mode A used — the slug
is recovered from the PR's branch name (`fix/<slug>`). One
directory for the life of a bug, across modes.

## Artifacts and file I/O

**All agent-produced files live under one directory:**
`<torch-neuronx-checkout>/agent_artifacts/<slug>/`. This is the only
place the skill writes. Nothing scatters to `/tmp/`, home, or
random subdirectories — everything one `ls` away for a human (or
a fleet log collector) who wants to inspect a session.

One slug per bug. Mode A picks the slug as the first action in
Step 1. Mode B **derives** the same slug from the branch name
(`fix/<slug>` → `<slug>`) during Step B1, so a PR opened by a
Mode-A session and later resumed by a Mode-B session shares one
artifact directory — no split between `pr-<N>/` and `<slug>/`.

### Layout (one directory for the life of a bug)

```
<torch-neuronx-checkout>/agent_artifacts/<slug>/
├── tests.txt                     (Step 1)   pinned nodeids
├── tests.txt.signatures.txt      (Step 1)   signature grouping
├── README.md                     (Step 2)   per-category orientation
├── notes.md                      (Step 2)   RCA log — fill as you investigate
├── unit_test_template.py         (Step 2)   seed for the project-tree unit test
├── rerun-baseline/               (Step 3)
│   ├── pytest.log
│   ├── report.json
│   └── summary.txt
├── verify_failing.sh             (Step 4)   uploaded to gist, NOT committed
├── repro.py                      (Step 4)   optional single-op Python repro
├── rerun-verify/                 (Step 8, and Mode B Step B3 re-verify)
│   ├── pytest.log
│   ├── report.json
│   ├── summary.txt
│   └── verify.md
├── pr_body.md                    (Step 9, updated by Mode B Step B5)
├── pr-metadata.json              (Mode B Step B1)   gh pr view --json
├── pr-line-comments.json         (Mode B Step B1)   gh api /pulls/N/comments
└── completion.json               (Step 10 / Step B6 / any blocked exit)
```

Re-verify runs during Mode B **overwrite** `rerun-verify/`; the
Step 3 baseline stays put as the authoritative "before" report.
The `pr_body.md` evolves in place across Mode B rounds — `gh pr
edit --body-file agent_artifacts/<slug>/pr_body.md` each time.

### Deriving the slug in Mode B

Mode B starts from a PR number, not a category. Recover the slug:

```bash
PR=<number>
gh pr checkout "$PR"
BRANCH=$(git branch --show-current)
SLUG="${BRANCH#fix/}"          # strips "fix/" prefix if present
mkdir -p "agent_artifacts/$SLUG"
```

If the PR was not opened by this skill, the branch won't start
with `fix/` — `SLUG` falls back to the full branch name, which is
still a valid artifact key. If two PRs share a branch name somehow
(rare), the user picks a disambiguating suffix explicitly.

### Repo-hygiene precondition

`agent_artifacts/` is agent-local scratch and **must not** be
committed. Before the skill writes anything, ensure the path is
git-ignored:

```bash
# From the torch-neuronx checkout root:
grep -q '^agent_artifacts/$' .gitignore || echo 'agent_artifacts/' >> .gitignore
```

If `.gitignore` already covers it, this is a no-op. If the repo
doesn't track `.gitignore` at all, stage the change (one-line
chore commit, separate from the fix).

### Files the skill does NOT write

- Anywhere under `/tmp/` (the old convention — now deprecated inside
  this skill).
- The torch-neuronx repo tree outside `agent_artifacts/`, except for
  the up-to-three intentional commits: `tests/python_ops/test_<op>.py`
  (only when a new fence test is authored — skipped when the pinned
  upstream test is reused as the fence, Step 6 decision),
  `torch_neuronx/...` source edits, and xfail-registry entries under
  `tests/pytorch_tests_native/skip_registry/` (when there are any to
  clear).
- Any path under `$HOME/` or the workspace root outside the
  checkout.

If a future extension of this skill needs a write outside
`agent_artifacts/<slug>/`, declare it here first.

## Companion skill: `torch-neuronx-docs`

This skill is a **workflow driver**. It tells you *what* to do step by
step, but it does not duplicate the structural knowledge of the
`torch_neuronx` codebase — that lives in the `torch-neuronx-docs`
skill under `.kiro/skills/torch-neuronx-docs/`.

Load both skills together. When the debug loop needs you to understand
a subsystem (which file owns ATen dispatch? where is the NEFF cache?
how do I rebuild the `.so`?), this skill cites the specific overview
doc to read. Every such citation is a path under
`docs/AI-*.md` — look it up there instead of
grepping from scratch.

Useful entry points:

| Debug-loop question | Go read |
|---|---|
| "what is this repo, end-to-end?" | `docs/AI-package-overview.md` |
| "which public API does the user code hit?" | `docs/AI-runtime-api.md` |
| "how does `torch.compile(backend='neuron')` work?" | `docs/AI-dynamo-backend.md` |
| "how is this ATen op registered in eager mode? what's the cast policy?" | `docs/AI-python-ops.md` |
| "where is `NeuronDispatcher`, the Python↔C++ boundary, or the `.so`?" | `docs/AI-csrc.md` |
| "how do collectives / `ProcessGroupNeuron` / DTensor work?" | `docs/AI-distributed.md` |
| "how does the NEFF cache work? where do artifacts go?" | `docs/AI-compilation-kernels.md` |
| "how do I run tests / what markers exist / what do the drivers do?" | `docs/AI-testing.md` |
| "how do I rebuild after a C++ edit?" | `docs/AI-build-infra.md` |

## Inputs

### Mode A — new fix

1. A pytest report from the failing run, in any of these formats:
   - **JSON** (from `pytest --json-report --json-report-file=report.json`)
   - **CSV** (with at minimum `nodeid` and `outcome`/`status`/`result` columns)
   - **plain log** (pytest console output saved to a file — the "short test
     summary info" section is parsed)
   - HTML is **not** supported directly; export JSON instead.
2. A **pin** — the specific bug we're fixing:
   - one or more exact test nodeids, **or**
   - a regex that matches the error message / traceback you care about,
   - or both.
3. A local checkout of the `torch-neuronx` repo (you'll edit source here).
4. Access to a Neuron-capable box. This skill emits the commands; the
   companion **`agent-isolation`** skill at
   `.kiro/skills/agent-isolation/SKILL.md` provides `agent-run`,
   `provision-worktree`, and `with-hw-lock` — read it before running
   anything on hardware, especially in a parallel batch where multiple
   debug agents share the host.

### Mode B — resume an existing PR

1. A PR number (and repo context via `gh`'s current-dir resolution, or
   `--repo OWNER/NAME`).
2. `gh` authenticated against the remote (`gh auth status` should show
   a logged-in account with repo access).
3. The same local checkout + hardware access as Mode A. `gh pr checkout`
   will switch to the PR's branch.

The PR body is the authoritative context-carrier in Mode B. This skill
expects Mode A to have populated it thoroughly; Mode B reads from it.

## Outputs

**Mode A** (new fix):
- An RCA workspace under `agent_artifacts/<slug>/` with
  investigation notes, a unit-test template, and a repro script stub.
- A failing test that fences the bug — either a new/updated unit test
  committed to the project's test tree (e.g.
  `tests/python_ops/test_<op>.py`), or the pinned upstream test when it
  already serves that role (Step 6 decision; no new test authored).
- Source edits under `torch_neuronx/` that clear the category.
- A verification report diffing before-and-after outcomes.
- A **draft PR** whose body links to a pytest-rerun shell script as
  a secret gist attachment (the script reruns the exact failing
  upstream nodeids this PR addresses; it is **not** committed to
  the repo — see Step 4 and the PR-body structure under Step 9).

**Mode B** (resume PR):
- New commits on the PR's branch addressing review comments.
- In-thread replies to every addressed review thread.
- Resolved review threads (only those actually addressed).
- Updated PR body if verification numbers changed.
- Round-summary comment at the issue level.

## Environment

`torch-neuronx` uses a **`uv`-managed venv** at `.venv/` at the repo
root. Every command this skill emits that runs `pytest`, `python3`, or
`./tools/build` assumes the environment resolves
`import torch_neuronx`. Two ways to make that true:

- **Activate:** `source .venv/bin/activate` once, then run commands
  directly.
- **`uv run` prefix:** `uv run python -m pytest …` without activating.

The project's own `./tools/run-test` / `./tools/run-test-parallel`
drivers auto-prefix `uv run` when `uv` is on `$PATH`, so those work
without activation. Plain `pytest` / `python` calls (including the
ones this skill's scripts emit) require activation or a manual
`uv run`.

Use **`uv pip`**, not plain `pip`, for install / list / show inside
this venv. Plain `pip` may not see uv-installed packages and can
install into the wrong environment.

### Health check before the first hardware run

```bash
# 1. Python stack resolves
python3 -c 'import torch; import torch_neuronx; print(torch.__version__, torch_neuronx.__version__)'

# 2. Neuron device is visible
python3 -c 'import torch; import torch_neuronx; assert torch.neuron.is_available(), "neuron not available"'

# 3. Dev deps for this skill's scripts
python3 -c 'import pytest_jsonreport'                      # should not raise
```

If any of these fail, fix the environment before proceeding — every
subsequent step will fail with confusing errors otherwise.

If you're running through `agent-isolation`'s `agent-run` (as part of a
parallel batch), run these checks **inside the wrapped shell** —
that's the shell that will actually execute `pytest`. In a Tier-2
worktree, also confirm `import torch_neuronx` resolves to the
worktree's source:

```bash
python3 -c 'import torch_neuronx, pathlib; print(pathlib.Path(torch_neuronx.__file__).resolve())'
# Expected: starts with /tmp/worktrees/<slug>/torch_neuronx/...
# Wrong:    starts with the canonical checkout path — the editable-install
#           MAPPING wasn't rewritten and `cd` into the worktree wasn't done.
```

### Expected dev dependencies

Project declares these as dev deps (present in a properly set-up venv):
`pytest-html`, `pytest-xdist`, `pytest-forked`, `pytest-cov`,
`pytest-json-report`, `pytest-shard`, `pytest-timeout`. If any are
missing:

```bash
uv pip install pytest-xdist pytest-json-report pytest-timeout
```

### Upstream-test overlay

Upstream PyTorch tests are cloned/overlaid into the repo from the
installed torch nightly. If the overlay is stale versus the torch
version in the venv, errors won't reproduce faithfully. The repo
ships a clone script at
`tests/pytorch_tests_native/clone_device_agnostic_tests.py` (or a
`run_upstream_tests.sh` driver); run it whenever torch in the venv
moves. See `docs/AI-testing.md` ("Upstream
PyTorch tests") for detail.

### Useful environment variables during debugging

Set these on the command line for one pytest invocation, not
globally — they meaningfully slow down or change execution.

| Variable | Effect |
|---|---|
| `NEURON_LAUNCH_BLOCKING=1` | Force synchronous execution. Tracebacks point at the real op. Essential for debugging ordering issues. |
| `TORCH_NEURONX_DEBUG_DIR=/path/to/artifacts` | Write compilation artifacts (FX, StableHLO, NEFF) here. |
| `TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS=1` | Keep the above after the run. |
| `TORCH_NEURONX_DISABLE_NEFF_CACHE=1` | Force recompilation every call — useful when suspecting a cache-staleness bug. |
| `TORCH_NEURONX_LOG_LEVEL=3` | C++ debug logs. Verbose, read carefully. |
| `NEURON_CC_FLAGS="…"` | Extra flags to `neuronx-cc`. |

See `docs/AI-runtime-api.md` and
`docs/AI-compilation-kernels.md` for the full list and cache
artifact layout.

### xfail / skip registry

Upstream tests use a **single source of truth** for skip/xfail at
`tests/pytorch_tests_native/neuron_test_registry.py`, with per-root-cause
entries under `tests/pytorch_tests_native/skip_registry/`. Step 8's
xfail cleanup edits this registry, not inline decorators. See
`docs/AI-testing.md` ("Upstream PyTorch tests").

## Running the tests

The repo ships three relevant test drivers plus plain `pytest`. Each
has a job — pick the right one for the step.

| Situation | Preferred command | Why |
|---|---|---|
| Inner loop: one unit test, many iterations | `pytest -xvs tests/…::Test::test_x` | Precise control, fastest iteration, `-xvs` stops at first fail with full stdout |
| Sanity check: whole file for the op you're fixing | `./tools/run-test tests/python_ops/test_<op>.py` | Respects project ignores, uses `uv run` when `uv` is on `$PATH` |
| Baseline / verify: the full category list | `$SK/scripts/rerun_category.sh …` (wraps `pytest`) | Structured JSON report needed by `verify_fixed.sh` |
| Full parallel run | `./tools/run-test-parallel` | xdist + forked + per-worker device pinning, core-count aware |
| C++ tests | `./tools/run-cpp-test` or `./tools/run-cpp-test-cmake` | Bazel / CMake gtest — required when you touched `torch_neuronx/csrc/` |

Never point `rerun_category.sh` at huggingface, vertical, or the
whole suite — it's category-sized (tens to hundreds of tests), not
suite-sized.

### xdist parallelism (`-n`)

`rerun_category.sh` accepts optional named flags after the three
positional arguments (`repo`, `tests.txt`, `out-dir`):

```bash
$SK/scripts/rerun_category.sh <repo> <tests.txt> <out-dir> [-n <num>] [--timeout <sec>]
```

Tune per test type (from `references/loop-playbook.md`):

| Test type | `-n` |
|---|---|
| Distributed (NCCL / FSDP / comm collectives) | 8 |
| Autograd (`gradcheck`, `gradgradcheck`, `test_ops_gradients.py`) | 16 |
| Other plain unit tests | up to 64 |

For under-30-test categories, **cap `-n` at or below the test count**
(worker init overhead dominates past that). When debugging a single
stuck/racy test, drop `-n` entirely — NRT state effects can differ
between single-process and parallel runs, so single-process is the
canonical repro.

### Reading pytest output

The scripts pass specific flags — what they mean:

| Flag | Effect |
|---|---|
| `--runxfail` | Ignore `@pytest.mark.xfail` so real tracebacks show. xfails-that-still-fail surface their error; xfails-that-now-pass report as `passed`. |
| `--tb=short` | One-line traceback per failure. Full tracebacks live in the longrepr field of the JSON report. |
| `-rs -rx` | Summary of skips and xfails at the end — the lines `select_category_tests.py` parses out of plain-log input. |
| `--json-report --json-report-file=…` | Structured per-test outcomes. `verify_fixed.sh` diffs this against the baseline. |
| `--timeout=600` | Per-test wall-clock cap. Gradcheck on Neuron can hit this — see `timeout` category in the playbook. |

Interpreting outcomes with `--runxfail` active:
- `passed` — fixed (was xfailed, now passes) OR was always passing.
- `failed` — real failure; crash message in `call.crash.message` of the
  JSON report.
- `skipped` — not run (environment, marker, or collection skip). If
  the JSON entry has `wasxfail` set, this was an xfail — treat as
  xfailed, not skipped. `select_category_tests.py` handles this.
- `xfailed` — visible only when `--runxfail` is NOT passed. This skill
  always passes it, so you'll rarely see this string.
- `error` — setup/collection error, not test logic. Include or exclude
  from selection with `--include-error-outcome`.

## Commit and push policy

One canonical rule set, used by both modes. The skill **never**
rewrites remote history.

### Commits per PR

A new PR from Mode A lands **three commits** on the feature branch,
**committed as each gate is verified** — not staged up at the end.
The cadence is:

1. Create the feature branch from fresh `origin/main` as soon as the
   baseline rerun confirms the signatures reproduce (after Step 3,
   before any `torch_neuronx/` edits).
2. Write the unit test. As soon as it **fails** on unfixed code with
   the expected signature, commit it. **Skip this commit if you're
   reusing the pinned upstream test as the fence** (Step 6 decision) —
   no new file, no commit.
3. Write the fix. As soon as the unit test **passes** and the whole
   op test file stays green, commit it.
4. Run verify. As soon as `verify.md` reports tests in the `fixed`
   bucket with no regressions, commit the registry cleanup for those
   tests.

Fewer is fine when there's nothing to ship in a given commit — no new
test because the upstream test is the fence (→ skip COMMIT 1), or no
registry entries to remove (→ skip COMMIT 3). Either drops you to two
commits; both drop you to one (the fix). More is fine when the fix
genuinely needs to be split. Each commit must be verified before it
lands. See Step 9 under "Three-commit structure (recap)" for the
staging detail.

Mode B **adds commits on top** of the existing three — one per review
round or per actionable comment. No squashing of either the original
three or the review-round additions.

### When to amend vs. new commit

| Situation | Action |
|---|---|
| Local commit, **not yet pushed anywhere** | `git commit --amend` is fine (typos, lint, forgotten file). |
| Local commit, **already pushed** to the PR branch | **New commit on top.** Never amend, never squash, never force-push. |
| Typo or lint fix caught immediately after committing, before first push | `git commit --amend`, then push for the first time. |
| Addressing review feedback (any size) | **New commit** with a descriptive message. |
| Undoing the most recent unpushed commit to restructure | `git reset --soft HEAD~1` (keeps changes staged). OK only while unpushed. |

### Forbidden operations

The skill **never** runs any of these:

- `git push --force` / `git push -f` / `git push --force-with-lease`
- `git rebase origin/main` followed by a push (rewrites pushed history)
- `git reset --hard` on a pushed branch
- `git commit --amend` on a pushed commit (then pushing)
- `git rebase -i` / interactive rebase
- `git filter-branch`

If the situation feels like it needs one of these, **stop and ask the
user.** The repo's steering rule is: *once a commit is in the remote,
it is immutable — fix forward with new commits.*

### Reviewer-pushed commits on your branch

GitHub lets reviewers push to PR branches. If this happens your next
push will be rejected:

```bash
git pull --ff-only          # fast-forward the reviewer's commits into local
# then continue with your normal commit + push
```

If `--ff-only` fails (reviewer force-pushed, or your work diverged),
stop and ask the user. Do not force around it.

### Do we squash?

**No.** Not during review, not before merge. Reasons:
- Squashing a pushed branch requires force-push, which is forbidden.
- GitHub's "Squash and merge" button does it at merge time — the
  reviewer decides.
- Review-round commits give reviewers a clean "what changed this
  round" diff.

If a specific repo requires a single-commit PR for merge, the human
flips the "Squash and merge" button or asks the agent to produce a
fresh PR from a clean branch (Mode A from the start).

### Conflicts with the base branch

If CI says the PR branch is behind `origin/main` and conflicts:

```bash
git fetch origin
git merge origin/main        # merge commit — preserves history
# resolve conflicts, commit, push
```

**Not** `git rebase origin/main && git push --force`. The merge
commit is uglier in history but it's additive, not destructive.

### Pre-commit hooks

Lint / format / whitespace checks are wired as pre-commit hooks in this repo
(`.pre-commit-config.yaml`: ruff, ruff-format, clang-format, trailing-whitespace,
end-of-file-fixer, check-yaml, check-added-large-files, pretty-format-json).
CI runs them and will fail the PR if any hook is dirty, so **every commit this
skill produces must be preceded by a passing pre-commit run** — not just a
final sweep before push.

This is a hard rule. If the environment doesn't have `pre-commit` installed
or the git hook wired, fix that first — do not skip the check and hope CI
catches it cheaply later.

#### One-time setup (do this before the first commit of the branch)

```bash
# Confirm pre-commit is on PATH (it's a dev dep; install if missing).
which pre-commit || uv pip install pre-commit

# Install the git hook so `git commit` triggers the checks automatically
# for every staged file. Idempotent; safe to re-run.
pre-commit install
```

If `which pre-commit` fails after `uv pip install pre-commit` too, check that
the venv is active (`source .venv/bin/activate`) or prefix with `uv run`.

#### Per-commit gate (scoped, fast)

Before every `git commit` the skill runs:

```bash
# 0. Pre-flight ruff on every Python file in <paths>. This is the
#    skill's single biggest commit-failure mode in batch runs:
#    ruff-check + ruff-format auto-rewrite first, pre-commit then
#    sees the rewrite, exits non-zero, and the agent commits the
#    rewritten files. But if `git add` was run BEFORE the rewrite,
#    the rewritten content isn't staged — the commit ships the
#    pre-rewrite version, pre-commit re-fails on the next push, and
#    the agent doesn't notice because there was no error in this
#    shell. Run ruff first, deterministically, before staging.
ruff check --fix <python paths>
ruff format <python paths>

# 1. Run the full hook set against exactly the files you are about
#    to commit. --files accepts the same set you'll pass to `git add`.
pre-commit run --files <path1> <path2> ...

# Some hooks (trailing-whitespace, end-of-file-fixer, pretty-format-json)
# also auto-rewrite. If pre-commit exits non-zero with file rewrites,
# re-run until exit 0 — then the working tree matches what hooks
# accept.
```

Scoping with `--files <paths>` keeps per-commit latency low (seconds on a
single test file) while still catching the exact hook set CI enforces.
For the registry-cleanup commit, `<paths>` may be several files — that's
still fine, it's just a smaller set than `--all-files`.

#### Post-commit verification (mandatory in batch runs)

After every `git commit`, **verify the commit actually landed** before
proceeding to the next step:

```bash
EXPECTED_MSG="<the commit message you just used>"
HEAD_MSG=$(git log -1 --pretty=%s)
if [[ "$HEAD_MSG" != "$EXPECTED_MSG" ]]; then
    echo "ERROR: commit did not land. HEAD message is '$HEAD_MSG', expected '$EXPECTED_MSG'."
    echo "       Likely cause: pre-commit hook rolled back its rewrite and the commit aborted silently."
    echo "       Inspect with: git status; pre-commit run --files <paths>"
    exit 1
fi
```

This catches the silent-rollback case the upstream-debugger-orchestrator
retro flagged: pre-commit can rewrite files, exit zero on the rewrite
itself, but the original `git commit` invocation that triggered the
hook still ends up not creating a commit because the pre-staged content
no longer matches the working-tree content. Without this check, the
agent moves on, force-pushes an incomplete branch, and the missing
commit is only noticed when CI red-Xes.

If `HEAD_MSG` doesn't match expected, **never amend or force-push to
recover** — re-stage and re-commit. The first commit literally didn't
exist; there's nothing to amend.

#### Final sweep before first push

Once the branch's commits exist locally, run one more full pass before the
first `git push`:

```bash
pre-commit run --all-files
```

If this rewrites anything, amend the offending file into whichever commit
introduced it (still pre-first-push, so amending is allowed) and re-run
until clean.

#### Fallback when pre-commit genuinely can't run

If the environment has a hard constraint preventing `pre-commit` (restricted
network, no venv write access, CI container without the dev deps), run the
underlying tools directly against the staged files — they are the hooks,
just invoked manually:

```bash
ruff check --fix <paths>
ruff format <paths>
# clang-format only if any C++ / header staged:
clang-format -i <paths>
```

This is a last resort, not a default. Flag the skipped pre-commit in the
PR body so the reviewer knows to re-run the hooks themselves before
merging.

## The 8-step loop (Mode A)

```
   pytest report + user pin (nodeid and/or error regex)
        │
        ▼
   1. ingest + pin          ── select_category_tests.py
        │
        ▼
   2. scaffold workspace    ── rca_workspace.py
        │                      (seeds unit tests, writes repro.py stub)
        ▼
   3. baseline rerun        ── rerun_category.sh against the pin list
        │                      (this is what "fix verified" is measured against)
        ▼
   4. minimal repro         ── build verify_failing.sh from tests.txt
        │                      ─ run against unfixed code to confirm repro
        │                      ─ upload as secret gist (PR attachment)
        ▼
   5. create feature branch ── fix/<slug> off fresh origin/main
        │                      (before any edit to torch_neuronx/ or tests/)
        ▼
   6. write unit test       ── must FAIL on unfixed code
        │                      → COMMIT 1 (test) as soon as it fails
        │                      ─ SKIP if reusing the upstream test as fence
        ▼
   7. fix + unit-loop       ── edit torch_neuronx/ → unit test (seconds)
        │                      → COMMIT 2 (fix) as soon as unit test passes
        ▼
   8. verify + registry     ── verify_fixed.sh: full category vs baseline
        │                      → COMMIT 3 (registry) for the `fixed` bucket
        ▼
   9. draft PR              ── push + gh pr create --draft, gist URL in body
        │
        ▼
  10. completion manifest   ── write agent_artifacts/<slug>/completion.json
                               (fleet handoff / interactive summary)
```

(Ten explicit steps — the section heading still says "8-step loop"
for historical continuity with the skill's original name. Step 5
"create the feature branch" is the gate between "investigating"
(agent-local scratch work) and "committing" (real branch, real history).
Steps 7 and 8 each end with an immediate commit the moment their
verification gate passes; Step 6 commits the new fence test the moment
it fails — or commits nothing (skips COMMIT 1) when the pinned upstream
test is reused as the fence (the SKIP branch in the diagram above).
Step 10 writes the machine-readable completion signal — see **Modes of
operation**.)

## Workflow

### Step 1 — ingest the report and expand the pin

**First, pick the category slug.** It's the user-facing handle for
everything that follows — the artifact directory name, the branch
name (`fix/<slug>`), and a component of the PR title. Short,
descriptive, lowercase-with-hyphens or underscores:
`dtype_mismatch_double_vs_float`, `jacobian_mismatch`, `timeout`.
Pick it once and reuse it. Set it as an environment variable so the
rest of the loop's commands stay copy-pasteable:

```bash
SLUG=dtype_mismatch_double_vs_float   # example
mkdir -p "agent_artifacts/$SLUG"
```

**Write the in-progress completion manifest now**, before any heavy
work. This claims the slug for fleet observers and ensures a manifest
exists even if the agent is later cancelled:

```bash
python3 - <<PY
import json, datetime, pathlib, os
slug = os.environ["SLUG"]
manifest = {
    "status":         "in_progress",
    "mode":           "A",
    "category":       slug,
    "branch":         None,
    "pr":             None,
    "gist":           None,
    "verification":   {"fixed": 0, "still_bad_same_signature": 0,
                       "still_bad_different_sig": 0, "regressed": 0,
                       "baseline_report": None, "verify_report": None},
    "blocked_reason": None,
    "timestamp":      datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
pathlib.Path(f"agent_artifacts/{slug}/completion.json").write_text(
    json.dumps(manifest, indent=2))
PY
```

Step 10 will overwrite this with the terminal status. Mode B does the
same write at Step B1 right after slug recovery.

Then run `scripts/select_category_tests.py` to:

1. Parse the report (JSON / CSV / log auto-detected).
2. Apply the pin (`--pin-nodeid` and/or `--pin-error`).
3. Compute a normalized error signature for every failing test.
4. Expand the pin to **every failing test sharing any pinned signature**.

```bash
SK=/path/to/this/skill   # .../pytorch-upstream-test-debugger

# Pin a specific failing test:
$SK/scripts/select_category_tests.py report.json \
    --pin-nodeid tests/pytorch_tests_native/test/test_ops_gradients.py::TestBwdGradientsNEURON::test_fn_grad_gather_neuron_float64 \
    --out agent_artifacts/<slug>/tests.txt

# Or pin by error message:
$SK/scripts/select_category_tests.py report.json \
    --pin-error "Double did not match Float" \
    --out agent_artifacts/<slug>/tests.txt

# Or pin from a file that mixes nodeids and regexes (lines prefixed
# `nodeid:` / `error:`, or bare lines = nodeids):
$SK/scripts/select_category_tests.py pytest.log \
    --pin-from-file pins.txt \
    --out agent_artifacts/<slug>/tests.txt
```

Outputs:
- `agent_artifacts/<slug>/tests.txt` — one pytest nodeid per line. The full category.
- `agent_artifacts/<slug>/tests.txt.signatures.txt` — human-readable breakdown showing
  how many tests fell into each signature group and a sample error
  message per group. Read this — it's your first sanity check that the
  pin captured the right bug.

If the pin only matches one test but `tests.txt` contains ten, that's
the skill working as intended: nine of those tests share the signature
and likely the same root cause. Fixing one should fix all.

(The slug was already picked at the top of this step — reuse the
same value for the branch name, PR title, and all subsequent
`agent_artifacts/<slug>/...` paths.)

### Step 2 — scaffold the RCA workspace

```bash
$SK/scripts/rca_workspace.py agent_artifacts/<slug>/tests.txt agent_artifacts/<slug> \
    --category dtype_mismatch_double_vs_float \
    --signatures agent_artifacts/<slug>/tests.txt.signatures.txt
```

Creates `agent_artifacts/<slug>/` with:
- `README.md` — per-category orientation.
- `tests.txt`, `signatures.txt` — copies of the selection artefacts.
- `unit_test_template.py` — pytest seeds, one per op name parsed from
  the failing nodeids. These are throwaway; the final unit test goes
  into the project tree in Step 6.
- `verify_failing.sh` — pytest rerun script stub (the PR's gist
  attachment). Generated from `tests.txt` by Step 4; contains the
  exact failing nodeids plus the skill's standard pytest flags
  (`--runxfail --tb=short -rs -rx -n 16 --timeout 600`). **Uploaded
  as a secret gist**, URL in PR body, **not** committed to the repo.
  You'll finish it in Step 4.
- `repro.py` *(optional)* — an auxiliary single-op Python repro for
  categories where pytest output is too noisy to see the root cause.
  If produced, add it as a second file to the same gist. Not
  required for every category.
- `notes.md` — RCA log. Fill as you investigate.

### Step 3 — baseline rerun

Before fixing anything, rerun the pinned tests with pytest + JSON
report. This is the **baseline** the verification step will diff
against.

**Prerequisite check:** the shell must have the `uv` venv active
(`source .venv/bin/activate`) and `python3 -c 'import torch_neuronx'`
must succeed. See the Environment section above. If hardware access
goes through a wrapper skill, the venv must be active inside the
wrapper's shell too — that's where `pytest` will actually run.

```bash
# -n and --timeout are optional named flags (see "Running the tests").
$SK/scripts/rerun_category.sh \
    /path/to/torch-neuronx-checkout \
    agent_artifacts/<slug>/tests.txt \
    agent_artifacts/<slug>/rerun-baseline \
    -n 16 \
    --timeout 600
```

The script runs `pytest` in the current shell. If your Neuron hardware
is behind an access wrapper, invoke this via the wrapper (or run its
internal `pytest ...` command inside the wrapper's shell). The wrapper
mechanics live in a separate skill — this skill doesn't care which one.

Outputs:
- `pytest.log` — full stdout+stderr
- `report.json` — structured per-test outcomes (used by verify)
- `summary.txt` — one line per test

Key flags used: `--runxfail` so xfails don't hide real failures
(xfailed-that-passes → `passed`, xfailed-that-fails → visible
traceback).

**Before you change any source:** confirm the baseline reproduces the
signatures you saw in the original report. If it doesn't — either your
checkout is on a different commit than the report was produced from,
or the environment differs. Stop and align before continuing.

### Step 4 — build a pytest rerun script (this becomes a gist linked from the PR)

The attachment is **not a single-op Python repro** — it's a shell
script that reruns **the exact failing upstream pytest nodeids** with
the same flags the skill uses, so a reviewer can directly verify that
the before/after outcomes match what the PR claims. This also
doubles as the evidence that justifies the xfail-registry cleanup
in Step 8: reviewers trust the removal because they can re-run the
script themselves and watch the tests pass.

Goal of the script:

- Take the `tests.txt` from Step 1 (the pinned category) and run
  `python -m pytest --runxfail --tb=short -rs -rx` against those
  nodeids.
- Print a one-line-per-test summary at the end (`PASSED` / `FAILED`
  / `XFAIL` / `ERROR`) so a human reviewer can scan the result.
- Self-contained — no reliance on the skill's scripts or agent-local
  paths. The script pastes the nodeid list inline.
- Works against any checkout: the user cloned torch-neuronx, checked
  out the PR branch, activated the venv, then runs the script.

Template — copy this into `agent_artifacts/<slug>/verify_failing.sh`:

```bash
#!/usr/bin/env bash
# verify_failing.sh — rerun the exact upstream tests this PR addresses.
#
# Usage (from any torch-neuronx checkout root, with the Neuron venv active):
#     bash verify_failing.sh
#
# Before this PR:  most tests FAIL or XFAIL with the signature below.
# After  this PR:  tests move to PASSED (or at minimum, a different signature).
#
# Category: <category-slug>
# Signature: <one-line error signature from Step 1>

set -euo pipefail

# --- Tests in scope (pasted from tests.txt) ---
TESTS=(
  "tests/pytorch_tests_native/test/test_decomp_shard0.py::TestDecompNEURON::test_comprehensive_index_copy_neuron_float16"
  "tests/pytorch_tests_native/test/test_decomp_shard1.py::TestDecompNEURON::test_comprehensive_index_copy_neuron_bool"
  # ... (one line per failing nodeid)
)

# --- Pytest flags (match skill baseline/verify conventions) ---
PYTEST_FLAGS=(
  --runxfail          # surface real failures instead of xfail-swallowing
  --tb=short
  -rs -rx
  -n 16               # xdist; tune per test type (see skill Environment)
  --timeout 600
)

# --- Run ---
python -m pytest "${PYTEST_FLAGS[@]}" "${TESTS[@]}"
```

Generate the `TESTS` array programmatically from the pinned
`tests.txt`:

```bash
{
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    echo 'TESTS=('
    sed 's/^/  "/; s/$/"/' agent_artifacts/<slug>/tests.txt
    echo ')'
    echo 'PYTEST_FLAGS=(--runxfail --tb=short -rs -rx -n 16 --timeout 600)'
    echo 'python -m pytest "${PYTEST_FLAGS[@]}" "${TESTS[@]}"'
} > agent_artifacts/<slug>/verify_failing.sh
chmod +x agent_artifacts/<slug>/verify_failing.sh
```

Run the script once on the **unfixed** code as a sanity check — the
failure count should match the Step 3 baseline and the signatures
should match the pin. Run it again on the fixed code at the end of
Step 8 — the failure count should match the `still_bad` bucket in
`verify.md` and the `fixed` bucket in `verify.md` should now show
`PASSED` lines.

This is the chain of evidence that justifies the xfail-registry
cleanup: script output shows the tests pass, verify.md's `fixed`
bucket cites the same nodeids, Step 8's commit removes exactly
those xfail entries. Anyone reviewing the PR can re-run the gist
themselves and reproduce the diff.

Once the script exits 0 on fixed code (or produces the expected
partial-fix output for a blocked signature), **upload as a secret
gist** — this is the PR attachment. Do **not** commit it to the repo.

```bash
gh gist create agent_artifacts/<slug>/verify_failing.sh \
    --filename verify_failing_<op>_<category>.sh \
    --desc "Verify script for <category> (torch-neuronx PR #<TBD>)"
# → https://gist.github.com/<user>/<gist_id>
```

Save the URL — it goes in the PR body under "Reproduction" (Step 9).
If you need to update the script later (e.g. after review feedback
trims the nodeid set), edit the gist in place with
`gh gist edit <gist_id>`; don't create new gists for the same PR.

#### Fallback: `gh` token without `gist` scope

`gh gist create` fails with "missing required scope: gist" if the
token doesn't have it. Many shared / agent / CI tokens don't —
adding the scope requires re-auth, which a batch run usually can't
do mid-flight. Detect the missing scope **before** running the loop:

```bash
gh auth status 2>&1 | grep -q "Token scopes:.*'gist'" || GIST_AVAILABLE=0
```

If `GIST_AVAILABLE=0`, **inline the verify script in the PR body's
Reproduction section as a fenced code block** instead of uploading.
The script's body is small (one nodeid array + the standard pytest
flags + one pytest invocation — usually <100 lines), so embedding it
inline is workable. Note explicitly in the body that the gist path
was unavailable in this environment, so a reviewer expecting a gist
URL knows where to look:

```markdown
## Reproduction

> **Note:** the agent's `gh` token lacks `gist` scope, so the verify
> script is embedded inline below instead of linked as a secret
> gist.

With the Neuron venv active from a torch-neuronx checkout, save the
script and run it:

\`\`\`bash
#!/usr/bin/env bash
# verify_failing.sh — rerun the upstream tests this PR addresses.
set -euo pipefail
TESTS=(
  "tests/pytorch_tests_native/test/test_..."
  ...
)
PYTEST_FLAGS=(--runxfail --tb=short -rs -rx -n 16 --timeout 600)
python -m pytest "${PYTEST_FLAGS[@]}" "${TESTS[@]}"
\`\`\`
```

This is the proven workaround from the upstream-debugger-orchestrator
batch run that hit this exact env. Don't commit the script to the
repo as a workaround — it's still agent-local scratch, and PR-body
inlining keeps it associated with the PR's lifetime (deleted when the
PR is merged or closed).

#### Optional: a Python-level repro alongside the pytest script

For some categories (e.g. a compiler crash, an NKI-kernel bug) the
pytest-level output is noisy and a 20-line Python repro isolates the
bug more clearly for a reviewer. In that case, add a second file to
the gist with `gh gist edit <gist_id> --add agent_artifacts/<slug>/repro.py`
so both live behind one URL. This is optional — the pytest rerun
script is the required attachment; a single-op Python script is
supporting context.

#### When to NOT fix — Neuron platform limitations

Neuron's backend does not support every dtype PyTorch defines. Before
treating a signature as a bug, check whether the underlying ask is
outside the platform's supported dtype set:

| Dtype | Neuron support | Typical test symptom |
|---|---|---|
| `float64` / `torch.double` | **Not supported** (silently cast to `float32` in some paths, errors in others) | OpInfo dtype tests parametrized with `float64` fail with accuracy mismatch, "supported dtypes … are incorrect", or surprise success depending on the op's cast policy |
| `complex64` / `complex128` | **Not supported** (PR #629 now rejects complex tensor allocation with a clean `TORCH_CHECK`) | OpInfo dtype tests parametrized with complex dtypes raise or segfault |

If the failing tests are purely parametrized over an unsupported
dtype and expect the op to work there, **the correct action is not
a fix** — it's an xfail-registry entry under the
`neuron_platform_limitation` category with a reason naming the dtype.
Do not extend Neuron's dtype coverage to chase the OpInfo allowlist;
do not silently cast to make the test pass. The limitation is real
and visible to users; the test must remain a tracked, categorized
skip rather than a source fix.

If the failing test is **NOT** about dtype (wrong shapes, bad index
handling, crashes on 0-d tensors, etc.), proceed with the normal fix
loop even if the traceback happens to involve `float64` or complex —
those are incidental to the real bug.

When in doubt, run the test with a supported dtype (`float32`,
`int32`, `bool`). If it passes, the bug is the dtype limitation
alone → xfail with platform-limitation reason. If it still fails, the
bug is real → continue to Step 5.

See `docs/AI-python-ops.md` for the cast
policy and `docs/environment_variables.md` (repo root) for the full
platform dtype matrix.

#### When to NOT fix — environment skew

If the failing signature is an `AttributeError`, `ImportError`, or
`ModuleNotFoundError` on a `torch.*` internal (e.g.
`torch.testing._comparison._unwrap_dtensor_for_comparison`,
`torch.distributed.tensor.DTensor`, a private helper in
`common_utils.py`), there is **no `torch_neuronx/` source bug** to
fix. The cause is overlay/venv version skew between the upstream
test overlay shipped under `tests/pytorch_tests_native/` and the
torch installed in `.venv/`.

Resolution belongs in Step 8's decision node ("If verify fails on a
missing torch internal — overlay/venv skew"), not here — but you
should recognize the signature *before* you start writing a fix.
The two valid fix paths are:

1. Refresh the overlay against the venv's torch:
   `python tests/pytorch_tests_native/clone_device_agnostic_tests.py`.
2. Bump the torch dep in `pyproject.toml` to a version compatible
   with the overlay, then `uv sync`.

What you must **not** do, even if it makes the symptom go away:

- Monkey-patch the venv's installed torch (`.venv/.../torch/...`).
  See "Never modify the venv's installed packages" in Critical
  rules.
- Add a backport shim to `conftest.py` or any test-patches helper.
  See "Fixes go in `torch_neuronx` source" in Critical rules.
- Reverse-engineer the upstream PyTorch fork's git history to find
  a "compatible commit" — the clone script already does the right
  thing.

If neither valid path resolves it, the issue is structural
(infrastructure, not a bug in the category you were asked to fix).
Write a `blocked` completion manifest and stop.

### Step 5 — create the feature branch (before any source edit)

The next steps commit to a real branch. Create it **now**, off a
freshly-fetched `origin/main`, so every subsequent commit already
belongs to a PR-ready history.

**Branch naming:** always `fix/<slug>`, always off `origin/main`. Slug
from the category + op: `fix/gather-float64-grad`,
`fix/timeout-linalg-svd`, `fix/index-copy-0d-tensor`. Short,
imperative, lowercase-with-hyphens. Must not collide with any
existing local or remote branch — pick a new slug if it does.

```bash
git fetch origin

BRANCH="fix/<slug>"

# Refuse to reuse a name — a stale branch from a previous attempt
# carries history you don't want. Pick a new slug if either check hits.
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    echo "ERROR: local branch $BRANCH already exists"; exit 1
fi
if git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null; then
    echo "ERROR: remote branch $BRANCH already exists"; exit 1
fi

git checkout -b "$BRANCH" origin/main
```

Do **not** `git rebase` an existing branch onto main — that rewrites
history and conflicts with the Commit and push policy. Start fresh.

From this point on, every code change is a commit candidate. The next
three steps each end with a commit the moment their verification gate
passes — you don't wait until the end of the loop.

### Step 6 — write a unit test that captures the bug, commit it

Locate the offending frame in `torch_neuronx/` — read it, don't guess.

**Before reading code in a directory you haven't touched yet:** look
for a guidance file in it (`AGENTS.md`, `SKILL.md`, `README.md`,
`CONTRIBUTING.md`) and read it first. Walk up the tree to the repo
root — guidance in a parent directory applies to its descendants.
This catches invariants and conventions you won't see from code alone.
(See Critical rules.)

To orient yourself in the area the traceback points at, consult the
companion `torch-neuronx-docs` skill first:

| Traceback lands in… | Read first |
|---|---|
| `torch_neuronx/csrc/...` | `docs/AI-csrc.md` — the dispatcher, pybind layer, allocators |
| `torch_neuronx/python_ops/...` | `docs/AI-python-ops.md` — op registration, cast policy, MLIR/NKI/XLA backends |
| `torch_neuronx/neuron_dynamo_backend/...` | `docs/AI-dynamo-backend.md` — FX passes, StableHLO pipeline, Executor |
| `torch_neuronx/distributed/...` or `ProcessGroupNeuron` | `docs/AI-distributed.md` — collectives + DTensor sharding |
| `torch_neuronx/kernels/...` (NEFF, compiler subprocess) | `docs/AI-compilation-kernels.md` |

Reading those takes 2–3 minutes and usually narrows the fix site to a
specific function or registration table — much cheaper than grepping
the source tree.

**First decide whether a new test is even needed.** The unit test
serves two jobs: a committed regression fence and a fast inner-loop
target. If the pinned upstream test *already* does both — it lives in
the repo's own test tree, fails specifically on this bug, and is small
and fast enough to loop on (seconds, not a 600 s gradcheck sweep) —
then a new unit test is pure duplication. **Skip Step 6's authoring and
COMMIT 1, and use that upstream nodeid as your Step 7 inner-loop
target.** Note in `notes.md` which upstream test you're reusing and why.

Write a new focused test only when the upstream one is unsuitable as a
fence: it's broad/parametrized over many cases, slow (gradcheck), or
lives only in the cloned/overlaid test source (not the repo's own tree,
so it isn't a durable in-repo regression fence). In that case:

**The real unit test belongs in the project's test tree**, not in the
RCA scratch dir. Put it in `tests/python_ops/test_<op>.py` (or
`test_<op>_backward.py`) and follow the conventions already there —
parametrized shapes, `torch.testing.assert_close` against CPU, any
existing helpers like `track_neuron_ops` / `assert_op_runs_on_neuron`.
Often adding one `pytest.param` to an existing `TEST_CONFIGS` list is
enough. (See `docs/AI-testing.md` for the full test-tree layout.)

Use `unit_test_template.py` in the RCA workspace only as a scratchpad /
seed — once the shape converges, move it into the project tree.

**Confirm the test FAILS on the unfixed code first.** A test that
doesn't fail without the fix isn't a regression fence. Log the exact
error message in `notes.md` — you'll reference it in the PR.

#### Commit gate — COMMIT 1 (unit test)

**Skip this commit entirely if you're reusing the upstream test** (the
decision above) — there's no new file to commit, and the PR becomes a
two-commit structure (fix · registry cleanup) instead of three. Go
straight to Step 7.

Otherwise: the moment the test fails with the expected signature,
commit it. This locks in "what breaks" before any source edit. Do
**not** wait for the fix to be ready.

```bash
# Run the repo's hook set against the one file we're committing.
# See "Pre-commit hooks" section above for install / fallback guidance.
# Re-run until exit 0; some hooks auto-rewrite and re-fail on first pass.
pre-commit run --files tests/python_ops/test_<op>.py

git add tests/python_ops/test_<op>.py
git commit -m "test(<op>): add failing regression test for <category>"
```

Message style: `test(<op>): add failing regression test for <category>`.

### Step 7 — fix in `torch_neuronx`, iterate on the unit test, commit when green

**How to author the fix:** `references/dev-playbook.md` — four
principles (think before coding · fix at the highest layer · surgical
changes · goal-driven). Read it before editing. In short: map the real
boundary first (use codegraph if `codegraph_status` succeeds, else
grep), fix at the highest layer that can express it (Python op
registration over C++ `csrc` over kernels; never the compiler or the
test framework), keep the diff surgical, and loop against the unit test
until you have a green you can point at.

Edit `torch_neuronx/` and rerun **only your fence test** in your tight
inner loop — the new unit test, or the reused upstream nodeid from the
Step 6 decision. Seconds per iteration.

```bash
# new unit test:
pytest -xvs tests/python_ops/test_<op>.py::<NewTestClass>::<new_test>
# or the reused upstream test:
pytest -xvs <upstream/nodeid::from::pin>
```

If the reused upstream test is too slow to loop on (e.g. a gradcheck
sweep), that's the signal you should have authored a focused unit test
in Step 6 — go back and do so. The fence has to be fast enough to be
your inner loop.

**If your edit touches C/C++/Cython** (anything under
`torch_neuronx/csrc/` or any `.cpp` / `.h` / `.cc` / `.pyx` file), run
the project's build step (`./tools/build` or equivalent) before the
next pytest invocation. Python re-import will **not** pick up a stale
`.so`. Symptom of forgetting: the test keeps failing identically to
before the edit. When in doubt, rebuild. For the full build surface
(Bazel, CMake fallback, `tools/` drivers, env flags), see
`docs/AI-build-infra.md`. More traps in
`references/loop-playbook.md`.

Log hypotheses in `notes.md` as you go. When the unit test goes green:

1. Run the **whole** test file for the op (`test_<op>.py`) to catch
   regressions against the sibling parametrizations.
2. Confirm no pre-existing tests flip red.
3. Only then commit (below). After committing, move to Step 8 for
   the full-category verification.

**If a fix attempt fails twice, stop tweaking.** Re-read the offending
frame and look for a different root cause. Incremental patches on a
misunderstood root cause burn budget fast.

#### Commit gate — COMMIT 2 (fix)

The moment the unit test (and the rest of the op's test file) passes
on the fixed code, commit the `torch_neuronx/` edits. Do **not** roll
the registry cleanup into this commit — it needs the full-category
verify to run first.

```bash
# Lint the files we're about to commit (same hook set CI runs).
# Includes clang-format for any C++ staged. See "Pre-commit hooks" above.
# Re-run until exit 0 — auto-rewriting hooks re-fail on first pass.
pre-commit run --files torch_neuronx/path/to/fix.py torch_neuronx/path/to/other.cpp

git add torch_neuronx/path/to/fix.py torch_neuronx/path/to/other.cpp
git commit -m "fix(<op>): <short imperative> - resolves <category>"
```

Message style: `fix(<op>): <short imperative> - resolves <category>`.
Keep the imperative short — "handle 0-d tensors in index_copy", not
"this commit fixes the thing where…".

### Step 8 — verify against the full category, commit registry cleanup

Re-run the full category list and diff against the Step 3 baseline:

```bash
# Use the SAME -n and --timeout as Step 3's baseline rerun — otherwise
# outcome-diff noise is possible (e.g. a flaky test that only triggers
# at high worker counts).
$SK/scripts/verify_fixed.sh \
    /path/to/torch-neuronx-checkout \
    agent_artifacts/<slug>/tests.txt \
    agent_artifacts/<slug>/rerun-verify \
    agent_artifacts/<slug>/rerun-baseline/report.json \
    -n 16 \
    --timeout 600
```

Use the same `-n` and `--timeout` as the baseline rerun — otherwise
outcome-diff noise is possible (e.g. a flaky test that only triggers
at high worker counts).

Output: `agent_artifacts/<slug>/rerun-verify/verify.md` with per-test buckets — `fixed`,
`regressed`, `still_bad`, `unchanged` — and a non-zero exit code if
there are regressions or no fixes.

Read the results carefully:

| Observation | Action |
|---|---|
| Same error signature as the original bug | **Fix is incomplete** — back to Step 7 |
| Different error signature (new root cause) | **Long tail** — leave xfailed with an updated reason, separate PR later |
| Was passing, now failing | **Regression** — revert the fix, rethink |

A fix that clears most tests but leaves a few with the **same**
signature is not done. A fix that clears most tests and leaves a few
with a **different** signature is done — document the long tail in the
PR and move on.

#### If verify fails on a missing torch internal — overlay/venv skew

If the verify rerun produces `AttributeError`, `ImportError`, or
`ModuleNotFoundError` referencing a symbol inside `torch/...` (e.g.
`torch.testing._comparison._unwrap_dtensor_for_comparison`,
`torch.distributed.tensor.DTensor`, a private helper in
`torch/testing/_internal/common_utils.py`), **the cause is almost
always overlay/venv version skew, not a bug in your fix.** The
upstream test overlay shipped with the repo was cloned at a
different torch version than the one currently in the venv, so the
overlay calls a helper the installed torch doesn't have (or the
installed torch calls a helper the overlay doesn't expose).

Resolution path, in order:

1. **Confirm the skew.** Compare the overlay file against the venv
   file for the missing symbol:
   ```bash
   grep -n '<missing_symbol>' \
     tests/pytorch_tests_native/torch/testing/_internal/common_utils.py \
     .venv/lib/python*/site-packages/torch/testing/_internal/common_utils.py
   ```
   If the symbol exists in one but not the other, you have skew.
2. **Refresh the overlay** against the venv's installed torch:
   ```bash
   python tests/pytorch_tests_native/clone_device_agnostic_tests.py
   ```
   Then re-run verify. This is the right fix in the common case
   (overlay is older than venv torch).
3. **If the overlay is newer than the venv torch** (overlay calls a
   helper that doesn't exist yet in the installed torch): bump the
   torch dep in `pyproject.toml` to a version that has the helper,
   re-sync (`uv sync` or `uv pip install -e .`), then re-run verify.
   This is a separate, environment-pinning concern from the bug fix
   — keep it in its own commit (or its own PR if it's contentious).
4. **Stop.** Do **not** dig into the upstream PyTorch fork's git
   history to reverse-engineer compatibility. Do **not** monkey-patch
   `.venv/.../torch/...` in place to backport the missing helper —
   that violates the "Never modify the venv" rule in Critical rules
   and produces a non-reproducible fix.

If neither (2) nor (3) resolves it (e.g. the overlay clone script is
broken, or no compatible torch version exists), this is no longer a
debugger-loop problem — it's an environment/infrastructure issue.
Stop and ask the user; write a `blocked` completion manifest with
`blocked_reason: "overlay/venv skew unresolvable: <missing_symbol>"`.

#### Commit gate — COMMIT 3 (registry cleanup)

As soon as `verify.md` lists the `fixed` bucket and the `regressed`
bucket is empty, remove xfail entries for exactly the tests in the
`fixed` bucket. Commit it. Don't touch entries for tests outside that
bucket.

xfail-registry hygiene:
- Remove xfail entries only for tests we **verifiably** fixed (the
  `fixed` bucket from `verify.md`). The registry lives at
  `tests/pytorch_tests_native/neuron_test_registry.py` with per-cause
  files under `tests/pytorch_tests_native/skip_registry/` — see
  `docs/AI-testing.md`.
- If a test moved xfail→xpass incidentally, leave its xfail in place —
  a future audit PR can clean that up.
- For still-failing tests that now fail with a **different** signature,
  leave them xfailed but update the inline reason (if present) to
  point at the new root cause — helps the next fix session.

```bash
# Lint the registry files we're about to commit. See "Pre-commit hooks" above.
# Registry edits are Python; ruff + pretty-format-json + whitespace hooks apply.
pre-commit run --files tests/pytorch_tests_native/skip_registry/*.py \
                       tests/pytorch_tests_native/neuron_test_registry.py

git add tests/pytorch_tests_native/skip_registry/*.py \
        tests/pytorch_tests_native/neuron_test_registry.py
git commit -m "chore(tests): clear xfails fixed by <category>"
```

If there are no registry entries to remove (e.g. the tests weren't
xfailed to begin with), skip this commit. Two commits is fine; don't
fabricate one.

### Step 9 — draft PR (push + create)

The feature branch already exists (Step 5) and the commits are
already in place (Steps 6, 7, 8). This step only pushes them and
opens the PR. Never amends, never squashes, never force-pushes.

#### Three-commit structure (recap)

The diff tells a story; the commits make it readable, and — unique
to this skill's cadence — each has already been verified before it
was committed:

| # | Commit | Committed in step | Message style |
|---|---|---|---|
| 1 | **Unit test** (fails on unfixed code) | Step 6 gate | `test(<op>): add failing regression test for <category>` |
| 2 | **Fix** (unit test now passes) | Step 7 gate | `fix(<op>): <short imperative> - resolves <category>` |
| 3 | **Registry cleanup** (verify.md `fixed` bucket) | Step 8 gate | `chore(tests): clear xfails fixed by <category>` |

Three is the full shape, not a requirement. Drop commit 1 if you reused
the pinned upstream test as the fence (Step 6 decision — no new test
file). Drop commit 3 if step 8 produced nothing to clean up. With both,
the PR is a single fix commit. If the fix genuinely spans two unrelated
concerns, it may be split into two commits at step 7 — same ordering
principle (test before fix, fix before registry).

The verify-failing shell script from Step 4 is a **secret gist**,
not a file in the repo — its URL goes in the PR body's Reproduction
section.

#### Pre-push checks

Before the first push, confirm:

1. **Pre-commit hooks pass**, including `clang-format` for any C++
   edits (`.cpp` / `.h` / `.hpp` / `.cc` / `.cxx` — see
   `.pre-commit-config.yaml`). You should have already run these at
   each commit gate; this is a final sweep.

   ```bash
   pre-commit run --all-files
   ```

   If `clang-format` rewrites any files belonging to a commit that
   has **not yet been pushed**, amend them into that commit — the
   "pre-first-push" exception in the Commit and push policy. Then
   re-run `pre-commit run --all-files` until clean.

2. **The test story still holds** (optional; skip if the inner loop
   already proved it). Check out each commit and confirm the unit
   test's outcome matches the commit's intent:
   ```bash
   git stash     # if any work-in-progress
   # Offsets below assume the FULL three-commit shape (test, fix, registry).
   # If you skipped COMMIT 1 (reused the upstream test as the fence) or
   # COMMIT 3 (no registry cleanup), the PR is 1 or 2 commits — `git log
   # --oneline` to map each commit by its message, then adjust the offsets.
   git checkout <test-commit> -- tests/python_ops/test_<op>.py torch_neuronx  # test FAILS  (skip if COMMIT 1 dropped)
   git checkout <fix-commit>  -- tests/python_ops/test_<op>.py torch_neuronx  # test PASSES
   git checkout HEAD          -- .                                            # restore working tree to tip
   ```

3. **No uncommitted or untracked files** that should have shipped
   with the PR got left behind. In particular, the gist files
   (`agent_artifacts/<slug>/verify_failing.sh`, and `repro.py` if you made
   one) stay out of the repo:
   ```bash
   git status
   ```

### Push and open

```bash
git push -u origin "$BRANCH"

# Compose the PR body in a file first — see PR-body structure below.
# Remember to paste the gist URL from Step 4 into the Reproduction
# section of the body before running the create command.
#
# The `torch-neuronx-agent` label marks the PR as agent-raised — human
# reviewers filter on it. The label is already created on the repo; if
# ever missing, `gh pr create --label` fails and you need to run
# `gh label create torch-neuronx-agent --color AB8FFF --description
# "Raised or revised by an automation agent"` once before retrying.
gh pr create --draft \
    --title "[torch_neuronx] fix <op>: <short summary>" \
    --body-file agent_artifacts/<slug>/pr_body.md \
    --label torch-neuronx-agent \
    --base main \
    --head "$BRANCH"
```

### PR-body structure

**The PR body is the canonical context-carrier.** Someone who lands on
this PR cold — with no access to `agent_artifacts/...` and no memory of our
session — must be able to understand: what broke, what we changed,
what was verified, what's still open, and how to reproduce. Don't
reference agent-local paths.

**Ordering rule:** custom content first, repo template last. Reviewers
see the interesting details at the top; the template's hygiene
checklist is the closer.

**Wrapping rule:** **do not hard-wrap mid-paragraph.** GitHub renders
markdown with paragraph reflow, so manual line breaks at column 80 (or
wherever) show up as awkward narrow text on wide screens. Each
paragraph is one long line — let the renderer wrap. Hard line breaks
are reserved for between paragraphs, list items, and code blocks. This
applies to every prose section of the body. Tables and code fences keep
their normal line structure.

**Keep it to three paragraphs.** This is a one-bug PR, not a design
doc — the reviewer wants what broke, why, and what changed, then the
proof. Three short prose paragraphs (error · RCA+test · fix), one
verified-line, the gist link, then the filled template. Resist the
urge to add sections; depth that doesn't fit goes in `notes.md` (not
shipped) or, for deferred bugs, a single Long-tail bullet.

Structure:

```markdown
## What broke
<One paragraph: which upstream tests failed and the error signature.
Fold in why-it-matters and the category slug here — don't give them
their own sections. Link the original failing run/report if safe to
share.>

## Root cause & test
<One paragraph: the root cause (the key finding, not a dump of
notes.md), and how it's fenced — the new unit test added, or the
upstream test reused as the fence (Step 6 decision).>

## Fix
<One paragraph: what changed, at which layer, and why this approach
over the alternatives. Note if the fix had to drop below the Python
layer and why.>

**Verified:** X of Y category tests now pass, 0 regressions, 0 still
failing with the same signature. Re-run via the gist below. <If any
remain in a different signature, that's the Long tail — one bullet,
below.>

## Reproduction
Verify script (secret gist): <GIST_URL_FROM_STEP_4>

Reruns the exact failing upstream nodeids this PR addresses, with the
same `--runxfail -n 16 --timeout 600` flags the skill uses for
baseline/verify. With the Neuron venv active from a torch-neuronx
checkout:
```bash
curl -sSL <GIST_RAW_URL> -o /tmp/verify_failing.sh
bash /tmp/verify_failing.sh
```
Before this PR the listed tests FAIL or XFAIL; after, they PASS. The
registry cleanup commit removes exactly the entries the script now
reports as PASSED. (The gist may include a second file `repro.py` — a
shorter single-op Python reproduction.)

<!-- Long tail — INCLUDE ONLY IF NON-EMPTY. One bullet per deferred
signature (tests still xfailed with a *different* signature after this
fix, or incidental xpasses left for a future audit). Each is a future
PR, not this one. Omit the whole section when there's nothing to
defer. -->
## Long tail
- `<signature>` — <one-line hypothesis>, separate PR.

---

<!-- PR template checklist at the END, filled honestly. Do not rewrite
the template — paste it verbatim and check the boxes. -->
<paste contents of .github/PULL_REQUEST_TEMPLATE.md here, with boxes
checked where applicable>
```

The "Verified:" line is the verification record Mode B updates and the
marker that this PR was opened by the skill — keep it even when the
counts are trivial.

**One bug per PR.** If Step 8 revealed additional distinct root causes,
add a Long-tail bullet and keep this PR focused. Each becomes a
separate pass through this skill. The one-bug rule is about scope, not
length — but a tight description helps the reviewer as much as a tight
scope does.

### Step 10 — write the completion manifest

After `gh pr create` succeeds, write the fleet completion manifest
to `agent_artifacts/<slug>/completion.json`. This is the machine-
readable end-of-session signal fleet runners consume. See **Modes of
operation** at the top for the full schema.

```bash
# Export vars so the Python heredoc below sees them via os.environ.
# (Without `export`, shell-local vars are not inherited by subprocesses.)
export SLUG BRANCH
export PR_URL=$(gh pr view "$BRANCH" --json url --jq .url)
export PR_NUM=$(gh pr view "$BRANCH" --json number --jq .number)
export GIST_URL GIST_ID                # already-set shell vars from Step 4

python3 - <<'PY'
import json, pathlib, datetime, os, subprocess

slug    = os.environ["SLUG"]
branch  = os.environ["BRANCH"]
pr_url  = os.environ["PR_URL"]
pr_num  = int(os.environ["PR_NUM"])
gist_url = os.environ.get("GIST_URL", "")
gist_id  = os.environ.get("GIST_ID", "")

root = pathlib.Path("agent_artifacts") / slug
verify_md = root / "rerun-verify" / "verify.md"
baseline  = root / "rerun-baseline" / "report.json"

# Parse the `fixed` / `regressed` / `still_bad_*` counts out of verify.md.
# verify_fixed.sh writes these buckets as Markdown sections — grep them.
def count_bucket(name):
    import re
    if not verify_md.exists(): return 0
    text = verify_md.read_text()
    m = re.search(rf"^## {name}\s*\(\s*(\d+)\s*\)", text, re.M)
    return int(m.group(1)) if m else 0

manifest = {
    "status": "done",   # set to blocked/no-repro/regressed at the corresponding gate
    "mode":   "A",
    "category": slug,
    "branch":   branch,
    "pr":       {"url": pr_url, "number": pr_num, "draft": True},
    "gist":     ({"url": gist_url, "id": gist_id} if gist_url else None),
    "verification": {
        "fixed":                    count_bucket("fixed"),
        "still_bad_same_signature": count_bucket("still_bad"),
        "still_bad_different_sig":  count_bucket("still_bad_different_signature"),
        "regressed":                count_bucket("regressed"),
        "baseline_report":          str(baseline),
        "verify_report":            str(verify_md),
    },
    "blocked_reason": None,
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}

out = root / "completion.json"
out.write_text(json.dumps(manifest, indent=2))
print(f"wrote {out}")
PY
```

**Blocked / no-repro / regressed exits also write completion.json.**
At any "stop and ask" gate in this skill — unclear pin scope,
unexpected traceback class, git conflict, baseline that doesn't
reproduce, verify showing regressions — before returning control
to the user (interactive) or exiting non-zero (fleet), write the
manifest with:

- `status`: `"blocked"` / `"no-repro"` / `"regressed"`
- `pr`: `null` if no PR was opened yet, otherwise the existing PR info
- `branch`: the feature branch if created, otherwise `null`
- `blocked_reason`: one-line human-readable explanation (e.g.
  `"baseline report did not reproduce pinned signatures — overlay may be stale"`)
- `verification`: whatever counts are available; zeros otherwise

Fleet orchestrators key off `status` and `blocked_reason` to decide
whether to retry, route to on-call, or record the session as
complete.

## Mode B — resume an existing PR

Use this mode when the user hands you a PR instead of a pytest report.
Goal: address review comments, rerun verification, push updates. The
PR body from Mode A is your context — read it first.

### Step B1 — fetch the PR and its context

```bash
PR=<number>

# Check out the PR branch and derive the slug for artifact keying
# (single agent_artifacts/<slug>/ for the life of the bug — same
# directory Mode A used if this PR was opened by that session).
gh pr checkout "$PR"
BRANCH=$(git branch --show-current)
SLUG="${BRANCH#fix/}"
mkdir -p "agent_artifacts/$SLUG"

# Metadata, body, status, reviews, comments, files — all in one shot:
gh pr view "$PR" --json number,title,state,isDraft,body,headRefName,baseRefName,commits,reviews,reviewThreads,comments,files,statusCheckRollup,mergeable,mergeStateStatus \
    > "agent_artifacts/$SLUG/pr-metadata.json"

# Human-readable for skimming:
gh pr view "$PR"
gh pr view "$PR" --comments

# Line-level review comments (not available via `gh pr view --comments`):
gh api "repos/:owner/:repo/pulls/$PR/comments" > "agent_artifacts/$SLUG/pr-line-comments.json"

# CI status:
gh pr checks "$PR"
```

Parse the JSON or read the files directly. Classify each open review
thread and comment:

| Class | Example | Action |
|---|---|---|
| `actionable-fix` | "This branch is unreachable. Remove it." | Edit the code. Reply in thread with "done in <sha>." |
| `actionable-clarify` | "Rename `x` to `foo` for consistency with `bar.py`." | Edit. Reply with the rename commit SHA. |
| `discussion` | "Why did you choose approach A over B?" | **Reply in thread first.** Don't silently change code. |
| `nit` | Style / wording. | Fix OR reply "declining this, here's why." Don't ignore. |
| `out-of-scope` | "While we're here, also fix Z." | Reply politely declining; note Z as a follow-up PR. |
| `already-addressed` | Reviewer missed it. | Reply pointing at the existing code/commit. |

**Do not silently resolve threads you didn't address.** Don't resolve a
thread that's a discussion until the reviewer agrees.

### Step B2 — refresh the workspace

The PR branch was already checked out in Step B1 when we derived the
slug. **Before doing anything else, check the PR's mergeability state**
— the metadata you fetched in Step B1 contains it. A branch with
unresolved conflicts against `main` poisons every subsequent step
(verify reruns produce nonsense, the next push gets rejected by
required-checks, reviewers can't re-test).

```bash
# Read the merge status from the metadata fetched in Step B1.
MERGE_STATE=$(jq -r .mergeStateStatus "agent_artifacts/$SLUG/pr-metadata.json")
MERGEABLE=$(jq   -r .mergeable        "agent_artifacts/$SLUG/pr-metadata.json")
echo "mergeable=$MERGEABLE  mergeStateStatus=$MERGE_STATE"
```

Action by status:

| `mergeStateStatus` | `mergeable` | What it means | Action |
|---|---|---|---|
| `CLEAN` | `MERGEABLE` | Branch is up to date and conflict-free | Continue to the env check below |
| `BEHIND` | `MERGEABLE` | Behind `main`, no conflicts | `git fetch origin && git merge origin/main && git push` |
| `DIRTY` | `CONFLICTING` | Conflicts with `main` | Resolve via merge — see steps below. **Never** `git rebase origin/main && git push --force`. |
| `BLOCKED` | `MERGEABLE` | Required checks failing or branch protection | Surface to the user; do not push around branch protection |
| `UNKNOWN` / `null` | `UNKNOWN` | GitHub hasn't computed yet | Wait ~5s and re-query once; if still unknown, surface and stop |

For `DIRTY` (conflicting with `main`):

```bash
git fetch origin
git merge origin/main
# Editor opens on conflicts. Resolve them. The fix is usually obvious
# (the PR's own changes vs. recent main commits in the same file).
# When in doubt about how to resolve a conflict, STOP and ask the
# user — silent guesses on a merge resolution are how regressions
# slip in.
git add <resolved files>
git commit             # default merge commit message is fine
git push
```

This adds one merge commit to the branch — that's the additive,
non-destructive resolution path the *Commit and push policy →
Conflicts with the base branch* section prescribes. Do not squash it,
do not amend prior commits to "absorb" the resolution.

**After resolving (or if `CLEAN`/`BEHIND`/`UNKNOWN`-recovered), continue:**

```bash
git pull --ff-only       # in case the reviewer pushed commits; see Commit policy

# Rerun the environment health check from the Environment section.
```

If `git pull --ff-only` fails (reviewer force-pushed or your work
diverged), **stop and ask the user** — do not force around it. See
*Commit and push policy → Reviewer-pushed commits on your branch*.
If CI later reports the branch as conflicting with `origin/main`
(despite a `CLEAN`/`BEHIND` status above — possible when reviewer
commits land between your check and your push), repeat the `DIRTY`
resolution: `git merge origin/main`, fix, commit, push. Never
`git rebase origin/main && git push --force`; see *Commit and push
policy → Conflicts with the base branch*.

If the PR was opened by a prior Mode-A session, the verification
numbers and category slug are in the PR body — use them. If not (the
PR came from somewhere else), you'll need to derive the category from
the PR's title/diff, then rerun Step 3 (baseline) and Step 8 (verify)
as-needed to rebuild context. Don't invent numbers.

### Step B3 — address comments

For each `actionable-*` comment:
1. Make the edit.
2. Rerun the affected unit test (Step 7 inner loop).
3. If the edit changed the fix at all, rerun Step 8 verify against the
   ORIGINAL baseline and update the **Verified:** line in the PR body.

Do not squash review-round commits into the original fix commit, and
never force-push. Follow the **Commit and push policy** section at the
top of this skill — specifically the rule that pushed history is
immutable. New substantive changes are new commits.

### Step B4 — push and reply

```bash
# Push updates
git push

# Ensure the agent-raised label is still on the PR. This is idempotent
# — adding an existing label is a no-op — so we unconditionally apply
# it in case the PR was opened by a human and then handed to the
# agent for review-round fixes.
gh pr edit "$PR" --add-label torch-neuronx-agent

# Reply to each addressed thread. Plain `gh pr comment` posts an
# ISSUE-level comment (bottom of the PR), which is rarely what you
# want — reviewers want replies in the thread they started. Use the
# GitHub API to reply in-thread:
gh api "repos/:owner/:repo/pulls/$PR/comments/<comment_id>/replies" \
    -f body="Done in <sha>. <optional explanation>"

# Mark review threads resolved (GraphQL — `gh pr` doesn't expose this):
gh api graphql -f query='
  mutation($tid: ID!) {
    resolveReviewThread(input: {threadId: $tid}) { thread { id } }
  }' -f tid=<thread_id>
```

Only resolve threads you actually addressed with a commit or an
accepted-by-the-reviewer reply. For discussion threads, wait for the
reviewer to resolve.

### Step B5 — update the PR body and status

If verification numbers changed, regenerate
`agent_artifacts/<slug>/pr_body.md` (same file Mode A wrote in
Step 9) with the new **Verified:** line, then push it:

```bash
gh pr edit "$PR" --body-file "agent_artifacts/$SLUG/pr_body.md"
```

Keep the template checklist at the end untouched unless a box's truth
value changed.

Leave the PR as draft unless the user explicitly asks you to mark it
ready. When all CI passes and all comments are addressed, post an
issue-level comment summarizing what changed this round:

```bash
gh pr comment $PR --body "Addressed review round. Changes: …"
```

### Step B6 — write the completion manifest

Same contract as Mode A Step 10, but keyed off the slug we derived
in Step B1 — so the manifest lands in the **same**
`agent_artifacts/<slug>/` directory Mode A used, not a sibling one:

```bash
# Export vars so the Python heredoc below sees them via os.environ.
export PR SLUG

python3 - <<'PY'
import json, pathlib, datetime, os, subprocess, re

pr   = os.environ["PR"]
slug = os.environ["SLUG"]
pr_url = subprocess.check_output(
    ["gh", "pr", "view", pr, "--json", "url", "--jq", ".url"]).decode().strip()

root      = pathlib.Path("agent_artifacts") / slug
verify_md = root / "rerun-verify" / "verify.md"
baseline  = root / "rerun-baseline" / "report.json"

def count_bucket(name):
    if not verify_md.exists(): return 0
    m = re.search(rf"^## {name}\s*\(\s*(\d+)\s*\)", verify_md.read_text(), re.M)
    return int(m.group(1)) if m else 0

manifest = {
    "status":   "done",       # or "blocked" with blocked_reason on early exit
    "mode":     "B",
    "category": slug,
    "branch":   subprocess.check_output(
        ["git", "branch", "--show-current"]).decode().strip(),
    "pr":       {"url": pr_url, "number": int(pr), "draft": True},
    "gist":     None,
    "verification": {
        "fixed":                    count_bucket("fixed"),
        "still_bad_same_signature": count_bucket("still_bad"),
        "still_bad_different_sig":  count_bucket("still_bad_different_signature"),
        "regressed":                count_bucket("regressed"),
        "baseline_report":          (str(baseline) if baseline.exists() else None),
        "verify_report":            (str(verify_md) if verify_md.exists() else None),
    },
    "blocked_reason": None,
    "timestamp":      datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
(root / "completion.json").write_text(json.dumps(manifest, indent=2))
print(f"wrote {root/'completion.json'}")
PY
```

If Step B3 didn't rerun verify (no code changes that round), the
counts are `0` across the board and `baseline_report` / `verify_report`
are `null` — fleet runners treat this as "review round addressed, no
verification delta". If verify did run, the counts come from the
updated `rerun-verify/verify.md`, which overwrote the previous
round's file in the same `agent_artifacts/<slug>/` directory.

### Mode B critical rules

- **The PR body is the source of truth.** Don't trust your
  `agent_artifacts/` workspace from a previous session — it may have been
  cleaned up, or you may be a different agent instance.
- **Reply before silently changing.** Ambiguous or discussion-type
  comments get a reply explaining what you plan; silent code changes
  to questioned behavior are worst-case.
- **One comment, one thread-resolution.** Don't mass-resolve.
- **If the PR is closed/merged/stale:** stop. Ask the user. Don't
  push to a merged PR's branch.
- **If the PR wasn't opened by this skill** (no **Verified:** line,
  unfamiliar structure): help anyway, but don't rewrite the existing
  description in the skill's format — add to it, don't replace.

## Critical rules

- **Fixes go in `torch_neuronx` source.** Not in `conftest.py`, not in
  a test-patches helper, not as a per-test tolerance override unless
  the underlying op is provably well-behaved. Anything that silently
  modifies test behaviour rather than fixing the op hides the bug.
- **Never modify the venv's installed packages.** Files under
  `.venv/lib/.../site-packages/` are reproducible from `pyproject.toml`
  + the lockfile and are not in source control. If a verify failure
  traces to a missing or changed symbol in `torch/...` (e.g.
  `AttributeError: module 'torch.testing._comparison' has no
  attribute '_unwrap_dtensor_for_comparison'`), the cause is
  overlay/venv version skew and the fix path is to (a) refresh the
  upstream-test overlay via
  `tests/pytorch_tests_native/clone_device_agnostic_tests.py`, or
  (b) bump the torch dep in `pyproject.toml` and re-sync — **never**
  patch `.venv/...` in place. Patching the venv is invisible to
  other developers, vaporizes on the next `uv sync`, and hides the
  real skew rather than resolving it. See Step 8's decision node
  ("If verify fails on a missing torch internal").
- **Read guidance files in a directory before reading its code.**
  When you first open any source directory, check for in-tree guidance
  files and read them first. The common ones, in priority order:
  `AGENTS.md`, `SKILL.md`, `README.md`, `CONTRIBUTING.md`,
  `.crux_template.md`. Walk upward from the file you care about to the
  repo root; guidance in a parent directory applies to its
  descendants. These files often document invariants, conventions,
  and gotchas that are not otherwise visible from the code — ignoring
  them is how agents produce "works locally, breaks CI" patches. Do
  this once per new directory; you don't have to re-read on every
  file open.
- **xfails are failures.** Every xfail is a tracked bug. Treat them
  exactly like a `failed` test in this loop.
- **All agent writes live under a single directory:
  `agent_artifacts/<slug>/`** — one place for the life of a bug,
  across modes. Mode A picks the slug in Step 1; Mode B derives the
  same slug from the PR's branch name (`fix/<slug>`) in Step B1. No
  `/tmp/`, no `$HOME/`, no parallel `pr-<N>/` siblings. The
  exceptions are the three in-repo commits (unit test, fix,
  xfail-registry cleanup). `agent_artifacts/` must be git-ignored in
  the torch-neuronx checkout — see **Artifacts and file I/O**.
- **Always write a completion manifest.** Step 10 (Mode A) and
  Step B6 (Mode B) write `agent_artifacts/<slug>/completion.json`
  describing the session outcome (done / blocked / no-repro /
  regressed), the PR URL if any, and the verification counts. Every
  "stop and ask" gate also writes a `blocked` manifest before
  returning control. This is how fleet orchestrators know the
  session is finished. See **Modes of operation**.
- **The verify-failing script is linked from the PR body as a secret
  gist, not committed.** It's a shell script that reruns the exact
  failing upstream pytest nodeids this PR addresses with the same
  `--runxfail` flags used for baseline/verify — so reviewers can
  reproduce the before/after diff themselves and trust the xfail
  cleanup. Without the link, the chain of evidence from "verify.md
  says these tests pass" to "the registry removes exactly those
  xfails" is broken. No gist URL in the body, no merge. The gist is
  created in Step 4 with `gh gist create --filename verify_failing_*.sh`.
- **Respect Neuron platform limitations — some signatures are not
  bugs.** `float64` and complex dtypes (`complex64` / `complex128`)
  are not supported on Neuron. OpInfo dtype tests that fail purely
  because they parametrize over an unsupported dtype get xfailed
  with a `neuron_platform_limitation` reason naming the dtype — not
  a source fix. Step 4's dtype-check decision node (under "When to
  NOT fix") belongs before Step 5; skipping it is how you end up
  writing a fix for a wontfix. Real bugs that happen to involve
  float64/complex (e.g. 0-d tensor crashes, wrong index handling)
  still go through the normal fix loop.
- **One bug per PR.** Long-tail failures with different signatures stay
  xfailed and become separate PRs.
- **A failing test must fence the bug before you fix it.** If you
  author a new unit test, it must fail first and gets committed the
  moment it fails; if the pinned upstream test already fences the bug
  (in-repo, specific, fast — Step 6 decision), reuse it and skip that
  commit. Don't author a duplicate of an upstream test that already
  does the job. Either way: the fix commits when the fence passes, and
  the registry cleanup commits when verify.md shows the `fixed` bucket.
  A test that's already passing before your fix isn't a fence. Commits
  are never batched at the end of the loop.
- **`.so` rebuild after C++/Cython edits.** Python re-import will not
  pick up a stale shared library. See `references/loop-playbook.md`.
- **Stop and ask** on: unclear scope ("should I fix both bugs in this
  PR?"), destructive git operations, anything that touches
  infrastructure outside the `torch_neuronx` tree.
- **The PR body is the context-carrier.** Write it so a cold reader
  with no access to your `agent_artifacts/` workspace understands the bug,
  fix, verification, and long tail. No agent-local paths. Custom
  content first, repo template at the end.
- **PRs stay draft.** This skill never opens a non-draft PR and never
  marks one ready. Reviewers do that.
- **Every agent-touched PR carries the `torch-neuronx-agent` label.**
  Added in Step 9 via `gh pr create --label torch-neuronx-agent` for
  PRs this skill raises, and in Step B4 via `gh pr edit --add-label`
  for PRs this skill revises. The label is idempotent — re-adding it
  is a no-op — so apply unconditionally. Human reviewers filter on
  it to triage agent-raised work.
- **Reply before silently changing behavior a reviewer asked about.**
  Ambiguous or discussion-type review comments get a reply first.
  Silent code changes to questioned behavior are worst-case.
- **Only resolve review threads you addressed.** Don't mass-resolve;
  don't resolve discussion threads unilaterally.
- **Access PRs through `gh` and `gh api`.** Not raw curl, not the web
  UI. Consistent tooling makes the commands reproducible.
- **Never rewrite remote history.** No `git push --force`, no
  `--force-with-lease`, no amending a pushed commit, no rebasing a
  pushed branch. Once a commit is on the remote it is immutable; fix
  forward with new commits. Full rule set in the **Commit and push
  policy** section.
- **Branches are always `fix/<slug>`, always fresh, always off
  `origin/main`, and always created in Step 5 — before any source
  edit.** The slug must not collide with any existing local or remote
  branch — pick a new one. The PR's base is always `main`.
- **PRs open with up to three commits** (test fails → fix passes →
  registry cleanup), each committed at its own verification gate (Steps
  6, 7, 8). Commit 1 is skipped when the pinned upstream test already
  serves as the regression fence (no new test authored); commit 3 is
  skipped when there's no registry to clean. See Step 9 "Three-commit
  structure (recap)."
- **Pre-commit hooks must pass before *every* commit, not just the first
  push.** The repo's `.pre-commit-config.yaml` (ruff / ruff-format /
  clang-format / trailing-whitespace / end-of-file-fixer / check-yaml /
  check-added-large-files / pretty-format-json) is what CI enforces, and
  CI will red-X a PR that smuggled a lint violation past a non-linted
  commit. At every commit gate: `pre-commit run --files <staged paths>`
  before `git commit`, re-run until clean (auto-rewriting hooks re-fail
  on first pass and then pass). Install with `uv pip install pre-commit`
  + `pre-commit install` if missing. Full guidance — including the
  fallback to invoke `ruff`/`clang-format` directly when `pre-commit`
  can't run — lives under **Environment → Pre-commit hooks**. A final
  `pre-commit run --all-files` sweep before the first `git push` catches
  anything the per-commit scope missed; any rewrites it produces get
  amended into the commit that introduced them (pre-first-push exception).

## Output artifacts

All paths below live under `agent_artifacts/<slug>/` inside the
torch-neuronx checkout. See **Artifacts and file I/O** above for the
full layout and the **Modes of operation** section for the
completion-manifest schema.

| Path | From | Purpose |
|---|---|---|
| `agent_artifacts/<slug>/tests.txt` | Step 1 | Pytest nodeids for the pinned category |
| `agent_artifacts/<slug>/tests.txt.signatures.txt` | Step 1 | Signature groups (sanity-check) |
| `agent_artifacts/<slug>/` | Step 2 | Investigation workspace (notes.md, README.md, unit_test_template.py) |
| `agent_artifacts/<slug>/rerun-baseline/report.json` | Step 3 | Before-fix outcomes |
| `agent_artifacts/<slug>/verify_failing.sh` → secret gist | Step 4 | Pytest rerun script for the failing nodeids; **uploaded as secret gist**, URL goes in PR body. Not committed. Optional second gist file: `repro.py` for single-op Python repro. |
| `fix/<slug>` branch off `origin/main` | Step 5 | Feature branch; created before any source edit |
| `tests/python_ops/test_<op>.py` | Step 6 | Permanent regression fence; committed as soon as it fails. Skipped when the pinned upstream test already fences the bug (Step 6 decision). |
| `torch_neuronx/...` edits | Step 7 | The actual fix; committed as soon as the fence test passes |
| `agent_artifacts/<slug>/rerun-verify/verify.md` | Step 8 | Before-vs-after diff; registry cleanup committed for the `fixed` bucket |
| `agent_artifacts/<slug>/pr_body.md` | Step 9 | Composed PR body, passed to `gh pr create --body-file` |
| Draft PR on the git host | Step 9 | The shipping artefact |
| `agent_artifacts/<slug>/completion.json` | Step 9 / any blocked exit | Fleet completion manifest — see **Modes of operation** |

See `references/loop-playbook.md` for per-signature debugging
heuristics and gotchas (C++ rebuild, overlay drift, xdist tuning).
