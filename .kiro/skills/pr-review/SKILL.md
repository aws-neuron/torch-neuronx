---
name: pr-review
description: Review a torch-neuronx GitHub PR or local branch for code quality, test adequacy, backward compatibility, security, and correct use of the project's infrastructure (@neuron_op / @register_aten, NKI wrappers, cast policy, NEFF cache, torch.compile backend="neuron" constraints, xfail registry, tools/run-test*). Use when the user says "review this PR", "code review", "self-review my branch", or gives a PR number / URL / commit range. Three modes — PR mode (fetches via `gh pr view` / `gh pr diff`), branch mode (diffs against `origin/main`), and commit-range mode. Review output reports problems only; nothing that CI already enforces. Cross-references `torch-neuronx-docs` for subsystem context.
---

# torch-neuronx PR Review Skill

Review `torch-neuronx` GitHub PRs — or a local branch before raising
one — focusing on what CI cannot check: design quality, use of
project infrastructure, test adequacy, backward compatibility of the
`torch_neuronx` public surface, thread/compilation safety, and
Neuron-specific correctness (dtype cast policy, contiguity, NEFF cache,
C++ dispatch / Python op fallback, torch.compile constraints).

This skill is the **reviewer companion** to
`pytorch-upstream-test-debugger` (which produces PRs). It does not
duplicate structural knowledge of the codebase — that lives in
`torch-neuronx-docs`.

## Companion skill: `torch-neuronx-docs`

This skill is a **review workflow driver**. The structural / subsystem
knowledge it relies on lives in the `torch-neuronx-docs` skill.

**Invocation is mandatory, not optional.** After mapping touched files
to subsystems in Step 1 of the workflow, invoke `torch-neuronx-docs`
once per touched subsystem (in parallel — independent reads):

```
Skill(skill="torch-neuronx-docs", args="04-python-ops.md")
Skill(skill="torch-neuronx-docs", args="07-compilation-kernels.md")
```

Do this **before** drafting any finding that depends on a contract
claim ("should this have used `@neuron_op`?", "does this break the
NEFF cache key?", "does VALUE_DEPENDENT preprocessing handle inplace
overloads?"). A finding grounded only in source-grep is fragile —
the docs codify the intended contract, and a mismatch between source
and contract is itself a finding.

| Reviewing code in… | Invoke `torch-neuronx-docs` with |
|---|---|
| `torch_neuronx/python_ops/` (eager ops, NKI wrappers, cast policy, `register_aten`, preprocessing protocol) | `04-python-ops.md` |
| `torch_neuronx/neuron_dynamo_backend/` (torch.compile path) | `03-dynamo-backend.md` |
| `torch_neuronx/csrc/` (C++ extension, dispatcher, streams) | `05-csrc.md` |
| `torch_neuronx/` top-level APIs (`torch.neuron.*`, Stream, Profiler, `nki_op`) | `02-runtime-api.md` |
| Distributed / `ProcessGroupNeuron` / collectives / DTensor | `06-distributed.md` |
| NEFF cache, compiler subprocess, artifacts | `07-compilation-kernels.md` |
| Tests (`tests/…`, drivers, markers, xfail registry) | `08-testing.md` |
| Build / CI / `tools/` drivers | `10-build-infra.md` |

**Skip only when** the diff is trivial enough that no finding will
turn on a subsystem contract (pure docstring / typo / comment-only
PRs). For any code change inside `torch_neuronx/`, invoke at least
the doc that covers the touched subsystem.

If a finding contradicts the doc, prefer the source as ground truth
(docs may lag) but call out the discrepancy in the review so the
docs can be updated.

## Usage Modes

### No argument

If the user asks for a PR review with no target, **do not invent one**.
Ask:

> What would you like me to review?
> - A PR number or URL (e.g. `#642` or the PR's GitHub URL)
> - A local branch (`review branch`) — reviews commits on HEAD not yet in `origin/main`
> - A commit range (`review HEAD~3..HEAD`)

Add `detailed` to any of the above to get file/line-level specific
comments in addition to categorized findings.

### PR mode

User provides a PR number or URL. **Always fetch via `gh`** — do not
switch to a local checkout even if the PR's branch is checked out,
and do not run any `git` commands in this mode. `gh pr diff` is the
source of truth for what's being reviewed.

```bash
# Metadata, body, commits, checks
gh pr view <number> --json number,title,body,state,baseRefName,headRefName,author,commits,files,reviews,comments,statusCheckRollup

# Full diff
gh pr diff <number>

# Existing review comments (so you don't duplicate points already raised)
gh pr view <number> --comments
gh api repos/{owner}/{repo}/pulls/<number>/comments   # inline/thread comments
```

If `gh` isn't available or the PR is on a fork you can't access, ask
the user to run the commands and paste the output. The PR's GitHub
URL can also be fed to your browser-fetch tooling.

Read existing review threads before adding yours — if a point has
already been raised, either don't repeat it or mark your finding as
"seconding @reviewer: …". Duplicated comments are noise.

**PR baseRefName sanity check.** This repo ships both `main` and
`mainline` as remote branches (historical — `mainline` is a legacy
branch that still exists). Real PRs merge into **`main`**; confirm
via `baseRefName` in the `gh pr view` output. If a PR targets
`mainline`, flag that as an issue — it's almost certainly wrong.

### Posting the review back to GitHub

**Default delivery mode depends on invocation:**

| Invoked by | Default | When the default applies |
|---|---|---|
| A human user, no instruction | **Report-only.** Print the review as a Markdown report; the user reads it and decides whether to post it. | The user can ask you to post afterward. |
| An orchestrator agent (e.g. `upstream-debugger-orchestrator`) | **Publish.** Post the review as a single GitHub review (one top-level body + N inline comments) via the Reviews API. | The orchestrator's prompt explicitly tells you to publish; treat that as authoritative. |
| Any caller saying "don't post" / "just summarize" / "report only" | **Report-only.** Override the publish default. | Honor the explicit override. |
| Any caller saying "post" / "publish" / "submit the review" | **Publish.** | Honor the explicit override. |

The orchestrator-driven default exists because the alternative (the
agent producing a markdown report that nobody sees because the
orchestrator only forwards a 1-line summary) is the failure mode
caught in the upstream-debugger-orchestrator retro: 11 of 13 reviewer
agents stayed silent when their prompt said "read-only review". When
the goal is "the PR has a review on it", publish is the right default.

When publishing, build a single review payload and post it via the
GitHub Reviews API. This creates one review with one top-level body
and N inline comments — visible on the PR's "Files changed" tab.

```bash
# Write the JSON payload
cat > /tmp/pr<number>_review.json <<'EOF'
{
  "body": "**[AI review]** One sentence on what the PR does. **Request Changes** — <top blocker>.",
  "event": "COMMENT",
  "comments": [
    {
      "path": "torch_neuronx/python_ops/torch_mlir/ops/indexing.py",
      "line": 292,
      "side": "RIGHT",
      "body": "**[AI review]** **[blocker]** <one-line headline>. <one paragraph of detail with file:line refs and the suggested fix>."
    }
  ]
}
EOF

gh api repos/<owner>/<repo>/pulls/<number>/reviews -X POST \
  --input /tmp/pr<number>_review.json
```

The top-level `body` is **one sentence** — what the PR does in your
own words, plus the verdict — not a paste of the PR description and
not a re-summary of every finding. See "Output Format" below for the
full shape.

Notes:

- `event`: pick the verdict by mapping findings to the right GitHub
  state:

  | Findings | `event` |
  |---|---|
  | No issues; CI clean; code looks good | `APPROVE` |
  | Style / nit / minor improvements only | `COMMENT` |
  | Substantive issue (logic bug, missing test, BC concern, security) | `REQUEST_CHANGES` |

  GitHub blocks `REQUEST_CHANGES` and `APPROVE` on your **own** PR —
  those return 422. For self-review (the PR was opened by the same
  account that's now reviewing), always use `"COMMENT"` regardless of
  finding severity, and put the verdict in the body text itself
  (`**Request Changes** — <top blocker>` style). The orchestrator-
  driven case is usually self-review (the agent that opened the PR
  is now reviewing it via a fresh subagent), so `"COMMENT"` is the
  expected event there.
- `line` must be a line **in the diff hunk**. Lines outside the hunk
  return 422; if so, anchor on the nearest changed line in the same
  file and reference the actual target in the comment text.
- `side: "RIGHT"` = the new version (post-PR). Use `"LEFT"` only when
  pointing at a line being removed.
- Tag every AI-generated comment with a bold `**[AI review]**`
  prefix, then a severity tag (`**[blocker]**` / `**[important]**` /
  `**[nit]**`) at the start of the `body` and every
  `comments[].body`. Provenance first, severity second:
  `**[AI review]** **[blocker]** <headline>. <detail>`. The top-level
  `body` only needs the `[AI review]` prefix (the verdict on the same
  line carries the severity signal there). Both prefixes must be
  inside the JSON string and use Markdown bold so the GitHub UI
  renders them. Use the same `[AI review]` tag whether the user is
  reviewing their own PR or someone else's — the tag marks provenance
  (AI), not authorship (self / other).

### Re-review: closing the loop on resolved findings

On the **second and later** iterations (the orchestrator re-runs you after the
Implementer committed fixes and the Verifier re-PASSed), the individual comment
threads from your *previous* review are still open in the GitHub UI even though
your new top-level body says "resolved". A human reviewer treats the unresolved-
conversation count as their checklist, so leave the threads reconciled — but only
in a factual, non-negotiated way:

- **Reply factually, linked to the fix commit.** For each of *your own prior
  `[AI review]` findings* that you have **independently re-confirmed** is fixed
  (re-read the new code; don't take the Implementer's word), post a one-line reply
  to that thread: `**[AI review]** Resolved in <sha> — <what changed>.` No
  argument about whether the finding was valid; just the resolution.
- **Then resolve the thread — self-review threads only.** After the factual reply,
  mark the conversation resolved via GraphQL `resolveReviewThread`. Do this **only
  for threads you (the AI reviewer) authored.** Never resolve or dismiss a thread
  authored by a **human** or an **external bot** (e.g. Copilot / github-code-
  quality) — that checkbox is the human's, and auto-dismissing hides a finding
  they may want.
- **A finding you now believe was wrong (not actually fixed, or never a real
  issue) is NOT self-cleared.** Do not reply-to-dismiss it; surface it and let the
  orchestrator route it to a human (see stop-and-ask). Reply-to-resolve is for
  genuinely-fixed findings only.
- **Gate on the Verifier.** Only post resolution replies once the orchestrator has
  told you the Verifier still returns PASS for this iteration; a resolution note on
  an unverified fix is exactly the false-consensus this guard exists to prevent.

```bash
# factual reply on a review-comment thread (COMMENT_ID from the pulls/.../comments list)
gh api repos/<owner>/<repo>/pulls/<number>/comments/<COMMENT_ID>/replies -X POST \
  -f body='**[AI review]** Resolved in <sha> — factored into weight_norm_interface_backward_grads().'

# resolve the thread (own self-review threads only) — needs the thread node id
gh api graphql -f query='mutation($id:ID!){resolveReviewThread(input:{threadId:$id}){thread{isResolved}}}' \
  -f id='<THREAD_NODE_ID>'
```

This is the **Reviewer's** job, not the Implementer's — the Implementer still
never touches PR comments. It is a natural extension of the re-review you already
publish (better placement of the "resolved" claim you're making anyway), not a new
channel for defending or negotiating the code.

### Branch mode

Review commits on the current branch that are not yet in
`origin/main`:

```bash
git -P fetch origin main
git -P log --oneline origin/main..HEAD                 # commits in scope
git -P diff --stat origin/main..HEAD                   # files + line counts
git -P diff origin/main..HEAD                          # the full diff
git -P diff --name-only origin/main..HEAD              # changed files
```

The review header should use the current branch name (from
`git -P branch --show-current`) and list the PR title the author will
use, derived from the commit log.

Before reviewing, verify the branch is not behind:

```bash
git merge-base --is-ancestor \
  $(git ls-remote origin main | cut -f1) HEAD \
  && echo "ahead/up-to-date" || echo "behind/diverged"
```

If behind, note that in the review — the author must sync before
opening the PR (or GitHub will refuse the merge until they do).

### Commit-range mode

`review HEAD~3..HEAD` or `review <sha1>..<sha2>`. Treat the range as
the diff to examine.

## Review philosophy

The torch-neuronx dispatcher, cast policy, NEFF cache key, and
torch.compile backend are **deeply cross-cutting**. One missed detail
can cause silent wrong-numerics or a cache-key collision across users:

- A new op that forgets `ensure_contiguous_on_device` will silently
  re-enter the dispatcher and loop.
- A new `_execute_impl` that doesn't declare `DISABLE_DTYPE_AUTOCAST`
  when it operates on `int64` input will get its inputs cast to int32
  underneath, corrupting indices.
- A new NKI kernel wired through `NKITorchMlirOpImpl` whose
  `backend_config` isn't part of the cache key will serve a stale NEFF
  the second time it's called with different constants.
- A new `torch.compile` FX pass that mutates the graph in place, not
  behind a decomposition table, will apply to every other backend too.

**Treat every changed line as potentially load-bearing.** Follow the
same 9 principles as pytorch upstream:

1. **Only report problems.** The review output contains issues,
   concerns, and actionable suggestions. Do not praise. Do not say "no
   concerns" outside the Summary. Omit any section with no findings.
2. **Investigate, don't guess.** When uncertain whether a checklist
   item applies, read the surrounding code (`read` + `grep` tools) or
   spawn a sub-agent to read related subsystems. A reviewer who
   guesses wrong provides negative value.
3. **Review the design, not just the implementation.** Flag
   side-channel communication (hidden flags set in one place and
   checked via `getattr` elsewhere), private booleans that swap
   between two fundamentally different code paths, and "internal"
   kwargs like `_internal=True` that gate public APIs.
4. **Focus on what CI cannot check.** Do not comment on ruff
   formatting, import order, `# type: ignore`, trailing whitespace,
   clang-format, or missing `__init__.py` — `lint.yml`, pre-commit,
   and `pull.yml` already block those. Focus on Neuron-specific
   correctness, pattern adherence, test adequacy, BC.
5. **Everything is a must-fix.** No nits. If it's worth mentioning,
   it's worth fixing. Every inconsistency degrades the codebase.
6. **Be specific and actionable.** Reference file paths and line
   numbers. Name the function/class/file the author should use.
7. **Match the immediate context.** Read how similar features are
   already implemented in the same file. Pattern mismatches inside a
   file are always wrong.
8. **Assume competence.** The author knows Neuron and PyTorch;
   explain only non-obvious context.
9. **No repetition.** Each observation appears in exactly one section.

### Using sub-agents

The checklist is large. You cannot hold the full context of every
infrastructure system in your head. For a medium CR, expect to spawn
**3–8 sub-agents in parallel** to investigate:

- Whether a similar op already exists and uses a different pattern.
- Whether a touched code path has existing tests under
  `tests/python_ops/` / `tests/neuron_dynamo_backend/` / etc.
- Whether the public `torch_neuronx.*` API being modified is exported
  in `torch_neuronx/__init__.py`.
- Whether a modified FX pass is invoked on any existing tested model.
- Whether the NEFF cache key covers every input that changes output
  shape / dtype.

Run independent investigations in the same tool-call block.

**Use Opus, not the default Haiku, for these calls.** Pass
`model: "opus"` on every `Agent` invocation in this skill — for
contract investigations, fact-check passes, and the optional
Step 5 self-check. Two reasons:

1. PR review findings rest on contract *synthesis* ("does this
   respect the preprocessing protocol?", "is this consistent with
   the cast policy?"), not just grep-and-cite. Haiku has fabricated
   subsystem-contract claims in this repo before; verify mode in
   `torch-neuronx-docs` exists in part because of that history.
2. Settings.json `model` only controls the main thread, not
   sub-agents. Without an explicit `model: "opus"` per call, the
   harness falls back to the built-in Haiku default for `Explore` /
   `general-purpose` — silently downgrading review quality.

Cost difference is acceptable for review quality.

## Review Workflow

### Step 1 — Understand context

1. Identify the PR/branch's stated purpose from
   title / description / linked issue. The PR template at
   `.github/PULL_REQUEST_TEMPLATE.md` defines the expected structure
   (Description, Motivation, Type of Change, Testing, Checklist) —
   if sections are empty, flag that.
2. Group the diff: new code, modified code, tests, docs, build
   infra.
3. List the changed files and map each to a subsystem:

   | Path prefix | Subsystem | Overview doc |
   |---|---|---|
   | `torch_neuronx/python_ops/` | eager ops | 04-python-ops.md |
   | `torch_neuronx/neuron_dynamo_backend/` | torch.compile | 03-dynamo-backend.md |
   | `torch_neuronx/csrc/` | C++ extension | 05-csrc.md |
   | `torch_neuronx/distributed/` | collectives, DTensor | 06-distributed.md |
   | `tests/` | tests | 08-testing.md |
   | `tools/` / `BUILD.bazel` / `CMakeLists.txt` / `setup.py` | build | 10-build-infra.md |
   | `docs/` / `*.md` | docs | — |

4. Read the unchanged code surrounding each significantly-changed file
   to learn the existing pattern. A class that already uses
   `CompilableOpImpl` + `_execute_impl` defines the pattern for new
   ops added to that file.

5. **Invoke `torch-neuronx-docs` for every touched subsystem** (in
   parallel). The mapping table is in the "Companion skill" section
   above. This loads the contract you'll be checking findings against
   in Step 2 — skipping it means findings rest on source-grep alone,
   which is fragile.

### Step 2 — Deep review

Walk **every changed line** in the diff against
[review-checklist.md](review-checklist.md). The checklist is grouped
so you can skip sections that don't apply to the touched paths.

### Step 3 — Check backward compatibility

Evaluate BC per [bc-guidelines.md](bc-guidelines.md). The public
surface is:

- Symbols exported by `torch_neuronx/__init__.py` (everything with a
  non-underscore name is public).
- The `torch.neuron.*` module APIs registered via
  `torch._register_device_module("neuron", neuron)`.
- The `'neuron'` device string itself and the tensor behaviors it
  implies (dtypes, contiguity, etc.).
- The contract of `torch.compile(backend="neuron")`: supported
  parameters, accepted dtypes, accepted shapes.
- Environment variables listed in `docs/environment_variables.md` —
  renaming / removing one is a BC break.
- NEFF cache artifacts on disk — changing the key schema silently
  invalidates users' caches; flag it.

For non-trivial BC questions, spawn a sub-agent to grep for callers
of the modified API.

### Step 4 — Formulate review

Structure the output as in "Output Format" below. Every finding maps
to a specific file and line in the diff.

### Step 5 — Fact-check (conditional)

Drafted findings can be wrong: a cited line number may have shifted,
a referenced helper may have been renamed, a "you bypassed X"
finding may turn out to be false because the diff actually does call
X two lines up. Fact-checking catches these before they hit the user
or the PR.

**Run this step when any of the following holds:**

- ≥ 5 findings drafted, OR
- any finding cites a file you did not read end-to-end, OR
- any finding asserts a pattern claim ("the rest of the file does
  X", "the convention here is Y") without you having verified the
  full file, OR
- the review will be posted directly as inline PR comments (the user
  can't easily edit them after the fact).

**Skip this step when** the review is small (≤ 4 findings) and you
re-read every cited file inline while drafting — re-reading
individual cited lines as you draft is sufficient.

When running it: spawn one sub-agent per reported issue, in
parallel, to re-read the cited code and return **valid**,
**invalid**, or **needs rewording**. Drop invalid issues. Reword or
soften low-confidence ones with an explicit "I'm not sure X — please
confirm" rather than a directive.

## Output Format

Two rules govern the shape of every review:

1. **The top-level body is short.** One sentence on what the PR does
   (in your own words — do **not** paste from the PR description; the
   author and other reviewers can read that themselves), then the
   verdict, then at most one line naming the top blocker(s). All
   detail belongs in labeled findings, not in the body.
2. **Every finding carries a severity tag** so the author can triage
   at a glance. Three tiers — choose the one that matches the
   highest-cost consequence of *not* fixing the finding:

   | Tag | When to use |
   |---|---|
   | **`[blocker]`** | Correctness, BC break, silent wrong-numerics, security, missing test for a bug fix or new functionality. Author should not merge without addressing. |
   | **`[important]`** | Pattern violation, design concern, infra-helper bypass, fact the author probably wants to know but could reasonably argue should ship as-is. Reviewer wouldn't approve over the author's objection without discussion. |
   | **`[nit]`** | Naming, comment wording, micro-readability, style preference inside the file's existing pattern. **Use sparingly** — most repos discourage nits in inline review. Skip if you can't articulate why this matters more than personal taste. |

   Inline comments use the same tag at the start of the body, after
   the `**[AI review]**` provenance prefix:
   `**[AI review]** **[blocker]** <one-line headline> — <detail>`.
   Categorized-section findings prefix the same way:
   `- **[blocker]** <headline>...`.

```markdown
## PR Review: #<number> — <title>
## Branch Review: <branch-name> (vs origin/main)

One sentence on what the PR does. **<Approve | Request Changes |
Needs Discussion>** — <one line on top blocker(s) if requesting
changes; otherwise omit>.

### Code Quality
[Problems only — abstractions, dead code, hidden flags, unclear names.
Each finding tagged.]

### Infrastructure
[Problems only — checklist violations: wrong pattern, bypassed helper,
ignored cache key, missing C++ dispatch registration, etc.]

### Testing
[Problems only — missing tests, wrong test file, manual loops where
device-generic tests exist, missing xfail-cleanup for fixed tests]

### API Design
[Problems only — flag-gated internals, new conventions without
justification, poor discoverability]

### Security
[Problems only — unsafe deserialization, credential leaks in logs,
execution of untrusted pickle, etc.]

### Thread / Compilation Safety
[Problems only — data races in the dispatcher, unprotected cache
mutation, NEFF cache key holes, GIL handling across Py↔C++ boundary]

### Backward Compatibility
[Problems only — removed / renamed public API, changed defaults,
changed env var names, cache-key schema changes that invalidate
existing artifacts]

### Performance
[Problems only — obvious regressions: extra Py↔device hops, forced
recompilation, redundant contiguous copies]
```

**Verdict heuristics.**
- Any `[blocker]` finding → **Request Changes** (or **Needs
  Discussion** if the blocker hinges on judgment, e.g. a BC
  question that should escalate to the package owners).
- Only `[important]` and `[nit]` findings → **Approve** is the
  default; the author decides which to address.
- Missing tests for new functionality or a bug fix is always a
  `[blocker]`.

**No `### Summary` section, no `### Recommendation` section.** The
verdict lives on the same line as the one-sentence summary at the
top — separate sections waste lines and tempt re-statement. Omit any
category section with no findings.

### Specific Comments (detailed review only)

Only include when the user asked for `detailed`. Do not repeat
observations already in the categorized sections. This section is for
additional file-specific feedback that doesn't fit a category:

```markdown
### Specific Comments
- `torch_neuronx/python_ops/gather.py:142` — extract the stride
  normalization into a named helper; used in 3 other impls already
- `tests/python_ops/test_gather.py:57` — missing coverage for
  int64 indices after the cast-policy change
- `torch_neuronx/csrc/core/dispatch/NeuronDispatcher.cpp:210` —
  this allocation is in the hot path; hoist above the loop
```

## Files to consult during a review

Read these rather than relying on memory:

- `CONTRIBUTING.md` — developer workflow, expected build / test
  commands, op-coverage recipe.
- `README.md` — user-facing contract for `torch.compile(backend="neuron")`
  (supported params, supported dtypes). Changes here are BC-visible.
- `docs/environment_variables.md` — canonical env-var list. Renames /
  removals are BC breaks.
- `docs/new_op.md` — if the CR adds an op, the author should have
  followed this recipe.
- `torch_neuronx/__init__.py` — the package entry point. What gets
  imported/registered here is public.
- `torch_neuronx/python_ops/__init__.py` — the `register_python_operations()`
  order. New ops must be registered in the right phase.
- `torch_neuronx/python_ops/base.py` — `Operation`,
  `OperationImplementation`, `ExecutionResult`. Any new op impl must
  follow this contract.
- `torch_neuronx/python_ops/auto_registration.py` — `@neuron_op`
  semantics, priority ordering, `AutoOperation` wrapper.
- `torch_neuronx/python_ops/torch_mlir/operation_registry.py` —
  `@register_aten` semantics.
- `torch_neuronx/python_ops/cast_policy.py` — `bounce_64bit`,
  `ensure_contiguous_on_device`, `validate_neuron_memory_format`. Any
  new Py↔device transfer must go through these.
- `tests/pytorch_tests_native/neuron_test_registry.py` — single
  source of truth for upstream-test skip/xfail. Fixes must remove
  their `SkipGroup` entry here, not flip `@pytest.mark.xfail`
  decorators inline.
- `conftest.py` — registered markers (`slow`, `multi_device`),
  filename conventions (`_lnc1`, `_all_cores`), the
  `ensure_neuron_initialized` / `reset_dynamo` fixtures tests must
  cooperate with.
- `.github/workflows/lint.yml` — scope of CI-enforced lint. Do not
  duplicate those findings in the review.
- `.github/workflows/pull.yml` — CI that runs on the CR. Know what
  it covers so you don't re-report it.

## Git conventions specific to this repo

These apply to **branch mode** and **commit-range mode** only.
**PR mode does not run any `git` commands** — `gh pr diff` and
`gh pr view` provide the diff, base ref, and metadata directly.

- Destination branch is **`main`**. Use `origin/main` in all `git`
  commands. A legacy `mainline` branch still exists on the remote but
  is not the PR target — if a PR is opened against it, flag that.
- Use `git -P` (`--no-pager`) on any command that can paginate.
- PRs are opened with `gh pr create`. Review threads live on the
  repo's GitHub PR page.
- Expect commit messages to follow Conventional Commits. A `sim:` or
  `taskei:` footer linking the tracking issue is common but not
  required.

## What to skip (CI covers it)

Do not report:

- ruff / black / isort formatting (`.github/workflows/lint.yml`)
- clang-format (`.pre-commit-config.yaml`)
- `# type: ignore` abuse — no mypy gate in CI today
- unit-test success/failure on the PR itself — `pull.yml` reports it
- missing `__init__.py` or Bazel target boilerplate that `pull.yml`
  would catch
- import order

## Running this skill as a subagent

This skill is designed to be invoked as a separate session. When the
user says "review #642" or "review this PR", load this skill, fetch
the PR via `gh`, and produce the structured review. Do not attempt to
fix the PR yourself in the same session — hand off to the author or to
`pytorch-upstream-test-debugger` Mode B, which is built to consume
review comments.
