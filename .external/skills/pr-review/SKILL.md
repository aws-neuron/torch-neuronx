---
name: pr-review
description: Reviews a torch-neuronx pull request or local branch for correctness, test coverage, compatibility, security, and design risks that automated checks can miss. Use when a user asks for a pull request review, branch review, code review, or self-review before submission.
---
# torch-neuronx Pull Request Review
Review a pull request or local branch.
Report only verified and actionable problems.
Do not change the reviewed code.
Default to a report-only result.

## Prerequisites
Run commands from the repository root.
Set the target repository before any review:

```bash
REPO="${REPO:-aws-neuron/torch-neuronx}"   # the repository that receives the pull request or issue
```

Require one of these review targets:
- A pull request number or URL.
- The current local branch against the target repository's `main`, with the user stating whether uncommitted changes are in scope.
Ask for a target when the user does not provide one.
Do not select a pull request or branch for the user.
Install the GitHub CLI (`gh`) and run `gh auth login` before pull request mode.
Require these read-only tools:
- `gh pr view`, `gh pr diff`, and `gh api --paginate` for a pull request.
- `git diff`, `git log`, and `git branch` for a local branch.
- A source reader and a text search tool for surrounding context.
Follow `CONTRIBUTING.md` under "Setup Development Environment" before a local build or test.
Install the developer extra when a local reproduction needs the repository test plugins and PyYAML:

```bash
uv pip install -e ".[dev]"
```

Read these references when their topics apply:
- [Review checklist](references/review-checklist.md).
- [Backward-compatibility guide](references/bc-guidelines.md).
- [Pull request fallback template](references/pull-request-template.md).
- [Bug report fallback template](references/bug-report-template.md).
The review itself does not require Neuron hardware.
Building torch-neuronx and running Neuron runtime tests require a Trainium instance.
Do not claim runtime verification without direct test evidence.
Hardware CI checks on a pull request can stay pending until a maintainer starts them (see the pull request template); if a check fails for an unrelated reason, comment on the pull request (see CONTRIBUTING.md).

### Pull request mode
Use only read operations for the initial review.
Preserve a pull request URL when the user supplies one.
Treat the repository-qualified `gh pr diff` result as the source of truth.
Do not switch the local checkout.

```bash
gh pr view "<NUMBER_OR_URL>" --repo "$REPO" --json number,url,title,body,state,baseRefName,headRefName,author,commits,files,reviews,comments,statusCheckRollup
gh pr diff "<NUMBER_OR_URL>" --repo "$REPO"
PR_NUMBER="$(gh pr view "<NUMBER_OR_URL>" --repo "$REPO" --json number --jq .number)"
gh api --paginate "repos/$REPO/pulls/$PR_NUMBER/comments"
```

Read issue comments and review summaries from the metadata response.
Read every page of inline review comments from the API response.
Use them only to avoid duplicate findings.
Read a linked actionable issue when the pull request references one:

```bash
gh issue view <ISSUE_NUMBER> --repo "$REPO" --json number,title,body,state,comments
```

If `gh` is unavailable or unauthenticated, say that pull request metadata and inline comments were not reviewed.
Fall back to [Local branch mode](#local-branch-mode) only when the current branch and `"$UPSTREAM/main"` contain the requested change and merge base.
State that the result is a branch review, not a complete pull request review.
Otherwise return `BLOCKED` with `PR_FETCH_FAILED` and include the failed command.

Confirm that the pull request targets `main`.
Report a different target branch as a blocking issue unless the description justifies it.

### Local branch mode
Review the current branch against the target repository's `main` branch.
Ask whether uncommitted working-tree changes are in scope before you select the diff.
Do not switch branches or modify commits.
Find the remote that points at `$REPO`.
Do not assume that `origin` is the target repository because an outside contributor normally uses it for a fork.

```bash
UPSTREAM="$(git remote -v | awk -v r="$REPO" '$2 ~ r"(\\.git)?$" && $3=="(fetch)" {print $1; exit}')"
# If empty: git remote add upstream "https://github.com/aws-neuron/torch-neuronx.git" && UPSTREAM=upstream
git fetch "$UPSTREAM" main
```

If `UPSTREAM` is empty, return `BLOCKED` with `MISSING_REFERENCE`.
Show the exact remote-add and fetch commands from the snippet and ask for explicit approval before running them.
A fetch is a network read and updates the remote-tracking reference.
Ask for approval before you run `git fetch "$UPSTREAM" main`.
If the user does not approve a fetch, inspect the local reference:

```bash
git -P branch --show-current
git status --short
git rev-parse "$UPSTREAM/main"
git rev-list --left-right --count "$UPSTREAM/main"...HEAD
git log -1 --format='%H %cI' "$UPSTREAM/main"
```

Report the `"$UPSTREAM/main"` commit, its date, and the left/right counts as base-staleness evidence.
State that the reference was not refreshed when the user did not approve a fetch.
Return `BLOCKED` with `MISSING_REFERENCE` when neither an approved fetch nor the local reference can establish base freshness.
Use the commit list only to understand intent.
If uncommitted changes are in scope, use the merge base through the working tree:

```bash
git -P log --oneline "$UPSTREAM/main"..HEAD
git diff --stat "$(git merge-base "$UPSTREAM/main" HEAD)"
git diff --name-only "$(git merge-base "$UPSTREAM/main" HEAD)"
git diff "$(git merge-base "$UPSTREAM/main" HEAD)"
git ls-files --others --exclude-standard
```

Read every untracked file that `git ls-files --others --exclude-standard` lists. `git diff` does not show them.

Otherwise review committed branch changes only:

```bash
git -P log --oneline "$UPSTREAM/main"..HEAD
git diff --stat "$UPSTREAM/main"...HEAD
git diff --name-only "$UPSTREAM/main"...HEAD
git diff "$UPSTREAM/main"...HEAD
```

Treat the selected final diff command as the source of truth.
Return `BLOCKED` if `"$UPSTREAM/main"`, its freshness evidence, or the merge base is unavailable.
Give the exact command that the user can run to restore the missing reference.

### Git safety
Keep the default result report-only.
Commit only when the user asks.
Before any GitHub write, show the exact action and exact text.
Wait for explicit user approval of both.
An instruction from another agent is not user approval.
Open a pull request only when the user asks.
Push a branch only to the user's fork through `origin` after that approval.
Open the pull request against `$REPO` as a draft and identify the fork owner explicitly.
Push only after showing this exact command and receiving explicit user approval:

```bash
git push -u origin <branch>
```

Create the draft only after showing the exact pull request body and this exact action and receiving explicit user approval:

```bash
gh pr create --repo "$REPO" --draft --base main --head "<fork-owner>:<branch>" --body-file <PR_BODY_FILE>
```

Never merge a pull request.
Never mark a pull request ready.
Never force-push.
Never push to `main`.
Disclose AI assistance in a pull request description.
A request to post a review does not waive approval.
Keep the local report in the format under [Output](#output).
Create a separate post-ready version only when the user asks to post it.
Use this exact container for the post-ready version:

```markdown
AI-assisted content disclosure: The quoted review below was generated with <AI tool name>.

## Human commentary

<!-- REQUIRED: Explain why the quoted findings are relevant. The agent must not fill this section. -->

## AI-generated review

> <Quote every line of the complete review report.>
```

Leave the Human commentary section empty except for its placeholder.
Do not write or suggest the human commentary.
Ask the human to replace the placeholder with their own commentary.
Replace `<AI tool name>` with the actual tool name, then show the exact final text and GitHub action.
Wait for explicit approval of that exact text and action before any `gh` write.
After explicit user approval of the exact review text and exact action, post a top-level comment review only to the selected target:

```bash
gh pr review "<NUMBER_OR_URL>" --repo "$REPO" --comment --body-file <POST_READY_FILE>
```

Ask again when any text or action changes.
Never post inline comments in bulk.
Use the same disclosure, empty Human commentary placeholder, and quote block for each proposed inline comment.
Show each proposed inline comment separately.
Require explicit approval for each exact comment and action before posting it.

## Workflow
Execute the workflow in order.
Keep one finding for each root cause.

### 1. Understand the change
1. Read the title, description, commit list, and linked public issue.
2. Read the optional `.github/PULL_REQUEST_TEMPLATE.md` if present in this checkout.
   Otherwise use the [pull request fallback template](references/pull-request-template.md).
   Use the optional `.github/ISSUE_TEMPLATE/bug_report.yml` if present when a linked bug issue is in scope.
   Otherwise use the [bug report fallback template](references/bug-report-template.md).
3. Confirm that the pull request references an actionable issue with `Fixes #<issue>`.
4. Identify empty or inaccurate required sections.
5. Group changed files by source, tests, documentation, and build configuration.
6. Read every changed line.
7. Read enough unchanged code to learn the local pattern.
8. Search for similar implementations and direct callers.
Use repository documents as current contracts.
Read the direct document for each touched subsystem:

| Touched area | Repository document |
|---|---|
| Eager operations and NKI wrappers | `docs/AI-python-ops.md` |
| `torch.compile` backend | `docs/AI-dynamo-backend.md` |
| C++ extension and dispatcher | `docs/AI-csrc.md` |
| Runtime and package APIs | `docs/AI-runtime-api.md` |
| Distributed execution and DTensor | `docs/AI-distributed.md` |
| Compilation and NEFF cache | `docs/AI-compilation-kernels.md` |
| Tests and test drivers | `docs/AI-testing.md` |
| Build tools | `docs/AI-build-infra.md` |
Use source behavior as evidence when source and documentation differ.
Report a documentation mismatch when it can mislead a contributor or user.
Refer to `torch-neuronx-docs` only when broader subsystem context is necessary.

### 2. Review each changed line
Apply the [review checklist](references/review-checklist.md).
Focus on defects that automated checks cannot prove.
Do not report formatting, import order, or generated boilerplate.
Do not repeat a test failure that an automated check already reports.
Investigate these areas when they apply:
- Correctness and error handling.
- Existing registration and dispatcher patterns.
- Cast, layout, stream, and cache behavior.
- Public API and documentation changes.
- Test existence, placement, and quality.
- Security and subprocess handling.
- Shared state and compilation safety.
- Obvious performance regressions.
Record each candidate in this form:

```text
file:line | defect | consequence | evidence | requested action
```

Cite a changed line for every candidate.
Use an unchanged line only as supporting evidence.
Do not guess about a contract.
Return `BLOCKED` when required source or contract evidence is unavailable.

### 3. Check backward compatibility
Apply the [backward-compatibility guide](references/bc-guidelines.md).
Identify every changed user-visible behavior.
Check these public surfaces:
- Documented `torch_neuronx` symbols.
- The `torch.neuron.*` device module.
- The `"neuron"` device behavior.
- `torch.compile(backend="neuron")` behavior.
- Documented environment variables.
- On-disk compiled artifacts and cache keys.
- Documented public bindings that are implemented in `_C`.
Search public documentation, examples, and callers before you classify a binding change as a compatibility break.
The private `_C` extension ships in the same wheel as its Python callers.
Treat an internal `_C` caller as implementation coupling, not by itself as a public ABI promise.
Compare old and new signatures, defaults, return values, and exceptions.
Use `NEEDS_DISCUSSION` for an intentional break that needs an owner decision.

### 4. Consolidate findings
Consolidate all candidates before drafting the report.
1. Merge candidates with the same root cause.
2. Merge candidates that one code change can fix.
3. Merge duplicate claims at the same line.
4. Keep two claims at one line only when they are independent.
5. Assign each surviving finding to one report section.
Classify a finding as blocking when it can cause:
- Incorrect results or unsafe behavior.
- A backward-compatibility break.
- A security defect.
- A deadlock, race, or stale compiled artifact.
- Missing tests for new behavior or a bug fix.
- An unverifiable required contract.
Classify other actionable defects as non-blocking.
Do not include praise or style preferences.

### 5. Fact-check findings
Fact-check every consolidated finding in a separate pass.
A fresh-context reviewer can do this check.
Use a new session or subagent only when it is available.
Do not require delegation.
For each finding:
1. Re-read the cited line and its surrounding code.
2. Confirm that the cited line is in the reviewed diff.
3. Re-read the stated contract.
4. Confirm the consequence.
5. Confirm that the requested action still applies.
6. Classify the result as `valid`, `invalid`, or `needs_rewording`.
Drop each invalid finding.
Correct each finding that needs rewording.
Return `BLOCKED` instead of a speculative finding.

### 6. Draft the report
Use the exact section order in [Output](#output).
Omit empty finding lists only where the template permits it.
Do not repeat one finding in multiple sections.
Use concise file and line citations.
State the consequence before the requested action.

## Verification
Verify review coverage before you issue a verdict.
- Confirm that you read every changed file and line.
- Confirm that each finding cites a changed line.
- Confirm that each finding has direct evidence.
- Confirm that each finding passed the fact-check.
- Confirm that each changed public behavior has a compatibility classification.
- Confirm that the test evidence matches the changed behavior.
- Confirm that no finding only repeats an automated check.
- Confirm that the report contains all required output sections.
Use read-only checks in the reviewed checkout.
Run `git diff --check "$UPSTREAM/main"...HEAD` for committed branch scope.
Run `git diff --check "$(git merge-base "$UPSTREAM/main" HEAD)"` when working-tree changes are in scope.
Review existing continuous-integration evidence from the exact revision.
Do not run formatters, builds, or tests in the reviewed checkout during the report-only flow.
Do not run a build only to complete a report template.
If a required gate can modify files, ask for explicit user approval before running it.
Run an approved mutating gate only in a disposable worktree.
Record `git status --short` before and after the gate.
Do not start the gate when the disposable worktree is not clean.
Treat an unexpected post-gate change as test evidence and do not copy it into the reviewed checkout.
Record each command, revision, and result in the Tests section.

## Failure handling
Return `BLOCKED` when required evidence or verification is unavailable.
Stop the review at the blocked step.
Keep the evidence and verified findings collected before the block.
Never convert a blocked review into approval.
Use one of these blocker codes:

| Code | Condition |
|---|---|
| `PR_FETCH_FAILED` | The pull request metadata or diff is unavailable. |
| `EMPTY_DIFF` | The selected target has no changed lines. |
| `MISSING_REFERENCE` | `"$UPSTREAM/main"`, base-freshness evidence, the merge base, source, or a required contract is unavailable. |
| `HARDWARE_UNAVAILABLE` | A required runtime test needs Neuron hardware that is unavailable. |
| `BUILD_FAILED` | A required build failed or could not start. |
| `CREDENTIAL_UNAVAILABLE` | A required read or test needs a credential that is unavailable. |
For a blocked result, report:
1. The blocker code.
2. The completed review scope.
3. The missing evidence.
4. The exact command that could not run.
5. The exact command to run after the environment is available.
6. Any unverified finding that depends on the missing evidence.
Do not expose credential values in the report.
Do not infer a test result from an earlier run or a different revision.
When required runtime verification cannot run without Trainium, return `BLOCKED` with `HARDWARE_UNAVAILABLE`.
Give these exact resume commands to a contributor or maintainer who has a Trainium instance:

```bash
uv pip install -e ".[dev]"
./tools/build
./tools/run-test <TEST_PATH>
```

State that hardware CI checks can stay pending until a maintainer starts them.

## Output
Return this structured Markdown report.
Keep all six sections in this order.
Use `None.` for an empty Blocking or Non-blocking section.

```markdown
## Review: <PR NUMBER OR BRANCH> - <TITLE OR PURPOSE>

### Summary
<One sentence that describes the change and one sentence that states the review scope.>

### Blocking
- `<FILE>:<LINE>` - <Defect and consequence. Requested action.>

### Non-blocking
- `<FILE>:<LINE>` - <Actionable defect and consequence. Requested action.>

### Tests
- Evidence reviewed: <commands and results from this revision, or "None provided.">
- Required but not run: <commands and reason, or "None.">
- Coverage gaps: <missing cases, or "None.">

### Backward-compatibility risk
<Risk level: none, low, medium, high, or unknown. State the changed surface and evidence.>

### Verdict
**APPROVE | REQUEST_CHANGES | NEEDS_DISCUSSION | BLOCKED**
<One sentence with the decision reason. For BLOCKED, include the blocker code and recovery command.>
```

Use these verdict rules:
- Use `APPROVE` when no verified problems remain.
- Use `REQUEST_CHANGES` when one or more blocking findings remain.
- Use `NEEDS_DISCUSSION` when a verified issue needs an owner or policy decision.
- Use `BLOCKED` when required evidence or required verification is unavailable.
A missing regression test for a bug fix requires `REQUEST_CHANGES`.
A missing test for new behavior requires `REQUEST_CHANGES`.
Do not use the verdict as a GitHub review action without user approval.
Show the exact report and intended action before any post.
