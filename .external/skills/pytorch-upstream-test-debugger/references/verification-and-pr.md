# Verification and PR guide

Use this guide for baseline comparison, acceptance, and a draft PR body.

## Runtime prerequisites

Require `git` for branch and source-history checks.
Require `uv` for environment setup.
Require `cc` or `gcc` to build `env_lock.so`.
Require `pytest`, `torch_neuronx`, `pytest-json-report`, and `pytest-timeout` for every rerun.
Require `pytest-xdist` when a rerun uses `-n`.
Require `pytest-forked` for distributed tests.
Install the declared development extra after following `CONTRIBUTING.md` section "Setup Development Environment":

```bash
uv pip install -e ".[dev]"
```

A missing rerun runtime, build dependency, or valid temporary directory returns `10` and means `BLOCKED`.
Report the failed check and the exact command that resumes the workflow.

The read-only issue and PR-overlap steps also require the GitHub CLI (`gh`).
Install it with the operating system package manager and run `gh auth login` once.
If it is missing or unauthenticated, return `BLOCKED` with `command -v gh && gh auth status` as the resume check.
Skip the overlap step only when the user explicitly accepts the overlap risk.

## Helper contracts

All helpers use repository-relative inputs.
They make no network calls.

### `select_category_tests.py`

This helper reads pytest JSON, CSV, or plain text.
It rejects duplicate node IDs in every supported report format.
It includes failed and xfailed outcomes.
It includes setup errors only with `--include-error-outcome`.

It expands an exact node ID or error regex to matching normalized signatures.
It writes a node-id file and a signature sidecar as one descriptor-anchored transaction.
It redacts sensitive diagnostic values before signature normalization and sidecar rendering.

Exit codes:

- `0`: It selected at least one test.
- `2`: The pin is absent or matches no failure.
- `3`: The report or pin syntax is invalid or unsupported.
- `4`: An input or output operation failed.

### `rca_workspace.py`

This helper creates notes, copied node IDs, optional signatures, test seeds, and a reproduction stub.
It uses only the Python standard library.
It rejects symlink destinations and existing generated files by default.
Use `--force` only after inspecting the workspace.
That flag replaces generated regular files in one rollback-safe transaction.
It never overwrites `notes.md`.
The helper validates and redacts structured signature fields before it writes `signatures.txt`.
An input `tests.txt` or signatures file may already be the matching workspace file.

Exit codes:

- `0`: It wrote all scaffold files.
- `2`: An input, seed limit, or workspace operation is invalid.

Generated files do not prove a device failure.
Run the generated reproduction in the project environment.

### `rerun_category.sh`

This helper runs exact node IDs with `--runxfail`.
It compiles and preloads the environment lock used by the official runner.
It uses `--forked` without `--timeout` for distributed targets.
It uses `--timeout` without `--forked` for other targets.
It rejects empty, duplicate, malformed, option-like, overlong, or excessive node-ID selections.
It passes validated node IDs after pytest's `--` option separator.
It rejects a symlink output directory and every symlink or non-regular artifact destination.
One `safe_output.py run` process opens and pins every output ancestor before pytest starts.
Pytest writes `pytest.log`, `report.json`, and `summary.txt` only in a private directory outside the output tree.
The same process reopens and verifies every ancestor before it installs through the held output descriptor.
It fails closed if any ancestor changed.
It redacts every text artifact and recursively redacts every JSON string value while keeping `report.json` valid.
It installs the complete set only after exact node-ID and status validation.

Normal pytest completion accepts only these status/report combinations:

- Status `0`: The fresh report contains no `failed` or `error` outcome.
- Status `1`: The fresh report contains at least one `failed` or `error` outcome.

Either inverse combination is invalid and returns `11` without installing fresh artifacts.
A status above `1` is propagated only as a non-comparable pytest runtime status after report validation.

Exit codes:

- `0`: Pytest passed and report generation succeeded.
- `1`: Pytest completed with a valid report that contains a failure.
- `2`: An argument or input is invalid, or pytest returned status `2` after fresh-report validation.
- `3` through `5`: Pytest returned that status after fresh-report validation.
- `10`: A required runtime, build, or temporary-directory prerequisite is missing.
- `11`: Helper output, I/O, or report validation failed.

A reproduced product failure usually returns `1`.
Do not convert that status to success.

### `verify_fixed.sh`

This helper reruns the exact selection.
It compares current results and signatures with the baseline.
It requires equal node-id sets.
It rejects a symlink output directory and every symlink or non-regular artifact destination.
One outer `safe_output.py run` process pins every output ancestor before the rerun starts.
The rerun and comparison write only to private directories outside the output tree.
It redacts both baseline and current diagnostics before signature normalization.
The outer process revalidates every ancestor, redacts all four artifacts, and installs them as one set through its held descriptor.
It fails closed if any output ancestor changed.

Exit codes:

- `0`: All selected tests are non-failing.
- `1`: A previously passing test regressed.
- `2`: Arguments or input are invalid, pytest returned status `2` before comparison, or at least one baseline failure still fails.
- `3` through `5`: Pytest returned that status before a comparable result was available.
- `10`: A rerun runtime, build, or temporary-directory prerequisite is missing.
- `11`: A report is invalid, node-id sets differ, or a helper I/O operation failed.

The report separates these buckets:

- `fixed`
- `regressed`
- `still_bad_same_signature`
- `still_bad_different_signature`
- `unchanged`

A changed signature remains a failure.
Decide separately whether it belongs to a new one-bug scope.

## Baseline discipline

Use the same node-id file for baseline and current runs.
Use the same worker count and timeout.
Use the same project commit except for the proposed fix and regression fence.

Keep the raw logs.
Do not compare only summary counts.

Require this evidence before xfail cleanup:

1. The baseline test fails or xfails with the selected signature.
2. The current test passes with `--runxfail`.
3. The verifier puts that node ID in `fixed`.
4. No selected test moved from pass to failure.

Remove only matching registry entries.
Do not remove a group member that was not verified.

## Different remaining signatures

A source fix can remove the selected signature and reveal another defect.
The verifier still returns `2` because the test does not pass.

Do not call that test fixed.
Leave its xfail entry in place.
Record the new signature as a separate follow-up.

Accept the current one-bug source change only when:

- The selected signature is absent from the complete category.
- The new signature has a clearly different root cause.
- No previously passing test regressed.
- The proposed PR claims only tests that now pass.

Otherwise return to the source fix.

## Commit structure

Commit only when the user asks.
Use additive history.

When all three changes exist, use this order:

1. A regression-fence commit after the test fails as expected.
2. A source-fix commit after the fence and affected file pass.
3. An xfail-cleanup commit after baseline verification passes.

Skip the first commit when an existing test is the fence.
Skip the third commit when no verified xfail entry exists.
Do not create placeholder commits.

Before each requested commit, run these applicable gates:

```bash
uv run pre-commit run --files torch_neuronx/path_to_fix.py tests/path_to_test.py
tools/run-test tests/path_to_test.py
tools/run-cpp-test //tests/csrc:TargetName
```

Run the Python command once per affected Python test file.
Run the C++ command for each affected target when `torch_neuronx/csrc` changes.
Pre-commit can modify files.
Run it only on the agent's own branch, inspect any rewrites, and rerun it until clean.
Return `BLOCKED` with exact unrun commands when any required gate cannot run.
Verify that each commit exists before the next step.
Never amend a pushed commit.
Never rebase and force-push a review branch.

## GitHub boundary

Default to a local report and local PR-body draft.
Set `REPO` to the repository that receives the issue or pull request.
Use the remote in `UPSTREAM` as the `main` base, and push only to the contributor's `origin` fork.

Before any GitHub write:

1. Show the exact branch, title, body, comment, or issue content.
2. State which repository and remote action will occur.
3. Wait for explicit user approval.

Require an actionable issue in `$REPO` before a pull request.
Verify an existing issue with `gh issue view --repo "$REPO" <issue-number>`.
If none exists, draft the issue from the optional repository issue form under `.github/` when present, or use [the fallback bug issue template](fallback-bug-issue.md).
Show the exact issue body and wait for explicit approval before `gh issue create --repo "$REPO"`.

For a pull request, stop before any push or PR creation.
Require the human to replace the empty Human commentary placeholder with their own explanation of relevance.
The agent must not fill or suggest that commentary.
Leave all first-person template checkboxes unchecked for the human.
Require the human to affirm: "I read and understand every changed file, and I take responsibility for these changes."
Use the same disclosure, human-commentary placeholder, and quote or code block for any AI-generated issue or comment.
Replace `<AI tool name>` with the actual tool name before showing the exact final content.
Show the exact target repository, branch, base, fork head, title, and body after the human edits them.
Wait for explicit approval of that exact content and action.

After explicit approval of the exact branch, title, body, repository, fork head, and write action, run:

```bash
FORK_OWNER=<fork-owner>
# Run each write only after explicit approval of the exact repository, branch, head, title, and body.
git push -u origin "$BRANCH_NAME"
# Run this write only after the same explicit approval of the exact action and content.
gh pr create --repo "$REPO" --draft --base main \
  --head "$FORK_OWNER:$BRANCH_NAME" --title '<title>' \
  --body-file agent_space/category_slug/pr_body.md
```

Never merge it or mark it ready.
Never push to `main`.
Disclose AI assistance, name the tool, and quote all agent-generated body content.

## Draft PR body template

Copy the optional repository pull request template from `.github/PULL_REQUEST_TEMPLATE.md` when present in this checkout.
Otherwise use [the bundled fallback pull request template](fallback-pr-template.md).
Use the optional repository bug issue form under `.github/ISSUE_TEMPLATE/` when present.
Otherwise use [the bundled fallback bug issue template](fallback-bug-issue.md).
Fill each applicable field.
Do not invent test results.
State that hardware CI checks on a pull request can stay pending until a maintainer starts them.

````markdown
AI-assisted content disclosure: The quoted pull request draft below was generated with <AI tool name>.

## Human commentary

<!-- REQUIRED: Explain why the quoted change is relevant. The agent must not fill this section. -->

## Human confirmation

- [ ] I read and understand every changed file.
- [ ] I take responsibility for these changes.

## AI-generated pull request body

> ## Problem
>
> <Tests and the shared error signature. State that xfails are tracked failures when applicable.>
>
> ## Related Issue
>
> Fixes #<public issue number>
>
> ## Root cause
>
> <Name the product source symbol and explain the incorrect behavior.>
>
> ## Fix
>
> <Describe the smallest source change and why it fixes this defect.>
>
> ## Verification
>
> Baseline:
>
> ```bash
> <exact baseline command>
> ```
>
> Current:
>
> ```bash
> <exact verification command>
> ```
>
> Results: <fixed count> fixed, <same-signature count> same-signature failures, <different-signature count> different-signature failures, <regression count> regressions.
>
> ## Follow-ups
>
> <Separate bugs only. Omit this section when none exist.>
>
> ---
>
> <copy and complete the repository pull request template, or the bundled fallback>
````

Save the result here:

```text
agent_space/<category_slug>/pr_body.md
```

Do not put local absolute paths in the body.
Use repository-relative paths and exact commands.

## Final evidence checklist

- The report selection contains one root cause.
- The baseline reproduces the selected signature.
- A test fences the defect before the fix.
- The root cause names a product source symbol.
- The fix changes product source, not upstream test behavior.
- The narrow fence passes.
- The affected test file passes.
- Native edits have a successful build.
- Baseline and current node-id sets match.
- The selected signature is absent.
- No passing test regressed.
- Xfail cleanup matches only verified passing tests.
- The PR body contains exact commands and actual results.
- The actionable public issue exists in `$REPO` and the PR body uses `Fixes #<issue>`.
- The PR body discloses AI assistance.
- Open PRs were checked by node ID or signature and by touched files.
- The human filled every first-person checkbox themselves.
- The human affirmed understanding and responsibility before any push or draft PR.
- No remote write occurred without explicit approval.
