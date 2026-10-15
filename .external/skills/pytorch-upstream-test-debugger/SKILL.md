---
name: pytorch-upstream-test-debugger
description: Groups failing or xfailing upstream PyTorch tests by error signature, reproduces one defect, fixes torch_neuronx source, verifies the result against a baseline, and drafts a PR body. Use when a pytest report or upstream test node ID shows a Neuron failure.
---

# Debug an upstream PyTorch test failure

Use one serial workflow for one defect.

Treat an xfail as a known failure.
Do not accept it as permanent test noise.

Fix product source under `torch_neuronx/`.
Never change an upstream test, a test patch, or a tolerance only to hide a defect.

Keep one error signature in one proposed PR.
Move a different signature to a separate investigation.

Read [Investigation guide](references/investigation.md) before you edit source.
Read [Verification and PR guide](references/verification-and-pr.md) before final verification.

## Prerequisites

Work from the `torch-neuronx` repository root.

Set the target repository and locate its remote:

```bash
REPO="${REPO:-aws-neuron/torch-neuronx}"   # the repository that receives the pull request or issue
UPSTREAM="$(git remote -v | awk -v r="$REPO" '$2 ~ r"(\\.git)?$" && $3=="(fetch)" {print $1; exit}')"
# If empty: git remote add upstream "https://github.com/aws-neuron/torch-neuronx.git" && UPSTREAM=upstream
git fetch "$UPSTREAM" main
```

For a fork checkout, `origin` must remain the contributor's fork.
If `UPSTREAM` is empty, run the optional add-remote command shown above and then fetch again.
Use `"$UPSTREAM/main"` as the base for branches, diffs, and merge-base checks.

Collect these inputs:

- A pytest JSON, CSV, or plain-text report, if available.
- One failing or xfailing pytest node ID, error regex, or both.
- A clean checkout with the project environment active.
- An actionable issue in `$REPO` that describes the defect.
- Neuron hardware for reproduction and final verification.

Follow `CONTRIBUTING.md` section "Setup Development Environment" for the build.
Install this skill's development dependencies in the active environment:

```bash
command -v uv || { echo 'BLOCKED: install uv and rerun this check'; exit 10; }
uv pip install -e ".[dev]"
```

Install the GitHub CLI (`gh`) with the operating system package manager and run `gh auth login` once.
Then verify authentication:

```bash
command -v gh || { echo 'BLOCKED: install the GitHub CLI and rerun this check'; exit 10; }
gh auth status
```

Verify the actionable issue before editing:

```bash
ISSUE_NUMBER=<issue-number>
gh issue view --repo "$REPO" "$ISSUE_NUMBER"
```

If no issue exists, use the [fallback bug issue template](references/fallback-bug-issue.md) when the repository issue form under `.github/` is absent.
Use the [fallback pull request template](references/fallback-pr-template.md) when the repository PR template under `.github/` is absent.
Show the exact issue content, disclose AI assistance, and wait for explicit approval before this write:

```bash
gh issue create --repo "$REPO" --title '<issue title>' --body-file agent_space/category_slug/issue_body.md
```

Set the skill path once:

```bash
SKILL_DIR=.external/skills/pytorch-upstream-test-debugger
```

Confirm the repository commands exist:

```bash
test -x tools/run-test
test -x tools/build
test -x tests/pytorch_tests_native/run_upstream_tests.sh
test -f tests/pytorch_tests_native/runner/run_upstream_tests.py
```

Confirm the required tools and active environment:

```bash
command -v git
command -v cc || command -v gcc
python3 -c 'import torch; import torch_neuronx; print(torch.__version__)'
python3 -c 'import torch; import torch_neuronx; assert torch.neuron.is_available()'
python3 -c 'import pytest, pytest_jsonreport, pytest_timeout'
```

Import `xdist` before a rerun that uses `-n`.
Import `pytest_forked` before a distributed rerun.
A missing import, C compiler, or valid temporary directory makes the rerun helper return `10`.
Return `BLOCKED` with the failed check and exact resume command.

Run the upstream driver help when you need its suite interface:

```bash
tests/pytorch_tests_native/run_upstream_tests.sh --help
```

Use `tests/pytorch_tests_native/runner/run_upstream_tests.py` only through its documented CLI.
Do not pass `--download-neff-cache` or `--refresh-neff-cache`.
Those options need a CI-provided storage location; a local run compiles every graph from cold.
Do not copy implementation details from the runner into a local command.

## Safety rules

Default to report-only.
Edit or commit only when the user asks.
Before any push, PR, issue, or comment, show the exact content and wait for explicit approval.
Use the disclosure, empty Human commentary placeholder, quoted agent content, and human confirmation from the PR guide.
The human must fill every first-person checkbox and affirm responsibility for every changed file.
Open only a draft PR after approval.
Never merge, mark ready, force-push, or push to `main`.
Disclose AI assistance and name the tool.

Create a fresh branch before a source edit.
Check both branch namespaces first:

```bash
BRANCH_NAME=fix_category_slug
git fetch "$UPSTREAM" main
git show-ref --verify --quiet "refs/heads/$BRANCH_NAME"
git ls-remote --heads origin "$BRANCH_NAME"
```

Exit zero from `show-ref` means the local branch exists.
Any output from `ls-remote` means the fork branch exists.
Stop on either collision and choose a new branch name.
Create the branch only after both checks show no collision:

```bash
BRANCH_NAME=fix_category_slug
git switch -c "$BRANCH_NAME" "$UPSTREAM/main"
```

Check open PR overlap in `$REPO` before editing.
This step requires `gh`:

```bash
SEARCH_TERM='node id or signature'
command -v gh
gh pr list --repo "$REPO" --state open --search "$SEARCH_TERM"
gh pr list --repo "$REPO" --state open --json number --jq '.[].number' | while read -r pr; do
  gh pr view --repo "$REPO" "$pr" --json number,title,url,files
done
```

Compare the returned file lists with the files this fix may touch.
Stop on ambiguous node-ID, signature, or touched-file overlap.
Ask the human which work should continue.
These commands are read-only.
If `gh` is missing, return `BLOCKED` and give `command -v gh && gh auth status` as the resume check.
Skip the overlap step only after the user explicitly accepts that risk.

Keep all temporary evidence under `agent_space/<category_slug>/`.
Keep that directory untracked.
Do not edit installed packages under `.venv/`.

## Workflow

### 1. Define one category

Choose a short category slug from the observed behavior.
Use lowercase ASCII letters, digits, and underscores.
Start it with a letter or underscore.
Use the same slug for all local evidence.
The RCA helper changes `-` to `_` only in generated Python identifiers.
It does not rename the workspace path.
Prefer underscores so paths and generated identifiers match.

Do not group tests by operation name alone.
Group tests by a normalized error signature.

If the user gave only a node ID, create a one-line selection and a fresh report.
Then select the signature from that report:

```bash
CATEGORY_SLUG=category_slug
PYTEST_NODE_ID='tests/path_to_test.py::test_name'
WORKSPACE="agent_space/$CATEGORY_SLUG"
mkdir -p "$WORKSPACE"
printf '%s\n' "$PYTEST_NODE_ID" > "$WORKSPACE/tests.txt"
bash "$SKILL_DIR/scripts/rerun_category.sh" \
  . \
  "$WORKSPACE/tests.txt" \
  "$WORKSPACE/rerun-discovery" \
  --timeout 600 || test "$?" -eq 1
python3 "$SKILL_DIR/scripts/select_category_tests.py" \
  "$WORKSPACE/rerun-discovery/report.json" \
  --pin-nodeid "$PYTEST_NODE_ID" \
  --out "$WORKSPACE/tests.txt"
```

The rerun helper uses the official environment lock and safe output handling.
It writes a valid redacted JSON report and signature sidecar.
Keep a one-test category when no other fresh failure shares its signature.

If the user gave a report, select and expand the category:

```bash
PYTEST_NODE_ID='tests/path_to_test.py::test_name'
python3 "$SKILL_DIR/scripts/select_category_tests.py" report.json \
  --pin-nodeid "$PYTEST_NODE_ID" \
  --out agent_space/category_slug/tests.txt
```

Use `--pin-error '<regex>'` instead when the signature is the stable pin.
Use both options when they reduce ambiguity.

Read these files before you continue:

- `agent_space/<category_slug>/tests.txt`
- `agent_space/<category_slug>/tests.txt.signatures.txt`

Split the selection if it contains unrelated signatures.
Do not fix two defects together.

### 2. Create an RCA workspace

Generate local notes and test seeds:

```bash
python3 "$SKILL_DIR/scripts/rca_workspace.py" \
  agent_space/category_slug/tests.txt \
  agent_space/category_slug \
  --category category_slug \
  --signatures agent_space/category_slug/tests.txt.signatures.txt
```

The generator does not require PyTorch or hardware.
It refuses to replace generated files unless `--force` is explicit.
`--force` replaces only generated regular files and never overwrites `notes.md`.
It accepts input files already at their workspace destinations.
The generated tests require the project environment when executed.

Record each hypothesis in `notes.md`.
Record the evidence that rejects or supports it.

### 3. Capture the baseline

Run the complete selected category before any fix:

```bash
bash "$SKILL_DIR/scripts/rerun_category.sh" \
  . \
  agent_space/category_slug/tests.txt \
  agent_space/category_slug/rerun-baseline \
  --timeout 600
```

A pytest failure returns pytest's nonzero status.
This is expected for a reproduced defect.

Require these baseline files:

- `rerun-baseline/pytest.log`
- `rerun-baseline/report.json`
- `rerun-baseline/summary.txt`

Compare the baseline signature with the input report.
Stop if the defect does not reproduce.
Do not change source for a stale or different failure.

Use the same worker and timeout options in baseline and verification.
Use one process for a race, hang, or one-test reproduction.

### 4. Locate the root cause

Start at the first product frame in the traceback.
Trace callers and registrations until you find where the wrong behavior starts.

Inspect the applicable source under `torch_neuronx/`.
Inspect nearby tests only to learn the expected style.
Inspect `tests/pytorch_tests_native/neuron_test_registry.py` for the xfail reason.
Inspect the matching file under `tests/pytorch_tests_native/skip_registry/` when applicable.

Classify the cause before you edit:

- A product-source defect that can be fixed here.
- A project environment or upstream-overlay mismatch.
- A compiler defect that needs `compiler-artifact-repro`.
- A documented platform limitation that must stay visible as an xfail.

Write one root-cause statement in `notes.md`.
Name the source symbol and the incorrect behavior.
Do not edit until that statement is specific.

### 5. Add or select a regression fence

Reuse the pinned upstream node ID when it is focused and fast.
Add a repository regression test when the upstream test is broad or slow.

A new regression test must fail before the fix.
It must fail with the selected signature.

Put a new product regression test in the applicable repository test directory.
Do not edit cloned upstream test content.

Run the narrow fence directly:

```bash
tools/run-test tests/path_to_test.py::test_name -xvs
```

If the user requested commits, commit the failing fence separately.
Run the repository formatting checks before the commit.

### 6. Fix the product source

Make the smallest source change that corrects the root cause.
Prefer the highest product layer that can express the fix.
Avoid unrelated cleanup.

Run only the fence during the inner loop.
Stop after two failed attempts with the same hypothesis.
Re-read the source and form a new hypothesis.

Run every affected Python test file after the fence passes:

```bash
tools/run-test tests/path_to_test.py
```

Build after a C++ or generated-code change.
When `torch_neuronx/csrc` changes, run its specific C++ target too:

```bash
tools/build
tools/run-cpp-test //tests/csrc:TargetName
```

Run the fence again after the build.
A Python import cannot load an unbuilt native source change.

If the user requested commits, commit the source fix separately.
Do not include xfail cleanup before final verification.

### 7. Verify against the baseline

Run the exact selected set with the baseline report:

```bash
bash "$SKILL_DIR/scripts/verify_fixed.sh" \
  . \
  agent_space/category_slug/tests.txt \
  agent_space/category_slug/rerun-verify \
  agent_space/category_slug/rerun-baseline/report.json \
  --timeout 600
```

Read `agent_space/<category_slug>/rerun-verify/verify.md`.
Do not rely only on the command exit status.

Apply these acceptance rules:

1. No test still has the selected error signature.
2. No previously passing test regressed.
3. Each claimed fixed test now passes.
4. The baseline and current reports contain the same node IDs.
5. Any different remaining signature has a separate scope and owner.

After a comparable rerun, the verifier returns `2` for any remaining failure.
Before comparison, it can also return `2` for invalid arguments, invalid input, or an interrupted rerun.
A changed signature is not automatically a fixed test.
Confirm that it is a separate defect before you accept the source fix.

Remove xfail entries only for tests in the verified `fixed` bucket.
Leave every unresolved or differently failing test registered.

If the user requested commits, commit verified xfail cleanup separately.
Do not create an empty cleanup commit.

### 8. Draft the PR body

Draft `agent_space/<category_slug>/pr_body.md` from the RCA scaffold.
Do not push or open a PR.
Keep the disclosure, empty Human commentary, human confirmation, unchecked first-person boxes, and quoted agent body.
After the human fills their section, require responsibility confirmation and explicit approval of the exact repository, branch, fork head, title, body, and action.

Keep one defect in the quoted draft.
Include the tests and signature, root cause symbol, regression fence, minimal fix, exact commands and results, verification buckets, distinct follow-ups, actionable issue, repository or fallback template, and AI disclosure.
Use the template in [Verification and PR guide](references/verification-and-pr.md).

## Verification

Run the host-only helper tests from the repository root:

```bash
python3 -m pytest -o addopts='' -p no:cacheprovider \
  .external/skills/pytorch-upstream-test-debugger/tests -q
```

Check shell syntax:

```bash
bash -n "$SKILL_DIR/scripts/rerun_category.sh"
bash -n "$SKILL_DIR/scripts/verify_fixed.sh"
```

Check helper interfaces:

```bash
python3 "$SKILL_DIR/scripts/select_category_tests.py" --help
python3 "$SKILL_DIR/scripts/rca_workspace.py" --help
python3 "$SKILL_DIR/scripts/safe_output.py" --help
python3 "$SKILL_DIR/scripts/redact.py" --help
bash "$SKILL_DIR/scripts/rerun_category.sh" --help
bash "$SKILL_DIR/scripts/verify_fixed.sh" --help
```

Run the narrow product test and the category verification on Neuron hardware.
Run the exact repository gates for the changed files:

```bash
uv run pre-commit run --files torch_neuronx/path_to_fix.py tests/path_to_test.py
tools/run-test tests/path_to_test.py
tools/run-cpp-test //tests/csrc:TargetName
```

Run `tools/run-test` once for every affected Python test file.
Run `tools/run-cpp-test <target>` when `torch_neuronx/csrc` changes.
Run pre-commit only on the agent's own branch because it can modify files.
Inspect its rewrites and rerun the same pre-commit command until it passes without rewrites.
Return `BLOCKED` when any required gate cannot run.

A complete result has host tests and device evidence.
Do not claim device verification from the host-only helper tests.

## Failure handling

Return `BLOCKED` when Neuron hardware, a build, or a credential is missing.
Treat rerun-helper exit `10` as a missing runtime, build, or temporary-directory prerequisite.
Stop source and registry cleanup at that point.

Report:

- The completed host-only checks.
- The missing prerequisite.
- The last successful workflow step.
- The exact unrun commands.
- The files that contain partial evidence.

Use this form:

```text
BLOCKED
Reason: <missing hardware, build dependency, or credential>
Completed: <commands and results>
Not run: <exact command>
Resume from: <workflow step and evidence path>
```

Use `BLOCKED` for an unresolved overlay mismatch or build failure too.
Do not label an environment failure as a product defect.

Use `NO_REPRO` when the baseline does not match the input signature.
Use `REGRESSED` when verification introduces a new failure.

Do not weaken a test to escape a blocked result.
Do not remove an xfail without passing device evidence.

Stop and report a suspected security issue.
Do not include exploit details in public output.
Keep sensitive evidence local and ask the user how to proceed.

Stop when a git conflict would require a destructive operation.
Report the conflicting paths and the non-destructive options.
Do not discard, overwrite, or reset existing work to resolve the conflict.

## Output

Return a concise report with this order:

1. Category and selected node IDs.
2. Reproduction signature and baseline result.
3. Root cause in `torch_neuronx/`.
4. Changed source and regression-test files.
5. Verification buckets and exact commands.
6. Remaining signatures and separate follow-ups.
7. Draft PR body path.
8. `DONE`, `BLOCKED`, `NO_REPRO`, or `REGRESSED`.

If no source edit was requested, return the RCA and proposed fix only.
If no commit was requested, leave all changes uncommitted.
If no GitHub write was approved, stop with the PR body draft.
