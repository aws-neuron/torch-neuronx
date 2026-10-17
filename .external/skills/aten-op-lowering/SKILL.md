---
name: aten-op-lowering
description: Implements and validates one native torch-neuronx lowering for one ATen operator. It covers reference analysis, dispatch, tests, registry changes, exact overload coverage, and deterministic acceptance. Use when an ATen operator uses CPU fallback or needs native Neuron support.
---
# Lower one ATen operator
Lower one operator family per invocation. Use upstream PyTorch behavior as the reference.
## Prerequisites
Run all commands from the torch-neuronx repository root. Read `README.md`, `CONTRIBUTING.md`, and `BAZEL_BUILD.md` first.

Set the target repository. Find the remote that points to it. An outside contributor normally uses `origin` for a fork and `upstream` for the target repository.
```bash
REPO="${REPO:-aws-neuron/torch-neuronx}"   # the repository that receives the pull request or issue
UPSTREAM_URL="${UPSTREAM_URL:-}"
if test -z "$UPSTREAM_URL"; then UPSTREAM_URL="https://github.com/aws-neuron/torch-neuronx.git"; fi
UPSTREAM="$(git remote -v | awk -v r="$REPO" '$2 ~ r"(\\.git)?$" && $3=="(fetch)" {print $1; exit}')"
# If empty: git remote add upstream "$UPSTREAM_URL" && UPSTREAM=upstream
git fetch "$UPSTREAM" main
```
Ask for approval before the network read. Use `"$UPSTREAM/main"` as the base for every diff, branch, worktree, and merge-base.

Follow `CONTRIBUTING.md` section "Setup Development Environment" for the build. Activate its virtual environment, then install the development extra required by the probe, verifier, and tests.
```bash
uv pip install -e ".[dev]"
```
Install the GitHub CLI (`gh`) and run `gh auth login` before using GitHub commands.

Use [Shell commands](references/commands.md#prepare-the-task) to create the quoted task environment, verify required paths and hardware, prepare both environments, install the upstream overlay, and create the unchanged base worktree. Complete those commands before the workflow.

Building torch-neuronx and running focused, upstream, coverage, and acceptance tests require a Trainium instance with the public Neuron SDK. If hardware is absent or busy, report `RESULT: BLOCKED` with `REASON: HARDWARE`. Include the exact resume commands from Failure handling. Never terminate another process. Never run `prerequisites.sh` on a non-Neuron host.

Hardware CI checks on a pull request can stay pending until a maintainer starts them (see the pull request template); if a check fails for an unrelated reason, comment on the pull request (see CONTRIBUTING.md). A pending hardware check does not require contributor action.

Treat setup, checkout, import, or build failure as `RESULT: BLOCKED` with `REASON: ENVIRONMENT`.

Read [Implementation lessons](references/lessons.md) before implementation.
Default to report-only work. Commit only when the user asks.
Show exact content and wait for explicit approval before each GitHub write. Push a branch only to the contributor's fork remote, `origin`. Open only a draft pull request against `"$REPO"`.
Never merge, mark ready, force-push, or push to `main`. Disclose AI assistance in the pull request description.
## Workflow
### 1. Confirm support and overloads
```bash
set -euo pipefail; source "agent_space/<task-slug>/env.sh"
./tools/list-implemented-ops | grep -i "$OP_TOKEN" || true
git grep -n "$PRIMARY_OP" -- torch_neuronx/python_ops tests || true
git grep -n "$OP_PACKET" -- torch_neuronx/neuron_dynamo_backend/decompositions.py || true
git log --oneline --all -- "*${OP_PACKET}*"
```
Stop if support is complete. List every schema.
Write one exact ATen name per line to `claimed_overloads.txt`. Put the primary operation first.
```bash
set -euo pipefail; source "agent_space/<task-slug>/env.sh"
python - <<PY
import torch
packet = getattr(torch.ops.aten, "$OP_PACKET")
print(packet.overloads())
for name in packet.overloads(): print(name, getattr(packet, name)._schema)
PY
printf '%s\n' "$PRIMARY_OP" 'aten::<operator>.out' > "$ARTIFACTS/claimed_overloads.txt"
test -s "$ARTIFACTS/claimed_overloads.txt"
```
### 2. Extract reference behavior
Read the schema, meta function, decomposition, native kernel, autograd rule, and tests.
```bash
set -euo pipefail; source "agent_space/<task-slug>/env.sh"
git -C "$PYTORCH_ROOT" grep -n "$OP_PACKET" -- aten torch tools test
git -C "$PYTORCH_ROOT" grep -n "$OP_TOKEN" -- tools/autograd/derivatives.yaml || true
```
Confirm defaults, errors, dtype promotion, layout, aliasing, and autograd with CPU experiments.
For a linear operator, recover its matrix from basis vectors in the calculation dtype.
```python
basis = torch.eye(input_size, dtype=torch.float32).reshape(input_size, *input_shape)
operator = reference(basis).reshape(input_size, -1).T
```
Build each one-dimensional factor with the same method. Compare the full matrix with `torch.kron` of the factors.
Compare backward with `operator.T`. Sweep representative sizes, scales, and flags. Write a CPU regression test for each discovered rule.
### 3. Probe autograd
```bash
set -euo pipefail; source "agent_space/<task-slug>/env.sh"
python .external/skills/aten-op-lowering/scripts/probe_op_autograd.py \
  --op "$PRIMARY_OP" --out "$ARTIFACTS/autograd_probe.json"
```
`none` needs no backward lowering.
`derived` uses differentiable primitives.
`dedicated_op` needs native coverage for each listed backward operator.
Resolve `unknown` from source before acceptance.
### 4. Implement the lowering
Read `torch_neuronx/python_ops/torch_mlir/ops/README.md`.
```bash
set -euo pipefail; source "agent_space/<task-slug>/env.sh"
git ls-files 'torch_neuronx/python_ops/torch_mlir/ops/*.py' 'tests/python_ops/*.py' \
  'tests/neuron_dynamo_backend/e2e/*.py'
git grep -n 'register_aten(' -- torch_neuronx/python_ops/torch_mlir/ops
git grep -n 'track_neuron_ops\|assert_op_runs_on_neuron' -- tests/python_ops
```
Use an existing family file when possible. Import a new family from `ops/__init__.py`.
Match adjacent registration patterns and metadata. Keep in-place handlers separate. Call the ATen operator from the handler.
Keep arithmetic values dynamic and graph structure static. Use output metadata, value preprocessing, and contiguity helpers when required.
Match PyTorch dtype checks, layout, aliasing, and errors. Share eager and compile math.
Implement each dedicated backward operator.
Search all uses before changing shared code.
Verify clean import registration.
```bash
set -euo pipefail; source "agent_space/<task-slug>/env.sh"
CLAIMED_OPS=$(paste -sd, "$ARTIFACTS/claimed_overloads.txt")
CLAIMED_OPS="$CLAIMED_OPS" uv run python - <<'PY'
import os
import torch_neuronx.python_ops.torch_mlir.ops  # noqa: F401
from torch_neuronx.python_ops.torch_mlir.registry import get_torch_mlir_function
missing = [op for op in os.environ["CLAIMED_OPS"].split(",") if get_torch_mlir_function(op) is None]
assert not missing, f"Clean import did not register: {missing}"
PY
uv run ./tools/build; ./tools/run-cpp-test
```
Run build and C++ tests only when native or generated code changed.
Establish the compile requirement from exact decomposition-table membership.
Write the evidence and an explicit `REQUIRE_COMPILE=0` or `1` to `env.sh` before coverage.
```bash
set -euo pipefail; source "agent_space/<task-slug>/env.sh"
REQUIRE_COMPILE=$(CLAIMED_OPS_FILE="$ARTIFACTS/claimed_overloads.txt" \
  COMPILE_EVIDENCE="$ARTIFACTS/compile-path-evidence.txt" python - <<'PY'
import os
from pathlib import Path
import torch
from torch_neuronx.neuron_dynamo_backend.decompositions import neuron_decompositions

def resolve(name):
    qualified = name.removeprefix("aten::")
    packet_name, _, overload = qualified.partition(".")
    return getattr(getattr(torch.ops.aten, packet_name), overload or "default")

names = Path(os.environ["CLAIMED_OPS_FILE"]).read_text().splitlines()
matched = [name for name in names if resolve(name) in neuron_decompositions]
Path(os.environ["COMPILE_EVIDENCE"]).write_text("\n".join(matched) + ("\n" if matched else ""))
print(1 if matched else 0)
PY
)
case "$REQUIRE_COMPILE" in 0|1) ;; *) exit 1 ;; esac
tmp=$(mktemp "$ARTIFACTS/env.sh.XXXXXX")
grep -v '^export REQUIRE_COMPILE=' "$ARTIFACTS/env.sh" > "$tmp"
printf 'export %s=%q\n' REQUIRE_COMPILE "$REQUIRE_COMPILE" >> "$tmp"
chmod 600 "$tmp"; mv "$tmp" "$ARTIFACTS/env.sh"
```
### 5. Add focused tests
Import tracking helpers from `tests.utils.neuron_test_utils`.
Wrap each device invocation in `with track_neuron_ops():`.
Synchronize before the native assertion.
Use this repository pattern:
```python
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

with track_neuron_ops():
    actual = torch.ops.aten.example.default(value_neuron)
    torch.neuron.synchronize()
    assert_op_runs_on_neuron("aten::example")
torch.testing.assert_close(actual.cpu(), expected_cpu)
```
Use CPU as the value reference.
Test exact overloads, edges, dtypes, promotion, broadcast, layout, aliases, mutation, forward, backward, and errors.
Confirm each regression test fails without the fix.
Capture every focused `tools/run-test` exit code under the evidence directory. Combine them so any nonzero code produces a nonzero unit result.
```bash
set -euo pipefail; source "agent_space/<task-slug>/env.sh"
printf '%s\n' "$FOCUSED_TEST" > "$ARTIFACTS/focused-tests.txt"
if test "$REQUIRE_COMPILE" -eq 1; then
  test -n "$COMPILE_TEST"; printf '%s\n' "$COMPILE_TEST" >> "$ARTIFACTS/focused-tests.txt"
fi
: > "$ARTIFACTS/focused-rc-files.txt"; index=0
while IFS= read -r test_file; do
  index=$((index + 1)); test -f "$test_file"
  log="$ARTIFACTS/focused-${index}.log"; rc_file="$ARTIFACTS/focused-${index}.rc"
  set +e; tools/run-test "$test_file" -q > "$log" 2>&1; rc=$?; set -e
  printf '%s\n' "$rc" > "$rc_file"; printf '%s\n' "$rc_file" >> "$ARTIFACTS/focused-rc-files.txt"
done < "$ARTIFACTS/focused-tests.txt"
test -s "$ARTIFACTS/focused-rc-files.txt"; unit_rc=0
while IFS= read -r rc_file; do
  test -s "$rc_file" || { printf 'Missing focused test exit code: %s\n' "$rc_file"; exit 1; }
  rc=$(cat "$rc_file"); case "$rc" in ''|*[!0-9]*) exit 1 ;; esac
  if test "$rc" -ne 0; then unit_rc=1; fi
done < "$ARTIFACTS/focused-rc-files.txt"
printf '%s\n' "$unit_rc" > "$ARTIFACTS/unit.rc"
```
### 6. Run upstream tests
Follow [Upstream collection and runner commands](references/commands.md#collect-and-run-upstream-tests). They capture every collection and runner exit code, merge only exact reports, and preserve candidate, baseline, and runxfail evidence. Review node IDs and remove similarly named operators. Never edit overlay tests.
### 7. Triage marks
Fix failures by default.
Compare each failure with the unchanged base.
Read [Platform limitations](references/platform-limitations.md) only when evidence suggests a limit.
Run changed marks normally and with `--runxfail`.
Remember that `--runxfail` disables xfail marks but not skip marks.
For a new skip, require an exact `skipped` runxfail outcome and a failed or errored unchanged baseline whose failure text matches the documented signature.
A matching signature does not grant acceptance.
Write each approved exception as `<exact-node-id>=<limitation-id>` in `accepted-limitations.txt`.
The verifier rejects unapproved, mismatched, dead, or baseline-regressing marks.
List every accepted skip or xfail in the draft pull request for human review.
Never broaden a mark to hide a defect.
Use `compiler-artifact-repro` for a reduced compiler defect.
Report `RESULT: BLOCKED` and `REASON: DECISION` when policy needs a decision.
### 8. Update coverage and compatibility data
Add every exact overload to `tests/vertical/internal/target_ops.py`.
Put routine builtin shapes in `tests/vertical/internal/neuron_shape_defaults_generated.py` inside `register_all_defaults`.
Do not put them in `neuron_shape_registry.py`.
That module loads the generated defaults at import time.
`refresh_cache_key_shapes.py` only regenerates its cache-key block.
Use it only for cache-key-derived shapes.
Add exact backward rows. Follow [Coverage commands](references/commands.md#run-coverage) to verify every shape registration and exact eager and required compile row.
Update `docs/torch_api_compatibility.md` from measured support.
Use `Native MLIR impl` for native registration.
### 9. Run deterministic acceptance
```bash
set -euo pipefail; source "agent_space/<task-slug>/env.sh"
python "$VERIFY" snapshot-registry --repo-root "$BASE_WORKTREE" --out "$ARTIFACTS/baseline-registry.json"
python "$VERIFY" snapshot-registry --repo-root "$CANDIDATE_ROOT" --out "$ARTIFACTS/candidate-registry.json"
```
Fail closed when the combined unit exit code is missing or invalid. Pass its file value to required `--unit-rc`. Load optional accepted limitations from their file.
```bash
set -euo pipefail; source "agent_space/<task-slug>/env.sh"
test -s "$ARTIFACTS/unit.rc" || { printf '%s\n' 'Missing combined focused test exit code.'; exit 1; }
UNIT_RC=$(cat "$ARTIFACTS/unit.rc"); case "$UNIT_RC" in ''|*[!0-9]*) exit 1 ;; esac
mapfile -t CLAIMED_OVERLOADS < "$ARTIFACTS/claimed_overloads.txt"; OVERLOAD_ARGS=()
for overload in "${CLAIMED_OVERLOADS[@]:1}"; do OVERLOAD_ARGS+=(--claimed-overload "$overload"); done
OPTIONAL_ARGS=()
if test -s "$ARTIFACTS/upstream-runxfail.json"; then
  OPTIONAL_ARGS+=(--upstream-report-runxfail "$ARTIFACTS/upstream-runxfail.json")
  OPTIONAL_ARGS+=(--runxfail-runner-rc "$ARTIFACTS/upstream-runxfail.rc")
fi
if test -s "$ARTIFACTS/accepted-limitations.txt"; then
  while IFS= read -r item; do OPTIONAL_ARGS+=(--accept-limitation "$item"); done < "$ARTIFACTS/accepted-limitations.txt"
fi
EAGER_CSV=$(find "$ARTIFACTS/coverage-eager" -maxdepth 1 -name 'target_ops_coverage_*.csv')
case "$REQUIRE_COMPILE" in 0|1) ;; *) exit 1 ;; esac
COMPILE_ARGS=()
if test "$REQUIRE_COMPILE" -eq 1; then
  COMPILE_CSV=$(find "$ARTIFACTS/coverage-compile" -maxdepth 1 -name 'target_ops_compile_coverage_*.csv')
  test -n "$COMPILE_CSV"
  COMPILE_ARGS=(--compile-coverage-csv "$COMPILE_CSV" --require-compile)
fi
python "$VERIFY" verify --op "$PRIMARY_OP" "${OVERLOAD_ARGS[@]}" \
  --upstream-report "$ARTIFACTS/upstream-candidate.json" --baseline-upstream-report "$ARTIFACTS/upstream-base.json" \
  --candidate-runner-rc "$ARTIFACTS/upstream-candidate.rc" \
  --baseline-runner-rc "$ARTIFACTS/upstream-base.rc" --op-nodeids "$ARTIFACTS/op_nodeids.txt" --baseline-registry "$ARTIFACTS/baseline-registry.json" \
  --candidate-registry "$ARTIFACTS/candidate-registry.json" \
  --limitations .external/skills/aten-op-lowering/references/platform-limitations.md \
  --coverage-csv "$EAGER_CSV" "${COMPILE_ARGS[@]}" \
  --autograd-probe "$ARTIFACTS/autograd_probe.json" --unit-rc "$(cat "$ARTIFACTS/unit.rc")" \
  --target-ops-file tests/vertical/internal/target_ops.py --out "$ARTIFACTS/verdict.json" "${OPTIONAL_ARGS[@]}"
```
Pass compile arguments if and only if `REQUIRE_COMPILE=1`.
The verifier checks the supplied runner codes, reports, node IDs, registry snapshots, limitations, coverage CSV files, autograd probe, and unit result.
It checks each claimed operator in the supplied target file and the repository target file.
It checks dedicated backward operators in the supplied coverage data.
It does not prove which commands produced the supplied files.

The agent must separately check clean registration, focused exit-code files, collection status, compile-path evidence, shapes, compatibility data, pre-commit, and the diff.
A passing verifier result does not replace these workflow checks.
The verifier JSON uses `verdict: BLOCKED` plus `reason: ENVIRONMENT` for detected runner or report infrastructure failures.
The workflow report uses `REASON: HARDWARE` or `REASON: DECISION` only for failures detected in earlier workflow steps.
Read every verdict check.
Fix failures and rerun their evidence commands.
### 10. Prepare a draft pull request plan
```bash
set -euo pipefail; source "agent_space/<task-slug>/env.sh"
pre-commit run --all-files; git diff --check; git status --short; git diff --stat; git diff
```
Read [Pull request body format](references/pr-body-format.md) now.
Write agent text to `agent_space/<task-slug>/pr-body.md`.
List every accepted limitation with its node ID, ID, signature, and reason.
Ask the human to review the list and write the reserved commentary.
Show the exact branch, title, and complete body.
Wait for approval before a push or draft pull request.
```bash
set -euo pipefail; source "agent_space/<task-slug>/env.sh"
test -z "$(git -C "$BASE_WORKTREE" status --short)"; git worktree remove "$BASE_WORKTREE"
```
## Verification
Before success, verify support state, clean registration, tracked device execution, focused exit-code files, zero combined unit result, collection statuses, exact upstream reports, registry changes, accepted limitations, shapes, coverage, compatibility data, pre-commit, diff checks, and verdict.
Record every command and exit status.
Do not report an unrun check as passed.
## Failure handling
Use only `RESULT: PASS`, `RESULT: FAIL`, or `RESULT: BLOCKED`.
For `RESULT: BLOCKED`, require `REASON: ENVIRONMENT`, `REASON: HARDWARE`, or `REASON: DECISION`.
Use `ENVIRONMENT` for missing setup, dependency, credential, checkout, build, or evidence access.
Use `HARDWARE` for missing, busy, or failed Neuron devices.
Use `DECISION` for a suspected upstream defect, new limitation, unsafe shared change, or destructive conflict.
Use `FAIL` for a defect, unapproved limitation, failed test, or incomplete evidence.
When a public bug issue is required, use the optional `.github/ISSUE_TEMPLATE/bug_report.yml` if present. Otherwise use [Bug issue fallback](references/bug-issue-format.md). Show the exact issue title and body, then wait for explicit approval before `gh issue create --repo "$REPO"`.
Report the failed command, output, completed work, omitted checks, and concrete resume commands.
For an environment block, report:
```text
RESULT: BLOCKED
REASON: ENVIRONMENT
Resume with:
  cd <candidate-repository>
  source agent_space/<task-slug>/env.sh
  source .venv/bin/activate
  uv pip install -e ".[dev]"
  ./tools/build
```
For a hardware block, report:
```text
RESULT: BLOCKED
REASON: HARDWARE
Resume with:
  cd <candidate-repository>
  source agent_space/<task-slug>/env.sh
  neuron-ls
  python3 .external/skills/aten-op-lowering/scripts/neuron_devices_free.py
  tools/run-test "$FOCUSED_TEST" -q
```
For a decision block, report:
```text
RESULT: BLOCKED
REASON: DECISION
Resume with:
  cd <candidate-repository>
  source agent_space/<task-slug>/env.sh
  cat .external/skills/aten-op-lowering/references/platform-limitations.md
  git diff -- tests/pytorch_tests_native/skip_registry
```
Never convert a block or failure into a broad skip.
## Output
Report one result for the requested operator family.
```text
RESULT: PASS | FAIL | BLOCKED
REASON: ENVIRONMENT | HARDWARE | DECISION  # Required only for BLOCKED.
Operator: aten::<operator>
Overloads: <exact claimed overloads>
Implementation: <files and overloads>
Reference: <source and CPU evidence>
Tests: <commands, exit-code files, and combined result>
Upstream: <collection, candidate, baseline, and runxfail status>
Coverage: <shape registration and exact eager and compile rows>
Registry: <added, removed, stale, dead, and accepted entries>
Accepted limitations: <node ID, ID, signature, human review, or none>
Documentation: <updated rows>
Acceptance: <verdict path and result>
Pull request: not opened; approval required
Resume: <concrete commands when blocked>
Open questions: <items or none>
```
Keep paths repository-relative.
State all omitted checks.
