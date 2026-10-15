# ATen lowering commands

Use these commands from the repository root. Replace each angle-bracket value. Each shell block is independent.

## Prepare the task

Use the merge base of the target repository's `main` branch and `HEAD`. Do not choose an arbitrary base.
Find the remote that points at `$REPO`. If it is absent, add `upstream` as shown below. Ask the user before `git fetch` because it is a network read.
Create the task environment before any other command block.
```bash
set -euo pipefail
REPO="${REPO:-aws-neuron/torch-neuronx}"
UPSTREAM_URL="${UPSTREAM_URL:-}"
if test -z "$UPSTREAM_URL"; then UPSTREAM_URL="https://github.com/aws-neuron/torch-neuronx.git"; fi
UPSTREAM="$(git remote -v | awk -v r="$REPO" '$2 ~ r"(\\.git)?$" && $3=="(fetch)" {print $1; exit}')"
if test -z "$UPSTREAM"; then
  git remote add upstream "$UPSTREAM_URL"
  UPSTREAM=upstream
fi
git fetch "$UPSTREAM" main
if ! git rev-parse --verify "$UPSTREAM/main" >/dev/null; then
  cat >&2 <<EOF
RESULT: BLOCKED
REASON: ENVIRONMENT
$UPSTREAM/main is absent. Resume with:
git fetch "$UPSTREAM" main
git rev-parse --verify "$UPSTREAM/main"
BASE_COMMIT="\$(git merge-base "$UPSTREAM/main" HEAD)"
EOF
  exit 2
fi
BASE_COMMIT="$(git merge-base "$UPSTREAM/main" HEAD)"
test -n "$BASE_COMMIT"
OP_TOKEN='aten::<operator>'
OP_PATTERN='^aten::[A-Za-z_]'
OP_PATTERN+='[A-Za-z0-9_]*(\.[A-Za-z_]'
OP_PATTERN+='[A-Za-z0-9_]*)?$'
[[ "$OP_TOKEN" =~ $OP_PATTERN ]] || {
  printf 'Invalid ATen operator token: %s\n' "$OP_TOKEN" >&2; exit 2;
}
OP_QUALIFIED="${OP_TOKEN#aten::}"; OP_PACKET="${OP_QUALIFIED%%.*}"
TASK_SLUG="aten-${OP_QUALIFIED//./-}"; CANDIDATE_ROOT="$PWD"
ARTIFACTS="$CANDIDATE_ROOT/agent_space/$TASK_SLUG"
BASE_WORKTREE="$(dirname "$CANDIDATE_ROOT")/${TASK_SLUG}-base"
PRIMARY_OP="$OP_TOKEN"; OP="$PRIMARY_OP"
PYTORCH_ROOT="$(dirname "$CANDIDATE_ROOT")/pytorch-${TASK_SLUG}"
VERIFY="$CANDIDATE_ROOT/.external/skills/aten-op-lowering/scripts/verify_op_acceptance.py"
FOCUSED_TEST='tests/python_ops/<family-test>.py'
COMPILE_TEST='tests/neuron_dynamo_backend/e2e/<family-test>.py'
VIRTUAL_ENV="$CANDIDATE_ROOT/.venv"; PATH="$VIRTUAL_ENV/bin:$PATH"
mkdir -p "$ARTIFACTS"
for NAME in REPO UPSTREAM_URL UPSTREAM TASK_SLUG CANDIDATE_ROOT ARTIFACTS BASE_COMMIT BASE_WORKTREE PRIMARY_OP OP \
  OP_TOKEN OP_QUALIFIED OP_PACKET PYTORCH_ROOT VERIFY FOCUSED_TEST COMPILE_TEST VIRTUAL_ENV PATH; do
  VALUE="${!NAME}"
  printf 'export %s=%q\n' "$NAME" "$VALUE"
done > "$ARTIFACTS/env.sh"
chmod 600 "$ARTIFACTS/env.sh"
```
Set `<task-slug>` to the value above. Start every later shell block from the repository root. Each block reloads all required state.
```bash
set -euo pipefail; source "agent_space/<task-slug>/env.sh"
git status --short; git branch --show-current
git ls-files README.md CONTRIBUTING.md BAZEL_BUILD.md pyproject.toml \
  tools/run-test tools/build tests/pytorch_tests_native/run_upstream_tests.sh \
  tests/pytorch_tests_native/clone_device_agnostic_tests.py \
  tests/vertical/internal/target_ops.py \
  tests/vertical/internal/neuron_shape_defaults_generated.py \
  tests/vertical/internal/neuron_shape_registry.py docs/torch_api_compatibility.md
```
Stop if a required path is missing.
Detect Neuron hardware before any setup action.
```bash
set -euo pipefail; source "agent_space/<task-slug>/env.sh"
command -v neuron-ls; neuron-ls
python3 .external/skills/aten-op-lowering/scripts/neuron_devices_free.py
python3 -c 'import torch_neuronx; print(torch_neuronx.__file__)'
```
If hardware is absent or busy, report `RESULT: BLOCKED` and `REASON: HARDWARE`. Never terminate another process. Never run `prerequisites.sh` on a non-Neuron host.
If system setup is missing, explain that `prerequisites.sh` changes system packages, uses `sudo`, and asks an interactive question. Show it and wait for explicit approval. Set up the project only after approval.
```bash
set -euo pipefail; source "agent_space/<task-slug>/env.sh"
uv venv; source .venv/bin/activate
uv pip install -e ".[dev]"; ./tools/build
python3 -c 'import torch_neuronx; print(torch_neuronx.__file__)'
python3 -c 'import expecttest, pytest, pytest_jsonreport'
```
Create an official PyTorch checkout at the pinned version.
```bash
set -euo pipefail; source "agent_space/<task-slug>/env.sh"
TORCH_PIN=$(python3 - <<'PY'
import re
from pathlib import Path

text = Path("pyproject.toml").read_text()
section = re.search(r"(?ms)^\[build-system\]\s*(.*?)(?=^\[|\Z)", text)
pins = re.findall(r"[\"']torch==([^\"']+)[\"']", section.group(1) if section else "")
if len(pins) != 1:
    raise SystemExit("The build-system table must contain one exact torch pin.")
print(pins[0])
PY
)
git clone --branch "v${TORCH_PIN}" --depth 1 https://github.com/pytorch/pytorch.git "$PYTORCH_ROOT"
test "$(git -C "$PYTORCH_ROOT" describe --tags --exact-match)" = "v${TORCH_PIN}"
```
The overlay helper deletes the `test` child of `tests/pytorch_tests_native` and replaces the `_internal` child under its `torch/testing` directory. Require clean destination status. If either destination exists, ask before replacement.
```bash
set -euo pipefail; source "agent_space/<task-slug>/env.sh"
OVERLAY_TEST='tests/pytorch_tests_native/test'
OVERLAY_INTERNAL='tests/pytorch_tests_native/torch/testing/_internal'
git status --short --untracked-files=all -- "$OVERLAY_TEST" "$OVERLAY_INTERNAL" > "$ARTIFACTS/overlay-status.txt"
test ! -s "$ARTIFACTS/overlay-status.txt" || { cat "$ARTIFACTS/overlay-status.txt"; exit 1; }
if test -e "$OVERLAY_TEST" || test -e "$OVERLAY_INTERNAL"; then
  printf '%s\n' 'Existing upstream overlay found. Approval is required before replacement.'; exit 2
fi
```
Run the next block only after explicit approval. It checks status again before replacement.
```bash
set -euo pipefail; source "agent_space/<task-slug>/env.sh"
git status --short --untracked-files=all -- tests/pytorch_tests_native/test \
  tests/pytorch_tests_native/torch/testing/_internal > "$ARTIFACTS/overlay-status.txt"
test ! -s "$ARTIFACTS/overlay-status.txt" || { cat "$ARTIFACTS/overlay-status.txt"; exit 1; }
python3 tests/pytorch_tests_native/clone_device_agnostic_tests.py --source-repo "$PYTORCH_ROOT"
```
Create and build an unchanged base worktree before edits.
```bash
set -euo pipefail; source "agent_space/<task-slug>/env.sh"
git worktree add "$BASE_WORKTREE" "$BASE_COMMIT"
test -z "$(git -C "$BASE_WORKTREE" status --short)"; : > "$ARTIFACTS/import-origins.txt"
(cd "$CANDIDATE_ROOT"; source .venv/bin/activate
 origin=$(python3 -c 'import torch_neuronx; print(torch_neuronx.__file__)')
 case "$origin" in "$CANDIDATE_ROOT"/*) ;; *) exit 1 ;; esac
 printf 'candidate=%s\n' "$origin" >> "$ARTIFACTS/import-origins.txt")
(cd "$BASE_WORKTREE"; uv venv; source .venv/bin/activate; uv pip install -e ".[dev]"; ./tools/build
 python3 tests/pytorch_tests_native/clone_device_agnostic_tests.py --source-repo "$PYTORCH_ROOT"
 origin=$(python3 -c 'import torch_neuronx; print(torch_neuronx.__file__)')
 case "$origin" in "$BASE_WORKTREE"/*) ;; *) exit 1 ;; esac
 printf 'base=%s\n' "$origin" >> "$ARTIFACTS/import-origins.txt")
cat "$ARTIFACTS/import-origins.txt"
```
Treat setup, checkout, import, or build failure as `RESULT: BLOCKED` with `REASON: ENVIRONMENT`.

## Collect and run upstream tests

Create `upstream-files.txt` with paths relative to the cloned PyTorch `test` directory. Check each collection exit status. Record a reason for zero matches.
```bash
set -euo pipefail; source "agent_space/<task-slug>/env.sh"
T='tests/pytorch_tests_native/test'; test -s "$ARTIFACTS/upstream-files.txt"
: > "$ARTIFACTS/op_nodeids.txt"; : > "$ARTIFACTS/collection-status.tsv"; index=0
while IFS= read -r file; do
  index=$((index + 1)); output="$ARTIFACTS/collect-${index}.txt"; matches="$ARTIFACTS/collect-${index}-nodeids.txt"
  set +e; python -m pytest "$T/$file" -k "$OP_PACKET" --collect-only -q > "$output" 2>&1; rc=$?; set -e
  if test "$rc" -ne 0; then printf '%s\t%s\tcollection failed\n' "$file" "$rc" >> "$ARTIFACTS/collection-status.tsv"; cat "$output"; exit "$rc"; fi
  grep -E "^${T}/.*::" "$output" > "$matches" || true
  if test -s "$matches"; then cat "$matches" >> "$ARTIFACTS/op_nodeids.txt"; reason='matched node IDs'
  else reason="zero matches for OP_PACKET=$OP_PACKET"; fi
  printf '%s\t0\t%s\n' "$file" "$reason" >> "$ARTIFACTS/collection-status.tsv"
done < "$ARTIFACTS/upstream-files.txt"
sort -u -o "$ARTIFACTS/op_nodeids.txt" "$ARTIFACTS/op_nodeids.txt"; test -s "$ARTIFACTS/op_nodeids.txt"
sed "s#^$T/##" "$ARTIFACTS/op_nodeids.txt" > "$ARTIFACTS/runner_items.txt"
test -s "$ARTIFACTS/runner_items.txt"; cat "$ARTIFACTS/collection-status.tsv"
```
Review node IDs and remove similarly named operators.
Store and reload runner items for every run. Capture each runner exit code under `set -e`. Merge every report that exists before classifying the exit code and report together.
```bash
set -euo pipefail; source "agent_space/<task-slug>/env.sh"
run_and_merge() {
  label=$1; root=$2; runxfail=$3; output="$ARTIFACTS/upstream-$label"
  paths="$ARTIFACTS/upstream-$label-paths.json"; merged="$ARTIFACTS/upstream-$label.json"
  set +e
  (cd "$root"; source .venv/bin/activate
   mapfile -t RUNNER_ITEMS < "$ARTIFACTS/runner_items.txt"; test "${#RUNNER_ITEMS[@]}" -gt 0
   if test "$runxfail" -eq 1; then
     PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} --runxfail" \
       tests/pytorch_tests_native/run_upstream_tests.sh --output-dir "$output" "${RUNNER_ITEMS[@]}"
   else
     tests/pytorch_tests_native/run_upstream_tests.sh --output-dir "$output" "${RUNNER_ITEMS[@]}"
   fi)
  runner_rc=$?; set -e
  printf '%s\n' "$runner_rc" > "$ARTIFACTS/upstream-$label.rc"
  python "$VERIFY" report-paths --output-dir "$output" \
    --items-file "$ARTIFACTS/runner_items.txt" --out "$paths"
  main_report=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["main"])' "$paths")
  report_state=missing
  if test -s "$main_report"; then
    set +e; python "$VERIFY" merge-reports --paths "$paths" --out "$merged"; merge_rc=$?; set -e
    if test "$merge_rc" -eq 0; then report_state=merged; else report_state=invalid; fi
  fi
  printf 'runner_rc=%s report=%s\n' "$runner_rc" "$report_state"
  test "$report_state" = merged || return 1
  if test "$runner_rc" -ne 0; then
    printf '%s\n' 'Runner returned nonzero; the verifier will classify the merged report.'
  fi
}
run_and_merge candidate "$CANDIDATE_ROOT" 0
run_and_merge base "$BASE_WORKTREE" 0
run_and_merge runxfail "$CANDIDATE_ROOT" 1
```

After a dispatch table change, run the global decomposition test.
```bash
set -euo pipefail; source "agent_space/<task-slug>/env.sh"
python -m pytest tests/pytorch_tests_native/test/test_decomp.py::HasDecompTest -q
```
Never edit overlay tests.

## Run coverage

Add exact backward rows and verify every shape registration.
```bash
set -euo pipefail; source "agent_space/<task-slug>/env.sh"
mapfile -t CLAIMED_OVERLOADS < "$ARTIFACTS/claimed_overloads.txt"
for op_name in "${CLAIMED_OVERLOADS[@]}"; do CHECK_OP="$op_name" python - <<'PY'
import os
from tests.vertical.internal.neuron_shape_registry import SHAPE_REGISTRY
op = os.environ["CHECK_OP"]
assert SHAPE_REGISTRY.has(op), f"No shape entry for {op}"
PY
done
```
Run eager and required compile coverage in fresh directories.
```bash
set -euo pipefail; source "agent_space/<task-slug>/env.sh"
OP_FILTER=$(paste -sd, "$ARTIFACTS/claimed_overloads.txt")
EAGER_CSV_DIR="$ARTIFACTS/coverage-eager"; rm -rf "$EAGER_CSV_DIR"; mkdir -p "$EAGER_CSV_DIR"
NEURON_COVERAGE_CSV_DIR="$EAGER_CSV_DIR" python -m pytest \
  tests/vertical/test_target_ops_coverage.py --ops "$OP_FILTER" -q
test "$(find "$EAGER_CSV_DIR" -maxdepth 1 -name 'target_ops_coverage_*.csv' | wc -l)" -eq 1
if test "$REQUIRE_COMPILE" -eq 1; then
  COMPILE_CSV_DIR="$ARTIFACTS/coverage-compile"; rm -rf "$COMPILE_CSV_DIR"; mkdir -p "$COMPILE_CSV_DIR"
  NEURON_COVERAGE_CSV_DIR="$COMPILE_CSV_DIR" python -m pytest \
    tests/vertical/test_target_ops_compile_coverage.py --ops "$OP_FILTER" -q
  test "$(find "$COMPILE_CSV_DIR" -maxdepth 1 -name 'target_ops_compile_coverage_*.csv' | wc -l)" -eq 1
else
  rm -rf "$ARTIFACTS/coverage-compile"
fi
```
Confirm each exact row passes.
