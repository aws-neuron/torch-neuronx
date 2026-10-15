#!/bin/bash
# Rerun a list of pytest tests from a torch-neuronx checkout and save output.
#
# Usage:
#   rerun_category.sh <torch-neuronx-root> <tests.txt> <output-dir> \
#       [-n <num_workers>] [--timeout <seconds>]
#
# The script runs `python -m pytest` in the current shell — if your
# Neuron hardware is behind an access wrapper, invoke this script
# through that wrapper (or run the pytest command yourself inside the
# wrapper's shell). See the hardware-wrapper skill for how to do that;
# this script is deliberately agnostic about it.
#
# Inputs:
#   tests.txt  — one pytest nodeid per line (from select_category_tests.py).
#              Lines are passed verbatim to pytest — they must match the
#              repo layout (e.g.
#              tests/pytorch_tests_native/test/test_ops_gradients.py::...).
#
# Optional flags:
#   -n <N>          pytest-xdist worker count. Tune per test type:
#                   distributed=8, autograd=16, others up to 64. For
#                   short lists (<30), cap at or below the test count.
#                   Default: no -n (single process).
#   --timeout <S>   pytest-timeout per test in seconds.
#                   Default: 600 (matches typical upstream-test runs).
#
# Outputs in <output-dir>:
#   pytest.log      — full pytest stdout+stderr (human-readable)
#   report.json     — pytest-json-report (structured per-test outcomes)
#   summary.txt     — one line per test: outcome + nodeid + first error line
#
# Uses --runxfail so xfail markers don't suppress real failure output:
#   - xfailed tests that still fail show their real traceback
#   - xfailed tests that now pass show up as 'passed'
#
# Environment prerequisites (fail fast if missing):
#   - A uv-managed venv is active (or `uv run` is used by the caller)
#     such that `python -m pytest` resolves and `import torch_neuronx`
#     succeeds.
#   - `pytest-json-report` installed. If missing: uv pip install pytest-json-report

set -u
REPO="${1:?torch-neuronx root required}"
TESTS_FILE="${2:?tests.txt required}"
OUT="${3:?output dir required}"
shift 3

# Defaults
NUM_WORKERS=""
TIMEOUT_SEC="600"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n)
            NUM_WORKERS="${2:?-n requires a value}"
            shift 2
            ;;
        --timeout)
            TIMEOUT_SEC="${2:?--timeout requires a value}"
            shift 2
            ;;
        *)
            echo "ERROR: unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

if [[ ! -d "$REPO" ]]; then
    echo "ERROR: repo not found: $REPO" >&2
    exit 2
fi
if [[ ! -f "$TESTS_FILE" ]]; then
    echo "ERROR: tests file not found: $TESTS_FILE" >&2
    exit 2
fi

mkdir -p "$OUT"
# Resolve OUT to absolute path so later file writes (which happen after
# `cd "$REPO"` below) land in the same directory that mkdir created —
# not in a same-named subdirectory of $REPO.
OUT="$(cd "$OUT" && pwd)"

# Environment health checks. Fail loudly, not mysteriously.
if ! python -c 'import torch_neuronx' 2>/dev/null; then
    echo "ERROR: 'import torch_neuronx' failed. Is the uv venv active?" >&2
    echo "       Try: source $REPO/.venv/bin/activate" >&2
    echo "       Or invoke this script with 'uv run' from the repo root." >&2
    exit 3
fi
if ! python -c 'import pytest_jsonreport' 2>/dev/null; then
    echo "ERROR: pytest-json-report not installed." >&2
    echo "       Install with: uv pip install pytest-json-report" >&2
    exit 3
fi

mapfile -t TESTS < <(grep -v '^[[:space:]]*$' "$TESTS_FILE")
N=${#TESTS[@]}
if [[ $N -eq 0 ]]; then
    echo "ERROR: no tests in $TESTS_FILE" >&2
    exit 2
fi

# Safety: if the caller asked for -n N with N > test count, clamp.
if [[ -n "$NUM_WORKERS" && "$NUM_WORKERS" -gt "$N" ]]; then
    echo "# clamping -n from $NUM_WORKERS to $N (one worker per test max)" >&2
    NUM_WORKERS="$N"
fi

echo "# Rerun $N tests from $TESTS_FILE" >&2
echo "# Repo:    $REPO" >&2
echo "# Output:  $OUT" >&2
echo "# Workers: ${NUM_WORKERS:-1 (single process)}" >&2
echo "# Timeout: ${TIMEOUT_SEC}s per test" >&2

cd "$REPO"

PYTEST_ARGS=(
    --runxfail
    -v -rs -rx
    --tb=short
    --timeout="$TIMEOUT_SEC"
    --json-report
    --json-report-file="$OUT/report.json"
)
if [[ -n "$NUM_WORKERS" ]]; then
    PYTEST_ARGS+=(-n "$NUM_WORKERS")
fi

python -m pytest \
    "${PYTEST_ARGS[@]}" \
    "${TESTS[@]}" \
    > "$OUT/pytest.log" 2>&1
EXIT=$?

# Compact per-test summary from the json report (stdlib only)
python - "$OUT/report.json" "$OUT/summary.txt" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
rep = json.loads(Path(sys.argv[1]).read_text())
lines = []
for t in rep.get("tests", []):
    nid = t["nodeid"]
    outcome = t["outcome"]
    msg = ""
    call = t.get("call") or {}
    crash = call.get("crash") or {}
    if crash.get("message"):
        msg = crash["message"].split("\n")[0][:200]
    lines.append(f"{outcome:10s}  {nid}  | {msg}")
Path(sys.argv[2]).write_text("\n".join(lines) + "\n")
c = Counter(t["outcome"] for t in rep.get("tests", []))
print(f"# summary: {len(lines)} tests -> {sys.argv[2]}", file=sys.stderr)
print(f"# outcomes: {dict(c)}", file=sys.stderr)
PY

echo "# pytest exit=$EXIT" >&2
exit 0
