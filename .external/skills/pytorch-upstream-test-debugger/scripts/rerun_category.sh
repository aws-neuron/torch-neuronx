#!/bin/bash
# Rerun selected pytest node IDs and save a validated JSON report plus summary.
#
# Safety choice: reproduce the official runner's env-lock preload and isolation
# split here. The official runner rewrites suite items, so it cannot preserve
# arbitrary explicit node IDs. Distributed targets use --forked without
# --timeout. Other targets use --timeout without --forked.

usage() {
    cat <<'EOF'
Usage: rerun_category.sh <repo-root> <tests.txt> <output-dir> [-n N] [--timeout SECONDS]

Run selected pytest node IDs. Write pytest.log, report.json, and summary.txt.
Return pytest's status after fresh-report validation.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

set -u

if [[ $# -lt 3 ]]; then
    usage >&2
    exit 2
fi

REPO="$1"
TESTS_FILE="$2"
OUT="$3"
shift 3

NUM_WORKERS=""
TIMEOUT_SEC="600"
PYTHON_BIN="${PYTHON:-python}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: -n requires a value" >&2
                exit 2
            fi
            NUM_WORKERS="$2"
            shift 2
            ;;
        --timeout)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --timeout requires a value" >&2
                exit 2
            fi
            TIMEOUT_SEC="$2"
            shift 2
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
case "$OUT" in
    /*) ;;
    *) OUT="$PWD/$OUT" ;;
esac

TMP_ROOT="${TMPDIR:-/tmp}"
if [[ "$TMP_ROOT" != /* || ! -d "$TMP_ROOT" || -L "$TMP_ROOT" ]]; then
    echo "ERROR: TMPDIR must be an absolute path to an existing real directory" >&2
    exit 10
fi

if [[ "${RERUN_SAFE_OUTPUT_WORKER:-0}" != "1" ]]; then
    WORKER_ARGS=("$REPO" "$TESTS_FILE" "$OUT")
    if [[ -n "$NUM_WORKERS" ]]; then
        WORKER_ARGS+=(-n "$NUM_WORKERS")
    fi
    WORKER_ARGS+=(--timeout "$TIMEOUT_SEC")
    exec python3 "$SCRIPT_DIR/safe_output.py" run \
        --dir "$OUT" \
        --name pytest.log \
        --name report.json \
        --name summary.txt \
        --redact-all \
        --error-exit 11 \
        -- env RERUN_SAFE_OUTPUT_WORKER=1 bash "$0" "${WORKER_ARGS[@]}"
fi

if [[ ! -d "$REPO" ]]; then
    echo "ERROR: repository directory not found: ${REPO##*/}" >&2
    exit 2
fi
if [[ ! -f "$TESTS_FILE" ]]; then
    echo "ERROR: tests file not found: ${TESTS_FILE##*/}" >&2
    exit 2
fi
if [[ ! "$TIMEOUT_SEC" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --timeout must be a positive integer" >&2
    exit 2
fi
if [[ -n "$NUM_WORKERS" && ! "$NUM_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: -n must be a positive integer" >&2
    exit 2
fi

WORK_DIR="$(mktemp -d "$TMP_ROOT/pytorch-upstream-rerun.XXXXXX")" || exit 11
trap 'rm -rf -- "$WORK_DIR"' EXIT
INSTALL_DIR="${SAFE_OUTPUT_STAGE:?safe output staging is unavailable}"
VALIDATED_TESTS="$WORK_DIR/tests.txt"
LOG_TEMP="$INSTALL_DIR/pytest.log"
REPORT_TEMP="$INSTALL_DIR/report.json"
SUMMARY_TEMP="$INSTALL_DIR/summary.txt"
ENV_LOCK_DIR="$WORK_DIR/env-lock"

if ! PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - \
    "$TESTS_FILE" "$REPO" >"$VALIDATED_TESTS" <<'PY'
import sys
from pathlib import Path

from report_signatures import MAX_RECORDS, validate_nodeid

try:
    text = Path(sys.argv[1]).read_text()
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines:
        raise ValueError("tests file contains no node IDs")
    if len(lines) > MAX_RECORDS:
        raise ValueError(f"tests file contains more than {MAX_RECORDS} node IDs")
    repo_root = Path(sys.argv[2]).resolve(strict=True)
    seen = set()
    for index, line in enumerate(lines, start=1):
        if not line or line != line.strip():
            raise ValueError(f"tests file line {index} is empty or has surrounding whitespace")
        nodeid = validate_nodeid(line, label=f"tests file line {index}")
        file_part = nodeid.partition("::")[0]
        resolved_file = (repo_root / file_part).resolve(strict=False)
        try:
            resolved_file.relative_to(repo_root)
        except ValueError:
            raise ValueError(f"tests file line {index} resolves outside the repository") from None
        if nodeid in seen:
            raise ValueError(f"tests file line {index} duplicates a node ID")
        seen.add(nodeid)
        print(nodeid)
except (OSError, UnicodeError, ValueError) as exc:
    print(f"ERROR: invalid tests file: {exc}", file=sys.stderr)
    sys.exit(2)
PY
then
    exit 2
fi
mapfile -t TESTS <"$VALIDATED_TESTS"
TEST_COUNT=${#TESTS[@]}

require_module() {
    local module="$1"
    local package="$2"
    if ! "$PYTHON_BIN" -c "import $module" 2>/dev/null; then
        echo "ERROR: required package is unavailable: $package" >&2
        exit 10
    fi
}
require_module pytest pytest
require_module torch_neuronx torch-neuronx
require_module pytest_jsonreport pytest-json-report
require_module pytest_timeout pytest-timeout
if [[ -n "$NUM_WORKERS" ]]; then
    require_module xdist pytest-xdist
fi

if [[ -n "$NUM_WORKERS" && "$NUM_WORKERS" -gt "$TEST_COUNT" ]]; then
    echo "# clamping workers from $NUM_WORKERS to $TEST_COUNT" >&2
    NUM_WORKERS="$TEST_COUNT"
fi

DISTRIBUTED=0
for nodeid in "${TESTS[@]}"; do
    test_path="${nodeid%%::*}"
    if [[ "$test_path" == distributed/* || "$test_path" == */distributed/* ]]; then
        DISTRIBUTED=1
        break
    fi
done
if [[ $DISTRIBUTED -eq 1 ]]; then
    require_module pytest_forked pytest-forked
fi

ENV_LOCK_SOURCE="$REPO/tests/pytorch_tests_native/runner/env_lock.c"
if [[ ! -f "$ENV_LOCK_SOURCE" ]]; then
    echo "ERROR: env-lock source is unavailable: tests/pytorch_tests_native/runner/env_lock.c" >&2
    exit 10
fi
if [[ -n "${CC:-}" ]]; then
    CC_BIN="$(command -v -- "$CC" || true)"
else
    CC_BIN="$(command -v cc || command -v gcc || true)"
fi
if [[ -z "$CC_BIN" ]]; then
    echo "ERROR: required package is unavailable: C compiler (cc or gcc)" >&2
    exit 10
fi
if ! mkdir -- "$ENV_LOCK_DIR"; then
    echo "ERROR: cannot create environment-lock staging directory" >&2
    exit 11
fi
ENV_LOCK_SO="$ENV_LOCK_DIR/env_lock.so"
if ! "$CC_BIN" -O2 -g -fPIC -shared -o "$ENV_LOCK_SO" "$ENV_LOCK_SOURCE" -lpthread -ldl \
    >"$ENV_LOCK_DIR/build.log" 2>&1; then
    echo "ERROR: env-lock shim compilation failed" >&2
    exit 10
fi

PYTEST_ARGS=(
    --runxfail
    -v -rs -rx
    --tb=short
    --json-report
    --json-report-file="$REPORT_TEMP"
)
if [[ $DISTRIBUTED -eq 1 ]]; then
    PYTEST_ARGS+=(--forked)
else
    PYTEST_ARGS+=(--timeout="$TIMEOUT_SEC")
fi
if [[ -n "$NUM_WORKERS" ]]; then
    PYTEST_ARGS+=(-n "$NUM_WORKERS")
fi

echo "# rerunning $TEST_COUNT selected tests" >&2
echo "# repository: ${REPO##*/}" >&2
echo "# output directory: ${OUT##*/}" >&2
echo "# workers: ${NUM_WORKERS:-1}" >&2
if [[ $DISTRIBUTED -eq 1 ]]; then
    echo "# isolation: forked distributed tests; no per-test timeout" >&2
else
    echo "# timeout: ${TIMEOUT_SEC}s per test" >&2
fi

if ! cd "$REPO"; then
    echo "ERROR: cannot enter repository directory" >&2
    exit 2
fi

LD_PRELOAD="$ENV_LOCK_SO${LD_PRELOAD:+ $LD_PRELOAD}" "$PYTHON_BIN" -m pytest \
    "${PYTEST_ARGS[@]}" -- "${TESTS[@]}" \
    >"$LOG_TEMP" 2>&1
PYTEST_EXIT=$?

if ! PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - \
    "$REPORT_TEMP" "$SUMMARY_TEMP" "$VALIDATED_TESTS" "$PYTEST_EXIT" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

from redact import redact_text
from report_signatures import (
    MAX_IDENTIFIER_CHARS,
    extract_failure_fields,
    validate_nodeid,
    validate_tests_array,
)

try:
    report = json.loads(Path(sys.argv[1]).read_text())
    tests = validate_tests_array(report)
    if not tests:
        raise ValueError("report must contain a non-empty tests array")
    expected = set(Path(sys.argv[3]).read_text().splitlines())
    actual = set()
    lines = []
    outcomes = Counter()
    for index, test in enumerate(tests):
        nodeid = validate_nodeid(test.get("nodeid"), label=f"tests[{index}].nodeid")
        if nodeid in actual:
            raise ValueError(f"duplicate report node ID: {nodeid}")
        actual.add(nodeid)
        outcome = test.get("outcome")
        if not isinstance(outcome, str) or not outcome:
            raise ValueError(f"tests[{index}] has no outcome")
        if len(outcome) > MAX_IDENTIFIER_CHARS:
            raise ValueError(f"tests[{index}].outcome is too long")
        message, longrepr, stage = extract_failure_fields(test)
        if not message and longrepr:
            detail_lines = [line.strip() for line in longrepr.splitlines() if line.strip()]
            message = detail_lines[-1] if detail_lines else ""
        if message and stage and stage != "call":
            message = f"[{stage}] {message}"
        lines.append(redact_text(f"{outcome:10s}  {nodeid}  | {message[:200]}"))
        outcomes[outcome] += 1
    if actual != expected:
        raise ValueError(
            "report node ID set differs from requested set: "
            f"missing={len(expected - actual)} extra={len(actual - expected)}"
        )
    pytest_exit = int(sys.argv[4])
    has_failures = bool(set(outcomes) & {"failed", "error"})
    if pytest_exit == 0 and has_failures:
        raise ValueError("pytest exited 0 but the fresh report contains failures")
    if pytest_exit == 1 and not has_failures:
        raise ValueError("pytest exited 1 but the fresh report contains no failures")
    Path(sys.argv[2]).write_text("\n".join(lines) + "\n")
except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
    print(f"ERROR: cannot validate pytest JSON report: {exc}", file=sys.stderr)
    sys.exit(11)

print(f"# summary: {len(lines)} tests", file=sys.stderr)
print(f"# outcomes: {dict(outcomes)}", file=sys.stderr)
PY
then
    echo "# pytest exit=$PYTEST_EXIT; report validation failed" >&2
    exit 11
fi

echo "# pytest exit=$PYTEST_EXIT" >&2
exit "$PYTEST_EXIT"
