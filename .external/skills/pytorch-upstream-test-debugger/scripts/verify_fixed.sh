#!/bin/bash
# Rerun selected tests and compare current outcomes/signatures with a baseline.
#
# Happy path, from the torch-neuronx repository root:
#   .external/skills/pytorch-upstream-test-debugger/scripts/verify_fixed.sh \
#       . category-tests.txt agent_space/<category-slug>/rerun-verify baseline-report.json
#
# Failure example: a changed failure signature is reported under
# still_bad_different_signature and exits 2 rather than being called fixed.
# Invalid reports or a rerun infrastructure failure exit non-zero.
#
# Exit codes: 0 all selected tests are non-failing; 1 regression from pass to
# failure; 2 one or more baseline failures still fail; pytest 3-5 are propagated;
# 10 means a prerequisite is missing; 11 means report or I/O failure.

usage() {
    cat <<'EOF'
Usage: verify_fixed.sh <repo-root> <tests.txt> <output-dir> <baseline.json> [-n N] [--timeout SECONDS]

Rerun selected tests. Compare outcomes and error signatures with a baseline.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

set -u

if [[ $# -lt 4 ]]; then
    usage >&2
    exit 2
fi

REPO="$1"
TESTS_FILE="$2"
OUT="$3"
BASELINE="$4"
shift 4

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

if [[ "${VERIFY_SAFE_OUTPUT_WORKER:-0}" != "1" ]]; then
    exec python3 "$SCRIPT_DIR/safe_output.py" run \
        --dir "$OUT" \
        --name pytest.log \
        --name report.json \
        --name summary.txt \
        --name verify.md \
        --redact-all \
        --error-exit 11 \
        -- env VERIFY_SAFE_OUTPUT_WORKER=1 bash "$0" \
        "$REPO" "$TESTS_FILE" "$OUT" "$BASELINE" "$@"
fi

if [[ ! -f "$BASELINE" ]]; then
    echo "ERROR: baseline report not found: ${BASELINE##*/}" >&2
    exit 11
fi

VERIFY_STAGE="$(mktemp -d "$TMP_ROOT/pytorch-upstream-verify.XXXXXX")" || exit 11
trap 'rm -rf -- "$VERIFY_STAGE"' EXIT
RERUN_STAGE="$VERIFY_STAGE/rerun"
if ! mkdir -- "$RERUN_STAGE"; then
    echo "ERROR: cannot create verification staging directory" >&2
    exit 11
fi

"$SCRIPT_DIR/rerun_category.sh" "$REPO" "$TESTS_FILE" "$RERUN_STAGE" "$@"
RERUN_EXIT=$?
# pytest exit 1 means tests failed and is valid input for comparison. Other
# non-zero statuses indicate usage, dependency, collection, or report failure.
if [[ $RERUN_EXIT -gt 1 ]]; then
    echo "ERROR: rerun failed before a comparable test result was available (exit=$RERUN_EXIT)" >&2
    exit "$RERUN_EXIT"
fi
if [[ ! -f "$RERUN_STAGE/report.json" ]]; then
    echo "ERROR: rerun did not produce report.json" >&2
    exit 11
fi

PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$BASELINE" "$RERUN_STAGE/report.json" "$VERIFY_STAGE/verify.md" <<'PY'
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from redact import redact_text
from report_signatures import (
    MAX_IDENTIFIER_CHARS,
    extract_failure_text,
    validate_nodeid,
    validate_tests_array,
)

PASS = {"passed"}
NOISE = (
    re.compile(r"0x[0-9a-fA-F]+"),
    re.compile(r"\b\d+\.\d+e[+-]?\d+\b"),
    re.compile(r"\b\d+\.\d+\b"),
    re.compile(r"\b\d{2,}\b"),
    re.compile(r"[\w./\\-]+:\d+:"),
    re.compile(r"tensor\([^)]*\)"),
)
KNOWN = (
    (re.compile(r"Double did not match Float|got dtype torch\.float64", re.I), "dtype_mismatch_double_vs_float"),
    (re.compile(r"Jacobian mismatch for output", re.I), "jacobian_mismatch"),
    (re.compile(r"gradcheck.*numerical", re.I), "gradcheck_numerical"),
    (re.compile(r"Compilation error occurred", re.I), "compiler_error_neuron"),
    (re.compile(r"out of memory", re.I), "oom"),
)


def load_report(path_text):
    path = Path(path_text)
    try:
        text = path.read_text()
    except OSError as exc:
        raise ValueError(
            f"cannot read report {path.name}: {exc.strerror or 'I/O error'}"
        ) from None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON report {path.name}: {exc.msg}") from None
    tests = validate_tests_array(data)
    if not tests:
        raise ValueError("report must contain a non-empty tests array")
    result = {}
    for index, test in enumerate(tests):
        nodeid = validate_nodeid(test.get("nodeid"), label=f"tests[{index}].nodeid")
        outcome = test.get("outcome")
        if not isinstance(outcome, str) or not outcome:
            raise ValueError(f"tests[{index}] has no outcome")
        if len(outcome) > MAX_IDENTIFIER_CHARS:
            raise ValueError(f"tests[{index}].outcome is too long")
        if nodeid in result:
            raise ValueError(f"duplicate nodeid: {nodeid}")
        result[nodeid] = test
    return result


def signature(test):
    text = redact_text(extract_failure_text(test))
    for pattern, label in KNOWN:
        if pattern.search(text):
            return label
    line = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("E   "):
            line = stripped[4:].strip()
            break
    if not line:
        line = next((raw.strip() for raw in text.splitlines() if raw.strip()), "")
    for pattern in NOISE:
        line = pattern.sub("<N>", line)
    return re.sub(r"\s+", " ", line).strip()[:160] or "unknown_error"


try:
    baseline = load_report(sys.argv[1])
    current = load_report(sys.argv[2])
    baseline_ids = set(baseline)
    current_ids = set(current)
    if baseline_ids != current_ids:
        missing = sorted(baseline_ids - current_ids)
        extra = sorted(current_ids - baseline_ids)
        raise ValueError(
            f"report nodeid sets differ: missing_current={len(missing)} extra_current={len(extra)}"
        )

    buckets = defaultdict(list)
    for nodeid in sorted(baseline_ids):
        before = baseline[nodeid]
        after = current[nodeid]
        before_outcome = before["outcome"]
        after_outcome = after["outcome"]
        if before_outcome in PASS and after_outcome in PASS:
            buckets["unchanged"].append(nodeid)
        elif before_outcome not in PASS and after_outcome in PASS:
            buckets["fixed"].append(nodeid)
        elif before_outcome in PASS and after_outcome not in PASS:
            buckets["regressed"].append((nodeid, after_outcome, signature(after)))
        else:
            before_signature = signature(before)
            after_signature = signature(after)
            key = (
                "still_bad_same_signature"
                if before_signature == after_signature
                else "still_bad_different_signature"
            )
            buckets[key].append(
                (nodeid, before_outcome, after_outcome, before_signature, after_signature)
            )

    lines = [
        "# Verify report",
        "",
        f"- fixed: {len(buckets['fixed'])}",
        f"- regressed: {len(buckets['regressed'])}",
        f"- still_bad_same_signature: {len(buckets['still_bad_same_signature'])}",
        f"- still_bad_different_signature: {len(buckets['still_bad_different_signature'])}",
        f"- unchanged: {len(buckets['unchanged'])}",
        "",
    ]
    if buckets["fixed"]:
        lines.extend([f"## fixed ({len(buckets['fixed'])})", ""])
        lines.extend(f"- `{nodeid}`" for nodeid in buckets["fixed"])
        lines.append("")
    if buckets["regressed"]:
        lines.extend([f"## regressed ({len(buckets['regressed'])})", ""])
        lines.extend(
            f"- `{nodeid}` -> {outcome}; signature=`{current_signature}`"
            for nodeid, outcome, current_signature in buckets["regressed"]
        )
        lines.append("")
    for key in ("still_bad_same_signature", "still_bad_different_signature"):
        if buckets[key]:
            lines.extend([f"## {key} ({len(buckets[key])})", ""])
            lines.extend(
                f"- `{nodeid}`: {before_outcome} -> {after_outcome}; "
                f"baseline=`{before_signature}`; current=`{after_signature}`"
                for nodeid, before_outcome, after_outcome, before_signature, after_signature in buckets[key]
            )
            lines.append("")

    Path(sys.argv[3]).write_text(redact_text("\n".join(lines) + "\n"))
except OSError as exc:
    print(
        f"ERROR: cannot write verification report: {exc.strerror or 'I/O error'}",
        file=sys.stderr,
    )
    sys.exit(11)
except ValueError as exc:
    print(f"ERROR: cannot compare verification reports: {exc}", file=sys.stderr)
    sys.exit(11)

print(
    "# verify: "
    f"fixed={len(buckets['fixed'])} "
    f"regressed={len(buckets['regressed'])} "
    f"same_signature={len(buckets['still_bad_same_signature'])} "
    f"different_signature={len(buckets['still_bad_different_signature'])} "
    f"unchanged={len(buckets['unchanged'])}",
    file=sys.stderr,
)

if buckets["regressed"]:
    sys.exit(1)
if buckets["still_bad_same_signature"] or buckets["still_bad_different_signature"]:
    sys.exit(2)
sys.exit(0)
PY

COMPARE_EXIT=$?
if [[ $COMPARE_EXIT -gt 2 ]]; then
    exit "$COMPARE_EXIT"
fi
for artifact in pytest.log report.json summary.txt; do
    if ! mv -- "$RERUN_STAGE/$artifact" "$SAFE_OUTPUT_STAGE/$artifact"; then
        echo "ERROR: cannot stage verification artifact: $artifact" >&2
        exit 11
    fi
done
if ! mv -- "$VERIFY_STAGE/verify.md" "$SAFE_OUTPUT_STAGE/verify.md"; then
    echo "ERROR: cannot stage verification report" >&2
    exit 11
fi
exit "$COMPARE_EXIT"
