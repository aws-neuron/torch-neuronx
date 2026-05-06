#!/bin/bash
# =============================================================================
# Upstream PyTorch Test Runner
# =============================================================================
#
# Runs upstream PyTorch test files/folders in a single pytest invocation.
# Supports sharding (via pytest-shard) and ignore patterns.
#
# Usage: ./run_upstream_tests.sh [OPTIONS] <item1> [<item2> ...]
#
# Options:
#   --output-dir DIR       Directory for test artifacts (default: /tmp/upstream_pytorch_artifacts)
#   --total-timeout SECS   Total time budget for all items (auto-calculates item timeout)
#   --shard-id ID          Shard ID (0-indexed) for pytest-shard
#   --num-shards N         Total number of shards for pytest-shard
#   --ignore PATH          Path to ignore (relative to tests/pytorch_tests_native/test/).
#                          Can be specified multiple times.
#
# Environment Variables:
#   CLASS_TIMEOUT          Total timeout in seconds (default: 10800, or auto-calculated)
#   TEST_TIMEOUT           Per-test timeout in seconds (default: 600)
#   NUM_WORKERS            Number of pytest-xdist workers (default: 64)
#   NUM_WORKERS_RETRY      Workers for the retry pass (default: 4)
#   RETRY_TIMEOUT          Wall-clock budget for retry in seconds (default: 1800)
#
# Timeout Behavior:
#   - Per-test timeout via pytest-timeout marks tests as failed (non-distributed)
#   - Distributed tests use --forked instead of per-test timeout
#
# Examples:
#   ./run_upstream_tests.sh test_ops.py
#   ./run_upstream_tests.sh --shard-id 0 --num-shards 4 test_ops.py
#   ./run_upstream_tests.sh --ignore distributed/fsdp --ignore distributed/checkpoint distributed
#   ./run_upstream_tests.sh test_nn.py test_torch.py
#
# =============================================================================

set -e

# Configuration
NUM_WORKERS=${NUM_WORKERS:-64}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_PATH="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="/tmp/upstream_pytorch_artifacts"
TOTAL_TIMEOUT=""
CLASS_TIMEOUT=${CLASS_TIMEOUT:-7200}  # 2 hours default
TEST_TIMEOUT=${TEST_TIMEOUT:-800}     # 15 minutes (temp reduce test flakiness with 10 minutes)
# Retry pass (fires only if main pytest exits with code 1 — see retry block below).
# Reduced concurrency avoids recreating cascading NRT conditions; separate
# timeout keeps retry from consuming a large share of the wall-clock budget.
NUM_WORKERS_RETRY=${NUM_WORKERS_RETRY:-4}
RETRY_TIMEOUT=${RETRY_TIMEOUT:-1800}  # 30 minutes
SHARD_ID=""
NUM_SHARDS=""
CLONE_SCRIPT="tests/pytorch_tests_native/clone_device_agnostic_tests.py"
declare -a IGNORE_PATHS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --total-timeout)
            TOTAL_TIMEOUT="$2"
            shift 2
            ;;
        --shard-id)
            SHARD_ID="$2"
            shift 2
            ;;
        --num-shards)
            NUM_SHARDS="$2"
            shift 2
            ;;
        --pytorch-fork-branch)
            PYTORCH_FORK_BRANCH="$2"
            shift 2
            ;;
        --ignore)
            IGNORE_PATHS+=("$2")
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Get items from positional args
if [ $# -gt 0 ]; then
    ITEMS=("$@")
else
    echo "Error: No test files/folders specified"
    exit 1
fi

if [ ${#ITEMS[@]} -eq 0 ]; then
    echo "Error: No items to run"
    exit 1
fi

# Calculate CLASS_TIMEOUT from total timeout if provided
# Reserve 20 minutes (1200s) for post-processing
if [ -n "$TOTAL_TIMEOUT" ]; then
    POST_PROCESSING_BUFFER=1200
    CLASS_TIMEOUT=$((TOTAL_TIMEOUT - POST_PROCESSING_BUFFER))
    echo "Auto-calculated CLASS_TIMEOUT: ${CLASS_TIMEOUT}s (total: ${TOTAL_TIMEOUT}s, buffer: ${POST_PROCESSING_BUFFER}s)"
fi

# Setup
mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/json_reports" "$OUTPUT_DIR/html_reports" "$OUTPUT_DIR/csv_reports"
export TEST_RUN_TIMESTAMP=${TEST_RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
cd "$BASE_PATH"

# Clone upstream PyTorch tests (skips if already exists with matching version)
python3 "$CLONE_SCRIPT" --pytorch-fork-branch "${PYTORCH_FORK_BRANCH:-main}" || echo "Cloning branch ${PYTORCH_FORK_BRANCH:-main} failed (non-fatal)"

echo "=== PyTorch Test Runner ==="
echo "Items to run: ${ITEMS[*]}"
echo "Workers: $NUM_WORKERS"
echo "Item timeout: ${CLASS_TIMEOUT}s, Test timeout: ${TEST_TIMEOUT}s"
echo "Output: $OUTPUT_DIR"
echo "================================="

FAILED_ITEMS=0
TOTAL_ITEMS=${#ITEMS[@]}

# Build shard args if provided
SHARD_ARGS=""
if [[ -n "$SHARD_ID" && -n "$NUM_SHARDS" ]]; then
    SHARD_ARGS="--shard-id=$SHARD_ID --num-shards=$NUM_SHARDS"
fi

# Build ignore args
IGNORE_ARGS=""
for IGN in "${IGNORE_PATHS[@]}"; do
    IGNORE_ARGS="$IGNORE_ARGS --ignore=tests/pytorch_tests_native/test/$IGN"
done

# Build pytest targets (all items in single invocation)
PYTEST_TARGETS=""
for ITEM in "${ITEMS[@]}"; do
    PYTEST_TARGETS="$PYTEST_TARGETS tests/pytorch_tests_native/test/$ITEM"
done

# Derive safe name from first item for report naming
SAFE_NAME="${ITEMS[0]//\//_}"
SAFE_NAME="${SAFE_NAME%.py}"
if [ ${#ITEMS[@]} -gt 1 ]; then
    SAFE_NAME="${SAFE_NAME}_and_${#ITEMS[@]}_more"
fi

REPORT_FILE="$OUTPUT_DIR/json_reports/test_results_${SAFE_NAME}.json"
HTML_FILE="$OUTPUT_DIR/html_reports/${SAFE_NAME}.html"
LOG_FILE="$OUTPUT_DIR/${SAFE_NAME}.log"

if [[ -n "$SHARD_ID" && -n "$NUM_SHARDS" ]]; then
    REPORT_FILE="$OUTPUT_DIR/json_reports/test_results_${SAFE_NAME}_shard${SHARD_ID}.json"
    HTML_FILE="$OUTPUT_DIR/html_reports/${SAFE_NAME}_shard${SHARD_ID}.html"
    LOG_FILE="$OUTPUT_DIR/${SAFE_NAME}_shard${SHARD_ID}.log"
    echo "Sharding: shard $SHARD_ID of $NUM_SHARDS"
fi

echo "=== Running: ${ITEMS[*]} ==="

# Use --forked for distributed tests, --timeout for others.
#
# NOTE on --forked + --last-failed: pytest's last-failed cache is written
# by the main pytest process (not the forked children), so --last-failed
# correctly replays the prior run's failures even under --forked.
TIMEOUT_ARGS="--timeout=$TEST_TIMEOUT --timeout-method=signal"
for ITEM in "${ITEMS[@]}"; do
    if [[ "$ITEM" == distributed || "$ITEM" == distributed/* ]]; then
        TIMEOUT_ARGS="--forked"
        break
    fi
done

# Pytest flags shared by both the main and retry invocations. Keeping these
# in a single array avoids subtle drift between the two runs.
# Note: $SHARD_ARGS / $IGNORE_ARGS / $TIMEOUT_ARGS are intentionally passed
# separately (unquoted) so the shell handles word-splitting — pytest must
# see e.g. '--shard-id=0' '--num-shards=4' as two tokens.
PYTEST_COMMON_ARGS=(
    --tb=short
    --maxfail=0
    --continue-on-collection-errors
    -v
)

set +e
# shellcheck disable=SC2086  # intentional word-splitting for $SHARD_ARGS/$IGNORE_ARGS/$TIMEOUT_ARGS/$PYTEST_TARGETS
timeout --signal=SIGINT --kill-after=300 $CLASS_TIMEOUT pytest -n "$NUM_WORKERS" \
    $SHARD_ARGS \
    $IGNORE_ARGS \
    --json-report \
    --json-report-file="$REPORT_FILE" \
    --html="$HTML_FILE" --self-contained-html \
    $TIMEOUT_ARGS \
    "${PYTEST_COMMON_ARGS[@]}" \
    $PYTEST_TARGETS > >(tee "$LOG_FILE") 2>&1
EXIT_CODE=$?
set -e

if [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 130 ] || [ $EXIT_CODE -eq 137 ]; then
    echo "WARNING: Hit timeout after ${CLASS_TIMEOUT}s (exit code: $EXIT_CODE)"
fi
if [ $EXIT_CODE -ne 0 ]; then
    FAILED_ITEMS=1
fi

# =============================================================================
# Single retry pass for failed tests
# =============================================================================
#
# pytest's --last-failed flag reuses the .pytest_cache populated by the
# main run to rerun only the tests that failed. This catches flaky
# failures caused by cascading NRT/worker crashes: when a worker
# segfaults, pytest-xdist marks the tests assigned to that worker as
# FAILED with 0.0s duration even though they never actually ran. Those
# victims typically pass on a second attempt.
#
# Only retry if:
#   - main pytest finished normally with failures (exit code 1)
#   - not on wall-clock timeout (124/130/137) — the cache is incomplete
#   - not on internal / usage / collection errors (>1) — retry won't help
#
# Retry results are written with a _retry suffix next to the main
# reports so the aggregator can identify which previously-failing
# tests passed on retry and count them as passes.
#
# Concurrency and timeout are controlled by NUM_WORKERS_RETRY and
# RETRY_TIMEOUT env vars (defaults set near the top of this script).
RETRY_EXIT_CODE=0
if [ $EXIT_CODE -eq 1 ]; then
    RETRY_REPORT_FILE="${REPORT_FILE%.json}_retry.json"
    RETRY_HTML_FILE="${HTML_FILE%.html}_retry.html"
    RETRY_LOG_FILE="${LOG_FILE%.log}_retry.log"

    echo ""
    echo "=== Retrying failed tests (single pass, ${NUM_WORKERS_RETRY} workers, timeout ${RETRY_TIMEOUT}s) ==="
    echo "Retry reports:"
    echo "  json: $RETRY_REPORT_FILE"
    echo "  html: $RETRY_HTML_FILE"
    echo "  log:  $RETRY_LOG_FILE"

    # --last-failed-no-failures=none: exit 5 if the pytest cache has no
    # recorded failures (defensive; shouldn't normally happen since
    # EXIT_CODE=1 implies failures occurred). Treat exit 5 as success.
    set +e
    # shellcheck disable=SC2086  # intentional word-splitting (see main invocation)
    timeout --signal=SIGINT --kill-after=300 $RETRY_TIMEOUT pytest -n "$NUM_WORKERS_RETRY" \
        --last-failed \
        --last-failed-no-failures=none \
        $SHARD_ARGS \
        $IGNORE_ARGS \
        --json-report \
        --json-report-file="$RETRY_REPORT_FILE" \
        --html="$RETRY_HTML_FILE" --self-contained-html \
        $TIMEOUT_ARGS \
        "${PYTEST_COMMON_ARGS[@]}" \
        $PYTEST_TARGETS > >(tee "$RETRY_LOG_FILE") 2>&1
    RETRY_EXIT_CODE=$?
    set -e

    # Exit 0: all previously-failing tests now pass.
    # Exit 5: no tests collected (cache empty — shouldn't happen here, treat as success).
    if [ $RETRY_EXIT_CODE -eq 0 ] || [ $RETRY_EXIT_CODE -eq 5 ]; then
        # Flakiness recovered. Override the final exit code so the pipeline
        # reports success; the aggregator will surface the passed_on_retry
        # count as a flakiness signal.
        echo ""
        echo "=== Retry SUCCESS: all previously-failing tests passed on retry ==="
        EXIT_CODE=0
        FAILED_ITEMS=0
    else
        echo ""
        echo "=== Retry incomplete: some tests still failing (retry exit: $RETRY_EXIT_CODE) ==="
    fi
fi

# Summary
echo ""
echo "=== All Items Complete ==="
echo "Total: $TOTAL_ITEMS, Failed: $FAILED_ITEMS"

if [ $FAILED_ITEMS -eq 0 ]; then
    echo "Status: SUCCESS"
else
    echo "Status: $FAILED_ITEMS/$TOTAL_ITEMS items had failures"
fi

# Generate HUD benchmark NDJSON from JSON reports
echo ""
echo "=== Generating HUD benchmark results ==="
PYTHONPATH="$BASE_PATH/tests:${PYTHONPATH:-}" python3 \
    "$BASE_PATH/tests/utils/generate_hud.py" \
    "Neuron PyTorch Upstream tests" "$OUTPUT_DIR/json_reports/test_results_*.json" "$OUTPUT_DIR" \
    || echo "WARNING: HUD generation failed (non-fatal)"

# Convert pytest JSON reports to per-class CSV. csv_reports/ is for reviewer
# convenience when inspecting a single node's artifact — the aggregator
# re-reads from json_reports/ directly.
echo ""
echo "=== Converting JSON reports to per-class CSVs ==="
python3 "$BASE_PATH/tests/pytorch_tests_native/runner/convert_reports.py" "$OUTPUT_DIR" \
    || echo "WARNING: CSV conversion failed (non-fatal)"

exit $EXIT_CODE
