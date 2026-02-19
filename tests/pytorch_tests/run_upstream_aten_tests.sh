#!/bin/bash
# =============================================================================
# Upstream PyTorch Aten Test Runner
# =============================================================================
#
# Runs upstream PyTorch test classes sequentially, one class at a time.
#
# Usage: ./run_upstream_aten_tests.sh [--output-dir DIR] [--spec-file FILE] [--class-list "Class1,Class2,..."] [class_name1] [class_name2] ...
#
# Options:
#   --output-dir DIR       Directory for test artifacts (default: /tmp/upstream_aten_artifacts)
#   --spec-file FILE       Spec file name in tests/pytorch_tests/specs/ (default: aten_full_suite_spec.json)
#   --class-list LIST      Comma-separated list of class names to run
#   --total-timeout SECS   Total time budget for all classes (auto-calculates CLASS_TIMEOUT)
#
# Environment Variables:
#   CLASS_TIMEOUT          Timeout per class in seconds (default: 14400, or auto-calculated from --total-timeout)
#   TEST_TIMEOUT           Timeout per test in seconds (default: 3600)
#   NUM_WORKERS            Number of parallel workers (default: 64)
#
# Timeout Behavior:
#   - Per-test timeout via pytest-timeout marks tests as failed
#   - Class-level timeout is a safety net to prevent hung classes from blocking the pipeline
#
# Examples:
#   ./run_upstream_aten_tests.sh TestNNDeviceType
#   ./run_upstream_aten_tests.sh --class-list "TestCommon,TestUnaryUfuncs,TestMeta"
#   ./run_upstream_aten_tests.sh --output-dir /path/to/output TestNNDeviceType TestModuleDeviceType
#   ./run_upstream_aten_tests.sh --spec-file aten_spec.json TestDropoutNNDeviceType
#
# =============================================================================

set -e

# Configuration
NUM_WORKERS=${NUM_WORKERS:-64}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_PATH="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="/tmp/upstream_aten_artifacts"
SPEC_FILE="aten_full_suite_spec.json"
CLASS_LIST=""
TOTAL_TIMEOUT=""
CLASS_TIMEOUT=${CLASS_TIMEOUT:-14400}  # 4 hours default
TEST_TIMEOUT=${TEST_TIMEOUT:-3600}      # 1 hour

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --spec-file)
            SPEC_FILE="$2"
            shift 2
            ;;
        --class-list)
            CLASS_LIST="$2"
            shift 2
            ;;
        --total-timeout)
            TOTAL_TIMEOUT="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Get class names - from --class-list, positional args, or spec file
if [ -n "$CLASS_LIST" ]; then
    # Parse comma-separated list into array
    IFS=',' read -ra CLASS_NAMES <<< "$CLASS_LIST"
elif [ $# -gt 0 ]; then
    CLASS_NAMES=("$@")
else
    # No class names provided - load all from spec file
    echo "No class names provided, loading all classes from $SPEC_FILE"
    CLASS_NAMES=($(python3 -c "
import json
with open('tests/pytorch_tests/specs/$SPEC_FILE') as f:
    data = json.load(f)
for cfg in data.get('test_configurations', []):
    if cfg.get('class'):
        print(cfg['class'])
"))
fi

if [ ${#CLASS_NAMES[@]} -eq 0 ]; then
    echo "Error: No classes found"
    exit 1
fi

# Calculate CLASS_TIMEOUT from total timeout if provided
# Reserve 5 minutes (300s) for post-processing
if [ -n "$TOTAL_TIMEOUT" ]; then
    POST_PROCESSING_BUFFER=300
    AVAILABLE_TIME=$((TOTAL_TIMEOUT - POST_PROCESSING_BUFFER))
    NUM_CLASSES=${#CLASS_NAMES[@]}
    CLASS_TIMEOUT=$((AVAILABLE_TIME / NUM_CLASSES))
    echo "Auto-calculated CLASS_TIMEOUT: ${CLASS_TIMEOUT}s (total: ${TOTAL_TIMEOUT}s, classes: ${NUM_CLASSES}, buffer: ${POST_PROCESSING_BUFFER}s)"
fi

# Setup
mkdir -p "$OUTPUT_DIR"
export NEURON_LAUNCH_BLOCKING=1
export TORCH_NEURONX_MLIR_ATEN_OPS=1
export TORCH_NEURONX_ENABLE_PROLOGUE=0
export TEST_RUN_TIMESTAMP=${TEST_RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
export ATEN_SPEC_FILE="$SPEC_FILE"
cd "$BASE_PATH"

# Clone upstream PyTorch tests (skips if already exists with matching version)
python3 tests/pytorch_tests/clone_tests.py

echo "=== PyTorch Test Class Runner ==="
echo "Classes to run: ${CLASS_NAMES[*]}"
echo "Spec file: $SPEC_FILE"
echo "Workers per class: $NUM_WORKERS"
echo "Class timeout: ${CLASS_TIMEOUT}s, Test timeout: ${TEST_TIMEOUT}s"
echo "Output: $OUTPUT_DIR"
echo "================================="

FAILED_CLASSES=0
TOTAL_CLASSES=${#CLASS_NAMES[@]}

# Run each class sequentially
for CLASS_NAME in "${CLASS_NAMES[@]}"; do
    echo ""
    echo "=== Running class: $CLASS_NAME ==="

    REPORT_FILE="$OUTPUT_DIR/test_results_${CLASS_NAME}.json"
    LOG_FILE="$OUTPUT_DIR/${CLASS_NAME}.log"

    # Set class name filter
    export ATEN_CLASS_NAME="$CLASS_NAME"

    # Run pytest with per-test timeout and class-level safety timeout
    set +e
    timeout --signal=SIGINT --kill-after=30 $CLASS_TIMEOUT pytest -n "$NUM_WORKERS" \
        --json-report \
        --json-report-file="$REPORT_FILE" \
        --timeout=$TEST_TIMEOUT \
        --timeout-method=signal \
        --tb=short \
        --maxfail=0 \
        --continue-on-collection-errors \
        -v \
        "$SCRIPT_DIR/test_aten.py" 2>&1 | tee "$LOG_FILE"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    # Log warning if class timeout was hit (124=timeout, 130=SIGINT, 137=SIGKILL)
    if [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 130 ] || [ $EXIT_CODE -eq 137 ]; then
        echo "WARNING: Class $CLASS_NAME hit class timeout after ${CLASS_TIMEOUT}s (exit code: $EXIT_CODE)"
    fi

    if [ $EXIT_CODE -ne 0 ]; then
        echo "Class $CLASS_NAME had failures (exit code: $EXIT_CODE)"
        FAILED_CLASSES=$((FAILED_CLASSES + 1))
    fi

    echo "=== Completed: $CLASS_NAME ==="
    echo "Report: $REPORT_FILE"
done

# Summary
echo ""
echo "=== All Classes Complete ==="
echo "Total: $TOTAL_CLASSES, Failed: $FAILED_CLASSES"

if [ $FAILED_CLASSES -eq 0 ]; then
    echo "Status: SUCCESS"
else
    echo "Status: $FAILED_CLASSES/$TOTAL_CLASSES classes had failures"
fi

exit 0
