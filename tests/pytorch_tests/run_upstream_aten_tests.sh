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

# Setup
mkdir -p "$OUTPUT_DIR"
export TORCH_NEURONX_SYNC_MODE=1
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

    # Set class name filter for test_aten.py
    export ATEN_CLASS_NAME="$CLASS_NAME"

    # Run pytest for single class
    # TEST_TIMEOUT: timeout per individual test in seconds (default 120s = 2 min)
    pytest -n "$NUM_WORKERS" \
        --json-report \
        --json-report-file="$REPORT_FILE" \
        --tb=short \
        --maxfail=0 \
        --continue-on-collection-errors \
        --timeout=${TEST_TIMEOUT:-120} \
        --timeout_method=signal \
        -v \
        "$SCRIPT_DIR/test_aten.py" 2>&1 | tee "$LOG_FILE" || {
            echo "Class $CLASS_NAME had failures"
            FAILED_CLASSES=$((FAILED_CLASSES + 1))
        }

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
