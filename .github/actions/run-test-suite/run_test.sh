#!/bin/bash
# Run a test command and propagate its exit code.
#
# Usage:
#   TEST_COMMAND="your command" run_test.sh

set -euo pipefail

TEST_COMMAND="${TEST_COMMAND:-}"

if [ -z "$TEST_COMMAND" ]; then
    echo "ERROR: TEST_COMMAND environment variable is required"
    exit 1
fi

# Environment info
echo "============================================="
echo "       NEURON ENVIRONMENT INFO               "
echo "============================================="
python3 --version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" || echo "PyTorch not available"
python3 -c "import neuronxcc; print(f'neuronx-cc: {neuronxcc.__version__}')" || echo "neuronx-cc not available"
pip show neuronx-cc 2>/dev/null | grep -E "^(Name|Version|Location):" || true
# Neuron system packages (driver / runtime-lib / collectives / tools) + their
# versions. dpkg is preferred (no apt cache needed); fall back to apt list.
# Guarded so a no-match/absent tool can't abort under `set -euo pipefail`.
echo "--- Neuron system packages (dpkg/apt) ---"
{ dpkg -l 2>/dev/null | grep -iE 'neuron' \
    || apt list --installed 2>/dev/null | grep -iE 'neuron'; } \
    || echo "  (no Neuron system packages found)"
# Relevant Python packages only — neuron*, nki/nkilib, and torch* (torch,
# torch-xla, torch-neuronx, torch-mlir). Keeps the log focused vs a full
# `pip list`. `pip` resolves to the active venv (same one `pip show` uses).
# Guarded so a no-match / pip failure can't abort under `set -euo pipefail`.
echo "--- Python packages (neuron / nki / torch) ---"
{ pip list 2>/dev/null | grep -iE 'neuron|nki|torch'; } \
    || echo "  (no matching python packages found)"
neuron-ls || /opt/aws/neuron/bin/neuron-ls || echo "neuron-ls not available"
echo "============================================="

export NEURON_LAUNCH_BLOCKING=0

# Use embedded torch-mlir instead of any pre-installed wheel
pip uninstall -y neuron-torch-mlir 2>/dev/null || true
TORCH_MLIR_PYTHON="${PWD}/third_party/torch-mlir/python"
if [ ! -f "$TORCH_MLIR_PYTHON/torch_mlir/_mlir_libs/_mlir.so" ]; then
    TORCH_MLIR_PYTHON="/opt/torch-neuronx-pr/third_party/torch-mlir/python"
fi
if [ -f "$TORCH_MLIR_PYTHON/torch_mlir/_mlir_libs/_mlir.so" ]; then
    export PYTHONPATH="${TORCH_MLIR_PYTHON}:${PYTHONPATH:-}"
else
    # No compiled .so in any source tree; remove to avoid shadowing the wheel
    rm -rf "${PWD}/third_party/torch-mlir/python/torch_mlir"
fi
echo "=== torch_mlir source: $(python3 -c 'import torch_mlir; print(torch_mlir.__path__)' 2>&1) ==="

# Run tests
mkdir -p outputs/test-reports
TEST_EXIT=0
bash -c "$TEST_COMMAND" || TEST_EXIT=$?

# Copy to output directory. A cp failure means we lose test artifacts (logs,
# JSON reports, coverage) — surface that as a CI failure rather than letting
# a green run hide a broken artifact path.
CP_EXIT=0
cp -r outputs/ /ustore/workload_output_rw/test-results 2>&1 || CP_EXIT=$?
if [ "$CP_EXIT" -ne 0 ]; then
    echo "ERROR: Failed to copy outputs to workload storage (exit $CP_EXIT)" >&2
    if [ "$TEST_EXIT" -eq 0 ]; then
        TEST_EXIT=2
    fi
fi

echo "Test exit code: $TEST_EXIT"
exit "$TEST_EXIT"
