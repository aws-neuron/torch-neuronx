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
neuron-ls || /opt/aws/neuron/bin/neuron-ls || echo "neuron-ls not available"
echo "============================================="

export NEURON_LAUNCH_BLOCKING=0

# Run tests
mkdir -p outputs/test-reports
TEST_EXIT=0
bash -c "$TEST_COMMAND" || TEST_EXIT=$?

# Copy to output directory
cp -r outputs/ /ustore/workload_output_rw/test-results 2>&1 || echo "WARNING: Failed to copy outputs to workload storage"

echo "Test exit code: $TEST_EXIT"
exit "$TEST_EXIT"
