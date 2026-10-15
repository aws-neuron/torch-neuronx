#!/usr/bin/env bash
# verify_env.sh — verify the ModuScope component-test environment is correctly
# bootstrapped on the current cluster. Run AFTER sourcing bootstrap_gpu_env.sh
# or bootstrap_trn_env.sh (it reads the exported env vars).
#
# Usage:
#   bash verify_env.sh [gpu|trn]
#
# Cluster arg is optional; if omitted it is inferred from TEST_MODE
# (TEST_MODE=neuron -> trn, else gpu). Exit 0 only if every check passes.

set -u

CLUSTER="${1:-}"
if [ -z "${CLUSTER}" ]; then
  if [ "${TEST_MODE:-}" = "neuron" ] || [ "${TARGET_DEVICE:-}" = "neuron" ]; then
    CLUSTER="trn"
  else
    CLUSTER="gpu"
  fi
fi

REF_PATH="${REF_PATH:-}"
TARGET_PATH="${TARGET_PATH:-}"

fails=0
pass() { echo "  [PASS] $1"; }
fail() { echo "  [FAIL] $1"; fails=$((fails + 1)); }

echo "[verify_env] cluster=${CLUSTER}  python=$(command -v python || echo MISSING)"
echo "[verify_env] REF_PATH=${REF_PATH}"
echo "[verify_env] TARGET_PATH=${TARGET_PATH}"
echo "[verify_env] PYTHONPATH=${PYTHONPATH:-<unset>}"
echo

# 1. python + torch
if python -c "import torch; print('torch', torch.__version__)" 2>/dev/null; then
  pass "import torch"
else
  fail "import torch"
fi

# 2. torch_neuronx (TRN only)
if [ "${CLUSTER}" = "trn" ]; then
  if python -c "import torch_neuronx; print('torch_neuronx', torch_neuronx.__version__)" 2>/dev/null; then
    pass "import torch_neuronx"
  else
    fail "import torch_neuronx"
  fi
fi

# 3. torchtitan importable from REF_PATH
if [ -n "${REF_PATH}" ] && \
   REF_PATH="${REF_PATH}" python -c "
import os, sys
sys.path.insert(0, os.environ['REF_PATH'])
import torchtitan
print('torchtitan(ref)', torchtitan.__file__)
" 2>/dev/null; then
  pass "torchtitan importable from REF_PATH"
else
  fail "torchtitan importable from REF_PATH"
fi

# 4. torchtitan importable from TARGET_PATH
if [ -n "${TARGET_PATH}" ] && \
   TARGET_PATH="${TARGET_PATH}" python -c "
import os, sys
sys.path.insert(0, os.environ['TARGET_PATH'])
import torchtitan
print('torchtitan(target)', torchtitan.__file__)
" 2>/dev/null; then
  pass "torchtitan importable from TARGET_PATH"
else
  fail "torchtitan importable from TARGET_PATH"
fi

# 5. conftest_base importable (from shared/ on PYTHONPATH)
if python -c "import conftest_base; print('conftest_base', conftest_base.__file__)" 2>/dev/null; then
  pass "import conftest_base"
else
  fail "import conftest_base (is <repo>/shared on PYTHONPATH?)"
fi

# 6. compare_3tensors importable (from NeuronTrainingEvaluations/src on PYTHONPATH)
if python -c "from neuron_training_evaluations.utils.tensor_compare import compare_3tensors; print('compare_3tensors ok')" 2>/dev/null; then
  pass "import compare_3tensors"
else
  fail "import compare_3tensors (is NeuronTrainingEvaluations/src on PYTHONPATH?)"
fi

echo
if [ "${fails}" -eq 0 ]; then
  echo "[verify_env] ALL CHECKS PASSED for ${CLUSTER}"
  exit 0
else
  echo "[verify_env] ${fails} CHECK(S) FAILED for ${CLUSTER}"
  exit 1
fi
