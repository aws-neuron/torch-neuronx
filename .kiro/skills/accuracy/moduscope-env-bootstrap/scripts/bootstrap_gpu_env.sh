#!/usr/bin/env bash
# bootstrap_gpu_env.sh — deterministically set up the GPU (the GPU cluster, CUDA) shell
# environment for ModuScope BF16 component tests in gpu_save mode.
#
# SOURCE this script (do NOT execute it) so the exports land in your shell:
#   source bootstrap_gpu_env.sh [REPO_ROOT] [MODEL_CONFIG]
#
# Params (positional, or pre-set as env vars):
#   REPO_ROOT     dir containing ModuScope/ and NeuronTrainingEvaluations/
#                 (default: $SHARED/e2e_root)
#   MODEL_CONFIG  registered torchtitan model (default: qwen3_30b_a3b)
#
# Optional env overrides:
#   VENV, REF_PATH, TARGET_PATH, COMPONENT_TEST_N_PER_DIST, PYTEST_INI, AWS_CREDS

# --- guard: must be sourced, not executed ---
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  echo "[bootstrap_gpu_env] ERROR: source this script, do not execute it:" >&2
  echo "    source ${BASH_SOURCE[0]} [REPO_ROOT] [MODEL_CONFIG]" >&2
  exit 1
fi

# --- parameters ---
REPO_ROOT="${1:-${REPO_ROOT:-$SHARED/e2e_root}}"
export MODEL_CONFIG="${2:-${MODEL_CONFIG:-qwen3_30b_a3b}}"
VENV="${VENV:-$SHARED/gpu_baselines_venv}"
AWS_CREDS="${AWS_CREDS:-$SHARED/aws_creds.sh}"
PYTEST_INI="${PYTEST_INI:-$SHARED/pytest_clean.ini}"

# GPU side: ref and target are the SAME tree (gpu_save only uploads, no compare).
export REF_PATH="${REF_PATH:-$SHARED/NTT_gpu_baselines}"
export TARGET_PATH="${TARGET_PATH:-$SHARED/NTT_gpu_baselines}"

# --- derived paths ---
TEST_DIR="${REPO_ROOT}/ModuScope/test/component_tests_torchtitan"
SHARED_DIR="${TEST_DIR}/shared"
NTE_SRC="${REPO_ROOT}/NeuronTrainingEvaluations/src"
export MODUSCOPE_TEST_DIR="${TEST_DIR}/bf16/tests"

# --- activate venv ---
if [ -f "${VENV}/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
else
  echo "[bootstrap_gpu_env] WARN: venv not found at ${VENV}/bin/activate" >&2
fi

# --- PYTHONPATH: shared + NeuronTrainingEvaluations/src first ---
export PYTHONPATH="${SHARED_DIR}:${NTE_SRC}:${PYTHONPATH}"

# --- gpu_save mode env (section 3/4 of briefing) ---
export TEST_MODE=gpu_save
export REF_DEVICE=cuda
export TARGET_DEVICE=cuda
export COMPONENT_TEST_N_PER_DIST="${COMPONENT_TEST_N_PER_DIST:-5}"

# --- AWS creds for S3 upload (rotatable STS, ~hours) ---
if [ -f "${AWS_CREDS}" ]; then
  # shellcheck disable=SC1090
  source "${AWS_CREDS}"
else
  echo "[bootstrap_gpu_env] WARN: AWS creds not found at ${AWS_CREDS} (S3 upload will fail)" >&2
fi

# --- drop the clean pytest config (ModuScope setup.cfg otherwise ignores this dir) ---
cat > "${PYTEST_INI}" <<'EOF'
[pytest]
addopts = -v -s --tb=short
testpaths = .
EOF
export PYTEST_CLEAN_INI="${PYTEST_INI}"

echo "[bootstrap_gpu_env] GPU env ready:"
echo "    MODEL_CONFIG=${MODEL_CONFIG}  TEST_MODE=${TEST_MODE}  N_PER_DIST=${COMPONENT_TEST_N_PER_DIST}"
echo "    REF_PATH=${REF_PATH}"
echo "    TARGET_PATH=${TARGET_PATH}"
echo "    test dir: ${MODUSCOPE_TEST_DIR}"
echo "    pytest ini: ${PYTEST_CLEAN_INI}"
echo "    next: bash $(dirname "${BASH_SOURCE[0]}")/verify_env.sh gpu"
