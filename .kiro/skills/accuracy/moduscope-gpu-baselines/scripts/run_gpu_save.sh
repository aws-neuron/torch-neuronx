#!/usr/bin/env bash
#
# run_gpu_save.sh — allocate an the GPU cluster node and run ModuScope TEST_MODE=gpu_save:
# compute fp32+bf16 references on CUDA and UPLOAD goldens to S3. NO compare (the test
# early-returns after upload). Run this AFTER ssh-ing into the GPU login host:
#   ssh -i ~/cgk-p5en.pem ubuntu@ec2-108-137-91-71.ap-southeast-3.compute.amazonaws.com
#
# Usage:
#   NODE=<gpu-node> \
#   MODEL_CONFIG=deepseekv3_16b \
#   TESTS="test_15_moe.py test_13_attention.py" \
#   N_PER_DIST=1 \
#   ./run_gpu_save.sh
#
# Parameters (env vars):
#   NODE         GPU compute node (default: auto-pick first idle in compute1)
#   MODEL_CONFIG registered model or group: deepseekv3_16b|llama4_17bx16e|qwen3_30b_a3b|gpt_oss_20b|moe|dense (default: deepseekv3_16b)
#   TESTS        space-separated pytest file(s) under bf16/tests (default: test_15_moe.py)
#   N_PER_DIST   COMPONENT_TEST_N_PER_DIST: 1=smoke (7 cfgs), 5=full (35 cfgs) (default: 1)
#   REPO_ROOT    GPU-side checkout root containing ModuScope + NeuronTrainingEvaluations
#                (default: $SHARED/e2e_root)
#   NTT_PATH     torchtitan tree used as REF_PATH and TARGET_PATH (default: $SHARED/NTT_gpu_baselines)
#   PARTITION    slurm partition (default: compute1)
#   VENV         GPU baselines venv (default: $SHARED/gpu_baselines_venv)
#
# This script is meant to be run by a human on the GPU login host. It performs a LIVE
# srun allocation and AWS S3 upload — do NOT run it in author/validate mode.

set -euo pipefail

MODEL_CONFIG="${MODEL_CONFIG:-deepseekv3_16b}"
TESTS="${TESTS:-test_15_moe.py}"
N_PER_DIST="${N_PER_DIST:-1}"
REPO_ROOT="${REPO_ROOT:-$SHARED/e2e_root}"
NTT_PATH="${NTT_PATH:-$SHARED/NTT_gpu_baselines}"
PARTITION="${PARTITION:-compute1}"
VENV="${VENV:-$SHARED/gpu_baselines_venv}"
SRUN="/opt/slurm/bin/srun"
SINFO="/opt/slurm/bin/sinfo"

CT_DIR="${REPO_ROOT}/ModuScope/test/component_tests_torchtitan"
SHARED_DIR="${CT_DIR}/shared"
NTE_SRC="${REPO_ROOT}/NeuronTrainingEvaluations/src"
TESTS_DIR="${CT_DIR}/bf16/tests"
PYTEST_INI="$SHARED/pytest_clean.ini"

# Auto-pick an idle node if NODE not given.
if [[ -z "${NODE:-}" ]]; then
  NODE="$("${SINFO}" -p "${PARTITION}" -t idle -h -o '%n' | head -n1 || true)"
  if [[ -z "${NODE}" ]]; then
    echo "ERROR: no idle node in partition ${PARTITION}; set NODE explicitly (sinfo -p ${PARTITION})." >&2
    exit 1
  fi
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG="$SHARED/pytest_gpu_save_${MODEL_CONFIG}_${TS}.log"

# Ensure the clean pytest config exists (ModuScope setup.cfg otherwise --ignores this dir).
if [[ ! -f "${PYTEST_INI}" ]]; then
  cat > "${PYTEST_INI}" <<'INI'
[pytest]
addopts = -v -s --tb=short
testpaths = .
INI
fi

# Build the space-separated list of pytest invocations.
PYTEST_CMDS=""
for t in ${TESTS}; do
  PYTEST_CMDS+="python -m pytest ${t} -c ${PYTEST_INI} -rs || true"$'\n'
done

echo "Launching gpu_save: node=${NODE} model=${MODEL_CONFIG} n_per_dist=${N_PER_DIST}"
echo "Tests: ${TESTS}"
echo "Log:   ${LOG}"

nohup "${SRUN}" -N 1 -w "${NODE}" -p "${PARTITION}" --no-kill bash -c "
  set -euo pipefail
  source $SHARED/aws_creds.sh
  source ${VENV}/bin/activate
  export REF_PATH=${NTT_PATH} TARGET_PATH=${NTT_PATH}
  export PYTHONPATH=${SHARED_DIR}:${NTE_SRC}:\$PYTHONPATH
  export TEST_MODE=gpu_save REF_DEVICE=cuda TARGET_DEVICE=cuda
  export MODEL_CONFIG=${MODEL_CONFIG}
  export COMPONENT_TEST_N_PER_DIST=${N_PER_DIST}
  cd ${TESTS_DIR}
  ${PYTEST_CMDS}
" > "${LOG}" 2>&1 &

echo "PID $! — tail -f ${LOG}"
echo "After it finishes, verify uploaded goldens (do NOT need the node):"
echo "  source $SHARED/aws_creds.sh"
echo "  aws s3 ls s3://<bucket>/titan/modules/moe/${MODEL_CONFIG}/"
echo "  # grep the log for the exact prefix per dist:"
echo "  grep '\[S3 UPLOAD\]' ${LOG}"
