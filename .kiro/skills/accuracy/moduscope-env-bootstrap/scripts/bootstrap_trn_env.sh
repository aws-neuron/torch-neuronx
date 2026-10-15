#!/usr/bin/env bash
# bootstrap_trn_env.sh — deterministically set up the TRN (the TRN cluster, Neuron) shell
# environment for ModuScope BF16 component tests in neuron (cross-cluster compare) mode.
#
# SOURCE this script (do NOT execute it) so the exports land in your shell:
#   source bootstrap_trn_env.sh [REPO_ROOT] [MODEL_CONFIG]
#
# Params (positional, or pre-set as env vars):
#   REPO_ROOT     dir containing ModuScope/ and NeuronTrainingEvaluations/
#                 (default: $SHARED/<neuron-env>)
#   MODEL_CONFIG  registered torchtitan model (default: qwen3_30b_a3b)
#
# Optional env overrides:
#   VENV, REF_PATH, TARGET_PATH, COMPONENT_TEST_N_PER_DIST, PYTEST_INI, AWS_CREDS
#
# CRITICAL VENV NOTE:
#   Use the PRODUCTION venv $SHARED/<neuron-env>/private-torch-neuronx/.venv
#   (torch_neuronx 2.11.3.x — complex64 .conj() no-op fix landed here).
#   DO NOT use $SHARED/<incompatible-venv>/torch-neuronx/.venv (torch_neuronx 0.1.0):
#   its complex64 .conj() bug corrupts RoPE/MLA backward and produces 50-90x bwd ratios
#   (every attention/block test fails). This script hard-defaults to the production venv.

# --- guard: must be sourced, not executed ---
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  echo "[bootstrap_trn_env] ERROR: source this script, do not execute it:" >&2
  echo "    source ${BASH_SOURCE[0]} [REPO_ROOT] [MODEL_CONFIG]" >&2
  exit 1
fi

# --- parameters ---
REPO_ROOT="${1:-${REPO_ROOT:-$SHARED/<neuron-env>}}"
export MODEL_CONFIG="${2:-${MODEL_CONFIG:-qwen3_30b_a3b}}"
VENV="${VENV:-$SHARED/<neuron-env>/private-torch-neuronx/.venv}"
AWS_CREDS="${AWS_CREDS:-$SHARED/aws_creds.sh}"
PYTEST_INI="${PYTEST_INI:-$SHARED/pytest_clean.ini}"

# Guard against the buggy <incompatible-venv> venv (complex64 conj bug).
case "${VENV}" in
  *<incompatible-venv>*)
    echo "[bootstrap_trn_env] WARN: VENV points at <incompatible-venv> (complex64 conj bug -> 50-90x ratios)." >&2
    echo "                   Override with the production private-torch-neuronx venv." >&2
    ;;
esac

# TRN side: ref = mainline (ootb) tree, target = NKI-optimized tree.
export REF_PATH="${REF_PATH:-$SHARED/<neuron-env>/torchtitan}"
export TARGET_PATH="${TARGET_PATH:-$SHARED/deepseek_v3_optimized/torchtitan}"

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
  echo "[bootstrap_trn_env] WARN: venv not found at ${VENV}/bin/activate" >&2
fi

# --- PYTHONPATH: shared + NeuronTrainingEvaluations/src first ---
export PYTHONPATH="${SHARED_DIR}:${NTE_SRC}:${PYTHONPATH}"

# --- neuron mode env (section 4/5 of briefing) ---
export TEST_MODE=neuron
export TARGET_DEVICE=neuron
export NEURON_RT_VISIBLE_CORES=0
export COMPONENT_TEST_N_PER_DIST="${COMPONENT_TEST_N_PER_DIST:-5}"

# Allow CPU fallback for unimplemented/complex ops (must be 0, not 1).
unset TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS
export TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=0

# Avoid the broken jenkins pytest plugin autoload.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

# --- AWS creds for S3 download (rotatable STS, ~hours) ---
if [ -f "${AWS_CREDS}" ]; then
  # shellcheck disable=SC1090
  source "${AWS_CREDS}"
else
  echo "[bootstrap_trn_env] WARN: AWS creds not found at ${AWS_CREDS} (S3 download will fail)" >&2
fi

# --- drop the clean pytest config (ModuScope setup.cfg otherwise ignores this dir) ---
cat > "${PYTEST_INI}" <<'EOF'
[pytest]
addopts = -v -s --tb=short
testpaths = .
EOF
export PYTEST_CLEAN_INI="${PYTEST_INI}"

echo "[bootstrap_trn_env] TRN env ready:"
echo "    MODEL_CONFIG=${MODEL_CONFIG}  TEST_MODE=${TEST_MODE}  N_PER_DIST=${COMPONENT_TEST_N_PER_DIST}"
echo "    VENV=${VENV}"
echo "    REF_PATH=${REF_PATH}"
echo "    TARGET_PATH=${TARGET_PATH}"
echo "    NEURON_RT_VISIBLE_CORES=${NEURON_RT_VISIBLE_CORES}  FALLBACK_ONLY=${TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS}"
echo "    test dir: ${MODUSCOPE_TEST_DIR}"
echo "    pytest ini: ${PYTEST_CLEAN_INI}"
echo "    next: bash $(dirname "${BASH_SOURCE[0]}")/verify_env.sh trn"
