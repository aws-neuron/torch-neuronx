#!/usr/bin/env bash
#
# run_trn_neuron.sh — launch a ModuScope component test on a TRN (Neuron) cluster
# compute node in the background, from the $LOGIN_HOST bastion.
#
# MODE=neuron      : download GPU goldens from S3, run target on Neuron, compare ratios
#                    (production cross-cluster path; requires gpu_save goldens on S3).
# MODE=neuron_cpu  : build the fp32+bf16 reference locally on CPU, run target on Neuron,
#                    compare. No S3 needed — fast TRN smoke test of a single test/dist.
#
# Run this FROM $LOGIN_HOST (it ssh-es to the compute node itself). It NEVER runs the
# test inline as a multi-line torchrun; it writes a self-contained runner to /shared
# and launches it with `setsid nohup ... </dev/null &` so it survives ssh logout.
#
# Parameters (all via env; NODE is required):
#   NODE         compute node, e.g. <trn-node>   (REQUIRED)
#   MODEL_CONFIG model name or group, e.g. deepseekv3_16b | qwen3_30b_a3b | moe   (default deepseekv3_16b)
#   TESTS        space-separated test files, e.g. "test_13_attention.py test_15_moe.py"
#                                                  (default "test_13_attention.py")
#   N_PER_DIST   COMPONENT_TEST_N_PER_DIST: 5 = full sweep (35 cfgs), 1 = smoke (7 cfgs)  (default 5)
#   MODE         neuron | neuron_cpu                                  (default neuron)
#
# Overridable paths (sensible the TRN cluster defaults baked in):
#   KEY          ssh key                  (default $SSH_KEY)
#   VENV         PRODUCTION torch_neuronx venv
#                (default $SHARED/<neuron-env>/private-torch-neuronx/.venv)
#   REPO         component_tests_torchtitan dir
#   REF_PATH     reference (mainline / plain MoE) torchtitan tree
#   TARGET_PATH  target (NKI-optimized) torchtitan tree
#   NTE_SRC      NeuronTrainingEvaluations/src
#   PYTEST_INI   clean pytest config       (default $SHARED/pytest_clean.ini)
#   LOG_DIR      where to write the run log (default $SHARED/moduscope_trn_logs)
#
set -euo pipefail

NODE="${NODE:?Set NODE, e.g. NODE=<trn-node>"
MODEL_CONFIG="${MODEL_CONFIG:-deepseekv3_16b}"
TESTS="${TESTS:-test_14_grouped_experts.py}"
N_PER_DIST="${N_PER_DIST:-5}"
MODE="${MODE:-neuron}"

case "$MODE" in
  neuron|neuron_cpu) ;;
  *) echo "ERROR: MODE must be 'neuron' or 'neuron_cpu' (got '$MODE')" >&2; exit 2 ;;
esac

KEY="${KEY:-$SSH_KEY}"
VENV="${VENV:-$SHARED/<neuron-env>/private-torch-neuronx/.venv}"
REPO="${REPO:-$SHARED/<neuron-env>/ModuScope/test/component_tests_torchtitan}"
REF_PATH="${REF_PATH:-$SHARED/<neuron-env>/torchtitan}"
TARGET_PATH="${TARGET_PATH:-$SHARED/deepseek_v3_optimized/torchtitan}"
NTE_SRC="${NTE_SRC:-$SHARED/<neuron-env>/NeuronTrainingEvaluations/src}"
PYTEST_INI="${PYTEST_INI:-$SHARED/pytest_clean.ini}"
LOG_DIR="${LOG_DIR:-$SHARED/moduscope_trn_logs}"

# WARNING guard: never use the old <incompatible-venv> venv (complex64 .conj() no-op bug
# → 50-90x bogus backward ratios). Production runs MUST use private-torch-neuronx.
case "$VENV" in
  *<incompatible-venv>*)
    echo "ERROR: VENV points at <incompatible-venv> (complex64 conj bug). Use private-torch-neuronx." >&2
    exit 3 ;;
esac

TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/trn_${MODE}_${MODEL_CONFIG}_${TS}.log"
RUNNER="${LOG_DIR}/.runner_${MODE}_${MODEL_CONFIG}_${TS}.sh"

mkdir -p "$LOG_DIR"

# Ensure a clean pytest.ini exists (ModuScope setup.cfg otherwise ignores this dir).
if [ ! -f "$PYTEST_INI" ]; then
  cat > "$PYTEST_INI" <<'INI'
[pytest]
addopts = -v -s --tb=short
testpaths = .
INI
fi

# S3-credential line only for production neuron mode (neuron_cpu uses a local CPU ref).
CREDS_LINE=""
if [ "$MODE" = "neuron" ]; then
  CREDS_LINE='source $SHARED/aws_creds.sh'
fi

# Write a self-contained remote runner to the shared FS ($LOGIN_HOST and the compute
# node both see /shared). Local vars are expanded here; \$PYTHONPATH stays literal so
# it expands on the remote node. This avoids nested ssh-quote hell.
cat > "$RUNNER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
${CREDS_LINE}
source "${VENV}/bin/activate"

# Allow complex/unimplemented ops to fall back to CPU (RoPE complex multiply etc.).
unset TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS || true
export TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=0
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

export REF_PATH="${REF_PATH}"
export TARGET_PATH="${TARGET_PATH}"
export PYTHONPATH="${REPO}/shared:${NTE_SRC}:\$PYTHONPATH"

export TEST_MODE="${MODE}"
export TARGET_DEVICE=neuron
export NEURON_RT_VISIBLE_CORES=0
export MODEL_CONFIG="${MODEL_CONFIG}"
export COMPONENT_TEST_N_PER_DIST="${N_PER_DIST}"

cd "${REPO}/bf16/tests"
echo "[run_trn_neuron] node=\$(hostname) mode=${MODE} model=${MODEL_CONFIG} tests='${TESTS}' N=${N_PER_DIST}"
set +e
python -m pytest ${TESTS} -c "${PYTEST_INI}" -rs
rc=\$?
set -e
echo "[run_trn_neuron] DONE rc=\$rc"
EOF
chmod +x "$RUNNER"

echo "Launching on ${NODE} (MODE=${MODE}, MODEL_CONFIG=${MODEL_CONFIG}, N_PER_DIST=${N_PER_DIST})"
echo "  runner: ${RUNNER}"
echo "  log   : ${LOG}"

# Background launch: setsid + nohup + </dev/null so it survives the ssh session.
ssh -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=no "$NODE" \
  "setsid nohup bash '${RUNNER}' > '${LOG}' 2>&1 </dev/null &"

echo "Launched. First NKI compile is 10-30 min — do NOT kill before you see [BACKWARD COMPARE] / compare lines."
echo "Tail with: tail -f '${LOG}'"
