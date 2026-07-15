#!/usr/bin/env bash
# parallel_launch_template.sh — fan out many ModuScope (test x node) cells in
# parallel, ONE run per node (no contention). Derived from the prior-art template
# moduscope-component-tests/references/parallel-launch-template.sh, generalized to
# accept an explicit (node,testfile) pair list and a MODEL_CONFIG.
#
# Usage:
#   bash parallel_launch_template.sh <CLUSTER> <MODEL_CONFIG> <PAIRS_FILE> [N_PER_DIST]
#     CLUSTER     : 'gpu' or 'trn'
#     MODEL_CONFIG: e.g. deepseekv3_16b | qwen3_30b_a3b | gpt_oss_20b | moe | dense
#     PAIRS_FILE  : text file, one "<node> <testfile.py>" pair per line (# comments ok)
#     N_PER_DIST  : COMPONENT_TEST_N_PER_DIST (default 5; 1 = 7-cfg smoke)
#
# Get free nodes first from the moduscope-node-select skill; put one cell per
# free node into PAIRS_FILE. Requires fresh creds (source aws_creds.sh) and the
# correct production venv per cluster.
#
# Prints the launch timestamp TS and the log directory; runs are detached so this
# returns immediately. Use monitor_runs.sh "$TS" to follow them.

set -euo pipefail

CLUSTER="${1:?need cluster: gpu|trn}"
MODEL="${2:?need MODEL_CONFIG}"
PAIRS_FILE="${3:?need pairs file: lines of '<node> <testfile.py>'}"
N="${4:-5}"
TS="$(date +%Y%m%d_%H%M%S)"

[[ -f "$PAIRS_FILE" ]] || { echo "ERROR: pairs file not found: $PAIRS_FILE" >&2; exit 1; }

# --- cluster-specific locations (grounded in the briefing) ---
KEY="${KEY:-$SSH_KEY}"
TRN_REPO=$SHARED/<neuron-env>/ModuScope/test/component_tests_torchtitan
TRN_EVALS=$SHARED/<neuron-env>/NeuronTrainingEvaluations/src
TRN_VENV=$SHARED/<neuron-env>/private-torch-neuronx/.venv
GPU_REPO=$SHARED/e2e_root/ModuScope/test/component_tests_torchtitan
GPU_EVALS=$SHARED/e2e_root/NeuronTrainingEvaluations/src
GPU_VENV=$SHARED/gpu_baselines_venv

LOGDIR="$SHARED/moduscope_parallel_logs/${MODEL}_${CLUSTER}_${TS}"
mkdir -p "$LOGDIR"
echo "TS=$TS"
echo "LOGDIR=$LOGDIR"

launch_trn() {
  local node="$1" test="$2"
  local log="$LOGDIR/${test%.py}_${node}.log"
  echo "[trn] $test on $node -> $log"
  # setsid+nohup detaches from the ssh session so the run survives disconnect.
  ssh -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=no "$node" "setsid nohup bash -c '
    source $SHARED/aws_creds.sh
    source $TRN_VENV/bin/activate
    unset TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS
    export TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=0
    export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
    export REF_PATH=$SHARED/<neuron-env>/torchtitan
    export TARGET_PATH=$SHARED/deepseek_v3_optimized/torchtitan
    export PYTHONPATH=$TRN_REPO/shared:$TRN_EVALS:\$PYTHONPATH
    export TEST_MODE=neuron TARGET_DEVICE=neuron NEURON_RT_VISIBLE_CORES=0
    export MODEL_CONFIG=$MODEL COMPONENT_TEST_N_PER_DIST=$N
    cd $TRN_REPO/bf16/tests
    python -m pytest $test -c $SHARED/pytest_clean.ini -rs
  ' > '$log' 2>&1 < /dev/null &"
}

launch_gpu() {
  local node="$1" test="$2"
  local log="$LOGDIR/${test%.py}_${node}.log"
  echo "[gpu] $test on $node -> $log"
  # srun holds the allocation; nohup+& detaches the local waiter.
  nohup /opt/slurm/bin/srun -N 1 -w "$node" -p compute1 --no-kill bash -c "
    source $SHARED/aws_creds.sh
    source $GPU_VENV/bin/activate
    export REF_PATH=$SHARED/NTT_gpu_baselines
    export TARGET_PATH=$SHARED/NTT_gpu_baselines
    export PYTHONPATH=$GPU_REPO/shared:$GPU_EVALS:\$PYTHONPATH
    export TEST_MODE=gpu_save REF_DEVICE=cuda TARGET_DEVICE=cuda
    export MODEL_CONFIG=$MODEL COMPONENT_TEST_N_PER_DIST=$N
    cd $GPU_REPO/bf16/tests
    python -m pytest $test -c $SHARED/pytest_clean.ini -rs
  " > "$log" 2>&1 &
}

COUNT=0
while read -r node test _rest; do
  [[ -z "${node:-}" || "$node" == \#* ]] && continue   # skip blanks/comments
  [[ -z "${test:-}" ]] && { echo "WARN: no testfile for node $node, skipping" >&2; continue; }
  case "$CLUSTER" in
    trn) launch_trn "$node" "$test" ;;
    gpu) launch_gpu "$node" "$test" ;;
    *)   echo "ERROR: unknown cluster '$CLUSTER' (use gpu|trn)" >&2; exit 1 ;;
  esac
  COUNT=$((COUNT + 1))
done < "$PAIRS_FILE"

echo ""
echo "Launched $COUNT cell(s). TS=$TS"
echo "Monitor with:  bash monitor_runs.sh $LOGDIR"
