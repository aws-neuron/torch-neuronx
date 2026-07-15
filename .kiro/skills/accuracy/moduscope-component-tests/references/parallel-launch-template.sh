#!/usr/bin/env bash
# Template — launch multiple component tests in parallel across multiple nodes
# Usage:
#   bash parallel-launch-template.sh <CLUSTER> <MODEL_CONFIG> [N_PER_DIST]
# Examples:
#   bash parallel-launch-template.sh gpu deepseekv3_16b 5
#   bash parallel-launch-template.sh trn deepseekv3_16b 5
#
# Requires:
#   - aws_creds.sh fresh
#   - Both clusters' shared filesystem accessible from current shell

set -euo pipefail

CLUSTER="${1:-trn}"           # 'gpu' or 'trn'
MODEL="${2:-deepseekv3_16b}"
N="${3:-5}"
TS=$(date +%Y%m%d_%H%M%S)

# === EDIT THESE PER YOUR FREE NODES ===
TRN_NODES=(<trn-node> <trn-node> <trn-node>)
GPU_NODES=(<gpu-node> <gpu-node> <gpu-node>)

# Test files to run, one per available node (parallel)
TESTS=(test_09_grouped_experts.py test_11_moe.py test_12_mla_attention.py)

if [[ "$CLUSTER" == "trn" ]]; then
  KEY=$SSH_KEY

  for i in "${!TESTS[@]}"; do
    NODE="${TRN_NODES[$i]}"
    TEST="${TESTS[$i]}"
    LOG="$SHARED/mla_diag_logs/pytest_trn_${TEST%.py}_${TS}.log"

    echo "[trn] launching $TEST on $NODE → $LOG"

    ssh -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=no "$NODE" "setsid nohup bash -c '
      source $SHARED/aws_creds.sh
      source $SHARED/<neuron-env>/private-torch-neuronx/.venv/bin/activate
      unset TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS
      export TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=0
      export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
      export REF_PATH=$SHARED/<neuron-env>/torchtitan
      export TARGET_PATH=$SHARED/deepseek_v3_optimized/torchtitan
      export PYTHONPATH=$SHARED/<neuron-env>/ModuScope/test/component_tests_torchtitan/shared:$SHARED/<neuron-env>/NeuronTrainingEvaluations/src:\$PYTHONPATH
      export TEST_MODE=neuron TARGET_DEVICE=neuron
      export MODEL_CONFIG=$MODEL COMPONENT_TEST_N_PER_DIST=$N NEURON_RT_VISIBLE_CORES=0
      cd $SHARED/<neuron-env>/ModuScope/test/component_tests_torchtitan/bf16/tests
      python -m pytest $TEST -c $SHARED/pytest_clean.ini -rs
    ' > '$LOG' 2>&1 < /dev/null &"
  done

elif [[ "$CLUSTER" == "gpu" ]]; then
  for i in "${!TESTS[@]}"; do
    NODE="${GPU_NODES[$i]}"
    TEST="${TESTS[$i]}"
    LOG="$SHARED/e2e_root/pytest_gpu_${TEST%.py}_${TS}.log"

    echo "[gpu] launching $TEST on $NODE → $LOG"

    nohup /opt/slurm/bin/srun -N 1 -w "$NODE" -p compute1 --no-kill bash -c "
      source $SHARED/aws_creds.sh
      source $SHARED/gpu_baselines_venv/bin/activate
      export REF_PATH=$SHARED/NTT_gpu_baselines
      export TARGET_PATH=$SHARED/NTT_gpu_baselines
      export PYTHONPATH=$SHARED/e2e_root/ModuScope/test/component_tests_torchtitan/shared:$SHARED/e2e_root/NeuronTrainingEvaluations/src:\$PYTHONPATH
      export TEST_MODE=gpu_save REF_DEVICE=cuda TARGET_DEVICE=cuda
      export MODEL_CONFIG=$MODEL COMPONENT_TEST_N_PER_DIST=$N
      cd $SHARED/e2e_root/ModuScope/test/component_tests_torchtitan/bf16/tests
      python -m pytest $TEST -c $SHARED/pytest_clean.ini -rs
    " > "$LOG" 2>&1 &
  done
else
  echo "Unknown cluster: $CLUSTER. Use 'gpu' or 'trn'."
  exit 1
fi

echo ""
echo "All launched. TS=$TS"
echo "Tail logs with:"
echo "  ls $SHARED/{mla_diag_logs,e2e_root}/pytest_*_${TS}.log"
echo ""
echo "Wait for completion:"
echo "  while [ \"\$(ssh -i \$KEY \$NODE 'ps -fu $USER | grep -v grep | grep -E pytest | wc -l')\" -gt 0 ]; do sleep 60; done"
