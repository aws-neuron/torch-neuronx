#!/bin/bash
# launch_apples.sh — Canonical wrapper for launching apples 10k runs on TRN or GPU
#
# Usage:
#   launch_apples.sh PLATFORM TOPOLOGY MODEL MODE [GBS] [LBS] [STEPS]
#
# PLATFORM: trn | gpu
# TOPOLOGY: 1n | 4n | 8n      (8n only valid for gpu+70b)
# MODEL: llama3_8b | llama3_70b | qwen3_32b | deepseek_v3_16b
# MODE: eager | compile
# GBS: default 16 for 1n, 64 for 4n/8n
# LBS: default 1 (use 4 for gpu 70b 8n)
# STEPS: default 10000
#
# Outputs the launched TAG to stdout. Logs go to $SHARED/logs/run_<TAG>.log[.rank{0..N}]

set -u

PLATFORM=$1
TOPO=$2
MODEL=$3
MODE=$4
GBS=${5:-}
LBS=${6:-1}
STEPS=${7:-10000}

# Default GBS by topology
if [ -z "$GBS" ]; then
    case "$TOPO" in
        1n)        GBS=16 ;;
        4n|8n)     GBS=64 ;;
        *)         echo "Unknown topology: $TOPO" >&2; exit 1 ;;
    esac
fi

TS=$(date -u +%Y%m%d_%H%M%S)
TAG="${MODEL}_${TOPO}_${MODE}_${PLATFORM}_${TS}"
PORT=$((29770 + RANDOM % 100))

case "$PLATFORM:$TOPO:$MODEL" in
    trn:1n:*)
        # Use <neuron-env> TIM-style launcher for 1N TRN
        SCRIPT=$SHARED/scripts_10k/run_on_node_accuracy_moe_tim.sh
        NODE=<trn-node>
        DUMP=$SHARED/<neuron-env>/runs/${MODEL}_1n/${MODE}_${TS}
        SEED=""
        [ "$MODEL" = "llama3_8b" ] && SEED=$SHARED/llama3_8b_seed_checkpoint/step-0
        ssh -F $SSH_CONFIG $LOGIN_HOST "
          ssh -i $SSH_KEY -o StrictHostKeyChecking=no $NODE \"
            setsid nohup bash $SCRIPT $TAG $MODEL $MODE \"$SEED\" $GBS $DUMP $STEPS \\
              > $SHARED/logs/run_$TAG.bgout 2>&1 < /dev/null &
          \"
        "
        ;;

    trn:4n:llama3_8b)
        SCRIPT=$SHARED/scripts_10k/run_4n_apples_trn.sh
        SEED=$SHARED/llama3_8b_seed_checkpoint/step-0
        VARIANT="trn_${MODE}"
        DUMP=$SHARED/<neuron-env>/runs/llama3_8b_4n_apples/${MODE}_10k_${TS}
        MASTER=<trn-node>
        ssh -F $SSH_CONFIG $LOGIN_HOST "
          for i in 0 1 2 3; do
            NODE=\$(printf '<trn-node>' \$((1+i)))
            ssh -i $SSH_KEY -o StrictHostKeyChecking=no \$NODE \"
              setsid nohup bash $SCRIPT $TAG llama3 llama3_8b $VARIANT \$i $MASTER $PORT $SEED $GBS $DUMP $STEPS \\
                > $SHARED/logs/run_$TAG.bgout.rank\$i 2>&1 < /dev/null &
            \"
          done
        "
        ;;

    trn:4n:llama3_70b)
        SCRIPT=$SHARED/scripts_10k/run_4n_tim_70b_trn_v2.sh
        DUMP=$SHARED/<neuron-env>/runs/llama3_70b_4n_gbs${GBS}/${MODE}_10k_${TS}
        MASTER=<trn-node>
        # First 4 nodes are eager, second 4 are compile - or pass via base
        BASE=${BASE:-13}
        COMPILE_ARG=""
        [ "$MODE" = "compile" ] && COMPILE_ARG='--compile.enable --compile.backend=neuron'
        ssh -F $SSH_CONFIG $LOGIN_HOST "
          for i in 0 1 2 3; do
            NODE=\$(printf '<trn-node>' \$(($BASE+i)))
            ssh -i $SSH_KEY -o StrictHostKeyChecking=no \$NODE \"
              setsid nohup bash $SCRIPT $TAG \$i $MASTER $PORT $STEPS $DUMP $GBS $LBS '$COMPILE_ARG' \\
                > $SHARED/logs/run_$TAG.bgout.rank\$i 2>&1 < /dev/null &
            \"
          done
        "
        ;;

    gpu:1n:*)
        SCRIPT=$SHARED/scripts_10k/run_on_node_gpu_apples.sh
        NODE=<gpu-node>
        DUMP=$SHARED/clean-may18/runs/${MODEL}_1n/${MODE}_${TS}
        ssh -i ~/cgk-p5en.pem -o StrictHostKeyChecking=no ubuntu@ec2-108-137-91-71.ap-southeast-3.compute.amazonaws.com "
          ssh -o StrictHostKeyChecking=no $NODE \"
            setsid nohup bash $SCRIPT $TAG $MODEL $MODE $GBS $DUMP $STEPS \\
              > $SHARED/logs/run_$TAG.bgout 2>&1 < /dev/null &
          \"
        "
        ;;

    gpu:4n:llama3_8b)
        SCRIPT=$SHARED/scripts_10k/run_4n_apples_gpu.sh
        SEED=$SHARED/llama3_8b_seed_checkpoint/step-0
        VARIANT="gpu_${MODE}"
        DUMP=$SHARED/clean-may18/runs/llama3_8b_4n_apples/${MODE}_10k_${TS}
        MASTER=<gpu-node>
        ssh -i ~/cgk-p5en.pem -o StrictHostKeyChecking=no ubuntu@ec2-108-137-91-71.ap-southeast-3.compute.amazonaws.com "
          NODES=(1-13 1-14 1-15 1-16)
          for i in 0 1 2 3; do
            NODE=\${NODES[\$i]}
            ssh -o StrictHostKeyChecking=no <gpu-node> \"
              setsid nohup bash $SCRIPT $TAG llama3 llama3_8b $VARIANT \$i $MASTER $PORT $GBS $DUMP $STEPS $SEED \\
                > $SHARED/logs/run_$TAG.bgout.rank\$i 2>&1 < /dev/null &
            \"
          done
        "
        ;;

    gpu:8n:llama3_70b)
        SCRIPT=$SHARED/scripts_10k/run_8n_tim_70b_gpu_v2.sh
        DUMP=$SHARED/clean-may18/runs/llama3_70b_8n_gbs${GBS}/${MODE}_10k_lbs${LBS}_${TS}
        MASTER=<gpu-node>
        ssh -i ~/cgk-p5en.pem -o StrictHostKeyChecking=no ubuntu@ec2-108-137-91-71.ap-southeast-3.compute.amazonaws.com "
          NODES=(1-4 1-6 1-7 1-8 1-9 1-10 1-11 1-12)
          for i in 0 1 2 3 4 5 6 7; do
            NODE=\${NODES[\$i]}
            ssh -o StrictHostKeyChecking=no <gpu-node> \"
              setsid nohup bash $SCRIPT $TAG \$i $MASTER $PORT $STEPS $DUMP $GBS $LBS \\
                > $SHARED/logs/run_$TAG.bgout.rank\$i 2>&1 < /dev/null &
            \"
          done
        "
        ;;

    *)
        echo "Unknown combination: $PLATFORM:$TOPO:$MODEL" >&2
        exit 1
        ;;
esac

echo "TAG=$TAG"
echo "Logs: $SHARED/logs/run_${TAG}.log[.rank{0..N}]"
echo "Dump: $DUMP"
