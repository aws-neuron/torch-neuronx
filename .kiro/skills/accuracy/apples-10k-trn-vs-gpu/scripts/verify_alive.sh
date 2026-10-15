#!/bin/bash
# verify_alive.sh — Quick liveness check for all active runs in active_runs_DO_NOT_KILL.json
#
# Outputs procs count and last log timestamp per run, useful to detect if any went silent.

set -u
RUNS_JSON=${1:-$WORKSPACE/active_runs_DO_NOT_KILL.json}

if [ ! -f "$RUNS_JSON" ]; then
    echo "ERROR: $RUNS_JSON not found" >&2
    exit 1
fi

echo "═══════════════════════════════════════════════════════════════════════"
echo "Liveness check at $(date -u +%FT%TZ)"
echo "═══════════════════════════════════════════════════════════════════════"

# Parse JSON (jq optional)
TAGS=$(grep -oE '"tag": *"[^"]+"' "$RUNS_JSON" | sed 's/"tag": *"//; s/"//')
IDS=$(grep -oE '"id": *"[^"]+"' "$RUNS_JSON" | sed 's/"id": *"//; s/"//')
PLATS=$(grep -oE '"platform": *"[^"]+"' "$RUNS_JSON" | sed 's/"platform": *"//; s/"//')
mapfile -t TAG_LIST <<< "$TAGS"
mapfile -t ID_LIST <<< "$IDS"
mapfile -t PLAT_LIST <<< "$PLATS"

# Get first node per run from JSON (rough parse)
NODES_RAW=$(awk '/"nodes":/{f=1; next} f && /\]/{f=0} f && /compute/' "$RUNS_JSON" | grep -oE 'compute[A-Za-z0-9-]+' | head -${#TAG_LIST[@]})
mapfile -t NODE_LIST <<< "$NODES_RAW"

for i in "${!TAG_LIST[@]}"; do
    TAG="${TAG_LIST[$i]}"
    ID="${ID_LIST[$i]}"
    PLAT="${PLAT_LIST[$i]}"
    NODE="${NODE_LIST[$i]}"

    # Pick log path
    if echo "$ID" | grep -q '_1n_'; then
        LOG=$SHARED/logs/run_${TAG}.log
    else
        LOG=$SHARED/logs/run_${TAG}.log.rank0
    fi

    if [ "$PLAT" = "TRN" ]; then
        STATS=$(ssh -F $SSH_CONFIG $LOGIN_HOST "
            P=\$(ssh -i $SSH_KEY -o ConnectTimeout=4 -o StrictHostKeyChecking=no $NODE 'pgrep -u $USER -f titan_train_wrapper | wc -l' 2>/dev/null)
            STEP=\$(grep -oE 'step:[ ]+[0-9]+' $LOG 2>/dev/null | tail -1 | grep -oE '[0-9]+')
            TS=\$(grep -E '^\\[titan\\].*step:' $LOG 2>/dev/null | tail -1 | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}')
            echo \"\$P|\$STEP|\$TS\"
        " 2>/dev/null)
    else
        STATS=$(ssh -i ~/cgk-p5en.pem -o StrictHostKeyChecking=no -o ConnectTimeout=10 ubuntu@ec2-108-137-91-71.ap-southeast-3.compute.amazonaws.com "
            P=\$(ssh -o ConnectTimeout=4 -o StrictHostKeyChecking=no $NODE 'pgrep -u ubuntu -f titan_train_wrapper | wc -l' 2>/dev/null)
            STEP=\$(grep -oE 'step:[ ]+[0-9]+' $LOG 2>/dev/null | tail -1 | grep -oE '[0-9]+')
            TS=\$(grep -E '^\\[titan\\].*step:' $LOG 2>/dev/null | tail -1 | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}')
            echo \"\$P|\$STEP|\$TS\"
        " 2>/dev/null)
    fi
    IFS='|' read -r PROCS STEP TS <<< "$STATS"

    EXP_PROCS=$(echo "$ID" | grep -q "70b_8n" && echo 10 || echo 66)
    [ "$PLAT" = "GPU" ] && EXP_PROCS=10

    if [ "$PROCS" = "$EXP_PROCS" ]; then
        STATUS=ALIVE
    elif [ "$PROCS" = "0" ] || [ -z "$PROCS" ]; then
        STATUS=DEAD
    else
        STATUS="DEGRADED($PROCS/$EXP_PROCS)"
    fi

    printf '  %-26s  procs=%-6s step=%-6s last_ts=%s  %s\n' "$ID" "${PROCS:-?}" "${STEP:-?}" "$TS" "$STATUS"
done
