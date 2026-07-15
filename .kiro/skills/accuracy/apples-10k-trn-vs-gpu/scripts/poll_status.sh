#!/bin/bash
# poll_status.sh — Status table of all 5/6 active 10k runs with delta-from-prev-poll
#
# Reads tags from $WORKSPACE/active_runs_DO_NOT_KILL.json (or fallback list).
# Caches previous step counts in /tmp/.apples_poll_state for delta computation.
#
# Output:
#                              nodes              prev → now    Δsteps  rate(/min)  loss   MFU     ETA→10k
#   TRN  8B 4N compile         3-1..3-4           4794 → 5267   +473    21.5        0.99   27.0%   3.7h
#   ...

set -u
STATE=/tmp/.apples_poll_state
NOW_EPOCH=$(date -u +%s)
NOW_ISO=$(date -u +%FT%TZ)

# Use jq if available, else fallback to grep
RUNS_JSON=${1:-$WORKSPACE/active_runs_DO_NOT_KILL.json}

if [ ! -f "$RUNS_JSON" ]; then
    echo "ERROR: $RUNS_JSON not found. Run snapshot_active_runs.sh first." >&2
    exit 1
fi

# Load previous state
PREV_EPOCH=0
declare -A PREV_STEP
if [ -f "$STATE" ]; then
    while IFS='|' read -r KEY VAL; do
        [ "$KEY" = "_epoch" ] && PREV_EPOCH=$VAL && continue
        PREV_STEP[$KEY]=$VAL
    done < "$STATE"
fi
ELAPSED_SEC=$((NOW_EPOCH - PREV_EPOCH))
ELAPSED_MIN=$((ELAPSED_SEC / 60))

echo "  Now: $NOW_ISO  (prev poll +${ELAPSED_MIN}min)"
printf '%-26s  %-18s  %-18s  %-8s  %-10s  %-7s  %-7s  %s\n' \
    "run" "nodes" "prev → now" "Δsteps" "rate(/min)" "loss" "MFU" "ETA→10k"
echo "──────────────────────────────────────────────────────────────────────────────────────────────────────────────"

> "$STATE.new"
echo "_epoch|$NOW_EPOCH" >> "$STATE.new"

# Parse JSON manually (jq optional)
TAGS=$(grep -oE '"tag": *"[^"]+"' "$RUNS_JSON" | sed 's/"tag": *"//; s/"//')
IDS=$(grep -oE '"id": *"[^"]+"' "$RUNS_JSON" | sed 's/"id": *"//; s/"//')
PLATFORMS=$(grep -oE '"platform": *"[^"]+"' "$RUNS_JSON" | sed 's/"platform": *"//; s/"//')

mapfile -t TAG_LIST <<< "$TAGS"
mapfile -t ID_LIST <<< "$IDS"
mapfile -t PLAT_LIST <<< "$PLATFORMS"

for i in "${!TAG_LIST[@]}"; do
    TAG="${TAG_LIST[$i]}"
    ID="${ID_LIST[$i]}"
    PLAT="${PLAT_LIST[$i]}"

    # Pick log file (1N has plain log, multi-N has .rank0)
    if echo "$ID" | grep -q '_1n_'; then
        LOG=$SHARED/logs/run_${TAG}.log
    else
        LOG=$SHARED/logs/run_${TAG}.log.rank0
    fi

    # Fetch latest step,loss,mfu
    if [ "$PLAT" = "TRN" ]; then
        LATEST=$(ssh -F $SSH_CONFIG $LOGIN_HOST "tail -200 $LOG 2>/dev/null | grep -E '\\[titan\\].*step:' | tail -1 | sed -E 's/\\x1b\\[[0-9;]*m//g'" 2>/dev/null)
    else
        LATEST=$(ssh -i ~/cgk-p5en.pem -o StrictHostKeyChecking=no -o ConnectTimeout=10 ubuntu@ec2-108-137-91-71.ap-southeast-3.compute.amazonaws.com "tail -200 $LOG 2>/dev/null | grep -E '\\[titan\\].*step:' | tail -1 | sed -E 's/\\x1b\\[[0-9;]*m//g'" 2>/dev/null)
    fi

    STEP=$(echo "$LATEST" | grep -oE 'step:[ ]+[0-9]+' | grep -oE '[0-9]+')
    LOSS=$(echo "$LATEST" | grep -oE 'loss:[ ]+[0-9.]+' | grep -oE '[0-9.]+')
    MFU=$(echo "$LATEST" | grep -oE 'mfu:[ ]+[0-9.]+%')

    if [ -z "$STEP" ]; then
        printf '%-26s  no log data\n' "$ID"
        continue
    fi

    PREV=${PREV_STEP[$ID]:-0}
    DELTA=$((STEP - PREV))
    if [ "$ELAPSED_MIN" -gt 0 ] && [ "$PREV" -gt 0 ]; then
        RATE=$(echo "scale=2; $DELTA / $ELAPSED_MIN" | bc)
        if [ "$DELTA" -gt 0 ]; then
            ETA_MIN=$(echo "scale=1; (10000 - $STEP) / $RATE" | bc)
            ETA_H=$(echo "scale=1; $ETA_MIN / 60" | bc)
            ETA="${ETA_H}h"
        else
            ETA="—"
        fi
    else
        RATE="—"
        ETA="—"
    fi

    # Pull nodes and topology hint from id
    NODES=$(grep -oE '"nodes": *\[[^]]+\]' "$RUNS_JSON" | head -$((i+1)) | tail -1 | grep -oE 'compute[A-Za-z0-9-]+' | sed 's/.*-\([0-9]\+-[0-9]\+\)/\1/' | head -8 | tr '\n' ',' | sed 's/,$//' | head -c 28)

    printf '%-26s  %-18s  %5s → %-5s  +%-6s  %-10s  %-7s  %-7s  %s\n' \
        "$ID" "$NODES" "$PREV" "$STEP" "$DELTA" "$RATE" "$LOSS" "${MFU#mfu: }" "$ETA"

    echo "$ID|$STEP" >> "$STATE.new"
done

mv "$STATE.new" "$STATE"
