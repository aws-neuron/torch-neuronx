#!/bin/bash
# snapshot_active_runs.sh — Generate JSON manifest of currently running 10k jobs
#
# Walks TRN nodes 3-1..3-16 and GPU nodes 1-1..1-16, identifies titan_train_wrapper
# processes by their tag (extracted from --dump-folder) and emits a JSON manifest.
#
# Usage: snapshot_active_runs.sh [output.json]   (default: stdout)

set -u
OUT=${1:-/dev/stdout}

NOW=$(date -u +%FT%TZ)
{
    echo '{'
    echo "  \"snapshot_utc\": \"$NOW\","
    echo '  "policy": "DO NOT KILL",'
    echo '  "active_runs": ['
} > "$OUT"

FIRST=1

# TRN side
ssh -F $SSH_CONFIG $LOGIN_HOST "
for n in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16; do
  RES=\$(ssh -i $SSH_KEY -o ConnectTimeout=4 -o StrictHostKeyChecking=no <trn-node> 'ps -ef --no-headers -u $USER 2>/dev/null | grep titan_train_wrapper | grep -v grep | head -1' 2>/dev/null)
  if [ -n \"\$RES\" ]; then
    TAG=\$(echo \"\$RES\" | grep -oE 'dump-folder=[^ ]+' | sed 's|.*/||')
    GBS=\$(echo \"\$RES\" | grep -oE 'global_batch_size=[0-9]+' | head -1 | sed 's/global_batch_size=//')
    CMP=\$(echo \"\$RES\" | grep -cE 'compile.enable')
    MODE=eager; [ \"\$CMP\" = \"1\" ] && MODE=compile
    MODEL=\$(echo \"\$RES\" | grep -oE 'config llama3:llama3_[0-9]+b|config qwen3:qwen3_[0-9]+b|config simple_fsdp.deepseek[^ ]+' | sed 's/config //')
    echo \"3-\$n|TRN|\$MODEL|\$MODE|\$GBS|\$TAG\"
  fi
done
" 2>/dev/null | while IFS='|' read -r NODE PLAT MODEL MODE GBS TAG; do
    [ -z "$TAG" ] && continue
    if [ "$FIRST" = "1" ]; then FIRST=0; else echo ',' >> "$OUT"; fi
    cat >> "$OUT" <<EOF
    {
      "node": "<trn-node>",
      "platform": "$PLAT",
      "model": "$MODEL",
      "mode": "$MODE",
      "gbs": $GBS,
      "tag": "$TAG"
    }
EOF
done

# GPU side
ssh -i ~/cgk-p5en.pem -o StrictHostKeyChecking=no ubuntu@ec2-108-137-91-71.ap-southeast-3.compute.amazonaws.com "
for n in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16; do
  RES=\$(ssh -o ConnectTimeout=4 -o StrictHostKeyChecking=no <gpu-node> 'ps -ef --no-headers -u ubuntu 2>/dev/null | grep titan_train_wrapper | grep -v grep | head -1' 2>/dev/null)
  if [ -n \"\$RES\" ]; then
    TAG=\$(echo \"\$RES\" | grep -oE 'dump-folder=[^ ]+' | sed 's|.*/||')
    GBS=\$(echo \"\$RES\" | grep -oE 'global_batch_size=[0-9]+' | head -1 | sed 's/global_batch_size=//')
    CMP=\$(echo \"\$RES\" | grep -cE 'compile.enable')
    MODE=eager; [ \"\$CMP\" = \"1\" ] && MODE=compile
    MODEL=\$(echo \"\$RES\" | grep -oE 'config llama3:llama3_[0-9]+b|config qwen3:qwen3_[0-9]+b|config simple_fsdp.deepseek[^ ]+' | sed 's/config //')
    echo \"1-\$n|GPU|\$MODEL|\$MODE|\$GBS|\$TAG\"
  fi
done
" 2>/dev/null | while IFS='|' read -r NODE PLAT MODEL MODE GBS TAG; do
    [ -z "$TAG" ] && continue
    if [ "$FIRST" = "1" ]; then FIRST=0; else echo ',' >> "$OUT"; fi
    cat >> "$OUT" <<EOF
    {
      "node": "<gpu-node>",
      "platform": "$PLAT",
      "model": "$MODEL",
      "mode": "$MODE",
      "gbs": $GBS,
      "tag": "$TAG"
    }
EOF
done

{
    echo ''
    echo '  ]'
    echo '}'
} >> "$OUT"
