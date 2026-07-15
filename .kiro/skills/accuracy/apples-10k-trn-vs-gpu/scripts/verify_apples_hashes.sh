#!/bin/bash
# verify_apples_hashes.sh — Verify cross-platform input hashes match for apples runs
#
# Usage: verify_apples_hashes.sh <trn_tag> <gpu_tag> [step]
#   step defaults to 1

set -u
TRN_TAG=$1
GPU_TAG=$2
STEP=${3:-1}

TMP_DIR=$(mktemp -d)
TRN_OUT=$TMP_DIR/trn_step${STEP}.txt
GPU_OUT=$TMP_DIR/gpu_step${STEP}.txt

# Pull all rank logs (try .rank0..N for multi-N, fallback to plain)
ssh -F $SSH_CONFIG $LOGIN_HOST "
> /tmp/__hashes_trn.txt
for r in 0 1 2 3 4 5 6 7; do
  L=$SHARED/logs/run_${TRN_TAG}.log.rank\$r
  [ -e \$L ] && grep 'INPUT_HASH' \$L 2>/dev/null | grep 'step=${STEP} ' >> /tmp/__hashes_trn.txt
done
[ ! -s /tmp/__hashes_trn.txt ] && grep 'INPUT_HASH' $SHARED/logs/run_${TRN_TAG}.log 2>/dev/null | grep 'step=${STEP} ' >> /tmp/__hashes_trn.txt
cat /tmp/__hashes_trn.txt
" > "$TRN_OUT"

ssh -i ~/cgk-p5en.pem -o StrictHostKeyChecking=no ubuntu@ec2-108-137-91-71.ap-southeast-3.compute.amazonaws.com "
> /tmp/__hashes_gpu.txt
for r in 0 1 2 3 4 5 6 7; do
  L=$SHARED/logs/run_${GPU_TAG}.log.rank\$r
  [ -e \$L ] && grep 'INPUT_HASH' \$L 2>/dev/null | grep 'step=${STEP} ' >> /tmp/__hashes_gpu.txt
done
[ ! -s /tmp/__hashes_gpu.txt ] && grep 'INPUT_HASH' $SHARED/logs/run_${GPU_TAG}.log 2>/dev/null | grep 'step=${STEP} ' >> /tmp/__hashes_gpu.txt
cat /tmp/__hashes_gpu.txt
" > "$GPU_OUT"

TRN_TOK=$(grep -oE 'tokens=\[[0-9, ]+\]' "$TRN_OUT" | sort -u)
GPU_TOK=$(grep -oE 'tokens=\[[0-9, ]+\]' "$GPU_OUT" | sort -u)

TRN_COUNT=$(echo "$TRN_TOK" | grep -c '.')
GPU_COUNT=$(echo "$GPU_TOK" | grep -c '.')
COMMON_COUNT=$(comm -12 <(echo "$TRN_TOK") <(echo "$GPU_TOK") | grep -c '.')

echo "═══════════════════════════════════════════════════════════════════════"
echo "Hash verification: TRN=$TRN_TAG vs GPU=$GPU_TAG at step=$STEP"
echo "═══════════════════════════════════════════════════════════════════════"
echo "TRN unique first-5-token sequences: $TRN_COUNT"
echo "GPU unique first-5-token sequences: $GPU_COUNT"
echo "Common between TRN and GPU: $COMMON_COUNT"
echo

if [ "$TRN_COUNT" -ge "$GPU_COUNT" ]; then
    SMALLER=$GPU_COUNT
    SMALLER_NAME="GPU"
else
    SMALLER=$TRN_COUNT
    SMALLER_NAME="TRN"
fi

if [ "$COMMON_COUNT" = "$SMALLER" ]; then
    echo "RESULT: PASS — All $SMALLER $SMALLER_NAME sequences appear in the larger set. APPLES VERIFIED."
    rm -rf "$TMP_DIR"
    exit 0
else
    echo "RESULT: FAIL — only $COMMON_COUNT of $SMALLER $SMALLER_NAME sequences match."
    echo "TRN-only:"
    comm -23 <(echo "$TRN_TOK") <(echo "$GPU_TOK") | head -5
    echo "GPU-only:"
    comm -13 <(echo "$TRN_TOK") <(echo "$GPU_TOK") | head -5
    rm -rf "$TMP_DIR"
    exit 1
fi
