#!/bin/bash
# extract_csv.sh — Convert a TorchTitan run log to CSV (step,loss,grad_norm,tps,tflops,mfu)
#
# Usage: extract_csv.sh <log_path> <output.csv>
#
# Reads from TRN or GPU log file, parses [titan] step lines, dedupes by step, sorts.

set -u
IN=$1
OUT=$2

# Strip ANSI color codes; parse step/loss/grad_norm/tps/tflops/mfu fields
grep -E '^\[titan\].*step:' "$IN" 2>/dev/null \
  | sed -E 's/\x1b\[[0-9;]*m//g' \
  | awk -F'[: ,]+' '
    {
      step=loss=gn=mfu=tps=tflops=""
      for(i=1;i<=NF;i++){
        if($i=="step")      step=$(i+1)
        if($i=="loss")      loss=$(i+1)
        if($i=="grad_norm") gn=$(i+1)
        if($i=="tps")       tps=$(i+1)
        if($i=="tflops")    tflops=$(i+1)
        if($i=="mfu")       mfu=$(i+1)
      }
      if(step!="") print step","loss","gn","tps","tflops","mfu
    }' \
  | sort -t, -k1 -n -u \
  > "$OUT"

echo "$(wc -l < "$OUT") rows -> $OUT"
