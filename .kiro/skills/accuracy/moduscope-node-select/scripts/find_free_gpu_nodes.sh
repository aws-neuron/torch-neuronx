#!/usr/bin/env bash
# find_free_gpu_nodes.sh — list idle GPU nodes on the GPU cluster compute1 partition.
#
# GPU nodes (<gpu-node>]) are allocated via Slurm, so
# Slurm's own view IS the truth: a node is free iff it is in state 'idle' and
# has no jobs in squeue. No ssh probe needed (unlike TRN reserved nodes).
#
# Run this FROM the GPU login host:
#   ssh -i ~/cgk-p5en.pem ubuntu@ec2-108-137-91-71.ap-southeast-3.compute.amazonaws.com
#
# Usage:  bash find_free_gpu_nodes.sh [PARTITION]
#   PARTITION defaults to compute1.
# Output: one verified-free node name per line on stdout; diagnostics on stderr.

set -euo pipefail

PARTITION="${1:-compute1}"
SINFO=/opt/slurm/bin/sinfo
SQUEUE=/opt/slurm/bin/squeue

[[ -x "$SINFO" ]]  || { echo "ERROR: $SINFO not found (run on GPU login host)" >&2; exit 1; }
[[ -x "$SQUEUE" ]] || { echo "ERROR: $SQUEUE not found (run on GPU login host)" >&2; exit 1; }

echo "== sinfo idle nodes in partition '$PARTITION' ==" >&2
# -h: no header, -o %N: node list, -t idle: only fully idle nodes.
IDLE_RAW="$("$SINFO" -h -p "$PARTITION" -t idle -o '%N' || true)"

if [[ -z "$IDLE_RAW" ]]; then
  echo "No idle nodes reported by sinfo in partition '$PARTITION'." >&2
  exit 0
fi

# Expand the Slurm hostlist (e.g. <gpu-node>,5]) into
# individual names using scontrol if available; fall back to raw token split.
if command -v /opt/slurm/bin/scontrol >/dev/null 2>&1; then
  IDLE_NODES="$(/opt/slurm/bin/scontrol show hostnames "$IDLE_RAW")"
else
  IDLE_NODES="$(echo "$IDLE_RAW" | tr ',' '\n')"
fi

# Nodes that currently have ANY squeue job (running or pending allocation).
BUSY_NODES="$("$SQUEUE" -h -p "$PARTITION" -o '%N' | tr ',' '\n' | sort -u || true)"

echo "== verified-free GPU nodes ==" >&2
COUNT=0
while IFS= read -r node; do
  [[ -z "$node" ]] && continue
  if echo "$BUSY_NODES" | grep -qx "$node"; then
    echo "  busy   : $node" >&2
  else
    echo "$node"            # machine-readable: free node to stdout
    echo "  FREE   : $node" >&2
    COUNT=$((COUNT + 1))
  fi
done <<< "$IDLE_NODES"

echo "== $COUNT free GPU node(s) ==" >&2
