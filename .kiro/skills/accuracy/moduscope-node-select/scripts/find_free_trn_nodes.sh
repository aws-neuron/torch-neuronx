#!/usr/bin/env bash
# find_free_trn_nodes.sh — list TRULY free TRN nodes on the TRN cluster compute2.
#
# TRN nodes (<trn-node>]) sit in a Slurm reservation, so
# sinfo/squeue show them as 'resv' — that is EXPECTED and does NOT mean busy.
# Real occupancy MUST be probed directly: a node is free iff it runs no python
# process owned by $USER.
#
# Run this FROM the TRN cluster login bastion (after `ssh $LOGIN_HOST`).
#
# Usage:  bash find_free_trn_nodes.sh [FIRST] [LAST]
#   FIRST/LAST = node index range (default 1..16).
# Output: one verified-free node name per line on stdout; diagnostics on stderr.

set -euo pipefail

FIRST="${1:-1}"
LAST="${2:-16}"
KEY="${KEY:-$SSH_KEY}"
PREFIX="<trn-node>"

[[ -f "$KEY" ]] || { echo "ERROR: ssh key not found: $KEY" >&2; exit 1; }

echo "== probing TRN nodes ${PREFIX}-[${FIRST}-${LAST}] (ignore 'resv' state) ==" >&2
COUNT=0
for i in $(seq "$FIRST" "$LAST"); do
  NODE="${PREFIX}-${i}"
  # Count $USER-owned python processes. 0 => free. Probe is read-only.
  NPROC="$(ssh -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=no \
            -o ConnectTimeout=10 "$NODE" \
            "ps -fu $USER | grep python | grep -v grep | wc -l" 2>/dev/null || echo "ERR")"

  if [[ "$NPROC" == "ERR" ]]; then
    echo "  unreachable : $NODE" >&2
  elif [[ "$NPROC" == "0" ]]; then
    echo "$NODE"                       # machine-readable: free node to stdout
    echo "  FREE        : $NODE" >&2
    COUNT=$((COUNT + 1))
  else
    echo "  busy ($NPROC) : $NODE" >&2
  fi
done

echo "== $COUNT free TRN node(s) ==" >&2
