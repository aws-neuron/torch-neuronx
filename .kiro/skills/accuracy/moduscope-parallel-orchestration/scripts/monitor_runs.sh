#!/usr/bin/env bash
# monitor_runs.sh — summarize the state of a parallel ModuScope launch by scanning
# its per-cell log files for terminal verdict markers.
#
# Usage:
#   bash monitor_runs.sh <LOGDIR> [--watch [SECONDS]]
#     LOGDIR : the directory printed by parallel_launch_template.sh
#     --watch: re-scan every SECONDS (default 60) until all logs are terminal
#
# Verdict markers emitted by the tests / tooling (grounded in conftest_base.py
# and generate_summary.py):
#   [PASS] [PASS-NaN] [PASS-zero] [PASS-near-zero] -> pass-class
#   [FAIL] [FAIL-targetNaN] [FAIL-zero]            -> fail-class
#   THRESHOLD_RESULT: PASS|FAIL                    -> per-run summary verdict
# A log is considered DONE when pytest prints its session epilogue
# ("passed", "failed", "error", or "no tests ran").

set -euo pipefail

LOGDIR="${1:?need LOGDIR (printed by parallel_launch_template.sh)}"
MODE="${2:-once}"
INTERVAL="${3:-60}"

[[ -d "$LOGDIR" ]] || { echo "ERROR: log dir not found: $LOGDIR" >&2; exit 1; }

scan_once() {
  local total=0 done=0 pass=0 fail=0
  shopt -s nullglob
  echo "== $(date '+%H:%M:%S')  scan of $LOGDIR =="
  for log in "$LOGDIR"/*.log; do
    total=$((total + 1))
    local name; name="$(basename "$log")"
    local p f thr state
    p="$(grep -cE '\[PASS(-NaN|-zero|-near-zero)?\]' "$log" 2>/dev/null || true)"
    f="$(grep -cE '\[FAIL(-targetNaN|-zero)?\]' "$log" 2>/dev/null || true)"
    thr="$(grep -oE 'THRESHOLD_RESULT: (PASS|FAIL)' "$log" 2>/dev/null | tail -1 || true)"
    if grep -qE '== .*(passed|failed|error|no tests ran).* ==' "$log" 2>/dev/null; then
      state="DONE"; done=$((done + 1))
    else
      state="running"
    fi
    [[ "$f" -gt 0 || "$thr" == *FAIL* ]] && fail=$((fail + 1)) || { [[ "$state" == "DONE" ]] && pass=$((pass + 1)); }
    printf '  %-7s pass=%-4s fail=%-4s %-22s %s\n' "$state" "$p" "$f" "${thr:-}" "$name"
  done
  echo "  -- $done/$total done; $pass clean, $fail with failures --"
  [[ "$done" -eq "$total" && "$total" -gt 0 ]] && return 0 || return 1
}

if [[ "$MODE" == "--watch" ]]; then
  while true; do
    if scan_once; then echo "All cells terminal."; break; fi
    sleep "$INTERVAL"
  done
else
  scan_once || true
fi
