#!/bin/bash
# Verify a fix by rerunning tests and diffing outcomes against a baseline run.
#
# Usage:
#   verify_fixed.sh <torch-neuronx-root> <tests.txt> <output-dir> \
#       <baseline-report.json> [-n <num_workers>] [--timeout <seconds>]
#
# Runs pytest via rerun_category.sh in the current shell (same
# hardware-wrapper caveat). Use the SAME -n and --timeout you used
# for the baseline; otherwise outcome-diff noise is possible.
# Diffs the per-test outcomes vs <baseline-report.json> and classifies:
#   - fixed     : was failed/xfailed in baseline, passed now
#   - regressed : was passed in baseline, failed now
#   - still_bad : failed in both
#   - unchanged : passed in both
#
# Emits <output-dir>/verify.md and exits:
#   0  all tests pass, OR fixed>=1 and no regressions
#   1  at least one regression
#   2  no tests fixed and some still fail

set -u
REPO="${1:?repo root}"
TESTS_FILE="${2:?tests.txt}"
OUT="${3:?output dir}"
BASELINE="${4:?baseline report.json}"
shift 4

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
"$SCRIPT_DIR/rerun_category.sh" "$REPO" "$TESTS_FILE" "$OUT" "$@"

python3 - "$BASELINE" "$OUT/report.json" "$OUT/verify.md" <<'PY'
import json, sys
from pathlib import Path
from collections import defaultdict

baseline = {t["nodeid"]: t["outcome"] for t in json.loads(Path(sys.argv[1]).read_text()).get("tests", [])}
now      = {t["nodeid"]: t["outcome"] for t in json.loads(Path(sys.argv[2]).read_text()).get("tests", [])}

buckets = defaultdict(list)
PASS = {"passed"}  # with --runxfail, xfailed-that-passes becomes 'passed'

for nid in sorted(set(baseline) | set(now)):
    b, n = baseline.get(nid, "<absent>"), now.get(nid, "<absent>")
    if b in PASS and n in PASS:
        buckets["unchanged"].append(nid)
    elif b not in PASS and n in PASS:
        buckets["fixed"].append(nid)
    elif b in PASS and n not in PASS:
        buckets["regressed"].append((nid, n))
    else:
        buckets["still_bad"].append((nid, b, n))

lines = ["# Verify report", ""]
lines.append(f"- fixed:     {len(buckets['fixed'])}")
lines.append(f"- regressed: {len(buckets['regressed'])}")
lines.append(f"- still_bad: {len(buckets['still_bad'])}")
lines.append(f"- unchanged: {len(buckets['unchanged'])}")
lines.append("")

if buckets["fixed"]:
    lines += [f"## fixed ({len(buckets['fixed'])})", ""]
    for n in buckets["fixed"]:
        lines.append(f"- `{n.split('::')[-1]}`")
    lines.append("")
if buckets["regressed"]:
    lines += [f"## regressed ({len(buckets['regressed'])})  — new failures, BLOCK THE FIX", ""]
    for nid, n in buckets["regressed"]:
        lines.append(f"- `{nid.split('::')[-1]}` -> {n}")
    lines.append("")
if buckets["still_bad"]:
    lines += [f"## still_bad ({len(buckets['still_bad'])})", ""]
    for nid, b, n in buckets["still_bad"]:
        lines.append(f"- `{nid.split('::')[-1]}` (baseline={b}, now={n})")
    lines.append("")

Path(sys.argv[3]).write_text("\n".join(lines) + "\n")
print(f"# verify: fixed={len(buckets['fixed'])} regressed={len(buckets['regressed'])} still_bad={len(buckets['still_bad'])} unchanged={len(buckets['unchanged'])}", file=sys.stderr)

# Exit code
if buckets["regressed"]:
    sys.exit(1)
if not buckets["fixed"] and buckets["still_bad"]:
    sys.exit(2)
sys.exit(0)
PY
