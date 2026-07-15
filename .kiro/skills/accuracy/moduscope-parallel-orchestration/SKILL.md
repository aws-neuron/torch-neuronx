---
name: moduscope-parallel-orchestration
description: Fan out many ModuScope component-test cells (test x node) concurrently across the TRN (Neuron) cluster and the GPU cluster clusters with one run per node (no contention), then monitor them. Use when asked to run many tests at once, scale a sweep across free nodes, launch in parallel, or check progress of a running batch. Critical knowledge — TRN cells launch via detached ssh using setsid+nohup+/dev/null so they survive disconnect, GPU cells via nohup srun backgrounded; progress is read by tailing per-cell logs and grepping for PASS/FAIL/THRESHOLD_RESULT markers. Pair with moduscope-node-select to get verified-free nodes first.
---

# ModuScope Parallel Orchestration

## Overview

Launch many ModuScope component-test cells at once — one cell (test file) per
verified-free node — so a full sweep finishes in wall-clock time proportional to a
single cell, not the sum. TRN cells run as detached ssh background jobs; GPU cells
run as backgrounded `srun` allocations. Monitoring is log-driven: scan each cell's
log for `[PASS]`/`[FAIL]`/`THRESHOLD_RESULT` markers.

## Usage

Use this skill when:
- Asked to "run many tests in parallel", "scale the sweep", or "launch across all free nodes"
- Distributing a (test × config) matrix over a list of free nodes
- Backgrounding long TRN/GPU runs so they survive ssh disconnect
- Checking progress / verdicts of an already-launched batch

Always get the free-node list from `moduscope-node-select` FIRST, then build a
`(node, testfile)` pair list — exactly one cell per node.

## Core Concepts

### One run per node (no contention)
ModuScope cells are heavy single-process jobs that pin an accelerator. Scheduling
two cells on the same node corrupts numerical results and skews ratios. The launch
template therefore consumes a pair list with one `<node> <testfile.py>` per line,
and you MUST ensure each node appears at most once.

### Detached background launch patterns (grounded in the briefing)
- **TRN** (from the `$LOGIN_HOST` bastion, reach node over ssh):
  ```bash
  ssh -i $KEY <node> "setsid nohup bash -c '<env + pytest>' > LOG 2>&1 < /dev/null &"
  ```
  `setsid` + `nohup` + `</dev/null` fully detach the run from the ssh session so it
  survives disconnect. `KEY=$SSH_KEY`.
- **GPU** (from the GPU login host):
  ```bash
  nohup /opt/slurm/bin/srun -N 1 -w <node> -p compute1 --no-kill bash -c '<env + pytest>' > LOG 2>&1 &
  ```
  `srun` holds the allocation on the chosen node; `nohup ... &` backgrounds the
  local waiter.

### Per-cluster environment (must be exact)
| | TRN (`TEST_MODE=neuron`) | GPU (`TEST_MODE=gpu_save`) |
|---|---|---|
| venv | `<neuron-env>/private-torch-neuronx/.venv` (NOT <incompatible-venv>) | `gpu_baselines_venv` |
| extra env | `TARGET_DEVICE=neuron NEURON_RT_VISIBLE_CORES=0`, `TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=0`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` | `REF_DEVICE=cuda TARGET_DEVICE=cuda` |
| pytest | `-c $SHARED/pytest_clean.ini` | same |
| PYTHONPATH | `<repo>/shared` + `<NeuronTrainingEvaluations>/src` | same (GPU trees) |

`$SHARED/` is NOT cross-cluster — S3 is the bridge. Refresh creds with
`source $SHARED/aws_creds.sh` on each side before launching.

### Monitoring by log markers
Verdicts come from `conftest_base.py` and `generate_summary.py`:
`[PASS]`/`[PASS-NaN]`/`[PASS-zero]`/`[PASS-near-zero]` (pass-class),
`[FAIL]`/`[FAIL-targetNaN]`/`[FAIL-zero]` (fail-class), and one
`THRESHOLD_RESULT: PASS|FAIL` per summarized run. A cell is done when pytest prints
its session epilogue (`passed`/`failed`/`error`/`no tests ran`). `monitor_runs.sh`
scans all logs and tallies these.

## Quick Reference

```bash
# 1. Build a pair list (one cell per free node) — e.g. from node-select output:
cat > /tmp/pairs.txt <<'EOF'
<trn-node>  test_13_attention.py
<trn-node>  test_14_grouped_experts.py
<trn-node>  test_15_moe.py
EOF

# 2. Launch all cells in parallel (returns immediately; runs are detached):
bash scripts/parallel_launch_template.sh trn deepseekv3_16b /tmp/pairs.txt 5

# 3. Monitor until every cell is terminal:
bash scripts/monitor_runs.sh <LOGDIR printed above> --watch 60
```

## Deterministic Scripts

### scripts/parallel_launch_template.sh
Takes `<CLUSTER> <MODEL_CONFIG> <PAIRS_FILE> [N_PER_DIST]`. Reads a
`<node> <testfile.py>` pair list (one per line, `#` comments allowed) and launches
each cell on its node using the detached TRN ssh pattern or the GPU `srun` pattern.
Writes one log per cell under `$SHARED/moduscope_parallel_logs/<...>/` and
prints the `TS` and `LOGDIR`. Generalizes the prior-art template at
`moduscope-component-tests/references/parallel-launch-template.sh`.

### scripts/monitor_runs.sh
Takes `<LOGDIR> [--watch [SECONDS]]`. Scans every `*.log` in the dir, counts
`[PASS]`/`[FAIL]` markers, surfaces the latest `THRESHOLD_RESULT`, and reports
done/running state per cell. With `--watch` it re-scans on an interval until all
cells reach a terminal pytest epilogue.

## How to test this skill

Structural / syntax check (no cluster access required):
```bash
bash -n scripts/parallel_launch_template.sh && \
bash -n scripts/monitor_runs.sh && \
echo "SYNTAX_OK"
```
Success criterion: prints `SYNTAX_OK` with exit code 0.

Offline behavior check for the monitor (no cluster needed): create a fake log dir
and confirm the marker tally works:
```bash
d=$(mktemp -d); printf '[PASS]\nTHRESHOLD_RESULT: PASS\n== 1 passed in 1s ==\n' > "$d/test_demo_node1.log"
bash scripts/monitor_runs.sh "$d"   # expect: 1/1 done; 1 clean, 0 with failures
```

## Common Mistakes

- **Two cells on one node.** Causes contention/corrupted ratios — one cell per node.
- **Foreground ssh/srun.** Omitting `setsid nohup ... </dev/null &` (TRN) or
  `nohup ... &` (GPU) ties the run to your session; it dies on disconnect.
- **Wrong TRN venv.** `<incompatible-venv>` has a complex64 `.conj()` bug — always use
  `<neuron-env>/private-torch-neuronx/.venv`.
- **Stale creds.** Re-`source aws_creds.sh` per cluster before launch; S3 is the
  only cross-cluster bridge.
