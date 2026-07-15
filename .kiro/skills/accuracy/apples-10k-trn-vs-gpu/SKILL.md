---
name: apples-10k-trn-vs-gpu
description: Run apples-to-apples 10k pipeline-equivalent training on TRN (Neuron) and GPU (CUDA) and compare loss curves. Use when launching cross-platform training experiments for llama3, qwen3, or deepseek_v3 models, debugging dataset/cmdline/topology mismatches between platforms, validating numerical equivalence between eager and torch.compile, monitoring multi-day production runs across many nodes, or generating side-by-side loss/grad_norm comparison plots. Triggers include "run 10k", "TRN vs GPU", "apples-to-apples", "cross-platform training", or "compare loss curves".
---

# Apples-to-Apples 10k Training: TRN vs GPU

## Overview
Launch cross-platform 10k-step training runs on TRN (Neuron) and GPU (CUDA) clusters with **bit-equivalent input data** and **matched cmdline**, then compare loss/grad_norm curves to validate numerical equivalence between platforms and between eager/torch.compile modes.

## Usage
Use this skill when:
- Asked to "run 10k", "launch 10k", "TRN vs GPU", "apples-to-apples training", "cross-platform comparison"
- Comparing eager vs `torch.compile --compile.backend=neuron` numerics
- Setting up llama3_8B / llama3_70B / qwen3_32B / deepseek_v3_16B training across both platforms
- Plotting cross-platform loss curves
- Diagnosing why TRN and GPU diverge unexpectedly (likely dataset, warmup, seq_len, topology, or seed checkpoint mismatch)
- Operating multi-day production runs across 8-16 nodes per platform with periodic status checks

## Core Concepts

### Apples-to-Apples Definition
Two runs are "apples" only if **all** of:
- Same `seq_len` (both 8192 for llama3_8B; both default for the model otherwise)
- Same `lr_scheduler.warmup_steps` (TIM-style: 10)
- Same `--dataloader.dataset=c4_local` (894 M tokens). **Never c4_test (3.7M tokens) for production** — it wraps every 7 steps at GBS=64 and causes massive overfit divergence.
- Same `--debug.deterministic --debug.seed=42`
- Same `seed_checkpoint` for shared init (when available)
- Same global batch size (GBS) — same 64 unique samples per step regardless of platform packaging
- Same total steps (10000)

Verify by comparing `INPUT_HASH rank=0 step=1 tokens=[…]` between TRN and GPU logs — first-5 tokens at rank 0 step 1 must match exactly.

### Platform Topology Mappings
GBS=64 is the canonical apples GBS. Map it to each platform like this:

| run config       | TP | DP_shard | DP_replicate | LBS | GBS | grad_accum | total procs |
|------------------|----|----------|--------------|-----|-----|------------|-------------|
| 1N TRN apples    | 4  | 16       | 0            | 1   | 16  | 1          | 64          |
| 1N GPU apples    | 4  | 2        | 0            | 1   | 16  | 8          | 8           |
| 4N TRN apples    | 4  | 16       | 4            | 1   | 64  | 1          | 256         |
| 4N GPU apples    | 4  | 2        | 4            | 1   | 64  | 8          | 32          |
| 70B 4N TRN GBS64 | 4  | 64       | 0            | 1   | 64  | 1          | 256         |
| 70B 8N GPU GBS64 | 4  | 16       | 0            | 4   | 64  | 1          | 64          |

Rule: `grad_accum = GBS / (LBS × DP_shard × max(1, DP_replicate))` and `total_procs = TP × DP_shard × max(1, DP_replicate)`.

### Stack Layout
- **TRN training**: `$SHARED/<neuron-env>/torchtitan` + `$SHARED/<neuron-env>/private-torch-neuronx/.venv`
- **GPU training**: `$SHARED/clean-may18/torchtitan` + `$SHARED/moduscope_venv`
- **Dataset**: `DATASET_BASE_PATH=$SHARED/c4_local_big` → `train.jsonl` (893 530 397 tokens)
- **Seed checkpoint** (when shared init needed): `$SHARED/llama3_8b_seed_checkpoint/step-0`
- **Logs**: `$SHARED/logs/run_<TAG>.log[.rank{0..N}]`

### Cluster Access
```bash
# TRN (the TRN cluster u24)
ssh -F $SSH_CONFIG $LOGIN_HOST
ssh -i $SSH_KEY -o StrictHostKeyChecking=no <trn-node>

# GPU (P5en, ap-southeast-3)
ssh -i ~/cgk-p5en.pem -o StrictHostKeyChecking=no ubuntu@ec2-108-137-91-71.ap-southeast-3.compute.amazonaws.com
ssh -o StrictHostKeyChecking=no <gpu-node>
```

## Workflow

### Step 1 — Launch a single platform/mode
The launcher scripts live at `$SHARED/scripts_10k/`. Pick the right one for your topology:

| topology      | TRN launcher                          | GPU launcher                              |
|---------------|---------------------------------------|-------------------------------------------|
| 1N            | `run_on_node_accuracy_moe_tim.sh`     | `run_on_node_gpu_apples.sh`               |
| 4N            | `run_4n_apples_trn.sh`                | `run_4n_apples_gpu.sh`                    |
| 4N TRN 70B    | `run_4n_tim_70b_trn_v2.sh`            | —                                         |
| 8N GPU 70B    | —                                     | `run_8n_tim_70b_gpu_v2.sh`                |

Each launcher takes positional args ending with `STEPS` and accepts an optional compile flag. See `scripts/launch_apples.sh` for the canonical wrapper used in this skill.

### Step 2 — Verify dataset and cmdline before letting it run for hours
After ~3 min, confirm:
```bash
grep 'pre-tokenizing\|pre-tokenized [0-9]' $SHARED/logs/run_<TAG>.log.rank0
# MUST show: c4_local ... 893530397 tokens
grep 'Trainer is initialized' $SHARED/logs/run_<TAG>.log.rank0 | head -1
# MUST show: warmup 10, sequence length 8192 (or model default for non-8B), correct GBS/LBS
```

### Step 3 — Verify input_hash matches across platforms
For a TRN/GPU pair at the same GBS, run `scripts/verify_apples_hashes.sh <trn_tag> <gpu_tag>`. It pulls step=1 input_hash lines from both, extracts `tokens=[...]` first-5-token sequences, compares them. The set of sequences from the smaller-LBS side **MUST** be a subset of the larger.

### Step 4 — Monitor pace and ETA
Use `scripts/poll_status.sh`. It produces a delta table comparing this poll vs the prior poll:
```
                     nodes              prev → now    Δsteps  rate(/min)  loss   MFU     ETA→10k
TRN  8B 4N compile   3-1..3-4           4794 → 5267    +473    21.5       0.99   27.0%   3.7h
TRN  70B 4N compile  3-9..3-12          4149 → 4215    + 66     3.0       1.52   30.6%   32 h
…
```
Run every 10–60 min during a multi-day campaign. Save snapshots in JSON (see `assets/active_runs_template.json`).

### Step 5 — Extract CSVs and plot
After any run reaches 10000 (or for partial comparisons), run `scripts/extract_csv.sh <tag> <output.csv>` to convert log lines to `step,loss,grad_norm,tps,tflops,mfu` (deduped, sorted). Then run `scripts/plot_4way.py` to generate a 4-panel comparison (loss, log(loss), grad_norm, log(grad_norm)) with distinct colors per platform/mode.

Color scheme (used in all our plots):
- TRN eager   = blue solid
- TRN compile = orange solid
- GPU eager   = green dashed
- GPU compile = red dashed

## Common Mistakes

### `c4_test` Dataset Bug
**Symptom**: TRN compile loss drops to 0.7 by step 1000 while TRN eager is at 2.0 — looks like compile is "cheating" or eager has bug. Both 70B runs have the same divergence.
**Cause**: Launcher hardcoded `--dataloader.dataset=c4_test` (3.7 M tokens). Wraps every ~7 steps at GBS=64 → severe overfit on tiny dataset.
**Fix**: `sed -i 's/--dataloader.dataset=c4_test/--dataloader.dataset=c4_local/g' <launcher>` then relaunch from step 0.
**Detection**: `grep 'pre-tokenized [0-9]' <log> | head -1` must show 893 530 397 tokens. If it shows 3 693 482, kill and relaunch.

### `warmup_steps` Mismatch (TRN=10, GPU=200)
**Symptom**: GPU loss curve looks "flatter" early than TRN.
**Cause**: GPU launcher inheriting torchtitan default `warmup_steps=200`, TRN explicitly setting 10.
**Fix**: GPU launcher must explicitly pass `--lr_scheduler.warmup_steps=10`.

### `seq_len` Mismatch for llama3_8B (TRN=8192, GPU=4096)
**Symptom**: TRN runs ~2× slower per token than GPU on llama3_8B; final losses differ by ~5%.
**Cause**: Some GPU configs default `seq_len=4096` for llama3_8B; TRN uses 8192.
**Fix**: GPU launcher must explicitly pass model config that sets seq_len=8192 (or override in cmdline).

### Killing Long Runs Aggressively
**Symptom**: `pkill -9 -f titan_train_wrapper` returns but procs remain alive 10s later.
**Cause**: pkill's regex match races against the long path; some procs survive.
**Fix**: Get explicit PID list and kill by PID:
```bash
PIDS=$(pgrep -f titan_train_wrapper); [ -n "$PIDS" ] && sudo kill -9 $PIDS
```
Repeat after `sleep 8` and verify `pgrep -f titan_train_wrapper | wc -l` returns 1 (the grep itself).

### Compile-Mode Grad-Norm Spikes at Step ~3800 on llama3_8B 1N
**Observation**: Both TRN compile AND GPU compile show grad_norm spikes at the **same step 3809** (peak ~125 on TRN, ~22 on GPU). Eager modes are clean. **NOT a hardware bug** — torch.compile/inductor numerical instability on a specific data sample. Loss stays smooth because gradient clipping absorbs the spikes.
**Action**: None — record as known artifact, do not relaunch.

### TRN Pre-tokenize Wait
**Observation**: TRN runs sit silently for ~8 minutes after launch. Last log line is `pre-tokenizing c4_local…` and nothing else.
**Cause**: Pre-tokenizing 894 M tokens single-threaded per process. Normal.
**Action**: Wait 8–10 min total. Compile runs add another 3-5 min for graph compile.

## Quick Reference

### Verify all 5/6 active runs are alive in one shot
```bash
scripts/verify_alive.sh         # Returns procs count + last log timestamp per run
```

### Generate a status snapshot JSON
```bash
scripts/snapshot_active_runs.sh > active_runs_DO_NOT_KILL.json
```

### Compute remaining time
`ETA_to_10k = (10000 - current_step) / rate_per_min / 60` hours

### Typical rates (steps/min, observed at the TRN cluster/P5en)
| run                  | rate   | notes                |
|----------------------|--------|----------------------|
| 1N TRN 8B eager      | 16     | 20% MFU              |
| 1N TRN 8B compile    | 20     | 29% MFU              |
| 1N GPU 8B eager      | 4      | 6% MFU               |
| 1N TRN 32B eager     | 5      | 19% MFU              |
| 1N GPU 32B eager     | 4.5    | 13% MFU              |
| 4N TRN 8B eager      | 16     | 20% MFU              |
| 4N TRN 8B compile    | 22     | 27% MFU              |
| 4N GPU 8B eager      | 2.1    | 3.4% MFU             |
| 4N TRN 70B eager     | 2.5    | 26% MFU              |
| 4N TRN 70B compile   | 3.0    | 30% MFU              |
| 8N GPU 70B eager LBS=4 | 1.1  | 7% MFU               |

## Final Loss Reference (10k apples runs we've completed)

| model         | TRN eager | TRN compile | GPU eager | GPU compile | spread % |
|---------------|-----------|-------------|-----------|-------------|----------|
| llama3_8B 1N  | 1.0828    | 1.0938      | 1.1041    | 1.0949      | 1.97 %   |
| llama3_8B 4N  | 0.8828    | 1.022 (s5267) | running | —           | tracking |
| qwen3_32B 1N  | 1.4521    | 1.4632      | 1.4912    | (killed)    | 2.69 %   |

**Pass criterion**: cross-platform spread within ~3% of mean loss = within bf16 noise band. Failures > 5% indicate a config/cmdline mismatch (revisit Common Mistakes).

## See Also
- `scripts/launch_apples.sh` — Canonical 10k launcher wrapper for all platforms
- `scripts/poll_status.sh` — Status table with delta tracking
- `scripts/verify_apples_hashes.sh` — Cross-platform hash verification
- `scripts/extract_csv.sh` — Log → CSV extractor
- `scripts/plot_4way.py` — 4-panel comparison plotter
- `scripts/snapshot_active_runs.sh` — Generate JSON manifest of running jobs
- `scripts/verify_alive.sh` — Liveness check for all active runs
- `references/launcher_patterns.md` — Detailed launcher cmdline patterns
- `references/run_recipes.md` — Concrete recipes per (model × topology × mode)
- `assets/active_runs_template.json` — Manifest template
