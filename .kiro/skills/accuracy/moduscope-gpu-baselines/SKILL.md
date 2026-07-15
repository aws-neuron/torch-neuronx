---
name: moduscope-gpu-baselines
description: Run ModuScope TEST_MODE=gpu_save on an the GPU cluster node to compute fp32+bf16 CUDA references for a torchtitan model and UPLOAD goldens to S3 (no compare — the test early-returns after upload). Use when capturing/refreshing GPU baselines for a new or existing model (deepseekv3_16b, llama4_17bx16e, qwen3_30b_a3b, gpt_oss_20b) before a TRN neuron compare, when asked to "save gpu baselines", "run gpu_save", "populate S3 goldens", or when a TRN run reports missing S3 artifacts. Covers ssh to the GPU login host, srun node allocation on partition compute1, the exact env-var set (REF_PATH=TARGET_PATH=NTT tree, PYTHONPATH, REF_DEVICE/TARGET_DEVICE=cuda), reading gpu_save FAILED as a NaN-poisoned edge-dist baseline (not a Neuron signal), and verifying uploads via aws s3 ls.
---

# ModuScope GPU Baselines (gpu_save)

## Overview

`TEST_MODE=gpu_save` is the **producer** half of the ModuScope cross-cluster workflow: it builds the
fp32 + bf16 reference for each distribution config on a CUDA GPU, **uploads the goldens to S3, and
early-returns with NO compare**. The TRN `neuron` run later downloads these goldens to compute the
`ratio = ||target−fp32|| / ||ref_bf16−fp32||` verdict. This skill runs that producer step on the
the GPU cluster cluster and verifies the uploads landed in S3.

## Usage

Use this skill when:
- Onboarding a new model and you need GPU goldens before any TRN compare can run
- A `TEST_MODE=neuron` run logs `[S3 LOAD]` failures / missing artifacts for a `(model, module, dist)`
- Refreshing baselines after a torchtitan ref-tree change
- Asked to "run gpu_save", "save/capture GPU baselines", or "populate S3 goldens"

This is the GPU/producer side only. For the TRN/consumer side and ratio debugging see the
`moduscope-component-tests` skill (`references/RUN-COMMANDS.md`).

## Core Concepts

### gpu_save data flow (grounded in `bf16/tests/conftest.py` + each test's Phase 1)

```
build ref_fp32 (cuda) ─┐
build ref_bf16 (cuda) ─┤→ fwd + (RUN_BWD) bwd → save_ref_artifacts(...) → [S3 UPLOAD] → return
                        └ NO Phase-3 compare in gpu_save
```

`_DEVICE_DEFAULTS["gpu_save"] = ("cuda","cuda")`. Each test's Phase 1 block guards
`if TEST_MODE in ("gpu","gpu_save",...)`, then `if TEST_MODE in ("gpu_save","cpu_save"): save → log
"[S3 UPLOAD] Saved baselines to <s3_path>" → return`. `TARGET_DEVICE` is irrelevant in this mode.

### What lands in S3 (one dir per dist config)

Layout: `s3://<bucket>/titan/modules/moe/<model_name>/<module>/<dist_name>/` containing
`config.json, weights.pt, input.pt, ref_fp32_output.pt, ref_bf16_output.pt, ref_fp32_grads.pt,
ref_bf16_grads.pt`. `<model_name>` = `MODEL_CONFIG`; `<module>` is derived per test (e.g. moe,
grouped_experts, attention, transformer_block); `<dist_name>` = `dist_cfg.name`. The test prints the
exact prefix on every dist — **always read the `[S3 UPLOAD]` log lines for ground truth** rather than
guessing the module dir.

### Reading a gpu_save FAILED — NOT a Neuron signal

In `gpu_save` there is **no compare assertion** (the test returns right after upload). A pytest
`FAILED`/error in this mode almost always means the **reference build itself produced a non-finite
tensor** on an edge distribution — extreme weight init (e.g. `log_normal`, aggressive `kaiming`) on
certain shapes overflows fp32/bf16, poisoning that dist's baseline. This is a **bad-reference
artifact, not a Neuron correctness signal**. The successfully-built dists still upload (the per-dist
upload precedes the return), so a partial sweep is normal — just re-run the poisoned dist or accept
that the TRN side will mark it `[PASS-NaN]` (GPU baseline non-finite ⇒ reference invalid, see
`shared/conftest_base.py`). You MUST NOT treat a gpu_save FAILED as a Neuron bug.

## Quick Reference — run it

`scripts/run_gpu_save.sh` parameterizes the whole launch by `NODE`, `MODEL_CONFIG`, `TESTS`,
`N_PER_DIST` (plus `REPO_ROOT`, `NTT_PATH`). Run it ON the GPU login host:

```bash
# 1. From your laptop, hop to the GPU cluster login host
ssh -i ~/cgk-p5en.pem ubuntu@ec2-108-137-91-71.ap-southeast-3.compute.amazonaws.com

# 2. Pick a free node (or let the script auto-pick)
/opt/slurm/bin/sinfo -p compute1 -t idle

# 3. Launch gpu_save (smoke = 1 cfg/family; full = 5)
NODE=<gpu-node> \
MODEL_CONFIG=deepseekv3_16b \
TESTS="test_15_moe.py test_13_attention.py" \
N_PER_DIST=1 \
./scripts/run_gpu_save.sh
```

### Manual launch (what the script does)

```bash
NODE=<gpu-node>
nohup /opt/slurm/bin/srun -N 1 -w $NODE -p compute1 --no-kill bash -c '
  source $SHARED/aws_creds.sh
  source $SHARED/gpu_baselines_venv/bin/activate
  export REF_PATH=$SHARED/NTT_gpu_baselines TARGET_PATH=$SHARED/NTT_gpu_baselines
  export PYTHONPATH=<REPO>/ModuScope/test/component_tests_torchtitan/shared:<REPO>/NeuronTrainingEvaluations/src:$PYTHONPATH
  export TEST_MODE=gpu_save REF_DEVICE=cuda TARGET_DEVICE=cuda
  export MODEL_CONFIG=deepseekv3_16b
  export COMPONENT_TEST_N_PER_DIST=1
  cd <REPO>/ModuScope/test/component_tests_torchtitan/bf16/tests
  python -m pytest test_15_moe.py -c $SHARED/pytest_clean.ini -rs
' > $SHARED/pytest_gpu_save_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Key env vars (gpu_save)

| Var | Value | Note |
|---|---|---|
| `TEST_MODE` | `gpu_save` | upload goldens, no compare |
| `REF_DEVICE` / `TARGET_DEVICE` | `cuda` / `cuda` | target ignored here |
| `REF_PATH` = `TARGET_PATH` | `$SHARED/NTT_gpu_baselines` | GPU torchtitan tree |
| `MODEL_CONFIG` | model or group | `deepseekv3_16b`, `llama4_17bx16e`, `qwen3_30b_a3b`, `gpt_oss_20b`, or `moe`/`dense` |
| `COMPONENT_TEST_N_PER_DIST` | `1` smoke / `5` full | 1→7 cfgs, 5→35 cfgs across 7 dist families |
| `PYTHONPATH` | `<repo>/shared` + `<NTE>/src` | required for collection |
| pytest | `-c $SHARED/pytest_clean.ini -rs` | ModuScope setup.cfg otherwise ignores the dir |

MoE tests are 08/13/14/15/16; dense tests are 00–07 (see `moduscope-component-tests` for the full
inventory and per-model attention/block quirks).

## How to test this skill

This skill is AUTHOR-only here; the launcher is for the future user on the live GPU host. Verify
without a cluster run:

```bash
# 1. Launcher is syntactically valid (no execution)
bash -n $SKILLS/moduscope-gpu-baselines/scripts/run_gpu_save.sh && echo "syntax OK"
```

Success criterion: prints `syntax OK` (exit 0).

After a *real* gpu_save run on the GPU host, the runnable success check is that S3 now lists one
directory per dist config for the model (do NOT run live in author mode):

```bash
source $SHARED/aws_creds.sh
aws s3 ls s3://<bucket>/titan/modules/moe/<model>/<module>/      # lists dist dirs e.g. normal_0/, kaiming_uniform_2/
```

Success criterion: the prefix exists and lists the expected dist sub-dirs (their count matches
`COMPONENT_TEST_N_PER_DIST` × the families that built finite baselines). Cross-check against the
`[S3 UPLOAD]` lines in the run log for the exact prefixes.

## Common Mistakes

- **Reading gpu_save FAILED as a Neuron bug** — it is a NaN-poisoned baseline on an edge dist; see Core Concepts.
- **Omitting `-c $SHARED/pytest_clean.ini`** — ModuScope `setup.cfg` `--ignore`s this dir and 0 tests collect.
- **Forgetting `source $SHARED/aws_creds.sh`** — STS creds expire (~hours); upload silently fails / pytest errors on S3.
- **Using the TRN env on GPU** — gpu_save uses `REF_DEVICE=TARGET_DEVICE=cuda`; do not set `NEURON_RT_VISIBLE_CORES` or the fallback-ops vars here.
- **Assuming the module dir name** — read the `[S3 UPLOAD]` log line; the `<module>` segment is test-derived.
