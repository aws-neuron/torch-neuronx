---
name: activation-compare
description: Localize WHERE and WHEN a real training run diverges between platforms (GPU/CPU reference vs Neuron/TRN) by dumping per-layer activations and/or weight checkpoints at multiple steps and comparing them with FrobRatio and CosSim per layer per step. The long-run debugging cut when a 50-step or 10k-step run drifts, a loss curve separates, or a LOSS SPIKE hits at step X — for a spike, compare per-parameter norms of the step X-1 checkpoint on both platforms (did a weight norm already drift?) and dump step-X activations to find the first layer that deviates. Finds the FIRST layer, step, or parameter where Neuron departs from the reference, then hands it to op-bisect. Covers the activation dump layout (layers.N/global_step_M/activation-output_raw), per-layer stat extraction, checkpoint per-parameter norm compare, and FrobRatio plus cosine similarity vs a bf16/fp32 baseline. Use to compare activations across steps, compare param or weight norms, find which layer drifts, or debug a loss spike at a step.
---

# Activation Compare — localize layer/step drift in a real run

## Overview

When a real training run (50-step, 10k-step) drifts or the loss curves of two platforms separate,
you need to know **which layer, at which step, first departs**. This skill dumps per-layer
activations (and/or weight checkpoints) at several steps on the reference (GPU/CPU) and the target
(Neuron/TRN), then computes **FrobRatio** and **cosine similarity** per layer per step to pinpoint
the first growth point. That layer then goes to `moduscope-op-bisect` for op-level root-cause. It
is the coarse-to-fine bridge between a full-model symptom and a single-op cause.

## Usage

Load this when asked to "compare activations across steps", "which layer drifts", "why does the
loss diverge at step N", or to debug a full-model GPU-vs-Neuron run that a module test can't
explain. It sits ABOVE `moduscope-op-bisect` (per-op within one forward/backward) and BELOW the
whole-run skills (`apples-10k-trn-vs-gpu`, `qwen3-mel-runs`, `titan-accuracy`) which produce the
dumps/checkpoints. Pair with `input-hash-verification` first — rule out a DATA desync before
chasing numerics.

## Core Concepts

### Two comparison surfaces

- **Activations** — per-layer forward outputs captured during the run:
  `<dumps>/layers.<N>/global_step_<M>/dp_<x>_tp_<y>_rank_<z>_activation-output_raw_0.pt`.
  Compare the same (layer, step, rank) across platforms.
- **Weights / checkpoints** — DCP checkpoints per step; compare key params (embed, `layers.k.q_proj`,
  `layers.k.mlp.gate_proj`, `lm_head`) across steps. Weight drift accumulates; activation drift is
  the instantaneous signal.

### The metrics (NOT the same scale as the component 3-tensor ratio — don't confuse them)

- **FrobRatio** = `‖ target − baseline ‖_F / ‖ baseline ‖_F` — relative Frobenius error vs a
  baseline (CPU/GPU **bf16** for an apples-to-apples floor, or **fp32** for the true gap). **~0 =
  identical.** This is a *relative-error* number, NOT the component 3-tensor ratio
  (`‖tgt_bf16−ref_fp32‖/‖ref_bf16−ref_fp32‖`, where ~1.0 = perfect / PASS<1.1) — different formula,
  different pass line. Keep the two straight when reporting.
- **CosSim** = cosine similarity of the flattened tensors — catches direction change even when
  norms match.
- **Signed-mean bias + max-relative-element** — as in `neuron-accuracy-issues`, FrobRatio pools and
  is direction-blind; also emit the signed mean (systematic bias accumulates over a long run) and the
  max element (a concentrated fault hides in the pooled norm).
- Read them as a **layer×step table**: the FIRST cell where FrobRatio jumps above the per-step
  bf16 floor (and CosSim drops) is where the target starts diverging.

### End-to-end pass thresholds (from the accuracy standard)

For a full run, the accept bar (see `neuron-accuracy-issues` §Exit criteria): per-step input hashes
match across all workers; **per-layer error ratio < 1.10**; **cosine > 0.999** (weights **> 0.9999**);
loss within **1–2%**; forced-routing **KL ~1e-3** (≈4e-4..4e-3 by model). Reference points: Llama3-8B
loss within 0.004%; shipped DeepSeek fp32 fix = 7.6e-4 loss diff.

## Workflow

1. **Rule out data desync** (`input-hash-verification`) — identical per-step input hashes on all
   ranks; a step-N loss split can be DATA, not numerics.
2. **Produce dumps** on both platforms for the same steps: enable per-layer activation dumps (and/or
   save DCP checkpoints) during the run (the run skills above drive this). Keep step set + parallel
   layout (dp/tp/rank) aligned across platforms.
3. **Extract stats** — `python scripts/extract_activation_stats.py <activations_dir> <out.pt>`
   collects per (layer/step/rank): frob_norm, mean, std, min, max, shape, dtype, a 100-value sample,
   and the flattened tensor for cross-platform compare.
4. **Compare per layer/step** — `scripts/compare_frobratio.py` computes FrobRatio + CosSim of
   target vs baseline for each key layer at each step and prints the layer×step table (see also the
   prior-art scripts `compare_10step_frob.py`, `ckpt_frob_ratio.py`, `full_compare_v5.py`).
5. **Localize + hand off** — the first layer/step whose FrobRatio exceeds the bf16 floor is the
   origin; drop to `moduscope-op-bisect` on THAT layer/module to find the exact op, then check
   `neuron-accuracy-issues` for a known cause.

## Loss-spike / step-X divergence root-cause (two-pronged)

When a **loss spike or divergence appears at step X** (on Neuron but not GPU, or worse on one),
run BOTH prongs — they answer different questions:

1. **Cause via param norms at X-1** — dump the **step X-1 checkpoint on both platforms** and compare
   **per-parameter norms** (`scripts/compare_param_norms.py`). A spike at X is often caused by a
   *slight* parameter-norm difference that accumulated through X-1 (e.g. one layer's weight norm
   drifting on Neuron), which then tips an update at X. Look for the param(s) whose GPU-vs-Neuron
   norm ratio departs most from 1.0 — that's the suspect that fed the spike.
2. **Where via step-X activations** — dump **step-X activations** on both platforms and run the
   per-layer FrobRatio/CosSim compare to find the **first layer where the deviation starts** at that
   step. That layer localizes the forward path that produced the spike.

Combine: the param-norm prong says *which weight was already off going into X*; the activation prong
says *which layer's forward first shows it at X*. If they point at the same layer, you have the
culprit — hand it to `moduscope-op-bisect`. If param norms match but activations diverge, the issue
is in the step-X compute (kernel/precision), not accumulated weights → check `neuron-accuracy-issues`
(e.g. a compiler-fusion grad-norm spike or an optimizer/warmup spike that is NOT a bug).

```bash
# prong 1: per-parameter norm compare of the X-1 checkpoints (GPU vs Neuron)
python scripts/compare_param_norms.py --ref <gpu_ckpt_step_{X-1}> --target <neuron_ckpt_step_{X-1}>
# prong 2: step-X activation first-deviation (see the workflow above)
python scripts/extract_activation_stats.py <gpu_step_X_dumps> /tmp/g.pt
python scripts/extract_activation_stats.py <neuron_step_X_dumps> /tmp/n.pt
python scripts/compare_frobratio.py --baseline /tmp/g.pt --target /tmp/n.pt
```

## Quick reference

```bash
# 1) extract per-layer stats + tensors from a dump tree
python scripts/extract_activation_stats.py /shared/.../trn_dumps  /tmp/trn_stats.pt
python scripts/extract_activation_stats.py /shared/.../gpu_dumps  /tmp/gpu_stats.pt
# 2) FrobRatio + CosSim, target vs baseline, per layer/step
python scripts/compare_frobratio.py --baseline /tmp/gpu_stats.pt --target /tmp/trn_stats.pt
```

## Common mistakes

- **Skipping the data-hash check.** A step-N divergence is often a DATA desync, not numerics —
  verify identical input hashes first.
- **Comparing across mismatched parallel layout.** Align dp/tp/rank (and gather sharded tensors)
  before comparing the same logical tensor.
- **Using fp32 as the floor when the target is bf16.** Compare against the **bf16** baseline for the
  apples-to-apples floor; fp32 shows the true gap but includes unavoidable rounding.
- **Chasing every drifting layer.** Only the FIRST layer/step that exceeds the floor is the origin;
  downstream layers inherit it. Then go op-level with op-bisect.
- **Reading one rank only.** Check all ranks (a single-expert/single-shard bug shows on one rank).

## How to test this skill

```bash
python -c "import ast; ast.parse(open('scripts/extract_activation_stats.py').read()); print('extract OK')"
python -c "import ast; ast.parse(open('scripts/compare_frobratio.py').read()); ast.parse(open('scripts/compare_param_norms.py').read()); print('compare OK')"
python scripts/selftest_frobratio.py   # synthetic layer/step + param-norm; injected layer & param localized
```
Success: scripts parse and the self-test localizes both the injected activation layer/step AND the
injected diverged parameter (the loss-spike prong).
