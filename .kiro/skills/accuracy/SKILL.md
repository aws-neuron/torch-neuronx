---
name: accuracy
description: Master index and router for the Neuron-vs-GPU/CPU model accuracy suite — 25 skills covering the full ladder from module/component equivalence tests, through op/NKI-level divergence localization, to full-model 5-step / 50-step / 10k-step training validation and per-layer activation comparison, plus the known-issues + precision catalog and large-model acceptance bounds. Load this to choose WHICH accuracy skill to use for a task (validate a new model, debug a failing module, root-cause a training divergence or loss spike, accept a large-model run without fp32) — it names the exact sub-skill and the testing level. Every sub-skill is also usable directly on its own. Covers dense and MoE models (DeepSeek-V3, Qwen3, Llama4, GPT-OSS, Llama3). Use when asked how to validate or debug model numerical accuracy on Neuron, which accuracy test/level to run, or where to start.
---

# Accuracy Suite — master index & router

## Overview

This folder is a **suite of 25 skills** for validating and debugging that a model computes the same
thing on Neuron (TRN) as on a GPU/CPU reference — from a single module up to a 10k-step training run.
This master skill routes you to the right one. **Each sub-skill is independently usable** (discovered
directly by name/description) — you do not have to go through this master; it just helps pick.

## Usage

Load this when the task is "validate/debug model accuracy on Neuron" but you're not sure which skill
or level applies: onboarding a new model, a failing/NaN/bad-ratio module, a training run that drifts
or spikes, or accepting a large-model run without an fp32 reference. It names the sub-skill + level;
then load that sub-skill.

## Core Concepts

- **One metric family** end to end: the component 3-tensor ratio `‖tgt_bf16−ref_fp32‖/‖ref_bf16−ref_fp32‖`
  (PASS < 1.1) at module/op level, and FrobRatio + CosSim per layer/step at run level — always paired
  with the **signed-mean bias** (the direction-blind ratio alone can't tell you long-run safety).
- **Pick the lowest level that answers the question.** Module test (minutes) and op-bisect before a
  10k-step run (days).
- **Coarse-to-fine for whole-model symptoms:** run symptom → activation-compare (which layer/step) →
  op-bisect (which op) → neuron-accuracy-issues (which known cause).

## Placeholder convention (set these before running any script)

The skills do **not** hardcode cluster or connection details — they are parameterized so anyone can
run them. Provide these from your own environment; if a skill needs a login host or node and you
have not supplied one, it asks you rather than assuming.

| Placeholder | Meaning | Example |
|---|---|---|
| `$SHARED` | shared cluster scratch dir | `/shared/$USER` |
| `$WORKSPACE` | your working/checkout dir | `/local/home/$USER` |
| `$VENV` | local torch venv for the CPU-only self-tests | `~/venvs/torch` |
| `$LOGIN_HOST` | cluster login host (ssh alias) | your cluster's login host |
| `$SSH_KEY` / `$SSH_CONFIG` | ssh key / ssh config for compute nodes | `~/.ssh/id_...` |
| `<trn-node>` / `<gpu-node>` | a TRN (Neuron) / GPU compute node you can access | your allocated node hostnames |
| `$USER` | your login (node process/reservation ownership) | (whoami) |
| `$SKILLS` | this skills dir | `.kiro/skills/accuracy` |

## Index (all 25, grouped) — load any directly

**Router / entry**
- `moduscope-accuracy-levels` — the testing-level ladder (L0 module → L1 op/NKI → L2 5-step → L3 50-step → L4 10k → L4+ activation-compare) with a decision tree.

**Debugging (find the cause)**
- `moduscope-accuracy-debug` — front door: triage → attribute env-vs-test → localize → characterize.
- `moduscope-op-bisect` — localize the EXACT op/line (eager torch.save per op, compiled-returns, NKI device_print via wrap_nki+NEURON_RT_DEBUG_OUTPUT_DIR); fwd + bwd.
- `activation-compare` — per-layer FrobRatio/CosSim across steps to find the first drifting layer/step; includes the loss-spike (X-1 param-norms + step-X activations) workflow.
- `moduscope-result-triage` — classify a result (PASS/FAIL/NaN/zero tags).
- `moduscope-test-roadblocks` — recovery playbook for runtime roadblocks.

**Knowledge / standards**
- `neuron-accuracy-issues` — catalog of the 9 known causes, precision RCs, CRs, the exit-criteria standard, prevention rules, and the accuracy summary docs.
- `large-model-accuracy-bounds` — accept a large-model TRN run WITHOUT fp32 (bf16-ULP perturbation envelope + grad-norm-diff + signed-mean bias).
- `moduscope-reference-fidelity` — build a faithful 1:1 reference wrapper by reading model source.

**Module/component level (L0) — build & run test_00..16**
- `moduscope-component-tests`, `moduscope-run-test`, `moduscope-dist-sweep`, `moduscope-add-test-module`,
  `moduscope-model-profile`, `moduscope-register-model`, `moduscope-statedict-mapping`,
  `moduscope-gpu-baselines`, `moduscope-trn-compare`, `moduscope-env-bootstrap`,
  `moduscope-node-select`, `moduscope-parallel-orchestration`, `moduscope-newmodel` (onboarding orchestrator), `moduscope-autopilot` (near-zero-touch).

**Full-model runs (L2–L4)**
- `titan-accuracy` — end-to-end GPU H200 vs TRN2 cross-platform validation.
- `apples-10k-trn-vs-gpu` — 10k-step apples-to-apples loss-curve comparison.

## How to route (quick)

| You want to… | Load |
|---|---|
| validate a NEW model from scratch | `moduscope-newmodel` / `moduscope-autopilot` (L0) |
| a module test failed / NaN / bad ratio | `moduscope-accuracy-debug` → `moduscope-op-bisect` |
| know if a divergence is a KNOWN issue / cite a CR | `neuron-accuracy-issues` |
| a training run drifts / loss splits at step N | `activation-compare` (+ loss-spike workflow) → op-bisect |
| accept a large-model run but fp32 is infeasible | `large-model-accuracy-bounds` |
| choose a testing level / depth | `moduscope-accuracy-levels` |
| run a full 10k comparison | `apples-10k-trn-vs-gpu` / `titan-accuracy` |

## Common mistakes

- **Launching a 10k run to debug a kernel** — use L0/L1 first (minutes vs days).
- **Debugging ops before localizing the layer** in a full-model drift — activation-compare first.
- **Judging on the ratio alone** — always pair with signed-mean bias + max-relative-element.
- **Not setting the placeholders** — scripts use `$SHARED`/`$VENV`/`$LOGIN_HOST` etc.; set them first.

## How to test this skill

The suite's portable self-tests run with only torch (no cluster):
```bash
python moduscope-op-bisect/scripts/selftest_bisect.py        # -> SELFTEST OK — origin = op05
python moduscope-op-bisect/scripts/selftest_realmodule.py    # -> origin 004_l2
python moduscope-op-bisect/scripts/selftest_variations.py    # -> SELFTEST_VARIATIONS OK
python activation-compare/scripts/selftest_frobratio.py      # -> localizes layer/step + param
python large-model-accuracy-bounds/scripts/selftest_bound.py # -> SELFTEST_BOUND OK
```
And every sub-skill is referenced above:
```bash
for s in $(ls -d */ | sed 's#/##'); do grep -q "$s" SKILL.md && echo "OK $s" || echo "MISSING $s"; done
```
Success: the self-tests pass and every sub-skill directory is indexed.
