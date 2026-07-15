---
name: large-model-accuracy-bounds
description: Accept or reject a Neuron (TRN) training run for LARGE models where computing an fp32 reference is infeasible, using two self-referential bf16 signals that need no fp32 baseline. (1) GPU bf16-ULP perturbation envelope — perturb the GPU embedding output by a bf16-ULP-sized error ONCE at step 0, run GPU-perturbed vs GPU-unperturbed, and accept TRN if its divergence from the unperturbed GPU bf16 run stays WITHIN that self-perturbation envelope (TRN is then no worse than a legitimate bf16 rounding wiggle). (2) Grad-norm difference trajectory — plot grad_norm(GPU bf16) minus grad_norm(TRN bf16) over steps; it should hover around 0 (about plus/minus 0.1) for roughly 90 percent of points, with early vs late training sections judged separately. Use when asked how to accept a large-model TRN training run without fp32, set a tolerance boundary via GPU perturbation, or formalize the grad-norm difference check for a long run.
---

# Large-Model Accuracy Bounds — accept a TRN run without fp32

## Overview

For large models an fp32 reference is often impossible (memory/time), so the component ratio
(`‖tgt_bf16 − ref_fp32‖ / ‖ref_bf16 − ref_fp32‖`) can't be computed on the real training run. This
skill gives two **self-referential bf16** acceptance signals — no fp32 needed — for a long training
run (L4): a **GPU bf16-ULP perturbation envelope** and a **grad-norm difference trajectory**. Both
compare GPU-bf16 against TRN-bf16 and ask "is TRN within legitimate bf16 noise?".

## Usage

Load this at the 10k-step / long-run level (`moduscope-accuracy-levels` L4) when you must **accept
or reject a large-model TRN run and fp32 is infeasible**, when asked to "set a tolerance boundary
by perturbing GPU", or to "formalize the grad-norm difference check". Localization of *where* a
run diverges stays in `activation-compare`; this skill decides *whether* the run is acceptable.

## Core Concepts

- **Self-perturbation envelope.** A single bf16-ULP-sized perturbation is a *legitimate* rounding
  event. If TRN's divergence from GPU is no larger than the divergence that such a perturbation
  induces on GPU-vs-itself, TRN is within bf16 noise → acceptable, without any fp32 reference.
- **bf16 ULP.** bf16 has a 7-bit stored mantissa (8-bit significand incl. the implicit bit); the
  unit round-off (half-ULP) is `2^-8 ≈ 0.0039` and the ULP spacing is `2^-7 ≈ 0.0078`. A perturbation
  of magnitude `2^-8` (`x *= (1 + 2^-8)`, or add `|x|·2^-8` noise) is one rounding step — a
  legitimate bf16 event.
- **Why this exists.** The accuracy standard names exactly this gap ("How much should you trust this
  number?"): we lack (i) the ratio distribution of known-good *alternate* implementations and (ii)
  the repeated-config run-to-run spread — so for large models the perturbation envelope is the
  stand-in for that missing variance bound.
- **Grad-norm difference hovers at 0.** With identical data/seed, `grad_norm(GPU_bf16) −
  grad_norm(TRN_bf16)` should hover around 0 across steps; persistent drift or many large spikes is
  the red flag. Early training (high LR/warmup) is noisier than the later steady state — judge the
  sections separately.

## Method 1 — GPU bf16-ULP perturbation envelope

1. **Baseline GPU run** (bf16, unperturbed): record an observable `O_ref(t)` per step — loss, a
   chosen layer's activation Frobenius norm, and/or grad-norm.
2. **Perturbed GPU run**: identical config, but **add a bf16-ULP error to the embedding output ONCE
   at step 0** (`emb_out *= (1 + 2^-8)` or add `|emb_out|·2^-8` noise), then let it run normally.
   Record `O_pert(t)`.
3. **Envelope** `E(t) = |O_pert(t) − O_ref(t)|` — how far one legitimate bf16 rounding moves the run.
4. **TRN run** (bf16, same data/seed): record `O_trn(t)`.
5. **Accept** if `|O_trn(t) − O_ref(t)| ≤ E(t)` for (nearly) all steps — TRN sits inside the
   bf16-legitimacy band. Steps where TRN exceeds the envelope are where TRN is worse than bf16 noise
   → localize with `activation-compare` + `moduscope-op-bisect`.

`scripts/perturbation_bound.py` computes the envelope and the per-step accept/exceed verdict.

## Method 2 — grad-norm difference trajectory

Plot `d(t) = grad_norm(GPU_bf16, t) − grad_norm(TRN_bf16, t)` over steps. Healthy = `d(t)` hovers
around 0, `|d(t)| ≤ ~0.1` for ~90% of points. Report: fraction within tolerance, mean/median `d`,
and split **early vs late** (warmup/high-LR vs steady state) since they behave differently.
`scripts/perturbation_bound.py --gradnorm` gives the hover stats. This is the run-level health
signal that complements the loss-curve match (`apples-10k-trn-vs-gpu`).

## Method 3 — signed-mean bias (the real long-horizon safety signal)

Over a long run, **an unbiased jitter averages out but a systematic bias accumulates ~linearly** and
quietly steers the optimizer — so the ratio/envelope (both magnitude-only, direction-blind) are NOT
enough. Track the **signed mean** of the per-layer (or per-step observable) difference `TRN_bf16 −
GPU_bf16`: it should sit near 0 with no persistent sign. A small ratio with a consistent-sign bias
is more dangerous over 100k steps than a larger but zero-mean jitter. Report signed-mean bias
alongside the envelope verdict and the grad-norm hover; a nonzero drifting bias is the flag that
"benign over a long run" is NOT safe to claim. (This is the companion the accuracy standard requires
next to the ratio — see `neuron-accuracy-issues`.)

## Interpretation

- **Inside envelope + grad-norm diff ~0** → accept: TRN is within legitimate bf16 behavior.
- **Outside envelope at some steps** → not automatically a bug, but TRN exceeds bf16 noise there →
  localize (activation-compare → op-bisect) and check `neuron-accuracy-issues`.
- **Grad-norm diff drifts / many spikes** → systematic; a same-step spike on BOTH chips is
  optimizer/warmup noise (not a bug), one-sided drift is real.

## Common mistakes

- **Perturbing every step / too large.** Perturb ONCE at step 0 at bf16-ULP scale; a large or
  repeated perturbation inflates the envelope and hides real TRN error.
- **Comparing fp32-GPU to bf16-TRN.** Both sides must be **bf16** — the whole point is a bf16-vs-bf16
  legitimacy band.
- **Judging all steps with one tolerance.** Early (warmup/high-LR) vs late steps differ; split them.
- **Using the envelope to LOCALIZE.** It only says accept/reject; use activation-compare + op-bisect
  to find where/why.

## How to test this skill

```bash
python -c "import ast; ast.parse(open('scripts/perturbation_bound.py').read()); print('parse OK')"
python scripts/selftest_bound.py   # within-envelope TRN accepted; outside-envelope rejected; grad-norm hover healthy
```
Success: the self-test accepts a within-envelope TRN, rejects an outside-envelope one, and reports a
healthy grad-norm-diff hover.
