---
name: neuron-accuracy-issues
description: The catalog of KNOWN Neuron-vs-GPU numerical-accuracy issues, their root causes, fixes/CRs, and the precision knowledge behind them — plus the one-page accuracy summary. Covers the nine shipped/understood bugs (SDPA bf16 softmax-times-V accumulator drift, MoE cross-expert bf16 HBM read-modify-write, already-bf16 gradient precision floor, softmax-backward cancellation at scale, complex64 RoPE backward 451x blowup, compiler matmul+elementwise fusion dropping a bf16 cast, fused memory op dropping a boundary tile zeroing an expert grad, test-harness traps, and warmup loss spikes), the 3-tensor ratio metric (target_bf16-ref_fp32 over ref_bf16-ref_fp32, pass under 1.1), precision RCs RC-1..RC-4, key CRs, and per-model status. Use to recognize a known accuracy cause fast, cite the CR/fix, understand a precision floor, or write an accuracy summary.
---

# Neuron Accuracy Issues — catalog + precision knowledge

## Overview

A reference catalog of every Neuron-vs-GPU numerical-accuracy issue we found across four MoE
models (DeepSeek-V3 16B, Llama4 17bx16e, Qwen3-30B-A3B, GPT-OSS 20B) plus dense Qwen3/Llama3,
with the root cause, the fix/CR, and the precision reasoning. Use it to **recognize a known cause
in seconds** instead of re-debugging, to **cite the fix**, and to **write an accuracy summary**.
This skill is self-contained — the catalog, metric, exit criteria, and prevention rules are below.

## Usage

Load this when you hit an accuracy divergence and want to know "is this a KNOWN issue", "what's the
fix / CR", "why is there a precision floor here", "what's the accuracy metric", or when asked to
"summarize the accuracy status" / "write the accuracy brief". Pair with `moduscope-accuracy-debug`
(the live debug loop) and `moduscope-op-bisect` (localize a new, uncatalogued divergence).

## Core Concepts

### The accuracy metric (3-tensor ratio)

```
ratio = ‖ target_bf16 − ref_fp32 ‖_F / ‖ ref_bf16 − ref_fp32 ‖_F      # ~1.0 perfect; PASS < 1.1 (1.2 prod)
```
Round every weight+input to bf16 ONCE, then run the fp32 reference on those same values — so Neuron
is never charged for rounding the reference dodged. `ABS_TOL=1e-5`, `ZERO_EPS=1e-30`. Guard tags:
`[PASS]/[FAIL]/[PASS-NaN]/[FAIL-targetNaN]/[PASS-zero]/[FAIL-zero]/[PASS-near-zero]`; target-NaN
where ref is finite = HARD FAIL. A score **below 1.0 is GOOD, not a warning** — Neuron was *more*
faithful than the bf16 reference (the RC-2 fp32-accumulator fix lands at 0.97); the line is an upper
bound, not a target, lower is better. The 1.1 line is **calibrated on the decoder models, not
variance-bounded** — a result hugging the threshold should be **re-run**, not read to two decimals
(reduction-order non-determinism can flip a borderline verdict).

**Companions to the ratio (do not judge on the ratio alone).** The Frobenius ratio is
*direction-blind* and *pools* error, so it hides two things — always emit alongside it:
- **signed-mean bias** per module — an unbiased jitter averages out over thousands of steps, but a
  *systematic* bias accumulates ~linearly and quietly steers the optimizer. **The bias, not the
  ratio, is what tells you whether drift is safe over a long run.**
- **max relative element** — a single concentrated fault can hide under an otherwise-fine pooled
  score; watch the largest relative element too.
- For spiky outputs (logits, router gates) the ratio is the wrong lens — use **top-K agreement /
  cosine / loss** (one "1.14 failure" had 100% top-1 and |Δloss|=0.001).

## The catalog — 9 causes (every failure traced to one of these; never the matmul engine)

| # | Bug | Root cause | Fix | CR / status |
|---|-----|-----------|-----|-------------|
| 1 | **SDPA bf16 accumulator** | attention keeps softmax·V total in bf16 (GPU fp32) → ~2.04× fwd drift, amplified ~50× by QK-norm | relax threshold; proven benign for convergence (architectural) | Shipped |
| 2 | **MoE cross-expert bf16 HBM RMW** | experts combined via bf16 read-modify-write, ~5 roundings/token (top_k=6) → 1.20 | fp32 HBM + fp32 SBUF accumulator, single final bf16 downcast → 0.97 | shipped (fp32-accumulation fix) |
| 3 | **already-bf16 gradient** | a grad multiplies two already-bf16 operands → precision floor 1.13–1.18 (proven fundamental: flat vs seq-length, matches CPU and GPU ref) | accepted fundamental floor | Understood |
| 4 | **softmax-backward cancellation** | subtraction cancels in bf16 at large scale (27× only at extreme artificial inputs; benign at realistic scale) | fp32 in that spot / test at realistic scale | Shipped |
| 5 | **complex64 RoPE backward** | complex RoPE bwd corrupts grads → **451×** (forward bit-identical, backward broken) + a stray autograd-detach | switch to cos/sin RoPE backend → 1.0 | Shipped (cured the training run; a separate narrower component-test attn-bwd issue is NOT the same bug) |
| 6 | **compiler fusion drops bf16 cast** | matmul+elementwise fusion keeps fp32 and drops a declared bf16 cast → grad-norm spike | **break** the fusion → spike ↓122× (polite ways — marking the op / allowing in-graph — do NOT work on Neuron) | Shipped |
| 7 | **fused memory op drops boundary tile** | one expert's grad = exactly 0 | pad-and-slice workaround (proper kernel fix open; Qwen3-30B-A3B full-MoE still 348× at long seq) | Workaround |
| 8 | **test-harness traps** | fake-clean reference, accidental compile, batch mismatch, router tie-flips, scrambled seed layout | cheap pre-checks; seed de-interleave | shipped (GPT-OSS de-interleave) |
| 9 | **warmup loss spike** | same-step spike on BOTH chips = optimizer/warmup noise, not a bug | warmup 10→3 cut it 8× | N/A |

Other key fix: forced routing (nullify router variance to isolate MoE-kernel numerics).

**Not RC-5:** a **50–90× backward blowup** is almost always the **wrong venv** (`<incompatible-venv>`'s
complex64 `.conj()` bug), an RC-8/env trap — use `<neuron-env>`/private-torch-neuronx. RC-5 is the
genuine 451× complex-RoPE bug, fixed by cos/sin.

## Precision RCs (from review01_precision — fact-checked)

- **RC-1 SDPA bf16 accumulator**: fwd ~2.0 on small-mag inputs; `grad_q` 1.13–1.60; Inf/NaN grads at
  large q/k scale (rms≳50–100); isolated repro 2.04. QK-norm rescales SDPA inputs ~50× (std 0.02→1.0)
  → Qwen3 borderline ~1.08, Llama3 (no QK-norm) safe. MLA `k_dim=192≠v_dim=128` forces MLIR
  decomposition. `mixed_precision=True` does NOT help.
- **RC-2 MoE bf16 HBM RMW loop**: fwd 1.20; grad(input) 1.45; grad(w2/w3) 1.15/1.14; top_k=6 → 5 bf16
  truncations/token. Fix = fp32 HBM + fp32 SBUF accumulator + single downcast; buffer_degree 3→2 →
  grad(w2/w3) 0.98/1.08. Costs ~39% more hardware time for that op. **Do NOT over-apply**: forcing
  the SwiGLU *activation* to fp32 too made it WORSE (1.68) — keep the *accumulator* fp32, not
  everything.
- **RC-3 / RC-4**: already-bf16 operand grad floor; large-scale softmax-bwd cancellation (grad_v worst
  in that regime).
- **d_logits**: ratio ~1.14 but top-1 argmax 100%, |Δloss|~0.001 → judge logits with top-K / cosine /
  loss, not the raw ratio.

## Per-model status (at synthesis time)

DeepSeek ~96%, Qwen3 ~98%, Llama4 ~95% pass; **GPT-OSS is a genuine architectural outlier**
(~50% post-rules but only **~23% raw on attention = 8/35**; block backward **blocked on
`bwmm_shard_on_I.py:992`**) — its four quirks (in-kernel fp32 sinks, sliding-window attention,
swiglu +1 up-bias clamp, gate/up interleave) need bespoke tests; do NOT loosen tolerance to hide the
residual attention drift. Benchmark: Llama3-8B matches GPU at **1.0000** (loss within 0.004%);
shipped DeepSeek fp32 fix landed a **7.6e-4** loss diff. Inference: prefill transfers directly;
decode needs a new metric (a flipped token breaks the fixed reference).

## Exit criteria (the standard)

The precise, hand-to-a-test-author spec
Thresholds come straight from the ratio's meaning: 1.1 = "≤10% worse than bf16
rounding" (clean), 1.2 = production (folds in measured bf16 noise just above 1.1).

- **Single module:** ratio + guard rules; PASS `<1.1` clean / `<1.2` prod. `ABS_TOL=1e-5`,
  `ZERO_EPS=1e-30`, `ZERO_BASELINE_ABS=1e-2`, `QK_NORM_BWD_TOLERANCE_RATIO=1.2`. Neuron-NaN where
  ref finite = hard fail. Coverage = full 7-distribution × N sweep (edge cases may fail but are
  always reported), **forward AND every gradient**, vs BOTH a CPU and a GPU reference. Emit the
  signed-mean bias + max-relative-element companions and **version-stamp** every result. Accept =
  zero real failures on non-edge inputs.
- **Kernel on real hardware:** `rtol=2e-2, atol=1e-5`, 100% compile, runs on-device, zero new
  failures vs the current baseline, routing counts match.
- **End-to-end run:** per-step input hashes match across all workers (separates data desync from
  math); per-layer ratio `<1.10`; cosine `>0.999` (weights `>0.9999`); loss within 1–2%;
  forced-routing KL near the ~1e-3 line (≈4e-4..4e-3 by model). Benchmark to beat: **Llama3-8B =
  ratio 1.0000**, loss within 0.004%.

## Prevention (design rules — each rule is a failure read backwards)

Move the cost from "a PE for 3 weeks" to "a reviewer for 30 seconds":
- **Never accumulate in bf16.** Any running total (MoE expert-combine, softmax normalizer, gradient
  reductions) stays fp32, round to bf16 **once** at the end (prevents RC-2 and RC-4). Precision is a
  scalpel — promoting the *matmul/activation* instead of the *accumulator* makes it WORSE (1.68).
- **Use cos/sin RoPE; never silently detach autograd** (closes the 451× RC-5 backward blowup).
- **Assert an honest reference** (bf16-rounded inputs, not clean fp32), **prove eager is eager**,
  **pin the env** (a stale build fakes 50–90×), **check weight layout before a long run** (a
  scrambled seed announces itself at step 1, not step 1000).
- **Gate on the ratio at merge:** kernel changes can't worsen a module ratio; compiler releases run
  the compile-vs-eager suite; new models/weight-format changes run the honest-reference sweep.

## Scope & open items (training-side)

This body of work is **training-side** accuracy. Known not-yet-covered / caveated areas:
- **Inference / decode accuracy** — separate problem; the fixed-reference ratio breaks once a decode
  token flips, so it needs a different metric. Not covered here.
- **Optimizer state (Adam m/v) comparison** — **do NOT hold it to ratio < 1.1**, especially
  norm-related moments: optimizer state is inherently sensitive early in training and legitimately
  won't match tightly. Judge the *weights* (param-norm compare, `activation-compare`) and the
  *grad-norm difference trajectory* (`large-model-accuracy-bounds`) instead.
- **Grad-norm difference trajectory** — formalized in `large-model-accuracy-bounds`: `grad_norm(GPU
  bf16) − grad_norm(TRN bf16)` should hover ~0 (|diff| ≤ ~0.1 for ~90% of points), early vs late
  sections judged separately.
- **Large-model acceptance without fp32** — use the GPU bf16-ULP perturbation envelope
  (`large-model-accuracy-bounds`), since the fp32-based ratio is infeasible at scale.
- **Real-workload input/weight distributions** for module tests — planned (needs a ~100k-step GPU
  run to capture from); see `moduscope-dist-sweep`.
- **Downstream eval-metric parity** — long-term, not started.

## Common mistakes

- **Re-debugging a known cause.** Check this catalog first; most divergences are one of the 9.
- **Calling a real NaN "just drift"** (or vice-versa). The tag distinguishes: finite ratio = drift/
  real-bug; `[FAIL-targetNaN]` = kernel blew up.
- **Judging logits by the raw ratio.** Use top-K / cosine / loss for the LM head.
- **Assuming a fix isn't shipped.** Most are shipped (see CRs); #7 is a workaround, #3 is a floor.

## How to test this skill

The catalog is a knowledge asset; validate it stays internally consistent:
```bash
grep -q "3-tensor ratio" SKILL.md && grep -q "RC-1" SKILL.md && echo "metric + taxonomy present"
```
Success: the metric and the RC taxonomy are present.
