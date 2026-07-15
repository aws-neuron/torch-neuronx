---
name: moduscope-test-roadblocks
description: Recovery playbook for RUNTIME roadblocks when developing or running ModuScope component tests on real GPU/Neuron hardware (errors that py_compile cannot catch). Generic recovery procedure — read the FULL source of the failing op to find the real API contract, mimic it, try the OTHER execution mode (eager vs compiled), then attribute env-vs-test by running the framework's OWN equivalent test — plus a TAILORED catalog of concrete roadblocks seen in this framework and their fixes: repo version skew (missing conftest_base helper / load_ref_module skip_clear), CPU op gaps (torch.histc Long, torch.bmm bf16), NKI Config construction idiom, NKI attention nki_attn_fwd output-aliasing (eager) / neuron-cc proto-parse crash (compiled), routing-tie skips, seq_len=8192, use_grouped_mm=False. Use when a generated test py_compiles but errors at collection/build/compile/run, on ImportError/TypeError/NotImplementedError/RuntimeError from conftest/torchtitan/NKI, or when choosing eager vs compiled mode.
---

# ModuScope Test Roadblocks — Recovery Playbook

## Overview

A generated component test can `py_compile` cleanly yet fail at collection / module-build / NKI-compile
/ execution on a real node. This skill is the recovery playbook: a generic procedure for ANY runtime
roadblock plus a tailored catalog of the concrete failures seen in this framework and their fixes. It
complements `moduscope-reference-fidelity` (generic: how to build a faithful wrapper) and
`moduscope-result-triage` (classify ratio failures). This skill is about runtime EXCEPTIONS, not ratios.

## Usage

Use this skill when a ModuScope test, after py_compile, hits a runtime error: an `ImportError`/`TypeError`
from `conftest`/`conftest_base`, a `NotImplementedError` from a torch op on CPU, a `RuntimeError` from an
NKI custom op, a neuron-cc `COMPILATION FAILED`, or any "works in py_compile, dies on the node" symptom.
Also use it to decide eager vs compiled mode.

## Core Concepts

### Generic recovery procedure (apply to ANY runtime roadblock)

1. **Do NOT assume the generated test is wrong.** Read the FULL source of the failing op / module /
   function — not just its signature: args, dtypes, shapes, device assumptions, what it returns, what it
   mutates/aliases. The real API contract is in the model/framework source.
2. **Mimic the contract exactly** in the test wrapper (the derive-don't-invent rule of
   `moduscope-reference-fidelity`). Most TypeErrors/shape errors are a contract you guessed instead of read.
3. **Try the OTHER execution mode.** Tests parametrize over `COMPILE_MODES` via `TORCH_COMPILE`
   (`0`=eager, `1`=compiled `torch.compile(backend="neuron")`, `both`). A kernel can fail in ONE mode only
   (eager custom-op aliasing checks; or neuron-cc graph compile). Run eager first (fastest signal), then
   compiled. Never conclude "broken" from a single mode.
4. **Attribute env-vs-test: run the FRAMEWORK's OWN equivalent test** (`test_NN_*.py`) in the SAME mode on
   the SAME node. If the canonical test fails identically → the fault is ENVIRONMENTAL (kernel / venv /
   branch / repo-version), the generated test is behaviorally equivalent, and it is NOT yours to fix. If
   the canonical PASSES and yours fails → diff the setup; the contract differs.
5. **If environmental/kernel:** document it, file to the owning team (Neuron / model), and OPTIONALLY apply
   a reduced-coverage workaround to keep the rest testable (e.g. run attention with `attn_backend="sdpa"`
   on Neuron to bypass a broken NKI attention kernel and still validate the surrounding paths) — clearly
   labelling it as a reduced-coverage variant, not the intended NKI comparison.

### Generic vs tailored (how the skills divide)

- **Generic** (`moduscope-reference-fidelity`): HOW to build a faithful 1:1 wrapper from source.
- **Tailored** (this skill): the concrete roadblocks THIS framework throws and their exact resolutions,
  plus the procedure above. When you hit something not in the catalog, fall back to the generic procedure.

## Known roadblocks catalog (empirically observed on live runs)

| Symptom (error) | Stage | Cause | Fix |
|---|---|---|---|
| `ImportError: cannot import name 'skip_if_topk_routing_ill_posed' from 'conftest'` | collect | Node repo OLDER than the repo you authored against (helper absent) | Import from `conftest_base` if present; else align/append the helper (it's pure torch). Repo-version skew, not a test bug. |
| `TypeError: load_ref_module() got an unexpected keyword argument 'skip_clear'` | collect/build | Same version skew — older `conftest_base` loader lacks `skip_clear` | Align `conftest_base.py` to the version the test targets (back up first), or drop the kwarg. |
| `NotImplementedError: "histogram_cpu" not implemented for 'Long'` | ref build (CPU) | MoE router/reorderer calls `torch.histc` on int64; unsupported on CPU | Module-top `torch.histc` Long→float→long monkeypatch (see `moduscope-reference-fidelity` "CPU-reference op gaps"). Needed for MoE (test_15) and blocks (test_16). |
| `[Errno 13] Permission denied: '/tmp/neuron_backend/...'` or `Failed to create directory: /tmp/hlo_cache/...` (also `COMPILATION FAILED: filesystem error: cannot create directories`) | NKI compile (Neuron) | freshly-allocated node has stale `/tmp/neuron_backend` / `/tmp/hlo_cache` owned by a PRIOR user → the neuron compile cache can't be written | **ALWAYS** `sudo chmod -R 777 /tmp` (or at least `/tmp/neuron_backend /tmp/hlo_cache`) at the START of every node run — bake it into the launcher. Environmental node-setup, NOT a test bug. |
| `RuntimeError ... bmm ... bf16` on CPU | ref build (CPU) | block combine uses `torch.bmm` bf16 on CPU | Module-top `torch.bmm` bf16→fp32 monkeypatch (blocks/test_16). |
| `AttributeError`/`TypeError` building `NKIxxx` | target build | Wrong Config construction idiom | grouped_experts: `cfg=ExpertsCls.Config(); cfg.dim/...; ExpertsCls(cfg)`. moe: `NKIMoE.Config(**kwargs).build(dim=...)`. Read the module's own usage. |
| `RuntimeError: nki_attn_fwd ... output ... must not also be an input ... return a clone` | forward (eager, Neuron) | NKI attention custom op returns an output aliasing an input (torch custom-op constraint) — **kernel bug in the optimized branch** | NOT test-fixable and NOT mode-fixable (compiled fails too). Confirm via the framework's own `test_13` (fails identically) → file to Neuron/model team. Optional reduced-coverage workaround: target `attn_backend="sdpa"` on Neuron. |
| `RuntimeError: nki_<op>_fwd ... output ... must not also be an input ... return a clone` for ANY nki custom op (e.g. `nki_moe_fwd` on qwen3 MoE, `nki_attn_fwd` on deepseek attention) | forward (Neuron) | SAME output-aliasing class, but it is **per-model / per-kernel**: deepseek `nki_moe_fwd` is FINE (test_15 passes) yet qwen3 `nki_moe_fwd` aliases (test_14/15 fail); deepseek `nki_attn_fwd` aliases yet qwen3 attention is sdpa (no custom op, passes). So check the SPECIFIC model+module kernel | Kernel bug in that model's NKI op — NOT test-fixable. Attribute via the framework's own `test_NN` for that model. No generic fallback for MoE (unlike attention's sdpa). File the exact `nki_<op>_fwd` to the Neuron/model team. **BEFORE filing, check two things that resolved it for qwen3 MoE:** (1) use the REQUIRED per-test mode — experts/MoE need `TORCH_COMPILE=1` (compiled); eager triggers the `nki_*_fwd` aliasing check; (2) ensure `TARGET_PATH` is the CURRENT branch — older trees (e.g. `deepseek_v3_optimized`) had a qwen3 MoE weight-grad `dma_copy bf16->f32` bug fixed in the newer tree (`cr_validation`). qwen3_30b_a3b passes test_14/15 with compiled mode + the current tree. |
| `COMPILATION FAILED ... Failed to parse proto ... module.mlir` (neuron-cc) | forward (compiled, Neuron) | neuron-cc compile crash on the same attention path | Same root cause as above, surfacing in compiled mode. Environmental; not test-fixable. |
| `attn_backend="sdpa"` fallback: forward PASSES but backward `grad(input)`/weight grads are `[FAIL-targetNaN]` | backward (Neuron, sdpa fallback for MLA) | Neuron SDPA BACKWARD produces NaN/Inf for MLA on this branch | Forward via the sdpa fallback gives a valid equivalence ratio (attention plumbing sound); backward is a separate Neuron SDPA-backward numerical issue — environmental, correctly tagged by `check_3tensor_result`. Treat the fallback as forward-meaningful; file the backward NaN to Neuron. |
| `pytest.skip("ill-posed top-k routing ...")` | run | A dist's fp32 top-k boundary is exactly tied (arbitrary tie-break) | Expected skip, not a failure (`skip_if_topk_routing_ill_posed`). |
| NKI MoE kernel shape assert | build | `seq_len` violates `(T//512)%16==0` | Use `seq_len=8192` for deepseek/qwen3/gpt_oss MoE (registry encodes it). |
| baseline `||ref_bf16-ref_fp32||==0` → degenerate ratio | compare | ref used `use_grouped_mm=True` (casts to bf16, collapses fp32 baseline) | Build ref with `use_grouped_mm=False` (for-loop path). |
| Concurrent module runs on different cores all report `[SBM] Stack OOM` with a SHRINKING `available=` (e.g. 3216→1296) | NKI compile | the SBM stack/scratch pool is contended across concurrent runs on one node → false OOMs | Run kernel-heavy module tests ONE AT A TIME (or few) per node; re-test a suspected OOM SOLO before believing it. |
| Scaling a large MoE down to fit one core hits a DIFFERENT NKI wall per shape: big hidden (384-512)→`Stack OOM`; small dim/odd head_dim (1024/64)→`Generated MLIR failed verification`; small hidden (96)→`nisa.dma_copy HWDGE requires same src/dst element type (bf16 vs f32)` | NKI compile | the model's NKI kernel is tuned for its REGISTERED shapes (e.g. qwen3 30B-A3B); arbitrary scaled shapes exercise untuned/buggy kernel paths | Kernel-side limits/bugs — NOT test-fixable by shape-tuning. Keep kernel-legal multiples (head_dim=128, dim & hidden multiples of 512) if you must scale, but expect kernel paths to break. For single-core module validation pick a model whose kernel already fits (deepseek-16B passes); escalate the large model to the Neuron/kernel team. |
| `[NCC_INKI016] Kernel validation exception: Stack out of memory` / `[SBM] Stack OOM: requested=N, available=M` (grouped_experts/moe target on Neuron, large-dim model e.g. qwen3 235B, dim=4096) | NKI kernel compile (fwd or bwd) | the NKI weight-grad kernel (`_compute_gate_up_projection_weight_grad`) needs more per-core stack than available for large dims. Independent of expert count (full 128 AND a 2-expert single-core proxy both OOM) and of `RUN_BWD` (the bwd kernel compiles regardless) | Per-core kernel RESOURCE limit — NOT test-fixable. A large MoE needs expert/tensor-parallel across MULTIPLE cores, which a single-process module test does not provide. Recoveries that do NOT help: a smaller `*-singlecore` proxy flavor, `RUN_BWD=0`, fewer experts. File to the Neuron/kernel team. For single-core module validation pick a model whose kernel fits (e.g. deepseek-16B passes). |

## Mode selection (eager vs compiled)

- `TORCH_COMPILE=0` (eager) — default; fastest to first signal; runs custom ops directly (eager aliasing
  checks active).
- `TORCH_COMPILE=1` (compiled) — `torch.compile(backend="neuron")`; exercises neuron-cc graph compile.
- `TORCH_COMPILE=both` — runs each test under both, parametrized (`-eager-`/`-compiled-` in the id).
- A backend error in ONE mode is NOT a verdict — try the other. A test that authors a module SHOULD work
  in both; when only one is viable on a given branch, document which and why.

## Common mistakes

- **Calling an environmental failure a test bug.** Always run the framework's own `test_NN` to attribute
  before "fixing" the generated test.
- **Concluding "broken" from one mode.** Eager and compiled exercise different code paths for NKI ops.
- **Guessing an API contract.** Read the full source; a `TypeError`/aliasing error usually means a
  contract you assumed instead of read.

## How to test this skill

```bash
<skill-builder-validator> \
  $SKILLS/moduscope-test-roadblocks/SKILL.md
grep -q "run the FRAMEWORK's OWN equivalent test" $SKILLS/moduscope-test-roadblocks/SKILL.md \
 && grep -q "nki_attn_fwd" $SKILLS/moduscope-test-roadblocks/SKILL.md && echo CATALOG_OK
```

Success: validator passes and `CATALOG_OK` prints (the attribution rule and the nki_attn_fwd entry are present).
