---
name: moduscope-run-test
description: Per-test knowledge base for the ModuScope BF16 component-equivalence suite (bf16/tests/ test_00..07 dense, test_08/13/14/15/16 MoE). Use to decide WHICH test file to run for a module, which models and TEST_MODE it applies to, what env knobs it honors (RUN_BWD, TORCH_COMPILE, SEQ_LEN, SDPA_FWD_TOLERANCE_RATIO), and the test-specific quirks before launching a run or triaging a failure — routing tie-break skips (08/15/16), capture_router_input (16), deepseek attn_backend=nki / gpt_oss flex+BlockMask+CPU-fallback (13), MoE layer_id (16), seq_len=8192 for qwen3/gpt_oss NKI MoE, gpt_oss +1 up-bias bake (14/15), and score_before_experts=False forward-insensitivity (deepseek). Author-only knowledge base; does not launch live cluster runs.
---

# ModuScope Per-Test Knowledge Base

## Overview

One reference card per test file in `bf16/tests/`. For each test it states the
module under test, which models it applies to, the supported `TEST_MODE`s, and the
test-specific quirks (routing tie-break skips, attention backends, seq_len, weight
re-layout, tolerance knobs) you MUST know before launching a run or triaging a
failure. This is a knowledge base — it does NOT itself launch cluster runs.

## Usage

- You MUST consult this skill before running a specific `test_NN_*.py` to confirm
  the model/TEST_MODE combination is valid and to set the right env knobs.
- Use the quick-reference table to map a module name → test file.
- Use the per-test sections to understand a quirk that appears in a failure log
  (e.g. a `pytest.skip("ill-posed top-k routing ...")`, an `xfail` on log_normal,
  or a `seq_len`-driven NKI kernel assertion).
- Pair with `moduscope-component-tests` (cluster/venv/workflow) and
  `moduscope-dist-sweep` (distribution sweep sizing).

## Core Concepts

### Test families and applicable models

- **Dense tests `test_00`–`test_07`** apply to dense models: `llama3_8b`,
  `llama3_70b`, `qwen3_0.6b/1.7b/4b/8b/14b/32b`. `test_04_feedforward` skips
  MoE-only models (no plain `FeedForward`).
- **MoE tests `test_08`, `test_13`–`test_16`** apply to the 4 registered MoE
  models: `deepseekv3_16b` (MLA), `llama4_17bx16e` (GQA), `qwen3_30b_a3b` (GQA),
  `gpt_oss_20b` (MHA flex + attention sinks + sliding window). Each `_run`
  `pytest.skip`s if `shape_cfg` is not an `MoEShapeConfig` / model not MoE.

### Two execution shapes (drives valid TEST_MODE)

- **Single-process self-check (no S3):** `test_00`, `test_05` (and the other dense
  tests), and `test_08`. They build `ref_fp32` + `ref_bf16` + `target` in ONE
  process and compare. Valid modes: `gpu`, `cpu`, `neuron_cpu` (local reference).
  There is no `gpu_save`/`neuron` golden split.
- **Cross-cluster golden workflow (S3):** `test_13`, `test_14`, `test_15`,
  `test_16`. Phase 1 builds the reference (`gpu`/`gpu_save`/`cpu`/`cpu_save`/
  `neuron_cpu`); `*_save` uploads to S3 and early-returns; `neuron` mode downloads
  the S3 golden and runs only the target. These are the heavy MoE modules.

### The assertion (all tests)

`compare_3tensors(ref_fp32, ref_bf16, target_bf16)` →
`check_3tensor_result(result, name, tol)`; PASS if
`ratio = ||target−fp32||_F / ||ref_bf16−fp32||_F < tol`. Default fwd `tol=1.1`;
backward grads use `BWD_TOLERANCE_RATIO=1.1` (and `QK_NORM_BWD_TOLERANCE_RATIO=1.2`
for `q_norm.weight`/`k_norm.weight` in `test_05`).

## Quick reference

### Required compile mode per test (CRITICAL — set TORCH_COMPILE accordingly)

Each MoE test is validated in a SPECIFIC `torch.compile` mode; running the wrong mode surfaces
spurious NKI errors (eager-only custom-op aliasing checks vs compiled-only graph issues):

| tests | required mode | env |
|---|---|---|
| `test_13` attention, `test_16` transformer_block | **EAGER** | `TORCH_COMPILE=0` (default) |
| `test_14` grouped_experts, `test_15` moe | **COMPILED** | `TORCH_COMPILE=1` |

So a full MoE-model sweep runs the attention/block tests eager and the experts/MoE tests compiled.
If experts/MoE fail in eager with `nki_*_fwd` output-aliasing, re-run them COMPILED before concluding;
if attention/block fail in compiled with a neuron-cc crash, re-run them EAGER.

| Test | Module | Models | TEST_MODE / S3 | Key knobs | Quirks |
|---|---|---|---|---|---|
| `test_00_rms_norm` | RMSNorm (`nn.RMSNorm`) | dense | gpu/cpu/neuron_cpu, no S3 | `RUN_BWD` | uses `DIST_CONFIGS_1D_SAFE` (1D weight excludes fan-based inits); trivial ratio ≈1.0 |
| `test_01_embedding` | token embedding | dense | local, no S3 | `RUN_BWD` | — |
| `test_02_rope` | RoPE | dense | local, no S3 | — | — |
| `test_03_linear_projections` | Col/Rowwise linear | dense | local, no S3 | — | — |
| `test_04_feedforward` | FeedForward (SwiGLU MLP) | dense | local, no S3 | `RUN_BWD` | **skips MoE-only models** |
| `test_05_attention` | GQAttention | dense | local, no S3 | `RUN_BWD`, `QK_NORM_BWD_TOLERANCE_RATIO=1.2` | `log_normal` dists are `xfail` (extreme magnitudes on Neuron) |
| `test_06_transformer_block` | dense block | dense | local, no S3 | `RUN_BWD` | — |
| `test_07_output_head` | output head | dense | local, no S3 | — | — |
| `test_08_masked_topk_router` | `MaskedScoresTopKRouter` (`common.moe.nki_moe`) | MoE | gpu/cpu/neuron_cpu, no S3 | `RUN_BWD` (fwd+bwd split) | ref==target same class → ratio ≈1.0; `skip_if_topk_routing_ill_posed`; 1D gate bias falls back to `normal` init |
| `test_13_attention` | Attention (MLA/GQA/MHA flex+sinks) | MoE | gpu_save→S3→neuron | `SDPA_FWD_TOLERANCE_RATIO` (defined but NOT wired — fwd uses 1.1), `SEQ_LEN` | deepseek target `attn_backend="nki"`; gpt_oss flex + `create_block_mask` + CPU `_NKIStyleAttention` fallback; per-model RoPE backend |
| `test_14_grouped_experts` | GroupedExperts (expert matmul only) | MoE | gpu_save→S3→neuron | `RUN_BWD` (split) | ref→NKI state remap; gpt_oss bake `+1` up-bias + undo `mlp2_bias/=tp`; `block_size=512` gpt_oss else 256; token-order reconcile |
| `test_15_moe` | full MoE (router+experts+reorderer+shared) | MoE | gpu_save→S3→neuron | `RUN_BWD` (split) | `skip_if_topk_routing_ill_posed` on flattened input; `torch.histc` Long-on-CPU patch; gpt_oss bake `+1`; deepseek `score_before_experts=False` fwd-insensitive |
| `test_16_transformer_block` | full block (attn+norms+MoE) | MoE | gpu_save→S3→neuron | `RUN_BWD` (split), `SEQ_LEN` | `layer_id` (deepseek=1 else 0); `capture_router_input` then skip; gpt_oss CPU `_SDPAFallback`; `torch.histc`+`torch.bmm` CPU patches |

`SEQ_LEN` note: `deepseekv3_16b` and `qwen3_30b_a3b` register `seq_len=8192`
(also `gpt_oss_20b`) because NKI MoE `BLOCK_SIZE=512` requires `(T//512) % 16 == 0`
(8192/512=16). The default 4096 yields 8 and fails kernel validation. Do NOT
override `SEQ_LEN` below the registered value for these models.

## Per-test detail (MoE)

### test_08 — MaskedScoresTopKRouter
Reference and target use the SAME class
(`torchtitan.models.common.moe.nki_moe.MaskedScoresTopKRouter`, resolved via
`shape_cfg.router_module_path`), so only arithmetic precision/device differs →
ratio should be ≈1.0. All three modules are built in one process (no S3). The
gate weight is drawn in fp32 then rounded to bf16 once so all builds share
byte-identical operands. The 1D gate bias uses a `normal` init fallback because
fan-based inits (kaiming/xavier, in `_NEEDS_2D`) require ≥2D. Before comparing,
`skip_if_topk_routing_ill_posed(ref_fp32, x1, top_k)` skips distributions whose
fp32 top-k boundary is exactly tied (arbitrary tie-break, no correct answer).
Separate `test_forward` and `test_backward`; backward uses a random bf16-rounded
cotangent (route_norm makes `out.sum()` gradient identically zero otherwise).

### test_13 — Attention (4 MoE models)
- **deepseek_v3:** MLA; target built with `attn_backend="nki"` (ref uses `sdpa`).
- **gpt_oss:** `Attention.Config(use_sliding_attention=True)`, `attn_backend="flex"`
  asserted; mask is a `BlockMask` from `create_block_mask` (causal + sliding
  window). On CPU the ref is patched to a custom autograd `_NKIStyleAttention`
  (`_gpt_oss_cpu_forward`) that implements GQA + sliding window + attention sinks,
  because the NKI flash wrapper has no CPU path.
- **llama4 / qwen3:** `GQAttention.Config` with the model's `rope_backend`.
- RoPE cache is rebuilt per model at `shape_cfg.seq_len`.
- **Forward tolerance — use `TOLERANCE_RATIO` (1.1); do NOT wire in `SDPA_FWD_TOLERANCE_RATIO`.**
  The canonical test_13 `_run` passes `fwd_tol = TOLERANCE_RATIO` (1.1) to the forward verdict.
  `SDPA_FWD_TOLERANCE_RATIO` (default 3.0) is DEFINED at module scope but the canonical test does
  NOT use it for the forward check — it is effectively dead code. To be faithful, author the forward
  check with `TOLERANCE_RATIO` (1.1); wiring 3.0 in makes the attention forward wrongly lenient.
  (Loosening to 3.0 is a deliberate opt-in for MLA drift, not the faithful default.)

### test_14 — GroupedExperts (expert matmul only)
Exercises the SwiGLU expert path (w1/w3 gate-up, w2 down) with no router. Ref→NKI
state-dict remap (`_map_experts_state_dict`): deepseek keys copy directly;
llama4/qwen3 fuse `w1`(gate)+`w3`(up) → `gate_up_proj_weight (E,D,2,I)` and
transpose `w2` → `down_proj_weight`; gpt_oss reshapes `mlp1_weight`/`mlp1_bias`,
transposes `mlp2_weight`, **bakes `+1` into the up-bias** (`mlp1_bias[:,1,:]+=1`,
NKI convention) and, in this single-process test, **undoes** the kernel's
`mlp2_bias /= tp_degree`. Expert config `block_size=512` for gpt_oss else 256.
The ref runs in expert-sorted (routed) order; outputs/grads are reconciled to
token order via `_to_token_order` before comparing.
**Authoring test_14 from scratch?** The ref and target have DIFFERENT forward
signatures (`GroupedExperts(x_sorted, num_tokens_per_expert)` vs
`NKIGroupedExperts(hidden, masked_scores, top_k)`) and need a deterministic routing
bridge. The complete copy-ready harness — `_build_masked_scores` (modulo routing),
`_prepare_routed_input`, `_combine_expert_output`, `_to_token_order`, `_build_experts`
(block_size 512/256), and the neuron wiring — is in
`references/grouped-experts-harness.md`. MUST read it before writing the routing logic
(do NOT reinvent a random-softmax routing).

### test_15 — full MoE
Router + experts + token reorderer + shared experts. `skip_if_topk_routing_ill_posed`
is called on the MoE's flattened `(bs*slen, dim)` input. `torch.histc` is patched
to accept Long on CPU. gpt_oss bakes `+1` into `experts.mlp1_bias[:,1,:]`. Backward
uses `out.sum().backward()`. For deepseek with `score_before_experts=False`, routing
bugs can be invisible in the forward output and only show up in weight grads — a
passing forward does NOT clear a routing change; check the backward grads too.

### test_16 — full TransformerBlock
The block class is shared ref/target; only the inner MoE differs (MoE/GptOssMoE vs
NKIMoE). `layer_id` must activate the MoE path: `deepseekv3_16b=1` (n_dense_layers=1),
all others `0`. `capture_router_input(block.moe.router)` registers a forward-pre-hook
to grab the REAL router input (post-attention hidden states, not the block input),
then `skip_if_topk_routing_ill_posed` uses it. gpt_oss CPU ref is patched
(`_patch_gptoss_for_cpu`) to replace FlexAttention with `_SDPAFallback` + a sink
scaling step. `torch.histc` and `torch.bmm` are patched for CPU bf16/Long.

## Common mistakes

- Running an S3 golden mode (`gpu_save`/`neuron`) for `test_00`/`test_05`/`test_08`
  — those are single-process and ignore the S3 split.
- Overriding `SEQ_LEN<8192` for deepseek/qwen3/gpt_oss MoE → NKI `BLOCK_SIZE`
  assertion failure (not a numeric bug).
- Treating a `pytest.skip("ill-posed top-k routing")` as a failure — it is an
  intentional skip on a tied routing boundary.
- Treating `log_normal_*` `xfail` in `test_05` as a regression — it is a known
  expected failure (extreme magnitudes on Neuron).
- Expecting `test_15`/`test_16` deepseek forward to expose a routing regression
  when `score_before_experts=False` — inspect weight grads.

## How to test this skill

This is a documentation skill; "runnable" means every test/model/quirk claim is
grounded in source. Verify the test inventory and the MoE model registry the table
is built from still match the repo:

```bash
REPO=$WORKSPACE/<neuron-env>-stage/ModuScope/test/component_tests_torchtitan
ls "$REPO"/bf16/tests/test_0[0-7]_*.py \
   "$REPO"/bf16/tests/test_08_*.py \
   "$REPO"/bf16/tests/test_1[3-6]_*.py
grep -nE 'deepseekv3_16b|llama4_17bx16e|qwen3_30b_a3b|gpt_oss_20b' "$REPO"/bf16/tests/conftest.py
grep -n 'skip_if_topk_routing_ill_posed\|capture_router_input' "$REPO"/shared/conftest_base.py
```

Success: all 13 test files list, the 4 MoE model entries print, and both routing
helpers resolve in `conftest_base.py`. (Grounding the quirk claims; no cluster run.)
