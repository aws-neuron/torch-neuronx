---
name: moduscope-model-profile
description: Introspect a torchtitan model's config tree and emit a JSON "model profile" (attention type MLA/GQA/MHA-flex+sinks, dim/layers/vocab/norm_eps, head counts, rope backend, full MoE params, required seq_len incl. the NKI MoE BLOCK_SIZE=512 constraint, and which test files apply). Use when onboarding a new model into the ModuScope BF16 component-test framework, before deciding TEST_MODE/SEQ_LEN/which tests to run, or whenever you need a single authoritative summary of a model's shape for deepseekv3_16b, qwen3_*, llama3_*, llama4_17bx16e, or gpt_oss_20b. Triggers include "model profile", "what attention does X use", "is X MoE or dense", "what seq_len for X".
---

# ModuScope Model Profile

## Overview

Given a torchtitan model name + a REF_PATH torchtitan tree, this skill produces a
deterministic JSON **model profile** that every later ModuScope step reads to decide
attention path, applicable test files, required seq_len, and MoE shape. It reuses the
test suite's own config loaders so the profile cannot drift from what the tests see.

## Usage

Use this skill when:
- Onboarding a NEW model into `component_tests_torchtitan` and you need its shape facts before editing `MODEL_REGISTRY` / `DEVICE_CLASS_MAP`.
- Deciding `TEST_MODE`, `SEQ_LEN`, `TP`, or which `test_*` files to run for a model.
- Answering "is this model dense or MoE?", "MLA or GQA?", "does seq_len satisfy the NKI MoE kernel constraint?".
- A downstream skill (run / triage / artifact) needs one authoritative shape summary instead of re-parsing torchtitan configs.

Run the script, read the JSON, act on it. Do NOT hand-parse torchtitan config classes — the profile already captured the fields, classified attention, and flagged seq_len problems.

## Core Concepts

### What a profile contains
Top-level keys emitted by `scripts/profile_model.py`:

| Key | Meaning |
|---|---|
| `model_name` / `torchtitan_model` / `flavor` | registry key, `torchtitan.models.<x>` package, config flavor (e.g. `16B`) |
| `group` | `dense` or `moe` — drives which tests apply |
| `block_class` / `block_module` / `config_attr` | block class name, its module, and the `{model}_configs` dict name (override) |
| `tp_degrees` / `seq_len` | TP degrees tested; required sequence length |
| `dim` / `n_layers` / `vocab_size` / `norm_eps` / `n_dense_layers` | core dims (deepseek `n_dense_layers=1` → block test uses `layer_id=1`) |
| `attention` | `{type, n_heads, n_kv_heads, head_dim, qk_norm, ...}` — see classification below |
| `rope` | rope backend config class + params (`theta`, `factor`, …) |
| `has_feed_forward` / `ff_hidden_dim` | dense SwiGLU MLP presence (MoE-only models → `false`) |
| `moe` | `null` for dense; else full MoE params (below) |
| `applicable_tests` | exact `test_*.py` list for this group |
| `seq_len_constraint` | NKI MoE BLOCK_SIZE=512 check + remediation note |
| `device_class_map` | role→(ref,target) class tuples (MoE models, best-effort from conftest_base) |
| `_sources` | provenance: which registry + loader were used |

### Attention classification (briefing rule, implemented in `_detect_attention`)
1. `MLA` — if `q_lora_rank` or `kv_lora_rank` present (deepseek_v3). Adds `mla` block with `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`.
2. `MHA_flex_sinks` — gpt_oss: MHA + FlexAttention + attention sinks (+ sliding window). Adds `flex` block.
3. `GQA` — `n_kv_heads < n_heads` (llama3/llama4/qwen3).
4. `MHA` — otherwise.

### MoE params (only when `group == moe`)
`num_experts`, `moe_hidden_dim` (full, unsharded), `top_k`, `num_shared_experts`,
`score_func` (softmax|sigmoid), `route_norm`, `route_scale`, `gate_bias`,
`score_before_experts`, `expert_class` (`GroupedExperts` vs `GptOssGroupedExperts`),
`swiglu_limit` (gpt_oss), `moe_module_path`, `router_module_path`. These are extracted
the same way conftest's `_build_moe_shape_config` builds `MoEShapeConfig`
(`type(moe).__qualname__` decides the expert class + module path).

### The NKI MoE seq_len constraint (MUST honor)
The NKI MoE kernel uses `BLOCK_SIZE=512` and requires `(seq_len // 512) % 16 == 0`.
8192 satisfies it (8192/512 = 16); the default 4096 yields 8 and FAILS kernel
validation. `deepseekv3_16b` and `qwen3_30b_a3b` therefore register `seq_len=8192`.

## HuggingFace entry (HF -> titan mapping)

ModuScope tests torchtitan modules, so a HuggingFace model must be MAPPED to a titan
`model_name`+`flavor` before profiling. Read the HF config (`transformers.AutoConfig.from_pretrained(id)`)
and classify from its fields, then map:

| HF signal (config / architectures) | Architecture | titan `model_name` |
|---|---|---|
| `q_lora_rank`/`kv_lora_rank` present (DeepseekV2/V3) | MLA + MoE | `deepseek_v3` |
| `Qwen3MoeForCausalLM` / `num_experts`>0 + qwen3 | GQA + MoE | `qwen3` (flavor `30B-A3B`) |
| `Qwen3ForCausalLM`, no experts | GQA dense | `qwen3` (dense flavor) |
| `Llama4*` / interleaved-MoE llama | GQA + MoE | `llama4` |
| gpt-oss (sliding window + attention sinks + experts) | MHA flex + MoE | `gpt_oss` |
| `LlamaForCausalLM` / `Llama3`, no experts | GQA dense | `llama3` |
| `MixtralForCausalLM` etc. with NO titan port | MoE | **ESCALATE** — needs a torchtitan port |

Classification heuristics (general, so the zoo can change): MoE if `num_experts`/`num_local_experts`/
`n_routed_experts` > 0; MLA if `q_lora_rank`/`kv_lora_rank`; GQA if `num_key_value_heads` <
`num_attention_heads`; gpt_oss if sliding-window + sinks. Map `top_k`/`num_experts`/`hidden_dim`/
`num_attention_heads` to the titan flavor. If the HF architecture has no titan equivalent, ESCALATE
(porting torchtitan is out of scope) — do NOT invent a mapping. Once mapped, profile the titan name
normally; the titan config is the authoritative shape source for the test.

## Registered model shapes & single-core kernel status (hardware-observed)

Module-level shapes come from the config; at `tp=4` the per-core expert interim is `moe_hidden_dim//4`.
Per-core NKI weight-grad kernel stack ≈ `16 B × (moe_hidden_dim//tp)`; one Neuron core has ~3216 B
(solo) — so single-core fits when `moe_hidden_dim//tp` ≲ 200. Concurrency shrinks the shared pool
(false OOMs) → run kernel-heavy module tests one at a time.

| model | dim | moe_hidden | experts/top_k | attn | seq_len | single-core kernel status (neuron_cpu) |
|---|---|---|---|---|---|---|
| `deepseekv3_16b` | 2048 | 1408 | 64 / 6 | MLA (nki) | 8192 | test_14/15 ✅ PASS fwd+bwd; test_13/16 ❌ `nki_attn_fwd` aliasing (MLA kernel) |
| `qwen3_30b_a3b` | 2048 | 768 | 128 / 8 | GQA (sdpa) | 8192 | ✅ ALL PASS fwd+bwd: test_13 (EAGER), test_14/15 (COMPILED) on the CURRENT target tree |
| `qwen3_235b` (A22B) | 4096 | 1536 | 128 / 8 | GQA (sdpa) | 8192 | ✅ test_15 MoE PASS fwd+bwd at **tp16** (per-core hidden 96), COMPILED + current tree (cr_validation). Full tp4 (hidden 384) needs more per-core stack — raise tp to shrink the per-core interim |
| `llama4_17bx16e` | — | — | — | GQA | 4096 | not yet single-core-validated here |
| `gpt_oss_20b` | — | — | — | MHA flex+sinks | 8192 | not yet single-core-validated here |

Takeaways for picking a quick single-core smoke: **deepseek-16B** (experts/MoE) and **qwen3_30b_a3b**
(all 3 modules) are validated-PASS combos. To fit a LARGE MoE (e.g. qwen3_235b) on one core, **raise the
TP degree on the REAL config** (tp8/tp16 → per-core interim `moe_hidden_dim//tp` of 192/96) so the kernel
fits while keeping correct numerics — qwen3_235b test_15 PASSES fwd+bwd at tp16. Do NOT hand-craft a
"singlecore" proxy with ad-hoc dims (it produced 150x-wrong numerics — the dims don't line up). NKI
custom-op aliasing (`nki_*_fwd`) and Stack-OOM seen on OLDER target trees / wrong mode are resolved by
COMPILED mode (experts/MoE) + a CURRENT target tree — see `moduscope-run-test` and `moduscope-test-roadblocks`.
The profile's `seq_len_constraint.satisfied` flag and `note` make this explicit — you
SHOULD NOT launch MoE tests for a model whose `satisfied` is `false`.

### Which tests apply
- `dense` → `test_00`–`test_07` (rms_norm, embedding, rope, linear, feedforward, attention, block, output head).
- `moe` → `test_08` (router), `test_13` (attention), `test_14` (grouped_experts), `test_15` (moe), `test_16` (transformer_block).

### Grounding & robustness
The script reuses `conftest_base.load_model_configs(REF_PATH, model_name, config_attr)`
— the exact function the tests call to import `{model}_configs`. If `conftest_base`
cannot be imported (e.g. `neuron_training_evaluations` is absent in an author/CI env),
it falls back to a byte-identical inline copy of that loader, so profiling still works.
The model list comes from the live `conftest.MODEL_REGISTRY` when importable, else an
embedded mirror. `_sources` in the output records which path was taken — check it if a
field looks stale.

## Deterministic Scripts

### scripts/profile_model.py
Emits the JSON profile for one model.

```bash
python scripts/profile_model.py --model <registry-key> --ref-path <REF_PATH> [options]
```
Required:
- `--model` registry key (e.g. `deepseekv3_16b`, `qwen3_8b`, `gpt_oss_20b`, `llama4_17bx16e`).
- `--ref-path` the REF (plain MoE) torchtitan tree — same value as `REF_PATH` in the test suite.

Optional:
- `--repo-root <component_tests_torchtitan>` to use the live `conftest.MODEL_REGISTRY` + `conftest_base.DEVICE_CLASS_MAP`. Omit to use the embedded mirror.
- `--seq-len N` override the required seq_len (re-checks the NKI constraint).
- `-o FILE` write JSON to a file instead of stdout.

Example (deepseek, MLA + MoE):
```bash
python scripts/profile_model.py \
  --model deepseekv3_16b \
  --ref-path /workspace/upstream_torchtitan \
  --repo-root $WORKSPACE/<neuron-env>-stage/ModuScope/test/component_tests_torchtitan \
  -o /tmp/deepseekv3_16b.profile.json
```

## How to test this skill

Structural (no cluster, no torchtitan tree needed — proves the script is sound and the
registry/CLI work):
```bash
cd $SKILLS/moduscope-model-profile
python -m py_compile scripts/profile_model.py && echo PY_COMPILE_OK
python scripts/profile_model.py --model bogus --ref-path /tmp/none   # prints "Unknown model ... Known (embedded-mirror): [...]"
```
Success = `PY_COMPILE_OK` printed and the bogus-model run lists the 12 known registry keys.

Functional (run by the future user where a torchtitan tree exists):
```bash
export REF_PATH=/workspace/upstream_torchtitan   # or the live REF_PATH
python scripts/profile_model.py --model deepseekv3_16b --ref-path "$REF_PATH" | python -m json.tool
```
Success = valid JSON where `group == "moe"`, `attention.type == "MLA"`,
`seq_len == 8192`, `seq_len_constraint.satisfied == true`, and `applicable_tests`
lists `test_08_masked_topk_router.py` … `test_16_transformer_block.py`.

## Common Mistakes

- **Profiling with the wrong `--ref-path`.** Use the REF (plain MoE) tree, not the NKI TARGET tree; expert/MoE *class names* differ between them — `device_class_map` already records both `ref` and `target`.
- **Ignoring `seq_len_constraint.satisfied`.** Launching MoE tests at seq_len=4096 fails the NKI kernel for the 8192-registered models. Use 8192 for `deepseekv3_16b` / `qwen3_30b_a3b` / `gpt_oss_20b`. NOTE `llama4_17bx16e` intentionally runs at 4096 — the constraint is ADVISORY; the registry's seq_len is authoritative, do not "fix" a model that already runs.
- **Trusting a stale field.** If `_sources.registry` is `embedded-mirror` but conftest.py changed, pass `--repo-root` to pull the live registry, or update `MODEL_REGISTRY_MIRROR` to match.
