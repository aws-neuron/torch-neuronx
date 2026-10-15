---
name: moduscope-statedict-mapping
description: Author and verify the reference→target (NKI) state-dict mapping for a NEW MoE model in the ModuScope BF16 component test framework (test_14 GroupedExperts, test_15 full MoE). Covers the three weight-fusion families (deepseek direct, llama4/qwen3 gate_up fuse, gpt_oss mlp1/mlp2 reshape), router and shared-expert keys, and the gpt_oss +1 up-bias / tp_degree single-process conventions. Use when onboarding a new MoE model and the NKI module load_state_dict fails on shape/key mismatch, when grad-comparison keys do not line up between ref and target, or when you need to confirm fused weight shapes before a cross-cluster run.
---

# ModuScope State-Dict Mapping (ref → NKI target)

## Overview
A NEW MoE model's reference module (plain torchtitan `GroupedExperts`/`MoE`) and its NKI target
(`NKIGroupedExperts`/`NKIMoE`) store the SAME math under DIFFERENT key names and tensor layouts.
test_14/test_15 bridge them with pure re-layout maps (stack/permute/transpose/reshape). This skill
documents the exact mapping per family with shapes and the real source logic, and ships
`scripts/verify_mapping.py` to assert every mapped tensor shape equals the target's `state_dict()`.

## Usage
Use this skill when onboarding a MoE model into `bf16/tests/` and you hit:
- `load_state_dict` / `set_weights_from_state` size-mismatch or missing/unexpected key errors when
  loading ref weights into the NKI target.
- Backward test reports `grad(<name>)` skipped or shape-mismatch — ref grad keys not lining up with
  fused target keys.
- A new model whose expert layout you must classify into one of the three mapping families.
- You want to confirm fused `gate_up_proj_weight` / reshaped `mlp1_weight` shapes BEFORE launching a
  cross-cluster (`gpu_save` → S3 → `neuron`) run.

Axis convention used below: **E** = num_experts, **D** = model dim (hidden state),
**I** = expert intermediate (`hidden_dim`, TP-sharded as `moe_hidden_dim // tp_degree`).
NOTE: test_14 source comments label the model-dim axis `H` and intermediate `I`; test_15 source
labels model-dim `D` and intermediate `H`. They denote the SAME axes — do not be misled by the
letter. This skill uses (E, I, D) consistently.

## Core Concepts

The mapping is invoked in two places, both keyed off `target_module.state_dict()`:
- **test_14** `_map_experts_state_dict(model_name, ref_state, target_module)` — experts only.
- **test_15** `_map_state_dict_ref_to_nki(model_name, ref_state, target_module)` →
  `_map_ref_to_target_layout` → per-key `_fuse_ref_weights` / `_reshape_weight` — full MoE (router +
  experts + shared).

Because the maps are pure re-layouts, the SAME function is reused to put ref **gradients** into the
target layout for backward comparison (test_14 calls `_map_experts_state_dict` on grad dicts;
test_15 calls `_map_ref_to_target_layout` with `fill_missing_with_zeros=False`).

Detection order in test_14 (verbatim intent): if ref keys == target keys AND shapes match → return
ref unchanged (DeepSeek); elif `gate_up_proj_weight` in target keys → llama4/qwen3 fuse; elif
`mlp1_weight` in target keys → gpt_oss reshape; else direct copy.

## The three mapping families

### 1. DeepSeek — identical keys, direct copy
Ref `GroupedExperts` and `NKIGroupedExperts` share keys `w1`, `w2`, `w3` with identical shapes →
no transform. test_14:
```python
if set(ref_state.keys()) == tgt_keys:
    shapes_match = all(
        ref_state[k].shape == getattr(target_module, k).shape for k in ref_state if hasattr(target_module, k)
    )
    if shapes_match:
        return ref_state
```
Shapes: `w1`,`w3` = gate/up `(E, I, D)`; `w2` = down `(E, D, I)`. Copied as-is.

### 2. llama4 / qwen3 — fuse gate+up, transpose down
Target replaces `w1`+`w3` with a single fused `gate_up_proj_weight` and renames/transposes `w2` to
`down_proj_weight`. test_14 (experts-only):
```python
w1 = ref_state["w1"]          # (E, I, D)  gate
w3 = ref_state["w3"]          # (E, I, D)  up
w2 = ref_state["w2"]          # (E, D, I)  down
gate_up = torch.stack([w1, w3], dim=1)            # (E, 2, I, D)
mapped["gate_up_proj_weight"] = gate_up.permute(0, 3, 1, 2).contiguous()  # (E, D, 2, I)
mapped["down_proj_weight"]    = w2.transpose(-2, -1).contiguous()         # (E, I, D)
```
test_15 `_fuse_ref_weights` does the same fuse keyed by target name (prefix-aware so it also handles
shared/nested expert keys), asserting the fused shape:
```python
fused = torch.stack([w1, w3], dim=1)              # (E, 2, I, D)
fused = fused.permute(0, 3, 1, 2).contiguous()    # (E, D, 2, I)
assert fused.shape == tgt_shape, f"gate_up_proj fuse shape mismatch: {fused.shape} vs {tgt_shape}"
```
and for down: `w2.transpose(-1, -2).contiguous()` → `(E, I, D)` (falls back to `w2.reshape(tgt_shape)`).

| ref key | ref shape | target key | target shape | op |
|---|---|---|---|---|
| w1 + w3 | (E, I, D) ×2 | gate_up_proj_weight | (E, D, 2, I) | `stack(dim=1)` then `permute(0,3,1,2)` |
| w2 | (E, D, I) | down_proj_weight | (E, I, D) | `transpose(-2,-1)` |

### 3. gpt_oss — reshape fused mlp + bias conventions
Ref `GptOssGroupedExperts` stores a pre-fused `mlp1` (gate+up interleaved on one axis) plus biases.
test_14:
```python
mlp1_w = ref_state["mlp1_weight"]   # (E, 2*I, D)
mlp1_b = ref_state["mlp1_bias"]     # (E, 2*I)
mlp2_w = ref_state["mlp2_weight"]   # (E, D, I)
mlp2_b = ref_state["mlp2_bias"]     # (E, D)
E = mlp1_w.shape[0]; H = mlp1_w.shape[2]; I = mlp1_w.shape[1] // 2   # H == D
mapped["mlp1_weight"] = mlp1_w.reshape(E, 2, I, H).permute(0, 3, 1, 2).contiguous()  # (E, D, 2, I)
mapped["mlp1_bias"]   = mlp1_b.reshape(E, 2, I).contiguous()                          # (E, 2, I)
mapped["mlp2_weight"] = mlp2_w.transpose(-2, -1).contiguous()                         # (E, I, D)
mapped["mlp2_bias"]   = mlp2_b                                                        # (E, D) — same
```
test_15 `_reshape_weight` performs the equivalent reshape (`reshape(E,2,H2//2,D).permute(0,3,1,2)`
for `mlp1_weight`; `reshape(E,2,H2//2)` for `mlp1_bias`).

| ref key | ref shape | target key | target shape | op |
|---|---|---|---|---|
| mlp1_weight | (E, 2I, D) | mlp1_weight | (E, D, 2, I) | `reshape(E,2,I,D)` then `permute(0,3,1,2)` |
| mlp1_bias | (E, 2I) | mlp1_bias | (E, 2, I) | `reshape(E,2,I)` |
| mlp2_weight | (E, D, I) | mlp2_weight | (E, I, D) | `transpose(-2,-1)` |
| mlp2_bias | (E, D) | mlp2_bias | (E, D) | identity |

**gpt_oss bias conventions (applied AFTER mapping, in the test — NOT inside the map fn):**
- **Bake +1 into the up-bias.** The NKI kernel expects up-bias already offset by 1; the GPU ref adds
  the +1 inside its swiglu activation. Both tests do:
  `target.experts.mlp1_bias.data[:, 1, :] += 1.0`  (slice `[:, 1, :]` = the *up* half of the (E,2,I) bias).
- **Single-process `mlp2_bias` / tp_degree undo — test_14 ONLY.** The NKI kernel divides `mlp2_bias`
  by `tp_degree` expecting a later all-reduce. In the single-core test there is no all-reduce, so
  test_14 multiplies it back: `target.mlp2_bias.data *= shape_cfg.tp_degree`.
  test_15 does NOT do this (it only bakes the +1). Preserve this difference when authoring a new
  gpt_oss-like model — do not add the tp_degree undo to test_15.

## Router and shared experts
Both are passed through `_map_state_dict_ref_to_nki` with matching keys/shapes, so they hit the
`ref_val.shape == tgt_shape` branch in `_map_ref_to_target_layout` and copy directly:
- **Router gate:** ref `router.gate.weight` (+ optional `router.gate.bias`) → identical target keys
  (ref `TokenChoiceTopKRouter` ↔ NKI `MaskedScoresTopKRouter` share gate-weight shapes).
- **Shared experts:** ref shared `FeedForward` (`w1`/`w2`/`w3`) → identical structure in the NKI MoE.
  (DeepSeek's `NKIMoE.Config` takes `num_shared_experts`; llama4/qwen3 NKI configs do not.)
Any target key absent from the ref dict is zero-filled (`fill_missing_with_zeros=True`) for weights,
and skipped (`None`) for grads.

## Onboarding a new MoE model — mapping checklist
1. Build ref + target via `load_class_for(model, "grouped_experts"|"moe", "ref"|"target")` and print
   `ref.state_dict()` vs `target.state_dict()` keys+shapes.
2. Classify the expert layout into one of the three families above. If the target exposes
   `gate_up_proj_weight` → family 2; `mlp1_weight` → family 3; identical keys+shapes → family 1.
3. If it is a NEW layout, add a branch to BOTH `_map_experts_state_dict` (test_14) and
   `_fuse_ref_weights`/`_reshape_weight` (test_15). Keep it a pure re-layout so it is grad-safe.
4. Apply any kernel bias conventions (e.g. gpt_oss +1 / tp_degree) AFTER `set_weights_from_state`,
   in the test body, never inside the map function.
5. Run `scripts/verify_mapping.py` for both modules and confirm the shape table is all ✅.

## Deterministic Scripts

### scripts/verify_mapping.py
Builds the ref and target modules for one model+module via the framework's `load_class_for`, runs
the REAL mapping function imported from `test_14_grouped_experts.py` / `test_15_moe.py`, and asserts
every mapped tensor's shape equals the target `state_dict()` shape. Prints a per-key table and exits
non-zero on any mismatch. This is the skill's testability anchor.

Run on a cluster node (TRN target needs the NKI tree in `TARGET_PATH`; for a quick shape-only check
both REF_PATH and TARGET_PATH may point at trees that expose the classes — no Neuron device needed
since nothing is executed, only constructed and mapped):
```bash
source $SHARED/<neuron-env>/private-torch-neuronx/.venv/bin/activate
export REF_PATH=/path/to/ref/torchtitan_tree
export TARGET_PATH=/path/to/nki/torchtitan_tree
export COMPONENT_TESTS_DIR=$WORKSPACE/<neuron-env>-stage/ModuScope/test/component_tests_torchtitan
python scripts/verify_mapping.py --model deepseekv3_16b --module grouped_experts --tp 4
python scripts/verify_mapping.py --model llama4_17bx16e  --module moe             --tp 4
```
Success criterion: prints `MAPPING OK` and exits 0; every row shows `match=True`.

## How to test this skill
Author-time (no cluster, structural only) — confirm the script is syntactically valid:
```bash
python -m py_compile scripts/verify_mapping.py && echo "py_compile OK"
```
and validate the skill itself:
```bash
<skill-builder-validator> \
  $SKILLS/moduscope-statedict-mapping/SKILL.md
```
Both must print success (`py_compile OK`; validator `✅ Validation passed`). Full runtime proof is the
`verify_mapping.py` invocation above, which a user runs on a node where REF_PATH/TARGET_PATH resolve.

## Common Mistakes
- **Confusing the H/D letters between test_14 and test_15.** They are swapped in comments; trust the
  shapes, not the letters. (E, I, D)=gate/up, (E, D, I)=down everywhere.
- **Putting the gpt_oss +1 / tp_degree fix-ups inside the map function.** They are kernel/runtime
  conventions applied to the live target module after loading, not part of the re-layout.
- **Adding the `mlp2_bias *= tp_degree` undo to test_15.** Only test_14 does it; test_15 must not.
- **Forgetting grads reuse the same map.** If you add a new expert family, the grad comparison breaks
  unless the map is a pure index re-layout (no value math).
