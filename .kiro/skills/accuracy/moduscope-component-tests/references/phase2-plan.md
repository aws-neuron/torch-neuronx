# Phase 2 Plan — Fix test_05 / test_06 + Complete TRN Compares

**Started**: 2026-05-26 04:14 UTC
**Goal**: Make test_05_attention.py and test_06_transformer_block.py use S3 (gpu_save → S3 → neuron pattern), then run cross-cluster compares for all 4 MoE models.

## Status overview

| Test | Pattern | Models needing rework |
|---|---|---|
| test_12_mla_attention | ✅ already uses S3 | — (DeepSeekV3 done) |
| test_12_gptoss_attention | ✅ already uses S3 (custom) | — (GPT-OSS GPU done, TRN done with high failures) |
| test_09_grouped_experts | ✅ already uses S3 | — (DeepSeekV3 done; Llama4/Qwen3/GPT-OSS TRN running) |
| test_11_moe | ✅ already uses S3 | — (all 4 done) |
| test_13_deepseek_v3_transformer_block | ✅ uses S3 | — (DeepSeekV3 GPU done, TRN running) |
| **test_05_attention** | ❌ NO S3 — local compare only | Llama4, Qwen3 |
| **test_06_transformer_block** | ❌ NO S3 — local compare only | Llama4, Qwen3, GPT-OSS |

## Phase 2 work plan

### Step 1: Code changes (one-time, ~30 min)
- [x] Identify test_05 and test_06 don't have TEST_MODE branching
- [ ] Modify `test_05_attention.py`: add `gpu_save` (save ref outputs/grads to S3) and `neuron` (load from S3, run target on neuron, compare) modes
- [ ] Modify `test_06_transformer_block.py`: same TEST_MODE branching
- [ ] Fix `rope_cache.to(REF_DEVICE)` bug for both
- [ ] Sync to TRN + GPU clusters

### Step 2: GPU baselines (5 jobs, parallel × 3 nodes = 2 rounds, ~1h total)
**Round 1 (parallel on 1-2, 1-3, 1-5)**:
- [ ] llama4_17bx16e test_05_attention (gpu_save)
- [ ] qwen3_30b_a3b test_05_attention (gpu_save)
- [ ] llama4_17bx16e test_06_transformer_block (gpu_save)

**Round 2 (parallel on 2 nodes)**:
- [ ] qwen3_30b_a3b test_06_transformer_block (gpu_save)
- [ ] gpt_oss_20b test_06_transformer_block (gpu_save)

### Step 3: TRN compares (5 jobs, parallel × 4 nodes, pipelined)
- [ ] llama4 test_05 (after GPU done)
- [ ] qwen3 test_05 (after GPU done)
- [ ] llama4 test_06 (after GPU done)
- [ ] qwen3 test_06 (after GPU done)
- [ ] gpt_oss test_06 (after GPU done)

## Resource pool

| Cluster | Nodes |
|---|---|
| GPU | 1-2, 1-3, 1-5 |
| TRN | 3-5, 3-6, 3-7, 3-8 |

Currently TRN nodes are running other compares (test_09 × 3, test_13). They'll free up over the next 30-90 min.

## Status tracking

Status updates appended to `$SKILLS/moduscope-component-tests/references/phase2-status.jsonl` every checkpoint.

Format:
```json
{"ts":"2026-05-26T04:14:00Z","phase":"step1","action":"start","detail":"..."}
```

## Tracking file structure

| File | Purpose |
|---|---|
| `phase2-plan.md` | THIS file (plan + checklist) |
| `phase2-status.jsonl` | Append-only event log (every 5 min + at task transitions) |
| `phase2-results.md` | Final aggregate when complete |
