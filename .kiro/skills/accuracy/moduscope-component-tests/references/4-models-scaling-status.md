# MoE Models Scaling — Component-Level Equivalence Tests

**Goal**: Run module-level numerical equivalence tests for **4 MoE models × 4 tests × 2 hardware** between the TRN (Neuron) cluster and the GPU cluster (CUDA).

**Last updated**: 2026-05-26 02:13 UTC

---

## Master matrix — all 4 models × 4 tests × 2 hardware

Legend: ✅ done · 🔄 running · ⏳ pending

| Model | Stage | Attention | Grouped Experts | MoE Block | **Transformer Block** |
|---|---|---|---|---|---|
| **DeepSeek-V3 16B** | GPU baseline | ✅ captured | ✅ captured | ✅ captured | **⏳ pending — only normal_0 captured via diag script, NOT full sweep** |
| **DeepSeek-V3 16B** | TRN compare | ✅ **5 fail/30 pass** | ✅ **71 fail/34 pass** | ✅ **3 fail/32 pass** | **✅ only normal_0 PASS — fwd 1.000, all 11 grads 0.972–1.005** |
| **Llama4 17bx16e** | GPU baseline | ✅ captured | 🔄 GPU 1-5 (~99%) | ✅ captured | **⏳ never launched** |
| **Llama4 17bx16e** | TRN compare | ✅ **30 pass/5 xpass** | ⏳ pending | 🔄 TRN 3-6 | **⏳ never launched** |
| **Qwen3 30B-A3B** | GPU baseline | ✅ captured | 🔄 GPU 1-2 (~50%) | ✅ captured | **⏳ never launched** |
| **Qwen3 30B-A3B** | TRN compare | ✅ **30 pass/5 xpass** | ⏳ pending | ✅ **4 fail/31 pass** | **⏳ never launched** |
| **GPT-OSS 20B** | GPU baseline | ✅ captured | 🔄 GPU 1-3 (~50%) | ✅ captured | **⏳ never launched** |
| **GPT-OSS 20B** | TRN compare | 🔄 TRN 3-5 | ⏳ pending | 🔄 TRN 3-7 | **⏳ never launched** |

---

## Transformer block tests — detailed status

**Test name mapping**:
| Model | Transformer block test |
|---|---|
| DeepSeek-V3 16B | `test_13_deepseek_v3_transformer_block.py` (model-specific — uses `TEST_LAYER_ID=1` to ensure MoE branch active) |
| Llama4 17bx16e | `test_06_transformer_block.py` (generic — uses `model_cfg.block_class_name="Llama4TransformerBlock"` from registry) |
| Qwen3 30B-A3B | `test_06_transformer_block.py` (generic — `Qwen3TransformerBlock`) |
| GPT-OSS 20B | `test_06_transformer_block.py` (generic — `GptOssTransformerBlock`, layer_id=0 hits sliding attention) |

**What each transformer block test exercises (e2e at module level)**:
- Embedding-free input tensor `(1, seq_len, dim)`
- Full block forward = `RMSNorm → Attention → Residual → RMSNorm → MoE/MLP → Residual`
- Backward through all weights including all expert MLPs, router, attention projections, RoPE-side state
- Tests grad ratios for every weight in the block

**Status per model**:

| Model | GPU baseline | TRN compare | Notes |
|---|---|---|---|
| **DeepSeek-V3 16B** | ⏳ partial — only normal_0 captured via `block_e2e_diag.py` (pytest-free reproducer) | ✅ only normal_0 — PASS | Full 35-config sweep via test_13 has NOT been run. Need `pytest test_13_*` with `COMPONENT_TEST_N_PER_DIST=5` |
| **Llama4 17bx16e** | ⏳ NEVER LAUNCHED | ⏳ NEVER LAUNCHED | Need `pytest test_06_transformer_block.py` |
| **Qwen3 30B-A3B** | ⏳ NEVER LAUNCHED | ⏳ NEVER LAUNCHED | Need `pytest test_06_transformer_block.py` |
| **GPT-OSS 20B** | ⏳ NEVER LAUNCHED | ⏳ NEVER LAUNCHED | Note: gpt_oss test_06 may hit sliding-window code path at `layer_id=0` (`use_sliding_attention = layer_id % 2 == 0`) — may need same BlockMask handling we did in `test_12_gptoss_attention.py` |

**Why no transformer block sweeps yet**: Phase A focused on attention + MoE + grouped experts (the 3 priority tests for cross-cluster equivalence). Transformer block is a SUPERSET of these — if the leaf tests pass, block-level should also pass. Block test is most useful when there's a composition bug not caught by leaf tests.

**ETA if launched**: Each model's transformer block test takes ~15–30 min on GPU (compile + 35 configs) and ~30–60 min on TRN (Neuron compile of larger graph + 35 configs). For all 3 new models: ~1.5h GPU + ~3h TRN sequential, or much less in parallel.

---

## Currently running (with nodes)

| Hardware | Node | Job | Started |
|---|---|---|---|
| **GPU 1-2** | qwen3 test_09_grouped_experts | ~33 min ago |
| **GPU 1-3** | gpt_oss test_09_grouped_experts | ~33 min ago |
| **GPU 1-5** | llama4 test_09_grouped_experts | 2h52m ago (~99% done) |
| **TRN 3-5** | gpt_oss test_12_gptoss_attention | ~10 min ago |
| **TRN 3-6** | llama4 test_11_moe | ~38 min ago |
| **TRN 3-7** | gpt_oss test_11_moe | ~10 min ago |

## Free nodes right now

| Pool | Free | Note |
|---|---|---|
| GPU {1-2, 1-3, 1-5} | **0 of 3** | all running test_09 |
| TRN {3-5, 3-6, 3-7, 3-8} | **1 of 4** (3-8) | available |

---

## Cluster pool — what we can use

### GPU pool (verified by `nvidia-smi`)
- ✅ usable: 1-2, 1-3, 1-5 (3 nodes)
- ❌ sheaak holds 1-1 (GPUs idle but slurm allocation held)
- ❌ heavy other-user compute on 1-4, 1-6, 1-7, 1-8, 1-9, 1-10, 1-11, 1-12, 1-13, 1-14, 1-15, 1-16 (~55 GB × 8 GPUs each, 100% util — invisible to `ps -fu ubuntu`)

### TRN pool (verified by `ps`)
- ✅ usable: 3-5, 3-6, 3-7, 3-8 (4 nodes)
- ❌ MY OWN long-running training jobs on 3-1..3-4 (llama3_8b 4-node compile), 3-9..3-12 (llama3_70b compile), 3-13..3-16 (llama3_70b eager) — DO NOT TOUCH

---

## Aggregate counts (4 models × 4 tests = 16 cells per stage)

| Status | GPU baselines (16 total) | TRN compares (16 total) |
|---|---|---|
| ✅ Done | 9 | 7 (deepseekv3 ×3 + deepseekv3 test_13 normal_0 + llama4 test_05 + qwen3 test_05 + qwen3 test_11) |
| 🔄 Running | 3 (test_09 × 3 models) | 3 (gpt_oss test_12, llama4 test_11, gpt_oss test_11) |
| ⏳ Pending | **4** (test_06 × 3 + test_13 full sweep) | **6** (test_09 × 3 + test_06 × 3) |

---

## Pending work breakdown

### TRN compares pending — wait for GPU first
| Model | Test | Why pending |
|---|---|---|
| Llama4 | test_09_grouped_experts | waiting for GPU 1-5 |
| Qwen3 | test_09_grouped_experts | waiting for GPU 1-2 |
| GPT-OSS | test_09_grouped_experts | waiting for GPU 1-3 |

### GPU baselines NOT YET LAUNCHED
| Model | Test |
|---|---|
| DeepSeek-V3 | test_13_transformer_block (full 35-config sweep) |
| Llama4 | test_06_transformer_block |
| Qwen3 | test_06_transformer_block |
| GPT-OSS | test_06_transformer_block |

After GPU baselines for test_06/test_13 captured, corresponding TRN compares would also be pending.

---

## Marginality summary (from completed TRN runs)

### DeepSeek-V3 test_12_mla_attention (5 fails, all SDPA-touching)

| Dist | Where | Ratio | Marginal? |
|---|---|---|---|
| normal_0 | fwd output | 2.0002 | ❌ NOT marginal — input std=1e-6 edge case |
| normal_2 | fwd output | 1.1343 | ✅ marginal (3.4% over) |
| uniform_0 | fwd output | 1.1110 | ✅ marginal (1.0% over) |
| xavier_normal_0 | bwd wq.weight | 1.1604 | ✅ marginal (5.5% over) |
| xavier_uniform_2 | bwd wq.weight | 1.1097 | ✅ marginal (0.97% over) |

### DeepSeek-V3 test_11_moe (3 fails)

| Dist | Where | Ratio | Marginal? |
|---|---|---|---|
| normal_1 | bwd experts.w1 | 1.2475 | ⚠️ borderline (13% over) |
| normal_3 | bwd experts.w2 | 1.1220 | ✅ marginal (2.0% over) |
| log_normal_4 | bwd experts.w1 | 1.4894 | ❌ NOT marginal (35% over, log_normal weight loc=-10 edge case) |

### DeepSeek-V3 test_09_grouped_experts (71 fails / 105 configs)

Pending per-fail analysis. Likely majority NaN-baseline propagation (similar pattern to test_12) — need NaN-safe rule applied to log to classify each fail.

### Qwen3 30B-A3B test_11_moe (4 fails)

Recently completed — fail details and ratios not yet inspected.

### DeepSeek-V3 test_13_transformer_block (only normal_0 — PASS)
| Stage | Ratio |
|---|---|
| fwd output | 1.000 |
| 11 weight grads | range 0.972–1.005 |
| input grad | 1.000 |

---

## Code changes (only 3, no torchtitan model code touched)

1. **`conftest_base.py`** (TRN + GPU): added stubs for missing symbols `QK_NORM_BWD_TOLERANCE_RATIO` and `maybe_compile` (no-op since module tests don't use compile).
2. **`bf16/tests/conftest.py`** (TRN + GPU): added `compile_mode` fixture returning `None`.
3. **NEW**: `bf16/tests/test_12_gptoss_attention.py` (~217 lines) — custom test for GPT-OSS Attention nn.Module. Uses real `BlockMask` constructed via `create_block_mask`, exercises actual production module forward (no monkey-patching).

## Process / node management actions taken

- **Only 2 kills** (with explicit user permission):
  - 1-8 PIDs 1041720+1041721 (qwen3 test_09, was on disallowed node) → relaunched on 1-2
  - 1-10 PIDs 9663+9664 (gpt_oss test_09, was on disallowed node) → relaunched on 1-3
- **Sheaak's job 2256** on 1-1 — 100% untouched
- **My llama3 8B/70B training jobs** on 3-1..3-4, 3-9..3-16 — 100% untouched

## Skills used

- **`$SKILLS/moduscope-component-tests/`**
  - `SKILL.md` — main workflow + scaling-to-new-models section
  - `references/sdpa-bf16-finding.md` — Neuron SDPA bf16 accumulator (root cause for deepseek attention fails)
  - `references/nan-safe-ratio.md` — NaN-safe ratio rule with code
  - `references/test12-sweep-analysis.md` — 35-config sweep details
  - `references/instrumentation-pattern.md` — `_dump_and_hook` instrumentation
  - `references/cluster-setup.md` — cluster topology + ssh keys
  - `references/parallel-launch-template.sh` — bash template for parallel multi-node launches
  - `references/4-models-scaling-status.md` — THIS FILE
