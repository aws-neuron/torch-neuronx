# Component Tests — Cross-cluster GPU↔Neuron equivalence

Module-level numerical equivalence tests for torchtitan models across the TRN (Neuron) cluster and the GPU cluster (CUDA).

## Quick setup — required env vars

The tests are entirely **env-var driven**. There's no config file.

### Common to both GPU and TRN

| Env var | Default | Purpose |
|---|---|---|
| `MODEL_CONFIG` | `qwen3_8b` | Which model to test (e.g. `deepseekv3_16b`, `llama4_17bx16e`, `qwen3_30b_a3b`, `gpt_oss_20b`) |
| `COMPONENT_TEST_N_PER_DIST` | 5 | How many dist configs per family (5 → 35 total configs across 7 families) |
| `RUN_BWD` | `1` | Set to `0` to skip backward pass |
| `S3_BASE` | `s3://<bucket>/titan/modules/moe` | Override S3 location for goldens |
| `ARTIFACT_CACHE` | `/tmp/moe_test_artifacts` | Local cache for downloaded artifacts |
| `BWD_TOLERANCE_RATIO` | `1.1` | Pass threshold for backward grad ratios |
| `QK_NORM_BWD_TOLERANCE_RATIO` | `1.2` | Looser threshold for q_norm/k_norm grads |
| `ZERO_BASELINE_ABS_THRESHOLD` | `0.01` | Threshold when baseline_fro=0 (zero-denom case) |

### GPU side (computes references, uploads to S3)

```bash
export TEST_MODE=gpu_save                                              # save baselines to S3
export REF_DEVICE=cuda                                                 # compute ref on GPU
export TARGET_DEVICE=cuda                                              # ignored in gpu_save mode
export REF_PATH=$SHARED/NTT_gpu_baselines                     # GPU torchtitan repo
export TARGET_PATH=$SHARED/NTT_gpu_baselines                  # same on GPU side
export MODEL_CONFIG=qwen3_30b_a3b
export COMPONENT_TEST_N_PER_DIST=5

# AWS creds for S3 upload
source $SHARED/aws_creds.sh   # or your own STS export
```

### TRN side (downloads refs, runs target on Neuron, compares)

```bash
export TEST_MODE=neuron                                                # load from S3
export TARGET_DEVICE=neuron                                            # run target on Neuron
export REF_PATH=$SHARED/<neuron-env>/torchtitan         # mainline_ootb branch
export TARGET_PATH=$SHARED/deepseek_v3_optimized/torchtitan  # optimized branch
export MODEL_CONFIG=qwen3_30b_a3b
export COMPONENT_TEST_N_PER_DIST=5
export NEURON_RT_VISIBLE_CORES=0

unset TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS
export TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=0             # allow CPU fallback
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1                                # avoid broken jenkins plugin

# AWS creds for S3 download
source $SHARED/aws_creds.sh
```

### Test mode summary

| `TEST_MODE` | What it does |
|---|---|
| `gpu` (default) | Build ref + target on REF_DEVICE / TARGET_DEVICE, compare locally. No S3. |
| `gpu_save` | Build ref on REF_DEVICE, **upload baselines to S3, NO compare**. |
| `neuron` | Download S3 baselines, build target on Neuron, compare ratios. |

## Test files

### Standard tests (preserved, S3-free, local-only compare)
- `test_00_*.py` … `test_05_attention.py` (GQAttention)
- `test_06_transformer_block.py` (generic block)
- `test_09_grouped_experts.py`, `test_10_token_reorderer.py`, `test_11_moe.py`
- `test_12_mla_attention.py` (DeepSeek-V3 MLA)
- `test_13_deepseek_v3_transformer_block.py` (DeepSeek-V3 block)

### S3-backed tests (NEW, cross-cluster GPU→Neuron compare)
- `test_05_attention_s3.py` — GQAttention with `TEST_MODE` branching, S3 baselines, backend-aware BlockMask
- `test_06_transformer_block_s3.py` — generic block, same S3 pattern
- `test_12_gptoss_attention.py` — custom test for GPT-OSS Attention (sliding window, sinks, custom inner_attention)

## Workflow

```
GPU side (gpu_save) → S3 → TRN side (neuron) → ratio compare
```

Run `gpu_save` mode on GPU first to populate baselines, then `neuron` mode on TRN to compare.

## Run commands

### GPU side (computes fp32+bf16 references, uploads to S3)

```bash
ssh -i ~/cgk-p5en.pem ubuntu@ec2-108-137-91-71.ap-southeast-3.compute.amazonaws.com

NODE=<gpu-node>          # any free node
TS=$(date +%Y%m%d_%H%M%S)
LOG=$SHARED/e2e_root/pytest_gpu_${TS}.log

# Drop pytest config (overrides ModuScope's setup.cfg --ignore)
cat > $SHARED/pytest_clean.ini <<EOF
[pytest]
addopts = -v -s --tb=short
testpaths = .
EOF

nohup /opt/slurm/bin/srun -N 1 -w $NODE -p compute1 --no-kill bash -c "
  source $SHARED/aws_creds.sh
  source $SHARED/gpu_baselines_venv/bin/activate
  export REF_PATH=$SHARED/NTT_gpu_baselines TARGET_PATH=$SHARED/NTT_gpu_baselines
  export PYTHONPATH=$SHARED/e2e_root/ModuScope/test/component_tests_torchtitan/shared:$SHARED/e2e_root/NeuronTrainingEvaluations/src:\$PYTHONPATH
  export TEST_MODE=gpu_save REF_DEVICE=cuda TARGET_DEVICE=cuda
  export MODEL_CONFIG=qwen3_30b_a3b               # any registered model
  export COMPONENT_TEST_N_PER_DIST=5              # 5 = full sweep, 1 = smoke

  cd $SHARED/e2e_root/ModuScope/test/component_tests_torchtitan/bf16/tests

  # Run any of:
  python -m pytest test_05_attention_s3.py        -c $SHARED/pytest_clean.ini -rs
  python -m pytest test_06_transformer_block_s3.py -c $SHARED/pytest_clean.ini -rs
  python -m pytest test_09_grouped_experts.py     -c $SHARED/pytest_clean.ini -rs
  python -m pytest test_11_moe.py                 -c $SHARED/pytest_clean.ini -rs
  python -m pytest test_12_mla_attention.py       -c $SHARED/pytest_clean.ini -rs
  python -m pytest test_12_gptoss_attention.py    -c $SHARED/pytest_clean.ini -rs
  python -m pytest test_13_deepseek_v3_transformer_block.py -c $SHARED/pytest_clean.ini -rs
" > $LOG 2>&1 &
```

### TRN side (downloads S3 refs, runs target on Neuron, compares)

```bash
# Login bastion → compute node
ssh $LOGIN_HOST
KEY=$SSH_KEY
NODE=<trn-node>              # any free node in your reservation

ssh -i $KEY $NODE "setsid nohup bash -c '
  source $SHARED/aws_creds.sh
  source $SHARED/<neuron-env>/private-torch-neuronx/.venv/bin/activate
  unset TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS
  export TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=0
  export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

  export REF_PATH=$SHARED/<neuron-env>/torchtitan
  export TARGET_PATH=$SHARED/deepseek_v3_optimized/torchtitan
  export PYTHONPATH=$SHARED/<neuron-env>/ModuScope/test/component_tests_torchtitan/shared:$SHARED/<neuron-env>/NeuronTrainingEvaluations/src:\$PYTHONPATH

  export TEST_MODE=neuron TARGET_DEVICE=neuron
  export MODEL_CONFIG=qwen3_30b_a3b
  export COMPONENT_TEST_N_PER_DIST=5
  export NEURON_RT_VISIBLE_CORES=0

  cd $SHARED/<neuron-env>/ModuScope/test/component_tests_torchtitan/bf16/tests

  python -m pytest test_05_attention_s3.py -c $SHARED/pytest_clean.ini -rs
' > $SHARED/mla_diag_logs/pytest_trn_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &"
```

## Test → Model mapping (what to run for each cell of the matrix)

For each (model, test type) pair, run the test file in the corresponding cell with `MODEL_CONFIG=<model>`.

| Model | **Attention** | **MoE** | **Transformer Block** | **Experts (Grouped)** |
|---|---|---|---|---|
| `deepseekv3_16b` | `test_12_mla_attention.py` | `test_11_moe.py` | `test_13_deepseek_v3_transformer_block.py` | `test_09_grouped_experts.py` |
| `llama4_17bx16e` | `test_05_attention_s3.py` | `test_11_moe.py` | `test_06_transformer_block_s3.py` | `test_09_grouped_experts.py` |
| `qwen3_30b_a3b` | `test_05_attention_s3.py` | `test_11_moe.py` | `test_06_transformer_block_s3.py` | `test_09_grouped_experts.py` |
| `gpt_oss_20b` | `test_12_gptoss_attention.py` | `test_11_moe.py` | `test_06_gptoss.py` | `test_09_grouped_experts.py` |

**Notes per model**:
- `deepseekv3_16b` — custom MLA + DeepSeek-specific block tests (`test_12_mla` + `test_13`).
- `llama4_17bx16e` — uses GQAttention `attn_backend="flex"` + complex RoPE; the `_s3` tests handle BlockMask + rope-on-device.
- `qwen3_30b_a3b` — uses GQAttention `attn_backend="sdpa"` + cos_sin RoPE; the `_s3` tests pass `None` mask for sdpa path.
- `gpt_oss_20b` — custom `Attention` class (sliding window + sinks); requires custom `test_12_gptoss_attention.py` and `test_06_gptoss.py` (generic test_05/06 are incompatible).

**⚠️ Verification status of new tests (as-of merge):**

| Test | Status |
|---|---|
| `test_05_attention_s3.py` | ✅ Verified — ran for llama4 + qwen3, produced TRN compare results |
| `test_06_transformer_block_s3.py` | ✅ Verified — ran for llama4 + qwen3, produced TRN compare results |
| `test_12_gptoss_attention.py` | ✅ Verified — ran for gpt_oss (8 pass / 27 fail on TRN) |
| `test_06_gptoss.py` | ❌ **NOT YET TESTED** — code written and synced, but never executed. Smoke-test before relying on results |

`test_09_grouped_experts.py` and `test_11_moe.py` work for all four MoE models out of the box (they use the model config registry).

## Supported models

| Model | Attention test | Block test | Notes |
|---|---|---|---|
| `deepseekv3_16b` | `test_12_mla_attention.py` (MLA) | `test_13_deepseek_v3_transformer_block.py` | Custom MLA + custom block |
| `llama4_17bx16e` | `test_05_attention_s3.py` (GQA, flex backend → BlockMask) | `test_06_transformer_block_s3.py` | Complex RoPE — needs `_to(REF_DEVICE)` fix |
| `qwen3_30b_a3b` | `test_05_attention_s3.py` (GQA, sdpa backend) | `test_06_transformer_block_s3.py` | cos_sin RoPE |
| `gpt_oss_20b` | `test_12_gptoss_attention.py` (custom) | `test_06_gptoss.py` (custom) | Custom Attention class with sliding window + sinks |

All models support `test_09_grouped_experts.py` and `test_11_moe.py`.

## S3 baselines (where GPU goldens live)

GPU `gpu_save` mode writes baselines to S3 at:

```
s3://<bucket>/titan/modules/moe/{model_name}/{module}/{test_id}/
```

**Tests print the S3 path explicitly during runs**:
- `gpu_save` mode logs `[S3 UPLOAD] Saved baselines to <s3_path>` after every dist
- `neuron` mode logs `[S3 LOAD] Loaded baselines from <s3_path>` before every compare

This means anyone reading the test log can immediately see which S3 prefix to point a follow-up TRN run at.

### Discover what's already on S3 (skip GPU-side rerun if baselines exist)

```bash
source $SHARED/aws_creds.sh

# What models have any baselines?
aws s3 ls s3://<bucket>/titan/modules/moe/

# What modules are saved for a model?
aws s3 ls s3://<bucket>/titan/modules/moe/qwen3_30b_a3b/

# What dists are saved for a (model, module)?
aws s3 ls s3://<bucket>/titan/modules/moe/qwen3_30b_a3b/attention/

# What's in a specific test_id directory?
aws s3 ls s3://<bucket>/titan/modules/moe/qwen3_30b_a3b/attention/normal_0/
```

If goldens already exist for your `(model, test, dist)`, you can skip GPU side entirely and run TRN compare directly with `TEST_MODE=neuron`.

### Path schema fields
- `model_name` = `MODEL_CONFIG` env var (e.g. `qwen3_30b_a3b`)
- `module` = test category, derived from each test file:
  - `test_05_attention_s3.py` → `attention/`
  - `test_06_transformer_block_s3.py` → `transformer_block/`
  - `test_06_gptoss.py` → `gptoss_transformer_block/`
  - `test_09_grouped_experts.py` → `experts/`
  - `test_11_moe.py` → `moe/`
  - `test_12_mla_attention.py` → `mla_attention/`
  - `test_12_gptoss_attention.py` → `gptoss_attention/`
  - `test_13_deepseek_v3_transformer_block.py` → `transformer_block/` (deepseek-specific)
- `test_id` = `dist_cfg.name` (e.g. `normal_0`, `kaiming_uniform_2`) plus `_<token_dist>` for test_09 (e.g. `normal_0_uniform`, `normal_0_skewed`)

Each `test_id` directory contains:
```
config.json              # test metadata (model, dist params, etc.)
weights.pt               # generated saved_state used by both ref and target
input.pt                 # x_cpu input tensor
ref_fp32_output.pt       # GPU fp32 reference output
ref_bf16_output.pt       # GPU bf16 baseline output
ref_fp32_grads.pt        # dict of fp32 grads per weight + "input"
ref_bf16_grads.pt        # dict of bf16 grads per weight + "input"
```

### Example: list goldens for a specific model

```bash
source $SHARED/aws_creds.sh   # or your own STS creds

# What modules have baselines for qwen3_30b_a3b?
aws s3 ls s3://<bucket>/titan/modules/moe/qwen3_30b_a3b/

# What dist configs are saved for attention?
aws s3 ls s3://<bucket>/titan/modules/moe/qwen3_30b_a3b/attention/

# What's in a specific config?
aws s3 ls s3://<bucket>/titan/modules/moe/qwen3_30b_a3b/attention/normal_0/
```

### Override S3 base (for staging, dev, or alternate buckets)

```bash
export S3_BASE=s3://my-team-bucket/my-prefix      # default: s3://<bucket>/titan/modules/moe
```

### Local artifact cache

After download, artifacts are cached at:
```
${ARTIFACT_CACHE:-/tmp/moe_test_artifacts}/{s3_path_without_scheme}/
```

You can pre-populate this dir to avoid re-downloads.



`COMPONENT_TEST_N_PER_DIST=5` generates 35 configs across 7 distribution families:
- normal, uniform, kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform, log_normal

`test_09_grouped_experts.py` adds 3 token-distribution strategies (uniform/skewed/single_expert) → 105 configs total.

## Pass criteria (`check_3tensor_result`)

```
ratio = ||target - fp32||_F / ||baseline - fp32||_F
PASS if ratio < BWD_TOLERANCE_RATIO (default 1.1)
```

### NaN / zero-denom rules (built into framework)

- `[PASS-NaN]` — GPU baseline produced NaN (overflow on extreme weight init); reference invalid
- `[FAIL-targetNaN]` — Neuron produced NaN where GPU was finite (real bug)
- `[PASS-zero]` — both baseline and target match fp32 (e.g. expert with no token assignment)
- `[FAIL-zero]` — baseline_fro=0 but target_fro exceeds `ZERO_BASELINE_ABS_THRESHOLD` (default 0.01)

## Common gotchas

1. **ModuScope `setup.cfg` ignores component_tests_torchtitan** by default. Always pass `-c $SHARED/pytest_clean.ini`.
2. **TRN side requires `TEST_MODE=neuron`** for S3 tests. If unset, defaults break behavior.
3. **`TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1` blocks complex ops**. `unset` it or `=0`.
4. **Llama4 GQAttention uses `attn_backend="flex"`** which requires a real `BlockMask`. The `_s3` test is backend-aware.
5. **First Neuron compile takes 10-30 min** (MoE kernels). Subsequent runs cache.

## Reference docs

- `references/SKILL.md` — full skill
- `references/sdpa-bf16-finding.md` — Neuron SDPA accumulator analysis
- `references/4-models-FINAL-results.md` — phase 2 sweep results
- `references/deepseek_debug.md` — fail-by-fail analysis for DeepSeek-V3

## Production AWS credentials

```bash
source $SHARED/aws_creds.sh   # rotate-able STS, ~few hour expiry
```

When rotating, replace contents of file with new exports.
