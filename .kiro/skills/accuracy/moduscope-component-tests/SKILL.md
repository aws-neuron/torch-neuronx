---
name: moduscope-component-tests
description: Run ModuScope BF16 module-level numerical-equivalence tests cross-cluster between the TRN (Neuron) cluster and the GPU cluster clusters using the gpu_save → S3 → neuron workflow. Use when running module tests for any torchtitan model (DeepSeek-V3 16B, Qwen3, Llama3, GptOss, etc.), debugging fwd/bwd grad failures, scaling test runs across all distribution configs and N_PER_DIST values, or adapting the framework to a new model. Critical knowledge — production runs MUST use the Neuron venv/private-torch-neuronx venv on TRN (older an incompatible venv has a complex64 .conj() bug); SDPA bf16 accumulator drift is the dominant failure mode for MLA/attention tests; bf16-baseline NaN cases need the NaN-safe ratio rule.
---

# ModuScope Component-Level Equivalence Tests

## Overview

Run BF16 module-level numerical-equivalence tests for any registered torchtitan model — cross-cluster between GPU (CUDA fp32+bf16 reference) and TRN (Neuron bf16 target). The framework asserts:

```
ratio = ||target - fp32||_F / ||ref_bf16 - fp32||_F < 1.1   (fwd output, each weight grad, input grad)
```

A passing ratio means Neuron's bf16 drift from fp32 is within ~10 % of CUDA's bf16 drift — i.e. Neuron and CUDA produce bf16 outputs of comparable precision relative to the fp32 reference.

## Usage

- Running any of `test_00..test_13_*.py` from `bf16/tests/` for any registered model (deepseekv3_16b/671b, qwen3_*, llama3_*, gpt_oss_*)
- Validating numerical equivalence between TRN (Neuron eager) and GPU (CUDA) for a model component
- Debugging large fwd/bwd ratios in any of the above tests
- Running the full distribution sweep (`COMPONENT_TEST_N_PER_DIST=5` → 35 configs across 7 distribution families) to characterize tolerance behaviour
- Adapting the test framework to a new model (or new optimizer-side branch) — see "Scaling to a new model" below

## Quick orientation — what each test covers

| Test | Module-under-test | DeepSeek-V3 applicable? | Configs at N_PER_DIST=5 |
|---|---|---|---|
| `test_00_rms_norm.py` | `RMSNorm` | ✅ | 35 |
| `test_01_embedding.py` | token embedding | ✅ | 35 |
| `test_02..07_*.py` | linear layers, ColwiseParallel/RowwiseParallel | varies | 35 each |
| `test_04_feedforward.py` | plain `FeedForward` (SwiGLU MLP) | ⚠️ skips for MoE-only models | 35 |
| `test_08_router.py` | `TokenChoiceTopKRouter` (MoE gate) | ✅ | 35 |
| `test_09_grouped_experts.py` | `GroupedExperts` (the N expert MLPs) | ✅ | 35 × 3 token-dist = **105** |
| `test_09_grouped_experts_noGMM.py` | same w/o grouped_mm | ✅ | 105 |
| `test_10_token_reorderer.py` | `TokenReorderer` | ✅ | 35 |
| `test_11_moe.py` | full MoE = router + reorderer + experts + shared_experts | ✅ — superset of 08+09+10 | 35 |
| `test_12_mla_attention.py` | `Attention` (MLA — multi-head latent attn for DeepSeek-V3) | ✅ | 35 |
| `test_13_deepseek_v3_transformer_block.py` | full block (MLA + MoE) | ✅ | 35 |

**For MLP-equivalent testing of an MoE model** — `test_11_moe.py` IS the "MLP block" test (the experts ARE the MLPs). `test_09_grouped_experts.py` is a finer-grained subset that exercises just the experts with deterministic token assignment.

## Core Concepts

### Two-cluster workflow

| Cluster | Where | Role |
|---|---|---|
| **the TRN (Neuron) cluster** (`<trn-node>]`) | $LOGIN_HOST bastion | runs target on Neuron with `TEST_MODE=neuron` |
| **the GPU cluster** (`<gpu-node>]`) | ec2-108-137-91-71 | runs reference on CUDA with `TEST_MODE=gpu_save` → uploads to S3 |
| **S3** | `s3://<bucket>/titan/modules/moe/{model}/{module}/{dist}/` | artifact transfer |

`TEST_MODE` env var picks workflow:
- `gpu_save` — GPU computes fp32+bf16 references, saves to S3, plus runs target locally
- `neuron` — TRN downloads S3 refs, runs target on Neuron, compares
- `gpu` — local-only (both ref and target on same GPU, no S3)

## Production venv (CRITICAL)

| Venv | torch_neuronx | Status |
|---|---|---|
| `$SHARED/<neuron-env>/private-torch-neuronx/.venv` | **2.11.3.0.1288+4b683e9e.dev** | ✅ production |
| `$SHARED/<incompatible-venv>/torch-neuronx/.venv` | 0.1.0 (very old) | ❌ has complex64 `.conj()` no-op bug |

If you ever see ratios >5× on `wq.weight`, `wkv_a.weight`, `attention_norm.weight`, or `input` — first thing to check is the venv. The old venv treats `complex.conj()` as identity → autograd silently uses wrong gradient through complex RoPE → 50×–90× error on weights downstream of RoPE.

## Standard end-to-end workflow

### Step 0 — Verify free nodes before launching

```bash
# TRN (from $LOGIN_HOST)
KEY=$SSH_KEY
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16; do
  N=<trn-node>
  busy=$(ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=4 $N "ps -fu $USER | grep -v grep | grep python.*pytest | wc -l" 2>/dev/null)
  echo "  $N: $USER-pytest procs=$busy"
done

# GPU (from gpu-login)
/opt/slurm/bin/squeue                  # see slurm jobs
/opt/slurm/bin/sinfo -p compute1 -t idle,alloc -o "%n %T"
```

When picking nodes for parallel runs: pick from the IDLE list and double-check that no other ubuntu/$USER user is on them.

### Step 1 — GPU side gpu_save (writes references to S3)

```bash
ssh -i ~/cgk-p5en.pem ubuntu@ec2-108-137-91-71.ap-southeast-3.compute.amazonaws.com

NODE=<gpu-node>   # pick a free idle node
TS=$(date +%Y%m%d_%H%M%S)
LOG=$SHARED/e2e_root/pytest_gpu_${TS}.log

# Drop a clean pytest.ini on /shared (compute nodes can't see /tmp from login)
cat > $SHARED/pytest_clean.ini <<'EOF'
[pytest]
addopts = -v -s --tb=short
testpaths = .
EOF

nohup /opt/slurm/bin/srun -N 1 -w $NODE -p compute1 --no-kill bash -c '
  source $SHARED/aws_creds.sh                                  # rotate STS creds first
  source $SHARED/gpu_baselines_venv/bin/activate
  export REF_PATH=$SHARED/NTT_gpu_baselines
  export TARGET_PATH=$SHARED/NTT_gpu_baselines
  export PYTHONPATH=$SHARED/e2e_root/ModuScope/test/component_tests_torchtitan/shared:$SHARED/e2e_root/NeuronTrainingEvaluations/src:$PYTHONPATH
  export TEST_MODE=gpu_save REF_DEVICE=cuda TARGET_DEVICE=cuda
  export MODEL_CONFIG=deepseekv3_16b           # ← change this for new models
  export COMPONENT_TEST_N_PER_DIST=5           # ← 1 = fast iter; 5 = full sweep (35 configs)
  cd $SHARED/e2e_root/ModuScope/test/component_tests_torchtitan/bf16/tests
  python -m pytest test_12_mla_attention.py test_13_deepseek_v3_transformer_block.py \
    -c $SHARED/pytest_clean.ini -rs
' > $LOG 2>&1 &
```

GPU-side "FAILED" in `gpu_save` mode = **NaN-poisoned baselines**, not a Neuron correctness signal. (REF_PATH = TARGET_PATH on GPU → both runs identical CUDA code; failures mean fp32 or bf16 reference itself produced NaN/inf for an edge-case dist.) The S3 artifacts ARE saved before the assert, so TRN-side runs are unaffected — they'll just propagate NaN through the comparison and report the same dists as fail-NaN.

### Step 2 — TRN side neuron compare

```bash
# from $LOGIN_HOST
KEY=$SSH_KEY
NODE=<trn-node>        # free node from your reservation

cat > $SHARED/pytest_clean.ini <<'EOF'
[pytest]
addopts = -v -s --tb=short
testpaths = .
EOF

ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=no $NODE "setsid nohup bash -c '
  source $SHARED/aws_creds.sh
  source $SHARED/<neuron-env>/private-torch-neuronx/.venv/bin/activate
  unset TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS
  export TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=0
  export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1                                      # avoids broken pytest_metadata jenkins import
  export REF_PATH=$SHARED/<neuron-env>/torchtitan
  export TARGET_PATH=$SHARED/deepseek_v3_optimized/torchtitan   # ← what we are validating
  export PYTHONPATH=$SHARED/<neuron-env>/ModuScope/test/component_tests_torchtitan/shared:$SHARED/<neuron-env>/NeuronTrainingEvaluations/src:\$PYTHONPATH
  export TEST_MODE=neuron TARGET_DEVICE=neuron
  export MODEL_CONFIG=deepseekv3_16b
  export COMPONENT_TEST_N_PER_DIST=5
  export NEURON_RT_VISIBLE_CORES=0
  cd $SHARED/<neuron-env>/ModuScope/test/component_tests_torchtitan/bf16/tests
  python -m pytest test_12_mla_attention.py test_13_deepseek_v3_transformer_block.py \
    -c $SHARED/pytest_clean.ini -rs
' > $SHARED/mla_diag_logs/pytest_trn_\$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &"
```

First-run wall time: test_12 ~4 min for 35 configs (compile cache hot). test_13 ~30 min. test_09 ~70 min for 105 configs (3× more configs × MoE compile size).

### Step 3 — Run multiple tests in parallel (when GPU/TRN have multiple free nodes)

To run test_09 + test_11 simultaneously, just pick **two different nodes** on each cluster. Each `srun` call binds to one node — no contention. Example launch script in `references/parallel-launch-template.sh`.

## Distribution sweep parameters

The framework's `DistConfig` generator is in `shared/distribution_config.py`:

```python
generate_test_configs(seed=42, edge_rate=0.3, n_per_dist=N_PER_DIST)
# → for each of 7 distribution families (normal, uniform, kaiming_normal,
#   kaiming_uniform, xavier_normal, xavier_uniform, log_normal):
#     for i in range(N_PER_DIST):
#         random_params(edge_case=random()<edge_rate)  # 30% edge cases
```

`NormalDistribution.random_params(edge_case=True)` picks `mean ∈ {0, ±1e-6, ±1e3}, std ∈ {1e-6, 1e-4, 100, 1000}` — extreme stress tests. **Production training never sees these magnitudes**; they're stress probes for the bf16 numerics.

To list exact params for any dist:
```bash
COMPONENT_TEST_N_PER_DIST=5 python -c '
import sys; sys.path.insert(0, "shared")
from distribution_config import generate_test_configs
for c in generate_test_configs(seed=42, edge_rate=0.3):
    print(c.name, c.weight_params, c.input_params)
'
```

(Helper: `$SHARED/dump_dist_params.py`.)

## NaN-safe ratio rule

Some dists produce NaN in the fp32 reference (overflow during edge-case computations). The naive ratio is `0/0 = NaN` and the test reports a meaningless failure. The NaN-safe rule:

```python
num   = ||target - fp32||_F
denom = ||ref_bf16 - fp32||_F

if NaN in num or denom:
    cleaned_num = ||nan_to_num(target - fp32)||_F   # zero out NaN positions
    PASS-NaN if cleaned_num < 1e-5 else FAIL-NaN
elif denom < 1e-30:
    PASS-zero if num < 1e-5 else FAIL-zero
else:
    ratio = num / denom
    PASS if ratio < 1.1 else FAIL
```

PASS-NaN means: on every position that was NOT NaN, target matches ref_bf16 bit-for-bit. The test is correctly identifying that Neuron isn't introducing extra error beyond the GPU's own NaN baseline.

Implementation in `$SHARED/mla_e2e_diag.py` (helper `report_compare`). The `conftest_base.py` framework version can be patched the same way (see `references/nan-safe-ratio.md`).

## Architectural failure modes

After running, group failures by which sub-op the failing weight/output traces back to:

| Failure pattern | Likely cause | Reference |
|---|---|---|
| All bwd grads on weights downstream of complex RoPE fail with **50×–90×** ratio | Old `<incompatible-venv>` venv (complex64 conj bug) | switch venv |
| Fwd output ratio ~2× on dists with tiny activations (`x.std=1e-6`) | Neuron `F.scaled_dot_product_attention` bf16 accumulator (CUDA `math` kernel uses fp32 accum) | `references/sdpa-bf16-finding.md` |
| Bwd `wq.weight` grad ratio ~1.1–1.6 on assorted xavier/kaiming dists | SDPA backward (same accumulator issue, bwd side) | `references/sdpa-bf16-finding.md` |
| Random PASS-NaN on uniform_0 / log_normal_0 / normal_4 / uniform_2 | Edge-case dist values caused fp32 overflow → cleaned ratio is 0 | NaN-safe rule above |
| Single dist fails, all weights ratio ~1.0 except one specific weight @ 1.5–2× | Real bug — instrument that weight's compute path with `_dump_and_hook` | `references/instrumentation-pattern.md` |

## Scaling to a new model

To extend this framework to a model other than `deepseekv3_16b`:

### 1. Confirm the model is registered

```bash
grep -rln "deepseekv3_16b\|qwen3_8b\|llama3_8b" $SHARED/<neuron-env>/ModuScope/test/component_tests_torchtitan/bf16/tests/conftest.py
```

`conftest.py:_BUILD_SHAPE_CONFIGS` builds shape configs for known model names. If your model isn't there, you need to:
- Add a `MoEShapeConfig` (or analogous `ShapeConfig` for non-MoE) entry in `conftest.py`
- Make sure `MODEL_CONFIG=<your_model_name>` resolves to a valid torchtitan registered model in both `REF_PATH` and `TARGET_PATH` torchtitan trees

### 2. Confirm MoE vs non-MoE branches

For a non-MoE model (Llama3, Qwen3 dense), `test_04_feedforward.py` is the relevant MLP test (skip 09/10/11). For an MoE model (DeepSeek-V3, GptOss, Qwen3 MoE), test_04 will skip and you want test_09/10/11.

### 3. Add the right tests to your sweep

| Model type | MLP-side tests | Attention test |
|---|---|---|
| Dense (Llama3, Qwen3 dense) | `test_04_feedforward.py` | `test_12_mla_attention.py` (only if MLA — else other test_12_*) |
| MoE (DeepSeek-V3, GptOss, Qwen3 MoE) | `test_09 + test_10 + test_11` (or just test_11 = superset) | `test_12_mla_attention.py` (only DeepSeek-V3) |

### 4. Run gpu_save first, then neuron (Steps 1-2 above)

S3 prefix `s3://<bucket>/titan/modules/moe/<MODEL_CONFIG>/<module>/<dist>/` is keyed by `MODEL_CONFIG`, so different models have isolated artifacts.

### 5. Compare ratios — anything systematic above 1.1 deserves localization

Use the framework's PASS-NaN rule for known overflow-prone dists. Real Neuron-vs-CUDA architectural drifts will show up as a consistent pattern across multiple dists touching the same op (SDPA, RoPE, RMSNorm, etc.).

### 6. Document new findings in references/

If you find a new architectural difference, add a `references/<finding-name>.md` describing:
- The minimal repro (e.g. `sdpa_repro.py`)
- Which dists trigger it
- The numerical ratio it produces
- Whether it affects production training (loss-curve evidence)

## Helper scripts

All in `$SHARED/` — see `references/helper-scripts.md` for full reference. Most-used:

| Script | Purpose |
|---|---|
| `mla_e2e_diag.py` | pytest-free reimpl of test_12 with NaN-safe ratio. Loads framework S3 artifacts. `--mode {gpu_save, neuron}` |
| `dist_input_stats.py` | Print min/max/mean/std/absmax/norm of q/k/v/output per dist for an MLA module |
| `dist_compare_nan_safe.py` | Standalone post-hoc comparator with NaN-safe rule |
| `sdpa_repro.py` | 25-line minimal SDPA repro (proves Neuron bf16 vs CUDA fp32 accumulator) |
| `dump_dist_params.py` | Print exact `weight_params` / `input_params` for any DistConfig name |
| `pytest_clean.ini` | Drop-in pytest config that overrides setup.cfg |

## Common mistakes

### Wrong venv (50×–90× bwd ratios)
- **Symptom**: Massive ratios on weights downstream of complex RoPE bwd
- **Fix**: `source $SHARED/<neuron-env>/private-torch-neuronx/.venv/bin/activate`

### setup.cfg pollutes pytest addopts
- **Symptom**: `unrecognized arguments: --pytorch_version=2.9.*`, or all tests skipped due to `--ignore=test/component_tests_torchtitan` and `-m sample_test`
- **Fix**: Always pass `-c $SHARED/pytest_clean.ini`

### TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1 blocks complex multiply
- **Symptom**: `RuntimeError: CPU fallback disabled because TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1`
- **Fix**: `unset` and `export ...=0` in the same shell as the python invocation

### Cross-cluster ssh-from-script doesn't propagate env
- **Symptom**: Run works interactively but fails when launched via wrapper
- **Fix**: ssh wrapper uses `bash -c '...'` (non-interactive subshell, no .bashrc). Set required env vars **inside** the bash -c block

### Empty `<dist>/` on S3 → "No artifacts found"
- **Symptom**: TRN-side `pytest -k normal_0` works but `kaiming_normal_0` skips
- **Fix**: Run gpu_save with `COMPONENT_TEST_N_PER_DIST=5` (no `-k` filter) to generate ALL dists before TRN side. Confirm with `aws s3 ls s3://<bucket>/titan/modules/moe/<model>/<module>/`

### Slurm srun queues forever (Resources)
- **Symptom**: srun job stays in `PD (Resources)` state
- **Fix**: A free node may have non-$USER ubuntu jobs. `squeue` to identify the holder; pick a different free node from `sinfo -t idle`

### NaN baseline on edge-case dist reported as FAIL
- **Symptom**: Test fails with "Error ratio = nan" or `inf` for `normal_4` / `uniform_2` / `log_normal_*`
- **Fix**: Use the NaN-safe rule. The dist's fp32 reference itself overflowed; this isn't a Neuron issue. PASS-NaN means cleaned numerator is 0, which is what we want

### "compile takes forever and times out" (test_13, test_11)
- **Expected**: First-time compile of MoE kernels via neuron-cc is ~10–15 min. Subsequent compiles use cache and are fast
- **Don't kill** before you see `[BACKWARD COMPARE]` lines

## Files & References

| Path | Contents |
|---|---|
| `references/cluster-setup.md` | Cluster topology, ssh keys, slurm reservations, paths |
| `references/instrumentation-pattern.md` | `_dump_and_hook` for fwd-norms + bwd-grad capture |
| `references/test12-sweep-analysis.md` | Concrete results from a 35-dist sweep on test_12 (5 fail / 30 pass) with exact ratios + dist params |
| `references/sdpa-bf16-finding.md` | The Neuron SDPA bf16 accumulator architectural finding + minimal repro |
| `references/nan-safe-ratio.md` | The NaN-safe rule + how to patch conftest_base.py |
| `references/helper-scripts.md` | Index of `$SHARED/*.py` helpers with usage examples |
| `references/parallel-launch-template.sh` | Bash template for parallel multi-test, multi-node launches |

## Sharing this skill

This skill lives at `$SKILLS/moduscope-component-tests/` — visible only to the local user. To share with others:

| Channel | How |
|---|---|
| **AIM Skills marketplace** (recommended) | `aim publish --skill moduscope-component-tests` (requires AIM agent setup); makes it pullable by other agents |
| **Internal package** | package it in your build system and declare the dependency; others depend on it |
| **Manual copy / GitHub** | Manual share — others copy directory to their `~/skills/` |

Until published, only this account uses it.
