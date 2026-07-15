---
name: moduscope-trn-compare
description: Run the TRN (Neuron) side of a ModuScope BF16 component-equivalence test on a TRN cluster compute node — TEST_MODE=neuron downloads GPU goldens from S3, runs the target module on Neuron, and reports ratios; TEST_MODE=neuron_cpu builds the reference locally on CPU for a quick TRN smoke. Use when asked to "run on TRN/Neuron", "compare ratios on the TRN cluster", "run the neuron-side compare", "neuron_cpu smoke", or to launch test_13/14/15/16 (or test_00-08) on TRN nodes after GPU gpu_save goldens exist. Covers the ssh-from-$LOGIN_HOST background launch, the PRODUCTION private-torch-neuronx venv (NOT an incompatible venv), the env vars, and the "No artifacts found" skip.
---

# ModuScope TRN (Neuron) Compare

## Overview

Run the **TRN/Neuron side** of a ModuScope component-equivalence test on a TRN cluster compute
node. `TEST_MODE=neuron` downloads the GPU fp32+bf16 goldens from S3, builds the target
module on Neuron, and asserts `ratio = ||target-fp32|| / ||ref_bf16-fp32|| < tol`.
`TEST_MODE=neuron_cpu` skips S3 and builds the reference locally on CPU — a fast smoke
that one test/dist runs on Neuron at all. This skill never debugs numerics; it launches.

## Usage

Use this skill when:
- Asked to "run on TRN", "run the neuron compare", "run on the TRN cluster node N", or "neuron_cpu smoke".
- GPU `gpu_save` goldens already exist on S3 and you want the cross-cluster compare (`TEST_MODE=neuron`).
- You want a quick local-reference validation that a test compiles/runs on Neuron (`TEST_MODE=neuron_cpu`).
- Launching any MoE test (`test_13_attention`, `test_14_grouped_experts`, `test_15_moe`,
  `test_16_transformer_block`, `test_08_masked_topk_router`) or dense test (`test_00`–`test_07`) on Neuron.

This skill is the consumer of goldens produced by the `gpu_save` side. To CREATE goldens
on GPU, use the GPU-save side of `moduscope-component-tests` first.

## Core Concepts

### TEST_MODE → devices (from conftest.py)

| `TEST_MODE` | REF_DEVICE | TARGET_DEVICE | S3 | Use |
|---|---|---|---|---|
| `neuron` | cpu (loaded from S3 golden) | neuron | **download** | production cross-cluster compare |
| `neuron_cpu` | cpu (built locally) | neuron | none | quick TRN validation, local ref |

Both map to `(ref=cpu, target=neuron)`. The difference is the reference source: `neuron`
loads GPU goldens from S3; `neuron_cpu` recomputes the reference on CPU in-process.

### The assertion (every test)

```
ratio = ||target_bf16 - ref_fp32||_F / ||ref_bf16 - ref_fp32||_F     PASS if ratio < tol (default 1.1)
```
NaN-safe / zero-denom verdicts (`[PASS-NaN]`, `[FAIL-targetNaN]`, `[PASS-zero]`,
`[FAIL-zero]`, `[PASS-near-zero]`) are handled inside `check_3tensor_result`.

### Cluster facts (the TRN (Neuron) cluster)

- Path: laptop → `ssh $LOGIN_HOST` (bastion) → `ssh -i $KEY <trn-node>]`.
- `KEY=$SSH_KEY`.
- **PRODUCTION venv (USE THIS):** `$SHARED/<neuron-env>/private-torch-neuronx/.venv`
  (torch_neuronx 2.11.3.0.1288+…; complex64 conj fixed).
- **NEVER use** `$SHARED/<incompatible-venv>/torch-neuronx/.venv` (0.1.0 — complex64
  `.conj()` no-op bug → 50–90× bogus backward ratios). The script aborts if `VENV` matches `<incompatible-venv>`.
- `$SHARED/` is shared across the TRN cluster login+compute, but is **NOT** shared with the GPU
  cluster — S3 is the only bridge between GPU goldens and TRN.

### Environment the TRN side MUST set

```bash
source $SHARED/aws_creds.sh                       # neuron mode only (S3 download)
source $SHARED/<neuron-env>/private-torch-neuronx/.venv/bin/activate
unset TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS
export TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=0  # allow complex/unimpl CPU fallback
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1                     # avoid broken jenkins plugin import
export REF_PATH=$SHARED/<neuron-env>/torchtitan          # mainline / plain MoE
export TARGET_PATH=$SHARED/deepseek_v3_optimized/torchtitan  # NKI-optimized
export PYTHONPATH=$REPO/shared:$NTE/src:$PYTHONPATH
export TEST_MODE=neuron TARGET_DEVICE=neuron NEURON_RT_VISIBLE_CORES=0
export MODEL_CONFIG=deepseekv3_16b
export COMPONENT_TEST_N_PER_DIST=5                         # 5 = 35 cfgs; 1 = 7-cfg smoke
python -m pytest <tests> -c $SHARED/pytest_clean.ini -rs
```

Always pass `-c $SHARED/pytest_clean.ini` — ModuScope's `setup.cfg` otherwise
ignores this directory and the tests collect to zero.

## How to run — use the script

`scripts/run_trn_neuron.sh` builds a self-contained remote runner on `/shared`, then
launches it on the compute node with `setsid nohup … </dev/null &` so it survives the ssh
session. Run it **from $LOGIN_HOST**. Parameters are env vars; `NODE` is required.

```bash
# Production cross-cluster compare (S3 goldens must already exist from gpu_save):
NODE=<trn-node> \
MODEL_CONFIG=deepseekv3_16b \
TESTS="test_13_attention.py test_15_moe.py" \
N_PER_DIST=5 MODE=neuron \
  bash scripts/run_trn_neuron.sh

# Quick local-reference TRN smoke (no S3, one test, fewest configs):
NODE=<trn-node> \
MODEL_CONFIG=deepseekv3_16b \
TESTS="test_13_attention.py" \
N_PER_DIST=1 MODE=neuron_cpu \
  bash scripts/run_trn_neuron.sh
```

The script prints the log path; follow it with `tail -f <log>`. Overridable paths (`KEY`,
`VENV`, `REPO`, `REF_PATH`, `TARGET_PATH`, `NTE_SRC`, `PYTEST_INI`, `LOG_DIR`) have the TRN cluster
defaults baked in. See the header of `scripts/run_trn_neuron.sh` for the full list.

### The launch pattern it encodes

```bash
KEY=$SSH_KEY
ssh -i $KEY <node> "setsid nohup bash -c '<env+pytest>' > LOG 2>&1 </dev/null &"
```

## Critical gotchas

1. **First NKI compile is 10–30 min** (MoE/attention kernels via neuronx-cc). Subsequent
   configs hit the NEFF cache and are fast. **Do NOT kill the run** before you see the
   `[BACKWARD COMPARE]` / compare lines — an idle-looking process is almost always compiling.
2. **"No artifacts found" skip** (`neuron` mode). `load_ref_artifacts` calls
   `pytest.skip("No artifacts at <s3_path>")` (or "Artifact download failed") when the S3
   golden for a `(model, module, dist)` is missing. Fix: run the GPU `gpu_save` side
   **full** first (`COMPONENT_TEST_N_PER_DIST=5`, no `-k` filter) so all dists exist, then
   re-launch. Confirm with `aws s3 ls s3://<bucket>/titan/modules/moe/<model>/<module>/`.
   For a fast check that needs no S3, use `MODE=neuron_cpu` instead.
3. **Cross-cluster env not propagated.** The ssh wrapper runs a non-interactive
   `bash -c` subshell (no `.bashrc`). All env (venv activate, REF_PATH, TEST_MODE, …) MUST
   be set **inside** that block. The script does this by baking everything into the
   generated runner — never rely on login-shell env on the compute node.
4. **Wrong venv → 50–90× backward ratios.** Symptom on weights downstream of complex RoPE.
   Always the production `private-torch-neuronx` venv; the script refuses `<incompatible-venv>`.
5. **`TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1` blocks complex multiply.** Must
   `unset` then `export …=0` in the same shell as pytest.
6. **Attention SDPA drift (test_13).** `SDPA_FWD_TOLERANCE_RATIO` (default 3.0) covers
   Neuron SDPA bf16-accumulator drift; deepseek target uses `attn_backend="nki"`. Not a bug.

## How to test this skill

Structurally verify the launcher is sound (no live cluster run):

```bash
bash -n scripts/run_trn_neuron.sh && echo "SYNTAX_OK"
```

Success criterion: prints `SYNTAX_OK` (exit 0). For a real but minimal end-to-end check
when you have a TRN cluster node, run the documented `neuron_cpu` smoke (`N_PER_DIST=1`, one test)
— it needs no S3 goldens and exercises the full launch + Neuron-target path; success is a
`PASSED`/`SKIPPED` line and a `[run_trn_neuron] DONE` marker in the log.

## Reference

Full env-var table, S3 layout, and the GPU `gpu_save` producer side live in the sibling
skill `moduscope-component-tests` (`references/RUN-COMMANDS.md`). This skill is the TRN
consumer of those goldens.
