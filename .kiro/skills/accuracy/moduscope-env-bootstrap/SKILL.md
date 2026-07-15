---
name: moduscope-env-bootstrap
description: Deterministically set up the shell environment on either ModuScope cluster (GPU the GPU cluster / CUDA, or TRN the TRN cluster / Neuron) so BF16 component-equivalence tests run correctly. Use when preparing a fresh login or compute node to run gpu_save or neuron mode tests, when tests fail with import errors (torchtitan/conftest_base/compare_3tensors not found), when pytest ignores the test dir, when Neuron backward ratios come back 50-90x (wrong venv), or before launching any GPU-to-S3-to-TRN cross-cluster run. Provides sourceable, parameterized bootstrap_gpu_env.sh / bootstrap_trn_env.sh plus verify_env.sh.
---

# ModuScope Env Bootstrap

## Overview

Set up a correct, reproducible shell environment on each ModuScope cluster so component
tests run without import errors, pytest-config surprises, or the wrong Neuron venv. One
sourceable script per cluster (GPU `gpu_save`, TRN `neuron`) plus a `verify_env.sh` that
fails loudly if anything is missing. The environment is 100% env-var driven — there is no
test config file, so getting these exports right is the whole job.

## Usage

Use this skill when:
- Preparing a fresh GPU login/compute node (`gpu_save`) or TRN compute node (`neuron`) to run tests.
- Tests fail importing `torchtitan`, `conftest_base`, or `compare_3tensors`.
- pytest collects nothing / ignores `component_tests_torchtitan` (ModuScope `setup.cfg` `--ignore`).
- Neuron backward ratios come back 50-90x (almost always the buggy `<incompatible-venv>` venv).
- About to launch a GPU→S3→TRN cross-cluster run and want the env locked down first.

## Core Concepts

### Two clusters, two scripts, one verifier

| | GPU (the GPU cluster, CUDA) | TRN (the TRN cluster, Neuron) |
|---|---|---|
| Script | `scripts/bootstrap_gpu_env.sh` | `scripts/bootstrap_trn_env.sh` |
| `TEST_MODE` | `gpu_save` (compute refs, upload to S3, no compare) | `neuron` (download refs, run target, compare) |
| Default `REPO_ROOT` | `$SHARED/e2e_root` | `$SHARED/<neuron-env>` |
| venv | `$SHARED/gpu_baselines_venv` | `$SHARED/<neuron-env>/private-torch-neuronx/.venv` |
| `REF_PATH` | `$SHARED/NTT_gpu_baselines` | `$SHARED/<neuron-env>/torchtitan` (mainline ootb) |
| `TARGET_PATH` | same as REF (gpu_save only uploads) | `$SHARED/deepseek_v3_optimized/torchtitan` (NKI optimized) |
| Devices | `REF_DEVICE=cuda TARGET_DEVICE=cuda` | `TARGET_DEVICE=neuron NEURON_RT_VISIBLE_CORES=0` |

`$SHARED/` is NOT cross-cluster — S3 is the bridge. Both scripts `source`
`$SHARED/aws_creds.sh` (rotatable STS, ~hours) and drop the clean pytest ini.

### What every cluster needs

1. The right **venv** activated.
2. **`PYTHONPATH`** = `<REPO_ROOT>/ModuScope/test/component_tests_torchtitan/shared` (for
   `conftest_base`) + `<REPO_ROOT>/NeuronTrainingEvaluations/src` (for `compare_3tensors`).
3. **`REF_PATH` / `TARGET_PATH`** pointing at the two torchtitan trees (added to `sys.path`
   by the test, not by `PYTHONPATH`).
4. The mode env vars (`TEST_MODE`, devices, `COMPONENT_TEST_N_PER_DIST`, …).
5. A **clean pytest ini** at `$SHARED/pytest_clean.ini`, because ModuScope's own
   `setup.cfg` `--ignore`s this directory. Always run pytest with `-c $PYTEST_CLEAN_INI`.

The ini content (also written by the scripts):
```ini
[pytest]
addopts = -v -s --tb=short
testpaths = .
```

### CRITICAL: TRN venv choice (the 50-90x trap)

`bootstrap_trn_env.sh` hard-defaults to the **production** venv
`$SHARED/<neuron-env>/private-torch-neuronx/.venv` (torch_neuronx 2.11.3.x).
NEVER use `$SHARED/<incompatible-venv>/torch-neuronx/.venv` (torch_neuronx 0.1.0):
it has a **complex64 `.conj()` bug** that corrupts RoPE/MLA complex math in the backward
pass, producing 50-90x backward grad ratios so every attention/block test fails. The
script warns if `VENV` is overridden to an `<incompatible-venv>` path.

## Quick Reference

GPU (run on an the GPU cluster node, inside the venv shell):
```bash
source scripts/bootstrap_gpu_env.sh [REPO_ROOT] [MODEL_CONFIG]   # e.g. ... qwen3_30b_a3b
bash   scripts/verify_env.sh gpu
cd "$MODUSCOPE_TEST_DIR"
python -m pytest test_15_moe.py -c "$PYTEST_CLEAN_INI" -rs
```

TRN (run on a TRN cluster compute node):
```bash
source scripts/bootstrap_trn_env.sh [REPO_ROOT] [MODEL_CONFIG]
bash   scripts/verify_env.sh trn
cd "$MODUSCOPE_TEST_DIR"
python -m pytest test_15_moe.py -c "$PYTEST_CLEAN_INI" -rs
```

Parameters (positional or pre-set env): `REPO_ROOT`, `MODEL_CONFIG`. Override anything else
via env before sourcing: `VENV`, `REF_PATH`, `TARGET_PATH`, `COMPONENT_TEST_N_PER_DIST`,
`PYTEST_INI`, `AWS_CREDS`.

### Exports set by the bootstrap scripts

| Var | GPU value | TRN value |
|---|---|---|
| `TEST_MODE` | `gpu_save` | `neuron` |
| `REF_DEVICE` / `TARGET_DEVICE` | `cuda` / `cuda` | (ref from S3) / `neuron` |
| `NEURON_RT_VISIBLE_CORES` | — | `0` |
| `TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS` | — | `0` (unset then re-export) |
| `PYTEST_DISABLE_PLUGIN_AUTOLOAD` | — | `1` |
| `COMPONENT_TEST_N_PER_DIST` | `5` (override) | `5` (override) |
| `MODUSCOPE_TEST_DIR` | `<REPO_ROOT>/.../bf16/tests` | same |
| `PYTEST_CLEAN_INI` | path to clean ini | same |

## How to test this skill

Structural verification (no live cluster, no AWS — safe anywhere):
```bash
# 1. All scripts are syntactically valid bash.
bash -n scripts/bootstrap_gpu_env.sh && \
bash -n scripts/bootstrap_trn_env.sh && \
bash -n scripts/verify_env.sh && echo "SYNTAX OK"

# 2. Sourcing sets the expected exports WITHOUT needing the real cluster.
#    Point everything at /tmp so no real venv/creds are touched; expect WARN lines, then
#    assert the key env vars landed.
( VENV=/tmp/none AWS_CREDS=/tmp/none PYTEST_INI=/tmp/pytest_clean.ini \
  source scripts/bootstrap_trn_env.sh /tmp/repo qwen3_30b_a3b
  test "$TEST_MODE" = neuron &&
  test "$TARGET_DEVICE" = neuron &&
  test "$TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS" = 0 &&
  test "$MODEL_CONFIG" = qwen3_30b_a3b &&
  test -f /tmp/pytest_clean.ini &&
  echo "TRN EXPORTS OK" )
```
Success criterion: prints `SYNTAX OK` and `TRN EXPORTS OK`. On a real node, the full proof
is `verify_env.sh` exiting `0` (all six import checks PASS) after sourcing the matching
bootstrap script.

## Common Mistakes

- **Skipping `chmod -R 777 /tmp` on a freshly-allocated node.** The Neuron compile cache
  (`/tmp/neuron_backend`, `/tmp/hlo_cache`) is often left owned by a PRIOR user, so the first NKI
  compile dies with `[Errno 13] Permission denied` / `cannot create directories`. **ALWAYS run
  `sudo chmod -R 777 /tmp` at the start of every node run** (bake it into the launcher). Environmental,
  not a test bug.
- **Executing instead of sourcing** the bootstrap scripts — exports vanish with the
  subshell. The scripts detect this and error out; use `source`.
- **Omitting `-c "$PYTEST_CLEAN_INI"`** — ModuScope's `setup.cfg` ignores the test dir and
  pytest collects nothing.
- **Wrong TRN venv** — `<incompatible-venv>` → complex64 conj bug → 50-90x ratios. Use the
  production `private-torch-neuronx/.venv` (the default).
- **Leaving `TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1`** — blocks complex ops on
  Neuron. The TRN script forces it to `0`.
- **`REF_PATH`/`TARGET_PATH` swapped on TRN** — ref MUST be mainline ootb, target the
  NKI-optimized tree, or you validate the wrong direction.

## Related skills
- `moduscope-trn-compare` / `moduscope-gpu-baselines` launch the tests; `moduscope-result-triage`
  interprets ratios; `moduscope-node-select` picks free nodes; `moduscope-test-roadblocks` recovers
  from runtime errors.
- **No-S3 single-node smoke (recommended first):** set `TEST_MODE=neuron_cpu` before sourcing the TRN
  script (CPU reference + Neuron target, skip `aws_creds`), or use `moduscope-trn-compare`'s
  `run_trn_neuron.sh MODE=neuron_cpu`.
- `moduscope-component-tests` is the LEGACY skill (older test numbering test_09/10/11/12); its
  `references/cluster-setup.md` / `RUN-COMMANDS.md` still have useful SSH topology.
