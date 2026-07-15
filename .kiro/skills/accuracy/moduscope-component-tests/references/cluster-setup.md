# Cluster setup — TRN (the TRN cluster) and GPU (the GPU cluster)

Two physically separate clusters, each with its own filesystem (`$SHARED/` is NOT cross-cluster). Workflow uses S3 (`s3://<bucket>/...`) for artifact transfer.

## the TRN (Neuron) cluster cluster

### SSH topology

```
local laptop ──ssh──> $LOGIN_HOST (bastion)
                       └──ssh -i $KEY──> <trn-node>
```

Where:
- `$LOGIN_HOST` is configured in `$SSH_CONFIG` as `Host $LOGIN_HOST` → `$LOGIN_HOST` (user: $USER, key: `~/mel_trn2.pem`)
- `$KEY = $SSH_KEY` for ssh from $LOGIN_HOST → compute nodes
- Compute nodes: `<trn-node>]` are in your slurm reservation (`$USER_reservation`, ends 2026-05-26 16:00 UTC). 4-day rolling reservation.

### Slurm

`/opt/slurm/bin/{sinfo,squeue,srun,scontrol}` available on $LOGIN_HOST. Compute nodes show as `resv` (reserved) state — that's expected for nodes in your reservation, not "busy". Check actual occupancy with:

```bash
ssh -i $KEY <trn-node> "ps -fu $USER | grep python | grep -v grep | wc -l"
```

### Production venv

```
$SHARED/<neuron-env>/private-torch-neuronx/.venv
```
- torch 2.11.0+cpu
- torch_neuronx 2.11.3.0.1288+4b683e9e.dev
- complex64 ops correct (post `.conj()` no-op fix)

DO NOT USE `$SHARED/<incompatible-venv>/torch-neuronx/.venv` (torch_neuronx 0.1.0) — has the complex `.conj()` bug. Older launcher scripts (`launch_single_node.sh`) hardcode this path; if you copy from those, change it.

### TorchTitan branches

```
REF_PATH    = $SHARED/<neuron-env>/torchtitan          # mainline_ootb
TARGET_PATH = $SHARED/deepseek_v3_optimized/torchtitan  # mainline_optimized (validates this)
```

Other clones (clean-may18, <incompatible-venv>{1,2,3}, fresh-may-1) exist for historical reasons; they all have the same deepseek_v3 model.py except deepseek_v3_optimized which has NKI attention path additions.

### Test framework

```
$SHARED/<neuron-env>/ModuScope/test/component_tests_torchtitan/bf16/tests/
  conftest.py              # synced from fresh-may-1 (has MoEShapeConfig + deepseekv3_16b registry)
  test_12_mla_attention.py
  test_12_mla_v2.py
  test_12_mla_diag.py
  test_12_mla_diag_v3.py
  test_13_deepseek_v3_transformer_block.py
  test_00_rms_norm.py … test_07_output_head.py
```

`NeuronTrainingEvaluations/src/` for `compare_3tensors`:
```
$SHARED/<neuron-env>/NeuronTrainingEvaluations/src/
```

## the GPU cluster cluster

### SSH topology

```
local laptop ──ssh -i ~/cgk-p5en.pem──> ec2-108-137-91-71.ap-southeast-3.compute.amazonaws.com (login)
                                          └──srun──> <gpu-node>] (GPU compute)
```

User on this cluster is `ubuntu`. Key: `~/cgk-p5en.pem`.

### Slurm

`/opt/slurm/bin/sinfo` shows ~16 GPU compute nodes. No personal reservation; allocate via srun:
```bash
nohup /opt/slurm/bin/srun -N 1 -w <gpu-node> -p compute1 --no-kill bash -c '...' > log 2>&1 &
```

### Production venv

```
$SHARED/gpu_baselines_venv/
```
- Python 3.12.3, pytest 9.0.3 — works for gpu_save mode.

### TorchTitan branch

```
$SHARED/NTT_gpu_baselines/
```
Used as both REF_PATH and TARGET_PATH on GPU side (same code; gpu_save just uploads).

### Test framework (synced from the TRN cluster)

```
$SHARED/e2e_root/ModuScope/test/component_tests_torchtitan/bf16/tests/
$SHARED/e2e_root/NeuronTrainingEvaluations/src/
```

If files are out of sync between clusters, re-tar from the TRN cluster:
```bash
ssh $LOGIN_HOST 'cd $SHARED/<neuron-env> && tar czf /tmp/m.tgz ModuScope/test/component_tests_torchtitan NeuronTrainingEvaluations/src'
scp $LOGIN_HOST:/tmp/m.tgz /tmp/
scp /tmp/m.tgz ubuntu@<gpu-login>:$SHARED/m.tgz
ssh ubuntu@<gpu-login> 'cd $SHARED/e2e_root && tar xzf $SHARED/m.tgz'
```

## S3

```
s3://<bucket>/titan/modules/moe/{model_name}/{module}/{dist_name}/
   config.json              (metadata)
   weights.pt               (saved_state)
   input.pt                 (x_cpu)
   ref_fp32_output.pt       (gpu fp32 forward output)
   ref_bf16_output.pt       (gpu bf16 forward output)
   ref_fp32_grads.pt        (dict of fp32 grads)
   ref_bf16_grads.pt        (dict of bf16 grads)
```

Fresh STS creds expire after a few hours; rotate with `source $SHARED/aws_creds.sh` (file should contain `export AWS_ACCESS_KEY_ID=…` etc.). When you get fresh creds from the user, replace this file's contents.

Test default `S3_BASE = s3://<bucket>/titan/modules/moe`. Override per run with `export S3_BASE=...`.

## Helpers

Wrappers in `$SHARED/`:
- `run_mla_e2e_trn.sh` — launches `mla_e2e_diag.py --mode neuron` on a TRN node
- `run_block_e2e_trn.sh` — same for `block_e2e_diag.py`
- `run_sdpa_dump_trn.sh` — launches `mla_sdpa_dump.py` (single-call, with optional bwd)
- `run_mla_diag_trn.sh` — launches `mla_norm_diag.py` (forward-only norms)

All four were updated to source `$SHARED/<neuron-env>/private-torch-neuronx/.venv/bin/activate`. If you find one still pointing at `<incompatible-venv>`, update it (sed -i 's|<incompatible-venv>/torch-neuronx|<neuron-env>/private-torch-neuronx|g').

Each accepts `FIXED_TS=<timestamp>` env var to align with a GPU-side run dir.

## Ad-hoc runners (pytest-free)

For quick iteration without setting up the full pytest framework:
- `$SHARED/mla_e2e_diag.py` — pytest-free reimpl of test_12_mla_attention.py
- `$SHARED/block_e2e_diag.py` — pytest-free reimpl of test_13
- `$SHARED/mla_norm_diag.py` — forward-only, prints norms per stage
- `$SHARED/mla_sdpa_dump.py` — single MLA fwd+bwd, dumps SDPA in/out via patched model.py

These are useful when ModuScope test framework isn't installed (e.g., on a fresh GPU node) or you want a single config (one dist) without pytest overhead.
