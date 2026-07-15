---
name: titan-accuracy
description: Run end-to-end TorchTitan cross-platform accuracy validation (GPU H200 vs TRN2 Neuron). Use when asked to validate numerical accuracy for a model, run accuracy comparison, compare GPU vs Neuron training, or scale accuracy tests to new models.
---

# TorchTitan Accuracy Validation

## Overview
End-to-end numerical accuracy validation of LLM training on GPU (H200) vs TRN2 (Neuron) using TorchTitan. Produces per-step FrobRatio, C_ratio, CosSim, loss, grad norm, and optimizer state comparisons.

## Usage
Use this skill when:
- Asked to run accuracy validation for a TorchTitan model
- Asked to compare GPU vs Neuron/TRN2 training numerics
- Scaling accuracy tests to new models (Qwen3, GPT-OSS, etc.)
- Asked to produce accuracy analysis MD files

## Core Concepts

### Metrics
- **FrobRatio** = `||trn - fp32|| / ||gpu_bf16 - fp32||` — PASS if ≤ 1.1
- **C_ratio** = `||trn - gpu_bf16|| / ||gpu_delta - gpu_bf16||` — PASS if < 1.0 (delta uses ε=2⁻⁷ on embedding)
- **CosSim** = cosine similarity between GPU and TRN tensors
- **Grad norm** = full model L2 gradient norm (all-reduced across TP/FSDP)
- **Opt state** = per-module post_weight, exp_avg (momentum), exp_avg_sq (variance)

### Infrastructure
- **Code:** `$SHARED/torchtitan-accuracy/` (identical on both clusters)
- **Branch:** `mainline_ootb` + `force_local_contiguous` fix + `cos_sin` RoPE with Llama scaling
- **Wrappers:** `accuracy_wrapper.py` (static dataset + opt state hook), `accuracy_perturb_wrapper.py` (+ ε perturbation)
- **Comparison:** `compare_4way_v2.py` on GPU node

### Clusters
- **GPU:** P5en H200, SSH via `ssh -i ~/cgk-p5en.pem ubuntu@ec2-108-137-91-71.ap-southeast-3.compute.amazonaws.com`, then `ssh <gpu-node>`
- **TRN:** the TRN (Neuron) cluster2, SSH via `ssh -i ~/mel_trn2.pem -o ProxyCommand=none $USER@$LOGIN_HOST`, then `ssh -i $SSH_KEY <gpu-node>`
- **S3:** `s3://<bucket>/$USER/`

## Workflow (Per Model)

### Step 1: Setup Config
Add `<model>_tensor_extract` config to the model's `config_registry.py`:
```python
def <model>_tensor_extract() -> Trainer.Config:
    config = <model_base_config>()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            TensorExtractorConverter.Config(
                output_folder="./tensor_dumps",
                depth="layer",
                module_filter="layers.<LAST_LAYER>",  # e.g. layers.31 for 32-layer model
                enable_activations=True,
                enable_gradients=True,
                enable_optimizer_state=True,
                max_disk_gb=500.0,
                time_steps=1,
                use_input_hash=True,
            ),
        ],
    )
    config.training = TrainingConfig(local_batch_size=1, seq_len=8192, steps=10, max_norm=1.0)
    config.metrics = MetricsProcessor.Config(log_freq=1)
    config.parallelism = ParallelismConfig(tensor_parallel_degree=4, data_parallel_shard_degree=-1)
    config.activation_checkpoint = ActivationCheckpointConfig(mode="selective", selective_ac_option="op")
    config.checkpoint = CheckpointManager.Config(enable=True, load_only=True)
    return config
```

You **MUST** sync config to both clusters after changes.

### Step 2: Generate Static Dataset + Seed Checkpoint
On GPU compute node (NOT head node):
```bash
# Dataset (160 samples for 10 steps × GBS=16)
python3 -c "import torch; torch.manual_seed(42); ids=torch.randint(0, VOCAB_SIZE, (160, 8192)); torch.save({'input_ids': ids, 'labels': ids.clone()}, 'static_dataset.pt')"

# Seed checkpoint (single GPU, TP1 DP1)
torchrun --nnodes 1 --nproc_per_node 1 --rdzv_id 100 --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
  torchtitan/train.py --module MODEL --config MODEL_tensor_extract \
  --training.steps=0 --dataloader.dataset=c4_test --hf_assets_path=./assets/hf/MODEL_ASSETS \
  --parallelism.tensor_parallel_degree=1 --parallelism.data_parallel_shard_degree=1 \
  --checkpoint.enable --checkpoint.create_seed_checkpoint --checkpoint.export_dtype=bfloat16 \
  --debug.seed=42 --debug.deterministic
```

Push checkpoint + dataset to S3, pull to the TRN cluster.

### Step 3: Run 4 Experiments
You **MUST** always `pkill -9 python; pkill -9 torchrun; sleep 2` before each run.
You **MUST** always `tee` output to a log file.
You **MUST** run on compute nodes, never head nodes.

**GPU bf16:**
```bash
torchrun --nnodes 1 --nproc_per_node 8 --rdzv_id N --rdzv_backend c10d --rdzv_endpoint localhost:PORT \
  accuracy_wrapper.py --module MODEL --config MODEL_tensor_extract \
  --dataloader.dataset_path=./static_dataset.pt --hf_assets_path=./assets/hf/MODEL_ASSETS \
  --training.global_batch_size=16 --checkpoint.load_only \
  --debug.seed=42 --debug.deterministic --metrics.log_freq=1
```

**GPU delta:** Same but use `accuracy_perturb_wrapper.py`

**GPU fp32:** Same as bf16 but add `--training.mixed_precision_param=float32 --training.mixed_precision_reduce=float32`

**TRN bf16:**
```bash
export NEURON_RT_NUM_CORES=64
export PYTHONPATH=$SHARED/torchtitan-accuracy:$PYTHONPATH
torchrun --nnodes 1 --nproc_per_node 64 ... accuracy_wrapper.py [same args] --checkpoint.load_only
```

You **MUST** install RT debs on TRN compute node: `sudo apt install -y --allow-downgrades $SHARED/artifacts-3-16/*.deb`
You **MUST** fix /tmp: `sudo rm -rf /tmp; sudo mkdir -p /tmp && sudo chmod 1777 /tmp`
You **MUST** use `PYTHONPATH` to override torchtitan import on the TRN cluster.

### Step 4: Transfer TRN Data to GPU
Only transfer first-TP-rank activation outputs per hash (160 files) + tp_0 opt state via S3.
You **MUST** use fresh S3 creds (ask user if expired).

### Step 5: Compare
Run on GPU compute node (all data pulled there):

**Activations (4-way):**
```bash
python3 compare_4way_v2.py  # reads from acc_v2_gpu_{bf16,fp32,delta}_opt and acc_v2_trn_bf16
```

**Opt state (4-way FrobRatio + C_ratio):**
```bash
python3 compare_opt_4way.py GPU_BF16_DIR TRN_DIR
```
You **MUST** exclude norm layers (attention_norm, ffn_norm) from FrobRatio — near-zero bf16↔fp32 denominator makes ratio unstable. C_ratio is valid for all modules.

**Loss + grad norm:** Extract from logs (strip ANSI: `sed "s/\x1B\[[0-9;]*m//g"`)

**Opt state (2-way CosSim + RelErr):**
```bash
python3 compare_opt_v2.py GPU_BF16_DIR TRN_DIR
```

### Step 6: Produce Report
Create `<model>_accuracy_analysis.md` with:
- Test configuration table
- 4-way activation comparison table (per step)
- Loss comparison table (all 4 runs)
- Grad norm comparison table
- Optimizer state comparison table
- Bugs found (if any)
- Conclusion with pass/fail verdict

Update `titan_accuracy_results_v2.md` summary table.

## Quick Reference

### File Inventory (all in torchtitan-accuracy/)
| File | Purpose |
|------|---------|
| `accuracy_patches.py` | Static dataset + opt state hook + scalar metrics (import only, no run) |
| `accuracy_wrapper.py` | Imports patches + runs training |
| `accuracy_perturb_wrapper.py` | Imports patches + adds ε=2⁻⁷ perturbation + runs |
| `compare_4way_v2.py` | 4-way activation comparison (FrobRatio + C_ratio + CosSim) |
| `compare_opt_4way.py` | 4-way opt state comparison (post_weight, exp_avg, exp_avg_sq) |
| `compare_opt_v2.py` | 2-way opt state comparison (CosSim + RelErr) |
| `torchtitan/components/tensor_extractor.py` | Enhanced extractor with per-hash, module_filter, opt state |
| `torchtitan/components/scalar_metrics_dumper.py` | Per-step scalar metrics to JSON |

### Local Source
All scripts live at `$WORKSPACE/workplace/torch-neuronx/src/torchtitan/`.

You **MUST** always edit locally and rsync to both clusters:
```bash
cd $WORKSPACE/workplace/torch-neuronx/src/torchtitan
rsync -az --force \
  --exclude="*.pyc" --exclude="__pycache__/" --exclude=".venv/" --exclude="build/" \
  --exclude="*.egg-info/" --exclude="outputs/" --exclude="tensor_dumps/" \
  --exclude=".git/" --exclude=".kiro/" --exclude="static_dataset.pt" \
  ./ -e "ssh -i ~/cgk-p5en.pem" \
  ubuntu@ec2-108-137-91-71.ap-southeast-3.compute.amazonaws.com:$SHARED/torchtitan-accuracy/

rsync -az --force \
  --exclude="*.pyc" --exclude="__pycache__/" --exclude=".venv/" --exclude="build/" \
  --exclude="*.egg-info/" --exclude="outputs/" --exclude="tensor_dumps/" \
  --exclude=".git/" --exclude=".kiro/" --exclude="static_dataset.pt" \
  ./ -e "ssh -i ~/mel_trn2.pem -o ProxyCommand=none" \
  $USER@$LOGIN_HOST:$SHARED/torchtitan-accuracy/
```

### Cluster Details

| Item | GPU | TRN |
|------|-----|-----|
| Venv | `$SHARED/moduscope_venv/` | `$SHARED/titan-llama/torch-neuronx/.venv/` + PYTHONPATH |
| nproc | 8 | 64 |
| FSDP | 2 | 16 |
| Grad accum | 8 | None |
| RT debs | N/A | `$SHARED/artifacts-3-16/*.deb` |
| HF assets | `./assets/hf/` (symlink) | `./assets/hf/` (symlink) |

## Common Mistakes

### Forgetting to kill stale processes
You **MUST** `pkill -9 python; pkill -9 torchrun` and on TRN also `sudo kill -9 $(sudo lsof -t /dev/neuron*)` before every run.

### DCP checkpoint OOM on TRN
TRN OOMs when saving DCP checkpoints. Use `--checkpoint.load_only` on TRN. Use our `accuracy_patches.py` hook for opt state instead.

### Comparing wrong TP ranks
GPU and TRN have different TP rank numbering per hash. Match by `tp_rank % 4 == 0` (first TP rank of each group).

### FSDP shard size mismatch for opt state
GPU FSDP2 rank 0 holds 1/2 of params, TRN FSDP16 rank 0 holds 1/16. Slice GPU tensor to match: `gpu_tensor[:trn_tensor.shape[0]]`.

### S3 token expiry
Tokens expire after ~12 hours. Ask user for fresh creds when needed.

### Overwriting checkpoints
Each run saves to same `outputs/checkpoint/` dir. Move data after each run before starting next.
