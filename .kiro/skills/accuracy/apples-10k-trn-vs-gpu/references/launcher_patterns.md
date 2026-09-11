# Launcher Cmdline Patterns

This reference details the actual cmdlines our launcher scripts emit, so you can debug or adapt them without reading the launcher source.

## Common cmdline (apples)
Both TRN and GPU runs share these flags. Any deviation breaks apples-to-apples.

```
--module modu_scope.data.titan_deterministic
--config <model_family>:<model_size>     # e.g. llama3:llama3_8b, qwen3:qwen3_32b
--training.steps=10000
--lr_scheduler.warmup_steps=10            # TIM-style — NOT torchtitan default 200
--dataloader.dataset=c4_local             # NOT c4_test (dataset bug, see SKILL.md)
--dataloader.global_batch_size=$GBS
--training.global_batch_size=$GBS
--training.local_batch_size=$LBS
--hf_assets_path=./tests/assets/tokenizer
--activation_checkpoint.mode=full
--metrics.log_freq=1
--debug.deterministic
--debug.seed=42
--debug.log_input_hash
--metrics.enable_tensorboard
--metrics.save_tb_folder=tb
--dump-folder=$DUMP
```

For compile, add: `--compile.enable --compile.backend=neuron` (TRN) or `--compile.enable` (GPU; backend default = inductor)

For seed checkpoint init: `--checkpoint.enable --checkpoint.load_only --checkpoint.initial_load_path=$SEED`

## Topology flags

### 1N TRN apples (GBS=16, LBS=1, grad_accum=1)
```
--parallelism.tensor_parallel_degree=4
--parallelism.data_parallel_shard_degree=16     # auto = 64/TP/dp_replicate = 16
--parallelism.data_parallel_replicate_degree=0  # 0 means "no replicate"
```
torchrun: `--nnodes=1 --nproc_per_node=64`

### 1N GPU apples (GBS=16, LBS=1, grad_accum=8)
```
--parallelism.tensor_parallel_degree=4
--parallelism.data_parallel_shard_degree=2      # 8 GPUs / 4 TP = 2 DP shards
```
torchrun: `--nnodes=1 --nproc_per_node=8`

### 4N TRN apples (GBS=64, LBS=1, grad_accum=1)
```
--parallelism.tensor_parallel_degree=4
--parallelism.data_parallel_shard_degree=16     # within-node FSDP
--parallelism.data_parallel_replicate_degree=4  # across-node replication
--optimizer.implementation=foreach              # required for replicate
```
torchrun: `--nnodes=4 --nproc_per_node=64`

### 4N GPU apples (GBS=64, LBS=1, grad_accum=8)
```
--parallelism.tensor_parallel_degree=4
--parallelism.data_parallel_shard_degree=2
--parallelism.data_parallel_replicate_degree=4
```
torchrun: `--nnodes=4 --nproc_per_node=8`

### 70B 4N TRN (GBS=64, LBS=1, grad_accum=1, no replicate)
```
--parallelism.tensor_parallel_degree=4
--parallelism.data_parallel_shard_degree=64     # full FSDP across all 256 procs
```
torchrun: `--nnodes=4 --nproc_per_node=64`

### 70B 8N GPU (GBS=64, LBS=4, grad_accum=1, no replicate)
```
--parallelism.tensor_parallel_degree=4
--parallelism.data_parallel_shard_degree=16
```
torchrun: `--nnodes=8 --nproc_per_node=8`

## TRN-specific Neuron environment
Set in TRN launcher before torchrun:
```
NEURON_LAUNCH_BLOCKING=0
NEURON_LOGICAL_NC_CONFIG=2
NEURON_RT_ALLOW_LEGACY_NEFF=1
NEURON_RT_DBG_ZEROCOPY=1
NEURON_RT_DISABLE_EXECUTION_BARRIER=0
NEURON_RT_MAP_HBM=1
NEURON_RT_NUM_CORES=64
NEURON_RT_VIRTUAL_CORE_SIZE=2
TORCH_NEURONX_CPP_PROCESS_GROUP=1
TORCH_NEURONX_DEBUG_DIR=$DUMP/xla_dump
TORCH_NEURONX_ENABLED_METRIC_TABLES=graph_stats
TORCH_NEURONX_ENABLE_CONCATENATION=0
# Preserve this baseline's single-stream, non-HostCC behavior.
TORCH_NEURONX_ENABLE_STREAMS=0
TORCH_NEURONX_ENABLE_HOST_CC=0
TORCH_NEURONX_ENABLE_LAZY_ALLOC=1
TORCH_NEURONX_ENABLE_STACK_TRACE=0
TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=0
TORCH_NEURONX_HLO_CACHE=1
TORCH_NEURONX_METRICS_DIR=$DUMP/graph_stats
TORCH_NEURONX_MLIR_ATEN_OPS=1
TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS=1
TORCH_NEURONX_RETAIN_DEVICE_MODE=0
```

## GPU-specific environment
```
# (intentionally minimal — let NCCL choose defaults)
# Avoid setting NCCL_NVLS_ENABLE=0 on H100/H200 — it disables intra-node NVLS
# fast path and costs ~3-5% per step. Inherited from older baseline launchers
# but not needed for our config.
```

## Pre-launch cleanup
TRN launcher runs (per node):
```
sudo pkill -9 -f torchrun || true
sudo pkill -9 -f titan_train_wrapper || true
sudo rm -rf /tmp/hlo_cache /tmp/neuron-compile-cache
sudo chmod 777 -R /tmp
sleep 3
```

GPU launcher runs:
```
pkill -9 -f torchrun || true
pkill -9 -f titan_train_wrapper || true
sleep 3
```

## Background detachment
All multi-node launches use this idiom on each node:
```
setsid nohup bash $SCRIPT $ARGS > $LOG.bgout.rank$i 2>&1 < /dev/null &
```
This survives SSH disconnect and detaches from the controlling terminal.

## Master/port discipline
Use a unique master port per launch to avoid collisions when relaunching after kill:
- 1N: no port needed (single process)
- 4N TRN: ports `2975x` (we used 29754, 29762, 29763, 29770 historically)
- 4N GPU: ports `2975x` (we used 29755)
- 8N GPU: ports `2975x` (we used 29764)
