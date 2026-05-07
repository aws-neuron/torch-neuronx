# Environment Variables

This document provides an overview of the different environment variables available to configure `torch_neuronx` behavior.

## `neuron` backend for `torch.compile`

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TORCH_NEURONX_DEBUG_DIR` | str | `/tmp/neuron_backend_<uuid>` | Directory where compilation artifacts (FX graphs, StableHLO IR, NEFF files) are saved. |
| `TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS` | bool | `False` | When `1`, keeps compilation artifacts after execution. Useful for debugging via IR inspection. |
| `TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION` | bool | `False` | When `1`, disables CPU fallback on Neuron execution failure. Fails fast instead. |
| `TORCH_NEURONX_RETAIN_DEVICE_MODE` | bool | `False` | When `1`, outputs retain original device placement instead of Neuron device. |
| `TORCH_NEURONX_DYNAMO_DISABLE_CPU_AUTOCOPY` | bool | `False` | When `1`, raises error if CPU tensors are passed as inputs instead of auto-copying to Neuron. |
| `TORCH_NEURONX_ENABLE_LAZY_ALLOC` | bool | `False` | When `1`, defers NRT tensor allocation to execution time; otherwise allocates eagerly. |

### Artifact Directory Structure

Both `torch.compile` and eager execution use the same unified job-based artifact hierarchy.
The working directory used by all neuronx-cc calls is preserved under TORCH_NEURONX_DEBUG_DIR.

```
{TORCH_NEURONX_DEBUG_DIR}/
├── {job_id}/
│   ├── fx_graphs/{model}_{ts}.fx.txt
│   ├── transformed_fx/{model}_{ts}.transformed.fx.txt
│   ├── raw/{model}_{ts}.torch_raw.mlir
│   ├── stablehlo/{model}_{ts}.stablehlo.mlir
│   └── torch_mlir_error/{model}_{ts}.mlir  # Populated when torch-mlir fails
└── neuronx-cc/
    └── {hash}/
        ├── log-neuron-cc.txt
        ├── module.mlir (or module.hlo.pb)
        ├── module.neff
        ├── metadata.txt
        └── compiler_args.txt
```

When `TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS=1`, all directories and their contents are preserved for debugging and analysis.

## Compiler

See [Neuron Compiler CLI Reference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/compiler/neuronx-cc/api-reference-guide/neuron-compiler-cli-reference-guide.html) for available flags.

## General

These variables affect eager execution (non-compiled operations).

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `NEURON_LAUNCH_BLOCKING` | bool | `False` | When `1`, forces synchronous execution. Useful for debugging. |
| `TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS` | bool | `False` | When `1`, only falls back to CPU for unimplemented ops, not execution errors. |
| `TORCH_NEURONX_ENABLE_STACK_TRACE` | bool | `False` | When `1`, captures Python stack traces per operation. Adds overhead. |
| `TORCH_NEURONX_DISABLE_NEFF_CACHE` | bool | `False` | When `1`, disables NEFF caching. Forces recompilation every time. |
| `TORCH_NEURONX_LOG_LEVEL` | int | `0` | When `3`, enable debug logs. |
| `TORCH_NEURONX_USE_PINNED_MALLOC` | bool | `False` | (Experimental) When `1`, uses driver-pinned host memory. |

## Neuron Runtime
- See [Neuron Runtime Configurable Parameters](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/nrt-configurable-parameters.html) for all runtime variables.
- See [Logical NeuronCore Configuration](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/logical-neuroncore-config.html) for LNC details.

## Distributed

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `NEURON_USE_SPLIT_ALLGATHER` | bool | `True` | Enables split-based allgather for better memory efficiency. |
| `NEURON_USE_SPLIT_REDUCE_SCATTER` | bool | `True` | Enables split-based reduce_scatter implementation. |
| `COLLECTIVE_BUCKETSIZE_IN_MB` | int | `512` | Maximum bucket size (MB) for collective operations. |
| `COLLECTIVE_MIN_BUCKETS` | int | `0` | Minimum buckets for collectives. When > 1, forces splitting small tensors. |

## Metrics

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TORCH_NEURONX_ENABLED_METRIC_TABLES` | str | `""` | Comma-separated metric tables to enable for CSV export. |
| `TORCH_NEURONX_METRICS_DIR` | str | `/tmp` | Directory for metric CSV files. |
