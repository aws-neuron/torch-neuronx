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

### Artifact Directory Structure

When `TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS=1`:

```
{TORCH_NEURONX_DEBUG_DIR}/
├── fx_graphs/
│   └── proc_{rank}/{model}_{timestamp}.fx.txt
├── stablehlo/
│   └── proc_{rank}/{model}_{timestamp}.stablehlo.mlir
├── raw/
│   └── proc_{rank}/{model}_{timestamp}.torch_raw.mlir
└── neff/
    └── proc_{rank}/{model}_{timestamp}.neff
```

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
