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
        ├── module.mlir
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

## Metrics

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TORCH_NEURONX_ENABLED_METRIC_TABLES` | str | `""` | Comma-separated metric tables to enable for CSV export. |
| `TORCH_NEURONX_METRICS_DIR` | str | `/tmp` | Directory for metric CSV files. |

## Distributed

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `NEURON_USE_SPLIT_ALLGATHER` | bool | `True` | Enables split-based allgather for better memory efficiency. |
| `NEURON_USE_SPLIT_REDUCE_SCATTER` | bool | `True` | Enables split-based reduce_scatter implementation. |
| `COLLECTIVE_BUCKETSIZE_IN_MB` | int | `512` | Maximum bucket size (MB) for collective operations. |
| `COLLECTIVE_MIN_BUCKETS` | int | `0` | Minimum buckets for collectives. When > 1, forces splitting small tensors. |

## Neuron Memory Allocator Configuration

The Neuron caching allocator's segment-pool tuning knobs are exposed
through a single environment variable:

```
TORCH_NEURONX_SEGMENT_ALLOCATOR_CONFIG=key1=value1,key2=value2,...
```

The value is a comma-separated list of `key=value` pairs. The first call
to the allocator parses the variable, validates the result, and snapshots
it into the per-device `NeuronSegmentPool` at construction. The snapshot
is fixed for the pool's lifetime. To change tuning, set the variable and
restart the process.

**Format**

- **Sizes**: raw bytes as a non-negative decimal integer. No size
  suffixes; `1048576` is valid, `1MB` is rejected with a warning.
- **Plain integers** (only `kRoundupPower2Divisions` today): same syntax,
  bare digits.
- Unknown keys are ignored with a warning.
- Invalid values are ignored with a warning; the field keeps its default.

**Default values** are tuned from training-workload experiments. Override
if running into OOM resulting from fragmentation.

**WARNING**: changing values can have drastic effects on
memory allocation training workloads and can potentially induce OOMs.
| Key                              | Type  | Default (bytes)  | Definition |
|----------------------------------|-------|------------------|------------|
| `kPoolBoundary`                  | bytes | `1048576` (1 MB) | Allocations below this go to the small pool, at or above to the large pool. The two pools have separate free sets so small allocations cannot fragment large-pool segments and vice versa. |
| `kSmallSegmentSize`              | bytes | `2097152` (2 MB) | Segment size NRT-allocates for small-pool requests. |
| `kMediumSegmentSize`             | bytes | `67108864` (64 MB) | Segment size NRT-allocates for large-pool requests in `[kPoolBoundary, kMediumThreshold)`. |
| `kMediumThreshold`               | bytes | `16777216` (16 MB) | Above this, each request gets its own per-request segment sized to the request rounded up to `kSegmentAlignment`. |
| `kSegmentAlignment`              | bytes | `2097152` (2 MB) | Alignment for per-request segments above `kMediumThreshold`. Must be a power of 2. |
| `kMinRoundedSize`                | bytes | `256`            | Floor for `RoundSize`. Requests smaller than this round up to it. |
| `kRoundupPower2Divisions`        | int   | `4`              | `N` in the power-of-two-with-N-divisions block rounding scheme - i.e. number of equal-sized divisions between the lower and higher power of 2 value of the requested size. `N=1` is pure pow2; `N=4` gives 25% max waste; higher `N` reduces waste at the cost of more distinct buckets, lower cache hit rate, and greater overhead of splitting/merging free blocks. Valid range: `[1, 64]`. |
| `kSmallSplitThreshold`           | bytes | `kMinRoundedSize` | Small-pool minimum split remainder. If carving a chosen free block would leave a remainder smaller than this, the block is handed out whole instead. |
| `kLargeSplitThreshold`           | bytes | `kPoolBoundary`  | Large-pool minimum split remainder. Same idea as above for the large pool. |
| `kMaxSplitSizeBytes`             | bytes | `SIZE_MAX`       | Cap above which a free block cannot be split. Default unlimited (all blocks splittable). Setting a finite cap keeps large blocks intact for future large requests, at the risk of cache misses for small requests when only above-cap blocks are free. Pairs with `kMaxNonSplitRoundingBytes`. |
| `kMaxNonSplitRoundingBytes`      | bytes | `20971520` (20 MB) | When a free block is non-splittable (`block.size > kMaxSplitSizeBytes`), allow it to serve a request whose size is within this tolerance of the block's size. Increase to widen reuse at the cost of more internal fragmentation. |

**Examples**

Cap large-block splitting at 512 MB; allow up to 64 MB rounding waste on
non-split reuse:

```bash
TORCH_NEURONX_SEGMENT_ALLOCATOR_CONFIG="kMaxSplitSizeBytes=536870912,kMaxNonSplitRoundingBytes=67108864"
```

Reduce internal fragmentation in the medium-size range by widening
rounding divisions to 8:

```bash
TORCH_NEURONX_SEGMENT_ALLOCATOR_CONFIG="kRoundupPower2Divisions=8"
```
