# Environment Variables

This document provides an overview of the different environment variables available to configure `torch_neuronx` behavior.

## `neuron` backend for `torch.compile`

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TORCH_NEURONX_DEBUG_DIR` | str | `/tmp/neuron_backend_<uuid>` | Directory where compilation artifacts (FX graphs, StableHLO IR, NEFF files) are saved. |
| `TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS` | bool | `False` | When `1`, keeps compilation artifacts after execution. Useful for debugging via IR inspection. |
| `TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION` | bool | `False` | When `1`, disables CPU fallback on Neuron execution failure. Fails fast instead. |
| `TORCH_NEURONX_COMPLEX_DECOMPOSE` | bool | `False` | Enables framework decomposition of complex operations before lowering. Requires PyTorch 2.13 or newer. When disabled, verified native complex operations use compiler lowering and other operations follow the existing fallback policy. |
| `TORCH_NEURONX_RETAIN_DEVICE_MODE` | bool | `False` | When `1`, outputs retain original device placement instead of Neuron device. |
| `TORCH_NEURONX_DYNAMO_DISABLE_CPU_AUTOCOPY` | bool | `False` | When `1`, raises error if CPU tensors are passed as inputs instead of auto-copying to Neuron. |
| `TORCH_NEURONX_ENABLE_LAZY_ALLOC` | bool | `False` | When `1`, defers NRT tensor allocation to execution time; otherwise allocates eagerly. |
| `TORCH_NEURONX_COMPILE_ONLY` | bool | `False` | When `1`, compile to NEFF without device execution. Ops compile but never run; `nrt_init` failure is non-fatal. Also set by `torch_neuronx.compiler.compile_only()`. |
| `NEURON_PLATFORM_TARGET_OVERRIDE` | str | (query live instance) | Target platform (e.g. `trn2`, `trn3`) for cross-compilation on a device-less host. Also set via `CompileOptions(target=...)`. |
| `TORCH_NEURONX_ATEN_HIERARCHY` | bool | `False` | When `1`, appends aten op hierarchy metadata to the graph for hierarchy based micro-modules flow compilation. |

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

## Persistent NEFF / HLO Cache

Compiled NEFFs and lowered HLO entries are cached on disk to avoid recompilation across runs. The cache is content-addressed; entries from one run can be reused by another as long as the inputs (HLO bytes, compiler config, compiler version) are unchanged.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TORCH_NEURONX_NEFF_CACHE_DIR` | path | `/tmp/neff_cache` | Directory for persistent NEFF cache. Point at a shared mount (e.g. FSx) to reuse compiled NEFFs across runs and instances. |
| `TORCH_NEURONX_HLO_CACHE_DIR` | path | `/tmp/hlo_cache` | Directory for persistent HLO cache. |
| `TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR` | path | `/tmp/local_cache` | Local directory for filelocks and scratch. **Must stay on a local POSIX filesystem** (NFS/Lustre lock semantics are unreliable). |
| `TORCH_NEURONX_NEFF_DISABLE_CACHE` | bool | `False` | When `1`, disables the persistent NEFF cache entirely. Every compile runs fresh. |
| `TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT` | int | `1200` | Seconds to wait for the per-key compile lock before giving up. |
| `TORCH_NEURONX_NEFF_KEY_FROM_HLO` | bool | `False` | When `1`, the persistent NEFF key is derived from the lowered HLO bytes plus compiler config (instead of the in-memory base_key). Lets benign Python edits re-trace and re-lower without busting the on-disk cache. Off by default; opt in to enable. |
| `NKI_ENABLE_TRACE_CACHE` | bool | `True` | When `1`, the NKI kernel compile cache will be persisted across processes at `/var/tmp/nki-trace-cache-uid<uid>/` (or `NKI_TRACE_CACHE_URL` if set). Note that only files under a `site-packages`/`dist-packages` tree that map to an installed wheel get captured in the caching process. An editable install, a vendored copy, or any tree with no `.dist-info` does not resolve to a version and will not be tracked, potentially resulting in a stale cache hit. Additionally, the kernel is assumed to be a pure function of its declared arguments and its source. Any value that steers control flow / which code path is taken should be passed as an argument to the kernel, otherwise a change to that value might not be detected and could also result in a stale cache hit. |
| `TORCH_CACHING_PRECOMPILE` | bool | `False` | When `1`, enable the PyTorch Dynamo precompile cache, which caches the backend result of the dynamo trace of any torch.compile-ed modules or functions. These will be automatically saved at TORCHINDUCTOR_CACHE_DIR (which is by default /tmp/torchinductor_<username>/) split by subdirectories per rank if applicable. For the neuron backend the HLO result will be cached, separate from the HLO cache. The TORCHINDUCTOR_CACHE_DIR should be deleted to clear this cache. Off by default. |
| `TORCHINDUCTOR_CACHE_DIR` | path | `/tmp/torchinductor_<username>/` | Directory for the torch inductor cache. Note that this is a PyTorch variable, and affects the location of other caches as well. Please refer to the PyTorch documentation for details on other caches. |

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
| `NEURON_DYN_NKI_COLLECTIVES` | bool | `False` | When `1`, routes eligible collective calls through the dynamic NKI kernel implementation. |

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
