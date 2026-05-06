# TorchNeuronX Statistics & Memory Usage Tracking

This document describes the statistics and memory tracking capabilities available in `torch_neuronx` for monitoring and debugging model execution on Neuron devices.

## **IMPORTANT: Synchronize before reading memory stats.**

> Neuron's lazy allocator issues `allocate()` and deallocation asynchronously: the
> Python thread only reserves a virtual address and submits a `kAlloc` op to the
> RuntimeWorker. The physical carve (`nrt_tensor_allocate`), the segment pool
> bookkeeping, and the stats counter updates all happen later, on the worker
> thread, when those ops execute.
>
> Stats returned by `memory_stats()`, `memory_allocated()`, `max_memory_allocated()`,
> etc. therefore reflect **worker-side physical state at the moment of the read**,
> not the set of outstanding `DataPtr`s on the Python side (which is how CUDA's
> equivalents behave). If Python runs ahead of the worker, the numbers you see
> may lag behind the most recent Python-side alloc/free calls.
>
> For an accurate point-in-time snapshot, call `torch.neuron.synchronize()` before reading.
> Similarly, call `synchronize()` before `reset_peak_memory_stats()` so pending ops land in
> the peak counter before it's cleared.

## Overview

The statistics system provides three types of instrumentation:

| Type | Purpose | Enabled By | Always On? |
|------|---------|------------|------------|
| **General Metrics** | Compilation stats, cache hits/misses, timing | `TORCH_NEURONX_METRICS_ENABLED=1` | No |
| **Memory Stats** | Current/peak memory usage, allocation counts | Always available | Yes |
| **Memory Snapshots** | Allocation history with stack traces | `_record_memory_history()` | No |

---

## At a Glance

```python
import torch
import torch_neuronx
import torch_neuronx.metrics as met
import torch.neuron.memory as mem

# === General Metrics ===
# Requires: export TORCH_NEURONX_METRICS_ENABLED=1
met.clear_all()
# ... your code ...
torch.neuron.synchronize()
print(met.report())

# === Memory Stats (always available) ===
# ... your code ...
torch.neuron.synchronize()
print(mem.memory_summary())

# === Memory Snapshots ===
mem._record_memory_history()
# ... your code ...
torch.neuron.synchronize()
mem._dump_snapshot("snapshot.pickle")
mem._record_memory_history(enabled=False)
```

---

## 1. General Metrics

Track compilation cache performance, timing, and other runtime statistics.

### Motivation

Understanding compilation behavior helps answer questions like:
- Which op took the longest to compile?
- How many times did we hit the compilation cache?
- How many ops were compiled vs reused?

### Enable

- Using environment variable - `export TORCH_NEURONX_METRICS_ENABLED=1`
- Using Python code - `torch_neuronx.metrics.set_enabled(True)`


### Usage

```python
import torch
import torch_neuronx
import torch_neuronx.metrics as met

# Clear previous stats (recommended)
met.clear_all()

# Your model code here
x = torch.randn(2, 3).to("neuron")
y = x + x
z = torch.mm(x, x.T)

# Ensure all operations complete
torch.neuron.synchronize()

# Generate report
print(met.report())
```

### Example Output

```
=== Neuron Metrics Report ===
Metric: CompilationCache.CompilationTime
  TotalSamples: 4
  Accumulator: 17s479ms254.371us
  Mean: 04s370ms813.593us
  Min: 04s260ms271.154us
  Max: 05s545ms190.693us

Metric: CompilationCache.NeffSize
  TotalSamples: 4
  Accumulator: 104.00KB
  Mean: 26.00KB
  Min: 11.00KB
  Max: 61.00KB

Counter: CompilationCache.Hits
  Value: 2

Counter: CompilationCache.Misses
  Value: 4
```

### Available metrics

Compilation stats are collected and exposed using `_C._get_compilation_cache_stats()` binding

**Counters:**

| Name | Description |
|------|-------------|
| `CompilationCache.Hits` | Total cache hits (in-memory + persistent) |
| `CompilationCache.Misses` | Total cache misses |
| `CompilationCache.InMemoryHits` | Hits from in-memory cache |
| `CompilationCache.InMemoryMisses` | Misses from in-memory cache |
| `CompilationCache.PersistentHits` | Hits from persistent (disk) cache |
| `CompilationCache.PersistentMisses` | Misses from persistent cache |

**Metrics:**

| Name | Description |
|------|-------------|
| `CompilationCache.CompilationTime` | Aggregated time spent compiling (with mean/min/max) |
| `CompilationCache.NeffSize` | Aggregated size of compiled NEFFs (with mean/min/max) |

### Available APIs

```python
import torch_neuronx.metrics as met

met.get_counter_names()      # List all counter names
met.get_counter_value(name)  # Get specific counter value
met.get_metric_names()       # List all metric names
met.get_metric_data(name)    # Get specific metric data
met.report()                 # Full metrics report
met.report_filtered(...)     # Filtered report
met.clear_counters()         # Reset counters
met.clear_metrics()          # Reset metrics
met.clear_all()              # Reset everything
met.is_enabled()             # Check if metrics enabled
met.set_enabled(True)        # Enable metrics collection
```

---

## 2. Memory Stats

Track device memory usage including allocations, cache, and peak values.

### Motivation

Memory stats help you:
- Monitor memory consumption over time
- Identify peak memory usage
- Debug unexpected OOM errors
- Compare actual vs expected memory footprint

### Usage

```python
import torch
import torch.nn.functional as F
import torch.neuron.memory as mem

# Your model code
x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to("neuron")
result = F.pad(x.unsqueeze(0).unsqueeze(1), (1, 1, 1, 1))

# Drop the output to let its block return to the pool's free set.
del result
torch.neuron.synchronize()

# Get memory summary
print(mem.memory_summary())

# Or get raw stats as dict
stats = mem.memory_stats()
```

### Example Output

```bash
# memory_summary() output -

===========================================================================
  Neuron Memory Summary, Device 0
===========================================================================
  Allocations: 2  |  Frees: 1
  OOMs: 0  |  Alloc Retries: 0
===========================================================================
  Metric              | Current    | Peak       | Allocated  | Freed
---------------------------------------------------------------------------
  Allocated Memory   |      256 B |      512 B |      512 B |      256 B
  Reserved Memory    |    2.00 MB |    2.00 MB |    2.00 MB |        0 B
  Active Memory      |       24 B |      104 B |      104 B |       80 B
===========================================================================
```

### Understanding the Metrics

Breakdown of above example:

- Input tensor: 2×3 × 4 bytes (float32) = 24 bytes requested. The pool
  rounds up to `kMinRoundedSize` = 256 B, so the block is **256 B**.
- Output tensor: 4×5 × 4 bytes (float32) = 80 bytes requested. Same
  rounding rule, so the block is also **256 B**.
- Both blocks are carved from a single 2 MiB small-pool segment (one
  `nrt_tensor_allocate` call). The segment stays resident until
  `empty_cache()`.

At printing time:

- 1 tensor freed (`del result` released the 80 B → 256 B block; it merged
  back into the segment's free space via `MergeNeighbors`).
- 1 tensor still active: the 24 B → 256 B block for `x`.
- `active_bytes.peak = 104 B` — what the user requested at peak (24 + 80).
- `allocated_bytes.peak = 512 B` — rounded block space at peak
  (256 + 256). The 512 − 104 = 408 B gap is internal fragmentation from
  `RoundSize` (tiny tensors always round up to 256 B).
- `reserved_bytes.current = 2 MB` — the segment itself is still held.
  `reserved − allocated = 2 MB − 256 B` is free-block space cached inside
  the segment for fast reuse.

Invariant: `reserved_bytes >= allocated_bytes >= active_bytes`. The gap
`reserved - allocated` is cached free-block space held inside segments, and
`allocated - active` is internal fragmentation from size rounding and split
thresholds.

| Metric | Description |
|--------|-------------|
| `reserved_bytes` | Physical NRT device memory held by the segment allocator. Covers both active block space and cached free-block space inside segments. |
| `allocated_bytes` | Sum of rounded block sizes for currently-active blocks. Includes internal fragmentation from the pool's `RoundSize`/split decisions. |
| `active_bytes` | Sum of user-requested byte counts for currently-active blocks. No rounding. |
| `num_alloc_retries` | Number of OOM recovery attempts (cache drain + retry). |
| `num_ooms` | Out-of-memory errors. |
| `num_tensor_frees` | Tensor deallocations processed by the allocator. |
| `allocation_requests` | Total `allocate()` requests (including zero-size and cache hits). |

Each byte-valued stat above is reported as a dict with `current`, `peak`,
`allocated` (historical total increases), and `freed` (historical total
decreases). `num_*` counters and `allocation_requests` are scalar ints.

### Available APIs

```python
import torch.neuron.memory as mem

mem.memory_stats(device=None)           # Dict of all stats
mem.memory_summary(device=None)         # Human-readable summary
mem.memory_allocated(device=None)       # Current active-block bytes (rounded)
mem.max_memory_allocated(device=None)   # Peak active-block bytes since last reset
mem.memory_reserved(device=None)        # Current segment (NRT) bytes held
mem.max_memory_reserved(device=None)    # Peak segment bytes since last reset
mem.reset_peak_memory_stats(device=None)       # Reset peak tracking
mem.reset_accumulated_memory_stats(device=None) # Reset counters
```

---

## 3. Memory Snapshots

Capture detailed allocation history with Python stack traces for visualization.

### Motivation

Memory snapshots help you:
- Visualize memory usage over time
- Identify which code paths allocate the most memory
- Debug memory leaks by tracing allocation sources
- Generate timeline views of tensor lifetimes

### Usage

```python
import torch
import torch.neuron.memory as mem

# Enable recording
mem._record_memory_history()

# Your model code
model = MyModel().to("neuron")
output = model(input_tensor)

torch.neuron.synchronize()

# Dump snapshot to file
mem._dump_snapshot("my_snapshot.pickle")

# Disable recording when done
mem._record_memory_history(enabled=False)
```

### Visualization

Upload the generated `.pickle` file to the [PyTorch Memory Visualizer](https://pytorch.org/memory_viz) to view:

- **Active Memory Timeline**: Shows live tensor memory over time
- **Allocator State History**: Detailed allocation/deallocation events with stack traces

### `_record_memory_history` Options

```python
mem._record_memory_history(
    enabled=True,       # True/"all" to enable, False/None to disable
    context="all",      # "all", "alloc", "state", or "never" for stack traces
    stacks="python",    # "python" (default)
    max_entries=10000,   # Max trace entries (ring buffer)
    device=None         # Device index
)
```

| `context` Value | Behavior |
|-----------------|----------|
| `"all"` | Capture stack traces for all events |
| `"alloc"` | Capture only for allocations |
| `"state"` | Capture for current state only |
| `"never"` | No stack traces |

### Available APIs

```python
import torch.neuron.memory as mem

mem._record_memory_history(...)  # Enable/disable recording
mem._snapshot()                  # Get snapshot as dict
mem._dump_snapshot(path)         # Save snapshot to pickle file
```

---

## 4. Adding Custom Telemetry

This section covers how to add custom metrics from C++ code in torch-neuronx.

### Overview

The torch-neuronx metrics system provides two main primitives:

1. **Counters** - Simple atomic integers for counting events
2. **Metrics** - Time-series data with samples, accumulator, and statistics (mean/min/max)

### Quick Reference: C++ Macros

| Use Case | Macro |
|----------|-------|
| Count events | `TORCH_NEURONX_COUNTER(name, value)` |
| Record numeric values | `TORCH_NEURONX_METRIC(name, value)` |
| Record byte sizes | `TORCH_NEURONX_BYTES_METRIC(name, value)` |
| Record timing (manual) | `TORCH_NEURONX_TIME_METRIC(name, start, end)` |
| Auto-time a scope | `TORCH_NEURONX_TIMED_SECTION(name)` |

### Use Case 1: Counting Events

**C++:**

```cpp
#include "torch_neuronx/csrc/core/metrics/NeuronMetrics.h"

void MyFunction() {
    // Increment counter by 1 each time this runs
    TORCH_NEURONX_COUNTER("MyFeature.CallCount", 1);

    // Or increment by arbitrary amount
    TORCH_NEURONX_COUNTER("MyFeature.ItemsProcessed", num_items);
}
```

**Python (access using API):**

```python
import torch_neuronx.metrics as met
met.set_enabled(True)

# After running code...
print(met.get_counter_value("MyFeature.CallCount"))
print(met.report())  # Full report
```

### Use Case 2: Auto-Scoped Timing

**C++:**

```cpp
void ExpensiveOperation() {
    TORCH_NEURONX_TIMED_SECTION("MyFeature.ExpensiveOpTime");
    // ... code to time ...
}  // Automatically records duration when scope exits
```

**Python (access using API):**

```python
data = met.get_metric_data("MyFeature.ExpensiveOpTime")
# Returns: {"accumulator": total_ns, "total_samples": N, "repr": "formatted"}
```

### Use Case 3: Manual Timing

```cpp
auto start = std::chrono::steady_clock::now();
// ... work ...
auto end = std::chrono::steady_clock::now();
TORCH_NEURONX_TIME_METRIC("MyFeature.SpecificPhase", start, end);
```

### Use Case 4: Byte Metrics

```cpp
TORCH_NEURONX_BYTES_METRIC("MyFeature.DataTransferred", bytes_sent);
```

Output formats as human-readable: 1.5MB, 256KB, etc.

### Use Case 5: Numeric Values

```cpp
TORCH_NEURONX_METRIC("MyFeature.BatchSize", batch_size);
TORCH_NEURONX_METRIC("MyFeature.Accuracy", accuracy_pct);
```

The report shows mean/min/max automatically.

### Use Case 6: Direct Class Usage

For more control (e.g., persistent metric objects):

```cpp
#include "torch_neuronx/csrc/core/metrics/NeuronMetrics.h"

namespace {
    // Static metric - registered once, reused
    at::neuron::metrics::NeuronMetric g_latency_metric(
        "MyFeature.Latency",
        at::neuron::metrics::MetricFnTime,  // Format as time
        1024  // Max samples to keep
    );

    at::neuron::metrics::NeuronCounter g_call_counter("MyFeature.Calls");
}

void MyFunction() {
    g_call_counter.AddValue(1);

    at::neuron::metrics::TimedSection timer(&g_latency_metric);
    // ... work ...
}  // Timer records on destruction
```

### Naming Convention

Use dot-separated hierarchical names: `Component.Subcomponent.MetricName`

**Examples:**
- `CompilationCache.InMemoryHits`
- `CompilationCache.PersistentMisses`
- `KernelLauncher.TotalLaunches`

### Concrete Example: Adding a New Feature Metric

**Say you want to track kernel launch frequency and latency:**

**1. C++ side (MyKernelLauncher.cpp):**

```cpp
#include "torch_neuronx/csrc/core/metrics/NeuronMetrics.h"

void LaunchKernel(const Kernel& k) {
    TORCH_NEURONX_COUNTER("KernelLauncher.TotalLaunches", 1);

    TORCH_NEURONX_TIMED_SECTION("KernelLauncher.LaunchLatency");
    // ... actual launch code ...
}
```

**2. Python side (query):**

```python
import torch_neuronx.metrics as met

met.set_enabled(True)
# ... run workload ...

print(f"Total launches: {met.get_counter_value('KernelLauncher.TotalLaunches')}")
latency = met.get_metric_data("KernelLauncher.LaunchLatency")
print(f"Avg latency: {latency['repr']}")
```

---

## Notes

- Always call `torch.neuron.synchronize()` before reading stats or calling `reset_peak_memory_stats()`. The allocator is asynchronous; unsynchronized reads reflect whatever state the RuntimeWorker happens to be in.
- Memory stats are always collected with no user-visible overhead.
- `reserved_bytes` is the physical NRT device memory held by the segment allocator. It stays resident across alloc/free cycles so blocks can be recycled without another NRT round-trip, and is only returned to NRT by `torch_neuronx.empty_cache()` (or the OOM recovery path, which drains caches automatically before retrying a failed allocation).
- Peak behavior on Neuron is measured against physical on-device state, not Python-side outstanding `DataPtr`s. Workloads that queue allocations in Python faster than the worker drains them can push `peak` above what a synchronous allocator (e.g. CUDA) would report; this is expected and reflects real in-flight device memory usage.
- General metrics and memory snapshots add minimal overhead and should be disabled in production.
- The snapshot format is compatible with PyTorch's memory visualization tool (although not all features can be used at the moment, since the tool is hardcoded for cuda).

---

## References

- [PyTorch CUDA Memory Documentation](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html)
- [PyTorch Memory Visualizer](https://pytorch.org/memory_viz)
