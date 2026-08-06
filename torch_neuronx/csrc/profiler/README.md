# Neuron Profiler

Hooks NRT profiling into `torch.profiler`. PyTorch has two separate profiler subsystems
and both need implementations for this to work.

## Quick Start

```python
import torch
import torch_neuronx
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.PrivateUse1]) as prof:
    x = torch.randn(10, 5, device="neuron")
    y = x @ x.T
# Traces written to current dir by NRT
```

With custom config:
```python
from torch._C._profiler import _ExperimentalConfig

config = "max_events_per_nc:10000;capture_enabled_for_nc:0,1;profile_output_dir:/tmp/traces"
with profile(
    activities=[ProfilerActivity.PrivateUse1, ProfilerActivity.CPU],
    experimental_config=_ExperimentalConfig(custom_profiler_config=config),
) as prof:
    # workload
    pass
```

With explicit _asynchronous_ system-profile export:
```python
from torch_neuronx.profiling import (
    ContinuousExportConfig,
    InspectExportMask,
    NeuronConfig,
    ProfileMode,
    request_export,
)

config = NeuronConfig(
    modes=[ProfileMode.RUNTIME],
    continuous_export=ContinuousExportConfig(),
)
with profile(activities=[ProfilerActivity.PrivateUse1], experimental_config=config):
    # workload steps1
    request_export(InspectExportMask.SYSTEM_PROFILE)
    # workload steps2.
    request_export(InspectExportMask.SYSTEM_PROFILE)
    # workload steps3.
    # step3 emitted events will be collected at session end.
```

With _synchronous_ system-profile export (blocks the calling thread until the drain completes; needs
no `ContinuousExportConfig()` — it is a direct `nrt_inspect_export` passthrough available in any
session):
```python
from torch_neuronx.profiling import InspectExportMask, NeuronConfig, ProfileMode, export_now

config = NeuronConfig(modes=[ProfileMode.RUNTIME])
with profile(activities=[ProfilerActivity.PrivateUse1], experimental_config=config):
    # workload steps1
    export_now(InspectExportMask.SYSTEM_PROFILE)  # returns once the drain is on disk
    # workload steps2
```

`modes` controls which activities NRT captures. Each export API or policy independently supplies the
mask to export, while `ContinuousExportConfig()` only enables the asynchronous worker.

Both APIs return `None` on success and raise `NRTProfilerError` on failure (`.status` carries the
verbatim NRT status for `export_now`, or the frontend `ExportStatus` gating reason for
`request_export`). The async drain itself runs later on a background thread; its outcome is
reported only in the Neuron logs.

## Architecture

### Profiler Registrations

The profiler requires two separate registration mechanisms:

**1. Kineto ActivityProfiler** (Registration.cpp) — Collects device traces
- Registered via `libkineto::api().registerProfilerFactory()`
- Called when `import torch_neuronx` runs `_C._register_profiler()`
- Creates `NeuronActivityProfilerSession` which calls NRT Profiling APIs

**2. ProfilerStubs** (TorchStubs.cpp) — CPU-side timing
- Registered via static initializer at module load
- The `enabled()` method is the gate: if it returns false when `PrivateUse1` is in the
    activity list, PyTorch does not like it.
- `record()` captures CPU timestamps for op correlation

### Session Lifecycle

```
profile().__enter__
  → Kineto calls our factory → NeuronActivityProfiler::configure()
  → Creates NeuronActivityProfilerSession
  → session.start()
      → nrt_inspect_config_allocate()
      → nrt_inspect_begin_with_options()

# (profiled workload executes)

profile().__exit__
  → session.stop()
      → nrt_inspect_stop()  # writes neuron trace to trace-output directory.
  → session.processTrace()
      → drains ProfilerEventBuffer
      → emits GenericTraceActivity pairs (dispatch + execution) per profiled op
```

The `processTrace()` callback drains the `ProfilerEventBuffer` and emits two
`GenericTraceActivity` entries per profiled operation: a `PRIVATEUSE1_RUNTIME` activity
for the host dispatch window (submit → execute_start) and a `PRIVATEUSE1_DRIVER` activity
for the device execution window (execute_start → execute_end). These activities include
flow arrows linking back to the originating CPU op via Kineto correlation IDs.

## Implementation Details

### NRT API Calls

| When | NRT Function |
|------|--------------|
| Session start | `nrt_inspect_begin_with_options()` |
| Explicit mid-session export | `nrt_inspect_export(mask)` |
| Session stop | `nrt_inspect_stop()` |

### Activity Type Mapping

| Kineto | NRT | What it captures |
|--------|-----|------------------|
| `PRIVATEUSE1_RUNTIME` | `system_profile` | Host-side NRT calls |
| `PRIVATEUSE1_DRIVER` | `device_profile` | On-device execution |
| `CPU_OP` | `cpu_util` | CPU utilization |

### Config Options

Passed via `_ExperimentalConfig(custom_profiler_config="key:value;key:value")`:

- `max_events_per_nc` — Event buffer size per NeuronCore
- `capture_enabled_for_nc` — Which NCs to profile ("0,1,2-5")
- `host_memory` — Profile host memory (bool)
- `profile_output_dir` — Where NRT writes traces
- `continuous_export` — Enables the async export worker for explicit `request_export(mask)` calls

## Testing

```sh
# Unit tests
./tools/run-cpp-test //tests/csrc:SessionTest
./tools/run-cpp-test //tests/csrc:NConfigParserTest
./tools/run-cpp-test //tests/csrc:ProfilerEventBufferTest
./tools/run-cpp-test //tests/csrc:CorrelationTrackerTest

# Integration
./tools/run-test profiler/test_profiler_integration.py -v
```
