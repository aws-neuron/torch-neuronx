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
