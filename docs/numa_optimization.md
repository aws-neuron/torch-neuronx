# NUMA Optimization for Neuron

## Problem

On multi-rank Trainium workloads (e.g., 64 ranks on TRN2.48xlarge), processes
can suffer from:

- **Cross-NUMA memory access** — without pinning, a Neuron device's CPU threads
  may run on a different NUMA node, causing host memory accesses to cross the
  interconnect

## Solution

`torch_neuronx.numa` provides NUMA CPU affinity — pins all CPU threads in each rank's process to the same NUMA node as its assigned Neuron device.

The API follows the same patterns as `torch.numa.binding` (PyTorch 2.11+),
adapted for Neuron's PrivateUse1 device model.

## Getting started

### Tier 1: Upgrade torch_neuronx (no code changes)

NUMA CPU affinity activates automatically on first device use (via
`_lazy_init`). No user code changes required.

Pins all CPU threads to the NUMA node of the assigned Neuron device. After
pinning, new memory allocations land on the local node via Linux's
[first-touch policy](https://www.kernel.org/doc/html/latest/admin-guide/mm/numa_memory_policy.html).
However, memory allocated before the pin (Python interpreter startup, imported
modules, early torch allocations) may already reside on the wrong node.

### Tier 2: Change launch command (no code changes)

Requires `numactl` installed (`apt install numactl`). If not present,
`neuron-numa-launch` falls back to plain `python` (Tier 1 still applies).

Adds `numactl --membind --cpunodebind` at process birth — before Python starts.
This ensures all memory allocations, including early startup, land on the
correct NUMA node.

Add `--no-python` and `neuron-numa-launch` to the launch command.
`neuron-numa-launch` is installed automatically with torch_neuronx.

```bash
# Before:
torchrun --nproc_per_node 64 ... ./train.py --args

# After:
torchrun --no-python --nproc_per_node 64 ... neuron-numa-launch ./train.py --args
```

`--no-python` tells torchrun to launch `neuron-numa-launch` as a binary
instead of importing it as a Python module. `neuron-numa-launch` then wraps
the Python process with `numactl --membind --cpunodebind`, ensuring all memory
allocations (including Python startup) land on the local NUMA node.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `NEURON_NUMA_PIN` | `1` | Enable NUMA CPU affinity. Set to `0` to disable. |

## Diagnostic API

```python
from torch_neuronx.numa import get_numa_status
print(get_numa_status())
# {'numa_pin_enabled': True, 'numa_pinned': True,
#  'current_affinity': [48, 49, ...], ...}
```

## What to know

- NUMA pinning runs **after** `_neuron_init()` because `neuron-ls` subprocesses
  contend with concurrent runtime init across 64 ranks. NRT threads are
  re-pinned after creation.
- Memory binding (`numactl --membind`) can only be set at process birth — hence
  the `neuron-numa-launch` wrapper.
- Topology is discovered dynamically via `neuron-ls --json-output` (called once
  per process, cached). No hardcoded per-instance-type tables.
