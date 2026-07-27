# Vertical Tests

ATen op coverage testing framework for Neuron using the `NeuronOpInfo` shape registry.

## Overview

Tests all ~1362 target ATen ops for coverage, performance, and correctness on Neuron. Uses an explicit shape registry — every op either has registered shapes or is marked `pending_registry`.

## Directory Structure
```
tests/vertical/
├── README.md
├── __init__.py
├── conftest.py                     # Aggregates coverage CSV, session hooks, --shape-registry
├── internal/                       # NeuronOpInfo framework
│   ├── __init__.py
│   ├── target_ops.py               # Curated list of ~1362 target ATen ops
│   ├── neuron_shape_registry.py    # Op-specific shape registry (CSV/JSON backed)
│   ├── neuron_shape_defaults_generated.py  # Auto-generated shapes
│   ├── neuron_op_info.py           # NeuronOpInfo extending PyTorch OpInfo
│   ├── neuron_op_db.py             # build_neuron_op_db() merging all sources
│   ├── generate_op_shape_registry.py       # Tool: cache keys → op_shape_registry.json
│   ├── test_generate_op_shape_registry.py  # Unit tests for the parser
├── test_target_ops_coverage.py             # Eager coverage + perf + correctness
├── test_target_ops_compile_coverage.py     # torch.compile coverage + perf + correctness
├── test_dynamic_aten_ops.py                # dynamic-shape NKI kernels perf (compared to static) + correctness
```

## Running ATen Ops Tests

```sh
# Run all eager coverage tests in parallel (recommended)
python -m pytest tests/vertical/test_target_ops_coverage.py -n 32 --tb=no -q

# Run all compile coverage tests in parallel
python -m pytest tests/vertical/test_target_ops_compile_coverage.py -n 32 --tb=no -q

# Run a single op (eager)
python -m pytest "tests/vertical/test_target_ops_coverage.py::TestTargetOpsCoverage::test_op[aten::linear[linalg]]" -v -s

# Run a single op (compile)
python -m pytest "tests/vertical/test_target_ops_compile_coverage.py::TestTargetOpsCompileCoverage::test_op_compile[aten::linear[linalg]]" -v -s

# Run ops matching a pattern
python -m pytest tests/vertical/test_target_ops_coverage.py -v -k "relu or gelu"

```

## Running Dynamic ATen Ops Tests

```sh

# Run all dynamic NKI kernels
python -m pytest tests/vertical/test_dynamic_aten_ops.py -n 8 --tb=no -q

# Run a single op
python -m pytest "tests/vertical/test_dynamic_aten_ops.py::TestDynamicAtenOps::test_op[aten::mm]" -v -s

# Run ops matching a pattern
python -m pytest tests/vertical/test_dynamic_aten_ops.py -v -k "silu or mm"

# Limit each op to its N largest shapes (default 10)
python -m pytest "tests/vertical/test_dynamic_aten_ops.py::TestDynamicAtenOps::test_op[aten::mm]" --max-shapes-per-op 2 -v -s

```

neuron-profile artifacts (ntff + NEFFs) are written under each test's pytest tmp dir: `/tmp/pytest-of-<user>/pytest-<N>/`.

## ATen Ops Coverage CSV

Results are aggregated into:
- **Eager**: `target_ops_coverage_<timestamp>.csv`
- **Compile**: `target_ops_compile_coverage_<timestamp>.csv`

| Column | Description |
|--------|-------------|
| `op_name` | Full ATen op name (e.g., `aten::add.Tensor`) |
| `op_group` | Category: elementwise, linalg, nn, reduction, indexing, etc. |
| `status` | `supported`, `missing`, `failed_lowering`, `pending_registry` |
| `supported_by` | `registered`, `composite`, `variant_covered`, `unknown` |
| `input_source` | `shape_registry_model`, `shape_registry_default`, `pytorch_db`, `pending_registry` |
| `input_element_count` | Total input element count of the slowest-executing shape (empty when no shape ran) |
| `neuron_primitive_ops` | ATen ops executed on Neuron |
| `fallback_primitive_ops` | ATen ops that fell back to CPU |
| `time_to_first_ms` | Time to first result (ms) |
| `exec_avg_ms` / `exec_min_ms` / `exec_max_ms` | Execution time stats |
| `error` | Error message if failed |
| `last_updated` | Timestamp |
| `correctness_test` | `pass`, `fail`, or empty |

## Dynamic ATen Ops CSV

Per-op rows are merged into one timestamped `dynamic_aten_ops_<timestamp>.csv`. Each row is one (op, shape, dtype) cell profiled in both **dynamic** (NKI kernel) and **static** (default lowering) modes.

| Column | Description |
|--------|-------------|
| `op_name` | Full ATen op name (e.g., `aten::add.Tensor`) |
| `input_shape` | Tensor-arg dims joined with `x`, args with `_` (scalar args skipped) |
| `dtype` | Float dtype of the swept sample (e.g., `bfloat16`) |
| `input_size_mb` | Total input size (sum of input-tensor bytes) in MiB |
| `dynamic_time_to_first_ms` / `static_time_to_first_ms` | Cold time-to-first (compile + load + execute), per mode |
| `dynamic_e2e_time_us` / `static_e2e_time_us` | Warm wall-clock latency incl. host overhead, per mode |
| `e2e_ratio` | `dynamic_e2e / static_e2e` (>1 ⇒ dynamic slower) |
| `dynamic_device_time_us` / `static_device_time_us` | On-device busy time (neuron-profile `total_exec_time`), per mode |
| `device_ratio` | `dynamic_device / static_device` |
| `correctness_test` | `pass` / `fail` — device output vs CPU reference |
| `last_updated` | Timestamp |
| `error` | Error message, or an informational note (e.g. `fell back` when the dynamic kernel declined and the op ran via the default lowering) |

## Input Source Priority

1. **Shape registry** (`internal/neuron_shape_registry.py`) — explicit shapes per op
2. **PyTorch op_db** — for base ops and `.default`/`.out` overloads only
3. **`pending_registry`** — no input source, op is skipped

## Adding Shapes for New Ops

Register in `internal/neuron_shape_registry.py` or `internal/neuron_shape_defaults_generated.py`:

```python
SHAPE_REGISTRY.register("aten::my_op",
    arg_shapes={"self": (256, 4096), "weight": (4096, 4096), "bias": None},
    source="manual")
```

For ops needing specific dtypes:

```python
SHAPE_REGISTRY.register("aten::my_index_op",
    arg_shapes={
        "self": (256, 4096),
        "index": {"shape": (64,), "dtype": "long", "low": 0, "high": 256},
        "mask": {"shape": (256, 4096), "dtype": "bool"},
    },
    source="manual")
```

## Loading Shapes from Model Traces

### End-to-end workflow

1. **Run a model** to generate compilation artifacts:
   ```sh
   export TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS=1
   torchrun --nproc-per-node=64 run_benchmarks.py --block qwen3_torchtitan --preset qwen3-8b-fsdp
   ```

2. **Generate JSON** from the cache keys in the artifacts:
   ```sh
   python -m tests.vertical.internal.generate_op_shape_registry \
       /tmp/neuron_backend/neuronx-cc -o op_shape_registry.json
   ```
   Or from a cache key text file (one key per line):
   ```sh
   python -m tests.vertical.internal.generate_op_shape_registry \
       --cache-keys aggregate_cache_keys.txt -o op_shape_registry.json
   ```

3. **Run coverage tests** with the generated shapes:
   ```sh
   python -m pytest tests/vertical/test_target_ops_coverage.py -n 64 \
       --shape-registry op_shape_registry.json --model-shapes-only --tb=no -q
   ```

### Pytest options

| Option | Description |
|--------|-------------|
| `--shape-registry <path>` | Load a JSON shape registry file into `SHAPE_REGISTRY` |
| `--model-shapes-only` | For ops with model-traced shapes, skip builtin/default shapes |

### Programmatic usage

```python
from internal.neuron_shape_registry import SHAPE_REGISTRY

# Load from JSON (output of model tracing)
SHAPE_REGISTRY.load_json("llama7b_shapes.json")

# Load from consolidated CSV
SHAPE_REGISTRY.load_csv("op_shapes.csv")

# Save consolidated
SHAPE_REGISTRY.save_csv("op_shapes.csv")
```

## Status Values

| Status | Meaning |
|--------|---------|
| `supported` | Op runs on Neuron without CPU fallback |
| `missing` | Op not implemented on Neuron |
| `failed_lowering` | Op registered but execution fails (real Neuron bug) |

## Input Source Values

| Input Source | Meaning |
|--------------|---------|
| `shape_registry_model` | Inputs from model-traced shapes (via `--shape-registry`) |
| `shape_registry_default` | Inputs from builtin/default shape registry entries |
| `pytorch_db` | Inputs from PyTorch's native op_db |
| `pending_registry` | No valid inputs available — needs shapes registered to test |

Note: When `input_source=pending_registry`, the `status` is inferred from dispatch classification (registered → `supported`, composite → `supported`, unknown → `missing`) rather than from actual execution.
