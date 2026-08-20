# Vertical Tests

ATen op coverage testing framework for Neuron using the `NeuronOpInfo` shape registry.

## Overview

Tests all ~1362 target ATen ops for coverage, performance, and correctness on Neuron. Uses an explicit shape registry — every op either has registered shapes or is marked `pending_registry`.

## Directory Structure
```
tests/vertical/
├── README.md
├── __init__.py
├── conftest.py                     # Aggregates coverage CSV, session hooks, shape-selection opts
├── internal/                       # NeuronOpInfo framework
│   ├── __init__.py
│   ├── target_ops.py               # Curated list of ~1362 target ATen ops
│   ├── neuron_shape_registry.py    # Op-specific shape registry (CSV/JSON backed)
│   ├── neuron_shape_defaults_generated.py  # Auto-generated shapes + model-traced sets
│   ├── neuron_op_info.py           # NeuronOpInfo extending PyTorch OpInfo
│   ├── neuron_op_db.py             # build_neuron_op_db() merging all sources
│   ├── generate_op_shape_registry.py       # Tool: cache keys → op_shape_registry.json
│   ├── test_generate_op_shape_registry.py  # Unit tests for the parser
├── accuracy/                       # Op-accuracy suite (measured ULP vs. derived budget)
│   ├── conftest.py                 # ROW_FIELDS CSV aggregation, --accumulator/--dtypes
│   ├── budgets.py                  # Per-op error budget models (the theory side)
│   ├── formats.py                  # NumericFormat: unit roundoff + reference upcast
│   ├── metrics.py                  # norms/ULP conversion, cell id, seeding
│   ├── test_metrics.py             # Phase 1: the error model, CPU-only, no device
│   └── test_accuracy.py            # Phase 2: on-device sweep, one row per cell
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

neuron-profile artifacts (ntff + NEFFs) are written under `aten_profiles/<cell>/` in the working directory (override with `AOP_PROFILE_DIR`); the path is recorded per row in the CSV's `profile_rel_path` column.


## Running Op Accuracy Tests

Measures each op's actual error against a budget *derived from its arithmetic*, in units of
the input format's unit roundoff (`u_in = 2**-precision`; bf16 → `2**-8`). The check is
`max_err_ulp <= budget_ulp`, where `max_err_ulp = (max|T-R| / max|R|) / u_in`. This is
supplementary to `torch.testing.assert_close`, not a replacement: upstream's bf16 `rtol` of
`0.016` is `4.1 u_in`, so it allows ~4 roundings' worth of error regardless of the op.

```sh
# Phase 1 -- the error model only. No device, unit test for accuracy related classes.
python -m pytest tests/vertical/accuracy/test_metrics.py -q

# Phase 2 -- on-device sweep against a model's traced op+shape list.
python -m pytest tests/vertical/accuracy/test_accuracy.py \
    --traced-model qwen3_235b_a22b --max-shapes-per-op 0 -n 32 --tb=no -q

# A single op, verbose
python -m pytest "tests/vertical/accuracy/test_accuracy.py::TestOpAccuracy::test_op[aten::bmm]" \
    --traced-model qwen3_235b_a22b --max-shapes-per-op 0 -s --log-cli-level=INFO

# No --traced-model: sweeps the builtin pool instead, kept to ops that have a budget model.
python -m pytest tests/vertical/accuracy/test_accuracy.py -n 32 --tb=no -q

# Ask what the budget would be if the accumulator were bf16 instead of fp32
python -m pytest tests/vertical/accuracy/test_accuracy.py --accumulator bfloat16 ...

# Include shapes above the shared 512Mi-element guard (NeuronOpInfo._MAX_TENSOR_ELEMS).
# For qwen3_235b_a22b that guard drops 3 of 181 entries -- the 151936-vocab lm_head
# projections in mm / linear / index.Tensor, at 622M elements each.
NEURON_MAX_TENSOR_ELEMS=700000000 python -m pytest tests/vertical/accuracy/test_accuracy.py \
    --traced-model qwen3_235b_a22b --max-shapes-per-op 0 -n 32 --tb=no -q
```

**Pass `--traced-model` to sweep a model.** It does two things — registers that set's shapes,
and restricts the run to them. Both are needed: registering alone would leave the op's
`builtin` shapes in the pool as well, so the sweep would report more cells than the set has
entries and would not be purely model-shaped. `--model-shapes-only` excludes `builtin` but
cannot separate two model sets registered together; `--traced-model` selects exactly one.

**Not passing it registers no model set at all**, so every other suite here measures exactly
the builtin pool it always has. See [Model-traced shape sets](#model-traced-shape-sets).

**Not a gate.** `test_op` passes as long as cells ran; the verdict lives in the CSV. An op
that could not be measured stays visible as a `status` rather than being folded into a pass.

### Accuracy-specific pytest options

| Option | Default | Description |
|--------|---------|-------------|
| `--accumulator` | `float32` | Assumed accumulator dtype. The parameter the budget turns on: at n=4096 fp32 budgets ~1.0 ULP, bf16 ~65 |
| `--reduction-order` | `sequential` | Growth model: `sequential` → `sqrt(n)`, `pairwise` → `sqrt(log2 n)`. Differ 25x at n=8192, but only when the accumulator is narrow |
| `--dtypes` | `bfloat16` | Comma-separated input dtypes to sweep |
| `--seed-base` | `0` | Re-roll every cell's input draw while keeping cells independent |

## Op Accuracy CSV

One row per `(op, shapes, dtype)` cell, aggregated into `op_accuracy_<timestamp>.csv`. The
budget's *inputs* are carried alongside its output so a verdict can be re-derived from the
row without re-running the op.

| Column | Description |
|--------|-------------|
| `cell_id` | 16-hex-char id of `(op, arg_shapes, dtype)`. Excludes the seed — the seed is derived from it |
| `op` / `op_group` | Full ATen op name; `TARGET_ATEN_OP_GROUP` category |
| `dtype` | Input dtype the cell was measured at |
| `arg_shapes` | JSON of the registry entry used |
| `unit` | `u_in`, the input format's unit roundoff (bf16 → `0.00390625`) |
| `n` | Reduction width the budget was derived from. `1` = elementwise, `0` = bit-exact group, empty = unmodelled |
| `u_acc` | Assumed accumulator unit roundoff (from `--accumulator`) |
| `order` | Growth model (from `--reduction-order`) |
| `max_err_ulp` | **The measurement.** `(max\|T-R\| / max\|R\|) / u_in` |
| `fro_err_ulp` | Same, Frobenius-normalized. Catches broad drift that L-inf dilutes |
| `budget_ulp` | **The prediction.** `const + coeff * g(n) * (u_acc / u_in)` |
| `verdict` | `within`, `over`, `invalid`, or empty when `status != ok` |
| `max_abs_err` / `max_abs_ref` / `fro_abs_err` / `fro_abs_ref` | The four raw scalars the ULP figures are computed from |
| `seed` | RNG seed, derived from `cell_id` + `--seed-base` |
| `gates` | JSON of the shape-selection knobs in effect (`max_shapes_per_op`, `shape_registry`, `traced_model`, ...) |
| `status` | See below |
| `error_cause_raw` | First 200 chars of the failure, when there is one |

### Verdicts

| Verdict | Meaning |
|---------|---------|
| `within` | `max_err_ulp <= budget_ulp` — the op is as accurate as its arithmetic predicts |
| `over` | Exceeded the budget. Includes `inf`, which is deliberate: an error against an exact-zero reference, or a bit-exact op that did not reproduce exactly |
| `invalid` | `max_err_ulp` is `nan` — the output held a `nan`, or both sides overflowed to `inf`. Not a comparison, so not counted either way |

### Statuses

| Status | Meaning |
|--------|---------|
| `ok` | Cell measured; read `verdict` |
| `no_budget` | Measured, but no budget model for this op. Stays countable rather than passing silently |
| `excluded` | `random` group — two correct RNG implementations disagree elementwise |
| `no_input_sample` | The op's schema could not be satisfied from the registry entry |
| `no_reference` | The CPU fp32 reference could not run, or its output arity disagreed with the device's |
| `unresolved` | Op name did not resolve to a callable |
| `failed_lowering` | Device allocation or execution failed (see `error_cause_raw`) |

## Running Dynamic Collectives Tests

Distributed all-gather benchmark. Each cell is one `(config, shape)` swept in both **dynamic** (NKI kernel, `NEURON_DYN_NKI_COLLECTIVES=1`) and **torch** (native) modes, each in its own cold `torchrun` job. Configs are defined in `_CONFIGS`: `tp` (contiguous group, nproc=4) and `fsdp` (stride-TP group, nproc=64).

```sh

# Run all cells (both configs). fsdp launches 64 ranks, so run serially (no -n):
python -m pytest tests/vertical/test_dynamic_collectives.py -v -s

# Run only the tp config (nproc=4) in parallel; each xdist worker owns a disjoint
# block of AG_CORE_BLOCK cores. Keep -n * AG_CORE_BLOCK within the logical-core
# budget (64 on TRN2 at LNC=2):
AG_CORE_BLOCK=4 python -m pytest tests/vertical/test_dynamic_collectives.py -k tp -n 4 -v -s

# Run a single cell (id is <config>-<h>x<w>)
python -m pytest "tests/vertical/test_dynamic_collectives.py::test_dynamic_allgather[tp-8192x4096]" -v -s

# or directly as a driver (launches the torchruns itself):
NEURON_LOGICAL_NC_CONFIG=2 AG_CSV=/tmp/ag.csv python3 tests/vertical/test_dynamic_collectives.py

```

neuron-profile artifacts (ntff + NEFFs) are written under `c10d_profiles/<h>x<w>/<mode>/` in the working directory (override with `AG_PROFILE_DIR`).

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
| `dynamic_device_time_us` / `static_device_time_us` | On-device wall-clock time (neuron-profile `total_time`), per mode |
| `device_ratio` | `dynamic_device / static_device` |
| `correctness_test` | `pass` / `fail` — device output vs CPU reference |
| `last_updated` | Timestamp |
| `error` | Error message, or an informational note (e.g. `fell back` when the dynamic kernel declined and the op ran via the default lowering) |

## Dynamic Collectives CSV

Per-cell rows are merged into one timestamped `dynamic_collectives_<timestamp>.csv`. Each row is one `(config, shape)` all-gather cell profiled in both **dynamic** (NKI kernel) and **torch** (native) modes. Tiling columns (`regime`, `*_tile_h`, `n_*`, `unroll`) are mode-independent and taken from the dynamic worker's on-device plan.

| Column | Description |
|--------|-------------|
| `op_name` | Collective op (`all_gather`) |
| `total_mib` | Gathered size in MiB (`world * h * w * elem_size / 2^20`) |
| `input_h` / `input_w` | Per-rank shard shape (`h` = seq, `w` = last dim) |
| `replica_group` | Ranks in this cell's all-gather group (== `replica_groups[0]`); stride-TP for fsdp (e.g. `[0, 4, ..., 60]`), contiguous otherwise |
| `regime` | Tiling regime name, or `native-fallback` when the plan is ineligible |
| `large_tile_h` / `medium_tile_h` / `small_tile_h` | Element tile sizes (pow2) the NEFF is baked with |
| `n_large` / `n_medium` / `n_small` / `n_rem` | Tile counts on the flat numel (`n_rem` = single small tail, 0 or 1) |
| `remainder_row_count` | Leftover elements below one small tile |
| `unroll` | Unroll variant (1/2/4) auto-picked from the tile count; `0` = native |
| `dynamic_e2e_us` / `torch_e2e_us` | Warm wall-clock latency incl. host overhead, per mode |
| `e2e_ratio` | `dynamic_e2e / torch_e2e` (>1 ⇒ dynamic slower) |
| `dynamic_device_us` / `torch_device_us` | On-device wall-clock time (neuron-profile `total_time`), per mode |
| `device_ratio` | `dynamic_device / torch_device` |
| `dynamic_cc_us` / `torch_cc_us` | On-device collective time (neuron-profile `cc_op_time`), per mode |
| `dynamic_bus_bw_gbps` / `torch_bus_bw_gbps` | Bus bandwidth (GiB/s) derived from device time (`busbw = algbw * (n-1)/n`) |
| `correctness_test` | `pass` / `fail` — gathered output vs locally-reconstructed reference |
| `last_updated` | Timestamp |

## Input Source Priority

1. **Shape registry** (`internal/neuron_shape_registry.py`) — explicit shapes per op
2. **PyTorch op_db** — for base ops and `.default`/`.out` overloads only
3. **`pending_registry`** — no input source, op is skipped

## Adding Shapes for New Ops

Register in `internal/neuron_shape_registry.py` or `internal/neuron_shape_defaults_generated.py`.
Entries added at the top level of `register_all_defaults` join the builtin pool every suite
sees; to add shapes that only one sweep picks up, put them in a model set instead (see
[Model-traced shape sets](#model-traced-shape-sets)).

```python
SHAPE_REGISTRY.register(
    "aten::my_op",
    arg_shapes={"self": (256, 4096), "weight": (4096, 4096), "bias": None},
    source="manual",
)
```

For ops needing specific dtypes:

```python
SHAPE_REGISTRY.register(
    "aten::my_index_op",
    arg_shapes={
        "self": (256, 4096),
        "index": {"shape": (64,), "dtype": "long", "low": 0, "high": 256},
        "mask": {"shape": (256, 4096), "dtype": "bool"},
    },
    source="manual",
)
```

## Loading Shapes from Model Traces

### Model-traced shape sets

A traced model's op+shape list is checked in to
`internal/neuron_shape_defaults_generated.py`, alongside the builtin defaults, so there is
one registry source rather than a second file to keep in sync. Each set contributes three
module-level names:

| Name | Purpose |
|------|---------|
| `TRACED_MODEL_SHAPES[<name>]` | The register function. `<name>` is what `--traced-model` accepts |
| `TRACED_MODEL_OPS[<name>]` | The frozenset of ops the set actually traced |
| `<NAME>` | The source tag its entries carry, so `SHAPE_REGISTRY.get(op, model=<NAME>)` works |

Currently: `qwen3_235b_a22b` — 181 entries over 54 ops, deduplicated on
`(op_name, arg_shapes)` across eager-dispatch and compiled-FX traces with counts summed.

**Registering a set is opt-in and off by default.** `register_all_defaults(r, traced_models=())`
takes an allowlist, and `--traced-model` is the only thing that fills it. That makes the gate
fail safe: adding a model to the generated module cannot change what any existing suite
measures until a run names it. Without the gate the new entries would join the shared
op-keyed pool and compete in the top-N-by-size cap in `NeuronOpInfo._registry_entries`,
silently displacing shapes the coverage suites measure today. An unknown `--traced-model` name
raises loudly.

Use `TRACED_MODEL_OPS[<name>]` — not `SHAPE_REGISTRY.all_ops(model=<name>)` — when you need the
list of ops a set traced. `propagate_to_compatible_overloads` copies entries *and their
source tag* onto schema-compatible `.out` / `.default` overloads, so `all_ops(model=)`
reports those synthesized overloads too (more ops rather than the actually traced).

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
| `--shape-registry <path>` | JSON shape registry file to load into `SHAPE_REGISTRY`, tagged `model_trace`. For a one-off trace that isn't checked in |
| `--traced-model <names>` | Comma-separated checked-in shape set(s) (keys of `TRACED_MODEL_SHAPES`). Registers them *and* restricts the run to them. Omit it and none are registered. Stricter than `--model-shapes-only`, which only excludes `builtin` and so still mixes every registered set together |
| `--model-shapes-only` | For ops with model-traced shapes, skip builtin/default shapes |
| `--max-shapes-per-op <n>` | Cap per op to the top-N entries by max tensor element count. `0` disables the cap |

### Programmatic usage

```python
from internal.neuron_shape_defaults_generated import (
    TRACED_MODEL_OPS,
    QWEN3_235B_A22B,
    register_all_defaults,
)
from internal.neuron_shape_registry import SHAPE_REGISTRY, NeuronShapeRegistry

# The process-wide registry holds builtin defaults, plus whatever --traced-model named.
SHAPE_REGISTRY.models()  # -> [] with no --traced-model; ["qwen3_235b_a22b"] with it

# A private registry with one model set registered on top of the builtin defaults.
reg = NeuronShapeRegistry()
register_all_defaults(reg.register, traced_models=[QWEN3_235B_A22B])
# that model's entries only, and the ops it traced (see the caveat above)
reg.get("aten::linear", model=QWEN3_235B_A22B)
TRACED_MODEL_OPS[QWEN3_235B_A22B]

# Load a one-off trace from JSON. Source defaults to the filename stem.
SHAPE_REGISTRY.load_json("llama7b_shapes.json")

# Load from / save to consolidated CSV
SHAPE_REGISTRY.load_csv("op_shapes.csv")
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
| `shape_registry_model` | Inputs from model-traced shapes (via `--traced-model` or `--shape-registry`) |
| `shape_registry_default` | Inputs from builtin/default shape registry entries |
| `pytorch_db` | Inputs from PyTorch's native op_db |
| `pending_registry` | No valid inputs available — needs shapes registered to test |

Note: When `input_source=pending_registry`, the `status` is inferred from dispatch classification (registered → `supported`, composite → `supported`, unknown → `missing`) rather than from actual execution.
