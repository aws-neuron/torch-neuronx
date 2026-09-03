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
│   ├── neuron_shape_defaults_generated.py  # Auto-generated shapes + smoke test subsets
│   ├── neuron_op_info.py           # NeuronOpInfo extending PyTorch OpInfo
│   ├── neuron_op_db.py             # build_neuron_op_db() merging all sources
│   ├── generate_op_shape_registry.py       # Tool: cache keys → op_shape_registry.json
│   ├── test_generate_op_shape_registry.py  # Unit tests for the parser
├── accuracy/                       # Op-accuracy suite (normwise relative error vs. a flat 1-ULP budget)
│   ├── conftest.py                 # ROW_FIELDS CSV aggregation, --dtypes/--seed-base
│   ├── budgets.py                  # The threshold and which groups it applies to
│   ├── formats.py                  # NumericFormat: unit roundoff + reference upcast
│   ├── metrics.py                  # norms, relative-error conversion, cell id, seeding
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

python -m pytest "tests/vertical/test_dynamic_aten_ops.py::TestDynamicAtenOps::test_op[aten::_grouped_mm]" -v -s

# Run ops matching a pattern
python -m pytest tests/vertical/test_dynamic_aten_ops.py -v -k "silu or mm"

# Limit each op to its N largest shapes (default 10)
python -m pytest "tests/vertical/test_dynamic_aten_ops.py::TestDynamicAtenOps::test_op[aten::mm]" --max-shapes-per-op 2 -v -s

```

neuron-profile artifacts (ntff + NEFFs) are written under `aten_profiles/<cell>/` in the working directory (override with `AOP_PROFILE_DIR`); the path is recorded per row in the CSV's `profile_rel_path` column.


## Running Op Accuracy Tests

Measures each op's **normwise relative error**, expressed in units of the input format's unit
roundoff (`unit = 2**-precision`; bf16 → `2**-8`). The check is
`max_rel_err_units <= budget_units`, where

```
max_rel_err_units = (max|T-R| / max|R|) / unit
```

**The budget is one flat constant: `2.0` units = 1 full ULP = the output format's machine
epsilon** (bf16 → `2**-7`).

**Only formats coarser than fp32 are gated.** At fp32 the accumulator *is* the input format,
so accumulation rather than rounding sets the scale and no constant bounds it — a correct
fp32 reduction over 4096 values drifts ~64 units. Those cells report `no_gate`.

```sh
# Phase 1 -- the error model only. No device, unit test for accuracy related classes.
python -m pytest tests/vertical/accuracy/test_metrics.py -q

# Phase 2 -- on-device sweep against a model's traced op+shape list. --max-shapes-per-op 0
# sweeps every registered shape; the dtype sweep defaults to both bf16 and fp32.
# qwen3_235b_ep64_tp4 is traced BOTH ways (eager + compiled); one run writes one CSV per
# path: op_accuracy_eager_<ts>.csv and op_accuracy_compiled_<ts>.csv.
python -m pytest tests/vertical/accuracy/test_accuracy.py \
    --traced-model qwen3_235b_ep64_tp4 --max-shapes-per-op 0 -n 32 --tb=no -q

# One dtype only
python -m pytest tests/vertical/accuracy/test_accuracy.py \
    --traced-model qwen3_235b_ep64_tp4 --max-shapes-per-op 0 --dtypes bfloat16 -n 32 --tb=no -q

# A single op, verbose
python -m pytest "tests/vertical/accuracy/test_accuracy.py::TestOpAccuracy::test_op[aten::bmm]" \
    --traced-model qwen3_235b_ep64_tp4 --max-shapes-per-op 0 -s --log-cli-level=INFO

# Include shapes above the shared 512Mi-element guard (NeuronOpInfo._MAX_TENSOR_ELEMS).
# No entry in qwen3_235b_ep64_tp4 hits it -- the largest tensor arg in the set is the 64-expert
# MoE weight stack at 64x1536x4096 = 402M elements -- so this is a knob for future sets.
NEURON_MAX_TENSOR_ELEMS=700000000 python -m pytest tests/vertical/accuracy/test_accuracy.py \
    --traced-model qwen3_235b_ep64_tp4 --max-shapes-per-op 0 -n 32 --tb=no -q
```

**`--traced-model` is required** — with no model the sweep fails loudly. It does two things —
registers that set's shapes, and restricts the run to them. Both are needed: registering alone
would leave the op's `builtin` shapes in the pool as well, so the sweep would report more cells
than the set has entries and would not be purely model-shaped. `--model-shapes-only` excludes
`builtin` but cannot separate two model sets registered together; the traced model selects
exactly one.

**`--max-shapes-per-op`** caps the shapes swept per op: `0` sweeps **all** registered shapes; a
positive `n` keeps only the `n` largest-tensor shapes per op. **`--dtypes`** defaults to both
bf16 and fp32; pass a comma-separated subset to narrow it.

**This gates on both metrics.** Each `(op, dtype)` is its own item; it fails if any cell is
over/invalid on the **rel-err budget** (bf16 only — fp32 has no rel-err verdict, `no_gate`) or
on the **Frobenius ratio** (every dtype). Every cell still emits a CSV row regardless. A cell
with an empty verdict — a device/lowering error (`failed_lowering`, …), `no_input_sample`,
`unresolved`, `no_reference` — is shown as `other` in the panel (never folded into a pass) but
is **not gated**: op runnability/lowering is owned by the vertical coverage suites
(`test_target_ops_*`); this accuracy suite only judges onboarded ops that actually run.

### CI

Runs as a first-class trn2 check in **PR Checks** (`pull.yml`, peer to the ATen/CPP suites) and
in **Dependency Guard** (`dependency-guard.yml`, blocks stable-image promotion). Each job runs
one command: `--traced-model qwen3_235b_ep64_tp4 --max-shapes-per-op 0` over both dtypes.
`NEURON_ACCURACY_OUTPUT_DIR=outputs/accuracy` puts the
per-path CSV(s) + a `step_summary.md` metrics panel under the uploaded artifact; the panel is
posted to the GitHub job summary via the `append-step-summary` action. It has two head-level
metrics — **relative error** (gates bf16) and **Frobenius ratio** (gates every dtype) — each
split by dtype into a Tests (ops+shapes) and an Ops table over
pass/fail/xfail/skip/xpass/other, plus per-(metric, dtype) failing-cell tables.

### Accuracy-specific pytest options

| Option | Default | Description |
|--------|---------|-------------|
| `--dtypes` | `bfloat16,float32` | Comma-separated input dtypes to sweep. fp32 has no rel-err verdict (`no_gate`) but is gated on the Frobenius ratio |
| `--seed-base` | `0` | Re-roll every cell's input draw while keeping cells independent |

## Op Accuracy CSV

One row per `(op, shapes, dtype, path)` cell, aggregated into one CSV per capture path (`op_accuracy_eager_<timestamp>.csv` / `op_accuracy_compiled_<timestamp>.csv`; a single unsplit run writes `op_accuracy_<timestamp>.csv`). `unit`
is carried alongside the measurement so a verdict can be re-read from the row without
re-running the op, and `arg_shapes` so anything shape-derived (reduction width, element count)
stays recomputable without re-running it either.

| Column | Description |
|--------|-------------|
| `cell_id` | 16-hex-char id of `(op, arg_shapes, dtype)`. Excludes the seed — the seed is derived from it |
| `op` / `op_group` | Full ATen op name; `TARGET_ATEN_OP_GROUP` category |
| `op_mode` | Which capture mode saw the op name in the capture: `eager_only` / `compile_only` / `both_modes`. Empty for subsets that declare no mode partition. Distinct from `op_group`, which sets the budget |
| `dtype` | Input dtype the cell was measured at |
| `path` | Capture path the cell came from (`eager` / `compiled`); the split key for the per-path CSVs |
| `arg_shapes` | JSON of the registry entry used. The source for any shape-derived figure — the suite no longer extracts a reduction width itself |
| `unit` | The input format's unit roundoff `2**-precision` = 1/2 ULP (bf16 → `0.00390625`). Multiply `max_rel_err_units` by this to get the raw relative error |
| `max_rel_err_units` | **The measurement.** Normwise relative error, L-inf: `(max\|T-R\| / max\|R\|) / unit` |
| `budget_units` | **The threshold.** `2.0` for arithmetic, `0.0` for bit-exact groups, empty when not gated |
| `relative_norm_verdict` | `within`, `over`, `invalid`, or empty when `status != ok` |
| `max_abs_err` / `max_abs_ref` | The two raw absolute scalars `max_rel_err_units` is computed from |
| `frobenius_abs_err_base` / `frobenius_ratio` / `frobenius_ratio_verdict` | Frobenius-ratio metric: denominator `‖cpu_low − ref_high‖`, the ratio `‖trn_low − ref_high‖ / ‖cpu_low − ref_high‖`, and its verdict (`within`/`over`/`invalid`, gate 1.1) |
| `seed` | RNG seed, derived from `cell_id` + `--seed-base` |
| `gates` | JSON of the shape-selection knobs in effect (`max_shapes_per_op`, `shape_registry`, `traced_model`, ...) |
| `status` | See below |
| `error_cause_raw` | First 200 chars of the failure, when there is one |

### Verdicts

| Verdict | Meaning |
|---------|---------|
| `within` | `max_rel_err_units <= budget_units` — the worst element is inside one ULP of the fp32 reference |
| `over` | Exceeded the budget. Includes `inf`, which is deliberate: an error against an exact-zero reference, or a bit-exact op that did not reproduce exactly |
| `invalid` | `max_rel_err_units` is `nan` — the output held a `nan`, or both sides overflowed to `inf`. Not a comparison, so not counted either way |

### Statuses

| Status | Meaning |
|--------|---------|
| `ok` | Cell measured; read `relative_norm_verdict` |
| `no_gate` | Measured, but the dtype is fp32 or finer, where a flat rounding bound is not the right instrument. Stays countable rather than passing silently |
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

# Run a single cell
python -m pytest "tests/vertical/test_dynamic_collectives.py::test_dynamic_collectives[all_gather-tp-8192x4096-bfloat16]" -v -s -n 1

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
sees; to add shapes that only one sweep picks up, put them in a smoke test subset instead (see
[Accuracy smoke test subsets](#accuracy-smoke-test-subsets)).

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

### Accuracy smoke test subsets

**Scope.** A subset is a small frozen op set for accuracy smoke tests. This is not a model
onboarding framework. Do not add a model here to get accuracy coverage for that model. The
first subset comes from a Qwen3-235B EP64/TP4 capture, You can repurpose or extend the subset,
but the intent is one small frozen subset.

The op+shape list of a subset is checked in to
`internal/neuron_shape_defaults_generated.py`, alongside the builtin defaults, so there is
one registry source rather than a second file to keep in sync. Each subset contributes four
module-level names:

| Name | Purpose |
|------|---------|
| `SMOKE_SUBSET_SHAPES[<name>]` | The register function. `<name>` is what `--traced-model` accepts |
| `SMOKE_SUBSET_OPS[<name>]` | The frozenset of ops the subset actually traced |
| `SMOKE_SUBSET_OP_MODE[<name>]` | Optional. `{op: mode}` for subsets that partition by capture mode; drives the `op_mode` CSV column |
| `<NAME>` | The source tag its entries carry, so `SHAPE_REGISTRY.get(op, model=<NAME>)` works |

`qwen3_235b_ep64_tp4` — 328 unique entries over 89 ops, from an end-to-end 64-rank capture of
`qwen3_235b_6layers_ep64_nle1_tp4_neuron_ep` (EP64 / 64 experts / top-8, TP4 × FSDP16,
SelectiveAC, seq 2048) on trn3.

The ops capture kit is on origin branch `opcap/qwen3-235b-ep64`, that branch's `tests/vertical/tools/opcap/README.md` has the procedure, the regeneration steps and the known gaps:

```sh
git fetch origin opcap/qwen3-235b-ep64
git show 16c27852:tests/vertical/tools/opcap/README.md
```

The subset is categorized on **two independent axes**.

**Axis 1 — `op_mode`, per op.** Which capture mode ever saw the op *name* at all:

| mode | meaning | ops |
|------|---------|-----|
| `eager_only` | seen in eager dispatch, never in a post-AOT graph | 33 |
| `compile_only` | post-AOT graphs only — introduced by AOT, never in model source | 10 |
| `both_modes` | seen in eager dispatch *and* in a post-AOT graph | 46 |

The full per-op attribution is in `tools/opcap/README.md` on the pinned
`opcap/qwen3-235b-ep64` commit.

**Axis 2 — capture path, per shape.** Which capture path actually observed *that exact call*,
recorded per cell during capture and carried on the entry as `OpShapeEntry.paths` (`eager` /
`compiled`, or both). This is the split key for the per-path CSVs: a cell traced on both paths is
measured once per path, and `accuracy/conftest.py` routes the results to
`op_accuracy_eager_<ts>.csv` and `op_accuracy_compiled_<ts>.csv` accordingly. A run whose entries
carry no path at all (builtin shapes, no `--traced-model`) is unsplit and writes the single
`op_accuracy_<ts>.csv`.

One constraint: a shape cannot have a path the op was never seen on, so an `eager_only` op's entries
are all eager and a `compile_only` op's are all compiled. A `both_modes` op is a mix — some shapes
were only seen in compile and some shape only in eager. The path allows us to track back where the
specific op shape is traced from.

`--max-shapes-per-op 0` is right for this set because the op shape registry is already pruned
with top-10 largest tensor cap, which leaves only 328 entries, and `aten::bmm` is declared in `QWEN3_235B_EP64_TP4_NO_SHAPE_OPS`, because the captured entries shapes is over the cap, so it
reports one `no_input_sample` in result CSV.

**Registering a subset is opt-in and off by default.** `register_all_defaults(r, smoke_subsets=())`
takes an allowlist, and `--traced-model` is the only thing that fills it. That makes the gate
fail safe: adding a subset to the generated module cannot change what any existing suite
measures until a run names it. Without the gate the new entries would join the shared
op-keyed pool and compete in the top-N-by-size cap in `NeuronOpInfo._registry_entries`,
silently displacing shapes the coverage suites measure today. An unknown `--traced-model` name
raises loudly.

Use `SMOKE_SUBSET_OPS[<name>]` — not `SHAPE_REGISTRY.all_ops(model=<name>)` — when you need the
list of ops a subset traced. `propagate_to_compatible_overloads` copies entries *and their
source tag* onto schema-compatible `.out` / `.default` overloads, so `all_ops(model=)`
reports those synthesized overloads too (more ops rather than the actually traced).

### Provenance pin

The `qwen3_235b_ep64_tp4` block in `internal/neuron_shape_defaults_generated.py` is generated output of
`tests/vertical/tools/opcap/gen_ep64_registry.py --model qwen3_235b_ep64_tp4`, which lives
on origin branch `opcap/qwen3-235b-ep64`.

```sh
# commit 7b73b5dc5eb4e7f68dfdc925701d3d528dbc747f
git fetch origin opcap/qwen3-235b-ep64
git show 7b73b5dc:tests/vertical/tools/opcap/README.md
```

sha256 (first 16) of the inputs and generators, so a re-derivation can be checked against the
block without trusting the branch pointer (paths relative to `tests/vertical/tools/opcap/`):

| sha256[:16] | file |
|---|---|
| `3b0c326890d2a731` | `captures/qwen3_235b_ep64_tp4/opcap_shapes.json` |
| `2902ae97407c846a` | `captures/qwen3_235b_ep64_tp4/opcap_ops.json` |
| `fcc04d5752454cfe` | `gen_ep64_registry.py` |
| `112d7e62bb36da88` | `gen_shape_registry.py` |

To reproduce: check that commit out, `python gen_ep64_registry.py --model qwen3_235b_ep64_tp4 -o
/tmp/block.py`, `ruff format --line-length 100 /tmp/block.py`, and diff against the block — it is
byte-for-byte identical. Nothing inside the block is hand-edited; regenerate and re-splice, then
update this pin. The capture procedure, the known gaps and the per-op mode attribution live in
that branch's `tests/vertical/tools/opcap/README.md`.

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
| `--traced-model <names>` | Comma-separated smoke test subset(s) (keys of `SMOKE_SUBSET_SHAPES`). Registers them *and* restricts the run to them. Omit it and none are registered. Stricter than `--model-shapes-only`, which only excludes `builtin` and so still mixes every registered subset together |
| `--model-shapes-only` | For ops with model-traced shapes, skip builtin/default shapes |
| `--max-shapes-per-op <n>` | Cap per op to the top-N entries by max tensor element count. `0` (or negative) disables the cap — sweep **all** shapes registered for the op |

### Programmatic usage

```python
from internal.neuron_shape_defaults_generated import (
    SMOKE_SUBSET_OPS,
    QWEN3_235B_EP64_TP4,
    register_all_defaults,
)
from internal.neuron_shape_registry import SHAPE_REGISTRY, NeuronShapeRegistry

# The process-wide registry holds builtin defaults, plus whatever --traced-model named.
SHAPE_REGISTRY.models()  # -> [] with no --traced-model; ["qwen3_235b_ep64_tp4"] with it

# A private registry with one subset registered on top of the builtin defaults.
reg = NeuronShapeRegistry()
register_all_defaults(reg.register, smoke_subsets=[QWEN3_235B_EP64_TP4])
# that subset's entries only, and the ops it traced (see the caveat above)
reg.get("aten::linear", model=QWEN3_235B_EP64_TP4)
SMOKE_SUBSET_OPS[QWEN3_235B_EP64_TP4]

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
