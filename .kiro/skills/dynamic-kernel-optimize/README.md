# Dynamic NKI Kernel Optimization — RUNBOOK

Self-contained procedure for optimizing a specific dynamic NKI kernel to close the
performance gap vs the baseline (static lowering for aten ops, native torch for collectives).

**Trigger:** "run the optimization loop for aten::<op>" (or the skill entry point).

---

## 0. Preconditions
- Trn2/Trn3 node with the Neuron toolchain.
- Full working tree at `/workspace/private-torch-neuronx`, built via `uv run ./tools/build`.
- NKI staging reference at `/workspace/private-nki-staging` (for API capabilities).
- The target op has a dynamic NKI kernel in `torch_neuronx/python_ops/torch_mlir/ops/dynamic_kernels/`.

## 1. Identify the target

Map the user's op name to files:
- **Kernel file:** `torch_neuronx/python_ops/torch_mlir/ops/dynamic_kernels/<op>_kernel.py`
  (or `dynamic_m_matmul_kernel.py` for mm, etc.)
- **Adapter file:** `torch_neuronx/python_ops/torch_mlir/ops/dynamic_kernels/adapters/<op>.py`
  (if it exists — handles shape dispatch + compile-key)
- **Registry:** `tests/vertical/internal/neuron_shape_registry.py` — shapes per op.

Op name mapping examples:
| User says | aten op | kernel file |
|-----------|---------|-------------|
| mm / matmul | aten::mm.default | dynamic_m_matmul_kernel.py |
| add | aten::add.Tensor | add_kernel.py |
| mul | aten::mul.Tensor | mul_kernel.py |
| silu | aten::silu.default | silu_kernel.py |
| embedding | aten::embedding.default | embedding_kernel.py |
| sum | aten::sum.dim_IntList | sum_kernel.py |
| gather | aten::gather.default | gather_kernel.py |
| scatter_add | aten::scatter_add_.default | scatter_add_kernel.py |

## 2. Select shapes

- If the user provided a shape: use that one.
- Otherwise: read the shape registry for this op and pick the **top-5 by input size** (most
  impactful shapes). The test selects shapes from `SHAPE_REGISTRY` capped by
  `NEURON_MAX_SHAPES_PER_OP` (default 10); pick the 5 largest.

## 3. Baseline measurement

**Resume:** If `experiments/dynamic_aten_ops/<op>/latest.csv` exists, copy it to
`baseline.csv` (if not already present) and start optimizing immediately — do NOT re-run
the benchmark. Parse `latest.csv` for the current performance state (device_ratio per shape).

**Fresh start:** If `latest.csv` does not exist in the workspace, run the measurement:
```bash
python -m pytest tests/vertical/test_dynamic_aten_ops.py -k "<op>" -v -s
```

This produces a CSV at the repo root: `dynamic_aten_ops_<timestamp>.csv` with columns:
- `op_name`, `input_shape`, `dtype`
- `dynamic_device_time_us`, `static_device_time_us`, `device_ratio`
- `correctness_test`, `profile_rel_path`

Parse the CSV. The key metric: `device_ratio = dynamic_device / static_device`.
A ratio > 1.0 means the dynamic kernel is slower; the goal is to approach 1.0.

Save the CSV as both `latest.csv` and `baseline.csv`:
```bash
mkdir -p experiments/dynamic_aten_ops/<op>
cp <newest dynamic_aten_ops_*.csv> experiments/dynamic_aten_ops/<op>/latest.csv
cp <newest dynamic_aten_ops_*.csv> experiments/dynamic_aten_ops/<op>/baseline.csv
```

After each optimization iteration, update `latest.csv` with the new measurement.
Use `baseline.csv` as the fixed reference to track total improvement across iterations.

## 4. Profile analysis

The test captures Neuron profiles for both modes. Use `neuron-explorer` or parse the NTFF:
```bash
# Profile artifacts are at the path in profile_rel_path column
/opt/aws/neuron/bin/neuron-explorer --ntff <profile_dir>/... --output-format csv
```

Compare static vs dynamic profiles:
- **Instruction count:** dynamic kernels often have more instructions (per-tile loop overhead).
- **DMA time:** look for excessive load/store cycles (bounce buffers, over-gather).
- **CC time:** collective compute should be similar; if not, tile shapes may differ.
- **Idle/stall time:** gaps between instructions indicate scheduling inefficiency.
- **Engine utilization:** DMA vs compute engine idle time.

## 5. Optimization loop (max 10 iterations)

Each iteration:
1. **Pick the worst shape** (highest device_ratio from latest measurement).
2. **Identify the bottleneck** from profile analysis (see `optimization_tips.md`).
3. **Propose ONE change** to the kernel (NKI-level optimization).
4. **Implement** in the kernel file.
5. **Rebuild:** `uv run ./tools/build`
6. **Re-measure:** run the vertical test for the target shapes.
7. **Evaluate:**
   - Correctness MUST pass.
   - device_ratio must improve beyond noise (±5%).
   - No shape should regress beyond noise.
8. **Accept or revert:**
   - Accept: leave the change, update baseline.
   - Reject: revert the kernel file, rebuild.
9. **Log** to `experiments/dynamic_aten_ops/<op>/experiment.md`.
10. **Stop check:** `touch experiments/dynamic_aten_ops/<op>/stop` to halt.

## 6. Convergence criteria

Stop early when:
- All measured shapes have device_ratio < 1.10 (within 10% of static).
- 3 consecutive iterations show no improvement.
- 10 iterations reached.
- Stop file present.

## 7. Experiment workspace

```
experiments/dynamic_aten_ops/<op>/
├── baseline.csv       # fixed reference from start of optimization (never overwritten)
├── latest.csv         # current measurement (updated after each iteration; resume anchor)
├── experiment.md      # iteration log
└── *.log              # build/run logs (optional)
```

After each measurement, copy the output CSV to `latest.csv`:
```bash
cp <newest dynamic_aten_ops_*.csv> experiments/dynamic_aten_ops/<op>/latest.csv
```

## 8. NKI API reference

For available NKI instructions, patterns, and constraints:
- `/workspace/private-nki-staging/python/nki/isa/` — ISA instruction definitions
- `/workspace/private-nki-staging/python/nki/language/` — high-level language APIs
- `/workspace/private-nki-staging/python/nki/collectives/` — collective ops
- `/workspace/private-nki-staging/docs/nisa/` — internal design docs
- `/workspace/private-nki-staging/docs/api/` — public API documentation
- Optimization tips: `optimization_tips.md` (this dir)

---
---

# §B — Dynamic Collectives Optimization

This section covers optimizing the **dynamic multi-tier NKI AllGather/AllReduce** kernels.
These differ from aten ops: they involve collective communication, multi-tier tiling tables,
and comparison against the native (torch) collective rather than a static lowering.

## B.1. Target files

- **Kernel:** `torch_neuronx/python_ops/torch_mlir/ops/dynamic_kernels/dynamic_collectives.py`
  → `dynamic_allgather_kernel` / `dynamic_allreduce_kernel` + `_run_gather_loop` / `_run_reduce_loop`
- **Tiling tables (Python):** same file — `_DEFAULT_SMALL` (tp/world≤8) and `_DEFAULT_LARGE` (fsdp/world>8)
- **Tiling tables (C++):** `torch_neuronx/csrc/distributed/DynamicAllGatherTiling.h`
  → `kDefaultSmall` / `kDefaultLarge`
- **Test:** `tests/vertical/test_dynamic_collectives.py`
- **Parity check:** `tests/distributed/dynamic_collective_ops/test_dynamic_allgather_tiling_parity.py`

## B.2. Metric

`device_ratio = dynamic_device_us / torch_device_us`

Both measured in the same session by the vertical test. A "slow row" = device_ratio > 1.10
and the row is dynamically-routed (not native-fallback).

## B.3. Two optimization levers

### Lever 1: Tiling table changes (lower risk)
Each regime = `(name, floor_gathered_bytes, (large, medium, small) gathered-byte targets)`.
Tables must satisfy:
- Tiles are powers of two, non-increasing (large ≥ medium ≥ small), nested
- Regime floors strictly descending; last floor == 0 (catch-all)
- **Both Python and C++ tables must be identical** (parity assert at lowering)
- A change requires a rebuild

**Pre-check before any build:**
```bash
python -m pytest tests/distributed/dynamic_collective_ops/test_dynamic_allgather_tiling_parity.py -q
```

**Scope of tiling changes — not limited to tuning existing regimes:**
The agent should evaluate whether the CURRENT regime structure (number of regimes, floor
boundaries, tile sizes) is optimal for the measured workload. Changes can include:
- Adjusting tile sizes within an existing regime
- **Adding new regimes** to better cover a message-size range that currently falls between
  two boundaries (e.g., a shape that gets a poor decomposition because it's just above a
  floor boundary)
- **Removing regimes** that are redundant (adjacent regimes with similar tile sizes)
- **Restructuring floor boundaries** to align with the actual message sizes in the workload
  (read the shapes from the CSV and cluster them to find natural breakpoints)
- **Splitting a regime** that serves too wide a range of message sizes with a single tile
  configuration (a 4x range within one regime may benefit from a split)

To evaluate regime restructuring: parse the CSV to extract all dynamically-routed message
sizes (gathered_bytes), group them by current regime assignment, and check whether a
different boundary placement would give each group a better tile decomposition (fewer tiles,
less over-gather, higher bus_bw utilization).

**Heuristics for tile changes:**
- **Single-tile row (numel == large tile): NO headroom.** The gap is HBM-bounce overhead;
  shrinking tiles only adds tiles → slower. Skip these.
- Low `bus_bw` + many tiles → enlarge that regime's tile targets (fewer, bigger collectives)
- `device_us >> cc_us` (overhead-bound) + multiple tiles → raise unroll or larger tile
- Big tail over-gather (remainder << small tile) → smaller small tile (but check neighbors)
- **Message sizes clustering near a floor boundary:** if many shapes are just above a floor
  and get a suboptimal decomposition, move the floor down or add a new regime above it

### Lever 2: Kernel-logic changes (higher risk, higher reward)
Modifying the kernel itself (e.g., DMA scatter pattern, buffer reuse, unroll structure).
**MUST be grounded in NKI FW source** at `/workspace/private-nki-staging`:
- `python/nki/collectives/_ops.py` — all_gather/all_reduce signatures + constraints
- `python/nki/isa/_copy.py` — dma_copy semantics, dge_mode, indirect addressing
- `lib/Frontend/Gather`, `docs/Gather2Design.md` — collective lowering design

## B.4. Procedure

**Resume:** If `experiments/dynamic_collectives/<collective>[/<reduce_op>]/latest.csv` exists,
copy it to `baseline.csv` (if not already present) and start optimizing immediately — do NOT
re-run the benchmark. Parse `latest.csv` for the current performance state.

**Fresh start:** If `latest.csv` does not exist in the workspace, run the full sweep:
```bash
# Serial — fsdp cells use all 64 cores; no pytest -n
python -m pytest tests/vertical/test_dynamic_collectives.py -k "all_gather" -v -s
```
Save the resulting CSV as both `latest.csv` and `baseline.csv`:
```bash
mkdir -p experiments/dynamic_collectives/all_gather
cp <newest dynamic_collectives_*.csv> experiments/dynamic_collectives/all_gather/latest.csv
cp <newest dynamic_collectives_*.csv> experiments/dynamic_collectives/all_gather/baseline.csv
```

After each optimization iteration, update `latest.csv` with the new measurement.
Use `baseline.csv` as the fixed reference to track total improvement.

**Optimization loop (each iteration):**
```bash
# 1. Parse latest.csv — identify slow rows (device_ratio > 1.10)
#    Attribute each slow row to (table, regime) from CSV columns.

# 2. Propose ONE change (tiling or kernel-logic)

# 3. For tiling changes: edit BOTH Python + C++ files, pre-check parity, rebuild
# For kernel changes: edit dynamic_collectives.py, rebuild
python -m pytest tests/distributed/dynamic_collective_ops/test_dynamic_allgather_tiling_parity.py -q
uv run ./tools/build

# 4. Re-measure (target config only if tiling; full sweep periodically)
python -m pytest tests/vertical/test_dynamic_collectives.py -k "all_gather and fsdp" -v -s

# 5. Accept or revert — update latest.csv on accept
# 6. Log to experiments/dynamic_collectives/<collective>[/<reduce_op>]/experiment.md
```

## B.5. Hard invariants for collectives

1. **Dynamic-loop / no-recompile contract:** The NEFF is traced at a MIN shape (`trace_numel`
   floor); the runtime `numel` scalar drives tile counts via `nl.fori_loop`, so ONE NEFF serves
   larger numels WITHOUT recompilation. No change may hardcode a runtime shape or force a
   per-numel recompile. Verify: `test_dynamic_reuse_same_bucket`.

2. **Python↔C++ table parity:** Both must be byte-for-byte identical. The C++ `ComputePlan`
   feeds the dispatch cache key + eligibility.

3. **Platform scoping:** Tiling is per-platform. If trn2 and trn3 share the `"default"` profile,
   fork a dedicated profile before editing so the other platform is frozen. Fork procedure:
   Python — add dedicated `_TRN2_SMALL`/`_TRN2_LARGE` tables, register in `_TILING_TABLES`,
   update `_profile_for_platform`; C++ — add `TilingProfile::kTrn2`, tables, `ProfileForPlatform`
   branch, `ResolveTable` case. Rebuild + run parity test (sweeps both platforms).

4. **Serial execution:** No `pytest -n` for collective tests. fsdp-16 cells consume all 64 cores.

## B.6. Experiment workspace

```
experiments/dynamic_collectives/<collective>[/<reduce_op>]/
├── baseline.csv       # fixed reference from start of optimization (never overwritten)
├── latest.csv         # current measurement (updated after each iteration; resume anchor)
├── experiment.md      # iteration log
├── .rollback/         # transient table backup (stash/revert)
└── *.log              # build/run logs
```

Where `<collective>` is the operation and `<reduce_op>` is included only for reductions:
- `experiments/dynamic_collectives/all_gather/`
- `experiments/dynamic_collectives/all_reduce/sum/`
- `experiments/dynamic_collectives/all_reduce/avg/`

On **resume**, the agent checks for `latest.csv` — if it exists, skip measurement and
start optimizing.
