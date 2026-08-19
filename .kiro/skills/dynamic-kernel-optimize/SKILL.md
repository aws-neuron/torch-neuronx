---
name: dynamic-kernel-optimize
description: Run the optimization loop for a specific dynamic NKI kernel — either an aten op (exp, silu, mm, etc.) or a dynamic collective (all_gather, all_reduce). Profiles both dynamic and baseline modes, identifies bottlenecks, proposes NKI-level or tiling-level optimizations, implements them, re-measures, and iterates. Use when the user says "run the optimization loop for dynamic aten::<op>", "optimize the dynamic <op> kernel", "tune the dynamic allgather", or "optimize the dynamic all_gather tiling". Targets torch_neuronx/python_ops/torch_mlir/ops/dynamic_kernels/ kernel files. References /workspace/private-nki-staging for NKI API capabilities. Cap at 10 iterations per invocation.
---

# Dynamic NKI Kernel Optimization Loop

**Read the runbook:** `README.md` (this dir) — procedure, invariants, and tips.
**Optimization tips:** `optimization_tips.md` (this dir) — accumulated NKI optimization patterns.

## When to use
The user wants to optimize a specific dynamic NKI kernel — either an **aten op** or a
**dynamic collective**:

### Aten ops:
- "run the optimization loop for dynamic aten::exp"
- "optimize the dynamic add kernel"
- "run the optimization loop for dynamic aten::mm for input shape 256x4096"

### Dynamic collectives:
- "tune the dynamic allgather tiling"
- "optimize the dynamic all_gather"
- "run the optimization loop for dynamic all_reduce"
- "run the optimization loop for dynamic all_reduce sum"
- "optimize the dynamic all_reduce avg"

The word "dynamic" must appear in the prompt to distinguish from static kernel work.
For all_reduce, the user may specify a reduce op (sum, avg, min, max). If not specified,
default to **sum**.

## What to do (summary — details in README.md)

### For aten ops (§A in README):
1. **Identify the op:** map user's op name to the kernel file in `dynamic_kernels/`.
2. **Select shapes:** use the user-specified shape, or pick top-5 from the shape registry.
3. **Baseline:** run `tests/vertical/test_dynamic_aten_ops.py -k "<op>"` to get static+dynamic
   device times and profiles. Parse the CSV output.
4. **Analyze:** compare dynamic vs static device_time; identify the bottleneck (DMA overhead,
   compute inefficiency, tile underutilization, unnecessary copies, etc.).
5. **Propose + implement** one optimization (consulting `optimization_tips.md` and
   `/workspace/private-nki-staging` for NKI API options).
6. **Rebuild + re-measure:** `uv run ./tools/build` then re-run the test.
7. **Decide:** accept (improvement beyond noise) or revert.
8. **Log** to `experiments/dynamic_aten_ops/<op>/experiment.md`.
9. **Iterate** (max 10) or stop if converged / stop file present.

### For dynamic collectives (§B in README):
1. **Identify the collective:** all_gather or all_reduce. For all_reduce, identify the
   reduce op (sum, avg, min, max) — default to **sum** if not specified by the user.
2. **Baseline:** run `tests/vertical/test_dynamic_collectives.py -k "all_gather"` (serial, no -n).
   Parse the CSV output (`dynamic_collectives_<ts>.csv`).
3. **Analyze:** compare dynamic vs torch (native) device_time; identify slow rows
   (device_ratio > 1.10). Attribute to (table, regime) from CSV columns.
4. **Choose optimization type:**
   - **Tiling change:** adjust regime tile sizes in BOTH Python (`dynamic_collectives.py`
     `_DEFAULT_SMALL/_DEFAULT_LARGE`) and C++ (`DynamicAllGatherTiling.h`
     `kDefaultSmall/kDefaultLarge`). Pre-check parity before build.
   - **Kernel-logic change:** modify the kernel itself (ground changes in NKI FW source).
5. **Rebuild + re-measure:** `uv run ./tools/build` then re-run the test (serial).
6. **Decide:** accept or revert. Tables must stay in lock-step (Python == C++).
7. **Log** to `experiments/dynamic_collectives/<collective>[/<reduce_op>]/experiment.md`.
8. **Iterate** or stop.

## Autonomy

This skill is fully autonomous for all operations within its scope. Do NOT ask for
confirmation before:
- Reading any file in the workspace (kernel source, tests, configs, profiles, NKI staging)
- Writing/modifying kernel files, experiment logs, or CSV artifacts
- Running `uv run ./tools/build` (rebuild after kernel changes)
- Running `python -m pytest tests/vertical/...` (benchmark/profile tests, may take 5+ min)
- Running `neuron-explorer` or other profiling tools
- Creating/deleting experiment workspace directories
- Running smoke tests (`python -c "..."`) to verify correctness
- Deleting/cleaning profile artifacts (`aten_profiles/`, `dynamic_collectives_*.csv`,
  `dynamic_aten_ops_*.csv`) between measurement runs
- Copying/moving CSV files for the experiment workspace

These are all local, reversible, single-machine operations on the developer's own workspace.
The optimization loop is inherently iterative (build→measure→analyze→change→repeat) and
asking permission at each step defeats the purpose. Proceed directly.

The only operations that still require confirmation:
- `git commit` / `git push` (explicit user intent needed)
- Modifying files outside `torch_neuronx/`, `experiments/`, and `tests/`

## Hard invariants
- Always rebuild after kernel/table changes (`uv run ./tools/build`).
- Verify correctness (test must PASS) before accepting a change.
- One change per iteration; revert on regression.
- Log every iteration (accepted or rejected) with measurements.
- Reference `/workspace/private-nki-staging` for NKI API capabilities — do NOT invent APIs.
- **Dynamic collectives only:** Python and C++ tiling tables MUST stay in lock-step.
- **Dynamic collectives only:** run vertical test STRICTLY SERIAL (no pytest -n) — fsdp-16
  cells use all 64 cores.
- **Dynamic collectives only:** never break the dynamic-loop / no-recompile contract — one
  NEFF must serve larger numels without recompilation.
