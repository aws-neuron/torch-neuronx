# Block Perf Group Configs

## MFU Threshold and Regression Gating

Each test entry has three MFU-related fields:

| Field | Purpose |
|-------|---------|
| `mfu_threshold` | Gate value passed directly to `--mfu-threshold` (= baseline × 0.95) |
| `mfu_baseline` | The true measured MFU when the threshold was last set |
| `mfu_baseline_date` | Date of that measurement (YYYY-MM-DD) |

`mfu_threshold` includes a 5% regression tolerance — if actual MFU drops below
this value, the test fails and blocks the PR.

## How to Update

When perf legitimately improves:

1. Measure the new MFU from a CI run
2. In the relevant JSON config, update:
   - `mfu_threshold` → `new_mfu × 0.95`
   - `mfu_baseline` → `new_mfu` (the true measured value)
   - `mfu_baseline_date` → today's date (YYYY-MM-DD)
3. Call out the change in the PR description

## Current Baselines

### Eager mode

| Test | Baseline | Threshold (×0.95) | Date |
|------|----------|------------------|------|
| qwen3-tp4-fsdp16 | 9.96% | 8.97% | 2026-04-29 |
| qwen3-tp1-fsdp64 | 24.14% | 21.73% | 2026-04-28 |
| qwen3-tp4 | 23.92% | 21.53% | 2026-04-25 |
| gpt-oss-tp4-fsdp16 | 6.91% | 6.22% | 2026-05-04 |
| gpt-oss-tp1-fsdp64 | 18.37% | 16.53% | 2026-04-24 |
| gpt-oss-tp4 | 15.62% | 14.06% | 2026-04-25 |

### Compile mode (`exec_mode: compile`)

| Test | Baseline | Threshold (×0.95) | Date |
|------|----------|------------------|------|
| qwen3-tp1-fsdp64-compile | 30.99% | 27.89% | 2026-05-29 |
| qwen3-tp4-fsdp16-compile | 23.57% | 22.39% | 2026-06-02 |
| qwen3-tp4-compile | 54.30% | 51.58% | 2026-06-02 |
| gpt-oss-tp1-fsdp64-compile | 22.47% | 20.22% | 2026-05-29 |
| gpt-oss-tp4-fsdp16-compile | 13.82% | 13.13% | 2026-06-02 |
| gpt-oss-tp4-compile | 48.98% | 46.53% | 2026-06-02 |

Baselines computed from a single full warmup_runs=20/benchmark_runs=20 pass;
threshold = `baseline × 0.95`.

TP4 compile-mode tests were previously skipped due to a non-contiguous DTensor view
error under `torch.compile` (PyTorch issue #173041). This is fixed via
`force_local_contiguous()` in the block definitions (ported from NeuronTorchTitan
`torchtitan/models/common/attention.py`).

## Config structure

Eager and compile-mode tests for each model are co-located in a single JSON:

- `qwen3.json` — all Qwen3 tests (eager + compile)
- `gpt_oss.json` — all GPT-OSS tests (eager + compile)
- `infra.json` — singlecore (eager + compile) and collective benchmarks

Tests with `exec_mode: compile` run the block wrapped in
`torch.compile(backend="neuron")`; the first warmup run absorbs compile cost.
Tests without `exec_mode` (or `exec_mode: eager`) run in standard eager mode.
