# Block Perf Group Configs

## Source-Controlled Goldens

MFU goldens are checked into the group config JSON files. Benchmark runs read these values but never update them automatically.

Each active `torchrun` test has an `mfu` object keyed by hardware platform:

| Field | Purpose |
|-------|---------|
| `mfu.<platform>.threshold` | Gate value passed to `--mfu-threshold` |
| `mfu.<platform>.baseline` | Benchmark-reported MFU when the threshold was captured |
| `mfu.<platform>.baseline_date` | Capture date in `YYYY-MM-DD` format |

New thresholds use `baseline * 0.90`, providing 10% regression tolerance. Existing TRN2 thresholds are preserved as captured; some newer entries use 5% tolerance. The reusable workflow can apply an additional multiplier, such as the nightly workflow's `0.95`, without changing the checked-in golden.

MFU is derived from the median iteration time, not the mean. A rare latency outlier can inflate the mean and trip the gate on an unrelated change, while the median reflects steady-state throughput. Use the MFU value reported by the benchmark rather than recomputing it from average latency.

## How To Update

1. Run the complete group config on the target platform with `warmup_runs=20`, `benchmark_runs=20`, and `--skip-mfu-threshold`.
2. Verify that every active case passed and retain the result JSON files plus image, source commit, driver, runtime, compiler, and collectives versions.
3. Update the target platform entry for each measured test:
   - `threshold`: reported MFU multiplied by `0.90`
   - `baseline`: reported MFU
   - `baseline_date`: capture date
4. Call out the platform, image, source commit, and changed goldens in the PR description.

Do not copy a golden from another hardware platform.

## Current Baselines

### TRN2 Eager

| Test | Baseline | Threshold | Date |
|------|----------|-----------|------|
| qwen3-tp4-fsdp16 | 9.96% | 8.97% | 2026-04-29 |
| qwen3-tp1-fsdp64 | 24.14% | 21.73% | 2026-04-28 |
| qwen3-tp4 | 23.92% | 21.53% | 2026-04-25 |
| gpt-oss-tp4-fsdp16 | 6.91% | 6.22% | 2026-05-04 |
| gpt-oss-tp1-fsdp64 | 18.37% | 16.53% | 2026-04-24 |
| gpt-oss-tp4 | 15.62% | 14.06% | 2026-04-25 |

### TRN2 Compile

| Test | Baseline | Threshold | Date |
|------|----------|-----------|------|
| qwen3-tp4-fsdp16-compile | 24.60% | 22.14% | 2026-07-10 |
| qwen3-tp1-fsdp64-compile | 30.99% | 27.89% | 2026-05-29 |
| qwen3-tp4-compile | 54.30% | 51.58% | 2026-06-02 |
| gpt-oss-tp4-fsdp16-compile | 13.82% | 13.13% | 2026-06-02 |
| gpt-oss-tp1-fsdp64-compile | 22.47% | 20.22% | 2026-05-29 |
| gpt-oss-tp4-compile | 48.98% | 46.53% | 2026-06-02 |

### TRN3PDS B1 Eager

| Test | Baseline | Threshold (x0.90) | Date |
|------|----------|-------------------|------|
| qwen3-tp4-fsdp16 | 14.30% | 12.87% | 2026-08-27 |
| qwen3-tp1-fsdp64 | 32.76% | 29.48% | 2026-08-27 |
| qwen3-tp4 | 32.72% | 29.45% | 2026-08-27 |
| gpt-oss-tp4-fsdp16 | 9.23% | 8.31% | 2026-08-27 |
| gpt-oss-tp1-fsdp64 | 23.98% | 21.58% | 2026-08-27 |
| gpt-oss-tp4 | 19.95% | 17.96% | 2026-08-27 |

### TRN3PDS B1 Compile

| Test | Baseline | Threshold (x0.90) | Date |
|------|----------|-------------------|------|
| qwen3-tp4-fsdp16-compile | 32.93% | 29.64% | 2026-08-27 |
| qwen3-tp1-fsdp64-compile | 37.99% | 34.19% | 2026-08-27 |
| qwen3-tp4-compile | 71.80% | 64.62% | 2026-08-27 |
| gpt-oss-tp4-fsdp16-compile | 21.96% | 19.76% | 2026-08-27 |
| gpt-oss-tp1-fsdp64-compile | 25.44% | 22.90% | 2026-08-27 |

The B1 baselines were captured on `trn3-dev1.48xlarge` from one full `warmup_runs=20` and `benchmark_runs=20` pass. `gpt-oss-tp4-compile` remains skipped because it times out in GitHub CI, so no B1 golden is recorded for that case.

## Config Structure

Eager and compile tests for each model are co-located in one JSON file:

- `qwen3.json`: all active Qwen3 eager and compile tests
- `gpt_oss.json`: all GPT-OSS eager and compile tests, including the intentionally skipped TP4 compile case
- `infra.json`: single-core eager and compile tests plus collective benchmarks; these are correctness/performance runs without MFU gates

Tests with `exec_mode: compile` wrap the block in `torch.compile(backend="neuron")`; the first warmup run absorbs compile cost. Tests without `exec_mode`, or with `exec_mode: eager`, use standard eager execution.
