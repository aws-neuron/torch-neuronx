# Block Perf Group Configs

## MFU Threshold and Regression Gating

Each test entry has three MFU-related fields:

| Field | Purpose |
|-------|---------|
| `mfu_threshold` | Gate value passed directly to `--mfu-threshold` (= baseline × 0.9) |
| `mfu_baseline` | The true measured MFU when the threshold was last set |
| `mfu_baseline_date` | Date of that measurement (YYYY-MM-DD) |

`mfu_threshold` includes a 10% regression tolerance — if actual MFU drops below
this value, the test fails and blocks the PR. This matches the old workflow behavior
where thresholds were stored as `baseline × 0.9`.

## How to Update

When perf legitimately improves:

1. Measure the new MFU from a CI run
2. In the relevant JSON config, update:
   - `mfu_threshold` → `new_mfu × 0.9`
   - `mfu_baseline` → `new_mfu` (the true measured value)
   - `mfu_baseline_date` → today's date (YYYY-MM-DD)
3. Call out the change in the PR description

## Current Baselines

| Test | Baseline | Threshold (×0.9) | Date |
|------|----------|-----------------|------|
| qwen3-tp4-fsdp16 | 9.96% | 8.97% | 2026-04-29 |
| qwen3-tp1-fsdp64 | 24.14% | 21.73% | 2026-04-28 |
| qwen3-tp4 | 23.92% | 21.53% | 2026-04-25 |
| gpt-oss-tp4-fsdp16 | 6.91% | 6.22% | 2026-05-04 |
| gpt-oss-tp1-fsdp64 | 18.37% | 16.53% | 2026-04-24 |
| gpt-oss-tp4 | 15.62% | 14.06% | 2026-04-25 |
