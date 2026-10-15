# Block Perf Group Configs

## Golden Model

The group config JSON files contain reconstructible MFU seed goldens. The repo-scoped persistent cache can hold a higher validated baseline for seven days, but it is never the sole source of truth: a missing, expired, corrupt, or incompatible cache object falls back to the checked-in seed for reads.

Each active `torchrun` test has an `mfu` object keyed by hardware platform:

| Field | Purpose |
|-------|---------|
| `mfu.<platform>.threshold` | Gate value passed to `--mfu-threshold` |
| `mfu.<platform>.baseline` | Reference MFU used to calculate the threshold |
| `mfu.<platform>.baseline_date` | Capture date in `YYYY-MM-DD` format |

Each test and platform uses a tolerance band from measured variance; see [PR #1755](https://github.com/aws-neuron/private-torch-neuronx/pull/1755).

FSDPv2 entries marked `host_cc: true` also have an `mfu_hostcc` map with the same fields. The default pass uses `mfu`; the HostCC pass uses `mfu_hostcc`. Each platform requires its own measured HostCC seed. A missing HostCC seed fails the gate setup.

MFU is derived from the median iteration time, not the mean. A rare latency outlier can inflate the mean and trip the gate on an unrelated change, while the median reflects steady-state throughput. Use the MFU value reported by the benchmark rather than recomputing it from average latency.

### Persistent Refresh

Every MFU-bearing block-perf group automatically hydrates persistent goldens. PR block tests and Dependency Guard have read-only access. Only the canonical `nightly.yml` run on `main` enables `refresh_persistent_goldens` for the configured full-test wheel. The shared workflows pass this default-off parameter to `_reusable-block-perf.yml` as `refresh-persistent-goldens`. That workflow checks the caller workflow and branch before it grants write access or requests a refresh. The full-test wheel refreshes TRN2 and TRN3PDS B1 tests:

- Every passing observation is appended to a rolling five-value history. The median becomes the candidate baseline, rounded to two decimals.
- The effective baseline is the greater of the current baseline and that candidate, so transient fast nodes cannot set the bar alone and sustained regressions cannot ratchet it downward.
- The object is rewritten even when the monotone floor retains the baseline. This records the observation and refreshes the seven-day TTL without hiding a regression.
- A failed test keeps its hydrated cache entry. Passing siblings can still update.
- If no MFU test passes, the group does not write.
- A writer on any branch other than `main` does not write.
- If an existing object cannot be downloaded or validated safely, readers use the checked-in seed and refuse to overwrite the cache. The run emits a warning but remains passing; if the object's seven-day TTL lapses, a later nightly safely recreates it from the seed.

A test that misses its MFU gate runs one more time. A passing retry counts as a pass for the group and the golden refresh. A retry can still fail.

Each cache object is scoped by platform and group under `block-perf-goldens/v1/`. Each test has a seed digest for its active benchmark configuration and checked-in platform golden. A changed test falls back to its seed without invalidating cached siblings. Payload schema v3 stores per-test digests and the rolling history. Older or malformed payloads are rejected rather than migrated.

HostCC objects use `block-perf-goldens/v1/hostcc/<platform>/<group>.json`. Default objects retain their existing paths. Both modes use the same refresh rules and seven-day TTL. Changes to a HostCC seed do not invalidate the default seed digest, or vice versa. Only the canonical nightly can refresh either mode.

## Updating Checked-In Seeds

1. Run the complete group config on the target platform with `warmup_runs=20`, `benchmark_runs=20`, and `--skip-mfu-threshold`.
2. Verify that every active case passed and retain the result JSON files plus image, source commit, driver, runtime, compiler, and collectives versions.
3. Update the target platform entry for each measured test:
   - `threshold`: reported MFU multiplied by the selected per-test tolerance band
   - `baseline`: reported MFU
   - `baseline_date`: capture date
4. Call out the platform, image, source commit, and changed goldens in the PR description.

Do not copy a golden from another hardware platform. Checked-in updates remain necessary when adding or materially changing a benchmark, intentionally resetting a baseline, or recovering after the persistent object expires.

For HostCC seeds, set `TORCH_NEURONX_ENABLE_STREAMS=1` and `TORCH_NEURONX_ENABLE_HOST_CC=1` before the capture command. Update `mfu_hostcc` from that platform's results. The explicit `--skip-mfu-threshold` capture option works before a seed exists. It cannot be combined with a persistent refresh.

The HostCC seeds were refreshed on 2026-10-05 (local date; execution on October 6 UTC) with 20 warmup and 20 measured iterations. All eight FSDPv2 cases passed on each platform, with separate Trn2 and Trn3PDS B1 measurements. The captures used main at `6fba17f4`, driver `2.x.12350.0`, runtime `2.x.83730.0-003bd74b0`, and collectives `2.x.82713.0-37113ea42`. Compiler `2.0.410046.0a0+18a8f51e` and NKI `0.7.0b1+36611804198.gd1177c75` matched the current CI build.

The driver version above came from `/sys/module/neuron/version` on each host. The capture script asserted that value before the benchmarks. It is the loaded host driver, not the DKMS package version inside the image.

Each seed retains the existing test/platform tolerance band. The default seeds remain unchanged.

## Current Baselines

On 2026-09-23, three Qwen3 seeds were reset: TRN2 TP1/FSDP64 compile, and TRN3PDS B1 TP1/FSDP64 eager and compile. Each new seed is the median MFU from five successful Python 3.14 stable nightly group runs, ending on 2026-09-23. Platform samples remain separate. This reset accepts historical performance losses. Other seeds remain unchanged. Future nightly refreshes retain the monotone baseline rule.

Source nightly runs: [35688374722](https://github.com/aws-neuron/private-torch-neuronx/actions/runs/35688374722), [35693726911](https://github.com/aws-neuron/private-torch-neuronx/actions/runs/35693726911), [35725519738](https://github.com/aws-neuron/private-torch-neuronx/actions/runs/35725519738), [35765240149](https://github.com/aws-neuron/private-torch-neuronx/actions/runs/35765240149), and [35801338296](https://github.com/aws-neuron/private-torch-neuronx/actions/runs/35801338296).

Four TRN3PDS B1 tolerance bands widen from 4% to 6%: GPT-OSS TP1/FSDP64 eager, GPT-OSS TP4/FSDP16 eager, Qwen3 TP4 compile, and Qwen3 TP4/FSDP16 compile. These bands provide at least 2% observed headroom against the current CI median references, including the initial partial history windows. The current sample has only five or six nightly observations per test. This margin does not guarantee future passes or assume retry recovery. Every other band remains unchanged. No checked-in baseline or threshold increases in this reset.

### TRN2 Eager

| Test | Baseline | MFU Gate | Date |
|------|----------|----------|------|
| qwen3-tp4-fsdp16 | 9.96% | 9.06% | 2026-04-29 |
| qwen3-tp1-fsdp64 | 24.14% | 23.17% | 2026-04-28 |
| qwen3-tp4 | 23.92% | 22.96% | 2026-04-25 |
| gpt-oss-tp4-fsdp16 | 6.91% | 6.29% | 2026-05-04 |
| gpt-oss-tp1-fsdp64 | 18.37% | 17.27% | 2026-04-24 |
| gpt-oss-tp4 | 15.62% | 14.68% | 2026-04-25 |

### TRN2 Compile

| Test | Baseline | MFU Gate | Date |
|------|----------|----------|------|
| qwen3-tp4-fsdp16-compile | 24.60% | 23.62% | 2026-07-10 |
| qwen3-tp1-fsdp64-compile | 29.47% | 27.70% | 2026-09-23 |
| qwen3-tp4-compile | 54.30% | 52.13% | 2026-06-02 |
| gpt-oss-tp4-fsdp16-compile | 13.82% | 13.27% | 2026-06-02 |
| gpt-oss-tp1-fsdp64-compile | 22.47% | 21.57% | 2026-05-29 |
| gpt-oss-tp4-compile | 48.98% | 46.04% | 2026-06-02 |

### TRN3PDS B1 Eager

| Test | Baseline | MFU Gate | Date |
|------|----------|----------|------|
| qwen3-tp4-fsdp16 | 14.30% | 13.73% | 2026-08-27 |
| qwen3-tp1-fsdp64 | 31.89% | 30.61% | 2026-09-23 |
| qwen3-tp4 | 32.72% | 29.78% | 2026-08-27 |
| gpt-oss-tp4-fsdp16 | 9.23% | 8.68% | 2026-08-27 |
| gpt-oss-tp1-fsdp64 | 23.98% | 22.54% | 2026-08-27 |
| gpt-oss-tp4 | 19.95% | 18.75% | 2026-08-27 |

### TRN3PDS B1 Compile

| Test | Baseline | MFU Gate | Date |
|------|----------|----------|------|
| qwen3-tp4-fsdp16-compile | 32.93% | 30.95% | 2026-08-27 |
| qwen3-tp1-fsdp64-compile | 36.04% | 34.60% | 2026-09-23 |
| qwen3-tp4-compile | 71.80% | 67.49% | 2026-08-27 |
| gpt-oss-tp4-fsdp16-compile | 21.96% | 21.08% | 2026-08-27 |
| gpt-oss-tp1-fsdp64-compile | 25.44% | 24.42% | 2026-08-27 |

`gpt-oss-tp4-compile` remains skipped because it times out in GitHub CI, so no B1 golden is recorded for that case.

## Config Structure

Eager and compile tests for each model are co-located in one JSON file:

- `qwen3.json`: all active Qwen3 eager and compile tests
- `gpt_oss.json`: all GPT-OSS eager and compile tests, including the intentionally skipped TP4 compile case
- `infra.json`: single-core eager and compile tests plus collective benchmarks; these are correctness/performance runs without MFU gates

Tests with `exec_mode: compile` wrap the block in `torch.compile(backend="neuron")`; the first warmup run absorbs compile cost. Tests without `exec_mode`, or with `exec_mode: eager`, use standard eager execution.

The reusable block workflow runs model groups at streams=1/HostCC=0, then repeats entries marked `host_cc: true` at streams=1/HostCC=1 on the same node. Those entries use FSDPv2. Both passes run for every caller, including pull requests, nightly, dependency guard, and SDK release. The infra group runs once at streams=1/HostCC=0 because it has no FSDPv2 entries. CI sets the baseline flags explicitly; the package default is unchanged by this PR.

Each mode has separate output and compiler-cache directories. Cache paths, including persistent cache URLs, have a `streams` or `hostcc` suffix. With MFU enforcement enabled, both passes check their own goldens. Only the canonical nightly on main can refresh goldens. Other callers have read-only access.

The workflow passes `--both-modes --hostcc-advisory` for model groups. A HostCC failure remains visible but does not fail the group. Baseline failures, interrupted children, launch errors, and artifact-copy failures remain strict. Without `--hostcc-advisory`, failures from either pass fail the group. The advisory option requires `--both-modes`. Each pass prints its status. The runner writes `block-perf-status.json` under the output directory after each child exits. This receipt records each mode's status, exit code, and whether its failure is advisory. A parent `KeyboardInterrupt` also writes the receipt before it propagates. Unstarted modes have status `not-run`.
