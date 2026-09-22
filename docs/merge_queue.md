# Merge Queue

This repo uses a GitHub **merge queue** as the pre-merge gate for the `main`
branch. The gate runs a smoke subset of the PyTorch upstream test suite on
Neuron hardware before a PR is allowed to merge.

The implementation lives in
[`.github/workflows/merge-queue.yml`](../.github/workflows/merge-queue.yml) and
delegates the actual test run to the reusable workflow
[`.github/workflows/_reusable-upstream-tests.yml`](../.github/workflows/_reusable-upstream-tests.yml).

## Why the gate runs on two events

GitHub couples merge-queue checks and PR required checks in a way that creates a
catch-22:

- A PR must **pass its required checks** before it can be **added** to the queue.
- A check that only runs on `merge_group` never reports a status on the PR
  itself, so the PR can never enter the queue — which means `merge_group` never
  fires, so the check never runs. Deadlock.

To break this, the required status check (`Merge Queue Gate Status`) runs on
**both** events:

| Event | Behavior |
|-------|----------|
| `pull_request` (→ `main`) | A **no-op** that passes instantly. It does **not** run the smoke suite — it only lets the PR enter the queue. |
| `merge_group` (`checks_requested` → `main`) | Mirrors the **real** smoke-suite result, so it actually **gates** the merge. |

> A *skipped* required check would stay "pending" forever and re-create the
> deadlock, so the gate job is written to always **run** and explicitly
> `exit 0` on the PR event rather than being skipped.

## Run names and PR links

The workflow gives each event a distinct run name so the Actions list shows
whether a run is informational or is gating a merge:

- `pull_request`: `[IGNORE NO-OP GATE] PR #<number>: <title>` identifies the
  required no-op check that only allows the PR to enter the queue.
- `merge_group`: `Merge Queue: <queue-ref>` identifies the real queue gate and
  exposes the queued PR token directly in the run list.

For `merge_group` events, the required gate also writes a job summary containing
the queue token, commit subject, links to every PR number found in the queue ref,
and the full queue ref. The summary is available whether the gate passes or
fails, providing a direct path from a queue run to the affected PRs. Summary
generation is informational (`continue-on-error: true`) and cannot change the
gate result.

## Job graph

```
pull_request event:
  merge-queue-gate (Merge Queue Gate Status)   ← no-op, exit 0

merge_group event:
  build-tests-image (Build Main Test Image)
        ↓ main_image
  pytorch-upstream-tests (Merge Queue Gate (smoke))   ← calls _reusable-upstream-tests.yml
        ↓ gate_passed / failure_count
  merge-queue-gate (Merge Queue Gate Status)          ← REQUIRED check, gates merge
```

### `build-tests-image`
- Runs **only** on `merge_group` (`if: github.event_name == 'merge_group'`).
  Plain PRs never build the image.
- Builds / waits for the nightly test image on a self-hosted Neuron build
  runner and exposes its URI as the `main_image` output.
- Timeout: 60 minutes.

### `pytorch-upstream-tests` (the smoke run)
- Runs only on `merge_group` and only if the image build succeeded.
- Calls `_reusable-upstream-tests.yml` with:
  - `dist-config: upstream_pytorch_test_distribution_config.smoke.json` — the
    1-node smoke subset (dynamo), not the full suite.
  - `max-failures: "0"` — **zero** failing tests allowed.
  - `priority: high` — jumps the internal-compute runner queue.

### `merge-queue-gate` (the required check)
- `name: Merge Queue Gate Status` — this is the status check configured as
  **required** in branch protection.
- `if: always()` so it runs on both events and regardless of upstream result.
- On `pull_request`: prints a note and `exit 0`.
- On `merge_group`: reads the reusable workflow's outputs and:
  - `exit 0` when `gate_passed == "true"` (failing tests within the
    `max-failures` threshold);
  - `exit 1` otherwise, emitting a `::error::` annotation with the failure
    count, threshold, and rollup result.

## Merge queue configuration (branch protection settings)

These settings are configured in the **branch protection rule for `main`** on
GitHub (Settings → Branches → `main` → "Require merge queue"). They are not part
of the workflow YAML — they live in the repository settings — but they govern
how the queue batches and times out the checks above.

### Build concurrency
- **Build concurrency: `5`** — up to 5 merge-queue groups (PR batches) can have
  their checks building/running at the same time. This bounds how many
  `merge_group` runs of the smoke suite execute concurrently.

### Merge limits
Limit the number of pull requests merged into the protected branch in a single
merge operation:

| Setting | Value |
|---------|-------|
| Minimum pull requests to merge | `1` |
| Maximum pull requests to merge | `1` |
| Only merge non-failing pull requests | Enabled |

This means each merge operation merges exactly **one** PR: with a minimum of 1,
a group forms as soon as a single PR enters the queue, and with a maximum of 1,
groups never batch multiple PRs. Only non-failing PRs are merged.

### Status check timeout
- **Consider check failed after: `300` minutes** — the maximum time a required
  status check has to report a conclusion before the queue treats it as
  **failed**.

> Note this 300-minute setting is the *branch-protection* timeout for the
> required check overall. It is separate from the per-job
> `timeout-minutes` values inside the workflows (e.g. `build-tests-image`'s
> 60-minute timeout and the reusable suite's 360-minute per-node default).

## The reusable suite (`_reusable-upstream-tests.yml`)

The smoke run reuses the same workflow that nightly uses, just with
a smaller dist-config and a failure-count gate. Its internal job graph:

```
compute-pytorch-upstream-matrix (ubuntu-latest)
      ↓ matrix
pytorch-upstream-test-suite (parallel trn2 nodes)
      ↓ per-node artifacts
aggregate-pytorch-upstream (ubuntu-latest)
      ↓ failure_count / gate_passed outputs + GH job summary
```

Key inputs used by the merge queue:

| Input | Merge-queue value | Meaning |
|-------|-------------------|---------|
| `image` | output of `build-tests-image` | Docker image to test against. |
| `dist-config` | `...smoke.json` | 1-node smoke subset (dynamo) for fast pipeline verification. |
| `max-failures` | `"0"` | Aggregator exits non-zero if failing tests exceed this. Empty = no gating (nightly behavior). |
| `priority` | `high` | Internal-compute priority; only the merge queue sets this. |

Outputs consumed by the gate:

| Output | Meaning |
|--------|---------|
| `failure_count` | Number of failing tests (`failed` + `error`) the aggregator counted. |
| `gate_passed` | `"true"`/`"false"` verdict of the `max-failures` gate. |
| `max_failures` | Echoes the threshold the run was gated against. |

## How a PR flows through

1. Open a PR against `main`. The `Merge Queue Gate Status` check reports a quick
   pass on the PR head (no smoke suite runs yet).
2. Once the PR is approved and other required checks pass, add it to the merge
   queue (e.g. via the GitHub "Merge when ready" button).
3. The `merge_group` event fires: the image builds, the smoke suite runs on
   Neuron hardware, and the aggregator computes a pass/fail verdict against
   `max-failures: 0`.
4. If the gate passes, the PR merges into `main`. If it fails, the PR is removed
   from the queue and the `::error::` annotation explains why.

## Related

- [`.github/workflows/nightly.yml`](../.github/workflows/nightly.yml) — nightly
  full upstream suite.
- `tests/pytorch_tests_native/runner/` — the matrix emitter, test runner, and
  results aggregator the reusable workflow invokes.
