# PyTorch Native Enablement

Run upstream PyTorch tests directly on Neuron — no wrappers, no JSON specs, no dynamic class generation. Tests are cloned from PyTorch, patched minimally for backend registration, and executed with pytest.

## Architecture

```
tests/pytorch_tests_native/
├── conftest.py                 # Overlay setup + skip/xfail hook + test duration
├── neuron_test_registry.py     # Single source of truth for skip/xfail
├── clone_tests.py              # Clone upstream tests (TODO: migrate)
├── torch/testing/_internal/    # Testing library overlay (cloned)
└── test/                       # Upstream PyTorch tests (cloned)
    └── distributed/
        └── _composable/fsdp/   # FSDP v2 tests
```

## Setup

Clone PyTorch tests from our pytorch fork. You might need access to this.

Then run the clone script which clones the testing module and the tests to the Eager package

```sh
git clone git@github.com:amazon-contributing/upstream-to-pytorch.git
python tests/pytorch_tests_native/clone_device_agnostic_tests.py --source-repo ../upstream-to-pytorch
```

Install the nightly and reinstall TorchNeuronEager:

```sh
# If you want to install another version replace the /cpu with what you want
# Ex: https://download.pytorch.org/whl/nightly/cu126

uv pip install -e .[dev] && \
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --reinstall-package torch
./tools/build
```

`uv pip list | grep torch` should show something like this, the version will differ `torch==2.12.0.dev202604+cu126`

## Running Tests

Ensure you have the right setup and try running one of the DDP tests

```sh
# Run pytorch test files directly
pytest tests/pytorch_tests_native/test/distributed/algorithms/ddp_comm_hooks/test_ddp_hooks.py::DistributedDataParallelCommHookTest::test_ddp_comm_hook_allreduce_hook -v -rs
```

Use `-rs` to show skip reasons in the summary. Use `-rx` to show xfail reasons.

## Persistent NEFF cache (CI)

NEFF compilation dominates the upstream suite wall-clock (a cold `test_ops`
shard recompiles ~27K NEFFs, ~50 min). The internal-compute fleet offers a
repo-scoped persistent cache for GitHub workloads, backed by S3: the runner
hydrates it to local NVMe so runs reuse compiled NEFFs instead of recompiling
from cold. It is fail-open: a missing label, missing env, or S3 error degrades
to a cold local compile and never blocks the run. The whole feature lives in
`runner/persistent_cache.py` (`PersistentNeffCache`); the runner just
instantiates it.

**How access works.** A job that carries the persistent-storage label
`neuron.internal_compute.persistent-storage:read` (or `:write`), alongside the
existing `neuron.internal_compute` label, gets the storage bucket and key-prefix
env vars injected into the pod (their names are the `STORAGE_BUCKET_ENV` /
`STORAGE_PREFIX_ENV` constants in `runner/persistent_cache.py`), pointing at this
repo's isolated cache. `read` grants read; `write` grants read + write.
`_reusable-upstream-tests.yml` adds this label automatically from the workflow inputs
(`:write` when the run uploads, else `:read`). Cache entries auto-expire after 7
days, so treat it as a reconstructible cache, not a source of truth.

`run_upstream_tests.py` flags:

| Flag | Effect |
|------|--------|
| `--download-neff-cache` | Before the run, download the NEFF cache to `/tmp/neff_cache` and enable HLO-derived NEFF keying (`TORCH_NEURONX_NEFF_KEY_FROM_HLO=1`), so NEFFs are reused whenever the lowered HLO is unchanged regardless of commit. |
| `--refresh-neff-cache` | After the run, upload the local cache back. Only effective with `--download-neff-cache`, the `:write` label, AND the `main` branch (enforced in the runner). |

Within the storage prefix the layout is keyed on compiler version and job name
(Hive-partitioned), so a `neuronx-cc` bump invalidates the cache automatically
and each shard owns its own prefix:

```
<storage-prefix>/version=<compiler>/<job_name>/neff_cache/
```

`<job_name>` is derived internally from the run's items + shard (the same stem
used for report artifacts), so it is unique per matrix node across the full
suite (no need to thread it in from the workflow). Nightly, Dependency Guard,
and Merge Queue use the same shard names. Each workflow can reuse the cache.

**Who reads vs writes.** Reads are on by default (`download-neff-cache: true`),
so the merge-queue gate and dependency-guard hydrate the seeded cache with no
extra config. Main Nightly and main Dependency Guard set
`refresh-neff-cache: true`. The runner permits uploads only from `main`.
PR and Merge Queue runs only read the shared cache.

Each job drops a `cache_metrics.json` sidecar (download/upload bytes + seconds,
NEFF count before/after, `persistent_cache_present`) that the aggregator folds
into the `cache_metrics` parquet table; `persistent_cache_present` is also
stamped onto `run_metrics`.

To bring up or verify S3 access on a small subset, dispatch
`manual-upstream-tests.yml` with `download-neff-cache: true`. Manual runs read
the cache but never upload. Main Nightly and main Dependency Guard seed it.

## Skip/Xfail Registry

`neuron_test_registry.py` is the single source of truth for all tests that need to be skipped or xfailed on Neuron. Applied automatically via `conftest.py` — no upstream patches needed.

### How it works

Each entry is a `SkipGroup` tied to one root cause. When the root cause is fixed, delete the entire group and all its tests are re-enabled.

```python
SkipGroup(
    reason="NCCL log comparison does not apply to Neuron",
    tests=["test_fully_shard_force_sum_both_reductions"],  # exact component match in nodeid
    patterns=["*nccl*"],                                     # glob against any nodeid component
    folders=["inductor/"],                                   # skip entire directory
)
```

Three granularity levels:
- **folders** — skip all tests under a directory (e.g., `"distributed/"`)
- **tests** — exact component match against the nodeid. The nodeid is split by `::` and `/` into components, and entries are matched as contiguous subsequences. This means you can target at any specificity:
  - `"test_coo"` — a single test function (across all files/classes)
  - `"TestSparse"` — an entire class (across all files)
  - `"test_sparse.py"` — an entire file
  - `"test_sparse.py::TestSparse"` — a class in a specific file
  - `"TestSparse::test_coo"` — a test in a specific class
  - `"test_sparse.py::TestSparse::test_coo"` — fully qualified
  - `"distributed::test_comm.py::TestComm"` — a class in a specific folder/file

  Partial prefixes never match: `"test_add"` won't match `"test_add_scalar"`.
- **patterns** — glob patterns matched against class and test name only (the `::` separated parts of the nodeid, excluding the file path). E.g., `"*WithLocalTensor"` matches a class, `"*nccl*"` matches a test method containing "nccl".

### Reason naming convention

Use a short, standardized label that describes the **capability gap**, not the symptom:

| Reason | What it covers |
|--------|---------------|
| `"Contiguous tensor required"` | Ops that fail because Neuron only supports contiguous tensors |
| `"Complex dtypes not supported"` | Tests using complex64/complex128 |
| `"Sparse ops not implemented"` | COO, CSR, semi-structured sparse |
| `"Float64 precision required"` | Tests that need true float64 (not just gradcheck tolerance) |
| `"FFT ops not implemented"` | torch.fft.* family |
| `"CUDA-specific feature"` | cudagraphs, NCCL, cuda streams |
| `"Op not implemented: <op_name>"` | Single missing aten op (use sparingly — group when possible) |
| `"Gradcheck float32 tolerance"` | Gradcheck failures from float64→float32 autocast |

### Rules

1. **Group by root cause, not by test file.** A `SkipGroup` for "Complex dtypes not supported" should contain tests from `test_ops.py`, `test_linalg.py`, `test_fft.py`, etc. — wherever complex tests live.
2. **One fix = one group deleted.** If fixing contiguous tensor support wouldn't also fix complex dtype tests, they belong in separate groups.
3. **Don't duplicate.** A test should appear in at most one group. The first matching group wins (skips checked before xfails).
4. **Prefer patterns over listing individual tests.** `"*_neuron_complex*"` is better than listing 200 test IDs — unless the pattern would over-match.
5. **Use xfail (not skip) when the test runs but gives wrong results.** Skip = crashes/hangs/missing feature. Xfail = runs but fails with known wrong output.

### Adding entries

Add a `SkipGroup` to `SKIP_GROUPS` (test can't run at all) or `XFAIL_GROUPS` (test runs but produces wrong results):

```python
SkipGroup(
    reason="Brief description of root cause",
    tests=["test_something", "test_another"],
)
```

## Design Principles

1. **No upstream modifications** — all customization via overlay, conftest hooks, and registry
2. **Root-cause grouping** — every skip/xfail is tied to a reason; fix the cause, delete the group
3. **Generic PrivateUse1** — solutions work for any PrivateUse1 backend, not Neuron-hardcoded
4. **100% categorized** — every test is passing, skipped-with-reason, or xfailed-with-root-cause
