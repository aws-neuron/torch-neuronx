# Testing

The matrix spans Python pytest, C++ gtest, native upstream PyTorch tests,
HuggingFace model tests, and vertical op-coverage tests.

> **Navigation → codegraph.** For the `tests/` tree layout, the per-subsystem
> test directories, and the `tools/` driver scripts as source, ask codegraph
> (or just `ls tests/`):
> `codegraph_explore "run_upstream_tests neuron_test_registry conftest distribution_config"`.
> This doc keeps the orchestration semantics, the marker/fixture contracts,
> the xfail-registry rules, and authority routing — the parts not obvious
> from the tree.

## Driver orchestration (contract)

`tools/run-test` (sequential), `run-cpp-test` (Bazel gtest), `run-cpp-test-cmake`,
`run-block-test`, `build-and-test`. Upstream PyTorch tests and the internal CI
nodes use `tests/pytorch_tests_native/runner/run_upstream_tests.py`. Flags + report
paths are in `CONTRIBUTING.md` (authoritative). Orchestration not documented there:

- **`run_upstream_tests.py --node <id>`** runs the entries of one node of
  `tests/pytorch_tests_native/internal_test_distribution_config.json` in series,
  the same way PR Checks, the nightly, and `manual-unit-tests.yml` do. Each entry
  sets its pytest items, ignores, worker count, `forked`, and env. The
  `aten-unit-tests` entry has `"root": true`. It runs `tests/`, but not
  `tests/tmp`, `tests/huggingface`, `tests/pytorch_tests*`, `tests/vertical`, and
  the items of the other entries. Thus a new test file runs in CI with no config
  change. An entry with `"also_in_root": true` also runs in the root run, for
  example to run the same file with a different env. The C++, MLIR LIT, and op
  accuracy tests are steps of their own CI job.
- **Retry contract:** a run exiting code 1 (test failures — not timeouts or
  worker crashes) gets **one `--last-failed` retry pass**; marked passed if
  all previously-failing tests pass on retry. `NUM_WORKERS_RETRY` (default 4).
  `RETRY_TIMEOUT` (default 1800 s) is the retry pass's outer wall-clock budget,
  not a per-test timeout. A non-distributed pass keeps `TEST_TIMEOUT` as its
  per-test ceiling; a forked pass has no pytest per-test timeout. The ATen node
  runs at 64 workers, and each run has some runtime-contention failures that the
  retry clears.
- **`forked: false` (process sharing):** with `forked: true`, every test's call phase runs
  in its own forked child (`--forked`), which re-pays Neuron runtime init per
  test. An entry with `forked: false` drops that so each xdist worker initializes once and
  shares the process across its tests, and adds a per-test `--timeout`
  (`pytest-timeout` cannot be combined with `pytest-forked`, so the shared run
  is the only place a timeout can apply). See the process-isolation contract
  below for the tests this does not apply to.
- **`tools/run-block-test`** sweeps blocks with/without
  `TORCH_NEURONX_ENABLE_CONCATENATION`; `--profile`, `--metrics`.

## Pytest config contracts (root `conftest.py`)

- **Markers:** `slow` (deselect `-m "not slow"`), `multi_device`,
  `needs_process_isolation` (see below).
- **Filename conventions (load-bearing):** `_lnc1` -> needs LNC=1; `_all_cores`
  -> all cores visible. Name a test wrong and it runs in the wrong core config.
- **`ensure_neuron_initialized`** (session) sets `NEURON_RT_NUM_CORES=1` for
  non-distributed tests; **`reset_dynamo`** (function) calls
  `torch._dynamo.reset()` before each test; xdist hook assigns each worker a
  unique neuron device.

### Process isolation (`needs_process_isolation`)

`forked: false` shares one process per xdist worker. Tests that cannot tolerate that
carry the marker. An entry with `isolated_pass: true` runs them in a second `--forked`
pass at 4 workers (`ISOLATED_NUM_WORKERS` in `runner/run_upstream_tests.py`):

```python
@pytest.mark.needs_process_isolation(
    reason="Deliberately OOMs the device; the SIGABRT in the OOM-report path "
    "aborts the whole xdist worker when tests share a process."
)
def test_hard_oom_records_counters(self): ...
```

`reason` is required. **Mark sparingly**: 40 tests out of ~12,000 today, because
marked tests leave the fast path and a large set rebuilds the cost `forked: false`
removes. Mark only for a crash that kills the worker (device OOM, SIGABRT), an
assertion about process-global state ("first op in this process", "this stream is
empty"), or a setting latched at first read. Mark the whole family rather than the
single test you saw abort, since these aborts are flaky.

**Do not mark for** a flaky failure (the `--last-failed` retry covers it), a slow
test, an env var that is re-read on every use (set and restore it), or a dirty
device (use `clean_device_state`). `xdist_group` is not a lighter version of the
marker: it puts tests on one worker in sequence, but an abort still kills that
worker.

Two traps:

- **`@pytest.mark.forked` cannot do this.** `neuron_init()`
  (`torch_neuronx/csrc/core/NeuronBindings.cpp`) installs a `pthread_atfork` handler that poisons
  inherited runtime state, so boxing is only safe from a process that has not yet
  initialized the runtime. Fork a worker that has already touched the device and
  every later device op in the child raises `Neuron runtime should not be used
  after fork()`.
- **A latched setting cannot be toggled per test.** `GetStreamConfig`
  (`torch_neuronx/csrc/core/streams/StreamConfig.h`) parses `TORCH_NEURONX_ENABLE_STREAMS` and
  `TORCH_NEURONX_ENABLE_HOST_CC` once, then keeps both values for the life of
  the process. Set both flags before Python starts. Use a fresh process for each configuration.
  An environment change after initialization cannot update the runtime consistently.
  Use `requires_nrt_streams` when a test only needs a stream separate from the default
  stream. Use `requires_nrt_streams_with_host_cc` only when a test asserts HostCC
  behavior, such as which copy path runs on a non-default stream; HostCC costs kernel
  performance, so do not enable it otherwise. Apply the decorator directly to a
  collected pytest test. The setup hook sets the flags before initialization fixtures
  run. Mark the test with `needs_process_isolation`. The `nrt_stream_config` marker
  sets the flags. The decorator also verifies them. A module that applies the marker
  through `pytestmark` must assert the expected flags.

## Upstream PyTorch tests (contract)

- **Native** (`tests/pytorch_tests_native/`): runs device-agnostic upstream
  tests directly. Skip and xfail entries live in files matching
  `tests/pytorch_tests_native/skip_registry/*_registry.py`; each `SkipGroup` is
  tied to one root cause and should be deleted when fixed.
  `tests/pytorch_tests_native/neuron_test_registry.py` loads those registries
  and applies their markers. The runner is
  `tests/pytorch_tests_native/runner/run_upstream_tests.py`.
  **One retry pass** on code 1 (re-runs failures via `--last-failed`, emits
  `*_retry.json`); timeouts/internal errors surface without retry
  (`tests/pytorch_tests_native/runner/run_upstream_tests.py::should_retry`). `tests/pytorch_tests_native/runner/convert_reports.py::apply_retry_outcomes` merges retries;
  retry-only passes get `passed_on_retry` status (flakiness tracking).

## Other suites

- **Compatibility suite** (`tests/huggingface/`): excluded by default. The run
  command is in `CONTRIBUTING.md`.
- **Vertical** (`tests/vertical/`): ATen op coverage and performance use
  `NeuronOpInfo` shape registry; opt-in `--include-vertical`. Two drivers over
  the same op set - `test_target_ops_coverage.py` (eager) and
  `test_target_ops_compile_coverage.py` (compile, emits roofline columns from
  `tests/vertical/internal/roofline.py`). Per-worker NeuronCore pinning via the directory's
  `tests/vertical/conftest.py`. Register new shapes in `tests/vertical/internal/neuron_shape_registry.py`.

## Writing a new test (recipe)

- **Python op:** `tests/python_ops/`, compare `.to("neuron")`
  result against CPU via `torch.testing.assert_close`.
- **Dynamo:** `tests/neuron_dynamo_backend/unit/`,
  `tests/neuron_dynamo_backend/component/`, or
  `tests/neuron_dynamo_backend/e2e/`.
- **Distributed:** inherit `tests/distributed/utils.py`, mark `@pytest.mark.multi_device`.
- **C++:** add `*Test.cpp` under `tests/csrc/` and register it with
  `torch_neuronx_cc_test()` in `tests/csrc/BUILD.bazel`.

## Authoritative human docs (cite, don't restate)

- Run commands, flags, coverage, compatibility suites, and high-level CI
  expectations are in `CONTRIBUTING.md`. The current required-check and job
  inventory is not maintained there.
- C++ test invocation and Bazel targets are in `BAZEL_BUILD.md`.
- Repository automation and the current required-check configuration are
  maintained outside the published source tree.

## See also

- `docs/AI-build-infra.md` - build system + `tools/` depth.
- `docs/AI-csrc.md` - C++ tree the gtest suite mirrors.
- `docs/AI-dynamo-backend.md` / `docs/AI-python-ops.md` - subsystems under test.
