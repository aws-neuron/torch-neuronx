# Testing

The matrix spans Python pytest, C++ gtest, wrapped upstream PyTorch tests,
HuggingFace model tests, and vertical op-coverage tests.

> **Navigation → codegraph.** For the `tests/` tree layout, the per-subsystem
> test directories, and the `tools/` driver scripts as source, ask codegraph
> (or just `ls tests/`):
> `codegraph_explore "run_upstream_tests neuron_test_registry conftest run-test-parallel"`.
> This doc keeps the orchestration semantics, the marker/fixture contracts,
> the xfail-registry rules, and authority routing — the parts not obvious
> from the tree.

## Driver orchestration (contract)

`tools/run-test` (sequential), `run-test-parallel` (xdist), `run-cpp-test`
(Bazel gtest), `run-cpp-test-cmake`, `run-block-test`, `build-and-test[-parallel]`.
Flags + report paths are in `CONTRIBUTING.md` (authoritative). Orchestration
not documented there:

- **`run-test-parallel`** defaults to `tests/` but **ignores** `tests/tmp`,
  `tests/huggingface`, `tests/pytorch_tests`, `tests/cpu_fallback`,
  `tests/vertical`. Sets `NEURON_LAUNCH_BLOCKING=0`. After the main suite it
  runs extra groups: distributed (fewer workers), multi-device (1 worker),
  LNC=1 all-cores, CPU fallback, host-CC multi-stream, vertical (opt-in),
  pinned malloc, streams, upstream-distributed.
- **Retry contract:** a group exiting code 1 (test failures — not timeouts or
  worker crashes) gets **one `--last-failed` retry pass**; marked passed if
  all previously-failing tests pass on retry. `NUM_WORKERS_RETRY` (default 4),
  `RETRY_TIMEOUT` (default 1800 s).
- **`--no-forked` (process sharing):** by default every test's call phase runs
  in its own forked child (`--forked`), which re-pays Neuron runtime init per
  test. `--no-forked` drops that so each xdist worker initializes once and
  shares the process across its tests, and adds a per-test `--timeout`
  (`pytest-timeout` cannot be combined with `pytest-forked`, so the shared run
  is the only place a timeout can apply). See the process-isolation contract
  below for the tests this does not apply to.
- **`run-block-test`** sweeps blocks with/without
  `TORCH_NEURONX_ENABLE_CONCATENATION`; `--profile`, `--metrics`.

## Pytest config contracts (root `conftest.py`)

- **Markers:** `slow` (deselect `-m "not slow"`), `multi_device`,
  `needs_process_isolation` (see below).
- **Filename conventions (load-bearing):** `_lnc1` → needs LNC=1; `_all_cores`
  → all cores visible. Name a test wrong and it runs in the wrong core config.
- **`ensure_neuron_initialized`** (session) sets `NEURON_RT_NUM_CORES=1` for
  non-distributed tests; **`reset_dynamo`** (function) calls
  `torch._dynamo.reset()` before each test; xdist hook assigns each worker a
  unique neuron device.

### Process isolation (`needs_process_isolation`)

`--no-forked` shares one process per xdist worker. Tests that cannot tolerate that
carry the marker and run in a second `--forked` pass at 4 workers
(`NO_FORKED_ISOLATED_WORKERS` in `tools/run-test-parallel`):

```python
@pytest.mark.needs_process_isolation(
    reason="Deliberately OOMs the device; the SIGABRT in the OOM-report path "
    "aborts the whole xdist worker when tests share a process."
)
def test_hard_oom_records_counters(self): ...
```

`reason` is required. **Mark sparingly**: 40 tests out of ~12,000 today, because
marked tests leave the fast path and a large set rebuilds the cost `--no-forked`
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
  (`csrc/core/NeuronBindings.cpp`) installs a `pthread_atfork` handler that poisons
  inherited runtime state, so boxing is only safe from a process that has not yet
  initialized the runtime. Fork a worker that has already touched the device and
  every later device op in the child raises `Neuron runtime should not be used
  after fork()`. `pytest-isolate` is the same mechanism and fails the same way.
- **A latched setting cannot be toggled per test.** `IsNonDefaultStreamWithHostCC`
  (`csrc/core/dispatch/NeuronDispatcher.cpp`) reads `TORCH_NEURONX_ENABLE_HOST_CC`
  into a `static const bool` at the first copy it dispatches, then keeps that value
  for the life of the process. An early test latches it to off; a later test sets it
  to `1`; `is_host_cc_enabled()` (`lru_cache`) and `NeuronStream.cpp` (`getenv` per
  call) now report on while the dispatcher still reports off. The test gets a real
  non-default stream while the dispatcher routes its copies down the default-stream
  path, and the worker aborts. Clearing the `lru_cache` makes this worse: it flips
  the two readers that re-read and leaves the latched one alone.

## Upstream PyTorch tests (contract)

- **Wrapper-based** (`tests/pytorch_tests/`): imports PyTorch's own test files
  via `utils.py`; `failing_test_methods/` holds xfail lists;
  `distributed/neuron_patch.py` patches NCCL→Neuron.
- **Native** (`tests/pytorch_tests_native/`): runs device-agnostic upstream
  tests directly. **`neuron_test_registry.py` is the single skip/xfail source
  of truth — each `SkipGroup` is tied to one root cause; delete the group when
  fixed.** Runner is Python (`runner/run_upstream_tests.py`; `.sh` is a shim).
  **One retry pass** on code 1 (re-runs failures via `--last-failed`, emits
  `*_retry.json`); timeouts/internal errors surface without retry
  (`should_retry`). `convert_reports.py:apply_retry_outcomes` merges retries;
  retry-only passes get `passed_on_retry` status (flakiness tracking).

## Other suites

- **HuggingFace** (`tests/huggingface/`): ~50 CausalLM models, excluded by
  default (~1h); run command in `CONTRIBUTING.md`.
- **Vertical** (`tests/vertical/`): ~1362 ATen ops for coverage/perf via
  `NeuronOpInfo` shape registry; opt-in `--include-vertical`. Two drivers over
  the same op set — `test_target_ops_coverage.py` (eager) and
  `test_target_ops_compile_coverage.py` (compile, emits roofline columns from
  `internal/roofline.py`). Per-worker NeuronCore pinning via the directory's
  `conftest.py`. Register new shapes in `internal/neuron_shape_registry.py`.

## Writing a new test (recipe)

- **Python op:** `tests/python_ops/test_<op>.py`, compare `.to("neuron")`
  result against CPU via `torch.testing.assert_close`.
- **Dynamo:** `tests/neuron_dynamo_backend/{unit,component,e2e}/`.
- **Distributed:** inherit `tests/distributed/utils.py`, mark `@pytest.mark.multi_device`.
- **C++:** `*Test.cpp` under `tests/csrc/...` + `neuron_cc_test()` in `BUILD.bazel`.

## Authoritative human docs (cite, don't restate)

- Run commands, flags, coverage, HuggingFace, CI expectations → `CONTRIBUTING.md`.
- C++ test invocation + Bazel targets → `BAZEL_BUILD.md`.
- CI workflow files live in `.github/workflows/` (`lint`, `pull`, `merge`,
  `nightly`, `_upstream-tests`, `_block-perf`).

## See also

- `AI-build-infra.md` — build system + `tools/` depth.
- `AI-csrc.md` — C++ tree the gtest suite mirrors.
- `AI-dynamo-backend.md` / `AI-python-ops.md` — subsystems under test.
