# torch-neuronx CR Review Checklist

This checklist covers areas CI cannot check. **Skip** items that lint,
ruff, clang-format, pre-commit, `lint.yml`, or `pull.yml` already
enforce — formatting, import order, trailing whitespace, type-ignore
syntax, bare file boilerplate.

Every item below is either a pattern violation, a correctness hole,
or a test-adequacy issue specific to `torch-neuronx`. Everything worth
mentioning is worth fixing — there are no nits.

---

## Code Quality

### Abstractions and design

- [ ] **Clear abstractions** — State management is explicit. Module-
      or instance-level attributes are set in `__init__` / class body,
      not dynamically via `setattr`/`getattr` at arbitrary call sites.
- [ ] **No side-channel communication** — A behavior change triggered
      by a hidden flag set in one file and checked via `getattr` in
      another is almost always wrong. Use a different function, a
      different class, or a different code path.
- [ ] **Proper interface, not on/off flags** — A private boolean that
      switches between two fundamentally different behaviors should be
      two separate code paths or a proper interface, not a flag.
- [ ] **Interface documentation** — New internal contracts between
      Python and C++, or between `Operation` → `OperationImplementation`
      → kernel, must have concrete documentation: what the caller
      provides, what the callee receives, what invariants hold, what
      the cleanup responsibility is. "This allows X" is motivation,
      not interface.
- [ ] **Match existing patterns in the same file** — Before accepting
      new code, read how similar features are already implemented in
      that file. If the file uses `@register_aten` and
      `TorchMlirOpImpl`, new ops in that file must use the same. If
      the file uses `Operation` + `OperationImplementation` + priority
      ordering, new impls must use the same.
- [ ] **No over-engineering** — Only requested changes are made; no
      speculative features.
- [ ] **No premature abstraction** — Helpers are only created when
      reused. Three similar lines is better than a one-use helper.
- [ ] **No trivial helpers** — 1–2 LOC helper functions used once are
      noise unless they meaningfully improve readability.

### API design

When the CR introduces a new API pattern, evaluate broader implications.

- [ ] **No flag-gated internal access** — Reject `_internal=True`
      kwargs and similar. Use a separate private function
      (`_my_internal_op()`) instead.
- [ ] **Pattern already exists?** — Before accepting a new pattern
      (decorator, fixture, registration convention), search the
      codebase for established ones. If this pattern is new, the CR
      is introducing a convention that needs strong justification.
- [ ] **Discoverable documentation** — If the new API is public, is it
      in `docs/`? Can it be documented without ambiguity?
- [ ] **BC implications going forward** — Does this pattern lock us
      into future constraints?
- [ ] **Testing implications** — Does this pattern require awkward
      test patterns (e.g., tests that set "forbidden" internal
      kwargs)?
- [ ] **UX implications** — Does this pattern surface cleanly in
      autocomplete, type hints, and docs? Or will users stumble over
      it?

### Public API documentation

- [ ] **Public symbols are re-exported** — Any callable intended to be
      public must be imported in `torch_neuronx/__init__.py` (or a
      documented submodule's `__init__.py`). A symbol not re-exported
      at `torch_neuronx.foo` is not public regardless of how
      non-underscored its name is.
- [ ] **`torch.neuron.*` surface** — New APIs registered via
      `torch._register_device_module("neuron", ...)` go under
      `torch.neuron.*` and are user-visible. Document in
      `docs/` and mention in the CR description.
- [ ] **New env vars documented** — Any new `TORCH_NEURONX_*` /
      `NEURON_*` env var must be added to `docs/environment_variables.md`
      with a description and default. CR should not merge without the
      doc update.

### Code clarity

- [ ] **Self-explanatory code** — Variable and function names convey
      intent; minimal comments needed.
- [ ] **Useful comments only** — Comments explain non-obvious context
      that cannot be inferred locally. Large comments use a
      `# Note [Title]` / `See Note [Title]` pattern (see
      `torch_neuronx/python_ops/base.py` for examples).
- [ ] **No BC-compat hacks left behind** — Unused code is deleted
      completely, not renamed with underscores or commented out.
- [ ] **Appropriate complexity** — Solutions are as simple as
      possible for current requirements.
- [ ] **Docs show correct patterns only** — Example code in `docs/`
      or `README.md` must compile and must reflect the recommended
      path (e.g., `backend="neuron"` with `dynamic=False`, not a
      counter-example followed by a correction).

### Initialization and module design

- [ ] **No fragile init ordering** — If multiple imports/calls must
      happen in a specific undocumented order, flag it. In this repo,
      `register_python_operations()` order matters
      (`_register_common_ops()` before `register_mlir_ops()` before
      `register_cpu_fallback_ops()`). New registrations must slot into
      the right phase, not be appended arbitrarily.
- [ ] **Idempotent registration** — `@neuron_op`, `@register_aten`,
      `aten_lib.impl(...)`, `_C._register_neuron_op(...)`,
      `torch.library.impl(...)` calls must be safe under repeated
      import. Duplicate registration must either be a no-op or raise
      with a clear message — never produce two live handlers for the
      same key.

---

## torch-neuronx Infrastructure

When a CR touches code in the scope of any item below, **stop and
investigate** whether the established helper/pattern should be used.
Cite the overview doc rather than re-deriving.

### Eager-mode op registration (`torch_neuronx/python_ops/`)

See `docs/AI-python-ops.md`.

- [ ] **`@neuron_op` / `@register_aten` instead of manual
      `aten_lib.impl`** — New ATen-bound ops use the decorators, not
      raw `torch.library.impl(...)` plumbing. The decorators handle
      priority, auto-wrapping for Dynamo guards, and registration
      timing.
- [ ] **Correct `Operation` / `OperationImplementation` contract** —
      New impls subclass `OperationImplementation` (or a typed
      variant: `UnaryOperation`, `BinaryOperation`,
      `ComparisonOperation`, `ReductionOperation`, `CompilableOpImpl`)
      and override `_execute_impl`, not `execute`. Overriding
      `execute` bypasses dtype autocast, empty shortcircuit, and
      output handling.
- [ ] **`can_handle()` is cheap and pure** — It runs on every
      dispatch; heavy work belongs in `_execute_impl`. No side
      effects, no tensor allocation.
- [ ] **Priority ordering justified** — A new impl's `priority` is
      higher than the MLIR fallback only if it can actually handle
      every shape/dtype the fallback can. Otherwise, dispatch will
      pick it, fail, and fall through — slower than not registering
      it at all.
- [ ] **`DISABLE_DTYPE_AUTOCAST` set when needed** — Ops whose
      semantics require exact input dtypes (e.g., index ops taking
      int64) must set `DISABLE_DTYPE_AUTOCAST = True`. The default
      autocast will silently truncate int64 → int32.
- [ ] **`DISABLE_EMPTY_SHORTCIRCUIT` set when needed** — Ops whose
      correctness depends on running even for 0-element tensors (side
      effects on `out=`, shape inference for downstream ops) must set
      the flag.
- [ ] **MLIR op files are one-per-family** — New `@register_aten`
      functions go into the existing file for their op family
      (`binary.py`, `reduction.py`, …), not a brand-new file unless
      the family is genuinely new.

### Cast policy and contiguity (`torch_neuronx/python_ops/cast_policy.py`)

Violations here cause silent wrong-numerics or infinite dispatcher
recursion.

- [ ] **64-bit handling uses `bounce_64bit`** — Any transfer path
      touching float64/int64 uses `bounce_64bit(dtype)` to decide,
      not ad-hoc `dtype in (torch.int64, torch.float64)` checks.
- [ ] **On-device contiguity uses `ensure_contiguous_on_device`** —
      For Neuron tensors, never call `tensor.contiguous()` directly
      from an op impl; it re-enters the dispatcher and loops. Use
      `ensure_contiguous_on_device()`.
- [ ] **Raw copies use the helpers** — New Py↔device transfers use
      `copy_cpu_to_neuron`, `copy_neuron_to_cpu`,
      `write_neuron_to_neuron` from `cast_policy.py`, not raw
      `.to('neuron')` / `.to('cpu')` loops.
- [ ] **Memory format validated** — `validate_neuron_memory_format`
      is called (or the op legitimately supports only contiguous, in
      which case it must reject non-contiguous inputs with a clear
      error, not silently corrupt).
- [ ] **Autocast warnings centralized** — Dtype-cast warnings go
      through `_warn_dtype_cast_once` (one per site), not ad-hoc
      `warnings.warn` calls that spam.

### NKI integration

See `docs/AI-python-ops.md` ("NKI backend").

- [ ] **NKI kernels wrapped via `NKITorchMlirOpImpl`** — Integration
      with the dispatcher goes through `NKITorchMlirOpImpl` to share
      the MLIR cache + execute path. Ad-hoc `nki.jit(...)` wired
      directly into a custom `_execute_impl` bypasses caching and
      will recompile on every call.
- [ ] **Cache key covers `backend_config` and `constant_args`** —
      `NKITorchMlirOpImpl`'s custom cache key hashes kernel identity
      + `backend_config` + `constant_args`. New NKI kernels whose
      output depends on a config knob must add that knob to the
      hashed set, or two calls with different configs will share a
      NEFF.
- [ ] **Public NKI wrapper uses `@nki_op`** — User-facing NKI ops
      use the `@nki_op("namespace::name", mutates_args=...)`
      decorator from `torch_neuronx`. The `mutates_args` set must
      list every output argument the kernel writes into, or
      Functionalization + autograd will mis-track aliasing.
- [ ] **Autograd backward provided where needed** — `@nki_op` with
      a trainable forward should declare a backward. Missing backward
      makes the op silently non-differentiable under
      `torch.compile` + autograd.

### C++ extension (`torch_neuronx/csrc/`)

See `docs/AI-csrc.md`.

- [ ] **Python-registered op also registered via
      `_C._register_neuron_op`** — Hot-path ops (gather, scatter_add,
      SDPA, nonzero, index) must have their C++ dispatch hook; the
      Python-only registration is a slow fallback. If the CR adds an
      op to one side, check whether the other side needs updating.
- [ ] **`NeuronDispatcher` lock discipline** — New paths that acquire
      the dispatcher's internal mutex must release it before calling
      into Python to avoid deadlocks during autograd.
- [ ] **RAII over manual resource management** — NRT handles,
      allocator scopes, profiler scopes: use RAII guards (see
      existing patterns in `csrc/core/runtime/`), not manual
      open/close pairs inside try/catch.
- [ ] **`NEURON_LAUNCH_BLOCKING` respected** — New async paths must
      honor the env var (force synchronous when set). Debugging
      depends on it.
- [ ] **GIL handling across Py↔C++ boundary** — Any C++ code that
      touches `PyObject*` holds the GIL. Long-running C++ work
      releases the GIL via `py::gil_scoped_release`. Borrowed
      references do not survive a GIL release.
- [ ] **`torch_neuronx::_C` module additions have pybind signatures
      with defaults** — Adding a required-without-default argument
      to an existing `_C` binding is a BC break for anyone pinning
      an older Python wheel.

### torch.compile backend (`torch_neuronx/neuron_dynamo_backend/`)

See `docs/AI-dynamo-backend.md`.

- [ ] **Supports only `mode="default"`, `dynamic=False`, contiguous
      inputs** — New paths don't silently enable an unsupported mode
      or accept dynamic shapes. The README is the public contract —
      any deviation is a BC-visible change.
- [ ] **FX passes are composable and local** — New graph transforms
      live under `neuron_dynamo_backend/fx/passes/` and operate on a
      single graph, not globally. They do not mutate the input graph
      module; they return a new one.
- [ ] **Decomposition table additions are guarded** — Entries added
      to the decomposition table must be op-specific, not
      global-catchall. A global entry affects every compile call.
- [ ] **NKI kernels in the compiled graph go through the integrated
      path** — `@nki_op`-wrapped ops must be preserved by the
      backend (no graph breaks at the nki_op boundary). New code
      that introduces a graph break at an nki_op should be flagged.
- [ ] **Cache integration** — If the CR changes a compile-time
      input to the graph (new decomposition, new pass, new env flag
      that alters lowering), the NEFF cache key must include it.
      See `docs/AI-compilation-kernels.md` for cache-key schema.
- [ ] **Vocab sharding / flex attention / caching_precompile
      interactions** — These are cross-cutting; touching any of them
      requires a test under
      `tests/neuron_dynamo_backend/component/`.

### NEFF cache & compilation (`torch_neuronx/python_ops/compilation/`, `csrc/core/compilation/`, `csrc/core/cache/`)

See `docs/AI-compilation-kernels.md`.

- [ ] **Cache key covers all inputs that affect output** — Shape,
      dtype, memory layout, NKI `backend_config`, `constant_args`,
      decomp-table identity, relevant env flags. A missed input is a
      silent wrong-result bug for any user with the stale key in
      their `TORCH_NEURONX_DEBUG_DIR`.
- [ ] **Cache-key schema changes are versioned** — If the CR changes
      how the key is computed, bump the schema version, or existing
      users will serve stale NEFFs. Note the bump in the CR.
- [ ] **`neuronx-cc` flag additions documented** — If new `NEURON_CC_FLAGS`
      or `CompilerSubprocess` args are added, document in
      `docs/environment_variables.md`.
- [ ] **Artifact writes respect `TORCH_NEURONX_DEBUG_DIR` and
      `TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS`** — New
      compilation paths write their artifacts under the configured
      dir and are deleted unless preservation is on. No hardcoded
      paths.

### Distributed (`torch_neuronx/distributed/`, `csrc/core/distributed/`)

See `docs/AI-distributed.md`.

- [ ] **`ProcessGroupNeuron` contract preserved** — New collectives
      register in the PG and follow the existing `NeuronWork` /
      `Watchdog` pattern. Do not add a parallel dispatch path.
- [ ] **DeviceMesh / DTensor compatibility** — New collectives must
      be reachable from DTensor sharding; add a test under
      `tests/distributed/dtensor/`.
- [ ] **Bucketing and premul-sum** — Any addition to `all_reduce`
      composes correctly with bucketing and premul-sum.
- [ ] **Test is multi-device-aware** — Marked with
      `@pytest.mark.multi_device`; placed under
      `tests/distributed/`; inherits from `tests/distributed/utils.py`.

### Python helpers (`torch_neuronx/__init__.py` and top-level module)

See `docs/AI-runtime-api.md`.

- [ ] **Entry points untouched unintentionally** — `torch.backends` and
      `torch_dynamo_backends` entry points in `pyproject.toml` are
      load-bearing. Any changes must be intentional and tested.
- [ ] **`_autoload` stays idempotent** — Importing `torch_neuronx`
      twice (directly + via an entry point) must be safe.
- [ ] **New env var defaults set in `__init__.py`** — Defaults live
      alongside `NEURON_LAUNCH_BLOCKING`, `TORCH_NEURONX_ENABLE_LAZY_ALLOC`.
      Setting a default inside an op file is a design smell.
- [ ] **`pytree` registration for custom containers** — Custom
      tensor-containing containers register with
      `torch.utils._pytree.register_pytree_node`.

### Utilities and logging

- [ ] **No `print()` in production paths** — Debugging uses
      `torch._logging` loggers / `torch_neuronx.log` (check
      `torch_neuronx/statistics/` for the pattern). `print()` in
      hot code floods user tests.
- [ ] **`torch._check` in op impls** — Python-level shape / dtype
      preconditions use `torch._check()` / `torch._check_is_size()`
      so meta / symbolic-shape tracing works.
- [ ] **No ad-hoc `warnings.warn`** — Use `_warn_dtype_cast_once`
      for dtype casts; the project's structured logging for anything
      else.

---

## Testing

See `docs/AI-testing.md`.

### Test existence

- [ ] **Tests exist** — New functionality has corresponding tests.
- [ ] **Regression tests for bug fixes** — A bug fix includes a test
      that fails without the fix and passes with it. If the bug was
      found via an upstream PyTorch test, the regression test is
      **removing that test's entry from
      `tests/pytorch_tests_native/neuron_test_registry.py`**, not a
      new one-off unit test.
- [ ] **xfail registry cleanup** — When a CR fixes a previously-xfailed
      upstream test, delete (or shrink) its `SkipGroup` in the
      registry. Flipping inline `@pytest.mark.xfail` decorators is
      not the right move — the project's single source of truth is
      the registry.
- [ ] **Tests are in the right place** — Tests go next to related
      tests. A new op goes in `tests/python_ops/test_<op>.py`. A new
      dynamo pass goes in `tests/neuron_dynamo_backend/unit/` or
      `.../component/`. A new collective goes in
      `tests/distributed/collective_ops/`.
- [ ] **New test file is rare** — Only justified when a genuinely
      new feature is introduced.

### Test patterns

- [ ] **Use `torch.testing.assert_close` for tensors** — Not
      `torch.allclose()`, not `==`, not bare `assert`.
- [ ] **Inputs built on CPU, moved to Neuron explicitly** —
      `x_cpu = torch.randn(...); x = x_cpu.to("neuron")`. Don't
      build directly on `device="neuron"` for correctness tests
      unless that is exactly what you're testing — doing so hides
      transfer-path bugs.
- [ ] **CPU reference for correctness** — Correctness tests compare
      Neuron output to the CPU implementation of the same op, not to
      a hand-written expected value, so the test tracks upstream
      behavior automatically.
- [ ] **No `time.time()` in tests** — Any perf measurement in tests
      uses `torch_neuronx` profiler or the repo's block-perf tools,
      not wall clock.
- [ ] **Device reset between tests if needed** — If a test changes
      global Neuron state (NRT cores, cache), it restores state in a
      fixture.
- [ ] **`reset_dynamo` fixture respected** — Dynamo tests assume the
      session-level `reset_dynamo` fixture from `conftest.py` clears
      state. Don't re-implement it inline.
- [ ] **`@pytest.mark.multi_device` used correctly** — Tests that
      need multiple Neuron cores carry the marker. Tests in
      `tests/distributed/` inherit it from the dir's `conftest.py`.
- [ ] **`_lnc1` / `_all_cores` filename conventions** — Tests with
      those suffixes are auto-configured by `conftest.py`. Don't
      duplicate that logic inside the test.
- [ ] **C++ tests register in `BUILD.bazel`** — New `*Test.cpp`
      under `tests/csrc/` must be added to `tests/csrc/BUILD.bazel`
      via `neuron_cc_test()` AND to `tests/csrc/CMakeLists.txt`.
      Missing either half is a build break in CI.
- [ ] **Upstream-test overlay not forked** — Do not modify upstream
      tests under `tests/pytorch_tests_native/`. If a test must
      behave differently on Neuron, the fix belongs in
      `torch_neuronx` source, not a test patch.

### Test quality

- [ ] **Edge cases covered** — Empty tensors, 0-dim, broadcasting
      edges, boundary dtypes (float16/bfloat16 in particular),
      non-contiguous inputs (when the op claims to support them).
- [ ] **Error conditions tested with `pytest.raises` + `match=`** —
      Bare `pytest.raises(SomeError)` without `match=` lets an
      unrelated exception pass the test.
- [ ] **No duplicated test logic** — Parametrize with
      `@pytest.mark.parametrize` instead of copy-pasting per dtype /
      per shape.
- [ ] **Skip-reasons are specific** — `pytest.skip("not yet
      supported on Neuron")` is too vague. Point to the tracking
      issue or the registry SkipGroup.

---

## Security

The threat model for `torch-neuronx` is narrower than upstream — we
don't serve arbitrary untrusted models over the network — but a few
patterns still apply.

- [ ] **No `torch.load(..., weights_only=False)`** — Default is
      `weights_only=True`; explicitly disabling it enables arbitrary
      code execution via pickle. Flag any new call that does.
- [ ] **No `pickle.load` / `pickle.loads` on caller-provided paths**
      — NEFF cache deserialization uses versioned schemas, not
      pickle. Any new pickle use in the cache path is a security
      regression.
- [ ] **No credentials or internal URLs in logs / errors / cache
      artifacts** — `TORCH_NEURONX_DEBUG_DIR` is written to users'
      disks; it must never contain auth material.
- [ ] **Subprocess invocations are argv-list, not shell** —
      `CompilerSubprocess` and any new `subprocess.Popen` / `run`
      pass a list, not `shell=True` with an interpolated string.
- [ ] **Environment variable names are not controlled by untrusted
      input** — `os.environ[user_string]` is a red flag.

---

## Thread safety & concurrency

### Python

- [ ] **No unprotected shared mutable state** — Caches (NEFF cache,
      HLO cache, `_NEURON_OPS_REGISTRY`) must be accessed under a
      lock or be inherently thread-safe.
- [ ] **Registration is single-threaded** — `@neuron_op`,
      `@register_aten`, and `aten_lib.impl(...)` happen at import
      time, which is GIL-serialized. Lazy/deferred registration
      from a worker thread must add its own lock.
- [ ] **Lock ordering** — When multiple locks are acquired
      (dispatcher lock + cache lock, for example), the order is the
      same everywhere.

### C++ / Py↔C++

- [ ] **No data races** — Shared mutable state is protected by
      mutexes or atomics with correct memory ordering.
- [ ] **RAII lock guards** — Prefer `std::lock_guard` /
      `std::unique_lock` over manual `lock()`/`unlock()`.
- [ ] **GIL held for `PyObject*` access** — Any code touching
      `PyObject*` (incref, decref, attribute access) holds the GIL.
      Code released via `py::gil_scoped_release` does not touch
      Python objects.
- [ ] **Borrowed references don't survive GIL release** — After
      `py::gil_scoped_release`, any `PyObject*` fetched via
      `PyTuple_GET_ITEM` / `PyList_GET_ITEM` is stale.
- [ ] **Decref-before-update hazard** — Replace the slot first, then
      `Py_DECREF` the old value. Decref can trigger `__del__` which
      re-enters.

### Neuron-specific

- [ ] **NRT / stream ordering explicit** — Cross-stream sync uses
      `NeuronEvent` / `NeuronStreamGuard`, not ad-hoc
      `synchronize()` calls.
- [ ] **Autograd worker thread safety** — Custom backward
      implementations are safe for concurrent execution across
      different nodes; they hold the GIL when touching Python.
- [ ] **Compilation is serialized per cache key** — Two threads
      asking for the same NEFF key must not race — one compiles,
      the other waits. Check `CompilationJob` / `CompilationCache`
      semantics before adding a parallel path.

---

## Backward compatibility

See [bc-guidelines.md](bc-guidelines.md).

- [ ] **No removed / renamed public `torch_neuronx.*` symbols** —
      Unless replaced by a deprecation shim that lives for at least
      one release.
- [ ] **No changed function signatures** on public APIs. New args
      must have defaults and come after existing ones.
- [ ] **No changed defaults** — Changing
      `torch.compile(backend="neuron", dynamic=False)` default,
      for example, silently changes user code.
- [ ] **Environment variable stability** — Renaming or removing
      `TORCH_NEURONX_*` / `NEURON_*` env vars is a BC break. If
      unavoidable, support both names for a release with a
      deprecation warning.
- [ ] **NEFF cache-key schema** — A silent cache-key change
      invalidates users' on-disk artifacts. If the schema changes,
      bump its version explicitly; the old artifacts should be
      ignored, not mis-served.
- [ ] **`_C` extension ABI** — Changing an existing
      `_C._register_*` binding's signature is a hard ABI break for
      any wheel pinned to an older Python-side package. Add a new
      binding; don't repurpose the old one.
- [ ] **Device-module surface** — `torch.neuron.is_available()`,
      `torch.neuron.current_device()`, `torch.neuron.device_count()`
      and similar are called by user code directly. Do not remove
      or change semantics.

---

## Performance

### Obvious regressions

- [ ] **No unnecessary Py↔device round-trips** — Every `.to('cpu')`
      in a hot path is suspect. CPU fallback is already slow; adding
      a second bounce is gratuitous.
- [ ] **No recompilation on every call** — A new path whose NEFF
      cache key includes a non-deterministic or per-call input
      (memory address, iteration count, timestamp) defeats caching.
- [ ] **No Python loops over tensor elements** — Use vectorized ops.
- [ ] **Contiguous materialization only when needed** — Don't call
      `ensure_contiguous_on_device` on inputs the op can handle in
      whatever layout it receives.

### Benchmarking

- [ ] **Perf claims backed by `tools/run-block-test`** — CRs
      claiming a speedup include numbers from `./tools/run-block-test`
      or `./tools/run-test-parallel --coverage`, not hand-rolled
      timing loops.
- [ ] **Regression tests for fixed perf bugs** — If the CR fixes a
      perf bug, add a bounded assertion in a block-perf test so the
      bug doesn't regress silently.

---

## Documentation

- [ ] **CR description matches the diff** — The description's "what
      changed" section lists every affected subsystem. A CR labeled
      "fix gather" that also touches `cast_policy.py` is lying in
      the title.
- [ ] **Env vars documented** — Any new / renamed env var is
      reflected in `docs/environment_variables.md`.
- [ ] **`README.md` updated** — User-facing changes to
      `torch.compile(backend="neuron")` support, supported dtypes,
      supported torch.compile params, or example code must be
      reflected in `README.md`.
- [ ] **`CONTRIBUTING.md` / `docs/new_op.md` updated** — If the CR
      introduces a new developer workflow or op-addition pattern,
      update the recipe.

---

## CI interaction

- [ ] **No skipping CI** — Tests aren't marked `skip` / `xfail`
      solely to make CI green. Every skip has a tracking issue or
      registry entry with a root cause.
- [ ] **No CI flakiness ignored** — If the CR's pipeline has a
      failing upstream or distributed test, verify it's a real
      regression vs. a known flake before approving.
- [ ] **Huggingface / vertical / distributed suites considered** —
      If the CR touches code they exercise, ensure the relevant
      extra suites ran. Main `pull.yml` excludes them by default.

---

## Review exit criteria

Before you sign off:

1. Every changed line has been read.
2. Every finding cites a file and line.
3. Every finding was fact-checked (sub-agent re-read the code).
4. The Recommendation is one of **Approve**, **Request Changes**, or
   **Needs Discussion**, and it follows the rule that missing tests
   → Request Changes.
5. You have not reported anything that CI (`lint.yml`, pre-commit,
   `pull.yml`) already catches.
