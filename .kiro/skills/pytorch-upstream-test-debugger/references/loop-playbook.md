# Loop playbook: category-specific fix guidance

Heuristics for the common failure signatures seen when running PyTorch
upstream gradient/gradcheck suites (e.g. `TestBwdGradientsNEURON`,
`test_ops_gradients.py`) against Neuron. Use as a starting hypothesis,
not a spec — every bug is different.

## `dtype_mismatch_double_vs_float`

**Symptom:** `Exception: Double did not match Float` or
`expected dtype torch.float32 ... got dtype torch.float64`.

**Root cause pattern:** a `float64` tensor survives into an op that
`torch_neuronx` only registered for `float32`. A silent CPU fallback
then widens the result back to float64, which disagrees with the
float32 reference.

**Where to look in `torch_neuronx`:**
- The op's ATen dispatch registration (the `PrivateUse1` / `neuron`
  keys). A missing float64 kernel silently falls through. See
  `docs/AI-csrc.md` for the dispatcher and
  `docs/AI-python-ops.md` for the Python-side op registration +
  cast policy.
- Any `cpu_fallback`-gated op list — if the op is in the allowlist,
  float64 inputs leak through.
- Autograd formula registrations that pin the gradient dtype to input
  dtype.

**Fix strategies:**
- Add a float64 kernel that internally casts to float32 and back (with a
  clear warning about precision).
- Route the op through its decomposition so it composes from float32
  ops only.
- Force-cast inputs at the dispatch boundary if downcasting is
  acceptable for the op's numerics.

## `jacobian_mismatch` / `gradcheck_numerical`

**Symptom:** `Jacobian mismatch for output N with respect to input M`
or `numerical vs analytical grad delta exceeds tolerance`.

**Root cause pattern:** gradcheck compares a numerical Jacobian (finite
differences) against the analytical one. Ops with high condition
numbers, catastrophic cancellation, or aggressive fp32 accumulation on
hardware can produce a real delta above the default tolerance. Genuine
compiler precision issues also end up here.

**Where to look:**
- First sanity-check: run the failing gradcheck with very loose
  tolerances (e.g. `rtol=1e-2, atol=1e-2`). If it passes, the fix is a
  per-op tolerance override.
- If tolerance can't close the gap, investigate the forward op's
  precision on Neuron — compare a single forward pass against CPU at
  `float32` and look for accumulation-order or cast-early-vs-late
  differences. The op's Neuron implementation map is in
  `docs/AI-python-ops.md` (MLIR / NKI / XLA
  backends).

**Escalation:** some of these are unfixable without native `float64`
hardware support. An RCA document explaining the op + numerical
analysis is acceptable; leave the test `xfail`ed with a clear reason
that points at the analysis.

## `timeout`

**Symptom:** `Failed: Timeout (>600.0s) from pytest-timeout`.

**Root cause pattern:** gradcheck invokes the forward pass many times
with finite differences. On Neuron, each call can trigger a
compile+execute cycle, and the cumulative cost can exceed pytest's
default timeout even for small tensors.

**Where to look:**
- Graph-cache hit rate. If every sample input has a different shape or
  dtype, the compiler recompiles on every call. Fix by making shapes /
  dtypes homogeneous across samples, or by warming the cache. The NEFF
  cache mechanism, cache key composition, and artifact directory layout
  are in `docs/AI-compilation-kernels.md` (and
  `docs/AI-dynamo-backend.md` for the Dynamo-side `CompileGraph`).
- Sequential compiler invocations. Check any compiler-flag env var
  (e.g. `NEURON_CC_FLAGS`) for a parallelism setting.

**Fix strategies:**
- Reduce gradcheck sample size for the op via its `OpInfo`
  `sample_inputs_func`.
- Warm the cache: run a shape-representative forward pass once before
  gradcheck, so subsequent calls hit the cache.

**Escalation:** if genuinely long-running, bump timeout to 1200 s via a
per-test marker rather than globally. That keeps the signal sharp for
other tests.

## `compiler_error_neuron`

**Symptom:** `Compilation error occurred on Neuron for operation=...`

**Root cause pattern:** the compiler can't lower the op's HLO to Neuron
ISA. Almost always a real compiler bug, not a `torch_neuronx` bug.

**Where to look:**
- The operation name from the error message. Cross-reference against
  your team's compiler issue tracker.
- Run the forward op on CPU first — if that works, compare HLO dumps
  between CPU and Neuron to isolate which op shape the compiler rejects.
  The StableHLO pipeline, the `TORCH_NEURONX_DEBUG_DIR` artifact layout,
  and how to dump MLIR are documented in
  `docs/AI-dynamo-backend.md`.

**Escalation:** compiler bugs generally can't be fixed at the
`torch_neuronx` layer. File a bug against the compiler team, leave an
RCA note, keep the test `xfail`ed with a reason pointing at the ticket.

## `unexpected_success`

**Symptom:** `Failed: Unexpected success`. A test marked expected-to-fail
now passes — either via a registry `XFAIL_GROUP` entry or an upstream
unittest `@expectedFailure` decorator.

**Fix:**
- **Registry-driven (the common case):** the test is xfailed by a group in
  `tests/pytorch_tests_native/neuron_test_registry.py`. Remove it from that
  `XFAIL_GROUP` (delete the whole group if it was its last member) — this is
  the normal Step-8 registry cleanup. The underlying op was fixed; the xfail
  is now stale.
- **Upstream `@expectedFailure`:** the decorator lives in upstream PyTorch
  source, which this repo runs directly and does **not** edit. Don't strip
  it. Track the divergence instead — skip the test via the registry with a
  reason noting it passes on Neuron but is xfailed upstream, or raise it
  upstream — same escalation pattern as `compiler_error_neuron`.

**Do NOT** double-xfail by adding a registry xfail on top of an upstream
`@expectedFailure`. Fix it in one place.

## Common traps

### C/C++ or Cython edits need a rebuild

The inner loop in the main skill shows Python-file edits: edit → ship →
pytest. The new `.py` file is picked up on re-import. But edits under
`torch_neuronx/csrc/`, or any `.cpp` / `.h` / `.cc` / `.pyx` file, compile
into the installed `.so` — Python re-import will **not** pick those up.
You must run `./tools/build` (or the equivalent for your setup) between
the edit and the pytest invocation.

Symptom of forgetting: "my fix isn't doing anything", test fails
identically to before the edit. The `.so` is stale; the source change
was never compiled in. When in doubt, rebuild. For the full build
surface (Bazel, CMake fallback, driver scripts, env flags), see
`docs/AI-build-infra.md`.

### The upstream-test overlay

Many repos that ship PyTorch upstream tests ship them as a clone of the
official PyTorch test source, optionally patched. There is usually a
`conftest.pytest_configure` (or similar) that copies an overlay into
the installed `torch.testing._internal` directory. If the overlay is
stale relative to the installed torch nightly, some errors reproduce
differently.

Symptom: you see a traceback line number or helper-function signature
that doesn't match the torch nightly you have installed. Re-run the
clone/overlay step (often a script like `clone_device_agnostic_tests.py`)
before debugging further.

### xdist worker count — tune per test type

pytest-xdist worker init has a fixed per-worker overhead (on Neuron
nodes, several seconds of driver + context setup each). So `-n` higher
is not always faster. Rough guide for trn2-class nodes:

| Test type | `-n` |
|---|---|
| Distributed tests (NCCL / FSDP / comm collectives) | 8 |
| Autograd tests (`gradcheck`, `gradgradcheck`, `test_ops_gradients.py`) | 16 |
| Other plain unit tests | up to 64 |

For short test lists (under 30), **cap the worker count at or below
the test count**. `-n 32` for 26 autograd tests is worse than `-n 16`
because init cost dominates.

When debugging a single stuck or racy test, drop `-n` entirely. NRT
state-machine effects can produce different failures at different
worker counts, so a single-process run is the canonical repro.

### Don't chase a fix through the test framework

If you are tempted to add an entry to a project-local "test patches"
file (anything that silently modifies test behaviour rather than
fixing the underlying op), stop and look again at the op. Hiding a bug
with a patch keeps the test green while the production path stays
broken. Fixes belong in `torch_neuronx` source, not in a pytest layer.
