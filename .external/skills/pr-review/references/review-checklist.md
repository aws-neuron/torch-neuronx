# torch-neuronx Pull Request Review Checklist

Use this checklist for defects that automated checks cannot identify.
Skip formatting, import order, generated boilerplate, and reported test status.
Investigate each applicable item in source before you create a finding.

## Code quality and design

### Interfaces and state

- [ ] Keep state explicit at module construction or object construction.
- [ ] Reject hidden cross-file flags that change behavior through `getattr` or `setattr`.
- [ ] Use separate functions or classes for fundamentally different code paths.
- [ ] Document new Python-to-C++ and operation-to-kernel contracts.
- [ ] State inputs, outputs, invariants, ownership, and cleanup duties.
- [ ] Match the established pattern in the same file and subsystem.
- [ ] Avoid speculative abstractions and one-use trivial helpers.
- [ ] Keep registration and initialization order explicit.
- [ ] Make repeated import and registration safe.

### Public API design

- [ ] Re-export each intended public callable from a documented package surface.
- [ ] Document new `torch.neuron.*` APIs in the change description.
- [ ] Document each new environment variable in `docs/environment_variables.md`.
- [ ] Avoid hidden control flags on a public function.
- [ ] Search for an established decorator, fixture, and registration pattern first.
- [ ] Check type hints, documentation, discoverability, and future compatibility.
- [ ] Make example code runnable and show only a supported path.

### Code clarity

- [ ] Use names that state intent.
- [ ] Use comments only for non-obvious context.
- [ ] Remove unused code instead of commenting it out.
- [ ] Keep the solution no more complex than the requirement.
- [ ] Make the change description match every affected subsystem.

## Eager operation registration

Read `docs/AI-python-ops.md` before you assess this section.

- [ ] Use `@neuron_op` or `@register_aten` instead of new manual registration plumbing.
- [ ] Subclass the correct `OperationImplementation` variant.
- [ ] Override `_execute_impl` by default so `OperationImplementation.execute` keeps common policy behavior.
- [ ] Override `execute` only when the implementation owns wrapper-level policy, autograd, or specialized dispatch.
- [ ] Compare valid examples before an override: `TorchMlirOpImpl.execute` owns preprocessing and int64 policy, `ContiguousBroadcastImpl.execute` owns autograd, and `GatherNKIImpl.execute` owns specialized NKI gather dispatch.
- [ ] Keep `can_handle()` cheap, pure, and free of tensor allocation.
- [ ] Justify priority against lower-priority implementations.
- [ ] Set `DISABLE_DTYPE_AUTOCAST` when exact input dtypes affect semantics.
- [ ] Set `DISABLE_EMPTY_SHORTCIRCUIT` when empty inputs must still execute.
- [ ] Put a new MLIR operation in the existing operation-family file when possible.
- [ ] Mark only graph-changing or specialization arguments as static.
- [ ] Keep a plain scalar operand dynamic unless specialization is necessary.
- [ ] Avoid a separate output function when one registration handles the output argument.
- [ ] Use separate overload functions only when registration metadata must differ.
- [ ] Keep Python and C++ registration consistent for a performance-sensitive operation.

## Cast policy and contiguity

Read the cast-policy section in `docs/AI-python-ops.md`.

- [ ] Use `bounce_64bit` for float64 and int64 transfer decisions.
- [ ] Use `ensure_contiguous_on_device` for Neuron tensors.
- [ ] Avoid `tensor.contiguous()` in a Neuron operation implementation.
- [ ] Use the standard CPU-to-Neuron, Neuron-to-CPU, and device copy helpers.
- [ ] Validate memory format or reject an unsupported layout with a clear error.
- [ ] Route dtype-cast warnings through the one-time warning helper.
- [ ] Test exact integer behavior when a cast can truncate indices.

## NKI integration

Read the NKI sections in `docs/AI-python-ops.md`.

- [ ] Match the neighboring NKI path instead of assuming one wrapper fits every operation.
- [ ] For `@nki_op`, trace `NKIHOPCaller` through the compile HOP path and the eager `_C._nki_dispatch_eager` path into `NkiDispatchCore`.
- [ ] Use `NKITorchMlirOpImpl` when the HOP or lowering path compiles the global NKI operation through torch-mlir.
- [ ] Use `register_dynamic_nki` and a `DynamicNkiEntry` for an ATen dynamic-kernel adapter that follows the existing adapter registry.
- [ ] For a local NKI dispatch key, include the kernel, constants, grid, tensor metadata, and other behavior not represented by the compiled IR.
- [ ] Do not duplicate identity inputs that serialized HLO or compiler configuration already encode in the persistent NEFF key.
- [ ] List every mutated argument name in `mutates_args`.
- [ ] Provide an autograd path for a trainable operation.
- [ ] Lower the separate backward operation when eager backward dispatches one.
- [ ] Test that both forward and backward execute on Neuron when support claims require it.
- [ ] Update `docs/torch_api_compatibility.md` only for the complete operation contract.

## C++ extension and dispatcher

Read `docs/AI-csrc.md` before you assess this section.

- [ ] Keep Python and C++ registration consistent when both paths apply.
- [ ] Release dispatcher locks before any call into Python.
- [ ] Use RAII guards for runtime handles, allocators, profiler scopes, and locks.
- [ ] Honor `NEURON_LAUNCH_BLOCKING` in each new asynchronous path.
- [ ] Hold the GIL while code accesses a `PyObject*`.
- [ ] Release the GIL for long C++ work that does not access Python objects.
- [ ] Do not keep a borrowed Python reference across a GIL release.
- [ ] Replace a Python container slot before the old value can run finalization code.
- [ ] Preserve a documented public binding name and signature or provide a migration path.
- [ ] Remember that the private `_C` extension and its Python callers ship in the same wheel.
- [ ] Search public documentation, examples, and external callers before assigning backward-compatibility impact to a direct `_C` change.

## torch.compile backend

Read `docs/AI-dynamo-backend.md` before you assess this section.

- [ ] Preserve the documented mode, shape, dtype, and input-layout contract.
- [ ] Put a new graph transform under the established FX pass directory.
- [ ] Keep graph transforms local and composable.
- [ ] Follow the local `NeuronFXPass` and `PassResult` convention.
- [ ] Mutate `gm.graph` in place when that matches the surrounding passes.
- [ ] Run `gm.graph.lint()` and `gm.recompile()` after a graph mutation.
- [ ] Set `PassResult.modified` to whether the pass actually changed the graph.
- [ ] Guard decomposition-table additions by the intended operation.
- [ ] Preserve an `@nki_op` boundary without an unintended graph break.
- [ ] Require an explicit identity input only for compile-time behavior that serialized IR and compiler configuration do not encode.
- [ ] Test cross-cutting attention, sharding, or precompile changes in the related suite.

## Compilation and NEFF cache

Read `docs/AI-compilation-kernels.md` before you assess this section.

- [ ] Identify whether the change touches the HLO precompile cache or the persistent NEFF cache.
- [ ] For the HLO precompile cache, inspect `torch_neuronx/python_ops/compilation/hlo_cache.py`, its key, manifest, and local schema version.
- [ ] For the default persistent NEFF path, preserve the content-addressed key from serialized HLO or MLIR bytes plus compiler configuration.
- [ ] Require an extra identity input only for output-changing behavior that is not encoded in serialized IR or compiler configuration.
- [ ] Prove artifacts created under old semantics become cache misses.
- [ ] Document environment variables in `docs/environment_variables.md`.
- [ ] Document compiler CLI flags in the compiler documentation that owns those flags.
- [ ] Respect `TORCH_NEURONX_DEBUG_DIR` for artifact output.
- [ ] Respect `TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS` for artifact retention.
- [ ] Avoid hardcoded artifact paths.
- [ ] Serialize compilation for one cache key so concurrent callers cannot race.

## Distributed execution

Read `docs/AI-distributed.md` before you assess this section.

- [ ] Extend the C++ `NeuronBackend`, `NeuronWork`, and `NeuronWatchdog` path.
- [ ] Keep a new collective reachable through DeviceMesh and DTensor when applicable.
- [ ] Test DTensor behavior under `tests/distributed/dtensor/`.
- [ ] Compose all-reduce changes with bucketing and premultiplied sum behavior.
- [ ] Mark multi-device tests with the repository's established marker.
- [ ] Preserve explicit stream and event ordering.

## Runtime and package entry points

Read `docs/AI-runtime-api.md` before you assess this section.

- [ ] Preserve package entry points unless the change is intentional and tested.
- [ ] Keep `_autoload` safe under repeated import.
- [ ] Define environment defaults at the package initialization boundary.
- [ ] Register custom tensor containers with `torch.utils._pytree`.
- [ ] Preserve the documented `torch.neuron.*` device-module surface.
- [ ] Use project logging instead of `print()` in production paths.
- [ ] Use traceable shape and dtype checks in operation implementations.

## Tests

Read `docs/AI-testing.md` before you assess this section.

### Test existence and placement

- [ ] Add tests for new behavior.
- [ ] Add a regression test that fails without each bug fix.
- [ ] Add, remove, or narrow registry entries under `tests/pytorch_tests_native/skip_registry/`; `neuron_test_registry.py` loads that registry.
- [ ] Avoid a source test patch that hides a backend defect.
- [ ] Put operation tests under `tests/python_ops/`.
- [ ] Put backend pass tests under the matching unit or component directory.
- [ ] Put collective tests under `tests/distributed/`.
- [ ] Create a new test file only for a new feature area.
- [ ] Register each new C++ test in Bazel. Add it to CMake only when the CMake build supports its dependencies.

### Test patterns

- [ ] Use `torch.testing.assert_close` for tensor comparisons.
- [ ] Build correctness inputs on CPU and move them to Neuron explicitly.
- [ ] Compare against the CPU implementation when it defines the contract.
- [ ] Use repository profiling tools instead of wall-clock timing in correctness tests.
- [ ] Restore modified global runtime state with a fixture.
- [ ] Reuse the Dynamo reset fixture established in the same test area, or add a scoped fixture that resets before and after the test.
- [ ] Use the repository's multi-device marker and filename conventions.
- [ ] Parametrize repeated dtype and shape cases.
- [ ] Use `pytest.raises(..., match=...)` for expected failures.
- [ ] Give each skip a specific public issue or registry reason.

### Coverage

- [ ] Cover empty tensors, scalar tensors, broadcasting edges, and boundary dtypes.
- [ ] Cover non-contiguous input when the operation claims to support it.
- [ ] Cover functional and output variants when both are public.
- [ ] Cover forward and backward for differentiable operations.
- [ ] Cover cache reuse with calls that differ in each output-changing constant.
- [ ] Cover repeated import for new registration logic.

## Security

- [ ] Avoid loading caller-provided pickle data.
- [ ] Avoid `torch.load(..., weights_only=False)` for caller-provided content.
- [ ] Keep credentials and non-public locations out of logs and artifacts.
- [ ] Pass subprocess arguments as a list.
- [ ] Avoid interpolated user input with `shell=True`.
- [ ] Prevent untrusted input from selecting an environment-variable name.
- [ ] Validate paths before a compiler or artifact process writes to them.

## Thread and compilation safety

### Python

- [ ] Protect shared mutable caches and registries.
- [ ] Add a lock for lazy registration outside import-time GIL serialization.
- [ ] Use one lock order across dispatcher, cache, and compilation paths.
- [ ] Avoid a check-then-write race around cache entries.

### C++ and Python boundaries

- [ ] Protect shared state with a mutex or a correctly ordered atomic.
- [ ] Use `std::lock_guard` or `std::unique_lock` for lock lifetime.
- [ ] Hold the GIL for Python object access.
- [ ] Do not use a borrowed Python reference after GIL release.
- [ ] Replace a container slot before decrementing the old reference.
- [ ] Make custom autograd backward safe across worker threads.
- [ ] Use Neuron events and stream guards for cross-stream synchronization.
- [ ] Serialize compilation for each cache key.

## Backward compatibility

Use [the compatibility guide](bc-guidelines.md) for the full review.

- [ ] Preserve public package and device-module symbols.
- [ ] Add optional parameters after existing parameters and provide defaults.
- [ ] Preserve defaults, return types, shapes, and exception families.
- [ ] Provide a deprecation path for a user-visible removal or rename.
- [ ] Preserve environment-variable compatibility.
- [ ] Distinguish HLO precompile-cache identity from the persistent NEFF content key and prove old artifacts miss after a semantics change.
- [ ] Preserve compatibility for documented public bindings; search public callers before treating a private `_C` change as a break.

## Performance

- [ ] Avoid unnecessary CPU-to-device and device-to-CPU transfers.
- [ ] Keep per-call values, addresses, and timestamps out of a compilation cache key.
- [ ] Avoid Python loops over tensor elements.
- [ ] Materialize a contiguous tensor only when the operation requires it.
- [ ] Back each performance claim with a reproducible public command and baseline.
- [ ] Add a stable regression check for a fixed performance defect when possible.

## Documentation

- [ ] Make the change description match every affected subsystem.
- [ ] Update `docs/environment_variables.md` for each environment-variable change.
- [ ] Update the owning compiler documentation for each compiler CLI flag change.
- [ ] Update `README.md` for a user-visible backend contract change.
- [ ] Update `CONTRIBUTING.md` for a new contributor workflow.
- [ ] Use the recipe in `docs/AI-python-ops.md` for an operation workflow change.
- [ ] Keep each code sample runnable and consistent with a supported path.

## Review exit criteria

1. Read every changed line.
2. Cite a changed file and line for every finding.
3. Consolidate all candidates before fact-checking.
4. Independently fact-check every surviving finding.
5. Remove findings that an automated gate fully proves.
6. Classify each changed user-visible behavior.
7. Select `APPROVE`, `REQUEST_CHANGES`, `NEEDS_DISCUSSION`, or `BLOCKED`.
8. Report test evidence and missing verification.
