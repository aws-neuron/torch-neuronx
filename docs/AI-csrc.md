# C++ Core (`torch_neuronx/csrc/`)

The `torch_neuronx._C` pybind11 extension. Owns device registration, the
ATen dispatcher, allocators, NRT wrappers, the compilation cache, the
concatenation engine, lazy materialization, streams, profiler, and the
distributed `Work`/`Watchdog` machinery. Built via Bazel (`//:_C.so`,
primary) or CMake (fallback, `_C.so` only - no torch-mlir libs).

> **Source navigation.** Run this command from the repository root:
>
> `rg -n 'NeuronDispatcher|NeuronBindings|CompilationCache|ConcatenationEngine|LazyTransformationManager|OperationExecutionEngine' torch_neuronx/csrc`
>
> If `rg` is unavailable, use `git grep -n -E` with the same pattern and paths.

## Contracts

**Dispatch (`torch_neuronx/csrc/core/dispatch/`).** This path handles ATen ops
registered from Python through `_register_neuron_op`. An `OpRegistration` table
is populated at import. Native `PrivateUse1` kernels registered in
`torch_neuronx/csrc/core/NeuronBindings.cpp` and the global `PrivateUse1` CPU
fallback are separate paths. For registered ops, a `DispatchCacheEntry` cache
is keyed by op name + input spec (shapes, dtypes, static arg values); **on hit
the path stays entirely in C++ with no GIL**. Cache miss: acquire GIL -> call
`lower_func` -> `ParseLoweringResult` extracts StableHLO bytes + specs ->
populate cache -> hand to `CompilationCache`. `DispatchWithCasting` upcasts
outputs when 64-bit inputs were downcast.

**Allocators (`torch_neuronx/csrc/core/allocator/`).** Device memory is always lazily bound: the
caller gets an `mmap` placeholder and the physical carve runs later on the
runtime worker through the segment allocator (carves large NRT slabs into
blocks; `lazy::AllocBind`/`DeallocRelease`). Pinned host
allocator gated by `TORCH_NEURONX_USE_PINNED_MALLOC`.

**Compilation cache (`torch_neuronx/csrc/core/compilation/`).** Thread-safe LRU with
shared/exclusive locking; `PendingCompilation` per-key coordination **prevents
duplicate compiles** of the same key. Limits via
`NEURON_COMPILATION_CACHE_MAX_SIZE_MB` / `_MAX_ENTRIES`. The service backend is the default. Set `NEURON_COMPILE_SERVICE=0` to select
`LocalCompilationBackend`; service setup failure also falls back to local.
The local backend invokes the kernel execution callback, which calls
`NeuronCompiler` directly. WLO compilation uses the local backend.

**Persistent NEFF cache (`torch_neuronx/csrc/core/cache/`).** The on-disk cache uses
temp files and atomic renames. Readers do not see a partial NEFF. It uses two-level
directory sharding: `{dir}/ab/cd/abcd...neff`. The directory comes from
`TORCH_NEURONX_NEFF_CACHE_DIR` and defaults to `/tmp/neff_cache`. Set
`TORCH_NEURONX_NEFF_DISABLE_CACHE` to disable it. `FileLock` normally coordinates
processes. After its timeout, it replaces the lock path without checking the owner.
A compile that exceeds the timeout can run concurrently in two processes.

**Concatenation engine (`torch_neuronx/csrc/core/concatenation/`).** Groups consecutive
compatible ops into one fused compile/execute via time-based buffering: ops
accumulate until timeout, `synchronize`, or a non-XLA op flushes them.

**Lazy materialization (`torch_neuronx/csrc/core/lazy_materialization/`).** Defers view ops
(slice/transpose/broadcast/reshape) as `TensorTransformation` descriptors so
the engine can elide/fuse them. Two engine entry points:
`ProcessOperationInputs` (prologue - pads/re-lays-out non-contiguous inputs
before the kernel) and `ProcessOperationOutputs` (epilogue - scatters /
inverse-transposes the contiguous result into a non-contiguous output). The
epilogue can emit a `SCATTER_IOTA` transformation gated by
`TORCH_NEURONX_ENABLE_SCATTER_IOTA`. (`OperationTransformations.*`, renamed
from `OperationPrologue.*`.)

**Execution engine (`torch_neuronx/csrc/core/OperationExecutionEngine.cpp`).** Global engine across
all streams. It coordinates preparation, asynchronous compilation, scheduling,
and completion. `NotifyExecutionReady` wakes the runtime worker when work can
advance. Queue-depth knobs: `DEFAULT_QUEUE_LOW_THRESHOLD = 15`,
`kDefaultMaxInflightPerXU = 8`.

**Distributed (`torch_neuronx/csrc/distributed/`).** C++ `NeuronBackend` is the **only** PG impl
(legacy Python `ProcessGroupNeuron` removed). `NeuronWork` (`c10d::Work`)
tracks async completion via streams/events with a thread-safe `TensorShelf`
keeping tensors alive in-flight; `detachStashedTensorShelf()` hands ownership
to the watchdog. `NeuronWatchdog` polls completion. The host caller drains completed shelves
while holding the GIL. `stop()` releases the GIL before joining the watchdog
thread, then performs final shelf cleanup when the GIL is available. See `docs/AI-distributed.md`.

## Python <-> C++ handoff

1. Python registers impls via `_C._register_neuron_op(name, python_fn,
   lower_fn, ...)` in `torch_neuronx/python_ops/__init__.py`.
2. Op call -> `NeuronDispatcher::dispatch`.
3. Cache miss -> `lower_func(...)` (GIL) -> `LoweringResult` (StableHLO + specs).
4. `CompilationCache` checks memory and persistent caches, selects the service
   backend by default or the local backend for `NEURON_COMPILE_SERVICE=0`, and
   hands local compilation to `KernelExecution::CompileToNeff`.
5. `KernelExecution::CompileToNeff` calls `NeuronCompiler`, which invokes
   `neuronx-cc`.
6. NEFF loaded into NRT and cached by compilation key plus device index in
   `ModelHandleCache`.
7. Execution runs async through `OperationExecutionEngine` + `NeuronStream`.

## Authoritative docs (cite, don't restate)

- `docs/cpp_dispatch_onboarding_guide.md`.
- C++ test invocation -> `BAZEL_BUILD.md` + `CONTRIBUTING.md`; tests mirror the
  `torch_neuronx/csrc/` tree under `tests/csrc/`.

## See also

- `docs/AI-runtime-api.md` - Python exposure of these facilities.
- `docs/AI-dynamo-backend.md` - the compile path calling `compile_graph`.
- `docs/AI-distributed.md` - `NeuronWork`/`NeuronWatchdog` in the PG.
- `docs/AI-compilation-kernels.md` - Python glue between dispatcher and `neuronx-cc`.
