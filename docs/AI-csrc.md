# C++ Core (`torch_neuronx/csrc/`)

The `torch_neuronx._C` pybind11 extension. Owns device registration, the
ATen dispatcher, allocators, NRT wrappers, the compilation cache, the
concatenation engine, lazy materialization, streams, profiler, and the
distributed `Work`/`Watchdog` machinery. Built via Bazel (`//:_C.so`,
primary) or CMake (fallback, `_C.so` only — no torch-mlir libs).

> **Navigation → codegraph.** This is a ~30-directory C++ tree; for the file
> map, class layouts, and bindings, ask codegraph:
> `codegraph_explore "NeuronDispatcher NeuronBindings CompilationCache ConcatenationEngine LazyTransformationManager OperationExecutionEngine"`.
> codegraph indexes the cpp sources, so it returns current signatures + call
> graphs the prose tables here used to duplicate. This doc keeps the
> dispatch/cache/concurrency contracts and the Python↔C++ handoff.

## Contracts

**Dispatch (`core/dispatch/NeuronDispatcher`).** Every ATen op on a Neuron
tensor flows through it. An `OpRegistration` table is populated by
`_register_neuron_op` from Python at import. A `DispatchCacheEntry` cache is
keyed by op name + input spec (shapes, dtypes, static arg values); **on hit
the path stays entirely in C++ with no GIL**. Cache miss: acquire GIL → call
`lower_func` → `ParseLoweringResult` extracts StableHLO bytes + specs →
populate cache → hand to `CompilationCache`. `DispatchWithCasting` upcasts
outputs when 64-bit inputs were downcast.

**Allocators (`core/allocator/`).** Device memory is always lazily bound: the
caller gets an `mmap` placeholder and the physical carve runs later on the
runtime worker through the segment allocator (carves large NRT slabs into
blocks; `lazy::AllocBind`/`DeallocRelease`). Pinned host
allocator gated by `TORCH_NEURONX_USE_PINNED_MALLOC`.

**Compilation cache (`core/compilation/`).** Thread-safe LRU with
shared/exclusive locking; `PendingCompilation` per-key coordination **prevents
duplicate compiles** of the same key. Limits via
`NEURON_COMPILATION_CACHE_MAX_SIZE_MB` / `_MAX_ENTRIES`. Backends: *Local*
(default, shells to `neuronx-cc` via the Python wrapper) and *Service*
(`NEURON_COMPILE_SERVICE=1`, out-of-process over a Unix socket).

**Persistent NEFF cache (`core/cache/`).** On-disk, **NFS-safe via atomic
writes** (temp + rename), two-level dir sharding `{dir}/ab/cd/abcd…neff`. Dir
from `TORCH_NEURONX_NEFF_CACHE_DIR` (fallback `/tmp/neff_cache`), disabled by
`TORCH_NEURONX_NEFF_DISABLE_CACHE`. `FileLock` is process-safe with timeout +
stale detection.

**Concatenation engine (`core/concatenation/`).** Groups consecutive
compatible ops into one fused compile/execute via time-based buffering: ops
accumulate until timeout, `synchronize`, or a non-XLA op flushes them.

**Lazy materialization (`core/lazy_materialization/`).** Defers view ops
(slice/transpose/broadcast/reshape) as `TensorTransformation` descriptors so
the engine can elide/fuse them. Two engine entry points:
`ProcessOperationInputs` (prologue — pads/re-lays-out non-contiguous inputs
before the kernel) and `ProcessOperationOutputs` (epilogue — scatters /
inverse-transposes the contiguous result into a non-contiguous output). The
epilogue can emit a `SCATTER_IOTA` transformation gated by
`TORCH_NEURONX_ENABLE_SCATTER_IOTA`. (`OperationTransformations.*`, renamed
from `OperationPrologue.*`.)

**Execution engine (`core/OperationExecutionEngine`).** Global engine across
all streams; pipeline `prepare → compile → schedule → execute → notify`.
Queue-depth knobs: `DEFAULT_QUEUE_LOW_THRESHOLD = 15`,
`kDefaultMaxInflightPerXU = 8`.

**Distributed (`distributed/`).** C++ `NeuronBackend` is the **only** PG impl
(legacy Python `ProcessGroupNeuron` removed). `NeuronWork` (`c10d::Work`)
tracks async completion via streams/events with a thread-safe `TensorShelf`
keeping tensors alive in-flight; `detachStashedTensorShelf()` hands ownership
to the watchdog. `NeuronWatchdog` polls completion, drains completed shelves
**on the main thread with the GIL held**, and releases the GIL before
`join()` to avoid deadlock. See `AI-distributed.md`.

## Python ↔ C++ handoff

1. Python registers impls via `_C._register_neuron_op(name, python_fn,
   lower_fn, ...)` in `python_ops/__init__.py`.
2. Op call → `NeuronDispatcher::dispatch`.
3. Cache miss → `lower_func(...)` (GIL) → `LoweringResult` (StableHLO + specs).
4. Compile via `CompilationCache` → `LocalCompilationBackend` →
   `neuronx_cc_wrapper` → `neuronx-cc`.
5. NEFF loaded into NRT, cached by hash in `ModelHandleCache`.
6. Execution runs async through `OperationExecutionEngine` + `NeuronStream`.

## Authoritative docs (cite, don't restate)

- `.internal_docs/DispatchArchitecture.md`,
  `NeuronSegmentAllocatorArchitecture.md`, `LazyAllocationArchitecture.md`,
  `EpilogueArchitecture.md`.
- `docs/cpp_dispatch_onboarding_guide.md`.
- C++ test invocation → `BAZEL_BUILD.md` + `CONTRIBUTING.md`; tests mirror the
  `csrc/` tree under `tests/csrc/`.

## See also

- `AI-runtime-api.md` — Python exposure of these facilities.
- `AI-dynamo-backend.md` — the compile path calling `compile_graph`.
- `AI-distributed.md` — `NeuronWork`/`NeuronWatchdog` in the PG.
- `AI-compilation-kernels.md` — Python glue between dispatcher and `neuronx-cc`.
