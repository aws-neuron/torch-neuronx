# Package Overview

`torch-neuronx` is a PyTorch backend for AWS Neuron devices that adds a
`'neuron'` device via PyTorch's `PrivateUse1` mechanism. For the elevator
pitch and usage, see `README.md` (authoritative). Two execution paths:

1. **Eager mode** - `tensor.to('neuron')` routes individual ATen ops
   through a C++ dispatcher and per-op implementations.
2. **`torch.compile` mode** - `torch.compile(model, backend="neuron")`
   compiles whole FX graphs to NEFF via a StableHLO pipeline and executes
   them through the Neuron Runtime (NRT).

> **Source navigation.** Run this command from the repository root:
>
> `rg -n '_autoload|neuron_backend|NeuronDispatcher|dispatch' pyproject.toml torch_neuronx`
>
> If `rg` is unavailable, use `git grep -n -E` with the same pattern and paths.

## How it's wired into PyTorch (contract)

`pyproject.toml` declares `torch.backends` -> `torch_neuronx:_autoload` and
`torch_dynamo_backends` ->
`torch_neuronx.neuron_dynamo_backend.backend:neuron_backend`.

`torch_neuronx/__init__.py` first prepares the bundled torch-mlir path, imports
PyTorch, and sets environment defaults. It then imports `_C`, registers the
device and profiler, initializes the C++ int64 policy, and disables Dynamo
tracing for `_C` callables. Next it imports the Python submodules and public
APIs. If no other backend owns `PrivateUse1`, it renames the backend, registers
`torch.neuron`, generates convenience methods, and later registers Python
operations. Read the file for the guarded patch registrations that follow.

## Subsystem index (routing)

| Subsystem | Path | Doc |
|---|---|---|
| User-facing Python API (device, streams, memory, metrics, profiler, NKI HOPs) | `torch_neuronx/` | `docs/AI-runtime-api.md` |
| Environment and bug-report collection | `torch_neuronx/collect_env.py` | `docs/AI-runtime-api.md` |
| `torch.compile` backend - FX -> StableHLO -> NEFF, FX passes, MegaCache | `torch_neuronx/neuron_dynamo_backend/` | `docs/AI-dynamo-backend.md` |
| Eager-mode op impls (MLIR/NKI), `@neuron_op`, cast policy | `torch_neuronx/python_ops/` | `docs/AI-python-ops.md` |
| C++ `_C` extension: dispatcher, allocators, cache, streams, profiler | `torch_neuronx/csrc/` | `docs/AI-csrc.md` |
| C++ compilation and persistent NEFF caches | `torch_neuronx/csrc/core/compilation/` + `torch_neuronx/csrc/core/cache/` | `docs/AI-compilation-kernels.md` |
| Python kernel base plus legacy and test-only compiler wrapper | `torch_neuronx/kernels/` | `docs/AI-compilation-kernels.md` |
| `"neuron"` c10d backend, collectives, DTensor sharding | `torch_neuronx/distributed/` + `torch_neuronx/csrc/distributed/` | `docs/AI-distributed.md` |
| Tests | `tests/` | `docs/AI-testing.md` |
| Examples | `examples/` | `docs/AI-examples.md` |
| Build system (Bazel primary, CMake fallback), vendored torch-mlir | `bazel/`, `BUILD.bazel`, `third_party/torch-mlir/` | `docs/AI-build-infra.md` |

## Two cross-cutting invariants

These anchor behavior across every subsystem; for the detailed behavior
and accuracy caveats, see `README.md` Limitations (authoritative).

- **64-bit handling differs by execution path.** Eager `float64` operations
  autocast to `float32`. Eager `int64` defaults to `native`, a per-op cascade:
  supported ops run as S64, non-native `DOWNCAST` ops use the packed int32 view
  and low 32 bits, and unsupported ops use CPU fallback. `packed` is an
  explicit mode that forces int64 operations through the packed view. The
  Dynamo `Executor` downcasts `int64`/`uint64` with `view()` and `float64` with
  `.to()`, then restores requested output dtypes.
  `torch_neuronx/int64_policy.py` is authoritative for the current default and
  per-op routing. `docs/packed_int64_design.md` is design background that
  predates the `native` default.
- **Contiguity is checked at kernel boundaries, with explicit layout support.**
  Prepared kernel tensors reject non-contiguous layouts by default, but callers
  that account for those layouts can set `allow_non_contiguous`. Eager dispatch
  can handle supported non-contiguous inputs with lazy-materialization
  prologues. See `docs/AI-csrc.md` for lazy materialization.

## Where to start

| Question | Start here |
|---|---|
| "Why is my model slow to compile?" | `docs/AI-dynamo-backend.md` + `NEURON_CC_FLAGS` |
| "How do I add support for a new ATen op?" | `docs/AI-python-ops.md` + `docs/cpp_dispatch_onboarding_guide.md` |
| "Why does my tensor get copied to CPU?" | `docs/AI-python-ops.md` (cast policy) + `docs/AI-csrc.md` (CPUFallbackExecutor) |
| "How do I debug a failed NEFF compile?" | `docs/AI-build-infra.md` + `TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS` |
| "How are collectives implemented?" | `docs/AI-distributed.md` |

## Authoritative human docs (cite, don't restate)

- Build/test quickstart -> `README.md`, `CONTRIBUTING.md`, `BAZEL_BUILD.md`.
- Env-var reference -> `docs/environment_variables.md`.
