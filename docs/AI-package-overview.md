# Package Overview

`torch-neuronx` is a PyTorch backend for AWS Neuron devices that adds a
`'neuron'` device via PyTorch's `PrivateUse1` mechanism. For the elevator
pitch and usage, see `README.md` (authoritative). Two execution paths:

1. **Eager mode** — `tensor.to('neuron')` routes individual ATen ops
   through a C++ dispatcher and per-op implementations.
2. **`torch.compile` mode** — `torch.compile(model, backend="neuron")`
   compiles whole FX graphs to NEFF via a StableHLO pipeline and executes
   them through the Neuron Runtime (NRT).

> **Navigation → codegraph.** For the import-time boot sequence, the
> per-op eager dispatch path, or the `torch.compile` pipeline as source,
> ask codegraph:
> `codegraph_explore "torch_neuronx __init__ _autoload neuron_backend NeuronDispatcher dispatch"`.
> This doc is the **subsystem index + the two cross-cutting invariants** —
> the part no single source file states.

## How it's wired into PyTorch (contract)

Two `pyproject.toml` entry points register the package automatically:
`torch.backends` → `torch_neuronx:_autoload` (registers the `'neuron'`
device) and `torch_dynamo_backends` → `neuron_backend`. On import,
`torch_neuronx/__init__.py` loads `_C`, renames the `PrivateUse1` backend
to `"neuron"`, registers `torch.neuron.*`, registers Python op impls, and
sets env-var defaults. Order matters — `_C` device registration precedes
everything. (Boot sequence as source: codegraph the symbols above.)

## Subsystem index (routing)

| Subsystem | Path | Doc |
|---|---|---|
| User-facing Python API (device, streams, memory, metrics, profiler, NKI HOPs) | `torch_neuronx/*.py` | `AI-runtime-api.md` |
| `torch.compile` backend — FX → StableHLO → NEFF, FX passes, MegaCache | `torch_neuronx/neuron_dynamo_backend/` | `AI-dynamo-backend.md` |
| Eager-mode op impls (MLIR/NKI), `@neuron_op`, cast policy | `torch_neuronx/python_ops/` | `AI-python-ops.md` |
| C++ `_C` extension: dispatcher, allocators, cache, streams, profiler | `torch_neuronx/csrc/` | `AI-csrc.md` |
| `"neuron"` c10d backend, collectives, DTensor sharding | `torch_neuronx/distributed/` + `csrc/distributed/` | `AI-distributed.md` |
| NEFF cache, `neuronx-cc` wrapper | `torch_neuronx/kernels/` | `AI-compilation-kernels.md` |
| Tests | `tests/` | `AI-testing.md` |
| Examples | `examples/` | `AI-examples.md` |
| Build system (Bazel primary, CMake fallback), vendored torch-mlir | `bazel/`, `BUILD.bazel`, `third_party/torch-mlir/` | `AI-build-infra.md` |

## Two cross-cutting invariants

These anchor behavior across every subsystem; for the detailed behavior
and accuracy caveats, see `README.md` Limitations (authoritative).

- **Neuron computes on 32-bit types.** `float64`/`uint64` are down-cast on
  entry, re-widened on exit. `int64` defaults to **packed mode**
  (`TORCH_NEURONX_INT64_MODE=packed`): kept losslessly on device as `int32`
  lane-pairs, with only arithmetic on the low 32 bits. The dynamo
  `Executor` path still down-casts via `view()`/`.to()`. Authoritative:
  `docs/packed_int64_design.md`; rules in `python_ops/cast_policy.py` +
  `int64_policy.py`.
- **Only contiguous tensors execute on-device.** Non-contiguous inputs are
  materialized via the `contiguous_*` op family (`python_ops/contiguous*.py`)
  or, for eager copies, `csrc/utils/CopyUtils.cpp`.

## Where to start

| Question | Start here |
|---|---|
| "Why is my model slow to compile?" | `AI-dynamo-backend.md` + `NEURON_CC_FLAGS` |
| "How do I add support for a new ATen op?" | `AI-python-ops.md` + `docs/new_op.md` |
| "Why does my tensor get copied to CPU?" | `AI-python-ops.md` (cast policy) + `AI-csrc.md` (CPUFallbackExecutor) |
| "How do I debug a failed NEFF compile?" | `AI-build-infra.md` + `TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS` |
| "How are collectives implemented?" | `AI-distributed.md` |

## Authoritative human docs (cite, don't restate)

- Build/test quickstart → `README.md`, `CONTRIBUTING.md`, `BAZEL_BUILD.md`.
- Env-var reference → `docs/environment_variables.md`.
