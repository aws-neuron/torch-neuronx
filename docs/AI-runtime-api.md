# Runtime API (`torch_neuronx` public surface)

The top-level Python modules in `torch_neuronx/` expose the public API
user code touches directly.

> **Navigation → codegraph.** For the full signature list of any API group
> (`torch.neuron.*`, `Stream`/`Event`, memory, metrics, `NeuronProfiler`,
> `@nki_op`/`wrap_nki`, `utils` helpers), ask codegraph:
> `codegraph_explore "torch_neuronx neuron streams memory metrics profiling NeuronProfiler nki_hop utils"`.
> This doc keeps only the load-order contract, non-obvious behaviors, and
> authority routing — not the API enumeration codegraph regenerates.

## Module load and device registration (contract)

`torch_neuronx/__init__.py` is the anchor. On import, in order:
1. Import `_C`, call `_C._register_device()` + `_C._register_profiler()`.
2. `torch.utils.rename_privateuse1_backend("neuron")`.
3. `torch._register_device_module("neuron", neuron)` → `torch.neuron.*` works.
4. `torch.utils.generate_methods_for_privateuse1_backend(...)`.
5. **Wrap every `_C.*` callable with `torch._dynamo.disable(...)`** — the
   C++ extension cannot be traced by Dynamo. (Load-bearing: forgetting this
   for a new `_C` binding makes it traceable and breaks compile.)
6. Set env-var defaults from `_build_config.py`.
7. Register the `_autoload()` entry point (no-op after first import).

**Fork safety:** the runtime may not be reinitialized in forked
subprocesses — use the `spawn` start method. `_lazy_init()` / `_after_fork()`
track init state + bad-fork detection.

## Non-obvious behaviors worth keeping

- **`set_rng_state` syncs Philox state to CPU** (guarded against
  `FakeTensorMode`) so CPU-fallback RNG respects DTensor's offset. On-device
  GPSIMD RNG state goes through `python_ops/nki_kernels/rng_state.py`.
- **AMP supported dtypes**: `[float32, float16, bfloat16]` — no float64.
- **`NeuronConfig.custom_profiler_config` must start with `neuron_profile:`** —
  `Profiler.cpp:hasNeuronProfileKey()` uses an `rfind(..., 0) == 0` prefix
  check, so ordering of the prefix matters.
- **`NeuronProfiler.export_trace`** writes the Chrome trace into the NRT
  instance output dir so PyTorch + NRT traces land together.
- **`TorchNeuronNKIKernelV3._convert_torch_input`** downcasts `float64`→`float32`,
  `int64`→`int32`, maps bool→`uint8`; enables float8 when the torch build
  supports it.
- **`global_version_hash()`** (`utils.py`) is a component of NEFF cache keys —
  changing it invalidates the cache.

## Authoritative human docs (cite, don't restate)

- Memory + metrics API tour (`memory_stats` flat/nested, `peak_memory_stats`,
  `segment_pool_analysis`, counters/reports, snapshot calls) →
  `docs/statistics/torch_neuronx_stats_readme.md`.
- NKI kernel registration workflow → `docs/cpp_dispatch_onboarding_guide.md`.
- NUMA pinning / `neuron-numa-launch` → `docs/numa_optimization.md`.
- Env-var reference → `docs/environment_variables.md`.

## See also

- `AI-dynamo-backend.md` — `torch.compile(backend="neuron")`.
- `AI-python-ops.md` — per-op implementation registration.
- `AI-csrc.md` — the C++ side behind `_C`.
