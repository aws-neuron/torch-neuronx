# Runtime API (`torch_neuronx` public surface)

The top-level Python modules in `torch_neuronx/` expose the public API
user code touches directly.

> **Source navigation.** Run this command from the repository root:
>
> `rg -n '_register_device|_lazy_init|NeuronProfiler|nki_op|wrap_nki|global_version_hash|collect_env' torch_neuronx`
>
> If `rg` is unavailable, use `git grep -n -E` with the same pattern and paths.

## Module load and device registration (contract)

`torch_neuronx/__init__.py` is the anchor. On import, it:
1. Prepares the bundled torch-mlir path, imports PyTorch, loads build defaults,
   and sets environment defaults.
2. Imports `_C`, calls `_C._register_device()` and `_C._register_profiler()`,
   initializes the C++ int64 policy, and wraps `_C` callables with
   `torch._dynamo.disable`.
3. Imports distributed, device, profiling, Python-op, memory, metrics, NKI,
   stream, and event APIs.
4. Checks whether another backend owns `PrivateUse1`. If not, it renames the
   backend to `neuron`, registers the `torch.neuron` module, and generates
   PrivateUse1 convenience methods.
5. Installs fork handling, registers Python operations under the same ownership
   guard, then applies mesh, random, graph-break, and optional auto-compile
   patches.

`_autoload` is the `pyproject.toml` backend entry point. It imports the package
once; it is defined before the main initialization sequence.

**Fork safety:** the runtime may not be reinitialized in forked
subprocesses - use the `spawn` start method. `_lazy_init()`,
`_snapshot_neuron_runtime_before_fork()`, and
`_handle_neuron_child_after_fork()` track initialization and bad-fork state.

## Non-obvious behaviors worth keeping

- **Environment and bug-report collection:**
  `python -m torch_neuronx.collect_env` reports Python and package versions,
  OS and driver data, instance and logical NeuronCore configuration, and whether
  PyTorch can use a Neuron device.
- **`set_rng_state` syncs Philox state to CPU** (guarded against
  `FakeTensorMode`) so CPU-fallback RNG respects DTensor's offset. On-device
  GPSIMD RNG state goes through `torch_neuronx/python_ops/nki_kernels/rng_state.py`.
- **AMP supported dtypes**: `[float32, float16, bfloat16]` - no float64.
- **`NeuronConfig.custom_profiler_config` must start with `neuron_profile:`** -
  `Profiler.cpp:hasNeuronProfileKey()` uses an `rfind(..., 0) == 0` prefix
  check, so ordering of the prefix matters.
- **`NeuronProfiler.export_trace`** writes the Chrome trace into the NRT
  instance output dir so PyTorch + NRT traces land together.
- **`global_version_hash()`** in `torch_neuronx/utils.py` hashes top-level
  package Python files; lowering sources under `python_ops`,
  `neuron_dynamo_backend`, `distributed`, `kernels`, and `csrc`; vendored
  torch-mlir sources; relevant package versions; and
  `TORCH_NEURONX_INT64_MODE`. It salts the in-memory cache key and persistent
  HLO cache key. The content-addressed persistent NEFF key does not use it; the
  legacy base-key mode does.

## Authoritative human docs (cite, don't restate)

- Memory + metrics API tour (`memory_stats` flat/nested, `peak_memory_stats`,
  `segment_pool_analysis`, counters/reports, snapshot calls) ->
  `docs/statistics/torch_neuronx_stats_readme.md`.
- NKI kernel registration workflow -> `docs/cpp_dispatch_onboarding_guide.md`.
- NUMA pinning / `neuron-numa-launch` -> `docs/numa_optimization.md`.
- Env-var reference -> `docs/environment_variables.md`.

## See also

- `docs/AI-dynamo-backend.md` - `torch.compile(backend="neuron")`.
- `docs/AI-python-ops.md` - per-op implementation registration.
- `docs/AI-csrc.md` - the C++ side behind `_C`.
