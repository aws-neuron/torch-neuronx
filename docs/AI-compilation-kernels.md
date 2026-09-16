# Compilation Kernels (`torch_neuronx/kernels/`)

Python glue between the ATen dispatcher / Dynamo backend and the
`neuronx-cc` compiler — **"how HLO turns into NEFF and gets cached."**
Provides the base kernel abstraction, a two-tier file-locked NEFF cache,
a CLI wrapper that makes direct `neuronx-cc` calls cached, and artifact
directory management.

> **Navigation → codegraph.** For the code structure, signatures, and call
> graphs of this package, ask codegraph (it returns current verbatim source —
> never stale):
> `codegraph_explore "BaseNeuronKernel CompilerSubprocess get_or_compile NEFFCacheKey CompilationJob neuronx_cc_wrapper"`.
> The `kernels/` file tree, the `get_or_compile` step flow,
> the `CompilerConfig` dataclass fields, and the `CompilationJob` artifact-path
> table are all recoverable that way. This doc keeps only what codegraph can't
> express: defaults, gated behaviors, multi-process invariants, and which human
> doc is authoritative.

## Contracts

**NEFF cache is two-tier and multi-process safe.** `CompilationCache` guards
compilation with: in-memory check → persistent-cache check → `FileLock` on the
persistent key → **double-check the persistent cache under the lock** (another
process may have just compiled) → compile → **atomic write** (temp file + rename, so
concurrent readers never see a partial NEFF). Lock timeout is
`TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT` (default in
`docs/environment_variables.md`). This ordering is the correctness contract — don't
reorder the double-check or the atomic write.

**The persistent NEFF key is content-addressed.**
`CompilationCacheKey::ComputePersistentKey` = `XXH3_128(hlo_bytes || 0xFF || compiler_config)`,
where `compiler_config` = `BuildCompilerExtension()` =
`platform_target_lnc_optlevel_compilerargs_v<compilerversion>[_wlo]` (compiler version
from `NEURONXCC_VERSION` via `PlatformUtils::GetCompilerVersion()`). A change in the
lowering, the compiler, or the IR gives different bytes and a different key, so
invalidation needs no source salt.

`TORCH_NEURONX_NEFF_KEY_FROM_HLO=0` selects the legacy key
(`XXH3_128(base_key || 0xFF || compiler_config)`), which breaks on any source or version
change because `base_key` carries `global_version_hash()`. Use it only to diagnose a
key problem. The Python gate in `neuron_dynamo_backend/compile.py::_content_key` must
stay identical to `utils::IsHloContentKeyEnabled()`.

**An IR-less kernel resolves its key from a hint.** `NeffDirectKernelExecution` carries
no IR, so it cannot recompute the content key. `CompilationCache` keeps an in-memory
`base_key -> persistent_key` map that the compile step records. LRU eviction drops the
NEFF but keeps the hint, so a later execution still finds the NEFF on disk.

**Two frontends do NOT share a NEFF today**, even for identical logical graphs. Two
things differ: the serialization (eager `str(module).encode()`, dynamo
`write_bytecode`) and the compiler config that the key folds in (eager
`GetXLAAdditionalArgs` + `-O1`; torch.compile
`--model-type transformer --auto-cast=none` + `NEURON_CC_FLAGS` + `-O2`). Sharing needs
both to agree.

**`torch_neuronx/kernels/compiler_subprocess.py` and `neuronx_cc_wrapper.py` are dead.**
`NEFFCacheKey` and `CompilerSubprocess.get_or_compile` have no production caller;
`NeuronCompiler.cpp` runs `neuronx-cc` directly. Do not treat them as the NEFF cache.

## Integration — who feeds this layer

Three lowering paths converge on the same NEFF cache:
- **Eager / torch_mlir** — `python_ops/torch_mlir/kernel.py::TorchMlirKernel`
  subclasses `BaseNeuronKernel` and `lower()`s the op to StableHLO, then submits
  it to the C++ `OperationExecutionEngine`, which drives compilation and NEFF
  caching through the C++ `CompilationCache` (see `AI-csrc.md`).
- **Dynamo** — `CompileGraph.compile` computes its own content key and calls
  `_C.compile_graph`; the C++ `CompilationCache` → `LocalCompilationBackend` →
  `NeuronCompiler` runs the `neuronx-cc` executable directly.
- **NKI HOP** — `nki_hop.py::TorchNeuronNKIKernelV3.CompileKernel` flows through
  the same C++ `compile_graph`.

**Persistent HLO cache** (`python_ops/compilation/hlo_cache.py`) complements the
NEFF cache: stores StableHLO bytes + IO spec so AOTAutograd's
`caching_precompile` can rebuild an `Executor` without re-tracing.

## Authoritative human docs (cite, don't restate)

- **All NEFF / HLO cache env vars** (`..._NEFF_CACHE_DIR`, `..._HLO_CACHE_DIR`,
  `..._NEFF_LOCAL_CACHE_DIR`, `..._NEFF_DISABLE_CACHE`, `..._NEFF_CACHE_LOCK_TIMEOUT`,
  `..._NEFF_KEY_FROM_HLO`) and the **artifact directory layout** under
  `TORCH_NEURONX_DEBUG_DIR` → `docs/environment_variables.md`
  ("Persistent NEFF / HLO Cache" + "Artifact Directory Structure").

## Debugging tips

- **Preserve artifacts**: `TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS=1`, then
  inspect `neuronx-cc/{hash}/log-neuron-cc.txt`.
- **Force recompile**: `TORCH_NEURONX_NEFF_DISABLE_CACHE=1`.
- **Stale lock**: a crashed compile leaves `{local_cache_dir}/{hash}/locks/compile.lock`;
  it blocks others for up to the lock timeout.
- **Spurious cache miss**: the persistent key hashes the IR bytes and
  `BuildCompilerExtension()`. Compare the two across the runs: if the IR bytes match,
  a compiler flag, the platform, the LNC count, or the compiler version moved.
- **Suspected key problem**: `TORCH_NEURONX_NEFF_KEY_FROM_HLO=0` selects the legacy
  key. If that changes the behavior, the content key is the cause.

## See also

- [AI-dynamo-backend.md](AI-dynamo-backend.md) — `CompileGraph` entry point.
- [AI-csrc.md](AI-csrc.md) — C++ `CompilationCache`, `LocalCompilationBackend`,
  `PersistentCacheBackend` this layer sits on.
- [AI-runtime-api.md](AI-runtime-api.md) — the `_C._nrt_*` runtime bindings the
  C++ execution engine drives.
