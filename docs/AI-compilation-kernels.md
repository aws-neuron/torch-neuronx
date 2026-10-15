# Compilation Kernels (`torch_neuronx/kernels/`)

Python glue between the ATen dispatcher / Dynamo backend and the
`neuronx-cc` compiler - **"how HLO turns into NEFF and gets cached."**
Provides the base kernel abstraction and describes the C++ file-locked NEFF
cache. The Python compiler wrapper remains for legacy and test coverage; the
production compile path is C++.

> **Source navigation.** Run this command from the repository root:
>
> `rg -n 'BaseNeuronKernel|CompilerSubprocess|get_or_compile|NEFFCacheKey|CompilationJob|neuronx_cc_wrapper' torch_neuronx/kernels torch_neuronx/python_ops torch_neuronx/neuron_dynamo_backend torch_neuronx/csrc`
>
> If `rg` is unavailable, use `git grep -n -E` with the same pattern and paths.

## Contracts

**NEFF cache has two tiers and normally coordinates compilers.** `CompilationCache`
uses an in-memory check, a persistent-cache check, `FileLock`, and a post-lock
persistent-cache check. It then compiles and uses an atomic write. Atomic writes
prevent readers from seeing a partial NEFF.

`TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT` controls the lock timeout. After this
timeout, `FileLock` replaces the lock path without checking if the owner process is
active. Two processes can compile the same key if a compile exceeds this timeout.
Set the timeout above the longest expected compile. Keep the post-lock check and
atomic write order.

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
key problem. The Python gate in `torch_neuronx/neuron_dynamo_backend/compile.py::_content_key` must
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

**The Python compiler wrapper is test and legacy code.**
`torch_neuronx/kernels/compiler_subprocess.py` and
`torch_neuronx/kernels/neuronx_cc_wrapper.py` have no production caller.
Production compilation reaches `NeuronCompiler` through the C++ cache and
kernel execution classes. Do not treat the Python wrapper as the active cache.

## Integration - who feeds this layer

The active paths converge on the C++ `CompilationCache`:
- **Eager / torch_mlir** -
  `torch_neuronx/python_ops/torch_mlir/kernel.py::TorchMlirKernel` lowers an op
  to StableHLO and submits it through `OperationExecutionEngine`.
- **Dynamo** - `CompileGraph.compile` calls `_C.compile_graph`.
  `CompilationCache` selects the service backend by default. Set
  `NEURON_COMPILE_SERVICE=0` to select `LocalCompilationBackend`. Service setup
  failure falls back to local compilation. The local backend invokes the
  kernel execution callback, whose `CompileToNeff` implementation calls
  `NeuronCompiler` and the `neuronx-cc` executable.
- **NKI HOP** - `torch_neuronx/nki_hop.py` defines `NKIKernelWrapper`,
  `nki_kernel_wrapper`, and `wrap_nki`. The HOP lowers through
  `torch_neuronx/python_ops/torch_mlir/nki_op_impl.py` during graph capture and
  then follows the normal graph compilation path.

**Persistent HLO cache** (`torch_neuronx/python_ops/compilation/hlo_cache.py`) complements the
NEFF cache: stores StableHLO bytes + IO spec so AOTAutograd's
`caching_precompile` can rebuild an `Executor` without re-tracing.

## Authoritative human docs (cite, don't restate)

- **All NEFF / HLO cache env vars** (`..._NEFF_CACHE_DIR`, `..._HLO_CACHE_DIR`,
  `..._NEFF_LOCAL_CACHE_DIR`, `..._NEFF_DISABLE_CACHE`, `..._NEFF_CACHE_LOCK_TIMEOUT`,
  `..._NEFF_KEY_FROM_HLO`) and the **artifact directory layout** under
  `TORCH_NEURONX_DEBUG_DIR` -> `docs/environment_variables.md`
  ("Persistent NEFF / HLO Cache" + "Artifact Directory Structure").

## Debugging tips

- **Preserve artifacts**: `TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS=1`, then
  inspect `neuronx-cc/{hash}/log-neuron-cc.txt`.
- **Force recompile**: `TORCH_NEURONX_NEFF_DISABLE_CACHE=1`.
- **Lock timeout**: `CompilationCache::GetLockPath` places each lock at
  `{local_cache_dir}/locks/{persistent_cache_key}.lock`. On timeout,
  `FileLock` deletes and recreates the lock path, then retries acquisition.
- **Spurious cache miss**: the persistent key hashes the IR bytes and
  `BuildCompilerExtension()`. Compare the two across the runs: if the IR bytes match,
  a compiler flag, the platform, the LNC count, or the compiler version moved.
- **Suspected key problem**: `TORCH_NEURONX_NEFF_KEY_FROM_HLO=0` selects the legacy
  key. If that changes the behavior, the content key is the cause.

## See also

- `docs/AI-dynamo-backend.md` - `CompileGraph` entry point.
- `docs/AI-csrc.md` - C++ `CompilationCache`, `LocalCompilationBackend`,
  `PersistentCacheBackend` this layer sits on.
- `docs/AI-runtime-api.md` - the `_C._nrt_*` runtime bindings the
  C++ execution engine drives.
