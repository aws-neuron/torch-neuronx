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
> The `kernels/` file tree, the `execute_neff` / `get_or_compile` step flows,
> the `CompilerConfig` dataclass fields, and the `CompilationJob` artifact-path
> table are all recoverable that way. This doc keeps only what codegraph can't
> express: defaults, gated behaviors, multi-process invariants, and which human
> doc is authoritative.

## Contracts

**NEFF cache is two-tier and multi-process safe.** `CompilerSubprocess.get_or_compile`
guards compilation with: NFS-cache check → `FileLock` at
`{local_cache_dir}/{hash}/locks/compile.lock` → **double-check NFS under the
lock** (another process may have just compiled) → compile → **atomic write**
(temp file + rename, so concurrent readers never see a partial NEFF) →
processes blocked on the lock read the finished NFS path *outside* the lock.
Lock timeout is `TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT` (default in
`docs/environment_variables.md`). This
ordering is the correctness contract — don't reorder the double-check or the
atomic write.

**Cache-key gate `TORCH_NEURONX_NEFF_KEY_FROM_HLO`.** Controls whether the key
includes a compiler/version salt:
- **Python / Dynamo** (`neuron_dynamo_backend/compile.py`, `CompileGraph.compile`):
  off (**default**) = `sha256(mlir_bytes) + ':' + global_version_hash()`;
  on = `sha256(mlir_bytes)` alone, so benign Python edits that don't change the
  lowered HLO still hit the on-disk cache.
- **C++ persistent** (`CompilationCacheKey`): on = HLO bytes + compiler config
  (compiler version from `NEURONXCC_VERSION` via
  `PlatformUtils::GetCompilerVersion()`); off = in-memory base key.

**`NEFFCacheKey` = HLO hash ⊕ config hash.** `_hash = sha256("{hlo_hash}:{config_hash}")`,
where `config_hash` folds in `target|model_type|opt_level|auto_cast|framework|lnc|version|sorted_flags`.
The split is deliberate: when only compiler flags change, `_hlo_hash` stays
constant while `_config_hash` moves — useful when debugging a spurious cache
miss. `CompilerConfig` defaults (`-O1`, `auto_cast="none"`, `model_type="transformer"`)
live in the dataclass.

## Integration — who feeds this layer

Three lowering paths converge on the same NEFF cache:
- **Eager / torch_mlir** — `python_ops/torch_mlir/kernel.py::TorchMlirKernel`
  subclasses `BaseNeuronKernel`; `_get_or_compile_neff` builds a
  `CompilationJob`, calls `CompilerSubprocess`, stashes in `_neff_cache`.
- **Dynamo** — `CompileGraph.compile` computes its own content key and calls
  `_C.compile_graph`; the C++ `CompilationCache` → `LocalCompilationBackend`
  runs `neuronx-cc` via the wrapper.
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
- **Spurious cache miss**: compare `_hlo_hash` vs `_config_hash` — if only the
  config moved, a flag changed, not the graph.

## See also

- [AI-dynamo-backend.md](AI-dynamo-backend.md) — `CompileGraph` entry point.
- [AI-csrc.md](AI-csrc.md) — C++ `CompilationCache`, `LocalCompilationBackend`,
  `PersistentCacheBackend` this layer sits on.
- [AI-runtime-api.md](AI-runtime-api.md) — the `_C._nrt_*` bindings
  `execute_neff` calls.
