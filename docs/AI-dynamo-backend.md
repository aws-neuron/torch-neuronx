# Dynamo Backend (`torch_neuronx/neuron_dynamo_backend/`)

Runs when you write `torch.compile(model, backend="neuron")`. Registered
via the `torch_dynamo_backends` → `neuron` entry point in `pyproject.toml`.

> **Navigation → codegraph.** For the compile pipeline as source — the
> `backend.py` → `compile.py` → `executor.py` flow, the FX-pass classes, the
> decomposition table — ask codegraph:
> `codegraph_explore "neuron_backend_fx_compiler CompileGraph Executor post_aot_processing convert_fx_to_stablehlo"`.
> The pipeline diagram, file map, FX-pass enumeration, and `Executor`
> step list are all recoverable that way. This doc keeps the gated
> behaviors, the cache-key formula, and the fallback contract.

## Pipeline (orientation)

`Dynamo → AOTAutograd → preprocess_graph → neuron_backend_fx_compiler`
(runs FX passes via `post_aot_processing`) `→ convert_fx_to_stablehlo`
(via vendored `torch_mlir`'s `FxImporter`) `→ CompileGraph.compile`
(`→ _C.compile_graph`) `→ Executor` (NEFF exec + dtype casts). Supported
`torch.compile` params are constrained (`fullgraph=True` recommended,
`dynamic=False`, `mode="default"`); full table in `README.md` (authoritative).

## Contracts & gated behaviors

**`CompileGraph` cache key** = `SHA256(mlir_bytes)`. It first strips non-deterministic
MLIR attrs (e.g. `torch.debug_dump_path`) so keys are consistent across ranks. With
`TORCH_NEURONX_NEFF_KEY_FROM_HLO=0` the key gains a `':' + global_version_hash()`
salt; see `AI-compilation-kernels.md`.

**Eager fallback defaults OFF.** `TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION`
is `setdefault`-ed to `"1"` at import, so the backend **re-raises by default**
instead of silently running the eager `GraphModule` (silent fallback was
masking real failures as false test passes). Set `=0` to re-enable the two
degradation paths (StableHLO-conversion failure, NEFF-execution failure).
The shared fallback fn replays the graph with the device-mesh placeholders +
shard-index input the compiler removed and re-inserts `None` outputs, so its
I/O matches the compiled path.

**`Executor` I/O dtype handling.** `_prepare_inputs` optionally auto-copies
CPU tensors to Neuron (off via `TORCH_NEURONX_DYNAMO_DISABLE_CPU_AUTOCOPY=1`),
forces contiguity, downcasts `int64`/`uint64`→`int32`/`uint32` via `view()`
and `float64`→`float32` via `.to()`; `_process_outputs` restores 64-bit
dtypes and re-inserts `None` outputs. Aliased/pass-through outputs reuse the
input buffer.

**SDPA decomposition.** `scaled_dot_product_fused_attention_overrideable`
has an NKI flash-attention fast path gated by `TORCH_NEURONX_ENABLE_NKI_SDPA`
(default on when `seqlen % 2048 == 0`). `native_batch_norm` family is
**unconditionally** decomposed through the standard FX→StableHLO path.

**MegaCache / precompile.** `NeuronOutputCode` **embeds StableHLO bytes
directly** (`hlo_bytes`) so a warm `caching_precompile` load is
self-contained — it re-invokes `compile_graph` with the embedded bytes
(hitting the persistent NEFF cache) without depending on the ephemeral
`/tmp/hlo_cache/`. The old `hlo_cache_key` field is a deprecated fallback.
Neither path retraces or recompiles. `NeuronNeffCacheArtifact` teaches
`save/load_cache_artifacts()` cross-machine NEFF sharing.

**FX passes that encode contracts** (not just transforms): `aliasing_analysis`
sets `input_output_alias` (KV-cache correctness); `dynamic_shape_analysis`
**bails out** if symbolic ints remain (dynamic shapes unsupported);
`vocab_sharding_spmd` SPMD-izes vocab-sharded embeddings;
`functionalize_copy_inplace_result` hoists `copy_` mutations so the compiler
can alias I/O. Use `settings.py` typed getters, not raw `os.environ`.

## Authoritative human docs (cite, don't restate)

- Decomposition authoring guide → `docs/decompositions.md`.
- Full env-var table + artifact layout → `docs/environment_variables.md`.
- Supported `torch.compile` params → `README.md`.

## See also

- `AI-python-ops.md` — shared MLIR lowering + NKI impls.
- `AI-compilation-kernels.md` — the NEFF cache, the content-addressed persistent key,
  and the `NEFF_KEY_FROM_HLO` gate.
- `AI-csrc.md` — `compile_graph` / `execute_compiled_graph` bindings.
- `AI-distributed.md` — how collectives survive the pipeline.
