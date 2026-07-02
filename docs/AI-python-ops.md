# Python Ops (`torch_neuronx/python_ops/`)

The **eager-mode** per-op implementations that run when you invoke an ATen
op on a Neuron tensor outside `torch.compile`. Two backends behind a common
`Operation` abstraction: **MLIR / torch-mlir / StableHLO** (`python_ops/torch_mlir/`)
and **NKI kernels** (`python_ops/nki_kernels/`). The legacy XLA scribe
backend was removed — everything lowers through torch-mlir (vendored
in-tree, see `AI-build-infra.md`).

> **Navigation → codegraph.** For the `Operation`/`OperationImplementation`
> class hierarchy, the `@neuron_op` / `@register_aten` decorators, the
> per-op-family files under `torch_mlir/ops/`, and the NKI kernel list, ask
> codegraph:
> `codegraph_explore "Operation OperationImplementation neuron_op register_aten register_python_operations bounce_64bit"`.
> File map, method signatures, and op-file enumeration are recoverable
> there. This doc keeps the dispatch contract, the cast/complex policies,
> and authority routing.

## Dispatch contract

ATen op on a Neuron tensor → PyTorch `PrivateUse1` key → **C++ fast path**
(`TORCH_NEURONX_CPP_DISPATCH=1`, default: NKI if available, else MLIR
`TorchMlirKernel`) **or** Python path (`=0`: `Operation._implementations`
priority-sorted, first `can_handle()` wins, else CPU fallback). Hot-path ops
(gather, scatter_add, SDPA, nonzero, index) are registered via
`_C._register_neuron_op` so C++ handles stride fix-up + output packing
without Python overhead.

**Registration order at import** (`register_python_operations()`):
1. `_register_common_ops()` (gather/scatter_add/nonzero/index/MHA/SDPA).
2. `register_mlir_ops()` → `auto_register_neuron_ops()` →
   `initialize_torch_mlir_backend()`.
3. `register_cpu_fallback_ops()` (linalg CPU stubs).

`@register_aten` queues into `_PENDING_OPERATIONS`, drained at init.
`@neuron_op` priority-wraps subclasses at module level (for pickle + Dynamo
`TYPE_MATCH` guard) and calls `aten_lib.impl(name, wrapper, "PrivateUse1")`.

`OperationImplementation.execute()` applies in order: (1)
`_cast_float64_to_float32`, (2) empty-tensor short-circuit, (3) `_execute_impl`,
(4) post-process (restore float64, handle `out=`). Ops opt out via
`DISABLE_DTYPE_AUTOCAST` / `DISABLE_EMPTY_SHORTCIRCUIT`.

## Cast & dtype policy (contract)

- **`bounce_64bit(dtype)`** → `True` for `float64` always; `True` for `int64`
  **only in `cpu_bounce` mode**. In the default **packed** mode int64
  transfers zero-copy as int32 lane-pairs (see `int64_policy.py` +
  `docs/packed_int64_design.md`, authoritative).
- **`DtypeAutocastContext`** promotes `float64`→`float32` (warns once per
  cast site). int64 promotion to int32 applies only in `cpu_bounce`/`native`,
  not packed mode.
- **Complex dtype allowlist** — complex tensors are **not** generally
  on-device. `NeuronDispatcher.cpp` holds a small static `kComplexSupportedOps`
  allowlist: currently the elementwise mul/div family (`mul.Tensor`,
  `mul.Scalar`, `mul.out`, `mul.Scalar_out`, `mul_`, `div.Tensor`, `div.out`,
  `div.out_mode`, `div_.Tensor`). Everything else with complex inputs →
  CPU fallback. `view_as_complex`/`view_as_real` are native zero-copy views.
- **`ensure_contiguous_on_device`** calls an internal contiguous op (avoids
  redispatch recursion); `validate_neuron_memory_format` allows
  contiguous/preserve/channels_last, raises otherwise.

**`nonzero` is not a preprocessing op** — it uses its own `NonzeroMLIRImpl`
registered via `_C._register_neuron_op` directly, sidestepping `@register_aten`.

**CPU fallback** has three cooperating layers: `cpu_fallback_registrations.py`
(linalg stubs, copy-to-CPU preserving strides via `as_strided`), the
`Operation` fallback chain, and C++ `CPUFallbackExecutor`. Gated by
`TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS`.

## Adding a new op (recipe)

1. Pick backend: MLIR (common) or NKI.
2. MLIR: `@register_aten("aten::my_op")` in `torch_mlir/ops/`. NKI: kernel in
   `nki_kernels/`, wrap with `NKITorchMlirOpImpl`. `@neuron_op` path: decorate
   + implement `can_handle`/`_execute_impl`.
3. For C++ dispatch: `_C._register_neuron_op(...)` in `__init__.py`.
4. Test under `tests/python_ops/test_<op>.py`.
5. See `torch_mlir/README.md` and `torch_mlir/ops/README.md`.

## Authoritative human docs (cite, don't restate)

- C++ dispatch flow, `Preprocessing` protocol (`SHAPE_ONLY` vs
  `VALUE_DEPENDENT`, the 3/4-tuple contract, `.out`/inplace handling), and
  ATen NKI override onboarding → `docs/cpp_dispatch_onboarding_guide.md`.
- Which ATen ops are supported / CPU-fallback / native →
  `docs/torch_api_compatibility.md` and `docs/ops.html`.
- `FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS` semantics → `docs/environment_variables.md`.

## See also

- `AI-dynamo-backend.md` — shares the MLIR lowering + decompositions.
- `AI-csrc.md` — C++ `NeuronDispatcher`, `_register_neuron_op`, `CPUFallbackExecutor`.
- `AI-runtime-api.md` — `_C.compile_graph` / `_C.execute_compiled_graph`.
