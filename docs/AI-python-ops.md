# Python Ops (`torch_neuronx/python_ops/`)

The **eager-mode** per-op implementations that run when you invoke an ATen
op on a Neuron tensor outside `torch.compile`. Two backends behind a common
`Operation` abstraction: **MLIR / torch-mlir / StableHLO** (`torch_neuronx/python_ops/torch_mlir/`)
and **NKI kernels** (`torch_neuronx/python_ops/nki_kernels/`). The legacy XLA scribe
backend was removed - everything lowers through torch-mlir (vendored
in-tree, see `docs/AI-build-infra.md`).

> **Source navigation.** Run this command from the repository root:
>
> `rg -n 'Operation|OperationImplementation|neuron_op|register_aten|register_python_operations|bounce_64bit' torch_neuronx/python_ops torch_neuronx/csrc/core/dispatch`
>
> If `rg` is unavailable, use `git grep -n -E` with the same pattern and paths.

## Dispatch contract

ATen op on a Neuron tensor -> PyTorch `PrivateUse1` key -> **C++ fast path**
(NKI if available, else MLIR `TorchMlirKernel`). An op with no lowerable kernel
is registered via `aten_lib.impl` instead and resolves through
`Operation._implementations`, priority-sorted, first `can_handle()` wins, else
CPU fallback. Hot-path ops
(gather, scatter_add, SDPA, nonzero, index) are registered via
`_C._register_neuron_op` so C++ handles stride fix-up + output packing
without Python overhead.

**Registration order at import:** `register_python_operations()` calls
`register_mlir_ops()`. That function imports operation modules, calls
`_register_common_ops()`, snapshots pending torch-mlir registrations,
auto-registers `@neuron_op` operations, and calls
`initialize_torch_mlir_backend()`.

`@register_aten` queues into `_PENDING_OPERATIONS`, drained at init.
`@neuron_op` queues implementation classes. Auto-registration creates the
module-level priority wrappers needed for pickle and Dynamo `TYPE_MATCH`
guards. It uses `_C._register_neuron_op` for lowerable implementations on the
C++ path and `aten_lib.impl(..., "PrivateUse1")` on the Python path or for a
pure Python implementation.

`OperationImplementation.execute()` applies in order: (1)
`_cast_float64_to_float32`, (2) empty-tensor short-circuit, (3) `_execute_impl`,
(4) post-process (restore float64, handle `out=`). Ops opt out via
`DISABLE_DTYPE_AUTOCAST` / `DISABLE_EMPTY_SHORTCIRCUIT`.

## Cast & dtype policy (contract)

- **`bounce_64bit(dtype)`** -> `True` for `float64` always. For `int64`, it is
  `True` in `cpu_bounce` and `cpu_fallback`, and `False` in `native`,
  `native_strict`, and `packed`.
- **int64 defaults to `native` mode.** Its per-op cascade runs supported ops as
  S64, runs non-native `DOWNCAST` ops through the packed int32 view, and sends
  unsupported ops such as matmul and linear to CPU fallback. Convolution reaches
  dispatch as `aten::convolution` and runs through the packed view. `packed` is
  an explicit mode that forces every int64 op through the packed view.
  `torch_neuronx/int64_policy.py` is authoritative for the current default and
  routing policy. `docs/packed_int64_design.md` is design background that
  predates the `native` default.
- **`DtypeAutocastContext`** promotes `float64`->`float32` and warns once per
  cast site. The int64 policy selects whether an operation keeps S64,
  downcasts on device, or falls back to CPU.
- **Complex support** is explicit. `NativeComplexOpsAllowlist()` in
  `torch_neuronx/csrc/core/dispatch/DispatchUtils.cpp` contains the current
  native mul, div, and abs overloads. `StructuralComplexHandlers()` in
  `torch_neuronx/csrc/core/dispatch/StructuralComplexOps.cpp` handles
  `contiguous`, `gather`, `index_select`, and `index.Tensor` through an
  interleaved real view when complex decomposition is enabled.
  `view_as_complex` and `view_as_real` have native registrations in
  `torch_neuronx/csrc/core/NeuronBindings.cpp`. Other complex operations use
  registered decomposition or fallback policy; consult these lists instead of
  assuming general complex support.
- **`ensure_contiguous_on_device`** calls an internal contiguous op (avoids
  redispatch recursion); `validate_neuron_memory_format` allows contiguous,
  preserve, channels-last, and channels-last-3d formats, and raises otherwise.

**`nonzero` is not a preprocessing op.** On the default C++ path, its
`NonzeroMLIRImpl` is registered directly through `_C._register_neuron_op`,
sidestepping `@register_aten`. The Python dispatch path uses `aten_lib.impl`.

**`grouped_mm` is eager-only.** At import, `TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS=1`
selects the ragged `@neuron_op` implementation. Otherwise, a
`Preprocessing.VALUE_DEPENDENT` `@register_aten` factory pads unaligned group
offsets. The Dynamo graph-break patch breaks on both grouped-matrix multiply
entry points.

**CPU fallback** has two active implementation layers. The Python `Operation`
fallback chain tries registered implementations and then calls
`_unhandled_cpu_fallback` when none succeeds. C++ error recovery owns a
`CPUFallbackExecutor` and invokes it for eligible compilation-failure recovery.
The Python fallback restriction is controlled by
`TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS`.

## Adding a new op (recipe)

1. Pick backend: MLIR (common) or NKI.
2. For MLIR, use `@register_aten("aten::my_op")` under
   `torch_neuronx/python_ops/torch_mlir/ops/`. For NKI, add the kernel under
   `torch_neuronx/python_ops/nki_kernels/` and use the current wrapper pattern.
3. For C++ dispatch, add `_C._register_neuron_op(...)` in
   `torch_neuronx/python_ops/__init__.py` when the operation needs that path.
4. Add a test under `tests/python_ops/`.
5. Read `torch_neuronx/python_ops/torch_mlir/README.md`,
   `torch_neuronx/python_ops/torch_mlir/ops/README.md`, and
   `docs/cpp_dispatch_onboarding_guide.md`.

## Authoritative human docs (cite, don't restate)

- C++ dispatch flow, `Preprocessing` protocol (`SHAPE_ONLY` vs
  `VALUE_DEPENDENT`, the 3/4-tuple contract, `.out`/inplace handling), and
  ATen NKI override onboarding -> `docs/cpp_dispatch_onboarding_guide.md`.
- Which ATen ops are supported / CPU-fallback / native ->
  `docs/torch_api_compatibility.md` and `docs/ops.html`.
- `FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS` semantics -> `docs/environment_variables.md`.

## See also

- `docs/AI-dynamo-backend.md` - shares the MLIR lowering + decompositions.
- `docs/AI-csrc.md` - C++ `NeuronDispatcher`, `_register_neuron_op`, `CPUFallbackExecutor`.
- `docs/AI-runtime-api.md` - `_C.compile_graph` / `_C.execute_compiled_graph`.
