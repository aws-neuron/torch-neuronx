# Vendored torch-mlir

Source: https://github.com/aws-neuron/torch-mlir

This is a vendored subset of torch-mlir containing only the files needed by
torch-neuronx — the `TorchToNeuron` + `TorchToStablehlo` lowering path. The
full torch-mlir repo includes backends (Tosa, Linalg, RefBackend) and tools
(ONNX importer, torch-mlir-opt) that are not used here and are not vendored.

The vendored commit is pinned in **`TORCH_MLIR_UPSTREAM_SHA`** (one-line file,
the single source of truth — this README does not duplicate it).

## What's included

- **Dialects**: `Torch`, `TorchConversion` (IR + Transforms).
- **Conversions**: `TorchToStablehlo`, `TorchToNeuron` (Neuron-specific,
  owned here — not upstream), and `Utils`.
- **CAPI** (`lib/CAPI/`): C API wrappers for dialect registration, types,
  ops, transforms.
- **Python** (`python/torch_mlir/`): `fx.py`, `compiler_utils.py`, `ir.py`,
  `passmanager.py`, `execution_engine.py`, `extras/fx_importer.py` (extended
  by `torch_neuronx`'s `NeuronFxImporter`), `extras/fx_decomp_util.py`,
  dialect bindings, generated ops files.
- **Pybind** (`python/TorchMLIRModule.cpp`, `csrc/`): the `_torchMlir`
  nanobind extension.

## What's excluded

- Conversions: `TorchToTosa`, `TorchToLinalg`, `TorchToArith`, `TorchToSCF`,
  `TorchToTensor`, `TorchConversionToMLProgram`, `TorchToTMTensor`,
  `TorchOnnxToTorch`.
- The `TMTensor` dialect.
- `RefBackend` runtime, and the `_mlirExecutionEngine.so` /
  `libmlir_*_runner_utils.so` native libs (no longer built or shipped).
- `projects/` (pt1, ltc, jit_ir_common, onnx_c_importer).
- `tools/` (torch-mlir-opt, lsp-server), `test/`, `test_pytorch/`,
  `build_tools/`, `docs/`.

## Updating from upstream

There is **no sync script** — the sync is a manual file copy of the vendored
subset above.

1. Copy the vendored files from `aws-neuron/torch-mlir` at the new commit;
   update `TORCH_MLIR_UPSTREAM_SHA`.
2. Review the diff and re-apply Neuron-specific patches if upstream touched
   them — notably `lib/Conversion/TorchToNeuron/`, which is owned here.
3. **Re-pin LLVM + StableHLO** in the repo's top-level `WORKSPACE` to whatever
   commits the new torch-mlir rev expects (see upstream's
   `externals/llvm-project` and `externals/stablehlo` at the new SHA). Update
   `LLVM_COMMIT` / `LLVM_SHA256` and `STABLEHLO_COMMIT`.
4. Rebuild: `./tools/build` (produces `_mlir.so`, `_torchMlir.so`,
   `_stablehlo.so`, `_C.so`).
5. Verify: `./tools/run-torch-mlir-lit-tests && ./tools/run-torch-mlir-e2e-tests`,
   then `./tools/run-test`.
