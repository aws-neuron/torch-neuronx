# Vendored torch-mlir

Source: https://github.com/aws-neuron/torch-mlir
Commit: c9eaae244b6973a0edae803d3ffcab70143e5cc4
Date: 2026-04-21

This is a vendored subset of torch-mlir containing only the files needed by
torch-neuronx. The full torch-mlir repo includes backends (Tosa, Linalg,
RefBackend) and tools (ONNX importer, torch-mlir-opt) that are not used here.

## What's included

- **Dialects**: Torch, TorchConversion, TMTensor (IR + Transforms)
- **Conversions**: TorchToStablehlo, TorchToNeuron, TorchToArith, TorchToSCF,
  TorchToTensor, TorchConversionToMLProgram, TorchToTMTensor, Utils
- **CAPI**: C API wrappers for dialect registration, types, ops, transforms
- **Python**: fx.py, compiler_utils.py, fx_importer.py, fx_decomp_util.py,
  dialect bindings, generated ops files
- **Pybind**: TorchMLIRModule.cpp (_torchMlir extension)

## What's excluded

- TorchToTosa, TorchToLinalg, TorchOnnxToTorch (full), RefBackend
- projects/ (pt1, ltc, jit_ir_common, onnx_c_importer)
- tools/ (torch-mlir-opt, lsp-server)
- test/, test_pytorch/, build_tools/, docs/

## Updating

1. Run `./sync_from_upstream.sh <new-commit>`
2. Review diffs, apply relevant changes
3. Update `TORCH_MLIR_UPSTREAM_SHA`
4. Build and test: `./tools/build`
