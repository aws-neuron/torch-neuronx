//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "torch-mlir/Conversion/TorchToNeuron/TorchToNeuron.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

#ifdef TORCH_MLIR_ENABLE_STABLEHLO
#include "stablehlo/transforms/Passes.h"
#include "torch-mlir/Conversion/TorchToStablehlo/ChloToStablehloCustomLowering.h"
#include "torch-mlir/Conversion/TorchToStablehlo/StablehloCustomLegalization.h"
#include "torch-mlir/Conversion/TorchToStablehlo/TorchToStablehlo.h"
#include "torch-mlir/Conversion/TorchToStablehlo/TorchToStablehloCustomLowering.h"
#endif

using namespace mlir;
using namespace mlir::torch;

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace reg {
#define GEN_PASS_REGISTRATION
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h.inc"
} // end namespace reg

void mlir::torch::registerTorchConversionPasses() {
  reg::registerPasses();
  mlir::PassPipelineRegistration<>(
      "torch-raw-to-neuron-backend-pipeline",
      "Pipeline lowering torch backend contract to Neuron backend "
      "contract.",
      TorchConversion::createTorchBackendToNeuronBackendPipeline);
#ifdef TORCH_MLIR_ENABLE_STABLEHLO
  mlir::PassPipelineRegistration<
      TorchConversion::StablehloBackendPipelineOptions>(
      "torch-backend-to-stablehlo-backend-pipeline",
      "Pipeline lowering torch backend contract to StableHLO backend "
      "contract.",
      TorchConversion::createTorchBackendToStablehloBackendPipeline);
#endif
}

void TorchConversion::createTorchBackendToNeuronBackendPipeline(
    OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createConvertTorchToNeuronPass());
}

#ifdef TORCH_MLIR_ENABLE_STABLEHLO
void TorchConversion::createTorchBackendToStablehloBackendPipeline(
    OpPassManager &pm,
    const TorchConversion::StablehloBackendPipelineOptions &options) {
  // Generate Stablehlo & Chlo ops.
  pm.addNestedPass<func::FuncOp>(createConvertTorchToStablehloPass(
      options.enableStaticShape, options.enableI32Index));
  // Lowering Chlo ops to Stablehlo
  pm.addNestedPass<func::FuncOp>(
      createConvertChloToStablehloCustomLoweringPass());
  pm.addNestedPass<func::FuncOp>(
      stablehlo::createChloLegalizeToStablehloPass());
  // Lowering remained ops to Arith

  // Clean up any non-canonical code introduced above..
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  // The resolution of `dim` ops tends to create identical ops. CSE them.
  pm.addNestedPass<func::FuncOp>(createCSEPass());

  // Finish the type conversion from `torch` types to the types of the
  // StableHLO backend contract.
  pm.addPass(
      TorchConversion::createFuncBackendTypeConversionForStablehloPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(
      TorchConversion::createFinalizingBackendTypeConversionForStablehloPass());

  // Verify that we have lowered to Stablehlo ops.
  pm.addPass(TorchConversion::createVerifyStablehloBackendContractPass());

  // Canonicalize Stablehlo dynamic ops to static ops
  pm.addNestedPass<func::FuncOp>(
      stablehlo::createStablehloCanonicalizeDynamismPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(stablehlo::createStablehloRefineShapesPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(
      stablehlo::createStablehloCanonicalizeDynamismPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  // Legalize deprecated ops to Stablehlo ops
  stablehlo::StablehloLegalizeDeprecatedOpsPassOptions stablehloOptions;
  stablehloOptions.failOnUnusedOps = false;
  pm.addNestedPass<func::FuncOp>(
      stablehlo::createStablehloLegalizeDeprecatedOpsPass(stablehloOptions));
  pm.addPass(createCanonicalizerPass());

  // Custom legalization/lowering passes
  pm.addNestedPass<func::FuncOp>(createStablehloCustomLegalizationPass());
  pm.addNestedPass<func::FuncOp>(
      createConvertTorchToStablehloCustomLoweringPass());
}
#endif
