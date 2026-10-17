//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/InitAll.h"

#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/IR/Dialect.h"
#include "torch-mlir/Conversion/Passes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"

#ifdef TORCH_MLIR_ENABLE_STABLEHLO
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/optimization/Passes.h"
#endif

void mlir::torch::registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::torch::Torch::TorchDialect>();
  registry.insert<mlir::torch::TorchConversion::TorchConversionDialect>();
#ifdef TORCH_MLIR_ENABLE_STABLEHLO
  registry.insert<mlir::stablehlo::StablehloDialect>();
  registry.insert<mlir::chlo::ChloDialect>();
#endif
}

void mlir::torch::registerAllExtensions(mlir::DialectRegistry &registry) {
  mlir::func::registerInlinerExtension(registry);
  tensor::registerInferTypeOpInterfaceExternalModels(registry);
}

// TODO: Break this up when backends are separated.
void mlir::torch::registerOptionalInputDialects(
    mlir::DialectRegistry &registry) {
  registry.insert<complex::ComplexDialect, linalg::LinalgDialect,
                  math::MathDialect, memref::MemRefDialect, scf::SCFDialect,
                  shape::ShapeDialect, sparse_tensor::SparseTensorDialect,
                  tensor::TensorDialect>();
}

void mlir::torch::registerAllPasses() {
  mlir::torch::registerTorchPasses();
  mlir::torch::registerTorchConversionPasses();
  mlir::torch::registerConversionPasses();

#ifdef TORCH_MLIR_ENABLE_STABLEHLO
  mlir::stablehlo::registerStablehloLegalizeToLinalgPass();
  mlir::stablehlo::registerStablehloAggressiveSimplificationPass();
  mlir::stablehlo::registerStablehloRefineShapesPass();
  mlir::stablehlo::registerStablehloConvertToSignlessPass();
  mlir::stablehlo::registerShapeLegalizeToStablehloPass();
  mlir::stablehlo::registerStablehloLegalizeDeprecatedOpsPass();
#endif
}
