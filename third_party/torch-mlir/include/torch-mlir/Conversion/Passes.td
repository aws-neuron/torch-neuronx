//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_CONVERSION_PASSES
#define TORCHMLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Torch conversions
//===----------------------------------------------------------------------===//

def ConvertTorchToNeuron : Pass<"convert-torch-to-neuron", "func::FuncOp"> {
  let summary = "Convert Torch ops to Neuron backend";
  let description = [{
    Convert Torch ops to Neuron backend operations.
  }];
  let constructor = "mlir::torch::createConvertTorchToNeuronPass()";
}

#ifdef TORCH_MLIR_ENABLE_STABLEHLO
def ConvertTorchToStablehlo : Pass<"convert-torch-to-stablehlo", "func::FuncOp"> {
  let summary = "Convert Torch ops to Stablehlo ops";
  let description = [{
    Convert Torch ops to Stablehlo ops.
  }];
  let constructor = "mlir::torch::createConvertTorchToStablehloPass()";

  // Specify any options.
  let options = [
    Option<"enableStaticShape", "enable-static-shape", "bool", /*default=*/"false",
           "Enable static shape conversion">,
    // The i64 calculation is much slower than i32 on some devices, such as
    // Nvidia GPU. One can truncate from i64 to i32 since dimension sizes
    // are unlikely to exceed the range of i32(4GiB)
    Option<"enableI32Index", "enable-i32-index", "bool", /*default=*/"false",
           "Enable truncate index from i64 to i32(unsafely)">,
  ];
}

def ConvertTorchToStablehloCustomLowering : Pass<"convert-torch-to-neuron-custom-lowering", "func::FuncOp"> {
   let summary = "Additional custom lowerings to stablehlo";
   let description = [{
     Additional custom lowerings run at the end of the torch->stablehlo pipeline for compatibility with neuronx-cc
   }];
   let constructor = "mlir::torch::createConvertTorchToStablehloCustomLoweringPass()";
 }

def StablehloCustomLegalization : Pass<"stablehlo-custom-legalization", "func::FuncOp"> {
   let summary = "Custom legalization for StableHLO scatter ops with integer types";
   let description = [{
     This pass legalizes StableHLO scatter operations that use integer types by
     upcasting them to F32, performing the operation, and then converting back.
     This is extracted from the legalizeScatterTypes function to be a standalone pass.
   }];
   let constructor = "mlir::torch::createStablehloCustomLegalizationPass()";
   let dependentDialects = ["::mlir::stablehlo::StablehloDialect"];
 }

def ConvertChloToStablehloCustomLowering : Pass<"convert-chlo-to-stablehlo-custom-lowering", "func::FuncOp"> {
   let summary = "Preliminary custom lowerings from chlo to stablehlo";
   let description = [{
     This pass adds chlo to stablehlo lowerings meant to preempt rewrite rules
     in the official stablehlo-provided chlo to stablehlo pass.
   }];
   let constructor = "mlir::torch::createConvertChloToStablehloCustomLoweringPass()";
 }

#endif

#endif // TORCHMLIR_CONVERSION_PASSES
