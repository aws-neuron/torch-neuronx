//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToStablehlo/StablehloCustomLegalization.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch-mlir/Conversion/Passes.h"
#include "torch-mlir/Conversion/TorchToStablehlo/StablehloLegalizeUtils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"

using namespace mlir;
using namespace mlir::torch;
namespace mlir::torch {

#define GEN_PASS_DEF_STABLEHLOCUSTOMLEGALIZATION
#include "torch-mlir/Conversion/Passes.h.inc"

namespace {

class ScatterLegalizationPattern
    : public OpRewritePattern<stablehlo::ScatterOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ScatterOp op,
                                PatternRewriter &rewriter) const override {
    auto inputs = op.getInputs();
    auto updates = op.getUpdates();
    if (inputs.size() == 0 || updates.size() == 0) {
      return rewriter.notifyMatchFailure(
          op, "stablehlo::ScatterOp must have non-empty inputs and updates");
    }

    auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    auto inputType = dyn_cast<RankedTensorType>(inputs[0].getType());
    auto updateType = dyn_cast<RankedTensorType>(updates[0].getType());
    if (!type || !inputType || !updateType) {
      return rewriter.notifyMatchFailure(
          op, "stablehlo::ScatterOp must have ranked tensor inputs, updates, "
              "and outputs");
    }

    if (inputType.getElementType() != updateType.getElementType()) {
      return rewriter.notifyMatchFailure(op,
                                         "stablehlo::ScatterOp must have the "
                                         "same dtype for inputs and updates");
    }

    // Only proceed if we have an integer type that needs upcasting
    auto intType = dyn_cast<IntegerType>(inputType.getElementType());
    if (!intType) {
      return success(); // Not an integer type, no need to legalize
    }

    // Upcast inputs and updates to F32
    SmallVector<Value> upcastedInputs;
    SmallVector<Value> upcastedUpdates;
    auto f32Type = rewriter.getF32Type();
    auto intermediateInputType =
        RankedTensorType::get(inputType.getShape(), f32Type);
    auto intermediateUpdateType =
        RankedTensorType::get(updateType.getShape(), f32Type);

    for (auto input : inputs) {
      auto convert = stablehlo::ConvertOp::create(rewriter, op.getLoc(),
                                                  intermediateInputType, input);
      upcastedInputs.emplace_back(convert);
    }

    for (auto update : updates) {
      auto convert = stablehlo::ConvertOp::create(
          rewriter, op.getLoc(), intermediateUpdateType, update);
      upcastedUpdates.emplace_back(convert);
    }

    auto intermediateScatterType =
        RankedTensorType::get(inputType.getShape(), f32Type);
    auto upcastedScatter = stablehlo::ScatterOp::create(
        rewriter, op.getLoc(), intermediateScatterType, upcastedInputs,
        op.getScatterIndices(), upcastedUpdates,
        op.getScatterDimensionNumbersAttr(), op.getIndicesAreSorted(),
        op.getUniqueIndices());

    // Copy the computation block from the original scatter op
    Region &originalRegion = op.getUpdateComputation();
    Region &newRegion = upcastedScatter.getUpdateComputation();

    if (!originalRegion.empty()) {
      Block &originalBlock = originalRegion.front();
      Block &newBlock = newRegion.emplaceBlock();

      // Add block arguments with F32 type - handle arbitrary number of
      // arguments
      auto blockArgumentType = RankedTensorType::get({}, f32Type);
      IRMapping valueMapping;

      // Map all block arguments from original to new block with proper types
      for (int64_t i = 0; i < originalBlock.getNumArguments(); ++i) {
        BlockArgument newArg =
            newBlock.addArgument(blockArgumentType, op.getLoc());
        valueMapping.map(originalBlock.getArgument(i), newArg);
      }

      // Clone operations from original block, converting types to F32
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(&newBlock);
        for (Operation &originalOp : originalBlock) {
          if (isa<stablehlo::ReturnOp>(originalOp)) {
            // Handle return op specially
            SmallVector<Value> newOperands;
            for (auto operand : originalOp.getOperands()) {
              auto mappedOperand = valueMapping.lookupOrDefault(operand);
              newOperands.push_back(mappedOperand);
            }
            stablehlo::ReturnOp::create(rewriter, op.getLoc(), newOperands);
          } else {
            // Clone other operations, converting result types to F32
            Operation *clonedOp = rewriter.clone(originalOp, valueMapping);

            // Update result types to F32 if they were originally integer types
            for (auto result : clonedOp->getResults()) {
              auto resultType = dyn_cast<RankedTensorType>(result.getType());
              if (resultType && resultType.getElementType() == intType) {
                auto newResultType =
                    RankedTensorType::get(resultType.getShape(), f32Type);
                result.setType(newResultType);
              }
            }
          }
        }
      }
    }

    // Convert results back to original type and replace the original op
    SmallVector<Value> finalResults;
    for (auto result : upcastedScatter.getResults()) {
      auto convertResult =
          stablehlo::ConvertOp::create(rewriter, op.getLoc(), type, result);
      finalResults.emplace_back(convertResult);
    }

    rewriter.replaceOp(op, finalResults);
    return success();
  }
};

/// Normalize non-standard bool (i1) function arguments.
///
/// On Neuron, bool tensors are stored as i8 internally. Non-standard values
/// (2, 3, 163, etc.) all mean True but the hardware treats raw bytes
/// literally during bool→numeric promotion. This keeps the function
/// signature as i1 (preserving type compatibility with producers) but
/// inserts normalization logic inside the function body:
///
///   func.func @main(%arg0: tensor<4xi1>) {
///     ... uses of %arg0 ...
///   }
///
/// into:
///
///   func.func @main(%arg0: tensor<4xi1>) {
///     %arg0_i8 = stablehlo.convert %arg0 : (tensor<4xi1>) -> tensor<4xi8>
///     %zero = stablehlo.constant dense<0> : tensor<4xi8>
///     %arg0_bool = stablehlo.compare NE, %arg0_i8, %zero
///     ... uses of %arg0_bool (replacing old %arg0) ...
///   }
///
static void normalizeBoolFuncArgs(func::FuncOp funcOp) {
  Block &entryBlock = funcOp.getBody().front();
  auto funcType = funcOp.getFunctionType();
  bool hasI1Args = false;

  for (auto argType : funcType.getInputs()) {
    auto tensorType = dyn_cast<RankedTensorType>(argType);
    if (tensorType && tensorType.getElementType().isInteger(1)) {
      hasI1Args = true;
      break;
    }
  }

  if (!hasI1Args)
    return;

  // NOTE: We do NOT change the function signature. The args stay as i1
  // so that producers (returning i1) and consumers (taking i1) have
  // matching types in the prologue/concatenation engine.

  OpBuilder builder(funcOp.getContext());
  builder.setInsertionPointToStart(&entryBlock);

  for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
    BlockArgument arg = entryBlock.getArgument(i);
    auto argType = dyn_cast<RankedTensorType>(funcType.getInput(i));
    if (!argType || !argType.getElementType().isInteger(1))
      continue;

    // Cast i1 → i8 to see the raw byte values
    auto i8Type = IntegerType::get(funcOp.getContext(), 8);
    auto i8TensorType = RankedTensorType::get(argType.getShape(), i8Type);
    auto toI8 = stablehlo::ConvertOp::create(builder, funcOp.getLoc(),
                                             i8TensorType, arg);

    // Compare NE with zero on i8 to normalize non-standard values.
    // Use scalar constant + broadcast to avoid materializing a full-sized
    // dense zero tensor in the IR.
    auto scalarI8Type = RankedTensorType::get({}, i8Type);
    auto scalarZero = stablehlo::ConstantOp::create(
        builder, funcOp.getLoc(),
        DenseElementsAttr::get(scalarI8Type, IntegerAttr::get(i8Type, 0)));
    auto zero = stablehlo::BroadcastInDimOp::create(
        builder, funcOp.getLoc(), i8TensorType, scalarZero.getResult(),
        builder.getDenseI64ArrayAttr({}));

    auto normalized = stablehlo::CompareOp::create(
        builder, funcOp.getLoc(), toI8.getResult(), zero.getResult(),
        stablehlo::ComparisonDirection::NE);

    // Replace all uses of the arg with the normalized i1 value
    // (except the convert we just created)
    Value normalizedVal = normalized.getResult();
    arg.replaceAllUsesExcept(normalizedVal, toI8.getOperation());
  }
}
class StablehloCustomLegalization
    : public impl::StablehloCustomLegalizationBase<
          StablehloCustomLegalization> {
public:
  using impl::StablehloCustomLegalizationBase<
      StablehloCustomLegalization>::StablehloCustomLegalizationBase;
  void runOnOperation() override {
    llvm::CrashRecoveryContext::Enable();
    llvm::CrashRecoveryContext CRC;
    LogicalResult result = failure();

    bool crashed = !CRC.RunSafely([&]() {
      func::FuncOp funcOp = getOperation();
      MLIRContext *context = &getContext();

      // Phase 1: Rewrite i1 function args to i8 with normalization.
      // This must happen before the pattern-based rewrite because it
      // changes the types that downstream ops see.
      normalizeBoolFuncArgs(funcOp);

      // Phase 2: Pattern-based legalization (scatter int→f32 upcast)
      ConversionTarget target(*context);
      target.addLegalDialect<stablehlo::StablehloDialect>();
      target.addDynamicallyLegalOp<stablehlo::ScatterOp>(
          [](stablehlo::ScatterOp op) {
            // Mark as illegal if it has integer element types that need
            // legalization
            if (op.getInputs().empty() || op.getUpdates().empty()) {
              return true; // Legal - empty inputs/updates
            }

            auto inputType =
                dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
            if (!inputType) {
              return true; // Legal - not ranked tensor
            }

            // Illegal if it's an integer type (needs upcasting to F32)
            return !isa<IntegerType>(inputType.getElementType());
          });

      RewritePatternSet patterns(context);
      patterns.add<ScatterLegalizationPattern>(context);
      result = applyPartialConversion(funcOp, target, std::move(patterns));
    });

    if (crashed || failed(result)) {
      if (crashed)
        getOperation()->emitError()
            << "StablehloCustomLegalization pass crashed unexpectedly";
      signalPassFailure();
    }
  }
};

} // namespace

// Default pass creation function (required by tablegen)
std::unique_ptr<OperationPass<func::FuncOp>>
createStablehloCustomLegalizationPass() {
  return std::make_unique<StablehloCustomLegalization>();
}

} // namespace mlir::torch
