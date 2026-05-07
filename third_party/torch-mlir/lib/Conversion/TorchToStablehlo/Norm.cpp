//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToStablehlo/TorchToStablehlo.h"

#include "PopulatePatterns.h"
#include "Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch-mlir/Conversion/TorchToStablehlo/StablehloLegalizeUtils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include <cmath>
#include <numeric>
#include <type_traits>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::torch_to_stablehlo;

namespace {
class ConvertAten_NativeBatchNormLegitFunctionalOp
    : public OpConversionPattern<Aten_NativeBatchNormLegitFunctionalOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Aten_NativeBatchNormLegitFunctionalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value input = adaptor.getInput();
    Value weight = adaptor.getWeight();
    Value bias = adaptor.getBias();

    // Create identity transform
    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t featureDim = inputType.getShape()[1];
    auto featureType =
        RankedTensorType::get({featureDim}, inputType.getElementType());

    if (isa<Torch::NoneType>(weight.getType())) {
      weight = stablehlo::ConstantOp::create(
          rewriter, loc,
          DenseElementsAttr::get(featureType, rewriter.getF32FloatAttr(1.0)));
    }
    if (isa<Torch::NoneType>(bias.getType())) {
      bias = stablehlo::ConstantOp::create(
          rewriter, loc,
          DenseElementsAttr::get(featureType, rewriter.getF32FloatAttr(0.0)));
    }

    Value runningMean = adaptor.getRunningMean();
    Value runningVar = adaptor.getRunningVar();

    double epsValue;
    if (!matchPattern(op.getEps(), m_TorchConstantFloat(&epsValue)))
      return rewriter.notifyMatchFailure(op,
                                         "Epsilon should be a constant float");
    auto epsAttr = rewriter.getF32FloatAttr(epsValue);

    // Feature dim is assumed to be 1 (BS, feature_dim, image_shape<...>)
    // since torch will ensure the appropriate shape
    bool training;
    if (!matchPattern(op.getTraining(), m_TorchConstantBool(&training)))
      return rewriter.notifyMatchFailure(op,
                                         "Training should be a constant bool");

    if (training) {
      auto batchNormOp = stablehlo::BatchNormTrainingOp::create(
          rewriter, loc, input, weight, bias, epsAttr,
          rewriter.getI64IntegerAttr(1));

      Value batchMean = batchNormOp.getResult(1);
      Value batchVar = batchNormOp.getResult(2);

      double momentumValue;
      if (!matchPattern(op.getMomentum(), m_TorchConstantFloat(&momentumValue)))
        return rewriter.notifyMatchFailure(
            op, "Momentum should be a constant float");

      auto scalarType = RankedTensorType::get({}, rewriter.getF32Type());
      Value momentum = stablehlo::ConstantOp::create(
          rewriter, loc,
          DenseElementsAttr::get(scalarType,
                                 rewriter.getF32FloatAttr(momentumValue)));
      Value one = stablehlo::ConstantOp::create(
          rewriter, loc,
          DenseElementsAttr::get(scalarType, rewriter.getF32FloatAttr(1.0)));
      Value oneMinusMomentum =
          stablehlo::SubtractOp::create(rewriter, loc, one, momentum);

      // Compute the new running mean/var via new_stat = momentum * batch_stat +
      // (1-momentum) * current_estimated_stat
      Value runningMeanMul =
          chlo::BroadcastMulOp::create(rewriter, loc, runningMean.getType(),
                                       oneMinusMomentum, runningMean, nullptr);
      Value batchMeanMul = chlo::BroadcastMulOp::create(
          rewriter, loc, batchMean.getType(), momentum, batchMean, nullptr);
      Value newMean =
          stablehlo::AddOp::create(rewriter, loc, runningMeanMul, batchMeanMul);

      Value runningVarMul =
          chlo::BroadcastMulOp::create(rewriter, loc, runningVar.getType(),
                                       oneMinusMomentum, runningVar, nullptr);
      Value batchVarMul = chlo::BroadcastMulOp::create(
          rewriter, loc, batchVar.getType(), momentum, batchVar, nullptr);
      Value newVar =
          stablehlo::AddOp::create(rewriter, loc, runningVarMul, batchVarMul);

      rewriter.replaceOp(
          op, {batchNormOp.getResult(0), weight, bias, newMean, newVar});
    } else {
      auto batchNormOp = stablehlo::BatchNormInferenceOp::create(
          rewriter, loc, input, weight, bias, runningMean, runningVar, epsAttr,
          rewriter.getI64IntegerAttr(1));
      rewriter.replaceOp(
          op, {batchNormOp.getResult(), weight, bias, runningMean, runningVar});
    }

    return success();
  }
};
} // namespace

namespace {
class ConvertAten_NativeBatchNormLegitNoTrainingOp
    : public OpConversionPattern<Aten_NativeBatchNormLegitNoTrainingOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Aten_NativeBatchNormLegitNoTrainingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value input = adaptor.getInput();
    Value weight = adaptor.getWeight();
    Value bias = adaptor.getBias();
    Value runningMean = adaptor.getRunningMean();
    Value runningVar = adaptor.getRunningVar();

    double epsValue;
    if (!matchPattern(op.getEps(), m_TorchConstantFloat(&epsValue)))
      return rewriter.notifyMatchFailure(op,
                                         "Epsilon should be a constant float");
    auto epsAttr = rewriter.getF32FloatAttr(epsValue);

    // Feature dim is assumed to be 1
    // Directly use the running mean & var for inference
    auto batchNormOp = stablehlo::BatchNormInferenceOp::create(
        rewriter, loc, input, weight, bias, runningMean, runningVar, epsAttr,
        rewriter.getI64IntegerAttr(1));

    // Create empty tensors for mean & var in inference
    auto emptyType = RankedTensorType::get({0}, rewriter.getF32Type());
    Value emptyMean = stablehlo::ConstantOp::create(
        rewriter, loc, DenseElementsAttr::get(emptyType, ArrayRef<float>{}));
    Value emptyVar = stablehlo::ConstantOp::create(
        rewriter, loc, DenseElementsAttr::get(emptyType, ArrayRef<float>{}));

    rewriter.replaceOp(op, {batchNormOp.getResult(), emptyMean, emptyVar});

    return success();
  }
};
} // namespace

namespace {
class ConvertAten_NativeBatchNormLegitNoStatsOp
    : public OpConversionPattern<Aten_NativeBatchNormLegitNoStatsOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Aten_NativeBatchNormLegitNoStatsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value input = adaptor.getInput();
    Value weight = adaptor.getWeight();
    Value bias = adaptor.getBias();

    double epsValue;
    if (!matchPattern(op.getEps(), m_TorchConstantFloat(&epsValue)))
      return rewriter.notifyMatchFailure(op,
                                         "Epsilon should be a constant float");
    auto epsAttr = rewriter.getF32FloatAttr(epsValue);

    // Feature dim is assumed to be 1
    auto batchNormOp = stablehlo::BatchNormTrainingOp::create(
        rewriter, loc, input, weight, bias, epsAttr,
        rewriter.getI64IntegerAttr(1));

    rewriter.replaceOp(op, {batchNormOp.getResult(0), batchNormOp.getResult(1),
                            batchNormOp.getResult(2)});

    return success();
  }
};
} // namespace

void mlir::torch::torch_to_stablehlo::populateNormOpPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, const TorchToStablehloOptions &options) {
  MLIRContext *context = patterns.getContext();

  target.addIllegalOp<Aten_NativeBatchNormLegitFunctionalOp>();
  patterns.add<ConvertAten_NativeBatchNormLegitFunctionalOp>(typeConverter,
                                                             context);

  target.addIllegalOp<Aten_NativeBatchNormLegitNoTrainingOp>();
  patterns.add<ConvertAten_NativeBatchNormLegitNoTrainingOp>(typeConverter,
                                                             context);

  target.addIllegalOp<Aten_NativeBatchNormLegitNoStatsOp>();
  patterns.add<ConvertAten_NativeBatchNormLegitNoStatsOp>(typeConverter,
                                                          context);
}
