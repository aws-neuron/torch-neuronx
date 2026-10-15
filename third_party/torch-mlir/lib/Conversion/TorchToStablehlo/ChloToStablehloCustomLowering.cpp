//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToStablehlo/ChloToStablehloCustomLowering.h"

#include "PopulatePatterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch-mlir/Conversion/Passes.h"
#include "torch-mlir/Conversion/TorchToStablehlo/StablehloLegalizeUtils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
namespace mlir::torch {
#define GEN_PASS_DEF_CONVERTCHLOTOSTABLEHLOCUSTOMLOWERING
#include "torch-mlir/Conversion/Passes.h.inc"

template <typename ChloOp, typename StablehloOp>
class ChloBroadcastOpConversionPattern : public OpConversionPattern<ChloOp> {
public:
  using OpConversionPattern<ChloOp>::OpConversionPattern;
  using OpAdaptor = typename ChloOp::Adaptor;

  LogicalResult
  matchAndRewrite(ChloOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "ChloBroadcastOpConversionPattern: "
                                         "result type is not a ranked tensor");
    }
    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    if (!lhsType || !rhsType) {
      return rewriter.notifyMatchFailure(
          op, "ChloBroadcastOpConversionPattern: operand types are not ranked "
              "tensors");
    }
    std::optional<Value> shapeTensor = std::nullopt;
    if (!resultType.hasStaticShape()) {
      SmallVector<Value> ops = {lhs, rhs};
      auto bcastInfo = hlo::getBroadcastResultShape(
          rewriter, op, ops, /*dimSizeIndexBits=*/64);
      if (failed(bcastInfo))
        return rewriter.notifyMatchFailure(
            op, "Could not infer broadcast result shape: at least 1 operand has "
                "incompatible dimension sizes or no operands match the "
                "broadcasted result type dimension sizes");
      shapeTensor = (*bcastInfo).first;
    }

    auto lhsResultType =
        RankedTensorType::get(resultType.getShape(), lhsType.getElementType());
    lhs = hlo::promoteAndBroadcast(rewriter, lhs, lhsResultType, shapeTensor);

    auto rhsResultType =
        RankedTensorType::get(resultType.getShape(), rhsType.getElementType());
    rhs = hlo::promoteAndBroadcast(rewriter, rhs, rhsResultType, shapeTensor);

    rewriter.replaceOpWithNewOp<StablehloOp>(op, resultType, lhs, rhs);

    return success();
  }
};

// Helper functions for attribute mapping
static std::optional<stablehlo::ComparisonDirection>
toStableHloComparisonDirection(chlo::ComparisonDirection value) {
  switch (value) {
  case chlo::ComparisonDirection::EQ:
    return stablehlo::ComparisonDirection::EQ;
  case chlo::ComparisonDirection::NE:
    return stablehlo::ComparisonDirection::NE;
  case chlo::ComparisonDirection::GE:
    return stablehlo::ComparisonDirection::GE;
  case chlo::ComparisonDirection::GT:
    return stablehlo::ComparisonDirection::GT;
  case chlo::ComparisonDirection::LE:
    return stablehlo::ComparisonDirection::LE;
  case chlo::ComparisonDirection::LT:
    return stablehlo::ComparisonDirection::LT;
  }
  return {};
}

static std::optional<stablehlo::ComparisonType>
toStableHloComparisonType(chlo::ComparisonType value) {
  switch (value) {
  case chlo::ComparisonType::NOTYPE:
    return stablehlo::ComparisonType::NOTYPE;
  case chlo::ComparisonType::FLOAT:
    return stablehlo::ComparisonType::FLOAT;
  case chlo::ComparisonType::TOTALORDER:
    return stablehlo::ComparisonType::TOTALORDER;
  case chlo::ComparisonType::SIGNED:
    return stablehlo::ComparisonType::SIGNED;
  case chlo::ComparisonType::UNSIGNED:
    return stablehlo::ComparisonType::UNSIGNED;
  }
  return {};
}

// Special case for chlo::BroadcastCompareOp
class ChloBroadcastCompareOpConversionPattern
    : public OpConversionPattern<chlo::BroadcastCompareOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename chlo::BroadcastCompareOp::Adaptor;

  LogicalResult
  matchAndRewrite(chlo::BroadcastCompareOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "ChloBroadcastCompareOpConversionPattern: result type is not a "
              "ranked tensor");
    }
    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    if (!lhsType || !rhsType) {
      return rewriter.notifyMatchFailure(
          op, "ChloBroadcastCompareOpConversionPattern: operand types are not "
              "ranked tensors");
    }
    std::optional<Value> shapeTensor = std::nullopt;
    if (!resultType.hasStaticShape()) {
      SmallVector<Value> ops = {lhs, rhs};
      auto bcastInfo = hlo::getBroadcastResultShape(
          rewriter, op, ops, /*dimSizeIndexBits=*/64);
      if (failed(bcastInfo))
        return rewriter.notifyMatchFailure(
            op, "Could not infer broadcast result shape: at least 1 operand has "
                "incompatible dimension sizes or no operands match the "
                "broadcasted result type dimension sizes");
      shapeTensor = (*bcastInfo).first;
    }

    auto lhsResultType =
        RankedTensorType::get(resultType.getShape(), lhsType.getElementType());
    lhs = hlo::promoteAndBroadcast(rewriter, lhs, lhsResultType, shapeTensor);

    auto rhsResultType =
        RankedTensorType::get(resultType.getShape(), rhsType.getElementType());
    rhs = hlo::promoteAndBroadcast(rewriter, rhs, rhsResultType, shapeTensor);

    // Map chlo compare attributes to stablehlo compare attributes
    auto chloDirection = op.getComparisonDirection();
    auto hloDirection = toStableHloComparisonDirection(chloDirection);
    if (!hloDirection) {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported comparison direction");
    }

    auto chloType = op.getCompareType().value_or(chlo::ComparisonType::NOTYPE);
    auto hloType = toStableHloComparisonType(chloType);
    if (!hloType) {
      return rewriter.notifyMatchFailure(op, "unsupported comparison type");
    }

    auto hloDirectionAttr = stablehlo::ComparisonDirectionAttr::get(
        rewriter.getContext(), *hloDirection);
    auto hloTypeAttr = op.getCompareType()
                           ? stablehlo::ComparisonTypeAttr::get(
                                 rewriter.getContext(), *hloType)
                           : nullptr;

    rewriter.replaceOpWithNewOp<stablehlo::CompareOp>(
        op, resultType, lhs, rhs, hloDirectionAttr, hloTypeAttr);

    return success();
  }
};

// Special case for chlo::BroadcastSelectOp
class ChloBroadcastSelectOpConversionPattern
    : public OpConversionPattern<chlo::BroadcastSelectOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename chlo::BroadcastSelectOp::Adaptor;

  LogicalResult
  matchAndRewrite(chlo::BroadcastSelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto pred = adaptor.getPred();
    auto lhs = adaptor.getOnTrue();
    auto rhs = adaptor.getOnFalse();
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "ChloBroadcastSelectOpConversionPattern: result type is not a "
              "ranked tensor");
    }
    auto predType = dyn_cast<RankedTensorType>(pred.getType());
    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    if (!predType || !lhsType || !rhsType) {
      return rewriter.notifyMatchFailure(
          op, "ChloBroadcastSelectOpConversionPattern: operand types are not "
              "ranked tensors");
    }
    std::optional<Value> shapeTensor = std::nullopt;
    if (!resultType.hasStaticShape()) {
      SmallVector<Value> ops = {pred, lhs, rhs};
      auto bcastInfo = hlo::getBroadcastResultShape(
          rewriter, op, ops, /*dimSizeIndexBits=*/64);
      if (failed(bcastInfo))
        return rewriter.notifyMatchFailure(
            op, "Could not infer broadcast result shape: at least 1 operand has "
                "incompatible dimension sizes or no operands match the "
                "broadcasted result type dimension sizes");
      shapeTensor = (*bcastInfo).first;
    }

    auto predResultType =
        RankedTensorType::get(resultType.getShape(), predType.getElementType());
    pred = hlo::promoteAndBroadcast(rewriter, pred, predResultType, shapeTensor);

    auto lhsResultType =
        RankedTensorType::get(resultType.getShape(), lhsType.getElementType());
    lhs = hlo::promoteAndBroadcast(rewriter, lhs, lhsResultType, shapeTensor);

    auto rhsResultType =
        RankedTensorType::get(resultType.getShape(), rhsType.getElementType());
    rhs = hlo::promoteAndBroadcast(rewriter, rhs, rhsResultType, shapeTensor);

    rewriter.replaceOpWithNewOp<stablehlo::SelectOp>(op, resultType, pred, lhs,
                                                     rhs);

    return success();
  }
};

namespace {
class ConvertChloToStablehloCustomLowering
    : public impl::ConvertChloToStablehloCustomLoweringBase<
          ConvertChloToStablehloCustomLowering> {
public:
  using impl::ConvertChloToStablehloCustomLoweringBase<
      ConvertChloToStablehloCustomLowering>::
      ConvertChloToStablehloCustomLoweringBase;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<stablehlo::StablehloDialect>();
    registry.insert<chlo::ChloDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    llvm::CrashRecoveryContext::Enable();
    llvm::CrashRecoveryContext CRC;
    LogicalResult result = failure();

    bool crashed = !CRC.RunSafely([&]() {
      MLIRContext *context = &getContext();
      ConversionTarget target(*context);
      target.addLegalDialect<stablehlo::StablehloDialect, tensor::TensorDialect,
                             arith::ArithDialect>();

      TypeConverter typeConverter;
      typeConverter.addConversion([](Type type) -> Type { return type; });
      TorchConversion::setupBackendTypeConversionForStablehlo(target,
                                                              typeConverter);

      RewritePatternSet patterns(context);

      // Add patterns for all CHLO broadcast operations
      target.addIllegalOp<chlo::BroadcastAddOp>();
      patterns.add<ChloBroadcastOpConversionPattern<chlo::BroadcastAddOp,
                                                    stablehlo::AddOp>>(
          typeConverter, context);

      target.addIllegalOp<chlo::BroadcastAndOp>();
      patterns.add<ChloBroadcastOpConversionPattern<chlo::BroadcastAndOp,
                                                    stablehlo::AndOp>>(
          typeConverter, context);

      target.addIllegalOp<chlo::BroadcastAtan2Op>();
      patterns.add<ChloBroadcastOpConversionPattern<chlo::BroadcastAtan2Op,
                                                    stablehlo::Atan2Op>>(
          typeConverter, context);

      target.addIllegalOp<chlo::BroadcastComplexOp>();
      patterns.add<ChloBroadcastOpConversionPattern<chlo::BroadcastComplexOp,
                                                    stablehlo::ComplexOp>>(
          typeConverter, context);

      target.addIllegalOp<chlo::BroadcastDivOp>();
      patterns.add<ChloBroadcastOpConversionPattern<chlo::BroadcastDivOp,
                                                    stablehlo::DivOp>>(
          typeConverter, context);

      target.addIllegalOp<chlo::BroadcastMaxOp>();
      patterns.add<ChloBroadcastOpConversionPattern<chlo::BroadcastMaxOp,
                                                    stablehlo::MaxOp>>(
          typeConverter, context);

      target.addIllegalOp<chlo::BroadcastMinOp>();
      patterns.add<ChloBroadcastOpConversionPattern<chlo::BroadcastMinOp,
                                                    stablehlo::MinOp>>(
          typeConverter, context);

      target.addIllegalOp<chlo::BroadcastMulOp>();
      patterns.add<ChloBroadcastOpConversionPattern<chlo::BroadcastMulOp,
                                                    stablehlo::MulOp>>(
          typeConverter, context);

      target.addIllegalOp<chlo::BroadcastNextAfterOp>();
      patterns.add<ChloBroadcastOpConversionPattern<chlo::BroadcastNextAfterOp,
                                                    chlo::NextAfterOp>>(
          typeConverter, context);

      target.addIllegalOp<chlo::BroadcastOrOp>();
      patterns.add<ChloBroadcastOpConversionPattern<chlo::BroadcastOrOp,
                                                    stablehlo::OrOp>>(
          typeConverter, context);

      target.addIllegalOp<chlo::BroadcastPolygammaOp>();
      patterns.add<ChloBroadcastOpConversionPattern<chlo::BroadcastPolygammaOp,
                                                    chlo::PolygammaOp>>(
          typeConverter, context);

      target.addIllegalOp<chlo::BroadcastPowOp>();
      patterns.add<ChloBroadcastOpConversionPattern<chlo::BroadcastPowOp,
                                                    stablehlo::PowOp>>(
          typeConverter, context);

      target.addIllegalOp<chlo::BroadcastRemOp>();
      patterns.add<ChloBroadcastOpConversionPattern<chlo::BroadcastRemOp,
                                                    stablehlo::RemOp>>(
          typeConverter, context);

      target.addIllegalOp<chlo::BroadcastShiftLeftOp>();
      patterns.add<ChloBroadcastOpConversionPattern<chlo::BroadcastShiftLeftOp,
                                                    stablehlo::ShiftLeftOp>>(
          typeConverter, context);

      target.addIllegalOp<chlo::BroadcastShiftRightArithmeticOp>();
      patterns.add<ChloBroadcastOpConversionPattern<
          chlo::BroadcastShiftRightArithmeticOp,
          stablehlo::ShiftRightArithmeticOp>>(typeConverter, context);

      target.addIllegalOp<chlo::BroadcastShiftRightLogicalOp>();
      patterns.add<ChloBroadcastOpConversionPattern<
          chlo::BroadcastShiftRightLogicalOp, stablehlo::ShiftRightLogicalOp>>(
          typeConverter, context);

      target.addIllegalOp<chlo::BroadcastSubOp>();
      patterns.add<ChloBroadcastOpConversionPattern<chlo::BroadcastSubOp,
                                                    stablehlo::SubtractOp>>(
          typeConverter, context);

      target.addIllegalOp<chlo::BroadcastXorOp>();
      patterns.add<ChloBroadcastOpConversionPattern<chlo::BroadcastXorOp,
                                                    stablehlo::XorOp>>(
          typeConverter, context);

      target.addIllegalOp<chlo::BroadcastZetaOp>();
      patterns.add<ChloBroadcastOpConversionPattern<chlo::BroadcastZetaOp,
                                                    chlo::ZetaOp>>(
          typeConverter, context);

      target.addIllegalOp<chlo::BroadcastCompareOp>();
      patterns.add<ChloBroadcastCompareOpConversionPattern>(typeConverter,
                                                            context);

      target.addIllegalOp<chlo::BroadcastSelectOp>();
      patterns.add<ChloBroadcastSelectOpConversionPattern>(typeConverter,
                                                           context);
      result =
          applyPartialConversion(getOperation(), target, std::move(patterns));
    });

    if (crashed || failed(result)) {
      if (crashed)
        getOperation()->emitError()
            << "ConvertChloToStablehloCustomLowering pass crashed unexpectedly";
      signalPassFailure();
    }
  }
};
} // namespace

// Default pass creation function (required by tablegen)
std::unique_ptr<OperationPass<func::FuncOp>>
createConvertChloToStablehloCustomLoweringPass() {
  return std::make_unique<ConvertChloToStablehloCustomLowering>();
}

} // namespace mlir::torch
