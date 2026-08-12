//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToStablehlo/TorchToStablehloCustomLowering.h"

#include "PopulatePatterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
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

#define GEN_PASS_DEF_CONVERTTORCHTOSTABLEHLOCUSTOMLOWERING
#include "torch-mlir/Conversion/Passes.h.inc"

class ConvertArithConstantOp : public OpConversionPattern<arith::ConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::ConstantOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto originalType = op.getType();
    auto value = adaptor.getValue();

    if (originalType.isIndex()) {
      if (auto attr = dyn_cast<IntegerAttr>(value)) {
        auto tensorType = RankedTensorType::get({}, rewriter.getI64Type());
        auto denseAttr = DenseElementsAttr::get(
            tensorType,
            rewriter.getIntegerAttr(rewriter.getI64Type(), attr.getInt()));
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, tensorType,
                                                           denseAttr);
        return success();
      }
      return rewriter.notifyMatchFailure(
          op, "unsupported index constant type - must contain an int attr");
    } else if (originalType.isIntOrFloat()) {
      if (auto attr = dyn_cast<TypedAttr>(value)) {
        auto tensorType = RankedTensorType::get({}, originalType);
        auto denseAttr = DenseElementsAttr::get(tensorType, attr);
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, tensorType,
                                                           denseAttr);
        return success();
      }
      return rewriter.notifyMatchFailure(
          op,
          "unsupported int/float constant type - must contain a typed attr");
    } else if (auto tensorType = dyn_cast<RankedTensorType>(originalType)) {
      if (auto attr = dyn_cast<ElementsAttr>(value)) {
        if (tensorType.getElementType().isIndex()) {
          auto i64Type = rewriter.getI64Type();
          auto i64TensorType =
              RankedTensorType::get(tensorType.getShape(), i64Type);
          SmallVector<int64_t> values;
          for (auto v : attr.getValues<IntegerAttr>())
            values.push_back(v.getInt());
          auto i64Attr = DenseIntElementsAttr::get(i64TensorType, values);
          rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, i64TensorType,
                                                             i64Attr);
        } else {
          rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, tensorType,
                                                             attr);
        }
        return success();
      }
      return rewriter.notifyMatchFailure(
          op, "unsupported ranked tensor constant type - must contain an "
              "elements attr");
    }
    return rewriter.notifyMatchFailure(
        op, "unsupported constant type - must either be a constant int/float "
            "or ranked tensor");
  }
};

class ConvertArithExtFOp : public OpConversionPattern<arith::ExtFOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::ExtFOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::ExtFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto input = adaptor.getOperands()[0];
    rewriter.replaceOpWithNewOp<stablehlo::ConvertOp>(op, type, input);
    return success();
  }
};

class ConvertArithExtSIOp : public OpConversionPattern<arith::ExtSIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::ExtSIOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::ExtSIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto input = adaptor.getOperands()[0];
    rewriter.replaceOpWithNewOp<stablehlo::ConvertOp>(op, type, input);
    return success();
  }
};

class ConvertArithExtUIOp : public OpConversionPattern<arith::ExtUIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::ExtUIOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::ExtUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto input = adaptor.getOperands()[0];
    rewriter.replaceOpWithNewOp<stablehlo::ConvertOp>(op, type, input);
    return success();
  }
};

class ConvertArithMaxSIOp : public OpConversionPattern<arith::MaxSIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::MaxSIOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::MaxSIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto lhs = adaptor.getOperands()[0];
    auto rhs = adaptor.getOperands()[1];
    rewriter.replaceOpWithNewOp<stablehlo::MaxOp>(op, type, lhs, rhs);
    return success();
  }
};

class ConvertArithMaxUIOp : public OpConversionPattern<arith::MaxUIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::MaxUIOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::MaxUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto lhs = adaptor.getOperands()[0];
    auto rhs = adaptor.getOperands()[1];
    rewriter.replaceOpWithNewOp<stablehlo::MaxOp>(op, type, lhs, rhs);
    return success();
  }
};

class ConvertArithMinSIOp : public OpConversionPattern<arith::MinSIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::MinSIOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::MinSIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto lhs = adaptor.getOperands()[0];
    auto rhs = adaptor.getOperands()[1];
    rewriter.replaceOpWithNewOp<stablehlo::MinOp>(op, type, lhs, rhs);
    return success();
  }
};

class ConvertArithMinUIOp : public OpConversionPattern<arith::MinUIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::MinUIOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::MinUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto lhs = adaptor.getOperands()[0];
    auto rhs = adaptor.getOperands()[1];
    rewriter.replaceOpWithNewOp<stablehlo::MinOp>(op, type, lhs, rhs);
    return success();
  }
};

class ConvertArithAddIOp : public OpConversionPattern<arith::AddIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::AddIOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::AddIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto lhs = adaptor.getOperands()[0];
    auto rhs = adaptor.getOperands()[1];
    rewriter.replaceOpWithNewOp<stablehlo::AddOp>(op, type, lhs, rhs);
    return success();
  }
};

class ConvertArithAddFOp : public OpConversionPattern<arith::AddFOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::AddFOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::AddFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto lhs = adaptor.getOperands()[0];
    auto rhs = adaptor.getOperands()[1];
    rewriter.replaceOpWithNewOp<stablehlo::AddOp>(op, type, lhs, rhs);
    return success();
  }
};

class ConvertArithAndIOp : public OpConversionPattern<arith::AndIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::AndIOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::AndIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto lhs = adaptor.getOperands()[0];
    auto rhs = adaptor.getOperands()[1];
    rewriter.replaceOpWithNewOp<stablehlo::AndOp>(op, type, lhs, rhs);
    return success();
  }
};

class ConvertArithBitcastOp : public OpConversionPattern<arith::BitcastOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::BitcastOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<stablehlo::ConvertOp>(op, type,
                                                      adaptor.getIn());
    return success();
  }
};

class ConvertArithCeilDivSIOp : public OpConversionPattern<arith::CeilDivSIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::CeilDivSIOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::CeilDivSIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    if (!lhsType || !rhsType) {
      return rewriter.notifyMatchFailure(
          op, "arith::CeilDivSIOp: lhs is not a ranked tensor");
    }
    auto elementType = lhsType.getElementType();
    auto intType = dyn_cast<IntegerType>(elementType);
    if (!intType) {
      return rewriter.notifyMatchFailure(
          op, "arith::CeilDivSIOp: element type is not integer");
    }

    // Convert to float64
    auto f64Type = rewriter.getF64Type();
    auto f64LhsType = RankedTensorType::get(lhsType.getShape(), f64Type,
                                            lhsType.getEncoding());
    auto f64RhsType = RankedTensorType::get(rhsType.getShape(), f64Type,
                                            rhsType.getEncoding());
    Value f64Lhs =
        stablehlo::ConvertOp::create(rewriter, op.getLoc(), f64LhsType, lhs);
    Value f64Rhs =
        stablehlo::ConvertOp::create(rewriter, op.getLoc(), f64RhsType, rhs);

    // Perform float division + ceil + back conversion
    Value divResult = stablehlo::DivOp::create(rewriter, op.getLoc(),
                                               f64LhsType, f64Lhs, f64Rhs);
    Value ceilResult =
        stablehlo::CeilOp::create(rewriter, op.getLoc(), f64LhsType, divResult);
    Value finalResult =
        stablehlo::ConvertOp::create(rewriter, op.getLoc(), type, ceilResult);
    rewriter.replaceOp(op, finalResult);
    return success();
  }
};

class ConvertArithCmpIOp : public OpConversionPattern<arith::CmpIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::CmpIOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto lhs = adaptor.getOperands()[0];
    auto rhs = adaptor.getOperands()[1];
    auto lhsType = dyn_cast<RankedTensorType>(
        getTypeConverter()->convertType(lhs.getType()));
    if (!lhsType) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag << "arith::CmpIOp has unranked lhs op";
      });
    }
    auto lhsElementType = lhsType.getElementType();

    stablehlo::ComparisonDirection cmpDirection;
    switch (op.getPredicate()) {
    case arith::CmpIPredicate::eq:
      cmpDirection = stablehlo::ComparisonDirection::EQ;
      break;
    case arith::CmpIPredicate::ne:
      cmpDirection = stablehlo::ComparisonDirection::NE;
      break;
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::ult:
      cmpDirection = stablehlo::ComparisonDirection::LT;
      break;
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::ule:
      cmpDirection = stablehlo::ComparisonDirection::LE;
      break;
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::ugt:
      cmpDirection = stablehlo::ComparisonDirection::GT;
      break;
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::uge:
      cmpDirection = stablehlo::ComparisonDirection::GE;
      break;
    }
    stablehlo::ComparisonType cmpType;
    if (auto intType = dyn_cast<mlir::IntegerType>(lhsElementType)) {
      if (intType.isSignedInteger()) {
        cmpType = stablehlo::ComparisonType::SIGNED;
      } else if (intType.isUnsignedInteger()) {
        cmpType = stablehlo::ComparisonType::UNSIGNED;
      } else if (intType.isSignlessInteger()) {
        cmpType = stablehlo::ComparisonType::UNSIGNED;
      } else {
        return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
          diag << "arith::CmpIOp lhs op has integer type which is not "
                  "signed/unsigned";
        });
      }
    } else if (auto floatType = dyn_cast<mlir::FloatType>(lhsElementType)) {
      cmpType = stablehlo::ComparisonType::FLOAT;
    } else if (auto complexType = dyn_cast<mlir::ComplexType>(lhsElementType)) {
      cmpType = stablehlo::ComparisonType::FLOAT;
    } else {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag << "arith::CmpIOp lhs op has type which is not "
                "integer/float/complex";
      });
    }

    auto cmpDirectionAttr = stablehlo::ComparisonDirectionAttr::get(
        rewriter.getContext(), cmpDirection);
    auto cmpTypeAttr =
        stablehlo::ComparisonTypeAttr::get(rewriter.getContext(), cmpType);
    rewriter.replaceOpWithNewOp<stablehlo::CompareOp>(
        op, type, lhs, rhs, cmpDirectionAttr, cmpTypeAttr);
    return success();
  }
};

class ConvertArithCmpFOp : public OpConversionPattern<arith::CmpFOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::CmpFOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::CmpFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto lhs = adaptor.getOperands()[0];
    auto rhs = adaptor.getOperands()[1];
    auto lhsType = dyn_cast<RankedTensorType>(
        getTypeConverter()->convertType(lhs.getType()));
    if (!lhsType) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag << "arith::CmpFOp has unranked lhs op";
      });
    }
    auto lhsElementType = lhsType.getElementType();

    stablehlo::ComparisonDirection cmpDirection;
    switch (op.getPredicate()) {
    case arith::CmpFPredicate::OEQ:
    case arith::CmpFPredicate::UEQ:
      cmpDirection = stablehlo::ComparisonDirection::EQ;
      break;
    case arith::CmpFPredicate::ONE:
    case arith::CmpFPredicate::UNE:
      cmpDirection = stablehlo::ComparisonDirection::NE;
      break;
    case arith::CmpFPredicate::OLT:
    case arith::CmpFPredicate::ULT:
      cmpDirection = stablehlo::ComparisonDirection::LT;
      break;
    case arith::CmpFPredicate::OLE:
    case arith::CmpFPredicate::ULE:
      cmpDirection = stablehlo::ComparisonDirection::LE;
      break;
    case arith::CmpFPredicate::OGT:
    case arith::CmpFPredicate::UGT:
      cmpDirection = stablehlo::ComparisonDirection::GT;
      break;
    case arith::CmpFPredicate::OGE:
    case arith::CmpFPredicate::UGE:
      cmpDirection = stablehlo::ComparisonDirection::GE;
      break;
    default:
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag << "Unsupported floating-point comparison predicate";
      });
    }
    stablehlo::ComparisonType cmpType;
    if (auto floatType = dyn_cast<mlir::FloatType>(lhsElementType)) {
      cmpType = stablehlo::ComparisonType::FLOAT;
    } else if (auto complexType = dyn_cast<mlir::ComplexType>(lhsElementType)) {
      cmpType = stablehlo::ComparisonType::FLOAT;
    } else {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag << "arith::CmpFOp lhs op has type which is not float/complex";
      });
    }

    auto cmpDirectionAttr = stablehlo::ComparisonDirectionAttr::get(
        rewriter.getContext(), cmpDirection);
    auto cmpTypeAttr =
        stablehlo::ComparisonTypeAttr::get(rewriter.getContext(), cmpType);
    rewriter.replaceOpWithNewOp<stablehlo::CompareOp>(
        op, type, lhs, rhs, cmpDirectionAttr, cmpTypeAttr);
    return success();
  }
};

class ConvertArithDivFOp : public OpConversionPattern<arith::DivFOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::DivFOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::DivFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto lhs = adaptor.getOperands()[0];
    auto rhs = adaptor.getOperands()[1];
    rewriter.replaceOpWithNewOp<stablehlo::DivOp>(op, type, lhs, rhs);
    return success();
  }
};

class ConvertArithDivSIOp : public OpConversionPattern<arith::DivSIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::DivSIOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::DivSIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    if (!lhsType || !rhsType) {
      return rewriter.notifyMatchFailure(
          op, "arith::DivSIOp: lhs is not a ranked tensor");
    }
    auto elementType = lhsType.getElementType();
    auto intType = dyn_cast<IntegerType>(elementType);
    if (!intType || !intType.isSignlessInteger()) {
      return rewriter.notifyMatchFailure(
          op, "arith::DivSIOp: element type is not signless integer");
    }
    rewriter.replaceOpWithNewOp<stablehlo::DivOp>(op, type, lhs, rhs);
    return success();
  }
};

class ConvertArithDivUIOp : public OpConversionPattern<arith::DivUIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::DivUIOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::DivUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    if (!lhsType || !rhsType) {
      return rewriter.notifyMatchFailure(
          op, "arith::DivUIOp: lhs is not a ranked tensor");
    }
    auto elementType = lhsType.getElementType();
    auto intType = dyn_cast<IntegerType>(elementType);
    if (!intType || !intType.isSignlessInteger()) {
      return rewriter.notifyMatchFailure(
          op, "arith::DivUIOp: element type is not signless integer");
    }

    // Bitcast convert lhs/rhs to unsigned type of same bitwidth
    auto unsignedElementType = IntegerType::get(
        rewriter.getContext(), intType.getWidth(), IntegerType::Unsigned);
    auto unsignedLhsType = RankedTensorType::get(
        lhsType.getShape(), unsignedElementType, lhsType.getEncoding());
    auto unsignedRhsType = RankedTensorType::get(
        rhsType.getShape(), unsignedElementType, rhsType.getEncoding());
    Value divLhs = stablehlo::BitcastConvertOp::create(rewriter, op.getLoc(),
                                                       unsignedLhsType, lhs);
    Value divRhs = stablehlo::BitcastConvertOp::create(rewriter, op.getLoc(),
                                                       unsignedRhsType, rhs);

    // Perform stablehlo::divide + convert back to signed
    Value divResult = stablehlo::DivOp::create(
        rewriter, op.getLoc(), divLhs.getType(), divLhs, divRhs);
    divResult = stablehlo::BitcastConvertOp::create(rewriter, op.getLoc(), type,
                                                    divResult);
    rewriter.replaceOp(op, divResult);
    return success();
  }
};

class ConvertArithFloorDivSIOp
    : public OpConversionPattern<arith::FloorDivSIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::FloorDivSIOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::FloorDivSIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    if (!lhsType || !rhsType) {
      return rewriter.notifyMatchFailure(
          op, "arith::FloorDivSIOp: lhs is not a ranked tensor");
    }
    auto elementType = lhsType.getElementType();
    auto intType = dyn_cast<IntegerType>(elementType);
    if (!intType) {
      return rewriter.notifyMatchFailure(
          op, "arith::FloorDivSIOp: element type is not integer");
    }

    // Convert to float64
    auto f64Type = rewriter.getF64Type();
    auto f64LhsType = RankedTensorType::get(lhsType.getShape(), f64Type,
                                            lhsType.getEncoding());
    auto f64RhsType = RankedTensorType::get(rhsType.getShape(), f64Type,
                                            rhsType.getEncoding());
    Value f64Lhs =
        stablehlo::ConvertOp::create(rewriter, op.getLoc(), f64LhsType, lhs);
    Value f64Rhs =
        stablehlo::ConvertOp::create(rewriter, op.getLoc(), f64RhsType, rhs);

    // Perform float division + floor + back conversion
    Value divResult = stablehlo::DivOp::create(rewriter, op.getLoc(),
                                               f64LhsType, f64Lhs, f64Rhs);
    Value ceilResult = stablehlo::FloorOp::create(rewriter, op.getLoc(),
                                                  f64LhsType, divResult);
    Value finalResult =
        stablehlo::ConvertOp::create(rewriter, op.getLoc(), type, ceilResult);
    rewriter.replaceOp(op, finalResult);
    return success();
  }
};

class ConvertArithFPToSIOp : public OpConversionPattern<arith::FPToSIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::FPToSIOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::FPToSIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<stablehlo::ConvertOp>(op, type,
                                                      adaptor.getIn());
    return success();
  }
};

class ConvertArithFPToUIOp : public OpConversionPattern<arith::FPToUIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::FPToUIOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::FPToUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<stablehlo::ConvertOp>(op, type,
                                                      adaptor.getIn());
    return success();
  }
};

class ConvertArithSelectOp : public OpConversionPattern<arith::SelectOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::SelectOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto pred = adaptor.getOperands()[0];
    auto lhs = adaptor.getOperands()[1];
    auto rhs = adaptor.getOperands()[2];
    rewriter.replaceOpWithNewOp<stablehlo::SelectOp>(op, type, pred, lhs, rhs);
    return success();
  }
};

class ConvertArithIndexCastOp : public OpConversionPattern<arith::IndexCastOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::IndexCastOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::IndexCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto input = adaptor.getIn();
    op.replaceAllUsesWith(input);
    op.erase();
    return success();
  }
};

class ConvertArithIndexCastUIOp
    : public OpConversionPattern<arith::IndexCastUIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::IndexCastUIOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::IndexCastUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Input is already casted by adapter type conversion
    auto input = adaptor.getIn();
    op.replaceAllUsesWith(input);
    op.erase();
    return success();
  }
};

class ConvertArithMaximumFOp : public OpConversionPattern<arith::MaximumFOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::MaximumFOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::MaximumFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto lhs = adaptor.getOperands()[0];
    auto rhs = adaptor.getOperands()[1];

    // Replace the op with stablehlo maximum
    rewriter.replaceOpWithNewOp<stablehlo::MaxOp>(op, type, lhs, rhs);
    return success();
  }
};

class ConvertArithMinimumFOp : public OpConversionPattern<arith::MinimumFOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::MinimumFOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::MinimumFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto lhs = adaptor.getOperands()[0];
    auto rhs = adaptor.getOperands()[1];

    rewriter.replaceOpWithNewOp<stablehlo::MinOp>(op, type, lhs, rhs);
    return success();
  }
};

class ConvertArithMulFOp : public OpConversionPattern<arith::MulFOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::MulFOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::MulFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto lhs = adaptor.getOperands()[0];
    auto rhs = adaptor.getOperands()[1];

    rewriter.replaceOpWithNewOp<stablehlo::MulOp>(op, type, lhs, rhs);
    return success();
  }
};

class ConvertArithTruncFOp : public OpConversionPattern<arith::TruncFOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::TruncFOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::TruncFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (op.getRoundingmode()) {
      return op->emitError("arith::TruncFOp: roundingMode is not supported");
    }
    if (op.getFastmath()) {
      return op->emitError("arith::TruncFOp: fastmath is not supported");
    }
    auto type = getTypeConverter()->convertType(op.getType());
    auto input = adaptor.getIn();

    rewriter.replaceOpWithNewOp<stablehlo::ConvertOp>(op, type, input);
    return success();
  }
};

class ConvertArithTruncIOp : public OpConversionPattern<arith::TruncIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::TruncIOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::TruncIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (op.getOverflowFlags() != mlir::arith::IntegerOverflowFlags::none) {
      return op->emitError("arith::TruncIOp: overflowFlags are not supported");
    }
    auto type = getTypeConverter()->convertType(op.getType());
    auto input = adaptor.getIn();

    rewriter.replaceOpWithNewOp<stablehlo::ConvertOp>(op, type, input);
    return success();
  }
};

class ConvertTensorFromElementsOp
    : public OpConversionPattern<tensor::FromElementsOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename tensor::FromElementsOp::Adaptor;
  LogicalResult
  matchAndRewrite(tensor::FromElementsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    int64_t concatLength = op.getElements().size();
    RankedTensorType type = dyn_cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));
    if (!type) {
      return rewriter.notifyMatchFailure(op,
                                         "type conversion for FromElementsOp "
                                         "failed to return a RankedTensorType");
    }
    Type elementTy = type.getElementType();
    auto inputs = adaptor.getOperands();
    SmallVector<Value> broadcastedInputs;
    for (auto input : inputs) {
      auto inputTy = dyn_cast<RankedTensorType>(input.getType());
      if (!inputTy) {
        return rewriter.notifyMatchFailure(
            op, "type conversion for FromElementsOp operand failed to return a "
                "RankedTensorType");
      }
      auto reshapeTy = RankedTensorType::get({1}, inputTy.getElementType());
      Value reshapeOp =
          stablehlo::ReshapeOp::create(rewriter, op.getLoc(), reshapeTy, input);
      broadcastedInputs.emplace_back(reshapeOp);
    }
    Type concatTy =
        RankedTensorType::get(ArrayRef<int64_t>({concatLength}), elementTy);
    Value concatOp = stablehlo::ConcatenateOp::create(
        rewriter, op.getLoc(), concatTy, broadcastedInputs, 0);
    rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(op, type, concatOp);
    return success();
  }
};

class ConvertArithMulIOp : public OpConversionPattern<arith::MulIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::MulIOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::MulIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    rewriter.replaceOpWithNewOp<stablehlo::MulOp>(op, type, lhs, rhs);
    return success();
  }
};

class ConvertArithNegFOp : public OpConversionPattern<arith::NegFOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::NegFOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::NegFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto operand = adaptor.getOperand();
    rewriter.replaceOpWithNewOp<stablehlo::NegOp>(op, type, operand);
    return success();
  }
};

class ConvertArithOrIOp : public OpConversionPattern<arith::OrIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::OrIOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::OrIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    rewriter.replaceOpWithNewOp<stablehlo::OrOp>(op, type, lhs, rhs);
    return success();
  }
};

class ConvertArithRemFOp : public OpConversionPattern<arith::RemFOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::RemFOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::RemFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = dyn_cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getType()));
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    rewriter.replaceOpWithNewOp<stablehlo::RemOp>(op, type, lhs, rhs);
    return success();
  }
};

class ConvertArithRemSIOp : public OpConversionPattern<arith::RemSIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::RemSIOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::RemSIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = dyn_cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getType()));
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    rewriter.replaceOpWithNewOp<stablehlo::RemOp>(op, type, lhs, rhs);
    return success();
  }
};

class ConvertArithRemUIOp : public OpConversionPattern<arith::RemUIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::RemUIOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::RemUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = dyn_cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getType()));
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    rewriter.replaceOpWithNewOp<stablehlo::RemOp>(op, type, lhs, rhs);
    return success();
  }
};

class ConvertArithShLIOp : public OpConversionPattern<arith::ShLIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::ShLIOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::ShLIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    rewriter.replaceOpWithNewOp<stablehlo::ShiftLeftOp>(op, type, lhs, rhs);
    return success();
  }
};

class ConvertArithShRSIOp : public OpConversionPattern<arith::ShRSIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::ShRSIOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::ShRSIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    rewriter.replaceOpWithNewOp<stablehlo::ShiftRightArithmeticOp>(op, type,
                                                                   lhs, rhs);
    return success();
  }
};

class ConvertArithShRUIOp : public OpConversionPattern<arith::ShRUIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::ShRUIOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::ShRUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    rewriter.replaceOpWithNewOp<stablehlo::ShiftRightLogicalOp>(op, type, lhs,
                                                                rhs);
    return success();
  }
};

class ConvertArithSIToFPOp : public OpConversionPattern<arith::SIToFPOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::SIToFPOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::SIToFPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto in = adaptor.getIn();
    rewriter.replaceOpWithNewOp<stablehlo::ConvertOp>(op, type, in);
    return success();
  }
};

class ConvertArithSubFOp : public OpConversionPattern<arith::SubFOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::SubFOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::SubFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    rewriter.replaceOpWithNewOp<stablehlo::SubtractOp>(op, type, lhs, rhs);
    return success();
  }
};

class ConvertArithSubIOp : public OpConversionPattern<arith::SubIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::SubIOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::SubIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    rewriter.replaceOpWithNewOp<stablehlo::SubtractOp>(op, type, lhs, rhs);
    return success();
  }
};

class ConvertArithUIToFPOp : public OpConversionPattern<arith::UIToFPOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::UIToFPOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::UIToFPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto in = adaptor.getIn();
    rewriter.replaceOpWithNewOp<stablehlo::ConvertOp>(op, type, in);
    return success();
  }
};

class ConvertArithXOrIOp : public OpConversionPattern<arith::XOrIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename arith::XOrIOp::Adaptor;
  LogicalResult
  matchAndRewrite(arith::XOrIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    rewriter.replaceOpWithNewOp<stablehlo::XorOp>(op, type, lhs, rhs);
    return success();
  }
};

class ConvertMathIPowIOp : public OpConversionPattern<math::IPowIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename math::IPowIOp::Adaptor;
  LogicalResult
  matchAndRewrite(math::IPowIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto type = getTypeConverter()->convertType(op.getType());
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    rewriter.replaceOpWithNewOp<stablehlo::PowOp>(op, type, lhs, rhs);
    return success();
  }
};

class ConvertTensorDimOp : public OpConversionPattern<tensor::DimOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename tensor::DimOp::Adaptor;

  LogicalResult
  matchAndRewrite(tensor::DimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto input = adaptor.getSource();
    auto dim = op.getConstantIndex();
    if (!dim)
      return rewriter.notifyMatchFailure(
          op, "tensor::DimOp does not have constant dim");

    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "input is not a ranked tensor");

    auto dimSize = inputType.getDimSize(*dim);
    auto i64Type = RankedTensorType::get({}, rewriter.getI64Type());

    if (ShapedType::isDynamic(dimSize)) {
      auto i32Type = RankedTensorType::get({}, rewriter.getI32Type());
      auto getDimSize = stablehlo::GetDimensionSizeOp::create(
          rewriter, op.getLoc(), i32Type, input,
          rewriter.getI64IntegerAttr(*dim));
      rewriter.replaceOpWithNewOp<stablehlo::ConvertOp>(op, i64Type,
                                                        getDimSize);
    } else {
      auto constantAttr = DenseElementsAttr::get(
          i64Type, rewriter.getI64IntegerAttr(dimSize));
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, i64Type,
                                                         constantAttr);
    }

    return success();
  }
};

class ConvertTensorExtractOp : public OpConversionPattern<tensor::ExtractOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename tensor::ExtractOp::Adaptor;
  LogicalResult
  matchAndRewrite(tensor::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto input = adaptor.getTensor();
    auto index = adaptor.getIndices();
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType) {
      return rewriter.notifyMatchFailure(
          op, "tensor::ExtractOp does not have ranked tensor input");
    }
    if (inputType.getShape().size() == 0) {
      op.replaceAllUsesWith(input);
      op.erase();
      return success();
    }
    SmallVector<int64_t> sliceSizes(inputType.getShape().size(), 1);
    rewriter.replaceOpWithNewOp<stablehlo::DynamicSliceOp>(
        op, input, index, rewriter.getDenseI64ArrayAttr(sliceSizes));
    return success();
  }
};

class ConvertTensorBitcastOp : public OpConversionPattern<tensor::BitcastOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename tensor::BitcastOp::Adaptor;
  LogicalResult
  matchAndRewrite(tensor::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Get the converted result type from the type converter
    Type resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType)
      return failure();

    // Replace the tensor bitcast with StableHLO bitcast_convert
    rewriter.replaceOpWithNewOp<stablehlo::BitcastConvertOp>(
        op, resultType, adaptor.getSource());

    return success();
  }
};

class ConvertTensorCastOp : public OpConversionPattern<tensor::CastOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename tensor::CastOp::Adaptor;

  LogicalResult
  matchAndRewrite(tensor::CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto input = adaptor.getSource();
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto resultType = dyn_cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getType()));

    if (!inputType || !resultType)
      return failure();

    // No-op cast (types identical or shape refinement with dynamic dims)
    if (inputType == resultType) {
      rewriter.replaceOp(op, input);
      return success();
    }

    if (resultType.hasStaticShape()) {
      rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(op, resultType, input);
      return success();
    }

    // Shape refinement cast (e.g., tensor<?x?x?xf32> -> tensor<1x?x?xf32>):
    // reconstruct the result shape at runtime using DynamicReshapeOp
    auto rank = resultType.getRank();
    auto resultShape = resultType.getShape();
    SmallVector<Value> dimValues;
    for (int64_t i = 0; i < rank; ++i) {
      if (resultShape[i] != ShapedType::kDynamic) {
        dimValues.push_back(arith::ConstantOp::create(
            rewriter, op.getLoc(),
            rewriter.getI64IntegerAttr(resultShape[i])));
      } else {
        Value dimIdx = arith::IndexCastOp::create(
            rewriter, op.getLoc(), rewriter.getI64Type(),
            tensor::DimOp::create(rewriter, op.getLoc(), input, i));
        dimValues.push_back(dimIdx);
      }
    }
    Value shapeTensor = tensor::FromElementsOp::create(
        rewriter, op.getLoc(),
        RankedTensorType::get({rank}, rewriter.getI64Type()), dimValues);
    rewriter.replaceOpWithNewOp<stablehlo::DynamicReshapeOp>(
        op, resultType, input, shapeTensor);
    return success();
  }
};

class ConvertTensorCollapseShapeOp
    : public OpConversionPattern<tensor::CollapseShapeOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename tensor::CollapseShapeOp::Adaptor;

  LogicalResult
  matchAndRewrite(tensor::CollapseShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto input = adaptor.getSrc();
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto resultType = dyn_cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getType()));

    if (!inputType || !resultType)
      return failure();

    // tensor.collapse_shape spec guarantees that the reassociation dims are
    // contiguous
    rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(op, resultType, input);

    return success();
  }
};

class ConvertTensorExpandShapeOp
    : public OpConversionPattern<tensor::ExpandShapeOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename tensor::ExpandShapeOp::Adaptor;

  LogicalResult
  matchAndRewrite(tensor::ExpandShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto input = adaptor.getSrc();
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto resultType = dyn_cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getType()));

    if (!inputType || !resultType)
      return failure();

    // tensor.expand_shape spec guarantees that the reassociation dims are
    // contiguous
    rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(op, resultType, input);

    return success();
  }
};

class ConvertTensorConcatOp : public OpConversionPattern<tensor::ConcatOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename tensor::ConcatOp::Adaptor;

  LogicalResult
  matchAndRewrite(tensor::ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Get the converted result type from the type converter
    Type resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType)
      return failure();
    // Get the concatenation dimension
    int64_t dim = op.getDim();
    // Replace the tensor concat with StableHLO concatenate
    rewriter.replaceOpWithNewOp<stablehlo::ConcatenateOp>(
        op, resultType, adaptor.getInputs(), dim);

    return success();
  }
};

class ConvertTensorEmptyOp : public OpConversionPattern<tensor::EmptyOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename tensor::EmptyOp::Adaptor;

  LogicalResult
  matchAndRewrite(tensor::EmptyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType)
      return failure();

    // Create a zero-initialized constant tensor
    auto zeroAttr = rewriter.getZeroAttr(resultType);
    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, zeroAttr);

    return success();
  }
};

class ConvertTensorReshapeOp : public OpConversionPattern<tensor::ReshapeOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename tensor::ReshapeOp::Adaptor;

  LogicalResult
  matchAndRewrite(tensor::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Get the converted result type
    auto resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType)
      return failure();

    auto resultTensorType = mlir::dyn_cast<RankedTensorType>(resultType);
    if (!resultTensorType)
      return failure();

    if (!resultTensorType.hasStaticShape())
      return rewriter.notifyMatchFailure(op,
                                         "only static shapes are supported");

    Value source = adaptor.getSource();
    auto sourceType = mlir::dyn_cast<RankedTensorType>(source.getType());
    if (!sourceType || !sourceType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "only static source shapes are supported");

    rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(op, resultTensorType,
                                                      source);

    return success();
  }
};

class ConvertTensorExtractSliceOp
    : public OpConversionPattern<tensor::ExtractSliceOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename tensor::ExtractSliceOp::Adaptor;

  LogicalResult
  matchAndRewrite(tensor::ExtractSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Get the converted result type
    auto resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType)
      return failure();

    auto resultTensorType = mlir::dyn_cast<RankedTensorType>(resultType);
    if (!resultTensorType)
      return failure();

    Location loc = op.getLoc();
    Value source = adaptor.getSource();

    // Get mixed offsets, sizes, and strides
    SmallVector<OpFoldResult> mixedOffsets = op.getMixedOffsets();
    SmallVector<OpFoldResult> mixedSizes = op.getMixedSizes();
    SmallVector<OpFoldResult> mixedStrides = op.getMixedStrides();

    // Check if strides and sizes are static
    bool stridesStatic = llvm::all_of(mixedStrides, [](OpFoldResult stride) {
      return llvm::isa<Attribute>(stride);
    });
    bool sizesStatic = llvm::all_of(mixedSizes, [](OpFoldResult size) {
      return llvm::isa<Attribute>(size);
    });

    // Strides and sizes must be static
    if (!stridesStatic || !sizesStatic)
      return rewriter.notifyMatchFailure(
          op, "only static strides and sizes are supported");

    // Check if all strides are 1
    bool allStridesOne = llvm::all_of(mixedStrides, [](OpFoldResult stride) {
      auto attr = llvm::cast<Attribute>(stride);
      return mlir::cast<IntegerAttr>(attr).getInt() == 1;
    });

    if (!allStridesOne)
      return rewriter.notifyMatchFailure(op, "only stride=1 is supported");

    // Extract static sizes
    SmallVector<int64_t> staticSizes;
    for (auto size : mixedSizes) {
      staticSizes.push_back(
          mlir::cast<IntegerAttr>(llvm::cast<Attribute>(size)).getInt());
    }
    // Check if offsets are all static
    bool offsetsStatic = llvm::all_of(mixedOffsets, [](OpFoldResult offset) {
      return llvm::isa<Attribute>(offset);
    });

    if (offsetsStatic) {
      // Case 1: All static - use SliceOp
      SmallVector<int64_t> staticOffsets;
      for (auto offset : mixedOffsets) {
        staticOffsets.push_back(
            mlir::cast<IntegerAttr>(llvm::cast<Attribute>(offset)).getInt());
      }

      // Calculate limits (offset + size)
      SmallVector<int64_t> staticLimits;
      for (size_t i = 0; i < staticOffsets.size(); ++i) {
        staticLimits.push_back(staticOffsets[i] + staticSizes[i]);
      }

      // Strides are all 1
      SmallVector<int64_t> staticStrides(mixedStrides.size(), 1);

      rewriter.replaceOpWithNewOp<stablehlo::SliceOp>(
          op, resultTensorType, source,
          rewriter.getDenseI64ArrayAttr(staticOffsets),
          rewriter.getDenseI64ArrayAttr(staticLimits),
          rewriter.getDenseI64ArrayAttr(staticStrides));

      return success();
    }

    // Case 2: Dynamic offsets, static sizes, stride=1 - use DynamicSliceOp
    SmallVector<Value> startIndices;

    // Get the converted dynamic offsets from the adaptor
    auto dynamicOffsets = adaptor.getOffsets();
    size_t dynamicOffsetIdx = 0;

    for (auto offset : mixedOffsets) {
      if (auto attr = llvm::dyn_cast<Attribute>(offset)) {
        // For static offsets, create a StableHLO constant directly
        int64_t offsetInt = mlir::cast<IntegerAttr>(attr).getInt();
        auto scalarTensorType =
            RankedTensorType::get({}, rewriter.getI64Type());
        auto constAttr = DenseIntElementsAttr::get(scalarTensorType, offsetInt);
        startIndices.push_back(
            stablehlo::ConstantOp::create(rewriter, loc, constAttr));
      } else {
        // For dynamic offsets, use the converted value from adaptor
        Value offsetValue = dynamicOffsets[dynamicOffsetIdx++];
        // The adaptor should have already converted this to a 0-D tensor
        auto offsetType =
            mlir::dyn_cast<RankedTensorType>(offsetValue.getType());
        if (!offsetType || offsetType.getRank() != 0) {
          return rewriter.notifyMatchFailure(
              op, "expected dynamic offset to be converted to 0-D tensor");
        }

        // Verify element type is i64
        if (!offsetType.getElementType().isInteger(64)) {
          // Need to convert to i64
          auto i64TensorType = RankedTensorType::get({}, rewriter.getI64Type());
          offsetValue = stablehlo::ConvertOp::create(
              rewriter, loc, i64TensorType, offsetValue);
        }
        startIndices.push_back(offsetValue);
      }
    }
    rewriter.replaceOpWithNewOp<stablehlo::DynamicSliceOp>(
        op, resultTensorType, source, startIndices,
        rewriter.getDenseI64ArrayAttr(staticSizes));

    return success();
  }
};

class ConvertTensorRankOp : public OpConversionPattern<tensor::RankOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename tensor::RankOp::Adaptor;

  LogicalResult
  matchAndRewrite(tensor::RankOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto inputType = dyn_cast<RankedTensorType>(adaptor.getTensor().getType());
    if (!inputType)
      return failure();

    int64_t rank = inputType.getRank();

    // Always create tensor<i64> constant for rank
    auto tensorType = RankedTensorType::get({}, rewriter.getI64Type());
    auto denseAttr = DenseElementsAttr::get(
        tensorType, rewriter.getIntegerAttr(rewriter.getI64Type(), rank));

    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, tensorType,
                                                       denseAttr);

    return success();
  }
};

class ConvertTensorSplatOp : public OpConversionPattern<tensor::SplatOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename tensor::SplatOp::Adaptor;

  LogicalResult
  matchAndRewrite(tensor::SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType)
      return failure();

    auto resultTensorType = llvm::dyn_cast<RankedTensorType>(resultType);
    // Only support static shapes
    if (!resultTensorType || !resultTensorType.hasStaticShape())
      return failure();

    Value input = adaptor.getInput();

    // Convert scalar to 0D tensor using tensor.from_elements
    if (!llvm::isa<TensorType>(input.getType())) {
      auto scalarTensorType = RankedTensorType::get({}, input.getType());
      input = tensor::FromElementsOp::create(
          rewriter, op.getLoc(), scalarTensorType, ValueRange{input});
    }
    rewriter.replaceOpWithNewOp<stablehlo::BroadcastInDimOp>(
        op, resultTensorType, input, rewriter.getDenseI64ArrayAttr({}));

    return success();
  }
};

class ConvertTensorPadOp : public OpConversionPattern<tensor::PadOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename tensor::PadOp::Adaptor;

  LogicalResult
  matchAndRewrite(tensor::PadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType)
      return failure();

    auto resultTensorType = llvm::dyn_cast<RankedTensorType>(resultType);
    // Only support static shapes
    if (!resultTensorType || !resultTensorType.hasStaticShape())
      return failure();

    Value source = adaptor.getSource();

    // If nofold is false and input/output shapes are the same, just fold away
    // the pad
    auto sourceTensorType = llvm::dyn_cast<RankedTensorType>(source.getType());
    if (!sourceTensorType)
      return failure();

    if (!op.getNofold() &&
        sourceTensorType.getShape() == resultTensorType.getShape()) {
      rewriter.replaceOp(op, source);
      return success();
    }

    // Low and high padding widths must be constants
    auto staticLow = adaptor.getStaticLow();
    auto staticHigh = adaptor.getStaticHigh();

    if (staticLow.empty() || staticHigh.empty())
      return failure();

    auto yieldOp =
        llvm::dyn_cast<tensor::YieldOp>(op.getRegion().front().getTerminator());
    if (!yieldOp)
      return failure();

    Value paddingValue = rewriter.getRemappedValue(yieldOp.getValue());

    auto paddingValueType =
        llvm::dyn_cast<RankedTensorType>(paddingValue.getType());
    if (!paddingValueType || paddingValueType.getRank() != 0)
      return failure();

    // StableHLO pad requires interior padding (all zeros for tensor.pad)
    auto rank = llvm::cast<RankedTensorType>(source.getType()).getRank();
    SmallVector<int64_t> interiorPadding(rank, 0);

    rewriter.replaceOpWithNewOp<stablehlo::PadOp>(
        op, resultTensorType, source, paddingValue, staticLow, staticHigh,
        rewriter.getDenseI64ArrayAttr(interiorPadding));

    return success();
  }
};

class ConvertTensorInsertOp : public OpConversionPattern<tensor::InsertOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename tensor::InsertOp::Adaptor;

  LogicalResult
  matchAndRewrite(tensor::InsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Get the converted result type
    auto resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType)
      return failure();

    auto resultTensorType = mlir::dyn_cast<RankedTensorType>(resultType);
    if (!resultTensorType)
      return failure();

    Location loc = op.getLoc();
    Value scalar = adaptor.getScalar();
    Value dest = adaptor.getDest();

    auto indices = adaptor.getIndices();
    SmallVector<Value> startIndices;

    for (Value index : indices) {
      auto indexType = mlir::dyn_cast<RankedTensorType>(index.getType());
      if (!indexType || indexType.getRank() != 0)
        return rewriter.notifyMatchFailure(
            op, "expected index to be converted to 0-D tensor");
      if (!indexType.getElementType().isInteger(64)) {
        // Need to convert to i64
        auto i64TensorType = RankedTensorType::get({}, rewriter.getI64Type());
        index =
            stablehlo::ConvertOp::create(rewriter, loc, i64TensorType, index);
      }

      startIndices.push_back(index);
    }

    // The scalar needs to be reshaped to match the rank of the destination
    // DynamicUpdateSliceOp requires update to have same rank as operand
    // with each dimension = 1 for a single element update
    auto scalarType = mlir::dyn_cast<RankedTensorType>(scalar.getType());

    int64_t destRank = resultTensorType.getRank();
    Value update = scalar;

    // Reshape scalar to match destination rank (all dimensions = 1)
    if (scalarType.getRank() != destRank) {
      SmallVector<int64_t> updateShape(destRank, 1);
      auto updateType =
          RankedTensorType::get(updateShape, scalarType.getElementType());
      update = stablehlo::ReshapeOp::create(rewriter, loc, updateType, scalar);
    }

    rewriter.replaceOpWithNewOp<stablehlo::DynamicUpdateSliceOp>(
        op, resultTensorType, dest, update, startIndices);

    return success();
  }
};

class ConvertTensorInsertSliceOp
    : public OpConversionPattern<tensor::InsertSliceOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename tensor::InsertSliceOp::Adaptor;

  LogicalResult
  matchAndRewrite(tensor::InsertSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Get the converted result type
    auto resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType)
      return failure();

    auto resultTensorType = mlir::dyn_cast<RankedTensorType>(resultType);
    if (!resultTensorType)
      return failure();

    Location loc = op.getLoc();
    Value source = adaptor.getSource(); // The slice to insert
    Value dest = adaptor.getDest();     // The destination tensor

    // Get mixed offsets, sizes, and strides
    SmallVector<OpFoldResult> mixedOffsets = op.getMixedOffsets();
    SmallVector<OpFoldResult> mixedSizes = op.getMixedSizes();
    SmallVector<OpFoldResult> mixedStrides = op.getMixedStrides();

    // Check if strides are static and all 1
    bool stridesStatic = llvm::all_of(mixedStrides, [](OpFoldResult stride) {
      return llvm::isa<Attribute>(stride);
    });

    if (!stridesStatic)
      return rewriter.notifyMatchFailure(op,
                                         "only static strides are supported");

    bool allStridesOne = llvm::all_of(mixedStrides, [](OpFoldResult stride) {
      auto attr = llvm::cast<Attribute>(stride);
      return mlir::cast<IntegerAttr>(attr).getInt() == 1;
    });

    if (!allStridesOne)
      return rewriter.notifyMatchFailure(op, "only stride=1 is supported");

    // Build start indices for DynamicUpdateSliceOp
    SmallVector<Value> startIndices;

    // Get the converted dynamic offsets from the adaptor
    auto dynamicOffsets = adaptor.getOffsets();
    size_t dynamicOffsetIdx = 0;

    for (auto offset : mixedOffsets) {
      if (auto attr = llvm::dyn_cast<Attribute>(offset)) {
        // For static offsets, create a StableHLO constant
        int64_t offsetInt = mlir::cast<IntegerAttr>(attr).getInt();
        auto scalarTensorType =
            RankedTensorType::get({}, rewriter.getI64Type());
        auto constAttr = DenseIntElementsAttr::get(scalarTensorType, offsetInt);
        startIndices.push_back(
            stablehlo::ConstantOp::create(rewriter, loc, constAttr));
      } else {
        // For dynamic offsets, use the converted value from adaptor
        Value offsetValue = dynamicOffsets[dynamicOffsetIdx++];

        // The adaptor should have already converted this to a 0-D tensor
        auto offsetType =
            mlir::dyn_cast<RankedTensorType>(offsetValue.getType());
        if (!offsetType || offsetType.getRank() != 0)
          return rewriter.notifyMatchFailure(
              op, "expected dynamic offset to be converted to 0-D tensor");

        // Verify element type is i64
        if (!offsetType.getElementType().isInteger(64)) {
          // Need to convert to i64
          auto i64TensorType = RankedTensorType::get({}, rewriter.getI64Type());
          offsetValue = stablehlo::ConvertOp::create(
              rewriter, loc, i64TensorType, offsetValue);
        }

        startIndices.push_back(offsetValue);
      }
    }
    rewriter.replaceOpWithNewOp<stablehlo::DynamicUpdateSliceOp>(
        op, resultTensorType, dest, source, startIndices);

    return success();
  }
};

// Convert shape.shape_of (which produces tensor<Nxindex>) into stablehlo
// get_dimension_size + concatenate (producing tensor<Nxi64>).
class ConvertShapeShapeOfOp
    : public OpConversionPattern<shape::ShapeOfOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(shape::ShapeOfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = adaptor.getArg();
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType)
      return failure();

    int64_t rank = inputType.getRank();
    auto i64Type = rewriter.getI64Type();
    auto scalarI32Type = RankedTensorType::get({}, rewriter.getI32Type());

    SmallVector<Value> dimValues;
    for (int64_t i = 0; i < rank; ++i) {
      Value dim = stablehlo::GetDimensionSizeOp::create(
          rewriter, loc, scalarI32Type, input,
          rewriter.getI64IntegerAttr(i));
      auto i64ScalarType = RankedTensorType::get({}, i64Type);
      Value dimI64 = stablehlo::ConvertOp::create(
          rewriter, loc, i64ScalarType, dim);
      auto i64_1dType = RankedTensorType::get({1}, i64Type);
      Value reshaped = stablehlo::ReshapeOp::create(
          rewriter, loc, i64_1dType, dimI64);
      dimValues.push_back(reshaped);
    }

    auto resultType = RankedTensorType::get({rank}, i64Type);
    Value result = stablehlo::ConcatenateOp::create(
        rewriter, loc, resultType, dimValues,
        rewriter.getI64IntegerAttr(0));
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Convert shape.num_elements (tensor<Nxindex> -> index) into a stablehlo
// reduce with multiplication, producing a scalar tensor<i64>.
class ConvertShapeNumElementsOp
    : public OpConversionPattern<shape::NumElementsOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(shape::NumElementsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value shape = adaptor.getShape();
    auto shapeType = dyn_cast<RankedTensorType>(shape.getType());
    if (!shapeType)
      return failure();

    auto i64Type = rewriter.getI64Type();
    auto scalarType = RankedTensorType::get({}, i64Type);

    if (shapeType.getElementType().isIndex()) {
      auto convertedType =
          RankedTensorType::get(shapeType.getShape(), i64Type);
      shape = stablehlo::ConvertOp::create(rewriter, loc, convertedType, shape);
      shapeType = convertedType;
    }

    Value initValue = stablehlo::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(i64Type, 1));
    auto reduceOp = stablehlo::ReduceOp::create(
        rewriter, loc, scalarType, ValueRange{shape}, ValueRange{initValue},
        rewriter.getDenseI64ArrayAttr({0}));

    Block &block = reduceOp.getBody().emplaceBlock();
    block.addArgument(scalarType, loc);
    block.addArgument(scalarType, loc);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&block);
      Value mul = stablehlo::MulOp::create(rewriter, loc, block.getArgument(0),
                                           block.getArgument(1));
      stablehlo::ReturnOp::create(rewriter, loc, mul);
    }

    rewriter.replaceOp(op, reduceOp.getResults()[0]);
    return success();
  }
};

namespace {
class ConvertTorchToStablehloCustomLowering
    : public impl::ConvertTorchToStablehloCustomLoweringBase<
          ConvertTorchToStablehloCustomLowering> {
public:
  using impl::ConvertTorchToStablehloCustomLoweringBase<
      ConvertTorchToStablehloCustomLowering>::
      ConvertTorchToStablehloCustomLoweringBase;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<stablehlo::StablehloDialect>();
    registry.insert<Torch::TorchDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<math::MathDialect>();
    registry.insert<shape::ShapeDialect>();
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
      target.addLegalDialect<chlo::ChloDialect, stablehlo::StablehloDialect,
                             func::FuncDialect>();

      TypeConverter typeConverter;
      typeConverter.addConversion([&context](Type type) -> Type {
        if (type.isIntOrFloat()) {
          return RankedTensorType::get({}, type);
        }
        if (type.isIndex()) {
          return RankedTensorType::get({}, IntegerType::get(context, 64));
        }
        if (auto tensorType = mlir::dyn_cast<RankedTensorType>(type)) {
          if (tensorType.getElementType().isIndex()) {
            return RankedTensorType::get(tensorType.getShape(),
                                         IntegerType::get(context, 64),
                                         tensorType.getEncoding());
          }
          return tensorType;
        }
        return type;
      });
      TorchConversion::setupBackendTypeConversionForStablehlo(target,
                                                              typeConverter);
      RewritePatternSet patterns(context);

      target.addIllegalOp<shape::ShapeOfOp>();
      patterns.add<ConvertShapeShapeOfOp>(typeConverter, context);

      target.addIllegalOp<shape::NumElementsOp>();
      patterns.add<ConvertShapeNumElementsOp>(typeConverter, context);

      target.addIllegalOp<tensor::DimOp>();
      patterns.add<ConvertTensorDimOp>(typeConverter, context);

      target.addIllegalOp<tensor::FromElementsOp>();
      patterns.add<ConvertTensorFromElementsOp>(typeConverter, context);

      target.addIllegalOp<tensor::ExtractOp>();
      patterns.add<ConvertTensorExtractOp>(typeConverter, context);

      target.addIllegalOp<tensor::BitcastOp>();
      patterns.add<ConvertTensorBitcastOp>(typeConverter, context);

      target.addIllegalOp<tensor::ConcatOp>();
      patterns.add<ConvertTensorConcatOp>(typeConverter, context);

      target.addIllegalOp<tensor::CastOp>();
      patterns.add<ConvertTensorCastOp>(typeConverter, context);

      target.addIllegalOp<tensor::CollapseShapeOp>();
      patterns.add<ConvertTensorCollapseShapeOp>(typeConverter, context);

      target.addIllegalOp<tensor::ExpandShapeOp>();
      patterns.add<ConvertTensorExpandShapeOp>(typeConverter, context);

      target.addIllegalOp<tensor::RankOp>();
      patterns.add<ConvertTensorRankOp>(typeConverter, context);

      target.addIllegalOp<tensor::ReshapeOp>();
      patterns.add<ConvertTensorReshapeOp>(typeConverter, context);

      target.addIllegalOp<tensor::EmptyOp>();
      patterns.add<ConvertTensorEmptyOp>(typeConverter, context);

      target.addIllegalOp<tensor::ExtractSliceOp>();
      patterns.add<ConvertTensorExtractSliceOp>(typeConverter, context);

      target.addIllegalOp<tensor::SplatOp>();
      patterns.add<ConvertTensorSplatOp>(typeConverter, context);

      target.addIllegalOp<tensor::InsertOp>();
      patterns.add<ConvertTensorInsertOp>(typeConverter, context);

      target.addIllegalOp<tensor::InsertSliceOp>();
      patterns.add<ConvertTensorInsertSliceOp>(typeConverter, context);

      target.addIllegalOp<tensor::PadOp>();
      patterns.add<ConvertTensorPadOp>(typeConverter, context);

      target.addIllegalOp<arith::ExtSIOp>();
      patterns.add<ConvertArithExtSIOp>(typeConverter, context);

      target.addIllegalOp<arith::MaxSIOp>();
      patterns.add<ConvertArithMaxSIOp>(typeConverter, context);

      target.addIllegalOp<arith::MinSIOp>();
      patterns.add<ConvertArithMinSIOp>(typeConverter, context);

      target.addIllegalOp<arith::AddIOp>();
      patterns.add<ConvertArithAddIOp>(typeConverter, context);

      target.addIllegalOp<arith::CmpIOp>();
      patterns.add<ConvertArithCmpIOp>(typeConverter, context);

      target.addIllegalOp<arith::SelectOp>();
      patterns.add<ConvertArithSelectOp>(typeConverter, context);

      target.addIllegalOp<arith::ConstantOp>();
      patterns.add<ConvertArithConstantOp>(typeConverter, context);

      target.addIllegalOp<arith::IndexCastOp>();
      patterns.add<ConvertArithIndexCastOp>(typeConverter, context);

      target.addIllegalOp<arith::ExtFOp>();
      patterns.add<ConvertArithExtFOp>(typeConverter, context);

      target.addIllegalOp<arith::ExtUIOp>();
      patterns.add<ConvertArithExtUIOp>(typeConverter, context);

      target.addIllegalOp<arith::MaxUIOp>();
      patterns.add<ConvertArithMaxUIOp>(typeConverter, context);

      target.addIllegalOp<arith::MinUIOp>();
      patterns.add<ConvertArithMinUIOp>(typeConverter, context);

      target.addIllegalOp<arith::AddFOp>();
      patterns.add<ConvertArithAddFOp>(typeConverter, context);

      target.addIllegalOp<arith::AndIOp>();
      patterns.add<ConvertArithAndIOp>(typeConverter, context);

      target.addIllegalOp<arith::BitcastOp>();
      patterns.add<ConvertArithBitcastOp>(typeConverter, context);

      target.addIllegalOp<arith::CmpFOp>();
      patterns.add<ConvertArithCmpFOp>(typeConverter, context);

      target.addIllegalOp<arith::DivFOp>();
      patterns.add<ConvertArithDivFOp>(typeConverter, context);

      target.addIllegalOp<arith::DivSIOp>();
      patterns.add<ConvertArithDivSIOp>(typeConverter, context);

      target.addIllegalOp<arith::DivUIOp>();
      patterns.add<ConvertArithDivUIOp>(typeConverter, context);

      target.addIllegalOp<arith::CeilDivSIOp>();
      patterns.add<ConvertArithCeilDivSIOp>(typeConverter, context);

      target.addIllegalOp<arith::FloorDivSIOp>();
      patterns.add<ConvertArithFloorDivSIOp>(typeConverter, context);

      target.addIllegalOp<arith::FPToSIOp>();
      patterns.add<ConvertArithFPToSIOp>(typeConverter, context);

      target.addIllegalOp<arith::FPToUIOp>();
      patterns.add<ConvertArithFPToUIOp>(typeConverter, context);

      target.addIllegalOp<arith::IndexCastUIOp>();
      patterns.add<ConvertArithIndexCastUIOp>(typeConverter, context);

      target.addIllegalOp<arith::MaximumFOp>();
      patterns.add<ConvertArithMaximumFOp>(typeConverter, context);

      target.addIllegalOp<arith::MinimumFOp>();
      patterns.add<ConvertArithMinimumFOp>(typeConverter, context);

      target.addIllegalOp<arith::MulFOp>();
      patterns.add<ConvertArithMulFOp>(typeConverter, context);

      target.addIllegalOp<arith::MulIOp>();
      patterns.add<ConvertArithMulIOp>(typeConverter, context);

      target.addIllegalOp<arith::NegFOp>();
      patterns.add<ConvertArithNegFOp>(typeConverter, context);

      target.addIllegalOp<arith::OrIOp>();
      patterns.add<ConvertArithOrIOp>(typeConverter, context);

      target.addIllegalOp<arith::RemFOp>();
      patterns.add<ConvertArithRemFOp>(typeConverter, context);

      target.addIllegalOp<arith::RemSIOp>();
      patterns.add<ConvertArithRemSIOp>(typeConverter, context);

      target.addIllegalOp<arith::RemUIOp>();
      patterns.add<ConvertArithRemUIOp>(typeConverter, context);

      target.addIllegalOp<arith::ShLIOp>();
      patterns.add<ConvertArithShLIOp>(typeConverter, context);

      target.addIllegalOp<arith::ShRSIOp>();
      patterns.add<ConvertArithShRSIOp>(typeConverter, context);

      target.addIllegalOp<arith::ShRUIOp>();
      patterns.add<ConvertArithShRUIOp>(typeConverter, context);

      target.addIllegalOp<arith::SIToFPOp>();
      patterns.add<ConvertArithSIToFPOp>(typeConverter, context);

      target.addIllegalOp<arith::SubFOp>();
      patterns.add<ConvertArithSubFOp>(typeConverter, context);

      target.addIllegalOp<arith::SubIOp>();
      patterns.add<ConvertArithSubIOp>(typeConverter, context);

      target.addIllegalOp<arith::TruncFOp>();
      patterns.add<ConvertArithTruncFOp>(typeConverter, context);

      target.addIllegalOp<arith::TruncIOp>();
      patterns.add<ConvertArithTruncIOp>(typeConverter, context);

      target.addIllegalOp<arith::UIToFPOp>();
      patterns.add<ConvertArithUIToFPOp>(typeConverter, context);

      target.addIllegalOp<arith::XOrIOp>();
      patterns.add<ConvertArithXOrIOp>(typeConverter, context);

      target.addIllegalOp<math::IPowIOp>();
      patterns.add<ConvertMathIPowIOp>(typeConverter, context);

      // Convert function signatures so scalar i64/f64 args become tensor<i64>/tensor<f64>.
      populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
          patterns, typeConverter);
      target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
        return typeConverter.isSignatureLegal(op.getFunctionType()) &&
               typeConverter.isLegal(&op.getBody());
      });
      populateReturnOpTypeConversionPattern(patterns, typeConverter);
      target.addDynamicallyLegalOp<func::ReturnOp>(
          [&](func::ReturnOp op) { return typeConverter.isLegal(op); });

      result =
          applyPartialConversion(getOperation(), target, std::move(patterns));
    });

    if (crashed || failed(result)) {
      if (crashed)
        getOperation()->emitError() << "ConvertTorchToStablehloCustomLowering "
                                       "pass crashed unexpectedly";
      signalPassFailure();
    }
  }
};
} // namespace

// Default pass creation function (required by tablegen)
std::unique_ptr<OperationPass<func::FuncOp>>
createConvertTorchToStablehloCustomLoweringPass() {
  return std::make_unique<ConvertTorchToStablehloCustomLowering>();
}

} // namespace mlir::torch
