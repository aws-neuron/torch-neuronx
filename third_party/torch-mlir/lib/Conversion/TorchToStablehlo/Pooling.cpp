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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch-mlir/Conversion/TorchToStablehlo/StablehloLegalizeUtils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include <numeric>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::torch_to_stablehlo;

static Value createInitialValueForAtenPoolingOp(Operation *op, Type elementTy,
                                                PatternRewriter &rewriter) {
  auto constType = RankedTensorType::get({}, elementTy);
  // Avg pooling
  if (isa<AtenAvgPool1dOp, AtenAdaptiveAvgPool2dOp, AtenAvgPool2dOp,
          AtenAvgPool3dOp, AtenCumsumOp>(op)) {
    if (isa<mlir::FloatType>(elementTy)) {
      auto constAttr = DenseElementsAttr::get(
          constType, {APFloat::getZero(
                         cast<mlir::FloatType>(elementTy).getFloatSemantics(),
                         /*negative=*/false)});
      return stablehlo::ConstantOp::create(rewriter, op->getLoc(), constType,
                                           constAttr);
    } else if (isa<mlir::IntegerType>(elementTy) &&
               elementTy.getIntOrFloatBitWidth() != 8) {
      auto constAttr = DenseElementsAttr::get(
          constType, {APInt::getZero(elementTy.getIntOrFloatBitWidth())});
      return stablehlo::ConstantOp::create(rewriter, op->getLoc(), constType,
                                           constAttr);
    }
  }

  // Max pooling
  if (isa<AtenMaxPool1dOp, AtenMaxPool2dOp, AtenMaxPool3dOp,
          AtenMaxPool1dWithIndicesOp, AtenMaxPool2dWithIndicesOp>(op)) {
    if (isa<mlir::FloatType>(elementTy)) {
      auto constAttr = DenseElementsAttr::get(
          constType,
          {APFloat::getInf(cast<mlir::FloatType>(elementTy).getFloatSemantics(),
                           /*negative=*/true)});
      return stablehlo::ConstantOp::create(rewriter, op->getLoc(), constType,
                                           constAttr);
    } else if (isa<mlir::IntegerType>(elementTy) &&
               elementTy.getIntOrFloatBitWidth() != 8) {
      auto constAttr = DenseElementsAttr::get(
          constType,
          {APInt::getSignedMinValue(elementTy.getIntOrFloatBitWidth())});
      return stablehlo::ConstantOp::create(rewriter, op->getLoc(), constType,
                                           constAttr);
    }
  }
  op->emitError("unimplemented lowering in AtenPoolingOp");
  return nullptr;
}

// Compute per-axis [low, high] padding for a pool op
// Ceil-mode adds trailing padding to match PyTorch's output size; see
// https://github.com/pytorch/pytorch/blob/c5de6ff079e3e5b453d6ff5190c90f02db458928/aten/src/ATen/native/Pool.h#L79
static SmallVector<std::pair<int64_t, int64_t>>
computePoolPadding(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> kernelSize,
                   ArrayRef<int64_t> stride, ArrayRef<int64_t> padding,
                   bool ceilMode, int64_t dim) {
  const int64_t rank = inputShape.size();
  const int64_t spatialStart = rank - dim;
  SmallVector<std::pair<int64_t, int64_t>> result(rank, {0, 0});

  for (int64_t i = 0; i < dim; ++i) {
    const int64_t axis = spatialStart + i;
    int64_t low = padding[i];
    int64_t high = padding[i];

    if (ceilMode) {
      const int64_t inputSize = inputShape[axis];
      if (inputSize != ShapedType::kDynamic) {
        const int64_t numerator =
            inputSize + 2 * padding[i] - (kernelSize[i] - 1) - 1;
        const int64_t floorOut = numerator / stride[i] + 1;
        const int64_t adj = stride[i] - 1;
        int64_t ceilOut = (numerator + adj) / stride[i] + 1;
        if ((ceilOut - 1) * stride[i] >= inputSize + padding[i])
          ceilOut--;
        if (ceilOut > floorOut) {
          const int64_t extra = (ceilOut - floorOut) * stride[i];
          high += extra;
        }
      }
    }

    result[axis] = {low, high};
  }
  return result;
}

static void splitPoolPaddingForStablehloPad(
    ArrayRef<std::pair<int64_t, int64_t>> poolPadding,
    SmallVectorImpl<int64_t> &padLow, SmallVectorImpl<int64_t> &padHigh,
    SmallVectorImpl<int64_t> &padInterior) {
  const int64_t rank = poolPadding.size();
  padLow.clear();
  padHigh.clear();
  padInterior.clear();
  padLow.reserve(rank);
  padHigh.reserve(rank);
  padInterior.assign(rank, 0);
  for (auto &p : poolPadding) {
    padLow.push_back(p.first);
    padHigh.push_back(p.second);
  }
}

// Pad `input` with `padValue` per the given [low, high] pairs; no-op if all
// padding is zero.
static Value applyExplicitPad(
    Location loc, Value input, Value padValue,
    ArrayRef<std::pair<int64_t, int64_t>> poolPadding,
    ConversionPatternRewriter &rewriter) {
  SmallVector<int64_t> padLow, padHigh, padInterior;
  splitPoolPaddingForStablehloPad(poolPadding, padLow, padHigh, padInterior);
  bool anyNonZero = false;
  for (auto v : padLow)
    anyNonZero |= (v != 0);
  for (auto v : padHigh)
    anyNonZero |= (v != 0);
  if (!anyNonZero)
    return input;
  return stablehlo::PadOp::create(
      rewriter, loc, input, padValue,
      rewriter.getDenseI64ArrayAttr(padLow),
      rewriter.getDenseI64ArrayAttr(padHigh),
      rewriter.getDenseI64ArrayAttr(padInterior));
}

// AtenMaxPool1dWithIndicesOp
template <>
LogicalResult ConvertAtenOp<AtenMaxPool1dWithIndicesOp>::matchAndRewrite(
    AtenMaxPool1dWithIndicesOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.getSelf();
  auto inputTy = cast<RankedTensorType>(input.getType());
  auto inputElemTy = inputTy.getElementType();
  auto inputShape = inputTy.getShape();
  auto inputRank = inputTy.getRank();

  auto outValTy =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType(0)));
  auto outIdxTy =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType(1)));

  if (inputRank <= 1) {
    return op.emitError(
        "max_pooling1d only supports inputs with rank higher than 1");
  }

  SmallVector<int64_t, 1> padding, kernelSize, stride, dilation;
  bool ceilMode = false;

  if (!(matchPattern(op.getKernelSize(),
                     m_TorchListOfConstantInts(kernelSize)))) {
    return rewriter.notifyMatchFailure(
        op, "non-const int kernel size unsupported!");
  }
  if (!(matchPattern(op.getStride(), m_TorchListOfConstantInts(stride)))) {
    return rewriter.notifyMatchFailure(op, "non-const int stride unsupported!");
  }
  if (!(matchPattern(op.getPadding(), m_TorchListOfConstantInts(padding)))) {
    return rewriter.notifyMatchFailure(op,
                                       "non-const int padding unsupported!");
  }
  if (!(matchPattern(op.getDilation(), m_TorchListOfConstantInts(dilation)))) {
    return rewriter.notifyMatchFailure(op,
                                       "non-const int dilation unsupported!");
  }
  if (!(matchPattern(op.getCeilMode(), m_TorchConstantBool(&ceilMode)))) {
    return rewriter.notifyMatchFailure(op,
                                       "non-const bool ceil_mode unsupported!");
  }

  SmallVector<int64_t> stablehloStride(inputRank, 1);
  SmallVector<int64_t> stablehloDilation(inputRank, 1);
  SmallVector<int64_t> stablehloKernelSize(inputRank, 1);
  SmallVector<int64_t> stablehloPadding(inputRank * 2, 0);

  std::copy(stride.begin(), stride.end(),
            stablehloStride.begin() + inputRank - 1);
  std::copy(dilation.begin(), dilation.end(),
            stablehloDilation.begin() + inputRank - 1);
  std::copy(kernelSize.begin(), kernelSize.end(),
            stablehloKernelSize.begin() + inputRank - 1);
  stablehloPadding[stablehloPadding.size() - 1] = padding[0];
  stablehloPadding[stablehloPadding.size() - 2] = padding[0];

  if (ceilMode) {
    // Match PyTorch output shape with extra padding. See
    // https://github.com/pytorch/pytorch/blob/c5de6ff079e3e5b453d6ff5190c90f02db458928/aten/src/ATen/native/Pool.h#L79
    const int64_t inputSize = inputShape[inputRank - 1];
    const int64_t numerator =
        (inputSize + 2 * padding[0] - dilation[0] * (kernelSize[0] - 1) - 1);
    const int64_t floor_output_size = (numerator) / stride[0] + 1;
    const int64_t adj = (stride[0] - 1);
    int64_t ceil_output_size = std::ceil((numerator + adj) / stride[0]) + 1;

    // Ensure last pooling starts inside input
    if ((ceil_output_size - 1) * stride[0] >= inputSize + padding[0]) {
      ceil_output_size--;
    }

    // Add extra padding to make output size same as torch
    if (ceil_output_size > floor_output_size) {
      const int64_t sizeDiff = ceil_output_size - floor_output_size;
      const int64_t extraPadding = sizeDiff * stride[0];
      stablehloPadding[stablehloPadding.size() - 1] += extraPadding;
    }
  }

  Value initVal = createInitialValueForAtenPoolingOp(op, inputElemTy, rewriter);

  auto windowDimensions = rewriter.getDenseI64ArrayAttr(stablehloKernelSize);
  auto windowStrides = rewriter.getDenseI64ArrayAttr(stablehloStride);
  auto windowDilations = rewriter.getDenseI64ArrayAttr(stablehloDilation);
  DenseIntElementsAttr pad = DenseIntElementsAttr::get(
      RankedTensorType::get(
          {static_cast<int64_t>(inputRank), static_cast<int64_t>(2)},
          rewriter.getI64Type()),
      stablehloPadding);
  DenseI64ArrayAttr baseDilations;

  auto inputShapeInfo = hlo::getDimIndexOfTensor(rewriter, op, input);
  if (failed(inputShapeInfo)) {
    return rewriter.notifyMatchFailure(
        op, "failed to get dimension sizes of the input");
  }
  auto inputShapeVec = *inputShapeInfo;
  auto inputShapeTensor = mlir::tensor::FromElementsOp::create(
      rewriter, op->getLoc(), inputShapeVec);

  // no need to reshape here for max_pool_1d. Need to make sure the iota
  // dimension. dim=inputRank-2 or dim=inputRank-1?
  auto indexTensor =
      stablehlo::DynamicIotaOp::create(
          rewriter, op->getLoc(),
          RankedTensorType::get(inputShape, rewriter.getI64Type()),
          inputShapeTensor, static_cast<uint64_t>(inputRank - 1))
          .getResult();
  Value initIdx = hlo::getConstTensor<int64_t>(rewriter, op, {0}, {}).value();

  auto reduceWindowOp = stablehlo::ReduceWindowOp::create(
      rewriter, op->getLoc(), mlir::TypeRange{outValTy, outIdxTy},
      mlir::ValueRange{input, indexTensor}, mlir::ValueRange{initVal, initIdx},
      windowDimensions, windowStrides, baseDilations, windowDilations, pad);

  // add block.
  Block &block = reduceWindowOp.getBody().emplaceBlock();
  auto blockValArgumentType = RankedTensorType::get({}, inputElemTy);
  auto blockIdxArgumentType = RankedTensorType::get({}, rewriter.getI64Type());
  auto compareResultType = RankedTensorType::get({}, rewriter.getI1Type());
  block.addArgument(blockValArgumentType, op->getLoc());
  block.addArgument(blockIdxArgumentType, op->getLoc());
  block.addArgument(blockValArgumentType, op->getLoc());
  block.addArgument(blockIdxArgumentType, op->getLoc());
  auto *firstValArg = block.args_begin();
  auto *firstIdxArg = std::next(firstValArg);
  auto *secondValArg = std::next(firstIdxArg);
  auto *secondIdxArg = std::next(secondValArg);

  stablehlo::ComparisonTypeAttr compareTypeAttr;
  if (isa<mlir::FloatType>(inputTy.getElementType())) {
    compareTypeAttr = stablehlo::ComparisonTypeAttr::get(
        rewriter.getContext(), stablehlo::ComparisonType::FLOAT);
  } else if (isa<mlir::IntegerType>(inputTy.getElementType())) {
    compareTypeAttr = stablehlo::ComparisonTypeAttr::get(
        rewriter.getContext(), stablehlo::ComparisonType::SIGNED);
  }

  stablehlo::ComparisonDirectionAttr compareGeDirectionAttr =
      stablehlo::ComparisonDirectionAttr::get(
          rewriter.getContext(), stablehlo::ComparisonDirection::GE);
  stablehlo::ComparisonDirectionAttr compareEqDirectionAttr =
      stablehlo::ComparisonDirectionAttr::get(
          rewriter.getContext(), stablehlo::ComparisonDirection::EQ);

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);

    Value compareGeResult = stablehlo::CompareOp::create(
        rewriter, op->getLoc(), compareResultType, *firstValArg, *secondValArg,
        compareGeDirectionAttr, compareTypeAttr);
    Value retValResult = stablehlo::SelectOp::create(
        rewriter, op->getLoc(), compareGeResult, *firstValArg, *secondValArg);

    // Get smaller index if compared values are equal.
    Value compareEqResult = stablehlo::CompareOp::create(
        rewriter, op->getLoc(), compareResultType, *firstValArg, *secondValArg,
        compareEqDirectionAttr, compareTypeAttr);
    Value minIdx = stablehlo::MinOp::create(rewriter, op->getLoc(),
                                            *firstIdxArg, *secondIdxArg);
    Value idxWithGeVal = stablehlo::SelectOp::create(
        rewriter, op->getLoc(), compareGeResult, *firstIdxArg, *secondIdxArg);
    Value retIdxResult = stablehlo::SelectOp::create(
        rewriter, op->getLoc(), compareEqResult, minIdx, idxWithGeVal);

    stablehlo::ReturnOp::create(rewriter, op->getLoc(),
                                mlir::ValueRange{retValResult, retIdxResult});
  }

  rewriter.replaceOp(op, reduceWindowOp.getResults());
  return success();
}

// AtenMaxPool2dWithIndicesOp
//
// Indices are row-major flat offsets into the H*W plane per (n, c) slice.
template <>
LogicalResult ConvertAtenOp<AtenMaxPool2dWithIndicesOp>::matchAndRewrite(
    AtenMaxPool2dWithIndicesOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value input = adaptor.getSelf();
  auto inputTy = cast<RankedTensorType>(input.getType());
  auto inputElemTy = inputTy.getElementType();
  auto inputShape = inputTy.getShape();
  auto inputRank = inputTy.getRank();
  auto outValTy =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType(0)));
  auto outIdxTy =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType(1)));

  if (inputRank <= 2) {
    return op.emitError(
        "max_pooling2d only supports inputs with rank higher than 2");
  }

  SmallVector<int64_t, 2> padding, kernelSize, stride, dilation;
  bool ceilMode = false;
  if (!matchPattern(op.getKernelSize(), m_TorchListOfConstantInts(kernelSize)))
    return rewriter.notifyMatchFailure(op, "non-const kernel_size");
  if (!matchPattern(op.getStride(), m_TorchListOfConstantInts(stride))) {
    if (isa<Torch::NoneType>(op.getStride().getType()) || stride.empty()) {
      stride = kernelSize;
    } else {
      return rewriter.notifyMatchFailure(op, "non-const stride");
    }
  }
  if (stride.empty())
    stride = kernelSize;
  if (!matchPattern(op.getPadding(), m_TorchListOfConstantInts(padding)))
    return rewriter.notifyMatchFailure(op, "non-const padding");
  if (!matchPattern(op.getDilation(), m_TorchListOfConstantInts(dilation)))
    return rewriter.notifyMatchFailure(op, "non-const dilation");
  if (!matchPattern(op.getCeilMode(), m_TorchConstantBool(&ceilMode)))
    return rewriter.notifyMatchFailure(op, "non-const ceil_mode");

  SmallVector<int64_t> winDims(inputRank, 1);
  SmallVector<int64_t> winStrides(inputRank, 1);
  std::copy(kernelSize.begin(), kernelSize.end(),
            winDims.begin() + inputRank - 2);
  std::copy(stride.begin(), stride.end(),
            winStrides.begin() + inputRank - 2);

  // Build an iota index tensor matching input shape, with values = h*W + w.
  auto inputShapeInfo = hlo::getDimIndexOfTensor(rewriter, op, input);
  if (failed(inputShapeInfo))
    return rewriter.notifyMatchFailure(
        op, "failed to get dimension sizes of the input");
  auto inputShapeVec = *inputShapeInfo;
  auto inputShapeTensor =
      tensor::FromElementsOp::create(rewriter, loc, inputShapeVec);

  SmallVector<Value> initIndexShapeVec;
  for (int64_t i = 0; i < inputRank - 2; i++)
    initIndexShapeVec.push_back(inputShapeVec[i]);
  initIndexShapeVec.push_back(arith::MulIOp::create(
      rewriter, loc, inputShapeVec[inputRank - 1],
      inputShapeVec[inputRank - 2]));
  auto initIndexShapeTensor =
      tensor::FromElementsOp::create(rewriter, loc, initIndexShapeVec);

  SmallVector<int64_t> initIndexShapeForType(inputShape.begin(),
                                             inputShape.end() - 2);
  if (inputShape[inputRank - 1] == ShapedType::kDynamic ||
      inputShape[inputRank - 2] == ShapedType::kDynamic) {
    initIndexShapeForType.push_back(ShapedType::kDynamic);
  } else {
    initIndexShapeForType.push_back(inputShape[inputRank - 1] *
                                    inputShape[inputRank - 2]);
  }

  Value initIndexTensor = stablehlo::DynamicIotaOp::create(
      rewriter, loc,
      RankedTensorType::get(initIndexShapeForType, rewriter.getI64Type()),
      initIndexShapeTensor, static_cast<uint64_t>(inputRank - 2));

  Value indexTensor = stablehlo::DynamicReshapeOp::create(
      rewriter, loc,
      RankedTensorType::get(inputShape, rewriter.getI64Type()),
      initIndexTensor, inputShapeTensor);

  auto poolPadding = computePoolPadding(inputShape, kernelSize, stride,
                                        padding, ceilMode, /*dim=*/2);

  Value padValueVal =
      createInitialValueForAtenPoolingOp(op, inputElemTy, rewriter);
  Value paddedInput =
      applyExplicitPad(loc, input, padValueVal, poolPadding, rewriter);

  // Padded index positions use 0 as a sentinel; the matching -inf value
  // never wins the GE compare, so these indices are never selected.
  Value padIndexVal =
      hlo::getConstTensor<int64_t>(rewriter, op, {0}, {}).value();
  Value paddedIndex =
      applyExplicitPad(loc, indexTensor, padIndexVal, poolPadding, rewriter);

  SmallVector<int64_t> noPadding(inputRank * 2, 0);
  DenseIntElementsAttr noPadAttr = DenseIntElementsAttr::get(
      RankedTensorType::get(
          {static_cast<int64_t>(inputRank), static_cast<int64_t>(2)},
          rewriter.getI64Type()),
      noPadding);
  DenseI64ArrayAttr baseDilations;
  DenseI64ArrayAttr windowDilations;

  auto reduceWindowOp = stablehlo::ReduceWindowOp::create(
      rewriter, loc, TypeRange{outValTy, outIdxTy},
      ValueRange{paddedInput, paddedIndex},
      ValueRange{padValueVal, padIndexVal},
      rewriter.getDenseI64ArrayAttr(winDims),
      rewriter.getDenseI64ArrayAttr(winStrides), baseDilations, windowDilations,
      noPadAttr);

  Block &block = reduceWindowOp.getBody().emplaceBlock();
  auto valArgTy = RankedTensorType::get({}, inputElemTy);
  auto idxArgTy = RankedTensorType::get({}, rewriter.getI64Type());
  auto boolTy = RankedTensorType::get({}, rewriter.getI1Type());
  block.addArgument(valArgTy, loc);
  block.addArgument(idxArgTy, loc);
  block.addArgument(valArgTy, loc);
  block.addArgument(idxArgTy, loc);
  auto *firstValArg = block.args_begin();
  auto *firstIdxArg = std::next(firstValArg);
  auto *secondValArg = std::next(firstIdxArg);
  auto *secondIdxArg = std::next(secondValArg);

  auto cmpType = isa<mlir::FloatType>(inputElemTy)
                     ? stablehlo::ComparisonType::FLOAT
                     : stablehlo::ComparisonType::SIGNED;
  auto cmpTypeAttr =
      stablehlo::ComparisonTypeAttr::get(rewriter.getContext(), cmpType);
  auto geDir = stablehlo::ComparisonDirectionAttr::get(
      rewriter.getContext(), stablehlo::ComparisonDirection::GE);
  auto eqDir = stablehlo::ComparisonDirectionAttr::get(
      rewriter.getContext(), stablehlo::ComparisonDirection::EQ);

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);
    Value ge = stablehlo::CompareOp::create(
        rewriter, loc, boolTy, *firstValArg, *secondValArg, geDir, cmpTypeAttr);
    Value retVal = stablehlo::SelectOp::create(rewriter, loc, ge, *firstValArg,
                                               *secondValArg);
    // Get smaller index if compared values are equal.
    Value eq = stablehlo::CompareOp::create(
        rewriter, loc, boolTy, *firstValArg, *secondValArg, eqDir, cmpTypeAttr);
    Value minIdx = stablehlo::MinOp::create(rewriter, loc, *firstIdxArg,
                                            *secondIdxArg);
    Value idxFromGe = stablehlo::SelectOp::create(rewriter, loc, ge,
                                                  *firstIdxArg, *secondIdxArg);
    Value retIdx =
        stablehlo::SelectOp::create(rewriter, loc, eq, minIdx, idxFromGe);
    stablehlo::ReturnOp::create(rewriter, loc, ValueRange{retVal, retIdx});
  }

  rewriter.replaceOp(op, reduceWindowOp.getResults());
  return success();
}

namespace {
template <typename AtenOpT, int Dim>
class ConvertAtenMaxPoolOp : public ConvertAtenOp<AtenOpT> {
public:
  using ConvertAtenOp<AtenOpT>::ConvertAtenOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    auto inputTy = cast<RankedTensorType>(input.getType());
    auto inputElemTy = inputTy.getElementType();
    auto inputRank = inputTy.getRank();
    auto outTy = cast<RankedTensorType>(
        ConvertAtenOp<AtenOpT>::getTypeConverter()->convertType(op.getType()));

    if (inputRank <= Dim) {
      return op.emitError(
          "max_pooling1d/2d only supports inputs with rank higher than 1/2");
    }
    SmallVector<int64_t, Dim> padding, kernelSize, stride, dilation;
    bool ceilMode = false;

    if (!(matchPattern(op.getKernelSize(),
                       m_TorchListOfConstantInts(kernelSize)))) {
      return rewriter.notifyMatchFailure(
          op, "non-const int kernel size unsupported!");
    }
    if (!(matchPattern(op.getStride(), m_TorchListOfConstantInts(stride)))) {
      return rewriter.notifyMatchFailure(op,
                                         "non-const int stride unsupported!");
    }
    if (!(matchPattern(op.getPadding(), m_TorchListOfConstantInts(padding)))) {
      return rewriter.notifyMatchFailure(op,
                                         "non-const int padding unsupported!");
    }
    if (!(matchPattern(op.getDilation(),
                       m_TorchListOfConstantInts(dilation)))) {
      return rewriter.notifyMatchFailure(op,
                                         "non-const int dilation unsupported!");
    }
    if (!(matchPattern(op.getCeilMode(), m_TorchConstantBool(&ceilMode)))) {
      return rewriter.notifyMatchFailure(
          op, "non-const bool ceil_mode unsupported!");
    }

    if (stride.empty()) {
      stride = kernelSize;
    }

    // prepend 1 to kernelSize, stride, dilation until they are of same rank
    // as input
    SmallVector<int64_t> stablehloStride(inputRank, 1);
    SmallVector<int64_t> stablehloDilation(inputRank, 1);
    SmallVector<int64_t> stablehloKernelSize(inputRank, 1);
    SmallVector<int64_t> stablehloPadding(inputRank * 2, 0);
    std::copy(dilation.begin(), dilation.end(),
              stablehloDilation.begin() + inputRank - Dim);
    std::copy(stride.begin(), stride.end(),
              stablehloStride.begin() + inputRank - Dim);
    std::copy(kernelSize.begin(), kernelSize.end(),
              stablehloKernelSize.begin() + inputRank - Dim);

    Value initVal =
        createInitialValueForAtenPoolingOp(op, inputElemTy, rewriter);

    if (Dim < 1 || Dim > 3) {
      assert(false && "Unsupported pooling dimension");
    }

    const size_t spatialIdxStart = inputRank - Dim;

    for (int i = 0; i < Dim; i++) {
      const size_t frontPadIdx = (spatialIdxStart + i) * 2;
      const size_t backPadIdx = (spatialIdxStart + i) * 2 + 1;

      // torch padding is symmetric
      stablehloPadding[frontPadIdx] = padding[i];
      stablehloPadding[backPadIdx] = padding[i];

      if (ceilMode) {
        // Match PyTorch output shape with extra padding. See
        // https://github.com/pytorch/pytorch/blob/c5de6ff079e3e5b453d6ff5190c90f02db458928/aten/src/ATen/native/Pool.h#L79
        // PyTorch output size formula:
        // 1. Calculate base output size:
        // output = (input + 2*pad - dilation*(kernel-1) - 1+adj) / stride + 1
        // where adj = (stride-1) if ceil_mode else 0
        // 2. Apply edge case correction:
        // if ((output-1) * stride >= input + pad_l) --output;

        const int64_t inputSize = inputTy.getDimSize(spatialIdxStart + i);
        const int64_t numerator = (inputSize + 2 * padding[i] -
                                   dilation[i] * (kernelSize[i] - 1) - 1);
        const int64_t floor_output_size = (numerator) / stride[i] + 1;
        const int64_t adj = (stride[i] - 1);
        int64_t ceil_output_size = std::ceil((numerator + adj) / stride[i]) + 1;

        // Ensure last pooling starts inside input
        if ((ceil_output_size - 1) * stride[i] >= inputSize + padding[i]) {
          ceil_output_size--;
        }

        // Add extra padding to make output size same as torch
        if (ceil_output_size > floor_output_size) {
          const int64_t sizeDiff = ceil_output_size - floor_output_size;
          const int64_t extraPadding = sizeDiff * stride[i];
          stablehloPadding[frontPadIdx] += extraPadding / 2;
          stablehloPadding[backPadIdx] += extraPadding - extraPadding / 2;
        }
      }
    }

    auto windowDimensions = rewriter.getDenseI64ArrayAttr(stablehloKernelSize);
    auto windowStrides = rewriter.getDenseI64ArrayAttr(stablehloStride);
    DenseI64ArrayAttr baseDilations;
    auto windowDilations = rewriter.getDenseI64ArrayAttr(stablehloDilation);

    DenseIntElementsAttr pad = DenseIntElementsAttr::get(
        RankedTensorType::get(
            {static_cast<int64_t>(inputRank), static_cast<int64_t>(2)},
            rewriter.getI64Type()),
        stablehloPadding);

    auto reduceWindowOp = stablehlo::ReduceWindowOp::create(
        rewriter, op->getLoc(), outTy, input, initVal, windowDimensions,
        windowStrides, baseDilations, windowDilations, pad);

    Block &block = reduceWindowOp.getBody().emplaceBlock();

    // Add bb argument
    auto blockArgumentType = RankedTensorType::get({}, inputElemTy);
    block.addArgument(blockArgumentType, op->getLoc());
    block.addArgument(blockArgumentType, op->getLoc());
    auto *firstArg = block.args_begin();
    auto secondArg = block.args_rbegin();

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&block);

      Value result = stablehlo::MaxOp::create(rewriter, op->getLoc(), *firstArg,
                                              *secondArg);
      stablehlo::ReturnOp::create(rewriter, op->getLoc(), result);
    }

    rewriter.replaceOp(op, reduceWindowOp.getResults());
    return success();
  }
};
} // namespace

namespace {
template <typename AtenOpT, int Dim>
class ConvertAtenAvgPoolOp : public ConvertAtenOp<AtenOpT> {
public:
  using ConvertAtenOp<AtenOpT>::ConvertAtenOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    RankedTensorType inputTy = cast<RankedTensorType>(input.getType());
    Type inputElemTy = inputTy.getElementType();
    int64_t inputRank = inputTy.getRank();
    RankedTensorType outTy = cast<RankedTensorType>(
        ConvertAtenOp<AtenOpT>::getTypeConverter()->convertType(op.getType()));
    auto outShape = outTy.getShape();

    if (inputRank <= Dim) {
      return op.emitError("avg_pooling1d/2d/3d only supports inputs with rank "
                          "higher than 1/2/3");
    }
    SmallVector<int64_t, Dim> padding, kernelSize, stride;
    bool ceilMode = false;
    bool countIncludePad = true;

    if (!(matchPattern(op.getKernelSize(),
                       m_TorchListOfConstantInts(kernelSize)))) {
      return rewriter.notifyMatchFailure(
          op, "non-const int kernel size unsupported!");
    }
    if (!(matchPattern(op.getStride(), m_TorchListOfConstantInts(stride)))) {
      return rewriter.notifyMatchFailure(op,
                                         "non-const int stride unsupported!");
    }
    if (!(matchPattern(op.getPadding(), m_TorchListOfConstantInts(padding)))) {
      return rewriter.notifyMatchFailure(op,
                                         "non-const int padding unsupported!");
    }
    if (!(matchPattern(op.getCeilMode(), m_TorchConstantBool(&ceilMode)))) {
      return rewriter.notifyMatchFailure(
          op, "non-const bool ceil_mode unsupported!");
    }
    if (!(matchPattern(op.getCountIncludePad(),
                       m_TorchConstantBool(&countIncludePad)))) {
      return rewriter.notifyMatchFailure(
          op, "non-const bool count_include_pad unsupported!");
    }

    if (stride.empty()) {
      stride = kernelSize;
    }

    if constexpr (std::is_same<AtenOpT, AtenAvgPool2dOp>()) {
      if (succeeded(checkNotNone(rewriter, op, op.getDivisorOverride())))
        return rewriter.notifyMatchFailure(
            op, "only None divisor_override supported for now!");
    }

    // Prepend 1 to kernelSize, stride, dilation until they are of same rank
    // as input
    SmallVector<int64_t> stablehloStride(inputRank, 1);
    SmallVector<int64_t> stablehloDilation(inputRank, 1);
    SmallVector<int64_t> stablehloKernelSize(inputRank, 1);
    SmallVector<int64_t> stablehloPadding(inputRank * 2, 0);

    std::copy(stride.begin(), stride.end(),
              stablehloStride.begin() + inputRank - Dim);
    std::copy(kernelSize.begin(), kernelSize.end(),
              stablehloKernelSize.begin() + inputRank - Dim);
    if (Dim == 1) {
      stablehloPadding[stablehloPadding.size() - 2] = padding[0];
      stablehloPadding[stablehloPadding.size() - 1] = padding[0];
    } else if (Dim == 2) {
      stablehloPadding[stablehloPadding.size() - 4] = padding[0];
      stablehloPadding[stablehloPadding.size() - 3] = padding[0];
      stablehloPadding[stablehloPadding.size() - 2] = padding[1];
      stablehloPadding[stablehloPadding.size() - 1] = padding[1];
    } else if (Dim == 3) {
      stablehloPadding[stablehloPadding.size() - 6] = padding[0];
      stablehloPadding[stablehloPadding.size() - 5] = padding[0];
      stablehloPadding[stablehloPadding.size() - 4] = padding[1];
      stablehloPadding[stablehloPadding.size() - 3] = padding[1];
      stablehloPadding[stablehloPadding.size() - 2] = padding[2];
      stablehloPadding[stablehloPadding.size() - 1] = padding[2];
    } else {
      assert(false && "Unsupported pooling dimension");
    }

    Value initVal =
        createInitialValueForAtenPoolingOp(op, inputElemTy, rewriter);

    auto windowDimensions = rewriter.getDenseI64ArrayAttr(stablehloKernelSize);
    auto windowStrides = rewriter.getDenseI64ArrayAttr(stablehloStride);
    DenseI64ArrayAttr baseDilations;
    auto windowDilations = rewriter.getDenseI64ArrayAttr(stablehloDilation);
    DenseIntElementsAttr pad = DenseIntElementsAttr::get(
        RankedTensorType::get(
            {static_cast<int64_t>(inputRank), static_cast<int64_t>(2)},
            rewriter.getI64Type()),
        stablehloPadding);

    auto reduceWindowSum = stablehlo::ReduceWindowOp::create(
        rewriter, op->getLoc(), outTy, input, initVal, windowDimensions,
        windowStrides, baseDilations, windowDilations, pad);

    Block &sumBlock = reduceWindowSum.getBody().emplaceBlock();

    // Add bb argument
    auto blockArgumentType = RankedTensorType::get({}, inputElemTy);
    sumBlock.addArgument(blockArgumentType, op->getLoc());
    sumBlock.addArgument(blockArgumentType, op->getLoc());
    auto firstArg = *sumBlock.args_begin();
    auto secondArg = *sumBlock.args_rbegin();

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&sumBlock);

      Value sumResult =
          stablehlo::AddOp::create(rewriter, op->getLoc(), firstArg, secondArg);
      stablehlo::ReturnOp::create(rewriter, op->getLoc(), sumResult);
    }

    // Use kernel size as the divisor
    if (countIncludePad) {
      Value divisor;
      if (Dim == 1) {
        divisor =
            hlo::getConstTensor<int64_t>(rewriter, op, {kernelSize[0]}, {})
                .value();
      } else if (Dim == 2) {
        divisor = hlo::getConstTensor<int64_t>(
                      rewriter, op, {kernelSize[0] * kernelSize[1]}, {})
                      .value();
      } else if (Dim == 3) {
        divisor = hlo::getConstTensor<int64_t>(
                      rewriter, op,
                      {kernelSize[0] * kernelSize[1] * kernelSize[2]}, {})
                      .value();
      } else {
        assert(false && "Unsupported pooling dimension");
      }
      divisor = hlo::promoteType(rewriter, op.getLoc(), divisor,
                                 outTy.getElementType());
      DenseI64ArrayAttr bcastDimensions;
      rewriter.replaceOpWithNewOp<mlir::chlo::BroadcastDivOp>(
          op, outTy, reduceWindowSum.getResult(0), divisor, bcastDimensions);
      return success();
    }

    // Use another mhlo.ReduceWindowOp to get the divisor
    Value windowSizeConst =
        hlo::getConstTensor<float>(rewriter, op, {1.0}, {}).value();
    windowSizeConst = hlo::promoteType(rewriter, op.getLoc(), windowSizeConst,
                                       outTy.getElementType());
    auto inputShapeVec = *hlo::getDimIndexOfTensor(rewriter, op, input);
    auto inputShapeTensor = mlir::tensor::FromElementsOp::create(
        rewriter, op->getLoc(), inputShapeVec);

    windowSizeConst = stablehlo::DynamicBroadcastInDimOp::create(
        rewriter, op->getLoc(),
        RankedTensorType::get(inputTy.getShape(), outTy.getElementType()),
        windowSizeConst, inputShapeTensor, rewriter.getDenseI64ArrayAttr({}));

    Value zero = createInitialValueForAtenPoolingOp(op, inputElemTy, rewriter);
    auto reduceWindowSize = stablehlo::ReduceWindowOp::create(
        rewriter, op->getLoc(), RankedTensorType::get(outShape, inputElemTy),
        windowSizeConst, zero, windowDimensions, windowStrides, baseDilations,
        windowDilations, pad);

    Block &sizeBlock = reduceWindowSize.getBody().emplaceBlock();

    // Add bb argument
    blockArgumentType = RankedTensorType::get({}, inputElemTy);
    sizeBlock.addArgument(blockArgumentType, op->getLoc());
    sizeBlock.addArgument(blockArgumentType, op->getLoc());
    firstArg = *sizeBlock.args_begin();
    secondArg = *sizeBlock.args_rbegin();

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&sizeBlock);

      Value sumResult =
          stablehlo::AddOp::create(rewriter, op->getLoc(), firstArg, secondArg);
      stablehlo::ReturnOp::create(rewriter, op->getLoc(), sumResult);
    }

    rewriter.replaceOpWithNewOp<stablehlo::DivOp>(
        op, outTy, reduceWindowSum.getResult(0), reduceWindowSize.getResult(0));
    return success();
  }
};
} // namespace

template <>
LogicalResult ConvertAtenOp<Aten_AdaptiveAvgPool2dBackwardOp>::matchAndRewrite(
    Aten_AdaptiveAvgPool2dBackwardOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value gradOutput = adaptor.getGradOutput();
  Value self = adaptor.getSelf();

  auto gradOutputTy = cast<RankedTensorType>(gradOutput.getType());
  auto selfTy = cast<RankedTensorType>(self.getType());
  auto gradOutputShape = gradOutputTy.getShape();
  auto selfShape = selfTy.getShape();
  auto gradOutputRank = gradOutputTy.getRank();

  auto outTy =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
  auto elemTy = outTy.getElementType();

  if (gradOutputRank != 4 || selfTy.getRank() != 4) {
    return rewriter.notifyMatchFailure(
        op, "adaptive_avg_pool2d_backward only supports 4D tensors");
  }

  // Check for static shapes
  for (int64_t i = 0; i < 4; i++) {
    if (gradOutputShape[i] == ShapedType::kDynamic ||
        selfShape[i] == ShapedType::kDynamic) {
      return rewriter.notifyMatchFailure(
          op,
          "adaptive_avg_pool2d_backward only supports static shapes for now");
    }
  }

  int64_t outputHeight = gradOutputShape[2];
  int64_t outputWidth = gradOutputShape[3];
  int64_t inputHeight = selfShape[2];
  int64_t inputWidth = selfShape[3];

  // Only support evenly sized windows for now, as it is very complex otherwise
  if (inputHeight % outputHeight != 0 || inputWidth % outputWidth != 0) {
    return rewriter.notifyMatchFailure(
        op, "unimplemented: input size must be divisible by output size");
  }

  int64_t strideH = inputHeight / outputHeight;
  int64_t strideW = inputWidth / outputWidth;
  int64_t kernelH = strideH;
  int64_t kernelW = strideW;

  // Normalize gradients by window size
  Value divisor =
      hlo::getConstTensor<int64_t>(rewriter, op, {kernelH * kernelW}, {})
          .value();
  divisor = hlo::promoteType(rewriter, op.getLoc(), divisor, elemTy);

  DenseI64ArrayAttr bcastDimensions;
  Value gradNormalized =
      chlo::BroadcastDivOp::create(rewriter, op.getLoc(), gradOutputTy,
                                   gradOutput, divisor, bcastDimensions);

  // Reshape: (N, C, H_out, W_out) -> (N, C, H_out, 1, W_out, 1)
  SmallVector<int64_t> reshapeShape = {
      gradOutputShape[0], gradOutputShape[1], outputHeight, 1, outputWidth, 1};
  Value reshaped = stablehlo::ReshapeOp::create(
      rewriter, op.getLoc(), RankedTensorType::get(reshapeShape, elemTy),
      gradNormalized);

  // Broadcast: (N, C, H_out, 1, W_out, 1) -> (N, C, H_out, stride_h, W_out,
  // stride_w)
  SmallVector<int64_t> broadcastShape = {gradOutputShape[0], gradOutputShape[1],
                                         outputHeight,       strideH,
                                         outputWidth,        strideW};
  Value broadcasted = stablehlo::BroadcastInDimOp::create(
      rewriter, op.getLoc(), RankedTensorType::get(broadcastShape, elemTy),
      reshaped, rewriter.getDenseI64ArrayAttr({0, 1, 2, 3, 4, 5}));

  // Reshape: (N, C, H_out, stride_h, W_out, stride_w) -> (N, C, H_in, W_in)
  // H_out * stride_h == H_in
  // W_out * stride_w == W_in
  Value gradInput =
      stablehlo::ReshapeOp::create(rewriter, op.getLoc(), outTy, broadcasted);

  rewriter.replaceOp(op, gradInput);
  return success();
}

// AtenMaxPool2dWithIndicesBackwardOp via stablehlo.select_and_scatter.
//
// The `indices` operand is unused — select_and_scatter re-derives winners
// from `self`. Overlapping windows accumulate via add.
template <>
LogicalResult
ConvertAtenOp<AtenMaxPool2dWithIndicesBackwardOp>::matchAndRewrite(
    AtenMaxPool2dWithIndicesBackwardOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  Location loc = op.getLoc();
  Value self = adaptor.getSelf();
  Value gradOutput = adaptor.getGradOutput();

  auto selfTy = cast<RankedTensorType>(self.getType());
  auto gradOutputTy = cast<RankedTensorType>(gradOutput.getType());
  auto outTy =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
  Type elemTy = outTy.getElementType();
  const int64_t rank = selfTy.getRank();

  if (rank != 4 || gradOutputTy.getRank() != 4) {
    return rewriter.notifyMatchFailure(op, "expected 4D tensors");
  }

  SmallVector<int64_t, 2> kernelSize, stride, padding, dilation;
  bool ceilMode = false;
  if (!matchPattern(op.getKernelSize(), m_TorchListOfConstantInts(kernelSize)))
    return rewriter.notifyMatchFailure(op, "non-const kernel_size unsupported");
  if (!matchPattern(op.getStride(), m_TorchListOfConstantInts(stride)))
    return rewriter.notifyMatchFailure(op, "non-const stride unsupported");
  if (stride.empty())
    stride = kernelSize;
  if (!matchPattern(op.getPadding(), m_TorchListOfConstantInts(padding)))
    return rewriter.notifyMatchFailure(op, "non-const padding unsupported");
  if (!matchPattern(op.getDilation(), m_TorchListOfConstantInts(dilation)))
    return rewriter.notifyMatchFailure(op, "non-const dilation unsupported");
  if (!matchPattern(op.getCeilMode(), m_TorchConstantBool(&ceilMode)))
    return rewriter.notifyMatchFailure(op, "non-const ceil_mode unsupported");

  // select_and_scatter has no dilation knob; reject dilation > 1
  for (int64_t d : dilation)
    if (d != 1)
      return rewriter.notifyMatchFailure(op, "dilation > 1 unsupported");

  SmallVector<int64_t> winDims(rank, 1);
  SmallVector<int64_t> winStrides(rank, 1);
  winDims[rank - 2] = kernelSize[0];
  winDims[rank - 1] = kernelSize[1];
  winStrides[rank - 2] = stride[0];
  winStrides[rank - 1] = stride[1];

  // Share padding with the forward so we reduce over the exact same shape.
  auto poolPadding = computePoolPadding(selfTy.getShape(), kernelSize, stride,
                                        padding, ceilMode, /*dim=*/2);

  SmallVector<int64_t> padLowHigh;
  padLowHigh.reserve(rank * 2);
  for (auto &p : poolPadding) {
    padLowHigh.push_back(p.first);
    padLowHigh.push_back(p.second);
  }
  auto padAttr = DenseIntElementsAttr::get(
      RankedTensorType::get({rank, 2}, rewriter.getI64Type()), padLowHigh);

  Value zero = hlo::getConstTensor<float>(rewriter, op, {0.0}, {}).value();
  zero = hlo::promoteType(rewriter, loc, zero, elemTy);

  auto scalarTy = RankedTensorType::get({}, elemTy);
  auto boolTy = RankedTensorType::get({}, rewriter.getI1Type());

  auto sasOp = stablehlo::SelectAndScatterOp::create(
      rewriter, loc, outTy, self, gradOutput, zero,
      rewriter.getDenseI64ArrayAttr(winDims),
      rewriter.getDenseI64ArrayAttr(winStrides), padAttr);

  // Select region: a >= b picks the max-pool winner.
  {
    Block &sel = sasOp.getSelect().emplaceBlock();
    sel.addArgument(scalarTy, loc);
    sel.addArgument(scalarTy, loc);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&sel);
    auto cmpType = isa<mlir::FloatType>(elemTy)
                       ? stablehlo::ComparisonType::FLOAT
                       : stablehlo::ComparisonType::SIGNED;
    Value cmp = stablehlo::CompareOp::create(
        rewriter, loc, boolTy, sel.getArgument(0), sel.getArgument(1),
        stablehlo::ComparisonDirectionAttr::get(
            rewriter.getContext(), stablehlo::ComparisonDirection::GE),
        stablehlo::ComparisonTypeAttr::get(rewriter.getContext(), cmpType));
    stablehlo::ReturnOp::create(rewriter, loc, cmp);
  }

  // Scatter region: sum gradients from overlapping windows.
  {
    Block &sc = sasOp.getScatter().emplaceBlock();
    sc.addArgument(scalarTy, loc);
    sc.addArgument(scalarTy, loc);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&sc);
    Value sum = stablehlo::AddOp::create(rewriter, loc, sc.getArgument(0),
                                         sc.getArgument(1));
    stablehlo::ReturnOp::create(rewriter, loc, sum);
  }

  rewriter.replaceOp(op, sasOp.getResult());
  return success();
}

// AtenCumsumOp
// Implements cumsum via triangular matrix multiplication:
//   cumsum(x, dim) = x @ triu(ones(N, N))
template <>
LogicalResult ConvertAtenOp<AtenCumsumOp>::matchAndRewrite(
    AtenCumsumOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.getSelf();
  auto inputTy = cast<RankedTensorType>(input.getType());
  auto outTy =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
  input =
      hlo::promoteType(rewriter, op.getLoc(), input, outTy.getElementType());
  inputTy = cast<RankedTensorType>(input.getType());
  auto inputElemTy = inputTy.getElementType();
  auto inputRank = inputTy.getRank();
  auto inputShape = inputTy.getShape();
  auto loc = op->getLoc();

  int64_t dim;
  if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim))) {
    return rewriter.notifyMatchFailure(
        op, "unimplemented: dim must be a constant int");
  }
  dim = toPositiveDim(dim, inputRank);
  if (!isValidDim(dim, inputRank)) {
    return rewriter.notifyMatchFailure(op, "dim is out of range");
  }

  int64_t N = inputShape[dim];
  bool isDynamic = inputTy.isDynamicDim(dim);
  bool needsTranspose = (dim != inputRank - 1);

  // Transpose input so that `dim` is the last axis if needed.
  Value matmulInput = input;
  SmallVector<int64_t> transposePermToLast;
  SmallVector<int64_t> transposedShape;
  if (needsTranspose) {
    for (int64_t i = 0; i < inputRank; ++i) {
      if (i != dim)
        transposePermToLast.push_back(i);
    }
    transposePermToLast.push_back(dim);

    for (int64_t i = 0; i < inputRank; ++i) {
      if (i != dim)
        transposedShape.push_back(inputShape[i]);
    }
    transposedShape.push_back(N);

    auto transposedTy = RankedTensorType::get(transposedShape, inputElemTy);
    matmulInput = stablehlo::TransposeOp::create(
        rewriter, loc, transposedTy, input,
        rewriter.getDenseI64ArrayAttr(transposePermToLast));
  } else {
    transposedShape.assign(inputShape.begin(), inputShape.end());
  }

  // Build upper-triangular ones matrix of shape [N, N].
  // M[k][i] = 1 iff k <= i, so result[i] = sum_{k<=i} input[k] (prefix sum).
  Value trilMatrix;
  if (!isDynamic) {
    auto trilType = RankedTensorType::get({N, N}, inputElemTy);
    if (isa<FloatType>(inputElemTy)) {
      auto floatTy = cast<FloatType>(inputElemTy);
      SmallVector<APFloat> apData;
      apData.reserve(N * N);
      for (int64_t row = 0; row < N; ++row) {
        for (int64_t col = 0; col < N; ++col) {
          bool inMask = (row <= col);
          apData.push_back(APFloat(floatTy.getFloatSemantics(),
                                   inMask ? 1 : 0));
        }
      }
      trilMatrix = stablehlo::ConstantOp::create(
          rewriter, loc, DenseElementsAttr::get(trilType, apData))
          .getResult();
    } else {
      auto intTy = cast<IntegerType>(inputElemTy);
      SmallVector<APInt> apData;
      apData.reserve(N * N);
      for (int64_t row = 0; row < N; ++row) {
        for (int64_t col = 0; col < N; ++col) {
          bool inMask = (row <= col);
          apData.push_back(APInt(intTy.getWidth(), inMask ? 1 : 0));
        }
      }
      trilMatrix = stablehlo::ConstantOp::create(
          rewriter, loc, DenseElementsAttr::get(trilType, apData))
          .getResult();
    }
  } else {
    // Dynamic: generate tril at runtime using iota + compare + convert
    Value dimSize = tensor::DimOp::create(rewriter, loc, matmulInput,
                                          inputRank - 1);
    Value dimSizeI64 = arith::IndexCastOp::create(rewriter, loc,
                                                   rewriter.getI64Type(),
                                                   dimSize);
    Value shapeTensor = tensor::FromElementsOp::create(
        rewriter, loc, ValueRange{dimSizeI64, dimSizeI64});

    auto dynamicNxNI64 =
        RankedTensorType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                              rewriter.getI64Type());
    Value rowIota = stablehlo::DynamicIotaOp::create(
        rewriter, loc, dynamicNxNI64, shapeTensor,
        rewriter.getI64IntegerAttr(0));
    Value colIota = stablehlo::DynamicIotaOp::create(
        rewriter, loc, dynamicNxNI64, shapeTensor,
        rewriter.getI64IntegerAttr(1));

    auto dynamicNxNI1 =
        RankedTensorType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                              rewriter.getI1Type());
    Value mask = stablehlo::CompareOp::create(
        rewriter, loc, dynamicNxNI1, rowIota, colIota,
        stablehlo::ComparisonDirectionAttr::get(
            rewriter.getContext(), stablehlo::ComparisonDirection::LE),
        stablehlo::ComparisonTypeAttr::get(
            rewriter.getContext(), stablehlo::ComparisonType::SIGNED));

    auto dynamicNxNElem =
        RankedTensorType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                              inputElemTy);
    trilMatrix = stablehlo::ConvertOp::create(rewriter, loc, dynamicNxNElem,
                                              mask);
  }

  // dot_general: matmulInput @ trilMatrix
  // Contracting: last dim of matmulInput with dim 0 of trilMatrix
  auto transposedTy = RankedTensorType::get(transposedShape, inputElemTy);
  stablehlo::DotDimensionNumbersAttr dotDimNumbers =
      stablehlo::DotDimensionNumbersAttr::get(
          rewriter.getContext(),
          /*lhsBatchingDimensions=*/{},
          /*rhsBatchingDimensions=*/{},
          /*lhsContractingDimensions=*/{inputRank - 1},
          /*rhsContractingDimensions=*/{0});

  Value dotResult = stablehlo::DotGeneralOp::create(
      rewriter, loc, transposedTy, matmulInput, trilMatrix,
      dotDimNumbers, nullptr, nullptr).getResult();

  // Transpose back if we transposed earlier
  Value result;
  if (needsTranspose) {
    SmallVector<int64_t> transposePermBack(inputRank);
    for (int64_t i = 0; i < inputRank; ++i) {
      transposePermBack[transposePermToLast[i]] = i;
    }
    result = stablehlo::TransposeOp::create(
        rewriter, loc, outTy, dotResult,
        rewriter.getDenseI64ArrayAttr(transposePermBack));
  } else {
    result = dotResult;
  }

  rewriter.replaceOp(op, result);
  return success();
}

// AtenIm2colOp
template <>
LogicalResult ConvertAtenOp<AtenIm2colOp>::matchAndRewrite(
    AtenIm2colOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op->getLoc();
  Value input = adaptor.getSelf();
  auto inputTy = cast<RankedTensorType>(input.getType());
  auto outTy =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));

  SmallVector<int64_t> kernelSize, dilation, padding, stride;
  if (!matchPattern(op.getKernelSize(), m_TorchListOfConstantInts(kernelSize)))
    return rewriter.notifyMatchFailure(op, "kernel_size must be constant");
  if (!matchPattern(op.getDilation(), m_TorchListOfConstantInts(dilation)))
    return rewriter.notifyMatchFailure(op, "dilation must be constant");
  if (!matchPattern(op.getPadding(), m_TorchListOfConstantInts(padding)))
    return rewriter.notifyMatchFailure(op, "padding must be constant");
  if (!matchPattern(op.getStride(), m_TorchListOfConstantInts(stride)))
    return rewriter.notifyMatchFailure(op, "stride must be constant");

  auto inputShape = inputTy.getShape();
  int64_t inputRank = inputTy.getRank();
  bool batched = inputRank == 4;

  if (!batched && inputRank != 3)
    return rewriter.notifyMatchFailure(op, "input must be 3D or 4D");

  if (!batched)
    input = stablehlo::ReshapeOp::create(
        rewriter, loc,
        RankedTensorType::get({1, inputShape[0], inputShape[1], inputShape[2]},
                              inputTy.getElementType()),
        input);

  auto batchedInputTy = cast<RankedTensorType>(input.getType());
  auto batchedShape = batchedInputTy.getShape();
  int64_t N = batchedShape[0], C = batchedShape[1], H = batchedShape[2],
          W = batchedShape[3];

  // Pad input
  Value paddedInput = input;
  if (padding[0] > 0 || padding[1] > 0) {
    SmallVector<int64_t> padLow = {0, 0, padding[0], padding[1]};
    SmallVector<int64_t> padHigh = {0, 0, padding[0], padding[1]};
    SmallVector<int64_t> padInterior(4, 0);
    Value padValue =
        hlo::getConstTensor<float>(rewriter, op, {0.0}, {}).value();
    padValue =
        hlo::promoteType(rewriter, loc, padValue, inputTy.getElementType());
    paddedInput = stablehlo::PadOp::create(
        rewriter, loc, input, padValue, rewriter.getDenseI64ArrayAttr(padLow),
        rewriter.getDenseI64ArrayAttr(padHigh),
        rewriter.getDenseI64ArrayAttr(padInterior));
  }

  auto indexTy = rewriter.getI32Type();

  if (H == ShapedType::kDynamic || W == ShapedType::kDynamic) {
    // Dynamic spatial path
    auto i64Ty = rewriter.getI64Type();
    auto constI64 = [&](int64_t v) {
      return arith::ConstantOp::create(rewriter, loc,
                                        rewriter.getI64IntegerAttr(v));
    };

    Value hDim = arith::IndexCastOp::create(
        rewriter, loc, i64Ty, tensor::DimOp::create(rewriter, loc, input, 2));
    Value wDim = arith::IndexCastOp::create(
        rewriter, loc, i64Ty, tensor::DimOp::create(rewriter, loc, input, 3));

    // outH = (H + 2*pad[0] - dilation[0]*(ks[0]-1) - 1) / stride[0] + 1
    Value outHI64 = arith::AddIOp::create(rewriter, loc, hDim,
                                           constI64(2 * padding[0]));
    outHI64 = arith::SubIOp::create(rewriter, loc, outHI64,
                                     constI64(dilation[0] * (kernelSize[0] - 1) + 1));
    outHI64 = arith::DivSIOp::create(rewriter, loc, outHI64, constI64(stride[0]));
    outHI64 = arith::AddIOp::create(rewriter, loc, outHI64, constI64(1));

    Value outWI64 = arith::AddIOp::create(rewriter, loc, wDim,
                                           constI64(2 * padding[1]));
    outWI64 = arith::SubIOp::create(rewriter, loc, outWI64,
                                     constI64(dilation[1] * (kernelSize[1] - 1) + 1));
    outWI64 = arith::DivSIOp::create(rewriter, loc, outWI64, constI64(stride[1]));
    outWI64 = arith::AddIOp::create(rewriter, loc, outWI64, constI64(1));

    Value oneI64 = constI64(1);
    Value ksHI64 = constI64(kernelSize[0]);
    Value ksWI64 = constI64(kernelSize[1]);

    // Static iotas for kernel dims
    Value iotaKh = stablehlo::IotaOp::create(
        rewriter, loc,
        RankedTensorType::get({kernelSize[0], 1, 1, 1}, indexTy), 0);
    Value iotaKw = stablehlo::IotaOp::create(
        rewriter, loc,
        RankedTensorType::get({1, kernelSize[1], 1, 1}, indexTy), 1);

    // Dynamic iotas for output spatial dims
    Value iotaOhShape = tensor::FromElementsOp::create(
        rewriter, loc, ValueRange{oneI64, oneI64, outHI64, oneI64});
    Value iotaOwShape = tensor::FromElementsOp::create(
        rewriter, loc, ValueRange{oneI64, oneI64, oneI64, outWI64});
    auto dynOhTy = RankedTensorType::get({1, 1, ShapedType::kDynamic, 1}, indexTy);
    auto dynOwTy = RankedTensorType::get({1, 1, 1, ShapedType::kDynamic}, indexTy);
    Value iotaOh = stablehlo::DynamicIotaOp::create(
        rewriter, loc, dynOhTy, iotaOhShape, rewriter.getI64IntegerAttr(2));
    Value iotaOw = stablehlo::DynamicIotaOp::create(
        rewriter, loc, dynOwTy, iotaOwShape, rewriter.getI64IntegerAttr(3));

    // Broadcast all to [kH, kW, outH, outW]
    Value fullShapeTensor = tensor::FromElementsOp::create(
        rewriter, loc, ValueRange{ksHI64, ksWI64, outHI64, outWI64});
    auto dynFullTy = RankedTensorType::get(
        {kernelSize[0], kernelSize[1], ShapedType::kDynamic, ShapedType::kDynamic},
        indexTy);
    iotaKh = stablehlo::DynamicBroadcastInDimOp::create(
        rewriter, loc, dynFullTy, iotaKh, fullShapeTensor,
        rewriter.getDenseI64ArrayAttr({0, 1, 2, 3}));
    iotaKw = stablehlo::DynamicBroadcastInDimOp::create(
        rewriter, loc, dynFullTy, iotaKw, fullShapeTensor,
        rewriter.getDenseI64ArrayAttr({0, 1, 2, 3}));
    iotaOh = stablehlo::DynamicBroadcastInDimOp::create(
        rewriter, loc, dynFullTy, iotaOh, fullShapeTensor,
        rewriter.getDenseI64ArrayAttr({0, 1, 2, 3}));
    iotaOw = stablehlo::DynamicBroadcastInDimOp::create(
        rewriter, loc, dynFullTy, iotaOw, fullShapeTensor,
        rewriter.getDenseI64ArrayAttr({0, 1, 2, 3}));

    // h_indices = kh * dilation[0] + oh * stride[0]
    Value khDilation = hlo::getConstTensor<int32_t>(
        rewriter, op, {static_cast<int32_t>(dilation[0])}, {}).value();
    Value hIndices = chlo::BroadcastMulOp::create(rewriter, loc, iotaKh, khDilation, nullptr);
    Value hStride = hlo::getConstTensor<int32_t>(
        rewriter, op, {static_cast<int32_t>(stride[0])}, {}).value();
    Value hOffset = chlo::BroadcastMulOp::create(rewriter, loc, iotaOh, hStride, nullptr);
    hIndices = chlo::BroadcastAddOp::create(rewriter, loc, hIndices, hOffset, nullptr);

    // w_indices = kw * dilation[1] + ow * stride[1]
    Value kwDilation = hlo::getConstTensor<int32_t>(
        rewriter, op, {static_cast<int32_t>(dilation[1])}, {}).value();
    Value wIndices = chlo::BroadcastMulOp::create(rewriter, loc, iotaKw, kwDilation, nullptr);
    Value wStride = hlo::getConstTensor<int32_t>(
        rewriter, op, {static_cast<int32_t>(stride[1])}, {}).value();
    Value wOffset = chlo::BroadcastMulOp::create(rewriter, loc, iotaOw, wStride, nullptr);
    wIndices = chlo::BroadcastAddOp::create(rewriter, loc, wIndices, wOffset, nullptr);

    // Flatten to [totalIndices, 1] and concatenate to [totalIndices, 2]
    Value totalIndicesI64 = arith::MulIOp::create(rewriter, loc,
        arith::MulIOp::create(rewriter, loc, constI64(kernelSize[0] * kernelSize[1]), outHI64),
        outWI64);
    Value reshapeShape = tensor::FromElementsOp::create(
        rewriter, loc, ValueRange{totalIndicesI64, oneI64});
    auto dynFlatIdx1Ty = RankedTensorType::get({ShapedType::kDynamic, 1}, indexTy);
    hIndices = stablehlo::DynamicReshapeOp::create(
        rewriter, loc, dynFlatIdx1Ty, hIndices, reshapeShape);
    wIndices = stablehlo::DynamicReshapeOp::create(
        rewriter, loc, dynFlatIdx1Ty, wIndices, reshapeShape);

    auto dynFlatIdx2Ty = RankedTensorType::get({ShapedType::kDynamic, 2}, indexTy);
    Value indicesTensor = stablehlo::ConcatenateOp::create(
        rewriter, loc, dynFlatIdx2Ty, ValueRange{hIndices, wIndices},
        rewriter.getI64IntegerAttr(1));

    // DynamicGather with slice_sizes = [N, C, 1, 1]
    auto gatherDims = stablehlo::GatherDimensionNumbersAttr::get(
        rewriter.getContext(),
        /*offsetDims=*/{0, 1}, /*collapsedSliceDims=*/{2, 3},
        /*operandBatchingDims=*/{}, /*startIndicesBatchingDims=*/{},
        /*startIndexMap=*/{2, 3}, /*indexVecDim=*/1);

    Value nDim = arith::IndexCastOp::create(
        rewriter, loc, i64Ty, tensor::DimOp::create(rewriter, loc, paddedInput, 0));
    Value cDim = arith::IndexCastOp::create(
        rewriter, loc, i64Ty, tensor::DimOp::create(rewriter, loc, paddedInput, 1));
    Value sliceSizesTensor = tensor::FromElementsOp::create(
        rewriter, loc, RankedTensorType::get({4}, i64Ty),
        ValueRange{nDim, cDim, oneI64, oneI64});

    auto gatherResultTy = RankedTensorType::get(
        {N, C, ShapedType::kDynamic}, inputTy.getElementType());
    Value result = stablehlo::DynamicGatherOp::create(
        rewriter, loc, gatherResultTy, paddedInput, indicesTensor,
        sliceSizesTensor, gatherDims);

    // Reshape to [N, C*kH*kW, outH*outW]
    Value cKernelProd;
    if (C != ShapedType::kDynamic) {
      cKernelProd = constI64(C * kernelSize[0] * kernelSize[1]);
    } else {
      cKernelProd = arith::MulIOp::create(rewriter, loc, cDim,
                                           constI64(kernelSize[0] * kernelSize[1]));
    }
    Value outHW = arith::MulIOp::create(rewriter, loc, outHI64, outWI64);

    if (!batched) {
      Value unbatchedShape = tensor::FromElementsOp::create(
          rewriter, loc, ValueRange{cKernelProd, outHW});
      result = stablehlo::DynamicReshapeOp::create(
          rewriter, loc, outTy, result, unbatchedShape);
    } else {
      Value outReshapeShape = tensor::FromElementsOp::create(
          rewriter, loc, ValueRange{nDim, cKernelProd, outHW});
      result = stablehlo::DynamicReshapeOp::create(
          rewriter, loc, outTy, result, outReshapeShape);
    }

    rewriter.replaceOp(op, result);
    return success();
  }

  // Static spatial path
  int64_t outH =
      (H + 2 * padding[0] - dilation[0] * (kernelSize[0] - 1) - 1) / stride[0] +
      1;
  int64_t outW =
      (W + 2 * padding[1] - dilation[1] * (kernelSize[1] - 1) - 1) / stride[1] +
      1;

  // Create separate iotas for each dimension and broadcast
  Value iotaKh = stablehlo::IotaOp::create(
      rewriter, loc, RankedTensorType::get({kernelSize[0], 1, 1, 1}, indexTy),
      0);
  Value iotaKw = stablehlo::IotaOp::create(
      rewriter, loc, RankedTensorType::get({1, kernelSize[1], 1, 1}, indexTy),
      1);
  Value iotaOh = stablehlo::IotaOp::create(
      rewriter, loc, RankedTensorType::get({1, 1, outH, 1}, indexTy), 2);
  Value iotaOw = stablehlo::IotaOp::create(
      rewriter, loc, RankedTensorType::get({1, 1, 1, outW}, indexTy), 3);

  // Broadcast to full shape [kernelSize[0], kernelSize[1], outH, outW]
  SmallVector<int64_t> fullShape = {kernelSize[0], kernelSize[1], outH, outW};
  auto fullTy = RankedTensorType::get(fullShape, indexTy);
  iotaKh = stablehlo::BroadcastInDimOp::create(
      rewriter, loc, fullTy, iotaKh,
      rewriter.getDenseI64ArrayAttr({0, 1, 2, 3}));
  iotaKw = stablehlo::BroadcastInDimOp::create(
      rewriter, loc, fullTy, iotaKw,
      rewriter.getDenseI64ArrayAttr({0, 1, 2, 3}));
  iotaOh = stablehlo::BroadcastInDimOp::create(
      rewriter, loc, fullTy, iotaOh,
      rewriter.getDenseI64ArrayAttr({0, 1, 2, 3}));
  iotaOw = stablehlo::BroadcastInDimOp::create(
      rewriter, loc, fullTy, iotaOw,
      rewriter.getDenseI64ArrayAttr({0, 1, 2, 3}));

  // Compute h indices: kh * dilation[0] + oh * stride[0]
  Value khDilation = hlo::getConstTensor<int32_t>(
                         rewriter, op, {static_cast<int32_t>(dilation[0])}, {})
                         .value();
  Value hIndices =
      chlo::BroadcastMulOp::create(rewriter, loc, iotaKh, khDilation, nullptr);
  Value hStride = hlo::getConstTensor<int32_t>(
                      rewriter, op, {static_cast<int32_t>(stride[0])}, {})
                      .value();
  Value hOffset =
      chlo::BroadcastMulOp::create(rewriter, loc, iotaOh, hStride, nullptr);
  hIndices =
      chlo::BroadcastAddOp::create(rewriter, loc, hIndices, hOffset, nullptr);

  // Compute w indices: kw * dilation[1] + ow * stride[1]
  Value kwDilation = hlo::getConstTensor<int32_t>(
                         rewriter, op, {static_cast<int32_t>(dilation[1])}, {})
                         .value();
  Value wIndices =
      chlo::BroadcastMulOp::create(rewriter, loc, iotaKw, kwDilation, nullptr);
  Value wStride = hlo::getConstTensor<int32_t>(
                      rewriter, op, {static_cast<int32_t>(stride[1])}, {})
                      .value();
  Value wOffset =
      chlo::BroadcastMulOp::create(rewriter, loc, iotaOw, wStride, nullptr);
  wIndices =
      chlo::BroadcastAddOp::create(rewriter, loc, wIndices, wOffset, nullptr);

  // Flatten and stack indices
  int64_t totalIndices = kernelSize[0] * kernelSize[1] * outH * outW;
  hIndices = stablehlo::ReshapeOp::create(
      rewriter, loc, RankedTensorType::get({totalIndices, 1}, indexTy),
      hIndices);
  wIndices = stablehlo::ReshapeOp::create(
      rewriter, loc, RankedTensorType::get({totalIndices, 1}, indexTy),
      wIndices);
  Value indicesTensor = stablehlo::ConcatenateOp::create(
      rewriter, loc, ValueRange{hIndices, wIndices}, 1);

  Value result = stablehlo::GatherOp::create(
      rewriter, loc,
      RankedTensorType::get({N, C, kernelSize[0] * kernelSize[1] * outH * outW},
                            inputTy.getElementType()),
      paddedInput, indicesTensor,
      stablehlo::GatherDimensionNumbersAttr::get(
          rewriter.getContext(),
          /*offsetDims=*/{0, 1}, /*collapsedSliceDims=*/{2, 3},
          /*operandBatchingDims=*/{}, /*startIndicesBatchingDims=*/{},
          /*startIndexMap=*/{2, 3}, /*indexVecDim=*/1),
      rewriter.getDenseI64ArrayAttr({N, C, 1, 1}));

  result = stablehlo::ReshapeOp::create(
      rewriter, loc,
      RankedTensorType::get({N, C * kernelSize[0] * kernelSize[1], outH * outW},
                            inputTy.getElementType()),
      result);

  if (!batched)
    result = stablehlo::ReshapeOp::create(rewriter, loc, outTy, result);

  rewriter.replaceOp(op, result);
  return success();
}

void mlir::torch::torch_to_stablehlo::populatePoolingOpPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, const TorchToStablehloOptions &options) {
  MLIRContext *context = patterns.getContext();
#define INSERT_ATEN_POOLING_PATTERN(AtenOp)                                    \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenOp<AtenOp>>(typeConverter, context, options)
  INSERT_ATEN_POOLING_PATTERN(AtenMaxPool1dWithIndicesOp);
  INSERT_ATEN_POOLING_PATTERN(AtenMaxPool2dWithIndicesOp);
  INSERT_ATEN_POOLING_PATTERN(AtenCumsumOp);
  INSERT_ATEN_POOLING_PATTERN(Aten_AdaptiveAvgPool2dBackwardOp);
  INSERT_ATEN_POOLING_PATTERN(AtenIm2colOp);
#undef INSERT_ATEN_POOLING_PATTERN

#define INSERT_ATEN_MAXPOOL_PATTERN(AtenOp, Dim)                               \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenMaxPoolOp<AtenOp, Dim>>(typeConverter, context,      \
                                                  options)
  INSERT_ATEN_MAXPOOL_PATTERN(AtenMaxPool1dOp, 1);
  INSERT_ATEN_MAXPOOL_PATTERN(AtenMaxPool2dOp, 2);
  INSERT_ATEN_MAXPOOL_PATTERN(AtenMaxPool3dOp, 3);
#undef INSERT_ATEN_MAXPOOL_PATTERN

#define INSERT_ATEN_AVGPOOL_PATTERN(AtenOp, Dim)                               \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenAvgPoolOp<AtenOp, Dim>>(typeConverter, context,      \
                                                  options)
  INSERT_ATEN_AVGPOOL_PATTERN(AtenAvgPool1dOp, 1);
  INSERT_ATEN_AVGPOOL_PATTERN(AtenAvgPool2dOp, 2);
  INSERT_ATEN_AVGPOOL_PATTERN(AtenAvgPool3dOp, 3);
#undef INSERT_ATEN_AVGPOOL_PATTERN

  target.addIllegalOp<AtenMaxPool2dWithIndicesBackwardOp>();
  patterns.add<ConvertAtenOp<AtenMaxPool2dWithIndicesBackwardOp>>(
      typeConverter, context, options);
}
