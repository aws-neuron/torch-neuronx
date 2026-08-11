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
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch-mlir/Conversion/TorchToStablehlo/StablehloLegalizeUtils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "llvm/Support/Debug.h"
#include <numeric>

#define DEBUG_TYPE "torch-to-stablehlo-viewlike"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::TorchConversion;
using namespace mlir::torch::torch_to_stablehlo;

namespace {
// A dimension index from torch.dialect might outside the range [0, dimSize].
// The function is used to normalize the input index into the range.
Value getNormalizedDimSizeInternal(PatternRewriter &rewriter, Operation *op,
                                   Value index, Value dimSize) {
  auto loc = op->getLoc();
  Value zero = arith::ConstantOp::create(
      rewriter, loc, rewriter.getIntegerAttr(rewriter.getI64Type(), 0));

  // To normalize index into range [-dimSize, dimSize]
  // index = min(max(-dimSize, index), dimSize)
  auto negDimSize = arith::SubIOp::create(rewriter, loc, zero, dimSize);
  index = arith::MaxSIOp::create(rewriter, loc, negDimSize, index);
  index = arith::MinSIOp::create(rewriter, loc, dimSize, index);

  auto dimSizePlusIndex = arith::AddIOp::create(rewriter, loc, dimSize, index);
  auto indexPositive = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::sge, index, zero);
  // get positive index: (index >=0) ? index: index + dimSize
  return arith::SelectOp::create(rewriter, loc, indexPositive, index,
                                 dimSizePlusIndex);
}

Value getDynamicSliceInternal(PatternRewriter &rewriter, Operation *op,
                              Type outTy, Value input, Value startIndex,
                              Value endIndex, Value step, size_t dimIndex,
                              ArrayRef<Value> dimSizes,
                              size_t dimSizeIndexBits) {
  auto loc = op->getLoc();
  // startIndex & endIndex has been normailized into range [0, dSize]
  Type intType = rewriter.getIntegerType(dimSizeIndexBits);
  Value zero = arith::ConstantOp::create(rewriter, loc,
                                         rewriter.getIntegerAttr(intType, 0));
  Value one = arith::ConstantOp::create(rewriter, loc,
                                        rewriter.getIntegerAttr(intType, 1));

  auto inputTy = cast<RankedTensorType>(input.getType());
  size_t rank = inputTy.getRank();

  // When stride is 1 and output shape is static, emit the simpler
  // stablehlo.dynamic_slice which takes scalar start indices and a static
  // slice_sizes attribute. This avoids real_dynamic_slice which some
  // downstream compilers cannot handle (as of 2026-04, neuronx-cc).
  //
  // Expected IR patterns for step (any of these are recognized):
  //   unrealized_conversion_cast(torch.constant.int 1) → i64
  //   arith.trunci(unrealized_conversion_cast(...)) → i32 (dimSizeIndexBits=32)
  //   arith.constant 1 : iN
  //   torch.constant.int 1
  auto outRankedTy = dyn_cast<RankedTensorType>(outTy);
  bool stepIsOne = false;
  {
    Value v = step;
    if (auto cast = v.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (cast.getInputs().size() == 1)
        v = cast.getInputs()[0];
    }
    if (auto trunc = v.getDefiningOp<arith::TruncIOp>())
      v = trunc.getIn();
    if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
        stepIsOne = intAttr.getInt() == 1;
    }
    if (!stepIsOne) {
      if (auto torchCst = v.getDefiningOp<Torch::ConstantIntOp>())
        stepIsOne = torchCst.getValue() == 1;
    }
  }
  if (stepIsOne && outRankedTy && outRankedTy.hasStaticShape()) {
    // DynamicSliceOp clamps start to [0, dim - size]. This is safe because
    // the caller normalizes start/end to [0, dimSize], so start <= dim - size
    // always holds when the output shape is static and step == 1.
    auto scalarType = RankedTensorType::get({}, rewriter.getI64Type());
    SmallVector<Value, 4> scalarStarts;
    for (size_t r = 0; r < rank; ++r) {
      Value idx = (r == dimIndex) ? startIndex : zero;
      // Widen to i64 to match slice_sizes attribute type.
      if (dimSizeIndexBits == 32)
        idx = arith::ExtSIOp::create(rewriter, loc, rewriter.getI64Type(), idx);
      scalarStarts.push_back(
          tensor::FromElementsOp::create(rewriter, loc, scalarType, idx)
              .getResult());
    }
    return stablehlo::DynamicSliceOp::create(
        rewriter, loc, outTy, input, scalarStarts,
        rewriter.getDenseI64ArrayAttr(outRankedTy.getShape()));
  }
  LLVM_DEBUG({
    if (outRankedTy && outRankedTy.hasStaticShape() && !stepIsOne)
      llvm::dbgs() << "getDynamicSliceInternal: static output shape but "
                      "could not prove step==1; falling back to "
                      "RealDynamicSliceOp\n";
  });

  // Fall through to RealDynamicSliceOp for non-unit strides or dynamic output.
  SmallVector<Value, 4> startIndices;
  SmallVector<Value, 4> endIndices;
  SmallVector<Value, 4> strides;
  startIndices.reserve(rank);
  endIndices.reserve(rank);
  strides.reserve(rank);

  auto endIndexIsZero = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::eq, endIndex, zero);
  endIndex = arith::SelectOp::create(rewriter, loc, endIndexIsZero,
                                     dimSizes[dimIndex], endIndex);

  for (size_t r = 0; r < rank; ++r) {
    if (r == dimIndex) {
      startIndices.push_back(startIndex);
      endIndices.push_back(endIndex);
      strides.push_back(step);
    } else {
      startIndices.push_back(zero);
      endIndices.push_back(dimSizes[r]);
      strides.push_back(one);
    }
  }

  auto startTensor =
      tensor::FromElementsOp::create(rewriter, loc, startIndices).getResult();
  auto endTensor =
      tensor::FromElementsOp::create(rewriter, loc, endIndices).getResult();
  auto stridesTensor =
      tensor::FromElementsOp::create(rewriter, loc, strides).getResult();

  return stablehlo::RealDynamicSliceOp::create(
      rewriter, loc, outTy, input, startTensor, endTensor, stridesTensor);
}

// Get a dynamic slice of the tensor from startIndex to endIndex with stride
// step on the specifed dimension. The input startIndex(default to 0),
// endIndex(default to dimSize), and step(default to 1) can be optional.
FailureOr<Value> getDynamicSlice(PatternRewriter &rewriter, Operation *op,
                                 Type outTy, Value input,
                                 std::optional<Value> startIndexOpt,
                                 std::optional<Value> endIndexOpt,
                                 std::optional<Value> stepOpt, int64_t dim,
                                 size_t dimSizeIndexBits) {
  auto loc = op->getLoc();
  auto inputTy = dyn_cast<RankedTensorType>(input.getType());
  auto rank = inputTy.getRank();

  dim = (dim + rank) % rank;
  Value dimSize = arith::IndexCastOp::create(
      rewriter, loc, rewriter.getI64Type(),
      tensor::DimOp::create(rewriter, loc, input, dim));

  Value normStartIndex =
      startIndexOpt
          ? getNormalizedDimSizeInternal(rewriter, op, *startIndexOpt, dimSize)
          : arith::ConstantOp::create(
                rewriter, loc,
                rewriter.getIntegerAttr(rewriter.getI64Type(), 0));
  Value normEndIndex =
      endIndexOpt
          ? getNormalizedDimSizeInternal(rewriter, op, *endIndexOpt, dimSize)
          : dimSize;
  Value step = stepOpt ? *stepOpt
                       : arith::ConstantOp::create(
                             rewriter, loc,
                             rewriter.getIntegerAttr(rewriter.getI64Type(), 1));

  if (dimSizeIndexBits == 32) {
    Type intType = rewriter.getIntegerType(dimSizeIndexBits);
    normStartIndex =
        arith::TruncIOp::create(rewriter, loc, intType, normStartIndex);
    normEndIndex =
        arith::TruncIOp::create(rewriter, loc, intType, normEndIndex);
    step = arith::TruncIOp::create(rewriter, loc, intType, step);
  }
  FailureOr<SmallVector<Value, 4>> dimSizesInfo =
      hlo::getDimSizesOfTensor(rewriter, op, input, dimSizeIndexBits);
  if (failed(dimSizesInfo))
    return rewriter.notifyMatchFailure(
        op, "failed to get dimension sizes of the input");

  auto dimSizes = *dimSizesInfo;
  return getDynamicSliceInternal(rewriter, op, outTy, input, normStartIndex,
                                 normEndIndex, step, dim, dimSizes,
                                 dimSizeIndexBits);
}

// This defines a template to construct ops whose legalizations are
// specialized.
template <typename AtenOpT>
class ConvertAtenViewOp : public ConvertAtenOp<AtenOpT> {
public:
  using ConvertAtenOp<AtenOpT>::ConvertAtenOp;
  using OpAdaptor = typename AtenOpT::Adaptor;

  unsigned getBitWidth(Type type) const {
    if (auto complexTy = dyn_cast<ComplexType>(type))
      return 2 * getBitWidth(complexTy.getElementType());
    return type.getIntOrFloatBitWidth();
  }

  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto rankType = dyn_cast<RankedTensorType>(adaptor.getSelf().getType());
    if (!rankType)
      return op.emitError("Only ranked tensor types are currently supported.");
    auto loc = op.getLoc();

    // support AtenViewDtypeOp
    if (isa<AtenViewDtypeOp>(op)) {
      auto self = adaptor.getSelf();
      auto baseResultTy = dyn_cast<BaseTensorType>(op.getType());

      auto operandElt = rankType.getElementType();
      auto targetElt = baseResultTy.getDtype();
      auto operandEltBitWidth = getBitWidth(operandElt);
      auto targetEltBitWidth = getBitWidth(targetElt);
      auto operandSizes = rankType.getShape();
      int64_t inputRank = rankType.getRank();

      auto resultType =
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              baseResultTy);
      auto resultRankedType = dyn_cast<RankedTensorType>(resultType);

      if (operandEltBitWidth == targetEltBitWidth) {
        // Same bitwidth: just bitcast_convert, no shape change.
        rewriter.replaceOpWithNewOp<stablehlo::BitcastConvertOp>(
            op, resultType, self);
        return success();
      } else if (operandEltBitWidth > targetEltBitWidth) {
        // Narrow (e.g., f32 -> u8):
        //   bitcast_convert adds trailing dim: [*, D] -> [*, D, ratio]
        //   reshape merges last two dims:      [*, D, ratio] -> [*, D*ratio]
        int64_t ratio = operandEltBitWidth / targetEltBitWidth;
        SmallVector<int64_t> bitcastShape(operandSizes);
        bitcastShape.push_back(ratio);
        auto bitcastType = RankedTensorType::get(
            bitcastShape, resultRankedType.getElementType());
        Value bitcast = stablehlo::BitcastConvertOp::create(
            rewriter, loc, bitcastType, self);

        if (resultRankedType.hasStaticShape()) {
          rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(op, resultType,
                                                            bitcast);
          return success();
        }

        // Dynamic: compute output shape by merging last two bitcast dims.
        auto dimSizes =
            hlo::getDimSizesOfTensor(rewriter, op, bitcast, 64);
        if (failed(dimSizes))
          return failure();
        int64_t bitcastRank = bitcastShape.size();
        SmallVector<Value, 4> outputDims;
        for (int64_t i = 0; i < bitcastRank - 2; ++i)
          outputDims.push_back((*dimSizes)[i]);
        Value lastDim = arith::MulIOp::create(
            rewriter, loc, (*dimSizes)[bitcastRank - 2],
            (*dimSizes)[bitcastRank - 1]);
        outputDims.push_back(lastDim);

        Value shapeTensor =
            tensor::FromElementsOp::create(rewriter, loc, outputDims);
        rewriter.replaceOpWithNewOp<stablehlo::DynamicReshapeOp>(
            op, resultType, bitcast, shapeTensor);
        return success();
      } else {
        // Widen (e.g., u8 -> f32):
        //   reshape splits last dim:         [*, D] -> [*, D/ratio, ratio]
        //   bitcast_convert removes last dim: [*, D/ratio, ratio] -> [*, D/ratio]
        int64_t ratio = targetEltBitWidth / operandEltBitWidth;
        int64_t lastDim = operandSizes[inputRank - 1];

        if (ShapedType::isDynamic(lastDim)) {
          return rewriter.notifyMatchFailure(
              op, "last dim must be static for widening view(dtype) "
                  "(must be divisible by targetBitWidth / operandBitWidth)");
        }
        if (lastDim % ratio != 0) {
          return rewriter.notifyMatchFailure(
              op, "last dim must be divisible by targetBitWidth / "
                  "operandBitWidth for widening view(dtype)");
        }

        // Build the intermediate shape: [*, lastDim/ratio, ratio]
        SmallVector<int64_t> reshapedShape(operandSizes.begin(),
                                          operandSizes.end() - 1);
        reshapedShape.push_back(lastDim / ratio);
        reshapedShape.push_back(ratio);
        auto reshapedType = RankedTensorType::get(
            reshapedShape, rankType.getElementType());

        Value reshaped;
        if (rankType.hasStaticShape()) {
          reshaped = stablehlo::ReshapeOp::create(rewriter, loc, reshapedType,
                                                  self);
        } else {
          auto dimSizes =
              hlo::getDimSizesOfTensor(rewriter, op, self, 64);
          if (failed(dimSizes))
            return failure();
          auto intType = rewriter.getI64Type();
          SmallVector<Value, 4> reshapeDims;
          for (int64_t i = 0; i < inputRank - 1; ++i)
            reshapeDims.push_back((*dimSizes)[i]);
          reshapeDims.push_back(arith::ConstantOp::create(
              rewriter, loc, rewriter.getIntegerAttr(intType, lastDim / ratio)));
          reshapeDims.push_back(arith::ConstantOp::create(
              rewriter, loc, rewriter.getIntegerAttr(intType, ratio)));
          Value shapeTensor =
              tensor::FromElementsOp::create(rewriter, loc, reshapeDims);
          reshaped = stablehlo::DynamicReshapeOp::create(
              rewriter, loc, reshapedType, self, shapeTensor);
        }

        // bitcast_convert consumes the trailing ratio dim.
        rewriter.replaceOpWithNewOp<stablehlo::BitcastConvertOp>(
            op, resultType, reshaped);
        return success();
      }
    }

    // collect Value of dims
    SmallVector<Value, 4> dimSizes;
    if (!getAtenViewOpSizes(op, adaptor, rewriter, dimSizes)) {
      return op.emitError("Dims size must be a list of Scalar");
    }

    if (dimSizes.size() == 0 || rankType.getRank() == 0) {
      rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(
          op,
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              op.getType()),
          adaptor.getSelf());
      return success();
    }

    // collect constant dim size which == -1
    SmallVector<size_t> negOneIndex;
    for (size_t i = 0; i < dimSizes.size(); i++) {
      int64_t dim;
      if (matchPattern(dimSizes[i], m_TorchConstantInt(&dim))) {
        if (dim == -1) {
          negOneIndex.push_back(i);
        }
      }
    }
    if (negOneIndex.size() > 1) {
      return op.emitError("Only support at most one -1 in view target dims");
    }

    std::for_each(dimSizes.begin(), dimSizes.end(), [&](Value &dSize) {
      dSize = ToI64Op::create(rewriter, loc, dSize).getResult();
      return dSize;
    });

    Value numel = shape::NumElementsOp::create(
        rewriter, loc,
        shape::ShapeOfOp::create(rewriter, loc, adaptor.getSelf()));
    numel =
        arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(), numel);

    // note: assuming that -1 doesn't arise from dynamic value
    if (negOneIndex.size() == 1) {
      size_t index = negOneIndex[0];
      Value realDim = numel;
      for (size_t i = 0; i < dimSizes.size(); i++) {
        if (i != index) {
          realDim = arith::DivUIOp::create(rewriter, loc, realDim, dimSizes[i]);
        }
      }
      // update -1 to realDim
      dimSizes[index] = realDim;
    }

    Value stablehloShape =
        tensor::FromElementsOp::create(rewriter, loc, dimSizes);
    rewriter.replaceOpWithNewOp<stablehlo::DynamicReshapeOp>(
        op,
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()),
        adaptor.getSelf(), stablehloShape);
    return success();
  }

  bool getAtenViewOpSizes(AtenOpT op, OpAdaptor adaptor,
                          ConversionPatternRewriter &rewriter,
                          SmallVector<Value, 4> &dimSizes) const;
};

template <>
bool ConvertAtenViewOp<AtenViewDtypeOp>::getAtenViewOpSizes(
    AtenViewDtypeOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter,
    SmallVector<Value, 4> &dimSizes) const {
  return false;
}

template <>
bool ConvertAtenViewOp<AtenViewOp>::getAtenViewOpSizes(
    AtenViewOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter,
    SmallVector<Value, 4> &dimSizes) const {
  return getListConstructElements(adaptor.getSize(), dimSizes);
}

template <>
bool ConvertAtenViewOp<AtenReshapeOp>::getAtenViewOpSizes(
    AtenReshapeOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter,
    SmallVector<Value, 4> &dimSizes) const {
  return getListConstructElements(adaptor.getShape(), dimSizes);
}
} // namespace

template <>
LogicalResult ConvertAtenOp<AtenSliceTensorOp>::matchAndRewrite(
    AtenSliceTensorOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto self = adaptor.getSelf();
  auto selfTy = cast<RankedTensorType>(self.getType());
  if (!selfTy)
    return op.emitError("only ranked tensor types are supported");
  auto outTy =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
  int64_t dim;
  if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
    return rewriter.notifyMatchFailure(
        op, "only constant dim is currently supported");
  int64_t inputRank = selfTy.getRank();
  dim = toPositiveDim(dim, inputRank);
  if (!isValidDim(dim, inputRank))
    return rewriter.notifyMatchFailure(op, "dim is statically invalid");

  auto getOptionalVal = [&](Value val) -> std::optional<Value> {
    if (isa<Torch::NoneType>(val.getType())) {
      return std::nullopt;
    }
    // Handle 0-d tensor start/end: extract scalar i64 from tensor<i64>
    if (auto tensorTy = dyn_cast<RankedTensorType>(val.getType())) {
      if (tensorTy.getRank() == 0 && tensorTy.getElementType().isInteger(64)) {
        val =
            tensor::ExtractOp::create(rewriter, op.getLoc(), val, ValueRange{});
        return val;
      }
      // Tensor type but not the expected 0-d i64 — signal failure.
      return std::nullopt;
    }
    return val;
  };

  std::optional<Value> start = getOptionalVal(adaptor.getStart());
  std::optional<Value> end = getOptionalVal(adaptor.getEnd());
  std::optional<Value> step = getOptionalVal(adaptor.getStep());

  // If start/end were tensors but not 0-d i64, getOptionalVal returned nullopt.
  // Detect this: a non-None original type that produced nullopt is an error.
  if (!start && !isa<Torch::NoneType>(op.getStart().getType()))
    return rewriter.notifyMatchFailure(
        op, "expected 0-d tensor<i64> for start, got unsupported tensor type");
  if (!end && !isa<Torch::NoneType>(op.getEnd().getType()))
    return rewriter.notifyMatchFailure(
        op, "expected 0-d tensor<i64> for end, got unsupported tensor type");

  FailureOr<Value> sliceInfo =
      getDynamicSlice(rewriter, op, outTy, self, start, end, step, dim,
                      options.dimSizeIndexBits);
  if (failed(sliceInfo))
    return op.emitError("can not create a dynmaic slice");

  auto slice = *sliceInfo;
  rewriter.replaceOp(op, slice);
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenSqueezeOp>::matchAndRewrite(
    AtenSqueezeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto self = adaptor.getSelf();
  auto selfTy = cast<RankedTensorType>(self.getType());
  if (!selfTy)
    return op.emitError("only ranked tensor types are supported");

  auto rank = selfTy.getRank();
  if (rank == 0)
    return rewriter.notifyMatchFailure(
        op, "The rank of tensor must be greater than 0");

  SmallVector<int64_t, 4> dims;
  dims.reserve(rank);
  for (int r = 0; r < rank; ++r) {
    auto dSize = selfTy.getShape()[r];
    if (dSize == ShapedType::kDynamic)
      return rewriter.notifyMatchFailure(
          op, "the size of the dimension being squeezed can't be unknown");
    if (dSize != 1)
      dims.push_back(r);
  }
  if (dims.size() == 0) {
    rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(
        op, getTypeConverter()->convertType(op.getType()), self);
    return success();
  }

  auto newDimSizesInfo = hlo::getDimIndexOfTensor(rewriter, op, self, dims);
  if (failed(newDimSizesInfo))
    return rewriter.notifyMatchFailure(
        op, "failed to get dimension sizes of the input");
  auto newDimSizes = *newDimSizesInfo;
  auto stablehloShape =
      tensor::FromElementsOp::create(rewriter, op.getLoc(), newDimSizes);
  rewriter.replaceOpWithNewOp<stablehlo::DynamicReshapeOp>(
      op, getTypeConverter()->convertType(op.getType()), self, stablehloShape);
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenSqueezeDimOp>::matchAndRewrite(
    AtenSqueezeDimOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto self = adaptor.getSelf();
  auto selfTy = cast<RankedTensorType>(self.getType());
  if (!selfTy)
    return op.emitError("only ranked tensor types are supported");

  auto rank = selfTy.getRank();
  if (rank == 0)
    return rewriter.notifyMatchFailure(
        op, "the rank of tensor must be greater than 0");

  int64_t dim;
  if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
    return rewriter.notifyMatchFailure(
        op, "only constant dim is currently supported");
  dim = toPositiveDim(dim, rank);
  if (!isValidDim(dim, rank))
    return rewriter.notifyMatchFailure(op, "dim is statically invalid");

  if (selfTy.getShape()[dim] != 1) {
    if (selfTy.getShape()[dim] == ShapedType::kDynamic)
      return rewriter.notifyMatchFailure(
          op, "the size of the dimension being squeezed is can't be unknown");

    rewriter.replaceOp(op, adaptor.getSelf());
    return success();
  }

  SmallVector<int64_t, 4> dims(rank);
  std::iota(dims.begin(), dims.end(), 0);
  dims.erase(dims.begin() + dim);
  if (dims.size() == 0) {
    rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(
        op, getTypeConverter()->convertType(op.getType()), self);
    return success();
  }
  auto newDimSizesInfo = hlo::getDimIndexOfTensor(rewriter, op, self, dims);
  if (failed(newDimSizesInfo))
    return rewriter.notifyMatchFailure(
        op, "failed to get dimension sizes of the input");
  auto newDimSizes = *newDimSizesInfo;
  auto stablehloShape =
      tensor::FromElementsOp::create(rewriter, op.getLoc(), newDimSizes);
  rewriter.replaceOpWithNewOp<stablehlo::DynamicReshapeOp>(
      op, getTypeConverter()->convertType(op.getType()), self, stablehloShape);
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenUnsqueezeOp>::matchAndRewrite(
    AtenUnsqueezeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto selfType = dyn_cast<TensorType>(adaptor.getSelf().getType());
  if (!selfType) {
    return op.emitError("only tensor types are currently supported");
  }

  int64_t dim;
  if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
    return op->emitError("dim must be a Scalar constant");
  int64_t inputRank =
      cast<RankedTensorType>(adaptor.getSelf().getType()).getRank();
  dim = toPositiveDim(dim, inputRank + 1);
  if (!isValidDim(dim, inputRank + 1))
    return rewriter.notifyMatchFailure(op, "dim is statically invalid");

  auto unsqzTensorInfo =
      hlo::unsqueezeTensor(rewriter, op, adaptor.getSelf(), {dim});
  if (failed(unsqzTensorInfo))
    return rewriter.notifyMatchFailure(op,
                                       "failed to create unsqueezed tensor");

  rewriter.replaceOp(op, *unsqzTensorInfo);
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenViewAsComplexOp>::matchAndRewrite(
    AtenViewAsComplexOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto inputType = dyn_cast<RankedTensorType>(adaptor.getSelf().getType());
  if (!inputType)
    return op.emitError("only ranked tensor types are supported");

  auto resultType =
      dyn_cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
  if (!resultType)
    return op.emitError("failed to convert result type");

  auto loc = op.getLoc();
  auto input = adaptor.getSelf();
  auto elementType = inputType.getElementType();

  // Extract real part: input[..., 0]
  auto rank = inputType.getRank();
  SmallVector<int64_t> startIndices(rank, 0);
  SmallVector<int64_t> limitIndices(inputType.getShape().begin(),
                                    inputType.getShape().end());
  limitIndices[rank - 1] = 1;
  SmallVector<int64_t> strides(rank, 1);

  auto realSlice = stablehlo::SliceOp::create(
      rewriter, loc, input,
      DenseI64ArrayAttr::get(rewriter.getContext(), startIndices),
      DenseI64ArrayAttr::get(rewriter.getContext(), limitIndices),
      DenseI64ArrayAttr::get(rewriter.getContext(), strides));

  // Extract imaginary part: input[..., 1]
  startIndices[rank - 1] = 1;
  limitIndices[rank - 1] = 2;

  auto imagSlice = stablehlo::SliceOp::create(
      rewriter, loc, input,
      DenseI64ArrayAttr::get(rewriter.getContext(), startIndices),
      DenseI64ArrayAttr::get(rewriter.getContext(), limitIndices),
      DenseI64ArrayAttr::get(rewriter.getContext(), strides));

  // Reshape to remove last dimension (keep element type as float)
  SmallVector<int64_t> reshapeShape(resultType.getShape().begin(),
                                    resultType.getShape().end());
  auto reshapeType = RankedTensorType::get(reshapeShape, elementType);

  auto realReshaped =
      stablehlo::ReshapeOp::create(rewriter, loc, reshapeType, realSlice);
  auto imagReshaped =
      stablehlo::ReshapeOp::create(rewriter, loc, reshapeType, imagSlice);

  // Construct complex: real + imag * 1j
  auto complexResult = stablehlo::ComplexOp::create(rewriter, loc, resultType,
                                                    realReshaped, imagReshaped);

  rewriter.replaceOp(op, complexResult);
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenViewAsRealOp>::matchAndRewrite(
    AtenViewAsRealOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto inputType = dyn_cast<RankedTensorType>(adaptor.getSelf().getType());
  if (!inputType)
    return op.emitError("only ranked tensor types are supported");

  auto resultType =
      dyn_cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
  if (!resultType)
    return op.emitError("failed to convert result type");

  auto loc = op.getLoc();
  auto input = adaptor.getSelf();
  auto elementType = resultType.getElementType();

  // Extract real and imaginary parts
  auto realPart = stablehlo::RealOp::create(rewriter, loc, input);
  auto imagPart = stablehlo::ImagOp::create(rewriter, loc, input);

  // Add dimension at the end for stacking
  SmallVector<int64_t> expandedShape(inputType.getShape().begin(),
                                     inputType.getShape().end());
  expandedShape.push_back(1);
  auto expandedType = RankedTensorType::get(expandedShape, elementType);

  auto realExpanded =
      stablehlo::ReshapeOp::create(rewriter, loc, expandedType, realPart);
  auto imagExpanded =
      stablehlo::ReshapeOp::create(rewriter, loc, expandedType, imagPart);

  // Concatenate along last dimension
  auto result = stablehlo::ConcatenateOp::create(
      rewriter, loc, resultType, ValueRange{realExpanded, imagExpanded},
      rewriter.getI64IntegerAttr(inputType.getRank()));

  rewriter.replaceOp(op, result);
  return success();
}

// aten.real: extract real part from complex tensor
template <>
LogicalResult ConvertAtenOp<AtenRealOp>::matchAndRewrite(
    AtenRealOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto inputType = dyn_cast<RankedTensorType>(adaptor.getSelf().getType());
  if (!inputType)
    return op.emitError("only ranked tensor types are supported");

  if (!isa<ComplexType>(inputType.getElementType()))
    return op.emitError("input must have complex element type");

  auto resultType =
      dyn_cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
  if (!resultType)
    return op.emitError("failed to convert result type");

  auto realResult =
      stablehlo::RealOp::create(rewriter, op.getLoc(), resultType,
                                adaptor.getSelf());
  rewriter.replaceOp(op, realResult);
  return success();
}

// aten.imag: extract imaginary part from complex tensor
template <>
LogicalResult ConvertAtenOp<AtenImagOp>::matchAndRewrite(
    AtenImagOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto inputType = dyn_cast<RankedTensorType>(adaptor.getSelf().getType());
  if (!inputType)
    return op.emitError("only ranked tensor types are supported");

  if (!isa<ComplexType>(inputType.getElementType()))
    return op.emitError("input must have complex element type");

  auto resultType =
      dyn_cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
  if (!resultType)
    return op.emitError("failed to convert result type");

  auto imagResult =
      stablehlo::ImagOp::create(rewriter, op.getLoc(), resultType,
                                adaptor.getSelf());
  rewriter.replaceOp(op, imagResult);
  return success();
}

// aten.complex: construct complex tensor from real and imaginary parts
template <>
LogicalResult ConvertAtenOp<AtenComplexOp>::matchAndRewrite(
    AtenComplexOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto realType = dyn_cast<RankedTensorType>(adaptor.getReal().getType());
  auto imagType = dyn_cast<RankedTensorType>(adaptor.getImag().getType());
  if (!realType || !imagType)
    return op.emitError("only ranked tensor types are supported");

  auto resultType =
      dyn_cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
  if (!resultType)
    return op.emitError("failed to convert result type");

  if (!isa<ComplexType>(resultType.getElementType()))
    return op.emitError("result must have complex element type");

  auto complexResult =
      stablehlo::ComplexOp::create(rewriter, op.getLoc(), resultType,
                                   adaptor.getReal(), adaptor.getImag());
  rewriter.replaceOp(op, complexResult);
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenDiagonalOp>::matchAndRewrite(
    AtenDiagonalOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto self = adaptor.getSelf();
  auto selfTy = cast<RankedTensorType>(self.getType());
  auto loc = op.getLoc();

  int64_t offset, dim1, dim2;
  if (!matchPattern(op.getOffset(), m_TorchConstantInt(&offset)) ||
      !matchPattern(op.getDim1(), m_TorchConstantInt(&dim1)) ||
      !matchPattern(op.getDim2(), m_TorchConstantInt(&dim2)))
    return rewriter.notifyMatchFailure(
        op, "offset, dim1, and dim2 must be constant");

  int64_t inputRank = selfTy.getRank();
  dim1 = toPositiveDim(dim1, inputRank);
  dim2 = toPositiveDim(dim2, inputRank);

  if (!isValidDim(dim1, inputRank) || !isValidDim(dim2, inputRank) ||
      dim1 == dim2)
    return rewriter.notifyMatchFailure(op, "invalid diagonal dims");

  auto shape = selfTy.getShape();
  auto i32Type = rewriter.getI32Type();
  bool hasAnyDynamic =
      llvm::any_of(shape, [](int64_t s) { return s == ShapedType::kDynamic; });

  // Build iota of length diagSize and compute idx1/idx2
  Value iota;
  if (!hasAnyDynamic) {
    int64_t diagSize;
    if (offset >= 0) {
      diagSize = std::min(shape[dim1], shape[dim2] - offset);
    } else {
      diagSize = std::min(shape[dim1] + offset, shape[dim2]);
    }
    if (diagSize <= 0)
      return rewriter.notifyMatchFailure(op, "diagonal size must be positive");

    auto iotaType = RankedTensorType::get({diagSize}, i32Type);
    iota = stablehlo::IotaOp::create(rewriter, loc, iotaType,
                                     rewriter.getI64IntegerAttr(0));
  } else {
    auto i64Ty = rewriter.getI64Type();
    Value dim1Size = arith::IndexCastOp::create(
        rewriter, loc, i64Ty,
        tensor::DimOp::create(rewriter, loc, self, dim1));
    Value dim2Size = arith::IndexCastOp::create(
        rewriter, loc, i64Ty,
        tensor::DimOp::create(rewriter, loc, self, dim2));

    Value diagSizeI64;
    if (offset >= 0) {
      Value offsetConst = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(offset));
      Value dim2MinusOffset =
          arith::SubIOp::create(rewriter, loc, dim2Size, offsetConst);
      diagSizeI64 =
          arith::MinSIOp::create(rewriter, loc, dim1Size, dim2MinusOffset);
    } else {
      Value offsetConst = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(-offset));
      Value dim1MinusOffset =
          arith::SubIOp::create(rewriter, loc, dim1Size, offsetConst);
      diagSizeI64 =
          arith::MinSIOp::create(rewriter, loc, dim1MinusOffset, dim2Size);
    }

    Value iotaShape = tensor::FromElementsOp::create(rewriter, loc,
                                                      ValueRange{diagSizeI64});
    auto dynIotaType = RankedTensorType::get({ShapedType::kDynamic}, i32Type);
    iota = stablehlo::DynamicIotaOp::create(
        rewriter, loc, dynIotaType, iotaShape, rewriter.getI64IntegerAttr(0));
  }

  // Compute idx1 and idx2 from iota + offset
  Value idx1, idx2;
  if (offset < 0) {
    Value offsetConst = stablehlo::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(-offset));
    idx1 = chlo::BroadcastAddOp::create(rewriter, loc, iota, offsetConst,
                                        nullptr);
    idx2 = iota;
  } else if (offset > 0) {
    Value offsetConst = stablehlo::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(offset));
    idx1 = iota;
    idx2 = chlo::BroadcastAddOp::create(rewriter, loc, iota, offsetConst,
                                        nullptr);
  } else {
    idx1 = idx2 = iota;
  }

  // Reshape idx1/idx2 from [diagSize] to [diagSize, 1] and concatenate to
  // [diagSize, 2]
  auto idx1Ty = cast<RankedTensorType>(idx1.getType());
  int64_t diagDim = idx1Ty.getShape()[0];
  Value startIndices;
  if (!hasAnyDynamic) {
    Value idx1Reshaped = stablehlo::ReshapeOp::create(
        rewriter, loc, RankedTensorType::get({diagDim, 1}, i32Type), idx1);
    Value idx2Reshaped = stablehlo::ReshapeOp::create(
        rewriter, loc, RankedTensorType::get({diagDim, 1}, i32Type), idx2);
    startIndices = stablehlo::ConcatenateOp::create(
        rewriter, loc, RankedTensorType::get({diagDim, 2}, i32Type),
        ValueRange{idx1Reshaped, idx2Reshaped},
        rewriter.getI64IntegerAttr(1));
  } else {
    Value diagSizeI64 = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getI64Type(),
        tensor::DimOp::create(
            rewriter, loc, idx1,
            arith::ConstantIndexOp::create(rewriter, loc, 0)));
    Value oneI64 = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(1));
    Value reshapeShape = tensor::FromElementsOp::create(
        rewriter, loc, ValueRange{diagSizeI64, oneI64});
    auto dynNx1Type =
        RankedTensorType::get({ShapedType::kDynamic, 1}, i32Type);
    Value idx1Reshaped = stablehlo::DynamicReshapeOp::create(
        rewriter, loc, dynNx1Type, idx1, reshapeShape);
    Value idx2Reshaped = stablehlo::DynamicReshapeOp::create(
        rewriter, loc, dynNx1Type, idx2, reshapeShape);
    auto dynNx2Type =
        RankedTensorType::get({ShapedType::kDynamic, 2}, i32Type);
    startIndices = stablehlo::ConcatenateOp::create(
        rewriter, loc, dynNx2Type, ValueRange{idx1Reshaped, idx2Reshaped},
        rewriter.getI64IntegerAttr(1));
  }

  // Gather dimension numbers (shared between static and dynamic)
  SmallVector<int64_t> offsetDimsVec(inputRank - 2);
  std::iota(offsetDimsVec.begin(), offsetDimsVec.end(), 0);
  SmallVector<int64_t> collapsedSliceDimsVec = {std::min(dim1, dim2),
                                                 std::max(dim1, dim2)};
  SmallVector<int64_t> startIndexMapVec = {std::min(dim1, dim2),
                                            std::max(dim1, dim2)};

  auto gatherDims = stablehlo::GatherDimensionNumbersAttr::get(
      rewriter.getContext(), offsetDimsVec, collapsedSliceDimsVec, {}, {},
      startIndexMapVec, 1);

  auto outTy =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));

  // Emit GatherOp (static) or DynamicGatherOp (dynamic)
  Value result;
  if (!hasAnyDynamic) {
    SmallVector<int64_t> sliceSizes(shape.begin(), shape.end());
    sliceSizes[dim1] = sliceSizes[dim2] = 1;
    result = stablehlo::GatherOp::create(
        rewriter, loc, outTy, self, startIndices, gatherDims,
        DenseI64ArrayAttr::get(rewriter.getContext(), sliceSizes));
  } else {
    SmallVector<Value> sliceSizeValues;
    for (int64_t i = 0; i < inputRank; ++i) {
      if (i == dim1 || i == dim2) {
        sliceSizeValues.push_back(arith::ConstantOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(1)));
      } else {
        Value dimVal = tensor::DimOp::create(rewriter, loc, self, i);
        sliceSizeValues.push_back(arith::IndexCastOp::create(
            rewriter, loc, rewriter.getI64Type(), dimVal));
      }
    }
    auto sliceSizesTensorType =
        RankedTensorType::get({inputRank}, rewriter.getI64Type());
    Value sliceSizesTensor = tensor::FromElementsOp::create(
        rewriter, loc, sliceSizesTensorType, sliceSizeValues);
    result = stablehlo::DynamicGatherOp::create(
        rewriter, loc, outTy, self, startIndices, sliceSizesTensor,
        gatherDims);
  }

  rewriter.replaceOp(op, result);
  return success();
}

template <>
LogicalResult ConvertAtenOp<PrimsCollapseOp>::matchAndRewrite(
    PrimsCollapseOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto selfType = dyn_cast<TensorType>(adaptor.getA().getType());
  if (!selfType) {
    return op.emitError("only tensor types are currently supported");
  }

  auto rank = selfType.getRank();
  if (rank == 0)
    return rewriter.notifyMatchFailure(
        op, "the rank of tensor must be greater than 0");

  int64_t start, end;
  if (!matchPattern(op.getStart(), m_TorchConstantInt(&start)))
    return rewriter.notifyMatchFailure(
        op, "only constant start is currently supported");
  if (!matchPattern(op.getEnd(), m_TorchConstantInt(&end)))
    return rewriter.notifyMatchFailure(
        op, "only constant end is currently supported");

  auto collapseTensorInfo =
      hlo::collapseTensor(rewriter, op, adaptor.getA(), start, end);
  if (failed(collapseTensorInfo))
    return rewriter.notifyMatchFailure(op, "failed to create collapsed tensor");

  rewriter.replaceOp(op, *collapseTensorInfo);
  return success();
}

template <>
LogicalResult ConvertAtenOp<PrimsSplitDimOp>::matchAndRewrite(
    PrimsSplitDimOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto selfType = dyn_cast<TensorType>(adaptor.getA().getType());
  if (!selfType) {
    return op.emitError("only tensor types are currently supported");
  }

  auto rank = selfType.getRank();
  if (rank == 0)
    return rewriter.notifyMatchFailure(
        op, "the rank of tensor must be greater than 0");

  int64_t dim;
  if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
    return rewriter.notifyMatchFailure(
        op, "only constant dim is currently supported");

  int64_t outerLength;
  if (matchPattern(op.getOuterLength(), m_TorchConstantInt(&outerLength))) {
    auto splitTensorInfo =
        hlo::splitTensor(rewriter, op, adaptor.getA(), dim, outerLength);
    if (failed(splitTensorInfo))
      return rewriter.notifyMatchFailure(op, "failed to create split tensor");
    rewriter.replaceOp(op, *splitTensorInfo);
    return success();
  }

  auto loc = op->getLoc();
  auto rankTy = cast<RankedTensorType>(adaptor.getA().getType());
  auto oldShape = rankTy.getShape();
  int64_t inputRank = rankTy.getRank();
  dim = toPositiveDim(dim, inputRank);

  auto outTy = cast<RankedTensorType>(
      OpConversionPattern<PrimsSplitDimOp>::getTypeConverter()->convertType(
          op.getType()));

  auto dimSizesInfo = hlo::getDimIndexOfTensor(rewriter, op, adaptor.getA());
  if (failed(dimSizesInfo))
    return failure();
  auto dimSizes = *dimSizesInfo;

  Value outerVal = adaptor.getOuterLength();
  if (!outerVal.getType().isIndex())
    outerVal = arith::IndexCastOp::create(rewriter, loc,
                                          rewriter.getIndexType(), outerVal);

  Value innerVal =
      arith::DivSIOp::create(rewriter, loc, dimSizes[dim], outerVal);

  SmallVector<Value> newDimSizes;
  for (int64_t k = 0; k < dim; ++k)
    newDimSizes.push_back(dimSizes[k]);
  newDimSizes.push_back(outerVal);
  newDimSizes.push_back(innerVal);
  for (int64_t k = dim + 1; k < inputRank; ++k)
    newDimSizes.push_back(dimSizes[k]);

  Value shapeTensor =
      tensor::FromElementsOp::create(rewriter, loc, newDimSizes);
  rewriter.replaceOpWithNewOp<stablehlo::DynamicReshapeOp>(
      op, outTy, adaptor.getA(), shapeTensor);
  return success();
}

void mlir::torch::torch_to_stablehlo::populateViewLikeOpPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, const TorchToStablehloOptions &options) {
  MLIRContext *context = patterns.getContext();

#define INSERT_ATENOP_PATTERN(AtenOp)                                          \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenOp<AtenOp>>(typeConverter, context, options)
  INSERT_ATENOP_PATTERN(AtenSliceTensorOp);
  INSERT_ATENOP_PATTERN(AtenSqueezeOp);
  INSERT_ATENOP_PATTERN(AtenSqueezeDimOp);
  INSERT_ATENOP_PATTERN(AtenUnsqueezeOp);
  INSERT_ATENOP_PATTERN(AtenViewAsComplexOp);
  INSERT_ATENOP_PATTERN(AtenViewAsRealOp);
  INSERT_ATENOP_PATTERN(AtenRealOp);
  INSERT_ATENOP_PATTERN(AtenImagOp);
  INSERT_ATENOP_PATTERN(AtenComplexOp);
  INSERT_ATENOP_PATTERN(AtenDiagonalOp);
  INSERT_ATENOP_PATTERN(PrimsCollapseOp);
  INSERT_ATENOP_PATTERN(PrimsSplitDimOp);
#undef INSERT_ATENOP_PATTERN

#define INSERT_VIEW_OP_PATTERN(AtenOp)                                         \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenViewOp<AtenOp>>(typeConverter, context, options)
  INSERT_VIEW_OP_PATTERN(AtenViewDtypeOp);
  INSERT_VIEW_OP_PATTERN(AtenViewOp);
  INSERT_VIEW_OP_PATTERN(AtenReshapeOp);
#undef INSERT_VIEW_OP_PATTERN
}
