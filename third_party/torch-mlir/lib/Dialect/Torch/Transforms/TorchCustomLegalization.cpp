//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include <cstdint>
#include <set>
using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
namespace mlir::torch::Torch {

#define GEN_PASS_DEF_TORCHCUSTOMLEGALIZATION
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h.inc"

template <typename AtenIndexPutLikeOpT>
class LegalizeAtenIndexPutLikeOp
    : public OpRewritePattern<AtenIndexPutLikeOpT> {
public:
  using OpRewritePattern<AtenIndexPutLikeOpT>::OpRewritePattern;

  // given an index_put op for which
  // - the indices are a boolean mask
  // - the final index count is known at compile time
  // -> transform the index_put op into (effectively) a nonzero op with static
  // shape
  //    (this avoids creating an actual nonzero op which has dynamic shape)
  Value decomposeStaticNonzeroOp(AtenIndexPutLikeOpT op, Value input,
                                 int64_t valuesSize,
                                 PatternRewriter &rewriter) const {

    // shared types and constants
    Location loc = op.getLoc();
    BaseTensorType inputType = dyn_cast<BaseTensorType>(input.getType());
    int64_t inputRank = inputType.getSizes().size();
    Value inputRankConst =
        AtenDimOp::create(rewriter, loc, rewriter.getType<IntType>(), input);
    Value inputNumElemsConst = AtenNumelOp::create(rewriter, loc, input);
    Value inputValuesSizeConst = ConstantIntOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(valuesSize));
    Value constantZero =
        ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(0));
    Value constantOne =
        ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));

    // int type used for the values of the result index tensor
    Type intType = rewriter.getIntegerType(32, /*isSigned=*/true);
    SmallVector<int64_t> resultShape({valuesSize, inputRank});
    BaseTensorType resultType = ValueTensorType::get(
        rewriter.getContext(), ArrayRef<int64_t>(resultShape), intType);
    Value intTypeValue = getDtypeIntValueForType(rewriter, loc, intType);
    Type intListTy = rewriter.getType<ListType>(rewriter.getType<IntType>());
    Type listType = rewriter.getType<ListType>(
        rewriter.getType<ValueTensorType>(std::nullopt, nullptr));

    // flatten the first R - 1 dimensions of input boolean mask
    // t_flat = t.flatten() # torch.flatten(t, 0, 0)
    int64_t flattenedSize = 1;
    for (auto size : inputType.getSizes()) {
      flattenedSize *= size;
    }
    auto flattendInputShape = SmallVector<int64_t>{flattenedSize};
    auto flattenedInputType = rewriter.getType<ValueTensorType>(
        flattendInputShape, inputType.getOptionalDtype());
    // %1 = torch.aten.flatten.using_ints %arg0, %int0, %int0_0 :
    auto inputDimsEnd = ConstantIntOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(inputRank - 1));
    // nonzero_mask = (t_flat != 0)
    auto boolMaskType = inputType.getWithSizesAndDtype(
        flattenedInputType.getOptionalSizes(), rewriter.getI1Type());
    Value boolMask = AtenFlattenUsingIntsOp::create(
        rewriter, loc, boolMaskType, input, constantZero /*inputDimsStart*/,
        inputDimsEnd /*inputDimsEnd*/);

    // convert boolean mask to indices int type
    // nonzero_mask = nonzero_mask.int()
    Value falseCst = ConstantBoolOp::create(rewriter, loc, false);
    Value noneCst = ConstantNoneOp::create(rewriter, loc);
    auto intMaskType = flattenedInputType.getWithSizesAndDtype(
        flattenedInputType.getOptionalSizes(), intType);
    Value intMask =
        AtenToDtypeOp::create(rewriter, loc, intMaskType, boolMask,
                              intTypeValue, falseCst, falseCst, noneCst);

    // construct the "destination indices" of each element in the boolean mask:
    // - each consecutive run of F values and the preceding T (if any) are all
    //   mapped to the same index
    // - the subsequent run is mapped to the next consecutive index
    // - the only exception is the initial run of F - these are grouped with the
    //   successor T
    // e.g. [F, F, T, F, F, T, F, F, T]
    // ->   [0, 0, 0, 0, 0, 1, 1, 1, 2]
    // destination_indices = torch.cumsum(nonzero_mask, 0) - 1
    Value cumulativeSum = AtenCumsumOp::create(rewriter, loc, intMaskType,
                                               intMask, constantZero, noneCst);
    Value subtracted =
        AtenSubScalarOp::create(rewriter, loc, intMaskType, cumulativeSum,
                                constantOne, /*alpha=*/constantOne);
    // destination_indices = torch.clamp(destination_indices, min=0)
    Value indices = AtenClampMinOp::create(rewriter, loc, intMaskType,
                                           subtracted, constantZero);

    // construct a tensor that effectively maps each element of the original
    // (non-flattened) input to its index
    // - the final tensor has rank R + 1 and shape (input.shape || R)
    //  (i.e., effectively the same shape as the input except each element
    //  is an index of length R)
    // - this is constructed by effectively concatenating the iota tensors on
    // dimension R
    //  iota(0) || iota(1) || ... || iota(R - 1)
    //  where each iota(i) tensor has shape (input.shape || 1)
    //  and increments values along dimension i

    // utility to transform tensor to desired shape using a reshape + broadcast
    auto reshapeAndBroadcast = [&loc, &rewriter, &resultType,
                                &intType](Value input,
                                          SmallVector<int64_t> &reshapeDims,
                                          Value &reshapeDimValue,
                                          SmallVector<int64_t> &broadcastDims,
                                          Value &broadcastDimValue) {
      assert(reshapeDims.size() == broadcastDims.size() &&
             "intermediate reshape type must have same type as final "
             "broadcast results");
      Type reshapeTy = resultType.getWithSizesAndDtype(
          ArrayRef<int64_t>(reshapeDims), intType);
      Value reshape = AtenReshapeOp::create(rewriter, loc, reshapeTy, input,
                                            reshapeDimValue);
      Type broadcastTy = resultType.getWithSizesAndDtype(
          ArrayRef<int64_t>(broadcastDims), intType);
      Value broadcast = AtenBroadcastToOp::create(rewriter, loc, broadcastTy,
                                                  reshape, broadcastDimValue);
      return broadcast;
    };

    // iota = torch.arange(len(t_flat)) * nonzero_mask
    SmallVector<Value> iotaValues;

    // partial index (pre-concatenation) - shape of the initial iota tensors
    // before concatenation i.e. (input.shape || 1)
    SmallVector<int64_t> partialIndexDimsShape(inputType.getSizes().begin(),
                                               inputType.getSizes().end());
    partialIndexDimsShape.emplace_back(1);
    SmallVector<Value> partialIndexDims;
    for (int64_t i = 0; i < inputRank; i++) {
      Value dimConstant =
          ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(i));
      Value dimSize = AtenSizeIntOp::create(rewriter, loc, input, dimConstant);
      partialIndexDims.emplace_back(dimSize);
    }
    partialIndexDims.emplace_back(constantOne);
    Value partialIndexDimsValue =
        PrimListConstructOp::create(rewriter, loc, intListTy, partialIndexDims);

    for (int64_t dim = 0; dim < inputRank; dim++) {
      // dimSize = inputType.getSizes()[dim] (written as shape[d] below)
      // note:
      // (i) dimSize is the compile-time dimension size (may be unknown)
      // (ii) dimSizeValue is the run-time dimension size
      int64_t dimSize = inputType.getSizes()[dim];
      Value dimConstant =
          ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(dim));
      Value dimSizeValue =
          AtenSizeIntOp::create(rewriter, loc, input, /*dim=*/dimConstant);

      // oneDimRange = arange(0, shape[d]) \in (shape[d])
      // [0, 1, ..., shape[d] - 1]
      SmallVector<int64_t> oneDimRangeShape({dimSize});
      Type oneDimRangeTy = resultType.getWithSizesAndDtype(
          ArrayRef<int64_t>(oneDimRangeShape), intType);
      Value oneDimRange = AtenArangeStartStepOp::create(
          rewriter, loc, oneDimRangeTy, /*start*/ constantZero,
          /*end*/ dimSizeValue,
          /*step*/ constantOne, noneCst, noneCst, noneCst, noneCst);

      // reshape = reshape(oneDimRange, shape) \in (1, ..., shape[d], ..., 1)
      //   where shape has rank = inputRank + 1 and
      //   (i) i != dim -> shape[i] = 1
      //   (ii) i == dim -> shape[i] = dimSize
      // e.g. rank=2, dim=0
      //   [[[0]], [[1]], [[dimShapeSize - 1]]]:
      SmallVector<int64_t> reshapeDimsShape(inputRank + 1, 1);
      reshapeDimsShape[dim] = dimSize;
      SmallVector<Value> reshapeDims(inputRank + 1, constantOne);
      reshapeDims[dim] = dimSizeValue;
      Value reshapeDimsValue =
          PrimListConstructOp::create(rewriter, loc, intListTy, reshapeDims);
      Value broadcastIota =
          reshapeAndBroadcast(oneDimRange, reshapeDimsShape, reshapeDimsValue,
                              partialIndexDimsShape, partialIndexDimsValue);
      iotaValues.emplace_back(broadcastIota);
    }
    // concat = cat([iota_0, ..., iota_d-1], inputRank) \in (shape[0], ...,
    // shape[inputRank - 1], inputRank) each element of concat is now exactly
    // its index:
    //  [[ [0, 0], [0, 1], ..., [0, dim[0] - 1] ],
    //   ...
    //   [ [dim[1] - 1, 0], [dim[1] - 1, 1], ..., [dim[1] - 1, dim[0] - 1] ] ]
    Value catListValue =
        PrimListConstructOp::create(rewriter, loc, listType, iotaValues);
    SmallVector<int64_t> fullIndexShape(inputType.getSizes().begin(),
                                        inputType.getSizes().end());
    fullIndexShape.emplace_back(inputRank);
    Type catTy = resultType.getWithSizesAndDtype(
        ArrayRef<int64_t>(fullIndexShape), intType);
    Value concat =
        AtenCatOp::create(rewriter, loc, catTy, catListValue, inputRankConst);

    // flattened reshape/broadcast sizes
    SmallVector<int64_t> flattenedReshapeSizeShape({flattenedSize, 1});
    SmallVector<Value> flattenedReshapeDims({inputNumElemsConst, constantOne});
    Value flattenedReshapeSizeValue = PrimListConstructOp::create(
        rewriter, loc, intListTy, flattenedReshapeDims);
    SmallVector<int64_t> flattenedBroadcastSizeShape(
        {flattenedSize, inputRank});
    SmallVector<Value> flattenedBroadcastDims(
        {inputNumElemsConst, inputRankConst});
    Value flattenedBroadcastSizeValue = PrimListConstructOp::create(
        rewriter, loc, intListTy, flattenedBroadcastDims);
    Type flattenedIndexTy = resultType.getWithSizesAndDtype(
        ArrayRef<int64_t>(flattenedBroadcastSizeShape), intType);

    // flatten into a list of indices and multiply by the boolean mask
    // [ [0, 0], [0, 1], ..., [dim[1] - 1, dim[0] - 1] ]
    // flattenConcat = flatten(concat, 0, inputRank - 1) \in (shape[0] * ... *
    // shape[inputRank - 1], inputRank)
    Value flattenConcat = AtenFlattenUsingIntsOp::create(
        rewriter, loc, flattenedIndexTy, concat,
        /*inputDimsStart=*/constantZero, /*inputDimsEnd=*/inputDimsEnd);
    Value broadcastMask = reshapeAndBroadcast(
        intMask, flattenedReshapeSizeShape, flattenedReshapeSizeValue,
        flattenedBroadcastSizeShape, flattenedBroadcastSizeValue);
    Value maskedIota = AtenMulTensorOp::create(rewriter, loc, flattenedIndexTy,
                                               flattenConcat, broadcastMask);
    // broadcast indices for each dim in rank
    Value broadcastIndices = reshapeAndBroadcast(
        indices, flattenedReshapeSizeShape, flattenedReshapeSizeValue,
        flattenedBroadcastSizeShape, flattenedBroadcastSizeValue);

    // use a scatter op to get the "true" indices:
    // - each index value in the flattened list is directly mapped
    //   and accumulated into the index given by the corresponding
    //   element of the cumulative sum
    // - since all false values are masked to zero and each
    //   unique index in the cumulative sum corresponds to exactly 1
    //   true value -> the sum in the scatter output will be exactly the
    //   true index
    // e.g. indices = [[0, 0], [0, 0], [0, 2], [0, 0], [1, 1], [1, 2]]
    //      cumsum =  [0,      0,      0,      0,      1,      2]
    // ->   output =  [[0, 2], [1, 1], [1, 2], [0, 0], [0, 0], [0, 0]]
    // - finally the slice output[:valuesSize] is returned
    //   (since valuesSize is known at compile time)

    // scatter_self = torch.zeros_like(t, dtype=torch.int64)
    // AtenFullLike doesn't support index type so we have to use int.
    Value zerosTensor = AtenZerosLikeOp::create(
        rewriter, loc, flattenedIndexTy, maskedIota, intTypeValue, noneCst,
        noneCst, noneCst, noneCst);
    // compacted = torch.scatter_add(
    //  scatter_self, dim=0, index=destination_indices_clamp, src=iota)
    Value scatteredTensor = AtenScatterAddOp::create(
        rewriter, loc, flattenedIndexTy, /*self*/ zerosTensor,
        /*dim=*/constantZero,
        /*index=*/broadcastIndices, /*src=*/maskedIota);
    // result_flat = compacted[:torch.sum(nonzero_mask)]
    Value narrowResult = AtenNarrowOp::create(
        rewriter, loc, resultType, /*self=*/scatteredTensor,
        /*dim=*/constantZero,
        /*start=*/constantZero, /*length=*/inputValuesSizeConst);
    return narrowResult;
  }

  // directly decompose the index_put into a select/where op
  // when the values are a splat
  void decomposeSplatValueToWhere(AtenIndexPutLikeOpT op, Value predicate,
                                  PatternRewriter &rewriter) const {
    // op inputs
    Value input = op.getSelf();
    Value values = op.getValues();
    Value accumulate = op.getAccumulate();

    // shared constants
    Location loc = op.getLoc();
    BaseTensorType inputType = cast<BaseTensorType>(input.getType());
    BaseTensorType valuesType = cast<BaseTensorType>(values.getType());
    ArrayRef<int64_t> inputSizes = inputType.getSizes();
    int64_t inputRank = inputSizes.size();
    Value constantOne =
        ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
    Value constantFalse =
        ConstantBoolOp::create(rewriter, loc, rewriter.getBoolAttr(false));

    // directly rewrite as a where op for scalar value
    SmallVector<Value> expandScalarShapeValues;
    for (int64_t dimSize : inputSizes) {
      Value dimConst = ConstantIntOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(dimSize));
      expandScalarShapeValues.emplace_back(dimConst);
    }
    Value expandScalarShapeListValue = PrimListConstructOp::create(
        rewriter, loc, rewriter.getType<ListType>(rewriter.getType<IntType>()),
        expandScalarShapeValues);

    // reshape + expand predicate
    BaseTensorType predicateType = cast<BaseTensorType>(predicate.getType());
    int64_t predicateRank = predicateType.getSizes().size();
    SmallVector<Value> reshapePredicateShapeValues;
    SmallVector<int64_t> reshapePredicateShape;
    for (int64_t i = 0; i < inputRank; i++) {
      if (i < predicateRank) {
        reshapePredicateShapeValues.emplace_back(ConstantIntOp::create(
            rewriter, loc,
            rewriter.getI64IntegerAttr(predicateType.getSizes()[i])));
        reshapePredicateShape.emplace_back(predicateType.getSizes()[i]);
      } else {
        reshapePredicateShapeValues.emplace_back(constantOne);
        reshapePredicateShape.emplace_back(1);
      }
    }
    Value reshapePredicateShapeListValue = PrimListConstructOp::create(
        rewriter, loc, rewriter.getType<ListType>(rewriter.getType<IntType>()),
        reshapePredicateShapeValues);
    Type reshapePredicateType = rewriter.getType<ValueTensorType>(
        reshapePredicateShape, predicateType.getDtype());
    Value reshapePredicate =
        AtenReshapeOp::create(rewriter, loc, reshapePredicateType, predicate,
                              reshapePredicateShapeListValue);
    Type expandPredicateType =
        rewriter.getType<ValueTensorType>(inputSizes, predicateType.getDtype());
    Value expandPredicate = AtenExpandOp::create(
        rewriter, loc, expandPredicateType, reshapePredicate,
        expandScalarShapeListValue, constantFalse);

    // reshape + expand value from scalar tensor
    SmallVector<int64_t> reshapeScalarShape(inputRank, 1);
    SmallVector<Value> reshapeScalarShapeValues(inputRank, constantOne);
    Value reshapeScalarShapeListValue = PrimListConstructOp::create(
        rewriter, loc, rewriter.getType<ListType>(rewriter.getType<IntType>()),
        reshapeScalarShapeValues);
    Type reshapeValueType = rewriter.getType<ValueTensorType>(
        reshapeScalarShape, valuesType.getDtype());
    Value reshapeValue = AtenReshapeOp::create(
        rewriter, loc, reshapeValueType, values, reshapeScalarShapeListValue);
    Type expandValueType = rewriter.getType<ValueTensorType>(
        inputType.getSizes(), valuesType.getDtype());
    Value expandValue =
        AtenExpandOp::create(rewriter, loc, expandValueType, reshapeValue,
                             expandScalarShapeListValue, constantFalse);

    // convert accumulate boolean to integer, then create full tensor
    Value accumulateInt = AtenIntBoolOp::create(rewriter, loc, accumulate);
    Type fullAccumulateType = rewriter.getType<ValueTensorType>(
        inputType.getSizes(), rewriter.getI1Type());
    Value noneCst = ConstantNoneOp::create(rewriter, loc);
    Value fullAccumulate = AtenFullOp::create(
        rewriter, loc, fullAccumulateType, expandScalarShapeListValue,
        accumulateInt, noneCst, noneCst, noneCst, noneCst);

    // where on expanded predicate + input + expanded value
    Value whereOnPredicateInputAloneOp = AtenWhereSelfOp::create(
        rewriter, loc, inputType, expandPredicate, expandValue, input);
    Value inputPlusValues = AtenAddTensorOp::create(
        rewriter, loc, inputType, input, expandValue, constantOne);
    Value whereOnPredicateInputPlusValuesOp = AtenWhereSelfOp::create(
        rewriter, loc, inputType, expandPredicate, inputPlusValues, input);
    auto whereOnAccumulate = AtenWhereSelfOp::create(
        rewriter, loc, inputType, fullAccumulate,
        whereOnPredicateInputPlusValuesOp, whereOnPredicateInputAloneOp);
    rewriter.replaceOp(op, whereOnAccumulate);
  }

  LogicalResult matchAndRewrite(AtenIndexPutLikeOpT op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value> indices;
    if (!getListConstructElements(op.getIndices(), indices)) {
      return rewriter.notifyMatchFailure(
          op, "LegalizeAtenIndexPut: failed to get elements of `indices`");
    }
    if (indices.size() == 0) {
      return failure();
    }

    // validate that either:
    // - all indices are integers
    // - all indices are booleans and there is only one index tensor
    bool anyIntIndex = false;
    bool anyBoolIndex = false;
    for (size_t i = 0; i < indices.size(); i++) {
      Value &v = indices[i];
      if (auto tensorType = dyn_cast<ValueTensorType>(v.getType())) {
        bool isBool = tensorType.getDtype() == rewriter.getI1Type();
        bool isInt = !isBool && isa<IntegerType>(tensorType.getDtype());
        anyIntIndex |= isInt;
        anyBoolIndex |= isBool;
      }
    }
    if (anyIntIndex && anyBoolIndex) {
      return rewriter.notifyMatchFailure(
          op,
          "LegalizeAtenIndexPut: `indices` must all be bool or int tensors");
    }
    if (anyBoolIndex && indices.size() > 1) {
      return rewriter.notifyMatchFailure(op,
                                         "LegalizeAtenIndexPut: `indices` must "
                                         "have length 1 for bool indices");
    }

    if (!anyBoolIndex) {
      return failure();
    }
    Value values = op.getValues();
    BaseTensorType valuesType = cast<BaseTensorType>(values.getType());
    int64_t valuesRank = valuesType.getSizes().size();
    // decompose entire op to aten.where
    if (valuesRank == 0) {
      decomposeSplatValueToWhere(op, indices[0], rewriter);
      return success();
    }
    // replace boolean mask tensor with the corresponding integer indices
    indices[0] = decomposeStaticNonzeroOp(op, indices[0], valuesRank, rewriter);
    Value newIndices = PrimListConstructOp::create(
        rewriter, loc,
        rewriter.getType<ListType>(
            rewriter.getType<ValueTensorType>(std::nullopt, nullptr)),
        indices);
    op.getIndicesMutable().set(newIndices);
    return success();
  }
};

namespace {
class TorchCustomLegalization
    : public impl::TorchCustomLegalizationBase<TorchCustomLegalization> {
private:
  template <typename AtenIndexPutLikeOpT>
  static bool checkHasBoolean(AtenIndexPutLikeOpT op) {
    SmallVector<Value> indices;
    if (!getListConstructElements(op.getIndices(), indices)) {
      return false;
    }
    bool anyBoolIndex = llvm::any_of(indices, [](Value index) {
      if (auto tensorType = dyn_cast<ValueTensorType>(index.getType())) {
        if (auto elementType = dyn_cast<IntegerType>(tensorType.getDtype())) {
          return elementType.getWidth() == 1;
        }
      }
      return false;
    });
    return !anyBoolIndex;
  }

public:
  void runOnOperation() override {
    llvm::CrashRecoveryContext::Enable();
    llvm::CrashRecoveryContext CRC;
    LogicalResult result = failure();

    bool crashed = !CRC.RunSafely([&]() {
      MLIRContext *context = &getContext();
      ConversionTarget target(*context);
      RewritePatternSet patterns(context);

      target.addDynamicallyLegalOp<AtenIndexPutOp>(
          checkHasBoolean<AtenIndexPutOp>);
      patterns.add<LegalizeAtenIndexPutLikeOp<AtenIndexPutOp>>(context);
      target.addDynamicallyLegalOp<Aten_UnsafeIndexPutHackedTwinOp>(
          checkHasBoolean<Aten_UnsafeIndexPutHackedTwinOp>);
      patterns.add<LegalizeAtenIndexPutLikeOp<Aten_UnsafeIndexPutHackedTwinOp>>(
          context);
      target.addDynamicallyLegalOp<Aten_IndexPutImplOp>(
          checkHasBoolean<Aten_IndexPutImplOp>);
      patterns.add<LegalizeAtenIndexPutLikeOp<Aten_IndexPutImplOp>>(context);

      GreedyRewriteConfig config;
      config.setUseTopDownTraversal(true);
      config.setMaxIterations(GreedyRewriteConfig::kNoLimit);
      result =
          applyPatternsGreedily(getOperation(), std::move(patterns), config);
    });

    if (crashed || failed(result)) {
      if (crashed)
        getOperation()->emitError()
            << "TorchCustomLegalizaton pass crashed unexpectedly";
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createTorchCustomLegalizationPass() {
  return std::make_unique<TorchCustomLegalization>();
}

} // namespace mlir::torch::Torch
