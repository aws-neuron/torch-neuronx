//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToNeuron/TorchToNeuron.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch-mlir/Conversion/Passes.h"
#include "torch-mlir/Conversion/TorchToStablehlo/StablehloLegalizeUtils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"
#include <regex>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
namespace mlir::torch {

#define GEN_PASS_DEF_CONVERTTORCHTONEURON
#include "torch-mlir/Conversion/Passes.h.inc"

namespace {
class ConvertNeuronOperatorOp : public OpConversionPattern<Torch::OperatorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename Torch::OperatorOp::Adaptor;

private:
  // Counter for generating unique channel IDs for collective operations.
  // Channel ID > 0 combined with use_global_device_ids enables flattenedIds
  // mode in StableHLO, which interprets replica group entries as global device
  // IDs rather than replica IDs. This is required for sub-group collectives
  // (e.g., TP groups) where global ranks like [40, 41, 42, 43] are used.
  mutable int64_t nextChannelId = 1;

  // Helper to create a channel handle for collective operations that use
  // global device IDs in replica groups.
  stablehlo::ChannelHandleAttr
  createChannelHandle(ConversionPatternRewriter &rewriter) const {
    return stablehlo::ChannelHandleAttr::get(rewriter.getContext(),
                                             nextChannelId++,
                                             /*type=*/1);
  }

  // Helper to create replica groups for collective operations
  // Returns {replicaGroupsAttr, groupSize}
  std::pair<DenseIntElementsAttr, int64_t>
  createReplicaGroups(ConversionPatternRewriter &rewriter, Value group) const {
    SmallVector<SmallVector<int64_t>> replicaIds;

    if (auto constOp = group.getDefiningOp<Torch::ConstantStrOp>()) {
      std::string groupStr = constOp.getValue().str();
      std::regex row_regex(R"(\[([\d,\s]+)\])");
      std::regex num_regex(R"(\d+)");

      auto rows_begin =
          std::sregex_iterator(groupStr.begin(), groupStr.end(), row_regex);
      auto rows_end = std::sregex_iterator();

      for (auto i = rows_begin; i != rows_end; ++i) {
        std::string row_str = (*i)[1].str();
        SmallVector<int64_t> row;

        auto nums_begin =
            std::sregex_iterator(row_str.begin(), row_str.end(), num_regex);
        auto nums_end = std::sregex_iterator();

        for (auto j = nums_begin; j != nums_end; ++j) {
          row.push_back(std::stoi(j->str()));
        }
        if (!row.empty()) {
          replicaIds.push_back(row);
        }
      }
    }

    // Fallback to single group with world size
    if (replicaIds.empty()) {
      int64_t worldSize = 1;
      if (const char *envWorldSize = std::getenv("WORLD_SIZE")) {
        worldSize = std::strtol(envWorldSize, nullptr, 10);
      }
      SmallVector<int64_t> defaultGroup;
      for (int64_t i = 0; i < worldSize; ++i) {
        defaultGroup.push_back(i);
      }
      replicaIds.push_back(defaultGroup);
    }

    int64_t groupSize = replicaIds[0].size();

    SmallVector<int64_t> flatIds;
    for (const auto &group : replicaIds) {
      for (int64_t id : group) {
        flatIds.push_back(id);
      }
    }

    auto attr = DenseIntElementsAttr::get(
        RankedTensorType::get(
            {static_cast<int64_t>(replicaIds.size()), groupSize},
            rewriter.getI64Type()),
        flatIds);
    return {attr, groupSize};
  }

  // Helper to create reduction computation block
  template <typename OpType>
  LogicalResult createReductionComputation(OpType &op,
                                           const std::string &reduceOp,
                                           RankedTensorType inputType,
                                           ConversionPatternRewriter &rewriter,
                                           Location loc) const {
    Block *block = rewriter.createBlock(&op.getComputation());
    auto scalarType = RankedTensorType::get({}, inputType.getElementType());
    block->addArgument(scalarType, loc);
    block->addArgument(scalarType, loc);

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(block);
    Value result;
    if (reduceOp == "sum" || reduceOp == "avg") {
      result = stablehlo::AddOp::create(rewriter, loc, block->getArgument(0),
                                        block->getArgument(1));
    } else if (reduceOp == "product" || reduceOp == "prod") {
      result = stablehlo::MulOp::create(rewriter, loc, block->getArgument(0),
                                        block->getArgument(1));
    } else if (reduceOp == "min") {
      result = stablehlo::MinOp::create(rewriter, loc, block->getArgument(0),
                                        block->getArgument(1));
    } else if (reduceOp == "max") {
      result = stablehlo::MaxOp::create(rewriter, loc, block->getArgument(0),
                                        block->getArgument(1));
    } else {
      return failure();
    }

    stablehlo::ReturnOp::create(rewriter, loc, result);
    return success();
  }

  // Helper to handle average division
  Value handleAverageReduction(Value result, const std::string &reduceOp,
                               int64_t groupSize, RankedTensorType outType,
                               ConversionPatternRewriter &rewriter,
                               Location loc) const {
    if (reduceOp == "avg" && groupSize > 1) {
      auto elementType = outType.getElementType();
      Attribute divisorAttr;
      if (isa<mlir::FloatType>(elementType)) {
        divisorAttr =
            rewriter.getFloatAttr(elementType, static_cast<double>(groupSize));
      } else {
        divisorAttr = rewriter.getIntegerAttr(elementType, groupSize);
      }
      Value divisor = stablehlo::ConstantOp::create(
          rewriter, loc, DenseElementsAttr::get(outType, divisorAttr));
      return stablehlo::DivOp::create(rewriter, loc, result, divisor);
    }
    return result;
  }

  // Helper to create output operand aliases
  LogicalResult
  createOutputOperandAliases(ArrayAttr outputAliasing, bool isMultiResult,
                             ConversionPatternRewriter &rewriter,
                             Torch::OperatorOp op,
                             ArrayAttr &outputOperandAliases) const {
    if (!outputAliasing) {
      outputOperandAliases = nullptr;
      return success();
    }

    SmallVector<Attribute> aliases;
    for (auto aliasAttr : outputAliasing) {
      auto tupleAttr = dyn_cast<ArrayAttr>(aliasAttr);
      if (!tupleAttr || tupleAttr.size() != 2) {
        return op->emitError("output_operand_aliases tuple must have exactly 2 "
                             "elements [operandIndex, outputIndex]");
      }

      auto operandIndexAttr = dyn_cast<IntegerAttr>(tupleAttr[0]);
      if (!operandIndexAttr) {
        return op->emitError(
            "output_operand_aliases tuple elements must be integers");
      }

      int64_t operandIndex = operandIndexAttr.getInt();

      if (isMultiResult) {
        auto outputIndexAttr = dyn_cast<IntegerAttr>(tupleAttr[1]);
        if (!outputIndexAttr) {
          return op->emitError(
              "output_operand_aliases tuple elements must be integers");
        }
        int64_t outputIndex = outputIndexAttr.getInt();

        auto alias = stablehlo::OutputOperandAliasAttr::get(
            rewriter.getContext(), rewriter.getDenseI64ArrayAttr({outputIndex}),
            operandIndex, rewriter.getDenseI64ArrayAttr({}));
        aliases.push_back(alias);
      } else {
        auto alias = stablehlo::OutputOperandAliasAttr::get(
            rewriter.getContext(), rewriter.getDenseI64ArrayAttr({}),
            operandIndex, rewriter.getDenseI64ArrayAttr({}));
        aliases.push_back(alias);
      }
    }
    outputOperandAliases = rewriter.getArrayAttr(aliases);
    return success();
  }

public:
  LogicalResult
  matchAndRewrite(Torch::OperatorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    std::string opName = op.getName().str();

    if (opName == "stablehlo.custom_call") {
      return CustomCallConversion(op, adaptor, rewriter);
    } else if (opName == "torch.neuron.rand") {
      return transformNeuronRand(op, adaptor, rewriter);
    } else if (opName == "torch._c10d_functional.all_reduce") {
      return transformAllReduce(op, adaptor, rewriter);
    } else if (opName == "torch._c10d_functional.all_reduce_coalesced") {
      return transformAllReduceCoalesced(op, adaptor, rewriter);
    } else if (opName == "torch._c10d_functional.all_gather_into_tensor") {
      return transformAllGather(op, adaptor, rewriter);
    } else if (opName ==
               "torch._c10d_functional.all_gather_into_tensor_coalesced") {
      return transformAllGatherCoalesced(op, adaptor, rewriter);
    } else if (opName == "torch._c10d_functional.reduce_scatter_tensor") {
      return transformReduceScatter(op, adaptor, rewriter);
    } else if (opName ==
               "torch._c10d_functional.reduce_scatter_tensor_coalesced") {
      return transformReduceScatterCoalesced(op, adaptor, rewriter);
    } else if (opName == "torch._c10d_functional.wait_tensor") {
      return transformWaitTensor(op, adaptor, rewriter);
    } else if (opName == "torch._c10d_functional.all_to_all_single") {
      return transformAllToAllSingle(op, adaptor, rewriter);
    } else if (opName == "torch._c10d_functional.broadcast") {
      return transformCollectiveBroadcast(op, adaptor, rewriter);
    } else if (opName == "torch._dtensor.shard_dim_alltoall") {
      return transformShardDimAllToAll(op, adaptor, rewriter);
    } else if (opName == "torch.neuron.fused_quantize_mx_scaled_mm") {
      return transformFusedQuantizeMxScaledMm(op, adaptor, rewriter);
    } else if (opName == "torch.neuron._scaled_mm") {
      return transformScaledMm(op, adaptor, rewriter);
    }

    return failure();
  }
  LogicalResult
  CustomCallConversion(Torch::OperatorOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    const TypeConverter *typeConverter = getTypeConverter();
    auto callTargetName = op->getAttrOfType<StringAttr>("call_target_name");
    auto backendConfig = op->getAttrOfType<StringAttr>("backend_config");
    ArrayAttr outputAliasing =
        op->getAttrOfType<ArrayAttr>("operand_output_aliases");

    if (!callTargetName)
      return op->emitError(
          "stablehlo.custom_call must have a call_target_name attribute");
    if (!backendConfig)
      backendConfig = rewriter.getStringAttr("");

    auto callTarget = callTargetName.str();

    // Check if the op is marked as having side effects (e.g. RNG kernels)
    bool hasSideEffect = false;
    if (auto sideEffectAttr = op->getAttrOfType<BoolAttr>("has_side_effect"))
      hasSideEffect = sideEffectAttr.getValue();

    SmallVector<Value> operands;
    SmallVector<Value> newOperands;

    for (unsigned int i = 0; i < op->getNumOperands(); i++) {
      operands.push_back(adaptor.getOperands()[i]);
    }
    for (Value input : operands) {
      RankedTensorType inputType = cast<RankedTensorType>(input.getType());
      if (inputType) {
        Type inElemTy = inputType.getElementType();
        Value newOperand =
            hlo::promoteType(rewriter, op.getLoc(), input, inElemTy);
        newOperands.push_back(newOperand);
      } else {
        //  Non-tensor arguments had already been included to the kernel while
        //  tracing
        // the backend_config. Hence, the kernel only takes tensor arguments.
        continue;
      }
    }
    // Create output operand aliases if needed
    ArrayAttr outputOperandAliases = nullptr;

    if (op->getNumResults() == 1) {
      auto resultTy = cast<TensorType>(
          typeConverter->convertType(op->getResult(0).getType()));
      if (failed(createOutputOperandAliases(outputAliasing, false, rewriter, op,
                                            outputOperandAliases))) {
        return failure();
      }
      auto newop = stablehlo::CustomCallOp::create(
          rewriter, loc, resultTy, newOperands, callTargetName, hasSideEffect,
          backendConfig, stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
          nullptr,             // called_computations
          nullptr,             // operand_layouts
          nullptr,             // result_layouts
          outputOperandAliases // use the created aliases
      );

      rewriter.replaceOp(op, newop);
      return success();
    } else if (op->getNumResults() > 1) {
      // Handle multi-result ops
      SmallVector<Type> resultTypes;
      for (auto result : op->getResults()) {
        auto convertedType = typeConverter->convertType(result.getType());
        if (!convertedType) {
          return op->emitError("Failed to convert result type for multi-result "
                               "stablehlo.custom_call");
        }
        resultTypes.push_back(cast<TensorType>(convertedType));
      }
      if (failed(createOutputOperandAliases(outputAliasing, true, rewriter, op,
                                            outputOperandAliases))) {
        return failure();
      }
      auto newop = stablehlo::CustomCallOp::create(
          rewriter, loc,
          resultTypes, // SmallVector<Type> -> TypeRange (implicit conversion)
          newOperands, callTargetName,
          hasSideEffect, // has_side_effect
          backendConfig, stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
          nullptr,             // called_computations
          nullptr,             // operand_layouts
          nullptr,             // result_layouts
          outputOperandAliases // output_operand_aliases
      );
      rewriter.replaceOp(op, newop);
      return success();
    } else {
      // TODO: is there a case where we need non-returning custom_call??
      return op->emitError("stablehlo.custom_call must return a value");
    }

    return failure();
  }

  LogicalResult transformAllReduce(Torch::OperatorOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
    ValueRange operands = adaptor.getOperands();
    if (operands.size() != 3)
      return rewriter.notifyMatchFailure(op, "expected 3 operands");

    Value input = operands[0];
    Value reduceOpValue = operands[1];
    Value group = operands[2];

    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "input must be ranked tensor");

    // Extract reduce operation (sum is default)
    std::string reduceOp = "";
    if (auto constOp = reduceOpValue.getDefiningOp<Torch::ConstantStrOp>()) {
      reduceOp = constOp.getValue().str();
    }

    auto replicaGroupsAttr = createReplicaGroups(rewriter, group).first;
    auto outType =
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType(0)));

    auto allReduceOp = stablehlo::AllReduceOp::create(
        rewriter, op.getLoc(), outType, input, replicaGroupsAttr,
        createChannelHandle(rewriter),
        /*use_global_device_ids=*/rewriter.getUnitAttr());

    if (failed(createReductionComputation(allReduceOp, reduceOp, inputType,
                                          rewriter, op.getLoc()))) {
      return rewriter.notifyMatchFailure(
          op, "reduce op must be one of the following: sum, avg, product/prod, "
              "min, or max");
    }

    rewriter.setInsertionPointAfter(allReduceOp);
    Value finalResult = allReduceOp.getResult(0);

    int64_t groupSize = replicaGroupsAttr.getType().getShape()[1];
    finalResult = handleAverageReduction(finalResult, reduceOp, groupSize,
                                         outType, rewriter, op.getLoc());

    rewriter.replaceOp(op, finalResult);
    return success();
  }

  LogicalResult transformNeuronRand(Torch::OperatorOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    auto outType =
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType(0)));

    // Extract seed from operands
    ValueRange operands = adaptor.getOperands();
    if (operands.size() < 2)
      return rewriter.notifyMatchFailure(op, "expected at least 2 operands");

    Value seed = operands[1];

    // Create initial state tensor [seed, 0] for rng_bit_generator
    auto ui64Type = rewriter.getIntegerType(64, /*isSigned=*/false);
    auto stateType = RankedTensorType::get({2}, ui64Type);

    // Convert seed to u64 and reshape to [1]
    auto scalarType = RankedTensorType::get({}, ui64Type);
    auto seedU64 =
        stablehlo::ConvertOp::create(rewriter, loc, scalarType, seed);
    auto seedReshaped = stablehlo::ReshapeOp::create(
        rewriter, loc, RankedTensorType::get({1}, ui64Type), seedU64);

    // Create zero and reshape to [1]
    auto zero = stablehlo::ConstantOp::create(
        rewriter, loc,
        DenseElementsAttr::get(scalarType,
                               rewriter.getIntegerAttr(ui64Type, 0)));
    auto zeroReshaped = stablehlo::ReshapeOp::create(
        rewriter, loc, RankedTensorType::get({1}, ui64Type), zero);

    // Concatenate to create [seed, 0]
    auto initialState = stablehlo::ConcatenateOp::create(
        rewriter, loc, ValueRange{seedReshaped, zeroReshaped}, 0);

    auto rngAlgorithm = stablehlo::RngAlgorithmAttr::get(
        rewriter.getContext(), stablehlo::RngAlgorithm::DEFAULT);

    // Generate u32 random bits
    auto ui32Type = rewriter.getIntegerType(32, /*isSigned=*/false);
    auto ui32OutputType = RankedTensorType::get(outType.getShape(), ui32Type);
    auto rngOp = stablehlo::RngBitGeneratorOp::create(
        rewriter, loc, stateType, ui32OutputType, rngAlgorithm, initialState);

    Value randomBits = rngOp.getOutput();

    // Shift right by 9 bits to get [0, 2^23)
    auto shift9 = stablehlo::ConstantOp::create(
        rewriter, loc,
        DenseElementsAttr::get(ui32OutputType,
                               rewriter.getIntegerAttr(ui32Type, 9)));
    auto shifted = stablehlo::ShiftRightLogicalOp::create(rewriter, loc,
                                                          randomBits, shift9);

    // Convert to f32
    auto f32Type =
        RankedTensorType::get(outType.getShape(), rewriter.getF32Type());
    auto asFloat =
        stablehlo::ConvertOp::create(rewriter, loc, f32Type, shifted);

    // Multiply by 1.1920929e-07 (1.0 / 2^23) to get [0, 1)
    auto scale = stablehlo::ConstantOp::create(
        rewriter, loc,
        DenseElementsAttr::get(f32Type,
                               rewriter.getF32FloatAttr(1.1920929e-07f)));
    Value result = stablehlo::MulOp::create(rewriter, loc, asFloat, scale);

    // Convert to target type if not f32
    if (outType.getElementType() != rewriter.getF32Type()) {
      result = stablehlo::ConvertOp::create(rewriter, loc, outType, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }

  LogicalResult
  transformAllReduceCoalesced(Torch::OperatorOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
    ValueRange operands = adaptor.getOperands();
    if (operands.size() != 3)
      return rewriter.notifyMatchFailure(op, "expected 3 operands");

    Value inputList = operands[0];
    Value reduceOpValue = operands[1];
    Value group = operands[2];

    SmallVector<Value> inputs;
    if (!getListConstructElements(inputList, inputs)) {
      return rewriter.notifyMatchFailure(
          op, "input should comes from a PrimListConstructOp");
    }

    // Extract reduce operation
    std::string reduceOp = "";
    if (auto constOp = reduceOpValue.getDefiningOp<Torch::ConstantStrOp>()) {
      reduceOp = constOp.getValue().str();
    }

    auto replicaGroupsAttr = createReplicaGroups(rewriter, group).first;

    SmallVector<Type> resultTypes;
    SmallVector<Value> convertedInputs;
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto convertedType = getTypeConverter()->convertType(inputs[i].getType());
      resultTypes.push_back(cast<RankedTensorType>(convertedType));

      Value convertedInput = getTypeConverter()->materializeTargetConversion(
          rewriter, op.getLoc(), convertedType, inputs[i]);
      convertedInputs.push_back(convertedInput);
    }

    auto allReduceOp = stablehlo::AllReduceOp::create(
        rewriter, op.getLoc(), resultTypes, convertedInputs, replicaGroupsAttr,
        createChannelHandle(rewriter),
        /*use_global_device_ids=*/rewriter.getUnitAttr());

    RankedTensorType inputType = cast<RankedTensorType>(resultTypes[0]);
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "input must be ranked tensor");

    if (failed(createReductionComputation(allReduceOp, reduceOp, inputType,
                                          rewriter, op.getLoc()))) {
      return rewriter.notifyMatchFailure(
          op, "reduce op must be one of the following: sum, avg, product/prod, "
              "min, or max");
    }

    rewriter.setInsertionPointAfter(allReduceOp);
    SmallVector<Value> results;
    int64_t groupSize = replicaGroupsAttr.getType().getShape()[1];
    for (size_t i = 0; i < allReduceOp.getResults().size(); ++i) {
      Value result = handleAverageReduction(
          allReduceOp.getResult(i), reduceOp, groupSize,
          cast<RankedTensorType>(resultTypes[i]), rewriter, op.getLoc());
      results.push_back(result);
    }

    // Create tuple with all results
    auto tupleType = rewriter.getTupleType(resultTypes);
    Value tupleResult =
        stablehlo::TupleOp::create(rewriter, op.getLoc(), tupleType, results);

    // Replace each use with get_tuple_element
    for (auto &use : llvm::make_early_inc_range(op->getUses())) {
      Operation *user = use.getOwner();
      if (auto listUnpack = dyn_cast<Torch::PrimListUnpackOp>(user)) {
        // Replace ListUnpack results with get_tuple_element operations
        SmallVector<Value> unpackResults;
        for (size_t i = 0; i < listUnpack.getNumResults(); ++i) {
          Value element = stablehlo::GetTupleElementOp::create(
              rewriter, listUnpack.getLoc(), resultTypes[i], tupleResult, i);
          unpackResults.push_back(element);
        }
        rewriter.replaceOp(listUnpack, unpackResults);
      }
    }

    rewriter.replaceOp(op, tupleResult);
    return success();
  }

  LogicalResult transformAllGather(Torch::OperatorOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
    ValueRange operands = adaptor.getOperands();
    if (operands.size() != 3)
      return rewriter.notifyMatchFailure(op, "expected 3 operands");

    Value input = operands[0];
    Value group = operands[2];

    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "input must be ranked tensor");

    auto replicaGroupsAttr = createReplicaGroups(rewriter, group).first;
    auto outType =
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType(0)));

    // all_gather_into_tensor always uses gather dimension 0
    auto gatherDimAttr = rewriter.getI64IntegerAttr(0);

    auto allGatherOp = stablehlo::AllGatherOp::create(
        rewriter, op.getLoc(), outType, ValueRange{input}, gatherDimAttr,
        replicaGroupsAttr, createChannelHandle(rewriter),
        /*use_global_device_ids=*/rewriter.getUnitAttr());

    rewriter.replaceOp(op, allGatherOp.getResult(0));
    return success();
  }

  LogicalResult
  transformAllGatherCoalesced(Torch::OperatorOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
    ValueRange operands = adaptor.getOperands();
    if (operands.size() != 3)
      return rewriter.notifyMatchFailure(op, "expected 3 operands");

    Value inputList = operands[0];
    Value group = operands[2];

    SmallVector<Value> inputs;
    if (!getListConstructElements(inputList, inputs)) {
      return rewriter.notifyMatchFailure(
          op, "input should come from a PrimListConstructOp");
    }

    auto replicaGroupsAttr = createReplicaGroups(rewriter, group).first;

    Value groupSizeValue = operands[1];
    if (auto castOp =
            groupSizeValue.getDefiningOp<UnrealizedConversionCastOp>()) {
      groupSizeValue = castOp.getInputs()[0];
    }

    int64_t groupSizeVal;
    if (!matchPattern(groupSizeValue,
                      Torch::m_TorchConstantInt(&groupSizeVal))) {
      return rewriter.notifyMatchFailure(
          op, "group size must be a constant integer");
    }

    SmallVector<Type> resultTypes;
    SmallVector<Value> convertedInputs;
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto convertedType = getTypeConverter()->convertType(inputs[i].getType());
      auto inputType = cast<RankedTensorType>(convertedType);

      Value convertedInput = getTypeConverter()->materializeTargetConversion(
          rewriter, op.getLoc(), convertedType, inputs[i]);
      convertedInputs.push_back(convertedInput);

      // Calculate result type with expanded gather dimension
      auto shape = inputType.getShape().vec();
      shape[0] = shape[0] * groupSizeVal;
      auto resultType =
          RankedTensorType::get(shape, inputType.getElementType());
      resultTypes.push_back(resultType);
    }

    auto gatherDimAttr = rewriter.getI64IntegerAttr(0);

    auto allGatherOp = stablehlo::AllGatherOp::create(
        rewriter, op.getLoc(), resultTypes, convertedInputs, gatherDimAttr,
        replicaGroupsAttr, createChannelHandle(rewriter),
        /*use_global_device_ids=*/rewriter.getUnitAttr());

    SmallVector<Value> results = allGatherOp.getResults();

    // Create tuple with all results
    auto tupleType = rewriter.getTupleType(resultTypes);
    Value tupleResult =
        stablehlo::TupleOp::create(rewriter, op.getLoc(), tupleType, results);

    // Replace each use with get_tuple_element
    for (auto &use : llvm::make_early_inc_range(op->getUses())) {
      Operation *user = use.getOwner();
      if (auto listUnpack = dyn_cast<Torch::PrimListUnpackOp>(user)) {
        SmallVector<Value> unpackResults;
        for (size_t i = 0; i < listUnpack.getNumResults(); ++i) {
          Value element = stablehlo::GetTupleElementOp::create(
              rewriter, listUnpack.getLoc(), resultTypes[i], tupleResult, i);
          unpackResults.push_back(element);
        }
        rewriter.replaceOp(listUnpack, unpackResults);
      }
    }

    rewriter.replaceOp(op, tupleResult);
    return success();
  }

  LogicalResult
  transformReduceScatter(Torch::OperatorOp op, OpAdaptor adaptor,
                         ConversionPatternRewriter &rewriter) const {
    ValueRange operands = adaptor.getOperands();
    if (operands.size() != 4)
      return rewriter.notifyMatchFailure(op, "expected 4 operands");

    Value input = operands[0];
    Value reduceOpValue = operands[1];
    Value group = operands[3];

    // TODO: Check group is legal
    if (auto constOp = group.getDefiningOp<Torch::ConstantStrOp>()) {
      ;
    }

    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "input must be ranked tensor");

    std::string reduceOp = "";
    if (auto constOp = reduceOpValue.getDefiningOp<Torch::ConstantStrOp>()) {
      reduceOp = constOp.getValue().str();
    }

    auto scatterDimAttr = rewriter.getI64IntegerAttr(0);
    auto replicaGroupsAttr = createReplicaGroups(rewriter, group).first;
    auto outType =
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType(0)));

    auto reduceScatterOp = stablehlo::ReduceScatterOp::create(
        rewriter, op.getLoc(), outType, input, scatterDimAttr,
        replicaGroupsAttr, createChannelHandle(rewriter),
        /*use_global_device_ids=*/rewriter.getUnitAttr());

    if (failed(createReductionComputation(reduceScatterOp, reduceOp, inputType,
                                          rewriter, op.getLoc()))) {
      return rewriter.notifyMatchFailure(
          op, "reduce op must be one of the following: sum, avg, product/prod, "
              "min, or max");
    }

    rewriter.setInsertionPointAfter(reduceScatterOp);
    Value finalResult = reduceScatterOp.getResult();

    int64_t groupSize = replicaGroupsAttr.getType().getShape()[1];
    finalResult = handleAverageReduction(finalResult, reduceOp, groupSize,
                                         outType, rewriter, op.getLoc());

    rewriter.replaceOp(op, finalResult);
    return success();
  }

  LogicalResult
  transformReduceScatterCoalesced(Torch::OperatorOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
    ValueRange operands = adaptor.getOperands();
    if (operands.size() != 4)
      return rewriter.notifyMatchFailure(op, "expected 4 operands");

    Value inputList = operands[0];
    Value reduceOpValue = operands[1];
    Value group = operands[3];

    SmallVector<Value> inputs;
    if (!getListConstructElements(inputList, inputs)) {
      return rewriter.notifyMatchFailure(
          op, "input should come from a PrimListConstructOp");
    }

    std::string reduceOp = "";
    if (auto constOp = reduceOpValue.getDefiningOp<Torch::ConstantStrOp>()) {
      reduceOp = constOp.getValue().str();
    }

    auto replicaGroupsAttr = createReplicaGroups(rewriter, group).first;

    Value groupSizeValue = operands[2];
    if (auto castOp =
            groupSizeValue.getDefiningOp<UnrealizedConversionCastOp>()) {
      groupSizeValue = castOp.getInputs()[0];
    }

    int64_t groupSizeVal;
    if (!matchPattern(groupSizeValue,
                      Torch::m_TorchConstantInt(&groupSizeVal))) {
      return rewriter.notifyMatchFailure(
          op, "group size must be a constant integer");
    }

    SmallVector<Type> resultTypes;
    SmallVector<Value> convertedInputs;
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto convertedType = getTypeConverter()->convertType(inputs[i].getType());
      auto inputType = cast<RankedTensorType>(convertedType);

      Value convertedInput = getTypeConverter()->materializeTargetConversion(
          rewriter, op.getLoc(), convertedType, inputs[i]);
      convertedInputs.push_back(convertedInput);

      // Calculate result type with reduced scatter dimension
      auto shape = inputType.getShape().vec();
      shape[0] = shape[0] / groupSizeVal;
      auto resultType =
          RankedTensorType::get(shape, inputType.getElementType());
      resultTypes.push_back(resultType);
    }

    auto scatterDimAttr = rewriter.getI64IntegerAttr(0);
    int64_t groupSize = replicaGroupsAttr.getType().getShape()[1];

    SmallVector<Value> results;
    for (size_t i = 0; i < convertedInputs.size(); ++i) {
      auto reduceScatterOp = stablehlo::ReduceScatterOp::create(
          rewriter, op.getLoc(), resultTypes[i], convertedInputs[i],
          scatterDimAttr, replicaGroupsAttr, createChannelHandle(rewriter),
          /*use_global_device_ids=*/rewriter.getUnitAttr());

      RankedTensorType inputType =
          cast<RankedTensorType>(convertedInputs[i].getType());
      if (failed(createReductionComputation(
              reduceScatterOp, reduceOp, inputType, rewriter, op.getLoc()))) {
        return rewriter.notifyMatchFailure(
            op, "reduce op must be one of the following: sum, avg, "
                "product/prod, min, or max");
      }

      rewriter.setInsertionPointAfter(reduceScatterOp);
      Value result = handleAverageReduction(
          reduceScatterOp.getResult(), reduceOp, groupSize,
          cast<RankedTensorType>(resultTypes[i]), rewriter, op.getLoc());
      results.push_back(result);
    }

    // Create tuple with all results
    auto tupleType = rewriter.getTupleType(resultTypes);
    Value tupleResult =
        stablehlo::TupleOp::create(rewriter, op.getLoc(), tupleType, results);

    // Replace each use with get_tuple_element
    for (auto &use : llvm::make_early_inc_range(op->getUses())) {
      Operation *user = use.getOwner();
      if (auto listUnpack = dyn_cast<Torch::PrimListUnpackOp>(user)) {
        SmallVector<Value> unpackResults;
        for (size_t i = 0; i < listUnpack.getNumResults(); ++i) {
          Value element = stablehlo::GetTupleElementOp::create(
              rewriter, listUnpack.getLoc(), resultTypes[i], tupleResult, i);
          unpackResults.push_back(element);
        }
        rewriter.replaceOp(listUnpack, unpackResults);
      }
    }

    rewriter.replaceOp(op, tupleResult);
    return success();
  }

  LogicalResult transformWaitTensor(Torch::OperatorOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
    ValueRange operands = adaptor.getOperands();
    if (operands.size() != 1)
      return rewriter.notifyMatchFailure(op, "expected 1 operand");

    // wait_tensor is identity - replace with input
    rewriter.replaceOp(op, operands[0]);
    return success();
  }

  // Lower all_to_all_single to StableHLO operations.
  //
  // PyTorch all_to_all_single(input, output_split_sizes, input_split_sizes,
  // group) semantics:
  // - input_split_sizes[i]: how much data rank i wants from the gathered tensor
  // - output_split_sizes[i]: how much data rank i contributes (unused in this
  // implementation)
  // - Each rank gathers all data, then extracts only what it needs based on
  // input_split_sizes
  //
  // Two lowering strategies:
  // 1. Uniform splits (all elements equal): Use AllToAllOp directly for
  // efficiency
  // 2. Non-uniform splits: Use AllGatherOp + slice operations to handle permute
  // gather patterns
  //    - AllGatherOp collects all tensors: [rank0_data, rank1_data, ...]
  //    - Slice operations extract portions based on input_split_sizes
  //    - Offset calculation: each rank contributes actualRankSize to gathered
  //    tensor
  //    - Example: input_splits=[0,4] means rank 0 wants 0 elements, rank 1
  //    wants 4 elements
  //      from gathered tensor layout [rank0_4_elements, rank1_4_elements]
  LogicalResult
  transformAllToAllSingle(Torch::OperatorOp op, OpAdaptor adaptor,
                          ConversionPatternRewriter &rewriter) const {
    ValueRange operands = adaptor.getOperands();
    if (operands.size() != 4)
      return rewriter.notifyMatchFailure(op, "expected 4 operands");

    Value input = operands[0];
    Value inputSplitSizes = operands[1];
    Value outputSplitSizes = operands[2];
    Value group = operands[3];

    // Handle non-uniform split sizes for permute gather
    SmallVector<int64_t> outputSplits, inputSplits;
    bool hasOutputSplits = false, hasInputSplits = false;

    if (!outputSplitSizes.getDefiningOp<Torch::ConstantNoneOp>()) {
      if (auto listOp =
              outputSplitSizes.getDefiningOp<Torch::PrimListConstructOp>()) {
        for (Value elem : listOp.getElements()) {
          int64_t val;
          if (!matchPattern(elem, Torch::m_TorchConstantInt(&val))) {
            return rewriter.notifyMatchFailure(
                op, "outputSplitSizes must be constants");
          }
          outputSplits.push_back(val);
        }
        hasOutputSplits = true;
      } else {
        return rewriter.notifyMatchFailure(
            op, "outputSplitSizes must be None or list");
      }
    }

    if (!inputSplitSizes.getDefiningOp<Torch::ConstantNoneOp>()) {
      if (auto listOp =
              inputSplitSizes.getDefiningOp<Torch::PrimListConstructOp>()) {
        for (Value elem : listOp.getElements()) {
          int64_t val;
          if (!matchPattern(elem, Torch::m_TorchConstantInt(&val))) {
            return rewriter.notifyMatchFailure(
                op, "inputSplitSizes must be constants");
          }
          inputSplits.push_back(val);
        }
        hasInputSplits = true;
      } else {
        return rewriter.notifyMatchFailure(
            op, "inputSplitSizes must be None or list");
      }
    }

    // TODO: Check group is legal
    if (auto constOp = group.getDefiningOp<Torch::ConstantStrOp>()) {
      ;
    }

    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "input must be ranked tensor");

    auto replicaGroupsAttr = createReplicaGroups(rewriter, group).first;
    auto outType =
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType(0)));

    // all_to_all uses split dimension 0
    uint64_t splitDim = 0;
    uint64_t concatDim = 0;
    int64_t worldSize = 1;
    if (const char *envWorldSize = std::getenv("WORLD_SIZE")) {
      worldSize = std::strtol(envWorldSize, nullptr, 10);
    }

    // Check if splits are uniform
    bool isUniform = true;
    if (hasOutputSplits && outputSplits.size() > 1) {
      int64_t firstVal = outputSplits[0];
      isUniform =
          std::all_of(outputSplits.begin() + 1, outputSplits.end(),
                      [firstVal](int64_t val) { return val == firstVal; });
    }
    if (hasInputSplits && inputSplits.size() > 1) {
      int64_t firstVal = inputSplits[0];
      isUniform =
          isUniform &&
          std::all_of(inputSplits.begin() + 1, inputSplits.end(),
                      [firstVal](int64_t val) { return val == firstVal; });
    }

    Value allToAllResult;
    if ((hasOutputSplits || hasInputSplits) && !isUniform) {
      // Use all_gather + slice for non-uniform splits
      auto gatheredType = RankedTensorType::get(
          {inputType.getDimSize(0) * worldSize}, inputType.getElementType());
      auto allGatherOp = stablehlo::AllGatherOp::create(
          rewriter, op.getLoc(), gatheredType, ValueRange{input},
          /*all_gather_dim=*/rewriter.getI64IntegerAttr(0), replicaGroupsAttr,
          createChannelHandle(rewriter),
          /*use_global_device_ids=*/rewriter.getUnitAttr());

      SmallVector<int64_t> inputSplitSizes =
          hasInputSplits ? inputSplits
                         : SmallVector<int64_t>(
                               worldSize, inputType.getDimSize(0) / worldSize);
      SmallVector<int64_t> outputSplitSizes =
          hasOutputSplits ? outputSplits : inputSplitSizes;

      SmallVector<Value> slices;
      int64_t offset = 0;
      int64_t actualRankSize = inputType.getDimSize(0);
      for (int64_t i = 0; i < worldSize; ++i) {
        int64_t inputSize = inputSplitSizes[i];
        if (inputSize > 0) {
          auto sliceType =
              RankedTensorType::get({inputSize}, inputType.getElementType());
          auto slice = stablehlo::SliceOp::create(
              rewriter, op.getLoc(), sliceType, allGatherOp.getResult(0),
              rewriter.getDenseI64ArrayAttr({offset}),
              rewriter.getDenseI64ArrayAttr({offset + inputSize}),
              rewriter.getDenseI64ArrayAttr({1}));
          slices.push_back(slice.getResult());
        }
        offset += actualRankSize;
      }

      if (slices.empty()) {
        // All input splits are zero - create empty slice
        auto emptyType = RankedTensorType::get({0}, inputType.getElementType());
        allToAllResult =
            stablehlo::SliceOp::create(rewriter, op.getLoc(), emptyType,
                                       allGatherOp.getResult(0),
                                       rewriter.getDenseI64ArrayAttr({0}),
                                       rewriter.getDenseI64ArrayAttr({0}),
                                       rewriter.getDenseI64ArrayAttr({1}))
                .getResult();
      } else {
        allToAllResult =
            stablehlo::ConcatenateOp::create(rewriter, op.getLoc(), slices, 0)
                .getResult();
      }
    } else {
      auto allToAllOp = stablehlo::AllToAllOp::create(
          rewriter, op.getLoc(), outType, input, splitDim, concatDim,
          /*split_count=*/worldSize, replicaGroupsAttr,
          createChannelHandle(rewriter));
      allToAllResult = allToAllOp.getResult(0);
    }

    rewriter.replaceOp(op, allToAllResult);
    return success();
  }

  // Lower _dtensor.shard_dim_alltoall.default to StableHLO AllToAllOp.
  // Args: (input_tensor, gather_dim, shard_dim, group_name)
  LogicalResult
  transformShardDimAllToAll(Torch::OperatorOp op, OpAdaptor adaptor,
                            ConversionPatternRewriter &rewriter) const {
    ValueRange operands = adaptor.getOperands();
    if (operands.size() != 4)
      return rewriter.notifyMatchFailure(op, "expected 4 operands");

    Value input = operands[0];
    Value gatherDimVal = operands[1];
    Value shardDimVal = operands[2];
    Value group = operands[3];

    int64_t gatherDim, shardDim;
    if (!extractConstantInt(gatherDimVal, gatherDim))
      return rewriter.notifyMatchFailure(op, "gather_dim must be a constant");
    if (!extractConstantInt(shardDimVal, shardDim))
      return rewriter.notifyMatchFailure(op, "shard_dim must be a constant");

    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "input must be ranked tensor");

    auto [replicaGroupsAttr, groupSize] = createReplicaGroups(rewriter, group);
    auto outType =
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType(0)));

    auto allToAllOp = stablehlo::AllToAllOp::create(
        rewriter, op.getLoc(), outType, input,
        /*split_dimension=*/static_cast<uint64_t>(shardDim),
        /*concat_dimension=*/static_cast<uint64_t>(gatherDim),
        /*split_count=*/groupSize, replicaGroupsAttr,
        createChannelHandle(rewriter));

    rewriter.replaceOp(op, allToAllOp.getResult(0));
    return success();
  }

  LogicalResult
  transformCollectiveBroadcast(Torch::OperatorOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
    ValueRange operands = adaptor.getOperands();
    if (operands.size() != 3)
      return rewriter.notifyMatchFailure(op, "expected 3 operands");

    Value input = operands[0];
    Value group = operands[2];

    auto inputType = dyn_cast<RankedTensorType>(
        getTypeConverter()->convertType(input.getType()));
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "input must be ranked tensor");

    // TODO: Check group is legal
    if (auto constOp = group.getDefiningOp<Torch::ConstantStrOp>()) {
      ;
    }

    auto replicaGroupsAttr = createReplicaGroups(rewriter, group).first;
    // Get output type from type converter
    auto outType =
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType(0)));

    auto broadcastOp = stablehlo::CollectiveBroadcastOp::create(
        rewriter, op.getLoc(), outType, input, replicaGroupsAttr,
        createChannelHandle(rewriter));

    rewriter.replaceOp(op, broadcastOp.getResult());
    return success();
  }

  // Helper to extract a constant integer from a torch.operator operand,
  // unwrapping UnrealizedConversionCastOp if needed.
  static bool extractConstantInt(Value operand, int64_t &result) {
    if (auto castOp = operand.getDefiningOp<UnrealizedConversionCastOp>()) {
      operand = castOp.getInputs()[0];
    }
    return matchPattern(operand, Torch::m_TorchConstantInt(&result));
  }

  LogicalResult transformScaledMm(Torch::OperatorOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    ValueRange operands = adaptor.getOperands();

    // Expect 8 operands: lhs, lhs_scale, rhs, rhs_scale, bias, scale_result,
    //                    out_dtype, fp8_dtype
    if (operands.size() != 8)
      return rewriter.notifyMatchFailure(op, "expected 8 operands");

    // 1. Validate lhs and rhs (must be packed ui32 ranked tensors)
    Value mat1 = operands[0]; // lhs
    Value mat2 = operands[2]; // rhs
    auto mat1Type = dyn_cast<RankedTensorType>(mat1.getType());
    auto mat2Type = dyn_cast<RankedTensorType>(mat2.getType());
    if (!mat1Type || !mat2Type)
      return rewriter.notifyMatchFailure(op,
                                         "lhs and rhs must be ranked tensors");
    if (!mat1Type.getElementType().isUnsignedInteger(32) ||
        !mat2Type.getElementType().isUnsignedInteger(32))
      return rewriter.notifyMatchFailure(
          op, "lhs and rhs element types must be ui32 (packed MX format)");
    int64_t mat1Rank = mat1Type.getRank();
    int64_t mat2Rank = mat2Type.getRank();
    if (mat1Rank < 2 || mat2Rank < 2)
      return rewriter.notifyMatchFailure(
          op, "lhs and rhs must be at least 2D tensors");

    // 2. Validate lhs_scale and rhs_scale (must be f8E8M0FNU ranked tensors)
    Value scaleA = operands[1]; // lhs_scale
    Value scaleB = operands[3]; // rhs_scale
    auto scaleAType = dyn_cast<RankedTensorType>(scaleA.getType());
    auto scaleBType = dyn_cast<RankedTensorType>(scaleB.getType());
    if (!scaleAType || !scaleBType)
      return rewriter.notifyMatchFailure(
          op, "lhs_scale and rhs_scale must be ranked tensors");
    if (!isa<Float8E8M0FNUType>(scaleAType.getElementType()) ||
        !isa<Float8E8M0FNUType>(scaleBType.getElementType()))
      return rewriter.notifyMatchFailure(
          op, "lhs_scale and rhs_scale element types must be f8E8M0FNU");
    // Validate scale contracting dim = packed contracting dim / 8
    // (block_size=32, pack_size=4 → 32/4 = 8 packed elements per scale block)
    int64_t packedKMat1 = mat1Type.getDimSize(mat1Rank - 1);
    int64_t packedKMat2 = mat2Type.getDimSize(mat2Rank - 1);
    int64_t scaleKA = scaleAType.getDimSize(mat1Rank - 1);
    int64_t scaleKB = scaleBType.getDimSize(mat2Rank - 1);
    if (packedKMat1 != ShapedType::kDynamic && scaleKA != ShapedType::kDynamic)
      if (scaleKA != packedKMat1 / 8)
        return rewriter.notifyMatchFailure(
            op, "lhs_scale contracting dimension size must be 1/8 of lhs "
                "contracting dimension size (block_size=32, pack_size=4)");
    if (packedKMat2 != ShapedType::kDynamic && scaleKB != ShapedType::kDynamic)
      if (scaleKB != packedKMat2 / 8)
        return rewriter.notifyMatchFailure(
            op, "rhs_scale contracting dimension size must be 1/8 of rhs "
                "contracting dimension size (block_size=32, pack_size=4)");

    // 3. Validate bias and scale_result are None (unsupported for now)
    if (!isa<Torch::NoneType>(operands[4].getType()))
      return rewriter.notifyMatchFailure(op, "bias must be None");
    if (!isa<Torch::NoneType>(operands[5].getType()))
      return rewriter.notifyMatchFailure(op, "scale_result must be None");

    // 4. Extract out_dtype (None defaults to bf16)
    std::string outDtypeStr;
    Type outElemType;
    if (isa<Torch::NoneType>(operands[6].getType())) {
      outDtypeStr = "BF16";
      outElemType = BFloat16Type::get(rewriter.getContext());
    } else {
      int64_t outDtypeInt;
      if (!extractConstantInt(operands[6], outDtypeInt))
        return rewriter.notifyMatchFailure(
            op, "out_dtype must be a constant integer or None");
      if (outDtypeInt ==
          static_cast<int64_t>(torch_upstream::ScalarType::BFloat16)) {
        outDtypeStr = "BF16";
        outElemType = BFloat16Type::get(rewriter.getContext());
      } else if (outDtypeInt ==
                 static_cast<int64_t>(torch_upstream::ScalarType::Float)) {
        outDtypeStr = "F32";
        outElemType = Float32Type::get(rewriter.getContext());
      } else if (outDtypeInt ==
                 static_cast<int64_t>(torch_upstream::ScalarType::Half)) {
        outDtypeStr = "F16";
        outElemType = Float16Type::get(rewriter.getContext());
      } else {
        return rewriter.notifyMatchFailure(
            op, "out_dtype must be BFloat16 (15), Float (6), or Half (5)");
      }
    }

    // 5. Extract fp8_dtype (None defaults to float8_e4m3fn)
    std::string fp8DtypeStr;
    if (isa<Torch::NoneType>(operands[7].getType())) {
      fp8DtypeStr = "float8_e4m3fn";
    } else {
      int64_t fp8DtypeInt;
      if (!extractConstantInt(operands[7], fp8DtypeInt))
        return rewriter.notifyMatchFailure(
            op, "fp8_dtype must be a constant integer or None");
      if (fp8DtypeInt ==
          static_cast<int64_t>(torch_upstream::ScalarType::Float8_e5m2)) {
        fp8DtypeStr = "float8_e5m2";
      } else if (fp8DtypeInt ==
                 static_cast<int64_t>(
                     torch_upstream::ScalarType::Float8_e4m3fn)) {
        fp8DtypeStr = "float8_e4m3fn";
      } else {
        return rewriter.notifyMatchFailure(
            op, "fp8_dtype must be Float8_e5m2 (23) or Float8_e4m3fn (24)");
      }
    }

    // 7. Build result shape [..., M, N]
    // mat1: [..., M, K/4] (packed), mat2: [..., N, K/4] (transposed, packed)
    SmallVector<int64_t> resultShape;
    for (int64_t i = 0; i < mat1Rank - 2; ++i)
      resultShape.push_back(mat1Type.getDimSize(i));
    resultShape.push_back(mat1Type.getDimSize(mat1Rank - 2)); // M
    resultShape.push_back(mat2Type.getDimSize(mat2Rank - 2)); // N
    auto resultType = RankedTensorType::get(resultShape, outElemType);

    // 8. Build scaled_dot_backend_config JSON
    auto formatDimList = [](const SmallVector<int64_t> &dims) -> std::string {
      std::string s = "[";
      for (size_t i = 0; i < dims.size(); ++i) {
        if (i > 0)
          s += ", ";
        s += std::to_string(dims[i]);
      }
      s += "]";
      return s;
    };

    SmallVector<int64_t> batchDims;
    for (int64_t i = 0; i < mat1Rank - 2; ++i)
      batchDims.push_back(i);
    int64_t contractDim = mat1Rank - 1;
    SmallVector<int64_t> contractDimVec = {contractDim};

    std::string dotConfigStr =
        "{\"scaled_dot_backend_config\": {"
        "\"lhs_batch_dimensions\": " +
        formatDimList(batchDims) +
        ", \"rhs_batch_dimensions\": " + formatDimList(batchDims) +
        ", \"lhs_contracting_dimensions\": " + formatDimList(contractDimVec) +
        ", \"rhs_contracting_dimensions\": " + formatDimList(contractDimVec) +
        ", \"dequantize_type\": \"" + outDtypeStr +
        "\", \"element_dtype\": \"" + fp8DtypeStr + "\"}}";
    auto dotConfig = rewriter.getStringAttr(dotConfigStr);
    auto dotCallTarget = rewriter.getStringAttr("__op$block_scaled_dot");

    // 9. Emit custom call: (lhs, rhs, lhs_scale, rhs_scale) -> result
    auto customCall = stablehlo::CustomCallOp::create(
        rewriter, loc, TypeRange{resultType},
        ValueRange{mat1, mat2, scaleA, scaleB}, dotCallTarget, false, dotConfig,
        stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL, nullptr, nullptr,
        nullptr, nullptr);

    rewriter.replaceOp(op, customCall.getResult(0));
    return success();
  }

  LogicalResult
  transformFusedQuantizeMxScaledMm(Torch::OperatorOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    ValueRange operands = adaptor.getOperands();

    // Expect 6 operands: lhs, rhs, bias, scale_result, out_dtype, fp8_dtype
    if (operands.size() != 6)
      return rewriter.notifyMatchFailure(op, "expected 6 operands");

    // --- Parse tensor operands (use adaptor for converted tensor values) ---
    Value lhs = operands[0];
    Value rhs = operands[1];

    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    if (!lhsType || !rhsType)
      return rewriter.notifyMatchFailure(op,
                                         "lhs and rhs must be ranked tensors");

    Type lhsElemType = lhsType.getElementType();
    Type rhsElemType = rhsType.getElementType();
    if ((!isa<BFloat16Type>(lhsElemType) && !lhsElemType.isF16()) ||
        (!isa<BFloat16Type>(rhsElemType) && !rhsElemType.isF16()))
      return rewriter.notifyMatchFailure(
          op, "lhs and rhs element types must be bf16 or f16");

    int64_t lhsRank = lhsType.getRank();
    int64_t rhsRank = rhsType.getRank();
    if (lhsRank < 2 || rhsRank < 2)
      return rewriter.notifyMatchFailure(
          op, "lhs and rhs must be at least 2D tensors");

    // --- Validate bias and scale_result are None (unsupported for now) ---
    if (!isa<Torch::NoneType>(operands[2].getType()))
      return rewriter.notifyMatchFailure(op, "bias must be None");
    if (!isa<Torch::NoneType>(operands[3].getType()))
      return rewriter.notifyMatchFailure(op, "scale_result must be None");

    // --- Extract out_dtype ---
    int64_t outDtypeInt;
    if (!extractConstantInt(operands[4], outDtypeInt))
      return rewriter.notifyMatchFailure(
          op, "out_dtype must be a constant integer");

    std::string dequantizeTypeStr;
    Type outElemType;
    if (outDtypeInt ==
        static_cast<int64_t>(torch_upstream::ScalarType::BFloat16)) {
      dequantizeTypeStr = "BF16";
      outElemType = BFloat16Type::get(rewriter.getContext());
    } else if (outDtypeInt ==
               static_cast<int64_t>(torch_upstream::ScalarType::Float)) {
      dequantizeTypeStr = "F32";
      outElemType = Float32Type::get(rewriter.getContext());
    } else if (outDtypeInt ==
               static_cast<int64_t>(torch_upstream::ScalarType::Half)) {
      dequantizeTypeStr = "F16";
      outElemType = Float16Type::get(rewriter.getContext());
    } else {
      return rewriter.notifyMatchFailure(
          op, "out_dtype must be BFloat16 (15), Float (6), or Half (5)");
    }

    // --- Extract fp8_dtype ---
    int64_t fp8DtypeInt;
    if (!extractConstantInt(operands[5], fp8DtypeInt))
      return rewriter.notifyMatchFailure(
          op, "fp8_dtype must be a constant integer");

    std::string fp8DtypeStr;
    if (fp8DtypeInt ==
        static_cast<int64_t>(torch_upstream::ScalarType::Float8_e5m2)) {
      fp8DtypeStr = "float8_e5m2";
    } else if (fp8DtypeInt == static_cast<int64_t>(
                                  torch_upstream::ScalarType::Float8_e4m3fn)) {
      fp8DtypeStr = "float8_e4m3fn";
    } else {
      return rewriter.notifyMatchFailure(
          op, "fp8_dtype must be Float8_e5m2 (23) or Float8_e4m3fn (24)");
    }

    // --- Validate quantization dimensions have static sizes ---
    // Both lhs and rhs quantize along their last dim (K dimension).
    // lhs: [..., M, K], rhs: [..., N, K] (transposed layout)
    int64_t lhsQuantDim = lhsRank - 1;
    int64_t rhsQuantDim = rhsRank - 1;
    int64_t lhsK = lhsType.getDimSize(lhsQuantDim);
    int64_t rhsK = rhsType.getDimSize(rhsQuantDim);
    if (lhsK == ShapedType::kDynamic || rhsK == ShapedType::kDynamic)
      return rewriter.notifyMatchFailure(
          op, "quantization dimensions (K) must have static sizes");

    constexpr int64_t blockSize = 32;

    // --- Common types ---
    auto ui32Type =
        IntegerType::get(rewriter.getContext(), 32, IntegerType::Unsigned);
    auto scaleElemType = Float8E8M0FNUType::get(rewriter.getContext());

    // --- QuantizeMX backend_config (same for both lhs and rhs) ---
    std::string quantConfigStr = "{\"dtype\": \"" + fp8DtypeStr +
                                 "\", \"dim\": -1"
                                 ", \"block_size\": " +
                                 std::to_string(blockSize) +
                                 ", \"scale_method\": \"EMAX\"}";
    auto quantConfig = rewriter.getStringAttr(quantConfigStr);
    auto quantCallTarget = rewriter.getStringAttr("QuantizeMX");

    // --- Create QuantizeMX for lhs ---
    // Input: lhs [..., M, K] → packed [..., M, K/4], scale [..., M, K/32]
    SmallVector<int64_t> lhsPackedShape(lhsType.getShape());
    lhsPackedShape[lhsQuantDim] = lhsK / 4;
    auto lhsPackedType = RankedTensorType::get(lhsPackedShape, ui32Type);

    SmallVector<int64_t> lhsScaleShape(lhsType.getShape());
    lhsScaleShape[lhsQuantDim] = (lhsK + blockSize - 1) / blockSize;
    auto lhsScaleType = RankedTensorType::get(lhsScaleShape, scaleElemType);

    Value convertedLhs = hlo::promoteType(rewriter, loc, lhs, lhsElemType);
    auto lhsQuantOp = stablehlo::CustomCallOp::create(
        rewriter, loc, TypeRange{lhsPackedType, lhsScaleType},
        ValueRange{convertedLhs}, quantCallTarget, false, quantConfig,
        stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL, nullptr, nullptr,
        nullptr, nullptr);

    // --- Create QuantizeMX for rhs ---
    // Input: rhs [..., N, K] → packed [..., N, K/4], scale [..., N, K/32]
    SmallVector<int64_t> rhsPackedShape(rhsType.getShape());
    rhsPackedShape[rhsQuantDim] = rhsK / 4;
    auto rhsPackedType = RankedTensorType::get(rhsPackedShape, ui32Type);

    SmallVector<int64_t> rhsScaleShape(rhsType.getShape());
    rhsScaleShape[rhsQuantDim] = (rhsK + blockSize - 1) / blockSize;
    auto rhsScaleType = RankedTensorType::get(rhsScaleShape, scaleElemType);

    Value convertedRhs = hlo::promoteType(rewriter, loc, rhs, rhsElemType);
    auto rhsQuantOp = stablehlo::CustomCallOp::create(
        rewriter, loc, TypeRange{rhsPackedType, rhsScaleType},
        ValueRange{convertedRhs}, quantCallTarget, false, quantConfig,
        stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL, nullptr, nullptr,
        nullptr, nullptr);

    // --- Create block_scaled_dot ---
    // lhs_packed [..., M, K/4] × rhs_packed [..., N, K/4] → [..., M, N]
    // Both contract on last dim (K/4).
    // Batch dims: [0 .. rank-3], contracting: [rank-1]
    SmallVector<int64_t> batchDims;
    for (int64_t i = 0; i < lhsRank - 2; ++i)
      batchDims.push_back(i);
    int64_t contractDim = lhsRank - 1;

    // Build result shape: [..., M, N]
    SmallVector<int64_t> mmShape;
    for (int64_t i = 0; i < lhsRank - 2; ++i)
      mmShape.push_back(lhsType.getDimSize(i));
    mmShape.push_back(lhsType.getDimSize(lhsRank - 2)); // M
    mmShape.push_back(rhsType.getDimSize(rhsRank - 2)); // N
    auto mmResultType = RankedTensorType::get(mmShape, outElemType);

    // Build block_scaled_dot backend_config JSON
    auto formatDimList = [](const SmallVector<int64_t> &dims) -> std::string {
      std::string s = "[";
      for (size_t i = 0; i < dims.size(); ++i) {
        if (i > 0)
          s += ", ";
        s += std::to_string(dims[i]);
      }
      s += "]";
      return s;
    };

    SmallVector<int64_t> contractDimVec = {contractDim};
    std::string dotConfigStr =
        "{\"scaled_dot_backend_config\": {"
        "\"lhs_batch_dimensions\": " +
        formatDimList(batchDims) +
        ", \"rhs_batch_dimensions\": " + formatDimList(batchDims) +
        ", \"lhs_contracting_dimensions\": " + formatDimList(contractDimVec) +
        ", \"rhs_contracting_dimensions\": " + formatDimList(contractDimVec) +
        ", \"dequantize_type\": \"" + dequantizeTypeStr +
        "\", \"element_dtype\": \"" + fp8DtypeStr + "\"}}";
    auto dotConfig = rewriter.getStringAttr(dotConfigStr);
    auto dotCallTarget = rewriter.getStringAttr("__op$block_scaled_dot");

    // Inputs: lhs_packed, rhs_packed, lhs_scale, rhs_scale
    SmallVector<Value> dotInputs = {
        lhsQuantOp.getResult(0), rhsQuantOp.getResult(0),
        lhsQuantOp.getResult(1), rhsQuantOp.getResult(1)};

    auto dotOp = stablehlo::CustomCallOp::create(
        rewriter, loc, TypeRange{mmResultType}, dotInputs, dotCallTarget, false,
        dotConfig, stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
        nullptr, nullptr, nullptr, nullptr);

    rewriter.replaceOp(op, dotOp.getResult(0));
    return success();
  }
};

class ConvertNeuronTopkOp : public OpConversionPattern<Torch::AtenTopkOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(Torch::AtenTopkOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value input = adaptor.getOperands()[0];
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType) {
      return rewriter.notifyMatchFailure(op, "input must be ranked tensor");
    }

    // Check if args are valid for custom call, only default args are supported
    int64_t kValue;
    if (!matchPattern(op.getK(), Torch::m_TorchConstantInt(&kValue))) {
      return rewriter.notifyMatchFailure(op, "k must be a constant integer");
    }

    bool largest;
    if (!matchPattern(op.getLargest(), Torch::m_TorchConstantBool(&largest))) {
      return rewriter.notifyMatchFailure(op,
                                         "largest must be a constant boolean");
    }

    bool sorted;
    if (!matchPattern(op.getSorted(), Torch::m_TorchConstantBool(&sorted)) ||
        !sorted) {
      return rewriter.notifyMatchFailure(
          op, "only sorted=True is supported for topk");
    }

    int64_t dim = -1;
    if (!matchPattern(op.getDim(), Torch::m_TorchConstantInt(&dim))) {
      if (!isa<Torch::NoneType>(op.getDim().getType())) {
        return rewriter.notifyMatchFailure(
            op, "dim must be None or constant integer");
      }
      dim = -1;
    }

    int64_t inputRank = inputType.getRank();
    if (dim < 0) {
      dim += inputRank;
    }

    // permute dim to last dim for compatibility with custom call
    bool needsPermute = (dim != inputRank - 1);
    SmallVector<int64_t> permutation;
    if (needsPermute) {
      for (int64_t i = 0; i < inputRank; ++i) {
        if (i != dim)
          permutation.push_back(i);
      }
      permutation.push_back(dim);
    }

    // Custom call reads K from the backend config
    std::string backendConfigStr = std::to_string(kValue);
    auto backendConfig = rewriter.getStringAttr(backendConfigStr);

    Value convertedInput =
        hlo::promoteType(rewriter, loc, input, inputType.getElementType());
    auto workingType = cast<RankedTensorType>(convertedInput.getType());

    if (needsPermute) {
      auto permutedType = RankedTensorType::get(
          llvm::to_vector(llvm::map_range(
              permutation,
              [&](int64_t i) { return workingType.getDimSize(i); })),
          workingType.getElementType());
      convertedInput = stablehlo::TransposeOp::create(
          rewriter, loc, permutedType, convertedInput,
          rewriter.getDenseI64ArrayAttr(permutation));
      workingType = permutedType;
    }

    // If largest=False, negate input to simulate smallest-k
    if (!largest) {
      convertedInput = stablehlo::NegOp::create(rewriter, loc, convertedInput);
    }

    SmallVector<Value> convertedOperands = {convertedInput};

    SmallVector<Type> convertedTypes;
    auto valuesShape = workingType.getShape().vec();
    valuesShape[valuesShape.size() - 1] = kValue;
    convertedTypes.push_back(
        RankedTensorType::get(valuesShape, workingType.getElementType()));
    convertedTypes.push_back(
        RankedTensorType::get(valuesShape, rewriter.getI64Type()));

    auto callTargetName = rewriter.getStringAttr("AwsNeuronTopK");

    auto customTopK = stablehlo::CustomCallOp::create(
        rewriter, op.getLoc(), convertedTypes, convertedOperands,
        callTargetName,
        false,         // has_side_effect
        backendConfig, // backend_config with K value
        stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
        nullptr, // called_computations
        nullptr, // operand_layouts
        nullptr, // result_layouts
        nullptr  // output_operand_aliases
    );

    SmallVector<Value> results(customTopK.getResults().begin(),
                               customTopK.getResults().end());
    if (!largest) {
      results[0] = stablehlo::NegOp::create(rewriter, loc, results[0]);
    }

    if (needsPermute) {
      SmallVector<int64_t> inversePermutation(inputRank);
      for (size_t i = 0; i < permutation.size(); ++i) {
        inversePermutation[permutation[i]] = i;
      }
      results[0] = stablehlo::TransposeOp::create(
          rewriter, loc, results[0],
          rewriter.getDenseI64ArrayAttr(inversePermutation));
      results[1] = stablehlo::TransposeOp::create(
          rewriter, loc, results[1],
          rewriter.getDenseI64ArrayAttr(inversePermutation));
    }

    rewriter.replaceOp(op, results);
    return success();
  }
};

class ConvertNeuronSortOp : public OpConversionPattern<Torch::AtenSortOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(Torch::AtenSortOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value input = adaptor.getOperands()[0];
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType) {
      return rewriter.notifyMatchFailure(op, "input must be ranked tensor");
    }

    int64_t dim;
    if (!matchPattern(op.getDim(), Torch::m_TorchConstantInt(&dim))) {
      return rewriter.notifyMatchFailure(op, "dim must be a constant integer");
    }

    int64_t inputRank = inputType.getRank();
    if (dim < 0)
      dim += inputRank;

    bool needsPermute = (dim != inputRank - 1);
    SmallVector<int64_t> permutation;
    if (needsPermute) {
      for (int64_t i = 0; i < inputRank; ++i) {
        if (i != dim)
          permutation.push_back(i);
      }
      permutation.push_back(dim);
    }

    bool descending;
    if (!matchPattern(op.getDescending(),
                      Torch::m_TorchConstantBool(&descending))) {
      return rewriter.notifyMatchFailure(
          op, "descending must be a constant boolean");
    }

    Value convertedInput =
        hlo::promoteType(rewriter, loc, input, inputType.getElementType());
    auto workingType = cast<RankedTensorType>(convertedInput.getType());

    if (needsPermute) {
      auto permutedType = RankedTensorType::get(
          llvm::to_vector(llvm::map_range(
              permutation,
              [&](int64_t i) { return workingType.getDimSize(i); })),
          workingType.getElementType());
      convertedInput = stablehlo::TransposeOp::create(
          rewriter, loc, permutedType, convertedInput,
          rewriter.getDenseI64ArrayAttr(permutation));
      workingType = permutedType;
    }

    int64_t kValue = workingType.getDimSize(workingType.getRank() - 1);
    if (kValue == ShapedType::kDynamic) {
      return rewriter.notifyMatchFailure(op, "sort dim must have static size");
    }

    if (kValue > 16384) {
      return rewriter.notifyMatchFailure(
          op, "sort dimension size " + std::to_string(kValue) +
                  " exceeds maximum of 16384 on dim " + std::to_string(dim));
    }

    std::string backendConfigStr = std::to_string(kValue);
    auto backendConfig = rewriter.getStringAttr(backendConfigStr);

    if (!descending) {
      convertedInput = stablehlo::NegOp::create(rewriter, loc, convertedInput);
    }

    SmallVector<Value> convertedOperands = {convertedInput};

    SmallVector<Type> convertedTypes;
    auto valuesShape = workingType.getShape().vec();
    convertedTypes.push_back(
        RankedTensorType::get(valuesShape, workingType.getElementType()));
    convertedTypes.push_back(
        RankedTensorType::get(valuesShape, rewriter.getI64Type()));

    auto callTargetName = rewriter.getStringAttr("AwsNeuronTopK");

    auto customTopK = stablehlo::CustomCallOp::create(
        rewriter, op.getLoc(), convertedTypes, convertedOperands,
        callTargetName, false, backendConfig,
        stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL, nullptr, nullptr,
        nullptr, nullptr);

    SmallVector<Value> results(customTopK.getResults().begin(),
                               customTopK.getResults().end());
    if (!descending) {
      results[0] = stablehlo::NegOp::create(rewriter, loc, results[0]);
    }

    if (needsPermute) {
      SmallVector<int64_t> inversePermutation(inputRank);
      for (size_t i = 0; i < permutation.size(); ++i) {
        inversePermutation[permutation[i]] = i;
      }
      results[0] = stablehlo::TransposeOp::create(
          rewriter, loc, results[0],
          rewriter.getDenseI64ArrayAttr(inversePermutation));
      results[1] = stablehlo::TransposeOp::create(
          rewriter, loc, results[1],
          rewriter.getDenseI64ArrayAttr(inversePermutation));
    }

    rewriter.replaceOp(op, results);
    return success();
  }
};

class ConvertNeuronSiluOp : public OpConversionPattern<Torch::AtenSiluOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(Torch::AtenSiluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value input = adaptor.getOperands()[0];
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType) {
      return rewriter.notifyMatchFailure(op, "input must be ranked tensor");
    }

    Value convertedInput =
        hlo::promoteType(rewriter, loc, input, inputType.getElementType());
    SmallVector<Value> convertedOperands = {convertedInput};

    Type convertedType = getTypeConverter()->convertType(op.getType());
    if (!convertedType) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }

    auto callTargetName = rewriter.getStringAttr("AwsNeuronSilu");

    auto customSilu = stablehlo::CustomCallOp::create(
        rewriter, op.getLoc(), convertedType, convertedOperands, callTargetName,
        false,   // has_side_effect
        nullptr, // backend_config
        stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
        nullptr, // called_computations
        nullptr, // operand_layouts
        nullptr, // result_layouts
        nullptr  // output_operand_aliases
    );

    rewriter.replaceOp(op, customSilu.getResult(0));
    return success();
  }
};

class ConvertNeuronQuantizeMxOp
    : public OpConversionPattern<Torch::AtenQuantizeMxOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(Torch::AtenQuantizeMxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // 1. Extract and validate input tensor
    Value input = adaptor.getOperands()[0];
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType) {
      return rewriter.notifyMatchFailure(op, "input must be ranked tensor");
    }

    Type inputElemType = inputType.getElementType();
    if (!isa<BFloat16Type>(inputElemType) && !inputElemType.isF16()) {
      return rewriter.notifyMatchFailure(
          op, "input element type must be bf16 or f16");
    }

    int64_t inputRank = inputType.getRank();

    // 2. Extract and validate dim
    int64_t origDim;
    if (!matchPattern(op.getDim(), Torch::m_TorchConstantInt(&origDim))) {
      return rewriter.notifyMatchFailure(op, "dim must be a constant integer");
    }
    // Normalize for validation and shape computation
    int64_t dim = origDim;
    if (dim < 0)
      dim += inputRank;
    if (dim != inputRank - 1 && dim != inputRank - 2) {
      return rewriter.notifyMatchFailure(
          op, "dim must be last (-1) or second-to-last (-2) dimension");
    }

    // 3. Extract and validate block_size
    int64_t blockSize;
    if (!matchPattern(op.getBlockSize(),
                      Torch::m_TorchConstantInt(&blockSize))) {
      return rewriter.notifyMatchFailure(
          op, "block_size must be a constant integer");
    }
    if (blockSize != 32) {
      return rewriter.notifyMatchFailure(
          op, "block_size must be 32 (MX specification requirement)");
    }

    // 4. Extract and validate dtype (ScalarType int -> MLIR type + string)
    int64_t dtypeInt;
    if (!matchPattern(op.getDtype(), Torch::m_TorchConstantInt(&dtypeInt))) {
      return rewriter.notifyMatchFailure(op,
                                         "dtype must be a constant integer");
    }

    std::string dtypeStr;
    if (dtypeInt ==
        static_cast<int64_t>(torch_upstream::ScalarType::Float8_e5m2)) {
      dtypeStr = "float8_e5m2";
    } else if (dtypeInt == static_cast<int64_t>(
                               torch_upstream::ScalarType::Float8_e4m3fn)) {
      dtypeStr = "float8_e4m3fn";
    } else {
      return rewriter.notifyMatchFailure(
          op, "dtype must be Float8_e5m2 (23) or Float8_e4m3fn (24)");
    }

    // 5. Extract and validate scale_calculation_mode
    int64_t scaleCalcMode;
    if (!matchPattern(op.getScaleCalculationMode(),
                      Torch::m_TorchConstantInt(&scaleCalcMode))) {
      return rewriter.notifyMatchFailure(
          op, "scale_calculation_mode must be a constant integer");
    }
    if (scaleCalcMode != 0) {
      return rewriter.notifyMatchFailure(
          op, "only scale_calculation_mode=0 (EMAX) is supported by Neuron");
    }

    // 6. Validate quantization dimension has static size
    int64_t dimSize = inputType.getDimSize(dim);
    if (dimSize == ShapedType::kDynamic) {
      return rewriter.notifyMatchFailure(
          op, "quantization dimension must have static size");
    }

    // 7. Build output types
    // Output #0: quantized data - packed uint32 (4 FP8 values per element)
    // Shape: same as input, but quantization dim divided by 4
    auto ui32Type =
        IntegerType::get(rewriter.getContext(), 32, IntegerType::Unsigned);
    SmallVector<int64_t> packedShape(inputType.getShape());
    packedShape[dim] = dimSize / 4;
    auto outputType = RankedTensorType::get(packedShape, ui32Type);

    // Output #1: scale tensor - f8E8M0FNU, dim replaced by ceil(dimSize/32)
    auto scaleElemType = Float8E8M0FNUType::get(rewriter.getContext());
    SmallVector<int64_t> scaleShape(inputType.getShape());
    scaleShape[dim] = (dimSize + blockSize - 1) / blockSize;
    auto scaleType = RankedTensorType::get(scaleShape, scaleElemType);

    // 8. Build JSON backend_config (use original dim, backend handles
    // negatives)
    std::string backendConfigStr =
        "{\"dtype\": \"" + dtypeStr +
        "\", \"dim\": " + std::to_string(origDim) +
        ", \"block_size\": " + std::to_string(blockSize) +
        ", \"scale_method\": \"EMAX\"}";
    auto backendConfig = rewriter.getStringAttr(backendConfigStr);

    // 9. Promote input type if needed
    Value convertedInput =
        hlo::promoteType(rewriter, loc, input, inputType.getElementType());

    // 10. Create the custom call
    SmallVector<Value> operands = {convertedInput};
    SmallVector<Type> resultTypes = {outputType, scaleType};

    auto callTargetName = rewriter.getStringAttr("QuantizeMX");
    auto customCall = stablehlo::CustomCallOp::create(
        rewriter, loc, resultTypes, operands, callTargetName,
        false,         // has_side_effect
        backendConfig, // JSON backend_config
        stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
        nullptr, // called_computations
        nullptr, // operand_layouts
        nullptr, // result_layouts
        nullptr  // output_operand_aliases
    );

    // 11. Handle scale type: the custom call always returns f8E8M0FNU.
    // PyTorch currently returns scale as uint8 (will change to f8E8M0FNU).
    // f8E8M0FNU and ui8 are both 8-bit types, so bitcast is lossless.
    // If the original torch op expects ui8, insert a bitcast_convert.
    Value quantizedResult = customCall.getResult(0);
    Value scaleResult = customCall.getResult(1);

    auto origScaleType = cast<Torch::ValueTensorType>(op.getScale().getType());
    Type origScaleDtype = origScaleType.getDtype();
    // Check if the original scale dtype is unsigned int8 (ui8)
    if (origScaleDtype.isUnsignedInteger(8)) {
      auto ui8Type =
          IntegerType::get(rewriter.getContext(), 8, IntegerType::Unsigned);
      auto bitcastScaleType = RankedTensorType::get(scaleShape, ui8Type);
      scaleResult = stablehlo::BitcastConvertOp::create(
          rewriter, loc, bitcastScaleType, scaleResult);
    }

    rewriter.replaceOp(op, {quantizedResult, scaleResult});
    return success();
  }
};

} // namespace

// Custom decomposition for AtenMaxPool2dWithIndicesOp using aten.im2col
namespace {
class DecomposeAtenMaxPool2dWithIndicesOp
    : public OpRewritePattern<AtenMaxPool2dWithIndicesOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenMaxPool2dWithIndicesOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    auto inputTy = cast<BaseTensorType>(input.getType());
    auto inputShape = inputTy.getSizes();
    auto inputRank = inputShape.size();

    if (inputRank != 4)
      return rewriter.notifyMatchFailure(op, "expected 4D input tensor");

    SmallVector<int64_t> kernelSize, stride, padding, dilation;
    bool ceilMode;

    if (!matchPattern(op.getKernelSize(),
                      m_TorchListOfConstantInts(kernelSize)))
      return rewriter.notifyMatchFailure(op, "non-const kernel_size");

    if (!matchPattern(op.getStride(), m_TorchListOfConstantInts(stride))) {
      if (isa<Torch::NoneType>(op.getStride().getType()))
        stride = kernelSize;
      else
        return rewriter.notifyMatchFailure(op, "non-const stride");
    }
    if (stride.empty())
      stride = kernelSize;

    if (!matchPattern(op.getPadding(), m_TorchListOfConstantInts(padding)))
      return rewriter.notifyMatchFailure(op, "non-const padding");
    if (!matchPattern(op.getDilation(), m_TorchListOfConstantInts(dilation)))
      return rewriter.notifyMatchFailure(op, "non-const dilation");
    if (!matchPattern(op.getCeilMode(), m_TorchConstantBool(&ceilMode)))
      return rewriter.notifyMatchFailure(op, "non-const ceil_mode");

    // Extract dimensions
    int64_t N = inputShape[0], C = inputShape[1];
    int64_t H = inputShape[2], W = inputShape[3];

    // Compute output dimensions
    int64_t outH =
        (H + 2 * padding[0] - dilation[0] * (kernelSize[0] - 1) - 1) /
            stride[0] +
        1;
    int64_t outW =
        (W + 2 * padding[1] - dilation[1] * (kernelSize[1] - 1) - 1) /
            stride[1] +
        1;

    // Create list values for im2col
    Value kernelSizeList = PrimListConstructOp::create(
        rewriter, loc, Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        ValueRange{
            ConstantIntOp::create(rewriter, loc,
                                  rewriter.getI64IntegerAttr(kernelSize[0])),
            ConstantIntOp::create(rewriter, loc,
                                  rewriter.getI64IntegerAttr(kernelSize[1]))});
    Value dilationList = PrimListConstructOp::create(
        rewriter, loc, Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        ValueRange{
            ConstantIntOp::create(rewriter, loc,
                                  rewriter.getI64IntegerAttr(dilation[0])),
            ConstantIntOp::create(rewriter, loc,
                                  rewriter.getI64IntegerAttr(dilation[1]))});
    Value paddingList = PrimListConstructOp::create(
        rewriter, loc, Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        ValueRange{ConstantIntOp::create(
                       rewriter, loc, rewriter.getI64IntegerAttr(padding[0])),
                   ConstantIntOp::create(
                       rewriter, loc, rewriter.getI64IntegerAttr(padding[1]))});
    Value strideList = PrimListConstructOp::create(
        rewriter, loc, Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        ValueRange{ConstantIntOp::create(rewriter, loc,
                                         rewriter.getI64IntegerAttr(stride[0])),
                   ConstantIntOp::create(
                       rewriter, loc, rewriter.getI64IntegerAttr(stride[1]))});

    // Use im2col to extract pooling windows: [N, C*kH*kW, outH*outW]
    int64_t colSize = C * kernelSize[0] * kernelSize[1];
    auto im2colTy = inputTy.getWithSizesAndDtype(
        llvm::ArrayRef<int64_t>{N, colSize, outH * outW},
        inputTy.getOptionalDtype());
    Value columns =
        AtenIm2colOp::create(rewriter, loc, im2colTy, input, kernelSizeList,
                             dilationList, paddingList, strideList);

    // Reshape to [N, C, kH*kW, outH*outW]
    SmallVector<Value> reshapeSize{
        ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(N)),
        ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(C)),
        ConstantIntOp::create(
            rewriter, loc,
            rewriter.getI64IntegerAttr(kernelSize[0] * kernelSize[1])),
        ConstantIntOp::create(rewriter, loc,
                              rewriter.getI64IntegerAttr(outH * outW))};
    Value reshapeSizeList = PrimListConstructOp::create(
        rewriter, loc, Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        reshapeSize);
    auto reshapeTy = inputTy.getWithSizesAndDtype(
        llvm::ArrayRef<int64_t>{N, C, kernelSize[0] * kernelSize[1],
                                outH * outW},
        inputTy.getOptionalDtype());
    Value reshaped = AtenReshapeOp::create(rewriter, loc, reshapeTy, columns,
                                           reshapeSizeList);

    // Compute max and argmax along dim=2 (kernel dimension)
    Value dim =
        ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(2));
    Value keepDim = ConstantBoolOp::create(rewriter, loc, false);

    auto maxTy = inputTy.getWithSizesAndDtype(
        llvm::ArrayRef<int64_t>{N, C, outH * outW}, inputTy.getOptionalDtype());
    auto idxTy = inputTy.getWithSizesAndDtype(
        llvm::ArrayRef<int64_t>{N, C, outH * outW},
        rewriter.getIntegerType(64, IntegerType::Signed));

    Value maxResult = AtenMaxDimOp::create(rewriter, loc, maxTy, idxTy,
                                           reshaped, dim, keepDim)
                          .getValues();
    Value windowIndices = AtenMaxDimOp::create(rewriter, loc, maxTy, idxTy,
                                               reshaped, dim, keepDim)
                              .getIndices();

    // Convert window indices to flattened image indices
    // windowIdx = kh * kW + kw, need to compute: (oh * stride[0] + kh) * W +
    // (ow * stride[1] + kw)
    Value kW = ConstantIntOp::create(rewriter, loc,
                                     rewriter.getI64IntegerAttr(kernelSize[1]));
    Value strideH = ConstantIntOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(stride[0]));
    Value strideW = ConstantIntOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(stride[1]));
    Value width =
        ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(W));

    // kh = windowIdx // kW, kw = windowIdx % kW
    Value kh = AtenFloorDivideScalarOp::create(rewriter, loc, idxTy,
                                               windowIndices, kW);
    Value kw =
        AtenRemainderScalarOp::create(rewriter, loc, idxTy, windowIndices, kW);

    // Reshape to [N, C, outH, outW] to get output positions
    SmallVector<Value> outSize{
        ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(N)),
        ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(C)),
        ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(outH)),
        ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(outW))};
    Value outSizeList = PrimListConstructOp::create(
        rewriter, loc, Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        outSize);
    auto finalIdxTy = inputTy.getWithSizesAndDtype(
        llvm::ArrayRef<int64_t>{N, C, outH, outW},
        rewriter.getIntegerType(64, IntegerType::Signed));

    kh = AtenReshapeOp::create(rewriter, loc, finalIdxTy, kh, outSizeList);
    kw = AtenReshapeOp::create(rewriter, loc, finalIdxTy, kw, outSizeList);

    // Create iota for oh, ow
    Value ohIota = AtenArangeStartStepOp::create(
        rewriter, loc,
        inputTy.getWithSizesAndDtype(
            llvm::ArrayRef<int64_t>{outH},
            rewriter.getIntegerType(64, IntegerType::Signed)),
        ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(0)),
        ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(outH)),
        ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1)),
        /*dtype=*/ConstantNoneOp::create(rewriter, loc),
        /*layout=*/ConstantNoneOp::create(rewriter, loc),
        /*device=*/ConstantNoneOp::create(rewriter, loc),
        /*pin_memory=*/ConstantNoneOp::create(rewriter, loc));
    Value owIota = AtenArangeStartStepOp::create(
        rewriter, loc,
        inputTy.getWithSizesAndDtype(
            llvm::ArrayRef<int64_t>{outW},
            rewriter.getIntegerType(64, IntegerType::Signed)),
        ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(0)),
        ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(outW)),
        ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1)),
        /*dtype=*/ConstantNoneOp::create(rewriter, loc),
        /*layout=*/ConstantNoneOp::create(rewriter, loc),
        /*device=*/ConstantNoneOp::create(rewriter, loc),
        /*pin_memory=*/ConstantNoneOp::create(rewriter, loc));

    // Reshape and broadcast oh, ow to [N, C, outH, outW]
    SmallVector<Value> ohShape{
        ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1)),
        ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1)),
        ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(outH)),
        ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1))};
    Value ohShapeList = PrimListConstructOp::create(
        rewriter, loc, Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        ohShape);
    ohIota = AtenReshapeOp::create(
        rewriter, loc,
        inputTy.getWithSizesAndDtype(
            llvm::ArrayRef<int64_t>{1, 1, outH, 1},
            rewriter.getIntegerType(64, IntegerType::Signed)),
        ohIota, ohShapeList);
    ohIota = AtenBroadcastToOp::create(rewriter, loc, finalIdxTy, ohIota,
                                       outSizeList);

    SmallVector<Value> owShape{
        ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1)),
        ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1)),
        ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1)),
        ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(outW))};
    Value owShapeList = PrimListConstructOp::create(
        rewriter, loc, Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        owShape);
    owIota = AtenReshapeOp::create(
        rewriter, loc,
        inputTy.getWithSizesAndDtype(
            llvm::ArrayRef<int64_t>{1, 1, 1, outW},
            rewriter.getIntegerType(64, IntegerType::Signed)),
        owIota, owShapeList);
    owIota = AtenBroadcastToOp::create(rewriter, loc, finalIdxTy, owIota,
                                       outSizeList);

    // h = oh * stride[0] + kh, w = ow * stride[1] + kw
    Value h =
        AtenMulScalarOp::create(rewriter, loc, finalIdxTy, ohIota, strideH);
    h = AtenAddTensorOp::create(
        rewriter, loc, finalIdxTy, h, kh,
        ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1)));
    Value w =
        AtenMulScalarOp::create(rewriter, loc, finalIdxTy, owIota, strideW);
    w = AtenAddTensorOp::create(
        rewriter, loc, finalIdxTy, w, kw,
        ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1)));

    // flatIdx = h * W + w
    Value indices =
        AtenMulScalarOp::create(rewriter, loc, finalIdxTy, h, width);
    indices = AtenAddTensorOp::create(
        rewriter, loc, finalIdxTy, indices, w,
        ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1)));

    // Reshape max output to [N, C, outH, outW]
    auto finalMaxTy = inputTy.getWithSizesAndDtype(
        llvm::ArrayRef<int64_t>{N, C, outH, outW}, inputTy.getOptionalDtype());
    Value finalMax = AtenReshapeOp::create(rewriter, loc, finalMaxTy, maxResult,
                                           outSizeList);

    rewriter.replaceOp(op, {finalMax, indices});
    return success();
  }
};
} // namespace

namespace {
class ConvertTorchToNeuron
    : public impl::ConvertTorchToNeuronBase<ConvertTorchToNeuron> {
public:
  using impl::ConvertTorchToNeuronBase<
      ConvertTorchToNeuron>::ConvertTorchToNeuronBase;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<stablehlo::StablehloDialect>();
    registry.insert<Torch::TorchDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<TorchConversion::TorchConversionDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    llvm::CrashRecoveryContext::Enable();
    llvm::CrashRecoveryContext CRC;
    LogicalResult result = failure();

    bool crashed = !CRC.RunSafely([&]() {
      MLIRContext *context = &getContext();
      ConversionTarget target(*context);
      target.addLegalDialect<stablehlo::StablehloDialect, Torch::TorchDialect,
                             func::FuncDialect,
                             TorchConversion::TorchConversionDialect>();

      TypeConverter typeConverter;
      typeConverter.addConversion([](Type type) { return type; });
      TorchConversion::setupBackendTypeConversionForStablehlo(target,
                                                              typeConverter);

      RewritePatternSet patterns(context);
      target.addIllegalOp<Torch::OperatorOp>();
      patterns.add<ConvertNeuronOperatorOp>(typeConverter, context);

      // Custom call kernel lowerings
      target.addIllegalOp<Torch::AtenTopkOp>();
      patterns.add<ConvertNeuronTopkOp>(typeConverter, context);

      target.addIllegalOp<Torch::AtenSortOp>();
      patterns.add<ConvertNeuronSortOp>(typeConverter, context);

      target.addIllegalOp<Torch::AtenSiluOp>();
      patterns.add<ConvertNeuronSiluOp>(typeConverter, context);

      target.addIllegalOp<Torch::AtenQuantizeMxOp>();
      patterns.add<ConvertNeuronQuantizeMxOp>(typeConverter, context);

      // Decomposition patterns
      target.addIllegalOp<Torch::AtenMaxPool2dWithIndicesOp>();
      patterns.add<DecomposeAtenMaxPool2dWithIndicesOp>(context);

      result =
          applyPartialConversion(getOperation(), target, std::move(patterns));
    });

    if (crashed || failed(result)) {
      if (crashed)
        getOperation()->emitError()
            << "ConvertTorchToNeuron pass crashed unexpectedly";
      signalPassFailure();
    }
  }
};
} // namespace

// Default pass creation function (required by tablegen)
std::unique_ptr<OperationPass<func::FuncOp>> createConvertTorchToNeuronPass() {
  return std::make_unique<ConvertTorchToNeuron>();
}

} // namespace mlir::torch
