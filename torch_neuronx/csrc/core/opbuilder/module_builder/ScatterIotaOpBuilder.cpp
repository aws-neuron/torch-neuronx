#include "ScatterIotaOpBuilder.h"

#include <numeric>
#include <stdexcept>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/opbuilder/utility/TypeUtils.h"

namespace torch_neuronx {

ScatterIotaOpBuilder::ScatterIotaOpBuilder(const std::vector<int64_t>& view_shape,
                                           const std::vector<int64_t>& view_strides,
                                           int64_t storage_offset, int64_t storage_span,
                                           const std::string& element_type,
                                           bool enable_verification)
    : OpModuleBuilder(enable_verification),
      view_shape_(view_shape),
      view_strides_(view_strides),
      storage_offset_(storage_offset),
      storage_span_(storage_span),
      element_type_(type_utils::stringToMlirType(getBuilder(), element_type)) {
  TORCH_NEURONX_DEBUG("ScatterIotaOpBuilder: Initializing");

  if (view_shape_.empty()) {
    throw std::invalid_argument("view_shape cannot be empty");
  }
  if (view_shape_.size() != view_strides_.size()) {
    throw std::invalid_argument("view_shape rank (" + std::to_string(view_shape_.size()) +
                                ") must match view_strides rank (" +
                                std::to_string(view_strides_.size()) + ")");
  }
  if (storage_span_ <= 0) {
    throw std::invalid_argument("storage_span must be positive (got " +
                                std::to_string(storage_span_) + ")");
  }
  if (storage_offset_ < 0) {
    throw std::invalid_argument("storage_offset must be non-negative (got " +
                                std::to_string(storage_offset_) + ")");
  }

  TORCH_NEURONX_DEBUG("ScatterIotaOpBuilder: Initialized successfully",
                      "view_shape_rank=", view_shape_.size(), "storage_span=", storage_span_,
                      "storage_offset=", storage_offset_);
}

mlir::OwningOpRef<mlir::ModuleOp> ScatterIotaOpBuilder::build() {
  TORCH_NEURONX_DEBUG("ScatterIotaOpBuilder: Starting module build");

  auto& builder = getBuilder();
  auto loc = builder.getUnknownLoc();

  // Create module
  auto module = builder.create<mlir::ModuleOp>(loc);
  auto& moduleBody = module.getBodyRegion().front();
  builder.setInsertionPointToEnd(&moduleBody);

  // Compute view numel
  int64_t view_numel = 1;
  for (auto s : view_shape_) view_numel *= s;

  // Type definitions
  auto updateType = mlir::RankedTensorType::get(view_shape_, element_type_);
  auto baseFlatType = mlir::RankedTensorType::get({storage_span_}, element_type_);
  auto i64Type = builder.getI64Type();

  // Function: @main(%update: tensor<view_shape>, %base_flat: tensor<storage_span>) ->
  // tensor<storage_span>
  auto funcType = builder.getFunctionType({updateType, baseFlatType}, {baseFlatType});
  auto mainFunc = createFunction("main", funcType);

  auto& funcBody = mainFunc.getBody().emplaceBlock();
  funcBody.addArgument(updateType, loc);
  funcBody.addArgument(baseFlatType, loc);
  builder.setInsertionPointToStart(&funcBody);

  auto updateArg = funcBody.getArgument(0);
  auto baseArg = funcBody.getArgument(1);

  int64_t ndims = view_shape_.size();

  // ====================================================================
  // Step 1: Compute flat indices using iota + stride arithmetic
  //
  // For each element at multi-index (i0, i1, ..., i_{n-1}):
  //   flat_idx = i0 * stride[0] + i1 * stride[1] + ... + i_{n-1} * stride[n-1] + offset
  //
  // We compute this using iota ops (one per dimension) and element-wise
  // multiply + add operations, all broadcast to view_shape.
  // ====================================================================

  auto viewI64Type = mlir::RankedTensorType::get(view_shape_, i64Type);

  // Start with offset constant
  auto offsetAttr =
      mlir::DenseElementsAttr::get(viewI64Type, builder.getI64IntegerAttr(storage_offset_));
  mlir::Value flatIndices =
      builder.create<mlir::stablehlo::ConstantOp>(loc, offsetAttr).getResult();

  // For each dimension, add iota_d * stride_d
  for (int64_t d = 0; d < ndims; ++d) {
    if (view_shape_[d] <= 1) continue;  // size-1 dims contribute nothing (iota always 0)

    // iota for dimension d: values [0, 1, ..., view_shape_[d]-1] broadcast to view_shape
    auto iotaOp =
        builder.create<mlir::stablehlo::IotaOp>(loc, viewI64Type, builder.getI64IntegerAttr(d));

    // stride constant broadcast to view_shape
    auto strideAttr =
        mlir::DenseElementsAttr::get(viewI64Type, builder.getI64IntegerAttr(view_strides_[d]));
    auto strideConst = builder.create<mlir::stablehlo::ConstantOp>(loc, strideAttr);

    // iota_d * stride_d
    auto mulOp = builder.create<mlir::stablehlo::MulOp>(loc, viewI64Type, iotaOp, strideConst);

    // accumulate: flat_indices += iota_d * stride_d
    flatIndices =
        builder.create<mlir::stablehlo::AddOp>(loc, viewI64Type, flatIndices, mulOp).getResult();
  }

  // ====================================================================
  // Step 2: Flatten update and reshape indices for scatter
  // ====================================================================

  // Flatten update: tensor<view_shape> -> tensor<view_numel>
  auto flatUpdateType = mlir::RankedTensorType::get({view_numel}, element_type_);
  auto flatUpdate = builder.create<mlir::stablehlo::ReshapeOp>(loc, flatUpdateType, updateArg);

  // Reshape indices: tensor<view_shape, i64> -> tensor<view_numel, 1, i64>
  // The scatter op expects indices of shape [num_updates, index_vector_dim_size]
  // where index_vector_dim_size = 1 for 1D scatter
  auto flatIndicesType = mlir::RankedTensorType::get({view_numel, 1}, i64Type);
  auto flatIndicesReshaped =
      builder.create<mlir::stablehlo::ReshapeOp>(loc, flatIndicesType, flatIndices);

  // ====================================================================
  // Step 3: Build scatter operation
  //
  // scatter(%base_flat, %flat_indices, %flat_update)
  //   with update_computation = overwrite (return rhs)
  //
  // scatter_dimension_numbers:
  //   update_window_dims = []
  //   inserted_window_dims = [0]
  //   scatter_dims_to_operand_dims = [0]
  //   index_vector_dim = 1
  // ====================================================================

  // Build scatter dimension numbers
  auto scatterDimNumbers =
      mlir::stablehlo::ScatterDimensionNumbersAttr::get(builder.getContext(),
                                                        /*updateWindowDims=*/{},
                                                        /*insertedWindowDims=*/{0},
                                                        /*inputBatchingDims=*/{},
                                                        /*scatterIndicesBatchingDims=*/{},
                                                        /*scatterDimsToOperandDims=*/{0},
                                                        /*indexVectorDim=*/1);

  // Create scatter op
  auto scatterOp = builder.create<mlir::stablehlo::ScatterOp>(
      loc,
      /*resultTypes=*/mlir::TypeRange{baseFlatType},
      /*inputs=*/mlir::ValueRange{baseArg},
      /*scatter_indices=*/flatIndicesReshaped.getResult(),
      /*updates=*/mlir::ValueRange{flatUpdate.getResult()},
      /*scatter_dimension_numbers=*/scatterDimNumbers,
      /*indices_are_sorted=*/false,
      /*unique_indices=*/true);  // Our indices are always unique (no overlapping writes)

  // Build the update computation region: overwrite (return the new value)
  {
    auto& region = scatterOp.getUpdateComputation();
    auto* block = new mlir::Block();
    region.push_back(block);

    // Two scalar arguments: existing value and update value
    auto scalarType = mlir::RankedTensorType::get({}, element_type_);
    block->addArgument(scalarType, loc);
    block->addArgument(scalarType, loc);

    // Return the update value (overwrite semantics)
    mlir::OpBuilder regionBuilder(block, block->end());
    regionBuilder.create<mlir::stablehlo::ReturnOp>(loc, block->getArgument(1));
  }

  // Return scatter result
  builder.create<mlir::func::ReturnOp>(loc, scatterOp.getResult(0));

  TORCH_NEURONX_DEBUG("ScatterIotaOpBuilder: Successfully built module", "view_numel=", view_numel,
                      "storage_span=", storage_span_);

  return mlir::OwningOpRef<mlir::ModuleOp>(module);
}

// These are required by OpModuleBuilder but not used since we override build()
mlir::Value ScatterIotaOpBuilder::buildOperation(mlir::func::FuncOp func,
                                                 mlir::OpBuilder& builder) {
  return func.getBody().front().getArgument(0);
}

mlir::RankedTensorType ScatterIotaOpBuilder::getInputType() {
  return mlir::RankedTensorType::get(view_shape_, element_type_);
}

mlir::RankedTensorType ScatterIotaOpBuilder::getOutputType() {
  return mlir::RankedTensorType::get({storage_span_}, element_type_);
}

}  // namespace torch_neuronx
