#ifndef TORCH_NEURONX_CSRC_CORE_STABLEHLO_MODULEBUILDER_SCATTER_IOTA_OP_BUILDER_H_
#define TORCH_NEURONX_CSRC_CORE_STABLEHLO_MODULEBUILDER_SCATTER_IOTA_OP_BUILDER_H_

#include <string>
#include <vector>

#include "OpModuleBuilder.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace torch_neuronx {

/**
 * @brief Builder for creating MLIR modules with scatter + iota index computation
 *
 * This replaces DynamicUpdateSliceOpBuilder for epilogue slice writeback.
 * Instead of inferring a shaped base tensor (ambiguous from strides), it uses
 * a FLAT 1D base (the entire storage span) and computes exact flat indices
 * using iota + stride arithmetic.
 *
 * Matches TorchNeuron's SCATTER_IOTA approach: unambiguous, handles any
 * stride pattern including strided slices.
 *
 * The generated MLIR function has the signature:
 *   func @main(%update: tensor<view_shape>, %base_flat: tensor<storage_span>) ->
 * tensor<storage_span>
 *
 * Internally:
 *   1. Generate iota per dimension of update shape
 *   2. Compute flat index = sum(iota_d * stride_d) + offset
 *   3. Flatten update and indices
 *   4. Scatter update elements into base_flat at computed indices
 *
 * @note Like DynamicUpdateSliceOpBuilder, this overrides build() to support
 *       2-input functions (%update, %base_flat).
 */
class ScatterIotaOpBuilder : public OpModuleBuilder {
 public:
  /**
   * @brief Constructor
   * @param view_shape Shape of the update tensor (the operation result / view shape)
   * @param view_strides Strides of the destination view in the original tensor
   * @param storage_offset Storage offset of the view into the flat storage
   * @param storage_span Total number of elements in the flat storage
   * @param element_type Element type as string (e.g., "f32", "bf16")
   * @param enable_verification Enable MLIR verification (default: false)
   */
  ScatterIotaOpBuilder(const std::vector<int64_t>& view_shape,
                       const std::vector<int64_t>& view_strides, int64_t storage_offset,
                       int64_t storage_span, const std::string& element_type,
                       bool enable_verification = false);

  virtual ~ScatterIotaOpBuilder() = default;

  /**
   * @brief Build and return the MLIR module with 2-input function
   *
   * Overrides OpModuleBuilder::build() to create a function with two inputs
   * (update tensor + flat base tensor) and scatter with iota-computed indices.
   *
   * @return MLIR module containing the scatter + iota operation
   */
  mlir::OwningOpRef<mlir::ModuleOp> build();

 protected:
  // Required by OpModuleBuilder but not used since we override build()
  mlir::Value buildOperation(mlir::func::FuncOp func, mlir::OpBuilder& builder) override;
  mlir::RankedTensorType getInputType() override;
  mlir::RankedTensorType getOutputType() override;

 private:
  std::vector<int64_t> view_shape_;
  std::vector<int64_t> view_strides_;
  int64_t storage_offset_;
  int64_t storage_span_;
  mlir::Type element_type_;
};

}  // namespace torch_neuronx

#endif  // TORCH_NEURONX_CSRC_CORE_STABLEHLO_MODULEBUILDER_SCATTER_IOTA_OP_BUILDER_H_
