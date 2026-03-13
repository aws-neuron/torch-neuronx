#ifndef TORCH_NEURONX_CSRC_CORE_STABLEHLO_MODULEBUILDER_BROADCAST_OP_BUILDER_H_
#define TORCH_NEURONX_CSRC_CORE_STABLEHLO_MODULEBUILDER_BROADCAST_OP_BUILDER_H_

#include <string>
#include <vector>

#include "OpModuleBuilder.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace torch_neuronx {

/**
 * @brief Builder for creating MLIR modules with StableHLO broadcast_in_dim operations
 *
 * This class inherits from OpModuleBuilder and implements the specific logic
 * for creating broadcast operations. It manages broadcast-specific attributes
 * like input shape, output shape, and broadcast dimensions mapping.
 *
 * The broadcast_in_dim operation expands dimensions and/or rank of an input tensor
 * by duplicating data. It follows StableHLO semantics where:
 * - Size-1 dimensions can broadcast to any size
 * - Other dimensions must exactly match the output dimension they map to
 */
class BroadcastOpBuilder : public OpModuleBuilder {
 public:
  /**
   * @brief Template constructor for element type
   * @tparam ElementType Either std::string or mlir::Type
   * @param input_shape Input tensor shape
   * @param output_shape Output tensor shape
   * @param broadcast_dimensions Maps input dims to output dims (size must equal input rank)
   * @param element_type Element type (string like "f32" or mlir::Type)
   * @param enable_verification Enable MLIR verification (default: false)
   * @throws std::invalid_argument if parameters are invalid
   */
  template <typename ElementType>
  BroadcastOpBuilder(const std::vector<int64_t>& input_shape,
                     const std::vector<int64_t>& output_shape,
                     const std::vector<int64_t>& broadcast_dimensions, ElementType element_type,
                     bool enable_verification = false);

  virtual ~BroadcastOpBuilder() = default;

 protected:
  /**
   * @brief Build the broadcast_in_dim operation
   * @param func Function to build operation in
   * @param builder OpBuilder positioned at function body start
   * @return Result value of the broadcast operation
   */
  mlir::Value buildOperation(mlir::func::FuncOp func, mlir::OpBuilder& builder) override;

  /**
   * @brief Get input tensor type
   * @return Input ranked tensor type
   */
  mlir::RankedTensorType getInputType() override;

  /**
   * @brief Get output tensor type
   * @return Output ranked tensor type
   */
  mlir::RankedTensorType getOutputType() override;

 private:
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;
  std::vector<int64_t> broadcast_dimensions_;
  mlir::Type element_type_;

  /**
   * @brief Validate broadcast dimensions and shape compatibility
   *
   * Validates according to StableHLO broadcast_in_dim semantics:
   * - Broadcast dimensions size must equal input rank
   * - Each broadcast dimension must be a valid output dimension index
   * - Broadcast dimensions must be unique
   * - Input dimensions must be 1 or match the corresponding output dimension
   *
   * @param input_shape Input tensor shape
   * @param output_shape Output tensor shape
   * @param broadcast_dimensions Broadcast dimension mapping
   * @throws std::invalid_argument if validation fails
   */
  static void validateShapeCompatibility(const std::vector<int64_t>& input_shape,
                                         const std::vector<int64_t>& output_shape,
                                         const std::vector<int64_t>& broadcast_dimensions);
};

}  // namespace torch_neuronx

#endif  // TORCH_NEURONX_CSRC_CORE_STABLEHLO_MODULEBUILDER_BROADCAST_OP_BUILDER_H_
