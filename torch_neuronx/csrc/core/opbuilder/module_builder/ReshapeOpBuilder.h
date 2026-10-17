#ifndef TORCH_NEURONX_CSRC_CORE_STABLEHLO_MODULEBUILDER_RESHAPE_OP_BUILDER_H_
#define TORCH_NEURONX_CSRC_CORE_STABLEHLO_MODULEBUILDER_RESHAPE_OP_BUILDER_H_

#include <string>
#include <vector>

#include "OpModuleBuilder.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace torch_neuronx {

/**
 * @brief Builder for creating MLIR modules with StableHLO reshape operations
 *
 * This class inherits from OpModuleBuilder and implements the specific logic
 * for creating reshape operations. It manages reshape-specific attributes
 * like input and output tensor shapes.
 */
class ReshapeOpBuilder : public OpModuleBuilder {
 public:
  /**
   * @brief Template constructor for element type
   * @tparam ElementType Either std::string or mlir::Type
   * @param input_shape Input tensor shape
   * @param output_shape Output tensor shape
   * @param element_type Element type (string like "f32" or mlir::Type)
   * @param enable_verification Enable MLIR verification (default: false)
   * @throws std::invalid_argument if parameters are invalid
   */
  template <typename ElementType>
  ReshapeOpBuilder(const std::vector<int64_t>& input_shape,
                   const std::vector<int64_t>& output_shape, ElementType element_type,
                   bool enable_verification = false);

  virtual ~ReshapeOpBuilder() = default;

 protected:
  /**
   * @brief Build the reshape operation
   * @param func Function to build operation in
   * @param builder OpBuilder positioned at function body start
   * @return Result value of the reshape operation
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
  mlir::Type element_type_;

  /**
   * @brief Validate shape compatibility
   * @param input_shape Input tensor shape
   * @param output_shape Output tensor shape
   * @throws std::invalid_argument if shapes are incompatible
   */
  static void validateShapeCompatibility(const std::vector<int64_t>& input_shape,
                                         const std::vector<int64_t>& output_shape);

  /**
   * @brief Compute total number of elements in a shape
   * @param shape Tensor shape
   * @return Total number of elements
   */
  static int64_t computeTotalElements(const std::vector<int64_t>& shape);
};

}  // namespace torch_neuronx

#endif  // TORCH_NEURONX_CSRC_CORE_STABLEHLO_MODULEBUILDER_RESHAPE_OP_BUILDER_H_
