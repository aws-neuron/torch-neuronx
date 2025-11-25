#ifndef TORCH_NEURONX_CSRC_CORE_STABLEHLO_MODULEBUILDER_TRANSPOSE_OP_BUILDER_H_
#define TORCH_NEURONX_CSRC_CORE_STABLEHLO_MODULEBUILDER_TRANSPOSE_OP_BUILDER_H_

#include <string>
#include <vector>

#include "OpModuleBuilder.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace torch_neuronx {

/**
 * @brief Builder for creating MLIR modules with StableHLO transpose operations
 *
 * This class inherits from OpModuleBuilder and implements the specific logic
 * for creating transpose operations. It manages transpose-specific attributes
 * like permutation and tensor shapes.
 */
class TransposeOpBuilder : public OpModuleBuilder {
 public:
  /**
   * @brief Template constructor for element type
   * @tparam ElementType Either std::string or mlir::Type
   * @param permutation Transpose permutation array
   * @param input_shape Input tensor shape
   * @param element_type Element type (string like "f32" or mlir::Type)
   * @param enable_verification Enable MLIR verification (default: false)
   * @throws std::invalid_argument if parameters are invalid
   */
  template <typename ElementType>
  TransposeOpBuilder(const std::vector<int64_t>& permutation,
                     const std::vector<int64_t>& input_shape, ElementType element_type,
                     bool enable_verification = false);

  virtual ~TransposeOpBuilder() = default;

 protected:
  /**
   * @brief Build the transpose operation
   * @param func Function to build operation in
   * @param builder OpBuilder positioned at function body start
   * @return Result value of the transpose operation
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
  std::vector<int64_t> permutation_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;
  mlir::Type element_type_;

  /**
   * @brief Compute output shape from input shape and permutation
   * @param input_shape Input tensor shape
   * @param permutation Transpose permutation
   * @return Output tensor shape
   */
  static std::vector<int64_t> computeOutputShape(const std::vector<int64_t>& input_shape,
                                                 const std::vector<int64_t>& permutation);

  /**
   * @brief Validate permutation array
   * @param permutation Permutation to validate
   * @throws std::invalid_argument if permutation is invalid
   */
  static void validatePermutation(const std::vector<int64_t>& permutation);
};

}  // namespace torch_neuronx

#endif  // TORCH_NEURONX_CSRC_CORE_STABLEHLO_MODULEBUILDER_TRANSPOSE_OP_BUILDER_H_
