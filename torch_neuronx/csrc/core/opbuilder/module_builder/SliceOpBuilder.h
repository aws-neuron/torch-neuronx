#ifndef TORCH_NEURONX_CSRC_CORE_STABLEHLO_MODULEBUILDER_SLICE_OP_BUILDER_H_
#define TORCH_NEURONX_CSRC_CORE_STABLEHLO_MODULEBUILDER_SLICE_OP_BUILDER_H_

#include <string>
#include <vector>

#include "OpModuleBuilder.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace torch_neuronx {

/**
 * @brief Builder for creating MLIR modules with StableHLO slice operations
 *
 * This class inherits from OpModuleBuilder and implements the specific logic
 * for creating slice operations. It manages slice-specific attributes
 * like start indices, limit indices, and strides.
 */
class SliceOpBuilder : public OpModuleBuilder {
 public:
  /**
   * @brief Template constructor for element type
   * @tparam ElementType Either std::string or mlir::Type
   * @param input_shape Input tensor shape
   * @param start_indices Starting indices for each dimension
   * @param limit_indices Ending indices (exclusive) for each dimension
   * @param strides Stride for each dimension
   * @param element_type Element type (string like "f32" or mlir::Type)
   * @param enable_verification Enable MLIR verification (default: false)
   * @throws std::invalid_argument if parameters are invalid
   */
  template <typename ElementType>
  SliceOpBuilder(const std::vector<int64_t>& input_shape, const std::vector<int64_t>& start_indices,
                 const std::vector<int64_t>& limit_indices, const std::vector<int64_t>& strides,
                 ElementType element_type, bool enable_verification = false);

  virtual ~SliceOpBuilder() = default;

 protected:
  /**
   * @brief Build the slice operation
   * @param func Function to build operation in
   * @param builder OpBuilder positioned at function body start
   * @return Result value of the slice operation
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
  std::vector<int64_t> start_indices_;
  std::vector<int64_t> limit_indices_;
  std::vector<int64_t> strides_;
  std::vector<int64_t> output_shape_;
  mlir::Type element_type_;

  /**
   * @brief Validate slice parameters
   * @param input_shape Input tensor shape
   * @param start_indices Starting indices
   * @param limit_indices Ending indices
   * @param strides Stride values
   * @throws std::invalid_argument if parameters are invalid
   */
  static void validateSliceParameters(const std::vector<int64_t>& input_shape,
                                      const std::vector<int64_t>& start_indices,
                                      const std::vector<int64_t>& limit_indices,
                                      const std::vector<int64_t>& strides);

  /**
   * @brief Compute output shape from slice parameters
   * @param input_shape Input tensor shape
   * @param start_indices Starting indices
   * @param limit_indices Ending indices
   * @param strides Stride values
   * @return Output tensor shape
   */
  static std::vector<int64_t> computeOutputShape(const std::vector<int64_t>& input_shape,
                                                 const std::vector<int64_t>& start_indices,
                                                 const std::vector<int64_t>& limit_indices,
                                                 const std::vector<int64_t>& strides);
};

}  // namespace torch_neuronx

#endif  // TORCH_NEURONX_CSRC_CORE_STABLEHLO_MODULEBUILDER_SLICE_OP_BUILDER_H_
