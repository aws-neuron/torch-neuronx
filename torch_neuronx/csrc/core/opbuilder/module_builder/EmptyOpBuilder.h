#ifndef TORCH_NEURONX_CSRC_CORE_STABLEHLO_MODULEBUILDER_EMPTY_OP_BUILDER_H_
#define TORCH_NEURONX_CSRC_CORE_STABLEHLO_MODULEBUILDER_EMPTY_OP_BUILDER_H_

#include <string>
#include <vector>

#include "OpModuleBuilder.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace torch_neuronx {

/**
 * @brief Builder for creating MLIR modules with no operations (identity/passthrough)
 *
 * This class inherits from OpModuleBuilder and implements a minimal operation
 * that simply passes the input to the output without any transformation.
 * This is useful for testing the module builder infrastructure or as a template
 * for creating new operations.
 */
class EmptyOpBuilder : public OpModuleBuilder {
 public:
  /**
   * @brief Template constructor for element type
   * @tparam ElementType Either std::string or mlir::Type
   * @param shape Tensor shape for both input and output
   * @param element_type Element type (string like "f32" or mlir::Type)
   * @param enable_verification Enable MLIR verification (default: false)
   * @throws std::invalid_argument if parameters are invalid
   */
  template <typename ElementType>
  EmptyOpBuilder(const std::vector<int64_t>& shape, ElementType element_type,
                 bool enable_verification = false);

  virtual ~EmptyOpBuilder() = default;

 protected:
  /**
   * @brief Build the empty operation (identity passthrough)
   * @param func Function to build operation in
   * @param builder OpBuilder positioned at function body start
   * @return Result value (the input argument unchanged)
   */
  mlir::Value buildOperation(mlir::func::FuncOp func, mlir::OpBuilder& builder) override;

  /**
   * @brief Get input tensor type
   * @return Input ranked tensor type
   */
  mlir::RankedTensorType getInputType() override;

  /**
   * @brief Get output tensor type
   * @return Output ranked tensor type (same as input)
   */
  mlir::RankedTensorType getOutputType() override;

 private:
  std::vector<int64_t> shape_;
  mlir::Type element_type_;
};

}  // namespace torch_neuronx

#endif  // TORCH_NEURONX_CSRC_CORE_STABLEHLO_MODULEBUILDER_EMPTY_OP_BUILDER_H_
