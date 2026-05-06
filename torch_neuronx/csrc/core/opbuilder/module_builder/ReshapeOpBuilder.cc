#include "ReshapeOpBuilder.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/opbuilder/utility/TypeUtils.h"

namespace torch_neuronx {

int64_t ReshapeOpBuilder::computeTotalElements(const std::vector<int64_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1),
                         std::multiplies<int64_t>());
}

void ReshapeOpBuilder::validateShapeCompatibility(const std::vector<int64_t>& input_shape,
                                                  const std::vector<int64_t>& output_shape) {
  int64_t input_elements = computeTotalElements(input_shape);
  int64_t output_elements = computeTotalElements(output_shape);

  if (input_elements != output_elements) {
    throw std::invalid_argument(
        "Input shape and output shape must have the same total number of elements (input: " +
        std::to_string(input_elements) + ", output: " + std::to_string(output_elements) + ")");
  }
}

// Template constructor implementation
template <typename ElementType>
ReshapeOpBuilder::ReshapeOpBuilder(const std::vector<int64_t>& input_shape,
                                   const std::vector<int64_t>& output_shape,
                                   ElementType element_type, bool enable_verification)
    : OpModuleBuilder(enable_verification),
      input_shape_(input_shape),
      output_shape_(output_shape),
      element_type_([&]() {
        if constexpr (std::is_same_v<ElementType, std::string> ||
                      std::is_same_v<ElementType, const char*>) {
          return type_utils::stringToMlirType(getBuilder(), element_type);
        } else {
          return element_type;
        }
      }()) {
  TORCH_NEURONX_DEBUG("ReshapeOpBuilder: Initializing");

  // Validate inputs
  if (input_shape_.empty()) {
    throw std::invalid_argument("input_shape cannot be empty");
  }

  if (output_shape_.empty()) {
    throw std::invalid_argument("output_shape cannot be empty");
  }

  for (size_t i = 0; i < input_shape_.size(); ++i) {
    if (input_shape_[i] <= 0) {
      throw std::invalid_argument("input_shape dimensions must be positive (got " +
                                  std::to_string(input_shape_[i]) + " at index " +
                                  std::to_string(i) + ")");
    }
  }

  for (size_t i = 0; i < output_shape_.size(); ++i) {
    if (output_shape_[i] <= 0) {
      throw std::invalid_argument("output_shape dimensions must be positive (got " +
                                  std::to_string(output_shape_[i]) + " at index " +
                                  std::to_string(i) + ")");
    }
  }

  // Validate shape compatibility
  validateShapeCompatibility(input_shape_, output_shape_);

  TORCH_NEURONX_DEBUG("ReshapeOpBuilder: Initialized successfully");
}

// Explicit template instantiations
template ReshapeOpBuilder::ReshapeOpBuilder(const std::vector<int64_t>&,
                                            const std::vector<int64_t>&, std::string, bool);

template ReshapeOpBuilder::ReshapeOpBuilder(const std::vector<int64_t>&,
                                            const std::vector<int64_t>&, const char*, bool);

template ReshapeOpBuilder::ReshapeOpBuilder(const std::vector<int64_t>&,
                                            const std::vector<int64_t>&, mlir::Type, bool);

mlir::RankedTensorType ReshapeOpBuilder::getInputType() {
  return mlir::RankedTensorType::get(input_shape_, element_type_);
}

mlir::RankedTensorType ReshapeOpBuilder::getOutputType() {
  return mlir::RankedTensorType::get(output_shape_, element_type_);
}

mlir::Value ReshapeOpBuilder::buildOperation(mlir::func::FuncOp func, mlir::OpBuilder& builder) {
  TORCH_NEURONX_DEBUG("ReshapeOpBuilder: Building reshape operation");

  // Get function argument
  auto arg = func.getBody().front().getArgument(0);

  // Get output type
  auto outputType = getOutputType();

  // Create reshape operation
  auto reshapeOp =
      builder.create<mlir::stablehlo::ReshapeOp>(builder.getUnknownLoc(), outputType, arg);

  TORCH_NEURONX_DEBUG("ReshapeOpBuilder: Reshape operation created successfully");

  return reshapeOp.getResult();
}

}  // namespace torch_neuronx
