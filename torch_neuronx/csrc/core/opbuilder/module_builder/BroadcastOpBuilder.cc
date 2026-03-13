#include "BroadcastOpBuilder.h"

#include <algorithm>
#include <set>
#include <stdexcept>

#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/opbuilder/utility/TypeUtils.h"

namespace torch_neuronx {

void BroadcastOpBuilder::validateShapeCompatibility(
    const std::vector<int64_t>& input_shape, const std::vector<int64_t>& output_shape,
    const std::vector<int64_t>& broadcast_dimensions) {
  // Validate broadcast dimensions size matches input rank
  if (broadcast_dimensions.size() != input_shape.size()) {
    throw std::invalid_argument(
        "broadcast_dimensions size (" + std::to_string(broadcast_dimensions.size()) +
        ") must equal input rank (" + std::to_string(input_shape.size()) + ")");
  }

  // Validate each broadcast dimension is within valid range
  for (size_t i = 0; i < broadcast_dimensions.size(); ++i) {
    int64_t broadcast_dim = broadcast_dimensions[i];
    if (broadcast_dim < 0 || broadcast_dim >= static_cast<int64_t>(output_shape.size())) {
      throw std::invalid_argument("broadcast_dimensions[" + std::to_string(i) +
                                  "] = " + std::to_string(broadcast_dim) + " is out of range [0, " +
                                  std::to_string(output_shape.size()) + ")");
    }
  }

  // Validate broadcast dimensions are unique
  std::set<int64_t> unique_dims(broadcast_dimensions.begin(), broadcast_dimensions.end());
  if (unique_dims.size() != broadcast_dimensions.size()) {
    throw std::invalid_argument("broadcast_dimensions must contain unique values");
  }

  // Validate dimension compatibility: each input dim must be 1 or match output dim
  for (size_t i = 0; i < input_shape.size(); ++i) {
    int64_t input_dim = input_shape[i];
    int64_t output_dim = output_shape[broadcast_dimensions[i]];

    if (input_dim != 1 && input_dim != output_dim) {
      throw std::invalid_argument(
          "Input dimension " + std::to_string(i) + " with size " + std::to_string(input_dim) +
          " must be either 1 or match output dimension " + std::to_string(broadcast_dimensions[i]) +
          " with size " + std::to_string(output_dim));
    }
  }
}

// Template constructor implementation
template <typename ElementType>
BroadcastOpBuilder::BroadcastOpBuilder(const std::vector<int64_t>& input_shape,
                                       const std::vector<int64_t>& output_shape,
                                       const std::vector<int64_t>& broadcast_dimensions,
                                       ElementType element_type, bool enable_verification)
    : OpModuleBuilder(enable_verification),
      input_shape_(input_shape),
      output_shape_(output_shape),
      broadcast_dimensions_(broadcast_dimensions),
      element_type_([&]() {
        if constexpr (std::is_same_v<ElementType, std::string> ||
                      std::is_same_v<ElementType, const char*>) {
          return type_utils::stringToMlirType(getBuilder(), element_type);
        } else {
          return element_type;
        }
      }()) {
  TORCH_NEURONX_DEBUG("BroadcastOpBuilder: Initializing");

  // Validate inputs
  if (input_shape_.empty()) {
    throw std::invalid_argument("input_shape cannot be empty");
  }

  if (output_shape_.empty()) {
    throw std::invalid_argument("output_shape cannot be empty");
  }

  if (broadcast_dimensions_.empty()) {
    throw std::invalid_argument("broadcast_dimensions cannot be empty");
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
  validateShapeCompatibility(input_shape_, output_shape_, broadcast_dimensions_);

  TORCH_NEURONX_DEBUG("BroadcastOpBuilder: Initialized successfully");
}

// Explicit template instantiations
template BroadcastOpBuilder::BroadcastOpBuilder(const std::vector<int64_t>&,
                                                const std::vector<int64_t>&,
                                                const std::vector<int64_t>&, std::string, bool);

template BroadcastOpBuilder::BroadcastOpBuilder(const std::vector<int64_t>&,
                                                const std::vector<int64_t>&,
                                                const std::vector<int64_t>&, const char*, bool);

template BroadcastOpBuilder::BroadcastOpBuilder(const std::vector<int64_t>&,
                                                const std::vector<int64_t>&,
                                                const std::vector<int64_t>&, mlir::Type, bool);

mlir::RankedTensorType BroadcastOpBuilder::getInputType() {
  return mlir::RankedTensorType::get(input_shape_, element_type_);
}

mlir::RankedTensorType BroadcastOpBuilder::getOutputType() {
  return mlir::RankedTensorType::get(output_shape_, element_type_);
}

mlir::Value BroadcastOpBuilder::buildOperation(mlir::func::FuncOp func, mlir::OpBuilder& builder) {
  TORCH_NEURONX_DEBUG("BroadcastOpBuilder: Building broadcast_in_dim operation");

  // Get function argument
  auto arg = func.getBody().front().getArgument(0);

  // Get output type
  auto outputType = getOutputType();

  // Create broadcast dimensions attribute
  auto broadcastDimsAttr = builder.getDenseI64ArrayAttr(broadcast_dimensions_);

  // Create broadcast_in_dim operation
  auto broadcastOp = builder.create<mlir::stablehlo::BroadcastInDimOp>(
      builder.getUnknownLoc(), outputType, arg, broadcastDimsAttr);

  TORCH_NEURONX_DEBUG("BroadcastOpBuilder: Broadcast operation created successfully");

  return broadcastOp.getResult();
}

}  // namespace torch_neuronx
