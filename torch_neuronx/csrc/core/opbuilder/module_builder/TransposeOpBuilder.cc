#include "TransposeOpBuilder.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/opbuilder/utility/TypeUtils.h"

namespace torch_neuronx {

void TransposeOpBuilder::validatePermutation(const std::vector<int64_t>& permutation) {
  if (permutation.empty()) {
    throw std::invalid_argument("Permutation cannot be empty");
  }

  std::unordered_set<int64_t> seen;
  for (size_t i = 0; i < permutation.size(); ++i) {
    if (permutation[i] < 0 || permutation[i] >= static_cast<int64_t>(permutation.size())) {
      throw std::invalid_argument("Permutation contains invalid index " +
                                  std::to_string(permutation[i]) + " (must be in range [0, " +
                                  std::to_string(permutation.size()) + "))");
    }
    if (seen.count(permutation[i]) > 0) {
      throw std::invalid_argument("Permutation contains duplicate index " +
                                  std::to_string(permutation[i]));
    }
    seen.insert(permutation[i]);
  }
}

std::vector<int64_t> TransposeOpBuilder::computeOutputShape(
    const std::vector<int64_t>& input_shape, const std::vector<int64_t>& permutation) {
  std::vector<int64_t> output_shape(input_shape.size());
  for (size_t i = 0; i < permutation.size(); ++i) {
    output_shape[i] = input_shape[permutation[i]];
  }
  return output_shape;
}

// Template constructor implementation
template <typename ElementType>
TransposeOpBuilder::TransposeOpBuilder(const std::vector<int64_t>& permutation,
                                       const std::vector<int64_t>& input_shape,
                                       ElementType element_type, bool enable_verification)
    : OpModuleBuilder(enable_verification),
      permutation_(permutation),
      input_shape_(input_shape),
      element_type_([&]() {
        if constexpr (std::is_same_v<ElementType, std::string> ||
                      std::is_same_v<ElementType, const char*>) {
          return type_utils::stringToMlirType(getBuilder(), element_type);
        } else {
          return element_type;
        }
      }()) {
  TORCH_NEURONX_DEBUG("TransposeOpBuilder: Initializing");

  // Validate inputs
  if (input_shape_.empty()) {
    throw std::invalid_argument("input_shape cannot be empty");
  }

  if (permutation_.size() != input_shape_.size()) {
    throw std::invalid_argument("Permutation size (" + std::to_string(permutation_.size()) +
                                ") must match input_shape rank (" +
                                std::to_string(input_shape_.size()) + ")");
  }

  validatePermutation(permutation_);

  for (size_t i = 0; i < input_shape_.size(); ++i) {
    if (input_shape_[i] <= 0) {
      throw std::invalid_argument("input_shape dimensions must be positive (got " +
                                  std::to_string(input_shape_[i]) + " at index " +
                                  std::to_string(i) + ")");
    }
  }

  // Compute output shape
  output_shape_ = computeOutputShape(input_shape_, permutation_);

  TORCH_NEURONX_DEBUG("TransposeOpBuilder: Initialized successfully");
}

// Explicit template instantiations
template TransposeOpBuilder::TransposeOpBuilder(const std::vector<int64_t>&,
                                                const std::vector<int64_t>&, std::string, bool);

template TransposeOpBuilder::TransposeOpBuilder(const std::vector<int64_t>&,
                                                const std::vector<int64_t>&, const char*, bool);

template TransposeOpBuilder::TransposeOpBuilder(const std::vector<int64_t>&,
                                                const std::vector<int64_t>&, mlir::Type, bool);

mlir::RankedTensorType TransposeOpBuilder::getInputType() {
  return mlir::RankedTensorType::get(input_shape_, element_type_);
}

mlir::RankedTensorType TransposeOpBuilder::getOutputType() {
  return mlir::RankedTensorType::get(output_shape_, element_type_);
}

mlir::Value TransposeOpBuilder::buildOperation(mlir::func::FuncOp func, mlir::OpBuilder& builder) {
  TORCH_NEURONX_DEBUG("TransposeOpBuilder: Building transpose operation");

  // Get function argument
  auto arg = func.getBody().front().getArgument(0);

  // Create permutation attribute
  auto permutationAttr = builder.getDenseI64ArrayAttr(permutation_);

  // Get output type
  auto outputType = getOutputType();

  // Create transpose operation
  auto transposeOp = builder.create<mlir::stablehlo::TransposeOp>(builder.getUnknownLoc(),
                                                                  outputType, arg, permutationAttr);

  TORCH_NEURONX_DEBUG("TransposeOpBuilder: Transpose operation created successfully");

  return transposeOp.getResult();
}

}  // namespace torch_neuronx
