#include "EmptyOpBuilder.h"

#include <stdexcept>

#include "mlir/IR/BuiltinTypes.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/opbuilder/utility/TypeUtils.h"

namespace torch_neuronx {

// Template constructor implementation
template <typename ElementType>
EmptyOpBuilder::EmptyOpBuilder(const std::vector<int64_t>& shape, ElementType element_type,
                               bool enable_verification)
    : OpModuleBuilder(enable_verification), shape_(shape), element_type_([&]() {
        if constexpr (std::is_same_v<ElementType, std::string> ||
                      std::is_same_v<ElementType, const char*>) {
          return type_utils::stringToMlirType(getBuilder(), element_type);
        } else {
          return element_type;
        }
      }()) {
  TORCH_NEURONX_DEBUG("EmptyOpBuilder: Initializing", "shape_size=", shape_.size());

  // Validate inputs
  // Note: Empty shape is valid - represents scalar tensors (0-dimensional)
  // Note: Zero-sized dimensions are valid - represent empty tensors (e.g., [0, 5])
  for (size_t i = 0; i < shape_.size(); ++i) {
    if (shape_[i] < 0) {
      throw std::invalid_argument("shape dimensions must be non-negative (got " +
                                  std::to_string(shape_[i]) + " at index " + std::to_string(i) +
                                  ")");
    }
  }

  TORCH_NEURONX_DEBUG("EmptyOpBuilder: Initialized successfully", "shape=", shape_);
}

// Explicit template instantiations
template EmptyOpBuilder::EmptyOpBuilder(const std::vector<int64_t>&, std::string, bool);

template EmptyOpBuilder::EmptyOpBuilder(const std::vector<int64_t>&, const char*, bool);

template EmptyOpBuilder::EmptyOpBuilder(const std::vector<int64_t>&, mlir::Type, bool);

mlir::RankedTensorType EmptyOpBuilder::getInputType() {
  return mlir::RankedTensorType::get(shape_, element_type_);
}

mlir::RankedTensorType EmptyOpBuilder::getOutputType() {
  return mlir::RankedTensorType::get(shape_, element_type_);
}

mlir::Value EmptyOpBuilder::buildOperation(mlir::func::FuncOp func, mlir::OpBuilder& builder) {
  TORCH_NEURONX_DEBUG("EmptyOpBuilder: Building empty operation (identity passthrough)");

  // Get function argument
  auto arg = func.getBody().front().getArgument(0);

  TORCH_NEURONX_DEBUG("EmptyOpBuilder: Empty operation created successfully");

  // Simply return the input argument without any transformation
  return arg;
}

}  // namespace torch_neuronx
