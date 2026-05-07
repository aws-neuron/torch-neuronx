#include "SliceOpBuilder.h"

#include <algorithm>
#include <stdexcept>

#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/opbuilder/utility/TypeUtils.h"

namespace torch_neuronx {

void SliceOpBuilder::validateSliceParameters(const std::vector<int64_t>& input_shape,
                                             const std::vector<int64_t>& start_indices,
                                             const std::vector<int64_t>& limit_indices,
                                             const std::vector<int64_t>& strides) {
  size_t rank = input_shape.size();

  // Check that all vectors have the same size
  if (start_indices.size() != rank) {
    throw std::invalid_argument("start_indices size (" + std::to_string(start_indices.size()) +
                                ") must match input_shape rank (" + std::to_string(rank) + ")");
  }

  if (limit_indices.size() != rank) {
    throw std::invalid_argument("limit_indices size (" + std::to_string(limit_indices.size()) +
                                ") must match input_shape rank (" + std::to_string(rank) + ")");
  }

  if (strides.size() != rank) {
    throw std::invalid_argument("strides size (" + std::to_string(strides.size()) +
                                ") must match input_shape rank (" + std::to_string(rank) + ")");
  }

  // Validate each dimension
  for (size_t i = 0; i < rank; ++i) {
    // Check stride is positive
    if (strides[i] <= 0) {
      throw std::invalid_argument("stride at index " + std::to_string(i) +
                                  " must be positive (got " + std::to_string(strides[i]) + ")");
    }

    // Check start index is within bounds
    if (start_indices[i] < 0 || start_indices[i] > input_shape[i]) {
      throw std::invalid_argument("start_indices at index " + std::to_string(i) +
                                  " must be in range [0, " + std::to_string(input_shape[i]) +
                                  "] (got " + std::to_string(start_indices[i]) + ")");
    }

    // Check limit index is within bounds
    if (limit_indices[i] < 0 || limit_indices[i] > input_shape[i]) {
      throw std::invalid_argument("limit_indices at index " + std::to_string(i) +
                                  " must be in range [0, " + std::to_string(input_shape[i]) +
                                  "] (got " + std::to_string(limit_indices[i]) + ")");
    }

    // Check that start < limit
    if (start_indices[i] > limit_indices[i]) {
      throw std::invalid_argument(
          "start_indices[" + std::to_string(i) + "] (" + std::to_string(start_indices[i]) +
          ") must be less than or equal to limit_indices[" + std::to_string(i) + "] (" +
          std::to_string(limit_indices[i]) + ")");
    }
  }
}

std::vector<int64_t> SliceOpBuilder::computeOutputShape(const std::vector<int64_t>& input_shape,
                                                        const std::vector<int64_t>& start_indices,
                                                        const std::vector<int64_t>& limit_indices,
                                                        const std::vector<int64_t>& strides) {
  std::vector<int64_t> output_shape(input_shape.size());
  for (size_t i = 0; i < input_shape.size(); ++i) {
    // Output size = ceil((limit - start) / stride)
    int64_t slice_size = limit_indices[i] - start_indices[i];
    output_shape[i] = (slice_size + strides[i] - 1) / strides[i];
  }
  return output_shape;
}

// Template constructor implementation
template <typename ElementType>
SliceOpBuilder::SliceOpBuilder(const std::vector<int64_t>& input_shape,
                               const std::vector<int64_t>& start_indices,
                               const std::vector<int64_t>& limit_indices,
                               const std::vector<int64_t>& strides, ElementType element_type,
                               bool enable_verification)
    : OpModuleBuilder(enable_verification),
      input_shape_(input_shape),
      start_indices_(start_indices),
      limit_indices_(limit_indices),
      strides_(strides),
      element_type_([&]() {
        if constexpr (std::is_same_v<ElementType, std::string> ||
                      std::is_same_v<ElementType, const char*>) {
          return type_utils::stringToMlirType(getBuilder(), element_type);
        } else {
          return element_type;
        }
      }()) {
  TORCH_NEURONX_DEBUG("SliceOpBuilder: Initializing");

  // Validate inputs
  if (input_shape_.empty()) {
    throw std::invalid_argument("input_shape cannot be empty");
  }

  for (size_t i = 0; i < input_shape_.size(); ++i) {
    if (input_shape_[i] <= 0) {
      throw std::invalid_argument("input_shape dimensions must be positive (got " +
                                  std::to_string(input_shape_[i]) + " at index " +
                                  std::to_string(i) + ")");
    }
  }

  // Validate slice parameters
  validateSliceParameters(input_shape_, start_indices_, limit_indices_, strides_);

  // Compute output shape
  output_shape_ = computeOutputShape(input_shape_, start_indices_, limit_indices_, strides_);

  TORCH_NEURONX_DEBUG("SliceOpBuilder: Initialized successfully");
}

// Explicit template instantiations
template SliceOpBuilder::SliceOpBuilder(const std::vector<int64_t>&, const std::vector<int64_t>&,
                                        const std::vector<int64_t>&, const std::vector<int64_t>&,
                                        std::string, bool);

template SliceOpBuilder::SliceOpBuilder(const std::vector<int64_t>&, const std::vector<int64_t>&,
                                        const std::vector<int64_t>&, const std::vector<int64_t>&,
                                        const char*, bool);

template SliceOpBuilder::SliceOpBuilder(const std::vector<int64_t>&, const std::vector<int64_t>&,
                                        const std::vector<int64_t>&, const std::vector<int64_t>&,
                                        mlir::Type, bool);

mlir::RankedTensorType SliceOpBuilder::getInputType() {
  return mlir::RankedTensorType::get(input_shape_, element_type_);
}

mlir::RankedTensorType SliceOpBuilder::getOutputType() {
  return mlir::RankedTensorType::get(output_shape_, element_type_);
}

mlir::Value SliceOpBuilder::buildOperation(mlir::func::FuncOp func, mlir::OpBuilder& builder) {
  TORCH_NEURONX_DEBUG("SliceOpBuilder: Building slice operation");

  // Get function argument
  auto arg = func.getBody().front().getArgument(0);

  // Create attributes for slice operation
  auto startIndicesAttr = builder.getDenseI64ArrayAttr(start_indices_);
  auto limitIndicesAttr = builder.getDenseI64ArrayAttr(limit_indices_);
  auto stridesAttr = builder.getDenseI64ArrayAttr(strides_);

  // Get output type
  auto outputType = getOutputType();

  // Create slice operation
  auto sliceOp = builder.create<mlir::stablehlo::SliceOp>(
      builder.getUnknownLoc(), outputType, arg, startIndicesAttr, limitIndicesAttr, stridesAttr);

  TORCH_NEURONX_DEBUG("SliceOpBuilder: Slice operation created successfully");

  return sliceOp.getResult();
}

}  // namespace torch_neuronx
