#pragma once

#include <ATen/Tensor.h>

#include <cstdint>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace c10_neuron {
namespace lazy {

// Transformation type enum
enum class TransformationType { TRANSPOSE, RESHAPE, BROADCAST, SLICE, NONE };

// Enhanced structure with complete transformation metadata and validation
struct TensorTransformation {
  TransformationType type;

  // Complete transformation specification - NO COMPUTATION NEEDED by consumers
  std::vector<int64_t> input_shape;   // Shape BEFORE this transformation
  std::vector<int64_t> output_shape;  // Shape AFTER this transformation
  std::string element_type;           // Element type (e.g., "bf16", "f32")

  // Permutation state (for TRANSPOSE) - provided by creator, no calculation needed
  std::vector<int64_t> current_perm;  // Permutation state BEFORE this transformation
  std::vector<int64_t> target_perm;   // Permutation state AFTER this transformation

  // Type-specific parameters
  std::vector<int64_t> params;  // For TRANSPOSE: permutation [0,2,1,3]
                                // For RESHAPE: empty (output_shape is sufficient)
                                // For SLICE: [start0, end0, start1, end1, ...]

  // Constructor with automatic validation
  TensorTransformation(TransformationType t, std::vector<int64_t> in_shape,
                       std::vector<int64_t> out_shape, std::string elem_type,
                       std::vector<int64_t> p = {}, std::vector<int64_t> curr_perm = {},
                       std::vector<int64_t> targ_perm = {})
      : type(t),
        input_shape(std::move(in_shape)),
        output_shape(std::move(out_shape)),
        element_type(std::move(elem_type)),
        current_perm(std::move(curr_perm)),
        target_perm(std::move(targ_perm)),
        params(std::move(p)) {
    // Validate on construction - catches errors immediately
    Validate();
  }

  // Validation method - verifies transformation is legal
  void Validate() const {
    // 1. Check shapes are non-empty and positive
    if (input_shape.empty()) {
      throw std::invalid_argument("TensorTransformation: input_shape cannot be empty");
    }
    if (output_shape.empty()) {
      throw std::invalid_argument("TensorTransformation: output_shape cannot be empty");
    }

    for (size_t i = 0; i < input_shape.size(); ++i) {
      if (input_shape[i] <= 0) {
        throw std::invalid_argument("TensorTransformation: input_shape[" + std::to_string(i) +
                                    "] = " + std::to_string(input_shape[i]) + " must be positive");
      }
    }

    for (size_t i = 0; i < output_shape.size(); ++i) {
      if (output_shape[i] <= 0) {
        throw std::invalid_argument("TensorTransformation: output_shape[" + std::to_string(i) +
                                    "] = " + std::to_string(output_shape[i]) + " must be positive");
      }
    }

    // 2. Check element type is non-empty
    if (element_type.empty()) {
      throw std::invalid_argument("TensorTransformation: element_type cannot be empty");
    }

    // 3. Type-specific validation
    switch (type) {
      case TransformationType::TRANSPOSE:
        ValidateTranspose();
        break;
      case TransformationType::RESHAPE:
        ValidateReshape();
        break;
      case TransformationType::BROADCAST:
        ValidateBroadcast();
        break;
      case TransformationType::SLICE:
        ValidateSlice();
        break;
      default:
        throw std::invalid_argument("TensorTransformation: Unknown transformation type");
    }
  }

 private:
  void ValidateTranspose() const {
    // Check permutation is provided
    if (params.empty()) {
      throw std::invalid_argument("TRANSPOSE: permutation (params) cannot be empty");
    }

    // Check permutation size matches input rank
    if (params.size() != input_shape.size()) {
      throw std::invalid_argument("TRANSPOSE: permutation size (" + std::to_string(params.size()) +
                                  ") must equal input rank (" + std::to_string(input_shape.size()) +
                                  ")");
    }

    // Check output rank matches input rank
    if (output_shape.size() != input_shape.size()) {
      throw std::invalid_argument("TRANSPOSE: output rank (" + std::to_string(output_shape.size()) +
                                  ") must equal input rank (" + std::to_string(input_shape.size()) +
                                  ")");
    }

    // Check permutation contains unique values in range [0, rank)
    std::unordered_set<int64_t> seen;
    for (size_t i = 0; i < params.size(); ++i) {
      int64_t perm_val = params[i];

      if (perm_val < 0 || perm_val >= static_cast<int64_t>(params.size())) {
        throw std::invalid_argument("TRANSPOSE: permutation[" + std::to_string(i) +
                                    "] = " + std::to_string(perm_val) + " is out of range [0, " +
                                    std::to_string(params.size()) + ")");
      }

      if (seen.count(perm_val)) {
        throw std::invalid_argument("TRANSPOSE: permutation contains duplicate value " +
                                    std::to_string(perm_val));
      }
      seen.insert(perm_val);
    }

    // Check output shape matches permutation applied to input shape
    for (size_t i = 0; i < output_shape.size(); ++i) {
      int64_t expected = input_shape[params[i]];
      if (output_shape[i] != expected) {
        std::stringstream ss;
        ss << "TRANSPOSE: output_shape[" << i << "] = " << output_shape[i] << " but expected "
           << expected << " (input_shape[" << params[i] << "])";
        throw std::invalid_argument(ss.str());
      }
    }
  }

  void ValidateReshape() const {
    // Check element count is preserved
    int64_t input_elements = ComputeElementCount(input_shape);
    int64_t output_elements = ComputeElementCount(output_shape);

    if (input_elements != output_elements) {
      throw std::invalid_argument("RESHAPE: element count mismatch. Input has " +
                                  std::to_string(input_elements) + " elements, output has " +
                                  std::to_string(output_elements) + " elements");
    }
  }

  void ValidateBroadcast() const {
    // Broadcast validation: output rank >= input rank
    if (output_shape.size() < input_shape.size()) {
      throw std::invalid_argument("BROADCAST: output rank (" + std::to_string(output_shape.size()) +
                                  ") must be >= input rank (" + std::to_string(input_shape.size()) +
                                  ")");
    }

    // Check broadcast rules (from right to left)
    size_t input_idx = input_shape.size();
    size_t output_idx = output_shape.size();

    while (input_idx > 0 && output_idx > 0) {
      --input_idx;
      --output_idx;

      int64_t in_dim = input_shape[input_idx];
      int64_t out_dim = output_shape[output_idx];

      // Valid broadcast: dim is 1 or dims match
      if (in_dim != 1 && in_dim != out_dim) {
        throw std::invalid_argument("BROADCAST: incompatible dimension at position " +
                                    std::to_string(input_idx) +
                                    ". Input dim = " + std::to_string(in_dim) +
                                    ", output dim = " + std::to_string(out_dim));
      }
    }
  }

  void ValidateSlice() const {
    // Slice validation: output rank == input rank
    if (output_shape.size() != input_shape.size()) {
      throw std::invalid_argument("SLICE: output rank must equal input rank");
    }

    // Each output dimension <= corresponding input dimension
    for (size_t i = 0; i < input_shape.size(); ++i) {
      if (output_shape[i] > input_shape[i]) {
        throw std::invalid_argument(
            "SLICE: output_shape[" + std::to_string(i) + "] = " + std::to_string(output_shape[i]) +
            " exceeds input_shape[" + std::to_string(i) + "] = " + std::to_string(input_shape[i]));
      }
    }
  }

  static int64_t ComputeElementCount(const std::vector<int64_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
  }
};

// Structure to track state across transformations
// This is updated by materializers as they process transformations
struct TransformationState {
  std::vector<int64_t> current_shape;  // Current tensor shape
  std::vector<int64_t> current_perm;   // Current permutation state
  std::string element_type;            // Element type string (e.g., "f32", "bf16")

  TransformationState(const std::vector<int64_t>& shape, const std::vector<int64_t>& perm,
                      const std::string& elem_type)
      : current_shape(shape), current_perm(perm), element_type(elem_type) {}
};

// Result of attempting to create a transformation from a tensor
struct TransformationCreationResult {
  bool is_supported;                    // Whether pattern was detected
  TensorTransformation transformation;  // The transformation metadata
  at::Tensor processed_input;           // Modified input tensor (e.g., base tensor for slice)
  std::string pattern_name;             // Name of detected pattern for logging
};

// ============================================================================
// Type-specific transformation parameters
// ============================================================================

// Parameters for transpose transformations
struct TransposeParams {
  std::vector<int64_t> current_perm;  // Current permutation state
  std::vector<int64_t> target_perm;   // Target permutation to apply
};

// Parameters for slice transformations
struct SliceParams {
  std::vector<int64_t> start_indices;  // Starting indices for each dimension
  std::vector<int64_t> end_indices;    // Ending indices for each dimension
  std::vector<int64_t> strides;        // Stride for each dimension
};

// Parameters for broadcast transformations
struct BroadcastParams {
  std::vector<int64_t> broadcast_dims;  // Dimensions to broadcast
};

// ============================================================================
// TransformationTask - Pure data structure
// ============================================================================

// Represents a single transformation to be applied to an input tensor
// This is a self-contained data structure with no references to tensors
// It carries state information from previous transformations for proper MLIR generation
struct TransformationTask {
  size_t input_index;                 // Index of input tensor this transforms
  TransformationType type;            // Type of transformation
  std::vector<int64_t> input_shape;   // Input shape for this transformation
  std::vector<int64_t> output_shape;  // Output shape after transformation

  // State carried from previous transformations
  std::string element_type;           // Element type (f32, f16, bf16, i32, etc.)
  std::vector<int64_t> current_perm;  // Current permutation state (for transpose tracking)

  // Type-specific parameters
  // Use std::monostate for RESHAPE (no additional params needed)
  std::variant<std::monostate, TransposeParams, SliceParams, BroadcastParams> params;
};

// ============================================================================
// TransformationHandler - Base class for transformation handlers
// ============================================================================

// Base class for transformation handlers
class TransformationHandler {
 public:
  virtual ~TransformationHandler() = default;

  // Virtual methods to be implemented by derived handlers
  virtual size_t FindGroupEnd(const std::vector<TensorTransformation>& transformations,
                              size_t start_index, const TransformationState& state) const = 0;

  virtual TransformationTask ProcessGroup(size_t input_index, TransformationState& state,
                                          const std::vector<TensorTransformation>& group,
                                          const std::string& op_name) const = 0;
};

}  // namespace lazy
}  // namespace c10_neuron
