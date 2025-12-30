#include "torch_neuronx/csrc/core/lazy_materialization/Transformations.h"

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/lazy_materialization/MlirGenerators.h"
#include "torch_neuronx/csrc/core/lazy_materialization/TypeUtils.h"

namespace c10_neuron {
namespace lazy {

// ============================================================================
// PATTERN DETECTION HELPER FUNCTIONS (from ContiguousTranspose/ContiguousSlice)
// ============================================================================

namespace {

// Helper: Compute row-major strides for a shape
std::vector<int64_t> ComputeRowMajorStrides(const std::vector<int64_t>& shape) {
  if (shape.empty()) {
    return {};
  }

  std::vector<int64_t> strides;
  strides.reserve(shape.size());
  int64_t running_stride = 1;

  // Build strides from right to left
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides.insert(strides.begin(), running_stride);
    running_stride *= shape[i];
  }

  return strides;
}

// Detect transpose pattern from tensor strides
std::optional<std::vector<int64_t>> detectTransposePattern(const at::Tensor& tensor) {
  auto shape = tensor.sizes();
  auto strides = tensor.strides();
  int64_t n = shape.size();

  if (n == 0) {
    TORCH_NEURONX_DEBUG("Transpose detection rejected: empty tensor (n=0)");
    return std::nullopt;
  }

  // Reject negative or zero strides (reversals not supported)
  for (int64_t i = 0; i < n; ++i) {
    if (strides[i] <= 0) {
      TORCH_NEURONX_DEBUG("Transpose detection rejected: negative or zero stride", "dim=", i,
                          "stride=", strides[i], "shape=", shape, "strides=", strides);
      return std::nullopt;
    }
  }

  // Sort axes by stride magnitude (descending) to determine base order
  std::vector<int64_t> axes(n);
  for (int64_t i = 0; i < n; ++i) {
    axes[i] = i;
  }

  std::sort(axes.begin(), axes.end(), [&](int64_t i, int64_t j) {
    // Sort by stride (descending), then by span, then by index
    if (strides[i] != strides[j]) {
      return strides[i] > strides[j];
    }
    int64_t span_i = strides[i] * (shape[i] > 0 ? shape[i] : 1);
    int64_t span_j = strides[j] * (shape[j] > 0 ? shape[j] : 1);
    if (span_i != span_j) {
      return span_i > span_j;
    }
    return i < j;
  });

  // Compute expected row-major strides for base shape
  std::vector<int64_t> base_shape;
  base_shape.reserve(n);
  for (int64_t axis : axes) {
    base_shape.push_back(shape[axis]);
  }
  auto base_row_major_strides = ComputeRowMajorStrides(base_shape);

  // Map expected strides back to current axis indices and verify
  std::unordered_map<int64_t, int64_t> pos_in_base;
  for (size_t idx = 0; idx < axes.size(); ++idx) {
    pos_in_base[axes[idx]] = idx;
  }

  for (int64_t i = 0; i < n; ++i) {
    int64_t expected_stride = base_row_major_strides[pos_in_base[i]];
    if (strides[i] != expected_stride) {
      TORCH_NEURONX_DEBUG("Transpose detection rejected: stride mismatch", "dim=", i,
                          "expected_stride=", expected_stride, "actual_stride=", strides[i],
                          "shape=", shape, "strides=", strides, "axes=", axes);
      return std::nullopt;
    }
  }

  // Build permutation mapping current axis -> base axis index
  std::vector<int64_t> permutation;
  permutation.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    permutation.push_back(pos_in_base[i]);
  }

  // Check if this is identity permutation (already contiguous)
  bool is_identity = true;
  for (int64_t i = 0; i < n; ++i) {
    if (permutation[i] != i) {
      is_identity = false;
      break;
    }
  }

  if (is_identity) {
    TORCH_NEURONX_DEBUG("Transpose detection rejected: identity permutation (already contiguous)",
                        "shape=", shape, "strides=", strides);
    return std::nullopt;
  }

  TORCH_NEURONX_DEBUG("Transpose pattern detected successfully!", "shape=", shape,
                      "strides=", strides, "permutation=", permutation);
  return permutation;
}

// Information about detected slice pattern
struct SliceInfo {
  size_t slice_axis;      // Which dimension is sliced
  int64_t original_size;  // Original size of sliced dimension
  int64_t start_offset;   // Where slice starts
};

// Information about detected broadcast pattern
struct BroadcastInfo {
  std::vector<int64_t> base_shape;      // Shape with size-1 for broadcasted dims
  std::vector<int64_t> broadcast_dims;  // Which dimensions are broadcasted
};

// Helper for slice detection
bool VerifySingleSliceAtAxis(const std::vector<int64_t>& shape, const std::vector<int64_t>& strides,
                             size_t slice_axis, int64_t original_size) {
  // Reconstruct what the original shape would have been
  std::vector<int64_t> original_shape = shape;
  original_shape[slice_axis] = original_size;

  // Check if current strides match what they would be for the original shape
  std::vector<int64_t> expected_original_strides = ComputeRowMajorStrides(original_shape);

  return strides == expected_original_strides;
}

// Detect slice pattern from tensor
std::optional<SliceInfo> DetectSlicePattern(const at::Tensor& tensor) {
  // Early exit if tensor is already contiguous
  if (tensor.is_contiguous()) {
    return std::nullopt;
  }

  std::vector<int64_t> shape(tensor.sizes().begin(), tensor.sizes().end());
  std::vector<int64_t> strides(tensor.strides().begin(), tensor.strides().end());
  int64_t storage_offset = tensor.storage_offset();

  // Check stride relationships: stride[i] should equal shape[i+1] * stride[i+1]
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    if (axis == shape.size() - 1) {
      // Last dimension: stride must be 1 (element size) - can't be sliced further
      if (strides[axis] != 1) {
        return std::nullopt;
      }
    } else {
      int64_t expected_stride = shape[axis + 1] * strides[axis + 1];
      if (strides[axis + 1] == 0) {
        return std::nullopt;
      }

      if (strides[axis] != expected_stride) {
        // This dimension has wrong stride - check if it's due to slice
        // The stride should be based on a larger size for the next dimension
        if (strides[axis] % strides[axis + 1] == 0) {
          int64_t implied_next_size = strides[axis] / strides[axis + 1];
          if (implied_next_size > shape[axis + 1] &&
              VerifySingleSliceAtAxis(shape, strides, axis + 1, implied_next_size)) {
            // Calculate start offset in the sliced dimension
            int64_t start_offset = storage_offset / strides[axis + 1];
            return SliceInfo{axis + 1, implied_next_size, start_offset};
          }
        }
        return std::nullopt;
      }
    }
  }

  return std::nullopt;
}

// Detect broadcast pattern from tensor strides
std::optional<BroadcastInfo> DetectBroadcastPattern(const at::Tensor& tensor) {
  auto shape = tensor.sizes();
  auto strides = tensor.strides();
  int64_t n = shape.size();

  if (n == 0) {
    TORCH_NEURONX_DEBUG("Broadcast detection rejected: empty tensor (n=0)");
    return std::nullopt;
  }

  // Identify broadcasted dimensions (stride=0, size>1)
  std::vector<int64_t> broadcasted_dims;
  for (int64_t i = 0; i < n; ++i) {
    if (strides[i] < 0) {
      TORCH_NEURONX_DEBUG("Broadcast detection rejected: negative stride", "dim=", i,
                          "stride=", strides[i], "shape=", shape, "strides=", strides);
      return std::nullopt;
    } else if (strides[i] == 0 && shape[i] > 1) {
      broadcasted_dims.push_back(i);
    }
  }

  // No broadcast dimensions found
  if (broadcasted_dims.empty()) {
    TORCH_NEURONX_DEBUG("Broadcast detection rejected: no broadcast dimensions found",
                        "shape=", shape, "strides=", strides);
    return std::nullopt;
  }

  // Construct base shape with size-1 for broadcasted dims
  std::vector<int64_t> base_shape(shape.begin(), shape.end());
  for (int64_t dim : broadcasted_dims) {
    base_shape[dim] = 1;
  }

  TORCH_NEURONX_DEBUG("Broadcast pattern detected successfully!", "shape=", shape,
                      "strides=", strides, "broadcast_dims=", broadcasted_dims,
                      "base_shape=", base_shape);

  return BroadcastInfo{base_shape, broadcasted_dims};
}

}  // anonymous namespace

// ============================================================================
// CREATORS IMPLEMENTATION
// ============================================================================

namespace Creators {

std::optional<TransformationCreationResult> TryCreateTranspose(const at::Tensor& input,
                                                               const std::string& op_name,
                                                               size_t input_index) {
  TORCH_NEURONX_DEBUG("TryCreateTranspose called", "op=", op_name, "input_index=", input_index,
                      "is_contiguous=", input.is_contiguous(), "shape=", input.sizes(),
                      "strides=", input.strides());

  // Step 1: Try to detect transpose pattern using centralized function
  auto transpose_perm = detectTransposePattern(input);
  if (!transpose_perm.has_value()) {
    TORCH_NEURONX_DEBUG("TryCreateTranspose: detectTransposePattern returned nullopt",
                        "op=", op_name, "input_index=", input_index);
    return std::nullopt;  // Not a transpose pattern
  }

  TORCH_NEURONX_DEBUG("Detected supported transpose pattern", "op=", op_name,
                      "input_index=", input_index, "perm_size=", transpose_perm.value().size());

  // Step 2: Compute the PHYSICAL shape (memory layout), not logical shape
  const auto& perm = transpose_perm.value();
  std::vector<int64_t> logical_shape(input.sizes().begin(), input.sizes().end());

  // Create inverse permutation
  std::vector<int64_t> inverse_perm(perm.size());
  for (size_t j = 0; j < perm.size(); ++j) {
    inverse_perm[perm[j]] = j;
  }

  // Apply inverse permutation to get physical shape
  std::vector<int64_t> input_shape(logical_shape.size());
  for (size_t j = 0; j < inverse_perm.size(); ++j) {
    input_shape[j] = logical_shape[inverse_perm[j]];
  }

  TORCH_NEURONX_DEBUG("Computed physical shape from logical", "op=", op_name,
                      "input_index=", input_index, "logical_shape=", logical_shape,
                      "physical_shape=", input_shape, "perm=", perm);

  // Step 3: Compute output shape by applying permutation to input shape
  std::vector<int64_t> output_shape(input_shape.size());
  for (size_t j = 0; j < perm.size(); ++j) {
    output_shape[j] = input_shape[perm[j]];
  }

  // Step 4: Get element type string
  std::string element_type = ScalarTypeToElementTypeString(input.scalar_type());

  // Step 5: Create identity permutation for current_perm (starting state)
  std::vector<int64_t> identity_perm(perm.size());
  std::iota(identity_perm.begin(), identity_perm.end(), 0);

  // Step 6: Create transformation with complete metadata
  TensorTransformation transform(TransformationType::TRANSPOSE, input_shape, output_shape,
                                 element_type,
                                 perm,           // params: the permutation to apply
                                 identity_perm,  // current_perm: identity (starting state)
                                 perm);          // target_perm: the final permutation we want

  // Step 7: Create a base tensor with identity strides (like TryCreateSlice)
  // This ensures the tensor is not detected as non-contiguous input
  std::vector<int64_t> base_strides(input_shape.size());
  int64_t running_stride = 1;
  for (int i = static_cast<int>(input_shape.size()) - 1; i >= 0; --i) {
    base_strides[i] = running_stride;
    running_stride *= input_shape[i];
  }

  // Create base tensor with physical shape and identity strides
  at::Tensor base_tensor = input.as_strided(c10::IntArrayRef(input_shape),
                                            c10::IntArrayRef(base_strides), input.storage_offset());

  TORCH_NEURONX_DEBUG("Created base tensor for transpose", "op=", op_name,
                      "input_index=", input_index, "base_shape=", base_tensor.sizes(),
                      "base_strides=", base_tensor.strides(), "logical_shape=", input.sizes());

  return TransformationCreationResult{.is_supported = true,
                                      .transformation = transform,
                                      .processed_input = base_tensor,
                                      .pattern_name = "TRANSPOSE"};
}

std::optional<TransformationCreationResult> TryCreateSlice(const at::Tensor& input,
                                                           const std::string& op_name,
                                                           size_t input_index) {
  // Step 1: Try to detect slice pattern
  auto slice_info = DetectSlicePattern(input);
  if (!slice_info.has_value()) {
    return std::nullopt;  // Not a slice pattern
  }

  TORCH_NEURONX_DEBUG("Detected supported slice pattern", "op=", op_name,
                      "input_index=", input_index, "slice_axis=", slice_info->slice_axis);

  // Step 2: INPUT SHAPE: Physical memory layout (with original_size)
  std::vector<int64_t> input_shape(input.sizes().begin(), input.sizes().end());
  input_shape[slice_info->slice_axis] = slice_info->original_size;

  // Step 3: OUTPUT SHAPE: Logical sliced view
  std::vector<int64_t> output_shape(input.sizes().begin(), input.sizes().end());

  // Step 4: PARAMS: Slice indices for all dimensions
  std::vector<int64_t> slice_params;
  for (size_t dim = 0; dim < input_shape.size(); ++dim) {
    if (dim == slice_info->slice_axis) {
      slice_params.push_back(slice_info->start_offset);
      slice_params.push_back(slice_info->start_offset + output_shape[dim]);
    } else {
      slice_params.push_back(0);
      slice_params.push_back(input_shape[dim]);
    }
  }

  // Step 5: Get element type
  std::string element_type = ScalarTypeToElementTypeString(input.scalar_type());

  // Step 6: Create identity permutation for current_perm (slices don't change permutation)
  std::vector<int64_t> identity_perm(input_shape.size());
  std::iota(identity_perm.begin(), identity_perm.end(), 0);

  // Step 7: Create SLICE transformation
  TensorTransformation transform(TransformationType::SLICE, input_shape, output_shape, element_type,
                                 slice_params,    // params: slice indices
                                 identity_perm,   // current_perm: identity (slices don't permute)
                                 identity_perm);  // target_perm: stays identity

  // Step 8: Create a base view using as_strided with FULL shape
  std::vector<int64_t> base_strides(input_shape.size());
  int64_t running_stride = 1;
  for (int i = static_cast<int>(input_shape.size()) - 1; i >= 0; --i) {
    base_strides[i] = running_stride;
    running_stride *= input_shape[i];
  }

  std::vector<int64_t> input_strides(input.strides().begin(), input.strides().end());
  int64_t base_storage_offset =
      input.storage_offset() - slice_info->start_offset * input_strides[slice_info->slice_axis];

  at::Tensor base_tensor = input.as_strided(c10::IntArrayRef(input_shape),
                                            c10::IntArrayRef(base_strides), base_storage_offset);

  TORCH_NEURONX_DEBUG("Created base tensor for slice", "op=", op_name, "input_index=", input_index,
                      "base_shape=", base_tensor.sizes(), "slice_view_shape=", input.sizes());

  return TransformationCreationResult{.is_supported = true,
                                      .transformation = transform,
                                      .processed_input = base_tensor,
                                      .pattern_name = "SLICE"};
}

std::optional<TransformationCreationResult> TryCreateBroadcast(const at::Tensor& input,
                                                               const std::string& op_name,
                                                               size_t input_index) {
  TORCH_NEURONX_DEBUG("TryCreateBroadcast called", "op=", op_name, "input_index=", input_index,
                      "is_contiguous=", input.is_contiguous(), "shape=", input.sizes(),
                      "strides=", input.strides());

  // Step 1: Try to detect broadcast pattern
  auto broadcast_info = DetectBroadcastPattern(input);
  if (!broadcast_info.has_value()) {
    TORCH_NEURONX_DEBUG("TryCreateBroadcast: DetectBroadcastPattern returned nullopt",
                        "op=", op_name, "input_index=", input_index);
    return std::nullopt;  // Not a broadcast pattern
  }

  TORCH_NEURONX_DEBUG("Detected supported broadcast pattern", "op=", op_name,
                      "input_index=", input_index,
                      "num_broadcast_dims=", broadcast_info->broadcast_dims.size());

  // Step 2: INPUT SHAPE: Base shape with size-1 for broadcasted dimensions
  std::vector<int64_t> input_shape = broadcast_info->base_shape;

  // Step 3: OUTPUT SHAPE: Current tensor shape (after broadcast)
  std::vector<int64_t> output_shape(input.sizes().begin(), input.sizes().end());

  // Step 4: PARAMS: Create broadcast_dimensions mapping for StableHLO
  // StableHLO broadcast_in_dim requires a complete mapping [0,1,2,...,rank-1]
  // that maps each input dimension to its corresponding output dimension
  std::vector<int64_t> broadcast_params(input_shape.size());
  std::iota(broadcast_params.begin(), broadcast_params.end(), 0);

  TORCH_NEURONX_DEBUG("TryCreateBroadcast: Computed broadcast_dimensions",
                      "input_shape=", input_shape, "output_shape=", output_shape,
                      "broadcast_dimensions=", broadcast_params,
                      "originally_broadcasted_dims=", broadcast_info->broadcast_dims);

  // Step 5: Get element type
  std::string element_type = ScalarTypeToElementTypeString(input.scalar_type());

  // Step 6: Create identity permutation (broadcasts don't change permutation)
  std::vector<int64_t> identity_perm(input_shape.size());
  std::iota(identity_perm.begin(), identity_perm.end(), 0);

  // Step 7: Create BROADCAST transformation
  TensorTransformation transform(TransformationType::BROADCAST, input_shape, output_shape,
                                 element_type,
                                 broadcast_params,  // params: broadcasted dimension indices
                                 identity_perm,     // current_perm: identity
                                 identity_perm);    // target_perm: stays identity

  // Step 8: Create base view with size-1 for broadcasted dimensions
  // Keep the same strides, just change the shape
  std::vector<int64_t> input_strides(input.strides().begin(), input.strides().end());
  at::Tensor base_view = input.as_strided(c10::IntArrayRef(input_shape),
                                          c10::IntArrayRef(input_strides), input.storage_offset());

  TORCH_NEURONX_DEBUG("Created base view for broadcast", "op=", op_name,
                      "input_index=", input_index, "base_shape=", base_view.sizes(),
                      "broadcast_view_shape=", input.sizes());

  return TransformationCreationResult{.is_supported = true,
                                      .transformation = transform,
                                      .processed_input = base_view,
                                      .pattern_name = "BROADCAST"};
}

}  // namespace Creators

// ============================================================================
// HANDLERS IMPLEMENTATION
// ============================================================================

namespace Handlers {

// TransposeHandler
size_t TransposeHandler::FindGroupEnd(const std::vector<TensorTransformation>& transformations,
                                      size_t start_index, const TransformationState& state) const {
  // Group all consecutive TRANSPOSE transformations starting from start_index
  size_t end = start_index;
  while (end < transformations.size() &&
         transformations[end].type == TransformationType::TRANSPOSE) {
    end++;
  }
  return end;  // Returns one past the last transpose (like old code)
}

TransformationTask TransposeHandler::ProcessGroup(size_t input_index, TransformationState& state,
                                                  const std::vector<TensorTransformation>& group,
                                                  const std::string& op_name) const {
  TORCH_NEURONX_DEBUG("Processing TRANSPOSE group", "op=", op_name, "input_index=", input_index,
                      "group_size=", group.size());

  // Compose all permutations in the group
  std::vector<int64_t> composed_perm;
  std::vector<int64_t> input_shape;
  std::vector<int64_t> current_perm_state;
  std::string element_type;

  for (size_t i = 0; i < group.size(); ++i) {
    const auto& transform = group[i];

    if (i == 0) {
      // First transformation - use its data
      composed_perm = transform.params;
      input_shape = transform.input_shape;
      current_perm_state = transform.current_perm;
      element_type = transform.element_type;
    } else {
      // Compose with previous permutation
      std::vector<int64_t> new_perm(composed_perm.size());
      for (size_t j = 0; j < composed_perm.size(); ++j) {
        new_perm[j] = composed_perm[transform.params[j]];
      }
      composed_perm = std::move(new_perm);
    }
  }

  // Compute final output shape by applying composed permutation
  std::vector<int64_t> output_shape(composed_perm.size());
  for (size_t j = 0; j < composed_perm.size(); ++j) {
    output_shape[j] = input_shape[composed_perm[j]];
  }

  // Update state for next transformation in chain
  state.current_shape = output_shape;

  // Build TransposeParams
  TransposeParams params;
  params.current_perm = current_perm_state;
  params.target_perm = composed_perm;

  // Return TransformationTask
  return TransformationTask{.input_index = input_index,
                            .type = TransformationType::TRANSPOSE,
                            .input_shape = input_shape,
                            .output_shape = output_shape,
                            .element_type = element_type,
                            .current_perm = current_perm_state,
                            .params = params};
}

// SliceHandler
size_t SliceHandler::FindGroupEnd(const std::vector<TensorTransformation>& transformations,
                                  size_t start_index, const TransformationState& state) const {
  // Process SLICE operations individually for now (no grouping)
  // Return start_index + 1 to include just this one transformation
  return start_index + 1;
}

TransformationTask SliceHandler::ProcessGroup(size_t input_index, TransformationState& state,
                                              const std::vector<TensorTransformation>& group,
                                              const std::string& op_name) const {
  TORCH_NEURONX_DEBUG("Processing SLICE group", "op=", op_name, "input_index=", input_index,
                      "group_size=", group.size());

  // Should only be one slice in the group for now
  const auto& transform = group[0];

  // Extract slice parameters (start and end indices for each dimension)
  std::vector<int64_t> start_indices;
  std::vector<int64_t> end_indices;
  std::vector<int64_t> strides;

  for (size_t i = 0; i < transform.params.size(); i += 2) {
    start_indices.push_back(transform.params[i]);
    end_indices.push_back(transform.params[i + 1]);
    strides.push_back(1);  // Default stride of 1
  }

  // Update state for next transformation in chain
  state.current_shape = transform.output_shape;

  // Build SliceParams
  SliceParams params;
  params.start_indices = start_indices;
  params.end_indices = end_indices;
  params.strides = strides;

  // Return TransformationTask
  return TransformationTask{.input_index = input_index,
                            .type = TransformationType::SLICE,
                            .input_shape = transform.input_shape,
                            .output_shape = transform.output_shape,
                            .element_type = transform.element_type,
                            .current_perm = state.current_perm,
                            .params = params};
}

// BroadcastHandler
size_t BroadcastHandler::FindGroupEnd(const std::vector<TensorTransformation>& transformations,
                                      size_t start_index, const TransformationState& state) const {
  // Process BROADCAST operations individually for now (no grouping)
  // Return start_index + 1 to include just this one transformation
  return start_index + 1;
}

TransformationTask BroadcastHandler::ProcessGroup(size_t input_index, TransformationState& state,
                                                  const std::vector<TensorTransformation>& group,
                                                  const std::string& op_name) const {
  TORCH_NEURONX_DEBUG("Processing BROADCAST group", "op=", op_name, "input_index=", input_index,
                      "group_size=", group.size());

  // Should only be one broadcast in the group for now
  const auto& transform = group[0];

  // Extract broadcast dimensions from params
  std::vector<int64_t> broadcast_dims = transform.params;

  TORCH_NEURONX_DEBUG(
      "BroadcastHandler: Creating transformation task", "input_shape=", transform.input_shape,
      "output_shape=", transform.output_shape, "broadcast_dims=", broadcast_dims,
      "broadcast_dims.size()=", broadcast_dims.size(), "input_rank=", transform.input_shape.size());

  // Update state for next transformation in chain
  state.current_shape = transform.output_shape;

  // Build BroadcastParams
  BroadcastParams params;
  params.broadcast_dims = broadcast_dims;

  // Return TransformationTask
  return TransformationTask{.input_index = input_index,
                            .type = TransformationType::BROADCAST,
                            .input_shape = transform.input_shape,
                            .output_shape = transform.output_shape,
                            .element_type = transform.element_type,
                            .current_perm = state.current_perm,
                            .params = params};
}

}  // namespace Handlers

}  // namespace lazy
}  // namespace c10_neuron
