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
// Uses max(dim, 1) so zero-sized dims don't collapse strides to 0,
// matching PyTorch CPU semantics (c10::contiguous_strides).
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
    running_stride *= std::max(shape[i], int64_t{1});
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
            // Reject multi-axis slices: if storage_offset >= strides[axis], the tensor
            // is also offset in a higher dimension, making this a multi-axis slice.
            if (storage_offset >= strides[axis]) {
              return std::nullopt;
            }
            // Calculate start offset in the sliced dimension only
            int64_t start_offset = storage_offset / strides[axis + 1];
            // Validate: offset must be perfectly explained by this single slice,
            // and the slice must fit within the original dimension bounds.
            if (storage_offset == start_offset * strides[axis + 1] &&
                start_offset + shape[axis + 1] <= implied_next_size) {
              // Storage bounds check: verify the implied base tensor fits within storage.
              // Without this check, strided-slice patterns (e.g., slice(step=3) producing
              // strides [18, 2, 1] on shape [3, 3, 2]) are misidentified as simple slices
              // with implied_next_size = stride[0]/stride[1] = 9, creating a base tensor
              // [3, 9, 2] = 54 elements that exceeds the 42-element storage.
              // The correct decomposition for strided slices is handled by TryCreateCompound.
              std::vector<int64_t> implied_shape = shape;
              implied_shape[axis + 1] = implied_next_size;
              int64_t implied_numel = 1;
              for (auto s : implied_shape) implied_numel *= s;
              int64_t storage_elements =
                  static_cast<int64_t>(tensor.storage().nbytes() / tensor.element_size());
              if (implied_numel > storage_elements) {
                TORCH_NEURONX_DEBUG(
                    "DetectSlicePattern rejected: implied base tensor exceeds storage",
                    " shape=", shape, " strides=", strides, " implied_shape=", implied_shape,
                    " implied_numel=", implied_numel, " storage_elements=", storage_elements,
                    " slice_axis=", axis + 1);
                return std::nullopt;
              }
              return SliceInfo{axis + 1, implied_next_size, start_offset};
            }
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

  // Step 2: Reject tensors with non-zero storage offset.
  // TryCreateTranspose only generates a TRANSPOSE prologue — it does NOT emit a SLICE
  // to account for the storage offset. This is fine when each input keeps its own
  // storage_offset in the TensorContext (NRT applies the offset before reading data).
  // However, input deduplication (ApplyInputDedup) sets storage_offset=0 for deduped
  // inputs, assuming the prologue MLIR handles the offset via SLICE operations.
  // If we allow non-zero offsets here, deduped ops like x+x read from offset 0 instead
  // of the correct offset, producing wrong results.
  // Deferring to TryCreateCompound (priority 50) is safe — it decomposes the offset
  // into a SLICE operation on a base tensor at offset 0.

  // TODO(mengchiy): Remove TryCreateTranspose and let TryCreateCompound handle all cases.
  // Then fix the base tensor shape inconsistency in compound by always computing the base
  // shape as the full storage extent from offset 0, ensuring all views of the same storage
  // get the same base shape.
  if (input.storage_offset() != 0) {
    TORCH_NEURONX_DEBUG("TryCreateTranspose: non-zero storage offset, deferring to compound",
                        "op=", op_name, "input_index=", input_index,
                        "storage_offset=", input.storage_offset());
    return std::nullopt;
  }

  TORCH_NEURONX_DEBUG("Detected supported transpose pattern", "op=", op_name,
                      "input_index=", input_index, "perm_size=", transpose_perm.value().size());

  // Step 3: Defense-in-depth — reject tensors whose physical shape doesn't cover
  // the full storage. The primary fix for FSDP shared-storage conflicts is in
  // ProcessInputTransformations (NeuronBindings.cpp), which detects when two inputs
  // share the same data_ptr() but have different base shapes and falls back to
  // contiguous_internal. This check is an additional safety layer that prevents
  // TryCreateTranspose from creating a small base tensor for partial-storage views,
  // deferring to TryCreateCompound which creates a more appropriate base shape.
  {
    int64_t numel = 1;
    for (auto s : input.sizes()) numel *= s;
    int64_t storage_elements =
        static_cast<int64_t>(input.storage().nbytes() / input.element_size());
    if (numel != storage_elements) {
      TORCH_NEURONX_DEBUG(
          "TryCreateTranspose: base tensor doesn't cover full storage, "
          "deferring to compound/contiguous_internal",
          "op=", op_name, "input_index=", input_index, "numel=", numel,
          "storage_elements=", storage_elements);
      return std::nullopt;
    }
  }

  // Step 4: Compute the PHYSICAL shape (memory layout), not logical shape
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

  // Step 5: Compute output shape by applying permutation to input shape
  std::vector<int64_t> output_shape(input_shape.size());
  for (size_t j = 0; j < perm.size(); ++j) {
    output_shape[j] = input_shape[perm[j]];
  }

  // Step 6: Get element type string
  std::string element_type = ScalarTypeToElementTypeString(input.scalar_type());

  // Step 7: Create identity permutation for current_perm (starting state)
  std::vector<int64_t> identity_perm(perm.size());
  std::iota(identity_perm.begin(), identity_perm.end(), 0);

  // Step 8: Create transformation with complete metadata
  TensorTransformation transform(TransformationType::TRANSPOSE, input_shape, output_shape,
                                 element_type,
                                 perm,           // params: the permutation to apply
                                 identity_perm,  // current_perm: identity (starting state)
                                 perm);          // target_perm: the final permutation we want

  // Step 9: Create a base tensor with identity strides (like TryCreateSlice)
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
                                      .transformations = {transform},
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

  // Reject if reconstructed base tensor would exceed 4GB — compiler limitation.
  // Matches Python ContiguousSliceMLIRImpl.can_handle() 4GB check.
  {
    // Divide first to avoid overflow: numel / size(axis) is exact (no remainder).
    int64_t base_numel =
        input.numel() / input.size(slice_info->slice_axis) * slice_info->original_size;
    int64_t base_bytes = base_numel * input.element_size();
    constexpr int64_t kMaxBytes = 4LL * 1024 * 1024 * 1024;
    if (base_bytes >= kMaxBytes) {
      TORCH_NEURONX_DEBUG("TryCreateSlice rejected: base tensor exceeds 4GB", "op=", op_name,
                          "input_index=", input_index, "base_bytes=", base_bytes);
      return std::nullopt;
    }
  }

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
                                      .transformations = {transform},
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

  // Step 1b: Verify non-broadcast dims are contiguous.
  // If not, this is a compound pattern (broadcast + transpose/slice) that
  // TryCreateCompound handles. Return nullopt to defer.
  {
    auto shape = input.sizes();
    auto strides = input.strides();
    std::vector<int64_t> expected_strides = ComputeRowMajorStrides(broadcast_info->base_shape);
    for (size_t i = 0; i < shape.size(); ++i) {
      // Skip stride validation for broadcast dimensions AND size-1 dimensions.
      // Dimensions of size 1 can have arbitrary mathematically meaningless strides.
      if (strides[i] == 0 || shape[i] == 1) continue;

      if (strides[i] != expected_strides[i]) {
        TORCH_NEURONX_DEBUG(
            "TryCreateBroadcast: non-broadcast dims not contiguous, "
            "deferring to TryCreateCompound",
            "dim=", i, "expected_stride=", expected_strides[i], "actual_stride=", strides[i]);
        return std::nullopt;
      }
    }
  }

  // Step 2: INPUT SHAPE: Base shape with size-1 for broadcasted dimensions
  std::vector<int64_t> input_shape = broadcast_info->base_shape;

  // Step 3: OUTPUT SHAPE: Current tensor shape (after broadcast)
  std::vector<int64_t> output_shape(input.sizes().begin(), input.sizes().end());

  // Step 4: PARAMS: Create broadcast_dimensions mapping for StableHLO
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
  at::Tensor base_view = input.as_strided(
      c10::IntArrayRef(input_shape), c10::IntArrayRef(input.strides()), input.storage_offset());

  TORCH_NEURONX_DEBUG("Created base view for broadcast", "op=", op_name,
                      "input_index=", input_index, "base_shape=", base_view.sizes(),
                      "broadcast_view_shape=", input.sizes());

  return TransformationCreationResult{.is_supported = true,
                                      .transformations = {transform},
                                      .processed_input = base_view,
                                      .pattern_name = "BROADCAST"};
}

// ============================================================================
// TryCreateCompound: Decompose arbitrary non-contiguous tensor views
//
// GOAL: Given a non-contiguous tensor (produced by any combination of
// transpose, slice, expand, select, unsqueeze, squeeze, permute, etc.),
// reconstruct a contiguous base tensor and a chain of SLICE → TRANSPOSE →
// BROADCAST operations that reproduce the original view.
//
// APPROACH (GCD-based stride factorization):
//   1. Separate broadcast dims (stride=0 or size=1) from non-broadcast dims.
//   2. Sort non-broadcast dims by stride descending — this gives the
//      "physical memory order" (outermost → innermost).
//   3. For each consecutive pair of sorted strides, use GCD to determine
//      the base dimension size. The key insight: if stride[i] is an integer
//      multiple of stride[i+1], then dimension i+1 has a well-defined extent
//      in the base tensor (= stride[i] / stride[i+1]). Any remainder becomes
//      a "step factor" (strided slice).
//   4. Decompose the storage offset into per-dimension start indices.
//   5. Build a contiguous base tensor with the computed shape.
//   6. Emit SLICE ops (one per dimension that needs slicing/stepping),
//      then TRANSPOSE (if physical order ≠ logical order), then BROADCAST
//      (if any dims were broadcast).
//
// LIMITATION:
// 1. The algorithm assumes each output dimension maps to exactly one base
//    dimension. This means:
//      - Supported: transpose, slice, expand, select, squeeze, unsqueeze, permute
//      - NOT supported: view/reshape that splits or merges dimensions (would need
//        RESHAPE in the chain), overlapping strides (unfold with step < size),
//        negative strides (flip)
// 2. Different views of the same shared storage (e.g., FSDP flat parameter buffers)
//    may produce different base tensor shapes through compound decomposition.
//    This is handled by:
//    a. ProcessInputTransformations (NeuronBindings.cpp): primary fix — detects
//       conflicting base shapes for inputs sharing the same data_ptr() and falls
//       back to contiguous_internal.
//    b. validateDepCompatibility (OpConcatUtils.cpp): defense-in-depth — returns
//       false for common input type mismatches during operation concatenation.
//    c. TryCreateTranspose Step 3: defense-in-depth — rejects partial-storage
//       views, deferring to TryCreateCompound.
//
// The GCD check (outer_stride_gcd[i] % base_strides[i+1] != 0) rejects
// patterns where stride ratios aren't integer multiples — this happens when
// view-split creates sub-dimensions of a single base dimension, making the
// strides coprime (e.g., strides [15, 5, 2, 1] → GCD(5,2)=1).
// ============================================================================
// Variable glossary for TryCreateCompound:
//   ndims              Total number of dimensions
//   num_non_broadcast  Count of non-broadcast (stride>0, size>1) dims
//   non_bc_dims        Original axis indices of non-broadcast dims
//   is_bc              Per-dim flag: true if broadcast (stride=0 or size=1)
//   sorted_nb_order    Indices into non_bc_dims sorted by stride descending
//   sorted_shape       Non-broadcast dim sizes in stride-descending order
//   sorted_strides     Non-broadcast dim strides in descending order
//   sorted_base_shape  Computed base shape for sorted non-broadcast dims
//   sorted_base_strides Computed contiguous strides for sorted base shape
//   step_factors       Per-dim step factor (1=contiguous, >1=strided slice)
//   outer_stride_gcd   Running GCD of strides from outermost to current dim
//   max_base_extent    Maximum valid base dimension size (from GCD)
//   min_required_extent Minimum physical footprint needed by a dim's slice
//   best_extent        Smallest valid divisor of max_base_extent for a dim
//   sorted_starts      Per-dim start index from offset decomposition
//   phys_to_logical    Maps physical dim position → original logical axis
//   logical_to_phys    Maps original logical axis → physical dim position
//   phys_base_shape    Full base shape (all dims, broadcast dims get size 1)
//   phys_starts        Per-dim start indices in physical order
//   phys_target_shape  Per-dim target sizes in physical order
//   phys_steps         Per-dim step factors in physical order
std::optional<TransformationCreationResult> TryCreateCompound(const at::Tensor& input,
                                                              const std::string& op_name,
                                                              size_t input_index) {
  auto shape = input.sizes();
  auto strides = input.strides();
  int64_t offset = input.storage_offset();
  int64_t ndims = shape.size();

  if (ndims == 0 || input.is_contiguous()) {
    TORCH_NEURONX_DEBUG("TryCreateCompound rejected: tensor is empty or contiguous", "op=", op_name,
                        "input_index=", input_index, "ndims=", ndims,
                        "is_contiguous=", input.is_contiguous());
    return std::nullopt;
  }

  // Step 1: Identify broadcast dims (stride=0 or size=1) vs non-broadcast dims.
  // Size-1 dims are treated as broadcast because their stride value is mathematically
  // irrelevant (the dimension contributes zero span regardless of stride value).
  // Including them in the non-broadcast set would pollute the GCD computation with
  // meaningless "junk" strides that cause false rejections.
  std::vector<int64_t> non_bc_dims;
  std::vector<bool> is_bc(ndims, false);
  for (int64_t i = 0; i < ndims; ++i) {
    if (strides[i] == 0 || shape[i] == 1) {
      is_bc[i] = true;
    } else {
      non_bc_dims.push_back(i);
    }
  }

  // Handle all-broadcast early return
  if (non_bc_dims.empty()) {
    std::string element_type = ScalarTypeToElementTypeString(input.scalar_type());
    at::Tensor base_tensor = input.as_strided({1}, {1}, offset);
    std::vector<TensorTransformation> transforms;
    std::vector<int64_t> all_ones_shape(ndims, 1);
    transforms.emplace_back(TransformationType::RESHAPE, std::vector<int64_t>{1}, all_ones_shape,
                            element_type);

    std::vector<int64_t> bc_params(ndims);
    std::iota(bc_params.begin(), bc_params.end(), 0);
    std::vector<int64_t> id_perm(ndims);
    std::iota(id_perm.begin(), id_perm.end(), 0);
    transforms.emplace_back(TransformationType::BROADCAST, all_ones_shape,
                            std::vector<int64_t>(shape.begin(), shape.end()), element_type,
                            bc_params, id_perm, id_perm);

    return TransformationCreationResult{true, std::move(transforms), base_tensor,
                                        "COMPOUND_ALL_BC"};
  }

  // Step 2: Sort non-broadcast dims by stride descending to get physical memory order.
  // The outermost dimension (largest stride) comes first.
  int64_t num_non_broadcast = non_bc_dims.size();
  std::vector<int64_t> sorted_nb_order(num_non_broadcast);
  std::iota(sorted_nb_order.begin(), sorted_nb_order.end(), 0);
  std::sort(sorted_nb_order.begin(), sorted_nb_order.end(), [&](int64_t a, int64_t b) {
    int64_t stride_a = strides[non_bc_dims[a]];
    int64_t stride_b = strides[non_bc_dims[b]];

    // Primary sort: stride magnitude descending
    if (stride_a != stride_b) return stride_a > stride_b;

    // Tie-breaker: maintain original relative axis order
    return non_bc_dims[a] < non_bc_dims[b];
  });

  int64_t total_dims = ndims;

  // Step 3: Extract sorted shapes/strides and factorize stride gaps.
  // For each consecutive pair of sorted strides, compute the base dimension size
  // using GCD. This determines how many contiguous elements each dimension spans
  // in the base tensor.
  std::vector<int64_t> sorted_shape;
  std::vector<int64_t> sorted_strides;
  for (int64_t idx : sorted_nb_order) {
    sorted_shape.push_back(shape[non_bc_dims[idx]]);
    sorted_strides.push_back(strides[non_bc_dims[idx]]);
  }

  std::vector<int64_t> sorted_base_shape = sorted_shape;
  std::vector<int64_t> step_factors(num_non_broadcast, 1);
  std::vector<int64_t> sorted_base_strides(num_non_broadcast, 1);

  if (num_non_broadcast > 0) {
    // Precompute running GCD of all strides from outermost to current position.
    // outer_stride_gcd[i] = GCD(sorted_strides[0], ..., sorted_strides[i]).
    // This bounds the maximum base dimension size at each level.
    //
    // Example: [B,S,H,D]=[1,8,4,8] transposed(1,2) then sliced [..., 4:]
    //   sorted_strides = [32, 8, 1] (S=8→stride 32, H=4→stride 8, D_half=4→stride 1)
    //   outer_stride_gcd = [32, GCD(32,8)=8, GCD(8,1)=1]
    std::vector<int64_t> outer_stride_gcd(num_non_broadcast);
    outer_stride_gcd[0] = sorted_strides[0];
    for (int i = 1; i < num_non_broadcast; ++i) {
      outer_stride_gcd[i] = std::gcd(outer_stride_gcd[i - 1], sorted_strides[i]);
    }

    // Innermost dimension: base stride is always 1, step is the actual stride.
    // Example: sorted_strides[-1]=1 → base_stride=1, step=1 (contiguous access)
    sorted_base_strides[num_non_broadcast - 1] = 1;
    step_factors[num_non_broadcast - 1] = sorted_strides[num_non_broadcast - 1];

    // Work from inner to outer: for each dimension, determine the base extent
    // and step factor by factorizing the stride gap using GCD.
    for (int64_t i = num_non_broadcast - 2; i >= 0; --i) {
      // GCD check: outer_stride_gcd[i] must be divisible by base_strides_sorted[i+1].
      // If not, the strides are coprime at this level — the layout can't be represented
      // as a rectangular grid (e.g., view-split after narrow creates this).
      if (outer_stride_gcd[i] % sorted_base_strides[i + 1] != 0) {
        TORCH_NEURONX_DEBUG("TryCreateCompound rejected: non-admissible layout (GCD check)",
                            "op=", op_name, "input_index=", input_index, "sorted_dim=", i,
                            "sorted_base_strides[i+1]=", sorted_base_strides[i + 1],
                            "outer_stride_gcd[i]=", outer_stride_gcd[i], "shape=", shape,
                            "strides=", strides, "sorted_shape=", sorted_shape,
                            "sorted_strides=", sorted_strides);
        return std::nullopt;
      }

      // max_base_extent: the maximum valid base dimension size, derived from GCD.
      // It represents how many contiguous elements the inner dimension could span
      // while still being compatible with all outer strides.
      // Example (i=0): outer_stride_gcd[0]=8, base_strides_sorted[1]=1
      //   → max_base_extent = 8/1 = 8 (dim 1 can have at most 8 elements in base)
      int64_t max_base_extent = outer_stride_gcd[i] / sorted_base_strides[i + 1];

      // min_required_extent: the minimum physical memory footprint needed by the
      // inner dimension's slice (accounting for step). Formula: (size-1)*step + 1.
      // Example: sorted_shape[1]=4, step_factors[1]=1
      //   → min_required_extent = (4-1)*1 + 1 = 4 (need 4 contiguous slots)
      //   Since 4 ≤ 8 (max_base_extent), this dimension fits.
      int64_t min_required_extent = (sorted_shape[i + 1] - 1) * step_factors[i + 1] + 1;

      // Find the smallest divisor of max_base_extent that is >= min_required_extent
      // and can physically accommodate the slice (including offset).
      // Uses O(√n) divisor enumeration instead of linear search.
      int64_t best_extent = -1;
      for (int64_t d = 1; d * d <= max_base_extent; ++d) {
        if (max_base_extent % d != 0) continue;
        // Check both d and max_base_extent/d
        for (int64_t candidate : {d, max_base_extent / d}) {
          if (candidate < min_required_extent) continue;
          int64_t start_idx = (offset / sorted_base_strides[i + 1]) % candidate;
          int64_t physical_span = start_idx + (sorted_shape[i + 1] - 1) * step_factors[i + 1] + 1;
          if (physical_span <= candidate) {
            if (best_extent == -1 || candidate < best_extent) {
              best_extent = candidate;
            }
          }
        }
      }
      if (best_extent == -1) {
        TORCH_NEURONX_DEBUG("TryCreateCompound rejected: no valid base factor found",
                            "op=", op_name, "input_index=", input_index, "sorted_dim=", i,
                            "max_base_extent=", max_base_extent,
                            "min_required_extent=", min_required_extent, "shape=", shape,
                            "strides=", strides, "offset=", offset);
        return std::nullopt;
      }

      // Update base shape and compute the base stride for the outer dimension.
      // base_strides_sorted[i] = product of all inner base extents.
      // Example (i=0): base_strides_sorted[0] = 1 * 4 = 4
      //   (dim 0 strides over rows of 4 elements each)
      sorted_base_shape[i + 1] = best_extent;
      sorted_base_strides[i] = sorted_base_strides[i + 1] * best_extent;

      // The remaining stride factor becomes the step (strided slice).
      // sorted_strides[i] = sorted_base_strides[i] * step_factors[i]
      // Example: sorted_strides[0]=32, sorted_base_strides[0]=4
      //   → step_factors[0] = 32/4 = 8 (take every 8th row → strided slice)
      step_factors[i] = sorted_strides[i] / sorted_base_strides[i];
    }
  }

  // Step 4: Offset decomposition — convert the flat storage offset into
  // per-dimension start indices using sorted_base_strides (NOT sorted_strides).
  // Example: offset=12, sorted_base_strides=[4,1] → starts=[3,0] (start at row 3)
  // TODO(mengchiy): Simplify by handling offset in flat space as an initial slice on the
  // base tensor (e.g., base[offset : offset + numel]), which would eliminate the
  // per-dimension decomposition and simplify the earlier min/max extent logic.
  int64_t remaining_offset = offset;
  std::vector<int64_t> sorted_starts(num_non_broadcast, 0);
  for (int64_t i = 0; i < num_non_broadcast; ++i) {
    if (sorted_base_strides[i] > 0 && remaining_offset >= sorted_base_strides[i]) {
      sorted_starts[i] = remaining_offset / sorted_base_strides[i];
      remaining_offset %= sorted_base_strides[i];
    }
  }

  if (remaining_offset != 0) {
    TORCH_NEURONX_DEBUG("TryCreateCompound rejected: offset decomposition failed", "op=", op_name,
                        "input_index=", input_index, "remaining_offset=", remaining_offset,
                        "original_offset=", offset, "shape=", shape, "strides=", strides);
    return std::nullopt;
  }

  // Enlarge outermost base shape to prevent out-of-bounds slicing
  if (num_non_broadcast > 0) {
    int64_t outermost_span = sorted_starts[0] + (sorted_shape[0] - 1) * step_factors[0] + 1;
    sorted_base_shape[0] = std::max(sorted_base_shape[0], outermost_span);
  }

  // Step 5: Build physical-to-logical dimension mapping.
  // Physical order = broadcast dims in original position + non-broadcast dims in stride-sorted
  // order. This mapping tells us how to transpose from physical layout back to logical layout.
  std::vector<int64_t> phys_to_logical;
  int sorted_dim_ptr = 0;
  for (int i = 0; i < ndims; ++i) {
    if (is_bc[i])
      phys_to_logical.push_back(i);
    else
      phys_to_logical.push_back(non_bc_dims[sorted_nb_order[sorted_dim_ptr++]]);
  }
  std::vector<int64_t> logical_to_phys(total_dims);
  for (int p = 0; p < total_dims; ++p) {
    logical_to_phys[phys_to_logical[p]] = p;
  }

  // Step 6: Construct the contiguous base tensor (physical bounding box).
  // Each non-broadcast dim gets the computed base extent; broadcast dims get size 1.
  std::vector<int64_t> phys_base_shape(total_dims);
  std::vector<int64_t> phys_starts(total_dims, 0);
  std::vector<int64_t> phys_target_shape(total_dims);

  int sorted_ptr = 0;
  for (int p = 0; p < total_dims; ++p) {
    int orig = phys_to_logical[p];
    if (orig < ndims && is_bc[orig]) {
      phys_base_shape[p] = 1;
      phys_starts[p] = 0;
      phys_target_shape[p] = 1;
    } else {
      int64_t logical_size = shape[orig];
      int64_t start_idx = sorted_starts[sorted_ptr];  // Native base coordinates
      int64_t step = step_factors[sorted_ptr];

      // Physical span accounts for the step
      int64_t physical_span = start_idx + (logical_size - 1) * step + 1;
      int64_t stride_gap_extent = sorted_base_shape[sorted_ptr];

      phys_base_shape[p] = std::max(stride_gap_extent, physical_span);
      phys_starts[p] = start_idx;
      phys_target_shape[p] = logical_size;
      sorted_ptr++;
    }
  }

  auto base_strides = ComputeRowMajorStrides(phys_base_shape);

  // Compute the base tensor's storage offset by subtracting the slice starts.
  // The base tensor starts earlier in storage; the SLICE ops will re-add the starts.
  // Example: offset=4 (from [..., D//2:]), phys_starts=[0,0,0,4], base_strides=[...,1]
  //   → base_offset = 4 - 4*1 = 0 (base tensor starts at storage position 0)
  int64_t base_offset = offset;
  for (int i = 0; i < total_dims; ++i) {
    base_offset -= phys_starts[i] * base_strides[i];
  }

  int64_t base_elements = 1;
  for (auto s : phys_base_shape) base_elements *= s;
  int64_t storage_elements = static_cast<int64_t>(input.storage().nbytes() / input.element_size());
  if (base_offset < 0 || base_offset + base_elements > storage_elements) {
    TORCH_NEURONX_DEBUG("TryCreateCompound rejected: base tensor out of storage bounds",
                        "op=", op_name, "input_index=", input_index, "base_offset=", base_offset,
                        "base_elements=", base_elements, "storage_elements=", storage_elements,
                        "shape=", shape, "strides=", strides, "offset=", offset);
    return std::nullopt;
  }

  // Note: The base tensor may not cover the full storage (e.g., FSDP shared buffers).
  // Type mismatches from different base shapes are handled gracefully by:
  // 1. ProcessInputTransformations (NeuronBindings.cpp): detects conflicting base shapes
  //    for inputs sharing the same data_ptr() and falls back to contiguous_internal.
  // 2. validateDepCompatibility (OpConcatUtils.cpp): returns false for common input type
  //    mismatches during operation concatenation, skipping concatenation gracefully.

  // Reject if base tensor would exceed 4GB — compiler limitation.
  // Matches Python ContiguousSliceMLIRImpl.can_handle() 4GB check.
  {
    int64_t base_numel = 1;
    for (auto s : phys_base_shape) base_numel *= s;
    int64_t base_bytes = base_numel * input.element_size();
    constexpr int64_t kMaxBytes = 4LL * 1024 * 1024 * 1024;
    if (base_bytes >= kMaxBytes) {
      TORCH_NEURONX_DEBUG("TryCreateCompound rejected: base tensor exceeds 4GB", "op=", op_name,
                          "input_index=", input_index, "base_bytes=", base_bytes);
      return std::nullopt;
    }
  }

  at::Tensor base_tensor = input.as_strided(c10::IntArrayRef(phys_base_shape),
                                            c10::IntArrayRef(base_strides), base_offset);

  // Step 7: Generate the transformation chain.
  // Emit one SLICE per dimension that needs slicing/stepping, then TRANSPOSE
  // (if physical order ≠ logical order), then BROADCAST (if any dims were broadcast).
  std::string element_type = ScalarTypeToElementTypeString(input.scalar_type());
  std::vector<TensorTransformation> transforms;
  std::vector<int64_t> cur_shape = phys_base_shape;
  std::vector<int64_t> id_perm_total(total_dims);
  std::iota(id_perm_total.begin(), id_perm_total.end(), 0);

  std::vector<int64_t> phys_steps(total_dims, 1);
  {
    int step_ptr = 0;
    for (int p = 0; p < total_dims; ++p) {
      int orig = phys_to_logical[p];
      if (orig < ndims && is_bc[orig]) {
        phys_steps[p] = 1;
      } else {
        phys_steps[p] = step_factors[step_ptr++];
      }
    }
  }

  for (int p = 0; p < total_dims; ++p) {
    if (phys_starts[p] > 0 || phys_base_shape[p] != phys_target_shape[p] || phys_steps[p] > 1) {
      std::vector<int64_t> next_shape = cur_shape;
      next_shape[p] = phys_target_shape[p];

      std::vector<int64_t> slice_params;
      for (int i = 0; i < total_dims; ++i) {
        if (i == p) {
          int64_t start = phys_starts[p];  // Already in base coordinates
          int64_t end = start + (phys_target_shape[p] - 1) * phys_steps[p] + 1;
          slice_params.push_back(start);
          slice_params.push_back(end);
          slice_params.push_back(phys_steps[p]);
        } else {
          slice_params.push_back(0);
          slice_params.push_back(cur_shape[i]);
          slice_params.push_back(1);
        }
      }
      transforms.emplace_back(TransformationType::SLICE, cur_shape, next_shape, element_type,
                              slice_params, id_perm_total, id_perm_total);
      cur_shape = next_shape;
    }
  }

  // TRANSPOSE (Move physical layout to logical layout)
  bool is_identity_perm = true;
  for (int i = 0; i < total_dims; ++i) {
    if (logical_to_phys[i] != i) {
      is_identity_perm = false;
      break;
    }
  }

  if (!is_identity_perm) {
    std::vector<int64_t> final_phys_shape(total_dims);
    for (int i = 0; i < total_dims; ++i) final_phys_shape[i] = cur_shape[logical_to_phys[i]];
    transforms.emplace_back(TransformationType::TRANSPOSE, cur_shape, final_phys_shape,
                            element_type, logical_to_phys, id_perm_total, logical_to_phys);
    cur_shape = final_phys_shape;
  }

  // BROADCAST
  std::vector<int64_t> final_shape(shape.begin(), shape.end());
  if (cur_shape != final_shape) {
    std::vector<int64_t> broadcast_params(ndims);
    std::iota(broadcast_params.begin(), broadcast_params.end(), 0);
    std::vector<int64_t> identity_perm_n(ndims);
    std::iota(identity_perm_n.begin(), identity_perm_n.end(), 0);
    transforms.emplace_back(TransformationType::BROADCAST, cur_shape, final_shape, element_type,
                            broadcast_params, identity_perm_n, identity_perm_n);
  }

  return TransformationCreationResult{true, std::move(transforms), base_tensor, "COMPOUND"};
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

  // Extract slice parameters: supports both pair encoding [start,end,...] and
  // triplet encoding [start,end,stride,...] from TryCreateCompound
  std::vector<int64_t> start_indices;
  std::vector<int64_t> end_indices;
  std::vector<int64_t> strides;

  size_t ndims = transform.input_shape.size();
  if (transform.params.size() == ndims * 3) {
    // Triplet encoding: [start, end, stride, start, end, stride, ...]
    for (size_t i = 0; i < transform.params.size(); i += 3) {
      start_indices.push_back(transform.params[i]);
      end_indices.push_back(transform.params[i + 1]);
      strides.push_back(transform.params[i + 2]);
    }
  } else {
    // Pair encoding: [start, end, start, end, ...] (legacy, stride=1)
    for (size_t i = 0; i < transform.params.size(); i += 2) {
      start_indices.push_back(transform.params[i]);
      end_indices.push_back(transform.params[i + 1]);
      strides.push_back(1);
    }
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
