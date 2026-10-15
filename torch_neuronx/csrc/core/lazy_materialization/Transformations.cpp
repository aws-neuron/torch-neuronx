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

// ============================================================================
// FOLDED DECOMPOSITION — Shared Types
// ============================================================================

struct FoldedDecompositionResult {
  at::Tensor base_tensor;
  std::vector<int64_t> base_shape;
  std::vector<TensorTransformation> transforms;
};

// ============================================================================
// SINGLE-GROUP DECOMPOSITION
// Searches for a single compact group of consecutive axes (in stride-sorted
// order) that form a folded split of a base axis. Restricted to rank >= 3.
//
// Example (narrow→view→permute):
//   x = randn(2, 3, 5).narrow(2, 0, 4).view(2, 3, 2, 2).permute(2, 0, 3, 1)
//   shape=[2, 2, 2, 3], strides=[2, 15, 1, 5]
//   → base [2, 3, 5] → SLICE [2, 3, 4] → RESHAPE [2, 3, 2, 2] → TRANSPOSE
// ============================================================================
std::optional<FoldedDecompositionResult> TrySingleGroupDecomposition(
    const at::Tensor& tensor, const std::vector<int64_t>& pre_order,
    const std::vector<int64_t>& pre_shape, const std::vector<int64_t>& pre_stride,
    const std::string& op_name, size_t input_index) {
  auto shape = tensor.sizes();
  auto strides = tensor.strides();
  int64_t ndims = shape.size();
  int64_t offset = tensor.storage_offset();

  if (ndims < 3) return std::nullopt;

  {
    for (int64_t group_len = 2; group_len <= ndims; ++group_len) {
      for (int64_t group_start = 1; group_start <= ndims - group_len; ++group_start) {
        int64_t group_end = group_start + group_len - 1;

        // Constraint: keep folded-axis inference narrow — don't allow large trailing groups
        if (group_end == ndims - 1 && group_len > 2) continue;

        // Compute group shape and numel
        int64_t group_numel = 1;
        for (int64_t pos = group_start; pos <= group_end; ++pos) {
          group_numel *= pre_shape[pos];
        }
        if (group_numel <= 1) continue;

        // Find the first non-unit dim in the group and the outer non-unit dim
        int64_t outer_pos = -1;
        for (int64_t pos = group_start - 1; pos >= 0; --pos) {
          if (pre_shape[pos] != 1) {
            outer_pos = pos;
            break;
          }
        }
        int64_t first_group_pos = -1;
        for (int64_t pos = group_start; pos <= group_end; ++pos) {
          if (pre_shape[pos] != 1) {
            first_group_pos = pos;
            break;
          }
        }
        int64_t last_group_pos = -1;
        for (int64_t pos = group_end; pos >= group_start; --pos) {
          if (pre_shape[pos] != 1) {
            last_group_pos = pos;
            break;
          }
        }
        if (outer_pos < 0 || first_group_pos < 0 || last_group_pos < 0) continue;

        // Check if outer stride == compact stride of the group (would mean no fold needed)
        // The compact outer stride is inner_stride * group_numel: if the group were a
        // contiguous sub-dim, this is what the next outer stride would be.
        int64_t inner_stride = pre_stride[last_group_pos];
        int64_t expected_compact_outer = inner_stride * group_numel;
        if (pre_stride[outer_pos] == expected_compact_outer) {
          // No stride gap — this isn't a folded pattern, skip
          continue;
        }

        // Verify group axes have compact strides (each = next × next_size, from right to left)
        int64_t expected_stride = inner_stride;
        bool compact = true;
        for (int64_t pos = group_end; pos >= group_start; --pos) {
          if (pre_shape[pos] == 1) continue;
          if (pre_stride[pos] != expected_stride) {
            compact = false;
            break;
          }
          expected_stride *= pre_shape[pos];
        }
        if (!compact) continue;

        // Compute folded base size: outer_stride / inner_stride
        int64_t outer_stride = pre_stride[outer_pos];
        if (inner_stride <= 0 || outer_stride % inner_stride != 0) continue;
        int64_t folded_base_size = outer_stride / inner_stride;
        if (folded_base_size < group_numel) continue;

        // Build base shape: replace the group with a single folded dimension
        std::vector<int64_t> base_shape;
        for (int64_t pos = 0; pos < group_start; ++pos) base_shape.push_back(pre_shape[pos]);
        base_shape.push_back(folded_base_size);
        for (int64_t pos = group_end + 1; pos < ndims; ++pos) base_shape.push_back(pre_shape[pos]);

        // Build the required shape (after slice, before reshape)
        std::vector<int64_t> required_shape;
        for (int64_t pos = 0; pos < group_start; ++pos) required_shape.push_back(pre_shape[pos]);
        required_shape.push_back(group_numel);
        for (int64_t pos = group_end + 1; pos < ndims; ++pos)
          required_shape.push_back(pre_shape[pos]);

        // Compute base strides (contiguous)
        auto base_strides = ComputeRowMajorStrides(base_shape);

        // Decompose storage offset into per-dimension starts
        int64_t remaining_offset = offset;
        std::vector<int64_t> starts(base_shape.size(), 0);
        for (size_t i = 0; i < base_shape.size(); ++i) {
          if (base_strides[i] > 0 && remaining_offset >= base_strides[i]) {
            starts[i] = remaining_offset / base_strides[i];
            remaining_offset %= base_strides[i];
          }
        }
        int64_t base_offset = remaining_offset;  // leftover becomes base_offset

        // Validate: slice must fit within base dims
        bool valid_slice = true;
        for (size_t i = 0; i < base_shape.size(); ++i) {
          if (starts[i] + required_shape[i] > base_shape[i]) {
            valid_slice = false;
            break;
          }
        }
        if (!valid_slice) continue;

        // Build permutation: pre_order maps pre-sorted position → original logical dim
        // We need perm such that applying permute(perm) to pre_shape gives logical shape
        std::vector<int64_t> perm(ndims);
        for (int64_t i = 0; i < ndims; ++i) {
          // Find where original dim i appears in pre_order
          for (int64_t j = 0; j < ndims; ++j) {
            if (pre_order[j] == i) {
              perm[i] = j;
              break;
            }
          }
        }

        // Verify reconstruction: apply transforms to base_shape and check strides match
        // Compute what the logical strides would be after slice→reshape→permute
        // Start with base contiguous strides after the fold
        std::vector<int64_t> folded_strides(ndims);
        {
          // Strides in pre-order after unfolding the group from base
          std::vector<int64_t> pre_reconstructed(ndims);
          int64_t folded_dim_stride = base_strides[group_start];  // stride of the folded base dim
          // Axes before group: same as base strides
          for (int64_t pos = 0; pos < group_start; ++pos) {
            pre_reconstructed[pos] = base_strides[pos];
          }
          // Group axes: compact from folded_dim_stride
          int64_t group_stride = folded_dim_stride;
          for (int64_t pos = group_end; pos >= group_start; --pos) {
            pre_reconstructed[pos] = group_stride;
            group_stride *= pre_shape[pos];
          }
          // Axes after group
          int64_t base_pos = group_start + 1;  // base dim after the folded one
          for (int64_t pos = group_end + 1; pos < ndims; ++pos) {
            pre_reconstructed[pos] = base_strides[base_pos];
            base_pos++;
          }

          // Apply permutation to get logical strides
          for (int64_t i = 0; i < ndims; ++i) {
            folded_strides[i] = pre_reconstructed[perm[i]];
          }
        }

        // Verify logical strides match (skip size-1 dims)
        bool strides_match = true;
        for (int64_t i = 0; i < ndims; ++i) {
          if (shape[i] == 1) continue;
          if (folded_strides[i] != strides[i]) {
            strides_match = false;
            break;
          }
        }
        if (!strides_match) continue;

        // Verify base tensor fits in storage
        int64_t base_elements = 1;
        for (auto s : base_shape) base_elements *= s;
        TORCH_INTERNAL_ASSERT(base_elements > 0,
                              "TrySingleGroupDecomposition: base_elements overflow detected, "
                              "base_shape product wrapped int64");
        int64_t storage_elements =
            static_cast<int64_t>(tensor.storage().nbytes() / tensor.element_size());
        if (base_offset < 0 || base_offset + base_elements > storage_elements) continue;

        // Reject if base tensor would exceed 4GB
        {
          int64_t base_bytes = base_elements * tensor.element_size();
          constexpr int64_t kMaxBytes = 4LL * 1024 * 1024 * 1024;
          if (base_bytes >= kMaxBytes) continue;
        }

        // === SUCCESS: Build the transformation chain ===
        std::string element_type = ScalarTypeToElementTypeString(tensor.scalar_type());

        // Create base tensor
        at::Tensor base_tensor = tensor.as_strided(c10::IntArrayRef(base_shape),
                                                   c10::IntArrayRef(base_strides), base_offset);

        std::vector<TensorTransformation> transforms;
        std::vector<int64_t> cur_shape = base_shape;
        int64_t base_rank = static_cast<int64_t>(base_shape.size());
        std::vector<int64_t> id_perm_base(base_rank);
        std::iota(id_perm_base.begin(), id_perm_base.end(), 0);

        // SLICE (if any dimension needs slicing)
        bool needs_slice = false;
        for (size_t i = 0; i < base_shape.size(); ++i) {
          if (starts[i] != 0 || required_shape[i] != base_shape[i]) {
            needs_slice = true;
            break;
          }
        }
        if (needs_slice) {
          std::vector<int64_t> slice_params;
          for (size_t i = 0; i < base_shape.size(); ++i) {
            slice_params.push_back(starts[i]);
            slice_params.push_back(starts[i] + required_shape[i]);
            slice_params.push_back(1);  // step=1
          }
          transforms.emplace_back(TransformationType::SLICE, cur_shape, required_shape,
                                  element_type, slice_params, id_perm_base, id_perm_base);
          cur_shape = required_shape;
        }

        // RESHAPE (unfold the folded dimension back into the group of sub-dimensions)
        transforms.emplace_back(TransformationType::RESHAPE, cur_shape, pre_shape, element_type);
        cur_shape = pre_shape;

        // TRANSPOSE (if perm is not identity)
        bool is_identity_perm = true;
        for (int64_t i = 0; i < ndims; ++i) {
          if (perm[i] != i) {
            is_identity_perm = false;
            break;
          }
        }
        if (!is_identity_perm) {
          std::vector<int64_t> logical_shape(ndims);
          for (int64_t i = 0; i < ndims; ++i) logical_shape[i] = cur_shape[perm[i]];
          std::vector<int64_t> id_perm_n(ndims);
          std::iota(id_perm_n.begin(), id_perm_n.end(), 0);
          transforms.emplace_back(TransformationType::TRANSPOSE, cur_shape, logical_shape,
                                  element_type, perm, id_perm_n, perm);
          cur_shape = logical_shape;
        }

        // Verify final shape matches
        std::vector<int64_t> expected_shape(shape.begin(), shape.end());
        if (cur_shape != expected_shape) continue;

        TORCH_NEURONX_DEBUG("TryFoldedDecomposition succeeded", "op=", op_name,
                            "input_index=", input_index, "base_shape=", base_shape,
                            "group_start=", group_start, "group_end=", group_end,
                            "folded_base_size=", folded_base_size);

        return FoldedDecompositionResult{base_tensor, base_shape, std::move(transforms)};
      }
    }
  }  // end single-group loop

  return std::nullopt;
}

// ============================================================================
// MULTI-GROUP DECOMPOSITION
// Discovers ALL compact groups simultaneously via right-to-left stride scan.
// Constructs a fully contiguous multi-dim grid base and emits:
//   RESHAPE → SLICE → RESHAPE → TRANSPOSE → BROADCAST
//
// Example (multi-split narrow→view→permute):
//   x = randn(3, 6).narrow(1, 1, 4).view(3, 2, 2).permute(1, 0, 2)
//   shape=[2, 3, 2], strides=[2, 6, 1], offset=1, storage=18
//   Groups: [{strides 6}, {strides 2, 1}] → base [3, 6]
//   → RESHAPE [18] → [3, 6] → SLICE → RESHAPE [3, 2, 2] → TRANSPOSE
// ============================================================================
std::optional<FoldedDecompositionResult> TryMultiGroupDecomposition(const at::Tensor& tensor,
                                                                    const std::string& op_name,
                                                                    size_t input_index) {
  auto shape = tensor.sizes();
  auto strides = tensor.strides();
  int64_t ndims = shape.size();
  int64_t offset = tensor.storage_offset();

  // Guard: skip if tensor is already contiguous (DecomposeStrides handles it above,
  // but the multi-group path must NOT accept contiguous tensors)
  if (tensor.is_contiguous()) return std::nullopt;

  {
    // Guard: reject negative strides (not supported)
    for (int64_t i = 0; i < ndims; ++i) {
      if (shape[i] > 1 && strides[i] < 0) return std::nullopt;
    }

    // Step 1: Filter active dims (size > 1 and stride > 0)
    std::vector<int64_t> active_indices;
    for (int64_t i = 0; i < ndims; ++i) {
      if (shape[i] > 1 && strides[i] > 0) {
        active_indices.push_back(i);
      }
    }
    if (active_indices.size() < 2) return std::nullopt;

    // Step 2: Sort active dims by stride descending
    std::sort(active_indices.begin(), active_indices.end(), [&](int64_t a, int64_t b) {
      if (strides[a] != strides[b]) return strides[a] > strides[b];
      return a < b;
    });

    int64_t num_active = static_cast<int64_t>(active_indices.size());
    std::vector<int64_t> act_shape(num_active);
    std::vector<int64_t> act_stride(num_active);
    for (int64_t i = 0; i < num_active; ++i) {
      act_shape[i] = shape[active_indices[i]];
      act_stride[i] = strides[active_indices[i]];
    }

    // Step 3: Discover compact groups (right-to-left scan)
    std::vector<CompactGroup> groups = DiscoverCompactGroups(act_shape, act_stride);

    // Guard: reject if no group contains more than one axis — this means no actual
    // folding occurred (each logical dim maps to its own physical dim). Such patterns
    // are simple slices, not folded reshapes, and should be handled by DecomposeStrides.
    {
      bool has_multi_axis_group = false;
      for (const auto& g : groups) {
        if (g.sorted_positions.size() > 1) {
          has_multi_axis_group = true;
          break;
        }
      }
      if (!has_multi_axis_group) return std::nullopt;
    }

    // Step 4: Construct the fully contiguous multi-dimensional base shape
    int64_t storage_elements =
        static_cast<int64_t>(tensor.storage().nbytes() / tensor.element_size());
    int64_t num_groups = static_cast<int64_t>(groups.size());

    // Grid layout: one axis per group, plus trailing dim if innermost stride > 1
    std::vector<int64_t> base_shape_mg;
    std::vector<int64_t> slice_starts_mg(num_groups, 0);
    std::vector<int64_t> slice_ends_mg(num_groups, 0);

    for (int64_t g = 0; g < num_groups; ++g) {
      int64_t group_dim_size = 0;
      if (g == 0) {
        // The outermost group's base dimension = storage_elements / inner_stride.
        // This must divide evenly — otherwise the storage size is not a multiple of
        // the group's stride (Limitation 2: non-integer stride ratio between storage
        // extent and outermost group stride). Falls back to contiguous copy.
        if (groups[g].inner_stride <= 0 || storage_elements % groups[g].inner_stride != 0) {
          return std::nullopt;
        }
        group_dim_size = storage_elements / groups[g].inner_stride;
      } else {
        int64_t outer_inner_stride = groups[g - 1].inner_stride;
        if (groups[g].inner_stride <= 0 || outer_inner_stride % groups[g].inner_stride != 0) {
          return std::nullopt;
        }
        group_dim_size = outer_inner_stride / groups[g].inner_stride;
      }
      base_shape_mg.push_back(group_dim_size);
    }

    // Append trailing unit dimension to anchor remaining element stride spaces
    if (groups.back().inner_stride > 1) {
      base_shape_mg.push_back(groups.back().inner_stride);
    }

    // Step 5: Decompose global storage offset completely across the grid rank
    int64_t grid_rank = static_cast<int64_t>(base_shape_mg.size());
    slice_starts_mg.resize(grid_rank, 0);
    slice_ends_mg.resize(grid_rank, 0);
    int64_t remaining_offset = offset;

    for (int64_t i = 0; i < grid_rank; ++i) {
      int64_t dim_stride = (i < num_groups) ? groups[i].inner_stride : 1;
      slice_starts_mg[i] = remaining_offset / dim_stride;
      remaining_offset %= dim_stride;

      int64_t required_size = (i < num_groups) ? groups[i].shape_product : 1;
      slice_ends_mg[i] = slice_starts_mg[i] + required_size;

      if (slice_ends_mg[i] > base_shape_mg[i]) return std::nullopt;
    }
    if (remaining_offset != 0) return std::nullopt;

    // Step 6: Memory safety and 4GB allocation checks
    int64_t base_elements = 1;
    for (auto s : base_shape_mg) base_elements *= s;
    if (base_elements > storage_elements) return std::nullopt;
    {
      int64_t base_bytes = base_elements * tensor.element_size();
      constexpr int64_t kMaxBytes = 4LL * 1024 * 1024 * 1024;
      if (base_bytes >= kMaxBytes) return std::nullopt;
    }

    // === SUCCESS: Construct Lowering Transform Graph ===
    std::string element_type = ScalarTypeToElementTypeString(tensor.scalar_type());

    // Emit flat 1D base tensor starting at storage offset 0
    at::Tensor base_tensor = tensor.as_strided({storage_elements}, {1}, 0);
    std::vector<TensorTransformation> transforms;

    // 1. RESHAPE: Flattened 1D storage → Contiguous Multi-Dim Grid
    std::vector<int64_t> cur_shape_mg = base_shape_mg;
    {
      std::vector<int64_t> flat = {storage_elements};
      std::vector<int64_t> id1 = {0};
      // No params needed for reshape — output_shape is sufficient
      transforms.emplace_back(TransformationType::RESHAPE, flat, cur_shape_mg, element_type);
    }

    // 2. SLICE: Carve out the active sub-volumes across group dimensions
    std::vector<int64_t> post_slice_shape;
    {
      std::vector<int64_t> slice_params;
      for (int64_t g = 0; g < num_groups; ++g) {
        slice_params.push_back(slice_starts_mg[g]);
        slice_params.push_back(slice_ends_mg[g]);
        slice_params.push_back(1);  // unit stride step
        post_slice_shape.push_back(groups[g].shape_product);
      }
      if (groups.back().inner_stride > 1) {
        // Use the dynamically computed coordinate for the trailing anchor dimension
        int64_t trailing_idx = grid_rank - 1;
        slice_params.push_back(slice_starts_mg[trailing_idx]);
        slice_params.push_back(slice_ends_mg[trailing_idx]);
        slice_params.push_back(1);
        post_slice_shape.push_back(slice_ends_mg[trailing_idx] - slice_starts_mg[trailing_idx]);
      }

      std::vector<int64_t> id_mg(cur_shape_mg.size());
      std::iota(id_mg.begin(), id_mg.end(), 0);
      transforms.emplace_back(TransformationType::SLICE, cur_shape_mg, post_slice_shape,
                              element_type, slice_params, id_mg, id_mg);
      cur_shape_mg = post_slice_shape;
    }

    // 3. RESHAPE: Explode flattened group dimensions back to active shapes
    std::vector<int64_t> reshape_target;
    for (int64_t i = 0; i < num_active; ++i) {
      reshape_target.push_back(act_shape[i]);
    }
    // If there's a trailing anchor dimension, include it temporarily then strip
    if (groups.back().inner_stride > 1) {
      reshape_target.push_back(1);
    }
    transforms.emplace_back(TransformationType::RESHAPE, cur_shape_mg, reshape_target,
                            element_type);
    cur_shape_mg = reshape_target;

    // Strip the trailing anchor dimension if it was added
    if (groups.back().inner_stride > 1) {
      std::vector<int64_t> stripped(cur_shape_mg.begin(), cur_shape_mg.end() - 1);
      transforms.emplace_back(TransformationType::RESHAPE, cur_shape_mg, stripped, element_type);
      cur_shape_mg = stripped;
    }

    // 4. TRANSPOSE: Map pre-sorted active dim order back to original logical order
    std::vector<int64_t> logical_active_order;
    for (int64_t i = 0; i < ndims; ++i) {
      if (shape[i] > 1 && strides[i] > 0) logical_active_order.push_back(i);
    }
    std::vector<int64_t> perm_mg(num_active);
    for (int64_t li = 0; li < num_active; ++li) {
      int64_t logical_dim = logical_active_order[li];
      for (int64_t si = 0; si < num_active; ++si) {
        if (active_indices[si] == logical_dim) {
          perm_mg[li] = si;
          break;
        }
      }
    }

    bool is_identity = true;
    for (int64_t i = 0; i < num_active; ++i) {
      if (perm_mg[i] != i) {
        is_identity = false;
        break;
      }
    }
    if (!is_identity) {
      std::vector<int64_t> transposed_shape(num_active);
      for (int64_t i = 0; i < num_active; ++i) transposed_shape[i] = cur_shape_mg[perm_mg[i]];
      std::vector<int64_t> id_active(num_active);
      std::iota(id_active.begin(), id_active.end(), 0);
      transforms.emplace_back(TransformationType::TRANSPOSE, cur_shape_mg, transposed_shape,
                              element_type, perm_mg, id_active, perm_mg);
      cur_shape_mg = transposed_shape;
    }

    // 5. BROADCAST: Inflate or inject non-active size-1 and broadcast layout dims
    std::vector<int64_t> full_shape(shape.begin(), shape.end());
    if (cur_shape_mg != full_shape) {
      std::vector<int64_t> with_bc_shape;
      int64_t active_ptr = 0;
      for (int64_t i = 0; i < ndims; ++i) {
        if (shape[i] <= 1 || strides[i] <= 0) {
          with_bc_shape.push_back(1);
        } else {
          with_bc_shape.push_back(cur_shape_mg[active_ptr++]);
        }
      }
      if (with_bc_shape != cur_shape_mg) {
        transforms.emplace_back(TransformationType::RESHAPE, cur_shape_mg, with_bc_shape,
                                element_type);
        cur_shape_mg = with_bc_shape;
      }
      if (cur_shape_mg != full_shape) {
        std::vector<int64_t> bc_params(ndims);
        std::iota(bc_params.begin(), bc_params.end(), 0);
        std::vector<int64_t> id_full(ndims);
        std::iota(id_full.begin(), id_full.end(), 0);
        transforms.emplace_back(TransformationType::BROADCAST, cur_shape_mg, full_shape,
                                element_type, bc_params, id_full, id_full);
      }
    }

    TORCH_NEURONX_DEBUG("TryFoldedDecomposition multi-group pipeline succeeded", "op=", op_name,
                        "input_index=", input_index, "num_groups=", num_groups);
    return FoldedDecompositionResult{base_tensor, {storage_elements}, std::move(transforms)};
  }

  return std::nullopt;
}

// ============================================================================
// FOLDED DECOMPOSITION DISPATCHER
//
// Fallback for non-contiguous tensors that DecomposeStrides rejects (GCD check
// fails due to view-split creating coprime stride ratios).
//
// Approach: Discover "compact groups" of consecutive dimensions (in stride-sorted
// order) where strides form an arithmetic progression. Each group maps to one
// base dimension. Emit: per-group SLICE → RESHAPE → TRANSPOSE → BROADCAST.
//
// A group boundary occurs at index i when:
//   pre_stride[i] != pre_stride[i+1] * pre_shape[i+1]
//
// For each group:
//   - inner_stride = smallest stride in the group
//   - shape_product = product of all shapes in the group
//   - base_dim_size = outer_stride / inner_stride (for inner groups)
//                   or storage_extent / inner_stride (for outermost group)
//
// Two strategies, tried in order:
//   1. TrySingleGroupDecomposition (rank >= 3):
//      Brute-force search for a single compact group within stride-sorted axes.
//      Emits: base → SLICE → RESHAPE → TRANSPOSE
//   2. TryMultiGroupDecomposition:
//      Discovers ALL compact groups via right-to-left stride scan.
//      Emits: flat storage → RESHAPE → SLICE → RESHAPE → TRANSPOSE → BROADCAST
// ============================================================================
std::optional<FoldedDecompositionResult> TryFoldedDecomposition(const at::Tensor& tensor,
                                                                const std::string& op_name,
                                                                size_t input_index) {
  auto shape = tensor.sizes();
  auto strides = tensor.strides();
  int64_t ndims = shape.size();

  // Reject negative strides (not supported).
  // Note: broadcast dims (stride=0) are allowed — they are handled by the
  // multi-group pipeline which filters them out and emits BROADCAST.
  for (int64_t i = 0; i < ndims; ++i) {
    if (shape[i] > 1 && strides[i] < 0) return std::nullopt;
  }

  // Sort axes by stride descending (pre-permutation order)
  std::vector<int64_t> pre_order(ndims);
  std::iota(pre_order.begin(), pre_order.end(), 0);
  std::sort(pre_order.begin(), pre_order.end(), [&](int64_t a, int64_t b) {
    if (strides[a] != strides[b]) return strides[a] > strides[b];
    return a < b;
  });

  std::vector<int64_t> pre_shape(ndims);
  std::vector<int64_t> pre_stride(ndims);
  for (int64_t i = 0; i < ndims; ++i) {
    pre_shape[i] = shape[pre_order[i]];
    pre_stride[i] = strides[pre_order[i]];
  }

  // Try single-group decomposition first (rank >= 3 only)
  auto single =
      TrySingleGroupDecomposition(tensor, pre_order, pre_shape, pre_stride, op_name, input_index);
  if (single.has_value()) return single;

  // Fall back to multi-group decomposition
  return TryMultiGroupDecomposition(tensor, op_name, input_index);
}

// ============================================================================
// GENERIC SCATTER_IOTA EPILOGUE FALLBACK
//
// Used by TryCreateEpilogue when the analytical decomposition
// (transpose ∘ slice of a contiguous base, via AnalyzeEpilogueLayout) cannot
// express the output's stride pattern, e.g.:
//   - Diagonal / padded-innermost-dim writes: shape=[5] strides=[6] (diag),
//     shape=[5,5] strides=[25,6] (diag_embed), shape=[2,2] strides=[1,2]
//     (as_strided diagonals), shape=[2] strides=[2] (linalg det `mul.out`).
//   - Select / non-multiple stride-gap writes: shape=[5,5] strides=[25,5]
//     (select on 3D), shape=[1,5,1] strides=[25,5,1] (cumprod/prod zero_),
//     shape=[20,20,5] strides=[400,20,2] (slice_scatter step=2).
//
// Emits a single SCATTER_IOTA using the raw output strides directly (no
// decomposition). Each view element maps to a flat index
//     flat_idx = offset + Σ iₖ · stride_k
// and stablehlo.scatter writes update[i] into base_flat[flat_idx].
//
// SAFETY (must be airtight — wrong scatter = silent miscompile):
//   1. Reuse the existing scatter_iota_enabled gate (default ON, env "0" to
//      disable). Disabling SCATTER_IOTA disables ALL scatter paths.
//   2. Defensive non-negative offset / stride sanity checks.
//   3. INJECTIVITY GUARD — sort active dims by stride desc; require
//      stride[outer] >= stride[inner] * size[inner] for every adjacent pair.
//      This guarantees no two view multi-indices map to the same flat index,
//      which is required because ScatterIotaOpBuilder hardcodes
//      unique_indices=true. Overlapping-stride patterns (e.g., `unfold` with
//      step < window) are correctly rejected here and fall through to the
//      CPU-bounce path in CopyBackToOriginal.
//   4. OUT-OF-BOUNDS REJECT (no clamping) — require max_flat_idx <
//      storage_elements. Clamping would alias / drop writes.
//   5. 4GB guard on storage_span * element_size.
// If any check fails we return nullopt — the existing fallback path
// (contiguous temp + CopyBackToOriginal) takes over, always correct.
// ============================================================================
std::optional<TransformationCreationResult> TryCreateGenericScatterEpilogue(
    const at::Tensor& output, c10::IntArrayRef shape, c10::IntArrayRef strides, int64_t offset,
    int64_t ndims, int64_t storage_elements, const std::string& op_name, size_t output_index) {
  // Gate: reuse the existing SCATTER_IOTA env switch.
  static const bool scatter_iota_enabled = []() {
    const char* env = std::getenv("TORCH_NEURONX_ENABLE_SCATTER_IOTA");
    return env == nullptr || std::string(env) != "0";
  }();
  if (!scatter_iota_enabled) {
    TORCH_NEURONX_DEBUG(
        "TryCreateGenericScatterEpilogue: skipped (TORCH_NEURONX_ENABLE_SCATTER_IOTA=0)",
        "op=", op_name, "output_index=", output_index);
    return std::nullopt;
  }

  // Defensive checks. The caller (TryCreateEpilogue) already rejects broadcast
  // / negative-stride patterns, but repeat here so this helper is self-contained
  // and safely callable from anywhere.
  if (offset < 0) {
    TORCH_NEURONX_DEBUG("TryCreateGenericScatterEpilogue: rejected (negative offset)",
                        "op=", op_name, "output_index=", output_index, "offset=", offset);
    return std::nullopt;
  }
  for (int64_t i = 0; i < ndims; ++i) {
    if (strides[i] < 0 || (strides[i] == 0 && shape[i] > 1)) {
      return std::nullopt;
    }
  }

  // Injectivity guard: sort ACTIVE (size>1) dims by stride descending and
  // require the inner dim's INCLUSIVE SPAN to fit within one outer stride step:
  //     (size[inner] - 1) * stride[inner] + 1  <=  stride[outer]
  //
  // This is the correct non-overlap condition for a lexicographic strided
  // layout — when the inner dim's max reachable offset (== span - 1) is
  // strictly less than stride[outer], no two view multi-indices can collide
  // on the same flat index, so stablehlo.scatter's unique_indices=true holds.
  //
  // NOTE: a previous version used `size[inner] * stride[inner] <= stride[outer]`
  // (the "next-step distance"), which was off by `stride[inner]` — too strict
  // by exactly one element. It rejected valid diagonal-style writes where the
  // inner span exactly fills one outer stride step, e.g.:
  //   [5, 5] strides=[25, 6]  (diag_embed): inner span = 4*6+1 = 25 == 25 ✓
  //   [3, 5] strides=[25, 6]  (linalg_det): inner span = 4*6+1 = 25 == 25 ✓
  //   [2, 3] strides=[9, 4]   (solve_tri):  inner span = 2*4+1 =  9 ==  9 ✓
  // These layouts are injective but the old guard rejected them.
  std::vector<int64_t> active_dims;
  active_dims.reserve(ndims);
  for (int64_t i = 0; i < ndims; ++i) {
    if (shape[i] > 1) active_dims.push_back(i);
  }
  std::sort(active_dims.begin(), active_dims.end(),
            [&](int64_t a, int64_t b) { return strides[a] > strides[b]; });
  for (size_t k = 0; k + 1 < active_dims.size(); ++k) {
    int64_t outer = active_dims[k];
    int64_t inner = active_dims[k + 1];
    int64_t inner_span = (shape[inner] - 1) * strides[inner] + 1;
    if (inner_span > strides[outer]) {
      TORCH_NEURONX_DEBUG(
          "TryCreateGenericScatterEpilogue: rejected (overlapping strides — inner span exceeds "
          "outer stride, would produce non-unique flat indices and violate "
          "stablehlo.scatter unique_indices=true)",
          "op=", op_name, "output_index=", output_index, "shape=", shape, "strides=", strides,
          "outer_dim=", outer, "outer_stride=", strides[outer], "inner_dim=", inner,
          "inner_stride=", strides[inner], "inner_size=", shape[inner], "inner_span=", inner_span);
      return std::nullopt;
    }
  }

  // Out-of-bounds reject (no clamping): max_flat_idx must be < storage_elements.
  int64_t max_flat_idx = offset;
  for (int64_t i = 0; i < ndims; ++i) {
    if (shape[i] > 1) max_flat_idx += (shape[i] - 1) * strides[i];
  }
  if (max_flat_idx >= storage_elements) {
    TORCH_NEURONX_DEBUG("TryCreateGenericScatterEpilogue: rejected (max flat index out of storage)",
                        "op=", op_name, "output_index=", output_index,
                        "max_flat_idx=", max_flat_idx, "storage_elements=", storage_elements);
    return std::nullopt;
  }

  int64_t scatter_storage_span = storage_elements;
  if (scatter_storage_span <= 0) {
    TORCH_NEURONX_DEBUG("TryCreateGenericScatterEpilogue: rejected (storage_span <= 0)",
                        "op=", op_name, "output_index=", output_index,
                        "storage_span=", scatter_storage_span);
    return std::nullopt;
  }

  // 4GB guard.
  {
    int64_t base_bytes = scatter_storage_span * output.element_size();
    constexpr int64_t kMaxBytes = 4LL * 1024 * 1024 * 1024;
    if (base_bytes >= kMaxBytes) {
      TORCH_NEURONX_DEBUG("TryCreateGenericScatterEpilogue: rejected (storage > 4GB)",
                          "op=", op_name, "output_index=", output_index, "base_bytes=", base_bytes);
      return std::nullopt;
    }
  }

  // Emit a single SCATTER_IOTA with raw output strides. params layout matches
  // ScatterIotaHandler / ScatterIotaOpBuilder:
  //   [stride_0, ..., stride_{n-1}, storage_offset, storage_span]
  std::string element_type = ScalarTypeToElementTypeString(output.scalar_type());
  std::vector<int64_t> logical_shape(shape.begin(), shape.end());
  std::vector<int64_t> id_perm(ndims);
  std::iota(id_perm.begin(), id_perm.end(), 0);

  std::vector<int64_t> scatter_params;
  scatter_params.reserve(ndims + 2);
  for (int64_t i = 0; i < ndims; ++i) scatter_params.push_back(strides[i]);
  scatter_params.push_back(offset);
  scatter_params.push_back(scatter_storage_span);

  std::vector<TensorTransformation> transforms;
  transforms.emplace_back(TransformationType::SCATTER_IOTA, logical_shape,
                          std::vector<int64_t>{scatter_storage_span}, element_type, scatter_params,
                          id_perm, id_perm);

  at::Tensor base_tensor = output.as_strided({scatter_storage_span}, {1}, 0);

  TORCH_NEURONX_DEBUG("TryCreateGenericScatterEpilogue: succeeded (EPILOGUE_GENERIC_SCATTER)",
                      "op=", op_name, "output_index=", output_index, "shape=", shape,
                      "strides=", strides, "offset=", offset,
                      "storage_span=", scatter_storage_span);

  return TransformationCreationResult{true, std::move(transforms), base_tensor,
                                      "EPILOGUE_GENERIC_SCATTER"};
}

}  // anonymous namespace

// ============================================================================
// INTERNAL UTILITIES IMPLEMENTATION
// ============================================================================

std::vector<CompactGroup> DiscoverCompactGroups(const std::vector<int64_t>& act_shape,
                                                const std::vector<int64_t>& act_stride) {
  int64_t num_active = static_cast<int64_t>(act_shape.size());
  if (num_active == 0) return {};

  std::vector<CompactGroup> groups;
  CompactGroup current_group;
  current_group.sorted_positions.push_back(num_active - 1);
  current_group.inner_stride = act_stride[num_active - 1];
  current_group.shape_product = act_shape[num_active - 1];

  for (int64_t i = num_active - 2; i >= 0; --i) {
    int64_t expected_stride = act_stride[i + 1] * act_shape[i + 1];
    if (act_stride[i] == expected_stride) {
      current_group.sorted_positions.push_back(i);
      current_group.shape_product *= act_shape[i];
    } else {
      groups.push_back(std::move(current_group));
      current_group = CompactGroup{};
      current_group.sorted_positions.push_back(i);
      current_group.inner_stride = act_stride[i];
      current_group.shape_product = act_shape[i];
    }
  }
  groups.push_back(std::move(current_group));
  std::reverse(groups.begin(), groups.end());
  return groups;
}

// ============================================================================
// STRIDE DECOMPOSITION IMPLEMENTATION
// ============================================================================

std::optional<StrideDecompositionResult> DecomposeStrides(const at::Tensor& tensor,
                                                          const std::string& op_name, size_t index,
                                                          bool anchor_at_view_offset) {
  auto shape = tensor.sizes();
  auto strides = tensor.strides();
  int64_t offset = anchor_at_view_offset ? 0 : tensor.storage_offset();
  int64_t ndims = shape.size();

  if (ndims == 0 || tensor.is_contiguous()) {
    TORCH_NEURONX_DEBUG("DecomposeStrides rejected: tensor is empty or contiguous", "op=", op_name,
                        "index=", index, "ndims=", ndims, "is_contiguous=", tensor.is_contiguous());
    return std::nullopt;
  }

  // Step 1: Identify broadcast dims (stride=0 or size=1) vs non-broadcast dims.
  std::vector<int64_t> non_bc_dims;
  std::vector<bool> is_bc(ndims, false);
  for (int64_t i = 0; i < ndims; ++i) {
    if (strides[i] == 0 || shape[i] == 1) {
      is_bc[i] = true;
    } else {
      non_bc_dims.push_back(i);
    }
  }

  // Handle all-broadcast: return a scalar base tensor with broadcast metadata
  if (non_bc_dims.empty()) {
    std::string element_type = ScalarTypeToElementTypeString(tensor.scalar_type());
    at::Tensor base_tensor = tensor.as_strided({1}, {1}, tensor.storage_offset());

    StrideDecompositionResult result;
    result.base_tensor = base_tensor;
    result.phys_base_shape = {1};
    result.phys_starts = {0};
    result.phys_target_shape = {1};
    result.phys_steps = {1};
    result.phys_to_logical = {};  // empty signals all-broadcast
    result.logical_to_phys = {};
    result.is_broadcast = is_bc;
    result.element_type = element_type;
    result.total_dims = ndims;
    return result;
  }

  // Step 2: Sort non-broadcast dims by stride descending.
  int64_t num_non_broadcast = non_bc_dims.size();
  std::vector<int64_t> sorted_nb_order(num_non_broadcast);
  std::iota(sorted_nb_order.begin(), sorted_nb_order.end(), 0);
  std::sort(sorted_nb_order.begin(), sorted_nb_order.end(), [&](int64_t a, int64_t b) {
    int64_t stride_a = strides[non_bc_dims[a]];
    int64_t stride_b = strides[non_bc_dims[b]];
    if (stride_a != stride_b) return stride_a > stride_b;
    return non_bc_dims[a] < non_bc_dims[b];
  });

  int64_t total_dims = ndims;

  // Step 3: Extract sorted shapes/strides and factorize stride gaps.
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
    std::vector<int64_t> outer_stride_gcd(num_non_broadcast);
    outer_stride_gcd[0] = sorted_strides[0];
    for (int i = 1; i < num_non_broadcast; ++i) {
      outer_stride_gcd[i] = std::gcd(outer_stride_gcd[i - 1], sorted_strides[i]);
    }

    sorted_base_strides[num_non_broadcast - 1] = 1;
    step_factors[num_non_broadcast - 1] = sorted_strides[num_non_broadcast - 1];

    for (int64_t i = num_non_broadcast - 2; i >= 0; --i) {
      if (outer_stride_gcd[i] % sorted_base_strides[i + 1] != 0) {
        TORCH_NEURONX_DEBUG("DecomposeStrides rejected: non-admissible layout (GCD check)",
                            "op=", op_name, "index=", index, "sorted_dim=", i);
        return std::nullopt;
      }

      int64_t max_base_extent = outer_stride_gcd[i] / sorted_base_strides[i + 1];
      int64_t min_required_extent = (sorted_shape[i + 1] - 1) * step_factors[i + 1] + 1;

      int64_t best_extent = -1;
      for (int64_t d = 1; d * d <= max_base_extent; ++d) {
        if (max_base_extent % d != 0) continue;
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
        TORCH_NEURONX_DEBUG("DecomposeStrides rejected: no valid base factor found", "op=", op_name,
                            "index=", index, "sorted_dim=", i);
        return std::nullopt;
      }

      sorted_base_shape[i + 1] = best_extent;
      sorted_base_strides[i] = sorted_base_strides[i + 1] * best_extent;
      step_factors[i] = sorted_strides[i] / sorted_base_strides[i];
    }
  }

  // Step 4: Offset decomposition
  int64_t remaining_offset = offset;
  std::vector<int64_t> sorted_starts(num_non_broadcast, 0);
  for (int64_t i = 0; i < num_non_broadcast; ++i) {
    if (sorted_base_strides[i] > 0 && remaining_offset >= sorted_base_strides[i]) {
      sorted_starts[i] = remaining_offset / sorted_base_strides[i];
      remaining_offset %= sorted_base_strides[i];
    }
  }

  if (remaining_offset != 0) {
    TORCH_NEURONX_DEBUG("DecomposeStrides rejected: offset decomposition failed", "op=", op_name,
                        "index=", index, "remaining_offset=", remaining_offset);
    return std::nullopt;
  }

  // Enlarge outermost base shape to prevent out-of-bounds slicing
  if (num_non_broadcast > 0) {
    int64_t outermost_span = sorted_starts[0] + (sorted_shape[0] - 1) * step_factors[0] + 1;
    sorted_base_shape[0] = std::max(sorted_base_shape[0], outermost_span);
  }

  // Step 5: Build physical-to-logical dimension mapping.
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

  // Step 6: Construct the contiguous base tensor.
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
      int64_t start_idx = sorted_starts[sorted_ptr];
      int64_t step = step_factors[sorted_ptr];
      int64_t physical_span = start_idx + (logical_size - 1) * step + 1;
      int64_t stride_gap_extent = sorted_base_shape[sorted_ptr];

      phys_base_shape[p] = std::max(stride_gap_extent, physical_span);
      phys_starts[p] = start_idx;
      phys_target_shape[p] = logical_size;
      sorted_ptr++;
    }
  }

  auto base_strides = ComputeRowMajorStrides(phys_base_shape);

  int64_t base_offset = tensor.storage_offset();
  for (int i = 0; i < total_dims; ++i) {
    base_offset -= phys_starts[i] * base_strides[i];
  }

  int64_t base_elements = 1;
  for (auto s : phys_base_shape) base_elements *= s;
  int64_t storage_elements =
      static_cast<int64_t>(tensor.storage().nbytes() / tensor.element_size());
  if (base_offset < 0 || base_offset + base_elements > storage_elements) {
    TORCH_NEURONX_DEBUG("DecomposeStrides rejected: base tensor out of storage bounds",
                        "op=", op_name, "index=", index, "base_offset=", base_offset,
                        "base_elements=", base_elements, "storage_elements=", storage_elements);
    return std::nullopt;
  }

  // Reject if base tensor would exceed 4GB
  {
    int64_t base_bytes = base_elements * tensor.element_size();
    constexpr int64_t kMaxBytes = 4LL * 1024 * 1024 * 1024;
    if (base_bytes >= kMaxBytes) {
      TORCH_NEURONX_DEBUG("DecomposeStrides rejected: base tensor exceeds 4GB", "op=", op_name,
                          "index=", index, "base_bytes=", base_bytes);
      return std::nullopt;
    }
  }

  at::Tensor base_tensor = tensor.as_strided(c10::IntArrayRef(phys_base_shape),
                                             c10::IntArrayRef(base_strides), base_offset);

  // Compute phys_steps for all dims
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

  std::string element_type = ScalarTypeToElementTypeString(tensor.scalar_type());

  StrideDecompositionResult result;
  result.base_tensor = base_tensor;
  result.phys_base_shape = phys_base_shape;
  result.phys_starts = phys_starts;
  result.phys_target_shape = phys_target_shape;
  result.phys_steps = phys_steps;
  result.phys_to_logical = phys_to_logical;
  result.logical_to_phys = logical_to_phys;
  result.is_broadcast = is_bc;
  result.element_type = element_type;
  result.total_dims = total_dims;
  return result;
}

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
// transpose, slice, expand, select, unsqueeze, squeeze, permute, view/reshape,
// etc.), reconstruct a contiguous base tensor and a chain of SLICE → RESHAPE →
// TRANSPOSE → BROADCAST operations that reproduce the original view.
//
// TWO-PHASE APPROACH:
//
// Phase 1 — GCD-based stride factorization (DecomposeStrides):
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
// Phase 2 — Folded decomposition fallback (TryFoldedDecomposition):
//   When the GCD check rejects a pattern (coprime stride ratios from view-split
//   on a narrowed tensor), TryFoldedDecomposition detects groups of consecutive
//   axes (in stride-sorted order) that form a compact split of a single base
//   axis. It emits SLICE → RESHAPE → TRANSPOSE to reconstruct the view.
//   Example: x.narrow(2,0,4).view(2,3,2,2).permute(2,0,3,1) has strides
//   [15, 5, 2, 1] where GCD(5,2)=1 — rejected by Phase 1, handled by Phase 2
//   as base [2,3,5] → SLICE → RESHAPE → TRANSPOSE.
//
// LIMITATION:
// 1. Uniform stride scaling (innermost active stride > 1): The multi-group pipeline
//    creates a virtual row-major grid and carves sub-volumes with step=1 SLICE only.
//    When all strides are scaled by a constant factor (e.g., strides=[32,16,8]),
//    the grid inflates to absorb spacing, but non-zero offsets push slice bounds
//    beyond physical allocation (slice_ends > base_shape). Falls back to contiguous.
// 2. Non-monotonic stride ratios (non-integer stride[i]/stride[i+1]): Base shape is
//    computed as stride[i]/stride[i+1]. Non-integer ratios (e.g., 49/21=2.33) break
//    divisor properties. Shared identical strides collapse groups incorrectly.
//    Falls back to contiguous.
//    NOTE: Fixing 1-2 requires adding step>1 SLICE to the multi-group pipeline, which
//    risks emitting multi-dimensional strided gathers that break memory coalescing in
//    the MLIR/StableHLO backend. The contiguous copy fallback is actually net-positive
//    for these layouts: a single sequential DMA burst guarantees 100% downstream
//    vectorization efficiency, recovering the copy latency immediately.
// 3. Overlapping strides (unfold with step < size): elements share memory in
//    multiple output positions. Cannot be represented as any combination of
//    SLICE + RESHAPE + TRANSPOSE + BROADCAST. Falls back to contiguous_generic_kernel.
// 4. Different views of the same shared storage (e.g., FSDP flat parameter buffers)
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
// SUPPORTED PATTERNS (all standard view operations):
//   transpose, slice, expand, select, squeeze, unsqueeze, permute,
//   view/reshape (split or merge dimensions), narrow, flip (as compute op)
//
// NOT SUPPORTED:
//   overlapping strides (unfold with step < size), diagonal
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
  // Use shared stride decomposition (Steps 1-6)
  auto decomp = DecomposeStrides(input, op_name, input_index);
  // Anchor the base at the view offset only when the offset-0 decomposition fails. An anchored base
  // reads from the view offset, so a shared-storage dedup input must keep the offset-0 base.
  if (!decomp.has_value() && input.storage_offset() != 0) {
    decomp = DecomposeStrides(input, op_name, input_index, /*anchor_at_view_offset=*/true);
  }
  if (!decomp.has_value()) {
    return std::nullopt;
  }

  auto shape = input.sizes();
  int64_t ndims = shape.size();
  const auto& d = decomp.value();

  // Handle all-broadcast early return (DecomposeStrides signals this with empty phys_to_logical)
  if (d.phys_to_logical.empty()) {
    std::vector<TensorTransformation> transforms;
    std::vector<int64_t> all_ones_shape(ndims, 1);
    transforms.emplace_back(TransformationType::RESHAPE, std::vector<int64_t>{1}, all_ones_shape,
                            d.element_type);

    std::vector<int64_t> bc_params(ndims);
    std::iota(bc_params.begin(), bc_params.end(), 0);
    std::vector<int64_t> id_perm(ndims);
    std::iota(id_perm.begin(), id_perm.end(), 0);
    transforms.emplace_back(TransformationType::BROADCAST, all_ones_shape,
                            std::vector<int64_t>(shape.begin(), shape.end()), d.element_type,
                            bc_params, id_perm, id_perm);

    return TransformationCreationResult{true, std::move(transforms), d.base_tensor,
                                        "COMPOUND_ALL_BC"};
  }

  // Step 7: Generate the forward transformation chain from decomposition result.
  // Emit one SLICE per dimension that needs slicing/stepping, then TRANSPOSE
  // (if physical order ≠ logical order), then BROADCAST (if any dims were broadcast).
  int64_t total_dims = d.total_dims;
  std::vector<TensorTransformation> transforms;
  std::vector<int64_t> cur_shape = d.phys_base_shape;
  std::vector<int64_t> id_perm_total(total_dims);
  std::iota(id_perm_total.begin(), id_perm_total.end(), 0);

  // SLICE ops
  for (int p = 0; p < total_dims; ++p) {
    if (d.phys_starts[p] > 0 || d.phys_base_shape[p] != d.phys_target_shape[p] ||
        d.phys_steps[p] > 1) {
      std::vector<int64_t> next_shape = cur_shape;
      next_shape[p] = d.phys_target_shape[p];

      std::vector<int64_t> slice_params;
      for (int i = 0; i < total_dims; ++i) {
        if (i == p) {
          int64_t start = d.phys_starts[p];
          int64_t end = start + (d.phys_target_shape[p] - 1) * d.phys_steps[p] + 1;
          slice_params.push_back(start);
          slice_params.push_back(end);
          slice_params.push_back(d.phys_steps[p]);
        } else {
          slice_params.push_back(0);
          slice_params.push_back(cur_shape[i]);
          slice_params.push_back(1);
        }
      }
      transforms.emplace_back(TransformationType::SLICE, cur_shape, next_shape, d.element_type,
                              slice_params, id_perm_total, id_perm_total);
      cur_shape = next_shape;
    }
  }

  // TRANSPOSE (Move physical layout to logical layout)
  bool is_identity_perm = true;
  for (int i = 0; i < total_dims; ++i) {
    if (d.logical_to_phys[i] != i) {
      is_identity_perm = false;
      break;
    }
  }

  if (!is_identity_perm) {
    std::vector<int64_t> final_phys_shape(total_dims);
    for (int i = 0; i < total_dims; ++i) final_phys_shape[i] = cur_shape[d.logical_to_phys[i]];
    transforms.emplace_back(TransformationType::TRANSPOSE, cur_shape, final_phys_shape,
                            d.element_type, d.logical_to_phys, id_perm_total, d.logical_to_phys);
    cur_shape = final_phys_shape;
  }

  // BROADCAST
  std::vector<int64_t> final_shape(shape.begin(), shape.end());
  if (cur_shape != final_shape) {
    std::vector<int64_t> broadcast_params(ndims);
    std::iota(broadcast_params.begin(), broadcast_params.end(), 0);
    std::vector<int64_t> identity_perm_n(ndims);
    std::iota(identity_perm_n.begin(), identity_perm_n.end(), 0);
    transforms.emplace_back(TransformationType::BROADCAST, cur_shape, final_shape, d.element_type,
                            broadcast_params, identity_perm_n, identity_perm_n);
  }

  return TransformationCreationResult{true, std::move(transforms), d.base_tensor, "COMPOUND"};
}

std::optional<TransformationCreationResult> TryCreateFoldedDecomposition(const at::Tensor& input,
                                                                         const std::string& op_name,
                                                                         size_t input_index) {
  auto folded = TryFoldedDecomposition(input, op_name, input_index);
  if (folded.has_value()) {
    return TransformationCreationResult{true, std::move(folded->transforms), folded->base_tensor,
                                        "COMPOUND_FOLDED"};
  }
  return std::nullopt;
}

// ============================================================================
// TryCreateEpilogue: Build INVERSE transformation chain for non-contiguous outputs
// ============================================================================

// Result of physical layout analysis for epilogue detection.
struct EpilogueLayoutInfo {
  std::vector<int64_t> active_dims;      // logical dims with size > 1
  std::vector<int64_t> phys_order;       // active dims sorted by stride descending
  std::vector<int64_t> phys_base;        // base shape per active dim (physical order)
  std::vector<int64_t> phys_view;        // view shape per active dim (physical order)
  std::vector<int64_t> phys_starts;      // per-dim slice starts (physical order)
  std::vector<int64_t> logical_to_phys;  // active-dim logical→physical mapping
  int64_t storage_elements;
  bool has_slice;
  bool is_transposed;
};

// Analyze physical layout: compute base shape from stride ratios, decompose offset.
// Returns nullopt if the tensor layout is unsupported (strided slice, broadcast, etc).
std::optional<EpilogueLayoutInfo> AnalyzeEpilogueLayout(c10::IntArrayRef shape,
                                                        c10::IntArrayRef strides, int64_t offset,
                                                        int64_t storage_elements,
                                                        const std::string& op_name,
                                                        size_t output_index) {
  int64_t ndims = shape.size();

  // Collect active dims and sort by stride descending (physical order)
  std::vector<int64_t> active_dims;
  for (int64_t i = 0; i < ndims; ++i) {
    if (shape[i] > 1) active_dims.push_back(i);
  }
  if (active_dims.empty()) return std::nullopt;

  std::vector<int64_t> phys_order = active_dims;
  std::sort(phys_order.begin(), phys_order.end(),
            [&](int64_t a, int64_t b) { return strides[a] > strides[b]; });
  int64_t num_active = static_cast<int64_t>(phys_order.size());

  // Compute base shape from stride ratios
  std::vector<int64_t> phys_base(num_active);
  std::vector<int64_t> phys_view(num_active);
  bool has_slice = false;

  for (int64_t pi = 0; pi < num_active; ++pi) {
    int64_t dim = phys_order[pi];
    phys_view[pi] = shape[dim];

    if (pi == 0) {
      phys_base[pi] = storage_elements / strides[dim];
    } else {
      int64_t outer_dim = phys_order[pi - 1];
      if (strides[dim] <= 0) return std::nullopt;
      phys_base[pi] = strides[outer_dim] / strides[dim];
      if (strides[outer_dim] != phys_base[pi] * strides[dim]) {
        TORCH_NEURONX_DEBUG("TryCreateEpilogue: non-contiguous stride gap", "op=", op_name,
                            "output_index=", output_index);
        return std::nullopt;
      }
    }

    if (pi == num_active - 1 && strides[dim] != 1) {
      TORCH_NEURONX_DEBUG("TryCreateEpilogue: innermost stride != 1", "op=", op_name,
                          "output_index=", output_index);
      return std::nullopt;
    }

    if (phys_base[pi] < phys_view[pi]) return std::nullopt;
    if (phys_base[pi] > phys_view[pi]) has_slice = true;
  }

  // Decompose offset into per-dim starts
  std::vector<int64_t> phys_starts(num_active, 0);
  {
    auto base_strides_phys = ComputeRowMajorStrides(phys_base);
    int64_t remaining = offset;
    for (int64_t pi = 0; pi < num_active; ++pi) {
      if (base_strides_phys[pi] > 0 && remaining >= base_strides_phys[pi]) {
        phys_starts[pi] = remaining / base_strides_phys[pi];
        remaining %= base_strides_phys[pi];
      }
    }
    if (remaining != 0) {
      TORCH_NEURONX_DEBUG("TryCreateEpilogue: offset decomposition failed", "op=", op_name,
                          "output_index=", output_index);
      return std::nullopt;
    }
    for (int64_t pi = 0; pi < num_active; ++pi) {
      if (phys_starts[pi] + phys_view[pi] > phys_base[pi]) return std::nullopt;
    }
  }

  // Build logical→physical mapping for active dims
  std::vector<int64_t> logical_to_phys_active(num_active);
  {
    std::unordered_map<int64_t, int64_t> dim_to_pi;
    for (int64_t pi = 0; pi < num_active; ++pi) dim_to_pi[phys_order[pi]] = pi;
    for (int64_t li = 0; li < num_active; ++li)
      logical_to_phys_active[li] = dim_to_pi[active_dims[li]];
  }

  bool is_transposed = false;
  for (int64_t i = 0; i < num_active; ++i) {
    if (logical_to_phys_active[i] != i) {
      is_transposed = true;
      break;
    }
  }

  if (!has_slice && !is_transposed) return std::nullopt;  // contiguous

  return EpilogueLayoutInfo{std::move(active_dims), std::move(phys_order),
                            std::move(phys_base),   std::move(phys_view),
                            std::move(phys_starts), std::move(logical_to_phys_active),
                            storage_elements,       has_slice,
                            is_transposed};
}

// Build full-ndims permutation vectors from active-dim layout analysis.
struct EpiloguePermutation {
  std::vector<int64_t> full_log_to_phys;  // logical_dim → physical_position
  std::vector<int64_t> full_phys_to_log;  // physical_position → logical_dim
  std::vector<int64_t> full_base_phys;    // base shape in physical order
  std::vector<int64_t> full_starts_phys;  // starts in physical order
  bool full_is_identity;
};

EpiloguePermutation BuildEpiloguePermutation(c10::IntArrayRef shape, c10::IntArrayRef strides,
                                             const EpilogueLayoutInfo& layout) {
  int64_t ndims = shape.size();
  int64_t num_active = static_cast<int64_t>(layout.active_dims.size());

  // Positions in full ndims occupied by active dims
  std::vector<int64_t> slots_for_active;
  for (int64_t i = 0; i < ndims; ++i) {
    if (shape[i] > 1) slots_for_active.push_back(i);
  }

  // Build full permutation: logical_dim → physical_position
  std::vector<int64_t> full_log_to_phys(ndims);
  for (int64_t i = 0; i < ndims; ++i) full_log_to_phys[i] = i;
  for (int64_t li = 0; li < num_active; ++li) {
    full_log_to_phys[slots_for_active[li]] = slots_for_active[layout.logical_to_phys[li]];
  }

  // Inverse: phys_position → logical_dim
  std::vector<int64_t> full_phys_to_log(ndims);
  for (int64_t i = 0; i < ndims; ++i) full_phys_to_log[full_log_to_phys[i]] = i;

  // Build full base shape and starts in physical order
  std::vector<int64_t> full_base_phys(ndims, 1);
  std::vector<int64_t> full_starts_phys(ndims, 0);
  for (int64_t p = 0; p < ndims; ++p) {
    int64_t log_dim = full_phys_to_log[p];
    if (shape[log_dim] <= 1) continue;
    for (int64_t li = 0; li < num_active; ++li) {
      if (layout.active_dims[li] == log_dim) {
        int64_t pi = layout.logical_to_phys[li];
        full_base_phys[p] = layout.phys_base[pi];
        full_starts_phys[p] = layout.phys_starts[pi];
        break;
      }
    }
  }

  bool full_is_identity = true;
  for (int64_t i = 0; i < ndims; ++i) {
    if (full_log_to_phys[i] != i) {
      full_is_identity = false;
      break;
    }
  }

  return EpiloguePermutation{std::move(full_log_to_phys), std::move(full_phys_to_log),
                             std::move(full_base_phys), std::move(full_starts_phys),
                             full_is_identity};
}

// ----------------------------------------------------------------------------
// Epilogue creation is a 2-tier strategy. Both tiers are sibling creators that
// return a TransformationCreationResult or nullopt:
//
//   Tier 1  AnalyzeEpilogueLayout  →  TryCreateDecomposedEpilogue
//           The view DECOMPOSES into  transpose ∘ slice  of a contiguous base.
//           Emits clean TRANSPOSE (+ SCATTER_IOTA only for the slice part).
//
//   Tier 2  TryCreateGenericScatterEpilogue
//           The view does NOT decompose (diagonal / select / non-multiple gap).
//           Falls back to a single RAW-STRIDE scatter (flat_idx = Σ iₖ·strideₖ).
//
//   none    Both tiers return nullopt → caller bounces through a contiguous
//           temp (CopyBackToOriginal), which is always correct.
//
// TryCreateEpilogue (below) is the orchestrator that wires these tiers together.
// ----------------------------------------------------------------------------

// Tier 1 creator: the layout was DECOMPOSED by AnalyzeEpilogueLayout into a
// transpose ∘ slice of a contiguous base. Build the inverse transformation
// chain (inverse TRANSPOSE, plus SCATTER_IOTA for the slice part) that writes a
// contiguous result back into the non-contiguous output's storage.
// Returns nullopt if storage bounds fail or SCATTER_IOTA is disabled for a slice.
std::optional<TransformationCreationResult> TryCreateDecomposedEpilogue(
    const at::Tensor& output, const EpilogueLayoutInfo& layout, c10::IntArrayRef shape,
    c10::IntArrayRef strides, int64_t offset, int64_t ndims, int64_t storage_elements,
    const std::string& op_name, size_t output_index) {
  // Build full-ndims permutation and base shape from the active-dim analysis.
  auto perm = BuildEpiloguePermutation(shape, strides, layout);

  // Verify storage bounds
  int64_t base_elements = 1;
  for (auto s : perm.full_base_phys) base_elements *= s;
  if (base_elements > storage_elements) return std::nullopt;
  {
    int64_t base_bytes = base_elements * output.element_size();
    constexpr int64_t kMaxBytes = 4LL * 1024 * 1024 * 1024;
    if (base_bytes >= kMaxBytes) return std::nullopt;
  }

  // Create base tensor
  auto base_cont_strides = ComputeRowMajorStrides(perm.full_base_phys);
  int64_t base_offset = offset;
  for (int64_t i = 0; i < ndims; ++i)
    base_offset -= perm.full_starts_phys[i] * base_cont_strides[i];
  if (base_offset < 0 || base_offset + base_elements > storage_elements) return std::nullopt;

  at::Tensor base_tensor = output.as_strided(c10::IntArrayRef(perm.full_base_phys),
                                             c10::IntArrayRef(base_cont_strides), base_offset);

  // Build transformation chain
  std::string element_type = ScalarTypeToElementTypeString(output.scalar_type());
  std::vector<int64_t> id_perm(ndims);
  std::iota(id_perm.begin(), id_perm.end(), 0);
  std::vector<int64_t> logical_shape(shape.begin(), shape.end());
  std::vector<TensorTransformation> transforms;
  std::vector<int64_t> cur_shape = logical_shape;

  // INVERSE TRANSPOSE
  if (!perm.full_is_identity) {
    std::vector<int64_t> physical_shape(ndims);
    for (int64_t p = 0; p < ndims; ++p) physical_shape[p] = logical_shape[perm.full_phys_to_log[p]];
    transforms.emplace_back(TransformationType::TRANSPOSE, cur_shape, physical_shape, element_type,
                            perm.full_phys_to_log, id_perm, perm.full_phys_to_log);
    cur_shape = physical_shape;
  }

  // SCATTER_IOTA for slice patterns.
  // Controlled by TORCH_NEURONX_ENABLE_SCATTER_IOTA (default ON).
  // When OFF (=0), patterns requiring SCATTER_IOTA fall back to contiguous temp +
  // CopyBackToOriginal.
  static const bool scatter_iota_enabled = []() {
    const char* env = std::getenv("TORCH_NEURONX_ENABLE_SCATTER_IOTA");
    return env == nullptr || std::string(env) != "0";
  }();
  if (layout.has_slice && !scatter_iota_enabled) {
    TORCH_NEURONX_DEBUG("TryCreateDecomposedEpilogue: SCATTER_IOTA disabled via env var",
                        "op=", op_name, "output_index=", output_index);
    return std::nullopt;
  }
  if (layout.has_slice) {
    std::vector<int64_t> phys_strides(ndims);
    for (int64_t p = 0; p < ndims; ++p) phys_strides[p] = strides[perm.full_phys_to_log[p]];

    int64_t scatter_storage_span = storage_elements;
    std::vector<int64_t> scatter_output_shape = {scatter_storage_span};

    std::vector<int64_t> scatter_params;
    scatter_params.insert(scatter_params.end(), phys_strides.begin(), phys_strides.end());
    scatter_params.push_back(offset);
    scatter_params.push_back(scatter_storage_span);

    transforms.emplace_back(TransformationType::SCATTER_IOTA, cur_shape, scatter_output_shape,
                            element_type, scatter_params, id_perm, id_perm);

    base_tensor = output.as_strided({scatter_storage_span}, {1}, 0);
  }

  std::string pattern_name = layout.has_slice
                                 ? (perm.full_is_identity ? "EPILOGUE_SLICE" : "EPILOGUE_COMPOUND")
                                 : "EPILOGUE_TRANSPOSE";
  return TransformationCreationResult{true, std::move(transforms), base_tensor, pattern_name};
}

// Orchestrator: pre-validate the output, then try the 2-tier strategy described
// above. See the strategy map comment preceding TryCreateDecomposedEpilogue.
std::optional<TransformationCreationResult> TryCreateEpilogue(const at::Tensor& output,
                                                              const std::string& op_name,
                                                              size_t output_index) {
  auto shape = output.sizes();
  auto strides = output.strides();
  int64_t offset = output.storage_offset();
  int64_t ndims = shape.size();

  if (ndims == 0 || output.is_contiguous()) return std::nullopt;

  // Reject broadcast and negative strides
  for (int64_t i = 0; i < ndims; ++i) {
    if (strides[i] < 0) return std::nullopt;
    if (strides[i] == 0 && shape[i] > 1) {
      TORCH_NEURONX_DEBUG("TryCreateEpilogue: broadcast view", "op=", op_name,
                          "output_index=", output_index);
      return std::nullopt;
    }
  }

  int64_t storage_elements =
      static_cast<int64_t>(output.storage().nbytes() / output.element_size());

  // Tier 1: can the layout be decomposed into transpose ∘ slice of a contiguous
  // base? If so, build the precise inverse transform chain.
  if (auto layout =
          AnalyzeEpilogueLayout(shape, strides, offset, storage_elements, op_name, output_index)) {
    return TryCreateDecomposedEpilogue(output, *layout, shape, strides, offset, ndims,
                                       storage_elements, op_name, output_index);
  }

  // Tier 2: the analytical decomposition failed. Fall back to a generic
  // raw-stride scatter for diagonal / select / non-multiple-stride-gap patterns.
  // Returns nullopt if any safety guard fails (overlap / OOB / 4GB / scatter
  // disabled), in which case TryCreateEpilogue returns nullopt and the caller's
  // CPU-bounce fallback (CopyBackToOriginal) takes over.
  return TryCreateGenericScatterEpilogue(output, shape, strides, offset, ndims, storage_elements,
                                         op_name, output_index);
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

// ScatterIotaHandler
size_t ScatterIotaHandler::FindGroupEnd(const std::vector<TensorTransformation>& transformations,
                                        size_t start_index,
                                        const TransformationState& state) const {
  return start_index + 1;  // Process individually
}

TransformationTask ScatterIotaHandler::ProcessGroup(size_t input_index, TransformationState& state,
                                                    const std::vector<TensorTransformation>& group,
                                                    const std::string& op_name) const {
  TORCH_NEURONX_DEBUG("Processing SCATTER_IOTA group", "op=", op_name, "input_index=", input_index);

  const auto& transform = group[0];

  // Extract strides, offset, storage_span from params
  // params layout: [stride_0, ..., stride_{n-1}, storage_offset, storage_span]
  size_t ndims = transform.input_shape.size();
  TORCH_CHECK(transform.params.size() == ndims + 2, "SCATTER_IOTA params size mismatch: expected ",
              ndims + 2, " got ", transform.params.size());

  std::vector<int64_t> view_strides(transform.params.begin(), transform.params.begin() + ndims);
  int64_t storage_offset = transform.params[ndims];
  int64_t storage_span = transform.params[ndims + 1];

  // Update state — output is 1D flat storage
  state.current_shape = transform.output_shape;

  // Build ScatterIotaParams
  ScatterIotaParams params;
  params.view_strides = view_strides;
  params.storage_offset = storage_offset;
  params.storage_span = storage_span;

  return TransformationTask{.input_index = input_index,
                            .type = TransformationType::SCATTER_IOTA,
                            .input_shape = transform.input_shape,
                            .output_shape = transform.output_shape,
                            .element_type = transform.element_type,
                            .current_perm = state.current_perm,
                            .params = params};
}

}  // namespace Handlers

}  // namespace lazy
}  // namespace c10_neuron
