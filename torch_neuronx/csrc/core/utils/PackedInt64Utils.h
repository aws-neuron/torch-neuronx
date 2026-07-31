#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace torch_neuronx {
namespace packed_int64 {

/// Compute physical shape for packed int64: appends trailing dim of 2.
/// Scalar (empty shape) becomes {2} (append the lane dim to the empty logical
/// shape). NOT {1, 2}: that collides with a 1-D {1} tensor and disagrees with
/// the output-packing / logical_view convention, which use {2} for a scalar.
inline std::vector<int64_t> physical_sizes(c10::IntArrayRef logical_sizes) {
  if (logical_sizes.empty()) {
    return {2};
  }
  std::vector<int64_t> out;
  out.reserve(logical_sizes.size() + 1);
  for (int64_t dim : logical_sizes) {
    out.push_back(dim);
  }
  out.push_back(2);
  return out;
}

/// Compute physical strides for packed int64: doubles all strides, appends 1.
/// Scalar (empty strides) becomes {1} (single contiguous lane axis).
inline std::vector<int64_t> physical_strides(c10::IntArrayRef logical_strides) {
  if (logical_strides.empty()) {
    return {1};
  }
  std::vector<int64_t> out;
  out.reserve(logical_strides.size() + 1);
  for (int64_t stride : logical_strides) {
    out.push_back(stride * 2);
  }
  out.push_back(1);
  return out;
}

/// Create a physical int32 view of a packed int64 tensor.
/// The returned tensor shares storage with the input but has dtype=int32,
/// shape=(*logical_shape, 2), strides=(*logical_strides*2, 1).
inline torch::Tensor physical_view(const torch::Tensor& t) {
  TORCH_CHECK(t.scalar_type() == c10::ScalarType::Long,
              "packed_int64::physical_view: expected int64 tensor, got ", t.scalar_type());

  auto sizes = physical_sizes(t.sizes());
  auto strides = physical_strides(t.strides());
  int64_t offset = static_cast<int64_t>(t.storage_offset()) * 2;

  // Create empty int32 tensor and set_ it to share the same storage
  auto out = torch::empty({0}, t.options().dtype(c10::ScalarType::Int));
  out.set_(t.storage(), offset, sizes, strides);
  return out;
}

/// Create a strided int32 view of just the low words (lane 0).
/// Shape = logical shape, strides = physical strides without trailing dim.
inline torch::Tensor low_word_alias(const torch::Tensor& t) {
  TORCH_CHECK(t.scalar_type() == c10::ScalarType::Long,
              "packed_int64::low_word_alias: expected int64 tensor, got ", t.scalar_type());

  auto sizes = t.sizes().vec();
  // Offset in int32 elements: storage_offset * 2 (each int64 = 2 int32s), lane 0
  int64_t offset = static_cast<int64_t>(t.storage_offset()) * 2;

  std::vector<int64_t> result_strides;
  if (!sizes.empty()) {
    // Non-scalar: double the logical strides, drop the trailing "1" lane stride.
    auto phys_strides = physical_strides(t.strides());
    result_strides.assign(phys_strides.begin(), phys_strides.end() - 1);
  }
  // Scalar (rank-0): sizes and strides are both empty; the offset alone selects
  // the single low word. (set_ requires size and stride lengths to match.)

  // Create int32 tensor sharing the same untyped storage
  auto out = at::empty({0}, t.options().dtype(c10::ScalarType::Int));
  out.set_(t.storage(), offset, sizes, result_strides);
  return out;
}

/// Reinterpret a contiguous packed int32 pair-stream as int64 (zero-copy).
/// Inverse of physical_view: given an int32 tensor of shape (*logical, 2) whose
/// storage already holds [low, high] pairs (e.g. a fused-materialize execution
/// output), returns an int64 tensor of shape (*logical) sharing that storage.
/// Used by UpcastExecutionOutputs to bind a fused packed int64 output without a
/// materialize copy.
inline torch::Tensor logical_view(const torch::Tensor& packed) {
  TORCH_CHECK(packed.scalar_type() == c10::ScalarType::Int,
              "packed_int64::logical_view: expected int32 tensor, got ", packed.scalar_type());
  TORCH_CHECK(packed.dim() >= 1 && packed.size(-1) == 2,
              "packed_int64::logical_view: expected trailing packed dim of 2, got shape ",
              packed.sizes());
  TORCH_CHECK(packed.is_contiguous(),
              "packed_int64::logical_view: expected contiguous packed input");

  auto psizes = packed.sizes();
  std::vector<int64_t> logical_sizes(psizes.begin(), psizes.end() - 1);

  // Contiguous int64 strides for the logical shape.
  std::vector<int64_t> logical_strides(logical_sizes.size());
  int64_t s = 1;
  for (int64_t i = static_cast<int64_t>(logical_sizes.size()) - 1; i >= 0; --i) {
    logical_strides[i] = s;
    s *= logical_sizes[i];
  }

  // int32 storage_offset -> int64 element offset (2 int32 per int64).
  int64_t offset = static_cast<int64_t>(packed.storage_offset()) / 2;

  auto out = at::empty({0}, packed.options().dtype(c10::ScalarType::Long));
  out.set_(packed.storage(), offset, logical_sizes, logical_strides);
  return out;
}

/// Create a strided int32 view of just the high words (lane 1).
/// Identical to low_word_alias but offset by one int32 slot, so it selects the
/// upper 32 bits of each packed int64 element. Used by the (optional) overflow
/// check to compare the stored high lane against the sign extension of the low
/// lane.
inline torch::Tensor high_word_alias(const torch::Tensor& t) {
  TORCH_CHECK(t.scalar_type() == c10::ScalarType::Long,
              "packed_int64::high_word_alias: expected int64 tensor, got ", t.scalar_type());

  auto sizes = t.sizes().vec();
  // Offset in int32 elements: storage_offset * 2, plus 1 to land on lane 1.
  int64_t offset = static_cast<int64_t>(t.storage_offset()) * 2 + 1;

  std::vector<int64_t> result_strides;
  if (!sizes.empty()) {
    auto phys_strides = physical_strides(t.strides());
    result_strides.assign(phys_strides.begin(), phys_strides.end() - 1);
  }
  // Scalar (rank-0): sizes and strides are both empty; the offset (lane 1)
  // alone selects the single high word.

  auto out = at::empty({0}, t.options().dtype(c10::ScalarType::Int));
  out.set_(t.storage(), offset, sizes, result_strides);
  return out;
}

/// Extract contiguous low words from a packed int64 tensor.
/// Allocates a new contiguous int32 buffer and copies via strided copy.
/// This is the C++ equivalent of packed_int64_execution_view(contiguous=True).
torch::Tensor execution_view(const torch::Tensor& t);

/// Materialize an int32 execution result into a freshly allocated packed int64
/// tensor. The high lane is sign-extended from the low word (high = low >> 31)
/// so negative results are preserved.
///
/// IMPORTANT: the output is allocated with torch::empty (which maps directly to
/// the empty_strided_neuron factory and does NOT redispatch). Allocating the
/// int64 output via composite ops such as torch::zeros would re-enter the
/// Neuron dispatcher, get downcast back to int32, and recurse infinitely
/// through UpcastExecutionOutputs -> NeuronSafeCast -> here.
///
/// This is the C++ equivalent of packed_storage.materialize_int64_from_i32.
torch::Tensor materialize_from_i32(const torch::Tensor& low_words);

}  // namespace packed_int64
}  // namespace torch_neuronx
