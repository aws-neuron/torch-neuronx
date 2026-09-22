#pragma once
#include <nrt/nrt.h>
#include <torch/torch.h>

#include "torch_neuronx/csrc/core/utils/TensorContext.h"

namespace torch_neuronx {
namespace utils {

void copy_cpu_to_neuron(const at::Tensor& src, at::Tensor& dst, bool non_blocking = false);
void copy_neuron_to_cpu(const at::Tensor& src, at::Tensor& dst, bool non_blocking = false);
void nrt_copy_neuron_to_cpu(nrt_tensor_t* src_data_ptr, void* dst_data_ptr, size_t src_offset,
                            size_t size);
void nrt_copy_cpu_to_neuron(void* src_data_ptr, nrt_tensor_t* dst_data_ptr, size_t dst_offset,
                            size_t size);
void copy_neuron_to_neuron(const at::Tensor& src, at::Tensor& dst);
// Internal submission helper. The dispatcher validates the NRT cast contract
// and materializes a contiguous source before calling it.
void cast_neuron_to_neuron(const at::Tensor& src, at::Tensor& dst);

/// Strided neuron->neuron copy using only NRT/stream primitives.
/// Submits one stream-ordered device copy op per contiguous span.
void copy_strided_neuron_to_neuron(const at::Tensor& src, at::Tensor& dst);

/// Strided neuron->neuron copy preferring the on-device NKI kernel.
/// Default entry point for a strided copy.
void nki_copy_strided_neuron_to_neuron(const at::Tensor& src, at::Tensor& dst);

nrt_tensor_t* get_nrt_tensor(const at::Tensor& tensor, bool non_blocking = false);
void write_to_noncontiguous_neuron(at::Tensor& dst, const at::Tensor& src);

inline bool is_nrt_tensor_cast_supported(c10::ScalarType src_dtype, c10::ScalarType dst_dtype) {
  using ST = c10::ScalarType;
  return (src_dtype == ST::BFloat16 && dst_dtype == ST::Float) ||
         (src_dtype == ST::Float && dst_dtype == ST::BFloat16) ||
         (src_dtype == ST::Half && dst_dtype == ST::Float) ||
         (src_dtype == ST::Float && dst_dtype == ST::Half);
}

/// On-device non-contiguous write — central entry point for all copy paths.
///
/// Writes contiguous `src` into potentially non-contiguous `dst` on Neuron,
/// using on-device MLIR transpose + flat D2D copy when dst is a simple
/// transpose pattern (the common case for attention Q/K/V layouts).
///
/// Falls back to write_to_noncontiguous_neuron (CPU bounce) for patterns
/// that are not simple transposes: broadcast (stride=0), sliced views
/// with gaps, non-dense storage, nonzero storage_offset, etc.
///
/// Preconditions (caller must ensure):
///   - Both tensors on the same Neuron device
///   - `src` is contiguous
///   - `src.dtype() == dst.dtype()` (asserted in debug builds; no dtype conversion)
void CopyToNonContiguous(at::Tensor& dst, const at::Tensor& src);

}  // namespace utils
}  // namespace torch_neuronx
