#pragma once

#include <ATen/Tensor.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>

namespace at {
namespace native {

// Neuron native function declarations
at::Tensor empty_neuron(c10::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt,
                        c10::optional<c10::Layout> layout_opt,
                        c10::optional<c10::Device> device_opt, c10::optional<bool> pin_memory_opt,
                        c10::optional<c10::MemoryFormat> memory_format_opt);

at::Tensor empty_strided_neuron(at::IntArrayRef size, at::IntArrayRef stride,
                                c10::optional<c10::ScalarType> dtype_opt,
                                c10::optional<c10::Layout> layout_opt,
                                c10::optional<c10::Device> device_opt,
                                c10::optional<bool> pin_memory_opt);

at::Tensor new_empty_neuron(const at::Tensor& self, at::IntArrayRef size,
                            c10::optional<c10::ScalarType> dtype,
                            c10::optional<c10::Layout> layout_opt,
                            c10::optional<c10::Device> device_opt,
                            c10::optional<bool> pin_memory_opt);

// Resize operations
const at::Tensor& resize_neuron(const at::Tensor& self, c10::IntArrayRef size,
                                std::optional<c10::MemoryFormat> memory_format);

// Clone operation
at::Tensor clone_neuron(const at::Tensor& self, c10::optional<c10::MemoryFormat> memory_format);

// Scalar extraction
at::Scalar _local_scalar_dense_neuron(const at::Tensor& self);
// Set operations - implements aten::set_ variants for Neuron tensors
// These operations modify tensor metadata to share storage with another tensor or storage

/**
 * @brief Implements aten::set_.source_Storage_storage_offset
 *
 * Sets the underlying storage, size, and strides. Changes to elements in one tensor will be
 * reflected in the other.
 *
 * @param result Target tensor to modify (must be Neuron tensor)
 * @param storage Storage to use (must be Neuron storage)
 * @param storage_offset Offset in the storage (must be non-negative)
 * @param size Desired size
 * @param stride Desired stride (empty for C-contiguous strides)
 * @return Reference to modified result tensor
 * @throws RuntimeError if storage_offset is negative or storage is not Neuron storage
 */
at::Tensor& set_storage_neuron(at::Tensor& result, c10::Storage storage, int64_t storage_offset,
                               c10::IntArrayRef size, c10::IntArrayRef stride);

/**
 * @brief Implements aten::set_.source_Storage
 *
 * Sets the underlying storage, inferring size from storage bytes and tensor dtype.
 * The tensor will be 1D with size = storage.nbytes() / dtype.itemsize().
 *
 * @param result Target tensor to modify (must be Neuron tensor)
 * @param source Storage to use (must be Neuron storage)
 * @return Reference to modified result tensor
 * @throws RuntimeError if storage is not Neuron storage or size calculation fails
 */
at::Tensor& set_neuron(at::Tensor& result, c10::Storage source);

/**
 * @brief Implements aten::set_.source_Tensor
 *
 * Sets tensor to share the same storage and have the same size and strides as source.
 * Changes to elements in one tensor will be reflected in the other.
 *
 * @param result Target tensor to modify (must be Neuron tensor)
 * @param source Source tensor to copy metadata from (must be Neuron tensor)
 * @return Reference to modified result tensor
 * @throws RuntimeError if either tensor is not on Neuron device
 */
at::Tensor& set_tensor_neuron(at::Tensor& result, const at::Tensor& source);

/**
 * @brief Implements aten::set_ (empty variant)
 *
 * Sets tensor to empty state with zero size and empty storage.
 *
 * @param result Target tensor to modify (must be Neuron tensor)
 * @return Reference to modified result tensor
 * @throws RuntimeError if tensor is not on Neuron device
 */
at::Tensor& set_empty_neuron(at::Tensor& result);

/**
 * @brief Implements aten::set_.source_Tensor_storage_offset
 *
 * Sets tensor to use source tensor's storage with additional offset and new size/stride.
 *
 * @param result Target tensor to modify (must be Neuron tensor)
 * @param source Source tensor providing storage (must be Neuron tensor, must be contiguous)
 * @param storage_offset Additional offset into source storage (must be non-negative)
 * @param size Desired tensor size
 * @param stride Desired tensor stride
 * @return Reference to modified result tensor
 * @throws RuntimeError if source is not contiguous or storage_offset is negative
 */
at::Tensor& set_tensor_storage_offset_neuron(at::Tensor& result, const at::Tensor& source,
                                             c10::SymInt storage_offset, c10::SymIntArrayRef size,
                                             c10::SymIntArrayRef stride);

// Type compatibility check for tensor data assignment
bool _has_compatible_shallow_copy_type_neuron(const at::Tensor& self, const at::Tensor& from);

}  // namespace native
}  // namespace at

// Copy operations in torch_neuronx namespace
namespace torch_neuronx {

// View operations
at::Tensor view_neuron(const at::Tensor& self, at::IntArrayRef size);

at::Tensor unfold_neuron(const at::Tensor& self, int64_t d, int64_t size, int64_t step);

at::Tensor as_strided_neuron(const at::Tensor& self, at::IntArrayRef size, at::IntArrayRef stride,
                             c10::optional<int64_t> storage_offset);

at::Tensor _reshape_alias_neuron(const at::Tensor& self, at::IntArrayRef sizes,
                                 at::IntArrayRef strides);

// Copy from and resize operation
at::Tensor _copy_from_and_resize_neuron(const at::Tensor& self, const at::Tensor& dst);

// Select backward
at::Tensor select_backward_neuron(const at::Tensor& grad, c10::SymIntArrayRef input_sizes,
                                  int64_t dim, c10::SymInt index);

// Slice backward
at::Tensor slice_backward_neuron(const at::Tensor& grad_output, c10::SymIntArrayRef input_sizes,
                                 int64_t dim, c10::SymInt start, c10::SymInt end, c10::SymInt step);

// argsort.stable
at::Tensor argsort_stable_neuron(const at::Tensor& self, bool stable, int64_t dim, bool descending);

void record_stream_neuron(at::Tensor& tensor, c10::Stream stream);

}  // namespace torch_neuronx
