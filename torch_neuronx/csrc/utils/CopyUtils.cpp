#include "CopyUtils.h"

#include "../core/NeuronStorageImpl.h"
#include "../core/allocator/NeuronCachingAllocator.h"
#include "../ops/InternalOps.h"
#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"

namespace torch_neuronx {
namespace utils {

nrt_tensor_t* get_nrt_tensor(const at::Tensor& tensor, bool non_blocking) {
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
              "Expected Neuron tensor, got ", tensor.device());

  // Get the storage implementation
  auto storage_impl = tensor.storage().unsafeGetStorageImpl();
  auto neuron_storage = static_cast<c10_neuron::NeuronStorageImpl*>(storage_impl);
  // Given we employ a synchronization level for all operations, this serves as
  // a validation for now.
  TORCH_CHECK(neuron_storage->is_valid(non_blocking), "Neuron tensor is not valid");
  return neuron_storage->neuron_tensor();
}

void copy_cpu_to_neuron(const at::Tensor& src, at::Tensor& dst, bool non_blocking) {
  TORCH_CHECK(src.is_cpu(), "Source tensor must be CPU tensor");
  TORCH_CHECK(dst.device().type() == c10::DeviceType::PrivateUse1,
              "Destination must be Neuron tensor");

  TORCH_CHECK(src.is_contiguous(),
              "Source CPU tensor must be contiguous. "
              "Please call .contiguous() on CPU tensor before copying to Neuron");
  TORCH_CHECK(dst.is_contiguous(), "Destination Neuron tensor must be contiguous");

  if (!non_blocking) {
    at::neuron::synchronize(dst.device().index());
  }

  nrt_tensor_t* nrt_dst = get_nrt_tensor(dst, non_blocking);
  TORCH_CHECK(nrt_dst != nullptr, "Failed to get Neuron tensor handle");

  // Calculate offset based on tensor's storage offset
  size_t byte_offset = dst.storage_offset() * dst.element_size();

  // Skip copy if there are no bytes to copy (empty tensor)
  if (src.nbytes() == 0) {
    return;
  }

  NRT_STATUS status = nrt_tensor_write(nrt_dst, src.data_ptr(), byte_offset, src.nbytes());

  TORCH_CHECK(status == NRT_SUCCESS, "Failed to copy data to Neuron device. Status: ", status);
}

void copy_neuron_to_cpu(const at::Tensor& src, at::Tensor& dst, bool non_blocking) {
  TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1, "Source must be Neuron tensor");
  TORCH_CHECK(dst.is_cpu(), "Destination tensor must be CPU tensor");

  // If source is not contiguous but we need contiguous data, make it contiguous first
  at::Tensor src_to_copy = src;
  if (!src.is_contiguous() && dst.is_contiguous()) {
    // Use internal contiguous operation to avoid dispatcher recursion
    src_to_copy = torch_neuronx::ops::contiguous_internal(src);
  }

  TORCH_CHECK(dst.is_contiguous(),
              "Destination CPU tensor must be contiguous for direct copy from Neuron");

  if (!non_blocking) {
    at::neuron::synchronize(src.device().index());
  }

  nrt_tensor_t* nrt_src = get_nrt_tensor(src_to_copy, non_blocking);
  TORCH_CHECK(nrt_src != nullptr, "Failed to get Neuron tensor handle");

  // Calculate offset based on tensor's storage offset
  size_t byte_offset = src_to_copy.storage_offset() * src_to_copy.element_size();

  // Skip copy if there are no bytes to copy (empty tensor)
  if (dst.nbytes() == 0) {
    return;
  }

  NRT_STATUS status = nrt_tensor_read(nrt_src, dst.data_ptr(), byte_offset, dst.nbytes());

  TORCH_CHECK(status == NRT_SUCCESS, "Failed to copy data from Neuron device. Status: ", status);
}

void copy_neuron_to_neuron(const at::Tensor& src, at::Tensor& dst, bool non_blocking) {
  TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1, "Source must be Neuron tensor");
  TORCH_CHECK(dst.device().type() == c10::DeviceType::PrivateUse1,
              "Destination must be Neuron tensor");

  // Check if on same device (required by nrt_tensor_copy for D2D)
  TORCH_CHECK(src.device().index() == dst.device().index(),
              "Device-to-device copy requires tensors on same Neuron core");

  // If source is not contiguous but we need contiguous data, make it contiguous first
  at::Tensor src_to_copy = src;
  if (!src.is_contiguous() && dst.is_contiguous()) {
    // Use internal contiguous operation to avoid dispatcher recursion
    src_to_copy = torch_neuronx::ops::contiguous_internal(src);
  }

  TORCH_CHECK(dst.is_contiguous(), "Destination Neuron tensor must be contiguous");

  if (!non_blocking) {
    at::neuron::synchronize(src.device().index());
    at::neuron::synchronize(dst.device().index());
  }

  nrt_tensor_t* nrt_src = get_nrt_tensor(src_to_copy, non_blocking);
  nrt_tensor_t* nrt_dst = get_nrt_tensor(dst, non_blocking);
  TORCH_CHECK(nrt_src != nullptr, "Failed to get source Neuron tensor handle");
  TORCH_CHECK(nrt_dst != nullptr, "Failed to get destination Neuron tensor handle");

  // Calculate offsets based on tensor's storage offset
  size_t src_byte_offset = src_to_copy.storage_offset() * src_to_copy.element_size();
  size_t dst_byte_offset = dst.storage_offset() * dst.element_size();

  // Ensure tensors have same size
  TORCH_CHECK(src_to_copy.nbytes() == dst.nbytes(),
              "Source and destination must have same number of bytes");

  // Skip copy if there are no bytes to copy (empty tensor)
  if (src_to_copy.nbytes() == 0) {
    return;
  }

  NRT_STATUS status = nrt_tensor_copy(nrt_src, src_byte_offset,  // src + offset
                                      nrt_dst, dst_byte_offset,  // dst + offset
                                      src_to_copy.nbytes());

  TORCH_CHECK(status == NRT_SUCCESS, "Failed to copy between Neuron tensors. Status: ", status);
}

}  // namespace utils
}  // namespace torch_neuronx
