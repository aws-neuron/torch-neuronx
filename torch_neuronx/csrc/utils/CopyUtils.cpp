#include "CopyUtils.h"

#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/NeuronStorageImpl.h"
#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"
#include "torch_neuronx/csrc/core/utils/PlatformUtils.h"
#include "torch_neuronx/csrc/ops/InternalOps.h"
#include "torch_neuronx/csrc/utils/NonTemporalMemcpy.h"

namespace torch_neuronx {
namespace utils {

nrt_tensor_t* get_nrt_tensor(const at::Tensor& tensor, bool non_blocking) {
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
              "Expected Neuron tensor, got ", tensor.device());

  // Get the storage implementation
  auto storage_impl = tensor.storage().unsafeGetStorageImpl();
  auto neuron_storage = static_cast<c10_neuron::NeuronStorageImpl*>(storage_impl);
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

  size_t size_bytes = src.nbytes();
  // Skip copy if there are no bytes to copy (empty tensor)
  if (size_bytes == 0) {
    return;
  }
  int device = dst.device().index();
  size_t byte_offset = dst.storage_offset() * dst.element_size();

  if (!at::neuron::utils::IsSyncModeEnabled()) {
    void* dst_ptr = dst.storage().mutable_data();
    auto dst_tensor_ptr = c10_neuron::NeuronCachingAllocator::findTensorPtr(dst_ptr);

    auto write_kernel = std::make_unique<at::neuron::WriteDirectKernelExecution>(
        "neuron::copy::cpu_to_neuron", src.data_ptr(),
        at::neuron::TensorDataRef{std::move(dst_tensor_ptr), dst_ptr}, byte_offset, size_bytes,
        device);

    bool use_bounce = non_blocking || src.numel() == 1;
    if (use_bounce) {
      write_kernel->AllocateBounceBuffer(size_bytes);
    }

    auto context = std::make_unique<at::neuron::OperationContext>(std::move(write_kernel));
    auto stream = at::neuron::getCurrentNeuronStream(device);
    at::neuron::SubmitOperationContext(stream, std::move(context));
    if (!use_bounce) {
      stream.synchronize();
    }
    return;
  }

  nrt_tensor_t* nrt_dst = get_nrt_tensor(dst, non_blocking);
  TORCH_CHECK(nrt_dst != nullptr, "Failed to get Neuron tensor handle");

  void* src_data_ptr = src.data_ptr();
  nrt_copy_cpu_to_neuron(src_data_ptr, nrt_dst, byte_offset, size_bytes);
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

  size_t size_bytes = dst.nbytes();
  // Skip copy if there are no bytes to copy (empty tensor)
  if (size_bytes == 0) {
    return;
  }

  int device = src.device().index();
  void* dst_data_ptr = dst.data_ptr();
  // Calculate offset based on tensor's storage offset
  size_t byte_offset = src_to_copy.storage_offset() * src_to_copy.element_size();

  if (!at::neuron::utils::IsSyncModeEnabled()) {
    auto stream = at::neuron::getCurrentNeuronStream(device);
    void* src_ptr = src_to_copy.storage().mutable_data();
    auto src_tensor_ptr = c10_neuron::NeuronCachingAllocator::findTensorPtr(src_ptr);
    auto read_kernel = std::make_unique<at::neuron::ReadDirectKernelExecution>(
        "neuron::copy::neuron_to_cpu",
        at::neuron::TensorDataRef{std::move(src_tensor_ptr), src_ptr}, dst_data_ptr, byte_offset,
        size_bytes, device);
    auto context = std::make_unique<at::neuron::OperationContext>(std::move(read_kernel));
    at::neuron::SubmitOperationContext(stream, std::move(context));
    stream.synchronize();
    return;
  }

  nrt_tensor_t* nrt_src = get_nrt_tensor(src_to_copy, non_blocking);
  TORCH_CHECK(nrt_src != nullptr, "Failed to get Neuron tensor handle");

  nrt_copy_neuron_to_cpu(nrt_src, dst_data_ptr, byte_offset, size_bytes);
}

void nrt_copy_neuron_to_cpu(nrt_tensor_t* src_data_ptr, void* dst_data_ptr, size_t src_offset,
                            size_t size) {
  TORCH_CHECK(src_data_ptr != nullptr, "Source data pointer must not be null");
  TORCH_CHECK(size > 0, "Destination size must be valid");
  NRT_STATUS status = nrt_tensor_read(src_data_ptr, dst_data_ptr, src_offset, size);
  if (status != NRT_SUCCESS) {
    throw std::runtime_error("Failed to copy data from Neuron device. Status: " +
                             std::to_string(status));
  }
}

void nrt_copy_cpu_to_neuron(void* src_data_ptr, nrt_tensor_t* dst_data_ptr, size_t dst_offset,
                            size_t size) {
  TORCH_CHECK(dst_data_ptr != nullptr, "Destination data pointer must not be null");
  TORCH_CHECK(size > 0, "Destination size must be valid");
  // Skip copy if there are no bytes to copy (empty tensor)
  NRT_STATUS status = nrt_tensor_write(dst_data_ptr, src_data_ptr, dst_offset, size);
  if (status != NRT_SUCCESS) {
    throw std::runtime_error("Failed to copy data to Neuron device. Status: " +
                             std::to_string(status));
  }
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

  // Ensure tensors have same size
  TORCH_CHECK(src_to_copy.nbytes() == dst.nbytes(),
              "Source and destination must have same number of bytes");

  int device = src.device().index();
  size_t size_bytes = src_to_copy.nbytes();
  // Skip copy if there are no bytes to copy (empty tensor)
  if (size_bytes == 0) {
    return;
  }

  // Calculate offsets based on tensor's storage offset
  size_t src_byte_offset = src_to_copy.storage_offset() * src_to_copy.element_size();
  size_t dst_byte_offset = dst.storage_offset() * dst.element_size();

  if (!at::neuron::utils::IsSyncModeEnabled()) {
    void* src_ptr = src_to_copy.storage().mutable_data();
    void* dst_ptr = dst.storage().mutable_data();
    auto src_tensor_ptr = c10_neuron::NeuronCachingAllocator::findTensorPtr(src_ptr);
    auto dst_tensor_ptr = c10_neuron::NeuronCachingAllocator::findTensorPtr(dst_ptr);
    auto copy_kernel = std::make_unique<at::neuron::CopyDirectKernelExecution>(
        "neuron::copy::neuron_to_neuron",
        at::neuron::TensorDataRef{std::move(src_tensor_ptr), src_ptr},
        at::neuron::TensorDataRef{std::move(dst_tensor_ptr), dst_ptr}, src_byte_offset,
        dst_byte_offset, size_bytes, device);
    auto context = std::make_unique<at::neuron::OperationContext>(std::move(copy_kernel));
    auto stream = at::neuron::getCurrentNeuronStream(device);
    at::neuron::SubmitOperationContext(stream, std::move(context));

    if (!non_blocking) {
      stream.synchronize();
    }
    return;
  }

  nrt_tensor_t* nrt_src = get_nrt_tensor(src_to_copy, non_blocking);
  nrt_tensor_t* nrt_dst = get_nrt_tensor(dst, non_blocking);
  TORCH_CHECK(nrt_src != nullptr, "Failed to get source Neuron tensor handle");
  TORCH_CHECK(nrt_dst != nullptr, "Failed to get destination Neuron tensor handle");
  NRT_STATUS status = nrt_tensor_copy(nrt_src, src_byte_offset,  // src + offset
                                      nrt_dst, dst_byte_offset,  // dst + offset
                                      size_bytes);

  TORCH_CHECK(status == NRT_SUCCESS, "Failed to copy between Neuron tensors. Status: ", status);
}

}  // namespace utils
}  // namespace torch_neuronx
