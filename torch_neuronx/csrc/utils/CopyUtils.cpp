#include "CopyUtils.h"

#include <c10/util/Logging.h>

#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/NeuronStorageImpl.h"
#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"
#include "torch_neuronx/csrc/core/allocator/NeuronHostAllocator.h"
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

  void* dst_ptr = dst.storage().mutable_data();
  auto dst_tensor_ptr = c10_neuron::NeuronCachingAllocator::findTensorPtr(dst_ptr);

  auto write_kernel = std::make_unique<at::neuron::WriteDirectKernelExecution>(
      "neuron::copy::cpu_to_neuron", src.data_ptr(),
      at::neuron::TensorDataRef{std::move(dst_tensor_ptr), dst_ptr}, byte_offset, size_bytes,
      device);

  auto stream = at::neuron::getCurrentNeuronStream(device);
  bool use_bounce = non_blocking || src.numel() == 1;

  if (use_bounce) {
    auto* host_alloc = torch_neuronx::GetNeuronHostAllocatorAsHostAllocator();
    bool recorded = host_alloc->record_event(src.data_ptr(), src.storage().data_ptr().get_context(),
                                             c10::Stream(stream.unwrap()));
    // record_event returns false when src wasn't allocated by NeuronHostAllocator
    // (e.g. regular CPU tensor), so fall back to bounce buffer for safe async copy.
    if (!recorded) {
      write_kernel->AllocateBounceBuffer(size_bytes);
    }
  }

  auto context = std::make_unique<at::neuron::OperationContext>(std::move(write_kernel));
  at::neuron::SubmitOperationContext(stream, std::move(context));
  if (!use_bounce) {
    stream.synchronize();
  }
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

  auto stream = at::neuron::getCurrentNeuronStream(device);
  void* src_ptr = src_to_copy.storage().mutable_data();
  auto src_tensor_ptr = c10_neuron::NeuronCachingAllocator::findTensorPtr(src_ptr);
  auto read_kernel = std::make_unique<at::neuron::ReadDirectKernelExecution>(
      "neuron::copy::neuron_to_cpu", at::neuron::TensorDataRef{std::move(src_tensor_ptr), src_ptr},
      dst_data_ptr, byte_offset, size_bytes, device);
  auto context = std::make_unique<at::neuron::OperationContext>(std::move(read_kernel));
  at::neuron::SubmitOperationContext(stream, std::move(context));
  if (non_blocking) {
    auto* host_alloc = torch_neuronx::GetNeuronHostAllocatorAsHostAllocator();
    bool recorded = host_alloc->record_event(dst_data_ptr, dst.storage().data_ptr().get_context(),
                                             c10::Stream(stream.unwrap()));
    // record_event returns false when dst wasn't allocated by NeuronHostAllocator
    // (e.g. regular CPU tensor), so fall back to synchronize for safety.
    if (!recorded) {
      stream.synchronize();
    }
  } else {
    stream.synchronize();
  }
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

void copy_neuron_to_neuron(const at::Tensor& src, at::Tensor& dst) {
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
}

}  // namespace utils
}  // namespace torch_neuronx
