#include "torch_neuronx/csrc/core/NeuronHooksInterface.h"

#include <ATen/Context.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/impl/GPUTrace.h>
#include <c10/util/CallOnce.h>

#include "torch_neuronx/csrc/Stream.h"
#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"

namespace torch_neuronx {

// Initialize the Neuron runtime if needed
void NeuronHooksInterface::init() const {
  // The Neuron runtime is initialized when the extension is loaded
  // in register_neuron_extension(). We don't need to do anything here.
}

// Check if Neuron device has a primary context
bool NeuronHooksInterface::hasPrimaryContext(c10::DeviceIndex device_index) const {
  // For Neuron, we consider the device active if it's within the valid range
  return device_index >= 0 && device_index < c10_neuron::device_count();
}

// Get default generator for the device
at::Generator NeuronHooksInterface::getDefaultGenerator(c10::DeviceIndex device_index) {
  // For now, we'll use CPU's default generator
  return at::detail::getDefaultCPUGenerator();
}

// Get a new generator for a specific device
at::Generator NeuronHooksInterface::getNewGenerator(c10::DeviceIndex device_index) const {
  // For now, we'll create a CPU generator
  return at::make_generator<at::CPUGeneratorImpl>(device_index);
}

// Resize storage for Neuron tensors
void NeuronHooksInterface::resizePrivateUse1Bytes(const c10::Storage& storage,
                                                  size_t new_bytes) const {
  // Check if storage is resizable
  TORCH_CHECK(storage.resizable(), "Cannot resize a non-resizable storage");

  // Get current size
  size_t old_bytes = storage.nbytes();

  // If sizes are the same, nothing to do
  if (old_bytes == new_bytes) {
    return;
  }

  c10::StorageImpl* storage_impl = storage.unsafeGetStorageImpl();

  // If new_bytes is 0, do not need to allocate
  if (new_bytes == 0) {
    storage_impl->set_data_ptr_noswap(at::DataPtr(nullptr, storage.device()));
    storage_impl->set_nbytes(0);
    return;
  }

  // Allocate new memory using the storage's allocator
  c10::Allocator* allocator = storage.allocator();
  TORCH_CHECK(allocator != nullptr, "Storage allocator is null");

  at::DataPtr new_data = allocator->allocate(new_bytes);

  int device_id = storage.device().index();

  // Get reference to the old data
  const at::DataPtr& old_data = storage_impl->data_ptr();

  // Copy old data to new allocation if there was data
  if (old_data.get() != nullptr && new_bytes > 0 && old_bytes > 0) {
    size_t copy_bytes = std::min(old_bytes, new_bytes);

    // Use NeuronCachingAllocator's copy function for device-to-device copy
    bool copy_success = c10_neuron::NeuronCachingAllocator::copyTensorData(
        new_data.get(), old_data.get(), copy_bytes, device_id);
    TORCH_CHECK(copy_success, "Failed to copy data during storage resize");
  }

  // Use PyTorch's noswap approach to avoid replacing the storage address
  storage_impl->set_data_ptr_noswap(std::move(new_data));

  // Update storage size
  storage_impl->set_nbytes(new_bytes);
}

// Check if Neuron runtime is available
bool NeuronHooksInterface::isAvailable() const { return c10_neuron::device_count() > 0; }

// Get the singleton instance of NeuronHooksInterface
at::PrivateUse1HooksInterface* get_neuron_hooks() {
  static at::PrivateUse1HooksInterface* neuron_hooks = nullptr;
  static c10::once_flag once;
  c10::call_once(once, [] { neuron_hooks = new NeuronHooksInterface(); });
  return neuron_hooks;
}

}  // namespace torch_neuronx
