#pragma once

#include <ATen/core/CachingHostAllocator.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/Device.h>
#include <c10/core/Storage.h>
#include <c10/util/Optional.h>

namespace torch_neuronx {

// NeuronHooksInterface implements the PrivateUse1HooksInterface to enable
// autograd support for Neuron tensors. This interface is required for
// gradient computation and other autograd operations.
class NeuronHooksInterface : public at::PrivateUse1HooksInterface {
 public:
  // Constructor
  NeuronHooksInterface() = default;

  // Destructor
  ~NeuronHooksInterface() override = default;

  // Initialize the Neuron runtime if needed
  void init() const override;

  // Check if a pointer is pinned memory (always true for Neuron) as all the tensors are pinned
  bool isPinnedPtr(const void* data) const override { return true; }

  // Get the device index
  c10::DeviceIndex getDeviceIndex(c10::DeviceIndex device_index) const { return device_index; }

  // Check if Neuron runtime is available
  bool hasPrimaryContext(c10::DeviceIndex device_index) const override;

  // Allocate pinned memory using the Neuron host allocator
  c10::Allocator* getPinnedMemoryAllocator() const override {
    return at::getHostAllocator(at::kPrivateUse1);
  }

  // Get the default generator for random number generation
  at::Generator getDefaultGenerator(c10::DeviceIndex device_index);

  // Get a new generator for a specific device
  at::Generator getNewGenerator(c10::DeviceIndex device_index) const;

  // Resize storage for PrivateUse1 (Neuron) tensors
  void resizePrivateUse1Bytes(const c10::Storage& storage, size_t new_bytes) const override;

  // Check if Neuron runtime is available
  bool isAvailable() const;
};

// Get the singleton instance of NeuronHooksInterface
at::PrivateUse1HooksInterface* get_neuron_hooks();

// Get the default generator for a Neuron device
const at::Generator& getDefaultNeuronGenerator(c10::DeviceIndex device_index = -1);

}  // namespace torch_neuronx
