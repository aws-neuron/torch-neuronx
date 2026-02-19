#pragma once

#include <ATen/CPUGeneratorImpl.h>
#include <c10/core/Device.h>

namespace torch_neuronx {

class NeuronGeneratorImpl : public at::CPUGeneratorImpl {
 public:
  NeuronGeneratorImpl(c10::DeviceIndex device_index) {
    device_ = c10::Device(c10::DeviceType::PrivateUse1, device_index);
    key_set_ = c10::DispatchKeySet(c10::DispatchKey::PrivateUse1);
  }
  ~NeuronGeneratorImpl() override = default;
};

const at::Generator& getDefaultNeuronGenerator(c10::DeviceIndex device_index = -1);

}  // namespace torch_neuronx
