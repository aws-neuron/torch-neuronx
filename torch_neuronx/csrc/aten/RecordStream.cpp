
#include <ATen/NativeFunctions.h>

#include "torch_neuronx/csrc/aten/NeuronNativeFunctions.h"
#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"

namespace torch_neuronx {

void record_stream_neuron(at::Tensor& tensor, c10::Stream stream) {
  // Validate that this is a neuron tensor
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
              "record_stream: expected neuron tensor, but got: ", tensor.device());
  TORCH_CHECK(stream.device_type() == c10::DeviceType::PrivateUse1,
              "record_stream: expected neuron stream, but got device type: ", stream.device_type());

  const c10::DataPtr& data_ptr = tensor.storage().data_ptr();
  c10_neuron::NeuronCachingAllocator::recordStream(data_ptr, stream);
}

}  // namespace torch_neuronx
