#include "torch_neuronx/csrc/core/NeuronBarrier.h"

#include <stdexcept>

#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/runtime/NRTUtils.h"

void nrt_barrier_impl(int32_t device_id, int32_t global_device_id, int32_t global_device_count) {
  torch_neuronx::maybe_lazy_init();
  // Use current_device() to ensure synchronization happens on the device this process sees,
  // not global_device_id which is only for NRT barrier coordination across processes
  at::neuron::synchronize(c10_neuron::current_device());
  NRT_STATUS status = at::neuron::nrt::Barrier(device_id, global_device_id, global_device_count);
  if (status != NRT_SUCCESS) {
    throw std::runtime_error("Failed to execute the device barrier " + std::to_string(status));
  }
}
