#include <ATen/core/CachingHostAllocator.h>
#include <c10/core/CPUAllocator.h>
#include <torch_neuron/csrc/core/allocator/NeuronHostAllocator.h>

namespace torch_neuron {

at::Allocator* GetNeuronHostAllocator() { return c10::GetCPUAllocator(); }

static bool register_host_allocator [[maybe_unused]] = []() {
  at::setHostAllocator(at::kPrivateUse1, static_cast<at::HostAllocator*>(GetNeuronHostAllocator()));
  return true;
}();

}  // namespace torch_neuron
