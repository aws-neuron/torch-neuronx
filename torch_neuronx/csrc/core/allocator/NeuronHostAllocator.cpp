#include <ATen/core/CachingHostAllocator.h>
#include <c10/core/CPUAllocator.h>
#include <torch_neuronx/csrc/core/allocator/NeuronHostAllocator.h>

namespace torch_neuronx {

at::Allocator* GetNeuronHostAllocator() { return c10::GetCPUAllocator(); }

static bool register_host_allocator [[maybe_unused]] = []() {
  at::setHostAllocator(at::kPrivateUse1, static_cast<at::HostAllocator*>(GetNeuronHostAllocator()));
  return true;
}();

}  // namespace torch_neuronx
