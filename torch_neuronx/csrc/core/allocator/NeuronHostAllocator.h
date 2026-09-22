#pragma once

#include <ATen/core/CachingHostAllocator.h>
#include <c10/core/Allocator.h>

#include <unordered_set>

namespace at::neuron {
class NeuronStream;
class NeuronEvent;
}  // namespace at::neuron

namespace torch_neuronx {

struct NeuronHostAllocatorImpl final
    : public at::CachingHostAllocatorImpl<at::neuron::NeuronStream, at::neuron::NeuronEvent> {
  void allocate_host_memory(size_t size, void** ptr) override;
  void free_block(at::HostBlock<at::neuron::NeuronStream>* block) override;
  void record_stream(std::optional<std::vector<at::neuron::NeuronEvent>>& events,
                     at::neuron::NeuronStream stream) override;
  bool query_event(at::neuron::NeuronEvent& event) override;

  // Check if a pointer was allocated by this allocator.
  bool is_pinned_ptr(const void* ptr);

 private:
  std::unordered_set<const void*> pinned_ptrs_;
};

at::Allocator* GetNeuronHostAllocator();
at::HostAllocator* GetNeuronHostAllocatorAsHostAllocator();

// Check if a pointer was allocated by the Neuron pinned host allocator.
bool IsNeuronPinnedPtr(const void* ptr);

}  // namespace torch_neuronx
