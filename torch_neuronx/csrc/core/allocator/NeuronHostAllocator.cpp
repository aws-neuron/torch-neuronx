#include "torch_neuronx/csrc/core/allocator/NeuronHostAllocator.h"

#include "torch_neuronx/csrc/c10/neuron/NeuronEvent.h"
#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/runtime/NRTUtils.h"

namespace torch_neuronx {

static bool use_pinned_malloc() {
  static const bool enabled = []() {
    const char* env = std::getenv("TORCH_NEURONX_USE_PINNED_MALLOC");
    return env && std::string(env) == "1";
  }();
  return enabled;
}

void NeuronHostAllocatorImpl::allocate_host_memory(size_t size, void** ptr) {
  if (size == 0) {
    *ptr = nullptr;
    return;
  }
  if (use_pinned_malloc()) {
    *ptr = at::neuron::nrt::PinnedMalloc(size);
    TORCH_CHECK(*ptr != nullptr, "nrt_pinned_malloc failed for size ", size);
  } else {
    int err = posix_memalign(ptr, 64, size);
    TORCH_CHECK(err == 0, "Failed to allocate pinned host memory, error: ", err);
  }
  pinned_ptrs_.insert(*ptr);
}

void NeuronHostAllocatorImpl::free_block(at::HostBlock<at::neuron::NeuronStream>* block) {
  pinned_ptrs_.erase(block->ptr_);
  if (use_pinned_malloc()) {
    at::neuron::nrt::PinnedFree(block->ptr_);
  } else {
    free(block->ptr_);
  }
}

bool NeuronHostAllocatorImpl::is_pinned_ptr(const void* ptr) { return pinned_ptrs_.count(ptr) > 0; }

void NeuronHostAllocatorImpl::record_stream(
    std::optional<std::vector<at::neuron::NeuronEvent>>& events, at::neuron::NeuronStream stream) {
  if (!events.has_value()) {
    events.emplace();
  }
  at::neuron::NeuronEvent event;
  event.record(stream);
  events->push_back(std::move(event));
}

bool NeuronHostAllocatorImpl::query_event(at::neuron::NeuronEvent& event) {
  bool completed = true;
  try {
    completed = event.query();
  } catch (const std::exception& e) {
    TORCH_NEURONX_ERROR("Neuron host allocator failed: ", e.what());
  }
  return completed;
}

}  // namespace torch_neuronx

namespace {
DECLARE_HOST_ALLOCATOR(NeuronCachingHostAllocator, torch_neuronx::NeuronHostAllocatorImpl,
                       neuron_host_deleter, neuron_host_allocator);
}  // namespace

namespace torch_neuronx {

at::Allocator* GetNeuronHostAllocator() { return &neuron_host_allocator; }

at::HostAllocator* GetNeuronHostAllocatorAsHostAllocator() { return &neuron_host_allocator; }

bool IsNeuronPinnedPtr(const void* ptr) { return neuron_host_allocator.impl_->is_pinned_ptr(ptr); }

static bool register_host_allocator [[maybe_unused]] = []() {
  at::setHostAllocator(at::kPrivateUse1, &neuron_host_allocator);
  return true;
}();

}  // namespace torch_neuronx
