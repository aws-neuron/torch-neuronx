#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Stream.h>
#include <c10/util/Registry.h>

#include <mutex>
#include <unordered_set>
#include <vector>

// Forward declarations
struct nrt_tensor;
typedef struct nrt_tensor nrt_tensor_t;

namespace at::neuron {
class NeuronStream;
}

namespace c10_neuron {
namespace NeuronCachingAllocator {

// Get the singleton allocator instance
c10::Allocator* get();

// Empty the cache (no-op for now since we don't have caching yet)
void emptyCache();

// Get cached blocks count
size_t getCachedBlocks();

// Copy data between neuron tensors
// Returns true if successful, false if tensors not found
bool copyTensorData(void* dest, const void* src, size_t size);

// Find the nrt_tensor for a given data pointer
// Returns the nrt_tensor_t* if found, nullptr otherwise
nrt_tensor_t* findTensor(void* ptr);

// Record that a tensor is used by a stream
void recordStream(const c10::DataPtr& data_ptr, c10::Stream stream);

}  // namespace NeuronCachingAllocator
}  // namespace c10_neuron
