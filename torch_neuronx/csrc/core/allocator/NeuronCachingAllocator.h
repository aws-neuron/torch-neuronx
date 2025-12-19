#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Stream.h>
#include <c10/util/Registry.h>

#include <cstddef>
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

// Forward declaration for internal allocator registry
struct AllocatorRegistry;

namespace NeuronCachingAllocator {

// Get the singleton allocator instance
c10::Allocator* get();

// Empty the cache (no-op for now since we don't have caching yet)
void emptyCache();

// Get cached blocks count
size_t getCachedBlocks();

// Copy data between neuron tensors
// Returns true if successful, false if tensors not found
bool copyTensorData(void* dest, void* src, size_t size, int device_id);

// Find the nrt_tensor for a given data pointer
// Returns the nrt_tensor_t* if found, nullptr otherwise
nrt_tensor_t* findTensor(void* ptr);

// Record the nrt_tensor for the given data pointer.
// If the tensor is already present, then throw an error.
void recordTensor(void* ptr, nrt_tensor_t* nrt_tensor);

std::vector<void*>& findTensorSlices(void* ptr);

// Add an entry for the slice data pointer with the original tensor. Once this the original
// tensor is deallocated, it will ensure that all other slices are deallocated.
void addSliceEntry(void* ptr, void* slice_ptr, size_t size, int device,
                   const at::neuron::NeuronStream& stream);

// Extract nrt_tensor from DataPtr context
// Returns the nrt_tensor_t* if the context is valid, nullptr otherwise
nrt_tensor_t* getTensorFromContext(void* ctx);

// Record that a tensor is used by a stream
void recordStream(const c10::DataPtr& data_ptr, c10::Stream stream);

// Reserve virtual address space
void* ReserveVirtualAddress();

// Release virtual address space
void ReleaseVirtualAddress(void* vaddr);

// Remove the entry from the registry
void remove(void* ptr);

// Get device-specific memory pool and add block to cache
void cacheBlock(int device, nrt_tensor_t* tensor, size_t size);

// Retrieve a cached block for the given device-specific memory pool and size.
// If empty, then return nullptr.
nrt_tensor_t* getCachedBlock(int device, size_t size);

}  // namespace NeuronCachingAllocator
}  // namespace c10_neuron
