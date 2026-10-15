#pragma once

#include <ATen/Tensor.h>
#include <c10/core/Allocator.h>
#include <c10/core/Stream.h>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Forward declarations
struct nrt_tensor;
typedef struct nrt_tensor nrt_tensor_t;

namespace at::neuron {
class NeuronEvent;
class NeuronStream;
}  // namespace at::neuron

namespace c10_neuron {

class NeuronSegmentPool;
struct Block;

namespace NeuronMemorySnapshot {
struct SegmentInfo;
struct StackContext;
}  // namespace NeuronMemorySnapshot

// Get per-device segment pool instance.
NeuronSegmentPool& GetSegmentPool(int device);

namespace NeuronCachingAllocator {

using TensorPtr = std::shared_ptr<nrt_tensor_t>;

inline bool IsScalarCacheEnabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("TORCH_NEURONX_ENABLE_SCALAR_CACHE");
    return env == nullptr || std::string(env) == "1";
  }();
  return enabled;
}

// Shutdown barrier.
//
// During normal operation the lazy-alloc path submits a kDealloc op per
// recorded stream so the physical free is ordered after any in-flight
// kernel that touched the memory. At process exit, streams have been
// drained and NRT teardown reclaims memory wholesale, so the ordering is
// meaningless — but the submission itself races the engine's worker
// shutdown.
//
// BeginShutdown() is a one-way fence: after it returns, every subsequent
// free() on the allocator short-circuits instead of submitting an op.
// Called from ~NeuronResourceManager before the engine Shutdown(), so
// producer submissions stop strictly before consumers are joined.
void BeginShutdown();
bool IsShuttingDown();

c10::Allocator* get();
void emptyCache();
void emptyNrtBufferCache();
size_t getCachedBlocks();
bool copyTensorData(void* dest, void* src, size_t size, int device_id);
// WARNING: Must be called from main thread only.
nrt_tensor_t* findTensor(void* ptr);
// WARNING: Must be called from main thread only.
TensorPtr findTensorPtr(void* ptr);

nrt_tensor_t* getTensorFromContext(void* ctx);
void recordStream(const c10::DataPtr& data_ptr, c10::Stream stream);
void remove(void* ptr);

// Get snapshot of current allocator state for memory debugging
std::vector<NeuronMemorySnapshot::SegmentInfo> getSnapshotSegments();

// Create a 0-dim Neuron tensor backed by scalar cache memory.
// Returns a tensor whose storage points to the cached slot — no per-tensor
// allocation or H2D write on cache hit. The tensor is read-only at the device
// level (writes would corrupt the shared cache).
// Returns nullopt if data_size is 0 or exceeds maximum scalar size.
std::optional<at::Tensor> createCachedScalarTensor(const void* host_data, size_t data_size,
                                                   c10::ScalarType dtype, int device_id);

namespace lazy {
// Bind physical memory to a lazy allocation VA.
void AllocBind(void* ptr, size_t size, int device_id);
// Release physical memory for a lazy allocation VA.
void DeallocRelease(void* ptr, size_t size, int device_id, c10::StreamId stream_id);
// Largest single physical binding observed (bytes).
size_t LargestAllocBytes();
}  // namespace lazy

}  // namespace NeuronCachingAllocator
}  // namespace c10_neuron
