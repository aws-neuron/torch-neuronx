#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Stream.h>

#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

// Forward declarations
struct nrt_tensor;
typedef struct nrt_tensor nrt_tensor_t;

namespace at::neuron {
class NeuronEvent;
class NeuronStream;
}  // namespace at::neuron

namespace c10_neuron {
namespace NeuronMemorySnapshot {
struct SegmentInfo;
struct StackContext;
}  // namespace NeuronMemorySnapshot

// NRT tensor pool using weak_ptr LIFO semantics for memory recycling.
class NrtTensorPool {
 public:
  using TensorPtr = std::shared_ptr<nrt_tensor_t>;

  // Result of tensor allocation - contains both the tensor pointer and virtual address
  struct TensorAllocationResult {
    TensorPtr tensor_ptr;
    void* data_ptr;  // Virtual address, cached to avoid repeated nrt_tensor_get_va calls
    std::shared_ptr<std::atomic<bool>> deallocation_recorded;  // Shared flag for recycling stats
  };

  explicit NrtTensorPool(int device_id);
  ~NrtTensorPool();

  NrtTensorPool(const NrtTensorPool&) = delete;
  NrtTensorPool& operator=(const NrtTensorPool&) = delete;

  // Allocate a tensor, optionally with stack context for snapshot tracing
  // Returns both the TensorPtr and the virtual address to avoid repeated nrt_tensor_get_va calls
  TensorAllocationResult Allocate(
      size_t size, c10::StreamId stream_id,
      const std::shared_ptr<NeuronMemorySnapshot::StackContext>& stack_ctx = nullptr);

  // Recycle a tensor back to the pool with deallocation metadata for stats tracking
  void Recycle(TensorPtr tensor, size_t size, c10::StreamId stream_id, void* data_ptr,
               const std::shared_ptr<NeuronMemorySnapshot::StackContext>& context,
               std::shared_ptr<std::atomic<bool>> deallocation_recorded);

  void DeferForRecordedStreams(TensorPtr tensor, size_t size, c10::StreamId allocation_stream,
                               std::vector<at::neuron::NeuronEvent> events, void* data_ptr,
                               const std::shared_ptr<NeuronMemorySnapshot::StackContext>& context,
                               std::shared_ptr<std::atomic<bool>> deallocation_recorded);
  void Clear();
  size_t PruneExpiredEntries();
  size_t GetFreelistSize() const;

 private:
  using WeakTensorPtr = std::weak_ptr<nrt_tensor_t>;

  // Entry in the recycle pool - contains weak_ptr and deallocation metadata
  struct RecycleEntry {
    WeakTensorPtr weak_ptr;
    void* data_ptr;  // Virtual address for deallocation recording
    std::shared_ptr<NeuronMemorySnapshot::StackContext> context;  // Stack trace of original alloc
    std::shared_ptr<std::atomic<bool>> deallocation_recorded;     // Flag to prevent double-counting
  };

  using Bucket = std::vector<RecycleEntry>;
  using SizeBuckets = std::unordered_map<size_t, Bucket>;

  struct RecordedStreamFree {
    TensorPtr tensor;
    size_t size;
    c10::StreamId allocation_stream;
    std::vector<at::neuron::NeuronEvent> events;
    void* data_ptr;  // Virtual address for deallocation recording
    std::shared_ptr<NeuronMemorySnapshot::StackContext> context;  // Stack trace
    std::shared_ptr<std::atomic<bool>> deallocation_recorded;     // Flag to prevent double-counting
  };

  TensorAllocationResult AllocateWithRetry(
      size_t size, c10::StreamId stream_id,
      const std::shared_ptr<NeuronMemorySnapshot::StackContext>& stack_ctx);

  TensorPtr TryRecycle(size_t size, c10::StreamId stream_id,
                       const std::shared_ptr<NeuronMemorySnapshot::StackContext>& new_stack_ctx,
                       void** out_data_ptr);
  void ProcessRecordedStreamFrees();

  std::unordered_map<c10::StreamId, SizeBuckets> stream_buckets_;
  std::vector<RecordedStreamFree> recorded_stream_frees_;
  mutable std::mutex mutex_;
  int device_id_;
};

// Get per-device tensor pool instance
NrtTensorPool& GetTensorPool(int device);

namespace NeuronCachingAllocator {

using TensorPtr = std::shared_ptr<nrt_tensor_t>;

c10::Allocator* get();
void emptyCache();
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

}  // namespace NeuronCachingAllocator
}  // namespace c10_neuron
