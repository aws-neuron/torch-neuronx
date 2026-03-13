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

// NRT tensor pool using weak_ptr LIFO semantics for memory recycling.
class NrtTensorPool {
 public:
  using TensorPtr = std::shared_ptr<nrt_tensor_t>;

  explicit NrtTensorPool(int device_id);
  ~NrtTensorPool();

  NrtTensorPool(const NrtTensorPool&) = delete;
  NrtTensorPool& operator=(const NrtTensorPool&) = delete;

  TensorPtr Allocate(size_t size, c10::StreamId stream_id);
  void Recycle(TensorPtr tensor, size_t size, c10::StreamId stream_id);
  void DeferForRecordedStreams(TensorPtr tensor, size_t size, c10::StreamId allocation_stream,
                               std::vector<at::neuron::NeuronEvent> events);
  void Clear();
  size_t PruneExpiredEntries();
  size_t GetFreelistSize() const;

 private:
  using WeakTensorPtr = std::weak_ptr<nrt_tensor_t>;
  using Bucket = std::vector<WeakTensorPtr>;
  using SizeBuckets = std::unordered_map<size_t, Bucket>;

  struct RecordedStreamFree {
    TensorPtr tensor;
    size_t size;
    c10::StreamId allocation_stream;
    std::vector<at::neuron::NeuronEvent> events;
  };

  TensorPtr AllocateWithRetry(size_t size, c10::StreamId stream_id);
  TensorPtr TryRecycle(size_t size, c10::StreamId stream_id);
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
nrt_tensor_t* findTensor(void* ptr);
TensorPtr findTensorPtr(void* ptr);
nrt_tensor_t* getTensorFromContext(void* ctx);
void recordStream(const c10::DataPtr& data_ptr, c10::Stream stream);
void remove(void* ptr);

}  // namespace NeuronCachingAllocator
}  // namespace c10_neuron
