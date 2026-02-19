#include "NeuronCachingAllocator.h"

#include <c10/core/DeviceType.h>
#include <fcntl.h>
#include <torch_neuronx/csrc/c10/neuron/NeuronEvent.h>
#include <torch_neuronx/csrc/c10/neuron/NeuronStream.h>
#include <torch_neuronx/csrc/core/KernelExecution.h>
#include <torch_neuronx/csrc/core/NeuronDevice.h>
#include <torch_neuronx/csrc/core/OperationContext.h>
#include <torch_neuronx/csrc/core/OperationExecutionEngine.h>
#include <torch_neuronx/csrc/core/allocator/ShelvesCleanup.h>
#include <torch_neuronx/csrc/core/metrics/NeuronMetrics.h>
#include <torch_neuronx/csrc/core/runtime/NRTUtils.h>
#include <torch_neuronx/csrc/core/utils/NeuronResourceManager.h>
#include <torch_neuronx/csrc/core/utils/PlatformUtils.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

extern "C" {
#include <nrt/nrt.h>
}

namespace c10_neuron {

NrtTensorPool::NrtTensorPool(int device_id) : device_id_(device_id) {}

NrtTensorPool::~NrtTensorPool() { Clear(); }

NrtTensorPool::TensorPtr NrtTensorPool::Allocate(size_t size, c10::StreamId stream_id) {
  return AllocateWithRetry(size, stream_id);
}

NrtTensorPool::TensorPtr NrtTensorPool::AllocateWithRetry(size_t size, c10::StreamId stream_id) {
  if (size == 0) {
    return nullptr;
  }

  if (auto recycled = TryRecycle(size, stream_id)) {
    return recycled;
  }

  nrt_tensor_t* raw = nullptr;
  NRT_STATUS status = at::neuron::nrt::AllocateTensor(
      NRT_TENSOR_PLACEMENT_DEVICE, c10_neuron::get_vnc_id(device_id_), size, nullptr, &raw);

  // Lambda deleter that also captures device and size for stats tracking
  auto deleter = [device = device_id_, size](nrt_tensor_t* tensor) {
    if (tensor) {
      at::neuron::nrt::FreeTensor(&tensor);
      at::neuron::metrics::RecordDeallocation(device, size);
    }
  };

  if (status == NRT_SUCCESS && raw) {
    at::neuron::metrics::RecordNewAllocation(device_id_, size);
    return TensorPtr(raw, deleter);
  }

  // OOM - prune expired entries and retry
  auto& stats =
      at::neuron::metrics::DeviceMemoryStatsRegistry::Instance().GetDeviceStats(device_id_);
  if (status == NRT_RESOURCE) {
    at::neuron::synchronize(device_id_);
    torch_neuronx::distributed::triggerShelvesCleanup();
    PruneExpiredEntries();
    status = at::neuron::nrt::AllocateTensor(
        NRT_TENSOR_PLACEMENT_DEVICE, c10_neuron::get_vnc_id(device_id_), size, nullptr, &raw);
    if (status == NRT_SUCCESS && raw) {
      // Track OOM retry
      stats.num_alloc_retries.fetch_add(1, std::memory_order_relaxed);
      at::neuron::metrics::RecordNewAllocation(device_id_, size);
      return TensorPtr(raw, deleter);
    }
  }

  std::string msg = "NRT allocation failed for size " + std::to_string(size);
  if (status == NRT_RESOURCE) {
    msg += " (OOM)";
  }
  // Track final OOM failure
  stats.num_ooms.fetch_add(1, std::memory_order_relaxed);
  throw std::runtime_error(msg);
}

NrtTensorPool::TensorPtr NrtTensorPool::TryRecycle(size_t size, c10::StreamId stream_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  ProcessRecordedStreamFrees();

  auto stream_it = stream_buckets_.find(stream_id);
  if (stream_it == stream_buckets_.end()) {
    return nullptr;
  }
  auto& buckets = stream_it->second;
  auto it = buckets.find(size);
  if (it == buckets.end()) {
    return nullptr;
  }

  Bucket& bucket = it->second;

  // LIFO: pop from back (most recently freed)
  while (!bucket.empty()) {
    WeakTensorPtr weak = std::move(bucket.back());
    bucket.pop_back();

    if (TensorPtr ptr = weak.lock()) {
      return ptr;
    }
  }

  return nullptr;
}

void NrtTensorPool::Recycle(TensorPtr tensor, size_t size, c10::StreamId stream_id) {
  if (!tensor || size == 0) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  stream_buckets_[stream_id][size].push_back(WeakTensorPtr(tensor));
}

void NrtTensorPool::DeferForRecordedStreams(TensorPtr tensor, size_t size,
                                            c10::StreamId allocation_stream,
                                            std::vector<at::neuron::NeuronEvent> events) {
  if (!tensor) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  recorded_stream_frees_.push_back({std::move(tensor), size, allocation_stream, std::move(events)});
}

void NrtTensorPool::ProcessRecordedStreamFrees() {
  auto it = recorded_stream_frees_.begin();
  while (it != recorded_stream_frees_.end()) {
    bool all_complete = true;
    for (const auto& event : it->events) {
      if (!event.query()) {
        all_complete = false;
        break;
      }
    }
    if (all_complete) {
      stream_buckets_[it->allocation_stream][it->size].push_back(WeakTensorPtr(it->tensor));
      it = recorded_stream_frees_.erase(it);
    } else {
      ++it;
    }
  }
}

void NrtTensorPool::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  stream_buckets_.clear();
  recorded_stream_frees_.clear();
}

size_t NrtTensorPool::PruneExpiredEntries() {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t pruned = 0;
  for (auto& [stream_id, buckets] : stream_buckets_) {
    for (auto& [size, bucket] : buckets) {
      auto new_end = std::remove_if(bucket.begin(), bucket.end(),
                                    [](const WeakTensorPtr& w) { return w.expired(); });
      pruned += std::distance(new_end, bucket.end());
      bucket.erase(new_end, bucket.end());
    }
  }
  return pruned;
}

size_t NrtTensorPool::GetFreelistSize() const {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t total = 0;
  for (const auto& [stream_id, buckets] : stream_buckets_) {
    for (const auto& [size, bucket] : buckets) {
      total += bucket.size();
    }
  }
  return total;
}

NrtTensorPool& GetTensorPool(int device) {
  return at::neuron::NeuronResourceManager::Instance().GetTensorPool(device);
}

namespace NeuronCachingAllocator {

namespace {

// Global registry - DataPtr deleter cannot capture allocator instance
struct AllocatorRegistry {
  struct AllocationInfo {
    void* ptr;
    nrt_tensor_t* tensor;
    NeuronCachingAllocator::TensorPtr tensor_ptr;
    size_t size;
    int device;
    at::neuron::NeuronStream allocation_stream;
    std::unordered_set<c10::StreamId> recorded_streams;
    std::mutex recorded_streams_mutex;

    AllocationInfo(void* ptr, nrt_tensor_t* t, size_t s, int dev,
                   const at::neuron::NeuronStream& alloc_stream)
        : ptr(ptr),
          tensor(t),
          tensor_ptr(nullptr),
          size(s),
          device(dev),
          allocation_stream(alloc_stream) {}
  };

  std::unordered_map<void*, std::unique_ptr<AllocationInfo>> allocations;
  std::mutex mutex;

  static AllocatorRegistry& getInstance() {
    static AllocatorRegistry instance;
    return instance;
  }

  AllocationInfo* add(void* ptr, nrt_tensor_t* tensor, size_t size, int device,
                      const at::neuron::NeuronStream& stream) {
    std::lock_guard<std::mutex> lock(mutex);
    auto info = std::make_unique<AllocationInfo>(ptr, tensor, size, device, stream);
    allocations[ptr] = std::move(info);
    return allocations[ptr].get();
  }

  void remove(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex);
    allocations.erase(ptr);
  }

  AllocationInfo* find(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = allocations.find(ptr);
    return (it != allocations.end()) ? it->second.get() : nullptr;
  }
};

using AllocationInfo = AllocatorRegistry::AllocationInfo;

void neuron_deleter(void* ctx) {
  AllocationInfo* info = static_cast<AllocationInfo*>(ctx);
  TORCH_CHECK(info != nullptr, "NeuronAllocator: Deleter called on nullptr context");
  void* ptr = info->ptr;
  int device = info->device;
  size_t size = info->size;
  auto stream_id = info->allocation_stream.id();

  if (info->tensor_ptr && size > 0 && device >= 0) {
    auto& pool = GetTensorPool(device);

    // Check for recorded streams
    std::vector<at::neuron::NeuronEvent> events;
    {
      std::lock_guard<std::mutex> lock(info->recorded_streams_mutex);
      if (!info->recorded_streams.empty()) {
        // Record event on each recorded stream at deallocation time
        for (auto recorded_stream_id : info->recorded_streams) {
          at::neuron::NeuronStream recorded_stream(recorded_stream_id,
                                                   static_cast<c10::DeviceIndex>(device),
                                                   at::neuron::NeuronStream::UNCHECKED);
          at::neuron::NeuronEvent event;
          event.record(recorded_stream);
          events.push_back(std::move(event));
        }
      }
    }

    if (events.empty()) {
      pool.Recycle(info->tensor_ptr, size, stream_id);
    } else {
      pool.DeferForRecordedStreams(info->tensor_ptr, size, stream_id, std::move(events));
    }
  }

  // Submit deallocation hint only when ops-concat is enabled (async mode only)
  auto& engine = at::neuron::NeuronResourceManager::Instance().GetOperationExecutionEngine();
  if (!at::neuron::utils::IsSyncModeEnabled() && engine.IsConcatenationEnabled()) {
    auto dealloc_stream = info->allocation_stream;
    auto hint_kernel = std::make_unique<at::neuron::HintDirectKernelExecution>(
        "neuron::hint::dealloc", ptr, size, device,
        at::neuron::HintDirectKernelExecution::HintType::kDeallocation);
    auto context = std::make_unique<at::neuron::OperationContext>(std::move(hint_kernel));
    at::neuron::SubmitOperationContext(dealloc_stream, std::move(context));
  }
  AllocatorRegistry::getInstance().remove(ptr);
}

class NeuronAllocator : public c10::Allocator {
 public:
  NeuronAllocator() = default;

  c10::DataPtr allocate(size_t size) override {
    auto device = c10_neuron::current_device();
    auto stream = at::neuron::getCurrentNeuronStream(device);
    // Get memory stats for this device
    auto& stats = at::neuron::metrics::DeviceMemoryStatsRegistry::Instance().GetDeviceStats(device);
    // Track every allocation request (even zero-size)
    stats.allocation_requests.fetch_add(1, std::memory_order_relaxed);

    if (size == 0) {
      return c10::DataPtr(nullptr, c10::Device(c10::DeviceType::PrivateUse1, device));
    }

    auto& allocator = GetTensorPool(device);
    auto tensor_ptr = allocator.Allocate(size, stream.id());

    void* ptr = nrt_tensor_get_va(tensor_ptr.get());
    TORCH_CHECK(ptr != nullptr, "Failed to get virtual address from NRT tensor");

    auto* alloc_info =
        AllocatorRegistry::getInstance().add(ptr, tensor_ptr.get(), size, device, stream);
    alloc_info->tensor_ptr = std::move(tensor_ptr);

    // Submit allocation hint only when ops-concat is enabled (async mode only)
    auto& engine = at::neuron::NeuronResourceManager::Instance().GetOperationExecutionEngine();
    if (!at::neuron::utils::IsSyncModeEnabled() && engine.IsConcatenationEnabled()) {
      auto hint_kernel = std::make_unique<at::neuron::HintDirectKernelExecution>(
          "neuron::hint::alloc", ptr, size, device,
          at::neuron::HintDirectKernelExecution::HintType::kAllocation);
      auto context = std::make_unique<at::neuron::OperationContext>(std::move(hint_kernel));
      at::neuron::SubmitOperationContext(stream, std::move(context));
    }

    return c10::DataPtr(ptr, alloc_info, &neuron_deleter,
                        c10::Device(c10::DeviceType::PrivateUse1, device));
  }

  void copy_data(void* dest, const void* src, std::size_t size) const override {
    copyTensorData(dest, const_cast<void*>(src), size, c10_neuron::current_device());
  }
};

std::once_flag allocator_init_flag;
NeuronAllocator* allocator_instance = nullptr;

NeuronAllocator* getAllocator() {
  std::call_once(allocator_init_flag, []() { allocator_instance = new NeuronAllocator(); });
  return allocator_instance;
}

}  // anonymous namespace

c10::Allocator* get() { return getAllocator(); }

void emptyCache() {
  int device_count = c10_neuron::device_count();
  for (int i = 0; i < device_count; ++i) {
    GetTensorPool(i).Clear();
  }
}

size_t getCachedBlocks() {
  size_t total = 0;
  int device_count = c10_neuron::device_count();
  for (int i = 0; i < device_count; ++i) {
    total += GetTensorPool(i).GetFreelistSize();
  }
  return total;
}

bool copyTensorData(void* dest, void* src, size_t size, int device_id) {
  if (!at::neuron::utils::IsSyncModeEnabled()) {
    auto src_tensor_ptr = findTensorPtr(src);
    auto dst_tensor_ptr = findTensorPtr(dest);
    auto copy_kernel = std::make_unique<at::neuron::CopyDirectKernelExecution>(
        "neuron::copy::neuron_to_neuron", at::neuron::TensorDataRef{std::move(src_tensor_ptr), src},
        at::neuron::TensorDataRef{std::move(dst_tensor_ptr), dest}, 0, 0, size, device_id);
    auto context = std::make_unique<at::neuron::OperationContext>(std::move(copy_kernel));
    auto current_stream = at::neuron::getCurrentNeuronStream(device_id);
    at::neuron::SubmitOperationContext(current_stream, std::move(context));
    return true;
  }

  nrt_tensor_t* src_tensor = findTensor(const_cast<void*>(src));
  nrt_tensor_t* dst_tensor = findTensor(dest);

  if (src_tensor && dst_tensor) {
    NRT_STATUS status = nrt_tensor_copy(src_tensor, 0, dst_tensor, 0, size);
    return status == NRT_SUCCESS;
  }
  return false;
}

nrt_tensor_t* findTensor(void* ptr) {
  auto* info = AllocatorRegistry::getInstance().find(ptr);
  return info ? info->tensor : nullptr;
}

TensorPtr findTensorPtr(void* ptr) {
  auto* info = AllocatorRegistry::getInstance().find(ptr);
  return info ? info->tensor_ptr : nullptr;
}

nrt_tensor_t* getTensorFromContext(void* ctx) {
  TORCH_CHECK(ctx != nullptr, "NeuronAllocator: Context is nullptr");
  auto* info = static_cast<AllocationInfo*>(ctx);
  return info->tensor;
}

void recordStream(const c10::DataPtr& data_ptr, c10::Stream stream) {
  void* ptr = data_ptr.get();
  if (!ptr) {
    return;
  }
  auto* info = AllocatorRegistry::getInstance().find(ptr);
  if (!info) {
    return;
  }
  auto stream_id = stream.id();
  // Skip if same as allocation stream
  if (stream_id == info->allocation_stream.id()) {
    return;
  }
  std::lock_guard<std::mutex> lock(info->recorded_streams_mutex);
  info->recorded_streams.insert(stream_id);
}

void remove(void* ptr) { AllocatorRegistry::getInstance().remove(ptr); }

}  // namespace NeuronCachingAllocator
}  // namespace c10_neuron
