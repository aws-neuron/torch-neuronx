#include "NeuronCachingAllocator.h"

#include <c10/core/DeviceType.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <torch_neuronx/csrc/c10/neuron/NeuronEvent.h>
#include <torch_neuronx/csrc/c10/neuron/NeuronStream.h>
#include <torch_neuronx/csrc/core/KernelExecution.h>
#include <torch_neuronx/csrc/core/NeuronDevice.h>
#include <torch_neuronx/csrc/core/OperationContext.h>
#include <torch_neuronx/csrc/core/OperationExecutionEngine.h>
#include <torch_neuronx/csrc/core/allocator/NeuronSegmentAllocator.h>
#include <torch_neuronx/csrc/core/allocator/ShelvesCleanup.h>
#include <torch_neuronx/csrc/core/metrics/NeuronMemorySnapshot.h>
#include <torch_neuronx/csrc/core/metrics/NeuronMetrics.h>
#include <torch_neuronx/csrc/core/runtime/NRTHandler.h>
#include <torch_neuronx/csrc/core/runtime/NRTUtils.h>
#include <torch_neuronx/csrc/core/utils/NeuronExceptions.h>
#include <torch_neuronx/csrc/core/utils/NeuronResourceManager.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

extern "C" {
#include <nrt/nrt.h>
}

namespace NeuronMemorySnapshot = c10_neuron::NeuronMemorySnapshot;

namespace c10_neuron {

// =============================================================================
// NrtTensorPool: host (non-segment-pool) allocator
// Used when lazy alloc is disabled (TORCH_NEURONX_ENABLE_LAZY_ALLOC=0).
// =============================================================================

NrtTensorPool::NrtTensorPool(int device_id) : device_id_(device_id) {}

NrtTensorPool::~NrtTensorPool() { Clear(); }

NrtTensorPool::TensorAllocationResult NrtTensorPool::Allocate(
    size_t size, c10::StreamId stream_id,
    const std::shared_ptr<NeuronMemorySnapshot::StackContext>& stack_ctx) {
  return AllocateWithRetry(size, stream_id, stack_ctx);
}

NrtTensorPool::TensorAllocationResult NrtTensorPool::AllocateWithRetry(
    size_t size, c10::StreamId stream_id,
    const std::shared_ptr<NeuronMemorySnapshot::StackContext>& stack_ctx) {
  if (size == 0) {
    return {nullptr, nullptr, nullptr};
  }

  void* recycled_ptr = nullptr;
  if (auto recycled = TryRecycle(size, stream_id, stack_ctx, &recycled_ptr)) {
    auto deallocation_recorded = std::make_shared<std::atomic<bool>>(false);
    return {std::move(recycled), recycled_ptr, std::move(deallocation_recorded)};
  }

  // Helper to finalize allocation with deleter and metrics
  auto finalize_allocation = [this, size, stream_id, stack_ctx](nrt_tensor_t* raw) {
    void* tensor_data_ptr = nrt_tensor_get_va(raw);

    // Shared flag to track if deallocation was already recorded.
    auto deallocation_recorded = std::make_shared<std::atomic<bool>>(false);

    auto deleter = [device = device_id_, size, stream_id, stack_ctx, tensor_data_ptr,
                    deallocation_recorded](nrt_tensor_t* tensor) {
      if (tensor) {
        bool was_recorded = deallocation_recorded->exchange(true, std::memory_order_acq_rel);
        if (!was_recorded) {
          at::neuron::metrics::RecordDeallocation(device, size, size, tensor_data_ptr, stream_id,
                                                  stack_ctx);
        }
        at::neuron::nrt::FreeTensor(&tensor);
        if (!was_recorded) {
          at::neuron::metrics::RecordDeallocationCompleted(device, size, size, tensor_data_ptr,
                                                           stream_id, stack_ctx);
        }
      }
    };

    at::neuron::metrics::RecordNewAllocation(device_id_, size, size, tensor_data_ptr, stream_id,
                                             stack_ctx);
    return NrtTensorPool::TensorAllocationResult{TensorPtr(raw, deleter), tensor_data_ptr,
                                                 deallocation_recorded};
  };

  nrt_tensor_t* raw = nullptr;
  NRT_STATUS status = at::neuron::nrt::AllocateTensor(
      NRT_TENSOR_PLACEMENT_DEVICE, c10_neuron::get_vnc_id(device_id_), size, nullptr, &raw);

  if (status == NRT_SUCCESS && raw) {
    return finalize_allocation(raw);
  }

  // OOM - prune expired entries and retry
  if (status == NRT_RESOURCE) {
    at::neuron::synchronize(device_id_);
    torch_neuronx::distributed::triggerShelvesCleanup();
    PruneExpiredEntries();
    status = at::neuron::nrt::AllocateTensor(
        NRT_TENSOR_PLACEMENT_DEVICE, c10_neuron::get_vnc_id(device_id_), size, nullptr, &raw);
    if (status == NRT_SUCCESS && raw) {
      at::neuron::metrics::RecordOOMRetry(device_id_, size, stream_id);
      return finalize_allocation(raw);
    }
  }

  std::string msg = "NRT allocation failed for size " + std::to_string(size);
  if (status == NRT_RESOURCE) {
    msg += " (OOM)";
  }
  at::neuron::metrics::RecordOOMFailure(device_id_, size, stream_id, stack_ctx);
  throw std::runtime_error(msg);
}

NrtTensorPool::TensorPtr NrtTensorPool::TryRecycle(
    size_t size, c10::StreamId stream_id,
    const std::shared_ptr<NeuronMemorySnapshot::StackContext>& new_stack_ctx, void** out_data_ptr) {
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
    RecycleEntry entry = std::move(bucket.back());
    bucket.pop_back();

    if (TensorPtr ptr = entry.weak_ptr.lock()) {
      void* data_ptr = entry.data_ptr;
      at::neuron::metrics::RecordNewAllocation(device_id_, size, size, data_ptr, stream_id,
                                               new_stack_ctx);

      // Return cached VA to caller
      if (out_data_ptr) {
        *out_data_ptr = data_ptr;
      }
      return ptr;
    }
  }

  return nullptr;
}

void NrtTensorPool::Recycle(TensorPtr tensor, size_t size, c10::StreamId stream_id, void* data_ptr,
                            const std::shared_ptr<NeuronMemorySnapshot::StackContext>& context,
                            std::shared_ptr<std::atomic<bool>> deallocation_recorded) {
  if (!tensor || size == 0) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  stream_buckets_[stream_id][size].push_back(
      RecycleEntry{WeakTensorPtr(tensor), data_ptr, context, std::move(deallocation_recorded)});
}

void NrtTensorPool::DeferForRecordedStreams(
    TensorPtr tensor, size_t size, c10::StreamId allocation_stream,
    std::vector<at::neuron::NeuronEvent> events, void* data_ptr,
    const std::shared_ptr<NeuronMemorySnapshot::StackContext>& context,
    std::shared_ptr<std::atomic<bool>> deallocation_recorded) {
  if (!tensor) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  recorded_stream_frees_.push_back({std::move(tensor), size, allocation_stream, std::move(events),
                                    data_ptr, context, std::move(deallocation_recorded)});
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
      // Create proper RecycleEntry with all metadata
      stream_buckets_[it->allocation_stream][it->size].push_back(
          RecycleEntry{WeakTensorPtr(it->tensor), it->data_ptr, it->context,
                       std::move(it->deallocation_recorded)});
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
      auto new_end = std::remove_if(bucket.begin(), bucket.end(), [](const RecycleEntry& entry) {
        return entry.weak_ptr.expired();
      });
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
      // Count only non-expired entries
      for (const auto& entry : bucket) {
        if (!entry.weak_ptr.expired()) {
          ++total;
        }
      }
    }
  }
  return total;
}

NrtTensorPool& GetTensorPool(int device) {
  return at::neuron::NeuronResourceManager::Instance().GetTensorPool(device);
}

NeuronSegmentPool& GetSegmentPool(int device) {
  return at::neuron::NeuronResourceManager::Instance().GetSegmentPool(device);
}

namespace NeuronCachingAllocator {

namespace {

std::atomic<bool> g_shutting_down{false};

using Block = c10_neuron::Block;

void* ReserveVirtualAddress(size_t size) {
  void* ptr = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  TORCH_CHECK(ptr != MAP_FAILED, "Internal error on allocation (mmap) of size ", size, ": ",
              strerror(errno));
  return ptr;
}

// Global registry — DataPtr deleter cannot capture allocator instance
struct AllocatorRegistry {
  struct AllocationInfo {
    void* ptr;             // Virtual address
    Block* block;          // Owning block in the segment pool (nullptr for lazy until bound)
    TensorPtr tensor_ptr;  // Shared ownership of the NRT slice handle
    size_t size;           // Requested size
    int device;            // Device index
    at::neuron::NeuronStream allocation_stream;
    std::unordered_set<c10::StreamId> recorded_streams;
    std::mutex recorded_streams_mutex;
    std::shared_ptr<NeuronMemorySnapshot::StackContext> context;  // Stack trace for snapshot
    std::shared_ptr<std::atomic<bool>> deallocation_recorded;     // Shared flag for recycling stats
    std::optional<at::neuron::NeuronEvent> alloc_event;           // Event recorded after AllocBind

    AllocationInfo(void* p, Block* b, size_t s, int dev,
                   const at::neuron::NeuronStream& alloc_stream,
                   std::shared_ptr<NeuronMemorySnapshot::StackContext> ctx = nullptr)
        : ptr(p),
          block(b),
          tensor_ptr(nullptr),
          size(s),
          device(dev),
          allocation_stream(alloc_stream),
          context(std::move(ctx)),
          deallocation_recorded(nullptr) {}
  };
  std::unordered_map<void*, std::unique_ptr<AllocationInfo>> allocations;
  std::mutex mutex;

  static AllocatorRegistry& getInstance() {
    static AllocatorRegistry instance;
    return instance;
  }

  AllocationInfo* add(void* ptr, Block* block, size_t size, int device,
                      const at::neuron::NeuronStream& stream,
                      std::shared_ptr<NeuronMemorySnapshot::StackContext> ctx = nullptr) {
    std::lock_guard<std::mutex> lock(mutex);
    auto info = std::make_unique<AllocationInfo>(ptr, block, size, device, stream, std::move(ctx));
    allocations[ptr] = std::move(info);
    return allocations[ptr].get();
  }

  void remove(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex);
    allocations.erase(ptr);
  }

  // Rekey an AllocationInfo under a new VA override, used by the lazy alloc
  // materialization to replace the virtual address placeholder with the real
  // NRT VA.
  void rekey(void* old_va, void* new_va) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = allocations.find(old_va);
    if (it == allocations.end()) return;
    auto info = std::move(it->second);
    info->ptr = new_va;
    allocations.erase(it);
    allocations[new_va] = std::move(info);
  }

  AllocationInfo* find(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = allocations.find(ptr);
    return (it != allocations.end()) ? it->second.get() : nullptr;
  }

  // Get snapshot context for a given pointer. Must be called with lock held.
  std::shared_ptr<NeuronMemorySnapshot::StackContext> getContext(void* ptr) {
    auto it = allocations.find(ptr);
    return (it != allocations.end()) ? it->second->context : nullptr;
  }
};

using AllocationInfo = AllocatorRegistry::AllocationInfo;

// Compile-time materializer hook for PyTorch 2.12+ (StorageImpl::set_materializer).
#if __has_include(<c10/core/StorageMaterializer.h>)
#define NEURON_HAS_MATERIALIZER 1
#include <c10/core/StorageMaterializer.h>

void neuron_materialize_lazy_alloc(c10::StorageImpl* storage) {
  void* old_va = storage->data_ptr().get();
  auto* info = AllocatorRegistry::getInstance().find(old_va);
  if (!info || !info->alloc_event) return;

  auto current_stream = at::neuron::getCurrentNeuronStream(info->device);
  info->alloc_event->block(current_stream);

  nrt_tensor_t* tensor = info->tensor_ptr ? info->tensor_ptr.get() : nullptr;
  EXECUTION_RUNTIME_CHECK(tensor, "materialize_lazy_alloc: tensor not bound");

  void* new_va = at::neuron::nrt::GetTensorVA(tensor);
  if (new_va == old_va) return;  // nothing to swap.

  AllocatorRegistry::getInstance().rekey(old_va, new_va);

  c10::DataPtr new_data_ptr(new_va, info, &neuron_deleter,
                            c10::Device(c10::DeviceType::PrivateUse1, info->device));
  auto old_data_ptr = storage->set_data_ptr_no_materialize(std::move(new_data_ptr));
  old_data_ptr.release_context();

  munmap(old_va, info->size);
  info->alloc_event.reset();
}
#else
#define NEURON_HAS_MATERIALIZER 0
#endif

void neuron_deleter(void* ctx) {
  AllocationInfo* info = static_cast<AllocationInfo*>(ctx);
  TORCH_CHECK(info != nullptr, "NeuronAllocator: Deleter called on nullptr context");

  void* ptr = info->ptr;
  int device = info->device;
  size_t size = info->size;

  if (IsShuttingDown()) {
    try {
      AllocatorRegistry::getInstance().remove(ptr);
    } catch (...) {
      // Process is exiting; registry entry leaks on exception, which
      // is strictly better than terminating.
    }
    return;
  }

  if (IsLazyAllocEnabled()) {
    // Always include the allocation stream in recorded_streams for handling cross
    // stream dependencies.
    std::vector<c10::StreamId> dealloc_streams;
    {
      std::lock_guard<std::mutex> lock(info->recorded_streams_mutex);
      info->recorded_streams.insert(info->allocation_stream.id());
      dealloc_streams.assign(info->recorded_streams.begin(), info->recorded_streams.end());
    }

    // Submit one kDealloc per stream — last to execute performs cleanup.
    for (auto sid : dealloc_streams) {
      at::neuron::NeuronStream stream(sid, static_cast<c10::DeviceIndex>(device),
                                      at::neuron::NeuronStream::UNCHECKED);
      auto kernel = std::make_unique<at::neuron::DeallocDirectKernelExecution>(
          "neuron::dealloc::lazy", at::neuron::TensorDataRef{nullptr, ptr}, size, device);
      auto ctx = std::make_unique<at::neuron::OperationContext>(std::move(kernel));
      at::neuron::SubmitOperationContext(stream, std::move(ctx));
    }

    return;
  }

  // Host-level allocations
  auto stream_id = info->allocation_stream.id();
  auto stack_ctx = info->context;
  auto deallocation_recorded = info->deallocation_recorded;

  if (deallocation_recorded && size > 0 && device >= 0) {
    bool was_recorded = deallocation_recorded->exchange(true, std::memory_order_acq_rel);
    if (!was_recorded) {
      at::neuron::metrics::RecordDeallocation(device, size, size, ptr, stream_id, stack_ctx);
      at::neuron::metrics::RecordDeallocationCompleted(device, size, size, ptr, stream_id,
                                                       stack_ctx);
    }
  }

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
      pool.Recycle(info->tensor_ptr, size, stream_id, ptr, stack_ctx,
                   std::move(deallocation_recorded));
    } else {
      pool.DeferForRecordedStreams(info->tensor_ptr, size, stream_id, std::move(events), ptr,
                                   stack_ctx, std::move(deallocation_recorded));
    }
  }

  // Submit deallocation hint via RecordHint (concat-only, bypasses compilation pipeline).
  auto& engine = at::neuron::NeuronResourceManager::Instance().GetOperationExecutionEngine();
  if (engine.IsConcatenationEnabled()) {
    auto dealloc_stream = info->allocation_stream;
    auto hint_kernel = std::make_unique<at::neuron::HintDirectKernelExecution>(
        "neuron::hint::dealloc", ptr, size, device,
        at::neuron::HintDirectKernelExecution::HintType::kDeallocation);
    auto context = std::make_unique<at::neuron::OperationContext>(std::move(hint_kernel));
    engine.RecordHint(std::move(context), dealloc_stream);
  }

  AllocatorRegistry::getInstance().remove(ptr);
}

class NeuronAllocator : public c10::Allocator {
 public:
  NeuronAllocator() = default;

  c10::DataPtr allocate(size_t size) override {
    torch_neuronx::maybe_lazy_init();
    auto device = c10_neuron::current_device();
    auto stream = at::neuron::getCurrentNeuronStream(device);
    // Get memory stats for this device
    auto& stats = at::neuron::metrics::DeviceMemoryStatsRegistry::Instance().GetDeviceStats(device);
    // Track every allocation request (even zero-size)
    stats.allocation_requests.fetch_add(1, std::memory_order_relaxed);

    if (size == 0) {
      return c10::DataPtr(nullptr, c10::Device(c10::DeviceType::PrivateUse1, device));
    }

    // Capture stack context ONCE at the beginning (must be done on Python thread)
    std::shared_ptr<NeuronMemorySnapshot::StackContext> stack_ctx = nullptr;
    if (NeuronMemorySnapshot::IsRecordingEnabled()) {
      auto record_ctx = NeuronMemorySnapshot::GetRecordContext();
      if (record_ctx == NeuronMemorySnapshot::RecordContext::ALL ||
          record_ctx == NeuronMemorySnapshot::RecordContext::ALLOC ||
          record_ctx == NeuronMemorySnapshot::RecordContext::STATE) {
        stack_ctx = NeuronMemorySnapshot::StackContext::capture(
            NeuronMemorySnapshot::ShouldIncludeCppFrames());
      }
    }

    if (IsLazyAllocEnabled()) {
      // Lazy path: reserve virtual address, defer physical allocation to worker thread
      void* ptr = ReserveVirtualAddress(size);
      auto* alloc_info =
          AllocatorRegistry::getInstance().add(ptr, nullptr, size, device, stream, stack_ctx);
      auto alloc_kernel = std::make_unique<at::neuron::AllocDirectKernelExecution>(
          "neuron::alloc::lazy", at::neuron::TensorDataRef{nullptr, ptr}, size, device);
      alloc_info->alloc_event = alloc_kernel->GetEvent();
      auto context = std::make_unique<at::neuron::OperationContext>(std::move(alloc_kernel));
      at::neuron::SubmitOperationContext(stream, std::move(context));

      return c10::DataPtr(ptr, alloc_info, &neuron_deleter,
                          c10::Device(c10::DeviceType::PrivateUse1, device));
    }

    // Non-lazy path: allocate from the NrtTensorPool
    auto& allocator = GetTensorPool(device);
    auto [tensor_ptr, ptr, deallocation_recorded] =
        allocator.Allocate(size, stream.id(), stack_ctx);

    TORCH_CHECK(ptr != nullptr, "Failed to get virtual address from NRT tensor");

    auto* alloc_info =
        AllocatorRegistry::getInstance().add(ptr, nullptr, size, device, stream, stack_ctx);
    alloc_info->tensor_ptr = std::move(tensor_ptr);
    alloc_info->deallocation_recorded = std::move(deallocation_recorded);

    // Submit allocation hint via RecordHint (concat-only, bypasses compilation pipeline).
    auto& engine = at::neuron::NeuronResourceManager::Instance().GetOperationExecutionEngine();
    if (engine.IsConcatenationEnabled()) {
      auto hint_kernel = std::make_unique<at::neuron::HintDirectKernelExecution>(
          "neuron::hint::alloc", ptr, size, device,
          at::neuron::HintDirectKernelExecution::HintType::kAllocation);
      auto context = std::make_unique<at::neuron::OperationContext>(std::move(hint_kernel));
      engine.RecordHint(std::move(context), stream);
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

void BeginShutdown() { g_shutting_down.store(true, std::memory_order_relaxed); }

bool IsShuttingDown() { return g_shutting_down.load(std::memory_order_relaxed); }

c10::Allocator* get() { return getAllocator(); }

void emptyCache() {
  int device_count = c10_neuron::device_count();
  if (IsLazyAllocEnabled()) {
    for (int i = 0; i < device_count; ++i) {
      at::neuron::synchronize(i);
    }
    for (int i = 0; i < device_count; ++i) {
      GetSegmentPool(i).EmptyCache();
    }
  } else {
    for (int i = 0; i < device_count; ++i) {
      GetTensorPool(i).Clear();
    }
  }
}

void emptyNrtBufferCache() {
  int device_count = c10_neuron::device_count();
  for (int i = 0; i < device_count; ++i) {
    GetSegmentPool(i).EmptyCache();
  }
}

size_t getCachedBlocks() {
  size_t total = 0;
  int device_count = c10_neuron::device_count();
  if (IsLazyAllocEnabled()) {
    for (int i = 0; i < device_count; ++i) {
      total += GetSegmentPool(i).GetFreeBlockCount();
    }
  } else {
    for (int i = 0; i < device_count; ++i) {
      total += GetTensorPool(i).GetFreelistSize();
    }
  }
  return total;
}

bool copyTensorData(void* dest, void* src, size_t size, int device_id) {
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

nrt_tensor_t* findTensor(void* ptr) {
  auto* info = AllocatorRegistry::getInstance().find(ptr);
  if (!info) return nullptr;
  // Segment-pool path: use block's NRT slice directly
  if (info->block && info->block->nrt_slice) return info->block->nrt_slice;
  // Non-lazy (NrtTensorPool) path: use shared_ptr
  return info->tensor_ptr ? info->tensor_ptr.get() : nullptr;
}

TensorPtr findTensorPtr(void* ptr) {
  auto* info = AllocatorRegistry::getInstance().find(ptr);
  if (!info) return nullptr;
  // Lazy (NeuronSegmentPool) path: wrap the raw slice in a no-op-deleter shared_ptr.
  // Lifetime is managed by the block.
  if (info->block && info->block->nrt_slice) {
    return TensorPtr(info->block->nrt_slice, [](nrt_tensor_t*) {});
  }
  // Non-lazy (NrtTensorPool) path: return the owning shared_ptr
  return info->tensor_ptr;
}

TensorPtr createSliceForView(void* base_ptr, size_t view_offset_bytes, size_t view_size_bytes,
                             const char* name) {
  auto* info = AllocatorRegistry::getInstance().find(base_ptr);

  // Determine the base NRT tensor handle
  nrt_tensor_t* base_tensor = nullptr;
  if (info && info->block && info->block->nrt_slice) {
    base_tensor = info->block->nrt_slice;
  } else if (info && info->tensor_ptr) {
    base_tensor = info->tensor_ptr.get();
  }
  TORCH_CHECK(base_tensor, "createSliceForView: no NRT tensor for base_ptr ", base_ptr);

  size_t block_size = info->block ? info->block->size : info->size;
  TORCH_CHECK(view_offset_bytes <= block_size && block_size - view_offset_bytes >= view_size_bytes,
              "createSliceForView: view [", view_offset_bytes, ", ",
              view_offset_bytes + view_size_bytes, ") exceeds block size ", block_size);

  if (view_offset_bytes == 0 && view_size_bytes == block_size) {
    // Return a no-op-deleter wrapper - lifetime managed by the block.
    return TensorPtr(base_tensor, [](nrt_tensor_t*) {});
  }

  // Create a sub-slice of the block's slice. The sub-slice is a short-lived
  // NRT handle; its deleter frees only the sub-slice, not the parent.
  nrt_tensor_t* slice = at::neuron::NRTHandler::AllocateSliceTensor(base_tensor, view_offset_bytes,
                                                                    view_size_bytes, name);
  return TensorPtr(slice, [](nrt_tensor_t* s) { at::neuron::nrt::FreeTensor(&s); });
}

nrt_tensor_t* getTensorFromContext(void* ctx) {
  TORCH_CHECK(ctx != nullptr, "NeuronAllocator: Context is nullptr");
  auto* info = static_cast<AllocationInfo*>(ctx);
  // Lazy (NeuronSegmentPool) path
  if (info->block && info->block->nrt_slice) return info->block->nrt_slice;
  // Non-lazy (NrtTensorPool) path
  TORCH_CHECK(info->tensor_ptr != nullptr, "NeuronAllocator: tensor_ptr is nullptr in context");
  return info->tensor_ptr.get();
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

// Get snapshot of current allocator state
// Returns SegmentInfo from NeuronMemorySnapshot
std::vector<NeuronMemorySnapshot::SegmentInfo> getSnapshotSegments() {
  if (IsLazyAllocEnabled()) {
    // Segment pool provides detailed block-level snapshots
    auto& registry = AllocatorRegistry::getInstance();
    auto context_resolver =
        [&registry](void* ptr) -> std::shared_ptr<NeuronMemorySnapshot::StackContext> {
      std::lock_guard<std::mutex> lock(registry.mutex);
      return registry.getContext(ptr);
    };

    std::vector<NeuronMemorySnapshot::SegmentInfo> result;
    int device_count = c10_neuron::device_count();
    for (int i = 0; i < device_count; ++i) {
      auto segs = GetSegmentPool(i).GetSnapshotSegments(context_resolver);
      result.insert(result.end(), std::make_move_iterator(segs.begin()),
                    std::make_move_iterator(segs.end()));
    }
    return result;
  }

  // Non-lazy: each active allocation is one segment with one block
  auto& registry = AllocatorRegistry::getInstance();
  std::lock_guard<std::mutex> lock(registry.mutex);
  std::vector<NeuronMemorySnapshot::SegmentInfo> result;
  result.reserve(registry.allocations.size());

  for (const auto& [ptr, info] : registry.allocations) {
    NeuronMemorySnapshot::SegmentInfo seg;
    seg.device = info->device;
    seg.address = reinterpret_cast<size_t>(info->ptr);
    seg.total_size = info->size;
    seg.allocated_size = info->size;
    seg.active_size = info->size;
    seg.stream = info->allocation_stream.id();
    seg.is_small = false;

    NeuronMemorySnapshot::BlockInfo block;
    block.size = info->size;
    block.requested_size = info->size;
    block.address = reinterpret_cast<size_t>(info->ptr);
    block.allocated = true;
    block.context = info->context;
    seg.blocks.push_back(std::move(block));

    result.push_back(std::move(seg));
  }
  return result;
}

namespace lazy {

void AllocBind(void* ptr, size_t size, int device_id) {
  auto* info = AllocatorRegistry::getInstance().find(ptr);
  EXECUTION_RUNTIME_CHECK(info, "lazy::AllocBind: AllocationInfo not found for ptr");

  // Idempotent: skip if already bound
  if (info->block) return;

  auto& pool = GetSegmentPool(device_id);
  Block* block = pool.Allocate(size);
  EXECUTION_RUNTIME_CHECK(
      block, "lazy::AllocBind: segment pool block not found, size=" + std::to_string(size));

  {
    std::lock_guard<std::mutex> lock(info->recorded_streams_mutex);
    info->block = block;
    info->deallocation_recorded = std::make_shared<std::atomic<bool>>(false);
  }

  // Record the ALLOC trace entry with on the real NRT VA (block->data_ptr).
  at::neuron::metrics::RecordNewAllocation(device_id, size, block->size, block->data_ptr,
                                           info->allocation_stream.id(), info->context);
}

void DeallocRelease(void* ptr, size_t size, int device_id, c10::StreamId stream_id) {
  auto* info = AllocatorRegistry::getInstance().find(ptr);
  EXECUTION_RUNTIME_CHECK(info, "lazy::DeallocRelease: AllocationInfo not found for ptr");

  {
    std::lock_guard<std::mutex> lock(info->recorded_streams_mutex);
    info->recorded_streams.erase(stream_id);
    if (!info->recorded_streams.empty()) {
      return;  // Other streams still pending
    }
  }

  // Capture block size and VA before Free. Merge may change block->size, and
  // the VA needs to match what AllocBind recorded for ALLOC/FREE pairing.
  Block* block = info->block;
  size_t block_size = block ? block->size : size;
  void* block_va = block ? block->data_ptr : ptr;

  // Free the block back to the segment pool.
  if (block) {
    auto& pool = GetSegmentPool(device_id);
    pool.Free(block);
  }

  at::neuron::metrics::RecordDeallocation(device_id, size, block_size, block_va,
                                          info->allocation_stream.id(), info->context);
  at::neuron::metrics::RecordDeallocationCompleted(device_id, size, block_size, block_va,
                                                   info->allocation_stream.id(), info->context);

  AllocatorRegistry::getInstance().remove(ptr);
  EXECUTION_RUNTIME_CHECK(munmap(ptr, size) == 0,
                          "Internal error on deallocation (munmap): ", strerror(errno));
}

}  // namespace lazy

}  // namespace NeuronCachingAllocator
}  // namespace c10_neuron
