#include "NeuronCachingAllocator.h"

#include <ATen/core/TensorBody.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <c10/core/TensorImpl.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <torch_neuronx/csrc/c10/neuron/NeuronEvent.h>
#include <torch_neuronx/csrc/c10/neuron/NeuronStream.h>
#include <torch_neuronx/csrc/core/KernelExecution.h>
#include <torch_neuronx/csrc/core/NeuronDevice.h>
#include <torch_neuronx/csrc/core/NeuronStorageImpl.h>
#include <torch_neuronx/csrc/core/OperationContext.h>
#include <torch_neuronx/csrc/core/OperationExecutionEngine.h>
#include <torch_neuronx/csrc/core/allocator/NeuronSegmentAllocator.h>
#include <torch_neuronx/csrc/core/metrics/NeuronMemorySnapshot.h>
#include <torch_neuronx/csrc/core/metrics/NeuronMetrics.h>
#include <torch_neuronx/csrc/core/runtime/NRTUtils.h>
#include <torch_neuronx/csrc/core/utils/NeuronExceptions.h>
#include <torch_neuronx/csrc/core/utils/NeuronResourceManager.h>
#include <unistd.h>

#include <atomic>
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
    void* ptr;     // Virtual address
    Block* block;  // Owning block in the segment pool (nullptr until bound)
    size_t size;   // Requested size
    int device;    // Device index
    at::neuron::NeuronStream allocation_stream;
    std::unordered_set<c10::StreamId> recorded_streams;
    std::mutex recorded_streams_mutex;
    std::shared_ptr<NeuronMemorySnapshot::StackContext> context;  // Stack trace for snapshot
    std::shared_ptr<at::neuron::EventImpl> alloc_event;           // Event recorded after AllocBind
    bool is_scalar = false;                                       // Scalar allocation flag
    // Scalar path only: slot in the ScalarDeviceMemoryCache (nullptr until
    // bound). The cache owns the lifetime, so this is a borrowed handle.
    nrt_tensor_t* scalar_slice = nullptr;

    AllocationInfo(void* p, Block* b, size_t s, int dev,
                   const at::neuron::NeuronStream& alloc_stream,
                   std::shared_ptr<NeuronMemorySnapshot::StackContext> ctx = nullptr)
        : ptr(p),
          block(b),
          size(s),
          device(dev),
          allocation_stream(alloc_stream),
          context(std::move(ctx)) {}
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

// Compile-time materializer hook for PyTorch 2.13+ (StorageImpl::set_materializer).
#if __has_include(<c10/core/StorageMaterializer.h>)
#define NEURON_HAS_MATERIALIZER 1
#include <c10/core/StorageMaterializer.h>

void neuron_deleter(void* ctx);

[[maybe_unused]] void neuron_materialize_lazy_alloc(c10::StorageImpl* storage) {
  void* old_va = storage->data_ptr().get();
  auto* info = AllocatorRegistry::getInstance().find(old_va);
  if (!info || !info->alloc_event) return;

  auto current_stream = at::neuron::getCurrentNeuronStream(info->device);
  if (!info->alloc_event->query()) current_stream.SubmitEventWait(info->alloc_event);

  nrt_tensor_t* tensor = info->block ? info->block->nrt_slice : nullptr;
  EXECUTION_RUNTIME_CHECK(tensor, "materialize_lazy_alloc: tensor not bound");

  void* new_va = at::neuron::nrt::GetTensorVA(tensor);
  if (new_va == old_va) return;  // nothing to swap.

  AllocatorRegistry::getInstance().rekey(old_va, new_va);

  c10::DataPtr new_data_ptr(new_va, info, &neuron_deleter,
                            c10::Device(c10::DeviceType::PrivateUse1, info->device));
  auto* neuron_storage = static_cast<c10_neuron::NeuronStorageImpl*>(storage);
  auto old_data_ptr =
      neuron_storage->replace_data_ptr_without_materializing(std::move(new_data_ptr));
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

  // Always include the allocation stream in recorded_streams for handling cross
  // stream dependencies.
  std::vector<c10::StreamId> dealloc_streams;
  {
    std::lock_guard<std::mutex> lock(info->recorded_streams_mutex);
    info->recorded_streams.insert(info->allocation_stream.id());
    // Record the current (free-time) stream. A cross-stream consumer
    // (for example an FSDP2 reduce-scatter output) can free this buffer from
    // a stream other than the allocation stream.
    // TODO - revert once https://github.com/pytorch/pytorch/pull/193686 lands upstream
    info->recorded_streams.insert(
        at::neuron::getCurrentNeuronStream(static_cast<c10::DeviceIndex>(device)).id());
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

    // Reserve a virtual address; physical binding is deferred to the worker thread.
    void* ptr = ReserveVirtualAddress(size);
    auto* alloc_info =
        AllocatorRegistry::getInstance().add(ptr, nullptr, size, device, stream, stack_ctx);
    auto alloc_kernel = std::make_unique<at::neuron::AllocDirectKernelExecution>(
        "neuron::alloc::lazy", at::neuron::TensorDataRef{nullptr, ptr}, size, device);
    alloc_info->alloc_event = alloc_kernel->GetEventImpl();
    auto context = std::make_unique<at::neuron::OperationContext>(std::move(alloc_kernel));
    at::neuron::SubmitOperationContext(stream, std::move(context));

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
  for (int i = 0; i < device_count; ++i) {
    at::neuron::synchronize(i);
  }
  for (int i = 0; i < device_count; ++i) {
    GetScalarCache(i).Clear();
    GetSegmentPool(i).EmptyCache();
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
  for (int i = 0; i < device_count; ++i) {
    total += GetSegmentPool(i).GetFreeBlockCount();
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
  // Scalar-cache path: use the cache slot.
  if (info->is_scalar) return info->scalar_slice;
  // Segment-pool path: use block's NRT slice directly
  return info->block ? info->block->nrt_slice : nullptr;
}

TensorPtr findTensorPtr(void* ptr) {
  nrt_tensor_t* slice = findTensor(ptr);
  if (!slice) return nullptr;
  // Wrap the raw slice in a no-op-deleter shared_ptr; the block or the scalar
  // cache owns the lifetime.
  return TensorPtr(slice, [](nrt_tensor_t*) {});
}

nrt_tensor_t* getTensorFromContext(void* ctx) {
  TORCH_CHECK(ctx != nullptr, "NeuronAllocator: Context is nullptr");
  auto* info = static_cast<AllocationInfo*>(ctx);
  if (info->is_scalar) {
    EXECUTION_RUNTIME_CHECK(info->scalar_slice, "NeuronAllocator: scalar not bound in context");
    return info->scalar_slice;
  }
  EXECUTION_RUNTIME_CHECK(info->block && info->block->nrt_slice,
                          "NeuronAllocator: tensor not bound in context");
  return info->block->nrt_slice;
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

std::optional<at::Tensor> createCachedScalarTensor(const void* host_data, size_t data_size,
                                                   c10::ScalarType dtype, int device_id) {
  constexpr size_t kSlotSize = c10_neuron::ScalarDeviceMemoryCache::kSlotSize;

  // No standard scalar dtype exceeds 16 bytes (largest, complex128 = 16 bytes). If
  // this fires, the caller passed a non-scalar; fall back to normal allocation.
  if (data_size > kSlotSize) {
    TORCH_WARN("Scalar size ", data_size, " bytes exceeds scalar cache slot size ", kSlotSize,
               "; falling back to standard allocation.");
    return std::nullopt;
  }

  // Fall back to the normal allocation + copy path when the cache is disabled.
  if (!IsScalarCacheEnabled()) {
    return std::nullopt;
  }

  // Separate allocation path, responsible for lazy init.
  torch_neuronx::maybe_lazy_init();

  // Resolve an unspecified device index (-1, e.g. scalar_tensor(device="neuron")
  // with no index) to the current device before indexing per-device state.
  if (device_id < 0) {
    device_id = c10_neuron::current_device();
  }

  // Build padded key from scalar value
  c10_neuron::ScalarKey key{};
  std::memcpy(key.bits.data(), host_data, data_size);

  // Get VA for scalar
  auto& cache = GetScalarCache(device_id);
  auto [ptr, is_new] = cache.GetOrReserveVA(key, data_size);

  auto& registry = AllocatorRegistry::getInstance();
  if (is_new) {
    auto stream = at::neuron::getCurrentNeuronStream(device_id);
    auto* new_info = registry.add(ptr, nullptr, kSlotSize, device_id, stream);
    new_info->is_scalar = true;
    // No alloc_event is set, so the storage materializer is a no-op; the tensor
    // keeps this placeholder VA and resolves the bound slice via findTensor.
    auto alloc_kernel = std::make_unique<at::neuron::AllocDirectKernelExecution>(
        "neuron::alloc::scalar", at::neuron::TensorDataRef{nullptr, ptr}, kSlotSize, device_id);
    at::neuron::SubmitOperationContext(
        stream, std::make_unique<at::neuron::OperationContext>(std::move(alloc_kernel)));

    // Write the value as a separate kWrite, ordered after the kAlloc.
    // Placeholder VA is resolved to the bound slice via findTensor at schedule time.
    // AllocateBounceBuffer copies the bytes into the op, so the key need not
    // outlive this call. Write the whole slot from the key, not data_size bytes from host_data.
    // The key carries no width, so a wider dtype can hit this entry.
    auto write_kernel = std::make_unique<at::neuron::WriteDirectKernelExecution>(
        "neuron::write::scalar", key.bits.data(), at::neuron::TensorDataRef{nullptr, ptr},
        /*dst_offset_bytes=*/0, kSlotSize, device_id);
    write_kernel->AllocateBounceBuffer(kSlotSize);
    at::neuron::SubmitOperationContext(
        stream, std::make_unique<at::neuron::OperationContext>(std::move(write_kernel)));
  }

  // Build a 0-dim tensor aliasing the shared scalar VA. The deleter frees
  // nothing: the VA and the slot are shared across every tensor with this value
  // and the cache owns them. It only drops the live count, which holds off
  // Clear() until the last such tensor dies.
  AllocationInfo* info = registry.find(ptr);
  EXECUTION_RUNTIME_CHECK(info, "createCachedScalarTensor: registry entry lost for scalar VA");
  cache.RetainTensor();
  c10::DataPtr data_ptr(
      ptr, /*ctx=*/info,
      /*deleter=*/
      [](void* ctx) {
        if (IsShuttingDown()) return;
        try {
          GetScalarCache(static_cast<AllocationInfo*>(ctx)->device).ReleaseTensor();
        } catch (...) {
          // A deleter must not throw. A lost decrement only keeps the cache.
        }
      },
      c10::Device(c10::DeviceType::PrivateUse1, device_id));
  auto storage = c10::Storage(c10::Storage::use_byte_size_t(), data_size, std::move(data_ptr),
                              /*allocator=*/nullptr, /*resizable=*/false);
  auto tensor = at::detail::make_tensor<at::TensorImpl>(
      std::move(storage), c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
      c10::scalarTypeToTypeMeta(dtype));
  // 0-dim scalar tensor: empty sizes and strides.
  tensor.unsafeGetTensorImpl()->set_sizes_and_strides(c10::IntArrayRef{}, c10::IntArrayRef{});
  return tensor;
}

namespace lazy {

// Largest single physical binding observed; sizes run-ahead headroom.
std::atomic<size_t> g_largest_alloc_bytes{0};

size_t LargestAllocBytes() { return g_largest_alloc_bytes.load(std::memory_order_relaxed); }

void AllocBind(void* ptr, size_t size, int device_id) {
  auto* info = AllocatorRegistry::getInstance().find(ptr);
  EXECUTION_RUNTIME_CHECK(info, "lazy::AllocBind: AllocationInfo not found for ptr");

  // Scalar allocations bind to a ScalarDeviceMemoryCache slot instead of a
  // segment-pool block.
  if (info->is_scalar) {
    if (info->scalar_slice) return;
    // Allocate slot. The value is written by the kWrite (see createCachedScalarTensor).
    nrt_tensor_t* slice = GetScalarCache(device_id).BindSlot();
    std::lock_guard<std::mutex> lock(info->recorded_streams_mutex);
    info->scalar_slice = slice;
    return;
  }

  // Idempotent: skip if already bound
  if (info->block) return;

  auto& pool = GetSegmentPool(device_id);
  Block* block = pool.Allocate(size);
  EXECUTION_RUNTIME_CHECK(
      block, "lazy::AllocBind: segment pool block not found, size=" + std::to_string(size));

  for (size_t prev = g_largest_alloc_bytes.load(std::memory_order_relaxed);
       size > prev &&
       !g_largest_alloc_bytes.compare_exchange_weak(prev, size, std::memory_order_relaxed);) {
  }

  {
    std::lock_guard<std::mutex> lock(info->recorded_streams_mutex);
    info->block = block;
  }

  // Record the ALLOC trace entry with on the real NRT VA (block->data_ptr).
  // Propagate the segment's trace_id so sub-block allocs can be correlated with
  // the runtime's TENSOR_ALLOC event for the parent segment. The id is only used
  // by the snapshot, so only query it (an NRT call) when memory recording is
  // enabled, matching the rest of the snapshot code.
  uint64_t trace_id = 0;
  if (c10_neuron::NeuronMemorySnapshot::IsRecordingEnabled() && block->root &&
      block->root->nrt_tensor) {
    trace_id = at::neuron::nrt::GetTensorTraceId(block->root->nrt_tensor.get());
  }
  at::neuron::metrics::RecordNewAllocation(device_id, size, block->size, block->data_ptr,
                                           info->allocation_stream.id(), info->context, trace_id);
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
