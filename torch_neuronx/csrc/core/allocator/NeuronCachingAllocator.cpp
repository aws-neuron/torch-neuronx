#include "NeuronCachingAllocator.h"

#include <c10/core/DeviceType.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <torch_neuronx/csrc/c10/neuron/NeuronStream.h>
#include <torch_neuronx/csrc/core/NeuronDevice.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <vector>

extern "C" {
#include <nrt/nrt.h>
}

namespace c10_neuron {
namespace NeuronCachingAllocator {

namespace {

// Get page size with proper error handling
inline size_t GetPageSize() {
  static size_t page_size = []() {
    long size = sysconf(_SC_PAGESIZE);
    TORCH_CHECK(size > 0, "Failed to get virtual address size");
    return static_cast<size_t>(size);
  }();
  return page_size;
}

// Utility function to check if all recorded streams have completed
bool areRecordedStreamsComplete(const std::unordered_set<c10::StreamId>& stream_ids, int device) {
  for (auto stream_id : stream_ids) {
    try {
      auto stream =
          at::neuron::NeuronStream(stream_id, device, at::neuron::NeuronStream::UNCHECKED);
      if (!stream.query()) {
        return false;
      }
    } catch (const std::exception&) {
      continue;
    }
  }
  return true;
}

// We need a global registry because DataPtr deleter cannot capture
// the allocator instance
struct AllocatorRegistry {
  struct AllocationInfo {
    void* ptr;  // Original pointer to the entry
    nrt_tensor_t* tensor;
    size_t size;
    int device;  // Add device index to track which device owns this allocation
    at::neuron::NeuronStream allocation_stream;          // Stream where allocated
    std::unordered_set<c10::StreamId> recorded_streams;  // Streams that have used this memory
    mutable std::mutex streams_mutex;                    // Protects recorded_streams

    AllocationInfo(void* ptr, nrt_tensor_t* t, size_t s, int dev,
                   const at::neuron::NeuronStream& alloc_stream)
        : ptr(ptr), tensor(t), size(s), device(dev), allocation_stream(alloc_stream) {}
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
    auto [it, inserted] = allocations.try_emplace(ptr, std::move(info));
    TORCH_CHECK(inserted, "NeuronAllocator: Duplicate allocation entry");
    return it->second.get();
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

  // Stream-aware methods
  void recordStream(void* ptr, const at::neuron::NeuronStream& stream) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = allocations.find(ptr);
    if (it != allocations.end()) {
      std::lock_guard<std::mutex> stream_lock(it->second->streams_mutex);
      it->second->recorded_streams.insert(stream.id());
    }
  }
};

// Type alias for convenience
using AllocationInfo = AllocatorRegistry::AllocationInfo;

// Simple memory pool that caches allocated blocks for reuse
class SimpleMemoryPool {
  struct Block {
    nrt_tensor_t* tensor;
    void* data_ptr;
    size_t size;
    std::chrono::steady_clock::time_point last_used;
    std::unordered_set<c10::StreamId> recorded_streams;  // Streams to wait for before reuse

    Block(nrt_tensor_t* t, void* ptr, size_t s, const std::unordered_set<c10::StreamId>& streams)
        : tensor(t),
          data_ptr(ptr),
          size(s),
          last_used(std::chrono::steady_clock::now()),
          recorded_streams(streams) {}
  };

  std::vector<Block> blocks_;
  mutable std::mutex mutex_;

 public:
  // Try to find a cached block of exact size where all recorded streams have completed
  std::pair<nrt_tensor_t*, void*> get_block(size_t size, int device) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Find exact size match where all recorded streams have completed
    auto it = std::find_if(blocks_.begin(), blocks_.end(), [this, size, device](const Block& b) {
      if (b.size != size) {
        return false;
      }
      return areRecordedStreamsComplete(b.recorded_streams, device);
    });

    if (it != blocks_.end()) {
      // Found a match - remove from cache and return
      nrt_tensor_t* tensor = it->tensor;
      void* data_ptr = it->data_ptr;
      blocks_.erase(it);
      return {tensor, data_ptr};
    }

    return {nullptr, nullptr};
  }

  // Add a freed block to the cache
  void add_block(nrt_tensor_t* tensor, void* data_ptr, size_t size,
                 const std::unordered_set<c10::StreamId>& recorded_streams) {
    std::lock_guard<std::mutex> lock(mutex_);
    blocks_.emplace_back(tensor, data_ptr, size, recorded_streams);
  }

  // Clear all cached blocks
  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& block : blocks_) {
      ReleaseVirtualAddress(block.data_ptr);
      nrt_tensor_free(&block.tensor);
    }
    blocks_.clear();
  }

  size_t cached_blocks() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return blocks_.size();
  }
};

// Per-device memory pools
// We dynamically initialize based on actual device count
std::vector<SimpleMemoryPool>* memory_pools = nullptr;
std::mutex pools_init_mutex;

// Get or initialize the memory pools
std::vector<SimpleMemoryPool>& getMemoryPools() {
  std::lock_guard<std::mutex> lock(pools_init_mutex);
  if (!memory_pools) {
    int device_count = c10_neuron::device_count();
    if (device_count <= 0) {
      throw std::runtime_error("No Neuron devices available. Cannot initialize memory pools.");
    }
    memory_pools = new std::vector<SimpleMemoryPool>(device_count);
  }
  return *memory_pools;
}

// Get pool for specific device
SimpleMemoryPool& getDevicePool(int device) {
  auto& pools = getMemoryPools();
  if (device < 0 || device >= static_cast<int>(pools.size())) {
    throw std::runtime_error("Invalid device index: " + std::to_string(device));
  }
  return pools[device];
}

// RAII helper to temporarily suppress stderr
class StderrSuppressor {
  int saved_stderr;
  int devnull;

 public:
  explicit StderrSuppressor(bool should_suppress) : saved_stderr(-1), devnull(-1) {
    if (!should_suppress) return;

    // Save current stderr
    saved_stderr = dup(STDERR_FILENO);
    if (saved_stderr == -1) return;

    // Open /dev/null
    devnull = open("/dev/null", O_WRONLY);
    if (devnull == -1) {
      close(saved_stderr);
      saved_stderr = -1;
      return;
    }

    // Redirect stderr to /dev/null
    dup2(devnull, STDERR_FILENO);
  }

  ~StderrSuppressor() {
    // Restore original stderr if we suppressed it
    if (saved_stderr != -1) {
      dup2(saved_stderr, STDERR_FILENO);
      close(saved_stderr);
    }

    if (devnull != -1) {
      close(devnull);
    }
  }

  // Delete copy/move to ensure RAII semantics
  StderrSuppressor(const StderrSuppressor&) = delete;
  StderrSuppressor& operator=(const StderrSuppressor&) = delete;
  StderrSuppressor(StderrSuppressor&&) = delete;
  StderrSuppressor& operator=(StderrSuppressor&&) = delete;
};

// Static deleter function that can be used with DataPtr
void neuron_deleter(void* ctx) {
  AllocationInfo* info = static_cast<AllocationInfo*>(ctx);
  TORCH_CHECK(info != nullptr, "NeuronAllocator: Deleter called on nullptr context");
  void* ptr = info->ptr;
  TORCH_CHECK(ptr != nullptr, "NeuronAllocator: Deleter called on nullptr pointer");
  nrt_tensor_t* tensor = info->tensor;
  size_t size = info->size;
  int device = info->device;
  std::unordered_set<c10::StreamId> recorded_streams;
  {
    std::lock_guard<std::mutex> stream_lock(info->streams_mutex);
    recorded_streams = info->recorded_streams;
  }
  // Remove from registry
  AllocatorRegistry::getInstance().remove(ptr);

  if (tensor && size > 0 && device >= 0) {
    // Add to the device-specific cache instead of freeing
    // TODO(Streams): Move the stream ownership to the cache.
    auto& device_pool = getDevicePool(device);
    device_pool.add_block(tensor, ptr, size, recorded_streams);
  } else {
    // This should never happen - it indicates a bug in the allocator
    ReleaseVirtualAddress(ptr);
    throw std::runtime_error(
        "NeuronAllocator: Deleter called on unregistered pointer. "
        "This indicates a serious bug - either double-free or missing registration.");
  }
}

class NeuronAllocator : public c10::Allocator {
 public:
  NeuronAllocator() = default;

  c10::DataPtr allocate(size_t size) override {
    // Get current device from guard
    auto device = c10_neuron::current_device();
    auto current_stream = at::neuron::getCurrentNeuronStream(device);

    // Handle zero-size allocation
    if (size == 0) {
      return c10::DataPtr(nullptr, c10::Device(c10::DeviceType::PrivateUse1, device));
    }

    // First, check the device-specific cache
    auto& device_pool = getDevicePool(device);
    auto [cached_tensor, cached_ptr] = device_pool.get_block(size, device);
    if (cached_tensor != nullptr) {
      // Register for cleanup tracking with device and stream info
      auto* alloc_info = AllocatorRegistry::getInstance().add(cached_ptr, cached_tensor, size,
                                                              device, current_stream);
      return c10::DataPtr(cached_ptr,
                          alloc_info,       // context
                          &neuron_deleter,  // static function pointer
                          c10::Device(c10::DeviceType::PrivateUse1, device));
    }

    // No cached block - allocate new

    nrt_tensor_t* tensor = nullptr;
    NRT_STATUS status;

    // If we have cached blocks, suppress error messages on first attempt
    // since we can recover from OOM by clearing cache
    {
      StderrSuppressor suppressor(device_pool.cached_blocks() > 0);

      status = nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE,     // tensor_placement
                                   c10_neuron::get_vnc_id(device),  // vnc (device index)
                                   size,                            // size
                                   nullptr,                         // name (optional)
                                   &tensor                          // output tensor
      );
    }  // suppressor destructor automatically restores stderr here

    // Check if we got OOM
    if (status == NRT_RESOURCE && device_pool.cached_blocks() > 0) {
      // Silently clear cache and retry - this is an implementation detail
      // Users don't need to know about automatic recovery

      // Clear the device-specific cache to free memory
      device_pool.clear();

      // Retry allocation
      status = nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, c10_neuron::get_vnc_id(device),
                                   size, nullptr, &tensor);
    }

    if (status != NRT_SUCCESS || !tensor) {
      std::string error_msg =
          "Failed to allocate Neuron tensor of size " + std::to_string(size) + " bytes";
      if (status == NRT_RESOURCE) {
        error_msg += " (out of memory)";
      }
      error_msg += ". Status: " + std::to_string(status);
      throw std::runtime_error(error_msg);
    }

    // Get virtual addr
    void* virtual_addr = ReserveVirtualAddress();
    if (!virtual_addr) {
      nrt_tensor_free(&tensor);
      throw std::runtime_error("Failed to allocate a virtual address");
    }

    // Store mapping for copy_data and deallocation with device and stream info
    auto* alloc_info =
        AllocatorRegistry::getInstance().add(virtual_addr, tensor, size, device, current_stream);

    // Create DataPtr with static deleter
    return c10::DataPtr(virtual_addr,
                        alloc_info,       // context
                        &neuron_deleter,  // static function pointer
                        c10::Device(c10::DeviceType::PrivateUse1, device));
  }

  void copy_data(void* dest, const void* src, std::size_t size) const override {
    // copy_data is only called internally by Allocator::clone()
    // We expect both pointers to be base pointers from our allocations
    nrt_tensor_t* src_tensor =
        c10_neuron::NeuronCachingAllocator::findTensor(const_cast<void*>(src));
    nrt_tensor_t* dst_tensor = c10_neuron::NeuronCachingAllocator::findTensor(dest);

    if (src_tensor && dst_tensor) {
      // Both are device tensors, use nrt_tensor_copy
      NRT_STATUS status = nrt_tensor_copy(src_tensor, 0,  // source tensor, offset
                                          dst_tensor, 0,  // dest tensor, offset
                                          size);
      if (status != NRT_SUCCESS) {
        throw std::runtime_error("nrt_tensor_copy failed");
      }
    } else {
      // This shouldn't happen in normal operation since clone()
      // should only be called with our allocations
      throw std::runtime_error(
          "NeuronAllocator::copy_data called with unknown pointers. "
          "Both source and destination must be allocated by this allocator.");
    }
  }
};

// Global singleton instance
std::once_flag allocator_init_flag;
NeuronAllocator* allocator_instance = nullptr;

NeuronAllocator* getAllocator() {
  std::call_once(allocator_init_flag, []() { allocator_instance = new NeuronAllocator(); });
  return allocator_instance;
}

}  // anonymous namespace

c10::Allocator* get() { return getAllocator(); }

void emptyCache() {
  // Clear all cached blocks across all devices (following CUDA's behavior)
  auto& pools = getMemoryPools();
  for (auto& pool : pools) {
    pool.clear();
  }
}

// Add a function to get cache statistics (useful for testing)
size_t getCachedBlocks() {
  size_t total = 0;
  auto& pools = getMemoryPools();
  for (const auto& pool : pools) {
    total += pool.cached_blocks();
  }
  return total;
}

bool copyTensorData(void* dest, const void* src, size_t size) {
  nrt_tensor_t* src_tensor = c10_neuron::NeuronCachingAllocator::findTensor(const_cast<void*>(src));
  nrt_tensor_t* dst_tensor = c10_neuron::NeuronCachingAllocator::findTensor(dest);

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

nrt_tensor_t* getTensorFromContext(void* ctx) {
  TORCH_CHECK(ctx != nullptr, "NeuronAllocator: Context is nullptr");
  auto* info = static_cast<AllocationInfo*>(ctx);
  return info->tensor;
}

// Stream-aware functionality
void recordStream(const c10::DataPtr& data_ptr, c10::Stream stream) {
  void* ptr = data_ptr.get();
  if (!ptr) {
    return;
  }
  at::neuron::NeuronStream neuron_stream(stream);
  AllocatorRegistry::getInstance().recordStream(ptr, neuron_stream);
}

// Helper function to reserve a virtual address to map entries
void* ReserveVirtualAddress() {
  void* vaddr =
      mmap(nullptr, GetPageSize(), PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (vaddr == MAP_FAILED) return nullptr;
  return vaddr;
}

// Helper function to release a virtual address
void ReleaseVirtualAddress(void* vaddr) {
  TORCH_CHECK(vaddr != nullptr, "Invalid address");
  munmap(vaddr, GetPageSize());
}

}  // namespace NeuronCachingAllocator
}  // namespace c10_neuron
