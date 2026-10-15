#include "ScalarDeviceMemoryCache.h"

#include <sys/mman.h>
#include <torch_neuronx/csrc/core/NeuronDevice.h>
#include <torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h>
#include <torch_neuronx/csrc/core/metrics/NeuronMetrics.h>
#include <torch_neuronx/csrc/core/runtime/NRTHandler.h>
#include <torch_neuronx/csrc/core/runtime/NRTUtils.h>
#include <torch_neuronx/csrc/core/utils/NeuronExceptions.h>
#include <torch_neuronx/csrc/core/utils/NeuronResourceManager.h>

#include <cstring>

namespace c10_neuron {

ScalarDeviceMemoryCache::ScalarDeviceMemoryCache(int device_id) : device_id_(device_id) {}

ScalarDeviceMemoryCache::~ScalarDeviceMemoryCache() {
  // At process/static teardown destruction order of dynamic shared objects relative to
  // this cache is unspecified, so any of them may already be gone and cause segfault.
  // The OS reclaims the mmap'd placeholder VAs and NRT tears down
  // device memory exit, so skip managed cleanup.
  if (NeuronCachingAllocator::IsShuttingDown()) {
    return;
  }
  Clear();
}

std::pair<void*, bool> ScalarDeviceMemoryCache::GetOrReserveVA(const ScalarKey& key,
                                                               size_t data_size) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = key_to_va_.find(key);
  if (it != key_to_va_.end()) {
    at::neuron::metrics::RecordScalarCacheHit(device_id_);
    return {it->second, /*is_new=*/false};
  }

  EXECUTION_RUNTIME_CHECK(data_size <= kSlotSize, "ScalarDeviceMemoryCache: scalar width ",
                          data_size, " exceeds the slot size ", kSlotSize,
                          ". Use the standard allocation path.");

  // Reserve a placeholder VA for this value. PROT_NONE: it's an address handle
  // (the AllocatorRegistry key + tensor data_ptr), never dereferenced on host.
  // Reserve a full slot, not data_size: a wider dtype can alias this entry and
  // Clear() munmaps kSlotSize.
  void* va =
      mmap(nullptr, kSlotSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  EXECUTION_RUNTIME_CHECK(
      va != MAP_FAILED, "ScalarDeviceMemoryCache: mmap failed for placeholder VA, size=", kSlotSize,
      ": ", strerror(errno));

  key_to_va_.emplace(key, va);
  at::neuron::metrics::RecordScalarCacheMiss(device_id_);
  return {va, /*is_new=*/true};
}

nrt_tensor_t* ScalarDeviceMemoryCache::BindSlot() {
  std::lock_guard<std::mutex> lock(mutex_);

  ScalarPage* page = nullptr;
  if (!pages_.empty() && pages_.back()->next_slot < kSlotsPerPage) {
    page = pages_.back().get();
  } else {
    page = AllocatePage();
  }

  size_t slot_idx = page->next_slot++;
  size_t offset = slot_idx * kSlotSize;

  // Allocate slice. The value is written separately by a kWrite.
  nrt_tensor_t* slice = at::neuron::NRTHandler::AllocateSliceTensor(page->tensor, offset, kSlotSize,
                                                                    "scalar_cache_slot");
  slices_.push_back(slice);
  return slice;
}

void ScalarDeviceMemoryCache::RetainTensor() {
  live_tensors_.fetch_add(1, std::memory_order_relaxed);
}

void ScalarDeviceMemoryCache::ReleaseTensor() {
  live_tensors_.fetch_sub(1, std::memory_order_release);
}

size_t ScalarDeviceMemoryCache::LiveTensorCount() const {
  return live_tensors_.load(std::memory_order_acquire);
}

void ScalarDeviceMemoryCache::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);

  // Keep the cache while a tensor still aliases a slot.
  if (live_tensors_.load(std::memory_order_acquire) > 0) {
    return;
  }

  // Drop each shared placeholder VA's registry entry, then munmap it.
  for (auto& [key, va] : key_to_va_) {
    NeuronCachingAllocator::remove(va);
    munmap(va, kSlotSize);
  }
  key_to_va_.clear();

  // Free the NRT slices, then the backing pages' root tensors, then drop pages.
  for (nrt_tensor_t* slice : slices_) {
    if (slice) {
      at::neuron::nrt::FreeTensor(&slice);
    }
  }
  slices_.clear();
  for (auto& page : pages_) {
    if (page && page->tensor) {
      at::neuron::nrt::FreeTensor(&page->tensor);
    }
  }
  pages_.clear();

  at::neuron::metrics::RecordScalarCacheClear(device_id_);
}

size_t ScalarDeviceMemoryCache::Size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return key_to_va_.size();
}

size_t ScalarDeviceMemoryCache::PageCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return pages_.size();
}

ScalarPage* ScalarDeviceMemoryCache::AllocatePage() {
  int vnc_id = c10_neuron::get_vnc_id(device_id_);
  nrt_tensor_t* raw = at::neuron::NRTHandler::AllocateTensor(NRT_TENSOR_PLACEMENT_DEVICE, vnc_id,
                                                             kPageSize, "scalar_cache_page");

  auto page = std::make_unique<ScalarPage>();
  page->tensor = raw;
  page->base_va = at::neuron::nrt::GetTensorVA(raw);
  page->next_slot = 0;

  ScalarPage* ptr = page.get();
  pages_.push_back(std::move(page));
  return ptr;
}

ScalarDeviceMemoryCache& GetScalarCache(int device) {
  return at::neuron::NeuronResourceManager::Instance().GetScalarCache(device);
}

}  // namespace c10_neuron
