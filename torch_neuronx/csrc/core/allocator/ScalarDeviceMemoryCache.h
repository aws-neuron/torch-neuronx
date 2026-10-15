#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

struct nrt_tensor;
typedef struct nrt_tensor nrt_tensor_t;

namespace c10_neuron {

// 128-bit key representing any scalar value, zero-padded to the width of
// complex128 (the widest scalar type at 16 bytes).
struct ScalarKey {
  alignas(16) std::array<uint8_t, 16> bits{};

  bool operator==(const ScalarKey& other) const { return bits == other.bits; }
};

struct ScalarKeyHash {
  size_t operator()(const ScalarKey& key) const {
    uint64_t lo, hi;
    std::memcpy(&lo, key.bits.data(), 8);
    std::memcpy(&hi, key.bits.data() + 8, 8);
    // Rotate hi to break symmetry for complex128 where real == imag
    return lo ^ (hi << 1 | hi >> 63);
  }
};

struct ScalarPage {
  nrt_tensor_t* tensor = nullptr;  // Root NRT tensor (4KiB allocation)
  void* base_va = nullptr;         // Base virtual address of the page
  size_t next_slot = 0;            // Next unallocated slot index
};

// Self-contained, per-device content-addressed cache for scalar tensors.
//
// A scalar's device presence is one logical unit owned  here:
//   ScalarKey (128-bit value)
//     -> placeholder VA (host mmap, reserved on the dispatch thread; used as the
//        tensor's data_ptr and the AllocatorRegistry key)
//     -> NRT slice into a shared 4KiB page (bound on the worker thread)
// All of it is freed together in Clear().
// Deduplication means identical bit patterns share one placeholder VA and
// one slice regardless of dtype; entries are permanent until Clear().
//
// Thread model:
//   - GetOrReserveVA runs on the dispatch (Python) thread: pure map + mmap, no
//     NRT. Returns the shared VA and whether it is newly reserved.
//   - BindSlot runs on the worker thread (from a scalar kAlloc's AllocBind):
//     packs a 16-byte NRT slice from a shared page. Called once per unique value.
//   - Clear runs on the dispatch thread after a device sync.
//
// Thread-safety: all public methods hold mutex_.
class ScalarDeviceMemoryCache {
 public:
  static constexpr size_t kPageSize = 4096;                       // 4KiB minimum NRT allocation
  static constexpr size_t kSlotSize = 16;                         // Padded for complex128
  static constexpr size_t kSlotsPerPage = kPageSize / kSlotSize;  // 256

  explicit ScalarDeviceMemoryCache(int device_id);
  ~ScalarDeviceMemoryCache();

  ScalarDeviceMemoryCache(const ScalarDeviceMemoryCache&) = delete;
  ScalarDeviceMemoryCache& operator=(const ScalarDeviceMemoryCache&) = delete;

  // Dispatch thread. Look up the shared placeholder VA for `key`. On a miss,
  // reserves a new placeholder VA (host mmap of kSlotSize) and records it, then
  // records hit/miss metrics. `data_size` (<= kSlotSize) is the caller's scalar
  // width; the reservation is always a full slot, because a wider dtype can
  // alias the same entry.
  // @return {shared placeholder VA, is_new} — is_new true iff this call reserved
  //         it (caller must then submit the kAlloc + kWrite for the value).
  std::pair<void*, bool> GetOrReserveVA(const ScalarKey& key, size_t data_size);

  // Worker thread. Pack a fresh 16-byte NRT slice from a shared page (allocating
  // a new page when the current one is full). Called once per unique value from
  // the scalar kAlloc's AllocBind; the value is written by a separate kWrite.
  // Returns the slice; it stays valid until Clear.
  nrt_tensor_t* BindSlot();

  // Count a tensor that aliases a cached slot. Clear() keeps the cache while
  // any such tensor is alive. Any thread.
  void RetainTensor();
  void ReleaseTensor();
  size_t LiveTensorCount() const;

  // Dispatch thread, after a device sync. Free everything atomically: drop each
  // placeholder VA's AllocatorRegistry entry, munmap it, free all NRT slices and
  // backing pages, and reset the cache-size metric.
  // Pages are freed whole, so one live tensor keeps every slot. Clear() is a
  // no-op while LiveTensorCount() > 0.
  void Clear();

  // Number of unique scalars currently cached.
  size_t Size() const;

  // Number of 4KiB pages allocated.
  size_t PageCount() const;

  int device_id() const { return device_id_; }

 private:
  ScalarPage* AllocatePage();

  int device_id_;
  mutable std::mutex mutex_;

  // Live tensors aliasing a cached slot. Atomic, not guarded by mutex_: the
  // tensor deleter must not block on a Clear() in progress.
  std::atomic<size_t> live_tensors_{0};

  // Dedup index: value -> shared placeholder VA.
  std::unordered_map<ScalarKey, void*, ScalarKeyHash> key_to_va_;
  // All NRT slices handed out by BindSlot, retained so Clear can free them.
  std::vector<nrt_tensor_t*> slices_;
  std::vector<std::unique_ptr<ScalarPage>> pages_;
};

// Per-device scalar cache instance (peer of the segment/tensor pools).
ScalarDeviceMemoryCache& GetScalarCache(int device);

}  // namespace c10_neuron
