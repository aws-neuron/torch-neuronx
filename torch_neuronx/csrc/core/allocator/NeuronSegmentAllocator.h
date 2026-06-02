#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

// Forward declarations for NRT types
struct nrt_tensor;
typedef struct nrt_tensor nrt_tensor_t;

namespace c10_neuron {
namespace NeuronMemorySnapshot {
struct SegmentInfo;
struct StackContext;
}  // namespace NeuronMemorySnapshot
}  // namespace c10_neuron

namespace c10_neuron {

// ============================================================================
// Configuration
// ============================================================================
//
// AllocatorConfig holds the segment-pool tuning knobs. Defaults match the
// historical compile-time constants. Overrides are picked up from the
// `TORCH_NEURONX_SEGMENT_ALLOCATOR_CONFIG` environment variable on the first call to
// `AllocatorConfig::Get()`. Format is comma-separated `key=value` pairs:
//
//   TORCH_NEURONX_SEGMENT_ALLOCATOR_CONFIG="kPoolBoundary=1048576,kMediumThreshold=33554432"
//
// All values are raw non-negative decimal integers (sizes are in bytes).
// All segment-pool internals read through `AllocatorConfig::Get()` once, at
// pool construction, into a per-pool `config_` snapshot — so a pool's
// behavior is fixed for its lifetime and unaffected by any later mutation.

struct AllocatorConfig {
  // Pool boundary: allocations below this go to small pool
  size_t kPoolBoundary = 1024 * 1024;  // 1 MB

  // Segment (super tensor) sizes
  size_t kSmallSegmentSize = 2 * 1024 * 1024;    // 2MB
  size_t kMediumSegmentSize = 64 * 1024 * 1024;  // 64MB
  size_t kMediumThreshold = 16 * 1024 * 1024;    // 16MB
  size_t kSegmentAlignment = 2 * 1024 * 1024;    // 2MB

  // Rounding: power-of-2 with N divisions.
  int kRoundupPower2Divisions = 4;
  size_t kMinRoundedSize = 256;  // 256B

  // Small threshold: a remainder smaller than the minimum
  // allocation size would never be useful.
  size_t kSmallSplitThreshold = kMinRoundedSize;
  // Large threshold: allocation smaller than the small pool boundary size
  // will never go to this pool.
  size_t kLargeSplitThreshold = kPoolBoundary;

  // Cap above which a free block cannot be split. Default = SIZE_MAX
  // (unlimited, all blocks re splittable). Setting a cap is a
  // fragmentation-control trade-off. Pairs with kMaxNonSplitRoundingBytes
  // for size-tolerant reuse of blocks this cap prevents from splitting.
  size_t kMaxSplitSizeBytes = SIZE_MAX;

  // When a free block is non-splittable (block.size > kMaxSplitSizeBytes),
  // allow it to serve a request whose size is within this tolerance of the
  // block's size. Increase to widen reuse at the cost of more internal fragmentation.
  size_t kMaxNonSplitRoundingBytes = 20 * 1024 * 1024;  // 20 MB

  /// Environment variable name parsed on the first call to `Get()`.
  /// Exposed so tests and scripts can reference the canonical name.
  static constexpr const char* kEnvVarName = "TORCH_NEURONX_SEGMENT_ALLOCATOR_CONFIG";

  /// Read-only access to the active config. First call latches the env-var
  /// overrides; subsequent calls return the same instance.
  static const AllocatorConfig& Get();
};

// ============================================================================
// Block: data structure for NRT allocations
// ============================================================================
//
// A root block is one allocated directly from NRT. When a root block is split,
// it shrinks and a new block is created for the remainder. Sub-blocks reference
// the root to obtain the base NRT tensor for slicing.
//
// Blocks within a root form a doubly-linked list in address order.
// Merging combines adjacent free blocks. When a root block is fully
// merged back (no splits remain), it can be returned to NRT (EmptyCache).

struct Block {
  size_t size = 0;            // Size of this block in bytes
  size_t requested_size = 0;  // Original requested size before rounding
  size_t offset = 0;          // Byte offset within the root block's NRT tensor
  bool active = false;        // Currently allocated status
  bool is_small = false;      // Belongs to small pool (< kPoolBoundary)

  // Shared ownership of the root segment's NRT tensor. Root blocks only.
  std::shared_ptr<nrt_tensor_t> nrt_tensor = nullptr;

  // NRT slice handle for this block. Created when the block is allocated,
  // freed when the block is returned to the pool. Only set on active blocks.
  // The slice's lifetime is bounded by the block's active period; the
  // underlying segment is kept alive by the root block's nrt_tensor.
  nrt_tensor_t* nrt_slice = nullptr;

  void* data_ptr = nullptr;  // Virtual address (root base VA + offset)

  Block* prev = nullptr;  // Previous block (lower address), nullptr if first
  Block* next = nullptr;  // Next block (higher address), nullptr if last
  Block* root = nullptr;  // Points to the root block (self if this is the root)

  /// True if this block was allocated directly from NRT (i.e., the segment)
  bool is_root() const { return root == this; }

  /// True if this block was split from a larger block
  bool is_split() const { return prev != nullptr || next != nullptr; }
};

// Compare blocks by size for best-fit.
// Tie-break on data_ptr for address-ordered first-fit among same-size blocks.
struct BlockSizeComparator {
  bool operator()(const Block* a, const Block* b) const {
    if (a->size != b->size) return a->size < b->size;
    return std::less<void*>()(a->data_ptr, b->data_ptr);
  }
};

using FreeBlockSet = std::set<Block*, BlockSizeComparator>;

// ============================================================================
// NeuronSegmentPool: per-device allocator with pool-based free sets
// ============================================================================

/// @brief NRT tensor allocator using segment-based pooling with best-fit block
///        allocation, splitting and merging.
///
/// Memory hierarchy:
///   Segment (super tensor, large NRT allocation)
///     └─ Block (sub-region, allocated via nrt_tensor_allocate_slice)
///         ├─ Block (active, returned to user)
///         └─ Block (free, in a single FreeBlockSet per pool category for reuse)
///
/// Two pool categories (small/large), each with a single free set:
///   - small (index 0): for allocations < kPoolBoundary
///   - large (index 1): for allocations >= kPoolBoundary
///
/// Allocation strategy:
///   1. Best-fit search the matching pool's free set
///   2. Allocate a new segment from NRT on miss
class NeuronSegmentPool {
 public:
  /// Default constructor. Snapshots the global AllocatorConfig::Get() (with
  /// any env-var overrides already applied and validated) into this pool.
  /// The snapshot is fixed for the pool's lifetime. Most callers go through
  /// this path via NeuronResourceManager::GetSegmentPool.
  explicit NeuronSegmentPool(int device_id);

  /// Config / test constructor. Uses the supplied `config` exactly
  /// (no env-var lookup), validating it the same way env-var-driven configs
  /// are validated. The pool's behavior for its lifetime is determined by
  /// this snapshot; subsequent global config changes do not affect it.
  NeuronSegmentPool(int device_id, AllocatorConfig config);

  ~NeuronSegmentPool();

  NeuronSegmentPool(const NeuronSegmentPool&) = delete;
  NeuronSegmentPool& operator=(const NeuronSegmentPool&) = delete;

  /// Allocate a block of at least `size` bytes.
  /// Searches the matching pool's free set first, then NRT.
  /// @param size Requested size in bytes (will be rounded up)
  /// @return Pointer to allocated block, or nullptr on failure.
  ///         Caller gets block->data_ptr for the virtual address.
  Block* Allocate(size_t size);

  /// Free a previously allocated block back to its pool's free set.
  /// @param block Block to free
  void Free(Block* block);

  /// Release all cached memory back to NRT.
  void EmptyCache();

  /// Return the total number of free (cached) blocks across both pools.
  size_t GetFreeBlockCount() const;

  /// Build a memory snapshot of all segments and their blocks.
  /// @param context_resolver Called for each active block to obtain its stack context.
  ///        Signature: (void* data_ptr) -> shared_ptr<StackContext>
  using ContextResolver = std::function<std::shared_ptr<NeuronMemorySnapshot::StackContext>(void*)>;
  std::vector<NeuronMemorySnapshot::SegmentInfo> GetSnapshotSegments(
      ContextResolver context_resolver = nullptr) const;

  int device_id() const { return device_id_; }

  /// Read-only access to this pool's frozen-at-construction config.
  const AllocatorConfig& config() const { return config_; }

  /// Return the total number of Block objects owned by the pool.
  size_t GetTotalBlockCount() const;

  // ---- Utilities ----

  /// Round a size using power-of-2 with N divisions, per this pool's config.
  size_t RoundSize(size_t size) const;

  /// Compute the segment size for a given allocation size, per this pool's
  /// config.
  size_t ComputeSegmentSize(size_t size) const;

 private:
  // Pool category index
  static constexpr int kSmallPool = 0;
  static constexpr int kLargePool = 1;
  static constexpr int kNumPools = 2;

  // ---- Core operations ----

  /// Find the best-fit free block in the matching pool.
  Block* FindFreeBlock(size_t size);

  /// Allocate a new segment (super tensor) from NRT.
  /// @param size Minimum usable size needed.
  /// @param is_small Whether this goes to the small pool.
  /// @return The initial block spanning the entire segment, or nullptr on OOM.
  Block* AllocateSegment(size_t size, bool is_small);

  /// Split a block if remainder is large enough.
  /// Returns the (possibly smaller) block to hand out.
  Block* MaybeSplit(Block* block, size_t size);

  /// Merge a free block with its free neighbors.
  /// Returns the merged block.
  Block* MergeNeighbors(Block* block);

  /// Empty internal cache, and return free segments to NRT. Must be called with lock held.
  void EmptyCacheInternal();

  /// Remove a block from its pool's free set.
  void RemoveFromFreeSet(Block* block);

  /// Insert a block into its pool's free set.
  void InsertIntoFreeSet(Block* block);

  /// Remove a dead block from all_blocks_ via swap-and-pop (O(1)).
  /// The block must be fully unlinked (not in free sets, prev/next chain, or ptr_to_block_).
  void RemoveDeadBlock(Block* block);

  /// Lock-free snapshot builder. Caller must hold mutex_.
  std::vector<NeuronMemorySnapshot::SegmentInfo> GetSnapshotSegmentsInternal(
      const ContextResolver& context_resolver = nullptr) const;

  // ---- Data ----
  int device_id_;
  AllocatorConfig config_;

  // One free set per pool category: pools_[0] = small, pools_[1] = large.
  FreeBlockSet pools_[kNumPools];

  // Owns all Block objects (stable pointers via unique_ptr)
  std::vector<std::unique_ptr<Block>> all_blocks_;

  // Reverse index: Block* -> index in all_blocks_ for O(1) swap-and-pop removal
  std::unordered_map<Block*, size_t> block_index_;

  // Lookup: data_ptr -> Block* for O(1) Free()
  std::unordered_map<void*, Block*> ptr_to_block_;

  mutable std::mutex mutex_;
};

}  // namespace c10_neuron
