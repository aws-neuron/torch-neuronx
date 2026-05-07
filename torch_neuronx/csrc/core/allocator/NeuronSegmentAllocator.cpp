#include "NeuronSegmentAllocator.h"

#include <c10/util/Exception.h>
#include <torch_neuronx/csrc/core/NeuronDevice.h>
#include <torch_neuronx/csrc/core/NeuronLogging.h>
#include <torch_neuronx/csrc/core/metrics/NeuronMemorySnapshot.h>
#include <torch_neuronx/csrc/core/metrics/NeuronMetrics.h>
#include <torch_neuronx/csrc/core/runtime/NRTHandler.h>
#include <torch_neuronx/csrc/core/runtime/NRTUtils.h>
#include <torch_neuronx/csrc/core/utils/NeuronExceptions.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

// Shorthand for segment pool debug logging
#define SEGMENT_POOL_DEBUG(...)                                                         \
  TORCH_NEURONX_LOG(torch_neuronx::LogLevel::DEBUG, torch_neuronx::LogCategory::MEMORY, \
                    "[SEGMENT_POOL]", __VA_ARGS__)

extern "C" {
#include <nrt/nrt.h>
}

namespace c10_neuron {

size_t NeuronSegmentPool::RoundSize(size_t size) {
  if (size == 0) return 0;
  if (size < AllocatorConfig::kMinRoundedSize) return AllocatorConfig::kMinRoundedSize;
  // For large allocations (above kMediumThreshold), skip power-of-2 rounding.
  // We return same-sized segments with alignment in ComputeSegmentSize.
  if (size >= AllocatorConfig::kMediumThreshold) return size;

  static_assert(AllocatorConfig::kMediumThreshold <= (size_t{1} << (sizeof(size_t) * 8 - 1)),
                "kMediumThreshold must be <= 2^(bits(size_t) - 1)");

  constexpr int N = AllocatorConfig::kRoundupPower2Divisions;

  // Find the enclosing power-of-2 interval: [lower_power, upper_power)
  // where lower_power <= size < upper_power
  size_t upper_power = 1;
  while (upper_power < size) {
    upper_power <<= 1;
  }
  // If size is exactly a power of 2, return it
  if (upper_power == size) return size;

  size_t lower_power = upper_power >> 1;
  size_t step = (upper_power - lower_power) / N;
  if (step == 0) step = 1;

  // Find the smallest division boundary >= size
  for (int i = 1; i <= N; ++i) {
    size_t boundary = lower_power + step * i;
    if (boundary >= size) return boundary;
  }
  return upper_power;
}

size_t NeuronSegmentPool::ComputeSegmentSize(size_t size) {
  if (size < AllocatorConfig::kPoolBoundary) {
    return AllocatorConfig::kSmallSegmentSize;
  } else if (size < AllocatorConfig::kMediumThreshold) {
    return AllocatorConfig::kMediumSegmentSize;
  } else {
    constexpr size_t alignment = AllocatorConfig::kSegmentAlignment;
    static_assert(alignment != 0 && (alignment & (alignment - 1)) == 0,
                  "kSegmentAlignment must be a power of 2");
    return (size + alignment - 1) & ~(alignment - 1);
  }
}

NeuronSegmentPool::NeuronSegmentPool(int device_id) : device_id_(device_id) {}

NeuronSegmentPool::~NeuronSegmentPool() { EmptyCache(); }

Block* NeuronSegmentPool::Allocate(size_t requested_size) {
  if (requested_size == 0) return nullptr;

  std::lock_guard<std::mutex> lock(mutex_);

  size_t size = RoundSize(requested_size);
  bool is_small = size < AllocatorConfig::kPoolBoundary;

  SEGMENT_POOL_DEBUG("Allocate requested_size=", requested_size, " rounded_size=", size,
                     " pool=", (is_small ? "small" : "large"), " device=", device_id_);

  // Try best-fit from free pool
  Block* block = FindFreeBlock(size);

  // If not found, allocate a new segment from NRT
  if (!block) {
    SEGMENT_POOL_DEBUG("No free block found, allocating new segment");
    block = AllocateSegment(size, is_small);
  }

  if (!block) {
    auto segments = GetSnapshotSegmentsInternal();
    size_t total_reserved = 0, total_active = 0;
    for (const auto& seg : segments) {
      total_reserved += seg.total_size;
      total_active += seg.active_size;
    }
    TORCH_NEURONX_LOG(torch_neuronx::LogLevel::ERROR, torch_neuronx::LogCategory::MEMORY,
                      "[SEGMENT_POOL] Allocation failed (OOM): requested_size=", requested_size,
                      " rounded_size=", size, " device=", device_id_, " segments=", segments.size(),
                      " reserved=", total_reserved, " active=", total_active,
                      " cached_free=", (total_reserved - total_active));
    return nullptr;
  }

  // Split if the block is larger than needed
  block = MaybeSplit(block, size);

  // Create NRT slice for this block
  nrt_tensor_t* root_tensor = block->root->nrt_tensor.get();
  block->nrt_slice =
      at::neuron::NRTHandler::AllocateSliceTensor(root_tensor, block->offset, block->size, nullptr);

  // Mark active and register in ptr_to_block.
  block->active = true;
  block->requested_size = requested_size;
  ptr_to_block_[block->data_ptr] = block;

  SEGMENT_POOL_DEBUG("Allocate completed ptr=", block->data_ptr, " size=", block->size,
                     " offset=", block->offset, " is_root=", block->is_root(),
                     " is_split=", block->is_split());
  return block;
}

void NeuronSegmentPool::Free(Block* block) {
  if (!block) return;

  std::lock_guard<std::mutex> lock(mutex_);

  // Free the NRT slice handle before returning the block to the pool.
  if (block->nrt_slice) {
    at::neuron::nrt::FreeTensor(&block->nrt_slice);
    block->nrt_slice = nullptr;
  }

  // Remove from ptr lookup
  if (block->data_ptr) {
    ptr_to_block_.erase(block->data_ptr);
  }

  // Mark inactive, merge with adjacent free neighbors, return to free set.
  block->active = false;
  block = MergeNeighbors(block);
  InsertIntoFreeSet(block);

  SEGMENT_POOL_DEBUG("Free to pool completed ptr=", block->data_ptr, " size=", block->size,
                     " is_root=", block->is_root());
}

Block* NeuronSegmentPool::FindFreeBlock(size_t size) {
  bool is_small = size < AllocatorConfig::kPoolBoundary;

  // Create a search key block on stack
  Block search_key;
  search_key.size = size;
  search_key.offset = 0;

  // Best-fit search within the matching pool.
  auto& free_set = pools_[is_small ? kSmallPool : kLargePool];
  auto it = free_set.lower_bound(&search_key);
  if (it != free_set.end()) {
    Block* block = *it;
    free_set.erase(it);
    SEGMENT_POOL_DEBUG("FindFreeBlock hit size=", block->size);
    return block;
  }

  return nullptr;
}

Block* NeuronSegmentPool::AllocateSegment(size_t size, bool is_small) {
  size_t segment_size = ComputeSegmentSize(size);

  // Throws ExecutionRuntimeException on OOM. The runtime
  // worker's ErrorRecovery catches this for OOM recovery.
  nrt_tensor_t* raw_tensor = at::neuron::NRTHandler::AllocateTensor(
      NRT_TENSOR_PLACEMENT_DEVICE, c10_neuron::get_vnc_id(device_id_), segment_size, nullptr);

  void* base_addr = at::neuron::nrt::GetTensorVA(raw_tensor);
  if (C10_UNLIKELY(base_addr == nullptr)) {
    at::neuron::nrt::FreeTensor(&raw_tensor);
    EXECUTION_RUNTIME_CHECK(false, "AllocateSegment: GetTensorVA returned null",
                            " segment_size=", segment_size, " device=", device_id_);
  }

  // Create the root block spanning the entire segment.
  // Wrap the NRT tensor in a shared_ptr so slices can keep it alive.
  auto block_ptr = std::make_unique<Block>();
  Block* block = block_ptr.get();
  block->size = segment_size;
  block->offset = 0;
  block->active = false;
  block->is_small = is_small;
  block->nrt_tensor = std::shared_ptr<nrt_tensor_t>(
      raw_tensor, [](nrt_tensor_t* t) { at::neuron::nrt::FreeTensor(&t); });
  block->data_ptr = base_addr;
  block->prev = nullptr;
  block->next = nullptr;
  block->root = block;  // Root points to itself

  all_blocks_.push_back(std::move(block_ptr));
  block_index_[block] = all_blocks_.size() - 1;

  // Track physical device memory reserved by the allocator
  at::neuron::metrics::RecordReserve(device_id_, segment_size);

  SEGMENT_POOL_DEBUG("Allocated new segment block, size=", segment_size);

  return block;
}

Block* NeuronSegmentPool::MaybeSplit(Block* block, size_t size) {
  size_t remaining = block->size - size;

  // Determine split threshold based on pool
  size_t threshold = block->is_small ? AllocatorConfig::kSmallSplitThreshold
                                     : AllocatorConfig::kLargeSplitThreshold;

  if (remaining < threshold) {
    return block;
  }

  // Split block: create a new block for the remainder.
  auto rest_ptr = std::make_unique<Block>();
  Block* rest = rest_ptr.get();
  rest->size = remaining;
  rest->offset = block->offset + size;
  rest->active = false;
  rest->is_small = block->is_small;
  rest->nrt_tensor = nullptr;  // Not a root
  rest->data_ptr = static_cast<char*>(block->data_ptr) + size;
  rest->root = block->root;

  // Link into the doubly-linked list
  rest->prev = block;
  rest->next = block->next;
  if (block->next) {
    block->next->prev = rest;
  }
  block->next = rest;

  // Shrink the original block
  block->size = size;

  // Add the remainder to the pool's free set
  InsertIntoFreeSet(rest);

  all_blocks_.push_back(std::move(rest_ptr));
  block_index_[rest] = all_blocks_.size() - 1;
  return block;
}

/// Check if two free blocks can be merged.
static bool CanMerge(const Block* a, const Block* b) {
  if (!a || a->active || !b || b->active) return false;
  return true;
}

Block* NeuronSegmentPool::MergeNeighbors(Block* block) {
  // Merge with previous block if free
  if (CanMerge(block, block->prev)) {
    Block* prev = block->prev;

    // Remove prev from its free set
    RemoveFromFreeSet(prev);

    // Absorb: prev takes over block's space
    prev->size += block->size;
    prev->next = block->next;
    if (block->next) {
      block->next->prev = prev;
    }

    // Remove the absorbed block from ownership
    RemoveDeadBlock(block);

    block = prev;
  }

  // Merge with next block if free
  if (CanMerge(block, block->next)) {
    Block* next_block = block->next;

    // Remove next from its free set
    RemoveFromFreeSet(next_block);

    // Absorb: block takes over next's space
    block->size += next_block->size;
    block->next = next_block->next;
    if (next_block->next) {
      next_block->next->prev = block;
    }

    // Remove the absorbed block from ownership
    RemoveDeadBlock(next_block);
  }

  return block;
}

void NeuronSegmentPool::EmptyCache() {
  std::lock_guard<std::mutex> lock(mutex_);
  EmptyCacheInternal();
}

void NeuronSegmentPool::EmptyCacheInternal() {
  SEGMENT_POOL_DEBUG("EmptyCache starting total_blocks=", all_blocks_.size());

  // Clear both pools' free sets
  for (int i = 0; i < kNumPools; ++i) {
    pools_[i].clear();
  }

  // Identify root segments that have no active sub-blocks
  std::unordered_set<Block*> freeable_roots;
  {
    std::unordered_set<Block*> roots_with_active;
    for (const auto& bp : all_blocks_) {
      if (bp->active && bp->root) {
        roots_with_active.insert(bp->root);
      }
    }
    for (const auto& bp : all_blocks_) {
      if (bp->is_root() && bp->nrt_tensor && !roots_with_active.count(bp.get())) {
        freeable_roots.insert(bp.get());
      }
    }
  }

  // Collect every block belonging to a freeable root
  std::vector<Block*> dead_blocks;
  dead_blocks.reserve(all_blocks_.size());
  for (const auto& bp : all_blocks_) {
    if (freeable_roots.count(bp->root)) {
      dead_blocks.push_back(bp.get());
    }
  }

  // Release NRT tensors for the freeable roots
  for (Block* root : freeable_roots) {
    size_t seg_size = root->size;
    root->nrt_tensor.reset();
    root->data_ptr = nullptr;
    // Track physical device memory released back to NRT
    at::neuron::metrics::RecordUnreserve(device_id_, seg_size);
  }

  // Remove freed roots and their sub-blocks from internal data structures
  for (Block* block : dead_blocks) {
    RemoveDeadBlock(block);
  }

  SEGMENT_POOL_DEBUG("EmptyCache completed freed_segments=", freeable_roots.size());
}

size_t NeuronSegmentPool::GetFreeBlockCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t total = 0;
  for (int i = 0; i < kNumPools; ++i) {
    total += pools_[i].size();
  }
  return total;
}

size_t NeuronSegmentPool::GetTotalBlockCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return all_blocks_.size();
}

std::vector<NeuronMemorySnapshot::SegmentInfo> NeuronSegmentPool::GetSnapshotSegmentsInternal(
    const ContextResolver& context_resolver) const {
  // Caller must hold mutex_.
  std::unordered_set<Block*> seen_roots;
  std::vector<NeuronMemorySnapshot::SegmentInfo> result;

  for (const auto& bp : all_blocks_) {
    Block* root = bp->root;
    if (!root || !seen_roots.insert(root).second) continue;
    if (!root->nrt_tensor) continue;

    NeuronMemorySnapshot::SegmentInfo seg;
    seg.device = device_id_;
    seg.address = reinterpret_cast<size_t>(root->data_ptr);
    // SegmentInfo::stream is part of the public snapshot API; the segment
    // allocator is stream-agnostic, so always report 0.
    seg.stream = 0;
    seg.is_small = root->is_small;

    size_t total = 0;
    size_t allocated = 0;
    size_t active = 0;

    for (Block* b = root; b != nullptr; b = b->next) {
      NeuronMemorySnapshot::BlockInfo binfo;
      binfo.size = b->size;
      binfo.requested_size = b->requested_size;
      binfo.address = reinterpret_cast<size_t>(b->data_ptr);
      binfo.allocated = b->active;

      if (b->active && context_resolver) {
        binfo.context = context_resolver(b->data_ptr);
      }

      total += b->size;
      if (b->active) {
        allocated += b->size;
        active += b->requested_size;
      }

      seg.blocks.push_back(std::move(binfo));
    }

    seg.total_size = total;
    seg.allocated_size = allocated;
    seg.active_size = active;
    result.push_back(std::move(seg));
  }

  return result;
}

std::vector<NeuronMemorySnapshot::SegmentInfo> NeuronSegmentPool::GetSnapshotSegments(
    ContextResolver context_resolver) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return GetSnapshotSegmentsInternal(context_resolver);
}

void NeuronSegmentPool::RemoveFromFreeSet(Block* block) {
  auto& free_set = pools_[block->is_small ? kSmallPool : kLargePool];
  free_set.erase(block);
}

void NeuronSegmentPool::InsertIntoFreeSet(Block* block) {
  pools_[block->is_small ? kSmallPool : kLargePool].insert(block);
}

void NeuronSegmentPool::RemoveDeadBlock(Block* block) {
  auto it = block_index_.find(block);
  if (it == block_index_.end()) return;

  size_t idx = it->second;
  size_t last = all_blocks_.size() - 1;

  if (idx != last) {
    // Swap with last element
    Block* moved = all_blocks_[last].get();
    std::swap(all_blocks_[idx], all_blocks_[last]);
    block_index_[moved] = idx;
  }

  block_index_.erase(it);
  all_blocks_.pop_back();
}

}  // namespace c10_neuron
