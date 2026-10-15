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
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

extern "C" {
#include <nrt/nrt.h>
}

namespace c10_neuron {

namespace {

// ---------------------------------------------------------------------------
// AllocatorConfig env-var parsing
// ---------------------------------------------------------------------------
//
// First call to AllocatorConfig::Get() latches in defaults, applies any
// AllocatorConfig::kEnvVarName override, validates the result, and returns
// a const reference. The default NeuronSegmentPool constructor copies that
// frozen singleton; the (device_id, config) constructor takes a
// user-supplied config and validates it directly.
// ---------------------------------------------------------------------------

// Field key strings used in the env-var spec. Centralized so log messages
// and the parser reference the same identifiers.
constexpr std::string_view kKeyPoolBoundary = "kPoolBoundary";
constexpr std::string_view kKeySmallSegmentSize = "kSmallSegmentSize";
constexpr std::string_view kKeyMediumSegmentSize = "kMediumSegmentSize";
constexpr std::string_view kKeyMediumThreshold = "kMediumThreshold";
constexpr std::string_view kKeySegmentAlignment = "kSegmentAlignment";
constexpr std::string_view kKeyMinRoundedSize = "kMinRoundedSize";
constexpr std::string_view kKeySmallSplitThreshold = "kSmallSplitThreshold";
constexpr std::string_view kKeyLargeSplitThreshold = "kLargeSplitThreshold";
constexpr std::string_view kKeyMaxSplitSizeBytes = "kMaxSplitSizeBytes";
constexpr std::string_view kKeyMaxNonSplitRoundingBytes = "kMaxNonSplitRoundingBytes";
constexpr std::string_view kKeyRoundupPower2Divisions = "kRoundupPower2Divisions";

// Largest representable threshold below the high bit of size_t so power-of-2
// expansion in RoundSize can't overflow.
constexpr size_t kMediumThresholdCap = (std::numeric_limits<size_t>::max)() / 2 + 1;

// Compile-time invariants on the in-class defaults. AllocatorConfig is an
// aggregate with constant default initializers, so the no-arg construction
// is a constant expression.
constexpr AllocatorConfig kAllocatorDefaults{};
static_assert(kAllocatorDefaults.kMediumThreshold > 0 &&
                  kAllocatorDefaults.kMediumThreshold <= kMediumThresholdCap,
              "default kMediumThreshold must be in (0, 2^(bits(size_t)-1)]");
static_assert(kAllocatorDefaults.kSegmentAlignment != 0 &&
                  (kAllocatorDefaults.kSegmentAlignment &
                   (kAllocatorDefaults.kSegmentAlignment - 1)) == 0,
              "default kSegmentAlignment must be a power of 2");
static_assert(kAllocatorDefaults.kMinRoundedSize > 0, "default kMinRoundedSize must be > 0");
static_assert(kAllocatorDefaults.kRoundupPower2Divisions > 0,
              "default kRoundupPower2Divisions must be > 0");

// Trim leading/trailing ASCII whitespace.
std::string_view Trim(std::string_view s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.remove_prefix(1);
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.remove_suffix(1);
  return s;
}

// Parse `s` (after trimming) as a non-negative decimal integer with no
// trailing characters. Returns nullopt on parse error or overflow.
std::optional<size_t> ParseUnsigned(std::string_view s) {
  s = Trim(s);
  if (s.empty()) return std::nullopt;
  for (char c : s) {
    if (!std::isdigit(static_cast<unsigned char>(c))) return std::nullopt;
  }
  try {
    size_t pos = 0;
    size_t value = std::stoull(std::string(s), &pos);
    if (pos != s.size()) return std::nullopt;
    return value;
  } catch (...) {
    return std::nullopt;
  }
}

// Apply a single key=value override to `cfg`. Logs and skips on parse error.
void ApplyOverride(AllocatorConfig& cfg, std::string_view key, std::string_view value) {
  // All size_t fields parse uniformly; route them through one table to avoid
  // a long if/else chain. Member pointers let the lookup write through to
  // the matching field.
  static const std::unordered_map<std::string_view, size_t AllocatorConfig::*> kSizeFields = {
      {kKeyPoolBoundary, &AllocatorConfig::kPoolBoundary},
      {kKeySmallSegmentSize, &AllocatorConfig::kSmallSegmentSize},
      {kKeyMediumSegmentSize, &AllocatorConfig::kMediumSegmentSize},
      {kKeyMediumThreshold, &AllocatorConfig::kMediumThreshold},
      {kKeySegmentAlignment, &AllocatorConfig::kSegmentAlignment},
      {kKeyMinRoundedSize, &AllocatorConfig::kMinRoundedSize},
      {kKeySmallSplitThreshold, &AllocatorConfig::kSmallSplitThreshold},
      {kKeyLargeSplitThreshold, &AllocatorConfig::kLargeSplitThreshold},
      {kKeyMaxSplitSizeBytes, &AllocatorConfig::kMaxSplitSizeBytes},
      {kKeyMaxNonSplitRoundingBytes, &AllocatorConfig::kMaxNonSplitRoundingBytes},
  };

  if (auto it = kSizeFields.find(key); it != kSizeFields.end()) {
    if (auto v = ParseUnsigned(value)) {
      cfg.*(it->second) = *v;
    } else {
      TORCH_NEURONX_ALLOCATOR_WARN("AllocatorConfig: invalid bytes value for ", key, "=", value);
    }
    return;
  }

  if (key == kKeyRoundupPower2Divisions) {
    if (auto v = ParseUnsigned(value); v && *v >= 1 && *v <= 64) {
      cfg.kRoundupPower2Divisions = static_cast<int>(*v);
    } else {
      TORCH_NEURONX_ALLOCATOR_WARN("AllocatorConfig: invalid value for ",
                                   kKeyRoundupPower2Divisions,
                                   " (expected integer in [1, 64])=", value);
    }
    return;
  }

  TORCH_NEURONX_ALLOCATOR_WARN("AllocatorConfig: unknown key in ", AllocatorConfig::kEnvVarName,
                               ", ignoring: ", key);
}

// Parse "a=1048576,b=67108864,..." into a sequence of ApplyOverride calls.
void ApplyEnvOverrides(AllocatorConfig& cfg) {
  const char* env = std::getenv(AllocatorConfig::kEnvVarName);
  if (!env || !*env) return;

  std::string_view spec(env);
  while (!spec.empty()) {
    auto comma = spec.find(',');
    std::string_view kv = (comma == std::string_view::npos) ? spec : spec.substr(0, comma);
    auto eq = kv.find('=');
    if (eq != std::string_view::npos) {
      auto key = Trim(kv.substr(0, eq));
      auto value = kv.substr(eq + 1);
      ApplyOverride(cfg, key, value);
    } else if (!kv.empty()) {
      TORCH_NEURONX_ALLOCATOR_WARN("AllocatorConfig: malformed token in ",
                                   AllocatorConfig::kEnvVarName, ": ", kv);
    }
    if (comma == std::string_view::npos) break;
    spec.remove_prefix(comma + 1);
  }
}

// Sanity-check a config and clamp obviously broken values back to defaults.
// Logs every clamp so users can find the issue.
void ValidateConfig(AllocatorConfig& cfg) {
  // Print original values (which may be negative for kRoundupPower2Divisions)
  // and replacement values without coercing to size_t.
  auto warn_replace = [](std::string_view name, auto bad, auto replacement) {
    TORCH_NEURONX_ALLOCATOR_WARN("AllocatorConfig: ", name, "=", bad, " is invalid, using ",
                                 replacement);
  };
  auto is_pow2 = [](size_t v) { return v != 0 && (v & (v - 1)) == 0; };

  // Per-field absolute invariants.
  if (cfg.kSegmentAlignment == 0 || !is_pow2(cfg.kSegmentAlignment)) {
    warn_replace("kSegmentAlignment (must be power of 2)", cfg.kSegmentAlignment,
                 kAllocatorDefaults.kSegmentAlignment);
    cfg.kSegmentAlignment = kAllocatorDefaults.kSegmentAlignment;
  }
  if (cfg.kMinRoundedSize == 0) {
    warn_replace("kMinRoundedSize (must be > 0)", cfg.kMinRoundedSize,
                 kAllocatorDefaults.kMinRoundedSize);
    cfg.kMinRoundedSize = kAllocatorDefaults.kMinRoundedSize;
  }
  if (cfg.kRoundupPower2Divisions <= 0) {
    warn_replace("kRoundupPower2Divisions (must be > 0)", cfg.kRoundupPower2Divisions,
                 kAllocatorDefaults.kRoundupPower2Divisions);
    cfg.kRoundupPower2Divisions = kAllocatorDefaults.kRoundupPower2Divisions;
  }
  if (cfg.kMediumThreshold == 0 || cfg.kMediumThreshold > kMediumThresholdCap) {
    warn_replace("kMediumThreshold (must be in (0, 2^(bits-1)])", cfg.kMediumThreshold,
                 kAllocatorDefaults.kMediumThreshold);
    cfg.kMediumThreshold = kAllocatorDefaults.kMediumThreshold;
  }

  // Cross-field invariants. The small pool only services allocations
  // < kPoolBoundary, so its segment must cover that range; the medium/large
  // pool's RoundSize and ComputeSegmentSize both assume kMediumThreshold sits
  // strictly above kPoolBoundary so the power-of-2 interval search has room.
  if (cfg.kSmallSegmentSize < cfg.kPoolBoundary) {
    warn_replace("kSmallSegmentSize (must be >= kPoolBoundary)", cfg.kSmallSegmentSize,
                 cfg.kPoolBoundary);
    cfg.kSmallSegmentSize = cfg.kPoolBoundary;
  }
  if (cfg.kMediumThreshold <= cfg.kPoolBoundary) {
    size_t replacement =
        (cfg.kPoolBoundary >= kMediumThresholdCap) ? kMediumThresholdCap : cfg.kPoolBoundary + 1;
    warn_replace("kMediumThreshold (must be > kPoolBoundary)", cfg.kMediumThreshold, replacement);
    cfg.kMediumThreshold = replacement;
  }
}

const AllocatorConfig& EnsureInitialized() {
  static std::mutex init_mutex;
  static AllocatorConfig cfg;
  static bool initialized = false;

  std::lock_guard<std::mutex> lock(init_mutex);
  if (!initialized) {
    ApplyEnvOverrides(cfg);
    ValidateConfig(cfg);
    initialized = true;
  }
  return cfg;
}

}  // anonymous namespace

const AllocatorConfig& AllocatorConfig::Get() { return EnsureInitialized(); }

// ---------------------------------------------------------------------------
// NeuronSegmentPool
// ---------------------------------------------------------------------------

// Default constructor: copy from the already validated global singleton.
NeuronSegmentPool::NeuronSegmentPool(int device_id)
    : device_id_(device_id), config_(AllocatorConfig::Get()) {}

// User-supplied config: validate before storing so a bad caller config
// produces the same warnings and clamps as a bad env-var override.
NeuronSegmentPool::NeuronSegmentPool(int device_id, AllocatorConfig config)
    : device_id_(device_id), config_(std::move(config)) {
  ValidateConfig(config_);
}

NeuronSegmentPool::~NeuronSegmentPool() { EmptyCache(); }

size_t NeuronSegmentPool::RoundSize(size_t size) const {
  if (size == 0) return 0;
  if (size < config_.kMinRoundedSize) return config_.kMinRoundedSize;
  // For large allocations (above kMediumThreshold), skip power-of-2 rounding.
  // Same-sized segments are emitted with alignment in ComputeSegmentSize.
  if (size >= config_.kMediumThreshold) return size;

  const int N = config_.kRoundupPower2Divisions;

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

size_t NeuronSegmentPool::ComputeSegmentSize(size_t size) const {
  if (size < config_.kPoolBoundary) {
    return config_.kSmallSegmentSize;
  } else if (size < config_.kMediumThreshold) {
    return config_.kMediumSegmentSize;
  }
  // size >= kMediumThreshold: round up to alignment.
  size_t alignment = config_.kSegmentAlignment;
  return (size + alignment - 1) & ~(alignment - 1);
}

Block* NeuronSegmentPool::Allocate(size_t requested_size) {
  if (requested_size == 0) return nullptr;

  std::lock_guard<std::mutex> lock(mutex_);

  size_t size = RoundSize(requested_size);
  bool is_small = size < config_.kPoolBoundary;

  TORCH_NEURONX_ALLOCATOR_DEBUG("Allocate requested_size=", requested_size, " rounded_size=", size,
                                " pool=", (is_small ? "small" : "large"), " device=", device_id_);

  // Try best-fit from free pool
  Block* block = FindFreeBlock(size);

  // If not found, allocate a new segment from NRT
  if (!block) {
    TORCH_NEURONX_ALLOCATOR_DEBUG("No free block found, allocating new segment");
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
                      "[SEGMENT_POOL] Allocation failed: requested_size=", requested_size,
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

  TORCH_NEURONX_ALLOCATOR_DEBUG("Allocate completed ptr=", block->data_ptr, " size=", block->size,
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

  TORCH_NEURONX_ALLOCATOR_DEBUG("Free to pool completed ptr=", block->data_ptr,
                                " size=", block->size, " is_root=", block->is_root());
}

Block* NeuronSegmentPool::FindFreeBlock(size_t size) {
  bool is_small = size < config_.kPoolBoundary;

  // Create a search key block on stack
  Block search_key;
  search_key.size = size;
  search_key.offset = 0;

  // Best-fit search within the matching pool.
  auto& free_set = pools_[is_small ? kSmallPool : kLargePool];
  auto it = free_set.lower_bound(&search_key);
  if (it != free_set.end()) {
    Block* block = *it;
    // Non-split-reuse policy: when a free block is above the
    // configured split cap (kMaxSplitSizeBytes), reject if the
    // implied waste exceeds kMaxNonSplitRoundingBytes
    // and let the caller allocate a fresh segment instead.
    if (block->size > config_.kMaxSplitSizeBytes &&
        block->size - size > config_.kMaxNonSplitRoundingBytes) {
      TORCH_NEURONX_ALLOCATOR_DEBUG(
          "FindFreeBlock rejecting non-split block: block_size=", block->size, " request=", size,
          " waste=", block->size - size,
          " kMaxNonSplitRoundingBytes=", config_.kMaxNonSplitRoundingBytes);
      return nullptr;
    }
    free_set.erase(it);
    TORCH_NEURONX_ALLOCATOR_DEBUG("FindFreeBlock hit size=", block->size);
    at::neuron::metrics::RecordAllocatorCacheHit(device_id_);
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

  // Record cache miss on successful allocation
  at::neuron::metrics::RecordAllocatorCacheMiss(device_id_);
  // Track physical device memory reserved from NRT.
  // Trace-only: emit SEGMENT_ALLOC with NRT trace correlation ID. The correlation
  // id is only needed by the snapshot, so only query it (an NRT call) when memory
  // recording is enabled, matching the rest of the snapshot code.
  uint64_t trace_id = c10_neuron::NeuronMemorySnapshot::IsRecordingEnabled()
                          ? at::neuron::nrt::GetTensorTraceId(raw_tensor)
                          : 0;
  at::neuron::metrics::RecordSegmentReserve(device_id_, base_addr, segment_size, trace_id);

  TORCH_NEURONX_ALLOCATOR_DEBUG("Allocated new segment block, size=", segment_size);

  return block;
}

Block* NeuronSegmentPool::MaybeSplit(Block* block, size_t size) {
  // Blocks larger than kMaxSplitSizeBytes are never split. Combined with the
  // FindFreeBlock acceptance check, this keeps large free blocks intact
  // for future large requests.
  if (block->size > config_.kMaxSplitSizeBytes) {
    TORCH_NEURONX_ALLOCATOR_DEBUG("MaybeSplit skip (above split cap) block_size=", block->size,
                                  " request=", size,
                                  " kMaxSplitSizeBytes=", config_.kMaxSplitSizeBytes);
    return block;
  }

  size_t remaining = block->size - size;

  // Determine split threshold based on pool
  size_t threshold = block->is_small ? config_.kSmallSplitThreshold : config_.kLargeSplitThreshold;

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
  at::neuron::metrics::RecordSplit(device_id_);
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

    at::neuron::metrics::RecordMerge(device_id_);
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

    at::neuron::metrics::RecordMerge(device_id_);
  }

  return block;
}

void NeuronSegmentPool::EmptyCache() {
  std::lock_guard<std::mutex> lock(mutex_);
  EmptyCacheInternal();
}

void NeuronSegmentPool::EmptyCacheInternal() {
  TORCH_NEURONX_ALLOCATOR_DEBUG("EmptyCache starting total_blocks=", all_blocks_.size());

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
    void* seg_base = root->data_ptr;
    root->nrt_tensor.reset();
    root->data_ptr = nullptr;
    // Track physical device memory released back to NRT
    // Trace-only: emit SEGMENT_FREE trace event (no-op when memory history recording disabled).
    at::neuron::metrics::RecordSegmentUnreserve(device_id_, seg_base, seg_size);
  }

  // Remove freed roots and their sub-blocks from internal data structures
  for (Block* block : dead_blocks) {
    RemoveDeadBlock(block);
  }

  TORCH_NEURONX_ALLOCATOR_DEBUG("EmptyCache completed freed_segments=", freeable_roots.size());
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
