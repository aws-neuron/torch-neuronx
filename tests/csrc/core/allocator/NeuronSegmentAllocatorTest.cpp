#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tests/csrc/mocks/MockNRT.h"
#include "torch_neuronx/csrc/core/allocator/NeuronSegmentAllocator.h"
#include "torch_neuronx/csrc/core/metrics/NeuronMemorySnapshot.h"
#include "torch_neuronx/csrc/core/utils/NeuronExceptions.h"

using namespace c10_neuron;
using namespace torch_neuronx::testing;
using ::testing::_;
using ::testing::AnyNumber;
using ::testing::AtLeast;
using ::testing::DoAll;
using ::testing::Invoke;
using ::testing::Return;
using ::testing::SetArgPointee;

// ============================================================================
// RoundSize tests
// ============================================================================

namespace {
struct DefaultConfigPool {
  NeuronSegmentPool pool{0, AllocatorConfig{}};
  const AllocatorConfig& cfg() const { return pool.config(); }
};
}  // namespace

TEST(RoundSizeTest, Zero) {
  DefaultConfigPool p;
  EXPECT_EQ(p.pool.RoundSize(0), 0);
}

TEST(RoundSizeTest, BelowMinimum) {
  DefaultConfigPool p;
  EXPECT_EQ(p.pool.RoundSize(255), p.cfg().kMinRoundedSize);
}

TEST(RoundSizeTest, ExactPowerOf2) {
  DefaultConfigPool p;
  EXPECT_EQ(p.pool.RoundSize(1024), 1024);
  EXPECT_EQ(p.pool.RoundSize(1 << 20), 1 << 20);  // 1MB
}

TEST(RoundSizeTest, LargeSizesPassThrough) {
  // Sizes at or above kMediumThreshold pass through without power-of-2 rounding.
  // ComputeSegmentSize handles alignment for these.
  DefaultConfigPool p;
  size_t large = p.cfg().kMediumThreshold + 1;
  EXPECT_EQ(p.pool.RoundSize(large), large);
}

// Helper: compute expected RoundSize for a given size using the same
// algorithm, parameterized by N = kRoundupPower2Divisions.
static size_t ExpectedRound(size_t size, const AllocatorConfig& cfg) {
  if (size == 0) return 0;
  if (size < cfg.kMinRoundedSize) return cfg.kMinRoundedSize;
  if (size >= cfg.kMediumThreshold) return size;
  const int N = cfg.kRoundupPower2Divisions;
  size_t upper = 1;
  while (upper < size) upper <<= 1;
  if (upper == size) return size;
  size_t lower = upper >> 1;
  size_t step = (upper - lower) / N;
  if (step == 0) step = 1;
  for (int i = 1; i <= N; ++i) {
    size_t b = lower + step * i;
    if (b >= size) return b;
  }
  return upper;
}

TEST(RoundSizeTest, NDivisions) {
  // Verify RoundSize matches expected divisions for power-of-2 intervals.
  // Works for any kRoundupPower2Divisions value.
  DefaultConfigPool p;
  const auto& cfg = p.cfg();
  const int N = cfg.kRoundupPower2Divisions;

  // Test in [512, 1024) interval
  size_t lo = 512, hi = 1024;
  size_t step = (hi - lo) / N;
  // Each division boundary should be exact
  for (int i = 1; i <= N; ++i) {
    size_t boundary = lo + step * i;
    EXPECT_EQ(p.pool.RoundSize(boundary), boundary);
  }
  // Just above lower -> rounds to first division
  EXPECT_EQ(p.pool.RoundSize(lo + 1), ExpectedRound(lo + 1, cfg));
  // Just above first division -> rounds to next
  if (N >= 2) {
    EXPECT_EQ(p.pool.RoundSize(lo + step + 1), ExpectedRound(lo + step + 1, cfg));
  }

  // Test a larger interval if below kMediumThreshold
  size_t MB = 1024 * 1024;
  if (2 * MB <= cfg.kMediumThreshold) {
    size_t lo2 = MB, hi2 = 2 * MB;
    size_t step2 = (hi2 - lo2) / N;
    for (int i = 1; i <= N; ++i) {
      size_t b = lo2 + step2 * i;
      EXPECT_EQ(p.pool.RoundSize(b), b);
    }
    EXPECT_EQ(p.pool.RoundSize(lo2 + 1), ExpectedRound(lo2 + 1, cfg));
  }
}

// ============================================================================
// ComputeSegmentSize tests
// ============================================================================

TEST(ComputeSegmentSizeTest, SmallPool) {
  // Anything below kPoolBoundary goes to small pool, kSmallSegmentSize
  DefaultConfigPool p;
  EXPECT_EQ(p.pool.ComputeSegmentSize(p.cfg().kPoolBoundary - 1), p.cfg().kSmallSegmentSize);
}

TEST(ComputeSegmentSizeTest, MediumPool) {
  // kPoolBoundary <= size < kMediumThreshold, kMediumSegmentSize
  DefaultConfigPool p;
  EXPECT_EQ(p.pool.ComputeSegmentSize(p.cfg().kPoolBoundary), p.cfg().kMediumSegmentSize);
  EXPECT_EQ(p.pool.ComputeSegmentSize(p.cfg().kMediumThreshold - 1), p.cfg().kMediumSegmentSize);
}

TEST(ComputeSegmentSizeTest, LargeRoundedTo2MB) {
  // size >= kMediumThreshold, round up to 2MB alignment (default config)
  DefaultConfigPool p;
  size_t MB = 1024 * 1024;
  EXPECT_EQ(p.pool.ComputeSegmentSize(p.cfg().kMediumThreshold), p.cfg().kMediumThreshold);
  EXPECT_EQ(p.pool.ComputeSegmentSize(100 * MB + 1), 102 * MB);
}

// ============================================================================
// Test fixture using MockNRT
// ============================================================================

class NeuronSegmentPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_session_ = std::make_unique<MockNRTSession>();
    next_addr_ = 0x10000;
    pool_ = std::make_unique<NeuronSegmentPool>(0);

    EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_allocate(_, _, _, _, _))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke(
            [this](nrt_tensor_placement_t, int32_t, size_t size, const char*, nrt_tensor_t** out) {
              auto* fake = reinterpret_cast<nrt_tensor_t*>(next_addr_);
              next_addr_ += size;
              *out = fake;
              return NRT_SUCCESS;
            }));

    EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_get_va(_))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([](const nrt_tensor_t* t) {
          return const_cast<void*>(reinterpret_cast<const void*>(t));
        }));

    EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_allocate_slice(_, _, _, _, _))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke(
            [](const nrt_tensor_t* base, size_t offset, size_t, const char*, nrt_tensor_t** out) {
              auto base_addr = reinterpret_cast<uintptr_t>(base);
              *out = reinterpret_cast<nrt_tensor_t*>(base_addr + offset);
              return NRT_SUCCESS;
            }));

    EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_free(_))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([](nrt_tensor_t** ptr) {
          if (ptr) *ptr = nullptr;
        }));
  }

  void TearDown() override {
    // Destroy pool first so its destructor's NRT calls hit the active mock.
    pool_.reset();
    mock_session_.reset();
  }

  std::unique_ptr<MockNRTSession> mock_session_;
  std::unique_ptr<NeuronSegmentPool> pool_;
  uintptr_t next_addr_ = 0x10000;
};

// ============================================================================
// Basic allocation tests
// ============================================================================

TEST_F(NeuronSegmentPoolTest, AllocateZero) {
  // No NRT calls expected — size 0 returns nullptr immediately.
  // Pool destructor: EmptyCache on empty pool, no NRT calls.
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_allocate(_, _, _, _, _)).Times(0);
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_get_va(_)).Times(0);
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_free(_)).Times(0);

  Block* b = pool_->Allocate(0);
  EXPECT_EQ(b, nullptr);
}

TEST_F(NeuronSegmentPoolTest, AllocateSmall) {
  Block* b = pool_->Allocate(1024);
  ASSERT_NE(b, nullptr);
  EXPECT_TRUE(b->active);
  EXPECT_NE(b->data_ptr, nullptr);
  EXPECT_GE(b->size, pool_->RoundSize(1024));
  EXPECT_TRUE(b->is_small);
  EXPECT_NE(b->nrt_slice, nullptr);
}

TEST_F(NeuronSegmentPoolTest, AllocateLarge) {
  size_t MB = 1024 * 1024;
  Block* b = pool_->Allocate(1 * MB);
  ASSERT_NE(b, nullptr);
  EXPECT_TRUE(b->active);
  EXPECT_FALSE(b->is_small);
  EXPECT_GE(b->size, 1 * MB);
}

TEST_F(NeuronSegmentPoolTest, AllocateThrowsOnOOM) {
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_allocate(_, _, _, _, _))
      .Times(1)
      .WillOnce(Return(NRT_RESOURCE));
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_free(_)).Times(0);

  EXPECT_THROW(pool_->Allocate(1024), torch_neuronx::ExecutionRuntimeException);
}

// ============================================================================
// nrt_slice lifecycle tests
// ============================================================================

TEST_F(NeuronSegmentPoolTest, AllocateSetsNrtSlice) {
  Block* b = pool_->Allocate(1024);
  ASSERT_NE(b, nullptr);
  EXPECT_NE(b->nrt_slice, nullptr);
}

TEST_F(NeuronSegmentPoolTest, FreeClearsNrtSlice) {
  Block* b = pool_->Allocate(1024);
  ASSERT_NE(b, nullptr);
  EXPECT_NE(b->nrt_slice, nullptr);

  // After Free, the slice should be released
  pool_->Free(b);
  // Block is merged back and nrt_slice freed via nrt_tensor_free
  EXPECT_EQ(b->nrt_slice, nullptr);
}

// ============================================================================
// Splitting tests
// ============================================================================

TEST_F(NeuronSegmentPoolTest, SplitCreatesRemainder) {
  Block* b = pool_->Allocate(1024);
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(b->size, pool_->RoundSize(1024));
  EXPECT_NE(b->next, nullptr);
  EXPECT_FALSE(b->next->active);
}

TEST_F(NeuronSegmentPoolTest, SplitReusesRemainder) {
  Block* b1 = pool_->Allocate(1024);
  Block* b2 = pool_->Allocate(1024);
  ASSERT_NE(b1, nullptr);
  ASSERT_NE(b2, nullptr);
  EXPECT_NE(b1->data_ptr, b2->data_ptr);
}

TEST_F(NeuronSegmentPoolTest, NoSplitBelowThreshold) {
  // Set up: create a free block flanked by an active block
  Block* b1 = pool_->Allocate(512);
  Block* b2 = pool_->Allocate(512);
  ASSERT_NE(b1, nullptr);
  ASSERT_NE(b2, nullptr);
  EXPECT_EQ(b1->size, 512u);
  EXPECT_EQ(b2->size, 512u);

  void* b1_ptr = b1->data_ptr;
  pool_->Free(b1);  // free block, can't merge (b2 active on the right)

  // Allocate from free block. Rounds up.
  // Remainder < threshold -> no split, gets entire block.
  Block* result = pool_->Allocate(350);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->data_ptr, b1_ptr);  // Reuses the same address
  EXPECT_EQ(result->size, 512u);        // Got the whole block (no split)
  EXPECT_TRUE(result->active);
}

// ============================================================================
// Free and merge tests
// ============================================================================

TEST_F(NeuronSegmentPoolTest, FreeMarksInactive) {
  Block* b = pool_->Allocate(1024);
  ASSERT_NE(b, nullptr);
  pool_->Free(b);
  EXPECT_FALSE(b->active);
}

TEST_F(NeuronSegmentPoolTest, FreeBlockIsReused) {
  Block* b1 = pool_->Allocate(1024);
  void* ptr1 = b1->data_ptr;
  pool_->Free(b1);

  Block* b2 = pool_->Allocate(1024);
  EXPECT_EQ(b2->data_ptr, ptr1);
}

TEST_F(NeuronSegmentPoolTest, MergeWithPrevBlock) {
  Block* b1 = pool_->Allocate(1024);
  Block* b2 = pool_->Allocate(1024);
  Block* b3 = pool_->Allocate(1024);
  ASSERT_NE(b1, nullptr);
  ASSERT_NE(b2, nullptr);
  ASSERT_NE(b3, nullptr);

  size_t b1_size = b1->size;
  size_t b2_size = b2->size;
  void* b1_ptr = b1->data_ptr;

  pool_->Free(b1);
  pool_->Free(b2);

  Block* merged = pool_->Allocate(b1_size + b2_size);
  ASSERT_NE(merged, nullptr);
  EXPECT_GE(merged->size, b1_size + b2_size);
  EXPECT_EQ(merged->data_ptr, b1_ptr);
}

TEST_F(NeuronSegmentPoolTest, MergeWithNextBlock) {
  Block* b1 = pool_->Allocate(1024);
  Block* b2 = pool_->Allocate(1024);
  ASSERT_NE(b1, nullptr);
  ASSERT_NE(b2, nullptr);

  size_t b1_size = b1->size;  // Save before Free (MergeNeighbors changes b1->size)

  pool_->Free(b2);
  pool_->Free(b1);

  Block* big = pool_->Allocate(b1_size + 1024);
  ASSERT_NE(big, nullptr);
}

TEST_F(NeuronSegmentPoolTest, FreeSetRetainsBlocksWithSameSizeAndOffset) {
  // Allocate from segment 1
  Block* b1 = pool_->Allocate(pool_->config().kMediumThreshold - 1);
  ASSERT_NE(b1, nullptr);

  // Now allocate to force segment 2
  Block* b2 = pool_->Allocate(pool_->config().kMediumThreshold - 1);
  ASSERT_NE(b2, nullptr);

  void* ptr1 = b1->data_ptr;
  void* ptr2 = b2->data_ptr;
  EXPECT_NE(ptr1, ptr2);

  // Free both — they have the same size
  pool_->Free(b1);
  pool_->Free(b2);

  // Both should be reusable
  Block* r1 = pool_->Allocate(pool_->config().kMediumThreshold - 1);
  Block* r2 = pool_->Allocate(pool_->config().kMediumThreshold - 1);
  ASSERT_NE(r1, nullptr);
  ASSERT_NE(r2, nullptr);
  EXPECT_NE(r1->data_ptr, r2->data_ptr);
}

// ============================================================================
// Block cleanup tests
// ============================================================================

TEST_F(NeuronSegmentPoolTest, MergeRemovesAbsorbedBlockFromOwnership) {
  // After merging two blocks, the absorbed block should be removed from
  // internal ownership (all_blocks_ shrinks via swap-and-pop).
  size_t blocks_before = pool_->GetTotalBlockCount();

  Block* b1 = pool_->Allocate(1024);
  Block* b2 = pool_->Allocate(1024);
  ASSERT_NE(b1, nullptr);
  ASSERT_NE(b2, nullptr);

  // After split: root block was split into b1 + b2 + remainder
  size_t blocks_after_alloc = pool_->GetTotalBlockCount();
  EXPECT_GT(blocks_after_alloc, blocks_before);

  // Free b2, then b1 — b1 merges with b2, absorbed block is removed
  pool_->Free(b2);
  pool_->Free(b1);

  size_t blocks_after_merge = pool_->GetTotalBlockCount();
  // Merge should have removed the absorbed block
  EXPECT_LT(blocks_after_merge, blocks_after_alloc);
}

TEST_F(NeuronSegmentPoolTest, EmptyCacheRemovesDeadBlocksFromOwnership) {
  // After EmptyCache frees a segment, all blocks belonging to that segment
  // should be cleaned up from all_blocks_.
  size_t blocks_before = pool_->GetTotalBlockCount();

  Block* b1 = pool_->Allocate(1024);
  Block* b2 = pool_->Allocate(1024);
  ASSERT_NE(b1, nullptr);
  ASSERT_NE(b2, nullptr);

  EXPECT_GT(pool_->GetTotalBlockCount(), blocks_before);

  pool_->Free(b1);
  pool_->Free(b2);
  pool_->EmptyCache();

  // All blocks from the freed segment should be removed
  EXPECT_EQ(pool_->GetTotalBlockCount(), blocks_before);
}

TEST_F(NeuronSegmentPoolTest, RepeatedSplitMergeCycleDoesNotLeakBlocks) {
  // Repeated alloc/free cycles with splitting and merging should not
  // cause unbounded growth of all_blocks_.
  size_t blocks_before = pool_->GetTotalBlockCount();

  for (int i = 0; i < 100; ++i) {
    Block* b1 = pool_->Allocate(1024);
    Block* b2 = pool_->Allocate(1024);
    ASSERT_NE(b1, nullptr);
    ASSERT_NE(b2, nullptr);
    pool_->Free(b1);
    pool_->Free(b2);
  }

  // With RemoveDeadBlock, all_blocks_ should not grow unboundedly.
  // After 100 cycles in the same segment, we expect 1 coalesced free block.
  size_t blocks_after = pool_->GetTotalBlockCount();
  EXPECT_EQ(blocks_after, blocks_before + 1);
}

// ============================================================================
// Best-fit behavior tests
// ============================================================================

TEST_F(NeuronSegmentPoolTest, BestFitSelectsSmallestFit) {
  Block* b1 = pool_->Allocate(1024);
  Block* b2 = pool_->Allocate(4096);
  Block* b3 = pool_->Allocate(2048);

  pool_->Free(b1);
  pool_->Free(b2);
  pool_->Free(b3);

  Block* result = pool_->Allocate(2048);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->size, pool_->RoundSize(2048));
}

TEST_F(NeuronSegmentPoolTest, BestFitFallsBackToFirstFit) {
  Block* b1 = pool_->Allocate(1024);
  Block* b2 = pool_->Allocate(1024);
  Block* b3 = pool_->Allocate(1024);
  ASSERT_NE(b1, nullptr);
  ASSERT_NE(b2, nullptr);
  ASSERT_NE(b3, nullptr);

  void* b1_ptr = b1->data_ptr;
  size_t b1_size = b1->size;

  pool_->Free(b3);
  pool_->Free(b1);

  Block* result = pool_->Allocate(b1_size);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->data_ptr, b1_ptr);
}

// ============================================================================
// Pool separation tests
// ============================================================================

TEST_F(NeuronSegmentPoolTest, SmallAndLargePoolsSeparate) {
  Block* small = pool_->Allocate(1024);
  ASSERT_NE(small, nullptr);
  EXPECT_TRUE(small->is_small);

  size_t MB = 1024 * 1024;
  Block* large = pool_->Allocate(1 * MB);
  ASSERT_NE(large, nullptr);
  EXPECT_FALSE(large->is_small);
  pool_->Free(small);
  pool_->Free(large);
}

// ============================================================================
// EmptyCache tests
// ============================================================================

TEST_F(NeuronSegmentPoolTest, EmptyCacheFreesUnusedSegments) {
  Block* b1 = pool_->Allocate(1024);
  Block* b2 = pool_->Allocate(1024);
  pool_->Free(b1);
  pool_->Free(b2);

  pool_->EmptyCache();
}

TEST_F(NeuronSegmentPoolTest, EmptyCacheKeepsActiveSegments) {
  Block* active = pool_->Allocate(1024);
  pool_->EmptyCache();
  EXPECT_TRUE(active->active);
  pool_->Free(active);
}

// ============================================================================
// Root tracking tests
// ============================================================================

TEST_F(NeuronSegmentPoolTest, RootPointsToSelf) {
  Block* b = pool_->Allocate(1024);
  ASSERT_NE(b, nullptr);
  EXPECT_NE(b->root, nullptr);
  EXPECT_TRUE(b->root->is_root());
  EXPECT_EQ(b->root->root, b->root);
}

TEST_F(NeuronSegmentPoolTest, SplitBlocksShareRoot) {
  Block* b1 = pool_->Allocate(1024);
  Block* b2 = pool_->Allocate(1024);
  ASSERT_NE(b1, nullptr);
  ASSERT_NE(b2, nullptr);
  EXPECT_EQ(b1->root, b2->root);
}

TEST_F(NeuronSegmentPoolTest, RootNrtTensorIsSharedPtr) {
  Block* b = pool_->Allocate(1024);
  ASSERT_NE(b, nullptr);
  EXPECT_NE(b->root->nrt_tensor, nullptr);
  EXPECT_NE(b->root->nrt_tensor.get(), nullptr);

  // Non-root blocks have null nrt_tensor
  if (b->next) {
    EXPECT_EQ(b->next->nrt_tensor, nullptr);
  }
}

// ============================================================================
// Multi-allocation stress test
// ============================================================================

TEST_F(NeuronSegmentPoolTest, ManySmallAllocations) {
  std::vector<Block*> blocks;
  for (int i = 0; i < 100; i++) {
    Block* b = pool_->Allocate(1024);
    ASSERT_NE(b, nullptr);
    blocks.push_back(b);
  }

  std::set<void*> ptrs;
  for (auto* b : blocks) {
    EXPECT_TRUE(ptrs.insert(b->data_ptr).second);
  }

  for (auto* b : blocks) {
    pool_->Free(b);
  }

  for (int i = 0; i < 100; i++) {
    Block* b = pool_->Allocate(1024);
    ASSERT_NE(b, nullptr);
    pool_->Free(b);
  }
}

TEST_F(NeuronSegmentPoolTest, GetFreeBlockCount) {
  EXPECT_EQ(pool_->GetFreeBlockCount(), 0u);

  Block* b = pool_->Allocate(1024);
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(pool_->GetFreeBlockCount(), 1u);  // remainder from split

  pool_->Free(b);
  EXPECT_EQ(pool_->GetFreeBlockCount(), 1u);  // merged root

  pool_->EmptyCache();
  EXPECT_EQ(pool_->GetFreeBlockCount(), 0u);
}

// ============================================================================
// GetSnapshotSegments tests
// ============================================================================

TEST_F(NeuronSegmentPoolTest, SnapshotEmptyPool) {
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_free(_)).Times(0);

  auto segs = pool_->GetSnapshotSegments();
  EXPECT_TRUE(segs.empty());
}

TEST_F(NeuronSegmentPoolTest, SnapshotSingleActiveBlock) {
  Block* b = pool_->Allocate(1024);
  ASSERT_NE(b, nullptr);

  auto segs = pool_->GetSnapshotSegments();
  ASSERT_EQ(segs.size(), 1u);

  auto& seg = segs[0];
  EXPECT_EQ(seg.device, 0);
  EXPECT_EQ(seg.address, reinterpret_cast<size_t>(b->root->data_ptr));
  EXPECT_EQ(seg.total_size, pool_->config().kSmallSegmentSize);
  EXPECT_TRUE(seg.is_small);
  EXPECT_GT(seg.allocated_size, 0u);
  EXPECT_GT(seg.active_size, 0u);

  // Segment should have multiple blocks (active + remainder from split)
  EXPECT_GE(seg.blocks.size(), 2u);

  // First block should be active
  EXPECT_TRUE(seg.blocks[0].allocated);
  EXPECT_EQ(seg.blocks[0].size, b->size);

  // Remainder block should be inactive
  EXPECT_FALSE(seg.blocks[1].allocated);
}

TEST_F(NeuronSegmentPoolTest, SnapshotAfterFreeShowsInactive) {
  Block* b = pool_->Allocate(1024);
  ASSERT_NE(b, nullptr);
  pool_->Free(b);

  auto segs = pool_->GetSnapshotSegments();
  ASSERT_EQ(segs.size(), 1u);

  // All blocks should be inactive after free + merge
  EXPECT_EQ(segs[0].allocated_size, 0u);
  EXPECT_EQ(segs[0].active_size, 0u);
  for (const auto& blk : segs[0].blocks) {
    EXPECT_FALSE(blk.allocated);
  }
}

TEST_F(NeuronSegmentPoolTest, SnapshotAfterEmptyCacheIsEmpty) {
  Block* b = pool_->Allocate(1024);
  ASSERT_NE(b, nullptr);
  pool_->Free(b);
  pool_->EmptyCache();

  auto segs = pool_->GetSnapshotSegments();
  EXPECT_TRUE(segs.empty());
}

TEST_F(NeuronSegmentPoolTest, SnapshotMultipleSegments) {
  // Small pool segment
  Block* small = pool_->Allocate(1024);
  ASSERT_NE(small, nullptr);

  // Large pool segment
  size_t MB = 1024 * 1024;
  Block* large = pool_->Allocate(1 * MB);
  ASSERT_NE(large, nullptr);

  auto segs = pool_->GetSnapshotSegments();
  ASSERT_EQ(segs.size(), 2u);

  // One should be small, one large
  bool found_small = false, found_large = false;
  for (const auto& seg : segs) {
    if (seg.is_small)
      found_small = true;
    else
      found_large = true;
  }
  EXPECT_TRUE(found_small);
  EXPECT_TRUE(found_large);
}

TEST_F(NeuronSegmentPoolTest, SnapshotContextResolver) {
  Block* b = pool_->Allocate(1024);
  ASSERT_NE(b, nullptr);

  auto fake_ctx = std::make_shared<c10_neuron::NeuronMemorySnapshot::StackContext>();
  void* expected_ptr = b->data_ptr;

  auto resolver =
      [&](void* ptr) -> std::shared_ptr<c10_neuron::NeuronMemorySnapshot::StackContext> {
    if (ptr == expected_ptr) return fake_ctx;
    return nullptr;
  };

  auto segs = pool_->GetSnapshotSegments(resolver);
  ASSERT_EQ(segs.size(), 1u);

  // Active block should have the context from the resolver
  bool found_context = false;
  for (const auto& blk : segs[0].blocks) {
    if (blk.allocated && blk.context == fake_ctx) {
      found_context = true;
    }
  }
  EXPECT_TRUE(found_context);
}
