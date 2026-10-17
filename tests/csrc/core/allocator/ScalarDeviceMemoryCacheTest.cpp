#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstring>

#include "tests/csrc/mocks/MockNRT.h"
#include "torch_neuronx/csrc/core/allocator/NeuronSegmentAllocator.h"
#include "torch_neuronx/csrc/core/allocator/ScalarDeviceMemoryCache.h"
#include "torch_neuronx/csrc/core/metrics/NeuronMetrics.h"
#include "torch_neuronx/csrc/core/utils/NeuronExceptions.h"

using namespace c10_neuron;
using namespace torch_neuronx::testing;
using ::testing::_;
using ::testing::AnyNumber;
using ::testing::DoAll;
using ::testing::Invoke;
using ::testing::Return;

// ============================================================================
// ScalarKey tests
// ============================================================================

TEST(ScalarKeyTest, EqualityForSameBits) {
  ScalarKey a{}, b{};
  float val = 3.14f;
  std::memcpy(a.bits.data(), &val, sizeof(val));
  std::memcpy(b.bits.data(), &val, sizeof(val));
  EXPECT_EQ(a, b);
}

TEST(ScalarKeyTest, InequalityForDifferentBits) {
  ScalarKey a{}, b{};
  int32_t va = 1, vb = 2;
  std::memcpy(a.bits.data(), &va, sizeof(va));
  std::memcpy(b.bits.data(), &vb, sizeof(vb));
  EXPECT_NE(a, b);
}

TEST(ScalarKeyTest, DifferentTypeSameValueDifferentKey) {
  ScalarKey a{}, b{};
  float fval = 1.0f;
  double dval = 1.0;
  std::memcpy(a.bits.data(), &fval, sizeof(fval));
  std::memcpy(b.bits.data(), &dval, sizeof(dval));
  EXPECT_NE(a, b);
}

TEST(ScalarKeyHashTest, SameKeysSameHash) {
  ScalarKey a{}, b{};
  int64_t val = 42;
  std::memcpy(a.bits.data(), &val, sizeof(val));
  std::memcpy(b.bits.data(), &val, sizeof(val));
  ScalarKeyHash hasher;
  EXPECT_EQ(hasher(a), hasher(b));
}

TEST(ScalarKeyHashTest, DifferentKeysDifferentHash) {
  ScalarKey a{}, b{};
  int64_t va = 1, vb = 2;
  std::memcpy(a.bits.data(), &va, sizeof(va));
  std::memcpy(b.bits.data(), &vb, sizeof(vb));
  ScalarKeyHash hasher;
  EXPECT_NE(hasher(a), hasher(b));
}

// ============================================================================
// ScalarDeviceMemoryCache test fixture
// ============================================================================

class ScalarDeviceMemoryCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_session_ = std::make_unique<MockNRTSession>();
    next_addr_ = 0x10000;

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

    EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_write(_, _, _, _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(NRT_SUCCESS));

    EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_free(_))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([](nrt_tensor_t** ptr) {
          if (ptr) *ptr = nullptr;
        }));

    cache_ = std::make_unique<ScalarDeviceMemoryCache>(/*device_id=*/0);
  }

  void TearDown() override {
    cache_.reset();
    mock_session_.reset();
  }

  ScalarKey MakeKey(const void* data, size_t size) {
    ScalarKey key{};
    std::memcpy(key.bits.data(), data, size);
    return key;
  }

  std::unique_ptr<MockNRTSession> mock_session_;
  std::unique_ptr<ScalarDeviceMemoryCache> cache_;
  uintptr_t next_addr_ = 0x10000;
};

// ============================================================================
// GetOrReserveVA — dispatch-side dedup + placeholder-VA reservation
// ============================================================================

TEST_F(ScalarDeviceMemoryCacheTest, ReserveVAAllocatesOnFirstAccess) {
  int32_t val = 42;
  auto [va, is_new] = cache_->GetOrReserveVA(MakeKey(&val, sizeof(val)), sizeof(val));

  EXPECT_NE(va, nullptr);
  EXPECT_TRUE(is_new);
  EXPECT_EQ(cache_->Size(), 1);
  // No pages yet — pages are only allocated by BindSlot on the worker thread.
  EXPECT_EQ(cache_->PageCount(), 0);
}

TEST_F(ScalarDeviceMemoryCacheTest, ReserveVAReturnsSameVAOnSecondAccess) {
  float val = 3.14f;
  auto key = MakeKey(&val, sizeof(val));

  auto [va1, new1] = cache_->GetOrReserveVA(key, sizeof(val));
  auto [va2, new2] = cache_->GetOrReserveVA(key, sizeof(val));

  EXPECT_EQ(va1, va2);
  EXPECT_TRUE(new1);
  EXPECT_FALSE(new2);  // hit
  EXPECT_EQ(cache_->Size(), 1);
}

TEST_F(ScalarDeviceMemoryCacheTest, DifferentValuesGetDifferentVAs) {
  int32_t v1 = 1, v2 = 2;
  auto [va1, new1] = cache_->GetOrReserveVA(MakeKey(&v1, sizeof(v1)), sizeof(v1));
  auto [va2, new2] = cache_->GetOrReserveVA(MakeKey(&v2, sizeof(v2)), sizeof(v2));

  EXPECT_NE(va1, va2);
  EXPECT_TRUE(new1);
  EXPECT_TRUE(new2);
  EXPECT_EQ(cache_->Size(), 2);
}

TEST_F(ScalarDeviceMemoryCacheTest, SameBitsShareVARegardlessOfDtype) {
  // int32_t bit pattern of float 1.0f dedups to the same entry.
  float fval = 1.0f;
  int32_t ival;
  std::memcpy(&ival, &fval, sizeof(ival));

  auto [va_f, new_f] = cache_->GetOrReserveVA(MakeKey(&fval, sizeof(fval)), sizeof(fval));
  auto [va_i, new_i] = cache_->GetOrReserveVA(MakeKey(&ival, sizeof(ival)), sizeof(ival));

  EXPECT_EQ(va_f, va_i);
  EXPECT_FALSE(new_i);  // same bits -> hit
  EXPECT_EQ(cache_->Size(), 1);
}

TEST_F(ScalarDeviceMemoryCacheTest, ZeroPaddingSharesVAAcrossWidths) {
  // The key holds no width, so every zero scalar shares one slot. The slot must
  // therefore hold the full zero-padded pattern.
  bool b = false;
  int32_t i32 = 0;
  int64_t i64 = 0;
  double f64 = 0.0;

  auto [va_b, new_b] = cache_->GetOrReserveVA(MakeKey(&b, sizeof(b)), sizeof(b));
  auto [va_32, new_32] = cache_->GetOrReserveVA(MakeKey(&i32, sizeof(i32)), sizeof(i32));
  auto [va_64, new_64] = cache_->GetOrReserveVA(MakeKey(&i64, sizeof(i64)), sizeof(i64));
  auto [va_f, new_f] = cache_->GetOrReserveVA(MakeKey(&f64, sizeof(f64)), sizeof(f64));

  EXPECT_TRUE(new_b);
  EXPECT_FALSE(new_32);
  EXPECT_FALSE(new_64);
  EXPECT_FALSE(new_f);
  EXPECT_EQ(va_32, va_b);
  EXPECT_EQ(va_64, va_b);
  EXPECT_EQ(va_f, va_b);
  EXPECT_EQ(cache_->Size(), 1);
}

TEST_F(ScalarDeviceMemoryCacheTest, DifferentPaddedPatternsGetDifferentVAs) {
  float f32 = 1.0f;
  double f64 = 1.0;
  auto [va32, new32] = cache_->GetOrReserveVA(MakeKey(&f32, sizeof(f32)), sizeof(f32));
  auto [va64, new64] = cache_->GetOrReserveVA(MakeKey(&f64, sizeof(f64)), sizeof(f64));

  EXPECT_NE(va32, va64);
  EXPECT_EQ(cache_->Size(), 2);
}

// ============================================================================
// BindSlot — worker-side slot packing
// ============================================================================

TEST_F(ScalarDeviceMemoryCacheTest, BindSlotReturnsValidSlice) {
  nrt_tensor_t* slice = cache_->BindSlot();
  EXPECT_NE(slice, nullptr);
  EXPECT_EQ(cache_->PageCount(), 1);
}

TEST_F(ScalarDeviceMemoryCacheTest, BindSlotDistinctSliceskSlotSizeApart) {
  nrt_tensor_t* s1 = cache_->BindSlot();
  nrt_tensor_t* s2 = cache_->BindSlot();
  EXPECT_NE(s1, s2);
  // Mock returns base+offset as the slice handle, so consecutive slots are
  // kSlotSize apart.
  auto diff = reinterpret_cast<uintptr_t>(s2) - reinterpret_cast<uintptr_t>(s1);
  EXPECT_EQ(diff, ScalarDeviceMemoryCache::kSlotSize);
}

TEST_F(ScalarDeviceMemoryCacheTest, BindSlotAllocatesNewPageWhenFull) {
  for (size_t i = 0; i < ScalarDeviceMemoryCache::kSlotsPerPage; ++i) {
    cache_->BindSlot();
  }
  EXPECT_EQ(cache_->PageCount(), 1);

  cache_->BindSlot();  // overflow -> new page
  EXPECT_EQ(cache_->PageCount(), 2);
}

TEST_F(ScalarDeviceMemoryCacheTest, BindSlotThrowsOnPageAllocationFailure) {
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_allocate(_, _, _, _, _))
      .Times(1)
      .WillOnce(Return(NRT_RESOURCE));

  EXPECT_THROW(cache_->BindSlot(), torch_neuronx::ExecutionRuntimeException);
}

// ============================================================================
// Clear
// ============================================================================

TEST_F(ScalarDeviceMemoryCacheTest, ClearReleasesAllResources) {
  // Reserve two distinct VAs (dispatch side) and bind their slots (worker side),
  // mirroring how a scalar miss drives the cache end to end.
  int32_t v1 = 1, v2 = 2;
  cache_->GetOrReserveVA(MakeKey(&v1, sizeof(v1)), sizeof(v1));
  cache_->BindSlot();
  cache_->GetOrReserveVA(MakeKey(&v2, sizeof(v2)), sizeof(v2));
  cache_->BindSlot();
  EXPECT_EQ(cache_->Size(), 2);
  EXPECT_EQ(cache_->PageCount(), 1);

  cache_->Clear();
  EXPECT_EQ(cache_->Size(), 0);
  EXPECT_EQ(cache_->PageCount(), 0);
}

TEST_F(ScalarDeviceMemoryCacheTest, ClearKeepsSlotsWhileATensorIsLive) {
  int32_t val = 7;
  cache_->GetOrReserveVA(MakeKey(&val, sizeof(val)), sizeof(val));
  cache_->BindSlot();
  cache_->RetainTensor();

  cache_->Clear();
  EXPECT_EQ(cache_->Size(), 1);
  EXPECT_EQ(cache_->PageCount(), 1);

  cache_->ReleaseTensor();
  EXPECT_EQ(cache_->LiveTensorCount(), 0);
  cache_->Clear();
  EXPECT_EQ(cache_->Size(), 0);
  EXPECT_EQ(cache_->PageCount(), 0);
}

TEST_F(ScalarDeviceMemoryCacheTest, CanReuseAfterClear) {
  int32_t val = 5;
  auto key = MakeKey(&val, sizeof(val));
  cache_->GetOrReserveVA(key, sizeof(val));
  cache_->Clear();

  auto [va, is_new] = cache_->GetOrReserveVA(key, sizeof(val));
  EXPECT_NE(va, nullptr);
  EXPECT_TRUE(is_new);  // fresh miss after clear
  EXPECT_EQ(cache_->Size(), 1);
}

// ============================================================================
// Metrics
// ============================================================================

namespace {
void ResetScalarMetrics(int device) {
  auto& stats = at::neuron::metrics::DeviceMemoryStatsRegistry::Instance().GetDeviceStats(device);
  stats.num_scalar_cache_hits.store(0, std::memory_order_relaxed);
  stats.num_scalar_cache_misses.store(0, std::memory_order_relaxed);
  stats.scalar_cache_size.store(0, std::memory_order_relaxed);
}
}  // namespace

TEST_F(ScalarDeviceMemoryCacheTest, MetricsMissIncrementsMissCountAndSize) {
  ResetScalarMetrics(0);
  auto& stats = at::neuron::metrics::DeviceMemoryStatsRegistry::Instance().GetDeviceStats(0);

  int32_t val = 42;
  cache_->GetOrReserveVA(MakeKey(&val, sizeof(val)), sizeof(val));

  EXPECT_EQ(stats.num_scalar_cache_misses.load(std::memory_order_relaxed), 1);
  EXPECT_EQ(stats.num_scalar_cache_hits.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(stats.scalar_cache_size.load(std::memory_order_relaxed), 1);
}

TEST_F(ScalarDeviceMemoryCacheTest, MetricsHitIncrementsHitCount) {
  ResetScalarMetrics(0);
  auto& stats = at::neuron::metrics::DeviceMemoryStatsRegistry::Instance().GetDeviceStats(0);

  int32_t val = 42;
  auto key = MakeKey(&val, sizeof(val));
  cache_->GetOrReserveVA(key, sizeof(val));  // miss
  cache_->GetOrReserveVA(key, sizeof(val));  // hit
  cache_->GetOrReserveVA(key, sizeof(val));  // hit

  EXPECT_EQ(stats.num_scalar_cache_misses.load(std::memory_order_relaxed), 1);
  EXPECT_EQ(stats.num_scalar_cache_hits.load(std::memory_order_relaxed), 2);
  EXPECT_EQ(stats.scalar_cache_size.load(std::memory_order_relaxed), 1);
}

TEST_F(ScalarDeviceMemoryCacheTest, MetricsClearResetsSizeToZero) {
  ResetScalarMetrics(0);
  auto& stats = at::neuron::metrics::DeviceMemoryStatsRegistry::Instance().GetDeviceStats(0);

  int32_t v1 = 1, v2 = 2, v3 = 3;
  cache_->GetOrReserveVA(MakeKey(&v1, sizeof(v1)), sizeof(v1));
  cache_->GetOrReserveVA(MakeKey(&v2, sizeof(v2)), sizeof(v2));
  cache_->GetOrReserveVA(MakeKey(&v3, sizeof(v3)), sizeof(v3));
  EXPECT_EQ(stats.scalar_cache_size.load(std::memory_order_relaxed), 3);

  cache_->Clear();
  EXPECT_EQ(stats.scalar_cache_size.load(std::memory_order_relaxed), 0);
  // Hits/misses are cumulative — not reset by Clear.
  EXPECT_EQ(stats.num_scalar_cache_misses.load(std::memory_order_relaxed), 3);
}
