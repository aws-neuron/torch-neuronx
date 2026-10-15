#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include "tests/csrc/mocks/MockNRT.h"
#include "tests/csrc/mocks/MockNeuronDevice.h"
#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"
#include "torch_neuronx/csrc/core/allocator/NeuronSegmentAllocator.h"
#include "torch_neuronx/csrc/core/allocator/ShelvesCleanup.h"
#include "torch_neuronx/csrc/core/metrics/NeuronMemorySnapshot.h"
#include "torch_neuronx/csrc/core/utils/NeuronResourceManager.h"

using namespace c10_neuron;
using namespace torch_neuronx::testing;
using ::testing::_;
using ::testing::AnyNumber;
using ::testing::DoAll;
using ::testing::Invoke;
using ::testing::Return;
using ::testing::SetArgPointee;

class NeuronCachingAllocatorTest : public ::testing::Test {
 protected:
  static bool streams_initialized_;

  void SetUp() override {
    mock_session_ = std::make_unique<MockNRTSession>();
    next_addr_ = 0x10000;

    // Mock device count: 1 fake Neuron device for testing
    c10_neuron::testing::SetMockDeviceCount(1);

    // Mock nrt_get_visible_vnc_count so stream pool init discovers 1 device
    EXPECT_CALL(*MockNRT::GetInstance(), nrt_get_visible_vnc_count(_))
        .Times(AnyNumber())
        .WillRepeatedly(DoAll(SetArgPointee<0>(1u), Return(NRT_SUCCESS)));

    // Initialize stream pools once (needed for NeuronEvent::record in neuron_deleter)
    if (!streams_initialized_) {
      at::neuron::InitializeStreamPools();
      streams_initialized_ = true;
    }

    // Mock NRT tensor allocation: return fake tensor pointers with unique addresses
    EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_allocate(_, _, _, _, _))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke(
            [this](nrt_tensor_placement_t, int32_t, size_t size, const char*, nrt_tensor_t** out) {
              auto* fake = reinterpret_cast<nrt_tensor_t*>(next_addr_);
              next_addr_ += std::max(size, static_cast<size_t>(0x1000));
              *out = fake;
              return NRT_SUCCESS;
            }));

    // nrt_tensor_get_va: return address of the tensor itself (identity mapping)
    EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_get_va(_))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([](const nrt_tensor_t* t) {
          return const_cast<void*>(reinterpret_cast<const void*>(t));
        }));

    // nrt_tensor_allocate_slice: return base + offset as fake slice handle
    EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_allocate_slice(_, _, _, _, _))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke(
            [](const nrt_tensor_t* base, size_t offset, size_t, const char*, nrt_tensor_t** out) {
              auto base_addr = reinterpret_cast<uintptr_t>(base);
              *out = reinterpret_cast<nrt_tensor_t*>(base_addr + offset);
              return NRT_SUCCESS;
            }));

    // nrt_tensor_free: just null out the pointer
    EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_free(_))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([](nrt_tensor_t** ptr) {
          if (ptr) *ptr = nullptr;
        }));

    // Mock async NRT operations so the worker thread can process kAlloc/kDealloc.
    // nrta_event_register_seq_id_completion: signal the eventfd immediately
    // so the runtime worker wakes from read(efd) and processes completions.
    EXPECT_CALL(*MockNRT::GetInstance(), nrta_event_register_seq_id_completion(_, _))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([](nrta_seq_t, int fd) -> NRT_STATUS {
          uint64_t val = 1;
          ::write(fd, &val, sizeof(val));
          return NRT_SUCCESS;
        }));

    // nrta_get_sequence: return UINT64_MAX to indicate all operations completed.
    EXPECT_CALL(*MockNRT::GetInstance(), nrta_get_sequence(_, _, _, _))
        .Times(AnyNumber())
        .WillRepeatedly(DoAll(SetArgPointee<3>(UINT64_MAX), Return(NRT_SUCCESS)));

    // nrta_tensor_write: the scalar cache miss path submits a kWrite for the
    // value. Set the async-status out-param (arg 6) to NRT_SUCCESS so the op's
    // completion status isn't the fail-closed NRT_FAILURE default, and a nonzero
    // sequence out-param (arg 7) to match how the other async NRT ops are mocked.
    EXPECT_CALL(*MockNRT::GetInstance(), nrta_tensor_write(_, _, _, _, _, _, _, _))
        .Times(AnyNumber())
        .WillRepeatedly(
            DoAll(SetArgPointee<6>(NRT_SUCCESS), SetArgPointee<7>(1u), Return(NRT_SUCCESS)));

    // Initialize the execution engine (needed for lazy alloc SubmitOperationContext)
    at::neuron::NeuronResourceManager::Instance();
  }

  void TearDown() override {
    // Synchronize all devices to complete pending async operations (lazy alloc kAlloc/kDealloc)
    for (int device = 0; device < c10_neuron::testing::device_count(); ++device) {
      at::neuron::synchronize(device);
    }
    // Drain segment pools back to NRT so state doesn't leak across tests.
    NeuronCachingAllocator::emptyCache();
    mock_session_.reset();
  }

  /// allocate and return the DataPtr.
  c10::DataPtr allocate(size_t size) {
    auto* allocator = NeuronCachingAllocator::get();
    return allocator->allocate(size);
  }

  /// Synchronize device 0 to wait for pending kAlloc/kDealloc to complete.
  /// Call this after allocate() for findTensors to work.
  void syncDevice() { at::neuron::synchronize(0); }

  // Get a valid non-default c10::Stream for device 0.
  static c10::Stream getNonDefaultStream() {
    return c10::Stream(c10::Stream::UNSAFE, c10::Device(c10::DeviceType::PrivateUse1, 0),
                       /*stream_id=*/32);
  }

  std::unique_ptr<MockNRTSession> mock_session_;
  uintptr_t next_addr_ = 0x10000;
};

bool NeuronCachingAllocatorTest::streams_initialized_ = false;

TEST_F(NeuronCachingAllocatorTest, AllocateReturnsValidDataPtr) {
  auto data_ptr = allocate(1024);

  EXPECT_NE(data_ptr.get(), nullptr);
  EXPECT_EQ(data_ptr.device().type(), c10::DeviceType::PrivateUse1);
  EXPECT_EQ(data_ptr.device().index(), 0);
}

TEST_F(NeuronCachingAllocatorTest, AllocateZeroReturnsNull) {
  auto data_ptr = allocate(0);

  EXPECT_EQ(data_ptr.get(), nullptr);
}

TEST_F(NeuronCachingAllocatorTest, MultipleAllocationsReturnDistinctPointers) {
  auto dp1 = allocate(1024);
  auto dp2 = allocate(2048);
  auto dp3 = allocate(512);

  EXPECT_NE(dp1.get(), nullptr);
  EXPECT_NE(dp2.get(), nullptr);
  EXPECT_NE(dp3.get(), nullptr);

  // All pointers must be distinct
  EXPECT_NE(dp1.get(), dp2.get());
  EXPECT_NE(dp1.get(), dp3.get());
  EXPECT_NE(dp2.get(), dp3.get());
}

TEST_F(NeuronCachingAllocatorTest, FindTensorReturnsNullForNullptr) {
  EXPECT_EQ(NeuronCachingAllocator::findTensor(nullptr), nullptr);
}

TEST_F(NeuronCachingAllocatorTest, FindTensorReturnsNullForUnknownPtr) {
  void* bogus = reinterpret_cast<void*>(0xDEADBEEF);
  EXPECT_EQ(NeuronCachingAllocator::findTensor(bogus), nullptr);
}

TEST_F(NeuronCachingAllocatorTest, FindTensorReturnsHandleForAllocatedPtr) {
  auto dp = allocate(1024);
  void* ptr = dp.get();
  ASSERT_NE(ptr, nullptr);

  syncDevice();

  nrt_tensor_t* handle = NeuronCachingAllocator::findTensor(ptr);
  EXPECT_NE(handle, nullptr);
}

TEST_F(NeuronCachingAllocatorTest, FindTensorPtrReturnsSharedPtrForAllocatedPtr) {
  auto dp = allocate(1024);
  void* ptr = dp.get();
  ASSERT_NE(ptr, nullptr);

  syncDevice();

  auto tensor_ptr = NeuronCachingAllocator::findTensorPtr(ptr);
  EXPECT_NE(tensor_ptr, nullptr);

  // The shared_ptr's deleter frees the NRT slice — should not crash
  tensor_ptr.reset();
}

TEST_F(NeuronCachingAllocatorTest, FindTensorPtrReturnsNullForUnknownPtr) {
  void* bogus = reinterpret_cast<void*>(0xDEADBEEF);
  EXPECT_EQ(NeuronCachingAllocator::findTensorPtr(bogus), nullptr);
}

TEST_F(NeuronCachingAllocatorTest, GetTensorFromContextThrowsOnNull) {
  EXPECT_THROW(NeuronCachingAllocator::getTensorFromContext(nullptr), c10::Error);
}

TEST_F(NeuronCachingAllocatorTest, GetTensorFromContextReturnsHandleForAllocatedPtr) {
  auto dp = allocate(1024);
  void* ctx = dp.get_context();
  ASSERT_NE(ctx, nullptr);

  syncDevice();

  nrt_tensor_t* handle = NeuronCachingAllocator::getTensorFromContext(ctx);
  EXPECT_NE(handle, nullptr);

  // Should match findTensor for the same allocation
  nrt_tensor_t* find_handle = NeuronCachingAllocator::findTensor(dp.get());
  EXPECT_EQ(handle, find_handle);
}

TEST_F(NeuronCachingAllocatorTest, DeleterRemovesFromRegistryOnDestruction) {
  void* ptr = nullptr;
  {
    auto dp = allocate(1024);
    ptr = dp.get();
    ASSERT_NE(ptr, nullptr);

    syncDevice();

    // Should be findable while alive
    EXPECT_NE(NeuronCachingAllocator::findTensor(ptr), nullptr);
  }
  // DataPtr destructed, neuron_deleter called, registry entry removed
  // For lazy alloc, the deleter submits kDealloc; sync to let it complete
  syncDevice();

  EXPECT_EQ(NeuronCachingAllocator::findTensor(ptr), nullptr);
}

TEST_F(NeuronCachingAllocatorTest, RecordStreamOnNullPtrDoesNotCrash) {
  c10::DataPtr empty_dp(nullptr, c10::Device(c10::DeviceType::PrivateUse1, 0));
  auto stream = getNonDefaultStream();
  EXPECT_NO_THROW(NeuronCachingAllocator::recordStream(empty_dp, stream));
}

TEST_F(NeuronCachingAllocatorTest, RecordStreamOnAllocatedPtrDoesNotCrash) {
  auto dp = allocate(1024);
  auto stream = getNonDefaultStream();
  EXPECT_NO_THROW(NeuronCachingAllocator::recordStream(dp, stream));
}

TEST_F(NeuronCachingAllocatorTest, SnapshotEmptyWhenNoActiveAllocations) {
  auto segments = NeuronCachingAllocator::getSnapshotSegments();
  EXPECT_TRUE(segments.empty());
}

TEST_F(NeuronCachingAllocatorTest, SnapshotReflectsActiveAllocations) {
  auto dp1 = allocate(1024);
  auto dp2 = allocate(2048);

  syncDevice();

  auto segments = NeuronCachingAllocator::getSnapshotSegments();
  // Should have at least one segment with active blocks
  EXPECT_GE(segments.size(), 1u);

  // Total active allocation size across all segments should account for dp1 + dp2
  size_t total_allocated = 0;
  for (const auto& seg : segments) {
    total_allocated += seg.allocated_size;
  }
  // Should be at least 1024 + 2048 = 3072 (may be rounded up by allocator)
  EXPECT_GE(total_allocated, 3072u);
}

TEST_F(NeuronCachingAllocatorTest, GetSegmentPoolReturnsSameInstanceForSameDevice) {
  auto& pool1 = c10_neuron::GetSegmentPool(0);
  auto& pool2 = c10_neuron::GetSegmentPool(0);
  EXPECT_EQ(&pool1, &pool2);
}

TEST_F(NeuronCachingAllocatorTest, EmptyCacheDoesNotCrash) {
  EXPECT_NO_THROW(NeuronCachingAllocator::emptyCache());
}

TEST_F(NeuronCachingAllocatorTest, RemoveUnknownPtrDoesNotCrash) {
  void* bogus = reinterpret_cast<void*>(0xDEADBEEF);
  EXPECT_NO_THROW(NeuronCachingAllocator::remove(bogus));
}

TEST_F(NeuronCachingAllocatorTest, FullLifecycleAllocFindRecordDestroy) {
  void* ptr = nullptr;
  nrt_tensor_t* handle = nullptr;
  {
    auto dp = allocate(2048);
    ptr = dp.get();
    ASSERT_NE(ptr, nullptr);

    syncDevice();

    // findTensor returns valid handle
    handle = NeuronCachingAllocator::findTensor(ptr);
    EXPECT_NE(handle, nullptr);

    // getTensorFromContext returns same handle
    void* ctx = dp.get_context();
    EXPECT_EQ(NeuronCachingAllocator::getTensorFromContext(ctx), handle);

    // recordStream on a non-default stream
    auto other_stream = getNonDefaultStream();
    NeuronCachingAllocator::recordStream(dp, other_stream);

    // Snapshot reflects the allocation
    auto segs = NeuronCachingAllocator::getSnapshotSegments();
    EXPECT_GE(segs.size(), 1u);
  }
  // After destruction, sync to let kDealloc complete
  syncDevice();

  // findTensor should return nullptr (removed from registry)
  EXPECT_EQ(NeuronCachingAllocator::findTensor(ptr), nullptr);
}

TEST_F(NeuronCachingAllocatorTest, CachedScalarTensorIsResolvableByFindTensor) {
  int32_t val = 7;
  auto t =
      NeuronCachingAllocator::createCachedScalarTensor(&val, sizeof(val), c10::ScalarType::Int, 0);
  ASSERT_TRUE(t.has_value());
  void* ptr = t->data_ptr();
  ASSERT_NE(ptr, nullptr);

  syncDevice();

  nrt_tensor_t* handle = NeuronCachingAllocator::findTensor(ptr);
  EXPECT_NE(handle, nullptr);

  // The storage context resolves to the same handle.
  void* ctx = t->storage().data_ptr().get_context();
  EXPECT_EQ(NeuronCachingAllocator::getTensorFromContext(ctx), handle);
}

TEST_F(NeuronCachingAllocatorTest, IdenticalScalarsShareVAAndSlice) {
  int32_t val = 123;
  auto a =
      NeuronCachingAllocator::createCachedScalarTensor(&val, sizeof(val), c10::ScalarType::Int, 0);
  auto b =
      NeuronCachingAllocator::createCachedScalarTensor(&val, sizeof(val), c10::ScalarType::Int, 0);
  ASSERT_TRUE(a.has_value() && b.has_value());

  syncDevice();

  // Identical bit patterns dedup to one shared placeholder VA (the second call
  // is a cache hit that submits no new op) and resolve to the same slice.
  EXPECT_EQ(a->data_ptr(), b->data_ptr());
  EXPECT_EQ(NeuronCachingAllocator::findTensor(a->data_ptr()),
            NeuronCachingAllocator::findTensor(b->data_ptr()));
}

TEST_F(NeuronCachingAllocatorTest, ScalarEntrySurvivesTensorDestructionUntilEmptyCache) {
  int32_t val = 55;
  void* ptr = nullptr;
  {
    auto t = NeuronCachingAllocator::createCachedScalarTensor(&val, sizeof(val),
                                                              c10::ScalarType::Int, 0);
    ASSERT_TRUE(t.has_value());
    ptr = t->data_ptr();
    syncDevice();
    EXPECT_NE(NeuronCachingAllocator::findTensor(ptr), nullptr);
  }
  // The scalar entry is permanent (no-op deleter, shared across tensors), so
  // destroying the tensor must not free it.
  syncDevice();
  EXPECT_NE(NeuronCachingAllocator::findTensor(ptr), nullptr);

  // Only emptyCache tears it down.
  NeuronCachingAllocator::emptyCache();
  EXPECT_EQ(NeuronCachingAllocator::findTensor(ptr), nullptr);
}

TEST_F(NeuronCachingAllocatorTest, DistinctScalarsResolveToDistinctSlices) {
  int32_t v1 = 1, v2 = 2;
  auto a =
      NeuronCachingAllocator::createCachedScalarTensor(&v1, sizeof(v1), c10::ScalarType::Int, 0);
  auto b =
      NeuronCachingAllocator::createCachedScalarTensor(&v2, sizeof(v2), c10::ScalarType::Int, 0);
  ASSERT_TRUE(a.has_value() && b.has_value());

  syncDevice();

  EXPECT_NE(a->data_ptr(), b->data_ptr());
  EXPECT_NE(NeuronCachingAllocator::findTensor(a->data_ptr()),
            NeuronCachingAllocator::findTensor(b->data_ptr()));
}

TEST_F(NeuronCachingAllocatorTest, OversizedScalarReturnsNullopt) {
  // Larger than kSlotSize (16 bytes) — not a real scalar; must fall back.
  char buf[32] = {};
  auto t =
      NeuronCachingAllocator::createCachedScalarTensor(buf, sizeof(buf), c10::ScalarType::Float, 0);
  EXPECT_FALSE(t.has_value());
}
