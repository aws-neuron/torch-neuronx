#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tests/csrc/mocks/MockNRT.h"
#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"
#include "torch_neuronx/csrc/core/allocator/ShelvesCleanup.h"

using namespace c10_neuron;
using namespace torch_neuronx::testing;
using ::testing::_;
using ::testing::DoAll;
using ::testing::InSequence;
using ::testing::Return;
using ::testing::SetArgPointee;

class NrtTensorPoolTest : public ::testing::Test {
 protected:
  void SetUp() override { mock_session_ = std::make_unique<MockNRTSession>(); }

  void TearDown() override { mock_session_.reset(); }

  std::unique_ptr<MockNRTSession> mock_session_;
};

TEST_F(NrtTensorPoolTest, AllocateReturnsValidPointer) {
  NrtTensorPool allocator(0);

  nrt_tensor_t* dummy_tensor = reinterpret_cast<nrt_tensor_t*>(0x1000);
  void* dummy_va = reinterpret_cast<void*>(0x5000);
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_allocate(_, _, 1024, _, _))
      .WillOnce(DoAll(SetArgPointee<4>(dummy_tensor), Return(NRT_SUCCESS)));
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_get_va(dummy_tensor)).WillOnce(Return(dummy_va));

  auto [tensor, data_ptr, dealloc_flag] = allocator.Allocate(1024, 0);
  EXPECT_NE(tensor, nullptr);
  EXPECT_EQ(tensor.get(), dummy_tensor);
  EXPECT_EQ(data_ptr, dummy_va);
  EXPECT_NE(dealloc_flag, nullptr);
  EXPECT_FALSE(dealloc_flag->load());
}

TEST_F(NrtTensorPoolTest, AllocateZeroReturnsNull) {
  NrtTensorPool allocator(0);
  auto [tensor, data_ptr, dealloc_flag] = allocator.Allocate(0, 0);
  EXPECT_EQ(tensor, nullptr);
  EXPECT_EQ(data_ptr, nullptr);
  EXPECT_EQ(dealloc_flag, nullptr);
}

TEST_F(NrtTensorPoolTest, RecycleAndReusesSameSize) {
  NrtTensorPool allocator(0);

  nrt_tensor_t* dummy_tensor = reinterpret_cast<nrt_tensor_t*>(0x1000);
  void* dummy_va = reinterpret_cast<void*>(0x5000);
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_allocate(_, _, 1024, _, _))
      .WillOnce(DoAll(SetArgPointee<4>(dummy_tensor), Return(NRT_SUCCESS)));
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_get_va(dummy_tensor))
      .WillRepeatedly(Return(dummy_va));

  auto [tensor1, data_ptr1, dealloc_flag1] = allocator.Allocate(1024, 0);
  auto tensor1_copy = tensor1;

  // Recycle the former original tensor
  allocator.Recycle(std::move(tensor1), 1024, 0, data_ptr1, nullptr, dealloc_flag1);

  // Next allocation should recycle the same original tensor
  auto [tensor2, data_ptr2, dealloc_flag2] = allocator.Allocate(1024, 0);
  EXPECT_EQ(tensor2.get(), dummy_tensor);
  EXPECT_EQ(tensor2.get(), tensor1_copy.get());
  EXPECT_EQ(data_ptr2, dummy_va);
}

TEST_F(NrtTensorPoolTest, ExpiredWeakPtrSkipped) {
  NrtTensorPool allocator(0);

  nrt_tensor_t* dummy_tensor1 = reinterpret_cast<nrt_tensor_t*>(0x1000);
  nrt_tensor_t* dummy_tensor2 = reinterpret_cast<nrt_tensor_t*>(0x2000);
  void* dummy_va1 = reinterpret_cast<void*>(0x5000);
  void* dummy_va2 = reinterpret_cast<void*>(0x6000);

  {
    ::testing::InSequence seq;
    EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_allocate(_, _, 1024, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(dummy_tensor1), Return(NRT_SUCCESS)));
    EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_get_va(dummy_tensor1))
        .WillOnce(Return(dummy_va1));
    EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_free(_)).Times(1);
    EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_allocate(_, _, 1024, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(dummy_tensor2), Return(NRT_SUCCESS)));
    EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_get_va(dummy_tensor2))
        .WillOnce(Return(dummy_va2));
    EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_free(_)).Times(1);
  }

  // Allocate but deference the tensor entirely, so it gets deallocated
  {
    auto [tensor1, data_ptr1, dealloc_flag1] = allocator.Allocate(1024, 0);
    allocator.Recycle(tensor1, 1024, 0, data_ptr1, nullptr, dealloc_flag1);
  }

  // Next allocation should allocate a fresh new tensor
  auto [tensor2, data_ptr2, dealloc_flag2] = allocator.Allocate(1024, 0);
  EXPECT_EQ(tensor2.get(), dummy_tensor2);
  EXPECT_EQ(data_ptr2, dummy_va2);
}

TEST_F(NrtTensorPoolTest, DifferentSizesNotRecycled) {
  NrtTensorPool allocator(0);

  nrt_tensor_t* dummy_tensor1 = reinterpret_cast<nrt_tensor_t*>(0x1000);
  nrt_tensor_t* dummy_tensor2 = reinterpret_cast<nrt_tensor_t*>(0x2000);
  void* dummy_va1 = reinterpret_cast<void*>(0x5000);
  void* dummy_va2 = reinterpret_cast<void*>(0x6000);
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_allocate(_, _, 1024, _, _))
      .WillOnce(DoAll(SetArgPointee<4>(dummy_tensor1), Return(NRT_SUCCESS)));
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_get_va(dummy_tensor1))
      .WillOnce(Return(dummy_va1));
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_allocate(_, _, 2048, _, _))
      .WillOnce(DoAll(SetArgPointee<4>(dummy_tensor2), Return(NRT_SUCCESS)));
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_get_va(dummy_tensor2))
      .WillOnce(Return(dummy_va2));

  auto [tensor1, data_ptr1, dealloc_flag1] = allocator.Allocate(1024, 0);
  auto tensor1_copy = tensor1;
  allocator.Recycle(std::move(tensor1), 1024, 0, data_ptr1, nullptr, dealloc_flag1);

  // Different size, so allocate a new tensor
  auto [tensor2, data_ptr2, dealloc_flag2] = allocator.Allocate(2048, 0);
  EXPECT_EQ(tensor2.get(), dummy_tensor2);
}

TEST_F(NrtTensorPoolTest, FreelistSizeTracking) {
  NrtTensorPool allocator(0);

  nrt_tensor_t* dummy_tensor = reinterpret_cast<nrt_tensor_t*>(0x1000);
  void* dummy_va = reinterpret_cast<void*>(0x5000);
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_allocate(_, _, _, _, _))
      .WillRepeatedly(DoAll(SetArgPointee<4>(dummy_tensor), Return(NRT_SUCCESS)));
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_get_va(_)).WillRepeatedly(Return(dummy_va));

  EXPECT_EQ(allocator.GetFreelistSize(), 0);

  auto [tensor, data_ptr, dealloc_flag] = allocator.Allocate(1024, 0);
  auto tensor_copy = tensor;
  allocator.Recycle(std::move(tensor), 1024, 0, data_ptr, nullptr, dealloc_flag);

  EXPECT_EQ(allocator.GetFreelistSize(), 1);

  allocator.Clear();
  EXPECT_EQ(allocator.GetFreelistSize(), 0);
}

TEST_F(NrtTensorPoolTest, LIFOOrder) {
  NrtTensorPool allocator(0);

  nrt_tensor_t* dummy1 = reinterpret_cast<nrt_tensor_t*>(0x1000);
  nrt_tensor_t* dummy2 = reinterpret_cast<nrt_tensor_t*>(0x2000);
  void* dummy_va1 = reinterpret_cast<void*>(0x5000);
  void* dummy_va2 = reinterpret_cast<void*>(0x6000);
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_allocate(_, _, 1024, _, _))
      .WillOnce(DoAll(SetArgPointee<4>(dummy1), Return(NRT_SUCCESS)))
      .WillOnce(DoAll(SetArgPointee<4>(dummy2), Return(NRT_SUCCESS)));
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_get_va(dummy1)).WillRepeatedly(Return(dummy_va1));
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_get_va(dummy2)).WillRepeatedly(Return(dummy_va2));

  auto [tensor1, data_ptr1, dealloc_flag1] = allocator.Allocate(1024, 0);
  auto [tensor2, data_ptr2, dealloc_flag2] = allocator.Allocate(1024, 0);
  auto copy1 = tensor1;
  auto copy2 = tensor2;

  // Recycle in order: tensor1, then tensor2
  allocator.Recycle(std::move(tensor1), 1024, 0, data_ptr1, nullptr, dealloc_flag1);
  allocator.Recycle(std::move(tensor2), 1024, 0, data_ptr2, nullptr, dealloc_flag2);

  // LIFO: should get tensor2 first (most recently recycled)
  auto [recycled, recycled_ptr, recycled_flag] = allocator.Allocate(1024, 0);
  EXPECT_EQ(recycled.get(), dummy2);
}

TEST_F(NrtTensorPoolTest, PruneExpiredEntries) {
  NrtTensorPool allocator(0);

  nrt_tensor_t* dummy1 = reinterpret_cast<nrt_tensor_t*>(0x1000);
  nrt_tensor_t* dummy2 = reinterpret_cast<nrt_tensor_t*>(0x2000);
  void* dummy_va1 = reinterpret_cast<void*>(0x5000);
  void* dummy_va2 = reinterpret_cast<void*>(0x6000);
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_allocate(_, _, 1024, _, _))
      .WillOnce(DoAll(SetArgPointee<4>(dummy1), Return(NRT_SUCCESS)))
      .WillOnce(DoAll(SetArgPointee<4>(dummy2), Return(NRT_SUCCESS)));
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_get_va(dummy1)).WillOnce(Return(dummy_va1));
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_get_va(dummy2)).WillOnce(Return(dummy_va2));
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_free(_)).Times(2);

  {
    auto [tensor1, data_ptr1, dealloc_flag1] = allocator.Allocate(1024, 0);
    auto [tensor2, data_ptr2, dealloc_flag2] = allocator.Allocate(1024, 0);
    allocator.Recycle(tensor1, 1024, 0, data_ptr1, nullptr, dealloc_flag1);
    allocator.Recycle(tensor2, 1024, 0, data_ptr2, nullptr, dealloc_flag2);
    EXPECT_EQ(allocator.GetFreelistSize(), 2);
  }

  // Prune should remove both expired entries
  size_t pruned = allocator.PruneExpiredEntries();
  EXPECT_EQ(pruned, 2);
  EXPECT_EQ(allocator.GetFreelistSize(), 0);
}

TEST_F(NrtTensorPoolTest, AllocateOOMRecovery) {
  NrtTensorPool allocator(0);

  nrt_tensor_t* dummy1 = reinterpret_cast<nrt_tensor_t*>(0x1000);
  nrt_tensor_t* dummy2 = reinterpret_cast<nrt_tensor_t*>(0x2000);
  void* dummy_va1 = reinterpret_cast<void*>(0x5000);
  void* dummy_va2 = reinterpret_cast<void*>(0x6000);

  // First allocation succeeds
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_allocate(_, _, 1024, _, _))
      .WillOnce(DoAll(SetArgPointee<4>(dummy1), Return(NRT_SUCCESS)));
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_get_va(dummy1)).WillOnce(Return(dummy_va1));
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_free(_)).Times(2);  // dummy1 + dummy2

  {
    auto [tensor1, data_ptr1, dealloc_flag1] = allocator.Allocate(1024, 0);
    allocator.Recycle(tensor1, 1024, 0, data_ptr1, nullptr, dealloc_flag1);
  }

  // Allocate: first attempt fails with OOM, prune clears expired, retry succeeds
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_allocate(_, _, 2048, _, _))
      .WillOnce(Return(NRT_RESOURCE))                                   // First attempt: OOM
      .WillOnce(DoAll(SetArgPointee<4>(dummy2), Return(NRT_SUCCESS)));  // Retry succeeds
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_get_va(dummy2)).WillOnce(Return(dummy_va2));

  auto [tensor2, data_ptr2, dealloc_flag2] = allocator.Allocate(2048, 0);
  EXPECT_EQ(tensor2.get(), dummy2);
  // Freelist should be empty after we prune expired entries
  EXPECT_EQ(allocator.GetFreelistSize(), 0);
}

TEST_F(NrtTensorPoolTest, OOMRetryTriggersShelvesCleanup) {
  NrtTensorPool allocator(0);

  nrt_tensor_t* dummy = reinterpret_cast<nrt_tensor_t*>(0x1000);

  // First attempt fails with OOM, retry succeeds
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_allocate(_, _, 1024, _, _))
      .WillOnce(Return(NRT_RESOURCE))
      .WillOnce(DoAll(SetArgPointee<4>(dummy), Return(NRT_SUCCESS)));
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_free(_)).Times(1);

  // Track cleanup trigger count
  size_t before = torch_neuronx::distributed::ShelvesCleanupRegistry::instance().getTriggerCount();

  auto tensor = allocator.Allocate(1024, 0);
  EXPECT_NE(tensor.tensor_ptr, nullptr);

  // Verify triggerShelvesCleanup was called once before alloc
  size_t after = torch_neuronx::distributed::ShelvesCleanupRegistry::instance().getTriggerCount();
  EXPECT_GE(after, before + 1);
}

TEST_F(NrtTensorPoolTest, AllocTriggersShelvesCleanup) {
  NrtTensorPool allocator(0);

  nrt_tensor_t* dummy = reinterpret_cast<nrt_tensor_t*>(0x1000);

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_allocate(_, _, 1024, _, _))
      .WillOnce(DoAll(SetArgPointee<4>(dummy), Return(NRT_SUCCESS)));
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_free(_)).Times(1);

  size_t before = torch_neuronx::distributed::ShelvesCleanupRegistry::instance().getTriggerCount();

  auto tensor = allocator.Allocate(1024, 0);
  EXPECT_NE(tensor.tensor_ptr, nullptr);

  // Cleanup is NOT called on normal allocation (only on OOM retry)
  size_t after = torch_neuronx::distributed::ShelvesCleanupRegistry::instance().getTriggerCount();
  EXPECT_EQ(after, before);
}

TEST_F(NrtTensorPoolTest, AllocateThrowsOnPersistentOOM) {
  NrtTensorPool allocator(0);

  // Both attempts fail with OOM
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_allocate(_, _, 1024, _, _))
      .WillOnce(Return(NRT_RESOURCE))
      .WillOnce(Return(NRT_RESOURCE));

  EXPECT_THROW(allocator.Allocate(1024, 0), std::runtime_error);
}

TEST_F(NrtTensorPoolTest, RecycledTensorSharedBetweenOldAndNewOwner) {
  NrtTensorPool allocator(0);

  nrt_tensor_t* dummy = reinterpret_cast<nrt_tensor_t*>(0x1000);
  void* dummy_va = reinterpret_cast<void*>(0x5000);
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_allocate(_, _, 1024, _, _))
      .WillOnce(DoAll(SetArgPointee<4>(dummy), Return(NRT_SUCCESS)));
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_get_va(dummy)).WillRepeatedly(Return(dummy_va));
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_free(_)).Times(1);

  auto [op_holds_tensor, data_ptr, dealloc_flag] =
      allocator.Allocate(1024, 0);  // Simulates operation holding ref

  // Recycled, but a reference is still maintained
  allocator.Recycle(op_holds_tensor, 1024, 0, data_ptr, nullptr, dealloc_flag);

  // New allocation gets the same recycled tensor
  auto [new_owner, new_data_ptr, new_dealloc_flag] = allocator.Allocate(1024, 0);
  EXPECT_EQ(new_owner.get(), dummy);

  // Both point to same tensor
  EXPECT_EQ(op_holds_tensor.get(), new_owner.get());
  EXPECT_EQ(op_holds_tensor.use_count(), 2);

  op_holds_tensor.reset();
  EXPECT_EQ(new_owner.use_count(), 1);

  // Tensor still valid for new owner
  EXPECT_EQ(new_owner.get(), dummy);
}

TEST_F(NrtTensorPoolTest, RecycledTensorFreedWhenNoReferencesRemain) {
  NrtTensorPool allocator(0);

  nrt_tensor_t* dummy1 = reinterpret_cast<nrt_tensor_t*>(0x1000);
  nrt_tensor_t* dummy2 = reinterpret_cast<nrt_tensor_t*>(0x2000);
  void* dummy_va1 = reinterpret_cast<void*>(0x5000);
  void* dummy_va2 = reinterpret_cast<void*>(0x6000);

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_allocate(_, _, 1024, _, _))
      .WillOnce(DoAll(SetArgPointee<4>(dummy1), Return(NRT_SUCCESS)))
      .WillOnce(DoAll(SetArgPointee<4>(dummy2), Return(NRT_SUCCESS)));
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_get_va(dummy1)).WillOnce(Return(dummy_va1));
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_get_va(dummy2)).WillOnce(Return(dummy_va2));
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_free(_)).Times(2);

  {
    auto [tensor, data_ptr, dealloc_flag] = allocator.Allocate(1024, 0);
    EXPECT_EQ(tensor.use_count(), 1);
    allocator.Recycle(tensor, 1024, 0, data_ptr, nullptr, dealloc_flag);
    EXPECT_EQ(tensor.use_count(), 1);  // Recycle doesn't increase refcount
  }

  // Next allocation should not be recycled
  auto [tensor2, data_ptr2, dealloc_flag2] = allocator.Allocate(1024, 0);
  EXPECT_EQ(tensor2.get(), dummy2);  // Different pointer = fresh allocation
}

TEST_F(NrtTensorPoolTest, StreamAwareRecycle) {
  NrtTensorPool allocator(0);

  nrt_tensor_t* dummy1 = reinterpret_cast<nrt_tensor_t*>(0x1000);
  nrt_tensor_t* dummy2 = reinterpret_cast<nrt_tensor_t*>(0x2000);
  void* dummy_va1 = reinterpret_cast<void*>(0x5000);
  void* dummy_va2 = reinterpret_cast<void*>(0x6000);

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_allocate(_, _, 1024, _, _))
      .WillOnce(DoAll(SetArgPointee<4>(dummy1), Return(NRT_SUCCESS)))
      .WillOnce(DoAll(SetArgPointee<4>(dummy2), Return(NRT_SUCCESS)));
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_get_va(dummy1)).WillRepeatedly(Return(dummy_va1));
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_get_va(dummy2)).WillRepeatedly(Return(dummy_va2));
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_free(_)).Times(2);

  c10::StreamId stream1 = 1;
  c10::StreamId stream2 = 2;

  // Allocate on stream 1
  auto [tensor1, data_ptr1, dealloc_flag1] = allocator.Allocate(1024, stream1);
  auto tensor1_copy = tensor1;

  // Recycle within stream 1
  allocator.Recycle(std::move(tensor1), 1024, stream1, data_ptr1, nullptr, dealloc_flag1);

  // Allocate on stream 2
  auto [tensor2, data_ptr2, dealloc_flag2] = allocator.Allocate(1024, stream2);
  EXPECT_EQ(tensor2.get(), dummy2);  // Fresh allocation, not recycled

  // Ensure is recycled strictly within stream 1 (rather than 2)
  auto [tensor3, data_ptr3, dealloc_flag3] = allocator.Allocate(1024, stream1);
  EXPECT_EQ(tensor3.get(), dummy1);
}

TEST_F(NrtTensorPoolTest, MultipleStreamsIndependent) {
  NrtTensorPool allocator(0);

  nrt_tensor_t* dummy1 = reinterpret_cast<nrt_tensor_t*>(0x1000);
  nrt_tensor_t* dummy2 = reinterpret_cast<nrt_tensor_t*>(0x2000);
  void* dummy_va1 = reinterpret_cast<void*>(0x5000);
  void* dummy_va2 = reinterpret_cast<void*>(0x6000);

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_allocate(_, _, 1024, _, _))
      .WillOnce(DoAll(SetArgPointee<4>(dummy1), Return(NRT_SUCCESS)))
      .WillOnce(DoAll(SetArgPointee<4>(dummy2), Return(NRT_SUCCESS)));
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_get_va(dummy1)).WillRepeatedly(Return(dummy_va1));
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_get_va(dummy2)).WillRepeatedly(Return(dummy_va2));
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_tensor_free(_)).Times(2);

  c10::StreamId stream1 = 1;
  c10::StreamId stream2 = 2;

  // Allocate and recycle on both streams
  auto [t1, t1_data_ptr, t1_dealloc_flag] = allocator.Allocate(1024, stream1);
  auto t1_copy = t1;
  allocator.Recycle(std::move(t1), 1024, stream1, t1_data_ptr, nullptr, t1_dealloc_flag);

  auto [t2, t2_data_ptr, t2_dealloc_flag] = allocator.Allocate(1024, stream2);
  auto t2_copy = t2;
  allocator.Recycle(std::move(t2), 1024, stream2, t2_data_ptr, nullptr, t2_dealloc_flag);

  // Each stream should recycle its own tensor
  auto [t1_recycled, t1_recycled_ptr, t1_recycled_flag] = allocator.Allocate(1024, stream1);
  EXPECT_EQ(t1_recycled.get(), dummy1);

  auto [t2_recycled, t2_recycled_ptr, t2_recycled_flag] = allocator.Allocate(1024, stream2);
  EXPECT_EQ(t2_recycled.get(), dummy2);
}
