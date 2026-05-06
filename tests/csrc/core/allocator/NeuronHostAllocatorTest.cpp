#include <gtest/gtest.h>

#include "tests/csrc/mocks/MockNRT.h"
#include "tests/csrc/utils/TestUtils.h"
#include "torch_neuronx/csrc/c10/neuron/NeuronEvent.h"
#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/allocator/NeuronHostAllocator.h"

class NeuronHostAllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_session_ = std::make_unique<torch_neuronx::testing::MockNRTSession>();
  }
  void TearDown() override { mock_session_.reset(); }
  std::unique_ptr<torch_neuronx::testing::MockNRTSession> mock_session_;
};

TEST_F(NeuronHostAllocatorTest, AllocateHostMemoryReturnsValidPointer) {
  torch_neuronx::NeuronHostAllocatorImpl impl;
  void* ptr = nullptr;
  impl.allocate_host_memory(1024, &ptr);
  EXPECT_NE(ptr, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0);
  free(ptr);
}

TEST_F(NeuronHostAllocatorTest, AllocateHostMemoryZeroSize) {
  torch_neuronx::NeuronHostAllocatorImpl impl;
  void* ptr = reinterpret_cast<void*>(0xDEAD);
  impl.allocate_host_memory(0, &ptr);
  EXPECT_EQ(ptr, nullptr);
}

TEST_F(NeuronHostAllocatorTest, FreeBlockReleasesMemorySanity) {
  auto* alloc = torch_neuronx::GetNeuronHostAllocator();
  {
    auto data_ptr = alloc->allocate(512);
  }
}

TEST_F(NeuronHostAllocatorTest, QueryEventDelegatesToNeuronEvent) {
  torch_neuronx::NeuronHostAllocatorImpl impl;
  at::neuron::NeuronEvent event;
  // Unrecorded event: query() returns !is_recorded() || is_completed()
  bool result = impl.query_event(event);
  EXPECT_TRUE(result);
}

TEST_F(NeuronHostAllocatorTest, GetNeuronHostAllocatorReturnsNonNull) {
  auto* alloc = torch_neuronx::GetNeuronHostAllocator();
  EXPECT_NE(alloc, nullptr);
}

TEST_F(NeuronHostAllocatorTest, GetNeuronHostAllocatorAsHostAllocatorReturnsNonNull) {
  auto* alloc = torch_neuronx::GetNeuronHostAllocatorAsHostAllocator();
  EXPECT_NE(alloc, nullptr);
}

TEST_F(NeuronHostAllocatorTest, AllocatorAllocateReturnsValidDataPtr) {
  auto* alloc = torch_neuronx::GetNeuronHostAllocator();
  auto data_ptr = alloc->allocate(1024);
  EXPECT_NE(data_ptr.get(), nullptr);
  EXPECT_EQ(data_ptr.device(), c10::Device(c10::DeviceType::CPU));
}

TEST_F(NeuronHostAllocatorTest, AllocatorAllocateZeroReturnsNull) {
  auto* alloc = torch_neuronx::GetNeuronHostAllocator();
  auto data_ptr = alloc->allocate(0);
  EXPECT_EQ(data_ptr.get(), nullptr);
}

TEST_F(NeuronHostAllocatorTest, AllocatorRecyclesMemory) {
  auto* alloc = torch_neuronx::GetNeuronHostAllocator();
  void* first_ptr = nullptr;
  {
    auto data_ptr = alloc->allocate(1024);
    first_ptr = data_ptr.get();
  }
  auto data_ptr2 = alloc->allocate(1024);
  EXPECT_EQ(data_ptr2.get(), first_ptr);
}

TEST_F(NeuronHostAllocatorTest, RecordEventReturnsTrueForKnownPtr) {
  auto* host_alloc = torch_neuronx::GetNeuronHostAllocatorAsHostAllocator();
  auto data_ptr = torch_neuronx::GetNeuronHostAllocator()->allocate(2048);
  c10::Stream stream(c10::Stream::UNSAFE, c10::Device(c10::DeviceType::PrivateUse1, 0), 1);
  bool recorded = host_alloc->record_event(data_ptr.get(), data_ptr.get_context(), stream);
  EXPECT_TRUE(recorded);
}

TEST_F(NeuronHostAllocatorTest, RecordEventReturnsFalseForUnknownPtr) {
  auto* host_alloc = torch_neuronx::GetNeuronHostAllocatorAsHostAllocator();
  void* random_ptr = reinterpret_cast<void*>(0x12345678);
  c10::Stream stream(c10::Stream::UNSAFE, c10::Device(c10::DeviceType::PrivateUse1, 0), 1);
  bool recorded = host_alloc->record_event(random_ptr, nullptr, stream);
  EXPECT_FALSE(recorded);
}

// ==================== isPinnedPtr / IsNeuronPinnedPtr tests ====================

TEST_F(NeuronHostAllocatorTest, IsPinnedPtrReturnsTrueForAllocatedPointer) {
  torch_neuronx::NeuronHostAllocatorImpl impl;
  void* ptr = nullptr;
  impl.allocate_host_memory(1024, &ptr);
  ASSERT_NE(ptr, nullptr);
  EXPECT_TRUE(impl.is_pinned_ptr(ptr));
  free(ptr);
}

TEST_F(NeuronHostAllocatorTest, IsPinnedPtrReturnsFalseForUnknownPointer) {
  torch_neuronx::NeuronHostAllocatorImpl impl;
  void* random_ptr = reinterpret_cast<void*>(0xDEADBEEF);
  EXPECT_FALSE(impl.is_pinned_ptr(random_ptr));
}

TEST_F(NeuronHostAllocatorTest, IsPinnedPtrZeroSizeAllocationDoesNotTrackNullptr) {
  torch_neuronx::NeuronHostAllocatorImpl impl;
  void* ptr = nullptr;
  impl.allocate_host_memory(0, &ptr);
  EXPECT_EQ(ptr, nullptr);
  EXPECT_FALSE(impl.is_pinned_ptr(nullptr));
}

TEST_F(NeuronHostAllocatorTest, IsPinnedPtrTracksMultipleAllocations) {
  torch_neuronx::NeuronHostAllocatorImpl impl;
  void* ptr1 = nullptr;
  void* ptr2 = nullptr;
  void* ptr3 = nullptr;
  impl.allocate_host_memory(64, &ptr1);
  impl.allocate_host_memory(128, &ptr2);
  impl.allocate_host_memory(256, &ptr3);

  EXPECT_TRUE(impl.is_pinned_ptr(ptr1));
  EXPECT_TRUE(impl.is_pinned_ptr(ptr2));
  EXPECT_TRUE(impl.is_pinned_ptr(ptr3));

  free(ptr1);
  free(ptr2);
  free(ptr3);
}

TEST_F(NeuronHostAllocatorTest, IsNeuronPinnedPtrWorksViaGlobalAllocator) {
  auto* alloc = torch_neuronx::GetNeuronHostAllocator();
  auto data_ptr = alloc->allocate(2048);
  void* ptr = data_ptr.get();
  ASSERT_NE(ptr, nullptr);
  EXPECT_TRUE(torch_neuronx::IsNeuronPinnedPtr(ptr));
}

TEST_F(NeuronHostAllocatorTest, IsNeuronPinnedPtrReturnsFalseForArbitraryPointer) {
  void* random_ptr = reinterpret_cast<void*>(0xCAFEBABE);
  EXPECT_FALSE(torch_neuronx::IsNeuronPinnedPtr(random_ptr));
}

TEST_F(NeuronHostAllocatorTest, IsNeuronPinnedPtrReturnsFalseForNullptr) {
  EXPECT_FALSE(torch_neuronx::IsNeuronPinnedPtr(nullptr));
}
