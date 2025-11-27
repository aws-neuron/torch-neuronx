#include <c10/core/DeviceType.h>
#include <c10/core/StorageImpl.h>
#include <gtest/gtest.h>

#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"

// Forward declare the registration function
namespace torch_neuronx {
void register_neuron_device();
}

class StorageImplTest : public ::testing::Test {
 protected:
  void SetUp() override { torch_neuronx::register_neuron_device(); }
};

TEST_F(StorageImplTest, CustomStorageCreatorRegistered) {
  // Test that our SetStorageImplCreate registration worked
  auto creator = c10::GetStorageImplCreate(c10::DeviceType::PrivateUse1);
  ASSERT_NE(creator, nullptr) << "Custom storage creator should be registered for PrivateUse1";
}

TEST_F(StorageImplTest, AllocatorRegistered) {
  // Test that allocator is registered for PrivateUse1
  auto allocator = c10::GetAllocator(c10::DeviceType::PrivateUse1);
  ASSERT_NE(allocator, nullptr) << "Allocator should be registered for PrivateUse1";
}
