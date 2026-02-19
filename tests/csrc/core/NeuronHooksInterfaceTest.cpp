#include <ATen/ATen.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Storage.h>
#include <c10/core/StorageImpl.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <chrono>
#include <cstdlib>
#include <future>
#include <thread>

#include "tests/csrc/mocks/MockNRT.h"
#include "tests/csrc/mocks/MockNeuronDevice.h"
#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/NeuronHooksInterface.h"
#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"
#include "torch_neuronx/csrc/core/utils/NeuronResourceManager.h"

// Forward declare the registration function
namespace torch_neuronx {
void register_neuron_device();
}

using namespace torch_neuronx::testing;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

class NeuronHooksInterfaceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize mock session first to avoid neuron runtime not found error
    mock_session_ = std::make_unique<MockNRTSession>();
    mock_nrt_ = torch_neuronx::testing::MockNRT::GetInstance();

    // Set up basic mock expectations for NRT initialization
    EXPECT_CALL(*mock_nrt_, nrt_init(_, _, _)).WillRepeatedly(Return(NRT_SUCCESS));
    EXPECT_CALL(*mock_nrt_, nrt_get_total_vnc_count(_))
        .WillRepeatedly(DoAll(SetArgPointee<0>(1), Return(NRT_SUCCESS)));
    EXPECT_CALL(*mock_nrt_, nrt_get_visible_vnc_count(_))
        .WillRepeatedly(DoAll(SetArgPointee<0>(1), Return(NRT_SUCCESS)));
    EXPECT_CALL(*mock_nrt_, nrt_close()).WillRepeatedly(Return());

    // Set up mock expectations for tensor allocation
    static std::atomic<uintptr_t> tensor_counter{0x12345678};
    static std::atomic<uintptr_t> va_counter{0x87654321};

    EXPECT_CALL(*mock_nrt_, nrt_tensor_allocate(_, _, _, _, _))
        .WillRepeatedly([](nrt_tensor_placement_t placement, int32_t logical_nc_id, size_t size,
                           const char* name, nrt_tensor_t** tensor) {
          // Create unique fake tensor pointers
          *tensor = reinterpret_cast<nrt_tensor_t*>(tensor_counter.fetch_add(0x1000));
          return NRT_SUCCESS;
        });

    // Set up mock expectations for tensor operations
    EXPECT_CALL(*mock_nrt_, nrt_tensor_free(_)).WillRepeatedly(Return());
    EXPECT_CALL(*mock_nrt_, nrt_tensor_get_size(_)).WillRepeatedly(Return(1024));
    EXPECT_CALL(*mock_nrt_, nrt_tensor_get_va(_))
        .WillRepeatedly(::testing::Invoke([](const nrt_tensor* tensor) -> void* {
          // Return unique VA for each call
          return reinterpret_cast<void*>(va_counter.fetch_add(0x1000));
        }));

    // Set up mock expectations for tensor copy operations
    EXPECT_CALL(*mock_nrt_, nrt_tensor_copy(_, _, _, _, _)).WillRepeatedly(Return(NRT_SUCCESS));
    EXPECT_CALL(*mock_nrt_, nrt_tensor_write(_, _, _, _)).WillRepeatedly(Return(NRT_SUCCESS));
    EXPECT_CALL(*mock_nrt_, nrt_tensor_read(_, _, _, _)).WillRepeatedly(Return(NRT_SUCCESS));

    // Register neuron device to ensure proper initialization
    torch_neuronx::register_neuron_device();

    at::neuron::InitializeStreamPools();
    at::neuron::NeuronResourceManager::Instance();
  }

  void TearDown() override {
    // Synchronize all devices to complete pending operations
    for (int device = 0; device < c10_neuron::testing::device_count(); ++device) {
      at::neuron::synchronize(device);
    }
    c10_neuron::NeuronCachingAllocator::emptyCache();
    ::testing::Mock::VerifyAndClearExpectations(mock_nrt_);
    mock_session_.reset();
  }

  std::unique_ptr<MockNRTSession> mock_session_;
  torch_neuronx::testing::MockNRT* mock_nrt_;
};

TEST_F(NeuronHooksInterfaceTest, ResizeWithNoswap) {
  // Test basic resize using noswap

  // Create storage directly using allocator
  auto allocator = c10::GetAllocator(c10::DeviceType::PrivateUse1);
  auto data_ptr = allocator->allocate(800);

  auto storage = c10::Storage(c10::Storage::use_byte_size_t{}, 800, std::move(data_ptr), allocator,
                              true  // resizable = true
  );

  size_t original_bytes = storage.nbytes();
  EXPECT_EQ(original_bytes, 800);

  // Get the hooks interface
  auto hooks = torch_neuronx::get_neuron_hooks();
  ASSERT_NE(hooks, nullptr);

  // Test resize to larger size
  size_t new_bytes = 4800;
  hooks->resizePrivateUse1Bytes(storage, new_bytes);

  // Verify the storage was resized correctly
  EXPECT_EQ(storage.nbytes(), new_bytes);

  // Test resize to smaller size
  size_t smaller_bytes = 200;
  hooks->resizePrivateUse1Bytes(storage, smaller_bytes);

  // Verify the storage was resized correctly
  EXPECT_EQ(storage.nbytes(), smaller_bytes);
}

TEST_F(NeuronHooksInterfaceTest, ResizePreservesStorageIdentity) {
  // Test that noswap approach preserves storage identity (doesn't replace storage address)
  auto device = at::Device(c10::DeviceType::PrivateUse1, 0);

  // Create a tensor on neuron device
  auto options = torch::TensorOptions().device(device).dtype(torch::kFloat32);
  auto tensor = torch::empty({10, 20}, options);

  // Get the storage and remember its identity
  auto storage = tensor.storage();
  c10::StorageImpl* storage_impl = storage.unsafeGetStorageImpl();
  void* original_storage_impl_ptr = static_cast<void*>(storage_impl);

  // Get the hooks interface
  auto hooks = torch_neuronx::get_neuron_hooks();

  // Resize the storage
  size_t new_bytes = 40 * 30 * sizeof(float);
  hooks->resizePrivateUse1Bytes(storage, new_bytes);

  // Verify storage identity is preserved (noswap behavior)
  c10::StorageImpl* new_storage_impl = storage.unsafeGetStorageImpl();
  void* new_storage_impl_ptr = static_cast<void*>(new_storage_impl);

  EXPECT_EQ(original_storage_impl_ptr, new_storage_impl_ptr)
      << "Storage implementation pointer should remain the same (noswap)";

  // Verify the storage was resized correctly
  EXPECT_EQ(storage.nbytes(), new_bytes);
}

TEST_F(NeuronHooksInterfaceTest, ConcurrentResizeOperations) {
  // Test concurrent resize operations
  auto device = at::Device(c10::DeviceType::PrivateUse1, 0);

  // Create a tensor on neuron device
  auto options = torch::TensorOptions().device(device).dtype(torch::kFloat32);
  auto tensor = torch::empty({50, 50}, options);

  // Get the storage and remember its identity
  auto storage = tensor.storage();
  c10::StorageImpl* original_storage_impl = storage.unsafeGetStorageImpl();
  void* original_storage_ptr = static_cast<void*>(original_storage_impl);

  auto hooks = torch_neuronx::get_neuron_hooks();

  const int num_concurrent_ops = 4;
  std::vector<std::future<bool>> futures;

  // Launch concurrent resize operations
  for (int i = 0; i < num_concurrent_ops; ++i) {
    futures.push_back(std::async(std::launch::async, [&, i]() -> bool {
      try {
        // Each async operation performs different resize sequence
        std::vector<size_t> resize_sequence;
        switch (i % 4) {
          case 0:
            resize_sequence = {100 * 100 * sizeof(float), 25 * 25 * sizeof(float)};
            break;
          case 1:
            resize_sequence = {75 * 75 * sizeof(float), 150 * 150 * sizeof(float)};
            break;
          case 2:
            resize_sequence = {200 * 200 * sizeof(float), 50 * 50 * sizeof(float)};
            break;
          case 3:
            resize_sequence = {120 * 120 * sizeof(float), 80 * 80 * sizeof(float)};
            break;
        }

        for (size_t new_bytes : resize_sequence) {
          hooks->resizePrivateUse1Bytes(storage, new_bytes);

          // Verify storage identity is preserved
          c10::StorageImpl* current_storage_impl = storage.unsafeGetStorageImpl();
          void* current_storage_ptr = static_cast<void*>(current_storage_impl);

          if (original_storage_ptr != current_storage_ptr) {
            return false;  // Storage identity not preserved
          }
        }

        return true;  // All operations succeeded with identity preserved

      } catch (...) {
        return false;  // Exception occurred
      }
    }));
  }

  // Wait for all concurrent operations and collect results
  std::vector<bool> results;
  for (auto& future : futures) {
    results.push_back(future.get());
  }

  // Verify all concurrent operations succeeded
  for (int i = 0; i < num_concurrent_ops; ++i) {
    EXPECT_TRUE(results[i]) << "Concurrent resize operation " << i
                            << " failed - storage identity not preserved";
  }

  // Final verification that storage identity is still preserved
  c10::StorageImpl* final_storage_impl = storage.unsafeGetStorageImpl();
  void* final_storage_ptr = static_cast<void*>(final_storage_impl);

  EXPECT_EQ(original_storage_ptr, final_storage_ptr)
      << "Storage identity should be preserved after all concurrent operations";
}

TEST_F(NeuronHooksInterfaceTest, ResizeZeroBytes) {
  // Test edge case: resize to zero bytes
  auto device = at::Device(c10::DeviceType::PrivateUse1, 0);

  auto options = torch::TensorOptions().device(device).dtype(torch::kFloat32);
  auto tensor = torch::empty({10, 20}, options);

  auto storage = tensor.storage();
  auto hooks = torch_neuronx::get_neuron_hooks();

  // Resize to zero bytes
  EXPECT_NO_THROW(hooks->resizePrivateUse1Bytes(storage, 0));
  EXPECT_EQ(storage.nbytes(), 0);
}

TEST_F(NeuronHooksInterfaceTest, ResizeToSameSize) {
  // Test resize to the same size (should be no-op)
  auto device = at::Device(c10::DeviceType::PrivateUse1, 0);

  auto options = torch::TensorOptions().device(device).dtype(torch::kFloat32);
  auto tensor = torch::empty({10, 20}, options);

  auto storage = tensor.storage();
  size_t original_bytes = storage.nbytes();
  c10::StorageImpl* original_storage_impl = storage.unsafeGetStorageImpl();

  auto hooks = torch_neuronx::get_neuron_hooks();

  // Resize to same size
  hooks->resizePrivateUse1Bytes(storage, original_bytes);

  // Verify nothing changed
  EXPECT_EQ(storage.nbytes(), original_bytes);

  c10::StorageImpl* post_resize_storage_impl = storage.unsafeGetStorageImpl();
  EXPECT_EQ(static_cast<void*>(original_storage_impl),
            static_cast<void*>(post_resize_storage_impl));
}

TEST_F(NeuronHooksInterfaceTest, ResizeNonResizableStorageFails) {
  // Test that resize fails on non-resizable storage

  // Create a non-resizable storage (if possible in this context)
  auto allocator = c10::GetAllocator(c10::DeviceType::PrivateUse1);
  auto data_ptr = allocator->allocate(1000);

  auto storage = c10::Storage(c10::Storage::use_byte_size_t{}, 1000, std::move(data_ptr), allocator,
                              false  // resizable = false
  );

  auto hooks = torch_neuronx::get_neuron_hooks();

  // This should throw an error
  EXPECT_THROW(hooks->resizePrivateUse1Bytes(storage, 2000), c10::Error);
}

TEST_F(NeuronHooksInterfaceTest, MultipleSequentialResizes) {
  // Test multiple sequential resizes to ensure noswap stability
  auto device = at::Device(c10::DeviceType::PrivateUse1, 0);

  auto options = torch::TensorOptions().device(device).dtype(torch::kFloat32);
  auto tensor = torch::empty({10, 10}, options);

  auto storage = tensor.storage();
  c10::StorageImpl* original_storage_impl = storage.unsafeGetStorageImpl();
  void* original_storage_ptr = static_cast<void*>(original_storage_impl);

  auto hooks = torch_neuronx::get_neuron_hooks();

  // Perform multiple resizes
  std::vector<size_t> sizes = {20 * 20 * sizeof(float), 50 * 50 * sizeof(float),
                               5 * 5 * sizeof(float), 100 * 100 * sizeof(float), 1 * sizeof(float)};

  for (size_t new_size : sizes) {
    hooks->resizePrivateUse1Bytes(storage, new_size);

    // Verify storage identity is always preserved
    c10::StorageImpl* current_storage_impl = storage.unsafeGetStorageImpl();
    void* current_storage_ptr = static_cast<void*>(current_storage_impl);

    EXPECT_EQ(original_storage_ptr, current_storage_ptr)
        << "Storage identity should be preserved across all resizes";

    EXPECT_EQ(storage.nbytes(), new_size) << "Storage size should match requested size";
  }
}
