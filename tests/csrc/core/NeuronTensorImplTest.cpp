#include <c10/core/DispatchKey.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/intrusive_ptr.h>
#include <gtest/gtest.h>

#include "tests/csrc/mocks/MockNRT.h"
#include "torch_neuronx/csrc/core/NeuronStorageImpl.h"
#include "torch_neuronx/csrc/core/NeuronTensorImpl.h"

using namespace torch_neuronx::testing;

namespace {

class NeuronTensorImplTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_nrt_ = torch_neuronx::testing::MockNRT::GetInstance();
    ::testing::Mock::VerifyAndClearExpectations(mock_nrt_);
  }

  void TearDown() override { ::testing::Mock::VerifyAndClearExpectations(mock_nrt_); }

  torch_neuronx::testing::MockNRT* mock_nrt_;
};

TEST_F(NeuronTensorImplTest, ShallowCopyFromCPUToNeuron) {
  // Create CPU tensor
  auto cpu_storage = c10::Storage(
      c10::make_intrusive<c10::StorageImpl>(c10::StorageImpl::use_byte_size_t(), 4 * sizeof(float),
                                            c10::GetAllocator(c10::DeviceType::CPU), true));

  auto cpu_impl = c10::make_intrusive<c10::TensorImpl>(
      std::move(cpu_storage), c10::DispatchKeySet(c10::DispatchKey::CPU),
      c10::scalarTypeToTypeMeta(c10::ScalarType::Float));

  cpu_impl->set_sizes_contiguous({4});

  // Create Neuron tensor
  auto neuron_storage = c10::Storage(c10::make_intrusive<c10_neuron::NeuronStorageImpl>(
      c10::StorageImpl::use_byte_size_t(), 4 * sizeof(float),
      c10::DataPtr(nullptr, c10::Device(c10::DeviceType::PrivateUse1, 0)),
      c10::GetAllocator(c10::DeviceType::CPU), true));

  auto neuron_impl = c10::make_intrusive<c10_neuron::NeuronTensorImpl>(
      std::move(neuron_storage), c10::scalarTypeToTypeMeta(c10::ScalarType::Float));

  // Verify initial Neuron dispatch keys
  auto initial_keys = neuron_impl->key_set();
  EXPECT_TRUE(initial_keys.has(c10::DispatchKey::PrivateUse1));
  EXPECT_TRUE(initial_keys.has(c10::DispatchKey::AutogradPrivateUse1));

  // Perform shallow copy from CPU to Neuron
  neuron_impl->shallow_copy_from(cpu_impl);

  // Verify dispatch keys after shallow copy - should have only CPU keys (no Neuron keys)
  auto final_keys = neuron_impl->key_set();
  EXPECT_TRUE(final_keys.has(c10::DispatchKey::CPU));
  EXPECT_FALSE(final_keys.has(c10::DispatchKey::PrivateUse1));
  EXPECT_FALSE(final_keys.has(c10::DispatchKey::AutogradPrivateUse1));
  EXPECT_TRUE(final_keys.has(c10::DispatchKey::AutogradCPU));
  EXPECT_TRUE(final_keys.has(c10::DispatchKey::ADInplaceOrView));
}

TEST_F(NeuronTensorImplTest, ShallowCopyPreservesMetadata) {
  // Create CPU tensor with specific metadata
  auto cpu_storage = c10::Storage(
      c10::make_intrusive<c10::StorageImpl>(c10::StorageImpl::use_byte_size_t(), 12 * sizeof(float),
                                            c10::GetAllocator(c10::DeviceType::CPU), true));

  auto cpu_impl = c10::make_intrusive<c10::TensorImpl>(
      std::move(cpu_storage), c10::DispatchKeySet(c10::DispatchKey::CPU),
      c10::scalarTypeToTypeMeta(c10::ScalarType::Float));

  cpu_impl->set_sizes_and_strides(c10::IntArrayRef({3, 4}), c10::IntArrayRef({4, 1}));
  cpu_impl->set_storage_offset(0);

  // Create Neuron tensor
  auto neuron_storage = c10::Storage(c10::make_intrusive<c10_neuron::NeuronStorageImpl>(
      c10::StorageImpl::use_byte_size_t(), 12 * sizeof(float),
      c10::DataPtr(nullptr, c10::Device(c10::DeviceType::PrivateUse1, 0)),
      c10::GetAllocator(c10::DeviceType::CPU), true));

  auto neuron_impl = c10::make_intrusive<c10_neuron::NeuronTensorImpl>(
      std::move(neuron_storage), c10::scalarTypeToTypeMeta(c10::ScalarType::Float));

  // Perform shallow copy and verify metadata preservation
  neuron_impl->shallow_copy_from(cpu_impl);

  EXPECT_EQ(neuron_impl->sizes(), cpu_impl->sizes());
  EXPECT_EQ(neuron_impl->strides(), cpu_impl->strides());
  EXPECT_EQ(neuron_impl->storage_offset(), cpu_impl->storage_offset());
  EXPECT_EQ(neuron_impl->dtype(), cpu_impl->dtype());
}

TEST_F(NeuronTensorImplTest, ShallowCopyFromNeuronToNeuron) {
  // Create source Neuron tensor
  auto src_storage = c10::Storage(c10::make_intrusive<c10_neuron::NeuronStorageImpl>(
      c10::StorageImpl::use_byte_size_t(), 4 * sizeof(float),
      c10::DataPtr(nullptr, c10::Device(c10::DeviceType::PrivateUse1, 0)),
      c10::GetAllocator(c10::DeviceType::CPU), true));

  auto src_impl = c10::make_intrusive<c10_neuron::NeuronTensorImpl>(
      std::move(src_storage), c10::scalarTypeToTypeMeta(c10::ScalarType::Float));

  src_impl->set_sizes_contiguous({4});

  // Create destination Neuron tensor
  auto dest_storage = c10::Storage(c10::make_intrusive<c10_neuron::NeuronStorageImpl>(
      c10::StorageImpl::use_byte_size_t(), 4 * sizeof(float),
      c10::DataPtr(nullptr, c10::Device(c10::DeviceType::PrivateUse1, 0)),
      c10::GetAllocator(c10::DeviceType::CPU), true));

  auto dest_impl = c10::make_intrusive<c10_neuron::NeuronTensorImpl>(
      std::move(dest_storage), c10::scalarTypeToTypeMeta(c10::ScalarType::Float));

  // Perform shallow copy from Neuron to Neuron
  dest_impl->shallow_copy_from(src_impl);

  // Verify dispatch keys after shallow copy - should have Neuron keys
  auto final_keys = dest_impl->key_set();
  EXPECT_TRUE(final_keys.has(c10::DispatchKey::PrivateUse1));
  EXPECT_TRUE(final_keys.has(c10::DispatchKey::AutogradPrivateUse1));
  EXPECT_TRUE(final_keys.has(c10::DispatchKey::ADInplaceOrView));
}

}  // namespace
