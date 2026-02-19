// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <numeric>
#include <string>
#include <vector>

#include "tests/csrc/mocks/MockNRT.h"
#include "tests/csrc/utils/TestUtils.h"
#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/NeuronStorageImpl.h"
#include "torch_neuronx/csrc/core/NeuronTensorImpl.h"
#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/lazy_materialization/LazyTransformationManager.h"
#include "torch_neuronx/csrc/core/lazy_materialization/OperationPrologue.h"
#include "torch_neuronx/csrc/core/lazy_materialization/TransformationTypes.h"

using namespace torch_neuronx::testing;
using namespace at::neuron::testing;

namespace c10_neuron {
namespace lazy {
namespace {

//============================================================================
// Test Utilities
//============================================================================

// Helper to check if a string contains a substring
bool Contains(const std::string& str, const std::string& substr) {
  return str.find(substr) != std::string::npos;
}

// Helper to create a Neuron tensor
at::Tensor CreateNeuronTensor(const std::vector<int64_t>& shape, c10::ScalarType dtype) {
  size_t element_size = c10::elementSize(dtype);
  size_t total_elements = 1;
  for (auto dim : shape) {
    total_elements *= dim;
  }
  size_t size_bytes = total_elements * element_size;

  // Create unique dummy data pointer
  static std::atomic<uintptr_t> dummy_address_counter{0x1000};
  void* dummy_data = reinterpret_cast<void*>(dummy_address_counter.fetch_add(0x1000));

  auto storage = c10::Storage(c10::make_intrusive<c10_neuron::NeuronStorageImpl>(
      c10::StorageImpl::use_byte_size_t(), size_bytes,
      c10::DataPtr(dummy_data, c10::Device(c10::DeviceType::PrivateUse1, 0)),
      c10::GetAllocator(c10::DeviceType::CPU), true));

  auto tensor_impl = c10::make_intrusive<c10_neuron::NeuronTensorImpl>(
      std::move(storage), c10::scalarTypeToTypeMeta(dtype));

  tensor_impl->set_sizes_contiguous(shape);
  return at::Tensor(tensor_impl);
}

// Sample operation MLIR strings from debug output
const std::string kMatmulOperationMLIR =
    R"(module @jit__aten_mm attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>) -> (tensor<128x512xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x256xf32>, tensor<256x512xf32>) -> tensor<128x512xf32>
    return %0 : tensor<128x512xf32>
  }
})";

// Operation MLIR for add - expects TRANSFORMED shapes [64x128]
const std::string kAddOperationMLIR =
    R"(module @jit_add_computation_default attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> (tensor<64x128xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
})";

// Operation MLIR for concat - expects TRANSFORMED shapes [128x128]
const std::string kConcatOperationMLIR =
    R"(module @jit_concat_computation attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> (tensor<256x128xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<256x128xf32>
    return %0 : tensor<256x128xf32>
  }
})";

const std::string kBmmOperationMLIR =
    R"(module @jit__aten_bmm attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x4x8xf32>, %arg1: tensor<1x8x8xf32>) -> (tensor<1x4x8xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.reshape %arg0 : (tensor<1x4x8xf32>) -> tensor<4x8xf32>
    %1 = stablehlo.dot_general %0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x8xf32>, tensor<1x8x8xf32>) -> tensor<4x1x8xf32>
    %2 = stablehlo.transpose %1, dims = [1, 0, 2] : (tensor<4x1x8xf32>) -> tensor<1x4x8xf32>
    return %2 : tensor<1x4x8xf32>
  }
})";

//============================================================================
// Helper to create XLACompilableKernelExecution for testing
//============================================================================

std::unique_ptr<at::neuron::XLACompilableKernelExecution> CreateTestKernel(
    const std::string& op_name, const std::string& op_mlir, const std::vector<void*>& input_ptrs,
    const std::vector<void*>& output_ptrs,
    const std::vector<at::neuron::TensorContext>& input_contexts,
    const std::vector<at::neuron::TensorContext>& output_contexts,
    const std::vector<std::vector<TensorTransformation>>& transforms) {
  std::vector<uint8_t> hlo_bytes(op_mlir.begin(), op_mlir.end());

  auto kernel = std::make_unique<at::neuron::XLACompilableKernelExecution>(
      op_name, create_fake_tensor_refs(input_ptrs), create_fake_tensor_refs(output_ptrs),
      input_contexts, output_contexts, "test_cache_key", hlo_bytes,
      false,  // has_collectives
      0);     // device_id

  // Set transformations if provided
  if (!transforms.empty()) {
    kernel->SetInputTransformations(transforms);
  }

  return kernel;
}

//============================================================================
// Test Fixture
//============================================================================

class LazyTransformationManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_session_ = std::make_unique<torch_neuronx::testing::MockNRTSession>();
    mock_nrt_ = torch_neuronx::testing::MockNRT::GetInstance();
    ::testing::Mock::VerifyAndClearExpectations(mock_nrt_);

    // Clear caches before each test
    LazyTransformationManager::ClearMergedOperationCache();
  }

  void TearDown() override {
    ::testing::Mock::VerifyAndClearExpectations(mock_nrt_);
    LazyTransformationManager::ClearMergedOperationCache();
    mock_session_.reset();
  }

  std::unique_ptr<torch_neuronx::testing::MockNRTSession> mock_session_;
  torch_neuronx::testing::MockNRT* mock_nrt_;
};

//============================================================================
// Test 1: Matmul with single transpose - order conservation
//============================================================================

TEST_F(LazyTransformationManagerTest, MatmulWithSingleTranspose_OrderConservation) {
  // Create tensors matching the debug output
  // Input 0: [256, 128] with transpose to [128, 256]
  // Input 1: [256, 512] without transformation
  auto tensor0 = CreateNeuronTensor({256, 128}, c10::ScalarType::Float);
  auto tensor1 = CreateNeuronTensor({256, 512}, c10::ScalarType::Float);

  std::vector<void*> input_ptrs = {tensor0.data_ptr(), tensor1.data_ptr()};
  std::vector<void*> output_ptrs = {reinterpret_cast<void*>(0x100000)};

  // Create transformation for input 0
  std::vector<int64_t> curr_perm = {0, 1};
  TensorTransformation transform0(TransformationType::TRANSPOSE, {256, 128},  // input_shape
                                  {128, 256},                                 // output_shape
                                  "f32",                                      // element_type
                                  {1, 0},     // params (permutation)
                                  curr_perm,  // current_perm
                                  {1, 0});    // target_perm

  std::vector<std::vector<TensorTransformation>> transforms = {{transform0}, {}};

  // Create operation context with tensor contexts
  std::vector<at::neuron::TensorContext> input_contexts = {
      at::neuron::TensorContext::FromTensor(tensor0),
      at::neuron::TensorContext::FromTensor(tensor1)};

  // Create kernel
  auto kernel = CreateTestKernel("aten::mm", kMatmulOperationMLIR, input_ptrs, output_ptrs,
                                 input_contexts, {}, transforms);

  // Get pointer before moving
  auto* kernel_ptr = dynamic_cast<at::neuron::XLACompilableKernelExecution*>(kernel.get());

  at::neuron::OperationContext op_context(std::move(kernel));

  // Process inputs
  LazyTransformationManager::ProcessOperationInputs(&op_context);

  // Verify MLIR was updated
  std::string merged_mlir(kernel_ptr->GetHloBytes().begin(), kernel_ptr->GetHloBytes().end());
  ASSERT_FALSE(merged_mlir.empty()) << "Merged MLIR should not be empty";

  // Verify merged main function has UNTRANSFORMED input shapes (before prologue transformation)
  EXPECT_TRUE(Contains(merged_mlir, "func.func @main(%arg0: tensor<256x128xf32>"))
      << "Merged main should have untransformed shape for input 0";
  EXPECT_TRUE(Contains(merged_mlir, "%arg1: tensor<256x512xf32>"))
      << "Merged main should have untransformed shape for input 1";

  // Verify transpose operation is present in the prologue
  EXPECT_TRUE(Contains(merged_mlir, "stablehlo.transpose"))
      << "Should contain transpose operation from prologue";
  EXPECT_TRUE(Contains(merged_mlir, "dims = [1, 0]"))
      << "Should have correct transpose permutation";

  // Verify transformed shapes appear in intermediate functions
  EXPECT_TRUE(Contains(merged_mlir, "tensor<128x256xf32>"))
      << "Should have transposed shape in intermediate results";

  // Verify original operation logic is preserved
  EXPECT_TRUE(Contains(merged_mlir, "stablehlo.dot_general"))
      << "Should contain original dot_general operation";

  // Verify dtype is preserved (f32)
  EXPECT_TRUE(Contains(merged_mlir, "f32")) << "Should preserve f32 dtype";
}

//============================================================================
// Test 2: Same operation with different transforms - different cache entries
//============================================================================

TEST_F(LazyTransformationManagerTest, SameOpDifferentTransforms_DifferentCacheEntries) {
  // First execution: Input 0 transposed
  {
    auto tensor0 = CreateNeuronTensor({256, 128}, c10::ScalarType::Float);
    auto tensor1 = CreateNeuronTensor({256, 512}, c10::ScalarType::Float);

    std::vector<void*> input_ptrs = {tensor0.data_ptr(), tensor1.data_ptr()};
    std::vector<void*> output_ptrs = {reinterpret_cast<void*>(0x100000)};

    std::vector<int64_t> curr_perm = {0, 1};
    TensorTransformation transform0(TransformationType::TRANSPOSE, {256, 128}, {128, 256}, "f32",
                                    {1, 0}, curr_perm, {1, 0});

    std::vector<std::vector<TensorTransformation>> transforms = {{transform0}, {}};

    std::vector<at::neuron::TensorContext> input_contexts = {
        at::neuron::TensorContext::FromTensor(tensor0),
        at::neuron::TensorContext::FromTensor(tensor1)};

    auto kernel = CreateTestKernel("aten::mm", kMatmulOperationMLIR, input_ptrs, output_ptrs,
                                   input_contexts, {}, transforms);
    at::neuron::OperationContext op_context(std::move(kernel));
    LazyTransformationManager::ProcessOperationInputs(&op_context);

    // Should have 1 cache entry
    EXPECT_EQ(LazyTransformationManager::GetMergedOperationCacheSize(), 1)
        << "Should have 1 cache entry after first execution";
  }

  // Second execution: Input 1 transposed instead (different shapes -> different operation MLIR)
  {
    auto tensor0 = CreateNeuronTensor({128, 256}, c10::ScalarType::Float);
    auto tensor1 = CreateNeuronTensor({256, 256}, c10::ScalarType::Float);

    std::vector<void*> input_ptrs = {tensor0.data_ptr(), tensor1.data_ptr()};
    std::vector<void*> output_ptrs = {reinterpret_cast<void*>(0x100001)};

    std::vector<int64_t> curr_perm = {0, 1};
    TensorTransformation transform1(TransformationType::TRANSPOSE, {256, 256}, {256, 256}, "f32",
                                    {1, 0}, curr_perm, {1, 0});

    std::vector<std::vector<TensorTransformation>> transforms = {{}, {transform1}};

    // Operation MLIR for second case: expects transformed shapes [128x256] x [256x256]
    std::string second_matmul_mlir =
        R"(module @jit__aten_mm attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<128x256xf32>, %arg1: tensor<256x256xf32>) -> (tensor<128x256xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x256xf32>, tensor<256x256xf32>) -> tensor<128x256xf32>
    return %0 : tensor<128x256xf32>
  }
})";

    std::vector<at::neuron::TensorContext> input_contexts = {
        at::neuron::TensorContext::FromTensor(tensor0),
        at::neuron::TensorContext::FromTensor(tensor1)};
    auto kernel = CreateTestKernel("aten::mm", second_matmul_mlir, input_ptrs, output_ptrs,
                                   input_contexts, {}, transforms);
    at::neuron::OperationContext op_context(std::move(kernel));
    LazyTransformationManager::ProcessOperationInputs(&op_context);

    // Should have 2 cache entries now (different transformations)
    EXPECT_EQ(LazyTransformationManager::GetMergedOperationCacheSize(), 2)
        << "Should have 2 cache entries with different transformations";
  }
}

//============================================================================
// Test 3: Cache hit verification - same transforms produce cache hit
//============================================================================

TEST_F(LazyTransformationManagerTest, CacheHit_SameTransforms) {
  // First execution
  auto tensor0_first = CreateNeuronTensor({256, 128}, c10::ScalarType::Float);
  auto tensor1_first = CreateNeuronTensor({256, 512}, c10::ScalarType::Float);

  std::vector<void*> input_ptrs_first = {tensor0_first.data_ptr(), tensor1_first.data_ptr()};
  std::vector<void*> output_ptrs = {reinterpret_cast<void*>(0x100000)};

  std::vector<int64_t> curr_perm = {0, 1};
  TensorTransformation transform0(TransformationType::TRANSPOSE, {256, 128}, {128, 256}, "f32",
                                  {1, 0}, curr_perm, {1, 0});

  std::vector<std::vector<TensorTransformation>> transforms = {{transform0}, {}};

  std::vector<at::neuron::TensorContext> input_contexts_first = {
      at::neuron::TensorContext::FromTensor(tensor0_first),
      at::neuron::TensorContext::FromTensor(tensor1_first)};

  auto kernel_first = CreateTestKernel("aten::mm", kMatmulOperationMLIR, input_ptrs_first,
                                       output_ptrs, input_contexts_first, {}, transforms);
  auto* kernel_ptr_first =
      dynamic_cast<at::neuron::XLACompilableKernelExecution*>(kernel_first.get());
  at::neuron::OperationContext op_context_first(std::move(kernel_first));
  LazyTransformationManager::ProcessOperationInputs(&op_context_first);

  std::string first_mlir(kernel_ptr_first->GetHloBytes().begin(),
                         kernel_ptr_first->GetHloBytes().end());
  EXPECT_EQ(LazyTransformationManager::GetMergedOperationCacheSize(), 1);

  // Second execution with same transformation pattern but different tensor instances
  auto tensor0_second = CreateNeuronTensor({256, 128}, c10::ScalarType::Float);
  auto tensor1_second = CreateNeuronTensor({256, 512}, c10::ScalarType::Float);

  std::vector<void*> input_ptrs_second = {tensor0_second.data_ptr(), tensor1_second.data_ptr()};

  std::vector<at::neuron::TensorContext> input_contexts_second = {
      at::neuron::TensorContext::FromTensor(tensor0_second),
      at::neuron::TensorContext::FromTensor(tensor1_second)};

  auto kernel_second = CreateTestKernel("aten::mm", kMatmulOperationMLIR, input_ptrs_second,
                                        output_ptrs, input_contexts_second, {}, transforms);
  auto* kernel_ptr_second =
      dynamic_cast<at::neuron::XLACompilableKernelExecution*>(kernel_second.get());
  at::neuron::OperationContext op_context_second(std::move(kernel_second));
  LazyTransformationManager::ProcessOperationInputs(&op_context_second);

  std::string second_mlir(kernel_ptr_second->GetHloBytes().begin(),
                          kernel_ptr_second->GetHloBytes().end());

  // Should still have only 1 cache entry (cache hit)
  EXPECT_EQ(LazyTransformationManager::GetMergedOperationCacheSize(), 1)
      << "Cache size should remain 1 on cache hit";

  // MLIRs should be identical
  EXPECT_EQ(first_mlir, second_mlir) << "Cached MLIR should be identical";
}

//============================================================================
// Test 4: Both inputs transformed - order and dtype preservation
//============================================================================

TEST_F(LazyTransformationManagerTest, BothInputsTransformed_OrderAndDtypePreservation) {
  // Both inputs [128, 64] transposed to [64, 128] for add operation
  auto tensor0 = CreateNeuronTensor({128, 64}, c10::ScalarType::Float);
  auto tensor1 = CreateNeuronTensor({128, 64}, c10::ScalarType::Float);

  std::vector<void*> input_ptrs = {tensor0.data_ptr(), tensor1.data_ptr()};
  std::vector<void*> output_ptrs = {reinterpret_cast<void*>(0x100000)};

  std::vector<int64_t> curr_perm = {0, 1};
  TensorTransformation transform0(TransformationType::TRANSPOSE, {128, 64}, {64, 128}, "f32",
                                  {1, 0}, curr_perm, {1, 0});
  TensorTransformation transform1(TransformationType::TRANSPOSE, {128, 64}, {64, 128}, "f32",
                                  {1, 0}, curr_perm, {1, 0});

  std::vector<std::vector<TensorTransformation>> transforms = {{transform0}, {transform1}};

  std::vector<at::neuron::TensorContext> input_contexts = {
      at::neuron::TensorContext::FromTensor(tensor0),
      at::neuron::TensorContext::FromTensor(tensor1)};

  auto kernel = CreateTestKernel("add_default", kAddOperationMLIR, input_ptrs, output_ptrs,
                                 input_contexts, {}, transforms);
  auto* kernel_ptr = dynamic_cast<at::neuron::XLACompilableKernelExecution*>(kernel.get());
  at::neuron::OperationContext op_context(std::move(kernel));
  LazyTransformationManager::ProcessOperationInputs(&op_context);

  std::string merged_mlir(kernel_ptr->GetHloBytes().begin(), kernel_ptr->GetHloBytes().end());

  // Verify both inputs are in order
  EXPECT_TRUE(Contains(merged_mlir, "%arg0:")) << "Should have arg0";
  EXPECT_TRUE(Contains(merged_mlir, "%arg1:")) << "Should have arg1";

  // Verify both inputs have transposed shape
  size_t arg0_pos = merged_mlir.find("%arg0:");
  size_t arg1_pos = merged_mlir.find("%arg1:");
  ASSERT_NE(arg0_pos, std::string::npos);
  ASSERT_NE(arg1_pos, std::string::npos);
  EXPECT_LT(arg0_pos, arg1_pos) << "arg0 should come before arg1";

  // Verify shapes after transformation [64x128]
  EXPECT_TRUE(Contains(merged_mlir, "tensor<64x128xf32>")) << "Should have transposed shape";

  // Verify both transpose operations are present
  size_t first_transpose = merged_mlir.find("stablehlo.transpose");
  ASSERT_NE(first_transpose, std::string::npos) << "Should have first transpose";

  size_t second_transpose = merged_mlir.find("stablehlo.transpose", first_transpose + 1);
  ASSERT_NE(second_transpose, std::string::npos) << "Should have second transpose";

  // Verify add operation is present
  EXPECT_TRUE(Contains(merged_mlir, "stablehlo.add")) << "Should contain add operation";
}

//============================================================================
// Test 5: Mixed transformation - one transformed, one direct
//============================================================================

TEST_F(LazyTransformationManagerTest, MixedTransformation_OneTransformedOneDirect) {
  // Input 0: [128, 128] transposed to [128, 128]
  // Input 1: [128, 128] without transformation
  auto tensor0 = CreateNeuronTensor({128, 128}, c10::ScalarType::Float);
  auto tensor1 = CreateNeuronTensor({128, 128}, c10::ScalarType::Float);

  std::vector<void*> input_ptrs = {tensor0.data_ptr(), tensor1.data_ptr()};
  std::vector<void*> output_ptrs = {reinterpret_cast<void*>(0x100000)};

  std::vector<int64_t> curr_perm = {0, 1};
  TensorTransformation transform0(TransformationType::TRANSPOSE, {128, 128}, {128, 128}, "f32",
                                  {1, 0}, curr_perm, {1, 0});

  std::vector<std::vector<TensorTransformation>> transforms = {{transform0}, {}};

  std::vector<at::neuron::TensorContext> input_contexts = {
      at::neuron::TensorContext::FromTensor(tensor0),
      at::neuron::TensorContext::FromTensor(tensor1)};

  auto kernel = CreateTestKernel("concat", kConcatOperationMLIR, input_ptrs, output_ptrs,
                                 input_contexts, {}, transforms);
  auto* kernel_ptr = dynamic_cast<at::neuron::XLACompilableKernelExecution*>(kernel.get());
  at::neuron::OperationContext op_context(std::move(kernel));
  LazyTransformationManager::ProcessOperationInputs(&op_context);

  std::string merged_mlir(kernel_ptr->GetHloBytes().begin(), kernel_ptr->GetHloBytes().end());

  // Verify order is preserved
  EXPECT_TRUE(Contains(merged_mlir, "%arg0:")) << "Should have arg0";
  EXPECT_TRUE(Contains(merged_mlir, "%arg1:")) << "Should have arg1";

  // Verify one transpose operation
  size_t transpose_count = 0;
  size_t pos = 0;
  while ((pos = merged_mlir.find("stablehlo.transpose", pos)) != std::string::npos) {
    transpose_count++;
    pos++;
  }
  EXPECT_EQ(transpose_count, 1) << "Should have exactly 1 transpose for the transformed input";

  // Verify concat operation is present
  EXPECT_TRUE(Contains(merged_mlir, "stablehlo.concatenate"))
      << "Should contain concatenate operation";
}

//============================================================================
// Test 6: Different dtypes - bfloat16
//============================================================================

TEST_F(LazyTransformationManagerTest, DifferentDtype_BFloat16) {
  // Same as matmul test but with bf16
  auto tensor0 = CreateNeuronTensor({256, 128}, c10::ScalarType::BFloat16);
  auto tensor1 = CreateNeuronTensor({256, 512}, c10::ScalarType::BFloat16);

  std::vector<void*> input_ptrs = {tensor0.data_ptr(), tensor1.data_ptr()};
  std::vector<void*> output_ptrs = {reinterpret_cast<void*>(0x100000)};

  std::vector<int64_t> curr_perm = {0, 1};
  TensorTransformation transform0(TransformationType::TRANSPOSE, {256, 128}, {128, 256}, "bf16",
                                  {1, 0}, curr_perm, {1, 0});

  std::vector<std::vector<TensorTransformation>> transforms = {{transform0}, {}};

  // Need to update operation MLIR to use bf16
  std::string bf16_matmul_mlir = kMatmulOperationMLIR;
  size_t pos = 0;
  while ((pos = bf16_matmul_mlir.find("f32", pos)) != std::string::npos) {
    bf16_matmul_mlir.replace(pos, 3, "bf16");
    pos += 4;
  }

  std::vector<at::neuron::TensorContext> input_contexts = {
      at::neuron::TensorContext::FromTensor(tensor0),
      at::neuron::TensorContext::FromTensor(tensor1)};

  auto kernel = CreateTestKernel("aten::mm", bf16_matmul_mlir, input_ptrs, output_ptrs,
                                 input_contexts, {}, transforms);
  auto* kernel_ptr = dynamic_cast<at::neuron::XLACompilableKernelExecution*>(kernel.get());
  at::neuron::OperationContext op_context(std::move(kernel));
  LazyTransformationManager::ProcessOperationInputs(&op_context);

  std::string merged_mlir(kernel_ptr->GetHloBytes().begin(), kernel_ptr->GetHloBytes().end());

  // Verify bf16 dtype is preserved
  EXPECT_TRUE(Contains(merged_mlir, "bf16")) << "Should preserve bf16 dtype";
  EXPECT_TRUE(Contains(merged_mlir, "tensor<128x256xbf16>"))
      << "Should have bf16 type in transposed shape";
  EXPECT_TRUE(Contains(merged_mlir, "tensor<256x512xbf16>"))
      << "Should have bf16 type in second input";
}

//============================================================================
// Test 7: 3D tensors with transpose
//============================================================================

TEST_F(LazyTransformationManagerTest, ThreeDimensionalTensor_Transpose) {
  // BMM with 3D transpose
  auto tensor0 = CreateNeuronTensor({1, 8, 4}, c10::ScalarType::Float);
  auto tensor1 = CreateNeuronTensor({1, 8, 8}, c10::ScalarType::Float);

  std::vector<void*> input_ptrs = {tensor0.data_ptr(), tensor1.data_ptr()};
  std::vector<void*> output_ptrs = {reinterpret_cast<void*>(0x100000)};

  std::vector<int64_t> curr_perm = {0, 1, 2};
  TensorTransformation transform0(TransformationType::TRANSPOSE, {1, 8, 4}, {1, 4, 8}, "f32",
                                  {0, 2, 1}, curr_perm, {0, 2, 1});

  std::vector<std::vector<TensorTransformation>> transforms = {{transform0}, {}};

  std::vector<at::neuron::TensorContext> input_contexts = {
      at::neuron::TensorContext::FromTensor(tensor0),
      at::neuron::TensorContext::FromTensor(tensor1)};

  auto kernel = CreateTestKernel("aten::bmm", kBmmOperationMLIR, input_ptrs, output_ptrs,
                                 input_contexts, {}, transforms);
  auto* kernel_ptr = dynamic_cast<at::neuron::XLACompilableKernelExecution*>(kernel.get());
  at::neuron::OperationContext op_context(std::move(kernel));
  LazyTransformationManager::ProcessOperationInputs(&op_context);

  std::string merged_mlir(kernel_ptr->GetHloBytes().begin(), kernel_ptr->GetHloBytes().end());

  // Verify 3D shapes are preserved
  EXPECT_TRUE(Contains(merged_mlir, "tensor<1x4x8xf32>"))
      << "Should have 3D transposed shape for input 0";
  EXPECT_TRUE(Contains(merged_mlir, "tensor<1x8x8xf32>")) << "Should have 3D shape for input 1";

  // Verify 3D transpose with permutation [0, 2, 1]
  EXPECT_TRUE(Contains(merged_mlir, "stablehlo.transpose")) << "Should contain transpose operation";

  // Verify original BMM operations are preserved
  EXPECT_TRUE(Contains(merged_mlir, "stablehlo.reshape"))
      << "Should contain reshape operation from original BMM";
  EXPECT_TRUE(Contains(merged_mlir, "stablehlo.dot_general"))
      << "Should contain dot_general operation";
}

}  // namespace
}  // namespace lazy
}  // namespace c10_neuron
