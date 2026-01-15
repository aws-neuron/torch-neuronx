#include <ATen/ATen.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/Storage.h>
#include <c10/util/intrusive_ptr.h>
#include <gtest/gtest.h>

#include <numeric>
#include <string>
#include <vector>

#include "tests/csrc/mocks/MockNRT.h"
#include "torch_neuronx/csrc/core/NeuronStorageImpl.h"
#include "torch_neuronx/csrc/core/NeuronTensorImpl.h"
#include "torch_neuronx/csrc/core/lazy_materialization/OperationPrologue.h"
#include "torch_neuronx/csrc/core/lazy_materialization/TransformationTypes.h"

using namespace torch_neuronx::testing;

namespace c10_neuron {
namespace lazy {
namespace {

//============================================================================
// Test Utilities
//============================================================================

// Helper function to check if a string contains a substring
bool Contains(const std::string& str, const std::string& substr) {
  return str.find(substr) != std::string::npos;
}

// Helper function to extract input pointers and contexts from tensors
void ExtractInputPtrsAndContexts(const std::vector<at::Tensor>& inputs,
                                 std::vector<void*>& input_ptrs,
                                 std::vector<at::neuron::TensorContext>& input_contexts) {
  input_ptrs.clear();
  input_contexts.clear();
  for (const auto& t : inputs) {
    input_ptrs.push_back(t.data_ptr());
    input_contexts.push_back(at::neuron::TensorContext::FromTensor(t));
  }
}

// Helper to create a Neuron tensor with NeuronStorageImpl
// Note: Transformations are no longer stored on the tensor itself,
// they are passed as a parameter to OperationPrologue::GeneratePrologue
at::Tensor CreateNeuronTensor(const std::vector<int64_t>& shape, c10::ScalarType dtype) {
  // Calculate total size in bytes
  size_t element_size = c10::elementSize(dtype);
  size_t total_elements = 1;
  for (auto dim : shape) {
    total_elements *= dim;
  }
  size_t size_bytes = total_elements * element_size;

  // Create a unique dummy data pointer for each tensor (not nullptr!)
  // Use a static counter to ensure each tensor gets a unique address
  static std::atomic<uintptr_t> dummy_address_counter{0x1000};
  void* dummy_data = reinterpret_cast<void*>(dummy_address_counter.fetch_add(0x1000));

  // Create NeuronStorageImpl with unique data pointer
  auto storage = c10::Storage(c10::make_intrusive<c10_neuron::NeuronStorageImpl>(
      c10::StorageImpl::use_byte_size_t(), size_bytes,
      c10::DataPtr(dummy_data, c10::Device(c10::DeviceType::PrivateUse1, 0)),
      c10::GetAllocator(c10::DeviceType::CPU), true));

  // Create NeuronTensorImpl
  auto tensor_impl = c10::make_intrusive<c10_neuron::NeuronTensorImpl>(
      std::move(storage), c10::scalarTypeToTypeMeta(dtype));

  // Set tensor shape
  tensor_impl->set_sizes_contiguous(shape);

  // Create at::Tensor from TensorImpl
  return at::Tensor(tensor_impl);
}

//============================================================================
// Test Fixture
//============================================================================

class OperationPrologueTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_nrt_ = torch_neuronx::testing::MockNRT::GetInstance();
    ::testing::Mock::VerifyAndClearExpectations(mock_nrt_);
  }

  void TearDown() override { ::testing::Mock::VerifyAndClearExpectations(mock_nrt_); }

  torch_neuronx::testing::MockNRT* mock_nrt_;
};

//============================================================================
// Test 1: Single Input, Single Transpose
//============================================================================

TEST_F(OperationPrologueTest, SingleInputSingleTranspose) {
  // Setup: Create tensor with one TRANSPOSE transformation
  std::vector<int64_t> shape = {2, 3, 4};
  std::vector<int64_t> transpose_params = {1, 0, 2};  // Swap first two dims

  // Output shape after applying permutation
  std::vector<int64_t> output_shape = {shape[1], shape[0], shape[2]};  // {3, 2, 4}

  // Create identity permutation for current state
  std::vector<int64_t> current_perm(shape.size());
  std::iota(current_perm.begin(), current_perm.end(), 0);  // {0, 1, 2}

  TensorTransformation transpose_transform(TransformationType::TRANSPOSE,
                                           shape,              // input_shape
                                           output_shape,       // output_shape
                                           "f32",              // element_type
                                           transpose_params,   // params (permutation)
                                           current_perm,       // current_perm
                                           transpose_params);  // target_perm

  auto tensor = CreateNeuronTensor(shape, c10::ScalarType::Float);

  std::vector<at::Tensor> inputs = {tensor};

  // Extract input pointers and create tensor contexts
  std::vector<void*> input_ptrs;
  std::vector<at::neuron::TensorContext> input_contexts;
  for (const auto& t : inputs) {
    input_ptrs.push_back(t.data_ptr());
    input_contexts.push_back(at::neuron::TensorContext::FromTensor(t));
  }

  // Transformations are passed as kernel_transforms parameter
  std::vector<std::vector<TensorTransformation>> kernel_transforms = {{transpose_transform}};

  // Compute cache key first
  std::string cache_key = OperationPrologue::ComputePrologueCacheKey(input_ptrs, input_contexts,
                                                                     "test_op", kernel_transforms);

  // Call: Generate prologue with cache key
  auto result = OperationPrologue::GeneratePrologue(input_ptrs, input_contexts, "test_op",
                                                    kernel_transforms, cache_key);

  // Assertions: Basic result structure
  EXPECT_TRUE(result.success) << "Prologue generation should succeed";
  EXPECT_TRUE(result.has_transformations) << "Should detect transformations";

  EXPECT_EQ(result.tasks.size(), 1) << "Should have 1 transformation task";
  EXPECT_EQ(result.tasks[0].type, TransformationType::TRANSPOSE);
  EXPECT_EQ(result.tasks[0].input_index, 0);
  EXPECT_EQ(result.tasks[0].input_shape, shape);

  EXPECT_FALSE(result.prologue_mlir_str.empty()) << "MLIR should be generated";
  EXPECT_TRUE(Contains(result.prologue_mlir_str, "stablehlo.transpose"))
      << "MLIR should contain transpose operation";

  EXPECT_FALSE(result.prologue_cache_key.empty()) << "Cache key should be generated";
}

//============================================================================
// Test 2: Single Input, Multiple Transpose Transformations
//============================================================================

TEST_F(OperationPrologueTest, SingleInputMultipleTransposeTransformations) {
  // Setup: Create tensor with 3 consecutive TRANSPOSE transformations
  // Note: RESHAPE is not yet implemented, so we test only TRANSPOSE
  std::vector<int64_t> shape = {4, 5, 6, 7};

  // Create transformations with proper shapes
  std::vector<int64_t> shape1 = {5, 4, 6, 7};  // After first transpose: swap dims 0,1
  std::vector<int64_t> shape2 = {5, 6, 4, 7};  // After second transpose: swap dims 1,2
  std::vector<int64_t> shape3 = {5, 6, 7, 4};  // After third transpose: swap dims 2,3

  // Create identity permutation and target permutations for each transformation
  std::vector<int64_t> curr_perm_4d(4);
  std::iota(curr_perm_4d.begin(), curr_perm_4d.end(), 0);  // {0, 1, 2, 3}

  std::vector<TensorTransformation> transformations = {
      TensorTransformation(TransformationType::TRANSPOSE, shape, shape1, "f32", {1, 0, 2, 3},
                           curr_perm_4d, {1, 0, 2, 3}),
      TensorTransformation(TransformationType::TRANSPOSE, shape1, shape2, "f32", {0, 2, 1, 3},
                           {1, 0, 2, 3}, {0, 2, 1, 3}),
      TensorTransformation(TransformationType::TRANSPOSE, shape2, shape3, "f32", {0, 1, 3, 2},
                           {0, 2, 1, 3}, {0, 1, 3, 2})};

  auto tensor = CreateNeuronTensor(shape, c10::ScalarType::Float);

  std::vector<at::Tensor> inputs = {tensor};

  // Extract input pointers and create tensor contexts
  std::vector<void*> input_ptrs;
  std::vector<at::neuron::TensorContext> input_contexts;
  ExtractInputPtrsAndContexts(inputs, input_ptrs, input_contexts);

  // Transformations are passed as kernel_transforms parameter
  std::vector<std::vector<TensorTransformation>> kernel_transforms = {transformations};

  // Compute cache key first
  std::string cache_key = OperationPrologue::ComputePrologueCacheKey(
      input_ptrs, input_contexts, "test_complex_op", kernel_transforms);

  // Call: Generate prologue with cache key
  auto result = OperationPrologue::GeneratePrologue(input_ptrs, input_contexts, "test_complex_op",
                                                    kernel_transforms, cache_key);

  // Assertions
  EXPECT_TRUE(result.success) << "Complex prologue generation should succeed";
  EXPECT_TRUE(result.has_transformations) << "Should detect transformations";

  // Consecutive transposes should be merged into a single task
  EXPECT_GE(result.tasks.size(), 1) << "Should have at least 1 task after grouping";
  EXPECT_LE(result.tasks.size(), 3) << "Should have at most 3 tasks (one per transformation)";

  EXPECT_FALSE(result.prologue_mlir_str.empty());
  EXPECT_FALSE(result.prologue_cache_key.empty());

  // Cache key should reflect all transformations
  EXPECT_GT(result.prologue_cache_key.length(), 50)
      << "Cache key should be substantial for multiple transformations";

  // Verify MLIR contains transpose operations
  EXPECT_TRUE(Contains(result.prologue_mlir_str, "stablehlo.transpose"))
      << "MLIR should contain transpose operations";
}

//============================================================================
// Test 3: Two Inputs, One Transformation Each
//============================================================================

TEST_F(OperationPrologueTest, TwoInputsOneTransformationEach) {
  // Setup: Create two tensors with different transformations

  // Tensor 0: 2D transpose
  std::vector<int64_t> shape0 = {8, 9};
  std::vector<int64_t> output_shape0 = {9, 8};  // After transpose

  std::vector<int64_t> curr_perm_2d(2);
  std::iota(curr_perm_2d.begin(), curr_perm_2d.end(), 0);  // {0, 1}

  TensorTransformation transform0(TransformationType::TRANSPOSE, shape0, output_shape0, "f32",
                                  {1, 0}, curr_perm_2d, {1, 0});

  auto tensor0 = CreateNeuronTensor(shape0, c10::ScalarType::Float);

  // Tensor 1: 3D transpose
  std::vector<int64_t> shape1 = {10, 11, 12};
  std::vector<int64_t> output_shape1 = {12, 10, 11};  // After transpose

  std::vector<int64_t> curr_perm_3d(3);
  std::iota(curr_perm_3d.begin(), curr_perm_3d.end(), 0);  // {0, 1, 2}

  TensorTransformation transform1(TransformationType::TRANSPOSE, shape1, output_shape1, "f32",
                                  {2, 0, 1}, curr_perm_3d, {2, 0, 1});

  auto tensor1 = CreateNeuronTensor(shape1, c10::ScalarType::Float);

  std::vector<at::Tensor> inputs = {tensor0, tensor1};

  // Extract input pointers and create tensor contexts
  std::vector<void*> input_ptrs;
  std::vector<at::neuron::TensorContext> input_contexts;
  ExtractInputPtrsAndContexts(inputs, input_ptrs, input_contexts);

  // Transformations are passed as kernel_transforms parameter
  std::vector<std::vector<TensorTransformation>> kernel_transforms = {{transform0}, {transform1}};

  // Compute cache key first
  std::string cache_key = OperationPrologue::ComputePrologueCacheKey(
      input_ptrs, input_contexts, "test_multi_input", kernel_transforms);

  // Call: Generate prologue with cache key
  auto result = OperationPrologue::GeneratePrologue(input_ptrs, input_contexts, "test_multi_input",
                                                    kernel_transforms, cache_key);

  // Assertions - Must succeed with 2 tasks
  ASSERT_TRUE(result.success) << "Multi-input prologue generation should succeed";
  ASSERT_EQ(result.tasks.size(), 2) << "Should have 2 transformation tasks";

  EXPECT_TRUE(result.has_transformations) << "Should detect transformations";

  // Verify task ordering and input indices
  EXPECT_EQ(result.tasks[0].input_index, 0);
  EXPECT_EQ(result.tasks[0].type, TransformationType::TRANSPOSE);

  EXPECT_EQ(result.tasks[1].input_index, 1);
  EXPECT_EQ(result.tasks[1].type, TransformationType::TRANSPOSE);

  EXPECT_FALSE(result.prologue_mlir_str.empty());
  EXPECT_FALSE(result.prologue_cache_key.empty());

  // The MLIR should contain operations for both transformations
  EXPECT_TRUE(Contains(result.prologue_mlir_str, "stablehlo.transpose"))
      << "MLIR should contain transpose operations";

  // Cache key should reflect both transformations
  EXPECT_TRUE(Contains(result.prologue_cache_key, "0_"))
      << "Cache key should reference first input";
}

//============================================================================
// Test 4: Cache Hit Verification
//============================================================================

TEST_F(OperationPrologueTest, CacheHitVerification) {
  // Setup: Create tensor with transformation
  std::vector<int64_t> shape = {3, 4, 5};
  std::vector<int64_t> output_shape = {5, 3, 4};  // After transpose

  std::vector<int64_t> curr_perm(3);
  std::iota(curr_perm.begin(), curr_perm.end(), 0);  // {0, 1, 2}

  TensorTransformation transform(TransformationType::TRANSPOSE, shape, output_shape, "f32",
                                 {2, 0, 1}, curr_perm, {2, 0, 1});

  auto tensor1 = CreateNeuronTensor(shape, c10::ScalarType::Float);

  std::vector<at::Tensor> inputs1 = {tensor1};

  // Extract input pointers and create tensor contexts
  std::vector<void*> input_ptrs1;
  std::vector<at::neuron::TensorContext> input_contexts1;
  ExtractInputPtrsAndContexts(inputs1, input_ptrs1, input_contexts1);

  // Transformations are passed as kernel_transforms parameter
  std::vector<std::vector<TensorTransformation>> kernel_transforms = {{transform}};

  // Compute cache key first
  std::string cache_key1 = OperationPrologue::ComputePrologueCacheKey(
      input_ptrs1, input_contexts1, "cache_test_op", kernel_transforms);

  // First call: Generate prologue with cache key
  auto result1 = OperationPrologue::GeneratePrologue(input_ptrs1, input_contexts1, "cache_test_op",
                                                     kernel_transforms, cache_key1);

  EXPECT_TRUE(result1.success) << "First prologue generation should succeed";
  EXPECT_TRUE(result1.has_transformations) << "Should detect transformations";

  // Create identical setup for second call
  auto tensor2 = CreateNeuronTensor(shape, c10::ScalarType::Float);

  std::vector<at::Tensor> inputs2 = {tensor2};

  // Extract input pointers and create tensor contexts
  std::vector<void*> input_ptrs2;
  std::vector<at::neuron::TensorContext> input_contexts2;
  ExtractInputPtrsAndContexts(inputs2, input_ptrs2, input_contexts2);

  // Compute cache key for second call
  std::string cache_key2 = OperationPrologue::ComputePrologueCacheKey(
      input_ptrs2, input_contexts2, "cache_test_op", kernel_transforms);

  // Second call: Should hit cache
  auto result2 = OperationPrologue::GeneratePrologue(input_ptrs2, input_contexts2, "cache_test_op",
                                                     kernel_transforms, cache_key2);

  EXPECT_TRUE(result2.success) << "Second prologue generation should succeed";
  EXPECT_TRUE(result2.has_transformations) << "Should detect transformations";

  // Cache keys should match for identical setups
  EXPECT_EQ(result1.prologue_cache_key, result2.prologue_cache_key)
      << "Cache keys should match for identical transformations";

  // MLIR strings should be identical (from cache)
  EXPECT_EQ(result1.prologue_mlir_str, result2.prologue_mlir_str)
      << "MLIR should be identical from cache";

  // Task counts should match
  EXPECT_EQ(result1.tasks.size(), result2.tasks.size());
}

//============================================================================
// Edge Case Tests
//============================================================================

TEST_F(OperationPrologueTest, EmptyInputVector) {
  // Test with no inputs
  std::vector<void*> input_ptrs;
  std::vector<at::neuron::TensorContext> input_contexts;

  // Compute cache key first
  std::string cache_key =
      OperationPrologue::ComputePrologueCacheKey(input_ptrs, input_contexts, "empty_test", {});

  auto result =
      OperationPrologue::GeneratePrologue(input_ptrs, input_contexts, "empty_test", {}, cache_key);

  EXPECT_TRUE(result.success) << "Should handle empty input gracefully";
  EXPECT_FALSE(result.has_transformations) << "No inputs means no transformations";
  EXPECT_TRUE(result.prologue_cache_key.empty())
      << "Cache key should be empty for no transformations";
}

TEST_F(OperationPrologueTest, TensorWithoutTransformations) {
  // Create a Neuron tensor without any transformations
  std::vector<int64_t> shape = {2, 3};
  auto tensor = CreateNeuronTensor(shape, c10::ScalarType::Float);

  std::vector<at::Tensor> inputs = {tensor};

  // Extract input pointers and create tensor contexts
  std::vector<void*> input_ptrs;
  std::vector<at::neuron::TensorContext> input_contexts;
  ExtractInputPtrsAndContexts(inputs, input_ptrs, input_contexts);

  // No transformations passed - compute cache key
  std::string cache_key = OperationPrologue::ComputePrologueCacheKey(input_ptrs, input_contexts,
                                                                     "no_transform_test", {});

  auto result = OperationPrologue::GeneratePrologue(input_ptrs, input_contexts, "no_transform_test",
                                                    {}, cache_key);

  EXPECT_TRUE(result.success);

  // Inputs without transformations now get NONE tasks
  // So has_transformations will be true, and MLIR will be generated for the NONE operation
  EXPECT_TRUE(result.has_transformations)
      << "With cache_fix.diff, NONE transformations are created";
  EXPECT_FALSE(result.prologue_mlir_str.empty())
      << "MLIR should be generated for NONE transformation (identity operation)";
  EXPECT_FALSE(result.prologue_cache_key.empty())
      << "Cache key should be generated for NONE transformation";

  // Verify that a NONE transformation task was created
  ASSERT_EQ(result.tasks.size(), 1) << "Should have 1 NONE transformation task";
  EXPECT_EQ(result.tasks[0].type, TransformationType::NONE)
      << "Task should be NONE type (identity operation)";
  EXPECT_EQ(result.tasks[0].input_index, 0) << "Task should be for input 0";
}

TEST_F(OperationPrologueTest, MixedNeuronAndCPUTensors) {
  // Create one Neuron tensor with transformation
  std::vector<int64_t> shape = {2, 3};
  std::vector<int64_t> output_shape = {3, 2};  // After transpose

  std::vector<int64_t> curr_perm(2);
  std::iota(curr_perm.begin(), curr_perm.end(), 0);  // {0, 1}

  TensorTransformation transform(TransformationType::TRANSPOSE, shape, output_shape, "f32", {1, 0},
                                 curr_perm, {1, 0});

  auto neuron_tensor = CreateNeuronTensor(shape, c10::ScalarType::Float);

  // Create one CPU tensor
  auto cpu_tensor = at::empty({2, 3}, at::TensorOptions().dtype(c10::ScalarType::Float));

  std::vector<at::Tensor> inputs = {neuron_tensor, cpu_tensor};

  // Extract input pointers and create tensor contexts
  std::vector<void*> input_ptrs;
  std::vector<at::neuron::TensorContext> input_contexts;
  ExtractInputPtrsAndContexts(inputs, input_ptrs, input_contexts);

  // Transformations are passed as kernel_transforms parameter - only for first input
  std::vector<std::vector<TensorTransformation>> kernel_transforms = {{transform}, {}};

  // Compute cache key first
  std::string cache_key = OperationPrologue::ComputePrologueCacheKey(
      input_ptrs, input_contexts, "mixed_test", kernel_transforms);

  auto result = OperationPrologue::GeneratePrologue(input_ptrs, input_contexts, "mixed_test",
                                                    kernel_transforms, cache_key);

  EXPECT_TRUE(result.success);

  // With the cache_fix.diff changes, inputs without transformations now get NONE tasks
  // So we should have 2 tasks: one TRANSPOSE for input 0, one NONE for input 1
  EXPECT_TRUE(result.has_transformations) << "Should detect transformations";
  EXPECT_EQ(result.tasks.size(), 2) << "Should have 2 tasks (TRANSPOSE + NONE)";
  EXPECT_EQ(result.tasks[0].input_index, 0) << "First task should be for input 0";
  EXPECT_EQ(result.tasks[0].type, TransformationType::TRANSPOSE);
  EXPECT_EQ(result.tasks[1].input_index, 1) << "Second task should be for input 1 (NONE)";
  EXPECT_EQ(result.tasks[1].type, TransformationType::NONE);
}

//============================================================================
// Test 5: Two Different Transformation Chains for Same Input Address
// This tests the STEP 5 logic for merging multiple transformation chains
//============================================================================

TEST_F(OperationPrologueTest, TwoChainsWithSameInputAddress) {
  // This test verifies that when we have multiple independent transformation chains
  // operating on different inputs, the STEP 5 merge logic correctly handles them
  // using unique slot addresses to avoid address conflicts during merge.

  // Create two separate inputs with different shapes and transformations
  // Input 0: 3D tensor with transpose
  std::vector<int64_t> shape0 = {2, 3, 4};
  std::vector<int64_t> output_shape0 = {3, 2, 4};  // After transpose

  std::vector<int64_t> curr_perm_3d(3);
  std::iota(curr_perm_3d.begin(), curr_perm_3d.end(), 0);  // {0, 1, 2}

  TensorTransformation transform0(TransformationType::TRANSPOSE, shape0, output_shape0, "f32",
                                  {1, 0, 2}, curr_perm_3d, {1, 0, 2});

  auto tensor0 = CreateNeuronTensor(shape0, c10::ScalarType::Float);

  // Input 1: 2D tensor with different transpose
  std::vector<int64_t> shape1 = {5, 6};
  std::vector<int64_t> output_shape1 = {6, 5};  // After transpose

  std::vector<int64_t> curr_perm_2d(2);
  std::iota(curr_perm_2d.begin(), curr_perm_2d.end(), 0);  // {0, 1}

  TensorTransformation transform1(TransformationType::TRANSPOSE, shape1, output_shape1, "f32",
                                  {1, 0}, curr_perm_2d, {1, 0});

  auto tensor1 = CreateNeuronTensor(shape1, c10::ScalarType::Float);

  // Create input list with both tensors
  std::vector<at::Tensor> inputs = {tensor0, tensor1};

  // Each input has its own transformation chain
  std::vector<std::vector<TensorTransformation>> kernel_transforms = {{transform0}, {transform1}};

  // Extract input pointers and create tensor contexts
  std::vector<void*> input_ptrs;
  std::vector<at::neuron::TensorContext> input_contexts;
  ExtractInputPtrsAndContexts(inputs, input_ptrs, input_contexts);

  // Compute cache key first
  std::string cache_key = OperationPrologue::ComputePrologueCacheKey(
      input_ptrs, input_contexts, "test_two_chains_merge", kernel_transforms);

  // Call: Generate prologue with cache key - this will exercise STEP 5's merge logic
  auto result = OperationPrologue::GeneratePrologue(
      input_ptrs, input_contexts, "test_two_chains_merge", kernel_transforms, cache_key);

  // Assertions: Verify successful merge
  ASSERT_TRUE(result.success) << "Merge should succeed";
  ASSERT_TRUE(result.has_transformations) << "Should detect transformations";

  // Should have 2 tasks, one for each input's transformation chain
  ASSERT_EQ(result.tasks.size(), 2) << "Should have 2 transformation tasks (one per input)";

  // Verify task details
  EXPECT_EQ(result.tasks[0].input_index, 0) << "First task should be for input 0";
  EXPECT_EQ(result.tasks[0].type, TransformationType::TRANSPOSE);
  EXPECT_EQ(result.tasks[0].input_shape, shape0);
  EXPECT_EQ(result.tasks[0].output_shape, output_shape0);

  EXPECT_EQ(result.tasks[1].input_index, 1) << "Second task should be for input 1";
  EXPECT_EQ(result.tasks[1].type, TransformationType::TRANSPOSE);
  EXPECT_EQ(result.tasks[1].input_shape, shape1);
  EXPECT_EQ(result.tasks[1].output_shape, output_shape1);

  // Verify MLIR was generated
  ASSERT_FALSE(result.prologue_mlir_str.empty()) << "MLIR should be generated";

  // MLIR should contain transpose operations for both chains
  EXPECT_TRUE(Contains(result.prologue_mlir_str, "stablehlo.transpose"))
      << "MLIR should contain transpose operations";

  // Verify cache key was generated
  EXPECT_FALSE(result.prologue_cache_key.empty()) << "Cache key should be generated";

  // Cache key should reflect both transformations
  EXPECT_GT(result.prologue_cache_key.length(), 50)
      << "Cache key should be substantial for multiple chains";

  // Verify input_was_transformed flags
  EXPECT_EQ(result.input_was_transformed.size(), 2);
  EXPECT_TRUE(result.input_was_transformed[0]) << "Input 0 should be marked as transformed";
  EXPECT_TRUE(result.input_was_transformed[1]) << "Input 1 should be marked as transformed";
}

//============================================================================
// Test 6: Same Tensor with Mixed Transformation - One Transformed, One Direct
// This tests concatenating the same tensor with and without transformation
// Pattern: torch.cat([tensor.T, tensor], dim=1)
//============================================================================

TEST_F(OperationPrologueTest, SameTensorMixedTransform) {
  // This test verifies the slot-based address mapping where the same tensor appears
  // multiple times with mixed transformations - one transformed, one direct.
  // The prologue must correctly handle:
  // - One transformation chain for the transposed input
  // - One direct (non-transformed) input
  // Both from the same base tensor address.

  // Create a single base tensor (square matrix for simplicity)
  std::vector<int64_t> base_shape = {128, 128};
  auto base_tensor = CreateNeuronTensor(base_shape, c10::ScalarType::Float);

  // Output shape after transpose: (128, 128) - still square
  std::vector<int64_t> transposed_shape = {128, 128};

  std::vector<int64_t> curr_perm(2);
  std::iota(curr_perm.begin(), curr_perm.end(), 0);  // {0, 1}

  // Create transpose transformation for first input
  TensorTransformation transform0(TransformationType::TRANSPOSE, base_shape, transposed_shape,
                                  "f32", {1, 0}, curr_perm, {1, 0});

  // First input: transformed (transpose)
  // Second input: direct (no transformation)
  // Both point to the same base tensor
  // This simulates: torch.cat([tensor.T, tensor], dim=1)
  std::vector<at::Tensor> inputs = {base_tensor, base_tensor};

  // First input has transformation, second has empty transformation list
  std::vector<std::vector<TensorTransformation>> kernel_transforms = {{transform0}, {}};

  // Extract input pointers and create tensor contexts
  std::vector<void*> input_ptrs;
  std::vector<at::neuron::TensorContext> input_contexts;
  ExtractInputPtrsAndContexts(inputs, input_ptrs, input_contexts);

  // Compute cache key first
  std::string cache_key = OperationPrologue::ComputePrologueCacheKey(
      input_ptrs, input_contexts, "test_concat_same_tensor_mixed", kernel_transforms);

  // Call: Generate prologue with cache key - this exercises STEP 5's merge logic
  // for handling mixed transformed and direct inputs with the same base address
  auto result = OperationPrologue::GeneratePrologue(
      input_ptrs, input_contexts, "test_concat_same_tensor_mixed", kernel_transforms, cache_key);

  // Assertions: Verify successful merge
  ASSERT_TRUE(result.success) << "Merge with mixed transformations should succeed";
  ASSERT_TRUE(result.has_transformations) << "Should detect transformations";

  // With cache_fix.diff changes, inputs without transformations now get NONE tasks
  // Should have 2 tasks: one TRANSPOSE for input 0, one NONE for input 1
  ASSERT_EQ(result.tasks.size(), 2) << "Should have 2 tasks (TRANSPOSE + NONE)";

  // Verify task details
  EXPECT_EQ(result.tasks[0].input_index, 0) << "First task should be for input 0 (transposed)";
  EXPECT_EQ(result.tasks[0].type, TransformationType::TRANSPOSE);
  EXPECT_EQ(result.tasks[0].input_shape, base_shape);
  EXPECT_EQ(result.tasks[0].output_shape, transposed_shape);

  EXPECT_EQ(result.tasks[1].input_index, 1) << "Second task should be for input 1 (NONE)";
  EXPECT_EQ(result.tasks[1].type, TransformationType::NONE);

  // Verify MLIR was generated
  ASSERT_FALSE(result.prologue_mlir_str.empty()) << "MLIR should be generated";

  // MLIR should contain transpose operation for the transformed input
  EXPECT_TRUE(Contains(result.prologue_mlir_str, "stablehlo.transpose"))
      << "MLIR should contain transpose operation";

  // Verify cache key was generated
  EXPECT_FALSE(result.prologue_cache_key.empty()) << "Cache key should be generated";

  // Verify input_was_transformed flags
  // Note: With cache_fix.diff, inputs without transformations now get NONE transformation tasks.
  // Even though NONE is semantically an identity operation (not really a transformation),
  // the implementation currently marks these as "transformed" in the input_was_transformed flag.
  EXPECT_EQ(result.input_was_transformed.size(), 2);
  EXPECT_TRUE(result.input_was_transformed[0])
      << "Input 0 should be marked as transformed (TRANSPOSE)";
  EXPECT_TRUE(result.input_was_transformed[1])
      << "Input 1 is marked as transformed (has NONE task)";

  // Verify that input 1's transformation is actually NONE (identity operation)
  EXPECT_EQ(result.tasks[1].type, TransformationType::NONE)
      << "Input 1 should have NONE transformation type (not a real transformation)";
}

//============================================================================
// Test 7: Three Independent Transformation Chains (native_multi_head_attn_mid scenario)
// This tests the cumulative mapping logic when merging 3 independent chains
//============================================================================

TEST_F(OperationPrologueTest, ThreeIndependentTransformationChains) {
  // This test simulates the native_multi_head_attn_mid scenario where 3 independent
  // tensors each have a transpose transformation that needs to be merged.
  // This specifically tests the cumulative tensor tracking logic that uses the
  // mapping from each merge step to correctly build up the final tensor list.

  // Create 3 separate tensors with the same shape (simulating Q, K, V tensors)
  std::vector<int64_t> shape = {1, 8, 768, 128};

  auto tensor0 = CreateNeuronTensor(shape, c10::ScalarType::Float);
  auto tensor1 = CreateNeuronTensor(shape, c10::ScalarType::Float);
  auto tensor2 = CreateNeuronTensor(shape, c10::ScalarType::Float);

  // Each tensor has the same transpose: [1, 8, 768, 128] -> [1, 8, 128, 768]
  std::vector<int64_t> output_shape = {1, 8, 128, 768};
  std::vector<int64_t> permutation = {0, 1, 3, 2};

  std::vector<int64_t> curr_perm(4);
  std::iota(curr_perm.begin(), curr_perm.end(), 0);  // {0, 1, 2, 3}

  TensorTransformation transform0(TransformationType::TRANSPOSE, shape, output_shape, "f32",
                                  permutation, curr_perm, permutation);
  TensorTransformation transform1(TransformationType::TRANSPOSE, shape, output_shape, "f32",
                                  permutation, curr_perm, permutation);
  TensorTransformation transform2(TransformationType::TRANSPOSE, shape, output_shape, "f32",
                                  permutation, curr_perm, permutation);

  std::vector<at::Tensor> inputs = {tensor0, tensor1, tensor2};
  std::vector<std::vector<TensorTransformation>> kernel_transforms = {
      {transform0}, {transform1}, {transform2}};

  // Extract input pointers and create tensor contexts
  std::vector<void*> input_ptrs;
  std::vector<at::neuron::TensorContext> input_contexts;
  ExtractInputPtrsAndContexts(inputs, input_ptrs, input_contexts);

  // Compute cache key first
  std::string cache_key = OperationPrologue::ComputePrologueCacheKey(
      input_ptrs, input_contexts, "test_three_chains", kernel_transforms);

  // Call: Generate prologue with cache key - this exercises the cumulative mapping logic
  auto result = OperationPrologue::GeneratePrologue(input_ptrs, input_contexts, "test_three_chains",
                                                    kernel_transforms, cache_key);

  // Assertions: Verify successful merge with cumulative tensor tracking
  ASSERT_TRUE(result.success) << "Three-chain merge should succeed";
  ASSERT_TRUE(result.has_transformations) << "Should detect transformations";

  // Should have 3 tasks, one for each input's transformation
  ASSERT_EQ(result.tasks.size(), 3) << "Should have 3 transformation tasks";

  // Verify each task
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(result.tasks[i].input_index, i) << "Task " << i << " should be for input " << i;
    EXPECT_EQ(result.tasks[i].type, TransformationType::TRANSPOSE);
    EXPECT_EQ(result.tasks[i].input_shape, shape);
    EXPECT_EQ(result.tasks[i].output_shape, output_shape);
  }

  // Verify MLIR was generated
  ASSERT_FALSE(result.prologue_mlir_str.empty()) << "MLIR should be generated";

  // MLIR should contain transpose operations
  EXPECT_TRUE(Contains(result.prologue_mlir_str, "stablehlo.transpose"))
      << "MLIR should contain transpose operations";

  // Verify cache key was generated
  EXPECT_FALSE(result.prologue_cache_key.empty()) << "Cache key should be generated";

  // Verify input_was_transformed flags
  EXPECT_EQ(result.input_was_transformed.size(), 3);
  EXPECT_TRUE(result.input_was_transformed[0]) << "Input 0 should be marked as transformed";
  EXPECT_TRUE(result.input_was_transformed[1]) << "Input 1 should be marked as transformed";
  EXPECT_TRUE(result.input_was_transformed[2]) << "Input 2 should be marked as transformed";
}

}  // namespace
}  // namespace lazy
}  // namespace c10_neuron
