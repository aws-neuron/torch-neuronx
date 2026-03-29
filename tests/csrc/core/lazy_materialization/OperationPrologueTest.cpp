#include <ATen/ATen.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/Storage.h>
#include <c10/util/intrusive_ptr.h>
#include <gtest/gtest.h>

#include <numeric>
#include <string>
#include <unordered_map>
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

// Helper function to check if a string starts with a prefix
bool StartsWith(const std::string& str, const std::string& prefix) {
  return str.rfind(prefix, 0) == 0;
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

// Helper to compute input deduplication map from pointers and contexts
void ComputeDedup(const std::vector<void*>& input_ptrs,
                  const std::vector<at::neuron::TensorContext>& input_contexts,
                  std::vector<size_t>& dedup_map, size_t& deduped_count) {
  std::unordered_map<uintptr_t, size_t> base_to_idx;
  dedup_map.resize(input_ptrs.size());
  deduped_count = 0;
  for (size_t i = 0; i < input_ptrs.size(); ++i) {
    uintptr_t base =
        reinterpret_cast<uintptr_t>(input_ptrs[i]) -
        (input_contexts[i].get_storage_offset() * input_contexts[i].get_element_size());
    auto it = base_to_idx.find(base);
    if (it != base_to_idx.end()) {
      dedup_map[i] = it->second;
    } else {
      base_to_idx[base] = deduped_count;
      dedup_map[i] = deduped_count;
      deduped_count++;
    }
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

// Helper to create a view tensor sharing storage with a base tensor at a given element offset.
// The view has shape `view_shape` and points into the same storage as `base` at `element_offset`.
at::Tensor CreateNeuronView(const at::Tensor& base, const std::vector<int64_t>& view_shape,
                            int64_t element_offset) {
  // Share storage with base — get a copy then move it
  c10::Storage storage = base.storage();

  auto tensor_impl = c10::make_intrusive<c10_neuron::NeuronTensorImpl>(
      std::move(storage), c10::scalarTypeToTypeMeta(base.scalar_type()));

  tensor_impl->set_sizes_contiguous(view_shape);
  tensor_impl->set_storage_offset(element_offset);

  return at::Tensor(tensor_impl);
}

// Count the number of function arguments (%argN) in the MLIR func signature line.
int CountMlirFuncArgs(const std::string& mlir) {
  // Find the func signature line and count %arg occurrences in it
  auto func_pos = mlir.find("func.func");
  if (func_pos == std::string::npos) return -1;
  auto paren_end = mlir.find(")", func_pos);
  if (paren_end == std::string::npos) return -1;
  std::string sig = mlir.substr(func_pos, paren_end - func_pos);
  int count = 0;
  size_t pos = 0;
  while ((pos = sig.find("%arg", pos)) != std::string::npos) {
    count++;
    pos += 4;
  }
  return count;
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
  std::vector<size_t> dm1;
  size_t dc1;
  ComputeDedup(input_ptrs, input_contexts, dm1, dc1);
  auto result = OperationPrologue::GeneratePrologue(input_ptrs, input_contexts, "test_op",
                                                    kernel_transforms, cache_key, dm1, dc1);

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
  std::vector<size_t> dm2;
  size_t dc2;
  ComputeDedup(input_ptrs, input_contexts, dm2, dc2);
  auto result = OperationPrologue::GeneratePrologue(input_ptrs, input_contexts, "test_complex_op",
                                                    kernel_transforms, cache_key, dm2, dc2);

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
  std::vector<size_t> dm3;
  size_t dc3;
  ComputeDedup(input_ptrs, input_contexts, dm3, dc3);
  auto result = OperationPrologue::GeneratePrologue(input_ptrs, input_contexts, "test_multi_input",
                                                    kernel_transforms, cache_key, dm3, dc3);

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
  EXPECT_TRUE(StartsWith(result.prologue_cache_key, "prologue--"))
      << "Cache key should start with prologue prefix";
  EXPECT_TRUE(Contains(result.prologue_cache_key, "i0_"))
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
  std::vector<size_t> dm4;
  size_t dc4;
  ComputeDedup(input_ptrs1, input_contexts1, dm4, dc4);
  auto result1 = OperationPrologue::GeneratePrologue(input_ptrs1, input_contexts1, "cache_test_op",
                                                     kernel_transforms, cache_key1, dm4, dc4);

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
  std::vector<size_t> dm5;
  size_t dc5;
  ComputeDedup(input_ptrs2, input_contexts2, dm5, dc5);
  auto result2 = OperationPrologue::GeneratePrologue(input_ptrs2, input_contexts2, "cache_test_op",
                                                     kernel_transforms, cache_key2, dm5, dc5);

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

  std::vector<size_t> dm6;
  size_t dc6;
  ComputeDedup(input_ptrs, input_contexts, dm6, dc6);
  auto result = OperationPrologue::GeneratePrologue(input_ptrs, input_contexts, "empty_test", {},
                                                    cache_key, dm6, dc6);

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

  std::vector<size_t> dm7;
  size_t dc7;
  ComputeDedup(input_ptrs, input_contexts, dm7, dc7);
  auto result = OperationPrologue::GeneratePrologue(input_ptrs, input_contexts, "no_transform_test",
                                                    {}, cache_key, dm7, dc7);

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

  std::vector<size_t> dm8;
  size_t dc8;
  ComputeDedup(input_ptrs, input_contexts, dm8, dc8);
  auto result = OperationPrologue::GeneratePrologue(input_ptrs, input_contexts, "mixed_test",
                                                    kernel_transforms, cache_key, dm8, dc8);

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
  std::vector<size_t> dm9;
  size_t dc9;
  ComputeDedup(input_ptrs, input_contexts, dm9, dc9);
  auto result = OperationPrologue::GeneratePrologue(
      input_ptrs, input_contexts, "test_two_chains_merge", kernel_transforms, cache_key, dm9, dc9);

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
  std::vector<size_t> dm10;
  size_t dc10;
  ComputeDedup(input_ptrs, input_contexts, dm10, dc10);
  auto result = OperationPrologue::GeneratePrologue(input_ptrs, input_contexts,
                                                    "test_concat_same_tensor_mixed",
                                                    kernel_transforms, cache_key, dm10, dc10);

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
  std::vector<size_t> dm11;
  size_t dc11;
  ComputeDedup(input_ptrs, input_contexts, dm11, dc11);
  auto result = OperationPrologue::GeneratePrologue(input_ptrs, input_contexts, "test_three_chains",
                                                    kernel_transforms, cache_key, dm11, dc11);

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

//============================================================================
// Input Deduplication Tests
// Validates that inputs sharing the same underlying storage are deduplicated
// into a single MLIR function argument during prologue generation.
//============================================================================

TEST_F(OperationPrologueTest, Dedup_AllSameStorage) {
  // 4 slice views from one [16,64] tensor → should dedup to 1 input
  auto base = CreateNeuronTensor({16, 64}, c10::ScalarType::Float);
  std::vector<int64_t> view_shape = {16, 16};

  auto v0 = CreateNeuronView(base, view_shape, 0);
  auto v1 = CreateNeuronView(base, view_shape, 16);
  auto v2 = CreateNeuronView(base, view_shape, 32);
  auto v3 = CreateNeuronView(base, view_shape, 48);

  std::vector<at::Tensor> inputs = {v0, v1, v2, v3};
  std::vector<void*> input_ptrs;
  std::vector<at::neuron::TensorContext> input_contexts;
  ExtractInputPtrsAndContexts(inputs, input_ptrs, input_contexts);

  // Each input has a SLICE transformation from [16,64] → [16,16]
  std::vector<std::vector<TensorTransformation>> kernel_transforms = {
      {TensorTransformation(TransformationType::SLICE, {16, 64}, {16, 16}, "f32", {0, 16, 0, 16})},
      {TensorTransformation(TransformationType::SLICE, {16, 64}, {16, 16}, "f32", {0, 16, 16, 32})},
      {TensorTransformation(TransformationType::SLICE, {16, 64}, {16, 16}, "f32", {0, 16, 32, 48})},
      {TensorTransformation(TransformationType::SLICE, {16, 64}, {16, 16}, "f32", {0, 16, 48, 64})},
  };

  auto cache_key = OperationPrologue::ComputePrologueCacheKey(input_ptrs, input_contexts,
                                                              "aten::cat", kernel_transforms);
  std::vector<size_t> dm12;
  size_t dc12;
  ComputeDedup(input_ptrs, input_contexts, dm12, dc12);
  auto result = OperationPrologue::GeneratePrologue(input_ptrs, input_contexts, "aten::cat",
                                                    kernel_transforms, cache_key, dm12, dc12);

  ASSERT_TRUE(result.success);
  ASSERT_TRUE(result.has_transformations);
  EXPECT_EQ(result.deduped_input_count, 1) << "All 4 views share storage, should dedup to 1";
  ASSERT_EQ(result.input_dedup_map.size(), 4);
  EXPECT_EQ(result.input_dedup_map[0], 0);
  EXPECT_EQ(result.input_dedup_map[1], 0);
  EXPECT_EQ(result.input_dedup_map[2], 0);
  EXPECT_EQ(result.input_dedup_map[3], 0);
  EXPECT_EQ(CountMlirFuncArgs(result.prologue_mlir_str), 1) << "MLIR should have 1 %arg. Got:\n"
                                                            << result.prologue_mlir_str;
}

TEST_F(OperationPrologueTest, Dedup_InterleavedTwoTensors) {
  // a0, b0, a1, b1 from two different tensors → should dedup to 2 inputs
  auto baseA = CreateNeuronTensor({16, 64}, c10::ScalarType::Float);
  auto baseB = CreateNeuronTensor({16, 64}, c10::ScalarType::Float);
  std::vector<int64_t> view_shape = {16, 16};

  auto a0 = CreateNeuronView(baseA, view_shape, 0);
  auto b0 = CreateNeuronView(baseB, view_shape, 0);
  auto a1 = CreateNeuronView(baseA, view_shape, 16);
  auto b1 = CreateNeuronView(baseB, view_shape, 16);

  std::vector<at::Tensor> inputs = {a0, b0, a1, b1};
  std::vector<void*> input_ptrs;
  std::vector<at::neuron::TensorContext> input_contexts;
  ExtractInputPtrsAndContexts(inputs, input_ptrs, input_contexts);

  std::vector<std::vector<TensorTransformation>> kernel_transforms = {
      {TensorTransformation(TransformationType::SLICE, {16, 64}, {16, 16}, "f32", {0, 16, 0, 16})},
      {TensorTransformation(TransformationType::SLICE, {16, 64}, {16, 16}, "f32", {0, 16, 0, 16})},
      {TensorTransformation(TransformationType::SLICE, {16, 64}, {16, 16}, "f32", {0, 16, 16, 32})},
      {TensorTransformation(TransformationType::SLICE, {16, 64}, {16, 16}, "f32", {0, 16, 16, 32})},
  };

  auto cache_key = OperationPrologue::ComputePrologueCacheKey(input_ptrs, input_contexts,
                                                              "aten::cat", kernel_transforms);
  std::vector<size_t> dm13;
  size_t dc13;
  ComputeDedup(input_ptrs, input_contexts, dm13, dc13);
  auto result = OperationPrologue::GeneratePrologue(input_ptrs, input_contexts, "aten::cat",
                                                    kernel_transforms, cache_key, dm13, dc13);

  ASSERT_TRUE(result.success);
  EXPECT_EQ(result.deduped_input_count, 2) << "Two tensors, should dedup to 2";
  EXPECT_EQ(result.input_dedup_map[0], 0);  // a0 → slot 0
  EXPECT_EQ(result.input_dedup_map[1], 1);  // b0 → slot 1
  EXPECT_EQ(result.input_dedup_map[2], 0);  // a1 → slot 0
  EXPECT_EQ(result.input_dedup_map[3], 1);  // b1 → slot 1
  EXPECT_EQ(CountMlirFuncArgs(result.prologue_mlir_str), 2) << "MLIR should have 2 %args. Got:\n"
                                                            << result.prologue_mlir_str;
}

TEST_F(OperationPrologueTest, Dedup_NoSharedStorage) {
  // 4 independent tensors — no dedup
  auto t0 = CreateNeuronTensor({16, 16}, c10::ScalarType::Float);
  auto t1 = CreateNeuronTensor({16, 16}, c10::ScalarType::Float);
  auto t2 = CreateNeuronTensor({16, 16}, c10::ScalarType::Float);
  auto t3 = CreateNeuronTensor({16, 16}, c10::ScalarType::Float);

  std::vector<at::Tensor> inputs = {t0, t1, t2, t3};
  std::vector<void*> input_ptrs;
  std::vector<at::neuron::TensorContext> input_contexts;
  ExtractInputPtrsAndContexts(inputs, input_ptrs, input_contexts);

  // NONE transforms (identity) — no kernel transforms
  std::vector<std::vector<TensorTransformation>> kernel_transforms = {{}, {}, {}, {}};

  auto cache_key = OperationPrologue::ComputePrologueCacheKey(input_ptrs, input_contexts,
                                                              "aten::cat", kernel_transforms);
  std::vector<size_t> dm14;
  size_t dc14;
  ComputeDedup(input_ptrs, input_contexts, dm14, dc14);
  auto result = OperationPrologue::GeneratePrologue(input_ptrs, input_contexts, "aten::cat",
                                                    kernel_transforms, cache_key, dm14, dc14);

  ASSERT_TRUE(result.success);
  EXPECT_EQ(result.deduped_input_count, 4) << "All different storage, no dedup";
  EXPECT_EQ(CountMlirFuncArgs(result.prologue_mlir_str), 4) << "MLIR should have 4 %args. Got:\n"
                                                            << result.prologue_mlir_str;
}

TEST_F(OperationPrologueTest, Dedup_MixedSharedAndIndependent) {
  // other, a0, a1, a2 — 1 independent + 3 from same storage → dedup to 2
  auto base = CreateNeuronTensor({16, 64}, c10::ScalarType::Float);
  auto other = CreateNeuronTensor({16, 16}, c10::ScalarType::Float);
  std::vector<int64_t> view_shape = {16, 16};

  auto a0 = CreateNeuronView(base, view_shape, 0);
  auto a1 = CreateNeuronView(base, view_shape, 16);
  auto a2 = CreateNeuronView(base, view_shape, 32);

  std::vector<at::Tensor> inputs = {other, a0, a1, a2};
  std::vector<void*> input_ptrs;
  std::vector<at::neuron::TensorContext> input_contexts;
  ExtractInputPtrsAndContexts(inputs, input_ptrs, input_contexts);

  std::vector<std::vector<TensorTransformation>> kernel_transforms = {
      {},  // other: no transform (NONE)
      {TensorTransformation(TransformationType::SLICE, {16, 64}, {16, 16}, "f32", {0, 16, 0, 16})},
      {TensorTransformation(TransformationType::SLICE, {16, 64}, {16, 16}, "f32", {0, 16, 16, 32})},
      {TensorTransformation(TransformationType::SLICE, {16, 64}, {16, 16}, "f32", {0, 16, 32, 48})},
  };

  auto cache_key = OperationPrologue::ComputePrologueCacheKey(input_ptrs, input_contexts,
                                                              "aten::cat", kernel_transforms);
  std::vector<size_t> dm15;
  size_t dc15;
  ComputeDedup(input_ptrs, input_contexts, dm15, dc15);
  auto result = OperationPrologue::GeneratePrologue(input_ptrs, input_contexts, "aten::cat",
                                                    kernel_transforms, cache_key, dm15, dc15);

  ASSERT_TRUE(result.success);
  EXPECT_EQ(result.deduped_input_count, 2) << "1 independent + 3 shared = 2 deduped";
  EXPECT_EQ(result.input_dedup_map[0], 0);  // other → slot 0
  EXPECT_EQ(result.input_dedup_map[1], 1);  // a0 → slot 1
  EXPECT_EQ(result.input_dedup_map[2], 1);  // a1 → slot 1
  EXPECT_EQ(result.input_dedup_map[3], 1);  // a2 → slot 1
  EXPECT_EQ(CountMlirFuncArgs(result.prologue_mlir_str), 2) << "MLIR should have 2 %args. Got:\n"
                                                            << result.prologue_mlir_str;
}

}  // namespace
}  // namespace lazy
}  // namespace c10_neuron
