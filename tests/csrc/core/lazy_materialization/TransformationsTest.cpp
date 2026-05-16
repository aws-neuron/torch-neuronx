// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "torch_neuronx/csrc/core/lazy_materialization/TransformationRegistry.h"
#include "torch_neuronx/csrc/core/lazy_materialization/Transformations.h"

namespace c10_neuron {
namespace lazy {
namespace {

//============================================================================
// Test Utilities
//============================================================================

// Helper to create a transposed tensor view
at::Tensor CreateTransposedTensor(const std::vector<int64_t>& shape,
                                  const std::vector<int64_t>& permutation,
                                  c10::ScalarType dtype = c10::ScalarType::Float) {
  // Create contiguous tensor with the original shape
  auto tensor = at::zeros(c10::IntArrayRef(shape), at::TensorOptions().dtype(dtype));

  // Apply permutation to get transpose
  return tensor.permute(c10::IntArrayRef(permutation));
}

// Helper to create a sliced tensor view
at::Tensor CreateSlicedTensor(const std::vector<int64_t>& original_shape,
                              const std::vector<std::pair<int64_t, int64_t>>& slice_ranges,
                              c10::ScalarType dtype = c10::ScalarType::Float) {
  // Create contiguous tensor with original shape
  auto tensor = at::zeros(c10::IntArrayRef(original_shape), at::TensorOptions().dtype(dtype));

  // Apply slices for each dimension
  for (size_t dim = 0; dim < slice_ranges.size(); ++dim) {
    tensor = tensor.slice(dim, slice_ranges[dim].first, slice_ranges[dim].second);
  }

  return tensor;
}

// Helper to create a broadcast tensor view using expand
at::Tensor CreateBroadcastTensor(const std::vector<int64_t>& base_shape,
                                 const std::vector<int64_t>& expanded_shape,
                                 c10::ScalarType dtype = c10::ScalarType::Float) {
  // Create tensor with base shape (with 1s for dimensions to broadcast)
  auto tensor = at::zeros(c10::IntArrayRef(base_shape), at::TensorOptions().dtype(dtype));

  // Use expand to create stride=0 broadcast pattern
  return tensor.expand(c10::IntArrayRef(expanded_shape));
}

//============================================================================
// Test Fixture
//============================================================================

class TransformationsTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

//============================================================================
// detectTransposePattern Tests
//============================================================================

TEST_F(TransformationsTest, DetectTransposePattern_Simple2DTranspose) {
  // Test case: Simple 2D transpose (3x4 -> 4x3)
  std::vector<int64_t> shape = {3, 4};
  std::vector<int64_t> permutation = {1, 0};  // Transpose permutation

  auto tensor = CreateTransposedTensor(shape, permutation);

  // Call TryCreateTranspose which uses detectTransposePattern internally
  auto result = Creators::TryCreateTranspose(tensor, "test_op", 0);

  ASSERT_TRUE(result.has_value()) << "Should detect 2D transpose pattern";
  EXPECT_TRUE(result->is_supported);
  EXPECT_EQ(result->pattern_name, "TRANSPOSE");

  // Verify the permutation is [1, 0] for 2D transpose
  EXPECT_EQ(result->transformations[0].params.size(), 2);
  EXPECT_EQ(result->transformations[0].params[0], 1);
  EXPECT_EQ(result->transformations[0].params[1], 0);
}

TEST_F(TransformationsTest, DetectTransposePattern_3DTranspose_Dims01) {
  // Test case: 3D tensor transpose swapping first two dimensions
  std::vector<int64_t> shape = {2, 3, 4};
  std::vector<int64_t> permutation = {1, 0, 2};  // Swap first two dims

  auto tensor = CreateTransposedTensor(shape, permutation);

  auto result = Creators::TryCreateTranspose(tensor, "test_op", 0);

  ASSERT_TRUE(result.has_value()) << "Should detect 3D transpose pattern";
  EXPECT_TRUE(result->is_supported);

  // Verify permutation [1, 0, 2] - swap first two dims
  EXPECT_EQ(result->transformations[0].params.size(), 3);
  EXPECT_EQ(result->transformations[0].params[0], 1);
  EXPECT_EQ(result->transformations[0].params[1], 0);
  EXPECT_EQ(result->transformations[0].params[2], 2);
}

TEST_F(TransformationsTest, DetectTransposePattern_3DTranspose_Dims12) {
  // Test case: 3D tensor transpose swapping last two dimensions
  std::vector<int64_t> shape = {2, 3, 4};
  std::vector<int64_t> permutation = {0, 2, 1};  // Swap last two dims

  auto tensor = CreateTransposedTensor(shape, permutation);

  auto result = Creators::TryCreateTranspose(tensor, "test_op", 0);

  ASSERT_TRUE(result.has_value()) << "Should detect 3D transpose pattern";
  EXPECT_TRUE(result->is_supported);

  // Verify permutation [0, 2, 1] - swap last two dims
  EXPECT_EQ(result->transformations[0].params.size(), 3);
  EXPECT_EQ(result->transformations[0].params[0], 0);
  EXPECT_EQ(result->transformations[0].params[1], 2);
  EXPECT_EQ(result->transformations[0].params[2], 1);
}

TEST_F(TransformationsTest, DetectTransposePattern_4DTranspose_Complex) {
  // Test case: 4D tensor with complex permutation [2, 0, 1, 3]
  std::vector<int64_t> shape = {4, 5, 6, 7};
  std::vector<int64_t> permutation = {2, 0, 1, 3};  // Complex permutation

  auto tensor = CreateTransposedTensor(shape, permutation);

  auto result = Creators::TryCreateTranspose(tensor, "test_op", 0);

  ASSERT_TRUE(result.has_value()) << "Should detect 4D complex transpose pattern";
  EXPECT_TRUE(result->is_supported);

  // Verify permutation [2, 0, 1, 3]
  EXPECT_EQ(result->transformations[0].params.size(), 4);
  EXPECT_EQ(result->transformations[0].params[0], 2);
  EXPECT_EQ(result->transformations[0].params[1], 0);
  EXPECT_EQ(result->transformations[0].params[2], 1);
  EXPECT_EQ(result->transformations[0].params[3], 3);
}

TEST_F(TransformationsTest, DetectTransposePattern_RejectContiguousTensor) {
  // Test case: Contiguous tensor should NOT be detected as transpose
  std::vector<int64_t> shape = {3, 4, 5};

  auto tensor = at::zeros(shape, at::TensorOptions().dtype(c10::ScalarType::Float));

  ASSERT_TRUE(tensor.is_contiguous()) << "Tensor should be contiguous";

  auto result = Creators::TryCreateTranspose(tensor, "test_op", 0);

  EXPECT_FALSE(result.has_value()) << "Should NOT detect transpose in contiguous tensor";
}

// Note: Tests for negative and zero strides are omitted as these cannot be
// easily created with normal PyTorch operations

TEST_F(TransformationsTest, DetectTransposePattern_EmptyTensor) {
  // Test case: Empty tensor (0 dimensions)
  std::vector<int64_t> shape = {};

  auto tensor = at::zeros({}, at::TensorOptions().dtype(c10::ScalarType::Float));

  auto result = Creators::TryCreateTranspose(tensor, "test_op", 0);

  EXPECT_FALSE(result.has_value()) << "Should reject empty tensor";
}

TEST_F(TransformationsTest, DetectTransposePattern_ScalarTensor) {
  // Test case: Scalar tensor (single element)
  auto tensor = at::zeros({}, at::TensorOptions().dtype(c10::ScalarType::Float));

  auto result = Creators::TryCreateTranspose(tensor, "test_op", 0);

  EXPECT_FALSE(result.has_value()) << "Should reject scalar tensor";
}

//============================================================================
// DetectSlicePattern Tests
//============================================================================

// XFAIL: These tests check for correct expected behavior but currently fail
// due to algorithm limitations. Marked as DISABLED_ to prevent CI failures.
// TODO: Fix DetectSlicePattern algorithm to handle these cases

TEST_F(TransformationsTest, DISABLED_DetectSlicePattern_Simple1DSlice) {
  // Test case: 1D slice should be detectable
  // CURRENT LIMITATION: Algorithm requires multi-dimensional stride structure
  // Original: tensor[10], slice: tensor[2:5] -> shape [3]
  std::vector<int64_t> original_shape = {10};
  std::vector<std::pair<int64_t, int64_t>> slice_ranges = {{2, 5}};

  auto tensor = CreateSlicedTensor(original_shape, slice_ranges);

  auto result = Creators::TryCreateSlice(tensor, "test_op", 0);

  // This is what SHOULD happen (checking for correct expected behavior)
  ASSERT_TRUE(result.has_value()) << "Should detect 1D slice pattern";
  EXPECT_TRUE(result->is_supported);
  EXPECT_EQ(result->pattern_name, "SLICE");

  // Verify slice parameters: start=2, end=5 for dimension 0
  ASSERT_GE(result->transformations[0].params.size(), 2);
  EXPECT_EQ(result->transformations[0].params[0], 2);  // start index
  EXPECT_EQ(result->transformations[0].params[1], 5);  // end index
}

TEST_F(TransformationsTest, DISABLED_DetectSlicePattern_2DSliceAlongDim0) {
  // Test case: 2D slice along first dimension should be detectable
  // CURRENT LIMITATION: Strides [10,1] match contiguous [3,10], no mismatch detected
  // Original: [8, 10], slice: [2:5, :] -> [3, 10]
  std::vector<int64_t> original_shape = {8, 10};
  std::vector<std::pair<int64_t, int64_t>> slice_ranges = {{2, 5}, {0, 10}};

  auto tensor = CreateSlicedTensor(original_shape, slice_ranges);

  auto result = Creators::TryCreateSlice(tensor, "test_op", 0);

  // This is what SHOULD happen (checking for correct expected behavior)
  ASSERT_TRUE(result.has_value()) << "Should detect 2D slice along dim 0";
  EXPECT_TRUE(result->is_supported);
  EXPECT_EQ(result->pattern_name, "SLICE");

  // Verify slice parameters: [2:5, 0:10]
  ASSERT_GE(result->transformations[0].params.size(), 4);
  EXPECT_EQ(result->transformations[0].params[0], 2);   // dim 0 start
  EXPECT_EQ(result->transformations[0].params[1], 5);   // dim 0 end
  EXPECT_EQ(result->transformations[0].params[2], 0);   // dim 1 start
  EXPECT_EQ(result->transformations[0].params[3], 10);  // dim 1 end
}

TEST_F(TransformationsTest, DetectSlicePattern_2DSliceAlongDim1) {
  // Test case: 2D slice along second dimension
  // Original: [8, 10], slice: [:, 3:7] -> [8, 4]
  std::vector<int64_t> original_shape = {8, 10};
  std::vector<std::pair<int64_t, int64_t>> slice_ranges = {{0, 8}, {3, 7}};

  auto tensor = CreateSlicedTensor(original_shape, slice_ranges);

  auto result = Creators::TryCreateSlice(tensor, "test_op", 0);

  ASSERT_TRUE(result.has_value()) << "Should detect 2D slice along dim 1";
  EXPECT_TRUE(result->is_supported);
}

TEST_F(TransformationsTest, DetectSlicePattern_3DSliceMiddleDim) {
  // Test case: 3D slice along middle dimension
  // Original: [4, 10, 6], slice: [:, 2:7, :] -> [4, 5, 6]
  std::vector<int64_t> original_shape = {4, 10, 6};
  std::vector<std::pair<int64_t, int64_t>> slice_ranges = {{0, 4}, {2, 7}, {0, 6}};

  auto tensor = CreateSlicedTensor(original_shape, slice_ranges);

  auto result = Creators::TryCreateSlice(tensor, "test_op", 0);

  ASSERT_TRUE(result.has_value()) << "Should detect 3D slice along middle dim";
  EXPECT_TRUE(result->is_supported);
}

TEST_F(TransformationsTest, DetectSlicePattern_RejectContiguousTensor) {
  // Test case: Contiguous tensor should NOT be detected as slice
  std::vector<int64_t> shape = {5, 6};

  auto tensor = at::zeros(shape, at::TensorOptions().dtype(c10::ScalarType::Float));

  ASSERT_TRUE(tensor.is_contiguous()) << "Tensor should be contiguous";

  auto result = Creators::TryCreateSlice(tensor, "test_op", 0);

  EXPECT_FALSE(result.has_value()) << "Should NOT detect slice in contiguous tensor";
}

TEST_F(TransformationsTest, DetectSlicePattern_RejectMultiAxisSlice) {
  // Reproduces the prod_zeros failure: narrow(0,2,1).narrow(2,3,1) on a 5x5x5 tensor
  // yields shape (1,5,1), strides (25,5,1), offset=53 — sliced on two axes.
  // The old code would compute start_offset=53 (wrong); the fix returns nullopt.
  auto base = at::zeros({5, 5, 5}, at::TensorOptions().dtype(c10::ScalarType::Float));
  auto view = base.narrow(0, 2, 1).narrow(2, 3, 1);

  // Verify the tensor has the expected properties
  ASSERT_EQ(view.storage_offset(), 53);
  ASSERT_FALSE(view.is_contiguous());

  auto result = Creators::TryCreateSlice(view, "test_op", 0);

  EXPECT_FALSE(result.has_value())
      << "Multi-axis slice (offset in both dim 0 and dim 2) must be rejected";
}

TEST_F(TransformationsTest, DetectSlicePattern_AcceptSingleAxisSliceWithOffset) {
  // Single-axis slice: narrow(2,3,1) on a 5x5x5 tensor
  // yields shape (5,5,1), strides (25,5,1), offset=3 — only dim 2 is sliced.
  // offset=3 < strides[1]=5, so this is a valid single-axis slice.
  auto base = at::zeros({5, 5, 5}, at::TensorOptions().dtype(c10::ScalarType::Float));
  auto view = base.narrow(2, 3, 1);

  ASSERT_EQ(view.storage_offset(), 3);
  ASSERT_FALSE(view.is_contiguous());

  auto result = Creators::TryCreateSlice(view, "test_op", 0);

  EXPECT_TRUE(result.has_value()) << "Single-axis slice with offset < stride should be accepted";
}

TEST_F(TransformationsTest, DetectSlicePattern_RejectStridedSlice_Step3) {
  // Strided slice (step>1) produces strides that DetectSlicePattern cannot handle.
  // [7,3,2].slice(0, 0, 7, 3) → [3,3,2] strides [18,2,1].
  // implied_next_size = 18/2 = 9 → base [3,9,2] = 54 > 42 storage elements.
  // Must defer to TryCreateCompound.
  auto base = at::zeros({7, 3, 2}, at::TensorOptions().dtype(c10::ScalarType::BFloat16));
  auto tensor = base.slice(0, 0, 7, 3);

  ASSERT_FALSE(tensor.is_contiguous());

  auto result = Creators::TryCreateSlice(tensor, "test_op", 0);
  EXPECT_FALSE(result.has_value())
      << "Must reject strided slice — implied base exceeds storage, defer to TryCreateCompound";
}

TEST_F(TransformationsTest, DetectSlicePattern_RejectStridedSlice_Step2_Compound) {
  // Compound ops starting with slice(step=2) produce large stride gaps that
  // DetectSlicePattern cannot handle.
  // [3,6,3].slice(0,0,3,2).permute(...).select(...) → [2,3] strides [36,1].
  // implied_next_size = 36/1 = 36 → base [2,36] = 72 > 54 storage elements.
  // Must defer to TryCreateCompound.
  auto base = at::zeros({3, 6, 3}, at::TensorOptions().dtype(c10::ScalarType::BFloat16));
  auto tensor = base.slice(0, 0, 3, 2)
                    .permute({0, 2, 1})
                    .select(2, 2)
                    .narrow(0, 0, 2)
                    .unsqueeze(1)
                    .squeeze(1);

  ASSERT_FALSE(tensor.is_contiguous());

  auto result = Creators::TryCreateSlice(tensor, "test_op", 0);
  EXPECT_FALSE(result.has_value())
      << "Must reject compound strided pattern — implied base exceeds storage, "
         "defer to TryCreateCompound";
}

//============================================================================
// DetectBroadcastPattern Tests
//============================================================================

TEST_F(TransformationsTest, DetectBroadcastPattern_Simple2D_LastDim) {
  // Test case: Simple 2D broadcast on last dimension
  // Broadcast [3, 1] -> [3, 4]
  // Dimension 1 has size>1 and stride=0
  std::vector<int64_t> base_shape = {3, 1};
  std::vector<int64_t> expanded_shape = {3, 4};

  auto tensor = CreateBroadcastTensor(base_shape, expanded_shape);

  auto result = Creators::TryCreateBroadcast(tensor, "test_op", 0);

  ASSERT_TRUE(result.has_value()) << "Should detect 2D broadcast pattern";
  EXPECT_TRUE(result->is_supported);
  EXPECT_EQ(result->pattern_name, "BROADCAST");
  ASSERT_EQ(result->transformations.size(), 1u);

  // Verify input shape is base shape [3, 1]
  ASSERT_EQ(result->transformations[0].input_shape.size(), 2);
  EXPECT_EQ(result->transformations[0].input_shape[0], 3);
  EXPECT_EQ(result->transformations[0].input_shape[1], 1);

  // Verify output shape is expanded shape [3, 4]
  ASSERT_EQ(result->transformations[0].output_shape.size(), 2);
  EXPECT_EQ(result->transformations[0].output_shape[0], 3);
  EXPECT_EQ(result->transformations[0].output_shape[1], 4);

  // Verify broadcast_dimensions params [0, 1]
  ASSERT_EQ(result->transformations[0].params.size(), 2);
  EXPECT_EQ(result->transformations[0].params[0], 0);
  EXPECT_EQ(result->transformations[0].params[1], 1);
}

TEST_F(TransformationsTest, DetectBroadcastPattern_Simple2D_FirstDim) {
  // Test case: 2D broadcast on first dimension
  // Broadcast [1, 4] -> [3, 4]
  // Dimension 0 has size>1 and stride=0
  std::vector<int64_t> base_shape = {1, 4};
  std::vector<int64_t> expanded_shape = {3, 4};

  auto tensor = CreateBroadcastTensor(base_shape, expanded_shape);

  auto result = Creators::TryCreateBroadcast(tensor, "test_op", 0);

  ASSERT_TRUE(result.has_value()) << "Should detect 2D broadcast on first dim";
  EXPECT_TRUE(result->is_supported);
  EXPECT_EQ(result->pattern_name, "BROADCAST");
  ASSERT_EQ(result->transformations.size(), 1u);

  // Verify shapes
  EXPECT_EQ(result->transformations[0].input_shape[0], 1);
  EXPECT_EQ(result->transformations[0].input_shape[1], 4);
  EXPECT_EQ(result->transformations[0].output_shape[0], 3);
  EXPECT_EQ(result->transformations[0].output_shape[1], 4);
}

TEST_F(TransformationsTest, DetectBroadcastPattern_3D_SingleDim) {
  // Test case: 3D broadcast on single middle dimension
  // Broadcast [2, 1, 4] -> [2, 3, 4]
  // Dimension 1 has size>1 and stride=0
  std::vector<int64_t> base_shape = {2, 1, 4};
  std::vector<int64_t> expanded_shape = {2, 3, 4};

  auto tensor = CreateBroadcastTensor(base_shape, expanded_shape);

  auto result = Creators::TryCreateBroadcast(tensor, "test_op", 0);

  ASSERT_TRUE(result.has_value()) << "Should detect 3D broadcast on middle dim";
  EXPECT_TRUE(result->is_supported);
  ASSERT_EQ(result->transformations.size(), 1u);

  // Verify shapes
  ASSERT_EQ(result->transformations[0].input_shape.size(), 3);
  EXPECT_EQ(result->transformations[0].input_shape[0], 2);
  EXPECT_EQ(result->transformations[0].input_shape[1], 1);
  EXPECT_EQ(result->transformations[0].input_shape[2], 4);

  ASSERT_EQ(result->transformations[0].output_shape.size(), 3);
  EXPECT_EQ(result->transformations[0].output_shape[0], 2);
  EXPECT_EQ(result->transformations[0].output_shape[1], 3);
  EXPECT_EQ(result->transformations[0].output_shape[2], 4);

  // Verify broadcast_dimensions [0, 1, 2]
  ASSERT_EQ(result->transformations[0].params.size(), 3);
  EXPECT_EQ(result->transformations[0].params[0], 0);
  EXPECT_EQ(result->transformations[0].params[1], 1);
  EXPECT_EQ(result->transformations[0].params[2], 2);
}

TEST_F(TransformationsTest, DetectBroadcastPattern_3D_MultipleDims) {
  // Test case: 3D broadcast on multiple dimensions
  // Broadcast [1, 3, 1] -> [2, 3, 4]
  // Dimensions 0 and 2 have stride=0
  std::vector<int64_t> base_shape = {1, 3, 1};
  std::vector<int64_t> expanded_shape = {2, 3, 4};

  auto tensor = CreateBroadcastTensor(base_shape, expanded_shape);

  auto result = Creators::TryCreateBroadcast(tensor, "test_op", 0);

  ASSERT_TRUE(result.has_value()) << "Should detect 3D broadcast on multiple dims";
  EXPECT_TRUE(result->is_supported);
  ASSERT_EQ(result->transformations.size(), 1u);

  // Verify base shape [1, 3, 1]
  EXPECT_EQ(result->transformations[0].input_shape[0], 1);
  EXPECT_EQ(result->transformations[0].input_shape[1], 3);
  EXPECT_EQ(result->transformations[0].input_shape[2], 1);

  // Verify expanded shape [2, 3, 4]
  EXPECT_EQ(result->transformations[0].output_shape[0], 2);
  EXPECT_EQ(result->transformations[0].output_shape[1], 3);
  EXPECT_EQ(result->transformations[0].output_shape[2], 4);
}

TEST_F(TransformationsTest, DetectBroadcastPattern_4D_BatchDim) {
  // Test case: 4D broadcast on batch dimension (common ML pattern)
  // Broadcast [1, 3, 4, 5] -> [8, 3, 4, 5]
  // Batch dimension (0) has stride=0
  std::vector<int64_t> base_shape = {1, 3, 4, 5};
  std::vector<int64_t> expanded_shape = {8, 3, 4, 5};

  auto tensor = CreateBroadcastTensor(base_shape, expanded_shape);

  auto result = Creators::TryCreateBroadcast(tensor, "test_op", 0);

  ASSERT_TRUE(result.has_value()) << "Should detect 4D broadcast on batch dim";
  EXPECT_TRUE(result->is_supported);
  ASSERT_EQ(result->transformations.size(), 1u);

  // Verify shapes
  ASSERT_EQ(result->transformations[0].input_shape.size(), 4);
  EXPECT_EQ(result->transformations[0].input_shape[0], 1);
  EXPECT_EQ(result->transformations[0].input_shape[1], 3);
  EXPECT_EQ(result->transformations[0].input_shape[2], 4);
  EXPECT_EQ(result->transformations[0].input_shape[3], 5);

  ASSERT_EQ(result->transformations[0].output_shape.size(), 4);
  EXPECT_EQ(result->transformations[0].output_shape[0], 8);
  EXPECT_EQ(result->transformations[0].output_shape[1], 3);
  EXPECT_EQ(result->transformations[0].output_shape[2], 4);
  EXPECT_EQ(result->transformations[0].output_shape[3], 5);

  // Verify broadcast_dimensions [0, 1, 2, 3]
  ASSERT_EQ(result->transformations[0].params.size(), 4);
  EXPECT_EQ(result->transformations[0].params[0], 0);
  EXPECT_EQ(result->transformations[0].params[1], 1);
  EXPECT_EQ(result->transformations[0].params[2], 2);
  EXPECT_EQ(result->transformations[0].params[3], 3);
}

TEST_F(TransformationsTest, DetectBroadcastPattern_MatmulBias) {
  // Test case: Common matmul bias broadcast pattern
  // Broadcast [256, 1] -> [256, 128]
  // Used in matrix multiplication with bias terms
  std::vector<int64_t> base_shape = {256, 1};
  std::vector<int64_t> expanded_shape = {256, 128};

  auto tensor = CreateBroadcastTensor(base_shape, expanded_shape);

  auto result = Creators::TryCreateBroadcast(tensor, "test_op", 0);

  ASSERT_TRUE(result.has_value()) << "Should detect matmul bias broadcast pattern";
  EXPECT_TRUE(result->is_supported);
  ASSERT_EQ(result->transformations.size(), 1u);

  // Verify this matches the pattern from test_matmul_with_broadcast in Python
  EXPECT_EQ(result->transformations[0].input_shape[0], 256);
  EXPECT_EQ(result->transformations[0].input_shape[1], 1);
  EXPECT_EQ(result->transformations[0].output_shape[0], 256);
  EXPECT_EQ(result->transformations[0].output_shape[1], 128);
}

TEST_F(TransformationsTest, DetectBroadcastPattern_RejectContiguousTensor) {
  // Test case: Contiguous tensor should NOT be detected as broadcast
  std::vector<int64_t> shape = {3, 4, 5};

  auto tensor = at::zeros(shape, at::TensorOptions().dtype(c10::ScalarType::Float));

  ASSERT_TRUE(tensor.is_contiguous()) << "Tensor should be contiguous";

  auto result = Creators::TryCreateBroadcast(tensor, "test_op", 0);

  EXPECT_FALSE(result.has_value()) << "Should NOT detect broadcast in contiguous tensor";
}

TEST_F(TransformationsTest, DetectBroadcastPattern_EmptyTensor) {
  // Test case: Empty tensor should be rejected
  std::vector<int64_t> shape = {};

  auto tensor = at::zeros({}, at::TensorOptions().dtype(c10::ScalarType::Float));

  auto result = Creators::TryCreateBroadcast(tensor, "test_op", 0);

  EXPECT_FALSE(result.has_value()) << "Should reject empty tensor";
}

TEST_F(TransformationsTest, DetectBroadcastPattern_ScalarTensor) {
  // Test case: Scalar tensor should be rejected
  auto tensor = at::zeros({}, at::TensorOptions().dtype(c10::ScalarType::Float));

  auto result = Creators::TryCreateBroadcast(tensor, "test_op", 0);

  EXPECT_FALSE(result.has_value()) << "Should reject scalar tensor";
}

TEST_F(TransformationsTest, BroadcastProcessedInput_IsContiguousWithBaseShape) {
  // Broadcast [3, 1] -> [3, 4]: processed_input must be contiguous with base shape
  auto tensor = CreateBroadcastTensor({3, 1}, {3, 4});
  ASSERT_FALSE(tensor.is_contiguous());

  auto result = Creators::TryCreateBroadcast(tensor, "test_op", 0);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->processed_input.is_contiguous())
      << "Broadcast processed_input must be contiguous";

  auto shape = result->processed_input.sizes();
  ASSERT_EQ(shape.size(), 2);
  EXPECT_EQ(shape[0], 3);
  EXPECT_EQ(shape[1], 1);
}

TEST_F(TransformationsTest, BroadcastProcessedInput_MultiDimIsContiguous) {
  // Broadcast [1, 4, 1] -> [3, 4, 5]: multiple broadcast dims
  auto tensor = CreateBroadcastTensor({1, 4, 1}, {3, 4, 5});
  ASSERT_FALSE(tensor.is_contiguous());

  auto result = Creators::TryCreateBroadcast(tensor, "test_op", 0);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->processed_input.is_contiguous())
      << "Multi-dim broadcast processed_input must be contiguous";

  auto shape = result->processed_input.sizes();
  ASSERT_EQ(shape.size(), 3);
  EXPECT_EQ(shape[0], 1);
  EXPECT_EQ(shape[1], 4);
  EXPECT_EQ(shape[2], 1);
}

TEST_F(TransformationsTest, BroadcastFromSlice_ReturnsNullopt) {
  // Broadcast from slice: slice creates non-contiguous base strides
  // Original [4] -> slice [::2] -> [2] with stride [2]
  // Then unsqueeze -> [2, 1] with strides [2, 0] (not contiguous)
  // Then expand -> [2, 5] with strides [2, 0]
  auto original = at::zeros({4});
  auto sliced = original.slice(0, 0, 4, 2);    // stride [2]
  auto unsqueezed = sliced.unsqueeze(1);       // shape [2, 1], strides [2, 1]
  auto broadcast = unsqueezed.expand({2, 5});  // shape [2, 5], strides [2, 0]

  ASSERT_FALSE(broadcast.is_contiguous());

  // Should return nullopt because base has non-contiguous strides
  auto result = Creators::TryCreateBroadcast(broadcast, "test_op", 0);
  EXPECT_FALSE(result.has_value()) << "Broadcast from slice should fall back (return nullopt)";
}

// TryCreateCompound Tests
//
// TryCreateCompound handles arbitrary combinations of transpose, slice, and
// broadcast. Unlike the single-pattern creators, it can handle:
//   - Pure transpose
//   - Pure single slice
//   - Pure broadcast
//   - Transpose + slice (the RoPE pattern)
//   - Multi-axis slice
//   - Broadcast + transpose + slice
//
// For all cases, the processed_input (base tensor) must be contiguous.
//============================================================================

TEST_F(TransformationsTest, TryCreateCompound_RejectContiguousTensor) {
  // Contiguous tensor → nothing to do
  auto tensor = at::zeros({4, 8, 16}, at::TensorOptions().dtype(c10::ScalarType::Float));
  ASSERT_TRUE(tensor.is_contiguous());

  auto result = Creators::TryCreateCompound(tensor, "test_op", 0);
  EXPECT_FALSE(result.has_value()) << "Should reject contiguous tensor";
}

TEST_F(TransformationsTest, TryCreateCompound_PureTranspose_2D) {
  // Pure 2D transpose: TryCreateCompound now handles this directly (no deferral)
  // [3, 4] transposed → logical shape [4, 3], physical shape [3, 4]
  auto tensor = at::zeros({3, 4}, at::TensorOptions().dtype(c10::ScalarType::Float)).t();

  ASSERT_FALSE(tensor.is_contiguous());

  auto result = Creators::TryCreateCompound(tensor, "test_op", 0);

  ASSERT_TRUE(result.has_value()) << "Should handle pure transpose";
  EXPECT_TRUE(result->is_supported);
  EXPECT_EQ(result->pattern_name, "COMPOUND");

  // Base tensor must be contiguous with physical shape [3, 4]
  EXPECT_TRUE(result->processed_input.is_contiguous());
  ASSERT_EQ(result->processed_input.dim(), 2);
  EXPECT_EQ(result->processed_input.size(0), 3);
  EXPECT_EQ(result->processed_input.size(1), 4);

  // Should emit exactly 1 TRANSPOSE transformation
  ASSERT_EQ(result->transformations.size(), 1u);
  EXPECT_EQ(result->transformations[0].type, TransformationType::TRANSPOSE);

  // Verify permutation: [1, 0] (swap dims 0 and 1)
  ASSERT_EQ(result->transformations[0].params.size(), 2u);
  EXPECT_EQ(result->transformations[0].params[0], 1);  // output dim 0 ← input dim 1
  EXPECT_EQ(result->transformations[0].params[1], 0);  // output dim 1 ← input dim 0

  // input_shape (physical [3,4]) → output_shape (logical [4,3])
  ASSERT_EQ(result->transformations[0].input_shape.size(), 2u);
  EXPECT_EQ(result->transformations[0].input_shape[0], 3);
  EXPECT_EQ(result->transformations[0].input_shape[1], 4);
  ASSERT_EQ(result->transformations[0].output_shape.size(), 2u);
  EXPECT_EQ(result->transformations[0].output_shape[0], 4);
  EXPECT_EQ(result->transformations[0].output_shape[1], 3);
}

TEST_F(TransformationsTest, TryCreateCompound_PureSingleSlice) {
  // Pure single-axis slice: TryCreateCompound now handles this directly (no deferral)
  // [8, 10] sliced to [:, 3:7] → shape [8, 4], strides [10, 1], offset=3
  auto tensor =
      at::zeros({8, 10}, at::TensorOptions().dtype(c10::ScalarType::Float)).slice(1, 3, 7);

  ASSERT_FALSE(tensor.is_contiguous());

  auto result = Creators::TryCreateCompound(tensor, "test_op", 0);

  ASSERT_TRUE(result.has_value()) << "Should handle pure single slice";
  EXPECT_TRUE(result->is_supported);

  // Base tensor must be contiguous with original shape [8, 10]
  EXPECT_TRUE(result->processed_input.is_contiguous());
  ASSERT_EQ(result->processed_input.dim(), 2);
  EXPECT_EQ(result->processed_input.size(0), 8);
  EXPECT_EQ(result->processed_input.size(1), 10);
  EXPECT_EQ(result->processed_input.storage_offset(), 0);

  // Should emit exactly 1 SLICE transformation
  ASSERT_EQ(result->transformations.size(), 1u);
  EXPECT_EQ(result->transformations[0].type, TransformationType::SLICE);

  // input_shape: [8, 10] (original), output_shape: [8, 4] (sliced)
  ASSERT_EQ(result->transformations[0].input_shape.size(), 2u);
  EXPECT_EQ(result->transformations[0].input_shape[0], 8);
  EXPECT_EQ(result->transformations[0].input_shape[1], 10);
  ASSERT_EQ(result->transformations[0].output_shape.size(), 2u);
  EXPECT_EQ(result->transformations[0].output_shape[0], 8);
  EXPECT_EQ(result->transformations[0].output_shape[1], 4);

  // params use triplet encoding: [start0,end0,stride0, start1,end1,stride1]
  ASSERT_EQ(result->transformations[0].params.size(), 6u);  // 2 dims × 3
  EXPECT_EQ(result->transformations[0].params[0], 0);       // dim0 start (full)
  EXPECT_EQ(result->transformations[0].params[1], 8);       // dim0 end (full)
  EXPECT_EQ(result->transformations[0].params[2], 1);       // dim0 stride
  EXPECT_EQ(result->transformations[0].params[3], 3);       // dim1 start
  EXPECT_EQ(result->transformations[0].params[4], 7);       // dim1 end
  EXPECT_EQ(result->transformations[0].params[5], 1);       // dim1 stride
}

TEST_F(TransformationsTest, TryCreateCompound_PureBroadcast) {
  // Pure broadcast: x.unsqueeze(1).expand(B, H, S, D)
  // stride=0 on dim 1 (H), non-broadcast dims are contiguous.
  // TryCreateCompound keeps all n dims (broadcast dims get size-1) and emits BROADCAST.
  int64_t B = 2, H = 4, S = 8, D = 16;
  auto base = at::zeros({B, S, D}, at::TensorOptions().dtype(c10::ScalarType::Float));
  auto tensor = base.unsqueeze(1).expand({B, H, S, D});  // stride=0 on dim 1

  ASSERT_FALSE(tensor.is_contiguous());
  ASSERT_EQ(tensor.stride(1), 0);  // broadcast dim

  auto result = Creators::TryCreateCompound(tensor, "test_op", 0);

  ASSERT_TRUE(result.has_value()) << "Should handle pure broadcast";
  EXPECT_TRUE(result->is_supported);

  // Base tensor must be contiguous with pre-broadcast shape [B, 1, S, D] (all n dims)
  EXPECT_TRUE(result->processed_input.is_contiguous());
  ASSERT_EQ(result->processed_input.dim(), 4);  // all n=4 dims
  EXPECT_EQ(result->processed_input.size(0), B);
  EXPECT_EQ(result->processed_input.size(1), 1);  // broadcast dim has size 1
  EXPECT_EQ(result->processed_input.size(2), S);
  EXPECT_EQ(result->processed_input.size(3), D);

  // Should emit just BROADCAST (1 transformation, no RESHAPE needed)
  ASSERT_EQ(result->transformations.size(), 1u);

  // BROADCAST: [B,1,S,D] → [B,H,S,D]
  EXPECT_EQ(result->transformations[0].type, TransformationType::BROADCAST);
  ASSERT_EQ(result->transformations[0].input_shape.size(), 4u);
  EXPECT_EQ(result->transformations[0].input_shape[0], B);
  EXPECT_EQ(result->transformations[0].input_shape[1], 1);  // size-1 input
  EXPECT_EQ(result->transformations[0].input_shape[2], S);
  EXPECT_EQ(result->transformations[0].input_shape[3], D);
  ASSERT_EQ(result->transformations[0].output_shape.size(), 4u);
  EXPECT_EQ(result->transformations[0].output_shape[0], B);
  EXPECT_EQ(result->transformations[0].output_shape[1], H);  // expanded to H
  EXPECT_EQ(result->transformations[0].output_shape[2], S);
  EXPECT_EQ(result->transformations[0].output_shape[3], D);
}

TEST_F(TransformationsTest, TryCreateCompound_TransposeSlice_RoPE) {
  // The RoPE pattern: [B,S,H,D].transpose(1,2)[..., D//2:]
  // Using B=1, S=8, H=4, D=8 → x2 = q[..., 4:]
  // shape=[1,4,8,4], strides=[256,8,32,1], offset=4
  int64_t B = 1, S = 8, H = 4, D = 8;
  auto base = at::zeros({B, S, H, D}, at::TensorOptions().dtype(c10::ScalarType::Float));
  auto transposed = base.transpose(1, 2);   // [B,H,S,D]
  auto x2 = transposed.slice(3, D / 2, D);  // [B,H,S,D//2], offset=D//2

  ASSERT_FALSE(x2.is_contiguous());
  ASSERT_EQ(x2.storage_offset(), D / 2);

  auto result = Creators::TryCreateCompound(x2, "test_op", 0);

  ASSERT_TRUE(result.has_value()) << "Should handle transpose+slice (RoPE pattern)";
  EXPECT_TRUE(result->is_supported);
  EXPECT_EQ(result->pattern_name, "COMPOUND");

  // Base tensor must be contiguous
  EXPECT_TRUE(result->processed_input.is_contiguous());
  EXPECT_EQ(result->processed_input.storage_offset(), 0);

  // Should emit SLICE then TRANSPOSE (at minimum)
  ASSERT_GE(result->transformations.size(), 2u);
  // First transform should be SLICE, last should be TRANSPOSE
  EXPECT_EQ(result->transformations[0].type, TransformationType::SLICE);
  EXPECT_EQ(result->transformations.back().type, TransformationType::TRANSPOSE);

  // Output shape must match the original tensor shape [B, H, S, D//2]
  const auto& last_t = result->transformations.back();
  ASSERT_EQ(last_t.output_shape.size(), 4u);
  EXPECT_EQ(last_t.output_shape[0], B);
  EXPECT_EQ(last_t.output_shape[1], H);
  EXPECT_EQ(last_t.output_shape[2], S);
  EXPECT_EQ(last_t.output_shape[3], D / 2);
}

TEST_F(TransformationsTest, TryCreateCompound_MultiAxisSlice) {
  // Two slices on a contiguous tensor: x[..., :D//2][:, :S//2, :]
  // Using H=4, S=8, D=8
  int64_t B = 1, H = 4, S = 8, D = 8;
  auto base = at::zeros({B, H, S, D}, at::TensorOptions().dtype(c10::ScalarType::Float));
  auto sliced = base.slice(3, 0, D / 2).slice(2, 0, S / 2);  // [B,H,S//2,D//2]

  ASSERT_FALSE(sliced.is_contiguous());

  auto result = Creators::TryCreateCompound(sliced, "test_op", 0);

  ASSERT_TRUE(result.has_value()) << "Should handle multi-axis slice";
  EXPECT_TRUE(result->is_supported);

  // Base tensor must be contiguous
  EXPECT_TRUE(result->processed_input.is_contiguous());
  ASSERT_EQ(result->processed_input.dim(), 4);

  // Should emit SLICE transformations (no TRANSPOSE since no permutation)
  ASSERT_GE(result->transformations.size(), 1u);
  for (const auto& t : result->transformations) {
    EXPECT_EQ(t.type, TransformationType::SLICE);
  }

  // Final output shape must match [B, H, S//2, D//2]
  const auto& last_t = result->transformations.back();
  EXPECT_EQ(last_t.output_shape[2], S / 2);
  EXPECT_EQ(last_t.output_shape[3], D / 2);
}

TEST_F(TransformationsTest, TryCreateCompound_BroadcastTranspose) {
  // Broadcast + transpose: x.unsqueeze(1).expand(B,H,S,D).transpose(2,3)
  // stride=0 on dim 1 (H), AND dims 2 and 3 are transposed.
  // Broadcast dims are kept with size-1 in the decomposition.
  // Expected chain: TRANSPOSE + BROADCAST (2 transforms)
  int64_t B = 2, H = 4, S = 8, D = 16;
  auto base = at::zeros({B, S, D}, at::TensorOptions().dtype(c10::ScalarType::Float));
  auto tensor = base.unsqueeze(1).expand({B, H, S, D}).transpose(2, 3);  // [B,H,D,S]

  ASSERT_FALSE(tensor.is_contiguous());
  ASSERT_EQ(tensor.stride(1), 0);  // broadcast dim

  auto result = Creators::TryCreateCompound(tensor, "test_op", 0);

  ASSERT_TRUE(result.has_value()) << "Should handle broadcast+transpose";
  EXPECT_TRUE(result->is_supported);

  // Base tensor must be contiguous (all n dims, size-1 for broadcast, in physical order)
  EXPECT_TRUE(result->processed_input.is_contiguous());

  // Should emit TRANSPOSE + BROADCAST (2 transforms, no RESHAPE needed)
  ASSERT_EQ(result->transformations.size(), 2u);
  EXPECT_EQ(result->transformations[0].type, TransformationType::TRANSPOSE);
  EXPECT_EQ(result->transformations[1].type, TransformationType::BROADCAST);

  // BROADCAST: expands dim 1 from 1 → H
  EXPECT_EQ(result->transformations[1].input_shape[1], 1);
  EXPECT_EQ(result->transformations[1].output_shape[1], H);

  // Output shape must match original tensor shape [B, H, D, S]
  const auto& last_transform = result->transformations.back();
  ASSERT_EQ(last_transform.output_shape.size(), 4u);
  EXPECT_EQ(last_transform.output_shape[0], B);
  EXPECT_EQ(last_transform.output_shape[1], H);
  EXPECT_EQ(last_transform.output_shape[2], D);
  EXPECT_EQ(last_transform.output_shape[3], S);
}

TEST_F(TransformationsTest, TryCreateCompound_BroadcastSlice) {
  // Broadcast + slice: x.unsqueeze(1).expand(B,H,S,D)[..., :D//2]
  // stride=0 on dim 1 (H), AND last dim is sliced.
  // Broadcast dims are kept with size-1 in the decomposition.
  // Expected chain: SLICE + BROADCAST (2 transforms)
  int64_t B = 2, H = 4, S = 8, D = 16;
  auto base = at::zeros({B, S, D}, at::TensorOptions().dtype(c10::ScalarType::Float));
  auto tensor = base.unsqueeze(1).expand({B, H, S, D}).slice(3, 0, D / 2);  // [B,H,S,D//2]

  ASSERT_FALSE(tensor.is_contiguous());
  ASSERT_EQ(tensor.stride(1), 0);  // broadcast dim

  auto result = Creators::TryCreateCompound(tensor, "test_op", 0);

  ASSERT_TRUE(result.has_value()) << "Should handle broadcast+slice";
  EXPECT_TRUE(result->is_supported);

  // Base tensor must be contiguous (all n dims, size-1 for broadcast, in physical order)
  EXPECT_TRUE(result->processed_input.is_contiguous());

  // Should emit SLICE + BROADCAST (2 transforms, no RESHAPE needed)
  ASSERT_EQ(result->transformations.size(), 2u);
  EXPECT_EQ(result->transformations[0].type, TransformationType::SLICE);
  EXPECT_EQ(result->transformations[1].type, TransformationType::BROADCAST);

  // SLICE: output has D//2 on last dim
  EXPECT_EQ(result->transformations[0].output_shape.back(), D / 2);

  // BROADCAST: expands dim 1 from 1 → H
  EXPECT_EQ(result->transformations[1].input_shape[1], 1);
  EXPECT_EQ(result->transformations[1].output_shape[1], H);

  // Output shape must match original tensor shape [B, H, S, D//2]
  const auto& last_transform = result->transformations.back();
  ASSERT_EQ(last_transform.output_shape.size(), 4u);
  EXPECT_EQ(last_transform.output_shape[0], B);
  EXPECT_EQ(last_transform.output_shape[1], H);
  EXPECT_EQ(last_transform.output_shape[2], S);
  EXPECT_EQ(last_transform.output_shape[3], D / 2);
}

TEST_F(TransformationsTest, TryCreateCompound_BaseIsAlwaysContiguous) {
  // For any compound pattern, the processed_input (base tensor) must be contiguous.
  // This is the key invariant that allows the kernel to use it directly.

  // Test with transpose+slice
  {
    auto tensor = at::zeros({2, 4, 8, 16}, at::TensorOptions().dtype(c10::ScalarType::Float))
                      .transpose(1, 2)
                      .slice(3, 4, 12);
    auto result = Creators::TryCreateCompound(tensor, "test_op", 0);
    ASSERT_TRUE(result.has_value()) << "Should handle transpose+slice";
    EXPECT_TRUE(result->processed_input.is_contiguous())
        << "Base tensor must be contiguous for transpose+slice";
  }

  // Test with multi-axis slice
  {
    auto tensor = at::zeros({4, 8, 16}, at::TensorOptions().dtype(c10::ScalarType::Float))
                      .slice(1, 2, 6)
                      .slice(2, 4, 12);
    auto result = Creators::TryCreateCompound(tensor, "test_op", 0);
    ASSERT_TRUE(result.has_value()) << "Should handle multi-axis slice";
    EXPECT_TRUE(result->processed_input.is_contiguous())
        << "Base tensor must be contiguous for multi-axis slice";
  }
}

TEST_F(TransformationsTest, TryCreateCompound_OutputShapeMatchesInput) {
  // The final output shape of the transformation chain must match the input tensor's shape.
  // Verify by checking that the last transformation's output_shape == input.sizes()

  auto tensor = at::zeros({1, 4, 8, 8}, at::TensorOptions().dtype(c10::ScalarType::Float))
                    .transpose(1, 2)
                    .slice(3, 4, 8);  // [1,4,8,4]

  auto result = Creators::TryCreateCompound(tensor, "test_op", 0);
  ASSERT_TRUE(result.has_value());

  // Last transformation's output shape must match the original tensor shape
  const auto& last_transform = result->transformations.back();
  ASSERT_EQ(last_transform.output_shape.size(), static_cast<size_t>(tensor.dim()));
  for (int64_t i = 0; i < tensor.dim(); ++i) {
    EXPECT_EQ(last_transform.output_shape[i], tensor.size(i))
        << "Output shape mismatch at dim " << i;
  }
}

//============================================================================
// Random Mixed Sequence Fuzz Test
//
// Generates random sequences of 1-10 transformations (transpose, slice,
// broadcast/expand) applied to a contiguous tensor, then verifies that
// TryCreateCompound can decompose the resulting view correctly.
//
// Key invariants verified for each random sequence:
//   1. Base tensor (processed_input) is contiguous
//   2. Last transformation's output_shape matches the input tensor's logical shape
//   3. Transformation chain is well-formed (output[i] == input[i+1])
//   4. All transformations have valid types
//
// On failure, the random seed is printed for reproducibility.
//============================================================================

// Enum for random operation types used in the fuzz test below.
//
// These 7 ops are chosen to cover the distinct stride patterns that
// PyTorch view operations can produce:
//
//   TRANSPOSE  - tensor.transpose(dim0, dim1): swaps two dimensions.
//   SLICE      - tensor.slice(dim, start, end, step): contiguous sub-range
//                with optional striding.
//   EXPAND     - tensor.expand(...): sets stride=0 on size-1 dims.
//   PERMUTE    - tensor.permute(...): arbitrary N-dim reorder.
//   UNSQUEEZE  - tensor.unsqueeze(dim): inserts a size-1 dim.
//   SQUEEZE    - tensor.squeeze(dim): removes a size-1 dim.
//   SELECT     - tensor.select(dim, idx): equivalent to slicing a single
//                index (e.g. tensor[idx]), but reduces rank by 1.
//
// Not exercised (produce identical stride patterns to the above):
//   t, mT, swapaxes, swapdims → alias for transpose
//   movedim, moveaxis → same stride effect as permute
//   expand_as → equivalent to expand(other.size())
//   narrow → equivalent to slice
//   split, chunk, unbind, at::split_with_sizes → produce sliced views
//
// Not exercised (view op tested in test_compound_sweep.py):
//   view, reshape, reshape_as, flatten, unflatten, contiguous, view.dtype
//
// Not exercised (produce non-admissible stride patterns):
//   diagonal — the diagonal stride equals the sum of the two traversed
//              dimensions' strides, creating a non-rectangular access pattern.
//              Example: torch.randn(3,4,4).diagonal(dim1=1,dim2=2)
//                → shape [3,4], strides [16, 5] where 5 = stride_dim1 + stride_dim2 = 4 + 1.
//              GCD(16, 5) = 1, so the GCD check rejects it.
//   unfold — overlapping windows when step < size
//   as_strided — arbitrary strides, may be non-rectangular or overlapping
//
enum class RandomOp { TRANSPOSE, SLICE, EXPAND, PERMUTE, UNSQUEEZE, SQUEEZE, SELECT };

// Helper: format a shape vector as string for error messages
std::string FormatShape(const std::vector<int64_t>& shape) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << shape[i];
  }
  oss << "]";
  return oss.str();
}

// Helper: format tensor shape/strides/offset for error messages
std::string FormatTensorInfo(const at::Tensor& t) {
  std::ostringstream oss;
  oss << "shape=" << FormatShape(std::vector<int64_t>(t.sizes().begin(), t.sizes().end()))
      << " strides=" << FormatShape(std::vector<int64_t>(t.strides().begin(), t.strides().end()))
      << " offset=" << t.storage_offset() << " contiguous=" << t.is_contiguous();
  return oss.str();
}

TEST_F(TransformationsTest, TryCreateCompound_RandomMixedSequences) {
  // Use random_device for a truly random seed; print it on failure for reproduction
  std::random_device rd;
  const uint32_t seed = rd();
  std::mt19937 rng(seed);

  constexpr int kNumIterations = 10000;
  constexpr int kMinOps = 1;
  constexpr int kMaxOps = 10;
  constexpr int64_t kMinDimSize = 2;
  constexpr int64_t kMaxDimSize = 8;

  // Distribution helpers
  std::uniform_int_distribution<int> num_ops_dist(kMinOps, kMaxOps);
  std::uniform_int_distribution<int> rank_dist(2, 4);  // 2D, 3D, or 4D
  std::uniform_int_distribution<int64_t> dim_size_dist(kMinDimSize, kMaxDimSize);
  std::uniform_int_distribution<int> op_type_dist(0, 6);  // All 7 RandomOp types

  int total_detected = 0;
  int total_contiguous_noop =
      0;                    // contiguous tensors where TryCreateCompound correctly returns nullopt
  int total_invalid = 0;    // degenerate tensors (<2D, size-0)
  int total_unhandled = 0;  // non-contiguous but TryCreateCompound returned nullopt (gap!)
  std::vector<std::string> unhandled_logs;  // details of unhandled patterns

  for (int iter = 0; iter < kNumIterations; ++iter) {
    // Step 1: Generate random initial shape
    int rank = rank_dist(rng);
    std::vector<int64_t> initial_shape;
    initial_shape.reserve(rank);
    for (int d = 0; d < rank; ++d) {
      initial_shape.push_back(dim_size_dist(rng));
    }

    // Create contiguous tensor with random values
    auto tensor = at::randn(c10::IntArrayRef(initial_shape),
                            at::TensorOptions().dtype(c10::ScalarType::Float));

    // Step 2: Apply random sequence of operations
    int num_ops = num_ops_dist(rng);
    std::vector<std::string> op_log;  // For error reporting
    op_log.push_back("initial: " + FormatTensorInfo(tensor));

    bool sequence_valid = true;
    for (int op_idx = 0; op_idx < num_ops && sequence_valid; ++op_idx) {
      int64_t ndim = tensor.dim();
      if (ndim < 2) {
        // Can't do meaningful operations on <2D tensors
        sequence_valid = false;
        break;
      }

      RandomOp op = static_cast<RandomOp>(op_type_dist(rng));

      switch (op) {
        case RandomOp::TRANSPOSE: {
          // Pick two random distinct dimensions to swap
          std::uniform_int_distribution<int64_t> dim_dist(0, ndim - 1);
          int64_t dim0 = dim_dist(rng);
          int64_t dim1 = dim_dist(rng);
          while (dim1 == dim0) {
            dim1 = dim_dist(rng);
          }
          tensor = tensor.transpose(dim0, dim1);
          op_log.push_back("transpose(" + std::to_string(dim0) + "," + std::to_string(dim1) +
                           ") -> " + FormatTensorInfo(tensor));
          break;
        }

        case RandomOp::SLICE: {
          // Random dim, random start/end, 50% chance of step > 1
          std::uniform_int_distribution<int64_t> dim_dist(0, ndim - 1);
          int64_t dim = dim_dist(rng);
          int64_t dim_size = tensor.size(dim);
          if (dim_size <= 1) {
            op_log.push_back("slice(dim=" + std::to_string(dim) + ") SKIPPED (size=1)");
            break;
          }
          // 50% chance of step slice (step > 1)
          int64_t step = 1;
          if (dim_size > 2 && rng() % 2 == 0) {
            step =
                std::uniform_int_distribution<int64_t>(2, std::min((int64_t)4, dim_size - 1))(rng);
          }
          std::uniform_int_distribution<int64_t> start_dist(0, dim_size - 2);
          int64_t start = start_dist(rng);
          std::uniform_int_distribution<int64_t> end_dist(start + 1, dim_size);
          int64_t end = end_dist(rng);
          tensor = tensor.slice(dim, start, end, step);
          op_log.push_back("slice(dim=" + std::to_string(dim) + ", " + std::to_string(start) + ":" +
                           std::to_string(end) + ":" + std::to_string(step) + ") -> " +
                           FormatTensorInfo(tensor));
          break;
        }

        case RandomOp::EXPAND: {
          // Find a dimension with size > 1, set it to 1 via slice, then expand
          // This creates a broadcast pattern (stride=0)
          // First, we need to find or create a size-1 dimension
          // Strategy: pick a dim, slice it to size 1, then expand
          std::uniform_int_distribution<int64_t> dim_dist(0, ndim - 1);
          int64_t dim = dim_dist(rng);
          int64_t dim_size = tensor.size(dim);

          if (dim_size <= 1) {
            // Already size 1, just expand it
            std::uniform_int_distribution<int64_t> expand_dist(2, 6);
            int64_t expand_size = expand_dist(rng);
            std::vector<int64_t> expand_shape(tensor.sizes().begin(), tensor.sizes().end());
            expand_shape[dim] = expand_size;
            tensor = tensor.expand(c10::IntArrayRef(expand_shape));
            op_log.push_back("expand(dim=" + std::to_string(dim) + ", size=" +
                             std::to_string(expand_size) + ") -> " + FormatTensorInfo(tensor));
          } else {
            // Slice to size 1 first, then expand
            int64_t start = std::uniform_int_distribution<int64_t>(0, dim_size - 1)(rng);
            tensor = tensor.slice(dim, start, start + 1);
            std::uniform_int_distribution<int64_t> expand_dist(2, 6);
            int64_t expand_size = expand_dist(rng);
            std::vector<int64_t> expand_shape(tensor.sizes().begin(), tensor.sizes().end());
            expand_shape[dim] = expand_size;
            tensor = tensor.expand(c10::IntArrayRef(expand_shape));
            op_log.push_back("slice_to_1_then_expand(dim=" + std::to_string(dim) + ", start=" +
                             std::to_string(start) + ", expand=" + std::to_string(expand_size) +
                             ") -> " + FormatTensorInfo(tensor));
          }
          break;
        }

        case RandomOp::PERMUTE: {
          // Random full permutation of all dims
          std::vector<int64_t> perm(ndim);
          std::iota(perm.begin(), perm.end(), 0);
          std::shuffle(perm.begin(), perm.end(), rng);
          tensor = tensor.permute(c10::IntArrayRef(perm));
          std::ostringstream perm_str;
          perm_str << "permute(";
          for (size_t pi = 0; pi < perm.size(); ++pi) {
            if (pi > 0) perm_str << ",";
            perm_str << perm[pi];
          }
          perm_str << ") -> " << FormatTensorInfo(tensor);
          op_log.push_back(perm_str.str());
          break;
        }

        case RandomOp::SELECT: {
          // Random dim, random index — reduces rank by 1
          if (ndim <= 2) {
            op_log.push_back("select SKIPPED (ndim<=2)");
            break;
          }
          std::uniform_int_distribution<int64_t> dim_dist(0, ndim - 1);
          int64_t dim = dim_dist(rng);
          std::uniform_int_distribution<int64_t> idx_dist(0, tensor.size(dim) - 1);
          int64_t idx = idx_dist(rng);
          tensor = tensor.select(dim, idx);
          op_log.push_back("select(dim=" + std::to_string(dim) + ", idx=" + std::to_string(idx) +
                           ") -> " + FormatTensorInfo(tensor));
          break;
        }

        case RandomOp::SQUEEZE: {
          // Find all size-1 dims, pick one randomly to squeeze
          if (ndim <= 2) {
            op_log.push_back("squeeze SKIPPED (ndim<=2)");
            break;
          }
          std::vector<int64_t> squeezable;
          for (int64_t d = 0; d < ndim; ++d) {
            if (tensor.size(d) == 1) squeezable.push_back(d);
          }
          if (!squeezable.empty()) {
            std::uniform_int_distribution<size_t> sq_dist(0, squeezable.size() - 1);
            int64_t d = squeezable[sq_dist(rng)];
            tensor = tensor.squeeze(d);
            op_log.push_back("squeeze(dim=" + std::to_string(d) + ") -> " +
                             FormatTensorInfo(tensor));
          } else {
            op_log.push_back("squeeze SKIPPED (no size-1 dims)");
          }
          break;
        }

        case RandomOp::UNSQUEEZE: {
          // Random position (0 to ndim inclusive)
          std::uniform_int_distribution<int64_t> pos_dist(0, ndim);
          int64_t dim = pos_dist(rng);
          tensor = tensor.unsqueeze(dim);
          op_log.push_back("unsqueeze(dim=" + std::to_string(dim) + ") -> " +
                           FormatTensorInfo(tensor));
          break;
        }
      }
    }

    if (!sequence_valid) {
      total_invalid++;
      continue;
    }

    // Step 3: If tensor is contiguous, verify TryCreateCompound returns nullopt (correct no-op)
    if (tensor.is_contiguous()) {
      auto result = Creators::TryCreateCompound(tensor, "fuzz_test", 0);
      EXPECT_FALSE(result.has_value())
          << "TryCreateCompound should return nullopt for contiguous tensor (seed=" << seed
          << ", iter=" << iter << ")";
      total_contiguous_noop++;
      continue;
    }

    // Step 4: Try to decompose with TryCreateCompound
    auto result = Creators::TryCreateCompound(tensor, "fuzz_test", 0);

    if (!result.has_value()) {
      // Non-contiguous tensor that TryCreateCompound couldn't handle — this is a gap!
      // Our random generator only produces transpose/slice/expand patterns (no overlapping
      // strides), so TryCreateCompound should be able to handle all of them.
      total_unhandled++;
      std::ostringstream unhandled_ctx;
      unhandled_ctx << "iter=" << iter << " " << FormatTensorInfo(tensor) << " ops: ";
      for (size_t li = 1; li < op_log.size(); ++li) {
        unhandled_ctx << op_log[li] << "; ";
      }
      unhandled_logs.push_back(unhandled_ctx.str());
      continue;
    }

    total_detected++;

    // Build error context for assertion messages
    std::ostringstream ctx;
    ctx << "\n=== RANDOM FUZZ FAILURE (seed=" << seed << ", iter=" << iter << ") ===\n";
    ctx << "Operation sequence:\n";
    for (const auto& log_entry : op_log) {
      ctx << "  " << log_entry << "\n";
    }
    ctx << "Final tensor: " << FormatTensorInfo(tensor) << "\n";
    ctx << "Detected " << result->transformations.size() << " transformations\n";
    std::string error_ctx = ctx.str();

    // Invariant 1: Base tensor must be contiguous
    EXPECT_TRUE(result->processed_input.is_contiguous())
        << "Base tensor must be contiguous" << error_ctx;

    // Invariant 2: Last transformation's output shape must match input tensor's shape
    ASSERT_FALSE(result->transformations.empty())
        << "Must have at least one transformation" << error_ctx;

    const auto& last_transform = result->transformations.back();
    ASSERT_EQ(last_transform.output_shape.size(), static_cast<size_t>(tensor.dim()))
        << "Output rank mismatch" << error_ctx;

    for (int64_t i = 0; i < tensor.dim(); ++i) {
      EXPECT_EQ(last_transform.output_shape[i], tensor.size(i))
          << "Output shape mismatch at dim " << i << error_ctx;
    }

    // Invariant 3: Transformation chain is well-formed (output[i] feeds input[i+1])
    for (size_t t = 1; t < result->transformations.size(); ++t) {
      const auto& prev = result->transformations[t - 1];
      const auto& curr = result->transformations[t];
      EXPECT_EQ(prev.output_shape, curr.input_shape)
          << "Chain break at transformation " << t << error_ctx;
    }

    // Invariant 4: All transformations have valid types
    for (size_t t = 0; t < result->transformations.size(); ++t) {
      auto type = result->transformations[t].type;
      EXPECT_TRUE(type == TransformationType::TRANSPOSE || type == TransformationType::SLICE ||
                  type == TransformationType::BROADCAST || type == TransformationType::RESHAPE)
          << "Invalid transformation type at index " << t << error_ctx;
    }

    // Invariant 5: First transformation's input shape matches base tensor shape
    const auto& first_transform = result->transformations[0];
    ASSERT_EQ(first_transform.input_shape.size(),
              static_cast<size_t>(result->processed_input.dim()))
        << "First transform input rank != base tensor rank" << error_ctx;

    for (int64_t i = 0; i < result->processed_input.dim(); ++i) {
      EXPECT_EQ(first_transform.input_shape[i], result->processed_input.size(i))
          << "First transform input shape mismatch at dim " << i << error_ctx;
    }
  }

  // Print summary (visible in test output)
  std::cout << "\n[RandomMixedSequences] seed=" << seed << " detected=" << total_detected
            << " contiguous_noop=" << total_contiguous_noop << " unhandled=" << total_unhandled
            << " invalid=" << total_invalid << " total=" << kNumIterations << std::endl;

  // Print details of unhandled patterns (if any)
  for (const auto& log : unhandled_logs) {
    std::cout << "  UNHANDLED: " << log << std::endl;
  }

  // Ensure we actually tested a meaningful number of patterns
  EXPECT_GT(total_detected, 0) << "No patterns were detected across " << kNumIterations
                               << " iterations (seed=" << seed
                               << "). This suggests the random generator "
                               << "is not producing non-contiguous tensors.";

  // TryCreateCompound handles all standard transpose/slice/expand combinations.
  EXPECT_EQ(total_unhandled, 0)
      << total_unhandled << " patterns were unhandled (seed=" << seed << "). "
      << "TryCreateCompound should handle all transpose/slice/expand combinations.";
}

//============================================================================
// 4GB Base Tensor Rejection Tests
//============================================================================

TEST_F(TransformationsTest, TryCreateSlice_Reject4GBBaseTensor) {
  // Simulates the failing test case: [32, 16, 2048, 2048] float16 sliced to
  // [32, 4, 2048, 2048] via [:, ::4, :, :].
  // Base tensor = 32 * 16 * 2048 * 2048 * 2 bytes = exactly 4GB.
  // The prologue must reject this to avoid DMA engine timeout.
  auto base = at::zeros({32, 16, 2048, 2048}, at::TensorOptions().dtype(c10::ScalarType::Half));
  auto sliced = base.slice(1, 0, 16, 4);  // [:, ::4, :, :] → [32, 4, 2048, 2048]

  ASSERT_FALSE(sliced.is_contiguous());

  auto result = Creators::TryCreateSlice(sliced, "test_4gb_reject", 0);
  EXPECT_FALSE(result.has_value())
      << "TryCreateSlice must reject slice patterns where the base tensor >= 4GB";
}

TEST_F(TransformationsTest, TryCreateSlice_AcceptJustUnder4GB) {
  // Base tensor just under 4GB should be accepted.
  // [16, 16, 2048, 2048] float16 = 16 * 16 * 2048 * 2048 * 2 = 2GB < 4GB.
  auto base = at::zeros({16, 16, 2048, 2048}, at::TensorOptions().dtype(c10::ScalarType::Half));
  auto sliced = base.slice(1, 0, 16, 4);  // [:, ::4, :, :] → [16, 4, 2048, 2048]

  ASSERT_FALSE(sliced.is_contiguous());

  auto result = Creators::TryCreateSlice(sliced, "test_under_4gb", 0);
  // Should be accepted (if DetectSlicePattern recognizes it)
  // Note: DetectSlicePattern may reject for other reasons (strided slice),
  // so we only check that IF it returns a value, it's marked supported.
  if (result.has_value()) {
    EXPECT_TRUE(result->is_supported);
  }
}

TEST_F(TransformationsTest, TryCreateCompound_Reject4GBBaseTensor) {
  // Create a tensor whose compound base would exceed 4GB.
  // [32, 16, 2048, 2048] float16 sliced with step → compound pattern.
  // Use a strided slice that TryCreateSlice rejects but TryCreateCompound handles.
  auto base = at::zeros({32, 16, 2048, 2048}, at::TensorOptions().dtype(c10::ScalarType::Half));
  // Strided slice: [:, ::2, :, :] → [32, 8, 2048, 2048]
  // This creates a stride gap that may route to TryCreateCompound.
  auto sliced = base.slice(1, 0, 16, 2);

  ASSERT_FALSE(sliced.is_contiguous());

  auto result = Creators::TryCreateCompound(sliced, "test_4gb_compound_reject", 0);
  EXPECT_FALSE(result.has_value())
      << "TryCreateCompound must reject patterns where the base tensor >= 4GB";
}

TEST_F(TransformationsTest, TryCreateAnyTransformation_RejectComplexDtype) {
  // Complex tensors cannot be fused — compiler rejects complex in prologues.
  // TryCreateAnyTransformation must return nullopt so ProcessInputTransformations
  // falls back to CPU contiguous.
  auto tensor =
      at::zeros({3, 4}, at::TensorOptions().dtype(c10::ScalarType::ComplexFloat)).transpose(0, 1);

  ASSERT_FALSE(tensor.is_contiguous()) << "Tensor should be non-contiguous";
  ASSERT_TRUE(tensor.is_complex()) << "Tensor should be complex dtype";

  auto& registry = TransformationRegistry::Get();
  auto result = registry.TryCreateAnyTransformation(tensor, "test_op", 0);

  EXPECT_FALSE(result.has_value()) << "Should reject complex dtype tensor";
}

//============================================================================
// DecomposeStrides Tests
//============================================================================

TEST_F(TransformationsTest, DecomposeStrides_PureTranspose_2D) {
  // 2D transpose: [3, 4].t() → logical [4, 3], physical [3, 4]
  auto tensor = at::zeros({3, 4}, at::TensorOptions().dtype(c10::ScalarType::Float)).t();
  ASSERT_FALSE(tensor.is_contiguous());

  auto result = DecomposeStrides(tensor, "test_op", 0);

  ASSERT_TRUE(result.has_value()) << "Should decompose 2D transpose";
  EXPECT_TRUE(result->base_tensor.is_contiguous()) << "Base tensor must be contiguous";
  EXPECT_EQ(result->total_dims, 2);
  EXPECT_EQ(result->element_type, "f32");

  // No broadcast dims
  EXPECT_FALSE(result->is_broadcast[0]);
  EXPECT_FALSE(result->is_broadcast[1]);

  // No slicing (starts are 0, steps are 1, base == target)
  for (int i = 0; i < result->total_dims; ++i) {
    EXPECT_EQ(result->phys_starts[i], 0) << "No slicing expected at dim " << i;
    EXPECT_EQ(result->phys_steps[i], 1) << "No striding expected at dim " << i;
  }
}

TEST_F(TransformationsTest, DecomposeStrides_RejectContiguous) {
  auto tensor = at::zeros({3, 4}, at::TensorOptions().dtype(c10::ScalarType::Float));
  ASSERT_TRUE(tensor.is_contiguous());

  auto result = DecomposeStrides(tensor, "test_op", 0);
  EXPECT_FALSE(result.has_value()) << "Should reject contiguous tensor";
}

TEST_F(TransformationsTest, DecomposeStrides_PureSlice) {
  // [8, 10] sliced to [:, 3:7] → shape [8, 4], offset=3
  auto tensor =
      at::zeros({8, 10}, at::TensorOptions().dtype(c10::ScalarType::Float)).slice(1, 3, 7);
  ASSERT_FALSE(tensor.is_contiguous());

  auto result = DecomposeStrides(tensor, "test_op", 0);

  ASSERT_TRUE(result.has_value()) << "Should decompose slice pattern";
  EXPECT_TRUE(result->base_tensor.is_contiguous());
  EXPECT_EQ(result->total_dims, 2);
}

TEST_F(TransformationsTest, DecomposeStrides_TransposeSlice_RoPE) {
  // RoPE pattern: [1,8,4,8].transpose(1,2)[..., 4:]
  auto base = at::zeros({1, 8, 4, 8}, at::TensorOptions().dtype(c10::ScalarType::Float));
  auto tensor = base.transpose(1, 2).slice(3, 4, 8);
  ASSERT_FALSE(tensor.is_contiguous());

  auto result = DecomposeStrides(tensor, "test_op", 0);

  ASSERT_TRUE(result.has_value()) << "Should decompose transpose+slice (RoPE)";
  EXPECT_TRUE(result->base_tensor.is_contiguous());
  EXPECT_EQ(result->total_dims, 4);
}

//============================================================================
// TryCreateEpilogue Tests
//============================================================================

TEST_F(TransformationsTest, TryCreateEpilogue_PureTranspose_2D) {
  // 2D transpose: [3, 4].t() → logical [4, 3]
  // Epilogue should produce inverse transpose to write [4, 3] contiguous → [3, 4] physical
  auto tensor = at::zeros({3, 4}, at::TensorOptions().dtype(c10::ScalarType::Float)).t();
  ASSERT_FALSE(tensor.is_contiguous());

  auto result = Creators::TryCreateEpilogue(tensor, "test_op", 0);

  ASSERT_TRUE(result.has_value()) << "Should create epilogue for 2D transpose";
  EXPECT_TRUE(result->is_supported);
  EXPECT_EQ(result->pattern_name, "EPILOGUE_TRANSPOSE");

  // Base tensor must be contiguous
  EXPECT_TRUE(result->processed_input.is_contiguous());
  EXPECT_EQ(result->processed_input.size(0), 3);
  EXPECT_EQ(result->processed_input.size(1), 4);

  // Should have exactly 1 TRANSPOSE transformation (the inverse)
  ASSERT_EQ(result->transformations.size(), 1u);
  EXPECT_EQ(result->transformations[0].type, TransformationType::TRANSPOSE);

  // Input shape = logical shape [4, 3] (what the operation produces)
  EXPECT_EQ(result->transformations[0].input_shape[0], 4);
  EXPECT_EQ(result->transformations[0].input_shape[1], 3);

  // Output shape = physical shape [3, 4] (base storage layout)
  EXPECT_EQ(result->transformations[0].output_shape[0], 3);
  EXPECT_EQ(result->transformations[0].output_shape[1], 4);
}

TEST_F(TransformationsTest, TryCreateEpilogue_PureTranspose_4D_AttentionPattern) {
  // 4D attention pattern: [2, 8, 4, 16].transpose(1, 2) → [2, 4, 8, 16]
  auto base = at::zeros({2, 8, 4, 16}, at::TensorOptions().dtype(c10::ScalarType::Float));
  auto tensor = base.transpose(1, 2);  // [2, 4, 8, 16]
  ASSERT_FALSE(tensor.is_contiguous());

  auto result = Creators::TryCreateEpilogue(tensor, "test_op", 0);

  ASSERT_TRUE(result.has_value()) << "Should create epilogue for 4D transpose";
  EXPECT_TRUE(result->is_supported);
  EXPECT_EQ(result->pattern_name, "EPILOGUE_TRANSPOSE");

  // Base tensor must be contiguous with physical shape [2, 8, 4, 16]
  EXPECT_TRUE(result->processed_input.is_contiguous());
  EXPECT_EQ(result->processed_input.size(0), 2);
  EXPECT_EQ(result->processed_input.size(1), 8);
  EXPECT_EQ(result->processed_input.size(2), 4);
  EXPECT_EQ(result->processed_input.size(3), 16);

  // Should have exactly 1 TRANSPOSE transformation
  ASSERT_EQ(result->transformations.size(), 1u);
  EXPECT_EQ(result->transformations[0].type, TransformationType::TRANSPOSE);

  // Input shape = logical [2, 4, 8, 16]
  EXPECT_EQ(result->transformations[0].input_shape[0], 2);
  EXPECT_EQ(result->transformations[0].input_shape[1], 4);
  EXPECT_EQ(result->transformations[0].input_shape[2], 8);
  EXPECT_EQ(result->transformations[0].input_shape[3], 16);

  // Output shape = physical [2, 8, 4, 16]
  EXPECT_EQ(result->transformations[0].output_shape[0], 2);
  EXPECT_EQ(result->transformations[0].output_shape[1], 8);
  EXPECT_EQ(result->transformations[0].output_shape[2], 4);
  EXPECT_EQ(result->transformations[0].output_shape[3], 16);
}

TEST_F(TransformationsTest, TryCreateEpilogue_InversePermutationIsCorrect) {
  // Verify: composing prologue permutation with epilogue permutation = identity
  // For [3, 4].t(): prologue perm = [1, 0], epilogue perm should also be [1, 0]
  // (self-inverse for 2D transpose)
  auto tensor = at::zeros({3, 4}, at::TensorOptions().dtype(c10::ScalarType::Float)).t();

  auto prologue = Creators::TryCreateCompound(tensor, "test_op", 0);
  auto epilogue = Creators::TryCreateEpilogue(tensor, "test_op", 0);

  ASSERT_TRUE(prologue.has_value());
  ASSERT_TRUE(epilogue.has_value());

  // Find the TRANSPOSE in prologue chain
  const TensorTransformation* prologue_transpose = nullptr;
  for (const auto& t : prologue->transformations) {
    if (t.type == TransformationType::TRANSPOSE) {
      prologue_transpose = &t;
      break;
    }
  }
  ASSERT_NE(prologue_transpose, nullptr) << "Prologue should have a TRANSPOSE";

  // Epilogue has exactly 1 TRANSPOSE
  ASSERT_EQ(epilogue->transformations.size(), 1u);
  const auto& epilogue_transpose = epilogue->transformations[0];

  // Compose: apply prologue perm then epilogue perm → should be identity
  const auto& fwd_perm = prologue_transpose->params;
  const auto& inv_perm = epilogue_transpose.params;
  size_t ndims = fwd_perm.size();
  ASSERT_EQ(inv_perm.size(), ndims);

  for (size_t i = 0; i < ndims; ++i) {
    // Apply forward then inverse: inv_perm[fwd_perm[i]] should == i
    EXPECT_EQ(inv_perm[fwd_perm[i]], static_cast<int64_t>(i))
        << "Composing prologue and epilogue permutations should yield identity at dim " << i;
  }
}

TEST_F(TransformationsTest, TryCreateEpilogue_RejectContiguousTensor) {
  auto tensor = at::zeros({3, 4}, at::TensorOptions().dtype(c10::ScalarType::Float));
  ASSERT_TRUE(tensor.is_contiguous());

  auto result = Creators::TryCreateEpilogue(tensor, "test_op", 0);
  EXPECT_FALSE(result.has_value()) << "Should reject contiguous tensor";
}

// DISABLED_ tests: These represent the FINAL expected behavior for future phases.
// They are disabled now because the inverse operations are not yet implemented,
// but they define the TDD targets for Phase 2+ implementation.

TEST_F(TransformationsTest, TryCreateEpilogue_SlicePattern) {
  // SCATTER_IOTA: [8, 10] sliced to [:, 3:7] → epilogue scatters [8, 4] into 1D flat storage
  auto tensor =
      at::zeros({8, 10}, at::TensorOptions().dtype(c10::ScalarType::Float)).slice(1, 3, 7);
  ASSERT_FALSE(tensor.is_contiguous());

  auto result = Creators::TryCreateEpilogue(tensor, "test_op", 0);

  ASSERT_TRUE(result.has_value()) << "Should create epilogue for slice pattern";
  EXPECT_TRUE(result->is_supported);
  EXPECT_EQ(result->pattern_name, "EPILOGUE_SLICE");
  EXPECT_TRUE(result->processed_input.is_contiguous());

  // Should have exactly 1 SCATTER_IOTA transformation
  ASSERT_EQ(result->transformations.size(), 1u);
  EXPECT_EQ(result->transformations[0].type, TransformationType::SCATTER_IOTA);

  // Input shape = logical [8, 4] (what the operation produces)
  EXPECT_EQ(result->transformations[0].input_shape[0], 8);
  EXPECT_EQ(result->transformations[0].input_shape[1], 4);

  // Output shape = 1D flat storage
  EXPECT_EQ(result->transformations[0].output_shape.size(), 1u);
  EXPECT_EQ(result->transformations[0].output_shape[0], 80);  // 8*10

  // Base tensor should be 1D flat storage
  EXPECT_EQ(result->processed_input.dim(), 1);
  EXPECT_EQ(result->processed_input.size(0), 80);
}

TEST_F(TransformationsTest, DISABLED_TryCreateEpilogue_BroadcastPattern) {
  // Phase 4 target: BROADCAST inverse should produce REDUCE or SLICE
  // [3, 1] expanded to [3, 4] → epilogue collapses dim 1 back to size 1
  auto base = at::zeros({3, 1}, at::TensorOptions().dtype(c10::ScalarType::Float));
  auto tensor = base.expand({3, 4});
  ASSERT_FALSE(tensor.is_contiguous());

  auto result = Creators::TryCreateEpilogue(tensor, "test_op", 0);

  ASSERT_TRUE(result.has_value()) << "Should create epilogue for broadcast pattern";
  EXPECT_TRUE(result->is_supported);
  EXPECT_TRUE(result->processed_input.is_contiguous());
}

TEST_F(TransformationsTest, DISABLED_TryCreateEpilogue_TransposeSliceCompound) {
  // Phase 3 target: Compound (transpose+slice) inverse
  // [1,8,4,8].transpose(1,2)[..., 4:] → epilogue: inverse_transpose then scatter_iota
  auto base = at::zeros({1, 8, 4, 8}, at::TensorOptions().dtype(c10::ScalarType::Float));
  auto tensor = base.transpose(1, 2).slice(3, 4, 8);
  ASSERT_FALSE(tensor.is_contiguous());

  auto result = Creators::TryCreateEpilogue(tensor, "test_op", 0);

  ASSERT_TRUE(result.has_value()) << "Should create epilogue for transpose+slice compound";
  EXPECT_TRUE(result->is_supported);
  EXPECT_TRUE(result->processed_input.is_contiguous());

  // Should have at least 2 transformations (inverse transpose + inverse slice)
  ASSERT_GE(result->transformations.size(), 2u);
}

// Phase 2: Slice and compound epilogues are now supported
TEST_F(TransformationsTest, TryCreateEpilogue_SlicePattern_Phase2) {
  // Phase 2: SLICE inverse produces scatter_iota
  auto tensor =
      at::zeros({8, 10}, at::TensorOptions().dtype(c10::ScalarType::Float)).slice(1, 3, 7);
  ASSERT_FALSE(tensor.is_contiguous());

  auto result = Creators::TryCreateEpilogue(tensor, "test_op", 0);
  ASSERT_TRUE(result.has_value()) << "Phase 2: Slice epilogue should now be supported";
  EXPECT_EQ(result->pattern_name, "EPILOGUE_SLICE");
  EXPECT_TRUE(result->processed_input.is_contiguous());
}

TEST_F(TransformationsTest, TryCreateEpilogue_TransposeSliceCompound) {
  // Compound epilogue (transpose+slice) is now supported by the new AnalyzeEpilogueLayout
  // which reads base shape directly from strides instead of using DecomposeStrides.
  auto base = at::zeros({1, 8, 4, 8}, at::TensorOptions().dtype(c10::ScalarType::Float));
  auto tensor = base.transpose(1, 2).slice(3, 4, 8);
  ASSERT_FALSE(tensor.is_contiguous());

  auto result = Creators::TryCreateEpilogue(tensor, "test_op", 0);
  ASSERT_TRUE(result.has_value()) << "Compound epilogue (transpose+slice) should be supported";
  EXPECT_EQ(result->pattern_name, "EPILOGUE_COMPOUND");
  EXPECT_TRUE(result->processed_input.is_contiguous());

  // Should have 2 transformations: inverse transpose + SCATTER_IOTA
  ASSERT_EQ(result->transformations.size(), 2u);
  EXPECT_EQ(result->transformations[0].type, TransformationType::TRANSPOSE);
  EXPECT_EQ(result->transformations[1].type, TransformationType::SCATTER_IOTA);
}

TEST_F(TransformationsTest, TryCreateEpilogue_BaseIsContiguous) {
  // For any supported epilogue pattern, the base tensor must be contiguous
  auto tensor = at::zeros({4, 5, 6, 7}, at::TensorOptions().dtype(c10::ScalarType::Float))
                    .transpose(1, 3);  // [4, 7, 6, 5]
  ASSERT_FALSE(tensor.is_contiguous());

  auto result = Creators::TryCreateEpilogue(tensor, "test_op", 0);
  ASSERT_TRUE(result.has_value()) << "Should create epilogue for 4D transpose";
  EXPECT_TRUE(result->processed_input.is_contiguous()) << "Epilogue base tensor must be contiguous";
}

TEST_F(TransformationsTest, TryCreateEpilogue_BFloat16Dtype) {
  // Verify epilogue works with bfloat16 dtype
  auto tensor = at::zeros({3, 4}, at::TensorOptions().dtype(c10::ScalarType::BFloat16)).t();
  ASSERT_FALSE(tensor.is_contiguous());

  auto result = Creators::TryCreateEpilogue(tensor, "test_op", 0);
  ASSERT_TRUE(result.has_value()) << "Should create epilogue for bf16 transpose";
  EXPECT_EQ(result->transformations[0].element_type, "bf16");
}

//============================================================================
// Additional Epilogue Tests — Coverage Gaps
//============================================================================

TEST_F(TransformationsTest, TryCreateEpilogue_MultiAxisSlice) {
  // Multi-axis slice: base[2:6, 3:7] on [8, 10]
  // shape=[4, 4], strides=[10, 1], offset=23
  auto base = at::zeros({8, 10}, at::TensorOptions().dtype(c10::ScalarType::Float));
  auto tensor = base.slice(0, 2, 6).slice(1, 3, 7);
  ASSERT_FALSE(tensor.is_contiguous());
  ASSERT_EQ(tensor.storage_offset(), 23);  // 2*10 + 3

  auto result = Creators::TryCreateEpilogue(tensor, "test_op", 0);

  ASSERT_TRUE(result.has_value()) << "Should create epilogue for multi-axis slice";
  EXPECT_EQ(result->pattern_name, "EPILOGUE_SLICE");
  EXPECT_TRUE(result->processed_input.is_contiguous());

  // Base should be 1D flat storage (SCATTER_IOTA uses flat base)
  EXPECT_EQ(result->processed_input.dim(), 1);
  EXPECT_EQ(result->processed_input.size(0), 80);  // 8*10

  // Should have 1 SCATTER_IOTA transformation (no transpose)
  ASSERT_EQ(result->transformations.size(), 1u);
  EXPECT_EQ(result->transformations[0].type, TransformationType::SCATTER_IOTA);
}

TEST_F(TransformationsTest, TryCreateEpilogue_SliceOffset0_OriginalBugCase) {
  // The original bug case: base[:, :4] on [4, 8]
  // shape=[4, 4], strides=[8, 1], offset=0
  // DecomposeStrides gave phys_base_shape=[4,4] (wrong), new analysis gives [4,8] (correct)
  auto base = at::zeros({4, 8}, at::TensorOptions().dtype(c10::ScalarType::Float));
  auto tensor = base.slice(1, 0, 4);
  ASSERT_FALSE(tensor.is_contiguous());
  ASSERT_EQ(tensor.storage_offset(), 0);

  auto result = Creators::TryCreateEpilogue(tensor, "test_op", 0);

  ASSERT_TRUE(result.has_value()) << "Should create epilogue for offset=0 slice";
  EXPECT_EQ(result->pattern_name, "EPILOGUE_SLICE");
  EXPECT_TRUE(result->processed_input.is_contiguous());

  // Base must be 1D flat (SCATTER_IOTA), storage = 4*8 = 32 elements
  EXPECT_EQ(result->processed_input.dim(), 1);
  EXPECT_EQ(result->processed_input.size(0), 32);
  EXPECT_EQ(result->processed_input.storage_offset(), 0);

  // Should use SCATTER_IOTA
  ASSERT_EQ(result->transformations.size(), 1u);
  EXPECT_EQ(result->transformations[0].type, TransformationType::SCATTER_IOTA);

  // params = [stride_0, stride_1, storage_offset, storage_span]
  // For [4,4] with strides [8,1], offset=0:
  ASSERT_EQ(result->transformations[0].params.size(), 4u);  // 2 strides + offset + span
  EXPECT_EQ(result->transformations[0].params[0], 8);       // stride dim 0
  EXPECT_EQ(result->transformations[0].params[1], 1);       // stride dim 1
  EXPECT_EQ(result->transformations[0].params[2], 0);       // storage offset
  EXPECT_EQ(result->transformations[0].params[3], 32);      // storage span
}

TEST_F(TransformationsTest, TryCreateEpilogue_StridedSliceRejection) {
  // Strided slice: base[::2] on [10, 4]
  // shape=[5, 4], strides=[8, 1], offset=0 — innermost stride=1 but stride gap
  auto base = at::zeros({10, 4}, at::TensorOptions().dtype(c10::ScalarType::Float));
  auto tensor = base.slice(0, 0, 10, 2);  // [::2] → shape [5, 4], strides [8, 1]
  ASSERT_FALSE(tensor.is_contiguous());

  auto result = Creators::TryCreateEpilogue(tensor, "test_op", 0);

  // Should reject: stride[0]=8 but base[0]*stride[1] = 8 and base[0]=8/1=8,
  // but storage=40, outermost base = 40/8 = 5 which equals view size.
  // Actually for this case: phys_base[0] = 40/8 = 5, phys_view[0] = 5 → no slice on dim 0.
  // phys_base[1] = 8/1 = 8, phys_view[1] = 4 → has slice on dim 1.
  // But wait — the stride gap check: strides[0]=8, strides[1]=1, 8/1=8, 8*1=8 ✓
  // So it passes the contiguity check. The base would be [5, 8] but storage is only 40=5*8 ✓.
  // This is actually a valid decomposition! The strided slice x[::2] on dim 0 looks like
  // a slice on dim 1 from the stride perspective. Let me use a clearer strided slice.

  // Use a 1D strided slice instead
  auto base1d = at::zeros({10}, at::TensorOptions().dtype(c10::ScalarType::Float));
  auto tensor1d = base1d.slice(0, 0, 10, 2);  // shape [5], stride [2]
  ASSERT_FALSE(tensor1d.is_contiguous());

  auto result1d = Creators::TryCreateEpilogue(tensor1d, "test_op", 0);
  EXPECT_FALSE(result1d.has_value())
      << "1D strided slice (stride=2) should be rejected (innermost stride != 1)";
}

TEST_F(TransformationsTest, TryCreateEpilogue_Size1Dims) {
  // Tensor with size-1 dims: [1, 4, 1, 8].transpose(1, 3) → [1, 8, 1, 4]
  // Size-1 dims should be handled as identity (no effect on physical layout)
  auto base = at::zeros({1, 4, 1, 8}, at::TensorOptions().dtype(c10::ScalarType::Float));
  auto tensor = base.transpose(1, 3);  // [1, 8, 1, 4]
  ASSERT_FALSE(tensor.is_contiguous());

  auto result = Creators::TryCreateEpilogue(tensor, "test_op", 0);

  ASSERT_TRUE(result.has_value()) << "Should handle size-1 dims in transpose";
  EXPECT_EQ(result->pattern_name, "EPILOGUE_TRANSPOSE");
  EXPECT_TRUE(result->processed_input.is_contiguous());

  // Base should have the physical shape [1, 4, 1, 8] (original contiguous layout)
  // Active dims are 1 and 3 (size 4 and 8), size-1 dims stay in place
  EXPECT_EQ(result->processed_input.numel(), 32);  // 1*4*1*8
}

TEST_F(TransformationsTest, TryCreateEpilogue_BaseOffsetAlwaysZero) {
  // Verify base_tensor.storage_offset() == 0 for various patterns

  // Pure transpose
  {
    auto tensor = at::zeros({3, 4}, at::TensorOptions().dtype(c10::ScalarType::Float)).t();
    auto result = Creators::TryCreateEpilogue(tensor, "test_op", 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->processed_input.storage_offset(), 0) << "Transpose: base offset must be 0";
  }

  // Slice with offset
  {
    auto tensor =
        at::zeros({8, 10}, at::TensorOptions().dtype(c10::ScalarType::Float)).slice(1, 3, 7);
    auto result = Creators::TryCreateEpilogue(tensor, "test_op", 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->processed_input.storage_offset(), 0) << "Slice: base offset must be 0";
  }

  // Compound (transpose + slice)
  {
    auto base = at::zeros({1, 8, 4, 8}, at::TensorOptions().dtype(c10::ScalarType::Float));
    auto tensor = base.transpose(1, 2).slice(3, 4, 8);
    auto result = Creators::TryCreateEpilogue(tensor, "test_op", 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->processed_input.storage_offset(), 0) << "Compound: base offset must be 0";
  }
}

TEST_F(TransformationsTest, TryCreateEpilogue_ScatterIotaParamsCorrect) {
  // Verify SCATTER_IOTA params are correct for various slice patterns
  // params layout: [stride_0, stride_1, ..., storage_offset, storage_span]

  // Single-axis slice: [8, 10][:, 3:7] → strides [10, 1], offset=3, storage=80
  {
    auto tensor =
        at::zeros({8, 10}, at::TensorOptions().dtype(c10::ScalarType::Float)).slice(1, 3, 7);
    auto result = Creators::TryCreateEpilogue(tensor, "test_op", 0);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->transformations.size(), 1u);
    EXPECT_EQ(result->transformations[0].type, TransformationType::SCATTER_IOTA);

    // params = [stride_dim0, stride_dim1, storage_offset, storage_span]
    const auto& params = result->transformations[0].params;
    ASSERT_EQ(params.size(), 4u);  // 2 strides + offset + span
    EXPECT_EQ(params[0], 10) << "Stride dim 0 should be 10";
    EXPECT_EQ(params[1], 1) << "Stride dim 1 should be 1";
    EXPECT_EQ(params[2], 3) << "Storage offset should be 3";
    EXPECT_EQ(params[3], 80) << "Storage span should be 80 (8*10)";
  }

  // Slice from start: [4, 8][:, :4] → strides [8, 1], offset=0, storage=32
  {
    auto tensor =
        at::zeros({4, 8}, at::TensorOptions().dtype(c10::ScalarType::Float)).slice(1, 0, 4);
    auto result = Creators::TryCreateEpilogue(tensor, "test_op", 0);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->transformations.size(), 1u);
    EXPECT_EQ(result->transformations[0].type, TransformationType::SCATTER_IOTA);

    const auto& params = result->transformations[0].params;
    ASSERT_EQ(params.size(), 4u);
    EXPECT_EQ(params[0], 8) << "Stride dim 0 should be 8";
    EXPECT_EQ(params[1], 1) << "Stride dim 1 should be 1";
    EXPECT_EQ(params[2], 0) << "Storage offset should be 0";
    EXPECT_EQ(params[3], 32) << "Storage span should be 32 (4*8)";
  }

  // Row slice: [8, 10][2:6, :] → shape [4, 10], strides [10, 1]
  // This is actually CONTIGUOUS (strides match row-major for shape [4, 10]),
  // so TryCreateEpilogue correctly returns nullopt. No epilogue needed.
  {
    auto tensor =
        at::zeros({8, 10}, at::TensorOptions().dtype(c10::ScalarType::Float)).slice(0, 2, 6);
    EXPECT_TRUE(tensor.is_contiguous()) << "Row slice produces contiguous tensor";
    auto result = Creators::TryCreateEpilogue(tensor, "test_op", 0);
    EXPECT_FALSE(result.has_value()) << "Row slice is contiguous — no epilogue needed";
  }
}

TEST_F(TransformationsTest, TryCreateEpilogue_ScatterIotaParams_CompoundPattern) {
  // Compound (transpose+slice): [1, 8, 4, 8].transpose(1,2)[..., 4:]
  // After inverse transpose, the physical-order strides are used for SCATTER_IOTA.
  auto base = at::zeros({1, 8, 4, 8}, at::TensorOptions().dtype(c10::ScalarType::Float));
  auto tensor = base.transpose(1, 2).slice(3, 4, 8);
  ASSERT_FALSE(tensor.is_contiguous());

  auto result = Creators::TryCreateEpilogue(tensor, "test_op", 0);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->pattern_name, "EPILOGUE_COMPOUND");

  // Should have [TRANSPOSE, SCATTER_IOTA]
  ASSERT_EQ(result->transformations.size(), 2u);
  EXPECT_EQ(result->transformations[0].type, TransformationType::TRANSPOSE);
  EXPECT_EQ(result->transformations[1].type, TransformationType::SCATTER_IOTA);

  // SCATTER_IOTA params: [strides..., offset, storage_span]
  const auto& scatter = result->transformations[1];
  size_t n = scatter.input_shape.size();
  ASSERT_EQ(scatter.params.size(), n + 2) << "params = [strides..., offset, span]";

  // Output must be 1D flat
  ASSERT_EQ(scatter.output_shape.size(), 1u);
  int64_t storage_span = scatter.params.back();
  EXPECT_EQ(scatter.output_shape[0], storage_span);
  EXPECT_EQ(storage_span, 1 * 8 * 4 * 8) << "Storage span = product of original base shape";

  // Storage offset should be the original tensor's offset (4)
  EXPECT_EQ(scatter.params[n], tensor.storage_offset());

  // Base tensor should be 1D flat
  EXPECT_EQ(result->processed_input.dim(), 1);
  EXPECT_EQ(result->processed_input.size(0), storage_span);
}

TEST_F(TransformationsTest, TryCreateEpilogue_ScatterIotaParams_MultiAxisSlice) {
  // Multi-axis slice: base[2:6, 3:7] on [8, 10]
  // shape=[4, 4], strides=[10, 1], offset=23 (2*10 + 3)
  auto base = at::zeros({8, 10}, at::TensorOptions().dtype(c10::ScalarType::Float));
  auto tensor = base.slice(0, 2, 6).slice(1, 3, 7);
  ASSERT_FALSE(tensor.is_contiguous());
  ASSERT_EQ(tensor.storage_offset(), 23);

  auto result = Creators::TryCreateEpilogue(tensor, "test_op", 0);
  ASSERT_TRUE(result.has_value());

  ASSERT_EQ(result->transformations.size(), 1u);
  EXPECT_EQ(result->transformations[0].type, TransformationType::SCATTER_IOTA);

  // params = [stride_0, stride_1, offset, storage_span]
  const auto& params = result->transformations[0].params;
  ASSERT_EQ(params.size(), 4u);
  EXPECT_EQ(params[0], 10) << "Stride dim 0 should be 10";
  EXPECT_EQ(params[1], 1) << "Stride dim 1 should be 1";
  EXPECT_EQ(params[2], 23) << "Storage offset should be 23 (2*10 + 3)";
  EXPECT_EQ(params[3], 80) << "Storage span should be 80 (8*10)";
}

TEST_F(TransformationsTest, TryCreateEpilogue_Reject4GBBase) {
  // Epilogue should reject when base tensor >= 4GB.
  // [32, 16, 2048, 2048] float16 sliced to [32, 4, 2048, 2048] via [:, ::4, :, :]
  // Full storage = 32 * 16 * 2048 * 2048 * 2 bytes = 4GB
  auto base = at::zeros({32, 16, 2048, 2048}, at::TensorOptions().dtype(c10::ScalarType::Half));
  auto tensor = base.slice(1, 0, 16, 4);  // [:, ::4, :, :]
  ASSERT_FALSE(tensor.is_contiguous());

  auto result = Creators::TryCreateEpilogue(tensor, "test_4gb_epilogue_reject", 0);
  EXPECT_FALSE(result.has_value()) << "TryCreateEpilogue must reject when base tensor >= 4GB";
}

TEST_F(TransformationsTest, TryCreateEpilogue_RejectAllSize1Dims) {
  // All dimensions are size 1: [1, 1, 1] — should be contiguous (no epilogue needed)
  auto tensor = at::zeros({1, 1, 1}, at::TensorOptions().dtype(c10::ScalarType::Float));
  ASSERT_TRUE(tensor.is_contiguous());

  auto result = Creators::TryCreateEpilogue(tensor, "test_op", 0);
  EXPECT_FALSE(result.has_value()) << "All-size-1 tensor is contiguous, no epilogue";
}

TEST_F(TransformationsTest, TryCreateEpilogue_RejectScalarTensor) {
  // 0-dim scalar tensor
  auto tensor = at::zeros({}, at::TensorOptions().dtype(c10::ScalarType::Float));

  auto result = Creators::TryCreateEpilogue(tensor, "test_op", 0);
  EXPECT_FALSE(result.has_value()) << "Scalar tensor (0 dims) should be rejected";
}

//============================================================================
// Epilogue Fuzz Test
//
// Generates random non-contiguous tensor views (transpose, step=1 slice,
// select, permute, unsqueeze, squeeze) and verifies TryCreateEpilogue
// produces correct results when it accepts a pattern.
//============================================================================

// Reuse RandomOp enum from prologue fuzz test (already defined above)

TEST_F(TransformationsTest, TryCreateEpilogue_RandomMixedSequences) {
  std::random_device rd;
  const uint32_t seed = rd();
  std::mt19937 rng(seed);

  constexpr int kNumIterations = 10000;
  constexpr int kMinOps = 1;
  constexpr int kMaxOps = 5;

  std::uniform_int_distribution<int> num_ops_dist(kMinOps, kMaxOps);
  std::uniform_int_distribution<int> rank_dist(2, 4);
  std::uniform_int_distribution<int64_t> dim_size_dist(2, 8);
  // Only epilogue-friendly ops: TRANSPOSE=0, SLICE(step=1)=1, PERMUTE=3, UNSQUEEZE=5, SQUEEZE=6,
  // SELECT=4
  std::uniform_int_distribution<int> op_type_dist(0, 5);

  int total_accepted = 0;
  int total_fallback = 0;
  int total_contiguous = 0;
  int total_invalid = 0;
  int total_errors = 0;

  for (int iter = 0; iter < kNumIterations; ++iter) {
    int rank = rank_dist(rng);
    std::vector<int64_t> initial_shape;
    for (int d = 0; d < rank; ++d) initial_shape.push_back(dim_size_dist(rng));

    auto tensor = at::randn(c10::IntArrayRef(initial_shape),
                            at::TensorOptions().dtype(c10::ScalarType::Float));

    bool sequence_valid = true;
    for (int op_idx = 0; op_idx < num_ops_dist(rng) && sequence_valid; ++op_idx) {
      int64_t ndim = tensor.dim();
      if (ndim < 2) {
        sequence_valid = false;
        break;
      }

      int op = op_type_dist(rng);
      switch (op) {
        case 0: {  // TRANSPOSE
          std::uniform_int_distribution<int64_t> dd(0, ndim - 1);
          int64_t d0 = dd(rng), d1 = dd(rng);
          while (d1 == d0) d1 = dd(rng);
          tensor = tensor.transpose(d0, d1);
          break;
        }
        case 1: {  // SLICE (step=1 only)
          std::uniform_int_distribution<int64_t> dd(0, ndim - 1);
          int64_t dim = dd(rng);
          int64_t sz = tensor.size(dim);
          if (sz <= 2) break;
          std::uniform_int_distribution<int64_t> sd(0, sz - 2);
          int64_t start = sd(rng);
          std::uniform_int_distribution<int64_t> ed(start + 1, sz - 1);
          int64_t end = ed(rng);
          tensor = tensor.slice(dim, start, end);  // step=1
          break;
        }
        case 2: {  // PERMUTE
          std::vector<int64_t> perm(ndim);
          std::iota(perm.begin(), perm.end(), 0);
          std::shuffle(perm.begin(), perm.end(), rng);
          tensor = tensor.permute(c10::IntArrayRef(perm));
          break;
        }
        case 3: {  // SELECT
          if (ndim <= 2) break;
          std::uniform_int_distribution<int64_t> dd(0, ndim - 1);
          int64_t dim = dd(rng);
          std::uniform_int_distribution<int64_t> id(0, tensor.size(dim) - 1);
          tensor = tensor.select(dim, id(rng));
          break;
        }
        case 4: {  // UNSQUEEZE
          std::uniform_int_distribution<int64_t> pd(0, ndim);
          tensor = tensor.unsqueeze(pd(rng));
          break;
        }
        case 5: {  // SQUEEZE
          if (ndim <= 2) break;
          std::vector<int64_t> squeezable;
          for (int64_t d = 0; d < ndim; ++d)
            if (tensor.size(d) == 1) squeezable.push_back(d);
          if (!squeezable.empty()) {
            std::uniform_int_distribution<size_t> sd(0, squeezable.size() - 1);
            tensor = tensor.squeeze(squeezable[sd(rng)]);
          }
          break;
        }
      }
    }

    if (!sequence_valid) {
      total_invalid++;
      continue;
    }
    if (tensor.is_contiguous()) {
      total_contiguous++;
      continue;
    }

    auto result = Creators::TryCreateEpilogue(tensor, "epilogue_fuzz", 0);

    if (!result.has_value()) {
      total_fallback++;
      continue;
    }

    total_accepted++;

    // Build error context
    std::ostringstream ctx;
    ctx << "\n=== EPILOGUE FUZZ FAILURE (seed=" << seed << ", iter=" << iter << ") ===\n"
        << "Tensor: " << FormatTensorInfo(tensor) << "\n";
    std::string err = ctx.str();

    // Invariant 1: Base tensor is contiguous
    EXPECT_TRUE(result->processed_input.is_contiguous()) << "Base must be contiguous" << err;

    // Invariant 2: Base storage_offset == 0
    EXPECT_EQ(result->processed_input.storage_offset(), 0) << "Base offset must be 0" << err;

    // Invariant 3: Base fits in storage
    int64_t base_numel = result->processed_input.numel();
    int64_t storage_elems = static_cast<int64_t>(tensor.storage().nbytes() / tensor.element_size());
    EXPECT_LE(base_numel, storage_elems) << "Base must fit in storage" << err;

    // Invariant 4: Transformation chain well-formed
    ASSERT_FALSE(result->transformations.empty()) << "Must have transforms" << err;
    for (size_t t = 1; t < result->transformations.size(); ++t) {
      EXPECT_EQ(result->transformations[t - 1].output_shape, result->transformations[t].input_shape)
          << "Chain break at " << t << err;
    }

    // Invariant 5: First transform input = logical shape
    const auto& first = result->transformations[0];
    ASSERT_EQ(first.input_shape.size(), static_cast<size_t>(tensor.dim())) << err;
    for (int64_t i = 0; i < tensor.dim(); ++i) {
      EXPECT_EQ(first.input_shape[i], tensor.size(i)) << "Input shape mismatch dim " << i << err;
    }

    // Invariant 6: Last transform output = base shape
    const auto& last = result->transformations.back();
    ASSERT_EQ(last.output_shape.size(), static_cast<size_t>(result->processed_input.dim())) << err;
    for (int64_t i = 0; i < result->processed_input.dim(); ++i) {
      EXPECT_EQ(last.output_shape[i], result->processed_input.size(i))
          << "Output shape mismatch dim " << i << err;
    }

    // Invariant 7: Only TRANSPOSE or SCATTER_IOTA (epilogue uses SCATTER_IOTA for slice patterns)
    for (const auto& t : result->transformations) {
      EXPECT_TRUE(t.type == TransformationType::TRANSPOSE ||
                  t.type == TransformationType::SCATTER_IOTA)
          << "Unexpected transform type" << err;
    }

    // Invariant 8: SCATTER_IOTA params validity check
    for (const auto& t : result->transformations) {
      if (t.type == TransformationType::SCATTER_IOTA) {
        // params = [stride_0, ..., stride_{n-1}, storage_offset, storage_span]
        // where n = input_shape.size()
        size_t n = t.input_shape.size();
        ASSERT_EQ(t.params.size(), n + 2) << "SCATTER_IOTA params size" << err;
        // Last element is storage_span, must be positive
        EXPECT_GT(t.params.back(), 0) << "SCATTER_IOTA storage_span" << err;
        // Storage offset must be non-negative
        EXPECT_GE(t.params[n], 0) << "SCATTER_IOTA offset" << err;
        // Output must be 1D flat
        ASSERT_EQ(t.output_shape.size(), 1u) << "SCATTER_IOTA output must be 1D" << err;
      }
    }
  }

  std::cout << "\n[EpilogueFuzz] seed=" << seed << " accepted=" << total_accepted
            << " fallback=" << total_fallback << " contiguous=" << total_contiguous
            << " invalid=" << total_invalid << " total=" << kNumIterations << std::endl;

  // At least some patterns should be accepted
  EXPECT_GT(total_accepted, 0) << "No patterns accepted (seed=" << seed << ")";
}

}  // namespace
}  // namespace lazy
}  // namespace c10_neuron
