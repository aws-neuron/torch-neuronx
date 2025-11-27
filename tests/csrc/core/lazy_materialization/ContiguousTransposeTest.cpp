#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <vector>

#include "torch_neuronx/csrc/core/lazy_materialization/ContiguousTranspose.h"

namespace torch_neuronx {
namespace {

// Helper function to check if a string contains a substring
bool contains(const std::string& str, const std::string& substr) {
  return str.find(substr) != std::string::npos;
}

//============================================================================
// Parameterized Tests for computeTransposePermutation
//============================================================================

struct PermutationTestCase {
  std::string name;
  std::vector<int64_t> source_perm;
  std::vector<int64_t> dest_perm;
  std::vector<int64_t> expected_transpose_perm;
};

class ComputeTransposePermutationTest : public ::testing::TestWithParam<PermutationTestCase> {};

TEST_P(ComputeTransposePermutationTest, ComputesCorrectPermutation) {
  const auto& test_case = GetParam();

  std::vector<int64_t> transpose_perm;
  EXPECT_NO_THROW({
    transpose_perm = computeTransposePermutation(test_case.source_perm, test_case.dest_perm);
  });

  ASSERT_EQ(transpose_perm.size(), test_case.expected_transpose_perm.size());
  for (size_t i = 0; i < transpose_perm.size(); ++i) {
    EXPECT_EQ(transpose_perm[i], test_case.expected_transpose_perm[i])
        << "Mismatch at index " << i << " for test case: " << test_case.name;
  }
}

INSTANTIATE_TEST_SUITE_P(
    PermutationTests, ComputeTransposePermutationTest,
    ::testing::Values(
        PermutationTestCase{"SimpleDimensionSwap", {0, 1, 2, 3}, {0, 2, 1, 3}, {0, 2, 1, 3}},
        PermutationTestCase{"DimensionRotation", {0, 1, 2}, {2, 0, 1}, {2, 0, 1}},
        PermutationTestCase{"IdentityPermutation", {0, 1, 2}, {0, 1, 2}, {0, 1, 2}},
        PermutationTestCase{"CompleteDimensionReversal", {0, 1, 2, 3}, {3, 2, 1, 0}, {3, 2, 1, 0}},
        PermutationTestCase{"NonIdentitySource", {1, 0, 2}, {2, 1, 0}, {2, 0, 1}}),
    [](const ::testing::TestParamInfo<PermutationTestCase>& info) { return info.param.name; });

//============================================================================
// Error Handling Tests for computeTransposePermutation
//============================================================================

TEST(ContiguousTransposeErrorTest, MismatchedPermutationSizes) {
  std::vector<int64_t> source_perm = {0, 1, 2};
  std::vector<int64_t> dest_perm = {0, 1, 2, 3};  // Different size

  EXPECT_THROW({ computeTransposePermutation(source_perm, dest_perm); }, std::invalid_argument);
}

// Note: Permutation validation tests (duplicate, out-of-range, empty) are covered
// in TransposeOpBuilderTest.cpp, which tests validation at the proper entry point.

//============================================================================
// Tests for String Element Type API
//============================================================================

TEST(ContiguousTransposeStringAPITest, GeneratesCorrectMLIR) {
  std::vector<int64_t> source_perm = {0, 1, 2, 3};
  std::vector<int64_t> dest_perm = {0, 2, 1, 3};
  std::vector<int64_t> input_shape = {2, 3, 4, 5};

  std::string mlir = generateContiguousTransposeMlir(source_perm, dest_perm, input_shape, "f32");

  EXPECT_TRUE(contains(mlir, "module"));
  EXPECT_TRUE(contains(mlir, "func.func @main"));
  EXPECT_TRUE(contains(mlir, "stablehlo.transpose"));
  EXPECT_TRUE(contains(mlir, "dims = [0, 2, 1, 3]"));
  EXPECT_TRUE(contains(mlir, "tensor<2x3x4x5xf32>"));
  EXPECT_TRUE(contains(mlir, "tensor<2x4x3x5xf32>"));
}

TEST(ContiguousTransposeStringAPITest, CreatesValidModule) {
  std::vector<int64_t> source_perm = {0, 1, 2};
  std::vector<int64_t> dest_perm = {2, 1, 0};
  std::vector<int64_t> input_shape = {2, 3, 4};

  auto module = createContiguousTransposeModule(source_perm, dest_perm, input_shape, "f32");
  ASSERT_TRUE(module);
  EXPECT_TRUE(module.get());
}

//============================================================================
// Integration Test
//============================================================================

TEST(ContiguousTransposeIntegrationTest, CompleteWorkflow) {
  std::vector<int64_t> source_perm = {0, 1, 2, 3};
  std::vector<int64_t> dest_perm = {0, 2, 1, 3};
  std::vector<int64_t> input_shape = {2, 3, 4, 5};

  // Step 1: Compute transpose permutation
  std::vector<int64_t> transpose_perm = computeTransposePermutation(source_perm, dest_perm);
  ASSERT_EQ(transpose_perm.size(), 4);
  EXPECT_EQ(transpose_perm[0], 0);
  EXPECT_EQ(transpose_perm[1], 2);
  EXPECT_EQ(transpose_perm[2], 1);
  EXPECT_EQ(transpose_perm[3], 3);

  // Step 2: Generate MLIR using the new API
  std::string mlir = generateContiguousTransposeMlir(source_perm, dest_perm, input_shape, "f32");

  // Step 3: Verify MLIR structure
  EXPECT_TRUE(contains(mlir, "module"));
  EXPECT_TRUE(contains(mlir, "stablehlo.transpose"));
  EXPECT_TRUE(contains(mlir, "dims = [0, 2, 1, 3]"));
}

}  // namespace
}  // namespace torch_neuronx
