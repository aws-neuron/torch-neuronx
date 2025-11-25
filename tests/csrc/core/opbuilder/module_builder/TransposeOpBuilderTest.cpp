#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <vector>

#include "torch_neuronx/csrc/core/opbuilder/module_builder/TransposeOpBuilder.h"
#include "torch_neuronx/csrc/core/opbuilder/utility/StableHloUtils.h"

namespace torch_neuronx {
namespace {

// Helper function to check if a string contains a substring
bool contains(const std::string& str, const std::string& substr) {
  return str.find(substr) != std::string::npos;
}

//============================================================================
// Constructor Tests - String Element Type
//============================================================================

TEST(TransposeOpBuilderConstructorTest, StringTypeConstructor) {
  std::vector<int64_t> permutation = {0, 2, 1, 3};
  std::vector<int64_t> input_shape = {2, 3, 4, 5};

  EXPECT_NO_THROW({ TransposeOpBuilder builder(permutation, input_shape, "f32"); });
}

TEST(TransposeOpBuilderConstructorTest, ValidatesPermutationSize) {
  std::vector<int64_t> permutation = {0, 1};     // Size 2
  std::vector<int64_t> input_shape = {2, 3, 4};  // Size 3 - mismatch!

  EXPECT_THROW(
      { TransposeOpBuilder builder(permutation, input_shape, "f32"); }, std::invalid_argument);
}

TEST(TransposeOpBuilderConstructorTest, ValidatesPermutationValues) {
  std::vector<int64_t> permutation = {0, 1, 5};  // 5 is out of range
  std::vector<int64_t> input_shape = {2, 3, 4};

  EXPECT_THROW(
      { TransposeOpBuilder builder(permutation, input_shape, "f32"); }, std::invalid_argument);
}

TEST(TransposeOpBuilderConstructorTest, ValidatesEmptyShape) {
  std::vector<int64_t> permutation = {};
  std::vector<int64_t> input_shape = {};

  EXPECT_THROW(
      { TransposeOpBuilder builder(permutation, input_shape, "f32"); }, std::invalid_argument);
}

TEST(TransposeOpBuilderConstructorTest, ValidatesNegativeDimension) {
  std::vector<int64_t> permutation = {0, 1, 2};
  std::vector<int64_t> input_shape = {2, -3, 4};  // Negative dimension

  EXPECT_THROW(
      { TransposeOpBuilder builder(permutation, input_shape, "f32"); }, std::invalid_argument);
}

//============================================================================
// Build Tests
//============================================================================

TEST(TransposeOpBuilderBuildTest, BuildsValidModule) {
  std::vector<int64_t> permutation = {0, 2, 1, 3};
  std::vector<int64_t> input_shape = {2, 3, 4, 5};

  TransposeOpBuilder builder(permutation, input_shape, "f32");

  auto module = builder.build();
  ASSERT_TRUE(module);
  EXPECT_TRUE(module.get());
}

TEST(TransposeOpBuilderBuildTest, GeneratesCorrectMLIR) {
  std::vector<int64_t> permutation = {0, 2, 1, 3};
  std::vector<int64_t> input_shape = {2, 3, 4, 5};

  TransposeOpBuilder builder(permutation, input_shape, "f32");
  auto module = builder.build();

  std::string mlir = stablehlo_utils::moduleToString(module.get());

  // Verify MLIR structure
  EXPECT_TRUE(contains(mlir, "module"));
  EXPECT_TRUE(contains(mlir, "func.func @main"));
  EXPECT_TRUE(contains(mlir, "stablehlo.transpose"));
  EXPECT_TRUE(contains(mlir, "dims = [0, 2, 1, 3]"));
  EXPECT_TRUE(contains(mlir, "tensor<2x3x4x5xf32>"));
  EXPECT_TRUE(contains(mlir, "tensor<2x4x3x5xf32>"));
}

TEST(TransposeOpBuilderBuildTest, MultipleBuildsSucceed) {
  std::vector<int64_t> permutation = {2, 1, 0};
  std::vector<int64_t> input_shape = {2, 3, 4};

  // Build multiple modules to ensure context management works
  TransposeOpBuilder builder1(permutation, input_shape, "f32");
  TransposeOpBuilder builder2(permutation, input_shape, "f16");
  TransposeOpBuilder builder3(permutation, input_shape, "i32");

  auto module1 = builder1.build();
  auto module2 = builder2.build();
  auto module3 = builder3.build();

  ASSERT_TRUE(module1);
  ASSERT_TRUE(module2);
  ASSERT_TRUE(module3);
}

//============================================================================
// Element Type Integration Test
// Note: Detailed element type testing should be in TypeUtilsTest
//============================================================================

TEST(TransposeOpBuilderIntegrationTest, InvalidElementType) {
  std::vector<int64_t> permutation = {0, 2, 1};
  std::vector<int64_t> input_shape = {2, 3, 4};

  // Verify that invalid types are caught (TypeUtils integration)
  EXPECT_THROW(
      { TransposeOpBuilder builder(permutation, input_shape, "invalid_type"); },
      std::invalid_argument);
}

//============================================================================
// Verification Mode Tests
//============================================================================

TEST(TransposeOpBuilderVerificationTest, VerificationDisabledByDefault) {
  std::vector<int64_t> permutation = {0, 2, 1};
  std::vector<int64_t> input_shape = {2, 3, 4};

  // Should succeed without verification
  TransposeOpBuilder builder(permutation, input_shape, "f32");
  auto module = builder.build();

  ASSERT_TRUE(module);
}

TEST(TransposeOpBuilderVerificationTest, VerificationCanBeEnabled) {
  std::vector<int64_t> permutation = {0, 2, 1};
  std::vector<int64_t> input_shape = {2, 3, 4};

  // Should succeed with verification enabled
  TransposeOpBuilder builder(permutation, input_shape, "f32", true);
  auto module = builder.build();

  ASSERT_TRUE(module);
}

//============================================================================
// Permutation Tests
//============================================================================

struct PermutationTestCase {
  std::string name;
  std::vector<int64_t> permutation;
  std::vector<int64_t> input_shape;
  std::string expected_dims;
};

class TransposeOpBuilderPermutationTest : public ::testing::TestWithParam<PermutationTestCase> {};

TEST_P(TransposeOpBuilderPermutationTest, GeneratesCorrectPermutation) {
  const auto& test_case = GetParam();

  TransposeOpBuilder builder(test_case.permutation, test_case.input_shape, "f32");
  auto module = builder.build();

  std::string mlir = stablehlo_utils::moduleToString(module.get());

  EXPECT_TRUE(contains(mlir, test_case.expected_dims))
      << "Expected dims: " << test_case.expected_dims << " in test case: " << test_case.name;
}

INSTANTIATE_TEST_SUITE_P(
    Permutations, TransposeOpBuilderPermutationTest,
    ::testing::Values(
        PermutationTestCase{
            "SimpleDimensionSwap", {0, 2, 1, 3}, {2, 3, 4, 5}, "dims = [0, 2, 1, 3]"},
        PermutationTestCase{"DimensionRotation", {2, 0, 1}, {2, 3, 4}, "dims = [2, 0, 1]"},
        PermutationTestCase{"IdentityPermutation", {0, 1, 2}, {2, 3, 4}, "dims = [0, 1, 2]"},
        PermutationTestCase{
            "CompleteDimensionReversal", {3, 2, 1, 0}, {2, 3, 4, 5}, "dims = [3, 2, 1, 0]"}),
    [](const ::testing::TestParamInfo<PermutationTestCase>& info) { return info.param.name; });

}  // namespace
}  // namespace torch_neuronx
