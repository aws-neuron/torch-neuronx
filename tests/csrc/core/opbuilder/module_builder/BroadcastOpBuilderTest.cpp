#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <vector>

#include "torch_neuronx/csrc/core/opbuilder/module_builder/BroadcastOpBuilder.h"
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

TEST(BroadcastOpBuilderConstructorTest, StringTypeConstructor) {
  std::vector<int64_t> input_shape = {1, 3};
  std::vector<int64_t> output_shape = {2, 3, 2};
  std::vector<int64_t> broadcast_dimensions = {2, 1};

  EXPECT_NO_THROW(
      { BroadcastOpBuilder builder(input_shape, output_shape, broadcast_dimensions, "f32"); });
}

TEST(BroadcastOpBuilderConstructorTest, ValidatesEmptyInputShape) {
  std::vector<int64_t> input_shape = {};
  std::vector<int64_t> output_shape = {2, 3};
  std::vector<int64_t> broadcast_dimensions = {};

  EXPECT_THROW(
      { BroadcastOpBuilder builder(input_shape, output_shape, broadcast_dimensions, "f32"); },
      std::invalid_argument);
}

TEST(BroadcastOpBuilderConstructorTest, ValidatesEmptyOutputShape) {
  std::vector<int64_t> input_shape = {3};
  std::vector<int64_t> output_shape = {};
  std::vector<int64_t> broadcast_dimensions = {0};

  EXPECT_THROW(
      { BroadcastOpBuilder builder(input_shape, output_shape, broadcast_dimensions, "f32"); },
      std::invalid_argument);
}

TEST(BroadcastOpBuilderConstructorTest, ValidatesEmptyBroadcastDimensions) {
  std::vector<int64_t> input_shape = {3};
  std::vector<int64_t> output_shape = {2, 3};
  std::vector<int64_t> broadcast_dimensions = {};

  EXPECT_THROW(
      { BroadcastOpBuilder builder(input_shape, output_shape, broadcast_dimensions, "f32"); },
      std::invalid_argument);
}

TEST(BroadcastOpBuilderConstructorTest, ValidatesNegativeDimension) {
  std::vector<int64_t> input_shape = {2, -3};  // Negative dimension
  std::vector<int64_t> output_shape = {2, 3};
  std::vector<int64_t> broadcast_dimensions = {0, 1};

  EXPECT_THROW(
      { BroadcastOpBuilder builder(input_shape, output_shape, broadcast_dimensions, "f32"); },
      std::invalid_argument);
}

TEST(BroadcastOpBuilderConstructorTest, ValidatesBroadcastDimensionsSizeMismatch) {
  std::vector<int64_t> input_shape = {2, 3};
  std::vector<int64_t> output_shape = {2, 3, 4};
  std::vector<int64_t> broadcast_dimensions = {0};  // Size 1, but input rank is 2

  EXPECT_THROW(
      { BroadcastOpBuilder builder(input_shape, output_shape, broadcast_dimensions, "f32"); },
      std::invalid_argument);
}

TEST(BroadcastOpBuilderConstructorTest, ValidatesBroadcastDimensionOutOfRange) {
  std::vector<int64_t> input_shape = {3};
  std::vector<int64_t> output_shape = {2, 3};
  std::vector<int64_t> broadcast_dimensions = {2};  // Out of range (max is 1)

  EXPECT_THROW(
      { BroadcastOpBuilder builder(input_shape, output_shape, broadcast_dimensions, "f32"); },
      std::invalid_argument);
}

TEST(BroadcastOpBuilderConstructorTest, ValidatesDuplicateBroadcastDimensions) {
  std::vector<int64_t> input_shape = {2, 3};
  std::vector<int64_t> output_shape = {2, 3, 4};
  std::vector<int64_t> broadcast_dimensions = {0, 0};  // Duplicate dimension

  EXPECT_THROW(
      { BroadcastOpBuilder builder(input_shape, output_shape, broadcast_dimensions, "f32"); },
      std::invalid_argument);
}

TEST(BroadcastOpBuilderConstructorTest, ValidatesIncompatibleDimensions) {
  std::vector<int64_t> input_shape = {3};
  std::vector<int64_t> output_shape = {2, 4};  // Input dim 3 doesn't match output dim 4
  std::vector<int64_t> broadcast_dimensions = {1};

  EXPECT_THROW(
      { BroadcastOpBuilder builder(input_shape, output_shape, broadcast_dimensions, "f32"); },
      std::invalid_argument);
}

//============================================================================
// Build Tests
//============================================================================

TEST(BroadcastOpBuilderBuildTest, BuildsValidModule) {
  std::vector<int64_t> input_shape = {1, 3};
  std::vector<int64_t> output_shape = {2, 3, 2};
  std::vector<int64_t> broadcast_dimensions = {2, 1};

  BroadcastOpBuilder builder(input_shape, output_shape, broadcast_dimensions, "f32", true);

  auto module = builder.build();
  ASSERT_TRUE(module);
  EXPECT_TRUE(module.get());
}

TEST(BroadcastOpBuilderBuildTest, GeneratesCorrectMLIR) {
  std::vector<int64_t> input_shape = {1, 3};
  std::vector<int64_t> output_shape = {2, 3, 2};
  std::vector<int64_t> broadcast_dimensions = {2, 1};

  BroadcastOpBuilder builder(input_shape, output_shape, broadcast_dimensions, "f32", true);
  auto module = builder.build();

  std::string mlir = stablehlo_utils::moduleToString(module.get());

  // Verify MLIR structure
  EXPECT_TRUE(contains(mlir, "module"));
  EXPECT_TRUE(contains(mlir, "func.func @main"));
  EXPECT_TRUE(contains(mlir, "stablehlo.broadcast_in_dim"));
  EXPECT_TRUE(contains(mlir, "tensor<1x3xf32>"));
  EXPECT_TRUE(contains(mlir, "tensor<2x3x2xf32>"));
}

TEST(BroadcastOpBuilderBuildTest, MultipleBuildsSucceed) {
  std::vector<int64_t> input_shape = {1, 3};
  std::vector<int64_t> output_shape = {2, 3, 2};
  std::vector<int64_t> broadcast_dimensions = {2, 1};

  // Build multiple modules to ensure context management works
  BroadcastOpBuilder builder1(input_shape, output_shape, broadcast_dimensions, "f32", true);
  BroadcastOpBuilder builder2(input_shape, output_shape, broadcast_dimensions, "f16", true);
  BroadcastOpBuilder builder3(input_shape, output_shape, broadcast_dimensions, "i32", true);

  auto module1 = builder1.build();
  auto module2 = builder2.build();
  auto module3 = builder3.build();

  ASSERT_TRUE(module1);
  ASSERT_TRUE(module2);
  ASSERT_TRUE(module3);
}

//============================================================================
// Element Type Integration Test
//============================================================================

TEST(BroadcastOpBuilderIntegrationTest, InvalidElementType) {
  std::vector<int64_t> input_shape = {1, 3};
  std::vector<int64_t> output_shape = {2, 3, 2};
  std::vector<int64_t> broadcast_dimensions = {2, 1};

  // Verify that invalid types are caught (TypeUtils integration)
  EXPECT_THROW(
      {
        BroadcastOpBuilder builder(input_shape, output_shape, broadcast_dimensions, "invalid_type");
      },
      std::invalid_argument);
}

//============================================================================
// Verification Mode Tests
//============================================================================

TEST(BroadcastOpBuilderVerificationTest, VerificationDisabledByDefault) {
  std::vector<int64_t> input_shape = {1, 3};
  std::vector<int64_t> output_shape = {2, 3, 2};
  std::vector<int64_t> broadcast_dimensions = {2, 1};

  // Should succeed without verification
  BroadcastOpBuilder builder(input_shape, output_shape, broadcast_dimensions, "f32");
  auto module = builder.build();

  ASSERT_TRUE(module);
}

TEST(BroadcastOpBuilderVerificationTest, VerificationCanBeEnabled) {
  std::vector<int64_t> input_shape = {1, 3};
  std::vector<int64_t> output_shape = {2, 3, 2};
  std::vector<int64_t> broadcast_dimensions = {2, 1};

  // Should succeed with verification enabled
  BroadcastOpBuilder builder(input_shape, output_shape, broadcast_dimensions, "f32", true);
  auto module = builder.build();

  ASSERT_TRUE(module);
}

//============================================================================
// Broadcast Pattern Tests
//============================================================================

struct BroadcastTestCase {
  std::string name;
  std::vector<int64_t> input_shape;
  std::vector<int64_t> output_shape;
  std::vector<int64_t> broadcast_dimensions;
  std::string expected_input_type;
  std::string expected_output_type;
};

class BroadcastOpBuilderPatternTest : public ::testing::TestWithParam<BroadcastTestCase> {};

TEST_P(BroadcastOpBuilderPatternTest, GeneratesCorrectBroadcast) {
  const auto& test_case = GetParam();

  BroadcastOpBuilder builder(test_case.input_shape, test_case.output_shape,
                             test_case.broadcast_dimensions, "f32", true);
  auto module = builder.build();

  std::string mlir = stablehlo_utils::moduleToString(module.get());

  EXPECT_TRUE(contains(mlir, test_case.expected_input_type))
      << "Expected input type: " << test_case.expected_input_type
      << " in test case: " << test_case.name;
  EXPECT_TRUE(contains(mlir, test_case.expected_output_type))
      << "Expected output type: " << test_case.expected_output_type
      << " in test case: " << test_case.name;
}

INSTANTIATE_TEST_SUITE_P(
    BroadcastPatterns, BroadcastOpBuilderPatternTest,
    ::testing::Values(
        // Scalar to tensor
        BroadcastTestCase{"ScalarToVector", {1}, {5}, {0}, "tensor<1xf32>", "tensor<5xf32>"},
        BroadcastTestCase{
            "ScalarToMatrix", {1, 1}, {3, 4}, {0, 1}, "tensor<1x1xf32>", "tensor<3x4xf32>"},
        // Vector to higher dimensions
        BroadcastTestCase{"VectorToMatrix", {3}, {2, 3}, {1}, "tensor<3xf32>", "tensor<2x3xf32>"},
        BroadcastTestCase{"VectorTo3D", {4}, {2, 3, 4}, {2}, "tensor<4xf32>", "tensor<2x3x4xf32>"},
        // Mixed size-1 broadcasting
        BroadcastTestCase{
            "MixedBroadcast", {1, 3}, {2, 3, 2}, {2, 1}, "tensor<1x3xf32>", "tensor<2x3x2xf32>"},
        // Same rank broadcasting
        BroadcastTestCase{"SameRankBroadcast",
                          {1, 3, 1},
                          {2, 3, 4},
                          {0, 1, 2},
                          "tensor<1x3x1xf32>",
                          "tensor<2x3x4xf32>"},
        // Identity (no actual broadcast)
        BroadcastTestCase{
            "Identity", {2, 3}, {2, 3}, {0, 1}, "tensor<2x3xf32>", "tensor<2x3xf32>"}),
    [](const ::testing::TestParamInfo<BroadcastTestCase>& info) { return info.param.name; });

}  // namespace
}  // namespace torch_neuronx
