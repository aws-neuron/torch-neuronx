#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <vector>

#include "torch_neuronx/csrc/core/opbuilder/module_builder/ReshapeOpBuilder.h"
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

TEST(ReshapeOpBuilderConstructorTest, StringTypeConstructor) {
  std::vector<int64_t> input_shape = {2, 3, 4};
  std::vector<int64_t> output_shape = {6, 4};

  EXPECT_NO_THROW({ ReshapeOpBuilder builder(input_shape, output_shape, "f32"); });
}

TEST(ReshapeOpBuilderConstructorTest, ValidatesEmptyInputShape) {
  std::vector<int64_t> input_shape = {};
  std::vector<int64_t> output_shape = {24};

  EXPECT_THROW(
      { ReshapeOpBuilder builder(input_shape, output_shape, "f32"); }, std::invalid_argument);
}

TEST(ReshapeOpBuilderConstructorTest, ValidatesEmptyOutputShape) {
  std::vector<int64_t> input_shape = {2, 3, 4};
  std::vector<int64_t> output_shape = {};

  EXPECT_THROW(
      { ReshapeOpBuilder builder(input_shape, output_shape, "f32"); }, std::invalid_argument);
}

TEST(ReshapeOpBuilderConstructorTest, ValidatesNegativeDimension) {
  std::vector<int64_t> input_shape = {2, -3, 4};  // Negative dimension
  std::vector<int64_t> output_shape = {24};

  EXPECT_THROW(
      { ReshapeOpBuilder builder(input_shape, output_shape, "f32"); }, std::invalid_argument);
}

TEST(ReshapeOpBuilderConstructorTest, ValidatesIncompatibleShapes) {
  std::vector<int64_t> input_shape = {2, 3, 4};  // 24 elements
  std::vector<int64_t> output_shape = {5, 5};    // 25 elements - mismatch!

  EXPECT_THROW(
      { ReshapeOpBuilder builder(input_shape, output_shape, "f32"); }, std::invalid_argument);
}

//============================================================================
// Build Tests
//============================================================================

TEST(ReshapeOpBuilderBuildTest, BuildsValidModule) {
  std::vector<int64_t> input_shape = {2, 3, 4};
  std::vector<int64_t> output_shape = {6, 4};

  ReshapeOpBuilder builder(input_shape, output_shape, "f32", true);

  auto module = builder.build();
  ASSERT_TRUE(module);
  EXPECT_TRUE(module.get());
}

TEST(ReshapeOpBuilderBuildTest, GeneratesCorrectMLIR) {
  std::vector<int64_t> input_shape = {2, 3, 4};
  std::vector<int64_t> output_shape = {6, 4};

  ReshapeOpBuilder builder(input_shape, output_shape, "f32", true);
  auto module = builder.build();

  std::string mlir = stablehlo_utils::moduleToString(module.get());

  // Verify MLIR structure
  EXPECT_TRUE(contains(mlir, "module"));
  EXPECT_TRUE(contains(mlir, "func.func @main"));
  EXPECT_TRUE(contains(mlir, "stablehlo.reshape"));
  EXPECT_TRUE(contains(mlir, "tensor<2x3x4xf32>"));
  EXPECT_TRUE(contains(mlir, "tensor<6x4xf32>"));
}

TEST(ReshapeOpBuilderBuildTest, MultipleBuildsSucceed) {
  std::vector<int64_t> input_shape = {2, 3, 4};
  std::vector<int64_t> output_shape = {6, 4};

  // Build multiple modules to ensure context management works
  ReshapeOpBuilder builder1(input_shape, output_shape, "f32", true);
  ReshapeOpBuilder builder2(input_shape, output_shape, "f16", true);
  ReshapeOpBuilder builder3(input_shape, output_shape, "i32", true);

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

TEST(ReshapeOpBuilderIntegrationTest, InvalidElementType) {
  std::vector<int64_t> input_shape = {2, 3, 4};
  std::vector<int64_t> output_shape = {6, 4};

  // Verify that invalid types are caught (TypeUtils integration)
  EXPECT_THROW(
      { ReshapeOpBuilder builder(input_shape, output_shape, "invalid_type"); },
      std::invalid_argument);
}

//============================================================================
// Verification Mode Tests
//============================================================================

TEST(ReshapeOpBuilderVerificationTest, VerificationDisabledByDefault) {
  std::vector<int64_t> input_shape = {2, 3, 4};
  std::vector<int64_t> output_shape = {6, 4};

  // Should succeed without verification
  ReshapeOpBuilder builder(input_shape, output_shape, "f32");
  auto module = builder.build();

  ASSERT_TRUE(module);
}

TEST(ReshapeOpBuilderVerificationTest, VerificationCanBeEnabled) {
  std::vector<int64_t> input_shape = {2, 3, 4};
  std::vector<int64_t> output_shape = {6, 4};

  // Should succeed with verification enabled
  ReshapeOpBuilder builder(input_shape, output_shape, "f32", true);
  auto module = builder.build();

  ASSERT_TRUE(module);
}

//============================================================================
// Reshape Pattern Tests
//============================================================================

struct ReshapeTestCase {
  std::string name;
  std::vector<int64_t> input_shape;
  std::vector<int64_t> output_shape;
  std::string expected_input_type;
  std::string expected_output_type;
};

class ReshapeOpBuilderPatternTest : public ::testing::TestWithParam<ReshapeTestCase> {};

TEST_P(ReshapeOpBuilderPatternTest, GeneratesCorrectReshape) {
  const auto& test_case = GetParam();

  ReshapeOpBuilder builder(test_case.input_shape, test_case.output_shape, "f32", true);
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
    ReshapePatterns, ReshapeOpBuilderPatternTest,
    ::testing::Values(
        ReshapeTestCase{"Flatten", {2, 3, 4}, {24}, "tensor<2x3x4xf32>", "tensor<24xf32>"},
        ReshapeTestCase{"Unflatten", {24}, {2, 3, 4}, "tensor<24xf32>", "tensor<2x3x4xf32>"},
        ReshapeTestCase{"DimensionReshape", {2, 6}, {3, 4}, "tensor<2x6xf32>", "tensor<3x4xf32>"},
        ReshapeTestCase{"DimensionExpand", {12}, {3, 2, 2}, "tensor<12xf32>", "tensor<3x2x2xf32>"}),
    [](const ::testing::TestParamInfo<ReshapeTestCase>& info) { return info.param.name; });

}  // namespace
}  // namespace torch_neuronx
