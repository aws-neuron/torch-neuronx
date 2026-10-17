#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <vector>

#include "torch_neuronx/csrc/core/opbuilder/module_builder/SliceOpBuilder.h"
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

TEST(SliceOpBuilderConstructorTest, StringTypeConstructor) {
  std::vector<int64_t> input_shape = {10, 20, 30};
  std::vector<int64_t> start_indices = {0, 5, 10};
  std::vector<int64_t> limit_indices = {5, 15, 25};
  std::vector<int64_t> strides = {1, 1, 1};

  EXPECT_NO_THROW(
      { SliceOpBuilder builder(input_shape, start_indices, limit_indices, strides, "f32"); });
}

TEST(SliceOpBuilderConstructorTest, ValidatesEmptyInputShape) {
  std::vector<int64_t> input_shape = {};
  std::vector<int64_t> start_indices = {};
  std::vector<int64_t> limit_indices = {};
  std::vector<int64_t> strides = {};

  EXPECT_THROW(
      { SliceOpBuilder builder(input_shape, start_indices, limit_indices, strides, "f32"); },
      std::invalid_argument);
}

TEST(SliceOpBuilderConstructorTest, ValidatesStartIndicesSize) {
  std::vector<int64_t> input_shape = {10, 20, 30};
  std::vector<int64_t> start_indices = {0, 5};  // Size 2, should be 3
  std::vector<int64_t> limit_indices = {5, 15, 25};
  std::vector<int64_t> strides = {1, 1, 1};

  EXPECT_THROW(
      { SliceOpBuilder builder(input_shape, start_indices, limit_indices, strides, "f32"); },
      std::invalid_argument);
}

TEST(SliceOpBuilderConstructorTest, ValidatesLimitIndicesSize) {
  std::vector<int64_t> input_shape = {10, 20, 30};
  std::vector<int64_t> start_indices = {0, 5, 10};
  std::vector<int64_t> limit_indices = {5, 15};  // Size 2, should be 3
  std::vector<int64_t> strides = {1, 1, 1};

  EXPECT_THROW(
      { SliceOpBuilder builder(input_shape, start_indices, limit_indices, strides, "f32"); },
      std::invalid_argument);
}

TEST(SliceOpBuilderConstructorTest, ValidatesStridesSize) {
  std::vector<int64_t> input_shape = {10, 20, 30};
  std::vector<int64_t> start_indices = {0, 5, 10};
  std::vector<int64_t> limit_indices = {5, 15, 25};
  std::vector<int64_t> strides = {1, 1};  // Size 2, should be 3

  EXPECT_THROW(
      { SliceOpBuilder builder(input_shape, start_indices, limit_indices, strides, "f32"); },
      std::invalid_argument);
}

TEST(SliceOpBuilderConstructorTest, ValidatesPositiveStrides) {
  std::vector<int64_t> input_shape = {10, 20, 30};
  std::vector<int64_t> start_indices = {0, 5, 10};
  std::vector<int64_t> limit_indices = {5, 15, 25};
  std::vector<int64_t> strides = {1, 0, 1};  // Stride of 0 is invalid

  EXPECT_THROW(
      { SliceOpBuilder builder(input_shape, start_indices, limit_indices, strides, "f32"); },
      std::invalid_argument);
}

TEST(SliceOpBuilderConstructorTest, ValidatesStartIndicesInBounds) {
  std::vector<int64_t> input_shape = {10, 20, 30};
  std::vector<int64_t> start_indices = {0, 5, 35};  // 35 > 30, out of bounds
  std::vector<int64_t> limit_indices = {5, 15, 40};
  std::vector<int64_t> strides = {1, 1, 1};

  EXPECT_THROW(
      { SliceOpBuilder builder(input_shape, start_indices, limit_indices, strides, "f32"); },
      std::invalid_argument);
}

TEST(SliceOpBuilderConstructorTest, ValidatesLimitIndicesInBounds) {
  std::vector<int64_t> input_shape = {10, 20, 30};
  std::vector<int64_t> start_indices = {0, 5, 10};
  std::vector<int64_t> limit_indices = {5, 25, 25};  // 25 > 20, out of bounds for dimension 1
  std::vector<int64_t> strides = {1, 1, 1};

  EXPECT_THROW(
      { SliceOpBuilder builder(input_shape, start_indices, limit_indices, strides, "f32"); },
      std::invalid_argument);
}

TEST(SliceOpBuilderConstructorTest, ValidatesStartLessThanLimit) {
  std::vector<int64_t> input_shape = {10, 20, 30};
  std::vector<int64_t> start_indices = {0, 15, 10};  // start[1] = 15
  std::vector<int64_t> limit_indices = {5, 10, 25};  // limit[1] = 10, start > limit
  std::vector<int64_t> strides = {1, 1, 1};

  EXPECT_THROW(
      { SliceOpBuilder builder(input_shape, start_indices, limit_indices, strides, "f32"); },
      std::invalid_argument);
}

TEST(SliceOpBuilderConstructorTest, ValidatesNegativeDimension) {
  std::vector<int64_t> input_shape = {10, -20, 30};  // Negative dimension
  std::vector<int64_t> start_indices = {0, 5, 10};
  std::vector<int64_t> limit_indices = {5, 15, 25};
  std::vector<int64_t> strides = {1, 1, 1};

  EXPECT_THROW(
      { SliceOpBuilder builder(input_shape, start_indices, limit_indices, strides, "f32"); },
      std::invalid_argument);
}

//============================================================================
// Build Tests
//============================================================================

TEST(SliceOpBuilderBuildTest, BuildsValidModule) {
  std::vector<int64_t> input_shape = {10, 20, 30};
  std::vector<int64_t> start_indices = {0, 5, 10};
  std::vector<int64_t> limit_indices = {5, 15, 25};
  std::vector<int64_t> strides = {1, 1, 1};

  SliceOpBuilder builder(input_shape, start_indices, limit_indices, strides, "f32", true);

  auto module = builder.build();
  ASSERT_TRUE(module);
  EXPECT_TRUE(module.get());
}

TEST(SliceOpBuilderBuildTest, GeneratesCorrectMLIR) {
  std::vector<int64_t> input_shape = {10, 20, 30};
  std::vector<int64_t> start_indices = {0, 5, 10};
  std::vector<int64_t> limit_indices = {5, 15, 25};
  std::vector<int64_t> strides = {1, 1, 1};

  SliceOpBuilder builder(input_shape, start_indices, limit_indices, strides, "f32", true);
  auto module = builder.build();

  std::string mlir = stablehlo_utils::moduleToString(module.get());

  // Verify MLIR structure
  EXPECT_TRUE(contains(mlir, "module"));
  EXPECT_TRUE(contains(mlir, "func.func @main"));
  EXPECT_TRUE(contains(mlir, "stablehlo.slice"));
  EXPECT_TRUE(contains(mlir, "tensor<10x20x30xf32>"));
  EXPECT_TRUE(contains(mlir, "tensor<5x10x15xf32>"));  // Output shape
  EXPECT_TRUE(contains(mlir, "[0:5, 5:15, 10:25]"));   // Slice notation
}

TEST(SliceOpBuilderBuildTest, MultipleBuildsSucceed) {
  std::vector<int64_t> input_shape = {10, 20, 30};
  std::vector<int64_t> start_indices = {0, 5, 10};
  std::vector<int64_t> limit_indices = {5, 15, 25};
  std::vector<int64_t> strides = {1, 1, 1};

  // Build multiple modules to ensure context management works
  SliceOpBuilder builder1(input_shape, start_indices, limit_indices, strides, "f32", true);
  SliceOpBuilder builder2(input_shape, start_indices, limit_indices, strides, "f16", true);
  SliceOpBuilder builder3(input_shape, start_indices, limit_indices, strides, "i32", true);

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

TEST(SliceOpBuilderIntegrationTest, InvalidElementType) {
  std::vector<int64_t> input_shape = {10, 20, 30};
  std::vector<int64_t> start_indices = {0, 5, 10};
  std::vector<int64_t> limit_indices = {5, 15, 25};
  std::vector<int64_t> strides = {1, 1, 1};

  // Verify that invalid types are caught (TypeUtils integration)
  EXPECT_THROW(
      {
        SliceOpBuilder builder(input_shape, start_indices, limit_indices, strides, "invalid_type");
      },
      std::invalid_argument);
}

//============================================================================
// Verification Mode Tests
//============================================================================

TEST(SliceOpBuilderVerificationTest, VerificationDisabledByDefault) {
  std::vector<int64_t> input_shape = {10, 20, 30};
  std::vector<int64_t> start_indices = {0, 5, 10};
  std::vector<int64_t> limit_indices = {5, 15, 25};
  std::vector<int64_t> strides = {1, 1, 1};

  // Should succeed without verification
  SliceOpBuilder builder(input_shape, start_indices, limit_indices, strides, "f32");
  auto module = builder.build();

  ASSERT_TRUE(module);
}

TEST(SliceOpBuilderVerificationTest, VerificationCanBeEnabled) {
  std::vector<int64_t> input_shape = {10, 20, 30};
  std::vector<int64_t> start_indices = {0, 5, 10};
  std::vector<int64_t> limit_indices = {5, 15, 25};
  std::vector<int64_t> strides = {1, 1, 1};

  // Should succeed with verification enabled
  SliceOpBuilder builder(input_shape, start_indices, limit_indices, strides, "f32", true);
  auto module = builder.build();

  ASSERT_TRUE(module);
}

//============================================================================
// Slice Pattern Tests
//============================================================================

struct SliceTestCase {
  std::string name;
  std::vector<int64_t> input_shape;
  std::vector<int64_t> start_indices;
  std::vector<int64_t> limit_indices;
  std::vector<int64_t> strides;
  std::string expected_input_type;
  std::string expected_output_type;
  std::string expected_start;
  std::string expected_limit;
  std::string expected_strides;
};

class SliceOpBuilderPatternTest : public ::testing::TestWithParam<SliceTestCase> {};

TEST_P(SliceOpBuilderPatternTest, GeneratesCorrectSlice) {
  const auto& test_case = GetParam();

  SliceOpBuilder builder(test_case.input_shape, test_case.start_indices, test_case.limit_indices,
                         test_case.strides, "f32", true);
  auto module = builder.build();

  std::string mlir = stablehlo_utils::moduleToString(module.get());

  EXPECT_TRUE(contains(mlir, test_case.expected_input_type))
      << "Expected input type: " << test_case.expected_input_type
      << " in test case: " << test_case.name;
  EXPECT_TRUE(contains(mlir, test_case.expected_output_type))
      << "Expected output type: " << test_case.expected_output_type
      << " in test case: " << test_case.name;
  EXPECT_TRUE(contains(mlir, test_case.expected_start))
      << "Expected start indices: " << test_case.expected_start
      << " in test case: " << test_case.name;
  EXPECT_TRUE(contains(mlir, test_case.expected_limit))
      << "Expected limit indices: " << test_case.expected_limit
      << " in test case: " << test_case.name;
  EXPECT_TRUE(contains(mlir, test_case.expected_strides))
      << "Expected strides: " << test_case.expected_strides << " in test case: " << test_case.name;
}

INSTANTIATE_TEST_SUITE_P(SlicePatterns, SliceOpBuilderPatternTest,
                         ::testing::Values(SliceTestCase{"BasicSlice",
                                                         {10, 20, 30},
                                                         {2, 5, 10},
                                                         {8, 15, 25},
                                                         {1, 1, 1},
                                                         "tensor<10x20x30xf32>",
                                                         "tensor<6x10x15xf32>",
                                                         "[2:8",
                                                         "5:15",
                                                         "10:25]"},
                                           SliceTestCase{"StrideSlice",
                                                         {20, 30},
                                                         {0, 0},
                                                         {20, 30},
                                                         {2, 3},
                                                         "tensor<20x30xf32>",
                                                         "tensor<10x10xf32>",
                                                         "[0:20:2",
                                                         "0:30:3",
                                                         "]"},
                                           SliceTestCase{"FullDimensionSlice",
                                                         {10, 20, 30},
                                                         {0, 0, 0},
                                                         {10, 20, 30},
                                                         {1, 1, 1},
                                                         "tensor<10x20x30xf32>",
                                                         "tensor<10x20x30xf32>",
                                                         "[0:10",
                                                         "0:20",
                                                         "0:30]"},
                                           SliceTestCase{"MultiDimensionalSlice",
                                                         {8, 16, 32},
                                                         {1, 4, 8},
                                                         {7, 12, 24},
                                                         {2, 2, 4},
                                                         "tensor<8x16x32xf32>",
                                                         "tensor<3x4x4xf32>",
                                                         "[1:7:2",
                                                         "4:12:2",
                                                         "8:24:4]"},
                                           SliceTestCase{"SingleElementSlice",
                                                         {10, 20},
                                                         {5, 10},
                                                         {6, 11},
                                                         {1, 1},
                                                         "tensor<10x20xf32>",
                                                         "tensor<1x1xf32>",
                                                         "[5:6",
                                                         "10:11",
                                                         "]"}),
                         [](const ::testing::TestParamInfo<SliceTestCase>& info) {
                           return info.param.name;
                         });

}  // namespace
}  // namespace torch_neuronx
