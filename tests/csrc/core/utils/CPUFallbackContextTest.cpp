#include <gtest/gtest.h>
#include <torch/torch.h>

#include "torch_neuronx/csrc/core/utils/CPUFallbackContext.h"

namespace at::neuron {

class CPUFallbackContextTest : public ::testing::Test {};

// Test default constructor creates empty context
TEST_F(CPUFallbackContextTest, DefaultConstructorCreatesEmptyContext) {
  CPUFallbackContext ctx;

  EXPECT_TRUE(ctx.original_inputs.empty());
  EXPECT_TRUE(ctx.tensor_metadata.empty());
  EXPECT_TRUE(ctx.tensor_data_ptrs.empty());
  EXPECT_TRUE(ctx.IsEmpty());
}

// Test constructor with inputs
TEST_F(CPUFallbackContextTest, ConstructorWithInputs) {
  auto tensor = torch::randn({2, 3});
  std::vector<c10::IValue> inputs = {c10::IValue(tensor), c10::IValue(42), c10::IValue(3.14)};

  CPUFallbackContext ctx(inputs);

  ASSERT_EQ(ctx.original_inputs.size(), 3);
  EXPECT_TRUE(ctx.original_inputs[0].isTensor());
  EXPECT_TRUE(ctx.original_inputs[1].isInt());
  EXPECT_EQ(ctx.original_inputs[1].toInt(), 42);
  EXPECT_TRUE(ctx.original_inputs[2].isDouble());
  EXPECT_DOUBLE_EQ(ctx.original_inputs[2].toDouble(), 3.14);

  EXPECT_FALSE(ctx.IsEmpty());
}

// Test IsEmpty returns true only when all fields are empty
TEST_F(CPUFallbackContextTest, IsEmptyReturnsTrueOnlyWhenAllFieldsEmpty) {
  // Empty context
  CPUFallbackContext empty_ctx;
  EXPECT_TRUE(empty_ctx.IsEmpty());

  // Context with only original_inputs
  CPUFallbackContext ctx_with_inputs;
  ctx_with_inputs.original_inputs.push_back(c10::IValue(1));
  EXPECT_FALSE(ctx_with_inputs.IsEmpty());
}

// Test move semantics work correctly
TEST_F(CPUFallbackContextTest, MoveConstructorWorks) {
  auto tensor = torch::randn({2});
  std::vector<c10::IValue> inputs = {c10::IValue(tensor), c10::IValue(42)};

  CPUFallbackContext original(inputs);
  CPUFallbackContext moved(std::move(original));

  ASSERT_EQ(moved.original_inputs.size(), 2);
  EXPECT_TRUE(moved.original_inputs[0].isTensor());
  EXPECT_TRUE(moved.original_inputs[1].isInt());
  EXPECT_EQ(moved.original_inputs[1].toInt(), 42);
}

// Test with various IValue types
TEST_F(CPUFallbackContextTest, SupportsVariousIValueTypes) {
  std::vector<c10::IValue> inputs;
  inputs.push_back(c10::IValue(42));       // int
  inputs.push_back(c10::IValue(3.14));     // double
  inputs.push_back(c10::IValue(true));     // bool
  inputs.push_back(c10::IValue("hello"));  // string
  inputs.push_back(c10::IValue());         // None

  CPUFallbackContext ctx(inputs);

  ASSERT_EQ(ctx.original_inputs.size(), 5);
  EXPECT_TRUE(ctx.original_inputs[0].isInt());
  EXPECT_TRUE(ctx.original_inputs[1].isDouble());
  EXPECT_TRUE(ctx.original_inputs[2].isBool());
  EXPECT_TRUE(ctx.original_inputs[3].isString());
  EXPECT_TRUE(ctx.original_inputs[4].isNone());
}

}  // namespace at::neuron
