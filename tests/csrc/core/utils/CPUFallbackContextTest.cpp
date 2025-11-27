#include <gtest/gtest.h>
#include <torch/torch.h>

#include "torch_neuronx/csrc/core/utils/CPUFallbackContext.h"

namespace at::neuron {

class CPUFallbackContextTest : public ::testing::Test {};

// Test default constructor creates empty context
TEST_F(CPUFallbackContextTest, DefaultConstructorCreatesEmptyContext) {
  CPUFallbackContext ctx;

  EXPECT_TRUE(ctx.static_args.empty());
  EXPECT_TRUE(ctx.static_arg_positions.empty());
  EXPECT_TRUE(ctx.list_arg_positions.empty());
  EXPECT_TRUE(ctx.list_arg_lengths.empty());
  EXPECT_TRUE(ctx.IsEmpty());
  EXPECT_FALSE(ctx.HasData());
}

// Test full constructor with all fields populated
TEST_F(CPUFallbackContextTest, FullConstructorPopulatesAllFields) {
  std::vector<c10::IValue> args = {c10::IValue(42), c10::IValue(3.14)};
  std::vector<int64_t> positions = {0, 2};
  std::vector<int64_t> list_pos = {1};
  std::vector<int64_t> list_len = {3};

  CPUFallbackContext ctx(args, positions, list_pos, list_len);

  ASSERT_EQ(ctx.static_args.size(), 2);
  EXPECT_EQ(ctx.static_args[0].toInt(), 42);
  EXPECT_DOUBLE_EQ(ctx.static_args[1].toDouble(), 3.14);

  ASSERT_EQ(ctx.static_arg_positions.size(), 2);
  EXPECT_EQ(ctx.static_arg_positions[0], 0);
  EXPECT_EQ(ctx.static_arg_positions[1], 2);

  ASSERT_EQ(ctx.list_arg_positions.size(), 1);
  EXPECT_EQ(ctx.list_arg_positions[0], 1);

  ASSERT_EQ(ctx.list_arg_lengths.size(), 1);
  EXPECT_EQ(ctx.list_arg_lengths[0], 3);

  EXPECT_FALSE(ctx.IsEmpty());
  EXPECT_TRUE(ctx.HasData());
}

// Test IsEmpty returns true only when all fields are empty
TEST_F(CPUFallbackContextTest, IsEmptyReturnsTrueOnlyWhenAllFieldsEmpty) {
  // Empty context
  CPUFallbackContext empty_ctx;
  EXPECT_TRUE(empty_ctx.IsEmpty());

  // Context with only static_args
  CPUFallbackContext ctx_with_args;
  ctx_with_args.static_args.push_back(c10::IValue(1));
  EXPECT_FALSE(ctx_with_args.IsEmpty());

  // Context with only static_arg_positions
  CPUFallbackContext ctx_with_positions;
  ctx_with_positions.static_arg_positions.push_back(0);
  EXPECT_FALSE(ctx_with_positions.IsEmpty());

  // Context with only list_arg_positions
  CPUFallbackContext ctx_with_list_pos;
  ctx_with_list_pos.list_arg_positions.push_back(1);
  EXPECT_FALSE(ctx_with_list_pos.IsEmpty());

  // Context with only list_arg_lengths
  CPUFallbackContext ctx_with_list_len;
  ctx_with_list_len.list_arg_lengths.push_back(2);
  EXPECT_FALSE(ctx_with_list_len.IsEmpty());
}

// Test HasData returns true when static_args or list_arg_positions are non-empty
TEST_F(CPUFallbackContextTest, HasDataReturnsTrueWhenRelevantFieldsPopulated) {
  // Empty context
  CPUFallbackContext empty_ctx;
  EXPECT_FALSE(empty_ctx.HasData());

  // Context with static_args
  CPUFallbackContext ctx_with_args;
  ctx_with_args.static_args.push_back(c10::IValue(1));
  EXPECT_TRUE(ctx_with_args.HasData());

  // Context with list_arg_positions
  CPUFallbackContext ctx_with_list_pos;
  ctx_with_list_pos.list_arg_positions.push_back(1);
  EXPECT_TRUE(ctx_with_list_pos.HasData());

  // Context with only static_arg_positions (no static_args)
  // HasData checks static_args and list_arg_positions, not positions
  CPUFallbackContext ctx_with_only_positions;
  ctx_with_only_positions.static_arg_positions.push_back(0);
  EXPECT_FALSE(ctx_with_only_positions.HasData());
}

// Test move semantics work correctly
TEST_F(CPUFallbackContextTest, MoveConstructorWorks) {
  std::vector<c10::IValue> args = {c10::IValue(42)};
  std::vector<int64_t> positions = {0};
  std::vector<int64_t> list_pos = {1};
  std::vector<int64_t> list_len = {3};

  CPUFallbackContext original(args, positions, list_pos, list_len);
  CPUFallbackContext moved(std::move(original));

  ASSERT_EQ(moved.static_args.size(), 1);
  EXPECT_EQ(moved.static_args[0].toInt(), 42);
  ASSERT_EQ(moved.static_arg_positions.size(), 1);
  EXPECT_EQ(moved.static_arg_positions[0], 0);
  ASSERT_EQ(moved.list_arg_positions.size(), 1);
  EXPECT_EQ(moved.list_arg_positions[0], 1);
  ASSERT_EQ(moved.list_arg_lengths.size(), 1);
  EXPECT_EQ(moved.list_arg_lengths[0], 3);
}

// Test with various IValue types
TEST_F(CPUFallbackContextTest, SupportsVariousIValueTypes) {
  std::vector<c10::IValue> args;
  args.push_back(c10::IValue(42));       // int
  args.push_back(c10::IValue(3.14));     // double
  args.push_back(c10::IValue(true));     // bool
  args.push_back(c10::IValue("hello"));  // string
  args.push_back(c10::IValue());         // None

  std::vector<int64_t> positions = {0, 1, 2, 3, 4};

  CPUFallbackContext ctx(args, positions, {}, {});

  ASSERT_EQ(ctx.static_args.size(), 5);
  EXPECT_TRUE(ctx.static_args[0].isInt());
  EXPECT_TRUE(ctx.static_args[1].isDouble());
  EXPECT_TRUE(ctx.static_args[2].isBool());
  EXPECT_TRUE(ctx.static_args[3].isString());
  EXPECT_TRUE(ctx.static_args[4].isNone());
}

}  // namespace at::neuron
