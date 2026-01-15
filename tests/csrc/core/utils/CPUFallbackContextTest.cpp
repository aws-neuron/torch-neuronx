#include <gtest/gtest.h>
#include <torch/torch.h>

#include "torch_neuronx/csrc/core/utils/CPUFallbackContext.h"

namespace at::neuron {

class CPUFallbackContextTest : public ::testing::Test {};

// Test default constructor creates empty context
TEST_F(CPUFallbackContextTest, DefaultConstructorCreatesEmptyContext) {
  CPUFallbackContext ctx;

  EXPECT_TRUE(ctx.input_metadata.empty());
  EXPECT_TRUE(ctx.tensor_metadata.empty());
  EXPECT_TRUE(ctx.tensor_data_ptrs.empty());
  EXPECT_TRUE(ctx.tensor_kwarg_names.empty());
  EXPECT_TRUE(ctx.IsEmpty());
}

// Test IsEmpty returns true only when all fields are empty
TEST_F(CPUFallbackContextTest, IsEmptyReturnsTrueOnlyWhenAllFieldsEmpty) {
  // Empty context
  CPUFallbackContext empty_ctx;
  EXPECT_TRUE(empty_ctx.IsEmpty());

  // Context with input_metadata
  CPUFallbackContext ctx_with_inputs;
  ctx_with_inputs.input_metadata.push_back(ScalarValue(c10::IValue(42)));
  EXPECT_FALSE(ctx_with_inputs.IsEmpty());

  // Context with only kwargs
  CPUFallbackContext ctx_with_kwargs;
  ctx_with_kwargs.original_kwargs.insert("alpha", c10::IValue(0.5));
  EXPECT_FALSE(ctx_with_kwargs.IsEmpty());
}

// Test InputMetadata variant types
TEST_F(CPUFallbackContextTest, InputMetadataVariantTypes) {
  // TensorMarker
  InputMetadata tensor_meta = TensorMarker{};
  EXPECT_TRUE(IsTensor(tensor_meta));
  EXPECT_FALSE(IsTensorList(tensor_meta));
  EXPECT_FALSE(IsScalar(tensor_meta));
  EXPECT_EQ(GetTensorCount(tensor_meta), 1);

  // TensorListMarker
  InputMetadata tensor_list_meta = TensorListMarker(3);
  EXPECT_FALSE(IsTensor(tensor_list_meta));
  EXPECT_TRUE(IsTensorList(tensor_list_meta));
  EXPECT_FALSE(IsScalar(tensor_list_meta));
  EXPECT_EQ(GetTensorCount(tensor_list_meta), 3);
  EXPECT_EQ(GetListSize(tensor_list_meta), 3);

  // ScalarValue with int
  InputMetadata int_meta = ScalarValue(c10::IValue(42));
  EXPECT_FALSE(IsTensor(int_meta));
  EXPECT_FALSE(IsTensorList(int_meta));
  EXPECT_TRUE(IsScalar(int_meta));
  EXPECT_EQ(GetTensorCount(int_meta), 0);
  EXPECT_EQ(GetScalarValue(int_meta).toInt(), 42);

  // ScalarValue with None
  InputMetadata none_meta = ScalarValue(c10::IValue());
  EXPECT_TRUE(IsScalar(none_meta));
  EXPECT_TRUE(GetScalarValue(none_meta).isNone());
}

// Test scalar value types
TEST_F(CPUFallbackContextTest, ScalarValueTypes) {
  // Int
  InputMetadata int_meta = ScalarValue(c10::IValue(42));
  EXPECT_TRUE(GetScalarValue(int_meta).isInt());
  EXPECT_EQ(GetScalarValue(int_meta).toInt(), 42);

  // Double
  InputMetadata double_meta = ScalarValue(c10::IValue(3.14));
  EXPECT_TRUE(GetScalarValue(double_meta).isDouble());
  EXPECT_DOUBLE_EQ(GetScalarValue(double_meta).toDouble(), 3.14);

  // Bool
  InputMetadata bool_meta = ScalarValue(c10::IValue(true));
  EXPECT_TRUE(GetScalarValue(bool_meta).isBool());
  EXPECT_EQ(GetScalarValue(bool_meta).toBool(), true);

  // String
  InputMetadata string_meta = ScalarValue(c10::IValue("hello"));
  EXPECT_TRUE(GetScalarValue(string_meta).isString());
  EXPECT_EQ(GetScalarValue(string_meta).toStringRef(), "hello");

  // IntList
  c10::List<int64_t> int_list;
  int_list.push_back(1);
  int_list.push_back(2);
  InputMetadata int_list_meta = ScalarValue(c10::IValue(int_list));
  EXPECT_TRUE(GetScalarValue(int_list_meta).isIntList());
}

// Test GetPositionalTensorCount
TEST_F(CPUFallbackContextTest, GetPositionalTensorCount) {
  CPUFallbackContext ctx;

  // Empty context
  EXPECT_EQ(ctx.GetPositionalTensorCount(), 0);

  // Add a single tensor
  ctx.input_metadata.push_back(TensorMarker{});
  EXPECT_EQ(ctx.GetPositionalTensorCount(), 1);

  // Add a scalar (shouldn't affect count)
  ctx.input_metadata.push_back(ScalarValue(c10::IValue(42)));
  EXPECT_EQ(ctx.GetPositionalTensorCount(), 1);

  // Add a tensor list with 3 tensors
  ctx.input_metadata.push_back(TensorListMarker(3));
  EXPECT_EQ(ctx.GetPositionalTensorCount(), 4);  // 1 + 3

  // Add another tensor
  ctx.input_metadata.push_back(TensorMarker{});
  EXPECT_EQ(ctx.GetPositionalTensorCount(), 5);  // 1 + 3 + 1
}

// Test tensor kwarg tracking
TEST_F(CPUFallbackContextTest, TensorKwargTracking) {
  CPUFallbackContext ctx;

  // Add positional tensors
  ctx.input_metadata.push_back(TensorMarker{});
  ctx.input_metadata.push_back(TensorListMarker(2));

  // Add tensor kwarg names
  ctx.tensor_kwarg_names.push_back("out");
  ctx.tensor_kwarg_names.push_back("weight");

  // Test IsKwargTensor
  EXPECT_TRUE(ctx.IsKwargTensor("out"));
  EXPECT_TRUE(ctx.IsKwargTensor("weight"));
  EXPECT_FALSE(ctx.IsKwargTensor("alpha"));
  EXPECT_FALSE(ctx.IsKwargTensor("nonexistent"));

  // Test GetKwargTensorIndex
  // Positional count = 1 + 2 = 3
  EXPECT_EQ(ctx.GetKwargTensorIndex("out"), 3);     // 3 + 0
  EXPECT_EQ(ctx.GetKwargTensorIndex("weight"), 4);  // 3 + 1
}

// Test GetTotalTensorCount
TEST_F(CPUFallbackContextTest, GetTotalTensorCount) {
  CPUFallbackContext ctx;

  // Add tensor metadata (simulating what ConvertToCPUFallbackContext does)
  ctx.tensor_metadata.push_back(TensorContext());
  ctx.tensor_metadata.push_back(TensorContext());
  ctx.tensor_metadata.push_back(TensorContext());

  EXPECT_EQ(ctx.GetTotalTensorCount(), 3);
}

// Test move semantics work correctly
TEST_F(CPUFallbackContextTest, MoveConstructorWorks) {
  CPUFallbackContext original;
  original.input_metadata.push_back(ScalarValue(c10::IValue(42)));
  original.input_metadata.push_back(TensorMarker{});
  original.tensor_kwarg_names.push_back("out");

  CPUFallbackContext moved(std::move(original));

  ASSERT_EQ(moved.input_metadata.size(), 2);
  EXPECT_TRUE(IsScalar(moved.input_metadata[0]));
  EXPECT_EQ(GetScalarValue(moved.input_metadata[0]).toInt(), 42);
  EXPECT_TRUE(IsTensor(moved.input_metadata[1]));
  EXPECT_EQ(moved.tensor_kwarg_names.size(), 1);
}

}  // namespace at::neuron
