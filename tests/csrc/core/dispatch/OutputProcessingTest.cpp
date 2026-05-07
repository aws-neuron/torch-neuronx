/**
 * Unit tests for OutputProcessing — allocation, reconstruction, and stack push.
 *
 * Tests use CPU tensors and device. No Python or Neuron hardware required.
 */

#include <gtest/gtest.h>

#include "torch_neuronx/csrc/core/dispatch/OutputProcessing.h"

namespace torch_neuronx {
namespace dispatch {
namespace {

static const torch::Device kCPU(torch::kCPU);

// ═══════════════════════════════════════════════════════════════
//  ExtractTensorInputs
// ═══════════════════════════════════════════════════════════════

TEST(ExtractTensorInputsTest, PlainTensors) {
  std::vector<c10::IValue> args = {torch::zeros({2}), torch::ones({3}), c10::IValue(42)};
  auto result = ExtractTensorInputs(args);
  EXPECT_EQ(result.size(), 2u);
}

TEST(ExtractTensorInputsTest, TensorList) {
  c10::List<at::Tensor> tl({torch::zeros({2}), torch::ones({3})});
  std::vector<c10::IValue> args = {c10::IValue(tl), c10::IValue(1)};
  auto result = ExtractTensorInputs(args);
  EXPECT_EQ(result.size(), 2u);
}

TEST(ExtractTensorInputsTest, UndefinedSkipped) {
  std::vector<c10::IValue> args = {torch::zeros({2}), c10::IValue(torch::Tensor())};
  EXPECT_EQ(ExtractTensorInputs(args).size(), 1u);
}

TEST(ExtractTensorInputsTest, EmptyArgs) { EXPECT_TRUE(ExtractTensorInputs({}).empty()); }

TEST(ExtractTensorInputsTest, MixedArgsOnlyTensors) {
  std::vector<c10::IValue> args = {c10::IValue(42), torch::zeros({2}), c10::IValue(true),
                                   torch::ones({3})};
  EXPECT_EQ(ExtractTensorInputs(args).size(), 2u);
}

// ═══════════════════════════════════════════════════════════════
//  AllocateExecutionOutputs
// ═══════════════════════════════════════════════════════════════

static c10::FunctionSchema AddSchema() {
  return c10::Dispatcher::singleton().findSchemaOrThrow("aten::add", "Tensor").schema();
}

TEST(AllocateExecutionOutputsTest, AllocatesFreshTensors) {
  DispatchCacheEntry entry;
  entry.output_specs = {{{2, 3}, c10::ScalarType::Float}};

  std::vector<c10::IValue> args = {torch::zeros({2, 3}), torch::ones({2, 3}), 1};
  std::vector<torch::Tensor> filtered = {torch::zeros({2, 3}), torch::ones({2, 3})};

  auto outputs = AllocateExecutionOutputs(entry, args, filtered, kCPU, AddSchema());
  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_EQ(outputs[0].sizes(), (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(outputs[0].scalar_type(), c10::ScalarType::Float);
  EXPECT_TRUE(outputs[0].device().is_cpu());
}

TEST(AllocateExecutionOutputsTest, MultipleOutputs) {
  DispatchCacheEntry entry;
  entry.output_specs = {
      {{4}, c10::ScalarType::Float},
      {{2, 2}, c10::ScalarType::Int},
  };

  std::vector<c10::IValue> args = {torch::zeros({4}), torch::ones({4}), 1};
  std::vector<torch::Tensor> filtered = {torch::zeros({4})};

  auto outputs = AllocateExecutionOutputs(entry, args, filtered, kCPU, AddSchema());
  ASSERT_EQ(outputs.size(), 2u);
  EXPECT_EQ(outputs[0].sizes(), (std::vector<int64_t>{4}));
  EXPECT_EQ(outputs[1].sizes(), (std::vector<int64_t>{2, 2}));
  EXPECT_EQ(outputs[1].scalar_type(), c10::ScalarType::Int);
}

TEST(AllocateExecutionOutputsTest, InputOutputAlias) {
  DispatchCacheEntry entry;
  entry.output_specs = {{{2, 3}, c10::ScalarType::Float}};
  entry.input_output_aliases = {{0, 0}};
  entry.is_inplace = true;

  auto input = torch::ones({2, 3}, torch::kFloat32);
  std::vector<c10::IValue> args = {input, torch::zeros({2, 3}), 1};
  std::vector<torch::Tensor> filtered = {input, torch::zeros({2, 3})};

  auto outputs = AllocateExecutionOutputs(entry, args, filtered, kCPU, AddSchema());
  ASSERT_EQ(outputs.size(), 1u);
  // Output should alias the input tensor
  EXPECT_EQ(outputs[0].data_ptr(), input.data_ptr());
}

TEST(AllocateExecutionOutputsTest, AliasDtypeMismatchAllocatesFresh) {
  DispatchCacheEntry entry;
  entry.output_specs = {{{2, 3}, c10::ScalarType::Int}};
  entry.input_output_aliases = {{0, 0}};
  entry.is_inplace = true;

  auto input = torch::ones({2, 3}, torch::kFloat32);
  std::vector<c10::IValue> args = {input, torch::zeros({2, 3}), 1};
  std::vector<torch::Tensor> filtered = {input};

  auto outputs = AllocateExecutionOutputs(entry, args, filtered, kCPU, AddSchema());
  ASSERT_EQ(outputs.size(), 1u);
  // Dtype mismatch — should NOT alias, allocates fresh
  EXPECT_NE(outputs[0].data_ptr(), input.data_ptr());
  EXPECT_EQ(outputs[0].scalar_type(), c10::ScalarType::Int);
}

// ═══════════════════════════════════════════════════════════════
//  ReconstructOutputs
// ═══════════════════════════════════════════════════════════════

TEST(ReconstructOutputsTest, FastPathNoReconstruction) {
  DispatchCacheEntry entry;
  entry.original_output_specs = {{{2, 3}, c10::ScalarType::Float}};
  // No kept_output_indices, no passthrough, sizes match → fast path

  auto exec = std::vector<torch::Tensor>{torch::ones({2, 3})};
  auto result = ReconstructOutputs(exec, entry, {}, kCPU);
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].data_ptr(), exec[0].data_ptr());
}

TEST(ReconstructOutputsTest, KeptOutputIndicesReorder) {
  DispatchCacheEntry entry;
  entry.original_output_specs = {
      {{2}, c10::ScalarType::Float},  // output 0
      {{3}, c10::ScalarType::Float},  // output 1
      {{4}, c10::ScalarType::Float},  // output 2
  };
  entry.kept_output_indices = {2, 0};  // exec[0]→orig[2], exec[1]→orig[0]

  auto t0 = torch::ones({4});
  auto t1 = torch::ones({2});
  auto result = ReconstructOutputs({t0, t1}, entry, {}, kCPU);
  ASSERT_EQ(result.size(), 3u);
  EXPECT_EQ(result[2].data_ptr(), t0.data_ptr());
  EXPECT_EQ(result[0].data_ptr(), t1.data_ptr());
  // output 1 should be allocated as empty with correct shape
  EXPECT_EQ(result[1].sizes(), (std::vector<int64_t>{3}));
}

TEST(ReconstructOutputsTest, PassthroughMapping) {
  DispatchCacheEntry entry;
  entry.original_output_specs = {
      {{2}, c10::ScalarType::Float},
      {{3}, c10::ScalarType::Float},
  };
  entry.kept_output_indices = {0};        // exec[0] → orig[0]
  entry.passthrough_mappings = {{1, 0}};  // orig[1] = input[0]

  auto exec_out = torch::ones({2});
  auto input_tensor = torch::zeros({3});
  std::vector<c10::IValue> args = {input_tensor};

  auto result = ReconstructOutputs({exec_out}, entry, args, kCPU);
  ASSERT_EQ(result.size(), 2u);
  EXPECT_EQ(result[0].data_ptr(), exec_out.data_ptr());
  EXPECT_EQ(result[1].data_ptr(), input_tensor.data_ptr());
}

TEST(ReconstructOutputsTest, UndefinedOutputForNoneDtype) {
  DispatchCacheEntry entry;
  entry.original_output_specs = {
      {{2}, c10::ScalarType::Float}, {{}, c10::ScalarType::Undefined},  // None output
  };
  entry.kept_output_indices = {0};

  auto result = ReconstructOutputs({torch::ones({2})}, entry, {}, kCPU);
  ASSERT_EQ(result.size(), 2u);
  EXPECT_TRUE(result[0].defined());
  EXPECT_FALSE(result[1].defined());  // Undefined = None output
}

// ═══════════════════════════════════════════════════════════════
//  UpcastExecutionOutputs
// ═══════════════════════════════════════════════════════════════

TEST(UpcastExecutionOutputsTest, NoInferredDtypesReturnsIdentical) {
  DispatchCacheEntry entry;
  // No inferred_output_dtypes → returns input unchanged
  auto exec = std::vector<torch::Tensor>{torch::ones({2, 3})};
  auto result = UpcastExecutionOutputs(exec, entry);
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].data_ptr(), exec[0].data_ptr());
}

TEST(UpcastExecutionOutputsTest, NoUpcastNeeded) {
  DispatchCacheEntry entry;
  entry.output_specs = {{{2, 3}, c10::ScalarType::Float}};
  entry.original_output_specs = {{{2, 3}, c10::ScalarType::Float}};
  entry.inferred_output_dtypes = {c10::ScalarType::Float};

  auto exec = std::vector<torch::Tensor>{torch::ones({2, 3}, torch::kFloat32)};
  auto result = UpcastExecutionOutputs(exec, entry);
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].data_ptr(), exec[0].data_ptr());
}

TEST(UpcastExecutionOutputsTest, Float32ToFloat64Upcast) {
  DispatchCacheEntry entry;
  entry.output_specs = {{{2, 3}, c10::ScalarType::Float}};
  entry.original_output_specs = {{{2, 3}, c10::ScalarType::Float}};
  entry.inferred_output_dtypes = {c10::ScalarType::Double};

  auto exec = std::vector<torch::Tensor>{torch::ones({2, 3}, torch::kFloat32)};
  auto result = UpcastExecutionOutputs(exec, entry);
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].scalar_type(), c10::ScalarType::Double);
  EXPECT_NEAR(result[0][0][0].item<double>(), 1.0, 1e-6);
}

TEST(UpcastExecutionOutputsTest, Int32ToInt64Upcast) {
  DispatchCacheEntry entry;
  entry.output_specs = {{{4}, c10::ScalarType::Int}};
  entry.original_output_specs = {{{4}, c10::ScalarType::Int}};
  entry.inferred_output_dtypes = {c10::ScalarType::Long};

  auto exec = std::vector<torch::Tensor>{torch::ones({4}, torch::kInt32)};
  auto result = UpcastExecutionOutputs(exec, entry);
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].scalar_type(), c10::ScalarType::Long);
}

TEST(UpcastExecutionOutputsTest, MultipleOutputsMixedUpcast) {
  DispatchCacheEntry entry;
  entry.output_specs = {
      {{4}, c10::ScalarType::Float},
      {{2}, c10::ScalarType::Int},
  };
  entry.original_output_specs = {
      {{4}, c10::ScalarType::Float},
      {{2}, c10::ScalarType::Int},
  };
  // Only upcast float to double, int stays int
  entry.inferred_output_dtypes = {c10::ScalarType::Double, c10::ScalarType::Int};

  auto exec = std::vector<torch::Tensor>{
      torch::ones({4}, torch::kFloat32),
      torch::ones({2}, torch::kInt32),
  };
  auto result = UpcastExecutionOutputs(exec, entry);
  ASSERT_EQ(result.size(), 2u);
  EXPECT_EQ(result[0].scalar_type(), c10::ScalarType::Double);
  EXPECT_EQ(result[1].scalar_type(), c10::ScalarType::Int);  // No int upcast
}

TEST(UpcastExecutionOutputsTest, CopyBackToOutTensor) {
  DispatchCacheEntry entry;
  entry.output_specs = {{{2, 3}, c10::ScalarType::Float}};
  entry.original_output_specs = {{{2, 3}, c10::ScalarType::Float}};
  entry.inferred_output_dtypes = {c10::ScalarType::Double};

  auto orig = torch::zeros({2, 3}, torch::kFloat64);
  auto exec = std::vector<torch::Tensor>{torch::ones({2, 3}, torch::kFloat32)};

  std::vector<std::pair<size_t, torch::Tensor>> out_tensors = {{0, orig}};
  auto result = UpcastExecutionOutputs(exec, entry, out_tensors);
  ASSERT_EQ(result.size(), 1u);
  // Should have copied the upcast result back into orig
  EXPECT_EQ(result[0].scalar_type(), c10::ScalarType::Double);
}

// ═══════════════════════════════════════════════════════════════
//  PushOutputsToStack
// ═══════════════════════════════════════════════════════════════

static c10::FunctionSchema CopySchema() {
  return c10::Dispatcher::singleton().findSchemaOrThrow("aten::copy_", "").schema();
}

// Schema with TensorList return: aten::split.Tensor → Tensor[]
static c10::FunctionSchema SplitSchema() {
  return c10::Dispatcher::singleton().findSchemaOrThrow("aten::split", "Tensor").schema();
}

TEST(PushOutputsToStackTest, SingleTensorReturn) {
  torch::jit::Stack stack;
  auto t = torch::ones({2, 3});
  PushOutputsToStack(&stack, {t}, CopySchema());
  ASSERT_EQ(stack.size(), 1u);
  EXPECT_EQ(stack[0].toTensor().data_ptr(), t.data_ptr());
}

TEST(PushOutputsToStackTest, TensorListReturn) {
  torch::jit::Stack stack;
  auto t1 = torch::zeros({2});
  auto t2 = torch::ones({3});
  PushOutputsToStack(&stack, {t1, t2}, SplitSchema());
  ASSERT_EQ(stack.size(), 1u);
  auto list = stack[0].toTensorList();
  EXPECT_EQ(list.size(), 2u);
}

TEST(PushOutputsToStackTest, EmptyOutputs) {
  torch::jit::Stack stack;
  PushOutputsToStack(&stack, {}, CopySchema());
  EXPECT_TRUE(stack.empty());
}

}  // namespace
}  // namespace dispatch
}  // namespace torch_neuronx
