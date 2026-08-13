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
//  AllocateExecutionOutputs — target_strides (SDPA pattern)
// ═══════════════════════════════════════════════════════════════

TEST(AllocateExecutionOutputsTest, TargetStrides_TransposedLayout) {
  // SDPA pattern: logical shape [2, 4, 8, 16] with strides from transpose(1,2)
  // target_strides = strides of [2, 4, 8, 16] viewed as [2, 8, 4, 16].transpose(1,2)
  // i.e. physical is [2, 8, 4, 16], logical is [2, 4, 8, 16]
  DispatchCacheEntry entry;
  // target strides for [2, 4, 8, 16] with physical layout [2, 8, 4, 16]:
  //   dim 0: stride = 8*4*16 = 512
  //   dim 1: stride = 16      (was dim 2 in physical)
  //   dim 2: stride = 4*16 = 64  (was dim 1 in physical)
  //   dim 3: stride = 1
  entry.output_specs = {{{2, 4, 8, 16}, c10::ScalarType::Float, {{512, 16, 64, 1}}}};

  std::vector<c10::IValue> args = {torch::zeros({2, 4, 8, 16}), torch::ones({2, 4, 8, 16}), 1};
  std::vector<torch::Tensor> filtered = {torch::zeros({2, 4, 8, 16})};

  auto outputs = AllocateExecutionOutputs(entry, args, filtered, kCPU, AddSchema());
  ASSERT_EQ(outputs.size(), 1u);

  // Logical shape must match spec
  EXPECT_EQ(outputs[0].sizes(), (std::vector<int64_t>{2, 4, 8, 16}));

  // Strides must match target_strides
  EXPECT_EQ(outputs[0].strides(), (std::vector<int64_t>{512, 16, 64, 1}));

  // Output should NOT be contiguous (it's a transposed view)
  EXPECT_FALSE(outputs[0].is_contiguous());

  // Total elements must be correct
  EXPECT_EQ(outputs[0].numel(), 2 * 4 * 8 * 16);

  // Underlying storage must be large enough
  EXPECT_GE(static_cast<int64_t>(outputs[0].storage().nbytes() / outputs[0].element_size()),
            outputs[0].numel());
}

TEST(AllocateExecutionOutputsTest, TargetStrides_2DTranspose) {
  // Simple 2D transpose: logical [3, 4], physical [4, 3]
  // target_strides = {1, 3} (column-major)
  DispatchCacheEntry entry;
  entry.output_specs = {{{3, 4}, c10::ScalarType::Float, {{1, 3}}}};

  std::vector<c10::IValue> args = {torch::zeros({3, 4}), torch::ones({3, 4}), 1};
  std::vector<torch::Tensor> filtered = {torch::zeros({3, 4})};

  auto outputs = AllocateExecutionOutputs(entry, args, filtered, kCPU, AddSchema());
  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_EQ(outputs[0].sizes(), (std::vector<int64_t>{3, 4}));
  EXPECT_EQ(outputs[0].strides(), (std::vector<int64_t>{1, 3}));
  EXPECT_FALSE(outputs[0].is_contiguous());
}

TEST(AllocateExecutionOutputsTest, TargetStrides_NulloptUsesContiguous) {
  // When target_strides is nullopt, should allocate contiguous as usual
  DispatchCacheEntry entry;
  entry.output_specs = {{{2, 3}, c10::ScalarType::Float, std::nullopt}};

  std::vector<c10::IValue> args = {torch::zeros({2, 3}), torch::ones({2, 3}), 1};
  std::vector<torch::Tensor> filtered = {torch::zeros({2, 3})};

  auto outputs = AllocateExecutionOutputs(entry, args, filtered, kCPU, AddSchema());
  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_TRUE(outputs[0].is_contiguous());
  EXPECT_EQ(outputs[0].strides(), (std::vector<int64_t>{3, 1}));
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
  // Not an identity op: a NEFF writes output 0 while output 1 is a passthrough.
  // A passthrough slot is still one of the op's returns, so it must hold a copy
  // rather than the input itself. An op shaped like `return input, mask` would
  // otherwise hand back its own argument.
  DispatchCacheEntry entry;
  entry.original_output_specs = {
      {{2}, c10::ScalarType::Float},
      {{3}, c10::ScalarType::Float},
  };
  entry.output_specs = {{{2}, c10::ScalarType::Float}};
  entry.kept_output_indices = {0};        // exec[0] → orig[0]
  entry.passthrough_mappings = {{1, 0}};  // orig[1] = input[0]
  ASSERT_FALSE(IsPureInputPassthrough(entry));

  auto exec_out = torch::ones({2});
  auto input_tensor = torch::zeros({3});
  std::vector<c10::IValue> args = {input_tensor};

  auto result = ReconstructOutputs({exec_out}, entry, args, kCPU);
  ASSERT_EQ(result.size(), 2u);
  // Output 0 is still the NEFF's own buffer, handed through untouched.
  EXPECT_EQ(result[0].data_ptr(), exec_out.data_ptr());
  EXPECT_TRUE(torch::equal(result[1], input_tensor));
  EXPECT_FALSE(result[1].is_alias_of(input_tensor));
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
//  IsPureInputPassthrough
// ═══════════════════════════════════════════════════════════════

TEST(IsPureInputPassthroughTest, FullCoverageSingleOutput) {
  // The only output is a bare placeholder, so nothing survived
  // capture and no NEFF exists to write an output buffer.
  DispatchCacheEntry entry;
  entry.original_output_specs = {{{2}, c10::ScalarType::Float}};
  entry.passthrough_mappings = {{0, 0}};
  EXPECT_TRUE(IsPureInputPassthrough(entry));
}

TEST(IsPureInputPassthroughTest, FullCoverageMultipleOutputs) {
  DispatchCacheEntry entry;
  entry.original_output_specs = {
      {{2}, c10::ScalarType::Float},
      {{3}, c10::ScalarType::Float},
  };
  entry.passthrough_mappings = {{0, 0}, {1, 1}};
  EXPECT_TRUE(IsPureInputPassthrough(entry));
}

TEST(IsPureInputPassthroughTest, PartialCoverageRejected) {
  // Output 0 is a bare placeholder; output 1 is a None (masked backward grad)
  // that carries no mapping. Treating this as a pure passthrough would return
  // an input clone for a slot that still needs real compute, so it must be
  // rejected and fall through to normal execution.
  DispatchCacheEntry entry;
  entry.original_output_specs = {
      {{2}, c10::ScalarType::Float},
      {{}, c10::ScalarType::Undefined},
  };
  entry.passthrough_mappings = {{0, 0}};
  EXPECT_FALSE(IsPureInputPassthrough(entry));
}

TEST(IsPureInputPassthroughTest, NoMappingsRejected) {
  DispatchCacheEntry entry;
  entry.original_output_specs = {{{2}, c10::ScalarType::Float}};
  EXPECT_FALSE(IsPureInputPassthrough(entry));
}

TEST(IsPureInputPassthroughTest, SurvivingOutputSpecsRejected) {
  // A NEFF exists and writes output_specs[0], so this is a normal op that
  // happens to also pass an input through, not a pure passthrough.
  DispatchCacheEntry entry;
  entry.original_output_specs = {
      {{2}, c10::ScalarType::Float},
      {{3}, c10::ScalarType::Float},
  };
  entry.output_specs = {{{2}, c10::ScalarType::Float}};
  entry.kept_output_indices = {0};
  entry.passthrough_mappings = {{1, 0}};
  EXPECT_FALSE(IsPureInputPassthrough(entry));
}

TEST(IsPureInputPassthroughTest, ReconstructOutputsClonesMappedInput) {
  // End-to-end for the identity case: with no exec outputs, every slot is
  // served from its mapped input rather than an unwritten buffer.
  DispatchCacheEntry entry;
  entry.original_output_specs = {{{3}, c10::ScalarType::Float}};
  entry.passthrough_mappings = {{0, 0}};
  ASSERT_TRUE(IsPureInputPassthrough(entry));

  auto input_tensor = torch::ones({3});
  std::vector<c10::IValue> args = {input_tensor};
  auto result = ReconstructOutputs(/*exec_outputs=*/{}, entry, args, kCPU);

  ASSERT_EQ(result.size(), 1u);
  ASSERT_TRUE(result[0].defined());
  EXPECT_TRUE(torch::equal(result[0], input_tensor));
  // A copy, not the input itself: PyTorch rejects an op that returns one of its
  // own inputs, comparing storages rather than tensor identity.
  EXPECT_FALSE(result[0].is_alias_of(input_tensor));
}

TEST(IsPureInputPassthroughTest, ReconstructOutputsClonesEachMappedInput) {
  // Multi-output identity: no slot may alias its input, and the two outputs
  // must not alias each other either (the same check covers both).
  DispatchCacheEntry entry;
  entry.original_output_specs = {
      {{2}, c10::ScalarType::Float},
      {{2}, c10::ScalarType::Float},
  };
  entry.passthrough_mappings = {{0, 0}, {1, 1}};
  ASSERT_TRUE(IsPureInputPassthrough(entry));

  auto in0 = torch::ones({2});
  auto in1 = torch::zeros({2});
  std::vector<c10::IValue> args = {in0, in1};
  auto result = ReconstructOutputs(/*exec_outputs=*/{}, entry, args, kCPU);

  ASSERT_EQ(result.size(), 2u);
  EXPECT_TRUE(torch::equal(result[0], in0));
  EXPECT_TRUE(torch::equal(result[1], in1));
  EXPECT_FALSE(result[0].is_alias_of(in0));
  EXPECT_FALSE(result[1].is_alias_of(in1));
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
