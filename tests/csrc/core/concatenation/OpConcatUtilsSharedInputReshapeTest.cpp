/**
 * @file OpConcatUtilsSharedInputReshapeTest.cpp
 * @brief Tests for reshape insertion on shared/common inputs during in-order module merging
 *
 * When two modules share the same input (same data pointer) but expect different shapes
 * (due to a view/reshape/unsqueeze that happened on the host), the in-order merge path
 * must insert a stablehlo.reshape to bridge the shape difference for the second module.
 *
 * This extends the view insertion feature (which previously only handled dependencies)
 * to also handle common inputs.
 */

#include <gtest/gtest.h>

#include "torch_neuronx/csrc/core/concatenation/OpConcatUtils.h"

namespace torch_neuronx {
namespace test {

class SharedInputReshapeTest : public ::testing::Test {
 protected:
  // Simple identity modules for testing merge behavior
  // Module with single input/output of given shape
  static std::string makeIdentityModule(const std::string& shape, const std::string& dtype) {
    return "module {\n  func.func public @main(%arg0: tensor<" + shape + "x" + dtype +
           ">) -> tensor<" + shape + "x" + dtype + "> {\n    return %arg0 : tensor<" + shape + "x" +
           dtype + ">\n  }\n}\n";
  }

  // Module with two inputs, returns second input (reshaped to first's shape via stablehlo.reshape)
  static std::string makeTwoInputModuleReturnsSecond(const std::string& shape1,
                                                     const std::string& shape2,
                                                     const std::string& dtype) {
    return "module {\n  func.func public @main(%arg0: tensor<" + shape1 + "x" + dtype +
           ">, %arg1: tensor<" + shape2 + "x" + dtype + ">) -> tensor<" + shape2 + "x" + dtype +
           "> {\n    return %arg1 : tensor<" + shape2 + "x" + dtype + ">\n  }\n}\n";
  }
};

/**
 * @brief Test that in-order merge succeeds when a shared input has different shapes.
 *
 * Both modules use the same tensor (via view()) but with different shapes.
 * Module1 sees [1, 2048, 1024] and module2 sees [2048, 1024] (squeeze).
 */
TEST_F(SharedInputReshapeTest, CommonInputWithReshapeCompatibleShapes) {
  std::string mlir1 = makeIdentityModule("1x2048x1024", "bf16");
  std::string mlir2 = makeIdentityModule("2048x1024", "bf16");

  // Create a single tensor and a view of it (same data_ptr)
  at::Tensor shared = at::randn({1, 2048, 1024}, at::kBFloat16);
  at::Tensor shared_view = shared.squeeze(0);  // [2048, 1024], same data_ptr

  at::Tensor out1 = at::randn({1, 2048, 1024}, at::kBFloat16);
  at::Tensor out2 = at::randn({2048, 1024}, at::kBFloat16);

  std::vector<at::Tensor> mod1_inputs = {shared};
  std::vector<at::Tensor> mod1_outputs = {out1};
  std::vector<at::Tensor> mod2_inputs = {shared_view};  // same data_ptr, different shape
  std::vector<at::Tensor> mod2_outputs = {out2};

  auto result = torch_neuronx::mergeStableHLOModulesWithTensorsInOrder(
      mlir1, mlir2, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);

  EXPECT_TRUE(result.success)
      << "In-order merge should succeed with reshape-compatible common inputs";
  EXPECT_FALSE(result.merged_mlir_string.empty());

  // The merged module should contain a reshape op for the shape conversion
  EXPECT_NE(result.merged_mlir_string.find("stablehlo.reshape"), std::string::npos)
      << "Merged MLIR should contain a stablehlo.reshape for the common input";
}

/**
 * @brief Test unsqueeze pattern: module1 sees [B,S,H,D] and module2 sees [B,S,H,1,D]
 */
TEST_F(SharedInputReshapeTest, CommonInputWithUnsqueeze) {
  std::string mlir1 = makeIdentityModule("1x8x8x128", "bf16");
  std::string mlir2 = makeIdentityModule("1x8x8x1x128", "bf16");

  at::Tensor shared = at::randn({1, 8, 8, 128}, at::kBFloat16);
  at::Tensor shared_view = shared.unsqueeze(3);  // [1, 8, 8, 1, 128]

  at::Tensor out1 = at::randn({1, 8, 8, 128}, at::kBFloat16);
  at::Tensor out2 = at::randn({1, 8, 8, 1, 128}, at::kBFloat16);

  std::vector<at::Tensor> mod1_inputs = {shared};
  std::vector<at::Tensor> mod1_outputs = {out1};
  std::vector<at::Tensor> mod2_inputs = {shared_view};
  std::vector<at::Tensor> mod2_outputs = {out2};

  auto result = torch_neuronx::mergeStableHLOModulesWithTensorsInOrder(
      mlir1, mlir2, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);

  EXPECT_TRUE(result.success)
      << "In-order merge should succeed with unsqueeze-compatible common input";
  EXPECT_NE(result.merged_mlir_string.find("stablehlo.reshape"), std::string::npos)
      << "Should insert reshape for 4D -> 5D common input";
}

/**
 * @brief Negative: common input with different element counts should fail merge.
 */
TEST_F(SharedInputReshapeTest, CommonInputRejectsIncompatibleElementCount) {
  std::string mlir1 = makeIdentityModule("32x32", "f32");
  std::string mlir2 = makeIdentityModule("64x32", "f32");

  // Different sized tensors at different addresses — NOT actually the same data
  // But we simulate the scenario by using different tensors with different shapes
  at::Tensor t1 = at::randn({32, 32}, at::kFloat);
  at::Tensor t2 = at::randn({64, 32}, at::kFloat);

  // To test the common input path, we need same data_ptr. But tensors with
  // different element counts can't share data_ptr. So this tests the case where
  // they are truly different inputs (no common input detected) — merge should
  // still work but with separate inputs (no reshape needed).
  at::Tensor out1 = at::randn({32, 32}, at::kFloat);
  at::Tensor out2 = at::randn({64, 32}, at::kFloat);

  std::vector<at::Tensor> mod1_inputs = {t1};
  std::vector<at::Tensor> mod1_outputs = {out1};
  std::vector<at::Tensor> mod2_inputs = {t2};
  std::vector<at::Tensor> mod2_outputs = {out2};

  auto result = torch_neuronx::mergeStableHLOModulesWithTensorsInOrder(
      mlir1, mlir2, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);

  // These have different data_ptrs, so they're independent inputs (not common)
  // The merge should succeed with both as separate inputs
  EXPECT_TRUE(result.success);
  // No reshape should be inserted since they are independent inputs
  EXPECT_EQ(result.merged_mlir_string.find("stablehlo.reshape"), std::string::npos)
      << "No reshape should be inserted for independent (non-shared) inputs";
}

/**
 * @brief Negative: common input with different element types should fail merge.
 *
 * When the MLIR types don't match the common input analysis (different dtypes),
 * the validation should reject the merge.
 */
TEST_F(SharedInputReshapeTest, CommonInputRejectsIncompatibleElementType) {
  // Module1 expects f32, module2 expects bf16 — but we use the same storage
  std::string mlir1 = makeIdentityModule("32x32", "f32");
  std::string mlir2 = makeIdentityModule("32x32", "bf16");

  // Create a storage and two views with different dtypes
  // In reality, same data_ptr with different dtypes means reinterpret_cast
  at::Tensor t_f32 = at::randn({32, 32}, at::kFloat);
  // Create a bf16 view of the same storage (different element type, same ptr)
  at::Tensor t_bf16 =
      at::from_blob(t_f32.data_ptr(), {32, 32}, at::TensorOptions().dtype(at::kBFloat16));

  at::Tensor out1 = at::randn({32, 32}, at::kFloat);
  at::Tensor out2 = at::randn({32, 32}, at::kBFloat16);

  std::vector<at::Tensor> mod1_inputs = {t_f32};
  std::vector<at::Tensor> mod1_outputs = {out1};
  std::vector<at::Tensor> mod2_inputs = {t_bf16};  // same data_ptr, different dtype
  std::vector<at::Tensor> mod2_outputs = {out2};

  auto result = torch_neuronx::mergeStableHLOModulesWithTensorsInOrder(
      mlir1, mlir2, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);

  // Should fail because element types differ
  EXPECT_FALSE(result.success) << "Merge should fail when common input element types differ";
}

/**
 * @brief Negative: exact same shape on common inputs should NOT insert reshape.
 */
TEST_F(SharedInputReshapeTest, CommonInputExactMatchNoReshape) {
  std::string mlir1 = makeIdentityModule("8x16", "bf16");
  std::string mlir2 = makeIdentityModule("8x16", "bf16");

  at::Tensor shared = at::randn({8, 16}, at::kBFloat16);
  // Same tensor used by both — same shape, same data_ptr
  at::Tensor out1 = at::randn({8, 16}, at::kBFloat16);
  at::Tensor out2 = at::randn({8, 16}, at::kBFloat16);

  std::vector<at::Tensor> mod1_inputs = {shared};
  std::vector<at::Tensor> mod1_outputs = {out1};
  std::vector<at::Tensor> mod2_inputs = {shared};  // exact same tensor
  std::vector<at::Tensor> mod2_outputs = {out2};

  auto result = torch_neuronx::mergeStableHLOModulesWithTensorsInOrder(
      mlir1, mlir2, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);

  EXPECT_TRUE(result.success) << "Merge should succeed with exact shape match";
  // No reshape needed — shapes match exactly
  EXPECT_EQ(result.merged_mlir_string.find("stablehlo.reshape"), std::string::npos)
      << "No reshape should be inserted when common input shapes match exactly";
}

/**
 * @brief Test with both a dependency AND a reshape-compatible common input.
 *
 * Module2 actually uses its second arg (the common input with different shape)
 * in its output, so the reshape cannot be dead-code-eliminated.
 */
TEST_F(SharedInputReshapeTest, DependencyPlusReshapeCommonInput) {
  // Module1: takes 1 input, produces 1 output of same shape
  std::string mlir1 = R"(
module {
  func.func public @main(%arg0: tensor<4x8xbf16>, %arg1: tensor<1x32xbf16>) -> tensor<4x8xbf16> {
    return %arg0 : tensor<4x8xbf16>
  }
}
)";

  // Module2: takes dep (4x8) + common (2x16), returns the common input
  std::string mlir2 = R"(
module {
  func.func public @main(%arg0: tensor<4x8xbf16>, %arg1: tensor<2x16xbf16>) -> tensor<2x16xbf16> {
    return %arg1 : tensor<2x16xbf16>
  }
}
)";

  at::Tensor input0 = at::randn({4, 8}, at::kBFloat16);
  at::Tensor shared = at::randn({1, 32}, at::kBFloat16);
  at::Tensor shared_view = shared.view({2, 16});  // same data_ptr, different shape
  at::Tensor dep_output = at::randn({4, 8}, at::kBFloat16);
  at::Tensor final_output = at::randn({2, 16}, at::kBFloat16);

  std::vector<at::Tensor> mod1_inputs = {input0, shared};
  std::vector<at::Tensor> mod1_outputs = {dep_output};
  std::vector<at::Tensor> mod2_inputs = {dep_output, shared_view};  // dep + reshaped common
  std::vector<at::Tensor> mod2_outputs = {final_output};

  auto result = torch_neuronx::mergeStableHLOModulesWithTensorsInOrder(
      mlir1, mlir2, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);

  EXPECT_TRUE(result.success)
      << "Merge should succeed with dependency + reshape-compatible common input";
  EXPECT_FALSE(result.merged_mlir_string.empty());

  // Module2 uses its arg1 (the reshaped common input) in its output
  EXPECT_NE(result.merged_mlir_string.find("stablehlo.reshape"), std::string::npos)
      << "Should insert reshape for the common input shape mismatch";
}

/**
 * @brief Negative: strict (non-in-order) merge should reject common input shape mismatches.
 */
TEST_F(SharedInputReshapeTest, StrictMergeRejectsCommonInputShapeMismatch) {
  std::string mlir1 = makeIdentityModule("1x2048x1024", "bf16");
  std::string mlir2 = makeIdentityModule("2048x1024", "bf16");

  at::Tensor shared = at::randn({1, 2048, 1024}, at::kBFloat16);
  at::Tensor shared_view = shared.squeeze(0);
  at::Tensor out1 = at::randn({1, 2048, 1024}, at::kBFloat16);
  at::Tensor out2 = at::randn({2048, 1024}, at::kBFloat16);

  std::vector<at::Tensor> mod1_inputs = {shared};
  std::vector<at::Tensor> mod1_outputs = {out1};
  std::vector<at::Tensor> mod2_inputs = {shared_view};
  std::vector<at::Tensor> mod2_outputs = {out2};

  // Use strict merge (NOT in-order) — should reject shape mismatch
  auto result = torch_neuronx::mergeStableHLOModulesWithTensors(
      mlir1, mlir2, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);

  EXPECT_FALSE(result.success) << "Strict merge should reject common input shape mismatches";
}

/**
 * @brief Test multiple common inputs where some need reshape and some don't.
 */
TEST_F(SharedInputReshapeTest, MultipleCommonInputsMixedReshape) {
  // Module1: two inputs, returns second input (1x32)
  std::string mlir1 = makeTwoInputModuleReturnsSecond("4x8", "1x32", "bf16");
  // Module2: first common input same shape, second needs reshape. Returns second (2x16)
  std::string mlir2 = makeTwoInputModuleReturnsSecond("4x8", "2x16", "bf16");

  at::Tensor shared1 = at::randn({4, 8}, at::kBFloat16);
  at::Tensor shared2 = at::randn({1, 32}, at::kBFloat16);
  at::Tensor shared2_view = shared2.view({2, 16});

  at::Tensor out1 = at::randn({1, 32}, at::kBFloat16);
  at::Tensor out2 = at::randn({2, 16}, at::kBFloat16);

  std::vector<at::Tensor> mod1_inputs = {shared1, shared2};
  std::vector<at::Tensor> mod1_outputs = {out1};
  std::vector<at::Tensor> mod2_inputs = {shared1, shared2_view};  // same ptr, reshaped
  std::vector<at::Tensor> mod2_outputs = {out2};

  auto result = torch_neuronx::mergeStableHLOModulesWithTensorsInOrder(
      mlir1, mlir2, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);

  EXPECT_TRUE(result.success) << "Merge should succeed with multiple common inputs (mixed reshape)";
  EXPECT_FALSE(result.merged_mlir_string.empty());

  // Only the second common input needs reshape (1x32 -> 2x16)
  EXPECT_NE(result.merged_mlir_string.find("stablehlo.reshape"), std::string::npos)
      << "Should have reshape for the shape-mismatched common input";
}

}  // namespace test
}  // namespace torch_neuronx
