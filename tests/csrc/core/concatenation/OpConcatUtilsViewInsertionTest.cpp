/**
 * @file OpConcatUtilsViewInsertionTest.cpp
 * @brief Tests for view/reshape insertion during in-order module merging
 *
 * When PyTorch view/reshape/unsqueeze operations are metadata-only (not dispatched
 * to Neuron), the concatenation system sees the same data pointer but different
 * MLIR tensor shapes. The in-order merge path handles this by validating element
 * count + type compatibility (instead of exact shape) and inserting stablehlo.reshape
 * ops to bridge shape differences.
 */

#include <gtest/gtest.h>

#include "torch_neuronx/csrc/core/concatenation/OpConcatUtils.h"

namespace torch_neuronx {
namespace test {

class ViewInsertionTest : public ::testing::Test {};

/**
 * @brief Test that in-order merge succeeds when shapes differ but element count matches.
 *
 * Reproduces the _fused_rms_norm + contiguous failure:
 *   Module1 output: tensor<1x2048x8x128xbf16>  (4D)
 *   Module2 input:  tensor<1x2048x8x1x128xbf16> (5D, unsqueeze)
 *
 * Same element count (2048*8*128 = 2097152), same element type (bf16).
 * The in-order merge should succeed by inserting a stablehlo.reshape.
 */
TEST_F(ViewInsertionTest, InOrderMergeWithReshapeCompatibleDependency) {
  std::string mlir1 = R"(
module @mod1 {
  func.func public @main(%arg0: tensor<1x2048x8x128xbf16>) -> tensor<1x2048x8x128xbf16> {
    return %arg0 : tensor<1x2048x8x128xbf16>
  }
}
)";

  std::string mlir2 = R"(
module @mod2 {
  func.func public @main(%arg0: tensor<1x2048x8x1x128xbf16>) -> tensor<1x2048x8x1x128xbf16> {
    return %arg0 : tensor<1x2048x8x1x128xbf16>
  }
}
)";

  // Module1 output feeds module2 input (same data_ptr, different shape via unsqueeze)
  at::Tensor input = at::randn({1, 2048, 8, 128}, at::kBFloat16);
  at::Tensor dep_output = at::randn({1, 2048, 8, 128}, at::kBFloat16);
  at::Tensor dep_view = dep_output.unsqueeze(3);  // [1, 2048, 8, 1, 128] same data_ptr
  at::Tensor final_output = at::randn({1, 2048, 8, 1, 128}, at::kBFloat16);

  std::vector<at::Tensor> mod1_inputs = {input};
  std::vector<at::Tensor> mod1_outputs = {dep_output};
  std::vector<at::Tensor> mod2_inputs = {dep_view};  // same data_ptr as dep_output
  std::vector<at::Tensor> mod2_outputs = {final_output};

  auto result = torch_neuronx::mergeStableHLOModulesWithTensorsInOrder(
      mlir1, mlir2, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs,
      /*verify_output=*/true);

  EXPECT_TRUE(result.success) << "In-order merge should succeed with reshape-compatible shapes";
  EXPECT_FALSE(result.merged_mlir_string.empty());

  // Verify the merged module contains a reshape op
  EXPECT_NE(result.merged_mlir_string.find("stablehlo.reshape"), std::string::npos)
      << "Merged module should contain a stablehlo.reshape op";
}

/**
 * @brief Test that in-order merge fails when element COUNTS differ.
 */
TEST_F(ViewInsertionTest, InOrderMergeRejectsIncompatibleElementCount) {
  // Module1: output has 1024 elements
  std::string mlir1 = R"(
module @mod1 {
  func.func public @main(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    return %arg0 : tensor<32x32xf32>
  }
}
)";

  // Module2: input has 2048 elements (different count!)
  std::string mlir2 = R"(
module @mod2 {
  func.func public @main(%arg0: tensor<64x32xf32>) -> tensor<64x32xf32> {
    return %arg0 : tensor<64x32xf32>
  }
}
)";

  // Create tensors where mod2 input has same data_ptr as mod1 output
  // (In reality, different element counts can't share data_ptr via view,
  //  but we simulate via from_blob to trigger the validation path)
  at::Tensor input = at::randn({32, 32}, at::kFloat);
  at::Tensor dep_output = at::randn({64, 32}, at::kFloat);  // 2048 elements storage
  at::Tensor dep_small_view =
      at::from_blob(dep_output.data_ptr(), {32, 32}, at::TensorOptions().dtype(at::kFloat));
  at::Tensor final_output = at::randn({64, 32}, at::kFloat);

  // mod1 outputs 32x32 (1024 elements), mod2 expects 64x32 (2048 elements) at same ptr
  std::vector<at::Tensor> mod1_inputs = {input};
  std::vector<at::Tensor> mod1_outputs = {dep_small_view};  // 32x32 view of dep_output
  std::vector<at::Tensor> mod2_inputs = {dep_output};       // 64x32, same data_ptr
  std::vector<at::Tensor> mod2_outputs = {final_output};

  auto result = torch_neuronx::mergeStableHLOModulesWithTensorsInOrder(
      mlir1, mlir2, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs,
      /*verify_output=*/true);

  EXPECT_FALSE(result.success) << "Merge should fail when element counts differ";
}

/**
 * @brief Test that in-order merge fails when element TYPES differ.
 */
TEST_F(ViewInsertionTest, InOrderMergeRejectsIncompatibleElementType) {
  // Module1: output is f32
  std::string mlir1 = R"(
module @mod1 {
  func.func public @main(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    return %arg0 : tensor<32x32xf32>
  }
}
)";

  // Module2: input is bf16 (different element type!)
  std::string mlir2 = R"(
module @mod2 {
  func.func public @main(%arg0: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
    return %arg0 : tensor<32x32xbf16>
  }
}
)";

  at::Tensor input = at::randn({32, 32}, at::kFloat);
  at::Tensor dep_output = at::randn({32, 32}, at::kFloat);
  // Create bf16 view of same storage (different element type, same ptr)
  at::Tensor dep_bf16_view =
      at::from_blob(dep_output.data_ptr(), {32, 32}, at::TensorOptions().dtype(at::kBFloat16));
  at::Tensor final_output = at::randn({32, 32}, at::kBFloat16);

  std::vector<at::Tensor> mod1_inputs = {input};
  std::vector<at::Tensor> mod1_outputs = {dep_output};
  std::vector<at::Tensor> mod2_inputs = {dep_bf16_view};  // same data_ptr, different dtype
  std::vector<at::Tensor> mod2_outputs = {final_output};

  auto result = torch_neuronx::mergeStableHLOModulesWithTensorsInOrder(
      mlir1, mlir2, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs,
      /*verify_output=*/true);

  EXPECT_FALSE(result.success) << "Merge should fail when element types differ";
}

/**
 * @brief Test that exact shape match still works (no reshape needed).
 */
TEST_F(ViewInsertionTest, InOrderMergeExactShapeMatchNoReshape) {
  std::string mlir1 = R"(
module @mod1 {
  func.func public @main(%arg0: tensor<4x8xbf16>) -> tensor<4x8xbf16> {
    return %arg0 : tensor<4x8xbf16>
  }
}
)";

  std::string mlir2 = R"(
module @mod2 {
  func.func public @main(%arg0: tensor<4x8xbf16>) -> tensor<4x8xbf16> {
    return %arg0 : tensor<4x8xbf16>
  }
}
)";

  at::Tensor input = at::randn({4, 8}, at::kBFloat16);
  at::Tensor dep_output = at::randn({4, 8}, at::kBFloat16);
  at::Tensor final_output = at::randn({4, 8}, at::kBFloat16);

  // Same data_ptr, same shape — dependency with no view
  std::vector<at::Tensor> mod1_inputs = {input};
  std::vector<at::Tensor> mod1_outputs = {dep_output};
  std::vector<at::Tensor> mod2_inputs = {dep_output};  // exact same tensor
  std::vector<at::Tensor> mod2_outputs = {final_output};

  auto result = torch_neuronx::mergeStableHLOModulesWithTensorsInOrder(
      mlir1, mlir2, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs,
      /*verify_output=*/true);

  EXPECT_TRUE(result.success) << "Merge should succeed with exact shape match";
  EXPECT_FALSE(result.merged_mlir_string.empty());

  // Verify NO reshape op when shapes match exactly
  EXPECT_EQ(result.merged_mlir_string.find("stablehlo.reshape"), std::string::npos)
      << "No reshape should be inserted when shapes match exactly";
}

/**
 * @brief Test multi-output module where one output has a view dependency.
 *
 * Simulates _fused_rms_norm (2 outputs: normalized + rstd) where the
 * downstream contiguous op has a view of one output.
 */
TEST_F(ViewInsertionTest, InOrderMergeMultiOutputWithViewDependency) {
  // Module1: 2 outputs (like _fused_rms_norm: normalized + rstd)
  std::string mlir1 = R"(
module @mod1 {
  func.func public @main(%arg0: tensor<1x2048x4096xbf16>, %arg1: tensor<4096xbf16>)
      -> (tensor<1x2048x4096xbf16>, tensor<1x2048x1xf32>) {
    %0 = stablehlo.constant dense<1.0> : tensor<1x2048x4096xbf16>
    %1 = stablehlo.constant dense<0.5> : tensor<1x2048x1xf32>
    return %0, %1 : tensor<1x2048x4096xbf16>, tensor<1x2048x1xf32>
  }
}
)";

  // Module2: input is a view of module1's first output (same data, reshaped)
  std::string mlir2 = R"(
module @mod2 {
  func.func public @main(%arg0: tensor<2048x4096xbf16>) -> tensor<2048x4096xbf16> {
    return %arg0 : tensor<2048x4096xbf16>
  }
}
)";

  at::Tensor input0 = at::randn({1, 2048, 4096}, at::kBFloat16);
  at::Tensor input1 = at::randn({4096}, at::kBFloat16);
  at::Tensor output0 = at::randn({1, 2048, 4096}, at::kBFloat16);  // normalized
  at::Tensor output1 = at::randn({1, 2048, 1}, at::kFloat);        // rstd
  at::Tensor output0_view = output0.view({2048, 4096});            // 3D -> 2D, same data_ptr
  at::Tensor final_output = at::randn({2048, 4096}, at::kBFloat16);

  std::vector<at::Tensor> mod1_inputs = {input0, input1};
  std::vector<at::Tensor> mod1_outputs = {output0, output1};
  std::vector<at::Tensor> mod2_inputs = {output0_view};  // depends on output0, reshaped
  std::vector<at::Tensor> mod2_outputs = {final_output};

  auto result = torch_neuronx::mergeStableHLOModulesWithTensorsInOrder(
      mlir1, mlir2, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs,
      /*verify_output=*/true);

  EXPECT_TRUE(result.success)
      << "In-order merge should succeed for multi-output module with view dependency";
  EXPECT_FALSE(result.merged_mlir_string.empty());
}

/**
 * @brief Test that strict (non-in-order) merge still rejects shape mismatches.
 *
 * The strict path (mergeStableHLOModulesWithTensors) should NOT allow reshape-compatible
 * shapes — only the in-order path has this relaxation.
 */
TEST_F(ViewInsertionTest, StrictMergeRejectsShapeMismatch) {
  std::string mlir1 = R"(
module @mod1 {
  func.func public @main(%arg0: tensor<1x2048x8x128xbf16>) -> tensor<1x2048x8x128xbf16> {
    return %arg0 : tensor<1x2048x8x128xbf16>
  }
}
)";

  std::string mlir2 = R"(
module @mod2 {
  func.func public @main(%arg0: tensor<1x2048x8x1x128xbf16>) -> tensor<1x2048x8x1x128xbf16> {
    return %arg0 : tensor<1x2048x8x1x128xbf16>
  }
}
)";

  at::Tensor input = at::randn({1, 2048, 8, 128}, at::kBFloat16);
  at::Tensor dep_output = at::randn({1, 2048, 8, 128}, at::kBFloat16);
  at::Tensor dep_view = dep_output.unsqueeze(3);  // [1, 2048, 8, 1, 128]
  at::Tensor final_output = at::randn({1, 2048, 8, 1, 128}, at::kBFloat16);

  std::vector<at::Tensor> mod1_inputs = {input};
  std::vector<at::Tensor> mod1_outputs = {dep_output};
  std::vector<at::Tensor> mod2_inputs = {dep_view};
  std::vector<at::Tensor> mod2_outputs = {final_output};

  // Use strict merge (NOT in-order) — should reject shape mismatch
  auto result = torch_neuronx::mergeStableHLOModulesWithTensors(
      mlir1, mlir2, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs,
      /*verify_output=*/true);

  EXPECT_FALSE(result.success) << "Strict merge should reject shape mismatches";
}

}  // namespace test
}  // namespace torch_neuronx
