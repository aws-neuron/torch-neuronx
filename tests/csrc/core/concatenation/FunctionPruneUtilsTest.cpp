#include <gtest/gtest.h>

#include <unordered_set>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch_neuronx/csrc/core/concatenation/FunctionPruneUtils.h"

class FunctionPruneUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_.getOrLoadDialect<mlir::func::FuncDialect>();
    context_.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();
  }

  // Helper methods to create fresh modules for each test
  // This prevents shared mutable state between tests

  // Module with 4 inputs but only 2 are actually used (args 0 and 2, not 1 and 3)
  mlir::OwningOpRef<mlir::ModuleOp> createModule4in4out() {
    std::string mlir_str = R"(
module @test_module {
  func.func public @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) {
    %c1 = stablehlo.constant dense<1.0> : tensor<f32>
    %c2 = stablehlo.constant dense<2.0> : tensor<f32>
    %0 = stablehlo.add %arg0, %c1 : tensor<f32>
    %1 = stablehlo.multiply %arg2, %c2 : tensor<f32>
    %2 = stablehlo.subtract %arg0, %c1 : tensor<f32>
    %3 = stablehlo.divide %arg2, %c2 : tensor<f32>
    return %0, %1, %2, %3 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
)";
    return mlir::parseSourceString<mlir::ModuleOp>(mlir_str, &context_);
  }

  // Module with 4 inputs where only arg0 is used (for PruneArgumentsKeepFirst test)
  mlir::OwningOpRef<mlir::ModuleOp> createModule4in4out_OnlyFirstUsed() {
    std::string mlir_str = R"(
module @test_module {
  func.func public @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) {
    %c1 = stablehlo.constant dense<1.0> : tensor<f32>
    %c2 = stablehlo.constant dense<2.0> : tensor<f32>
    %0 = stablehlo.add %arg0, %c1 : tensor<f32>
    %1 = stablehlo.multiply %arg0, %c2 : tensor<f32>
    %2 = stablehlo.subtract %arg0, %c1 : tensor<f32>
    %3 = stablehlo.divide %arg0, %c2 : tensor<f32>
    return %0, %1, %2, %3 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
)";
    return mlir::parseSourceString<mlir::ModuleOp>(mlir_str, &context_);
  }

  // Module with 4 inputs where only arg3 is used (for PruneArgumentsKeepLast test)
  mlir::OwningOpRef<mlir::ModuleOp> createModule4in4out_OnlyLastUsed() {
    std::string mlir_str = R"(
module @test_module {
  func.func public @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) {
    %c1 = stablehlo.constant dense<1.0> : tensor<f32>
    %c2 = stablehlo.constant dense<2.0> : tensor<f32>
    %0 = stablehlo.add %arg3, %c1 : tensor<f32>
    %1 = stablehlo.multiply %arg3, %c2 : tensor<f32>
    %2 = stablehlo.subtract %arg3, %c1 : tensor<f32>
    %3 = stablehlo.divide %arg3, %c2 : tensor<f32>
    return %0, %1, %2, %3 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
)";
    return mlir::parseSourceString<mlir::ModuleOp>(mlir_str, &context_);
  }

  // Module with 2 inputs where NEITHER is used (for EmptyIndicesToKeep test)
  mlir::OwningOpRef<mlir::ModuleOp> createModule2in2out_NoneUsed() {
    std::string mlir_str = R"(
module @test_module {
  func.func public @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    %c1 = stablehlo.constant dense<1.0> : tensor<f32>
    %c2 = stablehlo.constant dense<2.0> : tensor<f32>
    %0 = stablehlo.add %c1, %c2 : tensor<f32>
    %1 = stablehlo.multiply %c1, %c2 : tensor<f32>
    return %0, %1 : tensor<f32>, tensor<f32>
  }
}
)";
    return mlir::parseSourceString<mlir::ModuleOp>(mlir_str, &context_);
  }

  mlir::OwningOpRef<mlir::ModuleOp> createModule2in2out() {
    std::string mlir_str = R"(
module @test_module {
  func.func public @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    %1 = stablehlo.multiply %arg0, %arg1 : tensor<f32>
    return %0, %1 : tensor<f32>, tensor<f32>
  }
}
)";
    return mlir::parseSourceString<mlir::ModuleOp>(mlir_str, &context_);
  }

  mlir::OwningOpRef<mlir::ModuleOp> createModule1in3out() {
    std::string mlir_str = R"(
module @test_module {
  func.func public @main(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>) {
    %c1 = stablehlo.constant dense<1.0> : tensor<f32>
    %c2 = stablehlo.constant dense<2.0> : tensor<f32>
    %0 = stablehlo.add %arg0, %c1 : tensor<f32>
    %1 = stablehlo.multiply %arg0, %c2 : tensor<f32>
    %2 = stablehlo.subtract %arg0, %c1 : tensor<f32>
    return %0, %1, %2 : tensor<f32>, tensor<f32>, tensor<f32>
  }
}
)";
    return mlir::parseSourceString<mlir::ModuleOp>(mlir_str, &context_);
  }

  // Complex multi-function module with nested calls (real-world example)
  mlir::OwningOpRef<mlir::ModuleOp> createComplexMultiFunctionModule() {
    std::string mlir_str = R"(
module {
  func.func @main(%arg0: tensor<1x2048x16384xbf16>, %arg1: tensor<1x1x1xbf16>, %arg2: tensor<4096x16384xbf16>) -> (tensor<1x2048x16384xbf16>, tensor<4096x16384xbf16>, tensor<4096xbf16>) {
    %0:3 = call @submain1(%arg0, %arg1, %arg2) : (tensor<1x2048x16384xbf16>, tensor<1x1x1xbf16>, tensor<4096x16384xbf16>) -> (tensor<1x2048x16384xbf16>, tensor<1x2048x4096xbf16>, tensor<4096x16384xbf16>)
    %1:3 = call @submain2_2(%0#0, %0#1, %0#2) : (tensor<1x2048x16384xbf16>, tensor<1x2048x4096xbf16>, tensor<4096x16384xbf16>) -> (tensor<1x2048x16384xbf16>, tensor<4096x16384xbf16>, tensor<4096xbf16>)
    return %1#0, %1#1, %1#2 : tensor<1x2048x16384xbf16>, tensor<4096x16384xbf16>, tensor<4096xbf16>
  }
  func.func private @submain1(%arg0: tensor<1x2048x16384xbf16>, %arg1: tensor<1x1x1xbf16>, %arg2: tensor<4096x16384xbf16>) -> (tensor<1x2048x16384xbf16>, tensor<1x2048x4096xbf16>, tensor<4096x16384xbf16>) {
    %0:2 = call @submain1_1(%arg0, %arg1) : (tensor<1x2048x16384xbf16>, tensor<1x1x1xbf16>) -> (tensor<1x2048x16384xbf16>, tensor<1x2048x4096xbf16>)
    %1 = call @submain2_1(%arg2) : (tensor<4096x16384xbf16>) -> tensor<4096x16384xbf16>
    return %0#0, %0#1, %1 : tensor<1x2048x16384xbf16>, tensor<1x2048x4096xbf16>, tensor<4096x16384xbf16>
  }
  func.func private @submain1_1(%arg0: tensor<1x2048x16384xbf16>, %arg1: tensor<1x1x1xbf16>) -> (tensor<1x2048x16384xbf16>, tensor<1x2048x4096xbf16>) {
    %0 = call @submain1_1_1(%arg0) : (tensor<1x2048x16384xbf16>) -> tensor<1x2048x16384xbf16>
    %1 = call @submain2(%arg1) : (tensor<1x1x1xbf16>) -> tensor<1x2048x4096xbf16>
    return %0, %1 : tensor<1x2048x16384xbf16>, tensor<1x2048x4096xbf16>
  }
  func.func private @submain1_1_1(%arg0: tensor<1x2048x16384xbf16>) -> tensor<1x2048x16384xbf16> {
    return %arg0 : tensor<1x2048x16384xbf16>
  }
  func.func private @submain2(%arg0: tensor<1x1x1xbf16>) -> tensor<1x2048x4096xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<1x1x1xbf16>) -> tensor<1x2048x4096xbf16>
    return %0 : tensor<1x2048x4096xbf16>
  }
  func.func private @submain2_1(%arg0: tensor<4096x16384xbf16>) -> tensor<4096x16384xbf16> {
    return %arg0 : tensor<4096x16384xbf16>
  }
  func.func private @submain2_2(%arg0: tensor<1x2048x16384xbf16>, %arg1: tensor<1x2048x4096xbf16>, %arg2: tensor<4096x16384xbf16>) -> (tensor<1x2048x16384xbf16> {jax.result_info = "result[0]"}, tensor<4096x16384xbf16> {jax.result_info = "result[1]"}, tensor<4096xbf16> {jax.result_info = "result[2]"}) {
    %0 = stablehlo.dot_general %arg1, %arg2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x2048x4096xbf16>, tensor<4096x16384xbf16>) -> tensor<1x2048x16384xbf16>
    %1 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [0, 1] x [0, 1], precision = [DEFAULT, DEFAULT] : (tensor<1x2048x4096xbf16>, tensor<1x2048x16384xbf16>) -> tensor<4096x16384xbf16>
    %2 = stablehlo.convert %arg1 : (tensor<1x2048x4096xbf16>) -> tensor<1x2048x4096xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x2048x4096xf32>, tensor<f32>) -> tensor<4096xf32>
    %4 = stablehlo.convert %3 : (tensor<4096xf32>) -> tensor<4096xbf16>
    return %0, %1, %4 : tensor<1x2048x16384xbf16>, tensor<4096x16384xbf16>, tensor<4096xbf16>
  }
}
)";
    return mlir::parseSourceString<mlir::ModuleOp>(mlir_str, &context_);
  }

  mlir::MLIRContext context_;
};

// ============================================================================
// Section 1: Index-Based Argument Pruning Tests
// ============================================================================

TEST_F(FunctionPruneUtilsTest, PruneArgumentsKeepAll) {
  auto module = createModule4in4out();
  ASSERT_TRUE(module);
  auto func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(func);

  // Keep all arguments (no pruning)
  std::unordered_set<size_t> indicesToKeep = {0, 1, 2, 3};
  torch_neuronx::pruneFunctionArgumentsByIndices(func, indicesToKeep);

  EXPECT_EQ(func.getNumArguments(), 4) << "Should still have 4 arguments";
}

TEST_F(FunctionPruneUtilsTest, PruneMultipleArgumentsVerifyCorrectTypes) {
  // Test that pruning multiple arguments with different types keeps the correct ones
  // This test explicitly verifies the argument types to ensure the right args are kept
  std::string mlir_str = R"(
module @test_distinct_arg_shapes {
  func.func public @main(%arg0: tensor<1xf32>, %arg1: tensor<2xf32>, %arg2: tensor<3xf32>, %arg3: tensor<4xf32>) -> tensor<f32> {
    %c0 = stablehlo.constant dense<0.0> : tensor<f32>
    return %c0 : tensor<f32>
  }
}
)";

  auto module = mlir::parseSourceString<mlir::ModuleOp>(mlir_str, &context_);
  ASSERT_TRUE(module);
  auto func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(func);

  // Verify initial state - 4 arguments with distinct shapes
  ASSERT_EQ(func.getNumArguments(), 4);

  // Keep arguments at indices 0 and 3 (tensor<1xf32> and tensor<4xf32>)
  // Prune arguments at indices 1 and 2 (tensor<2xf32> and tensor<3xf32>)
  std::unordered_set<size_t> indicesToKeep = {0, 3};
  torch_neuronx::pruneFunctionArgumentsByIndices(func, indicesToKeep);

  // Verify count
  ASSERT_EQ(func.getNumArguments(), 2) << "Should have 2 arguments after pruning";

  // Verify the actual types to ensure the RIGHT arguments were kept
  auto argType0 = mlir::cast<mlir::RankedTensorType>(func.getArgument(0).getType());
  auto argType1 = mlir::cast<mlir::RankedTensorType>(func.getArgument(1).getType());

  EXPECT_EQ(argType0.getDimSize(0), 1)
      << "First argument should be tensor<1xf32> (original index 0)";
  EXPECT_EQ(argType1.getDimSize(0), 4)
      << "Second argument should be tensor<4xf32> (original index 3)";

  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module))) << "Module should still be valid";
}

TEST_F(FunctionPruneUtilsTest, PruneArgumentsInvalidIndex) {
  auto module = createModule4in4out();
  ASSERT_TRUE(module);
  auto func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(func);

  // Try to keep an out-of-range index
  std::unordered_set<size_t> indicesToKeep = {0, 10};  // 10 is out of range

  EXPECT_THROW(
      { torch_neuronx::pruneFunctionArgumentsByIndices(func, indicesToKeep); }, std::runtime_error);
}

// ============================================================================
// Section 2: Index-Based Result Pruning Tests
// ============================================================================

TEST_F(FunctionPruneUtilsTest, PruneResultsKeepAll) {
  auto module = createModule1in3out();
  ASSERT_TRUE(module);
  auto func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(func);

  // Keep all results (no pruning)
  std::unordered_set<size_t> indicesToKeep = {0, 1, 2};
  torch_neuronx::pruneFunctionResultsByIndices(func, indicesToKeep);

  EXPECT_EQ(func.getNumResults(), 3) << "Should still have 3 results";
}

TEST_F(FunctionPruneUtilsTest, PruneResultsInvalidIndex) {
  auto module = createModule1in3out();
  ASSERT_TRUE(module);
  auto func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(func);

  // Try to keep an out-of-range index
  std::unordered_set<size_t> indicesToKeep = {0, 10};  // 10 is out of range

  EXPECT_THROW(
      { torch_neuronx::pruneFunctionResultsByIndices(func, indicesToKeep); }, std::runtime_error);
}

TEST_F(FunctionPruneUtilsTest, PruneMultipleOutputsVerifyCorrectTypesByIndex) {
  // Test that pruning multiple outputs with different shapes keeps the correct ones
  // This test explicitly verifies the result types to ensure the right outputs are kept
  std::string mlir_str = R"(
module @test_distinct_shapes {
  func.func public @main(%arg0: tensor<f32>) -> (tensor<1xf32>, tensor<2xf32>, tensor<3xf32>, tensor<4xf32>) {
    %c1 = stablehlo.constant dense<1.0> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<4xf32>
    return %0, %1, %2, %3 : tensor<1xf32>, tensor<2xf32>, tensor<3xf32>, tensor<4xf32>
  }
}
)";

  auto module = mlir::parseSourceString<mlir::ModuleOp>(mlir_str, &context_);
  ASSERT_TRUE(module);
  auto func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(func);

  // Verify initial state - 4 outputs with distinct shapes
  ASSERT_EQ(func.getNumResults(), 4);

  // Keep outputs at indices 0 and 3 (tensor<1xf32> and tensor<4xf32>)
  // Prune outputs at indices 1 and 2 (tensor<2xf32> and tensor<3xf32>)
  std::unordered_set<size_t> indicesToKeep = {0, 3};
  torch_neuronx::pruneFunctionResultsByIndices(func, indicesToKeep);

  // Verify count
  ASSERT_EQ(func.getNumResults(), 2) << "Should have 2 results after pruning";

  // Verify the actual types to ensure the RIGHT outputs were kept
  auto resultType0 = mlir::cast<mlir::RankedTensorType>(func.getResultTypes()[0]);
  auto resultType1 = mlir::cast<mlir::RankedTensorType>(func.getResultTypes()[1]);

  EXPECT_EQ(resultType0.getDimSize(0), 1)
      << "First result should be tensor<1xf32> (original index 0)";
  EXPECT_EQ(resultType1.getDimSize(0), 4)
      << "Second result should be tensor<4xf32> (original index 3)";

  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module))) << "Module should still be valid";
}

TEST_F(FunctionPruneUtilsTest, PruneMultipleOutputsWithMultidimensionalShapes) {
  // Test with more complex multidimensional shapes to catch potential bugs
  std::string mlir_str = R"(
module @test_multidim_shapes {
  func.func public @main(%arg0: tensor<f32>) -> (tensor<1x2xf32>, tensor<3x4xf32>, tensor<5x6xf32>, tensor<7x8xf32>, tensor<9x10xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<1x2xf32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<3x4xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<5x6xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<7x8xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<9x10xf32>
    return %0, %1, %2, %3, %4 : tensor<1x2xf32>, tensor<3x4xf32>, tensor<5x6xf32>, tensor<7x8xf32>, tensor<9x10xf32>
  }
}
)";

  auto module = mlir::parseSourceString<mlir::ModuleOp>(mlir_str, &context_);
  ASSERT_TRUE(module);
  auto func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(func);

  ASSERT_EQ(func.getNumResults(), 5);

  // Keep outputs 1, 3, 4 (tensor<3x4xf32>, tensor<7x8xf32>, tensor<9x10xf32>)
  // Prune outputs 0, 2 (tensor<1x2xf32>, tensor<5x6xf32>)
  std::unordered_set<size_t> indicesToKeep = {1, 3, 4};
  torch_neuronx::pruneFunctionResultsByIndices(func, indicesToKeep);

  ASSERT_EQ(func.getNumResults(), 3) << "Should have 3 results after pruning";

  // Verify types by checking dimensions
  auto resultType0 = mlir::cast<mlir::RankedTensorType>(func.getResultTypes()[0]);
  auto resultType1 = mlir::cast<mlir::RankedTensorType>(func.getResultTypes()[1]);
  auto resultType2 = mlir::cast<mlir::RankedTensorType>(func.getResultTypes()[2]);

  // First kept output should be tensor<3x4xf32> (original index 1)
  EXPECT_EQ(resultType0.getDimSize(0), 3);
  EXPECT_EQ(resultType0.getDimSize(1), 4);

  // Second kept output should be tensor<7x8xf32> (original index 3)
  EXPECT_EQ(resultType1.getDimSize(0), 7);
  EXPECT_EQ(resultType1.getDimSize(1), 8);

  // Third kept output should be tensor<9x10xf32> (original index 4)
  EXPECT_EQ(resultType2.getDimSize(0), 9);
  EXPECT_EQ(resultType2.getDimSize(1), 10);

  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module))) << "Module should still be valid";
}

// ============================================================================
// Section 3: Combined Pruning Tests
// ============================================================================

TEST_F(FunctionPruneUtilsTest, PruneBothArgumentsAndResults) {
  auto module = createModule4in4out();
  ASSERT_TRUE(module);
  auto func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(func);

  // Prune arguments: keep 0, 2
  std::unordered_set<size_t> argsToKeep = {0, 2};
  torch_neuronx::pruneFunctionArgumentsByIndices(func, argsToKeep);

  // Prune results: keep 0, 3
  std::unordered_set<size_t> resultsToKeep = {0, 3};
  torch_neuronx::pruneFunctionResultsByIndices(func, resultsToKeep);

  EXPECT_EQ(func.getNumArguments(), 2) << "Should have 2 arguments after pruning";
  EXPECT_EQ(func.getNumResults(), 2) << "Should have 2 results after pruning";
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module))) << "Module should still be valid";
}

TEST_F(FunctionPruneUtilsTest, PruneIntermediateOutputScenario) {
  // Simulates the primary use case: Module A's output → Module B's input
  // We want to remove the intermediate output from the final merged function

  // Create a merged-like module where output 1 is intermediate
  std::string merged_mlir = R"(
module @merged {
  func.func public @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    %1 = stablehlo.multiply %0, %arg1 : tensor<f32>
    %2 = stablehlo.subtract %1, %arg0 : tensor<f32>
    return %0, %1, %2 : tensor<f32>, tensor<f32>, tensor<f32>
  }
}
)";

  auto module = mlir::parseSourceString<mlir::ModuleOp>(merged_mlir, &context_);
  ASSERT_TRUE(module);
  auto func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(func);

  // Remove intermediate output (index 1) - keep 0 and 2
  std::unordered_set<size_t> resultsToKeep = {0, 2};
  torch_neuronx::pruneFunctionResultsByIndices(func, resultsToKeep);

  EXPECT_EQ(func.getNumResults(), 2) << "Should have 2 results (intermediate removed)";
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module))) << "Module should still be valid";
}

// ============================================================================
// Section 4: Error Handling Tests
// ============================================================================

TEST_F(FunctionPruneUtilsTest, EmptyIndicesToKeep) {
  auto module = createModule2in2out_NoneUsed();
  ASSERT_TRUE(module);
  auto func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(func);

  // Empty set - remove all arguments (neither arg is used in this module)
  std::unordered_set<size_t> emptySet;
  torch_neuronx::pruneFunctionArgumentsByIndices(func, emptySet);

  EXPECT_EQ(func.getNumArguments(), 0) << "Should have 0 arguments after removing all";
}

// ============================================================================
// Section 5: Real-World Scenarios
// ============================================================================

TEST_F(FunctionPruneUtilsTest, TransposeMatmulIntermediateRemoval) {
  // Realistic scenario: Transpose → Matmul, remove transpose output from final outputs
  std::string transpose_matmul = R"(
module @transpose_matmul {
  func.func public @main(%arg0: tensor<8192x8192xbf16>, %arg1: tensor<2048x8192xbf16>) -> (tensor<8192x8192xbf16>, tensor<2048x8192xbf16>) {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<8192x8192xbf16>) -> tensor<8192x8192xbf16>
    %1 = stablehlo.dot_general %arg1, %0, contracting_dims = [1] x [0] : (tensor<2048x8192xbf16>, tensor<8192x8192xbf16>) -> tensor<2048x8192xbf16>
    return %0, %1 : tensor<8192x8192xbf16>, tensor<2048x8192xbf16>
  }
}
)";

  auto module = mlir::parseSourceString<mlir::ModuleOp>(transpose_matmul, &context_);
  ASSERT_TRUE(module);
  auto func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(func);

  // Keep only matmul output (index 1), prune transpose output (index 0)
  std::unordered_set<size_t> resultsToKeep = {1};
  torch_neuronx::pruneFunctionResultsByIndices(func, resultsToKeep);

  EXPECT_EQ(func.getNumArguments(), 2) << "Should keep both inputs";
  EXPECT_EQ(func.getNumResults(), 1) << "Should keep only final matmul output";
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module))) << "Module should still be valid";
}

TEST_F(FunctionPruneUtilsTest, ChainedOperationsWithIntermediates) {
  // Three chained operations: A → B → C, remove B from outputs
  std::string chained_ops = R"(
module @chained {
  func.func public @main(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>) {
    %c1 = stablehlo.constant dense<1.0> : tensor<f32>
    %c2 = stablehlo.constant dense<2.0> : tensor<f32>
    %0 = stablehlo.add %arg0, %c1 : tensor<f32>
    %1 = stablehlo.multiply %0, %c2 : tensor<f32>
    %2 = stablehlo.subtract %1, %c1 : tensor<f32>
    return %0, %1, %2 : tensor<f32>, tensor<f32>, tensor<f32>
  }
}
)";

  auto module = mlir::parseSourceString<mlir::ModuleOp>(chained_ops, &context_);
  ASSERT_TRUE(module);
  auto func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(func);

  // Keep first and last outputs, remove middle (intermediate)
  std::unordered_set<size_t> resultsToKeep = {0, 2};
  torch_neuronx::pruneFunctionResultsByIndices(func, resultsToKeep);

  EXPECT_EQ(func.getNumResults(), 2) << "Should have 2 results (intermediate removed)";
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module))) << "Module should still be valid";
}

TEST_F(FunctionPruneUtilsTest, PruneResultsVerifyCorrectIndicesKept) {
  // Test that pruning outputs keeps correct ones by verifying result count and types
  auto module = createModule4in4out();
  ASSERT_TRUE(module);
  auto func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(func);

  ASSERT_EQ(func.getNumResults(), 4) << "Should start with 4 results";

  // Keep outputs at indices 0 and 2
  std::unordered_set<size_t> indicesToKeep = {0, 2};
  torch_neuronx::pruneFunctionResultsByIndices(func, indicesToKeep);

  // Verify count
  EXPECT_EQ(func.getNumResults(), 2) << "Should have 2 results after pruning";
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module))) << "Module should still be valid";
}

TEST_F(FunctionPruneUtilsTest, ComplexMultiFunctionModulePruning) {
  // Real-world scenario: Complex module with multiple nested function calls
  // This module has:
  // - main function that calls submain1 and submain2_2
  // - submain1 calls submain1_1 and submain2_1
  // - submain1_1 calls submain1_1_1 and submain2
  // - submain2_2 performs dot_general, convert, reduce operations
  // Tests pruning on realistic nested function structure with large tensors

  auto module = createComplexMultiFunctionModule();
  ASSERT_TRUE(module) << "Complex multi-function module should parse correctly";

  // Verify module structure
  auto func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(func) << "main function should exist";
  EXPECT_EQ(func.getNumArguments(), 3) << "main should have 3 inputs";
  EXPECT_EQ(func.getNumResults(), 3) << "main should have 3 outputs";

  // Verify nested functions exist
  auto submain1 = module->lookupSymbol<mlir::func::FuncOp>("submain1");
  ASSERT_TRUE(submain1) << "submain1 function should exist";

  auto submain2_2 = module->lookupSymbol<mlir::func::FuncOp>("submain2_2");
  ASSERT_TRUE(submain2_2) << "submain2_2 function should exist";

  // Test pruning: Remove the last output (tensor<4096xbf16>) - keep first two outputs
  std::unordered_set<size_t> resultsToKeep = {0, 1};
  torch_neuronx::pruneFunctionResultsByIndices(func, resultsToKeep);

  EXPECT_EQ(func.getNumResults(), 2) << "Should have 2 results after pruning last output";
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)))
      << "Module should still be valid after pruning";

  // Test index-based pruning: Keep all inputs but only first and third outputs
  auto module2 = createComplexMultiFunctionModule();
  auto func2 = module2->lookupSymbol<mlir::func::FuncOp>("main");

  // Keep outputs at indices 0 and 2
  std::unordered_set<size_t> resultsToKeep2 = {0, 2};
  torch_neuronx::pruneFunctionResultsByIndices(func2, resultsToKeep2);

  EXPECT_EQ(func2.getNumArguments(), 3) << "Should keep all 3 inputs";
  EXPECT_EQ(func2.getNumResults(), 2) << "Should keep 2 outputs (first and third)";
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module2)))
      << "Module should remain valid after pruning";
}

// ============================================================================
// Section 6: Module-Level API Tests
// ============================================================================

TEST_F(FunctionPruneUtilsTest, ModuleLevelPruneArguments) {
  auto module = createModule4in4out();
  ASSERT_TRUE(module);

  // Use module-level API to prune arguments
  std::unordered_set<size_t> indicesToKeep = {0, 2};
  torch_neuronx::pruneModuleArgumentsByIndices(*module, indicesToKeep);

  // Verify the function was modified
  auto func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(func);
  EXPECT_EQ(func.getNumArguments(), 2) << "Should have 2 arguments after module-level pruning";
}

TEST_F(FunctionPruneUtilsTest, ModuleLevelPruneResults) {
  auto module = createModule1in3out();
  ASSERT_TRUE(module);

  // Use module-level API to prune results
  std::unordered_set<size_t> indicesToKeep = {0, 2};
  torch_neuronx::pruneModuleResultsByIndices(*module, indicesToKeep);

  // Verify the function was modified
  auto func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(func);
  EXPECT_EQ(func.getNumResults(), 2) << "Should have 2 results after module-level pruning";
}

TEST_F(FunctionPruneUtilsTest, ModuleLevelFunctionNotFound) {
  auto module = createModule4in4out();
  ASSERT_TRUE(module);

  std::unordered_set<size_t> indicesToKeep = {0};

  // Try to prune a non-existent function
  EXPECT_THROW(
      { torch_neuronx::pruneModuleArgumentsByIndices(*module, indicesToKeep, "nonexistent"); },
      std::runtime_error);
}

TEST_F(FunctionPruneUtilsTest, ModuleLevelComplexMultiFunctionPruning) {
  // Test module-level result pruning with complex real-world multi-function module
  auto module = createComplexMultiFunctionModule();
  ASSERT_TRUE(module);

  // Keep only first and last outputs
  std::unordered_set<size_t> indicesToKeep = {0, 2};
  torch_neuronx::pruneModuleResultsByIndices(*module, indicesToKeep);

  // Verify the main function was modified
  auto func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(func);
  EXPECT_EQ(func.getNumResults(), 2) << "Should have 2 results after module-level pruning";
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)))
      << "Module should still be valid after module-level pruning";
}

TEST_F(FunctionPruneUtilsTest, ModuleLevelComplexMultiFunctionPruningFirstAndThird) {
  // Test module-level index-based pruning with complex multi-function module
  // Keep all inputs, only prune the middle output
  auto module = createComplexMultiFunctionModule();
  ASSERT_TRUE(module);

  // Keep outputs at indices 0 and 2 (first and last)
  std::unordered_set<size_t> indicesToKeep = {0, 2};
  torch_neuronx::pruneModuleResultsByIndices(*module, indicesToKeep);

  // Verify the function was modified
  auto func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(func);
  EXPECT_EQ(func.getNumArguments(), 3) << "Should keep all 3 arguments";
  EXPECT_EQ(func.getNumResults(), 2) << "Should have 2 results after pruning middle output";
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)))
      << "Module should remain valid after module-level pruning";
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
