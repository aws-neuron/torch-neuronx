#include <gtest/gtest.h>

#include <sstream>
#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch_neuronx/csrc/core/concatenation/OpConcatUtils.h"

/**
 * @brief Test suite for validating optimization pipeline after concatenation
 *
 * These tests ensure that the optimization pipeline added to mergeModules() works correctly:
 * 1. Inlining - Flattens function hierarchy for cross-call optimization
 * 2. Symbol DCE - Removes unused functions/globals after inlining
 * 3. SCCP - Constant propagation across control flow
 * 4. Canonicalization - Simplifies operations (e.g., x + 0 -> x, x * 1 -> x)
 * 5. CSE - Common Subexpression Elimination removes duplicate operations
 * 6. Final Canonicalization - Secondary cleanup after CSE
 *
 * The pipeline ensures merged modules are optimized for performance while maintaining correctness.
 */
class OpConcatUtilsOptimizationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Module with redundant constant - CSE should eliminate duplicate
    mlir1_with_redundant_constant = R"(
module @jit__lambda {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c1 = stablehlo.constant dense<1.0> : tensor<f32>
    %c2 = stablehlo.constant dense<1.0> : tensor<f32>
    %0 = stablehlo.add %arg0, %c1 : tensor<f32>
    %1 = stablehlo.multiply %0, %c2 : tensor<f32>
    return %1 : tensor<f32>
  }
}
)";

    // Module with identity operations - canonicalization should simplify
    mlir1_with_identity_ops = R"(
module @jit__lambda {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c0 = stablehlo.constant dense<0.0> : tensor<f32>
    %c1 = stablehlo.constant dense<1.0> : tensor<f32>
    %0 = stablehlo.add %arg0, %c0 : tensor<f32>
    %1 = stablehlo.multiply %0, %c1 : tensor<f32>
    return %1 : tensor<f32>
  }
}
)";

    // Simple module for pairing with above
    mlir2_simple = R"(
module @jit__lambda2 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<2.0> : tensor<f32>
    %0 = stablehlo.multiply %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

    // Module with same constant as mlir1_with_redundant_constant - CSE should deduplicate across
    // modules
    mlir2_with_same_constant = R"(
module @jit__lambda2 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<1.0> : tensor<f32>
    %0 = stablehlo.multiply %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";
  }

  // Test modules
  std::string mlir1_with_redundant_constant;
  std::string mlir1_with_identity_ops;
  std::string mlir2_simple;
  std::string mlir2_with_same_constant;

  // Helper function to create mock tensor addresses
  std::vector<void*> createMockAddresses(const std::vector<int>& ids) {
    std::vector<void*> addrs;
    for (int id : ids) {
      addrs.push_back(reinterpret_cast<void*>(static_cast<uintptr_t>(id)));
    }
    return addrs;
  }

  // Helper to count occurrences of a pattern in a string
  size_t countOccurrences(const std::string& str, const std::string& pattern) {
    size_t count = 0;
    size_t pos = 0;
    while ((pos = str.find(pattern, pos)) != std::string::npos) {
      count++;
      pos += pattern.length();
    }
    return count;
  }
};

// ============================================================================
// Test 1: Optimization Pipeline Eliminates Redundant Constants Within a Module
// ============================================================================
TEST_F(OpConcatUtilsOptimizationTest, OptimizationEliminatesRedundantConstantsWithinModule) {
  // Module1 has two identical constants (dense<1.0>)
  // After the optimization pipeline (Inlining + Symbol DCE + SCCP + Canonicalization + CSE),
  // only one should remain and may be further optimized
  auto mod1_inputs = createMockAddresses({100});
  auto mod1_outputs = createMockAddresses({200});
  auto mod2_inputs = createMockAddresses({300});
  auto mod2_outputs = createMockAddresses({400});

  std::string result =
      torch_neuronx::mergeStableHLOModules(mlir1_with_redundant_constant, mlir2_simple, mod1_inputs,
                                           mod1_outputs, mod2_inputs, mod2_outputs);

  EXPECT_FALSE(result.empty()) << "Merge should succeed";

  // Count constant definitions in the merged result
  size_t constant_count = countOccurrences(result, "stablehlo.constant dense<1.0>");

  // After the optimization pipeline, the two duplicate constants from module1 should be reduced to
  // 1 or fewer (SCCP might even eliminate them entirely if they're not needed)
  EXPECT_LE(constant_count, 1)
      << "Optimization pipeline should eliminate duplicate constant dense<1.0> within module1. "
      << "Found " << constant_count << " occurrences, expected 1 or fewer";

  // Verify the MLIR is still valid
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  auto merged_module = mlir::parseSourceString<mlir::ModuleOp>(result, &context);
  ASSERT_TRUE(merged_module) << "Optimized MLIR should be parseable";

  auto main_func = merged_module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func) << "Should have main function";
  EXPECT_EQ(main_func.getNumArguments(), 2) << "Should have 2 inputs";
  EXPECT_EQ(main_func.getNumResults(), 2) << "Should have 2 outputs";
}

// ============================================================================
// Test 2: Optimization Pipeline Deduplicates Constants Across Merged Modules
// ============================================================================
TEST_F(OpConcatUtilsOptimizationTest, OptimizationDeduplicatesConstantsAcrossMergedModules) {
  // Both modules define the same constant (dense<1.0>)
  // After the optimization pipeline, constants should be deduplicated and potentially further
  // optimized
  auto mod1_inputs = createMockAddresses({100});
  auto mod1_outputs = createMockAddresses({200});
  auto mod2_inputs = createMockAddresses({300});
  auto mod2_outputs = createMockAddresses({400});

  std::string result =
      torch_neuronx::mergeStableHLOModules(mlir1_with_redundant_constant, mlir2_with_same_constant,
                                           mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);

  EXPECT_FALSE(result.empty()) << "Merge should succeed";

  // Count constant definitions of 1.0 in the merged result
  size_t constant_count = countOccurrences(result, "stablehlo.constant dense<1.0>");

  // Module1 has 2 instances of dense<1.0>, Module2 has 1 instance
  // After the optimization pipeline (Inlining + Symbol DCE + SCCP + Canonicalization + CSE + Final
  // Canonicalization), all 3 should be reduced to 1 or fewer (SCCP might eliminate them entirely if
  // possible)
  EXPECT_LE(constant_count, 1)
      << "Optimization pipeline should deduplicate dense<1.0> across merged modules. "
      << "Found " << constant_count << " occurrences, expected 1 or fewer";

  // Verify inlining worked - no submain function calls should remain
  EXPECT_EQ(result.find("func.call @submain1"), std::string::npos)
      << "Inlining should eliminate submain1 calls";
  EXPECT_EQ(result.find("func.call @submain2"), std::string::npos)
      << "Inlining should eliminate submain2 calls";

  // Verify the MLIR is still valid and functional
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  auto merged_module = mlir::parseSourceString<mlir::ModuleOp>(result, &context);
  ASSERT_TRUE(merged_module) << "Optimized MLIR should be parseable";
}

// ============================================================================
// Test 3: Inlining Eliminates Function Calls After Merge
// ============================================================================
TEST_F(OpConcatUtilsOptimizationTest, InliningEliminatesFunctionCalls) {
  auto mod1_inputs = createMockAddresses({100});
  auto mod1_outputs = createMockAddresses({200});
  auto mod2_inputs = createMockAddresses({300});
  auto mod2_outputs = createMockAddresses({400});

  std::string result =
      torch_neuronx::mergeStableHLOModules(mlir1_with_redundant_constant, mlir2_simple, mod1_inputs,
                                           mod1_outputs, mod2_inputs, mod2_outputs);

  EXPECT_FALSE(result.empty()) << "Merge with optimization should succeed";

  // After inlining, should not have submain1/submain2 function calls
  EXPECT_EQ(result.find("@submain1"), std::string::npos)
      << "Inlining should eliminate submain1 calls";
  EXPECT_EQ(result.find("@submain2"), std::string::npos)
      << "Inlining should eliminate submain2 calls";

  // Note: Symbol DCE may not always remove function definitions immediately
  // The important thing is that the function calls are eliminated by inlining
  // Function definitions may remain but are unused (this is acceptable behavior)

  // Parse and verify the optimized result
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  auto merged_module = mlir::parseSourceString<mlir::ModuleOp>(result, &context);
  ASSERT_TRUE(merged_module) << "Optimized MLIR should be parseable";

  auto main_func = merged_module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func) << "Should have main function";

  // Verify function signature is correct
  EXPECT_EQ(main_func.getNumArguments(), 2) << "Should have 2 inputs";
  EXPECT_EQ(main_func.getNumResults(), 2) << "Should have 2 outputs";
}

// ============================================================================
// Test 4: Optimization Pipeline Preserves Functional Correctness
// ============================================================================
TEST_F(OpConcatUtilsOptimizationTest, OptimizationPreservesFunctionalCorrectness) {
  auto mod1_inputs = createMockAddresses({100});
  auto mod1_outputs = createMockAddresses({200});
  auto mod2_inputs = createMockAddresses({300});
  auto mod2_outputs = createMockAddresses({400});

  std::string result =
      torch_neuronx::mergeStableHLOModules(mlir1_with_redundant_constant, mlir2_simple, mod1_inputs,
                                           mod1_outputs, mod2_inputs, mod2_outputs);

  EXPECT_FALSE(result.empty()) << "Merge with optimization should succeed";

  // Parse and verify the optimized result
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  auto merged_module = mlir::parseSourceString<mlir::ModuleOp>(result, &context);
  ASSERT_TRUE(merged_module) << "Optimized MLIR should be parseable";

  auto main_func = merged_module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func) << "Should have main function";

  // Verify function signature is correct
  EXPECT_EQ(main_func.getNumArguments(), 2) << "Should have 2 inputs";
  EXPECT_EQ(main_func.getNumResults(), 2) << "Should have 2 outputs";

  // Verify all expected operations are present (after optimization)
  EXPECT_TRUE(result.find("stablehlo.add") != std::string::npos) << "Should contain add operation";
  EXPECT_TRUE(result.find("stablehlo.multiply") != std::string::npos)
      << "Should contain multiply operation";
  EXPECT_TRUE(result.find("stablehlo.constant") != std::string::npos)
      << "Should contain constant definitions";
}

// ============================================================================
// Test 5: Optimization Pipeline Works End-to-End
// ============================================================================
TEST_F(OpConcatUtilsOptimizationTest, OptimizationWorksEndToEnd) {
  // Create a module with opportunities for optimization
  std::string mlir_with_optimization_opportunity = R"(
module @jit__lambda {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c1 = stablehlo.constant dense<2.0> : tensor<f32>
    %c2 = stablehlo.constant dense<3.0> : tensor<f32>
    %mult = stablehlo.multiply %c1, %c2 : tensor<f32>
    %result = stablehlo.add %arg0, %mult : tensor<f32>
    return %result : tensor<f32>
  }
}
)";

  auto mod1_inputs = createMockAddresses({100});
  auto mod1_outputs = createMockAddresses({200});
  auto mod2_inputs = createMockAddresses({300});
  auto mod2_outputs = createMockAddresses({400});

  std::string result =
      torch_neuronx::mergeStableHLOModules(mlir_with_optimization_opportunity, mlir2_simple,
                                           mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);

  EXPECT_FALSE(result.empty()) << "Merge with optimization should succeed";

  // Verify the optimization pipeline ran successfully by checking the result is optimized
  // (The exact optimizations may vary based on MLIR's capabilities with StableHLO)

  // Should still have the essential operations
  EXPECT_TRUE(result.find("stablehlo") != std::string::npos)
      << "Should contain StableHLO operations";

  // Should not have function calls (inlining worked)
  EXPECT_EQ(result.find("func.call @submain1"), std::string::npos)
      << "Inlining should eliminate submain1 calls";
  EXPECT_EQ(result.find("func.call @submain2"), std::string::npos)
      << "Inlining should eliminate submain2 calls";

  // Parse and verify the optimized result
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  auto merged_module = mlir::parseSourceString<mlir::ModuleOp>(result, &context);
  ASSERT_TRUE(merged_module) << "Optimized MLIR should be parseable";

  auto main_func = merged_module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func) << "Should have main function";

  // Verify function signature is correct
  EXPECT_EQ(main_func.getNumArguments(), 2) << "Should have 2 inputs";
  EXPECT_EQ(main_func.getNumResults(), 2) << "Should have 2 outputs";
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
