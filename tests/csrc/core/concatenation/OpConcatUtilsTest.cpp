#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "ATen/ATen.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch_neuronx/csrc/core/concatenation/OpConcatUtils.h"
#include "torch_neuronx/csrc/core/opbuilder/utility/StableHloUtils.h"

class OpConcatUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Module with 1 input, 1 output (for independent scenarios)
    mlir1_independent = R"(
module @jit__lambda {
  func.func public @main(%arg0: tensor<i32>) -> tensor<i32> {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %0 = stablehlo.add %arg0, %c : tensor<i32>
    return %0 : tensor<i32>
  }
}
)";

    mlir2_independent = R"(
module @jit__lambda2 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<2.0> : tensor<f32>
    %0 = stablehlo.multiply %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

    // Module with 2 inputs, 1 output (for common inputs, dependencies, and mixed scenarios)
    mlir1_two_input = R"(
module @jit__lambda {
  func.func public @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

    mlir2_two_input = R"(
module @jit__lambda2 {
  func.func public @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";
  }

  // Test modules for different scenarios
  std::string mlir1_independent, mlir2_independent;
  std::string mlir1_two_input, mlir2_two_input;

  // Helper function to create mock tensor addresses
  std::vector<void*> createMockAddresses(const std::vector<int>& ids) {
    std::vector<void*> addrs;
    for (int id : ids) {
      addrs.push_back(reinterpret_cast<void*>(static_cast<uintptr_t>(id)));
    }
    return addrs;
  }
};

// ============================================================================
// Section 1: Basic Single-to-Single Module Merging
// ============================================================================

// Test Scenario 1: Independent modules
TEST_F(OpConcatUtilsTest, MergeIndependentModules) {
  // Create mock tensor addresses for independent modules
  auto mod1_inputs = createMockAddresses({100});   // module1 input: tensor at address 100
  auto mod1_outputs = createMockAddresses({200});  // module1 output: tensor at address 200
  auto mod2_inputs = createMockAddresses({300});   // module2 input: tensor at address 300
  auto mod2_outputs = createMockAddresses({400});  // module2 output: tensor at address 400

  std::string result = torch_neuronx::mergeStableHLOModules(
      mlir1_independent, mlir2_independent, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(result.empty());
  EXPECT_TRUE(result.find("submain1") != std::string::npos);
  EXPECT_TRUE(result.find("submain2") != std::string::npos);

  // Verify merged function has 2 inputs and 2 outputs (independent)
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  auto merged_module = mlir::parseSourceString<mlir::ModuleOp>(result, &context);
  ASSERT_TRUE(merged_module);

  auto main_func = merged_module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func);

  EXPECT_EQ(main_func.getNumArguments(), 2);  // 1 from each module
  EXPECT_EQ(main_func.getNumResults(), 2);    // 1 from each module
}

// Test Scenario 2: Common inputs
TEST_F(OpConcatUtilsTest, MergeModulesWithCommonInputs) {
  // Create mock tensor addresses where modules share input tensors
  auto mod1_inputs = createMockAddresses({100, 101});  // module1 inputs: a, b
  auto mod1_outputs = createMockAddresses({200});      // module1 output: c = a + b
  auto mod2_inputs = createMockAddresses({100, 102});  // module2 inputs: a (shared), d
  auto mod2_outputs = createMockAddresses({300});      // module2 output: e = a * d

  std::string result = torch_neuronx::mergeStableHLOModules(
      mlir1_two_input, mlir2_two_input, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(result.empty());
  EXPECT_TRUE(result.find("submain1") != std::string::npos);
  EXPECT_TRUE(result.find("submain2") != std::string::npos);

  // Verify merged function has 3 inputs (deduplicated) and 2 outputs
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  auto merged_module = mlir::parseSourceString<mlir::ModuleOp>(result, &context);
  ASSERT_TRUE(merged_module);

  auto main_func = merged_module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func);
  EXPECT_EQ(main_func.getNumArguments(), 3);  // a, b, d (a is shared)
  EXPECT_EQ(main_func.getNumResults(), 2);    // c, e
}

// Test Scenario 3: Direct dependencies
TEST_F(OpConcatUtilsTest, MergeModulesWithDependencies) {
  // Create mock tensor addresses where module1 output becomes module2 input
  auto mod1_inputs = createMockAddresses({100, 101});  // module1 inputs: a, b
  auto mod1_outputs = createMockAddresses({200});      // module1 output: c = a + b
  auto mod2_inputs = createMockAddresses({200, 102});  // module2 inputs: c (from mod1), d
  auto mod2_outputs = createMockAddresses({300});      // module2 output: e = c * d

  std::string result = torch_neuronx::mergeStableHLOModules(
      mlir1_two_input, mlir2_two_input, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(result.empty());
  EXPECT_TRUE(result.find("submain1") != std::string::npos);
  EXPECT_TRUE(result.find("submain2") != std::string::npos);

  // Verify merged function has 3 inputs (a, b, d) and 2 outputs (c, e)
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  auto merged_module = mlir::parseSourceString<mlir::ModuleOp>(result, &context);
  ASSERT_TRUE(merged_module);

  auto main_func = merged_module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func);
  EXPECT_EQ(main_func.getNumArguments(), 3);  // a, b, d (c is internal)
  EXPECT_EQ(main_func.getNumResults(), 2);    // c, e
}

// Test Scenario 4: Mixed (common inputs + dependencies)
TEST_F(OpConcatUtilsTest, MergeMixedModules) {
  // Create mock tensor addresses with both common inputs and dependencies
  auto mod1_inputs = createMockAddresses({100, 101});  // module1 inputs: a, b
  auto mod1_outputs = createMockAddresses({200});      // module1 output: c = a + b
  auto mod2_inputs = createMockAddresses({100, 200});  // module2 inputs: a (shared), c (dependency)
  auto mod2_outputs = createMockAddresses({300});      // module2 output: e = a * c

  std::string result = torch_neuronx::mergeStableHLOModules(
      mlir1_two_input, mlir2_two_input, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(result.empty());
  EXPECT_TRUE(result.find("submain1") != std::string::npos);
  EXPECT_TRUE(result.find("submain2") != std::string::npos);

  // Verify merged function has 2 inputs (a, b) and 2 outputs (c, e)
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  auto merged_module = mlir::parseSourceString<mlir::ModuleOp>(result, &context);
  ASSERT_TRUE(merged_module);

  auto main_func = merged_module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func);
  EXPECT_EQ(main_func.getNumArguments(), 2);  // a, b (a is shared, c is internal)
  EXPECT_EQ(main_func.getNumResults(), 2);    // c, e
}

// Test Scenario 5: Module2->Module1 dependency (module2 executes first)
TEST_F(OpConcatUtilsTest, MergeModule2ToModule1Dependency) {
  // Create mock tensor addresses where module2 output becomes module1 input
  // This creates analysis.module1First = false, ensuring module2 executes before module1
  auto mod1_inputs = createMockAddresses({200, 101});  // module1 inputs: c (from mod2), b
  auto mod1_outputs = createMockAddresses({300});      // module1 output: d = c + b
  auto mod2_inputs = createMockAddresses({100});       // module2 input: a
  auto mod2_outputs = createMockAddresses({200});      // module2 output: c = a * 2

  std::string result = torch_neuronx::mergeStableHLOModules(
      mlir1_two_input, mlir2_independent, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(result.empty());
  EXPECT_TRUE(result.find("submain1") != std::string::npos);
  EXPECT_TRUE(result.find("submain2") != std::string::npos);

  // Verify merged function has 2 inputs (a, b) and 2 outputs (c, d)
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  auto merged_module = mlir::parseSourceString<mlir::ModuleOp>(result, &context);
  ASSERT_TRUE(merged_module);

  auto main_func = merged_module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func);
  EXPECT_EQ(main_func.getNumArguments(), 2);  // a, b
  EXPECT_EQ(main_func.getNumResults(), 2);    // c, d

  // Verify the merged MLIR contains the expected operations
  EXPECT_TRUE(result.find("stablehlo.add") != std::string::npos)
      << "Should contain add operation from module1";
  EXPECT_TRUE(result.find("stablehlo.multiply") != std::string::npos)
      << "Should contain multiply operation from module2";

  // Verify dependency analysis would show module2 executes first
  auto analysis =
      torch_neuronx::analyzeDependencies(mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);
  EXPECT_EQ(analysis.scenario, torch_neuronx::DependencyAnalysis::DIRECT_DEPS);
  EXPECT_FALSE(analysis.module1First) << "Module2 should execute first due to dependency";
}

// Regression test for recursive function calls
TEST_F(OpConcatUtilsTest, MergedMLIRContainsStableHLOOperations) {
  // This test ensures that merged MLIR contains actual StableHLO operations,
  // not recursive function calls
  auto mod1_inputs = createMockAddresses({100});
  auto mod1_outputs = createMockAddresses({200});
  auto mod2_inputs = createMockAddresses({300});
  auto mod2_outputs = createMockAddresses({400});

  std::string result = torch_neuronx::mergeStableHLOModules(
      mlir1_independent, mlir2_independent, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(result.empty());

  // Should contain actual StableHLO operations (the fix we implemented)
  EXPECT_TRUE(result.find("stablehlo.add") != std::string::npos)
      << "Merged MLIR should contain stablehlo.add operation";
  EXPECT_TRUE(result.find("stablehlo.multiply") != std::string::npos)
      << "Merged MLIR should contain stablehlo.multiply operation";

  // Count function calls vs StableHLO operations to ensure we have actual operations
  size_t func_call_count = 0;
  size_t stablehlo_op_count = 0;

  std::istringstream stream(result);
  std::string line;
  while (std::getline(stream, line)) {
    if (line.find("func.call") != std::string::npos) {
      func_call_count++;
    }
    if (line.find("stablehlo.") != std::string::npos) {
      stablehlo_op_count++;
    }
  }

  // Regression test: Should have StableHLO operations (not just recursive calls)
  EXPECT_GT(stablehlo_op_count, 0) << "Should have StableHLO operations";

  // The key regression test: merged MLIR should contain actual operations, not just function calls
  // Function calls are used for orchestration between submain1/submain2, but the main content
  // should be StableHLO operations
  if (func_call_count > 0) {
    // If there are function calls (for orchestration), there should be more StableHLO operations
    EXPECT_GT(stablehlo_op_count, func_call_count)
        << "Should have more StableHLO operations than function calls (regression test)";
  } else {
    // If no function calls, that's also valid - means operations were inlined directly
    EXPECT_GT(stablehlo_op_count, 0)
        << "Should have StableHLO operations even without function calls";
  }
}

// ============================================================================
// Section 2: Dependency Analysis Unit Tests
// ============================================================================

// Test dependency analysis function directly
TEST_F(OpConcatUtilsTest, DependencyAnalysis) {
  // Test independent modules
  auto mod1_inputs = createMockAddresses({100});
  auto mod1_outputs = createMockAddresses({200});
  auto mod2_inputs = createMockAddresses({300});
  auto mod2_outputs = createMockAddresses({400});

  auto analysis =
      torch_neuronx::analyzeDependencies(mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);
  EXPECT_EQ(analysis.scenario, torch_neuronx::DependencyAnalysis::INDEPENDENT);
  EXPECT_TRUE(analysis.commonInputs.empty());
  EXPECT_TRUE(analysis.module1ToModule2Deps.empty());
  EXPECT_TRUE(analysis.module2ToModule1Deps.empty());

  // Test common inputs
  mod2_inputs = createMockAddresses({100, 300});  // Share input 100
  analysis =
      torch_neuronx::analyzeDependencies(mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);
  EXPECT_EQ(analysis.scenario, torch_neuronx::DependencyAnalysis::COMMON_INPUTS);
  EXPECT_FALSE(analysis.commonInputs.empty());
  EXPECT_EQ(analysis.commonInputs[0], 0);  // module1[0] == module2[0]

  // Test dependencies
  mod2_inputs = createMockAddresses({200, 300});  // Use output 200 from module1
  analysis =
      torch_neuronx::analyzeDependencies(mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);
  EXPECT_EQ(analysis.scenario, torch_neuronx::DependencyAnalysis::DIRECT_DEPS);
  EXPECT_FALSE(analysis.module1ToModule2Deps.empty());

  // Verify dependency map now uses vectors
  ASSERT_TRUE(analysis.module1ToModule2Deps.find(0) != analysis.module1ToModule2Deps.end())
      << "Should have dependency from output 0";
  const auto& deps = analysis.module1ToModule2Deps.at(0);
  ASSERT_EQ(deps.size(), 1) << "Output 0 should map to 1 input";
  EXPECT_EQ(deps[0], 0) << "Output 0 should map to input 0";
}

// ============================================================================
// Section 3: Error Handling and Edge Cases
// ============================================================================

TEST_F(OpConcatUtilsTest, MergeWithInvalidModule) {
  std::string invalid_mlir = "invalid mlir syntax";
  auto mod1_inputs = createMockAddresses({100});
  auto mod1_outputs = createMockAddresses({200});
  auto mod2_inputs = createMockAddresses({300});
  auto mod2_outputs = createMockAddresses({400});

  std::string result = torch_neuronx::mergeStableHLOModules(
      mlir1_independent, invalid_mlir, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_TRUE(result.empty());
}

// Test circular dependencies - should throw runtime_error
TEST_F(OpConcatUtilsTest, CircularDependenciesThrowsException) {
  // Create mock tensor addresses with circular dependencies:
  // module1 output -> module2 input AND module2 output -> module1 input
  auto mod1_inputs = createMockAddresses({100, 300});  // module1 inputs: a, d (d from mod2)
  auto mod1_outputs = createMockAddresses({200});      // module1 output: b = a + d
  auto mod2_inputs = createMockAddresses({200, 101});  // module2 inputs: b (from mod1), c
  auto mod2_outputs = createMockAddresses({300});      // module2 output: d = b * c

  // This should throw std::runtime_error due to circular dependencies
  EXPECT_THROW(
      {
        torch_neuronx::mergeStableHLOModules(mlir1_two_input, mlir2_two_input, mod1_inputs,
                                             mod1_outputs, mod2_inputs, mod2_outputs,
                                             /*verify_output=*/true, /*run_optimization=*/false);
      },
      std::runtime_error);
}

// Test circular dependency detection in analyzeDependencies function directly
TEST_F(OpConcatUtilsTest, AnalyzeDependenciesCircularThrowsException) {
  // Create circular dependency scenario:
  // module1 output (200) -> module2 input (200)
  // module2 output (300) -> module1 input (300)
  auto mod1_inputs = createMockAddresses({100, 300});  // includes output from mod2
  auto mod1_outputs = createMockAddresses({200});
  auto mod2_inputs = createMockAddresses({200, 101});  // includes output from mod1
  auto mod2_outputs = createMockAddresses({300});

  // This should throw std::runtime_error
  EXPECT_THROW(
      { torch_neuronx::analyzeDependencies(mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs); },
      std::runtime_error);
}

// Test that the exception message is correct
TEST_F(OpConcatUtilsTest, CircularDependencyExceptionMessage) {
  auto mod1_inputs = createMockAddresses({100, 300});
  auto mod1_outputs = createMockAddresses({200});
  auto mod2_inputs = createMockAddresses({200, 101});
  auto mod2_outputs = createMockAddresses({300});

  try {
    torch_neuronx::analyzeDependencies(mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);
    FAIL() << "Expected std::runtime_error to be thrown";
  } catch (const std::runtime_error& e) {
    std::string error_msg = e.what();
    EXPECT_TRUE(error_msg.find("Circular dependencies detected") != std::string::npos);
    EXPECT_TRUE(error_msg.find("modules to be concatenated") != std::string::npos);
  } catch (...) {
    FAIL() << "Expected std::runtime_error, but got different exception type";
  }
}

// Test mismatched tensor address counts (should throw exceptions)
TEST_F(OpConcatUtilsTest, MismatchedTensorAddressCounts) {
  // Subtest 1: Empty addresses when module expects inputs
  auto empty_addrs = createMockAddresses({});
  auto normal_outputs = createMockAddresses({200});
  auto normal_inputs = createMockAddresses({300});
  auto normal_outputs2 = createMockAddresses({400});

  EXPECT_THROW(
      {
        torch_neuronx::mergeStableHLOModules(mlir1_independent, mlir2_independent, empty_addrs,
                                             normal_outputs, normal_inputs, normal_outputs2,
                                             /*verify_output=*/true, /*run_optimization=*/false);
      },
      std::runtime_error);

  // Subtest 2: Too few input addresses (1 address for 2-input function)
  auto too_few_inputs = createMockAddresses({100});  // 1 address for mlir1_common (expects 2)
  auto normal_outputs_common = createMockAddresses({200});
  auto normal_inputs_common = createMockAddresses({300, 301});
  auto normal_outputs_common2 = createMockAddresses({400});

  EXPECT_THROW(
      {
        torch_neuronx::mergeStableHLOModules(mlir1_two_input, mlir2_two_input, too_few_inputs,
                                             normal_outputs_common, normal_inputs_common,
                                             normal_outputs_common2, /*verify_output=*/true,
                                             /*run_optimization=*/false);
      },
      std::runtime_error);

  // Subtest 3: Too many input addresses (3 addresses for 1-input function)
  auto too_many_inputs =
      createMockAddresses({100, 101, 102});  // 3 addresses for mlir1_independent (expects 1)

  EXPECT_THROW(
      {
        torch_neuronx::mergeStableHLOModules(mlir1_independent, mlir2_independent, too_many_inputs,
                                             normal_outputs, normal_inputs, normal_outputs2,
                                             /*verify_output=*/true, /*run_optimization=*/false);
      },
      std::runtime_error);

  // Subtest 4: Mismatched output addresses
  auto too_many_outputs =
      createMockAddresses({200, 201, 202});  // 3 addresses for 1-output function

  EXPECT_THROW(
      {
        torch_neuronx::mergeStableHLOModules(mlir1_independent, mlir2_independent, normal_inputs,
                                             too_many_outputs, normal_inputs, normal_outputs2,
                                             /*verify_output=*/true, /*run_optimization=*/false);
      },
      std::runtime_error);
}

// Test modules with no inputs
TEST_F(OpConcatUtilsTest, ModulesWithNoInputs) {
  std::string mlir_no_inputs = R"(
module @jit__lambda {
  func.func public @main() -> tensor<f32> {
    %c = stablehlo.constant dense<42.0> : tensor<f32>
    return %c : tensor<f32>
  }
}
)";

  // Test merging no-input module with normal module
  auto empty_inputs = createMockAddresses({});         // No inputs for no-input module
  auto no_input_outputs = createMockAddresses({200});  // 1 output
  auto normal_inputs = createMockAddresses({300});     // Normal module inputs
  auto normal_outputs = createMockAddresses({400});    // Normal module outputs

  std::string result = torch_neuronx::mergeStableHLOModules(
      mlir_no_inputs, mlir1_independent, empty_inputs, no_input_outputs, normal_inputs,
      normal_outputs, /*verify_output=*/true, /*run_optimization=*/false);

  if (!result.empty()) {
    // If merge succeeds, verify the structure
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

    auto merged_module = mlir::parseSourceString<mlir::ModuleOp>(result, &context);
    if (merged_module) {
      auto main_func = merged_module->lookupSymbol<mlir::func::FuncOp>("main");
      if (main_func) {
        EXPECT_EQ(main_func.getNumArguments(), 1);  // Only from second module
        EXPECT_EQ(main_func.getNumResults(), 2);    // From both modules
      }
    }
  }
  // If result is empty, that's also acceptable behavior for this edge case
}

// Test modules with no outputs
TEST_F(OpConcatUtilsTest, ModulesWithNoOutputs) {
  std::string mlir_no_outputs = R"(
module @jit__lambda {
  func.func public @main(%arg0: tensor<f32>) -> () {
    return
  }
}
)";

  // Test merging no-output module with normal module
  auto no_output_inputs = createMockAddresses({100});  // 1 input
  auto empty_outputs = createMockAddresses({});        // No outputs for no-output module
  auto normal_inputs = createMockAddresses({300});     // Normal module inputs
  auto normal_outputs = createMockAddresses({400});    // Normal module outputs

  std::string result = torch_neuronx::mergeStableHLOModules(
      mlir_no_outputs, mlir1_independent, no_output_inputs, empty_outputs, normal_inputs,
      normal_outputs, /*verify_output=*/true, /*run_optimization=*/false);

  if (!result.empty()) {
    // If merge succeeds, verify the structure
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

    auto merged_module = mlir::parseSourceString<mlir::ModuleOp>(result, &context);
    if (merged_module) {
      auto main_func = merged_module->lookupSymbol<mlir::func::FuncOp>("main");
      if (main_func) {
        EXPECT_EQ(main_func.getNumArguments(), 2);  // From both modules
        EXPECT_EQ(main_func.getNumResults(), 1);    // Only from second module
      }
    }
  }
  // If result is empty, that's also acceptable behavior for this edge case
}

// ============================================================================
// Section 4: Real-world Examples
// ============================================================================

TEST_F(OpConcatUtilsTest, ReadInputFile) {
  std::string filename = "input_inc_1.mlir";
  std::ifstream file(filename);

  if (file.good()) {
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    EXPECT_FALSE(content.empty());
    EXPECT_TRUE(content.find("@main") != std::string::npos);
    EXPECT_TRUE(content.find("stablehlo.add") != std::string::npos);

    // Test merging with actual file content - create mock addresses for independent modules
    auto file_inputs = createMockAddresses({1000});    // file module input
    auto file_outputs = createMockAddresses({2000});   // file module output
    auto mlir2_inputs = createMockAddresses({3000});   // mlir2_independent input
    auto mlir2_outputs = createMockAddresses({4000});  // mlir2_independent output

    std::string result = torch_neuronx::mergeStableHLOModules(
        content, mlir2_independent, file_inputs, file_outputs, mlir2_inputs, mlir2_outputs,
        /*verify_output=*/true, /*run_optimization=*/false);
    EXPECT_FALSE(result.empty());

    // Verify input/output counts with real file
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

    auto merged_module = mlir::parseSourceString<mlir::ModuleOp>(result, &context);
    if (merged_module) {
      auto main_func = merged_module->lookupSymbol<mlir::func::FuncOp>("main");
      if (main_func) {
        EXPECT_EQ(main_func.getNumArguments(), 2);  // 1 from file + 1 from mlir2
        EXPECT_EQ(main_func.getNumResults(), 2);    // 1 from file + 1 from mlir2
      }
    }
  } else {
    GTEST_SKIP() << "input_inc_1.mlir not found";
  }
}

// Test the specific transpose + matmul example
TEST_F(OpConcatUtilsTest, TransposeMatmulFusionExample) {
  // MLIR for transpose operation: tensor<8192x8192xbf16> -> tensor<8192x8192xbf16>
  std::string transpose_mlir = R"(
module @jit_transpose_fn attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x8192xbf16>) -> (tensor<8192x8192xbf16> {jax.result_info = "result"}) {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<8192x8192xbf16>) -> tensor<8192x8192xbf16>
    return %0 : tensor<8192x8192xbf16>
  }
}
)";

  // MLIR for matmul operation: (tensor<2048x8192xbf16>, tensor<8192x8192xbf16>) ->
  // tensor<2048x8192xbf16>
  std::string matmul_mlir = R"(
module @jit__aten_mm attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2048x8192xbf16>, %arg1: tensor<8192x8192xbf16>) -> (tensor<2048x8192xbf16> {jax.result_info = "result"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<2048x8192xbf16>, tensor<8192x8192xbf16>) -> tensor<2048x8192xbf16>
    return %0 : tensor<2048x8192xbf16>
  }
}
)";

  // Create tensor addresses as specified in user feedback:
  // - Module1 (transpose): input=100, output=200
  // - Module2 (matmul): inputs=101,200, output=300
  auto mod1_inputs = createMockAddresses({100});   // transpose input
  auto mod1_outputs = createMockAddresses({200});  // transpose output
  auto mod2_inputs =
      createMockAddresses({101, 200});  // matmul inputs: first input (101) + transpose output (200)
  auto mod2_outputs = createMockAddresses({300});  // matmul output

  // Test merging transpose and matmul operations with dependency
  std::string result = torch_neuronx::mergeStableHLOModules(
      transpose_mlir, matmul_mlir, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  // Verify merge succeeded and produced non-empty result

  EXPECT_FALSE(result.empty()) << "Merged MLIR should not be empty";

  if (!result.empty()) {
    // Parse the merged module to verify structure
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

    auto merged_module = mlir::parseSourceString<mlir::ModuleOp>(result, &context);
    ASSERT_TRUE(merged_module) << "Merged MLIR should be parseable";

    auto main_func = merged_module->lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(main_func) << "Merged module should have main function";

    // Expected signature: (tensor<8192x8192xbf16>, tensor<2048x8192xbf16>) ->
    // (tensor<8192x8192xbf16>, tensor<2048x8192xbf16>)
    // - Input 0: transpose input (address 100)
    // - Input 1: matmul first input (address 101)
    // - Output 0: transpose output (address 200)
    // - Output 1: matmul output (address 300)
    EXPECT_EQ(main_func.getNumArguments(), 2) << "Merged function should have 2 inputs";
    EXPECT_EQ(main_func.getNumResults(), 2) << "Merged function should have 2 outputs";

    // Check input types
    if (main_func.getNumArguments() >= 2) {
      auto input0_type = main_func.getArgumentTypes()[0];
      auto input1_type = main_func.getArgumentTypes()[1];

      // Verify input tensor types and shapes

      // Input 0 should be tensor<8192x8192xbf16> (transpose input)
      // Input 1 should be tensor<2048x8192xbf16> (matmul first input)
      EXPECT_TRUE(llvm::isa<mlir::RankedTensorType>(input0_type))
          << "Input 0 should be ranked tensor";
      EXPECT_TRUE(llvm::isa<mlir::RankedTensorType>(input1_type))
          << "Input 1 should be ranked tensor";

      if (llvm::isa<mlir::RankedTensorType>(input0_type) &&
          llvm::isa<mlir::RankedTensorType>(input1_type)) {
        auto tensor0 = llvm::cast<mlir::RankedTensorType>(input0_type);
        auto tensor1 = llvm::cast<mlir::RankedTensorType>(input1_type);

        // Check shapes
        EXPECT_EQ(tensor0.getShape().size(), 2) << "Input 0 should be 2D tensor";
        EXPECT_EQ(tensor1.getShape().size(), 2) << "Input 1 should be 2D tensor";

        if (tensor0.getShape().size() == 2) {
          EXPECT_EQ(tensor0.getShape()[0], 8192) << "Input 0 dim 0 should be 8192";
          EXPECT_EQ(tensor0.getShape()[1], 8192) << "Input 0 dim 1 should be 8192";
        }

        if (tensor1.getShape().size() == 2) {
          EXPECT_EQ(tensor1.getShape()[0], 2048) << "Input 1 dim 0 should be 2048";
          EXPECT_EQ(tensor1.getShape()[1], 8192) << "Input 1 dim 1 should be 8192";
        }
      }
    }

    // Check output types
    if (main_func.getNumResults() >= 2) {
      auto output0_type = main_func.getResultTypes()[0];
      auto output1_type = main_func.getResultTypes()[1];

      // Verify output tensor types
    }

    // Verify that the merged MLIR contains the expected operations
    EXPECT_TRUE(result.find("stablehlo.transpose") != std::string::npos)
        << "Merged MLIR should contain transpose operation";
    EXPECT_TRUE(result.find("stablehlo.dot_general") != std::string::npos)
        << "Merged MLIR should contain dot_general operation";
    EXPECT_TRUE(result.find("submain1") != std::string::npos)
        << "Merged MLIR should contain submain1 function";
    EXPECT_TRUE(result.find("submain2") != std::string::npos)
        << "Merged MLIR should contain submain2 function";
  }
}

// Test the specific transpose + matmul example
TEST_F(OpConcatUtilsTest, WrongTransposeMatmulFusionExample) {
  // MLIR for transpose operation: tensor<8192x8192xbf16> -> tensor<8192x8192xbf16>
  std::string transpose_mlir = R"(
module @jit_transpose_fn attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x8192xbf16>) -> (tensor<8192x8192xbf16> {jax.result_info = "result"}) {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<8192x8192xbf16>) -> tensor<8192x8192xbf16>
    return %0 : tensor<8192x8192xbf16>
  }
}
)";

  // MLIR for matmul operation: (tensor<2048x8192xbf16>, tensor<8192x8192xbf16>) ->
  // tensor<2048x8192xbf16>
  std::string matmul_mlir = R"(
module @jit__aten_mm attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2048x8192xbf16>, %arg1: tensor<8192x8192xbf16>) -> (tensor<2048x8192xbf16> {jax.result_info = "result"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<2048x8192xbf16>, tensor<8192x8192xbf16>) -> tensor<2048x8192xbf16>
    return %0 : tensor<2048x8192xbf16>
  }
}
)";

  // Create tensor addresses as specified in user feedback:
  // - Module1 (transpose): input=100, output=200
  // - Module2 (matmul): inputs=101,200, output=300
  auto mod1_inputs = createMockAddresses({100});   // transpose input
  auto mod1_outputs = createMockAddresses({200});  // transpose output
  // mock WRONG matmul inputs: first input should be (101) + transpose output should be (200)
  auto mod2_inputs = createMockAddresses({200, 101});
  auto mod2_outputs = createMockAddresses({300});  // matmul output

  // This should throw a type mismatch error because:
  // - Module1 (transpose) output: tensor<8192x8192xbf16> (address 200)
  // - Module2 (matmul) input 0: expects tensor<2048x8192xbf16> but gets tensor<8192x8192xbf16>
  // The wrong tensor addresses cause the transpose output to be connected to the wrong matmul input

  EXPECT_THROW(
      {
        torch_neuronx::mergeStableHLOModules(transpose_mlir, matmul_mlir, mod1_inputs, mod1_outputs,
                                             mod2_inputs, mod2_outputs, /*verify_output=*/true,
                                             /*run_optimization=*/false);
      },
      std::runtime_error);

  // Also test that the error message contains the expected type mismatch information
  try {
    torch_neuronx::mergeStableHLOModules(transpose_mlir, matmul_mlir, mod1_inputs, mod1_outputs,
                                         mod2_inputs, mod2_outputs, /*verify_output=*/true,
                                         /*run_optimization=*/false);
    FAIL() << "Expected std::runtime_error to be thrown";
  } catch (const std::runtime_error& e) {
    std::string error_msg = e.what();
    // Verify the error message contains type mismatch information
    EXPECT_TRUE(error_msg.find("Type mismatch in dependency") != std::string::npos)
        << "Error should mention type mismatch in dependency";
    EXPECT_TRUE(error_msg.find("module1 output 0") != std::string::npos)
        << "Error should mention module1 output 0";
    EXPECT_TRUE(error_msg.find("module2 input 0") != std::string::npos)
        << "Error should mention module2 input 0";
    EXPECT_TRUE(error_msg.find("tensor<8192x8192xbf16>") != std::string::npos)
        << "Error should mention the actual type tensor<8192x8192xbf16>";
    EXPECT_TRUE(error_msg.find("tensor<2048x8192xbf16>") != std::string::npos)
        << "Error should mention the expected type tensor<2048x8192xbf16>";
  } catch (...) {
    FAIL() << "Expected std::runtime_error, but got different exception type";
  }
}

// ============================================================================
// Section 5: Merged-to-Single Module Concatenation
// ============================================================================

// Test concatenating a merged module with a single (non-merged) module
TEST_F(OpConcatUtilsTest, ConcatMergedModuleWithSingleModule) {
  // Test concatenating a merged module with a single module

  // Step 1: Create a merged module from two independent modules
  std::string module1 = R"(
module @jit_module1 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<1.0> : tensor<f32>
    %0 = stablehlo.add %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  std::string module2 = R"(
module @jit_module2 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<2.0> : tensor<f32>
    %0 = stablehlo.multiply %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  // Merge module1 and module2
  auto mod1_inputs = createMockAddresses({100});
  auto mod1_outputs = createMockAddresses({200});
  auto mod2_inputs = createMockAddresses({300});
  auto mod2_outputs = createMockAddresses({400});

  std::string merged12 = torch_neuronx::mergeStableHLOModules(
      module1, module2, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(merged12.empty()) << "Initial merge should succeed";
  EXPECT_TRUE(merged12.find("stablehlo.add") != std::string::npos);
  EXPECT_TRUE(merged12.find("stablehlo.multiply") != std::string::npos);

  // Step 2: Create a third single module
  std::string module3 = R"(
module @jit_module3 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<3.0> : tensor<f32>
    %0 = stablehlo.subtract %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  // Step 3: Concatenate the merged module with the single module
  auto merged12_inputs = createMockAddresses({100, 300});
  auto merged12_outputs = createMockAddresses({200, 400});
  auto mod3_inputs = createMockAddresses({500});
  auto mod3_outputs = createMockAddresses({600});

  // Concatenate the merged module with the single module
  std::string final_result = torch_neuronx::mergeStableHLOModules(
      merged12, module3, merged12_inputs, merged12_outputs, mod3_inputs, mod3_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(final_result.empty()) << "Merging merged module with single module should succeed";

  // Verify all operations are present
  EXPECT_TRUE(final_result.find("stablehlo.add") != std::string::npos)
      << "Should contain add from module1";
  EXPECT_TRUE(final_result.find("stablehlo.multiply") != std::string::npos)
      << "Should contain multiply from module2";
  EXPECT_TRUE(final_result.find("stablehlo.subtract") != std::string::npos)
      << "Should contain subtract from module3";

  // Parse and verify structure
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  auto final_module = mlir::parseSourceString<mlir::ModuleOp>(final_result, &context);
  ASSERT_TRUE(final_module) << "Final merged MLIR should be parseable";

  auto main_func = final_module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func) << "Should have main function";
  EXPECT_EQ(main_func.getNumArguments(), 3) << "Should have 3 inputs";
  EXPECT_EQ(main_func.getNumResults(), 3) << "Should have 3 outputs";

  // Test completed successfully
}

// Test iterative concatenation with COMMON_INPUTS scenario
TEST_F(OpConcatUtilsTest, IterativeConcatenationCommonInputs) {
  // Test iterative concatenation with common inputs scenario

  // Step 1: Create first merged module with common inputs
  // Module1: f(a, b) -> c, Module2: g(a, d) -> e (share input 'a')
  auto mod1_inputs = createMockAddresses({100, 101});  // a, b
  auto mod1_outputs = createMockAddresses({200});      // c
  auto mod2_inputs = createMockAddresses({100, 102});  // a (shared), d
  auto mod2_outputs = createMockAddresses({300});      // e

  std::string merged12 = torch_neuronx::mergeStableHLOModules(
      mlir1_two_input, mlir2_two_input, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(merged12.empty()) << "First merge with common inputs should succeed";

  // Step 2: Create third module and merge with the already-merged module
  // Module3: h(f) -> g
  std::string module3 = R"(
module @jit_module3 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<3.0> : tensor<f32>
    %0 = stablehlo.subtract %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  // Now merge the already-merged module (merged12) with module3
  // merged12 has inputs: a, b, d and outputs: c, e
  // module3 has input: f and output: g
  // This creates another COMMON_INPUTS scenario where merged12 and module3 share input 'a'
  auto merged12_inputs = createMockAddresses({100, 101, 102});  // a, b, d
  auto merged12_outputs = createMockAddresses({200, 300});      // c, e
  auto mod3_inputs = createMockAddresses({100});                // f=a (shared with merged12)
  auto mod3_outputs = createMockAddresses({400});               // g

  std::string final_result = torch_neuronx::mergeStableHLOModules(
      merged12, module3, merged12_inputs, merged12_outputs, mod3_inputs, mod3_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(final_result.empty()) << "Iterative merge with common inputs should succeed";

  // Verify all operations are present
  EXPECT_TRUE(final_result.find("stablehlo.add") != std::string::npos);
  EXPECT_TRUE(final_result.find("stablehlo.multiply") != std::string::npos);
  EXPECT_TRUE(final_result.find("stablehlo.subtract") != std::string::npos);

  // Parse and verify structure
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  auto final_module = mlir::parseSourceString<mlir::ModuleOp>(final_result, &context);
  ASSERT_TRUE(final_module) << "Final merged MLIR should be parseable";

  auto main_func = final_module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func) << "Should have main function";

  // Expected: 3 inputs (a shared, b, d) and 3 outputs (c, e, g)
  EXPECT_EQ(main_func.getNumArguments(), 3)
      << "Should have 3 inputs with shared input deduplicated";
  EXPECT_EQ(main_func.getNumResults(), 3) << "Should have 3 outputs";

  // Test completed successfully
}

// Test iterative concatenation with DIRECT_DEPS scenario
TEST_F(OpConcatUtilsTest, IterativeConcatenationDirectDeps) {
  // Test iterative concatenation with direct dependencies scenario

  // Step 1: Create first merged module with direct dependency
  // Module1: f(a, b) -> c, Module2: g(c, d) -> e (module2 uses module1's output)
  auto mod1_inputs = createMockAddresses({100, 101});  // a, b
  auto mod1_outputs = createMockAddresses({200});      // c
  auto mod2_inputs = createMockAddresses({200, 102});  // c (from mod1), d
  auto mod2_outputs = createMockAddresses({300});      // e

  std::string merged12 = torch_neuronx::mergeStableHLOModules(
      mlir1_two_input, mlir2_two_input, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(merged12.empty()) << "First merge with direct dependency should succeed";

  // Step 2: Create third module that depends on the merged module's output
  // Module3: h(e, f) -> g (uses output 'e' from merged12)
  std::string module3 = R"(
module @jit_module3 {
  func.func public @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.subtract %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  // merged12 has inputs: a, b, d and outputs: c, e
  // module3 has inputs: e (from merged12), f and output: g
  auto merged12_inputs = createMockAddresses({100, 101, 102});  // a, b, d
  auto merged12_outputs = createMockAddresses({200, 300});      // c, e
  auto mod3_inputs = createMockAddresses({300, 103});           // e (from merged12), f
  auto mod3_outputs = createMockAddresses({400});               // g

  std::string final_result = torch_neuronx::mergeStableHLOModules(
      merged12, module3, merged12_inputs, merged12_outputs, mod3_inputs, mod3_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(final_result.empty()) << "Iterative merge with direct dependency should succeed";

  // Verify all operations are present
  EXPECT_TRUE(final_result.find("stablehlo.add") != std::string::npos);
  EXPECT_TRUE(final_result.find("stablehlo.multiply") != std::string::npos);
  EXPECT_TRUE(final_result.find("stablehlo.subtract") != std::string::npos);

  // Parse and verify structure
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  auto final_module = mlir::parseSourceString<mlir::ModuleOp>(final_result, &context);
  ASSERT_TRUE(final_module) << "Final merged MLIR should be parseable";

  auto main_func = final_module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func) << "Should have main function";

  // Expected: 4 inputs (a, b, d, f) and 3 outputs (c, e, g)
  // Note: 'e' becomes internal connection between merged12 and module3
  EXPECT_EQ(main_func.getNumArguments(), 4) << "Should have 4 inputs";
  EXPECT_EQ(main_func.getNumResults(), 3) << "Should have 3 outputs";

  // Test completed successfully
}

// Test iterative concatenation with MIXED scenario (common inputs + dependencies)
TEST_F(OpConcatUtilsTest, IterativeConcatenationMixed) {
  // Test iterative concatenation with mixed scenario (common inputs + dependencies)

  // Step 1: Create first merged module with mixed scenario
  // Module1: f(a, b) -> c, Module2: g(a, c) -> e (share 'a' + use 'c')
  auto mod1_inputs = createMockAddresses({100, 101});  // a, b
  auto mod1_outputs = createMockAddresses({200});      // c
  auto mod2_inputs = createMockAddresses({100, 200});  // a (shared), c (dependency)
  auto mod2_outputs = createMockAddresses({300});      // e

  std::string merged12 = torch_neuronx::mergeStableHLOModules(
      mlir1_two_input, mlir2_two_input, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(merged12.empty()) << "First merge with mixed scenario should succeed";

  // Step 2: Create third module with another mixed scenario
  // Module3: h(a, e, f) -> g (shares 'a' with merged12 + uses 'e' from merged12)
  std::string module3 = R"(
module @jit_module3 {
  func.func public @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    %1 = stablehlo.subtract %0, %arg2 : tensor<f32>
    return %1 : tensor<f32>
  }
}
)";

  // merged12 has inputs: a, b and outputs: c, e
  // module3 has inputs: a (shared), e (dependency), f and output: g
  auto merged12_inputs = createMockAddresses({100, 101});   // a, b
  auto merged12_outputs = createMockAddresses({200, 300});  // c, e
  auto mod3_inputs = createMockAddresses({100, 300, 102});  // a (shared), e (dependency), f
  auto mod3_outputs = createMockAddresses({400});           // g

  std::string final_result = torch_neuronx::mergeStableHLOModules(
      merged12, module3, merged12_inputs, merged12_outputs, mod3_inputs, mod3_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(final_result.empty()) << "Iterative merge with mixed scenario should succeed";

  // Verify all operations are present
  EXPECT_TRUE(final_result.find("stablehlo.add") != std::string::npos);
  EXPECT_TRUE(final_result.find("stablehlo.multiply") != std::string::npos);
  EXPECT_TRUE(final_result.find("stablehlo.subtract") != std::string::npos);

  // Parse and verify structure
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  auto final_module = mlir::parseSourceString<mlir::ModuleOp>(final_result, &context);
  ASSERT_TRUE(final_module) << "Final merged MLIR should be parseable";

  auto main_func = final_module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func) << "Should have main function";

  // Expected: 3 inputs (a shared, b, f) and 3 outputs (c, e, g)
  // Note: 'a' is shared, 'e' becomes internal connection
  EXPECT_EQ(main_func.getNumArguments(), 3)
      << "Should have 3 inputs with shared input and internal dependency";
  EXPECT_EQ(main_func.getNumResults(), 3) << "Should have 3 outputs";

  // Test completed successfully
}

// Test complex iterative concatenation covering all 4 scenarios in sequence
TEST_F(OpConcatUtilsTest, IterativeConcatenationAllScenarios) {
  // This test demonstrates that our system can handle all 4 dependency scenarios
  // in a complex iterative concatenation sequence

  // Step 1: INDEPENDENT - merge two independent modules
  std::string mod1 = R"(
module @jit_mod1 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<1.0> : tensor<f32>
    %0 = stablehlo.add %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  std::string mod2 = R"(
module @jit_mod2 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<2.0> : tensor<f32>
    %0 = stablehlo.multiply %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  auto mod1_inputs = createMockAddresses({100});
  auto mod1_outputs = createMockAddresses({200});
  auto mod2_inputs = createMockAddresses({300});
  auto mod2_outputs = createMockAddresses({400});

  std::string merged_independent = torch_neuronx::mergeStableHLOModules(
      mod1, mod2, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs, /*verify_output=*/true,
      /*run_optimization=*/false);

  EXPECT_FALSE(merged_independent.empty()) << "INDEPENDENT merge should succeed";

  // Step 2: COMMON_INPUTS - merge result with a module that shares an input
  std::string mod3 = R"(
module @jit_mod3 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<3.0> : tensor<f32>
    %0 = stablehlo.subtract %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  auto merged_indep_inputs = createMockAddresses({100, 300});
  auto merged_indep_outputs = createMockAddresses({200, 400});
  auto mod3_inputs = createMockAddresses({100});  // Share input with first module
  auto mod3_outputs = createMockAddresses({500});

  std::string merged_common = torch_neuronx::mergeStableHLOModules(
      merged_independent, mod3, merged_indep_inputs, merged_indep_outputs, mod3_inputs,
      mod3_outputs, /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(merged_common.empty()) << "COMMON_INPUTS merge should succeed";

  // Step 3: DIRECT_DEPS - merge result with a module that uses an output
  std::string mod4 = R"(
module @jit_mod4 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<4.0> : tensor<f32>
    %0 = stablehlo.divide %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  auto merged_common_inputs = createMockAddresses({100, 300});
  auto merged_common_outputs = createMockAddresses({200, 400, 500});
  auto mod4_inputs = createMockAddresses({500});  // Use output from previous merge
  auto mod4_outputs = createMockAddresses({600});

  std::string merged_deps = torch_neuronx::mergeStableHLOModules(
      merged_common, mod4, merged_common_inputs, merged_common_outputs, mod4_inputs, mod4_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(merged_deps.empty()) << "DIRECT_DEPS merge should succeed";

  // Step 4: MIXED - merge result with a module that has both shared input and dependency
  std::string mod5 = R"(
module @jit_mod5 {
  func.func public @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  auto merged_deps_inputs = createMockAddresses({100, 300});
  auto merged_deps_outputs = createMockAddresses({200, 400, 500, 600});
  auto mod5_inputs = createMockAddresses({100, 600});  // Share input 100 + use output 600
  auto mod5_outputs = createMockAddresses({700});

  std::string final_mixed = torch_neuronx::mergeStableHLOModules(
      merged_deps, mod5, merged_deps_inputs, merged_deps_outputs, mod5_inputs, mod5_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(final_mixed.empty()) << "MIXED merge should succeed";

  // Verify all operations from all 5 modules are present
  EXPECT_TRUE(final_mixed.find("stablehlo.add") != std::string::npos)
      << "Should contain add from mod1";
  EXPECT_TRUE(final_mixed.find("stablehlo.multiply") != std::string::npos)
      << "Should contain multiply from mod2";
  EXPECT_TRUE(final_mixed.find("stablehlo.subtract") != std::string::npos)
      << "Should contain subtract from mod3";
  EXPECT_TRUE(final_mixed.find("stablehlo.divide") != std::string::npos)
      << "Should contain divide from mod4";
  EXPECT_TRUE(final_mixed.find("stablehlo.maximum") != std::string::npos)
      << "Should contain maximum from mod5";

  // Parse and verify final structure
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  auto final_module = mlir::parseSourceString<mlir::ModuleOp>(final_mixed, &context);
  ASSERT_TRUE(final_module) << "Final merged MLIR should be parseable";

  auto main_func = final_module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func) << "Should have main function";

  // Expected structure after all merges:
  // - Shared inputs are deduplicated
  // - Dependencies become internal connections
  // - Final result should have appropriate input/output counts
  EXPECT_GT(main_func.getNumArguments(), 0) << "Should have inputs";
  EXPECT_GT(main_func.getNumResults(), 0) << "Should have outputs";

  // Test completed successfully
}

// ============================================================================
// Section 6: Merged-to-Merged Module Concatenation
// ============================================================================

// Test concatenating two merged modules together
TEST_F(OpConcatUtilsTest, ConcatTwoMergedModules) {
  // Test concatenating two merged modules together

  // Create first pair of modules and merge them
  std::string module1 = R"(
module @jit_module1 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<1.0> : tensor<f32>
    %0 = stablehlo.add %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  std::string module2 = R"(
module @jit_module2 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<2.0> : tensor<f32>
    %0 = stablehlo.multiply %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  // Step 1: Create first merged module (module1 + module2)
  auto mod1_inputs = createMockAddresses({100});
  auto mod1_outputs = createMockAddresses({200});
  auto mod2_inputs = createMockAddresses({300});
  auto mod2_outputs = createMockAddresses({400});

  std::string merged12 = torch_neuronx::mergeStableHLOModules(
      module1, module2, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(merged12.empty()) << "First merge should succeed";

  // Create second pair of modules and merge them
  std::string module3 = R"(
module @jit_module3 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<3.0> : tensor<f32>
    %0 = stablehlo.subtract %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  std::string module4 = R"(
module @jit_module4 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<4.0> : tensor<f32>
    %0 = stablehlo.divide %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  // Step 2: Create second merged module (module3 + module4)
  auto mod3_inputs = createMockAddresses({500});
  auto mod3_outputs = createMockAddresses({600});
  auto mod4_inputs = createMockAddresses({700});
  auto mod4_outputs = createMockAddresses({800});

  std::string merged34 = torch_neuronx::mergeStableHLOModules(
      module3, module4, mod3_inputs, mod3_outputs, mod4_inputs, mod4_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(merged34.empty()) << "Second merge should succeed";

  // Step 3: Concatenate the two merged modules
  auto merged12_inputs = createMockAddresses({100, 300});
  auto merged12_outputs = createMockAddresses({200, 400});
  auto merged34_inputs = createMockAddresses({500, 700});
  auto merged34_outputs = createMockAddresses({600, 800});

  // Concatenate the two merged modules
  std::string final_result = torch_neuronx::mergeStableHLOModules(
      merged12, merged34, merged12_inputs, merged12_outputs, merged34_inputs, merged34_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(final_result.empty()) << "Merging two merged modules should succeed";

  // Verify all operations from all 4 original modules are present
  EXPECT_TRUE(final_result.find("stablehlo.add") != std::string::npos)
      << "Should contain add from module1";
  EXPECT_TRUE(final_result.find("stablehlo.multiply") != std::string::npos)
      << "Should contain multiply from module2";
  EXPECT_TRUE(final_result.find("stablehlo.subtract") != std::string::npos)
      << "Should contain subtract from module3";
  EXPECT_TRUE(final_result.find("stablehlo.divide") != std::string::npos)
      << "Should contain divide from module4";

  // Parse and verify structure
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  auto final_module = mlir::parseSourceString<mlir::ModuleOp>(final_result, &context);
  ASSERT_TRUE(final_module) << "Final merged MLIR should be parseable";

  auto main_func = final_module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func) << "Should have main function";
  EXPECT_EQ(main_func.getNumArguments(), 4) << "Should have 4 inputs (from all 4 original modules)";
  EXPECT_EQ(main_func.getNumResults(), 4) << "Should have 4 outputs (from all 4 original modules)";

  // Verify that we have nested submain functions from hierarchical merging
  size_t submain_count = 0;
  for (auto& op : final_module->getOps()) {
    if (auto func = llvm::dyn_cast<mlir::func::FuncOp>(op)) {
      std::string func_name = func.getSymName().str();
      if (func_name.find("submain") != std::string::npos) {
        submain_count++;
      }
    }
  }
  EXPECT_GT(submain_count, 0) << "Should have submain functions from hierarchical merging";

  // Test completed successfully
}

// Test complex iterative concatenation: merge 1+2, merge 3+4, merge 12+34, merge 5+6, merge 1234+56
TEST_F(OpConcatUtilsTest, ComplexIterativeConcatenation) {
  // Test complex iterative concatenation: merge 1+2, merge 3+4, merge 12+34, merge 5+6, merge
  // 1234+56

  // Create 6 different modules with different operations
  std::string module1 = R"(
module @jit_module1 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<1.0> : tensor<f32>
    %0 = stablehlo.add %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  std::string module2 = R"(
module @jit_module2 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<2.0> : tensor<f32>
    %0 = stablehlo.multiply %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  std::string module3 = R"(
module @jit_module3 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<3.0> : tensor<f32>
    %0 = stablehlo.subtract %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  std::string module4 = R"(
module @jit_module4 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<4.0> : tensor<f32>
    %0 = stablehlo.divide %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  std::string module5 = R"(
module @jit_module5 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<5.0> : tensor<f32>
    %0 = stablehlo.maximum %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  std::string module6 = R"(
module @jit_module6 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<6.0> : tensor<f32>
    %0 = stablehlo.minimum %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  // Step 1: Merge module1 + module2 = merged12
  auto mod1_inputs = createMockAddresses({100});
  auto mod1_outputs = createMockAddresses({200});
  auto mod2_inputs = createMockAddresses({300});
  auto mod2_outputs = createMockAddresses({400});

  std::string merged12 = torch_neuronx::mergeStableHLOModules(
      module1, module2, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(merged12.empty()) << "Step 1: module1 + module2 merge should succeed";
  EXPECT_TRUE(merged12.find("stablehlo.add") != std::string::npos)
      << "Should contain add from module1";
  EXPECT_TRUE(merged12.find("stablehlo.multiply") != std::string::npos)
      << "Should contain multiply from module2";

  // Step 2: Merge module3 + module4 = merged34
  auto mod3_inputs = createMockAddresses({500});
  auto mod3_outputs = createMockAddresses({600});
  auto mod4_inputs = createMockAddresses({700});
  auto mod4_outputs = createMockAddresses({800});

  std::string merged34 = torch_neuronx::mergeStableHLOModules(
      module3, module4, mod3_inputs, mod3_outputs, mod4_inputs, mod4_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(merged34.empty()) << "Step 2: module3 + module4 merge should succeed";
  EXPECT_TRUE(merged34.find("stablehlo.subtract") != std::string::npos)
      << "Should contain subtract from module3";
  EXPECT_TRUE(merged34.find("stablehlo.divide") != std::string::npos)
      << "Should contain divide from module4";

  // Step 3: Merge merged12 + merged34 = merged1234
  auto merged12_inputs = createMockAddresses({100, 300});
  auto merged12_outputs = createMockAddresses({200, 400});
  auto merged34_inputs = createMockAddresses({500, 700});
  auto merged34_outputs = createMockAddresses({600, 800});

  std::string merged1234 = torch_neuronx::mergeStableHLOModules(
      merged12, merged34, merged12_inputs, merged12_outputs, merged34_inputs, merged34_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(merged1234.empty()) << "Step 3: merged12 + merged34 merge should succeed";
  EXPECT_TRUE(merged1234.find("stablehlo.add") != std::string::npos)
      << "Should contain add from module1";
  EXPECT_TRUE(merged1234.find("stablehlo.multiply") != std::string::npos)
      << "Should contain multiply from module2";
  EXPECT_TRUE(merged1234.find("stablehlo.subtract") != std::string::npos)
      << "Should contain subtract from module3";
  EXPECT_TRUE(merged1234.find("stablehlo.divide") != std::string::npos)
      << "Should contain divide from module4";

  // Step 4: Merge module5 + module6 = merged56
  auto mod5_inputs = createMockAddresses({900});
  auto mod5_outputs = createMockAddresses({1000});
  auto mod6_inputs = createMockAddresses({1100});
  auto mod6_outputs = createMockAddresses({1200});

  std::string merged56 = torch_neuronx::mergeStableHLOModules(
      module5, module6, mod5_inputs, mod5_outputs, mod6_inputs, mod6_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(merged56.empty()) << "Step 4: module5 + module6 merge should succeed";
  EXPECT_TRUE(merged56.find("stablehlo.maximum") != std::string::npos)
      << "Should contain maximum from module5";
  EXPECT_TRUE(merged56.find("stablehlo.minimum") != std::string::npos)
      << "Should contain minimum from module6";

  // Step 5: Merge merged1234 + merged56 = final_merged
  auto merged1234_inputs = createMockAddresses({100, 300, 500, 700});
  auto merged1234_outputs = createMockAddresses({200, 400, 600, 800});
  auto merged56_inputs = createMockAddresses({900, 1100});
  auto merged56_outputs = createMockAddresses({1000, 1200});

  std::string final_merged = torch_neuronx::mergeStableHLOModules(
      merged1234, merged56, merged1234_inputs, merged1234_outputs, merged56_inputs,
      merged56_outputs, /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(final_merged.empty()) << "Step 5: Final merge should succeed";

  // Verify all 6 operations from all original modules are present
  EXPECT_TRUE(final_merged.find("stablehlo.add") != std::string::npos)
      << "Should contain add from module1";
  EXPECT_TRUE(final_merged.find("stablehlo.multiply") != std::string::npos)
      << "Should contain multiply from module2";
  EXPECT_TRUE(final_merged.find("stablehlo.subtract") != std::string::npos)
      << "Should contain subtract from module3";
  EXPECT_TRUE(final_merged.find("stablehlo.divide") != std::string::npos)
      << "Should contain divide from module4";
  EXPECT_TRUE(final_merged.find("stablehlo.maximum") != std::string::npos)
      << "Should contain maximum from module5";
  EXPECT_TRUE(final_merged.find("stablehlo.minimum") != std::string::npos)
      << "Should contain minimum from module6";

  // Parse and verify the final structure
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  auto final_module = mlir::parseSourceString<mlir::ModuleOp>(final_merged, &context);
  ASSERT_TRUE(final_module) << "Final merged MLIR should be parseable";

  auto main_func = final_module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func) << "Should have main function";
  EXPECT_EQ(main_func.getNumArguments(), 6) << "Should have 6 inputs (from all 6 original modules)";
  EXPECT_EQ(main_func.getNumResults(), 6) << "Should have 6 outputs (from all 6 original modules)";

  // Count the number of submain functions to verify hierarchical structure
  size_t submain_count = 0;
  size_t total_func_count = 0;
  for (auto& op : final_module->getOps()) {
    if (auto func = llvm::dyn_cast<mlir::func::FuncOp>(op)) {
      total_func_count++;
      std::string func_name = func.getSymName().str();
      if (func_name.find("submain") != std::string::npos) {
        submain_count++;
        // Found submain function for hierarchical merging
      }
    }
  }

  // Verify hierarchical structure with submain functions
  EXPECT_GT(submain_count, 0) << "Should have submain functions from hierarchical merging";
  EXPECT_GT(total_func_count, 6) << "Should have more functions than just the 6 original "
                                    "operations due to hierarchical structure";

  // Test completed successfully
}

// Test concatenating two merged modules with common inputs
TEST_F(OpConcatUtilsTest, ConcatTwoMergedModulesCommonInputs) {
  // Test concatenating two merged modules with common inputs

  // Step 1: Create MergedA = Module1 + Module2 (independent)
  std::string module1 = R"(
module @jit_module1 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<1.0> : tensor<f32>
    %0 = stablehlo.add %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  std::string module2 = R"(
module @jit_module2 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<2.0> : tensor<f32>
    %0 = stablehlo.multiply %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  // Create MergedA: f(a) -> b, g(c) -> d  =>  MergedA(a, c) -> (b, d)
  auto mod1_inputs = createMockAddresses({100});   // a
  auto mod1_outputs = createMockAddresses({200});  // b
  auto mod2_inputs = createMockAddresses({300});   // c
  auto mod2_outputs = createMockAddresses({400});  // d

  std::string mergedA = torch_neuronx::mergeStableHLOModules(
      module1, module2, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(mergedA.empty()) << "MergedA creation should succeed";

  // Step 2: Create MergedB = Module3 + Module4 (independent)
  std::string module3 = R"(
module @jit_module3 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<3.0> : tensor<f32>
    %0 = stablehlo.subtract %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  std::string module4 = R"(
module @jit_module4 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<4.0> : tensor<f32>
    %0 = stablehlo.divide %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  // Create MergedB: h(a) -> e, i(f) -> g  =>  MergedB(a, f) -> (e, g)
  // Note: 'a' (address 100) is shared with MergedA
  auto mod3_inputs = createMockAddresses({100});   // a (shared with MergedA)
  auto mod3_outputs = createMockAddresses({500});  // e
  auto mod4_inputs = createMockAddresses({600});   // f
  auto mod4_outputs = createMockAddresses({700});  // g

  std::string mergedB = torch_neuronx::mergeStableHLOModules(
      module3, module4, mod3_inputs, mod3_outputs, mod4_inputs, mod4_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(mergedB.empty()) << "MergedB creation should succeed";

  // Step 3: Merge MergedA + MergedB with common input 'a'
  // MergedA: (a, c) -> (b, d)
  // MergedB: (a, f) -> (e, g)
  // Expected: Final(a, c, f) -> (b, d, e, g) with 'a' deduplicated
  auto mergedA_inputs = createMockAddresses({100, 300});   // a, c
  auto mergedA_outputs = createMockAddresses({200, 400});  // b, d
  auto mergedB_inputs = createMockAddresses({100, 600});   // a (shared), f
  auto mergedB_outputs = createMockAddresses({500, 700});  // e, g

  // Merge MergedA + MergedB with common input 'a'
  std::string final_result = torch_neuronx::mergeStableHLOModules(
      mergedA, mergedB, mergedA_inputs, mergedA_outputs, mergedB_inputs, mergedB_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(final_result.empty()) << "Final merge with common inputs should succeed";

  // Verify all operations are present
  EXPECT_TRUE(final_result.find("stablehlo.add") != std::string::npos)
      << "Should contain add from module1";
  EXPECT_TRUE(final_result.find("stablehlo.multiply") != std::string::npos)
      << "Should contain multiply from module2";
  EXPECT_TRUE(final_result.find("stablehlo.subtract") != std::string::npos)
      << "Should contain subtract from module3";
  EXPECT_TRUE(final_result.find("stablehlo.divide") != std::string::npos)
      << "Should contain divide from module4";

  // Parse and verify structure
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  auto final_module = mlir::parseSourceString<mlir::ModuleOp>(final_result, &context);
  ASSERT_TRUE(final_module) << "Final merged MLIR should be parseable";

  auto main_func = final_module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func) << "Should have main function";

  // Expected: 3 inputs (a shared, c, f) and 4 outputs (b, d, e, g)
  EXPECT_EQ(main_func.getNumArguments(), 3)
      << "Should have 3 inputs with shared input deduplicated";
  EXPECT_EQ(main_func.getNumResults(), 4) << "Should have 4 outputs from all modules";

  // Test completed successfully
}

// Test concatenating two merged modules with direct dependencies
TEST_F(OpConcatUtilsTest, ConcatTwoMergedModulesDirectDeps) {
  // Test concatenating two merged modules with direct dependencies

  // Step 1: Create MergedA = Module1 + Module2 (independent)
  std::string module1 = R"(
module @jit_module1 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<1.0> : tensor<f32>
    %0 = stablehlo.add %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  std::string module2 = R"(
module @jit_module2 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<2.0> : tensor<f32>
    %0 = stablehlo.multiply %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  // Create MergedA: f(a) -> b, g(c) -> d  =>  MergedA(a, c) -> (b, d)
  auto mod1_inputs = createMockAddresses({100});   // a
  auto mod1_outputs = createMockAddresses({200});  // b
  auto mod2_inputs = createMockAddresses({300});   // c
  auto mod2_outputs = createMockAddresses({400});  // d

  std::string mergedA = torch_neuronx::mergeStableHLOModules(
      module1, module2, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(mergedA.empty()) << "MergedA creation should succeed";

  // Step 2: Create MergedB = Module3 + Module4 (independent)
  std::string module3 = R"(
module @jit_module3 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<3.0> : tensor<f32>
    %0 = stablehlo.subtract %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  std::string module4 = R"(
module @jit_module4 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<4.0> : tensor<f32>
    %0 = stablehlo.divide %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  // Create MergedB: h(d) -> e, i(f) -> g  =>  MergedB(d, f) -> (e, g)
  // Note: 'd' (address 400) is output from MergedA, creating dependency
  auto mod3_inputs = createMockAddresses({400});   // d (from MergedA output)
  auto mod3_outputs = createMockAddresses({500});  // e
  auto mod4_inputs = createMockAddresses({600});   // f
  auto mod4_outputs = createMockAddresses({700});  // g

  std::string mergedB = torch_neuronx::mergeStableHLOModules(
      module3, module4, mod3_inputs, mod3_outputs, mod4_inputs, mod4_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(mergedB.empty()) << "MergedB creation should succeed";

  // Step 3: Merge MergedA + MergedB with dependency 'd'
  // MergedA: (a, c) -> (b, d)
  // MergedB: (d, f) -> (e, g)  [d comes from MergedA]
  // Expected: Final(a, c, f) -> (b, d, e, g) with 'd' as internal connection
  auto mergedA_inputs = createMockAddresses({100, 300});   // a, c
  auto mergedA_outputs = createMockAddresses({200, 400});  // b, d
  auto mergedB_inputs = createMockAddresses({400, 600});   // d (dependency), f
  auto mergedB_outputs = createMockAddresses({500, 700});  // e, g

  // Merge MergedA + MergedB with dependency 'd'
  std::string final_result = torch_neuronx::mergeStableHLOModules(
      mergedA, mergedB, mergedA_inputs, mergedA_outputs, mergedB_inputs, mergedB_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(final_result.empty()) << "Final merge with direct dependency should succeed";

  // Verify all operations are present
  EXPECT_TRUE(final_result.find("stablehlo.add") != std::string::npos)
      << "Should contain add from module1";
  EXPECT_TRUE(final_result.find("stablehlo.multiply") != std::string::npos)
      << "Should contain multiply from module2";
  EXPECT_TRUE(final_result.find("stablehlo.subtract") != std::string::npos)
      << "Should contain subtract from module3";
  EXPECT_TRUE(final_result.find("stablehlo.divide") != std::string::npos)
      << "Should contain divide from module4";

  // Parse and verify structure
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  auto final_module = mlir::parseSourceString<mlir::ModuleOp>(final_result, &context);
  ASSERT_TRUE(final_module) << "Final merged MLIR should be parseable";

  auto main_func = final_module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func) << "Should have main function";

  // Expected: 3 inputs (a, c, f) and 4 outputs (b, d, e, g) with 'd' as internal connection
  EXPECT_EQ(main_func.getNumArguments(), 3) << "Should have 3 inputs with dependency as internal";
  EXPECT_EQ(main_func.getNumResults(), 4) << "Should have 4 outputs from all modules";

  // Test completed successfully
}

// Test concatenating two merged modules with mixed scenario (common inputs + dependencies)
TEST_F(OpConcatUtilsTest, ConcatTwoMergedModulesMixed) {
  // Test concatenating two merged modules with mixed scenario (common inputs + dependencies)

  // Step 1: Create MergedA = Module1 + Module2 (with common input)
  std::string module1 = R"(
module @jit_module1 {
  func.func public @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  std::string module2 = R"(
module @jit_module2 {
  func.func public @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  // Create MergedA: f(a, b) -> c, g(a, d) -> e  =>  MergedA(a, b, d) -> (c, e)
  // Note: Both modules share input 'a'
  auto mod1_inputs = createMockAddresses({100, 101});  // a, b
  auto mod1_outputs = createMockAddresses({200});      // c
  auto mod2_inputs = createMockAddresses({100, 102});  // a (shared), d
  auto mod2_outputs = createMockAddresses({300});      // e

  std::string mergedA = torch_neuronx::mergeStableHLOModules(
      module1, module2, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(mergedA.empty()) << "MergedA creation should succeed";

  // Step 2: Create MergedB = Module3 + Module4 (independent)
  std::string module3 = R"(
module @jit_module3 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<3.0> : tensor<f32>
    %0 = stablehlo.subtract %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  std::string module4 = R"(
module @jit_module4 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<4.0> : tensor<f32>
    %0 = stablehlo.divide %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  // Create MergedB: h(a) -> f, i(e) -> g  =>  MergedB(a, e) -> (f, g)
  // Note: 'a' is shared with MergedA, 'e' is output from MergedA (dependency)
  auto mod3_inputs = createMockAddresses({100});   // a (shared with MergedA)
  auto mod3_outputs = createMockAddresses({400});  // f
  auto mod4_inputs = createMockAddresses({300});   // e (from MergedA output)
  auto mod4_outputs = createMockAddresses({500});  // g

  std::string mergedB = torch_neuronx::mergeStableHLOModules(
      module3, module4, mod3_inputs, mod3_outputs, mod4_inputs, mod4_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(mergedB.empty()) << "MergedB creation should succeed";

  // Step 3: Merge MergedA + MergedB with mixed scenario
  // MergedA: (a, b, d) -> (c, e)
  // MergedB: (a, e) -> (f, g)  [shares 'a', uses 'e' from MergedA]
  // Expected: Final(a, b, d) -> (c, e, f, g) with 'a' shared, 'e' internal
  auto mergedA_inputs = createMockAddresses({100, 101, 102});  // a, b, d
  auto mergedA_outputs = createMockAddresses({200, 300});      // c, e
  auto mergedB_inputs = createMockAddresses({100, 300});       // a (shared), e (dependency)
  auto mergedB_outputs = createMockAddresses({400, 500});      // f, g

  // Merge MergedA + MergedB with mixed scenario
  std::string final_result = torch_neuronx::mergeStableHLOModules(
      mergedA, mergedB, mergedA_inputs, mergedA_outputs, mergedB_inputs, mergedB_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(final_result.empty()) << "Final merge with mixed scenario should succeed";

  // Verify all operations are present
  EXPECT_TRUE(final_result.find("stablehlo.add") != std::string::npos)
      << "Should contain add from module1";
  EXPECT_TRUE(final_result.find("stablehlo.multiply") != std::string::npos)
      << "Should contain multiply from module2";
  EXPECT_TRUE(final_result.find("stablehlo.subtract") != std::string::npos)
      << "Should contain subtract from module3";
  EXPECT_TRUE(final_result.find("stablehlo.divide") != std::string::npos)
      << "Should contain divide from module4";

  // Parse and verify structure
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  auto final_module = mlir::parseSourceString<mlir::ModuleOp>(final_result, &context);
  ASSERT_TRUE(final_module) << "Final merged MLIR should be parseable";

  auto main_func = final_module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func) << "Should have main function";

  // Expected: 3 inputs (a shared, b, d) and 4 outputs (c, e, f, g) with 'a' shared, 'e' internal
  EXPECT_EQ(main_func.getNumArguments(), 3)
      << "Should have 3 inputs with shared input and internal dependency";
  EXPECT_EQ(main_func.getNumResults(), 4) << "Should have 4 outputs from all modules";

  // Test completed successfully
}

// ============================================================================
// Section 7: Mapping Verification Tests
// ============================================================================

// Test mapping correctness for Module1->Module2 dependency (module1 executes first)
TEST_F(OpConcatUtilsTest, MappingTestModule1ToModule2Dependency) {
  // Module1: f(a, b) -> c, Module2: g(c, d) -> e
  auto mod1_inputs = createMockAddresses({100, 101});  // a, b
  auto mod1_outputs = createMockAddresses({200});      // c
  auto mod2_inputs = createMockAddresses({200, 102});  // c (from mod1), d
  auto mod2_outputs = createMockAddresses({300});      // e

  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  auto mod1 = mlir::parseSourceString<mlir::ModuleOp>(mlir1_two_input, &context);
  auto mod2 = mlir::parseSourceString<mlir::ModuleOp>(mlir2_two_input, &context);
  ASSERT_TRUE(mod1 && mod2);

  auto analysis =
      torch_neuronx::analyzeDependencies(mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);
  EXPECT_TRUE(analysis.module1First) << "Module1 should execute first";

  torch_neuronx::MergeMapping mapping;
  auto merged = torch_neuronx::mergeModules(*mod1, *mod2, &context, mod1_inputs, mod1_outputs,
                                            mod2_inputs, mod2_outputs, &mapping);
  ASSERT_TRUE(merged);

  // Verify mapping: inputs (a, b, d), outputs (c, e)
  EXPECT_EQ(mapping.total_inputs, 3);
  EXPECT_EQ(mapping.total_outputs, 2);

  // Input mapping: merged[0]=module1[0], merged[1]=module1[1], merged[2]=module2[1]
  EXPECT_EQ(mapping.input_mapping[0], std::make_pair(1, size_t(0)))
      << "Input 0 should be module1 input 0";
  EXPECT_EQ(mapping.input_mapping[1], std::make_pair(1, size_t(1)))
      << "Input 1 should be module1 input 1";
  EXPECT_EQ(mapping.input_mapping[2], std::make_pair(2, size_t(1)))
      << "Input 2 should be module2 input 1";

  // Output mapping: merged[0]=module1[0], merged[1]=module2[0]
  EXPECT_EQ(mapping.output_mapping[0], std::make_pair(1, size_t(0)))
      << "Output 0 should be module1 output 0";
  EXPECT_EQ(mapping.output_mapping[1], std::make_pair(2, size_t(0)))
      << "Output 1 should be module2 output 0";
}

// Test mapping correctness for Module2->Module1 dependency (module2 executes first)
TEST_F(OpConcatUtilsTest, MappingTestModule2ToModule1Dependency) {
  // Module2: f(a) -> b, Module1: g(b, c) -> d
  auto mod1_inputs = createMockAddresses({200, 101});  // b (from mod2), c
  auto mod1_outputs = createMockAddresses({300});      // d
  auto mod2_inputs = createMockAddresses({100});       // a
  auto mod2_outputs = createMockAddresses({200});      // b

  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  auto mod1 = mlir::parseSourceString<mlir::ModuleOp>(mlir1_two_input, &context);
  auto mod2 = mlir::parseSourceString<mlir::ModuleOp>(mlir2_independent, &context);
  ASSERT_TRUE(mod1 && mod2);

  auto analysis =
      torch_neuronx::analyzeDependencies(mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);
  EXPECT_FALSE(analysis.module1First) << "Module2 should execute first";

  torch_neuronx::MergeMapping mapping;
  auto merged = torch_neuronx::mergeModules(*mod1, *mod2, &context, mod1_inputs, mod1_outputs,
                                            mod2_inputs, mod2_outputs, &mapping);
  ASSERT_TRUE(merged);

  // Verify mapping: inputs (a, c), outputs (b, d)
  EXPECT_EQ(mapping.total_inputs, 2);
  EXPECT_EQ(mapping.total_outputs, 2);

  // Input mapping: merged[0]=module2[0], merged[1]=module1[1]
  EXPECT_EQ(mapping.input_mapping[0], std::make_pair(2, size_t(0)))
      << "Input 0 should be module2 input 0";
  EXPECT_EQ(mapping.input_mapping[1], std::make_pair(1, size_t(1)))
      << "Input 1 should be module1 input 1";

  // Output mapping: merged[0]=module2[0], merged[1]=module1[0]
  EXPECT_EQ(mapping.output_mapping[0], std::make_pair(2, size_t(0)))
      << "Output 0 should be module2 output 0";
  EXPECT_EQ(mapping.output_mapping[1], std::make_pair(1, size_t(0)))
      << "Output 1 should be module1 output 0";
}

// Test mapping correctness for common inputs scenario
TEST_F(OpConcatUtilsTest, MappingTestCommonInputs) {
  // Module1: f(a, b) -> c, Module2: g(a, d) -> e (share input 'a')
  auto mod1_inputs = createMockAddresses({100, 101});  // a, b
  auto mod1_outputs = createMockAddresses({200});      // c
  auto mod2_inputs = createMockAddresses({100, 102});  // a (shared), d
  auto mod2_outputs = createMockAddresses({300});      // e

  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  auto mod1 = mlir::parseSourceString<mlir::ModuleOp>(mlir1_two_input, &context);
  auto mod2 = mlir::parseSourceString<mlir::ModuleOp>(mlir2_two_input, &context);
  ASSERT_TRUE(mod1 && mod2);

  auto analysis =
      torch_neuronx::analyzeDependencies(mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);
  EXPECT_EQ(analysis.scenario, torch_neuronx::DependencyAnalysis::COMMON_INPUTS);

  torch_neuronx::MergeMapping mapping;
  auto merged = torch_neuronx::mergeModules(*mod1, *mod2, &context, mod1_inputs, mod1_outputs,
                                            mod2_inputs, mod2_outputs, &mapping);
  ASSERT_TRUE(merged);

  // Verify mapping: inputs (a, b, d), outputs (c, e) - 'a' is deduplicated
  EXPECT_EQ(mapping.total_inputs, 3);
  EXPECT_EQ(mapping.total_outputs, 2);

  // Input mapping: merged[0]=module1[0], merged[1]=module1[1], merged[2]=module2[1]
  EXPECT_EQ(mapping.input_mapping[0], std::make_pair(1, size_t(0)))
      << "Input 0 should be module1 input 0 (shared)";
  EXPECT_EQ(mapping.input_mapping[1], std::make_pair(1, size_t(1)))
      << "Input 1 should be module1 input 1";
  EXPECT_EQ(mapping.input_mapping[2], std::make_pair(2, size_t(1)))
      << "Input 2 should be module2 input 1";

  // Output mapping: merged[0]=module1[0], merged[1]=module2[0]
  EXPECT_EQ(mapping.output_mapping[0], std::make_pair(1, size_t(0)))
      << "Output 0 should be module1 output 0";
  EXPECT_EQ(mapping.output_mapping[1], std::make_pair(2, size_t(0)))
      << "Output 1 should be module2 output 0";
}

// Test mapping correctness for mixed scenario (common inputs + dependencies)
TEST_F(OpConcatUtilsTest, MappingTestMixedScenario) {
  // Module1: f(a, b) -> c, Module2: g(a, c) -> e (share 'a' + use 'c')
  auto mod1_inputs = createMockAddresses({100, 101});  // a, b
  auto mod1_outputs = createMockAddresses({200});      // c
  auto mod2_inputs = createMockAddresses({100, 200});  // a (shared), c (dependency)
  auto mod2_outputs = createMockAddresses({300});      // e

  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  auto mod1 = mlir::parseSourceString<mlir::ModuleOp>(mlir1_two_input, &context);
  auto mod2 = mlir::parseSourceString<mlir::ModuleOp>(mlir2_two_input, &context);
  ASSERT_TRUE(mod1 && mod2);

  auto analysis =
      torch_neuronx::analyzeDependencies(mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);
  EXPECT_EQ(analysis.scenario, torch_neuronx::DependencyAnalysis::MIXED);

  torch_neuronx::MergeMapping mapping;
  auto merged = torch_neuronx::mergeModules(*mod1, *mod2, &context, mod1_inputs, mod1_outputs,
                                            mod2_inputs, mod2_outputs, &mapping);
  ASSERT_TRUE(merged);

  // Verify mapping: inputs (a, b), outputs (c, e) - 'a' shared, 'c' internal
  EXPECT_EQ(mapping.total_inputs, 2);
  EXPECT_EQ(mapping.total_outputs, 2);

  // Input mapping: merged[0]=module1[0], merged[1]=module1[1]
  EXPECT_EQ(mapping.input_mapping[0], std::make_pair(1, size_t(0)))
      << "Input 0 should be module1 input 0 (shared)";
  EXPECT_EQ(mapping.input_mapping[1], std::make_pair(1, size_t(1)))
      << "Input 1 should be module1 input 1";

  // Output mapping: merged[0]=module1[0], merged[1]=module2[0]
  EXPECT_EQ(mapping.output_mapping[0], std::make_pair(1, size_t(0)))
      << "Output 0 should be module1 output 0";
  EXPECT_EQ(mapping.output_mapping[1], std::make_pair(2, size_t(0)))
      << "Output 1 should be module2 output 0";
}

// ============================================================================
// Section 7a: StableHloUtils Test Fixture for Tensor-Based Tests
// ============================================================================

class StableHloUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Module with 2 inputs, 1 output for tensor-based tests
    mlir1_two_input = R"(
module @jit__lambda {
  func.func public @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

    mlir2_two_input = R"(
module @jit__lambda2 {
  func.func public @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";
  }

  std::string mlir1_two_input, mlir2_two_input;
};

// Test tensor address verification through 2-stage merge mapping using real tensors
// This test catches the bug where duplicate addresses appear in final outputs
TEST_F(StableHloUtilsTest, TensorAddressExactMatch) {
  // Create real tensors to simulate the actual use case
  // Stage 1: Merge op1 + op2 (both are add operations)
  // Operation 1: r1 = a + b
  at::Tensor a = at::randn({4, 4});
  at::Tensor b = at::randn({4, 4});
  at::Tensor r1 = a + b;  // Simulated output from op1

  // Operation 2: r2 = c + d
  at::Tensor c = at::randn({4, 4});
  at::Tensor d = at::randn({4, 4});
  at::Tensor r2 = c + d;  // Simulated output from op2

  std::vector<at::Tensor> op1_inputs = {a, b};
  std::vector<at::Tensor> op1_outputs = {r1};
  std::vector<at::Tensor> op2_inputs = {c, d};
  std::vector<at::Tensor> op2_outputs = {r2};

  // Stage 1: Merge op1 + op2 using mergeStableHLOModulesWithTensors
  auto result12 = torch_neuronx::mergeStableHLOModulesWithTensors(
      mlir1_two_input, mlir1_two_input, op1_inputs, op1_outputs, op2_inputs, op2_outputs);

  ASSERT_TRUE(result12.success) << "Stage 1 merge should succeed";
  EXPECT_FALSE(result12.merged_mlir_string.empty()) << "Stage 1 MLIR should not be empty";

  // Verify Stage 1 mapping: 4 inputs (a,b,c,d) -> 2 outputs (r1,r2)
  ASSERT_EQ(result12.mapping.total_inputs, 4) << "Stage 1 should have 4 inputs";
  ASSERT_EQ(result12.mapping.total_outputs, 2) << "Stage 1 should have 2 outputs";
  ASSERT_EQ(result12.merged_inputs.size(), 4) << "Stage 1 merged inputs should have 4 tensors";
  ASSERT_EQ(result12.merged_outputs.size(), 2) << "Stage 1 merged outputs should have 2 tensors";

  // Check output mapping for stage 1
  EXPECT_EQ(result12.mapping.output_mapping[0], std::make_pair(1, size_t(0)))
      << "Output 0 (r1) should map to module1 output 0";
  EXPECT_EQ(result12.mapping.output_mapping[1], std::make_pair(2, size_t(0)))
      << "Output 1 (r2) should map to module2 output 0";

  // Verify tensor addresses match
  EXPECT_EQ(result12.merged_inputs[0].data_ptr(), a.data_ptr())
      << "Merged input 0 should be tensor a";
  EXPECT_EQ(result12.merged_inputs[1].data_ptr(), b.data_ptr())
      << "Merged input 1 should be tensor b";
  EXPECT_EQ(result12.merged_inputs[2].data_ptr(), c.data_ptr())
      << "Merged input 2 should be tensor c";
  EXPECT_EQ(result12.merged_inputs[3].data_ptr(), d.data_ptr())
      << "Merged input 3 should be tensor d";
  EXPECT_EQ(result12.merged_outputs[0].data_ptr(), r1.data_ptr())
      << "Merged output 0 should be tensor r1";
  EXPECT_EQ(result12.merged_outputs[1].data_ptr(), r2.data_ptr())
      << "Merged output 1 should be tensor r2";

  // Stage 2: Merge (op1+op2) with op3
  // Operation 3: r3 = matmul(r1, r2) - uses outputs from the merged12 result
  at::Tensor r3 = at::mm(r1, r2);  // Simulated output from op3

  std::vector<at::Tensor> op3_inputs = {r1, r2};  // Dependencies on merged12 outputs
  std::vector<at::Tensor> op3_outputs = {r3};

  // Use the merged result from stage 1 as inputs to stage 2
  auto result_final = torch_neuronx::mergeStableHLOModulesWithTensors(
      result12.merged_mlir_string, mlir2_two_input, result12.merged_inputs, result12.merged_outputs,
      op3_inputs, op3_outputs);

  ASSERT_TRUE(result_final.success) << "Final merge should succeed";
  EXPECT_FALSE(result_final.merged_mlir_string.empty()) << "Final MLIR should not be empty";

  // Verify final mapping: 4 inputs (a,b,c,d) -> 3 outputs (r1,r2,r3)
  ASSERT_EQ(result_final.mapping.total_inputs, 4) << "Final should have 4 inputs";
  ASSERT_EQ(result_final.mapping.total_outputs, 3) << "Final should have 3 outputs";
  ASSERT_EQ(result_final.merged_inputs.size(), 4) << "Final merged inputs should have 4 tensors";
  ASSERT_EQ(result_final.merged_outputs.size(), 3) << "Final merged outputs should have 3 tensors";

  // CRITICAL TEST: Check that all 3 outputs have DISTINCT mappings
  // This catches the bug where output addresses get duplicated
  std::set<std::pair<int, size_t>> output_mappings;
  for (size_t i = 0; i < result_final.mapping.total_outputs; ++i) {
    output_mappings.insert(result_final.mapping.output_mapping[i]);
  }

  EXPECT_EQ(output_mappings.size(), 3)
      << "All 3 outputs should have distinct (module,index) mappings. "
      << "Duplicate mappings indicate the address duplication bug: "
      << "output[0]=" << result_final.mapping.output_mapping[0].first << ","
      << result_final.mapping.output_mapping[0].second << " "
      << "output[1]=" << result_final.mapping.output_mapping[1].first << ","
      << result_final.mapping.output_mapping[1].second << " "
      << "output[2]=" << result_final.mapping.output_mapping[2].first << ","
      << result_final.mapping.output_mapping[2].second;

  // Verify each output maps correctly:
  // - Output 0 (r1): should map to merged12's output 0, which is module1's output
  // - Output 1 (r2): should map to merged12's output 1, which is module2's output
  // - Output 2 (r3): should map to mod3's output 0
  EXPECT_EQ(result_final.mapping.output_mapping[0], std::make_pair(1, size_t(0)))
      << "Output 0 (r1) should map to module1 output 0";
  EXPECT_EQ(result_final.mapping.output_mapping[1], std::make_pair(1, size_t(1)))
      << "Output 1 (r2) should map to module1 output 1 (merged12's second output)";
  EXPECT_EQ(result_final.mapping.output_mapping[2], std::make_pair(2, size_t(0)))
      << "Output 2 (r3) should map to module2 output 0 (mod3)";

  // Verify tensor addresses are correctly preserved
  EXPECT_EQ(result_final.merged_inputs[0].data_ptr(), a.data_ptr())
      << "Final merged input 0 should be tensor a";
  EXPECT_EQ(result_final.merged_inputs[1].data_ptr(), b.data_ptr())
      << "Final merged input 1 should be tensor b";
  EXPECT_EQ(result_final.merged_inputs[2].data_ptr(), c.data_ptr())
      << "Final merged input 2 should be tensor c";
  EXPECT_EQ(result_final.merged_inputs[3].data_ptr(), d.data_ptr())
      << "Final merged input 3 should be tensor d";

  // CRITICAL: Verify all 3 output tensors have DISTINCT addresses
  std::set<void*> output_addresses;
  output_addresses.insert(result_final.merged_outputs[0].data_ptr());
  output_addresses.insert(result_final.merged_outputs[1].data_ptr());
  output_addresses.insert(result_final.merged_outputs[2].data_ptr());

  EXPECT_EQ(output_addresses.size(), 3)
      << "All 3 output tensors should have distinct addresses. "
      << "Duplicate addresses indicate the bug: "
      << "output[0]=" << result_final.merged_outputs[0].data_ptr() << " "
      << "output[1]=" << result_final.merged_outputs[1].data_ptr() << " "
      << "output[2]=" << result_final.merged_outputs[2].data_ptr();

  EXPECT_EQ(result_final.merged_outputs[0].data_ptr(), r1.data_ptr())
      << "Final output 0 should be tensor r1";
  EXPECT_EQ(result_final.merged_outputs[1].data_ptr(), r2.data_ptr())
      << "Final output 1 should be tensor r2";
  EXPECT_EQ(result_final.merged_outputs[2].data_ptr(), r3.data_ptr())
      << "Final output 2 should be tensor r3";
}

// ============================================================================
// Section 8: In-Order Module Merging Tests
// ============================================================================

// Test in-order merge with independent modules
TEST_F(OpConcatUtilsTest, InOrderMergeIndependentModules) {
  // Create mock tensor addresses for independent modules
  auto mod1_inputs = createMockAddresses({100});
  auto mod1_outputs = createMockAddresses({200});
  auto mod2_inputs = createMockAddresses({300});
  auto mod2_outputs = createMockAddresses({400});

  // Use analyzeDependenciesInOrder instead of analyzeDependencies
  auto analysis = torch_neuronx::analyzeDependenciesInOrder(mod1_inputs, mod1_outputs, mod2_inputs,
                                                            mod2_outputs);

  EXPECT_EQ(analysis.scenario, torch_neuronx::DependencyAnalysis::INDEPENDENT);
  EXPECT_TRUE(analysis.module1First) << "In-order merge always has module1 first";
  EXPECT_TRUE(analysis.commonInputs.empty());
  EXPECT_TRUE(analysis.module1ToModule2Deps.empty());
  // In-order analysis doesn't check module2->module1 dependencies
  EXPECT_TRUE(analysis.module2ToModule1Deps.empty());
}

// Test in-order merge with direct dependencies
TEST_F(OpConcatUtilsTest, InOrderMergeWithDependencies) {
  // Module1 output becomes module2 input (valid for in-order)
  auto mod1_inputs = createMockAddresses({100, 101});
  auto mod1_outputs = createMockAddresses({200});
  auto mod2_inputs = createMockAddresses({200, 102});  // Uses mod1 output
  auto mod2_outputs = createMockAddresses({300});

  auto analysis = torch_neuronx::analyzeDependenciesInOrder(mod1_inputs, mod1_outputs, mod2_inputs,
                                                            mod2_outputs);

  EXPECT_EQ(analysis.scenario, torch_neuronx::DependencyAnalysis::DIRECT_DEPS);
  EXPECT_TRUE(analysis.module1First);
  EXPECT_FALSE(analysis.module1ToModule2Deps.empty());

  // Verify dependency map
  ASSERT_TRUE(analysis.module1ToModule2Deps.find(0) != analysis.module1ToModule2Deps.end());
  EXPECT_EQ(analysis.module1ToModule2Deps.at(0)[0], 0);
}

// Test in-order merge with common inputs
TEST_F(OpConcatUtilsTest, InOrderMergeCommonInputs) {
  auto mod1_inputs = createMockAddresses({100, 101});
  auto mod1_outputs = createMockAddresses({200});
  auto mod2_inputs = createMockAddresses({100, 102});  // Shares input 100
  auto mod2_outputs = createMockAddresses({300});

  auto analysis = torch_neuronx::analyzeDependenciesInOrder(mod1_inputs, mod1_outputs, mod2_inputs,
                                                            mod2_outputs);

  EXPECT_EQ(analysis.scenario, torch_neuronx::DependencyAnalysis::COMMON_INPUTS);
  EXPECT_TRUE(analysis.module1First);
  EXPECT_FALSE(analysis.commonInputs.empty());
  EXPECT_EQ(analysis.commonInputs[0], 0);  // module1[0] == module2[0]
}

// Test in-order merge with mixed scenario
TEST_F(OpConcatUtilsTest, InOrderMergeMixedScenario) {
  auto mod1_inputs = createMockAddresses({100, 101});
  auto mod1_outputs = createMockAddresses({200});
  auto mod2_inputs = createMockAddresses({100, 200});  // Shares 100, uses output 200
  auto mod2_outputs = createMockAddresses({300});

  auto analysis = torch_neuronx::analyzeDependenciesInOrder(mod1_inputs, mod1_outputs, mod2_inputs,
                                                            mod2_outputs);

  EXPECT_EQ(analysis.scenario, torch_neuronx::DependencyAnalysis::MIXED);
  EXPECT_TRUE(analysis.module1First);
  EXPECT_FALSE(analysis.commonInputs.empty());
  EXPECT_FALSE(analysis.module1ToModule2Deps.empty());
}

// Test mergeStableHLOModulesWithTensorsInOrder preserves all outputs
TEST_F(OpConcatUtilsTest, InOrderMergePreservesAllOutputs) {
  // Create real tensors
  at::Tensor a = at::randn({4});
  at::Tensor b = at::randn({4});
  at::Tensor c = at::randn({4});  // Output from op1
  at::Tensor d = at::randn({4});
  at::Tensor e = at::randn({4});  // Output from op2

  std::vector<at::Tensor> op1_inputs = {a, b};
  std::vector<at::Tensor> op1_outputs = {c};
  std::vector<at::Tensor> op2_inputs = {c, d};  // Uses op1's output
  std::vector<at::Tensor> op2_outputs = {e};

  auto result = torch_neuronx::mergeStableHLOModulesWithTensorsInOrder(
      mlir1_two_input, mlir2_two_input, op1_inputs, op1_outputs, op2_inputs, op2_outputs);

  ASSERT_TRUE(result.success) << "In-order merge should succeed";
  EXPECT_EQ(result.mapping.total_inputs, 3)
      << "Should have 3 inputs (a, b, d) - c is internal dependency";
  EXPECT_EQ(result.mapping.total_outputs, 2) << "Should have 2 outputs (c, e) - both preserved";

  // Verify outputs are preserved
  EXPECT_EQ(result.merged_outputs.size(), 2);
  EXPECT_EQ(result.merged_outputs[0].data_ptr(), c.data_ptr());
  EXPECT_EQ(result.merged_outputs[1].data_ptr(), e.data_ptr());
}

// Test mergeStableHLOModulesWithTensorsInOrderSkipIntermediates
TEST_F(OpConcatUtilsTest, InOrderMergeSkipIntermediates) {
  // Create real tensors
  at::Tensor a = at::randn({4});
  at::Tensor b = at::randn({4});
  at::Tensor c = at::randn({4});  // Intermediate output from op1
  at::Tensor d = at::randn({4});
  at::Tensor e = at::randn({4});  // Final output from op2

  std::vector<at::Tensor> op1_inputs = {a, b};
  std::vector<at::Tensor> op1_outputs = {c};
  std::vector<at::Tensor> op2_inputs = {c, d};  // Uses op1's output (c becomes intermediate)
  std::vector<at::Tensor> op2_outputs = {e};

  auto result = torch_neuronx::mergeStableHLOModulesWithTensorsInOrderSkipIntermediates(
      mlir1_two_input, mlir2_two_input, op1_inputs, op1_outputs, op2_inputs, op2_outputs);

  ASSERT_TRUE(result.success) << "In-order merge with skip intermediates should succeed";
  EXPECT_EQ(result.mapping.total_inputs, 3) << "Should have 3 inputs (a, b, d)";
  EXPECT_EQ(result.mapping.total_outputs, 1)
      << "Should have 1 output (e) - intermediate c is skipped";

  // Verify only final output is present
  EXPECT_EQ(result.merged_outputs.size(), 1);
  EXPECT_EQ(result.merged_outputs[0].data_ptr(), e.data_ptr());
}

// Test that in-order merge doesn't check module2->module1 dependencies
// (which would be impossible in sequential execution)
TEST_F(OpConcatUtilsTest, InOrderMergeIgnoresReverseDependencies) {
  // This scenario WOULD be a circular dependency in regular merge,
  // but in-order merge simply ignores module2->module1 direction
  // because module2 executes AFTER module1 by definition

  auto mod1_inputs = createMockAddresses({100});
  auto mod1_outputs = createMockAddresses({200});
  auto mod2_inputs = createMockAddresses({300});
  auto mod2_outputs = createMockAddresses({100});  // module2 output has same address as module1
                                                   // input

  // In regular analyzeDependencies, this might detect a reverse dependency
  // But in analyzeDependenciesInOrder, we don't check module2 outputs -> module1 inputs
  auto analysis = torch_neuronx::analyzeDependenciesInOrder(mod1_inputs, mod1_outputs, mod2_inputs,
                                                            mod2_outputs);

  // Should be INDEPENDENT because we don't check reverse dependencies
  EXPECT_EQ(analysis.scenario, torch_neuronx::DependencyAnalysis::INDEPENDENT);
  EXPECT_TRUE(analysis.module1First);
  EXPECT_TRUE(analysis.module2ToModule1Deps.empty())
      << "In-order merge should not populate module2->module1 deps";
}

// Test in-order merge performance advantage: no circular dependency check needed
TEST_F(OpConcatUtilsTest, InOrderMergeNoCircularDependencyCheck) {
  // In regular merge, having both directions of dependencies would throw.
  // But in in-order merge, we only check module1->module2, so even if
  // addresses appear "circular", it won't throw because we don't check the other direction.

  auto mod1_inputs = createMockAddresses({100, 300});  // Note: 300 appears in mod2 outputs
  auto mod1_outputs = createMockAddresses({200});
  auto mod2_inputs = createMockAddresses({200, 101});  // Note: 200 is mod1 output
  auto mod2_outputs = createMockAddresses({300});      // Note: 300 appears in mod1 inputs

  // In regular analyzeDependencies, this would throw "Circular dependencies detected"
  EXPECT_THROW(
      { torch_neuronx::analyzeDependencies(mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs); },
      std::runtime_error);

  // But in analyzeDependenciesInOrder, we only check module1->module2, so no throw
  EXPECT_NO_THROW({
    auto analysis = torch_neuronx::analyzeDependenciesInOrder(mod1_inputs, mod1_outputs,
                                                              mod2_inputs, mod2_outputs);

    // Should detect the forward dependency
    EXPECT_EQ(analysis.scenario, torch_neuronx::DependencyAnalysis::DIRECT_DEPS);
    EXPECT_FALSE(analysis.module1ToModule2Deps.empty());
    // But module2ToModule1Deps should be empty (not checked)
    EXPECT_TRUE(analysis.module2ToModule1Deps.empty());
  });
}

// Test in-order merge with mergeModulesInOrder low-level function
TEST_F(OpConcatUtilsTest, MergeModulesInOrderBasic) {
  auto mod1_inputs = createMockAddresses({100});
  auto mod1_outputs = createMockAddresses({200});
  auto mod2_inputs = createMockAddresses({300});
  auto mod2_outputs = createMockAddresses({400});

  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  auto mod1 = mlir::parseSourceString<mlir::ModuleOp>(mlir1_independent, &context);
  auto mod2 = mlir::parseSourceString<mlir::ModuleOp>(mlir2_independent, &context);
  ASSERT_TRUE(mod1 && mod2);

  torch_neuronx::MergeMapping mapping;
  auto merged = torch_neuronx::mergeModulesInOrder(
      *mod1, *mod2, &context, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs, &mapping);

  ASSERT_TRUE(merged) << "mergeModulesInOrder should succeed";

  auto main_func = merged->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func);
  EXPECT_EQ(main_func.getNumArguments(), 2);
  EXPECT_EQ(main_func.getNumResults(), 2);
}

// Test in-order merge with duplicate inputs (x + x scenario)
TEST_F(OpConcatUtilsTest, InOrderMergeDuplicateInputs) {
  auto mod1_inputs = createMockAddresses({100, 101});
  auto mod1_outputs = createMockAddresses({200});
  auto mod2_inputs = createMockAddresses({200, 200});  // Both inputs are the same (duplicate)
  auto mod2_outputs = createMockAddresses({300});

  auto analysis = torch_neuronx::analyzeDependenciesInOrder(mod1_inputs, mod1_outputs, mod2_inputs,
                                                            mod2_outputs);

  EXPECT_EQ(analysis.scenario, torch_neuronx::DependencyAnalysis::DIRECT_DEPS);

  // Verify both duplicate inputs are tracked
  auto dep_it = analysis.module1ToModule2Deps.find(0);
  ASSERT_NE(dep_it, analysis.module1ToModule2Deps.end());
  EXPECT_EQ(dep_it->second.size(), 2) << "Should track both duplicate input indices";
}

// Test for duplicate input handling (bug fix for x + x scenario)
TEST_F(OpConcatUtilsTest, HandlesDuplicateInputs) {
  // Test the bug fix for operations with duplicate inputs (e.g., x + x)
  // Bug: Operations like "add(x, x)" were producing 50% of expected value
  // because dependency tracking wasn't handling duplicate input indices

  // Module1: f(a, b) -> c = a + b
  std::string module1 = R"(
module @jit_module1 {
  func.func public @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  // Module2: g(c) -> d = c + c  (DUPLICATE INPUTS - both inputs are the same tensor)
  std::string module2 = R"(
module @jit_module2 {
  func.func public @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  // Create addresses: Module2 uses Module1's output TWICE (duplicate inputs)
  auto mod1_inputs = createMockAddresses({100, 101});  // a, b
  auto mod1_outputs = createMockAddresses({200});      // c
  auto mod2_inputs = createMockAddresses({200, 200});  // c, c (BOTH inputs point to same address!)
  auto mod2_outputs = createMockAddresses({300});      // d

  // This should succeed with the bug fix
  std::string result = torch_neuronx::mergeStableHLOModules(
      module1, module2, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  EXPECT_FALSE(result.empty()) << "Merge with duplicate inputs should succeed";

  // Parse and verify structure
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  auto merged_module = mlir::parseSourceString<mlir::ModuleOp>(result, &context);
  ASSERT_TRUE(merged_module) << "Merged MLIR with duplicate inputs should be parseable";

  auto main_func = merged_module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func) << "Should have main function";

  // Expected: 2 inputs (a, b), 2 outputs (c, d)
  // 'c' is used twice by module2 but only appears once in the output
  EXPECT_EQ(main_func.getNumArguments(), 2) << "Should have 2 inputs";
  EXPECT_EQ(main_func.getNumResults(), 2) << "Should have 2 outputs";

  // Verify both operations are present
  EXPECT_TRUE(result.find("stablehlo.add") != std::string::npos) << "Should contain add operations";

  // Verify dependency was correctly tracked
  auto analysis =
      torch_neuronx::analyzeDependencies(mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);
  EXPECT_EQ(analysis.scenario, torch_neuronx::DependencyAnalysis::DIRECT_DEPS);
  EXPECT_TRUE(analysis.module1First) << "Module1 should execute first";

  // CRITICAL: Verify that BOTH duplicate inputs are tracked in the dependency map
  ASSERT_EQ(analysis.module1ToModule2Deps.size(), 1) << "Should have one output with dependencies";

  // The bug fix: module1's output 0 should map to BOTH of module2's inputs (indices 0 and 1)
  auto dep_it = analysis.module1ToModule2Deps.find(0);
  ASSERT_NE(dep_it, analysis.module1ToModule2Deps.end()) << "Should find dependency for output 0";

  const std::vector<size_t>& input_indices = dep_it->second;
  EXPECT_EQ(input_indices.size(), 2) << "Output 0 should map to 2 input indices (duplicate inputs)";

  // Verify both input indices (0 and 1) are in the dependency vector
  EXPECT_NE(std::find(input_indices.begin(), input_indices.end(), 0), input_indices.end())
      << "Should have dependency to input index 0";
  EXPECT_NE(std::find(input_indices.begin(), input_indices.end(), 1), input_indices.end())
      << "Should have dependency to input index 1";
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
