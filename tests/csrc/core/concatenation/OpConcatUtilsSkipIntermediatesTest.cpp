#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch_neuronx/csrc/core/concatenation/OpConcatUtils.h"
#include "torch_neuronx/csrc/core/opbuilder/utility/StableHloUtils.h"

class StableHloUtilsSkipIntermediatesTest : public ::testing::Test {
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
// Section 1: Basic SkipIntermediates Merge Tests - Intermediate Output Exclusion
// ============================================================================

// Test Scenario 1: Independent modules (should behave same as conservative version)
TEST_F(StableHloUtilsSkipIntermediatesTest, SkipIntermediatesMergeIndependentModules) {
  // Create mock tensor addresses for independent modules
  auto mod1_inputs = createMockAddresses({100});   // module1 input: tensor at address 100
  auto mod1_outputs = createMockAddresses({200});  // module1 output: tensor at address 200
  auto mod2_inputs = createMockAddresses({300});   // module2 input: tensor at address 300
  auto mod2_outputs = createMockAddresses({400});  // module2 output: tensor at address 400

  // Create MLIR context and load dialects
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  // Parse modules
  auto mod1 = mlir::parseSourceString<mlir::ModuleOp>(mlir1_independent, &context);
  auto mod2 = mlir::parseSourceString<mlir::ModuleOp>(mlir2_independent, &context);

  ASSERT_TRUE(mod1) << "Module 1 should parse successfully";
  ASSERT_TRUE(mod2) << "Module 2 should parse successfully";

  // Analyze dependencies
  auto analysis =
      torch_neuronx::analyzeDependencies(mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);

  EXPECT_EQ(analysis.scenario, torch_neuronx::DependencyAnalysis::INDEPENDENT);

  // Call mergeModulesSkipIntermediates directly
  torch_neuronx::MergeMapping mapping;
  auto skip_intermediates_merged = torch_neuronx::mergeModulesSkipIntermediates(
      *mod1, *mod2, &context, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs, analysis,
      &mapping, /*verify_output=*/true, /*run_optimization=*/false);

  ASSERT_TRUE(skip_intermediates_merged)
      << "SkipIntermediates merge should succeed for independent modules";

  auto main_func = skip_intermediates_merged->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func) << "Merged module should have main function";

  // For independent modules, skip intermediates should behave same as conservative: 2 inputs, 2
  // outputs
  EXPECT_EQ(main_func.getNumArguments(), 2) << "Should have 2 inputs (1 from each module)";
  EXPECT_EQ(main_func.getNumResults(), 2) << "Should have 2 outputs (1 from each module)";

  // Verify the merged MLIR contains the expected operations
  std::string result = torch_neuronx::stablehlo_utils::moduleToString(*skip_intermediates_merged);
  EXPECT_TRUE(result.find("stablehlo.add") != std::string::npos) << "Should contain add operation";
  EXPECT_TRUE(result.find("stablehlo.multiply") != std::string::npos)
      << "Should contain multiply operation";
  EXPECT_TRUE(result.find("submain1") != std::string::npos) << "Should contain submain1 function";
  EXPECT_TRUE(result.find("submain2") != std::string::npos) << "Should contain submain2 function";
}

// Test Scenario 2: Common inputs (should behave same as conservative version)
TEST_F(StableHloUtilsSkipIntermediatesTest, SkipIntermediatesMergeModulesWithCommonInputs) {
  // Create mock tensor addresses where modules share input tensors
  auto mod1_inputs = createMockAddresses({100, 101});  // module1 inputs: a, b
  auto mod1_outputs = createMockAddresses({200});      // module1 output: c = a + b
  auto mod2_inputs = createMockAddresses({100, 102});  // module2 inputs: a (shared), d
  auto mod2_outputs = createMockAddresses({300});      // module2 output: e = a * d

  // Create MLIR context and load dialects
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  // Parse modules
  auto mod1 = mlir::parseSourceString<mlir::ModuleOp>(mlir1_two_input, &context);
  auto mod2 = mlir::parseSourceString<mlir::ModuleOp>(mlir2_two_input, &context);

  ASSERT_TRUE(mod1) << "Module 1 should parse successfully";
  ASSERT_TRUE(mod2) << "Module 2 should parse successfully";

  // Analyze dependencies
  auto analysis =
      torch_neuronx::analyzeDependencies(mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);

  EXPECT_EQ(analysis.scenario, torch_neuronx::DependencyAnalysis::COMMON_INPUTS);

  // Call mergeModulesSkipIntermediates directly
  torch_neuronx::MergeMapping mapping;
  auto skip_intermediates_merged = torch_neuronx::mergeModulesSkipIntermediates(
      *mod1, *mod2, &context, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs, analysis,
      &mapping, /*verify_output=*/true, /*run_optimization=*/false);

  ASSERT_TRUE(skip_intermediates_merged)
      << "SkipIntermediates merge should succeed for common inputs";

  auto main_func = skip_intermediates_merged->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func) << "Merged module should have main function";

  // For common inputs without dependencies, skip intermediates should behave same as conservative
  EXPECT_EQ(main_func.getNumArguments(), 3) << "Should have 3 inputs (a shared, b, d)";
  EXPECT_EQ(main_func.getNumResults(), 2) << "Should have 2 outputs (c, e)";
}

// Test Scenario 3: Direct dependencies - KEY DIFFERENCE FROM CONSERVATIVE VERSION
TEST_F(StableHloUtilsSkipIntermediatesTest, SkipIntermediatesMergeModulesWithDependencies) {
  // Create mock tensor addresses where module1 output becomes module2 input
  auto mod1_inputs = createMockAddresses({100, 101});  // module1 inputs: a, b
  auto mod1_outputs = createMockAddresses({200});      // module1 output: c = a + b
  auto mod2_inputs = createMockAddresses({200, 102});  // module2 inputs: c (from mod1), d
  auto mod2_outputs = createMockAddresses({300});      // module2 output: e = c * d

  // Create MLIR context and load dialects
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  // Parse modules
  auto mod1 = mlir::parseSourceString<mlir::ModuleOp>(mlir1_two_input, &context);
  auto mod2 = mlir::parseSourceString<mlir::ModuleOp>(mlir2_two_input, &context);

  ASSERT_TRUE(mod1) << "Module 1 should parse successfully";
  ASSERT_TRUE(mod2) << "Module 2 should parse successfully";

  // Analyze dependencies
  auto analysis =
      torch_neuronx::analyzeDependencies(mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);

  EXPECT_EQ(analysis.scenario, torch_neuronx::DependencyAnalysis::DIRECT_DEPS);

  // Call mergeModulesSkipIntermediates directly
  torch_neuronx::MergeMapping mapping;
  auto skip_intermediates_merged = torch_neuronx::mergeModulesSkipIntermediates(
      *mod1, *mod2, &context, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs, analysis,
      &mapping, /*verify_output=*/true, /*run_optimization=*/false);

  ASSERT_TRUE(skip_intermediates_merged)
      << "SkipIntermediates merge should succeed for dependencies";

  auto main_func = skip_intermediates_merged->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func) << "Merged module should have main function";

  // KEY DIFFERENCE: SkipIntermediates version excludes intermediate output 'c'
  // Conservative version: 3 inputs (a, b, d), 2 outputs (c, e)
  // SkipIntermediates version: 3 inputs (a, b, d), 1 output (e) - 'c' is excluded as intermediate
  EXPECT_EQ(main_func.getNumArguments(), 3) << "Should have 3 inputs (a, b, d)";
  EXPECT_EQ(main_func.getNumResults(), 1)
      << "Should have 1 output (e) - 'c' excluded as intermediate";
}

// Test Scenario 4: Mixed (common inputs + dependencies) - KEY DIFFERENCE FROM CONSERVATIVE VERSION
TEST_F(StableHloUtilsSkipIntermediatesTest, SkipIntermediatesMergeMixedModules) {
  // Create mock tensor addresses with both common inputs and dependencies
  auto mod1_inputs = createMockAddresses({100, 101});  // module1 inputs: a, b
  auto mod1_outputs = createMockAddresses({200});      // module1 output: c = a + b
  auto mod2_inputs = createMockAddresses({100, 200});  // module2 inputs: a (shared), c (dependency)
  auto mod2_outputs = createMockAddresses({300});      // module2 output: e = a * c

  // Create MLIR context and load dialects
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  // Parse modules
  auto mod1 = mlir::parseSourceString<mlir::ModuleOp>(mlir1_two_input, &context);
  auto mod2 = mlir::parseSourceString<mlir::ModuleOp>(mlir2_two_input, &context);

  ASSERT_TRUE(mod1) << "Module 1 should parse successfully";
  ASSERT_TRUE(mod2) << "Module 2 should parse successfully";

  // Analyze dependencies
  auto analysis =
      torch_neuronx::analyzeDependencies(mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);

  EXPECT_EQ(analysis.scenario, torch_neuronx::DependencyAnalysis::MIXED);

  // Call mergeModulesSkipIntermediates directly
  torch_neuronx::MergeMapping mapping;
  auto skip_intermediates_merged = torch_neuronx::mergeModulesSkipIntermediates(
      *mod1, *mod2, &context, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs, analysis,
      &mapping, /*verify_output=*/true, /*run_optimization=*/false);

  ASSERT_TRUE(skip_intermediates_merged)
      << "SkipIntermediates merge should succeed for mixed scenario";

  auto main_func = skip_intermediates_merged->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func) << "Merged module should have main function";

  // KEY DIFFERENCE: SkipIntermediates version excludes intermediate output 'c'
  // Conservative version: 2 inputs (a, b), 2 outputs (c, e)
  // SkipIntermediates version: 2 inputs (a, b), 1 output (e) - 'c' excluded as intermediate
  EXPECT_EQ(main_func.getNumArguments(), 2) << "Should have 2 inputs (a, b)";
  EXPECT_EQ(main_func.getNumResults(), 1)
      << "Should have 1 output (e) - 'c' excluded as intermediate";
}

// ============================================================================
// Section 2: Comparison Tests - Conservative vs SkipIntermediates Behavior
// ============================================================================

// Test comparing conservative vs skip intermediates merge for direct dependencies
TEST_F(StableHloUtilsSkipIntermediatesTest, CompareSkipIntermediatesVsConservativeDependencies) {
  auto mod1_inputs = createMockAddresses({100, 101});  // module1 inputs: a, b
  auto mod1_outputs = createMockAddresses({200});      // module1 output: c = a + b
  auto mod2_inputs = createMockAddresses({200, 102});  // module2 inputs: c (from mod1), d
  auto mod2_outputs = createMockAddresses({300});      // module2 output: e = c * d

  // Create conservative merged module
  std::string conservative_result = torch_neuronx::mergeStableHLOModules(
      mlir1_two_input, mlir2_two_input, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  // Create MLIR context and load dialects
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  // Parse modules for skip intermediates merge
  auto mod1 = mlir::parseSourceString<mlir::ModuleOp>(mlir1_two_input, &context);
  auto mod2 = mlir::parseSourceString<mlir::ModuleOp>(mlir2_two_input, &context);

  ASSERT_TRUE(mod1) << "Module 1 should parse successfully";
  ASSERT_TRUE(mod2) << "Module 2 should parse successfully";

  // Analyze dependencies
  auto analysis =
      torch_neuronx::analyzeDependencies(mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);

  EXPECT_EQ(analysis.scenario, torch_neuronx::DependencyAnalysis::DIRECT_DEPS);

  // Call mergeModulesSkipIntermediates directly
  torch_neuronx::MergeMapping mapping;
  auto skip_intermediates_merged = torch_neuronx::mergeModulesSkipIntermediates(
      *mod1, *mod2, &context, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs, analysis,
      &mapping, /*verify_output=*/true, /*run_optimization=*/false);

  ASSERT_FALSE(conservative_result.empty()) << "Conservative merge should succeed";
  ASSERT_TRUE(skip_intermediates_merged) << "SkipIntermediates merge should succeed";

  // Parse conservative result
  auto conservative_module = mlir::parseSourceString<mlir::ModuleOp>(conservative_result, &context);
  ASSERT_TRUE(conservative_module) << "Conservative merged MLIR should be parseable";

  auto conservative_main = conservative_module->lookupSymbol<mlir::func::FuncOp>("main");
  auto skip_intermediates_main =
      skip_intermediates_merged->lookupSymbol<mlir::func::FuncOp>("main");

  ASSERT_TRUE(conservative_main) << "Conservative module should have main function";
  ASSERT_TRUE(skip_intermediates_main) << "SkipIntermediates module should have main function";

  // CRITICAL: Compare input counts (should be EXACTLY the same - skip NEVER adds inputs)
  EXPECT_EQ(conservative_main.getNumArguments(), skip_intermediates_main.getNumArguments())
      << "Both versions should have same number of inputs";
  EXPECT_LE(skip_intermediates_main.getNumArguments(), conservative_main.getNumArguments())
      << "CONSTRAINT: Skip intermediate should NEVER add extra inputs vs non-skip";
  EXPECT_EQ(conservative_main.getNumArguments(), 3) << "Both should have 3 inputs";
  EXPECT_EQ(skip_intermediates_main.getNumArguments(), 3) << "Skip version should have 3 inputs";

  // Compare output counts (should be different - skip reduces outputs)
  EXPECT_EQ(conservative_main.getNumResults(), 2)
      << "Conservative version should have 2 outputs (c, e)";
  EXPECT_EQ(skip_intermediates_main.getNumResults(), 1)
      << "SkipIntermediates version should have 1 output (e only)";
  EXPECT_LT(skip_intermediates_main.getNumResults(), conservative_main.getNumResults())
      << "Skip intermediate should have fewer outputs than conservative";

  // Verify both contain the same operations
  std::string skip_intermediates_result =
      torch_neuronx::stablehlo_utils::moduleToString(*skip_intermediates_merged);
  EXPECT_TRUE(conservative_result.find("stablehlo.add") != std::string::npos);
  EXPECT_TRUE(skip_intermediates_result.find("stablehlo.add") != std::string::npos);
  EXPECT_TRUE(conservative_result.find("stablehlo.multiply") != std::string::npos);
  EXPECT_TRUE(skip_intermediates_result.find("stablehlo.multiply") != std::string::npos);
}

// Test comparing conservative vs skip intermediates merge for mixed scenario
TEST_F(StableHloUtilsSkipIntermediatesTest, CompareSkipIntermediatesVsConservativeMixed) {
  auto mod1_inputs = createMockAddresses({100, 101});  // module1 inputs: a, b
  auto mod1_outputs = createMockAddresses({200});      // module1 output: c = a + b
  auto mod2_inputs = createMockAddresses({100, 200});  // module2 inputs: a (shared), c (dependency)
  auto mod2_outputs = createMockAddresses({300});      // module2 output: e = a * c

  // Create conservative merged module
  std::string conservative_result = torch_neuronx::mergeStableHLOModules(
      mlir1_two_input, mlir2_two_input, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  // Create MLIR context and load dialects
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  // Parse modules for skip intermediates merge
  auto mod1 = mlir::parseSourceString<mlir::ModuleOp>(mlir1_two_input, &context);
  auto mod2 = mlir::parseSourceString<mlir::ModuleOp>(mlir2_two_input, &context);

  ASSERT_TRUE(mod1) << "Module 1 should parse successfully";
  ASSERT_TRUE(mod2) << "Module 2 should parse successfully";

  // Analyze dependencies
  auto analysis =
      torch_neuronx::analyzeDependencies(mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);

  EXPECT_EQ(analysis.scenario, torch_neuronx::DependencyAnalysis::MIXED);

  // Call mergeModulesSkipIntermediates directly
  torch_neuronx::MergeMapping mapping;
  auto skip_intermediates_merged = torch_neuronx::mergeModulesSkipIntermediates(
      *mod1, *mod2, &context, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs, analysis,
      &mapping, /*verify_output=*/true, /*run_optimization=*/false);

  ASSERT_FALSE(conservative_result.empty()) << "Conservative merge should succeed";
  ASSERT_TRUE(skip_intermediates_merged) << "SkipIntermediates merge should succeed";

  // Parse conservative result
  auto conservative_module = mlir::parseSourceString<mlir::ModuleOp>(conservative_result, &context);
  ASSERT_TRUE(conservative_module) << "Conservative merged MLIR should be parseable";

  auto conservative_main = conservative_module->lookupSymbol<mlir::func::FuncOp>("main");
  auto skip_intermediates_main =
      skip_intermediates_merged->lookupSymbol<mlir::func::FuncOp>("main");

  ASSERT_TRUE(conservative_main) << "Conservative module should have main function";
  ASSERT_TRUE(skip_intermediates_main) << "SkipIntermediates module should have main function";

  // Compare input counts (should be same)
  EXPECT_EQ(conservative_main.getNumArguments(), skip_intermediates_main.getNumArguments())
      << "Both versions should have same number of inputs";
  EXPECT_EQ(conservative_main.getNumArguments(), 2) << "Both should have 2 inputs";

  // Compare output counts (should be different)
  EXPECT_EQ(conservative_main.getNumResults(), 2)
      << "Conservative version should have 2 outputs (c, e)";
  EXPECT_EQ(skip_intermediates_main.getNumResults(), 1)
      << "SkipIntermediates version should have 1 output (e only)";
}

// ============================================================================
// Section 3: Critical Constraint Test - Skip NEVER Adds Inputs
// ============================================================================

// NEW TEST: Explicitly verify that skip intermediate NEVER adds extra inputs compared to
// conservative
TEST_F(StableHloUtilsSkipIntermediatesTest, SkipIntermediateNeverAddsExtraInputs) {
  // Test all dependency scenarios to ensure input counts are always <= conservative version

  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  // Test Scenario 1: Direct Dependencies
  {
    auto mod1_inputs = createMockAddresses({100, 101});
    auto mod1_outputs = createMockAddresses({200});
    auto mod2_inputs = createMockAddresses({200, 102});
    auto mod2_outputs = createMockAddresses({300});

    std::string conservative = torch_neuronx::mergeStableHLOModules(
        mlir1_two_input, mlir2_two_input, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs,
        /*verify_output=*/true, /*run_optimization=*/false);

    auto mod1 = mlir::parseSourceString<mlir::ModuleOp>(mlir1_two_input, &context);
    auto mod2 = mlir::parseSourceString<mlir::ModuleOp>(mlir2_two_input, &context);
    auto analysis =
        torch_neuronx::analyzeDependencies(mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);
    torch_neuronx::MergeMapping mapping;
    auto skip = torch_neuronx::mergeModulesSkipIntermediates(
        *mod1, *mod2, &context, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs, analysis,
        &mapping, /*verify_output=*/true, /*run_optimization=*/false);

    auto cons_mod = mlir::parseSourceString<mlir::ModuleOp>(conservative, &context);
    auto cons_main = cons_mod->lookupSymbol<mlir::func::FuncOp>("main");
    auto skip_main = skip->lookupSymbol<mlir::func::FuncOp>("main");

    EXPECT_LE(skip_main.getNumArguments(), cons_main.getNumArguments())
        << "DIRECT_DEPS: Skip should NEVER add extra inputs";
    EXPECT_EQ(skip_main.getNumArguments(), cons_main.getNumArguments())
        << "DIRECT_DEPS: Skip should have SAME input count as conservative";
  }

  // Test Scenario 2: Mixed (Common + Dependencies)
  {
    auto mod1_inputs = createMockAddresses({100, 101});
    auto mod1_outputs = createMockAddresses({200});
    auto mod2_inputs = createMockAddresses({100, 200});
    auto mod2_outputs = createMockAddresses({300});

    std::string conservative = torch_neuronx::mergeStableHLOModules(
        mlir1_two_input, mlir2_two_input, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs,
        /*verify_output=*/true, /*run_optimization=*/false);

    auto mod1 = mlir::parseSourceString<mlir::ModuleOp>(mlir1_two_input, &context);
    auto mod2 = mlir::parseSourceString<mlir::ModuleOp>(mlir2_two_input, &context);
    auto analysis =
        torch_neuronx::analyzeDependencies(mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);
    torch_neuronx::MergeMapping mapping;
    auto skip = torch_neuronx::mergeModulesSkipIntermediates(
        *mod1, *mod2, &context, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs, analysis,
        &mapping, /*verify_output=*/true, /*run_optimization=*/false);

    auto cons_mod = mlir::parseSourceString<mlir::ModuleOp>(conservative, &context);
    auto cons_main = cons_mod->lookupSymbol<mlir::func::FuncOp>("main");
    auto skip_main = skip->lookupSymbol<mlir::func::FuncOp>("main");

    EXPECT_LE(skip_main.getNumArguments(), cons_main.getNumArguments())
        << "MIXED: Skip should NEVER add extra inputs";
    EXPECT_EQ(skip_main.getNumArguments(), cons_main.getNumArguments())
        << "MIXED: Skip should have SAME input count as conservative";
  }

  // Test Scenario 3: Common Inputs Only
  {
    auto mod1_inputs = createMockAddresses({100, 101});
    auto mod1_outputs = createMockAddresses({200});
    auto mod2_inputs = createMockAddresses({100, 102});
    auto mod2_outputs = createMockAddresses({300});

    std::string conservative = torch_neuronx::mergeStableHLOModules(
        mlir1_two_input, mlir2_two_input, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs,
        /*verify_output=*/true, /*run_optimization=*/false);

    auto mod1 = mlir::parseSourceString<mlir::ModuleOp>(mlir1_two_input, &context);
    auto mod2 = mlir::parseSourceString<mlir::ModuleOp>(mlir2_two_input, &context);
    auto analysis =
        torch_neuronx::analyzeDependencies(mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);
    torch_neuronx::MergeMapping mapping;
    auto skip = torch_neuronx::mergeModulesSkipIntermediates(
        *mod1, *mod2, &context, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs, analysis,
        &mapping, /*verify_output=*/true, /*run_optimization=*/false);

    auto cons_mod = mlir::parseSourceString<mlir::ModuleOp>(conservative, &context);
    auto cons_main = cons_mod->lookupSymbol<mlir::func::FuncOp>("main");
    auto skip_main = skip->lookupSymbol<mlir::func::FuncOp>("main");

    EXPECT_LE(skip_main.getNumArguments(), cons_main.getNumArguments())
        << "COMMON_INPUTS: Skip should NEVER add extra inputs";
    EXPECT_EQ(skip_main.getNumArguments(), cons_main.getNumArguments())
        << "COMMON_INPUTS: Skip should have SAME input count as conservative";
  }

  // Test Scenario 4: Independent Modules
  {
    auto mod1_inputs = createMockAddresses({100});
    auto mod1_outputs = createMockAddresses({200});
    auto mod2_inputs = createMockAddresses({300});
    auto mod2_outputs = createMockAddresses({400});

    std::string conservative = torch_neuronx::mergeStableHLOModules(
        mlir1_independent, mlir2_independent, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs,
        /*verify_output=*/true, /*run_optimization=*/false);

    auto mod1 = mlir::parseSourceString<mlir::ModuleOp>(mlir1_independent, &context);
    auto mod2 = mlir::parseSourceString<mlir::ModuleOp>(mlir2_independent, &context);
    auto analysis =
        torch_neuronx::analyzeDependencies(mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);
    torch_neuronx::MergeMapping mapping;
    auto skip = torch_neuronx::mergeModulesSkipIntermediates(
        *mod1, *mod2, &context, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs, analysis,
        &mapping, /*verify_output=*/true, /*run_optimization=*/false);

    auto cons_mod = mlir::parseSourceString<mlir::ModuleOp>(conservative, &context);
    auto cons_main = cons_mod->lookupSymbol<mlir::func::FuncOp>("main");
    auto skip_main = skip->lookupSymbol<mlir::func::FuncOp>("main");

    EXPECT_LE(skip_main.getNumArguments(), cons_main.getNumArguments())
        << "INDEPENDENT: Skip should NEVER add extra inputs";
    EXPECT_EQ(skip_main.getNumArguments(), cons_main.getNumArguments())
        << "INDEPENDENT: Skip should have SAME input count as conservative";
  }
}

// ============================================================================
// Section 4: Complex Dependency Chain Tests
// ============================================================================

// Test multiple output dependencies - some intermediate, some final
TEST_F(StableHloUtilsSkipIntermediatesTest, SkipIntermediatesMultipleOutputDependencies) {
  // Module1: f(a, b) -> (c, d) - two outputs
  std::string module1_multi_output = R"(
module @jit_module1 {
  func.func public @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<f32>
    return %0, %1 : tensor<f32>, tensor<f32>
  }
}
)";

  // Module2: g(c, e) -> f - uses first output of module1, plus external input
  std::string module2_single_input = R"(
module @jit_module2 {
  func.func public @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  auto mod1_inputs = createMockAddresses({100, 101});   // a, b
  auto mod1_outputs = createMockAddresses({200, 201});  // c, d
  auto mod2_inputs = createMockAddresses({200, 102});   // c (from mod1), e
  auto mod2_outputs = createMockAddresses({300});       // f

  // Create MLIR context and load dialects
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  // Parse modules
  auto mod1 = mlir::parseSourceString<mlir::ModuleOp>(module1_multi_output, &context);
  auto mod2 = mlir::parseSourceString<mlir::ModuleOp>(module2_single_input, &context);

  ASSERT_TRUE(mod1) << "Module 1 should parse successfully";
  ASSERT_TRUE(mod2) << "Module 2 should parse successfully";

  // Analyze dependencies
  auto analysis =
      torch_neuronx::analyzeDependencies(mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);

  EXPECT_EQ(analysis.scenario, torch_neuronx::DependencyAnalysis::DIRECT_DEPS);

  // Call mergeModulesSkipIntermediates directly
  torch_neuronx::MergeMapping mapping;
  auto skip_intermediates_merged = torch_neuronx::mergeModulesSkipIntermediates(
      *mod1, *mod2, &context, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs, analysis,
      &mapping, /*verify_output=*/true, /*run_optimization=*/false);

  ASSERT_TRUE(skip_intermediates_merged) << "SkipIntermediates merge should succeed";

  auto main_func = skip_intermediates_merged->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func) << "Merged module should have main function";

  // Expected behavior:
  // - Inputs: a, b, e (3 inputs)
  // - Outputs: d, f (2 outputs) - 'c' is excluded as intermediate, 'd' is kept as final output
  EXPECT_EQ(main_func.getNumArguments(), 3) << "Should have 3 inputs (a, b, e)";
  EXPECT_EQ(main_func.getNumResults(), 2) << "Should have 2 outputs (d, f) - 'c' excluded";
}

// Test chain of dependencies: A -> B -> C
TEST_F(StableHloUtilsSkipIntermediatesTest, SkipIntermediatesChainedDependencies) {
  // This test requires creating a 3-module chain, but since we can only merge 2 at a time,
  // we'll test a 2-step process: first merge A+B, then merge (A+B)+C

  // Module A: f(x) -> y
  std::string moduleA = R"(
module @jit_moduleA {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<1.0> : tensor<f32>
    %0 = stablehlo.add %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  // Module B: g(y) -> z
  std::string moduleB = R"(
module @jit_moduleB {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<2.0> : tensor<f32>
    %0 = stablehlo.multiply %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  // Module C: h(z) -> w
  std::string moduleC = R"(
module @jit_moduleC {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<3.0> : tensor<f32>
    %0 = stablehlo.subtract %arg0, %c : tensor<f32>
    return %0 : tensor<f32>
  }
}
)";

  // Step 1: Merge A + B (A->B dependency)
  auto modA_inputs = createMockAddresses({100});   // x
  auto modA_outputs = createMockAddresses({200});  // y
  auto modB_inputs = createMockAddresses({200});   // y (from A)
  auto modB_outputs = createMockAddresses({300});  // z

  // Create MLIR context and load dialects for step 1
  mlir::MLIRContext contextAB;
  contextAB.getOrLoadDialect<mlir::func::FuncDialect>();
  contextAB.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  // Parse modules for step 1
  auto modA = mlir::parseSourceString<mlir::ModuleOp>(moduleA, &contextAB);
  auto modB = mlir::parseSourceString<mlir::ModuleOp>(moduleB, &contextAB);

  ASSERT_TRUE(modA) << "Module A should parse successfully";
  ASSERT_TRUE(modB) << "Module B should parse successfully";

  // Analyze dependencies for step 1
  auto analysisAB =
      torch_neuronx::analyzeDependencies(modA_inputs, modA_outputs, modB_inputs, modB_outputs);

  EXPECT_EQ(analysisAB.scenario, torch_neuronx::DependencyAnalysis::DIRECT_DEPS);

  // Call mergeModulesSkipIntermediates directly for step 1
  torch_neuronx::MergeMapping mappingAB;
  auto mergedAB = torch_neuronx::mergeModulesSkipIntermediates(
      *modA, *modB, &contextAB, modA_inputs, modA_outputs, modB_inputs, modB_outputs, analysisAB,
      &mappingAB, /*verify_output=*/true, /*run_optimization=*/false);

  ASSERT_TRUE(mergedAB) << "Merge A+B should succeed";

  auto mainAB = mergedAB->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(mainAB) << "Merged A+B should have main function";

  // A+B skip intermediates merge: input x, output z (y is intermediate and excluded)
  EXPECT_EQ(mainAB.getNumArguments(), 1) << "A+B should have 1 input (x)";
  EXPECT_EQ(mainAB.getNumResults(), 1) << "A+B should have 1 output (z) - y excluded";

  // Step 2: Merge (A+B) + C ((A+B)->C dependency)
  std::string mergedAB_str = torch_neuronx::stablehlo_utils::moduleToString(*mergedAB);

  auto mergedAB_inputs = createMockAddresses({100});   // x
  auto mergedAB_outputs = createMockAddresses({300});  // z
  auto modC_inputs = createMockAddresses({300});       // z (from A+B)
  auto modC_outputs = createMockAddresses({400});      // w

  // Create MLIR context and load dialects for step 2
  mlir::MLIRContext contextFinal;
  contextFinal.getOrLoadDialect<mlir::func::FuncDialect>();
  contextFinal.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  // Parse modules for step 2
  auto mergedAB_module = mlir::parseSourceString<mlir::ModuleOp>(mergedAB_str, &contextFinal);
  auto modC = mlir::parseSourceString<mlir::ModuleOp>(moduleC, &contextFinal);

  ASSERT_TRUE(mergedAB_module) << "Merged A+B module should parse successfully";
  ASSERT_TRUE(modC) << "Module C should parse successfully";

  // Analyze dependencies for step 2
  auto analysisFinal = torch_neuronx::analyzeDependencies(mergedAB_inputs, mergedAB_outputs,
                                                          modC_inputs, modC_outputs);

  EXPECT_EQ(analysisFinal.scenario, torch_neuronx::DependencyAnalysis::DIRECT_DEPS);

  // Call mergeModulesSkipIntermediates directly for step 2
  torch_neuronx::MergeMapping mappingFinal;
  auto finalMerged = torch_neuronx::mergeModulesSkipIntermediates(
      *mergedAB_module, *modC, &contextFinal, mergedAB_inputs, mergedAB_outputs, modC_inputs,
      modC_outputs, analysisFinal, &mappingFinal, /*verify_output=*/true,
      /*run_optimization=*/false);

  ASSERT_TRUE(finalMerged) << "Final merge should succeed";

  auto mainFinal = finalMerged->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(mainFinal) << "Final merged should have main function";

  // Final skip intermediates merge: input x, output w (z is intermediate and excluded)
  EXPECT_EQ(mainFinal.getNumArguments(), 1) << "Final should have 1 input (x)";
  EXPECT_EQ(mainFinal.getNumResults(), 1) << "Final should have 1 output (w) - z excluded";

  // Verify all operations are present
  std::string final_result = torch_neuronx::stablehlo_utils::moduleToString(*finalMerged);
  EXPECT_TRUE(final_result.find("stablehlo.add") != std::string::npos)
      << "Should contain add from moduleA";
  EXPECT_TRUE(final_result.find("stablehlo.multiply") != std::string::npos)
      << "Should contain multiply from moduleB";
  EXPECT_TRUE(final_result.find("stablehlo.subtract") != std::string::npos)
      << "Should contain subtract from moduleC";
}

// ============================================================================
// Section 4: Real-world Example Tests
// ============================================================================

// Test the transpose + matmul example with skip intermediates merge
TEST_F(StableHloUtilsSkipIntermediatesTest, SkipIntermediatesTransposeMatmulFusion) {
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

  // Create tensor addresses:
  // - Module1 (transpose): input=100, output=200
  // - Module2 (matmul): inputs=101,200, output=300
  auto mod1_inputs = createMockAddresses({100});   // transpose input
  auto mod1_outputs = createMockAddresses({200});  // transpose output
  auto mod2_inputs =
      createMockAddresses({101, 200});  // matmul inputs: first input (101) + transpose output (200)
  auto mod2_outputs = createMockAddresses({300});  // matmul output

  // Create MLIR context and load dialects
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  // Parse modules
  auto mod1 = mlir::parseSourceString<mlir::ModuleOp>(transpose_mlir, &context);
  auto mod2 = mlir::parseSourceString<mlir::ModuleOp>(matmul_mlir, &context);

  ASSERT_TRUE(mod1) << "Module 1 should parse successfully";
  ASSERT_TRUE(mod2) << "Module 2 should parse successfully";

  // Analyze dependencies
  auto analysis =
      torch_neuronx::analyzeDependencies(mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);

  EXPECT_EQ(analysis.scenario, torch_neuronx::DependencyAnalysis::DIRECT_DEPS);

  // Call mergeModulesSkipIntermediates directly
  torch_neuronx::MergeMapping mapping;
  auto skip_intermediates_merged = torch_neuronx::mergeModulesSkipIntermediates(
      *mod1, *mod2, &context, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs, analysis,
      &mapping, /*verify_output=*/true, /*run_optimization=*/false);

  ASSERT_TRUE(skip_intermediates_merged)
      << "SkipIntermediates transpose+matmul merge should succeed";

  auto main_func = skip_intermediates_merged->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func) << "Merged module should have main function";

  // KEY DIFFERENCE from conservative version:
  // Conservative version: 2 inputs, 2 outputs (transpose input, matmul first input) -> (transpose
  // output, matmul output) SkipIntermediates version: 2 inputs, 1 output (transpose input, matmul
  // first input) -> (matmul output only) The transpose output is excluded as it's intermediate
  EXPECT_EQ(main_func.getNumArguments(), 2) << "Should have 2 inputs";
  EXPECT_EQ(main_func.getNumResults(), 1) << "Should have 1 output (matmul result only)";

  // Verify operations are present
  std::string result = torch_neuronx::stablehlo_utils::moduleToString(*skip_intermediates_merged);
  EXPECT_TRUE(result.find("stablehlo.transpose") != std::string::npos)
      << "Should contain transpose operation";
  EXPECT_TRUE(result.find("stablehlo.dot_general") != std::string::npos)
      << "Should contain dot_general operation";

  // Check input types
  if (main_func.getNumArguments() >= 2) {
    auto input0_type = main_func.getArgumentTypes()[0];
    auto input1_type = main_func.getArgumentTypes()[1];

    EXPECT_TRUE(llvm::isa<mlir::RankedTensorType>(input0_type))
        << "Input 0 should be ranked tensor";
    EXPECT_TRUE(llvm::isa<mlir::RankedTensorType>(input1_type))
        << "Input 1 should be ranked tensor";

    if (llvm::isa<mlir::RankedTensorType>(input0_type) &&
        llvm::isa<mlir::RankedTensorType>(input1_type)) {
      auto tensor0 = llvm::cast<mlir::RankedTensorType>(input0_type);
      auto tensor1 = llvm::cast<mlir::RankedTensorType>(input1_type);

      // Input 0: tensor<8192x8192xbf16> (transpose input)
      // Input 1: tensor<2048x8192xbf16> (matmul first input)
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

  // Check output type - should only have matmul result
  if (main_func.getNumResults() >= 1) {
    auto output_type = main_func.getResultTypes()[0];
    EXPECT_TRUE(llvm::isa<mlir::RankedTensorType>(output_type)) << "Output should be ranked tensor";

    if (llvm::isa<mlir::RankedTensorType>(output_type)) {
      auto tensor_out = llvm::cast<mlir::RankedTensorType>(output_type);
      if (tensor_out.getShape().size() == 2) {
        EXPECT_EQ(tensor_out.getShape()[0], 2048) << "Output dim 0 should be 2048";
        EXPECT_EQ(tensor_out.getShape()[1], 8192) << "Output dim 1 should be 8192";
      }
    }
  }
}

// Test Scenario 5: Module2->Module1 dependency (module2 executes first)
TEST_F(StableHloUtilsSkipIntermediatesTest, SkipIntermediatesMergeModule2ToModule1Dependency) {
  // Create mock tensor addresses where module2 output becomes module1 input
  // This creates analysis.module1First = false, which was missing from other tests
  auto mod1_inputs = createMockAddresses({200, 101});  // module1 inputs: c (from mod2), b
  auto mod1_outputs = createMockAddresses({300});      // module1 output: d = c + b
  auto mod2_inputs = createMockAddresses({100});       // module2 input: a
  auto mod2_outputs = createMockAddresses({200});      // module2 output: c = a * 2

  // Create MLIR context and load dialects
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  // Parse modules
  auto mod1 = mlir::parseSourceString<mlir::ModuleOp>(mlir1_two_input, &context);
  auto mod2 = mlir::parseSourceString<mlir::ModuleOp>(mlir2_independent, &context);

  ASSERT_TRUE(mod1) << "Module 1 should parse successfully";
  ASSERT_TRUE(mod2) << "Module 2 should parse successfully";

  // Analyze dependencies
  auto analysis =
      torch_neuronx::analyzeDependencies(mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);

  EXPECT_EQ(analysis.scenario, torch_neuronx::DependencyAnalysis::DIRECT_DEPS);
  EXPECT_FALSE(analysis.module1First) << "Module2 should execute first due to dependency";

  // Call mergeModulesSkipIntermediates directly
  torch_neuronx::MergeMapping mapping;
  auto skip_intermediates_merged = torch_neuronx::mergeModulesSkipIntermediates(
      *mod1, *mod2, &context, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs, analysis,
      &mapping, /*verify_output=*/true, /*run_optimization=*/false);

  ASSERT_TRUE(skip_intermediates_merged)
      << "SkipIntermediates merge should succeed for module2->module1 dependency";

  auto main_func = skip_intermediates_merged->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func) << "Merged module should have main function";

  // Expected behavior:
  // - Inputs: a, b (2 inputs) - 'c' is intermediate
  // - Outputs: d (1 output) - 'c' is excluded as intermediate
  EXPECT_EQ(main_func.getNumArguments(), 2) << "Should have 2 inputs (a, b)";
  EXPECT_EQ(main_func.getNumResults(), 1)
      << "Should have 1 output (d) - 'c' excluded as intermediate";

  // Verify the merged MLIR contains the expected operations
  std::string result = torch_neuronx::stablehlo_utils::moduleToString(*skip_intermediates_merged);
  EXPECT_TRUE(result.find("stablehlo.add") != std::string::npos)
      << "Should contain add operation from module1";
  EXPECT_TRUE(result.find("stablehlo.multiply") != std::string::npos)
      << "Should contain multiply operation from module2";
  EXPECT_TRUE(result.find("submain1") != std::string::npos) << "Should contain submain1 function";
  EXPECT_TRUE(result.find("submain2") != std::string::npos) << "Should contain submain2 function";

  // This test would have failed with the original bug because the function calls
  // would be in the wrong order when analysis.module1First = false
}

// ============================================================================
// Section 5: Edge Cases and Error Handling
// ============================================================================

// Test skip intermediates merge with circular dependencies (should still throw)
TEST_F(StableHloUtilsSkipIntermediatesTest, SkipIntermediatesCircularDependenciesThrowsException) {
  // Create mock tensor addresses with circular dependencies:
  // module1 output -> module2 input AND module2 output -> module1 input
  auto mod1_inputs = createMockAddresses({100, 300});  // module1 inputs: a, d (d from mod2)
  auto mod1_outputs = createMockAddresses({200});      // module1 output: b = a + d
  auto mod2_inputs = createMockAddresses({200, 101});  // module2 inputs: b (from mod1), c
  auto mod2_outputs = createMockAddresses({300});      // module2 output: d = b * c

  // This should throw std::runtime_error due to circular dependencies
  EXPECT_THROW(
      {
        // Create MLIR context and load dialects
        mlir::MLIRContext context;
        context.getOrLoadDialect<mlir::func::FuncDialect>();
        context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

        // Parse modules
        auto mod1 = mlir::parseSourceString<mlir::ModuleOp>(mlir1_two_input, &context);
        auto mod2 = mlir::parseSourceString<mlir::ModuleOp>(mlir2_two_input, &context);

        ASSERT_TRUE(mod1) << "Module 1 should parse successfully";
        ASSERT_TRUE(mod2) << "Module 2 should parse successfully";

        // Analyze dependencies - this should throw due to circular dependencies
        auto analysis = torch_neuronx::analyzeDependencies(mod1_inputs, mod1_outputs, mod2_inputs,
                                                           mod2_outputs);
      },
      std::runtime_error);
}

// Test skip intermediates merge with invalid modules
TEST_F(StableHloUtilsSkipIntermediatesTest, SkipIntermediatesMergeWithInvalidModule) {
  std::string invalid_mlir = "invalid mlir syntax";
  auto mod1_inputs = createMockAddresses({100});
  auto mod1_outputs = createMockAddresses({200});
  auto mod2_inputs = createMockAddresses({300});
  auto mod2_outputs = createMockAddresses({400});

  // Create MLIR context and load dialects
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  // Parse modules - invalid module should fail to parse
  auto mod1 = mlir::parseSourceString<mlir::ModuleOp>(mlir1_independent, &context);
  auto mod2 = mlir::parseSourceString<mlir::ModuleOp>(invalid_mlir, &context);

  ASSERT_TRUE(mod1) << "Module 1 should parse successfully";
  EXPECT_FALSE(mod2) << "Invalid module should fail to parse";
}

// Test skip intermediates merge with mismatched tensor address counts
TEST_F(StableHloUtilsSkipIntermediatesTest, SkipIntermediatesMismatchedTensorAddressCounts) {
  // Too few input addresses (1 address for 2-input function)
  auto too_few_inputs = createMockAddresses({100});  // 1 address for mlir1_two_input (expects 2)
  auto normal_outputs = createMockAddresses({200});
  auto normal_inputs = createMockAddresses({300, 301});
  auto normal_outputs2 = createMockAddresses({400});

  EXPECT_THROW(
      {
        // Create MLIR context and load dialects
        mlir::MLIRContext context;
        context.getOrLoadDialect<mlir::func::FuncDialect>();
        context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

        // Parse modules
        auto mod1 = mlir::parseSourceString<mlir::ModuleOp>(mlir1_two_input, &context);
        auto mod2 = mlir::parseSourceString<mlir::ModuleOp>(mlir2_two_input, &context);

        ASSERT_TRUE(mod1) << "Module 1 should parse successfully";
        ASSERT_TRUE(mod2) << "Module 2 should parse successfully";

        // Call mergeModulesSkipIntermediates directly - this should throw due to mismatched tensor
        // counts The validation happens in mergeModules/mergeModulesSkipIntermediates, not in
        // analyzeDependencies
        auto analysis = torch_neuronx::analyzeDependencies(too_few_inputs, normal_outputs,
                                                           normal_inputs, normal_outputs2);

        torch_neuronx::MergeMapping mapping;
        auto skip_intermediates_merged = torch_neuronx::mergeModulesSkipIntermediates(
            *mod1, *mod2, &context, too_few_inputs, normal_outputs, normal_inputs, normal_outputs2,
            analysis, &mapping, /*verify_output=*/true, /*run_optimization=*/false);
      },
      std::runtime_error);
}

// ============================================================================
// Section 6: Mapping and Metadata Tests
// ============================================================================

// Test that skip intermediates merge produces correct mapping information
TEST_F(StableHloUtilsSkipIntermediatesTest, SkipIntermediatesMergeMappingInformation) {
  auto mod1_inputs = createMockAddresses({100, 101});  // module1 inputs: a, b
  auto mod1_outputs = createMockAddresses({200});      // module1 output: c = a + b
  auto mod2_inputs = createMockAddresses({200, 102});  // module2 inputs: c (from mod1), d
  auto mod2_outputs = createMockAddresses({300});      // module2 output: e = c * d

  // Create MLIR context and load dialects
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  // Parse modules
  auto mod1 = mlir::parseSourceString<mlir::ModuleOp>(mlir1_two_input, &context);
  auto mod2 = mlir::parseSourceString<mlir::ModuleOp>(mlir2_two_input, &context);

  ASSERT_TRUE(mod1) << "Module 1 should parse successfully";
  ASSERT_TRUE(mod2) << "Module 2 should parse successfully";

  // Analyze dependencies
  auto analysis =
      torch_neuronx::analyzeDependencies(mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);

  EXPECT_EQ(analysis.scenario, torch_neuronx::DependencyAnalysis::DIRECT_DEPS);

  // Call mergeModulesSkipIntermediates directly
  torch_neuronx::MergeMapping mapping;
  auto skip_intermediates_merged = torch_neuronx::mergeModulesSkipIntermediates(
      *mod1, *mod2, &context, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs, analysis,
      &mapping, /*verify_output=*/true, /*run_optimization=*/false);

  ASSERT_TRUE(skip_intermediates_merged) << "SkipIntermediates merge should succeed";

  // Verify mapping information
  EXPECT_EQ(mapping.total_inputs, 3) << "Should have 3 total inputs";
  EXPECT_EQ(mapping.total_outputs, 1)
      << "Should have 1 total output (skip intermediates excludes intermediate)";

  // Check input mapping
  EXPECT_EQ(mapping.input_mapping.size(), 3) << "Should have 3 input mappings";

  // Check output mapping - should only have 1 output (the final one)
  EXPECT_EQ(mapping.output_mapping.size(), 1)
      << "Should have 1 output mapping (intermediate excluded)";
}

// Test skip intermediates merge mapping for mixed scenario
TEST_F(StableHloUtilsSkipIntermediatesTest, SkipIntermediatesMergeMappingMixed) {
  auto mod1_inputs = createMockAddresses({100, 101});  // module1 inputs: a, b
  auto mod1_outputs = createMockAddresses({200});      // module1 output: c = a + b
  auto mod2_inputs = createMockAddresses({100, 200});  // module2 inputs: a (shared), c (dependency)
  auto mod2_outputs = createMockAddresses({300});      // module2 output: e = a * c

  // Create MLIR context and load dialects
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  // Parse modules
  auto mod1 = mlir::parseSourceString<mlir::ModuleOp>(mlir1_two_input, &context);
  auto mod2 = mlir::parseSourceString<mlir::ModuleOp>(mlir2_two_input, &context);

  ASSERT_TRUE(mod1) << "Module 1 should parse successfully";
  ASSERT_TRUE(mod2) << "Module 2 should parse successfully";

  // Analyze dependencies
  auto analysis =
      torch_neuronx::analyzeDependencies(mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);

  EXPECT_EQ(analysis.scenario, torch_neuronx::DependencyAnalysis::MIXED);

  // Call mergeModulesSkipIntermediates directly
  torch_neuronx::MergeMapping mapping;
  auto skip_intermediates_merged = torch_neuronx::mergeModulesSkipIntermediates(
      *mod1, *mod2, &context, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs, analysis,
      &mapping, /*verify_output=*/true, /*run_optimization=*/false);

  ASSERT_TRUE(skip_intermediates_merged) << "SkipIntermediates merge should succeed";

  // Verify mapping information for mixed scenario
  EXPECT_EQ(mapping.total_inputs, 2) << "Should have 2 total inputs";
  EXPECT_EQ(mapping.total_outputs, 1)
      << "Should have 1 total output (skip intermediates excludes intermediate)";

  // Check input mapping
  EXPECT_EQ(mapping.input_mapping.size(), 2) << "Should have 2 input mappings";

  // Check output mapping - should only have 1 output (the final one)
  EXPECT_EQ(mapping.output_mapping.size(), 1)
      << "Should have 1 output mapping (intermediate excluded)";
}

// ============================================================================
// Section 7: Performance and Optimization Tests
// ============================================================================

// Test that skip intermediates merge produces smaller output signatures
TEST_F(StableHloUtilsSkipIntermediatesTest, SkipIntermediatesProducesFewerOutputs) {
  // Test multiple scenarios to ensure skip intermediates consistently produces fewer outputs when
  // there are dependencies

  // Scenario 1: Direct dependencies
  auto mod1_inputs = createMockAddresses({100, 101});
  auto mod1_outputs = createMockAddresses({200});
  auto mod2_inputs = createMockAddresses({200, 102});
  auto mod2_outputs = createMockAddresses({300});

  // Create conservative merged module
  std::string conservative_result = torch_neuronx::mergeStableHLOModules(
      mlir1_two_input, mlir2_two_input, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs,
      /*verify_output=*/true, /*run_optimization=*/false);

  // Create MLIR context and load dialects
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  // Parse modules for skip intermediates merge
  auto mod1 = mlir::parseSourceString<mlir::ModuleOp>(mlir1_two_input, &context);
  auto mod2 = mlir::parseSourceString<mlir::ModuleOp>(mlir2_two_input, &context);

  ASSERT_TRUE(mod1) << "Module 1 should parse successfully";
  ASSERT_TRUE(mod2) << "Module 2 should parse successfully";

  // Analyze dependencies
  auto analysis =
      torch_neuronx::analyzeDependencies(mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);

  EXPECT_EQ(analysis.scenario, torch_neuronx::DependencyAnalysis::DIRECT_DEPS);

  // Call mergeModulesSkipIntermediates directly
  torch_neuronx::MergeMapping mapping;
  auto skip_intermediates_merged = torch_neuronx::mergeModulesSkipIntermediates(
      *mod1, *mod2, &context, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs, analysis,
      &mapping, /*verify_output=*/true, /*run_optimization=*/false);

  ASSERT_TRUE(skip_intermediates_merged);
  ASSERT_FALSE(conservative_result.empty());

  auto conservative_module = mlir::parseSourceString<mlir::ModuleOp>(conservative_result, &context);
  ASSERT_TRUE(conservative_module);

  auto conservative_main = conservative_module->lookupSymbol<mlir::func::FuncOp>("main");
  auto skip_intermediates_main =
      skip_intermediates_merged->lookupSymbol<mlir::func::FuncOp>("main");

  ASSERT_TRUE(conservative_main);
  ASSERT_TRUE(skip_intermediates_main);

  // SkipIntermediates should have fewer outputs
  EXPECT_LT(skip_intermediates_main.getNumResults(), conservative_main.getNumResults())
      << "SkipIntermediates version should have fewer outputs than conservative version";
}

// Test that skip intermediates merge still contains all necessary operations
TEST_F(StableHloUtilsSkipIntermediatesTest, SkipIntermediatesContainsAllOperations) {
  auto mod1_inputs = createMockAddresses({100, 101});
  auto mod1_outputs = createMockAddresses({200});
  auto mod2_inputs = createMockAddresses({200, 102});
  auto mod2_outputs = createMockAddresses({300});

  // Create MLIR context and load dialects
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  // Parse modules
  auto mod1 = mlir::parseSourceString<mlir::ModuleOp>(mlir1_two_input, &context);
  auto mod2 = mlir::parseSourceString<mlir::ModuleOp>(mlir2_two_input, &context);

  ASSERT_TRUE(mod1) << "Module 1 should parse successfully";
  ASSERT_TRUE(mod2) << "Module 2 should parse successfully";

  // Analyze dependencies
  auto analysis =
      torch_neuronx::analyzeDependencies(mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs);

  EXPECT_EQ(analysis.scenario, torch_neuronx::DependencyAnalysis::DIRECT_DEPS);

  // Call mergeModulesSkipIntermediates directly
  torch_neuronx::MergeMapping mapping;
  auto skip_intermediates_merged = torch_neuronx::mergeModulesSkipIntermediates(
      *mod1, *mod2, &context, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs, analysis,
      &mapping, /*verify_output=*/true, /*run_optimization=*/false);

  ASSERT_TRUE(skip_intermediates_merged);

  std::string result = torch_neuronx::stablehlo_utils::moduleToString(*skip_intermediates_merged);

  // Should contain all the original operations
  EXPECT_TRUE(result.find("stablehlo.add") != std::string::npos)
      << "Should contain add operation from module1";
  EXPECT_TRUE(result.find("stablehlo.multiply") != std::string::npos)
      << "Should contain multiply operation from module2";

  // Should contain submain functions
  EXPECT_TRUE(result.find("submain1") != std::string::npos) << "Should contain submain1 function";
  EXPECT_TRUE(result.find("submain2") != std::string::npos) << "Should contain submain2 function";

  // Should contain function calls in main
  EXPECT_TRUE(result.find("call @submain") != std::string::npos)
      << "Should contain function calls for orchestration";
}

// ============================================================================
// Section 8: Skip Intermediates Mapping Verification Tests
// ============================================================================

// Test mapping correctness for Module1->Module2 dependency with skip intermediates
TEST_F(StableHloUtilsSkipIntermediatesTest,
       MappingTestSkipIntermediatesModule1ToModule2Dependency) {
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
  auto merged = torch_neuronx::mergeModulesSkipIntermediates(
      *mod1, *mod2, &context, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs, analysis,
      &mapping, /*verify_output=*/true, /*run_optimization=*/false);
  ASSERT_TRUE(merged);

  // Verify mapping: inputs (a, b, d), outputs (e) - 'c' excluded as intermediate
  EXPECT_EQ(mapping.total_inputs, 3);
  EXPECT_EQ(mapping.total_outputs, 1);

  // Input mapping: merged[0]=module1[0], merged[1]=module1[1], merged[2]=module2[1]
  EXPECT_EQ(mapping.input_mapping[0], std::make_pair(1, size_t(0)))
      << "Input 0 should be module1 input 0";
  EXPECT_EQ(mapping.input_mapping[1], std::make_pair(1, size_t(1)))
      << "Input 1 should be module1 input 1";
  EXPECT_EQ(mapping.input_mapping[2], std::make_pair(2, size_t(1)))
      << "Input 2 should be module2 input 1";

  // Output mapping: merged[0]=module2[0] - 'c' excluded as intermediate
  EXPECT_EQ(mapping.output_mapping[0], std::make_pair(2, size_t(0)))
      << "Output 0 should be module2 output 0";
}

// Test mapping correctness for Module2->Module1 dependency with skip intermediates (CRITICAL TEST)
TEST_F(StableHloUtilsSkipIntermediatesTest,
       MappingTestSkipIntermediatesModule2ToModule1Dependency) {
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
  auto merged = torch_neuronx::mergeModulesSkipIntermediates(
      *mod1, *mod2, &context, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs, analysis,
      &mapping, /*verify_output=*/true, /*run_optimization=*/false);
  ASSERT_TRUE(merged);

  // Verify mapping: inputs (a, c), outputs (d) - 'b' excluded as intermediate
  EXPECT_EQ(mapping.total_inputs, 2);
  EXPECT_EQ(mapping.total_outputs, 1);

  // Input mapping: merged[0]=module2[0], merged[1]=module1[1]
  EXPECT_EQ(mapping.input_mapping[0], std::make_pair(2, size_t(0)))
      << "Input 0 should be module2 input 0";
  EXPECT_EQ(mapping.input_mapping[1], std::make_pair(1, size_t(1)))
      << "Input 1 should be module1 input 1";

  // Output mapping: merged[0]=module1[0] - 'b' excluded as intermediate
  EXPECT_EQ(mapping.output_mapping[0], std::make_pair(1, size_t(0)))
      << "Output 0 should be module1 output 0";
}

// Test mapping correctness for common inputs scenario with skip intermediates
TEST_F(StableHloUtilsSkipIntermediatesTest, MappingTestSkipIntermediatesCommonInputs) {
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
  auto merged = torch_neuronx::mergeModulesSkipIntermediates(
      *mod1, *mod2, &context, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs, analysis,
      &mapping, /*verify_output=*/true, /*run_optimization=*/false);
  ASSERT_TRUE(merged);

  // Verify mapping: inputs (a, b, d), outputs (c, e) - no intermediates to skip
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

// Test mapping correctness for mixed scenario with skip intermediates
TEST_F(StableHloUtilsSkipIntermediatesTest, MappingTestSkipIntermediatesMixedScenario) {
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
  auto merged = torch_neuronx::mergeModulesSkipIntermediates(
      *mod1, *mod2, &context, mod1_inputs, mod1_outputs, mod2_inputs, mod2_outputs, analysis,
      &mapping, /*verify_output=*/true, /*run_optimization=*/false);
  ASSERT_TRUE(merged);

  // Verify mapping: inputs (a, b), outputs (e) - 'c' excluded as intermediate
  EXPECT_EQ(mapping.total_inputs, 2);
  EXPECT_EQ(mapping.total_outputs, 1);

  // Input mapping: merged[0]=module1[0], merged[1]=module1[1]
  EXPECT_EQ(mapping.input_mapping[0], std::make_pair(1, size_t(0)))
      << "Input 0 should be module1 input 0 (shared)";
  EXPECT_EQ(mapping.input_mapping[1], std::make_pair(1, size_t(1)))
      << "Input 1 should be module1 input 1";

  // Output mapping: merged[0]=module2[0] - 'c' excluded as intermediate
  EXPECT_EQ(mapping.output_mapping[0], std::make_pair(2, size_t(0)))
      << "Output 0 should be module2 output 0";
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
