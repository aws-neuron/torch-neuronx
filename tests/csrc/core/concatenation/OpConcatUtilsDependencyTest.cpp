#include <gtest/gtest.h>

#include "torch_neuronx/csrc/core/concatenation/OpConcatUtils.h"

namespace torch_neuronx {
namespace test {

/**
 * @file OpConcatUtilsDependencyTest.cpp
 * @brief Tests for shared storage address handling in dependency analysis
 *
 * These tests verify that in-place operations and tensor views sharing storage addresses
 * are correctly classified as dependencies rather than common inputs, ensuring proper
 * module concatenation behavior.
 *
 * Dependency detection takes precedence over common input detection to prevent false
 * positives when tensor addresses are shared due to views or in-place operations.
 */

/**
 * @brief Verifies that shared storage addresses are correctly handled as dependencies
 *
 * This test reproduces the scenario from RoPE contiguous handling where:
 * - Module1 (prologue): Takes base tensor [1, 2048, 1, 256] at address 0x1000
 *                       Outputs sliced view [1, 2048, 1, 128] at same address 0x1000
 * - Module2 (operation): Takes sliced view [1, 2048, 1, 128] at address 0x1000
 *                        Produces output at different address
 *
 * Key behavior: Dependencies are detected first and dependency targets are excluded
 * from common input detection, correctly classifying this as DIRECT_DEPS scenario.
 *
 * This prevents type mismatches that would occur if module2 input[0] were incorrectly
 * detected as both a common input and a dependency, which would cause errors like:
 * "Type mismatch in common input: module1 input 0 (tensor<1x2048x1x256xbf16>)
 *  does not match module2 input 0 (tensor<1x2048x1x128xbf16>)"
 */
TEST(OpConcatUtilsDependencyTest, SharedStorageAddressShouldNotBeCommonInput) {
  // Simulate shared storage address (e.g., base tensor and its view)
  void* shared_address = reinterpret_cast<void*>(0x1000);

  // Module1 (prologue): Takes base tensor, outputs sliced view
  // Input: base tensor at shared_address
  // Output: sliced view (also at shared_address due to shared storage)
  std::vector<void*> module1_input_addrs = {shared_address};
  std::vector<void*> module1_output_addrs = {shared_address};

  // Module2 (operation): Takes sliced view, produces output
  // Input: sliced view at shared_address (this is a dependency, NOT a common input)
  // Output: operation result at different address
  std::vector<void*> module2_input_addrs = {shared_address};
  std::vector<void*> module2_output_addrs = {reinterpret_cast<void*>(0x2000)};

  // Analyze dependencies
  auto analysis = analyzeDependencies(module1_input_addrs, module1_output_addrs,
                                      module2_input_addrs, module2_output_addrs);

  // Verify correct dependency detection
  EXPECT_EQ(analysis.module1ToModule2Deps.size(), 1)
      << "Should detect one dependency: module1 output[0] -> module2 input[0]";
  ASSERT_EQ(analysis.module1ToModule2Deps[0].size(), 1)
      << "Output 0 should map to exactly one input";
  EXPECT_EQ(analysis.module1ToModule2Deps[0][0], 0)
      << "Dependency should map output index 0 to input index 0";

  // Verify NO false common input detection
  EXPECT_EQ(analysis.commonInputs.size(), 0)
      << "Module2 input[0] should NOT be detected as a common input when it's a dependency "
         "target. "
      << "Incorrect detection would cause type mismatch errors because the base tensor and "
         "sliced view "
      << "have different shapes but share the same storage address.";

  // Verify correct scenario classification
  EXPECT_EQ(analysis.scenario, DependencyAnalysis::DIRECT_DEPS)
      << "Expected DIRECT_DEPS scenario (not MIXED) since there are dependencies but no true "
         "common inputs";

  // Verify correct execution order
  EXPECT_TRUE(analysis.module1First)
      << "Module1 should execute first since module2 depends on module1's output";
}

/**
 * @brief Verifies that legitimate common inputs are correctly detected
 *
 * This test ensures that normal common input detection works correctly when inputs
 * are truly shared between modules without any dependency relationship.
 */
TEST(OpConcatUtilsDependencyTest, TrueCommonInputsAreDetected) {
  void* common_address = reinterpret_cast<void*>(0x1000);
  void* unique_address1 = reinterpret_cast<void*>(0x2000);
  void* unique_address2 = reinterpret_cast<void*>(0x3000);
  void* output_address1 = reinterpret_cast<void*>(0x4000);
  void* output_address2 = reinterpret_cast<void*>(0x5000);

  // Module1: Takes (common, unique1), outputs result1
  std::vector<void*> module1_input_addrs = {common_address, unique_address1};
  std::vector<void*> module1_output_addrs = {output_address1};

  // Module2: Takes (common, unique2), outputs result2
  // The common input is NOT a dependency - it's truly shared between modules
  std::vector<void*> module2_input_addrs = {common_address, unique_address2};
  std::vector<void*> module2_output_addrs = {output_address2};

  auto analysis = analyzeDependencies(module1_input_addrs, module1_output_addrs,
                                      module2_input_addrs, module2_output_addrs);

  // Verify common input is detected
  EXPECT_EQ(analysis.commonInputs.size(), 1)
      << "Should detect one common input shared by both modules";
  EXPECT_EQ(analysis.commonInputs[0], 0)
      << "Common input should map module1 input[0] to module2 input[0]";

  // Verify no false dependency detection
  EXPECT_EQ(analysis.module1ToModule2Deps.size(), 0)
      << "Should not detect any module1->module2 dependencies";
  EXPECT_EQ(analysis.module2ToModule1Deps.size(), 0)
      << "Should not detect any module2->module1 dependencies";

  // Verify correct scenario classification
  EXPECT_EQ(analysis.scenario, DependencyAnalysis::COMMON_INPUTS)
      << "Expected COMMON_INPUTS scenario since modules share inputs but have no dependencies";
}

/**
 * @brief Verifies correct handling of mixed scenario with distinct common inputs and dependencies
 *
 * This test verifies that when a module legitimately has BOTH common inputs AND dependencies,
 * but they are different inputs (not the same input misidentified), the analysis correctly
 * identifies each type.
 */
TEST(OpConcatUtilsDependencyTest, MixedScenarioWithDistinctCommonAndDependency) {
  void* common_address = reinterpret_cast<void*>(0x1000);
  void* unique_address = reinterpret_cast<void*>(0x2000);
  void* intermediate_address = reinterpret_cast<void*>(0x3000);
  void* output_address = reinterpret_cast<void*>(0x4000);

  // Module1: Takes (common, unique), outputs intermediate
  std::vector<void*> module1_input_addrs = {common_address, unique_address};
  std::vector<void*> module1_output_addrs = {intermediate_address};

  // Module2: Takes (common, intermediate), outputs result
  // - First input (common) is a true common input shared with module1
  // - Second input (intermediate) is a dependency from module1's output
  std::vector<void*> module2_input_addrs = {common_address, intermediate_address};
  std::vector<void*> module2_output_addrs = {output_address};

  auto analysis = analyzeDependencies(module1_input_addrs, module1_output_addrs,
                                      module2_input_addrs, module2_output_addrs);

  // Verify common input detection
  EXPECT_EQ(analysis.commonInputs.size(), 1)
      << "Should detect one common input (module1 input[0] == module2 input[0])";
  EXPECT_EQ(analysis.commonInputs[0], 0)
      << "Common input should map module1 input[0] to module2 input[0]";

  // Verify dependency detection
  EXPECT_EQ(analysis.module1ToModule2Deps.size(), 1)
      << "Should detect one dependency (module1 output feeds module2 input)";
  ASSERT_EQ(analysis.module1ToModule2Deps[0].size(), 1)
      << "Output 0 should map to exactly one input";
  EXPECT_EQ(analysis.module1ToModule2Deps[0][0], 1)
      << "Dependency should map module1 output[0] to module2 input[1]";

  // Verify correct scenario classification
  EXPECT_EQ(analysis.scenario, DependencyAnalysis::MIXED)
      << "Expected MIXED scenario since we have both common inputs and dependencies";
}

}  // namespace test
}  // namespace torch_neuronx
