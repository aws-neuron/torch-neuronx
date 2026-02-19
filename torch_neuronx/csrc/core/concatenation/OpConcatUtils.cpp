#include "OpConcatUtils.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/opbuilder/utility/StableHloUtils.h"

namespace torch_neuronx {

/**
 * @brief Analyzes tensor address relationships between two modules
 *
 * @throws std::runtime_error if circular dependencies are detected between modules
 */
DependencyAnalysis analyzeDependencies(const std::vector<void*>& module1_input_addrs,
                                       const std::vector<void*>& module1_output_addrs,
                                       const std::vector<void*>& module2_input_addrs,
                                       const std::vector<void*>& module2_output_addrs) {
  TORCH_NEURONX_TRACE_FUNCTION_WITH_ARGS(
      "mod1_inputs:", module1_input_addrs.size(), "mod1_outputs:", module1_output_addrs.size(),
      "mod2_inputs:", module2_input_addrs.size(), "mod2_outputs:", module2_output_addrs.size());

  // Input validation
  if (module1_input_addrs.empty() && module1_output_addrs.empty()) {
    TORCH_NEURONX_ERROR("Module1 cannot have both empty inputs and outputs");
    throw std::invalid_argument("Module1 cannot have both empty inputs and outputs");
  }

  if (module2_input_addrs.empty() && module2_output_addrs.empty()) {
    TORCH_NEURONX_ERROR("Module2 cannot have both empty inputs and outputs");
    throw std::invalid_argument("Module2 cannot have both empty inputs and outputs");
  }

  // Check for null pointers in addresses
  for (size_t i = 0; i < module1_input_addrs.size(); ++i) {
    if (!module1_input_addrs[i]) {
      throw std::invalid_argument("Module1 input address at index " + std::to_string(i) +
                                  " is null");
    }
  }

  for (size_t i = 0; i < module2_input_addrs.size(); ++i) {
    if (!module2_input_addrs[i]) {
      throw std::invalid_argument("Module2 input address at index " + std::to_string(i) +
                                  " is null");
    }
  }

  DependencyAnalysis analysis;

  TORCH_NEURONX_DEBUG("analyzeDependencies: Starting dependency analysis");

  // IMPORTANT: Find dependencies FIRST before checking for common inputs
  // This ensures dependency targets are not misidentified as common inputs
  // (critical for shared storage addresses from in-place ops and views)

  // Find dependencies: module1 outputs -> module2 inputs
  // Support duplicate inputs (e.g., x + x) by storing all matching input indices
  for (size_t i = 0; i < module1_output_addrs.size(); ++i) {
    for (size_t j = 0; j < module2_input_addrs.size(); ++j) {
      if (module1_output_addrs[i] == module2_input_addrs[j]) {
        analysis.module1ToModule2Deps[i].push_back(j);
        TORCH_NEURONX_DEBUG("Found dependency: module1_output[", i, "] -> module2_input[", j, "]");
      }
    }
  }

  // Find dependencies: module2 outputs -> module1 inputs
  // Support duplicate inputs (e.g., x + x) by storing all matching input indices
  for (size_t i = 0; i < module2_output_addrs.size(); ++i) {
    for (size_t j = 0; j < module1_input_addrs.size(); ++j) {
      if (module2_output_addrs[i] == module1_input_addrs[j]) {
        analysis.module2ToModule1Deps[i].push_back(j);
        TORCH_NEURONX_DEBUG("Found dependency: module2_output[", i, "] -> module1_input[", j, "]");
      }
    }
  }

  // Track which inputs are dependency targets
  // These should NOT be considered common inputs even if addresses match
  std::unordered_set<size_t> module1_dependency_targets;
  for (const auto& dep : analysis.module2ToModule1Deps) {
    // Insert all input indices from the dependency vector
    for (size_t target_idx : dep.second) {
      module1_dependency_targets.insert(target_idx);
    }
  }

  std::unordered_set<size_t> module2_dependency_targets;
  for (const auto& dep : analysis.module1ToModule2Deps) {
    // Insert all input indices from the dependency vector
    for (size_t target_idx : dep.second) {
      module2_dependency_targets.insert(target_idx);
    }
  }

  // Find common inputs, EXCLUDING dependency targets
  // This prevents false common input detection when storage is shared
  // (e.g., base tensor and its view/slice share the same address)
  for (size_t i = 0; i < module1_input_addrs.size(); ++i) {
    // Skip if this input is a dependency target
    if (module1_dependency_targets.count(i) > 0) {
      TORCH_NEURONX_DEBUG("Skipping module1 input[", i, "] - already a dependency target");
      continue;
    }

    for (size_t j = 0; j < module2_input_addrs.size(); ++j) {
      // Skip if this input is a dependency target
      if (module2_dependency_targets.count(j) > 0) {
        TORCH_NEURONX_DEBUG("Skipping module2 input[", j, "] - already a dependency target");
        continue;
      }

      if (module1_input_addrs[i] == module2_input_addrs[j]) {
        analysis.commonInputs[i] = j;
        TORCH_NEURONX_DEBUG("Found common input: module1[", i, "] == module2[", j, "]");
      }
    }
  }

  // Check for circular dependencies
  if (!analysis.module1ToModule2Deps.empty() && !analysis.module2ToModule1Deps.empty()) {
    throw std::runtime_error("Circular dependencies detected between modules to be concatenated!");
  }

  // Determine execution order
  if (!analysis.module2ToModule1Deps.empty()) {
    analysis.module1First = false;  // module2 must execute first
    TORCH_NEURONX_DEBUG("Module2 will execute first due to dependencies");
  } else {
    analysis.module1First = true;  // module1 executes first (default)
    TORCH_NEURONX_DEBUG("Module1 will execute first");
  }

  // Determine scenario type
  bool hasCommonInputs = !analysis.commonInputs.empty();
  bool hasDependencies =
      !analysis.module1ToModule2Deps.empty() || !analysis.module2ToModule1Deps.empty();

  if (hasCommonInputs && hasDependencies) {
    analysis.scenario = DependencyAnalysis::MIXED;
    TORCH_NEURONX_DEBUG("Detected MIXED scenario (common inputs + dependencies)");
  } else if (hasDependencies) {
    analysis.scenario = DependencyAnalysis::DIRECT_DEPS;
    TORCH_NEURONX_DEBUG("Detected DIRECT_DEPS scenario");
  } else if (hasCommonInputs) {
    analysis.scenario = DependencyAnalysis::COMMON_INPUTS;
    TORCH_NEURONX_DEBUG("Detected COMMON_INPUTS scenario");
  } else {
    analysis.scenario = DependencyAnalysis::INDEPENDENT;
    TORCH_NEURONX_DEBUG("Detected INDEPENDENT scenario");
  }

  return analysis;
}

/**
 * @brief Validates dependency compatibility between modules based on dependency analysis
 *
 * Checks both type and shape compatibility for dependencies and common inputs.
 *
 * @param main1 First module's main function
 * @param main2 Second module's main function
 * @param analysis Dependency analysis results
 *
 * @throws std::runtime_error if type or shape mismatches are detected in dependencies or common
 * inputs
 */
static void validateDepCompatibility(mlir::func::FuncOp main1, mlir::func::FuncOp main2,
                                     const DependencyAnalysis& analysis) {
  TORCH_NEURONX_DEBUG("validateDepCompatibility: Starting dependency compatibility validation");

  // Helper function to convert MLIR type to string for error messages
  auto typeToString = [](mlir::Type type) -> std::string {
    std::string result;
    llvm::raw_string_ostream stream(result);
    type.print(stream);
    return result;
  };

  // Helper function to validate shape compatibility between two types
  auto validateShapeCompatibility = [&typeToString](mlir::Type type1, mlir::Type type2,
                                                    const std::string& context) {
    // Try to cast both types to RankedTensorType to access shape information
    auto tensorType1 = llvm::dyn_cast<mlir::RankedTensorType>(type1);
    auto tensorType2 = llvm::dyn_cast<mlir::RankedTensorType>(type2);

    // If both are ranked tensors, validate their shapes
    if (tensorType1 && tensorType2) {
      auto shape1 = tensorType1.getShape();
      auto shape2 = tensorType2.getShape();

      // Check if ranks match
      if (shape1.size() != shape2.size()) {
        throw std::runtime_error("Shape rank mismatch in " + context + ": shape1 has rank " +
                                 std::to_string(shape1.size()) + " (" + typeToString(type1) +
                                 ") but shape2 has rank " + std::to_string(shape2.size()) + " (" +
                                 typeToString(type2) + ")");
      }

      // Check if each dimension matches
      for (size_t dim = 0; dim < shape1.size(); ++dim) {
        if (shape1[dim] != shape2[dim]) {
          throw std::runtime_error(
              "Shape mismatch in " + context + " at dimension " + std::to_string(dim) +
              ": shape1[" + std::to_string(dim) + "]=" + std::to_string(shape1[dim]) +
              " but shape2[" + std::to_string(dim) + "]=" + std::to_string(shape2[dim]) +
              " (full types: " + typeToString(type1) + " vs " + typeToString(type2) + ")");
        }
      }
      TORCH_NEURONX_DEBUG("Shape validation passed for ", context);
    }
  };

  // Validate module1 -> module2 dependencies
  // Support duplicate inputs by iterating through all input indices for each output
  for (const auto& dep : analysis.module1ToModule2Deps) {
    size_t output_idx = dep.first;
    const std::vector<size_t>& input_indices = dep.second;

    if (output_idx >= main1.getNumResults()) {
      throw std::runtime_error("Invalid dependency: module1 output index " +
                               std::to_string(output_idx) + " out of range");
    }

    auto output_type = main1.getResultTypes()[output_idx];

    // Validate each input that depends on this output
    for (size_t input_idx : input_indices) {
      if (input_idx >= main2.getNumArguments()) {
        throw std::runtime_error("Invalid dependency: module2 input index " +
                                 std::to_string(input_idx) + " out of range");
      }

      auto input_type = main2.getArgumentTypes()[input_idx];

      if (output_type != input_type) {
        throw std::runtime_error("Type mismatch in dependency: module1 output " +
                                 std::to_string(output_idx) + " (" + typeToString(output_type) +
                                 ") cannot connect to module2 input " + std::to_string(input_idx) +
                                 " (expected " + typeToString(input_type) + ")");
      }

      // Validate shape compatibility
      validateShapeCompatibility(output_type, input_type,
                                 "dependency module1_output[" + std::to_string(output_idx) +
                                     "] -> module2_input[" + std::to_string(input_idx) + "]");

      TORCH_NEURONX_DEBUG("Validated dependency: module1_output[", output_idx,
                          "] -> module2_input[", input_idx,
                          "] types match: ", typeToString(output_type));
    }
  }

  // Validate module2 -> module1 dependencies
  // Support duplicate inputs by iterating through all input indices for each output
  for (const auto& dep : analysis.module2ToModule1Deps) {
    size_t output_idx = dep.first;
    const std::vector<size_t>& input_indices = dep.second;

    if (output_idx >= main2.getNumResults()) {
      throw std::runtime_error("Invalid dependency: module2 output index " +
                               std::to_string(output_idx) + " out of range");
    }

    auto output_type = main2.getResultTypes()[output_idx];

    // Validate each input that depends on this output
    for (size_t input_idx : input_indices) {
      if (input_idx >= main1.getNumArguments()) {
        throw std::runtime_error("Invalid dependency: module1 input index " +
                                 std::to_string(input_idx) + " out of range");
      }

      auto input_type = main1.getArgumentTypes()[input_idx];

      if (output_type != input_type) {
        throw std::runtime_error("Type mismatch in dependency: module2 output " +
                                 std::to_string(output_idx) + " (" + typeToString(output_type) +
                                 ") cannot connect to module1 input " + std::to_string(input_idx) +
                                 " (expected " + typeToString(input_type) + ")");
      }

      // Validate shape compatibility
      validateShapeCompatibility(output_type, input_type,
                                 "dependency module2_output[" + std::to_string(output_idx) +
                                     "] -> module1_input[" + std::to_string(input_idx) + "]");

      TORCH_NEURONX_DEBUG("Validated dependency: module2_output[", output_idx,
                          "] -> module1_input[", input_idx,
                          "] types match: ", typeToString(output_type));
    }
  }

  // Validate common inputs have matching types
  for (const auto& common : analysis.commonInputs) {
    size_t module1_idx = common.first;
    size_t module2_idx = common.second;

    if (module1_idx >= main1.getNumArguments()) {
      throw std::runtime_error("Invalid common input: module1 input index " +
                               std::to_string(module1_idx) + " out of range");
    }
    if (module2_idx >= main2.getNumArguments()) {
      throw std::runtime_error("Invalid common input: module2 input index " +
                               std::to_string(module2_idx) + " out of range");
    }

    auto type1 = main1.getArgumentTypes()[module1_idx];
    auto type2 = main2.getArgumentTypes()[module2_idx];

    if (type1 != type2) {
      throw std::runtime_error("Type mismatch in common input: module1 input " +
                               std::to_string(module1_idx) + " (" + typeToString(type1) +
                               ") does not match module2 input " + std::to_string(module2_idx) +
                               " (" + typeToString(type2) + ")");
    }

    // Validate shape compatibility
    validateShapeCompatibility(type1, type2,
                               "common input module1[" + std::to_string(module1_idx) +
                                   "] vs module2[" + std::to_string(module2_idx) + "]");

    TORCH_NEURONX_DEBUG("Validated common input: module1[", module1_idx, "] == module2[",
                        module2_idx, "] types match: ", typeToString(type1));
  }

  TORCH_NEURONX_DEBUG("validateDepCompatibility: All dependency compatibility checks passed");
}

/**
 * @brief Updates function call references within a cloned function to use new unique names
 *
 * @param clonedFunc The cloned function to update
 * @param nameMapping Map from old function names to new unique names
 */
static void updateFunctionCallReferences(
    mlir::func::FuncOp clonedFunc,
    const std::unordered_map<std::string, std::string>& nameMapping) {
  TORCH_NEURONX_DEBUG("Updating function call references in ", clonedFunc.getSymName().str());

  // Walk through all operations in the function body
  clonedFunc.walk([&](mlir::func::CallOp callOp) {
    std::string calleeName = callOp.getCallee().str();

    // Check if this call needs to be updated
    auto it = nameMapping.find(calleeName);
    if (it != nameMapping.end()) {
      TORCH_NEURONX_DEBUG("Updating call from '", calleeName, "' to '", it->second, "'");
      callOp.setCalleeAttr(mlir::SymbolRefAttr::get(callOp.getContext(), it->second));
    }
  });
}

/**
 * @brief Clones all functions from source modules with unique names and updates internal references
 *
 * @param mergedModule Target module to clone functions into
 * @param module1 First source module
 * @param module2 Second source module
 * @param builder MLIR builder for creating operations
 * @param analysis Dependency analysis to determine execution order
 * @return std::pair<std::string, std::string> - unique names for main1 and main2 (in execution
 * order)
 */
static std::pair<std::string, std::string> cloneModuleFunctions(
    mlir::ModuleOp mergedModule, mlir::ModuleOp module1, mlir::ModuleOp module2,
    mlir::OpBuilder& builder, const DependencyAnalysis& analysis) {
  TORCH_NEURONX_DEBUG("cloneModuleFunctions: Starting function cloning with unique names");

  // Build name mappings for all functions in both modules
  std::unordered_map<std::string, std::string> module1NameMapping;
  std::unordered_map<std::string, std::string> module2NameMapping;

  // Keep track of names we've assigned to avoid duplicates
  std::unordered_set<std::string> assignedNames;

  // First, collect all existing names in the merged module
  for (auto& op : mergedModule.getOps()) {
    if (auto func = llvm::dyn_cast<mlir::func::FuncOp>(op)) {
      assignedNames.insert(func.getSymName().str());
    }
  }

  // Helper function to generate unique names with tracking
  auto generateUniqueNameWithTracking = [&](const std::string& baseName) -> std::string {
    std::string candidateName = baseName;
    int counter = 1;

    while (assignedNames.count(candidateName) > 0) {
      candidateName = baseName + "_" + std::to_string(counter);
      counter++;
    }

    assignedNames.insert(candidateName);
    TORCH_NEURONX_DEBUG("Generated and tracked unique name: '", candidateName, "'");
    return candidateName;
  };

  // Generate unique names for module1 functions
  TORCH_NEURONX_DEBUG("Generating unique names for module1 functions");
  for (auto& op : module1.getOps()) {
    if (auto func = llvm::dyn_cast<mlir::func::FuncOp>(op)) {
      std::string originalName = func.getSymName().str();
      if (originalName == "main") {
        // Main function gets renamed to submain1
        std::string uniqueName = generateUniqueNameWithTracking("submain1");
        module1NameMapping[originalName] = uniqueName;
        TORCH_NEURONX_DEBUG("Module1 main function: ", originalName, " -> ", uniqueName);
      } else {
        // Other functions keep their base name but get unique suffix if needed
        std::string uniqueName = generateUniqueNameWithTracking(originalName);
        module1NameMapping[originalName] = uniqueName;
        TORCH_NEURONX_DEBUG("Module1 other function: ", originalName, " -> ", uniqueName);
      }
    }
  }

  // Generate unique names for module2 functions
  TORCH_NEURONX_DEBUG("Generating unique names for module2 functions");
  for (auto& op : module2.getOps()) {
    if (auto func = llvm::dyn_cast<mlir::func::FuncOp>(op)) {
      std::string originalName = func.getSymName().str();
      if (originalName == "main") {
        // Always use submain2 for the second module's main function to maintain consistency with
        // tests
        std::string uniqueName = generateUniqueNameWithTracking("submain2");
        module2NameMapping[originalName] = uniqueName;
        TORCH_NEURONX_DEBUG("Module2 main function: ", originalName, " -> ", uniqueName);
      } else {
        // Other functions keep their base name but get unique suffix if needed
        std::string uniqueName = generateUniqueNameWithTracking(originalName);
        module2NameMapping[originalName] = uniqueName;
        TORCH_NEURONX_DEBUG("Module2 other function: ", originalName, " -> ", uniqueName);
      }
    }
  }

  // Clone functions from module1 with new names
  TORCH_NEURONX_DEBUG("Cloning functions from module1");
  for (auto& op : module1.getOps()) {
    if (auto func = llvm::dyn_cast<mlir::func::FuncOp>(op)) {
      std::string originalName = func.getSymName().str();
      std::string uniqueName = module1NameMapping[originalName];

      auto clonedFunc = llvm::cast<mlir::func::FuncOp>(builder.clone(*func.getOperation()));
      clonedFunc.setSymName(uniqueName);
      clonedFunc.setPrivate();

      // Update internal function call references
      updateFunctionCallReferences(clonedFunc, module1NameMapping);

      TORCH_NEURONX_DEBUG("Cloned function from module1: ", originalName, " -> ", uniqueName);
    }
  }

  // Clone functions from module2 with new names
  TORCH_NEURONX_DEBUG("Cloning functions from module2");
  for (auto& op : module2.getOps()) {
    if (auto func = llvm::dyn_cast<mlir::func::FuncOp>(op)) {
      std::string originalName = func.getSymName().str();
      std::string uniqueName = module2NameMapping[originalName];

      auto clonedFunc = llvm::cast<mlir::func::FuncOp>(builder.clone(*func.getOperation()));
      clonedFunc.setSymName(uniqueName);
      clonedFunc.setPrivate();

      // Update internal function call references
      updateFunctionCallReferences(clonedFunc, module2NameMapping);

      TORCH_NEURONX_DEBUG("Cloned function from module2: ", originalName, " -> ", uniqueName);
    }
  }

  // Return the unique names for the main functions in execution order
  std::string firstFuncName =
      analysis.module1First ? module1NameMapping["main"] : module2NameMapping["main"];
  std::string secondFuncName =
      analysis.module1First ? module2NameMapping["main"] : module1NameMapping["main"];

  TORCH_NEURONX_DEBUG("cloneModuleFunctions: Returning function names: first=", firstFuncName,
                      ", second=", secondFuncName);

  return std::make_pair(firstFuncName, secondFuncName);
}

/**
 * @brief Structure to hold common input mapping information during merge
 */
struct InputMappingInfo {
  llvm::SmallVector<mlir::Type> inputTypes;
  std::vector<size_t> firstModuleInputMapping;
  std::vector<std::optional<size_t>> secondModuleInputMapping;
};

/**
 * @brief Structure to hold output type and inclusion mask information
 */
struct OutputBuildInfo {
  llvm::SmallVector<mlir::Type> outputTypes;
  std::vector<bool> firstModuleOutputIncluded;
  std::vector<bool> secondModuleOutputIncluded;
};

/**
 * @brief Builds output types and inclusion masks considering skip sets
 *
 * This function determines which outputs should be included in the merged module
 * based on the provided skip sets. Outputs can be skipped for two reasons:
 * 1. Explicitly listed in module1_outputs_to_skip or module2_outputs_to_skip
 * 2. Identified as dependencies (outputs consumed as inputs by the other module)
 *
 * Skip sets use ORIGINAL MODULE INDICES (0-based indices within each module),
 * not execution-order or merged indices.
 *
 * @param firstMain First-executing module's main function
 * @param secondMain Second-executing module's main function
 * @param analysis Dependency analysis containing execution order information
 * @param module1_outputs_to_skip Set of Module1 output indices to skip
 * @param module2_outputs_to_skip Set of Module2 output indices to skip
 *
 * @return OutputBuildInfo containing output types and inclusion masks
 */
static OutputBuildInfo buildInclusionMasks(
    mlir::func::FuncOp firstMain, mlir::func::FuncOp secondMain, const DependencyAnalysis& analysis,
    const std::unordered_set<size_t>& module1_outputs_to_skip,
    const std::unordered_set<size_t>& module2_outputs_to_skip) {
  OutputBuildInfo info;

  TORCH_NEURONX_DEBUG("buildInclusionMasks: Building output types with skip sets");
  TORCH_NEURONX_DEBUG("Module1 outputs to skip: ", module1_outputs_to_skip.size());
  TORCH_NEURONX_DEBUG("Module2 outputs to skip: ", module2_outputs_to_skip.size());

  // Determine which skip set applies to which executing module
  const std::unordered_set<size_t>& firstModuleSkipSet =
      analysis.module1First ? module1_outputs_to_skip : module2_outputs_to_skip;
  const std::unordered_set<size_t>& secondModuleSkipSet =
      analysis.module1First ? module2_outputs_to_skip : module1_outputs_to_skip;

  // Build output types for first-executing module
  for (size_t i = 0; i < firstMain.getNumResults(); ++i) {
    bool shouldSkip = firstModuleSkipSet.count(i) > 0;

    if (!shouldSkip) {
      info.outputTypes.push_back(firstMain.getResultTypes()[i]);
      info.firstModuleOutputIncluded.push_back(true);
      TORCH_NEURONX_DEBUG("Including first-executing module output ", i);
    } else {
      info.firstModuleOutputIncluded.push_back(false);
      TORCH_NEURONX_DEBUG("SKIPPING first-executing module output ", i);
    }
  }

  // Build output types for second-executing module
  for (size_t i = 0; i < secondMain.getNumResults(); ++i) {
    bool shouldSkip = secondModuleSkipSet.count(i) > 0;

    if (!shouldSkip) {
      info.outputTypes.push_back(secondMain.getResultTypes()[i]);
      info.secondModuleOutputIncluded.push_back(true);
      TORCH_NEURONX_DEBUG("Including second-executing module output ", i);
    } else {
      info.secondModuleOutputIncluded.push_back(false);
      TORCH_NEURONX_DEBUG("SKIPPING second-executing module output ", i);
    }
  }

  size_t totalOriginal = firstMain.getNumResults() + secondMain.getNumResults();
  size_t totalIncluded = info.outputTypes.size();
  TORCH_NEURONX_DEBUG("buildInclusionMasks: Total outputs ", totalIncluded, " (reduced from ",
                      totalOriginal, ")");

  return info;
}

/**
 * @brief Validates tensor address counts and retrieves main functions
 *
 * @throws std::runtime_error if validation fails or main functions not found
 */
static std::pair<mlir::func::FuncOp, mlir::func::FuncOp> validateAndGetMainFunctions(
    mlir::ModuleOp module1, mlir::ModuleOp module2, const std::vector<void*>& module1_input_addrs,
    const std::vector<void*>& module1_output_addrs, const std::vector<void*>& module2_input_addrs,
    const std::vector<void*>& module2_output_addrs) {
  // Get main functions for each module
  auto main1 = module1.lookupSymbol<mlir::func::FuncOp>("main");
  auto main2 = module2.lookupSymbol<mlir::func::FuncOp>("main");

  if (!main1) {
    throw std::runtime_error("Main function not found in module1");
  }
  if (!main2) {
    throw std::runtime_error("Main function not found in module2");
  }

  // Validate tensor address counts match MLIR function signatures
  if (module1_input_addrs.size() != main1.getNumArguments()) {
    throw std::runtime_error("Module1 input address count (" +
                             std::to_string(module1_input_addrs.size()) +
                             ") does not match function argument count (" +
                             std::to_string(main1.getNumArguments()) + ")");
  }
  if (module1_output_addrs.size() != main1.getNumResults()) {
    throw std::runtime_error(
        "Module1 output address count (" + std::to_string(module1_output_addrs.size()) +
        ") does not match function result count (" + std::to_string(main1.getNumResults()) + ")");
  }
  if (module2_input_addrs.size() != main2.getNumArguments()) {
    throw std::runtime_error("Module2 input address count (" +
                             std::to_string(module2_input_addrs.size()) +
                             ") does not match function argument count (" +
                             std::to_string(main2.getNumArguments()) + ")");
  }
  if (module2_output_addrs.size() != main2.getNumResults()) {
    throw std::runtime_error(
        "Module2 output address count (" + std::to_string(module2_output_addrs.size()) +
        ") does not match function result count (" + std::to_string(main2.getNumResults()) + ")");
  }

  return std::make_pair(main1, main2);
}

/**
 * @brief Builds input types and mappings for merged function
 *
 * This handles deduplication of common inputs and dependency resolution
 * Now supports duplicate inputs (e.g., x + x) via vector-based dependencies
 */
static InputMappingInfo buildMappings(mlir::func::FuncOp firstMain, mlir::func::FuncOp secondMain,
                                      const DependencyAnalysis& analysis,
                                      const std::unordered_map<size_t, std::vector<size_t>>& deps) {
  InputMappingInfo info;

  // Add inputs from first-executing module
  for (size_t i = 0; i < firstMain.getNumArguments(); ++i) {
    info.inputTypes.push_back(firstMain.getArgumentTypes()[i]);
    info.firstModuleInputMapping.push_back(info.inputTypes.size() - 1);
  }

  // Add inputs from second-executing module, handling common inputs and dependencies
  for (size_t i = 0; i < secondMain.getNumArguments(); ++i) {
    // Check if this input index appears in ANY dependency vector
    bool isDependency = false;
    for (const auto& dep : deps) {
      const std::vector<size_t>& input_indices = dep.second;
      if (std::find(input_indices.begin(), input_indices.end(), i) != input_indices.end()) {
        isDependency = true;
        break;
      }
    }

    if (isDependency) {
      // This input comes from first module's output - mark as dependency with nullopt
      info.secondModuleInputMapping.push_back(std::nullopt);
      TORCH_NEURONX_DEBUG("Second module input ", i, " is a dependency");
    } else {
      // Check if this is a common input (shared between both modules)
      bool isCommon = false;
      std::optional<size_t> commonInputIdx = std::nullopt;

      if (analysis.module1First) {
        // Look for module2 input i in common inputs
        auto commonIt = std::find_if(analysis.commonInputs.begin(), analysis.commonInputs.end(),
                                     [i](const auto& pair) { return pair.second == i; });
        if (commonIt != analysis.commonInputs.end()) {
          commonInputIdx = info.firstModuleInputMapping[commonIt->first];
          isCommon = true;
        }
      } else {
        // Look for module1 input i in common inputs (reversed mapping)
        auto commonIt = std::find_if(analysis.commonInputs.begin(), analysis.commonInputs.end(),
                                     [i](const auto& pair) { return pair.first == i; });
        if (commonIt != analysis.commonInputs.end()) {
          commonInputIdx = info.firstModuleInputMapping[commonIt->second];
          isCommon = true;
        }
      }

      if (isCommon) {
        // This input is shared - map to existing input in merged function
        info.secondModuleInputMapping.push_back(commonInputIdx.value());
        TORCH_NEURONX_DEBUG("Second module input ", i, " maps to common input ",
                            commonInputIdx.value());
      } else {
        // This is a unique input - add new input to merged function
        info.inputTypes.push_back(secondMain.getArgumentTypes()[i]);
        info.secondModuleInputMapping.push_back(info.inputTypes.size() - 1);
        TORCH_NEURONX_DEBUG("Second module input ", i, " is unique, mapped to ",
                            (info.inputTypes.size() - 1));
      }
    }
  }

  return info;
}

/**
 * @brief Populates input mapping information if mapping pointer is provided
 */
static void populateInputMapping(MergeMapping* mapping, const InputMappingInfo& inputInfo,
                                 const DependencyAnalysis& analysis) {
  if (!mapping) return;

  // IMPORTANT: Module numbers are based on INPUT ARGUMENT ORDER to
  // mergeStableHLOModulesSkipIntermediates, NOT execution order. Module 1 = first argument
  // (module1), Module 2 = second argument (module2). This ensures the mapping reflects the original
  // function call order.

  // Determine which inputs come from module1 vs module2 based on execution order
  bool firstIsModule1 = analysis.module1First;

  // Build input mapping - First module inputs
  for (size_t i = 0; i < inputInfo.firstModuleInputMapping.size(); ++i) {
    size_t merged_idx = inputInfo.firstModuleInputMapping[i];
    int module_num = firstIsModule1 ? 1 : 2;
    mapping->input_mapping[merged_idx] = {module_num, i};
    TORCH_NEURONX_DEBUG("Input mapping: merged[", merged_idx, "] -> module", module_num, "[", i,
                        "]");
  }

  // Second module inputs (only non-dependency, non-common inputs)
  for (size_t i = 0; i < inputInfo.secondModuleInputMapping.size(); ++i) {
    if (inputInfo.secondModuleInputMapping[i].has_value()) {
      size_t merged_idx = inputInfo.secondModuleInputMapping[i].value();
      // Check if this is already mapped (common input case)
      if (mapping->input_mapping.find(merged_idx) == mapping->input_mapping.end()) {
        int module_num = firstIsModule1 ? 2 : 1;
        mapping->input_mapping[merged_idx] = {module_num, i};
        TORCH_NEURONX_DEBUG("Input mapping: merged[", merged_idx, "] -> module", module_num, "[", i,
                            "]");
      }
    }
  }
}

/**
 * @brief Populates output mapping information if mapping pointer is provided
 */
static void populateOutputMapping(MergeMapping* mapping, const OutputBuildInfo& outputInfo,
                                  const DependencyAnalysis& analysis) {
  if (!mapping) return;

  size_t output_idx = 0;

  // IMPORTANT: Module numbers are based on INPUT ARGUMENT ORDER to
  // mergeStableHLOModulesSkipIntermediates, NOT execution order. Module 1 = first argument
  // (module1), Module 2 = second argument (module2). This ensures the mapping reflects the original
  // function call order.

  // Determine which outputs come from module1 vs module2 based on execution order
  bool firstIsModule1 = analysis.module1First;

  // First-executing module outputs (only included ones)
  for (size_t i = 0; i < outputInfo.firstModuleOutputIncluded.size(); ++i) {
    if (outputInfo.firstModuleOutputIncluded[i]) {
      int module_num = firstIsModule1 ? 1 : 2;
      mapping->output_mapping[output_idx] = {module_num, i};
      TORCH_NEURONX_DEBUG("Output mapping: merged[", output_idx, "] -> module", module_num, "[", i,
                          "]");
      output_idx++;
    }
  }

  // Second-executing module outputs (only included ones)
  for (size_t i = 0; i < outputInfo.secondModuleOutputIncluded.size(); ++i) {
    if (outputInfo.secondModuleOutputIncluded[i]) {
      int module_num = firstIsModule1 ? 2 : 1;
      mapping->output_mapping[output_idx] = {module_num, i};
      TORCH_NEURONX_DEBUG("Output mapping: merged[", output_idx, "] -> module", module_num, "[", i,
                          "]");
      output_idx++;
    }
  }
}

/**
 * @brief Runs optimization passes on a merged MLIR module
 *
 * Applies a standard optimization pipeline including inlining, dead code elimination,
 * constant propagation, canonicalization, and common subexpression elimination.
 * Creates a backup before optimization if verification is enabled, restoring on failure.
 *
 * @param mergedModule The module to optimize (ownership transferred)
 * @param context MLIR context for pass manager
 * @param verify_output Whether to create backup and enable verification
 * @return The optimized module (or backup if optimization fails)
 */
static mlir::OwningOpRef<mlir::ModuleOp> runOptimizationPipeline(
    mlir::OwningOpRef<mlir::ModuleOp> mergedModule, mlir::MLIRContext* context,
    bool verify_output) {
  // Create backup of module before optimization in case optimization breaks verification
  mlir::OwningOpRef<mlir::ModuleOp> backupModule = nullptr;
  if (verify_output) {
    backupModule = mergedModule->clone();
    TORCH_NEURONX_DEBUG("runOptimizationPipeline: Created backup module before optimization");
  }

  TORCH_NEURONX_DEBUG(
      "runOptimizationPipeline: Running optimization passes (canonicalization + CSE)");

  // Register the func dialect's inliner extension to enable inlining of func.call operations
  mlir::DialectRegistry registry;
  mlir::func::registerInlinerExtension(registry);
  context->appendDialectRegistry(registry);

  mlir::PassManager pm(context);
  pm.enableVerifier(true);

  // 1. Inlining: Flatten the function hierarchy.
  pm.addPass(mlir::createInlinerPass());

  // 2. Symbol DCE: Remove any functions or globals that became unused after inlining.
  pm.addPass(mlir::createSymbolDCEPass());

  // 3. Constant Propagation: High-level constant folding across control flow.
  pm.addPass(mlir::createSCCPPass());

  // 4. Canonicalize: The main cleanup (folds 1+1, x*0, etc.)
  pm.addPass(mlir::createCanonicalizerPass());

  // 5. CSE: Common Subexpression Elimination.
  pm.addPass(mlir::createCSEPass());

  // 6. Final Canonicalize: Clean up any secondary opportunities created by CSE.
  pm.addPass(mlir::createCanonicalizerPass());

  if (mlir::failed(pm.run(*mergedModule))) {
    TORCH_NEURONX_DEBUG("WARNING: runOptimizationPipeline: Optimization pipeline failed mid-run.");

    if (backupModule) {
      TORCH_NEURONX_DEBUG("runOptimizationPipeline: Reverting to pre-optimization backup.");
      return backupModule;
    }
    throw std::runtime_error("Optimization failed and no backup available.");
  }

  TORCH_NEURONX_DEBUG(
      "runOptimizationPipeline: Optimization passes completed successfully (verified by "
      "PassManager)");
  return mergedModule;
}

/**
 * @brief Prepares arguments for the second-executing module
 * Now supports duplicate inputs (e.g., x + x) via vector-based dependencies
 */
static llvm::SmallVector<mlir::Value> prepareSecondModuleArguments(
    mlir::func::FuncOp secondMain,
    const std::vector<std::optional<size_t>>& secondModuleInputMapping,
    const std::unordered_map<size_t, std::vector<size_t>>& deps, mlir::func::CallOp firstCall,
    llvm::ArrayRef<mlir::BlockArgument> args) {
  llvm::SmallVector<mlir::Value> secondArgs;
  secondArgs.resize(secondMain.getNumArguments());

  TORCH_NEURONX_DEBUG("=== prepareSecondModuleArguments DEBUG ===");
  TORCH_NEURONX_DEBUG("secondMain num arguments: ", secondMain.getNumArguments());
  TORCH_NEURONX_DEBUG("firstCall num results: ", firstCall.getNumResults());
  TORCH_NEURONX_DEBUG("Dependency map size: ", deps.size());

  // Log all dependencies
  for (const auto& dep : deps) {
    TORCH_NEURONX_DEBUG("  Dependency: output[", dep.first, "] -> inputs[", dep.second.size(),
                        " indices]");
    for (size_t idx : dep.second) {
      TORCH_NEURONX_DEBUG("    -> input[", idx, "]");
    }
  }

  // Fill in arguments in the correct positions
  for (size_t i = 0; i < secondModuleInputMapping.size(); ++i) {
    if (!secondModuleInputMapping[i].has_value()) {
      // This input is a dependency - find which output it comes from
      // Search through all dependency vectors to find this input index
      bool found = false;
      for (const auto& dep : deps) {
        size_t output_idx = dep.first;
        const std::vector<size_t>& input_indices = dep.second;

        // Check if this input index is in this dependency's vector
        if (std::find(input_indices.begin(), input_indices.end(), i) != input_indices.end()) {
          secondArgs[i] = firstCall.getResult(output_idx);
          TORCH_NEURONX_DEBUG("✓ Mapped: secondModule input[", i, "] = firstCall.result[",
                              output_idx, "]");
          found = true;
          break;
        }
      }

      if (!found) {
        TORCH_NEURONX_DEBUG("✗ ERROR: No dependency mapping found for input ", i);
        throw std::runtime_error("Dependency mapping error - no matching output found for input " +
                                 std::to_string(i));
      }
    } else {
      // This input comes from merged function arguments
      secondArgs[i] = args[secondModuleInputMapping[i].value()];
      TORCH_NEURONX_DEBUG("✓ Mapped: secondModule input[", i, "] = mergedFunction arg[",
                          secondModuleInputMapping[i].value(), "]");
    }
  }

  TORCH_NEURONX_DEBUG("=== prepareSecondModuleArguments COMPLETE ===");
  return secondArgs;
}

/**
 * @brief Core merge implementation that takes pre-computed dependency analysis
 *
 * This is the low-level merge function that performs the actual MLIR merging.
 * It does NOT call analyzeDependencies - the caller must provide the analysis.
 *
 * @param module1 First MLIR module to merge
 * @param module2 Second MLIR module to merge
 * @param context MLIR context with required dialects loaded
 * @param module1_input_addrs Tensor addresses for module1 inputs
 * @param module1_output_addrs Tensor addresses for module1 outputs
 * @param module2_input_addrs Tensor addresses for module2 inputs
 * @param module2_output_addrs Tensor addresses for module2 outputs
 * @param analysis Pre-computed dependency analysis (from analyzeDependencies or
 * analyzeDependenciesInOrder)
 * @param mapping Optional pointer to capture mapping information
 * @param verify_output Whether to verify the merged module
 * @param module1_outputs_to_skip Set of module1 output indices to skip (for skip intermediates)
 * @param module2_outputs_to_skip Set of module2 output indices to skip (for skip intermediates)
 */
mlir::OwningOpRef<mlir::ModuleOp> mergeModulesWithAnalysis(
    mlir::ModuleOp module1, mlir::ModuleOp module2, mlir::MLIRContext* context,
    const std::vector<void*>& module1_input_addrs, const std::vector<void*>& module1_output_addrs,
    const std::vector<void*>& module2_input_addrs, const std::vector<void*>& module2_output_addrs,
    const DependencyAnalysis& analysis, MergeMapping* mapping = nullptr, bool verify_output = true,
    const std::unordered_set<size_t>& module1_outputs_to_skip = {},
    const std::unordered_set<size_t>& module2_outputs_to_skip = {}, bool run_optimization = true) {
  TORCH_NEURONX_TRACE_FUNCTION_WITH_ARGS(
      "mod1_inputs:", module1_input_addrs.size(), "mod1_outputs:", module1_output_addrs.size(),
      "mod2_inputs:", module2_input_addrs.size(), "mod2_outputs:", module2_output_addrs.size());
  TORCH_NEURONX_DEBUG("mergeModulesWithAnalysis: Starting module merge with pre-computed analysis");
  TORCH_NEURONX_DEBUG("mergeModulesWithAnalysis: Skip sets - Module1: ",
                      module1_outputs_to_skip.size(), " Module2: ", module2_outputs_to_skip.size());

  // Validate context parameter
  if (context == nullptr) {
    throw std::runtime_error("MLIRContext cannot be null");
  }

  mlir::OpBuilder builder(context);
  mlir::OwningOpRef<mlir::ModuleOp> mergedModule =
      builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
  auto& mergedBody = mergedModule->getBodyRegion().front();
  builder.setInsertionPointToEnd(&mergedBody);

  // Validate and get main functions using helper
  auto [main1, main2] =
      validateAndGetMainFunctions(module1, module2, module1_input_addrs, module1_output_addrs,
                                  module2_input_addrs, module2_output_addrs);

  TORCH_NEURONX_DEBUG("mergeModulesWithAnalysis: Tensor address validation passed");

  // Validate type and shape compatibility for dependencies and common inputs
  validateDepCompatibility(main1, main2, analysis);

  // Unified merge logic that handles all scenarios
  TORCH_NEURONX_DEBUG("mergeModulesWithAnalysis: Processing scenario ", analysis.scenario);

  // Determine execution order
  auto firstMain = analysis.module1First ? main1 : main2;
  auto secondMain = analysis.module1First ? main2 : main1;
  auto& deps =
      analysis.module1First ? analysis.module1ToModule2Deps : analysis.module2ToModule1Deps;

  // Build input types and mappings using helper function
  auto inputInfo = buildMappings(firstMain, secondMain, analysis, deps);

  // Build output types and inclusion masks using the new helper function
  auto outputInfo = buildInclusionMasks(firstMain, secondMain, analysis, module1_outputs_to_skip,
                                        module2_outputs_to_skip);

  TORCH_NEURONX_DEBUG("mergeModulesWithAnalysis: Merged function will have ",
                      inputInfo.inputTypes.size(), " inputs, ", outputInfo.outputTypes.size(),
                      " outputs");

  // Populate mapping information if requested (using helper functions)
  if (mapping) {
    mapping->total_inputs = inputInfo.inputTypes.size();
    mapping->total_outputs = outputInfo.outputTypes.size();

    populateInputMapping(mapping, inputInfo, analysis);
    populateOutputMapping(mapping, outputInfo, analysis);

    TORCH_NEURONX_DEBUG("Mapping populated - ", mapping->input_mapping.size(), " input mappings, ",
                        mapping->output_mapping.size(), " output mappings");
  }

  // Create merged main function
  auto funcType = builder.getFunctionType(inputInfo.inputTypes, outputInfo.outputTypes);
  auto mergedMain = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "main", funcType);
  mergedMain.setPublic();

  // Create function body
  auto& body = mergedMain.getBody().emplaceBlock();

  // Add arguments to the function body block
  for (size_t i = 0; i < inputInfo.inputTypes.size(); ++i) {
    body.addArgument(inputInfo.inputTypes[i], builder.getUnknownLoc());
  }

  builder.setInsertionPointToStart(&body);
  auto args = body.getArguments();

  // Clone all functions from both modules with unique names and update internal references
  builder.setInsertionPointToEnd(&mergedBody);
  auto [firstFuncName, secondFuncName] =
      cloneModuleFunctions(*mergedModule, module1, module2, builder, analysis);

  TORCH_NEURONX_DEBUG("Using function names for calls: first=", firstFuncName,
                      ", second=", secondFuncName);

  // Now go back to create the main function body with the function calls
  builder.setInsertionPointToStart(&body);

  // Prepare arguments for first-executing module
  llvm::SmallVector<mlir::Value> firstArgs;
  for (size_t idx : inputInfo.firstModuleInputMapping) {
    firstArgs.push_back(args[idx]);
  }

  // Call first-executing module
  auto firstCall = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), firstFuncName,
                                                      firstMain.getResultTypes(), firstArgs);

  // Prepare arguments for second-executing module using helper function
  auto secondArgs = prepareSecondModuleArguments(secondMain, inputInfo.secondModuleInputMapping,
                                                 deps, firstCall, args);

  // Call second-executing module
  auto secondCall = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), secondFuncName,
                                                       secondMain.getResultTypes(), secondArgs);

  // Combine results in original module order, but only include outputs not in skip sets
  llvm::SmallVector<mlir::Value> results;
  if (analysis.module1First) {
    // Add Module1 outputs (only included ones)
    for (size_t i = 0; i < outputInfo.firstModuleOutputIncluded.size(); ++i) {
      if (outputInfo.firstModuleOutputIncluded[i]) {
        results.push_back(firstCall.getResult(i));
      }
    }
    // Add Module2 outputs (only included ones)
    for (size_t i = 0; i < outputInfo.secondModuleOutputIncluded.size(); ++i) {
      if (outputInfo.secondModuleOutputIncluded[i]) {
        results.push_back(secondCall.getResult(i));
      }
    }
  } else {
    // When module2 executes first (module1First=false):
    // - firstMain = main2, secondMain = main1
    // - outputTypes was built as [firstMain outputs..., secondMain outputs...]
    // - So we must push results in the same order to match the function signature
    for (size_t i = 0; i < outputInfo.firstModuleOutputIncluded.size(); ++i) {
      if (outputInfo.firstModuleOutputIncluded[i]) {
        results.push_back(firstCall.getResult(i));  // firstMain (main2) outputs
      }
    }
    for (size_t i = 0; i < outputInfo.secondModuleOutputIncluded.size(); ++i) {
      if (outputInfo.secondModuleOutputIncluded[i]) {
        results.push_back(secondCall.getResult(i));  // secondMain (main1) outputs
      }
    }
  }

  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), results);

  TORCH_NEURONX_DEBUG("mergeModulesWithAnalysis: Successfully merged modules");

  // Verify the merged module for correctness (if enabled)
  if (verify_output) {
    if (mlir::failed(mlir::verify(*mergedModule))) {
      TORCH_NEURONX_DEBUG("ERROR: mergeModules: Merged module failed MLIR verification");
      throw std::runtime_error("Merged module failed MLIR verification - invalid IR generated");
    }
    TORCH_NEURONX_DEBUG("mergeModulesWithAnalysis: Merged module passed MLIR verification");
  } else {
    TORCH_NEURONX_DEBUG(
        "mergeModulesWithAnalysis: Skipping MLIR verification (verify_output=false)");
  }

  // Run optimization pipeline on the merged module (if enabled)
  if (run_optimization) {
    return runOptimizationPipeline(std::move(mergedModule), context, verify_output);
  }

  TORCH_NEURONX_DEBUG(
      "mergeModulesWithAnalysis: Skipping optimization pipeline (run_optimization=false)");
  return mergedModule;
}

/**
 * @brief Enhanced mergeModules with dependency analysis support (wrapper)
 *
 * This is a convenience wrapper that calls analyzeDependencies() internally,
 * then delegates to mergeModulesWithAnalysis().
 *
 * Merges two MLIR StableHLO modules based on tensor address analysis to handle:
 * 1. Independent modules
 * 2. Modules with common inputs
 * 3. Modules with direct dependencies
 * 4. Mixed scenarios (common inputs + dependencies)
 *
 * OUTPUT ORDERING BEHAVIOR:
 * The merged MLIR always returns outputs in ORIGINAL MODULE ORDER, regardless of execution order.
 * This means Module1 outputs always come first, then Module2 outputs, even if Module2 executes
 * before Module1 due to dependencies.
 */
mlir::OwningOpRef<mlir::ModuleOp> mergeModules(
    mlir::ModuleOp module1, mlir::ModuleOp module2, mlir::MLIRContext* context,
    const std::vector<void*>& module1_input_addrs, const std::vector<void*>& module1_output_addrs,
    const std::vector<void*>& module2_input_addrs, const std::vector<void*>& module2_output_addrs,
    MergeMapping* mapping, bool verify_output,
    const std::unordered_set<size_t>& module1_outputs_to_skip,
    const std::unordered_set<size_t>& module2_outputs_to_skip, bool run_optimization) {
  TORCH_NEURONX_TRACE_FUNCTION_WITH_ARGS(
      "mod1_inputs:", module1_input_addrs.size(), "mod1_outputs:", module1_output_addrs.size(),
      "mod2_inputs:", module2_input_addrs.size(), "mod2_outputs:", module2_output_addrs.size());
  TORCH_NEURONX_DEBUG("mergeModules: Starting enhanced module merge with dependency analysis");

  // Validate context parameter
  if (context == nullptr) {
    throw std::runtime_error("MLIRContext cannot be null");
  }

  // Analyze dependencies between modules (with circular dependency check)
  auto analysis = analyzeDependencies(module1_input_addrs, module1_output_addrs,
                                      module2_input_addrs, module2_output_addrs);

  // Delegate to mergeModulesWithAnalysis with pre-computed analysis
  return mergeModulesWithAnalysis(module1, module2, context, module1_input_addrs,
                                  module1_output_addrs, module2_input_addrs, module2_output_addrs,
                                  analysis, mapping, verify_output, module1_outputs_to_skip,
                                  module2_outputs_to_skip, run_optimization);
}

/**
 * @brief Enhanced interface to merge two StableHLO modules with dependency analysis
 *
 * Parses two MLIR text representations, analyzes tensor dependencies, and merges them
 * using the enhanced mergeModules() function that handles all 4 scenarios.
 *
 * @param mod_str1 MLIR text of first module
 * @param mod_str2 MLIR text of second module
 * @param module1_input_addrs Tensor addresses for module1 inputs
 * @param module1_output_addrs Tensor addresses for module1 outputs
 * @param module2_input_addrs Tensor addresses for module2 inputs
 * @param module2_output_addrs Tensor addresses for module2 outputs
 *
 * @return Merged module as MLIR text string, empty string on failure
 */
std::string mergeStableHLOModules(const std::string& mod_str1, const std::string& mod_str2,
                                  const std::vector<void*>& module1_input_addrs,
                                  const std::vector<void*>& module1_output_addrs,
                                  const std::vector<void*>& module2_input_addrs,
                                  const std::vector<void*>& module2_output_addrs,
                                  bool verify_output, bool run_optimization) {
  TORCH_NEURONX_TRACE_FUNCTION_WITH_ARGS("mod1_size:", mod_str1.size(),
                                         "mod2_size:", mod_str2.size());
  TORCH_NEURONX_DEBUG(
      "mergeStableHLOModules: Starting enhanced merge process with dependency analysis");

  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();
  context.getOrLoadDialect<mlir::chlo::ChloDialect>();

  TORCH_NEURONX_DEBUG("mergeStableHLOModules: Parsing first module");
  auto mod1 = mlir::parseSourceString<mlir::ModuleOp>(mod_str1, &context);
  if (!mod1) {
    TORCH_NEURONX_DEBUG("ERROR: mergeStableHLOModules: Failed to parse first module");
    return "";
  }

  TORCH_NEURONX_DEBUG("mergeStableHLOModules: Parsing second module");
  auto mod2 = mlir::parseSourceString<mlir::ModuleOp>(mod_str2, &context);
  if (!mod2) {
    TORCH_NEURONX_DEBUG("ERROR: mergeStableHLOModules: Failed to parse second module");
    return "";
  }

  TORCH_NEURONX_DEBUG("mergeStableHLOModules: Merging modules with dependency analysis");
  auto new_mod = mergeModules(*mod1, *mod2, &context, module1_input_addrs, module1_output_addrs,
                              module2_input_addrs, module2_output_addrs, nullptr, verify_output, {},
                              {}, run_optimization);

  TORCH_NEURONX_DEBUG("mergeStableHLOModules: Converting merged module to string");
  return stablehlo_utils::moduleToString(*new_mod);
}

/**
 * @brief String-based wrapper for mergeModulesSkipIntermediates that skips intermediate outputs
 *
 * This function takes MLIR strings, parses them, performs merge with skip intermediates,
 * and returns the result as a string.
 *
 * @param mod_str1 MLIR text of first module
 * @param mod_str2 MLIR text of second module
 * @param module1_input_addrs Tensor addresses for module1 inputs
 * @param module1_output_addrs Tensor addresses for module1 outputs
 * @param module2_input_addrs Tensor addresses for module2 inputs
 * @param module2_output_addrs Tensor addresses for module2 outputs
 * @param mapping Optional pointer to capture mapping information
 * @param verify_output Optional flag to enable/disable MLIR verification (default: true)
 *
 * @return Merged module as MLIR text string with intermediates skipped, empty string on failure
 */
std::string mergeStableHLOModulesSkipIntermediates(const std::string& mod_str1,
                                                   const std::string& mod_str2,
                                                   const std::vector<void*>& module1_input_addrs,
                                                   const std::vector<void*>& module1_output_addrs,
                                                   const std::vector<void*>& module2_input_addrs,
                                                   const std::vector<void*>& module2_output_addrs,
                                                   MergeMapping* mapping, bool verify_output) {
  TORCH_NEURONX_DEBUG("mergeStableHLOModulesSkipIntermediates: Starting skip intermediates merge");

  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();
  context.getOrLoadDialect<mlir::chlo::ChloDialect>();

  // Parse both modules
  auto module1 = mlir::parseSourceString<mlir::ModuleOp>(mod_str1, &context);
  if (!module1) {
    TORCH_NEURONX_DEBUG(
        "ERROR: mergeStableHLOModulesSkipIntermediates: Failed to parse first MLIR module");
    return "";
  }

  auto module2 = mlir::parseSourceString<mlir::ModuleOp>(mod_str2, &context);
  if (!module2) {
    TORCH_NEURONX_DEBUG(
        "ERROR: mergeStableHLOModulesSkipIntermediates: Failed to parse second MLIR module");
    return "";
  }

  // Analyze dependencies
  auto analysis = analyzeDependencies(module1_input_addrs, module1_output_addrs,
                                      module2_input_addrs, module2_output_addrs);

  // Merge with skip intermediates, passing mapping pointer through
  auto merged_module = mergeModulesSkipIntermediates(
      module1.get(), module2.get(), &context, module1_input_addrs, module1_output_addrs,
      module2_input_addrs, module2_output_addrs, analysis, mapping, verify_output);

  if (!merged_module) {
    TORCH_NEURONX_DEBUG("ERROR: mergeStableHLOModulesSkipIntermediates: Module merge failed");
    return "";
  }

  // Convert back to string
  return stablehlo_utils::moduleToString(merged_module.get());
}

/**
 * @brief Extracts tensor addresses from tensor vectors for dependency analysis
 *
 * @param module1_inputs Input tensors from module 1
 * @param module1_outputs Output tensors from module 1
 * @param module2_inputs Input tensors from module 2
 * @param module2_outputs Output tensors from module 2
 * @param module1_input_addrs Output vector for module1 input addresses
 * @param module1_output_addrs Output vector for module1 output addresses
 * @param module2_input_addrs Output vector for module2 input addresses
 * @param module2_output_addrs Output vector for module2 output addresses
 */
static void extractTensorAddresses(
    const std::vector<at::Tensor>& module1_inputs, const std::vector<at::Tensor>& module1_outputs,
    const std::vector<at::Tensor>& module2_inputs, const std::vector<at::Tensor>& module2_outputs,
    std::vector<void*>& module1_input_addrs, std::vector<void*>& module1_output_addrs,
    std::vector<void*>& module2_input_addrs, std::vector<void*>& module2_output_addrs) {
  for (const auto& tensor : module1_inputs) {
    module1_input_addrs.push_back(tensor.data_ptr());
  }
  for (const auto& tensor : module1_outputs) {
    module1_output_addrs.push_back(tensor.data_ptr());
  }
  for (const auto& tensor : module2_inputs) {
    module2_input_addrs.push_back(tensor.data_ptr());
  }
  for (const auto& tensor : module2_outputs) {
    module2_output_addrs.push_back(tensor.data_ptr());
  }
}

/**
 * @brief Helper function to populate merged inputs based on mapping
 *
 * @param merged_inputs Output vector to populate with merged input tensors
 * @param mapping Mapping information from merge operation
 * @param module1_inputs Input tensors from module 1
 * @param module2_inputs Input tensors from module 2
 */
static void populateMergedInputsFromMapping(std::vector<at::Tensor>& merged_inputs,
                                            const MergeMapping& mapping,
                                            const std::vector<at::Tensor>& module1_inputs,
                                            const std::vector<at::Tensor>& module2_inputs) {
  merged_inputs.resize(mapping.total_inputs);

  for (const auto& mapping_entry : mapping.input_mapping) {
    size_t merged_idx = mapping_entry.first;
    int module_num = mapping_entry.second.first;
    size_t original_idx = mapping_entry.second.second;

    // Get the tensor from the appropriate module
    if (module_num == 1) {
      if (original_idx < module1_inputs.size()) {
        merged_inputs[merged_idx] = module1_inputs[original_idx];
        TORCH_NEURONX_DEBUG("Merged input[", merged_idx, "] = module1_input[", original_idx, "]");
      }
    } else if (module_num == 2) {
      if (original_idx < module2_inputs.size()) {
        merged_inputs[merged_idx] = module2_inputs[original_idx];
        TORCH_NEURONX_DEBUG("Merged input[", merged_idx, "] = module2_input[", original_idx, "]");
      }
    }
  }
}

/**
 * @brief Helper function to populate merged outputs based on mapping
 *
 * @param merged_outputs Output vector to populate with merged output tensors
 * @param mapping Mapping information from merge operation
 * @param module1_outputs Output tensors from module 1
 * @param module2_outputs Output tensors from module 2
 */
static void populateMergedOutputsFromMapping(std::vector<at::Tensor>& merged_outputs,
                                             const MergeMapping& mapping,
                                             const std::vector<at::Tensor>& module1_outputs,
                                             const std::vector<at::Tensor>& module2_outputs) {
  merged_outputs.resize(mapping.total_outputs);

  for (const auto& mapping_entry : mapping.output_mapping) {
    size_t merged_idx = mapping_entry.first;
    int module_num = mapping_entry.second.first;
    size_t original_idx = mapping_entry.second.second;

    // Get the tensor from the appropriate module
    if (module_num == 1) {
      if (original_idx < module1_outputs.size()) {
        merged_outputs[merged_idx] = module1_outputs[original_idx];
        TORCH_NEURONX_DEBUG("Merged output[", merged_idx, "] = module1_output[", original_idx, "]");
      }
    } else if (module_num == 2) {
      if (original_idx < module2_outputs.size()) {
        merged_outputs[merged_idx] = module2_outputs[original_idx];
        TORCH_NEURONX_DEBUG("Merged output[", merged_idx, "] = module2_output[", original_idx, "]");
      }
    }
  }
}

/**
 * @brief Enhanced interface that returns both merged MLIR and correct tensor mappings
 *
 * This function builds the correct input/output tensor sets based on the MergeMapping
 * information from the dependency analysis. It ensures that the tensor sets match
 * what the merged MLIR expects.
 *
 * IMPORTANT: This function converts the merged MLIR to string format to avoid
 * context lifetime issues. The returned MergeResult contains a string representation
 * rather than a live MLIR module to prevent segmentation faults.
 */
MergeResult mergeStableHLOModulesWithTensors(const std::string& mod_str1,
                                             const std::string& mod_str2,
                                             const std::vector<at::Tensor>& module1_inputs,
                                             const std::vector<at::Tensor>& module1_outputs,
                                             const std::vector<at::Tensor>& module2_inputs,
                                             const std::vector<at::Tensor>& module2_outputs,
                                             bool verify_output) {
  TORCH_NEURONX_DEBUG(
      "mergeStableHLOModulesWithTensors: Starting enhanced merge with tensor mapping");

  MergeResult result;
  result.success = false;

  try {
    // Extract tensor addresses for dependency analysis using helper
    std::vector<void*> module1_input_addrs, module1_output_addrs;
    std::vector<void*> module2_input_addrs, module2_output_addrs;

    extractTensorAddresses(module1_inputs, module1_outputs, module2_inputs, module2_outputs,
                           module1_input_addrs, module1_output_addrs, module2_input_addrs,
                           module2_output_addrs);

    TORCH_NEURONX_DEBUG("mergeStableHLOModulesWithTensors: Extracted ", module1_input_addrs.size(),
                        " + ", module1_output_addrs.size(), " + ", module2_input_addrs.size(),
                        " + ", module2_output_addrs.size(), " tensor addresses");

    // Parse MLIR modules once (avoiding double computation)
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();
    context.getOrLoadDialect<mlir::chlo::ChloDialect>();

    auto mod1 = mlir::parseSourceString<mlir::ModuleOp>(mod_str1, &context);
    auto mod2 = mlir::parseSourceString<mlir::ModuleOp>(mod_str2, &context);

    if (!mod1 || !mod2) {
      TORCH_NEURONX_DEBUG("ERROR: mergeStableHLOModulesWithTensors: Failed to parse modules");
      return result;
    }

    // Get mapping information by calling mergeModules with mapping capture
    MergeMapping mapping;
    auto merged_module =
        mergeModules(*mod1, *mod2, &context, module1_input_addrs, module1_output_addrs,
                     module2_input_addrs, module2_output_addrs, &mapping, verify_output);

    if (!merged_module) {
      TORCH_NEURONX_DEBUG("ERROR: mergeStableHLOModulesWithTensors: Failed to merge modules");
      return result;
    }

    TORCH_NEURONX_DEBUG(
        "mergeStableHLOModulesWithTensors: Successfully merged modules, building tensor mappings");

    // Convert to string representation
    result.merged_mlir_string = stablehlo_utils::moduleToString(*merged_module);

    // Copy the mapping to the result
    result.mapping = mapping;

    // Build the merged tensor vectors based on the mapping information using helper functions
    populateMergedInputsFromMapping(result.merged_inputs, mapping, module1_inputs, module2_inputs);
    populateMergedOutputsFromMapping(result.merged_outputs, mapping, module1_outputs,
                                     module2_outputs);

    result.success = true;

    TORCH_NEURONX_DEBUG("mergeStableHLOModulesWithTensors: Merge completed successfully");
    TORCH_NEURONX_DEBUG("Merged tensors - ", result.merged_inputs.size(), " inputs, ",
                        result.merged_outputs.size(), " outputs");
    TORCH_NEURONX_DEBUG("Total inputs: ", mapping.total_inputs,
                        ", Total outputs: ", mapping.total_outputs);

  } catch (const std::exception& e) {
    TORCH_NEURONX_DEBUG("ERROR: mergeStableHLOModulesWithTensors: Exception during merge: ",
                        e.what());
    result.success = false;
  }

  return result;
}

/**
 * @brief Core skip intermediates merge function - now a thin wrapper around unified mergeModules()
 *
 * This function analyzes dependencies to identify intermediate outputs (outputs consumed as
 * inputs by the other module) and delegates to the unified mergeModules() function with
 * appropriate skip sets.
 *
 * This refactoring eliminates ~200 lines of code duplication while maintaining the same behavior.
 */
mlir::OwningOpRef<mlir::ModuleOp> mergeModulesSkipIntermediates(
    mlir::ModuleOp module1, mlir::ModuleOp module2, mlir::MLIRContext* context,
    const std::vector<void*>& module1_input_addrs, const std::vector<void*>& module1_output_addrs,
    const std::vector<void*>& module2_input_addrs, const std::vector<void*>& module2_output_addrs,
    const DependencyAnalysis& analysis, MergeMapping* mapping, bool verify_output,
    bool run_optimization) {
  TORCH_NEURONX_DEBUG("mergeModulesSkipIntermediates: Building skip sets from dependency analysis");

  // Build skip sets based on dependency analysis
  std::unordered_set<size_t> module1_outputs_to_skip;
  std::unordered_set<size_t> module2_outputs_to_skip;

  if (analysis.module1First) {
    // Module1 executes first, so skip Module1 outputs consumed by Module2
    for (const auto& dep : analysis.module1ToModule2Deps) {
      size_t output_idx = dep.first;
      const std::vector<size_t>& input_indices = dep.second;
      module1_outputs_to_skip.insert(output_idx);

      // Log all consuming input indices (handles duplicate inputs)
      for (size_t input_idx : input_indices) {
        TORCH_NEURONX_DEBUG("Marking Module1 output ", output_idx,
                            " as intermediate (consumed by Module2 input ", input_idx, ")");
      }
    }
  } else {
    // Module2 executes first, so skip Module2 outputs consumed by Module1
    for (const auto& dep : analysis.module2ToModule1Deps) {
      size_t output_idx = dep.first;
      const std::vector<size_t>& input_indices = dep.second;
      module2_outputs_to_skip.insert(output_idx);

      // Log all consuming input indices (handles duplicate inputs)
      for (size_t input_idx : input_indices) {
        TORCH_NEURONX_DEBUG("Marking Module2 output ", output_idx,
                            " as intermediate (consumed by Module1 input ", input_idx, ")");
      }
    }
  }

  TORCH_NEURONX_DEBUG(
      "mergeModulesSkipIntermediates: Skip sets built - Module1: ", module1_outputs_to_skip.size(),
      " outputs, Module2: ", module2_outputs_to_skip.size(), " outputs");

  // Delegate to unified mergeModules() with skip sets
  TORCH_NEURONX_DEBUG("mergeModulesSkipIntermediates: Delegating to unified mergeModules()");
  return mergeModules(module1, module2, context, module1_input_addrs, module1_output_addrs,
                      module2_input_addrs, module2_output_addrs, mapping, verify_output,
                      module1_outputs_to_skip, module2_outputs_to_skip, run_optimization);
}

/**
 * @brief Skip intermediates version of mergeStableHLOModulesWithTensors that treats dependencies as
 * intermediates
 *
 * This skip intermediates version differs from the conservative version in that if MLIR2 uses
 * output of MLIR1, then the used output of MLIR1 is NOT included in the output of the combined
 * MLIR. It is treated as an intermediate value and thus not exposed as an output.
 *
 * The key difference from the conservative version is in how we handle the output types and return
 * values:
 * - We identify which outputs are consumed as inputs by the other module
 * - We exclude those outputs from the final merged function signature
 * - We adjust the output mapping accordingly
 *
 * WARNING: This skips intermediate outputs that might be needed by external consumers.
 * Use only when you're certain that the dependent outputs are not needed externally.
 */
MergeResult mergeStableHLOModulesWithTensorsSkipIntermediates(
    const std::string& mod_str1, const std::string& mod_str2,
    const std::vector<at::Tensor>& module1_inputs, const std::vector<at::Tensor>& module1_outputs,
    const std::vector<at::Tensor>& module2_inputs, const std::vector<at::Tensor>& module2_outputs,
    bool verify_output) {
  TORCH_NEURONX_DEBUG(
      "mergeStableHLOModulesWithTensorsSkipIntermediates: Starting skip intermediates merge with "
      "tensor "
      "mapping");

  MergeResult result;
  result.success = false;

  try {
    // Extract tensor addresses for dependency analysis using helper
    std::vector<void*> module1_input_addrs, module1_output_addrs;
    std::vector<void*> module2_input_addrs, module2_output_addrs;

    extractTensorAddresses(module1_inputs, module1_outputs, module2_inputs, module2_outputs,
                           module1_input_addrs, module1_output_addrs, module2_input_addrs,
                           module2_output_addrs);

    TORCH_NEURONX_DEBUG("mergeStableHLOModulesWithTensorsSkipIntermediates: Extracted ",
                        module1_input_addrs.size(), " + ", module1_output_addrs.size(), " + ",
                        module2_input_addrs.size(), " + ", module2_output_addrs.size(),
                        " tensor addresses");

    // Analyze dependencies to identify which outputs should be treated as intermediates
    auto analysis = analyzeDependencies(module1_input_addrs, module1_output_addrs,
                                        module2_input_addrs, module2_output_addrs);

    TORCH_NEURONX_DEBUG(
        "mergeStableHLOModulesWithTensorsSkipIntermediates: Dependency analysis complete");
    TORCH_NEURONX_DEBUG("Module1->Module2 deps: ", analysis.module1ToModule2Deps.size());
    TORCH_NEURONX_DEBUG("Module2->Module1 deps: ", analysis.module2ToModule1Deps.size());

    // Parse MLIR modules for the skip intermediates merge
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();
    context.getOrLoadDialect<mlir::chlo::ChloDialect>();

    auto mod1 = mlir::parseSourceString<mlir::ModuleOp>(mod_str1, &context);
    auto mod2 = mlir::parseSourceString<mlir::ModuleOp>(mod_str2, &context);

    if (!mod1 || !mod2) {
      TORCH_NEURONX_DEBUG(
          "ERROR: mergeStableHLOModulesWithTensorsSkipIntermediates: Failed to parse modules");
      return result;
    }

    // Get main functions
    auto main1 = mod1->lookupSymbol<mlir::func::FuncOp>("main");
    auto main2 = mod2->lookupSymbol<mlir::func::FuncOp>("main");

    if (!main1 || !main2) {
      TORCH_NEURONX_DEBUG(
          "ERROR: mergeStableHLOModulesWithTensorsSkipIntermediates: Main functions not found");
      return result;
    }

    // Create the skip intermediates merged module using a modified merge approach
    auto skip_intermediates_merged = mergeModulesSkipIntermediates(
        *mod1, *mod2, &context, module1_input_addrs, module1_output_addrs, module2_input_addrs,
        module2_output_addrs, analysis, &result.mapping, verify_output);

    if (!skip_intermediates_merged) {
      TORCH_NEURONX_DEBUG(
          "ERROR: mergeStableHLOModulesWithTensorsSkipIntermediates: Failed to merge modules");
      return result;
    }

    // Convert to string
    result.merged_mlir_string = stablehlo_utils::moduleToString(*skip_intermediates_merged);

    // Build the merged tensor vectors based on the mapping information using helper functions
    populateMergedInputsFromMapping(result.merged_inputs, result.mapping, module1_inputs,
                                    module2_inputs);
    populateMergedOutputsFromMapping(result.merged_outputs, result.mapping, module1_outputs,
                                     module2_outputs);

    result.success = true;
    TORCH_NEURONX_DEBUG(
        "mergeStableHLOModulesWithTensorsSkipIntermediates: Skip intermediates merge completed "
        "successfully");
    TORCH_NEURONX_DEBUG("Merged tensors (unsafe) - ", result.merged_inputs.size(), " inputs, ",
                        result.merged_outputs.size(), " outputs");
    TORCH_NEURONX_DEBUG("Total inputs: ", result.mapping.total_inputs,
                        ", Total outputs: ", result.mapping.total_outputs);

  } catch (const std::exception& e) {
    TORCH_NEURONX_DEBUG(
        "ERROR: mergeStableHLOModulesWithTensorsSkipIntermediates: Exception during merge: ",
        e.what());
    result.success = false;
  }

  return result;
}

std::unique_ptr<StableHloNode> mergeStableHLOModulesWithTensorsSkipIntermediates(
    StableHloNode* node1, StableHloNode* node2, mlir::MLIRContext* context) {
  TORCH_NEURONX_TRACE_FUNCTION_WITH_ARGS("node1:", (node1 ? node1->op_name : "null"),
                                         "node2:", (node2 ? node2->op_name : "null"));

  TORCH_NEURONX_DEBUG("Starting unsafe skip intermediates merge for StableHloNodes",
                      "node1_op:", (node1 ? node1->op_name : "null"),
                      "node1_cache_key:", (node1 ? node1->cache_key : "null"),
                      "node1_inputs:", (node1 ? node1->inputs.size() : 0),
                      "node1_outputs:", (node1 ? node1->outputs.size() : 0),
                      "node2_op:", (node2 ? node2->op_name : "null"),
                      "node2_cache_key:", (node2 ? node2->cache_key : "null"),
                      "node2_inputs:", (node2 ? node2->inputs.size() : 0),
                      "node2_outputs:", (node2 ? node2->outputs.size() : 0),
                      "merge_type:", "unsafe_skip_intermediates");

  try {
    if (!node1 || !node2) {
      TORCH_NEURONX_WARN("Unsafe merge failed - null node provided",
                         "node1_valid:", (node1 ? "true" : "false"),
                         "node2_valid:", (node2 ? "true" : "false"));
      return std::unique_ptr<StableHloNode>();
    }

    const std::vector<void*>& module1_input_addrs = node1->inputs;
    const std::vector<void*>& module2_input_addrs = node2->inputs;
    const std::vector<void*>& module1_output_addrs = node1->outputs;
    const std::vector<void*>& module2_output_addrs = node2->outputs;

    auto& mod1 = *(node1->module);
    auto& mod2 = *(node2->module);

    if (!mod1 || !mod2) {
      TORCH_NEURONX_WARN(
          "WRAN: mergeStableHLOModulesWithTensorsSkipIntermediates: Failed to parse modules");
      return std::unique_ptr<StableHloNode>();
    }

    // Get main functions
    auto main1 = mod1->lookupSymbol<mlir::func::FuncOp>("main");
    auto main2 = mod2->lookupSymbol<mlir::func::FuncOp>("main");

    if (!main1 || !main2) {
      TORCH_NEURONX_DEBUG(
          "ERROR: mergeStableHLOModulesWithTensorsSkipIntermediates: Main functions not found");
      return std::unique_ptr<StableHloNode>();
    }

    // Analyze dependencies between modules
    auto analysis = analyzeDependencies(module1_input_addrs, module1_output_addrs,
                                        module2_input_addrs, module2_output_addrs);

    MergeMapping mapping;

    // Create the skip intermediates merged module using a modified merge approach
    auto skip_intermediates_merged = mergeModulesSkipIntermediates(
        *mod1, *mod2, context, module1_input_addrs, module1_output_addrs, module2_input_addrs,
        module2_output_addrs, analysis, &mapping);

    if (!skip_intermediates_merged) {
      TORCH_NEURONX_DEBUG(
          "ERROR: mergeStableHLOModulesWithTensorsSkipIntermediates: Failed to merge modules");
      return std::unique_ptr<StableHloNode>();
    }

    std::vector<void*> input_addrs;
    std::vector<void*> output_addrs;

    // Build tensor address arrays based on unsafe merge mapping (excludes intermediates)
    for (int i = 0; i < mapping.total_inputs; i++) {
      auto& mapping_entry = mapping.input_mapping[i];
      if (mapping_entry.first == 1) {
        // Input comes from node1
        input_addrs.push_back(node1->inputs[mapping_entry.second]);
      } else {
        // Input comes from node2
        input_addrs.push_back(node2->inputs[mapping_entry.second]);
      }
    }

    for (int i = 0; i < mapping.total_outputs; i++) {
      auto& mapping_entry = mapping.output_mapping[i];
      if (mapping_entry.first == 1) {
        // Output goes to node1's original output (non-intermediate only)
        output_addrs.push_back(node1->outputs[mapping_entry.second]);
      } else {
        // Output goes to node2's original output (non-intermediate only)
        output_addrs.push_back(node2->outputs[mapping_entry.second]);
      }
    }

    auto shared_module =
        std::make_shared<mlir::OwningOpRef<mlir::ModuleOp>>(std::move(skip_intermediates_merged));
    auto concat_node = std::make_unique<StableHloNode>(
        "", "", shared_module, std::move(input_addrs), std::move(output_addrs), false);

    // TODO: Add set merge mapping
    return concat_node;

  } catch (const std::exception& e) {
    TORCH_NEURONX_DEBUG(
        "ERROR: mergeStableHLOModulesWithTensorsSkipIntermediates: Exception during merge: ",
        e.what());
    return std::unique_ptr<StableHloNode>();
  }
}

/**
 * @brief Enhanced interface that returns both merged MLIR and correct tensor mappings
 *
 * This function builds the correct input/output tensor sets based on the MergeMapping
 * information from the dependency analysis. It ensures that the tensor sets match
 * what the merged MLIR expects.
 *
 * IMPORTANT: This function converts the merged MLIR to string format to avoid
 * context lifetime issues. The returned MergeResult contains a string representation
 * rather than a live MLIR module to prevent segmentation faults.
 */
std::unique_ptr<StableHloNode> mergeStableHLOModulesWithTensors(StableHloNode* node1,
                                                                StableHloNode* node2,
                                                                mlir::MLIRContext* context) {
  TORCH_NEURONX_TRACE_FUNCTION_WITH_ARGS("node1:", (node1 ? node1->op_name : "null"),
                                         "node2:", (node2 ? node2->op_name : "null"));

  TORCH_NEURONX_DEBUG("Starting safe enhanced merge for StableHloNodes",
                      "node1_op:", (node1 ? node1->op_name : "null"),
                      "node1_cache_key:", (node1 ? node1->cache_key : "null"),
                      "node1_inputs:", (node1 ? node1->inputs.size() : 0),
                      "node1_outputs:", (node1 ? node1->outputs.size() : 0),
                      "node2_op:", (node2 ? node2->op_name : "null"),
                      "node2_cache_key:", (node2 ? node2->cache_key : "null"),
                      "node2_inputs:", (node2 ? node2->inputs.size() : 0),
                      "node2_outputs:", (node2 ? node2->outputs.size() : 0),
                      "merge_type:", "safe_enhanced_merge");

  if (!node1 || !node2) {
    TORCH_NEURONX_WARN("Safe merge failed - null node provided",
                       "node1_valid:", (node1 ? "true" : "false"),
                       "node2_valid:", (node2 ? "true" : "false"));
    return std::unique_ptr<StableHloNode>();
  }

  const std::vector<void*>& module1_input_addrs = node1->inputs;
  const std::vector<void*>& module2_input_addrs = node2->inputs;
  const std::vector<void*>& module1_output_addrs = node1->outputs;
  const std::vector<void*>& module2_output_addrs = node2->outputs;

  auto& mod1 = *(node1->module);
  auto& mod2 = *(node2->module);

  if (!mod1 || !mod2) {
    TORCH_NEURONX_WARN("Safe merge failed - invalid MLIR modules in nodes",
                       "node1_module_valid:", (mod1 ? "true" : "false"),
                       "node2_module_valid:", (mod2 ? "true" : "false"),
                       "error_type:", "invalid_mlir_modules");
    return std::unique_ptr<StableHloNode>();
  }

  // Get main functions
  auto main1 = mod1->lookupSymbol<mlir::func::FuncOp>("main");
  auto main2 = mod2->lookupSymbol<mlir::func::FuncOp>("main");

  if (!main1 || !main2) {
    TORCH_NEURONX_DEBUG("ERROR: mergeStableHLOModulesWithTensors: Main functions not found");
    return std::unique_ptr<StableHloNode>();
  }

  auto mapping = std::make_unique<MergeMapping>();

  // Create the skip intermediates merged module using a modified merge approach
  auto merged = mergeModules(*mod1, *mod2, context, module1_input_addrs, module1_output_addrs,
                             module2_input_addrs, module2_output_addrs, mapping.get());

  if (!merged) {
    TORCH_NEURONX_DEBUG("ERROR: mergeStableHLOModulesWithTensors: Failed to merge modules");
    return std::unique_ptr<StableHloNode>();
  }

  std::vector<void*> input_addrs;
  std::vector<void*> output_addrs;

  // Build tensor address arrays based on safe merge mapping (preserves all outputs)
  for (int i = 0; i < mapping->total_inputs; i++) {
    auto& mapping_entry = mapping->input_mapping[i];
    if (mapping_entry.first == 1) {
      // Input comes from node1
      input_addrs.push_back(node1->inputs[mapping_entry.second]);
    } else {
      // Input comes from node2
      input_addrs.push_back(node2->inputs[mapping_entry.second]);
    }
  }

  for (int i = 0; i < mapping->total_outputs; i++) {
    auto& mapping_entry = mapping->output_mapping[i];
    if (mapping_entry.first == 1) {
      // Output goes to node1's original output (all outputs preserved)
      output_addrs.push_back(node1->outputs[mapping_entry.second]);
    } else {
      // Output goes to node2's original output (all outputs preserved)
      output_addrs.push_back(node2->outputs[mapping_entry.second]);
    }
  }

  // Build TensorDataRef and TensorContext using merge mapping from original nodes
  std::vector<at::neuron::TensorDataRef> input_data_refs;
  std::vector<at::neuron::TensorDataRef> output_data_refs;
  std::vector<at::neuron::TensorContext> input_contexts;
  std::vector<at::neuron::TensorContext> output_contexts;

  for (int i = 0; i < mapping->total_inputs; i++) {
    auto& mapping_entry = mapping->input_mapping[i];
    if (mapping_entry.first == 1) {
      // Input comes from node1
      if (mapping_entry.second < node1->input_data_refs.size()) {
        input_data_refs.push_back(node1->input_data_refs[mapping_entry.second]);
      }
      if (mapping_entry.second < node1->input_contexts.size()) {
        input_contexts.push_back(node1->input_contexts[mapping_entry.second]);
      }
    } else {
      // Input comes from node2
      if (mapping_entry.second < node2->input_data_refs.size()) {
        input_data_refs.push_back(node2->input_data_refs[mapping_entry.second]);
      }
      if (mapping_entry.second < node2->input_contexts.size()) {
        input_contexts.push_back(node2->input_contexts[mapping_entry.second]);
      }
    }
  }

  for (int i = 0; i < mapping->total_outputs; i++) {
    auto& mapping_entry = mapping->output_mapping[i];
    if (mapping_entry.first == 1) {
      // Output goes to node1's original output
      if (mapping_entry.second < node1->output_data_refs.size()) {
        output_data_refs.push_back(node1->output_data_refs[mapping_entry.second]);
      }
      if (mapping_entry.second < node1->output_contexts.size()) {
        output_contexts.push_back(node1->output_contexts[mapping_entry.second]);
      }
    } else {
      // Output goes to node2's original output
      if (mapping_entry.second < node2->output_data_refs.size()) {
        output_data_refs.push_back(node2->output_data_refs[mapping_entry.second]);
      }
      if (mapping_entry.second < node2->output_contexts.size()) {
        output_contexts.push_back(node2->output_contexts[mapping_entry.second]);
      }
    }
  }

  auto shared_module = std::make_shared<mlir::OwningOpRef<mlir::ModuleOp>>(std::move(merged));
  auto concat_node = std::make_unique<StableHloNode>("", "", shared_module, std::move(input_addrs),
                                                     std::move(output_addrs), false);
  // Populate TensorDataRef and TensorContext
  concat_node->input_data_refs = std::move(input_data_refs);
  concat_node->output_data_refs = std::move(output_data_refs);
  concat_node->input_contexts = std::move(input_contexts);
  concat_node->output_contexts = std::move(output_contexts);

  concat_node->SetMergeMapping(std::move(mapping));
  return concat_node;
}

// ============================================================================
// In-Order Module Merging Implementation
// ============================================================================

/**
 * @brief Simplified dependency analysis for in-order module merging
 *
 * This is an optimized version of analyzeDependencies that assumes module1
 * always executes before module2, eliminating bidirectional dependency
 * checks and circular dependency detection.
 */
DependencyAnalysis analyzeDependenciesInOrder(const std::vector<void*>& module1_input_addrs,
                                              const std::vector<void*>& module1_output_addrs,
                                              const std::vector<void*>& module2_input_addrs,
                                              const std::vector<void*>& module2_output_addrs) {
  TORCH_NEURONX_TRACE_FUNCTION_WITH_ARGS(
      "mod1_inputs:", module1_input_addrs.size(), "mod1_outputs:", module1_output_addrs.size(),
      "mod2_inputs:", module2_input_addrs.size(), "mod2_outputs:", module2_output_addrs.size());

  // Input validation
  if (module1_input_addrs.empty() && module1_output_addrs.empty()) {
    TORCH_NEURONX_ERROR("Module1 cannot have both empty inputs and outputs");
    throw std::invalid_argument("Module1 cannot have both empty inputs and outputs");
  }

  if (module2_input_addrs.empty() && module2_output_addrs.empty()) {
    TORCH_NEURONX_ERROR("Module2 cannot have both empty inputs and outputs");
    throw std::invalid_argument("Module2 cannot have both empty inputs and outputs");
  }

  // Check for null pointers in addresses
  for (size_t i = 0; i < module1_input_addrs.size(); ++i) {
    if (!module1_input_addrs[i]) {
      throw std::invalid_argument("Module1 input address at index " + std::to_string(i) +
                                  " is null");
    }
  }

  for (size_t i = 0; i < module2_input_addrs.size(); ++i) {
    if (!module2_input_addrs[i]) {
      throw std::invalid_argument("Module2 input address at index " + std::to_string(i) +
                                  " is null");
    }
  }

  DependencyAnalysis analysis;

  TORCH_NEURONX_DEBUG(
      "analyzeDependenciesInOrder: Starting simplified in-order dependency analysis");

  // In-order optimization: Only check module1 outputs -> module2 inputs
  // No need to check module2 outputs -> module1 inputs (impossible by construction)
  for (size_t i = 0; i < module1_output_addrs.size(); ++i) {
    for (size_t j = 0; j < module2_input_addrs.size(); ++j) {
      if (module1_output_addrs[i] == module2_input_addrs[j]) {
        analysis.module1ToModule2Deps[i].push_back(j);
        TORCH_NEURONX_DEBUG("Found dependency: module1_output[", i, "] -> module2_input[", j, "]");
      }
    }
  }

  // Track which inputs are dependency targets for common input filtering
  std::unordered_set<size_t> module2_dependency_targets;
  for (const auto& dep : analysis.module1ToModule2Deps) {
    for (size_t target_idx : dep.second) {
      module2_dependency_targets.insert(target_idx);
    }
  }

  // Find common inputs, EXCLUDING dependency targets
  for (size_t i = 0; i < module1_input_addrs.size(); ++i) {
    for (size_t j = 0; j < module2_input_addrs.size(); ++j) {
      // Skip if this input is a dependency target
      if (module2_dependency_targets.count(j) > 0) {
        continue;
      }

      if (module1_input_addrs[i] == module2_input_addrs[j]) {
        analysis.commonInputs[i] = j;
        TORCH_NEURONX_DEBUG("Found common input: module1[", i, "] == module2[", j, "]");
      }
    }
  }

  // In-order: module1 always executes first
  analysis.module1First = true;
  TORCH_NEURONX_DEBUG("In-order merge: Module1 executes first (by definition)");

  // Determine scenario type
  bool hasCommonInputs = !analysis.commonInputs.empty();
  bool hasDependencies = !analysis.module1ToModule2Deps.empty();

  if (hasCommonInputs && hasDependencies) {
    analysis.scenario = DependencyAnalysis::MIXED;
    TORCH_NEURONX_DEBUG("Detected MIXED scenario (common inputs + dependencies)");
  } else if (hasDependencies) {
    analysis.scenario = DependencyAnalysis::DIRECT_DEPS;
    TORCH_NEURONX_DEBUG("Detected DIRECT_DEPS scenario");
  } else if (hasCommonInputs) {
    analysis.scenario = DependencyAnalysis::COMMON_INPUTS;
    TORCH_NEURONX_DEBUG("Detected COMMON_INPUTS scenario");
  } else {
    analysis.scenario = DependencyAnalysis::INDEPENDENT;
    TORCH_NEURONX_DEBUG("Detected INDEPENDENT scenario");
  }

  return analysis;
}

/**
 * @brief Low-level in-order module merge implementation
 *
 * Core merge function that assumes module1 executes before module2.
 * Uses analyzeDependenciesInOrder for simplified dependency analysis,
 * then calls mergeModulesWithAnalysis to avoid re-analyzing dependencies.
 */
mlir::OwningOpRef<mlir::ModuleOp> mergeModulesInOrder(
    mlir::ModuleOp module1, mlir::ModuleOp module2, mlir::MLIRContext* context,
    const std::vector<void*>& module1_input_addrs, const std::vector<void*>& module1_output_addrs,
    const std::vector<void*>& module2_input_addrs, const std::vector<void*>& module2_output_addrs,
    MergeMapping* mapping, bool verify_output,
    const std::unordered_set<size_t>& module1_outputs_to_skip,
    const std::unordered_set<size_t>& module2_outputs_to_skip) {
  TORCH_NEURONX_TRACE_FUNCTION_WITH_ARGS(
      "mod1_inputs:", module1_input_addrs.size(), "mod1_outputs:", module1_output_addrs.size(),
      "mod2_inputs:", module2_input_addrs.size(), "mod2_outputs:", module2_output_addrs.size());
  TORCH_NEURONX_DEBUG(
      "mergeModulesInOrder: Starting in-order module merge (module1 before module2)");

  // Validate context parameter
  if (context == nullptr) {
    throw std::runtime_error("MLIRContext cannot be null");
  }

  // Use simplified in-order dependency analysis (skips reverse dependency check)
  auto analysis = analyzeDependenciesInOrder(module1_input_addrs, module1_output_addrs,
                                             module2_input_addrs, module2_output_addrs);

  // Delegate to mergeModulesWithAnalysis to avoid re-analyzing dependencies
  return mergeModulesWithAnalysis(module1, module2, context, module1_input_addrs,
                                  module1_output_addrs, module2_input_addrs, module2_output_addrs,
                                  analysis, mapping, verify_output, module1_outputs_to_skip,
                                  module2_outputs_to_skip);
}

/**
 * @brief High-level in-order merge with tensor management
 */
MergeResult mergeStableHLOModulesWithTensorsInOrder(const std::string& mod_str1,
                                                    const std::string& mod_str2,
                                                    const std::vector<at::Tensor>& module1_inputs,
                                                    const std::vector<at::Tensor>& module1_outputs,
                                                    const std::vector<at::Tensor>& module2_inputs,
                                                    const std::vector<at::Tensor>& module2_outputs,
                                                    bool verify_output) {
  TORCH_NEURONX_DEBUG("mergeStableHLOModulesWithTensorsInOrder: Starting in-order merge");

  MergeResult result;
  result.success = false;

  try {
    // Extract tensor addresses
    std::vector<void*> module1_input_addrs, module1_output_addrs;
    std::vector<void*> module2_input_addrs, module2_output_addrs;

    extractTensorAddresses(module1_inputs, module1_outputs, module2_inputs, module2_outputs,
                           module1_input_addrs, module1_output_addrs, module2_input_addrs,
                           module2_output_addrs);

    // Parse MLIR modules
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();
    context.getOrLoadDialect<mlir::chlo::ChloDialect>();

    auto mod1 = mlir::parseSourceString<mlir::ModuleOp>(mod_str1, &context);
    auto mod2 = mlir::parseSourceString<mlir::ModuleOp>(mod_str2, &context);

    if (!mod1 || !mod2) {
      TORCH_NEURONX_DEBUG(
          "ERROR: mergeStableHLOModulesWithTensorsInOrder: Failed to parse modules");
      return result;
    }

    // Use in-order merge
    MergeMapping mapping;
    auto merged_module =
        mergeModulesInOrder(*mod1, *mod2, &context, module1_input_addrs, module1_output_addrs,
                            module2_input_addrs, module2_output_addrs, &mapping, verify_output);

    if (!merged_module) {
      TORCH_NEURONX_DEBUG(
          "ERROR: mergeStableHLOModulesWithTensorsInOrder: Failed to merge modules");
      return result;
    }

    // Convert to string
    result.merged_mlir_string = stablehlo_utils::moduleToString(*merged_module);
    result.mapping = mapping;

    // Build merged tensor vectors
    populateMergedInputsFromMapping(result.merged_inputs, mapping, module1_inputs, module2_inputs);
    populateMergedOutputsFromMapping(result.merged_outputs, mapping, module1_outputs,
                                     module2_outputs);

    result.success = true;
    TORCH_NEURONX_DEBUG("mergeStableHLOModulesWithTensorsInOrder: Merge completed successfully");

  } catch (const std::exception& e) {
    TORCH_NEURONX_DEBUG("ERROR: mergeStableHLOModulesWithTensorsInOrder: Exception: ", e.what());
    result.success = false;
  }

  return result;
}

/**
 * @brief In-order merge with intermediate output skipping
 */
MergeResult mergeStableHLOModulesWithTensorsInOrderSkipIntermediates(
    const std::string& mod_str1, const std::string& mod_str2,
    const std::vector<at::Tensor>& module1_inputs, const std::vector<at::Tensor>& module1_outputs,
    const std::vector<at::Tensor>& module2_inputs, const std::vector<at::Tensor>& module2_outputs,
    bool verify_output) {
  TORCH_NEURONX_DEBUG("mergeStableHLOModulesWithTensorsInOrderSkipIntermediates: Starting");

  MergeResult result;
  result.success = false;

  try {
    // Extract tensor addresses
    std::vector<void*> module1_input_addrs, module1_output_addrs;
    std::vector<void*> module2_input_addrs, module2_output_addrs;

    extractTensorAddresses(module1_inputs, module1_outputs, module2_inputs, module2_outputs,
                           module1_input_addrs, module1_output_addrs, module2_input_addrs,
                           module2_output_addrs);

    // Use in-order dependency analysis
    auto analysis = analyzeDependenciesInOrder(module1_input_addrs, module1_output_addrs,
                                               module2_input_addrs, module2_output_addrs);

    // Build skip sets based on dependencies (module1 outputs consumed by module2)
    std::unordered_set<size_t> module1_outputs_to_skip;
    for (const auto& dep : analysis.module1ToModule2Deps) {
      module1_outputs_to_skip.insert(dep.first);
      TORCH_NEURONX_DEBUG("Marking Module1 output ", dep.first, " as intermediate");
    }

    // Parse MLIR modules
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();
    context.getOrLoadDialect<mlir::chlo::ChloDialect>();

    auto mod1 = mlir::parseSourceString<mlir::ModuleOp>(mod_str1, &context);
    auto mod2 = mlir::parseSourceString<mlir::ModuleOp>(mod_str2, &context);

    if (!mod1 || !mod2) {
      TORCH_NEURONX_DEBUG(
          "ERROR: mergeStableHLOModulesWithTensorsInOrderSkipIntermediates: Failed to parse");
      return result;
    }

    // Merge with skip sets
    MergeMapping mapping;
    auto merged_module = mergeModulesInOrder(
        *mod1, *mod2, &context, module1_input_addrs, module1_output_addrs, module2_input_addrs,
        module2_output_addrs, &mapping, verify_output, module1_outputs_to_skip, {});

    if (!merged_module) {
      TORCH_NEURONX_DEBUG(
          "ERROR: mergeStableHLOModulesWithTensorsInOrderSkipIntermediates: Merge failed");
      return result;
    }

    // Convert to string
    result.merged_mlir_string = stablehlo_utils::moduleToString(*merged_module);
    result.mapping = mapping;

    // Build merged tensor vectors
    populateMergedInputsFromMapping(result.merged_inputs, mapping, module1_inputs, module2_inputs);
    populateMergedOutputsFromMapping(result.merged_outputs, mapping, module1_outputs,
                                     module2_outputs);

    result.success = true;
    TORCH_NEURONX_DEBUG("mergeStableHLOModulesWithTensorsInOrderSkipIntermediates: Complete");

  } catch (const std::exception& e) {
    TORCH_NEURONX_DEBUG(
        "ERROR: mergeStableHLOModulesWithTensorsInOrderSkipIntermediates: Exception: ", e.what());
    result.success = false;
  }

  return result;
}

/**
 * @brief StableHloNode version of in-order merge preserving all outputs
 */
std::unique_ptr<StableHloNode> mergeStableHLOModulesWithTensorsInOrder(StableHloNode* node1,
                                                                       StableHloNode* node2,
                                                                       mlir::MLIRContext* context) {
  TORCH_NEURONX_TRACE_FUNCTION_WITH_ARGS("node1:", (node1 ? node1->op_name : "null"),
                                         "node2:", (node2 ? node2->op_name : "null"));

  TORCH_NEURONX_DEBUG("Starting in-order merge for StableHloNodes (preserving all outputs)");

  if (!node1 || !node2) {
    TORCH_NEURONX_WARN("In-order merge failed - null node provided");
    return std::unique_ptr<StableHloNode>();
  }

  const std::vector<void*>& module1_input_addrs = node1->inputs;
  const std::vector<void*>& module2_input_addrs = node2->inputs;
  const std::vector<void*>& module1_output_addrs = node1->outputs;
  const std::vector<void*>& module2_output_addrs = node2->outputs;

  auto& mod1 = *(node1->module);
  auto& mod2 = *(node2->module);

  if (!mod1 || !mod2) {
    TORCH_NEURONX_WARN("In-order merge failed - invalid MLIR modules");
    return std::unique_ptr<StableHloNode>();
  }

  auto main1 = mod1->lookupSymbol<mlir::func::FuncOp>("main");
  auto main2 = mod2->lookupSymbol<mlir::func::FuncOp>("main");

  if (!main1 || !main2) {
    TORCH_NEURONX_DEBUG("ERROR: mergeStableHLOModulesWithTensorsInOrder: Main functions not found");
    return std::unique_ptr<StableHloNode>();
  }

  auto mapping = std::make_unique<MergeMapping>();

  // Use in-order merge
  auto merged =
      mergeModulesInOrder(*mod1, *mod2, context, module1_input_addrs, module1_output_addrs,
                          module2_input_addrs, module2_output_addrs, mapping.get());

  if (!merged) {
    TORCH_NEURONX_DEBUG("ERROR: mergeStableHLOModulesWithTensorsInOrder: Merge failed");
    return std::unique_ptr<StableHloNode>();
  }

  // Build tensor address arrays
  std::vector<void*> input_addrs;
  std::vector<void*> output_addrs;

  for (size_t i = 0; i < mapping->total_inputs; i++) {
    auto& mapping_entry = mapping->input_mapping[i];
    if (mapping_entry.first == 1) {
      input_addrs.push_back(node1->inputs[mapping_entry.second]);
    } else {
      input_addrs.push_back(node2->inputs[mapping_entry.second]);
    }
  }

  for (size_t i = 0; i < mapping->total_outputs; i++) {
    auto& mapping_entry = mapping->output_mapping[i];
    if (mapping_entry.first == 1) {
      output_addrs.push_back(node1->outputs[mapping_entry.second]);
    } else {
      output_addrs.push_back(node2->outputs[mapping_entry.second]);
    }
  }

  // Build TensorDataRef and TensorContext
  std::vector<at::neuron::TensorDataRef> input_data_refs;
  std::vector<at::neuron::TensorDataRef> output_data_refs;
  std::vector<at::neuron::TensorContext> input_contexts;
  std::vector<at::neuron::TensorContext> output_contexts;

  for (size_t i = 0; i < mapping->total_inputs; i++) {
    auto& mapping_entry = mapping->input_mapping[i];
    if (mapping_entry.first == 1) {
      if (mapping_entry.second < node1->input_data_refs.size()) {
        input_data_refs.push_back(node1->input_data_refs[mapping_entry.second]);
      }
      if (mapping_entry.second < node1->input_contexts.size()) {
        input_contexts.push_back(node1->input_contexts[mapping_entry.second]);
      }
    } else {
      if (mapping_entry.second < node2->input_data_refs.size()) {
        input_data_refs.push_back(node2->input_data_refs[mapping_entry.second]);
      }
      if (mapping_entry.second < node2->input_contexts.size()) {
        input_contexts.push_back(node2->input_contexts[mapping_entry.second]);
      }
    }
  }

  for (size_t i = 0; i < mapping->total_outputs; i++) {
    auto& mapping_entry = mapping->output_mapping[i];
    if (mapping_entry.first == 1) {
      if (mapping_entry.second < node1->output_data_refs.size()) {
        output_data_refs.push_back(node1->output_data_refs[mapping_entry.second]);
      }
      if (mapping_entry.second < node1->output_contexts.size()) {
        output_contexts.push_back(node1->output_contexts[mapping_entry.second]);
      }
    } else {
      if (mapping_entry.second < node2->output_data_refs.size()) {
        output_data_refs.push_back(node2->output_data_refs[mapping_entry.second]);
      }
      if (mapping_entry.second < node2->output_contexts.size()) {
        output_contexts.push_back(node2->output_contexts[mapping_entry.second]);
      }
    }
  }

  auto shared_module = std::make_shared<mlir::OwningOpRef<mlir::ModuleOp>>(std::move(merged));
  auto concat_node = std::make_unique<StableHloNode>("", "", shared_module, std::move(input_addrs),
                                                     std::move(output_addrs), false);

  concat_node->input_data_refs = std::move(input_data_refs);
  concat_node->output_data_refs = std::move(output_data_refs);
  concat_node->input_contexts = std::move(input_contexts);
  concat_node->output_contexts = std::move(output_contexts);
  concat_node->SetMergeMapping(std::move(mapping));

  return concat_node;
}

/**
 * @brief StableHloNode version of in-order merge with intermediate skipping
 */
std::unique_ptr<StableHloNode> mergeStableHLOModulesWithTensorsInOrderSkipIntermediates(
    StableHloNode* node1, StableHloNode* node2, mlir::MLIRContext* context) {
  TORCH_NEURONX_TRACE_FUNCTION_WITH_ARGS("node1:", (node1 ? node1->op_name : "null"),
                                         "node2:", (node2 ? node2->op_name : "null"));

  TORCH_NEURONX_DEBUG("Starting in-order merge with skip intermediates for StableHloNodes");

  if (!node1 || !node2) {
    TORCH_NEURONX_WARN("In-order skip intermediates merge failed - null node provided");
    return std::unique_ptr<StableHloNode>();
  }

  const std::vector<void*>& module1_input_addrs = node1->inputs;
  const std::vector<void*>& module2_input_addrs = node2->inputs;
  const std::vector<void*>& module1_output_addrs = node1->outputs;
  const std::vector<void*>& module2_output_addrs = node2->outputs;

  auto& mod1 = *(node1->module);
  auto& mod2 = *(node2->module);

  if (!mod1 || !mod2) {
    TORCH_NEURONX_WARN("In-order skip intermediates merge failed - invalid MLIR modules");
    return std::unique_ptr<StableHloNode>();
  }

  auto main1 = mod1->lookupSymbol<mlir::func::FuncOp>("main");
  auto main2 = mod2->lookupSymbol<mlir::func::FuncOp>("main");

  if (!main1 || !main2) {
    TORCH_NEURONX_DEBUG(
        "ERROR: mergeStableHLOModulesWithTensorsInOrderSkipIntermediates: Main functions not "
        "found");
    return std::unique_ptr<StableHloNode>();
  }

  // Use in-order dependency analysis
  auto analysis = analyzeDependenciesInOrder(module1_input_addrs, module1_output_addrs,
                                             module2_input_addrs, module2_output_addrs);

  // Build skip set for intermediate outputs
  std::unordered_set<size_t> module1_outputs_to_skip;
  for (const auto& dep : analysis.module1ToModule2Deps) {
    module1_outputs_to_skip.insert(dep.first);
  }

  auto mapping = std::make_unique<MergeMapping>();

  // Merge with skip sets
  auto merged = mergeModulesInOrder(*mod1, *mod2, context, module1_input_addrs,
                                    module1_output_addrs, module2_input_addrs, module2_output_addrs,
                                    mapping.get(), true, module1_outputs_to_skip, {});

  if (!merged) {
    TORCH_NEURONX_DEBUG(
        "ERROR: mergeStableHLOModulesWithTensorsInOrderSkipIntermediates: Merge failed");
    return std::unique_ptr<StableHloNode>();
  }

  // Build tensor address arrays
  std::vector<void*> input_addrs;
  std::vector<void*> output_addrs;

  for (size_t i = 0; i < mapping->total_inputs; i++) {
    auto& mapping_entry = mapping->input_mapping[i];
    if (mapping_entry.first == 1) {
      input_addrs.push_back(node1->inputs[mapping_entry.second]);
    } else {
      input_addrs.push_back(node2->inputs[mapping_entry.second]);
    }
  }

  for (size_t i = 0; i < mapping->total_outputs; i++) {
    auto& mapping_entry = mapping->output_mapping[i];
    if (mapping_entry.first == 1) {
      output_addrs.push_back(node1->outputs[mapping_entry.second]);
    } else {
      output_addrs.push_back(node2->outputs[mapping_entry.second]);
    }
  }

  auto shared_module = std::make_shared<mlir::OwningOpRef<mlir::ModuleOp>>(std::move(merged));
  auto concat_node = std::make_unique<StableHloNode>("", "", shared_module, std::move(input_addrs),
                                                     std::move(output_addrs), false);

  return concat_node;
}

mlir::OwningOpRef<mlir::ModuleOp> parseMLIRModule(const std::vector<uint8_t>& hlo_bytes,
                                                  mlir::MLIRContext& context) {
  TORCH_NEURONX_TRACE_FUNCTION_WITH_ARGS("hlo_size:", hlo_bytes.size());

  // Check for empty input
  if (hlo_bytes.empty()) {
    TORCH_NEURONX_WARN("parseMLIRModule: Empty input data");
    std::cerr << "ERROR: parseMLIRModule: Empty input data" << std::endl;
    return nullptr;
  }

  // Convert bytes to string for parsing
  std::string mlir_text(hlo_bytes.begin(), hlo_bytes.end());

  // Try to parse as MLIR text first
  mlir::OwningOpRef<mlir::ModuleOp> module;

  // Check if this looks like MLIR text (starts with "module" or contains "stablehlo")
  if (mlir_text.find("module") != std::string::npos ||
      mlir_text.find("stablehlo") != std::string::npos) {
    TORCH_NEURONX_DEBUG("Parsing as MLIR text format");
    // Parse as MLIR text
    module = mlir::parseSourceString<mlir::ModuleOp>(mlir_text, &context);
  } else {
    TORCH_NEURONX_DEBUG("Parsing as MLIR bytecode format");
    // Try to parse as bytecode
    llvm::StringRef data(reinterpret_cast<const char*>(hlo_bytes.data()), hlo_bytes.size());
    module = mlir::parseSourceString<mlir::ModuleOp>(data, &context);
  }

  // Verify the parsed module if it exists
  if (module && mlir::failed(mlir::verify(*module))) {
    TORCH_NEURONX_WARN("parseMLIRModule: Parsed module failed MLIR verification");
    std::cerr << "ERROR: parseMLIRModule: Parsed module failed MLIR verification" << std::endl;
    return nullptr;
  }

  if (module) {
    TORCH_NEURONX_DEBUG("Successfully parsed and verified MLIR module");
  } else {
    TORCH_NEURONX_WARN("Failed to parse MLIR module");
  }

  return module;
}

}  // namespace torch_neuronx
