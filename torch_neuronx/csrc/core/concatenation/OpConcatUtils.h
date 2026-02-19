#ifndef TORCH_NEURONX_CSRC_CORE_CONCATENATION_OPCONCATUTILS_H_
#define TORCH_NEURONX_CSRC_CORE_CONCATENATION_OPCONCATUTILS_H_

/**
 * @file OpConcatUtils.h
 * @brief MLIR StableHLO Module Concatenation Utilities
 *
 * RECOMMENDED API USAGE:
 * ======================
 *
 * For most use cases, prefer these high-level functions that handle tensor management:
 *
 * 1. mergeStableHLOModulesWithTensors() - RECOMMENDED for general use
 *    - Automatically handles tensor address analysis and mapping
 *    - Preserves ALL outputs from both modules (conservative approach)
 *    - Safe for cases where you need to access intermediate results
 *    - Returns MergeResult with properly ordered input/output tensors
 *
 * 2. mergeStableHLOModulesWithTensorsSkipIntermediates() - Use with caution
 *    - Excludes intermediate outputs (outputs consumed as inputs by other module)
 *    - More efficient but may hide outputs needed by external consumers
 *    - Only use when certain that intermediate outputs are not needed
 *    - Returns smaller output tensor list
 *
 * Lower-level APIs (use only if you need direct MLIR manipulation):
 *
 * 3. mergeStableHLOModules() - Returns merged MLIR as string
 *    - Requires manual tensor address extraction
 *    - No automatic tensor mapping
 *    - Use for custom MLIR processing pipelines
 *
 * 4. analyzeDependencies() - Analyzes tensor address relationships
 *    - Returns DependencyAnalysis struct with module relationships
 *    - Use when you need to understand dependencies before merging
 *    - Required for mergeModulesSkipIntermediates()
 *
 * 5. mergeModules() - Lowest level, returns MLIR module directly
 *    - Requires MLIR context management
 *    - Must handle module lifetime carefully
 *    - Supports optional output skipping via skip sets
 *    - Use only for advanced MLIR operations
 *
 * 6. mergeModulesSkipIntermediates() - Low-level skip intermediates merge
 *    - Requires pre-computed DependencyAnalysis
 *    - Returns MLIR module with intermediate outputs excluded
 *    - Use when you need direct MLIR control with skip semantics
 *
 * IMPORTANT NOTES:
 * - All functions analyze dependencies automatically based on tensor addresses
 * - Circular dependencies are detected and will throw runtime_error
 * - Output ordering always maintains original module identity (Module1 first, then Module2)
 * - Functions handle 4 scenarios: INDEPENDENT, COMMON_INPUTS, DIRECT_DEPS, MIXED
 */

#include <ATen/Tensor.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "IrNode.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_neuronx {

// ============================================================================
// Data Structures for Module Merging
// ============================================================================
//
// These structures provide flexible module merging across different API levels:
//
// 1. DependencyAnalysis - Captures all relationships between modules
//    - Common inputs, dependencies, execution order, scenario classification
//    - Created by analyzeDependencies(), used throughout merge process
//
// 2. MergeMapping - Maps merged inputs/outputs to original modules
//    - Essential for correct tensor ordering when executing merged MLIR
//    - Used standalone via out-parameter in low-level mergeModules()
//    - Also embedded in MergeResult for high-level tensor-aware APIs
//
// 3. MergeResult - Complete package with MLIR, tensors, and embedded mapping
//    - Contains everything needed to execute merged module
//    - MergeMapping embedded (not separate) to keep all merge info together
//
// Design: Separation enables lightweight MLIR-only operations while providing
// cohesive packages for tensor management. Low-level APIs use out-parameters,
// high-level APIs return complete MergeResult.
//
// Usage: analyzeDependencies() -> mergeModules() -> MergeResult (high-level)
//                                               -> MergeMapping* (low-level)
// ============================================================================

/**
 * @brief Dependency analysis results between two modules
 *
 * Captures all relationships between two modules including common inputs,
 * dependencies, execution order, and scenario classification.
 */
struct DependencyAnalysis {
  // Indices of common inputs between modules (module1_idx -> module2_idx)
  std::unordered_map<size_t, size_t> commonInputs;

  // Dependencies: module1_output_idx -> [module2_input_idx, ...]
  // Uses vector to support duplicate inputs (e.g., x + x where both inputs come from same output)
  std::unordered_map<size_t, std::vector<size_t>> module1ToModule2Deps;

  // Dependencies: module2_output_idx -> [module1_input_idx, ...]
  // Uses vector to support duplicate inputs (e.g., x + x where both inputs come from same output)
  std::unordered_map<size_t, std::vector<size_t>> module2ToModule1Deps;

  // Execution order: true if module1 should execute first, false if module2 first
  bool module1First = true;

  // Scenario type classification
  enum ScenarioType {
    INDEPENDENT = 1,    // No relationships between modules
    COMMON_INPUTS = 2,  // Modules share input tensors
    DIRECT_DEPS = 3,    // One module's output feeds another's input
    MIXED = 4           // Both common inputs and dependencies
  } scenario = INDEPENDENT;
};

/**
 * @brief Mapping between merged function and original modules
 *
 * Tracks how inputs/outputs in the merged function correspond to
 * the original module inputs/outputs. Essential for correct tensor
 * ordering when executing the merged MLIR.
 */
struct MergeMapping {
  // Maps merged function input indices to original tensor info
  // Key: merged input index, Value: {module_num (1 or 2), original_index}
  std::unordered_map<size_t, std::pair<int, size_t>> input_mapping;

  // Maps merged function output indices to original tensor info
  // Key: merged output index, Value: {module_num (1 or 2), original_index}
  std::unordered_map<size_t, std::pair<int, size_t>> output_mapping;

  // Total number of inputs/outputs in merged function
  size_t total_inputs = 0;
  size_t total_outputs = 0;
};

/**
 * @brief Complete merge result with MLIR, tensors, and mapping
 *
 * Contains everything needed to execute a merged MLIR module:
 * - The merged MLIR as a string
 * - Correctly ordered input/output tensors
 * - Mapping information to trace back to original modules
 *
 * Note: MergeMapping is embedded rather than separate to keep all merge
 * information together. Consider accessing mapping through this structure
 * rather than passing MergeMapping separately.
 */
struct MergeResult {
  std::string merged_mlir_string;  // String representation of merged MLIR

  // Vectors containing the actual merged tensors in the correct order
  std::vector<at::Tensor> merged_inputs;   // Input tensors for the merged MLIR
  std::vector<at::Tensor> merged_outputs;  // Output tensors for the merged MLIR

  // Mapping information for tensor handling
  // NOTE: This contains the same information that would be in a standalone
  // MergeMapping. Prefer using MergeResult over separate MergeMapping when
  // working with tensors.
  MergeMapping mapping;

  bool success = false;  // Whether the merge was successful
};

// ============================================================================
// Function Declarations
// ============================================================================

/**
 * @brief Analyzes tensor address relationships between two modules
 *
 * @throws std::runtime_error if circular dependencies are detected between modules
 */
DependencyAnalysis analyzeDependencies(const std::vector<void*>& module1_input_addrs,
                                       const std::vector<void*>& module1_output_addrs,
                                       const std::vector<void*>& module2_input_addrs,
                                       const std::vector<void*>& module2_output_addrs);

/**
 * @brief Enhanced mergeModules with dependency analysis and flexible output skipping support
 *
 * Merges two MLIR StableHLO modules based on tensor address analysis to handle:
 * 1. Independent modules
 * 2. Modules with common inputs
 * 3. Modules with direct dependencies
 * 4. Mixed scenarios (common inputs + dependencies)
 *
 * Supports flexible output skipping via optional skip sets:
 * - Empty skip sets (default): Keeps all outputs from both modules
 * - Populated skip sets: Excludes specified outputs by original module index
 *
 * Skip sets use ORIGINAL MODULE INDICES (0-based within each module). Examples:
 * - module1_outputs_to_skip = {0, 2} excludes Module1's outputs at indices 0 and 2
 * - module2_outputs_to_skip = {1} excludes Module2's output at index 1
 *
 * Output ordering always maintains original module identity (Module1 outputs first,
 * then Module2 outputs), regardless of execution order.
 *
 * @param module1 First MLIR module to merge
 * @param module2 Second MLIR module to merge
 * @param context MLIR context with required dialects loaded (must not be null)
 * @param module1_input_addrs Tensor addresses for module1 inputs
 * @param module1_output_addrs Tensor addresses for module1 outputs
 * @param module2_input_addrs Tensor addresses for module2 inputs
 * @param module2_output_addrs Tensor addresses for module2 outputs
 * @param mapping Optional pointer to capture mapping information
 * @param verify_output Optional flag to enable/disable MLIR verification (default: true)
 * @param module1_outputs_to_skip Optional set of Module1 output indices to exclude (default: {})
 * @param module2_outputs_to_skip Optional set of Module2 output indices to exclude (default: {})
 *
 * @return OwningOpRef to merged module on success
 *
 * @throws std::runtime_error if circular dependencies are detected, tensor address counts mismatch,
 * main functions are not found, context is null, or verification fails (when enabled)
 */
mlir::OwningOpRef<mlir::ModuleOp> mergeModules(
    mlir::ModuleOp module1, mlir::ModuleOp module2, mlir::MLIRContext* context,
    const std::vector<void*>& module1_input_addrs, const std::vector<void*>& module1_output_addrs,
    const std::vector<void*>& module2_input_addrs, const std::vector<void*>& module2_output_addrs,
    MergeMapping* mapping = nullptr, bool verify_output = true,
    const std::unordered_set<size_t>& module1_outputs_to_skip = {},
    const std::unordered_set<size_t>& module2_outputs_to_skip = {}, bool run_optimization = true);

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
 * @param verify_output Optional flag to enable/disable MLIR verification (default: true)
 *
 * @return Merged module as MLIR text string, empty string on failure
 */
std::string mergeStableHLOModules(const std::string& mod_str1, const std::string& mod_str2,
                                  const std::vector<void*>& module1_input_addrs,
                                  const std::vector<void*>& module1_output_addrs,
                                  const std::vector<void*>& module2_input_addrs,
                                  const std::vector<void*>& module2_output_addrs,
                                  bool verify_output = true, bool run_optimization = true);

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
std::string mergeStableHLOModulesSkipIntermediates(
    const std::string& mod_str1, const std::string& mod_str2,
    const std::vector<void*>& module1_input_addrs, const std::vector<void*>& module1_output_addrs,
    const std::vector<void*>& module2_input_addrs, const std::vector<void*>& module2_output_addrs,
    MergeMapping* mapping = nullptr, bool verify_output = true);

/**
 * @brief Enhanced interface that returns both merged MLIR and correct tensor mappings
 *
 * This version returns the merged MLIR along with the correct input/output tensor mappings
 * that match what the merged MLIR expects, based on dependency analysis.
 *
 * @param mod_str1 MLIR text of first module
 * @param mod_str2 MLIR text of second module
 * @param module1_inputs Input tensors for module1 (ordered vector)
 * @param module1_outputs Output tensors for module1 (ordered vector)
 * @param module2_inputs Input tensors for module2 (ordered vector)
 * @param module2_outputs Output tensors for module2 (ordered vector)
 * @param verify_output Optional flag to enable/disable MLIR verification (default: true)
 *
 * @return MergeResult containing merged MLIR and correct tensor mappings
 */
MergeResult mergeStableHLOModulesWithTensors(const std::string& mod_str1,
                                             const std::string& mod_str2,
                                             const std::vector<at::Tensor>& module1_inputs,
                                             const std::vector<at::Tensor>& module1_outputs,
                                             const std::vector<at::Tensor>& module2_inputs,
                                             const std::vector<at::Tensor>& module2_outputs,
                                             bool verify_output = true);

/**
 * @brief Core skip intermediates merge function that excludes intermediate outputs from the merged
 * module
 *
 * This function performs the actual MLIR module merging but excludes outputs that are
 * consumed as inputs by the other module, treating them as intermediates.
 *
 * @param verify_output Optional flag to enable/disable MLIR verification (default: true)
 */
mlir::OwningOpRef<mlir::ModuleOp> mergeModulesSkipIntermediates(
    mlir::ModuleOp module1, mlir::ModuleOp module2, mlir::MLIRContext* context,
    const std::vector<void*>& module1_input_addrs, const std::vector<void*>& module1_output_addrs,
    const std::vector<void*>& module2_input_addrs, const std::vector<void*>& module2_output_addrs,
    const DependencyAnalysis& analysis, MergeMapping* mapping = nullptr, bool verify_output = true,
    bool run_optimization = true);

/**
 * @brief SKIP INTERMEDIATES version of mergeStableHLOModulesWithTensors that treats dependencies as
 * intermediates
 *
 * This skip intermediates version differs from the conservative version in that if MLIR2 uses
 * output of MLIR1, then the used output of MLIR1 is NOT included in the output of the combined
 * MLIR. It is treated as an intermediate value and thus not exposed as an output.
 *
 * WARNING: This is skip intermediates because it removes outputs that might be needed by external
 * consumers. Use only when you're certain that the dependent outputs are not needed externally.
 *
 * @param mod_str1 MLIR text of first module
 * @param mod_str2 MLIR text of second module
 * @param module1_inputs Input tensors for module1 (ordered vector)
 * @param module1_outputs Output tensors for module1 (ordered vector)
 * @param module2_inputs Input tensors for module2 (ordered vector)
 * @param module2_outputs Output tensors for module2 (ordered vector)
 * @param verify_output Optional flag to enable/disable MLIR verification (default: true)
 *
 * @return MergeResult containing merged MLIR with intermediate outputs removed
 */
MergeResult mergeStableHLOModulesWithTensorsSkipIntermediates(
    const std::string& mod_str1, const std::string& mod_str2,
    const std::vector<at::Tensor>& module1_inputs, const std::vector<at::Tensor>& module1_outputs,
    const std::vector<at::Tensor>& module2_inputs, const std::vector<at::Tensor>& module2_outputs,
    bool verify_output = true);

std::unique_ptr<StableHloNode> mergeStableHLOModulesWithTensorsSkipIntermediates(
    StableHloNode* node1, StableHloNode* node2, mlir::MLIRContext* context);

/**
 * @brief Safe merge of two StableHloNodes preserving all outputs
 *
 * Merges two StableHloNode objects using the safe merge strategy that
 * preserves all outputs from both modules. Creates a new StableHloNode
 * with the merged MLIR module and comprehensive tensor mappings.
 *
 * @param node1 First StableHloNode to merge
 * @param node2 Second StableHloNode to merge
 * @return Unique pointer to merged StableHloNode, nullptr on failure
 */
std::unique_ptr<StableHloNode> mergeStableHLOModulesWithTensors(StableHloNode* node1,
                                                                StableHloNode* node2,
                                                                mlir::MLIRContext* context);

// ============================================================================
// In-Order Module Merging APIs (Optimized for Sequential Execution)
// ============================================================================
//
// These APIs are optimized for the common case where module execution order is
// known at call time: module1 always executes before module2. This assumption
// eliminates the need for bidirectional dependency analysis and circular
// dependency detection.
//
// Key optimizations over general merging APIs:
// 1. No circular dependency check needed (impossible by construction)
// 2. No execution order determination (always module1 -> module2)
// 3. Simplified dependency analysis (only check module1_output -> module2_input)
// 4. Reduced computational overhead
//
// Use these APIs when you have sequential operations where the execution order
// is guaranteed (e.g., consecutive operations in an eager execution trace).
// ============================================================================

/**
 * @brief Optimized merge for sequential module execution (module1 before module2)
 *
 * This API merges two StableHLO modules assuming module1 always executes before
 * module2. This assumption removes the need for bidirectional dependency analysis
 * and circular dependency detection, making it more efficient for sequential
 * operation concatenation.
 *
 * Key differences from mergeStableHLOModulesWithTensors:
 * - No circular dependency check (impossible when order is known)
 * - No execution order analysis (always module1 first)
 * - Only checks module1_output -> module2_input dependencies
 * - More efficient for sequential eager execution patterns
 *
 * @param mod_str1 MLIR text of first module (executes first)
 * @param mod_str2 MLIR text of second module (executes second)
 * @param module1_inputs Input tensors for module1 (ordered vector)
 * @param module1_outputs Output tensors for module1 (ordered vector)
 * @param module2_inputs Input tensors for module2 (ordered vector)
 * @param module2_outputs Output tensors for module2 (ordered vector)
 * @param verify_output Optional flag to enable/disable MLIR verification (default: true)
 *
 * @return MergeResult containing merged MLIR and correct tensor mappings
 *
 * @note This function assumes module1 outputs may feed into module2 inputs,
 *       but module2 outputs NEVER feed into module1 inputs.
 */
MergeResult mergeStableHLOModulesWithTensorsInOrder(const std::string& mod_str1,
                                                    const std::string& mod_str2,
                                                    const std::vector<at::Tensor>& module1_inputs,
                                                    const std::vector<at::Tensor>& module1_outputs,
                                                    const std::vector<at::Tensor>& module2_inputs,
                                                    const std::vector<at::Tensor>& module2_outputs,
                                                    bool verify_output = true);

/**
 * @brief Skip intermediates version of in-order merge
 *
 * Combines the efficiency of in-order merging with intermediate output skipping.
 * Module1 outputs consumed by module2 inputs are treated as intermediates and
 * excluded from the final merged module outputs.
 *
 * @param mod_str1 MLIR text of first module (executes first)
 * @param mod_str2 MLIR text of second module (executes second)
 * @param module1_inputs Input tensors for module1 (ordered vector)
 * @param module1_outputs Output tensors for module1 (ordered vector)
 * @param module2_inputs Input tensors for module2 (ordered vector)
 * @param module2_outputs Output tensors for module2 (ordered vector)
 * @param verify_output Optional flag to enable/disable MLIR verification (default: true)
 *
 * @return MergeResult containing merged MLIR with intermediate outputs removed
 */
MergeResult mergeStableHLOModulesWithTensorsInOrderSkipIntermediates(
    const std::string& mod_str1, const std::string& mod_str2,
    const std::vector<at::Tensor>& module1_inputs, const std::vector<at::Tensor>& module1_outputs,
    const std::vector<at::Tensor>& module2_inputs, const std::vector<at::Tensor>& module2_outputs,
    bool verify_output = true);

/**
 * @brief StableHloNode version of in-order merge preserving all outputs
 *
 * Merges two StableHloNode objects assuming node1 executes before node2.
 * Creates a new StableHloNode with the merged MLIR module and comprehensive
 * tensor mappings. All outputs from both modules are preserved.
 *
 * @param node1 First StableHloNode to merge (executes first)
 * @param node2 Second StableHloNode to merge (executes second)
 * @param context MLIR context with required dialects loaded
 * @return Unique pointer to merged StableHloNode, nullptr on failure
 */
std::unique_ptr<StableHloNode> mergeStableHLOModulesWithTensorsInOrder(StableHloNode* node1,
                                                                       StableHloNode* node2,
                                                                       mlir::MLIRContext* context);

/**
 * @brief StableHloNode version of in-order merge with intermediate skipping
 *
 * Merges two StableHloNode objects assuming node1 executes before node2,
 * excluding intermediate outputs (node1 outputs consumed by node2 inputs).
 *
 * @param node1 First StableHloNode to merge (executes first)
 * @param node2 Second StableHloNode to merge (executes second)
 * @param context MLIR context with required dialects loaded
 * @return Unique pointer to merged StableHloNode, nullptr on failure
 */
std::unique_ptr<StableHloNode> mergeStableHLOModulesWithTensorsInOrderSkipIntermediates(
    StableHloNode* node1, StableHloNode* node2, mlir::MLIRContext* context);

/**
 * @brief Simplified dependency analysis for in-order module merging
 *
 * Analyzes dependencies assuming module1 always executes before module2.
 * This is an optimized version of analyzeDependencies that:
 * - Skips checking module2_output -> module1_input dependencies
 * - Skips circular dependency detection
 * - Always sets module1First = true
 *
 * @param module1_input_addrs Tensor addresses for module1 inputs
 * @param module1_output_addrs Tensor addresses for module1 outputs
 * @param module2_input_addrs Tensor addresses for module2 inputs
 * @param module2_output_addrs Tensor addresses for module2 outputs (not used for dependency check)
 *
 * @return DependencyAnalysis with module1First always true and only module1->module2 deps
 */
DependencyAnalysis analyzeDependenciesInOrder(const std::vector<void*>& module1_input_addrs,
                                              const std::vector<void*>& module1_output_addrs,
                                              const std::vector<void*>& module2_input_addrs,
                                              const std::vector<void*>& module2_output_addrs);

/**
 * @brief Low-level in-order module merge with MLIR context
 *
 * Core merge function for in-order module merging. Assumes module1 executes
 * before module2, eliminating bidirectional dependency analysis.
 *
 * @param module1 First MLIR module to merge (executes first)
 * @param module2 Second MLIR module to merge (executes second)
 * @param context MLIR context with required dialects loaded
 * @param module1_input_addrs Tensor addresses for module1 inputs
 * @param module1_output_addrs Tensor addresses for module1 outputs
 * @param module2_input_addrs Tensor addresses for module2 inputs
 * @param module2_output_addrs Tensor addresses for module2 outputs
 * @param mapping Optional pointer to capture mapping information
 * @param verify_output Optional flag to enable/disable MLIR verification
 * @param module1_outputs_to_skip Optional set of Module1 output indices to skip
 * @param module2_outputs_to_skip Optional set of Module2 output indices to skip
 *
 * @return OwningOpRef to merged module on success
 */
mlir::OwningOpRef<mlir::ModuleOp> mergeModulesInOrder(
    mlir::ModuleOp module1, mlir::ModuleOp module2, mlir::MLIRContext* context,
    const std::vector<void*>& module1_input_addrs, const std::vector<void*>& module1_output_addrs,
    const std::vector<void*>& module2_input_addrs, const std::vector<void*>& module2_output_addrs,
    MergeMapping* mapping = nullptr, bool verify_output = true,
    const std::unordered_set<size_t>& module1_outputs_to_skip = {},
    const std::unordered_set<size_t>& module2_outputs_to_skip = {});

}  // namespace torch_neuronx

#endif  // TORCH_NEURONX_CSRC_CORE_CONCATENATION_OPCONCATUTILS_H_
