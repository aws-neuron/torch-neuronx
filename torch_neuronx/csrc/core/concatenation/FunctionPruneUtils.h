#ifndef TORCH_NEURONX_CSRC_CORE_CONCATENATION_FUNCTIONPRUNEUTILS_H_
#define TORCH_NEURONX_CSRC_CORE_CONCATENATION_FUNCTIONPRUNEUTILS_H_

/**
 * @file FunctionPruneUtils.h
 * @brief MLIR Function Argument/Result Pruning Utilities
 *
 * This file provides utilities for pruning (removing) unwanted function arguments
 * and results from MLIR func::FuncOp operations. This is useful for "IR surgery"
 * where you need to clean up function signatures.
 *
 * API DESIGN:
 * ===========
 *
 * Index-Based API (Direct control)
 *   - pruneFunctionArgumentsByIndices() - Remove arguments by index
 *   - pruneFunctionResultsByIndices() - Remove results by index
 *   - pruneModuleArgumentsByIndices() - Module-level wrapper for arguments
 *   - pruneModuleResultsByIndices() - Module-level wrapper for results
 *   - Work with pre-computed index sets
 *   - Useful when you already know which indices to keep
 *
 * ERROR HANDLING:
 * ===============
 * All functions throw std::runtime_error on validation failures or invalid operations.
 * Use try-catch blocks when calling these utilities.
 *
 * USAGE EXAMPLE:
 * ==============
 *
 * Index-based pruning (keep only arguments 0, 2, 3)
 * ```cpp
 * try {
 *   std::unordered_set<size_t> indicesToKeep = {0, 2, 3};
 *   pruneFunctionArgumentsByIndices(funcOp, indicesToKeep);
 * } catch (const std::runtime_error& e) {
 *   // Handle error
 * }
 * ```
 *
 * IMPORTANT NOTES:
 * ================
 * - Both arguments and results can be pruned
 * - The func.return operation is automatically updated when results are pruned
 * - Function signature (FunctionType) is automatically updated
 * - Callers (func.call operations) are NOT automatically updated
 * - Pruning uses MLIR's built-in eraseArguments() and result type updates
 * - All validation errors throw std::runtime_error
 */

#include <string>
#include <unordered_set>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace torch_neuronx {

// ============================================================================
// Low-Level Index-Based API
// ============================================================================

/**
 * @brief Prune function arguments by keeping only specified indices
 *
 * This low-level function removes all function arguments whose indices are NOT
 * in the indicesToKeep set. The function signature is updated accordingly.
 *
 * @param funcOp The function operation to modify
 * @param indicesToKeep Set of argument indices to KEEP (0-based)
 *
 * @throws std::runtime_error if indices are out of range
 *
 * @note This function modifies funcOp in-place
 * @note Callers of this function are NOT automatically updated
 *
 * Example:
 * ```cpp
 * // Keep only arguments at indices 0, 2, 3 (remove 1, 4, etc.)
 * std::unordered_set<size_t> indicesToKeep = {0, 2, 3};
 * pruneFunctionArgumentsByIndices(funcOp, indicesToKeep);
 * ```
 */
void pruneFunctionArgumentsByIndices(mlir::func::FuncOp funcOp,
                                     const std::unordered_set<size_t>& indicesToKeep);

/**
 * @brief Prune function results by keeping only specified indices
 *
 * This low-level function removes all function results whose indices are NOT
 * in the indicesToKeep set. Both the function signature and the func.return
 * operation are updated accordingly.
 *
 * @param funcOp The function operation to modify
 * @param indicesToKeep Set of result indices to KEEP (0-based)
 *
 * @throws std::runtime_error if indices are out of range or func.return not found
 *
 * @note This function modifies funcOp in-place
 * @note The func.return operation inside the function is automatically updated
 * @note Callers of this function are NOT automatically updated
 *
 * Example:
 * ```cpp
 * // Keep only results at indices 0, 2 (remove 1, 3, etc.)
 * std::unordered_set<size_t> indicesToKeep = {0, 2};
 * pruneFunctionResultsByIndices(funcOp, indicesToKeep);
 * ```
 */
void pruneFunctionResultsByIndices(mlir::func::FuncOp funcOp,
                                   const std::unordered_set<size_t>& indicesToKeep);

// ============================================================================
// Module-Level Convenience Wrappers
// ============================================================================

/**
 * @brief Prune module's main function arguments by keeping only specified indices
 *
 * Convenience wrapper that operates at the module level. Looks up the specified
 * function (default "main") and calls pruneFunctionArgumentsByIndices().
 *
 * @param module The module containing the function to modify
 * @param indicesToKeep Set of argument indices to KEEP (0-based)
 * @param functionName Name of the function to prune (default: "main")
 *
 * @throws std::runtime_error if function not found or indices are out of range
 *
 * Example:
 * ```cpp
 * // Keep only arguments 0, 2, 3 from the main function
 * std::unordered_set<size_t> indicesToKeep = {0, 2, 3};
 * pruneModuleArgumentsByIndices(module, indicesToKeep);
 * ```
 */
void pruneModuleArgumentsByIndices(mlir::ModuleOp module,
                                   const std::unordered_set<size_t>& indicesToKeep,
                                   const std::string& functionName = "main");

/**
 * @brief Prune module's main function results by keeping only specified indices
 *
 * Convenience wrapper that operates at the module level. Looks up the specified
 * function (default "main") and calls pruneFunctionResultsByIndices().
 *
 * @param module The module containing the function to modify
 * @param indicesToKeep Set of result indices to KEEP (0-based)
 * @param functionName Name of the function to prune (default: "main")
 *
 * @throws std::runtime_error if function not found or indices are out of range
 *
 * Example:
 * ```cpp
 * // Keep only results 0, 2 from the main function
 * std::unordered_set<size_t> indicesToKeep = {0, 2};
 * pruneModuleResultsByIndices(module, indicesToKeep);
 * ```
 */
void pruneModuleResultsByIndices(mlir::ModuleOp module,
                                 const std::unordered_set<size_t>& indicesToKeep,
                                 const std::string& functionName = "main");

}  // namespace torch_neuronx

#endif  // TORCH_NEURONX_CSRC_CORE_CONCATENATION_FUNCTIONPRUNEUTILS_H_
