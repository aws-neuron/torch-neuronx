#include "FunctionPruneUtils.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"

namespace torch_neuronx {

/**
 * @brief Helper function to validate indices against a maximum value
 *
 * @throws std::runtime_error if any index is out of range
 */
static void validateIndices(const std::unordered_set<size_t>& indices, size_t maxIndex,
                            const std::string& context) {
  for (size_t idx : indices) {
    if (idx >= maxIndex) {
      throw std::runtime_error(context + ": Index " + std::to_string(idx) + " is out of range " +
                               "(max: " + std::to_string(maxIndex - 1) + ")");
    }
  }
}

/**
 * @brief Helper function to build BitVector for argument erasure
 *
 * Creates a BitVector where bits are set for arguments to ERASE (not keep).
 * MLIR's eraseArguments() takes a BitVector where set bits indicate deletion.
 */
static mlir::BitVector buildEraseVector(size_t totalCount,
                                        const std::unordered_set<size_t>& indicesToKeep) {
  mlir::BitVector eraseVector(totalCount);

  for (size_t i = 0; i < totalCount; ++i) {
    // Set bit if this index should be ERASED (not in keep set)
    if (indicesToKeep.find(i) == indicesToKeep.end()) {
      eraseVector.set(i);
    }
  }

  return eraseVector;
}

void pruneFunctionArgumentsByIndices(mlir::func::FuncOp funcOp,
                                     const std::unordered_set<size_t>& indicesToKeep) {
  size_t numArgs = funcOp.getNumArguments();

  TORCH_NEURONX_DEBUG("[FUNCTION_PRUNE] pruneFunctionArgumentsByIndices called",
                      "function=", funcOp.getSymName().str(), "total_args=", numArgs,
                      "args_to_keep=", indicesToKeep.size());

  // Validate indices
  validateIndices(indicesToKeep, numArgs, "pruneFunctionArgumentsByIndices");

  // If keeping all arguments, nothing to do
  if (indicesToKeep.size() == numArgs) {
    TORCH_NEURONX_DEBUG("[FUNCTION_PRUNE] No arguments to prune - keeping all");
    return;
  }

  // Build BitVector for arguments to erase
  mlir::BitVector argsToErase = buildEraseVector(numArgs, indicesToKeep);

  // Count how many we're erasing for logging
  size_t numToErase = argsToErase.count();
  TORCH_NEURONX_DEBUG("[FUNCTION_PRUNE] Erasing ", numToErase, " arguments");

  // Erase arguments from the entry block and update function signature
  // This single call handles both:
  // 1. Removing arguments from the entry block
  // 2. Updating the function's FunctionType
  if (mlir::failed(funcOp.eraseArguments(argsToErase))) {
    throw std::runtime_error(
        "pruneFunctionArgumentsByIndices: Failed to erase function arguments from " +
        funcOp.getSymName().str());
  }

  TORCH_NEURONX_DEBUG("[FUNCTION_PRUNE] Arguments pruned successfully",
                      "new_arg_count=", funcOp.getNumArguments());
}

void pruneFunctionResultsByIndices(mlir::func::FuncOp funcOp,
                                   const std::unordered_set<size_t>& indicesToKeep) {
  size_t numResults = funcOp.getNumResults();

  TORCH_NEURONX_DEBUG("[FUNCTION_PRUNE] pruneFunctionResultsByIndices called",
                      "function=", funcOp.getSymName().str(), "total_results=", numResults,
                      "results_to_keep=", indicesToKeep.size());

  // Validate indices
  validateIndices(indicesToKeep, numResults, "pruneFunctionResultsByIndices");

  // If keeping all results, nothing to do
  if (indicesToKeep.size() == numResults) {
    TORCH_NEURONX_DEBUG("[FUNCTION_PRUNE] No results to prune - keeping all");
    return;
  }

  // Find the return operation in the function
  mlir::func::ReturnOp returnOp;
  funcOp.walk([&](mlir::func::ReturnOp op) {
    returnOp = op;
    return mlir::WalkResult::interrupt();
  });

  if (!returnOp) {
    throw std::runtime_error("pruneFunctionResultsByIndices: func.return operation not found in " +
                             funcOp.getSymName().str());
  }

  // Build list of values to keep for the new return operation
  llvm::SmallVector<mlir::Value> newReturnValues;
  for (size_t i = 0; i < numResults; ++i) {
    if (indicesToKeep.find(i) != indicesToKeep.end()) {
      newReturnValues.push_back(returnOp.getOperand(i));
    }
  }

  TORCH_NEURONX_DEBUG("[FUNCTION_PRUNE] Keeping ", newReturnValues.size(), " of ", numResults,
                      " return values");

  // Create new return operation with pruned values
  mlir::OpBuilder builder(returnOp);
  builder.create<mlir::func::ReturnOp>(returnOp.getLoc(), newReturnValues);

  // Erase the old return operation
  returnOp.erase();

  // Update the function signature to match the new return types
  llvm::SmallVector<mlir::Type> newResultTypes;
  for (const auto& value : newReturnValues) {
    newResultTypes.push_back(value.getType());
  }

  // Get current input types (these don't change for result pruning)
  auto currentType = funcOp.getFunctionType();
  auto newFuncType = builder.getFunctionType(currentType.getInputs(), newResultTypes);
  funcOp.setFunctionType(newFuncType);

  TORCH_NEURONX_DEBUG("[FUNCTION_PRUNE] Results pruned successfully",
                      "new_result_count=", funcOp.getNumResults());
}

// ============================================================================
// Module-Level Convenience Wrappers
// ============================================================================

void pruneModuleArgumentsByIndices(mlir::ModuleOp module,
                                   const std::unordered_set<size_t>& indicesToKeep,
                                   const std::string& functionName) {
  // Look up the specified function
  auto func = module.lookupSymbol<mlir::func::FuncOp>(functionName);
  if (!func) {
    throw std::runtime_error("Function '" + functionName + "' not found in module");
  }

  // Delegate to function-level utility
  pruneFunctionArgumentsByIndices(func, indicesToKeep);
}

void pruneModuleResultsByIndices(mlir::ModuleOp module,
                                 const std::unordered_set<size_t>& indicesToKeep,
                                 const std::string& functionName) {
  // Look up the specified function
  auto func = module.lookupSymbol<mlir::func::FuncOp>(functionName);
  if (!func) {
    throw std::runtime_error("Function '" + functionName + "' not found in module");
  }

  // Delegate to function-level utility
  pruneFunctionResultsByIndices(func, indicesToKeep);
}

}  // namespace torch_neuronx
